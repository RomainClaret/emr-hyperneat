"""Building a substrate: the masks and weight matrices of a population, the locality penalty, the CPPN
queries (one output on a device, several outputs, the self-connections) and the per-level queries,
variances and accumulation of the streaming build."""
import functools
import threading
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.emrhyperneat_base import (
    HierarchicalGridStructure,
    _get_device_specific_query_fn,
    batch_query_population_positions,
    compute_hierarchical_variances_batch_jit,
    compute_subdivision_masks_batch_jit,
    cppn_query_rows,
)


# Locality Penalty for Geometry Seeding
# Implements distance-based weight scaling to favor local connections.
# Connections within locality_radius get full weight, distant ones are scaled down.

def compute_locality_penalty(
    source_coords: jnp.ndarray,
    target_coords: jnp.ndarray,
    locality_radius: float,
) -> jnp.ndarray:
    """Compute distance-based weight scaling for locality bias (geometry seeding).

    For each source-target pair, compute the Euclidean distance and scale
    weights based on whether the distance is within the locality radius.

    Args:
        source_coords: Source positions, shape (num_sources, 2) - (x, y) coordinates
        target_coords: Target positions, shape (num_targets, 2) - (x, y) coordinates
        locality_radius: Radius within which connections get full weight.
                        Connections beyond this are scaled down.

    Returns:
        Scale factors, shape (num_sources, num_targets) with values in [0, 1]:
        - 1.0 for distances <= locality_radius
        - locality_radius / distance for distances > locality_radius
    """
    # Compute pairwise distances: (num_sources, num_targets)
    # source_coords: (S, 2), target_coords: (T, 2)
    # Expand dims for broadcasting: (S, 1, 2) - (1, T, 2) = (S, T, 2)
    diff = source_coords[:, None, :] - target_coords[None, :, :]  # (S, T, 2)
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (S, T)

    # Compute scale: 1.0 for local, decay for distant
    # Scale = min(1.0, locality_radius / (distance + epsilon))
    epsilon = 1e-6
    scale = jnp.minimum(1.0, locality_radius / (distances + epsilon))

    return scale


def apply_locality_penalty_to_weights(
    weights: jnp.ndarray,
    source_coords: jnp.ndarray,
    target_coords: jnp.ndarray,
    locality_radius: Optional[float],
) -> jnp.ndarray:
    """Apply locality-based distance penalty to weight matrix.

    Args:
        weights: Weight matrix, shape (pop_size, num_sources, num_targets) or
                (num_sources, num_targets)
        source_coords: Source positions, shape (num_sources, 2)
        target_coords: Target positions, shape (num_targets, 2)
        locality_radius: Radius for locality penalty, or None to skip

    Returns:
        Scaled weights with same shape as input
    """
    if locality_radius is None:
        return weights

    # Compute scale factors: (num_sources, num_targets)
    scale = compute_locality_penalty(source_coords, target_coords, locality_radius)

    # Apply scaling - handles both (pop, S, T) and (S, T) shapes
    if weights.ndim == 3:
        # Broadcast scale: (1, S, T) to match (pop, S, T)
        return weights * scale[None, :, :]
    else:
        return weights * scale


# Self-Connection Query for Receptor Densities
# Proper CPPN-based method for extracting per-node receptor densities using
# self-connection queries at (x, y, x, y, bias).

def self_connection_rows(
    positions: jnp.ndarray,
    geometry_seeding_enabled: bool = False,
) -> jnp.ndarray:
    """CPPN input rows for the self-connection query at every position.

    A self-connection has source and target at the same position, ``(x, y, x, y, 1)``.
    The query reads per-node properties from it: a hidden node's activation and
    aggregation function in ``cppn_output`` mode, and its receptor densities under
    neuromodulation. Column order matches `cppn_query_rows` exactly:

      - 5-D: ``[x1, y1, x2, y2, bias]``
      - 7-D: ``[x1, y1, x2, y2, delta_x, delta_y, bias]`` (deltas before the bias,
        both exactly zero here, but present so the genome's input count matches)

    Args:
        positions: Positions to query (num_positions, 2).
        geometry_seeding_enabled: Add the two zero delta columns (7-D rows).

    Returns:
        (num_positions, 5) or (num_positions, 7) rows.
    """
    num_positions = positions.shape[0]
    bias = jnp.ones((num_positions, 1))
    if geometry_seeding_enabled:
        zero_delta = jnp.zeros((num_positions, 1))
        return jnp.concatenate(
            [positions, positions, zero_delta, zero_delta, bias], axis=1
        )  # (num_positions, 7)
    return jnp.concatenate([positions, positions, bias], axis=1)  # (num_positions, 5)


def batch_query_population_self_connections(
    state: Any,
    cppns_transformed: Tuple,
    positions: jnp.ndarray,
    cppn_forward: Any,
    num_cppn_outputs: int = 1,
    pop_chunk_size: int = 0,
    geometry_seeding_enabled: bool = False,
    device_id: int = 0,
) -> jnp.ndarray:
    """Query all CPPNs for self-connections at each position.

    Self-connections query CPPN with (x, y, x, y, bias) - same source and target.
    This is used for TRUE neuromodulation to extract per-node receptor densities,
    which are fixed properties of each hidden node (not connection-dependent).

    The query runs through the multi-output weight queries' compiled population query
    (:func:`query_cppn_multi_output`), cached per device and CPPN forward with the state
    traced. Called eagerly, ``cppn_forward`` would take the state as a static argument, which
    hashes by identity, so every generation's new state would trace and compile the CPPN
    forward again; the compiled query gives the same outputs, bit for bit on a CPU.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        positions: All positions to query (num_positions, 2)
        cppn_forward: JIT-compiled CPPN forward function
        num_cppn_outputs: Number of CPPN outputs
        pop_chunk_size: Population chunk size for memory optimization (0=no chunking)
        geometry_seeding_enabled: If True, the CPPN takes 7 inputs rather than 5, so
            the delta columns must be supplied here too. For a self-connection source
            and target coincide, so both deltas are exactly zero, but the columns
            still have to be present or the genome's input_idx will not match.
        device_id: Device index for multi-GPU

    Returns:
        (pop_size, num_positions, num_cppn_outputs) array of CPPN outputs at self-connections
    """
    pop_size = cppns_transformed[0].shape[0]

    # Build self-connection inputs. For self-connections, source == target.
    inputs = self_connection_rows(positions, geometry_seeding_enabled)

    if pop_chunk_size > 0:
        # Chunked processing for memory efficiency
        chunk_results = []
        for chunk_start in range(0, pop_size, pop_chunk_size):
            chunk_end = min(chunk_start + pop_chunk_size, pop_size)
            chunk_cppns = tuple(part[chunk_start:chunk_end] for part in cppns_transformed)
            chunk_results.append(query_cppn_multi_output(cppn_forward, state, inputs, chunk_cppns,
                                                         device_id=device_id))
        return jnp.concatenate(chunk_results, axis=0)
    # (pop_size, num_positions, num_cppn_outputs)
    return query_cppn_multi_output(cppn_forward, state, inputs, cppns_transformed,
                                   device_id=device_id)


# Feed-forward substrate: shared by the generation loop and the public API
# The standard path of run_generation_verbose builds and evaluates its substrate
# through these functions, and emr_hyperneat.substrate composes the same ones, so a
# substrate built outside the generation loop is the one EMR evaluates rather than a
# second implementation of it. Change them only together with that module's tests.

#: Weights whose magnitude is at or below this, after tanh scaling, are pruned. EMR
#: has always used this constant on every path; it is not a configuration option.
WEIGHT_THRESHOLD = 0.1


def substrate_masks(
    variance_weights: jnp.ndarray,
    grid: HierarchicalGridStructure,
    variance_threshold: float,
    initial_depth: int = 0,
    legacy_level0_variance: bool = False,
) -> jnp.ndarray:
    """Which grid positions become hidden nodes.

    The variance source is the raw CPPN weight output from the first substrate input
    to every grid position. A cell subdivides when the variance over its four children
    exceeds the threshold (or while its level is below ``initial_depth``), and every
    reached cell, at every level, is a hidden node: not only the leaves.

    Args:
        variance_weights: (pop_size, total_positions) raw CPPN weight outputs.
        grid: The multi-resolution grid from ``get_hierarchical_grid(max_depth)``.
        variance_threshold: Subdivision threshold.
        initial_depth: Levels subdivided unconditionally.
        legacy_level0_variance: The original standard-path level-0 rule (``variance_rule: legacy``).

    Returns:
        (pop_size, total_positions) boolean mask.

    Traceable: it reads the grid's level offsets from its static tuples,
    the arithmetic of compute_hierarchical_variances_batch and compute_subdivision_masks_batch,
    whose ``int(grid.level_offsets[level])`` raises ConcretizationTypeError inside ``jax.jit`` and
    ``lax.while_loop``.
    """
    level_variances = compute_hierarchical_variances_batch_jit(
        variance_weights,
        level_sizes=grid.level_sizes_static,
        level_offsets=grid.level_offsets_static,
        level_grid_sizes=grid.level_grid_sizes_static,
        num_levels=grid.num_levels,
        legacy_level0_variance=legacy_level0_variance,
    )
    return compute_subdivision_masks_batch_jit(
        level_variances,
        variance_threshold=variance_threshold,
        parent_indices_tuple=grid.parent_indices,
        level_offsets=grid.level_offsets_static,
        num_levels=grid.num_levels,
        total_positions=grid.total_positions,
        initial_depth=initial_depth,
        return_all_masks=False,
    )


def strong_default_float(x: jnp.ndarray) -> jnp.ndarray:
    """``x`` as a strongly typed array of JAX's default float: float32, or float64 under x64.

    Under x64 TensorNEAT's CPPN forward returns weakly typed float64, and a weak float64 weight that
    meets the float32 substrate inputs computes in float32. The on-device query
    (`query_population_on_device`) keeps that weak type, where 0.1.0's numpy round trip made the
    weights strong float64: without this cast, under x64 the feed-forward fitness turns float32, on
    the core and on the general path alike (best 0.7388442754745483 against 0.7388442976057666 in
    float64; XOR, population 30, depth 3, generation 0). Under float32 the values are unchanged, bit
    for bit.
    """
    return jnp.asarray(x, dtype=jax.dtypes.canonicalize_dtype(jnp.float64))


def substrate_weight_matrices(
    input_weights: jnp.ndarray,
    output_weights: jnp.ndarray,
    masks: jnp.ndarray,
    max_weight: float,
    weight_threshold: float = WEIGHT_THRESHOLD,
    locality_radius: Optional[float] = None,
    input_coords: Optional[jnp.ndarray] = None,
    output_coords: Optional[jnp.ndarray] = None,
    positions: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Input-to-hidden and hidden-to-output weights from raw CPPN weight outputs.

    A weight is ``tanh(cppn) * max_weight``, optionally scaled down with distance beyond
    ``locality_radius``, and kept only at a hidden position and when its magnitude exceeds
    ``weight_threshold``. The CPPN outputs are taken in JAX's default float
    (`strong_default_float`), whatever query backend produced them, so the weights compute in
    float64 under x64.

    Args:
        input_weights: (pop_size, num_inputs, total_positions) raw CPPN outputs,
            substrate input -> grid position.
        output_weights: (pop_size, num_outputs, total_positions) raw CPPN outputs,
            grid position -> substrate output.
        masks: (pop_size, total_positions) hidden-node mask from `substrate_masks`.
        max_weight: Weight scale.
        weight_threshold: Pruning threshold.
        locality_radius: Radius of the geometry-seeding locality penalty, or None.
        input_coords: (num_inputs, 2); needed only with ``locality_radius``.
        output_coords: (num_outputs, 2); needed only with ``locality_radius``.
        positions: (total_positions, 2) grid positions; needed only with
            ``locality_radius``.

    Returns:
        ``(W1, W2)``: W1 is (pop_size, num_inputs, total_positions), W2 is
        (pop_size, total_positions, num_outputs), zero wherever pruned.
    """
    # Strong default-float CPPN outputs (float64 under x64, where TensorNEAT's are weakly typed)
    input_weights = strong_default_float(input_weights)
    output_weights = strong_default_float(output_weights)

    # Broadcast mask: (pop_size, 1, total_positions) for weight masking
    active_mask_broadcast = masks[:, None, :]

    # Apply tanh + max_weight scaling, then mask
    W1_raw = jnp.tanh(input_weights) * max_weight

    # Locality penalty for geometry seeding. W1: input_coords -> positions.
    if locality_radius is not None:
        W1_raw = apply_locality_penalty_to_weights(
            W1_raw, input_coords, positions, locality_radius
        )

    W1 = jnp.where(
        active_mask_broadcast & (jnp.abs(W1_raw) > weight_threshold),
        W1_raw,
        0.0
    )

    W2_raw = jnp.tanh(output_weights) * max_weight

    # W2_raw is (pop, num_outputs, total_positions) before the transpose, so the
    # penalty is applied on the (pop, positions, outputs) view and moved back.
    if locality_radius is not None:
        W2_for_penalty = W2_raw.transpose(0, 2, 1)  # (pop, total_positions, num_outputs)
        W2_for_penalty = apply_locality_penalty_to_weights(
            W2_for_penalty, positions, output_coords, locality_radius
        )
        W2_raw = W2_for_penalty.transpose(0, 2, 1)  # Back to (pop, num_outputs, total_positions)

    W2 = jnp.where(
        active_mask_broadcast & (jnp.abs(W2_raw) > weight_threshold),
        W2_raw,
        0.0
    )
    W2 = W2.transpose(0, 2, 1)  # (pop, num_outputs, total_positions) -> (pop, total_positions, num_outputs)
    return W1, W2


#: Genomes per call of the one-output CPPN query, as batch_query_population_multi_source_chunked's
#: default: the standard path never passed another, and the same batches keep the same compiled
#: query.
QUERY_POPULATION_CHUNK = 100


def substrate_query_rows(
    coords: jnp.ndarray,
    positions: jnp.ndarray,
    outgoing: bool,
    geometry_seeding_enabled: bool = False,
) -> Tuple[jnp.ndarray, ...]:
    """Each source coordinate's CPPN input rows (`cppn_query_rows`), on the device.

    Args:
        coords: (num_sources, 2) substrate input or output coordinates.
        positions: (total_positions, 2) grid positions.
        outgoing: True for source -> position (the input side), False for position -> source.
        geometry_seeding_enabled: 7-D rows with the coordinate deltas.

    Returns:
        One (total_positions, input_dim) array per source.
    """
    return tuple(
        jnp.asarray(
            cppn_query_rows(np.asarray(coord), positions, outgoing, geometry_seeding_enabled)
        )
        for coord in np.asarray(coords)
    )


def query_population_on_device(
    state: Any,
    cppns_transformed: Tuple,
    rows: Sequence[jnp.ndarray],
    cppn_forward: Any,
    pop_chunk_size: int = QUERY_POPULATION_CHUNK,
    device_id: int = 0,
) -> jnp.ndarray:
    """The one-output CPPN outputs of a population from every source, kept on the device.

    The arithmetic of `batch_query_population_multi_source_chunked`: the same per-device query
    function on the same rows, in the same population chunks. That function stacked the chunks and
    sources through numpy, a copy of every output to the host and back each generation (it was
    written for one-device threads of the multi-GPU routes); here they are concatenated and stacked
    with jnp, which moves data only, so the values are the same bit for bit and the function is
    traceable.

    Args:
        state: Algorithm state.
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...).
        rows: Each source's rows (`substrate_query_rows`).
        cppn_forward: JIT-compiled CPPN forward function.
        pop_chunk_size: Genomes per query call.
        device_id: The device whose query function runs.

    Returns:
        (pop_size, num_sources, total_positions) CPPN outputs.
    """
    query_fn = _get_device_specific_query_fn(device_id, cppn_forward)
    pop_size = cppns_transformed[0].shape[0]
    per_source = []
    for source_rows in rows:
        chunks = [
            query_fn(
                state,
                source_rows,
                *(part[start : start + pop_chunk_size] for part in cppns_transformed),
            )
            for start in range(0, pop_size, pop_chunk_size)
        ]
        per_source.append(chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0))
    return jnp.transpose(jnp.stack(per_source, axis=0), (1, 0, 2))


# Level-by-Level Streaming Helper Functions
# These functions query the CPPN one hierarchical level at a time: one level's queries,
# and the next level's input queries for its variance, are resident at once; W1 and W2
# are still assembled at full size.


def query_level_positions(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    level_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    pop_chunk_size: int = 50,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    num_cppn_outputs: int = 1,
    use_leo: bool = False,
) -> jnp.ndarray:
    """Query CPPN for a single level's positions.

    This is a level-specific version of batch_query_population_multi_source_chunked
    that queries only the positions for a single hierarchical level.

    A CPPN emits more than one output for two independent reasons: LEO (weight +
    expression) and ``dynamic_functions.mode='cppn_output'`` (weight + per-node activation
    index). Both need the trailing output axis preserved, so the multi-output query path is
    selected by ``use_leo or num_cppn_outputs > 1``, the same condition
    ``discover_sparse_hh_vectorized_multi_hop`` uses. Sending a multi-output CPPN
    down the single-output branch folds the output axis into the position axis via
    ``.flatten()`` and produces ``(pop, sources, positions * num_outputs)``.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        level_positions: Positions for this level only (num_level_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Number of genomes to process at once
        device_id: Device index for multi-GPU
        geometry_seeding_enabled: If True, add delta inputs to CPPN
        num_cppn_outputs: CPPN output count. >1 also selects the multi-output path
        use_leo: LEO is on: the multi-output path, whose channel LEO_CHANNEL is LEO's

    Returns:
        With one output and no LEO: (pop_size, num_sources, num_level_positions) array
        Otherwise: (pop_size, num_sources, num_level_positions, num_outputs) array with
            the weight in channel 0
    """
    pop_size = cppns_transformed[0].shape[0]
    num_sources = source_coords.shape[0]

    results_list = []

    # Sequential loop over sources (typically 3-10 for XOR)
    for source_idx in range(num_sources):
        source_coord = source_coords[source_idx]

        # Chunked processing over population
        chunk_results = []
        for chunk_start in range(0, pop_size, pop_chunk_size):
            chunk_end = min(chunk_start + pop_chunk_size, pop_size)

            # Extract chunk of CPPNs
            chunk_cppns = (
                cppns_transformed[0][chunk_start:chunk_end],
                cppns_transformed[1][chunk_start:chunk_end],
                cppns_transformed[2][chunk_start:chunk_end],
                cppns_transformed[3][chunk_start:chunk_end],
            )

            if use_leo or num_cppn_outputs > 1:
                # Multi-output: returns (chunk_size, num_positions, num_outputs).
                # This path preserves the trailing axis; the single-output one
                # flattens it into the position axis.
                chunk_weights = batch_query_population_positions_multi_output(
                    state, chunk_cppns, source_coord, level_positions,
                    outgoing, cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=geometry_seeding_enabled
                )
            else:
                chunk_weights = batch_query_population_positions(
                    state, chunk_cppns, source_coord, level_positions,
                    outgoing, cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=geometry_seeding_enabled
                )
            chunk_results.append(chunk_weights)

        # Concatenate chunks for this source
        source_weights = jnp.concatenate(chunk_results, axis=0)
        results_list.append(source_weights)

    # Stack and transpose
    result = jnp.stack(results_list, axis=0)
    if use_leo or num_cppn_outputs > 1:
        # (num_sources, pop, positions, outputs) -> (pop, num_sources, positions, outputs)
        return jnp.transpose(result, (1, 0, 2, 3))
    else:
        # (num_sources, pop_size, num_positions) -> (pop_size, num_sources, num_positions)
        return jnp.transpose(result, (1, 0, 2))


def compute_single_level_variance_chunked(
    query_slice_fn,
    grid: HierarchicalGridStructure,
    level: int,
    pop_size: int,
    max_elements: int,
) -> np.ndarray:
    """Variance for one level, computed in slabs so memory never scales with level size.

    `compute_single_level_variance` needs the whole of level+1 resident at once. For deep
    levels that is the allocation `max_lookahead_positions` budgets; a level over the budget is
    computed here in slabs, because substituting zeros would silently stop subdivision at every
    level below (see emr_hyperneat/tests/test_depth_lookahead.py).

    Chunking is exact here because each parent's variance depends only on its own 2x2
    block of children. Parent row r draws from child rows 2r and 2r+1, so a slab of parent
    rows maps to a *contiguous* span of the flattened child array and can be queried on its
    own. Results are concatenated in parent-row order, matching the unchunked layout
    exactly.

    Args:
        query_slice_fn: Callable (start, end) -> (pop_size, end - start) array of level+1
            weights for that flat position span.
        grid: Pre-computed grid structure.
        level: Current level (0-indexed); variance is over level+1's children.
        pop_size: Number of genomes in this chunk.
        max_elements: Soft ceiling on elements (pop_size x positions) held at once.

    Returns:
        Variance array (pop_size, num_cells_at_level), identical to the unchunked result.

    Raises:
        ValueError: If a single parent row cannot fit within `max_elements`. Returning
            zeros here is what caused the silent depth cap, so this fails loudly instead.
    """
    next_level_size = int(grid.level_sizes_static[level + 1])
    child_grid_size = int(np.sqrt(next_level_size))
    parent_grid_size = child_grid_size // 2

    # A parent row needs 2 child rows resident: 2 * child_grid_size * pop_size elements.
    elements_per_parent_row = 2 * child_grid_size * pop_size
    rows_per_chunk = max_elements // max(1, elements_per_parent_row)

    if rows_per_chunk < 1:
        raise ValueError(
            f"Cannot compute level {level} variance within the memory budget: one parent "
            f"row of level {level + 1} needs {elements_per_parent_row:,} elements "
            f"(2 child rows x {child_grid_size:,} cols x pop {pop_size}) but the budget is "
            f"{max_elements:,}. Reduce the population chunk size or raise the budget. "
            f"Substituting zeros here would silently truncate the substrate at level "
            f"{level}, which is the defect this path exists to prevent."
        )

    parts = []
    for row_start in range(0, parent_grid_size, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, parent_grid_size)
        n_parent_rows = row_end - row_start

        flat_start = 2 * row_start * child_grid_size
        flat_end = 2 * row_end * child_grid_size
        weights = query_slice_fn(flat_start, flat_end)

        blocks = np.asarray(weights).reshape(
            pop_size, n_parent_rows, 2, parent_grid_size, 2
        )
        blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(
            pop_size, n_parent_rows, parent_grid_size, 4
        )
        parts.append(blocks.var(axis=-1).reshape(pop_size, n_parent_rows * parent_grid_size))

    return np.concatenate(parts, axis=1)


def legacy_level0_variance(level0_weights: jnp.ndarray) -> jnp.ndarray:
    """The original level-0 rule (``variance_rule: legacy``) for the paths that compute variance
    one level at a time.

    Streaming and depth_offload derive each level's variance from its children
    (``compute_single_level_variance``), which is the canonical rule. Under ``legacy`` the standard
    path instead takes one variance per genome over level 0's own four raw weights
    (``compute_hierarchical_variances_batch``), so level 0 subdivides all or nothing. This gives the
    other paths the same value, repeated over level 0's four cells so their per-parent arithmetic
    is unchanged, and ``legacy`` holds on every execution mode, including one chosen at run time.

    Args:
        level0_weights: (pop_size, 4) raw CPPN weights from the first substrate input to the
            level-0 positions.

    Returns:
        (pop_size, 4) variance: each genome's single level-0 value in all four cells.
    """
    return jnp.broadcast_to(jnp.var(level0_weights, axis=1, keepdims=True), level0_weights.shape)


def compute_single_level_variance(
    next_level_weights: jnp.ndarray,
    grid: 'HierarchicalGridStructure',
    level: int,
) -> jnp.ndarray:
    """Compute variance for a single level using weights from next level.

    This is a streaming version of variance computation that only requires
    the next level's weights, not all weights at once.

    Args:
        next_level_weights: CPPN outputs for level+1 (pop_size, num_next_positions)
        grid: Pre-computed grid structure
        level: Current level (0-indexed)

    Returns:
        Variance array (pop_size, num_cells_at_level)
    """
    pop_size = next_level_weights.shape[0]

    if level == 0:
        # Level 0: variance over the 4 cells from level 1
        # Level 1 has 16 cells, grouped into 4 blocks of 4
        child_grid_size = 4  # Level 1 is 4x4
        parent_grid_size = 2  # Level 0 is 2x2

        child_grids = next_level_weights.reshape(pop_size, child_grid_size, child_grid_size)
        reshaped = child_grids.reshape(pop_size, parent_grid_size, 2, parent_grid_size, 2)
        reshaped = reshaped.transpose(0, 1, 3, 2, 4)
        blocks = reshaped.reshape(pop_size, parent_grid_size, parent_grid_size, 4)
        variances = jnp.var(blocks, axis=-1)
        return variances.reshape(pop_size, parent_grid_size * parent_grid_size)
    else:
        # Higher levels: variance of 2x2 child blocks from next level
        next_level_size = grid.level_sizes_static[level + 1]
        child_grid_size = int(np.sqrt(next_level_size))
        parent_grid_size = child_grid_size // 2

        child_grids = next_level_weights.reshape(pop_size, child_grid_size, child_grid_size)
        reshaped = child_grids.reshape(pop_size, parent_grid_size, 2, parent_grid_size, 2)
        reshaped = reshaped.transpose(0, 1, 3, 2, 4)
        blocks = reshaped.reshape(pop_size, parent_grid_size, parent_grid_size, 4)
        variances = jnp.var(blocks, axis=-1)
        return variances.reshape(pop_size, parent_grid_size * parent_grid_size)


# Multi-Output CPPN Query Functions
# These functions handle CPPNs with multiple outputs (weight + secondary channels).
# The key difference from standard query functions is that they preserve
# the output dimension rather than flattening it.

# Per-device cache for multi-output query functions
_per_device_multi_output_vmap_cache: Dict[Tuple[int, int], Any] = {}
_per_device_multi_output_cache_lock = threading.Lock()


def _create_multi_output_query_fn_for_device(cppn_forward: Any, device_id: int):
    """Create a fresh multi-output query function bound to a specific device.

    Unlike the standard query function, this one does not flatten the output,
    preserving the (num_outputs,) dimension for each position.

    Args:
        cppn_forward: The CPPN forward function
        device_id: Target device index

    Returns:
        A fresh JIT-compiled query function for this device that returns
        (pop_size, num_positions, num_outputs) arrays
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    with jax.default_device(device):
        def query_population_vmap_impl(state, inputs, cppns_nodes, cppns_conns,
                                       cppns_conn_attrs, cppns_node_attrs):
            """Query all CPPNs at all positions, preserving output dimension."""
            def query_single_cppn(cppn_tuple):
                """Query one CPPN at all positions."""
                # DON'T flatten - preserve (num_positions, num_outputs) shape
                return jax.vmap(
                    lambda x: cppn_forward(state, cppn_tuple, x)
                )(inputs)

            return jax.vmap(
                query_single_cppn,
                in_axes=((0, 0, 0, 0),)
            )((cppns_nodes, cppns_conns, cppns_conn_attrs, cppns_node_attrs))

        return jax.jit(query_population_vmap_impl, device=device)


def _get_multi_output_query_fn(device_id: int, cppn_forward: Any):
    """Get or create a device-specific multi-output query function."""
    cache_key = (device_id, id(cppn_forward))

    with _per_device_multi_output_cache_lock:
        if cache_key not in _per_device_multi_output_vmap_cache:
            _per_device_multi_output_vmap_cache[cache_key] = (
                _create_multi_output_query_fn_for_device(cppn_forward, device_id)
            )
        return _per_device_multi_output_vmap_cache[cache_key]


def query_cppn_multi_output(
    cppn_forward: Any,
    state: Any,
    inputs: jnp.ndarray,
    cppns_transformed: Tuple,
    device_id: int = 0,
) -> jnp.ndarray:
    """Query CPPN preserving the output dimension (weight + secondary channels).

    Args:
        cppn_forward: JIT-compiled CPPN forward function
        state: Algorithm state
        inputs: Input coordinates (num_positions, input_dim)
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        device_id: Device index for multi-GPU

    Returns:
        Array of shape (pop_size, num_positions, 2) where:
        - [:, :, 0] are weight outputs
        - [:, :, 1] are secondary outputs
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Move all inputs to target device
    state_on_device = jax.tree.map(lambda x: jax.device_put(x, device), state)
    inputs_on_device = jax.device_put(inputs, device)
    cppns_on_device = tuple(jax.device_put(arr, device) for arr in cppns_transformed)

    # Get device-specific multi-output query function
    query_fn = _get_multi_output_query_fn(device_id, cppn_forward)

    with jax.default_device(device):
        return query_fn(
            state_on_device,
            inputs_on_device,
            cppns_on_device[0],
            cppns_on_device[1],
            cppns_on_device[2],
            cppns_on_device[3]
        )


def batch_query_population_positions_multi_output(
    state: Any,
    cppns_transformed: Tuple,
    source_coord: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    agent_coord: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    """Query CPPN preserving the output dimension (weight + secondary channels).

    This is the multi-output version of batch_query_population_positions.
    Instead of returning (pop_size, num_positions), it returns
    (pop_size, num_positions, 2) with weight and secondary channels.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coord: Single source coordinate (2,)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        device_id: Device index for multi-GPU
        geometry_seeding_enabled: If True, add delta_x, delta_y to CPPN inputs (7D instead of 5D)
        agent_coord: Optional (2,) agent team position for swarm mode.
            When provided, prepends [agent_x, agent_y] to CPPN inputs
            for position-aware multi-agent policy generation.

    Returns:
        (pop_size, num_positions, 2) array where:
        - [:, :, 0] are weight outputs
        - [:, :, 1] are secondary outputs
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    source_coord = jax.device_put(source_coord, device)
    target_positions = jax.device_put(target_positions, device)

    # Build CPPN inputs: the same rows as the single-output query (see cppn_query_rows).
    inputs_np = cppn_query_rows(
        source_coord, target_positions, outgoing, geometry_seeding_enabled, agent_coord
    )
    inputs = jax.device_put(inputs_np, device)

    return query_cppn_multi_output(
        cppn_forward, state, inputs, cppns_transformed, device_id
    )


def batch_query_population_multi_source_multi_output(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    pop_chunk_size: int = 100,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    agent_coord: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    """Query CPPN from multiple sources, preserving the output dimension.

    This is the multi-output version of batch_query_population_multi_source_chunked.
    Returns weight and secondary channels for each source->target pair.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Population chunk size for memory management
        device_id: Device index for multi-GPU
        geometry_seeding_enabled: If True, add delta_x, delta_y to CPPN inputs (7D instead of 5D)
        agent_coord: Optional (2,) agent team position for swarm mode.
            When provided, prepends [agent_x, agent_y] to CPPN inputs.

    Returns:
        (pop_size, num_sources, num_positions, 2) array where:
        - [:, :, :, 0] are weight outputs
        - [:, :, :, 1] are secondary outputs
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    source_coords = jax.device_put(source_coords, device)
    target_positions = jax.device_put(target_positions, device)
    cppns_transformed = tuple(jax.device_put(arr, device) for arr in cppns_transformed)

    pop_size = cppns_transformed[0].shape[0]
    num_sources = source_coords.shape[0]

    results_list = []

    for source_idx in range(num_sources):
        source_coord = source_coords[source_idx]

        chunk_results = []
        for chunk_start in range(0, pop_size, pop_chunk_size):
            chunk_end = min(chunk_start + pop_chunk_size, pop_size)

            chunk_cppns = (
                cppns_transformed[0][chunk_start:chunk_end],
                cppns_transformed[1][chunk_start:chunk_end],
                cppns_transformed[2][chunk_start:chunk_end],
                cppns_transformed[3][chunk_start:chunk_end],
            )

            chunk_weights_multi = batch_query_population_positions_multi_output(
                state, chunk_cppns, source_coord, target_positions,
                outgoing, cppn_forward, device_id, geometry_seeding_enabled,
                agent_coord=agent_coord,
            )
            chunk_weights_multi = jax.device_put(chunk_weights_multi, device)
            chunk_results.append(chunk_weights_multi)

        # Concatenate chunks
        chunk_results_np = [np.asarray(c) for c in chunk_results]
        source_weights_multi = np.concatenate(chunk_results_np, axis=0)
        results_list.append(source_weights_multi)

    # Stack: (num_sources, pop_size, num_positions, 2)
    # Transpose to: (pop_size, num_sources, num_positions, 2)
    stacked = np.stack(results_list, axis=0)
    return np.transpose(stacked, (1, 0, 2, 3))


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("start", "end"))
def _accumulate_level_slice_donated(
    acc: jnp.ndarray, block: jnp.ndarray, *, start: int, end: int
) -> jnp.ndarray:
    """Slice-set with the accumulator buffer donated.

    Streaming keeps W1/W2 fully device-resident and updates them per level with
    functional `.at[...].set`, which holds input and output alive at once and so
    transiently doubles W1. At depth 7 and population 1000, one forced-streaming run with the
    query chunked ran out of memory allocating the 786,432,000 B level-7 slice-update operand.
    Donating the accumulator lets XLA write in place. Donation aliases buffers; it never
    changes values. The caller must immediately rebind its reference, and does. Static
    slice bounds retrace once per level, a count bounded by the depth.
    """
    return acc.at[:, :, start:end].set(block)


def _accumulate_level_slice(
    acc: jnp.ndarray, block: jnp.ndarray, start: int, end: int
) -> jnp.ndarray:
    """Backend-dispatching level-slice accumulate.

    CPU ignores donation and warns per trace, so the donated variant is used
    only where it buys memory (GPU). Both variants compute the same values.
    """
    if jax.default_backend() == "gpu":
        return _accumulate_level_slice_donated(acc, block, start=int(start), end=int(end))
    return acc.at[:, :, start:end].set(block)
