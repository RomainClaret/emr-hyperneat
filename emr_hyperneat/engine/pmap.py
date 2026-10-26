"""The pmap routes' kernels: the dense and recurrent evaluation of one network and of a batch, the pmap
objects built from them, and the FULL_PIPELINE pipeline's masks and its data-parallel core."""
import functools
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from emr_hyperneat.emrhyperneat_base import (
    compute_hierarchical_variances_batch_jit,
    compute_subdivision_masks_batch_jit,
)
from emr_hyperneat.engine.config import SparseHiddenConnections
from emr_hyperneat.engine.evaluation import forward_hybrid_vmapped
from emr_hyperneat.engine.settling import settle_hidden


# PMAP Evaluation Functions (Zero IPC Overhead Multi-GPU)
# These functions enable native JAX multi-GPU evaluation without Python IPC.
# Key design:
# - W1, W2, h->h connection data are replicated across all devices
# - Input/target data is sharded across devices (each device gets a portion)
# - Evaluation runs in parallel, returns partial errors for aggregation


def _pmap_eval_single_dense(w1: jnp.ndarray, w2: jnp.ndarray,
                             inputs: jnp.ndarray, targets: jnp.ndarray,
                             activate_time: int) -> jnp.ndarray:
    """Evaluate single network in dense (feedforward) mode - pmap compatible.

    Pure JAX function with no Python control flow.

    Args:
        w1: Input->Hidden weights, shape (n_inputs, n_hidden)
        w2: Hidden->Output weights, shape (n_hidden, n_outputs)
        inputs: Input samples, shape (batch_size, n_inputs)
        targets: Target outputs, shape (batch_size, n_outputs)
        activate_time: Number of activation steps (typically 1 for feedforward)

    Returns:
        Sum of squared errors for this genome on this data shard (scalar)
    """
    # Feedforward: inputs @ w1 -> tanh -> @ w2 -> sigmoid
    hidden = jnp.tanh(inputs @ w1)
    outputs = jax.nn.sigmoid(hidden @ w2)
    errors = (outputs - targets) ** 2
    return jnp.sum(errors)


def _pmap_eval_single_hybrid(w1: jnp.ndarray, w2: jnp.ndarray,
                              hh_from: jnp.ndarray, hh_to: jnp.ndarray,
                              hh_weights: jnp.ndarray, hh_valid: jnp.ndarray,
                              inputs: jnp.ndarray, targets: jnp.ndarray,
                              activate_time: int, num_hidden: int) -> jnp.ndarray:
    """Evaluate single network with h->h connections - pmap compatible.

    The standard path's settling rule (:func:`settle_hidden` with tanh): ``h(0) = 0``, then
    ``tanh(inputs @ W1 + recurrent(h))`` for ``activate_time`` steps, so the data-parallel
    strategy settles a network as a single-device run does.

    Args:
        w1: Input->Hidden weights, shape (n_inputs, n_hidden)
        w2: Hidden->Output weights, shape (n_hidden, n_outputs)
        hh_from: H->H source indices, shape (max_connections,)
        hh_to: H->H target indices, shape (max_connections,)
        hh_weights: H->H connection weights, shape (max_connections,)
        hh_valid: H->H validity mask, shape (max_connections,)
        inputs: Input samples, shape (batch_size, n_inputs)
        targets: Target outputs, shape (batch_size, n_outputs)
        activate_time: Number of activation steps
        num_hidden: Number of hidden positions (unused; the shape carries it)

    Returns:
        Sum of squared errors for this genome on this data shard (scalar)
    """
    del num_hidden
    hidden = settle_hidden(inputs @ w1, hh_from, hh_to, hh_weights, hh_valid,
                           activate_time, jnp.tanh)
    outputs = jax.nn.sigmoid(hidden @ w2)
    errors = (outputs - targets) ** 2
    return jnp.sum(errors)


def _pmap_eval_batch_dense(W1: jnp.ndarray, W2: jnp.ndarray,
                            inputs: jnp.ndarray, targets: jnp.ndarray,
                            activate_time: int) -> jnp.ndarray:
    """Evaluate population in dense mode on a data shard.

    Args:
        W1: Population input->hidden weights, shape (pop_size, n_inputs, n_hidden)
        W2: Population hidden->output weights, shape (pop_size, n_hidden, n_outputs)
        inputs: Data shard inputs, shape (shard_size, n_inputs)
        targets: Data shard targets, shape (shard_size, n_outputs)
        activate_time: Activation steps

    Returns:
        Sum of squared errors per genome on this shard, shape (pop_size,)
    """
    # vmap over population: each genome evaluated on same data shard
    return jax.vmap(
        lambda w1, w2: _pmap_eval_single_dense(w1, w2, inputs, targets, activate_time),
        in_axes=(0, 0)
    )(W1, W2)


def _pmap_eval_batch_hybrid(W1: jnp.ndarray, W2: jnp.ndarray,
                             hh_from: jnp.ndarray, hh_to: jnp.ndarray,
                             hh_weights: jnp.ndarray, hh_valid: jnp.ndarray,
                             inputs: jnp.ndarray, targets: jnp.ndarray,
                             activate_time: int, num_hidden: int) -> jnp.ndarray:
    """Evaluate population with h->h on a data shard.

    Args:
        W1: Population input->hidden weights, shape (pop_size, n_inputs, n_hidden)
        W2: Population hidden->output weights, shape (pop_size, n_hidden, n_outputs)
        hh_from: H->H source indices per genome, shape (pop_size, max_connections)
        hh_to: H->H target indices per genome, shape (pop_size, max_connections)
        hh_weights: H->H weights per genome, shape (pop_size, max_connections)
        hh_valid: H->H validity per genome, shape (pop_size, max_connections)
        inputs: Data shard inputs, shape (shard_size, n_inputs)
        targets: Data shard targets, shape (shard_size, n_outputs)
        activate_time: Activation steps
        num_hidden: Number of hidden positions

    Returns:
        Sum of squared errors per genome on this shard, shape (pop_size,)
    """
    # vmap over population: each genome evaluated on same data shard
    return jax.vmap(
        lambda w1, w2, hf, ht, hw, hv: _pmap_eval_single_hybrid(
            w1, w2, hf, ht, hw, hv, inputs, targets, activate_time, num_hidden
        ),
        in_axes=(0, 0, 0, 0, 0, 0)
    )(W1, W2, hh_from, hh_to, hh_weights, hh_valid)


# Create pmap-wrapped evaluation functions
# in_axes: None = replicated (same on all devices), 0 = sharded (different per device)
# Dense mode: W1, W2 replicated; inputs, targets sharded
_PMAP_EVAL_DENSE = jax.pmap(
    _pmap_eval_batch_dense,
    axis_name='devices',
    in_axes=(None, None, 0, 0, None),  # W1, W2, inputs, targets, activate_time
    static_broadcasted_argnums=(4,),   # activate_time is static
)

# Hybrid mode: W1, W2, h->h all replicated; inputs, targets sharded
_PMAP_EVAL_HYBRID = jax.pmap(
    _pmap_eval_batch_hybrid,
    axis_name='devices',
    in_axes=(None, None, None, None, None, None, 0, 0, None, None),
    # W1, W2, hh_from, hh_to, hh_weights, hh_valid, inputs, targets, activate_time, num_hidden
    static_broadcasted_argnums=(8, 9),  # activate_time, num_hidden are static
)


# Full Pipeline with H->H Discovery (FULL_PIPELINE_PARALLEL)
# The complete pipeline (CPPN queries + variance + weight matrices + eval) runs inside pmap, one
# data shard per device; with h->h connections the masks and the connections come from the
# discovery outside it (EMRHyperNEAT._pmap_pipeline_hh).


def _full_pipeline_masks(
    cppns_transformed: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    variance_threshold: float,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[int, ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    state: Any,
    cppn_forward: Any,
    geometry_seeding_enabled: bool = False,
    initial_depth: int = 0,
    legacy_level0_variance: bool = False,
) -> jnp.ndarray:
    """The pmap pipeline's subdivision masks.

    The CPPN queried from the first input to every grid position, the hierarchical variances and
    the subdivision masks, as `_full_pipeline_with_hh_core` computes them inside the pmap without
    h->h connections. With h->h connections the runner computes them once outside the pmap
    (`_full_pipeline_masks_jit`), and the discovery and the weight matrices share them.

    Returns:
        (pop, total_positions) bool masks of active positions.
    """
    def query_single_position(cppn_tuple, target_pos):
        bias = jnp.array([1.0])
        source_coord = input_coords[0]
        if geometry_seeding_enabled:
            delta_x = target_pos[0] - source_coord[0]
            delta_y = target_pos[1] - source_coord[1]
            delta_input = jnp.array([delta_x, delta_y])
            inp = jnp.concatenate([source_coord, target_pos, delta_input, bias])
        else:
            inp = jnp.concatenate([source_coord, target_pos, bias])
        return cppn_forward(state, cppn_tuple, inp)

    def get_variance_weights(cppn_tuple):
        return jax.vmap(lambda pos: query_single_position(cppn_tuple, pos))(all_positions).flatten()

    all_weights_for_variance = jax.vmap(
        get_variance_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))
    level_variances = compute_hierarchical_variances_batch_jit(
        all_weights_for_variance,
        level_sizes=level_sizes,
        level_offsets=level_offsets,
        level_grid_sizes=level_grid_sizes,
        num_levels=num_levels,
        legacy_level0_variance=legacy_level0_variance,
    )
    masks_A, _, _ = compute_subdivision_masks_batch_jit(
        level_variances,
        variance_threshold=variance_threshold,
        initial_depth=initial_depth,
        parent_indices_tuple=parent_indices,
        level_offsets=level_offsets,
        num_levels=num_levels,
        total_positions=total_positions,
    )
    return masks_A


_full_pipeline_masks_jit = jax.jit(
    _full_pipeline_masks,
    static_argnums=(4, 5, 6, 8, 9, 11, 12, 13, 14),
)


def _full_pipeline_with_hh_core(
    cppns_transformed: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    variance_threshold: float,
    band_threshold: float,
    max_weight: float,
    weight_thresh: float,
    activate_time: int,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[int, ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    max_sparse_conns: int,
    iteration_level: int,
    allow_hh: bool,
    state: Any,
    cppn_forward: Any,
    geometry_seeding_enabled: bool = False,
    initial_depth: int = 0,
    legacy_level0_variance: bool = False,
    allow_backward: bool = True,
    allow_lateral: bool = True,
    allow_self_loops: bool = False,
    hh: Optional[Tuple] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full pipeline inside pmap: CPPN queries, masks, weight matrices and evaluation on one device.

    With h->h connections (``allow_hh`` and ``iteration_level`` > 0) the masks and the sparse
    connections come in through ``hh``, from the runner's discovery outside the pmap: the bounded
    discovery SINGLE_GPU and EVAL_ONLY_PARALLEL run (``discover_sparse_hh_vectorized_multi_hop``),
    direct connections only (iteration level 1; levels above 1 are refused on entry to this
    pipeline).
    In 0.1.0 the core discovered them itself, querying every genome's CPPN at every pair of
    grid positions in one vmap (chunked only above 30,000 positions): one buffer of about
    P x N^2 x 401 bytes (69.48 GiB at depth 4 and population 100, on any number of devices), so
    the recurrent route ran out of memory on GPUs where SINGLE_GPU ran.

    Args:
        cppns_transformed: Transformed CPPNs for this GPU
        all_positions: Hierarchical grid positions
        input_coords: Input coordinates
        output_coords: Output coordinates
        inputs_batch: Problem inputs (sharded or replicated depending on strategy)
        targets_batch: Problem targets
        variance_threshold: Subdivision threshold
        band_threshold: H->H threshold
        max_weight: Weight scaling
        weight_thresh: Minimum weight threshold
        activate_time: Forward pass iterations
        level_sizes: Tuple of cell counts per level
        level_offsets: Tuple of cumulative offsets
        level_grid_sizes: Tuple of grid dimensions
        parent_indices: Parent index arrays
        num_levels: Number of levels
        total_positions: Total positions in grid
        max_sparse_conns: Maximum sparse connections per genome
        iteration_level: H->H iteration level; 0 skips the h->h discovery, but the runners pass 1
            for a configured level 0 with h->h connections (`_pmap_iteration_level`)
        allow_hh: Enable h->h discovery
        state: Algorithm state
        cppn_forward: CPPN forward function
        hh: With h->h, ``(masks_A, from_indices, to_indices, weights, valid_mask, num_valid)``
            for this device's genomes, from the discovery outside the pmap; None without h->h

    Returns:
        Tuple of (fitnesses, total_hh_count, active_counts_sum, w1_nonzero_sum,
                  w2_nonzero_sum, w1_total, w2_total)

    Raises:
        ValueError: h->h connections without ``hh``.
    """
    # Helper: Query single CPPN at all positions from one source
    def query_cppn_single_source(cppn_tuple, source_coord, outgoing):
        def query_single_position(target_pos):
            bias = jnp.array([1.0])
            if geometry_seeding_enabled:
                # Geometry seeding: add delta (coordinate differences) as CPPN inputs (Risi & Stanley 2012)
                # Delta values allow CPPNs to learn direction-specific patterns
                # Gaussian activation on delta naturally biases toward local connections (peaks at delta=0)
                if outgoing:
                    delta_x = target_pos[0] - source_coord[0]
                    delta_y = target_pos[1] - source_coord[1]
                    delta_input = jnp.array([delta_x, delta_y])
                    inp = jnp.concatenate([source_coord, target_pos, delta_input, bias])
                else:
                    delta_x = source_coord[0] - target_pos[0]
                    delta_y = source_coord[1] - target_pos[1]
                    delta_input = jnp.array([delta_x, delta_y])
                    inp = jnp.concatenate([target_pos, source_coord, delta_input, bias])
            else:
                if outgoing:
                    inp = jnp.concatenate([source_coord, target_pos, bias])
                else:
                    inp = jnp.concatenate([target_pos, source_coord, bias])
            return cppn_forward(state, cppn_tuple, inp)
        return jax.vmap(query_single_position)(all_positions).flatten()

    # Helper: Query single CPPN from multiple sources
    def query_cppn_multi_source(cppn_tuple, source_coords, outgoing):
        return jax.vmap(
            lambda src: query_cppn_single_source(cppn_tuple, src, outgoing)
        )(source_coords)

    # Variance query, hierarchical variances and subdivision masks (_full_pipeline_masks), or,
    # with h->h, the masks the discovery outside the pmap used
    with_hh = allow_hh and iteration_level > 0
    if with_hh and hh is None:
        raise ValueError("the pmap pipeline takes its h->h connections from the discovery outside "
                         "the pmap (hh); it no longer discovers them itself")
    if with_hh:
        masks_A = hh[0]
    else:
        masks_A = _full_pipeline_masks(
            cppns_transformed, all_positions, input_coords, variance_threshold, level_sizes,
            level_offsets, level_grid_sizes, parent_indices, num_levels, total_positions, state,
            cppn_forward, geometry_seeding_enabled, initial_depth, legacy_level0_variance,
        )

    # Query CPPN for input weights
    def get_input_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, input_coords, outgoing=True)

    input_all_weights = jax.vmap(
        get_input_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))

    # Query CPPN for output weights
    def get_output_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, output_coords, outgoing=False)

    output_all_weights = jax.vmap(
        get_output_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))

    # Build weight matrices with mask
    mask_broadcast = masks_A[:, None, :]

    W1_raw = jnp.tanh(input_all_weights) * max_weight
    W2_raw = jnp.tanh(output_all_weights) * max_weight

    W1 = jnp.where(
        mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
        W1_raw, 0.0
    )
    W2_masked = jnp.where(
        mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
        W2_raw, 0.0
    )
    W2 = jnp.transpose(W2_masked, (0, 2, 1))

    # Compute substrate efficiency metrics for multi-GPU aggregation
    active_counts = jnp.sum(masks_A, axis=1)  # Per-genome active positions
    active_counts_sum = jnp.sum(active_counts)  # Total across population
    w1_nonzero_sum = jnp.sum(W1 != 0)
    w2_nonzero_sum = jnp.sum(W2 != 0)
    w1_total = jnp.array(W1.size, dtype=jnp.int32)
    w2_total = jnp.array(W2.size, dtype=jnp.int32)

    # The h->h connections the runner discovered outside the pmap
    sparse_hh = None
    if with_hh:
        sparse_hh = SparseHiddenConnections(
            from_indices=hh[1], to_indices=hh[2], weights=hh[3], valid_mask=hh[4], num_valid=hh[5],
        )

    # Evaluate all networks
    if sparse_hh is not None:
        # Hybrid evaluation with sparse h->h
        def eval_single_hybrid(w1, w2, from_idx, to_idx, hh_w, valid):
            outputs = forward_hybrid_vmapped(
                inputs_batch, w1, w2, from_idx, to_idx, hh_w, valid, activate_time
            )
            errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        fitnesses = jax.vmap(eval_single_hybrid)(
            W1, W2,
            sparse_hh.from_indices, sparse_hh.to_indices,
            sparse_hh.weights, sparse_hh.valid_mask,
        )
    else:
        # Dense-only evaluation
        def eval_single_dense(w1, w2):
            hidden = jnp.tanh(inputs_batch @ w1)
            outputs = jax.nn.sigmoid(hidden @ w2)
            errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        fitnesses = jax.vmap(eval_single_dense)(W1, W2)

    fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

    # Compute total h->h count for metrics
    if sparse_hh is not None:
        # Sum across population to get total h->h connections
        total_hh_count = jnp.sum(sparse_hh.num_valid)
    else:
        total_hh_count = jnp.array(0, dtype=jnp.int32)

    # Return fitnesses, h->h count, and substrate efficiency metrics
    return (fitnesses, total_hh_count,
            active_counts_sum, w1_nonzero_sum, w2_nonzero_sum, w1_total, w2_total)


# FULL_PIPELINE_PARALLEL: Split data across GPUs, replicate full population
# in_axes: (None=replicated) for cppns, positions, coords, scalars, state and hh (28)
#          (0=sharded) for inputs, targets
# parent_indices (index 14) contains JAX arrays so not static
@functools.partial(
    jax.pmap,
    in_axes=(None, None, None, None, 0, 0) + (None,) * 23,
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27),
)
def _pmap_data_parallel_full_hh(
    cppns_transformed,    # None: replicated
    all_positions,        # None: replicated
    input_coords,         # None: replicated
    output_coords,        # None: replicated
    inputs_shard,         # 0: sharded (each GPU gets data slice)
    targets_shard,        # 0: sharded
    variance_threshold,   # None: replicated
    band_threshold,       # None: replicated
    max_weight,           # None: replicated
    weight_thresh,        # None: replicated
    activate_time,        # static (10)
    level_sizes,          # static (11)
    level_offsets,        # static (12)
    level_grid_sizes,     # static (13)
    parent_indices,       # None: replicated (JAX arrays - not static)
    num_levels,           # static (15)
    total_positions,      # static (16)
    max_sparse_conns,     # static (17)
    iteration_level,      # static (18)
    allow_hh,             # static (19)
    state,                # None: replicated
    cppn_forward,         # static (21)
    geometry_seeding_enabled,  # static (22) - Risi & Stanley 2012 geometry seeding
    initial_depth,        # static (23) - forces subdivision through this level
    legacy_level0_variance,  # static (24) - variance_rule: legacy level-0 rule
    allow_backward=True,  # static (25) - connection class
    allow_lateral=True,   # static (26)
    allow_self_loops=False,  # static (27)
    hh=None,              # None: replicated - with h->h, the discovery outside the pmap
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Data-parallel: full population on each GPU, different data slices.

    Returns:
        Tuple of (fitnesses, total_hh_count, active_counts_sum, w1_nonzero_sum,
                  w2_nonzero_sum, w1_total, w2_total) where each is sharded per GPU.
    """
    return _full_pipeline_with_hh_core(
        cppns_transformed, all_positions, input_coords, output_coords,
        inputs_shard, targets_shard,
        variance_threshold, band_threshold, max_weight, weight_thresh,
        activate_time, level_sizes, level_offsets, level_grid_sizes,
        parent_indices, num_levels, total_positions,
        max_sparse_conns, iteration_level, allow_hh, state, cppn_forward,
        geometry_seeding_enabled, initial_depth, legacy_level0_variance,
        allow_backward, allow_lateral, allow_self_loops, hh,
    )
