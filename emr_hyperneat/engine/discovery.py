"""Hidden-to-hidden discovery: the constraint masks and the extraction of sparse connections, the multi-hop
expansions, union and per-genome discovery with their caches and budgets, the max_sparse_conns record
and refusal, and the h->h connection cache."""
import functools
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from emr_hyperneat.emrhyperneat_base import (
    HierarchicalGridStructure,
    batch_query_population_multi_source_chunked,
)
from emr_hyperneat.engine.config import EMRConfig, HH_CACHE_TRIGGERS, SparseHiddenConnections
from emr_hyperneat.engine.leo import (
    LEO_CHANNEL,
    compute_leo_validity_mask,
    compute_leo_validity_mask_replacement,
)
from emr_hyperneat.engine.substrate_build import batch_query_population_multi_source_multi_output


# Per-device caches for multi-GPU execution

# Cache for device-specific vmap'd extract_sparse functions
_extract_sparse_cache = {}
_extract_sparse_cache_lock = threading.Lock()


def _get_extract_sparse_fn(device_id: int, num_active: int, max_sparse_conns: int):
    """Get or create a device-specific extract_sparse JIT+vmap function.

    This avoids JAX trace caching issues in multi-GPU settings by ensuring
    each device has its own JIT-compiled traced function.

    The function is wrapped in both jax.vmap and jax.jit to ensure
    device-specific compilation.
    """
    cache_key = (device_id, num_active, max_sparse_conns)

    with _extract_sparse_cache_lock:
        if cache_key not in _extract_sparse_cache:
            devices = jax.devices()
            device = devices[device_id] if device_id < len(devices) else devices[0]

            with jax.default_device(device):
                def extract_sparse_for_genome(mask, weights, active_idx):
                    """Extract sparse connections from dense mask/weights."""
                    flat_mask = mask.flatten()
                    flat_weights = weights.flatten()

                    # Get indices of valid connections
                    num_valid_max = min(max_sparse_conns, flat_mask.shape[0])
                    valid_indices = jnp.nonzero(flat_mask, size=num_valid_max, fill_value=-1)[0]

                    # Convert flat index back to (from, to) in local coords
                    from_idx_local = valid_indices // num_active
                    to_idx_local = valid_indices % num_active

                    # Convert local indices to global position indices
                    from_idx_global = jnp.where(
                        valid_indices >= 0,
                        active_idx[from_idx_local],
                        -1
                    )
                    to_idx_global = jnp.where(
                        valid_indices >= 0,
                        active_idx[to_idx_local],
                        -1
                    )

                    # Get weights
                    conn_weights = jnp.where(
                        valid_indices >= 0,
                        flat_weights[valid_indices],
                        0.0
                    )

                    # Pad to max_sparse_conns if needed
                    if num_valid_max < max_sparse_conns:
                        pad_size = max_sparse_conns - num_valid_max
                        from_idx_global = jnp.pad(from_idx_global, (0, pad_size), constant_values=-1)
                        to_idx_global = jnp.pad(to_idx_global, (0, pad_size), constant_values=-1)
                        conn_weights = jnp.pad(conn_weights, (0, pad_size), constant_values=0.0)

                    return from_idx_global, to_idx_global, conn_weights

                # Create JIT-compiled vmap'd version within device context
                # The JIT ensures device-specific compilation
                vmap_fn = jax.jit(jax.vmap(extract_sparse_for_genome))
                _extract_sparse_cache[cache_key] = vmap_fn

        return _extract_sparse_cache[cache_key]


# Cache for device-specific constraint mask functions
_constraint_mask_cache = {}
_constraint_mask_cache_lock = threading.Lock()


def _get_constraint_mask_fn(device_id: int, allow_backward: bool, allow_lateral: bool, allow_self_loops: bool):
    """Get or create a device-specific constraint mask JIT function.

    This avoids JAX trace caching issues in multi-GPU settings by ensuring
    each device has its own JIT-compiled traced function.
    """
    cache_key = (device_id, allow_backward, allow_lateral, allow_self_loops)

    with _constraint_mask_cache_lock:
        if cache_key not in _constraint_mask_cache:
            devices = jax.devices()
            device = devices[device_id] if device_id < len(devices) else devices[0]

            with jax.default_device(device):
                @jax.jit
                def compute_constraint_mask(source_coords, target_positions):
                    """Compute Y-validity and self-loop masks.

                    Uses closure-captured config values to avoid trace-time issues.

                    Logic:
                    - Start with all True
                    - If not allow_backward: restrict to y_source <= y_target (forward + lateral)
                    - If not allow_lateral: further restrict to y_source != y_target (exclude same-y)

                    This ensures:
                    - hidden_only (backward=F, lateral=F): forward only (y < y)
                    - with_lateral (backward=F, lateral=T): forward + lateral (y <= y)
                    - with_backward (backward=T, lateral=F): forward + backward (y != y when combined)
                    - full_recurrent (all T): everything allowed
                    """
                    # Get Y coordinates
                    y_sources = source_coords[:, 1]
                    y_targets = target_positions[:, 1]

                    # Initialize with all True
                    y_valid = jnp.ones((source_coords.shape[0], target_positions.shape[0]), dtype=bool)

                    # Use <= instead of < to include lateral when not excluding backward
                    if not allow_backward:
                        y_valid = y_valid & (y_sources[:, None] <= y_targets[None, :])
                    if not allow_lateral:
                        y_valid = y_valid & (y_sources[:, None] != y_targets[None, :])

                    # Self-loop mask
                    coord_match = jnp.all(
                        source_coords[:, None, :] == target_positions[None, :, :],
                        axis=-1
                    )

                    if allow_self_loops:
                        # Add self-loops to y_valid, not just self_mask
                        # Self-loops have y==y, which would be excluded by y_valid otherwise
                        y_valid = y_valid | coord_match
                        self_mask = jnp.ones((source_coords.shape[0], target_positions.shape[0]), dtype=bool)
                    else:
                        self_mask = ~coord_match

                    return y_valid, self_mask

                _constraint_mask_cache[cache_key] = compute_constraint_mask

        return _constraint_mask_cache[cache_key]


# Connection Constraint Filtering

def get_connection_constraint_mask_batched(
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    config: Optional[EMRConfig] = None,
    device_id: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Connection constraint masks of the recurrence configuration, every source against every
    target.

    Connection Types:
    - Forward (y_source < y_target): Always allowed
    - Lateral (y_source == y_target): Configurable via allow_lateral
    - Backward (y_source > y_target): Configurable via allow_backward
    - Self-loop (x,y identical): Configurable via allow_self_loops

    Uses a device-specific JIT cache to avoid trace caching issues in multi-GPU settings.

    Args:
        source_coords: Source coordinates - shape (num_sources, 2)
        target_positions: Target positions - shape (num_targets, 2)
        config: Configuration for recurrence features
        device_id: Target device index for multi-GPU execution

    Returns:
        Tuple of:
        - y_valid: (num_sources, num_targets) bool mask
        - self_mask: (num_sources, num_targets) bool mask
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Extract config values
    allow_backward = config.allow_backward if config else True
    allow_lateral = config.allow_lateral if config else True
    allow_self_loops = config.allow_self_loops if config else False

    # Ensure inputs are on correct device
    source_coords = jax.device_put(source_coords, device)
    target_positions = jax.device_put(target_positions, device)

    # Get device-specific JIT function from cache
    compute_fn = _get_constraint_mask_fn(device_id, allow_backward, allow_lateral, allow_self_loops)

    # Call the device-specific JIT function
    y_valid, self_mask = compute_fn(source_coords, target_positions)

    # Ensure outputs are on correct device
    y_valid = jax.device_put(y_valid, device)
    self_mask = jax.device_put(self_mask, device)

    return y_valid, self_mask


# Multi-Hop Algorithm

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def compute_multi_hop_connections_matrix_power(
    adjacency_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    weight_threshold: float,
    iteration_level: int,
    hop_decay_factor: float,
    max_weight: float,
    second_factor: str = 'masked',
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-hop h->h expansion by repeated matrix products.

    Let ``A = where(valid_mask, adjacency_weights, 0)`` be the genome's direct connections. For
    ``iteration_level`` k the function forms the hop terms ``decay^(h-1) A^h`` for h = 1..k and
    keeps, per pair, the term of largest magnitude (a max merge, not a sum: a strong direct
    connection is never replaced by a weaker indirect one). Each product is thresholded at
    ``weight_threshold`` before the decay is applied; the result is clipped to ``max_weight`` and a
    pair is valid where ``|A_total| > weight_threshold``.

    ``second_factor`` chooses the right operand of every product (a static argument):

    - ``'masked'`` (default): ``A`` itself, so every hop is an existing direct connection, the
      composite connections stay within the genome's own positions, and hop h carries
      ``decay^(h-1)`` as stated above. The callers then intersect the result with the connection
      class (a forward edge composed with a backward one may land lateral or on its own source).
    - ``'raw'``: 0.1.0's code: the raw ``adjacency_weights`` (at the union call site these span
      the population's union of active positions, so a genome's composite connections could end
      on positions it did not have active, ignore its connection class and depend on the other
      genomes), with the decayed product carried into the next hop (hop 3 carried ``decay^3``).
      Kept bit for bit to reproduce recorded runs.

    JIT-compatible because ``iteration_level`` and ``second_factor`` are static: the Python loop
    compiles away into a fixed graph.

    Args:
        adjacency_weights: CPPN-derived weights, shape (num_active, num_active).
        valid_mask: Boolean mask of the direct connections, same shape.
        weight_threshold: Minimum magnitude a product must reach to count as a connection.
        iteration_level: Number of hops k (static).
        hop_decay_factor: Per-hop attenuation; hop h is scaled by decay^(h-1).
        max_weight: Clip magnitude of the accumulated weights.
        second_factor: ``'masked'`` or ``'raw'`` (static).

    Returns:
        ``(A_total, multi_hop_valid)``, both of shape (num_active, num_active).
    """
    if second_factor not in ('masked', 'raw'):
        raise ValueError(f"second_factor must be 'masked' or 'raw', got {second_factor!r}")
    A_current = jnp.where(valid_mask, adjacency_weights, 0.0)
    second = A_current if second_factor == 'masked' else adjacency_weights
    A_total = A_current

    # Static unroll: compiles away because iteration_level is static.
    for hop in range(1, iteration_level):
        A_power = A_current @ second
        A_power = jnp.where(jnp.abs(A_power) > weight_threshold, A_power, 0.0)
        A_decayed = A_power * (hop_decay_factor ** hop)
        A_total = jnp.where(jnp.abs(A_decayed) > jnp.abs(A_total), A_decayed, A_total)
        A_current = A_power if second_factor == 'masked' else A_decayed

    A_total = jnp.clip(A_total, -max_weight, max_weight)
    multi_hop_valid = jnp.abs(A_total) > weight_threshold
    if second_factor == 'masked':
        # The expansion never removes a direct connection. Under LEO replacement mode a direct
        # connection can carry |w| <= weight_threshold (the LEO gate replaced the weight gate), and
        # the rule above dropped it at level >= 2 (63 of 2,415 direct connections on XOR, depth 3,
        # population 30). Under compound or plain gating every direct cell has |w| >
        # weight_threshold and |A_total| >= |w| (the merge only replaces by a larger magnitude, and
        # |w| <= max_weight survives the clip), so this changes nothing there. 'raw' keeps 0.1.0's
        # code.
        multi_hop_valid = multi_hop_valid | valid_mask
    return A_total, multi_hop_valid


def compute_multi_hop_connections_fori_loop(
    adjacency_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    weight_threshold: float,
    iteration_level: int,
    hop_decay_factor: float,
    max_weight: float,
    second_factor: str = 'masked',
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The computation of ``compute_multi_hop_connections_matrix_power`` with ``lax.fori_loop``.

    Works with a dynamic ``iteration_level``. The decay is a traced float32 power here and a Python
    float there, so the two implementations can differ in the last bit. ``second_factor`` as above.
    """
    if second_factor not in ('masked', 'raw'):
        raise ValueError(f"second_factor must be 'masked' or 'raw', got {second_factor!r}")
    A_init = jnp.where(valid_mask, adjacency_weights, 0.0)
    second = A_init if second_factor == 'masked' else adjacency_weights

    def hop_step(hop_idx, carry):
        A_current, A_total = carry
        A_power = A_current @ second
        A_power = jnp.where(jnp.abs(A_power) > weight_threshold, A_power, 0.0)
        A_decayed = A_power * (hop_decay_factor ** (hop_idx + 1))
        A_total = jnp.where(jnp.abs(A_decayed) > jnp.abs(A_total), A_decayed, A_total)
        return ((A_power if second_factor == 'masked' else A_decayed), A_total)

    _, A_total = lax.fori_loop(0, iteration_level - 1, hop_step, (A_init, A_init))

    A_total = jnp.clip(A_total, -max_weight, max_weight)
    multi_hop_valid = jnp.abs(A_total) > weight_threshold
    if second_factor == 'masked':
        # The expansion never removes a direct connection. Under LEO replacement mode a direct
        # connection can carry |w| <= weight_threshold (the LEO gate replaced the weight gate), and
        # the rule above dropped it at level >= 2 (63 of 2,415 direct connections on XOR, depth 3,
        # population 30). Under compound or plain gating every direct cell has |w| >
        # weight_threshold and |A_total| >= |w| (the merge only replaces by a larger magnitude, and
        # |w| <= max_weight survives the clip), so this changes nothing there. 'raw' keeps 0.1.0's
        # code.
        multi_hop_valid = multi_hop_valid | valid_mask
    return A_total, multi_hop_valid


# Vectorized Discovery with Multi-Hop

# Per-device JIT cache for sparse discovery bitwise AND operations
# Without this, JAX reuses cached JIT traces across devices
# causing "Buffer on device cuda:X but replica assigned to cuda:Y" errors
_sparse_discovery_jit_cache = {}


def _get_sparse_discovery_combine_fn(device):
    """Get or create a device-specific JIT function for combining validity masks.

    This prevents JAX from reusing JIT traces across devices which causes
    device placement errors in multi-GPU execution.
    """
    cache_key = (id(device), device.id if hasattr(device, 'id') else 0)

    if cache_key not in _sparse_discovery_jit_cache:
        def combine_validity_masks(source_valid, target_valid, weight_valid, constraint_mask):
            """Combine validity masks with bitwise AND.

            All inputs should be on the target device before calling.
            """
            return source_valid & target_valid & weight_valid & constraint_mask

        # JIT compile with explicit device pinning
        _sparse_discovery_jit_cache[cache_key] = jax.jit(combine_validity_masks, device=device)

    return _sparse_discovery_jit_cache[cache_key]


#: The union discovery's compiled multi-hop expansions, by device and the settings they close over.
_union_multi_hop_cache: Dict[Tuple[Any, ...], Callable] = {}
_union_multi_hop_cache_lock = threading.Lock()


def _get_union_multi_hop_fn(device: Any, algorithm: str, band_threshold: float,
                            iteration_level: int, hop_decay_factor: float, max_weight: float,
                            second_factor: str) -> Callable:
    """The union discovery's multi-hop expansion over a population, ``jit(vmap(...))`` of
    :func:`compute_multi_hop_connections_matrix_power` (``algorithm`` 'matrix_power') or
    :func:`compute_multi_hop_connections_fori_loop`, made once per device and setting.

    Made on every call, the function would be traced and compiled again on every
    discovery, so every generation at iteration level 2 or more under ``hh_discovery_mode:
    union``. A setting keys with its type, and the expansion closes over the value as given.
    """
    settings = (algorithm, band_threshold, iteration_level, hop_decay_factor, max_weight,
                second_factor)
    key = (id(device), getattr(device, 'id', 0)) + tuple((type(v), v) for v in settings)
    with _union_multi_hop_cache_lock:
        expand_population = _union_multi_hop_cache.get(key)
        if expand_population is None:
            expand = (compute_multi_hop_connections_matrix_power if algorithm == "matrix_power"
                      else compute_multi_hop_connections_fori_loop)

            def expand_single_genome(adjacency, valid):
                return expand(adjacency, valid, band_threshold, iteration_level,
                              hop_decay_factor, max_weight, second_factor)

            expand_population = jax.jit(jax.vmap(expand_single_genome))
            _union_multi_hop_cache[key] = expand_population
    return expand_population


def _discover_sparse_hh_multi_hop_single_chunk(
    state: Any,
    cppns_transformed: Tuple,
    h_grid: HierarchicalGridStructure,
    masks_A: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    config: EMRConfig,
    cppn_forward: Any,
    pop_chunk_size: int = 100,
    verbose: bool = False,
    global_union_active: Optional[jnp.ndarray] = None,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    use_leo: bool = False,
    num_cppn_outputs: int = 1,
    leo_threshold: float = 0.2,
    leo_mode: str = 'compound',
    stats_out: Optional[Dict[str, Any]] = None,
) -> SparseHiddenConnections:
    """Fully vectorized sparse h->h discovery with multi-hop expansion.

    Pipeline:
    1. Query CPPN for all active->active position pairs
    2. Apply Y-coordinate constraints (from config)
    3. Apply weight threshold for direct connections
    4. Expand to multi-hop connections via matrix power
    5. Convert dense multi-hop matrix to sparse format

    Args:
        state: Algorithm state
        cppns_transformed: Transformed CPPNs (pop_size, ...)
        h_grid: Hierarchical grid structure
        masks_A: Active position masks - shape (pop, total_positions)
        band_threshold: Weight threshold
        max_weight: Weight scaling
        config: Extended configuration with multi-hop settings
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Population chunk size for queries
        global_union_active: Optional pre-computed global union of active positions.
            If provided, used instead of computing from masks_A. This ensures
            deterministic behavior in multi-GPU settings where masks_A is sharded.

    Returns:
        SparseHiddenConnections with multi-hop expanded connections
    """
    pop_size = cppns_transformed[0].shape[0]
    total_positions = h_grid.total_positions
    max_sparse_conns = config.max_sparse_conns

    # Move all input arrays to target device for multi-GPU execution
    # Without this, arrays stay on cuda:0 causing device placement errors
    # when this function is called from a cuda:1 thread
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Move h_grid positions to target device
    all_positions = jax.device_put(h_grid.all_positions, device)

    # Move masks_A to target device (critical for active_indices computation)
    masks_A = jax.device_put(masks_A, device)

    # Move cppns_transformed to target device
    cppns_transformed = tuple(jax.device_put(arr, device) for arr in cppns_transformed)

    # Move state to target device
    state = jax.tree.map(lambda x: jax.device_put(x, device), state)

    if getattr(config, 'hh_discovery_mode', 'union') == 'per_genome' and not config.use_dense_discovery:
        return _discover_sparse_hh_per_genome(
            state, cppns_transformed, h_grid, masks_A, band_threshold, max_weight, config,
            cppn_forward, verbose=verbose, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled, use_leo=use_leo,
            num_cppn_outputs=num_cppn_outputs, leo_threshold=leo_threshold, leo_mode=leo_mode,
            stats_out=stats_out,
        )

    # Get union of active positions across population
    # Use global union if provided (for multi-GPU determinism), otherwise compute locally
    if global_union_active is not None:
        union_active = jax.device_put(global_union_active, device)
    else:
        union_active = jnp.any(masks_A, axis=0)  # (total_positions,)
        union_active = jax.device_put(union_active, device)
    active_indices = jnp.nonzero(union_active, size=total_positions, fill_value=-1)[0]
    # Ensure active_indices is on correct device for subsequent indexing
    active_indices = jax.device_put(active_indices, device)
    num_active = int(jnp.sum(union_active))

    # Diagnostic logging (override with DEBUG_PHASE2=1 env var)
    _debug_phase2 = os.environ.get('DEBUG_PHASE2', '0') == '1'
    if verbose or _debug_phase2:
        print(f"[H->H Discovery] iteration_level={config.iteration_level}, "
              f"multi_hop_algorithm={config.multi_hop_algorithm}, "
              f"hop_decay_factor={config.hop_decay_factor}, "
              f"multi_hop_second_factor={config.multi_hop_second_factor}")
        print(f"[H->H Discovery] allow_backward={config.allow_backward}, "
              f"allow_lateral={config.allow_lateral}, "
              f"allow_self_loops={config.allow_self_loops}")
        print(f"[H->H Discovery] num_active={num_active}, pop_size={pop_size}")

    if num_active == 0:
        # No active positions - return empty connections
        return SparseHiddenConnections(
            from_indices=jnp.full((pop_size, max_sparse_conns), -1, dtype=jnp.int32),
            to_indices=jnp.full((pop_size, max_sparse_conns), -1, dtype=jnp.int32),
            weights=jnp.zeros((pop_size, max_sparse_conns), dtype=jnp.float32),
            valid_mask=jnp.zeros((pop_size, max_sparse_conns), dtype=bool),
            num_valid=jnp.zeros(pop_size, dtype=jnp.int32),
        )

    # Get active position coordinates
    active_coords = all_positions[active_indices[:num_active]]  # (num_active, 2)
    # Ensure active_coords is on correct device
    active_coords = jax.device_put(active_coords, device)

    # Query every active->active pair in one batch
    # Pass device_id to ensure device-specific vmap traces for multi-GPU
    if use_leo or num_cppn_outputs > 1:
        # Multi-output mode: use LEO query path to preserve output dimension.
        # batch_query_population_multi_source_chunked calls .flatten() which
        # corrupts multi-output CPPNs (turns (N, K) into (N*K,)); the multi-output
        # query path preserves the last dimension correctly.
        active_to_active_outputs = batch_query_population_multi_source_multi_output(
            state, cppns_transformed, active_coords, active_coords,
            outgoing=True, cppn_forward=cppn_forward,
            pop_chunk_size=pop_chunk_size, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled
        )
        # Extract weight channel (always output index 0)
        active_to_active_weights = active_to_active_outputs[:, :, :, 0]
        active_to_active_weights = jnp.tanh(active_to_active_weights) * max_weight
        # Extract LEO channel only when LEO is actually enabled
        if use_leo and active_to_active_outputs.shape[-1] > LEO_CHANNEL:
            active_to_active_leo = jnp.tanh(active_to_active_outputs[:, :, :, LEO_CHANNEL])
        else:
            active_to_active_leo = None
    else:
        # Standard single-output mode: Query returns (pop, num_active, num_active)
        active_to_active_weights = batch_query_population_multi_source_chunked(
            state, cppns_transformed, active_coords, active_coords,
            outgoing=True, cppn_forward=cppn_forward,
            pop_chunk_size=pop_chunk_size, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled
        )
        # Apply tanh activation and scale
        active_to_active_weights = jnp.tanh(active_to_active_weights) * max_weight
        active_to_active_leo = None

    # Apply Y-coordinate constraints
    # Pass device_id for device-aware execution
    y_valid, self_mask = get_connection_constraint_mask_batched(
        active_coords, active_coords, config, device_id=device_id
    )
    # Avoid fancy indexing ([None, :, :]) which triggers cached JIT traces
    # Use explicit reshape instead to avoid trace caching across devices
    combined_mask = y_valid & self_mask  # (num_active, num_active)
    combined_mask = jax.device_put(combined_mask, device)
    # Reshape to (1, num_active, num_active) for broadcasting
    constraint_mask = jnp.reshape(combined_mask, (1,) + combined_mask.shape)
    constraint_mask = jax.device_put(constraint_mask, device)

    # Step 3: Build validity masks
    if use_leo and active_to_active_leo is not None:
        # LEO gates existence, with the weight threshold ('compound') or alone
        # ('replacement'), at the configured threshold.
        if leo_mode == 'replacement':
            weight_valid = compute_leo_validity_mask_replacement(
                active_to_active_leo, leo_threshold=leo_threshold
            )
        else:
            weight_valid = compute_leo_validity_mask(
                active_to_active_weights, active_to_active_leo,
                weight_threshold=band_threshold, leo_threshold=leo_threshold
            )
    else:
        # Standard mode: Just weight threshold
        weight_valid = jnp.abs(active_to_active_weights) > band_threshold
    # Ensure weight_valid is on correct device
    weight_valid = jax.device_put(weight_valid, device)

    # Analyze CPPN weights by connection type (DEBUG_CPPN_WEIGHTS=1)
    _debug_cppn_weights = os.environ.get('DEBUG_CPPN_WEIGHTS', '0') == '1'
    if _debug_cppn_weights:
        # Categorize connections by y-coordinate relationship
        y_coords = active_coords[:, 1]  # (num_active,)
        y_sources = y_coords[:, None]  # (num_active, 1)
        y_targets = y_coords[None, :]  # (1, num_active)

        # Connection type masks (num_active, num_active)
        forward_mask = y_sources < y_targets  # y1 < y2
        backward_mask = y_sources > y_targets  # y1 > y2
        lateral_mask = y_sources == y_targets  # y1 == y2

        # Self-loop mask: same (x, y) coordinates
        x_coords = active_coords[:, 0]
        self_loop_mask = (x_coords[:, None] == x_coords[None, :]) & lateral_mask

        # Exclude self-loops from lateral
        lateral_only_mask = lateral_mask & ~self_loop_mask

        # Get raw weights (first genome as representative)
        raw_weights = active_to_active_weights[0]  # (num_active, num_active)

        # Stats per connection type
        def get_weight_stats(mask, weights):
            masked_w = jnp.where(mask, weights, jnp.nan)
            valid_w = masked_w[~jnp.isnan(masked_w)]
            if valid_w.size == 0:
                return 0, 0.0, 0.0, 0.0, 0.0, 0
            above_thresh = jnp.sum(jnp.abs(valid_w) > band_threshold)
            return (
                int(jnp.sum(mask)),  # total connections
                float(jnp.mean(jnp.abs(valid_w))),  # mean abs weight
                float(jnp.std(jnp.abs(valid_w))),   # std abs weight
                float(jnp.min(jnp.abs(valid_w))),   # min abs weight
                float(jnp.max(jnp.abs(valid_w))),   # max abs weight
                int(above_thresh),  # above threshold
            )

        print(f"\n[CPPN weight analysis] band_threshold={band_threshold}")
        print(f"{'Category':<15} {'Count':>8} {'Mean|w|':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'>thresh':>10} {'%pass':>8}")
        print("-" * 85)

        for name, mask in [
            ("forward", forward_mask),
            ("backward", backward_mask),
            ("lateral", lateral_only_mask),
            ("self_loop", self_loop_mask),
        ]:
            count, mean_w, std_w, min_w, max_w, above = get_weight_stats(mask, raw_weights)
            pct = 100.0 * above / max(count, 1)
            print(f"{name:<15} {count:>8} {mean_w:>10.6f} {std_w:>10.6f} {min_w:>10.6f} {max_w:>10.6f} {above:>10} {pct:>7.2f}%")

        # Overall stats
        total = int(forward_mask.sum() + backward_mask.sum() + lateral_only_mask.sum() + self_loop_mask.sum())
        weight_above = int(jnp.sum(jnp.abs(raw_weights) > band_threshold))
        print("-" * 85)
        print(f"{'TOTAL':<15} {total:>8} {'':<10} {'':<10} {'':<10} {'':<10} {weight_above:>10} {100.0*weight_above/max(total,1):>7.2f}%")
        print()

    if config.use_dense_discovery:
        # Dense discovery (vanilla-style): only use weight threshold + constraint
        # This finds ~15-20x more connections, better for hybrid at low populations
        # The per-genome active position filtering is skipped
        direct_valid = weight_valid & constraint_mask
        if verbose:
            print("[H->H Discovery] Dense mode: skipping per-genome filtering")
    else:
        # Sparse discovery (default): apply per-genome active position masks
        # Only counts connections where both source and target are active for that genome
        # This is more "correct" but finds fewer connections
        genome_source_active = masks_A[:, active_indices[:num_active]]  # (pop, num_active)
        genome_target_active = masks_A[:, active_indices[:num_active]]  # (pop, num_active)
        # Ensure on correct device
        genome_source_active = jax.device_put(genome_source_active, device)
        genome_target_active = jax.device_put(genome_target_active, device)

        source_valid = genome_source_active[:, :, None]  # (pop, num_active, 1)
        target_valid = genome_target_active[:, None, :]  # (pop, 1, num_active)

        # Ensure reshaped arrays are on correct device
        source_valid = jax.device_put(source_valid, device)
        target_valid = jax.device_put(target_valid, device)

        # Use per-device JIT function to avoid cross-device JIT trace reuse
        # Without this, JAX reuses JIT traces from cuda:0 on cuda:1 buffers causing errors
        combine_fn = _get_sparse_discovery_combine_fn(device)
        direct_valid = combine_fn(source_valid, target_valid, weight_valid, constraint_mask)

    # Direct connection stats
    if stats_out is not None:
        stats_out['direct_per_genome'] = jnp.sum(direct_valid, axis=(1, 2))

    if verbose:
        direct_count = int(jnp.sum(direct_valid))
        constraint_count = int(jnp.sum(constraint_mask))
        weight_count = int(jnp.sum(weight_valid))
        print(f"[H->H Discovery] constraint_mask_valid={constraint_count}, "
              f"weight_above_thresh={weight_count}, direct_valid={direct_count}")

    # Breakdown by connection type after all filtering (DEBUG_CPPN_WEIGHTS=1)
    if _debug_cppn_weights:
        # Re-create connection type masks
        y_coords = active_coords[:, 1]
        x_coords = active_coords[:, 0]
        y_sources = y_coords[:, None]
        y_targets = y_coords[None, :]
        forward_mask = y_sources < y_targets
        backward_mask = y_sources > y_targets
        lateral_mask = y_sources == y_targets
        self_loop_mask = (x_coords[:, None] == x_coords[None, :]) & lateral_mask
        lateral_only_mask = lateral_mask & ~self_loop_mask

        # Use first genome as representative (index [0] for the genome dimension)
        valid_first_genome = direct_valid[0] if direct_valid.ndim == 3 else direct_valid

        # Count by type
        fwd_valid = int(jnp.sum(valid_first_genome & forward_mask))
        bwd_valid = int(jnp.sum(valid_first_genome & backward_mask))
        lat_valid = int(jnp.sum(valid_first_genome & lateral_only_mask))
        self_valid = int(jnp.sum(valid_first_genome & self_loop_mask))
        total_valid = int(jnp.sum(valid_first_genome))

        print("\n[Filtered connection breakdown] (after constraint + weight + active filtering)")
        print(f"  forward:   {fwd_valid:>6} connections")
        print(f"  backward:  {bwd_valid:>6} connections")
        print(f"  lateral:   {lat_valid:>6} connections")
        print(f"  self_loop: {self_valid:>6} connections")
        print(f"  TOTAL:     {total_valid:>6} connections")
        print(f"  Config: allow_backward={config.allow_backward}, allow_lateral={config.allow_lateral}, allow_self_loops={config.allow_self_loops}")

        # Per-genome active position analysis
        if not config.use_dense_discovery:
            active_per_genome = jnp.sum(masks_A[:, active_indices[:num_active]], axis=1)
            mean_active = float(jnp.mean(active_per_genome))
            min_active = int(jnp.min(active_per_genome))
            max_active = int(jnp.max(active_per_genome))

            # Y-level distribution for first genome
            first_genome_active = masks_A[0, active_indices[:num_active]]
            active_y_coords = y_coords[first_genome_active]
            unique_y = jnp.unique(active_y_coords, size=num_active, fill_value=-1)
            unique_y = unique_y[unique_y >= 0]
            n_unique_y = int(unique_y.shape[0])

            print("\n[Per-genome active analysis]")
            print(f"  Active positions per genome: min={min_active}, mean={mean_active:.1f}, max={max_active}")
            print(f"  First genome: {int(jnp.sum(first_genome_active))} active positions across {n_unique_y} unique y-levels")

            # Count positions per y-level for first genome
            if n_unique_y > 0:
                for y_val in unique_y[:min(5, n_unique_y)]:  # Show first 5 y-levels
                    count = int(jnp.sum(active_y_coords == y_val))
                    print(f"    y={float(y_val):.3f}: {count} positions")

    # Multi-hop expansion (vmapped over population)
    if config.iteration_level > 1:
        expand_population = _get_union_multi_hop_fn(
            device, config.multi_hop_algorithm, band_threshold, config.iteration_level,
            config.hop_decay_factor, max_weight, config.multi_hop_second_factor)

        # Ensure inputs are on correct device before multi-hop expansion
        active_to_active_weights = jax.device_put(active_to_active_weights, device)
        direct_valid = jax.device_put(direct_valid, device)

        # Compute chunk size for multi-hop discovery to prevent OOM
        # The vmap creates (pop_size, num_active, num_active) intermediates
        multi_hop_chunk_size = config.multi_hop_chunk_size
        if multi_hop_chunk_size is None:
            # Auto-compute based on num_active positions
            if num_active < 100:
                multi_hop_chunk_size = 0  # No chunking needed
            elif num_active < 500:
                multi_hop_chunk_size = min(200, pop_size)
            else:
                multi_hop_chunk_size = min(50, pop_size)

        if verbose:
            print(f"[H->H Discovery] multi_hop_chunk_size={multi_hop_chunk_size}, "
                  f"num_active={num_active}, pop_size={pop_size}")

        # Use device context and JIT wrapper to ensure proper device compilation
        if multi_hop_chunk_size and multi_hop_chunk_size > 0 and pop_size > multi_hop_chunk_size:
            # Chunked processing to prevent OOM on large populations
            weight_chunks = []
            valid_chunks = []
            for chunk_start in range(0, pop_size, multi_hop_chunk_size):
                chunk_end = min(chunk_start + multi_hop_chunk_size, pop_size)
                with jax.default_device(device):
                    chunk_w, chunk_v = expand_population(
                        active_to_active_weights[chunk_start:chunk_end],
                        direct_valid[chunk_start:chunk_end]
                    )
                weight_chunks.append(chunk_w)
                valid_chunks.append(chunk_v)
            multi_hop_weights = jnp.concatenate(weight_chunks, axis=0)
            multi_hop_valid = jnp.concatenate(valid_chunks, axis=0)
        else:
            # Original: process all at once
            with jax.default_device(device):
                multi_hop_weights, multi_hop_valid = expand_population(
                    active_to_active_weights, direct_valid
                )

        if verbose:
            multi_hop_count = int(jnp.sum(multi_hop_valid))
            print(f"[H->H Discovery] Multi-hop expanded: multi_hop_valid={multi_hop_count} "
                  f"(added {multi_hop_count - int(jnp.sum(direct_valid))} transitive)")
    else:
        # No multi-hop (iteration_level = 1 means direct connections only)
        multi_hop_weights = jnp.where(direct_valid, active_to_active_weights, 0.0)
        multi_hop_valid = direct_valid
        if verbose:
            print("[H->H Discovery] No multi-hop (iteration_level=1): using direct only")

    if config.iteration_level > 1 and config.multi_hop_second_factor == 'masked':
        # Composites keep the preset's connection class: a forward edge composed with a backward
        # one lands lateral or on its own source, and those are dropped, so a with_backward network
        # holds only forward and backward edges at every level. 'raw' keeps 0.1.0's behavior bit for
        # bit.
        multi_hop_valid = multi_hop_valid & constraint_mask
        multi_hop_weights = jnp.where(multi_hop_valid, multi_hop_weights, 0.0)

    if stats_out is not None:
        # Direct pairs whose weight the max-magnitude merge swapped for a composite.
        direct_weights = jnp.where(direct_valid, active_to_active_weights, 0.0)
        stats_out['replaced_per_genome'] = jnp.sum(direct_valid & (multi_hop_weights != direct_weights), axis=(1, 2))

    if stats_out is not None:
        # Before the max_sparse_conns cap (the extraction keeps the first pairs in flat order).
        stats_out['uncapped_per_genome'] = jnp.sum(multi_hop_valid, axis=(1, 2))

    # Convert dense to sparse format using per-device cached vmap
    # The _get_extract_sparse_fn returns a device-specific traced vmap function
    # to avoid JAX trace caching issues across multi-GPU threads
    pop_size = multi_hop_valid.shape[0]
    active_indices_broadcast = jnp.broadcast_to(active_indices[None, :], (pop_size, active_indices.shape[0]))

    # Get device-specific vmap function
    extract_sparse_vmap = _get_extract_sparse_fn(device_id, num_active, max_sparse_conns)

    # Explicitly move all vmap inputs to target device
    # Even though upstream computation should be on device, ensure it here
    multi_hop_valid = jax.device_put(multi_hop_valid, device)
    multi_hop_weights = jax.device_put(multi_hop_weights, device)
    active_indices_broadcast = jax.device_put(active_indices_broadcast, device)

    with jax.default_device(device):
        from_indices, to_indices, weights = extract_sparse_vmap(
            multi_hop_valid, multi_hop_weights, active_indices_broadcast
        )

        # Create validity mask and count
        valid_mask = from_indices >= 0
        num_valid = jnp.sum(valid_mask, axis=1)

    # Final logging (DEBUG_PHASE2)
    if verbose or _debug_phase2:
        total_conns = int(jnp.sum(num_valid))
        mean_conns = float(jnp.mean(num_valid))
        print(f"[H->H Discovery] Complete: total_h2h_conns={total_conns}, "
              f"mean_per_genome={mean_conns:.1f}")

    return SparseHiddenConnections(
        from_indices=from_indices.astype(jnp.int32),
        to_indices=to_indices.astype(jnp.int32),
        weights=weights,
        valid_mask=valid_mask,
        num_valid=num_valid,
    )


class SparseCapExceeded(ValueError):
    """A genome's discovered h->h set exceeds ``max_sparse_conns`` under sparse_cap_policy
    'refuse', or under 'grow' beyond ``max_sparse_conns_ceiling``.

    Attributes:
        cap: The ``max_sparse_conns`` in force.
        uncapped: Each genome's count before the cap, in the order of the population the discovery
            ran on (in the substrate API, its batch), so a caller can leave out the genomes over
            the cap.
        over: The indices of the genomes over the cap.
        largest: The largest set: the cap that holds every genome's.
    """

    def __init__(self, message: str, cap: int, uncapped: Any):
        super().__init__(message)
        self.cap = int(cap)
        self.uncapped = np.asarray(uncapped, dtype=np.int64)
        self.over = np.flatnonzero(self.uncapped > self.cap)
        self.largest = int(self.uncapped.max(initial=0))

    def __reduce__(self):
        # Rebuilt from all three arguments, not args alone, so it crosses a process boundary
        return type(self), (self.args[0], self.cap, self.uncapped)


class SparseCapWarning(UserWarning):
    """Discovery truncated a genome's h->h set to ``max_sparse_conns`` (sparse_cap_policy
    'truncate'), or raised the cap (sparse_cap_policy 'grow')."""


def grown_sparse_cap(largest: int, ceiling: Optional[int]) -> Optional[int]:
    """The cap sparse_cap_policy 'grow' moves to: the largest set plus a quarter, rounded up to a
    multiple of 1,024 (headroom, so a run recompiles a few times rather than at every generation),
    at most ``ceiling``.

    Args:
        largest: The largest discovered set.
        ceiling: ``max_sparse_conns_ceiling``, or None.

    Returns:
        The new cap, or None when the largest set exceeds the ceiling.
    """
    largest = int(largest)
    if ceiling is not None and largest > int(ceiling):
        return None
    grown = -(-(largest * 5 // 4) // 1024) * 1024
    grown = max(grown, largest)
    return grown if ceiling is None else min(grown, int(ceiling))


# The largest set each cap has already warned about under 'truncate', so a run warns when a set
# grows rather than at every generation (the generation's metrics count every truncation)
_sparse_cap_truncation_warned: Dict[int, int] = {}
_sparse_cap_lock = threading.Lock()
# EMRHyperNEAT.set_sparse_cap's default for the ceiling, which None would clear
_KEEP_CEILING = object()


def apply_sparse_cap_policy(uncapped: Any, config: Any) -> Optional[int]:
    """Apply ``config.sparse_cap_policy`` to one discovery's per-genome counts before the cap.

    Args:
        uncapped: Per-genome connection counts before ``max_sparse_conns``, or None (nothing
            counted).
        config: An `EMRConfig` or an `emr_hyperneat.substrate.SubstrateConfig`: its
            ``max_sparse_conns``, ``sparse_cap_policy`` and ``max_sparse_conns_ceiling``.

    Returns:
        The cap to discover again at, under 'grow' when a set exceeds the cap in force; None when
        every set fits, or under 'truncate' (after a SparseCapWarning the first time a set this
        large is truncated at this cap).

    Raises:
        SparseCapExceeded: Under 'refuse', or under 'grow' when the largest set exceeds the
            ceiling.
    """
    if uncapped is None:
        return None
    counts = np.asarray(uncapped, dtype=np.int64)
    cap = int(config.max_sparse_conns)
    over = counts > cap
    if not over.any():
        return None
    largest = int(counts.max())
    policy = getattr(config, 'sparse_cap_policy', 'refuse')
    if policy == 'truncate':
        with _sparse_cap_lock:
            warn = largest > _sparse_cap_truncation_warned.get(cap, 0)
            if warn:
                _sparse_cap_truncation_warned[cap] = largest
        if warn:
            import warnings
            warnings.warn(
                f"recurrence.max_sparse_conns={cap} truncated the hidden-to-hidden connections of "
                f"{int(over.sum())} of {counts.shape[0]} genomes (largest set {largest}) to their "
                f"first {cap} pairs in (source, target) position order (sparse_cap_policy "
                f"'truncate'); max_sparse_conns {largest} keeps every set whole",
                SparseCapWarning, stacklevel=3)
        return None
    if policy == 'grow':
        ceiling = getattr(config, 'max_sparse_conns_ceiling', None)
        grown = grown_sparse_cap(largest, ceiling)
        if grown is not None:
            import warnings
            warnings.warn(
                f"recurrence.max_sparse_conns grew from {cap} to {grown}: {int(over.sum())} of "
                f"{counts.shape[0]} genomes discovered more hidden-to-hidden connections (largest "
                f"set {largest}; sparse_cap_policy 'grow'); max_sparse_conns {largest} starts a "
                "run there", SparseCapWarning, stacklevel=3)
            return grown
        raise SparseCapExceeded(
            f"{sparse_cap_refusal(counts, cap)} sparse_cap_policy 'grow' stops at "
            f"max_sparse_conns_ceiling={int(ceiling)}.", cap, counts)
    raise SparseCapExceeded(sparse_cap_refusal(counts, cap), cap, counts)


# The h->h cap's record of the run in progress: run_until_threshold starts it for its recurrent
# configuration, every discovery of that configuration adds to it, and _record_route puts it in
# the run's result
_sparse_cap_run: Dict[str, Any] = {}


def _sparse_cap_run_start(config: Any) -> None:
    """Start the run's cap record for ``config`` (an EMRConfig with hidden-to-hidden connections),
    or clear it."""
    with _sparse_cap_lock:
        _sparse_cap_run.clear()
        if config is not None and getattr(config, 'allow_hidden_to_hidden', False):
            _sparse_cap_run.update({'config': config, 'configured': int(config.max_sparse_conns),
                                    'largest': None, 'truncated': 0, 'grown': []})


def _sparse_cap_run_note(config: Any, uncapped: Any, grown_from: Optional[int] = None) -> None:
    """Add one discovery of the run's configuration to its record: the largest set, a truncation,
    a growth."""
    if uncapped is None:
        return
    with _sparse_cap_lock:
        if _sparse_cap_run.get('config') is not config:
            return
        largest = int(np.asarray(uncapped, dtype=np.int64).max(initial=0))
        _sparse_cap_run['largest'] = max(largest, _sparse_cap_run['largest'] or 0)
        if grown_from is not None:
            _sparse_cap_run['grown'].append([int(grown_from), int(config.max_sparse_conns)])
        elif largest > int(config.max_sparse_conns):
            _sparse_cap_run['truncated'] += 1


def sparse_cap_run_record() -> Optional[Dict[str, Any]]:
    """The cap record of the last run started (`run_until_threshold` puts it in the result as
    ``'sparse_cap'``).

    Returns:
        None without hidden-to-hidden connections; else ``max_sparse_conns`` (the cap the run ended
        with, the one its champion was evaluated under), ``configured_max_sparse_conns``,
        ``sparse_cap_policy``, ``max_sparse_conns_ceiling``, ``largest_set`` (the largest
        discovered, None before any discovery), ``truncated_discoveries`` (discoveries that
        truncated a genome) and ``grown`` (each growth, [from, to]).
    """
    with _sparse_cap_lock:
        if 'config' not in _sparse_cap_run:
            return None
        config = _sparse_cap_run['config']
        return {
            'max_sparse_conns': int(config.max_sparse_conns),
            'configured_max_sparse_conns': _sparse_cap_run['configured'],
            'sparse_cap_policy': config.sparse_cap_policy,
            'max_sparse_conns_ceiling': config.max_sparse_conns_ceiling,
            'largest_set': _sparse_cap_run['largest'],
            'truncated_discoveries': _sparse_cap_run['truncated'],
            'grown': [list(g) for g in _sparse_cap_run['grown']],
        }


# Diagnostics of the last h->h discovery call on this process: per-genome direct and
# total connection counts, direct pairs whose weight the multi-hop merge replaced, active positions,
# the union size and the settings in force, and each genome's count before the max_sparse_conns cap
# with how many connections the cap dropped. Filled by discover_sparse_hh_vectorized_multi_hop for
# the union and per-genome paths (not the pmap pipeline); read by experiment runners for their
# records and by the standard path's metrics (truncation only).
_last_discovery_stats: Dict[str, Any] = {}


def sparse_cap_refusal(uncapped: Any, cap: int) -> Optional[str]:
    """The refusal for genomes whose discovered h->h set exceeds ``max_sparse_conns``, or None.

    The cap sizes the padded connection arrays. Past it the extraction would keep a genome's first
    pairs in (source, target) position order, not its strongest, so the network evaluated would not
    be the one its CPPN describes. Every discovery, and the substrate API, refuses instead under
    sparse_cap_policy 'refuse', the default (`apply_sparse_cap_policy`).

    Args:
        uncapped: Per-genome connection counts before the cap.
        cap: ``recurrence.max_sparse_conns``.

    Returns:
        The message to raise, naming the genomes over the cap and the value that holds them all,
        or None when every set fits.
    """
    counts = np.asarray(uncapped, dtype=np.int64)
    over = counts > int(cap)
    if not over.any():
        return None
    largest = int(counts.max())
    return (f"recurrence.max_sparse_conns={int(cap)} cannot hold the hidden-to-hidden "
            f"connections discovered for {int(over.sum())} of {counts.shape[0]} genomes (largest "
            f"set {largest}): the cap would keep each genome's first pairs in (source, target) "
            "position order, not its strongest. Set recurrence.max_sparse_conns to at least "
            f"{largest}.")


def _record_discovery_stats(result: "SparseHiddenConnections", masks_A: jnp.ndarray, union_active: int,
                            config: "EMRConfig", chunk_stats: List[Dict[str, Any]]) -> None:
    """Store the last discovery's per-genome counts in ``_last_discovery_stats`` (numpy). The
    caller then applies the cap's policy (`apply_sparse_cap_policy`), after the record, so a caller
    that catches the refusal can read the counts."""
    direct = replaced = None
    if chunk_stats and all('direct_per_genome' in s for s in chunk_stats):
        direct = np.concatenate([np.asarray(s['direct_per_genome']) for s in chunk_stats]).astype(np.int64)
    if chunk_stats and all('replaced_per_genome' in s for s in chunk_stats):
        replaced = np.concatenate([np.asarray(s['replaced_per_genome']) for s in chunk_stats]).astype(np.int64)
    uncapped = truncated = None
    if chunk_stats and all('uncapped_per_genome' in s for s in chunk_stats):
        uncapped = np.concatenate(
            [np.asarray(s['uncapped_per_genome']) for s in chunk_stats]).astype(np.int64)
        truncated = np.maximum(uncapped - int(config.max_sparse_conns), 0)
    _last_discovery_stats.clear()
    _last_discovery_stats.update({
        'total_per_genome': np.asarray(result.num_valid).astype(np.int64),
        'direct_per_genome': direct,
        'replaced_per_genome': replaced,
        'uncapped_per_genome': uncapped,
        'truncated_per_genome': truncated,
        'max_sparse_conns': int(config.max_sparse_conns),
        'sparse_cap_policy': getattr(config, 'sparse_cap_policy', 'refuse'),
        'sparse_cap_grown_from': None,
        'active_per_genome': np.asarray(jnp.sum(masks_A, axis=1)).astype(np.int64),
        'active_masks': np.asarray(masks_A),
        'union_active': int(union_active),
        'iteration_level': int(config.iteration_level),
        'hh_discovery_mode': getattr(config, 'hh_discovery_mode', 'union'),
        'multi_hop_second_factor': getattr(config, 'multi_hop_second_factor', 'raw'),
    })


_per_genome_discovery_cache: Dict[Tuple, Any] = {}
_per_genome_discovery_cache_lock = threading.Lock()
# Ceiling on the m x S x S pair block of one per-genome core call. The bound that applies is the row
# budget below (`_per_genome_rows_for`, at most 2^16 pairs per call); this ceiling only caps it. Counted
# from the input rows alone, 2^24 pairs would be about 470 MB; XLA's own accounting puts one such call at
# 6.7 to 10.9 GB (450 to 652 bytes per pair through the CPPN's 50-node intermediates), which is why it is
# not the operative bound. Slicing never changes a value: every genome's block depends on that genome
# alone.
_PER_GENOME_PAIR_BUDGET = 1 << 24


def _bucket_size_for(count: int) -> int:
    """The smallest static bucket size (8, 16, 32, ...) that holds ``count`` positions."""
    size = 8
    while size < count:
        size *= 2
    return size


_PER_GENOME_ROW_BUDGET = 1 << 16


def _per_genome_rows_for(size: int) -> int:
    """Static number of genomes per compiled call for a bucket size.

    `max(1, min(slice, 2^16 // size^2))`, with `slice` the pair budget's ceiling for the size, so
    every call of a bucket size has the same shape (rows, size, ...). A slice of whatever is left of
    the bucket changes its member count almost every generation and recompiles every gather, the
    core, the extract and every scatter: 150 to 275 executables per generation, 94-99% of the
    per-genome discovery time on a CPU (XOR, depth 4, populations 100 and 1000). With fixed rows a
    bucket size compiles once per run. The pad rows repeat a real
    member and are scattered to an out-of-range index, so results are unchanged.
    """
    slice_len = max(1, _PER_GENOME_PAIR_BUDGET // (size * size))
    return max(1, min(slice_len, _PER_GENOME_ROW_BUDGET // (size * size)))


# Pairs per block of the per-genome core's CPPN query. Above it, the query of a slice runs over blocks of
# source positions in sequence.
_PER_GENOME_BLOCK_PAIRS = 1 << 18


def _per_genome_block_for(size: int, rows: int) -> int:
    """Source positions per block of the per-genome core's CPPN query, for ``rows`` genomes of a bucket size.

    The core queries every (source, target) pair of its slice. From a bucket size of 1024 on a slice is one
    genome and its ``size^2`` pairs, 4.2 million at 2048: 1.89 GB of XLA-CPU temporaries in one call, where
    the union path queries one source position at a time (0.20 GB for 1000 genomes against 300 targets).
    Above ``_PER_GENOME_BLOCK_PAIRS`` pairs per call the query runs over blocks of source positions in
    sequence (``lax.map``), at most that many pairs each; at or below it, one block. Every pair's arithmetic
    is unchanged, so the values are too. Bucket sizes are powers of two, so the block divides the size.
    """
    if rows * size * size <= _PER_GENOME_BLOCK_PAIRS:
        return size
    block = size
    while block > 1 and rows * block * size > _PER_GENOME_BLOCK_PAIRS:
        block //= 2
    return block


def hh_pair_weights(outputs: jnp.ndarray, max_weight: float,
                    band_threshold: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """A hidden-to-hidden pair's weight and whether it passes the band: per-genome discovery's criterion.

    ``weight = tanh(cppn) * max_weight`` on the CPPN's weight output (channel 0, taken after the
    CPPN's own output transform, so the weights are bounded by ``tanh(1) * max_weight`` when that
    transform is a tanh, as it is by default); a pair is a candidate connection when
    ``|weight| > band_threshold``. LEO gating replaces the band in the generation loop (the
    substrate API refuses LEO). The loop's per-genome core and the public substrate API call this
    function, so an explorer shows the criterion the loop applied.

    Args:
        outputs: CPPN outputs at the pairs, ``(..., num_cppn_outputs)``.
        max_weight: Weight scale.
        band_threshold: Band threshold.

    Returns:
        ``(weights, passes)``, both of shape ``outputs.shape[:-1]``.
    """
    weights = jnp.tanh(outputs[..., 0]) * max_weight
    return weights, jnp.abs(weights) > band_threshold


def per_genome_pair_connections(
    out: jnp.ndarray,
    coords: jnp.ndarray,
    valid: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    allow_backward: bool,
    allow_lateral: bool,
    allow_self_loops: bool,
    iteration_level: int,
    hop_decay_factor: float,
    use_fori: bool,
    use_leo: bool = False,
    leo_threshold: float = 0.2,
    leo_mode: str = 'compound',
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A bucket's hidden-to-hidden connections from the CPPN outputs at its genomes' position pairs.

    The arithmetic half of the per-genome discovery core (the other half is its CPPN query,
    :func:`_per_genome_pair_outputs`): the weight and band (:func:`hh_pair_weights`) or the LEO
    gate, the connection classes of the configuration (by y: backward ``y_source > y_target``,
    lateral equal, self-loops the same position), the pairs' validity, and the multi-hop expansion
    of the genome's own direct connections (masked second factor) whose composites keep the
    classes. The public substrate API runs this same function on the same shapes, so its
    connections are the loop's.

    Args:
        out: (m, size, size, outputs) CPPN outputs at (source, target) pairs.
        coords: (m, size, 2) the genomes' active positions; padding carries a real position's coordinates.
        valid: (m, size) False on padding.
        band_threshold, max_weight: As for :func:`hh_pair_weights`.
        allow_backward, allow_lateral, allow_self_loops: The configuration's connection classes.
        iteration_level: Hops composed (1: direct connections only).
        hop_decay_factor: Decay per hop.
        use_fori: The fori_loop expansion rather than the matrix power.
        use_leo, leo_threshold, leo_mode: LEO gating of the direct connections.

    Returns:
        ``(weights, valid, direct_valid, direct_weights)``, each (m, size, size): the connections after the
        expansion, and the direct ones (``direct_weights`` is zero off ``direct_valid``).
    """
    m, size = coords.shape[0], coords.shape[1]
    weights, weight_valid = hh_pair_weights(out, max_weight, band_threshold)
    if use_leo and out.shape[-1] > LEO_CHANNEL:
        leo = jnp.tanh(out[..., LEO_CHANNEL])
        if leo_mode == 'replacement':
            weight_valid = compute_leo_validity_mask_replacement(leo, leo_threshold=leo_threshold)
        else:
            weight_valid = compute_leo_validity_mask(
                weights, leo, weight_threshold=band_threshold, leo_threshold=leo_threshold
            )

    y = coords[..., 1]
    y_valid = jnp.ones((m, size, size), dtype=bool)
    if not allow_backward:
        y_valid = y_valid & (y[:, :, None] <= y[:, None, :])
    if not allow_lateral:
        y_valid = y_valid & (y[:, :, None] != y[:, None, :])
    coord_match = jnp.all(coords[:, :, None, :] == coords[:, None, :, :], axis=-1)
    if allow_self_loops:
        y_valid = y_valid | coord_match
        self_mask = jnp.ones((m, size, size), dtype=bool)
    else:
        self_mask = ~coord_match
    pair_valid = valid[:, :, None] & valid[:, None, :]
    direct_valid = weight_valid & y_valid & self_mask & pair_valid
    direct_weights = jnp.where(direct_valid, weights, 0.0)

    if iteration_level > 1:
        expand = (compute_multi_hop_connections_fori_loop if use_fori
                  else compute_multi_hop_connections_matrix_power)

        def expand_one(w, v):
            return expand(w, v, band_threshold, iteration_level, hop_decay_factor,
                          max_weight, 'masked')

        # Always the masked second factor: the union path does the same under
        # multi_hop_second_factor 'masked' (the default), and 'raw' is refused with
        # per_genome in EMRConfig.__post_init__, which is why the compile-cache key of
        # the discovery core does not carry the value.
        mh_weights, mh_valid = jax.vmap(expand_one)(direct_weights, direct_valid)
        # Composites keep the preset's connection class, as in the union path.
        mh_valid = mh_valid & y_valid & self_mask
        mh_weights = jnp.where(mh_valid, mh_weights, 0.0)
    else:
        mh_weights = direct_weights
        mh_valid = direct_valid
    return mh_weights, mh_valid, direct_valid, direct_weights


def _per_genome_pair_outputs(
    state: Any,
    nodes: jnp.ndarray,
    conns: jnp.ndarray,
    conn_attrs: jnp.ndarray,
    node_attrs: jnp.ndarray,
    coords: jnp.ndarray,
    cppn_forward: Any,
    geometry_seeding_enabled: bool,
) -> jnp.ndarray:
    """The per-genome core's CPPN query: every genome's CPPN at every (source, target) pair of its positions.

    Rows ``(source, target, optional deltas, 1)``, the layout of ``cppn_query_rows``. From a bucket size of 1024 on
    the query runs over blocks of source positions in sequence (:func:`_per_genome_block_for`); every pair's
    arithmetic is unchanged.

    Args:
        state: TensorNEAT state.
        nodes, conns, conn_attrs, node_attrs: The m genomes' transformed CPPNs.
        coords: (m, size, 2) the genomes' positions.
        cppn_forward: ``cppn_forward(state, cppn_tuple, row)``.
        geometry_seeding_enabled: 7-D rows with the signed deltas.

    Returns:
        (m, size, size, outputs).
    """
    m, size = coords.shape[0], coords.shape[1]

    def one_genome(cppn_tuple, rows_g):
        return jax.vmap(lambda x: cppn_forward(state, cppn_tuple, x))(rows_g)

    def query(sources):
        # CPPN outputs of the pairs (source in `sources`, any target): (m, b, size, outputs)
        b = sources.shape[1]
        src = jnp.broadcast_to(sources[:, :, None, :], (m, b, size, 2))
        tgt = jnp.broadcast_to(coords[:, None, :, :], (m, b, size, 2))
        parts = [src, tgt]
        if geometry_seeding_enabled:
            parts.append(tgt[..., 0:1] - src[..., 0:1])
            parts.append(tgt[..., 1:2] - src[..., 1:2])
        parts.append(jnp.ones((m, b, size, 1), dtype=coords.dtype))
        rows = jnp.concatenate(parts, axis=-1).reshape(m, b * size, -1)
        out = jax.vmap(one_genome, in_axes=((0, 0, 0, 0), 0))(
            (nodes, conns, conn_attrs, node_attrs), rows
        )
        return out.reshape(m, b, size, -1)

    block = _per_genome_block_for(size, m)
    if block >= size:
        return query(coords)
    # Blocks of source positions in sequence, so one block's intermediates are live at a
    # time; the blocks are stacked back in source order.
    blocks = coords.reshape(m, size // block, block, 2).transpose(1, 0, 2, 3)
    out = lax.map(query, blocks)
    return out.transpose(1, 0, 2, 3, 4).reshape(m, size, size, -1)


def _get_per_genome_discovery_fn(
    device_id: int,
    cppn_forward: Any,
    size: int,
    band_threshold: float,
    max_weight: float,
    config: EMRConfig,
    geometry_seeding_enabled: bool,
    use_leo: bool,
    num_cppn_outputs: int,
    leo_threshold: float,
    leo_mode: str,
):
    """Get or create the compiled per-bucket discovery core for one bucket size.

    The core computes, for a bucket of genomes that each own at most ``size`` active positions,
    the multi-hop h->h weights and validity over the genome's own position pairs. Its arithmetic
    is the union pipeline's, applied to the genome's own coordinates: the same
    ``cppn_forward(state, cppn_tuple, row)`` per pair with the row layout of ``cppn_query_rows``
    (source, target, optional deltas, bias one), tanh and ``max_weight``, the connection-class
    constraints of ``_get_constraint_mask_fn``, the band threshold or the LEO gate, and the
    multi-hop expansion over the genome's existing direct connections. Pairs whose endpoints are
    padding are never valid.
    """
    key = (device_id, id(cppn_forward), size, float(band_threshold), float(max_weight),
           bool(config.allow_backward), bool(config.allow_lateral), bool(config.allow_self_loops),
           int(config.iteration_level), config.multi_hop_algorithm, float(config.hop_decay_factor),
           bool(geometry_seeding_enabled), bool(use_leo), int(num_cppn_outputs),
           float(leo_threshold), leo_mode)
    with _per_genome_discovery_cache_lock:
        if key in _per_genome_discovery_cache:
            return _per_genome_discovery_cache[key]
        devices = jax.devices()
        device = devices[device_id] if device_id < len(devices) else devices[0]
        allow_backward = bool(config.allow_backward)
        allow_lateral = bool(config.allow_lateral)
        allow_self_loops = bool(config.allow_self_loops)
        iteration_level = int(config.iteration_level)
        hop_decay_factor = float(config.hop_decay_factor)
        use_fori = config.multi_hop_algorithm != "matrix_power"

        with jax.default_device(device):
            def core(state, nodes, conns, conn_attrs, node_attrs, coords, valid):
                # coords: (m, size, 2) float32, padded entries carry a real position's coordinates
                # valid:  (m, size) bool, False on padding
                out = _per_genome_pair_outputs(state, nodes, conns, conn_attrs, node_attrs, coords,
                                               cppn_forward, geometry_seeding_enabled)
                mh_weights, mh_valid, direct_valid, direct_weights = per_genome_pair_connections(
                    out, coords, valid, band_threshold, max_weight, allow_backward, allow_lateral,
                    allow_self_loops, iteration_level, hop_decay_factor, use_fori,
                    use_leo=use_leo, leo_threshold=leo_threshold, leo_mode=leo_mode,
                )
                replaced = jnp.sum(direct_valid & (mh_weights != direct_weights), axis=(1, 2))
                return mh_weights, mh_valid, jnp.sum(direct_valid, axis=(1, 2)), replaced

            compiled = jax.jit(core, device=device)
        _per_genome_discovery_cache[key] = compiled
        return compiled


def _discover_sparse_hh_per_genome(
    state: Any,
    cppns_transformed: Tuple,
    h_grid: HierarchicalGridStructure,
    masks_A: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    config: EMRConfig,
    cppn_forward: Any,
    verbose: bool = False,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    use_leo: bool = False,
    num_cppn_outputs: int = 1,
    leo_threshold: float = 0.2,
    leo_mode: str = 'compound',
    stats_out: Optional[Dict[str, Any]] = None,
) -> SparseHiddenConnections:
    """Sparse h->h discovery over each genome's own active positions (``hh_discovery_mode: per_genome``).

    The union pipeline queries every genome over the population's union of active positions
    (``pop x U x U`` pairs) and only then restricts each genome to its own positions. Here genomes
    are bucketed by their active count into static sizes 8, 16, 32, ...; each bucket is queried
    over its genomes' own positions (``m x S x S``), so the pair count follows the genomes' own
    active sets. The connections and weights are the union pipeline's for every genome, listed in
    the same order (ascending source position, then ascending target position), including the
    ``max_sparse_conns`` truncation rule: bit for bit at ``iteration_level`` 1, within a few float32
    ulps at level >= 2 (the products run over different array shapes) under the default
    ``multi_hop_second_factor: masked``, where both paths compose the genome's existing direct
    connections. ``raw`` (0.1.0's algebra, union path only) multiplied by the raw weights
    over the whole population union and made a genome's composites depend on the other genomes; it
    is refused together with this mode. Not used under ``use_dense_discovery``. Each bucket size is
    processed in slices of a fixed number of genomes (``_per_genome_rows_for``), padded with a
    repeated member whose rows are dropped at the scatter, so a size compiles once per run. From a
    bucket size of 1024 on, a slice's CPPN query runs over blocks of source positions in sequence
    (``_per_genome_block_for``, at most 2^18 pairs per block), so a call's temporaries stay near 120 MB
    whatever the bucket size (XLA-CPU accounting; 1.89 GB at size 2048 in one block).
    """
    pop_size = int(masks_A.shape[0])
    max_sparse_conns = config.max_sparse_conns
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]
    all_positions = jax.device_put(h_grid.all_positions, device)
    masks_A = jax.device_put(masks_A, device)
    cppns_transformed = tuple(jax.device_put(arr, device) for arr in cppns_transformed)
    state = jax.tree.map(lambda x: jax.device_put(x, device), state)

    counts = np.asarray(jnp.sum(masks_A, axis=1))
    from_all = jnp.full((pop_size, max_sparse_conns), -1, dtype=jnp.int32)
    to_all = jnp.full((pop_size, max_sparse_conns), -1, dtype=jnp.int32)
    w_all = jnp.zeros((pop_size, max_sparse_conns), dtype=jnp.float32)
    direct_all = jnp.zeros((pop_size,), dtype=jnp.int32)
    replaced_all = jnp.zeros((pop_size,), dtype=jnp.int32)
    uncapped_all = jnp.zeros((pop_size,), dtype=jnp.int32)
    if pop_size == 0 or int(counts.max()) == 0:
        if stats_out is not None:
            stats_out['direct_per_genome'] = direct_all
            stats_out['replaced_per_genome'] = replaced_all
            stats_out['uncapped_per_genome'] = uncapped_all
        valid_mask = from_all >= 0
        return SparseHiddenConnections(from_all, to_all, w_all, valid_mask, jnp.sum(valid_mask, axis=1))

    sizes = np.array([_bucket_size_for(int(c)) for c in counts])
    if verbose or os.environ.get('DEBUG_PHASE2', '0') == '1':
        hist = {int(s): int(np.sum(sizes == s)) for s in np.unique(sizes)}
        print(f"[H->H Discovery] per-genome mode: active per genome mean={counts.mean():.1f} "
              f"max={int(counts.max())}, buckets {hist}")

    for size in sorted(set(int(s) for s in sizes)):
        bucket_members = np.nonzero(sizes == size)[0]
        core = _get_per_genome_discovery_fn(
            device_id, cppn_forward, size, band_threshold, max_weight, config,
            geometry_seeding_enabled, use_leo, num_cppn_outputs, leo_threshold, leo_mode,
        )
        extract_sparse_vmap = _get_extract_sparse_fn(device_id, size, max_sparse_conns)
        rows = _per_genome_rows_for(size)
        for start in range(0, bucket_members.shape[0], rows):
            real = bucket_members[start:start + rows]
            # Every slice of a bucket size has the static shape (rows, size, ...): the gather index
            # repeats the slice's first member into the pad rows and the scatter index sends them to
            # pop_size, out of range, which `mode='drop'` discards. m is a batch dimension only in
            # the core (every reduction is per genome), so the pad rows change no result.
            pad = rows - real.shape[0]
            gather = jnp.asarray(np.concatenate([real, np.full(pad, real[0])]), dtype=jnp.int32)
            scatter = jnp.asarray(np.concatenate([real, np.full(pad, pop_size)]), dtype=jnp.int32)
            with jax.default_device(device):
                idx = jax.vmap(lambda row: jnp.nonzero(row, size=size, fill_value=-1)[0])(masks_A[gather])
                valid = idx >= 0
                idx_safe = jnp.where(valid, idx, 0)
                coords = all_positions[idx_safe]  # (rows, size, 2)
                slice_cppns = tuple(arr[gather] for arr in cppns_transformed)
                mh_weights, mh_valid, direct_b, replaced_b = core(
                    state, slice_cppns[0], slice_cppns[1], slice_cppns[2], slice_cppns[3], coords, valid
                )
                from_b, to_b, w_b = extract_sparse_vmap(mh_valid, mh_weights, idx_safe)
                from_all = from_all.at[scatter].set(from_b.astype(jnp.int32), mode='drop')
                to_all = to_all.at[scatter].set(to_b.astype(jnp.int32), mode='drop')
                w_all = w_all.at[scatter].set(w_b, mode='drop')
                direct_all = direct_all.at[scatter].set(direct_b.astype(jnp.int32), mode='drop')
                replaced_all = replaced_all.at[scatter].set(replaced_b.astype(jnp.int32), mode='drop')
                uncapped_all = uncapped_all.at[scatter].set(
                    jnp.sum(mh_valid, axis=(1, 2)).astype(jnp.int32), mode='drop')

    valid_mask = from_all >= 0
    num_valid = jnp.sum(valid_mask, axis=1)
    if stats_out is not None:
        stats_out['direct_per_genome'] = direct_all
        stats_out['replaced_per_genome'] = replaced_all
        stats_out['uncapped_per_genome'] = uncapped_all
    if verbose or os.environ.get('DEBUG_PHASE2', '0') == '1':
        print(f"[H->H Discovery] Complete (per-genome): total_h2h_conns={int(jnp.sum(num_valid))}, "
              f"mean_per_genome={float(jnp.mean(num_valid)):.1f}")
    return SparseHiddenConnections(
        from_indices=from_all, to_indices=to_all, weights=w_all, valid_mask=valid_mask, num_valid=num_valid,
    )


def discover_sparse_hh_vectorized_multi_hop(
    state: Any,
    cppns_transformed: Tuple,
    h_grid: HierarchicalGridStructure,
    masks_A: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    config: EMRConfig,
    cppn_forward: Any,
    pop_chunk_size: int = 100,
    verbose: bool = False,
    global_union_active: Optional[jnp.ndarray] = None,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    num_cppn_outputs: int = 1,
    discovery_population_chunk_size: Optional[int] = None,
    use_leo: bool = False,
    leo_threshold: float = 0.2,
    leo_mode: str = 'compound',
) -> SparseHiddenConnections:
    """Discover the population's h->h connections and apply the cap's policy.

    `_discover_sparse_hh_population` discovers at ``config.max_sparse_conns``; a genome whose set
    exceeds it raises under sparse_cap_policy 'refuse', keeps its first pairs under 'truncate', and
    under 'grow' the cap rises (`grown_sparse_cap`) in ``config`` itself, the run's configuration,
    so later generations start at the new width, and the population is discovered again at it. The
    arguments are `_discover_sparse_hh_population`'s.

    Raises:
        SparseCapExceeded: A genome over the cap under 'refuse', or over the ceiling under 'grow'.
    """
    def discover() -> SparseHiddenConnections:
        return _discover_sparse_hh_population(
            state, cppns_transformed, h_grid, masks_A, band_threshold, max_weight, config,
            cppn_forward, pop_chunk_size=pop_chunk_size, verbose=verbose,
            global_union_active=global_union_active, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled, use_leo=use_leo,
            num_cppn_outputs=num_cppn_outputs,
            discovery_population_chunk_size=discovery_population_chunk_size,
            leo_threshold=leo_threshold, leo_mode=leo_mode)

    result = discover()
    uncapped = _last_discovery_stats.get('uncapped_per_genome')
    grown = apply_sparse_cap_policy(uncapped, config)
    if grown is None:
        _sparse_cap_run_note(config, uncapped)
        return result
    with _sparse_cap_lock:
        grown_from = int(config.max_sparse_conns)
        config.max_sparse_conns = max(grown_from, grown)
    result = discover()
    uncapped = _last_discovery_stats.get('uncapped_per_genome')
    # The same population at a cap that holds its largest set: nothing exceeds it
    apply_sparse_cap_policy(uncapped, config)
    _last_discovery_stats['sparse_cap_grown_from'] = grown_from
    _sparse_cap_run_note(config, uncapped, grown_from=grown_from)
    return result


def _discover_sparse_hh_population(
    state: Any,
    cppns_transformed: Tuple,
    h_grid: HierarchicalGridStructure,
    masks_A: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    config: EMRConfig,
    cppn_forward: Any,
    pop_chunk_size: int = 100,
    verbose: bool = False,
    global_union_active: Optional[jnp.ndarray] = None,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    use_leo: bool = False,
    num_cppn_outputs: int = 1,
    discovery_population_chunk_size: Optional[int] = None,
    leo_threshold: float = 0.2,
    leo_mode: str = 'compound',
) -> SparseHiddenConnections:
    """Population-chunked front for h->h discovery.

    The single-chunk pipeline materializes population-dense `(pop, A, A)`
    tensors over the active-set union: recorded wall allocations of
    1000 x 692^2 x 4 and 1000 x 888^2 x 4 bytes. Per-genome work is
    independent once the union is fixed, so this wrapper (1) hoists the union
    computation so it stays population-global, then (2) slices the population
    through the unchanged pipeline and concatenates the sparse results.
    Chunking never changes values: with the union pinned, a slice of the full
    call and a call on the slice are the same computation.

    `discovery_population_chunk_size=None` sizes chunks from the measured
    working set (`execution.chunking.discovery_population_chunk`) and returns
    the whole population as one chunk whenever it fits, so a configuration that
    fits runs exactly the unchunked pipeline.
    """
    from emr_hyperneat.execution.chunking import (
        discovery_population_chunk,
    )

    pop_size = masks_A.shape[0]

    # Hoist the population-global union: computed here once, identical for
    # every chunk. (Inside the pipeline a None union is computed from the
    # masks it receives, which for a chunk would be the chunk's union:
    # smaller, and value-changing.)
    if global_union_active is None:
        global_union_active = jnp.any(masks_A, axis=0)
    num_active = int(jnp.sum(global_union_active))

    if discovery_population_chunk_size is None:
        chunk = discovery_population_chunk(pop_size, num_active)
    else:
        chunk = max(1, min(int(discovery_population_chunk_size), pop_size))
    if getattr(config, 'hh_discovery_mode', 'union') == 'per_genome' and not config.use_dense_discovery:
        # The per-genome route bounds its memory by the rows per bucket size, not by the union, so
        # the union-sized population split only multiplied its padded calls and changed the shapes
        # its eager steps see (bursts of recompiles in both modes on two RTX 2080 Ti: 36.6 s over
        # generations 2-29 at depth 5, population 1000). Whole population, one call; per-genome
        # results do not depend on how the population is sliced.
        chunk = pop_size

    if chunk >= pop_size:
        stats: Dict[str, Any] = {}
        result = _discover_sparse_hh_multi_hop_single_chunk(
            state, cppns_transformed, h_grid, masks_A,
            band_threshold, max_weight, config, cppn_forward,
            pop_chunk_size=pop_chunk_size, verbose=verbose,
            global_union_active=global_union_active, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled,
            use_leo=use_leo, num_cppn_outputs=num_cppn_outputs,
            leo_threshold=leo_threshold, leo_mode=leo_mode,
            stats_out=stats,
        )
        _record_discovery_stats(result, masks_A, num_active, config, [stats])
        return result

    if verbose:
        print(f"[H->H Discovery] population-chunked: pop={pop_size}, "
              f"chunk={chunk}, A={num_active}")

    parts = []
    chunk_stats: List[Dict[str, Any]] = []
    for start in range(0, pop_size, chunk):
        end = min(start + chunk, pop_size)
        chunk_cppns = tuple(arr[start:end] for arr in cppns_transformed)
        chunk_stats.append({})
        parts.append(_discover_sparse_hh_multi_hop_single_chunk(
            state, chunk_cppns, h_grid, masks_A[start:end],
            band_threshold, max_weight, config, cppn_forward,
            pop_chunk_size=pop_chunk_size, verbose=verbose and start == 0,
            global_union_active=global_union_active, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled,
            use_leo=use_leo, num_cppn_outputs=num_cppn_outputs,
            leo_threshold=leo_threshold, leo_mode=leo_mode,
            stats_out=chunk_stats[-1],
        ))

    result = SparseHiddenConnections(
        from_indices=jnp.concatenate([p.from_indices for p in parts], axis=0),
        to_indices=jnp.concatenate([p.to_indices for p in parts], axis=0),
        weights=jnp.concatenate([p.weights for p in parts], axis=0),
        valid_mask=jnp.concatenate([p.valid_mask for p in parts], axis=0),
        num_valid=jnp.concatenate([p.num_valid for p in parts], axis=0),
    )
    _record_discovery_stats(result, masks_A, num_active, config, chunk_stats)
    return result


# Cache Manager

class HHCacheManager:
    """Manages caching of sparse h->h connections across generations.

    Caching saves expensive Phase 2 discovery when the variance mask
    (which determines active positions) hasn't changed significantly.
    """

    def __init__(self, config: EMRConfig):
        """Initialize cache manager.

        Args:
            config: Extended configuration with caching settings
        """
        trigger = getattr(config, 'hh_cache_trigger', 'time_or_mask')
        if trigger not in HH_CACHE_TRIGGERS:
            raise ValueError(
                f"recurrence.hh_cache_trigger must be one of {HH_CACHE_TRIGGERS}, got {trigger!r}"
            )
        self.config = config
        self._cached_connections: Optional[SparseHiddenConnections] = None
        self._cached_variance_mask: Optional[jnp.ndarray] = None
        self._last_refresh_gen: int = -1
        self._refresh_count: int = 0

    def should_refresh(
        self,
        current_gen: int,
        current_mask: jnp.ndarray,
    ) -> bool:
        """Determine if h->h connections need refresh.

        Refresh if:
        1. Caching is disabled
        2. No cached connections exist
        3. Time-based: >= refresh_interval generations since last refresh
        4. Change-based: Variance mask changed by > threshold

        Args:
            current_gen: Current generation number
            current_mask: Current variance mask (pop, total_positions)

        Returns:
            True if refresh needed
        """
        if not self.config.hh_cache_enabled:
            return True

        if self._cached_connections is None:
            return True

        # Time-based check
        gens_since_refresh = current_gen - self._last_refresh_gen
        if gens_since_refresh >= self.config.hh_refresh_interval:
            return True

        # Change-based check ('time' skips it, and the host read of the change)
        if (getattr(self.config, 'hh_cache_trigger', 'time_or_mask') == 'time_or_mask'
                and self._cached_variance_mask is not None):
            # Compute mask change ratio (fraction of positions that changed)
            mask_diff = jnp.abs(
                current_mask.astype(jnp.float32) -
                self._cached_variance_mask.astype(jnp.float32)
            )
            change_ratio = float(jnp.mean(mask_diff))

            if change_ratio > self.config.hh_mask_change_threshold:
                return True

        return False

    def update_cache(
        self,
        connections: SparseHiddenConnections,
        variance_mask: jnp.ndarray,
        generation: int,
    ) -> None:
        """Update cached connections.

        Args:
            connections: New sparse connections
            variance_mask: Current variance mask
            generation: Current generation
        """
        self._cached_connections = connections
        self._cached_variance_mask = variance_mask
        self._last_refresh_gen = generation
        self._refresh_count += 1

    def get_cached(self) -> Optional[SparseHiddenConnections]:
        """Get cached connections if available."""
        return self._cached_connections

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'refresh_count': self._refresh_count,
            'last_refresh_gen': self._last_refresh_gen,
            'cache_enabled': self.config.hh_cache_enabled,
        }
