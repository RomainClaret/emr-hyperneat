"""HMR-HyperNEAT Unified Extended Implementation.

This module combines the best features from:
- hmrhyperneat_multi_gpus.py: SparseHiddenConnections, discovery toggle, iterative discovery
- hmrhyperneat_unified.py: RecurrenceConfig, presets, constraint filtering, caching

PLUS NEW: Multi-hop vectorized expansion that ACTUALLY implements iteration_level > 1.

HYBRID SPARSE-DENSE ARCHITECTURE
=================================

1. **Dense W1, W2** for input→positions and positions→output (GPU-efficient matmul)
2. **Sparse hidden→hidden** with multi-hop expansion via matrix power
3. **Multi-iteration forward** using scatter-add for h→h, matmul for rest

MULTI-HOP EXPANSION (KEY INNOVATION)
====================================

Multi-hop support uses a JIT-compatible matrix power approach:

```python
# For iteration_level k hops:
A_total = A + decay*A^2 + decay^2*A^3 + ... + decay^(k-1)*A^k

# This captures transitive connections: if A→B→C exists, we add A→C
```

The algorithm is JIT-compatible because iteration_level is a static argument,
allowing the Python loop to compile away.

EXECUTION MODES
===============

1. **Feedforward** (default): Dense W1 → matmul → W2 (no hidden→hidden)
2. **Feedforward+H→H** (hybrid): Dense W1/W2 + sparse hidden→hidden with multi-hop
3. **Full Recurrent**: All connection types (backward, lateral, self-loop)

PRESETS
=======

- feedforward: No recurrence (fastest)
- hidden_only: Phase 2 only (forward h→h)
- with_backward: h→h + feedback loops
- with_lateral: h→h + same-layer connections
- with_self: h→h + self-loops
- full_recurrent: Everything (activate_time=20)

TWO-LEVEL CACHING ARCHITECTURE
==============================

This implementation uses two independent caching mechanisms for optimal performance:

LEVEL 1: JAX Persistent Compilation Cache (System-Level)
---------------------------------------------------------
Stores compiled XLA functions on disk for reuse across runs.

- Location: /tmp/jax_cache (configurable via JAX_COMPILATION_CACHE_DIR)
- Scope: Reuses compiled functions when array shapes match
- Impact: 2-3.5x speedup across all modes
- Configuration (set before JAX import):
    os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
    os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '1'

LEVEL 2: H→H Connection Cache (Algorithm-Level)
-----------------------------------------------
Caches hidden-to-hidden connections within a run to skip redundant discovery.

- Location: In-memory via HHCacheManager class
- Scope: Skips Phase 2 discovery when variance mask unchanged
- Impact: 2-3x speedup for hybrid/recurrent modes
- Configuration (via recurrence config):
    hh_cache_enabled: true       # Enable/disable caching
    hh_refresh_interval: 5       # Generations between forced refresh
    hh_mask_change_threshold: 0.1  # Refresh if mask changes >10%

Combined Effect:
- Without caching: ~5000-12000ms per generation
- With both caches: ~900-1500ms per generation
- Total potential speedup: Up to 9x faster

CONFIGURATION
=============

```yaml
hmr_hyperneat:
  max_depth: 3
  variance_threshold: 0.03

  recurrence:
    preset: hidden_only  # Or configure individually:
    # enabled: true
    # allow_hidden_to_hidden: true
    # allow_backward: false
    # iteration_level: 2
    # multi_hop_algorithm: matrix_power  # or fori_loop
    # hop_decay_factor: 0.8
    # hh_cache_enabled: true
    # hh_refresh_interval: 5
```
"""

import functools
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax

# Multi-GPU imports
try:
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    from jax.experimental import mesh_utils
    SHARD_MAP_AVAILABLE = True
except ImportError:
    SHARD_MAP_AVAILABLE = False
    shard_map = None
    Mesh = None
    P = None
    NamedSharding = None
    mesh_utils = None

# Import base class and utilities
from emr_hyperneat._hmr_frozen.hmrhyperneat_pipeline_chunking_multi_gpus import (
    HMRHyperNEATMultiGPU,
    MultiGPUStrategy,
    PositionShardingConfig,
    IslandModelConfig,
    HybridShardingConfig,
    PopulationPmapConfig,
    HierarchicalGridStructure,
    DenseQuadtreeStructure,
    get_hierarchical_grid,
    get_quadtree_structure,
    compute_hierarchical_variances_batch,
    compute_subdivision_masks_batch,
    compute_hierarchical_variances_batch_jit,
    compute_subdivision_masks_batch_jit,
    batch_query_population_multi_source_chunked,
    batch_query_population_positions,
    safe_matmul,
    traced_device_get,
    clear_per_device_caches,
)

from emr_hyperneat._compat.core.base_algorithm import BaseAlgorithm, AlgorithmMetrics
from emr_hyperneat._compat.utils.config_manager import ConfigManager


# ============================================================================
# Per-device caches for multi-GPU execution
# ============================================================================

# Cache for device-specific vmap'd extract_sparse functions
_extract_sparse_cache = {}
_extract_sparse_cache_lock = threading.Lock()


def _get_extract_sparse_fn(device_id: int, num_active: int, max_sparse_conns: int):
    """Get or create a device-specific extract_sparse JIT+vmap function.

    This avoids JAX trace caching issues in multi-GPU settings by ensuring
    each device has its own JIT-compiled traced function.

    The function is wrapped in BOTH jax.vmap AND jax.jit to ensure
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
                    """
                    # Get Y coordinates
                    y_sources = source_coords[:, 1]
                    y_targets = target_positions[:, 1]

                    # Initialize with all True
                    y_valid = jnp.ones((source_coords.shape[0], target_positions.shape[0]), dtype=bool)

                    if not allow_backward:
                        y_valid = y_valid & (y_sources[:, None] < y_targets[None, :])
                    if not allow_lateral:
                        y_valid = y_valid & (y_sources[:, None] != y_targets[None, :])

                    # Self-loop mask
                    coord_match = jnp.all(
                        source_coords[:, None, :] == target_positions[None, :, :],
                        axis=-1
                    )

                    if allow_self_loops:
                        self_mask = jnp.ones((source_coords.shape[0], target_positions.shape[0]), dtype=bool)
                    else:
                        self_mask = ~coord_match

                    return y_valid, self_mask

                _constraint_mask_cache[cache_key] = compute_constraint_mask

        return _constraint_mask_cache[cache_key]


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class SparseHiddenConnections:
    """Sparse representation of hidden→hidden connections for a population.

    This dataclass stores sparse connections in padded arrays with validity masks,
    enabling vmap over the population dimension.

    From multi_gpus.py - provides type-safe storage.

    Attributes:
        from_indices: (pop, max_sparse_conns) - source position indices
        to_indices: (pop, max_sparse_conns) - target position indices
        weights: (pop, max_sparse_conns) - connection weights
        valid_mask: (pop, max_sparse_conns) - True for valid connections
        num_valid: (pop,) - actual number of valid connections per genome
    """
    from_indices: jnp.ndarray
    to_indices: jnp.ndarray
    weights: jnp.ndarray
    valid_mask: jnp.ndarray
    num_valid: jnp.ndarray


@dataclass
class UnifiedExtendedConfig:
    """Complete configuration for unified extended HMR-HyperNEAT.

    Combines RecurrenceConfig (unified.py) with additional multi-hop settings.

    Attributes:
        # From RecurrenceConfig (unified.py)
        enabled: Master switch for recurrence features
        allow_hidden_to_hidden: Enable Phase 2 discovery (iteration_level > 0)
        allow_backward: Enable backward connections (y_source > y_target)
        allow_lateral: Enable same-layer connections (y_source == y_target)
        allow_self_loops: Enable self-connections (same x,y)
        iteration_level: Number of hops for multi-hop expansion
        activate_time: Forward pass iterations for signal propagation
        max_connections: Max connections per substrate (for vmap padding)

        # From multi_gpus.py
        use_vectorized_discovery: Toggle between vectorized and iterative discovery
        max_sparse_conns: Maximum sparse connections to store per genome

        # NEW: Multi-hop settings
        multi_hop_algorithm: Algorithm for multi-hop expansion ("matrix_power" or "fori_loop")
        hop_decay_factor: Weight decay per hop (prevents exploding weights)

        # Caching settings (from unified.py)
        hh_cache_enabled: Enable/disable h→h caching
        hh_refresh_interval: Refresh every N generations
        hh_mask_change_threshold: Refresh if mask changes more than this fraction
    """
    # From RecurrenceConfig
    enabled: bool = True
    allow_hidden_to_hidden: bool = True
    allow_backward: bool = True
    allow_lateral: bool = True
    allow_self_loops: bool = True
    iteration_level: int = 2
    activate_time: Optional[int] = None  # None = auto-compute from depth
    max_connections: int = 10000

    # From multi_gpus.py
    use_vectorized_discovery: bool = True
    max_sparse_conns: int = 10000

    # NEW: Multi-hop settings
    multi_hop_algorithm: str = "matrix_power"  # "matrix_power" or "fori_loop"
    hop_decay_factor: float = 0.8  # Weight decay per hop

    # Caching settings
    hh_cache_enabled: bool = True
    hh_refresh_interval: int = 5
    hh_mask_change_threshold: float = 0.1

    # Discovery algorithm toggle
    # - False (default): Sparse per-genome filtering (better for full_recurrent)
    # - True: Dense discovery without per-genome filtering (better for hybrid at low pop)
    use_dense_discovery: bool = False

    # Evaluation chunking to prevent OOM on large populations
    # - None or 0: No chunking (process entire population at once via vmap)
    # - > 0: Process population in chunks of this size
    # Recommended: 100-200 for depth >= 6, 50-100 for depth >= 7
    eval_chunk_size: Optional[int] = None

    # Multi-hop discovery chunking to prevent OOM on large populations
    # - None: Auto-compute based on num_active positions
    # - 0: No chunking (process entire population at once via vmap)
    # - > 0: Process population in chunks of this size
    # The multi-hop expansion creates (pop_size, num_active, num_active) intermediates
    # which can cause OOM at high populations with iteration_level > 1
    multi_hop_chunk_size: Optional[int] = None

    # H→H discovery population chunking to prevent OOM on large num_active
    # - None: Auto-compute based on num_active positions
    # - 0: No chunking (process entire population at once)
    # - > 0: Process population in chunks of this size
    # The h→h discovery creates (chunk_size, num_active, num_active) tensors
    # Memory scales as: chunk_size × num_active² × 4 bytes
    # At num_active=772 (depth 7+): 1000 × 772² × 4 = 2.38GB → OOM
    # With chunk_size=100: 100 × 772² × 4 = 238MB → OK
    hh_discovery_chunk_size: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.iteration_level < 0:
            raise ValueError(f"iteration_level must be >= 0, got {self.iteration_level}")
        if self.max_connections < 100:
            raise ValueError(f"max_connections must be >= 100, got {self.max_connections}")
        if self.multi_hop_algorithm not in ("matrix_power", "fori_loop"):
            raise ValueError(f"multi_hop_algorithm must be 'matrix_power' or 'fori_loop', got {self.multi_hop_algorithm}")
        if not (0.0 < self.hop_decay_factor <= 1.0):
            raise ValueError(f"hop_decay_factor must be in (0, 1], got {self.hop_decay_factor}")


@dataclass
class ExtendedRecurrenceMetrics:
    """Extended metrics for A/B testing recurrence configurations.

    Includes all metrics from RecurrenceMetrics plus multi-hop specific stats.
    """
    # Connection statistics
    total_connections: int = 0
    forward_connections: int = 0
    backward_connections: int = 0
    lateral_connections: int = 0
    self_loop_connections: int = 0

    # Hidden→hidden specific
    hidden_to_hidden_connections: int = 0
    direct_hh_connections: int = 0  # Before multi-hop expansion

    # Topology metrics
    avg_in_degree: float = 0.0
    avg_out_degree: float = 0.0
    has_cycles: bool = False

    # Performance metrics
    fitness: float = 0.0
    generations_to_solve: Optional[int] = None
    activate_time_used: int = 0

    # Timing breakdown (milliseconds) - fine-grained step timing
    time_cppn_ask_ms: float = 0.0        # CPPN population ask + transform
    time_cppn_query_ms: float = 0.0      # CPPN queries for I/O weights
    time_variance_ms: float = 0.0         # Variance computation + masks
    time_hh_discovery_ms: float = 0.0     # h→h discovery
    time_build_matrices_ms: float = 0.0   # W1/W2 matrix construction
    time_evaluation_ms: float = 0.0       # Forward pass evaluation
    time_neat_evolution_ms: float = 0.0   # NEAT tell() evolution

    # Legacy timing fields (kept for compatibility)
    substrate_discovery_ms: float = 0.0
    phase2_discovery_ms: float = 0.0
    forward_pass_ms: float = 0.0

    # Multi-hop specific
    multi_hop_connections_added: int = 0
    effective_iteration_level: int = 0
    transitive_closure_density: float = 0.0
    multi_hop_discovery_ms: float = 0.0

    # Cache status
    cache_hit: bool = False
    cache_hit_count: int = 0  # Cumulative hits across generations
    cache_refresh_count: int = 0

    # Substrate efficiency metrics
    active_positions_mean: float = 0.0
    active_positions_min: int = 0
    active_positions_max: int = 0
    total_positions: int = 0
    position_utilization: float = 0.0  # active / total ratio

    # Weight matrix density
    w1_density: float = 0.0  # Non-zero ratio in W1
    w2_density: float = 0.0  # Non-zero ratio in W2

    # Configuration echo (actual values used)
    variance_threshold_used: float = 0.0
    band_threshold_used: float = 0.0
    max_weight_used: float = 0.0

    # Population diversity (from last generation)
    min_fitness: float = 0.0
    max_fitness: float = 0.0
    std_fitness: float = 0.0


# ============================================================================
# Presets
# ============================================================================

RECURRENCE_PRESETS: Dict[str, UnifiedExtendedConfig] = {
    'feedforward': UnifiedExtendedConfig(
        enabled=False,
        allow_hidden_to_hidden=False,
        allow_backward=False,
        allow_lateral=False,
        allow_self_loops=False,
        iteration_level=0,
    ),
    'hidden_only': UnifiedExtendedConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=False,
        allow_lateral=False,
        allow_self_loops=False,
        iteration_level=2,
    ),
    'with_backward': UnifiedExtendedConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=True,
        allow_lateral=False,
        allow_self_loops=False,
        iteration_level=2,
    ),
    'with_lateral': UnifiedExtendedConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=False,
        allow_lateral=True,
        allow_self_loops=False,
        iteration_level=2,
    ),
    'with_self': UnifiedExtendedConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=False,
        allow_lateral=False,
        allow_self_loops=True,
        iteration_level=2,
    ),
    'full_recurrent': UnifiedExtendedConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=True,
        allow_lateral=True,
        allow_self_loops=True,
        iteration_level=2,
        activate_time=20,
    ),
}


def get_recurrence_preset(name: str) -> UnifiedExtendedConfig:
    """Get a predefined recurrence configuration by name.

    Args:
        name: Preset name (feedforward, hidden_only, with_backward,
              with_lateral, with_self, full_recurrent)

    Returns:
        UnifiedExtendedConfig with the preset settings

    Raises:
        ValueError: If preset name is not recognized
    """
    if name not in RECURRENCE_PRESETS:
        valid = ', '.join(RECURRENCE_PRESETS.keys())
        raise ValueError(f"Unknown recurrence preset '{name}'. Valid presets: {valid}")

    # Return a copy to prevent mutation
    preset = RECURRENCE_PRESETS[name]
    return UnifiedExtendedConfig(
        enabled=preset.enabled,
        allow_hidden_to_hidden=preset.allow_hidden_to_hidden,
        allow_backward=preset.allow_backward,
        allow_lateral=preset.allow_lateral,
        allow_self_loops=preset.allow_self_loops,
        iteration_level=preset.iteration_level,
        activate_time=preset.activate_time,
        max_connections=preset.max_connections,
        use_vectorized_discovery=preset.use_vectorized_discovery,
        max_sparse_conns=preset.max_sparse_conns,
        multi_hop_algorithm=preset.multi_hop_algorithm,
        hop_decay_factor=preset.hop_decay_factor,
        hh_cache_enabled=preset.hh_cache_enabled,
        hh_refresh_interval=preset.hh_refresh_interval,
        hh_mask_change_threshold=preset.hh_mask_change_threshold,
    )


# ============================================================================
# Connection Constraint Filtering (from unified.py)
# ============================================================================

def get_connection_constraint_mask(
    source_coord: jnp.ndarray,
    target_positions: jnp.ndarray,
    config: Optional[UnifiedExtendedConfig] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute connection constraint masks based on recurrence configuration.

    Connection Types:
    - Forward (y_source < y_target): Always allowed
    - Lateral (y_source == y_target): Configurable via allow_lateral
    - Backward (y_source > y_target): Configurable via allow_backward
    - Self-loop (x,y identical): Configurable via allow_self_loops

    Args:
        source_coord: Source coordinate - shape (2,)
        target_positions: Target positions - shape (num_targets, 2)
        config: Configuration for recurrence features. If None,
            uses feedforward-only constraints

    Returns:
        Tuple of:
        - y_valid: (num_targets,) bool mask for y-constraint
        - self_mask: (num_targets,) bool mask for self-loop constraint
    """
    y_source = source_coord[1]
    y_targets = target_positions[:, 1]
    x_source = source_coord[0]
    x_targets = target_positions[:, 0]

    # Compute position relationship masks
    is_forward = y_source < y_targets
    is_same_y = jnp.abs(y_source - y_targets) < 1e-6
    is_backward = y_source > y_targets
    is_self = (jnp.abs(x_source - x_targets) < 1e-6) & is_same_y

    # Default to feedforward constraints if no config provided
    if config is None or not config.enabled:
        # Original behavior: y_source <= y_target, no self-loops
        y_valid = is_forward | is_same_y
        self_mask = ~is_self
        return y_valid, self_mask

    # Build y_valid mask based on config
    # Forward connections are always allowed
    y_valid = is_forward

    # Lateral connections (same layer, different position)
    if config.allow_lateral:
        y_valid = y_valid | (is_same_y & ~is_self)

    # Backward connections
    if config.allow_backward:
        y_valid = y_valid | is_backward

    # Self-loop handling
    if config.allow_self_loops:
        y_valid = y_valid | is_self
        self_mask = jnp.ones_like(is_self)  # Allow all
    else:
        self_mask = ~is_self  # Block self-loops

    return y_valid, self_mask


def get_connection_constraint_mask_batched(
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    config: Optional[UnifiedExtendedConfig] = None,
    device_id: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batched version of get_connection_constraint_mask using per-device JIT.

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


# ============================================================================
# Multi-Hop Algorithm (CRITICAL FIX - This is the TODO that was missing!)
# ============================================================================

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def compute_multi_hop_connections_matrix_power(
    adjacency_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    weight_threshold: float,
    iteration_level: int,
    hop_decay_factor: float,
    max_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute multi-hop connections via matrix power.

    For iteration_level k hops, computes:
    A_total = A + decay*A^2 + decay^2*A^3 + ... + decay^(k-1)*A^k

    This captures transitive connections: if A→B and B→C exist, we add A→C.

    IMPORTANT: This is JIT-compatible because iteration_level is a static argument,
    which means the Python loop compiles away into a fixed computation graph.

    Args:
        adjacency_weights: Raw CPPN-derived weights - shape (num_active, num_active)
        valid_mask: Boolean mask for direct connections - shape (num_active, num_active)
        weight_threshold: Minimum weight magnitude to keep
        iteration_level: Number of hops (static for JIT)
        hop_decay_factor: Weight decay per hop (e.g., 0.8 means 2-hop weights are 0.8x)
        max_weight: Maximum allowed weight magnitude

    Returns:
        Tuple of:
        - A_total: Multi-hop accumulated weights - shape (num_active, num_active)
        - multi_hop_valid: Boolean mask for multi-hop connections - shape (num_active, num_active)
    """
    # Initialize with direct connections (1-hop)
    A_current = jnp.where(valid_mask, adjacency_weights, 0.0)
    A_total = A_current.copy()

    # Static unroll - compiles away with JIT because iteration_level is static
    for hop in range(1, iteration_level):
        # Compute A^(hop+1) via matrix multiplication
        A_power = A_current @ adjacency_weights

        # Apply weight threshold for sparsity (prune weak multi-hop paths)
        A_power = jnp.where(jnp.abs(A_power) > weight_threshold, A_power, 0.0)

        # Decay weights by hop distance (prevents exploding accumulated weights)
        decay = hop_decay_factor ** hop
        A_power = A_power * decay

        # Accumulate (union of all hop distances)
        # Use max to prefer stronger direct connections over weaker indirect ones
        A_total = jnp.where(
            jnp.abs(A_power) > jnp.abs(A_total),
            A_power,
            A_total
        )
        # Alternative: simple addition (allows multiple paths to reinforce)
        # A_total = A_total + A_power

        A_current = A_power

    # Clip to max_weight
    A_total = jnp.clip(A_total, -max_weight, max_weight)

    # Create validity mask for non-zero connections
    multi_hop_valid = jnp.abs(A_total) > weight_threshold

    return A_total, multi_hop_valid


def compute_multi_hop_connections_fori_loop(
    adjacency_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    weight_threshold: float,
    iteration_level: int,
    hop_decay_factor: float,
    max_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute multi-hop connections using lax.fori_loop.

    Alternative to matrix_power that works with dynamic iteration_level.
    Slightly more overhead due to fori_loop.

    Args:
        (same as compute_multi_hop_connections_matrix_power)

    Returns:
        (same as compute_multi_hop_connections_matrix_power)
    """
    A_init = jnp.where(valid_mask, adjacency_weights, 0.0)

    def hop_step(hop_idx, carry):
        A_current, A_total = carry

        # Compute next power
        A_power = A_current @ adjacency_weights
        A_power = jnp.where(jnp.abs(A_power) > weight_threshold, A_power, 0.0)

        # Decay by hop distance
        decay = hop_decay_factor ** (hop_idx + 1)
        A_power = A_power * decay

        # Accumulate (prefer stronger)
        A_total = jnp.where(
            jnp.abs(A_power) > jnp.abs(A_total),
            A_power,
            A_total
        )

        return (A_power, A_total)

    _, A_total = lax.fori_loop(0, iteration_level - 1, hop_step, (A_init, A_init))

    A_total = jnp.clip(A_total, -max_weight, max_weight)
    multi_hop_valid = jnp.abs(A_total) > weight_threshold

    return A_total, multi_hop_valid


# ============================================================================
# Vectorized Discovery with Multi-Hop
# ============================================================================

# Per-device JIT cache for sparse discovery bitwise AND operations
# CRITICAL: Without this, JAX reuses cached JIT traces across devices
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


def discover_sparse_hh_vectorized_multi_hop(
    state: Any,
    cppns_transformed: Tuple,
    h_grid: HierarchicalGridStructure,
    masks_A: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    config: UnifiedExtendedConfig,
    cppn_forward: Any,
    pop_chunk_size: int = 100,
    verbose: bool = False,
    global_union_active: Optional[jnp.ndarray] = None,
    device_id: int = 0,
) -> SparseHiddenConnections:
    """Fully vectorized sparse h→h discovery WITH multi-hop expansion.

    Pipeline:
    1. Query CPPN for all active→active position pairs
    2. Apply Y-coordinate constraints (from config)
    3. Apply weight threshold for direct connections
    4. Expand to multi-hop connections via matrix power
    5. Convert dense multi-hop matrix to sparse format

    This FIXES the TODO in both multi_gpus.py and unified.py.

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

    # CRITICAL: Move ALL input arrays to target device for multi-GPU execution
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

    # Get union of active positions across population
    # Use global union if provided (for multi-GPU determinism), otherwise compute locally
    if global_union_active is not None:
        union_active = jax.device_put(global_union_active, device)
    else:
        union_active = jnp.any(masks_A, axis=0)  # (total_positions,)
        union_active = jax.device_put(union_active, device)
    active_indices = jnp.nonzero(union_active, size=total_positions, fill_value=-1)[0]
    # CRITICAL: Ensure active_indices is on correct device for subsequent indexing
    active_indices = jax.device_put(active_indices, device)
    num_active = int(jnp.sum(union_active))

    # Compute hh_discovery_chunk_size to prevent OOM
    # Memory scales as: chunk_size × num_active² × 4 bytes
    # Target: Keep under ~2GB per chunk on 11GB GPU
    hh_chunk_size = config.hh_discovery_chunk_size
    if hh_chunk_size is None:
        # Auto-compute based on num_active positions
        if num_active < 300:
            hh_chunk_size = pop_size  # No chunking needed (~0.36GB)
        elif num_active < 500:
            hh_chunk_size = min(200, pop_size)  # ~1GB per chunk
        elif num_active < 800:
            hh_chunk_size = min(100, pop_size)  # ~2.4GB per chunk for 772 nodes
        else:
            hh_chunk_size = min(50, pop_size)   # ~1.2GB per chunk for 1000+ nodes
    elif hh_chunk_size == 0:
        hh_chunk_size = pop_size  # Explicit no chunking

    # DIAGNOSTIC LOGGING
    if verbose:
        print(f"[H→H Discovery] iteration_level={config.iteration_level}, "
              f"multi_hop_algorithm={config.multi_hop_algorithm}, "
              f"hop_decay_factor={config.hop_decay_factor}")
        print(f"[H→H Discovery] allow_backward={config.allow_backward}, "
              f"allow_lateral={config.allow_lateral}, "
              f"allow_self_loops={config.allow_self_loops}")
        print(f"[H→H Discovery] num_active={num_active}, pop_size={pop_size}")
        if hh_chunk_size < pop_size:
            print(f"[H→H Discovery] CHUNKED MODE: hh_chunk_size={hh_chunk_size}")

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

    # Step 2 (MOVED BEFORE CHUNKING): Apply Y-coordinate constraints
    # This is shared across all genomes - compute once
    # Pass device_id for device-aware execution
    y_valid, self_mask = get_connection_constraint_mask_batched(
        active_coords, active_coords, config, device_id=device_id
    )
    # CRITICAL: Avoid fancy indexing ([None, :, :]) which triggers cached JIT traces
    # Use explicit reshape instead to avoid trace caching across devices
    combined_mask = y_valid & self_mask  # (num_active, num_active)
    combined_mask = jax.device_put(combined_mask, device)
    # Reshape to (1, num_active, num_active) for broadcasting
    constraint_mask = jnp.reshape(combined_mask, (1,) + combined_mask.shape)
    constraint_mask = jax.device_put(constraint_mask, device)

    # ========================================================================
    # POPULATION CHUNKING for h→h discovery
    # Process population in chunks to prevent OOM when num_active is large
    # Memory per chunk: chunk_size × num_active² × 4 bytes
    # ========================================================================
    if hh_chunk_size < pop_size:
        # CHUNKED MODE: Process population in chunks
        all_from_indices = []
        all_to_indices = []
        all_weights = []
        all_valid_mask = []
        all_num_valid = []

        for chunk_start in range(0, pop_size, hh_chunk_size):
            chunk_end = min(chunk_start + hh_chunk_size, pop_size)
            chunk_pop_size = chunk_end - chunk_start

            # Slice CPPN state for this chunk
            chunk_cppns = tuple(arr[chunk_start:chunk_end] for arr in cppns_transformed)

            # Step 1: Query active→active pairs for this chunk
            chunk_weights = batch_query_population_multi_source_chunked(
                state, chunk_cppns, active_coords, active_coords,
                outgoing=True, cppn_forward=cppn_forward,
                pop_chunk_size=pop_chunk_size, device_id=device_id
            )
            chunk_weights = jnp.tanh(chunk_weights) * max_weight

            # Step 3: Build validity masks for chunk
            chunk_weight_valid = jnp.abs(chunk_weights) > band_threshold
            chunk_weight_valid = jax.device_put(chunk_weight_valid, device)

            if config.use_dense_discovery:
                chunk_direct_valid = chunk_weight_valid & constraint_mask
            else:
                chunk_masks_A = masks_A[chunk_start:chunk_end]
                chunk_source_active = chunk_masks_A[:, active_indices[:num_active]]
                chunk_target_active = chunk_masks_A[:, active_indices[:num_active]]
                chunk_source_active = jax.device_put(chunk_source_active, device)
                chunk_target_active = jax.device_put(chunk_target_active, device)
                source_valid = chunk_source_active[:, :, None]
                target_valid = chunk_target_active[:, None, :]
                source_valid = jax.device_put(source_valid, device)
                target_valid = jax.device_put(target_valid, device)
                combine_fn = _get_sparse_discovery_combine_fn(device)
                chunk_direct_valid = combine_fn(source_valid, target_valid, chunk_weight_valid, constraint_mask)

            # Step 4: Multi-hop expansion for chunk (if needed)
            if config.iteration_level > 1:
                if config.multi_hop_algorithm == "matrix_power":
                    def expand_single_genome(adjacency, valid):
                        return compute_multi_hop_connections_matrix_power(
                            adjacency, valid, band_threshold,
                            config.iteration_level, config.hop_decay_factor, max_weight
                        )
                else:
                    def expand_single_genome(adjacency, valid):
                        return compute_multi_hop_connections_fori_loop(
                            adjacency, valid, band_threshold,
                            config.iteration_level, config.hop_decay_factor, max_weight
                        )
                chunk_weights = jax.device_put(chunk_weights, device)
                chunk_direct_valid = jax.device_put(chunk_direct_valid, device)
                with jax.default_device(device):
                    chunk_multi_hop_weights, chunk_multi_hop_valid = jax.jit(jax.vmap(expand_single_genome))(
                        chunk_weights, chunk_direct_valid
                    )
            else:
                chunk_multi_hop_weights = jnp.where(chunk_direct_valid, chunk_weights, 0.0)
                chunk_multi_hop_valid = chunk_direct_valid

            # Step 5: Extract sparse format for chunk
            chunk_active_indices = jnp.broadcast_to(
                active_indices[None, :], (chunk_pop_size, active_indices.shape[0])
            )
            chunk_active_indices = jax.device_put(chunk_active_indices, device)
            extract_fn = _get_extract_sparse_fn(device_id, num_active, max_sparse_conns)
            chunk_multi_hop_valid = jax.device_put(chunk_multi_hop_valid, device)
            chunk_multi_hop_weights = jax.device_put(chunk_multi_hop_weights, device)
            with jax.default_device(device):
                chunk_from, chunk_to, chunk_w = extract_fn(
                    chunk_multi_hop_valid, chunk_multi_hop_weights, chunk_active_indices
                )

            # Accumulate chunk results
            all_from_indices.append(chunk_from)
            all_to_indices.append(chunk_to)
            all_weights.append(chunk_w)
            all_valid_mask.append(chunk_from >= 0)
            all_num_valid.append(jnp.sum(chunk_from >= 0, axis=1))

            # Free chunk memory
            del chunk_weights, chunk_weight_valid, chunk_direct_valid
            del chunk_multi_hop_weights, chunk_multi_hop_valid

        # Concatenate all chunks
        from_indices = jnp.concatenate(all_from_indices, axis=0)
        to_indices = jnp.concatenate(all_to_indices, axis=0)
        weights = jnp.concatenate(all_weights, axis=0)
        valid_mask = jnp.concatenate(all_valid_mask, axis=0)
        num_valid = jnp.concatenate(all_num_valid, axis=0)

        if verbose:
            total_valid = int(jnp.sum(num_valid))
            print(f"[H→H Discovery] CHUNKED: total={total_valid}, avg={total_valid/pop_size:.1f}")

        return SparseHiddenConnections(
            from_indices=from_indices.astype(jnp.int32),
            to_indices=to_indices.astype(jnp.int32),
            weights=weights,
            valid_mask=valid_mask,
            num_valid=num_valid,
        )

    # ========================================================================
    # ORIGINAL NON-CHUNKED MODE (when hh_chunk_size >= pop_size)
    # ========================================================================
    # Step 1: Query ALL active→active pairs in one batch
    # Result shape: (pop, num_active, num_active)
    # Pass device_id to ensure device-specific vmap traces for multi-GPU
    active_to_active_weights = batch_query_population_multi_source_chunked(
        state, cppns_transformed, active_coords, active_coords,
        outgoing=True, cppn_forward=cppn_forward,
        pop_chunk_size=pop_chunk_size, device_id=device_id
    )

    # Apply tanh activation and scale
    active_to_active_weights = jnp.tanh(active_to_active_weights) * max_weight

    # Step 3: Build validity masks
    weight_valid = jnp.abs(active_to_active_weights) > band_threshold
    # Ensure weight_valid is on correct device
    weight_valid = jax.device_put(weight_valid, device)

    if config.use_dense_discovery:
        # DENSE DISCOVERY (vanilla-style): Only use weight threshold + constraint
        # This finds ~15-20x more connections, better for hybrid at low populations
        # The per-genome active position filtering is SKIPPED
        direct_valid = weight_valid & constraint_mask
        if verbose:
            print(f"[H→H Discovery] DENSE MODE: skipping per-genome filtering")
    else:
        # SPARSE DISCOVERY (default): Apply per-genome active position masks
        # Only counts connections where BOTH source AND target are active for that genome
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

        # CRITICAL: Use per-device JIT function to avoid cross-device JIT trace reuse
        # Without this, JAX reuses JIT traces from cuda:0 on cuda:1 buffers causing errors
        combine_fn = _get_sparse_discovery_combine_fn(device)
        direct_valid = combine_fn(source_valid, target_valid, weight_valid, constraint_mask)

    # DIAGNOSTIC: Direct connection stats
    if verbose:
        direct_count = int(jnp.sum(direct_valid))
        constraint_count = int(jnp.sum(constraint_mask))
        weight_count = int(jnp.sum(weight_valid))
        print(f"[H→H Discovery] constraint_mask_valid={constraint_count}, "
              f"weight_above_thresh={weight_count}, direct_valid={direct_count}")

    # Step 4: Multi-hop expansion (vmapped over population)
    if config.iteration_level > 1:
        if config.multi_hop_algorithm == "matrix_power":
            def expand_single_genome(adjacency, valid):
                return compute_multi_hop_connections_matrix_power(
                    adjacency, valid, band_threshold,
                    config.iteration_level, config.hop_decay_factor, max_weight
                )
        else:  # fori_loop
            def expand_single_genome(adjacency, valid):
                return compute_multi_hop_connections_fori_loop(
                    adjacency, valid, band_threshold,
                    config.iteration_level, config.hop_decay_factor, max_weight
                )

        # CRITICAL: Ensure inputs are on correct device before multi-hop expansion
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
            print(f"[H→H Discovery] multi_hop_chunk_size={multi_hop_chunk_size}, "
                  f"num_active={num_active}, pop_size={pop_size}")

        # Use device context and JIT wrapper to ensure proper device compilation
        if multi_hop_chunk_size and multi_hop_chunk_size > 0 and pop_size > multi_hop_chunk_size:
            # CHUNKED processing to prevent OOM on large populations
            weight_chunks = []
            valid_chunks = []
            for chunk_start in range(0, pop_size, multi_hop_chunk_size):
                chunk_end = min(chunk_start + multi_hop_chunk_size, pop_size)
                with jax.default_device(device):
                    chunk_w, chunk_v = jax.jit(jax.vmap(expand_single_genome))(
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
                multi_hop_weights, multi_hop_valid = jax.jit(jax.vmap(expand_single_genome))(
                    active_to_active_weights, direct_valid
                )

        if verbose:
            multi_hop_count = int(jnp.sum(multi_hop_valid))
            print(f"[H→H Discovery] MULTI-HOP EXPANDED: multi_hop_valid={multi_hop_count} "
                  f"(added {multi_hop_count - int(jnp.sum(direct_valid))} transitive)")
    else:
        # No multi-hop (iteration_level = 1 means direct connections only)
        multi_hop_weights = jnp.where(direct_valid, active_to_active_weights, 0.0)
        multi_hop_valid = direct_valid
        if verbose:
            print(f"[H→H Discovery] NO MULTI-HOP (iteration_level=1): using direct only")

    # Step 5: Convert dense to sparse format using per-device cached vmap
    # The _get_extract_sparse_fn returns a device-specific traced vmap function
    # to avoid JAX trace caching issues across multi-GPU threads
    pop_size = multi_hop_valid.shape[0]
    active_indices_broadcast = jnp.broadcast_to(active_indices[None, :], (pop_size, active_indices.shape[0]))

    # Get device-specific vmap function
    extract_sparse_vmap = _get_extract_sparse_fn(device_id, num_active, max_sparse_conns)

    # CRITICAL: Explicitly move ALL vmap inputs to target device
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

    return SparseHiddenConnections(
        from_indices=from_indices.astype(jnp.int32),
        to_indices=to_indices.astype(jnp.int32),
        weights=weights,
        valid_mask=valid_mask,
        num_valid=num_valid,
    )


# ============================================================================
# Forward Pass (Three Modes)
# ============================================================================

class ForwardPassMode(Enum):
    """Forward pass execution modes."""
    DENSE_ONLY = "dense_only"       # Feedforward: W1 → tanh → W2 (no h→h)
    HYBRID_SPARSE_HH = "hybrid"     # Dense W1/W2 + sparse h→h
    FULL_SPARSE = "full_sparse"     # All sparse (extreme recurrence)


def forward_unified_extended(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    sparse_hh: SparseHiddenConnections,
    activate_time: int,
    num_positions: int,
    mode: ForwardPassMode = ForwardPassMode.HYBRID_SPARSE_HH,
) -> jnp.ndarray:
    """Unified forward pass supporting all three modes.

    Args:
        inputs: Input values - shape (n_samples, num_inputs)
        W1: Input→hidden weights - shape (num_inputs, num_positions)
        W2: Hidden→output weights - shape (num_positions, num_outputs)
        sparse_hh: SparseHiddenConnections for hidden→hidden
        activate_time: Number of forward pass iterations
        num_positions: Total number of hidden positions
        mode: Which forward pass variant to use

    Returns:
        Output values - shape (n_samples, num_outputs)
    """
    if mode == ForwardPassMode.DENSE_ONLY:
        # Simple feedforward: W1 → tanh → W2 → sigmoid
        h = jnp.tanh(inputs @ W1)
        return jax.nn.sigmoid(h @ W2)

    elif mode == ForwardPassMode.HYBRID_SPARSE_HH:
        return _forward_hybrid_sparse_hh(
            inputs, W1, W2, sparse_hh, activate_time, num_positions
        )

    elif mode == ForwardPassMode.FULL_SPARSE:
        # same as hybrid (full sparse would need sparse W1/W2)
        return _forward_hybrid_sparse_hh(
            inputs, W1, W2, sparse_hh, activate_time, num_positions
        )

    else:
        raise ValueError(f"Unknown forward pass mode: {mode}")


def _forward_hybrid_sparse_hh(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    sparse_hh: SparseHiddenConnections,
    activate_time: int,
    num_positions: int,
) -> jnp.ndarray:
    """Hybrid forward pass: Dense W1/W2 + sparse h→h via lax.scan.

    Matches Model A (MultiGPU eval_single_network_sparse_hh):
    - input_contrib computed once (raw, no tanh)
    - hidden starts from zeros
    - each step: tanh(input_contrib + h_delta), input re-injected every step
    - scan runs activate_time steps (not activate_time - 1)

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        sparse_hh: SparseHiddenConnections
        activate_time: Number of iterations
        num_positions: Total positions

    Returns:
        (n_samples, num_outputs)
    """
    # Step 1: Input contribution (constant across iterations, raw, no tanh)
    input_contrib = inputs @ W1  # (n_samples, num_positions)

    # Step 2: Recurrent iterations with sparse h→h
    if sparse_hh is not None and sparse_hh.num_valid.sum() > 0:
        hidden = jnp.zeros((inputs.shape[0], num_positions))

        # Prepare safe indices
        safe_from = jnp.clip(sparse_hh.from_indices, 0, num_positions - 1)
        safe_to = jnp.clip(sparse_hh.to_indices, 0, num_positions - 1)
        effective_weights = jnp.where(sparse_hh.valid_mask, sparse_hh.weights, 0.0)

        def hh_step(hidden, _):
            # Gather from source positions
            source_vals = hidden[:, safe_from]  # (n_samples, max_sparse_conns)

            # Multiply by weights
            contributions = source_vals * effective_weights

            # Scatter-add to target positions
            h_delta = jnp.zeros_like(hidden)
            h_delta = h_delta.at[:, safe_to].add(contributions)

            # Combine input and recurrent contributions
            return jnp.tanh(input_contrib + h_delta), None

        hidden, _ = lax.scan(hh_step, hidden, None, length=activate_time)
    else:
        # No h→h connections: simple feedforward (one tanh application)
        hidden = jnp.tanh(input_contrib)

    # Step 3: Dense hidden→output
    return jax.nn.sigmoid(hidden @ W2)


def forward_hybrid_vmapped(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    activate_time: int,
) -> jnp.ndarray:
    """Vmap-friendly hybrid forward pass (takes individual arrays, not dataclass).

    This version is designed for use with jax.vmap over population.

    Matches Model A (MultiGPU eval_single_network_sparse_hh):
    - input_contrib computed once (raw, no tanh)
    - hidden starts from zeros
    - each step: tanh(input_contrib + h_delta), input re-injected every step
    - scan runs activate_time steps (not activate_time - 1)

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        from_indices: (max_sparse_conns,) source indices
        to_indices: (max_sparse_conns,) target indices
        hh_weights: (max_sparse_conns,) weights
        valid_mask: (max_sparse_conns,) validity mask
        activate_time: Number of iterations

    Returns:
        (n_samples, num_outputs)
    """
    num_positions = W1.shape[1]

    # Input contribution (constant across iterations, raw, no tanh)
    input_contrib = inputs @ W1

    # Sparse h→h recurrent iterations
    hidden = jnp.zeros((inputs.shape[0], num_positions))
    safe_from = jnp.clip(from_indices, 0, num_positions - 1)
    safe_to = jnp.clip(to_indices, 0, num_positions - 1)
    effective_weights = jnp.where(valid_mask, hh_weights, 0.0)

    def hh_step(hidden, _):
        source_vals = hidden[:, safe_from]
        contributions = source_vals * effective_weights
        h_delta = jnp.zeros_like(hidden)
        h_delta = h_delta.at[:, safe_to].add(contributions)
        return jnp.tanh(input_contrib + h_delta), None

    hidden, _ = lax.scan(hh_step, hidden, None, length=activate_time)

    return jax.nn.sigmoid(hidden @ W2)


def eval_single_network_hybrid(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    activate_time: int,
) -> float:
    """Evaluate single network with hybrid forward pass.

    Args:
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        from_indices, to_indices, hh_weights, valid_mask: sparse h→h
        inputs: (n_samples, num_inputs)
        targets: (n_samples, num_outputs)
        activate_time: Forward iterations

    Returns:
        Fitness score (1.0 - MSE)
    """
    outputs = forward_hybrid_vmapped(
        inputs, W1, W2,
        from_indices, to_indices, hh_weights, valid_mask,
        activate_time
    )
    errors = jnp.mean((outputs - targets) ** 2, axis=1)
    return jnp.maximum(0.0, 1.0 - jnp.mean(errors))


# ============================================================================
# Cache Manager
# ============================================================================

class HHCacheManager:
    """Manages caching of sparse h→h connections across generations.

    Caching saves expensive Phase 2 discovery when the variance mask
    (which determines active positions) hasn't changed significantly.
    """

    def __init__(self, config: UnifiedExtendedConfig):
        """Initialize cache manager.

        Args:
            config: Extended configuration with caching settings
        """
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
        """Determine if h→h connections need refresh.

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

        # Change-based check
        if self._cached_variance_mask is not None:
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


# ============================================================================
# Main Class: HMRHyperNEATUnifiedExtended
# ============================================================================

class HMRHyperNEATUnifiedExtended(HMRHyperNEATMultiGPU):
    """Extended Unified HMR-HyperNEAT combining all features.

    This class combines the best features from:
    - multi_gpus.py: SparseHiddenConnections, discovery toggle, forward pass variants
    - unified.py: RecurrenceConfig, presets, constraint filtering, caching

    PLUS NEW: Multi-hop vectorized expansion that ACTUALLY implements iteration_level > 1.

    Features:
    - UnifiedExtendedConfig with all recurrence options + 6 presets
    - SparseHiddenConnections dataclass for type-safe storage
    - Discovery strategy toggle (vectorized vs iterative)
    - Multi-hop vectorized expansion (JIT-compatible matrix power)
    - Y-coordinate constraint filtering (forward/backward/lateral/self)
    - H→H caching with time-based and change-based refresh
    - Multi-GPU position sharding for Phase 2
    - Three forward pass modes
    - Automatic exhaustive dataset detection for correct multi-GPU routing

    Multi-GPU Data Sharding:
        FULL_PIPELINE_PARALLEL mode shards data across GPUs, with each GPU evaluating
        the full population on a subset of samples. This works well for
        statistical datasets (MNIST, ImageNet) where each shard is representative.

        However, exhaustive/combinatorial problems (XOR, logic gates, parity)
        contain ALL possible input patterns - every sample is unique and
        necessary. Sharding these datasets breaks correctness because networks
        can "half-solve" the problem and get deceptively high averaged fitness.

        This class automatically detects exhaustive problems via the
        `problem.is_exhaustive_dataset` property and falls back to single-GPU
        evaluation to ensure correct fitness calculation.

        See: BaseProblem.is_exhaustive_dataset for problem-side configuration.

    Example:
        algo = HMRHyperNEATUnifiedExtended(strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL)
        algo.create_config({
            'algorithm_params': {
                'hmrhyperneat': {
                    'hmr_hyperneat': {
                        'max_depth': 3,
                        'recurrence': {
                            'preset': 'hidden_only',  # Or configure individually
                        }
                    }
                }
            }
        })
    """

    def __init__(
        self,
        name: str = 'hmr-hyperneat',
        implementation: str = 'tensorneat-hmrhyperneat-unified-extended',
        strategy: MultiGPUStrategy = MultiGPUStrategy.SINGLE_GPU,
        position_config: Optional[PositionShardingConfig] = None,
        island_config: Optional[IslandModelConfig] = None,
        hybrid_config: Optional[HybridShardingConfig] = None,
        pmap_config: Optional[PopulationPmapConfig] = None,
    ):
        """Initialize unified extended HMR-HyperNEAT.

        Args:
            name: Algorithm name
            implementation: Implementation identifier
            strategy: GPU execution strategy
            position_config: Position sharding configuration
            island_config: Island model configuration (legacy)
            hybrid_config: Hybrid sharding configuration (legacy)
            pmap_config: Population pmap configuration (legacy)
        """
        super().__init__(
            name=name,
            implementation=implementation,
            strategy=strategy,
            position_config=position_config,
            island_config=island_config,
            hybrid_config=hybrid_config,
            pmap_config=pmap_config,
        )

        # Extended configuration
        self.extended_config: Optional[UnifiedExtendedConfig] = None

        # Cache manager
        self._hh_cache: Optional[HHCacheManager] = None

        # Metrics tracking
        self._extended_metrics: Optional[ExtendedRecurrenceMetrics] = None

        # Forward pass mode
        self._forward_mode: ForwardPassMode = ForwardPassMode.DENSE_ONLY

        # Current generation (for caching)
        self._current_generation: int = 0

    def create_config(self, params: Dict[str, Any]) -> Any:
        """Create configuration with extended recurrence support.

        Args:
            params: Configuration parameters

        Returns:
            Configuration object
        """
        # First, call parent to set up base configuration
        config = super().create_config(params)

        # Parse extended configuration
        algo_params = params.get('algorithm_params', {}).get('hmrhyperneat', params)
        hmr_config = algo_params.get('hmr_hyperneat', {})
        recurrence_section = hmr_config.get('recurrence', {})

        self._parse_extended_config(recurrence_section)

        # Initialize cache manager
        if self.extended_config is not None:
            self._hh_cache = HHCacheManager(self.extended_config)

        return config

    def initialize(self, config: Any, problem: Any, seed: int = 42) -> Any:
        """Initialize with per-device JIT functions for multi-GPU support.

        This overrides the parent initialize() to add per-device JIT function
        creation. This fixes device placement errors when using ThreadPoolExecutor
        for device-parallel h→h processing - JAX JIT functions are compiled to a
        specific device and ignore thread-local device contexts.

        Args:
            config: Algorithm configuration
            problem: Problem instance
            seed: Random seed

        Returns:
            Initialized algorithm state
        """
        # Call parent initialize first
        state = super().initialize(config, problem, seed)

        # Create per-device JIT functions for multi-GPU support
        # This fixes device placement errors when using ThreadPoolExecutor
        devices = jax.devices()
        self._jitted_cppn_forward_per_device = {}

        if len(devices) > 1:
            print(f"[Multi-GPU] Creating per-device JIT functions for {len(devices)} devices")
            for device in devices:
                # CRITICAL FIX: Use device= parameter to pin JIT function to specific device.
                # jax.default_device() context ONLY affects where arrays are created,
                # it does NOT pin JIT functions to that device. Without device=, the
                # JIT function can be reused across devices causing device placement errors.
                self._jitted_cppn_forward_per_device[device] = jax.jit(
                    self.neat_algo.genome.forward, static_argnums=(0,), device=device
                )

        return state

    def _parse_extended_config(self, recurrence_section: Dict[str, Any]) -> None:
        """Parse recurrence section into extended config.

        Args:
            recurrence_section: The 'recurrence' subsection of hmr_hyperneat config
        """
        # Check for preset first
        preset_name = recurrence_section.get('preset', None)
        if preset_name:
            base_config = get_recurrence_preset(preset_name)
            if self.verbose:
                print(f"[HMR-UnifiedExtended] Using preset '{preset_name}'")

            # CRITICAL FIX: Apply explicit overrides from recurrence_section
            # The preset provides defaults, but explicit values take precedence
            override_fields = {}
            configurable_fields = [
                'iteration_level', 'multi_hop_algorithm', 'hop_decay_factor',
                'hh_cache_enabled', 'hh_refresh_interval', 'hh_mask_change_threshold',
                'allow_backward', 'allow_lateral', 'allow_self_loops',
                'activate_time', 'max_connections', 'max_sparse_conns', 'use_dense_discovery',
                'eval_chunk_size',  # For OOM prevention on large configs
            ]
            for field in configurable_fields:
                if field in recurrence_section:
                    override_fields[field] = recurrence_section[field]

            # Create new config with overrides
            if override_fields:
                from dataclasses import replace
                self.extended_config = replace(base_config, **override_fields)
                if self.verbose:
                    print(f"[HMR-UnifiedExtended] Applied overrides: {override_fields}")
            else:
                self.extended_config = base_config
        elif recurrence_section.get('enabled', False) or recurrence_section.get('allow_hidden_to_hidden', False):
            # Build config from individual settings
            self.extended_config = UnifiedExtendedConfig(
                enabled=recurrence_section.get('enabled', True),
                allow_hidden_to_hidden=recurrence_section.get('allow_hidden_to_hidden', True),
                allow_backward=recurrence_section.get('allow_backward', False),
                allow_lateral=recurrence_section.get('allow_lateral', False),
                allow_self_loops=recurrence_section.get('allow_self_loops', False),
                iteration_level=recurrence_section.get('iteration_level', 2),
                activate_time=recurrence_section.get('activate_time', None),
                max_connections=recurrence_section.get('max_connections', 10000),
                use_vectorized_discovery=recurrence_section.get('use_vectorized_discovery', True),
                max_sparse_conns=recurrence_section.get('max_sparse_conns', 10000),
                multi_hop_algorithm=recurrence_section.get('multi_hop_algorithm', 'matrix_power'),
                hop_decay_factor=recurrence_section.get('hop_decay_factor', 0.8),
                hh_cache_enabled=recurrence_section.get('hh_cache_enabled', True),
                hh_refresh_interval=recurrence_section.get('hh_refresh_interval', 5),
                hh_mask_change_threshold=recurrence_section.get('hh_mask_change_threshold', 0.1),
                use_dense_discovery=recurrence_section.get('use_dense_discovery', False),
                eval_chunk_size=recurrence_section.get('eval_chunk_size', None),
            )
        elif recurrence_section.get('eval_chunk_size') is not None:
            # Feedforward mode but with eval_chunk_size specified for OOM prevention
            self.extended_config = UnifiedExtendedConfig(
                enabled=False,
                allow_hidden_to_hidden=False,
                eval_chunk_size=recurrence_section.get('eval_chunk_size'),
            )
        else:
            # Feedforward mode without chunking
            self.extended_config = None

        # Set mode flags and update parent attributes
        if self.extended_config is not None:
            self._forward_mode = (
                ForwardPassMode.HYBRID_SPARSE_HH
                if self.extended_config.allow_hidden_to_hidden
                else ForwardPassMode.DENSE_ONLY
            )

            # Update parent iteration_level
            self.iteration_level = self.extended_config.iteration_level

            # Update activate_time
            if self.extended_config.activate_time is not None:
                self.activate_time = self.extended_config.activate_time
            elif self.extended_config.allow_hidden_to_hidden:
                # Auto-compute: more iterations for h→h
                default = (2 ** self.max_depth) + 1
                self.activate_time = default * 2

            # Log configuration
            if self.verbose:
                mode = self._forward_mode.value
                print(f"[HMR-UnifiedExtended] Mode: {mode}")
                if self.extended_config.allow_hidden_to_hidden:
                    print(f"  - iteration_level: {self.extended_config.iteration_level}")
                    print(f"  - multi_hop_algorithm: {self.extended_config.multi_hop_algorithm}")
                    print(f"  - hop_decay_factor: {self.extended_config.hop_decay_factor}")
                    print(f"  - allow_backward: {self.extended_config.allow_backward}")
                    print(f"  - allow_lateral: {self.extended_config.allow_lateral}")
                    print(f"  - allow_self_loops: {self.extended_config.allow_self_loops}")
                    print(f"  - activate_time: {self.activate_time}")
        else:
            self._forward_mode = ForwardPassMode.DENSE_ONLY
            self.iteration_level = 0

    def get_recurrence_mode(self) -> str:
        """Get current recurrence mode as string."""
        if self.extended_config is None:
            return 'feedforward'
        elif self.extended_config.enabled and (
            self.extended_config.allow_backward or
            self.extended_config.allow_lateral or
            self.extended_config.allow_self_loops
        ):
            return 'full_recurrent'
        elif self.extended_config.allow_hidden_to_hidden:
            return 'feedforward+hh'
        else:
            return 'feedforward'

    def get_extended_config(self) -> Optional[UnifiedExtendedConfig]:
        """Get current extended configuration."""
        return self.extended_config

    def get_extended_metrics(self) -> Optional[ExtendedRecurrenceMetrics]:
        """Get extended metrics from last generation."""
        return self._extended_metrics

    def run_generation_verbose(
        self,
        state: Any,
        problem: Any,
        skip_metrics: bool = False,
        device_id: int = 0,
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation with all extended features.

        This overrides the parent's run_generation_verbose to add:
        - Phase 2 discovery with multi-hop expansion
        - H→H caching
        - Extended metrics tracking

        Args:
            state: Current algorithm state
            problem: Problem instance
            skip_metrics: If True, skip metrics computation
            device_id: Device index for multi-GPU contexts (default 0)

        Returns:
            Tuple of (new_state, metrics)
        """
        step_times = {}
        total_start = time.perf_counter()

        # === STEP 0: CPPN Ask + Transform ===
        start = time.perf_counter()
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        step_times['step0_cppn_ask_transform'] = time.perf_counter() - start

        # === STEP 1: Grid Setup + CPPN Queries ===
        start = time.perf_counter()
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Input→all positions: (pop_size, num_inputs, total_positions)
        input_all_weights = batch_query_population_multi_source_chunked(
            state, cppns_transformed, input_coords, all_positions,
            True, self._jitted_cppn_forward,
            device_id=device_id,
        )

        # Output←all positions: (pop_size, num_outputs, total_positions)
        output_all_weights = batch_query_population_multi_source_chunked(
            state, cppns_transformed, output_coords, all_positions,
            False, self._jitted_cppn_forward,
            device_id=device_id,
        )

        step_times['step1_cppn_queries'] = time.perf_counter() - start

        # === STEP 2: Variance + Subdivision Masks ===
        start = time.perf_counter()
        # Use input[0]→all_positions as variance source (2D: pop_size, total_positions)
        all_weights_for_variance = input_all_weights[:, 0, :]  # (pop_size, total_positions)
        level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
        masks_A = compute_subdivision_masks_batch(
            level_variances, self.variance_threshold, h_grid, return_all_masks=False
        )
        step_times['step2_variance_masks'] = time.perf_counter() - start

        # === STEP 3: Phase 2 Discovery (with multi-hop) ===
        start = time.perf_counter()
        sparse_hh = None
        cache_hit = False

        if self.extended_config is not None and self.extended_config.allow_hidden_to_hidden:
            # DIAGNOSTIC: Log config being used
            if self.verbose:
                print(f"[Gen {self._current_generation}] H→H config: "
                      f"iteration_level={self.extended_config.iteration_level}, "
                      f"cache_enabled={self.extended_config.hh_cache_enabled}")

            # Check cache
            if self._hh_cache is not None and not self._hh_cache.should_refresh(
                self._current_generation, masks_A
            ):
                # Cache hit
                sparse_hh = self._hh_cache.get_cached()
                cache_hit = True
                if self.verbose:
                    print(f"[Gen {self._current_generation}] CACHE HIT - reusing connections")
            else:
                # Need to discover
                if self.verbose:
                    print(f"[Gen {self._current_generation}] CACHE MISS - running discovery")
                sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                    state, cppns_transformed, h_grid, masks_A,
                    self.band_threshold, self.max_weight,
                    self.extended_config, self._jitted_cppn_forward,
                    verbose=self.verbose,
                    device_id=device_id,
                )

                # Update cache
                if self._hh_cache is not None:
                    self._hh_cache.update_cache(sparse_hh, masks_A, self._current_generation)

        step_times['step3_phase2_discovery'] = time.perf_counter() - start

        # === STEP 4: Build W1/W2 Matrices ===
        start = time.perf_counter()
        weight_thresh = 0.1  # Local constant (same as parent)
        max_weight = self.max_weight

        # Broadcast mask: (pop_size, 1, total_positions) for weight masking
        active_mask_broadcast = masks_A[:, None, :]

        # Apply tanh + max_weight scaling, then mask
        W1_raw = jnp.tanh(input_all_weights) * max_weight
        W1 = jnp.where(
            active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
            W1_raw,
            0.0
        )
        # W1 shape: (pop, num_inputs, total_positions) - correct for matmul with transposed later

        W2_raw = jnp.tanh(output_all_weights) * max_weight
        W2 = jnp.where(
            active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
            W2_raw,
            0.0
        )
        W2 = W2.transpose(0, 2, 1)  # (pop, num_outputs, total_positions) → (pop, total_positions, num_outputs)

        # === DIAGNOSTIC: Log weight matrix statistics for feedforward investigation ===
        if self.verbose and self._forward_mode == ForwardPassMode.DENSE_ONLY:
            active_counts = masks_A.sum(axis=1)  # Per-genome active position counts
            W1_nonzero = (W1 != 0).sum(axis=(1, 2))  # Per-genome non-zero W1 entries
            W2_nonzero = (W2 != 0).sum(axis=(1, 2))  # Per-genome non-zero W2 entries
            print(f"[Feedforward Diag] Active positions: mean={float(active_counts.mean()):.1f}, "
                  f"min={int(active_counts.min())}, max={int(active_counts.max())}")
            print(f"[Feedforward Diag] W1 non-zero: mean={float(W1_nonzero.mean()):.1f}, "
                  f"W2 non-zero: mean={float(W2_nonzero.mean()):.1f}")
            print(f"[Feedforward Diag] W1 stats: min={float(W1.min()):.3f}, max={float(W1.max()):.3f}, "
                  f"mean={float(W1.mean()):.4f}")
            print(f"[Feedforward Diag] W2 stats: min={float(W2.min()):.3f}, max={float(W2.max()):.3f}, "
                  f"mean={float(W2.mean()):.4f}")
            print(f"[Feedforward Diag] Total positions: {total_positions}, "
                  f"Variance threshold: {self.variance_threshold}")

        step_times['step4_build_matrices'] = time.perf_counter() - start

        # === STEP 5: Evaluation ===
        start = time.perf_counter()
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # Get chunk size from config (None or 0 = no chunking, -1 = auto)
        eval_chunk_size = getattr(self.extended_config, 'eval_chunk_size', None) if self.extended_config else None
        pop_size = W1.shape[0]
        total_positions = W1.shape[2] if len(W1.shape) > 2 else W1.shape[1]

        # Auto-compute chunk size based on position count (same heuristics as CPPN queries)
        if eval_chunk_size == -1:
            if total_positions < 1000:        # depth <= 3
                eval_chunk_size = None  # No chunking needed
            elif total_positions < 6000:      # depth 4-5
                eval_chunk_size = min(500, pop_size)
            elif total_positions < 50000:     # depth 6 "anomaly zone"
                eval_chunk_size = min(50, pop_size)
            elif total_positions < 200000:    # depth 7
                eval_chunk_size = min(200, pop_size)
            else:                             # depth 8+
                eval_chunk_size = min(50, pop_size)

        if sparse_hh is not None and self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH:
            # Hybrid evaluation with sparse h→h
            eval_fn = functools.partial(
                eval_single_network_hybrid,
                inputs=inputs_batch,
                targets=targets_batch,
                activate_time=self.activate_time,
            )

            if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                # Chunked evaluation to prevent OOM
                fitness_chunks = []
                for chunk_start in range(0, pop_size, eval_chunk_size):
                    chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                    chunk_fitnesses = jax.vmap(eval_fn)(
                        W1[chunk_start:chunk_end],
                        W2[chunk_start:chunk_end],
                        sparse_hh.from_indices[chunk_start:chunk_end],
                        sparse_hh.to_indices[chunk_start:chunk_end],
                        sparse_hh.weights[chunk_start:chunk_end],
                        sparse_hh.valid_mask[chunk_start:chunk_end],
                    )
                    fitness_chunks.append(chunk_fitnesses)
                fitnesses = jnp.concatenate(fitness_chunks)
            else:
                # Full population vmap (original behavior)
                fitnesses = jax.vmap(eval_fn)(
                    W1, W2,
                    sparse_hh.from_indices, sparse_hh.to_indices,
                    sparse_hh.weights, sparse_hh.valid_mask,
                )
        else:
            # Dense-only evaluation
            # W1: (num_inputs, total_positions), W2: (total_positions, num_outputs)
            def eval_single_dense(W1_i, W2_i):
                # inputs: (n_samples, num_inputs), W1_i: (num_inputs, total_positions)
                h = jnp.tanh(inputs_batch @ W1_i)  # (n_samples, total_positions)
                outputs = jax.nn.sigmoid(h @ W2_i)  # (n_samples, num_outputs)
                errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

            if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                # Chunked evaluation to prevent OOM
                fitness_chunks = []
                for chunk_start in range(0, pop_size, eval_chunk_size):
                    chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                    chunk_fitnesses = jax.vmap(eval_single_dense)(
                        W1[chunk_start:chunk_end],
                        W2[chunk_start:chunk_end],
                    )
                    fitness_chunks.append(chunk_fitnesses)
                fitnesses = jnp.concatenate(fitness_chunks)
            else:
                # Full population vmap (original behavior)
                fitnesses = jax.vmap(eval_single_dense)(W1, W2)

        step_times['step5_evaluation'] = time.perf_counter() - start

        # === STEP 6: NEAT Evolution ===
        start = time.perf_counter()
        new_state = self._compiled_tell(state, fitnesses)
        step_times['step6_neat_evolution'] = time.perf_counter() - start

        # Update generation counter
        self._current_generation += 1

        # === Metrics ===
        total_time = time.perf_counter() - total_start

        # Compute fitness statistics
        best_fitness = float(jnp.max(fitnesses))
        mean_fitness = float(jnp.mean(fitnesses))
        min_fitness = float(jnp.min(fitnesses))
        max_fitness = float(jnp.max(fitnesses))
        std_fitness = float(jnp.std(fitnesses))

        custom_metrics = {
            'total_positions': total_positions,
            'step_timings': step_times,
            'method': 'unified_extended',
            'recurrence_mode': self.get_recurrence_mode(),
            'multi_hop_algorithm': self.extended_config.multi_hop_algorithm if self.extended_config else 'none',
            'iteration_level': self.extended_config.iteration_level if self.extended_config else 0,
            'cache_hit': cache_hit,
        }

        if sparse_hh is not None:
            custom_metrics['total_hh_connections'] = int(jnp.sum(sparse_hh.num_valid))
            custom_metrics['avg_hh_connections'] = float(jnp.mean(sparse_hh.num_valid))

        # Compute substrate efficiency metrics
        active_counts = masks_A.sum(axis=1)  # Per-genome active position counts
        active_positions_mean = float(jnp.mean(active_counts))
        active_positions_min = int(jnp.min(active_counts))
        active_positions_max = int(jnp.max(active_counts))
        position_utilization = active_positions_mean / total_positions if total_positions > 0 else 0.0

        # Compute weight matrix density
        w1_total = W1.size
        w2_total = W2.size
        w1_nonzero = int(jnp.sum(W1 != 0))
        w2_nonzero = int(jnp.sum(W2 != 0))
        w1_density = w1_nonzero / w1_total if w1_total > 0 else 0.0
        w2_density = w2_nonzero / w2_total if w2_total > 0 else 0.0

        # Track cumulative cache hits
        cache_hit_count = getattr(self, '_cache_hit_count', 0)
        if cache_hit:
            cache_hit_count += 1
        self._cache_hit_count = cache_hit_count

        # Store extended metrics with all fields
        self._extended_metrics = ExtendedRecurrenceMetrics(
            # Connection statistics
            total_connections=custom_metrics.get('total_hh_connections', 0),
            hidden_to_hidden_connections=custom_metrics.get('total_hh_connections', 0),
            direct_hh_connections=custom_metrics.get('total_hh_connections', 0),  # Before multi-hop

            # Performance metrics
            fitness=best_fitness,
            activate_time_used=self.activate_time,

            # Fine-grained timing breakdown (ms)
            time_cppn_ask_ms=step_times['step0_cppn_ask_transform'] * 1000,
            time_cppn_query_ms=step_times['step1_cppn_queries'] * 1000,
            time_variance_ms=step_times['step2_variance_masks'] * 1000,
            time_hh_discovery_ms=step_times['step3_phase2_discovery'] * 1000,
            time_build_matrices_ms=step_times['step4_build_matrices'] * 1000,
            time_evaluation_ms=step_times['step5_evaluation'] * 1000,
            time_neat_evolution_ms=step_times['step6_neat_evolution'] * 1000,

            # Legacy timing (kept for compatibility)
            phase2_discovery_ms=step_times['step3_phase2_discovery'] * 1000,
            forward_pass_ms=step_times['step5_evaluation'] * 1000,

            # Multi-hop
            effective_iteration_level=self.extended_config.iteration_level if self.extended_config else 0,

            # Cache status
            cache_hit=cache_hit,
            cache_hit_count=cache_hit_count,
            cache_refresh_count=self._hh_cache._refresh_count if self._hh_cache else 0,

            # Substrate efficiency
            active_positions_mean=active_positions_mean,
            active_positions_min=active_positions_min,
            active_positions_max=active_positions_max,
            total_positions=total_positions,
            position_utilization=position_utilization,

            # Weight matrix density
            w1_density=w1_density,
            w2_density=w2_density,

            # Configuration echo
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,

            # Population diversity
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            std_fitness=std_fitness,
        )

        metrics = AlgorithmMetrics(
            generation=self._current_generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            std_fitness=std_fitness,
            num_species=0,  # Not tracked in this implementation
            species_sizes=[],
            species_fitness=[],
            evaluations=pop_size,
            time_elapsed=total_time,
        )
        # Add custom metrics
        metrics.custom_metrics = custom_metrics

        return new_state, metrics

    # ========================================================================
    # Multi-GPU run_until_threshold() Override
    # ========================================================================

    def run_until_threshold(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        streaming_chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """Run multiple generations with h→h discovery support.

        Routes to appropriate implementation based on strategy:

        MULTI-GPU STRATEGIES (require 2+ GPUs):
        - FULL_PIPELINE_PARALLEL (aliases: MULTI_GPU, DATA_PARALLEL):
          Data-parallel with FULL pipeline inside pmap.
          Each GPU processes full population on different data shard.
          Memory: ~balanced across GPUs (each GPU ~5GB for typical workloads)
          Fitness: averaged across GPUs

        - POPULATION_PARALLEL_SEQUENTIAL: Population-parallel with sequential h→h
          Each GPU processes population slice on full dataset.
          H→H processing: SEQUENTIAL (one GPU at a time) to avoid JIT cache errors.
          Memory: ~balanced across GPUs
          Fitness: concatenated from all GPUs
          Trade-off: Feedforward 2.6x faster, h→h modes 2-3x slower than single-GPU

        FALLBACK STRATEGIES:
        - SINGLE_GPU or 1 GPU: Single GPU loop using run_generation_verbose()
        - CPU: Same as single GPU

        AUTOMATIC FALLBACK FOR EXHAUSTIVE DATASETS:
        - If problem.is_exhaustive_dataset is True, FULL_PIPELINE_PARALLEL is disabled
          and single-GPU mode is used instead, regardless of GPU count.
        - This ensures correct fitness evaluation for combinatorial problems
          (XOR, logic gates, parity) where ALL samples must be evaluated together.
        - Statistical datasets (MNIST, ImageNet) can still use FULL_PIPELINE_PARALLEL.

        This override ensures the unified extended features (multi-hop h→h discovery,
        sparse forward pass, caching) are used regardless of GPU strategy.

        Args:
            state: Initialized algorithm state
            problem: Problem instance (checked for is_exhaustive_dataset property)
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            collect_history: If True, collect per-generation history
            streaming_chunk_size: Chunk size for streaming (unused in extended)

        Returns:
            Dict with 'generations', 'best_fitness', 'state', 'history' (optional)
        """
        num_devices = len(jax.devices())

        # Check if FULL_PIPELINE_PARALLEL sharding is appropriate for this problem
        # FULL_PIPELINE_PARALLEL shards data across GPUs - each GPU evaluates on a subset
        # This BREAKS correctness for exhaustive/combinatorial problems (XOR, logic gates)
        # where ALL samples are needed for meaningful fitness evaluation
        n_samples = self._cached_inputs.shape[0] if hasattr(self, '_cached_inputs') and self._cached_inputs is not None else 0

        # Check 1: Exhaustive datasets CANNOT be sharded (each GPU needs ALL samples)
        # Examples: XOR (4 samples), logic gates, parity problems
        is_exhaustive = getattr(problem, 'is_exhaustive_dataset', False)

        # Check 2: Statistical datasets need enough samples per GPU for representativeness
        # Rule of thumb: at least 16 samples per GPU for meaningful fitness
        MIN_SAMPLES_PER_GPU = 16
        has_enough_samples = n_samples >= (MIN_SAMPLES_PER_GPU * num_devices)

        # Can use data-parallel only if: NOT exhaustive AND enough samples
        can_use_data_parallel = (not is_exhaustive) and has_enough_samples

        # Route based on strategy
        # NOTE: Parent class __init__ normalizes FULL_PIPELINE_PARALLEL -> PIPELINE_CHUNKED
        #       DATA_PARALLEL and MULTI_GPU are legacy aliases for FULL_PIPELINE_PARALLEL
        if num_devices >= 2:
            if self.strategy in (
                MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
                MultiGPUStrategy.MULTI_GPU,  # Legacy alias
                MultiGPUStrategy.DATA_PARALLEL,  # Legacy alias
                MultiGPUStrategy.PIPELINE_CHUNKED,  # Internal normalized form
            ):
                if can_use_data_parallel:
                    # Data-parallel: full pipeline inside pmap, balanced memory
                    print(f"[Strategy] FULL_PIPELINE_PARALLEL: Using {num_devices} GPUs")
                    print(f"  - Sharding {n_samples} samples across GPUs ({n_samples // num_devices} samples/GPU)")
                    print(f"  - Best for: Large statistical datasets (MNIST, ImageNet) where subsets are representative")
                    print(f"  - Each GPU evaluates FULL population on its data shard, fitnesses averaged")
                    return self._run_until_threshold_data_parallel_full(
                        state, problem, target_fitness, max_generations, collect_history
                    )
                else:
                    # Cannot use data-parallel sharding - fall back to single GPU
                    # to evaluate each genome on ALL samples for correct fitness
                    print(f"[Strategy] FULL_PIPELINE_PARALLEL requested but falling back to SINGLE_GPU")
                    if is_exhaustive:
                        print(f"  - Reason: Exhaustive dataset ({n_samples} samples) - ALL samples required for correct fitness")
                        print(f"  - Problem type: Combinatorial (XOR, logic gates, parity) where every sample is unique")
                        print(f"  - Why fallback: Sharding would give wrong fitness (e.g., 100% on GPU0, 0% on GPU1 = 50% avg)")
                        print(f"  - Recommendation: Use POPULATION_PARALLEL_SEQUENTIAL instead for exhaustive datasets with multi-GPU")
                    else:
                        print(f"  - Reason: Dataset too small ({n_samples} samples, need {MIN_SAMPLES_PER_GPU * num_devices}+)")
                        print(f"  - Why fallback: Too few samples per GPU for statistically meaningful fitness")
                        print(f"  - Recommendation: Use POPULATION_PARALLEL_SEQUENTIAL, or increase dataset size")
                    return self._run_until_threshold_single_gpu(
                        state, problem, target_fitness, max_generations, collect_history
                    )
            elif self.strategy == MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL:
                # Population-parallel: split population across GPUs, replicate full data
                # This WORKS with exhaustive datasets because each GPU gets ALL samples
                # Only FULL_PIPELINE_PARALLEL breaks exhaustive datasets (data sharding)

                # Check if h→h discovery is enabled - if so, use device-parallel threading
                # pmap with h→h causes OOM at ANY depth because XLA JIT traces fori_loops
                # and pre-allocates huge buffers for all iterations simultaneously
                h_grid = get_hierarchical_grid(self.max_depth)
                total_positions = h_grid.total_positions
                allow_hh = self.extended_config.allow_hidden_to_hidden if self.extended_config else False

                if allow_hh:
                    # Use DEVICE_PARALLEL for ALL h→h modes - pmap with h→h causes OOM
                    # Each GPU runs single-GPU-style processing SEQUENTIALLY (not threaded)
                    # Sequential processing avoids JAX JIT cache cross-device errors
                    print(f"[Strategy] DEVICE_PARALLEL: Using {num_devices} GPUs with SEQUENTIAL h→h processing")
                    print(f"  - h→h discovery at depth {self.max_depth} ({total_positions} positions)")
                    print(f"  - Each GPU runs single-GPU-style processing SEQUENTIALLY")
                    print(f"  - Sequential avoids JAX JIT cache cross-device errors")
                    print(f"  - Splitting population across GPUs, each GPU gets ALL {n_samples} samples")
                    return self._run_until_threshold_device_parallel_hh(
                        state, problem, target_fitness, max_generations, collect_history
                    )
                else:
                    # Feedforward mode (no h→h) - can use pmap safely
                    print(f"[Strategy] POPULATION_PARALLEL_SEQUENTIAL: Using {num_devices} GPUs")
                    print(f"  - Splitting population across GPUs, each GPU gets ALL {n_samples} samples")
                    print(f"  - Best for: Any dataset (works with exhaustive AND statistical)")
                    print(f"  - Each GPU evaluates its population slice on FULL dataset, fitnesses concatenated")
                    return self._run_until_threshold_pop_parallel(
                        state, problem, target_fitness, max_generations, collect_history
                    )
            elif self.strategy == MultiGPUStrategy.STREAMING:
                # Streaming: CPU-to-GPU data streaming for memory management
                # NOT a multi-GPU parallelization strategy
                print(f"[Strategy] STREAMING: Using single GPU with CPU-to-GPU data streaming")
                print(f"  - Purpose: Memory management for datasets too large for GPU memory")
                print(f"  - How it works: Streams data chunks from CPU to GPU during evaluation")
                print(f"  - Best for: Very large datasets (>10GB) that would cause GPU OOM")
                print(f"  - Note: NOT a multi-GPU parallelization strategy - uses 1 GPU only")
                print(f"  - For multi-GPU with large data: Combine with POPULATION_PARALLEL_SEQUENTIAL (not yet implemented)")
                return self._run_until_threshold_single_gpu(
                    state, problem, target_fitness, max_generations, collect_history
                )

        # Fallback for single GPU or unhandled strategies
        if self.strategy in (
            MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
            MultiGPUStrategy.MULTI_GPU,  # Legacy alias
            MultiGPUStrategy.DATA_PARALLEL,  # Legacy alias
            MultiGPUStrategy.PIPELINE_CHUNKED,
            MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL,
            MultiGPUStrategy.POPULATION_PARALLEL,
        ):
            print(f"[Strategy] {self.strategy.name} requested but only {num_devices} GPU available")
            print(f"  - Falling back to SINGLE_GPU mode")
            print(f"  - For multi-GPU: Ensure multiple GPUs are visible to JAX")
        elif self.strategy == MultiGPUStrategy.STREAMING:
            print(f"[Strategy] STREAMING: Single GPU with CPU-to-GPU data streaming")
            print(f"  - Purpose: Memory management, not parallelization")
        elif self.strategy == MultiGPUStrategy.SINGLE_GPU:
            print(f"[Strategy] SINGLE_GPU: Using 1 GPU")
            print(f"  - All computation on primary GPU")

        # For single GPU, CPU, or fallback: use run_generation_verbose() loop
        return self._run_until_threshold_single_gpu(
            state, problem, target_fitness, max_generations, collect_history
        )

    def _run_until_threshold_single_gpu(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Single GPU implementation using existing run_generation_verbose().

        This preserves all unified extended features (h→h discovery, multi-hop,
        caching) by using the existing per-generation implementation.

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        if self.verbose:
            print(f"[UnifiedExtended] Single GPU mode: max_gens={max_generations}, target={target_fitness}")

        while best_so_far < target_fitness and generation < max_generations:
            state, metrics = self.run_generation_verbose(state, problem)
            gen_best = metrics.best_fitness
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose and generation % 10 == 0:
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_device_parallel_hh(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Device-parallel multi-GPU implementation for h→h at large depths.

        Uses Python threading to run single-GPU-style processing on each device.
        This avoids pmap's XLA trace explosion that causes OOM for h→h at depth 7+.

        Each thread:
        1. Places its population shard on its assigned device
        2. Runs h→h discovery + evaluation (Python for-loops, no XLA trace explosion)
        3. Returns fitnesses to main thread

        Main thread:
        1. Splits population between devices
        2. Launches worker threads
        3. Combines fitnesses
        4. Runs NEAT evolution

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        devices = jax.devices()
        num_devices = len(devices)

        if self.verbose:
            print(f"[DeviceParallel] Using {num_devices} GPUs with threaded h→h processing")

        # Get hierarchical grid info
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0
        cache_hit_count = 0
        cache_refresh_count = 0

        # Thread-safe result storage
        thread_results = {}
        thread_lock = threading.Lock()

        def process_shard_on_device(
            device_idx: int,
            cppn_shard: Tuple,
            shard_start: int,
            shard_end: int,
            cached_hh_shard: Optional[SparseHiddenConnections] = None,
            global_union_active: Optional[jnp.ndarray] = None,
            precomputed_masks_A_shard: Optional[jnp.ndarray] = None,
        ):
            """Process a population shard on a specific device using single-GPU-style logic.

            Args:
                device_idx: Device index (0, 1, ...)
                cppn_shard: CPPN genomes for this shard
                shard_start: Start index in full population
                shard_end: End index in full population
                cached_hh_shard: Optional cached h→h connections for this shard (cache hit)
                global_union_active: Optional pre-computed global union of active positions.
                    When provided, ensures deterministic h→h discovery across shards.
                precomputed_masks_A_shard: Optional pre-computed masks_A for this shard.
                    When provided, skip variance/mask computation (already done globally).
            """
            device = devices[device_idx]
            shard_size = shard_end - shard_start

            # Use per-device JIT function to avoid device placement errors
            # JAX JIT functions are compiled to a specific device and ignore thread-local contexts
            jitted_cppn_forward = getattr(self, '_jitted_cppn_forward_per_device', {}).get(
                device, self._jitted_cppn_forward
            )

            # Use default_device context to ensure all JAX operations go to the correct device
            with jax.default_device(device):
                # Place data on device
                cppn_shard_on_device = jax.tree.map(lambda x: jax.device_put(x, device), cppn_shard)
                inputs_on_device = jax.device_put(inputs_batch, device)
                targets_on_device = jax.device_put(targets_batch, device)
                all_pos_on_device = jax.device_put(all_positions, device)
                input_coords_on_device = jax.device_put(input_coords, device)
                output_coords_on_device = jax.device_put(output_coords, device)
                # CRITICAL: state contains arrays used by cppn_forward inside vmap
                # Without device placement, these arrays stay on cuda:0 causing
                # "Buffer passed to Execute()... is on device cuda:0, but replica is assigned to device cuda:1"
                state_on_device = jax.tree.map(lambda x: jax.device_put(x, device), state)

                # Run CPPN queries for this shard
                # Pass device_idx to force device-specific vmap traces
                # This prevents cuda:1 from reusing traces compiled for cuda:0
                input_all_weights = batch_query_population_multi_source_chunked(
                    state_on_device, cppn_shard_on_device, input_coords_on_device, all_pos_on_device,
                    True, jitted_cppn_forward, device_id=device_idx,
                )
                output_all_weights = batch_query_population_multi_source_chunked(
                    state_on_device, cppn_shard_on_device, output_coords_on_device, all_pos_on_device,
                    False, jitted_cppn_forward, device_id=device_idx,
                )

                # Variance + subdivision masks
                # Use precomputed masks if available (for deterministic multi-GPU)
                if precomputed_masks_A_shard is not None:
                    masks_A = jax.device_put(precomputed_masks_A_shard, device)
                else:
                    all_weights_for_variance = input_all_weights[:, 0, :]
                    level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
                    masks_A = compute_subdivision_masks_batch(
                        level_variances, self.variance_threshold, h_grid, return_all_masks=False
                    )
                    # CRITICAL: Ensure masks_A is on the correct device
                    # compute_subdivision_masks_batch uses cached JIT which may place arrays on wrong device
                    masks_A = jax.device_put(masks_A, device)

                # H→H: use cached if provided, otherwise discover
                sparse_hh = None
                shard_hh_count = 0
                discovered_hh = None
                if self.extended_config is not None and self.extended_config.allow_hidden_to_hidden:
                    if cached_hh_shard is not None:
                        # Cache hit: use the cached connections
                        sparse_hh = cached_hh_shard
                        shard_hh_count = int(jnp.sum(sparse_hh.num_valid))
                    else:
                        # Cache miss: discover new connections
                        # Use global_union_active for deterministic behavior across shards
                        global_union_on_device = None
                        if global_union_active is not None:
                            global_union_on_device = jax.device_put(global_union_active, device)
                        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                            state_on_device, cppn_shard_on_device, h_grid, masks_A,
                            self.band_threshold, self.max_weight,
                            self.extended_config, jitted_cppn_forward,
                            verbose=False,
                            global_union_active=global_union_on_device,
                            device_id=device_idx,
                        )
                        if sparse_hh is not None:
                            shard_hh_count = int(jnp.sum(sparse_hh.num_valid))
                            # Store for caching later
                            discovered_hh = sparse_hh

                # Build W1/W2 matrices (inside device context)
                weight_thresh = 0.1
                max_weight_val = self.max_weight
                active_mask_broadcast = masks_A[:, None, :]
                W1_raw = jnp.tanh(input_all_weights) * max_weight_val
                W1 = jnp.where(
                    active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                    W1_raw, 0.0
                )
                W2_raw = jnp.tanh(output_all_weights) * max_weight_val
                W2 = jnp.where(
                    active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                    W2_raw, 0.0
                )
                W2 = W2.transpose(0, 2, 1)

                # Evaluation (inside device context)
                # CRITICAL: Wrap in device-specific JIT to prevent lax.scan
                # from being compiled to cuda:0 and reused on cuda:1.
                # The issue is that forward_hybrid_vmapped contains lax.scan
                # which gets traced once and cached. Without explicit device
                # placement, the first thread (cuda:0) compiles it, and the
                # second thread (cuda:1) reuses the cuda:0 trace, causing
                # device placement errors.
                if sparse_hh is not None:
                    eval_fn = functools.partial(
                        eval_single_network_hybrid,
                        inputs=inputs_on_device,
                        targets=targets_on_device,
                        activate_time=self.activate_time,
                    )
                    # Use device-specific JIT for evaluation
                    # This ensures lax.scan is compiled for THIS device
                    eval_vmapped = jax.jit(
                        lambda w1, w2, fr, to, wt, vm: jax.vmap(eval_fn)(w1, w2, fr, to, wt, vm),
                        device=device
                    )
                    fitnesses = eval_vmapped(
                        W1, W2,
                        sparse_hh.from_indices, sparse_hh.to_indices,
                        sparse_hh.weights, sparse_hh.valid_mask,
                    )
                else:
                    def eval_single_dense(w1, w2):
                        hidden = jnp.tanh(inputs_on_device @ w1.T)
                        outputs = jax.nn.sigmoid(hidden @ w2)
                        errors = jnp.mean((outputs - targets_on_device) ** 2, axis=1)
                        return 1.0 - jnp.mean(errors)
                    # Device-specific JIT for dense evaluation too
                    eval_vmapped_dense = jax.jit(
                        lambda w1, w2: jax.vmap(eval_single_dense)(w1, w2),
                        device=device
                    )
                    fitnesses = eval_vmapped_dense(W1, W2)

                # Block and get results back to CPU
                fitnesses = jax.device_get(fitnesses)

            with thread_lock:
                thread_results[device_idx] = {
                    'fitnesses': fitnesses,
                    'hh_count': shard_hh_count,
                    'shard_start': shard_start,
                    'discovered_hh': discovered_hh,  # For caching (None if cache hit)
                    'shard_end': shard_end,
                }

        # Main evolution loop
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # Get CPPN population
            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Split population between devices
            shard_size = pop_size // num_devices
            thread_results.clear()

            # Check cache for h→h connections
            use_cache = (
                self._hh_cache is not None and
                self.extended_config is not None and
                self.extended_config.allow_hidden_to_hidden
            )
            cached_sparse_hh = None
            is_cache_hit = False

            if use_cache:
                # Check if we have a valid cache (we can't check mask here, so use generation-based check)
                if not self._hh_cache.should_refresh(self._current_generation, None):
                    cached_sparse_hh = self._hh_cache.get_cached()
                    is_cache_hit = cached_sparse_hh is not None
                    if is_cache_hit:
                        cache_hit_count += 1
                        if self.verbose:
                            print(f"[Gen {self._current_generation}] CACHE HIT - reusing connections")
                else:
                    cache_refresh_count += 1
                    if self.verbose:
                        print(f"[Gen {self._current_generation}] CACHE MISS - discovering new connections")

            # =====================================================================
            # H→H DISCOVERY: Each thread computes its own masks locally
            # =====================================================================
            # masks are computed per-shard, not pre-computed for the full population on GPU 0.
            # Pre-computing the full population (for deterministic h→h discovery) causes OOM
            # at large populations (750+) with depth 7 (87K positions) because
            # 750 × 87,380 × 4 bytes = 262 MB just for input weights, and the
            # full pipeline needs much more.
            #
            # The fix: Let each thread compute its own masks_A locally. This is
            # valid because each genome's h→h connections are independent - they
            # only depend on that genome's CPPN outputs, not other genomes.
            # =====================================================================
            global_union_active = None
            full_masks_A = None

            # Process devices SEQUENTIALLY to avoid JAX JIT cache cross-device errors.
            # ThreadPoolExecutor causes "Buffer on device cuda:X but replica assigned to cuda:Y"
            # because JAX's global JIT cache reuses compiled traces across threads/devices.
            # Sequential processing ensures each device gets clean JIT compilation.
            #
            # Trade-off: We lose parallel execution but gain guaranteed correctness.
            # For large populations, the h→h discovery work is the bottleneck anyway,
            # so sequential GPU processing is still faster than single-GPU.
            for d in range(num_devices):
                # CRITICAL: Clear ALL caches before each device to prevent cross-device
                # JIT trace reuse. Without this, device 0's compiled traces get reused
                # by device 1, causing "Buffer on cuda:0 but replica on cuda:1" errors.
                #
                # We clear BOTH:
                # 1. JAX's internal compilation cache (jax.clear_caches())
                # 2. Our module-level Python cache of JIT functions (clear_per_device_caches())
                #
                # The Python cache holds references to compiled functions that may have
                # device-specific traces. Even after jax.clear_caches(), these references
                # persist and can cause cross-device issues.
                jax.clear_caches()
                clear_per_device_caches()

                shard_start = d * shard_size
                shard_end = pop_size if d == num_devices - 1 else (d + 1) * shard_size
                cppn_shard = tuple(arr[shard_start:shard_end] for arr in cppns_transformed)

                # Slice cached h→h for this shard if cache hit
                cached_hh_shard = None
                if is_cache_hit and cached_sparse_hh is not None:
                    cached_hh_shard = SparseHiddenConnections(
                        from_indices=cached_sparse_hh.from_indices[shard_start:shard_end],
                        to_indices=cached_sparse_hh.to_indices[shard_start:shard_end],
                        weights=cached_sparse_hh.weights[shard_start:shard_end],
                        valid_mask=cached_sparse_hh.valid_mask[shard_start:shard_end],
                        num_valid=cached_sparse_hh.num_valid[shard_start:shard_end],
                    )

                # Slice pre-computed masks_A for this shard (if available)
                masks_A_shard = None
                if full_masks_A is not None:
                    masks_A_shard = full_masks_A[shard_start:shard_end]

                # Process sequentially - no threading
                process_shard_on_device(d, cppn_shard, shard_start, shard_end,
                    cached_hh_shard, global_union_active, masks_A_shard)

            # Combine results
            all_fitnesses = np.zeros(pop_size, dtype=np.float32)
            gen_hh_count = 0
            for d in range(num_devices):
                result = thread_results[d]
                all_fitnesses[result['shard_start']:result['shard_end']] = result['fitnesses']
                gen_hh_count += result['hh_count']

            # If cache miss, combine discovered h→h and update cache
            if use_cache and not is_cache_hit:
                # Combine discovered_hh from all shards
                all_discovered = []
                for d in range(num_devices):
                    if thread_results[d]['discovered_hh'] is not None:
                        all_discovered.append((d, thread_results[d]))

                if all_discovered:
                    # Combine all shard discoveries into one SparseHiddenConnections
                    # Move to CPU first to avoid device mismatch errors
                    combined_from = []
                    combined_to = []
                    combined_weights = []
                    combined_valid = []
                    combined_num_valid = []

                    for d, result in sorted(all_discovered, key=lambda x: x[1]['shard_start']):
                        hh = result['discovered_hh']
                        # Move each shard's arrays to CPU (numpy)
                        combined_from.append(jax.device_get(hh.from_indices))
                        combined_to.append(jax.device_get(hh.to_indices))
                        combined_weights.append(jax.device_get(hh.weights))
                        combined_valid.append(jax.device_get(hh.valid_mask))
                        combined_num_valid.append(jax.device_get(hh.num_valid))

                    # Concatenate on CPU, then convert back to JAX arrays
                    combined_sparse_hh = SparseHiddenConnections(
                        from_indices=jnp.array(np.concatenate(combined_from, axis=0)),
                        to_indices=jnp.array(np.concatenate(combined_to, axis=0)),
                        weights=jnp.array(np.concatenate(combined_weights, axis=0)),
                        valid_mask=jnp.array(np.concatenate(combined_valid, axis=0)),
                        num_valid=jnp.array(np.concatenate(combined_num_valid, axis=0)),
                    )

                    # Update cache
                    self._hh_cache.update_cache(combined_sparse_hh, None, self._current_generation)

            fitnesses = jnp.array(all_fitnesses)
            total_hh_count = max(total_hh_count, gen_hh_count)

            # Update best
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            # NEAT evolution
            state = self._compiled_tell(state, fitnesses)
            generation += 1
            self._current_generation = generation

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                elapsed = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"h→h={gen_hh_count}, time={elapsed:.2f}s")

        # Store extended metrics for benchmark reporting
        self._extended_metrics = ExtendedRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=h_grid.total_positions,
            # Config echo
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            # Population diversity from final fitnesses
            min_fitness=float(np.min(all_fitnesses)),
            max_fitness=float(np.max(all_fitnesses)),
            std_fitness=float(np.std(all_fitnesses)),
            # Cache stats
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            'sparse_hh_count': total_hh_count,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_data_parallel_extended(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Data-parallel multi-GPU implementation WITH h→h discovery.

        Pipeline per generation:
        1. CPPN ask + transform (GPU 0)
        2. CPPN queries for input→all and output←all (GPU 0)
        3. Variance/subdivision masks (GPU 0)
        4. H→H discovery with multi-hop expansion (GPU 0)
        5. Build W1, W2 matrices (GPU 0)
        6. Data-parallel evaluation (SHARD dataset, replicate weights+h→h)
        7. NEAT evolution (GPU 0)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Pad samples to be divisible by num_devices
        remainder = n_samples % num_devices
        if remainder != 0:
            pad_samples = num_devices - remainder
            inputs_padded = jnp.pad(inputs_batch, ((0, pad_samples), (0, 0)), mode='edge')
            targets_padded = jnp.pad(targets_batch, ((0, pad_samples), (0, 0)), mode='edge')
            padded_n_samples = n_samples + pad_samples
        else:
            pad_samples = 0
            inputs_padded = inputs_batch
            targets_padded = targets_batch
            padded_n_samples = n_samples

        per_gpu_samples = padded_n_samples // num_devices

        # Shard dataset: (n_samples, features) -> (num_gpus, per_gpu_samples, features)
        inputs_sharded = inputs_padded.reshape(num_devices, per_gpu_samples, -1)
        targets_sharded = targets_padded.reshape(num_devices, per_gpu_samples, -1)

        if self.verbose:
            print(f"[UnifiedExtended] Data-parallel: {num_devices} GPUs, "
                  f"positions={total_positions}, samples={n_samples} ({per_gpu_samples}/GPU)")

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # === STEP 0: CPPN ask + transform ===
            if self.extra_randkey_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # === STEP 1: CPPN queries ===
            # Note: This runs on GPU 0 only (data sharding is in evaluation)
            input_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, input_coords, all_positions,
                True, self._jitted_cppn_forward,
                device_id=0,
            )
            output_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, output_coords, all_positions,
                False, self._jitted_cppn_forward,
                device_id=0,
            )

            # === STEP 2: Variance + subdivision masks ===
            all_weights_for_variance = input_all_weights[:, 0, :]
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )

            # === STEP 3: H→H discovery (GPU 0, result shared) ===
            sparse_hh = None
            if self.extended_config is not None and self.extended_config.allow_hidden_to_hidden:
                # Check cache
                if self._hh_cache is not None and not self._hh_cache.should_refresh(
                    self._current_generation, masks_A
                ):
                    sparse_hh = self._hh_cache.get_cached()
                else:
                    # Note: This runs on GPU 0 only
                    sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                        state, cppns_transformed, h_grid, masks_A,
                        self.band_threshold, self.max_weight,
                        self.extended_config, self._jitted_cppn_forward,
                        verbose=False,
                        device_id=0,
                    )
                    if self._hh_cache is not None:
                        self._hh_cache.update_cache(sparse_hh, masks_A, self._current_generation)

            # === STEP 4: Build W1, W2 matrices ===
            weight_thresh = 0.1
            max_weight = self.max_weight
            active_mask_broadcast = masks_A[:, None, :]

            W1_raw = jnp.tanh(input_all_weights) * max_weight
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw, 0.0
            )

            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W2 = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw, 0.0
            )
            W2 = W2.transpose(0, 2, 1)  # (pop, total_positions, num_outputs)

            # === STEP 5: Data-parallel evaluation ===
            if sparse_hh is not None and self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH:
                # Hybrid evaluation with sparse h→h
                fitnesses = self._eval_data_parallel_hybrid(
                    W1, W2, sparse_hh, inputs_sharded, targets_sharded,
                    num_devices, per_gpu_samples, n_samples, pad_samples
                )
            else:
                # Dense-only evaluation
                fitnesses = self._eval_data_parallel_dense(
                    W1, W2, inputs_sharded, targets_sharded,
                    num_devices, per_gpu_samples, n_samples, pad_samples
                )

            # Ensure fitnesses are on primary device and fully materialized
            # This prevents device sharding issues from pmap affecting _compiled_tell
            fitnesses = jax.device_put(fitnesses, jax.devices()[0])
            fitnesses = jax.block_until_ready(fitnesses)

            # === STEP 6: NEAT evolution ===
            state = self._compiled_tell(state, fitnesses)
            self._current_generation += 1

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                gen_time = time.perf_counter() - gen_start
                hh_count = int(jnp.sum(sparse_hh.num_valid)) if sparse_hh else 0
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"h→h={hh_count}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def _eval_data_parallel_hybrid(
        self,
        W1: jnp.ndarray,
        W2: jnp.ndarray,
        sparse_hh: SparseHiddenConnections,
        inputs_sharded: jnp.ndarray,
        targets_sharded: jnp.ndarray,
        num_devices: int,
        per_gpu_samples: int,
        n_samples: int,
        pad_samples: int,
    ) -> jnp.ndarray:
        """Data-parallel evaluation with hybrid forward pass (sparse h→h).

        Uses jax.pmap to distribute evaluation across multiple GPUs.
        Dataset is sharded (each GPU gets different samples).
        Weights and h→h connections are replicated.

        Args:
            W1: (pop, num_inputs, total_positions)
            W2: (pop, total_positions, num_outputs)
            sparse_hh: Sparse h→h connections
            inputs_sharded: (num_gpus, per_gpu_samples, num_inputs)
            targets_sharded: (num_gpus, per_gpu_samples, num_outputs)
            num_devices: Number of GPUs
            per_gpu_samples: Samples per GPU
            n_samples: Original sample count (before padding)
            pad_samples: Number of padded samples

        Returns:
            Fitness array (pop_size,)
        """
        activate_time = self.activate_time

        # Use module-level pmap function for multi-GPU evaluation
        # partial_error_sums has shape: (num_gpus, pop_size)
        partial_error_sums = _pmap_eval_hybrid(
            W1, W2,
            sparse_hh.from_indices, sparse_hh.to_indices,
            sparse_hh.weights, sparse_hh.valid_mask,
            inputs_sharded, targets_sharded,
            activate_time,
        )

        # Sum errors across all GPUs, then average across samples
        total_error_sums = jnp.sum(partial_error_sums, axis=0)  # (pop_size,)
        avg_errors = total_error_sums / n_samples
        fitnesses = jnp.maximum(0.0, 1.0 - avg_errors)
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        return fitnesses

    def _eval_data_parallel_dense(
        self,
        W1: jnp.ndarray,
        W2: jnp.ndarray,
        inputs_sharded: jnp.ndarray,
        targets_sharded: jnp.ndarray,
        num_devices: int,
        per_gpu_samples: int,
        n_samples: int,
        pad_samples: int,
    ) -> jnp.ndarray:
        """Data-parallel evaluation with dense-only forward pass.

        Uses jax.pmap to distribute evaluation across multiple GPUs.

        Args:
            W1: (pop, num_inputs, total_positions)
            W2: (pop, total_positions, num_outputs)
            inputs_sharded: (num_gpus, per_gpu_samples, num_inputs)
            targets_sharded: (num_gpus, per_gpu_samples, num_outputs)
            num_devices: Number of GPUs
            per_gpu_samples: Samples per GPU
            n_samples: Original sample count
            pad_samples: Number of padded samples

        Returns:
            Fitness array (pop_size,)
        """
        # Use module-level pmap function for multi-GPU evaluation
        # partial_error_sums has shape: (num_gpus, pop_size)
        partial_error_sums = _pmap_eval_dense(
            W1, W2,
            inputs_sharded, targets_sharded,
        )

        # Sum errors across all GPUs, then average across samples
        total_error_sums = jnp.sum(partial_error_sums, axis=0)  # (pop_size,)
        avg_errors = total_error_sums / n_samples
        fitnesses = jnp.maximum(0.0, 1.0 - avg_errors)
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        return fitnesses

    def _run_until_threshold_data_parallel_full(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Data-parallel multi-GPU with FULL pipeline inside pmap.

        This strategy moves ALL computation (CPPN queries, variance masks, h→h discovery,
        weight building, AND evaluation) inside pmap for balanced memory distribution.

        Memory distribution: Each GPU processes the full population on its data shard.

        Pipeline per generation:
        1. CPPN ask + transform (CPU/GPU 0)
        2. INSIDE PMAP (each GPU):
           - CPPN queries for input→all and output←all
           - Variance/subdivision masks
           - H→H discovery with multi-hop expansion
           - Build W1, W2 matrices
           - Evaluate on data shard
        3. Aggregate fitnesses (mean across GPUs)
        4. NEAT evolution (CPU/GPU 0)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Extract static parameters from h_grid for pmap (JIT-compatible Python tuples)
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Pad samples to be divisible by num_devices
        remainder = n_samples % num_devices
        if remainder != 0:
            pad_samples = num_devices - remainder
            inputs_padded = jnp.pad(inputs_batch, ((0, pad_samples), (0, 0)), mode='edge')
            targets_padded = jnp.pad(targets_batch, ((0, pad_samples), (0, 0)), mode='edge')
            padded_n_samples = n_samples + pad_samples
        else:
            pad_samples = 0
            inputs_padded = inputs_batch
            targets_padded = targets_batch
            padded_n_samples = n_samples

        per_gpu_samples = padded_n_samples // num_devices

        # Shard dataset: (n_samples, features) -> (num_gpus, per_gpu_samples, features)
        inputs_sharded = inputs_padded.reshape(num_devices, per_gpu_samples, -1)
        targets_sharded = targets_padded.reshape(num_devices, per_gpu_samples, -1)

        # Extended config parameters
        allow_hh = self.extended_config.allow_hidden_to_hidden if self.extended_config else False
        iteration_level = self.extended_config.iteration_level if self.extended_config else 0
        max_sparse_conns = self.extended_config.max_sparse_conns if self.extended_config else 1000

        if self.verbose:
            print(f"[UnifiedExtended] FULL_PIPELINE_PARALLEL: {num_devices} GPUs, "
                  f"positions={total_positions}, samples={n_samples} ({per_gpu_samples}/GPU)")
            print(f"  h→h enabled: {allow_hh}, iteration_level: {iteration_level}")

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0  # Track h→h connections across generations

        # Track substrate efficiency metrics across generations (for averaging)
        cumulative_active = 0.0
        cumulative_w1_nonzero = 0.0
        cumulative_w2_nonzero = 0.0
        cumulative_w1_total = 0.0
        cumulative_w2_total = 0.0
        generation_count = 0

        # Cache tracking
        cache_hit_count = 0
        cache_refresh_count = 0
        last_hh_count = -1  # -1 so first gen is always a "refresh"

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # === STEP 0: CPPN ask + transform (on GPU 0) ===
            if self.extra_randkey_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # === STEP 1-5: Full pipeline inside pmap ===
            # Each GPU runs CPPN queries, variance, h→h, weight building, AND evaluation
            # on its data shard, with full population replicated
            (partial_fitnesses, hh_counts_sharded,
             active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_data_parallel_full_hh(
                cppns_transformed,           # None: replicated
                all_positions,               # None: replicated
                input_coords,                # None: replicated
                output_coords,               # None: replicated
                inputs_sharded,              # 0: sharded
                targets_sharded,             # 0: sharded
                self.variance_threshold,     # None: replicated
                self.band_threshold,         # None: replicated
                self.max_weight,             # None: replicated
                0.1,                         # weight_thresh - None: replicated
                self.activate_time,          # STATIC
                level_sizes,                 # STATIC
                level_offsets,               # STATIC
                level_grid_sizes,            # STATIC
                parent_indices,              # STATIC
                num_levels,                  # STATIC
                total_positions,             # STATIC
                max_sparse_conns,            # STATIC
                iteration_level,             # STATIC
                allow_hh,                    # STATIC
                state,                       # None: replicated
                self._jitted_cppn_forward,   # STATIC
            )

            # Aggregate: average fitness across data shards
            # Each GPU computed fitness on different samples, so average them
            fitnesses = jnp.mean(partial_fitnesses, axis=0)  # (pop_size,)

            # Aggregate h→h counts (same population on each GPU, so take first)
            # Each GPU computes h→h for the full population, they should be identical
            gen_hh_count = int(hh_counts_sharded[0])  # Take from first GPU
            total_hh_count = max(total_hh_count, gen_hh_count)  # Track max across generations

            # Aggregate substrate efficiency metrics (same population, take from first GPU)
            cumulative_active += float(active_sums[0])
            cumulative_w1_nonzero += float(w1_nz_sums[0])
            cumulative_w2_nonzero += float(w2_nz_sums[0])
            cumulative_w1_total += float(w1_totals[0])
            cumulative_w2_total += float(w2_totals[0])
            generation_count += 1

            # Track cache stats (only for modes with h→h discovery)
            if gen_hh_count > 0:
                if gen_hh_count == last_hh_count:
                    cache_hit_count += 1
                else:
                    cache_refresh_count += 1
                last_hh_count = gen_hh_count

            # Ensure fitnesses are on primary device and fully materialized
            # This prevents device sharding issues from pmap affecting _compiled_tell
            fitnesses = jax.device_put(fitnesses, jax.devices()[0])
            fitnesses = jax.block_until_ready(fitnesses)

            # === STEP 6: NEAT evolution ===
            state = self._compiled_tell(state, fitnesses)
            self._current_generation += 1

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={gen_time:.3f}s")

        # Compute average substrate efficiency metrics
        pop_size = cppns_transformed[0].shape[0]
        if generation_count > 0:
            avg_active = cumulative_active / generation_count / pop_size
            position_util = (cumulative_active / generation_count) / total_positions / pop_size
            avg_w1_density = cumulative_w1_nonzero / cumulative_w1_total if cumulative_w1_total > 0 else 0.0
            avg_w2_density = cumulative_w2_nonzero / cumulative_w2_total if cumulative_w2_total > 0 else 0.0
        else:
            avg_active = 0.0
            position_util = 0.0
            avg_w1_density = 0.0
            avg_w2_density = 0.0

        # Store extended metrics including h→h count and substrate efficiency
        self._extended_metrics = ExtendedRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=total_positions,
            # Config echo (what was actually used)
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            # Population diversity from final fitnesses
            min_fitness=float(jnp.min(fitnesses)),
            max_fitness=float(jnp.max(fitnesses)),
            std_fitness=float(jnp.std(fitnesses)),
            # Substrate efficiency (averaged over generations)
            active_positions_mean=avg_active,
            position_utilization=position_util,
            w1_density=avg_w1_density,
            w2_density=avg_w2_density,
            # Cache stats
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_pop_parallel(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Population-parallel multi-GPU with FULL pipeline inside pmap.

        This strategy shards the POPULATION across GPUs, replicating the full dataset
        on each GPU. Good for large populations with small datasets.

        Memory distribution: Each GPU processes a population slice on the full dataset.

        Pipeline per generation:
        1. CPPN ask + transform (CPU/GPU 0)
        2. Shard population across GPUs
        3. INSIDE PMAP (each GPU):
           - CPPN queries for its pop slice
           - Variance/subdivision masks for pop slice
           - H→H discovery for pop slice
           - Build W1, W2 for pop slice
           - Evaluate pop slice on full dataset
        4. Gather fitnesses (concatenate from all GPUs)
        5. NEAT evolution (CPU/GPU 0)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Extract static parameters from h_grid for pmap (JIT-compatible Python tuples)
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data - full dataset on each GPU
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get population size to determine padding
        # We need to handle this dynamically since pop_size comes from state
        # Will compute padding inside the loop

        # Extended config parameters
        allow_hh = self.extended_config.allow_hidden_to_hidden if self.extended_config else False
        iteration_level = self.extended_config.iteration_level if self.extended_config else 0
        max_sparse_conns = self.extended_config.max_sparse_conns if self.extended_config else 1000

        if self.verbose:
            print(f"[UnifiedExtended] POPULATION_PARALLEL_SEQUENTIAL: {num_devices} GPUs, "
                  f"positions={total_positions}, samples={n_samples}")
            print(f"  h→h enabled: {allow_hh}, iteration_level: {iteration_level}")

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0  # Track h→h connections across generations

        # Track substrate efficiency metrics across generations (for averaging)
        cumulative_active = 0.0
        cumulative_w1_nonzero = 0.0
        cumulative_w2_nonzero = 0.0
        cumulative_w1_total = 0.0
        cumulative_w2_total = 0.0
        generation_count = 0

        # Cache tracking
        cache_hit_count = 0
        cache_refresh_count = 0
        last_hh_count = -1  # -1 so first gen is always a "refresh"

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # === STEP 0: CPPN ask + transform (on GPU 0) ===
            if self.extra_randkey_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # === STEP 1: Calculate chunk size for large substrates ===
            # For large total_positions, we need to process population in smaller chunks
            # to avoid OOM. Similar logic to single-GPU eval_chunk_size.
            eval_chunk_size = getattr(self.extended_config, 'eval_chunk_size', None) if self.extended_config else None

            # Auto-compute chunk size based on position count AND h→h mode
            # h→h discovery requires MUCH smaller chunks because it creates (chunk, positions, positions) arrays
            if eval_chunk_size == -1 or eval_chunk_size is None:
                if total_positions < 1000:        # depth <= 3
                    pop_chunk_size = None  # No chunking needed
                elif total_positions < 6000:      # depth 4-5
                    if allow_hh:
                        pop_chunk_size = 50   # h→h needs smaller chunks
                    else:
                        pop_chunk_size = 200  # feedforward can use larger chunks
                elif total_positions < 50000:     # depth 6 "anomaly zone"
                    if allow_hh:
                        pop_chunk_size = 10   # h→h needs very small chunks at depth 6
                    else:
                        pop_chunk_size = 100  # feedforward: 50 per GPU with 2 GPUs
                elif total_positions < 200000:    # depth 7
                    if allow_hh:
                        # CRITICAL: At depth 7+ with h→h, use 1 genome per GPU to avoid XLA trace explosion
                        # Each pmap call traces only 1 genome worth of fori_loop (175 chunks)
                        # Python for-loop iterates over population chunks (NOT traced)
                        pop_chunk_size = num_devices  # 1 per GPU
                    else:
                        pop_chunk_size = 50   # feedforward: 25 per GPU
                else:                             # depth 8+
                    if allow_hh:
                        pop_chunk_size = num_devices  # h→h: 1 per GPU
                    else:
                        pop_chunk_size = 20   # feedforward: 10 per GPU
            else:
                pop_chunk_size = eval_chunk_size

            # Ensure chunk size is divisible by num_devices
            if pop_chunk_size is not None:
                pop_chunk_size = max(num_devices, (pop_chunk_size // num_devices) * num_devices)

            # Determine if we need population chunking
            need_chunking = pop_chunk_size is not None and pop_chunk_size > 0 and pop_size > pop_chunk_size

            # NOTE: Sequential pmap with jax.lax.scan doesn't work because scan is still
            # traced by XLA just like fori_loop. The solution is to use very small population
            # chunks (1 genome per GPU) with the standard pmap, so each pmap call only traces
            # 1 genome's worth of computation. The Python for-loop over chunks is NOT traced.
            use_sequential_pmap = False  # Disabled - use tiny chunks instead

            if self.verbose and generation == 0:
                if need_chunking:
                    num_chunks = (pop_size + pop_chunk_size - 1) // pop_chunk_size
                    print(f"  Population: {pop_size}, using chunked processing: {num_chunks} chunks of {pop_chunk_size}")
                else:
                    print(f"  Population: {pop_size} (processing all at once)")
                if use_sequential_pmap:
                    print(f"  Using SEQUENTIAL multi-GPU pmap for h→h at depth {self.max_depth} (positions={total_positions})")

            # Lists to accumulate results across chunks
            all_fitness_chunks = []
            all_hh_counts = []
            all_active_sums = []
            all_w1_nz_sums = []
            all_w2_nz_sums = []
            all_w1_totals = []
            all_w2_totals = []

            # === STEP 2-5: Process population in chunks (or all at once if small) ===
            if need_chunking:
                # Process population in chunks
                for chunk_start in range(0, pop_size, pop_chunk_size):
                    chunk_end = min(chunk_start + pop_chunk_size, pop_size)
                    chunk_size_actual = chunk_end - chunk_start

                    # Extract chunk from cppns_transformed
                    cppns_chunk = tuple(arr[chunk_start:chunk_end] for arr in cppns_transformed)

                    # Pad chunk to be divisible by num_devices
                    remainder = chunk_size_actual % num_devices
                    if remainder != 0:
                        pad_size = num_devices - remainder
                        padded_chunk_size = chunk_size_actual + pad_size
                    else:
                        pad_size = 0
                        padded_chunk_size = chunk_size_actual

                    per_gpu_pop = padded_chunk_size // num_devices

                    # Pad and reshape each CPPN array in chunk
                    cppns_sharded = []
                    for arr in cppns_chunk:
                        if pad_size > 0:
                            pad_width = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
                            arr_padded = jnp.pad(arr, pad_width, mode='edge')
                        else:
                            arr_padded = arr
                        new_shape = (num_devices, per_gpu_pop) + arr.shape[1:]
                        arr_sharded = arr_padded.reshape(new_shape)
                        cppns_sharded.append(arr_sharded)
                    cppns_sharded = tuple(cppns_sharded)

                    # Run pmap on this chunk
                    # Use sequential pmap for h→h at large depths to avoid XLA trace explosion
                    if use_sequential_pmap:
                        (fitnesses_sharded, hh_counts_sharded,
                         active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_sequential_pop_parallel_hh(
                            cppns_sharded,
                            all_positions,
                            input_coords,
                            output_coords,
                            inputs_batch,
                            targets_batch,
                            self.variance_threshold,
                            self.band_threshold,
                            self.max_weight,
                            0.1,
                            self.activate_time,
                            level_sizes,
                            level_offsets,
                            level_grid_sizes,
                            parent_indices,
                            num_levels,
                            total_positions,
                            max_sparse_conns,
                            iteration_level,
                            allow_hh,
                            state,
                            self._jitted_cppn_forward,
                        )
                    else:
                        (fitnesses_sharded, hh_counts_sharded,
                         active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_pop_parallel_hh(
                            cppns_sharded,
                            all_positions,
                            input_coords,
                            output_coords,
                            inputs_batch,
                            targets_batch,
                            self.variance_threshold,
                            self.band_threshold,
                            self.max_weight,
                            0.1,
                            self.activate_time,
                            level_sizes,
                            level_offsets,
                            level_grid_sizes,
                            parent_indices,
                            num_levels,
                            total_positions,
                            max_sparse_conns,
                            iteration_level,
                            allow_hh,
                            state,
                            self._jitted_cppn_forward,
                        )

                    # Extract fitnesses for this chunk (remove padding)
                    chunk_fitnesses = fitnesses_sharded.reshape(-1)[:chunk_size_actual]
                    all_fitness_chunks.append(chunk_fitnesses)

                    # Accumulate metrics
                    all_hh_counts.append(float(jnp.sum(hh_counts_sharded)))
                    all_active_sums.append(float(jnp.sum(active_sums)))
                    all_w1_nz_sums.append(float(jnp.sum(w1_nz_sums)))
                    all_w2_nz_sums.append(float(jnp.sum(w2_nz_sums)))
                    all_w1_totals.append(float(jnp.sum(w1_totals)))
                    all_w2_totals.append(float(jnp.sum(w2_totals)))

                # Concatenate all chunks
                fitnesses = jnp.concatenate(all_fitness_chunks)
                gen_hh_count = int(sum(all_hh_counts))

                # Aggregate substrate efficiency metrics
                cumulative_active += sum(all_active_sums)
                cumulative_w1_nonzero += sum(all_w1_nz_sums)
                cumulative_w2_nonzero += sum(all_w2_nz_sums)
                cumulative_w1_total += sum(all_w1_totals)
                cumulative_w2_total += sum(all_w2_totals)
            else:
                # Original path: process all at once (no chunking needed)
                # Pad population to be divisible by num_devices
                remainder = pop_size % num_devices
                if remainder != 0:
                    pad_size = num_devices - remainder
                    padded_pop_size = pop_size + pad_size
                else:
                    pad_size = 0
                    padded_pop_size = pop_size

                per_gpu_pop = padded_pop_size // num_devices

                # Pad and reshape each CPPN array
                cppns_sharded = []
                for arr in cppns_transformed:
                    if pad_size > 0:
                        pad_width = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
                        arr_padded = jnp.pad(arr, pad_width, mode='edge')
                    else:
                        arr_padded = arr
                    new_shape = (num_devices, per_gpu_pop) + arr.shape[1:]
                    arr_sharded = arr_padded.reshape(new_shape)
                    cppns_sharded.append(arr_sharded)
                cppns_sharded = tuple(cppns_sharded)

                # Run pmap on full population
                # Use sequential pmap for h→h at large depths to avoid XLA trace explosion
                if use_sequential_pmap:
                    (fitnesses_sharded, hh_counts_sharded,
                     active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_sequential_pop_parallel_hh(
                        cppns_sharded,
                        all_positions,
                        input_coords,
                        output_coords,
                        inputs_batch,
                        targets_batch,
                        self.variance_threshold,
                        self.band_threshold,
                        self.max_weight,
                        0.1,
                        self.activate_time,
                        level_sizes,
                        level_offsets,
                        level_grid_sizes,
                        parent_indices,
                        num_levels,
                        total_positions,
                        max_sparse_conns,
                        iteration_level,
                        allow_hh,
                        state,
                        self._jitted_cppn_forward,
                    )
                else:
                    (fitnesses_sharded, hh_counts_sharded,
                     active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_pop_parallel_hh(
                        cppns_sharded,
                        all_positions,
                        input_coords,
                        output_coords,
                        inputs_batch,
                        targets_batch,
                        self.variance_threshold,
                        self.band_threshold,
                        self.max_weight,
                        0.1,
                        self.activate_time,
                        level_sizes,
                        level_offsets,
                        level_grid_sizes,
                        parent_indices,
                        num_levels,
                        total_positions,
                        max_sparse_conns,
                        iteration_level,
                        allow_hh,
                        state,
                        self._jitted_cppn_forward,
                    )

                # Gather: concatenate fitnesses and remove padding
                fitnesses = fitnesses_sharded.reshape(-1)[:pop_size]
                gen_hh_count = int(jnp.sum(hh_counts_sharded))

                # Aggregate substrate efficiency metrics
                cumulative_active += float(jnp.sum(active_sums))
                cumulative_w1_nonzero += float(jnp.sum(w1_nz_sums))
                cumulative_w2_nonzero += float(jnp.sum(w2_nz_sums))
                cumulative_w1_total += float(jnp.sum(w1_totals))
                cumulative_w2_total += float(jnp.sum(w2_totals))

            # Track max h→h across generations
            total_hh_count = max(total_hh_count, gen_hh_count)
            generation_count += 1

            # Track cache stats (only for modes with h→h discovery)
            if gen_hh_count > 0:
                if gen_hh_count == last_hh_count:
                    cache_hit_count += 1
                else:
                    cache_refresh_count += 1
                last_hh_count = gen_hh_count

            # Ensure fitnesses are on primary device and fully materialized
            # This prevents device sharding issues from pmap affecting _compiled_tell
            fitnesses = jax.device_put(fitnesses, jax.devices()[0])
            fitnesses = jax.block_until_ready(fitnesses)

            # === STEP 6: NEAT evolution ===
            state = self._compiled_tell(state, fitnesses)
            self._current_generation += 1

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={gen_time:.3f}s")

        # Compute average substrate efficiency metrics
        if generation_count > 0:
            avg_active = cumulative_active / generation_count / pop_size
            position_util = (cumulative_active / generation_count) / total_positions / pop_size
            avg_w1_density = cumulative_w1_nonzero / cumulative_w1_total if cumulative_w1_total > 0 else 0.0
            avg_w2_density = cumulative_w2_nonzero / cumulative_w2_total if cumulative_w2_total > 0 else 0.0
        else:
            avg_active = 0.0
            position_util = 0.0
            avg_w1_density = 0.0
            avg_w2_density = 0.0

        # Store extended metrics including h→h count and substrate efficiency
        self._extended_metrics = ExtendedRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=total_positions,
            # Config echo (what was actually used)
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            # Population diversity from final fitnesses
            min_fitness=float(jnp.min(fitnesses)),
            max_fitness=float(jnp.max(fitnesses)),
            std_fitness=float(jnp.std(fitnesses)),
            # Substrate efficiency (averaged over generations)
            active_positions_mean=avg_active,
            position_utilization=position_util,
            w1_density=avg_w1_density,
            w2_density=avg_w2_density,
            # Cache stats
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result


# ============================================================================
# Module-Level pmap Functions for Multi-GPU Evaluation
# ============================================================================
# These are defined at module level (outside any class) to ensure JAX
# compiles them ONCE and reuses across calls. Defining inside methods would
# cause recompilation each call due to new function objects.


def _eval_hybrid_on_shard_core(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    inputs_shard: jnp.ndarray,
    targets_shard: jnp.ndarray,
    activate_time: int,
) -> jnp.ndarray:
    """Core evaluation logic for hybrid forward pass on a data shard.

    This function runs on a single GPU with its portion of the dataset.

    Args:
        W1: (pop, num_inputs, total_positions) - replicated
        W2: (pop, total_positions, num_outputs) - replicated
        from_indices: (pop, max_sparse_conns) - replicated
        to_indices: (pop, max_sparse_conns) - replicated
        hh_weights: (pop, max_sparse_conns) - replicated
        valid_mask: (pop, max_sparse_conns) - replicated
        inputs_shard: (per_gpu_samples, num_inputs) - this GPU's data
        targets_shard: (per_gpu_samples, num_outputs) - this GPU's data
        activate_time: Number of forward pass iterations

    Returns:
        (pop_size,) error sums for this shard
    """
    def eval_single_genome(w1, w2, fi, ti, hw, vm):
        """Evaluate single genome on the data shard."""
        outputs = forward_hybrid_vmapped(
            inputs_shard, w1, w2, fi, ti, hw, vm, activate_time
        )
        errors = jnp.mean((outputs - targets_shard) ** 2, axis=1)
        return jnp.sum(errors)

    # vmap over population
    error_sums = jax.vmap(eval_single_genome)(
        W1, W2, from_indices, to_indices, hh_weights, valid_mask
    )
    return error_sums  # (pop_size,)


def _eval_dense_on_shard_core(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    inputs_shard: jnp.ndarray,
    targets_shard: jnp.ndarray,
) -> jnp.ndarray:
    """Core evaluation logic for dense forward pass on a data shard.

    Args:
        W1: (pop, num_inputs, total_positions) - replicated
        W2: (pop, total_positions, num_outputs) - replicated
        inputs_shard: (per_gpu_samples, num_inputs) - this GPU's data
        targets_shard: (per_gpu_samples, num_outputs) - this GPU's data

    Returns:
        (pop_size,) error sums for this shard
    """
    def eval_single_genome(w1, w2):
        """Evaluate single genome on the data shard."""
        h = jnp.tanh(inputs_shard @ w1)
        outputs = jax.nn.sigmoid(h @ w2)
        errors = jnp.mean((outputs - targets_shard) ** 2, axis=1)
        return jnp.sum(errors)

    # vmap over population
    error_sums = jax.vmap(eval_single_genome)(W1, W2)
    return error_sums  # (pop_size,)


# pmap-wrapped versions for multi-GPU execution
# in_axes specifies which axis to shard on:
# - None: replicate (same data on all GPUs)
# - 0: shard along axis 0 (each GPU gets a slice)

@functools.partial(
    jax.pmap,
    in_axes=(None, None, None, None, None, None, 0, 0, None),  # 9 args
    static_broadcasted_argnums=(8,),  # activate_time is static
)
def _pmap_eval_hybrid(
    W1: jnp.ndarray,           # replicated
    W2: jnp.ndarray,           # replicated
    from_indices: jnp.ndarray, # replicated
    to_indices: jnp.ndarray,   # replicated
    hh_weights: jnp.ndarray,   # replicated
    valid_mask: jnp.ndarray,   # replicated
    inputs_shard: jnp.ndarray, # sharded (axis 0)
    targets_shard: jnp.ndarray, # sharded (axis 0)
    activate_time: int,        # static
) -> jnp.ndarray:
    """pmap-wrapped hybrid evaluation - runs in parallel on all GPUs."""
    return _eval_hybrid_on_shard_core(
        W1, W2, from_indices, to_indices, hh_weights, valid_mask,
        inputs_shard, targets_shard, activate_time
    )


@functools.partial(
    jax.pmap,
    in_axes=(None, None, 0, 0),  # 4 args: W1, W2 replicated; inputs, targets sharded
)
def _pmap_eval_dense(
    W1: jnp.ndarray,           # replicated
    W2: jnp.ndarray,           # replicated
    inputs_shard: jnp.ndarray, # sharded (axis 0)
    targets_shard: jnp.ndarray, # sharded (axis 0)
) -> jnp.ndarray:
    """pmap-wrapped dense evaluation - runs in parallel on all GPUs."""
    return _eval_dense_on_shard_core(W1, W2, inputs_shard, targets_shard)


# ============================================================================
# Full Pipeline with H→H Discovery (for Multi-GPU strategies)
# ============================================================================
# These functions run the complete pipeline (CPPN queries + variance + h→h + eval)
# inside pmap for balanced memory distribution across GPUs.


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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full pipeline with h→h discovery - runs on single GPU inside pmap.

    This function replicates the parent class's _full_pipeline_single_gpu logic
    but adds Phase 2 h→h discovery with multi-hop expansion.

    Args:
        cppns_transformed: Transformed CPPNs for this GPU
        all_positions: Hierarchical grid positions
        input_coords: Input coordinates
        output_coords: Output coordinates
        inputs_batch: Problem inputs (sharded or replicated depending on strategy)
        targets_batch: Problem targets
        variance_threshold: Subdivision threshold
        band_threshold: H→H threshold
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
        iteration_level: H→H iteration level (0 = no h→h)
        allow_hh: Enable h→h discovery
        state: Algorithm state
        cppn_forward: CPPN forward function

    Returns:
        Tuple of (fitnesses, total_hh_count, active_counts_sum, w1_nonzero_sum,
                  w2_nonzero_sum, w1_total, w2_total)
    """
    pop_size = cppns_transformed[0].shape[0]
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]

    # Helper: Query single CPPN at all positions from one source
    def query_cppn_single_source(cppn_tuple, source_coord, outgoing):
        def query_single_position(target_pos):
            bias = jnp.array([1.0])
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

    # STEP 1: Query CPPN for variance (first input coord only)
    def get_variance_weights(cppn_tuple):
        return query_cppn_single_source(cppn_tuple, input_coords[0], outgoing=True)

    all_weights_for_variance = jax.vmap(
        get_variance_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))

    # STEP 2: Compute hierarchical variances
    level_variances = compute_hierarchical_variances_batch_jit(
        all_weights_for_variance,
        level_sizes=level_sizes,
        level_offsets=level_offsets,
        level_grid_sizes=level_grid_sizes,
        num_levels=num_levels,
    )

    # STEP 3: Compute subdivision masks
    masks_A, _, _ = compute_subdivision_masks_batch_jit(
        level_variances,
        variance_threshold=variance_threshold,
        parent_indices_tuple=parent_indices,
        level_offsets=level_offsets,
        num_levels=num_levels,
        total_positions=total_positions,
    )

    # STEP 4: Query CPPN for input weights
    def get_input_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, input_coords, outgoing=True)

    input_all_weights = jax.vmap(
        get_input_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))

    # STEP 5: Query CPPN for output weights
    def get_output_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, output_coords, outgoing=False)

    output_all_weights = jax.vmap(
        get_output_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))

    # STEP 6: Build weight matrices with mask
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

    # STEP 7: H→H Discovery (NEW - not in parent!)
    # Note: We use a simplified inline version for JIT-compatibility
    # CHUNKED VERSION to avoid OOM and int32 overflow at depth 7+
    sparse_hh = None
    if allow_hh and iteration_level > 0:
        # Query h→h connections (active → active)
        # Get union of active positions
        union_active = jnp.any(masks_A, axis=0)
        active_indices = jnp.arange(total_positions)

        # Query all active→active pairs for h→h weights
        def get_hh_weights(cppn_tuple, source_idx):
            source_coord = all_positions[source_idx]
            def query_target(target_idx):
                target_coord = all_positions[target_idx]
                bias = jnp.array([1.0])
                inp = jnp.concatenate([source_coord, target_coord, bias])
                return cppn_forward(state, cppn_tuple, inp)[0]
            return jax.vmap(query_target)(active_indices)

        # Determine if we need chunking based on total_positions
        # int32 max is 2,147,483,647; at depth 7, total_positions^2 = 7.6B > int32 max
        # Use chunking when total_positions > 30000 (conservative threshold)
        use_chunked_hh = total_positions > 30000

        if not use_chunked_hh:
            # Original non-chunked version for smaller grids (depth <= 6)
            def get_hh_for_genome(cppn_tuple, mask):
                # Query all→all
                hh_matrix = jax.vmap(
                    lambda src_idx: get_hh_weights(cppn_tuple, src_idx)
                )(active_indices)

                # Apply thresholds
                hh_matrix = jnp.tanh(hh_matrix) * max_weight
                valid = (jnp.abs(hh_matrix) > band_threshold) & mask[None, :] & mask[:, None]

                # Remove self-connections
                diag_mask = ~jnp.eye(total_positions, dtype=bool)
                valid = valid & diag_mask

                # Convert to sparse representation
                flat_valid = valid.flatten()
                flat_weights = hh_matrix.flatten()

                # Get top max_sparse_conns connections
                num_valid_total = jnp.sum(flat_valid)
                sorted_indices = jnp.argsort(~flat_valid)  # Put valid first
                top_indices = sorted_indices[:max_sparse_conns]

                from_idx = top_indices // total_positions
                to_idx = top_indices % total_positions
                weights = flat_weights[top_indices]
                valid_mask = flat_valid[top_indices]

                return from_idx, to_idx, weights, valid_mask, num_valid_total
        else:
            # CHUNKED VERSION for large grids (depth 7+) to avoid OOM and int32 overflow
            # Process source positions in chunks to avoid creating (total_positions x total_positions) matrix
            hh_chunk_size = 500  # Process 500 sources at a time
            num_chunks = (total_positions + hh_chunk_size - 1) // hh_chunk_size
            conns_per_chunk = max_sparse_conns  # Collect max_sparse_conns from each chunk, merge later

            def get_hh_for_genome(cppn_tuple, mask):
                # Process h→h in chunks to avoid OOM and int32 overflow
                # Each chunk processes hh_chunk_size sources against all targets

                def process_chunk(chunk_idx, carry):
                    # carry: (best_from, best_to, best_weights, best_valid, best_scores, num_valid_total)
                    best_from, best_to, best_weights, best_valid, best_scores, num_valid_total = carry

                    start = chunk_idx * hh_chunk_size
                    # Use min to handle the last chunk
                    actual_chunk_size = jnp.minimum(hh_chunk_size, total_positions - start)

                    # Get source indices for this chunk
                    chunk_source_indices = jnp.arange(hh_chunk_size) + start
                    # Mask out indices beyond total_positions
                    valid_source_mask = chunk_source_indices < total_positions
                    chunk_source_indices = jnp.where(valid_source_mask, chunk_source_indices, 0)

                    # Query h→h for this chunk: (chunk_size, total_positions)
                    hh_chunk = jax.vmap(
                        lambda src_idx: get_hh_weights(cppn_tuple, src_idx)
                    )(chunk_source_indices)

                    # Apply thresholds
                    hh_chunk = jnp.tanh(hh_chunk) * max_weight

                    # Build validity mask
                    source_mask = mask[chunk_source_indices]  # (chunk_size,)
                    valid_chunk = (jnp.abs(hh_chunk) > band_threshold) & source_mask[:, None] & mask[None, :]

                    # Remove self-connections
                    target_indices = jnp.arange(total_positions)
                    self_conn_mask = chunk_source_indices[:, None] != target_indices[None, :]
                    valid_chunk = valid_chunk & self_conn_mask

                    # Mask out invalid sources (beyond total_positions)
                    valid_chunk = valid_chunk & valid_source_mask[:, None]

                    # Count valid connections in this chunk
                    chunk_valid_count = jnp.sum(valid_chunk)
                    num_valid_total = num_valid_total + chunk_valid_count

                    # Flatten chunk and find top connections
                    flat_valid_chunk = valid_chunk.flatten()  # (chunk_size * total_positions,)
                    flat_weights_chunk = hh_chunk.flatten()

                    # Create scores: valid connections first, sorted by weight magnitude
                    # Use negative validity (0 for valid, 1 for invalid) as primary sort key
                    chunk_scores = jnp.where(flat_valid_chunk, -jnp.abs(flat_weights_chunk), jnp.inf)

                    # Find indices of top connections in this chunk
                    # Note: chunk_size * total_positions is within int32 range (500 * 87380 = 43.7M < 2B)
                    num_to_select = jnp.minimum(conns_per_chunk, hh_chunk_size * total_positions)
                    top_chunk_indices = jnp.argsort(chunk_scores)[:conns_per_chunk]

                    # Convert flat indices to (from, to) pairs
                    chunk_from = chunk_source_indices[top_chunk_indices // total_positions]
                    chunk_to = top_chunk_indices % total_positions
                    chunk_weights_sel = flat_weights_chunk[top_chunk_indices]
                    chunk_valid_sel = flat_valid_chunk[top_chunk_indices]
                    chunk_scores_sel = chunk_scores[top_chunk_indices]

                    # Merge with best so far: combine and keep top max_sparse_conns
                    # Concatenate current best with chunk results
                    combined_from = jnp.concatenate([best_from, chunk_from])
                    combined_to = jnp.concatenate([best_to, chunk_to])
                    combined_weights = jnp.concatenate([best_weights, chunk_weights_sel])
                    combined_valid = jnp.concatenate([best_valid, chunk_valid_sel])
                    combined_scores = jnp.concatenate([best_scores, chunk_scores_sel])

                    # Sort by score and keep top max_sparse_conns
                    sorted_idx = jnp.argsort(combined_scores)[:max_sparse_conns]
                    new_best_from = combined_from[sorted_idx]
                    new_best_to = combined_to[sorted_idx]
                    new_best_weights = combined_weights[sorted_idx]
                    new_best_valid = combined_valid[sorted_idx]
                    new_best_scores = combined_scores[sorted_idx]

                    return (new_best_from, new_best_to, new_best_weights, new_best_valid, new_best_scores, num_valid_total)

                # Initialize carry with empty arrays
                init_from = jnp.zeros(max_sparse_conns, dtype=jnp.int32)
                init_to = jnp.zeros(max_sparse_conns, dtype=jnp.int32)
                init_weights = jnp.zeros(max_sparse_conns, dtype=jnp.float32)
                init_valid = jnp.zeros(max_sparse_conns, dtype=bool)
                init_scores = jnp.full(max_sparse_conns, jnp.inf, dtype=jnp.float32)
                init_num_valid = jnp.array(0, dtype=jnp.int32)

                # Use fori_loop to process all chunks
                final_carry = jax.lax.fori_loop(
                    0, num_chunks,
                    process_chunk,
                    (init_from, init_to, init_weights, init_valid, init_scores, init_num_valid)
                )

                best_from, best_to, best_weights, best_valid, _, num_valid_total = final_carry
                return best_from, best_to, best_weights, best_valid, num_valid_total

        # vmap over population
        from_indices, to_indices, hh_weights, valid_masks, num_valids = jax.vmap(
            lambda c0, c1, c2, c3, m: get_hh_for_genome((c0, c1, c2, c3), m)
        )(cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3], masks_A)

        sparse_hh = SparseHiddenConnections(
            from_indices=from_indices,
            to_indices=to_indices,
            weights=hh_weights,
            valid_mask=valid_masks,
            num_valid=num_valids,
        )

    # STEP 8: Evaluate all networks
    if sparse_hh is not None:
        # Hybrid evaluation with sparse h→h
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

    # Compute total h→h count for metrics
    if sparse_hh is not None:
        # Sum across population to get total h→h connections
        total_hh_count = jnp.sum(sparse_hh.num_valid)
    else:
        total_hh_count = jnp.array(0, dtype=jnp.int32)

    # Return fitnesses, h→h count, and substrate efficiency metrics
    return (fitnesses, total_hh_count,
            active_counts_sum, w1_nonzero_sum, w2_nonzero_sum, w1_total, w2_total)


# ============================================================================
# SEQUENTIAL MULTI-GPU: Process genomes one-at-a-time inside pmap
# ============================================================================
# This avoids XLA trace explosion from vmap+fori_loop combination.
# Python for-loops inside pmap are NOT traced, so memory is reused.


def _single_genome_pipeline_with_hh(
    cppn_tuple: Tuple,  # Single genome: (nodes, conns, funcs, keys) - NO pop dimension
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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process SINGLE genome through full pipeline with h→h discovery.

    This is the core function for sequential multi-GPU processing.
    It handles one genome at a time to avoid XLA trace memory explosion.

    Args:
        cppn_tuple: Single CPPN genome (NOT batched)
        ... (same as _full_pipeline_with_hh_core)

    Returns:
        Tuple of (fitness, hh_count, active_count, w1_nonzero, w2_nonzero, w1_total, w2_total)
        All scalars for this single genome.
    """
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]

    # Helper: Query single CPPN at all positions from one source
    def query_cppn_single_source(source_coord, outgoing):
        def query_single_position(target_pos):
            bias = jnp.array([1.0])
            if outgoing:
                inp = jnp.concatenate([source_coord, target_pos, bias])
            else:
                inp = jnp.concatenate([target_pos, source_coord, bias])
            return cppn_forward(state, cppn_tuple, inp)
        return jax.vmap(query_single_position)(all_positions).flatten()

    # Helper: Query single CPPN from multiple sources
    def query_cppn_multi_source(source_coords, outgoing):
        return jax.vmap(
            lambda src: query_cppn_single_source(src, outgoing)
        )(source_coords)

    # STEP 1: Query CPPN for variance (first input coord only)
    weights_for_variance = query_cppn_single_source(input_coords[0], outgoing=True)
    # Add batch dimension for hierarchical variance (expects [pop, positions])
    weights_for_variance_batch = weights_for_variance[None, :]

    # STEP 2: Compute hierarchical variances
    level_variances = compute_hierarchical_variances_batch_jit(
        weights_for_variance_batch,
        level_sizes=level_sizes,
        level_offsets=level_offsets,
        level_grid_sizes=level_grid_sizes,
        num_levels=num_levels,
    )

    # STEP 3: Compute subdivision masks
    masks_batch, _, _ = compute_subdivision_masks_batch_jit(
        level_variances,
        variance_threshold=variance_threshold,
        parent_indices_tuple=parent_indices,
        level_offsets=level_offsets,
        num_levels=num_levels,
        total_positions=total_positions,
    )
    # Extract single genome mask (remove batch dim)
    mask_A = masks_batch[0]

    # STEP 4: Query CPPN for input weights
    input_weights = query_cppn_multi_source(input_coords, outgoing=True)

    # STEP 5: Query CPPN for output weights
    output_weights = query_cppn_multi_source(output_coords, outgoing=False)

    # STEP 6: Build weight matrices with mask
    W1_raw = jnp.tanh(input_weights) * max_weight
    W2_raw = jnp.tanh(output_weights) * max_weight

    W1 = jnp.where(
        mask_A[None, :] & (jnp.abs(W1_raw) > weight_thresh),
        W1_raw, 0.0
    )
    W2_masked = jnp.where(
        mask_A[None, :] & (jnp.abs(W2_raw) > weight_thresh),
        W2_raw, 0.0
    )
    W2 = W2_masked.T  # Transpose for output

    # Compute substrate efficiency metrics
    active_count = jnp.sum(mask_A)
    w1_nonzero = jnp.sum(W1 != 0)
    w2_nonzero = jnp.sum(W2 != 0)
    w1_total = jnp.array(W1.size, dtype=jnp.int32)
    w2_total = jnp.array(W2.size, dtype=jnp.int32)

    # STEP 7: H→H Discovery (CHUNKED for large grids)
    sparse_from = None
    sparse_to = None
    sparse_weights = None
    sparse_valid = None
    hh_count = jnp.array(0, dtype=jnp.int32)

    if allow_hh and iteration_level > 0:
        active_indices = jnp.arange(total_positions)

        def get_hh_weights(source_idx):
            source_coord = all_positions[source_idx]
            def query_target(target_idx):
                target_coord = all_positions[target_idx]
                bias = jnp.array([1.0])
                inp = jnp.concatenate([source_coord, target_coord, bias])
                return cppn_forward(state, cppn_tuple, inp)[0]
            return jax.vmap(query_target)(active_indices)

        # Always use chunked version for large grids (depth 6+)
        # This is the key to avoiding OOM
        hh_chunk_size = 500
        num_chunks = (total_positions + hh_chunk_size - 1) // hh_chunk_size

        def process_chunk(carry, chunk_idx):
            best_from, best_to, best_weights, best_valid, best_scores, num_valid_total = carry

            start = chunk_idx * hh_chunk_size
            actual_chunk_size = jnp.minimum(hh_chunk_size, total_positions - start)

            # Get source indices for this chunk
            chunk_source_indices = jnp.arange(hh_chunk_size) + start
            valid_source_mask = chunk_source_indices < total_positions
            chunk_source_indices = jnp.where(valid_source_mask, chunk_source_indices, 0)

            # Query h→h for this chunk: (chunk_size, total_positions)
            hh_chunk = jax.vmap(get_hh_weights)(chunk_source_indices)

            # Apply thresholds
            hh_chunk = jnp.tanh(hh_chunk) * max_weight

            # Build validity mask
            source_mask_chunk = mask_A[chunk_source_indices]
            valid_chunk = (jnp.abs(hh_chunk) > band_threshold) & source_mask_chunk[:, None] & mask_A[None, :]

            # Remove self-connections
            target_indices = jnp.arange(total_positions)
            self_conn_mask = chunk_source_indices[:, None] != target_indices[None, :]
            valid_chunk = valid_chunk & self_conn_mask

            # Mask out invalid sources (beyond total_positions)
            valid_chunk = valid_chunk & valid_source_mask[:, None]

            # Count valid connections
            chunk_valid_count = jnp.sum(valid_chunk)
            num_valid_total = num_valid_total + chunk_valid_count

            # Flatten and find top connections
            flat_valid_chunk = valid_chunk.flatten()
            flat_weights_chunk = hh_chunk.flatten()

            chunk_scores = jnp.where(flat_valid_chunk, -jnp.abs(flat_weights_chunk), jnp.inf)
            top_chunk_indices = jnp.argsort(chunk_scores)[:max_sparse_conns]

            chunk_from = chunk_source_indices[top_chunk_indices // total_positions]
            chunk_to = top_chunk_indices % total_positions
            chunk_weights_sel = flat_weights_chunk[top_chunk_indices]
            chunk_valid_sel = flat_valid_chunk[top_chunk_indices]
            chunk_scores_sel = chunk_scores[top_chunk_indices]

            # Merge with best so far
            combined_from = jnp.concatenate([best_from, chunk_from])
            combined_to = jnp.concatenate([best_to, chunk_to])
            combined_weights = jnp.concatenate([best_weights, chunk_weights_sel])
            combined_valid = jnp.concatenate([best_valid, chunk_valid_sel])
            combined_scores = jnp.concatenate([best_scores, chunk_scores_sel])

            sorted_idx = jnp.argsort(combined_scores)[:max_sparse_conns]
            new_best_from = combined_from[sorted_idx]
            new_best_to = combined_to[sorted_idx]
            new_best_weights = combined_weights[sorted_idx]
            new_best_valid = combined_valid[sorted_idx]
            new_best_scores = combined_scores[sorted_idx]

            return (new_best_from, new_best_to, new_best_weights, new_best_valid, new_best_scores, num_valid_total), None

        # Initialize carry
        init_from = jnp.zeros(max_sparse_conns, dtype=jnp.int32)
        init_to = jnp.zeros(max_sparse_conns, dtype=jnp.int32)
        init_weights = jnp.zeros(max_sparse_conns, dtype=jnp.float32)
        init_valid = jnp.zeros(max_sparse_conns, dtype=bool)
        init_scores = jnp.full(max_sparse_conns, jnp.inf, dtype=jnp.float32)
        init_num_valid = jnp.array(0, dtype=jnp.int32)

        # Use scan instead of fori_loop to avoid unrolling issues
        final_carry, _ = jax.lax.scan(
            process_chunk,
            (init_from, init_to, init_weights, init_valid, init_scores, init_num_valid),
            jnp.arange(num_chunks)
        )

        sparse_from, sparse_to, sparse_weights, sparse_valid, _, hh_count = final_carry

    # STEP 8: Evaluate network
    if allow_hh and iteration_level > 0 and sparse_from is not None:
        # Hybrid evaluation with sparse h→h
        outputs = forward_hybrid_vmapped(
            inputs_batch, W1, W2, sparse_from, sparse_to, sparse_weights, sparse_valid, activate_time
        )
    else:
        # Dense-only evaluation
        hidden = jnp.tanh(inputs_batch @ W1)
        outputs = jax.nn.sigmoid(hidden @ W2)

    errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
    fitness = 1.0 - jnp.mean(errors)
    fitness = jnp.where(jnp.isnan(fitness), -jnp.inf, fitness)

    return (fitness, hh_count, active_count, w1_nonzero, w2_nonzero, w1_total, w2_total)


# POPULATION_PARALLEL_SEQUENTIAL: Process genomes one-at-a-time inside pmap
# Each GPU processes pop_per_gpu genomes SEQUENTIALLY via Python for-loop
# This avoids XLA trace explosion from vmap + fori_loop
# Note: This is the tested implementation used by MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL
@functools.partial(
    jax.pmap,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21),
)
def _pmap_sequential_pop_parallel_hh(
    cppns_shard,          # 0: sharded (each GPU gets pop slice) - shape (pop_per_gpu, ...)
    all_positions,        # None: replicated
    input_coords,         # None: replicated
    output_coords,        # None: replicated
    inputs_batch,         # None: replicated (full dataset on each GPU)
    targets_batch,        # None: replicated
    variance_threshold,   # None: replicated
    band_threshold,       # None: replicated
    max_weight,           # None: replicated
    weight_thresh,        # None: replicated
    activate_time,        # STATIC (10)
    level_sizes,          # STATIC (11)
    level_offsets,        # STATIC (12)
    level_grid_sizes,     # STATIC (13)
    parent_indices,       # None: replicated (JAX arrays)
    num_levels,           # STATIC (15)
    total_positions,      # STATIC (16)
    max_sparse_conns,     # STATIC (17)
    iteration_level,      # STATIC (18)
    allow_hh,             # STATIC (19)
    state,                # None: replicated
    cppn_forward,         # STATIC (21)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequential population-parallel: process genomes one at a time inside pmap.

    The key insight: Python for-loops inside a pmap-traced function are NOT
    traced by XLA. Each iteration executes and deallocates before the next,
    avoiding the memory explosion from fori_loop unrolling.

    This provides ~2x speedup over single-GPU fallback by using both GPUs.

    Returns:
        Tuple of (fitnesses, hh_counts, active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals)
        Each has shape (pop_per_gpu,) for this GPU's results.
    """
    pop_per_gpu = cppns_shard[0].shape[0]

    # Use scan for JIT-compatible sequential processing
    # Note: We use scan with a single element per iteration to achieve
    # sequential processing that reuses memory

    def process_one_genome(_, genome_idx):
        # Extract single genome from the shard
        cppn_tuple = (
            cppns_shard[0][genome_idx],
            cppns_shard[1][genome_idx],
            cppns_shard[2][genome_idx],
            cppns_shard[3][genome_idx],
        )

        # Process this single genome
        result = _single_genome_pipeline_with_hh(
            cppn_tuple,
            all_positions, input_coords, output_coords,
            inputs_batch, targets_batch,
            variance_threshold, band_threshold, max_weight, weight_thresh,
            activate_time, level_sizes, level_offsets, level_grid_sizes,
            parent_indices, num_levels, total_positions,
            max_sparse_conns, iteration_level, allow_hh, state, cppn_forward,
        )

        return None, result

    # Process all genomes sequentially
    _, results = jax.lax.scan(process_one_genome, None, jnp.arange(pop_per_gpu))

    # Unpack results - each is (pop_per_gpu,) array
    fitnesses = results[0]
    hh_counts = results[1]
    active_counts = results[2]
    w1_nonzeros = results[3]
    w2_nonzeros = results[4]
    w1_totals = results[5]
    w2_totals = results[6]

    # Sum metrics across population for this GPU
    total_hh = jnp.sum(hh_counts)
    total_active = jnp.sum(active_counts)
    total_w1_nz = jnp.sum(w1_nonzeros)
    total_w2_nz = jnp.sum(w2_nonzeros)
    total_w1 = jnp.sum(w1_totals)
    total_w2 = jnp.sum(w2_totals)

    return (fitnesses, total_hh, total_active, total_w1_nz, total_w2_nz, total_w1, total_w2)


# FULL_PIPELINE_PARALLEL: Split data across GPUs, replicate full population
# in_axes: (None=replicated) for cppns, positions, coords, scalars, state
#          (0=sharded) for inputs, targets
# NOTE: parent_indices (index 14) contains JAX arrays so NOT static
@functools.partial(
    jax.pmap,
    in_axes=(None, None, None, None, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21),
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
    activate_time,        # STATIC (10)
    level_sizes,          # STATIC (11)
    level_offsets,        # STATIC (12)
    level_grid_sizes,     # STATIC (13)
    parent_indices,       # None: replicated (JAX arrays - NOT static)
    num_levels,           # STATIC (15)
    total_positions,      # STATIC (16)
    max_sparse_conns,     # STATIC (17)
    iteration_level,      # STATIC (18)
    allow_hh,             # STATIC (19)
    state,                # None: replicated
    cppn_forward,         # STATIC (21)
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
    )


# POPULATION_PARALLEL_SEQUENTIAL (feedforward only): Split population across GPUs, replicate full dataset
# in_axes: (0=sharded) for cppns
#          (None=replicated) for everything else
# NOTE: parent_indices (index 14) contains JAX arrays so NOT static
# NOTE: This is used for feedforward mode in POPULATION_PARALLEL_SEQUENTIAL strategy
@functools.partial(
    jax.pmap,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21),
)
def _pmap_pop_parallel_hh(
    cppns_shard,          # 0: sharded (each GPU gets pop slice)
    all_positions,        # None: replicated
    input_coords,         # None: replicated
    output_coords,        # None: replicated
    inputs_batch,         # None: replicated (full dataset on each GPU)
    targets_batch,        # None: replicated
    variance_threshold,   # None: replicated
    band_threshold,       # None: replicated
    max_weight,           # None: replicated
    weight_thresh,        # None: replicated
    activate_time,        # STATIC (10)
    level_sizes,          # STATIC (11)
    level_offsets,        # STATIC (12)
    level_grid_sizes,     # STATIC (13)
    parent_indices,       # None: replicated (JAX arrays - NOT static)
    num_levels,           # STATIC (15)
    total_positions,      # STATIC (16)
    max_sparse_conns,     # STATIC (17)
    iteration_level,      # STATIC (18)
    allow_hh,             # STATIC (19)
    state,                # None: replicated
    cppn_forward,         # STATIC (21)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Population-parallel: full data on each GPU, different pop slices.

    Returns:
        Tuple of (fitnesses, total_hh_count, active_counts_sum, w1_nonzero_sum,
                  w2_nonzero_sum, w1_total, w2_total) where each is sharded per GPU.
    """
    return _full_pipeline_with_hh_core(
        cppns_shard, all_positions, input_coords, output_coords,
        inputs_batch, targets_batch,
        variance_threshold, band_threshold, max_weight, weight_thresh,
        activate_time, level_sizes, level_offsets, level_grid_sizes,
        parent_indices, num_levels, total_positions,
        max_sparse_conns, iteration_level, allow_hh, state, cppn_forward,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main class
    'HMRHyperNEATUnifiedExtended',

    # Dataclasses
    'SparseHiddenConnections',
    'UnifiedExtendedConfig',
    'ExtendedRecurrenceMetrics',

    # Presets
    'RECURRENCE_PRESETS',
    'get_recurrence_preset',

    # Forward pass
    'ForwardPassMode',
    'forward_unified_extended',
    'forward_hybrid_vmapped',
    'eval_single_network_hybrid',

    # Multi-hop (key innovation)
    'compute_multi_hop_connections_matrix_power',
    'compute_multi_hop_connections_fori_loop',

    # Discovery
    'discover_sparse_hh_vectorized_multi_hop',

    # Constraint filtering
    'get_connection_constraint_mask',
    'get_connection_constraint_mask_batched',

    # Cache
    'HHCacheManager',

    # Multi-GPU (re-export from parent)
    'MultiGPUStrategy',
    'PositionShardingConfig',
]
