"""HMR-HyperNEAT with Multi-GPU Support.

This module extends HMR-HyperNEAT with multi-GPU parallelization for accelerating
deep-depth evolution. Uses full pipeline chunking for memory-efficient execution.

MAIN EXECUTION MODES
====================

1. SINGLE_GPU
   - Standard single-GPU execution
   - Use for debugging or single-GPU systems

2. FULL_PIPELINE_PARALLEL (RECOMMENDED for feedforward)
   - Full pipeline runs on each GPU: CPPN → variance → W1/W2 → eval
   - Dataset is SPLIT across GPUs, pipeline is REPLICATED
   - Works for ALL depths including depth 7+ with pop=1000 on 11 GiB GPUs
   - Peak memory: ~350 MB (vs 3-4 GB without chunking)
   - Speedup: 1.7x-4.3x over SINGLE_GPU

3. EVAL_ONLY_PARALLEL (RECOMMENDED for h→h modes)
   - Only evaluation parallelized, CPPN/h→h runs once on GPU 0
   - H→H caching enabled: 27s → 3ms after first generation
   - Dataset SHARDED, weights BROADCAST via native JAX
   - Best for recurrent modes where h→h caching saves significant time

4. CPPN_CHUNKED (FALLBACK)
   - Only chunks CPPN queries (NOT weight matrices)
   - Use as fallback if FULL_PIPELINE_PARALLEL has issues
   - Works for depths 1-6, OOMs at depth 7 with pop=1000

USAGE
=====

```python
from emr_hyperneat._hmr_frozen.hmrhyperneat_pipeline_chunking_multi_gpus import (
    HMRHyperNEATMultiGPU,
    MultiGPUStrategy,
)

# RECOMMENDED for feedforward: Full pipeline on each GPU
algo = HMRHyperNEATMultiGPU(strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL)

# RECOMMENDED for h→h modes: Only eval parallel, h→h cached
algo = HMRHyperNEATMultiGPU(strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL)

# FALLBACK: CPPN-only chunking (if FULL_PIPELINE_PARALLEL has issues)
algo = HMRHyperNEATMultiGPU(strategy=MultiGPUStrategy.CPPN_CHUNKED)

# Single GPU (for debugging)
algo = HMRHyperNEATMultiGPU(strategy=MultiGPUStrategy.SINGLE_GPU)

# IMPORTANT: Use run_until_threshold() for multi-GPU!
# run_generation() does NOT use multi-GPU sharding.
result = algo.run_until_threshold(state, problem, target_fitness=0.98, max_generations=100)
```

BENCHMARK RESULTS (December 2025, 3 trials)
===========================================

| Depth | Positions | SINGLE_GPU | FULL_PIPELINE | Speedup |
|-------|-----------|------------|---------------|---------|
| 1     | 20        | 49.44s     | 22.88s        | 2.16x   |
| 2     | 84        | 105.41s    | 24.34s        | 4.34x   |
| 3     | 340       | 65.66s     | 24.98s        | 2.64x   |
| 4     | 1,364     | 70.05s     | 28.94s        | 2.44x   |
| 5     | 5,460     | 89.71s     | 43.72s        | 2.06x   |
| 6     | 21,844    | 174.60s    | 104.57s       | 1.68x   |
| 7     | 87,380    | OOM        | 386.86s       | Works!  |

Hardware: 2x NVIDIA GPU (11 GiB each), JAX 0.4.38

LEGACY COMPATIBILITY
====================
For backward compatibility, the following aliases are supported:
- BASELINE → SINGLE_GPU
- MULTI_GPU → FULL_PIPELINE_PARALLEL
- DATA_PARALLEL → FULL_PIPELINE_PARALLEL
- PMAP_PARALLEL → EVAL_ONLY_PARALLEL
- POSITION_SHARDING_CHUNKED → CPPN_CHUNKED
- PIPELINE_CHUNKED → FULL_PIPELINE_PARALLEL

NOTE: Experimental strategies (ISLAND_MODEL, POPULATION_PMAP, etc.) are preserved
in hmrhyperneat_chunking_multi_gpus_experimental.py for reference.
"""

import functools
import threading
import time
import copy
import math
import os
import numpy as np
from typing import Any, Dict, Tuple, Set, List, Optional, NamedTuple, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from multiprocessing import Process, Queue, Event
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

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

from emr_hyperneat._compat.core.base_algorithm import BaseAlgorithm, AlgorithmMetrics
from emr_hyperneat._compat.utils.config_manager import ConfigManager
from emr_hyperneat._compat.adapters.tensorneat_adapter import TensorNEATAdapter

# Note: SparseHiddenConnections, discover_sparse_hh_vectorized_multi_hop, UnifiedExtendedConfig
# are imported lazily inside run_until_threshold_with_fixed_hh() to avoid circular imports


# ============================================================================
# Recurrence Configuration for H→H Caching
# ============================================================================

@dataclass
class RecurrenceConfig:
    """Configuration for recurrent network features with h→h caching.

    This dataclass controls which connection types are allowed during
    substrate discovery and how the forward pass handles recurrence.

    Connection Types:
    - Forward (y_source < y_target): Always allowed
    - Lateral (y_source == y_target): Same-layer connections
    - Backward (y_source > y_target): Feedback loops
    - Self-loop (x,y identical): Memory/state nodes

    Attributes:
        enabled: Master switch for recurrence features
        allow_hidden_to_hidden: Enable Phase 2 discovery (iteration_level > 0)
        allow_backward: Enable backward connections (y_source > y_target)
        allow_lateral: Enable same-layer connections (y_source == y_target)
        allow_self_loops: Enable self-connections (same x,y)
        iteration_level: Rounds of hidden→hidden discovery
        activate_time: Forward pass iterations for signal propagation
        max_connections: Max connections per substrate (for vmap padding)
    """
    enabled: bool = True
    allow_hidden_to_hidden: bool = True
    allow_backward: bool = True
    allow_lateral: bool = True
    allow_self_loops: bool = True
    iteration_level: int = 2
    activate_time: int = 5
    max_connections: int = 10000


# ============================================================================
# JAX-Compatible H→H Cache State for GPU-Resident Loop
# ============================================================================

@jax.tree_util.register_pytree_node_class
@dataclass
class CacheStateJAX:
    """JAX-traceable h→h cache state for use inside lax.while_loop.

    This dataclass provides dynamic h→h cache refresh capability within
    a GPU-resident while_loop. All fields are JAX arrays (no Python types)
    to ensure compatibility with JAX tracing.

    Key Design Decisions:
    - No Python ints/bools: Use JAX scalar arrays instead
    - No Optional types: Use is_valid flag instead of None checks
    - Pre-allocated arrays: Fixed shapes for all connections
    - Functional updates: Return new CacheStateJAX instead of mutation

    Attributes:
        # Cached connection data (from SparseHiddenConnections)
        from_indices: (pop, max_sparse_conns) - source position indices
        to_indices: (pop, max_sparse_conns) - target position indices
        weights: (pop, max_sparse_conns) - connection weights
        valid_mask: (pop, max_sparse_conns) - True for valid connections
        num_valid: (pop,) - number of valid connections per genome

        # Cached variance mask for change detection
        cached_variance_mask: (pop, total_positions) - masks_A when cached

        # Metadata (JAX scalars, not Python ints)
        last_refresh_gen: Scalar int32 - generation when last refreshed
        refresh_count: Scalar int32 - total number of refreshes
        is_valid: Scalar bool - True if cache contains valid data
    """
    # Connection data
    from_indices: jnp.ndarray
    to_indices: jnp.ndarray
    weights: jnp.ndarray
    valid_mask: jnp.ndarray
    num_valid: jnp.ndarray

    # Variance mask for change detection
    cached_variance_mask: jnp.ndarray

    # Metadata (JAX scalars)
    last_refresh_gen: jnp.ndarray
    refresh_count: jnp.ndarray
    is_valid: jnp.ndarray

    def tree_flatten(self):
        """Flatten for JAX pytree."""
        children = (
            self.from_indices,
            self.to_indices,
            self.weights,
            self.valid_mask,
            self.num_valid,
            self.cached_variance_mask,
            self.last_refresh_gen,
            self.refresh_count,
            self.is_valid,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX pytree."""
        return cls(*children)

    @staticmethod
    def create_empty(pop_size: int, max_sparse_conns: int, total_positions: int) -> 'CacheStateJAX':
        """Create empty cache state with pre-allocated arrays.

        Args:
            pop_size: Population size
            max_sparse_conns: Maximum sparse connections per genome
            total_positions: Total positions in hierarchical grid

        Returns:
            CacheStateJAX with zero-filled arrays and is_valid=False
        """
        return CacheStateJAX(
            from_indices=jnp.zeros((pop_size, max_sparse_conns), dtype=jnp.int32),
            to_indices=jnp.zeros((pop_size, max_sparse_conns), dtype=jnp.int32),
            weights=jnp.zeros((pop_size, max_sparse_conns), dtype=jnp.float32),
            valid_mask=jnp.zeros((pop_size, max_sparse_conns), dtype=jnp.bool_),
            num_valid=jnp.zeros((pop_size,), dtype=jnp.int32),
            cached_variance_mask=jnp.zeros((pop_size, total_positions), dtype=jnp.bool_),
            last_refresh_gen=jnp.array(-1, dtype=jnp.int32),
            refresh_count=jnp.array(0, dtype=jnp.int32),
            is_valid=jnp.array(False, dtype=jnp.bool_),
        )

    def update_from_sparse_hh(
        self,
        sparse_hh: Any,  # SparseHiddenConnections (imported lazily where needed)
        variance_mask: jnp.ndarray,
        generation: jnp.ndarray,
    ) -> 'CacheStateJAX':
        """Create new cache state with updated data (functional update).

        Args:
            sparse_hh: New sparse h→h connections
            variance_mask: Current masks_A
            generation: Current generation number

        Returns:
            New CacheStateJAX with updated data
        """
        return CacheStateJAX(
            from_indices=sparse_hh.from_indices,
            to_indices=sparse_hh.to_indices,
            weights=sparse_hh.weights,
            valid_mask=sparse_hh.valid_mask,
            num_valid=sparse_hh.num_valid,
            cached_variance_mask=variance_mask,
            last_refresh_gen=generation,
            refresh_count=self.refresh_count + 1,
            is_valid=jnp.array(True, dtype=jnp.bool_),
        )


def should_refresh_cache_jax(
    cache_state: CacheStateJAX,
    current_gen: jnp.ndarray,
    current_mask: jnp.ndarray,
    refresh_interval: int,
    mask_change_threshold: float,
) -> jnp.ndarray:
    """Pure JAX function to determine if cache refresh is needed.

    This function can be used inside jax.lax.while_loop because it:
    - Returns a JAX boolean scalar (not Python bool)
    - Uses only JAX operations (no Python conditionals)
    - Has no side effects

    Args:
        cache_state: Current cache state
        current_gen: Current generation (JAX scalar)
        current_mask: Current masks_A (pop, total_positions)
        refresh_interval: Refresh every N generations
        mask_change_threshold: Refresh if mask change ratio > this

    Returns:
        JAX boolean scalar indicating if refresh is needed
    """
    # Condition 1: Cache not valid yet
    not_valid = ~cache_state.is_valid

    # Condition 2: Time-based refresh
    gens_since = current_gen - cache_state.last_refresh_gen
    time_based = gens_since >= refresh_interval

    # Condition 3: Change-based refresh (variance mask changed significantly)
    mask_diff = jnp.abs(
        current_mask.astype(jnp.float32) -
        cache_state.cached_variance_mask.astype(jnp.float32)
    )
    change_ratio = jnp.mean(mask_diff)
    change_based = change_ratio > mask_change_threshold

    return not_valid | time_based | change_based


# ============================================================================
# Multi-GPU Strategy Selection
# ============================================================================

class MultiGPUStrategy(Enum):
    """Available GPU execution strategies.

    ============================================================================
    STRATEGY SELECTION GUIDE
    ============================================================================

    Use this decision tree to choose the right strategy:

        Dataset fits on single GPU?
        ├─ YES → SINGLE_GPU (fastest for single GPU)
        └─ NO  → Multiple GPUs available?
                 ├─ NO  → STREAMING (streams from CPU)
                 └─ YES → Dataset fits on combined GPU memory?
                          ├─ YES → FULL_PIPELINE_PARALLEL (default)
                          └─ NO  → STREAMING (streams from CPU)

    ============================================================================
    STRATEGY DETAILS
    ============================================================================

    SINGLE_GPU: Single GPU execution
        - Use for: Debugging, single-GPU systems, or small datasets
        - Memory: Full dataset + full population on one GPU

    FULL_PIPELINE_PARALLEL: Full pipeline on each GPU (RECOMMENDED DEFAULT)
        - Use for: Most multi-GPU scenarios, feedforward mode
        - How it works: Each GPU runs FULL pipeline (CPPN → variance → W1/W2 → eval)
        - Dataset is SPLIT across GPUs, pipeline is REPLICATED
        - Memory: Each GPU holds (dataset/num_gpus) + full population
        - Benefit: 2x effective GPU memory for datasets
        - Performance: Same speed as population-parallel (~1600ms/gen)
        - Note: For recurrent (h→h) modes, prefer EVAL_ONLY_PARALLEL for caching

    POPULATION_PARALLEL_SEQUENTIAL: Population-Parallel with Sequential h→h
        - Use for: Very large populations with small datasets
        - How it works: Splits POPULATION across GPUs, replicates dataset
        - H→H processing: SEQUENTIAL (one GPU at a time) to avoid JIT cache errors
        - Memory: Each GPU holds full dataset + (population/num_gpus)
        - When to prefer: population_size >> dataset_size (memory-wise)
        - Trade-off: 2-3x slower for h→h modes vs single-GPU, but 0 errors
        - Note: Feedforward mode is 2.6x faster (true parallel)

    POPULATION_PARALLEL: not implemented in this release
        - Would provide true parallel h→h processing across GPUs
        - Raises NotImplementedError
        - JAX JIT cache is globally shared across threads, causing
          device placement errors when processing h→h in parallel

    POPULATION_PARALLEL_PROCESS: ProcessPoolExecutor-based true parallel h→h
        - Use for: Large populations with h→h processing that need true parallelism
        - How it works: Spawns separate Python processes for each GPU
        - Each process has isolated JAX runtime and JIT cache
        - Memory: Same as POPULATION_PARALLEL_SEQUENTIAL
        - Benefit: True parallel execution without cross-device errors
        - Trade-off: Higher IPC overhead (data serialization), JIT warmup per process
        - Note: Best for computationally expensive h→h discovery (depth >= 4)

    PERSISTENT_PARALLEL: Persistent worker processes with FULL h→h pipeline parallelized
        - Use for: Maximum parallelization of h→h processing (CPPN queries + h→h discovery + evaluation)
        - How it works: Spawns persistent workers ONCE at init, workers reconstruct algorithm internally
        - Each worker has: Own JAX runtime, own cppn_forward JIT function, own JIT cache
        - Per generation: Only sends CPPN genome arrays (lightweight numpy), workers do full pipeline
        - Memory: Each GPU holds full dataset + (population/num_gpus)
        - Benefit: ~2x speedup vs sequential, amortized worker spawn overhead
        - Trade-off: ~3s initial worker spawn time, higher complexity
        - Best for: Long runs (100+ generations) where spawn overhead is amortized

    EVAL_ONLY_PARALLEL: Only evaluation parallelized (RECOMMENDED for h→h modes)
        - Use for: Recurrent modes with h→h caching, fastest multi-GPU execution
        - How it works:
          1. GPU 0 runs: CPPN queries → variance → H→H discovery WITH CACHING
          2. W1, W2, h→h BROADCAST to all GPUs via native JAX (no serialization)
          3. Dataset SHARDED across GPUs, weights REPLICATED
          4. ONLY evaluation runs in parallel via pmap
        - Memory: Each GPU holds (dataset/num_gpus) + full W1/W2/h→h
        - Benefit: ZERO IPC overhead + H→H caching (27s → 3ms after gen 0)
        - Trade-off: Requires pure JAX evaluation kernel (no Python branching)
        - Best for: Recurrent (h→h) modes where caching saves significant time
        - Note: For feedforward, FULL_PIPELINE_PARALLEL is equivalent

    STREAMING: CPU-to-GPU streaming for huge datasets
        - Use for: Datasets that exceed total GPU memory
        - How it works: Streams dataset chunks from CPU to GPU
        - Memory: Only chunk_size samples on GPU at a time
        - Trade-off: 2-10x slower due to CPU-GPU transfers
        - Benefit: Unlimited dataset size (limited only by CPU RAM)

    CPPN_CHUNKED: Legacy strategy (fallback)
        - Use as fallback if MULTI_GPU has issues
        - Only chunks CPPN queries (NOT weight matrices)

    ============================================================================
    PERFORMANCE COMPARISON (pop=300, depth=3, 2 GPUs)
    ============================================================================

    | Dataset  | FULL_PIPELINE | POP_PARALLEL | STREAMING-500 | STREAMING-1000 |
    |----------|---------------|--------------|---------------|----------------|
    | 2K       | ~1600ms       | ~1500ms      | ~5000ms       | ~2800ms        |
    | 5K       | ~1700ms       | ~1600ms      | ~11000ms      | ~5500ms        |
    | 10K      | ~1800ms       | ~1800ms      | ~22000ms      | ~11000ms       |

    Key insight: FULL_PIPELINE_PARALLEL ≈ POPULATION_PARALLEL in speed, but
    FULL_PIPELINE_PARALLEL provides 2x dataset memory capacity.

    For h→h modes: EVAL_ONLY_PARALLEL is preferred due to h→h caching.

    ============================================================================
    """
    # Production strategies
    SINGLE_GPU = "single_gpu"
    SINGLE_GPU_PYTHON_LOOP = "single_gpu_python_loop"  # Debugging: per-generation Python loop
    FULL_PIPELINE_PARALLEL = "full_pipeline_parallel"  # Full pipeline on each GPU (RECOMMENDED)
    POPULATION_PARALLEL_SEQUENTIAL = "population_parallel_sequential"  # Sequential h→h (TESTED, 0 errors)
    POPULATION_PARALLEL = "population_parallel"  # True parallel h→h (NOT IMPLEMENTED)
    POPULATION_PARALLEL_PROCESS = "population_parallel_process"  # ProcessPoolExecutor for true parallel h→h
    PERSISTENT_PARALLEL = "persistent_parallel"  # Persistent workers with full h→h pipeline parallelized
    EVAL_ONLY_PARALLEL = "eval_only_parallel"  # Only eval parallelized, h→h cached on GPU 0 (RECOMMENDED for h→h)
    STREAMING = "streaming"  # Streams from CPU for huge datasets
    CPPN_CHUNKED = "cppn_chunked"  # Legacy: only chunks CPPN queries

    # Legacy aliases (for backward compatibility)
    BASELINE = "baseline"  # Alias for SINGLE_GPU
    POSITION_SHARDING = "position"  # Legacy - use CPPN_CHUNKED
    POSITION_SHARDING_CHUNKED = "position_chunked"  # Alias for CPPN_CHUNKED
    PIPELINE_CHUNKED = "pipeline_chunked"  # Alias for FULL_PIPELINE_PARALLEL
    DATA_PARALLEL = "data_parallel"  # Alias for FULL_PIPELINE_PARALLEL
    MULTI_GPU = "multi_gpu"  # Alias for FULL_PIPELINE_PARALLEL (old name)
    PMAP_PARALLEL = "pmap_parallel"  # Alias for EVAL_ONLY_PARALLEL (old name)

    # Experimental strategies (kept for reference, may not work correctly)
    ISLAND_MODEL = "island"
    ISLAND_MODEL_V2 = "island_v2"
    SHARD_MAP = "shard_map"
    HYBRID = "hybrid"
    POPULATION_PMAP = "pmap"
    POSITION_PMAP = "position_pmap"


# ============================================================================
# Multi-GPU Configuration Classes
# ============================================================================

@dataclass
class PositionShardingConfig:
    """Configuration for position-level multi-GPU sharding.

    This shards the position dimension across GPUs, so each GPU processes
    a subset of positions for ALL candidates in the population.

    Example with 2 GPUs and 21,844 positions:
        GPU 0: positions 0-10,921 for all 1000 candidates
        GPU 1: positions 10,922-21,843 for all 1000 candidates

    Attributes:
        num_devices: Number of GPUs to use (None = auto-detect)
        mesh: JAX device mesh for sharding
        axis_name: Name of the sharding axis
    """
    num_devices: Optional[int] = None
    mesh: Any = field(default=None, repr=False)
    axis_name: str = "positions"

    def __post_init__(self):
        """Initialize mesh and device configuration."""
        if not SHARD_MAP_AVAILABLE:
            raise RuntimeError(
                "shard_map not available. Requires JAX 0.4.1+ with "
                "jax.experimental.shard_map support."
            )

        self.devices = jax.devices()
        if self.num_devices is None:
            self.num_devices = len(self.devices)
        else:
            self.num_devices = min(self.num_devices, len(self.devices))

        if self.num_devices < 2:
            raise ValueError(
                f"Position sharding requires at least 2 devices, got {self.num_devices}"
            )

        # Create device mesh
        device_array = mesh_utils.create_device_mesh((self.num_devices,))
        self.mesh = Mesh(device_array, axis_names=(self.axis_name,))

    def shard_positions(
        self,
        positions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, int]:
        """Shard positions array with padding if needed.

        Args:
            positions: Array of shape (num_positions, 2)

        Returns:
            Tuple of (padded_positions, pad_size)
            padded_positions has shape (padded_num_positions, 2) where
            padded_num_positions is divisible by num_devices
        """
        num_positions = positions.shape[0]

        # Pad to make divisible by num_devices
        remainder = num_positions % self.num_devices
        if remainder != 0:
            pad_size = self.num_devices - remainder
            padding = jnp.zeros((pad_size, positions.shape[1]), dtype=positions.dtype)
            padded_positions = jnp.concatenate([positions, padding], axis=0)
        else:
            pad_size = 0
            padded_positions = positions

        return padded_positions, pad_size

    def pad_positions(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Pad positions array to be divisible by num_devices.

        This is a convenience wrapper around shard_positions that only returns
        the padded array, storing the pad_size internally for later unpadding.

        Args:
            positions: Array of shape (num_positions, 2)

        Returns:
            Padded positions array
        """
        padded, self._last_pad_size = self.shard_positions(positions)
        self._last_original_size = positions.shape[0]
        return padded

    def unpad_results(
        self,
        results: jnp.ndarray,
        original_size: Optional[int] = None
    ) -> jnp.ndarray:
        """Remove padding from results array.

        Args:
            results: Array with padded position dimension (last axis)
            original_size: Original number of positions (if None, uses stored value)

        Returns:
            Results with padding removed from position dimension
        """
        if original_size is None:
            original_size = getattr(self, '_last_original_size', results.shape[-1])

        if original_size < results.shape[-1]:
            return results[..., :original_size]
        return results

    def get_positions_per_device(self, padded_num_positions: int) -> int:
        """Get number of positions per device after sharding.

        Args:
            padded_num_positions: Total number of padded positions

        Returns:
            Positions per device (padded_num_positions // num_devices)
        """
        return padded_num_positions // self.num_devices


@dataclass
class IslandConfig:
    """Configuration for a single island in the Island Model.

    Attributes:
        island_id: Unique identifier for this island
        gpu_id: GPU device ID to run on
        population_size: Number of candidates in this island's population
        seed: Random seed for reproducibility
        migration_rate: Fraction of best genomes to migrate (default 5%)
    """
    island_id: int
    gpu_id: int
    population_size: int
    seed: int
    migration_rate: float = 0.05


@dataclass
class MigrationPacket:
    """Data packet for genome migration between islands.

    Attributes:
        source_island: ID of the island sending genomes
        generation: Generation when migration occurred
        best_fitness: Fitness of the best migrating genome
        genome_nodes: Node genes as numpy array (for pickling)
        genome_conns: Connection genes as numpy array (for pickling)
    """
    source_island: int
    generation: int
    best_fitness: float
    genome_nodes: np.ndarray
    genome_conns: np.ndarray


@dataclass
class IslandModelConfig:
    """Configuration for Island Model multi-GPU execution.

    Attributes:
        num_islands: Number of islands (default = num_devices)
        migration_interval: Generations between migrations (default 20)
        migration_rate: Fraction of population to migrate (default 5%)
        stop_on_first_solution: Stop all islands when one finds solution
    """
    num_islands: Optional[int] = None
    migration_interval: int = 20
    migration_rate: float = 0.05
    stop_on_first_solution: bool = True

    def __post_init__(self):
        """Initialize island configuration."""
        num_devices = len(jax.devices())
        if self.num_islands is None:
            self.num_islands = num_devices
        else:
            self.num_islands = min(self.num_islands, num_devices)

        if self.num_islands < 2:
            raise ValueError(
                f"Island model requires at least 2 devices, got {self.num_islands}"
            )


@dataclass
class IslandModelConfigV2:
    """Configuration for Island Model V2 with optimizations.

    Key improvements over V1:
    - Working migration: migrants are actually injected into populations
    - Population splitting: total_pop is divided across islands
    - Configurable topology: ring, star, or random migration patterns

    Attributes:
        num_islands: Number of islands (default = num_devices)
        migration_interval: Generations between migrations (default 10)
        migration_rate: Fraction of population to migrate (default 5%)
        stop_on_first_solution: Stop all islands when one finds solution
        topology: Migration topology ('ring', 'star', 'random')
        split_population: If True, divide total population across islands
    """
    num_islands: Optional[int] = None
    migration_interval: int = 10  # More frequent than V1
    migration_rate: float = 0.05
    stop_on_first_solution: bool = True
    topology: str = "ring"  # ring, star, random
    split_population: bool = True  # New: split total pop across islands

    def __post_init__(self):
        """Initialize island configuration."""
        num_devices = len(jax.devices())
        if self.num_islands is None:
            self.num_islands = num_devices
        else:
            self.num_islands = min(self.num_islands, num_devices)

        if self.num_islands < 2:
            raise ValueError(
                f"Island model requires at least 2 devices, got {self.num_islands}"
            )

        if self.topology not in ("ring", "star", "random"):
            raise ValueError(
                f"Unknown topology: {self.topology}. Use 'ring', 'star', or 'random'"
            )


@dataclass
class HybridShardingConfig:
    """Configuration for hybrid position + population sharding.

    Creates a 2D mesh where:
    - Axis 0 shards positions
    - Axis 1 shards population

    Example with 4 GPUs, 21844 positions, 1000 candidates:
        GPU 0: positions 0-10921, candidates 0-499
        GPU 1: positions 0-10921, candidates 500-999
        GPU 2: positions 10922-21843, candidates 0-499
        GPU 3: positions 10922-21843, candidates 500-999
    """
    num_devices: Optional[int] = None
    position_devices: int = 2
    population_devices: int = 1
    mesh: Any = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize 2D mesh configuration."""
        if not SHARD_MAP_AVAILABLE:
            raise RuntimeError(
                "shard_map not available for hybrid sharding."
            )

        num_available = len(jax.devices())
        total_needed = self.position_devices * self.population_devices

        if total_needed > num_available:
            raise ValueError(
                f"Hybrid sharding needs {total_needed} devices "
                f"({self.position_devices} x {self.population_devices}), "
                f"but only {num_available} available."
            )

        # Create 2D device mesh
        device_array = mesh_utils.create_device_mesh(
            (self.position_devices, self.population_devices)
        )
        self.mesh = Mesh(device_array, axis_names=('positions', 'population'))


@dataclass
class PopulationPmapConfig:
    """Configuration for population-level pmap parallelism.

    This shards the POPULATION across GPUs using jax.pmap - the standard
    JAX approach for data parallelism. Each GPU processes its slice of the
    population independently, with no inter-GPU communication until the
    final gather.

    Example with 2 GPUs and 1000 candidates:
        GPU 0: candidates 0-499, ALL positions
        GPU 1: candidates 500-999, ALL positions

    This is more efficient than position sharding because:
    1. pmap has lower overhead than shard_map
    2. No need for check_rep=False workarounds
    3. Each GPU runs completely independent computation
    4. Memory usage scales better (each GPU only stores half population)

    Attributes:
        num_devices: Number of GPUs to use (None = auto-detect)
        devices: List of JAX devices
        pmap_axis_name: Name of the pmap axis for collective operations
    """
    num_devices: Optional[int] = None
    pmap_axis_name: str = "pop"
    devices: List[Any] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize device configuration."""
        all_devices = jax.devices()
        if self.num_devices is None:
            self.num_devices = len(all_devices)
        else:
            self.num_devices = min(self.num_devices, len(all_devices))

        if self.num_devices < 2:
            raise ValueError(
                f"Population pmap requires at least 2 devices, got {self.num_devices}"
            )

        self.devices = all_devices[:self.num_devices]

    def shard_population(
        self, pop_arrays: Tuple[jnp.ndarray, ...]
    ) -> Tuple[jnp.ndarray, ...]:
        """Shard population arrays across devices.

        Args:
            pop_arrays: Tuple of arrays, each (pop_size, ...)

        Returns:
            Tuple of sharded arrays, each (num_devices, pop_per_device, ...)
        """
        sharded = []
        for arr in pop_arrays:
            pop_size = arr.shape[0]
            pop_per_device = pop_size // self.num_devices
            # Reshape: (pop_size, ...) -> (num_devices, pop_per_device, ...)
            new_shape = (self.num_devices, pop_per_device) + arr.shape[1:]
            sharded.append(arr.reshape(new_shape))
        return tuple(sharded)

    def gather_results(
        self, sharded_results: jnp.ndarray
    ) -> jnp.ndarray:
        """Gather sharded results back to single array.

        Args:
            sharded_results: Array of shape (num_devices, pop_per_device, ...)

        Returns:
            Array of shape (pop_size, ...)
        """
        # Reshape: (num_devices, pop_per_device, ...) -> (pop_size, ...)
        return sharded_results.reshape(-1, *sharded_results.shape[2:])


# ============================================================================
# Device Sync Tracing (for debugging CPU<->GPU transfers)
# ============================================================================
#
# Enable tracing to identify CPU<->GPU synchronization points.
# Set _TRACE_DEVICE_SYNC = True to log all device_get calls.
# Set _TRACE_DEVICE_SYNC_VERBOSE = True to include stack traces.
#
# Usage:
#   1. Set _TRACE_DEVICE_SYNC = True
#   2. Run a benchmark
#   3. Check logs for [DEVICE_SYNC] messages
#   4. Use traced_device_get() instead of direct np.asarray/float() calls

_TRACE_DEVICE_SYNC = False
_TRACE_DEVICE_SYNC_VERBOSE = False  # Include stack traces
_SYNC_COUNTER = {'count': 0, 'generation': 0}  # Track syncs per generation

# Per-step timing instrumentation
_TRACE_STEP_TIMING = False  # Enable to log time breakdown per generation step
_STEP_TIMINGS = {}  # Accumulated step timings across generations

# DEPRECATED: Module-level constant kept for backwards compatibility only.
# Use hmr_hyperneat.sparse_forward_threshold in config instead.
# Values: -1 = disable sparse, 0 = always sparse (default), >0 = threshold.
_SPARSE_FORWARD_THRESHOLD = 0  # Not used - see self.sparse_forward_threshold


def traced_device_get(value: Any, name: str = "unknown") -> Any:
    """Get value from device with optional tracing.

    Use this instead of direct np.asarray(), float(), or jax.device_get()
    calls to enable sync point tracking.

    Args:
        value: JAX array or scalar to transfer to CPU
        name: Descriptive name for logging (e.g., "fitness_mean")

    Returns:
        CPU value (numpy array or Python scalar)

    Example:
        # Instead of: result = float(jnp.mean(fitnesses))
        # Use: result = traced_device_get(jnp.mean(fitnesses), "fitness_mean")
    """
    if _TRACE_DEVICE_SYNC:
        _SYNC_COUNTER['count'] += 1
        import traceback
        sync_num = _SYNC_COUNTER['count']
        gen = _SYNC_COUNTER['generation']
        print(f"[DEVICE_SYNC #{sync_num}] gen={gen} name={name}", flush=True)
        if _TRACE_DEVICE_SYNC_VERBOSE:
            traceback.print_stack(limit=6)

    # Perform the actual transfer
    if isinstance(value, jnp.ndarray):
        return np.asarray(value)
    else:
        # Scalar - use float() or int() depending on type
        return float(value) if hasattr(value, '__float__') else value


def reset_sync_counter(generation: int = 0):
    """Reset sync counter for a new generation."""
    _SYNC_COUNTER['count'] = 0
    _SYNC_COUNTER['generation'] = generation


def get_sync_count() -> int:
    """Get current sync count for this generation."""
    return _SYNC_COUNTER['count']


def reset_step_timings():
    """Reset accumulated step timings."""
    global _STEP_TIMINGS
    _STEP_TIMINGS = {}


def get_step_timings() -> dict:
    """Get accumulated step timings."""
    return _STEP_TIMINGS.copy()


# ============================================================================
# Platform-Specific Matrix Multiplication (CPU Fallback for CUDA)
# ============================================================================
# Set to True to enable CPU matmul fallback on NVIDIA CUDA backends.
# This is a workaround for CUDA library version mismatches that cause SIGSEGV.
# Fix: Run setup/platform/install_jax_cuda.sh to install matching nvidia-* packages.
_ENABLE_CPU_MATMUL_FALLBACK = False


def _detect_backend_type() -> str:
    """Detect JAX backend: 'cuda', 'metal', or 'cpu'."""
    try:
        for dev in jax.devices():
            dev_str = str(dev).lower()
            if 'cuda' in dev_str:
                return 'cuda'
            if 'metal' in dev_str:
                return 'metal'
        return 'cpu'
    except Exception:
        return 'cpu'


_BACKEND_TYPE = _detect_backend_type()
_USE_CPU_MATMUL_FALLBACK = (_BACKEND_TYPE == 'cuda') and _ENABLE_CPU_MATMUL_FALLBACK


def _cpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CPU matmul for jax.pure_callback."""
    return a @ b


def safe_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Matrix multiplication with optional CPU fallback for CUDA.

    By default uses native JAX matmul. If _ENABLE_CPU_MATMUL_FALLBACK is True
    and running on CUDA, falls back to CPU to avoid cuBLAS crashes from
    CUDA library version mismatches.

    Compatible with jax.vmap and jax.jit.
    """
    if not _USE_CPU_MATMUL_FALLBACK:
        return a @ b

    # CPU fallback via jax.pure_callback (slower but avoids CUDA crashes)
    result_shape = jax.ShapeDtypeStruct(
        shape=a.shape[:-1] + (b.shape[-1],),
        dtype=a.dtype
    )
    return jax.pure_callback(_cpu_matmul, result_shape, a, b)


def print_step_timing_summary(num_generations: int = 1):
    """Print summary of step timings.

    Args:
        num_generations: Number of generations to average over
    """
    if not _STEP_TIMINGS:
        print("No step timings recorded. Enable _TRACE_STEP_TIMING first.")
        return

    print("\n" + "=" * 70)
    print("STEP TIMING BREAKDOWN (per generation)")
    print("=" * 70)
    total = 0.0
    for step, total_time in sorted(_STEP_TIMINGS.items()):
        avg_ms = (total_time / num_generations) * 1000
        total += avg_ms
        print(f"  {step:30s}: {avg_ms:8.1f}ms")
    print("-" * 70)
    print(f"  {'TOTAL':30s}: {total:8.1f}ms")
    print("=" * 70)


# ============================================================================
# Dense Quadtree Data Structures
# ============================================================================

# Pre-computed child offsets (quadrant positions relative to parent)
CHILD_OFFSETS = jnp.array([
    [-0.5, -0.5],  # Child 0: bottom-left
    [-0.5, +0.5],  # Child 1: top-left
    [+0.5, +0.5],  # Child 2: top-right
    [+0.5, -0.5],  # Child 3: bottom-right
], dtype=jnp.float32)

# Neighbor offsets for band detection (left, right, top, bottom)
NEIGHBOR_OFFSETS = jnp.array([
    [-1.0, 0.0],   # left
    [+1.0, 0.0],   # right
    [0.0, -1.0],   # top (y decreases)
    [0.0, +1.0],   # bottom (y increases)
], dtype=jnp.float32)


class DenseQuadtreeStructure(NamedTuple):
    """Pre-computed quadtree spatial structure.

    This is computed ONCE at initialization and reused for all CPPNs.
    Only the weights change per CPPN; positions/widths are fixed.

    For max_depth=D:
    - Total nodes: sum(4^i for i=0..D) = (4^(D+1) - 1) / 3
    - Leaf nodes: 4^D

    Memory per source coordinate:
    - depth=3: 85 nodes, ~1.4 KB
    - depth=5: 1,365 nodes, ~22 KB
    - depth=7: 21,845 nodes, ~350 KB
    """
    # Node positions: (num_nodes, 2) - (x, y) centers
    positions: jnp.ndarray

    # Node widths: (num_nodes,) - half-width at each level
    widths: jnp.ndarray

    # Node levels: (num_nodes,) - depth level (0=root)
    levels: jnp.ndarray

    # Level offsets: (max_depth+2,) - start index of each level
    # level_offsets[d] = start of level d, level_offsets[d+1] = end of level d
    level_offsets: jnp.ndarray

    # Leaf mask: (num_nodes,) - True for leaf nodes
    leaf_mask: jnp.ndarray

    # Leaf positions only (for efficiency in band detection)
    leaf_positions: jnp.ndarray  # (num_leaves, 2)

    # Leaf widths (parent widths for band detection)
    leaf_widths: jnp.ndarray  # (num_leaves,)

    # Number of total nodes and leaves
    num_nodes: int
    num_leaves: int
    max_depth: int


def compute_num_nodes(max_depth: int) -> Tuple[int, int]:
    """Compute total nodes and leaf nodes for a complete quadtree.

    Args:
        max_depth: Maximum depth (0 = root only)

    Returns:
        (total_nodes, num_leaves)
    """
    # Total nodes = 1 + 4 + 16 + ... + 4^max_depth = (4^(max_depth+1) - 1) / 3
    total_nodes = (4 ** (max_depth + 1) - 1) // 3
    num_leaves = 4 ** max_depth
    return total_nodes, num_leaves


def compute_level_offsets(max_depth: int) -> jnp.ndarray:
    """Compute the starting index of each level in the flattened node array.

    Level 0: 1 node (root)
    Level 1: 4 nodes
    Level 2: 16 nodes
    ...
    Level d: 4^d nodes

    Returns:
        level_offsets: array of shape (max_depth+2,)
        level_offsets[d] = first index of level d
        level_offsets[max_depth+1] = total nodes (for bounds checking)
    """
    offsets = [0]
    for d in range(max_depth + 1):
        offsets.append(offsets[-1] + 4 ** d)
    return jnp.array(offsets, dtype=jnp.int32)


def precompute_quadtree_structure(max_depth: int) -> DenseQuadtreeStructure:
    """Pre-compute all quadtree node positions and widths.

    This creates the complete spatial structure for a quadtree of given depth.
    Called ONCE at algorithm initialization; reused for all CPPNs.

    The quadtree covers the space [-1, 1] x [-1, 1] with root at (0, 0).

    Args:
        max_depth: Maximum depth (0 = root only, 1 = root + 4 children, etc.)

    Returns:
        DenseQuadtreeStructure with all pre-computed positions and widths
    """
    num_nodes, num_leaves = compute_num_nodes(max_depth)
    level_offsets = compute_level_offsets(max_depth)

    # Initialize arrays
    positions = np.zeros((num_nodes, 2), dtype=np.float32)
    widths = np.zeros(num_nodes, dtype=np.float32)
    levels = np.zeros(num_nodes, dtype=np.int32)

    # Root node: center (0, 0), width 1.0 (covers [-1, 1])
    positions[0] = [0.0, 0.0]
    widths[0] = 1.0
    levels[0] = 0

    # Build level by level
    child_offsets_np = np.array(CHILD_OFFSETS)

    for depth in range(max_depth):
        level_start = int(level_offsets[depth])
        level_end = int(level_offsets[depth + 1])
        next_level_start = int(level_offsets[depth + 1])

        for parent_idx in range(level_start, level_end):
            parent_pos = positions[parent_idx]
            parent_width = widths[parent_idx]
            child_width = parent_width * 0.5

            # Compute 4 children positions
            for c in range(4):
                child_idx = next_level_start + (parent_idx - level_start) * 4 + c
                child_pos = parent_pos + child_offsets_np[c] * parent_width
                positions[child_idx] = child_pos
                widths[child_idx] = child_width
                levels[child_idx] = depth + 1

    # Create leaf mask (nodes at max_depth level)
    leaf_mask = np.zeros(num_nodes, dtype=bool)
    leaf_start = int(level_offsets[max_depth])
    leaf_mask[leaf_start:] = True

    # Extract leaf-only data for efficient band detection
    leaf_positions = positions[leaf_mask].copy()

    # Leaf widths are the parent widths (used for neighbor offset scaling)
    # For band detection, we use the parent width, not the leaf width
    # Parent width = 2 * leaf_width = positions at previous level
    leaf_widths = np.full(num_leaves, widths[leaf_start] * 2, dtype=np.float32)

    # Convert to JAX arrays
    return DenseQuadtreeStructure(
        positions=jnp.array(positions),
        widths=jnp.array(widths),
        levels=jnp.array(levels),
        level_offsets=jnp.array(level_offsets),
        leaf_mask=jnp.array(leaf_mask),
        leaf_positions=jnp.array(leaf_positions),
        leaf_widths=jnp.array(leaf_widths),
        num_nodes=num_nodes,
        num_leaves=num_leaves,
        max_depth=max_depth,
    )


# Cache of pre-computed structures for each max_depth (1-7)
_QUADTREE_CACHE: Dict[int, DenseQuadtreeStructure] = {}


def get_quadtree_structure(max_depth: int) -> DenseQuadtreeStructure:
    """Get pre-computed quadtree structure (cached).

    Structures are computed lazily and cached for reuse.

    Args:
        max_depth: Maximum depth (must be >= 1, no upper limit but depths > 7 are slow)

    Returns:
        Pre-computed DenseQuadtreeStructure

    Note:
        Position count formula: sum(4^(level+1) for level in 0..max_depth) = (4^(max_depth+2) - 4) / 3
        Depth 7 = 87,380 positions, Depth 8 = 349,524 positions, Depth 9 = 1,398,100 positions
        Memory and time scale approximately 4x per depth level.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    # Warn about high depths - position count grows as O(4^n)
    # Position formula: (4^(max_depth+2) - 4) / 3
    # Empirical GPU-resident timings (pop=1000, XOR problem):
    #   depth 1: 20 pos, ~2.6s | depth 2: 84 pos, ~4.6s | depth 3: 340 pos, ~2.1s
    #   depth 4: 1,364 pos, ~3.7s | depth 5: 5,460 pos, ~10.3s
    #   depth 6: 21,844 pos, ~46s | depth 7: 87,380 pos, ~26 min (1575s sparse)
    if max_depth > 7:
        total_positions = (4 ** (max_depth + 2) - 4) // 3
        import warnings
        warnings.warn(
            f"max_depth={max_depth} creates {total_positions:,} quadtree positions. "
            f"Position count grows as O(4^n): depth 8 = 349,524, depth 9 = 1,398,100. "
            f"Memory usage scales with population × positions × 4 bytes. "
            f"Empirical timings (pop=1000, GPU-resident, XOR): "
            f"depth 7 = ~26 min (87K pos), expect ~4x longer per additional depth level.",
            UserWarning,
            stacklevel=2
        )

    if max_depth not in _QUADTREE_CACHE:
        _QUADTREE_CACHE[max_depth] = precompute_quadtree_structure(max_depth)

    return _QUADTREE_CACHE[max_depth]


# ============================================================================
# Hierarchical Multi-Resolution Grid Structure
# ============================================================================
# This enables vmappable ES-HyperNEAT with real adaptive substrate discovery
# by pre-computing all positions at ALL resolution levels upfront.

class HierarchicalGridStructure(NamedTuple):
    """Pre-computed multi-resolution grid for vmappable ES-HyperNEAT.

    Unlike DenseQuadtreeStructure which is a complete quadtree, this structure
    stores positions at each resolution level SEPARATELY for variance-based
    subdivision decisions.

    Grid Layout (for max_depth=2):
        Level 0: 2×2 = 4 cells   (coarsest)
        Level 1: 4×4 = 16 cells
        Level 2: 8×8 = 64 cells  (finest)

    Each level-i cell maps to 4 children at level-(i+1).

    Memory: For max_depth=3: 4 + 16 + 64 + 256 = 340 positions (~5.4 KB)
    """
    # Positions at each level: level_positions[level] has shape (4^(level+1), 2)
    level_positions: Tuple[jnp.ndarray, ...]  # Tuple of [num_cells_at_level, 2]

    # Cell widths at each level (half the cell size)
    level_widths: Tuple[jnp.ndarray, ...]  # Tuple of [num_cells_at_level]

    # Parent indices: parent_indices[level][i] = index of parent at level-1
    # Level 0 has no parents (all are roots)
    parent_indices: Tuple[jnp.ndarray, ...]  # Tuple of [num_cells_at_level]

    # Neighbor positions for band detection: [num_cells, 4, 2] for each level
    # Neighbors: [left, right, top, bottom]
    neighbor_positions: Tuple[jnp.ndarray, ...]

    # Flattened all-level positions for unified CPPN queries
    all_positions: jnp.ndarray  # [total_cells, 2]

    # Level offsets into all_positions: level_offsets[i] = start of level i
    level_offsets: jnp.ndarray  # [num_levels + 1]

    # Metadata
    max_depth: int
    num_levels: int
    total_positions: int

    # Static metadata for JIT-compatible functions (Python tuples, not JAX arrays)
    # These enable use inside jax.lax.while_loop without ConcretizationTypeError
    level_sizes_static: Tuple[int, ...]      # (4, 16, 64, ...) cells per level
    level_offsets_static: Tuple[int, ...]    # (0, 4, 20, 84, ...) cumulative offsets
    level_grid_sizes_static: Tuple[int, ...] # (2, 4, 8, ...) grid dimension per level


def compute_hierarchical_level_counts(max_depth: int) -> List[int]:
    """Compute number of cells at each level.

    Level i has (2^(i+1))^2 = 4^(i+1) cells arranged in a 2^(i+1) x 2^(i+1) grid.

    Args:
        max_depth: Maximum depth (0-indexed, so max_depth=2 means levels 0,1,2)

    Returns:
        List of cell counts: [4, 16, 64, ...] for each level
    """
    return [4 ** (level + 1) for level in range(max_depth + 1)]


def precompute_hierarchical_grid(max_depth: int) -> HierarchicalGridStructure:
    """Pre-compute hierarchical multi-resolution grid structure.

    Creates a grid where:
    - Level 0: 2x2 grid (4 cells) covering [-1,1] x [-1,1]
    - Level 1: 4x4 grid (16 cells)
    - Level 2: 8x8 grid (64 cells)
    - etc.

    Each cell at level i maps to 4 children at level i+1.

    Args:
        max_depth: Maximum level (0 = only 2x2 grid)

    Returns:
        HierarchicalGridStructure with all pre-computed data
    """
    num_levels = max_depth + 1
    level_counts = compute_hierarchical_level_counts(max_depth)

    level_positions_list: List[np.ndarray] = []
    level_widths_list: List[np.ndarray] = []
    parent_indices_list: List[np.ndarray] = []
    neighbor_positions_list: List[np.ndarray] = []

    for level in range(num_levels):
        grid_size = 2 ** (level + 1)  # 2, 4, 8, 16, ...
        num_cells = grid_size * grid_size

        # Cell width: full space is 2.0 (-1 to 1), divided by grid_size
        cell_width = 2.0 / grid_size
        half_width = cell_width / 2.0

        # Generate positions (cell centers)
        # Cells are arranged row by row: (0,0), (0,1), ..., (0,n-1), (1,0), ...
        positions = np.zeros((num_cells, 2), dtype=np.float32)
        for row in range(grid_size):
            for col in range(grid_size):
                idx = row * grid_size + col
                # Center position: -1 + half_width + col * cell_width
                x = -1.0 + half_width + col * cell_width
                y = -1.0 + half_width + row * cell_width
                positions[idx] = [x, y]

        level_positions_list.append(positions)
        level_widths_list.append(np.full(num_cells, half_width, dtype=np.float32))

        # Parent indices: map each cell to its parent at level-1
        if level == 0:
            # Level 0 has no parents - use -1 as sentinel
            parent_indices_list.append(np.full(num_cells, -1, dtype=np.int32))
        else:
            parent_grid_size = grid_size // 2
            parent_indices = np.zeros(num_cells, dtype=np.int32)
            for row in range(grid_size):
                for col in range(grid_size):
                    idx = row * grid_size + col
                    parent_row = row // 2
                    parent_col = col // 2
                    parent_idx = parent_row * parent_grid_size + parent_col
                    parent_indices[idx] = parent_idx
            parent_indices_list.append(parent_indices)

        # Neighbor positions for band detection
        # Each cell has 4 neighbors: left, right, top, bottom
        # Use positions directly offset by cell_width
        neighbor_pos = np.zeros((num_cells, 4, 2), dtype=np.float32)
        for i in range(num_cells):
            pos = positions[i]
            neighbor_pos[i, 0] = pos + np.array([-cell_width, 0.0])  # left
            neighbor_pos[i, 1] = pos + np.array([+cell_width, 0.0])  # right
            neighbor_pos[i, 2] = pos + np.array([0.0, -cell_width])  # top (y up)
            neighbor_pos[i, 3] = pos + np.array([0.0, +cell_width])  # bottom
        neighbor_positions_list.append(neighbor_pos)

    # Compute flattened all-level positions and offsets
    all_positions = np.concatenate(level_positions_list, axis=0)
    level_offsets = np.zeros(num_levels + 1, dtype=np.int32)
    for i, count in enumerate(level_counts):
        level_offsets[i + 1] = level_offsets[i] + count

    total_positions = sum(level_counts)

    # Compute static metadata for JIT-compatible functions
    # These are Python tuples (not JAX arrays) for use inside jax.lax.while_loop
    level_sizes_static = tuple(level_counts)  # (4, 16, 64, ...) cells per level
    level_offsets_static = tuple(int(x) for x in level_offsets)  # (0, 4, 20, 84, ...)
    level_grid_sizes_static = tuple(2 ** (level + 1) for level in range(num_levels))  # (2, 4, 8, ...)

    # Convert to JAX arrays
    return HierarchicalGridStructure(
        level_positions=tuple(jnp.array(p) for p in level_positions_list),
        level_widths=tuple(jnp.array(w) for w in level_widths_list),
        parent_indices=tuple(jnp.array(p) for p in parent_indices_list),
        neighbor_positions=tuple(jnp.array(n) for n in neighbor_positions_list),
        all_positions=jnp.array(all_positions),
        level_offsets=jnp.array(level_offsets),
        max_depth=max_depth,
        num_levels=num_levels,
        total_positions=total_positions,
        level_sizes_static=level_sizes_static,
        level_offsets_static=level_offsets_static,
        level_grid_sizes_static=level_grid_sizes_static,
    )


# Cache for hierarchical grids
_HIERARCHICAL_GRID_CACHE: Dict[int, HierarchicalGridStructure] = {}


def get_hierarchical_grid(max_depth: int) -> HierarchicalGridStructure:
    """Get pre-computed hierarchical grid structure (cached).

    Args:
        max_depth: Maximum depth (any non-negative integer, but beware of exponential growth)

    Returns:
        Pre-computed HierarchicalGridStructure

    Warning:
        Grid size grows exponentially: 4^(depth+1) cells per level.
        - depth=3: 340 total positions, ~5 KB
        - depth=5: 5,460 total positions, ~87 KB
        - depth=6: 21,844 total positions, ~350 KB
        - depth=7: 87,380 total positions, ~1.4 MB

        Computation time scales similarly. Use the lowest depth that solves your problem.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")

    # Warn about exponential growth for large depths
    if max_depth > 5:
        import warnings
        total_positions = sum(4 ** (level + 1) for level in range(max_depth + 1))
        warnings.warn(
            f"max_depth={max_depth} creates {total_positions:,} grid positions. "
            f"Grid size grows exponentially (4x per depth level). "
            f"Computation time scales similarly. "
            f"Use the lowest max_depth that solves your problem - "
            f"higher depths waste compute without improving results for simple problems.",
            UserWarning,
            stacklevel=2
        )

    if max_depth not in _HIERARCHICAL_GRID_CACHE:
        _HIERARCHICAL_GRID_CACHE[max_depth] = precompute_hierarchical_grid(max_depth)

    return _HIERARCHICAL_GRID_CACHE[max_depth]


# ============================================================================
# Phase 2a: Hierarchical Variance and Subdivision Computation
# ============================================================================
# These functions compute variance at each level and determine which regions
# need subdivision based on ES-HyperNEAT semantics (high variance = subdivide).

def compute_hierarchical_variance_single(
    weights: jnp.ndarray,
    grid: HierarchicalGridStructure,
) -> Tuple[jnp.ndarray, ...]:
    """Compute variance at each level for a single genome's CPPN outputs.

    Variance is computed over 4 children for each parent cell.
    At level 0, we compute variance over the 4 cells (no parent).
    At level i>0, we group cells by their parent and compute variance.

    Args:
        weights: CPPN outputs for all positions [total_positions]
        grid: Pre-computed hierarchical grid structure

    Returns:
        Tuple of variance arrays, one per level (excluding finest level).
        level_variances[i] has shape (num_cells_at_level_i,)
    """
    level_variances = []

    for level in range(grid.num_levels):
        level_start = int(grid.level_offsets[level])
        level_end = int(grid.level_offsets[level + 1])
        level_weights = weights[level_start:level_end]

        if level == 0:
            # Level 0: compute single variance over all 4 cells
            variance = jnp.var(level_weights)
            level_variances.append(jnp.array([variance]))
        elif level < grid.num_levels - 1:
            # Intermediate levels: compute variance for each cell's children
            # Each cell has 4 children at level+1
            next_level_start = int(grid.level_offsets[level + 1])
            next_level_end = int(grid.level_offsets[level + 2])
            next_level_weights = weights[next_level_start:next_level_end]

            num_cells = level_end - level_start
            child_grid_size = int(np.sqrt(num_cells * 4))
            parent_grid_size = int(np.sqrt(num_cells))

            # Reshape to 2D grid, then compute 2x2 block variances
            child_grid = next_level_weights.reshape(child_grid_size, child_grid_size)

            # Compute variance of each 2x2 block
            variances = []
            for row in range(parent_grid_size):
                for col in range(parent_grid_size):
                    block = child_grid[row*2:(row+1)*2, col*2:(col+1)*2]
                    variances.append(jnp.var(block))

            level_variances.append(jnp.array(variances))
        else:
            # Finest level has no children - no variance to compute
            level_variances.append(jnp.zeros(level_end - level_start))

    return tuple(level_variances)


def compute_subdivision_mask_single(
    level_variances: Tuple[jnp.ndarray, ...],
    variance_threshold: float,
    grid: HierarchicalGridStructure,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute subdivision masks for a single genome.

    ES-HyperNEAT semantics: High variance means subdivide (explore finer detail).
    A cell is "active" if:
    - Level 0: always active (roots)
    - Level i>0: active if parent had high variance (was subdivided)
    - A cell is a "stopping point" if it has LOW variance OR is at finest level

    Returns three masks for the three representation options:
    - mask_A: All active positions from all levels (multi-resolution union)
    - mask_B: Only finest level positions that were "reached" by subdivision
    - mask_C: Stopping point positions (low variance or finest level)

    Args:
        level_variances: Tuple of variance arrays per level
        variance_threshold: Threshold for subdivision decision
        grid: Pre-computed grid structure

    Returns:
        (mask_A, mask_B, mask_C) - boolean masks for all_positions
    """
    num_levels = grid.num_levels

    # Track which cells are "reached" by subdivision at each level
    # Level 0 is always reached
    level_reached = []
    level_reached.append(jnp.ones(4, dtype=bool))  # Level 0: all 4 cells reached

    for level in range(1, num_levels):
        prev_level_reached = level_reached[level - 1]
        prev_variances = level_variances[level - 1]

        # A cell is subdivided if it was reached AND has high variance
        prev_subdivided = prev_level_reached & (prev_variances > variance_threshold)

        # Current level cells are reached if their parent was subdivided
        parent_indices = grid.parent_indices[level]
        current_reached = prev_subdivided[parent_indices]

        level_reached.append(current_reached)

    # Build the three masks

    # Mask A: All reached positions from all levels
    mask_A_parts = []
    for level in range(num_levels):
        mask_A_parts.append(level_reached[level])
    mask_A = jnp.concatenate(mask_A_parts)

    # Mask B: Only finest level positions that were reached
    mask_B = jnp.zeros(grid.total_positions, dtype=bool)
    finest_start = int(grid.level_offsets[num_levels - 1])
    mask_B = mask_B.at[finest_start:].set(level_reached[num_levels - 1])

    # Mask C: Stopping points (reached AND (low variance OR finest level))
    mask_C_parts = []
    for level in range(num_levels):
        reached = level_reached[level]
        if level == num_levels - 1:
            # Finest level: all reached cells are stopping points
            stopping = reached
        else:
            # Intermediate levels: stopping if reached AND low variance
            low_variance = level_variances[level] <= variance_threshold
            stopping = reached & low_variance
        mask_C_parts.append(stopping)
    mask_C = jnp.concatenate(mask_C_parts)

    return mask_A, mask_B, mask_C


def compute_hierarchical_variances_batch(
    all_weights: jnp.ndarray,
    grid: HierarchicalGridStructure,
) -> List[jnp.ndarray]:
    """Compute variances for a batch of genomes (vmappable).

    Args:
        all_weights: CPPN outputs [pop_size, total_positions]
        grid: Pre-computed grid structure

    Returns:
        List of variance arrays per level, each [pop_size, num_cells_at_level]
    """
    pop_size = all_weights.shape[0]
    level_variances_batch = []

    for level in range(grid.num_levels - 1):  # No variance at finest level
        level_start = int(grid.level_offsets[level])
        level_end = int(grid.level_offsets[level + 1])
        num_cells = level_end - level_start

        if level == 0:
            # Level 0: single variance over 4 cells
            level_weights = all_weights[:, level_start:level_end]
            variance = jnp.var(level_weights, axis=1, keepdims=True)
            level_variances_batch.append(variance)  # [pop_size, 1]
        else:
            # Higher levels: variance of 2x2 child blocks
            next_level_start = int(grid.level_offsets[level + 1])
            next_level_end = int(grid.level_offsets[level + 2])
            next_level_weights = all_weights[:, next_level_start:next_level_end]

            child_grid_size = int(np.sqrt(next_level_end - next_level_start))
            parent_grid_size = child_grid_size // 2

            # Reshape to [pop_size, child_grid, child_grid]
            child_grids = next_level_weights.reshape(pop_size, child_grid_size, child_grid_size)

            # Compute 2x2 block variances using reshape and var
            # Reshape to [pop_size, parent_grid, 2, parent_grid, 2]
            reshaped = child_grids.reshape(pop_size, parent_grid_size, 2, parent_grid_size, 2)
            # Transpose to [pop_size, parent_grid, parent_grid, 2, 2]
            reshaped = reshaped.transpose(0, 1, 3, 2, 4)
            # Reshape to [pop_size, parent_grid, parent_grid, 4]
            blocks = reshaped.reshape(pop_size, parent_grid_size, parent_grid_size, 4)
            # Compute variance over the 4 children
            variances = jnp.var(blocks, axis=-1)
            # Flatten to [pop_size, num_parent_cells]
            variances = variances.reshape(pop_size, parent_grid_size * parent_grid_size)

            level_variances_batch.append(variances)

    # Add zeros for finest level (no children)
    # MULTI-GPU FIX: Use numpy to avoid JIT cache cross-device errors
    # The jnp.zeros call uses a globally cached JIT trace that causes
    # "Buffer on device cuda:X but replica assigned to cuda:Y" errors
    finest_size = int(grid.level_offsets[grid.num_levels]) - int(grid.level_offsets[grid.num_levels - 1])
    level_variances_batch.append(np.zeros((pop_size, finest_size), dtype=np.float32))

    return level_variances_batch


def compute_subdivision_masks_batch(
    level_variances: List[jnp.ndarray],
    variance_threshold: float,
    grid: HierarchicalGridStructure,
    return_all_masks: bool = True,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Compute subdivision masks for a batch of genomes.

    Args:
        level_variances: List of [pop_size, num_cells] arrays per level
        variance_threshold: Threshold for subdivision
        grid: Pre-computed grid structure
        return_all_masks: If False, return only masks_A (saves ~0.66 GB at depth 8)

    Returns:
        If return_all_masks=True: (masks_A, masks_B, masks_C) - each [pop_size, total_positions] boolean
        If return_all_masks=False: masks_A only - [pop_size, total_positions] boolean
    """
    pop_size = level_variances[0].shape[0]
    num_levels = grid.num_levels

    # Track reached cells at each level
    level_reached = []
    level_reached.append(jnp.ones((pop_size, 4), dtype=bool))  # Level 0 always reached

    for level in range(1, num_levels):
        prev_reached = level_reached[level - 1]
        prev_variances = level_variances[level - 1]

        # Cells subdivided if reached AND high variance
        prev_subdivided = prev_reached & (prev_variances > variance_threshold)

        # Current level reached if parent was subdivided
        parent_indices = grid.parent_indices[level]
        # Gather parent subdivision status for each cell
        current_reached = prev_subdivided[:, parent_indices]

        level_reached.append(current_reached)

    # Build masks
    # Mask A: All reached positions
    masks_A = jnp.concatenate(level_reached, axis=1)

    # MEMORY OPTIMIZATION: Skip masks_B and masks_C if not needed
    if not return_all_masks:
        return masks_A

    # Mask B: Only finest level reached positions
    masks_B = jnp.zeros((pop_size, grid.total_positions), dtype=bool)
    finest_start = int(grid.level_offsets[num_levels - 1])
    masks_B = masks_B.at[:, finest_start:].set(level_reached[num_levels - 1])

    # Mask C: Stopping points (reached AND (low variance OR finest))
    masks_C_parts = []
    for level in range(num_levels):
        reached = level_reached[level]
        if level == num_levels - 1:
            stopping = reached
        else:
            low_variance = level_variances[level] <= variance_threshold
            stopping = reached & low_variance
        masks_C_parts.append(stopping)
    masks_C = jnp.concatenate(masks_C_parts, axis=1)

    return masks_A, masks_B, masks_C


# ============================================================================
# Phase 2a-JIT: JIT-Compatible Variance and Mask Functions
# ============================================================================
# These functions use static Python integers for indices instead of JAX arrays,
# enabling use inside jax.lax.while_loop without ConcretizationTypeError.


def compute_hierarchical_variances_batch_jit(
    all_weights: jnp.ndarray,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[int, ...],
    num_levels: int,
) -> Tuple[jnp.ndarray, ...]:
    """JIT-compatible variance computation using static level metadata.

    This function replaces compute_hierarchical_variances_batch() for use inside
    jax.lax.while_loop. All index parameters are Python integers (static), not
    JAX arrays, avoiding ConcretizationTypeError.

    Args:
        all_weights: CPPN outputs [pop_size, total_positions]
        level_sizes: Tuple of cell counts per level (4, 16, 64, ...)
        level_offsets: Tuple of cumulative offsets (0, 4, 20, 84, ...)
        level_grid_sizes: Tuple of grid dimensions (2, 4, 8, ...)
        num_levels: Number of levels (static Python int)

    Returns:
        Tuple of variance arrays per level, each [pop_size, num_cells_at_level]
    """
    pop_size = all_weights.shape[0]
    variances_list = []

    for level in range(num_levels - 1):  # No variance at finest level
        # Static Python integers - NO int() conversion needed!
        level_start = level_offsets[level]
        level_end = level_offsets[level + 1]
        level_size = level_sizes[level]

        if level == 0:
            # Level 0: single variance over 4 cells
            # Use array slicing with static indices
            level_weights = all_weights[:, level_start:level_end]
            variance = jnp.var(level_weights, axis=1, keepdims=True)
            variances_list.append(variance)  # [pop_size, 1]
        else:
            # Higher levels: variance of 2x2 child blocks
            next_start = level_offsets[level + 1]
            next_end = level_offsets[level + 2]
            child_grid_size = level_grid_sizes[level + 1]  # Static!
            parent_grid_size = level_grid_sizes[level]     # Static!

            # Use array slicing with static indices
            next_level_weights = all_weights[:, next_start:next_end]

            # Static reshape dimensions
            child_grids = next_level_weights.reshape(pop_size, child_grid_size, child_grid_size)
            # Reshape to [pop_size, parent_grid, 2, parent_grid, 2]
            reshaped = child_grids.reshape(pop_size, parent_grid_size, 2, parent_grid_size, 2)
            # Transpose to [pop_size, parent_grid, parent_grid, 2, 2]
            reshaped = reshaped.transpose(0, 1, 3, 2, 4)
            # Reshape to [pop_size, parent_grid, parent_grid, 4]
            blocks = reshaped.reshape(pop_size, parent_grid_size, parent_grid_size, 4)
            # Compute variance over the 4 children
            variances = jnp.var(blocks, axis=-1)
            # Flatten to [pop_size, num_parent_cells]
            variances = variances.reshape(pop_size, parent_grid_size * parent_grid_size)

            variances_list.append(variances)

    # Add zeros for finest level (no children)
    finest_size = level_sizes[-1]  # Static Python int!
    variances_list.append(jnp.zeros((pop_size, finest_size)))

    return tuple(variances_list)


def compute_subdivision_masks_batch_jit(
    level_variances: Tuple[jnp.ndarray, ...],
    variance_threshold: float,
    parent_indices_tuple: Tuple[jnp.ndarray, ...],
    level_offsets: Tuple[int, ...],
    num_levels: int,
    total_positions: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compatible mask computation using static metadata.

    This function replaces compute_subdivision_masks_batch() for use inside
    jax.lax.while_loop. All index parameters are Python integers (static).

    Args:
        level_variances: Tuple of [pop_size, num_cells] arrays per level
        variance_threshold: Threshold for subdivision (static Python float)
        parent_indices_tuple: Tuple of parent index arrays per level
        level_offsets: Tuple of cumulative offsets (0, 4, 20, 84, ...)
        num_levels: Number of levels (static Python int)
        total_positions: Total positions in grid (static Python int)

    Returns:
        (masks_A, masks_B, masks_C) - each [pop_size, total_positions] boolean
    """
    pop_size = level_variances[0].shape[0]

    # Track reached cells at each level
    level_reached = [jnp.ones((pop_size, 4), dtype=bool)]  # Level 0 always reached

    for level in range(1, num_levels):
        prev_reached = level_reached[level - 1]
        prev_variances = level_variances[level - 1]

        # Cells subdivided if reached AND high variance
        prev_subdivided = prev_reached & (prev_variances > variance_threshold)

        # Current level reached if parent was subdivided
        parent_indices = parent_indices_tuple[level]
        # Gather parent subdivision status for each cell
        current_reached = prev_subdivided[:, parent_indices]

        level_reached.append(current_reached)

    # Build masks using static indices
    masks_A = jnp.concatenate(level_reached, axis=1)

    # Mask B: Only finest level reached positions
    masks_B = jnp.zeros((pop_size, total_positions), dtype=bool)
    finest_start = level_offsets[num_levels - 1]  # Static Python int!
    masks_B = masks_B.at[:, finest_start:].set(level_reached[num_levels - 1])

    # Mask C: Stopping points (reached AND (low variance OR finest))
    masks_C_parts = []
    for level in range(num_levels):
        reached = level_reached[level]
        if level == num_levels - 1:
            stopping = reached
        else:
            low_variance = level_variances[level] <= variance_threshold
            stopping = reached & low_variance
        masks_C_parts.append(stopping)
    masks_C = jnp.concatenate(masks_C_parts, axis=1)

    return masks_A, masks_B, masks_C


# ============================================================================
# Phase 2b: Hierarchical Band Detection
# ============================================================================

def compute_band_detection_single_level(
    weights: jnp.ndarray,
    neighbor_weights: jnp.ndarray,
    band_threshold: float,
) -> jnp.ndarray:
    """Compute band detection for positions at a single level.

    Band detection checks if a position has consistent weight differences
    with its neighbors, indicating a meaningful "band" in the weight pattern.

    Band value = max(min(d_horizontal), min(d_vertical))
    where d = |weight - neighbor_weight|

    Args:
        weights: CPPN outputs at positions [num_positions]
        neighbor_weights: CPPN outputs at neighbor positions [num_positions, 4]
                         Order: [left, right, top, bottom]
        band_threshold: Minimum band value for position to be valid

    Returns:
        Boolean mask [num_positions] - True if band detection passes
    """
    # Compute weight differences to each neighbor
    diffs = jnp.abs(weights[:, None] - neighbor_weights)  # [num_positions, 4]

    # Horizontal: min of left and right
    d_horizontal = jnp.minimum(diffs[:, 0], diffs[:, 1])

    # Vertical: min of top and bottom
    d_vertical = jnp.minimum(diffs[:, 2], diffs[:, 3])

    # Band value: max of horizontal and vertical consistency
    band_values = jnp.maximum(d_horizontal, d_vertical)

    return band_values > band_threshold


def batch_query_neighbors_hierarchical(
    state: Any,
    cppn_transformed: Any,
    source_coord: jnp.ndarray,
    neighbor_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
) -> jnp.ndarray:
    """Query CPPN for neighbor positions at all cells.

    Args:
        state: Algorithm state
        cppn_transformed: Transformed CPPN
        source_coord: Source coordinate [2]
        neighbor_positions: Neighbor positions [num_cells, 4, 2]
        outgoing: Direction of connection
        cppn_forward: JIT-compiled forward function

    Returns:
        Neighbor weights [num_cells, 4]
    """
    num_cells = neighbor_positions.shape[0]
    # Flatten to [num_cells * 4, 2]
    flat_neighbors = neighbor_positions.reshape(-1, 2)

    # Build inputs
    num_queries = flat_neighbors.shape[0]
    source_tiled = jnp.tile(source_coord[None, :], (num_queries, 1))
    bias = jnp.ones((num_queries, 1))

    if outgoing:
        inputs = jnp.concatenate([source_tiled, flat_neighbors, bias], axis=1)
    else:
        inputs = jnp.concatenate([flat_neighbors, source_tiled, bias], axis=1)

    # Batched query
    flat_weights = jax.vmap(
        lambda x: cppn_forward(state, cppn_transformed, x)
    )(inputs)

    # Reshape back to [num_cells, 4]
    return flat_weights.reshape(num_cells, 4)


# ============================================================================
# Batched CPPN Queries
# ============================================================================

def batch_query_all_positions(
    state: Any,
    cppn_transformed: Any,
    source_coord: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
) -> jnp.ndarray:
    """Query CPPN for all target positions in ONE batched vmap call.

    This is the core optimization: instead of N sequential queries,
    we perform 1 batched query using JAX vmap.

    Args:
        state: Algorithm state
        cppn_transformed: Transformed CPPN for forward pass
        source_coord: Source coordinate (x, y) - shape (2,)
        target_positions: All target positions - shape (N, 2)
        outgoing: If True, query source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function

    Returns:
        Raw CPPN outputs - shape (N,)
    """
    num_positions = target_positions.shape[0]

    # Build CPPN inputs: [x1, y1, x2, y2, bias=1.0]
    source_tiled = jnp.tile(source_coord[None, :], (num_positions, 1))
    bias = jnp.ones((num_positions, 1))

    if outgoing:
        # source -> target
        inputs = jnp.concatenate([source_tiled, target_positions, bias], axis=1)
    else:
        # target -> source
        inputs = jnp.concatenate([target_positions, source_tiled, bias], axis=1)

    # Single batched CPPN evaluation
    weights = jax.vmap(
        lambda x: cppn_forward(state, cppn_transformed, x)
    )(inputs)

    return weights.flatten()


def vectorized_weight_sparsification(
    weights: jnp.ndarray,
    threshold: float = 0.2,
    max_weight: float = 8.0,
) -> jnp.ndarray:
    """Apply PUREPLES-compatible weight sparsification (vectorized).

    Weights below threshold are zeroed; weights above are scaled to [-max_weight, max_weight].

    This matches the exact formula from PUREPLES/ES-HyperNEAT:
    - If |weight| > threshold: scale to [-max_weight, max_weight]
    - Otherwise: zero

    Args:
        weights: Raw CPPN outputs - shape (N,)
        threshold: Sparsification threshold (default 0.2)
        max_weight: Maximum weight value (default 8.0)

    Returns:
        Sparsified weights - shape (N,)
    """
    # Handle NaN/Inf
    weights = jnp.where(jnp.isnan(weights) | jnp.isinf(weights), 0.0, weights)

    abs_weights = jnp.abs(weights)
    above_threshold = abs_weights > threshold

    # Scale weights that pass threshold
    # Positive: (w - threshold) / (1 - threshold) * max_weight
    # Negative: (w + threshold) / (1 - threshold) * max_weight
    scaled_positive = (weights - threshold) / (1.0 - threshold)
    scaled_negative = (weights + threshold) / (1.0 - threshold)
    scaled = jnp.where(weights > 0, scaled_positive, scaled_negative)
    scaled = jnp.clip(scaled * max_weight, -max_weight, max_weight)

    return jnp.where(above_threshold, scaled, 0.0)


# ============================================================================
# Population-Level Batch CPPN Pre-Query
# ============================================================================

# Device-specific CPPN query implementation using Python for-loop.
# This avoids JAX vmap trace caching issues that cause "Buffer passed to
# Execute()... is on device cuda:0, but replica is assigned to device cuda:1"
# errors in multi-GPU ThreadPoolExecutor execution.
#
# The issue: JAX vmap traces are cached globally, and when two threads create
# traces for the same input shape concurrently, one thread's trace can be
# reused by the other, causing device placement errors.
#
# The solution: Use Python for-loop instead of vmap for the inner loop.
# This is slightly slower but avoids the trace caching problem entirely.

def _query_cppn_sequential(cppn_forward, state, inputs, cppn_tuple):
    """Query CPPN at all positions using sequential loop.

    Uses Python for-loop to avoid JAX vmap trace caching issues in multi-GPU.
    """
    weights_list = []
    for i in range(inputs.shape[0]):
        w = cppn_forward(state, cppn_tuple, inputs[i])
        weights_list.append(w)
    return jnp.stack(weights_list).flatten()


def _query_population_sequential(cppn_forward, state, inputs, cppns_transformed):
    """Query ALL CPPNs at ALL positions using Python for-loops.

    This completely avoids JAX vmap trace caching by using Python for-loops
    for both the population and position dimensions. This is slower than vmap
    but prevents "Buffer passed to Execute()... is on device cuda:0, but replica
    is assigned to device cuda:1" errors in multi-GPU ThreadPoolExecutor execution.

    Args:
        cppn_forward: JIT-compiled CPPN forward function
        state: Algorithm state
        inputs: Input array (num_positions, input_dim)
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)

    Returns:
        (pop_size, num_positions) array of CPPN outputs
    """
    pop_size = cppns_transformed[0].shape[0]
    all_results = []

    # Python for-loop over population - NOT vmapped
    for i in range(pop_size):
        cppn_tuple = (
            cppns_transformed[0][i],
            cppns_transformed[1][i],
            cppns_transformed[2][i],
            cppns_transformed[3][i],
        )
        # Query all positions for this single CPPN
        weights = _query_cppn_sequential(cppn_forward, state, inputs, cppn_tuple)
        all_results.append(weights)

    return jnp.stack(all_results)  # (pop_size, num_positions)


# Per-device JIT function cache to ensure separate traces per device.
# FIXED v2: Use functools.partial to bind cppn_forward per-device, avoiding
# cross-device trace contamination. Each device gets its own JIT trace.
_per_device_vmap_cache = {}
_per_device_cache_lock = threading.Lock()


def clear_per_device_caches():
    """Clear all per-device JIT function caches.

    Call this before switching between devices to prevent stale traces
    from being reused across different device contexts.

    This is needed because jax.clear_caches() only clears JAX's internal
    compilation cache, not our Python-level cache of compiled functions.
    """
    global _per_device_vmap_cache
    with _per_device_cache_lock:
        _per_device_vmap_cache.clear()


def _create_query_fn_for_device(cppn_forward: Any, device_id: int):
    """Create a fresh query function bound to a specific device.

    This creates a NEW function each time, ensuring no trace contamination
    between devices. The function is JIT-compiled on first call with
    device-local data, creating device-specific traces.

    Args:
        cppn_forward: The CPPN forward function
        device_id: Target device index

    Returns:
        A fresh JIT-compiled query function for this device
    """
    # Get the target device
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Create a fresh function under the device context
    # This ensures any internal traces are created for this device
    with jax.default_device(device):
        def query_population_vmap_impl(state, inputs, cppns_nodes, cppns_conns,
                                       cppns_conn_attrs, cppns_node_attrs):
            """Query all CPPNs at all positions using vmap.

            state is an explicit parameter to ensure correct device placement.
            """
            def query_single_cppn(cppn_tuple):
                """Query one CPPN at all positions."""
                return jax.vmap(
                    lambda x: cppn_forward(state, cppn_tuple, x)
                )(inputs).flatten()

            return jax.vmap(
                query_single_cppn,
                in_axes=((0, 0, 0, 0),)
            )((cppns_nodes, cppns_conns, cppns_conn_attrs, cppns_node_attrs))

        # JIT compile with explicit device pinning
        # CRITICAL: Without device=, the JIT'd function can be reused across devices
        # causing "Buffer passed to Execute() ... is on device cuda:X" errors
        return jax.jit(query_population_vmap_impl, device=device)


def _get_device_specific_query_fn(device_id: int, cppn_forward: Any):
    """Get or create a device-specific JIT-compiled query function.

    Uses a cache to avoid recreating functions for the same device,
    but each device gets its own independently-traced function.

    Args:
        device_id: Device index for cache key
        cppn_forward: The CPPN forward function

    Returns:
        JIT-compiled query function for this device
    """
    # Cache key includes id(cppn_forward) to handle different CPPN functions
    cache_key = (device_id, id(cppn_forward))

    with _per_device_cache_lock:
        if cache_key not in _per_device_vmap_cache:
            # Create a fresh function for this device
            _per_device_vmap_cache[cache_key] = _create_query_fn_for_device(
                cppn_forward, device_id
            )
        return _per_device_vmap_cache[cache_key]


def _query_population_vmap_device_aware(cppn_forward, state, inputs,
                                         cppns_transformed, device_id):
    """Query ALL CPPNs at ALL positions using device-aware vmap.

    This uses per-device JIT-compiled functions to avoid trace sharing
    between devices in multi-GPU ThreadPoolExecutor execution.

    FIXED v2: Explicitly ensure all inputs are on the target device before
    calling the query function. This handles cases where arrays might have
    been created on a different device.

    Args:
        cppn_forward: JIT-compiled CPPN forward function
        state: Algorithm state (should already be on the target device)
        inputs: Input array (num_positions, input_dim)
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        device_id: Device index for trace isolation

    Returns:
        (pop_size, num_positions) array of CPPN outputs
    """
    # Get the target device
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # CRITICAL: Explicitly move ALL inputs to the target device
    # This ensures device-consistent execution even if some arrays
    # were created on a different device
    state_on_device = jax.tree.map(lambda x: jax.device_put(x, device), state)
    inputs_on_device = jax.device_put(inputs, device)
    cppns_on_device = tuple(jax.device_put(arr, device) for arr in cppns_transformed)

    # Get per-device query function (creates one if not cached)
    query_fn = _get_device_specific_query_fn(device_id, cppn_forward)

    # Execute under device context
    with jax.default_device(device):
        return query_fn(
            state_on_device,
            inputs_on_device,
            cppns_on_device[0],
            cppns_on_device[1],
            cppns_on_device[2],
            cppns_on_device[3]
        )


def batch_query_population_positions(
    state: Any,
    cppns_transformed: Tuple,
    source_coord: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
) -> jnp.ndarray:
    """Query ALL CPPNs at ALL target positions in ONE vmap call.

    This is the core optimization: instead of 1000 sequential calls
    (one per genome), we perform 1 batched call with double vmap.

    Memory usage: pop_size × num_positions × 4 bytes
    - 1000 × 1024 × 4 = ~4 MB (negligible)

    IMPORTANT: The device_id parameter selects a device-specific query function
    to ensure separate vmap trace caches for each device. This prevents
    "Buffer passed to Execute()... is on device cuda:0, but replica is assigned
    to device cuda:1" errors in multi-GPU execution with ThreadPoolExecutor.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coord: Single source coordinate (2,)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        device_id: Device index to select device-specific function (default 0)
        geometry_seeding_enabled: If True, add delta_x, delta_y to CPPN inputs (7D instead of 5D)

    Returns:
        (pop_size, num_positions) array of CPPN outputs
    """
    # CRITICAL: Get target device and ensure all inputs are on correct device
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Ensure input arrays are on correct device before building CPPN inputs
    source_coord = jax.device_put(source_coord, device)
    target_positions = jax.device_put(target_positions, device)

    pop_size = cppns_transformed[0].shape[0]
    num_positions = target_positions.shape[0]

    # Build CPPN inputs: [x1, y1, x2, y2, bias=1.0] or [x1, y1, x2, y2, delta_x, delta_y, bias=1.0]
    # CRITICAL: Use numpy to create arrays, then device_put to target device.
    # jnp.ones/tile use cached JIT traces that cause cross-device errors.
    # numpy arrays don't have this issue - they're just Python objects.
    source_np = np.tile(np.asarray(source_coord)[None, :], (num_positions, 1))
    source_tiled = jax.device_put(source_np, device)
    bias_np = np.ones((num_positions, 1), dtype=np.float32)
    bias = jax.device_put(bias_np, device)
    target_np = np.asarray(target_positions)
    target_positions = jax.device_put(target_np, device)

    if geometry_seeding_enabled:
        # Geometry seeding: add delta (coordinate differences) as CPPN inputs (Risi & Stanley 2012)
        # Delta values allow CPPNs to learn direction-specific patterns
        # Gaussian activation on delta naturally biases toward local connections (peaks at delta=0)
        if outgoing:
            delta_x = target_np[:, 0:1] - source_np[:, 0:1]
            delta_y = target_np[:, 1:2] - source_np[:, 1:2]
            inputs_np = np.concatenate([source_np, target_np, delta_x, delta_y, bias_np], axis=1)
        else:
            delta_x = source_np[:, 0:1] - target_np[:, 0:1]
            delta_y = source_np[:, 1:2] - target_np[:, 1:2]
            inputs_np = np.concatenate([target_np, source_np, delta_x, delta_y, bias_np], axis=1)
    else:
        # Standard 5D input: [x1, y1, x2, y2, bias]
        if outgoing:
            inputs_np = np.concatenate([source_np, target_np, bias_np], axis=1)
        else:
            inputs_np = np.concatenate([target_np, source_np, bias_np], axis=1)
    inputs = jax.device_put(inputs_np, device)

    # MULTI-GPU FIX: Use device-aware vmap with per-device JIT cache.
    # The issue: JAX vmap traces are cached globally. When feedforward depth 2 runs
    # first on GPU 0, it creates a trace. When hybrid depth 2 runs on GPU 1, JAX
    # reuses the same trace bound to GPU 0, causing device placement errors.
    #
    # Solution: Use per-device JIT-compiled functions with device_id in cache key.
    # Each device gets its own compiled trace, preventing cross-device reuse.
    # This maintains vmap performance while ensuring correct device placement.

    return _query_population_vmap_device_aware(
        cppn_forward, state, inputs, cppns_transformed, device_id
    )


def batch_query_population_positions_jax_pure(
    state: Any,
    cppns_transformed: Tuple,
    source_coord: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
) -> jnp.ndarray:
    """JAX-pure version of batch_query_population_positions for use in while_loop.

    Unlike batch_query_population_positions, this version uses ONLY JAX operations
    (no numpy conversions) so it can be traced inside jax.lax.while_loop.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coord: Single source coordinate (2,)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function

    Returns:
        (pop_size, num_positions) array of CPPN outputs
    """
    pop_size = cppns_transformed[0].shape[0]
    num_positions = target_positions.shape[0]

    # Build CPPN inputs: [x1, y1, x2, y2, bias=1.0]
    # Use jnp.tile (not np.tile) for vmap compatibility
    source_tiled = jnp.tile(source_coord[None, :], (num_positions, 1))
    bias = jnp.ones((num_positions, 1))

    if outgoing:
        inputs = jnp.concatenate([source_tiled, target_positions, bias], axis=1)
    else:
        inputs = jnp.concatenate([target_positions, source_tiled, bias], axis=1)

    # Inner function: query single CPPN at all positions
    def query_single_cppn(cppn_tuple):
        """Query one CPPN at all positions."""
        # vmap over positions
        weights = jax.vmap(
            lambda x: cppn_forward(state, cppn_tuple, x)
        )(inputs)
        return weights.flatten()

    # Outer vmap: over population
    all_weights = jax.vmap(
        query_single_cppn,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1],
       cppns_transformed[2], cppns_transformed[3]))

    return all_weights  # (pop_size, num_positions)


def batch_query_population_multi_source_jax_pure(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
) -> jnp.ndarray:
    """JAX-pure version of batch_query_population_multi_source for while_loop.

    Uses only JAX operations (no numpy) so it can be traced inside jax.lax.while_loop.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    # vmap over sources using JAX-pure query function
    def query_from_source(source_coord):
        return batch_query_population_positions_jax_pure(
            state, cppns_transformed, source_coord, target_positions,
            outgoing, cppn_forward
        )

    # Result: (num_sources, pop_size, num_positions)
    result = jax.vmap(query_from_source)(source_coords)

    # Transpose to (pop_size, num_sources, num_positions)
    return jnp.transpose(result, (1, 0, 2))


def batch_query_population_multi_source(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    device_id: int = 0,
) -> jnp.ndarray:
    """Query ALL CPPNs from ALL source coords to ALL target positions.

    This is an extended version for when we have multiple source coordinates
    (e.g., all input nodes or all hidden nodes).

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        device_id: Device index for multi-GPU execution (default 0)

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    # vmap over sources - pass device_id to ensure correct device placement
    def query_from_source(source_coord):
        return batch_query_population_positions(
            state, cppns_transformed, source_coord, target_positions,
            outgoing, cppn_forward, device_id
        )

    # Result: (num_sources, pop_size, num_positions)
    result = jax.vmap(query_from_source)(source_coords)

    # Transpose to (pop_size, num_sources, num_positions)
    return jnp.transpose(result, (1, 0, 2))


def batch_query_population_multi_source_chunked(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    pop_chunk_size: int = 100,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
) -> jnp.ndarray:
    """Chunked version of batch_query_population_multi_source.

    MEMORY OPTIMIZATION: Processes population in chunks to reduce peak memory.
    At depth 8 with pop=1000, chunking with size=100 reduces peak memory from
    139+ GB (if XLA unrolls all at once) to ~1-2 GB per chunk.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Number of genomes to process at once (default 100)
        device_id: Device index to force separate traces per device (default 0)
        geometry_seeding_enabled: If True, add delta_x, delta_y to CPPN inputs (7D instead of 5D)

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    # CRITICAL: Get target device for device-consistent operations
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Ensure inputs are on correct device
    source_coords = jax.device_put(source_coords, device)
    target_positions = jax.device_put(target_positions, device)
    cppns_transformed = tuple(jax.device_put(arr, device) for arr in cppns_transformed)

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

            # Extract chunk of CPPNs (already on correct device)
            chunk_cppns = (
                cppns_transformed[0][chunk_start:chunk_end],
                cppns_transformed[1][chunk_start:chunk_end],
                cppns_transformed[2][chunk_start:chunk_end],
                cppns_transformed[3][chunk_start:chunk_end],
            )

            # Query chunk (double vmap: chunk_pop x positions)
            # Pass device_id to ensure device-specific traces
            chunk_weights = batch_query_population_positions(
                state, chunk_cppns, source_coord, target_positions,
                outgoing, cppn_forward, device_id, geometry_seeding_enabled
            )
            # CRITICAL: Ensure chunk result is on correct device before appending
            chunk_weights = jax.device_put(chunk_weights, device)
            chunk_results.append(chunk_weights)

        # CRITICAL: Use numpy concatenate to avoid JAX JIT cache cross-device errors.
        # jnp.concatenate uses globally cached JIT traces.
        chunk_results_np = [np.asarray(c) for c in chunk_results]
        source_weights = np.concatenate(chunk_results_np, axis=0)
        results_list.append(source_weights)

    # Stack sources: (num_sources, pop_size, num_positions)
    # Transpose to: (pop_size, num_sources, num_positions)
    # CRITICAL: Use numpy for stack/transpose to avoid JAX JIT cache cross-device errors.
    # jnp.stack and jnp.transpose use globally cached JIT traces that can cause
    # "Buffer on device cuda:X but replica assigned to cuda:Y" errors.
    results_np = [np.asarray(r) for r in results_list]
    result_np = np.stack(results_np, axis=0)
    result_np = np.transpose(result_np, (1, 0, 2))
    return jax.device_put(result_np, device)


def precompute_all_query_positions(
    tree: DenseQuadtreeStructure,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-compute all positions that will be queried during discovery.

    Returns leaf positions and all neighbor positions (for band detection).
    These positions are fixed for a given quadtree structure.

    Args:
        tree: Pre-computed quadtree structure

    Returns:
        Tuple of:
        - leaf_positions: (num_leaves, 2)
        - neighbor_positions: (num_leaves * 4, 2) flattened neighbor positions
    """
    leaf_positions = tree.leaf_positions
    leaf_widths = tree.leaf_widths

    # Compute all neighbor positions: (num_leaves, 4, 2)
    neighbor_positions = (
        leaf_positions[:, None, :] +
        NEIGHBOR_OFFSETS[None, :, :] * leaf_widths[:, None, None]
    )

    # Flatten to (num_leaves * 4, 2)
    flat_neighbor_positions = neighbor_positions.reshape(-1, 2)

    return leaf_positions, flat_neighbor_positions


def vectorized_band_detection_prequery(
    leaf_weights: jnp.ndarray,
    prequeried_neighbor_weights: jnp.ndarray,
    active_mask: jnp.ndarray,
    band_threshold: float,
    max_weight: float = 8.0,
) -> jnp.ndarray:
    """Vectorized band detection using pre-queried neighbor weights.

    This is the pre-query version of vectorized_band_detection.
    Instead of calling the CPPN, it uses weights that were pre-computed.

    Args:
        leaf_weights: CPPN weights at leaves (already sparsified) - (num_leaves,)
        prequeried_neighbor_weights: Raw CPPN outputs at neighbors - (num_leaves * 4,)
        active_mask: Which leaves to check - (num_leaves,)
        band_threshold: Threshold for band detection
        max_weight: Maximum weight for sparsification

    Returns:
        Valid connection mask - (num_leaves,)
    """
    num_leaves = leaf_weights.shape[0]

    # Apply sparsification to neighbor weights
    neighbor_weights_flat = vectorized_weight_sparsification(
        prequeried_neighbor_weights, max_weight=max_weight
    )

    # Reshape: (num_leaves, 4)
    neighbor_weights = neighbor_weights_flat.reshape(num_leaves, 4)

    # Compute differences
    d_left = jnp.abs(leaf_weights - neighbor_weights[:, 0])
    d_right = jnp.abs(leaf_weights - neighbor_weights[:, 1])
    d_top = jnp.abs(leaf_weights - neighbor_weights[:, 2])
    d_bottom = jnp.abs(leaf_weights - neighbor_weights[:, 3])

    # Band formula: max(min(d_top, d_bottom), min(d_left, d_right))
    band_value = jnp.maximum(
        jnp.minimum(d_top, d_bottom),
        jnp.minimum(d_left, d_right)
    )

    # Valid connection if:
    # 1. Active (parent variance low enough)
    # 2. Band exceeds threshold
    # 3. Weight is non-zero
    valid = (
        active_mask &
        (band_value > band_threshold) &
        (leaf_weights != 0.0)
    )

    return valid


def dense_substrate_discovery_prequery(
    source_coord: jnp.ndarray,
    tree: DenseQuadtreeStructure,
    prequeried_leaf_weights: jnp.ndarray,
    prequeried_neighbor_weights: jnp.ndarray,
    initial_depth: int,
    variance_threshold: float,
    division_threshold: float,
    band_threshold: float,
    outgoing: bool,
    max_weight: float = 8.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-accelerated substrate discovery using pre-queried CPPN weights.

    This is the pre-query version of dense_substrate_discovery.
    Instead of calling the CPPN, it uses weights that were pre-computed
    for all CPPNs in the population.

    Args:
        source_coord: Source coordinate - shape (2,)
        tree: Pre-computed quadtree structure
        prequeried_leaf_weights: Raw CPPN outputs at leaves - (num_leaves,)
        prequeried_neighbor_weights: Raw CPPN outputs at neighbors - (num_leaves * 4,)
        initial_depth: Minimum subdivision depth
        variance_threshold: Threshold for extraction
        division_threshold: Threshold for subdivision
        band_threshold: Threshold for band detection
        outgoing: Direction (True=source->target, False=target->source)
        max_weight: Maximum connection weight

    Returns:
        Tuple of:
        - discovered_positions: (num_leaves, 2) with NaN for invalid
        - discovered_weights: (num_leaves,) with NaN for invalid
        - num_valid: scalar count of valid connections
    """
    leaf_positions = tree.leaf_positions

    # Step 1: Weight sparsification (vectorized)
    leaf_weights = vectorized_weight_sparsification(
        prequeried_leaf_weights, max_weight=max_weight
    )

    # Step 2: Hierarchical variance (scan, no recursion)
    variances = compute_hierarchical_variance(leaf_weights, tree)

    # Step 3: Compute active mask
    active_mask = compute_active_leaf_mask(
        variances, tree, initial_depth, division_threshold, variance_threshold
    )

    # Step 4: Band detection using pre-queried neighbor weights
    connection_mask = vectorized_band_detection_prequery(
        leaf_weights, prequeried_neighbor_weights,
        active_mask, band_threshold, max_weight
    )

    # Step 5: Y-constraint filter (vectorized)
    if outgoing:
        y_valid = source_coord[1] <= leaf_positions[:, 1]
    else:
        y_valid = leaf_positions[:, 1] <= source_coord[1]

    # Exclude self-connections
    not_self = ~(
        (jnp.abs(leaf_positions[:, 0] - source_coord[0]) < 1e-6) &
        (jnp.abs(leaf_positions[:, 1] - source_coord[1]) < 1e-6)
    )

    final_mask = connection_mask & y_valid & not_self

    # Step 6: Pack results with NaN padding (maintains static shape)
    discovered_positions = jnp.where(
        final_mask[:, None],
        leaf_positions,
        jnp.nan
    )
    discovered_weights = jnp.where(final_mask, leaf_weights, jnp.nan)

    return discovered_positions, discovered_weights, jnp.sum(final_mask)


# ============================================================================
# Hierarchical Variance Computation
# ============================================================================

def compute_hierarchical_variance(
    leaf_weights: jnp.ndarray,
    tree: DenseQuadtreeStructure,
) -> jnp.ndarray:
    """Compute variance for all nodes from leaves to root (bottom-up).

    Uses Python loop over levels (max 7 iterations) with JAX operations per level.
    This avoids JAX tracing issues with dynamic shapes inside fori_loop.

    The variance of a node is the variance of its 4 children's weights.
    Leaf nodes have variance 0 (no children).

    Args:
        leaf_weights: CPPN weights at leaf positions - shape (num_leaves,)
        tree: Pre-computed quadtree structure

    Returns:
        Variance for all nodes - shape (num_nodes,)
    """
    max_depth = tree.max_depth
    num_nodes = tree.num_nodes

    # Convert level_offsets to Python ints for indexing
    level_offsets_py = [int(tree.level_offsets[i]) for i in range(max_depth + 2)]

    # Initialize: all nodes get 0 variance
    # We'll store "representative weight" at each node (mean of children)
    variances = jnp.zeros(num_nodes)
    node_weights = jnp.zeros(num_nodes)

    # Leaves get their weights directly
    leaf_start = level_offsets_py[max_depth]
    node_weights = node_weights.at[leaf_start:].set(leaf_weights)

    # Process levels from bottom to top (max_depth-1 down to 0)
    # Using Python loop since max_depth <= 7 (small, fixed iteration count)
    for level in range(max_depth - 1, -1, -1):
        level_start = level_offsets_py[level]
        level_end = level_offsets_py[level + 1]
        num_nodes_at_level = level_end - level_start

        # Get indices of all nodes at this level
        node_indices = jnp.arange(level_start, level_end)

        # Compute child indices for each node
        # Children of node i are at: next_level_start + (i - level_start) * 4 + [0,1,2,3]
        next_level_start = level_offsets_py[level + 1]

        child_base = next_level_start + (node_indices - level_start) * 4
        child_indices = child_base[:, None] + jnp.arange(4)[None, :]  # (num_nodes, 4)

        # Get children weights
        child_weights = node_weights[child_indices]  # (num_nodes, 4)

        # Compute variance across 4 children
        node_variances = jnp.var(child_weights, axis=1)  # (num_nodes,)

        # Compute mean for parent's variance calculation
        node_means = jnp.mean(child_weights, axis=1)  # (num_nodes,)

        # Update arrays
        variances = variances.at[level_start:level_end].set(node_variances)
        node_weights = node_weights.at[level_start:level_end].set(node_means)

    return variances


def compute_active_leaf_mask(
    variances: jnp.ndarray,
    tree: DenseQuadtreeStructure,
    initial_depth: int,
    division_threshold: float,
    variance_threshold: float,
) -> jnp.ndarray:
    """Compute which leaf nodes should be checked for band discontinuity.

    A leaf is "active" (should check band) if we would have stopped subdividing
    at its parent due to:
    1. Parent at or above initial_depth AND parent variance <= division_threshold
    2. OR parent at max_depth (forced to stop)

    For variance_threshold in pruning: leaves with variance > variance_threshold
    are recursively explored; others are checked for band discontinuity.

    Since we're using a complete tree, ALL leaves at max_depth are candidates,
    but we use the parent's variance to determine if we should check band.

    Args:
        variances: Variance at each node - shape (num_nodes,)
        tree: Pre-computed quadtree structure
        initial_depth: Minimum subdivision depth (force subdivide until this)
        division_threshold: Variance threshold for continued subdivision
        variance_threshold: Variance threshold for extraction (band checking)

    Returns:
        Active mask for leaves - shape (num_leaves,)
    """
    max_depth = tree.max_depth
    level_offsets = tree.level_offsets

    # Parent level is max_depth - 1
    parent_level = max_depth - 1
    parent_start = level_offsets[parent_level]
    parent_end = level_offsets[parent_level + 1]

    # Get parent variances
    parent_variances = variances[parent_start:parent_end]  # (num_parents,)

    # Each parent has 4 leaf children
    # Expand to leaf mask: (num_parents, 4) -> (num_leaves,)
    leaf_parent_variances = jnp.repeat(parent_variances, 4)  # (num_leaves,)

    # In the working version, _variance(leaf) = 0 for single-point leaves
    # because variance of a single value is 0. Since 0 <= variance_threshold (0.03),
    # ALL leaves at max_depth automatically qualify for extraction/band-checking.
    #
    # The original bug was trying to filter based on parent variance, but that's
    # not how ES-HyperNEAT works. All max_depth leaves ARE the extraction candidates.
    # The band detection itself will filter out non-band positions.
    #
    # Making ALL leaves active (as they should be at max_depth):
    active = jnp.ones(tree.num_leaves, dtype=jnp.bool_)

    return active


# ============================================================================
# Vectorized Band Detection
# ============================================================================

def vectorized_band_detection(
    state: Any,
    cppn_transformed: Any,
    source_coord: jnp.ndarray,
    leaf_positions: jnp.ndarray,
    leaf_widths: jnp.ndarray,
    leaf_weights: jnp.ndarray,
    active_mask: jnp.ndarray,
    band_threshold: float,
    outgoing: bool,
    cppn_forward: Any,
) -> jnp.ndarray:
    """Vectorized band discontinuity detection across ALL leaves.

    For each active leaf, check if there's a band discontinuity by comparing
    the leaf weight with its 4 neighbors (left, right, top, bottom).

    Band formula: max(min(d_top, d_bottom), min(d_left, d_right)) > threshold

    Args:
        state: Algorithm state
        cppn_transformed: Transformed CPPN
        source_coord: Source coordinate - shape (2,)
        leaf_positions: Leaf node centers - shape (num_leaves, 2)
        leaf_widths: Parent widths for neighbor offset - shape (num_leaves,)
        leaf_weights: CPPN weights at leaves - shape (num_leaves,)
        active_mask: Which leaves to check - shape (num_leaves,)
        band_threshold: Threshold for band detection
        outgoing: Direction of connection query
        cppn_forward: JIT-compiled CPPN forward function

    Returns:
        Valid connection mask - shape (num_leaves,)
    """
    num_leaves = leaf_positions.shape[0]

    # Compute all neighbor positions: (num_leaves, 4, 2)
    # Neighbors are at: leaf_pos + NEIGHBOR_OFFSET * parent_width
    neighbor_positions = (
        leaf_positions[:, None, :] +
        NEIGHBOR_OFFSETS[None, :, :] * leaf_widths[:, None, None]
    )

    # Flatten for batch query: (num_leaves * 4, 2)
    flat_neighbor_positions = neighbor_positions.reshape(-1, 2)

    # Query CPPN for all neighbors in ONE call
    neighbor_weights_flat = batch_query_all_positions(
        state, cppn_transformed, source_coord,
        flat_neighbor_positions, outgoing, cppn_forward
    )

    # Apply same sparsification as leaf weights
    neighbor_weights_flat = vectorized_weight_sparsification(neighbor_weights_flat)

    # Reshape: (num_leaves, 4)
    neighbor_weights = neighbor_weights_flat.reshape(num_leaves, 4)

    # Compute differences
    d_left = jnp.abs(leaf_weights - neighbor_weights[:, 0])
    d_right = jnp.abs(leaf_weights - neighbor_weights[:, 1])
    d_top = jnp.abs(leaf_weights - neighbor_weights[:, 2])
    d_bottom = jnp.abs(leaf_weights - neighbor_weights[:, 3])

    # Band formula: max(min(d_top, d_bottom), min(d_left, d_right))
    band_value = jnp.maximum(
        jnp.minimum(d_top, d_bottom),
        jnp.minimum(d_left, d_right)
    )

    # Valid connection if:
    # 1. Active (parent variance low enough)
    # 2. Band exceeds threshold
    # 3. Weight is non-zero
    valid = (
        active_mask &
        (band_value > band_threshold) &
        (leaf_weights != 0.0)
    )

    return valid


# ============================================================================
# Complete Dense Discovery Pipeline
# ============================================================================

def dense_substrate_discovery(
    state: Any,
    cppn_transformed: Any,
    source_coord: jnp.ndarray,
    tree: DenseQuadtreeStructure,
    initial_depth: int,
    max_depth: int,
    variance_threshold: float,
    division_threshold: float,
    band_threshold: float,
    outgoing: bool,
    cppn_forward: Any,
    max_weight: float = 8.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-accelerated substrate discovery from a single source.

    Pipeline using vectorized JAX operations:
    1. Batch query ALL leaf positions (vmap)
    2. Vectorized weight sparsification
    3. Hierarchical variance computation (Python loop + JAX ops)
    4. Vectorized band detection
    5. Y-constraint filtering

    Args:
        state: Algorithm state
        cppn_transformed: Transformed CPPN
        source_coord: Source coordinate - shape (2,)
        tree: Pre-computed quadtree structure
        initial_depth: Minimum subdivision depth
        max_depth: Maximum subdivision depth
        variance_threshold: Threshold for extraction
        division_threshold: Threshold for subdivision
        band_threshold: Threshold for band detection
        outgoing: Direction (True=source->target, False=target->source)
        cppn_forward: JIT-compiled CPPN forward function
        max_weight: Maximum connection weight

    Returns:
        Tuple of:
        - discovered_positions: (num_leaves, 2) with NaN for invalid
        - discovered_weights: (num_leaves,) with NaN for invalid
        - num_valid: scalar count of valid connections
    """
    # Step 1: Batch query ALL leaf positions
    leaf_positions = tree.leaf_positions
    raw_weights = batch_query_all_positions(
        state, cppn_transformed, source_coord,
        leaf_positions, outgoing, cppn_forward
    )

    # Step 2: Weight sparsification (vectorized)
    leaf_weights = vectorized_weight_sparsification(raw_weights, max_weight=max_weight)

    # Step 3: Hierarchical variance (scan, no recursion)
    variances = compute_hierarchical_variance(leaf_weights, tree)

    # Step 4: Compute active mask
    active_mask = compute_active_leaf_mask(
        variances, tree, initial_depth, division_threshold, variance_threshold
    )

    # Step 5: Band detection (single batched neighbor query)
    connection_mask = vectorized_band_detection(
        state, cppn_transformed, source_coord,
        leaf_positions, tree.leaf_widths, leaf_weights,
        active_mask, band_threshold, outgoing, cppn_forward
    )

    # Step 6: Y-constraint filter (vectorized)
    # Outgoing: source.y < target.y (upward connections)
    # Incoming: target.y < source.y
    if outgoing:
        y_valid = source_coord[1] <= leaf_positions[:, 1]
    else:
        y_valid = leaf_positions[:, 1] <= source_coord[1]

    # Exclude self-connections
    not_self = ~(
        (jnp.abs(leaf_positions[:, 0] - source_coord[0]) < 1e-6) &
        (jnp.abs(leaf_positions[:, 1] - source_coord[1]) < 1e-6)
    )

    final_mask = connection_mask & y_valid & not_self

    # Step 7: Pack results with NaN padding (maintains static shape)
    discovered_positions = jnp.where(
        final_mask[:, None],
        leaf_positions,
        jnp.nan
    )
    discovered_weights = jnp.where(final_mask, leaf_weights, jnp.nan)

    return discovered_positions, discovered_weights, jnp.sum(final_mask)


def batch_discover_from_sources(
    state: Any,
    cppn_transformed: Any,
    source_coords: jnp.ndarray,
    tree: DenseQuadtreeStructure,
    initial_depth: int,
    max_depth: int,
    variance_threshold: float,
    division_threshold: float,
    band_threshold: float,
    outgoing: bool,
    cppn_forward: Any,
    max_weight: float = 8.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batch discovery from multiple source coordinates using vmap.

    Args:
        source_coords: Multiple source coordinates - shape (num_sources, 2)
        ... other args same as dense_substrate_discovery

    Returns:
        Tuple of:
        - discovered_positions: (num_sources, num_leaves, 2)
        - discovered_weights: (num_sources, num_leaves)
        - num_valid: (num_sources,)
    """
    # Note: We can't vmap over tree (static), but we can vmap over source_coords
    # Need non-jitted version for vmap
    def discover_single(source_coord):
        return dense_substrate_discovery(
            state, cppn_transformed, source_coord, tree,
            initial_depth, max_depth, variance_threshold, division_threshold,
            band_threshold, outgoing, cppn_forward, max_weight
        )

    return jax.vmap(discover_single)(source_coords)


# ============================================================================
# Connection Class (for compatibility with network building)
# ============================================================================

class Connection:
    """Connection between two spatial coordinates with weight.

    Kept for compatibility with _clean_net and substrate building.
    """
    def __init__(self, x1: float, y1: float, x2: float, y2: float, weight: float):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.weight = float(weight) if not math.isnan(float(weight)) else 0.0

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.x1, self.y1, self.x2, self.y2) == (other.x1, other.y1, other.x2, other.y2)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2))


# ============================================================================
# JAX-Native Connection Array Functions
# ============================================================================

def collect_valid_connections(
    source_coord: jnp.ndarray,
    positions: jnp.ndarray,
    weights: jnp.ndarray,
    outgoing: bool = True,
) -> np.ndarray:
    """Extract valid connections from discovery results as numpy array.

    This replaces the Python loop + Connection object creation pattern.

    Args:
        source_coord: Source coordinate (2,)
        positions: Discovered target positions (N, 2) with NaN for invalid
        weights: Discovered weights (N,) with NaN for invalid
        outgoing: If True, source->target; if False, target->source

    Returns:
        Array of valid connections (M, 5) with [x1, y1, x2, y2, weight]
    """
    # Convert to numpy for faster indexing
    positions_np = np.asarray(positions)
    weights_np = np.asarray(weights)
    source_np = np.asarray(source_coord)

    # Find valid (non-NaN) entries
    valid_mask = ~np.isnan(positions_np[:, 0])
    num_valid = np.sum(valid_mask)

    if num_valid == 0:
        return np.zeros((0, 5), dtype=np.float32)

    valid_positions = positions_np[valid_mask]
    valid_weights = weights_np[valid_mask]

    # Build connections array
    connections = np.zeros((num_valid, 5), dtype=np.float32)

    if outgoing:
        # source -> target
        connections[:, 0] = source_np[0]  # x1
        connections[:, 1] = source_np[1]  # y1
        connections[:, 2] = valid_positions[:, 0]  # x2
        connections[:, 3] = valid_positions[:, 1]  # y2
    else:
        # target -> source (reverse for incoming)
        connections[:, 0] = valid_positions[:, 0]  # x1
        connections[:, 1] = valid_positions[:, 1]  # y1
        connections[:, 2] = source_np[0]  # x2
        connections[:, 3] = source_np[1]  # y2

    connections[:, 4] = valid_weights

    return connections


def extract_hidden_nodes_from_connections(
    connections: np.ndarray,
    input_coords: np.ndarray,
    output_coords: np.ndarray,
) -> np.ndarray:
    """Extract hidden node coordinates from connections array.

    Args:
        connections: (N, 5) array of connections [x1, y1, x2, y2, weight]
        input_coords: (I, 2) array of input coordinates
        output_coords: (O, 2) array of output coordinates

    Returns:
        (H, 2) array of unique hidden node coordinates
    """
    if len(connections) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Get all source and target coordinates
    sources = connections[:, :2]
    targets = connections[:, 2:4]
    all_coords = np.vstack([sources, targets])

    # Round for comparison
    all_coords_r = np.round(all_coords, 5)
    unique_coords = np.unique(all_coords_r, axis=0)

    # Filter out inputs and outputs
    def coords_in_set(coords, coord_set, tol=1e-4):
        if len(coord_set) == 0:
            return np.zeros(len(coords), dtype=bool)
        diffs = np.abs(coords[:, None, :] - coord_set[None, :, :])
        distances = np.sum(diffs, axis=2)
        return np.any(distances < tol, axis=1)

    is_input = coords_in_set(unique_coords, np.round(input_coords, 5))
    is_output = coords_in_set(unique_coords, np.round(output_coords, 5))
    is_hidden = ~(is_input | is_output)

    return unique_coords[is_hidden]


def deduplicate_connections(connections: np.ndarray) -> np.ndarray:
    """Remove duplicate connections (same x1, y1, x2, y2), keeping first.

    Args:
        connections: (N, 5) array of connections

    Returns:
        (M, 5) array with duplicates removed
    """
    if len(connections) == 0:
        return connections

    # Round coordinates for comparison
    coords = np.round(connections[:, :4], 5)

    # Find unique rows
    _, unique_indices = np.unique(coords, axis=0, return_index=True)

    # Sort to maintain original order
    unique_indices = np.sort(unique_indices)

    return connections[unique_indices]


def build_substrate_from_arrays(
    conn_array: np.ndarray,
    input_coords: np.ndarray,
    output_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build TensorNEAT-compatible substrate from connection array.

    This replaces the Python-based _build_tensorneat_substrate with arrays.

    Args:
        conn_array: (N, 5) array of connections [x1, y1, x2, y2, weight]
        input_coords: (I, 2) input coordinates
        output_coords: (O, 2) output coordinates

    Returns:
        Tuple of:
        - nodes: (num_nodes, 1) node indices
        - conns: (M, 3) connections [from_idx, to_idx, weight]
    """
    num_inputs = len(input_coords)
    num_outputs = len(output_coords)

    if len(conn_array) == 0:
        # Minimal fallback: direct input->output connections
        num_nodes = num_inputs + num_outputs
        nodes = np.arange(num_nodes).reshape(-1, 1).astype(np.float32)

        # Random weights for fallback
        np.random.seed(42)
        conns = []
        for i in range(num_inputs):
            for o in range(num_outputs):
                conns.append([i, num_inputs + o, np.random.uniform(-0.5, 0.5)])
        conns = np.array(conns, dtype=np.float32)
        return nodes, conns

    # Build coordinate to index mapping
    # Inputs: indices 0 to num_inputs-1
    # Hidden: indices num_inputs to num_inputs+num_hidden-1
    # Outputs: last num_outputs indices

    # Collect all unique coordinates
    sources = conn_array[:, :2]
    targets = conn_array[:, 2:4]
    all_coords = np.vstack([sources, targets])
    all_coords_r = np.round(all_coords, 5)

    # Unique coordinates
    unique_coords, inverse = np.unique(all_coords_r, axis=0, return_inverse=True)

    # Classify coordinates
    def match_coords(coords, ref_coords, tol=1e-4):
        """Return indices of coords that match any ref_coord, or -1."""
        if len(ref_coords) == 0:
            return np.full(len(coords), -1, dtype=np.int32)
        diffs = np.abs(coords[:, None, :] - ref_coords[None, :, :])
        distances = np.sum(diffs, axis=2)
        matches = np.argmin(distances, axis=1)
        is_match = np.min(distances, axis=1) < tol
        return np.where(is_match, matches, -1)

    input_match = match_coords(unique_coords, np.round(input_coords, 5))
    output_match = match_coords(unique_coords, np.round(output_coords, 5))

    # Assign indices
    # coord_to_idx: unique_idx -> final node index
    coord_to_idx = np.full(len(unique_coords), -1, dtype=np.int32)

    # Inputs keep their original indices
    for i, m in enumerate(input_match):
        if m >= 0:
            coord_to_idx[i] = m

    # Hidden nodes: next indices after inputs
    hidden_idx = num_inputs
    for i in range(len(unique_coords)):
        if input_match[i] < 0 and output_match[i] < 0:
            coord_to_idx[i] = hidden_idx
            hidden_idx += 1

    num_hidden = hidden_idx - num_inputs

    # Outputs: after hidden
    output_start = num_inputs + num_hidden
    for i, m in enumerate(output_match):
        if m >= 0:
            coord_to_idx[i] = output_start + m

    num_nodes = num_inputs + num_hidden + num_outputs

    # Build nodes array
    nodes = np.arange(num_nodes).reshape(-1, 1).astype(np.float32)

    # Build connections array
    # Map source and target coordinates to indices
    num_conns = len(conn_array)
    source_unique_idx = inverse[:num_conns]
    target_unique_idx = inverse[num_conns:]

    from_indices = coord_to_idx[source_unique_idx]
    to_indices = coord_to_idx[target_unique_idx]
    weights = conn_array[:, 4]

    # Filter out invalid connections
    valid = (from_indices >= 0) & (to_indices >= 0)
    conns = np.stack([from_indices[valid], to_indices[valid], weights[valid]], axis=1)

    if len(conns) == 0:
        # Fallback if all connections filtered
        conns = []
        np.random.seed(42)
        for i in range(num_inputs):
            for o in range(num_outputs):
                conns.append([i, output_start + o, np.random.uniform(-0.5, 0.5)])
        conns = np.array(conns, dtype=np.float32)

    return nodes.astype(np.float32), conns.astype(np.float32)


def connections_to_array(connections: Set[Connection]) -> jnp.ndarray:
    """Convert Python Connection set to JAX array.

    Args:
        connections: Set of Connection objects

    Returns:
        JAX array of shape (N, 5) with [x1, y1, x2, y2, weight]
    """
    if len(connections) == 0:
        return jnp.zeros((0, 5), dtype=jnp.float32)

    conn_list = [[c.x1, c.y1, c.x2, c.y2, c.weight] for c in connections]
    return jnp.array(conn_list, dtype=jnp.float32)


def array_to_connections(conn_array: jnp.ndarray) -> Set[Connection]:
    """Convert JAX array back to Connection set.

    Args:
        conn_array: JAX array of shape (N, 5) with [x1, y1, x2, y2, weight]

    Returns:
        Set of Connection objects
    """
    connections = set()
    if conn_array.shape[0] == 0:
        return connections

    conn_np = np.array(conn_array)
    for row in conn_np:
        if not np.isnan(row[0]):  # Valid connection
            conn = Connection(row[0], row[1], row[2], row[3], row[4])
            connections.add(conn)
    return connections


def clean_connections_numpy(
    conn_array: np.ndarray,
    input_coords: np.ndarray,
    output_coords: np.ndarray,
    max_iterations: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy-accelerated network cleaning using graph reachability.

    This is significantly faster than the Python set-based approach because:
    1. Uses NumPy broadcasting for coordinate matching (vectorized)
    2. Uses boolean array operations instead of set operations

    Algorithm:
    1. Build adjacency information (source/target indices)
    2. Forward propagation from inputs
    3. Backward propagation from outputs
    4. Keep connections in intersection

    Args:
        conn_array: Connections as (N, 5) array [x1, y1, x2, y2, weight]
        input_coords: Input coordinates as (num_inputs, 2) array
        output_coords: Output coordinates as (num_outputs, 2) array
        max_iterations: Maximum iterations for reachability propagation

    Returns:
        Tuple of:
        - cleaned_connections: (M, 5) array of valid connections
        - hidden_nodes: (K, 2) array of valid hidden node coordinates
    """
    if len(conn_array) == 0:
        return conn_array, np.zeros((0, 2), dtype=np.float32)

    num_conns = len(conn_array)
    sources = conn_array[:, :2]  # (N, 2)
    targets = conn_array[:, 2:4]  # (N, 2)

    # Round for floating point comparison
    sources_r = np.round(sources, 5)
    targets_r = np.round(targets, 5)
    inputs_r = np.round(input_coords, 5)
    outputs_r = np.round(output_coords, 5)

    # Helper: check if coords match any in set (vectorized)
    def coords_in_set(coords, coord_set, tol=1e-4):
        """Check which coords are in coord_set. Returns (N,) bool array."""
        # coords: (N, 2), coord_set: (M, 2)
        # Use broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
        diffs = np.abs(coords[:, None, :] - coord_set[None, :, :])
        distances = np.sum(diffs, axis=2)  # (N, M)
        return np.any(distances < tol, axis=1)  # (N,)

    # Forward reachability: which connections are reachable from inputs?
    # A connection is reachable if its source is reachable
    forward_reachable = coords_in_set(sources_r, inputs_r)  # (N,)

    # Propagate forward: if source of conn_i matches target of a reachable conn
    for _ in range(max_iterations):
        if np.all(forward_reachable):
            break

        # Get targets of reachable connections
        reachable_targets = targets_r[forward_reachable]  # (K, 2)
        if len(reachable_targets) == 0:
            break

        # Check which non-reachable connections have sources in reachable_targets
        not_reachable = ~forward_reachable
        non_reachable_sources = sources_r[not_reachable]

        # Find matches: (num_not_reachable, num_reachable_targets)
        if len(non_reachable_sources) > 0:
            new_reachable = coords_in_set(non_reachable_sources, reachable_targets)
            # Update: mark newly reachable connections
            not_reachable_indices = np.where(not_reachable)[0]
            forward_reachable[not_reachable_indices[new_reachable]] = True
        else:
            break

    # Backward reachability: which connections can reach outputs?
    backward_reachable = coords_in_set(targets_r, outputs_r)  # (N,)

    # Propagate backward: if target of conn_i matches source of a backward-reachable conn
    for _ in range(max_iterations):
        if np.all(backward_reachable):
            break

        # Get sources of backward-reachable connections
        reachable_sources = sources_r[backward_reachable]  # (K, 2)
        if len(reachable_sources) == 0:
            break

        # Check which non-reachable connections have targets in reachable_sources
        not_reachable = ~backward_reachable
        non_reachable_targets = targets_r[not_reachable]

        if len(non_reachable_targets) > 0:
            new_reachable = coords_in_set(non_reachable_targets, reachable_sources)
            not_reachable_indices = np.where(not_reachable)[0]
            backward_reachable[not_reachable_indices[new_reachable]] = True
        else:
            break

    # Valid connections: reachable from inputs AND can reach outputs
    valid_mask = forward_reachable & backward_reachable
    cleaned_connections = conn_array[valid_mask]

    # Extract hidden nodes
    if len(cleaned_connections) > 0:
        valid_sources = cleaned_connections[:, :2]
        valid_targets = cleaned_connections[:, 2:4]
        all_nodes = np.vstack([valid_sources, valid_targets])

        # Get unique nodes (round for comparison)
        all_nodes_r = np.round(all_nodes, 5)
        unique_nodes = np.unique(all_nodes_r, axis=0)

        # Filter out inputs and outputs
        is_input = coords_in_set(unique_nodes, inputs_r)
        is_output = coords_in_set(unique_nodes, outputs_r)
        hidden_mask = ~(is_input | is_output)
        hidden_nodes = unique_nodes[hidden_mask]
    else:
        hidden_nodes = np.zeros((0, 2), dtype=np.float32)

    return cleaned_connections, hidden_nodes


def discover_substrate_arrays(
    state: Any,
    cppn_transformed: Any,
    tree: DenseQuadtreeStructure,
    input_coords: np.ndarray,
    output_coords: np.ndarray,
    initial_depth: int,
    max_depth: int,
    variance_threshold: float,
    division_threshold: float,
    band_threshold: float,
    max_weight: float,
    iteration_level: int,
    cppn_forward: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Array-based ES-HyperNEAT substrate discovery.

    This replaces the set-based _discover_substrate_es_jax with pure arrays.
    No Connection objects are created; everything stays as numpy arrays.

    Args:
        state: Algorithm state
        cppn_transformed: Transformed CPPN
        tree: Pre-computed quadtree structure
        input_coords: (I, 2) input coordinates
        output_coords: (O, 2) output coordinates
        initial_depth, max_depth, variance_threshold, etc.: ES-HyperNEAT params
        cppn_forward: JIT-compiled CPPN forward function

    Returns:
        Tuple of:
        - nodes: (num_nodes, 1) array of node indices
        - conns: (M, 3) array of [from_idx, to_idx, weight]
    """
    all_connections = []

    # Explore from inputs
    hidden_coords = []

    for i in range(len(input_coords)):
        source_coord = jnp.array(input_coords[i], dtype=jnp.float32)

        positions, weights, num_valid = dense_substrate_discovery(
            state, cppn_transformed, source_coord, tree,
            initial_depth, max_depth, variance_threshold, division_threshold,
            band_threshold, True, cppn_forward, max_weight
        )

        # Collect valid connections as array
        conns = collect_valid_connections(source_coord, positions, weights, outgoing=True)
        if len(conns) > 0:
            all_connections.append(conns)
            # Extract hidden coordinates (targets that aren't outputs)
            targets = conns[:, 2:4]
            hidden_coords.append(targets)

    # Fallback if no connections found
    if len(all_connections) == 0:
        center = np.array([[0.0, 0.0]], dtype=np.float32)
        hidden_coords = [center]
        for i in range(len(input_coords)):
            conn = np.array([[input_coords[i, 0], input_coords[i, 1], 0.0, 0.0, 0.5]], dtype=np.float32)
            all_connections.append(conn)

    # Combine Phase 1 hidden nodes
    if hidden_coords:
        hidden_set = np.vstack(hidden_coords)
        hidden_set = np.unique(np.round(hidden_set, 5), axis=0)
    else:
        hidden_set = np.zeros((0, 2), dtype=np.float32)

    unexplored = hidden_set.copy()

    # Explore from hidden nodes (iteration_level times)
    for iteration in range(iteration_level):
        if len(unexplored) == 0:
            break

        new_hidden = []

        for i in range(len(unexplored)):
            source_coord = jnp.array(unexplored[i], dtype=jnp.float32)

            positions, weights, num_valid = dense_substrate_discovery(
                state, cppn_transformed, source_coord, tree,
                initial_depth, max_depth, variance_threshold, division_threshold,
                band_threshold, True, cppn_forward, max_weight
            )

            conns = collect_valid_connections(source_coord, positions, weights, outgoing=True)
            if len(conns) > 0:
                all_connections.append(conns)
                targets = conns[:, 2:4]
                new_hidden.append(targets)

        if new_hidden:
            new_hidden_arr = np.vstack(new_hidden)
            new_hidden_arr = np.unique(np.round(new_hidden_arr, 5), axis=0)

            # Filter out already known hidden nodes
            def coords_in_set(coords, coord_set, tol=1e-4):
                if len(coord_set) == 0:
                    return np.zeros(len(coords), dtype=bool)
                diffs = np.abs(coords[:, None, :] - coord_set[None, :, :])
                distances = np.sum(diffs, axis=2)
                return np.any(distances < tol, axis=1)

            already_known = coords_in_set(new_hidden_arr, hidden_set)
            unexplored = new_hidden_arr[~already_known]

            if len(unexplored) > 0:
                hidden_set = np.vstack([hidden_set, unexplored])
                hidden_set = np.unique(np.round(hidden_set, 5), axis=0)
        else:
            unexplored = np.zeros((0, 2), dtype=np.float32)

    # Explore to outputs
    for i in range(len(output_coords)):
        source_coord = jnp.array(output_coords[i], dtype=jnp.float32)

        positions, weights, num_valid = dense_substrate_discovery(
            state, cppn_transformed, source_coord, tree,
            initial_depth, max_depth, variance_threshold, division_threshold,
            band_threshold, False, cppn_forward, max_weight
        )

        conns = collect_valid_connections(source_coord, positions, weights, outgoing=False)
        if len(conns) > 0:
            all_connections.append(conns)

    # Combine all connections
    if all_connections:
        combined = np.vstack(all_connections)
        # Deduplicate
        combined = deduplicate_connections(combined)
    else:
        combined = np.zeros((0, 5), dtype=np.float32)

    # Fallback: add output connections if missing
    if len(combined) > 0:
        targets = combined[:, 2:4]
        targets_r = np.round(targets, 5)
        outputs_r = np.round(output_coords, 5)

        def coords_in_set(coords, coord_set, tol=1e-4):
            if len(coord_set) == 0:
                return np.zeros(len(coords), dtype=bool)
            diffs = np.abs(coords[:, None, :] - coord_set[None, :, :])
            distances = np.sum(diffs, axis=2)
            return np.any(distances < tol, axis=1)

        has_output = np.any(coords_in_set(targets_r, outputs_r))

        if not has_output and len(hidden_set) > 0:
            # Find nearest hidden to each output
            for o in range(len(output_coords)):
                dists = np.sum((hidden_set - output_coords[o:o+1]) ** 2, axis=1)
                nearest_idx = np.argmin(dists)
                fallback_conn = np.array([[
                    hidden_set[nearest_idx, 0], hidden_set[nearest_idx, 1],
                    output_coords[o, 0], output_coords[o, 1], 0.5
                ]], dtype=np.float32)
                combined = np.vstack([combined, fallback_conn])

    # Clean network
    cleaned, _ = clean_connections_numpy(combined, input_coords, output_coords)

    # Build substrate
    nodes, conns = build_substrate_from_arrays(cleaned, input_coords, output_coords)

    return jnp.array(nodes), jnp.array(conns)


# ============================================================================
# Population-Level Parallelism - Padded Discovery for vmap
# ============================================================================

# Default padding sizes for fixed-shape arrays (enables vmap over population)
DEFAULT_MAX_NODES = 50      # Max nodes per substrate (inputs + hidden + outputs)
DEFAULT_MAX_CONNECTIONS = 100  # Max connections per substrate


def discover_substrate_padded(
    state: Any,
    cppn_transformed: Any,
    tree: DenseQuadtreeStructure,
    input_coords: np.ndarray,
    output_coords: np.ndarray,
    initial_depth: int,
    max_depth: int,
    variance_threshold: float,
    division_threshold: float,
    band_threshold: float,
    max_weight: float,
    iteration_level: int,
    cppn_forward: Any,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]:
    """Padded substrate discovery for vmap compatibility.

    Wraps discover_substrate_arrays and pads outputs to fixed sizes.
    Invalid entries are filled with NaN for weights and -1 for indices.

    Args:
        ... same as discover_substrate_arrays ...
        max_nodes: Maximum nodes to pad to
        max_connections: Maximum connections to pad to

    Returns:
        Tuple of:
        - padded_nodes: (max_nodes, 1) padded node indices (-1 for invalid)
        - padded_conns: (max_connections, 3) padded [from, to, weight] (NaN weight for invalid)
        - node_valid_mask: (max_nodes,) bool mask for valid nodes
        - conn_valid_mask: (max_connections,) bool mask for valid connections
        - num_nodes: actual number of valid nodes
        - num_conns: actual number of valid connections
    """
    # Run actual discovery
    nodes, conns = discover_substrate_arrays(
        state, cppn_transformed, tree,
        input_coords, output_coords,
        initial_depth, max_depth,
        variance_threshold, division_threshold,
        band_threshold, max_weight,
        iteration_level, cppn_forward
    )

    # Convert to numpy for manipulation
    nodes_np = np.asarray(nodes)
    conns_np = np.asarray(conns)

    num_nodes = len(nodes_np)
    num_conns = len(conns_np)

    # Pad nodes to fixed size
    padded_nodes = np.full((max_nodes, 1), -1.0, dtype=np.float32)
    if num_nodes > 0:
        actual_num = min(num_nodes, max_nodes)
        padded_nodes[:actual_num] = nodes_np[:actual_num]
    node_valid_mask = np.zeros(max_nodes, dtype=bool)
    node_valid_mask[:min(num_nodes, max_nodes)] = True

    # Pad connections to fixed size
    padded_conns = np.full((max_connections, 3), np.nan, dtype=np.float32)
    if num_conns > 0:
        actual_num = min(num_conns, max_connections)
        padded_conns[:actual_num] = conns_np[:actual_num]
    conn_valid_mask = np.zeros(max_connections, dtype=bool)
    conn_valid_mask[:min(num_conns, max_connections)] = True

    return (
        jnp.array(padded_nodes),
        jnp.array(padded_conns),
        jnp.array(node_valid_mask),
        jnp.array(conn_valid_mask),
        min(num_nodes, max_nodes),
        min(num_conns, max_connections),
    )


def forward_hyperneat_padded(
    nodes: jnp.ndarray,
    conns: jnp.ndarray,
    conn_valid_mask: jnp.ndarray,
    inputs: jnp.ndarray,
    num_inputs: int,
    num_outputs: int,
    activate_time: int,
) -> jnp.ndarray:
    """Forward pass with padded connections for vmap compatibility.

    This is a JAX-traceable version of _forward_hyperneat_style that works
    with padded arrays and validity masks.

    Args:
        nodes: (max_nodes, 1) padded node indices
        conns: (max_connections, 3) padded [from_idx, to_idx, weight]
        conn_valid_mask: (max_connections,) bool mask for valid connections
        inputs: (num_inputs,) input values
        num_inputs: number of input nodes
        num_outputs: number of output nodes
        activate_time: number of forward pass iterations

    Returns:
        (num_outputs,) output values after sigmoid
    """
    num_nodes = nodes.shape[0]
    output_start_idx = num_nodes - num_outputs

    # Extract connection components
    from_indices = conns[:, 0].astype(jnp.int32)
    to_indices = conns[:, 1].astype(jnp.int32)
    weights = conns[:, 2]

    # Combine NaN check with validity mask
    valid_weights = ~jnp.isnan(weights)
    valid_mask = conn_valid_mask & valid_weights

    # Clamp indices to valid range (JAX doesn't support negative indexing in scatter)
    safe_from = jnp.clip(from_indices, 0, num_nodes - 1)
    safe_to = jnp.clip(to_indices, 0, num_nodes - 1)

    # Initialize values
    vals = jnp.zeros(num_nodes)
    vals = vals.at[:num_inputs].set(inputs)

    # Forward pass iterations (fixed number for JIT)
    def forward_step(vals, _):
        new_vals = jnp.zeros(num_nodes)
        new_vals = new_vals.at[:num_inputs].set(inputs)

        # Aggregate weighted inputs (only for valid connections)
        aggregated = jnp.zeros(num_nodes)

        # Use where to zero out invalid connection contributions
        effective_weights = jnp.where(valid_mask, weights, 0.0)
        contributions = vals[safe_from] * effective_weights

        # Scatter-add contributions
        aggregated = aggregated.at[safe_to].add(contributions)

        # Apply activations
        # Hidden: tanh
        if output_start_idx > num_inputs:
            hidden_vals = jnp.tanh(aggregated[num_inputs:output_start_idx])
            new_vals = new_vals.at[num_inputs:output_start_idx].set(hidden_vals)

        # Output: no activation yet (applied after)
        output_vals = aggregated[output_start_idx:]
        new_vals = new_vals.at[output_start_idx:].set(output_vals)

        return new_vals, None

    # Run fixed iterations
    vals, _ = lax.scan(forward_step, vals, None, length=activate_time)

    # Final outputs with sigmoid
    raw_outputs = vals[-num_outputs:]
    return jax.nn.sigmoid(raw_outputs)


def evaluate_genome_padded(
    nodes: jnp.ndarray,
    conns: jnp.ndarray,
    conn_valid_mask: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    num_inputs: int,
    num_outputs: int,
    activate_time: int,
) -> float:
    """Evaluate a single genome with padded arrays.

    Args:
        nodes: (max_nodes, 1) padded nodes
        conns: (max_connections, 3) padded connections
        conn_valid_mask: (max_connections,) validity mask
        inputs_batch: (num_cases, num_inputs) batch of inputs
        targets_batch: (num_cases, num_outputs) batch of targets
        num_inputs, num_outputs: sizes
        activate_time: forward pass iterations

    Returns:
        Fitness score (1.0 - MSE)
    """
    # vmap forward pass over test cases
    def forward_single(inputs):
        return forward_hyperneat_padded(
            nodes, conns, conn_valid_mask, inputs,
            num_inputs, num_outputs, activate_time
        )

    outputs_batch = jax.vmap(forward_single)(inputs_batch)

    # Compute MSE fitness
    errors = jnp.mean((outputs_batch - targets_batch) ** 2, axis=1)
    avg_error = jnp.mean(errors)

    return jnp.maximum(0.0, 1.0 - avg_error)


# ============================================================================
# Integration - HMRHyperNEATAdaptiveChunking Class
# ============================================================================

class HMRHyperNEATAdaptiveChunking(BaseAlgorithm):
    """HMR-HyperNEAT with ADAPTIVE POPULATION CHUNKING.

    This is an extension of HMRHyperNEAT that adds intelligent
    auto-selection of chunk sizes based on empirical benchmark data.

    KEY FEATURE: Adaptive Chunking
    ==============================
    The optimal chunk size follows a NON-MONOTONIC relationship with position count:

    | Depth | Positions | Optimal Chunk | Speedup vs Worst |
    |-------|-----------|---------------|------------------|
    | 3     | 340       | 100           | 1.7x             |
    | 4     | 1,364     | 500           | 1.2x             |
    | 5     | 5,460     | 500           | 1.4x             |
    | 6     | 21,844    | 10            | 1.7x (ANOMALY!)  |
    | 7     | 87,380    | 200           | 1.9x             |
    | 8     | 349,524   | 50            | 4.9x             |

    The "depth 6 anomaly" is caused by XLA compilation behavior changes
    in the 10K-50K position range. This implementation automatically
    selects optimal chunk sizes to avoid this performance cliff.

    Configuration:
    - population_chunk_size: -1 (auto, default), 0 (no chunking), or manual value

    Execution Modes:
    - run_generation(): GPU-resident single generation (DEFAULT, recommended)
    - run_generation_verbose(): Python loop with per-step timing (for debugging)
    - run_until_threshold(): GPU-resident multi-generation with early stopping
    """

    def __init__(self, name: str = 'hmr-hyperneat',
                 implementation: str = 'tensorneat-hmrhyperneat-adaptive'):
        super().__init__(name=name, implementation=implementation)
        self.adapter = TensorNEATAdapter()
        self.lazy_metrics = True

        # ES-HyperNEAT parameters (set via create_config)
        self.initial_depth = None
        self.max_depth = None
        self.variance_threshold = None
        self.division_threshold = None
        self.band_threshold = None
        self.max_weight = None
        self.iteration_level = None

        # Substrate coordinates
        self.substrate_input_coords = None
        self.substrate_output_coords = None
        # OPTIMIZATION: Cached JAX arrays for coordinates (avoid per-generation conversion)
        self._cached_input_coords = None
        self._cached_output_coords = None

        # Pre-computed quadtree structure (cached per max_depth)
        self._quadtree: Optional[DenseQuadtreeStructure] = None

        # NEAT algorithm for CPPN evolution
        self.neat_algo = None
        self.pipeline = None
        self.hyper_genome = None

        # JIT-compiled functions
        self._jitted_cppn_forward = None
        self._compiled_ask = None
        self._compiled_transform_batch = None
        self._compiled_tell = None

        # Metrics
        self._config_metadata = None
        self._start_time = None
        self.verbose = False
        self._cppn_query_count = 0

        # Execution mode parameters (set via create_config from hmr_hyperneat section)
        self.sparse_forward_threshold = 0  # Default: always use sparse
        self.extra_randkey_split = True    # Default: adds extra random key split

    # ========================================================================
    # ES-HyperNEAT Discovery (JAX-accelerated)
    # ========================================================================

    def _discover_substrate_es_jax(
        self, state: Any, cppn_transformed: Any
    ) -> Tuple[Set, Set, Dict]:
        """Three-phase ES-HyperNEAT substrate discovery (JAX-accelerated).

        Returns:
            Tuple of (hidden_nodes, connections, phase_info)
        """
        hidden_nodes = set()
        connections1, connections2, connections3 = set(), set(), set()

        tree = self._quadtree

        # Explore from inputs
        input_coords_jax = jnp.array(self.substrate_input_coords, dtype=jnp.float32)

        for i, coord in enumerate(self.substrate_input_coords):
            source_coord = jnp.array(coord, dtype=jnp.float32)

            # Dense discovery
            positions, weights, num_valid = dense_substrate_discovery(
                state, cppn_transformed, source_coord, tree,
                self.initial_depth, self.max_depth,
                self.variance_threshold, self.division_threshold,
                self.band_threshold, True, self._jitted_cppn_forward,
                self.max_weight
            )

            # Convert to Python connections (for network building)
            positions_np = np.array(positions)
            weights_np = np.array(weights)

            for j in range(len(positions_np)):
                if not np.isnan(positions_np[j, 0]):
                    conn = Connection(
                        x1=float(coord[0]), y1=float(coord[1]),
                        x2=float(positions_np[j, 0]), y2=float(positions_np[j, 1]),
                        weight=float(weights_np[j])
                    )
                    connections1.add(conn)
                    hidden_nodes.add((float(positions_np[j, 0]), float(positions_np[j, 1])))

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)

        # Fallback if no connections found
        if len(connections1) == 0:
            center = (0.0, 0.0)
            hidden_nodes.add(center)
            for input_coord in self.substrate_input_coords:
                conn = Connection(
                    x1=input_coord[0], y1=input_coord[1],
                    x2=center[0], y2=center[1], weight=0.5
                )
                connections1.add(conn)
            unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)

        # Explore from hidden nodes (iteration_level times)
        for iteration in range(self.iteration_level):
            new_hidden_nodes = set()
            for coord in unexplored_hidden_nodes:
                source_coord = jnp.array(coord, dtype=jnp.float32)

                positions, weights, num_valid = dense_substrate_discovery(
                    state, cppn_transformed, source_coord, tree,
                    self.initial_depth, self.max_depth,
                    self.variance_threshold, self.division_threshold,
                    self.band_threshold, True, self._jitted_cppn_forward,
                    self.max_weight
                )

                positions_np = np.array(positions)
                weights_np = np.array(weights)

                for j in range(len(positions_np)):
                    if not np.isnan(positions_np[j, 0]):
                        target = (float(positions_np[j, 0]), float(positions_np[j, 1]))
                        conn = Connection(
                            x1=coord[0], y1=coord[1],
                            x2=target[0], y2=target[1],
                            weight=float(weights_np[j])
                        )
                        connections2.add(conn)
                        if target not in hidden_nodes:
                            new_hidden_nodes.add(target)

            hidden_nodes.update(new_hidden_nodes)
            unexplored_hidden_nodes = hidden_nodes - unexplored_hidden_nodes

        # Explore to outputs
        for coord in self.substrate_output_coords:
            source_coord = jnp.array(coord, dtype=jnp.float32)

            positions, weights, num_valid = dense_substrate_discovery(
                state, cppn_transformed, source_coord, tree,
                self.initial_depth, self.max_depth,
                self.variance_threshold, self.division_threshold,
                self.band_threshold, False, self._jitted_cppn_forward,
                self.max_weight
            )

            positions_np = np.array(positions)
            weights_np = np.array(weights)

            for j in range(len(positions_np)):
                if not np.isnan(positions_np[j, 0]):
                    conn = Connection(
                        x1=float(positions_np[j, 0]), y1=float(positions_np[j, 1]),
                        x2=float(coord[0]), y2=float(coord[1]),
                        weight=float(weights_np[j])
                    )
                    connections3.add(conn)

        # Fallback if no output connections
        if len(connections3) == 0 and len(hidden_nodes) > 0:
            for output_coord in self.substrate_output_coords:
                nearest_hidden = min(
                    hidden_nodes,
                    key=lambda h: ((h[0] - output_coord[0])**2 + (h[1] - output_coord[1])**2)**0.5
                )
                conn = Connection(
                    x1=nearest_hidden[0], y1=nearest_hidden[1],
                    x2=output_coord[0], y2=output_coord[1],
                    weight=0.5
                )
                connections3.add(conn)

        # Combine and clean
        connections = connections1.union(connections2).union(connections3)
        pre_clean = len(connections)

        result = self._clean_net(connections)
        post_clean_hidden, post_clean_connections_set = result

        phase_info = {
            'phase1_connections': len(connections1),
            'phase2_connections': len(connections2),
            'phase3_connections': len(connections3),
            'pre_clean_connections': pre_clean,
            'post_clean_connections': len(post_clean_connections_set),
            'pre_clean_hidden': len(hidden_nodes),
            'post_clean_hidden': len(post_clean_hidden),
        }

        return post_clean_hidden, post_clean_connections_set, phase_info

    # ========================================================================
    # Network Cleaning (NumPy-accelerated version)
    # ========================================================================

    def _clean_net(self, connections: Set[Connection]) -> Tuple[Set, Set]:
        """Clean network using NumPy-accelerated reachability algorithm.

        This is ~5-10x faster than the set-based approach for large networks
        due to vectorized coordinate matching.
        """
        if len(connections) == 0:
            return set(), set()

        # Convert to NumPy arrays
        conn_array = np.array(
            [[c.x1, c.y1, c.x2, c.y2, c.weight] for c in connections],
            dtype=np.float32
        )
        input_coords = np.array(self.substrate_input_coords, dtype=np.float32)
        output_coords = np.array(self.substrate_output_coords, dtype=np.float32)

        # Use NumPy-accelerated cleaning
        cleaned_conn_array, hidden_nodes_array = clean_connections_numpy(
            conn_array, input_coords, output_coords
        )

        # Convert back to sets for compatibility
        true_connections = set()
        for row in cleaned_conn_array:
            conn = Connection(row[0], row[1], row[2], row[3], row[4])
            true_connections.add(conn)

        true_nodes = set()
        for row in hidden_nodes_array:
            true_nodes.add((float(row[0]), float(row[1])))

        return true_nodes, true_connections

    # ========================================================================
    # Substrate Building (same as optimized version)
    # ========================================================================

    def _build_tensorneat_substrate(
        self, hidden_nodes: Set, connections: Set,
        state: Any = None, cppn_transformed: Any = None
    ) -> Tuple[Any, Any]:
        """Build TensorNEAT substrate from discovered nodes and connections."""
        coord_to_idx = {}

        num_inputs = len(self.substrate_input_coords)
        for i, coord in enumerate(self.substrate_input_coords):
            coord_to_idx[tuple(float(c) for c in coord)] = i

        output_coords_set = set(
            tuple(float(c) for c in coord) for coord in self.substrate_output_coords
        )

        all_hidden_coords = set()
        for conn in connections:
            coord1 = (conn.x1, conn.y1)
            coord2 = (conn.x2, conn.y2)
            if coord1 not in coord_to_idx and coord1 not in output_coords_set:
                all_hidden_coords.add(coord1)
            if coord2 not in coord_to_idx and coord2 not in output_coords_set:
                all_hidden_coords.add(coord2)

        hidden_idx = num_inputs
        for coord in sorted(all_hidden_coords):
            coord_to_idx[coord] = hidden_idx
            hidden_idx += 1

        for i, coord in enumerate(self.substrate_output_coords):
            coord_to_idx[tuple(float(c) for c in coord)] = hidden_idx + i

        num_nodes = len(coord_to_idx)
        nodes = np.zeros((num_nodes, 1))
        for idx in range(num_nodes):
            nodes[idx, 0] = idx

        conn_list = []
        for conn in connections:
            if (conn.x1, conn.y1) in coord_to_idx and (conn.x2, conn.y2) in coord_to_idx:
                from_idx = coord_to_idx[(conn.x1, conn.y1)]
                to_idx = coord_to_idx[(conn.x2, conn.y2)]
                conn_list.append([from_idx, to_idx, conn.weight])

        if len(conn_list) == 0:
            conn_list = self._create_minimal_substrate_fallback(num_nodes)

        conns = np.array(conn_list)

        return jnp.array(nodes), jnp.array(conns)

    def _create_minimal_substrate_fallback(self, num_nodes: int) -> list:
        """Create minimal fallback substrate."""
        import hashlib
        num_inputs = len(self.substrate_input_coords)
        num_outputs = len(self.substrate_output_coords)
        output_start_idx = num_nodes - num_outputs

        seed_str = f"{num_nodes}_{num_inputs}_{num_outputs}"
        seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
        seed = int(seed_hash[:8], 16)
        rng = np.random.RandomState(seed)

        conn_list = []
        for input_idx in range(num_inputs):
            for output_idx in range(output_start_idx, num_nodes):
                weight = rng.uniform(-0.5, 0.5)
                conn_list.append([input_idx, output_idx, weight])

        return conn_list

    # ========================================================================
    # Forward Pass (same as optimized version)
    # ========================================================================

    def _forward_hyperneat_style(self, nodes: Any, conns: Any, inputs: Any) -> Any:
        """Forward pass using HyperNEAT computational model."""
        num_nodes = nodes.shape[0]
        num_inputs = inputs.shape[0]
        num_outputs = len(self.substrate_output_coords)
        output_start_idx = num_nodes - num_outputs

        from_indices = conns[:, 0].astype(jnp.int32)
        to_indices = conns[:, 1].astype(jnp.int32)
        weights = conns[:, 2]
        valid_mask = ~jnp.isnan(weights)
        valid_from = from_indices[valid_mask]
        valid_to = to_indices[valid_mask]
        valid_weights = weights[valid_mask]

        vals = jnp.zeros(num_nodes)
        vals = vals.at[:num_inputs].set(inputs)

        for iteration in range(self.activate_time):
            new_vals = jnp.zeros(num_nodes)
            new_vals = new_vals.at[:num_inputs].set(inputs)

            aggregated = jnp.zeros(num_nodes)
            aggregated = aggregated.at[valid_to].add(vals[valid_from] * valid_weights)

            if output_start_idx > num_inputs:
                hidden_vals = jnp.tanh(aggregated[num_inputs:output_start_idx])
                new_vals = new_vals.at[num_inputs:output_start_idx].set(hidden_vals)

            output_vals = aggregated[output_start_idx:]
            new_vals = new_vals.at[output_start_idx:].set(output_vals)

            vals = new_vals

        raw_outputs = vals[-num_outputs:]
        return jax.nn.sigmoid(raw_outputs)

    def _evaluate_substrate(
        self, state: Any, substrate_net: Tuple[Any, Any], problem: Any
    ) -> float:
        """Evaluate substrate using vmap over test cases."""
        if substrate_net is None:
            return 0.0

        nodes, conns = substrate_net

        if hasattr(problem, 'get_data'):
            data = problem.get_data()
            inputs_list = [inp for inp, _ in data]
            targets_list = [target for _, target in data]
        elif hasattr(problem, 'get_test_cases'):
            test_cases = problem.get_test_cases()
            inputs_list = [tc['input'] for tc in test_cases]
            targets_list = [tc['target'] for tc in test_cases]
        else:
            return 0.0

        if len(inputs_list) == 0:
            return 0.0

        # Stack inputs and targets
        if hasattr(problem, 'use_bias') and problem.use_bias:
            inputs_batch = jnp.stack([jnp.array(inp, dtype=jnp.float32) for inp in inputs_list])
        else:
            inputs_batch = jnp.stack([
                jnp.concatenate([jnp.array(inp, dtype=jnp.float32), jnp.array([1.0])])
                for inp in inputs_list
            ])

        targets_batch = jnp.stack([jnp.array(t, dtype=jnp.float32) for t in targets_list])

        # vmap forward pass
        outputs_batch = jax.vmap(
            lambda inputs: self._forward_hyperneat_style(nodes, conns, inputs)
        )(inputs_batch)

        errors = jnp.mean((outputs_batch - targets_batch) ** 2, axis=1)
        avg_error = jnp.mean(errors)

        return max(0.0, 1.0 - float(avg_error))

    # ========================================================================
    # Configuration
    # ========================================================================

    def create_config(self, params: Dict[str, Any]) -> Any:
        """Create NEAT configuration for CPPN evolution."""
        if params.get('config_file') or params.get('preset'):
            config_manager = ConfigManager()
            hierarchical_config = config_manager.load_config(
                algorithm='hmrhyperneat',
                implementation='tensorneat',
                preset=params.get('preset', 'default'),
                config_file=params.get('config_file'),
                overrides=params.get('overrides', {})
            )
        else:
            hierarchical_config = params

        self._config_metadata = hierarchical_config

        algo_params = hierarchical_config.get('algorithm_params', {}).get('hmrhyperneat', {})
        if not algo_params:
            algo_params = hierarchical_config

        hmr_config = algo_params.get('hmr_hyperneat', {})
        self.initial_depth = hmr_config.get('initial_depth', 0)
        self.max_depth = hmr_config.get('max_depth', 1)
        self.variance_threshold = hmr_config.get('variance_threshold', 0.03)
        self.division_threshold = hmr_config.get('division_threshold', 0.5)
        self.band_threshold = hmr_config.get('band_threshold', 0.3)
        self.max_weight = hmr_config.get('max_weight', 8.0)
        self.iteration_level = hmr_config.get('iteration_level', 1)
        # DEPRECATION WARNING: iteration_level has no effect in this optimized implementation.
        # Original ES-HyperNEAT (PUREPLES) uses iteration_level to control hidden→hidden
        # connection discovery. This implementation omits hidden→hidden entirely for
        # JAX vectorization. See module docstring "Architecture Limitation" for details.
        if 'iteration_level' in hmr_config and hmr_config['iteration_level'] != 1:
            import warnings
            warnings.warn(
                f"iteration_level={self.iteration_level} has no effect in "
                "HMRHyperNEAT. This implementation omits hidden→hidden "
                "connections (which iteration_level controls in original ES-HyperNEAT). "
                "For hidden→hidden support, use PUREPLES ES-HyperNEAT instead.",
                DeprecationWarning,
                stacklevel=2
            )
        self.verbose = hmr_config.get('verbose', False)

        # HMR-HyperNEAT execution mode parameters
        # sparse_forward_threshold: Controls sparse forward pass behavior
        #   -1: Disable sparse (always use dense matrices)
        #    0: Always use sparse (slice to active positions only) - DEFAULT, fastest
        #   >0: Use sparse only when total_positions > threshold
        self.sparse_forward_threshold = hmr_config.get('sparse_forward_threshold', 0)

        # extra_randkey_split: Controls pre-tell() random key splitting behavior
        #   True (default): Adds extra split for different evolutionary trajectories
        #   False: Match EvoX adaptor behavior exactly
        self.extra_randkey_split = hmr_config.get('extra_randkey_split', True)

        # === Memory Optimization Toggles ===
        # fuse_w1_computation: Fuse W1_raw/W1 expression to eliminate intermediate array
        #   True: Use fused expression (saves ~2.6 GB at depth 8)
        #   False (default): Use separate computation (original behavior)
        self.fuse_w1_computation = hmr_config.get('fuse_w1_computation', False)

        # skip_unused_masks: Only compute masks_A, skip masks_B and masks_C
        #   True: Skip unused masks (saves ~0.66 GB at depth 8)
        #   False (default): Compute all masks (original behavior)
        self.skip_unused_masks = hmr_config.get('skip_unused_masks', False)

        # population_chunk_size: Process population in chunks during CPPN queries
        #   -1: AUTO - Use adaptive chunk size based on position count (RECOMMENDED)
        #   0: No chunking - vmap entire population at once
        #   >0: Manual override - process in chunks of this size
        # The adaptive model accounts for:
        #   - Python loop overhead (favors larger chunks)
        #   - XLA kernel compilation efficiency (non-monotonic with tensor size)
        #   - Memory bandwidth constraints (favors cache-friendly sizes)
        self.population_chunk_size = hmr_config.get('population_chunk_size', -1)  # Default: auto

        substrate_section = algo_params.get('substrate', {})
        self.substrate_input_coords = substrate_section.get('input_coords', [])
        self.substrate_output_coords = substrate_section.get('output_coords', [])
        self.output_activation = substrate_section.get('output_activation', 'sigmoid')
        self.hidden_activation = substrate_section.get('hidden_activation', 'tanh')
        default_activate_time = (2 ** self.max_depth) + 1
        self.activate_time = substrate_section.get('activate_time', default_activate_time)

        # Pre-compute quadtree structure
        self._quadtree = get_quadtree_structure(self.max_depth)

        # Parse NEAT configuration section
        # Config can be: {'neat': {'pop_size': 1000}} or {'population_size': 1000}
        neat_section = algo_params.get('neat', {})
        population_size = neat_section.get('pop_size',
                                           neat_section.get('population_size',
                                                           algo_params.get('population_size', 150)))

        # Parse CPPN configuration (allows overriding genome params like num_inputs)
        cppn_section = algo_params.get('cppn', {})
        cppn_genome_config = cppn_section.get('genome', {})

        # Build NEAT config
        # Default num_inputs=5 for [x1, y1, x2, y2, bias]
        # With geometry seeding: num_inputs=7 for [x1, y1, x2, y2, delta_x, delta_y, bias]
        flat_params = {
            'genome': {
                'num_inputs': cppn_genome_config.get('num_inputs', 5),
                'num_outputs': cppn_genome_config.get('num_outputs', 1),
                'num_hidden': 0,
                'feed_forward': True,
                'weight': {
                    'init_mean': 0.0, 'init_std': 1.0,
                    'min_value': -30.0, 'max_value': 30.0,
                    'mutate_power': 0.5, 'mutate_rate': 0.8, 'replace_rate': 0.1,
                },
                'bias': {
                    'init_mean': 0.0, 'init_std': 1.0,
                    'min_value': -30.0, 'max_value': 30.0,
                    'mutate_power': 0.5, 'mutate_rate': 0.7, 'replace_rate': 0.1,
                },
                'activation': {
                    'default': 'tanh',
                    'options': ['tanh', 'sin', 'gauss'],
                    'mutate_rate': 0.5,
                },
            },
            'population_size': population_size,
            'mutation': {
                'conn_add_prob': 0.5, 'conn_delete_prob': 0.5,
                'node_add_prob': 0.2, 'node_delete_prob': 0.2,
            },
            'species': {
                'compatibility_threshold': algo_params.get('neat_species', {}).get('compatibility_threshold', 3.0),
                'max_stagnation': algo_params.get('neat_species', {}).get('max_stagnation', 20),
                'species_elitism': algo_params.get('neat_species', {}).get('species_elitism', 15),
            },
            'selection': {
                'genome_elitism': 15, 'survival_threshold': 0.2,
            },
            'activation_options': ['tanh', 'sin', 'gauss'],
            'activation_default': 'tanh',
            'verbose': False,
        }

        # Override CPPN activation from external config if provided
        cppn_activation = cppn_section.get('activation', {})
        if cppn_activation:
            act_options = cppn_activation.get('options', ['tanh', 'sin', 'gauss'])
            act_default = cppn_activation.get('default', act_options[0])
            act_mutate = cppn_activation.get('mutate_rate', 0.5)
            flat_params['genome']['activation'] = {
                'default': act_default,
                'options': act_options,
                'mutate_rate': act_mutate,
            }
            flat_params['activation_options'] = act_options
            flat_params['activation_default'] = act_default
            print(f"[CPPN] Custom activation: options={act_options}, default={act_default}, mutate_rate={act_mutate}")

        self.neat_algo = self.adapter.build_neat_config(flat_params)
        self._jitted_cppn_forward = jax.jit(
            self.neat_algo.genome.forward, static_argnums=(0,)
        )

        from tensorneat.algorithm.hyperneat.hyperneat import HyperNEATNode, HyperNEATConn
        from tensorneat.genome import RecurrentGenome
        from tensorneat.common import ACT, AGG, State

        self.hyper_genome = RecurrentGenome(
            num_inputs=len(self.substrate_input_coords),
            num_outputs=len(self.substrate_output_coords),
            max_nodes=500, max_conns=2000,
            node_gene=HyperNEATNode(aggregation=AGG.sum, activation=ACT.tanh),
            conn_gene=HyperNEATConn(),
            activate_time=self.activate_time,
            output_transform=ACT.sigmoid
        )

        dummy_state = State()
        dummy_state = self.hyper_genome.setup(dummy_state)

        return self.neat_algo

    def initialize(self, config: Any, problem: Any, seed: int = 42) -> Any:
        """Initialize pipeline and JIT-compile NEAT operations."""
        from tensorneat.pipeline import Pipeline

        wrapped_problem = self._wrap_problem_for_pipeline(problem)

        self.pipeline = Pipeline(
            algorithm=config,
            problem=wrapped_problem,
            seed=seed
        )

        state = self.pipeline.setup()
        self.problem = problem
        self._start_time = time.time()

        self._compiled_ask = jax.jit(self.neat_algo.ask)
        self._compiled_transform_batch = jax.jit(
            jax.vmap(self.neat_algo.transform, in_axes=(None, (0, 0)))
        )
        self._compiled_tell = jax.jit(self.neat_algo.tell)

        # OPTIMIZATION: Cache problem data to avoid repeated list comprehension + array conversion
        data = problem.get_data()
        self._cached_inputs = jnp.array([d[0] for d in data], dtype=jnp.float32)
        self._cached_targets = jnp.array([d[1] for d in data], dtype=jnp.float32)

        # OPTIMIZATION: Cache coordinate arrays (avoid per-generation jnp.array conversion)
        self._cached_input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        self._cached_output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)

        return state

    def _wrap_problem_for_pipeline(self, problem: Any) -> Any:
        """Wrap problem for TensorNEAT pipeline compatibility."""
        # Determine CPPN input dimension: 5 normally, 7 with geometry seeding (delta_x, delta_y)
        cppn_input_dim = 7 if getattr(self, 'geometry_seeding_enabled', False) else 5

        class WrappedProblem:
            def __init__(self, inner_problem, input_dim):
                self.inner = inner_problem
                self.input_shape = (input_dim,)
                self.jitable = True

            def setup(self, state=None):
                from tensorneat.common import State
                return state if state else State()

            def evaluate(self, state, randkey, forward_func, transformed):
                return 0.0

        return WrappedProblem(problem, cppn_input_dim)

    # ========================================================================
    # Adaptive Chunking
    # ========================================================================

    def _compute_optimal_chunk_size(self, positions: int, pop_size: int, n_samples: int = 1000) -> int:
        """Compute optimal chunk size based on detected hardware cache hierarchy.

        ROOT CAUSE (XLA HLO Analysis):
        The intermediate tensor size relative to cache determines performance:
            intermediate = chunk_size × hidden_size × positions × 4 bytes
                         = chunk × 16 × positions × 4

        DATASET-AWARE MEMORY CONSTRAINT:
        For large datasets, network evaluation creates tensors of:
            memory = chunk_size × n_samples × positions × 4 bytes
        This must stay within GPU memory (~2GB target) to avoid OOM.

        Hardware-Adaptive Behavior:
        ---------------------------
        This method detects actual cache sizes at runtime instead of using
        hardcoded values, enabling optimal performance on:
        - Apple Silicon (M1/M2/M3/M4 variants with different P-core/E-core caches)
        - Intel/AMD CPUs (various L2/L3 configurations)
        - NVIDIA GPUs (with L2 caches from 6MB to 72MB)

        The "anomaly zone" (where small chunks outperform large) occurs when
        position counts fall in a range where XLA compilation decisions favor
        smaller memory footprints. This zone shifts based on actual cache sizes.

        Empirical Benchmark Results (M4 Max P-core: L1d=128KB, L2=16MB):
        ----------------------------------------------------------------
        | Depth | Positions | Row Size | Optimal Chunk | Speedup |
        |-------|-----------|----------|---------------|---------|
        | 3     | 340       | 1.3 KB   | 100           | 1.7x    |
        | 4     | 1,364     | 5.3 KB   | 500           | 1.2x    |
        | 5     | 5,460     | 21.4 KB  | 500           | 1.4x    |
        | 6     | 21,844    | 87.4 KB  | 10            | 1.7x    |
        | 7     | 87,380    | 350 KB   | 200           | 1.9x    |
        | 8     | 349,524   | 1.37 MB  | 50            | 11x     |
        ----------------------------------------------------------------

        Dataset-Aware Memory Caps (75K samples, depth 4 = 1,364 positions):
        ------------------------------------------------------------------
        | Dataset   | Positions | Base Chunk | Memory-Capped | Peak Memory |
        |-----------|-----------|------------|---------------|-------------|
        | XOR (4)   | 1,364     | 500        | 500           | 10 MB       |
        | 1K        | 1,364     | 500        | 366           | 2.0 GB      |
        | 75K       | 1,364     | 500        | 4             | 1.6 GB      |
        | 75K       | 5,460     | 500        | 1             | 1.6 GB      |
        ------------------------------------------------------------------

        Args:
            positions: Number of target positions to query (quadtree grid size)
            pop_size: Population size
            n_samples: Number of samples in dataset (for memory constraint)

        Returns:
            Recommended chunk size for optimal performance on current hardware
        """
        # Get cache info (cached at instance level for performance)
        if not hasattr(self, '_cache_info'):
            self._cache_info = self._detect_cache_hierarchy()

        cache = self._cache_info
        hidden_size = 16  # CPPN hidden layer size
        bytes_per_float = 4

        # Calculate intermediate tensor size for chunk=1
        # intermediate_per_chunk = 16 × positions × 4 bytes
        intermediate_per_chunk = hidden_size * positions * bytes_per_float

        # Target: keep intermediate tensor within L2 cache
        # chunk_max = L2 / (16 × positions × 4)
        if positions > 0:
            chunk_for_l2 = cache['l2'] // intermediate_per_chunk
        else:
            chunk_for_l2 = pop_size

        # Clamp chunk_for_l2 to reasonable bounds
        chunk_for_l2 = max(10, min(chunk_for_l2, pop_size))

        # Apply empirically-validated heuristics
        # The behavior differs significantly between Apple Silicon and NVIDIA GPUs:
        #
        # APPLE SILICON (M4 Max, L2=16MB):
        #   - "Anomaly zone" at 6K-50K positions where small chunks outperform large
        #   - This is due to XLA compilation decisions on Metal backend
        #
        # NVIDIA GPUs (RTX 2080 Ti, L2=6MB, benchmarked 2024-12-12):
        #   - NO anomaly zone observed - larger chunks are consistently better
        #   - Depth 1-3 (< 1000 positions): chunk=500 optimal, 1.2-1.3x faster than 100
        #   - Depth 4 (341 positions): chunk=100 slight edge, within 7%
        #   - Depth 5-6 (1-6K positions): chunk=500 optimal
        #   - GPU throughput benefits from larger batch sizes until OOM
        #
        # Platform detection: use L2 cache size as proxy for GPU vs Apple Silicon
        # NVIDIA GPUs have 4-72MB L2, Apple Silicon CPUs have 16-32MB L2
        # Key distinction: Apple Silicon reports cache via sysctl, NVIDIA via nvidia-smi

        import jax
        is_gpu = any(d.platform == 'gpu' for d in jax.devices())

        if is_gpu:
            # NVIDIA GPU path: larger chunks generally better
            # Empirical results from RTX 2080 Ti (L2=6MB):
            # | Depth | Positions | Best Chunk | Note |
            # |-------|-----------|------------|------|
            # | 1-3   | 5-85      | 500        | Large chunks reduce overhead |
            # | 4     | 341       | 100        | Slight edge at this size |
            # | 5-6   | 1365-5461 | 500        | Optimal |
            # | 7     | 21845     | OOM        | Need smaller chunks to fit in VRAM |

            if positions < 500:
                # Very small grids: large chunks reduce loop overhead
                base_chunk = min(500, pop_size)
            elif positions < 6000:
                # Small-medium grids: large chunks optimal
                base_chunk = min(500, pop_size)
            elif positions < 30000:
                # Medium grids: still benefit from large chunks on GPU
                # But be conservative to avoid OOM on 11GB GPUs
                base_chunk = min(200, pop_size)
            else:
                # Large grids (depth 7+): memory constrained
                # Use smaller chunks to avoid OOM
                base_chunk = min(50, pop_size)

        else:
            # Apple Silicon / CPU path: original logic with anomaly zone
            if positions < 1000:
                # Very small grids: loop overhead dominates, use medium chunks
                base_chunk = min(100, pop_size)

            elif positions < 6000:
                # Small-medium grids: larger chunks reduce loop overhead
                base_chunk = min(500, pop_size)

            elif positions < 50000:
                # ANOMALY ZONE (depth 6): XLA compiles inefficiently for medium position
                # counts with large chunks on Apple Silicon Metal backend.
                # Small chunks restore efficient memory access.
                # Scale chunk based on L2 cache: larger L2 → can use slightly larger chunks
                if cache['l2'] >= 40 * 1024 * 1024:  # GPU-class L2 cache
                    base_chunk = min(50, pop_size)
                else:
                    scaled_chunk = max(10, chunk_for_l2 // 5)  # Less conservative
                    base_chunk = min(scaled_chunk, 25, pop_size)

            elif positions < 200000:
                # Large grids (depth 7): medium chunks work well
                # Empirically, chunk=200 is optimal here even though it exceeds L2
                # XLA handles this range efficiently with good vectorization
                base_chunk = min(200, pop_size)

            else:
                # Very large grids (depth 8+): memory bandwidth critical
                # Smaller chunks (50) dramatically outperform larger chunks
                base_chunk = min(50, pop_size)

        # =====================================================================
        # Dataset-Aware Memory Constraint
        # =====================================================================
        # For large datasets (e.g., 75K samples), network evaluation creates tensors:
        #   memory = chunk_size × n_samples × positions × 4 bytes
        # Cap at 2GB to avoid OOM on GPUs with 11GB VRAM.
        #
        # Example: 75K samples × 1,364 positions × 500 chunk = 204 GB → OOM!
        #          75K samples × 1,364 positions × 4 chunk = 1.6 GB → OK
        # =====================================================================
        TARGET_MEMORY_GB = 2.0
        memory_per_network = n_samples * positions * bytes_per_float
        if memory_per_network > 0:
            max_chunk_for_memory = max(1, int(
                (TARGET_MEMORY_GB * 1e9) / memory_per_network
            ))
        else:
            max_chunk_for_memory = pop_size

        # Final chunk is minimum of position-based and memory-capped
        chunk_size = min(base_chunk, max_chunk_for_memory)

        return chunk_size

    def _detect_cache_hierarchy(self) -> dict:
        """Detect cache sizes with fallback defaults.

        Detects cache sizes at runtime using:
        - macOS: sysctl (with P-core preference for Apple Silicon)
        - Linux: /sys/devices/system/cpu/cpu0/cache/
        - GPU: nvidia-smi with known L2 cache lookup

        Returns:
            Dict with 'l1d', 'l2', 'l3' in bytes
        """
        from emr_hyperneat._compat.utils.hardware_info import CacheInfo
        import jax

        # Check if running on GPU
        try:
            devices = jax.devices()
            if devices and devices[0].platform.lower() in ('gpu', 'cuda'):
                gpu_info = CacheInfo.get_gpu_cache_info()
                if gpu_info['l2'] > 0:
                    # GPU: Use L2 as primary cache (no L1/L3 in same sense as CPU)
                    # GPU L2 caches are typically 6-72 MB
                    return {
                        'l1d': 128 * 1024,        # Shared memory proxy (128 KB)
                        'l2': gpu_info['l2'],
                        'l3': gpu_info['l2'],     # GPU L2 acts as effective L3
                    }
        except Exception:
            pass

        # CPU path
        try:
            cpu_cache = CacheInfo.get_cpu_cache_info()
        except Exception:
            cpu_cache = {}

        # Apply conservative defaults if detection failed
        # These are pessimistic to ensure safe chunking on unknown hardware
        defaults = {
            'l1d': 32 * 1024,      # 32 KB (common minimum)
            'l2': 512 * 1024,      # 512 KB (conservative)
            'l3': 8 * 1024 * 1024, # 8 MB (conservative)
        }

        return {
            'l1d': cpu_cache.get('l1d') or defaults['l1d'],
            'l2': cpu_cache.get('l2') or defaults['l2'],
            'l3': cpu_cache.get('l3') or defaults['l3'],
        }

    def _get_effective_chunk_size(self, positions: int, pop_size: int, n_samples: int = 1000) -> int:
        """Get the effective chunk size (auto-computed or manual override).

        Args:
            positions: Number of target positions
            pop_size: Population size
            n_samples: Number of samples in dataset (for memory constraint)

        Returns:
            Effective chunk size to use for CPPN queries
        """
        if self.population_chunk_size == -1:
            # Auto mode: compute optimal based on position count and dataset size
            chunk_size = self._compute_optimal_chunk_size(positions, pop_size, n_samples=n_samples)
            if self.verbose:
                import logging
                logging.info(
                    f"Adaptive chunking: auto-selected chunk_size={chunk_size} "
                    f"for {positions:,} positions, {n_samples:,} samples (pop={pop_size})"
                )
            return chunk_size
        elif self.population_chunk_size == 0:
            # No chunking - return full population
            return pop_size
        else:
            # Manual override
            return self.population_chunk_size

    # ========================================================================
    # Run Generation
    # ========================================================================

    def run_generation_verbose(
        self, state: Any, problem: Any, skip_metrics: bool = False
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation with verbose Python loop (for debugging/profiling).

        This method provides detailed per-step timing instrumentation at the cost
        of 8 GPU↔CPU synchronization points per generation.

        For production use, prefer:
        - run_generation(): GPU-resident single generation (single sync)
        - run_until_threshold(): GPU-resident multi-generation with early stopping

        Implementation details:
        - Hierarchical grid with variance-based subdivision
        - ALL positions from ALL levels as hidden nodes
        - GPU-accelerated CPPN queries via vmap
        - ~96-97% GPU-efficient on Apple Silicon

        Performance (XOR benchmark):
        - 100% solve rate (5/5 seeds)
        - ~510ms/generation on Apple M4
        - Average 44 generations to solve

        For max_depth=2: 4 + 16 + 64 = 84 total hidden node positions
        """
        gen_start = time.time()
        step_timings = {}  # Local step timings for this generation

        # Optional pre-tell random key split (off by default to match EvoX adaptor)
        # When enabled, shifts the random sequence for different evolutionary trajectories
        if self.extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # =====================================================================
        # STEP 0: CPPN ask + transform (get population and prepare for queries)
        # =====================================================================
        t0 = time.perf_counter()

        # Get CPPN population
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]

        # Batch transform all CPPNs
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        step_timings['step0_cppn_ask_transform'] = time.perf_counter() - t0

        # Get hierarchical grid for current max_depth
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions  # 4+16+64=84 for max_depth=2

        # OPTIMIZATION: Use cached coordinate arrays (avoids per-generation conversion)
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        # OPTIMIZATION: Use cached problem data instead of repeated conversion
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # =====================================================================
        # STEP 1+3 UNIFIED: Query ALL inputs→positions and outputs→positions
        # OPTIMIZATION: Eliminates redundant input[0]→positions query
        # input[0] is queried once, in STEP 1
        # Now: single query for ALL inputs, extract variance from input[0]
        # =====================================================================
        t1 = time.perf_counter()

        all_positions = h_grid.all_positions  # shape (total_positions, 2)
        total_positions = all_positions.shape[0]

        # ADAPTIVE CHUNKING: Compute optimal chunk size based on position count and dataset
        # This uses empirical benchmark data to select the best chunk size
        # For large datasets (75K+ samples), memory constraint dominates position-based heuristics
        n_samples = inputs_batch.shape[0]  # Dataset size for memory constraint
        effective_chunk_size = self._get_effective_chunk_size(
            positions=total_positions,
            pop_size=pop_size,
            n_samples=n_samples
        )

        # Use chunked query when chunk_size < pop_size
        # This reduces peak memory from 139+ GB to ~1-2 GB per chunk at depth 8
        if effective_chunk_size < pop_size:
            query_func = lambda state, cppns, sources, targets, outgoing, fwd: \
                batch_query_population_multi_source_chunked(
                    state, cppns, sources, targets, outgoing, fwd,
                    pop_chunk_size=effective_chunk_size
                )
        else:
            query_func = batch_query_population_multi_source

        # Input → all positions (outgoing): shape (pop_size, num_inputs, total_positions)
        # This includes input[0] which we use for variance computation
        input_all_weights = query_func(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward
        )

        # All → output (incoming): shape (pop_size, num_outputs, total_positions)
        output_all_weights = query_func(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward
        )

        # Extract variance weights from input[0] (zero-copy slice)
        # This replaces the separate STEP 1 query
        all_weights_for_variance = input_all_weights[:, 0, :]  # (pop_size, total_positions)

        step_timings['step1_unified_cppn_query'] = time.perf_counter() - t1

        # =====================================================================
        # STEP 2: Compute hierarchical variances and subdivision masks
        # =====================================================================
        t2 = time.perf_counter()

        level_variances = compute_hierarchical_variances_batch(
            all_weights_for_variance, h_grid
        )

        # Compute subdivision masks - we use masks_A for multi-resolution union
        # MEMORY OPTIMIZATION: skip_unused_masks=True skips masks_B and masks_C (~0.66 GB at depth 8)
        if self.skip_unused_masks:
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )
        else:
            masks_A, _, _ = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=True
            )
        # masks_A: (pop_size, total_positions) - True for ALL reached positions at ALL levels

        step_timings['step2_variance_masks'] = time.perf_counter() - t2

        # NOTE: STEP 3 timing is now included in STEP 1 (unified query)
        # Keeping step3 timing key for backwards compatibility with analysis tools
        step_timings['step3_weight_queries'] = 0.0  # Already included in step1

        # =====================================================================
        # STEP 4: Apply masks and build weight matrices
        # =====================================================================
        t4 = time.perf_counter()

        max_weight = self.max_weight
        weight_thresh = 0.1

        # Broadcast mask: (pop_size, 1, total_positions) for weight masking
        active_mask_broadcast = masks_A[:, None, :]

        if self.fuse_w1_computation:
            # MEMORY OPTIMIZATION: Fused expression eliminates intermediate arrays
            # XLA's common subexpression elimination will compute tanh*max_weight once
            # Saves ~3.9 GB at depth 8 (W1_raw + W2_raw intermediates eliminated)
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(jnp.tanh(input_all_weights) * max_weight) > weight_thresh),
                jnp.tanh(input_all_weights) * max_weight,
                0.0
            )
            W2_masked = jnp.where(
                active_mask_broadcast & (jnp.abs(jnp.tanh(output_all_weights) * max_weight) > weight_thresh),
                jnp.tanh(output_all_weights) * max_weight,
                0.0
            )
            # Transpose W2 for matrix multiply: (pop, total_positions, num_outputs)
            W2 = jnp.transpose(W2_masked, (0, 2, 1))
        else:
            # ORIGINAL: Use jnp.where instead of boolean multiplication
            # This avoids creating intermediate boolean mask arrays
            # Apply tanh activation and scale
            W1_raw = jnp.tanh(input_all_weights) * max_weight  # (pop, num_inputs, total_positions)
            W2_raw = jnp.tanh(output_all_weights) * max_weight  # (pop, num_outputs, total_positions)

            # Use jnp.where for combined mask: active position AND above weight threshold
            # This is more memory efficient than boolean multiplication
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )

            W2_raw = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )

            # Transpose W2 for matrix multiply: (pop, total_positions, num_outputs)
            W2 = jnp.transpose(W2_raw, (0, 2, 1))

        step_timings['step4_weight_matrix_build'] = time.perf_counter() - t4

        # =====================================================================
        # STEP 5: Evaluate ALL networks via vmap
        # OPTIMIZATION: Union mask sparse forward for high-depth configurations
        # =====================================================================
        t5 = time.perf_counter()

        # Use sparse forward pass if enabled (threshold >= 0) and positions exceed threshold
        total_positions = h_grid.total_positions
        use_sparse = (
            self.sparse_forward_threshold >= 0 and
            total_positions > self.sparse_forward_threshold
        )

        def eval_single_network(W1_single, W2_single, inputs, targets):
            """Evaluate single two-layer network with all-level hidden nodes."""
            hidden = jnp.tanh(safe_matmul(inputs, W1_single))  # (num_cases, total_positions)
            outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        if use_sparse:
            # Sparse path: slice weight matrices to only active positions
            union_mask = jnp.any(masks_A, axis=0)
            active_indices = jnp.nonzero(union_mask, size=total_positions, fill_value=0)[0]
            num_active = jnp.sum(union_mask)

            W1_active = jnp.take(W1, active_indices, axis=2)
            W2_active = jnp.take(W2, active_indices, axis=1)

            # Zero out padding (indices beyond num_active are invalid)
            valid_mask = jnp.arange(total_positions) < num_active
            W1_active = W1_active * valid_mask[None, None, :]
            W2_active = W2_active * valid_mask[None, :, None]

            fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
                W1_active, W2_active, inputs_batch, targets_batch
            )

            step_timings['sparse_num_active'] = float(num_active)
            step_timings['sparse_total_positions'] = float(total_positions)
            step_timings['sparse_ratio'] = float(num_active) / float(total_positions)
        else:
            # Dense path: use full weight matrices
            fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
                W1, W2, inputs_batch, targets_batch
            )
            step_timings['sparse_ratio'] = 1.0

        # Handle NaN
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        step_timings['step5_network_evaluation'] = time.perf_counter() - t5
        step_timings['step5_used_sparse'] = use_sparse

        # =====================================================================
        # STEP 6: NEAT evolution step
        # =====================================================================
        t6 = time.perf_counter()

        new_state = self._compiled_tell(state, fitnesses)

        step_timings['step6_neat_evolution'] = time.perf_counter() - t6

        # Compute actual hidden node counts per genome (active at all levels)
        # Keep as JAX arrays to avoid GPU→CPU sync until final metrics extraction
        active_counts = jnp.sum(masks_A, axis=1)  # (pop_size,)
        avg_hidden_jax = jnp.mean(active_counts)
        min_hidden_jax = jnp.min(active_counts)
        max_hidden_jax = jnp.max(active_counts)

        # =====================================================================
        # STEP 7: Metrics extraction (GPU→CPU sync point)
        # OPTIMIZATION: Batch all metrics into single array for single sync
        # When skip_metrics=True, skip GPU→CPU sync entirely (0 syncs)
        # =====================================================================
        t7 = time.perf_counter()

        if skip_metrics:
            # Skip all metrics extraction - return minimal metrics object
            # No GPU→CPU sync happens here - ALL data stays on GPU
            metrics = AlgorithmMetrics(
                generation=new_state.generation if hasattr(new_state, 'generation') else 0,
                best_fitness=0.0,  # no GPU sync on this path
                mean_fitness=0.0,
                min_fitness=0.0,
                max_fitness=0.0,
                std_fitness=0.0,
                num_species=0,
                species_sizes=[],
                species_fitness=[],
                evaluations=0,
                time_elapsed=time.time() - gen_start,
            )
            # Override custom_metrics to mark skip mode
            metrics.custom_metrics = {'skip_metrics': True}
            avg_hidden = 0.0
            min_hidden = 0.0
            max_hidden = 0.0
        else:
            # BATCHED EXTRACTION: Combine all 8 metrics into single JAX array
            # This reduces 8 GPU→CPU syncs to 1 sync (8x fewer kernel launches)
            # Order: [best_fit, mean_fit, min_fit, max_fit, std_fit, avg_hidden, min_hidden, max_hidden]
            metrics_batch_jax = jnp.array([
                jnp.max(fitnesses),
                jnp.mean(fitnesses),
                jnp.min(fitnesses),
                jnp.max(fitnesses),  # max_fitness (same as best)
                jnp.std(fitnesses),
                avg_hidden_jax,
                min_hidden_jax,
                max_hidden_jax,
            ])

            # SINGLE GPU→CPU sync for all metrics
            metrics_batch = traced_device_get(metrics_batch_jax, "metrics_batch")

            # Unpack metrics
            best_fitness = float(metrics_batch[0])
            mean_fitness = float(metrics_batch[1])
            min_fitness = float(metrics_batch[2])
            max_fitness = float(metrics_batch[3])
            std_fitness = float(metrics_batch[4])
            avg_hidden = float(metrics_batch[5])
            min_hidden = float(metrics_batch[6])
            max_hidden = float(metrics_batch[7])

            # Pre-extracted metrics dict for _create_metrics
            pre_extracted = {
                'best_fitness': best_fitness,
                'mean_fitness': mean_fitness,
                'min_fitness': min_fitness,
                'max_fitness': max_fitness,
                'std_fitness': std_fitness,
            }

            # Create metrics with pre-extracted values (no additional syncs)
            metrics = self._create_metrics(
                new_state, fitnesses, gen_start,
                avg_hidden,
                avg_hidden * (num_inputs + num_outputs),
                pre_extracted_metrics=pre_extracted
            )

        step_timings['step7_metrics_extraction'] = time.perf_counter() - t7

        # Accumulate step timings if tracing enabled
        if _TRACE_STEP_TIMING:
            for step_name, step_time in step_timings.items():
                _STEP_TIMINGS[step_name] = _STEP_TIMINGS.get(step_name, 0.0) + step_time
            # Print per-generation breakdown
            total_time = sum(step_timings.values())
            gen_num = new_state.generation if hasattr(new_state, 'generation') else 0
            print(f"[STEP_TIMING] Gen {gen_num}: total={total_time*1000:.1f}ms", flush=True)
            for step_name in sorted(step_timings.keys()):
                step_time = step_timings[step_name]
                pct = (step_time / total_time * 100) if total_time > 0 else 0
                print(f"  {step_name}: {step_time*1000:.1f}ms ({pct:.1f}%)", flush=True)

        # Add step timings to custom metrics (always, for external analysis)
        metrics.custom_metrics['step_timings'] = step_timings
        metrics.custom_metrics['method'] = 'vmapped_multiresA'
        metrics.custom_metrics['avg_hidden_nodes'] = avg_hidden
        metrics.custom_metrics['min_hidden_nodes'] = min_hidden
        metrics.custom_metrics['max_hidden_nodes'] = max_hidden
        metrics.custom_metrics['variance_threshold'] = self.variance_threshold
        metrics.custom_metrics['num_levels'] = h_grid.num_levels
        metrics.custom_metrics['total_positions'] = total_positions

        return new_state, metrics

    def run_generation(
        self,
        state: Any,
        problem: Any,
        skip_metrics: bool = False,
        verbose: bool = True,
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation (main entry point).

        This is the primary entry point for running a single generation.
        Uses the verbose (Python loop) implementation by default, which provides
        detailed per-step timing and is actually faster for single-generation runs.

        Args:
            state: Algorithm state from initialize()
            problem: Problem instance with get_data() method
            skip_metrics: If True, skip metrics extraction (0 syncs)
            verbose: If True (default), use Python loop with detailed timing

        Returns:
            Tuple of (new_state, AlgorithmMetrics)

        Performance Note:
            For single generations, the Python loop is faster than GPU-resident
            execution due to while_loop setup overhead.

            For multi-generation runs targeting a fitness threshold, use
            run_until_threshold() directly - it runs the entire loop on GPU with
            only one GPU↔CPU sync at the end, providing significant speedup.

        Example:
            # Single generations (use run_generation)
            state, metrics = algo.run_generation(state, problem)

            # Multi-generation with early stopping (use run_until_threshold)
            result = algo.run_until_threshold(
                state, problem,
                target_fitness=0.99,
                max_generations=100
            )
        """
        # Use verbose (Python loop) implementation - faster for single generations
        return self.run_generation_verbose(state, problem, skip_metrics)

    # Aliases for backwards compatibility (point to verbose implementation)
    run_generation_vmapped_multiresA = run_generation_verbose
    run_generation_vmapped_multires = run_generation_verbose

    # ========================================================================
    # GPU-Resident Multi-Generation Loop with Threshold-Based Early Stopping
    # ========================================================================

    def _pure_generation_step(
        self,
        state: Any,
        cppns_transformed: Any,
        h_grid: Any,
        input_coords: jnp.ndarray,
        output_coords: jnp.ndarray,
        inputs_batch: jnp.ndarray,
        targets_batch: jnp.ndarray,
        extra_randkey_split: bool = False,
    ) -> Tuple[Any, jnp.ndarray]:
        """Single generation step - PURE JAX, no CPU sync.

        This method contains only JIT-compatible operations extracted from
        run_generation(). It can be used inside jax.lax.while_loop for
        GPU-resident multi-generation evolution with threshold checking.

        Args:
            state: TensorNEAT algorithm state (JAX pytree)
            cppns_transformed: Pre-transformed CPPN population
            h_grid: Hierarchical grid configuration
            input_coords: Substrate input coordinates (JAX array)
            output_coords: Substrate output coordinates (JAX array)
            inputs_batch: Cached problem inputs (JAX array)
            targets_batch: Cached problem targets (JAX array)
            extra_randkey_split: If True, split random key before tell() for
                different evolutionary trajectories. Default False matches EvoX.

        Returns:
            Tuple of (new_state, fitnesses) where both are JAX arrays/pytrees.
            No GPU→CPU synchronization occurs.
        """
        # Optional pre-tell random key split (off by default to match EvoX adaptor)
        if extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # STEP 0: CPPN ask + transform
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Get grid info
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        # STEP 1: Query CPPN at all positions for variance computation
        source_coord = input_coords[0:1]
        all_weights_for_variance = batch_query_population_multi_source_jax_pure(
            state, cppns_transformed, source_coord,
            all_positions, True, self._jitted_cppn_forward
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        # STEP 2: Compute hierarchical variances and subdivision masks
        # Use JIT-compatible versions with static metadata for while_loop compatibility
        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # STEP 3: Query weights for input→all and all→output connections
        input_all_weights = batch_query_population_multi_source_jax_pure(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward
        )
        output_all_weights = batch_query_population_multi_source_jax_pure(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward
        )

        # STEP 4: Apply masks and build weight matrices
        max_weight = self.max_weight
        weight_thresh = 0.1

        W1_raw = jnp.tanh(input_all_weights) * max_weight
        W2_raw = jnp.tanh(output_all_weights) * max_weight

        active_mask_broadcast = masks_A[:, None, :]
        W1_combined_mask = active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh)
        W1 = W1_raw * W1_combined_mask

        W2_combined_mask = active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh)
        W2_raw = W2_raw * W2_combined_mask
        W2 = jnp.transpose(W2_raw, (0, 2, 1))

        # STEP 5: Evaluate ALL networks via vmap
        def eval_single_network(W1_single, W2_single, inputs, targets):
            # NOTE: Explicit args required for JAX CUDA compatibility (not closures).
            # The closure pattern works on Apple Metal but causes SIGSEGV on NVIDIA GPUs.
            # NOTE: Uses safe_matmul() to work around CUDA cuBLAS crash on matrices
            # larger than ~7x7 (JAX 0.4.38 + CUDA 12.4 + RTX 2080 Ti).
            hidden = jnp.tanh(safe_matmul(inputs, W1_single))
            outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        # in_axes: (0, 0, None, None) - vmap over W1/W2, broadcast inputs/targets
        fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
            W1, W2, inputs_batch, targets_batch
        )
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        # STEP 6: NEAT evolution step
        new_state = self._compiled_tell(state, fitnesses)

        return new_state, fitnesses

    def run_until_threshold(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Run multiple generations on GPU until fitness threshold is met.

        Uses jax.lax.while_loop to run generations entirely on GPU with
        GPU-side threshold checking. Only ONE GPU→CPU sync at the end.

        This enables early stopping when a solution is found without the
        overhead of per-generation CPU synchronization.

        Args:
            state: Initialized algorithm state from initialize()
            problem: Problem instance (must have get_data() method)
            target_fitness: Stop when jnp.max(fitnesses) >= target_fitness
            max_generations: Maximum generations before stopping
            collect_history: If True, collect per-generation best fitness history

        Returns:
            Dict with:
                'generations': int - Number of generations run
                'best_fitness': float - Best fitness achieved
                'state': Final algorithm state
                'history': (optional) array of per-generation best fitness

        Performance:
            - Current run_generation loop: 8 GPU→CPU syncs per generation
            - This method: 1 GPU→CPU sync total (at end)
            - Expected speedup: 1.82x → 2-3x for threshold-based runs

        Example:
            >>> state = algo.initialize(config, problem, seed=42)
            >>> result = algo.run_until_threshold(
            ...     state, problem,
            ...     target_fitness=0.98,
            ...     max_generations=100
            ... )
            >>> print(f"Solved in {result['generations']} generations")
            >>> print(f"Best fitness: {result['best_fitness']:.6f}")
        """
        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates as JAX arrays
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # Convert target to JAX array for GPU-side comparison
        target_fitness_jax = jnp.array(target_fitness, dtype=jnp.float32)
        max_gens_jax = jnp.array(max_generations, dtype=jnp.int32)

        # Get initial transformed CPPNs (will be recomputed in loop)
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Capture extra_randkey_split setting for use in loop body
        use_extra_split = self.extra_randkey_split

        if collect_history:
            # Version with history collection
            def loop_body(carry):
                generation, best_so_far, current_state, history = carry

                new_state, fitnesses = self._pure_generation_step(
                    current_state, cppns_transformed, h_grid,
                    input_coords, output_coords,
                    inputs_batch, targets_batch,
                    extra_randkey_split=use_extra_split
                )

                gen_best = jnp.max(fitnesses)
                best_so_far = jnp.maximum(best_so_far, gen_best)

                # Store in pre-allocated array
                history = history.at[generation].set(gen_best)

                return (generation + 1, best_so_far, new_state, history)

            def loop_condition(carry):
                generation, best_so_far, current_state, history = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            # Pre-allocate history array
            history = jnp.zeros(max_generations, dtype=jnp.float32)
            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state,
                history
            )

            final_gen, final_best, final_state, final_history = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            # SINGLE GPU→CPU sync at the very end
            final_gen_py = int(jax.device_get(final_gen))
            return {
                'generations': final_gen_py,
                'best_fitness': float(jax.device_get(final_best)),
                'state': final_state,
                'history': jax.device_get(final_history[:final_gen_py]),
            }

        else:
            # Version without history (minimal memory)
            def loop_body(carry):
                generation, best_so_far, current_state = carry

                new_state, fitnesses = self._pure_generation_step(
                    current_state, cppns_transformed, h_grid,
                    input_coords, output_coords,
                    inputs_batch, targets_batch,
                    extra_randkey_split=use_extra_split
                )

                gen_best = jnp.max(fitnesses)
                best_so_far = jnp.maximum(best_so_far, gen_best)

                return (generation + 1, best_so_far, new_state)

            def loop_condition(carry):
                generation, best_so_far, current_state = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state
            )

            final_gen, final_best, final_state = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            # SINGLE GPU→CPU sync at the very end
            return {
                'generations': int(jax.device_get(final_gen)),
                'best_fitness': float(jax.device_get(final_best)),
                'state': final_state,
            }

    def _create_metrics(
        self, state: Any, fitnesses: Any, gen_start: float,
        discovered_hidden: float, total_connections: float,
        pre_extracted_metrics: Optional[Dict[str, float]] = None
    ) -> AlgorithmMetrics:
        """Create AlgorithmMetrics with ES-HyperNEAT data.

        Args:
            state: Current algorithm state
            fitnesses: JAX array of fitness values
            gen_start: Start time of generation
            discovered_hidden: Number of discovered hidden nodes
            total_connections: Total number of connections
            pre_extracted_metrics: Optional dict with pre-extracted fitness stats
                (best_fitness, mean_fitness, min_fitness, max_fitness, std_fitness)
                If provided, skips GPU→CPU sync for these values.
        """
        generation = state.generation if hasattr(state, 'generation') else 0

        # Use pre-extracted metrics if available (batched extraction)
        # Otherwise fall back to individual extractions
        if pre_extracted_metrics is not None:
            best_fitness = pre_extracted_metrics['best_fitness']
            mean_fitness = pre_extracted_metrics['mean_fitness']
            min_fitness = pre_extracted_metrics['min_fitness']
            max_fitness = pre_extracted_metrics['max_fitness']
            std_fitness = pre_extracted_metrics['std_fitness']
        else:
            # Fallback: individual device_get calls (8 syncs)
            best_fitness = traced_device_get(jnp.max(fitnesses), "best_fitness")
            mean_fitness = traced_device_get(jnp.mean(fitnesses), "mean_fitness")
            min_fitness = traced_device_get(jnp.min(fitnesses), "min_fitness")
            max_fitness = traced_device_get(jnp.max(fitnesses), "max_fitness")
            std_fitness = traced_device_get(jnp.std(fitnesses), "std_fitness")

        evaluations = len(fitnesses)
        time_elapsed = time.time() - gen_start

        custom_metrics = {
            'discovered_hidden_nodes': discovered_hidden,
            'total_connections': total_connections,
            'generation_time': time_elapsed,
            'implementation': 'jax-optimized',
        }

        return AlgorithmMetrics(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            std_fitness=std_fitness,
            num_species=1,
            species_sizes=[len(fitnesses)],
            species_fitness=[mean_fitness],
            evaluations=evaluations,
            time_elapsed=time_elapsed,
            custom_metrics=custom_metrics
        )

    # ========================================================================
    # BaseAlgorithm Abstract Methods
    # ========================================================================

    def evaluate_genome(self, genome: Any, problem: Any) -> float:
        return 0.0

    def extract_network_info(self, state: Any) -> Any:
        return None

    def genome_to_phenotype(self, genome: Any) -> Any:
        return None

    def get_best_genome(self, state: Any) -> Any:
        if hasattr(self, 'neat_algo') and self.neat_algo is not None:
            pop = self.neat_algo.ask(state)
            if pop is not None and len(pop) > 0:
                return pop[0]
        return None

    def evaluate_on_data(
        self,
        state: Any,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> Dict[str, float]:
        """Evaluate the best genome on arbitrary input/target data.

        This method extracts the best CPPN from the population, builds its
        substrate network, and evaluates it on the provided data. Useful for
        test set evaluation after training.

        Args:
            state: Current algorithm state (contains population)
            inputs: Input data array of shape (n_samples, n_features)
            targets: Target data array of shape (n_samples, n_outputs)

        Returns:
            Dict with 'mse', 'fitness', and 'n_samples'
        """
        # Get CPPN population and transform
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Use cached coordinate arrays
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        # Convert inputs/targets to JAX arrays if needed
        if not isinstance(inputs, jnp.ndarray):
            inputs = jnp.array(inputs, dtype=jnp.float32)
        if not isinstance(targets, jnp.ndarray):
            targets = jnp.array(targets, dtype=jnp.float32)

        # Query CPPN for best genome (index 0 after sorting by fitness)
        all_positions = h_grid.all_positions

        # Input → all positions: shape (pop_size, num_inputs, total_positions)
        input_all_weights = batch_query_population_multi_source(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward
        )

        # All → output: shape (pop_size, num_outputs, total_positions)
        output_all_weights = batch_query_population_multi_source(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward
        )

        # Extract variance weights from input[0]
        all_weights_for_variance = input_all_weights[:, 0, :]

        # Compute hierarchical variances and subdivision masks (matching evaluate_batch logic)
        level_variances = compute_hierarchical_variances_batch(
            all_weights_for_variance, h_grid
        )
        masks_A, _, _ = compute_subdivision_masks_batch(
            level_variances, self.variance_threshold, h_grid
        )
        # masks_A: (pop_size, total_positions) - True for ALL reached positions

        # Apply tanh activation and scale (matching evaluate_batch Step 4)
        max_weight = self.max_weight
        weight_thresh = 0.1

        W1_raw = jnp.tanh(input_all_weights) * max_weight  # (pop, num_inputs, total_positions)
        W2_raw = jnp.tanh(output_all_weights) * max_weight  # (pop, num_outputs, total_positions)

        # Broadcast mask: (pop_size, 1, total_positions) for weight masking
        active_mask_broadcast = masks_A[:, None, :]

        # Use jnp.where for combined mask: active position AND above weight threshold
        W1_masked = jnp.where(
            active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
            W1_raw,
            0.0
        )
        W2_masked = jnp.where(
            active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
            W2_raw,
            0.0
        )

        # Get best genome's weights (index 0 after sorting by fitness)
        # W1_masked: (pop, num_inputs, total_positions)
        # W2_masked: (pop, num_outputs, total_positions)
        W1_best = W1_masked[0]  # (num_inputs, total_positions)
        W2_best = W2_masked[0].T  # Transpose to (total_positions, num_outputs)

        # Forward pass matching eval_single_network:
        # inputs: (n_samples, num_inputs)
        # W1_best: (num_inputs, total_positions) -> inputs @ W1_best = (n_samples, total_positions)
        # W2_best: (total_positions, num_outputs) -> hidden @ W2_best = (n_samples, num_outputs)
        hidden = jnp.tanh(inputs @ W1_best)  # (n_samples, total_positions)
        outputs = jax.nn.sigmoid(hidden @ W2_best)  # (n_samples, num_outputs)

        # Compute detailed statistics
        errors = (outputs - targets) ** 2
        mse = float(jnp.mean(errors))
        fitness = max(0.0, min(1.0, 1.0 - mse))

        # Per-sample errors for distribution analysis
        sample_errors = jnp.mean(errors, axis=1)  # MSE per sample

        # Substrate statistics
        mask_best = masks_A[0]  # Active positions for best genome
        active_positions = int(jnp.sum(mask_best))
        total_positions = h_grid.total_positions

        # Connection statistics (non-zero weights)
        W1_nonzero = jnp.sum(jnp.abs(W1_best) > 0)
        W2_nonzero = jnp.sum(jnp.abs(W2_best) > 0)
        total_connections = int(W1_nonzero + W2_nonzero)

        # Weight statistics
        W1_flat = W1_best.flatten()
        W2_flat = W2_best.flatten()
        W1_active = W1_flat[jnp.abs(W1_flat) > 0]
        W2_active = W2_flat[jnp.abs(W2_flat) > 0]

        return {
            'mse': mse,
            'fitness': fitness,
            'n_samples': int(inputs.shape[0]),
            # Error distribution
            'mse_std': float(jnp.std(sample_errors)),
            'mse_min': float(jnp.min(sample_errors)),
            'mse_max': float(jnp.max(sample_errors)),
            'mse_median': float(jnp.median(sample_errors)),
            # Substrate info
            'active_hidden_nodes': active_positions,
            'total_positions': total_positions,
            'total_connections': total_connections,
            'input_connections': int(W1_nonzero),
            'output_connections': int(W2_nonzero),
            # Weight stats
            'w1_mean': float(jnp.mean(W1_active)) if len(W1_active) > 0 else 0.0,
            'w1_std': float(jnp.std(W1_active)) if len(W1_active) > 0 else 0.0,
            'w2_mean': float(jnp.mean(W2_active)) if len(W2_active) > 0 else 0.0,
            'w2_std': float(jnp.std(W2_active)) if len(W2_active) > 0 else 0.0,
            # For substrate visualization
            '_mask': mask_best,
            '_W1': W1_best,
            '_W2': W2_best,
            '_h_grid': h_grid,
        }


# ============================================================================
# Multi-GPU Strategy Implementations
# ============================================================================

# ============================================================================
# Strategy 1: Position-Level Sharding
# ============================================================================

def batch_query_population_positions_sharded(
    state: Any,
    cppns_transformed: Tuple,
    source_coord: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    sharding_config: PositionShardingConfig,
) -> jnp.ndarray:
    """Query ALL CPPNs at ALL target positions using position-level sharding.

    This shards the position dimension across devices, so each GPU processes
    a subset of positions for all candidates. This is the key optimization
    that fixes the EvoX multi-GPU failure (which sharded by population).

    IMPORTANT: This function is designed to be called ONLY when multi-GPU
    position sharding is active. It has NO fallback to avoid Python `if`
    statements that would break JAX tracing inside while_loop.

    The caller (run_until_threshold in HMRHyperNEATMultiGPU) is
    responsible for dispatching to either the sharded or non-sharded path
    BEFORE entering any traced region.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coord: Single source coordinate (2,)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        sharding_config: Position sharding configuration

    Returns:
        (pop_size, num_positions) array of CPPN outputs

    Raises:
        RuntimeError: If shard_map is not available (should never happen
            because caller should check before using this function)
    """
    # NOTE: No Python `if` fallback here to maintain JAX tracing compatibility.
    # The caller handles dispatch BEFORE entering traced regions.

    # Pad positions to be divisible by num_devices
    padded_positions, pad_size = sharding_config.shard_positions(target_positions)
    num_padded_positions = padded_positions.shape[0]
    positions_per_device = num_padded_positions // sharding_config.num_devices

    # Build CPPN inputs for all positions
    source_tiled = jnp.tile(source_coord[None, :], (num_padded_positions, 1))
    bias = jnp.ones((num_padded_positions, 1))

    if outgoing:
        all_inputs = jnp.concatenate([source_tiled, padded_positions, bias], axis=1)
    else:
        all_inputs = jnp.concatenate([padded_positions, source_tiled, bias], axis=1)

    # Reshape for sharding: (num_devices, positions_per_device, input_dim)
    all_inputs_sharded = all_inputs.reshape(
        sharding_config.num_devices, positions_per_device, -1
    )

    mesh = sharding_config.mesh

    # Define sharded query function
    # NOTE: shard_map with P(axis_name) keeps the sharding dimension, so
    # inputs_shard shape is (1, positions_per_device, input_dim) after sharding.
    # We need to squeeze or index to get (positions_per_device, input_dim).
    # NOTE: check_rep=False is needed because the CPPN forward uses while_loop
    # internally and shard_map doesn't have a replication rule for it.
    @partial(shard_map, mesh=mesh,
             in_specs=(P(), P(sharding_config.axis_name)),
             out_specs=P(None, sharding_config.axis_name),
             check_rep=False)
    def sharded_query_positions(cppns_tuple, inputs_shard):
        """Query CPPNs on local position shard.

        Each device processes its local positions for ALL candidates.
        """
        # inputs_shard after shard_map: (1, positions_per_device, input_dim)
        # Squeeze to get: (positions_per_device, input_dim)
        local_inputs = inputs_shard[0]  # Shape: (positions_per_device, input_dim)
        # cppns_tuple: replicated across devices

        def query_single_cppn(cppn_tuple):
            """Query one CPPN at local positions."""
            weights = jax.vmap(
                lambda x: cppn_forward(state, cppn_tuple, x)
            )(local_inputs)
            return weights.flatten()

        # vmap over population
        all_weights = jax.vmap(
            query_single_cppn,
            in_axes=((0, 0, 0, 0),)
        )((cppns_tuple[0], cppns_tuple[1],
           cppns_tuple[2], cppns_tuple[3]))

        return all_weights  # (pop_size, positions_per_device)

    # Execute sharded query
    with mesh:
        result = sharded_query_positions(cppns_transformed, all_inputs_sharded)

    # Reshape result: (pop_size, num_padded_positions)
    pop_size = cppns_transformed[0].shape[0]
    result = result.reshape(pop_size, num_padded_positions)

    # Remove padding
    if pad_size > 0:
        result = result[:, :-pad_size]

    return result


def batch_query_population_multi_source_sharded(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    sharding_config: PositionShardingConfig,
) -> jnp.ndarray:
    """Query ALL CPPNs from ALL sources to ALL positions with position sharding.

    This is the main entry point for position-level parallelism across GPUs.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        sharding_config: Position sharding configuration

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    # vmap over sources (this is small, typically 3-10)
    def query_from_source(source_coord):
        return batch_query_population_positions_sharded(
            state, cppns_transformed, source_coord, target_positions,
            outgoing, cppn_forward, sharding_config
        )

    # Result: (num_sources, pop_size, num_positions)
    result = jax.vmap(query_from_source)(source_coords)

    # Transpose to (pop_size, num_sources, num_positions)
    return jnp.transpose(result, (1, 0, 2))


def batch_query_population_multi_source_chunked_sharded(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    sharding_config: PositionShardingConfig,
    pop_chunk_size: int = 100,
) -> jnp.ndarray:
    """Chunked + sharded version: chunks population, shards positions across GPUs.

    This combines two optimization strategies:
    1. Population chunking: Process population in smaller chunks to reduce peak memory
    2. Position sharding: Distribute positions across GPUs for parallel execution

    Use this for depth 6+ where:
    - Position count is high (5,461+ positions)
    - Memory is constrained
    - Multi-GPU speedup is beneficial

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        sharding_config: Position sharding configuration
        pop_chunk_size: Number of genomes to process at once (default 100)

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    pop_size = cppns_transformed[0].shape[0]
    num_sources = source_coords.shape[0]

    results_list = []

    # Iterate over sources (typically 3-10 for XOR)
    for source_idx in range(num_sources):
        source_coord = source_coords[source_idx]

        chunk_results = []
        # Chunked processing over population
        for chunk_start in range(0, pop_size, pop_chunk_size):
            chunk_end = min(chunk_start + pop_chunk_size, pop_size)

            # Extract chunk of CPPNs
            chunk_cppns = tuple(
                arr[chunk_start:chunk_end] for arr in cppns_transformed
            )

            # Use sharded query for this chunk - positions sharded across GPUs
            chunk_weights = batch_query_population_positions_sharded(
                state, chunk_cppns, source_coord, target_positions,
                outgoing, cppn_forward, sharding_config
            )
            chunk_results.append(chunk_weights)

        # Concatenate chunks for this source: (pop_size, num_positions)
        source_weights = jnp.concatenate(chunk_results, axis=0)
        results_list.append(source_weights)

    # Stack sources: (num_sources, pop_size, num_positions)
    result = jnp.stack(results_list, axis=0)
    # Transpose to: (pop_size, num_sources, num_positions)
    return jnp.transpose(result, (1, 0, 2))


# ============================================================================
# Full Pipeline Chunking (Weight Matrix + Network Evaluation)
# ============================================================================

def build_and_evaluate_chunked(
    input_all_weights: jnp.ndarray,
    output_all_weights: jnp.ndarray,
    masks_A: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    pop_chunk_size: int = 200,
    max_weight: float = 8.0,
    weight_thresh: float = 0.1,
) -> jnp.ndarray:
    """Build weight matrices AND evaluate networks in unified chunks (GPU-RESIDENT).

    CRITICAL: This function uses lax.fori_loop for GPU-resident execution.
    Processes W1/W2 construction AND network evaluation together per chunk,
    so we NEVER hold full (pop, inputs, positions) arrays.

    This is the KEY to enabling depth 7 (87,380 positions) with pop=1000 on 11 GiB GPUs.

    Memory flow per chunk (200 genomes at depth 7):
    1. Extract chunk of input_all_weights: (200, 4, 87380) = 280 MB
    2. Build W1_chunk, W2_chunk: (200, 4, 87380) + (200, 87380, 1) = 350 MB
    3. Evaluate fitness_chunk via vmap: (200,) = negligible
    4. DISCARD W1_chunk, W2_chunk before next iteration
    5. Peak: ~630 MB instead of 3-4 GB

    Args:
        input_all_weights: CPPN outputs for input→position (pop, inputs, positions)
        output_all_weights: CPPN outputs for position→output (pop, outputs, positions)
        masks_A: Active position masks (pop, positions)
        inputs_batch: Problem inputs (batch_size, inputs)
        targets_batch: Problem targets (batch_size, outputs)
        pop_chunk_size: Genomes per chunk (default 200)
        max_weight: Weight scaling factor
        weight_thresh: Pruning threshold

    Returns:
        fitnesses: (pop,) fitness values accumulated from all chunks
    """
    pop_size = input_all_weights.shape[0]
    num_chunks = (pop_size + pop_chunk_size - 1) // pop_chunk_size

    def eval_single_network(W1_single, W2_single, inputs, targets):
        """Evaluate a single network's fitness."""
        hidden = jnp.tanh(safe_matmul(inputs, W1_single))
        outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
        errors = jnp.mean((outputs - targets) ** 2, axis=1)
        return 1.0 - jnp.mean(errors)

    def process_chunk(chunk_idx, fitnesses_acc):
        """Process a single chunk - GPU-resident via lax.fori_loop."""
        chunk_start = chunk_idx * pop_chunk_size
        chunk_end = jnp.minimum(chunk_start + pop_chunk_size, pop_size)
        actual_chunk_size = chunk_end - chunk_start

        # Extract chunk slices using dynamic_slice for JIT compatibility
        chunk_input = jax.lax.dynamic_slice(
            input_all_weights,
            (chunk_start, 0, 0),
            (pop_chunk_size, input_all_weights.shape[1], input_all_weights.shape[2])
        )
        chunk_output = jax.lax.dynamic_slice(
            output_all_weights,
            (chunk_start, 0, 0),
            (pop_chunk_size, output_all_weights.shape[1], output_all_weights.shape[2])
        )
        chunk_masks = jax.lax.dynamic_slice(
            masks_A,
            (chunk_start, 0),
            (pop_chunk_size, masks_A.shape[1])
        )

        # Build weight matrices for THIS CHUNK ONLY
        W1_raw_chunk = jnp.tanh(chunk_input) * max_weight
        W2_raw_chunk = jnp.tanh(chunk_output) * max_weight

        # Apply masks
        mask_broadcast = chunk_masks[:, None, :]  # (chunk, 1, positions)
        W1_mask = mask_broadcast & (jnp.abs(W1_raw_chunk) > weight_thresh)
        W2_mask = mask_broadcast & (jnp.abs(W2_raw_chunk) > weight_thresh)

        W1_chunk = W1_raw_chunk * W1_mask
        W2_chunk = (W2_raw_chunk * W2_mask).transpose(0, 2, 1)  # (chunk, pos, outputs)

        # Evaluate THIS CHUNK
        fitnesses_chunk = jax.vmap(
            eval_single_network,
            in_axes=(0, 0, None, None)
        )(W1_chunk, W2_chunk, inputs_batch, targets_batch)

        # Write results to accumulator using dynamic_update_slice
        fitnesses_acc = jax.lax.dynamic_update_slice(
            fitnesses_acc, fitnesses_chunk, (chunk_start,)
        )

        return fitnesses_acc

    # Pad population to be divisible by chunk size for uniform slicing
    padded_pop_size = num_chunks * pop_chunk_size
    if padded_pop_size > pop_size:
        pad_size = padded_pop_size - pop_size
        input_all_weights = jnp.pad(
            input_all_weights, ((0, pad_size), (0, 0), (0, 0)), mode='constant'
        )
        output_all_weights = jnp.pad(
            output_all_weights, ((0, pad_size), (0, 0), (0, 0)), mode='constant'
        )
        masks_A = jnp.pad(masks_A, ((0, pad_size), (0, 0)), mode='constant')

    # Initialize fitness accumulator
    fitnesses = jnp.full(padded_pop_size, -jnp.inf, dtype=jnp.float32)

    # GPU-resident loop over chunks
    fitnesses = jax.lax.fori_loop(0, num_chunks, process_chunk, fitnesses)

    # Remove padding and handle NaN
    fitnesses = fitnesses[:pop_size]
    return jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)


def build_and_evaluate_chunked_multi_gpu(
    input_all_weights: jnp.ndarray,
    output_all_weights: jnp.ndarray,
    masks_A: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    pop_chunk_size: int = 200,
    max_weight: float = 8.0,
    weight_thresh: float = 0.1,
    num_gpus: int = 2,
) -> jnp.ndarray:
    """Build weight matrices AND evaluate networks using MULTI-GPU pmap.

    This distributes population chunks across multiple GPUs for parallel evaluation.
    Each GPU processes pop_size/num_gpus networks independently.

    For 75K samples with chunk_size=4:
    - Single GPU: Sequential chunks, ~1.6GB per chunk
    - Dual GPU: Parallel chunks, each GPU handles half the population

    Args:
        input_all_weights: CPPN outputs for input→position (pop, inputs, positions)
        output_all_weights: CPPN outputs for position→output (pop, outputs, positions)
        masks_A: Active position masks (pop, positions)
        inputs_batch: Problem inputs (batch_size, inputs)
        targets_batch: Problem targets (batch_size, outputs)
        pop_chunk_size: Genomes per chunk within each GPU (default 200)
        max_weight: Weight scaling factor
        weight_thresh: Pruning threshold
        num_gpus: Number of GPUs to distribute across

    Returns:
        fitnesses: (pop,) fitness values accumulated from all GPUs
    """
    pop_size = input_all_weights.shape[0]

    # Pad population to be divisible by num_gpus
    remainder = pop_size % num_gpus
    if remainder != 0:
        pad_size = num_gpus - remainder
        # Pad with zeros (will produce -inf fitness due to nan handling)
        input_all_weights = jnp.pad(
            input_all_weights, ((0, pad_size), (0, 0), (0, 0)), mode='constant'
        )
        output_all_weights = jnp.pad(
            output_all_weights, ((0, pad_size), (0, 0), (0, 0)), mode='constant'
        )
        masks_A = jnp.pad(masks_A, ((0, pad_size), (0, 0)), mode='constant')
    else:
        pad_size = 0

    padded_pop_size = input_all_weights.shape[0]
    per_gpu_size = padded_pop_size // num_gpus

    # Reshape to (num_gpus, per_gpu_size, ...)
    input_per_gpu = input_all_weights.reshape(num_gpus, per_gpu_size, *input_all_weights.shape[1:])
    output_per_gpu = output_all_weights.reshape(num_gpus, per_gpu_size, *output_all_weights.shape[1:])
    masks_per_gpu = masks_A.reshape(num_gpus, per_gpu_size, *masks_A.shape[1:])

    def eval_single_network(W1_single, W2_single, inputs, targets):
        """Evaluate a single network's fitness."""
        hidden = jnp.tanh(safe_matmul(inputs, W1_single))
        outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
        errors = jnp.mean((outputs - targets) ** 2, axis=1)
        return 1.0 - jnp.mean(errors)

    def process_gpu_chunk(gpu_inputs, gpu_outputs, gpu_masks, inputs_batch, targets_batch):
        """Process one GPU's share of the population using chunked evaluation.

        Uses lax.fori_loop for GPU-resident chunked processing on each device.
        """
        gpu_pop_size = gpu_inputs.shape[0]
        num_gpu_chunks = (gpu_pop_size + pop_chunk_size - 1) // pop_chunk_size

        def process_chunk(chunk_idx, fitnesses_acc):
            """Process a single chunk - GPU-resident via lax.fori_loop."""
            chunk_start = chunk_idx * pop_chunk_size

            # Extract chunk slices using dynamic_slice for JIT compatibility
            chunk_input = jax.lax.dynamic_slice(
                gpu_inputs,
                (chunk_start, 0, 0),
                (pop_chunk_size, gpu_inputs.shape[1], gpu_inputs.shape[2])
            )
            chunk_output = jax.lax.dynamic_slice(
                gpu_outputs,
                (chunk_start, 0, 0),
                (pop_chunk_size, gpu_outputs.shape[1], gpu_outputs.shape[2])
            )
            chunk_masks = jax.lax.dynamic_slice(
                gpu_masks,
                (chunk_start, 0),
                (pop_chunk_size, gpu_masks.shape[1])
            )

            # Build weight matrices
            W1_raw_chunk = jnp.tanh(chunk_input) * max_weight
            W2_raw_chunk = jnp.tanh(chunk_output) * max_weight

            # Apply masks
            mask_broadcast = chunk_masks[:, None, :]
            W1_mask = mask_broadcast & (jnp.abs(W1_raw_chunk) > weight_thresh)
            W2_mask = mask_broadcast & (jnp.abs(W2_raw_chunk) > weight_thresh)

            W1_chunk = W1_raw_chunk * W1_mask
            W2_chunk = (W2_raw_chunk * W2_mask).transpose(0, 2, 1)

            # Evaluate this chunk
            fitnesses_chunk = jax.vmap(
                eval_single_network,
                in_axes=(0, 0, None, None)
            )(W1_chunk, W2_chunk, inputs_batch, targets_batch)

            # Write results to accumulator
            fitnesses_acc = jax.lax.dynamic_update_slice(
                fitnesses_acc, fitnesses_chunk, (chunk_start,)
            )
            return fitnesses_acc

        # Pad to be divisible by chunk size
        padded_gpu_size = num_gpu_chunks * pop_chunk_size
        if padded_gpu_size > gpu_pop_size:
            pad_size = padded_gpu_size - gpu_pop_size
            gpu_inputs = jnp.pad(gpu_inputs, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            gpu_outputs = jnp.pad(gpu_outputs, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            gpu_masks = jnp.pad(gpu_masks, ((0, pad_size), (0, 0)), mode='constant')

        # Initialize and run GPU-resident loop
        fitnesses = jnp.full(padded_gpu_size, -jnp.inf, dtype=jnp.float32)
        fitnesses = jax.lax.fori_loop(0, num_gpu_chunks, process_chunk, fitnesses)

        # Return only the valid portion
        return fitnesses[:gpu_pop_size]

    # Use pmap to distribute across GPUs
    # Each GPU processes its share of the population
    pmapped_eval = jax.pmap(
        process_gpu_chunk,
        in_axes=(0, 0, 0, None, None),  # Shard first 3 args across GPUs
    )

    # Execute on all GPUs in parallel
    fitnesses_per_gpu = pmapped_eval(
        input_per_gpu, output_per_gpu, masks_per_gpu,
        inputs_batch, targets_batch
    )

    # Flatten results: (num_gpus, per_gpu_size) -> (padded_pop_size,)
    fitnesses = fitnesses_per_gpu.reshape(-1)

    # Remove padding if any
    if pad_size > 0:
        fitnesses = fitnesses[:pop_size]

    return jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)


# ============================================================================
# Strategy: Population-Level Pmap Parallelism
# ============================================================================

def batch_query_population_multi_source_pmap(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    pmap_config: PopulationPmapConfig,
) -> jnp.ndarray:
    """Query ALL CPPNs from ALL sources to ALL positions using population pmap.

    This shards the POPULATION dimension across GPUs using jax.pmap:
    - GPU 0: candidates 0-(pop_size/2-1) at ALL positions
    - GPU 1: candidates (pop_size/2)-(pop_size-1) at ALL positions

    This is more efficient than position sharding because:
    1. pmap has lower overhead than shard_map
    2. No synchronization during computation
    3. Each GPU runs completely independent CPPN evaluation
    4. Memory usage per GPU is halved

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        pmap_config: Population pmap configuration

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    num_devices = pmap_config.num_devices
    pop_size = cppns_transformed[0].shape[0]
    pop_per_device = pop_size // num_devices
    num_positions = target_positions.shape[0]
    num_sources = source_coords.shape[0]

    # Shard CPPNs across devices: (pop_size, ...) -> (num_devices, pop_per_device, ...)
    cppns_sharded = pmap_config.shard_population(cppns_transformed)

    # Build CPPN inputs for all source-position pairs
    # Shape: (num_sources, num_positions, 5) for (src_x, src_y, tgt_x, tgt_y, bias)
    def build_inputs_for_source(source_coord):
        source_tiled = jnp.tile(source_coord[None, :], (num_positions, 1))
        bias = jnp.ones((num_positions, 1))
        if outgoing:
            return jnp.concatenate([source_tiled, target_positions, bias], axis=1)
        else:
            return jnp.concatenate([target_positions, source_tiled, bias], axis=1)

    all_inputs = jax.vmap(build_inputs_for_source)(source_coords)
    # all_inputs shape: (num_sources, num_positions, 5)

    # Define pmap'd function that processes a slice of the population
    @partial(jax.pmap, axis_name=pmap_config.pmap_axis_name)
    def query_population_slice(cppns_slice):
        """Query one slice of population on all positions.

        Args:
            cppns_slice: Tuple of 4 arrays, each (pop_per_device, ...)

        Returns:
            (pop_per_device, num_sources, num_positions) array
        """
        def query_single_cppn(cppn_tuple):
            """Query single CPPN at all source-position pairs."""
            def query_at_source(inputs_for_source):
                # inputs_for_source: (num_positions, 5)
                weights = jax.vmap(
                    lambda x: cppn_forward(state, cppn_tuple, x)
                )(inputs_for_source)
                return weights.flatten()  # (num_positions,)

            # Query for all sources: (num_sources, num_positions)
            return jax.vmap(query_at_source)(all_inputs)

        # vmap over population slice
        return jax.vmap(
            query_single_cppn,
            in_axes=((0, 0, 0, 0),)
        )((cppns_slice[0], cppns_slice[1],
           cppns_slice[2], cppns_slice[3]))

    # Execute pmap'd query
    # Result shape: (num_devices, pop_per_device, num_sources, num_positions)
    result_sharded = query_population_slice(cppns_sharded)

    # Gather results: (pop_size, num_sources, num_positions)
    result = result_sharded.reshape(pop_size, num_sources, num_positions)

    return result


# ============================================================================
# Strategy: Position-Level Pmap Parallelism (for Large Depths)
# ============================================================================

def batch_query_population_multi_source_position_pmap(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    position_config: PositionShardingConfig,
) -> jnp.ndarray:
    """Query ALL CPPNs from ALL sources with POSITION-level pmap sharding.

    This shards the POSITION dimension across GPUs using device placement + pmap:
    - GPU 0: positions 0 to N/2 for ALL population members
    - GPU 1: positions N/2 to N for ALL population members

    This is designed for LARGE DEPTHS (7+) where:
    - Position count is massive (87,380 at depth 7, 349,524 at depth 8)
    - Single GPU cannot efficiently process all positions
    - Memory distribution is critical

    Unlike shard_map approach (which had check_rep=False overhead),
    this uses explicit data distribution with device_put_sharded/replicated.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2) - should be padded
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        position_config: Position sharding configuration

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    num_devices = position_config.num_devices
    devices = position_config.devices[:num_devices]
    pop_size = cppns_transformed[0].shape[0]
    num_positions = target_positions.shape[0]
    num_sources = source_coords.shape[0]
    positions_per_device = num_positions // num_devices

    # Reshape positions for device distribution: (num_devices, positions_per_device, 2)
    positions_sharded = target_positions.reshape(num_devices, positions_per_device, 2)

    # Build CPPN inputs for all source-position combinations per device shard
    # For each device, we need: (num_sources, positions_per_device, 5)
    def build_inputs_for_device_shard(positions_slice):
        """Build inputs for one device's position slice."""
        def build_for_source(source_coord):
            source_tiled = jnp.tile(source_coord[None, :], (positions_per_device, 1))
            bias = jnp.ones((positions_per_device, 1))
            if outgoing:
                return jnp.concatenate([source_tiled, positions_slice, bias], axis=1)
            else:
                return jnp.concatenate([positions_slice, source_tiled, bias], axis=1)
        return jax.vmap(build_for_source)(source_coords)

    # Build inputs for all device shards: (num_devices, num_sources, positions_per_device, 5)
    all_inputs_sharded = jax.vmap(build_inputs_for_device_shard)(positions_sharded)

    # Distribute data to devices
    # CPPNs: replicate to all devices (each device needs full population)
    # Inputs: shard across devices (each device gets its position slice)
    cppns_replicated = jax.device_put_replicated(cppns_transformed, devices)
    inputs_on_devices = jax.device_put_sharded(
        [all_inputs_sharded[i] for i in range(num_devices)], devices
    )

    # Define pmap'd function to query positions on each device
    @partial(jax.pmap, axis_name='device')
    def query_positions_on_device(cppns_tuple, inputs_for_device):
        """Query all CPPNs at local position slice.

        Args:
            cppns_tuple: Full CPPN population tuple (replicated)
            inputs_for_device: (num_sources, positions_per_device, 5) for this device

        Returns:
            (pop_size, num_sources, positions_per_device) array
        """
        def query_single_cppn(cppn_tuple):
            """Query one CPPN at all source-position pairs on this device."""
            def query_at_source(inputs_for_source):
                # inputs_for_source: (positions_per_device, 5)
                weights = jax.vmap(
                    lambda x: cppn_forward(state, cppn_tuple, x)
                )(inputs_for_source)
                # weights shape: (positions_per_device, 1) - squeeze trailing dim, not flatten
                return weights.squeeze(-1)  # (positions_per_device,)

            # Query for all sources: (num_sources, positions_per_device)
            return jax.vmap(query_at_source)(inputs_for_device)

        # vmap over population
        return jax.vmap(
            query_single_cppn,
            in_axes=((0, 0, 0, 0),)
        )((cppns_tuple[0], cppns_tuple[1],
           cppns_tuple[2], cppns_tuple[3]))

    # Execute pmap'd query
    # Result shape: (num_devices, pop_size, num_sources, positions_per_device)
    result_sharded = query_positions_on_device(cppns_replicated, inputs_on_devices)

    # Gather and reshape: (pop_size, num_sources, num_positions)
    # Transpose to move device dimension to end, then reshape
    # (num_devices, pop_size, num_sources, positions_per_device) ->
    # (pop_size, num_sources, num_devices, positions_per_device) ->
    # (pop_size, num_sources, num_positions)
    result = jnp.transpose(result_sharded, (1, 2, 0, 3))
    result = result.reshape(pop_size, num_sources, num_positions)

    return result


def query_positions_batched_multi_gpu(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    all_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    position_config: PositionShardingConfig,
    batch_size: int = 10000,  # Reduced from 50000 to avoid OOM at depth 7+
) -> jnp.ndarray:
    """Query positions in batches with multi-GPU distribution.

    For very large depths (8+), even distributed computation needs batching
    to manage memory. This function:
    1. Splits positions into manageable batches
    2. Pads each batch for even device distribution
    3. Distributes each batch across GPUs
    4. Concatenates results

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of CPPN arrays
        source_coords: (num_sources, 2) array of source coordinates
        all_positions: (total_positions, 2) array of ALL positions
        outgoing: Connection direction flag
        cppn_forward: CPPN forward function
        position_config: Multi-GPU configuration
        batch_size: Positions per batch (before padding)

    Returns:
        (pop_size, num_sources, total_positions) array of weights
    """
    total_positions = all_positions.shape[0]
    num_devices = position_config.num_devices

    if total_positions <= batch_size:
        # No batching needed - just pad and query
        padded = position_config.pad_positions(all_positions)
        result = batch_query_population_multi_source_position_pmap(
            state, cppns_transformed, source_coords, padded,
            outgoing, cppn_forward, position_config
        )
        return position_config.unpad_results(result, total_positions)

    # Process in batches
    results = []
    for start in range(0, total_positions, batch_size):
        end = min(start + batch_size, total_positions)
        batch_positions = all_positions[start:end]
        batch_size_actual = end - start

        # Pad batch for even distribution
        padded = position_config.pad_positions(batch_positions)

        # Query this batch across GPUs
        batch_result = batch_query_population_multi_source_position_pmap(
            state, cppns_transformed, source_coords, padded,
            outgoing, cppn_forward, position_config
        )

        # Unpad and collect
        batch_result = position_config.unpad_results(batch_result, batch_size_actual)
        results.append(batch_result)

    # Concatenate along position axis
    return jnp.concatenate(results, axis=-1)


# ============================================================================
# Optimized Population-Parallel Full Pipeline (No Data Reshuffling)
# ============================================================================

def _full_pipeline_single_gpu(
    cppns_transformed_slice: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[Tuple[int, int], ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    variance_threshold: float,
    max_weight: float,
    weight_thresh: float,
    cppn_forward: Any,
    state: Any,
) -> jnp.ndarray:
    """Run FULL HMR-HyperNEAT pipeline for a population slice on ONE GPU.

    This function processes the ENTIRE pipeline for a subset of the population:
    1. CPPN queries for variance
    2. Hierarchical variance computation
    3. Subdivision mask computation
    4. CPPN queries for input/output weights
    5. Weight matrix construction
    6. Network evaluation

    By keeping all data local to one GPU throughout, we avoid the overhead
    of reshuffling data between position-sharding and population-sharding.

    Args:
        cppns_transformed_slice: Tuple of CPPN arrays for this GPU's population slice
        all_positions: All hierarchical grid positions (num_positions, 2)
        input_coords: Substrate input coordinates (num_inputs, 2)
        output_coords: Substrate output coordinates (num_outputs, 2)
        inputs_batch: Problem inputs (n_samples, num_inputs)
        targets_batch: Problem targets (n_samples, num_outputs)
        level_sizes/offsets/grid_sizes: Hierarchical grid info
        parent_indices: Parent-child relationships for subdivision
        num_levels: Number of hierarchy levels
        total_positions: Total position count
        variance_threshold: Threshold for subdivision
        max_weight: Maximum weight value
        weight_thresh: Threshold for weight pruning
        cppn_forward: JIT-compiled CPPN forward function
        state: Algorithm state for CPPN queries

    Returns:
        Fitness values for this population slice (per_gpu_pop_size,)
    """
    per_gpu_pop = cppns_transformed_slice[0].shape[0]
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]
    num_positions = all_positions.shape[0]

    # Helper: Query single CPPN at all positions from one source
    def query_cppn_single_source(cppn_tuple, source_coord, outgoing):
        """Query one CPPN from source to all positions."""
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
        """Query one CPPN from all sources to all positions."""
        return jax.vmap(
            lambda src: query_cppn_single_source(cppn_tuple, src, outgoing)
        )(source_coords)

    # STEP 1: Query CPPN for variance (first input coord only)
    def get_variance_weights(cppn_tuple):
        return query_cppn_single_source(cppn_tuple, input_coords[0], outgoing=True)

    all_weights_for_variance = jax.vmap(
        get_variance_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed_slice[0], cppns_transformed_slice[1],
       cppns_transformed_slice[2], cppns_transformed_slice[3]))
    # Shape: (per_gpu_pop, num_positions)

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

    # STEP 4: Query CPPN for input weights (all inputs -> all positions)
    def get_input_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, input_coords, outgoing=True)

    input_all_weights = jax.vmap(
        get_input_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed_slice[0], cppns_transformed_slice[1],
       cppns_transformed_slice[2], cppns_transformed_slice[3]))
    # Shape: (per_gpu_pop, num_inputs, num_positions)

    # STEP 5: Query CPPN for output weights (all positions -> all outputs)
    def get_output_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, output_coords, outgoing=False)

    output_all_weights = jax.vmap(
        get_output_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed_slice[0], cppns_transformed_slice[1],
       cppns_transformed_slice[2], cppns_transformed_slice[3]))
    # Shape: (per_gpu_pop, num_outputs, num_positions)

    # STEP 6: Build weight matrices with mask
    mask_broadcast = masks_A[:, None, :]  # (per_gpu_pop, 1, num_positions)

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
    W2 = jnp.transpose(W2_masked, (0, 2, 1))  # (per_gpu_pop, num_positions, num_outputs)

    # STEP 7: Evaluate all networks
    def eval_single_network(W1_single, W2_single):
        hidden = jnp.tanh(inputs_batch @ W1_single)  # (n_samples, num_positions)
        outputs = jax.nn.sigmoid(hidden @ W2_single)  # (n_samples, num_outputs)
        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
        return 1.0 - jnp.mean(errors)

    fitnesses = jax.vmap(eval_single_network)(W1, W2)
    return jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)


def _full_pipeline_single_gpu_with_hh(
    cppns_transformed_slice: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    sparse_from_idx: jnp.ndarray,
    sparse_to_idx: jnp.ndarray,
    sparse_weights: jnp.ndarray,
    sparse_valid_mask: jnp.ndarray,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[Tuple[int, int], ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    variance_threshold: float,
    max_weight: float,
    weight_thresh: float,
    activate_time: int,
    cppn_forward: Any,
    state: Any,
    num_positions_param: int = None,  # Optional - if None, derived from all_positions
) -> jnp.ndarray:
    """Run FULL HMR-HyperNEAT pipeline with SPARSE H→H connections for a population slice.

    This function processes the ENTIRE pipeline including h→h recurrent connections
    for a subset of the population on ONE GPU:
    1. CPPN queries for variance
    2. Hierarchical variance computation
    3. Subdivision mask computation
    4. CPPN queries for input/output weights
    5. Weight matrix construction (W1, W2)
    6. Network evaluation WITH SPARSE H→H scatter-add

    This is the KEY function for achieving true multi-GPU speedup with h→h caching.
    By wrapping this in pmap, we get SINGLE pmap call per generation instead of
    6+ separate Python→JAX calls.

    Args:
        cppns_transformed_slice: Tuple of CPPN arrays for this GPU's population slice
        all_positions: All hierarchical grid positions (num_positions, 2)
        input_coords: Substrate input coordinates (num_inputs, 2)
        output_coords: Substrate output coordinates (num_outputs, 2)
        inputs_batch: Problem inputs (n_samples, num_inputs)
        targets_batch: Problem targets (n_samples, num_outputs)
        sparse_from_idx: Sparse h→h source indices (per_gpu_pop, max_sparse_conns)
        sparse_to_idx: Sparse h→h target indices (per_gpu_pop, max_sparse_conns)
        sparse_weights: Sparse h→h connection weights (per_gpu_pop, max_sparse_conns)
        sparse_valid_mask: Sparse h→h validity mask (per_gpu_pop, max_sparse_conns)
        level_sizes/offsets/grid_sizes: Hierarchical grid info
        parent_indices: Parent-child relationships for subdivision
        num_levels: Number of hierarchy levels
        total_positions: Total position count
        variance_threshold: Threshold for subdivision
        max_weight: Maximum weight value
        weight_thresh: Threshold for weight pruning
        activate_time: Number of recurrent iterations for h→h
        cppn_forward: JIT-compiled CPPN forward function
        state: Algorithm state for CPPN queries
        num_positions_param: Static num_positions (for pmap tracing)

    Returns:
        Fitness values for this population slice (per_gpu_pop_size,)
    """
    per_gpu_pop = cppns_transformed_slice[0].shape[0]
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]
    # Use provided num_positions if available (for static pmap args), else derive from array
    num_positions = num_positions_param if num_positions_param is not None else all_positions.shape[0]

    # Helper: Query single CPPN at all positions from one source
    def query_cppn_single_source(cppn_tuple, source_coord, outgoing):
        """Query one CPPN from source to all positions."""
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
        """Query one CPPN from all sources to all positions."""
        return jax.vmap(
            lambda src: query_cppn_single_source(cppn_tuple, src, outgoing)
        )(source_coords)

    # STEP 1: Query CPPN for variance (first input coord only)
    def get_variance_weights(cppn_tuple):
        return query_cppn_single_source(cppn_tuple, input_coords[0], outgoing=True)

    all_weights_for_variance = jax.vmap(
        get_variance_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed_slice[0], cppns_transformed_slice[1],
       cppns_transformed_slice[2], cppns_transformed_slice[3]))
    # Shape: (per_gpu_pop, num_positions)

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

    # STEP 4: Query CPPN for input weights (all inputs -> all positions)
    def get_input_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, input_coords, outgoing=True)

    input_all_weights = jax.vmap(
        get_input_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed_slice[0], cppns_transformed_slice[1],
       cppns_transformed_slice[2], cppns_transformed_slice[3]))
    # Shape: (per_gpu_pop, num_inputs, num_positions)

    # STEP 5: Query CPPN for output weights (all positions -> all outputs)
    def get_output_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, output_coords, outgoing=False)

    output_all_weights = jax.vmap(
        get_output_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed_slice[0], cppns_transformed_slice[1],
       cppns_transformed_slice[2], cppns_transformed_slice[3]))
    # Shape: (per_gpu_pop, num_outputs, num_positions)

    # STEP 6: Build weight matrices with mask
    mask_broadcast = masks_A[:, None, :]  # (per_gpu_pop, 1, num_positions)

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
    W2 = jnp.transpose(W2_masked, (0, 2, 1))  # (per_gpu_pop, num_positions, num_outputs)

    # STEP 7: Evaluate all networks WITH SPARSE H→H (recurrent)
    def eval_single_network_with_hh(W1_single, W2_single, from_idx, to_idx, hh_weights, hh_valid):
        """Evaluate single network with sparse h→h connections (recurrent)."""
        n_samples = inputs_batch.shape[0]
        num_pos = num_positions

        # Input contribution (constant across iterations)
        input_contrib = safe_matmul(inputs_batch, W1_single)  # (n_samples, num_positions)
        hidden = jnp.zeros((n_samples, num_pos))

        # Precompute safe indices and masked weights
        safe_from = jnp.clip(from_idx, 0, num_pos - 1)
        safe_to = jnp.clip(to_idx, 0, num_pos - 1)
        effective_weights = jnp.where(hh_valid, hh_weights, 0.0)

        def sparse_hh_step(hidden, _):
            """Single h→h iteration using sparse scatter-add."""
            # Gather from source positions: (n_samples, max_sparse_conns)
            source_vals = hidden[:, safe_from]

            # Multiply by connection weights
            contributions = source_vals * effective_weights

            # Scatter-add to target positions
            h_delta = jnp.zeros_like(hidden)
            h_delta = h_delta.at[:, safe_to].add(contributions)

            # Combine input and recurrent contributions
            return jnp.tanh(input_contrib + h_delta), None

        # Run recurrent iterations
        hidden_final, _ = jax.lax.scan(sparse_hh_step, hidden, None, length=activate_time)

        # Output layer
        outputs = jax.nn.sigmoid(safe_matmul(hidden_final, W2_single))
        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
        return 1.0 - jnp.mean(errors)

    fitnesses = jax.vmap(eval_single_network_with_hh)(
        W1, W2, sparse_from_idx, sparse_to_idx, sparse_weights, sparse_valid_mask
    )
    return jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)


def _full_pipeline_single_gpu_chunked(
    cppns_transformed_slice: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[Tuple[int, int], ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    variance_threshold: float,
    max_weight: float,
    weight_thresh: float,
    cppn_forward: Any,
    state: Any,
    pop_chunk_size: int = 50,
) -> jnp.ndarray:
    """Run FULL HMR-HyperNEAT pipeline with CHUNKING for large datasets.

    Same as _full_pipeline_single_gpu but processes individuals in smaller
    chunks to avoid OOM with large datasets. Uses lax.fori_loop for GPU
    efficiency.

    Memory constraint: With 20K samples and 1364 positions, each network
    evaluation uses ~100MB. Processing 50 networks at a time = 5GB peak.

    Args:
        ... (same as _full_pipeline_single_gpu)
        pop_chunk_size: Number of individuals to process at once (default 50)

    Returns:
        Fitness values for this population slice (per_gpu_pop_size,)
    """
    per_gpu_pop = cppns_transformed_slice[0].shape[0]
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]
    num_positions = all_positions.shape[0]

    # Ensure chunk size divides evenly (pad if needed)
    remainder = per_gpu_pop % pop_chunk_size
    if remainder != 0:
        pad_size = pop_chunk_size - remainder
        cppns_transformed_slice = tuple(
            jnp.pad(arr, ((0, pad_size),) + ((0, 0),) * (arr.ndim - 1), mode='constant')
            for arr in cppns_transformed_slice
        )
        padded_pop = per_gpu_pop + pad_size
    else:
        pad_size = 0
        padded_pop = per_gpu_pop

    num_chunks = padded_pop // pop_chunk_size

    # Pre-allocate output array
    all_fitnesses = jnp.zeros(padded_pop, dtype=jnp.float32)

    # Helper: Query single CPPN at all positions from one source
    def query_cppn_single_source(cppn_tuple, source_coord, outgoing):
        """Query one CPPN from source to all positions."""
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
        """Query one CPPN from all sources to all positions."""
        return jax.vmap(
            lambda src: query_cppn_single_source(cppn_tuple, src, outgoing)
        )(source_coords)

    def process_chunk(chunk_idx, fitnesses_acc):
        """Process one chunk of individuals."""
        start_idx = chunk_idx * pop_chunk_size
        end_idx = start_idx + pop_chunk_size

        # Extract chunk of CPPNs
        cppns_chunk = tuple(
            jax.lax.dynamic_slice(arr, (start_idx,) + (0,) * (arr.ndim - 1),
                                  (pop_chunk_size,) + arr.shape[1:])
            for arr in cppns_transformed_slice
        )

        # STEP 1: Query CPPN for variance (first input coord only)
        def get_variance_weights(cppn_tuple):
            return query_cppn_single_source(cppn_tuple, input_coords[0], outgoing=True)

        all_weights_for_variance = jax.vmap(
            get_variance_weights,
            in_axes=((0, 0, 0, 0),)
        )((cppns_chunk[0], cppns_chunk[1], cppns_chunk[2], cppns_chunk[3]))

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
        )((cppns_chunk[0], cppns_chunk[1], cppns_chunk[2], cppns_chunk[3]))

        # STEP 5: Query CPPN for output weights
        def get_output_weights(cppn_tuple):
            return query_cppn_multi_source(cppn_tuple, output_coords, outgoing=False)

        output_all_weights = jax.vmap(
            get_output_weights,
            in_axes=((0, 0, 0, 0),)
        )((cppns_chunk[0], cppns_chunk[1], cppns_chunk[2], cppns_chunk[3]))

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

        # STEP 7: Evaluate all networks in chunk
        def eval_single_network(W1_single, W2_single):
            hidden = jnp.tanh(inputs_batch @ W1_single)
            outputs = jax.nn.sigmoid(hidden @ W2_single)
            errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        chunk_fitnesses = jax.vmap(eval_single_network)(W1, W2)
        chunk_fitnesses = jnp.where(jnp.isnan(chunk_fitnesses), -jnp.inf, chunk_fitnesses)

        # Update accumulator
        return jax.lax.dynamic_update_slice(
            fitnesses_acc, chunk_fitnesses, (start_idx,)
        )

    # Process all chunks using fori_loop
    all_fitnesses = jax.lax.fori_loop(0, num_chunks, process_chunk, all_fitnesses)

    # Remove padding
    if pad_size > 0:
        all_fitnesses = all_fitnesses[:per_gpu_pop]

    return all_fitnesses


# ============================================================================
# MODULE-LEVEL PMAP FUNCTIONS (Fix for XLA recompilation bug)
# ============================================================================
# These pmap functions are defined at module level to ensure JAX caches their
# compiled versions properly. Defining pmap inside functions creates
# new function objects each call, triggering full XLA recompilation when input
# shapes change.
#
# By defining at module level:
# 1. Same Python function object across calls
# 2. JAX can properly cache compiled versions by shape
# 3. Shape changes only trigger incremental recompilation, not from scratch
# ============================================================================

# --- Population-Parallel Strategy: pmap functions ---
# These split POPULATION across GPUs, replicate dataset on each GPU

@partial(
    jax.pmap,
    # in_axes must have entry for ALL args (17 total), static ones are ignored but need position
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(9, 10, 11, 13, 14, 16),  # static Python tuples/ints/funcs (NOT parent_indices - it's JAX arrays)
)
def _pmap_pop_parallel_non_chunked(
    cppns_slice,           # 0: sharded (axis 0) - each GPU gets pop slice
    all_positions,         # 1: replicated
    input_coords,          # 2: replicated
    output_coords,         # 3: replicated
    inputs_batch,          # 4: replicated
    targets_batch,         # 5: replicated
    variance_threshold,    # 6: replicated
    max_weight,            # 7: replicated
    weight_thresh,         # 8: replicated
    level_sizes,           # 9: STATIC
    level_offsets,         # 10: STATIC
    level_grid_sizes,      # 11: STATIC
    parent_indices,        # 12: replicated (tuple of JAX arrays - NOT hashable)
    num_levels,            # 13: STATIC
    total_positions,       # 14: STATIC
    state,                 # 15: replicated (PyTree)
    cppn_forward,          # 16: STATIC
):
    """pmap wrapper for population-parallel non-chunked execution."""
    return _full_pipeline_single_gpu(
        cppns_slice,
        all_positions,
        input_coords,
        output_coords,
        inputs_batch,
        targets_batch,
        level_sizes,
        level_offsets,
        level_grid_sizes,
        parent_indices,
        num_levels,
        total_positions,
        variance_threshold,
        max_weight,
        weight_thresh,
        cppn_forward,
        state,
    )


@partial(
    jax.pmap,
    # in_axes must have entry for ALL args (18 total), static ones are ignored but need position
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(9, 10, 11, 12, 14, 15, 17),  # static Python tuples/ints/funcs (pop_chunk_size is int, NOT parent_indices - it's JAX arrays)
)
def _pmap_pop_parallel_chunked(
    cppns_slice,           # 0: sharded (axis 0) - each GPU gets pop slice
    all_positions,         # 1: replicated
    input_coords,          # 2: replicated
    output_coords,         # 3: replicated
    inputs_batch,          # 4: replicated
    targets_batch,         # 5: replicated
    variance_threshold,    # 6: replicated
    max_weight,            # 7: replicated
    weight_thresh,         # 8: replicated
    pop_chunk_size,        # 9: STATIC (int for chunking - must be static for if/else)
    level_sizes,           # 10: STATIC
    level_offsets,         # 11: STATIC
    level_grid_sizes,      # 12: STATIC
    parent_indices,        # 13: replicated (tuple of JAX arrays - NOT hashable)
    num_levels,            # 14: STATIC
    total_positions,       # 15: STATIC
    state,                 # 16: replicated (PyTree)
    cppn_forward,          # 17: STATIC
):
    """pmap wrapper for population-parallel chunked execution."""
    return _full_pipeline_single_gpu_chunked(
        cppns_slice,
        all_positions,
        input_coords,
        output_coords,
        inputs_batch,
        targets_batch,
        level_sizes,
        level_offsets,
        level_grid_sizes,
        parent_indices,
        num_levels,
        total_positions,
        variance_threshold,
        max_weight,
        weight_thresh,
        cppn_forward,
        state,
        pop_chunk_size=pop_chunk_size,
    )


# --- Data-Parallel Strategy: pmap functions ---
# These split DATASET across GPUs, replicate full population on each GPU

@partial(
    jax.pmap,
    # in_axes must have entry for ALL args (17 total), static ones are ignored but need position
    in_axes=(None, None, None, None, 0, 0, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(9, 10, 11, 13, 14, 16),  # static Python tuples/ints/funcs (NOT parent_indices - it's JAX arrays)
)
def _pmap_data_parallel_non_chunked(
    cppns,                 # 0: replicated (full pop on each GPU)
    all_positions,         # 1: replicated
    input_coords,          # 2: replicated
    output_coords,         # 3: replicated
    inputs_shard,          # 4: sharded (axis 0) - each GPU gets data slice
    targets_shard,         # 5: sharded (axis 0)
    variance_threshold,    # 6: replicated
    max_weight,            # 7: replicated
    weight_thresh,         # 8: replicated
    level_sizes,           # 9: STATIC
    level_offsets,         # 10: STATIC
    level_grid_sizes,      # 11: STATIC
    parent_indices,        # 12: replicated (tuple of JAX arrays - NOT hashable)
    num_levels,            # 13: STATIC
    total_positions,       # 14: STATIC
    state,                 # 15: replicated (PyTree)
    cppn_forward,          # 16: STATIC
):
    """pmap wrapper for data-parallel non-chunked execution."""
    return _full_pipeline_single_gpu(
        cppns,
        all_positions,
        input_coords,
        output_coords,
        inputs_shard,
        targets_shard,
        level_sizes,
        level_offsets,
        level_grid_sizes,
        parent_indices,
        num_levels,
        total_positions,
        variance_threshold,
        max_weight,
        weight_thresh,
        cppn_forward,
        state,
    )


@partial(
    jax.pmap,
    # in_axes must have entry for ALL args (18 total), static ones are ignored but need position
    in_axes=(None, None, None, None, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(9, 10, 11, 12, 14, 15, 17),  # static Python tuples/ints/funcs (pop_chunk_size is int, NOT parent_indices - it's JAX arrays)
)
def _pmap_data_parallel_chunked(
    cppns,                 # 0: replicated (full pop on each GPU)
    all_positions,         # 1: replicated
    input_coords,          # 2: replicated
    output_coords,         # 3: replicated
    inputs_shard,          # 4: sharded (axis 0) - each GPU gets data slice
    targets_shard,         # 5: sharded (axis 0)
    variance_threshold,    # 6: replicated
    max_weight,            # 7: replicated
    weight_thresh,         # 8: replicated
    pop_chunk_size,        # 9: STATIC (int for chunking - must be static for if/else)
    level_sizes,           # 10: STATIC
    level_offsets,         # 11: STATIC
    level_grid_sizes,      # 12: STATIC
    parent_indices,        # 13: replicated (tuple of JAX arrays - NOT hashable)
    num_levels,            # 14: STATIC
    total_positions,       # 15: STATIC
    state,                 # 16: replicated (PyTree)
    cppn_forward,          # 17: STATIC
):
    """pmap wrapper for data-parallel chunked execution."""
    return _full_pipeline_single_gpu_chunked(
        cppns,
        all_positions,
        input_coords,
        output_coords,
        inputs_shard,
        targets_shard,
        level_sizes,
        level_offsets,
        level_grid_sizes,
        parent_indices,
        num_levels,
        total_positions,
        variance_threshold,
        max_weight,
        weight_thresh,
        cppn_forward,
        state,
        pop_chunk_size=pop_chunk_size,
    )


# --- Population-Parallel Strategy with H→H Caching: pmap function ---
# This pmap wraps the ENTIRE generation pipeline (CPPN queries + masks + W1/W2 + h→h eval)
# into a SINGLE pmap call, achieving true multi-GPU parallelism.
#
# Key insight: Previous multi-GPU approaches (data-parallel, population-parallel) were
# 3x SLOWER than single-GPU because they made 6+ separate Python→JAX calls per generation.
# By wrapping EVERYTHING in one pmap, we reduce to 1 Python→JAX call per generation.

@partial(
    jax.pmap,
    in_axes=(0, None, None, None, None, None, 0, 0, 0, 0, None, None, None, None),  # 14 traceable args
    static_broadcasted_argnums=(14, 15, 16, 17, 18, 19, 20, 21, 22),  # 9 static args (must be hashable)
)
def _pmap_pop_parallel_with_hh(
    cppns_slice,           # 0: sharded (axis 0) - each GPU gets pop slice
    all_positions,         # 1: replicated
    input_coords,          # 2: replicated
    output_coords,         # 3: replicated
    inputs_batch,          # 4: replicated
    targets_batch,         # 5: replicated
    sparse_from_idx,       # 6: sharded (axis 0) - matches population sharding
    sparse_to_idx,         # 7: sharded (axis 0)
    sparse_weights,        # 8: sharded (axis 0)
    sparse_valid_mask,     # 9: sharded (axis 0)
    variance_threshold,    # 10: replicated
    max_weight,            # 11: replicated
    weight_thresh,         # 12: replicated
    state,                 # 13: replicated (PyTree)
    level_sizes,           # 14: STATIC - tuple of ints
    level_offsets,         # 15: STATIC - tuple of ints
    level_grid_sizes,      # 16: STATIC - tuple of tuple of ints
    parent_indices,        # 17: STATIC - tuple of tuple of ints
    num_levels,            # 18: STATIC - int
    total_positions,       # 19: STATIC - int
    activate_time,         # 20: STATIC - int (MUST be static for jax.lax.scan length!)
    cppn_forward,          # 21: STATIC - function
    num_positions,         # 22: STATIC - int
):
    """pmap wrapper for population-parallel with h→h caching.

    This is the UNIFIED pmap that achieves true multi-GPU speedup:
    - Population AND sparse h→h are sharded together
    - ENTIRE generation pipeline runs inside ONE pmap call
    - CPPN queries are parallelized across GPUs (the bottleneck!)
    - Each GPU processes its slice through complete pipeline

    Expected performance: ~400-600ms/gen (vs 2400ms with separate pmap calls)
    This makes multi-GPU 1.5-2x FASTER than single-GPU (vs 3x slower before).
    """
    return _full_pipeline_single_gpu_with_hh(
        cppns_slice,
        all_positions,
        input_coords,
        output_coords,
        inputs_batch,
        targets_batch,
        sparse_from_idx,
        sparse_to_idx,
        sparse_weights,
        sparse_valid_mask,
        level_sizes,
        level_offsets,
        level_grid_sizes,
        parent_indices,
        num_levels,
        total_positions,
        variance_threshold,
        max_weight,
        weight_thresh,
        activate_time,
        cppn_forward,
        state,
        num_positions_param=num_positions,
    )


def full_pipeline_population_parallel(
    cppns_transformed: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[Tuple[int, int], ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    variance_threshold: float,
    max_weight: float,
    weight_thresh: float,
    cppn_forward: Any,
    state: Any,
    num_gpus: int = 2,
    pop_chunk_size: Optional[int] = None,
) -> jnp.ndarray:
    """Run FULL pipeline with CONSISTENT population sharding across GPUs.

    This is the OPTIMIZED multi-GPU approach:
    - Population is split ONCE at the start
    - Each GPU runs the ENTIRE pipeline for its individuals
    - NO data reshuffling between stages
    - Results gathered only at the end

    This avoids the overhead of the previous approach which:
    - Used position sharding for CPPN queries
    - Used population sharding for evaluation
    - Required data reshuffling between these stages

    Args:
        cppns_transformed: Full population CPPNs (pop_size, ...)
        ... (same as _full_pipeline_single_gpu)
        num_gpus: Number of GPUs to use
        pop_chunk_size: If provided, use chunked processing to avoid OOM
                        with large datasets. If None, process all at once.

    Returns:
        Fitness values for entire population (pop_size,)
    """
    pop_size = cppns_transformed[0].shape[0]

    # Pad population to be divisible by num_gpus
    remainder = pop_size % num_gpus
    if remainder != 0:
        pad_size = num_gpus - remainder
        cppns_transformed = tuple(
            jnp.pad(arr, ((0, pad_size),) + ((0, 0),) * (arr.ndim - 1), mode='constant')
            for arr in cppns_transformed
        )
        padded_pop_size = pop_size + pad_size
    else:
        pad_size = 0
        padded_pop_size = pop_size

    per_gpu_pop = padded_pop_size // num_gpus

    # Reshape CPPNs for pmap: (pop_size, ...) -> (num_gpus, per_gpu_pop, ...)
    cppns_sharded = tuple(
        arr.reshape(num_gpus, per_gpu_pop, *arr.shape[1:])
        for arr in cppns_transformed
    )

    # Use module-level pmap functions so JAX caches their compiled versions
    # defining pmap inside this function creates new function objects
    # each call, causing full XLA recompilation when input shapes change.
    # By using module-level pmaps, we ensure JAX properly caches compiled versions.
    if pop_chunk_size is not None and pop_chunk_size > 0:
        # Use chunked version to avoid OOM with large datasets
        fitnesses_sharded = _pmap_pop_parallel_chunked(
            cppns_sharded,
            all_positions,
            input_coords,
            output_coords,
            inputs_batch,
            targets_batch,
            variance_threshold,
            max_weight,
            weight_thresh,
            pop_chunk_size,
            level_sizes,
            level_offsets,
            level_grid_sizes,
            parent_indices,
            num_levels,
            total_positions,
            state,
            cppn_forward,
        )
    else:
        # Use non-chunked version for small datasets (faster)
        fitnesses_sharded = _pmap_pop_parallel_non_chunked(
            cppns_sharded,
            all_positions,
            input_coords,
            output_coords,
            inputs_batch,
            targets_batch,
            variance_threshold,
            max_weight,
            weight_thresh,
            level_sizes,
            level_offsets,
            level_grid_sizes,
            parent_indices,
            num_levels,
            total_positions,
            state,
            cppn_forward,
        )

    # Gather results - shape (padded_pop_size,)
    fitnesses = fitnesses_sharded.reshape(-1)

    # Remove padding
    if pad_size > 0:
        fitnesses = fitnesses[:pop_size]

    return fitnesses


def full_pipeline_population_parallel_shardmap(
    cppns_transformed: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[Tuple[int, int], ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    variance_threshold: float,
    max_weight: float,
    weight_thresh: float,
    cppn_forward: Any,
    state: Any,
    mesh: Any,
    num_gpus: int = 2,
    pop_chunk_size: Optional[int] = None,
) -> jnp.ndarray:
    """Run FULL pipeline with shard_map for JIT-compatible multi-GPU execution.

    Unlike pmap, shard_map preserves sharding inside JIT/while_loop.
    This enables TRUE GPU-resident multi-GPU evolution.

    Args:
        cppns_transformed: Full population CPPNs (pop_size, ...)
        ... (same as full_pipeline_population_parallel)
        mesh: JAX Mesh for sharding (must have 'gpus' axis)
        num_gpus: Number of GPUs to use
        pop_chunk_size: If provided, use chunked processing to avoid OOM

    Returns:
        Fitness values for entire population (pop_size,)
    """
    if not SHARD_MAP_AVAILABLE:
        raise RuntimeError("shard_map not available in this JAX version")

    pop_size = cppns_transformed[0].shape[0]

    # Pad population to be divisible by num_gpus
    remainder = pop_size % num_gpus
    if remainder != 0:
        pad_size = num_gpus - remainder
        cppns_transformed = tuple(
            jnp.pad(arr, ((0, pad_size),) + ((0, 0),) * (arr.ndim - 1), mode='constant')
            for arr in cppns_transformed
        )
        padded_pop_size = pop_size + pad_size
    else:
        pad_size = 0
        padded_pop_size = pop_size

    per_gpu_pop = padded_pop_size // num_gpus

    # Reshape CPPNs for sharding: (pop_size, ...) -> (num_gpus, per_gpu_pop, ...)
    cppns_sharded = tuple(
        arr.reshape(num_gpus, per_gpu_pop, *arr.shape[1:])
        for arr in cppns_transformed
    )

    # Choose between chunked and non-chunked based on pop_chunk_size
    if pop_chunk_size is not None and pop_chunk_size > 0:
        # Use chunked version to avoid OOM with large datasets
        @partial(shard_map, mesh=mesh,
                 in_specs=(P('gpus'), P('gpus'), P('gpus'), P('gpus')),
                 out_specs=P('gpus'),
                 check_rep=False)
        def run_pipeline_on_gpu_shardmap(nodes_slice, conns_slice, acts_slice, extra_slice):
            cppns_slice = (nodes_slice, conns_slice, acts_slice, extra_slice)
            return _full_pipeline_single_gpu_chunked(
                cppns_slice,
                all_positions,
                input_coords,
                output_coords,
                inputs_batch,
                targets_batch,
                level_sizes,
                level_offsets,
                level_grid_sizes,
                parent_indices,
                num_levels,
                total_positions,
                variance_threshold,
                max_weight,
                weight_thresh,
                cppn_forward,
                state,
                pop_chunk_size=pop_chunk_size,
            )
    else:
        # Use non-chunked version for small datasets (faster)
        @partial(shard_map, mesh=mesh,
                 in_specs=(P('gpus'), P('gpus'), P('gpus'), P('gpus')),
                 out_specs=P('gpus'),
                 check_rep=False)
        def run_pipeline_on_gpu_shardmap(nodes_slice, conns_slice, acts_slice, extra_slice):
            cppns_slice = (nodes_slice, conns_slice, acts_slice, extra_slice)
            return _full_pipeline_single_gpu(
                cppns_slice,
                all_positions,
                input_coords,
                output_coords,
                inputs_batch,
                targets_batch,
                level_sizes,
                level_offsets,
                level_grid_sizes,
                parent_indices,
                num_levels,
                total_positions,
                variance_threshold,
                max_weight,
                weight_thresh,
                cppn_forward,
                state,
            )

    # Execute on all GPUs with shard_map
    with mesh:
        fitnesses_sharded = run_pipeline_on_gpu_shardmap(
            cppns_sharded[0], cppns_sharded[1], cppns_sharded[2], cppns_sharded[3]
        )

    # Gather results - shape (padded_pop_size,)
    fitnesses = fitnesses_sharded.reshape(-1)

    # Remove padding
    if pad_size > 0:
        fitnesses = fitnesses[:pop_size]

    return fitnesses


# ============================================================================
# Strategy: Data-Parallel (Dataset Sharding)
# ============================================================================

def full_pipeline_data_parallel(
    cppns_transformed: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[Tuple[int, int], ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    variance_threshold: float,
    max_weight: float,
    weight_thresh: float,
    cppn_forward: Any,
    state: Any,
    num_gpus: int = 2,
    pop_chunk_size: Optional[int] = None,
) -> jnp.ndarray:
    """Run pipeline with DATA-PARALLEL sharding (dataset split across GPUs).

    This strategy is optimal for LARGE DATASETS that don't fit on a single GPU:
    - Dataset is SPLIT across GPUs (each GPU gets different samples)
    - Population is REPLICATED on each GPU (full population on each)
    - Each GPU computes partial fitness on its data shard
    - Results are AVERAGED across GPUs

    Memory comparison (2 GPUs, pop=300, data=10K samples):
    - Population-Parallel: Each GPU needs FULL dataset (10K) + half pop (150)
    - Data-Parallel: Each GPU needs HALF dataset (5K) + full pop (300)
    - Winner depends on which is larger: population or dataset

    Args:
        cppns_transformed: Full population CPPNs (pop_size, ...)
        inputs_batch: Problem inputs (n_samples, num_inputs) - WILL BE SHARDED
        targets_batch: Problem targets (n_samples, num_outputs) - WILL BE SHARDED
        ... (same as full_pipeline_population_parallel)
        num_gpus: Number of GPUs to use
        pop_chunk_size: If provided, use chunked processing within each GPU

    Returns:
        Fitness values for entire population (pop_size,)
    """
    pop_size = cppns_transformed[0].shape[0]
    n_samples = inputs_batch.shape[0]

    # Pad n_samples to be divisible by num_gpus
    remainder = n_samples % num_gpus
    if remainder != 0:
        pad_samples = num_gpus - remainder
        inputs_batch = jnp.pad(inputs_batch, ((0, pad_samples), (0, 0)), mode='edge')
        targets_batch = jnp.pad(targets_batch, ((0, pad_samples), (0, 0)), mode='edge')
        padded_n_samples = n_samples + pad_samples
    else:
        pad_samples = 0
        padded_n_samples = n_samples

    per_gpu_samples = padded_n_samples // num_gpus

    # Shard dataset: (n_samples, features) -> (num_gpus, per_gpu_samples, features)
    inputs_sharded = inputs_batch.reshape(num_gpus, per_gpu_samples, -1)
    targets_sharded = targets_batch.reshape(num_gpus, per_gpu_samples, -1)

    # Use module-level pmap functions so JAX caches their compiled versions
    # defining pmap inside this function creates new function objects
    # each call, causing full XLA recompilation when input shapes change.
    # By using module-level pmaps, we ensure JAX properly caches compiled versions.
    if pop_chunk_size is not None and pop_chunk_size > 0:
        # Use chunked version to avoid OOM with large populations
        partial_fitnesses = _pmap_data_parallel_chunked(
            cppns_transformed,
            all_positions,
            input_coords,
            output_coords,
            inputs_sharded,
            targets_sharded,
            variance_threshold,
            max_weight,
            weight_thresh,
            pop_chunk_size,
            level_sizes,
            level_offsets,
            level_grid_sizes,
            parent_indices,
            num_levels,
            total_positions,
            state,
            cppn_forward,
        )
    else:
        # Use non-chunked version for smaller populations
        partial_fitnesses = _pmap_data_parallel_non_chunked(
            cppns_transformed,
            all_positions,
            input_coords,
            output_coords,
            inputs_sharded,
            targets_sharded,
            variance_threshold,
            max_weight,
            weight_thresh,
            level_sizes,
            level_offsets,
            level_grid_sizes,
            parent_indices,
            num_levels,
            total_positions,
            state,
            cppn_forward,
        )

    # Aggregate: average fitness across data shards
    # Each GPU computed fitness on different samples, so we average
    # Shape: (num_gpus, pop_size) -> (pop_size,)
    #
    # Note: If we padded samples, the padded samples are duplicates (mode='edge')
    # so averaging still gives correct result (no need for weighted average)
    fitnesses = jnp.mean(partial_fitnesses, axis=0)

    return fitnesses


# ============================================================================
# Strategy: Streaming (CPU-to-GPU Data Streaming)
# ============================================================================

def full_pipeline_streaming(
    cppns_transformed: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_cpu: Any,  # numpy array on CPU
    targets_cpu: Any,  # numpy array on CPU
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[Tuple[int, int], ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    variance_threshold: float,
    max_weight: float,
    weight_thresh: float,
    cppn_forward: Any,
    state: Any,
    data_chunk_size: int = 1000,
    num_gpus: int = 2,
    pop_chunk_size: Optional[int] = None,
) -> jnp.ndarray:
    """Run pipeline with STREAMING (load dataset chunks from CPU as needed).

    This strategy handles VERY LARGE DATASETS that exceed total GPU memory:
    - Dataset stays on CPU
    - Chunks are streamed to GPU one at a time
    - Each chunk is evaluated using multi-GPU (population-parallel)
    - Fitness is accumulated across chunks

    Trade-off:
    - Pro: Can handle datasets of ANY size (limited only by CPU RAM)
    - Con: CPU-GPU transfer overhead per chunk (~2ms per 4MB at ~2GB/s)

    Args:
        cppns_transformed: Full population CPPNs (pop_size, ...)
        inputs_cpu: Problem inputs as NUMPY array on CPU (n_samples, num_inputs)
        targets_cpu: Problem targets as NUMPY array on CPU (n_samples, num_outputs)
        ... (same as full_pipeline_population_parallel)
        data_chunk_size: Number of samples per chunk to stream
        num_gpus: Number of GPUs for population-parallel within each chunk
        pop_chunk_size: If provided, use chunked processing for population

    Returns:
        Fitness values for entire population (pop_size,)
    """
    import numpy as np

    pop_size = cppns_transformed[0].shape[0]
    n_samples = inputs_cpu.shape[0]
    num_chunks = (n_samples + data_chunk_size - 1) // data_chunk_size

    # Accumulate MSE (not fitness) for proper averaging
    # MSE = mean((pred - target)^2), fitness = 1 - MSE
    # To average MSE across chunks: weighted average by chunk size
    total_mse = jnp.zeros(pop_size, dtype=jnp.float32)
    total_samples = 0

    for chunk_idx in range(num_chunks):
        start = chunk_idx * data_chunk_size
        end = min(start + data_chunk_size, n_samples)
        actual_chunk_size = end - start

        # Stream chunk from CPU to GPU
        inputs_chunk = jax.device_put(jnp.array(inputs_cpu[start:end], dtype=jnp.float32))
        targets_chunk = jax.device_put(jnp.array(targets_cpu[start:end], dtype=jnp.float32))

        # Evaluate on this chunk using population-parallel multi-GPU
        # This returns fitness = 1 - MSE
        chunk_fitness = full_pipeline_population_parallel(
            cppns_transformed,
            all_positions,
            input_coords,
            output_coords,
            inputs_chunk,
            targets_chunk,
            level_sizes,
            level_offsets,
            level_grid_sizes,
            parent_indices,
            num_levels,
            total_positions,
            variance_threshold,
            max_weight,
            weight_thresh,
            cppn_forward,
            state,
            num_gpus=num_gpus,
            pop_chunk_size=pop_chunk_size,
        )

        # Convert fitness back to MSE for proper weighted averaging
        # fitness = 1 - MSE, so MSE = 1 - fitness
        chunk_mse = 1.0 - chunk_fitness

        # Accumulate weighted MSE
        total_mse = total_mse + chunk_mse * actual_chunk_size
        total_samples += actual_chunk_size

        # Hint to JAX GC (not strictly necessary but helps)
        del inputs_chunk, targets_chunk

    # Final fitness from averaged MSE
    avg_mse = total_mse / total_samples
    fitnesses = 1.0 - avg_mse

    return fitnesses


# ============================================================================
# Strategy 2: Island Model
# ============================================================================

class IslandModelRunner:
    """Asynchronous Island Model coordinator for multi-GPU evolution.

    Runs independent populations on each GPU with optional periodic migration
    of best genomes. This guarantees near-linear speedup because islands
    have NO synchronization during evolution.

    Usage:
        runner = IslandModelRunner(
            algorithm_class=HMRHyperNEATAdaptiveChunking,
            config_path="config.yaml",
            problem=xor_problem,
            config=IslandModelConfig(num_islands=2, migration_interval=20)
        )
        result = runner.run(target_fitness=0.98, max_generations=100)
    """

    def __init__(
        self,
        algorithm_class: type,
        config_path: Optional[str],
        problem: Any,
        config: IslandModelConfig,
        base_seed: int = 42,
        max_depth: int = 4,
        population_size: int = 1000,
    ):
        """Initialize Island Model runner.

        Args:
            algorithm_class: The algorithm class to instantiate per island
            config_path: Path to configuration file (can be None if max_depth/population_size provided)
            problem: The problem instance to solve
            config: Island model configuration
            base_seed: Base random seed (each island gets base_seed + island_id)
            max_depth: Max depth for HMR-HyperNEAT (used when config_path is None)
            population_size: Population size per island (used when config_path is None)
        """
        self.algorithm_class = algorithm_class
        self.config_path = config_path
        self.problem = problem
        self.config = config
        self.base_seed = base_seed
        self.max_depth = max_depth
        self.population_size = population_size

        # Inter-process communication
        self.migration_queues: List[Queue] = []
        self.result_queue = Queue()
        self.stop_event = Event()

        # Create migration queues (ring topology)
        for _ in range(config.num_islands):
            self.migration_queues.append(Queue())

    def run(
        self,
        target_fitness: float,
        max_generations: int,
    ) -> Dict[str, Any]:
        """Run Island Model evolution until solution found or max generations.

        Args:
            target_fitness: Fitness threshold for success
            max_generations: Maximum generations per island

        Returns:
            Dict with:
                - winner_island: ID of island that found solution
                - best_fitness: Best fitness achieved
                - generations: Number of generations run
                - total_time: Wall clock time
                - island_results: Per-island results
        """
        start_time = time.time()

        # Create and start worker processes
        processes: List[Process] = []
        for island_id in range(self.config.num_islands):
            p = Process(
                target=_island_worker,
                args=(
                    island_id,
                    island_id,  # gpu_id = island_id for simplicity
                    self.algorithm_class,
                    self.config_path,
                    self.max_depth,
                    self.population_size,
                    self.problem,
                    self.base_seed + island_id,
                    target_fitness,
                    max_generations,
                    self.config.migration_interval,
                    self.config.migration_rate,
                    self.migration_queues,
                    self.result_queue,
                    self.stop_event,
                )
            )
            p.start()
            processes.append(p)

        # Wait for first solution or all processes to complete
        island_results = []
        winner_island = None
        best_fitness = -float('inf')
        total_generations = 0

        while len(island_results) < self.config.num_islands:
            try:
                result = self.result_queue.get(timeout=1.0)
                island_results.append(result)

                if result['fitness'] > best_fitness:
                    best_fitness = result['fitness']
                    winner_island = result['island_id']
                    total_generations = result['generations']

                if result['solved'] and self.config.stop_on_first_solution:
                    self.stop_event.set()
                    break

            except Exception:
                # Check if any process is still alive
                alive = any(p.is_alive() for p in processes)
                if not alive:
                    break

        # Stop all remaining processes
        self.stop_event.set()
        for p in processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        total_time = time.time() - start_time

        return {
            'winner_island': winner_island,
            'best_fitness': best_fitness,
            'generations': total_generations,
            'total_time': total_time,
            'island_results': island_results,
            'solved': best_fitness >= target_fitness,
        }


def _island_worker(
    island_id: int,
    gpu_id: int,
    algorithm_class: type,
    config_path: Optional[str],
    max_depth: int,
    population_size: int,
    problem: Any,
    seed: int,
    target_fitness: float,
    max_generations: int,
    migration_interval: int,
    migration_rate: float,
    migration_queues: List[Queue],
    result_queue: Queue,
    stop_event: Event,
):
    """Worker process for single island.

    CRITICAL: Sets CUDA_VISIBLE_DEVICES BEFORE importing JAX to ensure
    this process only sees its assigned GPU.

    Args:
        island_id: Unique island identifier
        gpu_id: GPU device ID to use
        algorithm_class: Algorithm class to instantiate
        config_path: Configuration file path (can be None)
        max_depth: Max depth for HMR-HyperNEAT (used when config_path is None)
        population_size: Population size (used when config_path is None)
        problem: Problem to solve
        seed: Random seed
        target_fitness: Target fitness threshold
        max_generations: Max generations to run
        migration_interval: Generations between migrations
        migration_rate: Fraction of population to migrate
        migration_queues: List of queues for migration
        result_queue: Queue to report final results
        stop_event: Event to signal early stopping
    """
    # CRITICAL: Set GPU before importing JAX
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Now import JAX - it will only see the assigned GPU
    import jax
    import jax.numpy as jnp

    try:
        # Initialize algorithm
        if config_path is not None:
            algo = algorithm_class(config_path=config_path)
            state = algo.initialize(problem, seed=seed)
        else:
            # Create config when config_path is None
            algo_config = {
                "algorithm_params": {
                    "hmrhyperneat": {
                        "population_size": population_size,
                        "hmr_hyperneat": {
                            "initial_depth": 0,
                            "max_depth": max_depth,
                            "variance_threshold": 0.03,
                            "division_threshold": 0.5,
                            "band_threshold": 0.3,
                            "max_weight": 8.0,
                        },
                        "substrate": {
                            "input_coords": [(-1.0, -1.0), (1.0, -1.0), (0.0, -1.0)],
                            "output_coords": [(0.0, 1.0)],
                            "output_activation": "sigmoid",
                            "hidden_activation": "tanh",
                        },
                    }
                }
            }
            algo = algorithm_class()
            config = algo.create_config(algo_config)
            state = algo.initialize(config, problem, seed=seed)

        best_fitness = -float('inf')
        generation = 0

        while generation < max_generations and not stop_event.is_set():
            # Run one generation
            state, metrics = algo.run_generation(state, problem)
            generation += 1

            if metrics.best_fitness > best_fitness:
                best_fitness = metrics.best_fitness

            # Check for solution
            if best_fitness >= target_fitness:
                result_queue.put({
                    'island_id': island_id,
                    'fitness': best_fitness,
                    'generations': generation,
                    'solved': True,
                })
                stop_event.set()
                return

            # Periodic migration
            if generation % migration_interval == 0:
                _handle_migration(
                    island_id, state, algo, migration_rate,
                    migration_queues, generation
                )

        # Report final result
        result_queue.put({
            'island_id': island_id,
            'fitness': best_fitness,
            'generations': generation,
            'solved': best_fitness >= target_fitness,
        })

    except Exception as e:
        result_queue.put({
            'island_id': island_id,
            'fitness': -float('inf'),
            'generations': 0,
            'solved': False,
            'error': str(e),
        })


def _handle_migration(
    island_id: int,
    state: Any,
    algo: Any,
    migration_rate: float,
    migration_queues: List[Queue],
    generation: int,
):
    """Handle genome migration between islands.

    Uses ring topology: island i sends to island (i+1) % num_islands

    Args:
        island_id: Current island ID
        state: Algorithm state
        algo: Algorithm instance
        migration_rate: Fraction of population to migrate
        migration_queues: List of migration queues
        generation: Current generation
    """
    num_islands = len(migration_queues)

    # Extract best genomes to send
    try:
        pop = algo.neat_algo.ask(state)
        if pop is None or len(pop) == 0:
            return

        # Get fitness values
        fitnesses = state.fitnesses if hasattr(state, 'fitnesses') else None
        if fitnesses is None:
            return

        # Sort by fitness and get top migrants
        num_migrants = max(1, int(len(pop) * migration_rate))
        sorted_indices = np.argsort(np.asarray(fitnesses))[::-1]
        best_idx = sorted_indices[0]

        # Create migration packet (convert to numpy for pickling)
        packet = MigrationPacket(
            source_island=island_id,
            generation=generation,
            best_fitness=float(fitnesses[best_idx]),
            genome_nodes=np.asarray(pop[0][best_idx]) if len(pop) > 0 else np.array([]),
            genome_conns=np.asarray(pop[1][best_idx]) if len(pop) > 1 else np.array([]),
        )

        # Send to next island (ring topology)
        next_island = (island_id + 1) % num_islands
        migration_queues[next_island].put_nowait(packet)

    except Exception:
        pass  # Migration failure is non-fatal

    # Receive migrants from previous island
    try:
        prev_island = (island_id - 1) % num_islands
        while not migration_queues[island_id].empty():
            packet = migration_queues[island_id].get_nowait()
            # Injecting a migrant genome requires modifying NEAT state, which is complex
            # log receipt
            pass
    except Exception:
        pass


# ============================================================================
# Strategy 2b: Island Model V2 (Optimized with Working Migration)
# ============================================================================

class IslandModelRunnerV2:
    """Optimized Island Model coordinator with working migration.

    Key improvements over V1:
    1. Migration actually injects genomes into populations
    2. Population can be split across islands for better scaling
    3. Configurable migration topology (ring, star, random)

    Usage:
        runner = IslandModelRunnerV2(
            algorithm_class=HMRHyperNEATAdaptiveChunking,
            config_path="config.yaml",
            problem=xor_problem,
            config=IslandModelConfigV2(num_islands=2, topology="ring")
        )
        result = runner.run(target_fitness=0.98, max_generations=100)
    """

    def __init__(
        self,
        algorithm_class: type,
        problem: Any,
        config: IslandModelConfigV2,
        base_seed: int = 42,
        total_population: int = 1000,
        max_depth: int = 4,
    ):
        """Initialize Island Model V2 runner.

        Args:
            algorithm_class: The algorithm class to instantiate per island
            problem: The problem instance to solve
            config: Island model V2 configuration
            base_seed: Base random seed (each island gets base_seed + island_id)
            total_population: Total population size (split across islands if configured)
            max_depth: Max depth for HMR-HyperNEAT grid
        """
        self.algorithm_class = algorithm_class
        self.problem = problem
        self.config = config
        self.base_seed = base_seed
        self.total_population = total_population
        self.max_depth = max_depth

        # Calculate per-island population
        if config.split_population:
            self.pop_per_island = total_population // config.num_islands
        else:
            self.pop_per_island = total_population

        # Inter-process communication
        self.migration_queues: List[Queue] = []
        self.result_queue = Queue()
        self.stop_event = Event()

        # Create migration queues based on topology
        for _ in range(config.num_islands):
            self.migration_queues.append(Queue())

    def run(
        self,
        target_fitness: float,
        max_generations: int,
    ) -> Dict[str, Any]:
        """Run Island Model V2 evolution until solution found or max generations.

        Args:
            target_fitness: Fitness threshold for success
            max_generations: Maximum generations per island

        Returns:
            Dict with:
                - winner_island: ID of island that found solution
                - best_fitness: Best fitness achieved
                - generations: Number of generations run
                - total_time: Wall clock time
                - island_results: Per-island results
                - migrations_sent: Total migrations sent
                - migrations_received: Total migrations received
        """
        start_time = time.time()

        # Create and start worker processes
        processes: List[Process] = []
        for island_id in range(self.config.num_islands):
            p = Process(
                target=_island_worker_v2,
                args=(
                    island_id,
                    island_id,  # gpu_id = island_id
                    self.algorithm_class,
                    self.max_depth,
                    self.pop_per_island,
                    self.problem,
                    self.base_seed + island_id,
                    target_fitness,
                    max_generations,
                    self.config.migration_interval,
                    self.config.migration_rate,
                    self.config.topology,
                    self.migration_queues,
                    self.result_queue,
                    self.stop_event,
                )
            )
            p.start()
            processes.append(p)

        # Wait for results
        island_results = []
        winner_island = None
        best_fitness = -float('inf')
        total_generations = 0

        while len(island_results) < self.config.num_islands:
            try:
                result = self.result_queue.get(timeout=1.0)
                island_results.append(result)

                if result['fitness'] > best_fitness:
                    best_fitness = result['fitness']
                    winner_island = result['island_id']
                    total_generations = result['generations']

                if result['solved'] and self.config.stop_on_first_solution:
                    self.stop_event.set()
                    break

            except Exception:
                alive = any(p.is_alive() for p in processes)
                if not alive:
                    break

        # Stop all remaining processes
        self.stop_event.set()
        for p in processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        total_time = time.time() - start_time

        # Aggregate migration stats
        migrations_sent = sum(r.get('migrations_sent', 0) for r in island_results)
        migrations_received = sum(r.get('migrations_received', 0) for r in island_results)

        return {
            'winner_island': winner_island,
            'best_fitness': best_fitness,
            'generations': total_generations,
            'total_time': total_time,
            'island_results': island_results,
            'solved': best_fitness >= target_fitness,
            'migrations_sent': migrations_sent,
            'migrations_received': migrations_received,
            'pop_per_island': self.pop_per_island,
            'topology': self.config.topology,
        }


def _island_worker_v2(
    island_id: int,
    gpu_id: int,
    algorithm_class: type,
    max_depth: int,
    population_size: int,
    problem: Any,
    seed: int,
    target_fitness: float,
    max_generations: int,
    migration_interval: int,
    migration_rate: float,
    topology: str,
    migration_queues: List[Queue],
    result_queue: Queue,
    stop_event: Event,
):
    """V2 Worker process for single island with working migration.

    CRITICAL: Sets CUDA_VISIBLE_DEVICES BEFORE importing JAX.

    Args:
        island_id: Unique island identifier
        gpu_id: GPU device ID to use
        algorithm_class: Algorithm class to instantiate
        max_depth: Max depth for HMR-HyperNEAT grid
        population_size: Population size for this island
        problem: Problem to solve
        seed: Random seed
        target_fitness: Target fitness threshold
        max_generations: Max generations to run
        migration_interval: Generations between migrations
        migration_rate: Fraction of population to migrate
        topology: Migration topology (ring, star, random)
        migration_queues: List of queues for migration
        result_queue: Queue to report final results
        stop_event: Event to signal early stopping
    """
    # CRITICAL: Set GPU before importing JAX
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Now import JAX
    import jax
    import jax.numpy as jnp

    migrations_sent = 0
    migrations_received = 0

    try:
        # Create algorithm config
        algo_config = {
            "algorithm_params": {
                "hmrhyperneat": {
                    "population_size": population_size,
                    "hmr_hyperneat": {
                        "initial_depth": 0,
                        "max_depth": max_depth,
                        "variance_threshold": 0.03,
                        "division_threshold": 0.5,
                        "band_threshold": 0.3,
                        "max_weight": 8.0,
                    },
                    "substrate": {
                        "input_coords": [(-1.0, -1.0), (1.0, -1.0), (0.0, -1.0)],
                        "output_coords": [(0.0, 1.0)],
                        "output_activation": "sigmoid",
                        "hidden_activation": "tanh",
                    },
                }
            }
        }

        # Initialize algorithm
        algo = algorithm_class()
        config = algo.create_config(algo_config)
        state = algo.initialize(config, problem, seed=seed)

        best_fitness = -float('inf')
        generation = 0

        while generation < max_generations and not stop_event.is_set():
            # Run one generation
            state, metrics = algo.run_generation(state, problem)
            generation += 1

            if metrics.best_fitness > best_fitness:
                best_fitness = metrics.best_fitness

            # Check for solution
            if best_fitness >= target_fitness:
                result_queue.put({
                    'island_id': island_id,
                    'fitness': best_fitness,
                    'generations': generation,
                    'solved': True,
                    'migrations_sent': migrations_sent,
                    'migrations_received': migrations_received,
                })
                stop_event.set()
                return

            # Periodic migration
            if generation % migration_interval == 0:
                sent, received, state = _handle_migration_v2(
                    island_id, state, algo, migration_rate,
                    topology, migration_queues, generation
                )
                migrations_sent += sent
                migrations_received += received

        # Report final result
        result_queue.put({
            'island_id': island_id,
            'fitness': best_fitness,
            'generations': generation,
            'solved': best_fitness >= target_fitness,
            'migrations_sent': migrations_sent,
            'migrations_received': migrations_received,
        })

    except Exception as e:
        result_queue.put({
            'island_id': island_id,
            'fitness': -float('inf'),
            'generations': 0,
            'solved': False,
            'error': str(e),
            'migrations_sent': 0,
            'migrations_received': 0,
        })


def _handle_migration_v2(
    island_id: int,
    state: Any,
    algo: Any,
    migration_rate: float,
    topology: str,
    migration_queues: List[Queue],
    generation: int,
) -> Tuple[int, int, Any]:
    """Handle genome migration with actual injection.

    Args:
        island_id: Current island ID
        state: Algorithm state
        algo: Algorithm instance
        migration_rate: Fraction of population to migrate
        topology: Migration topology (ring, star, random)
        migration_queues: List of migration queues
        generation: Current generation

    Returns:
        Tuple of (migrations_sent, migrations_received, updated_state)
    """
    import jax.numpy as jnp
    import random

    num_islands = len(migration_queues)
    migrations_sent = 0
    migrations_received = 0

    # Extract best genomes to send
    try:
        pop = algo.neat_algo.ask(state)
        if pop is None or len(pop) == 0:
            return 0, 0, state

        # Get fitness values
        fitnesses = state.fitnesses if hasattr(state, 'fitnesses') else None
        if fitnesses is None:
            return 0, 0, state

        # Sort by fitness and get top migrant
        sorted_indices = np.argsort(np.asarray(fitnesses))[::-1]
        best_idx = sorted_indices[0]

        # Create migration packet
        packet = MigrationPacket(
            source_island=island_id,
            generation=generation,
            best_fitness=float(fitnesses[best_idx]),
            genome_nodes=np.asarray(pop[0][best_idx]) if len(pop) > 0 else np.array([]),
            genome_conns=np.asarray(pop[1][best_idx]) if len(pop) > 1 else np.array([]),
        )

        # Determine target islands based on topology
        if topology == "ring":
            # Send to next island in ring
            targets = [(island_id + 1) % num_islands]
        elif topology == "star":
            # Send to all other islands
            targets = [i for i in range(num_islands) if i != island_id]
        elif topology == "random":
            # Send to one random island
            candidates = [i for i in range(num_islands) if i != island_id]
            targets = [random.choice(candidates)] if candidates else []
        else:
            targets = []

        # Send packets
        for target in targets:
            try:
                migration_queues[target].put_nowait(packet)
                migrations_sent += 1
            except Exception:
                pass

    except Exception:
        pass

    # Receive and INJECT migrants
    try:
        while not migration_queues[island_id].empty():
            try:
                packet = migration_queues[island_id].get_nowait()
                migrations_received += 1

                # INJECT the migrant genome into population
                state = _inject_migrant_genome(
                    state, algo, packet.genome_nodes, packet.genome_conns
                )

            except Exception:
                pass
    except Exception:
        pass

    return migrations_sent, migrations_received, state


def _inject_migrant_genome(
    state: Any,
    algo: Any,
    migrant_nodes: np.ndarray,
    migrant_conns: np.ndarray,
) -> Any:
    """Inject a migrant genome into the population, replacing the worst individual.

    This is the key improvement in V2 - migrants are actually used!

    Args:
        state: Current algorithm state
        algo: Algorithm instance
        migrant_nodes: Node genes from migrant
        migrant_conns: Connection genes from migrant

    Returns:
        Updated state with migrant injected
    """
    import jax.numpy as jnp

    try:
        # Get current population
        pop = algo.neat_algo.ask(state)
        if pop is None or len(pop) < 2:
            return state

        # Get fitness values
        fitnesses = state.fitnesses if hasattr(state, 'fitnesses') else None
        if fitnesses is None:
            return state

        # Find worst individual to replace
        worst_idx = int(np.argmin(np.asarray(fitnesses)))

        # Get population arrays
        nodes_arr = pop[0]
        conns_arr = pop[1]

        # Check shape compatibility
        if migrant_nodes.shape != nodes_arr[worst_idx].shape:
            return state
        if migrant_conns.shape != conns_arr[worst_idx].shape:
            return state

        # Replace worst individual with migrant
        new_nodes = nodes_arr.at[worst_idx].set(jnp.array(migrant_nodes))
        new_conns = conns_arr.at[worst_idx].set(jnp.array(migrant_conns))

        # Update state with modified population
        # This depends on how the NEAT algorithm stores population
        # For TensorNEAT-based algorithms, we need to update the state properly
        if hasattr(state, '_replace'):
            # Named tuple style
            if hasattr(state, 'pop_nodes') and hasattr(state, 'pop_conns'):
                state = state._replace(pop_nodes=new_nodes, pop_conns=new_conns)
        elif hasattr(state, 'pop'):
            # Dictionary style population
            if isinstance(state.pop, tuple) and len(state.pop) >= 2:
                state.pop = (new_nodes, new_conns) + state.pop[2:]

        return state

    except Exception:
        # If injection fails, return unchanged state
        return state


# ============================================================================
# Strategy 3: shard_map Implementation
# ============================================================================

def create_position_mesh(num_devices: Optional[int] = None) -> Any:
    """Create JAX mesh for position-level sharding.

    Args:
        num_devices: Number of devices to use (None = all available)

    Returns:
        JAX Mesh object configured for position sharding
    """
    if not SHARD_MAP_AVAILABLE:
        raise RuntimeError("shard_map not available in this JAX version")

    devices = jax.devices()
    if num_devices is None:
        num_devices = len(devices)
    else:
        num_devices = min(num_devices, len(devices))

    device_array = mesh_utils.create_device_mesh((num_devices,))
    return Mesh(device_array, axis_names=('positions',))


def create_hybrid_mesh(
    position_devices: int = 2,
    population_devices: int = 1
) -> Any:
    """Create 2D JAX mesh for hybrid sharding.

    Args:
        position_devices: Devices for position axis
        population_devices: Devices for population axis

    Returns:
        JAX Mesh object configured for 2D sharding
    """
    if not SHARD_MAP_AVAILABLE:
        raise RuntimeError("shard_map not available in this JAX version")

    total_devices = position_devices * population_devices
    if total_devices > len(jax.devices()):
        raise ValueError(
            f"Requested {total_devices} devices but only {len(jax.devices())} available"
        )

    device_array = mesh_utils.create_device_mesh(
        (position_devices, population_devices)
    )
    return Mesh(device_array, axis_names=('positions', 'population'))


# ============================================================================
# Multi-GPU Wrapper Class
# ============================================================================

class HMRHyperNEATMultiGPU(HMRHyperNEATAdaptiveChunking):
    """HMR-HyperNEAT with multi-GPU parallelization support.

    This class extends HMRHyperNEATAdaptiveChunking with
    production GPU execution strategies:

    1. SINGLE_GPU: Single GPU execution (for debugging or single-GPU systems)

    2. FULL_PIPELINE_PARALLEL (RECOMMENDED for feedforward):
       - Full pipeline runs on each GPU: CPPN → variance → W1/W2 → eval
       - Dataset is SPLIT across GPUs, pipeline is REPLICATED
       - Works for ALL depths including depth 7+ with pop=1000
       - Speedup: 1.7x-4.3x over SINGLE_GPU
       - Peak memory: ~350 MB (vs 3-4 GB without chunking)

    3. EVAL_ONLY_PARALLEL (RECOMMENDED for h→h modes):
       - Only evaluation parallelized, CPPN/h→h runs once on GPU 0
       - H→H caching enabled: 27s → 3ms after first generation
       - Dataset SHARDED, weights BROADCAST via native JAX
       - Best for recurrent modes where h→h caching saves significant time

    4. CPPN_CHUNKED (FALLBACK): Only chunks CPPN queries, not weight matrices.
       - Use if FULL_PIPELINE_PARALLEL has issues
       - Works for depths 1-6, OOMs at depth 7 with pop=1000

    Usage:
        # RECOMMENDED for feedforward: Full pipeline on each GPU
        algo = HMRHyperNEATMultiGPU(strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL)

        # RECOMMENDED for h→h modes: Only eval parallel, h→h cached
        algo = HMRHyperNEATMultiGPU(strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL)

        # FALLBACK: CPPN-only chunking (if FULL_PIPELINE_PARALLEL has issues)
        algo = HMRHyperNEATMultiGPU(strategy=MultiGPUStrategy.CPPN_CHUNKED)

        # Single GPU (for debugging)
        algo = HMRHyperNEATMultiGPU(strategy=MultiGPUStrategy.SINGLE_GPU)

        # IMPORTANT: Use run_until_threshold() for multi-GPU!
        result = algo.run_until_threshold(state, problem, 0.98, 100)

    Legacy aliases for backward compatibility:
        - BASELINE → SINGLE_GPU
        - MULTI_GPU → FULL_PIPELINE_PARALLEL
        - DATA_PARALLEL → FULL_PIPELINE_PARALLEL
        - PMAP_PARALLEL → EVAL_ONLY_PARALLEL
        - POSITION_SHARDING_CHUNKED → CPPN_CHUNKED
        - PIPELINE_CHUNKED → FULL_PIPELINE_PARALLEL
    """

    def __init__(
        self,
        name: str = 'hmr-hyperneat',
        implementation: str = 'tensorneat-hmrhyperneat-multi-gpu',
        strategy: MultiGPUStrategy = MultiGPUStrategy.SINGLE_GPU,
        position_config: Optional[PositionShardingConfig] = None,
        island_config: Optional[IslandModelConfig] = None,
        hybrid_config: Optional[HybridShardingConfig] = None,
        pmap_config: Optional[PopulationPmapConfig] = None,
        recurrence_config: Optional[RecurrenceConfig] = None,
    ):
        """Initialize multi-GPU HMR-HyperNEAT.

        Args:
            name: Algorithm name
            implementation: Implementation identifier
            strategy: GPU execution strategy (SINGLE_GPU, FULL_PIPELINE_PARALLEL, or EVAL_ONLY_PARALLEL)
            position_config: Configuration for position sharding (optional)
            island_config: Configuration for island model (legacy, optional)
            hybrid_config: Configuration for hybrid sharding (legacy, optional)
            pmap_config: Configuration for population pmap (legacy, optional)
            recurrence_config: Configuration for recurrent network h→h caching (optional).
                If provided, enables run_until_threshold_with_fixed_hh() method.
        """
        super().__init__(name=name, implementation=implementation)

        # Store recurrence config for h→h caching support
        self.recurrence_config = recurrence_config

        # Normalize strategy names (map user-facing names to internal implementation names)
        # FULL_PIPELINE_PARALLEL (recommended for feedforward) → PIPELINE_CHUNKED (internal)
        # EVAL_ONLY_PARALLEL (recommended for h→h) → direct handling
        # CPPN_CHUNKED (fallback) → POSITION_SHARDING_CHUNKED (CPPN-only chunking)
        if strategy == MultiGPUStrategy.SINGLE_GPU:
            strategy = MultiGPUStrategy.BASELINE
        elif strategy in (MultiGPUStrategy.FULL_PIPELINE_PARALLEL, MultiGPUStrategy.MULTI_GPU, MultiGPUStrategy.DATA_PARALLEL):
            strategy = MultiGPUStrategy.PIPELINE_CHUNKED  # Full pipeline chunking
        elif strategy in (MultiGPUStrategy.EVAL_ONLY_PARALLEL, MultiGPUStrategy.PMAP_PARALLEL):
            pass  # Keep as-is, handled by run_until_threshold routing
        elif strategy == MultiGPUStrategy.CPPN_CHUNKED:
            strategy = MultiGPUStrategy.POSITION_SHARDING_CHUNKED  # CPPN-only chunking

        self.strategy = strategy
        self._position_config = position_config
        self._island_config = island_config
        self._hybrid_config = hybrid_config
        self._pmap_config = pmap_config

        # Initialize strategy-specific components
        self._setup_strategy()

    def _setup_strategy(self):
        """Initialize components for the selected strategy."""
        num_devices = len(jax.devices())

        if self.strategy == MultiGPUStrategy.BASELINE:
            # No special setup needed
            pass

        elif self.strategy == MultiGPUStrategy.POSITION_SHARDING:
            if self._position_config is None:
                if num_devices >= 2:
                    self._position_config = PositionShardingConfig()
                else:
                    print(f"Warning: Position sharding requires 2+ devices, "
                          f"got {num_devices}. Falling back to baseline.")
                    self.strategy = MultiGPUStrategy.BASELINE

        elif self.strategy == MultiGPUStrategy.POSITION_SHARDING_CHUNKED:
            # Same as POSITION_SHARDING but will use chunked population processing
            if self._position_config is None:
                if num_devices >= 2:
                    self._position_config = PositionShardingConfig()
                else:
                    print(f"Warning: Position sharding chunked requires 2+ devices, "
                          f"got {num_devices}. Falling back to baseline.")
                    self.strategy = MultiGPUStrategy.BASELINE

        elif self.strategy == MultiGPUStrategy.PIPELINE_CHUNKED:
            # Full pipeline chunking for depth 7+ (chunks W1/W2 construction AND evaluation)
            if self._position_config is None:
                if num_devices >= 2:
                    self._position_config = PositionShardingConfig()
                else:
                    print(f"Warning: Pipeline chunked requires 2+ devices, "
                          f"got {num_devices}. Falling back to baseline.")
                    self.strategy = MultiGPUStrategy.BASELINE

        elif self.strategy == MultiGPUStrategy.ISLAND_MODEL:
            if self._island_config is None:
                if num_devices >= 2:
                    self._island_config = IslandModelConfig()
                else:
                    print(f"Warning: Island model requires 2+ devices, "
                          f"got {num_devices}. Falling back to baseline.")
                    self.strategy = MultiGPUStrategy.BASELINE

        elif self.strategy == MultiGPUStrategy.SHARD_MAP:
            if not SHARD_MAP_AVAILABLE:
                print("Warning: shard_map not available. Falling back to baseline.")
                self.strategy = MultiGPUStrategy.BASELINE
            elif num_devices < 2:
                print(f"Warning: shard_map requires 2+ devices, "
                      f"got {num_devices}. Falling back to baseline.")
                self.strategy = MultiGPUStrategy.BASELINE

        elif self.strategy == MultiGPUStrategy.HYBRID:
            if self._hybrid_config is None:
                if num_devices >= 2:
                    self._hybrid_config = HybridShardingConfig(
                        position_devices=min(2, num_devices),
                        population_devices=1
                    )
                else:
                    print(f"Warning: Hybrid sharding requires 2+ devices, "
                          f"got {num_devices}. Falling back to baseline.")
                    self.strategy = MultiGPUStrategy.BASELINE

        elif self.strategy == MultiGPUStrategy.POPULATION_PMAP:
            if self._pmap_config is None:
                if num_devices >= 2:
                    self._pmap_config = PopulationPmapConfig()
                else:
                    print(f"Warning: Population pmap requires 2+ devices, "
                          f"got {num_devices}. Falling back to baseline.")
                    self.strategy = MultiGPUStrategy.BASELINE

        elif self.strategy == MultiGPUStrategy.POSITION_PMAP:
            # Position-level pmap uses PositionShardingConfig for device management
            if self._position_config is None:
                if num_devices >= 2:
                    self._position_config = PositionShardingConfig()
                else:
                    print(f"Warning: Position pmap requires 2+ devices, "
                          f"got {num_devices}. Falling back to baseline.")
                    self.strategy = MultiGPUStrategy.BASELINE

    def get_strategy(self) -> MultiGPUStrategy:
        """Get the active multi-GPU strategy."""
        return self.strategy

    def get_num_devices(self) -> int:
        """Get number of devices being used."""
        if self.strategy == MultiGPUStrategy.BASELINE:
            return 1
        elif self.strategy in (MultiGPUStrategy.POSITION_SHARDING, MultiGPUStrategy.POSITION_SHARDING_CHUNKED):
            return self._position_config.num_devices if self._position_config else 1
        elif self.strategy == MultiGPUStrategy.ISLAND_MODEL:
            return self._island_config.num_islands if self._island_config else 1
        elif self.strategy == MultiGPUStrategy.HYBRID:
            cfg = self._hybrid_config
            return cfg.position_devices * cfg.population_devices if cfg else 1
        elif self.strategy == MultiGPUStrategy.POPULATION_PMAP:
            return self._pmap_config.num_devices if self._pmap_config else 1
        elif self.strategy == MultiGPUStrategy.POSITION_PMAP:
            return self._position_config.num_devices if self._position_config else 1
        return len(jax.devices())

    def run_until_threshold_multi_gpu(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
    ) -> Dict[str, Any]:
        """Run evolution with multi-GPU support.

        For Island Model, this spawns separate processes.
        For other strategies, uses the parent implementation with sharding.

        Args:
            state: Initial algorithm state
            problem: Problem to solve
            target_fitness: Fitness threshold for success
            max_generations: Maximum generations

        Returns:
            Dict with evolution results
        """
        if self.strategy == MultiGPUStrategy.ISLAND_MODEL:
            # Use Island Model runner
            if self._island_config is None:
                self._island_config = IslandModelConfig()

            runner = IslandModelRunner(
                algorithm_class=HMRHyperNEATAdaptiveChunking,
                config_path=self.config_manager.config_path if hasattr(self.config_manager, 'config_path') else None,
                problem=problem,
                config=self._island_config,
            )
            return runner.run(target_fitness, max_generations)

        else:
            # Use parent implementation (baseline or with position sharding)
            return super().run_until_threshold(state, problem, target_fitness, max_generations)

    def _select_query_function(
        self,
        total_positions: int,
        pop_size: int,
    ) -> Callable:
        """Select the appropriate query function based on strategy.

        Args:
            total_positions: Number of positions to query
            pop_size: Population size

        Returns:
            Query function to use
        """
        if self.strategy == MultiGPUStrategy.POSITION_SHARDING and self._position_config:
            # Use position-sharded query
            return lambda state, cppns, sources, targets, outgoing, forward: \
                batch_query_population_multi_source_sharded(
                    state, cppns, sources, targets, outgoing, forward,
                    self._position_config
                )

        elif self.strategy == MultiGPUStrategy.HYBRID and self._hybrid_config:
            # fall back to chunked
            return batch_query_population_multi_source_chunked

        else:
            # Use parent selection (baseline or chunked)
            return super()._select_query_function(total_positions, pop_size)

    def run_generation_verbose(
        self, state: Any, problem: Any, skip_metrics: bool = False
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation with multi-GPU sharding if enabled.

        For POSITION_SHARDING strategy, uses multi-GPU shard_map queries.
        For POPULATION_PMAP strategy, uses multi-GPU pmap queries.
        For POSITION_PMAP strategy, uses pmap with position sharding (for large depths).
        For other strategies, uses the parent implementation.
        """
        # Handle POPULATION_PMAP strategy
        if self.strategy == MultiGPUStrategy.POPULATION_PMAP and self._pmap_config is not None:
            return self._run_generation_pmap(state, problem, skip_metrics)

        # Handle POSITION_PMAP strategy (position-level pmap for large depths)
        if self.strategy == MultiGPUStrategy.POSITION_PMAP and self._position_config is not None:
            return self._run_generation_position_pmap(state, problem, skip_metrics)

        # For non-position-sharding strategies, use parent implementation
        if self.strategy != MultiGPUStrategy.POSITION_SHARDING or self._position_config is None:
            return super().run_generation_verbose(state, problem, skip_metrics)

        # POSITION_SHARDING: Use multi-GPU shard_map queries
        gen_start = time.time()
        step_timings = {}

        # Optional pre-tell random key split
        if self.extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # STEP 0: CPPN ask + transform
        t0 = time.perf_counter()
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        step_timings['step0_cppn_ask_transform'] = time.perf_counter() - t0

        # Get hierarchical grid for current max_depth
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions

        # Use cached coordinate arrays
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # STEP 1+3 UNIFIED: Query with position sharding
        t1 = time.perf_counter()

        all_positions = h_grid.all_positions
        total_positions = all_positions.shape[0]

        # USE SHARDED QUERY FUNCTION for multi-GPU parallelism
        position_config = self._position_config

        # Input → all positions (outgoing): shape (pop_size, num_inputs, total_positions)
        input_all_weights = batch_query_population_multi_source_sharded(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward,
            position_config
        )

        # All → output (incoming): shape (pop_size, num_outputs, total_positions)
        output_all_weights = batch_query_population_multi_source_sharded(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward,
            position_config
        )

        # Extract variance weights from input[0]
        all_weights_for_variance = input_all_weights[:, 0, :]

        step_timings['step1_unified_cppn_query'] = time.perf_counter() - t1

        # STEP 2: Compute hierarchical variances and subdivision masks
        t2 = time.perf_counter()

        level_variances = compute_hierarchical_variances_batch(
            all_weights_for_variance, h_grid
        )

        if self.skip_unused_masks:
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )
        else:
            masks_A, _, _ = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=True
            )

        step_timings['step2_variance_masks'] = time.perf_counter() - t2
        step_timings['step3_weight_queries'] = 0.0  # Already included in step1

        # STEP 4: Apply masks and build weight matrices
        t4 = time.perf_counter()

        max_weight = self.max_weight
        weight_thresh = 0.1
        active_mask_broadcast = masks_A[:, None, :]

        if self.fuse_w1_computation:
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(jnp.tanh(input_all_weights) * max_weight) > weight_thresh),
                jnp.tanh(input_all_weights) * max_weight,
                0.0
            )
            W2_masked = jnp.where(
                active_mask_broadcast & (jnp.abs(jnp.tanh(output_all_weights) * max_weight) > weight_thresh),
                jnp.tanh(output_all_weights) * max_weight,
                0.0
            )
            W2 = jnp.transpose(W2_masked, (0, 2, 1))
        else:
            W1_raw = jnp.tanh(input_all_weights) * max_weight
            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )
            W2_raw = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )
            W2 = jnp.transpose(W2_raw, (0, 2, 1))

        step_timings['step4_weight_matrix_build'] = time.perf_counter() - t4

        # STEP 5: Evaluate ALL networks via vmap
        t5 = time.perf_counter()

        total_positions = h_grid.total_positions
        use_sparse = (
            self.sparse_forward_threshold >= 0 and
            total_positions > self.sparse_forward_threshold
        )

        if use_sparse:
            union_mask = jnp.any(masks_A, axis=0)
            active_indices = jnp.where(union_mask)[0]
            num_active = active_indices.shape[0]
            W1_sparse = W1[:, :, active_indices]
            W2_sparse = W2[:, active_indices, :]

            def eval_single_network_sparse(W1_single, W2_single, inputs, targets):
                hidden = jnp.tanh(safe_matmul(inputs, W1_single))
                outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
                errors = jnp.mean((outputs - targets) ** 2, axis=1)
                return 1.0 - jnp.mean(errors)

            fitnesses = jax.vmap(eval_single_network_sparse, in_axes=(0, 0, None, None))(
                W1_sparse, W2_sparse, inputs_batch, targets_batch
            )
        else:
            def eval_single_network(W1_single, W2_single, inputs, targets):
                hidden = jnp.tanh(safe_matmul(inputs, W1_single))
                outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
                errors = jnp.mean((outputs - targets) ** 2, axis=1)
                return 1.0 - jnp.mean(errors)

            fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
                W1, W2, inputs_batch, targets_batch
            )

        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
        step_timings['step5_network_eval'] = time.perf_counter() - t5

        # STEP 6: NEAT evolution step
        t6 = time.perf_counter()
        new_state = self._compiled_tell(state, fitnesses)
        step_timings['step6_tell'] = time.perf_counter() - t6

        # Build metrics matching parent class format
        t7 = time.perf_counter()

        if skip_metrics:
            metrics = AlgorithmMetrics(
                generation=new_state.generation if hasattr(new_state, 'generation') else 0,
                best_fitness=0.0,
                mean_fitness=0.0,
                min_fitness=0.0,
                max_fitness=0.0,
                std_fitness=0.0,
                num_species=0,
                species_sizes=[],
                species_fitness=[],
                evaluations=0,
                time_elapsed=time.time() - gen_start,
            )
            metrics.custom_metrics = {'skip_metrics': True}
        else:
            # Extract metrics with single sync
            fitnesses_np = np.array(jax.device_get(fitnesses))
            best_fitness = float(np.max(fitnesses_np))
            mean_fitness = float(np.mean(fitnesses_np))
            min_fitness = float(np.min(fitnesses_np))
            std_fitness = float(np.std(fitnesses_np))

            metrics = AlgorithmMetrics(
                generation=new_state.generation if hasattr(new_state, 'generation') else 0,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                min_fitness=min_fitness,
                max_fitness=best_fitness,
                std_fitness=std_fitness,
                num_species=0,
                species_sizes=[],
                species_fitness=[],
                evaluations=pop_size,
                time_elapsed=time.time() - gen_start,
            )

        step_timings['step7_metrics_extraction'] = time.perf_counter() - t7

        # Initialize custom_metrics if None
        if metrics.custom_metrics is None:
            metrics.custom_metrics = {}

        # Add step timings to custom metrics
        metrics.custom_metrics['step_timings'] = step_timings
        metrics.custom_metrics['method'] = 'multi_gpu_position_sharding'
        metrics.custom_metrics['num_devices'] = self._position_config.num_devices
        metrics.custom_metrics['variance_threshold'] = self.variance_threshold
        metrics.custom_metrics['total_positions'] = total_positions

        return new_state, metrics

    def _run_generation_pmap(
        self, state: Any, problem: Any, skip_metrics: bool = False
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation using pmap-based population sharding.

        This is the cleanest multi-GPU approach:
        - Population is split across GPUs
        - Each GPU processes its slice independently
        - No synchronization until results are gathered

        This should give true ~2x speedup for 2 GPUs because:
        - No shard_map overhead
        - No check_rep=False workaround needed
        - Each GPU does half the CPPN queries
        """
        gen_start = time.time()
        step_timings = {}
        pmap_config = self._pmap_config

        # Optional pre-tell random key split
        if self.extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # STEP 0: CPPN ask + transform
        t0 = time.perf_counter()
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        step_timings['step0_cppn_ask_transform'] = time.perf_counter() - t0

        # Get hierarchical grid for current max_depth
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions

        # Use cached coordinate arrays
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # STEP 1+3 UNIFIED: Query with population pmap
        t1 = time.perf_counter()

        all_positions = h_grid.all_positions

        # USE PMAP QUERY FUNCTION for multi-GPU parallelism
        # Input → all positions (outgoing): shape (pop_size, num_inputs, total_positions)
        input_all_weights = batch_query_population_multi_source_pmap(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward,
            pmap_config
        )

        # All → output (incoming): shape (pop_size, num_outputs, total_positions)
        output_all_weights = batch_query_population_multi_source_pmap(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward,
            pmap_config
        )

        # Extract variance weights from input[0]
        all_weights_for_variance = input_all_weights[:, 0, :]

        step_timings['step1_unified_cppn_query'] = time.perf_counter() - t1

        # STEP 2: Compute hierarchical variances and subdivision masks
        t2 = time.perf_counter()

        level_variances = compute_hierarchical_variances_batch(
            all_weights_for_variance, h_grid
        )

        if self.skip_unused_masks:
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )
        else:
            masks_A, _, _ = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=True
            )

        step_timings['step2_variance_masks'] = time.perf_counter() - t2
        step_timings['step3_weight_queries'] = 0.0  # Already included in step1

        # STEP 4: Apply masks and build weight matrices
        t4 = time.perf_counter()

        max_weight = self.max_weight
        weight_thresh = 0.1
        active_mask_broadcast = masks_A[:, None, :]

        if self.fuse_w1_computation:
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(jnp.tanh(input_all_weights) * max_weight) > weight_thresh),
                jnp.tanh(input_all_weights) * max_weight,
                0.0
            )
            W2_masked = jnp.where(
                active_mask_broadcast & (jnp.abs(jnp.tanh(output_all_weights) * max_weight) > weight_thresh),
                jnp.tanh(output_all_weights) * max_weight,
                0.0
            )
            W2 = jnp.transpose(W2_masked, (0, 2, 1))
        else:
            W1_raw = jnp.tanh(input_all_weights) * max_weight
            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )
            W2_raw = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )
            W2 = jnp.transpose(W2_raw, (0, 2, 1))

        step_timings['step4_weight_matrix_build'] = time.perf_counter() - t4

        # STEP 5: Evaluate ALL networks via vmap
        t5 = time.perf_counter()

        use_sparse = (
            self.sparse_forward_threshold >= 0 and
            total_positions > self.sparse_forward_threshold
        )

        if use_sparse:
            union_mask = jnp.any(masks_A, axis=0)
            active_indices = jnp.where(union_mask)[0]
            W1_sparse = W1[:, :, active_indices]
            W2_sparse = W2[:, active_indices, :]

            def eval_single_network_sparse(W1_single, W2_single, inputs, targets):
                hidden = jnp.tanh(safe_matmul(inputs, W1_single))
                outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
                errors = jnp.mean((outputs - targets) ** 2, axis=1)
                return 1.0 - jnp.mean(errors)

            fitnesses = jax.vmap(eval_single_network_sparse, in_axes=(0, 0, None, None))(
                W1_sparse, W2_sparse, inputs_batch, targets_batch
            )
        else:
            def eval_single_network(W1_single, W2_single, inputs, targets):
                hidden = jnp.tanh(safe_matmul(inputs, W1_single))
                outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
                errors = jnp.mean((outputs - targets) ** 2, axis=1)
                return 1.0 - jnp.mean(errors)

            fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
                W1, W2, inputs_batch, targets_batch
            )

        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
        step_timings['step5_network_eval'] = time.perf_counter() - t5

        # STEP 6: NEAT evolution step
        t6 = time.perf_counter()
        new_state = self._compiled_tell(state, fitnesses)
        step_timings['step6_tell'] = time.perf_counter() - t6

        # Build metrics
        t7 = time.perf_counter()

        if skip_metrics:
            metrics = AlgorithmMetrics(
                generation=new_state.generation if hasattr(new_state, 'generation') else 0,
                best_fitness=0.0,
                mean_fitness=0.0,
                min_fitness=0.0,
                max_fitness=0.0,
                std_fitness=0.0,
                num_species=0,
                species_sizes=[],
                species_fitness=[],
                evaluations=0,
                time_elapsed=time.time() - gen_start,
            )
            metrics.custom_metrics = {'skip_metrics': True}
        else:
            # Extract metrics with single sync
            fitnesses_np = np.array(jax.device_get(fitnesses))
            best_fitness = float(np.max(fitnesses_np))
            mean_fitness = float(np.mean(fitnesses_np))
            min_fitness = float(np.min(fitnesses_np))
            std_fitness = float(np.std(fitnesses_np))

            metrics = AlgorithmMetrics(
                generation=new_state.generation if hasattr(new_state, 'generation') else 0,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                min_fitness=min_fitness,
                max_fitness=best_fitness,
                std_fitness=std_fitness,
                num_species=0,
                species_sizes=[],
                species_fitness=[],
                evaluations=pop_size,
                time_elapsed=time.time() - gen_start,
            )

        step_timings['step7_metrics_extraction'] = time.perf_counter() - t7

        # Initialize custom_metrics if None
        if metrics.custom_metrics is None:
            metrics.custom_metrics = {}

        # Add step timings to custom metrics
        metrics.custom_metrics['step_timings'] = step_timings
        metrics.custom_metrics['method'] = 'multi_gpu_population_pmap'
        metrics.custom_metrics['num_devices'] = pmap_config.num_devices
        metrics.custom_metrics['variance_threshold'] = self.variance_threshold
        metrics.custom_metrics['total_positions'] = total_positions

        return new_state, metrics

    def _run_generation_position_pmap(
        self, state: Any, problem: Any, skip_metrics: bool = False
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation using pmap-based POSITION sharding.

        This is for LARGE DEPTHS (7+) where the bottleneck is position count:
        - Depth 7: 87,380 positions
        - Depth 8: 349,524 positions

        Key difference from _run_generation_pmap (population sharding):
        - This shards POSITIONS across GPUs (not population)
        - Each GPU processes half the positions for ALL population members
        - Better when position count >> population size

        Uses query_positions_batched_multi_gpu which handles:
        - Padding for even distribution
        - Batching for very large depths (8+)
        """
        gen_start = time.time()
        step_timings = {}
        position_config = self._position_config

        # Optional pre-tell random key split
        if self.extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # STEP 0: CPPN ask + transform
        t0 = time.perf_counter()
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        step_timings['step0_cppn_ask_transform'] = time.perf_counter() - t0

        # Get hierarchical grid for current max_depth
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions

        # Use cached coordinate arrays
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # STEP 1+3 UNIFIED: Query with POSITION pmap
        t1 = time.perf_counter()

        all_positions = h_grid.all_positions

        # USE CHUNKED BASELINE QUERY FUNCTION - the pmap version has a bug causing XOR plateau
        # Using chunked version for memory efficiency at large depths (6+)
        # Input → all positions (outgoing): shape (pop_size, num_inputs, total_positions)
        input_all_weights = batch_query_population_multi_source_chunked(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward,
            pop_chunk_size=100  # Process 100 genomes at a time to avoid OOM
        )

        # All → output (incoming): shape (pop_size, num_outputs, total_positions)
        output_all_weights = batch_query_population_multi_source_chunked(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward,
            pop_chunk_size=100  # Process 100 genomes at a time to avoid OOM
        )

        # Extract variance weights from input[0]
        all_weights_for_variance = input_all_weights[:, 0, :]

        step_timings['step1_unified_cppn_query'] = time.perf_counter() - t1

        # STEP 2: Compute hierarchical variances and subdivision masks
        t2 = time.perf_counter()

        level_variances = compute_hierarchical_variances_batch(
            all_weights_for_variance, h_grid
        )

        if self.skip_unused_masks:
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )
        else:
            masks_A, _, _ = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=True
            )

        step_timings['step2_variance_masks'] = time.perf_counter() - t2
        step_timings['step3_weight_queries'] = 0.0  # Already included in step1

        # STEP 4: Apply masks and build weight matrices
        t4 = time.perf_counter()

        max_weight = self.max_weight
        weight_thresh = 0.1
        active_mask_broadcast = masks_A[:, None, :]

        if self.fuse_w1_computation:
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(jnp.tanh(input_all_weights) * max_weight) > weight_thresh),
                jnp.tanh(input_all_weights) * max_weight,
                0.0
            )
            W2_masked = jnp.where(
                active_mask_broadcast & (jnp.abs(jnp.tanh(output_all_weights) * max_weight) > weight_thresh),
                jnp.tanh(output_all_weights) * max_weight,
                0.0
            )
            W2 = jnp.transpose(W2_masked, (0, 2, 1))
        else:
            W1_raw = jnp.tanh(input_all_weights) * max_weight
            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )
            W2_raw = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )
            W2 = jnp.transpose(W2_raw, (0, 2, 1))

        step_timings['step4_weight_matrix_build'] = time.perf_counter() - t4

        # STEP 5: Evaluate ALL networks via vmap
        t5 = time.perf_counter()

        use_sparse = (
            self.sparse_forward_threshold >= 0 and
            total_positions > self.sparse_forward_threshold
        )

        if use_sparse:
            union_mask = jnp.any(masks_A, axis=0)
            active_indices = jnp.where(union_mask)[0]
            W1_sparse = W1[:, :, active_indices]
            W2_sparse = W2[:, active_indices, :]

            def eval_single_network_sparse(W1_single, W2_single, inputs, targets):
                hidden = jnp.tanh(safe_matmul(inputs, W1_single))
                outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
                errors = jnp.mean((outputs - targets) ** 2, axis=1)
                return 1.0 - jnp.mean(errors)

            fitnesses = jax.vmap(eval_single_network_sparse, in_axes=(0, 0, None, None))(
                W1_sparse, W2_sparse, inputs_batch, targets_batch
            )
        else:
            def eval_single_network(W1_single, W2_single, inputs, targets):
                hidden = jnp.tanh(safe_matmul(inputs, W1_single))
                outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
                errors = jnp.mean((outputs - targets) ** 2, axis=1)
                return 1.0 - jnp.mean(errors)

            fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
                W1, W2, inputs_batch, targets_batch
            )

        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
        step_timings['step5_network_eval'] = time.perf_counter() - t5

        # STEP 6: NEAT evolution step
        t6 = time.perf_counter()
        new_state = self._compiled_tell(state, fitnesses)
        step_timings['step6_tell'] = time.perf_counter() - t6

        # Build metrics
        t7 = time.perf_counter()

        if skip_metrics:
            metrics = AlgorithmMetrics(
                generation=new_state.generation if hasattr(new_state, 'generation') else 0,
                best_fitness=0.0,
                mean_fitness=0.0,
                min_fitness=0.0,
                max_fitness=0.0,
                std_fitness=0.0,
                num_species=0,
                species_sizes=[],
                species_fitness=[],
                evaluations=0,
                time_elapsed=time.time() - gen_start,
            )
            metrics.custom_metrics = {'skip_metrics': True}
        else:
            # Extract metrics with single sync
            fitnesses_np = np.array(jax.device_get(fitnesses))
            best_fitness = float(np.max(fitnesses_np))
            mean_fitness = float(np.mean(fitnesses_np))
            min_fitness = float(np.min(fitnesses_np))
            std_fitness = float(np.std(fitnesses_np))

            metrics = AlgorithmMetrics(
                generation=new_state.generation if hasattr(new_state, 'generation') else 0,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                min_fitness=min_fitness,
                max_fitness=best_fitness,
                std_fitness=std_fitness,
                num_species=0,
                species_sizes=[],
                species_fitness=[],
                evaluations=pop_size,
                time_elapsed=time.time() - gen_start,
            )

        step_timings['step7_metrics_extraction'] = time.perf_counter() - t7

        # Initialize custom_metrics if None
        if metrics.custom_metrics is None:
            metrics.custom_metrics = {}

        # Add step timings to custom metrics
        metrics.custom_metrics['step_timings'] = step_timings
        metrics.custom_metrics['method'] = 'multi_gpu_position_pmap'
        metrics.custom_metrics['num_devices'] = position_config.num_devices
        metrics.custom_metrics['variance_threshold'] = self.variance_threshold
        metrics.custom_metrics['total_positions'] = total_positions

        return new_state, metrics

    def _pure_generation_step_multi_gpu(
        self,
        state: Any,
        cppns_transformed: Any,
        h_grid: Any,
        input_coords: jnp.ndarray,
        output_coords: jnp.ndarray,
        inputs_batch: jnp.ndarray,
        targets_batch: jnp.ndarray,
        position_config: PositionShardingConfig,
        extra_randkey_split: bool = False,
    ) -> Tuple[Any, jnp.ndarray]:
        """Single generation step with position sharding across GPUs.

        This is a separate method from _pure_generation_step because we cannot
        use Python `if` statements to select between sharded and non-sharded
        queries inside a traced region (jax.lax.while_loop).

        The caller (run_until_threshold) dispatches to either this method or
        the parent's _pure_generation_step BEFORE entering the while_loop.

        Args:
            state: TensorNEAT algorithm state (JAX pytree)
            cppns_transformed: Pre-transformed CPPN population
            h_grid: Hierarchical grid configuration
            input_coords: Substrate input coordinates (JAX array)
            output_coords: Substrate output coordinates (JAX array)
            inputs_batch: Cached problem inputs (JAX array)
            targets_batch: Cached problem targets (JAX array)
            position_config: Position sharding configuration
            extra_randkey_split: If True, split random key before tell()

        Returns:
            Tuple of (new_state, fitnesses) where both are JAX arrays/pytrees.
        """
        # Optional pre-tell random key split
        if extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # STEP 0: CPPN ask + transform
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Replicate CPPNs to all devices in mesh for shard_map multi-GPU execution
        # Without this, CPPNs only exist on GPU 0 and shard_map cannot distribute work
        replicated_sharding = NamedSharding(position_config.mesh, P())
        cppns_transformed = tuple(
            jax.device_put(arr, replicated_sharding) for arr in cppns_transformed
        )

        # Get grid info
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        # STEP 1: Query CPPN at all positions for variance computation
        # USING SHARDED VERSION for multi-GPU parallelism
        source_coord = input_coords[0:1]
        all_weights_for_variance = batch_query_population_multi_source_sharded(
            state, cppns_transformed, source_coord,
            all_positions, True, self._jitted_cppn_forward,
            position_config
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        # STEP 2: Compute hierarchical variances and subdivision masks
        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # STEP 3: Query weights for input→all and all→output connections
        # USING SHARDED VERSION for multi-GPU parallelism
        input_all_weights = batch_query_population_multi_source_sharded(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward,
            position_config
        )
        output_all_weights = batch_query_population_multi_source_sharded(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward,
            position_config
        )

        # STEP 4: Apply masks and build weight matrices
        max_weight = self.max_weight
        weight_thresh = 0.1

        W1_raw = jnp.tanh(input_all_weights) * max_weight
        W2_raw = jnp.tanh(output_all_weights) * max_weight

        active_mask_broadcast = masks_A[:, None, :]
        W1_combined_mask = active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh)
        W1 = W1_raw * W1_combined_mask

        W2_combined_mask = active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh)
        W2_raw = W2_raw * W2_combined_mask
        W2 = jnp.transpose(W2_raw, (0, 2, 1))

        # STEP 5: Evaluate ALL networks via vmap
        def eval_single_network(W1_single, W2_single, inputs, targets):
            hidden = jnp.tanh(safe_matmul(inputs, W1_single))
            outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
            W1, W2, inputs_batch, targets_batch
        )
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        # STEP 6: NEAT evolution step
        new_state = self._compiled_tell(state, fitnesses)

        return new_state, fitnesses

    def _pure_generation_step_multi_gpu_chunked(
        self,
        state: Any,
        cppns_transformed: Any,
        h_grid: Any,
        input_coords: jnp.ndarray,
        output_coords: jnp.ndarray,
        inputs_batch: jnp.ndarray,
        targets_batch: jnp.ndarray,
        position_config: PositionShardingConfig,
        pop_chunk_size: int,
        extra_randkey_split: bool = False,
    ) -> Tuple[Any, jnp.ndarray]:
        """Single generation step with position sharding AND population chunking.

        This combines two optimization strategies:
        1. Position sharding: Distribute positions across GPUs for parallelism
        2. Population chunking: Process population in chunks to reduce peak memory

        Use this for depth 6+ where:
        - Position count is high (5,461+ positions)
        - Memory is constrained (would OOM without chunking)
        - Multi-GPU speedup is beneficial

        Args:
            state: TensorNEAT algorithm state (JAX pytree)
            cppns_transformed: Pre-transformed CPPN population
            h_grid: Hierarchical grid configuration
            input_coords: Substrate input coordinates (JAX array)
            output_coords: Substrate output coordinates (JAX array)
            inputs_batch: Cached problem inputs (JAX array)
            targets_batch: Cached problem targets (JAX array)
            position_config: Position sharding configuration
            pop_chunk_size: Chunk size for population processing
            extra_randkey_split: If True, split random key before tell()

        Returns:
            Tuple of (new_state, fitnesses) where both are JAX arrays/pytrees.
        """
        # Optional pre-tell random key split
        if extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # STEP 0: CPPN ask + transform
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Replicate CPPNs to all devices in mesh for shard_map multi-GPU execution
        # Without this, CPPNs only exist on GPU 0 and shard_map cannot distribute work
        replicated_sharding = NamedSharding(position_config.mesh, P())
        cppns_transformed = tuple(
            jax.device_put(arr, replicated_sharding) for arr in cppns_transformed
        )

        # Get grid info
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        # STEP 1: Query CPPN at all positions for variance computation
        # USING CHUNKED+SHARDED VERSION for memory efficiency + multi-GPU
        source_coord = input_coords[0:1]
        all_weights_for_variance = batch_query_population_multi_source_chunked_sharded(
            state, cppns_transformed, source_coord,
            all_positions, True, self._jitted_cppn_forward,
            position_config, pop_chunk_size
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        # STEP 2: Compute hierarchical variances and subdivision masks
        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # STEP 3: Query weights for input→all and all→output connections
        # USING CHUNKED+SHARDED VERSION for memory efficiency + multi-GPU
        input_all_weights = batch_query_population_multi_source_chunked_sharded(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward,
            position_config, pop_chunk_size
        )
        output_all_weights = batch_query_population_multi_source_chunked_sharded(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward,
            position_config, pop_chunk_size
        )

        # STEP 4: Apply masks and build weight matrices
        max_weight = self.max_weight
        weight_thresh = 0.1

        W1_raw = jnp.tanh(input_all_weights) * max_weight
        W2_raw = jnp.tanh(output_all_weights) * max_weight

        active_mask_broadcast = masks_A[:, None, :]
        W1_combined_mask = active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh)
        W1 = W1_raw * W1_combined_mask

        W2_combined_mask = active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh)
        W2_raw = W2_raw * W2_combined_mask
        W2 = jnp.transpose(W2_raw, (0, 2, 1))

        # STEP 5: Evaluate ALL networks via vmap
        def eval_single_network(W1_single, W2_single, inputs, targets):
            hidden = jnp.tanh(safe_matmul(inputs, W1_single))
            outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
            W1, W2, inputs_batch, targets_batch
        )
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        # STEP 6: NEAT evolution step
        new_state = self._compiled_tell(state, fitnesses)

        return new_state, fitnesses

    def _pure_generation_step_pipeline_chunked(
        self,
        state: Any,
        cppns_transformed: Any,
        h_grid: Any,
        input_coords: jnp.ndarray,
        output_coords: jnp.ndarray,
        inputs_batch: jnp.ndarray,
        targets_batch: jnp.ndarray,
        position_config: PositionShardingConfig,
        pop_chunk_size: int,
        extra_randkey_split: bool = False,
    ) -> Tuple[Any, jnp.ndarray]:
        """Single generation step with FULL PIPELINE chunking.

        Chunks ALL memory-intensive operations:
        1. CPPN queries (existing chunking via batch_query_population_multi_source_chunked_sharded)
        2. Weight matrix construction + evaluation (NEW unified chunking via build_and_evaluate_chunked)

        This enables depth 7 (87,380 positions) with pop=1000 on 11 GiB GPUs.
        Peak memory: ~600 MB instead of 3-4 GB.

        Args:
            state: TensorNEAT algorithm state (JAX pytree)
            cppns_transformed: Pre-transformed CPPN population
            h_grid: Hierarchical grid configuration
            input_coords: Substrate input coordinates (JAX array)
            output_coords: Substrate output coordinates (JAX array)
            inputs_batch: Cached problem inputs (JAX array)
            targets_batch: Cached problem targets (JAX array)
            position_config: Position sharding configuration
            pop_chunk_size: Chunk size for population processing
            extra_randkey_split: If True, split random key before tell()

        Returns:
            Tuple of (new_state, fitnesses) where both are JAX arrays/pytrees.
        """
        # Optional pre-tell random key split
        if extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # STEP 0: CPPN ask + transform
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Replicate CPPNs to all devices in mesh for shard_map multi-GPU execution
        replicated_sharding = NamedSharding(position_config.mesh, P())
        cppns_transformed = tuple(
            jax.device_put(arr, replicated_sharding) for arr in cppns_transformed
        )

        # Get grid info
        all_positions = h_grid.all_positions

        # STEP 1: Query CPPN for variance (CHUNKED via existing function)
        source_coord = input_coords[0:1]
        all_weights_for_variance = batch_query_population_multi_source_chunked_sharded(
            state, cppns_transformed, source_coord,
            all_positions, True, self._jitted_cppn_forward,
            position_config, pop_chunk_size
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        # STEP 2: Compute hierarchical variances and masks
        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # STEP 3: Query weights (CHUNKED via existing function)
        input_all_weights = batch_query_population_multi_source_chunked_sharded(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward,
            position_config, pop_chunk_size
        )
        output_all_weights = batch_query_population_multi_source_chunked_sharded(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward,
            position_config, pop_chunk_size
        )

        # STEP 4+5: Build weight matrices AND evaluate (NEW UNIFIED CHUNKING)
        # This is the KEY CHANGE - unified function never holds full W1/W2 in memory
        # NOTE: Multi-GPU pmap evaluation conflicts with shard_map CPPN queries.
        # Using single-GPU chunked evaluation which is GPU-resident and memory-efficient.
        fitnesses = build_and_evaluate_chunked(
            input_all_weights,
            output_all_weights,
            masks_A,
            inputs_batch,
            targets_batch,
            pop_chunk_size=pop_chunk_size,
            max_weight=self.max_weight,
            weight_thresh=0.1,
        )

        # STEP 6: NEAT evolution step
        new_state = self._compiled_tell(state, fitnesses)

        return new_state, fitnesses

    def run_until_threshold(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        streaming_chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """Run multiple generations on GPU until fitness threshold is met.

        Routes to the appropriate implementation based on strategy:

        ┌──────────────────────────────────┬─────────────────────────────────────────────────┐
        │ Strategy                         │ Implementation                                  │
        ├──────────────────────────────────┼─────────────────────────────────────────────────┤
        │ MULTI_GPU (default)              │ Data-Parallel: splits dataset across GPUs      │
        │ DATA_PARALLEL                    │ Same as MULTI_GPU                              │
        │ POPULATION_PARALLEL_SEQUENTIAL   │ Splits population, sequential h→h (tested)     │
        │ POPULATION_PARALLEL              │ Not implemented in this release                │
        │ STREAMING                        │ Streams dataset chunks from CPU                │
        │ SINGLE_GPU                       │ Single GPU execution                           │
        │ CPPN_CHUNKED                     │ Legacy: CPPN-only chunking                     │
        └──────────────────────────────────┴─────────────────────────────────────────────────┘

        When to use each:
        - MULTI_GPU: Default. Best for most cases. 2x dataset memory capacity.
        - POPULATION_PARALLEL_SEQUENTIAL: When population >> dataset (memory-wise).
          Note: H→H modes are 2-3x slower than single-GPU, but 0 errors.
        - STREAMING: When dataset exceeds total GPU memory
        - SINGLE_GPU: Debugging, single-GPU systems, or h→h-heavy workloads

        Args:
            state: Initialized algorithm state from initialize()
            problem: Problem instance (must have get_data() method)
            target_fitness: Stop when jnp.max(fitnesses) >= target_fitness
            max_generations: Maximum generations before stopping
            collect_history: If True, collect per-generation best fitness history
            streaming_chunk_size: Chunk size for STREAMING strategy (default 1000)

        Returns:
            Dict with evolution results
        """
        # ====================================================================
        # STRATEGY ROUTING
        # ====================================================================
        #
        # FULL_PIPELINE_PARALLEL (+ aliases MULTI_GPU, DATA_PARALLEL) → Full pipeline on each GPU
        #   - Recommended for feedforward: same speed as population-parallel but 2x memory
        #   - Each GPU runs: CPPN → variance → W1/W2 → eval on data shard
        #   - Each GPU: (dataset / num_gpus) + full population
        #
        # EVAL_ONLY_PARALLEL (+ alias PMAP_PARALLEL) → Only eval parallel, h→h cached on GPU 0
        #   - Recommended for h→h modes: caching saves ~27s → 3ms per generation
        #   - GPU 0: CPPN → variance → h→h (cached) → W1/W2
        #   - All GPUs: eval on data shards via pmap
        #
        # POPULATION_PARALLEL_SEQUENTIAL → Population-Parallel (splits population)
        #   - Use when: population >> dataset (memory-wise)
        #   - Each GPU: full dataset + (population / num_gpus)
        #   - H→H: Sequential (one GPU at a time) to avoid JIT cache errors
        #   - Trade-off: H→H modes 2-3x slower than single-GPU, but 0 errors
        #
        # POPULATION_PARALLEL → not implemented in this release
        #   - Future: True parallel h→h processing across GPUs
        #
        # STREAMING → Streaming from CPU
        #   - Use when: dataset exceeds total GPU memory
        #   - Each GPU: chunk_size samples at a time
        #   - Trade-off: 2-10x slower due to CPU-GPU transfers
        #
        # ====================================================================

        num_devices = len(jax.devices())

        # Route FULL_PIPELINE_PARALLEL and aliases to Data-Parallel implementation
        if self.strategy in (MultiGPUStrategy.FULL_PIPELINE_PARALLEL, MultiGPUStrategy.MULTI_GPU, MultiGPUStrategy.DATA_PARALLEL):
            if num_devices >= 2:
                return self.run_until_threshold_data_parallel(
                    state, problem, target_fitness, max_generations, collect_history
                )
            else:
                # Fallback to single GPU if only 1 device
                return super().run_until_threshold(
                    state, problem, target_fitness, max_generations, collect_history
                )

        # Route POPULATION_PARALLEL_SEQUENTIAL to Population-Parallel implementation
        if self.strategy == MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL:
            if num_devices >= 2:
                return self.run_until_threshold_population_parallel(
                    state, problem, target_fitness, max_generations, collect_history
                )
            else:
                return super().run_until_threshold(
                    state, problem, target_fitness, max_generations, collect_history
                )

        # POPULATION_PARALLEL (true parallel) - not implemented in this release
        if self.strategy == MultiGPUStrategy.POPULATION_PARALLEL:
            raise NotImplementedError(
                "POPULATION_PARALLEL (true parallel h→h) is not yet implemented. "
                "Use POPULATION_PARALLEL_SEQUENTIAL instead, which processes h→h modes "
                "sequentially to avoid JAX JIT cache cross-device errors. "
                "See benchmarks/results/multi_gpu_vs_single_gpu_comparison.md for details."
            )

        # Route STREAMING to Streaming implementation
        if self.strategy == MultiGPUStrategy.STREAMING:
            return self.run_until_threshold_streaming(
                state, problem, target_fitness, max_generations,
                data_chunk_size=streaming_chunk_size, collect_history=collect_history
            )

        # Route EVAL_ONLY_PARALLEL and alias PMAP_PARALLEL
        # NOTE: This strategy is implemented in the unified_extended_dynamic_functions_full.py file
        # It requires the unified implementation's h→h caching infrastructure.
        if self.strategy in (MultiGPUStrategy.EVAL_ONLY_PARALLEL, MultiGPUStrategy.PMAP_PARALLEL):
            # This strategy requires the _run_until_threshold_pmap method which is defined
            # in HMRHyperNEATUnifiedExtendedDynamicFunctions. Fall through to super().
            pass  # Will be handled by parent class if it has the method

        # Legacy strategies: POSITION_SHARDING, POSITION_SHARDING_CHUNKED, PIPELINE_CHUNKED
        use_legacy_multi_gpu = self.strategy in (
            MultiGPUStrategy.POSITION_SHARDING,
            MultiGPUStrategy.POSITION_SHARDING_CHUNKED,
            MultiGPUStrategy.PIPELINE_CHUNKED,
        )
        if not use_legacy_multi_gpu or self._position_config is None:
            return super().run_until_threshold(
                state, problem, target_fitness, max_generations, collect_history
            )

        # POSITION_SHARDING or POSITION_SHARDING_CHUNKED: Use multi-GPU generation step
        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates as JAX arrays
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # Convert target to JAX array for GPU-side comparison
        target_fitness_jax = jnp.array(target_fitness, dtype=jnp.float32)
        max_gens_jax = jnp.array(max_generations, dtype=jnp.int32)

        # Get initial transformed CPPNs (will be recomputed in loop)
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Capture settings for use in loop body
        use_extra_split = self.extra_randkey_split
        position_config = self._position_config

        # For POSITION_SHARDING_CHUNKED or PIPELINE_CHUNKED, get the effective chunk size
        use_chunked = self.strategy == MultiGPUStrategy.POSITION_SHARDING_CHUNKED
        use_pipeline_chunked = self.strategy == MultiGPUStrategy.PIPELINE_CHUNKED
        if use_chunked or use_pipeline_chunked:
            pop_size = cppn_population[0].shape[0]
            positions = h_grid.total_positions
            n_samples = inputs_batch.shape[0]  # Dataset size for memory constraint
            pop_chunk_size = self._get_effective_chunk_size(positions, pop_size, n_samples=n_samples)
        else:
            pop_chunk_size = None

        if collect_history:
            # Version with history collection
            if use_pipeline_chunked:
                # PIPELINE_CHUNKED: Use full pipeline chunking (for depth 7+)
                def loop_body(carry):
                    generation, best_so_far, current_state, history = carry

                    new_state, fitnesses = self._pure_generation_step_pipeline_chunked(
                        current_state, cppns_transformed, h_grid,
                        input_coords, output_coords,
                        inputs_batch, targets_batch,
                        position_config, pop_chunk_size,
                        extra_randkey_split=use_extra_split
                    )

                    gen_best = jnp.max(fitnesses)
                    best_so_far = jnp.maximum(best_so_far, gen_best)
                    history = history.at[generation].set(gen_best)

                    return (generation + 1, best_so_far, new_state, history)
            elif use_chunked:
                # POSITION_SHARDING_CHUNKED: Use chunked generation step
                def loop_body(carry):
                    generation, best_so_far, current_state, history = carry

                    new_state, fitnesses = self._pure_generation_step_multi_gpu_chunked(
                        current_state, cppns_transformed, h_grid,
                        input_coords, output_coords,
                        inputs_batch, targets_batch,
                        position_config, pop_chunk_size,
                        extra_randkey_split=use_extra_split
                    )

                    gen_best = jnp.max(fitnesses)
                    best_so_far = jnp.maximum(best_so_far, gen_best)
                    history = history.at[generation].set(gen_best)

                    return (generation + 1, best_so_far, new_state, history)
            else:
                # POSITION_SHARDING: Use standard multi-GPU generation step
                def loop_body(carry):
                    generation, best_so_far, current_state, history = carry

                    new_state, fitnesses = self._pure_generation_step_multi_gpu(
                        current_state, cppns_transformed, h_grid,
                        input_coords, output_coords,
                        inputs_batch, targets_batch,
                        position_config,
                        extra_randkey_split=use_extra_split
                    )

                    gen_best = jnp.max(fitnesses)
                    best_so_far = jnp.maximum(best_so_far, gen_best)
                    history = history.at[generation].set(gen_best)

                    return (generation + 1, best_so_far, new_state, history)

            def loop_condition(carry):
                generation, best_so_far, current_state, history = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            history = jnp.zeros(max_generations, dtype=jnp.float32)
            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state,
                history
            )

            final_gen, final_best, final_state, final_history = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            final_gen_py = int(jax.device_get(final_gen))
            return {
                'generations': final_gen_py,
                'best_fitness': float(jax.device_get(final_best)),
                'state': final_state,
                'history': jax.device_get(final_history[:final_gen_py]),
            }

        else:
            # Version without history (minimal memory)
            if use_pipeline_chunked:
                # PIPELINE_CHUNKED: Use full pipeline chunking (for depth 7+)
                def loop_body(carry):
                    generation, best_so_far, current_state = carry

                    new_state, fitnesses = self._pure_generation_step_pipeline_chunked(
                        current_state, cppns_transformed, h_grid,
                        input_coords, output_coords,
                        inputs_batch, targets_batch,
                        position_config, pop_chunk_size,
                        extra_randkey_split=use_extra_split
                    )

                    gen_best = jnp.max(fitnesses)
                    best_so_far = jnp.maximum(best_so_far, gen_best)

                    return (generation + 1, best_so_far, new_state)
            elif use_chunked:
                # POSITION_SHARDING_CHUNKED: Use chunked generation step
                def loop_body(carry):
                    generation, best_so_far, current_state = carry

                    new_state, fitnesses = self._pure_generation_step_multi_gpu_chunked(
                        current_state, cppns_transformed, h_grid,
                        input_coords, output_coords,
                        inputs_batch, targets_batch,
                        position_config, pop_chunk_size,
                        extra_randkey_split=use_extra_split
                    )

                    gen_best = jnp.max(fitnesses)
                    best_so_far = jnp.maximum(best_so_far, gen_best)

                    return (generation + 1, best_so_far, new_state)
            else:
                # POSITION_SHARDING: Use standard multi-GPU generation step
                def loop_body(carry):
                    generation, best_so_far, current_state = carry

                    new_state, fitnesses = self._pure_generation_step_multi_gpu(
                        current_state, cppns_transformed, h_grid,
                        input_coords, output_coords,
                        inputs_batch, targets_batch,
                        position_config,
                        extra_randkey_split=use_extra_split
                    )

                    gen_best = jnp.max(fitnesses)
                    best_so_far = jnp.maximum(best_so_far, gen_best)

                    return (generation + 1, best_so_far, new_state)

            def loop_condition(carry):
                generation, best_so_far, current_state = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state
            )

            final_gen, final_best, final_state = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            final_gen_py = int(jax.device_get(final_gen))
            return {
                'generations': final_gen_py,
                'best_fitness': float(jax.device_get(final_best)),
                'state': final_state,
            }

    def run_until_threshold_dual_gpu_eval(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run generations with TRUE DUAL-GPU evaluation using Python loop + pmap.

        Unlike run_until_threshold (which uses jax.lax.while_loop and single-GPU eval),
        this method uses a Python while loop to enable pmap-based multi-GPU evaluation.

        For large datasets (75K+ samples), evaluation dominates runtime. This method
        distributes evaluation across both GPUs for ~2x speedup on that bottleneck.

        Trade-off:
        - Python loop has slightly more overhead than jax.lax.while_loop
        - But pmap evaluation gives ~2x speedup for large datasets
        - Net win when evaluation >> control flow overhead

        Args:
            state: Initialized algorithm state
            problem: Problem instance with get_data()
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            collect_history: If True, collect per-generation fitness history
            verbose: If True, print per-generation progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state', and optionally 'history'
        """
        num_devices = len(jax.devices())
        if num_devices < 2:
            if verbose:
                print("WARNING: Only 1 GPU detected, falling back to standard method")
            return self.run_until_threshold(
                state, problem, target_fitness, max_generations, collect_history
            )

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get initial population info
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        positions = h_grid.total_positions

        # Calculate chunk size with dataset-aware memory constraint
        pop_chunk_size = self._get_effective_chunk_size(positions, pop_size, n_samples=n_samples)

        if verbose:
            print(f"Dual-GPU eval: {num_devices} GPUs, pop={pop_size}, "
                  f"positions={positions}, n_samples={n_samples}, chunk={pop_chunk_size}")

        # Settings
        position_config = self._position_config
        use_extra_split = self.extra_randkey_split
        all_positions = h_grid.all_positions

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter() if verbose else None

            # STEP 0: CPPN ask + transform
            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Replicate CPPNs for shard_map CPPN queries
            replicated_sharding = NamedSharding(position_config.mesh, P())
            cppns_transformed = tuple(
                jax.device_put(arr, replicated_sharding) for arr in cppns_transformed
            )

            # STEP 1: Query CPPN for variance (multi-GPU position sharding)
            source_coord = input_coords[0:1]
            all_weights_for_variance = batch_query_population_multi_source_chunked_sharded(
                state, cppns_transformed, source_coord,
                all_positions, True, self._jitted_cppn_forward,
                position_config, pop_chunk_size
            )
            all_weights_for_variance = all_weights_for_variance[:, 0, :]

            # STEP 2: Compute hierarchical variances and masks
            level_variances = compute_hierarchical_variances_batch_jit(
                all_weights_for_variance,
                level_sizes=h_grid.level_sizes_static,
                level_offsets=h_grid.level_offsets_static,
                level_grid_sizes=h_grid.level_grid_sizes_static,
                num_levels=h_grid.num_levels,
            )
            masks_A, _, _ = compute_subdivision_masks_batch_jit(
                level_variances,
                variance_threshold=self.variance_threshold,
                parent_indices_tuple=h_grid.parent_indices,
                level_offsets=h_grid.level_offsets_static,
                num_levels=h_grid.num_levels,
                total_positions=h_grid.total_positions,
            )

            # STEP 3: Query weights (multi-GPU position sharding)
            input_all_weights = batch_query_population_multi_source_chunked_sharded(
                state, cppns_transformed, input_coords,
                all_positions, True, self._jitted_cppn_forward,
                position_config, pop_chunk_size
            )
            output_all_weights = batch_query_population_multi_source_chunked_sharded(
                state, cppns_transformed, output_coords,
                all_positions, False, self._jitted_cppn_forward,
                position_config, pop_chunk_size
            )

            # STEP 4+5: Build and evaluate with DUAL-GPU pmap
            # This is the KEY difference - uses pmap to split population across GPUs
            fitnesses = build_and_evaluate_chunked_multi_gpu(
                input_all_weights,
                output_all_weights,
                masks_A,
                inputs_batch,
                targets_batch,
                pop_chunk_size=pop_chunk_size,
                max_weight=self.max_weight,
                weight_thresh=0.1,
                num_gpus=num_devices,
            )

            # STEP 6: NEAT evolution
            if use_extra_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            state = self._compiled_tell(state, fitnesses)

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall_best={best_so_far:.4f}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def run_until_threshold_population_parallel(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run generations with OPTIMIZED population-parallel multi-GPU execution.

        This is the MOST EFFICIENT multi-GPU approach:
        - Population is split ONCE at the start
        - Each GPU runs the ENTIRE pipeline (CPPN + variance + masks + weights + eval)
        - NO data reshuffling between stages
        - Results gathered only for NEAT selection

        Compared to run_until_threshold_dual_gpu_eval:
        - OLD: position-shard for CPPN, population-shard for eval → data reshuffling
        - NEW: population-shard for EVERYTHING → no reshuffling, ~2x speedup

        Args:
            state: Initialized algorithm state
            problem: Problem instance with get_data()
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            collect_history: If True, collect per-generation fitness history
            verbose: If True, print per-generation progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state', and optionally 'history'
        """
        num_devices = len(jax.devices())
        if num_devices < 2:
            if verbose:
                print("WARNING: Only 1 GPU detected, falling back to standard method")
            return self.run_until_threshold(
                state, problem, target_fitness, max_generations, collect_history
            )

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get grid info for variance/mask computation
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels
        total_positions = h_grid.total_positions

        # Compute memory-aware chunk size for large datasets
        # Memory per chunk = chunk_size × n_samples × positions × 4 bytes (float32)
        # This is because vmap creates arrays of shape (chunk_size, n_samples, positions)
        # Target: 2GB per GPU to leave room for other allocations (CPPN weights, etc)
        TARGET_MEMORY_GB = 2.0
        bytes_per_element = 4
        memory_per_network = n_samples * total_positions * bytes_per_element

        if memory_per_network > 0:
            max_chunk_for_memory = max(1, int(
                (TARGET_MEMORY_GB * 1e9) / memory_per_network
            ))
        else:
            max_chunk_for_memory = None  # No chunking needed

        # Chunk size will be computed on first iteration (need pop_size from state)
        pop_chunk_size = None  # Will be set on first iteration
        chunk_size_computed = False

        # Settings
        use_extra_split = self.extra_randkey_split

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        # Python while loop for control flow (required for pmap)
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter() if verbose else None

            # STEP 0: CPPN ask + transform
            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Compute chunk size on first iteration (need pop_size from cppn_population)
            if not chunk_size_computed:
                pop_size = cppn_population[0].shape[0]
                per_gpu_pop = (pop_size + num_devices - 1) // num_devices

                # Use chunking if processing all at once would exceed memory limit
                if max_chunk_for_memory is not None and per_gpu_pop > max_chunk_for_memory:
                    pop_chunk_size = max_chunk_for_memory
                else:
                    pop_chunk_size = None  # No chunking needed

                if verbose:
                    print(f"Population-parallel: {num_devices} GPUs, "
                          f"positions={total_positions}, n_samples={n_samples}")
                    if pop_chunk_size is not None:
                        print(f"  Memory-aware chunking: {pop_chunk_size} individuals per chunk")

                chunk_size_computed = True

            # Run FULL pipeline with population-parallel execution
            fitnesses = full_pipeline_population_parallel(
                cppns_transformed,
                all_positions,
                input_coords,
                output_coords,
                inputs_batch,
                targets_batch,
                level_sizes,
                level_offsets,
                level_grid_sizes,
                parent_indices,
                num_levels,
                total_positions,
                self.variance_threshold,
                self.max_weight,
                0.1,  # weight_thresh
                self._jitted_cppn_forward,
                state,
                num_gpus=num_devices,
                pop_chunk_size=pop_chunk_size,  # Memory-aware chunking
            )

            # NEAT evolution step
            if use_extra_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            state = self._compiled_tell(state, fitnesses)

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall_best={best_so_far:.4f}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def _setup_population_parallel(self, state, problem, verbose=False):
        """Common setup for population-parallel methods. Returns setup dict."""
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get grid info
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels
        total_positions = h_grid.total_positions

        # Compute memory-aware chunk size
        TARGET_MEMORY_GB = 2.0
        memory_per_network = n_samples * total_positions * 4

        if memory_per_network > 0:
            max_chunk_for_memory = max(1, int((TARGET_MEMORY_GB * 1e9) / memory_per_network))
        else:
            max_chunk_for_memory = None

        return {
            'num_devices': num_devices,
            'h_grid': h_grid,
            'input_coords': input_coords,
            'output_coords': output_coords,
            'all_positions': all_positions,
            'inputs_batch': inputs_batch,
            'targets_batch': targets_batch,
            'n_samples': n_samples,
            'level_sizes': level_sizes,
            'level_offsets': level_offsets,
            'level_grid_sizes': level_grid_sizes,
            'parent_indices': parent_indices,
            'num_levels': num_levels,
            'total_positions': total_positions,
            'max_chunk_for_memory': max_chunk_for_memory,
            'use_extra_split': self.extra_randkey_split,
        }

    def _run_one_generation_population_parallel(self, state, setup, pop_chunk_size):
        """Run one generation with population-parallel execution. Returns (state, fitnesses)."""
        # CPPN ask + transform
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Run FULL pipeline with population-parallel execution
        fitnesses = full_pipeline_population_parallel(
            cppns_transformed,
            setup['all_positions'],
            setup['input_coords'],
            setup['output_coords'],
            setup['inputs_batch'],
            setup['targets_batch'],
            setup['level_sizes'],
            setup['level_offsets'],
            setup['level_grid_sizes'],
            setup['parent_indices'],
            setup['num_levels'],
            setup['total_positions'],
            self.variance_threshold,
            self.max_weight,
            0.1,  # weight_thresh
            self._jitted_cppn_forward,
            state,
            num_gpus=setup['num_devices'],
            pop_chunk_size=pop_chunk_size,
        )

        # NEAT evolution step
        if setup['use_extra_split']:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        state = self._compiled_tell(state, fitnesses)
        return state, fitnesses

    # =========================================================================
    # Option 1: Fixed Generations (Zero Sync Until End)
    # =========================================================================

    def run_fixed_generations_population_parallel(
        self,
        state: Any,
        problem: Any,
        num_generations: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run fixed number of generations with ZERO per-generation sync.

        This is the fastest approach - no CPU-GPU synchronization until the
        very end. Use when you don't need early stopping.

        Args:
            state: Initialized algorithm state
            problem: Problem instance
            num_generations: Exact number of generations to run
            verbose: Print progress (only at start/end to avoid sync)

        Returns:
            Dict with 'generations', 'best_fitness', 'state'
        """
        num_devices = len(jax.devices())
        if num_devices < 2:
            if verbose:
                print("WARNING: Only 1 GPU detected")

        setup = self._setup_population_parallel(state, problem, verbose)

        # Compute chunk size from first ask
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        per_gpu_pop = (pop_size + num_devices - 1) // num_devices

        if setup['max_chunk_for_memory'] is not None and per_gpu_pop > setup['max_chunk_for_memory']:
            pop_chunk_size = setup['max_chunk_for_memory']
        else:
            pop_chunk_size = None

        if verbose:
            print(f"Fixed generations: {num_generations} gens, {num_devices} GPUs, "
                  f"positions={setup['total_positions']}, n_samples={setup['n_samples']}")
            if pop_chunk_size:
                print(f"  Memory-aware chunking: {pop_chunk_size} individuals per chunk")
            print("  Running (no sync until end)...")

        # Run all generations WITHOUT any sync
        start_time = time.perf_counter()
        for gen in range(num_generations):
            state, fitnesses = self._run_one_generation_population_parallel(
                state, setup, pop_chunk_size
            )
            # NO SYNC HERE - just keep going

        # Single sync at the very end
        final_best = float(jnp.max(fitnesses))
        total_time = time.perf_counter() - start_time

        if verbose:
            print(f"  Completed {num_generations} gens in {total_time:.2f}s "
                  f"({total_time/num_generations*1000:.1f}ms/gen), best={final_best:.4f}")

        return {
            'generations': num_generations,
            'best_fitness': final_best,
            'state': state,
        }

    # =========================================================================
    # Option 2: Periodic Sync (Sync Every K Generations)
    # =========================================================================

    def run_until_threshold_periodic_sync(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        sync_interval: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run with periodic sync every K generations.

        Reduces sync overhead by factor of K while maintaining early stopping.
        May overshoot target by up to K-1 generations.

        Args:
            state: Initialized algorithm state
            problem: Problem instance
            target_fitness: Stop when best >= target
            max_generations: Maximum generations
            sync_interval: Generations between syncs (default 10)
            verbose: Print progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state'
        """
        num_devices = len(jax.devices())
        if num_devices < 2:
            if verbose:
                print("WARNING: Only 1 GPU detected")

        setup = self._setup_population_parallel(state, problem, verbose)

        # Compute chunk size from first ask
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        per_gpu_pop = (pop_size + num_devices - 1) // num_devices

        if setup['max_chunk_for_memory'] is not None and per_gpu_pop > setup['max_chunk_for_memory']:
            pop_chunk_size = setup['max_chunk_for_memory']
        else:
            pop_chunk_size = None

        if verbose:
            print(f"Periodic sync: {num_devices} GPUs, sync every {sync_interval} gens, "
                  f"positions={setup['total_positions']}, n_samples={setup['n_samples']}")
            if pop_chunk_size:
                print(f"  Memory-aware chunking: {pop_chunk_size} individuals per chunk")

        generation = 0
        best_so_far = -float('inf')
        start_time = time.perf_counter()

        while best_so_far < target_fitness and generation < max_generations:
            batch_start = time.perf_counter()

            # Run sync_interval generations without checking
            gens_this_batch = min(sync_interval, max_generations - generation)
            for _ in range(gens_this_batch):
                state, fitnesses = self._run_one_generation_population_parallel(
                    state, setup, pop_chunk_size
                )
                generation += 1

            # Sync once per batch
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            if verbose:
                batch_time = time.perf_counter() - batch_start
                print(f"  Gens {generation-gens_this_batch+1}-{generation}: "
                      f"best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={batch_time:.2f}s ({batch_time/gens_this_batch*1000:.1f}ms/gen)")

        total_time = time.perf_counter() - start_time
        if verbose:
            print(f"  Total: {generation} gens in {total_time:.2f}s")

        return {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }

    # =========================================================================
    # Option 3: Overlapped Sync (Queue Next Gen While Syncing)
    # =========================================================================

    def run_until_threshold_overlapped(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run with overlapped execution - queue next gen while syncing.

        GPU stays busy during CPU sync by queuing the next generation's
        work before waiting for the current sync to complete.

        Args:
            state: Initialized algorithm state
            problem: Problem instance
            target_fitness: Stop when best >= target
            max_generations: Maximum generations
            verbose: Print progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state'
        """
        num_devices = len(jax.devices())
        if num_devices < 2:
            if verbose:
                print("WARNING: Only 1 GPU detected")

        setup = self._setup_population_parallel(state, problem, verbose)

        # Compute chunk size from first ask
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        per_gpu_pop = (pop_size + num_devices - 1) // num_devices

        if setup['max_chunk_for_memory'] is not None and per_gpu_pop > setup['max_chunk_for_memory']:
            pop_chunk_size = setup['max_chunk_for_memory']
        else:
            pop_chunk_size = None

        if verbose:
            print(f"Overlapped sync: {num_devices} GPUs, "
                  f"positions={setup['total_positions']}, n_samples={setup['n_samples']}")
            if pop_chunk_size:
                print(f"  Memory-aware chunking: {pop_chunk_size} individuals per chunk")

        best_so_far = -float('inf')
        start_time = time.perf_counter()

        # First generation
        state, fitnesses = self._run_one_generation_population_parallel(
            state, setup, pop_chunk_size
        )
        generation = 1

        while generation < max_generations:
            gen_start = time.perf_counter()

            # Queue NEXT generation's work BEFORE syncing current
            next_state, next_fitnesses = self._run_one_generation_population_parallel(
                state, setup, pop_chunk_size
            )

            # NOW sync previous (GPU busy with next gen)
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            if verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={gen_time*1000:.1f}ms")

            if best_so_far >= target_fitness:
                break

            # Move to next
            fitnesses = next_fitnesses
            state = next_state
            generation += 1

        # Final sync for last generation
        gen_best = float(jnp.max(fitnesses))
        best_so_far = max(best_so_far, gen_best)

        total_time = time.perf_counter() - start_time
        if verbose:
            print(f"  Total: {generation} gens in {total_time:.2f}s")

        return {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }

    # =========================================================================
    # Option 4: Combined (Periodic + Overlapped)
    # =========================================================================

    def run_until_threshold_periodic_overlapped(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        sync_interval: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run with periodic sync + overlapped execution.

        Best of both: minimal syncs (every K gens) + GPU pipelining.

        Args:
            state: Initialized algorithm state
            problem: Problem instance
            target_fitness: Stop when best >= target
            max_generations: Maximum generations
            sync_interval: Generations between syncs (default 10)
            verbose: Print progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state'
        """
        num_devices = len(jax.devices())
        if num_devices < 2:
            if verbose:
                print("WARNING: Only 1 GPU detected")

        setup = self._setup_population_parallel(state, problem, verbose)

        # Compute chunk size from first ask
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        per_gpu_pop = (pop_size + num_devices - 1) // num_devices

        if setup['max_chunk_for_memory'] is not None and per_gpu_pop > setup['max_chunk_for_memory']:
            pop_chunk_size = setup['max_chunk_for_memory']
        else:
            pop_chunk_size = None

        if verbose:
            print(f"Periodic+Overlapped: {num_devices} GPUs, sync every {sync_interval} gens, "
                  f"positions={setup['total_positions']}, n_samples={setup['n_samples']}")
            if pop_chunk_size:
                print(f"  Memory-aware chunking: {pop_chunk_size} individuals per chunk")

        generation = 0
        best_so_far = -float('inf')
        start_time = time.perf_counter()

        while best_so_far < target_fitness and generation < max_generations:
            batch_start = time.perf_counter()

            # Run sync_interval generations without checking
            gens_this_batch = min(sync_interval, max_generations - generation)
            for _ in range(gens_this_batch):
                state, fitnesses = self._run_one_generation_population_parallel(
                    state, setup, pop_chunk_size
                )
                generation += 1

            # Start queuing NEXT batch's first gen while syncing current
            if generation < max_generations and best_so_far < target_fitness:
                # Queue next batch's first generation (GPU starts working)
                next_cppn = self._compiled_ask(state)
                next_cppn_t = self._compiled_transform_batch(state, next_cppn)
                # Don't wait - let GPU start while we sync

            # NOW sync (GPU busy with next batch's ask/transform)
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            if verbose:
                batch_time = time.perf_counter() - batch_start
                print(f"  Gens {generation-gens_this_batch+1}-{generation}: "
                      f"best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={batch_time:.2f}s ({batch_time/gens_this_batch*1000:.1f}ms/gen)")

        total_time = time.perf_counter() - start_time
        if verbose:
            print(f"  Total: {generation} gens in {total_time:.2f}s")

        return {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }

    # =========================================================================
    # GPU-RESIDENT Multi-GPU Evolution (True lax.while_loop)
    # =========================================================================

    def run_until_threshold_gpu_resident_multi_gpu(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Run generations with TRUE GPU-resident multi-GPU execution.

        Uses jax.lax.while_loop to run ALL generations on GPU with
        GPU-side threshold checking. Only ONE GPU→CPU sync at the end.

        This is the FASTEST approach for multi-GPU evolution:
        - Population is split across GPUs via pmap
        - Entire evolution loop stays on GPU (no Python round-trips)
        - Threshold checking happens on GPU
        - Single sync at the very end to retrieve results

        Compared to run_until_threshold_population_parallel:
        - OLD: Python while loop → ~20 round trips through interpreter
        - NEW: lax.while_loop → 0 Python round trips during execution

        Args:
            state: Initialized algorithm state
            problem: Problem instance with get_data()
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            collect_history: If True, collect per-generation fitness history

        Returns:
            Dict with 'generations', 'best_fitness', 'state', and optionally 'history'

        Note:
            This method does NOT support memory-aware chunking. For large
            datasets (>20K samples), use run_until_threshold_population_parallel
            which supports chunking but uses Python loops.
        """
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # Get grid info for variance/mask computation
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels
        total_positions = h_grid.total_positions

        # Convert thresholds to JAX arrays for GPU-side comparison
        target_fitness_jax = jnp.array(target_fitness, dtype=jnp.float32)
        max_gens_jax = jnp.array(max_generations, dtype=jnp.int32)

        # Capture settings for use in loop body
        use_extra_split = self.extra_randkey_split
        variance_threshold = self.variance_threshold
        max_weight = self.max_weight
        weight_thresh = 0.1

        # Pre-create pmap function ONCE (not inside loop)
        # Using non-chunked version since we can't dynamically choose inside while_loop
        @jax.pmap
        def run_pipeline_on_gpu(cppns_slice):
            return _full_pipeline_single_gpu(
                cppns_slice,
                all_positions,
                input_coords,
                output_coords,
                inputs_batch,
                targets_batch,
                level_sizes,
                level_offsets,
                level_grid_sizes,
                parent_indices,
                num_levels,
                total_positions,
                variance_threshold,
                max_weight,
                weight_thresh,
                self._jitted_cppn_forward,
                state,
            )

        # Get population size from first ask
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]

        # Compute padding for even GPU split
        remainder = pop_size % num_devices
        if remainder != 0:
            pad_size = num_devices - remainder
            padded_pop_size = pop_size + pad_size
        else:
            pad_size = 0
            padded_pop_size = pop_size

        per_gpu_pop = padded_pop_size // num_devices

        def shard_cppns(cppns_transformed):
            """Pad and reshape CPPNs for pmap distribution."""
            if pad_size > 0:
                cppns_padded = tuple(
                    jnp.pad(arr, ((0, pad_size),) + ((0, 0),) * (arr.ndim - 1), mode='constant')
                    for arr in cppns_transformed
                )
            else:
                cppns_padded = cppns_transformed

            # Reshape: (pop_size, ...) -> (num_gpus, per_gpu_pop, ...)
            return tuple(
                arr.reshape(num_devices, per_gpu_pop, *arr.shape[1:])
                for arr in cppns_padded
            )

        def gather_fitnesses(sharded_fitnesses):
            """Gather fitnesses from GPUs and remove padding."""
            # sharded_fitnesses: (num_gpus, per_gpu_pop)
            all_fitnesses = sharded_fitnesses.reshape(-1)  # (padded_pop_size,)
            if pad_size > 0:
                return all_fitnesses[:pop_size]
            return all_fitnesses

        if collect_history:
            # Version with history collection
            def loop_body(carry):
                generation, best_so_far, current_state, history = carry

                # Optional pre-tell random key split
                if use_extra_split:
                    randkey_, randkey = jax.random.split(current_state.randkey)
                    current_state = current_state.update(randkey=randkey)

                # CPPN ask + transform
                cppn_pop = self._compiled_ask(current_state)
                cppns_transformed = self._compiled_transform_batch(current_state, cppn_pop)

                # Shard population across GPUs
                cppns_sharded = shard_cppns(cppns_transformed)

                # Run pipeline on all GPUs (pmap)
                sharded_fitnesses = run_pipeline_on_gpu(cppns_sharded)

                # Gather results
                fitnesses = gather_fitnesses(sharded_fitnesses)

                # NEAT evolution step
                new_state = self._compiled_tell(current_state, fitnesses)

                # Update tracking (GPU-side operations)
                gen_best = jnp.max(fitnesses)
                best_so_far = jnp.maximum(best_so_far, gen_best)

                # Store in pre-allocated array
                history = history.at[generation].set(gen_best)

                return (generation + 1, best_so_far, new_state, history)

            def loop_condition(carry):
                generation, best_so_far, current_state, history = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            # Pre-allocate history array
            history = jnp.zeros(max_generations, dtype=jnp.float32)
            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state,
                history
            )

            final_gen, final_best, final_state, final_history = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            # SINGLE GPU→CPU sync at the very end
            final_gen_py = int(jax.device_get(final_gen))
            return {
                'generations': final_gen_py,
                'best_fitness': float(jax.device_get(final_best)),
                'state': final_state,
                'history': jax.device_get(final_history[:final_gen_py]),
            }

        else:
            # Version without history (minimal memory)
            def loop_body(carry):
                generation, best_so_far, current_state = carry

                # Optional pre-tell random key split
                if use_extra_split:
                    randkey_, randkey = jax.random.split(current_state.randkey)
                    current_state = current_state.update(randkey=randkey)

                # CPPN ask + transform
                cppn_pop = self._compiled_ask(current_state)
                cppns_transformed = self._compiled_transform_batch(current_state, cppn_pop)

                # Shard population across GPUs
                cppns_sharded = shard_cppns(cppns_transformed)

                # Run pipeline on all GPUs (pmap)
                sharded_fitnesses = run_pipeline_on_gpu(cppns_sharded)

                # Gather results
                fitnesses = gather_fitnesses(sharded_fitnesses)

                # NEAT evolution step
                new_state = self._compiled_tell(current_state, fitnesses)

                # Update tracking (GPU-side operations)
                gen_best = jnp.max(fitnesses)
                best_so_far = jnp.maximum(best_so_far, gen_best)

                return (generation + 1, best_so_far, new_state)

            def loop_condition(carry):
                generation, best_so_far, current_state = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state
            )

            final_gen, final_best, final_state = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            # SINGLE GPU→CPU sync - pack scalars and transfer together
            results_packed = jnp.stack([final_gen.astype(jnp.float32), final_best])
            results_cpu = jax.device_get(results_packed)
            return {
                'generations': int(results_cpu[0]),
                'best_fitness': float(results_cpu[1]),
                'state': final_state,
            }

    # =========================================================================
    # GPU-RESIDENT Multi-GPU with SHARD_MAP (Supports chunking + while_loop)
    # =========================================================================

    def run_until_threshold_gpu_resident_shardmap(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        pop_chunk_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run generations with TRUE GPU-resident multi-GPU using shard_map.

        Uses jax.lax.while_loop + shard_map for:
        - TRUE multi-GPU execution (shard_map preserves sharding inside JIT)
        - GPU-resident loop (no Python round-trips)
        - Dataset-aware chunking (optional, for large datasets)

        This is the ULTIMATE approach combining:
        - Multi-GPU: shard_map distributes population across GPUs
        - GPU-resident: lax.while_loop keeps entire loop on GPU
        - Chunking: lax.fori_loop for memory-aware processing

        Args:
            state: Initialized algorithm state
            problem: Problem instance with get_data()
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            collect_history: If True, collect per-generation fitness history
            pop_chunk_size: If provided, use chunked processing to avoid OOM
                           with large datasets. If None, auto-compute based on memory.

        Returns:
            Dict with 'generations', 'best_fitness', 'state', and optionally 'history'
        """
        if not SHARD_MAP_AVAILABLE:
            raise RuntimeError("shard_map not available in this JAX version")

        num_devices = len(jax.devices())

        # Create mesh for population sharding
        device_mesh = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(device_mesh, axis_names=('gpus',))

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get grid info for variance/mask computation
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels
        total_positions = h_grid.total_positions

        # Compute memory-aware chunk size if not provided
        if pop_chunk_size is None:
            TARGET_MEMORY_GB = 2.0
            memory_per_network = n_samples * total_positions * 4
            if memory_per_network > 0:
                max_chunk_for_memory = max(1, int((TARGET_MEMORY_GB * 1e9) / memory_per_network))
            else:
                max_chunk_for_memory = None
        else:
            max_chunk_for_memory = pop_chunk_size

        # Convert thresholds to JAX arrays for GPU-side comparison
        target_fitness_jax = jnp.array(target_fitness, dtype=jnp.float32)
        max_gens_jax = jnp.array(max_generations, dtype=jnp.int32)

        # Capture settings for use in loop body
        use_extra_split = self.extra_randkey_split
        variance_threshold = self.variance_threshold
        max_weight = self.max_weight
        weight_thresh = 0.1

        # Get population size from first ask
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]

        # Compute padding for even GPU split
        remainder = pop_size % num_devices
        if remainder != 0:
            pad_size = num_devices - remainder
            padded_pop_size = pop_size + pad_size
        else:
            pad_size = 0
            padded_pop_size = pop_size

        per_gpu_pop = padded_pop_size // num_devices

        # Determine if chunking is needed
        if max_chunk_for_memory is not None and per_gpu_pop > max_chunk_for_memory:
            use_chunking = True
            chunk_size = max_chunk_for_memory
        else:
            use_chunking = False
            chunk_size = None

        # Pre-create shard_map function ONCE (not inside loop)
        if use_chunking:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P('gpus'), P('gpus'), P('gpus'), P('gpus')),
                     out_specs=P('gpus'),
                     check_rep=False)
            def run_pipeline_shardmap(nodes_slice, conns_slice, acts_slice, extra_slice):
                cppns_slice = (nodes_slice, conns_slice, acts_slice, extra_slice)
                return _full_pipeline_single_gpu_chunked(
                    cppns_slice,
                    all_positions,
                    input_coords,
                    output_coords,
                    inputs_batch,
                    targets_batch,
                    level_sizes,
                    level_offsets,
                    level_grid_sizes,
                    parent_indices,
                    num_levels,
                    total_positions,
                    variance_threshold,
                    max_weight,
                    weight_thresh,
                    self._jitted_cppn_forward,
                    state,
                    pop_chunk_size=chunk_size,
                )
        else:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P('gpus'), P('gpus'), P('gpus'), P('gpus')),
                     out_specs=P('gpus'),
                     check_rep=False)
            def run_pipeline_shardmap(nodes_slice, conns_slice, acts_slice, extra_slice):
                cppns_slice = (nodes_slice, conns_slice, acts_slice, extra_slice)
                return _full_pipeline_single_gpu(
                    cppns_slice,
                    all_positions,
                    input_coords,
                    output_coords,
                    inputs_batch,
                    targets_batch,
                    level_sizes,
                    level_offsets,
                    level_grid_sizes,
                    parent_indices,
                    num_levels,
                    total_positions,
                    variance_threshold,
                    max_weight,
                    weight_thresh,
                    self._jitted_cppn_forward,
                    state,
                )

        def shard_cppns(cppns_transformed):
            """Pad and reshape CPPNs for shard_map distribution."""
            if pad_size > 0:
                cppns_padded = tuple(
                    jnp.pad(arr, ((0, pad_size),) + ((0, 0),) * (arr.ndim - 1), mode='constant')
                    for arr in cppns_transformed
                )
            else:
                cppns_padded = cppns_transformed

            # Reshape: (pop_size, ...) -> (num_gpus, per_gpu_pop, ...)
            return tuple(
                arr.reshape(num_devices, per_gpu_pop, *arr.shape[1:])
                for arr in cppns_padded
            )

        def gather_fitnesses(sharded_fitnesses):
            """Gather fitnesses from GPUs and remove padding."""
            all_fitnesses = sharded_fitnesses.reshape(-1)
            if pad_size > 0:
                return all_fitnesses[:pop_size]
            return all_fitnesses

        if collect_history:
            def loop_body(carry):
                generation, best_so_far, current_state, history = carry

                if use_extra_split:
                    randkey_, randkey = jax.random.split(current_state.randkey)
                    current_state = current_state.update(randkey=randkey)

                cppn_pop = self._compiled_ask(current_state)
                cppns_transformed = self._compiled_transform_batch(current_state, cppn_pop)
                cppns_sharded = shard_cppns(cppns_transformed)

                # Run pipeline with shard_map (preserves sharding inside while_loop)
                with mesh:
                    sharded_fitnesses = run_pipeline_shardmap(
                        cppns_sharded[0], cppns_sharded[1],
                        cppns_sharded[2], cppns_sharded[3]
                    )

                fitnesses = gather_fitnesses(sharded_fitnesses)
                new_state = self._compiled_tell(current_state, fitnesses)

                gen_best = jnp.max(fitnesses)
                best_so_far = jnp.maximum(best_so_far, gen_best)
                history = history.at[generation].set(gen_best)

                return (generation + 1, best_so_far, new_state, history)

            def loop_condition(carry):
                generation, best_so_far, current_state, history = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            history = jnp.zeros(max_generations, dtype=jnp.float32)
            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state,
                history
            )

            final_gen, final_best, final_state, final_history = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            final_gen_py = int(jax.device_get(final_gen))
            return {
                'generations': final_gen_py,
                'best_fitness': float(jax.device_get(final_best)),
                'state': final_state,
                'history': jax.device_get(final_history[:final_gen_py]),
            }

        else:
            def loop_body(carry):
                generation, best_so_far, current_state = carry

                if use_extra_split:
                    randkey_, randkey = jax.random.split(current_state.randkey)
                    current_state = current_state.update(randkey=randkey)

                cppn_pop = self._compiled_ask(current_state)
                cppns_transformed = self._compiled_transform_batch(current_state, cppn_pop)
                cppns_sharded = shard_cppns(cppns_transformed)

                # Run pipeline with shard_map (preserves sharding inside while_loop)
                with mesh:
                    sharded_fitnesses = run_pipeline_shardmap(
                        cppns_sharded[0], cppns_sharded[1],
                        cppns_sharded[2], cppns_sharded[3]
                    )

                fitnesses = gather_fitnesses(sharded_fitnesses)
                new_state = self._compiled_tell(current_state, fitnesses)

                gen_best = jnp.max(fitnesses)
                best_so_far = jnp.maximum(best_so_far, gen_best)

                return (generation + 1, best_so_far, new_state)

            def loop_condition(carry):
                generation, best_so_far, current_state = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state
            )

            final_gen, final_best, final_state = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            results_packed = jnp.stack([final_gen.astype(jnp.float32), final_best])
            results_cpu = jax.device_get(results_packed)
            return {
                'generations': int(results_cpu[0]),
                'best_fitness': float(results_cpu[1]),
                'state': final_state,
            }

    # =========================================================================
    # BATCHED Multi-GPU Evolution (N generations per Python iteration)
    # =========================================================================

    def run_until_threshold_batched_multi_gpu(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        batch_size: int = 10,
        collect_history: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run generations with BATCHED multi-GPU execution.

        Runs N generations per Python iteration to reduce Python overhead while
        maintaining true multi-GPU execution via pmap.

        This approach works around the limitation that pmap inside lax.while_loop
        collapses to single GPU, and shard_map has compatibility issues with
        TensorNEAT's CPPN forward function.

        Key insight: Python overhead is ~X ms per iteration. By running N
        generations per iteration, we reduce overhead by N× while still getting
        true multi-GPU execution.

        Args:
            state: Initialized algorithm state
            problem: Problem instance with get_data()
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            batch_size: Generations per Python iteration (default 10)
            collect_history: If True, collect per-generation fitness history
            verbose: If True, print per-batch progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state', and optionally 'history'
        """
        num_devices = len(jax.devices())
        if num_devices < 2:
            if verbose:
                print("WARNING: Only 1 GPU detected")

        setup = self._setup_population_parallel(state, problem, verbose)

        # Compute chunk size from first ask
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        per_gpu_pop = (pop_size + num_devices - 1) // num_devices

        if setup['max_chunk_for_memory'] is not None and per_gpu_pop > setup['max_chunk_for_memory']:
            pop_chunk_size = setup['max_chunk_for_memory']
        else:
            pop_chunk_size = None

        if verbose:
            print(f"Batched multi-GPU: {num_devices} GPUs, {batch_size} gens/batch, "
                  f"positions={setup['total_positions']}, n_samples={setup['n_samples']}")
            if pop_chunk_size:
                print(f"  Memory-aware chunking: {pop_chunk_size} individuals per chunk")

        generation = 0
        best_so_far = -float('inf')
        history = [] if collect_history else None
        start_time = time.perf_counter()

        while best_so_far < target_fitness and generation < max_generations:
            batch_start = time.perf_counter()

            # Run batch_size generations without checking threshold
            gens_this_batch = min(batch_size, max_generations - generation)
            batch_history = []

            for _ in range(gens_this_batch):
                state, fitnesses = self._run_one_generation_population_parallel(
                    state, setup, pop_chunk_size
                )
                generation += 1

                if collect_history:
                    # GPU-side max, will sync at end of batch
                    batch_history.append(jnp.max(fitnesses))

            # Single sync per batch
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            if collect_history:
                # Sync all history values at once
                for h in batch_history:
                    history.append(float(h))

            if verbose:
                batch_time = time.perf_counter() - batch_start
                print(f"  Gens {generation-gens_this_batch+1}-{generation}: "
                      f"best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={batch_time:.2f}s ({batch_time/gens_this_batch*1000:.1f}ms/gen)")

        total_time = time.perf_counter() - start_time
        if verbose:
            print(f"  Total: {generation} gens in {total_time:.2f}s "
                  f"({total_time/generation*1000:.1f}ms/gen avg)")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def run_until_threshold_data_parallel(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run generations with DATA-PARALLEL multi-GPU execution.

        This strategy shards the DATASET across GPUs (not the population):
        - Dataset is SPLIT across GPUs (each GPU gets different samples)
        - Population is REPLICATED on each GPU (full population on each)
        - Each GPU computes partial fitness on its data shard
        - Results are AVERAGED across GPUs

        Best for: LARGE DATASETS that don't fit on a single GPU

        Memory comparison (2 GPUs, pop=300, data=10K samples):
        - Population-Parallel: Each GPU needs FULL dataset (10K) + half pop (150)
        - Data-Parallel: Each GPU needs HALF dataset (5K) + full pop (300)

        Args:
            state: Initialized algorithm state
            problem: Problem instance with get_data()
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            collect_history: If True, collect per-generation fitness history
            verbose: If True, print per-generation progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state', and optionally 'history'
        """
        num_devices = len(jax.devices())
        if num_devices < 2:
            if verbose:
                print("WARNING: Only 1 GPU detected, falling back to standard method")
            return self.run_until_threshold(
                state, problem, target_fitness, max_generations, collect_history
            )

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get grid info
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels
        total_positions = h_grid.total_positions

        # Compute memory-aware chunk size for population (since full pop on each GPU)
        TARGET_MEMORY_GB = 2.0
        per_gpu_samples = n_samples // num_devices
        memory_per_network = per_gpu_samples * total_positions * 4

        if memory_per_network > 0:
            max_chunk_for_memory = max(1, int((TARGET_MEMORY_GB * 1e9) / memory_per_network))
        else:
            max_chunk_for_memory = None

        # Settings
        use_extra_split = self.extra_randkey_split

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        pop_chunk_size = None
        chunk_size_computed = False

        if verbose:
            print(f"Data-parallel: {num_devices} GPUs, "
                  f"positions={total_positions}, n_samples={n_samples}")
            print(f"  Each GPU gets {n_samples // num_devices} samples, full population")

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter() if verbose else None

            # CPPN ask + transform
            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Compute chunk size on first iteration
            if not chunk_size_computed:
                pop_size = cppn_population[0].shape[0]
                if max_chunk_for_memory is not None and pop_size > max_chunk_for_memory:
                    pop_chunk_size = max_chunk_for_memory
                else:
                    pop_chunk_size = None
                if verbose and pop_chunk_size is not None:
                    print(f"  Memory-aware chunking: {pop_chunk_size} individuals per chunk")
                chunk_size_computed = True

            # Run pipeline with DATA-PARALLEL execution
            fitnesses = full_pipeline_data_parallel(
                cppns_transformed,
                all_positions,
                input_coords,
                output_coords,
                inputs_batch,
                targets_batch,
                level_sizes,
                level_offsets,
                level_grid_sizes,
                parent_indices,
                num_levels,
                total_positions,
                self.variance_threshold,
                self.max_weight,
                0.1,  # weight_thresh
                self._jitted_cppn_forward,
                state,
                num_gpus=num_devices,
                pop_chunk_size=pop_chunk_size,
            )

            # NEAT evolution step
            if use_extra_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            state = self._compiled_tell(state, fitnesses)

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall_best={best_so_far:.4f}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def run_until_threshold_streaming(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        data_chunk_size: int = 1000,
        collect_history: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run generations with STREAMING data from CPU.

        This strategy handles VERY LARGE DATASETS that exceed total GPU memory:
        - Dataset stays on CPU (as numpy array)
        - Chunks are streamed to GPU one at a time
        - Each chunk is evaluated using multi-GPU (population-parallel)
        - Fitness is accumulated across chunks

        Best for: Datasets that exceed total GPU memory

        Trade-off:
        - Pro: Can handle datasets of ANY size (limited only by CPU RAM)
        - Con: CPU-GPU transfer overhead per chunk (~2ms per 4MB at ~2GB/s)

        Args:
            state: Initialized algorithm state
            problem: Problem instance with get_data()
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            data_chunk_size: Number of samples per chunk to stream from CPU
            collect_history: If True, collect per-generation fitness history
            verbose: If True, print per-generation progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state', and optionally 'history'
        """
        import numpy as np

        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates (these stay on GPU)
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions

        # Keep problem data on CPU for streaming
        data = problem.get_data()
        inputs_cpu = np.array([d[0] for d in data], dtype=np.float32)
        targets_cpu = np.array([d[1] for d in data], dtype=np.float32)
        n_samples = inputs_cpu.shape[0]
        num_chunks = (n_samples + data_chunk_size - 1) // data_chunk_size

        # Get grid info
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels
        total_positions = h_grid.total_positions

        # Compute memory-aware chunk size for population
        TARGET_MEMORY_GB = 2.0
        memory_per_network = data_chunk_size * total_positions * 4

        if memory_per_network > 0:
            max_chunk_for_memory = max(1, int((TARGET_MEMORY_GB * 1e9) / memory_per_network))
        else:
            max_chunk_for_memory = None

        # Settings
        use_extra_split = self.extra_randkey_split

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        pop_chunk_size = None
        chunk_size_computed = False

        if verbose:
            print(f"Streaming: {num_devices} GPUs, "
                  f"positions={total_positions}, n_samples={n_samples}")
            print(f"  Data chunk size: {data_chunk_size} samples ({num_chunks} chunks)")

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter() if verbose else None

            # CPPN ask + transform
            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Compute chunk size on first iteration
            if not chunk_size_computed:
                pop_size = cppn_population[0].shape[0]
                per_gpu_pop = (pop_size + num_devices - 1) // num_devices
                if max_chunk_for_memory is not None and per_gpu_pop > max_chunk_for_memory:
                    pop_chunk_size = max_chunk_for_memory
                else:
                    pop_chunk_size = None
                if verbose and pop_chunk_size is not None:
                    print(f"  Population chunking: {pop_chunk_size} individuals per chunk")
                chunk_size_computed = True

            # Run pipeline with STREAMING execution
            fitnesses = full_pipeline_streaming(
                cppns_transformed,
                all_positions,
                input_coords,
                output_coords,
                inputs_cpu,
                targets_cpu,
                level_sizes,
                level_offsets,
                level_grid_sizes,
                parent_indices,
                num_levels,
                total_positions,
                self.variance_threshold,
                self.max_weight,
                0.1,  # weight_thresh
                self._jitted_cppn_forward,
                state,
                data_chunk_size=data_chunk_size,
                num_gpus=num_devices,
                pop_chunk_size=pop_chunk_size,
            )

            # NEAT evolution step
            if use_extra_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            state = self._compiled_tell(state, fitnesses)

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall_best={best_so_far:.4f}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    # ========================================================================
    # H→H Caching Methods for Recurrent Network Support
    # ========================================================================

    def _pure_generation_step_with_fixed_hh(
        self,
        state: Any,
        cppns_transformed: Tuple,
        h_grid: Any,  # HierarchicalGrid
        input_coords: jnp.ndarray,
        output_coords: jnp.ndarray,
        inputs_batch: jnp.ndarray,
        targets_batch: jnp.ndarray,
        sparse_hh: Any,  # SparseHiddenConnections (imported lazily)
        activate_time: int,
        extra_randkey_split: bool = False,
    ) -> Tuple[Any, jnp.ndarray]:
        """Single generation step with FIXED sparse h→h connections - PURE JAX.

        Like _pure_generation_step but uses pre-computed sparse h→h
        connections instead of computing W3 every generation. This eliminates
        the expensive batch_query_population_all_to_all() call per generation.

        The sparse_hh is computed ONCE before the while_loop and passed in.

        Args:
            state: TensorNEAT algorithm state (JAX pytree)
            cppns_transformed: Pre-transformed CPPN population
            h_grid: Hierarchical grid configuration
            input_coords: Substrate input coordinates
            output_coords: Substrate output coordinates
            inputs_batch: Cached problem inputs
            targets_batch: Cached problem targets
            sparse_hh: Pre-computed sparse h→h connections (FIXED throughout loop)
            activate_time: Number of recurrent iterations
            extra_randkey_split: Extra random key split before tell()

        Returns:
            Tuple of (new_state, fitnesses) - no GPU→CPU sync

        Performance:
            - No O(N²) CPPN query per generation (vs _pure_generation_step_recurrent)
            - Uses sparse scatter-add instead of dense matmul
            - Can be used inside jax.lax.while_loop for GPU-resident evolution
        """
        # Optional pre-tell random key split
        if extra_randkey_split:
            randkey_, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # STEP 0: CPPN ask + transform
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Get grid info
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # STEP 1: Query CPPN for variance computation
        # Use JAX-pure version for while_loop compatibility (no numpy operations)
        source_coord = input_coords[0:1]
        all_weights_for_variance = batch_query_population_multi_source_jax_pure(
            state, cppns_transformed, source_coord,
            all_positions, True, self._jitted_cppn_forward
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        # STEP 2: Compute masks
        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # STEP 3: Query W1 (input→hidden) and W2 (hidden→output)
        # Use JAX-pure versions for while_loop compatibility
        input_all_weights = batch_query_population_multi_source_jax_pure(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward
        )
        output_all_weights = batch_query_population_multi_source_jax_pure(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward
        )

        # STEP 4: Apply masks and build W1/W2 weight matrices
        # (NO W3 computation - using pre-computed sparse_hh instead)
        max_weight = self.max_weight
        weight_thresh = 0.1

        # W1: input→hidden
        W1_raw = jnp.tanh(input_all_weights) * max_weight
        active_mask_broadcast = masks_A[:, None, :]
        W1_combined_mask = active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh)
        W1 = W1_raw * W1_combined_mask

        # W2: hidden→output
        W2_raw = jnp.tanh(output_all_weights) * max_weight
        W2_combined_mask = active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh)
        W2_raw = W2_raw * W2_combined_mask
        W2 = jnp.transpose(W2_raw, (0, 2, 1))

        # STEP 5: Evaluate ALL networks via vmap with sparse h→h
        num_positions = total_positions

        def eval_single_network_sparse_hh(
            W1_single, W2_single,
            from_idx, to_idx, hh_weights, hh_valid_mask,
            inputs, targets, act_time
        ):
            """Evaluate single network with SPARSE h→h connections."""
            # Input contribution (constant across iterations)
            input_contrib = safe_matmul(inputs, W1_single)  # (n_samples, num_positions)
            hidden = jnp.zeros((inputs.shape[0], num_positions))

            # Precompute safe indices and masked weights
            safe_from = jnp.clip(from_idx, 0, num_positions - 1)
            safe_to = jnp.clip(to_idx, 0, num_positions - 1)
            effective_weights = jnp.where(hh_valid_mask, hh_weights, 0.0)

            def sparse_hh_step(hidden, _):
                """Single h→h iteration using sparse scatter-add."""
                # Gather from source positions: (n_samples, max_sparse_conns)
                source_vals = hidden[:, safe_from]

                # Multiply by connection weights
                contributions = source_vals * effective_weights

                # Scatter-add to target positions
                h_delta = jnp.zeros_like(hidden)
                h_delta = h_delta.at[:, safe_to].add(contributions)

                # Combine input and recurrent contributions
                return jnp.tanh(input_contrib + h_delta), None

            # Run recurrent iterations
            hidden_final, _ = jax.lax.scan(sparse_hh_step, hidden, None, length=act_time)

            # Output layer
            outputs = jax.nn.sigmoid(safe_matmul(hidden_final, W2_single))
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        # vmap over population, broadcast inputs/targets/activate_time
        fitnesses = jax.vmap(
            eval_single_network_sparse_hh,
            in_axes=(0, 0, 0, 0, 0, 0, None, None, None)
        )(
            W1, W2,
            sparse_hh.from_indices, sparse_hh.to_indices,
            sparse_hh.weights, sparse_hh.valid_mask,
            inputs_batch, targets_batch, activate_time
        )
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        # STEP 6: NEAT evolution step
        new_state = self._compiled_tell(state, fitnesses)

        return new_state, fitnesses

    def run_until_threshold_with_fixed_hh(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        extended_config: Optional[Any] = None,  # UnifiedExtendedConfig, imported lazily
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run evolution with FIXED sparse h→h connections - DISPATCHER.

        Routes to appropriate implementation based on self.strategy:
        - SINGLE_GPU, STREAMING, CPPN_CHUNKED: GPU-resident single-GPU loop
        - ALL multi-GPU strategies: UNIFIED pmap (SINGLE pmap call per generation)

        The UNIFIED pmap approach wraps the ENTIRE generation pipeline (CPPN queries +
        masks + W1/W2 + sparse h→h eval) into a SINGLE pmap call, achieving true
        multi-GPU parallelism. This is 4-6x faster than the previous multi-GPU
        implementations which made 6+ separate Python→JAX calls per generation.

        Expected performance:
        - Single-GPU: ~800ms/gen (baseline)
        - Multi-GPU (unified): ~400-600ms/gen (1.3-2x faster than single-GPU)
        - Multi-GPU (old): ~2400ms/gen (3x SLOWER - deprecated)

        All implementations:
        - Pre-compute sparse h→h connections ONCE (outside loop)
        - Use fixed h→h topology throughout evolution
        - Return sparse_hh_stats with discovery metrics

        Args:
            state: Initialized algorithm state from initialize()
            problem: Problem instance (must have get_data() method)
            target_fitness: Stop when jnp.max(fitnesses) >= target_fitness
            max_generations: Maximum generations before stopping
            collect_history: If True, collect per-generation best fitness
            extended_config: Optional UnifiedExtendedConfig for h→h discovery.
            verbose: If True, print per-generation progress

        Returns:
            Dict with 'generations', 'best_fitness', 'state', 'sparse_hh_stats',
            and optionally 'history'.
        """
        num_devices = len(jax.devices())

        # Check recurrence is enabled (common to all implementations)
        if self.recurrence_config is None or not self.recurrence_config.enabled:
            raise ValueError(
                "run_until_threshold_with_fixed_hh requires recurrence to be enabled. "
                "Set recurrence_config when creating the algorithm, or use run_until_threshold for feedforward networks."
            )

        # Route based on strategy and device count
        # Single GPU strategies (or only 1 GPU available)
        if num_devices < 2 or self.strategy in (
            MultiGPUStrategy.SINGLE_GPU,
            MultiGPUStrategy.BASELINE,
            MultiGPUStrategy.STREAMING,
            MultiGPUStrategy.CPPN_CHUNKED,
            MultiGPUStrategy.POSITION_SHARDING_CHUNKED,
        ):
            return self._run_until_threshold_with_fixed_hh_single_gpu(
                state, problem, target_fitness, max_generations,
                collect_history, extended_config
            )

        # ALL multi-GPU strategies use UNIFIED pmap (SINGLE pmap call per generation)
        # This is 4-6x faster than the old approach which made 6+ separate JAX calls
        if self.strategy in (
            MultiGPUStrategy.DATA_PARALLEL,
            MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
            MultiGPUStrategy.MULTI_GPU,
            MultiGPUStrategy.PIPELINE_CHUNKED,
            MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL,
            MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
            MultiGPUStrategy.EVAL_ONLY_PARALLEL,
            MultiGPUStrategy.PERSISTENT_PARALLEL,
        ):
            return self._run_until_threshold_pmap_unified_with_fixed_hh(
                state, problem, target_fitness, max_generations,
                collect_history, extended_config, verbose
            )

        # Fallback to single GPU
        return self._run_until_threshold_with_fixed_hh_single_gpu(
            state, problem, target_fitness, max_generations,
            collect_history, extended_config
        )

    def _run_until_threshold_with_fixed_hh_single_gpu(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        extended_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Single-GPU implementation with GPU-resident while_loop.

        Uses jax.lax.while_loop for minimal GPU→CPU synchronization.
        All computation stays on GPU 0 until final result retrieval.
        """
        import time as time_module

        # Lazy import to avoid circular dependency
        from emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended_dynamic_functions_full import (
            SparseHiddenConnections,
            discover_sparse_hh_vectorized_multi_hop,
            UnifiedExtendedConfig,
        )

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates as JAX arrays
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # Get activate_time
        activate_time = self.recurrence_config.activate_time

        # Convert target to JAX arrays for GPU-side comparison
        target_fitness_jax = jnp.array(target_fitness, dtype=jnp.float32)
        max_gens_jax = jnp.array(max_generations, dtype=jnp.int32)

        # Capture settings for use in loop body
        use_extra_split = getattr(self, 'extra_randkey_split', False)

        # ======================================================================
        # Pre-compute sparse h→h connections ONCE (outside while_loop)
        # ======================================================================
        discovery_start = time_module.time()

        # Get initial transformed CPPNs
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)

        # Compute variance masks from initial population
        # Use chunked version which is vmap-safe (no np.asarray inside vmap)
        source_coord = input_coords[0:1]
        pop_size = cppns_transformed[0].shape[0]
        all_weights_for_variance = batch_query_population_multi_source_chunked(
            state, cppns_transformed, source_coord,
            h_grid.all_positions, True, self._jitted_cppn_forward,
            pop_chunk_size=min(100, pop_size)
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # Create UnifiedExtendedConfig if not provided
        if extended_config is None:
            extended_config = UnifiedExtendedConfig(
                enabled=self.recurrence_config.enabled,
                allow_hidden_to_hidden=self.recurrence_config.allow_hidden_to_hidden,
                allow_backward=self.recurrence_config.allow_backward,
                allow_lateral=self.recurrence_config.allow_lateral,
                allow_self_loops=self.recurrence_config.allow_self_loops,
                iteration_level=self.recurrence_config.iteration_level,
                activate_time=self.recurrence_config.activate_time,
                max_connections=self.recurrence_config.max_connections,
                max_sparse_conns=getattr(self.recurrence_config, 'max_connections', 10000),
            )

        # Discover sparse h→h connections ONCE (pop_size already defined above)
        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
            state=state,
            cppns_transformed=cppns_transformed,
            h_grid=h_grid,
            masks_A=masks_A,
            band_threshold=self.band_threshold,
            max_weight=self.max_weight,
            config=extended_config,
            cppn_forward=self._jitted_cppn_forward,
            pop_chunk_size=min(100, pop_size),
            verbose=False,
        )

        discovery_time = time_module.time() - discovery_start
        total_hh_conns = int(jax.device_get(sparse_hh.num_valid.sum()))
        avg_hh_conns = float(jax.device_get(sparse_hh.num_valid.mean()))

        # ======================================================================
        # GPU-resident while_loop with fixed h→h
        # ======================================================================

        if collect_history:
            # Version with history collection
            def loop_body(carry):
                generation, best_so_far, current_state, history = carry

                new_state, fitnesses = self._pure_generation_step_with_fixed_hh(
                    current_state, cppns_transformed, h_grid,
                    input_coords, output_coords,
                    inputs_batch, targets_batch,
                    sparse_hh, activate_time,
                    extra_randkey_split=use_extra_split
                )

                gen_best = jnp.max(fitnesses)
                best_so_far = jnp.maximum(best_so_far, gen_best)
                history = history.at[generation].set(gen_best)

                return (generation + 1, best_so_far, new_state, history)

            def loop_condition(carry):
                generation, best_so_far, current_state, history = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            # Pre-allocate history array
            history = jnp.zeros(max_generations, dtype=jnp.float32)
            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state,
                history
            )

            final_gen, final_best, final_state, final_history = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            # SINGLE GPU→CPU sync at the very end
            final_gen_py = int(jax.device_get(final_gen))
            return {
                'generations': final_gen_py,
                'best_fitness': float(jax.device_get(final_best)),
                'state': final_state,
                'history': jax.device_get(final_history[:final_gen_py]),
                'sparse_hh_stats': {
                    'discovery_time_ms': discovery_time * 1000,
                    'hh_connections': total_hh_conns,
                    'total_connections': total_hh_conns,
                    'avg_connections_per_genome': avg_hh_conns,
                },
            }

        else:
            # Version without history (minimal memory)
            def loop_body(carry):
                generation, best_so_far, current_state = carry

                new_state, fitnesses = self._pure_generation_step_with_fixed_hh(
                    current_state, cppns_transformed, h_grid,
                    input_coords, output_coords,
                    inputs_batch, targets_batch,
                    sparse_hh, activate_time,
                    extra_randkey_split=use_extra_split
                )

                gen_best = jnp.max(fitnesses)
                best_so_far = jnp.maximum(best_so_far, gen_best)

                return (generation + 1, best_so_far, new_state)

            def loop_condition(carry):
                generation, best_so_far, current_state = carry
                return (best_so_far < target_fitness_jax) & (generation < max_gens_jax)

            initial_carry = (
                jnp.array(0, dtype=jnp.int32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                state
            )

            final_gen, final_best, final_state = jax.lax.while_loop(
                loop_condition, loop_body, initial_carry
            )

            # SINGLE GPU→CPU sync - pack scalars and transfer together
            results_packed = jnp.stack([final_gen.astype(jnp.float32), final_best])
            results_cpu = jax.device_get(results_packed)
            return {
                'generations': int(results_cpu[0]),
                'best_fitness': float(results_cpu[1]),
                'state': final_state,
                'sparse_hh_stats': {
                    'discovery_time_ms': discovery_time * 1000,
                    'hh_connections': total_hh_conns,
                    'total_connections': total_hh_conns,
                    'avg_connections_per_genome': avg_hh_conns,
                },
            }

    def _run_until_threshold_data_parallel_with_fixed_hh(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        extended_config: Optional[Any] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Data-parallel multi-GPU implementation with fixed h→h caching.

        Strategy: Dataset is SHARDED across GPUs, population is REPLICATED.
        - Each GPU evaluates FULL population on its data shard
        - H→H connections are discovered ONCE and replicated to all GPUs
        - Fitness values are averaged across GPUs

        Uses jax.pmap for native multi-device execution (no Python multiprocessing).
        """
        import time as time_module

        # Lazy imports
        from emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended_dynamic_functions_full import (
            SparseHiddenConnections,
            discover_sparse_hh_vectorized_multi_hop,
            UnifiedExtendedConfig,
        )

        num_devices = len(jax.devices())
        devices = jax.devices()

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions
        total_positions = h_grid.total_positions

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get activate_time
        activate_time = self.recurrence_config.activate_time

        # Capture settings
        use_extra_split = getattr(self, 'extra_randkey_split', False)

        # ======================================================================
        # Pre-compute sparse h→h connections ONCE
        # ======================================================================
        discovery_start = time_module.time()

        # Get initial transformed CPPNs
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        pop_size = cppns_transformed[0].shape[0]

        # Compute variance masks
        source_coord = input_coords[0:1]
        all_weights_for_variance = batch_query_population_multi_source_chunked(
            state, cppns_transformed, source_coord,
            h_grid.all_positions, True, self._jitted_cppn_forward,
            pop_chunk_size=min(100, pop_size)
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # Create UnifiedExtendedConfig if not provided
        if extended_config is None:
            extended_config = UnifiedExtendedConfig(
                enabled=self.recurrence_config.enabled,
                allow_hidden_to_hidden=self.recurrence_config.allow_hidden_to_hidden,
                allow_backward=self.recurrence_config.allow_backward,
                allow_lateral=self.recurrence_config.allow_lateral,
                allow_self_loops=self.recurrence_config.allow_self_loops,
                iteration_level=self.recurrence_config.iteration_level,
                activate_time=self.recurrence_config.activate_time,
                max_connections=self.recurrence_config.max_connections,
                max_sparse_conns=getattr(self.recurrence_config, 'max_connections', 10000),
            )

        # Discover sparse h→h connections ONCE
        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
            state=state,
            cppns_transformed=cppns_transformed,
            h_grid=h_grid,
            masks_A=masks_A,
            band_threshold=self.band_threshold,
            max_weight=self.max_weight,
            config=extended_config,
            cppn_forward=self._jitted_cppn_forward,
            pop_chunk_size=min(100, pop_size),
            verbose=False,
        )

        discovery_time = time_module.time() - discovery_start
        total_hh_conns = int(jax.device_get(sparse_hh.num_valid.sum()))
        avg_hh_conns = float(jax.device_get(sparse_hh.num_valid.mean()))

        # ======================================================================
        # Shard dataset, replicate population and h→h
        # ======================================================================

        # Pad samples to be divisible by num_devices
        per_gpu_samples = (n_samples + num_devices - 1) // num_devices
        padded_samples = per_gpu_samples * num_devices
        if padded_samples > n_samples:
            pad_size = padded_samples - n_samples
            inputs_batch = jnp.concatenate([
                inputs_batch,
                jnp.zeros((pad_size, inputs_batch.shape[1]), dtype=inputs_batch.dtype)
            ], axis=0)
            targets_batch = jnp.concatenate([
                targets_batch,
                jnp.zeros((pad_size, targets_batch.shape[1]), dtype=targets_batch.dtype)
            ], axis=0)

        # Reshape for sharding: (num_devices, per_gpu_samples, ...)
        inputs_sharded = inputs_batch.reshape(num_devices, per_gpu_samples, -1)
        targets_sharded = targets_batch.reshape(num_devices, per_gpu_samples, -1)

        if verbose:
            print(f"Data-parallel multi-GPU: {num_devices} GPUs")
            print(f"  Positions: {total_positions}, Population: {pop_size}")
            print(f"  Samples: {n_samples} → {per_gpu_samples} per GPU")
            print(f"  H→H connections: {total_hh_conns} total, {avg_hh_conns:.1f} avg")

        # ======================================================================
        # Python while loop with pmap evaluation
        # ======================================================================

        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        # Grid parameters for evaluation
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels

        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time_module.perf_counter() if verbose else None

            # CPPN ask + transform
            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Compute variance and masks (needed per-generation as CPPNs evolve)
            all_weights_for_variance = batch_query_population_multi_source_jax_pure(
                state, cppns_transformed, source_coord,
                all_positions, True, self._jitted_cppn_forward
            )
            all_weights_for_variance = all_weights_for_variance[:, 0, :]

            level_variances = compute_hierarchical_variances_batch_jit(
                all_weights_for_variance,
                level_sizes=level_sizes,
                level_offsets=level_offsets,
                level_grid_sizes=level_grid_sizes,
                num_levels=num_levels,
            )
            masks_A, _, _ = compute_subdivision_masks_batch_jit(
                level_variances,
                variance_threshold=self.variance_threshold,
                parent_indices_tuple=parent_indices,
                level_offsets=level_offsets,
                num_levels=num_levels,
                total_positions=total_positions,
            )

            # Query W1 and W2
            input_all_weights = batch_query_population_multi_source_jax_pure(
                state, cppns_transformed, input_coords,
                all_positions, True, self._jitted_cppn_forward
            )
            output_all_weights = batch_query_population_multi_source_jax_pure(
                state, cppns_transformed, output_coords,
                all_positions, False, self._jitted_cppn_forward
            )

            # Build W1 and W2 matrices
            max_weight = self.max_weight
            weight_thresh = 0.1

            W1_raw = jnp.tanh(input_all_weights) * max_weight
            active_mask_broadcast = masks_A[:, None, :]
            W1_combined_mask = active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh)
            W1 = W1_raw * W1_combined_mask

            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W2_combined_mask = active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh)
            W2_raw = W2_raw * W2_combined_mask
            W2 = jnp.transpose(W2_raw, (0, 2, 1))

            # Replicate W1, W2, sparse_hh across devices (add leading axis)
            W1_replicated = jnp.broadcast_to(W1[None, ...], (num_devices,) + W1.shape)
            W2_replicated = jnp.broadcast_to(W2[None, ...], (num_devices,) + W2.shape)

            # Sparse h→h arrays: replicate across devices
            from_idx_rep = jnp.broadcast_to(sparse_hh.from_indices[None, ...], (num_devices,) + sparse_hh.from_indices.shape)
            to_idx_rep = jnp.broadcast_to(sparse_hh.to_indices[None, ...], (num_devices,) + sparse_hh.to_indices.shape)
            weights_rep = jnp.broadcast_to(sparse_hh.weights[None, ...], (num_devices,) + sparse_hh.weights.shape)
            valid_mask_rep = jnp.broadcast_to(sparse_hh.valid_mask[None, ...], (num_devices,) + sparse_hh.valid_mask.shape)

            # pmap evaluation: data sharded, population replicated
            # Use the existing eval_single_network_sparse_hh logic via vmap inside pmap
            def eval_pop_on_data_shard(W1_pop, W2_pop, from_idx, to_idx, hh_weights, hh_valid,
                                        inputs_shard, targets_shard):
                """Evaluate full population on one data shard."""
                num_pos = total_positions

                def eval_single_network_sparse_hh(W1_single, W2_single, f_idx, t_idx, hw, hv):
                    input_contrib = safe_matmul(inputs_shard, W1_single)
                    hidden = jnp.zeros((inputs_shard.shape[0], num_pos))

                    safe_from = jnp.clip(f_idx, 0, num_pos - 1)
                    safe_to = jnp.clip(t_idx, 0, num_pos - 1)
                    effective_weights = jnp.where(hv, hw, 0.0)

                    def sparse_hh_step(hidden, _):
                        source_vals = hidden[:, safe_from]
                        contributions = source_vals * effective_weights
                        h_delta = jnp.zeros_like(hidden)
                        h_delta = h_delta.at[:, safe_to].add(contributions)
                        return jnp.tanh(input_contrib + h_delta), None

                    hidden_final, _ = jax.lax.scan(sparse_hh_step, hidden, None, length=activate_time)
                    outputs = jax.nn.sigmoid(safe_matmul(hidden_final, W2_single))
                    errors = jnp.mean((outputs - targets_shard) ** 2, axis=1)
                    return 1.0 - jnp.mean(errors)

                # vmap over population
                return jax.vmap(eval_single_network_sparse_hh)(
                    W1_pop, W2_pop, from_idx, to_idx, hh_weights, hh_valid
                )

            # Create pmap function
            pmap_eval = jax.pmap(
                eval_pop_on_data_shard,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0),  # All arrays have leading device axis
            )

            # Run pmap evaluation
            partial_fitnesses = pmap_eval(
                W1_replicated, W2_replicated,
                from_idx_rep, to_idx_rep, weights_rep, valid_mask_rep,
                inputs_sharded, targets_sharded
            )

            # Average fitnesses across GPUs (data shards)
            # partial_fitnesses shape: (num_devices, pop_size)
            # For data-parallel, we average the fitness from different data shards
            fitnesses = jnp.mean(partial_fitnesses, axis=0)
            fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

            # Optional random key split
            if use_extra_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            # NEAT evolution step
            state = self._compiled_tell(state, fitnesses)

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if verbose:
                gen_time = time_module.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            'sparse_hh_stats': {
                'discovery_time_ms': discovery_time * 1000,
                'hh_connections': total_hh_conns,
                'total_connections': total_hh_conns,
                'avg_connections_per_genome': avg_hh_conns,
            },
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_population_parallel_with_fixed_hh(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        extended_config: Optional[Any] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Population-parallel multi-GPU implementation with fixed h→h caching.

        Strategy: Population is SHARDED across GPUs, dataset is REPLICATED.
        - Each GPU evaluates its population slice on FULL dataset
        - H→H connections are discovered ONCE and sharded by population
        - Fitness values are concatenated across GPUs

        Uses jax.pmap for native multi-device execution (no Python multiprocessing).
        """
        import time as time_module

        # Lazy imports
        from emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended_dynamic_functions_full import (
            SparseHiddenConnections,
            discover_sparse_hh_vectorized_multi_hop,
            UnifiedExtendedConfig,
        )

        num_devices = len(jax.devices())
        devices = jax.devices()

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions
        total_positions = h_grid.total_positions

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get activate_time
        activate_time = self.recurrence_config.activate_time

        # Capture settings
        use_extra_split = getattr(self, 'extra_randkey_split', False)

        # ======================================================================
        # Pre-compute sparse h→h connections ONCE
        # ======================================================================
        discovery_start = time_module.time()

        # Get initial transformed CPPNs
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        pop_size = cppns_transformed[0].shape[0]

        # Compute variance masks
        source_coord = input_coords[0:1]
        all_weights_for_variance = batch_query_population_multi_source_chunked(
            state, cppns_transformed, source_coord,
            h_grid.all_positions, True, self._jitted_cppn_forward,
            pop_chunk_size=min(100, pop_size)
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # Create UnifiedExtendedConfig if not provided
        if extended_config is None:
            extended_config = UnifiedExtendedConfig(
                enabled=self.recurrence_config.enabled,
                allow_hidden_to_hidden=self.recurrence_config.allow_hidden_to_hidden,
                allow_backward=self.recurrence_config.allow_backward,
                allow_lateral=self.recurrence_config.allow_lateral,
                allow_self_loops=self.recurrence_config.allow_self_loops,
                iteration_level=self.recurrence_config.iteration_level,
                activate_time=self.recurrence_config.activate_time,
                max_connections=self.recurrence_config.max_connections,
                max_sparse_conns=getattr(self.recurrence_config, 'max_connections', 10000),
            )

        # Discover sparse h→h connections ONCE
        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
            state=state,
            cppns_transformed=cppns_transformed,
            h_grid=h_grid,
            masks_A=masks_A,
            band_threshold=self.band_threshold,
            max_weight=self.max_weight,
            config=extended_config,
            cppn_forward=self._jitted_cppn_forward,
            pop_chunk_size=min(100, pop_size),
            verbose=False,
        )

        discovery_time = time_module.time() - discovery_start
        total_hh_conns = int(jax.device_get(sparse_hh.num_valid.sum()))
        avg_hh_conns = float(jax.device_get(sparse_hh.num_valid.mean()))

        # ======================================================================
        # Pad population for even sharding
        # ======================================================================

        # Pad population to be divisible by num_devices
        per_gpu_pop = (pop_size + num_devices - 1) // num_devices
        padded_pop_size = per_gpu_pop * num_devices
        pad_needed = padded_pop_size - pop_size

        if verbose:
            print(f"Population-parallel multi-GPU: {num_devices} GPUs")
            print(f"  Positions: {total_positions}, Samples: {n_samples}")
            print(f"  Population: {pop_size} → {per_gpu_pop} per GPU")
            print(f"  H→H connections: {total_hh_conns} total, {avg_hh_conns:.1f} avg")

        # ======================================================================
        # Python while loop with pmap evaluation
        # ======================================================================

        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        # Grid parameters for evaluation
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels

        # Replicate dataset across devices (add leading axis)
        inputs_replicated = jnp.broadcast_to(inputs_batch[None, ...], (num_devices,) + inputs_batch.shape)
        targets_replicated = jnp.broadcast_to(targets_batch[None, ...], (num_devices,) + targets_batch.shape)

        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time_module.perf_counter() if verbose else None

            # CPPN ask + transform
            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Compute variance and masks
            all_weights_for_variance = batch_query_population_multi_source_jax_pure(
                state, cppns_transformed, source_coord,
                all_positions, True, self._jitted_cppn_forward
            )
            all_weights_for_variance = all_weights_for_variance[:, 0, :]

            level_variances = compute_hierarchical_variances_batch_jit(
                all_weights_for_variance,
                level_sizes=level_sizes,
                level_offsets=level_offsets,
                level_grid_sizes=level_grid_sizes,
                num_levels=num_levels,
            )
            masks_A, _, _ = compute_subdivision_masks_batch_jit(
                level_variances,
                variance_threshold=self.variance_threshold,
                parent_indices_tuple=parent_indices,
                level_offsets=level_offsets,
                num_levels=num_levels,
                total_positions=total_positions,
            )

            # Query W1 and W2
            input_all_weights = batch_query_population_multi_source_jax_pure(
                state, cppns_transformed, input_coords,
                all_positions, True, self._jitted_cppn_forward
            )
            output_all_weights = batch_query_population_multi_source_jax_pure(
                state, cppns_transformed, output_coords,
                all_positions, False, self._jitted_cppn_forward
            )

            # Build W1 and W2 matrices
            max_weight = self.max_weight
            weight_thresh = 0.1

            W1_raw = jnp.tanh(input_all_weights) * max_weight
            active_mask_broadcast = masks_A[:, None, :]
            W1_combined_mask = active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh)
            W1 = W1_raw * W1_combined_mask

            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W2_combined_mask = active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh)
            W2_raw = W2_raw * W2_combined_mask
            W2 = jnp.transpose(W2_raw, (0, 2, 1))

            # Pad arrays if needed for even sharding
            if pad_needed > 0:
                W1 = jnp.concatenate([W1, jnp.zeros((pad_needed,) + W1.shape[1:], dtype=W1.dtype)], axis=0)
                W2 = jnp.concatenate([W2, jnp.zeros((pad_needed,) + W2.shape[1:], dtype=W2.dtype)], axis=0)

                # Pad sparse h→h arrays
                from_idx_padded = jnp.concatenate([
                    sparse_hh.from_indices,
                    jnp.zeros((pad_needed, sparse_hh.from_indices.shape[1]), dtype=sparse_hh.from_indices.dtype)
                ], axis=0)
                to_idx_padded = jnp.concatenate([
                    sparse_hh.to_indices,
                    jnp.zeros((pad_needed, sparse_hh.to_indices.shape[1]), dtype=sparse_hh.to_indices.dtype)
                ], axis=0)
                weights_padded = jnp.concatenate([
                    sparse_hh.weights,
                    jnp.zeros((pad_needed, sparse_hh.weights.shape[1]), dtype=sparse_hh.weights.dtype)
                ], axis=0)
                valid_mask_padded = jnp.concatenate([
                    sparse_hh.valid_mask,
                    jnp.zeros((pad_needed, sparse_hh.valid_mask.shape[1]), dtype=sparse_hh.valid_mask.dtype)
                ], axis=0)
            else:
                from_idx_padded = sparse_hh.from_indices
                to_idx_padded = sparse_hh.to_indices
                weights_padded = sparse_hh.weights
                valid_mask_padded = sparse_hh.valid_mask

            # Reshape for sharding: (num_devices, per_gpu_pop, ...)
            W1_sharded = W1.reshape(num_devices, per_gpu_pop, *W1.shape[1:])
            W2_sharded = W2.reshape(num_devices, per_gpu_pop, *W2.shape[1:])
            from_idx_sharded = from_idx_padded.reshape(num_devices, per_gpu_pop, -1)
            to_idx_sharded = to_idx_padded.reshape(num_devices, per_gpu_pop, -1)
            weights_sharded = weights_padded.reshape(num_devices, per_gpu_pop, -1)
            valid_mask_sharded = valid_mask_padded.reshape(num_devices, per_gpu_pop, -1)

            # pmap evaluation: population sharded, data replicated
            def eval_pop_shard_on_full_data(W1_shard, W2_shard, from_idx, to_idx, hh_weights, hh_valid,
                                            inputs_full, targets_full):
                """Evaluate population shard on full dataset."""
                num_pos = total_positions

                def eval_single_network_sparse_hh(W1_single, W2_single, f_idx, t_idx, hw, hv):
                    input_contrib = safe_matmul(inputs_full, W1_single)
                    hidden = jnp.zeros((inputs_full.shape[0], num_pos))

                    safe_from = jnp.clip(f_idx, 0, num_pos - 1)
                    safe_to = jnp.clip(t_idx, 0, num_pos - 1)
                    effective_weights = jnp.where(hv, hw, 0.0)

                    def sparse_hh_step(hidden, _):
                        source_vals = hidden[:, safe_from]
                        contributions = source_vals * effective_weights
                        h_delta = jnp.zeros_like(hidden)
                        h_delta = h_delta.at[:, safe_to].add(contributions)
                        return jnp.tanh(input_contrib + h_delta), None

                    hidden_final, _ = jax.lax.scan(sparse_hh_step, hidden, None, length=activate_time)
                    outputs = jax.nn.sigmoid(safe_matmul(hidden_final, W2_single))
                    errors = jnp.mean((outputs - targets_full) ** 2, axis=1)
                    return 1.0 - jnp.mean(errors)

                # vmap over population shard
                return jax.vmap(eval_single_network_sparse_hh)(
                    W1_shard, W2_shard, from_idx, to_idx, hh_weights, hh_valid
                )

            # Create pmap function
            pmap_eval = jax.pmap(
                eval_pop_shard_on_full_data,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0),  # All arrays have leading device axis
            )

            # Run pmap evaluation
            fitnesses_sharded = pmap_eval(
                W1_sharded, W2_sharded,
                from_idx_sharded, to_idx_sharded, weights_sharded, valid_mask_sharded,
                inputs_replicated, targets_replicated
            )

            # Concatenate fitnesses across GPUs and trim padding
            # fitnesses_sharded shape: (num_devices, per_gpu_pop)
            fitnesses = fitnesses_sharded.reshape(-1)[:pop_size]
            fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

            # Optional random key split
            if use_extra_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            # NEAT evolution step
            state = self._compiled_tell(state, fitnesses)

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if verbose:
                gen_time = time_module.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            'sparse_hh_stats': {
                'discovery_time_ms': discovery_time * 1000,
                'hh_connections': total_hh_conns,
                'total_connections': total_hh_conns,
                'avg_connections_per_genome': avg_hh_conns,
            },
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_pmap_unified_with_fixed_hh(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        extended_config: Optional[Any] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """UNIFIED multi-GPU implementation with SINGLE pmap call per generation.

        This is the OPTIMIZED multi-GPU approach that achieves true speedup:
        - ENTIRE generation pipeline runs inside ONE pmap call
        - CPPN queries are parallelized across GPUs (the bottleneck!)
        - Each GPU processes its population slice through complete pipeline
        - Reduces Python→JAX calls from 6+ to 1 per generation

        Expected performance: ~400-600ms/gen (vs 2400ms with separate pmap calls)
        This makes multi-GPU 1.5-2x FASTER than single-GPU (vs 3x slower before).

        Uses the module-level _pmap_pop_parallel_with_hh() function which wraps
        _full_pipeline_single_gpu_with_hh() to include CPPN queries, masks,
        W1/W2 building, and sparse h→h evaluation in a SINGLE pmap call.
        """
        import time as time_module

        # Lazy imports
        from emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended_dynamic_functions_full import (
            SparseHiddenConnections,
            discover_sparse_hh_vectorized_multi_hop,
            UnifiedExtendedConfig,
        )

        num_devices = len(jax.devices())
        devices = jax.devices()

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)

        # Prepare coordinates
        input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)
        all_positions = h_grid.all_positions
        total_positions = h_grid.total_positions

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get activate_time
        activate_time = self.recurrence_config.activate_time

        # Capture settings
        use_extra_split = getattr(self, 'extra_randkey_split', False)

        # ======================================================================
        # Pre-compute sparse h→h connections ONCE
        # ======================================================================
        discovery_start = time_module.time()

        # Get initial transformed CPPNs
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        pop_size = cppns_transformed[0].shape[0]

        # Compute variance masks for h→h discovery
        source_coord = input_coords[0:1]
        all_weights_for_variance = batch_query_population_multi_source_chunked(
            state, cppns_transformed, source_coord,
            h_grid.all_positions, True, self._jitted_cppn_forward,
            pop_chunk_size=min(100, pop_size)
        )
        all_weights_for_variance = all_weights_for_variance[:, 0, :]

        level_variances = compute_hierarchical_variances_batch_jit(
            all_weights_for_variance,
            level_sizes=h_grid.level_sizes_static,
            level_offsets=h_grid.level_offsets_static,
            level_grid_sizes=h_grid.level_grid_sizes_static,
            num_levels=h_grid.num_levels,
        )
        masks_A, _, _ = compute_subdivision_masks_batch_jit(
            level_variances,
            variance_threshold=self.variance_threshold,
            parent_indices_tuple=h_grid.parent_indices,
            level_offsets=h_grid.level_offsets_static,
            num_levels=h_grid.num_levels,
            total_positions=h_grid.total_positions,
        )

        # Create extended config for h→h discovery
        if extended_config is None:
            extended_config = UnifiedExtendedConfig(
                allow_hidden_to_hidden=self.recurrence_config.allow_hidden_to_hidden,
                allow_backward=self.recurrence_config.allow_backward,
                allow_lateral=self.recurrence_config.allow_lateral,
                allow_self_loops=self.recurrence_config.allow_self_loops,
                iteration_level=self.recurrence_config.iteration_level,
                activate_time=self.recurrence_config.activate_time,
                max_connections=self.recurrence_config.max_connections,
                max_sparse_conns=getattr(self.recurrence_config, 'max_connections', 10000),
            )

        # Discover sparse h→h connections ONCE
        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
            state=state,
            cppns_transformed=cppns_transformed,
            h_grid=h_grid,
            masks_A=masks_A,
            band_threshold=self.band_threshold,
            max_weight=self.max_weight,
            config=extended_config,
            cppn_forward=self._jitted_cppn_forward,
            pop_chunk_size=min(100, pop_size),
            verbose=False,
        )

        discovery_time = time_module.time() - discovery_start
        total_hh_conns = int(jax.device_get(sparse_hh.num_valid.sum()))
        avg_hh_conns = float(jax.device_get(sparse_hh.num_valid.mean()))

        # ======================================================================
        # Prepare population and h→h for sharding
        # ======================================================================

        # Pad population to be divisible by num_devices
        per_gpu_pop = (pop_size + num_devices - 1) // num_devices
        padded_pop_size = per_gpu_pop * num_devices
        pad_needed = padded_pop_size - pop_size

        # Pad sparse h→h arrays if needed
        if pad_needed > 0:
            from_idx_padded = jnp.concatenate([
                sparse_hh.from_indices,
                jnp.zeros((pad_needed, sparse_hh.from_indices.shape[1]), dtype=sparse_hh.from_indices.dtype)
            ], axis=0)
            to_idx_padded = jnp.concatenate([
                sparse_hh.to_indices,
                jnp.zeros((pad_needed, sparse_hh.to_indices.shape[1]), dtype=sparse_hh.to_indices.dtype)
            ], axis=0)
            weights_padded = jnp.concatenate([
                sparse_hh.weights,
                jnp.zeros((pad_needed, sparse_hh.weights.shape[1]), dtype=sparse_hh.weights.dtype)
            ], axis=0)
            valid_mask_padded = jnp.concatenate([
                sparse_hh.valid_mask,
                jnp.zeros((pad_needed, sparse_hh.valid_mask.shape[1]), dtype=sparse_hh.valid_mask.dtype)
            ], axis=0)
        else:
            from_idx_padded = sparse_hh.from_indices
            to_idx_padded = sparse_hh.to_indices
            weights_padded = sparse_hh.weights
            valid_mask_padded = sparse_hh.valid_mask

        # Pre-shard sparse h→h arrays: (padded_pop_size, ...) -> (num_devices, per_gpu_pop, ...)
        sparse_from_sharded = from_idx_padded.reshape(num_devices, per_gpu_pop, -1)
        sparse_to_sharded = to_idx_padded.reshape(num_devices, per_gpu_pop, -1)
        sparse_weights_sharded = weights_padded.reshape(num_devices, per_gpu_pop, -1)
        sparse_valid_sharded = valid_mask_padded.reshape(num_devices, per_gpu_pop, -1)

        # Grid parameters (static) - must be hashable for pmap static_broadcasted_argnums
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        # Convert parent_indices from JAX arrays to hashable Python tuples
        parent_indices = tuple(
            tuple(int(x) for x in arr.tolist()) if hasattr(arr, 'tolist') else tuple(arr)
            for arr in h_grid.parent_indices
        )
        num_levels = h_grid.num_levels

        if verbose:
            print(f"UNIFIED pmap multi-GPU: {num_devices} GPUs")
            print(f"  Positions: {total_positions}, Population: {pop_size} (padded: {padded_pop_size})")
            print(f"  Per GPU: {per_gpu_pop} individuals")
            print(f"  H→H connections: {total_hh_conns} total, {avg_hh_conns:.1f} avg")

        # ======================================================================
        # Python while loop with SINGLE UNIFIED pmap call per generation
        # ======================================================================

        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time_module.perf_counter() if verbose else None

            # CPPN ask + transform (only Python→JAX call besides the pmap)
            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Pad CPPNs if needed
            if pad_needed > 0:
                cppns_padded = []
                for arr in cppns_transformed:
                    pad_shape = (pad_needed,) + arr.shape[1:]
                    padded = jnp.concatenate([arr, jnp.zeros(pad_shape, dtype=arr.dtype)], axis=0)
                    cppns_padded.append(padded)
                cppns_transformed = tuple(cppns_padded)

            # Shard CPPNs: (padded_pop_size, ...) -> (num_devices, per_gpu_pop, ...)
            cppns_sharded = tuple(
                arr.reshape(num_devices, per_gpu_pop, *arr.shape[1:])
                for arr in cppns_transformed
            )

            # SINGLE UNIFIED PMAP CALL - ENTIRE generation pipeline
            # This includes: CPPN queries + masks + W1/W2 + sparse h→h eval
            fitnesses_sharded = _pmap_pop_parallel_with_hh(
                cppns_sharded,           # 0: population sharded
                all_positions,           # 1: replicated
                input_coords,            # 2: replicated
                output_coords,           # 3: replicated
                inputs_batch,            # 4: replicated
                targets_batch,           # 5: replicated
                sparse_from_sharded,     # 6: population sharded
                sparse_to_sharded,       # 7: population sharded
                sparse_weights_sharded,  # 8: population sharded
                sparse_valid_sharded,    # 9: population sharded
                self.variance_threshold, # 10: replicated
                self.max_weight,         # 11: replicated
                0.1,                     # 12: weight_thresh replicated
                state,                   # 13: replicated PyTree
                level_sizes,             # 14: STATIC
                level_offsets,           # 15: STATIC
                level_grid_sizes,        # 16: STATIC
                parent_indices,          # 17: STATIC
                num_levels,              # 18: STATIC
                total_positions,         # 19: STATIC
                activate_time,           # 20: STATIC (must be concrete for jax.lax.scan!)
                self._jitted_cppn_forward,  # 21: STATIC
                total_positions,         # 22: STATIC (num_positions for inner function)
            )

            # Concatenate fitnesses and trim padding
            # fitnesses_sharded shape: (num_devices, per_gpu_pop)
            fitnesses = fitnesses_sharded.reshape(-1)[:pop_size]
            fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

            # Optional random key split
            if use_extra_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            # NEAT evolution step
            state = self._compiled_tell(state, fitnesses)

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if verbose:
                gen_time = time_module.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            'sparse_hh_stats': {
                'discovery_time_ms': discovery_time * 1000,
                'hh_connections': total_hh_conns,
                'total_connections': total_hh_conns,
                'avg_connections_per_genome': avg_hh_conns,
            },
        }
        if collect_history:
            result['history'] = history

        return result


# ============================================================================
# Utility Functions
# ============================================================================

def get_available_strategies() -> List[str]:
    """Get list of available GPU execution strategies based on hardware.

    Returns:
        List of strategy names that can be used on this system.
        Primary strategies: "single_gpu", "full_pipeline_parallel", "eval_only_parallel", "cppn_chunked"
        Legacy aliases also included for backward compatibility.
    """
    # Primary production strategies
    strategies = [MultiGPUStrategy.SINGLE_GPU.value]
    num_devices = len(jax.devices())

    if num_devices >= 2 and SHARD_MAP_AVAILABLE:
        strategies.append(MultiGPUStrategy.FULL_PIPELINE_PARALLEL.value)  # Full pipeline on each GPU (RECOMMENDED for feedforward)
        strategies.append(MultiGPUStrategy.EVAL_ONLY_PARALLEL.value)  # Only eval parallel, h→h cached (RECOMMENDED for h→h)
        strategies.append(MultiGPUStrategy.CPPN_CHUNKED.value)  # CPPN-only chunking (FALLBACK)

    # Also include legacy aliases for backward compatibility
    strategies.append(MultiGPUStrategy.BASELINE.value)  # Alias for SINGLE_GPU
    if num_devices >= 2 and SHARD_MAP_AVAILABLE:
        strategies.append(MultiGPUStrategy.POSITION_SHARDING_CHUNKED.value)  # Alias for CPPN_CHUNKED
        strategies.append(MultiGPUStrategy.PIPELINE_CHUNKED.value)  # Alias for MULTI_GPU

    return strategies


def print_multi_gpu_info():
    """Print information about multi-GPU capabilities."""
    devices = jax.devices()
    print("=" * 60)
    print("MULTI-GPU INFORMATION")
    print("=" * 60)
    print(f"JAX devices: {devices}")
    print(f"Number of devices: {len(devices)}")
    print(f"shard_map available: {SHARD_MAP_AVAILABLE}")
    print(f"Available strategies: {get_available_strategies()}")
    print("=" * 60)
