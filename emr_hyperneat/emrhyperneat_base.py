"""The EMR-HyperNEAT engine: substrate discovery and execution beneath the public class.

This module holds the layer beneath the public `EMRHyperNEAT` class (defined in
`emrhyperneat.py`): the eager multi-resolution grid and its variance masking, the
recurrence/cache configuration objects, the execution-mode plumbing, and the
`MultiGPUStrategy` names (see that enum's docstring for what each strategy splits and
replicates; `EMRHyperNEAT.run_until_threshold` routes them).

Two facts callers get wrong most often:

- Multi-GPU engages only through ``run_until_threshold()``; ``run_generation()``
  is single-device by design.
- Recurrence is the configuration's ``emr_hyperneat.recurrence`` block;
  ``run_until_threshold_with_fixed_hh()`` and the ``recurrence_config`` argument are
  refused.

The user-facing story lives in docs/architecture.md ("How a generation is
computed"); this docstring intentionally stays short.
"""

import threading
import numpy as np
from typing import Any, Dict, Tuple, List, Optional, NamedTuple, Union
from dataclasses import dataclass
from enum import Enum
import jax
import jax.numpy as jnp
from functools import lru_cache

from emr_hyperneat._compat.core.base_algorithm import BaseAlgorithm
from emr_hyperneat._compat.adapters.tensorneat_adapter import TensorNEATAdapter


# Recurrence Configuration for H->H Caching

@dataclass
class RecurrenceConfig:
    """Configuration for recurrent network features with h->h caching.

    This dataclass controls which connection types are allowed during
    substrate discovery and how the forward pass handles recurrence.

    Connection Types:
    - Forward (y_source < y_target): Always allowed
    - Lateral (y_source == y_target): Same-layer connections
    - Backward (y_source > y_target): Feedback loops
    - Self-loop (x,y identical): Memory/state nodes

    Attributes:
        enabled: Master switch for recurrence features
        allow_hidden_to_hidden: Discover hidden-to-hidden connections
        allow_backward: Enable backward connections (y_source > y_target)
        allow_lateral: Enable same-layer connections (y_source == y_target)
        allow_self_loops: Enable self-connections (same x,y)
        iteration_level: Hops of the h->h multi-hop expansion (0 feed-forward, 1 direct
            connections only, k adds composites up to k hops)
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


# Multi-GPU Strategy Selection

class MultiGPUStrategy(Enum):
    """Available GPU execution strategies.

    Strategy selection guide

    Use this decision tree to choose the right strategy:

        Dataset fits on single GPU?
        |- yes -> SINGLE_GPU
        `- no  -> Multiple GPUs available?
                 |- no  -> none (STREAMING splits the CPPN queries by level, not the data)
                 `- yes -> Recurrent (h->h) mode?
                          |- yes -> EVAL_ONLY_PARALLEL (any iteration level)
                          `- no  -> FULL_PIPELINE_PARALLEL (data parallel)

    Strategy details

    SINGLE_GPU: Single GPU execution
        - Use for: Debugging, single-GPU systems, or small datasets
        - Memory: Full dataset + full population on one GPU

    FULL_PIPELINE_PARALLEL: Full pipeline on each GPU
        - Use for: a dataset too large for one device, feedforward mode
        - How it works: Each GPU runs the full pipeline (CPPN -> variance -> W1/W2 -> eval)
        - Dataset is split across GPUs, pipeline is replicated
        - Memory: Each GPU holds (dataset/num_gpus) + full population
        - In EMRHyperNEAT, a sample count that is not a multiple of the device count is padded
          with rows that add no error, so a sharded run scores as a single-device run (0.1.0
          padded with copies of the last sample, which then counted several times)
        - Note: in EMRHyperNEAT, discovers h->h at iteration level 1 only and refuses higher
          levels; for recurrent (h->h) modes, prefer EVAL_ONLY_PARALLEL

    POPULATION_PARALLEL_SEQUENTIAL: alias for POPULATION_PARALLEL_PROCESS
        - Kept so older configs keep parsing; the parser maps it onto the process arm
        - Splits the population across processes, replicates the dataset
        - Memory: each worker holds the full dataset and its shard, on its own device
        - Not recommended: see POPULATION_PARALLEL_PROCESS below

    POPULATION_PARALLEL: in EMRHyperNEAT, an alias for POPULATION_PMAP

    POPULATION_PMAP, POSITION_SHARDING, HYBRID: in EMRHyperNEAT, the split routes
        - How it works: the standard generation, on one device but for its one-output CPPN
          query and its evaluation, which run on a grid of devices: POPULATION_PMAP splits the
          genomes, POSITION_SHARDING the grid positions (the query) and the genomes (the
          evaluation), HYBRID both (emr_hyperneat.parallel); results are gathered on the
          first device
        - Every feature of the standard generation runs (h->h at any iteration level, per-node
          functions, aggregation, neuromodulation, LEO, palette evolution); on a CPU each
          generation equals SINGLE_GPU's bit for bit
        - Memory: a device holds its share of the query, in chunks of 100 genomes, and its
          share of the evaluation; the first device also holds the masks, the h->h discovery
          and W1/W2
        - Names: POPULATION_PARALLEL is POPULATION_PMAP; POSITION_PMAP, SHARD_MAP,
          CPPN_CHUNKED and POSITION_SHARDING_CHUNKED are POSITION_SHARDING

    ISLAND_MODEL, ISLAND_MODEL_V2: in EMRHyperNEAT, the island model (emr_hyperneat.islands)
        - Islands of the population evolve apart in lockstep, each a SINGLE_GPU run of its own
          population and seed, with optional migration (ring, star or random)

    POPULATION_PARALLEL_PROCESS: ProcessPoolExecutor-based population parallelism
        - Use for: nothing in particular; SINGLE_GPU and EVAL_ONLY_PARALLEL serialize nothing
        - How it works: Spawns one Python process per device, each started with that device's
          CUDA_VISIBLE_DEVICES (start_device_workers in multi_gpu_worker.py); h->h connections
          are discovered on GPU 0 and scored with the kernel EVAL_ONLY_PARALLEL uses
        - Each process has isolated JAX runtime and JIT cache
        - Memory: Same as POPULATION_PARALLEL_SEQUENTIAL
        - Benefit: isolated JAX runtimes, so the globally shared JIT cache cannot
          produce cross-device placement errors
        - Trade-off: IPC overhead (data serialization), JIT warm-up per process
        - Not recommended. This is the least exercised path in this release.
          A worker that fails raises. In 0.1.0 a failed worker left its shard at fitness
          0, every worker failed with hidden-to-hidden connections (a shape error in its h->h
          sum), and every worker ran on the first GPU. Development measurements made outside
          this repository, with every worker on the first GPU, found a best fitness of zero at
          population 1000, where every worker ran out of GPU memory, and found the path slower
          than SINGLE_GPU at populations 100 and 300. Those measurements are not reproducible
          from here.

    PERSISTENT_PARALLEL: alias for POPULATION_PARALLEL_PROCESS
        - Kept so older configs keep parsing; nothing ever started the persistent workers the
          name stood for, and this package no longer has them
        - Not recommended: see POPULATION_PARALLEL_PROCESS above

    EVAL_ONLY_PARALLEL: Only evaluation parallelized
        - Use for: Recurrent modes; discovery, multi-hop included, stays on GPU 0
        - How it works:
          1. GPU 0 runs: CPPN queries -> variance -> h->h discovery (cached if hh_cache_enabled)
          2. W1, W2, h->h broadcast to all GPUs via native JAX (no serialization)
          3. Dataset sharded across GPUs, weights replicated
          4. Only evaluation runs in parallel via pmap
        - Memory: Each GPU holds (dataset/num_gpus) + full W1/W2/h->h
        - Benefit: no IPC serialization; keeping discovery on one device lets it run
          multi-hop, and lets the h->h cache (off by default) survive across generations
        - Trade-off: Requires pure JAX evaluation kernel (no Python branching)
        - Best for: Recurrent (h->h) modes at any iteration level
        - A sample count that is not a multiple of the device count is padded with rows that
          add no error (in 0.1.0 they had zero targets and counted 0.25 per output)
        - Note: For feedforward, FULL_PIPELINE_PARALLEL scores the same

    STREAMING: one device, execution mode streaming
        - The dataset stays on the device; the CPPN is queried one level at a time, for
          substrates too large to query at once

    CPPN_CHUNKED: in EMRHyperNEAT, POSITION_SHARDING
        - The CPPN query split by grid position on two or more devices; on one device, the
          one-device loop

    A note on speed

    This docstring gives no per-strategy timings or speed ratios: the figures in the
    development notes have no result file behind them, and later measurements disagreed
    with them. Choose a strategy by the memory and correctness properties above, then
    measure on your own hardware.
    """
    # Production strategies
    SINGLE_GPU = "single_gpu"
    SINGLE_GPU_PYTHON_LOOP = "single_gpu_python_loop"  # Debugging: per-generation Python loop
    FULL_PIPELINE_PARALLEL = "full_pipeline_parallel"  # Full pipeline on each GPU
    POPULATION_PARALLEL_SEQUENTIAL = "population_parallel_sequential"  # Alias for the process arm
    POPULATION_PARALLEL = "population_parallel"  # EMRHyperNEAT: POPULATION_PMAP
    POPULATION_PARALLEL_PROCESS = "population_parallel_process"  # Process pool; not recommended
    PERSISTENT_PARALLEL = "persistent_parallel"  # Parsed as POPULATION_PARALLEL_PROCESS
    EVAL_ONLY_PARALLEL = "eval_only_parallel"  # Only eval parallelized, h->h on GPU 0
    STREAMING = "streaming"  # One device, the CPPN queried level by level
    CPPN_CHUNKED = "cppn_chunked"  # EMRHyperNEAT: POSITION_SHARDING

    # Legacy aliases (for backward compatibility)
    BASELINE = "baseline"  # Alias for SINGLE_GPU
    POSITION_SHARDING = "position"  # EMRHyperNEAT: the position split route
    POSITION_SHARDING_CHUNKED = "position_chunked"  # EMRHyperNEAT: POSITION_SHARDING
    PIPELINE_CHUNKED = "pipeline_chunked"  # Alias for FULL_PIPELINE_PARALLEL
    DATA_PARALLEL = "data_parallel"  # Alias for FULL_PIPELINE_PARALLEL
    MULTI_GPU = "multi_gpu"  # Alias for FULL_PIPELINE_PARALLEL (old name)
    PMAP_PARALLEL = "pmap_parallel"  # Alias for EVAL_ONLY_PARALLEL (old name)

    # EMRHyperNEAT's split routes and island model
    ISLAND_MODEL = "island"  # EMRHyperNEAT: the island model (emr_hyperneat.islands)
    ISLAND_MODEL_V2 = "island_v2"  # EMRHyperNEAT: ISLAND_MODEL
    SHARD_MAP = "shard_map"  # EMRHyperNEAT: POSITION_SHARDING
    HYBRID = "hybrid"  # EMRHyperNEAT: the 2-D split route
    POPULATION_PMAP = "pmap"  # EMRHyperNEAT: the population split route
    POSITION_PMAP = "position_pmap"  # EMRHyperNEAT: POSITION_SHARDING


# Platform-Specific Matrix Multiplication (CPU Fallback for CUDA)
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


# Dense Quadtree Data Structures

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

    This is computed once at initialization and reused for all CPPNs.
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
    Called once at algorithm initialization; reused for all CPPNs.

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
        # Node count of a complete quadtree: sum(4^i for i in 0..max_depth). The
        # multi-resolution grid formula, (4^(max_depth+2)-4)/3, would overstate this
        # structure's size by 4x.
        total_positions = (4 ** (max_depth + 1) - 1) // 3
        import warnings
        warnings.warn(
            f"max_depth={max_depth} creates {total_positions:,} quadtree positions. "
            f"Node count grows as O(4^n): depth 8 = 87,381, depth 9 = 349,525. "
            f"Memory scales with nodes x 17 bytes (positions, widths, levels, leaf mask). "
            f"This is the ES-HyperNEAT-lineage complete quadtree, read only by the node-list "
            f"model (recurrence.network_model 'node_list'). The multi-resolution paths use "
            f"get_hierarchical_grid(), which reports its own (larger) grid position count.",
            UserWarning,
            stacklevel=2
        )

    if max_depth not in _QUADTREE_CACHE:
        _QUADTREE_CACHE[max_depth] = precompute_quadtree_structure(max_depth)

    return _QUADTREE_CACHE[max_depth]


# Hierarchical Multi-Resolution Grid Structure
# This enables vmappable ES-HyperNEAT with real adaptive substrate discovery
# by pre-computing all positions at all resolution levels upfront.

class HierarchicalGridStructure(NamedTuple):
    """Pre-computed multi-resolution grid for vmappable ES-HyperNEAT.

    Unlike DenseQuadtreeStructure which is a complete quadtree, this structure
    stores positions at each resolution level separately for variance-based
    subdivision decisions.

    Grid Layout (for max_depth=2):
        Level 0: 2x2 = 4 cells   (coarsest)
        Level 1: 4x4 = 16 cells
        Level 2: 8x8 = 64 cells  (finest)

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


# CPU-resident hierarchical grid (single-precision, host memory)
# The GPU grid above materializes every candidate position as a device array,
# which becomes the binding constraint at extreme depth: depth 12 alone is
# 89.5M positions. These CPU-resident variants keep the same geometry in host
# memory as float32/NumPy, so the depth-offload execution path can stream
# levels from the host instead of holding the whole grid on device.
#
# Geometry is identical to `precompute_hierarchical_grid`; only residency and precision differ.


class HierarchicalGridStructureSingle(NamedTuple):
    """CPU-resident hierarchical grid for depth offloading.

    Unlike HierarchicalGridStructure which stores JAX arrays on GPU,
    this version keeps everything as NumPy arrays on CPU to avoid
    GPU memory allocation for ultra-deep substrates (depth 12+).

    Fields are transferred to GPU on-demand, one level at a time.
    This enables processing of depth 13+ (358M+ positions) on GPUs
    with limited VRAM by keeping the grid structure entirely on CPU.

    Flow: Grid init (CPU) -> Transfer level chunk -> Process -> Free -> Next chunk
    """
    # Positions at each level as NumPy arrays (CPU)
    # Each level has shape (4^(level+1), 2) for x,y coordinates
    level_positions: Tuple[np.ndarray, ...]

    # Parent indices: parent_indices[level][i] = index of parent at level-1
    # Level 0 has no parents (uses -1 as sentinel)
    parent_indices: Tuple[np.ndarray, ...]

    # Flattened all-level positions for recurrent/hidden_only modes
    # Used by discover_sparse_hh_vectorized_multi_hop for h->h connection discovery
    # Shape: (total_positions, 2) - concatenation of all level_positions
    all_positions: np.ndarray

    # Metadata
    max_depth: int
    num_levels: int
    total_positions: int

    # Static metadata for JIT-compatible functions (Python tuples, not JAX arrays)
    level_sizes_static: Tuple[int, ...]      # (4, 16, 64, ...) cells per level
    level_offsets_static: Tuple[int, ...]    # (0, 4, 20, 84, ...) cumulative offsets


def precompute_hierarchical_grid_cpu(max_depth: int) -> HierarchicalGridStructureSingle:
    """Pre-compute hierarchical multi-resolution grid structure on CPU only.

    This is the CPU-resident version of precompute_hierarchical_grid().
    All arrays remain as NumPy (on CPU) - no JAX conversion happens.

    This enables processing of ultra-deep substrates (depth 12+) by:
    1. Keeping the grid structure entirely on CPU
    2. Only transferring needed level positions to GPU on-demand
    3. Freeing GPU memory after processing each level

    Args:
        max_depth: Maximum level (0 = only 2x2 grid)

    Returns:
        HierarchicalGridStructureSingle with NumPy arrays (CPU-resident)
    """
    num_levels = max_depth + 1
    level_counts = compute_hierarchical_level_counts(max_depth)

    level_positions_list: List[np.ndarray] = []
    parent_indices_list: List[np.ndarray] = []

    for level in range(num_levels):
        grid_size = 2 ** (level + 1)  # 2, 4, 8, 16, ...
        num_cells = grid_size * grid_size

        # Cell width: full space is 2.0 (-1 to 1), divided by grid_size
        cell_width = 2.0 / grid_size
        half_width = cell_width / 2.0

        # Generate positions (cell centers)
        positions = np.zeros((num_cells, 2), dtype=np.float32)
        for row in range(grid_size):
            for col in range(grid_size):
                idx = row * grid_size + col
                x = -1.0 + half_width + col * cell_width
                y = -1.0 + half_width + row * cell_width
                positions[idx] = [x, y]

        level_positions_list.append(positions)

        # Parent indices: map each cell to its parent at level-1
        if level == 0:
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

    # Compute static metadata
    total_positions = sum(level_counts)
    level_offsets = [0]
    for count in level_counts:
        level_offsets.append(level_offsets[-1] + count)

    level_sizes_static = tuple(level_counts)
    level_offsets_static = tuple(level_offsets)

    # Compute all_positions for recurrent/hidden_only modes (h->h discovery)
    # This concatenates all level positions into a single array for use by
    # discover_sparse_hh_vectorized_multi_hop which needs to look up positions
    # by global index
    all_positions = np.concatenate(level_positions_list, axis=0)

    # Return CPU-resident structure (no JAX conversion)
    return HierarchicalGridStructureSingle(
        level_positions=tuple(level_positions_list),
        parent_indices=tuple(parent_indices_list),
        all_positions=all_positions,
        max_depth=max_depth,
        num_levels=num_levels,
        total_positions=total_positions,
        level_sizes_static=level_sizes_static,
        level_offsets_static=level_offsets_static,
    )


@lru_cache(maxsize=16)
def get_hierarchical_grid_cpu(max_depth: int) -> HierarchicalGridStructureSingle:
    """Get pre-computed CPU-resident hierarchical grid structure (cached).

    This is the CPU-resident version for depth offloading.
    Unlike get_hierarchical_grid() which returns JAX arrays on GPU,
    this returns NumPy arrays that remain on CPU.

    Use this for ultra-deep substrates (depth 12+) where the full grid
    structure would exceed GPU memory.

    Args:
        max_depth: Maximum depth (any non-negative integer)

    Returns:
        Pre-computed HierarchicalGridStructureSingle (CPU-resident)

    Note:
        For depth 13: grid uses ~4.3 GB CPU RAM (vs 17+ GB on GPU)
        - level_positions: 358M x 2 x 4 = 2.86 GB
        - parent_indices: 358M x 4 = 1.43 GB
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")

    return precompute_hierarchical_grid_cpu(max_depth)


def _neat_operator_block(algo_params: Dict[str, Any], name: str,
                         defaults: Dict[str, Any]) -> Dict[str, Any]:
    """One NEAT operator block (``neat_mutation``, ``neat_species``, ``neat_selection``).

    Returns the defaults overridden by the configured values. An unknown key raises, so a
    misspelled setting is never dropped silently.
    """
    block = algo_params.get(name) or {}
    unknown = sorted(set(block) - set(defaults))
    if unknown:
        raise ValueError(f"{name} has no setting {unknown}; its settings are {sorted(defaults)}")
    return {**defaults, **block}


# Hierarchical Variance and Subdivision Computation
# These functions compute variance at each level and determine which regions
# need subdivision based on ES-HyperNEAT semantics (high variance = subdivide).

def compute_hierarchical_variances_batch(
    all_weights: jnp.ndarray,
    grid: HierarchicalGridStructure,
    legacy_level0_variance: bool = False,
) -> List[jnp.ndarray]:
    """Compute variances for a batch of genomes (vmappable).

    Args:
        all_weights: CPPN outputs [pop_size, total_positions]
        grid: Pre-computed grid structure
        legacy_level0_variance: If True, level 0 uses the original standard-path rule:
            one scalar per genome over level 0's own four weights, so the whole level
            subdivides all-or-nothing. Default False: per-parent variance over the four
            level-1 children, which is the rule every other level uses, the one the
            streaming/offload path always used, and the one the papers describe.

    Returns:
        List of variance arrays per level, each [pop_size, num_cells_at_level]
    """
    pop_size = all_weights.shape[0]
    level_variances_batch = []

    for level in range(grid.num_levels - 1):  # No variance at finest level
        level_start = int(grid.level_offsets[level])
        level_end = int(grid.level_offsets[level + 1])

        if level == 0 and legacy_level0_variance:
            # Legacy level 0 (the original standard path): one scalar per genome over
            # level 0's own four weights, so the level subdivides all-or-nothing.
            level_weights = all_weights[:, level_start:level_end]
            variance = jnp.var(level_weights, axis=1, keepdims=True)
            level_variances_batch.append(variance)  # [pop_size, 1]
        else:
            # Canonical, level 0 included: variance of the 2x2 child blocks at level+1
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
    # Use numpy to avoid JIT cache cross-device errors
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
    initial_depth: int = 0,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Compute subdivision masks for a batch of genomes.

    Args:
        level_variances: List of [pop_size, num_cells] arrays per level
        variance_threshold: Threshold for subdivision
        grid: Pre-computed grid structure
        return_all_masks: If False, return only masks_A (saves ~0.66 GB at depth 8)
        initial_depth: Force full subdivision through this level (canonical
            ES-HyperNEAT subdivides on (level < initial_depth) OR variance;
            0 forces nothing and reproduces the variance-only mask exactly)

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

        # Canonical ES-HyperNEAT subdivides on two clauses:
        #   (level < initial_depth) or (variance > threshold)
        # A parent at level-1 is forced when level-1 < initial_depth, i.e.
        # level <= initial_depth. Static Python branch: unrolls at trace time.
        if level <= initial_depth:
            prev_subdivided = prev_reached
        else:
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

    # Skip masks_B and masks_C if not needed, so they are not allocated
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


# JIT-Compatible Variance and Mask Functions
# These functions use static Python integers for indices instead of JAX arrays,
# enabling use inside jax.lax.while_loop without ConcretizationTypeError.


def compute_hierarchical_variances_batch_jit(
    all_weights: jnp.ndarray,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[int, ...],
    num_levels: int,
    legacy_level0_variance: bool = False,
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
        # Static Python integers: no int() conversion needed
        level_start = level_offsets[level]
        level_end = level_offsets[level + 1]

        if level == 0 and legacy_level0_variance:
            # Legacy level 0 (the original standard path): one scalar per genome over
            # level 0's own four weights, so the level subdivides all-or-nothing.
            level_weights = all_weights[:, level_start:level_end]
            variance = jnp.var(level_weights, axis=1, keepdims=True)
            variances_list.append(variance)  # [pop_size, 1]
        else:
            # Canonical, level 0 included: variance of the 2x2 child blocks at level+1
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
    initial_depth: int = 0,
    return_all_masks: bool = True,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
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
        initial_depth: Force full subdivision through this level (static
            Python int; 0 forces nothing, which gives exactly the variance-only mask)
        return_all_masks: If False, return masks_A only, as compute_subdivision_masks_batch does

    Returns:
        (masks_A, masks_B, masks_C) - each [pop_size, total_positions] boolean; masks_A alone when
        return_all_masks is False
    """
    pop_size = level_variances[0].shape[0]

    # Track reached cells at each level
    level_reached = [jnp.ones((pop_size, 4), dtype=bool)]  # Level 0 always reached

    for level in range(1, num_levels):
        prev_reached = level_reached[level - 1]
        prev_variances = level_variances[level - 1]

        # Canonical ES-HyperNEAT subdivides on two clauses:
        #   (level < initial_depth) or (variance > threshold)
        # Forced when level <= initial_depth. Static branch, unrolls at trace time.
        if level <= initial_depth:
            prev_subdivided = prev_reached
        else:
            # Cells subdivided if reached AND high variance
            prev_subdivided = prev_reached & (prev_variances > variance_threshold)

        # Current level reached if parent was subdivided
        parent_indices = parent_indices_tuple[level]
        # Gather parent subdivision status for each cell
        current_reached = prev_subdivided[:, parent_indices]

        level_reached.append(current_reached)

    # Build masks using static indices
    masks_A = jnp.concatenate(level_reached, axis=1)
    if not return_all_masks:
        return masks_A

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


# Population-Level Batch CPPN Pre-Query

# Per-device JIT function cache to ensure separate traces per device.
# _create_query_fn_for_device binds cppn_forward in a closure and jits it for one device,
# avoiding cross-device trace contamination. Each device gets its own JIT trace.
_per_device_vmap_cache = {}
_per_device_cache_lock = threading.Lock()


def _create_query_fn_for_device(cppn_forward: Any, device_id: int):
    """Create a fresh query function bound to a specific device.

    This creates a new function each time, ensuring no trace contamination
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
        # Without device=, the JIT'd function can be reused across devices
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
    """Query all CPPNs at all positions using device-aware vmap.

    This uses per-device JIT-compiled functions to avoid trace sharing
    between devices in multi-GPU ThreadPoolExecutor execution.

    It moves all inputs to the target device before calling the query function. This
    handles cases where arrays might have been created on a different device.

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

    # Explicitly move all inputs to the target device
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


def cppn_query_rows(
    source_coord: Any,
    target_positions: Any,
    outgoing: bool,
    geometry_seeding_enabled: bool = False,
    agent_coord: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The CPPN input rows for one source coordinate and every target position.

    This is the single definition of EMR's query layout: every query helper builds
    its rows here, and `emr_hyperneat.substrate.substrate_queries` hands the same
    rows to consumers that evaluate CPPNs themselves.

    Column order:
      - 5-D: ``[x1, y1, x2, y2, 1]``, source first when ``outgoing``, target first
        otherwise. The trailing column is a literal one.
      - 7-D (geometry seeding): ``[x1, y1, x2, y2, dx, dy, 1]`` with the signed
        deltas before the bias, ``dx = x2 - x1``.
      - Swarm mode prepends ``[agent_x, agent_y]``.

    Built with numpy rather than jnp: jnp.ones/tile use cached JIT traces that cause
    cross-device errors in multi-GPU runs, and numpy arrays do not.

    Args:
        source_coord: One source coordinate (2,).
        target_positions: Target positions (num_positions, 2).
        outgoing: True for source -> target, False for target -> source.
        geometry_seeding_enabled: Add the delta columns (7-D rows).
        agent_coord: Optional (2,) agent team position for swarm mode.

    Returns:
        (num_positions, input_dim) rows in the coordinates' dtype (float32 in EMR).
    """
    num_positions = target_positions.shape[0]
    source_np = np.tile(np.asarray(source_coord)[None, :], (num_positions, 1))
    bias_np = np.ones((num_positions, 1), dtype=np.float32)
    target_np = np.asarray(target_positions)

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

    # Swarm mode: prepend agent team position to CPPN inputs
    # Extends input from [x1, y1, x2, y2, bias] to [agent_x, agent_y, x1, y1, x2, y2, bias]
    if agent_coord is not None:
        agent_np = np.tile(np.asarray(agent_coord)[None, :], (num_positions, 1))
        inputs_np = np.concatenate([agent_np, inputs_np], axis=1)

    return inputs_np


def batch_query_population_positions(
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
    """Query all CPPNs at all target positions in one vmap call.

    This is the core optimization: instead of 1000 sequential calls
    (one per genome), we perform 1 batched call with double vmap.

    Memory usage: pop_size x num_positions x 4 bytes
    - 1000 x 1024 x 4 = ~4 MB (negligible)

    The device_id parameter selects a device-specific query function
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
        agent_coord: Optional (2,) agent team position for swarm mode.
            When provided, prepends [agent_x, agent_y] to CPPN inputs.

    Returns:
        (pop_size, num_positions) array of CPPN outputs
    """
    # Get target device and ensure all inputs are on correct device
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Ensure input arrays are on correct device before building CPPN inputs
    source_coord = jax.device_put(source_coord, device)
    target_positions = jax.device_put(target_positions, device)

    # Build CPPN inputs: [x1, y1, x2, y2, bias=1.0] or [x1, y1, x2, y2, delta_x, delta_y, bias=1.0].
    # Rows are built with numpy, then device_put to the target device (see cppn_query_rows).
    inputs_np = cppn_query_rows(
        source_coord, target_positions, outgoing, geometry_seeding_enabled, agent_coord
    )
    inputs = jax.device_put(inputs_np, device)

    # Use device-aware vmap with per-device JIT cache.
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
    agent_coord: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    """Query all CPPNs from all source coords to all target positions, in population chunks.

    Processes population in chunks to reduce peak memory.
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
        agent_coord: Optional (2,) agent team position for swarm mode.
            When provided, prepends [agent_x, agent_y] to CPPN inputs.

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    # Get target device for device-consistent operations
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
                outgoing, cppn_forward, device_id, geometry_seeding_enabled,
                agent_coord=agent_coord,
            )
            # Ensure chunk result is on correct device before appending
            chunk_weights = jax.device_put(chunk_weights, device)
            chunk_results.append(chunk_weights)

        # Use numpy concatenate to avoid JAX JIT cache cross-device errors.
        # jnp.concatenate uses globally cached JIT traces.
        chunk_results_np = [np.asarray(c) for c in chunk_results]
        source_weights = np.concatenate(chunk_results_np, axis=0)
        results_list.append(source_weights)

    # Stack sources: (num_sources, pop_size, num_positions)
    # Transpose to: (pop_size, num_sources, num_positions)
    # Use numpy for stack/transpose to avoid JAX JIT cache cross-device errors.
    # jnp.stack and jnp.transpose use globally cached JIT traces that can cause
    # "Buffer on device cuda:X but replica assigned to cuda:Y" errors.
    results_np = [np.asarray(r) for r in results_list]
    result_np = np.stack(results_np, axis=0)
    result_np = np.transpose(result_np, (1, 0, 2))
    return jax.device_put(result_np, device)


# The CPPN node count the query chunking assumes for the CPPN's forward carry when the caller has
# no transformed CPPN at hand (EMRHyperNEAT._calculate_optimal_pos_chunk_size)
DEFAULT_MAX_NODES = 50

#: Top-level keys of a configuration that would be loaded from a file
FILE_CONFIGURATION_KEYS = ('preset', 'config_file', 'overrides')


def refuse_file_configuration(params: Dict[str, Any]) -> None:
    """Raise on a top-level ``preset``, ``config_file`` or ``overrides`` that has a value.

    The three keys asked the vendored ConfigManager to load the configuration from a file and
    apply overrides to it, and no such file ships with the package: the load raised ConfigError
    after two warnings, in every release. Set to None or empty they did nothing, and still do.

    Args:
        params: The configuration passed to ``create_config``.

    Raises:
        ValueError: One of the keys has a value.
    """
    given = [key for key in FILE_CONFIGURATION_KEYS if params.get(key)]
    if given:
        verb = 'asks' if len(given) == 1 else 'ask'
        raise ValueError(
            f"top-level {' and '.join(given)} {verb} for a configuration loaded from a file, and "
            "none ships with emr_hyperneat: pass the whole configuration as a dict "
            "(docs/configuration.md); a recurrence preset is "
            "algorithm_params.emrhyperneat.emr_hyperneat.recurrence.preset")


# Integration - EMRHyperNEATChunking Class

class EMRHyperNEATChunking(BaseAlgorithm):
    """The base class beneath EMRHyperNEAT: the base settings and the TensorNEAT setup.

    create_config reads the base settings of the emr_hyperneat section and builds the CPPN's NEAT
    algorithm; initialize sets up the TensorNEAT pipeline, compiles ask, transform and tell, and
    caches the problem's data and the substrate coordinates. EMRHyperNEAT runs the generations:
    run_generation here only implements BaseAlgorithm's abstract method.

    Configuration:
    - population_chunk_size: -1 (auto, default), 0 (no chunking), or manual value, which
      EMRHyperNEAT's CPPN queries read
    """

    #: Whether this class's substrate discovery consults `division_threshold`.
    #: False for the whole EMR tree: canonical ES-HyperNEAT subdivides on
    #: division_threshold and prunes on variance_threshold, but the eager
    #: multi-resolution path subdivides on variance_threshold and nothing here
    #: reads division_threshold. create_config warns rather than storing a value
    #: that does nothing.
    _SUBDIVISION_READS_DIVISION_THRESHOLD = False


    def __init__(self, name: str = 'emr-hyperneat',
                 implementation: str = 'tensorneat-emrhyperneat-adaptive'):
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
        # Cached JAX arrays for coordinates (avoid per-generation conversion)
        self._cached_input_coords = None
        self._cached_output_coords = None


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
        self.verbose = False

        # Execution mode parameters (set via create_config from emr_hyperneat section)
        self.sparse_forward_threshold = 0  # Default: always use sparse
        self.extra_randkey_split = True    # Default: adds extra random key split

    # Configuration

    def create_config(self, params: Dict[str, Any]) -> Any:
        """Create NEAT configuration for CPPN evolution."""
        refuse_file_configuration(params)
        hierarchical_config = params

        self._config_metadata = hierarchical_config

        algo_params = hierarchical_config.get('algorithm_params', {}).get('emrhyperneat', {})
        if not algo_params:
            algo_params = hierarchical_config

        hmr_config = algo_params.get('emr_hyperneat', {})
        if not hmr_config and (hierarchical_config.get('algorithm_params', {}).get('hmrhyperneat') is not None
                               or 'hmr_hyperneat' in algo_params):
            # An HMR-shaped configuration: EMR reads none of it, and a warning alone (as
            # below) would let the run go ahead with every setting at its default.
            raise ValueError(
                "this configuration is shaped for the HMR engine (release 0.1.0: "
                "algorithm_params.hmrhyperneat or hmr_hyperneat); EMR reads "
                "algorithm_params.emrhyperneat.emr_hyperneat")
        if not hmr_config:
            # Every setting below reads from this block. Absent, they all take their
            # defaults and the run proceeds looking healthy, which is exactly what an
            # HMR-shaped or misnested config produces: max_depth 1, canonical variance
            # rule, standard execution, dynamic functions off.
            import warnings
            warnings.warn(
                "No 'emr_hyperneat' configuration block found under "
                "algorithm_params.emrhyperneat; every algorithm setting (max_depth, "
                "variance_threshold, execution mode, dynamic functions) is taking its "
                "default. Check the nesting of your config.",
                UserWarning,
                stacklevel=2,
            )
        self.initial_depth = hmr_config.get('initial_depth', 0)
        # Level-0 subdivision variance rule. 'canonical' (default): per-parent variance
        # over the four level-1 children, which is the rule the papers describe, the
        # one every other level uses, and the one streaming/depth_offload always used.
        # 'legacy': the original standard-path rule, one scalar over level 0's own four
        # weights subdividing the level all-or-nothing, which every published
        # standard-path run used; pin it to reproduce those numbers bit for bit.
        self.variance_rule = hmr_config.get('variance_rule', 'canonical')
        if self.variance_rule not in ('canonical', 'legacy'):
            raise ValueError(
                f"variance_rule must be 'canonical' or 'legacy', got {self.variance_rule!r}")
        self._legacy_level0_variance = self.variance_rule == 'legacy'
        if hmr_config.get('verbose', False):
            print(f"[EMR-HyperNEAT] variance_rule={self.variance_rule}")
        self.max_depth = hmr_config.get('max_depth', 1)
        self.variance_threshold = hmr_config.get('variance_threshold', 0.03)
        self.division_threshold = hmr_config.get('division_threshold', 0.5)
        self.band_threshold = hmr_config.get('band_threshold', 0.3)
        if self.band_threshold < 0:
            # A negative band makes every zero cell a connection: the padded slots of the per-genome
            # buckets and the positions outside a genome's active set (1,544 zero-weight duplicates
            # per_genome, 129,684 outside-set connections union, on XOR, depth 3, population 30, at
            # band -0.01).
            raise ValueError(f"band_threshold must be >= 0, got {self.band_threshold}")
        self.max_weight = hmr_config.get('max_weight', 8.0)
        self.iteration_level = hmr_config.get('iteration_level', 1)
        # A top-level iteration_level is never read: hidden->hidden discovery and its level are
        # configured in the emr_hyperneat.recurrence block.
        if 'iteration_level' in hmr_config and hmr_config['iteration_level'] != 1:
            import warnings
            warnings.warn(
                f"emr_hyperneat.iteration_level={self.iteration_level} has no effect; "
                "hidden->hidden connections and their iteration level are configured in "
                "emr_hyperneat.recurrence (e.g. recurrence.iteration_level).",
                DeprecationWarning,
                stacklevel=2
            )
        # See _SUBDIVISION_READS_DIVISION_THRESHOLD: nothing on this path reads it.
        if ('division_threshold' in hmr_config
                and not self._SUBDIVISION_READS_DIVISION_THRESHOLD):
            import warnings
            warnings.warn(
                f"division_threshold={self.division_threshold} has no effect in "
                f"{type(self).__name__}. The eager multi-resolution path subdivides on "
                f"variance_threshold={self.variance_threshold} instead; canonical "
                "ES-HyperNEAT uses division_threshold for subdivision and "
                "variance_threshold only for pruning. Set variance_threshold, or "
                "initial_depth to force subdivision to a given level.",
                DeprecationWarning,
                stacklevel=2
            )
        self.verbose = hmr_config.get('verbose', False)
        self.fitness_mode = hmr_config.get('fitness_mode', 'mse')

        # HMR's base-engine memory and speed settings: sparse_forward_threshold (-1 dense, 0
        # sparse, above 0 sparse beyond that many positions), and below fuse_w1_computation and
        # skip_unused_masks. EMRHyperNEAT.create_config checks them, and nothing reads them: it
        # computes the reached-position mask alone and narrows the evaluation by compaction, so no
        # value changes a result.
        self.sparse_forward_threshold = hmr_config.get('sparse_forward_threshold', 0)

        # extra_randkey_split: Controls pre-tell() random key splitting behavior
        #   True (default): Adds extra split for different evolutionary trajectories
        #   False: Match EvoX adaptor behavior exactly
        self.extra_randkey_split = hmr_config.get('extra_randkey_split', True)

        # fuse_w1_computation and skip_unused_masks: see sparse_forward_threshold above
        self.fuse_w1_computation = hmr_config.get('fuse_w1_computation', False)
        self.skip_unused_masks = hmr_config.get('skip_unused_masks', False)

        # population_chunk_size: Process population in chunks during CPPN queries
        #   -1: AUTO - chunk only when the query's carry would not fit the shard budget (default)
        #   0: No chunking - vmap entire population at once
        #   >0: Manual override - process in chunks of this size
        self.population_chunk_size = hmr_config.get('population_chunk_size', -1)  # Default: auto

        substrate_section = algo_params.get('substrate', {})
        self.substrate_input_coords = substrate_section.get('input_coords', [])
        self.substrate_output_coords = substrate_section.get('output_coords', [])
        default_activate_time = (2 ** self.max_depth) + 1
        self.activate_time = substrate_section.get('activate_time', default_activate_time)
        # The defaults of dynamic_functions.hidden_activation and output_activation, as in HMR's
        # dynamic-functions variant (EMRHyperNEAT._parse_dynamic_functions_config).
        self.hidden_activation = substrate_section.get('hidden_activation', 'tanh')
        self.output_activation = substrate_section.get('output_activation', 'sigmoid')

        # The DenseQuadtreeStructure is not built here: it is the ES-HyperNEAT-lineage
        # structure, and only the node-list model reads it (get_quadtree_structure, at that
        # model's depth). The multi-resolution paths (depth_offload / streaming) use
        # HierarchicalGridStructure via get_hierarchical_grid() and never touch it, so building
        # it here would charge every EMR run for a structure it never reads: ~1.5 GB of arrays
        # at max_depth=13 ((4^14-1)/3 = 89,478,485 nodes), retained in _QUADTREE_CACHE for the
        # process lifetime. It would also emit a "creates N quadtree positions" warning into EMR
        # logs, which reads as though the multi-resolution path uses a quadtree.

        # Parse swarm mode configuration (multi-agent policy geometry)
        # When enabled, CPPN gets 2 extra inputs (agent_x, agent_y) prepended
        # to generate position-aware policies for each agent in a team
        swarm_section = algo_params.get('swarm', {})
        self.swarm_mode = swarm_section.get('enabled', False)
        self.swarm_num_agents = swarm_section.get('num_agents', 1)
        self.swarm_team_layout = swarm_section.get('team_layout', 'line')

        # Parse NEAT configuration section
        # Config can be: {'neat': {'pop_size': 1000}} or {'population_size': 1000}
        neat_section = algo_params.get('neat', {})
        population_size = neat_section.get('pop_size',
                                           neat_section.get('population_size',
                                                           algo_params.get('population_size', 150)))
        # Retained for memory-aware execution-mode resolution (the AUTO gate
        # needs pop x positions before any arrays exist).
        self.population_size = population_size

        # Parse CPPN configuration (allows overriding genome params like num_inputs)
        cppn_section = algo_params.get('cppn', {})
        cppn_genome_config = cppn_section.get('genome', {})

        # Build NEAT config
        # Default num_inputs=5 for [x1, y1, x2, y2, bias]
        # With geometry seeding: num_inputs=7 for [x1, y1, x2, y2, delta_x, delta_y, bias]
        # With swarm mode: +2 for [agent_x, agent_y, ...] prepended
        default_num_inputs = 5
        if self.swarm_mode:
            default_num_inputs += 2  # agent_x, agent_y prepended
        flat_params = {
            'genome': {
                'num_inputs': cppn_genome_config.get('num_inputs', default_num_inputs),
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
            # NEAT operator settings from neat_mutation, neat_species and neat_selection, over
            # the defaults below, which are the values 0.1.0 applied to every run whatever a
            # configuration asked for.
            'mutation': _neat_operator_block(algo_params, 'neat_mutation', {
                'conn_add_prob': 0.5, 'conn_delete_prob': 0.5,
                'node_add_prob': 0.2, 'node_delete_prob': 0.2,
            }),
            'species': _neat_operator_block(algo_params, 'neat_species', {
                'compatibility_threshold': 3.0,
                'max_stagnation': 20, 'species_elitism': 15,
            }),
            'selection': _neat_operator_block(algo_params, 'neat_selection', {
                'genome_elitism': 15, 'survival_threshold': 0.2,
            }),
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
                'default': act_default, 'options': act_options, 'mutate_rate': act_mutate,
            }
            flat_params['activation_options'] = act_options
            flat_params['activation_default'] = act_default
            print(f"[CPPN] Custom activation: options={act_options}, "
                  f"default={act_default}, mutate_rate={act_mutate}")

        self.neat_algo = self.adapter.build_neat_config(flat_params)
        # Under JAX's x64 mode TensorNEAT's species update traces lax.cond branches of different
        # dtypes and stops at the first tell; this gives its species module a dtype-unifying cond
        # (x64_compat.py). Under float32 it does nothing.
        from emr_hyperneat.x64_compat import ensure_x64_species_dtypes
        ensure_x64_species_dtypes()
        self._jitted_cppn_forward = jax.jit(
            self.neat_algo.genome.forward, static_argnums=(0,)
        )

        from tensorneat.algorithm.hyperneat.hyperneat import HyperNEATNode, HyperNEATConn
        from tensorneat.genome import RecurrentGenome
        from tensorneat.common import ACT, AGG, State

        n_in = len(self.substrate_input_coords)
        n_out = len(self.substrate_output_coords)
        min_conns = n_in * n_out  # fully-connected initial layer
        dynamic_max_conns = max(2000, int(min_conns * 1.2))
        dynamic_max_nodes = max(500, n_in + n_out + 100)

        self.hyper_genome = RecurrentGenome(
            num_inputs=n_in,
            num_outputs=n_out,
            max_nodes=dynamic_max_nodes, max_conns=dynamic_max_conns,
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

        self._compiled_ask = jax.jit(self.neat_algo.ask)
        self._compiled_transform_batch = jax.jit(
            jax.vmap(self.neat_algo.transform, in_axes=(None, (0, 0)))
        )
        self._compiled_tell = jax.jit(self.neat_algo.tell)

        # Cache problem data to avoid repeated list comprehension + array conversion
        data = problem.get_data()
        self._cached_inputs = jnp.array([d[0] for d in data], dtype=jnp.float32)
        self._cached_targets = jnp.array([d[1] for d in data], dtype=jnp.float32)

        # Cache coordinate arrays (avoid per-generation jnp.array conversion)
        self._cached_input_coords = jnp.array(self.substrate_input_coords, dtype=jnp.float32)
        self._cached_output_coords = jnp.array(self.substrate_output_coords, dtype=jnp.float32)

        return state

    def _wrap_problem_for_pipeline(self, problem: Any) -> Any:
        """Wrap problem for TensorNEAT pipeline compatibility."""
        # Determine CPPN input dimension:
        # Base: 5 [x1, y1, x2, y2, bias]
        # +2 with geometry seeding [delta_x, delta_y]
        # +2 with swarm mode [agent_x, agent_y] prepended
        cppn_input_dim = 5
        if getattr(self, 'geometry_seeding_enabled', False):
            cppn_input_dim += 2
        if getattr(self, 'swarm_mode', False):
            cppn_input_dim += 2

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

    def run_generation(self, state: Any, problem: Any) -> Tuple[Any, Any]:
        """Not run on this class: EMRHyperNEAT overrides it. It implements BaseAlgorithm's abstract
        method so that EMRHyperNEATMultiGPU can be constructed.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("run_generation runs on EMRHyperNEAT")

    # BaseAlgorithm Abstract Methods

    def evaluate_genome(self, genome: Any, problem: Any) -> float:
        """Required by the abstract base; EMR-HyperNEAT evaluates whole populations and returns 0.0
        here. `evaluate_genome_on_problem` scores one genome on a problem."""
        return 0.0

    def extract_network_info(self, state: Any) -> Any:
        """Required by the abstract base and not used by EMR-HyperNEAT: returns None."""
        return None

    def genome_to_phenotype(self, genome: Any) -> Any:
        """Required by the abstract base and not used by EMR-HyperNEAT: returns None.
        `population_substrates` builds the networks a population evaluates."""
        return None

    def get_best_genome(self, state: Any) -> Tuple[Any, Any]:
        """The champion of the last generation evaluated: its CPPN genes ``(nodes, conns)``.

        The genome `get_best_genome_with_fitness` returns, from the record run_generation keeps;
        ``state`` is not read (the abstract base's signature). Release 0.1.0 returned what
        TensorNEAT's `ask` returns first for ``state``: the whole population's node array.

        Args:
            state: Unused.

        Returns:
            ``(nodes, conns)``.

        Raises:
            RuntimeError: no generation has been evaluated.
        """
        record = getattr(self, '_last_eval', None)
        if record is None:
            raise RuntimeError(
                "no evaluated generation recorded a champion: run run_generation first (of "
                "run_until_threshold's multi-device routes, only the split routes and the island "
                "model record one)"
            )
        nodes, conns = record['champion_genome']
        return nodes, conns


# The Strategy Class Beneath EMRHyperNEAT

class EMRHyperNEATMultiGPU(EMRHyperNEATChunking):
    """The strategy beneath EMRHyperNEAT, and the refusal of fixed h->h runs.

    This class stores the strategy name (`self.strategy`, SINGLE_GPU as BASELINE) and refuses
    run_until_threshold_with_fixed_hh and its recurrence_config argument. It runs no multi-device
    route of its own: EMRHyperNEAT routes every MultiGPUStrategy name in run_until_threshold
    (STRATEGY_NAME_MAP, in engine/parallel.py), and replaces the strategy stored here with the route
    it chose.
    """

    def __init__(
        self,
        name: str = 'emr-hyperneat',
        implementation: str = 'tensorneat-emrhyperneat-multi-gpu',
        strategy: MultiGPUStrategy = MultiGPUStrategy.SINGLE_GPU,
        recurrence_config: Optional[RecurrenceConfig] = None,
    ):
        """Initialize multi-GPU EMR-HyperNEAT.

        Args:
            name: Algorithm name
            implementation: Implementation identifier
            strategy: The strategy name, stored as given (SINGLE_GPU as BASELINE)
            recurrence_config: Refused when given. It configured only
                run_until_threshold_with_fixed_hh, which is refused; recurrence is the
                configuration's ``emr_hyperneat.recurrence`` block.

        Raises:
            ValueError: recurrence_config is given.
        """
        if recurrence_config is not None:
            raise ValueError(
                "recurrence_config configured only run_until_threshold_with_fixed_hh, which is "
                "refused (it scored genomes with hidden-to-hidden connections discovered for other "
                "genomes); set recurrence in the configuration's emr_hyperneat.recurrence block "
                "instead.")
        super().__init__(name=name, implementation=implementation)

        # SINGLE_GPU is the base class's BASELINE
        if strategy == MultiGPUStrategy.SINGLE_GPU:
            strategy = MultiGPUStrategy.BASELINE
        self.strategy = strategy

    # Fixed h->h runs: refused

    def run_until_threshold_with_fixed_hh(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Refused: a fixed h->h run scores genomes with connections discovered for other genomes.

        Such a runner discovers every population slot's hidden-to-hidden connections once, from
        the generation-0 population, and keeps them for the whole run. After the first
        reproduction a slot holds a different genome, which is then scored with the
        connections discovered for the genome that held the slot at generation 0, so its
        fitness describes a network it does not encode. Configure recurrence in the
        ``emr_hyperneat.recurrence`` block instead: `run_generation` and `run_until_threshold`
        then discover each genome's own connections every generation.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "run_until_threshold_with_fixed_hh is refused: it discovered the hidden-to-hidden "
            "connections once, from the generation-0 population, and kept each population slot's "
            "set for the whole run, so after the first reproduction a genome was scored with the "
            "connections discovered for another genome. Configure recurrence in the "
            "emr_hyperneat.recurrence block and call run_generation or run_until_threshold, which "
            "discover each genome's own connections every generation.")
