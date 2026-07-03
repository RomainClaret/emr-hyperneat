"""HMR-HyperNEAT (Hierarchical Multi-Resolution HyperNEAT) TensorNEAT implementation.

HMR-HyperNEAT is a GPU-optimized variant of ES-HyperNEAT that achieves adaptive
resolution through pre-computed hierarchical grids and variance-based masking,
enabling efficient batch processing across entire populations.

Key Features:
1. Pre-computed hierarchical grid instead of dynamic quadtree
2. Batch processing via JAX vmap instead of sequential per-genome discovery
3. Fixed position sets with variance-based masking instead of variable topology
4. GPU-resident execution option with single GPU↔CPU sync

API
===

**Execution Modes:**

1. Single Generation: `run_generation(state, problem)`
   - Uses Python loop implementation (faster for single generations)
   - Detailed per-step timing available
   - Recommended for step-by-step execution

2. Verbose Mode: `run_generation_verbose(state, problem)`
   - Same as run_generation() with explicit verbose naming
   - Python loop with per-step timing instrumentation
   - Useful for debugging and profiling

3. GPU-Resident Multi-Generation: `run_until_threshold(state, problem, target_fitness, max_generations)`
   - GPU-resident loop via jax.lax.while_loop
   - Single GPU↔CPU sync at the end
   - **RECOMMENDED for production runs** - significantly faster for multi-generation evolution
   - Best for runs targeting a fitness threshold with early stopping

Benchmark Results (XOR, pop=1000)
=================================

max_depth=2 (84 positions):
- 100% solve rate (5/5 seeds)
- ~510ms/gen on Apple Silicon M4
- ~44 generations to solve

max_depth=1 (20 positions):
- 100% solve rate
- ~400ms/gen on Apple Silicon M4
- ~35 generations to solve

Configuration
=============

Key parameters in hmr_hyperneat config:
- initial_depth: Starting resolution (default: 0)
- max_depth: Maximum subdivision depth (1-3 recommended)
- variance_threshold: Threshold for position activation (default: 0.03)
- division_threshold: Threshold for quadtree subdivision
- band_threshold: Connection band threshold
- max_weight: Maximum connection weight
- iteration_level: DEPRECATED - ignored in this implementation (see Architecture Note)

Architecture Limitation
======================

This optimized implementation uses a SIMPLIFIED feedforward architecture:
- Input → Hidden → Output (no hidden→hidden connections)
- Forward pass: hidden = tanh(inputs @ W1), outputs = sigmoid(hidden @ W2)

Original ES-HyperNEAT (PUREPLES) creates THREE connection types:
1. connections1: Input → Hidden (from exploring inputs via quadtree)
2. connections2: Hidden → Hidden (from `iteration_level` iterations exploring FROM hidden)
3. connections3: Hidden → Output (from exploring to outputs)

This optimized version OMITS hidden→hidden connections (connections2) entirely.
The `iteration_level` parameter controls how many rounds of hidden→hidden
discovery occur in the original algorithm - here it is ignored because:
1. Hidden→hidden requires iterative/recurrent forward propagation
2. Iterative propagation breaks JAX vmap vectorization across population
3. Simple feedforward (W1, W2 only) enables efficient parallel evaluation

Consequence: `iteration_level` parameter has NO EFFECT in this implementation.
For problems requiring hidden→hidden connections (deeper compositional reasoning),
use the PUREPLES-based ES-HyperNEAT implementation instead.
"""

import functools
import time
import copy
import math
import os
import numpy as np
from typing import Any, Dict, Tuple, Set, List, Optional, NamedTuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import jax.numpy as jnp
from jax import lax

from emr_hyperneat._compat.core.base_algorithm import BaseAlgorithm, AlgorithmMetrics
from emr_hyperneat._compat.utils.config_manager import ConfigManager
from emr_hyperneat._compat.adapters.tensorneat_adapter import TensorNEATAdapter


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
    finest_size = int(grid.level_offsets[grid.num_levels]) - int(grid.level_offsets[grid.num_levels - 1])
    level_variances_batch.append(jnp.zeros((pop_size, finest_size)))

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

def batch_query_population_positions(
    state: Any,
    cppns_transformed: Tuple,
    source_coord: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
) -> jnp.ndarray:
    """Query ALL CPPNs at ALL target positions in ONE vmap call.

    This is the core optimization: instead of 1000 sequential calls
    (one per genome), we perform 1 batched call with double vmap.

    Memory usage: pop_size × num_positions × 4 bytes
    - 1000 × 1024 × 4 = ~4 MB (negligible)

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
    # Need to vmap over the tuple elements (nodes, conns, conn_attrs, node_attrs)
    all_weights = jax.vmap(
        query_single_cppn,
        in_axes=((0, 0, 0, 0),)  # vmap over first axis of each tuple element
    )((cppns_transformed[0], cppns_transformed[1],
       cppns_transformed[2], cppns_transformed[3]))

    return all_weights  # (pop_size, num_positions)


def batch_query_population_multi_source(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
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

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
    """
    # vmap over sources
    def query_from_source(source_coord):
        return batch_query_population_positions(
            state, cppns_transformed, source_coord, target_positions,
            outgoing, cppn_forward
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

    Returns:
        (pop_size, num_sources, num_positions) array of CPPN outputs
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

            # Query chunk (double vmap: chunk_pop x positions)
            chunk_weights = batch_query_population_positions(
                state, chunk_cppns, source_coord, target_positions,
                outgoing, cppn_forward
            )
            chunk_results.append(chunk_weights)

        # Concatenate chunks for this source
        source_weights = jnp.concatenate(chunk_results, axis=0)
        results_list.append(source_weights)

    # Stack sources: (num_sources, pop_size, num_positions)
    # Transpose to: (pop_size, num_sources, num_positions)
    result = jnp.stack(results_list, axis=0)
    return jnp.transpose(result, (1, 0, 2))


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
# Integration - HMRHyperNEAT Class
# ============================================================================

class HMRHyperNEAT(BaseAlgorithm):
    """HMR-HyperNEAT (Hierarchical Multi-Resolution HyperNEAT) with full JAX/GPU substrate discovery.

    This implementation achieves ES-HyperNEAT's semantic goal (adaptive resolution) through:
    - Pre-computed hierarchical grid instead of dynamic quadtree
    - Batch processing via JAX vmap instead of sequential per-genome discovery
    - Fixed position sets with variance-based masking instead of variable topology
    - GPU-resident execution option with single GPU↔CPU sync

    Execution Modes:
    - run_generation(): GPU-resident single generation (DEFAULT, recommended)
    - run_generation_verbose(): Python loop with per-step timing (for debugging)
    - run_until_threshold(): GPU-resident multi-generation with early stopping
    """

    def __init__(self, name: str = 'hmr-hyperneat',
                 implementation: str = 'tensorneat-hmrhyperneat'):
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
        #   0 (default): No chunking - vmap entire population at once
        #   >0: Process in chunks of this size (reduces peak memory)
        self.population_chunk_size = hmr_config.get('population_chunk_size', 0)

        substrate_section = algo_params.get('substrate', {})
        self.substrate_input_coords = substrate_section.get('input_coords', [])
        self.substrate_output_coords = substrate_section.get('output_coords', [])
        self.output_activation = substrate_section.get('output_activation', 'sigmoid')
        self.hidden_activation = substrate_section.get('hidden_activation', 'tanh')
        default_activate_time = (2 ** self.max_depth) + 1
        self.activate_time = substrate_section.get('activate_time', default_activate_time)

        # Pre-compute quadtree structure
        self._quadtree = get_quadtree_structure(self.max_depth)

        # Build NEAT config
        flat_params = {
            'genome': {
                'num_inputs': 5,
                'num_outputs': 1,
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
            'population_size': algo_params.get('population_size', 150),
            'mutation': {
                'conn_add_prob': 0.5, 'conn_delete_prob': 0.5,
                'node_add_prob': 0.2, 'node_delete_prob': 0.2,
            },
            'species': {
                'compatibility_threshold': 3.0,
                'max_stagnation': 20, 'species_elitism': 15,
            },
            'selection': {
                'genome_elitism': 15, 'survival_threshold': 0.2,
            },
            'activation_options': ['tanh', 'sin', 'gauss'],
            'activation_default': 'tanh',
            'verbose': False,
        }

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
        class WrappedProblem:
            def __init__(self, inner_problem):
                self.inner = inner_problem
                self.input_shape = (5,)
                self.jitable = True

            def setup(self, state=None):
                from tensorneat.common import State
                return state if state else State()

            def evaluate(self, state, randkey, forward_func, transformed):
                return 0.0

        return WrappedProblem(problem)

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

        # MEMORY OPTIMIZATION: Use chunked query when population_chunk_size > 0
        # This reduces peak memory from 139+ GB to ~1-2 GB per chunk at depth 8
        if self.population_chunk_size > 0:
            query_func = lambda state, cppns, sources, targets, outgoing, fwd: \
                batch_query_population_multi_source_chunked(
                    state, cppns, sources, targets, outgoing, fwd,
                    pop_chunk_size=self.population_chunk_size
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
        all_weights_for_variance = batch_query_population_multi_source(
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
        input_all_weights = batch_query_population_multi_source(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward
        )
        output_all_weights = batch_query_population_multi_source(
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
