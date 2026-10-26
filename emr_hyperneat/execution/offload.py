"""Depth-offload execution: host-resident matrices, chunked device evaluation.

Beyond roughly depth 11 the weight matrices for a whole population do not fit in
device memory at all: depth 11 is 22.4M candidate positions, about 380 MB of
W1/W2/mask per genome. Depth offload assembles the matrices in host RAM (or a
memmap when even that is tight) and moves only one chunk at a time to the device
to be evaluated.

Why this is a whole generation rather than a step
------------------------------------------------
Streaming swaps a single step inside the normal generation loop, which is why it
composes with everything else the loop does. Depth offload cannot be factored that
way: never materializing the full matrices means its build and its evaluation are
fused, so it necessarily owns the generation. That asymmetry is inherent to the
feature, not an artifact of how it was written: any factoring keeps a separate
generation path here.

The consequence of owning the generation: a separate generation path does not
automatically pick up features added to the main loop. This path applies geometry
seeding itself and refuses what it does not apply, such as locality_radius
(`refuse_depth_offload`). Anything added to the standard loop has to be
considered here too.

Every function takes the algorithm instance as its first argument. That makes the
coupling explicit (these read 34 distinct attributes of it), and it is what
lets depth offload be selected by config on one class instead of requiring users
to instantiate a different subclass.

Import direction is one-way: this module imports from the algorithm module, and the
dispatcher imports this module lazily at call time so the cycle never forms.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.emrhyperneat_base import (
    HierarchicalGridStructureSingle,
    get_hierarchical_grid_cpu,
)
from emr_hyperneat.emrhyperneat import (
    AlgorithmMetrics,
    ForwardPassMode,
    HierarchicalGridStructure,
    batch_query_population_positions,
    compute_single_level_variance,
    compute_single_level_variance_chunked,
    discover_sparse_hh_vectorized_multi_hop,
    fitness_statistics,
    legacy_level0_variance,
    get_hierarchical_grid,
    safe_matmul,
    sanitize_fitness,
    species_summary,
)

from .chunking import population_chunk_for_query, recurrent_chunk_sizes
from .compaction import (
    compact_weights,
    find_active_positions,
    remap_sparse_hh,
)
from .streaming import evaluate_population_recurrent_double_chunked

__all__ = [
    "run_generation_offload",
    "query_level_positions_chunked",
    "process_level_streaming",
    "estimate_memory_requirements",
    "estimate_offload_feasibility",
]


# Level-wise helpers (pure: no algorithm state)

def query_level_positions_chunked(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    level_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    pop_chunk_size: int = 10,
    pos_chunk_size: int = 100_000,
    geometry_seeding_enabled: bool = False,
) -> jnp.ndarray:
    """Query CPPN for a level's positions with double chunking.

    This version chunks both population and positions to handle ultra-deep
    levels (4M+ positions) without OOM.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        level_positions: Positions for this level only (num_level_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Number of genomes to process at once
        pos_chunk_size: Number of positions to process at once

    Returns:
        (pop_size, num_sources, num_level_positions) array of CPPN outputs
    """
    pop_size = cppns_transformed[0].shape[0]
    num_sources = source_coords.shape[0]
    num_positions = level_positions.shape[0]

    results_list = []

    # Sequential loop over sources (typically 3-10 for XOR)
    for source_idx in range(num_sources):
        source_coord = source_coords[source_idx]

        # Double-chunked processing: population x positions
        all_source_results = []

        for pop_start in range(0, pop_size, pop_chunk_size):
            pop_end = min(pop_start + pop_chunk_size, pop_size)

            # Extract population chunk
            chunk_cppns = (
                cppns_transformed[0][pop_start:pop_end],
                cppns_transformed[1][pop_start:pop_end],
                cppns_transformed[2][pop_start:pop_end],
                cppns_transformed[3][pop_start:pop_end],
            )

            # Position chunking for large levels
            pos_chunks = []
            for pos_start in range(0, num_positions, pos_chunk_size):
                pos_end = min(pos_start + pos_chunk_size, num_positions)
                pos_chunk = level_positions[pos_start:pos_end]

                # Query this (pop_chunk, pos_chunk) combination
                chunk_weights = batch_query_population_positions(
                    state, chunk_cppns, source_coord, pos_chunk,
                    outgoing, cppn_forward,
                    geometry_seeding_enabled=geometry_seeding_enabled,
                )  # (chunk_pop_size, chunk_pos_size)
                pos_chunks.append(chunk_weights)

            # Concatenate position chunks: (chunk_pop_size, num_positions)
            pop_chunk_result = jnp.concatenate(pos_chunks, axis=1)
            all_source_results.append(pop_chunk_result)

        # Concatenate population chunks: (pop_size, num_positions)
        source_weights = jnp.concatenate(all_source_results, axis=0)
        results_list.append(source_weights)

    # Stack sources: (num_sources, pop_size, num_level_positions)
    # Transpose to: (pop_size, num_sources, num_level_positions)
    result = jnp.stack(results_list, axis=0)
    return jnp.transpose(result, (1, 0, 2))


def process_level_streaming(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    level_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    pop_chunk_size: int,
    pos_chunk_size: int,
    W_cpu: np.ndarray,
    level_start: int,
    max_weight: float,
    weight_thresh: float,
    reached_mask: np.ndarray,
    geometry_seeding_enabled: bool = False,
) -> int:
    """Process a level and stream results directly to CPU array.

    Instead of building full level result on GPU then copying, this processes
    position chunks and copies each chunk immediately to CPU.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        level_positions: Positions for this level only (num_level_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Number of genomes to process at once
        pos_chunk_size: Number of positions to process at once
        W_cpu: Pre-allocated CPU array to write results into (pop, sources, total_positions)
        level_start: Starting offset in W_cpu for this level
        max_weight: Maximum weight magnitude
        weight_thresh: Threshold for weight masking
        reached_mask: Boolean mask for active positions (pop, level_positions)

    Returns:
        Number of GPU-to-CPU transfers performed
    """
    pop_size = cppns_transformed[0].shape[0]
    num_sources = source_coords.shape[0]
    num_positions = level_positions.shape[0]
    transfers = 0

    # Process in position chunks and offload immediately
    for pos_start in range(0, num_positions, pos_chunk_size):
        pos_end = min(pos_start + pos_chunk_size, num_positions)
        pos_chunk = level_positions[pos_start:pos_end]

        # Get reached mask for this position chunk
        mask_chunk_cpu = reached_mask[:, pos_start:pos_end]
        mask_chunk_gpu = jax.device_put(mask_chunk_cpu)

        # Process each source separately to minimize GPU memory
        for source_idx in range(num_sources):
            source_coord = source_coords[source_idx]

            # Process population in chunks
            for pop_start_idx in range(0, pop_size, pop_chunk_size):
                pop_end_idx = min(pop_start_idx + pop_chunk_size, pop_size)

                chunk_cppns = (
                    cppns_transformed[0][pop_start_idx:pop_end_idx],
                    cppns_transformed[1][pop_start_idx:pop_end_idx],
                    cppns_transformed[2][pop_start_idx:pop_end_idx],
                    cppns_transformed[3][pop_start_idx:pop_end_idx],
                )

                # Query CPPN for this chunk
                chunk_weights = batch_query_population_positions(
                    state, chunk_cppns, source_coord, pos_chunk,
                    outgoing, cppn_forward,
                    geometry_seeding_enabled=geometry_seeding_enabled,
                )  # (chunk_pop_size, chunk_pos_size)

                # Apply weight processing
                W_raw = jnp.tanh(chunk_weights) * max_weight
                mask_slice = mask_chunk_gpu[pop_start_idx:pop_end_idx]
                W_processed = jnp.where(
                    mask_slice & (jnp.abs(W_raw) > weight_thresh),
                    W_raw,
                    0.0
                )

                # Immediately offload to CPU
                W_cpu[pop_start_idx:pop_end_idx, source_idx, level_start + pos_start:level_start + pos_end] = np.asarray(W_processed)
                transfers += 1

                # Free GPU memory
                del chunk_weights, W_raw, W_processed

        del mask_chunk_gpu

    return transfers


def estimate_memory_requirements(
    depth: int,
    pop_size: int,
    num_inputs: int = 3,
    num_outputs: int = 1,
    target_cpu_gb: float = 32.0,
) -> Dict[str, Any]:
    """Estimate memory requirements for a given configuration.

    Args:
        depth: Maximum depth (0-indexed)
        pop_size: Population size
        num_inputs: Number of input nodes
        num_outputs: Number of output nodes
        target_cpu_gb: Target CPU RAM for population chunking (default 32 GB)

    Returns:
        Dictionary with memory estimates in GB and chunking recommendations
    """
    # Position count formula
    total_positions = (4 ** (depth + 2) - 4) // 3

    # Level sizes
    level_sizes = [4 ** (l + 1) for l in range(depth + 1)]
    max_level_size = max(level_sizes)

    bytes_per_float = 4

    # GPU memory (with depth offloading)
    # Active level buffer (query result)
    active_level_gpu = pop_size * num_inputs * max_level_size * bytes_per_float
    # Variance lookahead (next level)
    next_level_gpu = pop_size * num_inputs * max_level_size * 4 * bytes_per_float  # 4x for next level
    # CPPN population (rough estimate)
    cppn_gpu = pop_size * 1000 * bytes_per_float  # ~1K params per CPPN

    gpu_peak_bytes = active_level_gpu + next_level_gpu + cppn_gpu

    # CPU memory (full accumulators)
    w1_cpu = pop_size * num_inputs * total_positions * bytes_per_float
    w2_cpu = pop_size * num_outputs * total_positions * bytes_per_float
    masks_cpu = pop_size * total_positions  # bool

    cpu_bytes = w1_cpu + w2_cpu + masks_cpu

    # Population chunking calculations
    bytes_per_genome = (
        (num_inputs + num_outputs) * total_positions * bytes_per_float +
        total_positions  # masks (bool)
    )
    target_bytes = target_cpu_gb * 1e9 * 0.9  # 10% safety margin
    pop_chunk_size = int(target_bytes / bytes_per_genome)
    pop_chunk_size = max(10, min(pop_chunk_size, pop_size))
    needs_chunking = pop_chunk_size < pop_size
    num_chunks = (pop_size + pop_chunk_size - 1) // pop_chunk_size if needs_chunking else 1

    return {
        'total_positions': total_positions,
        'max_level_size': max_level_size,
        'gpu_peak_gb': gpu_peak_bytes / 1e9,
        'cpu_required_gb': cpu_bytes / 1e9,
        'w1_size_gb': w1_cpu / 1e9,
        'w2_size_gb': w2_cpu / 1e9,
        # Population chunking
        'bytes_per_genome_gb': bytes_per_genome / 1e9,
        'needs_pop_chunking': needs_chunking,
        'pop_chunk_size': pop_chunk_size,
        'num_pop_chunks': num_chunks,
        'memory_per_chunk_gb': (pop_chunk_size * bytes_per_genome) / 1e9,
    }


# Generation strategy

def _evaluates_hidden_to_hidden(algo) -> bool:
    """Whether a generation discovers and settles hidden-to-hidden connections.

    The standard path's rule: the forward mode the configuration resolves to
    (`resolve_recurrence`, which follows ``allow_hidden_to_hidden``) and a recurrence
    configuration that allows the connections. It checks neither ``enabled`` nor
    ``iteration_level > 0``: the configuration refuses ``enabled: false`` with
    ``allow_hidden_to_hidden: true``, and iteration level 0 discovers direct connections
    only, as level 1 does, so this path settles them as the standard path does.
    """
    return (
        algo._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH
        and algo._recurrence_config is not None
        and algo._recurrence_config.allow_hidden_to_hidden
    )


def _refuse_hh_cache_on_population_chunks(algo, state) -> None:
    """Refuse the population-chunked loop with the h->h cache on, before the generation's work.

    The loop scores each chunk without the population-level cache manager
    (``consult_cache_manager=False``), so with ``recurrence.hh_cache_enabled`` on it would
    rediscover the hidden-to-hidden connections every generation where the unchunked loop reuses
    them, and the two would differ from generation 1 on without a word. The
    configuration alone does not decide the route: AUTO can resolve to depth offload at run time
    on a GPU, and the chunk size depends on the substrate's inputs and outputs. So the check runs
    here, at the route's entry, with the numbers the loop itself uses.

    Args:
        algo: The algorithm instance, after ``initialize``.
        state: The state the generation starts from.

    Raises:
        ValueError: The h->h cache is on and the population would be split into chunks.
    """
    cache = getattr(algo, '_hh_cache', None)
    if cache is None or not cache.config.hh_cache_enabled or not _evaluates_hidden_to_hidden(algo):
        return
    use_cpu_grid = algo._use_cpu_grid or algo.max_depth >= 12
    h_grid = (get_hierarchical_grid_cpu if use_cpu_grid else get_hierarchical_grid)(algo.max_depth)
    pop_size = int(state.pop_nodes.shape[0])
    chunk = algo._calculate_population_chunk_size(
        pop_size, h_grid.total_positions,
        algo._cached_input_coords.shape[0], algo._cached_output_coords.shape[0],
    )
    if chunk < pop_size:
        raise ValueError(
            f"depth offload would split the population of {pop_size} into chunks of {chunk} "
            f"genomes (execution.max_population_chunk, or the weight matrices above "
            f"execution.target_cpu_gb), and that loop does not consult the population-level "
            f"h->h cache manager: with recurrence.hh_cache_enabled on it would rediscover the "
            f"hidden-to-hidden connections every generation and silently differ from the "
            f"unchunked loop. Run depth offload unchunked (execution.max_population_chunk at "
            f"least the population, or a larger execution.target_cpu_gb), or turn the cache "
            f"off (recurrence.hh_cache_enabled: false, the default)."
        )


def run_generation_offload(
    algo,
    state,
    problem,
    skip_metrics: bool = False,
    cached_sparse_hh: "Optional[Any]" = None,
) -> Tuple[Any, AlgorithmMetrics]:
    """Run one generation with depth-wise CPU offloading.

    This processes CPPN queries one level at a time on GPU, immediately
    offloading completed levels to CPU RAM. This enables depths 11+
    that would otherwise OOM.

    Memory Pattern:
    - GPU: Only holds active level + next level (for variance)
    - CPU: Accumulates all completed levels
    - Transfer: Level weights GPU->CPU after each level
    - Evaluation: Chunks streamed CPU->GPU

    Args:
        state: Current algorithm state
        problem: Problem instance with get_data() method
        skip_metrics: If True, skip detailed metric collection

    Returns:
        Tuple of (new_state, metrics)
    """
    # On a GPU, execution.mode 'auto' can select this path at run time
    algo.refuse_depth_offload()
    _refuse_hh_cache_on_population_chunks(algo, state)
    step_timings = {}
    algo._gpu_transfers_count = 0
    # This path records no champion, so evaluate_on_data refuses rather than score an
    # earlier generation's
    algo._last_eval = None
    # Execution mode is a memory choice and must draw the keys the standard path draws: the
    # pre-ask split only under extra_randkey_split, as run_generation_verbose does.
    if getattr(algo, '_repro_extra_randkey_split', False):
        _, randkey = jax.random.split(state.randkey)
        state = state.update(randkey=randkey)

    # Step 0: CPPN ask + transform (same as standard)
    t0 = time.perf_counter()

    cppn_population = algo._compiled_ask(state)
    pop_size = cppn_population[0].shape[0]
    cppns_transformed = algo._compiled_transform_batch(state, cppn_population)

    step_timings['step0_cppn_ask_transform'] = time.perf_counter() - t0

    # Get hierarchical grid
    # Use CPU-resident grid for depth 12+ to avoid GPU OOM on grid structure alone
    use_cpu_grid = algo._use_cpu_grid or algo.max_depth >= 12
    if use_cpu_grid:
        h_grid = get_hierarchical_grid_cpu(algo.max_depth)
        if algo._offload_verbose:
            print(f"  [DepthOffload] Using CPU-resident grid for depth {algo.max_depth}")
    else:
        h_grid = get_hierarchical_grid(algo.max_depth)
    total_positions = h_grid.total_positions
    num_levels = h_grid.num_levels

    # Cached coordinates
    input_coords = algo._cached_input_coords
    output_coords = algo._cached_output_coords
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]

    # Cached problem data
    inputs_batch = algo._cached_inputs
    targets_batch = algo._cached_targets

    # Population chunking check
    # For depth 10+ with large populations, the full weight matrices exceed
    # available RAM. Calculate if we need to process in population chunks.
    pop_chunk_size_for_ram = algo._calculate_population_chunk_size(
        pop_size, total_positions, num_inputs, num_outputs
    )

    use_population_chunking = pop_chunk_size_for_ram < pop_size

    if use_population_chunking:
        if algo._offload_verbose:
            print(f"  [PopChunking] Enabled: pop={pop_size}, chunk={pop_chunk_size_for_ram}, "
                  f"target_cpu_gb={algo._target_cpu_gb}")
        return _run_generation_population_chunked(
        algo,
            state, problem, cppns_transformed, h_grid, use_cpu_grid,
            pop_size, pop_chunk_size_for_ram, total_positions, num_levels,
            num_inputs, num_outputs, input_coords, output_coords,
            inputs_batch, targets_batch, skip_metrics, step_timings, t0
        )

    # Step 1: Depth offload, building the weight matrices in CPU storage. This path
    # serves populations that fit in RAM; larger ones take the chunked path above.
    t1 = time.perf_counter()

    # Unlike the standard path, allocate the accumulators on the CPU
    W1_cpu = np.zeros((pop_size, num_inputs, total_positions), dtype=np.float32)
    W2_cpu = np.zeros((pop_size, num_outputs, total_positions), dtype=np.float32)
    masks_cpu = np.zeros((pop_size, total_positions), dtype=np.bool_)

    # Track CPU memory
    cpu_bytes = (W1_cpu.nbytes + W2_cpu.nbytes + masks_cpu.nbytes)
    algo._cpu_memory_peak_gb = cpu_bytes / 1e9

    # Track level variances and reached masks for subdivision
    level_variances = []
    level_reached = []
    level_reached.append(np.ones((pop_size, 4), dtype=bool))  # Level 0 always reached
    # variance_rule legacy: set at level 0, read where level 1's reach is derived
    legacy_v0 = None

    # Per-level reached counts. A level stuck at 0 means the substrate stopped short
    # of the requested max_depth; these counts make that visible (a depth sweep
    # without them ran for 188 days against a level-3 substrate).
    algo._last_reached_per_level = [0] * num_levels

    # Cache for next-level weights (for variance lookahead)
    cached_next_input_weights = None

    max_weight = algo.max_weight
    weight_thresh = 0.1

    # Chunk sizing for CPPN queries. Position chunks first: they bound the slab
    # any single query call actually touches. The sizing counts the forward carry:
    # without it, the formula gives 62,500-position shards against a 0.5 GB target
    # (the recorded 25.0 GB depth-7 allocation).
    cppn_max_nodes = cppns_transformed[0].shape[1]  # (nodes, conns, ...) padded axis
    pos_chunk_size_w1 = algo._calculate_optimal_pos_chunk_size(
        pop_size, num_inputs, max_nodes=cppn_max_nodes)
    pos_chunk_size_w2 = algo._calculate_optimal_pos_chunk_size(
        pop_size, num_outputs, max_nodes=cppn_max_nodes)

    # Population chunk size. -1 is AUTO: chunk only when one query call's carry
    # would not fit the shard budget. The gate budgets against the realized slab,
    # because a call never touches more positions than the larger position chunk
    # allows. A gate sized against the largest level, while the position axis
    # consumes the same target independently, leaves the realized per-call carry
    # (pop_chunk x pos_chunk) 53x under budget, paid in kernel launches
    # (a measured 1.97x overhead).
    # Both figures: XOR, feed-forward (no h->h connections, so no cache), depth 7,
    # population 1000, one RTX 2080 Ti 11 GB, jax 0.4.38. The carry was ~9.4 MB per
    # call against the 0.5 GB target, and auto mode with that gate took 50.171 s per
    # generation against 25.425 s for forced offload with population chunk 10
    # (seeds 42-51).
    if algo.population_chunk_size > 0:
        pop_chunk_size = algo.population_chunk_size
    elif algo.population_chunk_size == -1:
        largest_level = max(
            int(h_grid.level_offsets_static[lv + 1] - h_grid.level_offsets_static[lv])
            for lv in range(num_levels)
        )
        realized_slab = min(largest_level,
                            max(pos_chunk_size_w1, pos_chunk_size_w2))
        pop_chunk_size = population_chunk_for_query(
            pop_size, realized_slab, cppn_max_nodes, algo._target_shard_gb)
    else:
        pop_chunk_size = pop_size

    if algo._offload_verbose:
        print(f"  [DepthOffload] Position chunks: W1={pos_chunk_size_w1:,}, W2={pos_chunk_size_w2:,}")

    level_timings = []

    for level in range(num_levels):
        t_level = time.perf_counter()

        level_start = h_grid.level_offsets_static[level]
        level_end = h_grid.level_offsets_static[level + 1]
        num_level_positions = level_end - level_start

        # Get level positions - transfer to GPU on-demand for CPU grid
        if use_cpu_grid:
            # CPU grid: positions are NumPy arrays, transfer to GPU
            level_positions = jax.device_put(h_grid.level_positions[level])
        else:
            # GPU grid: positions are already JAX arrays on GPU
            level_positions = h_grid.level_positions[level]

        # Get reached mask for this level (computed on CPU for efficiency)
        if level == 0:
            current_reached_mask = level_reached[0]
        else:
            # Compute reached from previous level's subdivision
            prev_reached = level_reached[level - 1]
            prev_variance = (legacy_v0 if (level == 1 and legacy_v0 is not None)
                             else level_variances[level - 1])
            # Canonical ES-HyperNEAT subdivides on two clauses (naive/eshyperneat/eshyperneat.py:441):
            #   (level < initial_depth) or (variance > threshold)
            # The eager path needs both clauses: variance alone stops subdividing at
            # level 2-3, so with the variance clause only, max_depth 8..13 cannot
            # differ. A parent at level-1 is forced when level-1 < initial_depth, i.e.
            # level <= initial_depth.
            if level <= algo.initial_depth:
                prev_subdivided = prev_reached
            else:
                prev_subdivided = prev_reached & (prev_variance > algo.variance_threshold)

            parent_indices = np.asarray(h_grid.parent_indices[level])
            current_reached_mask = prev_subdivided[:, parent_indices]
            level_reached.append(current_reached_mask)

        _reached_count = int(np.count_nonzero(current_reached_mask))
        algo._last_reached_per_level[level] += _reached_count
        # Once a level is unreached every level below it is too, and processing them
        # writes exactly zeros: jnp.where(mask & ..., W1_raw, 0.0) against a
        # pre-zeroed buffer. On by default (skip_dead_levels): it changes cost, not results.
        if algo._skip_dead_levels and _reached_count == 0:
            break

        if level == 0 and algo._legacy_level0_variance:
            # variance_rule legacy: level 0's own variance (four positions, one extra query)
            legacy_v0 = np.asarray(legacy_level0_variance(query_level_positions_chunked(
                state, cppns_transformed, input_coords, level_positions,
                True, algo._jitted_cppn_forward, pop_chunk_size, pos_chunk_size_w1,
                geometry_seeding_enabled=algo.geometry_seeding_enabled,
            )[:, 0, :]))

        # Check if this level is too large for standard processing
        use_streaming_for_level = num_level_positions > algo._max_lookahead_positions

        if use_streaming_for_level:
            # Streaming mode: Process position chunks and offload immediately
            # This avoids holding the full level result on GPU
            if algo._offload_verbose:
                print(f"    [Using streaming mode for level {level} ({num_level_positions:,} positions)]")

            # Use streaming to process and offload W1 (input weights)
            transfers_w1 = process_level_streaming(
                state, cppns_transformed, input_coords, level_positions,
                True, algo._jitted_cppn_forward, pop_chunk_size,
                pos_chunk_size_w1, W1_cpu, level_start,
                max_weight, weight_thresh, current_reached_mask,
                geometry_seeding_enabled=algo.geometry_seeding_enabled,
            )
            algo._gpu_transfers_count += transfers_w1

            # Use streaming to process and offload W2 (output weights)
            transfers_w2 = process_level_streaming(
                state, cppns_transformed, output_coords, level_positions,
                False, algo._jitted_cppn_forward, pop_chunk_size,
                pos_chunk_size_w2, W2_cpu, level_start,
                max_weight, weight_thresh, current_reached_mask,
                geometry_seeding_enabled=algo.geometry_seeding_enabled,
            )
            algo._gpu_transfers_count += transfers_w2

            # Store mask (already on CPU)
            masks_cpu[:, level_start:level_end] = current_reached_mask

            # A streaming level is large, but its variance still has to be real:
            # zeros would leave the next level's reached mask empty and stop the
            # substrate here. Slab it, and skip entirely if nothing is reached.
            if level < num_levels - 1 and bool(np.any(current_reached_mask)):
                next_level_positions = (jax.device_put(h_grid.level_positions[level + 1]) if use_cpu_grid
                                else h_grid.level_positions[level + 1])

                def _query_next_slice(flat_start, flat_end):
                    sliced = query_level_positions_chunked(
                        state, cppns_transformed, input_coords,
                        next_level_positions[flat_start:flat_end],
                        True, algo._jitted_cppn_forward, pop_chunk_size,
                        pos_chunk_size_w1,
                        geometry_seeding_enabled=algo.geometry_seeding_enabled,
                    )
                    return np.asarray(sliced[:, 0, :])

                level_variances.append(compute_single_level_variance_chunked(
                    _query_next_slice, h_grid, level, pop_size,
                    max(1, algo._max_lookahead_positions) * pop_size,
                ))
            else:
                level_variances.append(np.zeros((pop_size, num_level_positions)))
            cached_next_input_weights = None

        else:
            # Standard mode: Query full level, then offload
            # Convert reached mask to JAX for GPU operations
            current_reached_mask_gpu = jax.device_put(current_reached_mask)

            # GPU: Query CPPN for this level's positions (with position chunking)
            if cached_next_input_weights is not None and level > 0:
                level_input_weights = cached_next_input_weights
            else:
                level_input_weights = query_level_positions_chunked(
                    state, cppns_transformed, input_coords, level_positions,
                    True, algo._jitted_cppn_forward, pop_chunk_size,
                    pos_chunk_size_w1,
                    geometry_seeding_enabled=algo.geometry_seeding_enabled,
                )

            level_output_weights = query_level_positions_chunked(
                state, cppns_transformed, output_coords, level_positions,
                False, algo._jitted_cppn_forward, pop_chunk_size,
                pos_chunk_size_w2,
                geometry_seeding_enabled=algo.geometry_seeding_enabled,
            )

            # GPU: Compute variance for this level (lookahead to next level)
            # Skip the lookahead when nothing is reached: subdivision below is
            # `prev_reached & (...)`, all-False regardless of the variance, so the
            # query would be pure waste: 357.9M positions per genome at depth 13.
            if level < num_levels - 1 and bool(np.any(current_reached_mask)):
                # Get next level positions - transfer to GPU on-demand for CPU grid
                if use_cpu_grid:
                    next_level_positions = jax.device_put(h_grid.level_positions[level + 1])
                else:
                    next_level_positions = h_grid.level_positions[level + 1]
                next_level_size = next_level_positions.shape[0]

                # Two ways a level can be too large to materialize in one shot:
                #   truncate_variance_lookahead=True  -> zeros, exactly as documented. The
                #     substrate changes: zeros fail `variance > threshold` everywhere, so the
                #     next level's reached mask comes out empty and subdivision halts here.
                #     Opt-in, because that is a real change to what gets evolved.
                #   otherwise -> slab it. Bounded memory and the exact same substrate, which
                #     is what the default should give.
                if next_level_size > algo._max_lookahead_positions and algo._truncate_variance_lookahead:
                    if algo._offload_verbose:
                        print(f"    [Truncating lookahead: next level has {next_level_size:,} positions > {algo._max_lookahead_positions:,}, so the substrate halts at level {level}]")
                    level_variances.append(np.zeros((pop_size, num_level_positions)))
                    cached_next_input_weights = None
                elif next_level_size > algo._max_lookahead_positions:
                    if algo._offload_verbose:
                        print(f"    [Chunked lookahead: next level has {next_level_size:,} positions > {algo._max_lookahead_positions:,}]")

                    def _query_next_slice(flat_start, flat_end):
                        sliced = query_level_positions_chunked(
                            state, cppns_transformed, input_coords,
                            next_level_positions[flat_start:flat_end],
                            True, algo._jitted_cppn_forward, pop_chunk_size,
                            pos_chunk_size_w1,
                            geometry_seeding_enabled=algo.geometry_seeding_enabled,
                        )
                        return np.asarray(sliced[:, 0, :])

                    level_variances.append(compute_single_level_variance_chunked(
                        _query_next_slice, h_grid, level, pop_size,
                        max(1, algo._max_lookahead_positions) * pop_size,
                    ))
                    cached_next_input_weights = None
                else:
                    cached_next_input_weights = query_level_positions_chunked(
                        state, cppns_transformed, input_coords, next_level_positions,
                        True, algo._jitted_cppn_forward, pop_chunk_size,
                        pos_chunk_size_w1,
                        geometry_seeding_enabled=algo.geometry_seeding_enabled,
                    )

                    level_variance_gpu = compute_single_level_variance(
                        cached_next_input_weights[:, 0, :],
                        h_grid,
                        level,
                    )
                    level_variance = np.asarray(level_variance_gpu)
                    level_variances.append(level_variance)
                    algo._gpu_transfers_count += 1
            else:
                level_variances.append(np.zeros((pop_size, num_level_positions)))
                cached_next_input_weights = None

            # GPU: Apply weight processing
            active_mask_broadcast = current_reached_mask_gpu[:, None, :]

            W1_raw = jnp.tanh(level_input_weights) * max_weight
            W1_level = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )

            W2_raw = jnp.tanh(level_output_weights) * max_weight
            W2_level = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )

            # Offload to CPU immediately
            W1_cpu[:, :, level_start:level_end] = np.asarray(W1_level)
            W2_cpu[:, :, level_start:level_end] = np.asarray(W2_level)
            masks_cpu[:, level_start:level_end] = np.asarray(current_reached_mask_gpu)

            algo._gpu_transfers_count += 3  # W1, W2, masks

            # GPU: Free memory immediately
            del level_input_weights, level_output_weights
            del W1_raw, W2_raw, W1_level, W2_level
            del current_reached_mask_gpu, active_mask_broadcast

        level_time = time.perf_counter() - t_level
        level_timings.append({
            'level': level,
            'positions': num_level_positions,
            'time_s': level_time,
        })

        if algo._offload_verbose:
            print(f"  Level {level}: {num_level_positions:,} positions, "
                  f"{level_time:.2f}s, transferred to CPU")

    step_timings['step1_depth_offload_build'] = time.perf_counter() - t1
    step_timings['gpu_to_cpu_transfers'] = algo._gpu_transfers_count
    step_timings['cpu_memory_gb'] = algo._cpu_memory_peak_gb

    # Step 2: Build masks (already on CPU)
    t2 = time.perf_counter()
    masks_A_cpu = masks_cpu  # Already accumulated on CPU
    step_timings['step2_mask_build'] = time.perf_counter() - t2

    # Step 5: Evaluate all networks with chunked weight streaming
    t5 = time.perf_counter()

    is_recurrent = _evaluates_hidden_to_hidden(algo)

    if is_recurrent:
        # Recurrent mode: Need full weights on GPU for h->h discovery
        # Stream back in chunks for memory safety
        fitnesses = _evaluate_recurrent_chunked(
        algo,
            state, cppns_transformed, W1_cpu, W2_cpu, masks_A_cpu,
            h_grid, inputs_batch, targets_batch,
            pop_size, pop_chunk_size, step_timings,
            cached_sparse_hh=cached_sparse_hh,
        )
        step_timings['mode'] = 'recurrent_depth_offload'
    else:
        # Feedforward mode: Chunked evaluation
        fitnesses = _evaluate_feedforward_chunked(
        algo,
            W1_cpu, W2_cpu, masks_A_cpu,
            inputs_batch, targets_batch,
            h_grid, step_timings
        )
        step_timings['mode'] = 'feedforward_depth_offload'

    fitnesses, nan_fitness_count = sanitize_fitness(fitnesses)

    step_timings['step5_network_evaluation'] = time.perf_counter() - t5

    # Step 6: Selection (tell)
    t6 = time.perf_counter()

    new_state = algo._compiled_tell(*algo._before_tell(state, fitnesses))
    # Generation counter: run_generation_verbose increments this on the
    # standard/streaming paths; the offload path never did, which would pin the
    # h->h cache's time-based refresh rule at generation 0 forever.
    algo._current_generation += 1

    step_timings['step6_tell'] = time.perf_counter() - t6

    # Build metrics
    time_elapsed = time.perf_counter() - t0

    # The evaluated population's species: state is the state before tell.
    num_species, species_sizes, species_fitness = species_summary(state.species, fitnesses)
    if skip_metrics:
        best_fitness = float(jnp.max(fitnesses))
        generation = int(new_state.generation)

        metrics = AlgorithmMetrics(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=0.0,
            min_fitness=0.0,
            max_fitness=best_fitness,
            std_fitness=0.0,
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=pop_size,
            time_elapsed=time_elapsed,
            custom_metrics={'mode': 'depth_offload', 'skip_metrics': True,
                            'nan_fitness_count': nan_fitness_count},
        )
    else:
        best_fitness, mean_fitness, min_fitness, std_fitness = fitness_statistics(
            fitnesses, nan_fitness_count)
        generation = int(new_state.generation)

        custom_metrics = {
            'mode': 'depth_offload',
            'nan_fitness_count': nan_fitness_count,
            'variance_threshold': algo.variance_threshold,
            'total_positions': total_positions,
            'num_levels': num_levels,
            'cpu_memory_gb': algo._cpu_memory_peak_gb,
            'gpu_transfers': algo._gpu_transfers_count,
        }
        custom_metrics.update(step_timings)

        metrics = AlgorithmMetrics(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=best_fitness,
            std_fitness=std_fitness,
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=pop_size,
            time_elapsed=time_elapsed,
            custom_metrics=custom_metrics,
        )

    return new_state, metrics


def _run_generation_population_chunked(
    algo,
    state,
    problem,
    cppns_transformed: Tuple,
    h_grid: Union[HierarchicalGridStructure, HierarchicalGridStructureSingle],
    use_cpu_grid: bool,
    pop_size: int,
    pop_chunk_size: int,
    total_positions: int,
    num_levels: int,
    num_inputs: int,
    num_outputs: int,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    skip_metrics: bool,
    step_timings: Dict[str, Any],
    t0: float,
) -> Tuple[Any, AlgorithmMetrics]:
    """Run generation with population chunking for RAM-limited scenarios.

    This method processes the population in chunks when the full weight
    matrices would exceed available CPU RAM. Each chunk is processed
    through all levels independently, evaluated, and the fitnesses are
    collected.

    Args:
        state: Current algorithm state
        problem: Problem instance
        cppns_transformed: Transformed CPPNs for full population
        h_grid: Hierarchical grid structure
        use_cpu_grid: Whether using CPU-resident grid
        pop_size: Full population size
        pop_chunk_size: Size of each population chunk
        total_positions: Total positions in substrate
        num_levels: Number of depth levels
        num_inputs: Number of input nodes
        num_outputs: Number of output nodes
        input_coords: Input coordinates
        output_coords: Output coordinates
        inputs_batch: Input data for evaluation
        targets_batch: Target data for evaluation
        skip_metrics: Whether to skip detailed metrics
        step_timings: Dictionary for timing stats
        t0: Start time for total elapsed calculation

    Returns:
        Tuple of (new_state, metrics)
    """
    t1 = time.perf_counter()

    # Ensure memmap directory exists if we might use it
    if algo._use_memmap:
        os.makedirs(algo._memmap_path, exist_ok=True)

    max_weight = algo.max_weight
    weight_thresh = 0.1

    # Chunk sizing within each pop chunk: position chunks first, then the AUTO
    # population gate against the realized slab (see the unchunked path for the
    # 53x-under-budget rationale this ordering fixes).
    cppn_max_nodes = cppns_transformed[0].shape[1]
    pos_chunk_size_w1 = algo._calculate_optimal_pos_chunk_size(
        pop_chunk_size, num_inputs, max_nodes=cppn_max_nodes)
    pos_chunk_size_w2 = algo._calculate_optimal_pos_chunk_size(
        pop_chunk_size, num_outputs, max_nodes=cppn_max_nodes)
    if algo.population_chunk_size > 0:
        cppn_pop_chunk = min(algo.population_chunk_size, pop_chunk_size)
    elif algo.population_chunk_size == -1:
        largest_level = max(
            int(h_grid.level_offsets_static[lv + 1] - h_grid.level_offsets_static[lv])
            for lv in range(num_levels)
        )
        realized_slab = min(largest_level,
                            max(pos_chunk_size_w1, pos_chunk_size_w2))
        cppn_pop_chunk = population_chunk_for_query(
            pop_chunk_size, realized_slab, cppn_max_nodes, algo._target_shard_gb)
    else:
        cppn_pop_chunk = pop_chunk_size

    # Collect fitnesses from all chunks
    all_fitnesses = []
    num_chunks = (pop_size + pop_chunk_size - 1) // pop_chunk_size
    # Generation-level h->h accumulators (recurrent path); published to the
    # algo._last_* fields after the loop so consumers see population totals.
    gen_hh_total = 0
    gen_hh_max = 0
    gen_hh_discovery_ms = 0.0

    if algo._offload_verbose:
        print(f"  [PopChunking] Processing {num_chunks} chunks of {pop_chunk_size} genomes")
        print(f"  [PopChunking] Position chunks: W1={pos_chunk_size_w1:,}, W2={pos_chunk_size_w2:,}")

    is_recurrent = _evaluates_hidden_to_hidden(algo)

    # Track whether any chunk used memmap
    any_chunk_used_memmap = False

    # Pre-transfer level positions to GPU once (avoids redundant transfers per chunk)
    # At depth 13: 357.9M positions x 30 chunks = 85.8 GB redundant transfers avoided
    gpu_level_positions = {}
    if use_cpu_grid:
        for level in range(num_levels):
            gpu_level_positions[level] = jax.device_put(h_grid.level_positions[level])
        if algo._offload_verbose:
            print(f"  [PopChunking] Pre-transferred {num_levels} levels to GPU (cached)")

    # Accumulated across population chunks; see the non-chunked path above.
    algo._last_reached_per_level = [0] * num_levels

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * pop_chunk_size
        chunk_end = min(chunk_start + pop_chunk_size, pop_size)
        actual_chunk_size = chunk_end - chunk_start

        if algo._offload_verbose:
            print(f"  [PopChunking] Chunk {chunk_idx + 1}/{num_chunks}: "
                  f"genomes [{chunk_start}:{chunk_end}]")

        # Slice CPPNs for this chunk
        chunk_cppns = (
            cppns_transformed[0][chunk_start:chunk_end],
            cppns_transformed[1][chunk_start:chunk_end],
            cppns_transformed[2][chunk_start:chunk_end],
            cppns_transformed[3][chunk_start:chunk_end],
        )

        # Check if memmap needed (chunk arrays > target_cpu_gb)
        bytes_per_genome = (num_inputs + num_outputs) * total_positions * 4 + total_positions
        total_chunk_bytes = actual_chunk_size * bytes_per_genome
        use_memmap_for_chunk = (
            algo._use_memmap and
            total_chunk_bytes > algo._target_cpu_gb * 1e9 * 0.9
        )

        memmap_dir = None  # Track for cleanup

        if use_memmap_for_chunk:
            any_chunk_used_memmap = True
            # Create temp directory for this chunk's memmap files
            memmap_dir = tempfile.mkdtemp(prefix=f'emr_d{num_levels}_c{chunk_idx}_', dir=algo._memmap_path)

            if algo._offload_verbose:
                chunk_gb = total_chunk_bytes / 1e9
                print(f"    [Memmap] Chunk {chunk_idx + 1} requires {chunk_gb:.1f} GB, using disk-backed arrays")

            # Single combined memmap file with lazy initialization (sparse file)
            # Calculate sizes for each array
            w1_elements = actual_chunk_size * num_inputs * total_positions
            w2_elements = actual_chunk_size * num_outputs * total_positions
            mask_elements = actual_chunk_size * total_positions

            # Total bytes: float32 for weights, uint8 for masks
            w1_bytes = w1_elements * 4
            w2_bytes = w2_elements * 4
            mask_bytes = mask_elements  # uint8 = 1 byte
            total_bytes = w1_bytes + w2_bytes + mask_bytes

            combined_path = os.path.join(memmap_dir, 'combined.dat')

            # Create sparse file (no disk write for zeros on Linux ext4/xfs)
            with open(combined_path, 'wb') as f:
                f.truncate(total_bytes)

            # Map views into the combined file at calculated offsets
            w1_offset = 0
            w2_offset = w1_bytes
            mask_offset = w2_offset + w2_bytes

            W1_cpu = np.memmap(combined_path, dtype=np.float32, mode='r+',
                shape=(actual_chunk_size, num_inputs, total_positions),
                offset=w1_offset)
            W2_cpu = np.memmap(combined_path, dtype=np.float32, mode='r+',
                shape=(actual_chunk_size, num_outputs, total_positions),
                offset=w2_offset)
            # Use uint8 for masks (bool_ doesn't support offset reliably)
            masks_cpu = np.memmap(combined_path, dtype=np.uint8, mode='r+',
                shape=(actual_chunk_size, total_positions),
                offset=mask_offset)
        else:
            # Standard in-memory allocation (depths 8-12 with smaller chunks)
            W1_cpu = np.zeros((actual_chunk_size, num_inputs, total_positions), dtype=np.float32)
            W2_cpu = np.zeros((actual_chunk_size, num_outputs, total_positions), dtype=np.float32)
            masks_cpu = np.zeros((actual_chunk_size, total_positions), dtype=np.bool_)

        # Track CPU memory for this chunk
        chunk_cpu_bytes = (W1_cpu.nbytes + W2_cpu.nbytes + masks_cpu.nbytes)
        algo._cpu_memory_peak_gb = max(algo._cpu_memory_peak_gb, chunk_cpu_bytes / 1e9)

        # Track level variances and reached masks for this chunk
        level_variances = []
        level_reached = []
        level_reached.append(np.ones((actual_chunk_size, 4), dtype=bool))
        # variance_rule legacy: set at level 0, read where level 1's reach is derived
        legacy_v0 = None

        cached_next_input_weights = None

        # Process all levels for this chunk
        for level in range(num_levels):
            level_start = h_grid.level_offsets_static[level]
            level_end = h_grid.level_offsets_static[level + 1]
            num_level_positions = level_end - level_start

            # Get level positions (use GPU cache to avoid redundant transfers)
            if use_cpu_grid:
                level_positions = gpu_level_positions[level]
            else:
                level_positions = h_grid.level_positions[level]

            # Get reached mask for this level
            if level == 0:
                current_reached_mask = level_reached[0]
            else:
                prev_reached = level_reached[level - 1]
                prev_variance = (legacy_v0 if (level == 1 and legacy_v0 is not None)
                                 else level_variances[level - 1])
                # Canonical ES-HyperNEAT subdivides on two clauses (naive/eshyperneat/eshyperneat.py:441):
                #   (level < initial_depth) or (variance > threshold)
                # The eager path needs both clauses: variance alone stops subdividing at
                # level 2-3, so with the variance clause only, max_depth 8..13 cannot
                # differ. A parent at level-1 is forced when level-1 < initial_depth, i.e.
                # level <= initial_depth.
                if level <= algo.initial_depth:
                    prev_subdivided = prev_reached
                else:
                    prev_subdivided = prev_reached & (prev_variance > algo.variance_threshold)
                parent_indices = np.asarray(h_grid.parent_indices[level])
                current_reached_mask = prev_subdivided[:, parent_indices]
                level_reached.append(current_reached_mask)

            _reached_count = int(np.count_nonzero(current_reached_mask))
            algo._last_reached_per_level[level] += _reached_count
            # Once a level is unreached every level below it is too, and processing them
            # writes exactly zeros: jnp.where(mask & ..., W1_raw, 0.0) against a
            # pre-zeroed buffer. On by default (skip_dead_levels): it changes cost, not results.
            if algo._skip_dead_levels and _reached_count == 0:
                break

            if level == 0 and algo._legacy_level0_variance:
                # variance_rule legacy: level 0's own variance (four positions, one extra query)
                legacy_v0 = np.asarray(legacy_level0_variance(query_level_positions_chunked(
                    state, chunk_cppns, input_coords, level_positions,
                    True, algo._jitted_cppn_forward, cppn_pop_chunk, pos_chunk_size_w1,
                    geometry_seeding_enabled=algo.geometry_seeding_enabled,
                )[:, 0, :]))

            use_streaming_for_level = num_level_positions > algo._max_lookahead_positions

            if use_streaming_for_level:
                # Streaming mode for large levels
                transfers_w1 = process_level_streaming(
                    state, chunk_cppns, input_coords, level_positions,
                    True, algo._jitted_cppn_forward, cppn_pop_chunk,
                    pos_chunk_size_w1, W1_cpu, level_start,
                    max_weight, weight_thresh, current_reached_mask,
                    geometry_seeding_enabled=algo.geometry_seeding_enabled,
                )
                algo._gpu_transfers_count += transfers_w1

                transfers_w2 = process_level_streaming(
                    state, chunk_cppns, output_coords, level_positions,
                    False, algo._jitted_cppn_forward, cppn_pop_chunk,
                    pos_chunk_size_w2, W2_cpu, level_start,
                    max_weight, weight_thresh, current_reached_mask,
                    geometry_seeding_enabled=algo.geometry_seeding_enabled,
                )
                algo._gpu_transfers_count += transfers_w2

                # Convert bool to uint8 for memmap compatibility
                masks_cpu[:, level_start:level_end] = current_reached_mask.astype(np.uint8)
                # A streaming level is large, but its variance still has to be real:
                # zeros would leave the next level's reached mask empty and stop the
                # substrate here. Slab it, and skip entirely if nothing is reached.
                if level < num_levels - 1 and bool(np.any(current_reached_mask)):
                    next_level_positions = (gpu_level_positions[level + 1] if use_cpu_grid
                                            else h_grid.level_positions[level + 1])

                    def _query_next_slice(flat_start, flat_end):
                        sliced = query_level_positions_chunked(
                            state, chunk_cppns, input_coords,
                            next_level_positions[flat_start:flat_end],
                            True, algo._jitted_cppn_forward, cppn_pop_chunk,
                            pos_chunk_size_w1,
                            geometry_seeding_enabled=algo.geometry_seeding_enabled,
                        )
                        return np.asarray(sliced[:, 0, :])

                    level_variances.append(compute_single_level_variance_chunked(
                        _query_next_slice, h_grid, level, actual_chunk_size,
                        max(1, algo._max_lookahead_positions) * actual_chunk_size,
                    ))
                else:
                    level_variances.append(np.zeros((actual_chunk_size, num_level_positions)))
                cached_next_input_weights = None

            else:
                # Standard mode
                current_reached_mask_gpu = jax.device_put(current_reached_mask)

                if cached_next_input_weights is not None and level > 0:
                    level_input_weights = cached_next_input_weights
                else:
                    level_input_weights = query_level_positions_chunked(
                        state, chunk_cppns, input_coords, level_positions,
                        True, algo._jitted_cppn_forward, cppn_pop_chunk,
                        pos_chunk_size_w1,
                        geometry_seeding_enabled=algo.geometry_seeding_enabled,
                    )

                level_output_weights = query_level_positions_chunked(
                    state, chunk_cppns, output_coords, level_positions,
                    False, algo._jitted_cppn_forward, cppn_pop_chunk,
                    pos_chunk_size_w2,
                    geometry_seeding_enabled=algo.geometry_seeding_enabled,
                )

                # Compute variance for next level (use GPU cache for positions)
                # Skip the lookahead when nothing is reached: subdivision below is
                # `prev_reached & (...)`, all-False regardless of the variance, so the
                # query would be pure waste: 357.9M positions per genome at depth 13.
                if level < num_levels - 1 and bool(np.any(current_reached_mask)):
                    if use_cpu_grid:
                        next_level_positions = gpu_level_positions[level + 1]
                    else:
                        next_level_positions = h_grid.level_positions[level + 1]
                    next_level_size = next_level_positions.shape[0]

                    # Two ways a level can be too large to materialize in one shot:
                    #   truncate_variance_lookahead=True  -> zeros, exactly as documented. The
                    #     substrate changes: zeros fail `variance > threshold` everywhere, so the
                    #     next level's reached mask comes out empty and subdivision halts here.
                    #     Opt-in, because that is a real change to what gets evolved.
                    #   otherwise -> slab it. Bounded memory and the exact same substrate, which
                    #     is what the default should give.
                    if next_level_size > algo._max_lookahead_positions and algo._truncate_variance_lookahead:
                        level_variances.append(np.zeros((actual_chunk_size, num_level_positions)))
                        cached_next_input_weights = None
                    elif next_level_size > algo._max_lookahead_positions:

                        def _query_next_slice(flat_start, flat_end):
                            sliced = query_level_positions_chunked(
                                state, chunk_cppns, input_coords,
                                next_level_positions[flat_start:flat_end],
                                True, algo._jitted_cppn_forward, cppn_pop_chunk,
                                pos_chunk_size_w1,
                                geometry_seeding_enabled=algo.geometry_seeding_enabled,
                            )
                            return np.asarray(sliced[:, 0, :])

                        level_variances.append(compute_single_level_variance_chunked(
                            _query_next_slice, h_grid, level, actual_chunk_size,
                            max(1, algo._max_lookahead_positions) * actual_chunk_size,
                        ))
                        cached_next_input_weights = None
                    else:
                        cached_next_input_weights = query_level_positions_chunked(
                            state, chunk_cppns, input_coords, next_level_positions,
                            True, algo._jitted_cppn_forward, cppn_pop_chunk,
                            pos_chunk_size_w1,
                            geometry_seeding_enabled=algo.geometry_seeding_enabled,
                        )
                        level_variance_gpu = compute_single_level_variance(
                            cached_next_input_weights[:, 0, :],
                            h_grid,
                            level,
                        )
                        level_variance = np.asarray(level_variance_gpu)
                        level_variances.append(level_variance)
                        algo._gpu_transfers_count += 1
                else:
                    level_variances.append(np.zeros((actual_chunk_size, num_level_positions)))
                    cached_next_input_weights = None

                # Apply weight processing
                active_mask_broadcast = current_reached_mask_gpu[:, None, :]

                W1_raw = jnp.tanh(level_input_weights) * max_weight
                W1_level = jnp.where(
                    active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                    W1_raw,
                    0.0
                )

                W2_raw = jnp.tanh(level_output_weights) * max_weight
                W2_level = jnp.where(
                    active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                    W2_raw,
                    0.0
                )

                # Offload to CPU (convert mask to uint8 for memmap compatibility)
                W1_cpu[:, :, level_start:level_end] = np.asarray(W1_level)
                W2_cpu[:, :, level_start:level_end] = np.asarray(W2_level)
                masks_cpu[:, level_start:level_end] = np.asarray(current_reached_mask_gpu).astype(np.uint8)

                algo._gpu_transfers_count += 3

                # Free GPU memory
                del level_input_weights, level_output_weights
                del W1_raw, W2_raw, W1_level, W2_level
                del current_reached_mask_gpu, active_mask_broadcast

        # Evaluate this chunk. Each helper writes its step timings into the dict
        # it is handed and (recurrent path) stamps algo._last_*hh* with this
        # chunk's discovery stats, so both are accumulated across chunks; passing
        # {} here would report the last chunk's h->h connection count as the
        # generation's total.
        chunk_timings: Dict[str, Any] = {}
        if is_recurrent:
            chunk_fitnesses = _evaluate_recurrent_chunked(
        algo,
                state, chunk_cppns, W1_cpu, W2_cpu, masks_cpu,
                h_grid, inputs_batch, targets_batch,
                actual_chunk_size, cppn_pop_chunk, chunk_timings,
                consult_cache_manager=False,
            )
            gen_hh_total += int(algo._last_hh_connections)
            gen_hh_max = max(gen_hh_max, int(algo._last_max_hh_connections))
            gen_hh_discovery_ms += float(algo._last_hh_discovery_time_ms)
        else:
            chunk_fitnesses = _evaluate_feedforward_chunked(
        algo,
                W1_cpu, W2_cpu, masks_cpu,
                inputs_batch, targets_batch,
                h_grid, chunk_timings
            )
        for key, value in chunk_timings.items():
            if isinstance(value, (int, float)):
                step_timings[key] = step_timings.get(key, 0.0) + value
            else:
                step_timings[key] = value

        all_fitnesses.append(chunk_fitnesses)

        # Free chunk arrays and cleanup memmap files
        del W1_cpu, W2_cpu, masks_cpu
        gc.collect()

        # Remove memmap directory if we used disk-backed arrays
        if use_memmap_for_chunk and memmap_dir is not None:
            try:
                shutil.rmtree(memmap_dir, ignore_errors=True)
                if algo._offload_verbose:
                    print(f"    [Memmap] Cleaned up {memmap_dir}")
            except Exception as e:
                # Log but don't fail if cleanup fails
                if algo._offload_verbose:
                    print(f"    [Memmap] Warning: cleanup failed: {e}")

        if algo._offload_verbose:
            best_chunk = float(jnp.max(chunk_fitnesses))
            print(f"    Chunk {chunk_idx + 1} best fitness: {best_chunk:.6f}")

    # Clean up GPU position cache after all chunks complete
    if use_cpu_grid and gpu_level_positions:
        del gpu_level_positions
        gpu_level_positions = {}
        if algo._offload_verbose:
            print("  [PopChunking] Cleaned up GPU position cache")

    step_timings['step1_depth_offload_build'] = time.perf_counter() - t1
    step_timings['gpu_to_cpu_transfers'] = algo._gpu_transfers_count
    step_timings['cpu_memory_gb'] = algo._cpu_memory_peak_gb
    step_timings['population_chunks'] = num_chunks
    step_timings['population_chunk_size'] = pop_chunk_size
    step_timings['mode'] = 'population_chunked_depth_offload'
    step_timings['used_memmap'] = any_chunk_used_memmap

    # Publish population-level h->h stats: totals across chunks, not the last
    # chunk's values. avg is total/pop; max is the max over chunks.
    if is_recurrent and num_chunks > 0:
        algo._last_hh_connections = gen_hh_total
        algo._last_avg_hh_connections = gen_hh_total / max(pop_size, 1)
        algo._last_hh_discovery_time_ms = gen_hh_discovery_ms
        step_timings['gen_hh_total'] = gen_hh_total
        step_timings['gen_hh_max_chunk'] = gen_hh_max

    # Concatenate all fitnesses
    fitnesses = jnp.concatenate(all_fitnesses, axis=0)
    fitnesses, nan_fitness_count = sanitize_fitness(fitnesses)

    # Selection (tell)
    t6 = time.perf_counter()

    new_state = algo._compiled_tell(*algo._before_tell(state, fitnesses))
    # Generation counter: run_generation_verbose increments this on the
    # standard/streaming paths; the offload path never did, which would pin the
    # h->h cache's time-based refresh rule at generation 0 forever.
    algo._current_generation += 1

    step_timings['step6_tell'] = time.perf_counter() - t6

    # Build metrics
    time_elapsed = time.perf_counter() - t0

    # The evaluated population's species: state is the state before tell.
    num_species, species_sizes, species_fitness = species_summary(state.species, fitnesses)
    if skip_metrics:
        best_fitness = float(jnp.max(fitnesses))
        generation = int(new_state.generation)

        metrics = AlgorithmMetrics(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=0.0,
            min_fitness=0.0,
            max_fitness=best_fitness,
            std_fitness=0.0,
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=pop_size,
            time_elapsed=time_elapsed,
            custom_metrics={'mode': 'population_chunked', 'skip_metrics': True,
                            'nan_fitness_count': nan_fitness_count},
        )
    else:
        best_fitness, mean_fitness, min_fitness, std_fitness = fitness_statistics(
            fitnesses, nan_fitness_count)
        generation = int(new_state.generation)

        custom_metrics = {
            'mode': 'population_chunked',
            'nan_fitness_count': nan_fitness_count,
            'variance_threshold': algo.variance_threshold,
            'total_positions': total_positions,
            'num_levels': num_levels,
            'cpu_memory_gb': algo._cpu_memory_peak_gb,
            'gpu_transfers': algo._gpu_transfers_count,
            'population_chunks': num_chunks,
            'population_chunk_size': pop_chunk_size,
        }
        custom_metrics.update(step_timings)

        metrics = AlgorithmMetrics(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=best_fitness,
            std_fitness=std_fitness,
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=pop_size,
            time_elapsed=time_elapsed,
            custom_metrics=custom_metrics,
        )

    return new_state, metrics


def _evaluate_feedforward_chunked(
    algo,
    W1_cpu: np.ndarray,
    W2_cpu: np.ndarray,
    masks_cpu: np.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    h_grid: Union[HierarchicalGridStructure, HierarchicalGridStructureSingle],
    step_timings: Dict[str, Any],
) -> jnp.ndarray:
    """Evaluate feedforward networks by streaming weights from CPU.

    This implementation streams weight matrix chunks from CPU to GPU,
    computes partial hidden activations, and accumulates results.

    For depth 11 (22.4M positions):
    - Instead of loading 36GB W1 at once -> load in ~10 chunks
    - Each chunk: ~3GB, compute partial hidden, free, next chunk

    Args:
        W1_cpu: Input weights on CPU - (pop, inputs, positions)
        W2_cpu: Output weights on CPU - (pop, outputs, positions)
        masks_cpu: Active masks on CPU - (pop, positions)
        inputs_batch: Input data on GPU - (samples, inputs)
        targets_batch: Target data on GPU - (samples, outputs)
        h_grid: Hierarchical grid structure
        step_timings: Dictionary to record timing stats

    Returns:
        Fitness scores - (pop,)
    """
    pop_size = W1_cpu.shape[0]
    num_inputs = W1_cpu.shape[1]
    total_positions = W1_cpu.shape[2]
    n_samples = inputs_batch.shape[0]
    num_outputs = targets_batch.shape[1]

    t_eval = time.perf_counter()

    # Determine chunking strategy
    # Calculate optimal chunk size based on configured target GPU memory
    target_chunk_bytes = algo._target_eval_shard_gb * 1e9
    bytes_per_position = (pop_size * num_inputs * 4 +  # W1 slice
                           pop_size * num_outputs * 4 + # W2 slice
                           pop_size * 4 +               # mask slice
                           pop_size * n_samples * 4)    # hidden slice
    max_positions_per_chunk = int(target_chunk_bytes / bytes_per_position)
    max_positions_per_chunk = max(1000, min(max_positions_per_chunk, algo._eval_chunk_positions))

    # Hidden activations - computed chunk by chunk, accumulated on GPU
    # For very large positions, also chunk the hidden accumulator
    chunk_ranges = []

    pos_start = 0
    while pos_start < total_positions:
        pos_end = min(pos_start + max_positions_per_chunk, total_positions)
        chunk_ranges.append((pos_start, pos_end))
        pos_start = pos_end

    step_timings['eval_num_chunks'] = len(chunk_ranges)
    step_timings['eval_positions_per_chunk'] = max_positions_per_chunk

    # Accumulator for partial outputs (pre-sigmoid sum)
    # output = sigmoid(sum over chunks of: hidden_chunk @ W2_chunk.T)
    output_pre_sigmoid = jnp.zeros((pop_size, n_samples, num_outputs), dtype=jnp.float32)

    # Process chunks: compute hidden and partial output in one pass
    # This avoids storing all hidden activations at once
    for pos_start, pos_end in chunk_ranges:

        # Transfer weight chunks to GPU (convert uint8 masks back to bool)
        W1_chunk = jax.device_put(W1_cpu[:, :, pos_start:pos_end])
        W2_chunk = jax.device_put(W2_cpu[:, :, pos_start:pos_end])
        mask_chunk = jax.device_put(masks_cpu[:, pos_start:pos_end].astype(np.bool_))

        # Compute hidden and partial output for this chunk
        # hidden_chunk = tanh(inputs @ W1_chunk) * mask
        # partial_output = hidden_chunk @ W2_chunk.T (pre-sigmoid)

        def compute_chunk_contribution(w1_single, w2_single, mask_single, inputs):
            """Compute partial output contribution for one genome's chunk."""
            # inputs: (samples, num_inputs), w1_single: (num_inputs, chunk_size)
            hidden = jnp.tanh(safe_matmul(inputs, w1_single))  # (samples, chunk_size)
            hidden_masked = hidden * mask_single[None, :]  # Apply mask
            # w2_single: (num_outputs, chunk_size) -> need (chunk_size, num_outputs)
            w2_t = w2_single.T
            partial_out = safe_matmul(hidden_masked, w2_t)  # (samples, num_outputs)
            return partial_out

        chunk_contribution = jax.vmap(
            compute_chunk_contribution,
            in_axes=(0, 0, 0, None)
        )(W1_chunk, W2_chunk, mask_chunk, inputs_batch)  # (pop, samples, outputs)

        # Accumulate partial output
        output_pre_sigmoid = output_pre_sigmoid + chunk_contribution

        # Free GPU memory
        del W1_chunk, W2_chunk, mask_chunk, chunk_contribution

        algo._gpu_transfers_count += 3

    # Apply sigmoid and compute fitness
    outputs = jax.nn.sigmoid(output_pre_sigmoid)
    errors = jnp.mean((outputs - targets_batch[None, :, :]) ** 2, axis=(1, 2))
    fitnesses = 1.0 - errors

    step_timings['eval_time'] = time.perf_counter() - t_eval

    return fitnesses


def _evaluate_recurrent_chunked(
    algo,
    state: Any,
    cppns_transformed: Tuple,
    W1_cpu: np.ndarray,
    W2_cpu: np.ndarray,
    masks_cpu: np.ndarray,
    h_grid: Union[HierarchicalGridStructure, HierarchicalGridStructureSingle],
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    pop_size: int,
    pop_chunk_size: int,
    step_timings: Dict[str, Any],
    cached_sparse_hh: "Optional[Any]" = None,
    consult_cache_manager: bool = True,
) -> jnp.ndarray:
    """Evaluate recurrent networks with depth offloading.

    Both GPU grid (HierarchicalGridStructure) and CPU grid
    (HierarchicalGridStructureSingle) are supported. The CPU grid includes
    an all_positions field for h->h discovery, enabling depth 12+ recurrent
    and hidden_only modes.

    For recurrent mode, we need h->h connections which require the full
    weight matrices. We handle this by:
    1. Streaming W1/W2 back to GPU
    2. Discovering h->h connections
    3. Using double-chunked evaluation (population + samples)

    Args:
        state: Algorithm state
        cppns_transformed: Transformed CPPNs
        W1_cpu: Input weights on CPU
        W2_cpu: Output weights on CPU
        masks_cpu: Active masks on CPU
        h_grid: Hierarchical grid structure
        inputs_batch: Input data
        targets_batch: Target data
        pop_size: Population size
        pop_chunk_size: Chunk size for population
        step_timings: Timing dictionary

    Returns:
        Fitness scores - (pop,)
    """
    total_positions = W1_cpu.shape[2]

    # Transfer to GPU for h->h discovery.
    t_transfer = time.perf_counter()

    # Only the masks go across yet. The weights are transferred after compaction
    # below, so the full (pop, ., total_positions) arrays are never resident on the
    # device: that is the allocation this path exists to avoid.
    # Convert uint8 masks back to bool (memmap uses uint8 for offset compatibility)
    masks_A = jax.device_put(np.asarray(masks_cpu).astype(np.bool_))

    algo._gpu_transfers_count += 1
    step_timings['recurrent_transfer_time'] = time.perf_counter() - t_transfer

    # h->h connections: an explicit set first (a pass-through for a caller that
    # holds one; nothing in the package passes one), then the
    # cache manager under the same should_refresh contract as the hybrid path,
    # then discovery.
    t_hh = time.perf_counter()
    sparse_hh = cached_sparse_hh
    cache_hit = sparse_hh is not None
    # The manager caches POPULATION-level connections; the population-chunked
    # caller hands in per-chunk masks whose shapes differ chunk to chunk, so it
    # must neither consult nor update the manager (it sets
    # consult_cache_manager=False and keeps its per-chunk re-discovery).
    if sparse_hh is None and consult_cache_manager \
            and getattr(algo, '_hh_cache', None) is not None \
            and not algo._hh_cache.should_refresh(algo._current_generation, masks_A):
        sparse_hh = algo._hh_cache.get_cached()
        cache_hit = sparse_hh is not None
    if sparse_hh is None:
        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                state=state,
                cppns_transformed=cppns_transformed,
                h_grid=h_grid,
                masks_A=masks_A,
                # EMRConfig has no band_threshold field, so a hasattr guard on it is
                    # always false and falls back to 0.1 instead of the configured value.
                    # Use the substrate's setting, matching the non-offload path.
                    band_threshold=algo.band_threshold,
                max_weight=algo.max_weight,
                config=algo._recurrence_config,
                cppn_forward=algo._jitted_cppn_forward,
                pop_chunk_size=pop_chunk_size,
                verbose=algo._offload_verbose,
        )
        if consult_cache_manager and getattr(algo, '_hh_cache', None) is not None:
            algo._hh_cache.update_cache(sparse_hh, masks_A, algo._current_generation)
    step_timings['recurrent_hh_discovery'] = time.perf_counter() - t_hh
    step_timings['recurrent_hh_cache_hit'] = bool(cache_hit)

    # Track connection counts for sparse_hh_stats
    total_connections = int(jnp.sum(sparse_hh.num_valid))
    avg_connections = float(jnp.mean(sparse_hh.num_valid))
    max_connections = int(jnp.max(sparse_hh.num_valid))

    # Store for run_until_threshold to retrieve
    algo._last_hh_discovery_time_ms = step_timings['recurrent_hh_discovery'] * 1000
    algo._last_hh_connections = total_connections
    algo._last_avg_hh_connections = avg_connections
    # Recorded for the chunked caller, which accumulates it.
    algo._last_max_hh_connections = max_connections
    # A genome over `max_sparse_conns` never gets here: the discovery refuses it
    # (sparse_cap_refusal), on every path. A cached set passed the same rule when found.

    # Log connection counts
    if algo._offload_verbose:
        print(f"  [h->h Discovery] total={total_connections:,}, "
              f"avg={avg_connections:.1f}, max={max_connections:,}, "
              f"time={step_timings['recurrent_hh_discovery']:.2f}s")

    # Active-set compaction
    # The recurrent forward builds a dense (P, P) Whh over the substrate ceiling,
    # which is 30.5 GB per genome at depth 7 while the realized substrate is a
    # handful of positions. The feedforward path escapes this by chunking
    # positions; the recurrent one cannot, because the h->h scan gathers and
    # scatters across arbitrary positions so no chunk is self-contained.
    #
    # So evaluate on the union of positions any genome actually reached. Discovery
    # has to happen first (it needs the full masks and returns global indices),
    # after which the weights are gathered on the host and only the compact
    # arrays are transferred.
    t_compact = time.perf_counter()
    active = find_active_positions(masks_cpu)

    if active.num_active == 0:
        # Nothing was reached, so every hidden unit stays at zero and each output
        # is sigmoid(0) = 0.5. Short-circuit rather than feeding empty arrays into
        # vmap, which would fail on a zero-sized position axis.
        n_out = W2_cpu.shape[1]
        outputs = jnp.full(
            (pop_size, targets_batch.shape[0], n_out), 0.5, dtype=jnp.float32)
        errors = jnp.mean((outputs - targets_batch[None, :, :]) ** 2, axis=(1, 2))
        step_timings['recurrent_compact_time'] = time.perf_counter() - t_compact
        step_timings['recurrent_num_active'] = 0
        return jnp.maximum(0.0, 1.0 - errors)

    W1_gpu, W2_gpu = compact_weights(W1_cpu, W2_cpu, active)
    sparse_hh = remap_sparse_hh(sparse_hh, active)
    eval_positions = active.num_active

    algo._gpu_transfers_count += 2
    step_timings['recurrent_compact_time'] = time.perf_counter() - t_compact
    step_timings['recurrent_num_active'] = eval_positions

    if algo._offload_verbose:
        print(f"  [Compact] evaluating on {eval_positions:,} active of "
              f"{total_positions:,} positions "
              f"({active.reduction:,.0f}x reduction)")

    # Transpose W2 for evaluation
    W2_transposed = jnp.transpose(W2_gpu, (0, 2, 1))

    # The resolved value, not the raw config field. EMRConfig.activate_time is
    # Optional and defaults to None, with create_config deriving
    # (2 ** max_depth + 1) * 2 when h->h is on. Reading the config field works
    # only when offload is reached through set_recurrence_config with an
    # explicit value; every evaluation path in the algorithm, this one
    # included, uses the resolved attribute.
    activate_time = algo.activate_time

    # Calculate optimal chunk sizes
    n_samples = inputs_batch.shape[0]
    pop_chunk, sample_chunk = recurrent_chunk_sizes(
        pop_size=pop_size,
        # The compact space, not the ceiling. Sizing against total_positions would
        # budget memory for arrays that are never allocated and shrink the chunks
        # to nothing, discarding the reduction compaction just bought.
        positions=eval_positions,
        n_samples=n_samples,
        activate_time=activate_time,
        target_memory_gb=algo._recurrent_target_memory_gb,
    )

    step_timings['recurrent_pop_chunk'] = pop_chunk
    step_timings['recurrent_sample_chunk'] = sample_chunk

    # Double-chunked evaluation
    fitnesses = evaluate_population_recurrent_double_chunked(
        W1=W1_gpu,
        W2=W2_transposed,
        sparse_hh=sparse_hh,
        inputs_batch=inputs_batch,
        targets_batch=targets_batch,
        activate_time=activate_time,
        pop_chunk_size=pop_chunk,
        sample_chunk_size=sample_chunk,
    )

    return fitnesses


def estimate_offload_feasibility(
    depth: int = 11,
    pop_size: int = 50,
    target_cpu_gb: float = 16.0,
) -> Dict[str, Any]:
    """Report whether a (depth, population) pair is runnable on this machine.

    Args:
        depth: Target substrate depth
        pop_size: Population size
        target_cpu_gb: Target CPU RAM for population chunking


    Note:
        Formerly named `test_depth_offload`, which pytest collected as a
        test case. It is a calculator, not a test: it asserts nothing.

    Returns:
        Dictionary with memory estimates and feasibility assessment
    """
    estimates = estimate_memory_requirements(depth, pop_size, target_cpu_gb=target_cpu_gb)

    # Check feasibility
    gpu_fits = estimates['gpu_peak_gb'] < 10.0  # 10GB threshold for 11GB GPU
    cpu_required = estimates['cpu_required_gb']

    import psutil
    cpu_available_gb = psutil.virtual_memory().available / 1e9

    # With population chunking, we only need memory for one chunk
    memory_per_chunk = estimates['memory_per_chunk_gb']
    cpu_fits_with_chunking = cpu_available_gb > memory_per_chunk * 1.2  # 20% margin

    result = {
        **estimates,
        'gpu_fits_11gb': gpu_fits,
        'cpu_available_gb': cpu_available_gb,
        'cpu_fits': cpu_available_gb > cpu_required * 1.2,  # Without chunking
        'cpu_fits_with_chunking': cpu_fits_with_chunking,  # With chunking
        'feasible': gpu_fits and cpu_fits_with_chunking,
    }

    print(f"Depth {depth}, pop {pop_size}:")
    print(f"  Total positions: {estimates['total_positions']:,}")
    print(f"  GPU peak: {estimates['gpu_peak_gb']:.2f} GB ({'fits' if gpu_fits else 'OOM'})")
    print(f"  CPU required (full): {cpu_required:.2f} GB")
    print(f"  CPU available: {cpu_available_gb:.1f} GB")
    if estimates['needs_pop_chunking']:
        print(f"  Population chunking: {estimates['num_pop_chunks']} chunks of {estimates['pop_chunk_size']} genomes")
        print(f"  Memory per chunk: {memory_per_chunk:.2f} GB ({'fits' if cpu_fits_with_chunking else 'OOM'})")
    else:
        print("  Population chunking: Not needed")
    print(f"  Feasible: {'yes' if result['feasible'] else 'no'}")

    return result
