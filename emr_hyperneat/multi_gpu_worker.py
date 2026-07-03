"""Multi-GPU Worker Module for True Parallel Processing.

This module provides worker functions that can be spawned in separate processes
via ProcessPoolExecutor. Each worker has its own JAX runtime and JIT cache,
avoiding cross-device contamination.

CRITICAL: CUDA_VISIBLE_DEVICES must be set BEFORE importing JAX.
This is handled automatically by the worker initialization.

Architecture:
    - Main process: Computes W1, W2 matrices and h→h discovery on GPU 0
    - Worker processes: Perform evaluation in parallel on their assigned GPUs

This approach parallelizes the evaluation phase while avoiding the complexity
of CPPN forward function serialization.

Usage:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(max_workers=2) as executor:
        future = executor.submit(evaluate_shard_worker, device_id, shard_input)
        result = future.result()
"""

import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List


@dataclass
class WorkerConfig:
    """Configuration for a worker process."""
    device_id: int  # Physical device ID (0, 1, ...)
    activate_time: int
    allow_hidden_to_hidden: bool


@dataclass
class EvalShardInput:
    """Input data for evaluation-only worker (all numpy arrays for IPC).

    This is the simplified input for parallel evaluation.
    W1, W2 matrices and h→h connections are pre-computed in main process.
    """
    shard_start: int
    shard_end: int
    shard_size: int

    # Pre-computed weight matrices (numpy arrays)
    W1: np.ndarray  # shape: (shard_size, num_inputs, num_positions)
    W2: np.ndarray  # shape: (shard_size, num_positions, num_outputs)

    # Task data (numpy arrays)
    inputs: np.ndarray   # shape: (batch_size, input_dim)
    targets: np.ndarray  # shape: (batch_size, output_dim)

    # Optional h→h connections (numpy arrays)
    hh_from: Optional[np.ndarray] = None      # shape: (shard_size, max_hh_connections)
    hh_to: Optional[np.ndarray] = None        # shape: (shard_size, max_hh_connections)
    hh_weights: Optional[np.ndarray] = None   # shape: (shard_size, max_hh_connections)
    hh_valid: Optional[np.ndarray] = None     # shape: (shard_size, max_hh_connections)
    hh_num_valid: Optional[np.ndarray] = None # shape: (shard_size,)

    # Evaluation parameters
    activate_time: int = 10


# Legacy dataclass for backward compatibility (will be removed later)
@dataclass
class ShardInput:
    """Input data for a shard (all numpy arrays for IPC).

    DEPRECATED: Use EvalShardInput for evaluation-only parallelization.
    """
    shard_start: int
    shard_end: int

    # CPPN genome data (numpy arrays)
    cppn_nodes: np.ndarray  # shape: (shard_size, max_nodes, node_features)
    cppn_conns: np.ndarray  # shape: (shard_size, max_conns, conn_features)

    # Coordinates (numpy arrays)
    all_positions: np.ndarray  # shape: (num_positions, 3)
    input_coords: np.ndarray   # shape: (num_inputs, 3)
    output_coords: np.ndarray  # shape: (num_outputs, 3)

    # Task data (numpy arrays)
    inputs: np.ndarray   # shape: (batch_size, input_dim)
    targets: np.ndarray  # shape: (batch_size, output_dim)

    # Hierarchical grid data for variance computation
    h_grid_data: Dict[str, Any] = field(default_factory=dict)

    # Optional cached h→h connections (numpy arrays)
    cached_hh_from: Optional[np.ndarray] = None
    cached_hh_to: Optional[np.ndarray] = None
    cached_hh_weights: Optional[np.ndarray] = None
    cached_hh_valid: Optional[np.ndarray] = None
    cached_hh_num_valid: Optional[np.ndarray] = None

    # Optional pre-computed masks
    precomputed_masks_A: Optional[np.ndarray] = None


@dataclass
class EvalShardOutput:
    """Output from evaluation-only worker (all numpy arrays for IPC)."""
    shard_start: int
    shard_end: int
    fitnesses: np.ndarray  # shape: (shard_size,)

    # Timing info
    init_time_ms: float = 0.0
    transfer_time_ms: float = 0.0
    eval_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Error info (if any)
    error: Optional[str] = None


# Legacy dataclass for backward compatibility
@dataclass
class ShardOutput:
    """Output from a shard (all numpy arrays for IPC).

    DEPRECATED: Use EvalShardOutput for evaluation-only parallelization.
    """
    shard_start: int
    shard_end: int
    fitnesses: np.ndarray  # shape: (shard_size,)
    hh_count: int

    # Discovered h→h connections (if cache miss)
    discovered_hh_from: Optional[np.ndarray] = None
    discovered_hh_to: Optional[np.ndarray] = None
    discovered_hh_weights: Optional[np.ndarray] = None
    discovered_hh_valid: Optional[np.ndarray] = None
    discovered_hh_num_valid: Optional[np.ndarray] = None

    # Timing info
    init_time_ms: float = 0.0
    cppn_time_ms: float = 0.0
    hh_time_ms: float = 0.0
    eval_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Error info (if any)
    error: Optional[str] = None


# =============================================================================
# Persistent Worker Dataclasses (for true parallel h→h processing)
# =============================================================================

@dataclass
class PersistentWorkerConfig:
    """Configuration sent to worker at spawn time (once).

    This contains everything the worker needs to reconstruct the algorithm
    and process multiple generations without re-initialization.
    """
    # Required fields (no defaults) must come first
    gpu_id: int
    algorithm_config: Dict[str, Any]  # Serialized algo config dict
    problem_type: str                 # 'xor', 'and', etc.
    seed: int

    # EMR-specific config
    max_depth: int
    variance_threshold: float
    band_threshold: float
    max_weight: float
    activate_time: int
    allow_hidden_to_hidden: bool
    iteration_level: int

    # Substrate coordinates (numpy arrays)
    input_coords: np.ndarray   # shape: (num_inputs, 2)
    output_coords: np.ndarray  # shape: (num_outputs, 2)

    # Optional fields (with defaults) must come last
    hh_refresh_interval: int = 1
    hh_mask_change_threshold: float = 0.1  # For worker-local h→h caching
    hh_cache_enabled: bool = False          # Enable worker-local h→h caching
    pop_size: int = 100
    species_size: int = 10

    # Static task data (sent once, not per-generation)
    # When provided, workers don't need inputs/targets in per-gen tasks
    inputs: Optional[np.ndarray] = None   # shape: (batch_size, input_dim)
    targets: Optional[np.ndarray] = None  # shape: (batch_size, output_dim)


@dataclass
class EvalOnlyTask:
    """Lightweight evaluation-only task for centralized W1/W2/h→h computation.

    This is sent when main process does ALL expensive work:
    - CPPN queries (W1, W2 computation)
    - H→H discovery with caching

    Workers only need to do the final evaluation step.
    Much more efficient than GenerationTask because:
    1. No CPPN forward passes in workers (already done)
    2. H→H uses cached connections from main process
    3. Workers become simple evaluators
    """
    generation: int
    shard_start: int
    shard_end: int

    # Pre-computed weight matrices from main process
    W1: np.ndarray           # shape: (shard_size, num_inputs, num_positions)
    W2: np.ndarray           # shape: (shard_size, num_positions, num_outputs)

    # Pre-computed h→h connections (from main process cache)
    # None for feedforward mode
    hh_from: Optional[np.ndarray] = None      # shape: (shard_size, max_hh)
    hh_to: Optional[np.ndarray] = None        # shape: (shard_size, max_hh)
    hh_weights: Optional[np.ndarray] = None   # shape: (shard_size, max_hh)
    hh_valid: Optional[np.ndarray] = None     # shape: (shard_size, max_hh)


@dataclass
class GenerationTask:
    """Task sent to worker each generation (lightweight).

    Only contains the CPPN genomes which change each generation.
    All static config is in PersistentWorkerConfig (sent once).
    """
    generation: int
    shard_start: int
    shard_end: int

    # CPPN genomes for this shard (numpy arrays)
    cppn_nodes: np.ndarray   # shape: (shard_size, max_nodes, node_features)
    cppn_conns: np.ndarray   # shape: (shard_size, max_conns, conn_features)

    # Task data (may be static across generations but included for flexibility)
    inputs: np.ndarray       # shape: (batch_size, input_dim)
    targets: np.ndarray      # shape: (batch_size, output_dim)

    # Optional cached h→h (from previous generation if hh_refresh_interval > 1)
    cached_hh_from: Optional[np.ndarray] = None
    cached_hh_to: Optional[np.ndarray] = None
    cached_hh_weights: Optional[np.ndarray] = None
    cached_hh_valid: Optional[np.ndarray] = None


@dataclass
class GenerationResult:
    """Result from worker each generation."""
    shard_start: int
    shard_end: int
    fitnesses: np.ndarray    # shape: (shard_size,)

    # Discovered h→h connections (for caching in main process)
    hh_from: Optional[np.ndarray] = None
    hh_to: Optional[np.ndarray] = None
    hh_weights: Optional[np.ndarray] = None
    hh_valid: Optional[np.ndarray] = None
    hh_count: int = 0

    # Timing breakdown (milliseconds)
    cppn_time_ms: float = 0.0
    hh_time_ms: float = 0.0
    eval_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Error info (if any)
    error: Optional[str] = None


# Global worker state (set during initialization)
_worker_device_id: Optional[int] = None
_worker_jax_initialized: bool = False


def evaluate_shard_worker(
    device_id: int,
    shard_input: EvalShardInput,
) -> EvalShardOutput:
    """Evaluate a population shard on a single GPU.

    This function runs in a separate process with its own JAX runtime.
    All inputs/outputs are numpy arrays for IPC serialization.

    CRITICAL: Sets CUDA_VISIBLE_DEVICES BEFORE importing JAX.

    Args:
        device_id: Physical GPU device ID (0, 1, ...)
        shard_input: Pre-computed W1, W2, h→h data for evaluation

    Returns:
        EvalShardOutput with fitnesses and timing info
    """
    total_start = time.perf_counter()

    try:
        # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing JAX
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        # Import JAX after setting environment
        import jax
        import jax.numpy as jnp
        from jax import lax

        init_time = (time.perf_counter() - total_start) * 1000

        device = jax.devices()[0]  # Single device due to CUDA_VISIBLE_DEVICES

        with jax.default_device(device):
            transfer_start = time.perf_counter()

            # Convert numpy arrays to JAX arrays on device
            W1 = jax.device_put(jnp.array(shard_input.W1), device)
            W2 = jax.device_put(jnp.array(shard_input.W2), device)
            inputs = jax.device_put(jnp.array(shard_input.inputs), device)
            targets = jax.device_put(jnp.array(shard_input.targets), device)

            # Convert h→h connections if present
            has_hh = shard_input.hh_from is not None
            if has_hh:
                hh_from = jax.device_put(jnp.array(shard_input.hh_from), device)
                hh_to = jax.device_put(jnp.array(shard_input.hh_to), device)
                hh_weights = jax.device_put(jnp.array(shard_input.hh_weights), device)
                hh_valid = jax.device_put(jnp.array(shard_input.hh_valid), device)

            transfer_time = (time.perf_counter() - transfer_start) * 1000

            # Evaluation
            eval_start = time.perf_counter()
            activate_time = shard_input.activate_time

            if has_hh:
                # Hybrid evaluation with h→h connections
                def eval_single_hybrid(w1, w2, fr, to, wt, vm):
                    """Evaluate single network with h→h connections."""
                    num_hidden = w1.shape[1]

                    def step_fn(hidden, _):
                        # Input to hidden
                        pre_hidden = inputs @ w1

                        # H→H contribution (sparse)
                        hh_contrib = jnp.zeros(num_hidden)

                        def add_hh_connection(carry, idx):
                            hh_c, fr_, to_, wt_, vm_ = carry
                            valid = vm_[idx]
                            src = fr_[idx]
                            dst = to_[idx]
                            weight = wt_[idx]
                            hh_c = hh_c.at[dst].add(
                                jnp.where(valid, hidden[src] * weight, 0.0)
                            )
                            return (hh_c, fr_, to_, wt_, vm_), None

                        num_conns = fr.shape[0]
                        (hh_contrib, _, _, _, _), _ = lax.scan(
                            add_hh_connection,
                            (hh_contrib, fr, to, wt, vm),
                            jnp.arange(num_conns),
                        )

                        new_hidden = jnp.tanh(pre_hidden + hh_contrib)
                        return new_hidden, None

                    hidden_init = jnp.zeros((inputs.shape[0], num_hidden))
                    hidden_final, _ = lax.scan(step_fn, hidden_init, None, length=activate_time)

                    outputs = jax.nn.sigmoid(hidden_final @ w2)
                    errors = jnp.mean((outputs - targets) ** 2, axis=1)
                    return 1.0 - jnp.mean(errors)

                # Device-specific JIT for evaluation
                eval_vmapped = jax.jit(
                    lambda w1, w2, fr, to, wt, vm: jax.vmap(eval_single_hybrid)(
                        w1, w2, fr, to, wt, vm
                    ),
                    device=device,
                )

                fitnesses = eval_vmapped(W1, W2, hh_from, hh_to, hh_weights, hh_valid)

            else:
                # Dense evaluation (no h→h)
                def eval_single_dense(w1, w2):
                    """Evaluate single network without h→h connections."""
                    hidden = jnp.tanh(inputs @ w1)
                    outputs = jax.nn.sigmoid(hidden @ w2)
                    errors = jnp.mean((outputs - targets) ** 2, axis=1)
                    return 1.0 - jnp.mean(errors)

                # Device-specific JIT for evaluation
                eval_vmapped = jax.jit(
                    lambda w1, w2: jax.vmap(eval_single_dense)(w1, w2),
                    device=device,
                )

                fitnesses = eval_vmapped(W1, W2)

            # Block and convert to numpy
            fitnesses_np = np.array(jax.device_get(fitnesses))

            eval_time = (time.perf_counter() - eval_start) * 1000
            total_time = (time.perf_counter() - total_start) * 1000

            return EvalShardOutput(
                shard_start=shard_input.shard_start,
                shard_end=shard_input.shard_end,
                fitnesses=fitnesses_np,
                init_time_ms=init_time,
                transfer_time_ms=transfer_time,
                eval_time_ms=eval_time,
                total_time_ms=total_time,
                error=None,
            )

    except Exception as e:
        import traceback
        return EvalShardOutput(
            shard_start=shard_input.shard_start,
            shard_end=shard_input.shard_end,
            fitnesses=np.zeros(shard_input.shard_size),
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            total_time_ms=(time.perf_counter() - total_start) * 1000,
        )


# Legacy worker functions (for backward compatibility)
def worker_init(device_id: int):
    """Initialize worker process with specific GPU.

    DEPRECATED: evaluate_shard_worker handles device setup internally.
    """
    global _worker_device_id, _worker_jax_initialized
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    _worker_device_id = device_id
    _worker_jax_initialized = False


def _init_jax():
    """Lazily initialize JAX in the worker process.

    DEPRECATED: evaluate_shard_worker handles JAX init internally.
    """
    global _worker_jax_initialized

    if _worker_jax_initialized:
        return

    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(
            f"Worker {_worker_device_id} expected 1 device, got {len(devices)}: {devices}"
        )

    _worker_jax_initialized = True


def process_shard_worker(
    config: WorkerConfig,
    shard_input: ShardInput,
    state_arrays: Dict[str, np.ndarray],
) -> ShardOutput:
    """Process a population shard on a single GPU.

    DEPRECATED: Use evaluate_shard_worker instead.
    This is kept for backward compatibility but returns dummy results.
    """
    total_start = time.perf_counter()

    return ShardOutput(
        shard_start=shard_input.shard_start,
        shard_end=shard_input.shard_end,
        fitnesses=np.ones(shard_input.shard_end - shard_input.shard_start) * 0.5,
        hh_count=0,
        error="DEPRECATED: Use evaluate_shard_worker instead",
        total_time_ms=(time.perf_counter() - total_start) * 1000,
    )


# =============================================================================
# Persistent Worker Implementation (true parallel h→h processing)
# =============================================================================

def persistent_worker_main(
    config: PersistentWorkerConfig,
    task_queue,  # mp.Queue
    result_queue,  # mp.Queue
    shutdown_event,  # mp.Event
) -> None:
    """Main loop for persistent worker process.

    This worker stays alive across all generations, maintaining its own:
    - JAX runtime (isolated via CUDA_VISIBLE_DEVICES)
    - Algorithm instance with cppn_forward JIT function
    - Hierarchical grid (pre-computed once)

    CRITICAL: CUDA_VISIBLE_DEVICES must be set BEFORE importing JAX.

    Args:
        config: Worker configuration (sent once at spawn)
        task_queue: Queue to receive GenerationTask objects
        result_queue: Queue to send GenerationResult objects
        shutdown_event: Event to signal worker shutdown
    """
    import queue as queue_module  # Avoid name collision

    # 1. Set GPU and memory management BEFORE importing JAX
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    # Prevent JAX from preallocating all GPU memory
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Allow JAX to grow memory as needed (platform allocator)
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    try:
        # 2. Import JAX (after environment is set)
        import jax
        import jax.numpy as jnp
        from jax import random

        # 3. Import algorithm components (imports JAX-dependent modules)
        from emr_hyperneat.emrhyperneat import (
            EMRHyperNEAT,
            HHCacheManager,
            EMRConfig,
        )
        from emr_hyperneat.emrhyperneat_base import (
            get_hierarchical_grid,
            batch_query_population_multi_source_chunked,
            compute_hierarchical_variances_batch,
            compute_subdivision_masks_batch,
        )
        from emr_hyperneat._compat.problems.benchmarks import (
            XORProblem, ANDProblem, ORProblem, NANDProblem, NORProblem,
        )

        # Map problem type string to class
        problem_map = {
            'xor': XORProblem,
            'and': ANDProblem,
            'or': ORProblem,
            'nand': NANDProblem,
            'nor': NORProblem,
        }

        # 4. Create problem instance
        problem_class = problem_map.get(config.problem_type.lower(), XORProblem)
        problem = problem_class()

        # 5. Create algorithm instance (this creates its own cppn_forward)
        algo = EMRHyperNEAT()
        neat_config = algo.create_config(config.algorithm_config)
        state = algo.initialize(neat_config, problem, seed=config.seed)

        # 6. Get the JITted CPPN forward function
        cppn_forward = jax.jit(
            algo.neat_algo.genome.forward,
            static_argnums=(0,),
        )

        # 6b. Get the JITted transform function (transforms (nodes, conns) -> cppns_transformed)
        transform_batch = jax.jit(
            jax.vmap(algo.neat_algo.transform, in_axes=(None, (0, 0)))
        )

        # 7. Pre-compute hierarchical grid (reused across generations)
        h_grid = get_hierarchical_grid(config.max_depth)

        # 8. Convert coordinates to JAX arrays
        input_coords_jax = jnp.array(config.input_coords)
        output_coords_jax = jnp.array(config.output_coords)

        # 9. Get the device (should be only one due to CUDA_VISIBLE_DEVICES)
        device = jax.devices()[0]

        # 10. Pre-convert static task data if provided in config (sent once)
        # This avoids IPC overhead for inputs/targets every generation
        inputs_jax = None
        targets_jax = None
        if config.inputs is not None and config.targets is not None:
            inputs_jax = jax.device_put(jnp.array(config.inputs), device)
            targets_jax = jax.device_put(jnp.array(config.targets), device)

        # 11. Initialize worker-local h→h cache (if enabled)
        # This allows each worker to cache its h→h connections between generations
        hh_cache = None
        extended_config = None
        print(f"[Worker {config.gpu_id}] h→h cache: enabled={config.hh_cache_enabled}, allow_hh={config.allow_hidden_to_hidden}", flush=True)
        if config.hh_cache_enabled and config.allow_hidden_to_hidden:
            print(f"[Worker {config.gpu_id}] Initializing HHCacheManager with refresh_interval={config.hh_refresh_interval}, threshold={config.hh_mask_change_threshold}", flush=True)
            extended_config = EMRConfig(
                enabled=True,
                allow_hidden_to_hidden=True,
                allow_backward=True,
                allow_lateral=True,
                allow_self_loops=True,
                iteration_level=config.iteration_level,
                activate_time=config.activate_time,
                max_connections=10000,
                use_vectorized_discovery=True,
                max_sparse_conns=10000,
                multi_hop_algorithm="matrix_power",
                hop_decay_factor=0.8,
                hh_cache_enabled=True,
                hh_refresh_interval=config.hh_refresh_interval,
                hh_mask_change_threshold=config.hh_mask_change_threshold,
                use_dense_discovery=False,
            )
            hh_cache = HHCacheManager(extended_config)

        # Signal successful initialization
        result_queue.put(GenerationResult(
            shard_start=0,
            shard_end=0,
            fitnesses=np.array([]),
            error=None,
            total_time_ms=0.0,
        ))

        # 11. Main worker loop
        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break

                # Handle both task types
                if isinstance(task, EvalOnlyTask):
                    # FAST PATH: Evaluation-only (main process did W1/W2/h→h)
                    # Use pre-converted inputs/targets from config
                    if inputs_jax is None or targets_jax is None:
                        result = GenerationResult(
                            shard_start=task.shard_start,
                            shard_end=task.shard_end,
                            fitnesses=np.zeros(task.shard_end - task.shard_start),
                            error="EvalOnlyTask requires inputs/targets in PersistentWorkerConfig",
                        )
                    else:
                        result = _process_eval_only_task(
                            task=task,
                            config=config,
                            inputs=inputs_jax,
                            targets=targets_jax,
                            device=device,
                        )
                else:
                    # FULL PIPELINE PATH: Worker does CPPN queries + h→h + eval (GenerationTask)
                    # With worker-local caching for h→h connections
                    result = _process_generation_shard(
                        task=task,
                        config=config,
                        state=state,
                        cppn_forward=cppn_forward,
                        transform_batch=transform_batch,
                        h_grid=h_grid,
                        input_coords=input_coords_jax,
                        output_coords=output_coords_jax,
                        problem=problem,
                        device=device,
                        hh_cache=hh_cache,
                        extended_config=extended_config,
                    )
                result_queue.put(result)

            except queue_module.Empty:
                continue
            except Exception as e:
                import traceback
                result_queue.put(GenerationResult(
                    shard_start=task.shard_start if task else 0,
                    shard_end=task.shard_end if task else 0,
                    fitnesses=np.zeros(task.shard_end - task.shard_start if task else 0),
                    error=f"Worker {config.gpu_id} error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                ))

    except Exception as e:
        import traceback
        result_queue.put(GenerationResult(
            shard_start=0,
            shard_end=0,
            fitnesses=np.array([]),
            error=f"Worker {config.gpu_id} init error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))


def _process_generation_shard(
    task: GenerationTask,
    config: PersistentWorkerConfig,
    state,
    cppn_forward,
    transform_batch,
    h_grid,
    input_coords,
    output_coords,
    problem,
    device,
    hh_cache=None,
    extended_config=None,
) -> GenerationResult:
    """Process one generation's shard on this worker's GPU.

    This implements the FULL h→h pipeline:
    1. CPPN queries (input→positions, output←positions)
    2. Variance computation and mask generation
    3. H→H discovery (if enabled and not cached) WITH WORKER-LOCAL CACHING
    4. Network evaluation

    Args:
        task: GenerationTask with CPPN genomes and task data
        config: Worker config with algorithm parameters
        state: Algorithm state (for genome forward function)
        cppn_forward: JITted CPPN forward function
        transform_batch: JITted transform function for CPPN population
        h_grid: Pre-computed hierarchical grid
        input_coords: Input substrate coordinates (JAX array)
        output_coords: Output substrate coordinates (JAX array)
        problem: Problem instance
        device: JAX device for this worker
        hh_cache: Optional HHCacheManager for worker-local h→h caching
        extended_config: Optional EMRConfig for h→h discovery

    Returns:
        GenerationResult with fitnesses and timing info
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from emr_hyperneat.emrhyperneat_base import (
        batch_query_population_multi_source_chunked,
        compute_hierarchical_variances_batch,
        compute_subdivision_masks_batch,
    )
    from emr_hyperneat.emrhyperneat import (
        discover_sparse_hh_vectorized_multi_hop,
        EMRConfig,
    )

    total_start = time.perf_counter()
    shard_size = task.shard_end - task.shard_start

    try:
        with jax.default_device(device):
            # Convert task data to JAX arrays on device
            cppn_nodes = jax.device_put(jnp.array(task.cppn_nodes), device)
            cppn_conns = jax.device_put(jnp.array(task.cppn_conns), device)
            inputs = jax.device_put(jnp.array(task.inputs), device)
            targets = jax.device_put(jnp.array(task.targets), device)

            # Get all positions from hierarchical grid
            all_positions = h_grid.all_positions

            # ==================================================================
            # 1. Transform CPPNs and Query
            # ==================================================================
            cppn_start = time.perf_counter()

            # Transform (nodes, conns) -> cppns_transformed tuple of 4 arrays
            # This is necessary for batch_query_population_multi_source_chunked
            cppns_transformed = transform_batch(state, (cppn_nodes, cppn_conns))

            # Query input→all positions (for W1)
            # W1 shape: (shard_size, num_inputs, num_positions)
            W1 = batch_query_population_multi_source_chunked(
                state, cppns_transformed,
                input_coords, all_positions,
                outgoing=True,
                cppn_forward=cppn_forward,
                pop_chunk_size=50,
                device_id=0,  # Worker only sees one device due to CUDA_VISIBLE_DEVICES
            )

            # Query output←all positions (for W2)
            # Returns: (shard_size, num_outputs, num_positions)
            # Need to transpose to: (shard_size, num_positions, num_outputs)
            W2_raw = batch_query_population_multi_source_chunked(
                state, cppns_transformed,
                output_coords, all_positions,
                outgoing=False,
                cppn_forward=cppn_forward,
                pop_chunk_size=50,
                device_id=0,
            )
            # ==================================================================
            # 1b. Apply Weight Processing (CRITICAL for XOR)
            # ==================================================================
            # Match main algorithm's processing at lines 5994-6008
            # Without this, raw CPPN outputs are unbounded and dense,
            # preventing proper XOR learning (0.75 fitness plateau)
            weight_thresh = 0.1
            max_weight_val = config.max_weight

            # Compute masks_A for W1/W2 masking (need to do this for ALL modes, not just h→h)
            # Use first input's weights for variance computation
            all_weights_for_variance = W1[:, 0, :]  # (shard_size, total_positions)
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
            masks_A = compute_subdivision_masks_batch(
                level_variances, config.variance_threshold, h_grid, return_all_masks=False
            )

            # Broadcast mask for (pop, inputs/outputs, positions) shape
            active_mask_broadcast = masks_A[:, None, :]

            # Apply tanh + scaling + masking to W1
            W1_raw = jnp.tanh(W1) * max_weight_val
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw, 0.0
            )

            # Apply tanh + scaling + masking to W2 (before transpose)
            W2_pre = jnp.tanh(W2_raw) * max_weight_val
            W2_masked = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_pre) > weight_thresh),
                W2_pre, 0.0
            )
            # Transpose: (shard_size, num_outputs, num_positions) -> (shard_size, num_positions, num_outputs)
            W2 = jnp.transpose(W2_masked, (0, 2, 1))

            cppn_time_ms = (time.perf_counter() - cppn_start) * 1000

            # ==================================================================
            # 2. H→H Discovery (if enabled)
            # ==================================================================
            hh_start = time.perf_counter()
            hh_from = None
            hh_to = None
            hh_weights = None
            hh_valid = None
            hh_count = 0

            cache_hit = False
            if config.allow_hidden_to_hidden:
                # Check for worker-local cache first (most efficient path)
                if hh_cache is not None and extended_config is not None:
                    # Worker-local caching: check if we can use cached h→h
                    if not hh_cache.should_refresh(task.generation, masks_A):
                        # CACHE HIT: Reuse cached h→h connections
                        cached = hh_cache.get_cached()
                        if cached is not None:
                            hh_from = cached.from_indices
                            hh_to = cached.to_indices
                            hh_weights = cached.weights
                            hh_valid = cached.valid_mask
                            hh_count = int(jnp.sum(cached.num_valid))
                            cache_hit = True

                if not cache_hit:
                    # CACHE MISS: Need to discover h→h
                    # Create or use provided extended_config
                    local_extended_config = extended_config if extended_config is not None else EMRConfig(
                        enabled=True,
                        allow_hidden_to_hidden=config.allow_hidden_to_hidden,
                        allow_backward=True,
                        allow_lateral=True,
                        allow_self_loops=True,
                        iteration_level=config.iteration_level,
                        activate_time=config.activate_time,
                        max_connections=10000,
                        use_vectorized_discovery=True,
                        max_sparse_conns=10000,
                        multi_hop_algorithm="matrix_power",
                        hop_decay_factor=0.8,
                        hh_cache_enabled=False,
                        hh_refresh_interval=config.hh_refresh_interval,
                        use_dense_discovery=False,
                    )

                    # Discover h→h connections
                    sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                        state, cppns_transformed, h_grid, masks_A,
                        config.band_threshold, config.max_weight,
                        local_extended_config, cppn_forward,
                        pop_chunk_size=50,
                        verbose=False,
                        global_union_active=None,
                        device_id=0,
                    )

                    # Extract sparse connection data
                    hh_from = sparse_hh.from_indices
                    hh_to = sparse_hh.to_indices
                    hh_weights = sparse_hh.weights
                    hh_valid = sparse_hh.valid_mask
                    hh_count = int(jnp.sum(sparse_hh.num_valid))

                    # Update worker-local cache
                    if hh_cache is not None:
                        hh_cache.update_cache(sparse_hh, masks_A, task.generation)

            elif task.cached_hh_from is not None:
                # Use task-provided cached h→h connections (from main process)
                hh_from = jax.device_put(jnp.array(task.cached_hh_from), device)
                hh_to = jax.device_put(jnp.array(task.cached_hh_to), device)
                hh_weights = jax.device_put(jnp.array(task.cached_hh_weights), device)
                hh_valid = jax.device_put(jnp.array(task.cached_hh_valid), device)

            hh_time_ms = (time.perf_counter() - hh_start) * 1000

            # ==================================================================
            # 3. Network Evaluation
            # ==================================================================
            eval_start = time.perf_counter()

            has_hh = hh_from is not None

            if has_hh:
                # Evaluation with h→h connections
                fitnesses = _evaluate_networks_with_hh(
                    W1, W2, hh_from, hh_to, hh_weights, hh_valid,
                    inputs, targets, config.activate_time, device,
                )
            else:
                # Simple feedforward evaluation (no h→h)
                fitnesses = _evaluate_networks_dense(
                    W1, W2, inputs, targets, device,
                )

            # Block and convert to numpy
            fitnesses_np = np.array(jax.device_get(fitnesses))

            eval_time_ms = (time.perf_counter() - eval_start) * 1000
            total_time_ms = (time.perf_counter() - total_start) * 1000

            return GenerationResult(
                shard_start=task.shard_start,
                shard_end=task.shard_end,
                fitnesses=fitnesses_np,
                hh_from=np.array(jax.device_get(hh_from)) if hh_from is not None else None,
                hh_to=np.array(jax.device_get(hh_to)) if hh_to is not None else None,
                hh_weights=np.array(jax.device_get(hh_weights)) if hh_weights is not None else None,
                hh_valid=np.array(jax.device_get(hh_valid)) if hh_valid is not None else None,
                hh_count=hh_count,
                cppn_time_ms=cppn_time_ms,
                hh_time_ms=hh_time_ms,
                eval_time_ms=eval_time_ms,
                total_time_ms=total_time_ms,
                error=None,
            )

    except Exception as e:
        import traceback
        return GenerationResult(
            shard_start=task.shard_start,
            shard_end=task.shard_end,
            fitnesses=np.zeros(shard_size),
            error=f"Shard processing error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
            total_time_ms=(time.perf_counter() - total_start) * 1000,
        )


def _batch_query_cppn_chunked(
    cppn_nodes,
    cppn_conns,
    source_coords,
    target_coords,
    cppn_forward,
    state,
    is_input: bool,
    chunk_size: int = 50,
):
    """Query CPPN for weight matrices, processing population in chunks.

    Args:
        cppn_nodes: CPPN node data (shard_size, max_nodes, features)
        cppn_conns: CPPN connection data (shard_size, max_conns, features)
        source_coords: Source coordinates (num_sources, 2)
        target_coords: Target coordinates (num_targets, 2)
        cppn_forward: JITted CPPN forward function
        state: Algorithm state
        is_input: True for input→hidden, False for hidden→output
        chunk_size: Population chunk size

    Returns:
        Weight matrix (shard_size, num_sources, num_targets) if is_input
        else (shard_size, num_targets, num_sources)
    """
    import jax.numpy as jnp

    shard_size = cppn_nodes.shape[0]
    num_sources = source_coords.shape[0]
    num_targets = target_coords.shape[0]

    # Simplified CPPN query - actual implementation would use
    # batch_query_population_multi_source_chunked from the main module
    # returns random weights (multi-GPU sync not implemented in this release)

    if is_input:
        # W1: (shard_size, num_sources, num_targets)
        return jnp.zeros((shard_size, num_sources, num_targets))
    else:
        # W2: (shard_size, num_targets, num_sources)
        return jnp.zeros((shard_size, num_targets, num_sources))


def _evaluate_networks_dense(
    W1, W2, inputs, targets, device,
):
    """Evaluate networks without h→h connections (feedforward only).

    Args:
        W1: Input→hidden weights (shard_size, num_inputs, num_hidden)
        W2: Hidden→output weights (shard_size, num_hidden, num_outputs)
        inputs: Task inputs (batch_size, input_dim)
        targets: Task targets (batch_size, output_dim)
        device: JAX device

    Returns:
        Fitness values (shard_size,)
    """
    import jax
    import jax.numpy as jnp

    def eval_single(w1, w2):
        """Evaluate single network."""
        # Forward pass
        hidden = jnp.tanh(inputs @ w1)  # (batch, hidden)
        outputs = jax.nn.sigmoid(hidden @ w2)  # (batch, output)

        # Compute fitness (1 - MSE)
        mse = jnp.mean((outputs - targets) ** 2)
        return 1.0 - mse

    # Vectorize over population
    eval_fn = jax.jit(jax.vmap(eval_single), device=device)
    return eval_fn(W1, W2)


def _evaluate_networks_with_hh(
    W1, W2, hh_from, hh_to, hh_weights, hh_valid,
    inputs, targets, activate_time, device,
):
    """Evaluate networks with h→h connections (recurrent).

    Uses vectorized scatter_add for efficient sparse h→h aggregation,
    matching the approach in the main algorithm.

    Args:
        W1: Input→hidden weights (shard_size, num_inputs, num_hidden)
        W2: Hidden→output weights (shard_size, num_hidden, num_outputs)
        hh_from: Source indices for h→h (shard_size, max_hh)
        hh_to: Target indices for h→h (shard_size, max_hh)
        hh_weights: H→H connection weights (shard_size, max_hh)
        hh_valid: Valid mask for h→h (shard_size, max_hh)
        inputs: Task inputs (batch_size, input_dim)
        targets: Task targets (batch_size, output_dim)
        activate_time: Number of activation steps
        device: JAX device

    Returns:
        Fitness values (shard_size,)
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    def eval_single(w1, w2, fr, to, wt, vm):
        """Evaluate single network with h→h connections using vectorized scatter_add."""
        num_hidden = w1.shape[1]
        batch_size = inputs.shape[0]

        # Precompute safe indices (clip to valid range)
        safe_from = jnp.clip(fr, 0, num_hidden - 1)
        safe_to = jnp.clip(to, 0, num_hidden - 1)
        # Effective weights: zero out invalid connections
        effective_hh_w = jnp.where(vm, wt, 0.0)

        def step_fn(hidden, _):
            # Input→hidden (W1: (num_inputs, num_hidden))
            pre_hidden = inputs @ w1  # (batch, hidden)

            # H→H contribution using vectorized scatter_add
            # source_vals: (batch, num_conns) - gather from hidden state
            source_vals = hidden[:, safe_from]  # (batch, max_hh)
            # contributions: (batch, num_conns)
            contributions = source_vals * effective_hh_w  # broadcast weight over batch
            # Scatter-add to target positions
            hh_contrib = jnp.zeros_like(hidden)
            hh_contrib = hh_contrib.at[:, safe_to].add(contributions)

            # Combine and activate
            new_hidden = jnp.tanh(pre_hidden + hh_contrib)
            return new_hidden, None

        # Initialize hidden state
        hidden_init = jnp.zeros((batch_size, num_hidden))

        # Run activation steps using lax.scan (JIT-friendly)
        hidden_final, _ = lax.scan(step_fn, hidden_init, None, length=activate_time)

        # Output (W2: (num_hidden, num_outputs))
        outputs = jax.nn.sigmoid(hidden_final @ w2)

        # Compute fitness (1 - MSE)
        mse = jnp.mean((outputs - targets) ** 2)
        return 1.0 - mse

    # Vectorize over population
    eval_fn = jax.jit(
        jax.vmap(eval_single),
        device=device,
    )
    return eval_fn(W1, W2, hh_from, hh_to, hh_weights, hh_valid)


def _process_eval_only_task(
    task: 'EvalOnlyTask',
    config: PersistentWorkerConfig,
    inputs: 'jnp.ndarray',
    targets: 'jnp.ndarray',
    device: Any,
) -> GenerationResult:
    """Process evaluation-only task (main process did W1/W2/h→h computation).

    This is the FAST path where:
    - Main process computed W1, W2 matrices centrally (on GPU 0)
    - Main process discovered h→h connections with caching
    - Worker only needs to evaluate fitness

    Args:
        task: EvalOnlyTask with pre-computed W1, W2, h→h data
        config: Worker config with activate_time
        inputs: Task inputs (JAX array on device)
        targets: Task targets (JAX array on device)
        device: JAX device for this worker

    Returns:
        GenerationResult with fitnesses and timing info
    """
    import jax
    import jax.numpy as jnp

    total_start = time.perf_counter()
    shard_size = task.shard_end - task.shard_start

    try:
        with jax.default_device(device):
            # Transfer pre-computed arrays to device
            transfer_start = time.perf_counter()

            W1 = jax.device_put(jnp.array(task.W1), device)
            W2 = jax.device_put(jnp.array(task.W2), device)

            # Transfer h→h if present
            has_hh = task.hh_from is not None
            if has_hh:
                hh_from = jax.device_put(jnp.array(task.hh_from), device)
                hh_to = jax.device_put(jnp.array(task.hh_to), device)
                hh_weights = jax.device_put(jnp.array(task.hh_weights), device)
                hh_valid = jax.device_put(jnp.array(task.hh_valid), device)

            transfer_time_ms = (time.perf_counter() - transfer_start) * 1000

            # Evaluation
            eval_start = time.perf_counter()

            if has_hh:
                fitnesses = _evaluate_networks_with_hh(
                    W1, W2, hh_from, hh_to, hh_weights, hh_valid,
                    inputs, targets, config.activate_time, device,
                )
            else:
                fitnesses = _evaluate_networks_dense(
                    W1, W2, inputs, targets, device,
                )

            # Block and convert to numpy
            fitnesses_np = np.array(jax.device_get(fitnesses))

            eval_time_ms = (time.perf_counter() - eval_start) * 1000
            total_time_ms = (time.perf_counter() - total_start) * 1000

            return GenerationResult(
                shard_start=task.shard_start,
                shard_end=task.shard_end,
                fitnesses=fitnesses_np,
                # No h→h discovery in workers - main process has it
                hh_from=None,
                hh_to=None,
                hh_weights=None,
                hh_valid=None,
                hh_count=0,
                cppn_time_ms=0.0,  # No CPPN work in workers
                hh_time_ms=transfer_time_ms,  # Just transfer time
                eval_time_ms=eval_time_ms,
                total_time_ms=total_time_ms,
                error=None,
            )

    except Exception as e:
        import traceback
        return GenerationResult(
            shard_start=task.shard_start,
            shard_end=task.shard_end,
            fitnesses=np.zeros(shard_size),
            error=f"EvalOnly processing error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
            total_time_ms=(time.perf_counter() - total_start) * 1000,
        )


def test_worker_isolation():
    """Test that worker processes have isolated JAX runtimes.

    Run this to verify the ProcessPoolExecutor setup works correctly.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # MUST use spawn to get fresh Python interpreters
    mp.set_start_method('spawn', force=True)

    def test_device_in_worker(device_id: int) -> Dict[str, Any]:
        """Worker function that reports its device configuration."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        import jax

        devices = jax.devices()
        backend = jax.default_backend()

        return {
            'device_id': device_id,
            'visible_devices': str(os.environ.get("CUDA_VISIBLE_DEVICES")),
            'jax_devices': [str(d) for d in devices],
            'backend': backend,
            'pid': os.getpid(),
        }

    num_gpus = 2  # Assuming 2 GPUs

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(test_device_in_worker, d) for d in range(num_gpus)]
        results = [f.result() for f in futures]

    print("\n=== Worker Isolation Test Results ===")
    for r in results:
        print(f"Device {r['device_id']}: PID={r['pid']}, "
              f"CUDA_VISIBLE_DEVICES={r['visible_devices']}, "
              f"JAX sees: {r['jax_devices']}, backend={r['backend']}")

    # Verify isolation
    pids = [r['pid'] for r in results]
    if len(set(pids)) == num_gpus:
        print("✓ All workers have separate PIDs (process isolation confirmed)")
    else:
        print("✗ WARNING: Workers share PIDs (process isolation may be broken)")

    return results


if __name__ == "__main__":
    # Run isolation test when module is executed directly
    test_worker_isolation()
