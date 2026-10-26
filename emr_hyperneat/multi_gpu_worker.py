"""Multi-GPU Worker Module for True Parallel Processing.

This module provides worker functions that can be spawned in separate processes
via ProcessPoolExecutor. Each worker has its own JAX runtime and JIT cache,
avoiding cross-device contamination.

CUDA_VISIBLE_DEVICES must be set before JAX initializes, and the package initializes JAX at
import (unpickling a worker function imports it), so a worker cannot set the variable itself:
in 0.1.0 each worker set it inside the function, too late, and every worker ran on the first
GPU. start_device_workers starts one single-worker pool per device with that device's variable
already in the child's environment.

Architecture:
    - Main process: Computes W1, W2 matrices and h->h discovery on GPU 0
    - Worker processes: evaluate population shards in parallel, one per device

This approach parallelizes the evaluation phase while avoiding the complexity
of CPPN forward function serialization.

Usage:
    import multiprocessing as mp

    pools = start_device_workers(2, mp.get_context('spawn'))
    futures = [pools[d].submit(evaluate_shard_worker, d, shard_inputs[d]) for d in range(2)]
    fitnesses = combine_shard_fitnesses([f.result() for f in futures], pop_size)
"""

import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass
class EvalShardInput:
    """Input data for evaluation-only worker (all numpy arrays for IPC).

    This is the simplified input for parallel evaluation.
    W1, W2 matrices and h->h connections are pre-computed in main process.
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

    # Optional h->h connections (numpy arrays)
    hh_from: Optional[np.ndarray] = None      # shape: (shard_size, max_hh_connections)
    hh_to: Optional[np.ndarray] = None        # shape: (shard_size, max_hh_connections)
    hh_weights: Optional[np.ndarray] = None   # shape: (shard_size, max_hh_connections)
    hh_valid: Optional[np.ndarray] = None     # shape: (shard_size, max_hh_connections)
    hh_num_valid: Optional[np.ndarray] = None # shape: (shard_size,)

    # Evaluation parameters
    activate_time: int = 10


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


def evaluate_shard_worker(
    device_id: int,
    shard_input: EvalShardInput,
) -> EvalShardOutput:
    """Evaluate a population shard on a single GPU.

    This function runs in a separate process with its own JAX runtime, started by
    start_device_workers with this device's CUDA_VISIBLE_DEVICES, so on GPUs it sees one GPU.
    All inputs/outputs are numpy arrays for IPC serialization. With h->h connections it scores
    the shard with the kernel EVAL_ONLY_PARALLEL uses (_pmap_eval_batch_hybrid, the standard
    path's settling rule), so a shard scores as a single-device run does.

    Args:
        device_id: Index of the device among the parent's devices (0, 1, ...)
        shard_input: Pre-computed W1, W2, h->h data for evaluation

    Returns:
        EvalShardOutput with fitnesses and timing info; on any error, `error` holds it and
        the caller raises (combine_shard_fitnesses)
    """
    total_start = time.perf_counter()

    try:
        import jax
        import jax.numpy as jnp

        init_time = (time.perf_counter() - total_start) * 1000

        gpus = [d for d in jax.devices() if d.platform == 'gpu']
        if len(gpus) > 1:
            raise RuntimeError(
                f"the worker for device {device_id} sees {len(gpus)} GPUs: its "
                f"CUDA_VISIBLE_DEVICES ({os.environ.get('CUDA_VISIBLE_DEVICES')!r}) was not set "
                f"before JAX started. Start workers with start_device_workers; a main script "
                f"that sets CUDA_VISIBLE_DEVICES at import overrides the per-worker value")
        device = jax.devices()[0]

        with jax.default_device(device):
            transfer_start = time.perf_counter()

            # Convert numpy arrays to JAX arrays on device
            W1 = jax.device_put(jnp.array(shard_input.W1), device)
            W2 = jax.device_put(jnp.array(shard_input.W2), device)
            inputs = jax.device_put(jnp.array(shard_input.inputs), device)
            targets = jax.device_put(jnp.array(shard_input.targets), device)

            # Convert h->h connections if present
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
                # The kernel EVAL_ONLY_PARALLEL uses: settle_hidden with tanh (h(0) = 0, then
                # tanh(inputs @ W1 + recurrent(h)) for activate_time steps) and a sigmoid output,
                # summed squared error per genome. In 0.1.0 this branch carried its own loop,
                # whose h->h sum indexed the sample axis, so every shard failed.
                from emr_hyperneat.emrhyperneat import _pmap_eval_batch_hybrid
                n_errors = inputs.shape[0] * targets.shape[1]

                eval_vmapped = jax.jit(
                    lambda w1, w2, fr, to, wt, vm: 1.0 - _pmap_eval_batch_hybrid(
                        w1, w2, fr, to, wt, vm, inputs, targets, activate_time, w1.shape[-1]
                    ) / n_errors,
                    device=device,
                )

                fitnesses = eval_vmapped(W1, W2, hh_from, hh_to, hh_weights, hh_valid)

            else:
                # Dense evaluation (no h->h)
                def eval_single_dense(w1, w2):
                    """Evaluate single network without h->h connections."""
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


def start_device_workers(num_devices: int, mp_context: Any) -> List[Any]:
    """One single-worker process pool per device, each started with its device's GPU.

    The worker process inherits the parent's environment when it starts, so the parent sets
    CUDA_VISIBLE_DEVICES to the device (the d-th entry of its own CUDA_VISIBLE_DEVICES when that
    is set, as under Slurm, else d) and XLA_PYTHON_CLIENT_PREALLOCATE=false (unless one of the
    XLA_PYTHON_CLIENT_* memory variables is set, which the child then inherits), forces the start
    with a trivial task, and restores its own environment. A pool per device pins each shard to
    its device: one pool with several workers serves tasks in any order. On CPU devices the
    variable changes nothing.

    The parent's matmul precision and x64 mode go to the workers the same way, as
    JAX_DEFAULT_MATMUL_PRECISION and JAX_ENABLE_X64: a spawned worker reads only its environment
    when its JAX starts, so a setting the parent made in process (jax.config.update, as a conftest
    or a script does) would be lost. Without them a worker runs at JAX's defaults, as 0.1.0's
    workers always did: TF32 matmuls on an Ampere or newer GPU (1.0e-4 from SINGLE_GPU on two L40S,
    1.2e-7 on V100s), and float32 under an in-process x64.

    Args:
        num_devices: Number of pools, one per device
        mp_context: A multiprocessing context (spawn: the package holds JAX state that fork
            would copy)

    Returns:
        The pools, in device order. The caller shuts them down.
    """
    from concurrent.futures import ProcessPoolExecutor

    import jax

    precision = jax.config.jax_default_matmul_precision
    numeric = {'JAX_DEFAULT_MATMUL_PRECISION': None if precision is None else str(precision),
               'JAX_ENABLE_X64': '1' if jax.config.jax_enable_x64 else '0'}
    parent = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    parent_ids = [part.strip() for part in parent.split(',') if part.strip()]
    memory_settings = ('XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                       'XLA_PYTHON_CLIENT_ALLOCATOR')
    user_memory = any(name in os.environ for name in memory_settings)
    saved = {name: os.environ.get(name)
             for name in ('CUDA_VISIBLE_DEVICES', 'XLA_PYTHON_CLIENT_PREALLOCATE', *numeric)}
    pools: List[Any] = []
    try:
        for name, value in numeric.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for d in range(num_devices):
            os.environ['CUDA_VISIBLE_DEVICES'] = parent_ids[d] if d < len(parent_ids) else str(d)
            if not user_memory:
                os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            pool = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
            pools.append(pool)
            pool.submit(os.getpid).result()  # the worker process starts here, with this environment
    except BaseException:
        for pool in pools:
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return pools


def combine_shard_fitnesses(results: List[EvalShardOutput], pop_size: int) -> np.ndarray:
    """The population's fitness from the shards' results, in population order.

    Raises:
        RuntimeError: a shard failed. In 0.1.0 its genomes kept a fitness of 0 and evolution
            went on, so a failure (an h->h shape error, a worker out of GPU memory) only showed
            as a run that never improved.
    """
    fitnesses = np.zeros(pop_size, dtype=np.float32)
    for result in results:
        if result.error:
            raise RuntimeError(f"POPULATION_PARALLEL_PROCESS: the worker for genomes "
                               f"{result.shard_start}-{result.shard_end} failed: {result.error}")
        fitnesses[result.shard_start:result.shard_end] = result.fitnesses
    return fitnesses
