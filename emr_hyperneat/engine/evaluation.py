"""Evaluating a population: the single-task fitness, the feed-forward and recurrent forwards, the chunked
and scanned evaluation, and the records a generation's evaluation returns."""
import functools
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from emr_hyperneat.engine.settling import _stable_settle_step, settle_hidden, settle_hidden_hmr


def _eval_single_fitness(outputs, targets_batch, mode):
    """Compute fitness for a single individual in single-task evaluation.

    Supports multiple fitness modes for classification tasks.
    Called inside vmap'd evaluation functions, mode is a Python string
    resolved at JAX trace time (only one branch is compiled).
    """
    if mode == 'accuracy':
        n_out = outputs.shape[-1]
        if n_out == 1:
            # Binary: threshold at 0.5
            predictions = (outputs > 0.5).astype(jnp.float32)
            return jnp.mean(predictions == targets_batch)
        else:
            predictions = jnp.argmax(outputs, axis=-1)
            labels = jnp.argmax(targets_batch, axis=-1)
            return jnp.mean(predictions == labels)
    elif mode == 'hybrid':
        n_out = outputs.shape[-1]
        if n_out == 1:
            predictions = (outputs > 0.5).astype(jnp.float32)
            accuracy = jnp.mean(predictions == targets_batch)
        else:
            predictions = jnp.argmax(outputs, axis=-1)
            labels = jnp.argmax(targets_batch, axis=-1)
            accuracy = jnp.mean(predictions == labels)
        mse = jnp.mean((outputs - targets_batch) ** 2)
        return 0.8 * accuracy + 0.2 * (1.0 - mse)
    elif mode == 'bce':
        eps = 1e-7
        outputs_safe = jnp.clip(outputs, eps, 1.0 - eps)
        bce = -jnp.mean(
            targets_batch * jnp.log(outputs_safe) +
            (1.0 - targets_batch) * jnp.log(1.0 - outputs_safe)
        )
        return 1.0 / (1.0 + bce)
    elif mode == 'soft_accuracy':
        n_out = outputs.shape[-1]
        if n_out == 1:
            # Binary: sigmoid-based soft accuracy
            T = 10.0
            soft_pred = jax.nn.sigmoid((outputs - 0.5) * T)
            return jnp.mean(soft_pred * targets_batch + (1.0 - soft_pred) * (1.0 - targets_batch))
        else:
            # Multi-class: softmax-based soft accuracy
            outputs_sharpened = jax.nn.softmax(outputs * 10.0, axis=-1)
            return jnp.mean(jnp.sum(outputs_sharpened * targets_batch, axis=-1))
    elif mode == 'ce':
        # Cross-entropy for multi-class classification.
        # Assumes outputs are softmax-activated (probabilities summing to 1)
        # and targets are one-hot encoded. Matches DES-HyperNEAT: fitness = exp(-CE)
        eps = 1e-7
        outputs_safe = jnp.clip(outputs, eps, 1.0 - eps)
        ce = -jnp.mean(jnp.sum(targets_batch * jnp.log(outputs_safe), axis=-1))
        return jnp.exp(-ce)
    elif mode == 'acc_mse':
        # Accuracy with 0.01 of (1 - MSE) as a tie-break, as the multi-task evaluators and HMR's
        # neuromodulation variant define it
        return (_eval_single_fitness(outputs, targets_batch, 'accuracy')
                + 0.01 * _eval_single_fitness(outputs, targets_batch, 'mse'))
    elif mode == 'accuracy_confident':
        # Accuracy plus 0.1 of the outputs' mean distance from 0.5, scaled to [0, 1]
        return (_eval_single_fitness(outputs, targets_batch, 'accuracy')
                + 0.1 * jnp.mean(jnp.abs(outputs - 0.5) * 2.0))
    else:  # 'mse' (FITNESS_MODES is checked at configuration)
        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))


def evaluate_population_on_device(
    forward_one: Callable,
    genome_args: Tuple,
    chunk_by_scan: bool,
    eval_chunk_size: Optional[int],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    fitness_mode: str,
    whole_chunks: bool = False,
) -> jnp.ndarray:
    """A population's fitness under a generation's forward, on the device that holds its arrays.

    One vmap over the population, or chunks of ``eval_chunk_size`` genomes when the population is
    above it, by a scan where the forward asks for one (see `_build_forward`). ``whole_chunks``
    takes the chunked form even when the population is not above the chunk: a split's device
    evaluating some of one device's chunks evaluates them as one device does (a one-chunk scan
    equals that chunk inside a longer scan; the vmap does not, by an ulp).
    """
    def eval_one(*args):
        outputs = forward_one(inputs, *args)
        return _eval_single_fitness(outputs, targets, fitness_mode)

    pop_size = genome_args[0].shape[0]
    if eval_chunk_size and eval_chunk_size > 0 and (whole_chunks or pop_size > eval_chunk_size):
        if chunk_by_scan:
            # Scan-based: one compiled dispatch, not ceil(pop/chunk) launches: the
            # loop form measured +116-126% at d5/chunk 10. Compiled once per shape: the
            # forward is the same object every generation.
            return chunked_forward_eval(
                forward_one, fitness_mode, inputs, targets, eval_chunk_size, *genome_args
            )
        fitness_chunks = []
        for chunk_start in range(0, pop_size, eval_chunk_size):
            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
            fitness_chunks.append(jax.vmap(eval_one)(*[
                None if arg is None else arg[chunk_start:chunk_end]
                for arg in genome_args
            ]))
        return jnp.concatenate(fitness_chunks)
    return jax.vmap(eval_one)(*genome_args)


class FeedforwardGeneration(NamedTuple):
    """One plain feed-forward generation, as `EMRHyperNEAT._feedforward_generation` computes it."""
    #: The state the population was asked from, after the opt-in pre-ask key split (the state before
    #: tell)
    state: Any
    #: The state after tell
    new_state: Any
    #: (pop_size,) fitness as told: NaN set to -inf
    fitnesses: jnp.ndarray
    #: How many fitness values were NaN (an int32 device scalar)
    nan_count: jnp.ndarray
    cppn_population: Tuple
    cppns_transformed: Tuple
    #: (pop_size, total_positions) active positions
    masks: jnp.ndarray
    W1: jnp.ndarray
    W2: jnp.ndarray
    #: The per-genome forward and the population arrays it evaluated (see _build_forward)
    forward_one: Callable
    genome_args: Tuple


class EvaluatedGeneration(NamedTuple):
    """What the host side of a generation reads once the population is told: the champion record,
    the counters and the metrics (`EMRHyperNEAT._finish_generation`)."""
    #: The state the population was evaluated under (before tell), and after tell
    state: Any
    new_state: Any
    fitnesses: jnp.ndarray
    nan_count: int
    cppn_population: Tuple
    forward_one: Callable
    genome_args: Tuple
    #: The ceiling-sized mask, and the mask on the evaluation axis (compacted on the recurrent path)
    masks_full: jnp.ndarray
    masks_eval: jnp.ndarray
    #: The ceiling position of each evaluated position, or None (the identity)
    eval_positions: Optional[jnp.ndarray]
    act_indices: Optional[jnp.ndarray]
    ih_agg_indices: Optional[jnp.ndarray]
    #: The weights as evaluated
    W1: jnp.ndarray
    W2: jnp.ndarray
    sparse_hh: Optional[Any]
    cache_hit: bool
    #: Virtual positions padding the compact axis, the ceiling's positions, the real evaluated
    #: positions
    compact_pad: int
    ceiling_positions: int
    total_positions: int
    num_levels: int
    pop_size: int


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

    The plain configuration of :func:`settle_hidden`: sum input drive, tanh hidden nodes,
    sigmoid outputs.

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
    hidden = settle_hidden(inputs @ W1, from_indices, to_indices, hh_weights, valid_mask,
                           activate_time, jnp.tanh)
    return jax.nn.sigmoid(hidden @ W2)


@functools.lru_cache(maxsize=None)
def _plain_recurrent_forward(activate_time: int, settling_rule: str = 'emr') -> Callable:
    """The plain recurrent path's per-genome forward, made once per step count and settling rule:
    the same object every generation, so :func:`chunked_forward_eval` compiles its evaluation once
    per shape.

    'emr': :func:`forward_hybrid_vmapped`. 'hmr': HMR's plain recurrent forward, tanh(x W1), tanh
    settling steps (:func:`settle_hidden_hmr`), sigmoid output.
    """
    if settling_rule == 'hmr':
        def forward_one(inputs, W1, W2, from_indices, to_indices, hh_weights, valid_mask):
            h = settle_hidden_hmr(
                inputs @ W1,
                from_indices,
                to_indices,
                hh_weights,
                valid_mask,
                activate_time,
                jnp.tanh,
                jnp.tanh,
            )
            return jax.nn.sigmoid(h @ W2)
        return forward_one

    def forward_one(inputs, W1, W2, from_indices, to_indices, hh_weights, valid_mask):
        return forward_hybrid_vmapped(inputs, W1, W2, from_indices, to_indices, hh_weights,
                                      valid_mask, activate_time)
    return forward_one


@functools.lru_cache(maxsize=None)
def _node_list_forward(activate_time: int) -> Callable:
    """The node-list model's per-genome forward, made once per step count, so the evaluation
    compiles once per shape.

    HMR's node list updated every node at once from the last step's values, inputs clamped, hidden
    nodes tanh and outputs linear, and took the sigmoid of the outputs after ``activate_time``
    steps. The outputs therefore read the hidden nodes one step behind:
    ``h_t = tanh(x W_in + h_(t-1) W_hh)`` for ``t = 1 .. T-1``, then ``sigmoid(h_(T-1) W_out)``.
    The hidden steps are the engine's settling step for tanh nodes (:func:`_stable_settle_step`),
    so the scan is the plain recurrent path's. HMR summed each node's inputs in its connection
    list's order, Python set order; the sums here run in a fixed order, so the outputs equal HMR's
    up to float32 rounding.

    Args:
        activate_time: Steps T, at least 2 (the configuration checks it).

    Returns:
        ``forward_one(inputs, W_in, W_hh, W_out)``: (n_samples, n_outputs) for one genome's blocks.
    """
    step = _stable_settle_step(jnp.tanh, False, False, 4)

    def forward_one(inputs, W_in, W_hh, W_out):
        drive = inputs @ W_in
        (hidden, *_), _ = lax.scan(step, (jnp.zeros_like(drive), drive, W_hh), None,
                                   length=activate_time - 1)
        return jax.nn.sigmoid(hidden @ W_out)
    return forward_one


def _auto_eval_chunk_size(eval_length: int, pop_size: int) -> Optional[int]:
    """The evaluation's genome chunk under ``eval_chunk_size: -1``, from the length the evaluation compiles for.

    The length includes compaction padding: the chunking is part of the compiled shape, so counts that share a padded
    length must share it. Keyed to the real count, the 1024 bucket (961 to 1,024 real positions) would switch to
    chunks at 1,000 and compile the chunked path at a length already seen.
    """
    if eval_length < 1000:        # depth <= 3
        return None
    if eval_length < 6000:        # depth 4-5
        return min(500, pop_size)
    if eval_length < 50000:       # depth 6 "anomaly zone"
        return min(50, pop_size)
    if eval_length < 200000:      # depth 7
        return min(200, pop_size)
    return min(50, pop_size)      # depth 8+


def chunked_vmap_eval(eval_fn, chunk: int, *arrays: jnp.ndarray) -> jnp.ndarray:
    """Population-chunked evaluation as one compiled dispatch.

    The Python-loop form issues ceil(pop/chunk) sequential device launches and
    is launch-bound: a depth-5 probe measured +116-126% per generation
    at chunk 10 (100 launches). `lax.map` compiles the chunk loop into the
    computation, with the same peak memory (one chunk's lanes live at a time) and one
    launch. The population is padded to a chunk multiple; padding lanes are
    evaluated and dropped, so results equal the loop form: bitwise on XLA:CPU,
    to float32 rounding on a GPU, where the padded last chunk is lowered at the
    full chunk shape and the loop form's last call at the remainder's shape.

    Args:
        eval_fn: Per-genome evaluation function.
        chunk: Genomes per chunk (> 0).
        *arrays: Per-genome arrays, each with the population on axis 0.

    Returns:
        (pop,) fitness array, equal to vmapping eval_fn per chunk (see above).
    """
    stacked, pop, padded = _stack_chunks(arrays, chunk)
    mapped = jax.lax.map(lambda xs: jax.vmap(eval_fn)(*xs), stacked)
    return mapped.reshape(padded)[:pop]


def _stack_chunks(arrays: Tuple[jnp.ndarray, ...],
                  chunk: int) -> Tuple[Tuple[jnp.ndarray, ...], int, int]:
    """The per-genome arrays padded to a chunk multiple and stacked as (chunks, chunk, ...).

    Returns:
        ``(stacked, pop, padded)``: the stacked arrays, the population and its padded size.
    """
    pop = arrays[0].shape[0]
    num_chunks = -(-pop // chunk)
    padded = num_chunks * chunk

    def stack(a: jnp.ndarray) -> jnp.ndarray:
        if padded != pop:
            pad_width = ((0, padded - pop),) + ((0, 0),) * (a.ndim - 1)
            a = jnp.pad(a, pad_width)
        return a.reshape((num_chunks, chunk) + a.shape[1:])

    return tuple(stack(a) for a in arrays), pop, padded


@functools.lru_cache(maxsize=64)
def _chunked_forward_evaluation(forward_one: Callable, fitness_mode: str) -> Callable:
    """The compiled chunk loop of :func:`chunked_forward_eval`, made once per forward and fitness
    mode; ``jax.jit`` compiles it once per shape."""
    def evaluate(inputs, targets, stacked):
        def eval_one(*args):
            return _eval_single_fitness(forward_one(inputs, *args), targets, fitness_mode)
        return jax.lax.map(lambda xs: jax.vmap(eval_one)(*xs), stacked)
    return jax.jit(evaluate)


def chunked_forward_eval(forward_one: Callable, fitness_mode: str, inputs: jnp.ndarray,
                         targets: jnp.ndarray, chunk: int, *arrays: jnp.ndarray) -> jnp.ndarray:
    """:func:`chunked_vmap_eval` of a forward made once, compiled once per shape.

    ``lax.map`` builds its scan body on every call, so outside ``jit`` the chunk loop of
    :func:`chunked_vmap_eval` is traced and compiled again on every call, once per generation.
    Here the loop runs inside a ``jax.jit`` kept per forward and fitness mode
    (:func:`_chunked_forward_evaluation`), with the inputs and targets as its arguments, so a
    forward that is the same object every generation (:func:`_plain_recurrent_forward`)
    compiles it once per shape. The same computation as :func:`chunked_vmap_eval` with
    ``eval_fn`` the fitness of ``forward_one`` on ``inputs`` and ``targets``.

    Args:
        forward_one: ``forward_one(inputs, *genome_arrays)``, one genome's outputs.
        fitness_mode: The fitness of :func:`_eval_single_fitness`.
        inputs: (n_samples, num_inputs).
        targets: (n_samples, num_outputs).
        chunk: Genomes per chunk (> 0).
        *arrays: Per-genome arrays, each with the population on axis 0.

    Returns:
        (pop,) fitness array.
    """
    stacked, pop, padded = _stack_chunks(arrays, chunk)
    mapped = _chunked_forward_evaluation(forward_one, fitness_mode)(inputs, targets, stacked)
    return mapped.reshape(padded)[:pop]


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
        from_indices, to_indices, hh_weights, valid_mask: sparse h->h
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
