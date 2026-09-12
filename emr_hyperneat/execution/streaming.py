"""Streaming execution: level-by-level matrix building, chunked evaluation.

Most of streaming already lives on the main algorithm class, as the
`_build_matrices_streaming` step selected inside the normal generation loop. That
factoring is the correct one -- it swaps a step rather than forking the loop, so it
composes with `locality_radius`, LEO, geometry seeding, dynamic functions and
neuromodulation, all of which the now-removed streaming subclass silently dropped.

What could not stay on the class is the recurrent evaluator below, because it is
the one piece of the subclass with no counterpart on the parent.

Import direction: this module imports from the algorithm module, never the reverse.
The dispatcher imports it lazily at call time, which is what keeps that one-way.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from emr_hyperneat.emrhyperneat import (  # noqa: E501
    SparseHiddenConnections,
    eval_single_network_hybrid,
)

__all__ = ["evaluate_population_recurrent_double_chunked"]


def evaluate_population_recurrent_double_chunked(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    sparse_hh: SparseHiddenConnections,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    activate_time: int,
    pop_chunk_size: int = 10,
    sample_chunk_size: int = 50,
) -> jnp.ndarray:
    """Evaluate recurrent networks chunked over BOTH population and samples.

    Chunking the sample axis alone is not enough for recurrent evaluation. XLA
    unrolls `lax.scan`, so every one of `activate_time` iterations is held live at
    once, and the population axis is still vmapped whole. The product is what
    OOMs; both axes have to be cut.

    Depth 7, pop 1000, 500 samples, activate_time 20:
        sample-chunked only: 19 x 1000 x 500 x 87380 x 4 = 331 GB (OOM)
        double-chunked:      19 x   10 x  50 x 87380 x 4 = 3.3 GB

    Fitness is accumulated as MSE weighted by actual chunk length rather than as a
    mean of chunk fitnesses, so a trailing short chunk does not get equal weight
    with a full one.

    Args:
        W1: Input-to-hidden weights, shape (pop, num_inputs, positions).
        W2: Hidden-to-output weights, shape (pop, positions, num_outputs).
        sparse_hh: Hidden-to-hidden connections for the population.
        inputs_batch: Inputs, shape (n_samples, num_inputs).
        targets_batch: Targets, shape (n_samples, num_outputs).
        activate_time: Forward-pass iterations per evaluation.
        pop_chunk_size: Genomes per population chunk.
        sample_chunk_size: Samples per sample chunk.

    Returns:
        Fitness per genome, shape (pop,).
    """
    pop_size = W1.shape[0]
    n_samples = inputs_batch.shape[0]

    total_mse = jnp.zeros(pop_size)
    total_samples = 0

    for sample_start in range(0, n_samples, sample_chunk_size):
        sample_end = min(sample_start + sample_chunk_size, n_samples)
        inputs_chunk = inputs_batch[sample_start:sample_end]
        targets_chunk = targets_batch[sample_start:sample_end]
        actual_sample_chunk = sample_end - sample_start

        fitness_chunks = []
        for pop_start in range(0, pop_size, pop_chunk_size):
            pop_end = min(pop_start + pop_chunk_size, pop_size)
            chunk_fitnesses = jax.vmap(
                functools.partial(
                    eval_single_network_hybrid,
                    inputs=inputs_chunk,
                    targets=targets_chunk,
                    activate_time=activate_time,
                )
            )(
                W1[pop_start:pop_end],
                W2[pop_start:pop_end],
                sparse_hh.from_indices[pop_start:pop_end],
                sparse_hh.to_indices[pop_start:pop_end],
                sparse_hh.weights[pop_start:pop_end],
                sparse_hh.valid_mask[pop_start:pop_end],
            )
            fitness_chunks.append(chunk_fitnesses)

        chunk_fitnesses_all = jnp.concatenate(fitness_chunks)
        chunk_mse = 1.0 - chunk_fitnesses_all
        total_mse = total_mse + chunk_mse * actual_sample_chunk
        total_samples += actual_sample_chunk

    return 1.0 - (total_mse / total_samples)
