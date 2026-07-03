#!/usr/bin/env python3
"""Multi-Head Neuromodulation with Per-Head Activation Palettes.

This extends multi-head neuromodulation by assigning different activation functions
to different heads based on their task specialization.

Key Insight from Palette Evolution Research:
- XOR is parity-like: needs oscillatory functions (sin, burst)
- AND/OR/NAND/NOR are threshold gates: need bounded functions (tanh, sigmoid)
- Using the wrong activation type makes tasks unsolvable or slow

Per-Head Activation Strategy:
- XOR head: sin (oscillatory - matches parity structure)
- AND head: tanh (bounded - threshold gate)
- OR head: sigmoid (bounded - threshold gate)
- NAND head: tanh (bounded - inverted threshold)
- NOR head: tanh (bounded - inverted threshold)

This should break the 50% floor that multi-head with uniform tanh hits.

Usage:
    python papers/emr-neuromodulation/scripts/runners/multihead_palette_neuromodulation.py
    python papers/emr-neuromodulation/scripts/runners/multihead_palette_neuromodulation.py --benchmark
"""

import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple, Optional, Callable
import numpy as np


import jax
import jax.numpy as jnp
from functools import partial

from emr_hyperneat._hmr_frozen.hmrhyperneat_neuromodulation_functions import (
    HMRHyperNEAT,
    MultiTaskConfig,
    get_nt_for_task,
    NT_PRESETS_4,
)

# R7: optional EMR backend for the HMR->EMR migration A/B test. Try-guarded so
# HMR-only environments still import this module.
try:
    from emr_hyperneat.emrhyperneat import (
        EMRHyperNEAT,
    )
except Exception:
    EMRHyperNEAT = None


# ============================================================================
# Constants
# ============================================================================

ALL_TASKS = ['xor', 'and', 'or', 'nand', 'nor']

INPUTS = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TRUTH_TABLES = {
    'xor': jnp.array([[0.0], [1.0], [1.0], [0.0]]),
    'and': jnp.array([[0.0], [0.0], [0.0], [1.0]]),
    'or': jnp.array([[0.0], [1.0], [1.0], [1.0]]),
    'nand': jnp.array([[1.0], [1.0], [1.0], [0.0]]),
    'nor': jnp.array([[1.0], [0.0], [0.0], [0.0]]),
    'xnor': jnp.array([[1.0], [0.0], [0.0], [1.0]]),
    'imply': jnp.array([[1.0], [1.0], [0.0], [1.0]]),
    'nimply': jnp.array([[0.0], [0.0], [1.0], [0.0]]),
    'converse_imply': jnp.array([[1.0], [0.0], [1.0], [1.0]]),
    'converse_nimply': jnp.array([[0.0], [1.0], [0.0], [0.0]]),
}

# Define head-specific NT profiles using OUTPUT INVERSION approach
# Format: [DA, 5HT, NE, no_invert] where NT4=0.0 enables output inversion
# NOTE: This differs from NT_PRESETS_4 which uses distinct NT vectors for all tasks (NT4=1.0)
HEAD_NT_PROFILES = {
    'xor': jnp.array([0.95, 0.05, 0.95, 1.0]),   # HIGH-LOW-HIGH, no inversion
    'and': jnp.array([0.10, 0.90, 0.10, 1.0]),   # LOW-HIGH-LOW, no inversion
    'or': jnp.array([0.50, 0.50, 0.50, 1.0]),    # BALANCED, no inversion
    'nand': jnp.array([0.10, 0.90, 0.10, 0.0]),  # Same as AND but WITH OUTPUT INVERSION
    'nor': jnp.array([0.50, 0.50, 0.50, 0.0]),   # Same as OR but WITH OUTPUT INVERSION
}

# ============================================================================
# Per-Head Activation Functions - THE KEY INNOVATION
# ============================================================================

# Oscillatory activations (for parity-like problems)
def burst_activation(x):
    """Burst activation - oscillatory, good for parity."""
    return jnp.sin(x) * jnp.exp(-jnp.abs(x) / 4.0)

def dampened_sin(x):
    """Dampened sine - oscillatory with decay."""
    return jnp.sin(x) / (1.0 + jnp.abs(x) * 0.1)

def gcu(x):
    """Growing Cosine Unit - oscillatory but growing."""
    return x * jnp.cos(x)

def sin2x(x):
    """Double-frequency sine."""
    return jnp.sin(2.0 * x)

# Head-specific activation functions
# Based on palette evolution findings:
# - XOR needs oscillatory (sin achieves 100% at 3.3 gens, vs tanh failing)
# - AND/OR/NAND/NOR are threshold gates (tanh/sigmoid work well)
HEAD_ACTIVATIONS = {
    'xor': jnp.sin,           # Oscillatory - matches parity structure
    'and': jnp.tanh,          # Bounded - threshold gate
    'or': jax.nn.sigmoid,     # Bounded - threshold gate
    'nand': jnp.tanh,         # Bounded - inverted threshold
    'nor': jnp.tanh,          # Bounded - inverted threshold
}

# Alternative: All oscillatory for parity heads
HEAD_ACTIVATIONS_AGGRESSIVE = {
    'xor': jnp.sin,           # Pure oscillatory
    'and': jnp.tanh,          # Bounded
    'or': jnp.tanh,           # Bounded
    'nand': jnp.tanh,         # Bounded
    'nor': jnp.tanh,          # Bounded
}

# Preset blend weights
PRESET_BLEND_WEIGHTS = {
    '5head': {
        'xor': jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        'and': jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        'or': jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        'nand': jnp.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        'nor': jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MultiHeadPaletteResult:
    """Result of multi-head palette neuromodulation experiment."""
    experiment_id: str
    task_combination: List[str]
    num_tasks: int
    num_heads: int
    blend_mode: str
    palette_mode: str  # 'per_head' or 'uniform'
    aggregation: str
    seed: int
    converged: bool
    convergence_gen: Optional[int]
    individual_min_fitness: float
    per_task_fitness: Dict[str, float]
    blend_weights: Optional[Dict[str, List[float]]]
    head_activations: Dict[str, str]
    runtime_seconds: float
    total_generations: int
    fitness_history: List[Dict[str, float]] = field(default_factory=list)


# ============================================================================
# Forward Pass with Per-Head Activation
# ============================================================================

def forward_single_head_with_activation(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    receptor_densities: jnp.ndarray,
    base_gains: jnp.ndarray,
    head_nt: jnp.ndarray,
    inputs: jnp.ndarray,
    activation_fn: Callable,
    modulation_strength: float = 5.0,
):
    """Forward pass through a single head with specific NT and activation.

    Args:
        W1, W2: Weight matrices
        receptor_densities: Per-neuron receptors, shape (num_positions, 4)
        base_gains: Per-neuron base gains, shape (num_positions,)
        head_nt: NT vector for this head, shape (4,)
        inputs: Input data
        activation_fn: Activation function for this head (e.g., jnp.sin, jnp.tanh)
        modulation_strength: Modulation scaling

    Returns:
        outputs: Network outputs
    """
    def safe_matmul(a, b):
        result = jnp.matmul(a, b)
        return jnp.where(jnp.isnan(result), 0.0, result)

    # Compute modulation
    modulation = receptor_densities[:, :3] @ head_nt[:3]

    # Effective gains
    effective_gains = base_gains + modulation_strength * modulation
    effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
    effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)

    # Gates
    gates = jax.nn.sigmoid(modulation)

    # Modulation bias
    modulation_bias = modulation * modulation_strength

    # Forward pass with head-specific activation
    pre_hidden = safe_matmul(inputs, W1)
    hidden = activation_fn(effective_gains * pre_hidden + modulation_bias)
    hidden = hidden * gates

    # Output layer
    outputs = jax.nn.sigmoid(safe_matmul(hidden, W2))

    # Output inversion
    invert_signal = head_nt[3]
    invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
    inverted = 1.0 - outputs
    outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

    return outputs


# Create specialized forward functions for each activation type
# This is necessary because JAX JIT requires static activation functions
def _make_forward_fn(activation_fn):
    """Create a forward function with a specific activation."""
    @partial(jax.jit, static_argnums=())
    def _forward(W1, W2, receptor_densities, base_gains, head_nt, inputs, modulation_strength):
        return forward_single_head_with_activation(
            W1, W2, receptor_densities, base_gains, head_nt, inputs,
            activation_fn, modulation_strength
        )
    return _forward

# Pre-compile forward functions for each activation type
FORWARD_FNS = {
    'sin': _make_forward_fn(jnp.sin),
    'tanh': _make_forward_fn(jnp.tanh),
    'sigmoid': _make_forward_fn(jax.nn.sigmoid),
    'relu': _make_forward_fn(jax.nn.relu),
    'burst': _make_forward_fn(burst_activation),
    'cos': _make_forward_fn(jnp.cos),
    'sin2x': _make_forward_fn(sin2x),
    'gcu': _make_forward_fn(gcu),
    'dampened_sin': _make_forward_fn(dampened_sin),
}


def eval_single_network_with_palette(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    receptor_densities: jnp.ndarray,
    base_gains: jnp.ndarray,
    head_nt: jnp.ndarray,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    activation_name: str,
    modulation_strength: float = 5.0,
):
    """Evaluate network with specific activation.

    Args:
        W1, W2, receptor_densities, base_gains: Network parameters
        head_nt: NT vector for this head
        inputs: Input data
        targets: Target outputs
        activation_name: Name of activation ('sin', 'tanh', 'sigmoid', etc.)
        modulation_strength: Modulation scaling

    Returns:
        fitness: Accuracy
    """
    forward_fn = FORWARD_FNS[activation_name]
    outputs = forward_fn(W1, W2, receptor_densities, base_gains, head_nt, inputs, modulation_strength)

    predictions = (outputs > 0.5).astype(jnp.float32)
    correct = jnp.sum(predictions == targets)
    accuracy = correct / (targets.shape[0] * targets.shape[1])

    return accuracy


# Vmapped versions for each activation
def _make_eval_population(activation_name):
    """Create population evaluation function for specific activation."""
    forward_fn = FORWARD_FNS[activation_name]

    def _eval_single(W1, W2, rd, bg, nt, inputs, targets, mod_strength):
        outputs = forward_fn(W1, W2, rd, bg, nt, inputs, mod_strength)
        predictions = (outputs > 0.5).astype(jnp.float32)
        correct = jnp.sum(predictions == targets)
        return correct / (targets.shape[0] * targets.shape[1])

    return jax.vmap(
        _eval_single,
        in_axes=(0, 0, 0, 0, None, None, None, None)
    )

EVAL_POPULATION_FNS = {
    'sin': _make_eval_population('sin'),
    'tanh': _make_eval_population('tanh'),
    'sigmoid': _make_eval_population('sigmoid'),
    'relu': _make_eval_population('relu'),
    'burst': _make_eval_population('burst'),
}


# ============================================================================
# Reference Problem
# ============================================================================

class RefProblem:
    """Reference problem for algorithm initialization."""
    jitable = True
    input_shape = (2,)
    output_shape = (1,)

    def __init__(self, task_names):
        self.task_names = task_names
        self.inputs = INPUTS
        self.targets = TRUTH_TABLES[task_names[0]]
        self.input_coords = [[-1.0, 0.0], [1.0, 0.0]]
        self.output_coords = [[0.0, 1.0]]

    def get_data(self):
        return list(zip(self.inputs.tolist(), self.targets.tolist()))


# ============================================================================
# Multi-Head Palette Experiment Runner
# ============================================================================

def run_multihead_palette_experiment(
    task_names: List[str],
    palette_mode: str = 'per_head',
    blend_mode: str = 'fixed',
    aggregation: str = 'min',
    seed: int = 42,
    generations: int = 200,
    population: int = 750,
    max_depth: int = 4,
    success_threshold: float = 0.90,
    verbose: bool = True,
    nt_profiles: Optional[Dict[str, jnp.ndarray]] = None,
    modulation_strength: float = 5.0,
    backend: str = 'emr',  # R7: migrated to EMR (validated to reproduce HMR/paper). Pass 'hmr' for the exact published baseline.
) -> MultiHeadPaletteResult:
    """Run multi-head neuromodulation with per-head activation palettes.

    Args:
        task_names: List of task names
        palette_mode: 'per_head' (task-specific activations) or 'uniform' (all tanh)
        blend_mode: 'fixed' or 'evolved'
        aggregation: 'min' or 'product'
        seed: Random seed
        generations: Max generations
        population: Population size
        max_depth: HMR max depth
        success_threshold: Threshold for convergence
        verbose: Print progress
        nt_profiles: Optional dict mapping task names to NT vectors.
            Defaults to HEAD_NT_PROFILES (Schema B) if None.
        modulation_strength: Neuromodulation gain scaling factor (default: 5.0).

    Returns:
        MultiHeadPaletteResult
    """
    start_time = time.time()
    num_heads = len(task_names)

    exp_id = f"palette_{palette_mode}__{blend_mode}__{'+'.join(task_names)}__{aggregation}__seed{seed}"

    if verbose:
        print(f"\n{'='*70}")
        print(f"Multi-Head PALETTE Experiment: {exp_id}")
        print(f"Tasks: {task_names}")
        print(f"Palette mode: {palette_mode}")
        print(f"Blend: {blend_mode}, Aggregation: {aggregation}")
        print(f"{'='*70}")

    # Set up head configurations (NT + activation)
    head_names = task_names  # One head per task
    profiles = nt_profiles if nt_profiles is not None else HEAD_NT_PROFILES
    head_nts = jnp.stack([profiles[t] for t in head_names])

    # Activation function names per head
    # Parity-class tasks need oscillatory activation (sin)
    PARITY_TASKS = {'xor', 'xnor'}
    if palette_mode == 'per_head':
        head_activation_names = {
            t: 'sin' if t in PARITY_TASKS else 'tanh'
            for t in head_names
        }
    elif palette_mode == 'uniform_sin':
        # Uniform sin - all tasks use sin activation
        head_activation_names = {t: 'sin' for t in head_names}
    else:
        # Uniform - all tanh (baseline)
        head_activation_names = {t: 'tanh' for t in head_names}

    if verbose:
        print(f"Head activations: {head_activation_names}")

    # Get blend weights
    if blend_mode == 'fixed':
        # Direct mapping: task i -> head i
        blend_weights_dict = {
            t: jnp.eye(num_heads)[i]
            for i, t in enumerate(task_names)
        }
    else:
        rng_key = jax.random.PRNGKey(seed)
        blend_weights_dict = None

    # Initialize algorithm
    problem = RefProblem(task_names)

    if backend == 'emr':
        # R7 migration: EMR's neuromod config differs from HMR (verified via probe).
        # The mode MUST be exactly 'true_neuromodulation' (EMR errors on the HMR mode
        # string and only enables true neuromod on the exact-match mode). EMR's decode
        # sources base_gains/receptors from the first-input weight (use_self_connection_
        # query=False, not config-settable here) -- the validated divergence from HMR's
        # self-connection source. The fitness/forward path below is backend-agnostic.
        if EMRHyperNEAT is None:
            raise RuntimeError("backend='emr' requested but EMRHyperNEAT failed to import")
        config_params = {
            'algorithm_params': {
                'emrhyperneat': {
                    'population_size': population,
                    'substrate': {
                        'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                        'output_coords': [[0.0, 1.0]],
                    },
                    'emr_hyperneat': {
                        'initial_depth': 0,
                        'max_depth': max_depth,
                        'variance_threshold': 0.03,
                        'neuromodulation': {
                            'mode': 'true_neuromodulation',
                            'modulation_strength': modulation_strength,
                            'num_nt_types': 4,
                            'receptor_from_weight': True,
                            'receptor_derivation': 'tanh',
                            # R7 exact reproduction: source receptors/base-gains from the CPPN
                            # self-connection query (the HMR/paper decode), not the W1 approximation.
                            'use_self_connection_query': True,
                        },
                    },
                    'neat_species': {
                        'compatibility_threshold': 2.5,
                        'max_stagnation': 40,
                    },
                },
            },
        }
        algo = EMRHyperNEAT()
    else:
        config_params = {
            'algorithm_params': {
                'hmrhyperneat': {
                    'population_size': population,
                    'substrate': {
                        'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                        'output_coords': [[0.0, 1.0]],
                    },
                    'hmr_hyperneat': {
                        'initial_depth': 0,
                        'max_depth': max_depth,
                        'variance_threshold': 0.03,
                        'neuromodulation': {
                            'mode': 'true_neuromodulation_4nt_option_a_tanh',
                            'modulation_strength': modulation_strength,
                        },
                    },
                    'neat_species': {
                        'compatibility_threshold': 2.5,
                        'max_stagnation': 40,
                    },
                },
            },
        }
        algo = HMRHyperNEAT()
    neat_config = algo.create_config(config_params)
    state = algo.initialize(neat_config, problem, seed=seed)

    # For evolved blend weights
    evolved_blend = None
    if blend_mode == 'evolved':
        rng_key = jax.random.PRNGKey(seed)
        evolved_blend = jax.random.uniform(
            rng_key, (population, num_heads, num_heads),
            minval=0.0, maxval=1.0
        )

    # Evolution loop
    converged = False
    convergence_gen = None
    best_per_task = {}
    fitness_history = []

    for gen in range(generations):
        # Build weight matrices
        first_task_name = task_names[0]

        class _DummyProblem:
            input_shape = (2,)
            output_shape = (1,)
            jitable = True
            def __init__(self_inner):
                self_inner.inputs = INPUTS
                self_inner.targets = TRUTH_TABLES[first_task_name]
                self_inner.input_coords = [[-1.0, 0.0], [1.0, 0.0]]
                self_inner.output_coords = [[0.0, 1.0]]
            def get_data(self_inner):
                return []

        dummy_problem = _DummyProblem()
        _, _ = algo.run_generation_verbose(state, dummy_problem, skip_metrics=True)

        # Get weight matrices
        W1 = algo._cached_W1
        W2 = algo._cached_W2
        rd = algo._neuromod_true['receptor_densities']
        bg = algo._neuromod_true['base_gains']

        # Evaluate on each task using the appropriate head
        per_task_fitness = {}

        for task_idx, task_name in enumerate(task_names):
            inputs = INPUTS
            targets = TRUTH_TABLES[task_name]

            # Get blend weights for this task
            if blend_mode == 'fixed':
                task_blend = blend_weights_dict[task_name]
            else:
                task_blend = evolved_blend[:, task_idx, :]

            # Find the head with highest weight for this task
            if blend_mode == 'fixed':
                selected_head_idx = int(jnp.argmax(task_blend))
            else:
                # Per-individual selection (use mode across population)
                selected_head_idx = task_idx  # Simplified: direct mapping

            # Get the activation for this head
            selected_head_name = head_names[selected_head_idx]
            activation_name = head_activation_names[selected_head_name]

            # Get the NT for this head
            head_nt = head_nts[selected_head_idx]

            # Evaluate population with this head's activation
            eval_fn = EVAL_POPULATION_FNS[activation_name]
            fitness = eval_fn(W1, W2, rd, bg, head_nt, inputs, targets, modulation_strength)

            per_task_fitness[task_name] = fitness

        # Aggregate fitness
        fitness_stack = jnp.stack(list(per_task_fitness.values()), axis=0)

        if aggregation == 'min':
            aggregated_fitness = jnp.min(fitness_stack, axis=0)
        elif aggregation == 'product':
            aggregated_fitness = jnp.prod(fitness_stack, axis=0)
        else:
            aggregated_fitness = jnp.mean(fitness_stack, axis=0)

        # Update state
        new_state = algo._compiled_tell(state, aggregated_fitness)
        state = new_state

        # Find best generalist
        best_idx = int(jnp.nanargmax(aggregated_fitness))
        best_per_task = {
            t: float(per_task_fitness[t][best_idx])
            for t in per_task_fitness.keys()
        }
        min_fitness = min(best_per_task.values())

        # Record history
        fitness_history.append({
            'generation': gen,
            'min_fitness': min_fitness,
            **best_per_task
        })

        # Mutate evolved blend weights if needed
        if blend_mode == 'evolved':
            rng_key, mut_key = jax.random.split(rng_key)
            k1, k2 = jax.random.split(mut_key)
            mask = jax.random.uniform(k1, evolved_blend.shape) < 0.3
            noise = jax.random.normal(k2, evolved_blend.shape) * 0.15
            evolved_blend = jnp.where(mask, evolved_blend + noise, evolved_blend)
            evolved_blend = jnp.clip(evolved_blend, 0.0, 1.0)

        # Check convergence
        if all(f >= success_threshold for f in best_per_task.values()):
            if not converged:
                converged = True
                convergence_gen = gen
                if verbose:
                    print(f"\n*** CONVERGED at generation {gen}! ***")
                    print(f"  Best individual per-task: {best_per_task}")
                break

        if verbose and gen % 10 == 0:
            task_str = ' '.join([f"{t}:{best_per_task[t]:.2f}" for t in task_names])
            print(f"Gen {gen:3d}: min={min_fitness:.4f} | {task_str}")

    runtime = time.time() - start_time

    # Extract final blend weights
    final_blend_weights = None
    if blend_mode == 'fixed':
        final_blend_weights = {
            task: blend_weights_dict[task].tolist()
            for task in task_names
        }
    elif evolved_blend is not None:
        final_blend_weights = {
            task: evolved_blend[best_idx, i].tolist()
            for i, task in enumerate(task_names)
        }

    result = MultiHeadPaletteResult(
        experiment_id=exp_id,
        task_combination=task_names,
        num_tasks=len(task_names),
        num_heads=num_heads,
        blend_mode=blend_mode,
        palette_mode=palette_mode,
        aggregation=aggregation,
        seed=seed,
        converged=converged,
        convergence_gen=convergence_gen,
        individual_min_fitness=min(best_per_task.values()),
        per_task_fitness=best_per_task,
        blend_weights=final_blend_weights,
        head_activations=head_activation_names,
        runtime_seconds=runtime,
        total_generations=generations,
        fitness_history=fitness_history,
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"  Converged: {converged}" + (f" at gen {convergence_gen}" if converged else ""))
        print(f"  Individual min fitness: {result.individual_min_fitness:.4f}")
        print(f"  Per-task: {best_per_task}")
        print(f"  Head activations: {head_activation_names}")
        print(f"  Runtime: {runtime:.1f}s")

    return result


def run_5task_comparison(
    seeds: List[int] = None,
    generations: int = 200,
    population: int = 750,
    verbose: bool = True,
) -> Dict[str, List[MultiHeadPaletteResult]]:
    """Run comparison between uniform tanh and per-head palettes on 5 tasks.

    Args:
        seeds: Random seeds
        generations: Max generations
        population: Population size
        verbose: Print progress

    Returns:
        Dict with 'uniform' and 'per_head' results
    """
    if seeds is None:
        seeds = [42, 123, 456]

    results = {'uniform': [], 'per_head': []}

    for palette_mode in ['uniform', 'per_head']:
        print(f"\n{'#'*70}")
        print(f"# PALETTE MODE: {palette_mode.upper()}")
        print(f"{'#'*70}")

        for seed in seeds:
            result = run_multihead_palette_experiment(
                task_names=ALL_TASKS,
                palette_mode=palette_mode,
                blend_mode='fixed',
                aggregation='min',
                seed=seed,
                generations=generations,
                population=population,
                verbose=verbose,
            )
            results[palette_mode].append(result)

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"5-TASK COMPARISON SUMMARY")
    print(f"{'='*70}")

    for mode in ['uniform', 'per_head']:
        mode_results = results[mode]
        converged = sum(1 for r in mode_results if r.converged)
        avg_min = np.mean([r.individual_min_fitness for r in mode_results])

        print(f"\n{mode.upper()} mode (all {'tanh' if mode == 'uniform' else 'per-head'}):")
        print(f"  Converged: {converged}/{len(mode_results)}")
        print(f"  Average min fitness: {avg_min:.4f}")

        for r in mode_results:
            status = f"gen {r.convergence_gen}" if r.converged else "NOT CONVERGED"
            print(f"    Seed {r.seed}: min={r.individual_min_fitness:.4f} ({status})")

    # Highlight improvement
    uniform_avg = np.mean([r.individual_min_fitness for r in results['uniform']])
    perhead_avg = np.mean([r.individual_min_fitness for r in results['per_head']])

    improvement = (perhead_avg - uniform_avg) / uniform_avg * 100
    print(f"\n{'='*70}")
    print(f"IMPROVEMENT: {improvement:+.1f}% ({uniform_avg:.4f} -> {perhead_avg:.4f})")
    print(f"{'='*70}")

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Head Palette Neuromodulation")
    parser.add_argument('--tasks', type=str, default='xor+and+or+nand+nor',
                        help='Tasks separated by + (default: all 5)')
    parser.add_argument('--palette', type=str, default='per_head',
                        choices=['per_head', 'uniform', 'uniform_sin'],
                        help='Palette mode (default: per_head)')
    parser.add_argument('--blend', type=str, default='fixed',
                        choices=['fixed', 'evolved'],
                        help='Blend weight mode (default: fixed)')
    parser.add_argument('--aggregation', type=str, default='min',
                        choices=['min', 'product', 'mean'],
                        help='Aggregation method (default: min)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--generations', type=int, default=200,
                        help='Max generations (default: 200)')
    parser.add_argument('--population', type=int, default=750,
                        help='Population size (default: 750)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run 5-task benchmark comparing uniform vs per_head')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (50 gens, pop 200)')

    args = parser.parse_args()

    task_names = args.tasks.split('+')

    if args.quick:
        args.generations = 50
        args.population = 200

    if args.benchmark:
        results = run_5task_comparison(
            generations=args.generations,
            population=args.population,
        )

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = Path('results') / f'palette_comparison_5task_{timestamp}.json'
        results_file.parent.mkdir(exist_ok=True)

        # Flatten results for JSON
        all_results = []
        for mode, mode_results in results.items():
            for r in mode_results:
                all_results.append(asdict(r))

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")
    else:
        result = run_multihead_palette_experiment(
            task_names=task_names,
            palette_mode=args.palette,
            blend_mode=args.blend,
            aggregation=args.aggregation,
            seed=args.seed,
            generations=args.generations,
            population=args.population,
        )

        # Save result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = Path('results') / f'palette_{args.palette}_{timestamp}.json'
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        print(f"\nResult saved to: {results_file}")


if __name__ == '__main__':
    main()
