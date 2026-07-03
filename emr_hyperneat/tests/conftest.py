"""Shared fixtures and utilities for EMR-HyperNEAT unit tests.

This module provides:
- Problem fixtures (XOR, AND, OR, NAND, NOR, Parity)
- Configuration factories for all feature combinations
- Evolution runner utilities
- Assertion helpers
- Test result dataclass
"""

import sys
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import pytest
import numpy as np
import jax
import jax.numpy as jnp

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent / 'src'))

# Import the main implementation
from emr_hyperneat.emrhyperneat import (
    EMRHyperNEAT,
    EMRConfig,
    NeuromodulationConfig,
    MultiTaskConfig,
    SparseHiddenConnections,
    ForwardPassMode,
    HHCacheManager,
    NT_TASK_PRESETS,  # For multi-task NT vectors
)


# Import multi-GPU components
from emr_hyperneat.emrhyperneat_base import (
    MultiGPUStrategy,
    PositionShardingConfig,
    IslandModelConfig,
    HybridShardingConfig,
    PopulationPmapConfig,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_SEED = 42
STANDARD_SEEDS = [42, 123, 456]
FITNESS_THRESHOLD = 0.95
QUICK_GENERATIONS = 5
STANDARD_GENERATIONS = 15
COMPREHENSIVE_GENERATIONS = 50
DEFAULT_POPULATION = 100
DEFAULT_MAX_DEPTH = 2

# Multi-task inputs and targets (for MultiTaskProblem class)
LOGIC_GATE_INPUTS = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
LOGIC_GATE_TARGETS = {
    'xor': jnp.array([[0.0], [1.0], [1.0], [0.0]]),
    'and': jnp.array([[0.0], [0.0], [0.0], [1.0]]),
    'or': jnp.array([[0.0], [1.0], [1.0], [1.0]]),
    'nand': jnp.array([[1.0], [1.0], [1.0], [0.0]]),
    'nor': jnp.array([[1.0], [0.0], [0.0], [0.0]]),
}


# =============================================================================
# Problem Classes
# =============================================================================

class XORProblem:
    """XOR logic gate problem."""
    input_shape = (3,)  # 2 inputs + bias
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = FITNESS_THRESHOLD

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[0.0], [1.0], [1.0], [0.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


class ANDProblem:
    """AND logic gate problem."""
    input_shape = (3,)
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = FITNESS_THRESHOLD

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[0.0], [0.0], [0.0], [1.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


class ORProblem:
    """OR logic gate problem."""
    input_shape = (3,)
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = FITNESS_THRESHOLD

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[0.0], [1.0], [1.0], [1.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


class NANDProblem:
    """NAND logic gate problem."""
    input_shape = (3,)
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = FITNESS_THRESHOLD

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[1.0], [1.0], [1.0], [0.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


class NORProblem:
    """NOR logic gate problem."""
    input_shape = (3,)
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = FITNESS_THRESHOLD

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[1.0], [0.0], [0.0], [0.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


class ParityProblem:
    """N-bit parity problem."""
    jitable = True
    use_bias = True

    def __init__(self, n_bits: int = 3):
        self.n_bits = n_bits
        self.input_shape = (n_bits + 1,)  # n inputs + bias
        self.output_shape = (1,)
        self.fitness_threshold = FITNESS_THRESHOLD
        self._generate_data()

    def _generate_data(self):
        self.inputs = []
        self.targets = []
        for i in range(2 ** self.n_bits):
            bits = [(i >> j) & 1 for j in range(self.n_bits)]
            self.inputs.append([float(b) for b in bits] + [1.0])  # Add bias
            self.targets.append([float(sum(bits) % 2)])

    def get_data(self):
        return list(zip(self.inputs, self.targets))


class MultiTaskProblem:
    """Problem wrapper for multi-task evaluation.

    Unlike XORProblem etc., this uses 2D input (no bias) to match the benchmark
    at experiments/neuromodulation/true_zero_forgetting_recurrence_benchmark.py.
    """
    input_shape = (2,)
    output_shape = (1,)
    jitable = True
    use_bias = False

    def __init__(self, task_name: str):
        if task_name not in LOGIC_GATE_TARGETS:
            raise ValueError(f"Unknown task: {task_name}. Must be one of {list(LOGIC_GATE_TARGETS.keys())}")
        self.name = task_name
        self.inputs = LOGIC_GATE_INPUTS
        self.targets = LOGIC_GATE_TARGETS[task_name]
        self.fitness_threshold = FITNESS_THRESHOLD

    def get_data(self):
        return list(zip(
            [np.array(x) for x in self.inputs.tolist()],
            [np.array(y) for y in self.targets.tolist()]
        ))


# =============================================================================
# Test Result Dataclass
# =============================================================================

@dataclass
class TestResult:
    """Complete test result with all relevant metrics."""
    # Test metadata
    test_name: str = ""
    config_name: str = ""
    seed: int = DEFAULT_SEED

    # Fitness metrics
    fitness_history: List[float] = field(default_factory=list)
    final_fitness: float = 0.0
    best_fitness: float = 0.0
    solved: bool = False
    generations_to_solve: Optional[int] = None
    fitness_threshold: float = FITNESS_THRESHOLD

    # Timing metrics
    gen_times: List[float] = field(default_factory=list)
    total_time: float = 0.0
    avg_gen_time_ms: float = 0.0

    # Topology metrics
    num_hidden_positions: int = 0
    num_hh_connections: int = 0

    # Error tracking
    error: Optional[str] = None


# =============================================================================
# Configuration Factories
# =============================================================================

def create_base_config(
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
    variance_threshold: float = 0.03,
    division_threshold: float = 0.03,
    band_threshold: float = 0.3,
    max_weight: float = 3.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Create base algorithm configuration."""
    return {
        'algorithm_params': {
            'emrhyperneat': {
                'population_size': population_size,
                'emr_hyperneat': {
                    'initial_depth': 1,
                    'max_depth': max_depth,
                    'variance_threshold': variance_threshold,
                    'division_threshold': division_threshold,
                    'band_threshold': band_threshold,
                    'max_weight': max_weight,
                    'verbose': verbose,
                    'recurrence': {},
                    'dynamic_functions': {},
                    'aggregation': {},
                    'neuromodulation': {},
                },
                'substrate': {
                    'input_coords': [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],
                    'output_coords': [(0.0, 1.0)],
                    'output_activation': 'sigmoid',
                    'hidden_activation': 'tanh',
                },
            }
        }
    }


# Recurrence presets
RECURRENCE_PRESETS = {
    'feedforward': {
        'preset': 'feedforward',
        'enabled': False,
        'allow_hidden_to_hidden': False,
        'allow_backward': False,
        'allow_lateral': False,
        'allow_self_loops': False,
        'iteration_level': 0,
    },
    'hidden_only': {
        'preset': 'hidden_only',
        'enabled': True,
        'allow_hidden_to_hidden': True,
        'allow_backward': False,
        'allow_lateral': False,
        'allow_self_loops': False,
        'iteration_level': 2,
    },
    'with_backward': {
        'preset': 'with_backward',
        'enabled': True,
        'allow_hidden_to_hidden': True,
        'allow_backward': True,
        'allow_lateral': False,
        'allow_self_loops': False,
        'iteration_level': 2,
    },
    'with_lateral': {
        'preset': 'with_lateral',
        'enabled': True,
        'allow_hidden_to_hidden': True,
        'allow_backward': False,
        'allow_lateral': True,
        'allow_self_loops': False,
        'iteration_level': 2,
    },
    'with_self': {
        'preset': 'with_self',
        'enabled': True,
        'allow_hidden_to_hidden': True,
        'allow_backward': False,
        'allow_lateral': False,
        'allow_self_loops': True,
        'iteration_level': 2,
    },
    'full_recurrent': {
        'preset': 'full_recurrent',
        'enabled': True,
        'allow_hidden_to_hidden': True,
        'allow_backward': True,
        'allow_lateral': True,
        'allow_self_loops': True,
        'iteration_level': 2,
        'activate_time': 20,
    },
}


# Dynamic function presets
DYNAMIC_FUNCTION_PRESETS = {
    'disabled': {'mode': 'disabled'},
    'global_tanh': {'mode': 'global', 'hidden_activation': 'tanh', 'output_activation': 'sigmoid'},
    'global_sin': {'mode': 'global', 'hidden_activation': 'sin', 'output_activation': 'sigmoid'},
    'global_relu': {'mode': 'global', 'hidden_activation': 'relu', 'output_activation': 'sigmoid'},
    'cppn_output_4': {'mode': 'cppn_output', 'num_activations': 4},
    'cppn_output_6': {'mode': 'cppn_output', 'num_activations': 6},
    'weight_interp_sign': {'mode': 'weight_interpretation', 'num_activations': 4, 'interpretation': 'sign'},
    'weight_interp_magnitude': {'mode': 'weight_interpretation', 'num_activations': 6, 'interpretation': 'magnitude'},
    'weight_interp_variance': {'mode': 'weight_interpretation', 'num_activations': 6, 'interpretation': 'variance'},
    'random_fixed': {'mode': 'random_fixed', 'num_activations': 6},
    'random_generation': {'mode': 'random_generation', 'num_activations': 6},
}


# Aggregation presets
AGGREGATION_PRESETS = {
    'disabled': {'mode': 'disabled'},
    'global_sum': {'mode': 'global', 'global_aggregation': 'sum'},
    'global_mean': {'mode': 'global', 'global_aggregation': 'mean'},
    'global_max': {'mode': 'global', 'global_aggregation': 'max'},
    'global_min': {'mode': 'global', 'global_aggregation': 'min'},
    'global_product': {'mode': 'global', 'global_aggregation': 'product'},
    'weight_interp': {'mode': 'weight_interpretation', 'num_aggregations': 4},
}


# Neuromodulation presets
NEUROMODULATION_PRESETS = {
    'disabled': {'enabled': False, 'mode': 'disabled'},
    'static_gating': {
        'enabled': True,
        'mode': 'static_gating',
        'gate_from_cppn': True,
        'gate_scaling': 'sigmoid',
    },
    'context_gating': {
        'enabled': True,
        'mode': 'context_gating',
        'gate_from_cppn': True,
        'context_dim': 4,
        'context_influence': 0.5,
    },
    'modulatory_neurons': {
        'enabled': True,
        'mode': 'modulatory_neurons',
        'mod_neuron_ratio': 0.1,
        'mod_connection_type': 'multiplicative',
        'mod_decay': 0.9,
    },
    'true_neuromodulation_4nt': {
        'enabled': True,
        'mode': 'true_neuromodulation',
        'num_nt_types': 4,
        'receptor_from_weight': True,
        'receptor_derivation': 'tanh',
        'modulation_strength': 2.0,
        'modulation_mode': 'full',
    },
}


# Geometry seeding presets
GEOMETRY_SEEDING_PRESETS = {
    'disabled': {'enabled': False},
    'default': {'enabled': True, 'seed_weight': -1.0, 'use_7d_inputs': True},
    'positive_weight': {'enabled': True, 'seed_weight': 1.0, 'use_7d_inputs': True},
    'zero_weight': {'enabled': True, 'seed_weight': 0.0, 'use_7d_inputs': True},
    '5d_inputs': {'enabled': True, 'seed_weight': -1.0, 'use_7d_inputs': False},
}


def create_config_with_recurrence(
    preset: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
    iteration_level: int = 2,
    hh_cache_enabled: bool = True,
) -> Dict[str, Any]:
    """Create configuration for specific recurrence/connection type."""
    if preset not in RECURRENCE_PRESETS:
        raise ValueError(f"Unknown recurrence preset: {preset}")

    config = create_base_config(max_depth=max_depth, population_size=population_size)
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

    rec_config = RECURRENCE_PRESETS[preset].copy()
    rec_config['iteration_level'] = iteration_level
    rec_config['hh_cache_enabled'] = hh_cache_enabled
    hmr['recurrence'] = rec_config

    hmr['dynamic_functions'] = {'mode': 'disabled'}
    hmr['aggregation'] = {'mode': 'disabled'}
    hmr['neuromodulation'] = {'enabled': False}

    return config


def create_config_with_dynamic_functions(
    mode: str,
    recurrence_preset: str = 'feedforward',
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
    hh_activation_mode: str = 'initial_only',
    **overrides,
) -> Dict[str, Any]:
    """Create configuration for specific dynamic functions mode."""
    if mode not in DYNAMIC_FUNCTION_PRESETS:
        raise ValueError(f"Unknown dynamic functions mode: {mode}")

    config = create_base_config(max_depth=max_depth, population_size=population_size)
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

    if recurrence_preset in RECURRENCE_PRESETS:
        hmr['recurrence'] = RECURRENCE_PRESETS[recurrence_preset].copy()
    else:
        hmr['recurrence'] = {'preset': 'feedforward', 'iteration_level': 0}

    df_config = DYNAMIC_FUNCTION_PRESETS[mode].copy()
    df_config['hh_activation_mode'] = hh_activation_mode
    df_config.update(overrides)
    hmr['dynamic_functions'] = df_config

    if df_config.get('mode') == 'global':
        substrate = config['algorithm_params']['emrhyperneat']['substrate']
        substrate['hidden_activation'] = df_config.get('hidden_activation', 'tanh')
        substrate['output_activation'] = df_config.get('output_activation', 'sigmoid')

    hmr['aggregation'] = {'mode': 'disabled'}
    hmr['neuromodulation'] = {'enabled': False}

    return config


def create_config_with_aggregation(
    mode: str,
    recurrence_preset: str = 'feedforward',
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
    hh_aggregation_mode: str = 'sum',
    **overrides,
) -> Dict[str, Any]:
    """Create configuration for specific aggregation mode."""
    if mode not in AGGREGATION_PRESETS:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    config = create_base_config(max_depth=max_depth, population_size=population_size)
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

    if recurrence_preset in RECURRENCE_PRESETS:
        hmr['recurrence'] = RECURRENCE_PRESETS[recurrence_preset].copy()
    else:
        hmr['recurrence'] = {'preset': 'feedforward', 'iteration_level': 0}

    agg_config = AGGREGATION_PRESETS[mode].copy()
    agg_config['hh_aggregation_mode'] = hh_aggregation_mode
    agg_config.update(overrides)
    hmr['aggregation'] = agg_config

    hmr['dynamic_functions'] = {'mode': 'disabled'}
    hmr['neuromodulation'] = {'enabled': False}

    return config


def create_config_with_neuromodulation(
    mode: str,
    recurrence_preset: str = 'hidden_only',
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
    **overrides,
) -> Dict[str, Any]:
    """Create configuration for specific neuromodulation mode."""
    if mode not in NEUROMODULATION_PRESETS:
        raise ValueError(f"Unknown neuromodulation mode: {mode}")

    config = create_base_config(max_depth=max_depth, population_size=population_size)
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

    if recurrence_preset in RECURRENCE_PRESETS:
        hmr['recurrence'] = RECURRENCE_PRESETS[recurrence_preset].copy()
    else:
        hmr['recurrence'] = {'preset': 'feedforward', 'iteration_level': 0}

    nm_config = NEUROMODULATION_PRESETS[mode].copy()
    nm_config.update(overrides)
    hmr['neuromodulation'] = nm_config

    hmr['dynamic_functions'] = {'mode': 'disabled'}
    hmr['aggregation'] = {'mode': 'disabled'}

    return config


def create_config_with_streaming(
    enable_streaming: bool = True,
    population_chunk_size: int = 50,
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
    recurrence_preset: str = 'feedforward',
) -> Dict[str, Any]:
    """Create configuration with streaming mode enabled."""
    config = create_base_config(max_depth=max_depth, population_size=population_size)
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

    if recurrence_preset in RECURRENCE_PRESETS:
        hmr['recurrence'] = RECURRENCE_PRESETS[recurrence_preset].copy()
    else:
        hmr['recurrence'] = {'preset': 'feedforward', 'iteration_level': 0}

    hmr['enable_streaming'] = enable_streaming
    hmr['population_chunk_size'] = population_chunk_size

    hmr['dynamic_functions'] = {'mode': 'disabled'}
    hmr['aggregation'] = {'mode': 'disabled'}
    hmr['neuromodulation'] = {'enabled': False}

    return config


def create_config_with_multi_task(
    task_names: List[str],
    neuromod_mode: str = 'true_neuromodulation_4nt',
    fitness_aggregation: str = 'min',
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
) -> Dict[str, Any]:
    """Create configuration for multi-task learning.

    Uses 2-input substrate to match MultiTaskProblem (logic gate problems).
    """
    config = create_config_with_neuromodulation(
        mode=neuromod_mode,
        recurrence_preset='hidden_only',
        max_depth=max_depth,
        population_size=population_size,
    )

    # Override substrate to use 2 inputs (matching MultiTaskProblem/logic gates)
    # Based on working benchmark: experiments/neuromodulation/true_zero_forgetting_recurrence_benchmark.py
    config['algorithm_params']['emrhyperneat']['substrate'] = {
        'input_coords': [[-1.0, 0.0], [1.0, 0.0]],  # 2 inputs for logic gates
        'output_coords': [[0.0, 1.0]],
        'output_activation': 'sigmoid',
        'hidden_activation': 'tanh',
    }

    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    hmr['multitask'] = {
        'enabled': True,
        'num_tasks': len(task_names),
        'task_names': task_names,
        'fitness_aggregation': fitness_aggregation,
    }

    return config


def create_config_with_geometry_seeding(
    preset: str = 'default',
    recurrence_preset: str = 'feedforward',
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
    **overrides,
) -> Dict[str, Any]:
    """Create configuration for geometry seeding.

    Geometry seeding adds distance-based locality penalty to weights,
    favoring local connections. Optionally uses 7D CPPN inputs
    (x1, y1, x2, y2, d, delta_x, delta_y) instead of standard 5D.

    Args:
        preset: One of GEOMETRY_SEEDING_PRESETS keys
        recurrence_preset: Recurrence configuration preset
        max_depth: Maximum EMR depth
        population_size: Population size for evolution
        **overrides: Override specific geometry seeding config values

    Returns:
        Configuration dictionary with geometry seeding enabled.
    """
    if preset not in GEOMETRY_SEEDING_PRESETS:
        raise ValueError(f"Unknown geometry seeding preset: {preset}")

    config = create_base_config(max_depth=max_depth, population_size=population_size)
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

    if recurrence_preset in RECURRENCE_PRESETS:
        hmr['recurrence'] = RECURRENCE_PRESETS[recurrence_preset].copy()
    else:
        hmr['recurrence'] = {'preset': 'feedforward', 'iteration_level': 0}

    gs_config = GEOMETRY_SEEDING_PRESETS[preset].copy()
    gs_config.update(overrides)
    hmr['geometry_seeding'] = gs_config

    hmr['dynamic_functions'] = {'mode': 'disabled'}
    hmr['aggregation'] = {'mode': 'disabled'}
    hmr['neuromodulation'] = {'enabled': False}

    return config


def create_config_with_combined_recurrence(
    recurrence_preset: str = 'hidden_only',
    max_depth: int = DEFAULT_MAX_DEPTH,
    population_size: int = DEFAULT_POPULATION,
    num_activations: int = 6,
    num_aggregations: int = 4,
) -> Dict[str, Any]:
    """Create config combining dynamic functions + aggregation (multi-output CPPN) + recurrence.

    This tests the combination where num_cppn_outputs > 1
    and recurrence is enabled, which requires the multi-output query path for H→H discovery.

    Args:
        recurrence_preset: Recurrence configuration preset
        max_depth: Maximum EMR depth
        population_size: Population size for evolution
        num_activations: Number of activation functions in the palette
        num_aggregations: Number of aggregation functions in the palette

    Returns:
        Configuration dictionary with combined activation+aggregation+recurrence.
    """
    config = create_base_config(max_depth=max_depth, population_size=population_size)
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

    if recurrence_preset in RECURRENCE_PRESETS:
        hmr['recurrence'] = RECURRENCE_PRESETS[recurrence_preset].copy()
    else:
        hmr['recurrence'] = {'preset': 'feedforward', 'iteration_level': 0}

    hmr['dynamic_functions'] = {
        'mode': 'cppn_output',
        'num_activations': num_activations,
        'output_activation': 'sigmoid',
    }
    hmr['aggregation'] = {
        'mode': 'cppn_output',
        'num_aggregations': num_aggregations,
        'use_true_aggregation': True,
    }
    hmr['neuromodulation'] = {'enabled': False}

    return config


def run_quick_evolution(
    algorithm: EMRHyperNEAT,
    config: Dict[str, Any],
    problem: Any,
    generations: int = QUICK_GENERATIONS,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
) -> TestResult:
    """Run quick evolution and return results."""
    result = TestResult(
        test_name="quick_evolution",
        seed=seed,
        fitness_threshold=getattr(problem, 'fitness_threshold', FITNESS_THRESHOLD),
    )

    try:
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, problem, seed=seed)
    except Exception as e:
        result.error = f"Initialization failed: {e}"
        return result

    fitness_history = []
    gen_times = []
    solved = False
    generations_to_solve = None

    for gen in range(generations):
        start = time.time()
        try:
            state, metrics = algorithm.run_generation(state, problem, verbose=verbose)
        except Exception as e:
            result.error = f"Generation {gen} failed: {e}"
            result.fitness_history = fitness_history
            result.gen_times = gen_times
            return result

        elapsed = time.time() - start
        best_fit = float(metrics.best_fitness)
        fitness_history.append(best_fit)
        gen_times.append(elapsed * 1000)

        if not solved and best_fit >= result.fitness_threshold:
            solved = True
            generations_to_solve = gen

        if verbose:
            print(f"  Gen {gen}: fitness={best_fit:.6f}, time={elapsed*1000:.1f}ms")

    result.fitness_history = fitness_history
    result.final_fitness = fitness_history[-1] if fitness_history else 0.0
    result.best_fitness = max(fitness_history) if fitness_history else 0.0
    result.solved = solved
    result.generations_to_solve = generations_to_solve
    result.gen_times = gen_times
    result.total_time = sum(gen_times) / 1000
    result.avg_gen_time_ms = float(np.mean(gen_times)) if gen_times else 0.0

    return result


def run_quick_evolution_multi_gpu(
    algorithm: EMRHyperNEAT,
    config: Dict[str, Any],
    problem: Any,
    generations: int = QUICK_GENERATIONS,
    seed: int = DEFAULT_SEED,
    target_fitness: float = 0.95,
) -> TestResult:
    """Run quick evolution using run_until_threshold for proper multi-GPU testing.

    Unlike run_quick_evolution() which uses run_generation() in a loop (single-GPU only),
    this function uses run_until_threshold() which triggers multi-GPU routing.

    Args:
        algorithm: EMR-HyperNEAT algorithm with multi-GPU strategy set
        config: Configuration dict
        problem: Problem instance
        generations: Max generations (used as max_generations)
        seed: Random seed
        target_fitness: Target fitness threshold

    Returns:
        TestResult with evolution metrics
    """
    result = TestResult(
        test_name="quick_evolution_multi_gpu",
        seed=seed,
        fitness_threshold=getattr(problem, 'fitness_threshold', FITNESS_THRESHOLD),
    )

    try:
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, problem, seed=seed)
    except Exception as e:
        result.error = f"Initialization failed: {e}"
        return result

    try:
        start = time.time()
        evolution_result = algorithm.run_until_threshold(
            state, problem,
            target_fitness=target_fitness,
            max_generations=generations,
            collect_history=True
        )
        elapsed = time.time() - start

        result.final_fitness = evolution_result.get('best_fitness', 0.0)
        result.best_fitness = evolution_result.get('best_fitness', 0.0)
        result.fitness_history = evolution_result.get('history', [])
        result.solved = result.best_fitness >= target_fitness
        result.generations_to_solve = evolution_result.get('generations', generations)
        result.total_time = elapsed
        result.avg_gen_time_ms = (elapsed * 1000) / max(1, evolution_result.get('generations', 1))

    except Exception as e:
        result.error = f"Evolution failed: {e}"

    return result


def run_multitask_evolution(
    algorithm: EMRHyperNEAT,
    config: Dict[str, Any],
    problems: List[Any],
    nt_vectors: List[jnp.ndarray],
    generations: int = QUICK_GENERATIONS,
    seed: int = DEFAULT_SEED,
    aggregation_method: str = 'mean',
    verbose: bool = False,
) -> TestResult:
    """Run multi-task evolution with multiple problems and NT vectors.

    Based on: experiments/neuromodulation/true_zero_forgetting_recurrence_benchmark.py

    Args:
        algorithm: The EMR-HyperNEAT algorithm instance.
        config: Configuration dictionary.
        problems: List of problem instances (one per task).
        nt_vectors: List of neurotransmitter vectors (one per task).
        generations: Number of generations to run.
        seed: Random seed for reproducibility.
        aggregation_method: Fitness aggregation method (mean, min, weighted, product, softmin, harmonic).
        verbose: Whether to print progress.

    Returns:
        TestResult with fitness history and metrics.
    """
    result = TestResult(
        test_name="multitask_evolution",
        seed=seed,
        fitness_threshold=getattr(problems[0], 'fitness_threshold', FITNESS_THRESHOLD),
    )

    try:
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, problems[0], seed=seed)
    except Exception as e:
        result.error = f"Initialization failed: {e}"
        return result

    fitness_history = []
    gen_times = []
    solved = False
    generations_to_solve = None

    for gen in range(generations):
        start = time.time()
        try:
            # Call run_generation_multitask directly (like the benchmark does)
            state, metrics = algorithm.run_generation_multitask(
                state,
                problems=problems,
                neurotransmitters=nt_vectors,
                aggregation_method=aggregation_method,
            )
        except Exception as e:
            result.error = f"Generation {gen} failed: {e}"
            result.fitness_history = fitness_history
            result.gen_times = gen_times
            return result

        elapsed = time.time() - start
        best_fit = float(metrics.best_fitness)
        fitness_history.append(best_fit)
        gen_times.append(elapsed * 1000)

        if not solved and best_fit >= result.fitness_threshold:
            solved = True
            generations_to_solve = gen

        if verbose:
            print(f"  Gen {gen}: fitness={best_fit:.6f}, time={elapsed*1000:.1f}ms")

    result.fitness_history = fitness_history
    result.final_fitness = fitness_history[-1] if fitness_history else 0.0
    result.best_fitness = max(fitness_history) if fitness_history else 0.0
    result.solved = solved
    result.generations_to_solve = generations_to_solve
    result.gen_times = gen_times
    result.total_time = sum(gen_times) / 1000
    result.avg_gen_time_ms = float(np.mean(gen_times)) if gen_times else 0.0

    return result


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_no_errors(result: TestResult, message: str = ""):
    """Assert that the result has no errors."""
    if result.error:
        pytest.fail(f"{message}: {result.error}" if message else result.error)


def assert_fitness_above(result: TestResult, threshold: float, message: str = ""):
    """Assert that best fitness is above threshold."""
    if result.best_fitness < threshold:
        msg = f"Best fitness {result.best_fitness:.4f} < threshold {threshold:.4f}"
        pytest.fail(f"{message}: {msg}" if message else msg)


def assert_solved(result: TestResult, message: str = ""):
    """Assert that the problem was solved."""
    if not result.solved:
        msg = f"Problem not solved - best fitness {result.best_fitness:.4f} < threshold {result.fitness_threshold:.4f}"
        pytest.fail(f"{message}: {msg}" if message else msg)


def assert_positive_fitness(result: TestResult, message: str = ""):
    """Assert that fitness is positive (evolution is working)."""
    if result.best_fitness <= 0:
        msg = f"Non-positive fitness: {result.best_fitness:.4f}"
        pytest.fail(f"{message}: {msg}" if message else msg)


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def xor_problem() -> XORProblem:
    """Provide XOR problem instance."""
    return XORProblem()


@pytest.fixture
def and_problem() -> ANDProblem:
    """Provide AND problem instance."""
    return ANDProblem()


@pytest.fixture
def or_problem() -> ORProblem:
    """Provide OR problem instance."""
    return ORProblem()


@pytest.fixture
def nand_problem() -> NANDProblem:
    """Provide NAND problem instance."""
    return NANDProblem()


@pytest.fixture
def nor_problem() -> NORProblem:
    """Provide NOR problem instance."""
    return NORProblem()


@pytest.fixture
def parity3_problem() -> ParityProblem:
    """Provide 3-bit parity problem instance."""
    return ParityProblem(n_bits=3)


@pytest.fixture
def parity4_problem() -> ParityProblem:
    """Provide 4-bit parity problem instance."""
    return ParityProblem(n_bits=4)


@pytest.fixture
def algorithm() -> EMRHyperNEAT:
    """Provide algorithm instance."""
    return EMRHyperNEAT()


@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Provide base configuration."""
    return create_base_config()


@pytest.fixture
def feedforward_config() -> Dict[str, Any]:
    """Provide feedforward-only configuration."""
    return create_config_with_recurrence('feedforward')


@pytest.fixture
def hidden_only_config() -> Dict[str, Any]:
    """Provide hidden-only recurrence configuration."""
    return create_config_with_recurrence('hidden_only')


@pytest.fixture
def full_recurrent_config() -> Dict[str, Any]:
    """Provide full recurrent configuration."""
    return create_config_with_recurrence('full_recurrent')


@pytest.fixture
def two_task_problems() -> Dict[str, Any]:
    """Provide 2-task problem bundle (xor, and) with NT vectors."""
    task_names = ['xor', 'and']
    return {
        'problems': [MultiTaskProblem(name) for name in task_names],
        'task_names': task_names,
        'nt_vectors': [jnp.array(NT_TASK_PRESETS[name]) for name in task_names],
    }


@pytest.fixture
def three_task_problems() -> Dict[str, Any]:
    """Provide 3-task problem bundle (xor, and, or) with NT vectors."""
    task_names = ['xor', 'and', 'or']
    return {
        'problems': [MultiTaskProblem(name) for name in task_names],
        'task_names': task_names,
        'nt_vectors': [jnp.array(NT_TASK_PRESETS[name]) for name in task_names],
    }


@pytest.fixture
def five_task_problems() -> Dict[str, Any]:
    """Provide 5-task problem bundle with NT vectors."""
    task_names = ['xor', 'and', 'or', 'nand', 'nor']
    return {
        'problems': [MultiTaskProblem(name) for name in task_names],
        'task_names': task_names,
        'nt_vectors': [jnp.array(NT_TASK_PRESETS[name]) for name in task_names],
    }


# =============================================================================
# Skip Markers
# =============================================================================

def requires_multi_gpu():
    """Skip test if multiple GPUs not available."""
    num_devices = len(jax.devices())
    return pytest.mark.skipif(
        num_devices < 2,
        reason=f"Requires 2+ GPUs, only {num_devices} available"
    )


def requires_cuda():
    """Skip test if CUDA not available."""
    backend = jax.default_backend()
    return pytest.mark.skipif(
        backend != 'gpu',
        reason=f"Requires CUDA GPU, backend is {backend}"
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    'DEFAULT_SEED',
    'STANDARD_SEEDS',
    'FITNESS_THRESHOLD',
    'QUICK_GENERATIONS',
    'STANDARD_GENERATIONS',
    'COMPREHENSIVE_GENERATIONS',
    'DEFAULT_POPULATION',
    'DEFAULT_MAX_DEPTH',

    # Problem classes
    'XORProblem',
    'ANDProblem',
    'ORProblem',
    'NANDProblem',
    'NORProblem',
    'ParityProblem',
    'MultiTaskProblem',

    # Multi-task constants
    'LOGIC_GATE_INPUTS',
    'LOGIC_GATE_TARGETS',
    'NT_TASK_PRESETS',

    # Test result
    'TestResult',

    # Presets
    'RECURRENCE_PRESETS',
    'DYNAMIC_FUNCTION_PRESETS',
    'AGGREGATION_PRESETS',
    'NEUROMODULATION_PRESETS',
    'GEOMETRY_SEEDING_PRESETS',

    # Config factories
    'create_base_config',
    'create_config_with_recurrence',
    'create_config_with_dynamic_functions',
    'create_config_with_aggregation',
    'create_config_with_neuromodulation',
    'create_config_with_streaming',
    'create_config_with_multi_task',
    'create_config_with_geometry_seeding',

    # Evolution runner
    'run_quick_evolution',
    'run_quick_evolution_multi_gpu',
    'run_multitask_evolution',

    # Assertions
    'assert_no_errors',
    'assert_fitness_above',
    'assert_solved',
    'assert_positive_fitness',

    # Skip markers
    'requires_multi_gpu',
    'requires_cuda',

    # Implementation imports
    'EMRHyperNEAT',
    'EMRConfig',
    'NeuromodulationConfig',
    'MultiTaskConfig',
    'SparseHiddenConnections',
    'ForwardPassMode',
    'HHCacheManager',
    'MultiGPUStrategy',
    'PositionShardingConfig',
    'IslandModelConfig',
    'HybridShardingConfig',
    'PopulationPmapConfig',
]
