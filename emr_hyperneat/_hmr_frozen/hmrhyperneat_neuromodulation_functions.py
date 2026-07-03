"""HMR-HyperNEAT with Neuromodulation - Evolved Multi-Behavior Networks.

This is an EXPERIMENTAL extension of HMR-HyperNEAT that implements neuromodulation,
allowing the same network to exhibit multiple behaviors based on context signals.

Neuromodulation enables:
1. Activity gating: Same weights, different effective connectivity per context
2. Context sensitivity: Network behavior changes based on task/input context
3. Modulatory neurons: Neurons that control other neurons, not just propagate signal

Neuromodulation Levels (Progressive)
=====================================
Level 1 - Static Gating (~100 LOC):
  - CPPN outputs per-connection gate values [0, 1]
  - Gates multiply connection weights: effective_weight = weight * gate
  - Simplest form, static once evolved

Level 2 - Context-Dependent Gating (XdG-style, ~200 LOC):
  - Task context vector modulates gates dynamically
  - Same network, different behaviors per context
  - Inspired by XdG achieving 95% across 100 tasks (Masse et al. 2018)

Level 3 - Modulatory Neurons (Soltoggio-style, ~500 LOC):
  - Two neuron types: standard (signal) and modulatory (regulation)
  - Modulatory neurons influence activation/plasticity of targets
  - Full evolved neuromodulatory topology

Configuration Example
====================
```yaml
hmr_hyperneat:
  neuromodulation:
    mode: 'static_gating'  # disabled, static_gating, xdg_style, modulatory, full
    # Level 1: Static Gating
    gate_threshold: 0.5
    gate_scaling: 'sigmoid'  # sigmoid, binary, soft_threshold
    # Level 2: Context Gating
    context_dim: 4
    context_influence: 0.5
    # Level 3: Modulatory Neurons
    mod_neuron_ratio: 0.1
    mod_connection_type: 'multiplicative'  # multiplicative, additive, gated
```

Inherits All Dynamic Functions Modes
=====================================
All modes from hmrhyperneat_dynamic_functions.py are preserved:
- 'disabled': Original hardcoded tanh/sigmoid (baseline)
- 'global': All hidden nodes use same configurable activation
- 'cppn_output': CPPN outputs activation index per node
- 'weight_interpretation': Derive activation from weight patterns
- 'modular': Three orthogonal layers (activation, sparsity, scaling)

Base Algorithm
==============
Based on HMR-HyperNEAT - a GPU-optimized variant of ES-HyperNEAT that achieves
adaptive resolution through pre-computed hierarchical grids and variance-based
masking, enabling efficient batch processing across entire populations.

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
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import jax.numpy as jnp
from jax import lax

from emr_hyperneat._compat.core.base_algorithm import BaseAlgorithm, AlgorithmMetrics
from emr_hyperneat._compat.utils.config_manager import ConfigManager
from emr_hyperneat._compat.adapters.tensorneat_adapter import TensorNEATAdapter


# ============================================================================
# Neuromodulation Configuration
# ============================================================================
# Modular neuromodulation system with three progressive levels that can be
# enabled independently or combined.

@dataclass
class NeuromodulationConfig:
    """Configuration for neuromodulation features.

    Three independent levels can be enabled alone or in combination:
    - Level 1: Static Gating - CPPN outputs per-connection gate values
    - Level 2: Context Gating - Task context modulates gates dynamically
    - Level 3: Modulatory Neurons - Two neuron types (standard/modulatory)

    Attributes:
        enabled: Master switch for neuromodulation features

        # Level 1: Static Gating (CPPN-based)
        static_gating: Enable per-connection gates from CPPN output
        gate_threshold: Threshold for binary/soft_threshold scaling (default 0.5)
        gate_scaling: How to convert raw CPPN output to gate value
            - 'sigmoid': Continuous gates in [0, 1]
            - 'binary': Hard threshold to 0 or 1
            - 'soft_threshold': Soft ramp from threshold

        # Level 2: Context-Dependent Gating (XdG-style)
        context_gating: Enable context-dependent gate modulation
        context_dim: Dimensionality of context vector (default 4)
        context_influence: How much context modulates base gates [0, 1] (default 0.5)
        context_source: Where context comes from
            - 'input': Derived from input statistics
            - 'task_id': One-hot encoded task identifier
            - 'learned': Evolved context per task (future)

        # Level 3: Modulatory Neurons (Soltoggio-style)
        modulatory_neurons: Enable modulatory neuron types
        mod_neuron_ratio: Fraction of hidden neurons that are modulatory (default 0.1)
        mod_connection_type: How modulatory signal affects targets
            - 'multiplicative': target = activation * (1 + mod_signal)
            - 'additive': target = activation + mod_signal
            - 'gated': target = activation * sigmoid(mod_signal)
        mod_decay: Decay factor for modulatory accumulator (default 0.9)
    """
    enabled: bool = False

    # Level 1: Static Gating
    static_gating: bool = False
    gate_threshold: float = 0.5
    gate_scaling: str = 'sigmoid'  # 'sigmoid', 'binary', 'soft_threshold'

    # Level 2: Context-Dependent Gating
    context_gating: bool = False
    context_dim: int = 4
    context_influence: float = 0.5
    context_source: str = 'input'  # 'input', 'task_id', 'learned'

    # Level 3: Modulatory Neurons
    modulatory_neurons: bool = False
    mod_neuron_ratio: float = 0.1
    mod_connection_type: str = 'multiplicative'  # 'multiplicative', 'additive', 'gated'
    mod_decay: float = 0.9

    # Level 4: TRUE Neuromodulation (Neurotransmitter-based)
    # This is TRUE biological neuromodulation where:
    # - ALL network parameters (weights, receptor_densities) are FIXED after evolution
    # - ONLY the neurotransmitter signal varies per task
    # - Different NT levels → different activation behaviors → different network outputs
    # - ZERO task-specific learned parameters
    true_neuromodulation: bool = False
    num_nt_types: int = 2  # Number of neurotransmitter types (start with 2)
    modulation_strength: float = 2.0  # How strongly NT affects activation (higher = more differentiation)

    # Option A vs B for receptor density source:
    # - Option A (receptor_from_weight=True): Derive receptor densities from weight output
    #   Uses single CPPN output, derives receptor via receptor_derivation method
    # - Option B (receptor_from_weight=False): Separate CPPN outputs for receptor densities
    #   CPPN outputs: [weight, receptor_0, receptor_1, ...] (num_nt_types extra outputs)
    receptor_from_weight: bool = False  # If True, use Option A (single CPPN output)
    receptor_derivation: str = 'tanh'  # Method to derive receptor from weight: 'abs', 'normalized', 'tanh'


# ============================================================================
# Multi-Task Configuration for TRUE Neuromodulation
# ============================================================================

@dataclass
class MultiTaskConfig:
    """Configuration for multi-task evolution with TRUE neuromodulation.

    This enables TRUE multi-task evolution where:
    - NEAT selection uses AGGREGATED fitness across all tasks
    - Each task is evaluated with its task-specific NT vector
    - CPPN evolution happens ONCE per generation (not per task)

    Attributes:
        tasks: Dict mapping task_name -> (inputs, targets)
            Example: {'xor': (xor_inputs, xor_targets), 'and': (and_inputs, and_targets)}
        aggregation: How to combine task fitnesses for NEAT selection
            - 'mean': Average fitness across tasks (default)
            - 'min': Pessimistic - worst task determines fitness
            - 'weighted': Weighted average using `weights` dict
            - 'product': Geometric mean (encourages balance)
            - 'softmin': Differentiable min (weighted avg favoring low values)
            - 'harmonic': Harmonic mean (heavily penalizes low values)
        weights: Optional task weights for 'weighted' aggregation
            Example: {'xor': 2.0, 'and': 1.0} - XOR is twice as important
        nt_vectors: Optional custom NT vectors per task
            If None, uses get_nt_for_task() from NT_PRESETS
            Example: {'xor': jnp.array([1.0, 0.0]), 'and': jnp.array([0.0, 1.0])}
        modulation_penalty: Weight for modulation variance bonus (default 0.0 = disabled)
            When > 0, rewards networks that produce different modulation per task.
            This prevents "universal solutions" that ignore NT differentiation.
            Recommended range: 0.01-0.1
        fitness_mode: Fitness function to use for evaluation. Options:
            - 'mse': Mean squared error (default, allows fuzzy outputs)
            - 'accuracy': Pure binary accuracy (discrete: 0.25 steps)
            - 'acc_mse': Accuracy + MSE tiebreaker (best of both)
            - 'hybrid': Weighted 0.8*accuracy + 0.2*(1-MSE)
            - 'bce': Binary cross-entropy (standard classification)
            - 'soft_accuracy': Continuous approximation with temperature
    """
    tasks: Dict[str, Tuple[Any, Any]]  # task_name -> (inputs, targets)
    aggregation: str = 'mean'  # 'mean', 'min', 'weighted', 'product', 'softmin', 'harmonic'
    weights: Optional[Dict[str, float]] = None
    nt_vectors: Optional[Dict[str, Any]] = None  # Custom NT vectors per task
    modulation_penalty: float = 0.0  # Weight for modulation variance bonus
    fitness_mode: str = 'mse'  # 'mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy'

    # Softmin parameters
    softmin_temperature: float = 0.1  # Lower = sharper (more like true min)

    # Generalist bonus mechanisms
    generalist_bonus_type: str = 'none'  # 'none', 'min_bonus', 'variance_penalty', 'threshold_bonus'
    generalist_bonus_weight: float = 0.0  # Weight for the bonus/penalty
    generalist_threshold: float = 0.9  # Threshold for 'threshold_bonus' type

    # Modulation mechanism control
    modulation_mode: str = 'full'  # 'full', 'gating_only', 'gain_bias_only'
    modulation_strength_override: Optional[float] = None  # Override default 5.0 if set

    # Specialization bonus (confusion matrix gap)
    specialization_bonus_weight: float = 0.0  # Weight for NT-task alignment bonus
    confusion_eval_frequency: int = 1  # Compute confusion matrix every N generations

    # Subspace orthogonality bonus (Liu & Wang 2024 mechanism)
    # Forces different tasks to occupy orthogonal activation subspaces
    orthogonality_bonus_weight: float = 0.0  # 0.1-0.5 recommended; 0.0 = disabled
    orthogonality_metric: str = 'cosine_mean'  # 'cosine_mean', 'cosine_max', 'correlation'

    # Dendritic branch-specific gating (Liu & Wang 2024 SST mechanism)
    # Different spatial regions respond to different NT components
    branch_gating_mode: str = 'none'  # 'none', 'spatial', 'hierarchical'
    # - 'none': All neurons receive same modulation signal (current behavior)
    # - 'spatial': Divide neurons by x,y quadrants, each quadrant responds to different NT
    # - 'hierarchical': Neurons at different depths respond to different NTs

    # Two-Module Architecture (Liu & Wang 2024 PFC/Sensorimotor separation)
    # Separate hidden layer into rule-maintenance (PFC) and task-execution (Sensorimotor)
    two_module_mode: str = 'none'  # 'none', 'parallel', 'sequential'
    # - 'none': Standard single hidden layer (current behavior)
    # - 'parallel': PFC and Sensorimotor process independently, combine outputs
    # - 'sequential': PFC output feeds into Sensorimotor processing (top-down signal)


@dataclass
class MultiTaskMetrics:
    """Metrics from multi-task evolution with per-task breakdown.

    Attributes:
        aggregated_fitness: Best aggregated fitness (used for NEAT selection)
        mean_aggregated_fitness: Mean aggregated fitness across population
        per_task_best: Best fitness for each task (NOTE: max across population per task,
            may come from DIFFERENT individuals - use for population diversity tracking)
        per_task_mean: Mean fitness for each task across population
        best_generalist_per_task: Per-task fitness for THE SINGLE BEST GENERALIST
            (argmax of aggregated fitness). This is what you want for single-network
            performance claims.
        best_generalist_idx: Index of best generalist in population
        generation: Current generation number
    """
    aggregated_fitness: float = 0.0
    mean_aggregated_fitness: float = 0.0
    per_task_best: Dict[str, float] = field(default_factory=dict)
    per_task_mean: Dict[str, float] = field(default_factory=dict)
    best_generalist_per_task: Dict[str, float] = field(default_factory=dict)
    best_generalist_idx: int = 0
    generation: int = 0


# Preset configurations for common neuromodulation setups
NEUROMODULATION_PRESETS: Dict[str, NeuromodulationConfig] = {
    # Baseline: No neuromodulation
    'disabled': NeuromodulationConfig(enabled=False),

    # Level 1 only: Static per-connection gates from CPPN
    'static_gating': NeuromodulationConfig(
        enabled=True,
        static_gating=True,
    ),

    # Level 1 + 2: XdG-style context-dependent gating
    'xdg_style': NeuromodulationConfig(
        enabled=True,
        static_gating=True,
        context_gating=True,
        context_dim=4,
        context_influence=0.5,
    ),

    # Level 3 only: Modulatory neurons without gating
    'modulatory_only': NeuromodulationConfig(
        enabled=True,
        modulatory_neurons=True,
        mod_neuron_ratio=0.1,
    ),

    # All levels: Full neuromodulation stack
    'full': NeuromodulationConfig(
        enabled=True,
        static_gating=True,
        context_gating=True,
        context_dim=4,
        context_influence=0.5,
        modulatory_neurons=True,
        mod_neuron_ratio=0.1,
    ),

    # Level 4: TRUE Neuromodulation
    # Biologically-grounded: FIXED receptor densities + VARIABLE NT signal
    # Same network solves multiple tasks with different NT vectors
    # ZERO task-specific learned parameters
    # Option B: Separate CPPN outputs for receptor densities (default)
    'true_neuromodulation': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=2,
        modulation_strength=2.0,
        receptor_from_weight=False,  # Option B: separate CPPN outputs
    ),

    # Option A variants: Derive receptor densities from weight output
    # Uses single CPPN output, derives receptor via different methods
    'true_neuromodulation_option_a_abs': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=2,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='abs',  # receptor_0 = |w|, receptor_1 = 1 - |w|
    ),
    'true_neuromodulation_option_a_normalized': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=2,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='normalized',  # receptor_0 = (w+1)/2, receptor_1 = 1 - receptor_0
    ),
    'true_neuromodulation_option_a_tanh': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=2,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',  # receptor_0 = tanh(w), receptor_1 = tanh(-w)
    ),

    # ==========================================================================
    # Level 4: TRUE Neuromodulation with 4 NT Types
    # ==========================================================================
    # Biologically-inspired 4 NT system:
    # - NT1 (Dopamine-like): Reward, sharp decisions
    # - NT2 (Serotonin-like): Modulatory, stabilizing
    # - NT3 (Norepinephrine-like): Arousal, attention
    # - NT4 (Acetylcholine-like): Learning, plasticity

    # 4 NT Option B: Separate CPPN outputs (1 weight + 4 receptor densities = 5 outputs)
    'true_neuromodulation_4nt': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=4,
        modulation_strength=2.0,
        receptor_from_weight=False,  # Option B: 5 CPPN outputs needed
    ),

    # 4 NT Option A variants: Derive 4 receptors from weight (only 1 CPPN output!)
    'true_neuromodulation_4nt_option_a_abs': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=4,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='abs',
    ),
    'true_neuromodulation_4nt_option_a_normalized': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=4,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='normalized',
    ),
    'true_neuromodulation_4nt_option_a_tanh': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=4,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',  # Recommended for 4 NT types
    ),
    'true_neuromodulation_4nt_option_a_fourier': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=4,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='fourier',  # Orthogonal basis - avoids anti-correlation
    ),

    # ==========================================================================
    # Level 4: TRUE Neuromodulation with Variable NT Types (2, 3, 5, 6)
    # ==========================================================================
    # These modes support different NT dimensionalities for ablation studies.
    # Key insight: 5 tasks need 5 distinct directions. With N<5 dimensions,
    # some tasks will have high similarity and be confused by the network.

    # 2 NT Types: Simple but insufficient for 5+ tasks
    'true_neuromodulation_2nt_option_a_tanh': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=2,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',
    ),

    # 3 NT Types: Better separation with Fibonacci sphere
    'true_neuromodulation_3nt_option_a_tanh': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=3,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',
    ),

    # 5 NT Types: RECOMMENDED - Perfect orthogonal separation for 5 tasks
    # Uses one-hot encoding in NT_PRESETS_5 for 0.0 cosine similarity
    'true_neuromodulation_5nt_option_a_tanh': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=5,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',
    ),

    # 6 NT Types: Extra capacity for >5 tasks
    'true_neuromodulation_6nt_option_a_tanh': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=6,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',
    ),
}


def get_neuromodulation_config(mode: str, custom_config: Optional[Dict] = None) -> NeuromodulationConfig:
    """Get neuromodulation config from preset name or custom dict.

    Args:
        mode: Preset name ('disabled', 'static_gating', 'xdg_style', 'modulatory_only', 'full')
              or 'custom' for fully custom configuration
        custom_config: Optional dict to override preset values or provide custom config

    Returns:
        NeuromodulationConfig instance
    """
    if mode == 'custom' and custom_config:
        return NeuromodulationConfig(**custom_config)

    if mode not in NEUROMODULATION_PRESETS:
        print(f"[Neuromodulation] Warning: Unknown mode '{mode}', using 'disabled'")
        mode = 'disabled'

    config = NEUROMODULATION_PRESETS[mode]

    # Override with custom values if provided
    if custom_config:
        config_dict = {
            'enabled': config.enabled,
            'static_gating': config.static_gating,
            'gate_threshold': config.gate_threshold,
            'gate_scaling': config.gate_scaling,
            'context_gating': config.context_gating,
            'context_dim': config.context_dim,
            'context_influence': config.context_influence,
            'context_source': config.context_source,
            'modulatory_neurons': config.modulatory_neurons,
            'mod_neuron_ratio': config.mod_neuron_ratio,
            'mod_connection_type': config.mod_connection_type,
            'mod_decay': config.mod_decay,
            # Level 4: TRUE Neuromodulation
            'true_neuromodulation': config.true_neuromodulation,
            'num_nt_types': config.num_nt_types,
            'modulation_strength': config.modulation_strength,
            # Option A vs B receptor derivation
            'receptor_from_weight': config.receptor_from_weight,
            'receptor_derivation': config.receptor_derivation,
        }
        config_dict.update(custom_config)
        return NeuromodulationConfig(**config_dict)

    return config


# ============================================================================
# TRUE Neuromodulation: Neurotransmitter Presets (Level 4)
# ============================================================================
# Predefined NT vectors for common tasks.
# These are NOT learned - they are hand-specified or derived from task properties.
# The key insight: ONLY the NT vector changes per task, all network parameters stay FIXED.

# 2 NT Types: Simple dopamine-like / serotonin-like dichotomy
NT_PRESETS_2: Dict[str, jnp.ndarray] = {
    # Basic logic tasks with 2 NT types
    'xor': jnp.array([1.0, 0.0]),      # High NT1 for XOR (needs sharp nonlinear decisions)
    'and': jnp.array([0.0, 1.0]),      # High NT2 for AND (different activation pattern)
    'or': jnp.array([0.5, 0.5]),       # Balanced for OR (simple threshold)
    'nand': jnp.array([0.8, 0.2]),     # Similar to XOR but shifted
    'nor': jnp.array([0.2, 0.8]),      # Similar to OR but shifted
    'identity': jnp.array([0.3, 0.7]), # Pass-through behavior
}

# 4 NT Types: Inspired by biological neurotransmitter systems
# NT1 (Dopamine-like): Reward signals, sharp decisions, reinforcement
# NT2 (Serotonin-like): Modulatory, stabilizing, inhibitory
# NT3 (Norepinephrine-like): Arousal, attention, vigilance
# NT4: Reserved (kept at 1.0 for all tasks - output inversion removed)
#
# NOTE: Each task gets DISTINCT NT1-3 values for different hidden representations.
# Previous design used identical NT1-3 for AND/NAND (and OR/NOR) with NT4 for
# output inversion. This failed because inversion of 0.5 outputs is ambiguous.
# Now each task has unique modulation - no inversion needed.
NT_PRESETS_4: Dict[str, jnp.ndarray] = {
    # Logic tasks with 4 NT types
    # RESTORED: Original vectors that achieved 100% on all 5 tasks
    # Key insight: Mixed values (0.95, 0.05, 0.5) work better than pure orthogonal (0, 1)
    # Each task has DISTINCT NT1-3 values for different hidden layer representations
    # NT4=1.0 for all (no output inversion - it doesn't work reliably)
    'xor': jnp.array([0.95, 0.05, 0.95, 1.0]),  # HIGH-LOW-HIGH: distinct for XOR
    'and': jnp.array([0.1, 0.9, 0.1, 1.0]),     # LOW-HIGH-LOW
    'or': jnp.array([0.5, 0.5, 0.5, 1.0]),      # BALANCED
    'nand': jnp.array([0.9, 0.1, 0.5, 1.0]),    # HIGH-LOW-MID (distinct from AND!)
    'nor': jnp.array([0.1, 0.5, 0.9, 1.0]),     # LOW-MID-HIGH (distinct from OR!)
    # Other tasks
    'parity4': jnp.array([0.8, 0.2, 0.6, 1.0]),
    'identity': jnp.array([0.5, 0.5, 0.5, 1.0]),
    'mnist': jnp.array([0.6, 0.4, 0.7, 1.0]),
    'iris': jnp.array([0.5, 0.5, 0.5, 1.0]),
}

# 5 NT Types: One-hot encoding for PERFECT task separation
# CRITICAL: With 4 NT types, XOR and NAND have 0.960 cosine similarity!
# One-hot encoding guarantees 0.0 similarity between ALL task pairs.
# This is mathematically optimal for 5 distinct tasks.
NT_PRESETS_5: Dict[str, jnp.ndarray] = {
    # Logic tasks with 5 NT types - ORTHOGONAL (one-hot encoding)
    # Each task has a unique "channel" - perfect task discrimination
    'xor':  jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # Channel 1
    'and':  jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]),  # Channel 2
    'or':   jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]),  # Channel 3
    'nand': jnp.array([0.0, 0.0, 0.0, 1.0, 0.0]),  # Channel 4
    'nor':  jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]),  # Channel 5
    # Other tasks - use mixed patterns
    'parity4': jnp.array([0.8, 0.2, 0.0, 0.0, 0.0]),
    'identity': jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    'mnist': jnp.array([0.6, 0.4, 0.0, 0.0, 0.0]),
    'iris': jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
}

# 6 NT Types: Orthogonal + extra capacity (for >5 tasks)
NT_PRESETS_6: Dict[str, jnp.ndarray] = {
    'xor':  jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'and':  jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    'or':   jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    'nand': jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    'nor':  jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    # Channel 6 reserved for additional tasks
    'parity4': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    'identity': jnp.array([0.17, 0.17, 0.17, 0.17, 0.17, 0.17]),
    'mnist': jnp.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    'iris': jnp.array([0.17, 0.17, 0.17, 0.17, 0.17, 0.17]),
}

# 3 NT Types: Fibonacci-sphere optimized for maximum separation in 3D
# Uses golden angle distribution for evenly spaced points on sphere
# Better than NT_PRESETS_2 but still limited by 3D constraint
NT_PRESETS_3_FIBONACCI: Dict[str, jnp.ndarray] = {
    # Logic tasks - Fibonacci sphere distribution (normalized to [0,1])
    'xor':  jnp.array([0.800, 0.500, 0.900]),  # Point 1
    'and':  jnp.array([0.162, 0.190, 0.700]),  # Point 2
    'or':   jnp.array([0.544, 0.998, 0.500]),  # Point 3
    'nand': jnp.array([0.779, 0.136, 0.300]),  # Point 4
    'nor':  jnp.array([0.205, 0.552, 0.100]),  # Point 5
    # Other tasks
    'parity4': jnp.array([0.6, 0.3, 0.5]),
    'identity': jnp.array([0.5, 0.5, 0.5]),
    'mnist': jnp.array([0.7, 0.4, 0.6]),
    'iris': jnp.array([0.5, 0.5, 0.5]),
}

# Dynamic preset selector based on num_nt_types
NT_PRESETS: Dict[str, jnp.ndarray] = NT_PRESETS_2  # Default to 2 for backward compatibility


def get_nt_for_task(
    task_name: str,
    num_nt_types: int = 2,
    preset_name: Optional[str] = None
) -> jnp.ndarray:
    """Get predefined neurotransmitter vector for a task.

    Selects from appropriate preset dictionary based on num_nt_types or preset_name.

    Preset options:
    - num_nt_types=2: Uses NT_PRESETS_2 (dopamine-like / serotonin-like)
    - num_nt_types=3 or preset_name='fibonacci': Uses NT_PRESETS_3_FIBONACCI (optimally separated)
    - num_nt_types=4: Uses NT_PRESETS_4 (DA / 5-HT / NE / ACh inspired)
    - num_nt_types=5: Uses NT_PRESETS_5 (one-hot orthogonal - RECOMMENDED for 5 tasks!)
    - num_nt_types=6: Uses NT_PRESETS_6 (orthogonal + extra capacity)

    Args:
        task_name: Name of the task (lowercase)
        num_nt_types: Number of NT types (2, 3, 4, 5, or 6)
        preset_name: Optional explicit preset name ('fibonacci', 'onehot5', 'onehot6')

    Returns:
        Neurotransmitter vector with shape (num_nt_types,)
    """
    task_key = task_name.lower()

    # Select appropriate preset dictionary
    if preset_name == 'fibonacci' or (num_nt_types == 3 and preset_name is None):
        presets = NT_PRESETS_3_FIBONACCI
    elif preset_name == 'onehot5' or num_nt_types == 5:
        presets = NT_PRESETS_5
    elif preset_name == 'onehot6' or num_nt_types == 6:
        presets = NT_PRESETS_6
    elif num_nt_types == 4:
        presets = NT_PRESETS_4
    else:
        presets = NT_PRESETS_2

    if task_key in presets:
        nt = presets[task_key]
        if len(nt) < num_nt_types:
            # Pad with 0.5 (neutral) if more NT types needed
            nt = jnp.pad(nt, (0, num_nt_types - len(nt)), constant_values=0.5)
        return nt[:num_nt_types]
    else:
        # Default: uniform NT levels for unknown tasks
        return jnp.ones((num_nt_types,)) / num_nt_types


def compute_fitness(outputs: jnp.ndarray, targets: jnp.ndarray, mode: str = 'mse') -> float:
    """Compute fitness using specified mode.

    Different fitness functions have different properties:
    - MSE: Continuous, allows fuzzy outputs, can plateau at ~93.75%
    - Accuracy: Discrete (0.25 steps for 4-sample tasks), strong selection
    - Acc+MSE: Combines accuracy with MSE gradient as tiebreaker
    - Hybrid: Weighted combination of accuracy and MSE
    - BCE: Standard binary cross-entropy for classification
    - Soft accuracy: Continuous approximation of accuracy using temperature

    Args:
        outputs: Network outputs, shape (num_samples, num_outputs) or flattened
        targets: Target values, same shape as outputs
        mode: Fitness mode ('mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy')

    Returns:
        Fitness value in [0, 1] range (higher = better)
    """
    # Flatten if needed
    outputs = outputs.flatten()
    targets = targets.flatten()

    if mode == 'mse':
        # Current: MSE-based (allows fuzzy outputs)
        mse = jnp.mean((outputs - targets) ** 2)
        return 1.0 - mse

    elif mode == 'accuracy':
        # Pure binary accuracy - discrete values
        predictions = (outputs > 0.5).astype(jnp.float32)
        return jnp.mean(predictions == targets)

    elif mode == 'acc_mse':
        # Accuracy + MSE tiebreaker
        # Primary: binary accuracy (discrete)
        # Secondary: small MSE component for gradient when accuracy is equal
        predictions = (outputs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(predictions == targets)
        mse = jnp.mean((outputs - targets) ** 2)
        # Scale MSE to be a small tiebreaker (max contribution ~0.01)
        return accuracy + 0.01 * (1.0 - mse)

    elif mode == 'hybrid':
        # Weighted hybrid: 80% accuracy + 20% MSE
        # Balances discrete selection with continuous gradient
        predictions = (outputs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(predictions == targets)
        mse = jnp.mean((outputs - targets) ** 2)
        return 0.8 * accuracy + 0.2 * (1.0 - mse)

    elif mode == 'bce':
        # Binary cross-entropy (standard classification loss)
        # Converted to fitness (higher = better)
        eps = 1e-7
        outputs_safe = jnp.clip(outputs, eps, 1.0 - eps)
        bce = -jnp.mean(
            targets * jnp.log(outputs_safe) +
            (1.0 - targets) * jnp.log(1.0 - outputs_safe)
        )
        # Convert to fitness: perfect = 1.0, random = ~0.5
        return 1.0 / (1.0 + bce)

    elif mode == 'soft_accuracy':
        # Soft accuracy with temperature
        # Continuous approximation of binary accuracy
        # Higher temperature = sharper boundary at 0.5
        T = 10.0  # Temperature
        # Soft prediction: sigmoid steepens around 0.5
        soft_pred = jax.nn.sigmoid((outputs - 0.5) * T)
        # Soft accuracy: how close soft_pred is to target
        return jnp.mean(soft_pred * targets + (1.0 - soft_pred) * (1.0 - targets))

    elif mode == 'accuracy_confident':
        # Accuracy + confidence bonus (penalize outputs near 0.5)
        # This is CRITICAL for output inversion to work
        # Networks need to output values far from 0.5 so inversion is effective
        predictions = (outputs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(predictions == targets)
        # Confidence = distance from 0.5 (0 to 0.5 range, normalized to 0-1)
        confidence = jnp.mean(jnp.abs(outputs - 0.5) * 2.0)
        # Combine: accuracy is primary, confidence is secondary
        return accuracy + 0.1 * confidence

    else:
        raise ValueError(f"Unknown fitness mode: {mode}. "
                        f"Valid modes: 'mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy', 'accuracy_confident'")


def compute_subspace_orthogonality(
    hidden_list: List[jnp.ndarray],
    metric: str = 'cosine_mean',
) -> float:
    """Compute orthogonality bonus from hidden activations across tasks.

    Based on Liu & Wang (2024) finding: "Different rules occupy nearly ORTHOGONAL SUBSPACES"
    and "When SST neurons silenced, subspaces collapsed, performance = chance".

    This function measures how orthogonal task-specific activation patterns are.
    Higher orthogonality = better task separation = more likely to break the 75% barrier.

    Args:
        hidden_list: List of hidden activations per task, each shape (num_cases, total_positions)
        metric: Method to compute orthogonality
            - 'cosine_mean': Mean absolute cosine similarity (default, recommended)
            - 'cosine_max': Maximum absolute cosine similarity (strictest)
            - 'correlation': Mean absolute Pearson correlation

    Returns:
        Orthogonality bonus in [0, 1] where 1.0 = perfectly orthogonal (best)
    """
    num_tasks = len(hidden_list)
    if num_tasks < 2:
        return 1.0  # Single task is trivially orthogonal

    # Mean activation per task: shape (num_tasks, total_positions)
    # This creates a "task fingerprint" in activation space
    mean_activations = jnp.stack([jnp.mean(h, axis=0) for h in hidden_list])

    # Normalize to unit vectors
    norms = jnp.linalg.norm(mean_activations, axis=1, keepdims=True) + 1e-8
    norm_activations = mean_activations / norms

    # Cosine similarity matrix: S[i,j] = dot(task_i, task_j)
    # Diagonal = 1.0 (self-similarity), off-diagonal = cross-task similarity
    sim_matrix = norm_activations @ norm_activations.T

    # Off-diagonal mask (exclude self-similarity)
    mask = 1 - jnp.eye(num_tasks)

    if metric == 'cosine_mean':
        # Mean of absolute off-diagonal similarities
        # Lower similarity -> higher orthogonality
        off_diag = jnp.abs(sim_matrix * mask)
        mean_sim = jnp.sum(off_diag) / (num_tasks * (num_tasks - 1))
        return 1.0 - jnp.clip(mean_sim, 0.0, 1.0)

    elif metric == 'cosine_max':
        # Maximum absolute similarity (strictest criterion)
        # Even one highly correlated pair reduces bonus
        off_diag = jnp.abs(sim_matrix * mask)
        max_sim = jnp.max(off_diag)
        return 1.0 - jnp.clip(max_sim, 0.0, 1.0)

    elif metric == 'correlation':
        # Pearson correlation instead of cosine similarity
        # Centers activations first (removes mean)
        centered = mean_activations - jnp.mean(mean_activations, axis=1, keepdims=True)
        std_norms = jnp.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
        centered_norm = centered / std_norms
        corr_matrix = centered_norm @ centered_norm.T
        off_diag = jnp.abs(corr_matrix * mask)
        mean_corr = jnp.sum(off_diag) / (num_tasks * (num_tasks - 1))
        return 1.0 - jnp.clip(mean_corr, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown orthogonality metric: {metric}. "
                        f"Valid options: 'cosine_mean', 'cosine_max', 'correlation'")


def compute_population_orthogonality(
    per_task_hidden: Dict[str, jnp.ndarray],
    metric: str = 'cosine_mean',
) -> jnp.ndarray:
    """Compute orthogonality bonus for entire population.

    Vectorized version that computes orthogonality bonuses for all individuals
    in the population simultaneously.

    Args:
        per_task_hidden: Dict mapping task_name -> hidden activations array
            Each array has shape (pop_size, num_cases, total_positions)
        metric: Orthogonality metric (see compute_subspace_orthogonality)

    Returns:
        Array of orthogonality bonuses, shape (pop_size,)
    """
    task_names = list(per_task_hidden.keys())
    num_tasks = len(task_names)

    if num_tasks < 2:
        # Single task - return perfect orthogonality for everyone
        first_hidden = per_task_hidden[task_names[0]]
        pop_size = first_hidden.shape[0]
        return jnp.ones(pop_size)

    # Stack hidden activations: (num_tasks, pop_size, num_cases, total_positions)
    hidden_stack = jnp.stack([per_task_hidden[t] for t in task_names], axis=0)

    # Mean over cases -> task fingerprints: (num_tasks, pop_size, total_positions)
    mean_activations = jnp.mean(hidden_stack, axis=2)

    # Transpose for easier processing: (pop_size, num_tasks, total_positions)
    mean_activations = jnp.transpose(mean_activations, (1, 0, 2))

    # Normalize per-task fingerprints to unit vectors
    norms = jnp.linalg.norm(mean_activations, axis=2, keepdims=True) + 1e-8
    norm_activations = mean_activations / norms

    # Cosine similarity matrices per individual: (pop_size, num_tasks, num_tasks)
    # sim_matrix[p, i, j] = dot(task_i, task_j) for individual p
    sim_matrices = jnp.einsum('ptn,pmn->ptm', norm_activations, norm_activations)

    # Off-diagonal mask
    mask = 1 - jnp.eye(num_tasks)

    if metric == 'cosine_mean':
        off_diag = jnp.abs(sim_matrices) * mask  # (pop_size, num_tasks, num_tasks)
        mean_sim = jnp.sum(off_diag, axis=(1, 2)) / (num_tasks * (num_tasks - 1))
        return 1.0 - jnp.clip(mean_sim, 0.0, 1.0)

    elif metric == 'cosine_max':
        off_diag = jnp.abs(sim_matrices) * mask
        max_sim = jnp.max(off_diag, axis=(1, 2))
        return 1.0 - jnp.clip(max_sim, 0.0, 1.0)

    elif metric == 'correlation':
        # Center activations per individual per task
        centered = mean_activations - jnp.mean(mean_activations, axis=2, keepdims=True)
        std_norms = jnp.linalg.norm(centered, axis=2, keepdims=True) + 1e-8
        centered_norm = centered / std_norms
        corr_matrices = jnp.einsum('ptn,pmn->ptm', centered_norm, centered_norm)
        off_diag = jnp.abs(corr_matrices) * mask
        mean_corr = jnp.sum(off_diag, axis=(1, 2)) / (num_tasks * (num_tasks - 1))
        return 1.0 - jnp.clip(mean_corr, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown orthogonality metric: {metric}")


def compute_branch_specific_modulation(
    receptor_densities_single: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    all_positions: jnp.ndarray,
    branch_mode: str = 'none',
) -> jnp.ndarray:
    """Compute branch-specific NT routing inspired by Liu & Wang (2024).

    In Liu & Wang (2024), SST interneurons target specific dendritic branches,
    enabling branch-specific modulation that creates task-selective subnetworks.

    This function implements a simplified version: different spatial regions
    of the hidden layer respond to different NT components.

    Args:
        receptor_densities_single: Per-neuron NT sensitivity, shape (total_positions, num_nt_types)
        neurotransmitter: NT vector for current task, shape (num_nt_types,)
        all_positions: Spatial coordinates of all positions, shape (total_positions, 2)
        branch_mode: Gating mode
            - 'none': Standard receptor @ NT dot product (all neurons, same signal)
            - 'spatial': Quadrant-based routing (4 regions, 4 NT components)
            - 'hierarchical': Depth-based routing (different y-levels get different NTs)

    Returns:
        Per-neuron modulation signal, shape (total_positions,)
    """
    if branch_mode == 'none':
        # Original behavior: all neurons receive linear combination of NTs
        return receptor_densities_single[:, :3] @ neurotransmitter[:3]

    elif branch_mode == 'spatial':
        # Divide neurons into 4 quadrants by x,y coordinates
        # Each quadrant responds primarily to one NT component
        # This creates task-selective subnetworks
        x = all_positions[:, 0]
        y = all_positions[:, 1]

        # Initialize modulation array
        modulation = jnp.zeros(receptor_densities_single.shape[0])

        # Quadrant 0 (x<0, y<0): responds to NT[0]
        # Biological analog: proximal dendrites
        mask_0 = (x < 0.0) & (y < 0.0)
        mod_0 = receptor_densities_single[:, 0] * neurotransmitter[0]
        modulation = jnp.where(mask_0, mod_0, modulation)

        # Quadrant 1 (x>=0, y<0): responds to NT[1]
        # Biological analog: mid-apical dendrites
        mask_1 = (x >= 0.0) & (y < 0.0)
        mod_1 = receptor_densities_single[:, 1] * neurotransmitter[1]
        modulation = jnp.where(mask_1, mod_1, modulation)

        # Quadrant 2 (x<0, y>=0): responds to NT[2]
        # Biological analog: distal dendrites
        mask_2 = (x < 0.0) & (y >= 0.0)
        mod_2 = receptor_densities_single[:, 2] * neurotransmitter[2]
        modulation = jnp.where(mask_2, mod_2, modulation)

        # Quadrant 3 (x>=0, y>=0): mixed signal (integration zone)
        # Biological analog: soma integration
        mask_3 = (x >= 0.0) & (y >= 0.0)
        mod_3 = receptor_densities_single[:, :3] @ neurotransmitter[:3]
        modulation = jnp.where(mask_3, mod_3, modulation)

        return modulation

    elif branch_mode == 'hierarchical':
        # Divide neurons by y-coordinate (depth/layer)
        # Lower y = earlier processing, higher y = later processing
        y = all_positions[:, 1]

        # Initialize modulation array
        modulation = jnp.zeros(receptor_densities_single.shape[0])

        # Layer 1 (y < -0.33): responds to NT[0] (early processing)
        mask_1 = y < -0.33
        mod_1 = receptor_densities_single[:, 0] * neurotransmitter[0]
        modulation = jnp.where(mask_1, mod_1, modulation)

        # Layer 2 (-0.33 <= y < 0.33): responds to NT[1] (mid processing)
        mask_2 = (y >= -0.33) & (y < 0.33)
        mod_2 = receptor_densities_single[:, 1] * neurotransmitter[1]
        modulation = jnp.where(mask_2, mod_2, modulation)

        # Layer 3 (y >= 0.33): responds to NT[2] (late/integration)
        mask_3 = y >= 0.33
        mod_3 = receptor_densities_single[:, 2] * neurotransmitter[2]
        modulation = jnp.where(mask_3, mod_3, modulation)

        return modulation

    else:
        raise ValueError(f"Unknown branch_gating_mode: {branch_mode}. "
                        f"Valid options: 'none', 'spatial', 'hierarchical'")


def derive_receptor_from_weight(
    weight: jnp.ndarray,
    method: str = 'tanh',
    num_nt_types: int = 2,
) -> jnp.ndarray:
    """Derive receptor densities from weight value (Option A).

    This is the key function for Option A TRUE neuromodulation:
    Instead of separate CPPN outputs for receptor densities, we derive
    them from the weight output using one of three methods.

    For 4 NT types, we use both the magnitude and sign of the weight to
    create 4 receptor density values, inspired by biological NT systems:
    - NT1 (Dopamine-like): Based on |w| magnitude
    - NT2 (Serotonin-like): Inverse of |w| magnitude
    - NT3 (Norepinephrine-like): Based on sign(w) * |w|
    - NT4 (Acetylcholine-like): Based on |w| variance from center (0.5)

    Args:
        weight: Weight values from CPPN, any shape (typically (pop, positions) or scalar)
        method: Derivation method
            - 'abs': receptor_0 = |w|, receptor_1 = 1 - |w|
              Interpretation: Strong connections (large |w|) → high NT1 sensitivity
            - 'normalized': receptor_0 = (w+1)/2, receptor_1 = 1 - receptor_0
              Interpretation: Maps [-1, 1] to [0, 1], complementary receptors
            - 'tanh': receptor_0 = (tanh(w)+1)/2, receptor_1 = (tanh(-w)+1)/2
              Interpretation: Soft saturation, allows both receptors to be active
        num_nt_types: Number of NT types (2 or 4 supported)

    Returns:
        Receptor densities with shape (..., num_nt_types)
        Last axis has length num_nt_types

    Example:
        weight = jnp.array([[-0.5, 0.0, 0.5, 1.0]])  # shape (1, 4)
        receptors = derive_receptor_from_weight(weight, 'tanh')  # shape (1, 4, 2)
        receptors_4nt = derive_receptor_from_weight(weight, 'tanh', num_nt_types=4)  # shape (1, 4, 4)
    """
    if num_nt_types not in (2, 3, 4, 5, 6):
        raise ValueError(
            f"derive_receptor_from_weight supports num_nt_types=2, 3, 4, 5, or 6, "
            f"got {num_nt_types}. For other NT counts, use Option B (separate CPPN outputs)."
        )

    # Initialize receptor list
    receptors = []

    if method == 'abs':
        # |w| → strong connection = high sensitivity to NT1
        abs_w = jnp.clip(jnp.abs(weight), 0.0, 1.0)
        receptors.append(abs_w)                                        # receptor_0
        receptors.append(1.0 - abs_w)                                  # receptor_1
        if num_nt_types >= 3:
            # NT3: Based on positive weights (excitatory-associated)
            receptors.append(jnp.clip((weight + 1.0) / 2.0, 0.0, 1.0))  # receptor_2
        if num_nt_types >= 4:
            # NT4: Based on deviation from center - high when |w| ≈ 0.5
            receptors.append(1.0 - jnp.abs(abs_w - 0.5) * 2.0)          # receptor_3
        if num_nt_types >= 5:
            # NT5: Quadratic response to extreme weights
            receptors.append(abs_w ** 2)                                # receptor_4
        if num_nt_types >= 6:
            # NT6: Inverse quadratic - high for moderate weights
            receptors.append(1.0 - abs_w ** 2)                          # receptor_5

    elif method == 'normalized':
        # Map [-1, 1] to [0, 1] linearly
        receptor_0 = jnp.clip((weight + 1.0) / 2.0, 0.0, 1.0)
        receptors.append(receptor_0)                                    # receptor_0
        receptors.append(1.0 - receptor_0)                              # receptor_1
        if num_nt_types >= 3:
            # NT3: Squared magnitude (stronger response to extreme weights)
            receptors.append(weight ** 2)                               # receptor_2
        if num_nt_types >= 4:
            # NT4: Quadratic center preference
            receptors.append(1.0 - (weight ** 2))                       # receptor_3
        if num_nt_types >= 5:
            # NT5: Cubic asymmetry
            receptors.append(jnp.clip((weight ** 3 + 1.0) / 2.0, 0.0, 1.0))  # receptor_4
        if num_nt_types >= 6:
            # NT6: Sign-magnitude interaction
            receptors.append(jnp.clip(jnp.abs(weight) * jnp.sign(weight + 0.5), 0.0, 1.0))

    elif method == 'tanh':
        # Soft saturation with tanh - the recommended method
        tanh_w = jnp.tanh(weight)
        receptors.append((tanh_w + 1.0) / 2.0)                          # receptor_0: High for positive
        receptors.append((-tanh_w + 1.0) / 2.0)                         # receptor_1: High for negative
        if num_nt_types >= 3:
            # NT3: Sigmoid of magnitude - smooth activation
            receptors.append(jax.nn.sigmoid(jnp.abs(weight) * 2.0 - 1.0))  # receptor_2
        if num_nt_types >= 4:
            # NT4: Inverse sigmoid - high for weak weights
            receptors.append(jax.nn.sigmoid(1.0 - jnp.abs(weight) * 2.0))  # receptor_3
        if num_nt_types >= 5:
            # NT5: Tanh of double - faster saturation
            tanh_2w = jnp.tanh(2.0 * weight)
            receptors.append((tanh_2w + 1.0) / 2.0)                     # receptor_4
        if num_nt_types >= 6:
            # NT6: Complementary to NT5
            receptors.append((-jnp.tanh(2.0 * weight) + 1.0) / 2.0)     # receptor_5

    elif method == 'fourier':
        # Fourier basis - ORTHOGONAL decomposition (NO anti-correlation!)
        # sin(x) and cos(x) are orthogonal: their correlation = 0
        receptors.append((jnp.sin(weight * jnp.pi) + 1.0) / 2.0)        # sin(πw)
        receptors.append((jnp.cos(weight * jnp.pi) + 1.0) / 2.0)        # cos(πw)
        if num_nt_types >= 3:
            # Higher harmonic
            receptors.append((jnp.sin(2.0 * weight * jnp.pi) + 1.0) / 2.0)  # sin(2πw)
        if num_nt_types >= 4:
            receptors.append((jnp.cos(2.0 * weight * jnp.pi) + 1.0) / 2.0)  # cos(2πw)
        if num_nt_types >= 5:
            receptors.append((jnp.sin(3.0 * weight * jnp.pi) + 1.0) / 2.0)  # sin(3πw)
        if num_nt_types >= 6:
            receptors.append((jnp.cos(3.0 * weight * jnp.pi) + 1.0) / 2.0)  # cos(3πw)

    elif method == 'softmax':
        # Competitive allocation - receptors sum to 1.0
        # Each receptor competes for "attention" from the NT vector
        # This breaks anti-correlation by making receptors mutually exclusive
        raw_scores = []
        raw_scores.append(jnp.tanh(weight))                             # Base response
        raw_scores.append(jnp.tanh(weight - 0.5))                       # Shifted down
        if num_nt_types >= 3:
            raw_scores.append(jnp.tanh(weight + 0.5))                   # Shifted up
        if num_nt_types >= 4:
            raw_scores.append(jnp.sin(weight * jnp.pi))                 # Periodic
        if num_nt_types >= 5:
            raw_scores.append(jnp.cos(weight * jnp.pi))                 # Periodic orthogonal
        if num_nt_types >= 6:
            raw_scores.append(jnp.tanh(2.0 * weight))                   # Faster saturation

        # Apply softmax to get competitive allocation (sum to 1.0)
        scores_stack = jnp.stack(raw_scores[:num_nt_types], axis=-1)
        softmax_receptors = jax.nn.softmax(scores_stack, axis=-1)

        # Return directly - already stacked
        return softmax_receptors

    elif method == 'orthogonal':
        # Generate orthogonal receptor basis using rotation from single weight
        # Each receptor spans a different direction in NT space
        # This ensures NO forced correlations between receptors
        angle = weight * jnp.pi  # Map weight to rotation angle

        # Generate orthogonal basis vectors via rotation
        # For 2D: standard rotation
        receptors.append((jnp.cos(angle) + 1.0) / 2.0)                  # receptor_0
        receptors.append((jnp.sin(angle) + 1.0) / 2.0)                  # receptor_1

        if num_nt_types >= 3:
            # For 3D+: use different rotation frequencies to maintain orthogonality
            receptors.append((jnp.cos(2.0 * angle + jnp.pi / 3) + 1.0) / 2.0)  # receptor_2

        if num_nt_types >= 4:
            receptors.append((jnp.sin(2.0 * angle + jnp.pi / 6) + 1.0) / 2.0)  # receptor_3

        if num_nt_types >= 5:
            # Higher frequency for more independence
            receptors.append((jnp.cos(3.0 * angle + jnp.pi / 4) + 1.0) / 2.0)  # receptor_4

        if num_nt_types >= 6:
            receptors.append((jnp.sin(3.0 * angle + jnp.pi / 5) + 1.0) / 2.0)  # receptor_5

    elif method == 'phase_shifted':
        # Phase-shifted Fourier - better separation for XOR discrimination
        # Uses non-standard phases to break symmetries between tasks
        receptors.append((jnp.sin(weight * jnp.pi) + 1.0) / 2.0)                    # sin(πw)
        receptors.append((jnp.cos(weight * jnp.pi + jnp.pi / 4) + 1.0) / 2.0)       # cos(πw + π/4)
        if num_nt_types >= 3:
            receptors.append((jnp.sin(2.0 * weight * jnp.pi + jnp.pi / 3) + 1.0) / 2.0)
        if num_nt_types >= 4:
            receptors.append((jnp.cos(2.0 * weight * jnp.pi + jnp.pi / 6) + 1.0) / 2.0)
        if num_nt_types >= 5:
            receptors.append((jnp.sin(3.0 * weight * jnp.pi - jnp.pi / 4) + 1.0) / 2.0)
        if num_nt_types >= 6:
            receptors.append((jnp.cos(3.0 * weight * jnp.pi - jnp.pi / 5) + 1.0) / 2.0)

    else:
        raise ValueError(
            f"Unknown receptor derivation method: '{method}'. "
            f"Valid methods: 'abs', 'normalized', 'tanh', 'fourier', 'softmax', 'orthogonal', 'phase_shifted'"
        )

    return jnp.stack(receptors[:num_nt_types], axis=-1)


# ============================================================================
# Neuromodulation Gate Scaling Functions (Level 1)
# ============================================================================

def apply_gate_scaling(gate_raw: jnp.ndarray, scaling: str, threshold: float = 0.5) -> jnp.ndarray:
    """Convert raw CPPN gate output to [0, 1] gate value.

    Args:
        gate_raw: Raw gate values from CPPN (typically in [-1, 1] from tanh)
        scaling: Scaling method
            - 'sigmoid': Smooth mapping to [0, 1]
            - 'binary': Hard threshold to 0 or 1
            - 'soft_threshold': Values below threshold become 0, above ramp to 1
        threshold: Threshold value for binary/soft_threshold methods

    Returns:
        Gate values in [0, 1] with same shape as input
    """
    if scaling == 'sigmoid':
        # Standard sigmoid for smooth gates
        return jax.nn.sigmoid(gate_raw)
    elif scaling == 'binary':
        # Hard binary: 0 if below threshold, 1 if above
        sig = jax.nn.sigmoid(gate_raw)
        return jnp.where(sig > threshold, 1.0, 0.0)
    elif scaling == 'soft_threshold':
        # Soft threshold: ramp from threshold to 1
        sig = jax.nn.sigmoid(gate_raw)
        # Below threshold -> 0, above threshold -> scaled to [0, 1]
        return jnp.where(sig > threshold, (sig - threshold) / (1.0 - threshold + 1e-8), 0.0)
    else:
        # Default fallback to sigmoid
        return jax.nn.sigmoid(gate_raw)


# ============================================================================
# Context-Dependent Gating Functions (Level 2)
# ============================================================================

def derive_context_from_input(inputs: jnp.ndarray, method: str = 'statistics') -> jnp.ndarray:
    """Derive context vector from input patterns.

    Args:
        inputs: Input data, shape (batch, num_inputs) or (num_inputs,)
        method: How to derive context
            - 'statistics': [mean, std, min, max] of inputs
            - 'mean': Just the mean value
            - 'pca': First N principal components (future)

    Returns:
        Context vector, shape (context_dim,)
    """
    # Flatten to 1D for statistics if needed
    flat = inputs.flatten() if inputs.ndim > 1 else inputs

    if method == 'mean':
        return jnp.array([jnp.mean(flat)])
    elif method == 'statistics':
        return jnp.array([
            jnp.mean(flat),
            jnp.std(flat) + 1e-8,  # Avoid zero std
            jnp.min(flat),
            jnp.max(flat),
        ])
    else:
        # Default to statistics
        return jnp.array([jnp.mean(flat), jnp.std(flat) + 1e-8, jnp.min(flat), jnp.max(flat)])


def project_context_to_gates(
    context: jnp.ndarray,
    base_gates_logit: jnp.ndarray,
    context_weights: jnp.ndarray,
    context_bias: jnp.ndarray,
    influence: float = 0.5,
) -> jnp.ndarray:
    """Modulate gate values based on context vector.

    Combines base gates (from CPPN) with context-dependent modulation.

    Args:
        context: Context vector, shape (context_dim,)
        base_gates_logit: Pre-sigmoid gate values from CPPN, shape (num_connections,)
        context_weights: Projection weights, shape (context_dim, num_connections)
        context_bias: Projection bias, shape (num_connections,)
        influence: How much context affects gates [0, 1]

    Returns:
        Final gate values in [0, 1], shape (num_connections,)
    """
    # Project context to connection space
    context_projection = jnp.dot(context, context_weights) + context_bias

    # Combine base gates with context modulation
    combined_logit = base_gates_logit + influence * context_projection

    return jax.nn.sigmoid(combined_logit)


# ============================================================================
# Modulatory Neuron Functions (Level 3)
# ============================================================================

def assign_node_types(
    num_hidden: int,
    node_type_raw: jnp.ndarray,
    mod_ratio: float,
) -> jnp.ndarray:
    """Assign node types: 0=standard, 1=modulatory.

    Uses CPPN output to determine which hidden nodes are modulatory.
    Input and output nodes are always standard (type 0).

    Args:
        num_hidden: Number of hidden nodes
        node_type_raw: Raw CPPN output for node type, shape (num_hidden,)
        mod_ratio: Target fraction of modulatory neurons

    Returns:
        Node types, shape (num_hidden,) with 0=standard, 1=modulatory
    """
    # Use percentile-based assignment to get roughly mod_ratio modulatory
    # Higher raw values -> modulatory
    threshold = jnp.percentile(node_type_raw, (1.0 - mod_ratio) * 100)
    node_types = (node_type_raw > threshold).astype(jnp.int32)
    return node_types


def compute_modulation_signal(
    node_activations: jnp.ndarray,
    node_types: jnp.ndarray,
    W_mod: jnp.ndarray,
) -> jnp.ndarray:
    """Compute modulation signal from modulatory neurons.

    Modulatory neurons (type 1) send modulation signals to standard neurons.
    The modulation signal affects how standard neurons process information.

    Args:
        node_activations: Current activations of all hidden nodes, shape (num_hidden,)
        node_types: Node types (0=standard, 1=modulatory), shape (num_hidden,)
        W_mod: Modulation weight matrix, shape (num_hidden, num_hidden)
               W_mod[i,j] = how much node i modulates node j

    Returns:
        Modulation signal for each node, shape (num_hidden,)
    """
    # Only modulatory neurons contribute to modulation signal
    # Create mask: 1 for modulatory neurons, 0 for standard
    mod_mask = (node_types == 1).astype(jnp.float32)

    # Modulation contribution: activation * is_modulatory
    mod_contributions = node_activations * mod_mask

    # Propagate through modulation weights
    # Result: sum of weighted modulation signals reaching each node
    mod_signal = jnp.dot(mod_contributions, W_mod)

    # Squash to bounded range
    return jnp.tanh(mod_signal)


def apply_modulation(
    activations: jnp.ndarray,
    mod_signal: jnp.ndarray,
    mod_type: str = 'multiplicative',
) -> jnp.ndarray:
    """Apply modulation signal to node activations.

    Args:
        activations: Pre-modulation activations, shape (batch, num_nodes) or (num_nodes,)
        mod_signal: Modulation signal from modulatory neurons, shape (num_nodes,)
        mod_type: How modulation affects activations
            - 'multiplicative': output = activation * (1 + mod_signal)
            - 'additive': output = activation + mod_signal
            - 'gated': output = activation * sigmoid(mod_signal)

    Returns:
        Modulated activations with same shape as input
    """
    if mod_type == 'multiplicative':
        # Multiplicative: modulation scales the activation
        # mod_signal in [-1, 1] from tanh, so (1 + mod) in [0, 2]
        return activations * (1.0 + mod_signal)
    elif mod_type == 'additive':
        # Additive: modulation shifts the activation
        return activations + mod_signal
    elif mod_type == 'gated':
        # Gated: modulation controls how much signal passes through
        gate = jax.nn.sigmoid(mod_signal)
        return activations * gate
    else:
        # Default to multiplicative
        return activations * (1.0 + mod_signal)


# ============================================================================
# CPPN Output Index Constants for Neuromodulation
# ============================================================================
# When neuromodulation is enabled, CPPN outputs additional values beyond weights

CPPN_WEIGHT = 0           # Connection weight (always present)
CPPN_ACTIVATION = 1       # Activation index (if dynamic_functions uses cppn_output)
CPPN_GATE = 2             # Gate value (if static_gating or context_gating)
CPPN_MOD_FACTOR = 3       # Modulation factor (if modulatory_neurons)
CPPN_NODE_TYPE = 4        # Node type raw value (if modulatory_neurons)
# CPPN_RECEPTOR_BASE = dynamically computed (if true_neuromodulation)
# Receptor density outputs start after all other outputs


def get_cppn_num_outputs(
    dynamic_functions_mode: str,
    neuromod_config: NeuromodulationConfig,
) -> int:
    """Compute number of CPPN outputs needed for current configuration.

    Args:
        dynamic_functions_mode: Mode from dynamic_functions config
        neuromod_config: Neuromodulation configuration

    Returns:
        Number of CPPN outputs required
    """
    # Base: 1 output for weight
    outputs = 1

    # +1 for activation index in cppn_output mode
    if dynamic_functions_mode == 'cppn_output':
        outputs += 1

    # +1 for gate if any gating is enabled
    if neuromod_config.enabled and (neuromod_config.static_gating or neuromod_config.context_gating):
        outputs += 1

    # +2 for modulatory neurons (mod_factor + node_type)
    if neuromod_config.enabled and neuromod_config.modulatory_neurons:
        outputs += 2

    # +num_nt_types for TRUE neuromodulation receptor densities (Option B only)
    # Each NT type gets its own CPPN output for per-node receptor sensitivity
    # Option A (receptor_from_weight=True): Derive receptor from weight, no extra outputs needed
    # Option B (receptor_from_weight=False): Separate CPPN outputs for receptor densities
    if neuromod_config.enabled and neuromod_config.true_neuromodulation:
        if not neuromod_config.receptor_from_weight:
            # Option B: separate outputs for receptor densities
            outputs += neuromod_config.num_nt_types
        # Option A: no additional outputs, derive from weight

    return outputs


def get_cppn_receptor_base_idx(
    dynamic_functions_mode: str,
    neuromod_config: NeuromodulationConfig,
) -> int:
    """Get the starting CPPN output index for receptor density outputs.

    Receptor densities come AFTER all other outputs:
    - 0: weight (always)
    - 1: activation (if cppn_output mode)
    - next: gate (if gating enabled)
    - next 2: mod_factor, node_type (if modulatory)
    - RECEPTOR_BASE: first receptor density output

    Args:
        dynamic_functions_mode: Mode from dynamic_functions config
        neuromod_config: Neuromodulation configuration

    Returns:
        Starting index for receptor density outputs in CPPN
    """
    base_idx = 1  # Always have weight at 0

    if dynamic_functions_mode == 'cppn_output':
        base_idx += 1  # activation

    if neuromod_config.enabled and (neuromod_config.static_gating or neuromod_config.context_gating):
        base_idx += 1  # gate

    if neuromod_config.enabled and neuromod_config.modulatory_neurons:
        base_idx += 2  # mod_factor + node_type

    return base_idx


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
# Dynamic Activation Functions Support
# ============================================================================
# This section provides infrastructure for evolved/configurable activation functions.
# Part of progressive A/B testing study.

# Available activation functions for dynamic selection
ACTIVATION_FUNCTIONS = {
    # Original functions
    'tanh': jnp.tanh,
    'sigmoid': jax.nn.sigmoid,
    'relu': jax.nn.relu,
    'sin': jnp.sin,
    'gauss': lambda x: jnp.exp(-x**2),
    'identity': lambda x: x,
    'lelu': lambda x: jnp.where(x > 0, x, 0.01 * x),  # Leaky ReLU
    'abs': jnp.abs,
    'scaled_tanh': lambda x: jnp.tanh(x) * 3.0,
    # Static activation functions with shapes inspired by Izhikevich neuron response curves.
    # NOT actual spiking dynamics - just convenient nonlinearities that vaguely resemble
    # the input-output characteristics of different neocortical neuron types.
    # Reference: https://www.izhikevich.org/publications/spikes.htm
    'softplus': lambda x: jnp.log1p(jnp.exp(x)),  # Smooth ReLU
    'rs_adapt': lambda x: jnp.tanh(x) * (1 - 0.3 * jnp.abs(x)),  # Self-attenuating tanh
    'fs_fast': lambda x: jax.nn.relu(x) * 2.0,  # Scaled ReLU
    'lts_low': lambda x: jax.nn.sigmoid(x * 2 - 0.5),  # Shifted sigmoid
    'burst': lambda x: jnp.tanh(x) + 0.5 * jnp.sin(x * 3),  # Tanh + sine oscillation
    'resonator': lambda x: jnp.sin(x) * jnp.exp(-jnp.abs(x) / 3),  # Damped sine
}

# Ordered list for indexing (used in cppn_output and weight_interpretation modes)
# Original functions first (preserve backward compatibility), then Izhikevich-inspired
ACTIVATION_LIST = [
    # Original functions (indices 0-6)
    'tanh', 'sigmoid', 'relu', 'identity', 'sin', 'gauss', 'lelu',
    # Izhikevich-inspired (indices 7-12)
    'softplus', 'rs_adapt', 'fs_fast', 'lts_low', 'burst', 'resonator',
]


def continuous_to_index(raw: jnp.ndarray, num_options: int) -> jnp.ndarray:
    """Convert continuous CPPN output to discrete function index.

    Maps values (typically in [-1, 1] from tanh) to discrete indices [0, num_options-1].
    Uses tanh to normalize before scaling to handle any input range.

    Args:
        raw: Continuous values from CPPN
        num_options: Number of function options

    Returns:
        Integer indices in [0, num_options-1]
    """
    # Normalize to [0, 1] then scale to [0, num_options)
    scaled = (jnp.tanh(raw) + 1.0) / 2.0 * num_options
    return jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, num_options - 1)


def grouped_activation_forward(
    pre_activation: jnp.ndarray,
    act_indices: jnp.ndarray,
    num_activations: int = 4,
) -> jnp.ndarray:
    """Apply different activation functions to different nodes via grouping.

    JAX-efficient implementation: applies each function to all nodes, then
    uses masks to combine results. Avoids per-node branching.

    Args:
        pre_activation: Pre-activation values, shape (batch, num_nodes)
        act_indices: Per-node activation indices, shape (num_nodes,)
        num_activations: Number of activation functions available

    Returns:
        Activated values with same shape as pre_activation
    """
    # Get activation functions as list
    activation_funcs = [ACTIVATION_FUNCTIONS[name] for name in ACTIVATION_LIST[:num_activations]]

    result = jnp.zeros_like(pre_activation)
    for idx, func in enumerate(activation_funcs):
        mask = (act_indices == idx)
        activated = func(pre_activation)
        # Broadcast mask across batch dimension
        result = jnp.where(mask[None, :] if pre_activation.ndim == 2 else mask, activated, result)
    return result


# Weight interpretation methods for Method B
def sign_based_activation_index(incoming_weights: jnp.ndarray, num_funcs: int = 4) -> jnp.ndarray:
    """Derive activation index from sign of incoming weights.

    Maps: Positive → tanh (0), Negative → relu (2), Mixed → sigmoid (1)
    """
    mean_sign = jnp.mean(jnp.sign(incoming_weights))
    return continuous_to_index(mean_sign, num_funcs)


def magnitude_based_activation_index(incoming_weights: jnp.ndarray, num_funcs: int = 4) -> jnp.ndarray:
    """Derive activation index from magnitude of incoming weights.

    Maps: High magnitude → bounded (tanh/sigmoid), Low → unbounded (relu/identity)
    """
    mean_abs = jnp.mean(jnp.abs(incoming_weights))
    normalized = jnp.tanh(mean_abs * 2 - 1)
    return continuous_to_index(normalized, num_funcs)


def variance_based_activation_index(incoming_weights: jnp.ndarray, num_funcs: int = 4) -> jnp.ndarray:
    """Derive activation index from variance of incoming weights.

    Maps: High variance → nonlinear, Low variance → linear
    """
    variance = jnp.var(incoming_weights)
    normalized = jnp.tanh(variance * 5)
    return continuous_to_index(normalized, num_funcs)


# ============================================================================
# Biologically-Inspired Sparsity Mechanisms (GPU-Safe)
# ============================================================================
# These mechanisms apply region-appropriate sparsity based on hippocampal organization:
# - Level 0 (input-adjacent): Very sparse like DG (~5% active)
# - Level 1 (middle): Moderate like CA3 (~20% active)
# - Level 2+ (output-adjacent): Denser like CA1 (~40% active)


def compute_level_indices(level_offsets_static: Tuple[int, ...], total_positions: int) -> jnp.ndarray:
    """Compute level index for each position in the hierarchical grid.

    GPU-safe: Uses only jnp.where masking with static level boundaries.

    Args:
        level_offsets_static: Tuple of cumulative offsets (0, 4, 20, 84, ...)
        total_positions: Total number of positions in grid

    Returns:
        Array of shape (total_positions,) with level index for each position
    """
    num_levels = len(level_offsets_static) - 1
    position_indices = jnp.arange(total_positions)

    # Start with highest level (last valid level index)
    level_indices = jnp.full(total_positions, num_levels - 1, dtype=jnp.int32)

    # Work backwards from highest to lowest level
    # Each position belongs to the highest level whose offset is <= its index
    for level in range(num_levels - 1, -1, -1):
        level_start = level_offsets_static[level]
        level_end = level_offsets_static[level + 1]
        in_level = (position_indices >= level_start) & (position_indices < level_end)
        level_indices = jnp.where(in_level, level, level_indices)

    return level_indices


def compute_hierarchical_sparsity_thresholds(
    level_indices: jnp.ndarray,
    sparsity_config: Dict[str, float],
) -> jnp.ndarray:
    """Compute per-position sparsity threshold based on hierarchical level.

    Biological inspiration:
    - DG (dentate gyrus): ~2-4% active neurons (very sparse, pattern separation)
    - CA3: ~22% active neurons (moderate sparsity)
    - CA1: ~42% active neurons (relatively dense, pattern completion)

    GPU-safe: Uses only jnp.where masking.

    Args:
        level_indices: Per-position level index, shape (total_positions,)
        sparsity_config: Dict with 'level_0', 'level_1', 'level_2_plus' sparsity targets

    Returns:
        Activation threshold per position, shape (total_positions,)
        Higher threshold = more sparse (fewer nodes pass)
    """
    # Default sparsity targets (matching biological values)
    level_0_sparsity = sparsity_config.get('level_0', 0.05)   # 5% active (DG-like)
    level_1_sparsity = sparsity_config.get('level_1', 0.20)   # 20% active (CA3-like)
    level_2_plus_sparsity = sparsity_config.get('level_2_plus', 0.40)  # 40% active (CA1-like)

    # Convert sparsity targets to activation thresholds
    # Threshold is percentile of activations to zero out (1 - sparsity = fraction zeroed)
    # Higher threshold = more values zeroed = more sparse
    # Note: This is an approximation - actual sparsity depends on activation distribution
    thresh_0 = 1.0 - level_0_sparsity      # 0.95 → zero out bottom 95%
    thresh_1 = 1.0 - level_1_sparsity      # 0.80 → zero out bottom 80%
    thresh_2_plus = 1.0 - level_2_plus_sparsity  # 0.60 → zero out bottom 60%

    # Assign threshold based on level (pure JAX masking)
    level_0_mask = (level_indices == 0)
    level_1_mask = (level_indices == 1)
    # Level 2+ gets the remaining

    thresholds = jnp.where(level_0_mask, thresh_0,
                  jnp.where(level_1_mask, thresh_1, thresh_2_plus))

    return thresholds


def apply_hierarchical_sparsity(
    hidden: jnp.ndarray,
    sparsity_thresholds: jnp.ndarray,
) -> jnp.ndarray:
    """Apply per-node sparsity threshold to hidden activations.

    GPU-safe: Uses only jnp.where masking.

    Args:
        hidden: Hidden activations, shape (num_samples, total_positions)
        sparsity_thresholds: Per-position threshold, shape (total_positions,)
            Higher threshold = more positions zeroed

    Returns:
        Sparse hidden activations with same shape
    """
    # Compute per-sample threshold values
    # For each sample, zero out activations whose absolute value is below
    # the threshold-th percentile of that sample's activations
    # This is an approximation using absolute value ranking

    # Get sorted absolute values per sample
    sorted_abs = jnp.sort(jnp.abs(hidden), axis=-1)  # (num_samples, total_positions)

    # For each position, determine the threshold value based on its sparsity target
    # threshold_indices = (sparsity_thresholds * total_positions).astype(int)
    # Instead of per-position percentile (complex), use a simpler approach:
    # Zero out if abs(activation) < threshold_value where threshold_value
    # is computed per-sample as the k-th percentile

    # Simplified approach: Use sparsity_thresholds as direct activation thresholds
    # Scale by a factor to match typical tanh output range [-1, 1]
    # Threshold of 0.95 means zero out if |activation| < 0.95 (very sparse)
    # Threshold of 0.60 means zero out if |activation| < 0.60 (less sparse)

    # For tanh outputs, typical values are in [-1, 1]
    # Map threshold to actual value threshold: higher threshold = higher bar to pass
    # Actually, let's use percentile-based thresholding per sample

    # Compute threshold value per sample per position using the percentile
    total_positions = hidden.shape[-1]
    # For each position, get its target percentile from sparsity_thresholds
    # Then look up that percentile in the sorted values
    percentile_indices = (sparsity_thresholds * (total_positions - 1)).astype(jnp.int32)

    # Gather the threshold value for each position from sorted_abs
    # This is tricky because each position has a different percentile index
    # Use advanced indexing: for position i, get sorted_abs[:, percentile_indices[i]]
    # Shape: (num_samples, total_positions)
    threshold_values = jnp.take_along_axis(
        sorted_abs,
        percentile_indices[None, :],  # (1, total_positions)
        axis=-1
    )

    # Zero out activations below threshold
    return jnp.where(jnp.abs(hidden) >= threshold_values, hidden, 0.0)


def sparse_activation_wta(hidden: jnp.ndarray, k_percent: float = 0.1) -> jnp.ndarray:
    """Winner-Take-All: Keep only top k% of activations per sample.

    GPU-safe: Uses only jnp.sort and jnp.where.

    Args:
        hidden: Hidden activations, shape (num_samples, total_positions)
        k_percent: Fraction of positions to keep active (0.1 = 10%)

    Returns:
        Sparse activations with only top k% non-zero
    """
    num_positions = hidden.shape[-1]
    k = max(1, int(num_positions * k_percent))  # Static: computed once

    # Get k-th largest absolute value per sample
    sorted_abs = jnp.sort(jnp.abs(hidden), axis=-1)  # Sort ascending
    threshold = sorted_abs[:, -k][:, None]  # k-th from end = k-th largest

    # Zero out below threshold
    return jnp.where(jnp.abs(hidden) >= threshold, hidden, 0.0)


# ============================================================================
# Critical Periods - Meta-Evolutionary Plasticity Modulation
# ============================================================================
# Biological inspiration: Critical periods in neural development where
# plasticity is high early, then gradually closes.
#
# Implementation note: This modifies NEAT mutation rates outside the
# vmapped evaluation. Requires coordination with TensorNEAT's mutation
# infrastructure for full implementation.


def compute_critical_period_plasticity(
    generation: int,
    max_generations: int,
    config: Dict[str, float],
) -> float:
    """Compute plasticity factor based on critical period phase.

    Three phases:
    - Phase 1 (0 - phase1_end): Full plasticity (1.0)
    - Phase 2 (phase1_end - phase2_end): Linear decline to min_plasticity
    - Phase 3 (phase2_end - 1.0): Minimal plasticity (fine-tuning only)

    Args:
        generation: Current generation number
        max_generations: Maximum generations for this run
        config: Dict with 'phase1_end', 'phase2_end', 'min_plasticity'

    Returns:
        Plasticity factor in [min_plasticity, 1.0]
    """
    progress = generation / max(1, max_generations)

    phase1_end = config.get('phase1_end', 0.2)
    phase2_end = config.get('phase2_end', 0.5)
    min_plasticity = config.get('min_plasticity', 0.3)

    if progress < phase1_end:
        # Full plasticity
        return 1.0
    elif progress < phase2_end:
        # Linear decline
        phase2_progress = (progress - phase1_end) / (phase2_end - phase1_end)
        return 1.0 - phase2_progress * (1.0 - min_plasticity)
    else:
        # Minimal plasticity
        return min_plasticity


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
    num_cppn_outputs: int = 1,
) -> jnp.ndarray:
    """Query ALL CPPNs at ALL target positions in ONE vmap call.

    This is the core optimization: instead of 1000 sequential calls
    (one per genome), we perform 1 batched call with double vmap.

    Memory usage: pop_size × num_positions × num_outputs × 4 bytes
    - 1000 × 1024 × 1 × 4 = ~4 MB (negligible)
    - 1000 × 1024 × 2 × 4 = ~8 MB (still negligible)

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coord: Single source coordinate (2,)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        num_cppn_outputs: Number of CPPN outputs (1=weight only, 2=weight+activation)

    Returns:
        If num_cppn_outputs=1: (pop_size, num_positions) array of weights
        If num_cppn_outputs>1: (pop_size, num_positions, num_outputs) array
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
        # vmap over positions - returns (num_positions, num_outputs)
        outputs = jax.vmap(
            lambda x: cppn_forward(state, cppn_tuple, x)
        )(inputs)
        if num_cppn_outputs == 1:
            return outputs.flatten()  # (num_positions,) for backward compat
        else:
            return outputs  # (num_positions, num_outputs) for multi-output

    # Outer vmap: over population
    # Need to vmap over the tuple elements (nodes, conns, conn_attrs, node_attrs)
    all_outputs = jax.vmap(
        query_single_cppn,
        in_axes=((0, 0, 0, 0),)  # vmap over first axis of each tuple element
    )((cppns_transformed[0], cppns_transformed[1],
       cppns_transformed[2], cppns_transformed[3]))

    return all_outputs  # (pop_size, num_positions) or (pop_size, num_positions, num_outputs)


def batch_query_population_multi_source(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    num_cppn_outputs: int = 1,
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
        num_cppn_outputs: Number of CPPN outputs (1=weight only, 2=weight+activation)

    Returns:
        If num_cppn_outputs=1: (pop_size, num_sources, num_positions) array
        If num_cppn_outputs>1: (pop_size, num_sources, num_positions, num_outputs) array
    """
    # vmap over sources
    def query_from_source(source_coord):
        return batch_query_population_positions(
            state, cppns_transformed, source_coord, target_positions,
            outgoing, cppn_forward, num_cppn_outputs
        )

    # Result shape depends on num_cppn_outputs:
    # - num_cppn_outputs=1: (num_sources, pop_size, num_positions)
    # - num_cppn_outputs>1: (num_sources, pop_size, num_positions, num_outputs)
    result = jax.vmap(query_from_source)(source_coords)

    # Transpose to move pop_size first
    if num_cppn_outputs == 1:
        # (num_sources, pop_size, num_positions) -> (pop_size, num_sources, num_positions)
        return jnp.transpose(result, (1, 0, 2))
    else:
        # (num_sources, pop_size, num_positions, num_outputs) -> (pop_size, num_sources, num_positions, num_outputs)
        return jnp.transpose(result, (1, 0, 2, 3))


def batch_query_population_multi_source_chunked(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    pop_chunk_size: int = 100,
    num_cppn_outputs: int = 1,
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
        num_cppn_outputs: Number of CPPN outputs (1=weight only, 2=weight+activation)

    Returns:
        If num_cppn_outputs=1: (pop_size, num_sources, num_positions) array
        If num_cppn_outputs>1: (pop_size, num_sources, num_positions, num_outputs) array
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
            chunk_outputs = batch_query_population_positions(
                state, chunk_cppns, source_coord, target_positions,
                outgoing, cppn_forward, num_cppn_outputs
            )
            chunk_results.append(chunk_outputs)

        # Concatenate chunks for this source
        source_outputs = jnp.concatenate(chunk_results, axis=0)
        results_list.append(source_outputs)

    # Stack sources and transpose to move pop_size first
    result = jnp.stack(results_list, axis=0)
    if num_cppn_outputs == 1:
        # (num_sources, pop_size, num_positions) -> (pop_size, num_sources, num_positions)
        return jnp.transpose(result, (1, 0, 2))
    else:
        # (num_sources, pop_size, num_positions, num_outputs) -> (pop_size, num_sources, num_positions, num_outputs)
        return jnp.transpose(result, (1, 0, 2, 3))


def batch_query_population_self_connections(
    state: Any,
    cppns_transformed: Tuple,
    positions: jnp.ndarray,
    cppn_forward: Any,
    num_cppn_outputs: int = 1,
    pop_chunk_size: int = 0,
) -> jnp.ndarray:
    """Query ALL CPPNs for self-connections at each position.

    Self-connections query CPPN with (x, y, x, y, bias) - same source and target.
    This is used for TRUE neuromodulation to extract per-node receptor densities,
    which are FIXED properties of each hidden node (not connection-dependent).

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        positions: All positions to query (num_positions, 2)
        cppn_forward: JIT-compiled CPPN forward function
        num_cppn_outputs: Number of CPPN outputs
        pop_chunk_size: Population chunk size for memory optimization (0=no chunking)

    Returns:
        (pop_size, num_positions, num_cppn_outputs) array of CPPN outputs at self-connections
    """
    pop_size = cppns_transformed[0].shape[0]
    num_positions = positions.shape[0]

    # Build self-connection inputs: [x, y, x, y, bias=1.0]
    # For self-connections, source == target
    bias = jnp.ones((num_positions, 1))
    inputs = jnp.concatenate([positions, positions, bias], axis=1)  # (num_positions, 5)

    def query_single_cppn(cppn_tuple):
        """Query one CPPN at all positions for self-connections."""
        outputs = jax.vmap(
            lambda x: cppn_forward(state, cppn_tuple, x)
        )(inputs)
        return outputs  # (num_positions, num_cppn_outputs)

    if pop_chunk_size > 0:
        # Chunked processing for memory efficiency
        chunk_results = []
        for chunk_start in range(0, pop_size, pop_chunk_size):
            chunk_end = min(chunk_start + pop_chunk_size, pop_size)
            chunk_cppns = (
                cppns_transformed[0][chunk_start:chunk_end],
                cppns_transformed[1][chunk_start:chunk_end],
                cppns_transformed[2][chunk_start:chunk_end],
                cppns_transformed[3][chunk_start:chunk_end],
            )
            chunk_outputs = jax.vmap(
                query_single_cppn,
                in_axes=((0, 0, 0, 0),)
            )((chunk_cppns[0], chunk_cppns[1], chunk_cppns[2], chunk_cppns[3]))
            chunk_results.append(chunk_outputs)
        return jnp.concatenate(chunk_results, axis=0)
    else:
        # Single vmap over entire population
        all_outputs = jax.vmap(
            query_single_cppn,
            in_axes=((0, 0, 0, 0),)
        )((cppns_transformed[0], cppns_transformed[1],
           cppns_transformed[2], cppns_transformed[3]))
        return all_outputs  # (pop_size, num_positions, num_cppn_outputs)


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

        # Neuromodulation configuration (set via create_config)
        self.neuromod_config: Optional[NeuromodulationConfig] = None
        self._neuromod_gate_values = None      # Per-connection gates for Level 1
        self._neuromod_context_weights = None  # Context projection weights for Level 2
        self._neuromod_context_bias = None     # Context projection bias for Level 2
        self._neuromod_node_types = None       # Node types (0=standard, 1=modulatory) for Level 3
        self._neuromod_W_mod = None            # Modulation weight matrix for Level 3

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
        """Forward pass using HyperNEAT computational model.

        Uses configured activation functions from dynamic_functions config.
        """
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

        # Use configured activation functions
        hidden_act = self._hidden_act_func if self._hidden_act_func is not None else jnp.tanh
        output_act = self._output_act_func if self._output_act_func is not None else jax.nn.sigmoid

        vals = jnp.zeros(num_nodes)
        vals = vals.at[:num_inputs].set(inputs)

        for iteration in range(self.activate_time):
            new_vals = jnp.zeros(num_nodes)
            new_vals = new_vals.at[:num_inputs].set(inputs)

            aggregated = jnp.zeros(num_nodes)
            aggregated = aggregated.at[valid_to].add(vals[valid_from] * valid_weights)

            if output_start_idx > num_inputs:
                hidden_vals = hidden_act(aggregated[num_inputs:output_start_idx])
                new_vals = new_vals.at[num_inputs:output_start_idx].set(hidden_vals)

            output_vals = aggregated[output_start_idx:]
            new_vals = new_vals.at[output_start_idx:].set(output_vals)

            vals = new_vals

        raw_outputs = vals[-num_outputs:]
        return output_act(raw_outputs)

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

        # =====================================================================
        # Dynamic Functions Configuration
        # =====================================================================
        dynamic_funcs = hmr_config.get('dynamic_functions', {})

        # Mode selection:
        # - 'disabled': Original hardcoded tanh/sigmoid (baseline)
        # - 'global': All hidden nodes use same configurable activation
        # - 'cppn_output': CPPN outputs activation index per node
        # - 'weight_interpretation': Derive activation from weight patterns
        # - 'random_fixed': Random activation per node, fixed at initialization (negative control)
        # - 'random_generation': Random activation per node, re-randomized each generation (negative control)
        # - 'modular': Granular combination with three orthogonal layers
        self.dynamic_functions_mode = dynamic_funcs.get('mode', 'disabled')

        # For random modes: store seed for reproducibility
        self._random_mode_seed = dynamic_funcs.get('random_seed', None)
        self._random_fixed_indices = None  # Populated during first generation for random_fixed

        # For 'global' mode: which activation to use
        # Overrides self.hidden_activation if mode='global'
        self.df_hidden_activation = dynamic_funcs.get('hidden_activation', self.hidden_activation)
        self.df_output_activation = dynamic_funcs.get('output_activation', self.output_activation)

        # For 'cppn_output' mode: number of activation functions available
        self.df_num_activations = dynamic_funcs.get('num_activations', 4)

        # For 'weight_interpretation' mode: interpretation method
        # Options: 'sign', 'magnitude', 'magnitude_bio', 'variance',
        #          'hierarchical_sparsity', 'sparsity_threshold', 'stp_inspired', 'combined'
        self.df_interpretation = dynamic_funcs.get('interpretation', 'sign')

        # Sparsity configuration for biologically-inspired mechanisms
        sparsity_config = dynamic_funcs.get('sparsity', {})
        self.df_sparsity_config = {
            'level_0': sparsity_config.get('level_0', 0.05),      # DG-like: 5% active
            'level_1': sparsity_config.get('level_1', 0.20),      # CA3-like: 20% active
            'level_2_plus': sparsity_config.get('level_2_plus', 0.40),  # CA1-like: 40% active
            'wta_k_percent': sparsity_config.get('wta_k_percent', 0.10),  # WTA: 10% active
        }

        # Pre-computed sparsity structures (populated lazily in _evaluate_fitness)
        self._level_indices = None
        self._sparsity_thresholds = None

        # Critical Periods configuration (meta-evolutionary, modifies mutation rates)
        # NOTE: Full implementation requires TensorNEAT modifications.
        # Current implementation: stores config for future use.
        critical_periods = dynamic_funcs.get('critical_periods', {})
        self.df_critical_periods_enabled = critical_periods.get('enabled', False)
        self.df_critical_periods_config = {
            'phase1_end': critical_periods.get('phase1_end', 0.2),      # 0-20%: Full plasticity
            'phase2_end': critical_periods.get('phase2_end', 0.5),      # 20-50%: Declining
            'min_plasticity': critical_periods.get('min_plasticity', 0.3),  # Minimum plasticity
        }

        # =====================================================================
        # MODULAR MODE: Granular combination with three orthogonal layers
        # =====================================================================
        # Solves the problem of the broken 'combined' mode (0.81 fitness on XOR)
        # by allowing independent configuration of:
        #   Layer 1: Activation selection (magnitude_bio, sign, variance, cppn_output)
        #   Layer 2: Sparsity (none, hierarchical, threshold, wta)
        #   Layer 3: Weight scaling (none, stp_inspired)
        modular_config = dynamic_funcs.get('modular', {})
        self.df_modular_config = {
            # Layer 1: Activation selection method (required)
            'activation_method': modular_config.get('activation_method', 'magnitude_bio'),
            # Layer 2: Sparsity method (optional, default='none')
            'sparsity_method': modular_config.get('sparsity_method', 'none'),
            # Layer 3: Weight scaling method (optional, default='none')
            'scaling_method': modular_config.get('scaling_method', 'none'),
        }
        # Modular sparsity sub-config
        modular_sparsity = modular_config.get('sparsity', {})
        self.df_modular_sparsity = {
            'level_0': modular_sparsity.get('level_0', 0.05),
            'level_1': modular_sparsity.get('level_1', 0.20),
            'level_2_plus': modular_sparsity.get('level_2_plus', 0.40),
            'wta_k_percent': modular_sparsity.get('wta_k_percent', 0.10),
        }

        # Internal state flags for modular mode (set during _evaluate_fitness)
        self._modular_use_hierarchical_sparsity = False
        self._modular_use_node_sparsity = False
        self._modular_use_wta_sparsity = False
        self._modular_use_stp_scaling = False

        # Resolve hidden activation function for 'global' and 'disabled' modes
        if self.dynamic_functions_mode in ('disabled', 'global'):
            act_name = self.df_hidden_activation if self.dynamic_functions_mode == 'global' else 'tanh'
            self._hidden_act_func = ACTIVATION_FUNCTIONS.get(act_name, jnp.tanh)
            self._output_act_func = ACTIVATION_FUNCTIONS.get(self.df_output_activation, jax.nn.sigmoid)
        else:
            # For cppn_output, weight_interpretation, and random modes, we use grouped_activation_forward
            self._hidden_act_func = None
            self._output_act_func = ACTIVATION_FUNCTIONS.get(self.df_output_activation, jax.nn.sigmoid)

        # Log configuration
        if self.dynamic_functions_mode != 'disabled':
            print(f"[Dynamic Functions] Mode: {self.dynamic_functions_mode}")
            if self.dynamic_functions_mode == 'global':
                print(f"[Dynamic Functions] Hidden activation: {self.df_hidden_activation}")
            elif self.dynamic_functions_mode == 'cppn_output':
                print(f"[Dynamic Functions] Num activations: {self.df_num_activations}")
            elif self.dynamic_functions_mode == 'weight_interpretation':
                print(f"[Dynamic Functions] Interpretation: {self.df_interpretation}")
                if self.df_interpretation in ('hierarchical_sparsity', 'sparsity_threshold'):
                    print(f"[Dynamic Functions] Sparsity config: {self.df_sparsity_config}")
            elif self.dynamic_functions_mode == 'modular':
                print(f"[Dynamic Functions] Modular config:")
                print(f"  - Activation method: {self.df_modular_config['activation_method']}")
                print(f"  - Sparsity method: {self.df_modular_config['sparsity_method']}")
                print(f"  - Scaling method: {self.df_modular_config['scaling_method']}")
                if self.df_modular_config['sparsity_method'] != 'none':
                    print(f"  - Sparsity params: {self.df_modular_sparsity}")
            elif self.dynamic_functions_mode in ('random_fixed', 'random_generation'):
                pass  # Continue to next condition

        # Log critical periods if enabled (separate from dynamic_functions mode)
        if self.df_critical_periods_enabled:
            print(f"[Critical Periods] Enabled with config: {self.df_critical_periods_config}")
            print(f"[Critical Periods] NOTE: Full implementation requires TensorNEAT mutation scaling")

        # Continue logging for random modes
        if self.dynamic_functions_mode in ('random_fixed', 'random_generation'):
            print(f"[Dynamic Functions] Num activations: {self.df_num_activations}")
            print(f"[Dynamic Functions] Random seed: {self._random_mode_seed or 'from state'}")

        # =====================================================================
        # NEUROMODULATION Configuration
        # =====================================================================
        # Three progressive levels of neuromodulation:
        #   Level 1: Static Gating - CPPN outputs per-connection gates
        #   Level 2: Context-Dependent Gating (XdG-style) - context modulates gates
        #   Level 3: Modulatory Neurons (Soltoggio-style) - two neuron types
        neuromod_section = hmr_config.get('neuromodulation', {})
        neuromod_mode = neuromod_section.get('mode', 'disabled')

        # Get custom config overrides
        custom_neuromod = {
            k: v for k, v in neuromod_section.items()
            if k not in ('mode',)  # mode is handled separately
        }

        # Parse configuration using presets and custom overrides
        self.neuromod_config = get_neuromodulation_config(neuromod_mode, custom_neuromod if custom_neuromod else None)

        # Log neuromodulation configuration
        if self.neuromod_config.enabled:
            print(f"[Neuromodulation] Mode: {neuromod_mode}")
            if self.neuromod_config.static_gating:
                print(f"[Neuromodulation] Level 1 - Static Gating:")
                print(f"  - Gate scaling: {self.neuromod_config.gate_scaling}")
                print(f"  - Gate threshold: {self.neuromod_config.gate_threshold}")
            if self.neuromod_config.context_gating:
                print(f"[Neuromodulation] Level 2 - Context-Dependent Gating:")
                print(f"  - Context dim: {self.neuromod_config.context_dim}")
                print(f"  - Context influence: {self.neuromod_config.context_influence}")
                print(f"  - Context source: {self.neuromod_config.context_source}")
            if self.neuromod_config.modulatory_neurons:
                print(f"[Neuromodulation] Level 3 - Modulatory Neurons:")
                print(f"  - Mod neuron ratio: {self.neuromod_config.mod_neuron_ratio}")
                print(f"  - Mod connection type: {self.neuromod_config.mod_connection_type}")
                print(f"  - Mod decay: {self.neuromod_config.mod_decay}")
            if self.neuromod_config.true_neuromodulation:
                print(f"[Neuromodulation] Level 4 - TRUE Neuromodulation:")
                print(f"  - Num NT types: {self.neuromod_config.num_nt_types}")
                print(f"  - Modulation strength: {self.neuromod_config.modulation_strength}")
                print(f"  - CPPN receptor base index: {get_cppn_receptor_base_idx(self.dynamic_functions_mode, self.neuromod_config)}")

        # Pre-compute quadtree structure
        self._quadtree = get_quadtree_structure(self.max_depth)

        # Determine CPPN output count based on dynamic_functions mode AND neuromodulation
        # Mode='cppn_output' needs 2 outputs: [weight, activation_raw]
        # Neuromodulation adds additional outputs for gates, mod_factor, node_type
        self.cppn_num_outputs = get_cppn_num_outputs(self.dynamic_functions_mode, self.neuromod_config)

        # Build NEAT config
        flat_params = {
            'genome': {
                'num_inputs': 5,
                'num_outputs': self.cppn_num_outputs,
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
                'conn_add_prob': algo_params.get('neat_mutation', {}).get('conn_add_prob', 0.5),
                'conn_delete_prob': 0.5,
                'node_add_prob': algo_params.get('neat_mutation', {}).get('node_add_prob', 0.2),
                'node_delete_prob': 0.2,
            },
            'species': {
                'compatibility_threshold': algo_params.get('neat_species', {}).get('compatibility_threshold', 3.0),
                'max_stagnation': algo_params.get('neat_species', {}).get('max_stagnation', 20),
                'species_elitism': algo_params.get('neat_species', {}).get('species_elitism', 15),
            },
            'selection': {
                'genome_elitism': algo_params.get('neat_selection', {}).get('genome_elitism', 15),
                'survival_threshold': algo_params.get('neat_selection', {}).get('survival_threshold', 0.2),
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
        num_cppn_outputs = self.cppn_num_outputs  # 1 for disabled/global, 2 for cppn_output
        if self.population_chunk_size > 0:
            query_func = lambda state, cppns, sources, targets, outgoing, fwd: \
                batch_query_population_multi_source_chunked(
                    state, cppns, sources, targets, outgoing, fwd,
                    pop_chunk_size=self.population_chunk_size,
                    num_cppn_outputs=num_cppn_outputs
                )
        else:
            query_func = lambda state, cppns, sources, targets, outgoing, fwd: \
                batch_query_population_multi_source(
                    state, cppns, sources, targets, outgoing, fwd,
                    num_cppn_outputs=num_cppn_outputs
                )

        # Query CPPN - shape depends on num_cppn_outputs:
        # - 1 output: (pop_size, num_inputs, total_positions)
        # - 2 outputs: (pop_size, num_inputs, total_positions, 2) where [:,:,:,0]=weight, [:,:,:,1]=activation
        input_all_cppn = query_func(
            state, cppns_transformed, input_coords,
            all_positions, True, self._jitted_cppn_forward
        )

        # All → output (incoming)
        output_all_cppn = query_func(
            state, cppns_transformed, output_coords,
            all_positions, False, self._jitted_cppn_forward
        )

        # Extract weights (and optionally activation outputs) from CPPN results
        if num_cppn_outputs == 1:
            # Single output: weights only - shapes: (pop_size, num_inputs/outputs, total_positions)
            input_all_weights = input_all_cppn
            output_all_weights = output_all_cppn

            # Check if we're in weight_interpretation mode
            if self.dynamic_functions_mode == 'weight_interpretation':
                # Derive activation indices from weight patterns (no extra CPPN output needed)
                # Use mean across all inputs for each position: (pop_size, num_inputs, total_positions) -> (pop_size, total_positions)
                mean_weights = jnp.mean(input_all_weights, axis=1)

                # Select interpretation method
                if self.df_interpretation == 'sign':
                    # Sign-based: positive weights → one set, negative → another
                    mean_sign = jnp.mean(jnp.sign(input_all_weights), axis=1)
                    hidden_activation_indices = continuous_to_index(mean_sign, self.df_num_activations)
                elif self.df_interpretation == 'magnitude':
                    # Magnitude-based (computational): high magnitude → unbounded, low → bounded
                    # NOTE: This is NOT biologically accurate - see docs for correction
                    mean_abs = jnp.mean(jnp.abs(input_all_weights), axis=1)
                    normalized = jnp.tanh(mean_abs * 2 - 1)  # Center around typical values
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)
                elif self.df_interpretation == 'magnitude_bio':
                    # Magnitude-based (biological): high magnitude → bounded (saturation), low → unbounded
                    # Matches divisive normalization in biology: strong input → bounded response
                    # See Carandini & Heeger (2012) - Normalization as a canonical neural computation
                    mean_abs = jnp.mean(jnp.abs(input_all_weights), axis=1)
                    # INVERTED: negate to get high magnitude → low index (tanh) → bounded
                    normalized = -jnp.tanh(mean_abs * 2 - 1)  # Negate to invert mapping
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)
                elif self.df_interpretation == 'variance':
                    # Variance-based: high variance → nonlinear, low variance → linear
                    weight_var = jnp.var(input_all_weights, axis=1)
                    normalized = jnp.tanh(weight_var * 5)  # Scale variance
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)

                # ================================================================
                # NEW: Biologically-Inspired Sparsity Mechanisms (GPU-Safe)
                # ================================================================

                elif self.df_interpretation == 'hierarchical_sparsity':
                    # Region-appropriate sparsity based on hierarchical level
                    # Level 0 (input-adjacent): Very sparse like DG (~5% active)
                    # Level 1 (middle): Moderate like CA3 (~20% active)
                    # Level 2+ (output-adjacent): Denser like CA1 (~40% active)
                    # Uses magnitude_bio for activation selection + level-based sparsity

                    # Activation selection: Use magnitude_bio (best performing)
                    mean_abs = jnp.mean(jnp.abs(input_all_weights), axis=1)
                    normalized = -jnp.tanh(mean_abs * 2 - 1)  # Biological mapping
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)

                    # Pre-compute level indices and sparsity thresholds (lazy init)
                    if self._level_indices is None:
                        self._level_indices = compute_level_indices(
                            h_grid.level_offsets_static, h_grid.total_positions
                        )
                        self._sparsity_thresholds = compute_hierarchical_sparsity_thresholds(
                            self._level_indices, self.df_sparsity_config
                        )

                elif self.df_interpretation == 'sparsity_threshold':
                    # Per-node sparsity threshold derived from incoming weight magnitude
                    # Low weight magnitude → high threshold (sparse)
                    # High weight magnitude → low threshold (dense)
                    mean_abs = jnp.mean(jnp.abs(input_all_weights), axis=1)

                    # Activation selection: magnitude_bio
                    normalized = -jnp.tanh(mean_abs * 2 - 1)
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)

                    # Compute per-node thresholds: low magnitude → sparse, high → dense
                    # Range: [0.5, 0.95] for threshold (zeroing out 50-95% of activations)
                    self._node_sparsity_thresholds = jax.nn.sigmoid(-mean_abs * 2 + 1) * 0.45 + 0.5

                elif self.df_interpretation == 'stp_inspired':
                    # STP-inspired (static approximation): scale weights by depression factor
                    # High weight magnitude → high U (release probability) → more depression
                    # This is computed once per evaluation, not per-timestep (static approx)
                    mean_abs = jnp.mean(jnp.abs(input_all_weights), axis=1)

                    # Activation selection: magnitude_bio
                    normalized = -jnp.tanh(mean_abs * 2 - 1)
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)

                    # U (release probability): high magnitude → high U → more depression
                    # Range: [0.1, 0.8]
                    U = jax.nn.sigmoid(mean_abs * 2) * 0.7 + 0.1  # (pop_size, total_positions)

                    # Depression factor: 1 - U * decay (effective weight scaling)
                    # Range: [0.6, 0.95] - high U → low effective weight
                    self._stp_depression_factor = 1.0 - U * 0.5

                elif self.df_interpretation == 'combined':
                    # Combined: fuse magnitude + variance + sign ratio
                    mean_abs = jnp.mean(jnp.abs(input_all_weights), axis=1)
                    weight_var = jnp.var(input_all_weights, axis=1)
                    pos_ratio = jnp.mean((input_all_weights > 0).astype(jnp.float32), axis=1)

                    # Weighted combination (coefficients tuned empirically)
                    combined = (0.5 * jnp.tanh(mean_abs * 2) +
                                0.3 * jnp.tanh(weight_var * 5) +
                                0.2 * (pos_ratio - 0.5) * 2)

                    hidden_activation_indices = continuous_to_index(combined, self.df_num_activations)

                else:
                    # Default fallback to sign
                    mean_sign = jnp.mean(jnp.sign(input_all_weights), axis=1)
                    hidden_activation_indices = continuous_to_index(mean_sign, self.df_num_activations)

            elif self.dynamic_functions_mode == 'random_fixed':
                # NEGATIVE CONTROL: Random activation assignment, fixed at first generation
                # Tests: "Is CPPN selection better than random assignment?"
                # Shape: (pop_size, total_positions)
                pop_size = input_all_weights.shape[0]
                total_positions = input_all_weights.shape[2]

                if self._random_fixed_indices is None:
                    # First generation: generate and store random indices
                    if self._random_mode_seed is not None:
                        key = jax.random.PRNGKey(self._random_mode_seed)
                    else:
                        key = state.randkey
                    self._random_fixed_indices = jax.random.randint(
                        key, shape=(pop_size, total_positions),
                        minval=0, maxval=self.df_num_activations
                    )
                hidden_activation_indices = self._random_fixed_indices

            elif self.dynamic_functions_mode == 'random_generation':
                # NEGATIVE CONTROL: Random activation assignment, re-randomized each generation
                # Tests: "Does consistent activation assignment matter?"
                # Shape: (pop_size, total_positions)
                pop_size = input_all_weights.shape[0]
                total_positions = input_all_weights.shape[2]

                # Use different key each generation (split from state.randkey)
                key = jax.random.fold_in(state.randkey, 9999)  # Use arbitrary fold value
                hidden_activation_indices = jax.random.randint(
                    key, shape=(pop_size, total_positions),
                    minval=0, maxval=self.df_num_activations
                )

            elif self.dynamic_functions_mode == 'modular':
                # ================================================================
                # MODULAR MODE: Three orthogonal layers of configuration
                # ================================================================
                # Replaces the broken 'combined' mode with granular control
                # Layer 1: Activation selection (required)
                # Layer 2: Sparsity (optional)
                # Layer 3: Weight scaling (optional)

                modular_cfg = self.df_modular_config
                activation_method = modular_cfg['activation_method']
                sparsity_method = modular_cfg['sparsity_method']
                scaling_method = modular_cfg['scaling_method']

                # Reset modular flags
                self._modular_use_hierarchical_sparsity = False
                self._modular_use_node_sparsity = False
                self._modular_use_wta_sparsity = False
                self._modular_use_stp_scaling = False

                # LAYER 1: Activation Selection (always applied)
                mean_abs = jnp.mean(jnp.abs(input_all_weights), axis=1)

                if activation_method == 'none':
                    # No dynamic activation: all nodes use tanh (index 0)
                    # This allows testing sparsity/scaling mechanisms in isolation
                    # Shape must be (pop_size, total_positions) to match other modes
                    hidden_activation_indices = jnp.zeros(mean_abs.shape, dtype=jnp.int32)
                elif activation_method == 'magnitude_bio':
                    # Biological: high magnitude → bounded (like divisive normalization)
                    normalized = -jnp.tanh(mean_abs * 2 - 1)
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)
                elif activation_method == 'magnitude':
                    # Computational: high magnitude → unbounded
                    normalized = jnp.tanh(mean_abs * 2 - 1)
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)
                elif activation_method == 'sign':
                    # Sign-based: positive weights → one set, negative → another
                    mean_sign = jnp.mean(jnp.sign(input_all_weights), axis=1)
                    hidden_activation_indices = continuous_to_index(mean_sign, self.df_num_activations)
                elif activation_method == 'variance':
                    # Variance-based: high variance → nonlinear
                    weight_var = jnp.var(input_all_weights, axis=1)
                    normalized = jnp.tanh(weight_var * 5)
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)
                else:
                    # Default fallback to magnitude_bio
                    normalized = -jnp.tanh(mean_abs * 2 - 1)
                    hidden_activation_indices = continuous_to_index(normalized, self.df_num_activations)

                # LAYER 2: Sparsity Selection (independent of activation)
                if sparsity_method == 'hierarchical':
                    # Pre-compute level indices and sparsity thresholds (lazy init)
                    if self._level_indices is None:
                        self._level_indices = compute_level_indices(
                            h_grid.level_offsets_static, h_grid.total_positions
                        )
                        self._sparsity_thresholds = compute_hierarchical_sparsity_thresholds(
                            self._level_indices, self.df_modular_sparsity
                        )
                    self._modular_use_hierarchical_sparsity = True

                elif sparsity_method == 'threshold':
                    # Per-node threshold from weight magnitude
                    # Low magnitude → high threshold (sparse), high → low threshold (dense)
                    self._node_sparsity_thresholds = jax.nn.sigmoid(-mean_abs * 2 + 1) * 0.45 + 0.5
                    self._modular_use_node_sparsity = True

                elif sparsity_method == 'wta':
                    # Winner-Take-All: keep only top k%
                    self._modular_use_wta_sparsity = True

                # else: sparsity_method == 'none' - no flags set

                # LAYER 3: Weight Scaling (independent of activation and sparsity)
                if scaling_method == 'stp_inspired':
                    # STP-inspired: high magnitude → high U → more depression
                    U = jax.nn.sigmoid(mean_abs * 2) * 0.7 + 0.1  # [0.1, 0.8]
                    self._stp_depression_factor = 1.0 - U * 0.5  # [0.6, 0.95]
                    self._modular_use_stp_scaling = True
                # else: scaling_method == 'none' - no scaling

            else:
                # disabled or global mode: use fixed activation
                hidden_activation_indices = None
        else:
            # Multi-output (cppn_output mode): split into weights and activation_raw
            # Shapes: (pop_size, num_inputs/outputs, total_positions, 2)
            input_all_weights = input_all_cppn[:, :, :, 0]    # (pop_size, num_inputs, total_positions)
            output_all_weights = output_all_cppn[:, :, :, 0]  # (pop_size, num_outputs, total_positions)

            # Extract activation_raw from input connections (average across all inputs per position)
            # Shape: (pop_size, num_inputs, total_positions) -> mean over inputs -> (pop_size, total_positions)
            input_activation_raw = input_all_cppn[:, :, :, 1]
            mean_activation_raw = jnp.mean(input_activation_raw, axis=1)  # (pop_size, total_positions)

            # Convert continuous activation output to discrete function indices
            # Using continuous_to_index: maps [-inf, inf] -> [0, num_activations)
            hidden_activation_indices = continuous_to_index(mean_activation_raw, self.df_num_activations)
            # Shape: (pop_size, total_positions) of int32 indices

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
        # STEP 4.5: Extract and process gate values (NEUROMODULATION)
        # =====================================================================
        # If neuromodulation is enabled, extract gate values from CPPN outputs

        G1 = None  # Input→Hidden gates (or logits if context_gating)
        G2 = None  # Hidden→Output gates (or logits if context_gating)
        G1_logit = None  # Raw logits for context gating
        G2_logit = None

        if self.neuromod_config.enabled and (self.neuromod_config.static_gating or self.neuromod_config.context_gating):
            # Determine gate output index based on current configuration
            # If cppn_output mode: gates at index 2 (0=weight, 1=activation, 2=gate)
            # Otherwise: gates at index 1 (0=weight, 1=gate)
            gate_idx = 2 if self.dynamic_functions_mode == 'cppn_output' else 1

            if num_cppn_outputs > gate_idx:
                # Extract gate_raw values from CPPN output
                # Shapes: (pop_size, num_inputs/outputs, total_positions)
                input_gate_raw = input_all_cppn[:, :, :, gate_idx]
                output_gate_raw = output_all_cppn[:, :, :, gate_idx]

                if self.neuromod_config.context_gating:
                    # For context gating, store raw logits (pre-sigmoid)
                    # Context will modulate these during evaluation
                    G1_logit = jnp.where(active_mask_broadcast, input_gate_raw, 0.0)
                    G2_logit_masked = jnp.where(active_mask_broadcast, output_gate_raw, 0.0)
                    G2_logit = jnp.transpose(G2_logit_masked, (0, 2, 1))
                    self._neuromod_gate_values = {'G1_logit': G1_logit, 'G2_logit': G2_logit}
                else:
                    # For static gating, apply gate scaling to convert to [0, 1] range
                    G1_raw = apply_gate_scaling(
                        input_gate_raw,
                        self.neuromod_config.gate_scaling,
                        self.neuromod_config.gate_threshold
                    )
                    G2_raw = apply_gate_scaling(
                        output_gate_raw,
                        self.neuromod_config.gate_scaling,
                        self.neuromod_config.gate_threshold
                    )

                    # Apply mask (gates only apply to active positions)
                    G1 = jnp.where(active_mask_broadcast, G1_raw, 0.0)
                    G2_masked = jnp.where(active_mask_broadcast, G2_raw, 0.0)
                    # Transpose G2 to match W2 shape: (pop, total_positions, num_outputs)
                    G2 = jnp.transpose(G2_masked, (0, 2, 1))

                    # Store for potential visualization/analysis
                    self._neuromod_gate_values = {'G1': G1, 'G2': G2}
            else:
                # CPPN doesn't have gate outputs - this shouldn't happen if config is correct
                print(f"[Neuromodulation] Warning: CPPN has {num_cppn_outputs} outputs, "
                      f"but gate_idx={gate_idx}. Disabling gating for this run.")

        # =====================================================================
        # STEP 4.6: Extract modulatory neuron data (NEUROMODULATION Level 3)
        # =====================================================================
        # If modulatory_neurons is enabled, extract node_type and mod_factor from CPPN
        # These determine which hidden nodes are modulatory and their influence strength

        node_types = None  # (pop_size, total_positions) - 0=standard, 1=modulatory
        W_mod = None       # (pop_size, total_positions, total_positions) - modulation weights

        if self.neuromod_config.enabled and self.neuromod_config.modulatory_neurons:
            # Determine indices for mod_factor and node_type in CPPN output
            # Layout depends on what other outputs are enabled:
            # Base: 0=weight
            # If cppn_output: 1=activation
            # If gating: next=gate
            # If modulatory: next=mod_factor, next+1=node_type
            base_idx = 1 if self.dynamic_functions_mode == 'cppn_output' else 0
            if self.neuromod_config.static_gating or self.neuromod_config.context_gating:
                base_idx += 1
            mod_factor_idx = base_idx + 1
            node_type_idx = base_idx + 2

            if num_cppn_outputs > node_type_idx:
                # Extract node_type_raw for each hidden position across all networks
                # Use input_all_cppn averaged across input positions as proxy for node type
                # Shape: (pop_size, num_inputs, total_positions, num_outputs)
                # Average across input dimension to get (pop_size, total_positions)
                node_type_raw = jnp.mean(input_all_cppn[:, :, :, node_type_idx], axis=1)

                # Compute node types per network
                # (pop_size, total_positions) with 0=standard, 1=modulatory
                def compute_node_types_single(raw):
                    return assign_node_types(
                        num_hidden=raw.shape[0],
                        node_type_raw=raw,
                        mod_ratio=self.neuromod_config.mod_neuron_ratio,
                    )

                node_types = jax.vmap(compute_node_types_single)(node_type_raw)

                # Extract modulation factors for hidden→hidden modulation
                # mod_factor determines strength of modulatory influence
                # Shape: (pop_size, num_inputs, total_positions) → use for W_mod rows
                mod_factor_raw = jnp.mean(input_all_cppn[:, :, :, mod_factor_idx], axis=1)

                # Build modulation weight matrix: W_mod[i,j] = how much node i modulates node j
                # Use outer product of mod_factor with itself as simple proxy
                # In full implementation, this would come from CPPN query with hidden coords
                # Shape: (pop_size, total_positions, total_positions)
                W_mod = jnp.einsum('pi,pj->pij', mod_factor_raw, jnp.ones_like(mod_factor_raw))
                W_mod = jax.nn.tanh(W_mod)  # Bound to [-1, 1]

                # Mask by node types: only modulatory→standard connections active
                # Create mask: (pop_size, total_positions, total_positions)
                mod_mask = (node_types[:, :, None] == 1).astype(jnp.float32)  # From modulatory
                std_mask = (node_types[:, None, :] == 0).astype(jnp.float32)  # To standard
                W_mod = W_mod * mod_mask * std_mask

                # Store for visualization/analysis
                self._neuromod_modulatory = {
                    'node_types': node_types,
                    'W_mod': W_mod,
                    'mod_ratio_actual': jnp.mean(node_types.astype(jnp.float32)),
                }
            else:
                print(f"[Neuromodulation] Warning: CPPN has {num_cppn_outputs} outputs, "
                      f"but need {node_type_idx + 1} for modulatory neurons. Disabling.")

        # =====================================================================
        # STEP 4.7: Extract receptor densities (TRUE NEUROMODULATION Level 4)
        # =====================================================================
        # If true_neuromodulation is enabled, extract per-node receptor densities
        # from CPPN self-connections. These determine how each hidden node responds
        # to different neurotransmitter types.
        #
        # KEY INSIGHT: Query CPPN at (x,y,x,y,bias) for per-NODE properties.
        # This gives spatially-organized receptor densities based on position.
        #
        # Receptor densities are FIXED after evolution - only the NT vector changes.

        receptor_densities = None  # (pop_size, total_positions, num_nt_types)
        base_gains = None          # (pop_size, total_positions) - base activation gains

        if self.neuromod_config.enabled and self.neuromod_config.true_neuromodulation:
            num_nt_types = self.neuromod_config.num_nt_types
            use_option_a = self.neuromod_config.receptor_from_weight

            # Query CPPN at self-connections (x,y,x,y,bias) for per-node properties
            # This extracts spatially-organized receptor densities
            self_conn_outputs = batch_query_population_self_connections(
                state, cppns_transformed, all_positions,
                self._jitted_cppn_forward, num_cppn_outputs,
                pop_chunk_size=self.population_chunk_size
            )
            # Shape: (pop_size, total_positions, num_cppn_outputs)

            if use_option_a:
                # =====================================================================
                # OPTION A: Derive receptor densities from weight output
                # =====================================================================
                # Use single CPPN output (weight), derive receptor via specified method
                weight_values = self_conn_outputs[:, :, 0]  # (pop_size, total_positions)

                # Derive receptor densities from weight using specified method
                receptor_densities = derive_receptor_from_weight(
                    weight_values,
                    method=self.neuromod_config.receptor_derivation,
                    num_nt_types=num_nt_types,
                )
                # Shape: (pop_size, total_positions, num_nt_types)

                # Convert to [-1, 1] range to match Option B (enables both excitatory and inhibitory)
                receptor_densities = receptor_densities * 2.0 - 1.0

                # Base gains derived from weight magnitude
                base_gains = jnp.abs(weight_values) * 0.5 + 0.5
                # Shape: (pop_size, total_positions) - base gain per neuron, [0.5, 1.0]

                if self.verbose:
                    print(f"[TRUE Neuromodulation - Option A] Derived receptor densities "
                          f"from weight using '{self.neuromod_config.receptor_derivation}': "
                          f"{receptor_densities.shape}, base_gains: {base_gains.shape}")
            else:
                # =====================================================================
                # OPTION B: Separate CPPN outputs for receptor densities
                # =====================================================================
                # Get starting index for receptor outputs in CPPN
                receptor_base_idx = get_cppn_receptor_base_idx(
                    self.dynamic_functions_mode, self.neuromod_config
                )

                if num_cppn_outputs >= receptor_base_idx + num_nt_types:
                    # Extract receptor densities for each NT type
                    # receptor_densities[i,j,k] = sensitivity of neuron j in network i to NT type k
                    receptor_densities = self_conn_outputs[:, :, receptor_base_idx:receptor_base_idx + num_nt_types]
                    # Shape: (pop_size, total_positions, num_nt_types)

                    # STABILITY FIX 1: Clip extreme values BEFORE tanh
                    # Raw CPPN outputs can be ±1000, causing numerical issues
                    receptor_densities = jnp.clip(receptor_densities, -10.0, 10.0)

                    # Apply tanh to bound receptor densities to [-1, 1]
                    # This creates structured receptor profiles:
                    # - Positive values: excitatory response to NT
                    # - Negative values: inhibitory response to NT
                    receptor_densities = jnp.tanh(receptor_densities)

                    # STABILITY FIX 2: Replace any NaN with 0 (neutral modulation)
                    receptor_densities = jnp.where(
                        jnp.isnan(receptor_densities), 0.0, receptor_densities
                    )

                    # Base gains derived from weight output (index 0) at self-connections
                    # This creates position-dependent base gains
                    base_weight = jnp.clip(self_conn_outputs[:, :, 0], -10.0, 10.0)  # STABILITY FIX
                    base_gains = jnp.abs(base_weight) * 0.5 + 0.5
                    # Shape: (pop_size, total_positions) - base gain per neuron, [0.5, 1.0]
                    base_gains = jnp.where(jnp.isnan(base_gains), 0.75, base_gains)  # STABILITY FIX

                    if self.verbose:
                        print(f"[TRUE Neuromodulation - Option B] Extracted receptor densities: "
                              f"{receptor_densities.shape}, base_gains: {base_gains.shape}")
                else:
                    print(f"[Neuromodulation] Warning: CPPN has {num_cppn_outputs} outputs, "
                          f"but need {receptor_base_idx + num_nt_types} for Option B. "
                          f"Consider using Option A (receptor_from_weight=True).")
                    receptor_densities = None
                    base_gains = None

            # Store for visualization/analysis (only if successfully extracted)
            if receptor_densities is not None:
                self._neuromod_true = {
                    'receptor_densities': receptor_densities,
                    'base_gains': base_gains,
                    'num_nt_types': num_nt_types,
                    'modulation_strength': self.neuromod_config.modulation_strength,
                    'option': 'A' if use_option_a else 'B',
                    'derivation_method': self.neuromod_config.receptor_derivation if use_option_a else None,
                }

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

        # Capture activation functions in closure for eval_single_network
        # This allows mode-dependent activation selection without passing extra args
        hidden_act = self._hidden_act_func if self._hidden_act_func is not None else jnp.tanh
        output_act = self._output_act_func if self._output_act_func is not None else jax.nn.sigmoid
        num_activations = self.df_num_activations  # for grouped_activation_forward
        use_per_node_activation = hidden_activation_indices is not None

        def eval_single_network(W1_single, W2_single, inputs, targets):
            """Evaluate single two-layer network with all-level hidden nodes.

            Uses configured activation functions via closure:
            - hidden_act: From dynamic_functions config (default: tanh)
            - output_act: From dynamic_functions config (default: sigmoid)
            """
            hidden = hidden_act(safe_matmul(inputs, W1_single))  # (num_cases, total_positions)
            outputs = output_act(safe_matmul(hidden, W2_single))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        def eval_single_network_per_node_activation(W1_single, W2_single, act_indices, inputs, targets):
            """Evaluate network with per-node activation (Method A: CPPN output).

            Uses grouped_activation_forward to apply different activations per hidden node.
            """
            hidden_pre = safe_matmul(inputs, W1_single)  # (num_cases, total_positions)
            hidden = grouped_activation_forward(hidden_pre, act_indices, num_activations)
            outputs = output_act(safe_matmul(hidden, W2_single))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        # ================================================================
        # NEW: Sparsity-enhanced evaluation functions
        # ================================================================

        # Capture sparsity config in closures
        df_interpretation = self.df_interpretation
        sparsity_thresholds = self._sparsity_thresholds  # For hierarchical_sparsity
        # Use modular sparsity config if in modular mode, otherwise use weight_interpretation config
        if self.dynamic_functions_mode == 'modular':
            wta_k_percent = self.df_modular_sparsity.get('wta_k_percent', 0.10)
        else:
            wta_k_percent = self.df_sparsity_config.get('wta_k_percent', 0.10)

        def eval_single_network_with_hierarchical_sparsity(
            W1_single, W2_single, act_indices, sparsity_thresh, inputs, targets
        ):
            """Evaluate network with per-node activation AND hierarchical sparsity.

            Applies region-appropriate sparsity after activation:
            - Level 0 (input-adjacent): Very sparse like DG (~5% active)
            - Level 1 (middle): Moderate like CA3 (~20% active)
            - Level 2+ (output-adjacent): Denser like CA1 (~40% active)
            """
            hidden_pre = safe_matmul(inputs, W1_single)  # (num_cases, total_positions)
            hidden = grouped_activation_forward(hidden_pre, act_indices, num_activations)
            # Apply hierarchical sparsity
            hidden = apply_hierarchical_sparsity(hidden, sparsity_thresh)
            outputs = output_act(safe_matmul(hidden, W2_single))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        def eval_single_network_with_node_sparsity(
            W1_single, W2_single, act_indices, node_thresh, inputs, targets
        ):
            """Evaluate network with per-node sparsity thresholds.

            Each node has its own threshold derived from incoming weight magnitude.
            Low magnitude → high threshold (sparse), high magnitude → low threshold (dense).
            """
            hidden_pre = safe_matmul(inputs, W1_single)  # (num_cases, total_positions)
            hidden = grouped_activation_forward(hidden_pre, act_indices, num_activations)
            # Apply per-node sparsity using the node-specific thresholds
            # node_thresh: (total_positions,) with values in [0.5, 0.95]
            hidden = apply_hierarchical_sparsity(hidden, node_thresh)
            outputs = output_act(safe_matmul(hidden, W2_single))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        def eval_single_network_with_wta(W1_single, W2_single, act_indices, inputs, targets):
            """Evaluate network with Winner-Take-All sparsity.

            Keeps only top k% of activations per sample.
            """
            hidden_pre = safe_matmul(inputs, W1_single)  # (num_cases, total_positions)
            hidden = grouped_activation_forward(hidden_pre, act_indices, num_activations)
            # Apply WTA sparsity
            hidden = sparse_activation_wta(hidden, wta_k_percent)
            outputs = output_act(safe_matmul(hidden, W2_single))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        # ================================================================
        # NEUROMODULATION: Gated evaluation functions (Level 1)
        # ================================================================
        # Gate values multiply effective weights: W_eff = W * gates

        # Capture neuromodulation config
        neuromod_config = self.neuromod_config
        use_static_gating = neuromod_config.enabled and neuromod_config.static_gating

        def eval_single_network_gated(W1_single, W2_single, G1_single, G2_single, inputs, targets):
            """Evaluate network with per-connection gating (Level 1 neuromodulation).

            Gating provides activity modulation - same network can have different
            effective connectivity based on evolved gate patterns.

            Args:
                W1_single: Input→Hidden weights, shape (num_inputs, total_positions)
                W2_single: Hidden→Output weights, shape (total_positions, num_outputs)
                G1_single: Input→Hidden gates, shape (num_inputs, total_positions)
                G2_single: Hidden→Output gates, shape (total_positions, num_outputs)
            """
            # Apply gates to weights: effective_weight = weight * gate
            W1_gated = W1_single * G1_single
            W2_gated = W2_single * G2_single

            hidden = hidden_act(safe_matmul(inputs, W1_gated))  # (num_cases, total_positions)
            outputs = output_act(safe_matmul(hidden, W2_gated))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        def eval_single_network_gated_per_node_activation(
            W1_single, W2_single, G1_single, G2_single, act_indices, inputs, targets
        ):
            """Evaluate gated network with per-node activation functions.

            Combines Level 1 gating with per-node activation from CPPN/weight interpretation.
            """
            # Apply gates to weights
            W1_gated = W1_single * G1_single
            W2_gated = W2_single * G2_single

            hidden_pre = safe_matmul(inputs, W1_gated)  # (num_cases, total_positions)
            hidden = grouped_activation_forward(hidden_pre, act_indices, num_activations)
            outputs = output_act(safe_matmul(hidden, W2_gated))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        # ================================================================
        # NEUROMODULATION: Context-Dependent Gated evaluation (Level 2)
        # ================================================================
        # XdG-style: Context derived from input modulates gates per-sample

        use_context_gating = neuromod_config.enabled and neuromod_config.context_gating
        context_influence = neuromod_config.context_influence if use_context_gating else 0.0

        def eval_single_network_context_gated(
            W1_single, W2_single, G1_logit_single, G2_logit_single, inputs, targets
        ):
            """Evaluate network with context-dependent gating (Level 2 neuromodulation).

            Gates are modulated per-sample based on input statistics.
            This allows different effective connectivity for different input patterns.

            Args:
                W1_single: Input→Hidden weights, shape (num_inputs, total_positions)
                W2_single: Hidden→Output weights, shape (total_positions, num_outputs)
                G1_logit_single: Input→Hidden gate logits (pre-sigmoid), shape (num_inputs, total_positions)
                G2_logit_single: Hidden→Output gate logits (pre-sigmoid), shape (total_positions, num_outputs)
            """
            num_cases = inputs.shape[0]

            def process_single_case(carry, x):
                """Process a single input case with context-modulated gates."""
                input_case, target_case = x

                # Derive context from this input (4-dim: mean, std, min, max)
                context = derive_context_from_input(input_case, 'statistics')

                # Simple context modulation: add scaled context mean to gate logits
                # This creates input-dependent gating
                context_scale = context_influence * jnp.mean(context)

                # Apply context-modulated gates
                G1 = jax.nn.sigmoid(G1_logit_single + context_scale)
                G2 = jax.nn.sigmoid(G2_logit_single + context_scale)

                W1_gated = W1_single * G1
                W2_gated = W2_single * G2

                # Forward pass for this single case
                hidden = hidden_act(jnp.dot(input_case, W1_gated))
                output = output_act(jnp.dot(hidden, W2_gated))

                error = jnp.mean((output - target_case) ** 2)
                return carry, error

            # Process all cases
            _, errors = jax.lax.scan(
                process_single_case,
                None,
                (inputs, targets)
            )

            return 1.0 - jnp.mean(errors)

        # ================================================================
        # NEUROMODULATION: TRUE Task-Gated evaluation (Multi-Task)
        # ================================================================
        # CRITICAL: This is the CORRECT implementation for multi-task neuromodulation.
        # Unlike eval_single_network_context_gated which derives context from input,
        # this function takes an EXTERNAL task_context parameter that is INDEPENDENT
        # of the inputs - enabling the SAME substrate to solve MULTIPLE tasks.

        def eval_single_network_task_gated(
            W1_single, W2_single, G1_logit_single, G2_logit_single, task_context, inputs, targets
        ):
            """Evaluate network with EXTERNAL task-context gating (TRUE multi-task neuromodulation).

            This is the KEY function for multi-task learning. The task_context is:
            - EXTERNAL: Not derived from inputs
            - TASK-SPECIFIC: Different for each task (e.g., one-hot encoding)
            - BEHAVIOR-SWITCHING: Same weights + different context = different behavior

            Args:
                W1_single: Input→Hidden weights, shape (num_inputs, total_positions)
                W2_single: Hidden→Output weights, shape (total_positions, num_outputs)
                G1_logit_single: Input→Hidden gate logits (pre-sigmoid), shape (num_inputs, total_positions)
                G2_logit_single: Hidden→Output gate logits (pre-sigmoid), shape (total_positions, num_outputs)
                task_context: EXTERNAL task context vector, shape (num_tasks,) or (context_size,)
                              This is NOT derived from inputs - it specifies WHICH task to solve.
                inputs: Input data, shape (num_cases, num_inputs)
                targets: Target data, shape (num_cases, num_outputs)

            Returns:
                Fitness score (1.0 - mean error)
            """
            # Compute task-based gate modulation from EXTERNAL context
            # This is the key difference from context_gated: context is a parameter, not derived
            context_scale = context_influence * jnp.mean(task_context)

            # Apply task-specific gating to weights
            G1 = jax.nn.sigmoid(G1_logit_single + context_scale)
            G2 = jax.nn.sigmoid(G2_logit_single + context_scale)

            W1_gated = W1_single * G1
            W2_gated = W2_single * G2

            # Forward pass - same weights for all inputs, but gated by task context
            hidden = hidden_act(safe_matmul(inputs, W1_gated))  # (num_cases, total_positions)
            outputs = output_act(safe_matmul(hidden, W2_gated))  # (num_cases, num_outputs)

            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        def eval_single_network_task_gated_per_position(
            W1_single, W2_single, G1_logit_single, G2_logit_single, task_context, inputs, targets
        ):
            """Evaluate with per-position task context modulation.

            More sophisticated version: task context is broadcast to each position
            and used to modulate gates at each position differently.

            This allows the CPPN to evolve position-specific responses to task context.
            """
            num_positions = W1_single.shape[1]

            # Expand task context to match position count
            # If context has fewer dims than positions, tile it
            if len(task_context) < num_positions:
                # Tile context to fill positions
                repeats = (num_positions // len(task_context)) + 1
                expanded_context = jnp.tile(task_context, repeats)[:num_positions]
            else:
                expanded_context = task_context[:num_positions]

            # Per-position gate modulation: different positions can respond differently
            # Shape: (num_positions,) - each position gets its own modulation scale
            position_scales = context_influence * expanded_context

            # Apply position-specific modulation to gates
            # G1_logit: (num_inputs, num_positions) -> add position_scales to each row
            G1 = jax.nn.sigmoid(G1_logit_single + position_scales[None, :])
            G2 = jax.nn.sigmoid(G2_logit_single + position_scales[:, None])

            W1_gated = W1_single * G1
            W2_gated = W2_single * G2

            # Forward pass
            hidden = hidden_act(safe_matmul(inputs, W1_gated))
            outputs = output_act(safe_matmul(hidden, W2_gated))

            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        # Create vmapped version for evaluating entire population with task context
        eval_population_task_gated = jax.vmap(
            eval_single_network_task_gated,
            in_axes=(0, 0, 0, 0, None, None, None)  # task_context, inputs, targets are shared
        )

        eval_population_task_gated_per_position = jax.vmap(
            eval_single_network_task_gated_per_position,
            in_axes=(0, 0, 0, 0, None, None, None)
        )

        # Store task-gated evaluation functions for multi-task usage
        self._eval_single_network_task_gated = eval_single_network_task_gated
        self._eval_single_network_task_gated_per_position = eval_single_network_task_gated_per_position
        self._eval_population_task_gated = eval_population_task_gated
        self._eval_population_task_gated_per_position = eval_population_task_gated_per_position

        # ================================================================
        # TRUE NEUROMODULATION: Neurotransmitter-Based Forward Pass (Level 4)
        # ================================================================
        # This is TRUE biological neuromodulation where:
        # - ALL network parameters (W1, W2, receptor_densities) are FIXED after evolution
        # - ONLY the neurotransmitter signal varies per task
        # - Different NT levels → different activation behaviors → different network outputs
        # - ZERO task-specific learned parameters

        use_true_neuromodulation = neuromod_config.enabled and neuromod_config.true_neuromodulation
        true_mod_strength = neuromod_config.modulation_strength if use_true_neuromodulation else 2.0

        def eval_single_network_true_neuromodulated(
            W1_single, W2_single, receptor_densities_single, base_gains_single, neurotransmitter, inputs, targets, fitness_mode='mse',
            modulation_mode='full', modulation_strength_override=None
        ):
            """Evaluate network with TRUE neuromodulation (Level 4).

            This is the KEY function for TRUE multi-task neuromodulation:
            - ALL parameters (W1, W2, receptor_densities, base_gains) are FIXED
            - ONLY the neurotransmitter vector varies per task
            - Same network produces different behaviors via NT-modulated activation

            The core insight: receptor_densities @ neurotransmitter gives per-neuron modulation.
            This modulation affects both:
            1. GAIN of activation functions (how steep the response is)
            2. GATING of neuron contributions (which neurons are active)

            Args:
                W1_single: Input→Hidden weights, shape (num_inputs, total_positions) - FIXED
                W2_single: Hidden→Output weights, shape (total_positions, num_outputs) - FIXED
                receptor_densities_single: Per-neuron NT sensitivity, shape (total_positions, num_nt_types) - FIXED
                base_gains_single: Base activation gain per neuron, shape (total_positions,) - FIXED
                neurotransmitter: NT levels, shape (num_nt_types,) - VARIABLE (only this changes per task!)
                inputs: Input data, shape (num_cases, num_inputs)
                targets: Target data, shape (num_cases, num_outputs)
                fitness_mode: Fitness function to use ('mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy')
                modulation_mode: Modulation mechanism to use:
                    - 'full': Gain + Bias + Gating (default, original behavior)
                    - 'gating_only': Only use gating, no gain/bias modulation
                    - 'gain_bias_only': Only gain + bias, no gating
                modulation_strength_override: Override default modulation strength (5.0)

            Returns:
                Fitness score based on fitness_mode
            """
            # Use override if provided, otherwise use default
            mod_strength = modulation_strength_override if modulation_strength_override is not None else true_mod_strength

            # Compute per-neuron modulation from FIXED receptors and VARIABLE NT
            # modulation[i] = sum_j(receptor_densities[i,j] * NT[j])
            # Shape: (total_positions,)
            # NOTE: Only use first 3 NT components for modulation
            # NT4 is reserved for output inversion control
            modulation = receptor_densities_single[:, :3] @ neurotransmitter[:3]

            # === FORWARD PASS (mode-dependent) ===
            pre_hidden = safe_matmul(inputs, W1_single)  # (num_cases, total_positions)

            if modulation_mode == 'gating_only':
                # GATING-ONLY: Standard activation, only gating varies by task
                # This is the simplest modulation - just select which neurons are active
                gates = jax.nn.sigmoid(modulation * mod_strength)
                hidden = jnp.tanh(pre_hidden)
                hidden = hidden * gates

            elif modulation_mode == 'gain_bias_only':
                # GAIN+BIAS ONLY: No gating, full gain and bias modulation
                # This tests if gating is necessary or if gain/bias alone suffice
                effective_gains = base_gains_single + mod_strength * modulation
                effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                modulation_bias = modulation * mod_strength
                hidden = jnp.tanh(effective_gains * pre_hidden + modulation_bias)
                # No gating applied

            else:  # 'full' mode (default)
                # FULL MODULATION: Gain + Bias + Gating (original behavior)
                # === GAIN MODULATION ===
                effective_gains = base_gains_single + mod_strength * modulation
                effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)

                # === GATING MODULATION ===
                gates = jax.nn.sigmoid(modulation)

                # === MODULATION BIAS ===
                modulation_bias = modulation * mod_strength

                # Apply gain-modulated activation WITH BIAS
                hidden = jnp.tanh(effective_gains * pre_hidden + modulation_bias)
                # Apply gating
                hidden = hidden * gates

            # Output layer: sigmoid for binary classification
            outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))  # (num_cases, num_outputs)

            # === OUTPUT INVERSION (for complementary tasks) ===
            # Use the 4th NT component to control output inversion
            # NT4 = 0.0: fully invert output (for NAND, NOR)
            # NT4 = 1.0: keep output as-is (for AND, OR)
            # NT4 in between: interpolate smoothly
            invert_signal = neurotransmitter[3]  # 4th NT component (index 3)
            # Smooth interpolation: 0=invert, 1=keep
            invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
            # Linear interpolation between inverted and original
            # When invert_weight=1: outputs unchanged
            # When invert_weight=0: outputs = 1 - outputs (inverted)
            inverted = 1.0 - outputs
            outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted
            # Debug: uncomment to verify inversion is active
            # jax.debug.print("NT4={x}, invert_weight={y}", x=invert_signal, y=invert_weight)

            # Compute fitness using the specified mode
            return compute_fitness(outputs, targets, fitness_mode)

        def eval_single_network_true_neuromodulated_with_hidden(
            W1_single, W2_single, receptor_densities_single, base_gains_single, neurotransmitter, inputs, targets, fitness_mode='mse',
            modulation_mode='full', modulation_strength_override=None
        ):
            """Evaluate network with TRUE neuromodulation AND return hidden activations.

            Same as eval_single_network_true_neuromodulated but additionally returns
            the hidden layer activations for subspace orthogonality computation.

            This is needed for the orthogonality bonus (Liu & Wang 2024):
            Different tasks should occupy orthogonal activation subspaces.

            Args:
                Same as eval_single_network_true_neuromodulated

            Returns:
                Tuple of (fitness, hidden_activations)
                - fitness: Fitness score based on fitness_mode
                - hidden_activations: Hidden layer activations, shape (num_cases, total_positions)
            """
            # Use override if provided, otherwise use default
            mod_strength = modulation_strength_override if modulation_strength_override is not None else true_mod_strength

            # Compute per-neuron modulation from FIXED receptors and VARIABLE NT
            modulation = receptor_densities_single[:, :3] @ neurotransmitter[:3]

            # === FORWARD PASS (mode-dependent) ===
            pre_hidden = safe_matmul(inputs, W1_single)  # (num_cases, total_positions)

            if modulation_mode == 'gating_only':
                gates = jax.nn.sigmoid(modulation * mod_strength)
                hidden = jnp.tanh(pre_hidden)
                hidden = hidden * gates

            elif modulation_mode == 'gain_bias_only':
                effective_gains = base_gains_single + mod_strength * modulation
                effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                modulation_bias = modulation * mod_strength
                hidden = jnp.tanh(effective_gains * pre_hidden + modulation_bias)

            else:  # 'full' mode (default)
                effective_gains = base_gains_single + mod_strength * modulation
                effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                gates = jax.nn.sigmoid(modulation)
                modulation_bias = modulation * mod_strength
                hidden = jnp.tanh(effective_gains * pre_hidden + modulation_bias)
                hidden = hidden * gates

            # Output layer: sigmoid for binary classification
            outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))  # (num_cases, num_outputs)

            # === OUTPUT INVERSION (for complementary tasks) ===
            invert_signal = neurotransmitter[3]
            invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
            inverted = 1.0 - outputs
            outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

            # Return BOTH fitness and hidden activations
            fitness = compute_fitness(outputs, targets, fitness_mode)
            return fitness, hidden

        def eval_single_network_true_neuromodulated_multitask(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            task_nts, task_inputs_list, task_targets_list
        ):
            """Evaluate network on MULTIPLE tasks with different NT vectors.

            The SAME network (same W1, W2, receptor_densities) is evaluated on
            multiple tasks. Each task has its own NT vector, inputs, and targets.
            Fitness = average performance across all tasks.

            This demonstrates TRUE neuromodulation: identical weights, different behaviors.

            Args:
                W1_single, W2_single, receptor_densities_single, base_gains_single: FIXED parameters
                task_nts: NT vectors for each task, shape (num_tasks, num_nt_types)
                task_inputs_list: List of input arrays, one per task
                task_targets_list: List of target arrays, one per task

            Returns:
                Average fitness across all tasks
            """
            num_tasks = task_nts.shape[0]
            total_fitness = 0.0

            for task_idx in range(num_tasks):
                nt = task_nts[task_idx]
                inputs = task_inputs_list[task_idx]
                targets = task_targets_list[task_idx]

                fitness = eval_single_network_true_neuromodulated(
                    W1_single, W2_single, receptor_densities_single, base_gains_single,
                    nt, inputs, targets
                )
                total_fitness = total_fitness + fitness

            return total_fitness / num_tasks

        # Create vmapped versions for population evaluation
        # Note: fitness_mode, modulation_mode, modulation_strength_override are shared (None)
        eval_population_true_neuromodulated = jax.vmap(
            eval_single_network_true_neuromodulated,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None)  # NT, inputs, targets, fitness_mode, modulation_mode, mod_strength are shared
        )

        # Vmapped version that also returns hidden activations for orthogonality computation
        # Returns tuple: (fitness_array, hidden_array) where shapes are (pop_size,) and (pop_size, num_cases, total_positions)
        eval_population_true_neuromodulated_with_hidden = jax.vmap(
            eval_single_network_true_neuromodulated_with_hidden,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None)
        )

        # Store for external access (e.g., multi-task evaluation)
        self._eval_single_network_true_neuromodulated = eval_single_network_true_neuromodulated
        self._eval_single_network_true_neuromodulated_multitask = eval_single_network_true_neuromodulated_multitask
        self._eval_population_true_neuromodulated = eval_population_true_neuromodulated
        self._eval_single_network_true_neuromodulated_with_hidden = eval_single_network_true_neuromodulated_with_hidden
        self._eval_population_true_neuromodulated_with_hidden = eval_population_true_neuromodulated_with_hidden

        # ================================================================
        # Branch-Specific Gating Evaluation Functions
        # ================================================================
        # These functions implement dendritic branch-specific modulation
        # inspired by Liu & Wang (2024) SST-mediated gating.

        def eval_single_network_branch_gated(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            neurotransmitter, inputs, targets, all_positions,
            fitness_mode='mse', modulation_mode='full', modulation_strength_override=None,
            branch_mode='spatial'
        ):
            """Evaluate network with branch-specific NT routing.

            Similar to eval_single_network_true_neuromodulated but uses
            compute_branch_specific_modulation() to route different NT
            components to different spatial regions.

            Args:
                Same as eval_single_network_true_neuromodulated plus:
                all_positions: Spatial coordinates, shape (total_positions, 2)
                branch_mode: 'none', 'spatial', or 'hierarchical'

            Returns:
                Fitness score based on fitness_mode
            """
            mod_strength = modulation_strength_override if modulation_strength_override is not None else true_mod_strength

            # Use branch-specific modulation
            modulation = compute_branch_specific_modulation(
                receptor_densities_single, neurotransmitter, all_positions, branch_mode
            )

            # === FORWARD PASS (mode-dependent) ===
            pre_hidden = safe_matmul(inputs, W1_single)

            if modulation_mode == 'gating_only':
                gates = jax.nn.sigmoid(modulation * mod_strength)
                hidden = jnp.tanh(pre_hidden)
                hidden = hidden * gates

            elif modulation_mode == 'gain_bias_only':
                effective_gains = base_gains_single + mod_strength * modulation
                effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                modulation_bias = modulation * mod_strength
                hidden = jnp.tanh(effective_gains * pre_hidden + modulation_bias)

            else:  # 'full' mode
                effective_gains = base_gains_single + mod_strength * modulation
                effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                gates = jax.nn.sigmoid(modulation)
                modulation_bias = modulation * mod_strength
                hidden = jnp.tanh(effective_gains * pre_hidden + modulation_bias)
                hidden = hidden * gates

            outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))

            # Output inversion
            invert_signal = neurotransmitter[3]
            invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
            inverted = 1.0 - outputs
            outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

            return compute_fitness(outputs, targets, fitness_mode)

        def eval_single_network_branch_gated_with_hidden(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            neurotransmitter, inputs, targets, all_positions,
            fitness_mode='mse', modulation_mode='full', modulation_strength_override=None,
            branch_mode='spatial'
        ):
            """Branch-gated evaluation that also returns hidden activations."""
            mod_strength = modulation_strength_override if modulation_strength_override is not None else true_mod_strength

            modulation = compute_branch_specific_modulation(
                receptor_densities_single, neurotransmitter, all_positions, branch_mode
            )

            pre_hidden = safe_matmul(inputs, W1_single)

            if modulation_mode == 'gating_only':
                gates = jax.nn.sigmoid(modulation * mod_strength)
                hidden = jnp.tanh(pre_hidden)
                hidden = hidden * gates

            elif modulation_mode == 'gain_bias_only':
                effective_gains = base_gains_single + mod_strength * modulation
                effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                modulation_bias = modulation * mod_strength
                hidden = jnp.tanh(effective_gains * pre_hidden + modulation_bias)

            else:  # 'full' mode
                effective_gains = base_gains_single + mod_strength * modulation
                effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                gates = jax.nn.sigmoid(modulation)
                modulation_bias = modulation * mod_strength
                hidden = jnp.tanh(effective_gains * pre_hidden + modulation_bias)
                hidden = hidden * gates

            outputs = jax.nn.sigmoid(safe_matmul(hidden, W2_single))

            invert_signal = neurotransmitter[3]
            invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
            inverted = 1.0 - outputs
            outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

            fitness = compute_fitness(outputs, targets, fitness_mode)
            return fitness, hidden

        # Vmapped versions for population evaluation
        # Note: all_positions, fitness_mode, modulation_mode, mod_strength, branch_mode are shared (None)
        eval_population_branch_gated = jax.vmap(
            eval_single_network_branch_gated,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None)
        )

        eval_population_branch_gated_with_hidden = jax.vmap(
            eval_single_network_branch_gated_with_hidden,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None)
        )

        # Store branch-gated functions
        self._eval_single_network_branch_gated = eval_single_network_branch_gated
        self._eval_population_branch_gated = eval_population_branch_gated
        self._eval_single_network_branch_gated_with_hidden = eval_single_network_branch_gated_with_hidden
        self._eval_population_branch_gated_with_hidden = eval_population_branch_gated_with_hidden

        # ================================================================
        # Two-Module Architecture Evaluation Functions
        # ================================================================
        # Inspired by Liu & Wang (2024) PFC + Sensorimotor separation.
        # Partitions hidden layer into two modules by spatial region:
        # - PFC (x < 0.0): Rule maintenance, uses NT[0:2]
        # - Sensorimotor (x >= 0.0): Task execution, uses NT[1:3]

        def eval_single_network_two_module(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            neurotransmitter, inputs, targets, all_positions,
            fitness_mode='mse', modulation_mode='full', modulation_strength_override=None,
            two_module_mode='parallel'
        ):
            """Evaluate network with two-module architecture.

            Partitions hidden layer into PFC and Sensorimotor modules based on
            spatial x-coordinate. Each module uses different NT components.

            Architecture:
                Inputs -> PFC (x<0) -+-> Outputs
                      |             |
                      `-> SM (x>=0) -+
                           ^
                           | (sequential mode: PFC feeds into SM)

            Args:
                Same as eval_single_network_true_neuromodulated plus:
                all_positions: Spatial coordinates, shape (total_positions, 2)
                two_module_mode: 'parallel' or 'sequential'
                    - 'parallel': Both modules process independently, combine outputs
                    - 'sequential': PFC output feeds into Sensorimotor processing

            Returns:
                Fitness score based on fitness_mode
            """
            mod_strength = modulation_strength_override if modulation_strength_override is not None else true_mod_strength

            # === Spatial partitioning by x-coordinate ===
            x_coords = all_positions[:, 0]
            pfc_mask = x_coords < 0.0  # Left hemisphere = PFC (rule)
            sm_mask = x_coords >= 0.0   # Right hemisphere = Sensorimotor (execution)

            # Count positions in each module
            n_pfc = jnp.sum(pfc_mask)
            n_sm = jnp.sum(sm_mask)

            # === PFC MODULE (Rule Maintenance) ===
            # Uses NT[0:2] for modulation - distinct rule encoding
            W1_pfc = W1_single[:, pfc_mask]
            W2_pfc = W2_single[pfc_mask, :]
            rd_pfc = receptor_densities_single[pfc_mask, :]
            bg_pfc = base_gains_single[pfc_mask]

            pre_hidden_pfc = safe_matmul(inputs, W1_pfc)  # (num_cases, n_pfc)
            mod_pfc = rd_pfc[:, :2] @ neurotransmitter[:2]  # NT[0:2] for PFC

            if modulation_mode == 'gating_only':
                gates_pfc = jax.nn.sigmoid(mod_pfc * mod_strength)
                hidden_pfc = jnp.tanh(pre_hidden_pfc) * gates_pfc
            elif modulation_mode == 'gain_bias_only':
                effective_gains_pfc = bg_pfc + mod_strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                modulation_bias_pfc = mod_pfc * mod_strength
                hidden_pfc = jnp.tanh(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
            else:  # 'full' mode
                effective_gains_pfc = bg_pfc + mod_strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                gates_pfc = jax.nn.sigmoid(mod_pfc)
                modulation_bias_pfc = mod_pfc * mod_strength
                hidden_pfc = jnp.tanh(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
                hidden_pfc = hidden_pfc * gates_pfc

            # === SENSORIMOTOR MODULE (Task Execution) ===
            # Uses NT[1:3] for modulation - overlapping NT component enables coordination
            W1_sm = W1_single[:, sm_mask]
            W2_sm = W2_single[sm_mask, :]
            rd_sm = receptor_densities_single[sm_mask, :]
            bg_sm = base_gains_single[sm_mask]

            pre_hidden_sm = safe_matmul(inputs, W1_sm)  # (num_cases, n_sm)

            # Sequential mode: Add top-down PFC signal to sensorimotor input
            if two_module_mode == 'sequential':
                # Mean PFC activation provides top-down rule signal
                pfc_mean = jnp.mean(hidden_pfc, axis=-1, keepdims=True)  # (num_cases, 1)
                pre_hidden_sm = pre_hidden_sm + pfc_mean  # Broadcast to all SM positions

            mod_sm = rd_sm[:, 1:3] @ neurotransmitter[1:3]  # NT[1:3] for SM

            if modulation_mode == 'gating_only':
                gates_sm = jax.nn.sigmoid(mod_sm * mod_strength)
                hidden_sm = jnp.tanh(pre_hidden_sm) * gates_sm
            elif modulation_mode == 'gain_bias_only':
                effective_gains_sm = bg_sm + mod_strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                modulation_bias_sm = mod_sm * mod_strength
                hidden_sm = jnp.tanh(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
            else:  # 'full' mode
                effective_gains_sm = bg_sm + mod_strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                gates_sm = jax.nn.sigmoid(mod_sm)
                modulation_bias_sm = mod_sm * mod_strength
                hidden_sm = jnp.tanh(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
                hidden_sm = hidden_sm * gates_sm

            # === OUTPUT INTEGRATION ===
            # Combine outputs from both modules
            output_pfc = safe_matmul(hidden_pfc, W2_pfc)
            output_sm = safe_matmul(hidden_sm, W2_sm)
            outputs = jax.nn.sigmoid(output_pfc + output_sm)

            # Output inversion using NT[3]
            invert_signal = neurotransmitter[3]
            invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
            inverted = 1.0 - outputs
            outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

            return compute_fitness(outputs, targets, fitness_mode)

        def eval_single_network_two_module_with_hidden(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            neurotransmitter, inputs, targets, all_positions,
            fitness_mode='mse', modulation_mode='full', modulation_strength_override=None,
            two_module_mode='parallel'
        ):
            """Two-module evaluation that also returns hidden activations."""
            mod_strength = modulation_strength_override if modulation_strength_override is not None else true_mod_strength

            x_coords = all_positions[:, 0]
            pfc_mask = x_coords < 0.0
            sm_mask = x_coords >= 0.0

            # === PFC MODULE ===
            W1_pfc = W1_single[:, pfc_mask]
            rd_pfc = receptor_densities_single[pfc_mask, :]
            bg_pfc = base_gains_single[pfc_mask]
            W2_pfc = W2_single[pfc_mask, :]

            pre_hidden_pfc = safe_matmul(inputs, W1_pfc)
            mod_pfc = rd_pfc[:, :2] @ neurotransmitter[:2]

            if modulation_mode == 'gating_only':
                gates_pfc = jax.nn.sigmoid(mod_pfc * mod_strength)
                hidden_pfc = jnp.tanh(pre_hidden_pfc) * gates_pfc
            elif modulation_mode == 'gain_bias_only':
                effective_gains_pfc = bg_pfc + mod_strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                modulation_bias_pfc = mod_pfc * mod_strength
                hidden_pfc = jnp.tanh(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
            else:
                effective_gains_pfc = bg_pfc + mod_strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                gates_pfc = jax.nn.sigmoid(mod_pfc)
                modulation_bias_pfc = mod_pfc * mod_strength
                hidden_pfc = jnp.tanh(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
                hidden_pfc = hidden_pfc * gates_pfc

            # === SENSORIMOTOR MODULE ===
            W1_sm = W1_single[:, sm_mask]
            rd_sm = receptor_densities_single[sm_mask, :]
            bg_sm = base_gains_single[sm_mask]
            W2_sm = W2_single[sm_mask, :]

            pre_hidden_sm = safe_matmul(inputs, W1_sm)
            if two_module_mode == 'sequential':
                pfc_mean = jnp.mean(hidden_pfc, axis=-1, keepdims=True)
                pre_hidden_sm = pre_hidden_sm + pfc_mean

            mod_sm = rd_sm[:, 1:3] @ neurotransmitter[1:3]

            if modulation_mode == 'gating_only':
                gates_sm = jax.nn.sigmoid(mod_sm * mod_strength)
                hidden_sm = jnp.tanh(pre_hidden_sm) * gates_sm
            elif modulation_mode == 'gain_bias_only':
                effective_gains_sm = bg_sm + mod_strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                modulation_bias_sm = mod_sm * mod_strength
                hidden_sm = jnp.tanh(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
            else:
                effective_gains_sm = bg_sm + mod_strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                gates_sm = jax.nn.sigmoid(mod_sm)
                modulation_bias_sm = mod_sm * mod_strength
                hidden_sm = jnp.tanh(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
                hidden_sm = hidden_sm * gates_sm

            # Combine outputs
            output_pfc = safe_matmul(hidden_pfc, W2_pfc)
            output_sm = safe_matmul(hidden_sm, W2_sm)
            outputs = jax.nn.sigmoid(output_pfc + output_sm)

            invert_signal = neurotransmitter[3]
            invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
            inverted = 1.0 - outputs
            outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

            fitness = compute_fitness(outputs, targets, fitness_mode)

            # Combine hidden for orthogonality computation - pad to total_positions
            # Use zeros for the "other" module positions
            total_positions = all_positions.shape[0]
            num_cases = inputs.shape[0]
            combined_hidden = jnp.zeros((num_cases, total_positions))
            combined_hidden = combined_hidden.at[:, pfc_mask].set(hidden_pfc)
            combined_hidden = combined_hidden.at[:, sm_mask].set(hidden_sm)

            return fitness, combined_hidden

        # Vmapped versions for population evaluation
        eval_population_two_module = jax.vmap(
            eval_single_network_two_module,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None)
        )

        eval_population_two_module_with_hidden = jax.vmap(
            eval_single_network_two_module_with_hidden,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None)
        )

        # Store two-module functions
        self._eval_single_network_two_module = eval_single_network_two_module
        self._eval_population_two_module = eval_population_two_module
        self._eval_single_network_two_module_with_hidden = eval_single_network_two_module_with_hidden
        self._eval_population_two_module_with_hidden = eval_population_two_module_with_hidden

        # ================================================================
        # NEUROMODULATION: Modulatory Neuron evaluation (Level 3)
        # ================================================================
        # Two-phase processing: standard neurons propagate, then modulatory neurons
        # influence the activations of standard neurons

        use_modulatory = neuromod_config.enabled and neuromod_config.modulatory_neurons
        mod_connection_type = neuromod_config.mod_connection_type if use_modulatory else 'multiplicative'

        def eval_single_network_modulatory(
            W1_single, W2_single, node_types_single, W_mod_single, inputs, targets
        ):
            """Evaluate network with modulatory neurons (Level 3 neuromodulation).

            Two-phase forward pass per sample:
            1. Standard forward pass to get initial hidden activations
            2. Modulatory neurons influence standard neuron activations

            Args:
                W1_single: Input→Hidden weights, shape (num_inputs, total_positions)
                W2_single: Hidden→Output weights, shape (total_positions, num_outputs)
                node_types_single: Node types (0=standard, 1=modulatory), shape (total_positions,)
                W_mod_single: Modulation weights, shape (total_positions, total_positions)
            """
            # Standard forward pass
            hidden_pre = safe_matmul(inputs, W1_single)  # (num_cases, total_positions)
            hidden = hidden_act(hidden_pre)

            # Compute modulation signal from modulatory neurons
            # Modulatory neurons (type 1) influence standard neurons (type 0)
            mod_signal = compute_modulation_signal(
                hidden[0] if hidden.ndim > 1 else hidden,  # Use first case for static modulation
                node_types_single,
                W_mod_single,
            )

            # Apply modulation to hidden activations
            hidden_modulated = apply_modulation(hidden, mod_signal, mod_connection_type)

            # Output layer (uses modulated hidden activations)
            outputs = output_act(safe_matmul(hidden_modulated, W2_single))  # (num_cases, num_outputs)
            errors = jnp.mean((outputs - targets) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        def eval_single_network_modulatory_per_case(
            W1_single, W2_single, node_types_single, W_mod_single, inputs, targets
        ):
            """Evaluate network with modulatory neurons, per-case modulation.

            Like eval_single_network_modulatory but computes modulation signal
            per input case (more expensive but more dynamic).
            """
            def process_single_case(carry, x):
                input_case, target_case = x

                # Standard forward
                hidden_pre = jnp.dot(input_case, W1_single)
                hidden = hidden_act(hidden_pre)

                # Compute modulation signal for this input's hidden state
                mod_signal = compute_modulation_signal(
                    hidden,
                    node_types_single,
                    W_mod_single,
                )

                # Apply modulation
                hidden_modulated = apply_modulation(hidden, mod_signal, mod_connection_type)

                # Output
                output = output_act(jnp.dot(hidden_modulated, W2_single))
                error = jnp.mean((output - target_case) ** 2)
                return carry, error

            _, errors = jax.lax.scan(process_single_case, None, (inputs, targets))
            return 1.0 - jnp.mean(errors)

        # Determine which evaluation path to use based on interpretation mode
        # Support both weight_interpretation mode AND modular mode
        use_hierarchical_sparsity = (
            (self.dynamic_functions_mode == 'weight_interpretation' and
             df_interpretation == 'hierarchical_sparsity') or
            (self.dynamic_functions_mode == 'modular' and
             self._modular_use_hierarchical_sparsity)
        )
        use_node_sparsity = (
            (self.dynamic_functions_mode == 'weight_interpretation' and
             df_interpretation == 'sparsity_threshold') or
            (self.dynamic_functions_mode == 'modular' and
             self._modular_use_node_sparsity)
        )
        use_wta_sparsity = (
            self.dynamic_functions_mode == 'modular' and
            self._modular_use_wta_sparsity
        )
        use_stp = (
            (self.dynamic_functions_mode == 'weight_interpretation' and
             df_interpretation == 'stp_inspired') or
            (self.dynamic_functions_mode == 'modular' and
             self._modular_use_stp_scaling)
        )

        # Apply STP depression factor to W1 if using stp_inspired
        if use_stp and hasattr(self, '_stp_depression_factor'):
            # Scale W1 columns by depression factor: (pop_size, total_positions)
            # W1 shape: (pop_size, num_inputs, total_positions)
            W1 = W1 * self._stp_depression_factor[:, None, :]

        # Cache weight matrices for external evaluation methods
        # (e.g., evaluate_with_neurotransmitter, evaluate_with_task_context)
        self._cached_W1 = W1
        self._cached_W2 = W2

        if use_sparse:
            # Sparse path: slice weight matrices to only active positions
            union_mask = jnp.any(masks_A, axis=0)
            active_indices = jnp.nonzero(union_mask, size=total_positions, fill_value=0)[0]
            num_active = jnp.sum(union_mask)

            W1_active = jnp.take(W1, active_indices, axis=2)
            W2_active = jnp.take(W2, active_indices, axis=1)

            # Slice gates/logits for active positions if using neuromodulation
            G1_active = None
            G2_active = None
            G1_logit_active = None
            G2_logit_active = None
            if use_context_gating and G1_logit is not None and G2_logit is not None:
                G1_logit_active = jnp.take(G1_logit, active_indices, axis=2)
                G2_logit_active = jnp.take(G2_logit, active_indices, axis=1)
            elif use_static_gating and G1 is not None and G2 is not None:
                G1_active = jnp.take(G1, active_indices, axis=2)
                G2_active = jnp.take(G2, active_indices, axis=1)

            # Zero out padding (indices beyond num_active are invalid)
            valid_mask = jnp.arange(total_positions) < num_active
            W1_active = W1_active * valid_mask[None, None, :]
            W2_active = W2_active * valid_mask[None, :, None]
            if G1_active is not None:
                G1_active = G1_active * valid_mask[None, None, :]
                G2_active = G2_active * valid_mask[None, :, None]
            if G1_logit_active is not None:
                G1_logit_active = G1_logit_active * valid_mask[None, None, :]
                G2_logit_active = G2_logit_active * valid_mask[None, :, None]

            # Slice modulatory data for active positions if using Level 3
            node_types_active = None
            W_mod_active = None
            if use_modulatory and node_types is not None and W_mod is not None:
                node_types_active = jnp.take(node_types, active_indices, axis=1)
                node_types_active = jnp.where(valid_mask[None, :], node_types_active, 0)
                # W_mod is (pop, total_pos, total_pos) - slice both dimensions
                W_mod_active = jnp.take(jnp.take(W_mod, active_indices, axis=1), active_indices, axis=2)
                W_mod_active = W_mod_active * valid_mask[None, :, None] * valid_mask[None, None, :]

            # Slice TRUE neuromodulation data for active positions if using Level 4
            receptor_densities_active = None
            base_gains_active = None
            if use_true_neuromodulation and hasattr(self, '_neuromod_true') and self._neuromod_true is not None:
                rd = self._neuromod_true['receptor_densities']  # (pop, total_pos, num_nt)
                bg = self._neuromod_true['base_gains']  # (pop, total_pos)
                receptor_densities_active = jnp.take(rd, active_indices, axis=1)
                receptor_densities_active = receptor_densities_active * valid_mask[None, :, None]
                base_gains_active = jnp.take(bg, active_indices, axis=1)
                base_gains_active = base_gains_active * valid_mask[None, :]

            # NEUROMODULATION Level 4: TRUE neuromodulation (sparse)
            # Check Level 4 FIRST - highest priority when enabled
            if use_true_neuromodulation and receptor_densities_active is not None and base_gains_active is not None:
                # Get NT vector for current task (default to XOR for single-task evolution)
                task_name = getattr(self, '_current_task_name', 'xor')
                num_nt = self._neuromod_true['num_nt_types']
                nt_vector = get_nt_for_task(task_name, num_nt)
                # Use 'mse' for single-task (backward compatible)
                # Pass 'full' modulation mode and None for strength override (use default)
                fitnesses = self._eval_population_true_neuromodulated(
                    W1_active, W2_active, receptor_densities_active, base_gains_active,
                    nt_vector, inputs_batch, targets_batch, 'mse', 'full', None
                )
            elif use_modulatory and node_types_active is not None and W_mod_active is not None:
                # NEUROMODULATION Level 3: Modulatory neurons (sparse)
                fitnesses = jax.vmap(
                    eval_single_network_modulatory,
                    in_axes=(0, 0, 0, 0, None, None)
                )(
                    W1_active, W2_active, node_types_active, W_mod_active,
                    inputs_batch, targets_batch
                )
            elif use_context_gating and G1_logit_active is not None and G2_logit_active is not None:
                # NEUROMODULATION Level 2: Context-dependent gating (sparse)
                fitnesses = jax.vmap(
                    eval_single_network_context_gated,
                    in_axes=(0, 0, 0, 0, None, None)
                )(
                    W1_active, W2_active, G1_logit_active, G2_logit_active, inputs_batch, targets_batch
                )
            elif use_static_gating and G1_active is not None and G2_active is not None:
                # NEUROMODULATION: Use gated evaluation with sparse matrices
                if use_per_node_activation:
                    act_indices_active = jnp.take(hidden_activation_indices, active_indices, axis=1)
                    act_indices_active = jnp.where(valid_mask[None, :], act_indices_active, 0)
                    fitnesses = jax.vmap(
                        eval_single_network_gated_per_node_activation,
                        in_axes=(0, 0, 0, 0, 0, None, None)
                    )(
                        W1_active, W2_active, G1_active, G2_active, act_indices_active,
                        inputs_batch, targets_batch
                    )
                else:
                    fitnesses = jax.vmap(
                        eval_single_network_gated,
                        in_axes=(0, 0, 0, 0, None, None)
                    )(
                        W1_active, W2_active, G1_active, G2_active, inputs_batch, targets_batch
                    )
            elif use_hierarchical_sparsity and sparsity_thresholds is not None:
                # Slice sparsity thresholds for active positions
                sparsity_thresh_active = jnp.take(sparsity_thresholds, active_indices, axis=0)
                sparsity_thresh_active = jnp.where(valid_mask, sparsity_thresh_active, 0.5)
                act_indices_active = jnp.take(hidden_activation_indices, active_indices, axis=1)
                act_indices_active = jnp.where(valid_mask[None, :], act_indices_active, 0)
                fitnesses = jax.vmap(
                    eval_single_network_with_hierarchical_sparsity,
                    in_axes=(0, 0, 0, None, None, None)
                )(
                    W1_active, W2_active, act_indices_active, sparsity_thresh_active,
                    inputs_batch, targets_batch
                )
            elif use_node_sparsity and hasattr(self, '_node_sparsity_thresholds'):
                # Slice node sparsity thresholds for active positions
                node_thresh_active = jnp.take(self._node_sparsity_thresholds, active_indices, axis=1)
                node_thresh_active = jnp.where(valid_mask[None, :], node_thresh_active, 0.5)
                act_indices_active = jnp.take(hidden_activation_indices, active_indices, axis=1)
                act_indices_active = jnp.where(valid_mask[None, :], act_indices_active, 0)
                fitnesses = jax.vmap(
                    eval_single_network_with_node_sparsity,
                    in_axes=(0, 0, 0, 0, None, None)
                )(
                    W1_active, W2_active, act_indices_active, node_thresh_active,
                    inputs_batch, targets_batch
                )
            elif use_wta_sparsity and use_per_node_activation:
                # WTA sparsity with per-node activation (modular mode)
                act_indices_active = jnp.take(hidden_activation_indices, active_indices, axis=1)
                act_indices_active = jnp.where(valid_mask[None, :], act_indices_active, 0)
                fitnesses = jax.vmap(eval_single_network_with_wta, in_axes=(0, 0, 0, None, None))(
                    W1_active, W2_active, act_indices_active, inputs_batch, targets_batch
                )
            elif use_per_node_activation:
                # Also slice activation indices for active positions
                act_indices_active = jnp.take(hidden_activation_indices, active_indices, axis=1)
                act_indices_active = jnp.where(valid_mask[None, :], act_indices_active, 0)
                fitnesses = jax.vmap(eval_single_network_per_node_activation, in_axes=(0, 0, 0, None, None))(
                    W1_active, W2_active, act_indices_active, inputs_batch, targets_batch
                )
            else:
                fitnesses = jax.vmap(eval_single_network, in_axes=(0, 0, None, None))(
                    W1_active, W2_active, inputs_batch, targets_batch
                )

            step_timings['sparse_num_active'] = float(num_active)
            step_timings['sparse_total_positions'] = float(total_positions)
            step_timings['sparse_ratio'] = float(num_active) / float(total_positions)
        else:
            # Dense path: use full weight matrices
            # NEUROMODULATION Level 4: TRUE neuromodulation (dense)
            # Check Level 4 FIRST - highest priority when enabled
            if use_true_neuromodulation and hasattr(self, '_neuromod_true') and self._neuromod_true is not None:
                rd = self._neuromod_true['receptor_densities']  # (pop, total_pos, num_nt)
                bg = self._neuromod_true['base_gains']  # (pop, total_pos)
                # Get NT vector for current task (default to XOR for single-task evolution)
                task_name = getattr(self, '_current_task_name', 'xor')
                num_nt = self._neuromod_true['num_nt_types']
                nt_vector = get_nt_for_task(task_name, num_nt)
                # Use 'mse' for single-task (backward compatible)
                # Pass 'full' modulation mode and None for strength override (use default)
                fitnesses = self._eval_population_true_neuromodulated(
                    W1, W2, rd, bg, nt_vector, inputs_batch, targets_batch, 'mse', 'full', None
                )
            elif use_modulatory and node_types is not None and W_mod is not None:
                # NEUROMODULATION Level 3: Modulatory neurons (dense)
                fitnesses = jax.vmap(
                    eval_single_network_modulatory,
                    in_axes=(0, 0, 0, 0, None, None)
                )(
                    W1, W2, node_types, W_mod, inputs_batch, targets_batch
                )
            elif use_context_gating and G1_logit is not None and G2_logit is not None:
                # NEUROMODULATION Level 2: Context-dependent gating
                # Gates are modulated per-sample based on input statistics
                fitnesses = jax.vmap(
                    eval_single_network_context_gated,
                    in_axes=(0, 0, 0, 0, None, None)
                )(
                    W1, W2, G1_logit, G2_logit, inputs_batch, targets_batch
                )
            elif use_static_gating and G1 is not None and G2 is not None:
                # NEUROMODULATION Level 1: Static gating
                if use_per_node_activation:
                    # Gated with per-node activation
                    fitnesses = jax.vmap(
                        eval_single_network_gated_per_node_activation,
                        in_axes=(0, 0, 0, 0, 0, None, None)
                    )(
                        W1, W2, G1, G2, hidden_activation_indices, inputs_batch, targets_batch
                    )
                else:
                    # Gated with fixed activation
                    fitnesses = jax.vmap(
                        eval_single_network_gated,
                        in_axes=(0, 0, 0, 0, None, None)
                    )(
                        W1, W2, G1, G2, inputs_batch, targets_batch
                    )
            elif use_hierarchical_sparsity and sparsity_thresholds is not None:
                # Hierarchical sparsity: level-based sparsity thresholds
                fitnesses = jax.vmap(
                    eval_single_network_with_hierarchical_sparsity,
                    in_axes=(0, 0, 0, None, None, None)
                )(
                    W1, W2, hidden_activation_indices, sparsity_thresholds,
                    inputs_batch, targets_batch
                )
            elif use_node_sparsity and hasattr(self, '_node_sparsity_thresholds'):
                # Per-node sparsity: weight-derived thresholds
                fitnesses = jax.vmap(
                    eval_single_network_with_node_sparsity,
                    in_axes=(0, 0, 0, 0, None, None)
                )(
                    W1, W2, hidden_activation_indices, self._node_sparsity_thresholds,
                    inputs_batch, targets_batch
                )
            elif use_wta_sparsity and use_per_node_activation:
                # WTA sparsity with per-node activation (modular mode)
                fitnesses = jax.vmap(eval_single_network_with_wta, in_axes=(0, 0, 0, None, None))(
                    W1, W2, hidden_activation_indices, inputs_batch, targets_batch
                )
            elif use_per_node_activation:
                # Standard per-node activation (no sparsity)
                fitnesses = jax.vmap(eval_single_network_per_node_activation, in_axes=(0, 0, 0, None, None))(
                    W1, W2, hidden_activation_indices, inputs_batch, targets_batch
                )
            else:
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
    # TRUE Multi-Task Evolution with Aggregated NEAT Selection
    # ========================================================================

    def run_generation_multitask(
        self,
        state: Any,
        multi_task_config: 'MultiTaskConfig',
        skip_metrics: bool = False,
    ) -> Tuple[Any, 'MultiTaskMetrics']:
        """Run one generation of TRUE multi-task neuromodulated evolution.

        This is the KEY method for proper multi-task evolution where:
        1. CPPN evolution happens ONCE per generation
        2. Fitness is computed on ALL tasks with task-specific NT vectors
        3. AGGREGATED multi-task fitness is used for NEAT selection
        4. Same network architecture solves multiple tasks

        Args:
            state: Algorithm state from initialize()
            multi_task_config: MultiTaskConfig with tasks, aggregation, etc.
            skip_metrics: If True, skip metrics extraction

        Returns:
            Tuple of (new_state, MultiTaskMetrics)

        Raises:
            RuntimeError: If TRUE neuromodulation is not enabled
            ValueError: If aggregation method is unknown

        Example:
            multi_task_config = MultiTaskConfig(
                tasks={
                    'xor': (xor_inputs, xor_targets),
                    'and': (and_inputs, and_targets),
                    'or': (or_inputs, or_targets),
                },
                aggregation='mean',
            )
            for gen in range(100):
                state, metrics = algo.run_generation_multitask(state, multi_task_config)
                print(f"Gen {gen}: aggregated={metrics.aggregated_fitness:.4f}")
        """
        import time

        # Validate TRUE neuromodulation is enabled
        if not (self.neuromod_config.enabled and self.neuromod_config.true_neuromodulation):
            raise RuntimeError(
                "run_generation_multitask() requires TRUE neuromodulation. "
                "Set neuromodulation mode to 'true_neuromodulation' or similar."
            )

        gen_start = time.time()

        # Get first task to use as "dummy" problem for W1/W2 construction
        first_task_name = next(iter(multi_task_config.tasks.keys()))
        first_inputs, first_targets = multi_task_config.tasks[first_task_name]

        # Create a minimal problem object for run_generation_verbose's internals
        class _DummyProblem:
            input_shape = (first_inputs.shape[-1],)
            output_shape = (first_targets.shape[-1],)
            jitable = True
            def __init__(self_inner):
                self_inner.inputs = first_inputs
                self_inner.targets = first_targets
                self_inner.input_coords = getattr(self, '_cached_input_coords', [[-1.0, 0.0], [1.0, 0.0]])
                self_inner.output_coords = getattr(self, '_cached_output_coords', [[0.0, 1.0]])
            def get_data(self_inner):
                return list(zip(first_inputs.tolist(), first_targets.tolist()))

        dummy_problem = _DummyProblem()

        # Run single-task evolution to build W1, W2 and set up _neuromod_true
        # This builds all the infrastructure we need
        _, _ = self.run_generation_verbose(state, dummy_problem, skip_metrics=True)

        # Now W1, W2 are cached and _neuromod_true is populated
        # Compute multi-task fitness using aggregation
        if not hasattr(self, '_cached_W1') or self._cached_W1 is None:
            raise RuntimeError("Weight matrices not cached. Ensure TRUE neuromodulation is active.")

        if not hasattr(self, '_neuromod_true') or self._neuromod_true is None:
            raise RuntimeError("TRUE neuromodulation data not available. Check configuration.")

        W1 = self._cached_W1
        W2 = self._cached_W2
        rd = self._neuromod_true['receptor_densities']
        bg = self._neuromod_true['base_gains']
        num_nt = self._neuromod_true['num_nt_types']

        # Check if we need hidden activations for orthogonality bonus
        collect_hidden = multi_task_config.orthogonality_bonus_weight > 0

        # Check if branch-specific gating is enabled
        use_branch_gating = multi_task_config.branch_gating_mode != 'none'

        # Check if two-module architecture is enabled
        use_two_module = multi_task_config.two_module_mode != 'none'

        # Get all_positions for branch gating or two-module (if needed)
        all_positions = None
        if use_branch_gating or use_two_module:
            h_grid = get_hierarchical_grid(self.max_depth)
            all_positions = h_grid.all_positions

        # Compute fitness for each task (and optionally hidden activations)
        per_task_fitness = {}
        per_task_hidden = {} if collect_hidden else None
        for task_name, (inputs, targets) in multi_task_config.tasks.items():
            # Get NT vector for this task
            if multi_task_config.nt_vectors and task_name in multi_task_config.nt_vectors:
                nt_vector = multi_task_config.nt_vectors[task_name]
            else:
                nt_vector = get_nt_for_task(task_name, num_nt)

            # Evaluate population on this task
            inputs_arr = jnp.asarray(inputs)
            targets_arr = jnp.asarray(targets)

            if use_two_module:
                # Use two-module architecture (takes priority over branch gating)
                if collect_hidden:
                    fitness, hidden = self._eval_population_two_module_with_hidden(
                        W1, W2, rd, bg, nt_vector, inputs_arr, targets_arr, all_positions,
                        multi_task_config.fitness_mode,
                        multi_task_config.modulation_mode,
                        multi_task_config.modulation_strength_override,
                        multi_task_config.two_module_mode
                    )
                    per_task_hidden[task_name] = hidden
                else:
                    fitness = self._eval_population_two_module(
                        W1, W2, rd, bg, nt_vector, inputs_arr, targets_arr, all_positions,
                        multi_task_config.fitness_mode,
                        multi_task_config.modulation_mode,
                        multi_task_config.modulation_strength_override,
                        multi_task_config.two_module_mode
                    )
            elif use_branch_gating:
                # Use branch-specific gating functions
                if collect_hidden:
                    fitness, hidden = self._eval_population_branch_gated_with_hidden(
                        W1, W2, rd, bg, nt_vector, inputs_arr, targets_arr, all_positions,
                        multi_task_config.fitness_mode,
                        multi_task_config.modulation_mode,
                        multi_task_config.modulation_strength_override,
                        multi_task_config.branch_gating_mode
                    )
                    per_task_hidden[task_name] = hidden
                else:
                    fitness = self._eval_population_branch_gated(
                        W1, W2, rd, bg, nt_vector, inputs_arr, targets_arr, all_positions,
                        multi_task_config.fitness_mode,
                        multi_task_config.modulation_mode,
                        multi_task_config.modulation_strength_override,
                        multi_task_config.branch_gating_mode
                    )
            elif collect_hidden:
                # Use the variant that also returns hidden activations
                fitness, hidden = self._eval_population_true_neuromodulated_with_hidden(
                    W1, W2, rd, bg, nt_vector, inputs_arr, targets_arr,
                    multi_task_config.fitness_mode,
                    multi_task_config.modulation_mode,
                    multi_task_config.modulation_strength_override
                )
                per_task_hidden[task_name] = hidden  # Shape: (pop_size, num_cases, total_positions)
            else:
                fitness = self._eval_population_true_neuromodulated(
                    W1, W2, rd, bg, nt_vector, inputs_arr, targets_arr,
                    multi_task_config.fitness_mode,
                    multi_task_config.modulation_mode,
                    multi_task_config.modulation_strength_override
                )
            per_task_fitness[task_name] = fitness

        # Aggregate fitness across tasks
        fitness_stack = jnp.stack(list(per_task_fitness.values()), axis=0)

        if multi_task_config.aggregation == 'mean':
            aggregated_fitness = jnp.mean(fitness_stack, axis=0)
        elif multi_task_config.aggregation == 'min':
            aggregated_fitness = jnp.min(fitness_stack, axis=0)
        elif multi_task_config.aggregation == 'weighted':
            if multi_task_config.weights is None:
                raise ValueError("Weighted aggregation requires 'weights' in MultiTaskConfig")
            weights = jnp.array([multi_task_config.weights.get(t, 1.0) for t in multi_task_config.tasks.keys()])
            weights = weights / jnp.sum(weights)  # Normalize
            aggregated_fitness = jnp.sum(fitness_stack * weights[:, None], axis=0)
        elif multi_task_config.aggregation == 'product':
            aggregated_fitness = jnp.prod(fitness_stack, axis=0)
        elif multi_task_config.aggregation == 'softmin':
            # Softmin: weighted average where weights favor lower values
            # Lower fitness tasks get higher weight -> forces optimization of worst tasks
            # Temperature controls sharpness: low temp -> approaches true min
            temp = getattr(multi_task_config, 'softmin_temperature', 0.1)
            # Negate fitness so softmax gives higher weight to lower fitness
            weights = jax.nn.softmax(-fitness_stack / temp, axis=0)
            aggregated_fitness = jnp.sum(fitness_stack * weights, axis=0)
        elif multi_task_config.aggregation == 'harmonic':
            # Harmonic mean: n / sum(1/x_i) - heavily penalizes low values
            # Like product but with gradient even for non-zero low values
            eps = 1e-6  # Prevent division by zero
            n_tasks = fitness_stack.shape[0]
            aggregated_fitness = n_tasks / jnp.sum(1.0 / (fitness_stack + eps), axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {multi_task_config.aggregation}")

        # Apply modulation variance penalty/bonus if enabled
        # This rewards networks that produce different modulation per task,
        # preventing "universal solutions" that ignore NT differentiation.
        if multi_task_config.modulation_penalty > 0.0:
            # Get NT vectors for all tasks
            task_nts = []
            for task_name in multi_task_config.tasks.keys():
                if multi_task_config.nt_vectors and task_name in multi_task_config.nt_vectors:
                    nt = multi_task_config.nt_vectors[task_name]
                else:
                    nt = get_nt_for_task(task_name, num_nt)
                task_nts.append(nt)
            task_nts = jnp.stack(task_nts)  # (num_tasks, num_nt)

            # Compute modulation for each task using mean receptor density per individual
            # rd shape: (pop_size, positions, num_nt) -> mean over positions: (pop_size, num_nt)
            rd_mean = jnp.mean(rd, axis=1)

            # Compute modulation for each task: mod[i,t] = rd_mean[i,:] @ task_nts[t,:]
            # Shape: (pop_size, num_tasks)
            modulations = jnp.einsum('pn,tn->pt', rd_mean, task_nts)

            # Variance across tasks per individual (higher = using NT differently per task)
            mod_variance = jnp.var(modulations, axis=1)  # (pop_size,)

            # Add variance bonus to fitness (high variance = good, used NT differentiation)
            # Scale so bonus is meaningful but doesn't dominate task fitness
            variance_bonus = multi_task_config.modulation_penalty * mod_variance
            aggregated_fitness = aggregated_fitness + variance_bonus

        # Apply generalist bonus/penalty mechanisms
        if multi_task_config.generalist_bonus_type == 'min_bonus':
            # Add bonus proportional to worst-task fitness
            # Encourages improving the weakest task
            min_fitness = jnp.min(fitness_stack, axis=0)
            aggregated_fitness = aggregated_fitness + multi_task_config.generalist_bonus_weight * min_fitness

        elif multi_task_config.generalist_bonus_type == 'variance_penalty':
            # Penalize high variance across tasks (encourages balanced performance)
            fitness_variance = jnp.var(fitness_stack, axis=0)
            aggregated_fitness = aggregated_fitness - multi_task_config.generalist_bonus_weight * fitness_variance

        elif multi_task_config.generalist_bonus_type == 'threshold_bonus':
            # Big bonus if ALL tasks exceed threshold (encourages generalists)
            threshold = multi_task_config.generalist_threshold
            all_above = jnp.all(fitness_stack >= threshold, axis=0).astype(jnp.float32)
            aggregated_fitness = aggregated_fitness + multi_task_config.generalist_bonus_weight * all_above

        # Apply specialization bonus (confusion matrix gap) if enabled
        # This rewards networks where correct NT gives HIGH fitness and wrong NT gives LOW fitness
        if multi_task_config.specialization_bonus_weight > 0:
            # Build confusion matrix: performance[task_idx, nt_idx] = fitness using NT[nt_idx] on task[task_idx]
            # This is computationally expensive (5x5=25 evaluations vs 5)
            # To reduce cost, we sample 2 off-diagonal NTs per task instead of all 4

            task_names = list(multi_task_config.tasks.keys())
            n_tasks = len(task_names)

            # Get all NT vectors (one per task)
            all_nt_vectors = []
            for task_name in task_names:
                if multi_task_config.nt_vectors and task_name in multi_task_config.nt_vectors:
                    nt = multi_task_config.nt_vectors[task_name]
                else:
                    nt = get_nt_for_task(task_name, num_nt)
                all_nt_vectors.append(nt)

            # Diagonal elements: fitness using CORRECT NT (already computed in per_task_fitness)
            diagonal_fitness = fitness_stack  # Shape: (n_tasks, pop_size)

            # Off-diagonal: fitness using WRONG NT for each task
            # Sample 2 wrong NTs per task to reduce computation (3x cost instead of 5x)
            off_diagonal_fitness_list = []

            for t_idx, task_name in enumerate(task_names):
                task_inputs, task_targets = multi_task_config.tasks[task_name]
                inputs_arr = jnp.asarray(task_inputs)
                targets_arr = jnp.asarray(task_targets)

                # Sample 2 wrong NT indices (deterministic for reproducibility)
                wrong_nt_indices = [(t_idx + 1) % n_tasks, (t_idx + 2) % n_tasks]

                for wrong_nt_idx in wrong_nt_indices:
                    wrong_nt_vector = all_nt_vectors[wrong_nt_idx]
                    # Evaluate task with WRONG NT vector
                    wrong_fitness = self._eval_population_true_neuromodulated(
                        W1, W2, rd, bg, wrong_nt_vector, inputs_arr, targets_arr,
                        multi_task_config.fitness_mode,
                        multi_task_config.modulation_mode,
                        multi_task_config.modulation_strength_override
                    )
                    off_diagonal_fitness_list.append(wrong_fitness)

            # Stack off-diagonal: (n_tasks * 2, pop_size)
            off_diagonal_stack = jnp.stack(off_diagonal_fitness_list, axis=0)

            # Compute means per individual
            # diagonal_mean: mean fitness when using correct NT
            diagonal_mean = jnp.mean(diagonal_fitness, axis=0)  # (pop_size,)

            # off_diagonal_mean: mean fitness when using wrong NT
            off_diagonal_mean = jnp.mean(off_diagonal_stack, axis=0)  # (pop_size,)

            # Specialization gap: high when correct NT >> wrong NT
            # This means the network truly USES the NT to differentiate tasks
            specialization_gap = diagonal_mean - off_diagonal_mean

            # Add bonus: networks with large gap get rewarded
            # Gap can range from ~-1.0 to ~+1.0, typically want gap > 0.3
            aggregated_fitness = aggregated_fitness + multi_task_config.specialization_bonus_weight * specialization_gap

        # ================================================================
        # Subspace Orthogonality Bonus (Liu & Wang 2024)
        # ================================================================
        # Force different tasks to occupy orthogonal activation subspaces.
        # Based on Liu & Wang (2024): "Different rules occupy nearly ORTHOGONAL SUBSPACES"
        # and "When SST neurons silenced, subspaces collapsed, performance = chance".
        if multi_task_config.orthogonality_bonus_weight > 0 and per_task_hidden is not None:
            # Compute population-wide orthogonality bonuses
            ortho_bonuses = compute_population_orthogonality(
                per_task_hidden,
                metric=multi_task_config.orthogonality_metric
            )
            # Add orthogonality bonus to aggregated fitness
            # Higher orthogonality (more separation) = higher bonus
            aggregated_fitness = aggregated_fitness + multi_task_config.orthogonality_bonus_weight * ortho_bonuses

        # Use AGGREGATED fitness for NEAT selection (this is the KEY difference!)
        new_state = self._compiled_tell(state, aggregated_fitness)

        # Find best GENERALIST (single individual with highest aggregated fitness)
        # This is THE network that would be deployed for multi-task use
        best_gen_idx = int(jnp.nanargmax(aggregated_fitness))
        best_generalist_per_task = {
            t: float(per_task_fitness[t][best_gen_idx])
            for t in per_task_fitness.keys()
        }

        # Build metrics (use nanmax/nanmean to handle degenerate networks with nan fitness)
        # NOTE: per_task_best shows population-wide max per task (may be different individuals)
        # best_generalist_per_task shows per-task scores for ONE individual (the true benchmark)
        metrics = MultiTaskMetrics(
            aggregated_fitness=float(jnp.nanmax(aggregated_fitness)),
            mean_aggregated_fitness=float(jnp.nanmean(aggregated_fitness)),
            per_task_best={t: float(jnp.nanmax(f)) for t, f in per_task_fitness.items()},
            per_task_mean={t: float(jnp.nanmean(f)) for t, f in per_task_fitness.items()},
            best_generalist_per_task=best_generalist_per_task,
            best_generalist_idx=best_gen_idx,
            generation=new_state.generation if hasattr(new_state, 'generation') else 0,
        )

        return new_state, metrics

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
            # This JIT method uses hardcoded tanh/sigmoid for
            # CUDA compatibility. For dynamic activations, use run_generation() instead.
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

    # ========================================================================
    # Multi-Task Neuromodulation Methods
    # ========================================================================
    # These methods enable TRUE multi-task neuromodulation where the SAME
    # substrate can solve MULTIPLE tasks by changing ONLY the external
    # task context signal.

    def evaluate_with_task_context(
        self,
        state: Any,
        task_context: Any,
        inputs: Any,
        targets: Any,
    ) -> Any:
        """Evaluate population on a specific task using EXTERNAL task context.

        This is the KEY method for multi-task neuromodulation. It uses:
        - EXTERNAL task_context: Not derived from inputs
        - Same weights for all tasks: Only context changes
        - Gates modulated by context: Different effective connectivity per task

        Args:
            state: NEAT state (contains population)
            task_context: EXTERNAL task context vector (e.g., one-hot task ID)
            inputs: Input data for this task, shape (num_cases, num_inputs)
            targets: Target data for this task, shape (num_cases, num_outputs)

        Returns:
            Array of fitness values for the population
        """
        import jax.numpy as jnp

        # Get weights and gates from state
        if not hasattr(self, '_eval_population_task_gated'):
            raise RuntimeError(
                "Task-gated evaluation not available. "
                "Initialize with neuromodulation.static_gating=True"
            )

        # Get weight matrices from cached evaluation data
        W1 = self._cached_W1
        W2 = self._cached_W2
        G1_logit = self._cached_G1_logit
        G2_logit = self._cached_G2_logit

        # Convert inputs/targets to jax arrays
        inputs_arr = jnp.array(inputs)
        targets_arr = jnp.array(targets)
        task_context_arr = jnp.array(task_context)

        # Evaluate population with task context
        fitnesses = self._eval_population_task_gated(
            W1, W2, G1_logit, G2_logit,
            task_context_arr, inputs_arr, targets_arr
        )

        return fitnesses

    def evaluate_with_neurotransmitter(
        self,
        task_name: str,
        inputs: Any,
        targets: Any,
    ) -> Any:
        """Evaluate population on a specific task using TRUE neuromodulation.

        This is the KEY method for TRUE neuromodulation multi-task learning:
        - Uses FIXED receptor_densities and base_gains (from CPPN)
        - Uses task-specific neurotransmitter vector
        - Same network produces different behaviors via NT modulation

        Args:
            task_name: Name of the task (e.g., 'xor', 'and', 'or') for NT lookup
            inputs: Input data for this task, shape (num_cases, num_inputs)
            targets: Target data for this task, shape (num_cases, num_outputs)

        Returns:
            Array of fitness values for the population
        """
        import jax.numpy as jnp

        # Verify TRUE neuromodulation is enabled and data is available
        if not hasattr(self, '_neuromod_true') or self._neuromod_true is None:
            raise RuntimeError(
                "TRUE neuromodulation not available. "
                "Initialize with neuromodulation.true_neuromodulation=True"
            )
        if not hasattr(self, '_eval_population_true_neuromodulated'):
            raise RuntimeError(
                "TRUE neuromodulation evaluation function not available."
            )

        # Get cached weight matrices
        W1 = self._cached_W1
        W2 = self._cached_W2
        rd = self._neuromod_true['receptor_densities']
        bg = self._neuromod_true['base_gains']
        num_nt = self._neuromod_true['num_nt_types']

        # Get NT vector for task
        nt_vector = get_nt_for_task(task_name, num_nt)

        # Convert inputs/targets to jax arrays
        inputs_arr = jnp.array(inputs)
        targets_arr = jnp.array(targets)

        # Evaluate population with TRUE neuromodulation (using 'mse' for backward compat)
        # Pass 'full' modulation mode and None for strength override (use default)
        fitnesses = self._eval_population_true_neuromodulated(
            W1, W2, rd, bg, nt_vector, inputs_arr, targets_arr, 'mse', 'full', None
        )

        return fitnesses

    def evaluate_with_custom_nt(
        self,
        custom_nt_vector: Any,
        inputs: Any,
        targets: Any,
    ) -> Any:
        """Evaluate population with a CUSTOM NT vector (not derived from task name).

        This is used for validation experiments to test cross-task NT application.
        For example: testing XOR's NT vector on AND's data to verify neuromodulation
        is actually changing network behavior.

        Args:
            custom_nt_vector: Custom NT vector (shape: num_nt_types)
            inputs: Input data for evaluation
            targets: Target data for evaluation

        Returns:
            Fitness values for each individual in the population
        """
        import jax.numpy as jnp

        if not (self.neuromod_config.enabled and self.neuromod_config.true_neuromodulation):
            raise RuntimeError("evaluate_with_custom_nt requires TRUE neuromodulation to be enabled")

        if self._neuromod_true is None:
            raise RuntimeError("TRUE neuromodulation not initialized. Call run_generation first.")

        # Get cached weights and neuromodulation parameters
        W1 = self._cached_W1
        W2 = self._cached_W2
        rd = self._neuromod_true['receptor_densities']
        bg = self._neuromod_true['base_gains']

        # Convert NT vector to JAX array
        nt_vector = jnp.asarray(custom_nt_vector)

        # Validate NT vector shape
        expected_nt_types = self.neuromod_config.num_nt_types
        if nt_vector.shape != (expected_nt_types,):
            raise ValueError(f"NT vector shape {nt_vector.shape} != expected ({expected_nt_types},)")

        # Convert inputs/targets
        inputs_arr = jnp.asarray(inputs)
        targets_arr = jnp.asarray(targets)

        # Evaluate population with the custom NT vector (using 'mse' for backward compat)
        # Pass 'full' modulation mode and None for strength override (use default)
        fitnesses = self._eval_population_true_neuromodulated(
            W1, W2, rd, bg, nt_vector, inputs_arr, targets_arr, 'mse', 'full', None
        )

        return fitnesses

    def evaluate_multi_task_true_neuromodulated(
        self,
        tasks: dict,
        aggregation: str = 'mean',
    ) -> Any:
        """Evaluate population on multiple tasks using TRUE neuromodulation.

        This enables evolution where fitness = performance across ALL tasks
        using TRUE neuromodulation (same network, different NTs).

        Args:
            tasks: Dict mapping task_name → (inputs, targets)
                   e.g., {'xor': (xor_inputs, xor_targets), 'and': (...), ...}
            aggregation: How to combine task fitnesses ('mean', 'min', 'product')

        Returns:
            Aggregated fitness values for the population
        """
        import jax.numpy as jnp

        task_fitnesses = []
        for task_name, (inputs, targets) in tasks.items():
            fitnesses = self.evaluate_with_neurotransmitter(task_name, inputs, targets)
            task_fitnesses.append(fitnesses)

        # Stack: shape (num_tasks, pop_size)
        task_fitnesses = jnp.stack(task_fitnesses, axis=0)

        # Aggregate across tasks
        if aggregation == 'mean':
            return jnp.mean(task_fitnesses, axis=0)
        elif aggregation == 'min':
            return jnp.min(task_fitnesses, axis=0)
        elif aggregation == 'product':
            return jnp.prod(task_fitnesses, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def evaluate_multi_task(
        self,
        state: Any,
        multi_task_problem: Any,
        aggregation: str = 'mean',
    ) -> Any:
        """Evaluate population on multiple tasks using multi-task neuromodulation.

        This enables evolution where fitness = performance across ALL tasks.

        Args:
            state: NEAT state
            multi_task_problem: MultiTaskProblem instance with multiple tasks
            aggregation: How to combine task fitnesses ('mean', 'min', 'product')

        Returns:
            Aggregated fitness values for the population
        """
        import jax.numpy as jnp

        task_fitnesses = []
        for task_id in range(multi_task_problem.num_tasks):
            # Get task-specific data and context
            inputs, targets = multi_task_problem.get_task_data(task_id)
            task_context = multi_task_problem.get_task_context(task_id)

            # Evaluate population on this task
            fitnesses = self.evaluate_with_task_context(
                state, task_context, inputs, targets
            )
            task_fitnesses.append(fitnesses)

        # Stack: shape (num_tasks, pop_size)
        task_fitnesses = jnp.stack(task_fitnesses, axis=0)

        # Aggregate across tasks
        if aggregation == 'mean':
            return jnp.mean(task_fitnesses, axis=0)
        elif aggregation == 'min':
            return jnp.min(task_fitnesses, axis=0)
        elif aggregation == 'product':
            return jnp.prod(task_fitnesses, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def run_multi_task_generation(
        self,
        state: Any,
        multi_task_problem: Any,
        aggregation: str = 'mean',
    ) -> tuple:
        """Run one generation of multi-task evolution.

        Uses multi-task fitness (performance across ALL tasks) for selection.

        Args:
            state: NEAT state
            multi_task_problem: MultiTaskProblem instance
            aggregation: How to combine task fitnesses ('mean', 'min', 'product')

        Returns:
            (new_state, metrics) tuple
        """
        import jax.numpy as jnp
        from dataclasses import dataclass

        @dataclass
        class MultiTaskMetrics:
            best_fitness: float
            mean_fitness: float
            task_fitnesses: list  # Best fitness per task
            aggregated_fitness: float

        # Get multi-task fitness
        fitnesses = self.evaluate_multi_task(state, multi_task_problem, aggregation)

        # Update state with aggregated fitnesses
        state = self.neat_algo.tell(state, fitnesses)

        # Compute per-task metrics for best individual
        best_idx = jnp.argmax(fitnesses)
        task_best_fitnesses = []
        for task_id in range(multi_task_problem.num_tasks):
            inputs, targets = multi_task_problem.get_task_data(task_id)
            task_context = multi_task_problem.get_task_context(task_id)
            task_fits = self.evaluate_with_task_context(state, task_context, inputs, targets)
            task_best_fitnesses.append(float(task_fits[best_idx]))

        metrics = MultiTaskMetrics(
            best_fitness=float(jnp.max(fitnesses)),
            mean_fitness=float(jnp.mean(fitnesses)),
            task_fitnesses=task_best_fitnesses,
            aggregated_fitness=float(fitnesses[best_idx]),
        )

        return state, metrics
