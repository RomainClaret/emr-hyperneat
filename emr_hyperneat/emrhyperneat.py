"""EMR-HyperNEAT Unified Extended with Dynamic Functions and Neuromodulation.

This is the FULL implementation combining:
- emrhyperneat_unified_extended.py: All multi-GPU, sparse H→H, multi-hop, caching features
- emrhyperneat_dynamic_functions_aggregation.py: Per-node activation/aggregation functions
- emrhyperneat_neuromodulation_functions.py: All 4 levels of neuromodulation

FEATURES FROM UNIFIED_EXTENDED:
- SparseHiddenConnections, discovery toggle, iterative discovery
- RecurrenceConfig, presets, constraint filtering, caching
- Multi-hop vectorized expansion (iteration_level > 1)

FEATURES FROM DYNAMIC_FUNCTIONS_AGGREGATION:
- 18 activation functions (tanh, sigmoid, relu, sin, burst, resonator, etc.)
- 6 aggregation functions (sum, mean, max, min, product, maxabs)
- 7 selection modes (disabled, global, cppn_output, weight_interpretation, etc.)
- Palette system for custom activation function subsets
- Per-node dynamic function selection

NEUROMODULATION FEATURES (4 LEVELS):
- Level 1: Static Gating - CPPN outputs per-connection gate values
- Level 2: Context-Dependent Gating (XdG-style) - Task context modulates gates
- Level 3: Modulatory Neurons (Soltoggio-style) - Dedicated modulatory neuron population
- Level 4: TRUE Neuromodulation - NT vectors + receptor densities + gain modulation

TRUE NEUROMODULATION (Level 4) KEY FEATURES:
- Neurotransmitter vectors: [DA, 5HT, NE, ACh] (2-6 NT types)
- Receptor density matrices per neuron: (pop_size, total_positions, num_nt_types)
- 7 receptor derivation methods: tanh, abs, normalized, fourier, softmax, orthogonal, phase_shifted
- 3 modulation modes: gating_only, gain_bias_only, full
- H→H connections are ALSO modulated (not just Input→Hidden)
- Multi-task support with 6 aggregation methods (mean, min, weighted, product, softmin, harmonic)

NEUROMODULATION + DYNAMIC FUNCTIONS (Orthogonal Features):
- Dynamic functions: Per-node STRUCTURAL choice (which activation function)
- Neuromodulation: Per-node PARAMETRIC modulation (how much gain/gating)
- Both can be enabled simultaneously for maximum flexibility

HYBRID SPARSE-DENSE ARCHITECTURE
=================================

1. **Dense W1, W2** for input→positions and positions→output (GPU-efficient matmul)
2. **Sparse hidden→hidden** with multi-hop expansion via matrix power
3. **Multi-iteration forward** using scatter-add for h→h, matmul for rest

MULTI-HOP EXPANSION (KEY INNOVATION)
====================================

Multi-hop support uses a JIT-compatible matrix power approach:

```python
# For iteration_level k hops:
A_total = A + decay*A^2 + decay^2*A^3 + ... + decay^(k-1)*A^k

# This captures transitive connections: if A→B→C exists, we add A→C
```

The algorithm is JIT-compatible because iteration_level is a static argument,
allowing the Python loop to compile away.

EXECUTION MODES
===============

1. **Feedforward** (default): Dense W1 → matmul → W2 (no hidden→hidden)
2. **Feedforward+H→H** (hybrid): Dense W1/W2 + sparse hidden→hidden with multi-hop
3. **Full Recurrent**: All connection types (backward, lateral, self-loop)

PRESETS
=======

- feedforward: No recurrence (fastest)
- hidden_only: Phase 2 only (forward h→h)
- with_backward: h→h + feedback loops
- with_lateral: h→h + same-layer connections
- with_self: h→h + self-loops
- full_recurrent: Everything (activate_time=20)

AGGREGATION MODES
=================

H→H AGGREGATION IMPLEMENTATION
------------------------------
The `hh_aggregation_mode` config option controls H→H aggregation behavior:

- `hh_aggregation_mode='sum'` (default): Uses fast scatter_add for H→H contributions
- `hh_aggregation_mode='dynamic'`: Enables per-node aggregation for H→H connections
  - Uses segment operations (sum, mean, max, min) instead of scatter_add
  - Each hidden node can use a different aggregation function
  - Aggregation indices computed via same interpretation logic as activation indices

Helper functions for H→H aggregation:
- `segment_sum_2d()`: Batched segment sum using scatter_add
- `segment_max_2d()`: Batched segment max with -inf initialization
- `segment_min_2d()`: Batched segment min with +inf initialization
- `segment_count_2d()`: Count valid elements per segment
- `scatter_aggregate_by_target()`: Main function that applies per-node aggregation

TWO-LEVEL CACHING ARCHITECTURE
==============================

This implementation uses two independent caching mechanisms for optimal performance:

LEVEL 1: JAX Persistent Compilation Cache (System-Level)
---------------------------------------------------------
Stores compiled XLA functions on disk for reuse across runs.

- Location: /tmp/jax_cache (configurable via JAX_COMPILATION_CACHE_DIR)
- Scope: Reuses compiled functions when array shapes match
- Impact: 2-3.5x speedup across all modes
- Configuration (set before JAX import):
    os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
    os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '1'

LEVEL 2: H→H Connection Cache (Algorithm-Level)
-----------------------------------------------
Caches hidden-to-hidden connections within a run to skip redundant discovery.

- Location: In-memory via HHCacheManager class
- Scope: Skips Phase 2 discovery when variance mask unchanged
- Impact: 2-3x speedup for hybrid/recurrent modes
- Configuration (via recurrence config):
    hh_cache_enabled: true       # Enable/disable caching
    hh_refresh_interval: 5       # Generations between forced refresh
    hh_mask_change_threshold: 0.1  # Refresh if mask changes >10%

Combined Effect:
- Without caching: ~5000-12000ms per generation
- With both caches: ~900-1500ms per generation
- Total potential speedup: Up to 9x faster

CONFIGURATION
=============

```yaml
emr_hyperneat:
  max_depth: 3
  variance_threshold: 0.03

  recurrence:
    preset: hidden_only  # Or configure individually:
    # enabled: true
    # allow_hidden_to_hidden: true
    # allow_backward: false
    # iteration_level: 2
    # multi_hop_algorithm: matrix_power  # or fori_loop
    # hop_decay_factor: 0.8
    # hh_cache_enabled: true
    # hh_refresh_interval: 5
```
"""

import functools
import time
import os
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Any, Callable, Dict, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax

# Multi-GPU imports
try:
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    from jax.experimental import mesh_utils
    SHARD_MAP_AVAILABLE = True
except ImportError:
    SHARD_MAP_AVAILABLE = False
    shard_map = None
    Mesh = None
    P = None
    NamedSharding = None
    mesh_utils = None

# Import base class and utilities
from emr_hyperneat.emrhyperneat_base import (
    EMRHyperNEATMultiGPU,
    MultiGPUStrategy,
    PositionShardingConfig,
    IslandModelConfig,
    HybridShardingConfig,
    PopulationPmapConfig,
    HierarchicalGridStructure,
    DenseQuadtreeStructure,
    get_hierarchical_grid,
    get_quadtree_structure,
    compute_hierarchical_variances_batch,
    compute_subdivision_masks_batch,
    compute_hierarchical_variances_batch_jit,
    compute_subdivision_masks_batch_jit,
    batch_query_population_multi_source_chunked,
    batch_query_population_positions,
    safe_matmul,
    traced_device_get,
    clear_per_device_caches,
)

from emr_hyperneat._compat.core.base_algorithm import BaseAlgorithm, AlgorithmMetrics
from emr_hyperneat._compat.utils.config_manager import ConfigManager


# ============================================================================
# Locality Penalty for Geometry Seeding
# ============================================================================
# Implements distance-based weight scaling to favor local connections.
# Connections within locality_radius get full weight, distant ones are scaled down.

def compute_locality_penalty(
    source_coords: jnp.ndarray,
    target_coords: jnp.ndarray,
    locality_radius: float,
) -> jnp.ndarray:
    """Compute distance-based weight scaling for locality bias (geometry seeding).

    For each source-target pair, compute the Euclidean distance and scale
    weights based on whether the distance is within the locality radius.

    Args:
        source_coords: Source positions, shape (num_sources, 2) - (x, y) coordinates
        target_coords: Target positions, shape (num_targets, 2) - (x, y) coordinates
        locality_radius: Radius within which connections get full weight.
                        Connections beyond this are scaled down.

    Returns:
        Scale factors, shape (num_sources, num_targets) with values in [0, 1]:
        - 1.0 for distances <= locality_radius
        - locality_radius / distance for distances > locality_radius
    """
    # Compute pairwise distances: (num_sources, num_targets)
    # source_coords: (S, 2), target_coords: (T, 2)
    # Expand dims for broadcasting: (S, 1, 2) - (1, T, 2) = (S, T, 2)
    diff = source_coords[:, None, :] - target_coords[None, :, :]  # (S, T, 2)
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (S, T)

    # Compute scale: 1.0 for local, decay for distant
    # Scale = min(1.0, locality_radius / (distance + epsilon))
    epsilon = 1e-6
    scale = jnp.minimum(1.0, locality_radius / (distances + epsilon))

    return scale


def apply_locality_penalty_to_weights(
    weights: jnp.ndarray,
    source_coords: jnp.ndarray,
    target_coords: jnp.ndarray,
    locality_radius: Optional[float],
) -> jnp.ndarray:
    """Apply locality-based distance penalty to weight matrix.

    Args:
        weights: Weight matrix, shape (pop_size, num_sources, num_targets) or
                (num_sources, num_targets)
        source_coords: Source positions, shape (num_sources, 2)
        target_coords: Target positions, shape (num_targets, 2)
        locality_radius: Radius for locality penalty, or None to skip

    Returns:
        Scaled weights with same shape as input
    """
    if locality_radius is None:
        return weights

    # Compute scale factors: (num_sources, num_targets)
    scale = compute_locality_penalty(source_coords, target_coords, locality_radius)

    # Apply scaling - handles both (pop, S, T) and (S, T) shapes
    if weights.ndim == 3:
        # Broadcast scale: (1, S, T) to match (pop, S, T)
        return weights * scale[None, :, :]
    else:
        return weights * scale


# ============================================================================
# Neuromodulation Configuration
# ============================================================================
# Modular neuromodulation system with four progressive levels that can be
# enabled independently or combined.

@dataclass
class NeuromodulationConfig:
    """Configuration for neuromodulation features.

    Four independent levels can be enabled alone or in combination:
    - Level 1: Static Gating - CPPN outputs per-connection gate values
    - Level 2: Context Gating - Task context modulates gates dynamically
    - Level 3: Modulatory Neurons - Two neuron types (standard/modulatory)
    - Level 4: TRUE Neuromodulation - NT vectors + receptor densities

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

        # Level 4: TRUE Neuromodulation (Neurotransmitter-based)
        true_neuromodulation: Enable TRUE biological neuromodulation
        num_nt_types: Number of neurotransmitter types (2-6, default 4)
        modulation_strength: How strongly NT affects activation (default 2.0)
        receptor_from_weight: Option A (True) vs Option B (False) for receptor derivation
        receptor_derivation: Method to derive receptor from weight
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

    # Level 4: TRUE Neuromodulation
    true_neuromodulation: bool = False
    num_nt_types: int = 4  # 2, 3, 4, 5, or 6 NT types
    modulation_strength: float = 2.0

    # Option A vs B for receptor density source:
    # - Option A (receptor_from_weight=True): Derive receptor densities from weight output
    # - Option B (receptor_from_weight=False): Separate CPPN outputs for receptor densities
    receptor_from_weight: bool = True
    receptor_derivation: str = 'tanh'  # 'abs', 'normalized', 'tanh', 'fourier', 'softmax', 'orthogonal', 'phase_shifted'

    # Mode string for convenience (derived from individual flags)
    mode: str = 'disabled'  # 'disabled', 'static_gating', 'context_gating', 'modulatory_neurons', 'true_neuromodulation'

    # Additional parameters
    gate_hardness: float = 10.0  # For soft_threshold gate scaling
    modulation_mode: str = 'full'  # 'full', 'gating_only', 'gain_bias_only'
    use_output_inversion: bool = True  # Whether NT4 controls output inversion
    branch_gating: bool = False  # Branch-specific gating (e.g., IH vs HH)

    # Self-connection query for receptor densities (proper CPPN-based method)
    # When True, uses batch_query_population_self_connections() to get receptor
    # densities from CPPN output at (x,y,x,y,bias) instead of deriving from weights
    use_self_connection_query: bool = False  # Default: derive from weights (current behavior)


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task evolution with neuromodulation.

    Enables multi-task evolution where:
    - NEAT selection uses AGGREGATED fitness across all tasks
    - Each task is evaluated with its task-specific NT vector
    - CPPN evolution happens ONCE per generation (not per task)

    Attributes:
        enabled: Whether multi-task mode is active
        num_tasks: Number of tasks to evaluate
        task_names: Optional list of task names (for NT vector lookup)
        nt_per_task: Optional custom NT vectors per task
        fitness_aggregation: How to combine task fitnesses
            - 'mean': Average fitness across tasks (default)
            - 'min': Minimum fitness (hardest task matters most)
            - 'weighted': Weighted average using task_weights
            - 'product': Geometric mean
            - 'softmin': Soft minimum
            - 'harmonic': Harmonic mean
        task_weights: Optional weights for 'weighted' aggregation
        joint_evolution: If True, evolve on all tasks jointly (default True)
        orthogonality_bonus: Bonus for orthogonal NT vectors (default 0.0)
        specialization_bonus: Bonus for task-specialized activations (default 0.0)
        hidden_activation: Global hidden layer activation function name (default: 'tanh')
            Use this when all tasks should use the same activation function.
            Available: 'tanh', 'sigmoid', 'relu', 'sin', 'gauss', 'identity',
                      'lelu', 'burst', 'resonator', etc. (see ACTIVATION_FUNCTIONS)
        per_task_activation: Optional per-task activation functions (overrides hidden_activation)
            Dict mapping task name to activation function name.
            Example: {'xor': 'sin', 'and': 'tanh', 'or': 'tanh'}
    """
    enabled: bool = False
    num_tasks: int = 2
    task_names: Optional[List[str]] = None
    nt_per_task: Optional[List[Any]] = None
    fitness_aggregation: str = 'mean'
    task_weights: Optional[List[float]] = None
    joint_evolution: bool = True
    orthogonality_bonus: float = 0.0
    specialization_bonus: float = 0.0
    # Dynamic activation function support for multi-task neuromodulation
    hidden_activation: str = 'tanh'  # Global activation for all tasks
    per_task_activation: Optional[Dict[str, str]] = None  # Per-task activation override

    # ==== EXTENDED FIELDS (ported from neuromodulation_functions.py) ====
    # These are opt-in features that preserve default behavior when not set

    # Alternative task specification (Dict mapping task_name -> (inputs, targets))
    tasks: Optional[Dict[str, Tuple[Any, Any]]] = None

    # Fitness mode for evaluation
    fitness_mode: str = 'mse'  # 'mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy', 'ce'

    # Modulation penalty: rewards networks that produce different modulation per task
    modulation_penalty: float = 0.0  # 0.01-0.1 recommended; 0.0 = disabled

    # Softmin parameters (for 'softmin' aggregation)
    softmin_temperature: float = 0.1  # Lower = sharper (more like true min)

    # Generalist bonus mechanisms
    generalist_bonus_type: str = 'none'  # 'none', 'min_bonus', 'variance_penalty', 'threshold_bonus'
    generalist_bonus_weight: float = 0.0  # Weight for the bonus/penalty
    generalist_threshold: float = 0.9  # Threshold for 'threshold_bonus' type

    # Modulation mechanism control
    modulation_mode: str = 'full'  # 'full', 'gating_only', 'gain_bias_only'
    modulation_strength_override: Optional[float] = None  # Override default if set

    # Specialization bonus (confusion matrix gap)
    specialization_bonus_weight: float = 0.0  # Weight for NT-task alignment bonus
    confusion_eval_frequency: int = 0  # Compute confusion matrix every N generations (0 = disabled)

    # Subspace orthogonality bonus (Liu & Wang 2024 mechanism)
    orthogonality_bonus_weight: float = 0.0  # 0.1-0.5 recommended; 0.0 = disabled
    orthogonality_metric: str = 'cosine_mean'  # 'cosine_mean', 'cosine_max', 'correlation'

    # Dendritic branch-specific gating (Liu & Wang 2024 SST mechanism)
    branch_gating_mode: str = 'none'  # 'none', 'spatial', 'hierarchical'

    # Two-Module Architecture (Liu & Wang 2024 PFC/Sensorimotor separation)
    two_module_mode: str = 'none'  # 'none', 'parallel', 'sequential'


@dataclass
class MultiTaskMetrics:
    """Metrics from multi-task evolution with per-task breakdown."""
    aggregated_fitness: float = 0.0
    mean_aggregated_fitness: float = 0.0
    per_task_best: Dict[str, float] = field(default_factory=dict)
    per_task_mean: Dict[str, float] = field(default_factory=dict)
    best_generalist_per_task: Dict[str, float] = field(default_factory=dict)
    best_generalist_idx: int = 0
    generation: int = 0


# Preset configurations for common neuromodulation setups
NEUROMODULATION_PRESETS: Dict[str, NeuromodulationConfig] = {
    'disabled': NeuromodulationConfig(enabled=False),

    'static_gating': NeuromodulationConfig(
        enabled=True,
        static_gating=True,
    ),

    'xdg_style': NeuromodulationConfig(
        enabled=True,
        static_gating=True,
        context_gating=True,
        context_dim=4,
        context_influence=0.5,
    ),

    'modulatory_only': NeuromodulationConfig(
        enabled=True,
        modulatory_neurons=True,
        mod_neuron_ratio=0.1,
    ),

    'full': NeuromodulationConfig(
        enabled=True,
        static_gating=True,
        context_gating=True,
        context_dim=4,
        context_influence=0.5,
        modulatory_neurons=True,
        mod_neuron_ratio=0.1,
    ),

    'true_neuromodulation': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=4,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',
    ),

    'true_neuromodulation_4nt': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=4,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',
    ),

    'true_neuromodulation_5nt': NeuromodulationConfig(
        enabled=True,
        true_neuromodulation=True,
        num_nt_types=5,
        modulation_strength=2.0,
        receptor_from_weight=True,
        receptor_derivation='tanh',
    ),
}


def get_neuromodulation_config(mode: str, custom_config: Optional[Dict] = None) -> NeuromodulationConfig:
    """Get neuromodulation config from preset name or custom dict."""
    if mode == 'custom' and custom_config:
        return NeuromodulationConfig(**custom_config)

    if mode not in NEUROMODULATION_PRESETS:
        mode = 'disabled'

    config = NEUROMODULATION_PRESETS[mode]

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
            'true_neuromodulation': config.true_neuromodulation,
            'num_nt_types': config.num_nt_types,
            'modulation_strength': config.modulation_strength,
            'receptor_from_weight': config.receptor_from_weight,
            'receptor_derivation': config.receptor_derivation,
        }
        config_dict.update(custom_config)
        return NeuromodulationConfig(**config_dict)

    return config


# ============================================================================
# TRUE Neuromodulation: Neurotransmitter Presets (Level 4)
# ============================================================================

# 2 NT Types: Simple dopamine-like / serotonin-like dichotomy
NT_PRESETS_2: Dict[str, jnp.ndarray] = {
    'xor': jnp.array([1.0, 0.0]),
    'and': jnp.array([0.0, 1.0]),
    'or': jnp.array([0.5, 0.5]),
    'nand': jnp.array([0.8, 0.2]),
    'nor': jnp.array([0.2, 0.8]),
    'identity': jnp.array([0.3, 0.7]),
}

# 4 NT Types: [Dopamine, Serotonin, Norepinephrine, ACh]
NT_PRESETS_4: Dict[str, jnp.ndarray] = {
    'xor': jnp.array([0.95, 0.05, 0.95, 1.0]),
    'and': jnp.array([0.1, 0.9, 0.1, 1.0]),
    'or': jnp.array([0.5, 0.5, 0.5, 1.0]),
    'nand': jnp.array([0.9, 0.1, 0.5, 1.0]),
    'nor': jnp.array([0.1, 0.5, 0.9, 1.0]),
    'parity4': jnp.array([0.8, 0.2, 0.6, 1.0]),
    'identity': jnp.array([0.5, 0.5, 0.5, 1.0]),
}

# 5 NT Types: One-hot encoding for PERFECT task separation
NT_PRESETS_5: Dict[str, jnp.ndarray] = {
    'xor':  jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    'and':  jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    'or':   jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    'nand': jnp.array([0.0, 0.0, 0.0, 1.0, 0.0]),
    'nor':  jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    'parity4': jnp.array([0.8, 0.2, 0.0, 0.0, 0.0]),
    'identity': jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
}

# 3 NT Types: Fibonacci-sphere optimized
NT_PRESETS_3_FIBONACCI: Dict[str, jnp.ndarray] = {
    'xor':  jnp.array([0.800, 0.500, 0.900]),
    'and':  jnp.array([0.162, 0.190, 0.700]),
    'or':   jnp.array([0.544, 0.998, 0.500]),
    'nand': jnp.array([0.779, 0.136, 0.300]),
    'nor':  jnp.array([0.205, 0.552, 0.100]),
    'parity4': jnp.array([0.6, 0.3, 0.5]),
    'identity': jnp.array([0.5, 0.5, 0.5]),
}

# 6 NT Types: Orthogonal + extra capacity
NT_PRESETS_6: Dict[str, jnp.ndarray] = {
    'xor':  jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'and':  jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    'or':   jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    'nand': jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    'nor':  jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    'parity4': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    'identity': jnp.array([0.17, 0.17, 0.17, 0.17, 0.17, 0.17]),
}

# Default task presets (4-dimensional for [DA, 5HT, NE, ACh])
# These are the optimized vectors proven to achieve 100% accuracy on 5 tasks
NT_TASK_PRESETS: Dict[str, jnp.ndarray] = {
    'xor': jnp.array([0.95, 0.05, 0.95, 1.0]),   # High DA, low 5HT, high NE, output=normal
    'and': jnp.array([0.10, 0.90, 0.10, 1.0]),   # Low DA, high 5HT, low NE, output=normal
    'or': jnp.array([0.50, 0.50, 0.90, 1.0]),    # Medium DA, medium 5HT, high NE
    'nand': jnp.array([0.10, 0.90, 0.10, 0.0]),  # Like AND but inverted output
    'nor': jnp.array([0.50, 0.50, 0.90, 0.0]),   # Like OR but inverted output
    'classification': jnp.array([0.7, 0.3, 0.5, 1.0]),  # General classification
    'regression': jnp.array([0.5, 0.5, 0.5, 1.0]),      # Balanced for regression
    'memory': jnp.array([0.3, 0.7, 0.4, 1.0]),          # High 5HT for memory tasks
    'sequential': jnp.array([0.6, 0.4, 0.8, 1.0]),      # High NE for sequential
}


# Swarm behavior NT presets (4-dimensional for [DA, 5HT, NE, ACh])
# These enable behavior switching in multi-agent teams via neuromodulation.
# Same receptor densities, same gain modulation, only NT vector changes.
SWARM_NT_PRESETS: Dict[str, jnp.ndarray] = {
    'flock':     jnp.array([0.10, 0.90, 0.10, 1.0]),   # High 5HT: cohesion/alignment
    'forage':    jnp.array([0.95, 0.05, 0.50, 1.0]),   # High DA: explore/exploit
    'evade':     jnp.array([0.50, 0.20, 0.95, 0.0]),   # High NE + inverted: reactive escape
    'disperse':  jnp.array([0.50, 0.10, 0.95, 1.0]),   # High NE: spread out
    'aggregate': jnp.array([0.30, 0.70, 0.30, 1.0]),   # Med-high 5HT: cluster
    'patrol':    jnp.array([0.70, 0.30, 0.60, 1.0]),   # High DA+NE: systematic coverage
    'idle':      jnp.array([0.10, 0.10, 0.10, 1.0]),   # Baseline: minimal modulation
}


def get_nt_for_task(task_name: str, num_nt_types: int = 4, preset_name: Optional[str] = None) -> jnp.ndarray:
    """Get predefined neurotransmitter vector for a task."""
    task_key = task_name.lower()

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
            nt = jnp.pad(nt, (0, num_nt_types - len(nt)), constant_values=0.5)
        return nt[:num_nt_types]
    else:
        return jnp.ones((num_nt_types,)) / num_nt_types


def derive_receptor_from_weight(
    weight: jnp.ndarray,
    method: str = 'tanh',
    num_nt_types: int = 4,
) -> jnp.ndarray:
    """Derive receptor densities from weight value (Option A).

    Supports methods: 'abs', 'normalized', 'tanh', 'fourier', 'softmax', 'orthogonal', 'phase_shifted'
    """
    if num_nt_types not in (2, 3, 4, 5, 6):
        raise ValueError(f"num_nt_types must be 2-6, got {num_nt_types}")

    receptors = []

    if method == 'abs':
        abs_w = jnp.clip(jnp.abs(weight), 0.0, 1.0)
        receptors.append(abs_w)
        receptors.append(1.0 - abs_w)
        if num_nt_types >= 3:
            receptors.append(jnp.clip((weight + 1.0) / 2.0, 0.0, 1.0))
        if num_nt_types >= 4:
            receptors.append(1.0 - jnp.abs(abs_w - 0.5) * 2.0)
        if num_nt_types >= 5:
            receptors.append(abs_w ** 2)
        if num_nt_types >= 6:
            receptors.append(1.0 - abs_w ** 2)

    elif method == 'normalized':
        receptor_0 = jnp.clip((weight + 1.0) / 2.0, 0.0, 1.0)
        receptors.append(receptor_0)
        receptors.append(1.0 - receptor_0)
        if num_nt_types >= 3:
            receptors.append(weight ** 2)
        if num_nt_types >= 4:
            receptors.append(1.0 - (weight ** 2))
        if num_nt_types >= 5:
            receptors.append(jnp.clip((weight ** 3 + 1.0) / 2.0, 0.0, 1.0))
        if num_nt_types >= 6:
            receptors.append(jnp.clip(jnp.abs(weight) * jnp.sign(weight + 0.5), 0.0, 1.0))

    elif method == 'tanh':
        tanh_w = jnp.tanh(weight)
        receptors.append((tanh_w + 1.0) / 2.0)
        receptors.append((-tanh_w + 1.0) / 2.0)
        if num_nt_types >= 3:
            receptors.append(jax.nn.sigmoid(jnp.abs(weight) * 2.0 - 1.0))
        if num_nt_types >= 4:
            receptors.append(jax.nn.sigmoid(1.0 - jnp.abs(weight) * 2.0))
        if num_nt_types >= 5:
            tanh_2w = jnp.tanh(2.0 * weight)
            receptors.append((tanh_2w + 1.0) / 2.0)
        if num_nt_types >= 6:
            receptors.append((-jnp.tanh(2.0 * weight) + 1.0) / 2.0)

    elif method == 'fourier':
        receptors.append((jnp.sin(weight * jnp.pi) + 1.0) / 2.0)
        receptors.append((jnp.cos(weight * jnp.pi) + 1.0) / 2.0)
        if num_nt_types >= 3:
            receptors.append((jnp.sin(2.0 * weight * jnp.pi) + 1.0) / 2.0)
        if num_nt_types >= 4:
            receptors.append((jnp.cos(2.0 * weight * jnp.pi) + 1.0) / 2.0)
        if num_nt_types >= 5:
            receptors.append((jnp.sin(3.0 * weight * jnp.pi) + 1.0) / 2.0)
        if num_nt_types >= 6:
            receptors.append((jnp.cos(3.0 * weight * jnp.pi) + 1.0) / 2.0)

    elif method == 'softmax':
        raw_scores = [jnp.tanh(weight), jnp.tanh(weight - 0.5)]
        if num_nt_types >= 3:
            raw_scores.append(jnp.tanh(weight + 0.5))
        if num_nt_types >= 4:
            raw_scores.append(jnp.sin(weight * jnp.pi))
        if num_nt_types >= 5:
            raw_scores.append(jnp.cos(weight * jnp.pi))
        if num_nt_types >= 6:
            raw_scores.append(jnp.tanh(2.0 * weight))
        scores_stack = jnp.stack(raw_scores[:num_nt_types], axis=-1)
        return jax.nn.softmax(scores_stack, axis=-1)

    elif method == 'orthogonal':
        angle = weight * jnp.pi
        receptors.append((jnp.cos(angle) + 1.0) / 2.0)
        receptors.append((jnp.sin(angle) + 1.0) / 2.0)
        if num_nt_types >= 3:
            receptors.append((jnp.cos(2.0 * angle + jnp.pi / 3) + 1.0) / 2.0)
        if num_nt_types >= 4:
            receptors.append((jnp.sin(2.0 * angle + jnp.pi / 6) + 1.0) / 2.0)
        if num_nt_types >= 5:
            receptors.append((jnp.cos(3.0 * angle + jnp.pi / 4) + 1.0) / 2.0)
        if num_nt_types >= 6:
            receptors.append((jnp.sin(3.0 * angle + jnp.pi / 5) + 1.0) / 2.0)

    elif method == 'phase_shifted':
        receptors.append((jnp.sin(weight * jnp.pi) + 1.0) / 2.0)
        receptors.append((jnp.cos(weight * jnp.pi + jnp.pi / 4) + 1.0) / 2.0)
        if num_nt_types >= 3:
            receptors.append((jnp.sin(2.0 * weight * jnp.pi + jnp.pi / 3) + 1.0) / 2.0)
        if num_nt_types >= 4:
            receptors.append((jnp.cos(2.0 * weight * jnp.pi + jnp.pi / 6) + 1.0) / 2.0)
        if num_nt_types >= 5:
            receptors.append((jnp.sin(3.0 * weight * jnp.pi - jnp.pi / 4) + 1.0) / 2.0)
        if num_nt_types >= 6:
            receptors.append((jnp.cos(3.0 * weight * jnp.pi - jnp.pi / 5) + 1.0) / 2.0)

    else:
        raise ValueError(f"Unknown receptor derivation method: '{method}'")

    return jnp.stack(receptors[:num_nt_types], axis=-1)


def compute_fitness(outputs: jnp.ndarray, targets: jnp.ndarray, mode: str = 'mse') -> float:
    """Compute fitness using specified mode ('mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy')."""
    outputs = outputs.flatten()
    targets = targets.flatten()

    if mode == 'mse':
        mse = jnp.mean((outputs - targets) ** 2)
        return 1.0 - mse
    elif mode == 'accuracy':
        predictions = (outputs > 0.5).astype(jnp.float32)
        return jnp.mean(predictions == targets)
    elif mode == 'acc_mse':
        predictions = (outputs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(predictions == targets)
        mse = jnp.mean((outputs - targets) ** 2)
        return accuracy + 0.01 * (1.0 - mse)
    elif mode == 'hybrid':
        predictions = (outputs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(predictions == targets)
        mse = jnp.mean((outputs - targets) ** 2)
        return 0.8 * accuracy + 0.2 * (1.0 - mse)
    elif mode == 'bce':
        eps = 1e-7
        outputs_safe = jnp.clip(outputs, eps, 1.0 - eps)
        bce = -jnp.mean(
            targets * jnp.log(outputs_safe) +
            (1.0 - targets) * jnp.log(1.0 - outputs_safe)
        )
        return 1.0 / (1.0 + bce)
    elif mode == 'soft_accuracy':
        T = 10.0
        soft_pred = jax.nn.sigmoid((outputs - 0.5) * T)
        return jnp.mean(soft_pred * targets + (1.0 - soft_pred) * (1.0 - targets))
    else:
        mse = jnp.mean((outputs - targets) ** 2)
        return 1.0 - mse


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
    else:  # 'mse' default
        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))


def apply_gate_scaling(
    gate_raw: jnp.ndarray,
    scaling: str,
    threshold: float = 0.5,
    hardness: float = 10.0,
) -> jnp.ndarray:
    """Convert raw CPPN gate output to [0, 1] gate value.

    Args:
        gate_raw: Raw gate values from CPPN
        scaling: Scaling method ('sigmoid', 'binary', 'soft_threshold')
        threshold: Threshold for binary/soft_threshold scaling
        hardness: Sigmoid steepness (higher = sharper transition)
    """
    if scaling == 'sigmoid':
        return jax.nn.sigmoid(gate_raw * hardness)
    elif scaling == 'binary':
        sig = jax.nn.sigmoid(gate_raw * hardness)
        return jnp.where(sig > threshold, 1.0, 0.0)
    elif scaling == 'soft_threshold':
        sig = jax.nn.sigmoid(gate_raw * hardness)
        return jnp.where(sig > threshold, (sig - threshold) / (1.0 - threshold + 1e-8), 0.0)
    else:
        return jax.nn.sigmoid(gate_raw)


def derive_context_from_input(inputs: jnp.ndarray, method: str = 'statistics') -> jnp.ndarray:
    """Derive context vector from input patterns."""
    flat = inputs.flatten() if inputs.ndim > 1 else inputs

    if method == 'mean':
        return jnp.array([jnp.mean(flat)])
    elif method == 'statistics':
        return jnp.array([
            jnp.mean(flat),
            jnp.std(flat) + 1e-8,
            jnp.min(flat),
            jnp.max(flat),
        ])
    else:
        return jnp.array([jnp.mean(flat)])


def compute_modulation_with_branch_gating(
    receptor_densities_single: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    all_positions: jnp.ndarray,
    branch_mode: str = 'none',
) -> jnp.ndarray:
    """Compute per-neuron modulation with optional branch-specific gating."""
    if branch_mode == 'none':
        return receptor_densities_single[:, :3] @ neurotransmitter[:3]

    elif branch_mode == 'spatial':
        x = all_positions[:, 0]
        y = all_positions[:, 1]
        modulation = jnp.zeros(receptor_densities_single.shape[0])

        mask_0 = (x < 0.0) & (y < 0.0)
        mod_0 = receptor_densities_single[:, 0] * neurotransmitter[0]
        modulation = jnp.where(mask_0, mod_0, modulation)

        mask_1 = (x >= 0.0) & (y < 0.0)
        mod_1 = receptor_densities_single[:, 1] * neurotransmitter[1]
        modulation = jnp.where(mask_1, mod_1, modulation)

        mask_2 = (x < 0.0) & (y >= 0.0)
        mod_2 = receptor_densities_single[:, 2] * neurotransmitter[2]
        modulation = jnp.where(mask_2, mod_2, modulation)

        mask_3 = (x >= 0.0) & (y >= 0.0)
        mod_3 = receptor_densities_single[:, :3] @ neurotransmitter[:3]
        modulation = jnp.where(mask_3, mod_3, modulation)

        return modulation

    elif branch_mode == 'hierarchical':
        y = all_positions[:, 1]
        modulation = jnp.zeros(receptor_densities_single.shape[0])

        mask_1 = y < -0.33
        mod_1 = receptor_densities_single[:, 0] * neurotransmitter[0]
        modulation = jnp.where(mask_1, mod_1, modulation)

        mask_2 = (y >= -0.33) & (y < 0.33)
        mod_2 = receptor_densities_single[:, 1] * neurotransmitter[1]
        modulation = jnp.where(mask_2, mod_2, modulation)

        mask_3 = y >= 0.33
        mod_3 = receptor_densities_single[:, 2] * neurotransmitter[2]
        modulation = jnp.where(mask_3, mod_3, modulation)

        return modulation

    else:
        raise ValueError(f"Unknown branch_gating_mode: {branch_mode}")


# ============================================================================
# Self-Connection Query for Receptor Densities
# ============================================================================
# Proper CPPN-based method for extracting per-node receptor densities using
# self-connection queries at (x, y, x, y, bias).

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


# ============================================================================
# Orthogonality Metrics (Liu & Wang 2024 Mechanism)
# ============================================================================
# Functions for computing subspace orthogonality bonuses to encourage
# different tasks to occupy orthogonal activation subspaces.

def compute_subspace_orthogonality(
    hidden_list: List[jnp.ndarray],
    metric: str = 'cosine_mean',
) -> float:
    """Compute orthogonality bonus from hidden activations across tasks.

    Based on Liu & Wang (2024) finding: "Different rules occupy nearly ORTHOGONAL SUBSPACES"
    and "When SST neurons silenced, subspaces collapsed, performance = chance".

    This function measures how orthogonal task-specific activation patterns are.
    Higher orthogonality = better task separation.

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


# ============================================================================
# Dynamic Activation Functions Support
# ============================================================================
# This section provides infrastructure for evolved/configurable activation functions.
# Copied from emrhyperneat_dynamic_functions_aggregation.py

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
    'softplus': lambda x: jnp.log1p(jnp.exp(x)),  # Smooth ReLU
    'rs_adapt': lambda x: jnp.tanh(x) * (1 - 0.3 * jnp.abs(x)),  # Self-attenuating tanh
    'fs_fast': lambda x: jax.nn.relu(x) * 2.0,  # Scaled ReLU
    'lts_low': lambda x: jax.nn.sigmoid(x * 2 - 0.5),  # Shifted sigmoid
    'burst': lambda x: jnp.tanh(x) + 0.5 * jnp.sin(x * 3),  # Tanh + sine oscillation
    'resonator': lambda x: jnp.sin(x) * jnp.exp(-jnp.abs(x) / 3),  # Damped sine
    # bio-inspired activations (indices 13-17)
    'osc_adapt': lambda x: jnp.sin(x) * (1 - 0.2 * jnp.abs(x)),  # Oscillatory + adaptive attenuation
    'gain_mod': lambda x: x / (1 + jnp.abs(x)),  # Cortical divisive normalization
    'receptive': lambda x: jnp.exp(-x**2) * jnp.cos(2*x),  # Localized oscillatory response
    'band_pass': lambda x: jnp.exp(-jnp.abs(x - 1)) - jnp.exp(-jnp.abs(x + 1)),  # Intermediate value filter
    'integrate': lambda x: jnp.tanh(x) * (1 + 0.2 * jnp.exp(-jnp.abs(x))),  # Membrane dynamics inspired
}

# Ordered list for indexing (used in cppn_output and weight_interpretation modes)
ACTIVATION_LIST = [
    # Original functions (indices 0-6)
    'tanh', 'sigmoid', 'relu', 'identity', 'sin', 'gauss', 'lelu',
    # Izhikevich-inspired (indices 7-12)
    'softplus', 'rs_adapt', 'fs_fast', 'lts_low', 'burst', 'resonator',
    # bio-inspired (indices 13-17)
    'osc_adapt', 'gain_mod', 'receptive', 'band_pass', 'integrate',
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
    activation_funcs = [ACTIVATION_FUNCTIONS[name] for name in ACTIVATION_LIST[:num_activations]]

    result = jnp.zeros_like(pre_activation)
    for idx, func in enumerate(activation_funcs):
        mask = (act_indices == idx)
        activated = func(pre_activation)
        result = jnp.where(mask[None, :] if pre_activation.ndim == 2 else mask, activated, result)
    return result


# ============================================================================
# Palette Support for Custom Activation Function Sets
# ============================================================================
# This enables palette evolution strategies to work with specific activation
# function subsets (e.g., sin-only for parity problems).

PALETTE_CONFIGS = {
    'default': [0, 1, 2, 3],           # tanh, sigmoid, relu, identity
    'oscillatory': [4, 11, 12, 5],     # sin, burst, resonator, gauss
    'sin_only': [4],                   # sin - optimal for parity (43x speedup)
    'parity_optimal': [4, 11, 12],     # sin, burst, resonator
    'classification': [0, 1, 2, 3],    # tanh, sigmoid, relu, identity
    'full': list(range(18)),           # all 18 activations (indices 0-17)
    # bio-inspired palettes
    'bio_oscillatory': [4, 11, 12, 13, 15],  # sin, burst, resonator, osc_adapt, receptive
    'bio_adaptive': [8, 9, 10, 14, 17],      # rs_adapt, fs_fast, lts_low, gain_mod, integrate
    'phase4_all': list(range(13, 18)),       # new Phase 4 functions only
}


def continuous_to_palette_index(
    raw: jnp.ndarray,
    palette: jnp.ndarray,
) -> jnp.ndarray:
    """Convert continuous CPPN output to activation index using custom palette.

    Instead of mapping to indices 0..N-1, maps to specific indices in palette array.
    This allows testing hypothesis that palette choice (which activations) matters.

    Args:
        raw: Continuous values from CPPN
        palette: Array of activation indices to use, e.g., [4, 11, 12] for sin, burst, resonator

    Returns:
        Integer indices from the palette array
    """
    num_options = len(palette)
    scaled = (jnp.tanh(raw) + 1.0) / 2.0 * num_options
    position = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, num_options - 1)
    return palette[position]


def grouped_activation_forward_with_palette(
    pre_activation: jnp.ndarray,
    act_indices: jnp.ndarray,
    palette: jnp.ndarray,
) -> jnp.ndarray:
    """Apply activation functions from custom palette to nodes.

    Like grouped_activation_forward but act_indices contains actual ACTIVATION_LIST
    indices (e.g., 4 for sin, 11 for burst) rather than sequential 0..N-1.

    Args:
        pre_activation: Pre-activation values, shape (batch, num_nodes)
        act_indices: Per-node activation indices from ACTIVATION_LIST, shape (num_nodes,)
        palette: Array of activation indices that may be present, for efficient iteration

    Returns:
        Activated values with same shape as pre_activation
    """
    result = jnp.zeros_like(pre_activation)
    for act_idx in palette:
        func = ACTIVATION_FUNCTIONS[ACTIVATION_LIST[act_idx]]
        mask = (act_indices == act_idx)
        activated = func(pre_activation)
        result = jnp.where(mask[None, :] if pre_activation.ndim == 2 else mask, activated, result)
    return result


# ============================================================================
# Weight Interpretation Methods for Activation Selection
# ============================================================================

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
# Dynamic Aggregation Functions Support
# ============================================================================
# This section provides infrastructure for evolved/configurable aggregation functions.
# Note: Matrix multiplication (W @ x) inherently performs SUM aggregation.
# To support other aggregations, we need true per-node aggregation or approximation.

AGGREGATION_FUNCTIONS = {
    'sum': lambda z: jnp.sum(z, axis=1),
    'mean': lambda z: jnp.mean(z, axis=1),
    'max': lambda z: jnp.max(z, axis=1),
    'min': lambda z: jnp.min(z, axis=1),
    'product': lambda z: jnp.prod(jnp.clip(z, -10, 10), axis=1),  # Clip to prevent NaN/Inf
    'maxabs': lambda z: jnp.take_along_axis(z, jnp.argmax(jnp.abs(z), axis=1, keepdims=True), axis=1).squeeze(1),
}

AGGREGATION_LIST = ['sum', 'mean', 'max', 'min', 'product', 'maxabs']


def grouped_aggregation_forward(
    weighted_inputs: jnp.ndarray,
    agg_indices: jnp.ndarray,
    num_aggregations: int = 4,
) -> jnp.ndarray:
    """Apply different aggregation functions to different nodes via grouping.

    JAX-efficient implementation: applies each aggregation to all nodes, then
    uses masks to combine results. Avoids per-node branching.

    Args:
        weighted_inputs: Weighted input contributions, shape (batch, num_inputs, num_nodes)
        agg_indices: Per-node aggregation indices, shape (num_nodes,)
        num_aggregations: Number of aggregation functions available

    Returns:
        Aggregated values with shape (batch, num_nodes)

    Example:
        # For a batch of 4 samples, 3 inputs, 5 nodes:
        # weighted_inputs shape: (4, 3, 5)
        # agg_indices shape: (5,) with values in [0, num_aggregations-1]
        # Output shape: (4, 5)
    """
    agg_funcs = [AGGREGATION_FUNCTIONS[name] for name in AGGREGATION_LIST[:num_aggregations]]

    batch_size = weighted_inputs.shape[0]
    num_nodes = weighted_inputs.shape[2]

    result = jnp.zeros((batch_size, num_nodes))
    for idx, func in enumerate(agg_funcs):
        mask = (agg_indices == idx)  # (num_nodes,)
        aggregated = func(weighted_inputs)  # (batch, num_nodes)
        result = jnp.where(mask[None, :], aggregated, result)
    return result


def aggregation_from_weight_interpretation(
    weights: jnp.ndarray,
    interpretation: str,
    num_aggregations: int,
) -> jnp.ndarray:
    """Derive aggregation indices from weight patterns.

    Hypothesis: Aggregation should relate to weight DISTRIBUTION.
    - High variance → max/min (select dominant input)
    - Low variance → mean (average inputs equally)
    - Mixed signs → sum (preserve cancellation effects)
    - Uniform signs → product (AND-like behavior)

    Args:
        weights: Weight matrix, shape (pop_size, num_inputs, num_nodes)
        interpretation: Method name: 'magnitude_bio', 'variance', 'sign_uniformity'
        num_aggregations: Number of aggregation functions available

    Returns:
        Aggregation indices, shape (pop_size, num_nodes)
    """
    if interpretation == 'magnitude_bio':
        # High magnitude → sum (preserve scale), low → mean (normalize)
        mean_abs = jnp.mean(jnp.abs(weights), axis=1)  # (pop_size, num_nodes)
        normalized = -jnp.tanh(mean_abs * 2 - 1)  # Inverted like activation
        return continuous_to_index(normalized, num_aggregations)

    elif interpretation == 'variance':
        # High variance → max (winner-take-all), low → mean (equal contribution)
        weight_var = jnp.var(weights, axis=1)  # (pop_size, num_nodes)
        normalized = jnp.tanh(weight_var * 5)
        return continuous_to_index(normalized, num_aggregations)

    elif interpretation == 'sign_uniformity':
        # All same sign → product (AND-gate), mixed → sum (preserve cancellation)
        pos_ratio = jnp.mean((weights > 0).astype(jnp.float32), axis=1)  # (pop_size, num_nodes)
        sign_uniformity = 2 * jnp.abs(pos_ratio - 0.5)  # 0 = mixed, 1 = uniform
        return continuous_to_index(sign_uniformity, num_aggregations)

    elif interpretation == 'fan_in':
        # Based on number of significant inputs (above threshold)
        significant = jnp.abs(weights) > 0.1  # (pop_size, num_inputs, num_nodes)
        fan_in = jnp.sum(significant.astype(jnp.float32), axis=1)  # (pop_size, num_nodes)
        max_fan_in = weights.shape[1]
        normalized = jnp.tanh((fan_in / max_fan_in) * 2 - 1)
        return continuous_to_index(normalized, num_aggregations)

    else:
        # Default: magnitude_bio (same as activation)
        mean_abs = jnp.mean(jnp.abs(weights), axis=1)
        normalized = -jnp.tanh(mean_abs * 2 - 1)
        return continuous_to_index(normalized, num_aggregations)


def compute_aggregation_correction_factors(
    agg_indices: jnp.ndarray,
    num_inputs: int,
    num_aggregations: int = 4,
) -> jnp.ndarray:
    """Compute correction factors for approximation-based aggregation.

    When using matmul (which does sum), we can approximate other aggregations
    by multiplying by correction factors:
    - sum: factor = 1.0 (no change)
    - mean: factor = 1.0 / num_inputs
    - max/min: factor = 1.0 (not approximable, falls back to sum)
    - product: factor = 1.0 (not approximable, falls back to sum)
    - maxabs: factor = 1.0 (not approximable, falls back to sum)

    Note: Only 'sum' and 'mean' are faithfully represented by this approximation.
    Other aggregations require true per-node aggregation for accuracy.

    Args:
        agg_indices: Per-node aggregation indices, shape (num_nodes,) or (pop_size, num_nodes)
        num_inputs: Number of inputs to the layer
        num_aggregations: Number of aggregation functions available

    Returns:
        Correction factors with same shape as agg_indices (broadcastable)
    """
    factors = jnp.ones_like(agg_indices, dtype=jnp.float32)

    # mean = index 1 in AGGREGATION_LIST → divide by num_inputs
    mean_idx = 1  # AGGREGATION_LIST.index('mean')
    factors = jnp.where(agg_indices == mean_idx, 1.0 / num_inputs, factors)

    return factors


# ============================================================================
# PMAP Evaluation Functions (Zero IPC Overhead Multi-GPU)
# ============================================================================
# These functions enable native JAX multi-GPU evaluation without Python IPC.
# Key design:
# - W1, W2, h→h connection data are REPLICATED across all devices
# - Input/target data is SHARDED across devices (each device gets a portion)
# - Evaluation runs in parallel, returns partial errors for aggregation


def _pmap_eval_single_dense(w1: jnp.ndarray, w2: jnp.ndarray,
                             inputs: jnp.ndarray, targets: jnp.ndarray,
                             activate_time: int) -> jnp.ndarray:
    """Evaluate single network in dense (feedforward) mode - pmap compatible.

    Pure JAX function with no Python control flow.

    Args:
        w1: Input→Hidden weights, shape (n_inputs, n_hidden)
        w2: Hidden→Output weights, shape (n_hidden, n_outputs)
        inputs: Input samples, shape (batch_size, n_inputs)
        targets: Target outputs, shape (batch_size, n_outputs)
        activate_time: Number of activation steps (typically 1 for feedforward)

    Returns:
        Sum of squared errors for this genome on this data shard (scalar)
    """
    # Feedforward: inputs @ w1 → tanh → @ w2 → sigmoid
    hidden = jnp.tanh(inputs @ w1)
    outputs = jax.nn.sigmoid(hidden @ w2)
    errors = (outputs - targets) ** 2
    return jnp.sum(errors)


def _pmap_eval_single_hybrid(w1: jnp.ndarray, w2: jnp.ndarray,
                              hh_from: jnp.ndarray, hh_to: jnp.ndarray,
                              hh_weights: jnp.ndarray, hh_valid: jnp.ndarray,
                              inputs: jnp.ndarray, targets: jnp.ndarray,
                              activate_time: int, num_hidden: int) -> jnp.ndarray:
    """Evaluate single network with h→h connections - pmap compatible.

    Pure JAX function using lax.scan for activation steps.
    Matches _forward_hybrid_sparse_hh behavior exactly:
    1. Initialize with h = tanh(inputs @ W1)
    2. Run activate_time - 1 more steps with residual: tanh(h + h_delta)

    Args:
        w1: Input→Hidden weights, shape (n_inputs, n_hidden)
        w2: Hidden→Output weights, shape (n_hidden, n_outputs)
        hh_from: H→H source indices, shape (max_connections,)
        hh_to: H→H target indices, shape (max_connections,)
        hh_weights: H→H connection weights, shape (max_connections,)
        hh_valid: H→H validity mask, shape (max_connections,)
        inputs: Input samples, shape (batch_size, n_inputs)
        targets: Target outputs, shape (batch_size, n_outputs)
        activate_time: Number of activation steps
        num_hidden: Number of hidden positions

    Returns:
        Sum of squared errors for this genome on this data shard (scalar)
    """
    # Step 1: Dense input→hidden (matches single-GPU)
    h = jnp.tanh(inputs @ w1)  # (batch_size, num_hidden)

    # Step 2: Sparse h→h iterations with RESIDUAL connection (matches single-GPU)
    # Only run if activate_time > 1 and we have valid h→h connections
    num_valid = jnp.sum(hh_valid)

    def run_hh_steps(h_init):
        # Clip indices to valid range for safe scatter operations
        safe_from = jnp.clip(hh_from, 0, num_hidden - 1)
        safe_to = jnp.clip(hh_to, 0, num_hidden - 1)
        effective_hh_w = jnp.where(hh_valid, hh_weights, 0.0)

        def hh_step(h, _):
            """Single h→h step with residual connection."""
            # Gather source values
            source_vals = h[:, safe_from]  # (batch_size, max_connections)

            # Multiply by weights
            contributions = source_vals * effective_hh_w

            # Scatter-add to target positions
            h_delta = jnp.zeros_like(h)
            h_delta = h_delta.at[:, safe_to].add(contributions)

            # Update with RESIDUAL connection and activation (matches single-GPU)
            return jnp.tanh(h + h_delta), None

        # Run activate_time - 1 MORE steps (first step was input→hidden)
        h_final, _ = lax.scan(hh_step, h_init, None, length=activate_time - 1)
        return h_final

    # Use lax.cond to handle the case where we skip h→h steps
    h_final = lax.cond(
        (activate_time > 1) & (num_valid > 0),
        run_hh_steps,
        lambda h: h,
        h
    )

    # Step 3: Dense hidden→output
    outputs = jax.nn.sigmoid(h_final @ w2)
    errors = (outputs - targets) ** 2
    return jnp.sum(errors)


def _pmap_eval_batch_dense(W1: jnp.ndarray, W2: jnp.ndarray,
                            inputs: jnp.ndarray, targets: jnp.ndarray,
                            activate_time: int) -> jnp.ndarray:
    """Evaluate population in dense mode on a data shard.

    Args:
        W1: Population input→hidden weights, shape (pop_size, n_inputs, n_hidden)
        W2: Population hidden→output weights, shape (pop_size, n_hidden, n_outputs)
        inputs: Data shard inputs, shape (shard_size, n_inputs)
        targets: Data shard targets, shape (shard_size, n_outputs)
        activate_time: Activation steps

    Returns:
        Sum of squared errors per genome on this shard, shape (pop_size,)
    """
    # vmap over population: each genome evaluated on same data shard
    return jax.vmap(
        lambda w1, w2: _pmap_eval_single_dense(w1, w2, inputs, targets, activate_time),
        in_axes=(0, 0)
    )(W1, W2)


def _pmap_eval_batch_hybrid(W1: jnp.ndarray, W2: jnp.ndarray,
                             hh_from: jnp.ndarray, hh_to: jnp.ndarray,
                             hh_weights: jnp.ndarray, hh_valid: jnp.ndarray,
                             inputs: jnp.ndarray, targets: jnp.ndarray,
                             activate_time: int, num_hidden: int) -> jnp.ndarray:
    """Evaluate population with h→h on a data shard.

    Args:
        W1: Population input→hidden weights, shape (pop_size, n_inputs, n_hidden)
        W2: Population hidden→output weights, shape (pop_size, n_hidden, n_outputs)
        hh_from: H→H source indices per genome, shape (pop_size, max_connections)
        hh_to: H→H target indices per genome, shape (pop_size, max_connections)
        hh_weights: H→H weights per genome, shape (pop_size, max_connections)
        hh_valid: H→H validity per genome, shape (pop_size, max_connections)
        inputs: Data shard inputs, shape (shard_size, n_inputs)
        targets: Data shard targets, shape (shard_size, n_outputs)
        activate_time: Activation steps
        num_hidden: Number of hidden positions

    Returns:
        Sum of squared errors per genome on this shard, shape (pop_size,)
    """
    # vmap over population: each genome evaluated on same data shard
    return jax.vmap(
        lambda w1, w2, hf, ht, hw, hv: _pmap_eval_single_hybrid(
            w1, w2, hf, ht, hw, hv, inputs, targets, activate_time, num_hidden
        ),
        in_axes=(0, 0, 0, 0, 0, 0)
    )(W1, W2, hh_from, hh_to, hh_weights, hh_valid)


# Create pmap-wrapped evaluation functions
# in_axes: None = replicated (same on all devices), 0 = sharded (different per device)
# Dense mode: W1, W2 replicated; inputs, targets sharded
_PMAP_EVAL_DENSE = jax.pmap(
    _pmap_eval_batch_dense,
    axis_name='devices',
    in_axes=(None, None, 0, 0, None),  # W1, W2, inputs, targets, activate_time
    static_broadcasted_argnums=(4,),   # activate_time is static
)

# Hybrid mode: W1, W2, h→h all replicated; inputs, targets sharded
_PMAP_EVAL_HYBRID = jax.pmap(
    _pmap_eval_batch_hybrid,
    axis_name='devices',
    in_axes=(None, None, None, None, None, None, 0, 0, None, None),
    # W1, W2, hh_from, hh_to, hh_weights, hh_valid, inputs, targets, activate_time, num_hidden
    static_broadcasted_argnums=(8, 9),  # activate_time, num_hidden are static
)


# ============================================================================
# H→H Aggregation Helper Functions (Segment Operations)
# ============================================================================

def segment_sum_2d(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    """Segment sum for batched 2D data.

    Args:
        data: Input data, shape (n_samples, n_elements)
        segment_ids: Segment IDs for each element, shape (n_elements,)
        num_segments: Number of output segments (positions)

    Returns:
        Summed segments, shape (n_samples, num_segments)
    """
    n_samples = data.shape[0]
    result = jnp.zeros((n_samples, num_segments), dtype=data.dtype)
    # Use scatter_add for sum - this is what JAX optimizes best
    return result.at[:, segment_ids].add(data)


def segment_max_2d(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Segment max for batched 2D data.

    Args:
        data: Input data, shape (n_samples, n_elements)
        segment_ids: Segment IDs for each element, shape (n_elements,)
        num_segments: Number of output segments (positions)
        valid_mask: Boolean mask for valid elements, shape (n_elements,)

    Returns:
        Max per segment, shape (n_samples, num_segments)
    """
    n_samples = data.shape[0]
    # Initialize with -inf so max works correctly
    result = jnp.full((n_samples, num_segments), -jnp.inf, dtype=data.dtype)
    # Mask invalid data to -inf
    masked_data = jnp.where(valid_mask[None, :], data, -jnp.inf)
    # JAX scatter max
    result = result.at[:, segment_ids].max(masked_data)
    # Replace -inf with 0 for segments with no contributions
    return jnp.where(result == -jnp.inf, 0.0, result)


def segment_min_2d(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Segment min for batched 2D data.

    Args:
        data: Input data, shape (n_samples, n_elements)
        segment_ids: Segment IDs for each element, shape (n_elements,)
        num_segments: Number of output segments (positions)
        valid_mask: Boolean mask for valid elements, shape (n_elements,)

    Returns:
        Min per segment, shape (n_samples, num_segments)
    """
    n_samples = data.shape[0]
    # Initialize with +inf so min works correctly
    result = jnp.full((n_samples, num_segments), jnp.inf, dtype=data.dtype)
    # Mask invalid data to +inf
    masked_data = jnp.where(valid_mask[None, :], data, jnp.inf)
    # JAX scatter min
    result = result.at[:, segment_ids].min(masked_data)
    # Replace +inf with 0 for segments with no contributions
    return jnp.where(result == jnp.inf, 0.0, result)


def segment_count_2d(
    segment_ids: jnp.ndarray,
    num_segments: int,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Count valid elements per segment.

    Args:
        segment_ids: Segment IDs for each element, shape (n_elements,)
        num_segments: Number of output segments (positions)
        valid_mask: Boolean mask for valid elements, shape (n_elements,)

    Returns:
        Count per segment, shape (num_segments,)
    """
    counts = jnp.zeros((num_segments,), dtype=jnp.float32)
    valid_counts = valid_mask.astype(jnp.float32)
    return counts.at[segment_ids].add(valid_counts)


def scatter_aggregate_by_target(
    contributions: jnp.ndarray,
    target_indices: jnp.ndarray,
    num_positions: int,
    hh_agg_indices: jnp.ndarray,
    valid_mask: jnp.ndarray,
    num_aggregations: int = 4,
) -> jnp.ndarray:
    """Aggregate H→H contributions with per-node aggregation function.

    This function replaces simple scatter_add with per-node aggregation,
    enabling different aggregation modes (sum, mean, max, min, etc.) for
    different hidden nodes in H→H connections.

    Args:
        contributions: H→H contributions, shape (n_samples, max_sparse_conns)
        target_indices: Target node indices, shape (max_sparse_conns,)
        num_positions: Number of hidden positions
        hh_agg_indices: Per-node aggregation indices, shape (num_positions,)
            Values in [0, num_aggregations-1] mapping to AGGREGATION_LIST
        valid_mask: Boolean mask for valid connections, shape (max_sparse_conns,)
        num_aggregations: Number of aggregation functions available

    Returns:
        Aggregated deltas, shape (n_samples, num_positions)

    Note:
        Uses segment operations for sum (efficient), manual masking for others.
        Aggregation types from AGGREGATION_LIST:
        - 0: sum (default, fastest)
        - 1: mean (sum / count)
        - 2: max
        - 3: min
        - 4: product (approximated as sum for now)
        - 5: maxabs (max of absolute values)
    """
    n_samples = contributions.shape[0]

    # Pre-compute aggregation results for all types
    # Sum is always computed as base
    sum_result = segment_sum_2d(contributions, target_indices, num_positions)

    # Start with sum as default
    result = sum_result

    if num_aggregations > 1:
        # Mean: sum / count
        counts = segment_count_2d(target_indices, num_positions, valid_mask)
        safe_counts = jnp.maximum(counts, 1.0)  # Avoid division by zero
        mean_result = sum_result / safe_counts[None, :]

        # Apply mean where agg_indices == 1
        mean_mask = (hh_agg_indices == 1)  # (num_positions,)
        result = jnp.where(mean_mask[None, :], mean_result, result)

    if num_aggregations > 2:
        # Max
        max_result = segment_max_2d(contributions, target_indices, num_positions, valid_mask)
        max_mask = (hh_agg_indices == 2)
        result = jnp.where(max_mask[None, :], max_result, result)

    if num_aggregations > 3:
        # Min
        min_result = segment_min_2d(contributions, target_indices, num_positions, valid_mask)
        min_mask = (hh_agg_indices == 3)
        result = jnp.where(min_mask[None, :], min_result, result)

    # Note: product (4) and maxabs (5) fall back to sum
    # These are harder to implement efficiently with segment operations
    # and are rarely used in practice

    return result


# ============================================================================
# Per-device caches for multi-GPU execution
# ============================================================================

# Cache for device-specific vmap'd extract_sparse functions
_extract_sparse_cache = {}
_extract_sparse_cache_lock = threading.Lock()


def _get_extract_sparse_fn(device_id: int, num_active: int, max_sparse_conns: int):
    """Get or create a device-specific extract_sparse JIT+vmap function.

    This avoids JAX trace caching issues in multi-GPU settings by ensuring
    each device has its own JIT-compiled traced function.

    The function is wrapped in BOTH jax.vmap AND jax.jit to ensure
    device-specific compilation.
    """
    cache_key = (device_id, num_active, max_sparse_conns)

    with _extract_sparse_cache_lock:
        if cache_key not in _extract_sparse_cache:
            devices = jax.devices()
            device = devices[device_id] if device_id < len(devices) else devices[0]

            with jax.default_device(device):
                def extract_sparse_for_genome(mask, weights, active_idx):
                    """Extract sparse connections from dense mask/weights."""
                    flat_mask = mask.flatten()
                    flat_weights = weights.flatten()

                    # Get indices of valid connections
                    num_valid_max = min(max_sparse_conns, flat_mask.shape[0])
                    valid_indices = jnp.nonzero(flat_mask, size=num_valid_max, fill_value=-1)[0]

                    # Convert flat index back to (from, to) in local coords
                    from_idx_local = valid_indices // num_active
                    to_idx_local = valid_indices % num_active

                    # Convert local indices to global position indices
                    from_idx_global = jnp.where(
                        valid_indices >= 0,
                        active_idx[from_idx_local],
                        -1
                    )
                    to_idx_global = jnp.where(
                        valid_indices >= 0,
                        active_idx[to_idx_local],
                        -1
                    )

                    # Get weights
                    conn_weights = jnp.where(
                        valid_indices >= 0,
                        flat_weights[valid_indices],
                        0.0
                    )

                    # Pad to max_sparse_conns if needed
                    if num_valid_max < max_sparse_conns:
                        pad_size = max_sparse_conns - num_valid_max
                        from_idx_global = jnp.pad(from_idx_global, (0, pad_size), constant_values=-1)
                        to_idx_global = jnp.pad(to_idx_global, (0, pad_size), constant_values=-1)
                        conn_weights = jnp.pad(conn_weights, (0, pad_size), constant_values=0.0)

                    return from_idx_global, to_idx_global, conn_weights

                # Create JIT-compiled vmap'd version within device context
                # The JIT ensures device-specific compilation
                vmap_fn = jax.jit(jax.vmap(extract_sparse_for_genome))
                _extract_sparse_cache[cache_key] = vmap_fn

        return _extract_sparse_cache[cache_key]


# Cache for device-specific constraint mask functions
_constraint_mask_cache = {}
_constraint_mask_cache_lock = threading.Lock()


def _get_constraint_mask_fn(device_id: int, allow_backward: bool, allow_lateral: bool, allow_self_loops: bool):
    """Get or create a device-specific constraint mask JIT function.

    This avoids JAX trace caching issues in multi-GPU settings by ensuring
    each device has its own JIT-compiled traced function.
    """
    cache_key = (device_id, allow_backward, allow_lateral, allow_self_loops)

    with _constraint_mask_cache_lock:
        if cache_key not in _constraint_mask_cache:
            devices = jax.devices()
            device = devices[device_id] if device_id < len(devices) else devices[0]

            with jax.default_device(device):
                @jax.jit
                def compute_constraint_mask(source_coords, target_positions):
                    """Compute Y-validity and self-loop masks.

                    Uses closure-captured config values to avoid trace-time issues.

                    Logic:
                    - Start with all True
                    - If not allow_backward: restrict to y_source <= y_target (forward + lateral)
                    - If not allow_lateral: further restrict to y_source != y_target (exclude same-y)

                    This ensures:
                    - hidden_only (backward=F, lateral=F): forward only (y < y)
                    - with_lateral (backward=F, lateral=T): forward + lateral (y <= y)
                    - with_backward (backward=T, lateral=F): forward + backward (y != y when combined)
                    - full_recurrent (all T): everything allowed
                    """
                    # Get Y coordinates
                    y_sources = source_coords[:, 1]
                    y_targets = target_positions[:, 1]

                    # Initialize with all True
                    y_valid = jnp.ones((source_coords.shape[0], target_positions.shape[0]), dtype=bool)

                    # Use <= instead of < to include lateral when not excluding backward
                    if not allow_backward:
                        y_valid = y_valid & (y_sources[:, None] <= y_targets[None, :])
                    if not allow_lateral:
                        y_valid = y_valid & (y_sources[:, None] != y_targets[None, :])

                    # Self-loop mask
                    coord_match = jnp.all(
                        source_coords[:, None, :] == target_positions[None, :, :],
                        axis=-1
                    )

                    if allow_self_loops:
                        # Add self-loops to y_valid, not just self_mask
                        # Self-loops have y==y, which would be excluded by y_valid otherwise
                        y_valid = y_valid | coord_match
                        self_mask = jnp.ones((source_coords.shape[0], target_positions.shape[0]), dtype=bool)
                    else:
                        self_mask = ~coord_match

                    return y_valid, self_mask

                _constraint_mask_cache[cache_key] = compute_constraint_mask

        return _constraint_mask_cache[cache_key]


# ============================================================================
# Level-by-Level Streaming Helper Functions
# ============================================================================
# These functions enable memory-efficient processing of hierarchical grids by
# querying CPPN positions one level at a time, reducing peak memory from
# O(total_positions) to O(max_level_size).
#
# Memory reduction example (depth 7):
#   Standard: ~87K positions at once → 10-20 GB on CPU
#   Streaming: ~65K positions max (largest level) → 2-3 GB on CPU
#
# Trade-off: 3-10x slower per generation due to Python loop overhead.
# ============================================================================


def query_level_positions(
    state: Any,
    cppns_transformed: Tuple,
    source_coords: jnp.ndarray,
    level_positions: jnp.ndarray,
    outgoing: bool,
    cppn_forward: Any,
    pop_chunk_size: int = 50,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
) -> jnp.ndarray:
    """Query CPPN for a single level's positions.

    This is a level-specific version of batch_query_population_multi_source_chunked
    that queries only the positions for a single hierarchical level.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        level_positions: Positions for this level only (num_level_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Number of genomes to process at once
        device_id: Device index for multi-GPU
        geometry_seeding_enabled: If True, add delta inputs to CPPN

    Returns:
        (pop_size, num_sources, num_level_positions) array of CPPN outputs
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

            chunk_weights = batch_query_population_positions(
                state, chunk_cppns, source_coord, level_positions,
                outgoing, cppn_forward,
                device_id=device_id,
                geometry_seeding_enabled=geometry_seeding_enabled
            )
            chunk_results.append(chunk_weights)

        # Concatenate chunks for this source
        source_weights = jnp.concatenate(chunk_results, axis=0)
        results_list.append(source_weights)

    # Stack and transpose
    result = jnp.stack(results_list, axis=0)
    # (num_sources, pop_size, num_positions) -> (pop_size, num_sources, num_positions)
    return jnp.transpose(result, (1, 0, 2))


def compute_single_level_variance(
    next_level_weights: jnp.ndarray,
    grid: 'HierarchicalGridStructure',
    level: int,
) -> jnp.ndarray:
    """Compute variance for a single level using weights from next level.

    This is a streaming version of variance computation that only requires
    the next level's weights, not all weights at once.

    Args:
        next_level_weights: CPPN outputs for level+1 (pop_size, num_next_positions)
        grid: Pre-computed grid structure
        level: Current level (0-indexed)

    Returns:
        Variance array (pop_size, num_cells_at_level)
    """
    pop_size = next_level_weights.shape[0]

    if level == 0:
        # Level 0: variance over the 4 cells from level 1
        # Level 1 has 16 cells, grouped into 4 blocks of 4
        child_grid_size = 4  # Level 1 is 4x4
        parent_grid_size = 2  # Level 0 is 2x2

        child_grids = next_level_weights.reshape(pop_size, child_grid_size, child_grid_size)
        reshaped = child_grids.reshape(pop_size, parent_grid_size, 2, parent_grid_size, 2)
        reshaped = reshaped.transpose(0, 1, 3, 2, 4)
        blocks = reshaped.reshape(pop_size, parent_grid_size, parent_grid_size, 4)
        variances = jnp.var(blocks, axis=-1)
        return variances.reshape(pop_size, parent_grid_size * parent_grid_size)
    else:
        # Higher levels: variance of 2x2 child blocks from next level
        next_level_size = grid.level_sizes_static[level + 1]
        child_grid_size = int(np.sqrt(next_level_size))
        parent_grid_size = child_grid_size // 2

        child_grids = next_level_weights.reshape(pop_size, child_grid_size, child_grid_size)
        reshaped = child_grids.reshape(pop_size, parent_grid_size, 2, parent_grid_size, 2)
        reshaped = reshaped.transpose(0, 1, 3, 2, 4)
        blocks = reshaped.reshape(pop_size, parent_grid_size, parent_grid_size, 4)
        variances = jnp.var(blocks, axis=-1)
        return variances.reshape(pop_size, parent_grid_size * parent_grid_size)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class SparseHiddenConnections:
    """Sparse representation of hidden→hidden connections for a population.

    This dataclass stores sparse connections in padded arrays with validity masks,
    enabling vmap over the population dimension.

    Provides type-safe storage.

    Attributes:
        from_indices: (pop, max_sparse_conns) - source position indices
        to_indices: (pop, max_sparse_conns) - target position indices
        weights: (pop, max_sparse_conns) - connection weights
        valid_mask: (pop, max_sparse_conns) - True for valid connections
        num_valid: (pop,) - actual number of valid connections per genome
    """
    from_indices: jnp.ndarray
    to_indices: jnp.ndarray
    weights: jnp.ndarray
    valid_mask: jnp.ndarray
    num_valid: jnp.ndarray


@dataclass
class EMRConfig:
    """Complete configuration for unified extended EMR-HyperNEAT.

    Combines RecurrenceConfig (unified.py) with additional multi-hop settings.

    Attributes:
        # From RecurrenceConfig (unified.py)
        enabled: Master switch for recurrence features
        allow_hidden_to_hidden: Enable Phase 2 discovery (iteration_level > 0)
        allow_backward: Enable backward connections (y_source > y_target)
        allow_lateral: Enable same-layer connections (y_source == y_target)
        allow_self_loops: Enable self-connections (same x,y)
        iteration_level: Number of hops for multi-hop expansion
        activate_time: Forward pass iterations for signal propagation
        max_connections: Max connections per substrate (for vmap padding)

        use_vectorized_discovery: Toggle between vectorized and iterative discovery
        max_sparse_conns: Maximum sparse connections to store per genome

        # NEW: Multi-hop settings
        multi_hop_algorithm: Algorithm for multi-hop expansion ("matrix_power" or "fori_loop")
        hop_decay_factor: Weight decay per hop (prevents exploding weights)

        # Caching settings
        hh_cache_enabled: Enable/disable h→h caching
        hh_refresh_interval: Refresh every N generations
        hh_mask_change_threshold: Refresh if mask changes more than this fraction
    """
    # From RecurrenceConfig
    enabled: bool = True
    allow_hidden_to_hidden: bool = True
    allow_backward: bool = True
    allow_lateral: bool = True
    allow_self_loops: bool = True
    iteration_level: int = 2
    activate_time: Optional[int] = None  # None = auto-compute from depth
    max_connections: int = 10000

    use_vectorized_discovery: bool = True
    max_sparse_conns: int = 10000

    # NEW: Multi-hop settings
    multi_hop_algorithm: str = "matrix_power"  # "matrix_power" or "fori_loop"
    hop_decay_factor: float = 0.8  # Weight decay per hop

    # Caching settings
    hh_cache_enabled: bool = True
    hh_refresh_interval: int = 5
    hh_mask_change_threshold: float = 0.1

    # Discovery algorithm toggle
    # - False (default): Sparse per-genome filtering (better for full_recurrent)
    # - True: Dense discovery without per-genome filtering (better for hybrid at low pop)
    use_dense_discovery: bool = False

    # Evaluation chunking to prevent OOM on large populations
    # - None or 0: No chunking (process entire population at once via vmap)
    # - > 0: Process population in chunks of this size
    # Recommended: 100-200 for depth >= 6, 50-100 for depth >= 7
    eval_chunk_size: Optional[int] = None

    # Multi-hop discovery chunking to prevent OOM on large populations
    # - None: Auto-compute based on num_active positions
    # - 0: No chunking (process entire population at once via vmap)
    # - > 0: Process population in chunks of this size
    # The multi-hop expansion creates (pop_size, num_active, num_active) intermediates
    # which can cause OOM at high populations with iteration_level > 1
    multi_hop_chunk_size: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.iteration_level < 0:
            raise ValueError(f"iteration_level must be >= 0, got {self.iteration_level}")
        if self.max_connections < 100:
            raise ValueError(f"max_connections must be >= 100, got {self.max_connections}")
        if self.multi_hop_algorithm not in ("matrix_power", "fori_loop"):
            raise ValueError(f"multi_hop_algorithm must be 'matrix_power' or 'fori_loop', got {self.multi_hop_algorithm}")
        if not (0.0 < self.hop_decay_factor <= 1.0):
            raise ValueError(f"hop_decay_factor must be in (0, 1], got {self.hop_decay_factor}")


@dataclass
class EMRRecurrenceMetrics:
    """Extended metrics for A/B testing recurrence configurations.

    Includes all metrics from RecurrenceMetrics plus multi-hop specific stats.
    """
    # Connection statistics
    total_connections: int = 0
    forward_connections: int = 0
    backward_connections: int = 0
    lateral_connections: int = 0
    self_loop_connections: int = 0

    # Hidden→hidden specific
    hidden_to_hidden_connections: int = 0
    direct_hh_connections: int = 0  # Before multi-hop expansion

    # Topology metrics
    avg_in_degree: float = 0.0
    avg_out_degree: float = 0.0
    has_cycles: bool = False

    # Performance metrics
    fitness: float = 0.0
    generations_to_solve: Optional[int] = None
    activate_time_used: int = 0

    # Timing breakdown (milliseconds) - fine-grained step timing
    time_cppn_ask_ms: float = 0.0        # CPPN population ask + transform
    time_cppn_query_ms: float = 0.0      # CPPN queries for I/O weights
    time_variance_ms: float = 0.0         # Variance computation + masks
    time_hh_discovery_ms: float = 0.0     # h→h discovery
    time_build_matrices_ms: float = 0.0   # W1/W2 matrix construction
    time_evaluation_ms: float = 0.0       # Forward pass evaluation
    time_neat_evolution_ms: float = 0.0   # NEAT tell() evolution

    # Legacy timing fields (kept for compatibility)
    substrate_discovery_ms: float = 0.0
    phase2_discovery_ms: float = 0.0
    forward_pass_ms: float = 0.0

    # Multi-hop specific
    multi_hop_connections_added: int = 0
    effective_iteration_level: int = 0
    transitive_closure_density: float = 0.0
    multi_hop_discovery_ms: float = 0.0

    # Cache status
    cache_hit: bool = False
    cache_hit_count: int = 0  # Cumulative hits across generations
    cache_refresh_count: int = 0

    # Substrate efficiency metrics
    active_positions_mean: float = 0.0
    active_positions_min: int = 0
    active_positions_max: int = 0
    total_positions: int = 0
    position_utilization: float = 0.0  # active / total ratio

    # Weight matrix density
    w1_density: float = 0.0  # Non-zero ratio in W1
    w2_density: float = 0.0  # Non-zero ratio in W2

    # Configuration echo (actual values used)
    variance_threshold_used: float = 0.0
    band_threshold_used: float = 0.0
    max_weight_used: float = 0.0

    # Population diversity (from last generation)
    min_fitness: float = 0.0
    max_fitness: float = 0.0
    std_fitness: float = 0.0


# ============================================================================
# Presets
# ============================================================================

RECURRENCE_PRESETS: Dict[str, EMRConfig] = {
    'feedforward': EMRConfig(
        enabled=False,
        allow_hidden_to_hidden=False,
        allow_backward=False,
        allow_lateral=False,
        allow_self_loops=False,
        iteration_level=0,
    ),
    'hidden_only': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=False,
        allow_lateral=False,
        allow_self_loops=False,
        iteration_level=2,
    ),
    'with_backward': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=True,
        allow_lateral=False,
        allow_self_loops=False,
        iteration_level=2,
    ),
    'with_lateral': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=False,
        allow_lateral=True,
        allow_self_loops=False,
        iteration_level=2,
    ),
    'with_self': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=False,
        allow_lateral=False,
        allow_self_loops=True,
        iteration_level=2,
    ),
    'full_recurrent': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=True,
        allow_lateral=True,
        allow_self_loops=True,
        iteration_level=2,
        activate_time=20,
    ),
}


def get_recurrence_preset(name: str) -> EMRConfig:
    """Get a predefined recurrence configuration by name.

    Args:
        name: Preset name (feedforward, hidden_only, with_backward,
              with_lateral, with_self, full_recurrent)

    Returns:
        EMRConfig with the preset settings

    Raises:
        ValueError: If preset name is not recognized
    """
    if name not in RECURRENCE_PRESETS:
        valid = ', '.join(RECURRENCE_PRESETS.keys())
        raise ValueError(f"Unknown recurrence preset '{name}'. Valid presets: {valid}")

    # Return a copy to prevent mutation
    preset = RECURRENCE_PRESETS[name]
    return EMRConfig(
        enabled=preset.enabled,
        allow_hidden_to_hidden=preset.allow_hidden_to_hidden,
        allow_backward=preset.allow_backward,
        allow_lateral=preset.allow_lateral,
        allow_self_loops=preset.allow_self_loops,
        iteration_level=preset.iteration_level,
        activate_time=preset.activate_time,
        max_connections=preset.max_connections,
        use_vectorized_discovery=preset.use_vectorized_discovery,
        max_sparse_conns=preset.max_sparse_conns,
        multi_hop_algorithm=preset.multi_hop_algorithm,
        hop_decay_factor=preset.hop_decay_factor,
        hh_cache_enabled=preset.hh_cache_enabled,
        hh_refresh_interval=preset.hh_refresh_interval,
        hh_mask_change_threshold=preset.hh_mask_change_threshold,
    )


# ============================================================================
# Multi-Output CPPN Query Functions
# ============================================================================
# These functions handle CPPNs with multiple outputs (weight + secondary channels).
# The key difference from standard query functions is that they preserve
# the output dimension rather than flattening it.
# ============================================================================

# Per-device cache for multi-output query functions
_per_device_multi_output_vmap_cache: Dict[Tuple[int, int], Any] = {}
_per_device_multi_output_cache_lock = threading.Lock()


def _create_multi_output_query_fn_for_device(cppn_forward: Any, device_id: int):
    """Create a fresh multi-output query function bound to a specific device.

    Unlike the standard query function, this one does NOT flatten the output,
    preserving the (num_outputs,) dimension for each position.

    Args:
        cppn_forward: The CPPN forward function
        device_id: Target device index

    Returns:
        A fresh JIT-compiled query function for this device that returns
        (pop_size, num_positions, num_outputs) arrays
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    with jax.default_device(device):
        def query_population_vmap_impl(state, inputs, cppns_nodes, cppns_conns,
                                       cppns_conn_attrs, cppns_node_attrs):
            """Query all CPPNs at all positions, preserving output dimension."""
            def query_single_cppn(cppn_tuple):
                """Query one CPPN at all positions."""
                # DON'T flatten - preserve (num_positions, num_outputs) shape
                return jax.vmap(
                    lambda x: cppn_forward(state, cppn_tuple, x)
                )(inputs)

            return jax.vmap(
                query_single_cppn,
                in_axes=((0, 0, 0, 0),)
            )((cppns_nodes, cppns_conns, cppns_conn_attrs, cppns_node_attrs))

        return jax.jit(query_population_vmap_impl, device=device)


def _get_multi_output_query_fn(device_id: int, cppn_forward: Any):
    """Get or create a device-specific multi-output query function."""
    cache_key = (device_id, id(cppn_forward))

    with _per_device_multi_output_cache_lock:
        if cache_key not in _per_device_multi_output_vmap_cache:
            _per_device_multi_output_vmap_cache[cache_key] = _create_multi_output_query_fn_for_device(
                cppn_forward, device_id
            )
        return _per_device_multi_output_vmap_cache[cache_key]


def query_cppn_multi_output(
    cppn_forward: Any,
    state: Any,
    inputs: jnp.ndarray,
    cppns_transformed: Tuple,
    device_id: int = 0,
) -> jnp.ndarray:
    """Query CPPN preserving the output dimension (weight + secondary channels).

    Args:
        cppn_forward: JIT-compiled CPPN forward function
        state: Algorithm state
        inputs: Input coordinates (num_positions, input_dim)
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        device_id: Device index for multi-GPU

    Returns:
        Array of shape (pop_size, num_positions, 2) where:
        - [:, :, 0] are weight outputs
        - [:, :, 1] are secondary outputs
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Move all inputs to target device
    state_on_device = jax.tree.map(lambda x: jax.device_put(x, device), state)
    inputs_on_device = jax.device_put(inputs, device)
    cppns_on_device = tuple(jax.device_put(arr, device) for arr in cppns_transformed)

    # Get device-specific multi-output query function
    query_fn = _get_multi_output_query_fn(device_id, cppn_forward)

    with jax.default_device(device):
        return query_fn(
            state_on_device,
            inputs_on_device,
            cppns_on_device[0],
            cppns_on_device[1],
            cppns_on_device[2],
            cppns_on_device[3]
        )


def batch_query_population_positions_multi_output(
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
    """Query CPPN preserving the output dimension (weight + secondary channels).

    This is the multi-output version of batch_query_population_positions.
    Instead of returning (pop_size, num_positions), it returns
    (pop_size, num_positions, 2) with weight and secondary channels.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coord: Single source coordinate (2,)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        device_id: Device index for multi-GPU
        geometry_seeding_enabled: If True, add delta_x, delta_y to CPPN inputs (7D instead of 5D)
        agent_coord: Optional (2,) agent team position for swarm mode.
            When provided, prepends [agent_x, agent_y] to CPPN inputs
            for position-aware multi-agent policy generation.

    Returns:
        (pop_size, num_positions, 2) array where:
        - [:, :, 0] are weight outputs
        - [:, :, 1] are secondary outputs
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    source_coord = jax.device_put(source_coord, device)
    target_positions = jax.device_put(target_positions, device)

    num_positions = target_positions.shape[0]

    # Build CPPN inputs
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

    inputs = jax.device_put(inputs_np, device)

    return query_cppn_multi_output(
        cppn_forward, state, inputs, cppns_transformed, device_id
    )


def batch_query_population_multi_source_multi_output(
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
    """Query CPPN from multiple sources, preserving the output dimension.

    This is the multi-output version of batch_query_population_multi_source_chunked.
    Returns weight and secondary channels for each source->target pair.

    Args:
        state: Algorithm state
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...)
        source_coords: Multiple source coordinates (num_sources, 2)
        target_positions: All target positions (num_positions, 2)
        outgoing: If True, source->target; if False, target->source
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Population chunk size for memory management
        device_id: Device index for multi-GPU
        geometry_seeding_enabled: If True, add delta_x, delta_y to CPPN inputs (7D instead of 5D)
        agent_coord: Optional (2,) agent team position for swarm mode.
            When provided, prepends [agent_x, agent_y] to CPPN inputs.

    Returns:
        (pop_size, num_sources, num_positions, 2) array where:
        - [:, :, :, 0] are weight outputs
        - [:, :, :, 1] are secondary outputs
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    source_coords = jax.device_put(source_coords, device)
    target_positions = jax.device_put(target_positions, device)
    cppns_transformed = tuple(jax.device_put(arr, device) for arr in cppns_transformed)

    pop_size = cppns_transformed[0].shape[0]
    num_sources = source_coords.shape[0]

    results_list = []

    for source_idx in range(num_sources):
        source_coord = source_coords[source_idx]

        chunk_results = []
        for chunk_start in range(0, pop_size, pop_chunk_size):
            chunk_end = min(chunk_start + pop_chunk_size, pop_size)

            chunk_cppns = (
                cppns_transformed[0][chunk_start:chunk_end],
                cppns_transformed[1][chunk_start:chunk_end],
                cppns_transformed[2][chunk_start:chunk_end],
                cppns_transformed[3][chunk_start:chunk_end],
            )

            chunk_weights_multi = batch_query_population_positions_multi_output(
                state, chunk_cppns, source_coord, target_positions,
                outgoing, cppn_forward, device_id, geometry_seeding_enabled,
                agent_coord=agent_coord,
            )
            chunk_weights_multi = jax.device_put(chunk_weights_multi, device)
            chunk_results.append(chunk_weights_multi)

        # Concatenate chunks
        chunk_results_np = [np.asarray(c) for c in chunk_results]
        source_weights_multi = np.concatenate(chunk_results_np, axis=0)
        results_list.append(source_weights_multi)

    # Stack: (num_sources, pop_size, num_positions, 2)
    # Transpose to: (pop_size, num_sources, num_positions, 2)
    stacked = np.stack(results_list, axis=0)
    return np.transpose(stacked, (1, 0, 2, 3))


def build_swarm_weight_matrices(
    state: Any,
    cppns_transformed: Tuple,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    all_positions: jnp.ndarray,
    team_positions: np.ndarray,
    cppn_forward: Any,
    variance_threshold: float,
    band_threshold: float,
    max_weight: float,
    geometry_seeding_enabled: bool = False,
    device_id: int = 0,
    h_grid: Any = None,
    use_neuromodulation: bool = False,
    use_dynamic_functions: bool = False,
    num_activations: int = 18,
) -> Tuple:
    """Build W1/W2 weight matrices for all agents from a single CPPN population.

    For each agent at team position (agent_x, agent_y), queries the CPPN with
    agent coordinates prepended to generate agent-specific substrates.

    This is the core of multi-agent policy geometry: same CPPN, different
    agent positions -> different weight matrices -> different policies.

    When h_grid is provided, applies hierarchical variance-based position
    filtering (matching the single-agent EMR-HyperNEAT path).

    Args:
        state: Algorithm state.
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...).
        input_coords: (num_inputs, 2) substrate input coordinates.
        output_coords: (num_outputs, 2) substrate output coordinates.
        all_positions: (total_positions, 2) all hierarchical grid positions.
        team_positions: (num_agents, 2) agent team positions in [-1, 1].
        cppn_forward: JIT-compiled CPPN forward function.
        variance_threshold: Threshold for variance-based position filtering.
        band_threshold: Weight threshold for connection pruning.
        max_weight: Maximum weight value for scaling.
        geometry_seeding_enabled: Whether geometry seeding is active.
        device_id: Device index for multi-GPU.
        h_grid: HierarchicalGridStructure for variance filtering (None = skip).
        use_neuromodulation: If True, compute receptor densities and base gains.
        use_dynamic_functions: If True, compute per-node activation indices.
        num_activations: Number of activation functions for dynamic selection.

    Returns:
        Tuple of (W1_agents, W2_agents, receptor_densities, base_gains, act_indices):
            W1_agents: (pop_size, num_agents, num_inputs, total_positions)
            W2_agents: (pop_size, num_agents, total_positions, num_outputs)
            receptor_densities: (pop_size, num_agents, total_positions, 4) or None
            base_gains: (pop_size, num_agents, total_positions) or None
            act_indices: (pop_size, num_agents, total_positions) or None
    """
    num_agents = team_positions.shape[0]
    pop_size = cppns_transformed[0].shape[0]
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]
    total_positions = all_positions.shape[0]

    W1_all = []
    W2_all = []
    receptor_densities_list = [] if use_neuromodulation else None
    base_gains_list = [] if use_neuromodulation else None
    act_indices_list = [] if use_dynamic_functions else None

    for agent_idx in range(num_agents):
        agent_coord = np.asarray(team_positions[agent_idx])

        # Query CPPN for this agent's input->hidden weights
        # Returns: (pop_size, num_inputs, total_positions)
        input_weights = batch_query_population_multi_source_chunked(
            state, cppns_transformed, input_coords, all_positions,
            True, cppn_forward, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled,
            agent_coord=agent_coord,
        )

        # Query CPPN for this agent's hidden->output weights
        # Returns: (pop_size, num_outputs, total_positions)
        output_weights = batch_query_population_multi_source_chunked(
            state, cppns_transformed, output_coords, all_positions,
            False, cppn_forward, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled,
            agent_coord=agent_coord,
        )

        # --- Variance filtering (EMR-HyperNEAT adaptive resolution) ---
        if h_grid is not None and variance_threshold > 0:
            variance_source = jnp.array(input_weights[:, 0, :])
            level_variances = compute_hierarchical_variances_batch(variance_source, h_grid)
            masks_A = compute_subdivision_masks_batch(
                level_variances, variance_threshold, h_grid, return_all_masks=False
            )
            masks_np = np.asarray(masks_A)  # (pop_size, total_positions)
        else:
            masks_np = None

        # --- Neuromodulation data (from raw weights, before scaling) ---
        if use_neuromodulation:
            weight_jax = jnp.array(input_weights[:, 0, :])
            # NaN protection: evolved CPPNs can produce NaN for some genomes.
            # Replace NaN inputs with 0.0 before derivation to get neutral receptors.
            weight_jax_safe = jnp.where(jnp.isnan(weight_jax), 0.0, weight_jax)
            receptor_d = derive_receptor_from_weight(weight_jax_safe, method='tanh', num_nt_types=4)
            weight_norm = jnp.tanh(weight_jax_safe)
            base_g = jnp.abs(weight_norm) * 0.5 + 0.5
            receptor_densities_list.append(np.asarray(receptor_d))
            base_gains_list.append(np.asarray(base_g))

        # --- Dynamic function indices (from raw weights, before scaling) ---
        if use_dynamic_functions:
            act_raw = jnp.array(input_weights[:, 0, :])
            act_raw_safe = jnp.where(jnp.isnan(act_raw), 0.0, act_raw)
            act_idx = continuous_to_index(act_raw_safe, num_activations)
            act_indices_list.append(np.asarray(act_idx))

        # --- Weight scaling and thresholding ---
        weight_thresh = 0.1
        input_weights = np.tanh(input_weights) * max_weight
        output_weights = np.tanh(output_weights) * max_weight

        # Apply variance mask (zero out inactive positions)
        if masks_np is not None:
            input_weights = np.where(masks_np[:, None, :], input_weights, 0.0)
            output_weights = np.where(masks_np[:, None, :], output_weights, 0.0)

        # Apply weight threshold (zero out weak connections)
        input_weights = np.where(np.abs(input_weights) > weight_thresh, input_weights, 0.0)
        output_weights = np.where(np.abs(output_weights) > weight_thresh, output_weights, 0.0)

        W1_all.append(input_weights)
        W2_all.append(output_weights)

    # Stack and transpose to (pop_size, num_agents, ...)
    W1_agents = np.transpose(np.stack(W1_all, axis=0), (1, 0, 2, 3))
    W2_stacked = np.transpose(np.stack(W2_all, axis=0), (1, 0, 2, 3))
    W2_agents = np.transpose(W2_stacked, (0, 1, 3, 2))

    # Stack optional neuromodulation outputs
    if use_neuromodulation:
        rd_stacked = np.stack(receptor_densities_list, axis=0)
        receptor_densities_all = np.transpose(rd_stacked, (1, 0, 2, 3))
        bg_stacked = np.stack(base_gains_list, axis=0)
        base_gains_all = np.transpose(bg_stacked, (1, 0, 2))
    else:
        receptor_densities_all = None
        base_gains_all = None

    # Stack optional dynamic function outputs
    if use_dynamic_functions:
        ai_stacked = np.stack(act_indices_list, axis=0)
        act_indices_all = np.transpose(ai_stacked, (1, 0, 2))
    else:
        act_indices_all = None

    return W1_agents, W2_agents, receptor_densities_all, base_gains_all, act_indices_all


# ============================================================================
# Connection Constraint Filtering
# ============================================================================

def get_connection_constraint_mask(
    source_coord: jnp.ndarray,
    target_positions: jnp.ndarray,
    config: Optional[EMRConfig] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute connection constraint masks based on recurrence configuration.

    Connection Types:
    - Forward (y_source < y_target): Always allowed
    - Lateral (y_source == y_target): Configurable via allow_lateral
    - Backward (y_source > y_target): Configurable via allow_backward
    - Self-loop (x,y identical): Configurable via allow_self_loops

    Args:
        source_coord: Source coordinate - shape (2,)
        target_positions: Target positions - shape (num_targets, 2)
        config: Configuration for recurrence features. If None,
            uses feedforward-only constraints

    Returns:
        Tuple of:
        - y_valid: (num_targets,) bool mask for y-constraint
        - self_mask: (num_targets,) bool mask for self-loop constraint
    """
    y_source = source_coord[1]
    y_targets = target_positions[:, 1]
    x_source = source_coord[0]
    x_targets = target_positions[:, 0]

    # Compute position relationship masks
    is_forward = y_source < y_targets
    is_same_y = jnp.abs(y_source - y_targets) < 1e-6
    is_backward = y_source > y_targets
    is_self = (jnp.abs(x_source - x_targets) < 1e-6) & is_same_y

    # Default to feedforward constraints if no config provided
    if config is None or not config.enabled:
        # Original behavior: y_source <= y_target, no self-loops
        y_valid = is_forward | is_same_y
        self_mask = ~is_self
        return y_valid, self_mask

    # Build y_valid mask based on config
    # Forward connections are always allowed
    y_valid = is_forward

    # Lateral connections (same layer, different position)
    if config.allow_lateral:
        y_valid = y_valid | (is_same_y & ~is_self)

    # Backward connections
    if config.allow_backward:
        y_valid = y_valid | is_backward

    # Self-loop handling
    if config.allow_self_loops:
        y_valid = y_valid | is_self
        self_mask = jnp.ones_like(is_self)  # Allow all
    else:
        self_mask = ~is_self  # Block self-loops

    return y_valid, self_mask


def get_connection_constraint_mask_batched(
    source_coords: jnp.ndarray,
    target_positions: jnp.ndarray,
    config: Optional[EMRConfig] = None,
    device_id: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batched version of get_connection_constraint_mask using per-device JIT.

    Uses a device-specific JIT cache to avoid trace caching issues in multi-GPU settings.

    Args:
        source_coords: Source coordinates - shape (num_sources, 2)
        target_positions: Target positions - shape (num_targets, 2)
        config: Configuration for recurrence features
        device_id: Target device index for multi-GPU execution

    Returns:
        Tuple of:
        - y_valid: (num_sources, num_targets) bool mask
        - self_mask: (num_sources, num_targets) bool mask
    """
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Extract config values
    allow_backward = config.allow_backward if config else True
    allow_lateral = config.allow_lateral if config else True
    allow_self_loops = config.allow_self_loops if config else False

    # Ensure inputs are on correct device
    source_coords = jax.device_put(source_coords, device)
    target_positions = jax.device_put(target_positions, device)

    # Get device-specific JIT function from cache
    compute_fn = _get_constraint_mask_fn(device_id, allow_backward, allow_lateral, allow_self_loops)

    # Call the device-specific JIT function
    y_valid, self_mask = compute_fn(source_coords, target_positions)

    # Ensure outputs are on correct device
    y_valid = jax.device_put(y_valid, device)
    self_mask = jax.device_put(self_mask, device)

    return y_valid, self_mask


# ============================================================================
# Multi-Hop Algorithm
# ============================================================================

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def compute_multi_hop_connections_matrix_power(
    adjacency_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    weight_threshold: float,
    iteration_level: int,
    hop_decay_factor: float,
    max_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute multi-hop connections via matrix power.

    For iteration_level k hops, computes:
    A_total = A + decay*A^2 + decay^2*A^3 + ... + decay^(k-1)*A^k

    This captures transitive connections: if A→B and B→C exist, we add A→C.

    IMPORTANT: This is JIT-compatible because iteration_level is a static argument,
    which means the Python loop compiles away into a fixed computation graph.

    Args:
        adjacency_weights: Raw CPPN-derived weights - shape (num_active, num_active)
        valid_mask: Boolean mask for direct connections - shape (num_active, num_active)
        weight_threshold: Minimum weight magnitude to keep
        iteration_level: Number of hops (static for JIT)
        hop_decay_factor: Weight decay per hop (e.g., 0.8 means 2-hop weights are 0.8x)
        max_weight: Maximum allowed weight magnitude

    Returns:
        Tuple of:
        - A_total: Multi-hop accumulated weights - shape (num_active, num_active)
        - multi_hop_valid: Boolean mask for multi-hop connections - shape (num_active, num_active)
    """
    # Initialize with direct connections (1-hop)
    A_current = jnp.where(valid_mask, adjacency_weights, 0.0)
    A_total = A_current.copy()

    # Static unroll - compiles away with JIT because iteration_level is static
    for hop in range(1, iteration_level):
        # Compute A^(hop+1) via matrix multiplication
        A_power = A_current @ adjacency_weights

        # Apply weight threshold for sparsity (prune weak multi-hop paths)
        A_power = jnp.where(jnp.abs(A_power) > weight_threshold, A_power, 0.0)

        # Decay weights by hop distance (prevents exploding accumulated weights)
        decay = hop_decay_factor ** hop
        A_power = A_power * decay

        # Accumulate (union of all hop distances)
        # Use max to prefer stronger direct connections over weaker indirect ones
        A_total = jnp.where(
            jnp.abs(A_power) > jnp.abs(A_total),
            A_power,
            A_total
        )
        # Alternative: simple addition (allows multiple paths to reinforce)
        # A_total = A_total + A_power

        A_current = A_power

    # Clip to max_weight
    A_total = jnp.clip(A_total, -max_weight, max_weight)

    # Create validity mask for non-zero connections
    multi_hop_valid = jnp.abs(A_total) > weight_threshold

    return A_total, multi_hop_valid


def compute_multi_hop_connections_fori_loop(
    adjacency_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    weight_threshold: float,
    iteration_level: int,
    hop_decay_factor: float,
    max_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute multi-hop connections using lax.fori_loop.

    Alternative to matrix_power that works with dynamic iteration_level.
    Slightly more overhead due to fori_loop.

    Args:
        (same as compute_multi_hop_connections_matrix_power)

    Returns:
        (same as compute_multi_hop_connections_matrix_power)
    """
    A_init = jnp.where(valid_mask, adjacency_weights, 0.0)

    def hop_step(hop_idx, carry):
        A_current, A_total = carry

        # Compute next power
        A_power = A_current @ adjacency_weights
        A_power = jnp.where(jnp.abs(A_power) > weight_threshold, A_power, 0.0)

        # Decay by hop distance
        decay = hop_decay_factor ** (hop_idx + 1)
        A_power = A_power * decay

        # Accumulate (prefer stronger)
        A_total = jnp.where(
            jnp.abs(A_power) > jnp.abs(A_total),
            A_power,
            A_total
        )

        return (A_power, A_total)

    _, A_total = lax.fori_loop(0, iteration_level - 1, hop_step, (A_init, A_init))

    A_total = jnp.clip(A_total, -max_weight, max_weight)
    multi_hop_valid = jnp.abs(A_total) > weight_threshold

    return A_total, multi_hop_valid


# ============================================================================
# Vectorized Discovery with Multi-Hop
# ============================================================================

# Per-device JIT cache for sparse discovery bitwise AND operations
# CRITICAL: Without this, JAX reuses cached JIT traces across devices
# causing "Buffer on device cuda:X but replica assigned to cuda:Y" errors
_sparse_discovery_jit_cache = {}


def _get_sparse_discovery_combine_fn(device):
    """Get or create a device-specific JIT function for combining validity masks.

    This prevents JAX from reusing JIT traces across devices which causes
    device placement errors in multi-GPU execution.
    """
    cache_key = (id(device), device.id if hasattr(device, 'id') else 0)

    if cache_key not in _sparse_discovery_jit_cache:
        def combine_validity_masks(source_valid, target_valid, weight_valid, constraint_mask):
            """Combine validity masks with bitwise AND.

            All inputs should be on the target device before calling.
            """
            return source_valid & target_valid & weight_valid & constraint_mask

        # JIT compile with explicit device pinning
        _sparse_discovery_jit_cache[cache_key] = jax.jit(combine_validity_masks, device=device)

    return _sparse_discovery_jit_cache[cache_key]


def discover_sparse_hh_vectorized_multi_hop(
    state: Any,
    cppns_transformed: Tuple,
    h_grid: HierarchicalGridStructure,
    masks_A: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    config: EMRConfig,
    cppn_forward: Any,
    pop_chunk_size: int = 100,
    verbose: bool = False,
    global_union_active: Optional[jnp.ndarray] = None,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    num_cppn_outputs: int = 1,
) -> SparseHiddenConnections:
    """Fully vectorized sparse h→h discovery WITH multi-hop expansion.

    Pipeline:
    1. Query CPPN for all active→active position pairs
    2. Apply Y-coordinate constraints (from config)
    3. Apply weight threshold for direct connections
    4. Expand to multi-hop connections via matrix power
    5. Convert dense multi-hop matrix to sparse format

    Args:
        state: Algorithm state
        cppns_transformed: Transformed CPPNs (pop_size, ...)
        h_grid: Hierarchical grid structure
        masks_A: Active position masks - shape (pop, total_positions)
        band_threshold: Weight threshold
        max_weight: Weight scaling
        config: Extended configuration with multi-hop settings
        cppn_forward: JIT-compiled CPPN forward function
        pop_chunk_size: Population chunk size for queries
        global_union_active: Optional pre-computed global union of active positions.
            If provided, used instead of computing from masks_A. This ensures
            deterministic behavior in multi-GPU settings where masks_A is sharded.

    Returns:
        SparseHiddenConnections with multi-hop expanded connections
    """
    pop_size = cppns_transformed[0].shape[0]
    total_positions = h_grid.total_positions
    max_sparse_conns = config.max_sparse_conns

    # CRITICAL: Move ALL input arrays to target device for multi-GPU execution
    # Without this, arrays stay on cuda:0 causing device placement errors
    # when this function is called from a cuda:1 thread
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]

    # Move h_grid positions to target device
    all_positions = jax.device_put(h_grid.all_positions, device)

    # Move masks_A to target device (critical for active_indices computation)
    masks_A = jax.device_put(masks_A, device)

    # Move cppns_transformed to target device
    cppns_transformed = tuple(jax.device_put(arr, device) for arr in cppns_transformed)

    # Move state to target device
    state = jax.tree.map(lambda x: jax.device_put(x, device), state)

    # Get union of active positions across population
    # Use global union if provided (for multi-GPU determinism), otherwise compute locally
    if global_union_active is not None:
        union_active = jax.device_put(global_union_active, device)
    else:
        union_active = jnp.any(masks_A, axis=0)  # (total_positions,)
        union_active = jax.device_put(union_active, device)
    active_indices = jnp.nonzero(union_active, size=total_positions, fill_value=-1)[0]
    # CRITICAL: Ensure active_indices is on correct device for subsequent indexing
    active_indices = jax.device_put(active_indices, device)
    num_active = int(jnp.sum(union_active))

    # DIAGNOSTIC LOGGING (override with DEBUG_PHASE2=1 env var)
    _debug_phase2 = os.environ.get('DEBUG_PHASE2', '0') == '1'
    if verbose or _debug_phase2:
        print(f"[H→H Discovery] iteration_level={config.iteration_level}, "
              f"multi_hop_algorithm={config.multi_hop_algorithm}, "
              f"hop_decay_factor={config.hop_decay_factor}")
        print(f"[H→H Discovery] allow_backward={config.allow_backward}, "
              f"allow_lateral={config.allow_lateral}, "
              f"allow_self_loops={config.allow_self_loops}")
        print(f"[H→H Discovery] num_active={num_active}, pop_size={pop_size}")

    if num_active == 0:
        # No active positions - return empty connections
        return SparseHiddenConnections(
            from_indices=jnp.full((pop_size, max_sparse_conns), -1, dtype=jnp.int32),
            to_indices=jnp.full((pop_size, max_sparse_conns), -1, dtype=jnp.int32),
            weights=jnp.zeros((pop_size, max_sparse_conns), dtype=jnp.float32),
            valid_mask=jnp.zeros((pop_size, max_sparse_conns), dtype=bool),
            num_valid=jnp.zeros(pop_size, dtype=jnp.int32),
        )

    # Get active position coordinates
    active_coords = all_positions[active_indices[:num_active]]  # (num_active, 2)
    # Ensure active_coords is on correct device
    active_coords = jax.device_put(active_coords, device)

    # Step 1: Query ALL active→active pairs in one batch
    # Pass device_id to ensure device-specific vmap traces for multi-GPU
    if num_cppn_outputs > 1:
        # Multi-output mode: preserve the CPPN output dimension.
        # batch_query_population_multi_source_chunked calls .flatten() which
        # corrupts multi-output CPPNs (turns (N, K) into (N*K,)); the multi-output
        # query path preserves the last dimension correctly.
        active_to_active_outputs = batch_query_population_multi_source_multi_output(
            state, cppns_transformed, active_coords, active_coords,
            outgoing=True, cppn_forward=cppn_forward,
            pop_chunk_size=pop_chunk_size, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled
        )
        # Extract weight channel (always output index 0)
        active_to_active_weights = active_to_active_outputs[:, :, :, 0]
        active_to_active_weights = jnp.tanh(active_to_active_weights) * max_weight
    else:
        # Standard single-output mode: Query returns (pop, num_active, num_active)
        active_to_active_weights = batch_query_population_multi_source_chunked(
            state, cppns_transformed, active_coords, active_coords,
            outgoing=True, cppn_forward=cppn_forward,
            pop_chunk_size=pop_chunk_size, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled
        )
        # Apply tanh activation and scale
        active_to_active_weights = jnp.tanh(active_to_active_weights) * max_weight

    # Step 2: Apply Y-coordinate constraints
    # Pass device_id for device-aware execution
    y_valid, self_mask = get_connection_constraint_mask_batched(
        active_coords, active_coords, config, device_id=device_id
    )
    # CRITICAL: Avoid fancy indexing ([None, :, :]) which triggers cached JIT traces
    # Use explicit reshape instead to avoid trace caching across devices
    combined_mask = y_valid & self_mask  # (num_active, num_active)
    combined_mask = jax.device_put(combined_mask, device)
    # Reshape to (1, num_active, num_active) for broadcasting
    constraint_mask = jnp.reshape(combined_mask, (1,) + combined_mask.shape)
    constraint_mask = jax.device_put(constraint_mask, device)

    # Step 3: Build validity mask (weight threshold)
    weight_valid = jnp.abs(active_to_active_weights) > band_threshold
    # Ensure weight_valid is on correct device
    weight_valid = jax.device_put(weight_valid, device)

    # DIAGNOSTIC: Analyze CPPN weights by connection type (DEBUG_CPPN_WEIGHTS=1)
    _debug_cppn_weights = os.environ.get('DEBUG_CPPN_WEIGHTS', '0') == '1'
    if _debug_cppn_weights:
        # Categorize connections by y-coordinate relationship
        y_coords = active_coords[:, 1]  # (num_active,)
        y_sources = y_coords[:, None]  # (num_active, 1)
        y_targets = y_coords[None, :]  # (1, num_active)

        # Connection type masks (num_active, num_active)
        forward_mask = y_sources < y_targets  # y1 < y2
        backward_mask = y_sources > y_targets  # y1 > y2
        lateral_mask = y_sources == y_targets  # y1 == y2

        # Self-loop mask: same (x, y) coordinates
        x_coords = active_coords[:, 0]
        self_loop_mask = (x_coords[:, None] == x_coords[None, :]) & lateral_mask

        # Exclude self-loops from lateral
        lateral_only_mask = lateral_mask & ~self_loop_mask

        # Get raw weights (first genome as representative)
        raw_weights = active_to_active_weights[0]  # (num_active, num_active)
        abs_weights = jnp.abs(raw_weights)

        # Stats per connection type
        def get_weight_stats(mask, weights):
            masked_w = jnp.where(mask, weights, jnp.nan)
            valid_w = masked_w[~jnp.isnan(masked_w)]
            if valid_w.size == 0:
                return 0, 0.0, 0.0, 0.0, 0.0, 0
            above_thresh = jnp.sum(jnp.abs(valid_w) > band_threshold)
            return (
                int(jnp.sum(mask)),  # total connections
                float(jnp.mean(jnp.abs(valid_w))),  # mean abs weight
                float(jnp.std(jnp.abs(valid_w))),   # std abs weight
                float(jnp.min(jnp.abs(valid_w))),   # min abs weight
                float(jnp.max(jnp.abs(valid_w))),   # max abs weight
                int(above_thresh),  # above threshold
            )

        print(f"\n[CPPN WEIGHT ANALYSIS] band_threshold={band_threshold}")
        print(f"{'Category':<15} {'Count':>8} {'Mean|w|':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'>thresh':>10} {'%pass':>8}")
        print("-" * 85)

        for name, mask in [
            ("forward", forward_mask),
            ("backward", backward_mask),
            ("lateral", lateral_only_mask),
            ("self_loop", self_loop_mask),
        ]:
            count, mean_w, std_w, min_w, max_w, above = get_weight_stats(mask, raw_weights)
            pct = 100.0 * above / max(count, 1)
            print(f"{name:<15} {count:>8} {mean_w:>10.6f} {std_w:>10.6f} {min_w:>10.6f} {max_w:>10.6f} {above:>10} {pct:>7.2f}%")

        # Overall stats
        total = int(forward_mask.sum() + backward_mask.sum() + lateral_only_mask.sum() + self_loop_mask.sum())
        weight_above = int(jnp.sum(jnp.abs(raw_weights) > band_threshold))
        print("-" * 85)
        print(f"{'TOTAL':<15} {total:>8} {'':<10} {'':<10} {'':<10} {'':<10} {weight_above:>10} {100.0*weight_above/max(total,1):>7.2f}%")
        print()

    if config.use_dense_discovery:
        # DENSE DISCOVERY (vanilla-style): Only use weight threshold + constraint
        # This finds ~15-20x more connections, better for hybrid at low populations
        # The per-genome active position filtering is SKIPPED
        direct_valid = weight_valid & constraint_mask
        if verbose:
            print(f"[H→H Discovery] DENSE MODE: skipping per-genome filtering")
    else:
        # SPARSE DISCOVERY (default): Apply per-genome active position masks
        # Only counts connections where BOTH source AND target are active for that genome
        # This is more "correct" but finds fewer connections
        genome_source_active = masks_A[:, active_indices[:num_active]]  # (pop, num_active)
        genome_target_active = masks_A[:, active_indices[:num_active]]  # (pop, num_active)
        # Ensure on correct device
        genome_source_active = jax.device_put(genome_source_active, device)
        genome_target_active = jax.device_put(genome_target_active, device)

        source_valid = genome_source_active[:, :, None]  # (pop, num_active, 1)
        target_valid = genome_target_active[:, None, :]  # (pop, 1, num_active)

        # Ensure reshaped arrays are on correct device
        source_valid = jax.device_put(source_valid, device)
        target_valid = jax.device_put(target_valid, device)

        # CRITICAL: Use per-device JIT function to avoid cross-device JIT trace reuse
        # Without this, JAX reuses JIT traces from cuda:0 on cuda:1 buffers causing errors
        combine_fn = _get_sparse_discovery_combine_fn(device)
        direct_valid = combine_fn(source_valid, target_valid, weight_valid, constraint_mask)

    # DIAGNOSTIC: Direct connection stats
    if verbose:
        direct_count = int(jnp.sum(direct_valid))
        constraint_count = int(jnp.sum(constraint_mask))
        weight_count = int(jnp.sum(weight_valid))
        print(f"[H→H Discovery] constraint_mask_valid={constraint_count}, "
              f"weight_above_thresh={weight_count}, direct_valid={direct_count}")

    # DIAGNOSTIC: Breakdown by connection type after all filtering (DEBUG_CPPN_WEIGHTS=1)
    if _debug_cppn_weights:
        # Re-create connection type masks
        y_coords = active_coords[:, 1]
        x_coords = active_coords[:, 0]
        y_sources = y_coords[:, None]
        y_targets = y_coords[None, :]
        forward_mask = y_sources < y_targets
        backward_mask = y_sources > y_targets
        lateral_mask = y_sources == y_targets
        self_loop_mask = (x_coords[:, None] == x_coords[None, :]) & lateral_mask
        lateral_only_mask = lateral_mask & ~self_loop_mask

        # Use first genome as representative (index [0] for the genome dimension)
        valid_first_genome = direct_valid[0] if direct_valid.ndim == 3 else direct_valid

        # Count by type
        fwd_valid = int(jnp.sum(valid_first_genome & forward_mask))
        bwd_valid = int(jnp.sum(valid_first_genome & backward_mask))
        lat_valid = int(jnp.sum(valid_first_genome & lateral_only_mask))
        self_valid = int(jnp.sum(valid_first_genome & self_loop_mask))
        total_valid = int(jnp.sum(valid_first_genome))

        print(f"\n[FILTERED CONNECTION BREAKDOWN] (after constraint + weight + active filtering)")
        print(f"  forward:   {fwd_valid:>6} connections")
        print(f"  backward:  {bwd_valid:>6} connections")
        print(f"  lateral:   {lat_valid:>6} connections")
        print(f"  self_loop: {self_valid:>6} connections")
        print(f"  TOTAL:     {total_valid:>6} connections")
        print(f"  Config: allow_backward={config.allow_backward}, allow_lateral={config.allow_lateral}, allow_self_loops={config.allow_self_loops}")

        # Per-genome active position analysis
        if not config.use_dense_discovery:
            active_per_genome = jnp.sum(masks_A[:, active_indices[:num_active]], axis=1)
            mean_active = float(jnp.mean(active_per_genome))
            min_active = int(jnp.min(active_per_genome))
            max_active = int(jnp.max(active_per_genome))

            # Y-level distribution for first genome
            first_genome_active = masks_A[0, active_indices[:num_active]]
            active_y_coords = y_coords[first_genome_active]
            unique_y = jnp.unique(active_y_coords, size=num_active, fill_value=-1)
            unique_y = unique_y[unique_y >= 0]
            n_unique_y = int(unique_y.shape[0])

            print(f"\n[PER-GENOME ACTIVE ANALYSIS]")
            print(f"  Active positions per genome: min={min_active}, mean={mean_active:.1f}, max={max_active}")
            print(f"  First genome: {int(jnp.sum(first_genome_active))} active positions across {n_unique_y} unique y-levels")

            # Count positions per y-level for first genome
            if n_unique_y > 0:
                for y_val in unique_y[:min(5, n_unique_y)]:  # Show first 5 y-levels
                    count = int(jnp.sum(active_y_coords == y_val))
                    print(f"    y={float(y_val):.3f}: {count} positions")

    # Step 4: Multi-hop expansion (vmapped over population)
    if config.iteration_level > 1:
        if config.multi_hop_algorithm == "matrix_power":
            def expand_single_genome(adjacency, valid):
                return compute_multi_hop_connections_matrix_power(
                    adjacency, valid, band_threshold,
                    config.iteration_level, config.hop_decay_factor, max_weight
                )
        else:  # fori_loop
            def expand_single_genome(adjacency, valid):
                return compute_multi_hop_connections_fori_loop(
                    adjacency, valid, band_threshold,
                    config.iteration_level, config.hop_decay_factor, max_weight
                )

        # CRITICAL: Ensure inputs are on correct device before multi-hop expansion
        active_to_active_weights = jax.device_put(active_to_active_weights, device)
        direct_valid = jax.device_put(direct_valid, device)

        # Compute chunk size for multi-hop discovery to prevent OOM
        # The vmap creates (pop_size, num_active, num_active) intermediates
        multi_hop_chunk_size = config.multi_hop_chunk_size
        if multi_hop_chunk_size is None:
            # Auto-compute based on num_active positions
            if num_active < 100:
                multi_hop_chunk_size = 0  # No chunking needed
            elif num_active < 500:
                multi_hop_chunk_size = min(200, pop_size)
            else:
                multi_hop_chunk_size = min(50, pop_size)

        if verbose:
            print(f"[H→H Discovery] multi_hop_chunk_size={multi_hop_chunk_size}, "
                  f"num_active={num_active}, pop_size={pop_size}")

        # Use device context and JIT wrapper to ensure proper device compilation
        if multi_hop_chunk_size and multi_hop_chunk_size > 0 and pop_size > multi_hop_chunk_size:
            # CHUNKED processing to prevent OOM on large populations
            weight_chunks = []
            valid_chunks = []
            for chunk_start in range(0, pop_size, multi_hop_chunk_size):
                chunk_end = min(chunk_start + multi_hop_chunk_size, pop_size)
                with jax.default_device(device):
                    chunk_w, chunk_v = jax.jit(jax.vmap(expand_single_genome))(
                        active_to_active_weights[chunk_start:chunk_end],
                        direct_valid[chunk_start:chunk_end]
                    )
                weight_chunks.append(chunk_w)
                valid_chunks.append(chunk_v)
            multi_hop_weights = jnp.concatenate(weight_chunks, axis=0)
            multi_hop_valid = jnp.concatenate(valid_chunks, axis=0)
        else:
            # Original: process all at once
            with jax.default_device(device):
                multi_hop_weights, multi_hop_valid = jax.jit(jax.vmap(expand_single_genome))(
                    active_to_active_weights, direct_valid
                )

        if verbose:
            multi_hop_count = int(jnp.sum(multi_hop_valid))
            print(f"[H→H Discovery] MULTI-HOP EXPANDED: multi_hop_valid={multi_hop_count} "
                  f"(added {multi_hop_count - int(jnp.sum(direct_valid))} transitive)")
    else:
        # No multi-hop (iteration_level = 1 means direct connections only)
        multi_hop_weights = jnp.where(direct_valid, active_to_active_weights, 0.0)
        multi_hop_valid = direct_valid
        if verbose:
            print(f"[H→H Discovery] NO MULTI-HOP (iteration_level=1): using direct only")

    # Step 5: Convert dense to sparse format using per-device cached vmap
    # The _get_extract_sparse_fn returns a device-specific traced vmap function
    # to avoid JAX trace caching issues across multi-GPU threads
    pop_size = multi_hop_valid.shape[0]
    active_indices_broadcast = jnp.broadcast_to(active_indices[None, :], (pop_size, active_indices.shape[0]))

    # Get device-specific vmap function
    extract_sparse_vmap = _get_extract_sparse_fn(device_id, num_active, max_sparse_conns)

    # CRITICAL: Explicitly move ALL vmap inputs to target device
    # Even though upstream computation should be on device, ensure it here
    multi_hop_valid = jax.device_put(multi_hop_valid, device)
    multi_hop_weights = jax.device_put(multi_hop_weights, device)
    active_indices_broadcast = jax.device_put(active_indices_broadcast, device)

    with jax.default_device(device):
        from_indices, to_indices, weights = extract_sparse_vmap(
            multi_hop_valid, multi_hop_weights, active_indices_broadcast
        )

        # Create validity mask and count
        valid_mask = from_indices >= 0
        num_valid = jnp.sum(valid_mask, axis=1)

    # Final logging (DEBUG_PHASE2)
    if verbose or _debug_phase2:
        total_conns = int(jnp.sum(num_valid))
        mean_conns = float(jnp.mean(num_valid))
        print(f"[H→H Discovery] COMPLETE: total_h2h_conns={total_conns}, "
              f"mean_per_genome={mean_conns:.1f}")

    return SparseHiddenConnections(
        from_indices=from_indices.astype(jnp.int32),
        to_indices=to_indices.astype(jnp.int32),
        weights=weights,
        valid_mask=valid_mask,
        num_valid=num_valid,
    )


# ============================================================================
# Forward Pass (Three Modes)
# ============================================================================

class ForwardPassMode(Enum):
    """Forward pass execution modes."""
    DENSE_ONLY = "dense_only"       # Feedforward: W1 → tanh → W2 (no h→h)
    HYBRID_SPARSE_HH = "hybrid"     # Dense W1/W2 + sparse h→h
    FULL_SPARSE = "full_sparse"     # All sparse (extreme recurrence)


def forward_unified_extended(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    sparse_hh: SparseHiddenConnections,
    activate_time: int,
    num_positions: int,
    mode: ForwardPassMode = ForwardPassMode.HYBRID_SPARSE_HH,
) -> jnp.ndarray:
    """Unified forward pass supporting all three modes.

    Args:
        inputs: Input values - shape (n_samples, num_inputs)
        W1: Input→hidden weights - shape (num_inputs, num_positions)
        W2: Hidden→output weights - shape (num_positions, num_outputs)
        sparse_hh: SparseHiddenConnections for hidden→hidden
        activate_time: Number of forward pass iterations
        num_positions: Total number of hidden positions
        mode: Which forward pass variant to use

    Returns:
        Output values - shape (n_samples, num_outputs)
    """
    if mode == ForwardPassMode.DENSE_ONLY:
        # Simple feedforward: W1 → tanh → W2 → sigmoid
        h = jnp.tanh(inputs @ W1)
        return jax.nn.sigmoid(h @ W2)

    elif mode == ForwardPassMode.HYBRID_SPARSE_HH:
        return _forward_hybrid_sparse_hh(
            inputs, W1, W2, sparse_hh, activate_time, num_positions
        )

    elif mode == ForwardPassMode.FULL_SPARSE:
        # same as hybrid (full sparse would need sparse W1/W2)
        return _forward_hybrid_sparse_hh(
            inputs, W1, W2, sparse_hh, activate_time, num_positions
        )

    else:
        raise ValueError(f"Unknown forward pass mode: {mode}")


def _forward_hybrid_sparse_hh(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    sparse_hh: SparseHiddenConnections,
    activate_time: int,
    num_positions: int,
) -> jnp.ndarray:
    """Hybrid forward pass: Dense W1/W2 + sparse h→h via lax.scan.

    Matches Model A (MultiGPU eval_single_network_sparse_hh):
    - input_contrib computed once (raw, no tanh)
    - hidden starts from zeros
    - each step: tanh(input_contrib + h_delta), input re-injected every step
    - scan runs activate_time steps (not activate_time - 1)

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        sparse_hh: SparseHiddenConnections
        activate_time: Number of iterations
        num_positions: Total positions

    Returns:
        (n_samples, num_outputs)
    """
    # Step 1: Input contribution (constant across iterations, raw, no tanh)
    input_contrib = inputs @ W1  # (n_samples, num_positions)

    # Step 2: Recurrent iterations with sparse h→h
    if sparse_hh is not None and sparse_hh.num_valid.sum() > 0:
        hidden = jnp.zeros((inputs.shape[0], num_positions))

        # Prepare safe indices
        safe_from = jnp.clip(sparse_hh.from_indices, 0, num_positions - 1)
        safe_to = jnp.clip(sparse_hh.to_indices, 0, num_positions - 1)
        effective_weights = jnp.where(sparse_hh.valid_mask, sparse_hh.weights, 0.0)

        def hh_step(hidden, _):
            # Gather from source positions
            source_vals = hidden[:, safe_from]  # (n_samples, max_sparse_conns)

            # Multiply by weights
            contributions = source_vals * effective_weights

            # Scatter-add to target positions
            h_delta = jnp.zeros_like(hidden)
            h_delta = h_delta.at[:, safe_to].add(contributions)

            # Combine input and recurrent contributions
            return jnp.tanh(input_contrib + h_delta), None

        hidden, _ = lax.scan(hh_step, hidden, None, length=activate_time)
    else:
        # No h→h connections: simple feedforward (one tanh application)
        hidden = jnp.tanh(input_contrib)

    # Step 3: Dense hidden→output
    return jax.nn.sigmoid(hidden @ W2)


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

    This version is designed for use with jax.vmap over population.

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
    num_positions = W1.shape[1]

    # Input contribution (constant across iterations, raw, no tanh)
    input_contrib = inputs @ W1

    # Sparse h→h recurrent iterations
    hidden = jnp.zeros((inputs.shape[0], num_positions))
    safe_from = jnp.clip(from_indices, 0, num_positions - 1)
    safe_to = jnp.clip(to_indices, 0, num_positions - 1)
    effective_weights = jnp.where(valid_mask, hh_weights, 0.0)

    def hh_step(hidden, _):
        source_vals = hidden[:, safe_from]
        contributions = source_vals * effective_weights
        h_delta = jnp.zeros_like(hidden)
        h_delta = h_delta.at[:, safe_to].add(contributions)
        return jnp.tanh(input_contrib + h_delta), None

    hidden, _ = lax.scan(hh_step, hidden, None, length=activate_time)

    return jax.nn.sigmoid(hidden @ W2)


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
        from_indices, to_indices, hh_weights, valid_mask: sparse h→h
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


# ============================================================================
# Dynamic Forward Pass Functions (with per-node activation/aggregation)
# ============================================================================

def forward_hybrid_vmapped_dynamic(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    activate_time: int,
    act_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Optional[Callable] = None,
    output_act_fn: Optional[Callable] = None,
    num_activations: int = 4,
    palette: Optional[jnp.ndarray] = None,
    hh_activation_mode: str = 'initial_only',
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
) -> jnp.ndarray:
    """Vmap-friendly hybrid forward pass with dynamic activation functions.

    This extends forward_hybrid_vmapped to support per-node activation functions
    and per-node H→H aggregation.

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        from_indices: (max_sparse_conns,) source indices
        to_indices: (max_sparse_conns,) target indices
        hh_weights: (max_sparse_conns,) weights
        valid_mask: (max_sparse_conns,) validity mask
        activate_time: Number of iterations
        act_indices: Optional (num_positions,) per-node activation indices.
            If None, uses hidden_act_fn for all nodes.
        hidden_act_fn: Optional activation function for 'disabled'/'global' modes.
            Used when act_indices is None.
        output_act_fn: Output activation function. Defaults to sigmoid.
        num_activations: Number of activation functions for grouped forward.
        palette: Optional custom activation palette indices.
        hh_activation_mode: 'initial_only' or 'every_iteration'
        hh_agg_indices: Optional (num_positions,) per-node H→H aggregation indices.
            If None or hh_aggregation_mode='sum', uses scatter_add (sum).
        hh_aggregation_mode: 'sum' (default) or 'dynamic' for per-node aggregation.
        num_hh_aggregations: Number of aggregation functions for dynamic H→H.

    Returns:
        (n_samples, num_outputs)
    """
    num_positions = W1.shape[1]
    output_fn = output_act_fn if output_act_fn is not None else jax.nn.sigmoid

    # Step 1: Input contribution (constant across iterations, raw, no activation)
    input_contrib = inputs @ W1  # (n_samples, num_positions)

    # Step 2: Recurrent iterations with sparse h→h
    hidden = jnp.zeros((inputs.shape[0], num_positions))
    safe_from = jnp.clip(from_indices, 0, num_positions - 1)
    safe_to = jnp.clip(to_indices, 0, num_positions - 1)
    effective_weights = jnp.where(valid_mask, hh_weights, 0.0)

    # Choose aggregation method for H→H contributions
    use_dynamic_hh_agg = (hh_aggregation_mode == 'dynamic' and hh_agg_indices is not None)

    def compute_h_delta(hidden, contributions):
        """Compute H→H delta with appropriate aggregation."""
        if use_dynamic_hh_agg:
            return scatter_aggregate_by_target(
                contributions, safe_to, num_positions,
                hh_agg_indices, valid_mask, num_hh_aggregations
            )
        else:
            # Default: scatter_add (sum aggregation)
            h_delta = jnp.zeros_like(hidden)
            return h_delta.at[:, safe_to].add(contributions)

    if hh_activation_mode == 'every_iteration' and act_indices is not None:
        # Apply per-node activations on every H→H iteration
        if palette is not None:
            def hh_step(hidden, _):
                source_vals = hidden[:, safe_from]
                contributions = source_vals * effective_weights
                h_delta = compute_h_delta(hidden, contributions)
                return grouped_activation_forward_with_palette(input_contrib + h_delta, act_indices, palette), None
        else:
            def hh_step(hidden, _):
                source_vals = hidden[:, safe_from]
                contributions = source_vals * effective_weights
                h_delta = compute_h_delta(hidden, contributions)
                return grouped_activation_forward(input_contrib + h_delta, act_indices, num_activations), None
    else:
        # Default: use tanh for H→H iterations (initial_only mode)
        def hh_step(hidden, _):
            source_vals = hidden[:, safe_from]
            contributions = source_vals * effective_weights
            h_delta = compute_h_delta(hidden, contributions)
            return jnp.tanh(input_contrib + h_delta), None

    hidden, _ = lax.scan(hh_step, hidden, None, length=activate_time)

    # Step 3: Dense hidden→output with output activation
    return output_fn(hidden @ W2)


def forward_hybrid_vmapped_neuromodulated(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    activate_time: int,
    receptor_densities: jnp.ndarray,
    base_gains: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    mod_strength: float,
    modulation_mode: str = 'full',
    use_output_inversion: bool = True,
    num_nt: int = 4,
    act_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Optional[Callable] = None,
    output_act_fn: Optional[Callable] = None,
    num_activations: int = 4,
    palette: Optional[jnp.ndarray] = None,
    hh_activation_mode: str = 'initial_only',
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
) -> jnp.ndarray:
    """Vmap-friendly hybrid forward pass with TRUE neuromodulation.

    Neuromodulation is applied to BOTH:
    1. Input→Hidden stage
    2. H→H iterations (per user requirement)

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        from_indices: (max_sparse_conns,) source indices
        to_indices: (max_sparse_conns,) target indices
        hh_weights: (max_sparse_conns,) weights
        valid_mask: (max_sparse_conns,) validity mask
        activate_time: Number of iterations
        receptor_densities: (num_positions, num_nt) per-node receptor densities
        base_gains: (num_positions,) per-node base gains
        neurotransmitter: (num_nt,) NT vector for task
        mod_strength: Modulation strength factor
        modulation_mode: 'full', 'gating_only', or 'gain_bias_only'
        use_output_inversion: Whether NT4 controls output inversion
        num_nt: Number of neurotransmitter types
        act_indices: Optional (num_positions,) per-node activation indices.
        hidden_act_fn: Optional activation function for 'disabled'/'global' modes.
        output_act_fn: Output activation function. Defaults to sigmoid.
        num_activations: Number of activation functions for grouped forward.
        palette: Optional custom activation palette indices.
        hh_activation_mode: 'initial_only' or 'every_iteration'
        hh_agg_indices: Optional (num_positions,) per-node H→H aggregation indices.
        hh_aggregation_mode: 'sum' (default) or 'dynamic' for per-node aggregation.
        num_hh_aggregations: Number of aggregation functions for dynamic H→H.

    Returns:
        (n_samples, num_outputs)
    """
    num_positions = W1.shape[1]
    output_fn = output_act_fn if output_act_fn is not None else jax.nn.sigmoid

    # Compute modulation from receptors × NT (once, reused for all iterations)
    nt_mod = neurotransmitter[:min(3, num_nt)]
    modulation = receptor_densities[:, :min(3, num_nt)] @ nt_mod  # (num_positions,)
    effective_gains = base_gains + mod_strength * modulation
    gates = jax.nn.sigmoid(modulation * 5.0)

    # Step 1: Input contribution (constant across iterations, raw, no activation)
    input_contrib = inputs @ W1  # (n_samples, num_positions)

    # Step 2: Recurrent iterations with sparse h→h (NEUROMODULATED)
    hidden = jnp.zeros((inputs.shape[0], num_positions))
    safe_from = jnp.clip(from_indices, 0, num_positions - 1)
    safe_to = jnp.clip(to_indices, 0, num_positions - 1)
    effective_weights = jnp.where(valid_mask, hh_weights, 0.0)

    use_dynamic_hh_agg = (hh_aggregation_mode == 'dynamic' and hh_agg_indices is not None)

    def compute_h_delta(hidden, contributions):
        """Compute H→H delta with appropriate aggregation."""
        if use_dynamic_hh_agg:
            return scatter_aggregate_by_target(
                contributions, safe_to, num_positions,
                hh_agg_indices, valid_mask, num_hh_aggregations
            )
        else:
            h_delta = jnp.zeros_like(hidden)
            return h_delta.at[:, safe_to].add(contributions)

    if hh_activation_mode == 'every_iteration' and act_indices is not None:
        # Apply per-node activations + neuromodulation on every H→H iteration
        if palette is not None:
            def hh_step(hidden, _):
                source_vals = hidden[:, safe_from]
                contributions = source_vals * effective_weights
                h_delta = compute_h_delta(hidden, contributions)
                # Apply gain modulation to H→H
                if modulation_mode == 'gain_bias_only' or modulation_mode == 'full':
                    h_updated = grouped_activation_forward_with_palette(
                        effective_gains * (input_contrib + h_delta), act_indices, palette
                    )
                else:
                    h_updated = grouped_activation_forward_with_palette(
                        input_contrib + h_delta, act_indices, palette
                    )
                # Apply gates
                if modulation_mode == 'gating_only' or modulation_mode == 'full':
                    h_updated = h_updated * gates
                return h_updated, None
        else:
            def hh_step(hidden, _):
                source_vals = hidden[:, safe_from]
                contributions = source_vals * effective_weights
                h_delta = compute_h_delta(hidden, contributions)
                # Apply gain modulation to H→H
                if modulation_mode == 'gain_bias_only' or modulation_mode == 'full':
                    h_updated = grouped_activation_forward(
                        effective_gains * (input_contrib + h_delta), act_indices, num_activations
                    )
                else:
                    h_updated = grouped_activation_forward(
                        input_contrib + h_delta, act_indices, num_activations
                    )
                # Apply gates
                if modulation_mode == 'gating_only' or modulation_mode == 'full':
                    h_updated = h_updated * gates
                return h_updated, None
    else:
        # Default: use tanh for H→H iterations but still apply gates
        def hh_step(hidden, _):
            source_vals = hidden[:, safe_from]
            contributions = source_vals * effective_weights
            h_delta = compute_h_delta(hidden, contributions)
            # Apply gain modulation if full mode
            if modulation_mode == 'gain_bias_only' or modulation_mode == 'full':
                h_updated = jnp.tanh(effective_gains * (input_contrib + h_delta))
            else:
                h_updated = jnp.tanh(input_contrib + h_delta)
            # Apply gates
            if modulation_mode == 'gating_only' or modulation_mode == 'full':
                h_updated = h_updated * gates
            return h_updated, None

    hidden, _ = lax.scan(hh_step, hidden, None, length=activate_time)

    # Step 3: Dense hidden→output with output activation
    outputs = output_fn(hidden @ W2)

    # NT4 output inversion control
    if use_output_inversion and num_nt >= 4:
        invert = neurotransmitter[3]
        outputs = invert * outputs + (1 - invert) * (1 - outputs)

    return outputs


def eval_single_network_hybrid_neuromodulated(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    activate_time: int,
    receptor_densities: jnp.ndarray,
    base_gains: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    mod_strength: float,
    modulation_mode: str = 'full',
    use_output_inversion: bool = True,
    num_nt: int = 4,
    act_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Optional[Callable] = None,
    output_act_fn: Optional[Callable] = None,
    num_activations: int = 4,
    palette: Optional[jnp.ndarray] = None,
    hh_activation_mode: str = 'initial_only',
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
) -> float:
    """Evaluate single network with TRUE neuromodulation.

    Neuromodulation is applied to BOTH Input→Hidden AND H→H iterations.

    Returns:
        Fitness score (1.0 - MSE)
    """
    outputs = forward_hybrid_vmapped_neuromodulated(
        inputs, W1, W2,
        from_indices, to_indices, hh_weights, valid_mask,
        activate_time,
        receptor_densities, base_gains, neurotransmitter, mod_strength,
        modulation_mode, use_output_inversion, num_nt,
        act_indices=act_indices,
        hidden_act_fn=hidden_act_fn,
        output_act_fn=output_act_fn,
        num_activations=num_activations,
        palette=palette,
        hh_activation_mode=hh_activation_mode,
        hh_agg_indices=hh_agg_indices,
        hh_aggregation_mode=hh_aggregation_mode,
        num_hh_aggregations=num_hh_aggregations,
    )
    errors = jnp.mean((outputs - targets) ** 2, axis=1)
    return jnp.maximum(0.0, 1.0 - jnp.mean(errors))


def eval_single_network_hybrid_dynamic(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    activate_time: int,
    act_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Optional[Callable] = None,
    output_act_fn: Optional[Callable] = None,
    num_activations: int = 4,
    palette: Optional[jnp.ndarray] = None,
    hh_activation_mode: str = 'initial_only',
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
) -> float:
    """Evaluate single network with dynamic activation functions.

    Args:
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        from_indices, to_indices, hh_weights, valid_mask: sparse h→h
        inputs: (n_samples, num_inputs)
        targets: (n_samples, num_outputs)
        activate_time: Forward iterations
        act_indices: Optional per-node activation indices
        hidden_act_fn: Optional activation function for 'disabled'/'global' modes
        output_act_fn: Output activation function
        num_activations: Number of activation functions
        palette: Optional custom activation palette
        hh_activation_mode: 'initial_only' or 'every_iteration'
        hh_agg_indices: Optional per-node H→H aggregation indices
        hh_aggregation_mode: 'sum' or 'dynamic' for H→H aggregation
        num_hh_aggregations: Number of H→H aggregation functions

    Returns:
        Fitness score (1.0 - MSE)
    """
    outputs = forward_hybrid_vmapped_dynamic(
        inputs, W1, W2,
        from_indices, to_indices, hh_weights, valid_mask,
        activate_time,
        act_indices=act_indices,
        hidden_act_fn=hidden_act_fn,
        output_act_fn=output_act_fn,
        num_activations=num_activations,
        palette=palette,
        hh_activation_mode=hh_activation_mode,
        hh_agg_indices=hh_agg_indices,
        hh_aggregation_mode=hh_aggregation_mode,
        num_hh_aggregations=num_hh_aggregations,
    )
    errors = jnp.mean((outputs - targets) ** 2, axis=1)
    return jnp.maximum(0.0, 1.0 - jnp.mean(errors))


# ============================================================================
# Cache Manager
# ============================================================================

class HHCacheManager:
    """Manages caching of sparse h→h connections across generations.

    Caching saves expensive Phase 2 discovery when the variance mask
    (which determines active positions) hasn't changed significantly.
    """

    def __init__(self, config: EMRConfig):
        """Initialize cache manager.

        Args:
            config: Extended configuration with caching settings
        """
        self.config = config
        self._cached_connections: Optional[SparseHiddenConnections] = None
        self._cached_variance_mask: Optional[jnp.ndarray] = None
        self._last_refresh_gen: int = -1
        self._refresh_count: int = 0

    def should_refresh(
        self,
        current_gen: int,
        current_mask: jnp.ndarray,
    ) -> bool:
        """Determine if h→h connections need refresh.

        Refresh if:
        1. Caching is disabled
        2. No cached connections exist
        3. Time-based: >= refresh_interval generations since last refresh
        4. Change-based: Variance mask changed by > threshold

        Args:
            current_gen: Current generation number
            current_mask: Current variance mask (pop, total_positions)

        Returns:
            True if refresh needed
        """
        if not self.config.hh_cache_enabled:
            return True

        if self._cached_connections is None:
            return True

        # Time-based check
        gens_since_refresh = current_gen - self._last_refresh_gen
        if gens_since_refresh >= self.config.hh_refresh_interval:
            return True

        # Change-based check
        if self._cached_variance_mask is not None:
            # Compute mask change ratio (fraction of positions that changed)
            mask_diff = jnp.abs(
                current_mask.astype(jnp.float32) -
                self._cached_variance_mask.astype(jnp.float32)
            )
            change_ratio = float(jnp.mean(mask_diff))

            if change_ratio > self.config.hh_mask_change_threshold:
                return True

        return False

    def update_cache(
        self,
        connections: SparseHiddenConnections,
        variance_mask: jnp.ndarray,
        generation: int,
    ) -> None:
        """Update cached connections.

        Args:
            connections: New sparse connections
            variance_mask: Current variance mask
            generation: Current generation
        """
        self._cached_connections = connections
        self._cached_variance_mask = variance_mask
        self._last_refresh_gen = generation
        self._refresh_count += 1

    def get_cached(self) -> Optional[SparseHiddenConnections]:
        """Get cached connections if available."""
        return self._cached_connections

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'refresh_count': self._refresh_count,
            'last_refresh_gen': self._last_refresh_gen,
            'cache_enabled': self.config.hh_cache_enabled,
        }


# ============================================================================
# Main Class: EMRHyperNEAT
# ============================================================================

class EMRHyperNEAT(EMRHyperNEATMultiGPU):
    """Unified Extended EMR-HyperNEAT with Dynamic Functions + Neuromodulation.

    This is the FULL implementation combining:
    - unified_extended.py: All multi-GPU, sparse H→H, multi-hop, caching features
    - dynamic_functions_aggregation.py: Per-node activation/aggregation functions
    - neuromodulation_functions.py: 4 neuromodulation levels + multi-task support

    Features from unified_extended:
    - EMRConfig with all recurrence options + 6 presets
    - SparseHiddenConnections dataclass for type-safe storage
    - Discovery strategy toggle (vectorized vs iterative)
    - Multi-hop vectorized expansion (JIT-compatible matrix power)
    - Y-coordinate constraint filtering (forward/backward/lateral/self)
    - H→H caching with time-based and change-based refresh
    - Multi-GPU position sharding for Phase 2
    - Three forward pass modes
    - Automatic exhaustive dataset detection for correct multi-GPU routing

    Features from dynamic_functions_aggregation:
    - 18 activation functions (tanh, sigmoid, relu, sin, burst, resonator, etc.)
    - 6 aggregation functions (sum, mean, max, min, product, maxabs)
    - 7 selection modes (disabled, global, cppn_output, weight_interpretation, etc.)
    - Palette system for custom activation function subsets
    - Per-node dynamic function selection

    Features from neuromodulation (4 levels):
    - Level 1: static_gating - Per-connection gate values [0, 1]
    - Level 2: context_gating - XdG-style task context modulates gates
    - Level 3: modulatory_neurons - Soltoggio-style dedicated modulatory neurons
    - Level 4: true_neuromodulation - NT vectors + receptor densities + gain modulation
    - Multi-task support with 6 fitness aggregation methods
    - Neuromodulation applied to BOTH Input→Hidden AND H→H iterations

    Dynamic Functions Configuration:
        dynamic_functions:
          mode: 'disabled'           # disabled, global, cppn_output, weight_interpretation
          hidden_activation: 'tanh'  # For global mode
          output_activation: 'sigmoid'
          num_activations: 4         # For cppn_output mode
          palette: 'sin_only'        # Optional: use specific activation subset
          hh_activation_mode: 'initial_only'  # initial_only or every_iteration

          # CPPN output configuration (for cppn_output mode):
          num_cppn_outputs: 1        # Number of CPPN outputs (1=derive from weights, 2+=dedicated outputs)
          cppn_output_indices:       # Which CPPN output index to use for each purpose
            weight: 0                # Index for weight output (always 0)
            activation: 0            # Index for activation output (0 when num_cppn_outputs=1)
            aggregation: 0           # Index for aggregation output (for future use)

        aggregation:
          mode: 'disabled'           # disabled, global, cppn_output, weight_interpretation
          global_function: 'sum'
          num_aggregations: 4
          use_true_aggregation: true
          hh_aggregation_mode: 'sum'  # sum or dynamic

    Neuromodulation Configuration:
        neuromodulation:
          mode: 'disabled'           # disabled, static_gating, context_gating,
                                     # modulatory_neurons, true_neuromodulation

          # Static gating (Level 1)
          gate_threshold: 0.5
          gate_scaling: 'sigmoid'    # sigmoid, hard, linear

          # Context gating - XdG style (Level 2)
          context_dim: 4
          context_influence: 0.5
          context_source: 'task_id'  # task_id, input_derived, learned

          # Modulatory neurons - Soltoggio style (Level 3)
          mod_neuron_ratio: 0.1
          mod_connection_type: 'multiplicative'  # multiplicative, additive, gating
          mod_decay: 0.9

          # TRUE neuromodulation (Level 4)
          num_nt_types: 4            # [DA, 5HT, NE, ACh]
          modulation_strength: 2.0
          receptor_from_weight: true # Option A (true) vs B (CPPN outputs)
          receptor_derivation: 'tanh' # tanh, abs, normalized, fourier, softmax,
                                      # orthogonal, phase_shifted
          modulation_mode: 'full'    # full, gating_only, gain_bias_only
          use_output_inversion: true # NT4 controls output inversion

        multitask:
          enabled: false
          num_tasks: 2
          task_names: ['xor', 'and']
          fitness_aggregation: 'mean'  # mean, min, weighted, product, softmin, harmonic
          task_weights: null
          orthogonality_bonus: 0.0
          specialization_bonus: 0.0

    Example with neuromodulation:
        algo = EMRHyperNEAT()
        algo.create_config({
            'algorithm_params': {
                'emrhyperneat': {
                    'emr_hyperneat': {
                        'max_depth': 3,
                        'recurrence': {'preset': 'hidden_only'},
                        'dynamic_functions': {
                            'mode': 'weight_interpretation',
                            'num_activations': 6
                        },
                        'neuromodulation': {
                            'mode': 'true_neuromodulation',
                            'num_nt_types': 4,
                            'receptor_from_weight': True
                        },
                        'multitask': {
                            'enabled': True,
                            'num_tasks': 5,
                            'task_names': ['xor', 'and', 'or', 'nand', 'nor'],
                            'fitness_aggregation': 'min'
                        }
                    }
                }
            }
        })
    """

    def __init__(
        self,
        name: str = 'emr-hyperneat',
        implementation: str = 'tensorneat-emrhyperneat-unified-extended-dynamic-functions',
        strategy: MultiGPUStrategy = MultiGPUStrategy.SINGLE_GPU,
        position_config: Optional[PositionShardingConfig] = None,
        island_config: Optional[IslandModelConfig] = None,
        hybrid_config: Optional[HybridShardingConfig] = None,
        pmap_config: Optional[PopulationPmapConfig] = None,
    ):
        """Initialize unified extended EMR-HyperNEAT.

        Args:
            name: Algorithm name
            implementation: Implementation identifier
            strategy: GPU execution strategy
            position_config: Position sharding configuration
            island_config: Island model configuration (legacy)
            hybrid_config: Hybrid sharding configuration (legacy)
            pmap_config: Population pmap configuration (legacy)
        """
        super().__init__(
            name=name,
            implementation=implementation,
            strategy=strategy,
            position_config=position_config,
            island_config=island_config,
            hybrid_config=hybrid_config,
            pmap_config=pmap_config,
        )

        # Extended configuration
        self.extended_config: Optional[EMRConfig] = None

        # Cache manager
        self._hh_cache: Optional[HHCacheManager] = None

        # Metrics tracking
        self._extended_metrics: Optional[EMRRecurrenceMetrics] = None

        # Forward pass mode
        self._forward_mode: ForwardPassMode = ForwardPassMode.DENSE_ONLY

        # Current generation (for caching)
        self._current_generation: int = 0

        # Fitness rescaling: when set, converts fitness from 1-MSE to R²
        # R² = 1 - MSE/target_variance, spreading the useful range
        self._fitness_rescale_variance: Optional[float] = None

        # ====================================================================
        # Dynamic Functions Configuration (from dynamic_functions_aggregation.py)
        # ====================================================================

        # Activation function settings
        self.dynamic_functions_mode: str = 'disabled'  # disabled, global, cppn_output, weight_interpretation, random_fixed, random_generation, modular
        self.df_hidden_activation: str = 'tanh'
        self.df_output_activation: str = 'sigmoid'
        self.df_num_activations: int = 4
        self.df_palette: Optional[jnp.ndarray] = None  # Custom palette indices
        self.df_palette_name: Optional[str] = None  # Named palette
        self.df_interpretation: str = 'magnitude_bio'  # For weight_interpretation mode
        self.hh_activation_mode: str = 'initial_only'  # initial_only or every_iteration

        # Aggregation function settings
        self.agg_mode: str = 'disabled'  # disabled, global, cppn_output, weight_interpretation
        self.agg_global_function: str = 'sum'
        self.agg_num_aggregations: int = 4
        self.agg_use_true_aggregation: bool = True  # True = accurate, False = approximation
        self.agg_interpretation: str = 'magnitude_bio'
        # hh_aggregation_mode controls H→H aggregation behavior:
        # - 'sum': Fast scatter_add (default, best performance)
        # - 'dynamic': Per-node aggregation using segment operations
        self.hh_aggregation_mode: str = 'sum'  # sum or dynamic

        # Sparsity settings (from dynamic_functions_aggregation.py)
        self.df_sparsity_config: Dict[str, float] = {
            'level_0': 0.05,  # DG-like, 5% active
            'level_1': 0.20,  # CA3-like, 20% active
            'level_2_plus': 0.40,  # CA1-like, 40% active
        }
        self._sparsity_thresholds: Optional[jnp.ndarray] = None

        # Cached activation/aggregation functions
        self._hidden_act_fn: Optional[Callable] = None
        self._output_act_fn: Optional[Callable] = None

        # CPPN output configuration for cppn_output mode
        # num_cppn_outputs: How many outputs the CPPN has (default 1)
        # When num_cppn_outputs=1, we use the same weight output for activation derivation
        # When num_cppn_outputs>1, each output can be dedicated (weight, activation, aggregation)
        self.num_cppn_outputs: int = 1
        self._cppn_output_indices: Dict[str, int] = {
            'weight': 0,
            'activation': 0,  # Default to same as weight when num_cppn_outputs=1
            'aggregation': 0,
        }
        # Cache for multi-output CPPN queries (future use)
        self._cached_cppn_outputs: Optional[jnp.ndarray] = None

        # Random mode state
        self._random_mode_seed: Optional[int] = None
        self._random_fixed_indices: Optional[jnp.ndarray] = None
        self._random_agg_seed: Optional[int] = None
        self._random_fixed_agg_indices: Optional[jnp.ndarray] = None

        # Level indices for hierarchical sparsity
        self._level_indices: Optional[jnp.ndarray] = None

        # Modular mode configuration
        self.df_modular_config: Dict[str, str] = {
            'activation_method': 'magnitude_bio',
            'sparsity_method': 'none',
            'scaling_method': 'none',
            'aggregation_method': 'none',
        }
        self.df_modular_sparsity: Dict[str, float] = {
            'level_0': 0.05,
            'level_1': 0.20,
            'level_2_plus': 0.40,
            'wta_k_percent': 0.10,
        }

        # Critical periods configuration
        self.df_critical_periods_enabled: bool = False
        self.df_critical_periods_config: Dict[str, float] = {
            'phase1_end': 0.2,
            'phase2_end': 0.5,
            'min_plasticity': 0.3,
        }

        # ====================================================================
        # Neuromodulation Configuration (4 levels supported)
        # ====================================================================
        # Level 1: static_gating - Per-connection gate values [0, 1]
        # Level 2: context_gating - XdG-style context modulates gates
        # Level 3: modulatory_neurons - Soltoggio-style dedicated modulatory neurons
        # Level 4: true_neuromodulation - NT vectors + receptor densities + gain modulation
        # ====================================================================

        # Neuromodulation config (parsed from user config in create_config)
        self.neuromod_config: Optional[NeuromodulationConfig] = None

        # TRUE neuromodulation storage (populated during _process_cppn_outputs)
        # - receptor_densities: (pop_size, total_positions, num_nt_types)
        # - base_gains: (pop_size, total_positions)
        # - modulation_strength: float
        # - option: 'A' (from weight) or 'B' (dedicated CPPN outputs)
        self._neuromod_true: Dict[str, Any] = {
            'receptor_densities': None,
            'base_gains': None,
            'num_nt_types': 4,
            'modulation_strength': 2.0,
            'option': 'A',  # Default: derive from weights
        }

        # Static gating storage
        self._static_gates: Optional[jnp.ndarray] = None  # (pop_size, total_positions)

        # Context gating storage (XdG-style)
        self._context_gates: Optional[jnp.ndarray] = None  # (pop_size, context_dim, total_positions)
        self._current_context: Optional[jnp.ndarray] = None  # (context_dim,)

        # Modulatory neurons storage (Soltoggio-style)
        self._mod_neuron_mask: Optional[jnp.ndarray] = None  # Boolean mask of modulatory neurons
        self._mod_connection_weights: Optional[jnp.ndarray] = None  # Modulatory connections
        self._mod_neuron_states: Optional[jnp.ndarray] = None  # Current modulatory neuron activity

        # ====================================================================
        # Multi-Task Configuration
        # ====================================================================

        # Multi-task config (parsed from user config in create_config)
        self.multitask_config: Optional[MultiTaskConfig] = None

        # Task-specific neurotransmitter vectors
        # - nt_vectors: (num_tasks, num_nt_types)
        # - task_fitnesses: (pop_size, num_tasks)
        self._multitask_state: Dict[str, Any] = {
            'nt_vectors': None,
            'task_fitnesses': None,
            'task_names': [],
            'current_task_idx': 0,
        }

        # ====================================================================
        # Level-by-Level Streaming Configuration
        # ====================================================================
        # When enabled, CPPN queries are done one hierarchical level at a time
        # instead of all positions at once. This reduces peak memory from
        # O(total_positions) to O(max_level_size), enabling depth 7+ on CPU.
        #
        # Trade-off: 3-10x slower per generation due to Python loop overhead.
        # ====================================================================

        self.enable_streaming: bool = False  # Default: standard (faster)
        self._streaming_verbose: bool = False  # Per-level timing if True


    def create_config(self, params: Dict[str, Any]) -> Any:
        """Create configuration with extended recurrence support.

        Args:
            params: Configuration parameters

        Returns:
            Configuration object
        """
        # Parse extended configuration BEFORE calling parent
        algo_params = params.get('algorithm_params', {}).get('emrhyperneat', params)
        hmr_config = algo_params.get('emr_hyperneat', {})

        # Opt-in reproduction flag (default False -> EMR's existing behavior is UNCHANGED).
        # When True, EMR performs HMR's extra per-generation pre-ask jax.random.split.
        # HMR's default (extra_randkey_split=True) produced the published results, so this
        # lets reproduction benchmarks recover HMR's exact per-seed trajectories in EMR.
        # NOTE: stored under a DISTINCT name so the base class's own `extra_randkey_split`
        # attribute (which defaults True but is unused by EMR's eval) cannot leak in.
        self._repro_extra_randkey_split = hmr_config.get('extra_randkey_split', False)

        # Check if geometry seeding is enabled - need to set CPPN num_inputs to 7
        geo_config = hmr_config.get('geometry_seeding', {})
        geo_seeding_enabled = geo_config.get('enabled', False)

        # Auto-compute required CPPN outputs from dynamic_functions + aggregation config
        dynamic_funcs = hmr_config.get('dynamic_functions', {})
        agg_config_section = hmr_config.get('aggregation', {})
        required_outputs = 1  # weight always needs output 0
        if dynamic_funcs.get('mode') == 'cppn_output':
            required_outputs += 1  # activation needs its own output
        if agg_config_section.get('mode') == 'cppn_output':
            required_outputs += 1  # aggregation needs its own output

        # Allow explicit override (for ablation configs like 2-output variants)
        explicit_num = dynamic_funcs.get('num_cppn_outputs', None)
        if explicit_num is not None:
            required_outputs = explicit_num

        # Determine if we need to deepcopy and inject CPPN genome params
        needs_deepcopy = geo_seeding_enabled or required_outputs > 1

        if needs_deepcopy:
            import copy
            params = copy.deepcopy(params)

            # Navigate to or create the cppn.genome path
            if 'algorithm_params' not in params:
                params['algorithm_params'] = {}
            if 'emrhyperneat' not in params['algorithm_params']:
                params['algorithm_params']['emrhyperneat'] = {}
            if 'cppn' not in params['algorithm_params']['emrhyperneat']:
                params['algorithm_params']['emrhyperneat']['cppn'] = {}
            if 'genome' not in params['algorithm_params']['emrhyperneat']['cppn']:
                params['algorithm_params']['emrhyperneat']['cppn']['genome'] = {}

            cppn_genome = params['algorithm_params']['emrhyperneat']['cppn']['genome']

            if geo_seeding_enabled:
                # Geometry seeding requires 7 CPPN inputs: [x1, y1, x2, y2, delta_x, delta_y, bias]
                cppn_genome['num_inputs'] = 7
                if self.verbose:
                    print(f"[EMR-HyperNEAT] Geometry seeding: CPPN num_inputs set to 7 [x1, y1, x2, y2, delta_x, delta_y, bias]")

            if required_outputs > 1:
                # Inject num_outputs so NEAT genome is built with correct output count
                cppn_genome['num_outputs'] = required_outputs
                if self.verbose:
                    print(f"[EMR-HyperNEAT] Auto-computed CPPN num_outputs={required_outputs} "
                          f"(act_mode={dynamic_funcs.get('mode', 'disabled')}, "
                          f"agg_mode={agg_config_section.get('mode', 'disabled')})")

        # Store for later use by _parse_dynamic_functions_config
        self._auto_num_cppn_outputs = required_outputs

        # Parse fitness rescaling (converts 1-MSE fitness to R² for compressed landscapes)
        rescale_var = hmr_config.get('fitness_rescale_variance', None)
        if rescale_var is not None and rescale_var > 0:
            self._fitness_rescale_variance = float(rescale_var)
            if self.verbose:
                print(f"[EMR-HyperNEAT] Fitness rescaling enabled: "
                      f"target_variance={rescale_var:.6f}, "
                      f"fitness = R² = 1 - MSE/var")

        # Now call parent to set up base configuration
        config = super().create_config(params)
        recurrence_section = hmr_config.get('recurrence', {})

        # Parse multi_gpu_strategy from config (string -> enum)
        strategy_str = hmr_config.get('multi_gpu_strategy', None)
        if strategy_str is not None:
            strategy_str_upper = strategy_str.upper()
            # ============================================================================
            # 5 AVAILABLE GPU STRATEGIES
            # ============================================================================
            # Decision tree:
            #   Dataset fits on single GPU?
            #   ├─ YES → SINGLE_GPU (fastest)
            #   └─ NO  → Multiple GPUs available?
            #            ├─ NO  → STREAMING (streams from CPU)
            #            └─ YES → Recurrent/h→h mode?
            #                     ├─ YES → EVAL_ONLY_PARALLEL (h→h caching)
            #                     └─ NO  → FULL_PIPELINE_PARALLEL (data parallel)
            #
            # POPULATION_PARALLEL_PROCESS: For very large populations with h→h
            # ============================================================================
            strategy_map = {
                # Primary 5 strategies
                'SINGLE_GPU': MultiGPUStrategy.SINGLE_GPU,
                'FULL_PIPELINE_PARALLEL': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
                'STREAMING': MultiGPUStrategy.STREAMING,
                'EVAL_ONLY_PARALLEL': MultiGPUStrategy.EVAL_ONLY_PARALLEL,
                'POPULATION_PARALLEL_PROCESS': MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
                # Legacy aliases (map to primary strategies for backward compatibility)
                'BASELINE': MultiGPUStrategy.SINGLE_GPU,
                'MULTI_GPU': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
                'DATA_PARALLEL': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
                'PMAP_PARALLEL': MultiGPUStrategy.EVAL_ONLY_PARALLEL,
                'PIPELINE_CHUNKED': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
                'CPPN_CHUNKED': MultiGPUStrategy.STREAMING,
                'POSITION_SHARDING_CHUNKED': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
                'POPULATION_PARALLEL_SEQUENTIAL': MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
                'PERSISTENT_PARALLEL': MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
            }
            if strategy_str_upper in strategy_map:
                self.strategy = strategy_map[strategy_str_upper]
                if self.verbose:
                    print(f"[HMR-UnifiedExtended] Strategy from config: {strategy_str} → {self.strategy}")
            else:
                print(f"Warning: Unknown multi_gpu_strategy '{strategy_str}', keeping default: {self.strategy}")

        self._parse_extended_config(recurrence_section)

        # Parse dynamic functions configuration
        self._parse_dynamic_functions_config(hmr_config)

        # Parse neuromodulation configuration (4 levels + multi-task)
        self._parse_neuromodulation_config(hmr_config)

        # Parse streaming configuration
        self.enable_streaming = hmr_config.get('enable_streaming', False)
        self._streaming_verbose = hmr_config.get('streaming_verbose', False)


        # Parse locality radius for geometry seeding (distance-based weight penalty)
        # None = no penalty (default), positive float = penalty for distances > radius
        self.locality_radius = hmr_config.get('locality_radius', None)
        if self.locality_radius is not None and self.verbose:
            print(f"[EMR-HyperNEAT] Locality radius enabled: {self.locality_radius}")

        # Parse geometry seeding configuration (Risi & Stanley 2012 paper approach)
        # This adds distance as an explicit CPPN input and seeds initial topology
        geo_config = hmr_config.get('geometry_seeding', {})
        self.geometry_seeding_enabled = geo_config.get('enabled', False)
        self.geometry_seeding_weight = geo_config.get('seed_weight', -1.0)
        if self.geometry_seeding_enabled and self.verbose:
            print(f"[EMR-HyperNEAT] Geometry seeding ENABLED (seed_weight={self.geometry_seeding_weight})")
            print(f"[EMR-HyperNEAT] CPPN inputs: [x1, y1, x2, y2, delta_x, delta_y, bias] (7 dimensions)")

        # Log swarm mode status
        if self.swarm_mode and self.verbose:
            cppn_dims = 7 + (2 if self.geometry_seeding_enabled else 0)
            print(f"[EMR-HyperNEAT] SWARM MODE ENABLED: {self.swarm_num_agents} agents, "
                  f"layout={self.swarm_team_layout}")
            print(f"[EMR-HyperNEAT] CPPN inputs: [agent_x, agent_y, x1, y1, x2, y2, bias] "
                  f"({cppn_dims} dimensions)")

        # Initialize cache manager
        if self.extended_config is not None:
            self._hh_cache = HHCacheManager(self.extended_config)

        return config

    def initialize(self, config: Any, problem: Any, seed: int = 42) -> Any:
        """Initialize with per-device JIT functions for multi-GPU support.

        This overrides the parent initialize() to add per-device JIT function
        creation. This fixes device placement errors when using ThreadPoolExecutor
        for device-parallel h→h processing - JAX JIT functions are compiled to a
        specific device and ignore thread-local device contexts.

        Args:
            config: Algorithm configuration
            problem: Problem instance
            seed: Random seed

        Returns:
            Initialized algorithm state
        """
        # Call parent initialize first
        state = super().initialize(config, problem, seed)

        # Create per-device JIT functions for multi-GPU support
        # This fixes device placement errors when using ThreadPoolExecutor
        devices = jax.devices()
        self._jitted_cppn_forward_per_device = {}

        if len(devices) > 1:
            print(f"[Multi-GPU] Creating per-device JIT functions for {len(devices)} devices")
            for device in devices:
                # Use device= parameter to pin JIT function to specific device.
                # jax.default_device() context ONLY affects where arrays are created,
                # it does NOT pin JIT functions to that device. Without device=, the
                # JIT function can be reused across devices causing device placement errors.
                self._jitted_cppn_forward_per_device[device] = jax.jit(
                    self.neat_algo.genome.forward, static_argnums=(0,), device=device
                )

        return state

    def refresh_problem_cache(self, problem: Any) -> None:
        """Refresh cached problem data for MVG (Modularly Varying Goals) support.

        This method re-reads the problem data and updates the cached inputs and
        targets. Use this when the problem's targets change during evolution
        (e.g., Kashtan MVG where goals alternate every N generations).

        Args:
            problem: Problem instance with get_data() method.

        Example:
            # In evolution loop with MVG problem:
            for gen in range(max_generations):
                new_goal = problem.set_generation(gen)
                if problem.needs_cache_refresh():
                    algo.refresh_problem_cache(problem)
                    problem.mark_cache_refreshed()
                state, metrics = algo.run_generation(state, problem)
        """
        # Get fresh data from problem
        if hasattr(problem, 'get_data'):
            data = problem.get_data()
            self._cached_inputs = jnp.array([d[0] for d in data], dtype=jnp.float32)
            self._cached_targets = jnp.array([d[1] for d in data], dtype=jnp.float32)
        elif hasattr(problem, 'inputs') and hasattr(problem, 'targets'):
            self._cached_inputs = jnp.array(problem.inputs, dtype=jnp.float32)
            self._cached_targets = jnp.array(problem.targets, dtype=jnp.float32)
        else:
            raise ValueError(
                "Problem must have get_data() method or inputs/targets attributes"
            )

        # Mark problem cache as refreshed if it supports this interface
        if hasattr(problem, 'mark_cache_refreshed'):
            problem.mark_cache_refreshed()

    def _setup_pmap_evaluation(self) -> None:
        """One-time setup for pmap-based multi-GPU evaluation.

        This method pre-shards the dataset across devices and validates the
        configuration for pmap compatibility. Called once before the first
        pmap generation loop.

        Sets:
            self._pmap_devices: List of JAX devices
            self._pmap_num_devices: Number of devices
            self._pmap_inputs_sharded: Inputs sharded across devices
            self._pmap_targets_sharded: Targets sharded across devices
            self._pmap_samples_per_device: Samples per device (padded if needed)
            self._pmap_original_n_samples: Original sample count (for error aggregation)
        """
        self._pmap_devices = jax.devices()
        self._pmap_num_devices = len(self._pmap_devices)

        if self._pmap_num_devices < 2:
            raise ValueError(
                f"EVAL_ONLY_PARALLEL requires 2+ devices, found {self._pmap_num_devices}. "
                f"Use SINGLE_GPU strategy instead."
            )

        # Get dataset info
        inputs = self._cached_inputs  # (n_samples, n_inputs)
        targets = self._cached_targets  # (n_samples, n_outputs)
        n_samples = inputs.shape[0]
        self._pmap_original_n_samples = n_samples

        # Pad dataset if not divisible by device count
        samples_per_device = (n_samples + self._pmap_num_devices - 1) // self._pmap_num_devices
        padded_total = samples_per_device * self._pmap_num_devices
        pad_needed = padded_total - n_samples

        if pad_needed > 0:
            # Pad with zeros (will be masked out in error calculation)
            inputs_padded = jnp.pad(inputs, ((0, pad_needed), (0, 0)), mode='constant')
            targets_padded = jnp.pad(targets, ((0, pad_needed), (0, 0)), mode='constant')
            if self.verbose:
                print(f"[EVAL_ONLY_PARALLEL] Padded dataset: {n_samples} → {padded_total} samples "
                      f"({samples_per_device}/device)")
        else:
            inputs_padded = inputs
            targets_padded = targets

        # Reshape to (num_devices, samples_per_device, features)
        self._pmap_inputs_sharded = inputs_padded.reshape(
            self._pmap_num_devices, samples_per_device, inputs.shape[1]
        )
        self._pmap_targets_sharded = targets_padded.reshape(
            self._pmap_num_devices, samples_per_device, targets.shape[1]
        )
        self._pmap_samples_per_device = samples_per_device

        if self.verbose:
            print(f"[EVAL_ONLY_PARALLEL] Dataset sharded: {self._pmap_inputs_sharded.shape}")
            print(f"[EVAL_ONLY_PARALLEL] Devices: {[str(d) for d in self._pmap_devices]}")

    def _parse_extended_config(self, recurrence_section: Dict[str, Any]) -> None:
        """Parse recurrence section into extended config.

        Args:
            recurrence_section: The 'recurrence' subsection of emr_hyperneat config
        """
        # Check for preset first
        preset_name = recurrence_section.get('preset', None)
        if preset_name:
            base_config = get_recurrence_preset(preset_name)
            if self.verbose:
                print(f"[HMR-UnifiedExtended] Using preset '{preset_name}'")

            # Apply explicit overrides from recurrence_section
            # The preset provides defaults, but explicit values take precedence
            override_fields = {}
            configurable_fields = [
                'iteration_level', 'multi_hop_algorithm', 'hop_decay_factor',
                'hh_cache_enabled', 'hh_refresh_interval', 'hh_mask_change_threshold',
                'allow_backward', 'allow_lateral', 'allow_self_loops',
                'activate_time', 'max_connections', 'max_sparse_conns', 'use_dense_discovery',
                'eval_chunk_size',  # For OOM prevention on large configs
            ]
            for field in configurable_fields:
                if field in recurrence_section:
                    override_fields[field] = recurrence_section[field]

            # Create new config with overrides
            if override_fields:
                from dataclasses import replace
                self.extended_config = replace(base_config, **override_fields)
                if self.verbose:
                    print(f"[HMR-UnifiedExtended] Applied overrides: {override_fields}")
            else:
                self.extended_config = base_config
        elif recurrence_section.get('enabled', False) or recurrence_section.get('allow_hidden_to_hidden', False):
            # Build config from individual settings
            self.extended_config = EMRConfig(
                enabled=recurrence_section.get('enabled', True),
                allow_hidden_to_hidden=recurrence_section.get('allow_hidden_to_hidden', True),
                allow_backward=recurrence_section.get('allow_backward', False),
                allow_lateral=recurrence_section.get('allow_lateral', False),
                allow_self_loops=recurrence_section.get('allow_self_loops', False),
                iteration_level=recurrence_section.get('iteration_level', 2),
                activate_time=recurrence_section.get('activate_time', None),
                max_connections=recurrence_section.get('max_connections', 10000),
                use_vectorized_discovery=recurrence_section.get('use_vectorized_discovery', True),
                max_sparse_conns=recurrence_section.get('max_sparse_conns', 10000),
                multi_hop_algorithm=recurrence_section.get('multi_hop_algorithm', 'matrix_power'),
                hop_decay_factor=recurrence_section.get('hop_decay_factor', 0.8),
                hh_cache_enabled=recurrence_section.get('hh_cache_enabled', True),
                hh_refresh_interval=recurrence_section.get('hh_refresh_interval', 5),
                hh_mask_change_threshold=recurrence_section.get('hh_mask_change_threshold', 0.1),
                use_dense_discovery=recurrence_section.get('use_dense_discovery', False),
                eval_chunk_size=recurrence_section.get('eval_chunk_size', None),
            )
        elif recurrence_section.get('eval_chunk_size') is not None:
            # Feedforward mode but with eval_chunk_size specified for OOM prevention
            self.extended_config = EMRConfig(
                enabled=False,
                allow_hidden_to_hidden=False,
                eval_chunk_size=recurrence_section.get('eval_chunk_size'),
            )
        else:
            # Feedforward mode without chunking
            self.extended_config = None

        # Set mode flags and update parent attributes
        if self.extended_config is not None:
            self._forward_mode = (
                ForwardPassMode.HYBRID_SPARSE_HH
                if self.extended_config.allow_hidden_to_hidden
                else ForwardPassMode.DENSE_ONLY
            )

            # Update parent iteration_level
            self.iteration_level = self.extended_config.iteration_level

            # Update activate_time
            if self.extended_config.activate_time is not None:
                self.activate_time = self.extended_config.activate_time
            elif self.extended_config.allow_hidden_to_hidden:
                # Auto-compute: more iterations for h→h
                default = (2 ** self.max_depth) + 1
                self.activate_time = default * 2

            # Log configuration
            if self.verbose:
                mode = self._forward_mode.value
                print(f"[HMR-UnifiedExtended] Mode: {mode}")
                if self.extended_config.allow_hidden_to_hidden:
                    print(f"  - iteration_level: {self.extended_config.iteration_level}")
                    print(f"  - multi_hop_algorithm: {self.extended_config.multi_hop_algorithm}")
                    print(f"  - hop_decay_factor: {self.extended_config.hop_decay_factor}")
                    print(f"  - allow_backward: {self.extended_config.allow_backward}")
                    print(f"  - allow_lateral: {self.extended_config.allow_lateral}")
                    print(f"  - allow_self_loops: {self.extended_config.allow_self_loops}")
                    print(f"  - activate_time: {self.activate_time}")
        else:
            self._forward_mode = ForwardPassMode.DENSE_ONLY
            self.iteration_level = 0

    def _parse_dynamic_functions_config(self, hmr_config: Dict[str, Any]) -> None:
        """Parse dynamic functions and aggregation configuration.

        Args:
            hmr_config: The 'emr_hyperneat' section of config
        """
        dynamic_funcs = hmr_config.get('dynamic_functions', {})

        # Mode selection:
        # - 'disabled': Original hardcoded tanh/sigmoid (baseline)
        # - 'global': All hidden nodes use same configurable activation
        # - 'cppn_output': CPPN outputs activation index per node
        # - 'weight_interpretation': Derive activation from weight patterns
        # - 'random_fixed': Random activation per node, fixed at initialization
        # - 'random_generation': Random re-assigned each generation
        # - 'modular': Orthogonal configuration
        self.dynamic_functions_mode = dynamic_funcs.get('mode', 'disabled')

        # For random modes: store seed for reproducibility
        self._random_mode_seed = dynamic_funcs.get('random_seed', None)

        # For 'global' mode: which activation to use
        self.df_hidden_activation = dynamic_funcs.get('hidden_activation', 'tanh')
        self.df_output_activation = dynamic_funcs.get('output_activation', 'sigmoid')

        # For 'cppn_output' mode: number of activation functions available
        self.df_num_activations = dynamic_funcs.get('num_activations', 4)

        # CPPN output configuration for cppn_output mode
        # num_cppn_outputs: How many outputs the CPPN has (default from auto-compute)
        # When num_cppn_outputs=1, we derive activation from the same weight output
        # When num_cppn_outputs>1, each output can be dedicated to different purposes
        self.num_cppn_outputs = dynamic_funcs.get(
            'num_cppn_outputs', getattr(self, '_auto_num_cppn_outputs', 1)
        )

        # cppn_output_indices: Which CPPN output index to use for each purpose
        # Default: all use index 0 (single-output CPPN compatibility)
        cppn_indices = dynamic_funcs.get('cppn_output_indices', {})
        self._cppn_output_indices = {
            'weight': cppn_indices.get('weight', 0),
            'activation': cppn_indices.get('activation', 0 if self.num_cppn_outputs == 1 else 1),
            'aggregation': cppn_indices.get('aggregation', 0 if self.num_cppn_outputs <= 2 else 2),
        }

        # Validate CPPN output indices
        for purpose, idx in self._cppn_output_indices.items():
            if idx >= self.num_cppn_outputs:
                raise ValueError(
                    f"CPPN output index for '{purpose}' ({idx}) >= num_cppn_outputs ({self.num_cppn_outputs}). "
                    f"Increase num_cppn_outputs or fix cppn_output_indices."
                )

        # H→H activation mode: controls how activations are applied during H→H iterations
        # - 'initial_only': Apply dynamic activations on input→hidden, use fixed tanh for H→H
        # - 'every_iteration': Apply per-node dynamic activations on EVERY H→H iteration
        self.hh_activation_mode = dynamic_funcs.get('hh_activation_mode', 'initial_only')

        # Palette configuration
        palette_config = dynamic_funcs.get('palette', None)
        if palette_config is None:
            self.df_palette = None
            self.df_palette_name = None
        elif isinstance(palette_config, str):
            # Preset name
            if palette_config in PALETTE_CONFIGS:
                self.df_palette = jnp.array(PALETTE_CONFIGS[palette_config], dtype=jnp.int32)
                self.df_palette_name = palette_config
            else:
                raise ValueError(f"Unknown palette preset: {palette_config}. "
                               f"Available: {list(PALETTE_CONFIGS.keys())}")
        else:
            # Explicit list of indices
            self.df_palette = jnp.array(palette_config, dtype=jnp.int32)
            self.df_palette_name = 'custom'

        # For 'weight_interpretation' mode: interpretation method
        self.df_interpretation = dynamic_funcs.get('interpretation', 'magnitude_bio')

        # Sparsity configuration for biologically-inspired mechanisms
        sparsity_config = dynamic_funcs.get('sparsity', {})
        self.df_sparsity_config = {
            'level_0': sparsity_config.get('level_0', 0.05),
            'level_1': sparsity_config.get('level_1', 0.20),
            'level_2_plus': sparsity_config.get('level_2_plus', 0.40),
            'wta_k_percent': sparsity_config.get('wta_k_percent', 0.10),
        }

        # Critical Periods configuration
        critical_periods = dynamic_funcs.get('critical_periods', {})
        self.df_critical_periods_enabled = critical_periods.get('enabled', False)
        self.df_critical_periods_config = {
            'phase1_end': critical_periods.get('phase1_end', 0.2),
            'phase2_end': critical_periods.get('phase2_end', 0.5),
            'min_plasticity': critical_periods.get('min_plasticity', 0.3),
        }

        # Modular mode configuration
        modular_config = dynamic_funcs.get('modular', {})
        self.df_modular_config = {
            'activation_method': modular_config.get('activation_method', 'magnitude_bio'),
            'sparsity_method': modular_config.get('sparsity_method', 'none'),
            'scaling_method': modular_config.get('scaling_method', 'none'),
        }
        modular_sparsity = modular_config.get('sparsity', {})
        self.df_modular_sparsity = {
            'level_0': modular_sparsity.get('level_0', 0.05),
            'level_1': modular_sparsity.get('level_1', 0.20),
            'level_2_plus': modular_sparsity.get('level_2_plus', 0.40),
            'wta_k_percent': modular_sparsity.get('wta_k_percent', 0.10),
        }

        # =====================================================================
        # AGGREGATION FUNCTIONS CONFIGURATION
        # =====================================================================
        # Note: aggregation is a sibling of dynamic_functions in the config,
        # so we read it from hmr_config, not from dynamic_funcs
        aggregation_config = hmr_config.get('aggregation', {})
        self.agg_mode = aggregation_config.get('mode', 'disabled')

        # For random modes: store seed for reproducibility
        self._random_agg_seed = aggregation_config.get('random_seed', None)

        # For 'global' mode: which aggregation to use
        self.agg_global_function = aggregation_config.get('global_function', 'sum')

        # For 'cppn_output' mode: number of aggregation functions available
        self.agg_num_aggregations = aggregation_config.get('num_aggregations', 4)

        # For 'weight_interpretation' mode: interpretation method
        self.agg_interpretation = aggregation_config.get('interpretation', 'magnitude_bio')

        # Whether to use true per-node aggregation or approximation
        self.agg_use_true_aggregation = aggregation_config.get('use_true_aggregation', True)

        # H→H aggregation mode: controls how aggregations are applied during H→H iterations
        # - 'sum': Standard scatter-add (current behavior, efficient)
        # - 'dynamic': Per-node aggregation using segment operations (more complex)
        self.hh_aggregation_mode = aggregation_config.get('hh_aggregation_mode', 'sum')

        # Extend modular config to include aggregation method
        self.df_modular_config['aggregation_method'] = modular_config.get('aggregation_method', 'none')

        # Resolve hidden activation function for 'global' and 'disabled' modes
        if self.dynamic_functions_mode in ('disabled', 'global'):
            act_name = self.df_hidden_activation if self.dynamic_functions_mode == 'global' else 'tanh'
            self._hidden_act_fn = ACTIVATION_FUNCTIONS.get(act_name, jnp.tanh)
            self._output_act_fn = ACTIVATION_FUNCTIONS.get(self.df_output_activation, jax.nn.sigmoid)
        else:
            # For cppn_output, weight_interpretation, etc., we use grouped_activation_forward
            self._hidden_act_fn = None
            self._output_act_fn = ACTIVATION_FUNCTIONS.get(self.df_output_activation, jax.nn.sigmoid)

        # Log configuration
        if self.verbose:
            if self.dynamic_functions_mode != 'disabled':
                print(f"[Dynamic Functions] Mode: {self.dynamic_functions_mode}")
                if self.dynamic_functions_mode == 'global':
                    print(f"  - Hidden activation: {self.df_hidden_activation}")
                elif self.dynamic_functions_mode == 'cppn_output':
                    print(f"  - Num activations: {self.df_num_activations}")
                    print(f"  - Num CPPN outputs: {self.num_cppn_outputs}")
                    print(f"  - CPPN output indices: {self._cppn_output_indices}")
                    if self.df_palette_name:
                        print(f"  - Palette: {self.df_palette_name} ({len(self.df_palette)} functions)")
                elif self.dynamic_functions_mode == 'weight_interpretation':
                    print(f"  - Interpretation: {self.df_interpretation}")
                elif self.dynamic_functions_mode == 'modular':
                    print(f"  - Activation method: {self.df_modular_config['activation_method']}")
                    print(f"  - Sparsity method: {self.df_modular_config['sparsity_method']}")
                    print(f"  - Scaling method: {self.df_modular_config['scaling_method']}")
                print(f"  - H→H activation mode: {self.hh_activation_mode}")

            if self.agg_mode != 'disabled':
                print(f"[Aggregation Functions] Mode: {self.agg_mode}")
                if self.agg_mode == 'global':
                    print(f"  - Global function: {self.agg_global_function}")
                elif self.agg_mode == 'cppn_output':
                    print(f"  - Num aggregations: {self.agg_num_aggregations}")
                elif self.agg_mode == 'weight_interpretation':
                    print(f"  - Interpretation: {self.agg_interpretation}")
                print(f"  - True aggregation: {self.agg_use_true_aggregation}")
                print(f"  - H→H aggregation mode: {self.hh_aggregation_mode}")

    def _parse_neuromodulation_config(self, hmr_config: Dict[str, Any]) -> None:
        """Parse neuromodulation and multi-task configuration.

        Supports 4 neuromodulation levels:
        1. static_gating: Per-connection gate values [0, 1]
        2. context_gating: XdG-style context modulates gates
        3. modulatory_neurons: Soltoggio-style dedicated modulatory neurons
        4. true_neuromodulation: NT vectors + receptor densities + gain modulation

        Args:
            hmr_config: The 'emr_hyperneat' section of config
        """
        neuromod_config = hmr_config.get('neuromodulation', {})

        # Mode selection (default: disabled = original behavior)
        mode = neuromod_config.get('mode', 'disabled')

        # Check for preset
        preset = neuromod_config.get('preset', None)
        if preset and preset in NEUROMODULATION_PRESETS:
            base = get_neuromodulation_config(preset)
            # Override with explicit settings
            self.neuromod_config = NeuromodulationConfig(
                mode=neuromod_config.get('mode', base.mode),
                # Static gating
                gate_threshold=neuromod_config.get('gate_threshold', base.gate_threshold),
                gate_scaling=neuromod_config.get('gate_scaling', base.gate_scaling),
                gate_hardness=neuromod_config.get('gate_hardness', base.gate_hardness),
                # Context gating (XdG-style)
                context_dim=neuromod_config.get('context_dim', base.context_dim),
                context_influence=neuromod_config.get('context_influence', base.context_influence),
                context_source=neuromod_config.get('context_source', base.context_source),
                # Modulatory neurons (Soltoggio-style)
                mod_neuron_ratio=neuromod_config.get('mod_neuron_ratio', base.mod_neuron_ratio),
                mod_connection_type=neuromod_config.get('mod_connection_type', base.mod_connection_type),
                mod_decay=neuromod_config.get('mod_decay', base.mod_decay),
                # TRUE neuromodulation
                num_nt_types=neuromod_config.get('num_nt_types', base.num_nt_types),
                modulation_strength=neuromod_config.get('modulation_strength', base.modulation_strength),
                receptor_from_weight=neuromod_config.get('receptor_from_weight', base.receptor_from_weight),
                receptor_derivation=neuromod_config.get('receptor_derivation', base.receptor_derivation),
                modulation_mode=neuromod_config.get('modulation_mode', base.modulation_mode),
                use_output_inversion=neuromod_config.get('use_output_inversion', base.use_output_inversion),
                # Branch-specific gating
                branch_gating=neuromod_config.get('branch_gating', base.branch_gating),
            )
        elif mode != 'disabled':
            # Build config from individual settings
            # Set mode-specific flags based on the mode string
            self.neuromod_config = NeuromodulationConfig(
                enabled=True,  # Any non-disabled mode is enabled
                mode=mode,
                # Mode-specific flags
                static_gating=(mode == 'static_gating'),
                context_gating=(mode == 'context_gating'),
                modulatory_neurons=(mode == 'modulatory_neurons'),
                true_neuromodulation=(mode == 'true_neuromodulation'),
                # Static gating params
                gate_threshold=neuromod_config.get('gate_threshold', 0.5),
                gate_scaling=neuromod_config.get('gate_scaling', 'sigmoid'),
                gate_hardness=neuromod_config.get('gate_hardness', 10.0),
                # Context gating (XdG-style)
                context_dim=neuromod_config.get('context_dim', 4),
                context_influence=neuromod_config.get('context_influence', 0.5),
                context_source=neuromod_config.get('context_source', 'task_id'),
                # Modulatory neurons (Soltoggio-style)
                mod_neuron_ratio=neuromod_config.get('mod_neuron_ratio', 0.1),
                mod_connection_type=neuromod_config.get('mod_connection_type', 'multiplicative'),
                mod_decay=neuromod_config.get('mod_decay', 0.9),
                # TRUE neuromodulation
                num_nt_types=neuromod_config.get('num_nt_types', 4),
                modulation_strength=neuromod_config.get('modulation_strength', 2.0),
                receptor_from_weight=neuromod_config.get('receptor_from_weight', True),
                receptor_derivation=neuromod_config.get('receptor_derivation', 'tanh'),
                modulation_mode=neuromod_config.get('modulation_mode', 'full'),
                use_output_inversion=neuromod_config.get('use_output_inversion', True),
                # Branch-specific gating
                branch_gating=neuromod_config.get('branch_gating', False),
                # R7: opt-in self-connection-query decode (reproduces the HMR/paper
                # receptor/base-gain source). Default False = existing EMR behavior.
                use_self_connection_query=neuromod_config.get('use_self_connection_query', False),
            )
        else:
            # Disabled mode - original behavior
            self.neuromod_config = None

        # Update TRUE neuromodulation storage if enabled
        if self.neuromod_config and self.neuromod_config.mode == 'true_neuromodulation':
            self._neuromod_true = {
                'receptor_densities': None,  # Set during substrate building
                'base_gains': None,  # Set during substrate building
                'num_nt_types': self.neuromod_config.num_nt_types,
                'modulation_strength': self.neuromod_config.modulation_strength,
                'option': 'A' if self.neuromod_config.receptor_from_weight else 'B',
            }

        # =====================================================================
        # MULTI-TASK CONFIGURATION
        # =====================================================================
        multitask_config = hmr_config.get('multitask', {})
        if multitask_config.get('enabled', False):
            self.multitask_config = MultiTaskConfig(
                enabled=True,
                num_tasks=multitask_config.get('num_tasks', 2),
                task_names=multitask_config.get('task_names', None),
                nt_per_task=multitask_config.get('nt_per_task', None),
                fitness_aggregation=multitask_config.get('fitness_aggregation', 'mean'),
                task_weights=multitask_config.get('task_weights', None),
                joint_evolution=multitask_config.get('joint_evolution', True),
                orthogonality_bonus=multitask_config.get('orthogonality_bonus', 0.0),
                specialization_bonus=multitask_config.get('specialization_bonus', 0.0),
                # Dynamic activation support for per-task activation functions
                hidden_activation=multitask_config.get('hidden_activation', 'tanh'),
                per_task_activation=multitask_config.get('per_task_activation', None),
            )

            # Initialize NT vectors if true_neuromodulation is enabled
            if (self.neuromod_config and
                self.neuromod_config.mode == 'true_neuromodulation'):
                num_tasks = self.multitask_config.num_tasks
                num_nt = self.neuromod_config.num_nt_types

                # Get NT vectors for each task
                if self.multitask_config.nt_per_task is not None:
                    # Use user-specified vectors
                    nt_vectors = jnp.array(self.multitask_config.nt_per_task)
                else:
                    # Use default presets
                    task_names = (self.multitask_config.task_names or
                                  [f'task_{i}' for i in range(num_tasks)])
                    nt_vectors = jnp.array([
                        get_nt_for_task(name, num_nt) for name in task_names
                    ])

                self._multitask_state = {
                    'nt_vectors': nt_vectors,
                    'task_fitnesses': None,
                    'task_names': (self.multitask_config.task_names or
                                   [f'task_{i}' for i in range(num_tasks)]),
                    'current_task_idx': 0,
                }
        else:
            self.multitask_config = None

        # Log neuromodulation configuration
        if self.verbose and self.neuromod_config:
            mode = self.neuromod_config.mode
            print(f"[Neuromodulation] Mode: {mode}")
            if mode == 'static_gating':
                print(f"  - Gate threshold: {self.neuromod_config.gate_threshold}")
                print(f"  - Gate scaling: {self.neuromod_config.gate_scaling}")
            elif mode == 'context_gating':
                print(f"  - Context dim: {self.neuromod_config.context_dim}")
                print(f"  - Context influence: {self.neuromod_config.context_influence}")
                print(f"  - Context source: {self.neuromod_config.context_source}")
            elif mode == 'modulatory_neurons':
                print(f"  - Mod neuron ratio: {self.neuromod_config.mod_neuron_ratio}")
                print(f"  - Connection type: {self.neuromod_config.mod_connection_type}")
                print(f"  - Decay: {self.neuromod_config.mod_decay}")
            elif mode == 'true_neuromodulation':
                print(f"  - Num NT types: {self.neuromod_config.num_nt_types}")
                print(f"  - Modulation strength: {self.neuromod_config.modulation_strength}")
                print(f"  - Receptor from weight: {self.neuromod_config.receptor_from_weight}")
                print(f"  - Receptor derivation: {self.neuromod_config.receptor_derivation}")
                print(f"  - Modulation mode: {self.neuromod_config.modulation_mode}")
                print(f"  - Output inversion: {self.neuromod_config.use_output_inversion}")

            if self.multitask_config:
                print(f"[Multi-Task] Enabled")
                print(f"  - Num tasks: {self.multitask_config.num_tasks}")
                print(f"  - Fitness aggregation: {self.multitask_config.fitness_aggregation}")
                if self.multitask_config.orthogonality_bonus > 0:
                    print(f"  - Orthogonality bonus: {self.multitask_config.orthogonality_bonus}")
                if self.multitask_config.specialization_bonus > 0:
                    print(f"  - Specialization bonus: {self.multitask_config.specialization_bonus}")
                # Show activation function configuration
                if self.multitask_config.per_task_activation:
                    print(f"  - Per-task activations: {self.multitask_config.per_task_activation}")
                elif self.multitask_config.hidden_activation != 'tanh':
                    print(f"  - Hidden activation: {self.multitask_config.hidden_activation}")

    def get_recurrence_mode(self) -> str:
        """Get current recurrence mode as string."""
        if self.extended_config is None:
            return 'feedforward'
        elif self.extended_config.enabled and (
            self.extended_config.allow_backward or
            self.extended_config.allow_lateral or
            self.extended_config.allow_self_loops
        ):
            return 'full_recurrent'
        elif self.extended_config.allow_hidden_to_hidden:
            return 'feedforward+hh'
        else:
            return 'feedforward'

    def get_extended_config(self) -> Optional[EMRConfig]:
        """Get current extended configuration."""
        return self.extended_config

    def get_extended_metrics(self) -> Optional[EMRRecurrenceMetrics]:
        """Get extended metrics from last generation."""
        return self._extended_metrics

    def get_best_activation_indices(self) -> Dict[str, Any]:
        """Extract per-node activation and aggregation function indices for the best individual.

        Must be called after run_generation_verbose() while cached state is valid.
        Returns dict with activation_histogram and aggregation_histogram (Counter-like dicts).
        """
        result = {}
        if self._cached_fitnesses is None:
            return result

        best_idx = int(jnp.argmax(self._cached_fitnesses))

        if self._cached_act_indices is not None:
            act_idx = np.array(self._cached_act_indices[best_idx])
            num_act = len(self.df_palette) if self.df_palette is not None else self.df_num_activations
            names = ACTIVATION_LIST[:num_act]
            hist = {}
            for idx in act_idx:
                name = names[int(idx)] if int(idx) < len(names) else f'unknown_{idx}'
                hist[name] = hist.get(name, 0) + 1
            result['activation_histogram'] = hist
            result['num_positions'] = len(act_idx)

        if self._cached_ih_agg_indices is not None:
            agg_idx = np.array(self._cached_ih_agg_indices[best_idx])
            names = AGGREGATION_LIST[:self.agg_num_aggregations]
            hist = {}
            for idx in agg_idx:
                name = names[int(idx)] if int(idx) < len(names) else f'unknown_{idx}'
                hist[name] = hist.get(name, 0) + 1
            result['aggregation_histogram'] = hist

        return result

    def evaluate_best_all_modes(self) -> Optional[Dict[str, float]]:
        """Compute all 6 fitness metrics for the best individual using cached state.

        Re-runs the forward pass for a single individual (cheap) and evaluates with
        each fitness mode. Must be called after run_generation_verbose().
        """
        if self._cached_fitnesses is None or self._cached_W1 is None:
            return None

        best_idx = int(jnp.argmax(self._cached_fitnesses))
        W1_i = self._cached_W1[best_idx]
        W2_i = self._cached_W2[best_idx]
        inputs = self._cached_inputs
        targets = self._cached_targets

        # I→H: handle per-node aggregation or standard matmul
        if self._cached_ih_agg_indices is not None and self.agg_use_true_aggregation:
            agg_idx = self._cached_ih_agg_indices[best_idx]
            weighted = inputs[:, :, None] * W1_i[None, :, :]
            pre_h = grouped_aggregation_forward(weighted, agg_idx, self.agg_num_aggregations)
        else:
            pre_h = inputs @ W1_i

        # Hidden activation
        if self._cached_act_indices is not None:
            act_idx = self._cached_act_indices[best_idx]
            if self.df_palette is not None:
                h = grouped_activation_forward_with_palette(
                    pre_h[None], act_idx, self.df_palette
                )[0]
            else:
                h = grouped_activation_forward(pre_h[None], act_idx, self.df_num_activations)[0]
        elif self._hidden_act_fn is not None:
            h = self._hidden_act_fn(pre_h)
        else:
            h = jnp.tanh(pre_h)

        outputs = jax.nn.sigmoid(h @ W2_i)

        # Compute all 6 fitness modes using _eval_single_fitness (handles multiclass)
        modes = ['mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy']
        result = {}
        for mode in modes:
            if mode == 'acc_mse':
                # acc_mse not in _eval_single_fitness, compute manually
                acc = float(_eval_single_fitness(outputs, targets, 'accuracy'))
                mse_fit = float(_eval_single_fitness(outputs, targets, 'mse'))
                result[mode] = acc + 0.01 * mse_fit
            else:
                result[mode] = float(_eval_single_fitness(outputs, targets, mode))
        return result

    def run_generation(
        self,
        state: Any,
        problem: Any,
        skip_metrics: bool = False,
        verbose: bool = True,
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation, dispatching to multi-task if enabled.

        Overrides parent's run_generation to add multi-task dispatch.
        When multitask_config is enabled, routes to run_generation_multitask()
        which evaluates on all tasks with different NT vectors.

        Args:
            state: Algorithm state
            problem: Problem instance (or MultiTaskProblem for multi-task)
            skip_metrics: Skip metrics computation
            verbose: Verbose output

        Returns:
            Tuple of (new_state, metrics)
        """
        # Check if multi-task mode is enabled
        if (self.multitask_config is not None and
            getattr(self.multitask_config, 'enabled', False)):
            # Multi-task mode: extract individual problems from MultiTaskProblem
            if hasattr(problem, 'tasks'):
                problems = problem.tasks
            elif hasattr(problem, 'problems'):
                problems = problem.problems
            else:
                problems = [problem]

            # Get NT vectors from state
            neurotransmitters = self._multitask_state.get('nt_vectors', None)

            return self.run_generation_multitask(
                state,
                problems,
                neurotransmitters,
                aggregation_method=self.multitask_config.fitness_aggregation,
                aggregation_weights=self.multitask_config.task_weights,
                orthogonality_bonus=self.multitask_config.orthogonality_bonus,
                specialization_bonus=self.multitask_config.specialization_bonus,
            )
        else:
            # Single-task mode: use standard implementation
            return self.run_generation_verbose(state, problem, skip_metrics)

    def run_generation_verbose(
        self,
        state: Any,
        problem: Any,
        skip_metrics: bool = False,
        device_id: int = 0,
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation with all extended features.

        This overrides the parent's run_generation_verbose to add:
        - Phase 2 discovery with multi-hop expansion
        - H→H caching
        - Extended metrics tracking

        Args:
            state: Current algorithm state
            problem: Problem instance
            skip_metrics: If True, skip metrics computation
            device_id: Device index for multi-GPU contexts (default 0)

        Returns:
            Tuple of (new_state, metrics)
        """
        step_times = {}
        total_start = time.perf_counter()

        # Opt-in reproduction of HMR's pre-ask key split (default off; see create_config).
        # Mirrors hmrhyperneat_dynamic_functions.run_generation_verbose so that
        # extra_randkey_split=True reproduces the published per-seed results bit-for-bit.
        if getattr(self, '_repro_extra_randkey_split', False):
            _, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # === STEP 0: CPPN Ask + Transform ===
        start = time.perf_counter()
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        self._cached_cppns_transformed = cppns_transformed  # Cache for external eval (swarm)
        step_times['step0_cppn_ask_transform'] = time.perf_counter() - start

        # === STEP 1-4: Build W1/W2 (STREAMING or STANDARD path) ===
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions

        if self.enable_streaming:
            # =================================================================
            # STREAMING PATH: Level-by-level CPPN queries for memory efficiency
            # =================================================================
            # Reduces peak memory from O(total_positions) to O(max_level_size)
            # Enables depth 7+ on CPU without OOM
            # Trade-off: 3-10x slower per generation
            # =================================================================
            if self.verbose:
                print(f"[Streaming] Enabled for depth {self.max_depth} ({total_positions} positions)")

            W1, W2, masks_A = self._build_matrices_streaming(
                state, cppns_transformed, h_grid, step_times
            )

            step_times['step1_cppn_queries'] = 0.0  # Included in streaming_build
            step_times['step2_variance_masks'] = 0.0  # Included in streaming_build
            step_times['step4_build_matrices'] = step_times.get('streaming_build', 0.0)
        else:
            # =================================================================
            # STANDARD PATH: Query all positions at once (faster, more memory)
            # =================================================================
            start = time.perf_counter()
            all_positions = h_grid.all_positions

            input_coords = self._cached_input_coords
            output_coords = self._cached_output_coords

            # -----------------------------------------------------------------
            # CPPN Query: standard (single output) or multi-output
            # -----------------------------------------------------------------
            if self.num_cppn_outputs > 1:
                # Multi-output CPPN (activation/aggregation have dedicated outputs)
                # Use the multi-output query path which preserves the output dimension
                # Returns: (pop_size, num_inputs, total_positions, num_cppn_outputs)
                input_all_multi = batch_query_population_multi_source_multi_output(
                    state, cppns_transformed, input_coords, all_positions,
                    True, self._jitted_cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )
                # Extract weight channel (output 0)
                input_all_weights = input_all_multi[:, :, :, 0]  # (pop, inputs, positions)

                output_all_multi = batch_query_population_multi_source_multi_output(
                    state, cppns_transformed, output_coords, all_positions,
                    False, self._jitted_cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )
                output_all_weights = output_all_multi[:, :, :, 0]  # (pop, outputs, positions)

                if self.verbose:
                    print(f"[Multi-output CPPN] {self.num_cppn_outputs} outputs, "
                          f"indices={self._cppn_output_indices}")
            else:
                # Standard mode: Query CPPN for 1 output (weight only)
                # Input→all positions: (pop_size, num_inputs, total_positions)
                input_all_weights = batch_query_population_multi_source_chunked(
                    state, cppns_transformed, input_coords, all_positions,
                    True, self._jitted_cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )

                # Output←all positions: (pop_size, num_outputs, total_positions)
                output_all_weights = batch_query_population_multi_source_chunked(
                    state, cppns_transformed, output_coords, all_positions,
                    False, self._jitted_cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )

            step_times['step1_cppn_queries'] = time.perf_counter() - start

            # === STEP 2: Variance + Subdivision Masks ===
            start = time.perf_counter()
            # Use input[0]→all_positions as variance source (2D: pop_size, total_positions)
            all_weights_for_variance = input_all_weights[:, 0, :]  # (pop_size, total_positions)
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )
            step_times['step2_variance_masks'] = time.perf_counter() - start

            # === STEP 4: Build W1/W2 Matrices ===
            start = time.perf_counter()
            weight_thresh = 0.1  # Local constant (same as parent)
            max_weight = self.max_weight

            # Broadcast mask: (pop_size, 1, total_positions) for weight masking
            active_mask_broadcast = masks_A[:, None, :]

            # -----------------------------------------------------------------
            # Weight Processing: standard threshold-based sparsification
            # -----------------------------------------------------------------
            # Standard mode: simple threshold-based sparsification
            # Apply tanh + max_weight scaling, then mask
            W1_raw = jnp.tanh(input_all_weights) * max_weight

            # Apply locality penalty for geometry seeding (if configured)
            # W1: input_coords → all_positions, so source=input, target=hidden
            if self.locality_radius is not None:
                W1_raw = apply_locality_penalty_to_weights(
                    W1_raw, input_coords, all_positions, self.locality_radius
                )
                if self.verbose:
                    print(f"[Locality] Applied penalty (r={self.locality_radius}) to W1")

            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )
            # W1 shape: (pop, num_inputs, total_positions) - correct for matmul with transposed later

            W2_raw = jnp.tanh(output_all_weights) * max_weight

            # Apply locality penalty for W2: all_positions → output_coords
            # Note: W2_raw is (pop, num_outputs, total_positions) before transpose
            # So source=outputs, target=hidden - we need to apply it correctly
            if self.locality_radius is not None:
                # Transpose temporarily to match (pop, hidden, outputs) for penalty
                W2_for_penalty = W2_raw.transpose(0, 2, 1)  # (pop, total_positions, num_outputs)
                W2_for_penalty = apply_locality_penalty_to_weights(
                    W2_for_penalty, all_positions, output_coords, self.locality_radius
                )
                W2_raw = W2_for_penalty.transpose(0, 2, 1)  # Back to (pop, num_outputs, total_positions)
                if self.verbose:
                    print(f"[Locality] Applied penalty (r={self.locality_radius}) to W2")

            W2 = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )
            W2 = W2.transpose(0, 2, 1)  # (pop, num_outputs, total_positions) → (pop, total_positions, num_outputs)

            step_times['step4_build_matrices'] = time.perf_counter() - start

        # === STEP 3: Phase 2 Discovery (with multi-hop) ===
        # This step runs AFTER matrices are built (needs masks_A)
        start = time.perf_counter()
        sparse_hh = None
        cache_hit = False

        if self.extended_config is not None and self.extended_config.allow_hidden_to_hidden:
            # DIAGNOSTIC: Log config being used
            if self.verbose:
                print(f"[Gen {self._current_generation}] H→H config: "
                      f"iteration_level={self.extended_config.iteration_level}, "
                      f"cache_enabled={self.extended_config.hh_cache_enabled}")

            # Check cache
            if self._hh_cache is not None and not self._hh_cache.should_refresh(
                self._current_generation, masks_A
            ):
                # Cache hit
                sparse_hh = self._hh_cache.get_cached()
                cache_hit = True
                if self.verbose:
                    print(f"[Gen {self._current_generation}] CACHE HIT - reusing connections")
            else:
                # Need to discover
                if self.verbose:
                    print(f"[Gen {self._current_generation}] CACHE MISS - running discovery")
                sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                    state, cppns_transformed, h_grid, masks_A,
                    self.band_threshold, self.max_weight,
                    self.extended_config, self._jitted_cppn_forward,
                    verbose=self.verbose,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                    num_cppn_outputs=self.num_cppn_outputs,
                )

                # Update cache
                if self._hh_cache is not None:
                    self._hh_cache.update_cache(sparse_hh, masks_A, self._current_generation)

        step_times['step3_phase2_discovery'] = time.perf_counter() - start

        # Cache weight matrices for multi-task evaluation
        # These are used by run_generation_multitask() to evaluate on multiple tasks
        # with different NT vectors without rebuilding W1/W2
        self._cached_W1 = W1
        self._cached_W2 = W2

        # Query self-connections for per-node properties (activation/aggregation)
        # when using multi-output CPPNs with dedicated outputs
        self._cached_self_conn = None
        _need_self_conn = self.num_cppn_outputs > 1 and (
            self.dynamic_functions_mode == 'cppn_output' or self.agg_mode == 'cppn_output'
        )
        # R7: also query self-connections when the neuromod decode is configured to use
        # them as the receptor/base-gain source (the HMR/paper decode). Default off.
        _need_self_conn = _need_self_conn or (
            self.neuromod_config is not None
            and getattr(self.neuromod_config, 'use_self_connection_query', False)
        )
        if _need_self_conn:
            self_conn_positions = h_grid.all_positions
            self_conn_outputs = batch_query_population_self_connections(
                state, cppns_transformed, self_conn_positions,
                self._jitted_cppn_forward,
                num_cppn_outputs=self.num_cppn_outputs,
            )
            self._cached_self_conn = self_conn_outputs
            if self.verbose:
                print(f"[CPPN Self-Conn] shape={self_conn_outputs.shape}, "
                      f"indices={self._cppn_output_indices}")

        # Cache sparse H→H connections for multi-task evaluation
        # This enables run_generation_multitask() to use H→H connections
        self._cached_sparse_hh = sparse_hh

        # === DIAGNOSTIC: Log weight matrix statistics for feedforward investigation ===
        if self.verbose and self._forward_mode == ForwardPassMode.DENSE_ONLY:
            active_counts = masks_A.sum(axis=1)  # Per-genome active position counts
            W1_nonzero = (W1 != 0).sum(axis=(1, 2))  # Per-genome non-zero W1 entries
            W2_nonzero = (W2 != 0).sum(axis=(1, 2))  # Per-genome non-zero W2 entries
            print(f"[Feedforward Diag] Active positions: mean={float(active_counts.mean()):.1f}, "
                  f"min={int(active_counts.min())}, max={int(active_counts.max())}")
            print(f"[Feedforward Diag] W1 non-zero: mean={float(W1_nonzero.mean()):.1f}, "
                  f"W2 non-zero: mean={float(W2_nonzero.mean()):.1f}")
            print(f"[Feedforward Diag] W1 stats: min={float(W1.min()):.3f}, max={float(W1.max()):.3f}, "
                  f"mean={float(W1.mean()):.4f}")
            print(f"[Feedforward Diag] W2 stats: min={float(W2.min()):.3f}, max={float(W2.max()):.3f}, "
                  f"mean={float(W2.mean()):.4f}")
            print(f"[Feedforward Diag] Total positions: {total_positions}, "
                  f"Variance threshold: {self.variance_threshold}")

        # === STEP 5: Evaluation ===
        start = time.perf_counter()
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # Get chunk size from config (None or 0 = no chunking, -1 = auto)
        eval_chunk_size = getattr(self.extended_config, 'eval_chunk_size', None) if self.extended_config else None
        pop_size = W1.shape[0]
        total_positions = W1.shape[2] if len(W1.shape) > 2 else W1.shape[1]

        # Auto-compute chunk size based on position count (same heuristics as CPPN queries)
        if eval_chunk_size == -1:
            if total_positions < 1000:        # depth <= 3
                eval_chunk_size = None  # No chunking needed
            elif total_positions < 6000:      # depth 4-5
                eval_chunk_size = min(500, pop_size)
            elif total_positions < 50000:     # depth 6 "anomaly zone"
                eval_chunk_size = min(50, pop_size)
            elif total_positions < 200000:    # depth 7
                eval_chunk_size = min(200, pop_size)
            else:                             # depth 8+
                eval_chunk_size = min(50, pop_size)

        # =========================================================================
        # DYNAMIC FUNCTIONS: Compute per-node activation indices based on mode
        # =========================================================================
        act_indices = None  # Per-node activation indices (num_positions,)
        use_dynamic_eval = self.dynamic_functions_mode not in ('disabled',)

        if self.dynamic_functions_mode == 'global':
            # Global mode: all nodes use same activation (no per-node indices needed)
            # The hidden_act_fn is already set in _parse_dynamic_functions_config
            act_indices = None
        elif self.dynamic_functions_mode == 'weight_interpretation':
            # Derive activation indices from W1 weight patterns
            # W1 shape: (pop, num_inputs, num_positions)
            # For each position, compute activation index based on incoming weights
            if self.df_interpretation == 'magnitude_bio':
                # Use mean absolute weight to determine activation
                mean_abs = jnp.mean(jnp.abs(W1), axis=1)  # (pop, num_positions)
                normalized = -jnp.tanh(mean_abs * 2 - 1)
                act_indices = continuous_to_index(normalized, self.df_num_activations)
            elif self.df_interpretation == 'variance':
                weight_var = jnp.var(W1, axis=1)  # (pop, num_positions)
                normalized = jnp.tanh(weight_var * 5)
                act_indices = continuous_to_index(normalized, self.df_num_activations)
            elif self.df_interpretation == 'sign':
                mean_sign = jnp.mean(jnp.sign(W1), axis=1)  # (pop, num_positions)
                act_indices = continuous_to_index(mean_sign, self.df_num_activations)
            else:
                # Default to magnitude_bio
                mean_abs = jnp.mean(jnp.abs(W1), axis=1)
                normalized = -jnp.tanh(mean_abs * 2 - 1)
                act_indices = continuous_to_index(normalized, self.df_num_activations)
        elif self.dynamic_functions_mode == 'random_fixed':
            # Random per-node activation, fixed at first generation
            if self._random_fixed_indices is None:
                rng_seed = self._random_mode_seed or 42
                key = jax.random.PRNGKey(rng_seed)
                num_activations = len(self.df_palette) if self.df_palette is not None else self.df_num_activations
                self._random_fixed_indices = jax.random.randint(
                    key, (total_positions,), 0, num_activations, dtype=jnp.int32
                )
                if self.df_palette is not None:
                    # Map to actual palette indices
                    self._random_fixed_indices = self.df_palette[self._random_fixed_indices]
            # Broadcast to all genomes
            act_indices = jnp.broadcast_to(self._random_fixed_indices, (pop_size, total_positions))
        elif self.dynamic_functions_mode == 'random_generation':
            # Random per-node activation, re-randomized each generation
            key = jax.random.PRNGKey(self._current_generation + (self._random_mode_seed or 0))
            num_activations = len(self.df_palette) if self.df_palette is not None else self.df_num_activations
            random_indices = jax.random.randint(
                key, (total_positions,), 0, num_activations, dtype=jnp.int32
            )
            if self.df_palette is not None:
                random_indices = self.df_palette[random_indices]
            act_indices = jnp.broadcast_to(random_indices, (pop_size, total_positions))
        elif self.dynamic_functions_mode == 'cppn_output':
            # CPPN output mode: Derive activation indices from CPPN output
            num_activations = len(self.df_palette) if self.df_palette is not None else self.df_num_activations
            if self.num_cppn_outputs > 1 and self._cached_self_conn is not None:
                # Use dedicated CPPN output for activation (multi-output CPPN)
                act_output_idx = self._cppn_output_indices['activation']
                activation_raw = self._cached_self_conn[:, :, act_output_idx]  # (pop, positions)
                activation_raw = jnp.where(masks_A, activation_raw, 0.0)
            else:
                # Fallback: derive from weight matrix (1-output CPPN)
                # W1 shape: (pop, num_inputs, num_positions)
                activation_raw = jnp.mean(W1, axis=1)  # (pop, num_positions)
            act_indices = continuous_to_index(activation_raw, num_activations)
            if self.df_palette is not None:
                act_indices = self.df_palette[act_indices]

        # Step 4b: Compute H→H aggregation indices if dynamic mode enabled
        hh_agg_indices = None
        if self.hh_aggregation_mode == 'dynamic':
            if self.num_cppn_outputs > 1 and self._cached_self_conn is not None and self.agg_mode == 'cppn_output':
                # Use dedicated CPPN output for H→H aggregation (multi-output CPPN)
                agg_output_idx = self._cppn_output_indices['aggregation']
                agg_raw = self._cached_self_conn[:, :, agg_output_idx]  # (pop, positions)
                hh_agg_indices = continuous_to_index(agg_raw, self.agg_num_aggregations)
            else:
                # Fallback: Use weight interpretation logic for H→H aggregation
                if self.agg_interpretation == 'magnitude_bio':
                    mean_abs = jnp.mean(jnp.abs(W1), axis=1)  # (pop, num_positions)
                    normalized = -jnp.tanh(mean_abs * 2 - 1)
                    hh_agg_indices = continuous_to_index(normalized, self.agg_num_aggregations)
                elif self.agg_interpretation == 'variance':
                    weight_var = jnp.var(W1, axis=1)  # (pop, num_positions)
                    normalized = jnp.tanh(weight_var * 5)
                    hh_agg_indices = continuous_to_index(normalized, self.agg_num_aggregations)
                else:
                    # Default to magnitude_bio
                    mean_abs = jnp.mean(jnp.abs(W1), axis=1)
                    normalized = -jnp.tanh(mean_abs * 2 - 1)
                    hh_agg_indices = continuous_to_index(normalized, self.agg_num_aggregations)

        # Step 4c: Compute I→H aggregation indices based on aggregation mode
        ih_agg_indices = None
        ih_global_agg_fn = None
        if self.agg_mode == 'global':
            ih_global_agg_fn = AGGREGATION_FUNCTIONS.get(
                self.agg_global_function, AGGREGATION_FUNCTIONS['sum']
            )
        elif self.agg_mode == 'cppn_output':
            if self.num_cppn_outputs > 1 and self._cached_self_conn is not None:
                # Use dedicated CPPN output for aggregation (multi-output CPPN)
                agg_output_idx = self._cppn_output_indices['aggregation']
                agg_raw = self._cached_self_conn[:, :, agg_output_idx]  # (pop, positions)
                agg_raw = jnp.where(masks_A, agg_raw, 0.0)
            else:
                # Fallback: derive from weight matrix (1-output CPPN)
                agg_raw = jnp.mean(W1, axis=1)  # (pop, num_positions)
            ih_agg_indices = continuous_to_index(agg_raw, self.agg_num_aggregations)
        elif self.agg_mode == 'weight_interpretation':
            ih_agg_indices = aggregation_from_weight_interpretation(
                W1, self.agg_interpretation, self.agg_num_aggregations
            )
        elif self.agg_mode == 'random_fixed':
            if self._random_fixed_agg_indices is None:
                key = jax.random.PRNGKey(self._random_agg_seed or 42)
                self._random_fixed_agg_indices = jax.random.randint(
                    key, (total_positions,), 0, self.agg_num_aggregations, dtype=jnp.int32
                )
            ih_agg_indices = jnp.broadcast_to(
                self._random_fixed_agg_indices, (pop_size, total_positions)
            )
        elif self.agg_mode == 'random_generation':
            key = jax.random.PRNGKey(
                self._current_generation + (self._random_agg_seed or 0)
            )
            random_agg = jax.random.randint(
                key, (total_positions,), 0, self.agg_num_aggregations, dtype=jnp.int32
            )
            ih_agg_indices = jnp.broadcast_to(random_agg, (pop_size, total_positions))

        use_ih_aggregation = (ih_agg_indices is not None or ih_global_agg_fn is not None)

        # Cache per-node function indices for NAS gradient training extraction
        self._cached_act_indices = act_indices
        self._cached_hh_agg_indices = hh_agg_indices
        self._cached_ih_agg_indices = ih_agg_indices

        # =========================================================================
        # NEUROMODULATION: Derive receptor densities, base gains, and gates
        # =========================================================================
        # Storage for neuromodulation data (set to None if disabled)
        neuromod_data = None
        if self.neuromod_config and self.neuromod_config.mode != 'disabled':
            neuromod_mode = self.neuromod_config.mode
            neuromod_data = {'mode': neuromod_mode}

            if neuromod_mode == 'static_gating':
                # Level 1: Static gating from weight interpretation
                # Gates are derived from mean absolute weight magnitude
                # Higher magnitude weights → more active connections
                mean_abs_w = jnp.mean(jnp.abs(W1), axis=1)  # (pop, num_positions)
                gate_raw = jnp.tanh(mean_abs_w * 2)  # Normalize to roughly [-1, 1]
                static_gates = apply_gate_scaling(
                    gate_raw,
                    self.neuromod_config.gate_scaling,
                    self.neuromod_config.gate_threshold,
                    self.neuromod_config.gate_hardness,
                )  # (pop, num_positions)
                neuromod_data['static_gates'] = static_gates
                self._static_gates = static_gates

            elif neuromod_mode == 'context_gating':
                # Level 2: Context-dependent gating (XdG-style)
                # Gate values depend on task context
                context_dim = self.neuromod_config.context_dim
                context_influence = self.neuromod_config.context_influence

                # Derive base gates from weights (same as static gating)
                mean_abs_w = jnp.mean(jnp.abs(W1), axis=1)  # (pop, num_positions)
                base_gates = jnp.tanh(mean_abs_w * 2)

                # Derive per-context gate modulation from W1 patterns
                # Use variance across inputs to create context-specific modulation
                weight_var = jnp.var(W1, axis=1)  # (pop, num_positions)
                context_modulation = jnp.zeros((pop_size, context_dim, total_positions))

                # Create context-specific modulations using different weight patterns
                for ctx_idx in range(context_dim):
                    # Rotate/phase-shift the weight patterns for different contexts
                    phase_shift = ctx_idx * (2 * jnp.pi / context_dim)
                    ctx_pattern = jnp.sin(weight_var * 5 + phase_shift)
                    context_modulation = context_modulation.at[:, ctx_idx, :].set(ctx_pattern)

                neuromod_data['base_gates'] = base_gates
                neuromod_data['context_modulation'] = context_modulation
                neuromod_data['context_influence'] = context_influence
                self._context_gates = context_modulation

            elif neuromod_mode == 'modulatory_neurons':
                # Level 3: Modulatory neurons (Soltoggio-style)
                # A subset of neurons act as modulatory neurons
                mod_ratio = self.neuromod_config.mod_neuron_ratio
                num_mod_neurons = max(1, int(total_positions * mod_ratio))

                # Select modulatory neurons based on weight patterns
                # Neurons with highest variance become modulatory
                weight_var = jnp.var(W1, axis=1)  # (pop, num_positions)
                # Get top-k positions by variance for each genome
                # For simplicity, use same modulatory positions for all genomes
                mean_var = jnp.mean(weight_var, axis=0)  # (num_positions,)
                mod_indices = jnp.argsort(mean_var)[-num_mod_neurons:]
                mod_mask = jnp.zeros(total_positions, dtype=jnp.bool_)
                mod_mask = mod_mask.at[mod_indices].set(True)

                # Modulatory connections: derived from W1 patterns
                # Modulatory neurons affect regular neurons multiplicatively
                mod_connection_weights = jnp.tanh(weight_var) * self.neuromod_config.mod_decay

                neuromod_data['mod_mask'] = mod_mask
                neuromod_data['mod_connection_weights'] = mod_connection_weights
                neuromod_data['mod_connection_type'] = self.neuromod_config.mod_connection_type
                neuromod_data['mod_decay'] = self.neuromod_config.mod_decay
                self._mod_neuron_mask = mod_mask
                self._mod_connection_weights = mod_connection_weights

            elif neuromod_mode == 'true_neuromodulation':
                # Level 4: TRUE neuromodulation
                # NT vectors × receptor densities → gain modulation
                num_nt = self.neuromod_config.num_nt_types
                mod_strength = self.neuromod_config.modulation_strength
                receptor_derivation = self.neuromod_config.receptor_derivation

                if self.neuromod_config.receptor_from_weight:
                    # Option A: Derive receptor densities from weights
                    # W1 shape: (pop, num_inputs, num_positions)
                    #
                    # Use FIRST input's weights (not mean) to match original's single-value-per-position approach.
                    # The original uses CPPN output at self-connection coordinates (x,y,x,y,bias),
                    # which gives one value per position. Using first input's weights approximates this
                    # by giving one value per position rather than averaging multiple values.
                    #
                    # This is important for receptor_densities because:
                    # - mean() smooths out the values, losing the variation needed for diverse modulation
                    # - first input gives a consistent per-position property similar to self-connection query
                    # R7: choose the per-position weight source for receptor/base-gain.
                    if (getattr(self.neuromod_config, 'use_self_connection_query', False)
                            and self._cached_self_conn is not None):
                        # Exact HMR/paper reproduction: raw CPPN self-connection query at
                        # (x,y,x,y,bias); index 0 = weight output -- the SAME source HMR uses
                        # (weight_values = self_conn_outputs[:, :, 0]). Already bounded ~[-1, 1].
                        weight_normalized = self._cached_self_conn[:, :, 0]
                    else:
                        # Default EMR approximation: first input's weight, normalized to ~[-1, 1].
                        # W1 = tanh(cppn_output) * max_weight, so W1/max_weight ~= tanh(cppn_output).
                        weight_for_receptor = W1[:, 0, :]  # (pop, num_positions) - use first input
                        weight_normalized = weight_for_receptor / self.max_weight

                    receptor_densities = derive_receptor_from_weight(
                        weight=weight_normalized, method=receptor_derivation, num_nt_types=num_nt
                    )  # (pop, num_positions, num_nt)

                    # Convert to [-1, 1] range to match Option B (enables both excitatory and inhibitory)
                    # This matches the original emrhyperneat_neuromodulation_functions.py
                    receptor_densities = receptor_densities * 2.0 - 1.0

                    # Base gains: derived from weight magnitude, range [0.5, 1.0]
                    # Use normalized weight (already in [-1, 1]) to match original
                    base_gains = jnp.abs(weight_normalized) * 0.5 + 0.5
                else:
                    # Option B: Use dedicated CPPN outputs for receptor densities
                    # This requires multi-output CPPN support
                    raise NotImplementedError(
                        "receptor_from_weight=False requires multi-output CPPN support. "
                        "Use receptor_from_weight=True (Option A) for now."
                    )

                # Store for use in evaluation
                neuromod_data['receptor_densities'] = receptor_densities
                neuromod_data['base_gains'] = base_gains
                neuromod_data['num_nt'] = num_nt
                neuromod_data['mod_strength'] = mod_strength
                neuromod_data['modulation_mode'] = self.neuromod_config.modulation_mode
                neuromod_data['use_output_inversion'] = self.neuromod_config.use_output_inversion
                neuromod_data['branch_gating'] = self.neuromod_config.branch_gating

                # Get default NT vector for single-task evaluation
                if self.multitask_config and self._multitask_state.get('nt_vectors') is not None:
                    # Use first task's NT vector by default
                    neuromod_data['neurotransmitter'] = self._multitask_state['nt_vectors'][0]
                else:
                    # Default NT vector: [0.5, 0.5, 0.5, 1.0]
                    neuromod_data['neurotransmitter'] = jnp.array(
                        [0.5] * min(3, num_nt) + [1.0] if num_nt >= 4 else [0.5] * num_nt
                    )

                # Store in class attribute
                self._neuromod_true['receptor_densities'] = receptor_densities
                self._neuromod_true['base_gains'] = base_gains

        if sparse_hh is not None and self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH:
            # Hybrid evaluation with sparse h→h

            # Check for neuromodulation FIRST
            if neuromod_data is not None:
                # =====================================================================
                # NEUROMODULATED HYBRID SPARSE H→H EVALUATION
                # Applies neuromodulation to BOTH Input→Hidden AND H→H iterations
                # =====================================================================
                neuromod_mode = neuromod_data['mode']
                hidden_act_fn = self._hidden_act_fn if self._hidden_act_fn is not None else jnp.tanh
                output_act_fn = self._output_act_fn if self._output_act_fn is not None else jax.nn.sigmoid
                palette = self.df_palette
                num_activations = self.df_num_activations
                activate_time = self.activate_time

                if neuromod_mode == 'static_gating':
                    # Level 1: Static gating with H→H
                    static_gates = neuromod_data['static_gates']

                    def eval_single_hybrid_neuromod_static(
                        W1_i, W2_i, from_idx, to_idx, hh_w, mask, act_idx, gates
                    ):
                        """Hybrid forward with static gating applied to both I→H and H→H."""
                        # Input→Hidden
                        pre_h = inputs_batch @ W1_i
                        if act_idx is None:
                            h = hidden_act_fn(pre_h)
                        elif palette is not None:
                            h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                        else:
                            h = grouped_activation_forward(pre_h, act_idx, num_activations)
                        h = h * gates  # Apply static gates

                        # H→H iterations with gating
                        # Precompute safe indices and masked weights (JIT-compatible)
                        num_hidden = h.shape[1]
                        safe_from = jnp.clip(from_idx, 0, num_hidden - 1)
                        safe_to = jnp.clip(to_idx, 0, num_hidden - 1)
                        effective_hh_w = jnp.where(mask, hh_w, 0.0)

                        for _ in range(activate_time):
                            # Sparse H→H aggregation (JIT-compatible vectorized version)
                            source_vals = h[:, safe_from]  # (batch, num_conns)
                            contributions = source_vals * effective_hh_w  # (batch, num_conns)
                            hh_contrib = jnp.zeros_like(h)
                            hh_contrib = hh_contrib.at[:, safe_to].add(contributions)
                            # Apply gating to H→H contribution
                            hh_contrib = hh_contrib * gates
                            h = h + jnp.tanh(hh_contrib)

                        outputs = output_act_fn(h @ W2_i)
                        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

                    if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                        fitness_chunks = []
                        for chunk_start in range(0, pop_size, eval_chunk_size):
                            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                            chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                            chunk_fitnesses = jax.vmap(eval_single_hybrid_neuromod_static)(
                                W1[chunk_start:chunk_end],
                                W2[chunk_start:chunk_end],
                                sparse_hh.from_indices[chunk_start:chunk_end],
                                sparse_hh.to_indices[chunk_start:chunk_end],
                                sparse_hh.weights[chunk_start:chunk_end],
                                sparse_hh.valid_mask[chunk_start:chunk_end],
                                chunk_act,
                                static_gates[chunk_start:chunk_end],
                            )
                            fitness_chunks.append(chunk_fitnesses)
                        fitnesses = jnp.concatenate(fitness_chunks)
                    else:
                        fitnesses = jax.vmap(eval_single_hybrid_neuromod_static)(
                            W1, W2,
                            sparse_hh.from_indices, sparse_hh.to_indices,
                            sparse_hh.weights, sparse_hh.valid_mask,
                            act_indices, static_gates
                        )

                elif neuromod_mode == 'context_gating':
                    # Level 2: Context-dependent gating with H→H (XdG-style)
                    base_gates = neuromod_data['base_gates']
                    context_mod = neuromod_data['context_modulation']
                    context_influence = neuromod_data['context_influence']
                    context_idx = self._multitask_state.get('current_task_idx', 0)

                    def eval_single_hybrid_neuromod_context(
                        W1_i, W2_i, from_idx, to_idx, hh_w, mask, act_idx, base_g, ctx_mod
                    ):
                        """Hybrid forward with context gating applied to both I→H and H→H."""
                        # Compute context-modulated gates
                        gates = jax.nn.sigmoid(base_g + context_influence * ctx_mod[context_idx])

                        # Input→Hidden
                        pre_h = inputs_batch @ W1_i
                        if act_idx is None:
                            h = hidden_act_fn(pre_h)
                        elif palette is not None:
                            h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                        else:
                            h = grouped_activation_forward(pre_h, act_idx, num_activations)
                        h = h * gates

                        # H→H iterations with context gating
                        # Precompute safe indices and masked weights (JIT-compatible)
                        num_hidden = h.shape[1]
                        safe_from = jnp.clip(from_idx, 0, num_hidden - 1)
                        safe_to = jnp.clip(to_idx, 0, num_hidden - 1)
                        effective_hh_w = jnp.where(mask, hh_w, 0.0)

                        for _ in range(activate_time):
                            # Sparse H→H aggregation (JIT-compatible vectorized version)
                            source_vals = h[:, safe_from]
                            contributions = source_vals * effective_hh_w
                            hh_contrib = jnp.zeros_like(h)
                            hh_contrib = hh_contrib.at[:, safe_to].add(contributions)
                            hh_contrib = hh_contrib * gates
                            h = h + jnp.tanh(hh_contrib)

                        outputs = output_act_fn(h @ W2_i)
                        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

                    if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                        fitness_chunks = []
                        for chunk_start in range(0, pop_size, eval_chunk_size):
                            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                            chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                            chunk_fitnesses = jax.vmap(eval_single_hybrid_neuromod_context)(
                                W1[chunk_start:chunk_end],
                                W2[chunk_start:chunk_end],
                                sparse_hh.from_indices[chunk_start:chunk_end],
                                sparse_hh.to_indices[chunk_start:chunk_end],
                                sparse_hh.weights[chunk_start:chunk_end],
                                sparse_hh.valid_mask[chunk_start:chunk_end],
                                chunk_act,
                                base_gates[chunk_start:chunk_end],
                                context_mod[chunk_start:chunk_end],
                            )
                            fitness_chunks.append(chunk_fitnesses)
                        fitnesses = jnp.concatenate(fitness_chunks)
                    else:
                        fitnesses = jax.vmap(eval_single_hybrid_neuromod_context)(
                            W1, W2,
                            sparse_hh.from_indices, sparse_hh.to_indices,
                            sparse_hh.weights, sparse_hh.valid_mask,
                            act_indices, base_gates, context_mod
                        )

                elif neuromod_mode == 'modulatory_neurons':
                    # Level 3: Modulatory neurons with H→H (Soltoggio-style)
                    mod_mask = neuromod_data['mod_mask']
                    mod_weights = neuromod_data['mod_connection_weights']
                    mod_type = neuromod_data['mod_connection_type']
                    mod_decay = neuromod_data['mod_decay']

                    def eval_single_hybrid_neuromod_neurons(
                        W1_i, W2_i, from_idx, to_idx, hh_w, mask, act_idx, mod_w
                    ):
                        """Hybrid forward with modulatory neurons affecting both I→H and H→H."""
                        # Input→Hidden
                        pre_h = inputs_batch @ W1_i
                        if act_idx is None:
                            h = hidden_act_fn(pre_h)
                        elif palette is not None:
                            h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                        else:
                            h = grouped_activation_forward(pre_h, act_idx, num_activations)

                        # Compute initial modulatory signal
                        mod_activity = jnp.mean(h[:, mod_mask], axis=1, keepdims=True)
                        mod_signal = jnp.tanh(mod_activity) * mod_decay

                        # Apply initial modulation
                        if mod_type == 'multiplicative':
                            h = h * (1.0 + mod_signal * mod_w)
                        elif mod_type == 'additive':
                            h = h + mod_signal * mod_w
                        else:  # gating
                            h = h * jax.nn.sigmoid(mod_signal * mod_w)

                        # H→H iterations with modulatory neuron influence
                        # Precompute safe indices and masked weights (JIT-compatible)
                        num_hidden = h.shape[1]
                        safe_from = jnp.clip(from_idx, 0, num_hidden - 1)
                        safe_to = jnp.clip(to_idx, 0, num_hidden - 1)
                        effective_hh_w = jnp.where(mask, hh_w, 0.0)

                        for _ in range(activate_time):
                            # Sparse H→H aggregation (JIT-compatible vectorized version)
                            source_vals = h[:, safe_from]
                            contributions = source_vals * effective_hh_w
                            hh_contrib = jnp.zeros_like(h)
                            hh_contrib = hh_contrib.at[:, safe_to].add(contributions)

                            # Update modulatory signal from current hidden state
                            mod_activity = jnp.mean(h[:, mod_mask], axis=1, keepdims=True)
                            mod_signal = jnp.tanh(mod_activity) * mod_decay

                            # Apply modulation to H→H contribution
                            if mod_type == 'multiplicative':
                                hh_contrib = hh_contrib * (1.0 + mod_signal * mod_w)
                            elif mod_type == 'additive':
                                hh_contrib = hh_contrib + mod_signal * mod_w
                            else:
                                hh_contrib = hh_contrib * jax.nn.sigmoid(mod_signal * mod_w)

                            h = h + jnp.tanh(hh_contrib)

                        outputs = output_act_fn(h @ W2_i)
                        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

                    if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                        fitness_chunks = []
                        for chunk_start in range(0, pop_size, eval_chunk_size):
                            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                            chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                            chunk_fitnesses = jax.vmap(eval_single_hybrid_neuromod_neurons)(
                                W1[chunk_start:chunk_end],
                                W2[chunk_start:chunk_end],
                                sparse_hh.from_indices[chunk_start:chunk_end],
                                sparse_hh.to_indices[chunk_start:chunk_end],
                                sparse_hh.weights[chunk_start:chunk_end],
                                sparse_hh.valid_mask[chunk_start:chunk_end],
                                chunk_act,
                                mod_weights[chunk_start:chunk_end],
                            )
                            fitness_chunks.append(chunk_fitnesses)
                        fitnesses = jnp.concatenate(fitness_chunks)
                    else:
                        fitnesses = jax.vmap(eval_single_hybrid_neuromod_neurons)(
                            W1, W2,
                            sparse_hh.from_indices, sparse_hh.to_indices,
                            sparse_hh.weights, sparse_hh.valid_mask,
                            act_indices, mod_weights
                        )

                elif neuromod_mode == 'true_neuromodulation':
                    # Level 4: TRUE neuromodulation with H→H (NT vectors + receptor densities)
                    # Uses the pre-defined forward_hybrid_vmapped_neuromodulated function
                    receptor_densities = neuromod_data['receptor_densities']
                    base_gains = neuromod_data['base_gains']
                    neurotransmitter = neuromod_data['neurotransmitter']
                    mod_strength = neuromod_data['mod_strength']
                    modulation_mode = neuromod_data['modulation_mode']
                    use_output_inversion = neuromod_data['use_output_inversion']
                    num_nt = neuromod_data['num_nt']
                    hh_agg_mode = self.hh_aggregation_mode
                    num_hh_agg = self.agg_num_aggregations

                    def eval_single_hybrid_true_neuromod(
                        W1_i, W2_i, from_idx, to_idx, hh_w, mask,
                        act_idx, hh_agg_idx, receptors, base_g
                    ):
                        """Hybrid forward with TRUE neuromodulation on both I→H and H→H."""
                        # Compute modulation from receptors × NT
                        nt_mod = neurotransmitter[:min(3, num_nt)]
                        modulation = receptors[:, :min(3, num_nt)] @ nt_mod
                        effective_gains = base_g + mod_strength * modulation
                        gates = jax.nn.sigmoid(modulation * 5.0)

                        # Input→Hidden with neuromodulation
                        pre_h = inputs_batch @ W1_i

                        if modulation_mode == 'gating_only':
                            if act_idx is None:
                                h = hidden_act_fn(pre_h)
                            elif palette is not None:
                                h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                            else:
                                h = grouped_activation_forward(pre_h, act_idx, num_activations)
                            h = h * gates
                        elif modulation_mode == 'gain_bias_only':
                            if act_idx is None:
                                h = hidden_act_fn(effective_gains * pre_h)
                            elif palette is not None:
                                h = grouped_activation_forward_with_palette(
                                    effective_gains * pre_h, act_idx, palette
                                )
                            else:
                                h = grouped_activation_forward(
                                    effective_gains * pre_h, act_idx, num_activations
                                )
                        else:  # 'full' mode
                            if act_idx is None:
                                h = hidden_act_fn(effective_gains * pre_h)
                            elif palette is not None:
                                h = grouped_activation_forward_with_palette(
                                    effective_gains * pre_h, act_idx, palette
                                )
                            else:
                                h = grouped_activation_forward(
                                    effective_gains * pre_h, act_idx, num_activations
                                )
                            h = h * gates

                        # H→H iterations with TRUE neuromodulation
                        # Precompute safe indices and masked weights (JIT-compatible)
                        num_hidden = h.shape[1]
                        safe_from = jnp.clip(from_idx, 0, num_hidden - 1)
                        safe_to = jnp.clip(to_idx, 0, num_hidden - 1)
                        effective_hh_w = jnp.where(mask, hh_w, 0.0)

                        for _ in range(activate_time):
                            # Sparse H→H aggregation (JIT-compatible vectorized version)
                            source_vals = h[:, safe_from]
                            contributions = source_vals * effective_hh_w
                            hh_contrib = jnp.zeros_like(h)
                            hh_contrib = hh_contrib.at[:, safe_to].add(contributions)

                            # Apply neuromodulation to H→H contribution
                            if modulation_mode == 'gating_only':
                                hh_update = jnp.tanh(hh_contrib) * gates
                            elif modulation_mode == 'gain_bias_only':
                                hh_update = jnp.tanh(effective_gains * hh_contrib)
                            else:  # 'full'
                                hh_update = jnp.tanh(effective_gains * hh_contrib) * gates

                            h = h + hh_update

                        # Hidden→Output
                        outputs = output_act_fn(h @ W2_i)

                        # NT4 output inversion control
                        if use_output_inversion and num_nt >= 4:
                            invert = neurotransmitter[3]
                            outputs = invert * outputs + (1 - invert) * (1 - outputs)

                        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

                    if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                        fitness_chunks = []
                        for chunk_start in range(0, pop_size, eval_chunk_size):
                            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                            chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                            chunk_hh_agg = hh_agg_indices[chunk_start:chunk_end] if hh_agg_indices is not None else None
                            chunk_fitnesses = jax.vmap(eval_single_hybrid_true_neuromod)(
                                W1[chunk_start:chunk_end],
                                W2[chunk_start:chunk_end],
                                sparse_hh.from_indices[chunk_start:chunk_end],
                                sparse_hh.to_indices[chunk_start:chunk_end],
                                sparse_hh.weights[chunk_start:chunk_end],
                                sparse_hh.valid_mask[chunk_start:chunk_end],
                                chunk_act,
                                chunk_hh_agg,
                                receptor_densities[chunk_start:chunk_end],
                                base_gains[chunk_start:chunk_end],
                            )
                            fitness_chunks.append(chunk_fitnesses)
                        fitnesses = jnp.concatenate(fitness_chunks)
                    else:
                        fitnesses = jax.vmap(eval_single_hybrid_true_neuromod)(
                            W1, W2,
                            sparse_hh.from_indices, sparse_hh.to_indices,
                            sparse_hh.weights, sparse_hh.valid_mask,
                            act_indices, hh_agg_indices,
                            receptor_densities, base_gains
                        )
                else:
                    raise ValueError(f"Unknown neuromodulation mode: {neuromod_mode}")

            elif use_dynamic_eval and (act_indices is not None or self._hidden_act_fn is not None):
                # Use dynamic evaluation with per-node or global activation functions
                # Create wrapper function to avoid functools.partial issues with vmap
                hidden_act_fn = self._hidden_act_fn
                output_act_fn = self._output_act_fn
                num_activations = self.df_num_activations
                palette = self.df_palette
                hh_activation_mode = self.hh_activation_mode
                activate_time = self.activate_time
                hh_aggregation_mode = self.hh_aggregation_mode
                num_hh_aggregations = self.agg_num_aggregations

                def eval_single_hybrid_dynamic_wrapper(
                    W1_i, W2_i, from_idx, to_idx, hh_w, mask, act_idx, hh_agg_idx
                ):
                    return eval_single_network_hybrid_dynamic(
                        W1=W1_i,
                        W2=W2_i,
                        from_indices=from_idx,
                        to_indices=to_idx,
                        hh_weights=hh_w,
                        valid_mask=mask,
                        inputs=inputs_batch,
                        targets=targets_batch,
                        activate_time=activate_time,
                        act_indices=act_idx,
                        hidden_act_fn=hidden_act_fn,
                        output_act_fn=output_act_fn,
                        num_activations=num_activations,
                        palette=palette,
                        hh_activation_mode=hh_activation_mode,
                        hh_agg_indices=hh_agg_idx,
                        hh_aggregation_mode=hh_aggregation_mode,
                        num_hh_aggregations=num_hh_aggregations,
                    )

                if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                    # Chunked evaluation with dynamic functions
                    fitness_chunks = []
                    for chunk_start in range(0, pop_size, eval_chunk_size):
                        chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                        # Get act_indices and hh_agg_indices for this chunk
                        chunk_act_indices = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                        chunk_hh_agg_indices = hh_agg_indices[chunk_start:chunk_end] if hh_agg_indices is not None else None
                        chunk_fitnesses = jax.vmap(eval_single_hybrid_dynamic_wrapper)(
                            W1[chunk_start:chunk_end],
                            W2[chunk_start:chunk_end],
                            sparse_hh.from_indices[chunk_start:chunk_end],
                            sparse_hh.to_indices[chunk_start:chunk_end],
                            sparse_hh.weights[chunk_start:chunk_end],
                            sparse_hh.valid_mask[chunk_start:chunk_end],
                            chunk_act_indices,
                            chunk_hh_agg_indices,
                        )
                        fitness_chunks.append(chunk_fitnesses)
                    fitnesses = jnp.concatenate(fitness_chunks)
                else:
                    # Full population vmap with dynamic functions
                    fitnesses = jax.vmap(eval_single_hybrid_dynamic_wrapper)(
                        W1, W2,
                        sparse_hh.from_indices, sparse_hh.to_indices,
                        sparse_hh.weights, sparse_hh.valid_mask,
                        act_indices,
                        hh_agg_indices,
                    )
            else:
                # Original evaluation (disabled mode)
                eval_fn = functools.partial(
                    eval_single_network_hybrid,
                    inputs=inputs_batch,
                    targets=targets_batch,
                    activate_time=self.activate_time,
                )

                if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                    # Chunked evaluation to prevent OOM
                    fitness_chunks = []
                    for chunk_start in range(0, pop_size, eval_chunk_size):
                        chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                        chunk_fitnesses = jax.vmap(eval_fn)(
                            W1[chunk_start:chunk_end],
                            W2[chunk_start:chunk_end],
                            sparse_hh.from_indices[chunk_start:chunk_end],
                            sparse_hh.to_indices[chunk_start:chunk_end],
                            sparse_hh.weights[chunk_start:chunk_end],
                            sparse_hh.valid_mask[chunk_start:chunk_end],
                        )
                        fitness_chunks.append(chunk_fitnesses)
                    fitnesses = jnp.concatenate(fitness_chunks)
                else:
                    # Full population vmap (original behavior)
                    fitnesses = jax.vmap(eval_fn)(
                        W1, W2,
                        sparse_hh.from_indices, sparse_hh.to_indices,
                        sparse_hh.weights, sparse_hh.valid_mask,
                    )
        else:
            # Dense-only evaluation (DENSE_ONLY mode)
            hidden_act_fn = self._hidden_act_fn if self._hidden_act_fn is not None else jnp.tanh
            output_act_fn = self._output_act_fn if self._output_act_fn is not None else jax.nn.sigmoid
            palette = self.df_palette
            num_activations = self.df_num_activations

            # Check for neuromodulation
            if neuromod_data is not None:
                # =====================================================================
                # NEUROMODULATED DENSE-ONLY EVALUATION
                # =====================================================================
                neuromod_mode = neuromod_data['mode']

                if neuromod_mode == 'static_gating':
                    # Level 1: Static gating
                    static_gates = neuromod_data['static_gates']

                    def eval_single_dense_neuromod_static(W1_i, W2_i, act_idx, gates):
                        pre_h = inputs_batch @ W1_i  # (n_samples, total_positions)

                        if act_idx is None:
                            h = hidden_act_fn(pre_h)
                        elif palette is not None:
                            h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                        else:
                            h = grouped_activation_forward(pre_h, act_idx, num_activations)

                        # Apply static gates
                        h = h * gates  # (n_samples, total_positions)

                        outputs = output_act_fn(h @ W2_i)
                        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

                    if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                        fitness_chunks = []
                        for chunk_start in range(0, pop_size, eval_chunk_size):
                            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                            chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                            chunk_fitnesses = jax.vmap(eval_single_dense_neuromod_static)(
                                W1[chunk_start:chunk_end],
                                W2[chunk_start:chunk_end],
                                chunk_act,
                                static_gates[chunk_start:chunk_end],
                            )
                            fitness_chunks.append(chunk_fitnesses)
                        fitnesses = jnp.concatenate(fitness_chunks)
                    else:
                        fitnesses = jax.vmap(eval_single_dense_neuromod_static)(
                            W1, W2, act_indices, static_gates
                        )

                elif neuromod_mode == 'context_gating':
                    # Level 2: Context-dependent gating (XdG-style)
                    base_gates = neuromod_data['base_gates']
                    context_mod = neuromod_data['context_modulation']
                    context_influence = neuromod_data['context_influence']
                    # Use current context or default to first context
                    context_idx = self._multitask_state.get('current_task_idx', 0)

                    def eval_single_dense_neuromod_context(W1_i, W2_i, act_idx, base_g, ctx_mod):
                        pre_h = inputs_batch @ W1_i

                        if act_idx is None:
                            h = hidden_act_fn(pre_h)
                        elif palette is not None:
                            h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                        else:
                            h = grouped_activation_forward(pre_h, act_idx, num_activations)

                        # Compute context-modulated gates
                        gates = jax.nn.sigmoid(
                            base_g + context_influence * ctx_mod[context_idx]
                        )
                        h = h * gates

                        outputs = output_act_fn(h @ W2_i)
                        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

                    if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                        fitness_chunks = []
                        for chunk_start in range(0, pop_size, eval_chunk_size):
                            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                            chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                            chunk_fitnesses = jax.vmap(eval_single_dense_neuromod_context)(
                                W1[chunk_start:chunk_end],
                                W2[chunk_start:chunk_end],
                                chunk_act,
                                base_gates[chunk_start:chunk_end],
                                context_mod[chunk_start:chunk_end],
                            )
                            fitness_chunks.append(chunk_fitnesses)
                        fitnesses = jnp.concatenate(fitness_chunks)
                    else:
                        fitnesses = jax.vmap(eval_single_dense_neuromod_context)(
                            W1, W2, act_indices, base_gates, context_mod
                        )

                elif neuromod_mode == 'modulatory_neurons':
                    # Level 3: Modulatory neurons (Soltoggio-style)
                    mod_mask = neuromod_data['mod_mask']
                    mod_weights = neuromod_data['mod_connection_weights']
                    mod_type = neuromod_data['mod_connection_type']
                    mod_decay = neuromod_data['mod_decay']

                    def eval_single_dense_neuromod_neurons(W1_i, W2_i, act_idx, mod_w):
                        pre_h = inputs_batch @ W1_i

                        if act_idx is None:
                            h = hidden_act_fn(pre_h)
                        elif palette is not None:
                            h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                        else:
                            h = grouped_activation_forward(pre_h, act_idx, num_activations)

                        # Compute modulatory neuron output
                        mod_activity = jnp.mean(h[:, mod_mask], axis=1, keepdims=True)  # (n_samples, 1)
                        mod_signal = jnp.tanh(mod_activity) * mod_decay

                        # Apply modulation
                        if mod_type == 'multiplicative':
                            h = h * (1.0 + mod_signal * mod_w)
                        elif mod_type == 'additive':
                            h = h + mod_signal * mod_w
                        else:  # gating
                            h = h * jax.nn.sigmoid(mod_signal * mod_w)

                        outputs = output_act_fn(h @ W2_i)
                        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

                    if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                        fitness_chunks = []
                        for chunk_start in range(0, pop_size, eval_chunk_size):
                            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                            chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                            chunk_fitnesses = jax.vmap(eval_single_dense_neuromod_neurons)(
                                W1[chunk_start:chunk_end],
                                W2[chunk_start:chunk_end],
                                chunk_act,
                                mod_weights[chunk_start:chunk_end],
                            )
                            fitness_chunks.append(chunk_fitnesses)
                        fitnesses = jnp.concatenate(fitness_chunks)
                    else:
                        fitnesses = jax.vmap(eval_single_dense_neuromod_neurons)(
                            W1, W2, act_indices, mod_weights
                        )

                elif neuromod_mode == 'true_neuromodulation':
                    # Level 4: TRUE neuromodulation (NT vectors + receptor densities)
                    receptor_densities = neuromod_data['receptor_densities']
                    base_gains = neuromod_data['base_gains']
                    neurotransmitter = neuromod_data['neurotransmitter']
                    mod_strength = neuromod_data['mod_strength']
                    modulation_mode = neuromod_data['modulation_mode']
                    use_output_inversion = neuromod_data['use_output_inversion']
                    num_nt = neuromod_data['num_nt']

                    def eval_single_dense_true_neuromod(
                        W1_i, W2_i, act_idx, receptors, base_g
                    ):
                        pre_h = inputs_batch @ W1_i  # (n_samples, total_positions)

                        # Compute modulation from receptors × NT
                        # receptors: (total_positions, num_nt), NT: (num_nt,)
                        # Use first 3 NT types for modulation
                        nt_mod = neurotransmitter[:min(3, num_nt)]
                        modulation = receptors[:, :min(3, num_nt)] @ nt_mod  # (total_positions,)

                        # Compute effective gains and gates
                        effective_gains = base_g + mod_strength * modulation
                        gates = jax.nn.sigmoid(modulation * 5.0)

                        # Apply modulation based on mode
                        if modulation_mode == 'gating_only':
                            # Only gates, no gain modulation
                            if act_idx is None:
                                h = hidden_act_fn(pre_h)
                            elif palette is not None:
                                h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                            else:
                                h = grouped_activation_forward(pre_h, act_idx, num_activations)
                            h = h * gates
                        elif modulation_mode == 'gain_bias_only':
                            # Only gain modulation, no gating
                            if act_idx is None:
                                h = hidden_act_fn(effective_gains * pre_h)
                            elif palette is not None:
                                h = grouped_activation_forward_with_palette(
                                    effective_gains * pre_h, act_idx, palette
                                )
                            else:
                                h = grouped_activation_forward(
                                    effective_gains * pre_h, act_idx, num_activations
                                )
                        else:  # 'full' mode
                            # Both gain modulation AND gating
                            if act_idx is None:
                                h = hidden_act_fn(effective_gains * pre_h)
                            elif palette is not None:
                                h = grouped_activation_forward_with_palette(
                                    effective_gains * pre_h, act_idx, palette
                                )
                            else:
                                h = grouped_activation_forward(
                                    effective_gains * pre_h, act_idx, num_activations
                                )
                            h = h * gates

                        outputs = output_act_fn(h @ W2_i)  # (n_samples, num_outputs)

                        # NT4 output inversion control
                        if use_output_inversion and num_nt >= 4:
                            invert = neurotransmitter[3]
                            outputs = invert * outputs + (1 - invert) * (1 - outputs)

                        errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
                        return jnp.maximum(0.0, 1.0 - jnp.mean(errors))

                    if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                        fitness_chunks = []
                        for chunk_start in range(0, pop_size, eval_chunk_size):
                            chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                            chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                            chunk_fitnesses = jax.vmap(eval_single_dense_true_neuromod)(
                                W1[chunk_start:chunk_end],
                                W2[chunk_start:chunk_end],
                                chunk_act,
                                receptor_densities[chunk_start:chunk_end],
                                base_gains[chunk_start:chunk_end],
                            )
                            fitness_chunks.append(chunk_fitnesses)
                        fitnesses = jnp.concatenate(fitness_chunks)
                    else:
                        fitnesses = jax.vmap(eval_single_dense_true_neuromod)(
                            W1, W2, act_indices, receptor_densities, base_gains
                        )
                else:
                    raise ValueError(f"Unknown neuromodulation mode: {neuromod_mode}")

            elif (use_dynamic_eval and (act_indices is not None or self._hidden_act_fn is not None)) or use_ih_aggregation:
                # Dynamic evaluation with optional I→H aggregation
                # Handles: dynamic activation only, aggregation only, or both combined
                use_global_agg = ih_global_agg_fn is not None
                use_per_node_agg = ih_agg_indices is not None
                use_true_agg = self.agg_use_true_aggregation
                num_agg = self.agg_num_aggregations
                fitness_mode_val = self.fitness_mode

                # Prepare agg_indices for vmap (dummy zeros if not per-node)
                if ih_agg_indices is not None:
                    ih_agg_for_vmap = ih_agg_indices
                else:
                    ih_agg_for_vmap = jnp.zeros((pop_size, total_positions), dtype=jnp.int32)

                def eval_single_dense_dynamic(W1_i, W2_i, act_idx, agg_idx):
                    # I→H with optional aggregation
                    if use_global_agg:
                        # Global aggregation: all nodes use same function
                        weighted = inputs_batch[:, :, None] * W1_i[None, :, :]
                        pre_h = ih_global_agg_fn(weighted)
                    elif use_per_node_agg and use_true_agg:
                        # True per-node aggregation
                        weighted = inputs_batch[:, :, None] * W1_i[None, :, :]
                        pre_h = grouped_aggregation_forward(weighted, agg_idx, num_agg)
                    elif use_per_node_agg:
                        # Approximated aggregation: matmul + correction factors
                        pre_h = inputs_batch @ W1_i
                        factors = compute_aggregation_correction_factors(
                            agg_idx, W1_i.shape[0], num_agg
                        )
                        pre_h = pre_h * factors
                    else:
                        # Standard matmul (sum aggregation)
                        pre_h = inputs_batch @ W1_i

                    # Apply activation
                    if act_idx is None:
                        h = hidden_act_fn(pre_h)
                    elif palette is not None:
                        h = grouped_activation_forward_with_palette(pre_h, act_idx, palette)
                    else:
                        h = grouped_activation_forward(pre_h, act_idx, num_activations)

                    outputs = output_act_fn(h @ W2_i)
                    return _eval_single_fitness(outputs, targets_batch, fitness_mode_val)

                if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                    fitness_chunks = []
                    for chunk_start in range(0, pop_size, eval_chunk_size):
                        chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                        chunk_act = act_indices[chunk_start:chunk_end] if act_indices is not None else None
                        chunk_agg = ih_agg_for_vmap[chunk_start:chunk_end]
                        chunk_fitnesses = jax.vmap(eval_single_dense_dynamic)(
                            W1[chunk_start:chunk_end],
                            W2[chunk_start:chunk_end],
                            chunk_act,
                            chunk_agg,
                        )
                        fitness_chunks.append(chunk_fitnesses)
                    fitnesses = jnp.concatenate(fitness_chunks)
                else:
                    fitnesses = jax.vmap(eval_single_dense_dynamic)(
                        W1, W2, act_indices, ih_agg_for_vmap
                    )
            else:
                # Original dense-only evaluation (disabled mode)
                # W1: (num_inputs, total_positions), W2: (total_positions, num_outputs)
                fitness_mode_val = self.fitness_mode

                def eval_single_dense(W1_i, W2_i):
                    # inputs: (n_samples, num_inputs), W1_i: (num_inputs, total_positions)
                    h = jnp.tanh(inputs_batch @ W1_i)  # (n_samples, total_positions)
                    outputs = jax.nn.sigmoid(h @ W2_i)  # (n_samples, num_outputs)
                    return _eval_single_fitness(outputs, targets_batch, fitness_mode_val)

                if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
                    # Chunked evaluation to prevent OOM
                    fitness_chunks = []
                    for chunk_start in range(0, pop_size, eval_chunk_size):
                        chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                        chunk_fitnesses = jax.vmap(eval_single_dense)(
                            W1[chunk_start:chunk_end],
                            W2[chunk_start:chunk_end],
                        )
                        fitness_chunks.append(chunk_fitnesses)
                    fitnesses = jnp.concatenate(fitness_chunks)
                else:
                    # Full population vmap (original behavior)
                    fitnesses = jax.vmap(eval_single_dense)(W1, W2)

        step_times['step5_evaluation'] = time.perf_counter() - start

        # === STEP 5b: Fitness Rescaling (optional) ===
        # When target variance is small, 1-MSE fitness compresses the useful range.
        # Convert to R² = 1 - MSE/var to spread the signal for evolutionary selection.
        if self._fitness_rescale_variance is not None:
            mse_values = 1.0 - fitnesses  # Recover MSE from 1-MSE fitness
            r2_values = 1.0 - mse_values / self._fitness_rescale_variance
            fitnesses = jnp.maximum(0.0, r2_values)

        # Cache per-individual fitness for external access (e.g., Baldwin Effect)
        self._cached_fitnesses = fitnesses

        # === STEP 6: NEAT Evolution ===
        start = time.perf_counter()
        new_state = self._compiled_tell(state, fitnesses)
        step_times['step6_neat_evolution'] = time.perf_counter() - start

        # Update generation counter
        self._current_generation += 1

        # Cache fitnesses for NAS substrate extraction
        self._last_fitnesses = fitnesses

        # === Metrics ===
        total_time = time.perf_counter() - total_start

        # Compute fitness statistics
        best_fitness = float(jnp.max(fitnesses))
        mean_fitness = float(jnp.mean(fitnesses))
        min_fitness = float(jnp.min(fitnesses))
        max_fitness = float(jnp.max(fitnesses))
        std_fitness = float(jnp.std(fitnesses))

        custom_metrics = {
            'total_positions': total_positions,
            'step_timings': step_times,
            'method': 'unified_extended',
            'recurrence_mode': self.get_recurrence_mode(),
            'multi_hop_algorithm': self.extended_config.multi_hop_algorithm if self.extended_config else 'none',
            'iteration_level': self.extended_config.iteration_level if self.extended_config else 0,
            'cache_hit': cache_hit,
        }

        if sparse_hh is not None:
            custom_metrics['total_hh_connections'] = int(jnp.sum(sparse_hh.num_valid))
            custom_metrics['avg_hh_connections'] = float(jnp.mean(sparse_hh.num_valid))

        # Compute substrate efficiency metrics
        active_counts = masks_A.sum(axis=1)  # Per-genome active position counts
        active_positions_mean = float(jnp.mean(active_counts))
        active_positions_min = int(jnp.min(active_counts))
        active_positions_max = int(jnp.max(active_counts))
        position_utilization = active_positions_mean / total_positions if total_positions > 0 else 0.0

        # Add hidden node counts to custom_metrics for benchmark reporting
        custom_metrics['avg_hidden_nodes'] = active_positions_mean
        custom_metrics['min_hidden_nodes'] = active_positions_min
        custom_metrics['max_hidden_nodes'] = active_positions_max
        custom_metrics['position_utilization'] = position_utilization

        # Compute weight matrix density
        w1_total = W1.size
        w2_total = W2.size
        w1_nonzero = int(jnp.sum(W1 != 0))
        w2_nonzero = int(jnp.sum(W2 != 0))
        w1_density = w1_nonzero / w1_total if w1_total > 0 else 0.0
        w2_density = w2_nonzero / w2_total if w2_total > 0 else 0.0

        # Track cumulative cache hits
        cache_hit_count = getattr(self, '_cache_hit_count', 0)
        if cache_hit:
            cache_hit_count += 1
        self._cache_hit_count = cache_hit_count

        # Store extended metrics with all fields
        self._extended_metrics = EMRRecurrenceMetrics(
            # Connection statistics
            total_connections=custom_metrics.get('total_hh_connections', 0),
            hidden_to_hidden_connections=custom_metrics.get('total_hh_connections', 0),
            direct_hh_connections=custom_metrics.get('total_hh_connections', 0),  # Before multi-hop

            # Performance metrics
            fitness=best_fitness,
            activate_time_used=self.activate_time,

            # Fine-grained timing breakdown (ms)
            time_cppn_ask_ms=step_times['step0_cppn_ask_transform'] * 1000,
            time_cppn_query_ms=step_times['step1_cppn_queries'] * 1000,
            time_variance_ms=step_times['step2_variance_masks'] * 1000,
            time_hh_discovery_ms=step_times['step3_phase2_discovery'] * 1000,
            time_build_matrices_ms=step_times['step4_build_matrices'] * 1000,
            time_evaluation_ms=step_times['step5_evaluation'] * 1000,
            time_neat_evolution_ms=step_times['step6_neat_evolution'] * 1000,

            # Legacy timing (kept for compatibility)
            phase2_discovery_ms=step_times['step3_phase2_discovery'] * 1000,
            forward_pass_ms=step_times['step5_evaluation'] * 1000,

            # Multi-hop
            effective_iteration_level=self.extended_config.iteration_level if self.extended_config else 0,

            # Cache status
            cache_hit=cache_hit,
            cache_hit_count=cache_hit_count,
            cache_refresh_count=self._hh_cache._refresh_count if self._hh_cache else 0,

            # Substrate efficiency
            active_positions_mean=active_positions_mean,
            active_positions_min=active_positions_min,
            active_positions_max=active_positions_max,
            total_positions=total_positions,
            position_utilization=position_utilization,

            # Weight matrix density
            w1_density=w1_density,
            w2_density=w2_density,

            # Configuration echo
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,

            # Population diversity
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            std_fitness=std_fitness,
        )

        metrics = AlgorithmMetrics(
            generation=self._current_generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            std_fitness=std_fitness,
            num_species=0,  # Not tracked in this implementation
            species_sizes=[],
            species_fitness=[],
            evaluations=pop_size,
            time_elapsed=total_time,
        )
        # Add custom metrics
        metrics.custom_metrics = custom_metrics

        return new_state, metrics

    # ========================================================================
    # Multi-Task Evolution Support
    # ========================================================================

    def _create_eval_population_true_neuromodulated(
        self,
        has_hh: bool = False,
        activate_time: int = 0,
        hidden_act_fn: Optional[Callable] = None,
    ):
        """Create vmapped true neuromodulation evaluation function.

        This creates a JIT-compiled vmapped function that evaluates an entire
        population on a single task with a specific NT vector. Used by
        run_generation_multitask() for efficient multi-task evaluation.

        Args:
            has_hh: Whether H→H connections are available
            activate_time: Number of H→H iterations (if has_hh=True)
            hidden_act_fn: Optional activation function for hidden layer.
                If None, defaults to jnp.tanh. Can be any callable from
                ACTIVATION_FUNCTIONS (e.g., jnp.sin, jax.nn.relu, etc.)

        Returns:
            A vmapped function with signature:
            - If has_hh=False:
              (W1, W2, receptor_densities, base_gains, NT_vector, inputs, targets,
               fitness_mode, modulation_mode, modulation_strength_override)
            - If has_hh=True:
              (W1, W2, receptor_densities, base_gains, NT_vector, inputs, targets,
               from_indices, to_indices, hh_weights, valid_mask,
               fitness_mode, modulation_mode, modulation_strength_override)
            -> fitness_array of shape (pop_size,)
        """
        # Capture neuromodulation config in closure
        mod_strength = self.neuromod_config.modulation_strength if self.neuromod_config else 5.0
        # Use provided activation or default to tanh
        act_fn = hidden_act_fn if hidden_act_fn is not None else jnp.tanh

        if has_hh and activate_time > 0:
            # H→H enabled version with iteration loop
            def eval_single_network_true_neuromodulated_hh(
                W1_single, W2_single, receptor_densities_single, base_gains_single,
                neurotransmitter, inputs, targets,
                from_idx, to_idx, hh_w, hh_mask,
                fitness_mode='mse', modulation_mode='full', modulation_strength_override=None
            ):
                """Evaluate single network with TRUE neuromodulation AND H→H connections.

                This extends the feedforward version to include H→H iterations:
                - W1, W2, receptor_densities, base_gains are FIXED
                - H→H sparse connections (from_idx, to_idx, hh_w, hh_mask) are FIXED
                - ONLY neurotransmitter vector varies per task
                - After initial I→H activation, applies H→H iterations with neuromodulation

                Args:
                    W1_single: Input→Hidden weights, shape (num_inputs, total_positions)
                    W2_single: Hidden→Output weights, shape (total_positions, num_outputs)
                    receptor_densities_single: Per-neuron NT sensitivity, shape (total_positions, num_nt)
                    base_gains_single: Base activation gain per neuron, shape (total_positions,)
                    neurotransmitter: NT levels, shape (num_nt,) - ONLY this changes per task!
                    inputs: Input data, shape (num_cases, num_inputs)
                    targets: Target data, shape (num_cases, num_outputs)
                    from_idx: Source indices for H→H connections, shape (max_conns,)
                    to_idx: Target indices for H→H connections, shape (max_conns,)
                    hh_w: H→H connection weights, shape (max_conns,)
                    hh_mask: Valid connection mask, shape (max_conns,)
                    fitness_mode: Fitness function ('mse', 'accuracy', etc.)
                    modulation_mode: 'full', 'gating_only', 'gain_bias_only'
                    modulation_strength_override: Override default modulation strength

                Returns:
                    Fitness score for this network on this task
                """
                # Use override if provided, otherwise use default
                strength = modulation_strength_override if modulation_strength_override is not None else mod_strength

                # Compute per-neuron modulation from FIXED receptors and VARIABLE NT
                modulation = receptor_densities_single[:, :3] @ neurotransmitter[:3]

                # Initial I→H forward pass
                pre_hidden = inputs @ W1_single  # (num_cases, total_positions)

                if modulation_mode == 'gating_only':
                    gates = jax.nn.sigmoid(modulation * strength)
                    hidden = act_fn(pre_hidden) * gates
                elif modulation_mode == 'gain_bias_only':
                    effective_gains = base_gains_single + strength * modulation
                    effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                    effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                    modulation_bias = modulation * strength
                    hidden = act_fn(effective_gains * pre_hidden + modulation_bias)
                    gates = None  # Not used in gain_bias_only mode
                else:  # 'full' mode
                    effective_gains = base_gains_single + strength * modulation
                    effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                    effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                    gates = jax.nn.sigmoid(modulation)
                    modulation_bias = modulation * strength
                    hidden = act_fn(effective_gains * pre_hidden + modulation_bias) * gates

                # H→H iterations with neuromodulation
                num_hidden = hidden.shape[1]
                safe_from = jnp.clip(from_idx, 0, num_hidden - 1)
                safe_to = jnp.clip(to_idx, 0, num_hidden - 1)
                effective_hh_w = jnp.where(hh_mask, hh_w, 0.0)

                # Static unroll for JIT compatibility (activate_time is compile-time constant)
                for _ in range(activate_time):
                    # Gather from source positions
                    source_vals = hidden[:, safe_from]  # (num_cases, max_conns)
                    contributions = source_vals * effective_hh_w  # (num_cases, max_conns)

                    # Scatter-add to target positions
                    hh_contrib = jnp.zeros_like(hidden)
                    hh_contrib = hh_contrib.at[:, safe_to].add(contributions)

                    # Apply neuromodulation to H→H contribution
                    if modulation_mode == 'gating_only':
                        hh_update = act_fn(hh_contrib) * gates
                    elif modulation_mode == 'gain_bias_only':
                        hh_update = act_fn(effective_gains * hh_contrib + modulation_bias)
                    else:  # 'full'
                        hh_update = act_fn(effective_gains * hh_contrib + modulation_bias) * gates

                    hidden = hidden + hh_update

                # Output layer
                outputs = jax.nn.sigmoid(hidden @ W2_single)

                # Output inversion using NT[3]
                invert_signal = neurotransmitter[3]
                invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
                inverted = 1.0 - outputs
                outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

                return compute_fitness(outputs, targets, fitness_mode)

            # Vmap over population (axis 0 for all per-genome arrays)
            eval_population = jax.vmap(
                eval_single_network_true_neuromodulated_hh,
                in_axes=(0, 0, 0, 0, None, None, None, 0, 0, 0, 0, None, None, None)
            )

        else:
            # Original feedforward-only version
            def eval_single_network_true_neuromodulated(
                W1_single, W2_single, receptor_densities_single, base_gains_single,
                neurotransmitter, inputs, targets, fitness_mode='mse',
                modulation_mode='full', modulation_strength_override=None
            ):
                """Evaluate single network with TRUE neuromodulation (feedforward only).

                This is the KEY function for multi-task neuromodulation:
                - W1, W2, receptor_densities, base_gains are FIXED (from run_generation_verbose)
                - ONLY neurotransmitter vector varies per task
                - Same network produces different behaviors via NT-modulated activation

                Args:
                    W1_single: Input→Hidden weights, shape (num_inputs, total_positions)
                    W2_single: Hidden→Output weights, shape (total_positions, num_outputs)
                    receptor_densities_single: Per-neuron NT sensitivity, shape (total_positions, num_nt)
                    base_gains_single: Base activation gain per neuron, shape (total_positions,)
                    neurotransmitter: NT levels, shape (num_nt,) - ONLY this changes per task!
                    inputs: Input data, shape (num_cases, num_inputs)
                    targets: Target data, shape (num_cases, num_outputs)
                    fitness_mode: Fitness function ('mse', 'accuracy', etc.)
                    modulation_mode: 'full', 'gating_only', 'gain_bias_only'
                    modulation_strength_override: Override default modulation strength

                Returns:
                    Fitness score for this network on this task
                """
                # Use override if provided, otherwise use default
                strength = modulation_strength_override if modulation_strength_override is not None else mod_strength

                # Compute per-neuron modulation from FIXED receptors and VARIABLE NT
                # Only use first 3 NT components for modulation (NT[3] is for output inversion)
                modulation = receptor_densities_single[:, :3] @ neurotransmitter[:3]

                # Forward pass
                pre_hidden = inputs @ W1_single  # (num_cases, total_positions)

                if modulation_mode == 'gating_only':
                    # GATING-ONLY: Standard activation, only gating varies by task
                    gates = jax.nn.sigmoid(modulation * strength)
                    hidden = act_fn(pre_hidden) * gates
                elif modulation_mode == 'gain_bias_only':
                    # GAIN+BIAS ONLY: No gating, full gain and bias modulation
                    effective_gains = base_gains_single + strength * modulation
                    effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                    effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                    modulation_bias = modulation * strength
                    hidden = act_fn(effective_gains * pre_hidden + modulation_bias)
                else:  # 'full' mode (default)
                    # FULL MODULATION: Gain + Bias + Gating
                    effective_gains = base_gains_single + strength * modulation
                    effective_gains = jnp.clip(effective_gains, 0.1, 5.0)
                    effective_gains = jnp.where(jnp.isnan(effective_gains), 1.0, effective_gains)
                    gates = jax.nn.sigmoid(modulation)
                    modulation_bias = modulation * strength
                    hidden = act_fn(effective_gains * pre_hidden + modulation_bias) * gates

                # Output layer
                outputs = jax.nn.sigmoid(hidden @ W2_single)

                # Output inversion using NT[3] (for complementary tasks like NAND, NOR)
                invert_signal = neurotransmitter[3]
                invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
                inverted = 1.0 - outputs
                outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

                # Compute fitness
                return compute_fitness(outputs, targets, fitness_mode)

            # Vmap over population dimension (axis 0 for W1, W2, receptor_densities, base_gains)
            eval_population = jax.vmap(
                eval_single_network_true_neuromodulated,
                in_axes=(0, 0, 0, 0, None, None, None, None, None, None)
            )

        return eval_population

    def _create_eval_population_two_module(
        self,
        all_positions: jnp.ndarray,
        hidden_act_fn: Optional[Callable] = None,
    ):
        """Create vmapped two-module evaluation function (PFC/Sensorimotor separation).

        This creates a JIT-compiled vmapped function that evaluates networks using
        the two-module architecture from Liu & Wang (2024):
        - PFC module (x < 0): Rule maintenance, uses NT[0:2]
        - Sensorimotor module (x >= 0): Task execution, uses NT[1:3]

        Args:
            all_positions: Spatial coordinates of all positions, shape (total_positions, 2)
            hidden_act_fn: Optional activation function for hidden layer.

        Returns:
            A vmapped function with signature:
            (W1, W2, receptor_densities, base_gains, NT_vector, inputs, targets,
             fitness_mode, modulation_mode, modulation_strength_override, two_module_mode)
            -> fitness_array of shape (pop_size,)
        """
        # Capture neuromodulation config in closure
        mod_strength = self.neuromod_config.modulation_strength if self.neuromod_config else 5.0
        # Use provided activation or default to tanh
        act_fn = hidden_act_fn if hidden_act_fn is not None else jnp.tanh

        # Precompute spatial masks (these are constant for all evaluations)
        x_coords = all_positions[:, 0]
        pfc_mask = x_coords < 0.0  # Left hemisphere = PFC (rule)
        sm_mask = x_coords >= 0.0   # Right hemisphere = Sensorimotor (execution)

        def eval_single_network_two_module(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            neurotransmitter, inputs, targets,
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
                W1_single: Input→Hidden weights, shape (num_inputs, total_positions)
                W2_single: Hidden→Output weights, shape (total_positions, num_outputs)
                receptor_densities_single: Per-neuron NT sensitivity, shape (total_positions, num_nt)
                base_gains_single: Base activation gain per neuron, shape (total_positions,)
                neurotransmitter: NT levels, shape (num_nt,)
                inputs: Input data, shape (num_cases, num_inputs)
                targets: Target data, shape (num_cases, num_outputs)
                fitness_mode: Fitness function ('mse', 'accuracy', etc.)
                modulation_mode: 'full', 'gating_only', 'gain_bias_only'
                modulation_strength_override: Override default modulation strength
                two_module_mode: 'parallel' or 'sequential'

            Returns:
                Fitness score based on fitness_mode
            """
            strength = modulation_strength_override if modulation_strength_override is not None else mod_strength

            # === PFC MODULE (Rule Maintenance) ===
            # Uses NT[0:2] for modulation - distinct rule encoding
            W1_pfc = W1_single[:, pfc_mask]
            W2_pfc = W2_single[pfc_mask, :]
            rd_pfc = receptor_densities_single[pfc_mask, :]
            bg_pfc = base_gains_single[pfc_mask]

            pre_hidden_pfc = inputs @ W1_pfc  # (num_cases, n_pfc)
            mod_pfc = rd_pfc[:, :2] @ neurotransmitter[:2]  # NT[0:2] for PFC

            if modulation_mode == 'gating_only':
                gates_pfc = jax.nn.sigmoid(mod_pfc * strength)
                hidden_pfc = act_fn(pre_hidden_pfc) * gates_pfc
            elif modulation_mode == 'gain_bias_only':
                effective_gains_pfc = bg_pfc + strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                modulation_bias_pfc = mod_pfc * strength
                hidden_pfc = act_fn(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
            else:  # 'full' mode
                effective_gains_pfc = bg_pfc + strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                gates_pfc = jax.nn.sigmoid(mod_pfc)
                modulation_bias_pfc = mod_pfc * strength
                hidden_pfc = act_fn(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
                hidden_pfc = hidden_pfc * gates_pfc

            # === SENSORIMOTOR MODULE (Task Execution) ===
            # Uses NT[1:3] for modulation - overlapping NT component enables coordination
            W1_sm = W1_single[:, sm_mask]
            W2_sm = W2_single[sm_mask, :]
            rd_sm = receptor_densities_single[sm_mask, :]
            bg_sm = base_gains_single[sm_mask]

            pre_hidden_sm = inputs @ W1_sm  # (num_cases, n_sm)

            # Sequential mode: Add top-down PFC signal to sensorimotor input
            if two_module_mode == 'sequential':
                # Mean PFC activation provides top-down rule signal
                pfc_mean = jnp.mean(hidden_pfc, axis=-1, keepdims=True)  # (num_cases, 1)
                pre_hidden_sm = pre_hidden_sm + pfc_mean  # Broadcast to all SM positions

            mod_sm = rd_sm[:, 1:3] @ neurotransmitter[1:3]  # NT[1:3] for SM

            if modulation_mode == 'gating_only':
                gates_sm = jax.nn.sigmoid(mod_sm * strength)
                hidden_sm = act_fn(pre_hidden_sm) * gates_sm
            elif modulation_mode == 'gain_bias_only':
                effective_gains_sm = bg_sm + strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                modulation_bias_sm = mod_sm * strength
                hidden_sm = act_fn(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
            else:  # 'full' mode
                effective_gains_sm = bg_sm + strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                gates_sm = jax.nn.sigmoid(mod_sm)
                modulation_bias_sm = mod_sm * strength
                hidden_sm = act_fn(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
                hidden_sm = hidden_sm * gates_sm

            # === COMBINE OUTPUTS ===
            output_pfc = hidden_pfc @ W2_pfc  # (num_cases, num_outputs)
            output_sm = hidden_sm @ W2_sm    # (num_cases, num_outputs)
            outputs = jax.nn.sigmoid(output_pfc + output_sm)

            # Output inversion using NT[3]
            invert_signal = neurotransmitter[3]
            invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
            inverted = 1.0 - outputs
            outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

            return compute_fitness(outputs, targets, fitness_mode)

        def eval_single_network_two_module_with_hidden(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            neurotransmitter, inputs, targets,
            fitness_mode='mse', modulation_mode='full', modulation_strength_override=None,
            two_module_mode='parallel'
        ):
            """Two-module evaluation that also returns hidden activations."""
            strength = modulation_strength_override if modulation_strength_override is not None else mod_strength

            # === PFC MODULE ===
            W1_pfc = W1_single[:, pfc_mask]
            rd_pfc = receptor_densities_single[pfc_mask, :]
            bg_pfc = base_gains_single[pfc_mask]
            W2_pfc = W2_single[pfc_mask, :]

            pre_hidden_pfc = inputs @ W1_pfc
            mod_pfc = rd_pfc[:, :2] @ neurotransmitter[:2]

            if modulation_mode == 'gating_only':
                gates_pfc = jax.nn.sigmoid(mod_pfc * strength)
                hidden_pfc = act_fn(pre_hidden_pfc) * gates_pfc
            elif modulation_mode == 'gain_bias_only':
                effective_gains_pfc = bg_pfc + strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                modulation_bias_pfc = mod_pfc * strength
                hidden_pfc = act_fn(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
            else:
                effective_gains_pfc = bg_pfc + strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                gates_pfc = jax.nn.sigmoid(mod_pfc)
                modulation_bias_pfc = mod_pfc * strength
                hidden_pfc = act_fn(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
                hidden_pfc = hidden_pfc * gates_pfc

            # === SENSORIMOTOR MODULE ===
            W1_sm = W1_single[:, sm_mask]
            rd_sm = receptor_densities_single[sm_mask, :]
            bg_sm = base_gains_single[sm_mask]
            W2_sm = W2_single[sm_mask, :]

            pre_hidden_sm = inputs @ W1_sm
            if two_module_mode == 'sequential':
                pfc_mean = jnp.mean(hidden_pfc, axis=-1, keepdims=True)
                pre_hidden_sm = pre_hidden_sm + pfc_mean

            mod_sm = rd_sm[:, 1:3] @ neurotransmitter[1:3]

            if modulation_mode == 'gating_only':
                gates_sm = jax.nn.sigmoid(mod_sm * strength)
                hidden_sm = act_fn(pre_hidden_sm) * gates_sm
            elif modulation_mode == 'gain_bias_only':
                effective_gains_sm = bg_sm + strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                modulation_bias_sm = mod_sm * strength
                hidden_sm = act_fn(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
            else:
                effective_gains_sm = bg_sm + strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                gates_sm = jax.nn.sigmoid(mod_sm)
                modulation_bias_sm = mod_sm * strength
                hidden_sm = act_fn(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
                hidden_sm = hidden_sm * gates_sm

            # Combine outputs
            output_pfc = hidden_pfc @ W2_pfc
            output_sm = hidden_sm @ W2_sm
            outputs = jax.nn.sigmoid(output_pfc + output_sm)

            invert_signal = neurotransmitter[3]
            invert_weight = jnp.clip(invert_signal, 0.0, 1.0)
            inverted = 1.0 - outputs
            outputs = invert_weight * outputs + (1.0 - invert_weight) * inverted

            fitness = compute_fitness(outputs, targets, fitness_mode)

            # Combine hidden for orthogonality computation - pad to total_positions
            total_positions = all_positions.shape[0]
            num_cases = inputs.shape[0]
            combined_hidden = jnp.zeros((num_cases, total_positions))
            combined_hidden = combined_hidden.at[:, pfc_mask].set(hidden_pfc)
            combined_hidden = combined_hidden.at[:, sm_mask].set(hidden_sm)

            return fitness, combined_hidden

        # Create vmapped versions
        eval_population_two_module = jax.vmap(
            eval_single_network_two_module,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None)
        )

        eval_population_two_module_with_hidden = jax.vmap(
            eval_single_network_two_module_with_hidden,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None)
        )

        return eval_population_two_module, eval_population_two_module_with_hidden

    def _build_matrices_streaming(
        self,
        state: Any,
        cppns_transformed: Tuple,
        h_grid: 'HierarchicalGridStructure',
        step_times: Dict[str, float],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Build W1, W2, masks_A level-by-level for memory efficiency.

        This method processes one hierarchical level at a time instead of
        querying all positions at once, reducing peak memory from
        O(total_positions) to O(max_level_size).

        Memory reduction example (depth 7):
            Standard: 87K positions × pop × sources → 10-20 GB on CPU
            Streaming: 65K positions max (largest level) → 2-3 GB on CPU

        Trade-off: 3-10x slower per generation due to Python loop overhead.

        Args:
            state: Algorithm state
            cppns_transformed: Transformed CPPN parameters
            h_grid: Pre-computed hierarchical grid structure
            step_times: Dict to record timing info

        Returns:
            Tuple of (W1, W2, masks_A):
                W1: (pop, num_inputs, total_positions) - input→hidden weights
                W2: (pop, total_positions, num_outputs) - hidden→output weights
                masks_A: (pop, total_positions) - active position masks
        """
        t_stream_start = time.perf_counter()

        pop_size = cppns_transformed[0].shape[0]
        num_levels = h_grid.num_levels
        total_positions = h_grid.total_positions

        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        max_weight = self.max_weight
        weight_thresh = 0.1
        pop_chunk_size = self.population_chunk_size if self.population_chunk_size > 0 else pop_size

        # Pre-allocate weight matrix accumulators
        W1_accumulated = jnp.zeros((pop_size, num_inputs, total_positions), dtype=jnp.float32)
        W2_accumulated = jnp.zeros((pop_size, num_outputs, total_positions), dtype=jnp.float32)

        # Track level variances and reached masks for subdivision
        level_variances = []
        level_reached = []
        level_reached.append(jnp.ones((pop_size, 4), dtype=bool))  # Level 0 always reached

        # Cache for next-level weights (for variance lookahead)
        cached_next_input_weights = None

        if self.verbose:
            print(f"[Streaming] Processing {num_levels} levels, {total_positions} total positions")

        for level in range(num_levels):
            level_start = h_grid.level_offsets_static[level]
            level_end = h_grid.level_offsets_static[level + 1]
            num_level_positions = level_end - level_start
            level_positions = h_grid.level_positions[level]

            t_level = time.perf_counter()

            # -----------------------------------------------------------------
            # Get reached mask for THIS level FIRST (before querying)
            # -----------------------------------------------------------------
            if level == 0:
                current_reached_mask = level_reached[0]
            else:
                # Compute reached from previous level's subdivision
                prev_reached = level_reached[level - 1]
                prev_variance = level_variances[level - 1]
                prev_subdivided = prev_reached & (prev_variance > self.variance_threshold)

                parent_indices = h_grid.parent_indices[level]
                current_reached_mask = prev_subdivided[:, parent_indices]
                level_reached.append(current_reached_mask)

            # -----------------------------------------------------------------
            # Query CPPN for this level's positions
            # -----------------------------------------------------------------

            # Input weights for this level
            if cached_next_input_weights is not None and level > 0:
                level_input_weights = cached_next_input_weights
            else:
                level_input_weights = query_level_positions(
                    state, cppns_transformed, input_coords, level_positions,
                    True, self._jitted_cppn_forward, pop_chunk_size,
                    device_id=0,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )

            # Output weights for this level (always query fresh)
            level_output_weights = query_level_positions(
                state, cppns_transformed, output_coords, level_positions,
                False, self._jitted_cppn_forward, pop_chunk_size,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )

            # -----------------------------------------------------------------
            # Compute variance for this level (lookahead to next level)
            # -----------------------------------------------------------------

            if level < num_levels - 1:
                # Query next level for variance computation
                next_level_positions = h_grid.level_positions[level + 1]
                cached_next_input_weights = query_level_positions(
                    state, cppns_transformed, input_coords, next_level_positions,
                    True, self._jitted_cppn_forward, pop_chunk_size,
                    device_id=0,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )

                # Compute variance from children
                variance_weights = cached_next_input_weights[:, 0, :]
                level_variance = compute_single_level_variance(
                    variance_weights,
                    h_grid,
                    level,
                )
                level_variances.append(level_variance)
            else:
                # Finest level - no children, no variance
                level_variances.append(jnp.zeros((pop_size, num_level_positions)))
                cached_next_input_weights = None

            # -----------------------------------------------------------------
            # Apply weight processing and store in accumulators
            # -----------------------------------------------------------------

            active_mask_broadcast = current_reached_mask[:, None, :]

            # Standard mode: simple weight thresholding
            # level_input_weights shape: (pop, num_sources, num_positions)
            W1_raw = jnp.tanh(level_input_weights) * max_weight

            # Apply locality penalty for geometry seeding (if configured)
            if self.locality_radius is not None:
                W1_raw = apply_locality_penalty_to_weights(
                    W1_raw, input_coords, level_positions, self.locality_radius
                )

            W1_level = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )

            # Process output weights
            W2_raw = jnp.tanh(level_output_weights) * max_weight

            # Apply locality penalty for W2: level_positions → output_coords
            if self.locality_radius is not None:
                # W2_raw is (pop, num_outputs, level_positions), need to transpose for penalty
                W2_for_penalty = W2_raw.transpose(0, 2, 1)  # (pop, level_positions, num_outputs)
                W2_for_penalty = apply_locality_penalty_to_weights(
                    W2_for_penalty, level_positions, output_coords, self.locality_radius
                )
                W2_raw = W2_for_penalty.transpose(0, 2, 1)  # Back to (pop, num_outputs, level_positions)

            W2_level = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )

            # Store in accumulators (slice update)
            W1_accumulated = W1_accumulated.at[:, :, level_start:level_end].set(W1_level)
            W2_accumulated = W2_accumulated.at[:, :, level_start:level_end].set(W2_level)

            # Explicit memory cleanup
            del level_input_weights, level_output_weights, W1_raw, W2_raw
            del W1_level, W2_level

            if self._streaming_verbose:
                level_time = time.perf_counter() - t_level
                print(f"  Level {level}: {num_level_positions} positions, {level_time*1000:.1f}ms")

        # Build full masks_A from level_reached
        masks_A = jnp.concatenate(level_reached, axis=1)

        # Final weight matrices
        W1 = W1_accumulated
        W2 = W2_accumulated.transpose(0, 2, 1)  # (pop, num_outputs, total_positions) → (pop, total_positions, num_outputs)

        step_times['streaming_build'] = time.perf_counter() - t_stream_start

        if self.verbose:
            print(f"[Streaming] Complete: {step_times['streaming_build']*1000:.1f}ms total")

        return W1, W2, masks_A

    def run_generation_multitask(
        self,
        state: Any,
        problems: List[Any],
        neurotransmitters: Optional[List[jnp.ndarray]] = None,
        aggregation_method: str = 'mean',
        aggregation_weights: Optional[List[float]] = None,
        orthogonality_bonus: float = 0.0,
        specialization_bonus: float = 0.0,
    ) -> Tuple[Any, 'AlgorithmMetrics']:
        """Run one generation of multi-task evolution with TRUE neuromodulation.

        This is the KEY method for proper multi-task evolution where:
        1. CPPN evolution happens ONCE per generation
        2. W1, W2, receptor_densities, base_gains are built ONCE (via run_generation_verbose)
        3. Fitness is computed on ALL tasks with task-specific NT vectors
        4. AGGREGATED multi-task fitness is used for NEAT selection
        5. Same network architecture solves multiple tasks

        The approach mirrors the working implementation from
        emrhyperneat_neuromodulation_functions.py.

        Args:
            state: Current algorithm state
            problems: List of problems (tasks) to evaluate on
            neurotransmitters: Optional list of NT vectors for each task.
                If None, uses default XOR, AND, OR, NAND, NOR presets.
            aggregation_method: How to combine per-task fitnesses:
                - 'mean': Average fitness across tasks
                - 'min': Minimum fitness (hardest task matters most)
                - 'weighted': Weighted average using aggregation_weights
                - 'product': Geometric mean (product^(1/n))
                - 'softmin': Soft minimum (smoother than min)
                - 'harmonic': Harmonic mean (emphasizes poor performance)
            aggregation_weights: Weights for 'weighted' method (must sum to 1)
            orthogonality_bonus: Bonus for having orthogonal NT vectors
            specialization_bonus: Bonus for having distinct activation patterns per task

        Returns:
            Tuple of (new_state, metrics) with multi-task metrics

        Raises:
            ValueError: If neuromodulation is disabled or mode not true_neuromodulation
        """
        import time as time_module

        total_start = time_module.perf_counter()

        # Validate neuromodulation is enabled
        if self.neuromod_config is None or self.neuromod_config.mode == 'disabled':
            raise ValueError(
                "run_generation_multitask() requires neuromodulation to be enabled. "
                "Set 'neuromodulation': {'mode': 'true_neuromodulation', ...} in config."
            )

        if self.neuromod_config.mode != 'true_neuromodulation':
            raise ValueError(
                f"run_generation_multitask() requires 'true_neuromodulation' mode, "
                f"got '{self.neuromod_config.mode}'. Use run_generation_verbose() "
                f"for other neuromodulation modes."
            )

        num_tasks = len(problems)

        # Setup NT vectors
        if neurotransmitters is None:
            # Use default task presets
            task_names = ['xor', 'and', 'or', 'nand', 'nor']
            neurotransmitters = [
                NT_TASK_PRESETS.get(task_names[i % len(task_names)], NT_TASK_PRESETS['xor'])
                for i in range(num_tasks)
            ]
        elif len(neurotransmitters) != num_tasks:
            raise ValueError(
                f"Number of NT vectors ({len(neurotransmitters)}) must match "
                f"number of problems ({num_tasks})"
            )

        # Validate aggregation weights
        if aggregation_method == 'weighted':
            if aggregation_weights is None:
                aggregation_weights = [1.0 / num_tasks] * num_tasks
            elif len(aggregation_weights) != num_tasks:
                raise ValueError(
                    f"Number of weights ({len(aggregation_weights)}) must match "
                    f"number of problems ({num_tasks})"
                )
            # Normalize weights
            total_weight = sum(aggregation_weights)
            aggregation_weights = [w / total_weight for w in aggregation_weights]

        # Store current multi-task state
        self._multitask_state['num_tasks'] = num_tasks
        self._multitask_state['task_names'] = [
            getattr(p, 'name', f'task_{i}') for i, p in enumerate(problems)
        ]

        # ====================================================================
        # STEP 1: Call run_generation_verbose ONCE to build W1, W2 and cache them
        # ====================================================================
        # Use first problem to build weight matrices
        # This caches: self._cached_W1, self._cached_W2, self._neuromod_true
        first_problem = problems[0]
        _, _ = self.run_generation_verbose(state, first_problem, skip_metrics=True)

        # ====================================================================
        # STEP 2: Extract cached data
        # ====================================================================
        if not hasattr(self, '_cached_W1') or self._cached_W1 is None:
            raise RuntimeError(
                "Weight matrices not cached. Ensure run_generation_verbose() was called "
                "with TRUE neuromodulation enabled."
            )
        if not hasattr(self, '_neuromod_true') or self._neuromod_true is None:
            raise RuntimeError(
                "TRUE neuromodulation data not available. Check configuration."
            )

        W1 = self._cached_W1
        W2 = self._cached_W2
        rd = self._neuromod_true['receptor_densities']
        bg = self._neuromod_true['base_gains']

        # ====================================================================
        # STEP 3: Check for H→H connections and setup activation functions
        # ====================================================================
        # Check for cached sparse H→H connections from run_generation_verbose
        sparse_hh = getattr(self, '_cached_sparse_hh', None)
        has_hh = (
            sparse_hh is not None and
            self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH
        )
        activate_time = self.activate_time if has_hh else 0

        # Determine per-task activation functions from config
        # Priority: per_task_activation > hidden_activation > 'tanh' (default)
        multitask_cfg = getattr(self, 'multitask_config', None)
        per_task_act_names = []  # List of activation names per task
        task_names = self._multitask_state.get('task_names', [f'task_{i}' for i in range(num_tasks)])

        for task_idx, task_name in enumerate(task_names):
            if multitask_cfg is not None and multitask_cfg.per_task_activation:
                # Per-task activation specified
                act_name = multitask_cfg.per_task_activation.get(task_name, 'tanh')
            elif multitask_cfg is not None and multitask_cfg.hidden_activation:
                # Global activation specified
                act_name = multitask_cfg.hidden_activation
            else:
                # Default to tanh for backwards compatibility
                act_name = 'tanh'
            per_task_act_names.append(act_name)

        # ==== Extract extended config values (ported from neuromodulation_functions.py) ====
        # These use safe defaults to preserve existing behavior
        cfg_fitness_mode = 'mse'
        cfg_modulation_mode = 'full'
        cfg_modulation_strength_override = None
        cfg_two_module_mode = 'none'
        cfg_orthogonality_bonus_weight = orthogonality_bonus  # Use function arg as default
        cfg_orthogonality_metric = 'cosine_mean'
        cfg_generalist_bonus_type = 'none'
        cfg_generalist_bonus_weight = 0.0
        cfg_generalist_threshold = 0.9
        cfg_softmin_temperature = 0.1

        if multitask_cfg is not None:
            cfg_fitness_mode = getattr(multitask_cfg, 'fitness_mode', 'mse')
            cfg_modulation_mode = getattr(multitask_cfg, 'modulation_mode', 'full')
            cfg_modulation_strength_override = getattr(multitask_cfg, 'modulation_strength_override', None)
            cfg_two_module_mode = getattr(multitask_cfg, 'two_module_mode', 'none')
            # Use config orthogonality_bonus_weight if set, otherwise use function arg
            if hasattr(multitask_cfg, 'orthogonality_bonus_weight') and multitask_cfg.orthogonality_bonus_weight > 0:
                cfg_orthogonality_bonus_weight = multitask_cfg.orthogonality_bonus_weight
            cfg_orthogonality_metric = getattr(multitask_cfg, 'orthogonality_metric', 'cosine_mean')
            cfg_generalist_bonus_type = getattr(multitask_cfg, 'generalist_bonus_type', 'none')
            cfg_generalist_bonus_weight = getattr(multitask_cfg, 'generalist_bonus_weight', 0.0)
            cfg_generalist_threshold = getattr(multitask_cfg, 'generalist_threshold', 0.9)
            cfg_softmin_temperature = getattr(multitask_cfg, 'softmin_temperature', 0.1)

        # Create eval functions for unique activation functions (cache to avoid redundant JIT)
        unique_activations = set(per_task_act_names)
        eval_fn_cache = {}
        eval_fn_cache_two_module = {}  # For two-module mode

        # Get all_positions for two-module evaluation (if needed)
        all_positions = None
        if cfg_two_module_mode != 'none':
            h_grid = getattr(self, '_cached_h_grid', None)
            if h_grid is not None:
                all_positions = h_grid.all_positions
            else:
                # Fall back to standard mode if positions not available
                cfg_two_module_mode = 'none'
                if self.verbose:
                    print("[MultiTask] Warning: two_module_mode requires cached positions, falling back to 'none'")

        for act_name in unique_activations:
            act_fn = ACTIVATION_FUNCTIONS.get(act_name, jnp.tanh)
            eval_fn_cache[act_name] = self._create_eval_population_true_neuromodulated(
                has_hh=has_hh,
                activate_time=activate_time,
                hidden_act_fn=act_fn,
            )
            # Also create two-module eval function if needed
            if cfg_two_module_mode != 'none' and all_positions is not None:
                eval_fn_cache_two_module[act_name] = self._create_eval_population_two_module(
                    all_positions=all_positions,
                    hidden_act_fn=act_fn,
                )

        if self.verbose and len(unique_activations) > 1:
            print(f"[MultiTask] Using {len(unique_activations)} unique activation functions: {unique_activations}")
        if self.verbose and cfg_two_module_mode != 'none':
            print(f"[MultiTask] Using two-module architecture mode: {cfg_two_module_mode}")

        # ====================================================================
        # STEP 4: Evaluate on each task with different NT vectors
        # ====================================================================
        per_task_fitnesses = []
        per_task_times = []

        for task_idx, (problem, nt_vector) in enumerate(zip(problems, neurotransmitters)):
            task_start = time_module.perf_counter()

            # Get inputs and targets for this task
            if hasattr(problem, 'get_inputs'):
                inputs = jnp.array(problem.get_inputs())
                targets = jnp.array(problem.get_targets())
            elif hasattr(problem, 'inputs'):
                inputs = jnp.array(problem.inputs)
                targets = jnp.array(problem.targets)
            else:
                # Try get_data() method (returns list of (input, target) tuples)
                data = problem.get_data()
                inputs = jnp.array([d[0] for d in data])
                targets = jnp.array([d[1] for d in data])

            # Ensure NT vector is a JAX array
            nt_vector = jnp.array(nt_vector)

            # Get the appropriate eval function for this task's activation
            task_act_name = per_task_act_names[task_idx]

            # Use two-module evaluation if configured
            if cfg_two_module_mode != 'none' and task_act_name in eval_fn_cache_two_module:
                eval_fn_pair = eval_fn_cache_two_module[task_act_name]
                eval_population_two_mod = eval_fn_pair[0]  # Without hidden return
                fitness = eval_population_two_mod(
                    W1, W2, rd, bg, nt_vector, inputs, targets,
                    cfg_fitness_mode, cfg_modulation_mode, cfg_modulation_strength_override,
                    cfg_two_module_mode
                )
            elif has_hh and sparse_hh is not None:
                # Pass H→H connection data for recurrent evaluation
                eval_population = eval_fn_cache[task_act_name]
                fitness = eval_population(
                    W1, W2, rd, bg, nt_vector, inputs, targets,
                    sparse_hh.from_indices,  # (pop_size, max_conns)
                    sparse_hh.to_indices,
                    sparse_hh.weights,
                    sparse_hh.valid_mask,
                    cfg_fitness_mode, cfg_modulation_mode, cfg_modulation_strength_override
                )
            else:
                # Feedforward-only evaluation (no H→H)
                eval_population = eval_fn_cache[task_act_name]
                fitness = eval_population(
                    W1, W2, rd, bg, nt_vector, inputs, targets,
                    cfg_fitness_mode, cfg_modulation_mode, cfg_modulation_strength_override
                )
            per_task_fitnesses.append(fitness)
            per_task_times.append(time_module.perf_counter() - task_start)

        # ====================================================================
        # STEP 5: Aggregate fitnesses
        # ====================================================================
        stacked_fitnesses = jnp.stack(per_task_fitnesses, axis=0)  # (num_tasks, pop_size)

        if aggregation_method == 'mean':
            aggregated_fitness = jnp.mean(stacked_fitnesses, axis=0)
        elif aggregation_method == 'min':
            aggregated_fitness = jnp.min(stacked_fitnesses, axis=0)
        elif aggregation_method == 'weighted':
            weights = jnp.array(aggregation_weights)
            aggregated_fitness = jnp.sum(stacked_fitnesses * weights[:, None], axis=0)
        elif aggregation_method == 'product':
            # Geometric mean: (f1 * f2 * ... * fn)^(1/n)
            eps = 1e-8
            log_fitness = jnp.sum(jnp.log(stacked_fitnesses + eps), axis=0)
            aggregated_fitness = jnp.exp(log_fitness / num_tasks)
        elif aggregation_method == 'softmin':
            # Softmin: weighted average where smaller values get more weight
            # Use config temperature (lower = sharper, more like true min)
            neg_scaled = -stacked_fitnesses / cfg_softmin_temperature
            weights = jax.nn.softmax(neg_scaled, axis=0)
            aggregated_fitness = jnp.sum(stacked_fitnesses * weights, axis=0)
        elif aggregation_method == 'harmonic':
            # Harmonic mean: n / (1/f1 + 1/f2 + ... + 1/fn)
            eps = 1e-8
            reciprocal_sum = jnp.sum(1.0 / (stacked_fitnesses + eps), axis=0)
            aggregated_fitness = num_tasks / reciprocal_sum
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Apply orthogonality bonus (reward distinct NT vectors)
        # Use config value if set, otherwise use function argument
        effective_orthogonality_bonus = cfg_orthogonality_bonus_weight
        if effective_orthogonality_bonus > 0.0:
            nt_matrix = jnp.stack(neurotransmitters, axis=0)
            norms = jnp.linalg.norm(nt_matrix, axis=1, keepdims=True)
            normalized = nt_matrix / (norms + 1e-8)
            similarity_matrix = normalized @ normalized.T
            mask = ~jnp.eye(num_tasks, dtype=bool)

            # Use configured orthogonality metric
            if cfg_orthogonality_metric == 'cosine_mean':
                avg_similarity = jnp.mean(jnp.abs(similarity_matrix[mask]))
                orthogonality = 1.0 - avg_similarity
            elif cfg_orthogonality_metric == 'cosine_max':
                max_similarity = jnp.max(jnp.abs(similarity_matrix[mask]))
                orthogonality = 1.0 - max_similarity
            elif cfg_orthogonality_metric == 'correlation':
                # For correlation, we'd need hidden activations - use cosine_mean as fallback
                avg_similarity = jnp.mean(jnp.abs(similarity_matrix[mask]))
                orthogonality = 1.0 - avg_similarity
            else:
                avg_similarity = jnp.mean(jnp.abs(similarity_matrix[mask]))
                orthogonality = 1.0 - avg_similarity

            aggregated_fitness = aggregated_fitness + effective_orthogonality_bonus * orthogonality

        # Apply specialization bonus (reward different activation patterns per task)
        if specialization_bonus > 0.0:
            fitness_variance = jnp.var(stacked_fitnesses, axis=0)
            specialization = jnp.mean(fitness_variance)
            aggregated_fitness = aggregated_fitness + specialization_bonus * specialization

        # Apply generalist bonus (ported from neuromodulation_functions.py)
        if cfg_generalist_bonus_type != 'none' and cfg_generalist_bonus_weight > 0.0:
            if cfg_generalist_bonus_type == 'min_bonus':
                # Bonus based on minimum fitness across tasks (rewards consistent performance)
                min_fitnesses = jnp.min(stacked_fitnesses, axis=0)
                generalist_bonus = cfg_generalist_bonus_weight * min_fitnesses
                aggregated_fitness = aggregated_fitness + generalist_bonus
            elif cfg_generalist_bonus_type == 'variance_penalty':
                # Penalty for high variance across tasks (encourages balanced performance)
                fitness_variance_per_ind = jnp.var(stacked_fitnesses, axis=0)
                variance_penalty = cfg_generalist_bonus_weight * fitness_variance_per_ind
                aggregated_fitness = aggregated_fitness - variance_penalty
            elif cfg_generalist_bonus_type == 'threshold_bonus':
                # Bonus if all tasks exceed threshold
                above_threshold = stacked_fitnesses >= cfg_generalist_threshold
                all_above = jnp.all(above_threshold, axis=0)
                threshold_bonus = jnp.where(all_above, cfg_generalist_bonus_weight, 0.0)
                aggregated_fitness = aggregated_fitness + threshold_bonus

        # ====================================================================
        # STEP 6: Run NEAT selection with aggregated fitness
        # ====================================================================
        new_state = self._compiled_tell(state, aggregated_fitness)

        # Update generation counter
        self._current_generation += 1

        # Build multi-task metrics
        total_time = time_module.perf_counter() - total_start

        best_fitness = float(jnp.max(aggregated_fitness))
        mean_fitness = float(jnp.mean(aggregated_fitness))
        min_fitness = float(jnp.min(aggregated_fitness))
        std_fitness = float(jnp.std(aggregated_fitness))

        # Find best generalist (single individual with highest aggregated fitness)
        best_gen_idx = int(jnp.argmax(aggregated_fitness))
        best_generalist_per_task = {
            f'task_{i}': float(per_task_fitnesses[i][best_gen_idx])
            for i in range(num_tasks)
        }

        custom_metrics = {
            'method': 'multitask',
            'num_tasks': num_tasks,
            'aggregation_method': aggregation_method,
            'per_task_best': [float(jnp.max(f)) for f in per_task_fitnesses],
            'per_task_mean': [float(jnp.mean(f)) for f in per_task_fitnesses],
            'per_task_times_ms': [t * 1000 for t in per_task_times],
            'best_generalist_idx': best_gen_idx,
            'best_generalist_per_task': best_generalist_per_task,
            'orthogonality_bonus_applied': effective_orthogonality_bonus if effective_orthogonality_bonus > 0 else None,
            'specialization_bonus_applied': specialization_bonus if specialization_bonus > 0 else None,
            # Extended features (ported from neuromodulation_functions.py)
            'fitness_mode': cfg_fitness_mode,
            'modulation_mode': cfg_modulation_mode,
            'two_module_mode': cfg_two_module_mode if cfg_two_module_mode != 'none' else None,
            'orthogonality_metric': cfg_orthogonality_metric,
            'generalist_bonus_type': cfg_generalist_bonus_type if cfg_generalist_bonus_type != 'none' else None,
            'generalist_bonus_weight': cfg_generalist_bonus_weight if cfg_generalist_bonus_weight > 0 else None,
        }

        # Update fitness history for backward transfer tracking
        if 'fitness_history' not in self._multitask_state:
            self._multitask_state['fitness_history'] = []
        self._multitask_state['fitness_history'].append(
            [float(jnp.max(f)) for f in per_task_fitnesses]
        )

        total_time = time_module.perf_counter() - total_start
        metrics = AlgorithmMetrics(
            generation=self._current_generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=best_fitness,
            std_fitness=std_fitness,
            num_species=1,
            species_sizes=[len(aggregated_fitness)],  # All in one "species"
            species_fitness=[float(best_fitness)],
            evaluations=len(aggregated_fitness) * num_tasks,  # pop_size * num_tasks
            time_elapsed=total_time,
            custom_metrics=custom_metrics,
        )

        return new_state, metrics

    def set_task_neurotransmitter(self, task_idx: int, nt_vector: jnp.ndarray) -> None:
        """Set the neurotransmitter vector for a specific task.

        Args:
            task_idx: Task index
            nt_vector: NT vector (typically 4D: [DA, 5HT, NE, ACh])
        """
        self._multitask_state['current_task_idx'] = task_idx
        self._multitask_state['current_nt_vector'] = nt_vector

        if self.neuromod_config is not None:
            self.neuromod_config.neurotransmitter = nt_vector

    def get_task_neurotransmitter(self, task_name: str) -> jnp.ndarray:
        """Get the preset neurotransmitter vector for a named task.

        Args:
            task_name: One of 'xor', 'and', 'or', 'nand', 'nor', 'classification',
                      'regression', 'memory', 'sequential'

        Returns:
            NT vector for the task
        """
        return NT_TASK_PRESETS.get(task_name, NT_TASK_PRESETS['xor'])

    def evaluate_with_neurotransmitter(
        self,
        nt_vector: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        activation_name: Optional[str] = None,
        fitness_mode: str = 'mse',
        modulation_mode: str = 'full',
    ) -> jnp.ndarray:
        """Evaluate population with a specific neurotransmitter vector and activation.

        This method allows evaluating the current population on arbitrary data
        with a custom NT vector and activation function. Useful for:
        - Testing networks on novel tasks without re-evolution
        - Probing multi-task capabilities
        - Benchmarking with different activation functions

        Prerequisites:
        - Must have called run_generation_verbose() or run_generation_multitask()
          at least once to cache W1, W2, receptor_densities, base_gains

        Args:
            nt_vector: Neurotransmitter vector, shape (num_nt,) typically (4,)
            inputs: Input data, shape (num_cases, num_inputs)
            targets: Target data, shape (num_cases, num_outputs)
            activation_name: Optional activation function name from ACTIVATION_FUNCTIONS.
                If None, uses 'tanh'. Examples: 'sin', 'relu', 'burst', 'gauss'
            fitness_mode: Fitness computation mode ('mse', 'accuracy', 'bce', etc.)
            modulation_mode: Neuromodulation mode ('full', 'gating_only', 'gain_bias_only')

        Returns:
            Fitness array of shape (pop_size,)

        Raises:
            RuntimeError: If weight matrices not cached (run_generation_verbose first)

        Example:
            >>> algo.run_generation_verbose(state, xor_problem)  # Cache weights
            >>> # Test with sin activation on a new task
            >>> fitness = algo.evaluate_with_neurotransmitter(
            ...     nt_vector=jnp.array([0.9, 0.1, 0.9, 1.0]),
            ...     inputs=jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            ...     targets=jnp.array([[0], [1], [1], [0]]),
            ...     activation_name='sin'
            ... )
        """
        # Validate cached data exists
        if not hasattr(self, '_cached_W1') or self._cached_W1 is None:
            raise RuntimeError(
                "Weight matrices not cached. Call run_generation_verbose() or "
                "run_generation_multitask() first to build and cache W1, W2."
            )
        if not hasattr(self, '_neuromod_true') or self._neuromod_true is None:
            raise RuntimeError(
                "TRUE neuromodulation data not available. Ensure neuromodulation "
                "is enabled with mode='true_neuromodulation'."
            )

        W1 = self._cached_W1
        W2 = self._cached_W2
        rd = self._neuromod_true['receptor_densities']
        bg = self._neuromod_true['base_gains']

        # Check for H→H connections
        sparse_hh = getattr(self, '_cached_sparse_hh', None)
        has_hh = (
            sparse_hh is not None and
            self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH
        )
        activate_time = self.activate_time if has_hh else 0

        # Get activation function
        act_fn = ACTIVATION_FUNCTIONS.get(activation_name, jnp.tanh) if activation_name else jnp.tanh

        # Create evaluation function
        eval_population = self._create_eval_population_true_neuromodulated(
            has_hh=has_hh,
            activate_time=activate_time,
            hidden_act_fn=act_fn,
        )

        # Ensure arrays are JAX arrays
        nt_vector = jnp.array(nt_vector)
        inputs = jnp.array(inputs)
        targets = jnp.array(targets)

        # Evaluate
        if has_hh and sparse_hh is not None:
            fitness = eval_population(
                W1, W2, rd, bg, nt_vector, inputs, targets,
                sparse_hh.from_indices,
                sparse_hh.to_indices,
                sparse_hh.weights,
                sparse_hh.valid_mask,
                fitness_mode, modulation_mode, None
            )
        else:
            fitness = eval_population(
                W1, W2, rd, bg, nt_vector, inputs, targets,
                fitness_mode, modulation_mode, None
            )

        return fitness

    # ========================================================================
    # Test Evaluation with Dynamic Function Support
    # ========================================================================

    def evaluate_on_data(
        self,
        state: Any,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> Dict[str, float]:
        """Evaluate the best genome on arbitrary input/target data.

        Overrides base class to preserve CPPN-assigned per-node activation
        functions when dynamic_functions_mode == 'cppn_output'. The base class
        uses hardcoded jnp.tanh for all hidden nodes, which produces incorrect
        (severely pessimistic) results when per-node activations were evolved.

        Args:
            state: Current algorithm state (contains population)
            inputs: Input data array of shape (n_samples, n_features)
            targets: Target data array of shape (n_samples, n_outputs)

        Returns:
            Dict with 'mse', 'fitness', 'n_samples', and substrate statistics.
            Includes 'activation_indices' and 'activation_names' when dynamic
            functions are enabled.
        """
        # Get base result (substrate W1/W2/mask are correct; forward pass may be wrong)
        result = super().evaluate_on_data(state, inputs, targets)

        # If dynamic functions disabled, the base class forward pass is correct
        if getattr(self, 'dynamic_functions_mode', 'disabled') == 'disabled':
            return result

        # Dynamic functions are enabled, recompute forward pass with per-node activations
        W1_best = result['_W1']  # (num_inputs, total_positions)
        W2_best = result['_W2']  # (total_positions, num_outputs)
        mask_best = result['_mask']  # (total_positions,)
        h_grid = result['_h_grid']

        # Convert inputs/targets to JAX arrays if needed
        if not isinstance(inputs, jnp.ndarray):
            inputs = jnp.array(inputs, dtype=jnp.float32)
        if not isinstance(targets, jnp.ndarray):
            targets = jnp.array(targets, dtype=jnp.float32)

        # --- Derive per-node activation indices ---
        act_indices = None

        if self.dynamic_functions_mode == 'cppn_output' and self.num_cppn_outputs > 1:
            # Query self-connections to get activation channel for best genome
            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)
            all_positions = h_grid.all_positions

            self_conn_outputs = batch_query_population_self_connections(
                state, cppns_transformed, all_positions,
                self._jitted_cppn_forward,
                num_cppn_outputs=self.num_cppn_outputs,
            )
            # self_conn_outputs: (pop_size, num_positions, num_cppn_outputs)
            # Extract best genome (index 0), activation channel
            act_output_idx = self._cppn_output_indices.get('activation', 1)
            activation_raw = self_conn_outputs[0, :, act_output_idx]  # (num_positions,)
            activation_raw = jnp.where(mask_best, activation_raw, 0.0)

            num_activations = len(self.df_palette) if self.df_palette is not None else self.df_num_activations
            act_indices = continuous_to_index(activation_raw, num_activations)
            if self.df_palette is not None:
                act_indices = self.df_palette[act_indices]

        elif self.dynamic_functions_mode == 'cppn_output' and self.num_cppn_outputs == 1:
            # Single-output CPPN: derive activation from mean of W1
            activation_raw = jnp.mean(W1_best, axis=0)  # (total_positions,)
            num_activations = len(self.df_palette) if self.df_palette is not None else self.df_num_activations
            act_indices = continuous_to_index(activation_raw, num_activations)
            if self.df_palette is not None:
                act_indices = self.df_palette[act_indices]

        elif self.dynamic_functions_mode == 'global':
            # Global mode: all nodes use the same activation (stored in self._hidden_act_func)
            act_name = getattr(self, 'df_global_activation', 'tanh')
            act_func = ACTIVATION_FUNCTIONS.get(act_name, jnp.tanh)
            hidden_pre = inputs @ W1_best  # (n_samples, total_positions)
            hidden = act_func(hidden_pre)
            output_act_name = getattr(self, 'df_output_activation', 'sigmoid')
            output_act = ACTIVATION_FUNCTIONS.get(output_act_name, jax.nn.sigmoid)
            outputs = output_act(hidden @ W2_best)
            # Recompute metrics
            errors = (outputs - targets) ** 2
            mse = float(jnp.mean(errors))
            fitness = max(0.0, min(1.0, 1.0 - mse))
            sample_errors = jnp.mean(errors, axis=1)
            result.update({
                'mse': mse, 'fitness': fitness,
                'mse_std': float(jnp.std(sample_errors)),
                'mse_min': float(jnp.min(sample_errors)),
                'mse_max': float(jnp.max(sample_errors)),
                'mse_median': float(jnp.median(sample_errors)),
                'dynamic_functions_mode': 'global',
                'global_activation': act_name,
            })
            return result

        # If we have per-node activation indices, recompute forward pass
        if act_indices is not None:
            hidden_pre = inputs @ W1_best  # (n_samples, total_positions)

            # Apply per-node activations
            if self.df_palette is not None:
                hidden = grouped_activation_forward_with_palette(
                    hidden_pre, act_indices, jnp.array(self.df_palette)
                )
            else:
                num_activations = getattr(self, 'df_num_activations', 6)
                hidden = grouped_activation_forward(hidden_pre, act_indices, num_activations)

            # Output activation
            output_act_name = getattr(self, 'df_output_activation', 'sigmoid')
            output_act = ACTIVATION_FUNCTIONS.get(output_act_name, jax.nn.sigmoid)
            outputs = output_act(hidden @ W2_best)

            # Recompute metrics
            errors = (outputs - targets) ** 2
            mse = float(jnp.mean(errors))
            fitness = max(0.0, min(1.0, 1.0 - mse))
            sample_errors = jnp.mean(errors, axis=1)

            # Build activation name counts for logging
            act_indices_np = np.array(act_indices)
            act_names = {}
            for idx in range(len(ACTIVATION_LIST)):
                count = int(np.sum(act_indices_np == idx))
                if count > 0:
                    act_names[ACTIVATION_LIST[idx]] = count

            # Build per-level activation counts
            activation_counts_per_level = {}
            if h_grid is not None and hasattr(h_grid, 'level_offsets_static'):
                offsets = h_grid.level_offsets_static
                num_levels = len(offsets) - 1
                for lvl in range(num_levels):
                    lvl_start = offsets[lvl]
                    lvl_end = offsets[lvl + 1]
                    lvl_indices = act_indices_np[lvl_start:lvl_end]
                    lvl_counts = {}
                    for idx in range(len(ACTIVATION_LIST)):
                        count = int(np.sum(lvl_indices == idx))
                        if count > 0:
                            lvl_counts[ACTIVATION_LIST[idx]] = count
                    activation_counts_per_level[f'level_{lvl}'] = {
                        'num_positions': int(lvl_end - lvl_start),
                        'counts': lvl_counts,
                    }

            result.update({
                'mse': mse, 'fitness': fitness,
                'mse_std': float(jnp.std(sample_errors)),
                'mse_min': float(jnp.min(sample_errors)),
                'mse_max': float(jnp.max(sample_errors)),
                'mse_median': float(jnp.median(sample_errors)),
                'dynamic_functions_mode': self.dynamic_functions_mode,
                'activation_counts': act_names,
                'activation_counts_per_level': activation_counts_per_level,
                'num_activations_used': len(act_names),
            })

        return result

    # ========================================================================
    # Multi-GPU run_until_threshold() Override
    # ========================================================================

    def run_until_threshold(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        streaming_chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """Run multiple generations with h→h discovery support.

        Routes to appropriate implementation based on strategy:

        ============================================================================
        5 AVAILABLE GPU STRATEGIES
        ============================================================================

        Decision tree:
          Dataset fits on single GPU?
          ├─ YES → SINGLE_GPU (fastest)
          └─ NO  → Multiple GPUs available?
                   ├─ NO  → STREAMING (streams from CPU)
                   └─ YES → Recurrent/h→h mode?
                            ├─ YES → EVAL_ONLY_PARALLEL (h→h caching)
                            └─ NO  → FULL_PIPELINE_PARALLEL (data parallel)

          POPULATION_PARALLEL_PROCESS: For very large populations with h→h

        1. SINGLE_GPU (default):
           Single GPU execution using run_generation_verbose()
           Fastest when memory fits; uses all unified extended features

        2. FULL_PIPELINE_PARALLEL:
           Full pipeline runs on each GPU: CPPN → variance → W1/W2 → eval
           Dataset is SPLIT across GPUs, pipeline is REPLICATED
           Recommended for: Feedforward mode, large statistical datasets (MNIST)
           Memory: ~balanced across GPUs

        3. STREAMING:
           CPU→GPU data streaming for memory management
           Uses single GPU but streams data in chunks
           Recommended for: Huge datasets that cause GPU OOM

        4. EVAL_ONLY_PARALLEL:
           Only evaluation parallelized, CPPN/h→h runs once on GPU 0
           H→H caching enabled: 27s → 3ms after first generation
           Recommended for: Recurrent (h→h) modes
           Memory: ~balanced across GPUs

        5. POPULATION_PARALLEL_PROCESS:
           ProcessPoolExecutor-based true parallel processing
           Each GPU runs in separate Python process with isolated JAX runtime
           Recommended for: Very large populations with expensive h→h

        FALLBACK BEHAVIOR:
        - Multi-GPU strategies fall back to SINGLE_GPU if only 1 GPU available
        - FULL_PIPELINE_PARALLEL falls back to SINGLE_GPU for exhaustive datasets

        AUTOMATIC FALLBACK FOR EXHAUSTIVE DATASETS:
        - If problem.is_exhaustive_dataset is True, FULL_PIPELINE_PARALLEL is disabled
          and single-GPU mode is used instead, regardless of GPU count.
        - This ensures correct fitness evaluation for combinatorial problems
          (XOR, logic gates, parity) where ALL samples must be evaluated together.
        - Statistical datasets (MNIST, ImageNet) can still use FULL_PIPELINE_PARALLEL.

        This override ensures the unified extended features (multi-hop h→h discovery,
        sparse forward pass, caching) are used regardless of GPU strategy.

        Args:
            state: Initialized algorithm state
            problem: Problem instance (checked for is_exhaustive_dataset property)
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            collect_history: If True, collect per-generation history
            streaming_chunk_size: Chunk size for streaming (unused in extended)

        Returns:
            Dict with 'generations', 'best_fitness', 'state', 'history' (optional)
        """
        num_devices = len(jax.devices())

        # Check if FULL_PIPELINE_PARALLEL sharding is appropriate for this problem
        # FULL_PIPELINE_PARALLEL shards data across GPUs - each GPU evaluates on a subset
        # This BREAKS correctness for exhaustive/combinatorial problems (XOR, logic gates)
        # where ALL samples are needed for meaningful fitness evaluation
        n_samples = self._cached_inputs.shape[0] if hasattr(self, '_cached_inputs') and self._cached_inputs is not None else 0

        # Check 1: Exhaustive datasets CANNOT be sharded (each GPU needs ALL samples)
        # Examples: XOR (4 samples), logic gates, parity problems
        is_exhaustive = getattr(problem, 'is_exhaustive_dataset', False)

        # Check 2: Statistical datasets need enough samples per GPU for representativeness
        # Rule of thumb: at least 16 samples per GPU for meaningful fitness
        MIN_SAMPLES_PER_GPU = 16
        has_enough_samples = n_samples >= (MIN_SAMPLES_PER_GPU * num_devices)

        # Can use data-parallel only if: NOT exhaustive AND enough samples
        can_use_data_parallel = (not is_exhaustive) and has_enough_samples

        # Route based on strategy
        # ============================================================================
        # 5 AVAILABLE GPU STRATEGIES (legacy aliases normalized at config parsing)
        # ============================================================================
        # 1. SINGLE_GPU          - Single GPU execution (default, fastest)
        # 2. FULL_PIPELINE_PARALLEL - Multi-GPU data parallel (OOM prevention)
        # 3. STREAMING           - CPU→GPU streaming (huge datasets, OOM safe)
        # 4. EVAL_ONLY_PARALLEL  - Multi-GPU for recurrent/h→h modes
        # 5. POPULATION_PARALLEL_PROCESS - ProcessPool for large populations
        # ============================================================================
        if num_devices >= 2:
            if self.strategy == MultiGPUStrategy.FULL_PIPELINE_PARALLEL:
                if can_use_data_parallel:
                    # Data-parallel: full pipeline inside pmap, balanced memory
                    print(f"[Strategy] FULL_PIPELINE_PARALLEL: Using {num_devices} GPUs")
                    print(f"  - Sharding {n_samples} samples across GPUs ({n_samples // num_devices} samples/GPU)")
                    print(f"  - Best for: Large statistical datasets (MNIST, ImageNet) where subsets are representative")
                    print(f"  - Each GPU evaluates FULL population on its data shard, fitnesses averaged")
                    return self._run_until_threshold_data_parallel_full(
                        state, problem, target_fitness, max_generations, collect_history
                    )
                else:
                    # Cannot use data-parallel sharding - fall back to single GPU
                    # to evaluate each genome on ALL samples for correct fitness
                    print(f"[Strategy] FULL_PIPELINE_PARALLEL requested but falling back to SINGLE_GPU")
                    if is_exhaustive:
                        print(f"  - Reason: Exhaustive dataset ({n_samples} samples) - ALL samples required for correct fitness")
                        print(f"  - Problem type: Combinatorial (XOR, logic gates, parity) where every sample is unique")
                        print(f"  - Why fallback: Sharding would give wrong fitness (e.g., 100% on GPU0, 0% on GPU1 = 50% avg)")
                        print(f"  - Recommendation: Use POPULATION_PARALLEL_PROCESS instead for exhaustive datasets with multi-GPU")
                    else:
                        print(f"  - Reason: Dataset too small ({n_samples} samples, need {MIN_SAMPLES_PER_GPU * num_devices}+)")
                        print(f"  - Why fallback: Too few samples per GPU for statistically meaningful fitness")
                        print(f"  - Recommendation: Use POPULATION_PARALLEL_PROCESS, or increase dataset size")
                    return self._run_until_threshold_single_gpu(
                        state, problem, target_fitness, max_generations, collect_history
                    )
            elif self.strategy == MultiGPUStrategy.POPULATION_PARALLEL_PROCESS:
                # ProcessPoolExecutor-based true parallel h→h processing
                # Each GPU runs in a separate Python process with isolated JAX runtime
                h_grid = get_hierarchical_grid(self.max_depth)
                total_positions = h_grid.total_positions
                allow_hh = self.extended_config.allow_hidden_to_hidden if self.extended_config else False

                print(f"[Strategy] POPULATION_PARALLEL_PROCESS: Using {num_devices} GPUs with ProcessPoolExecutor")
                print(f"  - Each GPU runs in separate Python process (isolated JAX runtime)")
                print(f"  - True parallel execution without JIT cache cross-device errors")
                print(f"  - h→h discovery at depth {self.max_depth} ({total_positions} positions)")
                print(f"  - Splitting population across GPUs, each GPU gets ALL {n_samples} samples")
                print(f"  - Trade-off: Higher IPC overhead but true parallelism")

                return self._run_until_threshold_device_parallel_process(
                    state, problem, target_fitness, max_generations, collect_history
                )

            elif self.strategy == MultiGPUStrategy.EVAL_ONLY_PARALLEL:
                # Native jax.pmap multi-GPU with zero IPC overhead (RECOMMENDED for h→h)
                h_grid = get_hierarchical_grid(self.max_depth)
                total_positions = h_grid.total_positions
                has_hh = self.extended_config and self.extended_config.allow_hidden_to_hidden

                print(f"[Strategy] EVAL_ONLY_PARALLEL: Only eval parallelized, h→h cached on GPU 0")
                print(f"  - CPPN queries + h→h discovery + W1/W2: GPU 0 only (h→h cache preserved)")
                print(f"  - Evaluation: jax.pmap across {num_devices} GPUs (true parallel)")
                print(f"  - Data sharding: Each GPU evaluates on {n_samples // num_devices}+ samples")
                print(f"  - H→H caching: {'ENABLED (GPU 0)' if has_hh else 'N/A (feedforward)'}")
                print(f"  - Depth {self.max_depth} ({total_positions} positions)")
                print(f"  - Best for: Recurrent modes where h→h caching saves time")

                return self._run_until_threshold_pmap(
                    state, problem, target_fitness, max_generations, collect_history
                )

            elif self.strategy == MultiGPUStrategy.STREAMING:
                # Streaming: Level-by-level CPPN queries for memory efficiency
                # Enables depth 7+ on CPU without OOM
                h_grid = get_hierarchical_grid(self.max_depth)
                print(f"[Strategy] STREAMING: Level-by-level CPPN queries for memory efficiency")
                print(f"  - Purpose: Reduce peak memory from O(total_positions) to O(max_level_size)")
                print(f"  - How it works: Queries CPPN one level at a time, accumulates W1/W2")
                print(f"  - Best for: Depth 7+ on CPU, or low-memory GPUs")
                print(f"  - Trade-off: 3-10x slower per generation")
                print(f"  - Depth {self.max_depth} ({h_grid.total_positions} positions)")

                # Enable streaming mode for this run
                original_streaming = self.enable_streaming
                self.enable_streaming = True
                try:
                    return self._run_until_threshold_single_gpu(
                        state, problem, target_fitness, max_generations, collect_history
                    )
                finally:
                    self.enable_streaming = original_streaming

        # Fallback for single GPU or when multi-GPU strategies run on single device
        # Note: Legacy aliases (MULTI_GPU, DATA_PARALLEL, etc.) are normalized to
        # primary strategies at config parsing time, so only check primary 5 here
        if self.strategy in (
            MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
            MultiGPUStrategy.EVAL_ONLY_PARALLEL,
            MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
        ):
            print(f"[Strategy] {self.strategy.name} requested but only {num_devices} GPU available")
            print(f"  - Falling back to SINGLE_GPU mode")
            print(f"  - For multi-GPU: Ensure multiple GPUs are visible to JAX")
        elif self.strategy == MultiGPUStrategy.STREAMING:
            # Streaming: Level-by-level CPPN queries for memory efficiency
            h_grid = get_hierarchical_grid(self.max_depth)
            print(f"[Strategy] STREAMING: Level-by-level CPPN queries for memory efficiency")
            print(f"  - Purpose: Reduce peak memory from O(total_positions) to O(max_level_size)")
            print(f"  - Best for: Depth 7+ on CPU, or low-memory GPUs")
            print(f"  - Trade-off: 3-10x slower per generation")
            print(f"  - Depth {self.max_depth} ({h_grid.total_positions} positions)")

            # Enable streaming mode for this run
            original_streaming = self.enable_streaming
            self.enable_streaming = True
            try:
                return self._run_until_threshold_single_gpu(
                    state, problem, target_fitness, max_generations, collect_history
                )
            finally:
                self.enable_streaming = original_streaming

        elif self.strategy == MultiGPUStrategy.SINGLE_GPU:
            print(f"[Strategy] SINGLE_GPU: Using 1 GPU")
            print(f"  - All computation on primary GPU")

        # For single GPU, CPU, or fallback: use run_generation_verbose() loop
        return self._run_until_threshold_single_gpu(
            state, problem, target_fitness, max_generations, collect_history
        )

    def _run_until_threshold_single_gpu(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Single GPU implementation using existing run_generation_verbose().

        This preserves all unified extended features (h→h discovery, multi-hop,
        caching) by using the existing per-generation implementation.

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        if self.verbose:
            print(f"[UnifiedExtended] Single GPU mode: max_gens={max_generations}, target={target_fitness}")

        while best_so_far < target_fitness and generation < max_generations:
            state, metrics = self.run_generation_verbose(state, problem)
            gen_best = metrics.best_fitness
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose and generation % 10 == 0:
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_device_parallel_hh(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Device-parallel multi-GPU implementation for h→h at large depths.

        Uses Python threading to run single-GPU-style processing on each device.
        This avoids pmap's XLA trace explosion that causes OOM for h→h at depth 7+.

        Each thread:
        1. Places its population shard on its assigned device
        2. Runs h→h discovery + evaluation (Python for-loops, no XLA trace explosion)
        3. Returns fitnesses to main thread

        Main thread:
        1. Splits population between devices
        2. Launches worker threads
        3. Combines fitnesses
        4. Runs NEAT evolution

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        devices = jax.devices()
        num_devices = len(devices)

        if self.verbose:
            print(f"[DeviceParallel] Using {num_devices} GPUs with threaded h→h processing")

        # Get hierarchical grid info
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0
        cache_hit_count = 0
        cache_refresh_count = 0

        # Thread-safe result storage
        thread_results = {}
        thread_lock = threading.Lock()

        def process_shard_on_device(
            device_idx: int,
            cppn_shard: Tuple,
            shard_start: int,
            shard_end: int,
            cached_hh_shard: Optional[SparseHiddenConnections] = None,
            global_union_active: Optional[jnp.ndarray] = None,
            precomputed_masks_A_shard: Optional[jnp.ndarray] = None,
        ):
            """Process a population shard on a specific device using single-GPU-style logic.

            Args:
                device_idx: Device index (0, 1, ...)
                cppn_shard: CPPN genomes for this shard
                shard_start: Start index in full population
                shard_end: End index in full population
                cached_hh_shard: Optional cached h→h connections for this shard (cache hit)
                global_union_active: Optional pre-computed global union of active positions.
                    When provided, ensures deterministic h→h discovery across shards.
                precomputed_masks_A_shard: Optional pre-computed masks_A for this shard.
                    When provided, skip variance/mask computation (already done globally).
            """
            device = devices[device_idx]
            shard_size = shard_end - shard_start

            # Use per-device JIT function to avoid device placement errors
            # JAX JIT functions are compiled to a specific device and ignore thread-local contexts
            jitted_cppn_forward = getattr(self, '_jitted_cppn_forward_per_device', {}).get(
                device, self._jitted_cppn_forward
            )

            # Use default_device context to ensure all JAX operations go to the correct device
            with jax.default_device(device):
                # Place data on device
                cppn_shard_on_device = jax.tree.map(lambda x: jax.device_put(x, device), cppn_shard)
                inputs_on_device = jax.device_put(inputs_batch, device)
                targets_on_device = jax.device_put(targets_batch, device)
                all_pos_on_device = jax.device_put(all_positions, device)
                input_coords_on_device = jax.device_put(input_coords, device)
                output_coords_on_device = jax.device_put(output_coords, device)
                # CRITICAL: state contains arrays used by cppn_forward inside vmap
                # Without device placement, these arrays stay on cuda:0 causing
                # "Buffer passed to Execute()... is on device cuda:0, but replica is assigned to device cuda:1"
                state_on_device = jax.tree.map(lambda x: jax.device_put(x, device), state)

                # Run CPPN queries for this shard
                # Pass device_idx to force device-specific vmap traces
                # This prevents cuda:1 from reusing traces compiled for cuda:0
                input_all_weights = batch_query_population_multi_source_chunked(
                    state_on_device, cppn_shard_on_device, input_coords_on_device, all_pos_on_device,
                    True, jitted_cppn_forward, device_id=device_idx,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )
                output_all_weights = batch_query_population_multi_source_chunked(
                    state_on_device, cppn_shard_on_device, output_coords_on_device, all_pos_on_device,
                    False, jitted_cppn_forward, device_id=device_idx,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )

                # Variance + subdivision masks
                # Use precomputed masks if available (for deterministic multi-GPU)
                if precomputed_masks_A_shard is not None:
                    masks_A = jax.device_put(precomputed_masks_A_shard, device)
                else:
                    all_weights_for_variance = input_all_weights[:, 0, :]
                    level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
                    masks_A = compute_subdivision_masks_batch(
                        level_variances, self.variance_threshold, h_grid, return_all_masks=False
                    )
                    # CRITICAL: Ensure masks_A is on the correct device
                    # compute_subdivision_masks_batch uses cached JIT which may place arrays on wrong device
                    masks_A = jax.device_put(masks_A, device)

                # H→H: use cached if provided, otherwise discover
                sparse_hh = None
                shard_hh_count = 0
                discovered_hh = None
                if self.extended_config is not None and self.extended_config.allow_hidden_to_hidden:
                    if cached_hh_shard is not None:
                        # Cache hit: use the cached connections
                        sparse_hh = cached_hh_shard
                        shard_hh_count = int(jnp.sum(sparse_hh.num_valid))
                    else:
                        # Cache miss: discover new connections
                        # Use global_union_active for deterministic behavior across shards
                        global_union_on_device = None
                        if global_union_active is not None:
                            global_union_on_device = jax.device_put(global_union_active, device)
                        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                            state_on_device, cppn_shard_on_device, h_grid, masks_A,
                            self.band_threshold, self.max_weight,
                            self.extended_config, jitted_cppn_forward,
                            verbose=False,
                            global_union_active=global_union_on_device,
                            device_id=device_idx,
                            geometry_seeding_enabled=self.geometry_seeding_enabled,
                            num_cppn_outputs=getattr(self, 'num_cppn_outputs', 1),
                        )
                        if sparse_hh is not None:
                            shard_hh_count = int(jnp.sum(sparse_hh.num_valid))
                            # Store for caching later
                            discovered_hh = sparse_hh

                # Build W1/W2 matrices (inside device context)
                weight_thresh = 0.1
                max_weight_val = self.max_weight
                active_mask_broadcast = masks_A[:, None, :]
                W1_raw = jnp.tanh(input_all_weights) * max_weight_val
                W1 = jnp.where(
                    active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                    W1_raw, 0.0
                )
                W2_raw = jnp.tanh(output_all_weights) * max_weight_val
                W2 = jnp.where(
                    active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                    W2_raw, 0.0
                )
                W2 = W2.transpose(0, 2, 1)

                # Evaluation (inside device context)
                # CRITICAL: Wrap in device-specific JIT to prevent lax.scan
                # from being compiled to cuda:0 and reused on cuda:1.
                # The issue is that forward_hybrid_vmapped contains lax.scan
                # which gets traced once and cached. Without explicit device
                # placement, the first thread (cuda:0) compiles it, and the
                # second thread (cuda:1) reuses the cuda:0 trace, causing
                # device placement errors.
                if sparse_hh is not None:
                    eval_fn = functools.partial(
                        eval_single_network_hybrid,
                        inputs=inputs_on_device,
                        targets=targets_on_device,
                        activate_time=self.activate_time,
                    )
                    # Use device-specific JIT for evaluation
                    # This ensures lax.scan is compiled for THIS device
                    eval_vmapped = jax.jit(
                        lambda w1, w2, fr, to, wt, vm: jax.vmap(eval_fn)(w1, w2, fr, to, wt, vm),
                        device=device
                    )
                    fitnesses = eval_vmapped(
                        W1, W2,
                        sparse_hh.from_indices, sparse_hh.to_indices,
                        sparse_hh.weights, sparse_hh.valid_mask,
                    )
                else:
                    def eval_single_dense(w1, w2):
                        hidden = jnp.tanh(inputs_on_device @ w1.T)
                        outputs = jax.nn.sigmoid(hidden @ w2)
                        errors = jnp.mean((outputs - targets_on_device) ** 2, axis=1)
                        return 1.0 - jnp.mean(errors)
                    # Device-specific JIT for dense evaluation too
                    eval_vmapped_dense = jax.jit(
                        lambda w1, w2: jax.vmap(eval_single_dense)(w1, w2),
                        device=device
                    )
                    fitnesses = eval_vmapped_dense(W1, W2)

                # Block and get results back to CPU
                fitnesses = jax.device_get(fitnesses)

            with thread_lock:
                thread_results[device_idx] = {
                    'fitnesses': fitnesses,
                    'hh_count': shard_hh_count,
                    'shard_start': shard_start,
                    'discovered_hh': discovered_hh,  # For caching (None if cache hit)
                    'shard_end': shard_end,
                }

        # Main evolution loop
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # Get CPPN population
            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Split population between devices
            shard_size = pop_size // num_devices
            thread_results.clear()

            # Check cache for h→h connections
            use_cache = (
                self._hh_cache is not None and
                self.extended_config is not None and
                self.extended_config.allow_hidden_to_hidden
            )
            cached_sparse_hh = None
            is_cache_hit = False

            if use_cache:
                # Check if we have a valid cache (we can't check mask here, so use generation-based check)
                if not self._hh_cache.should_refresh(self._current_generation, None):
                    cached_sparse_hh = self._hh_cache.get_cached()
                    is_cache_hit = cached_sparse_hh is not None
                    if is_cache_hit:
                        cache_hit_count += 1
                        if self.verbose:
                            print(f"[Gen {self._current_generation}] CACHE HIT - reusing connections")
                else:
                    cache_refresh_count += 1
                    if self.verbose:
                        print(f"[Gen {self._current_generation}] CACHE MISS - discovering new connections")

            # =====================================================================
            # H→H DISCOVERY: Each thread computes its own masks locally
            # =====================================================================
            # masks are computed per-shard, not pre-computed for the full population on GPU 0.
            # Pre-computing the full population (for deterministic h→h discovery) causes OOM
            # at large populations (750+) with depth 7 (87K positions) because
            # 750 × 87,380 × 4 bytes = 262 MB just for input weights, and the
            # full pipeline needs much more.
            #
            # The fix: Let each thread compute its own masks_A locally. This is
            # valid because each genome's h→h connections are independent - they
            # only depend on that genome's CPPN outputs, not other genomes.
            # =====================================================================
            global_union_active = None
            full_masks_A = None

            # Process devices SEQUENTIALLY to avoid JAX JIT cache cross-device errors.
            # ThreadPoolExecutor causes "Buffer on device cuda:X but replica assigned to cuda:Y"
            # because JAX's global JIT cache reuses compiled traces across threads/devices.
            # Sequential processing ensures each device gets clean JIT compilation.
            #
            # Trade-off: We lose parallel execution but gain guaranteed correctness.
            # For large populations, the h→h discovery work is the bottleneck anyway,
            # so sequential GPU processing is still faster than single-GPU.
            for d in range(num_devices):
                # CRITICAL: Clear ALL caches before each device to prevent cross-device
                # JIT trace reuse. Without this, device 0's compiled traces get reused
                # by device 1, causing "Buffer on cuda:0 but replica on cuda:1" errors.
                #
                # We clear BOTH:
                # 1. JAX's internal compilation cache (jax.clear_caches())
                # 2. Our module-level Python cache of JIT functions (clear_per_device_caches())
                #
                # The Python cache holds references to compiled functions that may have
                # device-specific traces. Even after jax.clear_caches(), these references
                # persist and can cause cross-device issues.
                jax.clear_caches()
                clear_per_device_caches()

                shard_start = d * shard_size
                shard_end = pop_size if d == num_devices - 1 else (d + 1) * shard_size
                cppn_shard = tuple(arr[shard_start:shard_end] for arr in cppns_transformed)

                # Slice cached h→h for this shard if cache hit
                cached_hh_shard = None
                if is_cache_hit and cached_sparse_hh is not None:
                    cached_hh_shard = SparseHiddenConnections(
                        from_indices=cached_sparse_hh.from_indices[shard_start:shard_end],
                        to_indices=cached_sparse_hh.to_indices[shard_start:shard_end],
                        weights=cached_sparse_hh.weights[shard_start:shard_end],
                        valid_mask=cached_sparse_hh.valid_mask[shard_start:shard_end],
                        num_valid=cached_sparse_hh.num_valid[shard_start:shard_end],
                    )

                # Slice pre-computed masks_A for this shard (if available)
                masks_A_shard = None
                if full_masks_A is not None:
                    masks_A_shard = full_masks_A[shard_start:shard_end]

                # Process sequentially - no threading
                process_shard_on_device(d, cppn_shard, shard_start, shard_end,
                    cached_hh_shard, global_union_active, masks_A_shard)

            # Combine results
            all_fitnesses = np.zeros(pop_size, dtype=np.float32)
            gen_hh_count = 0
            for d in range(num_devices):
                result = thread_results[d]
                all_fitnesses[result['shard_start']:result['shard_end']] = result['fitnesses']
                gen_hh_count += result['hh_count']

            # If cache miss, combine discovered h→h and update cache
            if use_cache and not is_cache_hit:
                # Combine discovered_hh from all shards
                all_discovered = []
                for d in range(num_devices):
                    if thread_results[d]['discovered_hh'] is not None:
                        all_discovered.append((d, thread_results[d]))

                if all_discovered:
                    # Combine all shard discoveries into one SparseHiddenConnections
                    # Move to CPU first to avoid device mismatch errors
                    combined_from = []
                    combined_to = []
                    combined_weights = []
                    combined_valid = []
                    combined_num_valid = []

                    for d, result in sorted(all_discovered, key=lambda x: x[1]['shard_start']):
                        hh = result['discovered_hh']
                        # Move each shard's arrays to CPU (numpy)
                        combined_from.append(jax.device_get(hh.from_indices))
                        combined_to.append(jax.device_get(hh.to_indices))
                        combined_weights.append(jax.device_get(hh.weights))
                        combined_valid.append(jax.device_get(hh.valid_mask))
                        combined_num_valid.append(jax.device_get(hh.num_valid))

                    # Concatenate on CPU, then convert back to JAX arrays
                    combined_sparse_hh = SparseHiddenConnections(
                        from_indices=jnp.array(np.concatenate(combined_from, axis=0)),
                        to_indices=jnp.array(np.concatenate(combined_to, axis=0)),
                        weights=jnp.array(np.concatenate(combined_weights, axis=0)),
                        valid_mask=jnp.array(np.concatenate(combined_valid, axis=0)),
                        num_valid=jnp.array(np.concatenate(combined_num_valid, axis=0)),
                    )

                    # Update cache
                    self._hh_cache.update_cache(combined_sparse_hh, None, self._current_generation)

            fitnesses = jnp.array(all_fitnesses)
            total_hh_count = max(total_hh_count, gen_hh_count)

            # Update best
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            # NEAT evolution
            state = self._compiled_tell(state, fitnesses)
            generation += 1
            self._current_generation = generation

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                elapsed = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"h→h={gen_hh_count}, time={elapsed:.2f}s")

        # Store extended metrics for benchmark reporting
        self._extended_metrics = EMRRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=h_grid.total_positions,
            # Config echo
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            # Population diversity from final fitnesses
            min_fitness=float(np.min(all_fitnesses)),
            max_fitness=float(np.max(all_fitnesses)),
            std_fitness=float(np.std(all_fitnesses)),
            # Cache stats
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            'sparse_hh_count': total_hh_count,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_device_parallel_process(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """ProcessPoolExecutor-based true parallel multi-GPU implementation.

        Uses separate Python processes for each GPU to achieve true parallelism.
        Each process has isolated JAX runtime and JIT cache, avoiding the
        cross-device contamination errors that plague ThreadPoolExecutor.

        Architecture:
        - Main process: Computes W1, W2, h→h on GPU 0, handles NEAT evolution
        - Worker processes: Each evaluates a population shard on its assigned GPU

        Trade-offs:
        - PRO: True parallel evaluation
        - PRO: No cross-device JIT cache errors
        - CON: CPPN queries still sequential (on GPU 0)
        - CON: IPC overhead (data serialization)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        from concurrent.futures import ProcessPoolExecutor
        from .multi_gpu_worker import evaluate_shard_worker, EvalShardInput, EvalShardOutput

        devices = jax.devices()
        num_devices = len(devices)

        print(f"[ProcessPoolExecutor] Using {num_devices} GPUs with parallel evaluation")

        # Get hierarchical grid info
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # Convert task data to numpy for worker IPC
        inputs_np = np.array(jax.device_get(inputs_batch))
        targets_np = np.array(jax.device_get(targets_batch))

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0
        cache_hit_count = 0
        cache_refresh_count = 0

        # Get spawn context for ProcessPoolExecutor
        ctx = mp.get_context('spawn')

        # Main evolution loop
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # =====================================================================
            # CPPN queries and h→h discovery on GPU 0 (sequential)
            # =====================================================================
            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # CPPN queries on GPU 0
            input_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, input_coords, all_positions,
                True, self._jitted_cppn_forward, device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )
            output_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, output_coords, all_positions,
                False, self._jitted_cppn_forward, device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )

            # Variance + masks
            all_weights_for_variance = input_all_weights[:, 0, :]
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )

            # H→H discovery
            sparse_hh = None
            gen_hh_count = 0
            allow_hh = self.extended_config is not None and self.extended_config.allow_hidden_to_hidden

            if allow_hh:
                # Check cache
                use_cache = self._hh_cache is not None
                is_cache_hit = False

                if use_cache:
                    if not self._hh_cache.should_refresh(self._current_generation, None):
                        sparse_hh = self._hh_cache.get_cached()
                        is_cache_hit = sparse_hh is not None
                        if is_cache_hit:
                            cache_hit_count += 1

                if not is_cache_hit:
                    cache_refresh_count += 1
                    sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                        state, cppns_transformed, h_grid, masks_A,
                        self.band_threshold, self.max_weight,
                        self.extended_config, self._jitted_cppn_forward,
                        verbose=False, device_id=0,
                        geometry_seeding_enabled=self.geometry_seeding_enabled,
                        num_cppn_outputs=self.num_cppn_outputs,
                    )
                    if use_cache and sparse_hh is not None:
                        self._hh_cache.update_cache(sparse_hh, None, self._current_generation)

                if sparse_hh is not None:
                    gen_hh_count = int(jnp.sum(sparse_hh.num_valid))

            # Build W1, W2 matrices
            weight_thresh = 0.1
            max_weight_val = self.max_weight
            active_mask_broadcast = masks_A[:, None, :]
            W1_raw = jnp.tanh(input_all_weights) * max_weight_val
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw, 0.0
            )
            W2_raw = jnp.tanh(output_all_weights) * max_weight_val
            W2 = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw, 0.0
            )
            W2 = W2.transpose(0, 2, 1)

            cppn_time = time.perf_counter() - gen_start

            # =====================================================================
            # Parallel evaluation using ProcessPoolExecutor
            # =====================================================================
            eval_start = time.perf_counter()

            # Convert to numpy for IPC
            W1_np = np.array(jax.device_get(W1))
            W2_np = np.array(jax.device_get(W2))

            hh_from_np = None
            hh_to_np = None
            hh_weights_np = None
            hh_valid_np = None
            hh_num_valid_np = None

            if sparse_hh is not None:
                hh_from_np = np.array(jax.device_get(sparse_hh.from_indices))
                hh_to_np = np.array(jax.device_get(sparse_hh.to_indices))
                hh_weights_np = np.array(jax.device_get(sparse_hh.weights))
                hh_valid_np = np.array(jax.device_get(sparse_hh.valid_mask))
                hh_num_valid_np = np.array(jax.device_get(sparse_hh.num_valid))

            # Create shards
            shard_size = pop_size // num_devices
            shard_inputs = []

            for d in range(num_devices):
                shard_start = d * shard_size
                shard_end = pop_size if d == num_devices - 1 else (d + 1) * shard_size
                actual_shard_size = shard_end - shard_start

                shard_input = EvalShardInput(
                    shard_start=shard_start,
                    shard_end=shard_end,
                    shard_size=actual_shard_size,
                    W1=W1_np[shard_start:shard_end],
                    W2=W2_np[shard_start:shard_end],
                    inputs=inputs_np,
                    targets=targets_np,
                    hh_from=hh_from_np[shard_start:shard_end] if hh_from_np is not None else None,
                    hh_to=hh_to_np[shard_start:shard_end] if hh_to_np is not None else None,
                    hh_weights=hh_weights_np[shard_start:shard_end] if hh_weights_np is not None else None,
                    hh_valid=hh_valid_np[shard_start:shard_end] if hh_valid_np is not None else None,
                    hh_num_valid=hh_num_valid_np[shard_start:shard_end] if hh_num_valid_np is not None else None,
                    activate_time=self.activate_time,
                )
                shard_inputs.append(shard_input)

            # Submit to workers
            with ProcessPoolExecutor(max_workers=num_devices, mp_context=ctx) as executor:
                futures = [
                    executor.submit(evaluate_shard_worker, d, shard_inputs[d])
                    for d in range(num_devices)
                ]
                results = [f.result(timeout=300) for f in futures]

            # Combine results
            all_fitnesses = np.zeros(pop_size, dtype=np.float32)
            for result in results:
                if result.error:
                    print(f"[ERROR] Shard {result.shard_start}-{result.shard_end}: {result.error}")
                all_fitnesses[result.shard_start:result.shard_end] = result.fitnesses

            eval_time = time.perf_counter() - eval_start

            fitnesses = jnp.array(all_fitnesses)
            total_hh_count = max(total_hh_count, gen_hh_count)

            # Update best
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            # NEAT evolution
            state = self._compiled_tell(state, fitnesses)
            generation += 1
            self._current_generation = generation

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                elapsed = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"h→h={gen_hh_count}, cppn={cppn_time:.2f}s, eval={eval_time:.2f}s, total={elapsed:.2f}s")

        # Store extended metrics
        self._extended_metrics = EMRRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=h_grid.total_positions,
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            min_fitness=float(np.min(all_fitnesses)),
            max_fitness=float(np.max(all_fitnesses)),
            std_fitness=float(np.std(all_fitnesses)),
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            'sparse_hh_count': total_hh_count,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_persistent_parallel(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Persistent worker multi-GPU with CENTRALIZED W1/W2/h→h computation + distributed eval.

        OPTIMIZED ARCHITECTURE (v2):
        - Main process (GPU 0) does ALL expensive work: CPPN queries, W1/W2, h→h discovery
        - H→H discovery uses HHCacheManager for ~80% cache hits (2-3x faster than v1)
        - Workers ONLY do evaluation (simple, fast, embarrassingly parallel)

        Architecture:
            Main Process (GPU 0)                    Workers (GPU 0, 1)
            ─────────────────────                   ─────────────────────
            1. CPPN queries (full pop)
            2. Compute W1, W2 matrices
            3. Variance masks
            4. H→H discovery WITH CACHE ←─── Cache hit: skip discovery!
            5. Split data, send to workers ───→ 6. EVAL ONLY (no CPPN, no h→h)
            8. Combine fitnesses ←─────────── 7. Return fitnesses
            9. NEAT evolution

        Benefits vs v1:
        - H→H caching: Cache hit rate ~80%, skipping expensive discovery
        - Feedforward: Workers only eval (100-200ms vs 2-5s per gen in v1)
        - H→H modes: Cache hits = eval only, cache miss = still centralized

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        from .multi_gpu_worker import (
            persistent_worker_main,
            PersistentWorkerConfig,
            GenerationTask,
            GenerationResult,
        )

        devices = jax.devices()
        num_devices = len(devices)

        print(f"[PERSISTENT_PARALLEL] Spawning {num_devices} persistent workers...")

        # Get hierarchical grid info
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions

        # Prepare coordinates for workers (numpy arrays)
        input_coords_np = np.array(jax.device_get(self._cached_input_coords))
        output_coords_np = np.array(jax.device_get(self._cached_output_coords))

        # Prepare task data (numpy arrays)
        inputs_np = np.array(jax.device_get(self._cached_inputs))
        targets_np = np.array(jax.device_get(self._cached_targets))

        # Get problem type for workers
        problem_type = getattr(problem, 'name', 'xor').lower()

        # Get population size from actual population arrays
        # This is how other run_* methods get pop_size - from the ask() result
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        print(f"[PERSISTENT_PARALLEL] Population size: {pop_size}")

        # Create algorithm config dict for workers
        # This will be used by workers to reconstruct the algorithm
        worker_algo_config = {
            'algorithm_params': {
                'emrhyperneat': {
                    'emr_hyperneat': {
                        'initial_depth': 0,
                        'max_depth': self.max_depth,
                        'variance_threshold': self.variance_threshold,
                        'band_threshold': self.band_threshold,
                        'max_weight': self.max_weight,
                        'activate_time': self.activate_time,
                        'extended_config': {
                            'enabled': self.extended_config.enabled if self.extended_config else False,
                            'allow_hidden_to_hidden': self.extended_config.allow_hidden_to_hidden if self.extended_config else False,
                            'iteration_level': self.extended_config.iteration_level if self.extended_config else 0,
                            'hh_refresh_interval': self.extended_config.hh_refresh_interval if self.extended_config else 1,
                        },
                    },
                    'substrate': {
                        'input_coords': input_coords_np.tolist(),
                        'output_coords': output_coords_np.tolist(),
                    },
                    'neat': {
                        'pop_size': pop_size,
                        'species_size': 10,
                    },
                }
            }
        }

        # Spawn persistent workers
        ctx = mp.get_context('spawn')
        task_queues = []
        result_queues = []
        processes = []
        shutdown_event = ctx.Event()

        spawn_start = time.perf_counter()

        for gpu_id in range(num_devices):
            task_q = ctx.Queue()
            result_q = ctx.Queue()

            # Enable worker-local caching for h→h modes
            has_hh = self.extended_config.allow_hidden_to_hidden if self.extended_config else False
            worker_config = PersistentWorkerConfig(
                gpu_id=gpu_id,
                algorithm_config=worker_algo_config,
                problem_type=problem_type,
                seed=42 + gpu_id,  # Different seed per worker
                max_depth=self.max_depth,
                variance_threshold=self.variance_threshold,
                band_threshold=self.band_threshold,
                max_weight=self.max_weight,
                activate_time=self.activate_time,
                allow_hidden_to_hidden=has_hh,
                iteration_level=self.extended_config.iteration_level if self.extended_config else 0,
                hh_refresh_interval=self.extended_config.hh_refresh_interval if self.extended_config else 10,
                hh_mask_change_threshold=self.extended_config.hh_mask_change_threshold if self.extended_config else 0.9,
                hh_cache_enabled=has_hh,  # Enable worker-local caching for h→h modes
                input_coords=input_coords_np,
                output_coords=output_coords_np,
                pop_size=pop_size,
                species_size=10,
                # Include static task data (sent once, not per-generation)
                inputs=inputs_np,
                targets=targets_np,
            )

            p = ctx.Process(
                target=persistent_worker_main,
                args=(worker_config, task_q, result_q, shutdown_event),
                daemon=True,
            )
            p.start()

            task_queues.append(task_q)
            result_queues.append(result_q)
            processes.append(p)

        # Wait for worker initialization (each sends empty result on success)
        print(f"[PERSISTENT_PARALLEL] Waiting for worker initialization...")
        for gpu_id in range(num_devices):
            try:
                init_result = result_queues[gpu_id].get(timeout=60)
                if init_result.error:
                    raise RuntimeError(f"Worker {gpu_id} init error: {init_result.error}")
                print(f"[PERSISTENT_PARALLEL] Worker {gpu_id} initialized successfully")
            except Exception as e:
                print(f"[PERSISTENT_PARALLEL] Worker {gpu_id} init failed: {e}")
                shutdown_event.set()
                for p in processes:
                    p.terminate()
                raise

        spawn_time = time.perf_counter() - spawn_start
        print(f"[PERSISTENT_PARALLEL] All workers initialized in {spawn_time:.2f}s")

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0
        cache_hit_count = 0
        cache_refresh_count = 0

        # Get hierarchical grid for centralized computation
        h_grid = get_hierarchical_grid(self.max_depth)
        all_positions = h_grid.all_positions
        total_positions = h_grid.total_positions

        # Get device 0 for centralized computation
        device_0 = jax.devices()[0]

        # Initialize HHCacheManager if not already done
        if self._hh_cache is None and self.extended_config is not None:
            self._hh_cache = HHCacheManager(self.extended_config)

        try:
            # Main evolution loop
            # WORKER-LOCAL CACHING APPROACH:
            # - Main process only sends CPPN genomes (small)
            # - Each worker does its own CPPN queries + W1/W2 + h→h + eval
            # - Workers cache their h→h connections locally between generations
            # - This minimizes IPC overhead and enables true parallel computation

            while best_so_far < target_fitness and generation < max_generations:
                gen_start = time.perf_counter()

                # ============================================================
                # GET CPPN POPULATION (small IPC)
                # ============================================================

                cppn_population = self._compiled_ask(state)
                pop_size = cppn_population[0].shape[0]

                # Convert to numpy for IPC - CPPN genomes are much smaller than W1/W2
                cppn_nodes_np = np.array(jax.device_get(cppn_population[0]))
                cppn_conns_np = np.array(jax.device_get(cppn_population[1]))

                cppn_time_ms = 0.0  # Workers will track their own CPPN time
                hh_time_ms = 0.0    # Workers will track their own h→h time
                total_hh_count = 0

                # ============================================================
                # DISTRIBUTE TO WORKERS (GenerationTask with CPPN genomes)
                # ============================================================

                shard_size = pop_size // num_devices

                for gpu_id in range(num_devices):
                    shard_start = gpu_id * shard_size
                    shard_end = pop_size if gpu_id == num_devices - 1 else (gpu_id + 1) * shard_size

                    # Send GenerationTask with CPPN genomes only
                    # Workers will do CPPN queries + W1/W2 + h→h (with caching) + eval
                    task = GenerationTask(
                        generation=generation,
                        shard_start=shard_start,
                        shard_end=shard_end,
                        cppn_nodes=cppn_nodes_np[shard_start:shard_end],
                        cppn_conns=cppn_conns_np[shard_start:shard_end],
                        inputs=inputs_np,  # Small for XOR (4 samples)
                        targets=targets_np,
                        # No cached h→h - workers use their own local cache
                        cached_hh_from=None,
                        cached_hh_to=None,
                        cached_hh_weights=None,
                        cached_hh_valid=None,
                    )
                    task_queues[gpu_id].put(task)

                # ============================================================
                # COLLECT RESULTS FROM WORKERS
                # ============================================================

                all_fitnesses = np.zeros(pop_size, dtype=np.float32)
                max_cppn_time = 0.0
                max_hh_time = 0.0
                max_eval_time = 0.0
                gen_hh_count = 0

                for gpu_id in range(num_devices):
                    try:
                        result = result_queues[gpu_id].get(timeout=300)

                        if result.error:
                            print(f"[PERSISTENT_PARALLEL] Worker {gpu_id} error: {result.error}")
                            continue

                        all_fitnesses[result.shard_start:result.shard_end] = result.fitnesses

                        # Track timing from workers (take max since they run in parallel)
                        max_cppn_time = max(max_cppn_time, result.cppn_time_ms or 0.0)
                        max_hh_time = max(max_hh_time, result.hh_time_ms or 0.0)
                        max_eval_time = max(max_eval_time, result.eval_time_ms or 0.0)
                        gen_hh_count += result.hh_count or 0

                    except Exception as e:
                        print(f"[PERSISTENT_PARALLEL] Failed to get result from worker {gpu_id}: {e}")

                # Update totals
                total_hh_count = gen_hh_count
                cppn_time_ms = max_cppn_time
                hh_time_ms = max_hh_time

                # Convert to JAX array for NEAT update
                fitnesses = jnp.array(all_fitnesses)

                # Update best
                gen_best = float(jnp.max(fitnesses))
                best_so_far = max(best_so_far, gen_best)

                # NEAT evolution
                state = self._compiled_tell(state, fitnesses)
                generation += 1
                self._current_generation = generation

                if collect_history:
                    history.append(gen_best)

                if self.verbose:
                    elapsed = time.perf_counter() - gen_start
                    # Workers track their own cache - hh_time≈0 means cache hit
                    cache_status = "HIT" if hh_time_ms < 10 else "MISS"
                    print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                          f"h→h={total_hh_count} ({cache_status}), cppn={cppn_time_ms:.0f}ms, "
                          f"hh_disc={hh_time_ms:.0f}ms, eval={max_eval_time:.0f}ms, total={elapsed:.2f}s",
                          flush=True)

        finally:
            # Clean shutdown
            print(f"[PERSISTENT_PARALLEL] Shutting down workers...")
            shutdown_event.set()

            # Send poison pills
            for q in task_queues:
                try:
                    q.put(None)
                except:
                    pass

            # Wait for processes
            for p in processes:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()

        # Store extended metrics
        self._extended_metrics = EMRRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=total_positions,
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            min_fitness=float(np.min(all_fitnesses)),
            max_fitness=float(np.max(all_fitnesses)),
            std_fitness=float(np.std(all_fitnesses)),
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            'sparse_hh_count': total_hh_count,
            'spawn_time_s': spawn_time,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_pmap(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Native jax.pmap multi-GPU evaluation with zero IPC overhead.

        This is the recommended multi-GPU strategy. Key design:
        - Phase 1-3: GPU 0 only (stateful): CPPN queries, h→h discovery WITH CACHE, W1/W2
        - Phase 4-5: pmap evaluation: W1/W2/h→h replicated, data sharded, true parallel
        - Phase 6: NEAT evolution on GPU 0

        Benefits:
        - ZERO IPC overhead (no Python pickle serialization)
        - H→H caching preserved (cache lives on GPU 0)
        - Expected 1.3-1.9x speedup over single GPU

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        # One-time pmap setup (shard dataset across devices)
        self._setup_pmap_evaluation()

        # Get hierarchical grid info
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords
        n_inputs = len(input_coords)
        n_outputs = len(output_coords)

        # Check if h→h is enabled
        has_hh = self.extended_config and self.extended_config.allow_hidden_to_hidden
        activate_time = self.activate_time if has_hh else 1

        # Tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0
        cache_hit_count = 0
        cache_refresh_count = 0

        # Prepare EMR-HyperNEAT arrays (reusable across generations)
        all_positions_arr = jnp.array(all_positions)
        input_coords_arr = jnp.array(input_coords)
        output_coords_arr = jnp.array(output_coords)

        # Padded sample count (includes any padding from setup)
        n_samples_padded = self._pmap_num_devices * self._pmap_samples_per_device

        if self.verbose:
            print(f"[EVAL_ONLY_PARALLEL] Starting evolution:")
            print(f"  - Devices: {self._pmap_num_devices}")
            print(f"  - Positions: {total_positions}")
            print(f"  - H→H: {'ENABLED (with cache)' if has_hh else 'DISABLED (feedforward)'}")
            print(f"  - Activate time: {activate_time}")
            print(f"  - Samples: {self._pmap_original_n_samples} original, {n_samples_padded} padded")

        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # ================================================================
            # CPPN QUERIES (GPU 0)
            # ================================================================
            cppn_start = time.perf_counter()

            # Get CPPN population and transform
            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]  # Get population size from nodes array
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Batch CPPN queries for input→positions (W1)
            # outgoing=True: input coords as source, all positions as target
            input_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, input_coords_arr, all_positions_arr,
                True, self._jitted_cppn_forward,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )  # (pop_size, n_inputs, total_positions)

            # Batch CPPN queries for positions→output (W2)
            # outgoing=False: output coords as source (but we want position→output)
            output_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, output_coords_arr, all_positions_arr,
                False, self._jitted_cppn_forward,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )  # (pop_size, n_outputs, total_positions)

            cppn_time_ms = (time.perf_counter() - cppn_start) * 1000

            # ================================================================
            # VARIANCE → MASKS (GPU 0)
            # ================================================================
            mask_start = time.perf_counter()

            # Compute hierarchical variances and subdivision masks
            # Use input[0]→all_positions as variance source (2D: pop_size, total_positions)
            all_weights_for_variance = input_all_weights[:, 0, :]  # (pop_size, total_positions)
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )

            mask_time_ms = (time.perf_counter() - mask_start) * 1000

            # ================================================================
            # H→H DISCOVERY WITH CACHE (GPU 0)
            # ================================================================
            hh_start = time.perf_counter()
            hh_from = None
            hh_to = None
            hh_weights = None
            hh_valid = None

            if has_hh:
                # Check cache
                cache_hit = False
                if self._hh_cache is not None:
                    cache_hit = not self._hh_cache.should_refresh(generation, masks_A)

                if cache_hit:
                    # Cache HIT - reuse cached h→h connections
                    cached_hh = self._hh_cache.get_cached()
                    if cached_hh is not None:
                        hh_from = cached_hh.from_indices
                        hh_to = cached_hh.to_indices
                        hh_weights = cached_hh.weights
                        hh_valid = cached_hh.valid_mask
                        total_hh_count = int(jnp.sum(hh_valid[0]))  # From first genome
                        cache_hit_count += 1
                else:
                    # Cache MISS - discover h→h connections
                    # Note: discover_sparse_hh_vectorized_multi_hop does its own CPPN queries internally
                    sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                        state=state,
                        cppns_transformed=cppns_transformed,
                        h_grid=h_grid,
                        masks_A=masks_A,
                        band_threshold=self.band_threshold,
                        max_weight=self.max_weight,
                        config=self.extended_config,
                        cppn_forward=self._jitted_cppn_forward,
                        pop_chunk_size=pop_size,
                        verbose=self.verbose,
                        device_id=0,
                        geometry_seeding_enabled=self.geometry_seeding_enabled,
                        num_cppn_outputs=self.num_cppn_outputs,
                    )

                    hh_from = sparse_hh.from_indices
                    hh_to = sparse_hh.to_indices
                    hh_weights = sparse_hh.weights
                    hh_valid = sparse_hh.valid_mask
                    total_hh_count = int(sparse_hh.num_valid[0])  # Count from first genome

                    # Update cache
                    if self._hh_cache is not None:
                        self._hh_cache.update_cache(sparse_hh, masks_A, generation)
                    cache_refresh_count += 1

            hh_time_ms = (time.perf_counter() - hh_start) * 1000

            # ================================================================
            # BUILD W1, W2 MATRICES (GPU 0)
            # ================================================================
            build_start = time.perf_counter()

            # Apply masking and build final weight matrices (same as single-GPU)
            weight_thresh = 0.1  # Same as parent
            max_weight = self.max_weight

            # Broadcast mask: (pop_size, 1, total_positions) for weight masking
            active_mask_broadcast = masks_A[:, None, :]

            # Apply tanh + max_weight scaling, then mask
            W1_raw = jnp.tanh(input_all_weights) * max_weight
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )  # (pop_size, n_inputs, total_positions)

            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W2_masked = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )
            W2 = W2_masked.transpose(0, 2, 1)  # (pop, n_outputs, total_positions) → (pop, total_positions, n_outputs)

            build_time_ms = (time.perf_counter() - build_start) * 1000

            # ================================================================
            # PMAP EVALUATION (TRUE PARALLEL)
            # ================================================================
            eval_start = time.perf_counter()

            if has_hh and hh_from is not None:
                # Hybrid mode with h→h connections
                # pmap: W1, W2, h→h replicated; inputs, targets sharded
                partial_errors = _PMAP_EVAL_HYBRID(
                    W1, W2,
                    hh_from, hh_to, hh_weights, hh_valid,
                    self._pmap_inputs_sharded,
                    self._pmap_targets_sharded,
                    activate_time,
                    total_positions,
                )  # (num_devices, pop_size)
            else:
                # Dense feedforward mode
                partial_errors = _PMAP_EVAL_DENSE(
                    W1, W2,
                    self._pmap_inputs_sharded,
                    self._pmap_targets_sharded,
                    activate_time,
                )  # (num_devices, pop_size)

            # Aggregate errors across devices
            total_errors = jnp.sum(partial_errors, axis=0)  # (pop_size,)

            # Compute fitness: 1.0 - MSE
            # Note: Use ORIGINAL sample count, not padded, for correct normalization
            n_outputs_per_sample = self._cached_targets.shape[1]
            mse = total_errors / (self._pmap_original_n_samples * n_outputs_per_sample)
            fitnesses = 1.0 - mse

            eval_time_ms = (time.perf_counter() - eval_start) * 1000

            # ================================================================
            # NEAT EVOLUTION (GPU 0)
            # ================================================================
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            state = self._compiled_tell(state, fitnesses)
            generation += 1
            self._current_generation = generation

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                elapsed = time.perf_counter() - gen_start
                cache_status = "HIT" if (has_hh and cache_hit_count > 0 and (cache_hit_count + cache_refresh_count) > 0 and cache_hit_count == generation) else ("MISS" if has_hh else "N/A")
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"h→h={total_hh_count} ({cache_status}), "
                      f"cppn={cppn_time_ms:.0f}ms, hh={hh_time_ms:.0f}ms, "
                      f"eval={eval_time_ms:.0f}ms, total={elapsed*1000:.0f}ms",
                      flush=True)

        # Store extended metrics
        self._extended_metrics = EMRRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=total_positions,
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            min_fitness=float(jnp.min(fitnesses)),
            max_fitness=float(jnp.max(fitnesses)),
            std_fitness=float(jnp.std(fitnesses)),
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            'sparse_hh_count': total_hh_count,
        }
        if collect_history:
            result['history'] = history

        return result

    def _verify_process_pool_infrastructure(self):
        """Verify that ProcessPoolExecutor with spawn method works correctly.

        This tests the basic infrastructure needed for true parallel GPU execution:
        1. Spawn method creates fresh Python processes
        2. Each process can set CUDA_VISIBLE_DEVICES independently
        3. Each process has isolated JAX runtime

        Called once to confirm the approach is viable before full implementation.
        """
        devices = jax.devices()
        num_devices = len(devices)

        if num_devices < 2:
            print("  [Verify] Skipping - only 1 device available")
            return

        print(f"  [Verify] Testing ProcessPoolExecutor with {num_devices} GPUs...")

        # Test function that runs in worker process
        def test_worker(device_id: int) -> Dict[str, Any]:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

            import jax
            import jax.numpy as jnp

            devices = jax.devices()
            backend = jax.default_backend()

            # Simple computation to verify GPU access
            @jax.jit
            def simple_compute():
                return jnp.sum(jnp.ones((100, 100)))

            result = float(simple_compute())

            return {
                'device_id': device_id,
                'pid': os.getpid(),
                'jax_devices': [str(d) for d in devices],
                'backend': backend,
                'result': result,
            }

        # Use spawn to get fresh processes
        ctx = mp.get_context('spawn')

        try:
            with ProcessPoolExecutor(max_workers=num_devices, mp_context=ctx) as executor:
                futures = [executor.submit(test_worker, d) for d in range(num_devices)]
                results = [f.result(timeout=30) for f in futures]

            # Verify results
            pids = [r['pid'] for r in results]
            all_separate = len(set(pids)) == num_devices
            all_single_gpu = all(len(r['jax_devices']) == 1 for r in results)

            if all_separate and all_single_gpu:
                print(f"  [Verify] ✓ ProcessPoolExecutor working correctly")
                print(f"  [Verify]   - All {num_devices} workers have separate PIDs")
                print(f"  [Verify]   - Each worker sees exactly 1 GPU")
            else:
                print(f"  [Verify] ⚠ ProcessPoolExecutor issues detected")
                if not all_separate:
                    print(f"  [Verify]   - Workers sharing PIDs: {pids}")
                if not all_single_gpu:
                    print(f"  [Verify]   - Workers see multiple GPUs")

        except Exception as e:
            print(f"  [Verify] ✗ ProcessPoolExecutor failed: {e}")

    def _run_until_threshold_data_parallel_extended(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Data-parallel multi-GPU implementation WITH h→h discovery.

        Pipeline per generation:
        1. CPPN ask + transform (GPU 0)
        2. CPPN queries for input→all and output←all (GPU 0)
        3. Variance/subdivision masks (GPU 0)
        4. H→H discovery with multi-hop expansion (GPU 0)
        5. Build W1, W2 matrices (GPU 0)
        6. Data-parallel evaluation (SHARD dataset, replicate weights+h→h)
        7. NEAT evolution (GPU 0)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Pad samples to be divisible by num_devices
        remainder = n_samples % num_devices
        if remainder != 0:
            pad_samples = num_devices - remainder
            inputs_padded = jnp.pad(inputs_batch, ((0, pad_samples), (0, 0)), mode='edge')
            targets_padded = jnp.pad(targets_batch, ((0, pad_samples), (0, 0)), mode='edge')
            padded_n_samples = n_samples + pad_samples
        else:
            pad_samples = 0
            inputs_padded = inputs_batch
            targets_padded = targets_batch
            padded_n_samples = n_samples

        per_gpu_samples = padded_n_samples // num_devices

        # Shard dataset: (n_samples, features) -> (num_gpus, per_gpu_samples, features)
        inputs_sharded = inputs_padded.reshape(num_devices, per_gpu_samples, -1)
        targets_sharded = targets_padded.reshape(num_devices, per_gpu_samples, -1)

        if self.verbose:
            print(f"[UnifiedExtended] Data-parallel: {num_devices} GPUs, "
                  f"positions={total_positions}, samples={n_samples} ({per_gpu_samples}/GPU)")

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # === STEP 0: CPPN ask + transform ===
            if self.extra_randkey_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # === STEP 1: CPPN queries ===
            # Note: This runs on GPU 0 only (data sharding is in evaluation)
            input_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, input_coords, all_positions,
                True, self._jitted_cppn_forward,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )
            output_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, output_coords, all_positions,
                False, self._jitted_cppn_forward,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )

            # === STEP 2: Variance + subdivision masks ===
            all_weights_for_variance = input_all_weights[:, 0, :]
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False
            )

            # === STEP 3: H→H discovery (GPU 0, result shared) ===
            sparse_hh = None
            if self.extended_config is not None and self.extended_config.allow_hidden_to_hidden:
                # Check cache
                if self._hh_cache is not None and not self._hh_cache.should_refresh(
                    self._current_generation, masks_A
                ):
                    sparse_hh = self._hh_cache.get_cached()
                else:
                    # Note: This runs on GPU 0 only
                    sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                        state, cppns_transformed, h_grid, masks_A,
                        self.band_threshold, self.max_weight,
                        self.extended_config, self._jitted_cppn_forward,
                        verbose=False,
                        device_id=0,
                        geometry_seeding_enabled=self.geometry_seeding_enabled,
                        num_cppn_outputs=self.num_cppn_outputs,
                    )
                    if self._hh_cache is not None:
                        self._hh_cache.update_cache(sparse_hh, masks_A, self._current_generation)

            # === STEP 4: Build W1, W2 matrices ===
            weight_thresh = 0.1
            max_weight = self.max_weight
            active_mask_broadcast = masks_A[:, None, :]

            W1_raw = jnp.tanh(input_all_weights) * max_weight
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw, 0.0
            )

            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W2 = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw, 0.0
            )
            W2 = W2.transpose(0, 2, 1)  # (pop, total_positions, num_outputs)

            # === STEP 5: Data-parallel evaluation ===
            if sparse_hh is not None and self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH:
                # Hybrid evaluation with sparse h→h
                fitnesses = self._eval_data_parallel_hybrid(
                    W1, W2, sparse_hh, inputs_sharded, targets_sharded,
                    num_devices, per_gpu_samples, n_samples, pad_samples
                )
            else:
                # Dense-only evaluation
                fitnesses = self._eval_data_parallel_dense(
                    W1, W2, inputs_sharded, targets_sharded,
                    num_devices, per_gpu_samples, n_samples, pad_samples
                )

            # Ensure fitnesses are on primary device and fully materialized
            # This prevents device sharding issues from pmap affecting _compiled_tell
            fitnesses = jax.device_put(fitnesses, jax.devices()[0])
            fitnesses = jax.block_until_ready(fitnesses)

            # === STEP 6: NEAT evolution ===
            state = self._compiled_tell(state, fitnesses)
            self._current_generation += 1

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                gen_time = time.perf_counter() - gen_start
                hh_count = int(jnp.sum(sparse_hh.num_valid)) if sparse_hh else 0
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"h→h={hh_count}, time={gen_time:.3f}s")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def _eval_data_parallel_hybrid(
        self,
        W1: jnp.ndarray,
        W2: jnp.ndarray,
        sparse_hh: SparseHiddenConnections,
        inputs_sharded: jnp.ndarray,
        targets_sharded: jnp.ndarray,
        num_devices: int,
        per_gpu_samples: int,
        n_samples: int,
        pad_samples: int,
    ) -> jnp.ndarray:
        """Data-parallel evaluation with hybrid forward pass (sparse h→h).

        Uses jax.pmap to distribute evaluation across multiple GPUs.
        Dataset is sharded (each GPU gets different samples).
        Weights and h→h connections are replicated.

        Args:
            W1: (pop, num_inputs, total_positions)
            W2: (pop, total_positions, num_outputs)
            sparse_hh: Sparse h→h connections
            inputs_sharded: (num_gpus, per_gpu_samples, num_inputs)
            targets_sharded: (num_gpus, per_gpu_samples, num_outputs)
            num_devices: Number of GPUs
            per_gpu_samples: Samples per GPU
            n_samples: Original sample count (before padding)
            pad_samples: Number of padded samples

        Returns:
            Fitness array (pop_size,)
        """
        activate_time = self.activate_time

        # Use module-level pmap function for multi-GPU evaluation
        # partial_error_sums has shape: (num_gpus, pop_size)
        partial_error_sums = _pmap_eval_hybrid(
            W1, W2,
            sparse_hh.from_indices, sparse_hh.to_indices,
            sparse_hh.weights, sparse_hh.valid_mask,
            inputs_sharded, targets_sharded,
            activate_time,
        )

        # Sum errors across all GPUs, then average across samples
        total_error_sums = jnp.sum(partial_error_sums, axis=0)  # (pop_size,)
        avg_errors = total_error_sums / n_samples
        fitnesses = jnp.maximum(0.0, 1.0 - avg_errors)
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        return fitnesses

    def _eval_data_parallel_dense(
        self,
        W1: jnp.ndarray,
        W2: jnp.ndarray,
        inputs_sharded: jnp.ndarray,
        targets_sharded: jnp.ndarray,
        num_devices: int,
        per_gpu_samples: int,
        n_samples: int,
        pad_samples: int,
    ) -> jnp.ndarray:
        """Data-parallel evaluation with dense-only forward pass.

        Uses jax.pmap to distribute evaluation across multiple GPUs.

        Args:
            W1: (pop, num_inputs, total_positions)
            W2: (pop, total_positions, num_outputs)
            inputs_sharded: (num_gpus, per_gpu_samples, num_inputs)
            targets_sharded: (num_gpus, per_gpu_samples, num_outputs)
            num_devices: Number of GPUs
            per_gpu_samples: Samples per GPU
            n_samples: Original sample count
            pad_samples: Number of padded samples

        Returns:
            Fitness array (pop_size,)
        """
        # Use module-level pmap function for multi-GPU evaluation
        # partial_error_sums has shape: (num_gpus, pop_size)
        partial_error_sums = _pmap_eval_dense(
            W1, W2,
            inputs_sharded, targets_sharded,
        )

        # Sum errors across all GPUs, then average across samples
        total_error_sums = jnp.sum(partial_error_sums, axis=0)  # (pop_size,)
        avg_errors = total_error_sums / n_samples
        fitnesses = jnp.maximum(0.0, 1.0 - avg_errors)
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        return fitnesses

    def _run_until_threshold_data_parallel_full(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Data-parallel multi-GPU with FULL pipeline inside pmap.

        This strategy moves ALL computation (CPPN queries, variance masks, h→h discovery,
        weight building, AND evaluation) inside pmap for balanced memory distribution.

        Memory distribution: Each GPU processes the full population on its data shard.

        Pipeline per generation:
        1. CPPN ask + transform (CPU/GPU 0)
        2. INSIDE PMAP (each GPU):
           - CPPN queries for input→all and output←all
           - Variance/subdivision masks
           - H→H discovery with multi-hop expansion
           - Build W1, W2 matrices
           - Evaluate on data shard
        3. Aggregate fitnesses (mean across GPUs)
        4. NEAT evolution (CPU/GPU 0)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Extract static parameters from h_grid for pmap (JIT-compatible Python tuples)
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Pad samples to be divisible by num_devices
        remainder = n_samples % num_devices
        if remainder != 0:
            pad_samples = num_devices - remainder
            inputs_padded = jnp.pad(inputs_batch, ((0, pad_samples), (0, 0)), mode='edge')
            targets_padded = jnp.pad(targets_batch, ((0, pad_samples), (0, 0)), mode='edge')
            padded_n_samples = n_samples + pad_samples
        else:
            pad_samples = 0
            inputs_padded = inputs_batch
            targets_padded = targets_batch
            padded_n_samples = n_samples

        per_gpu_samples = padded_n_samples // num_devices

        # Shard dataset: (n_samples, features) -> (num_gpus, per_gpu_samples, features)
        inputs_sharded = inputs_padded.reshape(num_devices, per_gpu_samples, -1)
        targets_sharded = targets_padded.reshape(num_devices, per_gpu_samples, -1)

        # Extended config parameters
        allow_hh = self.extended_config.allow_hidden_to_hidden if self.extended_config else False
        iteration_level = self.extended_config.iteration_level if self.extended_config else 0
        max_sparse_conns = self.extended_config.max_sparse_conns if self.extended_config else 1000

        if self.verbose:
            print(f"[UnifiedExtended] FULL_PIPELINE_PARALLEL: {num_devices} GPUs, "
                  f"positions={total_positions}, samples={n_samples} ({per_gpu_samples}/GPU)")
            print(f"  h→h enabled: {allow_hh}, iteration_level: {iteration_level}")

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0  # Track h→h connections across generations

        # Track substrate efficiency metrics across generations (for averaging)
        cumulative_active = 0.0
        cumulative_w1_nonzero = 0.0
        cumulative_w2_nonzero = 0.0
        cumulative_w1_total = 0.0
        cumulative_w2_total = 0.0
        generation_count = 0

        # Cache tracking
        cache_hit_count = 0
        cache_refresh_count = 0
        last_hh_count = -1  # -1 so first gen is always a "refresh"

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # === STEP 0: CPPN ask + transform (on GPU 0) ===
            if self.extra_randkey_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # === STEP 1-5: Full pipeline inside pmap ===
            # Each GPU runs CPPN queries, variance, h→h, weight building, AND evaluation
            # on its data shard, with full population replicated
            (partial_fitnesses, hh_counts_sharded,
             active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_data_parallel_full_hh(
                cppns_transformed,           # None: replicated
                all_positions,               # None: replicated
                input_coords,                # None: replicated
                output_coords,               # None: replicated
                inputs_sharded,              # 0: sharded
                targets_sharded,             # 0: sharded
                self.variance_threshold,     # None: replicated
                self.band_threshold,         # None: replicated
                self.max_weight,             # None: replicated
                0.1,                         # weight_thresh - None: replicated
                self.activate_time,          # STATIC
                level_sizes,                 # STATIC
                level_offsets,               # STATIC
                level_grid_sizes,            # STATIC
                parent_indices,              # STATIC
                num_levels,                  # STATIC
                total_positions,             # STATIC
                max_sparse_conns,            # STATIC
                iteration_level,             # STATIC
                allow_hh,                    # STATIC
                state,                       # None: replicated
                self._jitted_cppn_forward,   # STATIC
                self.geometry_seeding_enabled,  # STATIC - Risi & Stanley 2012
            )

            # Aggregate: average fitness across data shards
            # Each GPU computed fitness on different samples, so average them
            fitnesses = jnp.mean(partial_fitnesses, axis=0)  # (pop_size,)

            # Aggregate h→h counts (same population on each GPU, so take first)
            # Each GPU computes h→h for the full population, they should be identical
            gen_hh_count = int(hh_counts_sharded[0])  # Take from first GPU
            total_hh_count = max(total_hh_count, gen_hh_count)  # Track max across generations

            # Aggregate substrate efficiency metrics (same population, take from first GPU)
            cumulative_active += float(active_sums[0])
            cumulative_w1_nonzero += float(w1_nz_sums[0])
            cumulative_w2_nonzero += float(w2_nz_sums[0])
            cumulative_w1_total += float(w1_totals[0])
            cumulative_w2_total += float(w2_totals[0])
            generation_count += 1

            # Track cache stats (only for modes with h→h discovery)
            if gen_hh_count > 0:
                if gen_hh_count == last_hh_count:
                    cache_hit_count += 1
                else:
                    cache_refresh_count += 1
                last_hh_count = gen_hh_count

            # Ensure fitnesses are on primary device and fully materialized
            # This prevents device sharding issues from pmap affecting _compiled_tell
            fitnesses = jax.device_put(fitnesses, jax.devices()[0])
            fitnesses = jax.block_until_ready(fitnesses)

            # === STEP 6: NEAT evolution ===
            state = self._compiled_tell(state, fitnesses)
            self._current_generation += 1

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={gen_time:.3f}s")

        # Compute average substrate efficiency metrics
        pop_size = cppns_transformed[0].shape[0]
        if generation_count > 0:
            avg_active = cumulative_active / generation_count / pop_size
            position_util = (cumulative_active / generation_count) / total_positions / pop_size
            avg_w1_density = cumulative_w1_nonzero / cumulative_w1_total if cumulative_w1_total > 0 else 0.0
            avg_w2_density = cumulative_w2_nonzero / cumulative_w2_total if cumulative_w2_total > 0 else 0.0
        else:
            avg_active = 0.0
            position_util = 0.0
            avg_w1_density = 0.0
            avg_w2_density = 0.0

        # Store extended metrics including h→h count and substrate efficiency
        self._extended_metrics = EMRRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=total_positions,
            # Config echo (what was actually used)
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            # Population diversity from final fitnesses
            min_fitness=float(jnp.min(fitnesses)),
            max_fitness=float(jnp.max(fitnesses)),
            std_fitness=float(jnp.std(fitnesses)),
            # Substrate efficiency (averaged over generations)
            active_positions_mean=avg_active,
            position_utilization=position_util,
            w1_density=avg_w1_density,
            w2_density=avg_w2_density,
            # Cache stats
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_pop_parallel(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Population-parallel multi-GPU with FULL pipeline inside pmap.

        This strategy shards the POPULATION across GPUs, replicating the full dataset
        on each GPU. Good for large populations with small datasets.

        Memory distribution: Each GPU processes a population slice on the full dataset.

        Pipeline per generation:
        1. CPPN ask + transform (CPU/GPU 0)
        2. Shard population across GPUs
        3. INSIDE PMAP (each GPU):
           - CPPN queries for its pop slice
           - Variance/subdivision masks for pop slice
           - H→H discovery for pop slice
           - Build W1, W2 for pop slice
           - Evaluate pop slice on full dataset
        4. Gather fitnesses (concatenate from all GPUs)
        5. NEAT evolution (CPU/GPU 0)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Extract static parameters from h_grid for pmap (JIT-compatible Python tuples)
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data - full dataset on each GPU
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Get population size to determine padding
        # We need to handle this dynamically since pop_size comes from state
        # Will compute padding inside the loop

        # Extended config parameters
        allow_hh = self.extended_config.allow_hidden_to_hidden if self.extended_config else False
        iteration_level = self.extended_config.iteration_level if self.extended_config else 0
        max_sparse_conns = self.extended_config.max_sparse_conns if self.extended_config else 1000

        if self.verbose:
            print(f"[UnifiedExtended] POPULATION_PARALLEL_SEQUENTIAL: {num_devices} GPUs, "
                  f"positions={total_positions}, samples={n_samples}")
            print(f"  h→h enabled: {allow_hh}, iteration_level: {iteration_level}")

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0  # Track h→h connections across generations

        # Track substrate efficiency metrics across generations (for averaging)
        cumulative_active = 0.0
        cumulative_w1_nonzero = 0.0
        cumulative_w2_nonzero = 0.0
        cumulative_w1_total = 0.0
        cumulative_w2_total = 0.0
        generation_count = 0

        # Cache tracking
        cache_hit_count = 0
        cache_refresh_count = 0
        last_hh_count = -1  # -1 so first gen is always a "refresh"

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # === STEP 0: CPPN ask + transform (on GPU 0) ===
            if self.extra_randkey_split:
                randkey_, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # === STEP 1: Calculate chunk size for large substrates ===
            # For large total_positions, we need to process population in smaller chunks
            # to avoid OOM. Similar logic to single-GPU eval_chunk_size.
            eval_chunk_size = getattr(self.extended_config, 'eval_chunk_size', None) if self.extended_config else None

            # Auto-compute chunk size based on position count AND h→h mode
            # h→h discovery requires MUCH smaller chunks because it creates (chunk, positions, positions) arrays
            if eval_chunk_size == -1 or eval_chunk_size is None:
                if total_positions < 1000:        # depth <= 3
                    pop_chunk_size = None  # No chunking needed
                elif total_positions < 6000:      # depth 4-5
                    if allow_hh:
                        pop_chunk_size = 50   # h→h needs smaller chunks
                    else:
                        pop_chunk_size = 200  # feedforward can use larger chunks
                elif total_positions < 50000:     # depth 6 "anomaly zone"
                    if allow_hh:
                        pop_chunk_size = 10   # h→h needs very small chunks at depth 6
                    else:
                        pop_chunk_size = 100  # feedforward: 50 per GPU with 2 GPUs
                elif total_positions < 200000:    # depth 7
                    if allow_hh:
                        # CRITICAL: At depth 7+ with h→h, use 1 genome per GPU to avoid XLA trace explosion
                        # Each pmap call traces only 1 genome worth of fori_loop (175 chunks)
                        # Python for-loop iterates over population chunks (NOT traced)
                        pop_chunk_size = num_devices  # 1 per GPU
                    else:
                        pop_chunk_size = 50   # feedforward: 25 per GPU
                else:                             # depth 8+
                    if allow_hh:
                        pop_chunk_size = num_devices  # h→h: 1 per GPU
                    else:
                        pop_chunk_size = 20   # feedforward: 10 per GPU
            else:
                pop_chunk_size = eval_chunk_size

            # Ensure chunk size is divisible by num_devices
            if pop_chunk_size is not None:
                pop_chunk_size = max(num_devices, (pop_chunk_size // num_devices) * num_devices)

            # Determine if we need population chunking
            need_chunking = pop_chunk_size is not None and pop_chunk_size > 0 and pop_size > pop_chunk_size

            # NOTE: Sequential pmap with jax.lax.scan doesn't work because scan is still
            # traced by XLA just like fori_loop. The solution is to use very small population
            # chunks (1 genome per GPU) with the standard pmap, so each pmap call only traces
            # 1 genome's worth of computation. The Python for-loop over chunks is NOT traced.
            use_sequential_pmap = False  # Disabled - use tiny chunks instead

            if self.verbose and generation == 0:
                if need_chunking:
                    num_chunks = (pop_size + pop_chunk_size - 1) // pop_chunk_size
                    print(f"  Population: {pop_size}, using chunked processing: {num_chunks} chunks of {pop_chunk_size}")
                else:
                    print(f"  Population: {pop_size} (processing all at once)")
                if use_sequential_pmap:
                    print(f"  Using SEQUENTIAL multi-GPU pmap for h→h at depth {self.max_depth} (positions={total_positions})")

            # Lists to accumulate results across chunks
            all_fitness_chunks = []
            all_hh_counts = []
            all_active_sums = []
            all_w1_nz_sums = []
            all_w2_nz_sums = []
            all_w1_totals = []
            all_w2_totals = []

            # === STEP 2-5: Process population in chunks (or all at once if small) ===
            if need_chunking:
                # Process population in chunks
                for chunk_start in range(0, pop_size, pop_chunk_size):
                    chunk_end = min(chunk_start + pop_chunk_size, pop_size)
                    chunk_size_actual = chunk_end - chunk_start

                    # Extract chunk from cppns_transformed
                    cppns_chunk = tuple(arr[chunk_start:chunk_end] for arr in cppns_transformed)

                    # Pad chunk to be divisible by num_devices
                    remainder = chunk_size_actual % num_devices
                    if remainder != 0:
                        pad_size = num_devices - remainder
                        padded_chunk_size = chunk_size_actual + pad_size
                    else:
                        pad_size = 0
                        padded_chunk_size = chunk_size_actual

                    per_gpu_pop = padded_chunk_size // num_devices

                    # Pad and reshape each CPPN array in chunk
                    cppns_sharded = []
                    for arr in cppns_chunk:
                        if pad_size > 0:
                            pad_width = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
                            arr_padded = jnp.pad(arr, pad_width, mode='edge')
                        else:
                            arr_padded = arr
                        new_shape = (num_devices, per_gpu_pop) + arr.shape[1:]
                        arr_sharded = arr_padded.reshape(new_shape)
                        cppns_sharded.append(arr_sharded)
                    cppns_sharded = tuple(cppns_sharded)

                    # Run pmap on this chunk
                    # Use sequential pmap for h→h at large depths to avoid XLA trace explosion
                    if use_sequential_pmap:
                        (fitnesses_sharded, hh_counts_sharded,
                         active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_sequential_pop_parallel_hh(
                            cppns_sharded,
                            all_positions,
                            input_coords,
                            output_coords,
                            inputs_batch,
                            targets_batch,
                            self.variance_threshold,
                            self.band_threshold,
                            self.max_weight,
                            0.1,
                            self.activate_time,
                            level_sizes,
                            level_offsets,
                            level_grid_sizes,
                            parent_indices,
                            num_levels,
                            total_positions,
                            max_sparse_conns,
                            iteration_level,
                            allow_hh,
                            state,
                            self._jitted_cppn_forward,
                            self.geometry_seeding_enabled,
                        )
                    else:
                        (fitnesses_sharded, hh_counts_sharded,
                         active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_pop_parallel_hh(
                            cppns_sharded,
                            all_positions,
                            input_coords,
                            output_coords,
                            inputs_batch,
                            targets_batch,
                            self.variance_threshold,
                            self.band_threshold,
                            self.max_weight,
                            0.1,
                            self.activate_time,
                            level_sizes,
                            level_offsets,
                            level_grid_sizes,
                            parent_indices,
                            num_levels,
                            total_positions,
                            max_sparse_conns,
                            iteration_level,
                            allow_hh,
                            state,
                            self._jitted_cppn_forward,
                            self.geometry_seeding_enabled,
                        )

                    # Extract fitnesses for this chunk (remove padding)
                    chunk_fitnesses = fitnesses_sharded.reshape(-1)[:chunk_size_actual]
                    all_fitness_chunks.append(chunk_fitnesses)

                    # Accumulate metrics
                    all_hh_counts.append(float(jnp.sum(hh_counts_sharded)))
                    all_active_sums.append(float(jnp.sum(active_sums)))
                    all_w1_nz_sums.append(float(jnp.sum(w1_nz_sums)))
                    all_w2_nz_sums.append(float(jnp.sum(w2_nz_sums)))
                    all_w1_totals.append(float(jnp.sum(w1_totals)))
                    all_w2_totals.append(float(jnp.sum(w2_totals)))

                # Concatenate all chunks
                fitnesses = jnp.concatenate(all_fitness_chunks)
                gen_hh_count = int(sum(all_hh_counts))

                # Aggregate substrate efficiency metrics
                cumulative_active += sum(all_active_sums)
                cumulative_w1_nonzero += sum(all_w1_nz_sums)
                cumulative_w2_nonzero += sum(all_w2_nz_sums)
                cumulative_w1_total += sum(all_w1_totals)
                cumulative_w2_total += sum(all_w2_totals)
            else:
                # Original path: process all at once (no chunking needed)
                # Pad population to be divisible by num_devices
                remainder = pop_size % num_devices
                if remainder != 0:
                    pad_size = num_devices - remainder
                    padded_pop_size = pop_size + pad_size
                else:
                    pad_size = 0
                    padded_pop_size = pop_size

                per_gpu_pop = padded_pop_size // num_devices

                # Pad and reshape each CPPN array
                cppns_sharded = []
                for arr in cppns_transformed:
                    if pad_size > 0:
                        pad_width = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
                        arr_padded = jnp.pad(arr, pad_width, mode='edge')
                    else:
                        arr_padded = arr
                    new_shape = (num_devices, per_gpu_pop) + arr.shape[1:]
                    arr_sharded = arr_padded.reshape(new_shape)
                    cppns_sharded.append(arr_sharded)
                cppns_sharded = tuple(cppns_sharded)

                # Run pmap on full population
                # Use sequential pmap for h→h at large depths to avoid XLA trace explosion
                if use_sequential_pmap:
                    (fitnesses_sharded, hh_counts_sharded,
                     active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_sequential_pop_parallel_hh(
                        cppns_sharded,
                        all_positions,
                        input_coords,
                        output_coords,
                        inputs_batch,
                        targets_batch,
                        self.variance_threshold,
                        self.band_threshold,
                        self.max_weight,
                        0.1,
                        self.activate_time,
                        level_sizes,
                        level_offsets,
                        level_grid_sizes,
                        parent_indices,
                        num_levels,
                        total_positions,
                        max_sparse_conns,
                        iteration_level,
                        allow_hh,
                        state,
                        self._jitted_cppn_forward,
                        self.geometry_seeding_enabled,
                    )
                else:
                    (fitnesses_sharded, hh_counts_sharded,
                     active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_pop_parallel_hh(
                        cppns_sharded,
                        all_positions,
                        input_coords,
                        output_coords,
                        inputs_batch,
                        targets_batch,
                        self.variance_threshold,
                        self.band_threshold,
                        self.max_weight,
                        0.1,
                        self.activate_time,
                        level_sizes,
                        level_offsets,
                        level_grid_sizes,
                        parent_indices,
                        num_levels,
                        total_positions,
                        max_sparse_conns,
                        iteration_level,
                        allow_hh,
                        state,
                        self._jitted_cppn_forward,
                        self.geometry_seeding_enabled,
                    )

                # Gather: concatenate fitnesses and remove padding
                fitnesses = fitnesses_sharded.reshape(-1)[:pop_size]
                gen_hh_count = int(jnp.sum(hh_counts_sharded))

                # Aggregate substrate efficiency metrics
                cumulative_active += float(jnp.sum(active_sums))
                cumulative_w1_nonzero += float(jnp.sum(w1_nz_sums))
                cumulative_w2_nonzero += float(jnp.sum(w2_nz_sums))
                cumulative_w1_total += float(jnp.sum(w1_totals))
                cumulative_w2_total += float(jnp.sum(w2_totals))

            # Track max h→h across generations
            total_hh_count = max(total_hh_count, gen_hh_count)
            generation_count += 1

            # Track cache stats (only for modes with h→h discovery)
            if gen_hh_count > 0:
                if gen_hh_count == last_hh_count:
                    cache_hit_count += 1
                else:
                    cache_refresh_count += 1
                last_hh_count = gen_hh_count

            # Ensure fitnesses are on primary device and fully materialized
            # This prevents device sharding issues from pmap affecting _compiled_tell
            fitnesses = jax.device_put(fitnesses, jax.devices()[0])
            fitnesses = jax.block_until_ready(fitnesses)

            # === STEP 6: NEAT evolution ===
            state = self._compiled_tell(state, fitnesses)
            self._current_generation += 1

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={gen_time:.3f}s")

        # Compute average substrate efficiency metrics
        if generation_count > 0:
            avg_active = cumulative_active / generation_count / pop_size
            position_util = (cumulative_active / generation_count) / total_positions / pop_size
            avg_w1_density = cumulative_w1_nonzero / cumulative_w1_total if cumulative_w1_total > 0 else 0.0
            avg_w2_density = cumulative_w2_nonzero / cumulative_w2_total if cumulative_w2_total > 0 else 0.0
        else:
            avg_active = 0.0
            position_util = 0.0
            avg_w1_density = 0.0
            avg_w2_density = 0.0

        # Store extended metrics including h→h count and substrate efficiency
        self._extended_metrics = EMRRecurrenceMetrics(
            hidden_to_hidden_connections=total_hh_count,
            total_positions=total_positions,
            # Config echo (what was actually used)
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            # Population diversity from final fitnesses
            min_fitness=float(jnp.min(fitnesses)),
            max_fitness=float(jnp.max(fitnesses)),
            std_fitness=float(jnp.std(fitnesses)),
            # Substrate efficiency (averaged over generations)
            active_positions_mean=avg_active,
            position_utilization=position_util,
            w1_density=avg_w1_density,
            w2_density=avg_w2_density,
            # Cache stats
            cache_hit_count=cache_hit_count,
            cache_refresh_count=cache_refresh_count,
        )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
        }
        if collect_history:
            result['history'] = history

        return result


# ============================================================================
# Module-Level pmap Functions for Multi-GPU Evaluation
# ============================================================================
# These are defined at module level (outside any class) to ensure JAX
# compiles them ONCE and reuses across calls. Defining inside methods would
# cause recompilation each call due to new function objects.


def _eval_hybrid_on_shard_core(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    inputs_shard: jnp.ndarray,
    targets_shard: jnp.ndarray,
    activate_time: int,
) -> jnp.ndarray:
    """Core evaluation logic for hybrid forward pass on a data shard.

    This function runs on a single GPU with its portion of the dataset.

    Args:
        W1: (pop, num_inputs, total_positions) - replicated
        W2: (pop, total_positions, num_outputs) - replicated
        from_indices: (pop, max_sparse_conns) - replicated
        to_indices: (pop, max_sparse_conns) - replicated
        hh_weights: (pop, max_sparse_conns) - replicated
        valid_mask: (pop, max_sparse_conns) - replicated
        inputs_shard: (per_gpu_samples, num_inputs) - this GPU's data
        targets_shard: (per_gpu_samples, num_outputs) - this GPU's data
        activate_time: Number of forward pass iterations

    Returns:
        (pop_size,) error sums for this shard
    """
    def eval_single_genome(w1, w2, fi, ti, hw, vm):
        """Evaluate single genome on the data shard."""
        outputs = forward_hybrid_vmapped(
            inputs_shard, w1, w2, fi, ti, hw, vm, activate_time
        )
        errors = jnp.mean((outputs - targets_shard) ** 2, axis=1)
        return jnp.sum(errors)

    # vmap over population
    error_sums = jax.vmap(eval_single_genome)(
        W1, W2, from_indices, to_indices, hh_weights, valid_mask
    )
    return error_sums  # (pop_size,)


def _eval_dense_on_shard_core(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    inputs_shard: jnp.ndarray,
    targets_shard: jnp.ndarray,
) -> jnp.ndarray:
    """Core evaluation logic for dense forward pass on a data shard.

    Args:
        W1: (pop, num_inputs, total_positions) - replicated
        W2: (pop, total_positions, num_outputs) - replicated
        inputs_shard: (per_gpu_samples, num_inputs) - this GPU's data
        targets_shard: (per_gpu_samples, num_outputs) - this GPU's data

    Returns:
        (pop_size,) error sums for this shard
    """
    def eval_single_genome(w1, w2):
        """Evaluate single genome on the data shard."""
        h = jnp.tanh(inputs_shard @ w1)
        outputs = jax.nn.sigmoid(h @ w2)
        errors = jnp.mean((outputs - targets_shard) ** 2, axis=1)
        return jnp.sum(errors)

    # vmap over population
    error_sums = jax.vmap(eval_single_genome)(W1, W2)
    return error_sums  # (pop_size,)


# pmap-wrapped versions for multi-GPU execution
# in_axes specifies which axis to shard on:
# - None: replicate (same data on all GPUs)
# - 0: shard along axis 0 (each GPU gets a slice)

@functools.partial(
    jax.pmap,
    in_axes=(None, None, None, None, None, None, 0, 0, None),  # 9 args
    static_broadcasted_argnums=(8,),  # activate_time is static
)
def _pmap_eval_hybrid(
    W1: jnp.ndarray,           # replicated
    W2: jnp.ndarray,           # replicated
    from_indices: jnp.ndarray, # replicated
    to_indices: jnp.ndarray,   # replicated
    hh_weights: jnp.ndarray,   # replicated
    valid_mask: jnp.ndarray,   # replicated
    inputs_shard: jnp.ndarray, # sharded (axis 0)
    targets_shard: jnp.ndarray, # sharded (axis 0)
    activate_time: int,        # static
) -> jnp.ndarray:
    """pmap-wrapped hybrid evaluation - runs in parallel on all GPUs."""
    return _eval_hybrid_on_shard_core(
        W1, W2, from_indices, to_indices, hh_weights, valid_mask,
        inputs_shard, targets_shard, activate_time
    )


@functools.partial(
    jax.pmap,
    in_axes=(None, None, 0, 0),  # 4 args: W1, W2 replicated; inputs, targets sharded
)
def _pmap_eval_dense(
    W1: jnp.ndarray,           # replicated
    W2: jnp.ndarray,           # replicated
    inputs_shard: jnp.ndarray, # sharded (axis 0)
    targets_shard: jnp.ndarray, # sharded (axis 0)
) -> jnp.ndarray:
    """pmap-wrapped dense evaluation - runs in parallel on all GPUs."""
    return _eval_dense_on_shard_core(W1, W2, inputs_shard, targets_shard)


# ============================================================================
# Full Pipeline with H→H Discovery (for Multi-GPU strategies)
# ============================================================================
# These functions run the complete pipeline (CPPN queries + variance + h→h + eval)
# inside pmap for balanced memory distribution across GPUs.


def _full_pipeline_with_hh_core(
    cppns_transformed: Tuple,
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    variance_threshold: float,
    band_threshold: float,
    max_weight: float,
    weight_thresh: float,
    activate_time: int,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[int, ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    max_sparse_conns: int,
    iteration_level: int,
    allow_hh: bool,
    state: Any,
    cppn_forward: Any,
    geometry_seeding_enabled: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full pipeline with h→h discovery - runs on single GPU inside pmap.

    This function replicates the parent class's _full_pipeline_single_gpu logic
    but adds Phase 2 h→h discovery with multi-hop expansion.

    Args:
        cppns_transformed: Transformed CPPNs for this GPU
        all_positions: Hierarchical grid positions
        input_coords: Input coordinates
        output_coords: Output coordinates
        inputs_batch: Problem inputs (sharded or replicated depending on strategy)
        targets_batch: Problem targets
        variance_threshold: Subdivision threshold
        band_threshold: H→H threshold
        max_weight: Weight scaling
        weight_thresh: Minimum weight threshold
        activate_time: Forward pass iterations
        level_sizes: Tuple of cell counts per level
        level_offsets: Tuple of cumulative offsets
        level_grid_sizes: Tuple of grid dimensions
        parent_indices: Parent index arrays
        num_levels: Number of levels
        total_positions: Total positions in grid
        max_sparse_conns: Maximum sparse connections per genome
        iteration_level: H→H iteration level (0 = no h→h)
        allow_hh: Enable h→h discovery
        state: Algorithm state
        cppn_forward: CPPN forward function

    Returns:
        Tuple of (fitnesses, total_hh_count, active_counts_sum, w1_nonzero_sum,
                  w2_nonzero_sum, w1_total, w2_total)
    """
    pop_size = cppns_transformed[0].shape[0]
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]

    # Helper: Query single CPPN at all positions from one source
    def query_cppn_single_source(cppn_tuple, source_coord, outgoing):
        def query_single_position(target_pos):
            bias = jnp.array([1.0])
            if geometry_seeding_enabled:
                # Geometry seeding: add delta (coordinate differences) as CPPN inputs (Risi & Stanley 2012)
                # Delta values allow CPPNs to learn direction-specific patterns
                # Gaussian activation on delta naturally biases toward local connections (peaks at delta=0)
                if outgoing:
                    delta_x = target_pos[0] - source_coord[0]
                    delta_y = target_pos[1] - source_coord[1]
                    delta_input = jnp.array([delta_x, delta_y])
                    inp = jnp.concatenate([source_coord, target_pos, delta_input, bias])
                else:
                    delta_x = source_coord[0] - target_pos[0]
                    delta_y = source_coord[1] - target_pos[1]
                    delta_input = jnp.array([delta_x, delta_y])
                    inp = jnp.concatenate([target_pos, source_coord, delta_input, bias])
            else:
                if outgoing:
                    inp = jnp.concatenate([source_coord, target_pos, bias])
                else:
                    inp = jnp.concatenate([target_pos, source_coord, bias])
            return cppn_forward(state, cppn_tuple, inp)
        return jax.vmap(query_single_position)(all_positions).flatten()

    # Helper: Query single CPPN from multiple sources
    def query_cppn_multi_source(cppn_tuple, source_coords, outgoing):
        return jax.vmap(
            lambda src: query_cppn_single_source(cppn_tuple, src, outgoing)
        )(source_coords)

    # STEP 1: Query CPPN for variance (first input coord only)
    def get_variance_weights(cppn_tuple):
        return query_cppn_single_source(cppn_tuple, input_coords[0], outgoing=True)

    all_weights_for_variance = jax.vmap(
        get_variance_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))

    # STEP 2: Compute hierarchical variances
    level_variances = compute_hierarchical_variances_batch_jit(
        all_weights_for_variance,
        level_sizes=level_sizes,
        level_offsets=level_offsets,
        level_grid_sizes=level_grid_sizes,
        num_levels=num_levels,
    )

    # STEP 3: Compute subdivision masks
    masks_A, _, _ = compute_subdivision_masks_batch_jit(
        level_variances,
        variance_threshold=variance_threshold,
        parent_indices_tuple=parent_indices,
        level_offsets=level_offsets,
        num_levels=num_levels,
        total_positions=total_positions,
    )

    # STEP 4: Query CPPN for input weights
    def get_input_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, input_coords, outgoing=True)

    input_all_weights = jax.vmap(
        get_input_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))

    # STEP 5: Query CPPN for output weights
    def get_output_weights(cppn_tuple):
        return query_cppn_multi_source(cppn_tuple, output_coords, outgoing=False)

    output_all_weights = jax.vmap(
        get_output_weights,
        in_axes=((0, 0, 0, 0),)
    )((cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3]))

    # STEP 6: Build weight matrices with mask
    mask_broadcast = masks_A[:, None, :]

    W1_raw = jnp.tanh(input_all_weights) * max_weight
    W2_raw = jnp.tanh(output_all_weights) * max_weight

    W1 = jnp.where(
        mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
        W1_raw, 0.0
    )
    W2_masked = jnp.where(
        mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
        W2_raw, 0.0
    )
    W2 = jnp.transpose(W2_masked, (0, 2, 1))

    # Compute substrate efficiency metrics for multi-GPU aggregation
    active_counts = jnp.sum(masks_A, axis=1)  # Per-genome active positions
    active_counts_sum = jnp.sum(active_counts)  # Total across population
    w1_nonzero_sum = jnp.sum(W1 != 0)
    w2_nonzero_sum = jnp.sum(W2 != 0)
    w1_total = jnp.array(W1.size, dtype=jnp.int32)
    w2_total = jnp.array(W2.size, dtype=jnp.int32)

    # STEP 7: H→H Discovery (NEW - not in parent!)
    # Note: We use a simplified inline version for JIT-compatibility
    # CHUNKED VERSION to avoid OOM and int32 overflow at depth 7+
    sparse_hh = None
    if allow_hh and iteration_level > 0:
        # Query h→h connections (active → active)
        # Get union of active positions
        union_active = jnp.any(masks_A, axis=0)
        active_indices = jnp.arange(total_positions)

        # Query all active→active pairs for h→h weights
        def get_hh_weights(cppn_tuple, source_idx):
            source_coord = all_positions[source_idx]
            def query_target(target_idx):
                target_coord = all_positions[target_idx]
                bias = jnp.array([1.0])
                if geometry_seeding_enabled:
                    # Geometry seeding: add delta (coordinate differences) as CPPN inputs
                    delta_x = target_coord[0] - source_coord[0]
                    delta_y = target_coord[1] - source_coord[1]
                    delta_input = jnp.array([delta_x, delta_y])
                    inp = jnp.concatenate([source_coord, target_coord, delta_input, bias])
                else:
                    inp = jnp.concatenate([source_coord, target_coord, bias])
                return cppn_forward(state, cppn_tuple, inp)[0]
            return jax.vmap(query_target)(active_indices)

        # Determine if we need chunking based on total_positions
        # int32 max is 2,147,483,647; at depth 7, total_positions^2 = 7.6B > int32 max
        # Use chunking when total_positions > 30000 (conservative threshold)
        use_chunked_hh = total_positions > 30000

        if not use_chunked_hh:
            # Original non-chunked version for smaller grids (depth <= 6)
            def get_hh_for_genome(cppn_tuple, mask):
                # Query all→all
                hh_matrix = jax.vmap(
                    lambda src_idx: get_hh_weights(cppn_tuple, src_idx)
                )(active_indices)

                # Apply thresholds
                hh_matrix = jnp.tanh(hh_matrix) * max_weight
                valid = (jnp.abs(hh_matrix) > band_threshold) & mask[None, :] & mask[:, None]

                # Remove self-connections
                diag_mask = ~jnp.eye(total_positions, dtype=bool)
                valid = valid & diag_mask

                # Convert to sparse representation
                flat_valid = valid.flatten()
                flat_weights = hh_matrix.flatten()

                # Get top max_sparse_conns connections
                num_valid_total = jnp.sum(flat_valid)
                sorted_indices = jnp.argsort(~flat_valid)  # Put valid first
                top_indices = sorted_indices[:max_sparse_conns]

                from_idx = top_indices // total_positions
                to_idx = top_indices % total_positions
                weights = flat_weights[top_indices]
                valid_mask = flat_valid[top_indices]

                return from_idx, to_idx, weights, valid_mask, num_valid_total
        else:
            # CHUNKED VERSION for large grids (depth 7+) to avoid OOM and int32 overflow
            # Process source positions in chunks to avoid creating (total_positions x total_positions) matrix
            hh_chunk_size = 500  # Process 500 sources at a time
            num_chunks = (total_positions + hh_chunk_size - 1) // hh_chunk_size
            conns_per_chunk = max_sparse_conns  # Collect max_sparse_conns from each chunk, merge later

            def get_hh_for_genome(cppn_tuple, mask):
                # Process h→h in chunks to avoid OOM and int32 overflow
                # Each chunk processes hh_chunk_size sources against all targets

                def process_chunk(chunk_idx, carry):
                    # carry: (best_from, best_to, best_weights, best_valid, best_scores, num_valid_total)
                    best_from, best_to, best_weights, best_valid, best_scores, num_valid_total = carry

                    start = chunk_idx * hh_chunk_size
                    # Use min to handle the last chunk
                    actual_chunk_size = jnp.minimum(hh_chunk_size, total_positions - start)

                    # Get source indices for this chunk
                    chunk_source_indices = jnp.arange(hh_chunk_size) + start
                    # Mask out indices beyond total_positions
                    valid_source_mask = chunk_source_indices < total_positions
                    chunk_source_indices = jnp.where(valid_source_mask, chunk_source_indices, 0)

                    # Query h→h for this chunk: (chunk_size, total_positions)
                    hh_chunk = jax.vmap(
                        lambda src_idx: get_hh_weights(cppn_tuple, src_idx)
                    )(chunk_source_indices)

                    # Apply thresholds
                    hh_chunk = jnp.tanh(hh_chunk) * max_weight

                    # Build validity mask
                    source_mask = mask[chunk_source_indices]  # (chunk_size,)
                    valid_chunk = (jnp.abs(hh_chunk) > band_threshold) & source_mask[:, None] & mask[None, :]

                    # Remove self-connections
                    target_indices = jnp.arange(total_positions)
                    self_conn_mask = chunk_source_indices[:, None] != target_indices[None, :]
                    valid_chunk = valid_chunk & self_conn_mask

                    # Mask out invalid sources (beyond total_positions)
                    valid_chunk = valid_chunk & valid_source_mask[:, None]

                    # Count valid connections in this chunk
                    chunk_valid_count = jnp.sum(valid_chunk)
                    num_valid_total = num_valid_total + chunk_valid_count

                    # Flatten chunk and find top connections
                    flat_valid_chunk = valid_chunk.flatten()  # (chunk_size * total_positions,)
                    flat_weights_chunk = hh_chunk.flatten()

                    # Create scores: valid connections first, sorted by weight magnitude
                    # Use negative validity (0 for valid, 1 for invalid) as primary sort key
                    chunk_scores = jnp.where(flat_valid_chunk, -jnp.abs(flat_weights_chunk), jnp.inf)

                    # Find indices of top connections in this chunk
                    # Note: chunk_size * total_positions is within int32 range (500 * 87380 = 43.7M < 2B)
                    num_to_select = jnp.minimum(conns_per_chunk, hh_chunk_size * total_positions)
                    top_chunk_indices = jnp.argsort(chunk_scores)[:conns_per_chunk]

                    # Convert flat indices to (from, to) pairs
                    chunk_from = chunk_source_indices[top_chunk_indices // total_positions]
                    chunk_to = top_chunk_indices % total_positions
                    chunk_weights_sel = flat_weights_chunk[top_chunk_indices]
                    chunk_valid_sel = flat_valid_chunk[top_chunk_indices]
                    chunk_scores_sel = chunk_scores[top_chunk_indices]

                    # Merge with best so far: combine and keep top max_sparse_conns
                    # Concatenate current best with chunk results
                    combined_from = jnp.concatenate([best_from, chunk_from])
                    combined_to = jnp.concatenate([best_to, chunk_to])
                    combined_weights = jnp.concatenate([best_weights, chunk_weights_sel])
                    combined_valid = jnp.concatenate([best_valid, chunk_valid_sel])
                    combined_scores = jnp.concatenate([best_scores, chunk_scores_sel])

                    # Sort by score and keep top max_sparse_conns
                    sorted_idx = jnp.argsort(combined_scores)[:max_sparse_conns]
                    new_best_from = combined_from[sorted_idx]
                    new_best_to = combined_to[sorted_idx]
                    new_best_weights = combined_weights[sorted_idx]
                    new_best_valid = combined_valid[sorted_idx]
                    new_best_scores = combined_scores[sorted_idx]

                    return (new_best_from, new_best_to, new_best_weights, new_best_valid, new_best_scores, num_valid_total)

                # Initialize carry with empty arrays
                init_from = jnp.zeros(max_sparse_conns, dtype=jnp.int32)
                init_to = jnp.zeros(max_sparse_conns, dtype=jnp.int32)
                init_weights = jnp.zeros(max_sparse_conns, dtype=jnp.float32)
                init_valid = jnp.zeros(max_sparse_conns, dtype=bool)
                init_scores = jnp.full(max_sparse_conns, jnp.inf, dtype=jnp.float32)
                init_num_valid = jnp.array(0, dtype=jnp.int32)

                # Use fori_loop to process all chunks
                final_carry = jax.lax.fori_loop(
                    0, num_chunks,
                    process_chunk,
                    (init_from, init_to, init_weights, init_valid, init_scores, init_num_valid)
                )

                best_from, best_to, best_weights, best_valid, _, num_valid_total = final_carry
                return best_from, best_to, best_weights, best_valid, num_valid_total

        # vmap over population
        from_indices, to_indices, hh_weights, valid_masks, num_valids = jax.vmap(
            lambda c0, c1, c2, c3, m: get_hh_for_genome((c0, c1, c2, c3), m)
        )(cppns_transformed[0], cppns_transformed[1], cppns_transformed[2], cppns_transformed[3], masks_A)

        sparse_hh = SparseHiddenConnections(
            from_indices=from_indices,
            to_indices=to_indices,
            weights=hh_weights,
            valid_mask=valid_masks,
            num_valid=num_valids,
        )

    # STEP 8: Evaluate all networks
    if sparse_hh is not None:
        # Hybrid evaluation with sparse h→h
        def eval_single_hybrid(w1, w2, from_idx, to_idx, hh_w, valid):
            outputs = forward_hybrid_vmapped(
                inputs_batch, w1, w2, from_idx, to_idx, hh_w, valid, activate_time
            )
            errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        fitnesses = jax.vmap(eval_single_hybrid)(
            W1, W2,
            sparse_hh.from_indices, sparse_hh.to_indices,
            sparse_hh.weights, sparse_hh.valid_mask,
        )
    else:
        # Dense-only evaluation
        def eval_single_dense(w1, w2):
            hidden = jnp.tanh(inputs_batch @ w1)
            outputs = jax.nn.sigmoid(hidden @ w2)
            errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
            return 1.0 - jnp.mean(errors)

        fitnesses = jax.vmap(eval_single_dense)(W1, W2)

    fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

    # Compute total h→h count for metrics
    if sparse_hh is not None:
        # Sum across population to get total h→h connections
        total_hh_count = jnp.sum(sparse_hh.num_valid)
    else:
        total_hh_count = jnp.array(0, dtype=jnp.int32)

    # Return fitnesses, h→h count, and substrate efficiency metrics
    return (fitnesses, total_hh_count,
            active_counts_sum, w1_nonzero_sum, w2_nonzero_sum, w1_total, w2_total)


# ============================================================================
# SEQUENTIAL MULTI-GPU: Process genomes one-at-a-time inside pmap
# ============================================================================
# This avoids XLA trace explosion from vmap+fori_loop combination.
# Python for-loops inside pmap are NOT traced, so memory is reused.


def _single_genome_pipeline_with_hh(
    cppn_tuple: Tuple,  # Single genome: (nodes, conns, funcs, keys) - NO pop dimension
    all_positions: jnp.ndarray,
    input_coords: jnp.ndarray,
    output_coords: jnp.ndarray,
    inputs_batch: jnp.ndarray,
    targets_batch: jnp.ndarray,
    variance_threshold: float,
    band_threshold: float,
    max_weight: float,
    weight_thresh: float,
    activate_time: int,
    level_sizes: Tuple[int, ...],
    level_offsets: Tuple[int, ...],
    level_grid_sizes: Tuple[int, ...],
    parent_indices: Tuple,
    num_levels: int,
    total_positions: int,
    max_sparse_conns: int,
    iteration_level: int,
    allow_hh: bool,
    state: Any,
    cppn_forward: Any,
    geometry_seeding_enabled: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process SINGLE genome through full pipeline with h→h discovery.

    This is the core function for sequential multi-GPU processing.
    It handles one genome at a time to avoid XLA trace memory explosion.

    Args:
        cppn_tuple: Single CPPN genome (NOT batched)
        ... (same as _full_pipeline_with_hh_core)

    Returns:
        Tuple of (fitness, hh_count, active_count, w1_nonzero, w2_nonzero, w1_total, w2_total)
        All scalars for this single genome.
    """
    num_inputs = input_coords.shape[0]
    num_outputs = output_coords.shape[0]

    # Helper: Query single CPPN at all positions from one source
    def query_cppn_single_source(source_coord, outgoing):
        def query_single_position(target_pos):
            bias = jnp.array([1.0])
            if geometry_seeding_enabled:
                # Geometry seeding: add delta (coordinate differences) as CPPN inputs (Risi & Stanley 2012)
                # Delta values allow CPPNs to learn direction-specific patterns
                # Gaussian activation on delta naturally biases toward local connections (peaks at delta=0)
                if outgoing:
                    delta_x = target_pos[0] - source_coord[0]
                    delta_y = target_pos[1] - source_coord[1]
                    delta_input = jnp.array([delta_x, delta_y])
                    inp = jnp.concatenate([source_coord, target_pos, delta_input, bias])
                else:
                    delta_x = source_coord[0] - target_pos[0]
                    delta_y = source_coord[1] - target_pos[1]
                    delta_input = jnp.array([delta_x, delta_y])
                    inp = jnp.concatenate([target_pos, source_coord, delta_input, bias])
            else:
                if outgoing:
                    inp = jnp.concatenate([source_coord, target_pos, bias])
                else:
                    inp = jnp.concatenate([target_pos, source_coord, bias])
            return cppn_forward(state, cppn_tuple, inp)
        return jax.vmap(query_single_position)(all_positions).flatten()

    # Helper: Query single CPPN from multiple sources
    def query_cppn_multi_source(source_coords, outgoing):
        return jax.vmap(
            lambda src: query_cppn_single_source(src, outgoing)
        )(source_coords)

    # STEP 1: Query CPPN for variance (first input coord only)
    weights_for_variance = query_cppn_single_source(input_coords[0], outgoing=True)
    # Add batch dimension for hierarchical variance (expects [pop, positions])
    weights_for_variance_batch = weights_for_variance[None, :]

    # STEP 2: Compute hierarchical variances
    level_variances = compute_hierarchical_variances_batch_jit(
        weights_for_variance_batch,
        level_sizes=level_sizes,
        level_offsets=level_offsets,
        level_grid_sizes=level_grid_sizes,
        num_levels=num_levels,
    )

    # STEP 3: Compute subdivision masks
    masks_batch, _, _ = compute_subdivision_masks_batch_jit(
        level_variances,
        variance_threshold=variance_threshold,
        parent_indices_tuple=parent_indices,
        level_offsets=level_offsets,
        num_levels=num_levels,
        total_positions=total_positions,
    )
    # Extract single genome mask (remove batch dim)
    mask_A = masks_batch[0]

    # STEP 4: Query CPPN for input weights
    input_weights = query_cppn_multi_source(input_coords, outgoing=True)

    # STEP 5: Query CPPN for output weights
    output_weights = query_cppn_multi_source(output_coords, outgoing=False)

    # STEP 6: Build weight matrices with mask
    W1_raw = jnp.tanh(input_weights) * max_weight
    W2_raw = jnp.tanh(output_weights) * max_weight

    W1 = jnp.where(
        mask_A[None, :] & (jnp.abs(W1_raw) > weight_thresh),
        W1_raw, 0.0
    )
    W2_masked = jnp.where(
        mask_A[None, :] & (jnp.abs(W2_raw) > weight_thresh),
        W2_raw, 0.0
    )
    W2 = W2_masked.T  # Transpose for output

    # Compute substrate efficiency metrics
    active_count = jnp.sum(mask_A)
    w1_nonzero = jnp.sum(W1 != 0)
    w2_nonzero = jnp.sum(W2 != 0)
    w1_total = jnp.array(W1.size, dtype=jnp.int32)
    w2_total = jnp.array(W2.size, dtype=jnp.int32)

    # STEP 7: H→H Discovery (CHUNKED for large grids)
    sparse_from = None
    sparse_to = None
    sparse_weights = None
    sparse_valid = None
    hh_count = jnp.array(0, dtype=jnp.int32)

    if allow_hh and iteration_level > 0:
        active_indices = jnp.arange(total_positions)

        def get_hh_weights(source_idx):
            source_coord = all_positions[source_idx]
            def query_target(target_idx):
                target_coord = all_positions[target_idx]
                bias = jnp.array([1.0])
                if geometry_seeding_enabled:
                    # Geometry seeding: add delta (coordinate differences) as CPPN inputs
                    delta_x = target_coord[0] - source_coord[0]
                    delta_y = target_coord[1] - source_coord[1]
                    delta_input = jnp.array([delta_x, delta_y])
                    inp = jnp.concatenate([source_coord, target_coord, delta_input, bias])
                else:
                    inp = jnp.concatenate([source_coord, target_coord, bias])
                return cppn_forward(state, cppn_tuple, inp)[0]
            return jax.vmap(query_target)(active_indices)

        # Always use chunked version for large grids (depth 6+)
        # This is the key to avoiding OOM
        hh_chunk_size = 500
        num_chunks = (total_positions + hh_chunk_size - 1) // hh_chunk_size

        def process_chunk(carry, chunk_idx):
            best_from, best_to, best_weights, best_valid, best_scores, num_valid_total = carry

            start = chunk_idx * hh_chunk_size
            actual_chunk_size = jnp.minimum(hh_chunk_size, total_positions - start)

            # Get source indices for this chunk
            chunk_source_indices = jnp.arange(hh_chunk_size) + start
            valid_source_mask = chunk_source_indices < total_positions
            chunk_source_indices = jnp.where(valid_source_mask, chunk_source_indices, 0)

            # Query h→h for this chunk: (chunk_size, total_positions)
            hh_chunk = jax.vmap(get_hh_weights)(chunk_source_indices)

            # Apply thresholds
            hh_chunk = jnp.tanh(hh_chunk) * max_weight

            # Build validity mask
            source_mask_chunk = mask_A[chunk_source_indices]
            valid_chunk = (jnp.abs(hh_chunk) > band_threshold) & source_mask_chunk[:, None] & mask_A[None, :]

            # Remove self-connections
            target_indices = jnp.arange(total_positions)
            self_conn_mask = chunk_source_indices[:, None] != target_indices[None, :]
            valid_chunk = valid_chunk & self_conn_mask

            # Mask out invalid sources (beyond total_positions)
            valid_chunk = valid_chunk & valid_source_mask[:, None]

            # Count valid connections
            chunk_valid_count = jnp.sum(valid_chunk)
            num_valid_total = num_valid_total + chunk_valid_count

            # Flatten and find top connections
            flat_valid_chunk = valid_chunk.flatten()
            flat_weights_chunk = hh_chunk.flatten()

            chunk_scores = jnp.where(flat_valid_chunk, -jnp.abs(flat_weights_chunk), jnp.inf)
            top_chunk_indices = jnp.argsort(chunk_scores)[:max_sparse_conns]

            chunk_from = chunk_source_indices[top_chunk_indices // total_positions]
            chunk_to = top_chunk_indices % total_positions
            chunk_weights_sel = flat_weights_chunk[top_chunk_indices]
            chunk_valid_sel = flat_valid_chunk[top_chunk_indices]
            chunk_scores_sel = chunk_scores[top_chunk_indices]

            # Merge with best so far
            combined_from = jnp.concatenate([best_from, chunk_from])
            combined_to = jnp.concatenate([best_to, chunk_to])
            combined_weights = jnp.concatenate([best_weights, chunk_weights_sel])
            combined_valid = jnp.concatenate([best_valid, chunk_valid_sel])
            combined_scores = jnp.concatenate([best_scores, chunk_scores_sel])

            sorted_idx = jnp.argsort(combined_scores)[:max_sparse_conns]
            new_best_from = combined_from[sorted_idx]
            new_best_to = combined_to[sorted_idx]
            new_best_weights = combined_weights[sorted_idx]
            new_best_valid = combined_valid[sorted_idx]
            new_best_scores = combined_scores[sorted_idx]

            return (new_best_from, new_best_to, new_best_weights, new_best_valid, new_best_scores, num_valid_total), None

        # Initialize carry
        init_from = jnp.zeros(max_sparse_conns, dtype=jnp.int32)
        init_to = jnp.zeros(max_sparse_conns, dtype=jnp.int32)
        init_weights = jnp.zeros(max_sparse_conns, dtype=jnp.float32)
        init_valid = jnp.zeros(max_sparse_conns, dtype=bool)
        init_scores = jnp.full(max_sparse_conns, jnp.inf, dtype=jnp.float32)
        init_num_valid = jnp.array(0, dtype=jnp.int32)

        # Use scan instead of fori_loop to avoid unrolling issues
        final_carry, _ = jax.lax.scan(
            process_chunk,
            (init_from, init_to, init_weights, init_valid, init_scores, init_num_valid),
            jnp.arange(num_chunks)
        )

        sparse_from, sparse_to, sparse_weights, sparse_valid, _, hh_count = final_carry

    # STEP 8: Evaluate network
    if allow_hh and iteration_level > 0 and sparse_from is not None:
        # Hybrid evaluation with sparse h→h
        outputs = forward_hybrid_vmapped(
            inputs_batch, W1, W2, sparse_from, sparse_to, sparse_weights, sparse_valid, activate_time
        )
    else:
        # Dense-only evaluation
        hidden = jnp.tanh(inputs_batch @ W1)
        outputs = jax.nn.sigmoid(hidden @ W2)

    errors = jnp.mean((outputs - targets_batch) ** 2, axis=1)
    fitness = 1.0 - jnp.mean(errors)
    fitness = jnp.where(jnp.isnan(fitness), -jnp.inf, fitness)

    return (fitness, hh_count, active_count, w1_nonzero, w2_nonzero, w1_total, w2_total)


# POPULATION_PARALLEL_SEQUENTIAL: Process genomes one-at-a-time inside pmap
# Each GPU processes pop_per_gpu genomes SEQUENTIALLY via Python for-loop
# This avoids XLA trace explosion from vmap + fori_loop
@functools.partial(
    jax.pmap,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22),
)
def _pmap_sequential_pop_parallel_hh(
    cppns_shard,          # 0: sharded (each GPU gets pop slice) - shape (pop_per_gpu, ...)
    all_positions,        # None: replicated
    input_coords,         # None: replicated
    output_coords,        # None: replicated
    inputs_batch,         # None: replicated (full dataset on each GPU)
    targets_batch,        # None: replicated
    variance_threshold,   # None: replicated
    band_threshold,       # None: replicated
    max_weight,           # None: replicated
    weight_thresh,        # None: replicated
    activate_time,        # STATIC (10)
    level_sizes,          # STATIC (11)
    level_offsets,        # STATIC (12)
    level_grid_sizes,     # STATIC (13)
    parent_indices,       # None: replicated (JAX arrays)
    num_levels,           # STATIC (15)
    total_positions,      # STATIC (16)
    max_sparse_conns,     # STATIC (17)
    iteration_level,      # STATIC (18)
    allow_hh,             # STATIC (19)
    state,                # None: replicated
    cppn_forward,         # STATIC (21)
    geometry_seeding_enabled,  # STATIC (22) - Risi & Stanley 2012 geometry seeding
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequential population-parallel: process genomes one at a time inside pmap.

    The key insight: Python for-loops inside a pmap-traced function are NOT
    traced by XLA. Each iteration executes and deallocates before the next,
    avoiding the memory explosion from fori_loop unrolling.

    This provides ~2x speedup over single-GPU fallback by using both GPUs.

    Returns:
        Tuple of (fitnesses, hh_counts, active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals)
        Each has shape (pop_per_gpu,) for this GPU's results.
    """
    pop_per_gpu = cppns_shard[0].shape[0]

    # Use scan for JIT-compatible sequential processing
    # Note: We use scan with a single element per iteration to achieve
    # sequential processing that reuses memory

    def process_one_genome(_, genome_idx):
        # Extract single genome from the shard
        cppn_tuple = (
            cppns_shard[0][genome_idx],
            cppns_shard[1][genome_idx],
            cppns_shard[2][genome_idx],
            cppns_shard[3][genome_idx],
        )

        # Process this single genome
        result = _single_genome_pipeline_with_hh(
            cppn_tuple,
            all_positions, input_coords, output_coords,
            inputs_batch, targets_batch,
            variance_threshold, band_threshold, max_weight, weight_thresh,
            activate_time, level_sizes, level_offsets, level_grid_sizes,
            parent_indices, num_levels, total_positions,
            max_sparse_conns, iteration_level, allow_hh, state, cppn_forward,
            geometry_seeding_enabled,
        )

        return None, result

    # Process all genomes sequentially
    _, results = jax.lax.scan(process_one_genome, None, jnp.arange(pop_per_gpu))

    # Unpack results - each is (pop_per_gpu,) array
    fitnesses = results[0]
    hh_counts = results[1]
    active_counts = results[2]
    w1_nonzeros = results[3]
    w2_nonzeros = results[4]
    w1_totals = results[5]
    w2_totals = results[6]

    # Sum metrics across population for this GPU
    total_hh = jnp.sum(hh_counts)
    total_active = jnp.sum(active_counts)
    total_w1_nz = jnp.sum(w1_nonzeros)
    total_w2_nz = jnp.sum(w2_nonzeros)
    total_w1 = jnp.sum(w1_totals)
    total_w2 = jnp.sum(w2_totals)

    return (fitnesses, total_hh, total_active, total_w1_nz, total_w2_nz, total_w1, total_w2)


# FULL_PIPELINE_PARALLEL: Split data across GPUs, replicate full population
# in_axes: (None=replicated) for cppns, positions, coords, scalars, state
#          (0=sharded) for inputs, targets
# NOTE: parent_indices (index 14) contains JAX arrays so NOT static
@functools.partial(
    jax.pmap,
    in_axes=(None, None, None, None, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22),
)
def _pmap_data_parallel_full_hh(
    cppns_transformed,    # None: replicated
    all_positions,        # None: replicated
    input_coords,         # None: replicated
    output_coords,        # None: replicated
    inputs_shard,         # 0: sharded (each GPU gets data slice)
    targets_shard,        # 0: sharded
    variance_threshold,   # None: replicated
    band_threshold,       # None: replicated
    max_weight,           # None: replicated
    weight_thresh,        # None: replicated
    activate_time,        # STATIC (10)
    level_sizes,          # STATIC (11)
    level_offsets,        # STATIC (12)
    level_grid_sizes,     # STATIC (13)
    parent_indices,       # None: replicated (JAX arrays - NOT static)
    num_levels,           # STATIC (15)
    total_positions,      # STATIC (16)
    max_sparse_conns,     # STATIC (17)
    iteration_level,      # STATIC (18)
    allow_hh,             # STATIC (19)
    state,                # None: replicated
    cppn_forward,         # STATIC (21)
    geometry_seeding_enabled,  # STATIC (22) - Risi & Stanley 2012 geometry seeding
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Data-parallel: full population on each GPU, different data slices.

    Returns:
        Tuple of (fitnesses, total_hh_count, active_counts_sum, w1_nonzero_sum,
                  w2_nonzero_sum, w1_total, w2_total) where each is sharded per GPU.
    """
    return _full_pipeline_with_hh_core(
        cppns_transformed, all_positions, input_coords, output_coords,
        inputs_shard, targets_shard,
        variance_threshold, band_threshold, max_weight, weight_thresh,
        activate_time, level_sizes, level_offsets, level_grid_sizes,
        parent_indices, num_levels, total_positions,
        max_sparse_conns, iteration_level, allow_hh, state, cppn_forward,
        geometry_seeding_enabled,
    )


# POPULATION_PARALLEL: Split population across GPUs, replicate full dataset
# in_axes: (0=sharded) for cppns
#          (None=replicated) for everything else
# NOTE: parent_indices (index 14) contains JAX arrays so NOT static
@functools.partial(
    jax.pmap,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22),
)
def _pmap_pop_parallel_hh(
    cppns_shard,          # 0: sharded (each GPU gets pop slice)
    all_positions,        # None: replicated
    input_coords,         # None: replicated
    output_coords,        # None: replicated
    inputs_batch,         # None: replicated (full dataset on each GPU)
    targets_batch,        # None: replicated
    variance_threshold,   # None: replicated
    band_threshold,       # None: replicated
    max_weight,           # None: replicated
    weight_thresh,        # None: replicated
    activate_time,        # STATIC (10)
    level_sizes,          # STATIC (11)
    level_offsets,        # STATIC (12)
    level_grid_sizes,     # STATIC (13)
    parent_indices,       # None: replicated (JAX arrays - NOT static)
    num_levels,           # STATIC (15)
    total_positions,      # STATIC (16)
    max_sparse_conns,     # STATIC (17)
    iteration_level,      # STATIC (18)
    allow_hh,             # STATIC (19)
    state,                # None: replicated
    cppn_forward,         # STATIC (21)
    geometry_seeding_enabled,  # STATIC (22) - Risi & Stanley 2012 geometry seeding
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Population-parallel: full data on each GPU, different pop slices.

    Returns:
        Tuple of (fitnesses, total_hh_count, active_counts_sum, w1_nonzero_sum,
                  w2_nonzero_sum, w1_total, w2_total) where each is sharded per GPU.
    """
    return _full_pipeline_with_hh_core(
        cppns_shard, all_positions, input_coords, output_coords,
        inputs_batch, targets_batch,
        variance_threshold, band_threshold, max_weight, weight_thresh,
        activate_time, level_sizes, level_offsets, level_grid_sizes,
        parent_indices, num_levels, total_positions,
        max_sparse_conns, iteration_level, allow_hh, state, cppn_forward,
        geometry_seeding_enabled,
    )


# ============================================================================
# Class Aliases
# ============================================================================

# Alias for "Full" variant (with neuromodulation support)
EMRHyperNEATFull = EMRHyperNEAT

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main class (both names)
    'EMRHyperNEAT',
    'EMRHyperNEATFull',

    # Neuromodulation dataclasses and config
    'NeuromodulationConfig',
    'MultiTaskConfig',
    'NEUROMODULATION_PRESETS',
    'NT_TASK_PRESETS',
    'derive_receptor_from_weight',

    # Dataclasses
    'SparseHiddenConnections',
    'EMRConfig',
    'EMRRecurrenceMetrics',

    # Presets
    'RECURRENCE_PRESETS',
    'get_recurrence_preset',

    # Forward pass
    'ForwardPassMode',
    'forward_unified_extended',
    'forward_hybrid_vmapped',
    'eval_single_network_hybrid',

    # Multi-hop (key innovation)
    'compute_multi_hop_connections_matrix_power',
    'compute_multi_hop_connections_fori_loop',

    # Discovery
    'discover_sparse_hh_vectorized_multi_hop',

    # Constraint filtering
    'get_connection_constraint_mask',
    'get_connection_constraint_mask_batched',

    # Cache
    'HHCacheManager',

    # Multi-GPU (re-export from parent)
    'MultiGPUStrategy',
    'PositionShardingConfig',
]
