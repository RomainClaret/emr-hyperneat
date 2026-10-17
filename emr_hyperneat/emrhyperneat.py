"""The public `EMRHyperNEAT` class and its feature layer.

Everything a user configures lives here: per-node dynamic activation functions
(the 18-function `ACTIVATION_LIST` and its palettes), per-node aggregation,
neuromodulation with multi-task evaluation, and h→h recurrence with its caches.
The substrate engine and multi-GPU dispatch beneath it are in
`emrhyperneat_base.py`; the mechanism end to end is documented in
docs/architecture.md ("How a generation is computed").

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

MULTI-HOP EXPANSION
===================

With A the genome's direct h->h connections (the masked adjacency), iteration_level k forms the
hop terms decay^(h-1) A^h for h = 1..k and keeps, per pair, the term of largest magnitude; each
product is thresholded before the decay. Every hop is an existing direct connection
(`recurrence.multi_hop_second_factor: masked`, the default since 0.2.0). `raw` reproduces
0.1.0's expansion, which multiplied by the raw weights over the population's union of active
positions, so a genome's composite connections could end on positions it did not have active,
ignore its connection class and depend on the other genomes. JIT-compatible because
iteration_level and the second factor are static, so the Python loop compiles away.

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

The recurrent presets discover direct h->h connections only (iteration_level 1) since 0.2.0; in
0.1.0 they composed two hops (level 2). Set recurrence.iteration_level to compose connections.

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
HHCacheManager reuses one population-wide h->h connection set for up to hh_refresh_interval
generations. Off by default since 0.2.0 (on in 0.1.0): reproduction moves genomes between
population slots, so between refreshes a genome is scored with the recurrent connections
discovered for whichever genome held its slot. Set hh_cache_enabled: true only to reproduce a
cache-on campaign.

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
    # iteration_level: 1  # the presets' level since 0.2.0; 2 composes connections
    # multi_hop_algorithm: matrix_power  # or fori_loop
    # hop_decay_factor: 0.8
    # multi_hop_second_factor: masked  # or raw, the 0.1.0 algebra
    # hh_cache_enabled: false  # default since 0.2.0
    # hh_refresh_interval: 5
```
"""

import dataclasses
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
    cppn_query_rows,
    safe_matmul,
    traced_device_get,
    clear_per_device_caches,
)

from emr_hyperneat._compat.core.base_algorithm import BaseAlgorithm, AlgorithmMetrics
from emr_hyperneat._compat.utils.config_manager import ConfigManager

# Execution modes (memory residency). Only the vocabulary and the chunk arithmetic
# are imported here; the strategy modules import from this one, so the dispatcher
# imports them lazily at call time to keep the dependency one-way.
from emr_hyperneat.execution import (  # noqa: E501
    EXECUTION_KEYS,
    ExecutionConfig,
    ExecutionMode,
    parse_execution_config,
)
from emr_hyperneat.execution.chunking import (  # noqa: E501
    optimal_position_chunk_size,
    population_chunk_for_query,
    population_chunk_size,
)


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
        mod_neuron_ratio: Fraction of a genome's active hidden positions that are modulatory, at least one
            (default 0.1)
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
    # The neuromodulated node: 'paper' (the published rule, on every path) or 'hmr_full' (HMR FULL's
    # rule, which 0.1.0's single-task path computed); see true_neuromodulation_node_fn.
    node_rule: str = 'paper'
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
        orthogonality_bonus: Must stay 0.0: a non-zero value raises (one value added to every genome)
        specialization_bonus: Must stay 0.0: a non-zero value raises (one value added to every genome)
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
    fitness_mode: str = 'mse'  # MULTITASK_FITNESS_MODES ('ce' and unknown names raise)

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
    orthogonality_bonus_weight: float = 0.0  # any non-zero value raises in run_generation_multitask (a constant)
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


#: The fitness modes of the multi-task evaluators (compute_fitness).
MULTITASK_FITNESS_MODES = ('mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy', 'accuracy_confident')
#: Fitness modes the single-task evaluation computes (emr_hyperneat.fitness_mode, _eval_single_fitness).
#: create_config raises on any other name; 0.1.0 scored it, and acc_mse and accuracy_confident, as MSE.
FITNESS_MODES = ('mse', 'accuracy', 'hybrid', 'bce', 'soft_accuracy', 'ce', 'acc_mse', 'accuracy_confident')
#: Accepted values of the multi-task settings (create_config raises on any other).
MULTITASK_AGGREGATIONS = ('mean', 'min', 'weighted', 'product', 'softmin', 'harmonic')
#: Weight interpretations of dynamic_functions and aggregation mode 'weight_interpretation'.
#: create_config raises on any other name; 0.1.0 ran magnitude_bio for it (HMR's dynamic-functions
#: variant had more, among them an inverted 'magnitude', which EMR never implemented).
ACTIVATION_INTERPRETATIONS = ('magnitude_bio', 'variance', 'sign')
AGGREGATION_INTERPRETATIONS = ('magnitude_bio', 'variance', 'sign_uniformity', 'fan_in')
GENERALIST_BONUS_TYPES = ('none', 'min_bonus', 'variance_penalty', 'threshold_bonus')
TWO_MODULE_MODES = ('none', 'parallel', 'sequential')
MODULATION_MODES = ('full', 'gating_only', 'gain_bias_only')


def compute_fitness(outputs: jnp.ndarray, targets: jnp.ndarray, mode: str = 'mse') -> float:
    """The multi-task evaluators' fitness over flattened outputs and targets: 'mse', 'accuracy', 'acc_mse', 'hybrid',
    'bce', 'soft_accuracy' and 'accuracy_confident' (accuracy plus 0.1 of the outputs' mean distance from 0.5, scaled
    to [0, 1]); MULTITASK_FITNESS_MODES. In 0.1.0 'accuracy_confident', 'ce' and any unknown name computed MSE.

    Raises:
        ValueError: 'ce', categorical cross-entropy over softmax outputs: these evaluators score sigmoid outputs of
            Boolean tasks, where it ranks a constant output above a solver (a constant 0.9999 scores 0.99995, a
            0.99/0.01 XOR solver 0.99499); and an unknown mode.
    """
    if mode == 'ce':
        raise ValueError("fitness mode 'ce' is categorical cross-entropy over softmax outputs; the multi-task "
                         "evaluators score sigmoid outputs, where it ranks a constant output above a solver: "
                         "use 'bce'")
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
    elif mode == 'accuracy_confident':
        predictions = (outputs > 0.5).astype(jnp.float32)
        return jnp.mean(predictions == targets) + 0.1 * jnp.mean(jnp.abs(outputs - 0.5) * 2.0)
    raise ValueError(f"unknown fitness mode {mode!r}; the multi-task modes are {MULTITASK_FITNESS_MODES}")


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
        # neuromodulation variant define it; in 0.1.0 this name computed MSE here
        return (_eval_single_fitness(outputs, targets_batch, 'accuracy')
                + 0.01 * _eval_single_fitness(outputs, targets_batch, 'mse'))
    elif mode == 'accuracy_confident':
        # Accuracy plus 0.1 of the outputs' mean distance from 0.5, scaled to [0, 1]; in 0.1.0 this
        # name computed MSE here
        return (_eval_single_fitness(outputs, targets_batch, 'accuracy')
                + 0.1 * jnp.mean(jnp.abs(outputs - 0.5) * 2.0))
    else:  # 'mse' (FITNESS_MODES is checked at configuration)
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

def self_connection_rows(
    positions: jnp.ndarray,
    geometry_seeding_enabled: bool = False,
) -> jnp.ndarray:
    """CPPN input rows for the self-connection query at every position.

    A self-connection has source and target at the same position, ``(x, y, x, y, 1)``.
    The query reads per-node properties from it: a hidden node's activation and
    aggregation function in ``cppn_output`` mode, and its receptor densities under
    neuromodulation. Column order matches `cppn_query_rows` exactly:

      - 5-D: ``[x1, y1, x2, y2, bias]``
      - 7-D: ``[x1, y1, x2, y2, delta_x, delta_y, bias]`` (deltas before the bias,
        both exactly zero here, but present so the genome's input count matches)

    Args:
        positions: Positions to query (num_positions, 2).
        geometry_seeding_enabled: Add the two zero delta columns (7-D rows).

    Returns:
        (num_positions, 5) or (num_positions, 7) rows.
    """
    num_positions = positions.shape[0]
    bias = jnp.ones((num_positions, 1))
    if geometry_seeding_enabled:
        zero_delta = jnp.zeros((num_positions, 1))
        return jnp.concatenate(
            [positions, positions, zero_delta, zero_delta, bias], axis=1
        )  # (num_positions, 7)
    return jnp.concatenate([positions, positions, bias], axis=1)  # (num_positions, 5)


def batch_query_population_self_connections(
    state: Any,
    cppns_transformed: Tuple,
    positions: jnp.ndarray,
    cppn_forward: Any,
    num_cppn_outputs: int = 1,
    pop_chunk_size: int = 0,
    geometry_seeding_enabled: bool = False,
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
        geometry_seeding_enabled: If True, the CPPN takes 7 inputs rather than 5, so
            the delta columns must be supplied here too. For a self-connection source
            and target coincide, so both deltas are exactly zero -- but the columns
            still have to be present or the genome's input_idx will not match.

    Returns:
        (pop_size, num_positions, num_cppn_outputs) array of CPPN outputs at self-connections
    """
    pop_size = cppns_transformed[0].shape[0]

    # Build self-connection inputs. For self-connections, source == target.
    inputs = self_connection_rows(positions, geometry_seeding_enabled)

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
    # Cross-output normalization for multiclass output heads (normalizes across the
    # last axis = output units). Only meaningful as an OUTPUT activation on multi-output
    # substrates; pairs with the categorical 'ce' fitness mode (probabilities summing to 1).
    # Deliberately absent from ACTIVATION_LIST below: it is a per-substrate output choice,
    # not an evolvable per-node one, and adding it there would shift every palette index.
    'softmax': lambda x: jax.nn.softmax(x, axis=-1),
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


#: How a value in [-1, 1] becomes one of K function indices (dynamic_functions.index_decode,
#: aggregation.index_decode). 'tanh', the default and 0.1.0's decode, applies tanh before binning; the values
#: it bins (the CPPN's channels, whose output transform is tanh, and every weight-interpretation signal) are
#: already in [-1, 1], so tanh squeezes them into [-0.762, 0.762] and the end bins of a palette of 9 or more
#: functions are never reached (18 functions: positions 2 to 15). 'uniform' bins the value itself, every bin
#: equally wide.
INDEX_DECODES = ('tanh', 'uniform')


def continuous_to_index(raw: jnp.ndarray, num_options: int, decode: str = 'tanh') -> jnp.ndarray:
    """Convert a continuous value to a discrete function index in [0, num_options - 1].

    Args:
        raw: Continuous values, in [-1, 1] for every caller in this module (CPPN channels and
            weight-interpretation signals).
        num_options: Number of function options.
        decode: 'tanh' (the default): bin tanh(raw); 'uniform': bin raw itself, clipped to [-1, 1]
            (INDEX_DECODES).

    Returns:
        Integer indices in [0, num_options-1]
    """
    if decode == 'uniform':
        scaled = (jnp.clip(raw, -1.0, 1.0) + 1.0) / 2.0 * num_options
    else:
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
    decode: str = 'tanh',
) -> jnp.ndarray:
    """Convert continuous CPPN output to activation index using custom palette.

    Instead of mapping to indices 0..N-1, maps to specific indices in palette array.
    This allows testing hypothesis that palette choice (which activations) matters.

    Args:
        raw: Continuous values from CPPN
        palette: Array of activation indices to use, e.g., [4, 11, 12] for sin, burst, resonator
        decode: 'tanh' (the default) or 'uniform', as in `continuous_to_index`.

    Returns:
        Integer indices from the palette array
    """
    position = continuous_to_index(raw, len(palette), decode)
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
        palette: Activation indices that may be present, for efficient iteration. Concrete
            (a sequence or a concrete array): it selects Python functions while tracing.

    Returns:
        Activated values with same shape as pre_activation
    """
    result = jnp.zeros_like(pre_activation)
    # Python ints, not the array's elements: iterating a jax array inside lax.scan stages
    # the unstack, and a traced index cannot select a function. In 0.1.0 this made every
    # recurrent palette configuration fail under hh_activation_mode 'every_iteration'.
    for act_idx in np.asarray(palette).tolist():
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
    decode: str = 'tanh',
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
        decode: 'tanh' or 'uniform' (INDEX_DECODES)

    Returns:
        Aggregation indices, shape (pop_size, num_nodes)
    """
    if interpretation == 'magnitude_bio':
        # High magnitude → sum (preserve scale), low → mean (normalize)
        mean_abs = jnp.mean(jnp.abs(weights), axis=1)  # (pop_size, num_nodes)
        normalized = -jnp.tanh(mean_abs * 2 - 1)  # Inverted like activation
        return continuous_to_index(normalized, num_aggregations, decode)

    elif interpretation == 'variance':
        # High variance → max (winner-take-all), low → mean (equal contribution)
        weight_var = jnp.var(weights, axis=1)  # (pop_size, num_nodes)
        normalized = jnp.tanh(weight_var * 5)
        return continuous_to_index(normalized, num_aggregations, decode)

    elif interpretation == 'sign_uniformity':
        # All same sign → product (AND-gate), mixed → sum (preserve cancellation)
        pos_ratio = jnp.mean((weights > 0).astype(jnp.float32), axis=1)  # (pop_size, num_nodes)
        sign_uniformity = 2 * jnp.abs(pos_ratio - 0.5)  # 0 = mixed, 1 = uniform
        return continuous_to_index(sign_uniformity, num_aggregations, decode)

    elif interpretation == 'fan_in':
        # Based on number of significant inputs (above threshold)
        significant = jnp.abs(weights) > 0.1  # (pop_size, num_inputs, num_nodes)
        fan_in = jnp.sum(significant.astype(jnp.float32), axis=1)  # (pop_size, num_nodes)
        max_fan_in = weights.shape[1]
        normalized = jnp.tanh((fan_in / max_fan_in) * 2 - 1)
        return continuous_to_index(normalized, num_aggregations, decode)

    else:
        # Default: magnitude_bio (same as activation)
        mean_abs = jnp.mean(jnp.abs(weights), axis=1)
        normalized = -jnp.tanh(mean_abs * 2 - 1)
        return continuous_to_index(normalized, num_aggregations, decode)


#: Draws of the random function modes. 'genome': one draw per population slot from the run's key,
#: as HMR drew its negative controls; 'shared': 0.1.0's draw, one assignment for every genome from a
#: fixed key.
RANDOM_SCOPES = ('genome', 'shared')


def random_function_indices(scope: str, seed: Optional[int], run_key: jnp.ndarray, salt: int,
                            generation: Optional[int], num_options: int,
                            palette: Optional[jnp.ndarray], pop_size: int,
                            total_positions: int) -> jnp.ndarray:
    """One random function index per node, for the random_fixed and random_generation modes.

    Scope 'genome' draws each population slot independently, from the run's key folded with
    ``salt`` (a fresh key every generation, so random_generation redraws) or, when ``seed`` is set,
    from the seed (the same draw in every run; folded with the generation for random_generation).
    Scope 'shared' is 0.1.0's draw, bit for bit: one assignment for every genome from
    ``PRNGKey(seed or 42)``, or ``PRNGKey(generation + (seed or 0))`` per generation, so every
    genome and every run of a configuration drew the same. It is kept to reproduce runs made that
    way; as a control it compares the evolved assignment with one random assignment per run, not
    with random assignments.

    Args:
        scope: 'genome' or 'shared'.
        seed: The block's random_seed, or None.
        run_key: The run's key this generation (state.randkey).
        salt: Separates this draw from the run's other uses of the key.
        generation: The generation for a per-generation draw, None for a fixed one.
        num_options: Functions to draw from (the palette's size with a palette).
        palette: ACTIVATION_LIST indices to map the draw to, or None.
        pop_size: Population size, the first dimension of the result.
        total_positions: Grid positions, the second dimension of the result.

    Returns:
        Indices of shape (pop_size, total_positions).
    """
    if scope == 'shared':
        key = jax.random.PRNGKey((seed or 42) if generation is None else generation + (seed or 0))
        draw = jax.random.randint(key, (total_positions,), 0, num_options, dtype=jnp.int32)
    else:
        if seed is None:
            key = jax.random.fold_in(run_key, salt)
        else:
            key = jax.random.PRNGKey(seed)
            if generation is not None:
                key = jax.random.fold_in(key, generation)
        draw = jax.random.randint(key, (pop_size, total_positions), 0, num_options,
                                  dtype=jnp.int32)
    if palette is not None:
        draw = palette[draw]
    return jnp.broadcast_to(draw, (pop_size, total_positions))


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
# Feed-forward substrate: shared by the generation loop and the public API
# ============================================================================
# The standard path of run_generation_verbose builds and evaluates its substrate
# through these functions, and emr_hyperneat.substrate composes the same ones, so a
# substrate built outside the generation loop is the one EMR evaluates rather than a
# second implementation of it. Change them only together with that module's tests.

#: Weights whose magnitude is at or below this, after tanh scaling, are pruned. EMR
#: has always used this constant on every path; it is not a configuration option.
WEIGHT_THRESHOLD = 0.1


def substrate_masks(
    variance_weights: jnp.ndarray,
    grid: HierarchicalGridStructure,
    variance_threshold: float,
    initial_depth: int = 0,
    legacy_level0_variance: bool = False,
) -> jnp.ndarray:
    """Which grid positions become hidden nodes.

    The variance source is the raw CPPN weight output from the first substrate input
    to every grid position. A cell subdivides when the variance over its four children
    exceeds the threshold (or while its level is below ``initial_depth``), and every
    reached cell, at every level, is a hidden node: not only the leaves.

    Args:
        variance_weights: (pop_size, total_positions) raw CPPN weight outputs.
        grid: The multi-resolution grid from ``get_hierarchical_grid(max_depth)``.
        variance_threshold: Subdivision threshold.
        initial_depth: Levels subdivided unconditionally.
        legacy_level0_variance: The original standard-path level-0 rule (``variance_rule: legacy``).

    Returns:
        (pop_size, total_positions) boolean mask.
    """
    level_variances = compute_hierarchical_variances_batch(
        variance_weights, grid, legacy_level0_variance=legacy_level0_variance
    )
    return compute_subdivision_masks_batch(
        level_variances, variance_threshold, grid, return_all_masks=False,
        initial_depth=initial_depth,
    )


def substrate_weight_matrices(
    input_weights: jnp.ndarray,
    output_weights: jnp.ndarray,
    masks: jnp.ndarray,
    max_weight: float,
    weight_threshold: float = WEIGHT_THRESHOLD,
    locality_radius: Optional[float] = None,
    input_coords: Optional[jnp.ndarray] = None,
    output_coords: Optional[jnp.ndarray] = None,
    positions: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Input-to-hidden and hidden-to-output weights from raw CPPN weight outputs.

    A weight is ``tanh(cppn) * max_weight``, optionally scaled down with distance
    beyond ``locality_radius``, and kept only at a hidden position and when its
    magnitude exceeds ``weight_threshold``.

    Args:
        input_weights: (pop_size, num_inputs, total_positions) raw CPPN outputs,
            substrate input -> grid position.
        output_weights: (pop_size, num_outputs, total_positions) raw CPPN outputs,
            grid position -> substrate output.
        masks: (pop_size, total_positions) hidden-node mask from `substrate_masks`.
        max_weight: Weight scale.
        weight_threshold: Pruning threshold.
        locality_radius: Radius of the geometry-seeding locality penalty, or None.
        input_coords: (num_inputs, 2); needed only with ``locality_radius``.
        output_coords: (num_outputs, 2); needed only with ``locality_radius``.
        positions: (total_positions, 2) grid positions; needed only with
            ``locality_radius``.

    Returns:
        ``(W1, W2)``: W1 is (pop_size, num_inputs, total_positions), W2 is
        (pop_size, total_positions, num_outputs), zero wherever pruned.
    """
    # Broadcast mask: (pop_size, 1, total_positions) for weight masking
    active_mask_broadcast = masks[:, None, :]

    # Apply tanh + max_weight scaling, then mask
    W1_raw = jnp.tanh(input_weights) * max_weight

    # Locality penalty for geometry seeding. W1: input_coords -> positions.
    if locality_radius is not None:
        W1_raw = apply_locality_penalty_to_weights(
            W1_raw, input_coords, positions, locality_radius
        )

    W1 = jnp.where(
        active_mask_broadcast & (jnp.abs(W1_raw) > weight_threshold),
        W1_raw,
        0.0
    )

    W2_raw = jnp.tanh(output_weights) * max_weight

    # W2_raw is (pop, num_outputs, total_positions) before the transpose, so the
    # penalty is applied on the (pop, positions, outputs) view and moved back.
    if locality_radius is not None:
        W2_for_penalty = W2_raw.transpose(0, 2, 1)  # (pop, total_positions, num_outputs)
        W2_for_penalty = apply_locality_penalty_to_weights(
            W2_for_penalty, positions, output_coords, locality_radius
        )
        W2_raw = W2_for_penalty.transpose(0, 2, 1)  # Back to (pop, num_outputs, total_positions)

    W2 = jnp.where(
        active_mask_broadcast & (jnp.abs(W2_raw) > weight_threshold),
        W2_raw,
        0.0
    )
    W2 = W2.transpose(0, 2, 1)  # (pop, num_outputs, total_positions) → (pop, total_positions, num_outputs)
    return W1, W2


def cppn_output_function_indices(
    self_conn: Optional[jnp.ndarray],
    masks: jnp.ndarray,
    W1: jnp.ndarray,
    channel: int,
    num_options: int,
    num_cppn_outputs: int,
    palette: Optional[jnp.ndarray] = None,
    decode: str = 'tanh',
) -> jnp.ndarray:
    """Per-node function indices in ``cppn_output`` mode.

    With a dedicated CPPN output, a hidden node's index comes from that output at the
    node's self-connection ``(x, y, x, y, 1)``, zeroed where the mask drops the node,
    and binned by `continuous_to_index`. A one-output CPPN has no dedicated channel,
    and the index is derived from the node's mean incoming weight instead.

    Args:
        self_conn: (pop_size, total_positions, num_cppn_outputs) self-connection query,
            or None.
        masks: (pop_size, total_positions) hidden-node mask.
        W1: (pop_size, num_inputs, total_positions), used by the one-output fallback.
        channel: CPPN output channel that carries this function choice.
        num_options: Number of bins (the palette size when a palette is given).
        num_cppn_outputs: The CPPN's output count.
        palette: Optional array of ACTIVATION_LIST indices; bins are mapped through it.
        decode: 'tanh' (the default) or 'uniform' (INDEX_DECODES). The one-output fallback's mean
            weight is unbounded and is squashed with tanh under both decodes.

    Returns:
        (pop_size, total_positions) int32 indices (ACTIVATION_LIST or
        AGGREGATION_LIST positions).
    """
    if num_cppn_outputs > 1 and self_conn is not None:
        raw = self_conn[:, :, channel]  # (pop, positions)
        raw = jnp.where(masks, raw, 0.0)
    else:
        # Fallback: derive from weight matrix (1-output CPPN); the mean weight is unbounded
        raw = jnp.mean(W1, axis=1)  # (pop, num_positions)
        if decode == 'uniform':
            raw = jnp.tanh(raw)
    indices = continuous_to_index(raw, num_options, decode)
    if palette is not None:
        indices = palette[indices]
    return indices


def input_drive(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    agg_indices: Optional[jnp.ndarray] = None,
    global_agg_fn: Optional[Callable] = None,
    per_node_aggregation: bool = False,
    true_aggregation: bool = True,
    num_aggregations: int = 4,
) -> jnp.ndarray:
    """What hidden nodes receive from the substrate inputs, before their activation.

    Aggregation at a hidden node is a global function, a per-node function (exact, or
    approximated by a matmul with correction factors), or the sum. Every evaluation path,
    feed-forward, recurrent and neuromodulated, takes its input drive from here, which is what
    makes a recurrent substrate with no hidden-to-hidden connection compute the feed-forward
    one's arithmetic (see :func:`settle_hidden`).

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        agg_indices: (num_positions,) aggregation indices; read only when
            ``per_node_aggregation``.
        global_agg_fn: One aggregation function for every hidden node.
        per_node_aggregation: Use ``agg_indices``.
        true_aggregation: Exact per-node aggregation rather than the approximation.
        num_aggregations: Size of the leading AGGREGATION_LIST slice.

    Returns:
        (n_samples, num_positions)
    """
    if global_agg_fn is not None:
        # Global aggregation: all nodes use same function
        weighted = inputs[:, :, None] * W1[None, :, :]
        return global_agg_fn(weighted)
    if per_node_aggregation and true_aggregation:
        # True per-node aggregation
        weighted = inputs[:, :, None] * W1[None, :, :]
        return grouped_aggregation_forward(weighted, agg_indices, num_aggregations)
    if per_node_aggregation:
        # Approximated aggregation: matmul + correction factors
        pre_h = inputs @ W1
        factors = compute_aggregation_correction_factors(
            agg_indices, W1.shape[0], num_aggregations
        )
        return pre_h * factors
    # Standard matmul (sum aggregation)
    return inputs @ W1


def hidden_activation(
    pre: jnp.ndarray,
    act_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Callable = jnp.tanh,
    palette: Optional[jnp.ndarray] = None,
    num_activations: int = 4,
) -> jnp.ndarray:
    """Hidden nodes' activation: per node when ``act_indices`` is given, else ``hidden_act_fn``.

    Args:
        pre: (n_samples, num_positions) pre-activation.
        act_indices: Optional (num_positions,) activation indices.
        hidden_act_fn: The activation of every node when ``act_indices`` is None.
        palette: Optional ACTIVATION_LIST indices; ``act_indices`` then holds those
            indices rather than positions 0..N-1.
        num_activations: Size of the leading ACTIVATION_LIST slice without a palette.

    Returns:
        (n_samples, num_positions)
    """
    if act_indices is None:
        return hidden_act_fn(pre)
    if palette is not None:
        return grouped_activation_forward_with_palette(pre, act_indices, palette)
    return grouped_activation_forward(pre, act_indices, num_activations)


def dense_substrate_forward(
    inputs: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    act_indices: Optional[jnp.ndarray] = None,
    agg_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Callable = jnp.tanh,
    output_act_fn: Callable = jax.nn.sigmoid,
    palette: Optional[jnp.ndarray] = None,
    num_activations: int = 4,
    global_agg_fn: Optional[Callable] = None,
    per_node_aggregation: bool = False,
    true_aggregation: bool = True,
    num_aggregations: int = 4,
) -> jnp.ndarray:
    """One feed-forward substrate's outputs: input -> hidden -> output.

    The input drive of :func:`input_drive`, the activation of :func:`hidden_activation`,
    then the output activation.

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        act_indices: Optional (num_positions,) activation indices.
        agg_indices: (num_positions,) aggregation indices; read only when
            ``per_node_aggregation``.
        hidden_act_fn: Hidden activation when ``act_indices`` is None.
        output_act_fn: Output activation.
        palette: Optional ACTIVATION_LIST indices; ``act_indices`` then holds those
            indices rather than positions 0..N-1.
        num_activations: Size of the leading ACTIVATION_LIST slice without a palette.
        global_agg_fn: One aggregation function for every hidden node.
        per_node_aggregation: Use ``agg_indices``.
        true_aggregation: Exact per-node aggregation rather than the approximation.
        num_aggregations: Size of the leading AGGREGATION_LIST slice.

    Returns:
        (n_samples, num_outputs)
    """
    pre_h = input_drive(inputs, W1, agg_indices, global_agg_fn, per_node_aggregation,
                        true_aggregation, num_aggregations)
    h = hidden_activation(pre_h, act_indices, hidden_act_fn, palette, num_activations)
    return output_act_fn(h @ W2)


# ============================================================================
# Recurrent substrate: one settling rule for every recurrent evaluation
# ============================================================================

#: How the settling steps of the recurrent path activate hidden nodes. ``every_iteration``
#: (the default) applies the configured activation, per node or the one global function, at
#: every step; ``tanh`` settles through tanh whatever the configuration says. There is no
#: ``initial_only``: the name described an initial activated step the forward pass did not take,
#: so in 0.1.0 that mode applied no configured activation at all; it is rejected with an explanation.
HH_ACTIVATION_MODES = ('every_iteration', 'tanh')


def node_activation(
    hh_activation_mode: str,
    act_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Optional[Callable] = None,
    palette: Optional[jnp.ndarray] = None,
    num_activations: int = 4,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """The activation a settling step applies to hidden nodes.

    Args:
        hh_activation_mode: One of :data:`HH_ACTIVATION_MODES`.
        act_indices: Optional (num_positions,) per-node activation indices.
        hidden_act_fn: The one activation of every node when there are no indices; tanh when
            None.
        palette: Optional ACTIVATION_LIST indices the indices refer to.
        num_activations: Size of the leading ACTIVATION_LIST slice without a palette.

    Returns:
        A function from pre-activation to activation. Without indices it is ``hidden_act_fn``
        itself (tanh when None), which :func:`hidden_activation` would apply alone: the same
        object on every call, so a registry function's settling step is made once
        (:func:`settle_hidden`).

    Raises:
        ValueError: An unknown mode.
    """
    if hh_activation_mode == 'tanh':
        return jnp.tanh
    if hh_activation_mode != 'every_iteration':
        raise ValueError(
            f"hh_activation_mode must be one of {HH_ACTIVATION_MODES}, "
            f"got {hh_activation_mode!r}"
        )
    hidden_fn = hidden_act_fn if hidden_act_fn is not None else jnp.tanh
    if act_indices is None:
        return hidden_fn
    return functools.partial(
        hidden_activation, act_indices=act_indices, hidden_act_fn=hidden_fn,
        palette=palette, num_activations=num_activations,
    )


#: Node functions whose settling step is made once and reused: the activation registry's
#: functions, which stay the same objects for the life of the module (``jnp.tanh`` among them).
_STABLE_NODE_FNS = tuple(ACTIVATION_FUNCTIONS.values())

#: Settling steps of the stable node functions, by (node function, dynamic recurrent input,
#: return_states, number of h->h aggregations); bounded by the registry and those settings.
_SETTLE_STEPS: Dict[Tuple[Any, ...], Callable] = {}


def _recurrent_input(hidden: jnp.ndarray, operands: Tuple[jnp.ndarray, ...], dynamic: bool,
                     num_hh_aggregations: int) -> jnp.ndarray:
    """The hidden-to-hidden input of one settling step (see :func:`settle_hidden`)."""
    if dynamic:
        safe_from, safe_to, effective_weights, hh_agg_indices, valid_mask = operands
        return scatter_aggregate_by_target(
            hidden[:, safe_from] * effective_weights, safe_to, hidden.shape[1],
            hh_agg_indices, valid_mask, num_hh_aggregations
        )
    (Whh,) = operands
    return hidden @ Whh


def _stable_settle_step(node_fn: Callable[[jnp.ndarray], jnp.ndarray], dynamic: bool,
                        return_states: bool, num_hh_aggregations: int) -> Callable:
    """The settling step of a stable node function, made once and reused.

    ``lax.scan`` traces its step through a cache keyed on the step's function object, and the
    generation's evaluation runs the scan under ``vmap`` without ``jit``, which compiles it
    once per traced step. A step made inside every call is a new object every time, so the
    scan is traced and compiled again on every call, once per generation. This one takes the
    drive and the recurrent operands from the carry and closes over nothing that changes
    between calls, so a run compiles it once per shape.
    """
    key = (node_fn, dynamic, return_states, num_hh_aggregations if dynamic else None)
    step = _SETTLE_STEPS.get(key)
    if step is None:
        def step(carry, _):
            hidden, drive, *operands = carry
            new = node_fn(drive + _recurrent_input(hidden, operands, dynamic, num_hh_aggregations))
            return (new, drive, *operands), (new if return_states else None)
        _SETTLE_STEPS[key] = step
    return step


def settle_hidden(
    drive: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    activate_time: int,
    node_fn: Callable[[jnp.ndarray], jnp.ndarray],
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
    return_states: bool = False,
) -> Any:
    """Settle a recurrent substrate's hidden layer.

    The one rule of every recurrent evaluation: ``h(0) = 0`` and, for ``activate_time``
    steps, ``h(t+1) = node_fn(drive + recurrent(h(t)))``. ``drive`` is the input drive of
    :func:`input_drive`, constant across steps and re-injected at each; ``recurrent`` sums the
    hidden-to-hidden contributions (or aggregates them per node in
    ``hh_aggregation_mode='dynamic'``); ``node_fn`` is the node's whole transformation,
    activation and any neuromodulation, the same one the feed-forward evaluation applies. With
    no hidden-to-hidden connection every step returns ``node_fn(drive)``, so the recurrent path
    computes the feed-forward substrate's arithmetic. On CPU the two agree bit for bit, except
    for compound activations such as ``resonator`` and ``osc_adapt``, which XLA can lower
    differently inside the settling loop: those differ by a float32 ulp.

    A node function of the activation registry (``jnp.tanh`` on the plain recurrent path, the
    configured function under a global activation, :func:`node_activation`) settles through a
    step made once (:func:`_stable_settle_step`), so repeated calls with the same shapes
    compile nothing new. Any other node function, such as the partials and closures of
    per-node activation and neuromodulation, which are new objects on every call, gets a step
    made per call and is traced and compiled again on every call.

    Args:
        drive: (n_samples, num_positions) input drive.
        from_indices: (max_sparse_conns,) source indices.
        to_indices: (max_sparse_conns,) target indices.
        hh_weights: (max_sparse_conns,) weights.
        valid_mask: (max_sparse_conns,) validity mask.
        activate_time: Number of settling steps, at least 1 (static).
        node_fn: Pre-activation to activation, per node.
        hh_agg_indices: Per-node hidden-to-hidden aggregation indices, for
            ``hh_aggregation_mode='dynamic'``.
        hh_aggregation_mode: ``'sum'`` or ``'dynamic'``.
        num_hh_aggregations: Number of aggregation functions for the dynamic mode.
        return_states: Also return the hidden state after each step (the substrate API's settling view);
            the settled state is computed the same way either way.

    Returns:
        (n_samples, num_positions) settled hidden activity; with ``return_states`` the pair
        ``(hidden, states)``, states (activate_time, n_samples, num_positions), the last equal to hidden.

    Raises:
        ValueError: ``activate_time`` below 1, which would return the zero state.
    """
    if activate_time < 1:
        raise ValueError(
            f"activate_time must be at least 1 on the recurrent path, got {activate_time}"
        )
    num_positions = drive.shape[1]
    hidden = jnp.zeros((drive.shape[0], num_positions))
    safe_from = jnp.clip(from_indices, 0, num_positions - 1)
    safe_to = jnp.clip(to_indices, 0, num_positions - 1)
    effective_weights = jnp.where(valid_mask, hh_weights, 0.0)

    dynamic = hh_aggregation_mode == 'dynamic' and hh_agg_indices is not None
    if dynamic:
        operands = (safe_from, safe_to, effective_weights, hh_agg_indices, valid_mask)
    else:
        # The sparse h->h weights as one dense matrix (Whh[i, j] = sum of weights i->j), so
        # each settling step is a matmul instead of a per-step scatter-add, the CPU bottleneck
        # that made recurrent topologies look infeasible. Equal to the scatter-add up to
        # float32 rounding (max abs diff ~5e-7); all zero without h->h connections.
        operands = (jnp.zeros((num_positions, num_positions)).at[safe_from, safe_to].add(
            effective_weights
        ),)

    if any(node_fn is fn for fn in _STABLE_NODE_FNS):
        step = _stable_settle_step(node_fn, dynamic, return_states, num_hh_aggregations)
        (hidden, *_), states = lax.scan(step, (hidden, drive) + operands, None,
                                        length=activate_time)
    else:
        # Any other node function, typically a partial or closure made per call: caching its
        # step would add an entry per call and never reuse one
        def step(hidden, _):
            new = node_fn(drive + _recurrent_input(hidden, operands, dynamic,
                                                   num_hh_aggregations))
            return new, (new if return_states else None)

        hidden, states = lax.scan(step, hidden, None, length=activate_time)
    if return_states:
        return hidden, states
    return hidden


#: The neuromodulated node's rules (neuromodulation.node_rule).
NEUROMODULATION_NODE_RULES = ('paper', 'hmr_full')


def modulatory_neuron_masks(weight_var: jnp.ndarray, active: jnp.ndarray, ratio: float) -> jnp.ndarray:
    """Each genome's modulatory neurons (neuromodulation mode modulatory_neurons): `ratio` of its own active
    positions, at least one, those of highest input-weight variance.

    0.1.0 gave every genome the same positions, the top of the population's mean variance, so a genome's network
    depended on the rest of the population and could use positions it did not have; and it counted them as a fraction
    of every grid position, so at depths 3 to 5 nearly every genome's whole active set was modulatory.

    Args:
        weight_var: (pop_size, positions) variance of each position's input weights.
        active: (pop_size, positions) active-position mask (concrete: the counts are taken on the host).
        ratio: mod_neuron_ratio, the modulatory fraction of a genome's active positions.

    Returns:
        (pop_size, positions) boolean mask.
    """
    score = jnp.where(active, weight_var, -jnp.inf)
    # Each position's rank by descending variance within its genome (a tie: the lower position first)
    rank = jnp.argsort(jnp.argsort(-score, axis=1), axis=1)
    # Counted in float64 on the host, as Python's int(n * ratio) and 0.1.0's grid count are: float32 can differ
    # (0.7 of 90 active positions is 63 in float32, 62 in float64)
    active_count = np.asarray(jnp.sum(active, axis=1))
    count = jnp.asarray(np.maximum(1, np.floor(active_count * float(ratio))).astype(np.int32))
    return (rank < count[:, None]) & active


def true_neuromodulation_node_fn(
    act_fn: Callable[[jnp.ndarray], jnp.ndarray],
    receptors: jnp.ndarray,
    base_gains: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    mod_strength: float,
    num_nt: int,
    modulation_mode: str,
    rule: str = 'paper',
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """A node under true neuromodulation, as a function of its pre-activation.

    The first three neurotransmitter channels select among a node's receptors,
    m = receptors[:, :3] @ nt[:3]; the fourth only inverts outputs (:func:`invert_outputs`).

    'paper' (the published rule; HMR's neuromodulation variant, and the rule of every EMR path):
    gating_only act(pre) * sigmoid(s m); otherwise gains clip(bg + s m, 0.1, 5) (NaN to 1) and
    bias s m, gain_bias_only act(gains pre + bias), full act(gains pre + bias) * sigmoid(m).
    'hmr_full' (HMR FULL's rule, which 0.1.0's single-task path computed): gains bg + s m, gate
    sigmoid(5 m); gating_only act(pre) * gate, gain_bias_only act(gains pre), full
    act(gains pre) * gate.

    Args:
        act_fn: The node's activation.
        receptors: (num_positions, num_nt) receptor densities.
        base_gains: (num_positions,) base gains.
        neurotransmitter: The task's neurotransmitter vector.
        mod_strength: Modulation strength s.
        num_nt: Number of neurotransmitter channels.
        modulation_mode: 'full', 'gating_only' or 'gain_bias_only'.
        rule: 'paper' or 'hmr_full'.

    Returns:
        The node function of the pre-activation.
    """
    k = min(3, num_nt)
    modulation = receptors[:, :k] @ neurotransmitter[:k]
    if rule == 'hmr_full':
        gains, gates = base_gains + mod_strength * modulation, jax.nn.sigmoid(modulation * 5.0)
        if modulation_mode == 'gating_only':
            return lambda pre: act_fn(pre) * gates
        if modulation_mode == 'gain_bias_only':
            return lambda pre: act_fn(gains * pre)
        return lambda pre: act_fn(gains * pre) * gates
    if modulation_mode == 'gating_only':
        gates = jax.nn.sigmoid(modulation * mod_strength)
        return lambda pre: act_fn(pre) * gates
    gains = jnp.clip(base_gains + mod_strength * modulation, 0.1, 5.0)
    gains = jnp.where(jnp.isnan(gains), 1.0, gains)
    bias = modulation * mod_strength
    if modulation_mode == 'gain_bias_only':
        return lambda pre: act_fn(gains * pre + bias)
    gates = jax.nn.sigmoid(modulation)
    return lambda pre: act_fn(gains * pre + bias) * gates


def invert_outputs(
    outputs: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    use_output_inversion: bool,
    num_nt: int,
    rule: str = 'paper',
) -> jnp.ndarray:
    """The fourth neurotransmitter channel's output inversion, clipped to [0, 1] under the paper's rule.

    Only a vector of four or more channels has an inversion channel. 0.1.0's multi-task path read
    nt[3] whatever the count, and JAX clamps an out-of-range index to the last element, so with two
    or three neurotransmitters a modulation channel also inverted the outputs.
    """
    if use_output_inversion and num_nt >= 4:
        invert = jnp.clip(neurotransmitter[3], 0.0, 1.0) if rule == 'paper' else neurotransmitter[3]
        return invert * outputs + (1 - invert) * (1 - outputs)
    return outputs


def sanitize_fitness(fitnesses: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
    """NaN fitness to -inf, as TensorNEAT's own pipeline does, and how many there were.

    The generation loop calls a jitted ``tell`` directly, which bypasses TensorNEAT's guard,
    so without this a NaN (an overflowing settling, a NaN target) reaches selection and the
    reported best fitness. A population without NaN is returned unchanged.

    Returns:
        ``(fitnesses, nan_count)``.
    """
    nan = jnp.isnan(fitnesses)
    count = int(jnp.sum(nan))
    if count == 0:
        return fitnesses, 0
    return jnp.where(nan, -jnp.inf, fitnesses), count


def fitness_statistics(fitnesses: jnp.ndarray,
                       nan_count: int) -> Tuple[float, float, float, float]:
    """Best, mean, min and std of a population as told, the last three over its finite entries.

    A NaN fitness is told as -inf (``sanitize_fitness``); counting it would report a mean of -inf.

    Args:
        fitnesses: (pop_size,) fitness after ``sanitize_fitness``.
        nan_count: How many entries ``sanitize_fitness`` replaced.

    Returns:
        ``(best, mean, min, std)``; the last three are NaN when no entry is finite.
    """
    best = float(jnp.max(fitnesses))
    finite = fitnesses if nan_count == 0 else fitnesses[jnp.isfinite(fitnesses)]
    if not finite.size:
        return best, float('nan'), float('nan'), float('nan')
    return best, float(jnp.mean(finite)), float(jnp.min(finite)), float(jnp.std(finite))


def species_summary(species_state: Any,
                    fitnesses: jnp.ndarray) -> Tuple[int, List[int], List[float]]:
    """The CPPN species of an evaluated population, as TensorNEAT's speciation assigned them.

    ``species_state`` is ``state.species`` of the state the population was evaluated under, before
    ``tell``: its ``idx2species`` holds each genome's species key and its ``member_count`` the size
    of each slot (NaN for an empty slot). After ``tell`` the same fields describe the next
    population, so passing the new state would pair this generation's fitness with the next
    generation's species.

    Args:
        species_state: TensorNEAT's species state (``state.species``).
        fitnesses: (pop_size,) fitness the population was told with.

    Returns:
        ``(num_species, sizes, best)``: the number of non-empty species, their sizes and the best
        fitness of each, in TensorNEAT's slot order.
    """
    counts, keys, members, fitness = (np.asarray(a) for a in jax.device_get((
        species_state.member_count, species_state.species_keys, species_state.idx2species,
        fitnesses)))
    live = np.flatnonzero(np.nan_to_num(counts, nan=0.0) > 0)
    sizes = [int(counts[s]) for s in live]
    best = [float(np.max(fitness[members == keys[s]])) for s in live]
    return len(live), sizes, best


def topology_summary(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    masks: jnp.ndarray,
    hh_counts: Optional[jnp.ndarray] = None,
) -> Dict[str, float]:
    """Population mean and spread of each genome's hidden nodes and expressed connections.

    A hidden node is an active substrate position. A connection is expressed when its weight
    survived pruning: a nonzero entry of W1 (input -> hidden) or W2 (hidden -> output), plus the
    genome's discovered h->h connections.

    Args:
        W1: (pop_size, num_inputs, positions) input weights as evaluated.
        W2: (pop_size, positions, num_outputs) output weights as evaluated.
        masks: (pop_size, positions) active-position mask.
        hh_counts: (pop_size,) h->h connection counts, or None on a feed-forward run.

    Returns:
        ``hidden_nodes_mean``, ``hidden_nodes_std``, ``connections_mean`` and ``connections_std``,
        the population fields of ``AlgorithmMetrics``.
    """
    connections = jnp.sum(W1 != 0, axis=(1, 2)) + jnp.sum(W2 != 0, axis=(1, 2))
    if hh_counts is not None:
        connections = connections + hh_counts
    hidden, connections = (np.asarray(a) for a in jax.device_get(
        (jnp.sum(masks, axis=1), connections)))
    return {
        'hidden_nodes_mean': float(np.mean(hidden)),
        'hidden_nodes_std': float(np.std(hidden)),
        'connections_mean': float(np.mean(connections)),
        'connections_std': float(np.std(connections)),
    }


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

    The standard path's settling rule (:func:`settle_hidden` with tanh): ``h(0) = 0``, then
    ``tanh(inputs @ W1 + recurrent(h))`` for ``activate_time`` steps. In 0.1.0 this
    evaluator kept an older rule (an activated initial state, a residual step, one step
    fewer), so the data-parallel strategy scored the same network differently from a
    single-device run.

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
        num_hidden: Number of hidden positions (unused; the shape carries it)

    Returns:
        Sum of squared errors for this genome on this data shard (scalar)
    """
    del num_hidden
    hidden = settle_hidden(inputs @ w1, hh_from, hh_to, hh_weights, hh_valid,
                           activate_time, jnp.tanh)
    outputs = jax.nn.sigmoid(hidden @ w2)
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
    num_cppn_outputs: int = 1,
) -> jnp.ndarray:
    """Query CPPN for a single level's positions.

    This is a level-specific version of batch_query_population_multi_source_chunked
    that queries only the positions for a single hierarchical level.

    A CPPN can emit more than one output -- notably
    ``dynamic_functions.mode='cppn_output'`` (weight + per-node activation index).
    That needs the trailing output axis preserved, so the multi-output query path is
    selected by ``num_cppn_outputs > 1`` -- the same condition
    ``discover_sparse_hh_vectorized_multi_hop`` uses. Sending a multi-output CPPN
    down the single-output branch folds the output axis into the position axis via
    ``.flatten()`` and produces ``(pop, sources, positions * num_outputs)``.

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
        num_cppn_outputs: CPPN output count. >1 selects the multi-output path

    Returns:
        If num_cppn_outputs == 1: (pop_size, num_sources, num_level_positions) array
        If num_cppn_outputs > 1: (pop_size, num_sources, num_level_positions,
            num_outputs) array with the weight in channel 0
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

            if num_cppn_outputs > 1:
                # Multi-output: returns (chunk_size, num_positions, num_outputs).
                # This path preserves the trailing axis; the single-output one
                # flattens it into the position axis.
                chunk_weights = batch_query_population_positions_multi_output(
                    state, chunk_cppns, source_coord, level_positions,
                    outgoing, cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=geometry_seeding_enabled
                )
            else:
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
    if num_cppn_outputs > 1:
        # (num_sources, pop, positions, outputs) -> (pop, num_sources, positions, outputs)
        return jnp.transpose(result, (1, 0, 2, 3))
    else:
        # (num_sources, pop_size, num_positions) -> (pop_size, num_sources, num_positions)
        return jnp.transpose(result, (1, 0, 2))


def compute_single_level_variance_chunked(
    query_slice_fn,
    grid: HierarchicalGridStructure,
    level: int,
    pop_size: int,
    max_elements: int,
) -> np.ndarray:
    """Variance for one level, computed in slabs so memory never scales with level size.

    `compute_single_level_variance` needs the whole of level+1 resident at once. For deep
    levels that is the allocation the old `max_lookahead_positions` cap was defending
    against -- and the cap's response was to substitute zeros, which silently killed every
    level below (see emr_hyperneat/tests/test_depth_lookahead.py).

    Chunking is exact here because each parent's variance depends only on its own 2x2
    block of children. Parent row r draws from child rows 2r and 2r+1, so a slab of parent
    rows maps to a *contiguous* span of the flattened child array and can be queried on its
    own. Results are concatenated in parent-row order, matching the unchunked layout
    exactly.

    Args:
        query_slice_fn: Callable (start, end) -> (pop_size, end - start) array of level+1
            weights for that flat position span.
        grid: Pre-computed grid structure.
        level: Current level (0-indexed); variance is over level+1's children.
        pop_size: Number of genomes in this chunk.
        max_elements: Soft ceiling on elements (pop_size x positions) held at once.

    Returns:
        Variance array (pop_size, num_cells_at_level), identical to the unchunked result.

    Raises:
        ValueError: If a single parent row cannot fit within `max_elements`. Returning
            zeros here is what caused the silent depth cap, so this fails loudly instead.
    """
    next_level_size = int(grid.level_sizes_static[level + 1])
    child_grid_size = int(np.sqrt(next_level_size))
    parent_grid_size = child_grid_size // 2

    # A parent row needs 2 child rows resident: 2 * child_grid_size * pop_size elements.
    elements_per_parent_row = 2 * child_grid_size * pop_size
    rows_per_chunk = max_elements // max(1, elements_per_parent_row)

    if rows_per_chunk < 1:
        raise ValueError(
            f"Cannot compute level {level} variance within the memory budget: one parent "
            f"row of level {level + 1} needs {elements_per_parent_row:,} elements "
            f"(2 child rows x {child_grid_size:,} cols x pop {pop_size}) but the budget is "
            f"{max_elements:,}. Reduce the population chunk size or raise the budget. "
            f"Substituting zeros here would silently truncate the substrate at level "
            f"{level}, which is the defect this path exists to prevent."
        )

    parts = []
    for row_start in range(0, parent_grid_size, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, parent_grid_size)
        n_parent_rows = row_end - row_start

        flat_start = 2 * row_start * child_grid_size
        flat_end = 2 * row_end * child_grid_size
        weights = query_slice_fn(flat_start, flat_end)

        blocks = np.asarray(weights).reshape(
            pop_size, n_parent_rows, 2, parent_grid_size, 2
        )
        blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(
            pop_size, n_parent_rows, parent_grid_size, 4
        )
        parts.append(blocks.var(axis=-1).reshape(pop_size, n_parent_rows * parent_grid_size))

    return np.concatenate(parts, axis=1)


#: Why a key is not a setting, for the keys configurations are known to use. Every other unknown key
#: gets the list of the block's settings.
_UNKNOWN_KEY_HINTS: Dict[Tuple[str, str], str] = {
    ('dynamic_functions', 'aggregation'): (
        "aggregation is its own block next to dynamic_functions in emr_hyperneat (HMR's aggregation "
        "variant read it nested); nested, it was ignored and the run used sum aggregation with a "
        "one-output CPPN"),
    ('dynamic_functions', 'activation_palette'): (
        "the palette is dynamic_functions.palette, a preset name or a list of ACTIVATION_LIST indices "
        "or names; this key was ignored, so the run had no palette"),
    ('dynamic_functions', 'global_activation'): (
        "the one hidden activation of mode 'global' is dynamic_functions.hidden_activation"),
    ('geometry_seeding', 'use_7d_inputs'): (
        "geometry seeding always queries 7-D rows (x1, y1, x2, y2, delta_x, delta_y, bias) and 5-D "
        "rows without it; nothing read this key, so use_7d_inputs false ran 7-D inputs"),
    ('dynamic_functions', 'live_strategy'): (
        "HMR's live palette strategy is not part of EMR; set algo.df_palette from the runner"),
    ('dynamic_functions', 'interp_method'): "the setting is dynamic_functions.interpretation",
    ('dynamic_functions', 'weight_interpretation_method'): (
        "the setting is dynamic_functions.interpretation"),
    ('dynamic_functions', 'method'): "the setting is dynamic_functions.interpretation",
    ('dynamic_functions', 'num_outputs'): (
        "the CPPN's output count follows from the configuration; dynamic_functions.num_cppn_outputs "
        "forces it"),
    ('dynamic_functions', 'global_function'): "the global aggregation is aggregation.global_function",
    ('aggregation', 'global_aggregation'): (
        "the function of aggregation mode 'global' is aggregation.global_function (the misspelled key "
        "was ignored, so such a configuration ran 'sum')"),
    ('aggregation', 'aggregation_func'): (
        "the function of aggregation mode 'global' is aggregation.global_function"),
    ('emr_hyperneat', 'extended'): (
        "the recurrence settings are emr_hyperneat.recurrence; this block was ignored, so the run was "
        "feed-forward"),
    ('emr_hyperneat', 'strategy'): "the setting is emr_hyperneat.multi_gpu_strategy",
    ('emr_hyperneat', 'num_cppn_outputs'): (
        "the CPPN's output count follows from dynamic_functions and aggregation; "
        "dynamic_functions.num_cppn_outputs forces it"),
    ('emr_hyperneat', 'hmr_hyperneat'): (
        "this is HMR's block name; EMR reads algorithm_params.emrhyperneat.emr_hyperneat"),
    ('emr_hyperneat', 'activate_time'): (
        "the settling steps are recurrence.activate_time or substrate.activate_time"),
    ('emr_hyperneat', 'pop_size'): (
        "the population size is algorithm_params.emrhyperneat.population_size (or neat.pop_size)"),
    ('emr_hyperneat', 'leo'): "link-expression output (LEO) is not part of EMR",
    ('recurrence', 'recurrence_preset'): (
        "the preset is recurrence.preset (HMR's extended block named it so)"),
    ('recurrence', 'cache_enabled'): "the setting is recurrence.hh_cache_enabled",
    ('recurrence', 'use_cache'): "the setting is recurrence.hh_cache_enabled",
    ('neuromodulation', 'gate_from_cppn'): (
        "EMR derives its gates from the incoming weights; HMR's neuromodulation variant evolved them "
        "as a CPPN output per connection, which is not part of EMR"),
    ('neuromodulation', 'static_gating'): (
        "a level is chosen with neuromodulation.mode or neuromodulation.preset"),
    ('neuromodulation', 'context_gating'): (
        "a level is chosen with neuromodulation.mode or neuromodulation.preset"),
    ('neuromodulation', 'modulatory_neurons'): (
        "a level is chosen with neuromodulation.mode or neuromodulation.preset"),
    ('neuromodulation', 'true_neuromodulation'): (
        "a level is chosen with neuromodulation.mode or neuromodulation.preset"),
    ('neuromodulation', 'num_nt'): "the setting is neuromodulation.num_nt_types",
    ('neuromodulation', 'context_derivation'): (
        "nothing ever read it (only tests set it); context gating takes its context from "
        "neuromodulation.context_source ('input', 'task_id', 'learned')"),
}


def _refuse_unknown_keys(section: str, block: Any, known: frozenset) -> None:
    """Raise on a key of a configuration block that EMR does not read.

    In 0.1.0 such a key was ignored, so a misspelled or misplaced setting ran the default without a
    word. The message names the right key when it is known (``_UNKNOWN_KEY_HINTS``), and the block's
    settings otherwise.

    Args:
        section: The block's name, as the message prints it.
        block: The block; anything but a dict is left to the parser.
        known: The keys EMR reads in this block.

    Raises:
        ValueError: The block has a key outside ``known``.
    """
    if not isinstance(block, dict):
        return
    unknown = sorted(k for k in block if k not in known)
    if unknown:
        key = unknown[0]
        hint = _UNKNOWN_KEY_HINTS.get((section, key), f"the settings of {section} are {sorted(known)}")
        more = f" (also not settings: {unknown[1:]})" if len(unknown) > 1 else ""
        raise ValueError(f"{section}.{key} is not a setting; {hint}{more}")


def legacy_level0_variance(level0_weights: jnp.ndarray) -> jnp.ndarray:
    """The original level-0 rule (``variance_rule: legacy``) for the paths that compute variance
    one level at a time.

    Streaming and depth_offload derive each level's variance from its children
    (``compute_single_level_variance``), which is the canonical rule. Under ``legacy`` the standard
    path instead takes one variance per genome over level 0's own four raw weights
    (``compute_hierarchical_variances_batch``), so level 0 subdivides all or nothing. This gives the
    other paths the same value, repeated over level 0's four cells so their per-parent arithmetic
    is unchanged. In 0.1.0 those paths ignored ``legacy``, with a warning at configuration and none
    when the mode changed at run time.

    Args:
        level0_weights: (pop_size, 4) raw CPPN weights from the first substrate input to the
            level-0 positions.

    Returns:
        (pop_size, 4) variance: each genome's single level-0 value in all four cells.
    """
    return jnp.broadcast_to(jnp.var(level0_weights, axis=1, keepdims=True), level0_weights.shape)


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
        enabled: Must agree with allow_hidden_to_hidden (False with it True is refused)
        allow_hidden_to_hidden: Discover hidden-to-hidden connections and settle the
            recurrent forward pass; the forward mode follows this flag
        allow_backward: Enable backward connections (y_source > y_target)
        allow_lateral: Enable same-layer connections (y_source == y_target)
        allow_self_loops: Enable self-connections (same x,y)
        iteration_level: Number of hops for multi-hop expansion
        activate_time: Forward pass iterations for signal propagation
        max_connections: Max connections per substrate (for vmap padding)

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
    iteration_level: int = 1  # 2 in 0.1.0 (see RECURRENCE_PRESETS)
    activate_time: Optional[int] = None  # None = auto-compute from depth
    max_connections: int = 10000

    # (use_vectorized_discovery lived here but was never read to branch
    # anything -- deleted as a dead knob; configs that still carry the key are
    # silently ignored.)
    max_sparse_conns: int = 10000

    # NEW: Multi-hop settings
    multi_hop_algorithm: str = "matrix_power"  # "matrix_power" or "fori_loop"
    hop_decay_factor: float = 0.8  # Weight decay per hop
    # Right operand of every multi-hop product: 'masked' composes the genome's own direct
    # connections (the documented formula); 'raw' multiplies by the raw weights over the
    # population union, as 0.1.0 did, kept to reproduce recorded level-2 runs.
    # Refused together with hh_discovery_mode 'per_genome' (the leak needs the union).
    multi_hop_second_factor: str = 'masked'

    # Caching settings. Off by default since 0.2.0: the cache indexes the discovered h->h set by
    # population slot and reuses it for up to hh_refresh_interval generations while reproduction
    # moves genomes between slots, so a genome is scored with another genome's recurrent
    # connections between refreshes. Set True only to reproduce a cache-on campaign.
    hh_cache_enabled: bool = False
    hh_refresh_interval: int = 5
    hh_mask_change_threshold: float = 0.1

    # Discovery algorithm toggle
    # - False (default): Sparse per-genome filtering (better for full_recurrent)
    # - True: Dense discovery without per-genome filtering (better for hybrid at low pop)
    use_dense_discovery: bool = False
    # How the h->h candidate pairs are queried:
    # - 'union': one CPPN query over the population's union of active positions, then each
    #   genome restricted to its own positions (the published path; cost ~ pop x U^2)
    # - 'per_genome': each genome queried over its own active positions only, in buckets of
    #   static size (cost ~ sum of n_i^2). Same connections and weights as 'union' at every
    #   iteration_level with multi_hop_second_factor 'masked'; the two settings are orthogonal
    #   (this one is the query shape, that one the algebra of the hops). Ignored
    #   under use_dense_discovery, where the union is part of the semantics, and by the pmap
    #   (multi-GPU) pipeline, which has its own inline discovery over the whole grid per genome
    #   at iteration level 1 (levels above 1 are refused there).
    # Default per_genome: on XOR, population 1000, an RTX 2080 Ti, cache off, it was faster than union
    # at every depth 1-7 and every seed (1.52x at the least, depth 7; 5-12x at depths 2-6) with identical
    # trajectories. Left unset, it resolves in __post_init__: 'per_genome', or 'union' under
    # multi_hop_second_factor 'raw', which exists only on the union path.
    hh_discovery_mode: Optional[str] = None

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
        if self.hh_discovery_mode is None:
            self.hh_discovery_mode = 'union' if self.multi_hop_second_factor == 'raw' else 'per_genome'
        if self.hh_discovery_mode not in ('union', 'per_genome'):
            raise ValueError(
                f"hh_discovery_mode must be 'union' or 'per_genome', got {self.hh_discovery_mode!r}"
            )
        if self.multi_hop_second_factor not in ('masked', 'raw'):
            raise ValueError(
                f"multi_hop_second_factor must be 'masked' or 'raw', got {self.multi_hop_second_factor!r}"
            )
        if self.hh_discovery_mode == 'per_genome' and self.multi_hop_second_factor == 'raw':
            raise ValueError(
                "multi_hop_second_factor 'raw' multiplies by the population union's weights and "
                "cannot be combined with hh_discovery_mode 'per_genome'"
            )


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

# The recurrent presets discover direct connections only (iteration level 1) since 0.2.0. A pre-registered
# study of the level (1,650 runs) found level 2 better in one (problem, preset) cell of 25 and worse in 11
# of its 60 tests (Holm p < 0.05), and its rule moved the presets to level 1. Level 2 stays available: set
# recurrence.iteration_level.
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
        iteration_level=1,
    ),
    'with_backward': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=True,
        allow_lateral=False,
        allow_self_loops=False,
        iteration_level=1,
    ),
    'with_lateral': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=False,
        allow_lateral=True,
        allow_self_loops=False,
        iteration_level=1,
    ),
    'with_self': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=False,
        allow_lateral=False,
        allow_self_loops=True,
        iteration_level=1,
    ),
    'full_recurrent': EMRConfig(
        enabled=True,
        allow_hidden_to_hidden=True,
        allow_backward=True,
        allow_lateral=True,
        allow_self_loops=True,
        iteration_level=1,
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

    # Return a copy to prevent mutation. dataclasses.replace copies every field and re-runs
    # __post_init__. 0.1.0's hand-written constructor listed 14 fields and dropped the rest
    # (use_dense_discovery, eval_chunk_size, multi_hop_chunk_size) from any preset that set them;
    # it would also drop hh_discovery_mode and multi_hop_second_factor.
    return dataclasses.replace(RECURRENCE_PRESETS[name])


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

    # Build CPPN inputs: the same rows as the single-output query (see cppn_query_rows).
    inputs_np = cppn_query_rows(
        source_coord, target_positions, outgoing, geometry_seeding_enabled, agent_coord
    )
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
    initial_depth: int = 0,
    legacy_level0_variance: bool = False,
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
            level_variances = compute_hierarchical_variances_batch(variance_source, h_grid, legacy_level0_variance=legacy_level0_variance)
            masks_A = compute_subdivision_masks_batch(
                level_variances, variance_threshold, h_grid, return_all_masks=False,
                initial_depth=initial_depth,
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

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def compute_multi_hop_connections_matrix_power(
    adjacency_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    weight_threshold: float,
    iteration_level: int,
    hop_decay_factor: float,
    max_weight: float,
    second_factor: str = 'masked',
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-hop h->h expansion by repeated matrix products.

    Let ``A = where(valid_mask, adjacency_weights, 0)`` be the genome's direct connections. For
    ``iteration_level`` k the function forms the hop terms ``decay^(h-1) A^h`` for h = 1..k and
    keeps, per pair, the term of largest magnitude (a max merge, not a sum: a strong direct
    connection is never replaced by a weaker indirect one). Each product is thresholded at
    ``weight_threshold`` before the decay is applied; the result is clipped to ``max_weight`` and a
    pair is valid where ``|A_total| > weight_threshold``.

    ``second_factor`` chooses the right operand of every product (a static argument):

    - ``'masked'`` (default): ``A`` itself, so every hop is an existing direct connection, the
      composite connections stay within the genome's own positions, and hop h carries
      ``decay^(h-1)`` as stated above. The callers then intersect the result with the connection
      class (a forward edge composed with a backward one may land lateral or on its own source).
    - ``'raw'``: 0.1.0's code: the raw ``adjacency_weights`` (at the union call site these span
      the population's union of active positions, so a genome's composite connections could end
      on positions it did not have active, ignore its connection class and depend on the other
      genomes), with the decayed product carried into the next hop (hop 3 carried ``decay^3``).
      Kept bit for bit to reproduce recorded runs.

    JIT-compatible because ``iteration_level`` and ``second_factor`` are static: the Python loop
    compiles away into a fixed graph.

    Args:
        adjacency_weights: CPPN-derived weights, shape (num_active, num_active).
        valid_mask: Boolean mask of the direct connections, same shape.
        weight_threshold: Minimum magnitude a product must reach to count as a connection.
        iteration_level: Number of hops k (static).
        hop_decay_factor: Per-hop attenuation; hop h is scaled by decay^(h-1).
        max_weight: Clip magnitude of the accumulated weights.
        second_factor: ``'masked'`` or ``'raw'`` (static).

    Returns:
        ``(A_total, multi_hop_valid)``, both of shape (num_active, num_active).
    """
    if second_factor not in ('masked', 'raw'):
        raise ValueError(f"second_factor must be 'masked' or 'raw', got {second_factor!r}")
    A_current = jnp.where(valid_mask, adjacency_weights, 0.0)
    second = A_current if second_factor == 'masked' else adjacency_weights
    A_total = A_current

    # Static unroll: compiles away because iteration_level is static.
    for hop in range(1, iteration_level):
        A_power = A_current @ second
        A_power = jnp.where(jnp.abs(A_power) > weight_threshold, A_power, 0.0)
        A_decayed = A_power * (hop_decay_factor ** hop)
        A_total = jnp.where(jnp.abs(A_decayed) > jnp.abs(A_total), A_decayed, A_total)
        A_current = A_power if second_factor == 'masked' else A_decayed

    A_total = jnp.clip(A_total, -max_weight, max_weight)
    multi_hop_valid = jnp.abs(A_total) > weight_threshold
    if second_factor == 'masked':
        # The expansion never removes a direct connection. A caller's valid_mask can mark a direct
        # connection with |w| <= weight_threshold (validity from a gate other than the weight band),
        # and the rule above dropped it at level >= 2. Under the band gate, the only one this package
        # applies, every direct cell has |w| > weight_threshold and |A_total| >= |w| (the merge only
        # replaces by a larger magnitude, and |w| <= max_weight survives the clip), so this changes
        # nothing there. 'raw' keeps 0.1.0's code.
        multi_hop_valid = multi_hop_valid | valid_mask
    return A_total, multi_hop_valid


def compute_multi_hop_connections_fori_loop(
    adjacency_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    weight_threshold: float,
    iteration_level: int,
    hop_decay_factor: float,
    max_weight: float,
    second_factor: str = 'masked',
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The computation of ``compute_multi_hop_connections_matrix_power`` with ``lax.fori_loop``.

    Works with a dynamic ``iteration_level``. The decay is a traced float32 power here and a Python
    float there, so the two implementations can differ in the last bit. ``second_factor`` as above.
    """
    if second_factor not in ('masked', 'raw'):
        raise ValueError(f"second_factor must be 'masked' or 'raw', got {second_factor!r}")
    A_init = jnp.where(valid_mask, adjacency_weights, 0.0)
    second = A_init if second_factor == 'masked' else adjacency_weights

    def hop_step(hop_idx, carry):
        A_current, A_total = carry
        A_power = A_current @ second
        A_power = jnp.where(jnp.abs(A_power) > weight_threshold, A_power, 0.0)
        A_decayed = A_power * (hop_decay_factor ** (hop_idx + 1))
        A_total = jnp.where(jnp.abs(A_decayed) > jnp.abs(A_total), A_decayed, A_total)
        return ((A_power if second_factor == 'masked' else A_decayed), A_total)

    _, A_total = lax.fori_loop(0, iteration_level - 1, hop_step, (A_init, A_init))

    A_total = jnp.clip(A_total, -max_weight, max_weight)
    multi_hop_valid = jnp.abs(A_total) > weight_threshold
    if second_factor == 'masked':
        # The expansion never removes a direct connection. A caller's valid_mask can mark a direct
        # connection with |w| <= weight_threshold (validity from a gate other than the weight band),
        # and the rule above dropped it at level >= 2. Under the band gate, the only one this package
        # applies, every direct cell has |w| > weight_threshold and |A_total| >= |w| (the merge only
        # replaces by a larger magnitude, and |w| <= max_weight survives the clip), so this changes
        # nothing there. 'raw' keeps 0.1.0's code.
        multi_hop_valid = multi_hop_valid | valid_mask
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


def _discover_sparse_hh_multi_hop_single_chunk(
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
    stats_out: Optional[Dict[str, Any]] = None,
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

    if getattr(config, 'hh_discovery_mode', 'union') == 'per_genome' and not config.use_dense_discovery:
        return _discover_sparse_hh_per_genome(
            state, cppns_transformed, h_grid, masks_A, band_threshold, max_weight, config,
            cppn_forward, verbose=verbose, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled, num_cppn_outputs=num_cppn_outputs,
            stats_out=stats_out,
        )

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
              f"hop_decay_factor={config.hop_decay_factor}, "
              f"multi_hop_second_factor={config.multi_hop_second_factor}")
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
    if stats_out is not None:
        stats_out['direct_per_genome'] = jnp.sum(direct_valid, axis=(1, 2))

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
                    config.iteration_level, config.hop_decay_factor, max_weight,
                    config.multi_hop_second_factor,
                )
        else:  # fori_loop
            def expand_single_genome(adjacency, valid):
                return compute_multi_hop_connections_fori_loop(
                    adjacency, valid, band_threshold,
                    config.iteration_level, config.hop_decay_factor, max_weight,
                    config.multi_hop_second_factor,
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

    if config.iteration_level > 1 and config.multi_hop_second_factor == 'masked':
        # Composites keep the preset's connection class: a forward edge composed with a backward
        # one lands lateral or on its own source, and those are dropped, so a with_backward network
        # holds only forward and backward edges at every level. 'raw' keeps 0.1.0's behavior bit for
        # bit.
        multi_hop_valid = multi_hop_valid & constraint_mask
        multi_hop_weights = jnp.where(multi_hop_valid, multi_hop_weights, 0.0)

    if stats_out is not None:
        # Direct pairs whose weight the max-magnitude merge swapped for a composite.
        direct_weights = jnp.where(direct_valid, active_to_active_weights, 0.0)
        stats_out['replaced_per_genome'] = jnp.sum(direct_valid & (multi_hop_weights != direct_weights), axis=(1, 2))
        # Before the max_sparse_conns cap (the extraction keeps the first pairs in flat order).
        stats_out['uncapped_per_genome'] = jnp.sum(multi_hop_valid, axis=(1, 2))

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




# Diagnostics of the last h->h discovery call on this process: per-genome direct and
# total connection counts, direct pairs whose weight the multi-hop merge replaced, active positions,
# the union size and the settings in force, and each genome's count before the max_sparse_conns cap
# with how many connections the cap dropped. Filled by discover_sparse_hh_vectorized_multi_hop for
# the union and per-genome paths (not the pmap pipeline); read by experiment runners for their
# records and by the standard path's metrics (truncation only).
_last_discovery_stats: Dict[str, Any] = {}
# The cap warning is given once per process; the counts are in every discovery's record.
_warned_sparse_cap = False


def _record_discovery_stats(result: "SparseHiddenConnections", masks_A: jnp.ndarray, union_active: int,
                            config: "EMRConfig", chunk_stats: List[Dict[str, Any]]) -> None:
    """Store the last discovery's per-genome counts in ``_last_discovery_stats`` (numpy)."""
    direct = replaced = None
    if chunk_stats and all('direct_per_genome' in s for s in chunk_stats):
        direct = np.concatenate([np.asarray(s['direct_per_genome']) for s in chunk_stats]).astype(np.int64)
    if chunk_stats and all('replaced_per_genome' in s for s in chunk_stats):
        replaced = np.concatenate([np.asarray(s['replaced_per_genome']) for s in chunk_stats]).astype(np.int64)
    uncapped = truncated = None
    if chunk_stats and all('uncapped_per_genome' in s for s in chunk_stats):
        uncapped = np.concatenate(
            [np.asarray(s['uncapped_per_genome']) for s in chunk_stats]).astype(np.int64)
        truncated = np.maximum(uncapped - int(config.max_sparse_conns), 0)
        global _warned_sparse_cap
        if not _warned_sparse_cap and int(truncated.max(initial=0)) > 0:
            _warned_sparse_cap = True
            import warnings
            warnings.warn(
                f"max_sparse_conns={config.max_sparse_conns} truncated the h->h connections of "
                f"{int(np.count_nonzero(truncated))} of {truncated.shape[0]} genomes in one "
                f"discovery (largest set {int(uncapped.max())}); the cap keeps a genome's first "
                "pairs in (source, target) position order, not its strongest. Raise "
                "recurrence.max_sparse_conns to keep them all. Per-genome counts: "
                "_last_discovery_stats['truncated_per_genome']; this warning is given once per "
                "process.", UserWarning, stacklevel=2)
    _last_discovery_stats.clear()
    _last_discovery_stats.update({
        'total_per_genome': np.asarray(result.num_valid).astype(np.int64),
        'direct_per_genome': direct,
        'replaced_per_genome': replaced,
        'uncapped_per_genome': uncapped,
        'truncated_per_genome': truncated,
        'max_sparse_conns': int(config.max_sparse_conns),
        'active_per_genome': np.asarray(jnp.sum(masks_A, axis=1)).astype(np.int64),
        'active_masks': np.asarray(masks_A),
        'union_active': int(union_active),
        'iteration_level': int(config.iteration_level),
        'hh_discovery_mode': getattr(config, 'hh_discovery_mode', 'union'),
        'multi_hop_second_factor': getattr(config, 'multi_hop_second_factor', 'raw'),
    })


def _validated_hh_discovery_mode(value: Any) -> Optional[str]:
    """Refuse an unknown h->h discovery mode instead of silently running the default.

    None (the key absent) is passed through: EMRConfig resolves it to 'per_genome', or to 'union'
    under multi_hop_second_factor 'raw'.
    """
    if value is None:
        return None
    if value not in ('union', 'per_genome'):
        raise ValueError(
            f"recurrence.hh_discovery_mode must be 'union' or 'per_genome', got {value!r}"
        )
    return value


_per_genome_discovery_cache: Dict[Tuple, Any] = {}
_per_genome_discovery_cache_lock = threading.Lock()
# Ceiling on the m x S x S pair block of one per-genome core call. The bound that applies is the row
# budget below (`_per_genome_rows_for`, at most 2^16 pairs per call); this ceiling only caps it. Counted
# from the input rows alone, 2^24 pairs would be about 470 MB; XLA's own accounting puts one such call at
# 6.7 to 10.9 GB (450 to 652 bytes per pair through the CPPN's 50-node intermediates), which is why it is
# not the operative bound. Slicing never changes a value: every genome's block depends on that genome
# alone.
_PER_GENOME_PAIR_BUDGET = 1 << 24


def _bucket_size_for(count: int) -> int:
    """The smallest static bucket size (8, 16, 32, ...) that holds ``count`` positions."""
    size = 8
    while size < count:
        size *= 2
    return size


_PER_GENOME_ROW_BUDGET = 1 << 16


def _per_genome_rows_for(size: int) -> int:
    """Static number of genomes per compiled call for a bucket size.

    `max(1, min(slice, 2^16 // size^2))`, with `slice` the pair budget's ceiling for the size, so
    every call of a bucket size has the same shape (rows, size, ...). A slice of whatever is left of
    the bucket changes its member count almost every generation and recompiles every gather, the
    core, the extract and every scatter: 150 to 275 executables per generation, 94-99% of the
    per-genome discovery time on a CPU (XOR, depth 4, populations 100 and 1000). With fixed rows a
    bucket size compiles once per run. The pad rows repeat a real
    member and are scattered to an out-of-range index, so results are unchanged.
    """
    slice_len = max(1, _PER_GENOME_PAIR_BUDGET // (size * size))
    return max(1, min(slice_len, _PER_GENOME_ROW_BUDGET // (size * size)))


# Pairs per block of the per-genome core's CPPN query. Above it, the query of a slice runs over blocks of
# source positions in sequence.
_PER_GENOME_BLOCK_PAIRS = 1 << 18


def _per_genome_block_for(size: int, rows: int) -> int:
    """Source positions per block of the per-genome core's CPPN query, for ``rows`` genomes of a bucket size.

    The core queries every (source, target) pair of its slice. From a bucket size of 1024 on a slice is one
    genome and its ``size^2`` pairs, 4.2 million at 2048: 1.89 GB of XLA-CPU temporaries in one call, where
    the union path queries one source position at a time (0.20 GB for 1000 genomes against 300 targets).
    Above ``_PER_GENOME_BLOCK_PAIRS`` pairs per call the query runs over blocks of source positions in
    sequence (``lax.map``), at most that many pairs each; at or below it, one block. Every pair's arithmetic
    is unchanged, so the values are too. Bucket sizes are powers of two, so the block divides the size.
    """
    if rows * size * size <= _PER_GENOME_BLOCK_PAIRS:
        return size
    block = size
    while block > 1 and rows * block * size > _PER_GENOME_BLOCK_PAIRS:
        block //= 2
    return block


def hh_pair_weights(outputs: jnp.ndarray, max_weight: float,
                    band_threshold: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """A hidden-to-hidden pair's weight and whether it passes the band: per-genome discovery's criterion.

    ``weight = tanh(cppn) * max_weight`` on the CPPN's weight output (channel 0, taken after the CPPN's own output
    transform, so the weights are bounded by ``tanh(1) * max_weight`` when that transform is a tanh, as it is by
    default); a pair is a candidate connection when ``|weight| > band_threshold``. The generation loop's per-genome
    core and the public substrate API both call this function, so an explorer shows the criterion the loop applied.

    Args:
        outputs: CPPN outputs at the pairs, ``(..., num_cppn_outputs)``.
        max_weight: Weight scale.
        band_threshold: Band threshold.

    Returns:
        ``(weights, passes)``, both of shape ``outputs.shape[:-1]``.
    """
    weights = jnp.tanh(outputs[..., 0]) * max_weight
    return weights, jnp.abs(weights) > band_threshold


def per_genome_pair_connections(
    out: jnp.ndarray,
    coords: jnp.ndarray,
    valid: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    allow_backward: bool,
    allow_lateral: bool,
    allow_self_loops: bool,
    iteration_level: int,
    hop_decay_factor: float,
    use_fori: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A bucket's hidden-to-hidden connections from the CPPN outputs at its genomes' position pairs.

    The arithmetic half of the per-genome discovery core (the other half is its CPPN query,
    :func:`_per_genome_pair_outputs`): the weight and band (:func:`hh_pair_weights`), the connection classes of the
    configuration (by y: backward ``y_source > y_target``, lateral equal, self-loops the same position), the pairs'
    validity, and the multi-hop expansion of the genome's own direct connections (masked second factor) whose
    composites keep the classes. The public substrate API runs this same function on the same shapes, so its
    connections are the loop's.

    Args:
        out: (m, size, size, outputs) CPPN outputs at (source, target) pairs.
        coords: (m, size, 2) the genomes' active positions; padding carries a real position's coordinates.
        valid: (m, size) False on padding.
        band_threshold, max_weight: As for :func:`hh_pair_weights`.
        allow_backward, allow_lateral, allow_self_loops: The configuration's connection classes.
        iteration_level: Hops composed (1: direct connections only).
        hop_decay_factor: Decay per hop.
        use_fori: The fori_loop expansion rather than the matrix power.

    Returns:
        ``(weights, valid, direct_valid, direct_weights)``, each (m, size, size): the connections after the
        expansion, and the direct ones (``direct_weights`` is zero off ``direct_valid``).
    """
    m, size = coords.shape[0], coords.shape[1]
    weights, weight_valid = hh_pair_weights(out, max_weight, band_threshold)

    y = coords[..., 1]
    y_valid = jnp.ones((m, size, size), dtype=bool)
    if not allow_backward:
        y_valid = y_valid & (y[:, :, None] <= y[:, None, :])
    if not allow_lateral:
        y_valid = y_valid & (y[:, :, None] != y[:, None, :])
    coord_match = jnp.all(coords[:, :, None, :] == coords[:, None, :, :], axis=-1)
    if allow_self_loops:
        y_valid = y_valid | coord_match
        self_mask = jnp.ones((m, size, size), dtype=bool)
    else:
        self_mask = ~coord_match
    pair_valid = valid[:, :, None] & valid[:, None, :]
    direct_valid = weight_valid & y_valid & self_mask & pair_valid
    direct_weights = jnp.where(direct_valid, weights, 0.0)

    if iteration_level > 1:
        expand = (compute_multi_hop_connections_fori_loop if use_fori
                  else compute_multi_hop_connections_matrix_power)

        def expand_one(w, v):
            return expand(w, v, band_threshold, iteration_level, hop_decay_factor,
                          max_weight, 'masked')

        # Always the masked second factor: the union path does the same under
        # multi_hop_second_factor 'masked' (the default), and 'raw' is refused with
        # per_genome in EMRConfig.__post_init__, which is why the compile-cache key of
        # the discovery core does not carry the value.
        mh_weights, mh_valid = jax.vmap(expand_one)(direct_weights, direct_valid)
        # Composites keep the preset's connection class, as in the union path.
        mh_valid = mh_valid & y_valid & self_mask
        mh_weights = jnp.where(mh_valid, mh_weights, 0.0)
    else:
        mh_weights = direct_weights
        mh_valid = direct_valid
    return mh_weights, mh_valid, direct_valid, direct_weights


def _per_genome_pair_outputs(
    state: Any,
    nodes: jnp.ndarray,
    conns: jnp.ndarray,
    conn_attrs: jnp.ndarray,
    node_attrs: jnp.ndarray,
    coords: jnp.ndarray,
    cppn_forward: Any,
    geometry_seeding_enabled: bool,
) -> jnp.ndarray:
    """The per-genome core's CPPN query: every genome's CPPN at every (source, target) pair of its positions.

    Rows ``(source, target, optional deltas, 1)``, the layout of ``cppn_query_rows``. From a bucket size of 1024 on
    the query runs over blocks of source positions in sequence (:func:`_per_genome_block_for`); every pair's
    arithmetic is unchanged.

    Args:
        state: TensorNEAT state.
        nodes, conns, conn_attrs, node_attrs: The m genomes' transformed CPPNs.
        coords: (m, size, 2) the genomes' positions.
        cppn_forward: ``cppn_forward(state, cppn_tuple, row)``.
        geometry_seeding_enabled: 7-D rows with the signed deltas.

    Returns:
        (m, size, size, outputs).
    """
    m, size = coords.shape[0], coords.shape[1]

    def one_genome(cppn_tuple, rows_g):
        return jax.vmap(lambda x: cppn_forward(state, cppn_tuple, x))(rows_g)

    def query(sources):
        # CPPN outputs of the pairs (source in `sources`, any target): (m, b, size, outputs)
        b = sources.shape[1]
        src = jnp.broadcast_to(sources[:, :, None, :], (m, b, size, 2))
        tgt = jnp.broadcast_to(coords[:, None, :, :], (m, b, size, 2))
        parts = [src, tgt]
        if geometry_seeding_enabled:
            parts.append(tgt[..., 0:1] - src[..., 0:1])
            parts.append(tgt[..., 1:2] - src[..., 1:2])
        parts.append(jnp.ones((m, b, size, 1), dtype=coords.dtype))
        rows = jnp.concatenate(parts, axis=-1).reshape(m, b * size, -1)
        out = jax.vmap(one_genome, in_axes=((0, 0, 0, 0), 0))(
            (nodes, conns, conn_attrs, node_attrs), rows
        )
        return out.reshape(m, b, size, -1)

    block = _per_genome_block_for(size, m)
    if block >= size:
        return query(coords)
    # Blocks of source positions in sequence, so one block's intermediates are live at a
    # time; the blocks are stacked back in source order.
    blocks = coords.reshape(m, size // block, block, 2).transpose(1, 0, 2, 3)
    out = lax.map(query, blocks)
    return out.transpose(1, 0, 2, 3, 4).reshape(m, size, size, -1)


def _get_per_genome_discovery_fn(
    device_id: int,
    cppn_forward: Any,
    size: int,
    band_threshold: float,
    max_weight: float,
    config: EMRConfig,
    geometry_seeding_enabled: bool,
    num_cppn_outputs: int,
):
    """Get or create the compiled per-bucket discovery core for one bucket size.

    The core computes, for a bucket of genomes that each own at most ``size`` active positions,
    the multi-hop h->h weights and validity over the genome's own position pairs. Its arithmetic
    is the union pipeline's, applied to the genome's own coordinates: the same
    ``cppn_forward(state, cppn_tuple, row)`` per pair with the row layout of ``cppn_query_rows``
    (source, target, optional deltas, bias one), tanh and ``max_weight``, the connection-class
    constraints of ``_get_constraint_mask_fn``, the band threshold, and the multi-hop expansion
    over the genome's existing direct connections. Pairs whose endpoints are padding are never valid.
    """
    key = (device_id, id(cppn_forward), size, float(band_threshold), float(max_weight),
           bool(config.allow_backward), bool(config.allow_lateral), bool(config.allow_self_loops),
           int(config.iteration_level), config.multi_hop_algorithm, float(config.hop_decay_factor),
           bool(geometry_seeding_enabled), int(num_cppn_outputs))
    with _per_genome_discovery_cache_lock:
        if key in _per_genome_discovery_cache:
            return _per_genome_discovery_cache[key]
        devices = jax.devices()
        device = devices[device_id] if device_id < len(devices) else devices[0]
        allow_backward = bool(config.allow_backward)
        allow_lateral = bool(config.allow_lateral)
        allow_self_loops = bool(config.allow_self_loops)
        iteration_level = int(config.iteration_level)
        hop_decay_factor = float(config.hop_decay_factor)
        use_fori = config.multi_hop_algorithm != "matrix_power"

        with jax.default_device(device):
            def core(state, nodes, conns, conn_attrs, node_attrs, coords, valid):
                # coords: (m, size, 2) float32, padded entries carry a real position's coordinates
                # valid:  (m, size) bool, False on padding
                out = _per_genome_pair_outputs(state, nodes, conns, conn_attrs, node_attrs, coords,
                                               cppn_forward, geometry_seeding_enabled)
                mh_weights, mh_valid, direct_valid, direct_weights = per_genome_pair_connections(
                    out, coords, valid, band_threshold, max_weight, allow_backward, allow_lateral,
                    allow_self_loops, iteration_level, hop_decay_factor, use_fori,
                )
                replaced = jnp.sum(direct_valid & (mh_weights != direct_weights), axis=(1, 2))
                return mh_weights, mh_valid, jnp.sum(direct_valid, axis=(1, 2)), replaced

            compiled = jax.jit(core, device=device)
        _per_genome_discovery_cache[key] = compiled
        return compiled


def _discover_sparse_hh_per_genome(
    state: Any,
    cppns_transformed: Tuple,
    h_grid: HierarchicalGridStructure,
    masks_A: jnp.ndarray,
    band_threshold: float,
    max_weight: float,
    config: EMRConfig,
    cppn_forward: Any,
    verbose: bool = False,
    device_id: int = 0,
    geometry_seeding_enabled: bool = False,
    num_cppn_outputs: int = 1,
    stats_out: Optional[Dict[str, Any]] = None,
) -> SparseHiddenConnections:
    """Sparse h->h discovery over each genome's own active positions (``hh_discovery_mode: per_genome``).

    The union pipeline queries every genome over the population's union of active positions
    (``pop x U x U`` pairs) and only then restricts each genome to its own positions. Here genomes
    are bucketed by their active count into static sizes 8, 16, 32, ...; each bucket is queried
    over its genomes' own positions (``m x S x S``), so the pair count follows the genomes' own
    active sets. The connections and weights are the union pipeline's for every genome, listed in
    the same order (ascending source position, then ascending target position), including the
    ``max_sparse_conns`` truncation rule: bit for bit at ``iteration_level`` 1, within a few float32
    ulps at level >= 2 (the products run over different array shapes) under the default
    ``multi_hop_second_factor: masked``, where both paths compose the genome's existing direct
    connections. ``raw`` (0.1.0's algebra, union path only) multiplied by the raw weights
    over the whole population union and made a genome's composites depend on the other genomes; it
    is refused together with this mode. Not used under ``use_dense_discovery``. Each bucket size is
    processed in slices of a fixed number of genomes (``_per_genome_rows_for``), padded with a
    repeated member whose rows are dropped at the scatter, so a size compiles once per run. From a
    bucket size of 1024 on, a slice's CPPN query runs over blocks of source positions in sequence
    (``_per_genome_block_for``, at most 2^18 pairs per block), so a call's temporaries stay near 120 MB
    whatever the bucket size (XLA-CPU accounting; 1.89 GB at size 2048 in one block).
    """
    pop_size = int(masks_A.shape[0])
    max_sparse_conns = config.max_sparse_conns
    devices = jax.devices()
    device = devices[device_id] if device_id < len(devices) else devices[0]
    all_positions = jax.device_put(h_grid.all_positions, device)
    masks_A = jax.device_put(masks_A, device)
    cppns_transformed = tuple(jax.device_put(arr, device) for arr in cppns_transformed)
    state = jax.tree.map(lambda x: jax.device_put(x, device), state)

    counts = np.asarray(jnp.sum(masks_A, axis=1))
    from_all = jnp.full((pop_size, max_sparse_conns), -1, dtype=jnp.int32)
    to_all = jnp.full((pop_size, max_sparse_conns), -1, dtype=jnp.int32)
    w_all = jnp.zeros((pop_size, max_sparse_conns), dtype=jnp.float32)
    direct_all = jnp.zeros((pop_size,), dtype=jnp.int32)
    replaced_all = jnp.zeros((pop_size,), dtype=jnp.int32)
    uncapped_all = jnp.zeros((pop_size,), dtype=jnp.int32)
    if pop_size == 0 or int(counts.max()) == 0:
        if stats_out is not None:
            stats_out['direct_per_genome'] = direct_all
            stats_out['replaced_per_genome'] = replaced_all
            stats_out['uncapped_per_genome'] = uncapped_all
        valid_mask = from_all >= 0
        return SparseHiddenConnections(from_all, to_all, w_all, valid_mask, jnp.sum(valid_mask, axis=1))

    sizes = np.array([_bucket_size_for(int(c)) for c in counts])
    if verbose or os.environ.get('DEBUG_PHASE2', '0') == '1':
        hist = {int(s): int(np.sum(sizes == s)) for s in np.unique(sizes)}
        print(f"[H->H Discovery] per-genome mode: active per genome mean={counts.mean():.1f} "
              f"max={int(counts.max())}, buckets {hist}")

    for size in sorted(set(int(s) for s in sizes)):
        bucket_members = np.nonzero(sizes == size)[0]
        core = _get_per_genome_discovery_fn(
            device_id, cppn_forward, size, band_threshold, max_weight, config,
            geometry_seeding_enabled, num_cppn_outputs,
        )
        extract_sparse_vmap = _get_extract_sparse_fn(device_id, size, max_sparse_conns)
        rows = _per_genome_rows_for(size)
        for start in range(0, bucket_members.shape[0], rows):
            real = bucket_members[start:start + rows]
            # Every slice of a bucket size has the static shape (rows, size, ...): the gather index
            # repeats the slice's first member into the pad rows and the scatter index sends them to
            # pop_size, out of range, which `mode='drop'` discards. m is a batch dimension only in
            # the core (every reduction is per genome), so the pad rows change no result.
            pad = rows - real.shape[0]
            gather = jnp.asarray(np.concatenate([real, np.full(pad, real[0])]), dtype=jnp.int32)
            scatter = jnp.asarray(np.concatenate([real, np.full(pad, pop_size)]), dtype=jnp.int32)
            with jax.default_device(device):
                idx = jax.vmap(lambda row: jnp.nonzero(row, size=size, fill_value=-1)[0])(masks_A[gather])
                valid = idx >= 0
                idx_safe = jnp.where(valid, idx, 0)
                coords = all_positions[idx_safe]  # (rows, size, 2)
                slice_cppns = tuple(arr[gather] for arr in cppns_transformed)
                mh_weights, mh_valid, direct_b, replaced_b = core(
                    state, slice_cppns[0], slice_cppns[1], slice_cppns[2], slice_cppns[3], coords, valid
                )
                from_b, to_b, w_b = extract_sparse_vmap(mh_valid, mh_weights, idx_safe)
                from_all = from_all.at[scatter].set(from_b.astype(jnp.int32), mode='drop')
                to_all = to_all.at[scatter].set(to_b.astype(jnp.int32), mode='drop')
                w_all = w_all.at[scatter].set(w_b, mode='drop')
                direct_all = direct_all.at[scatter].set(direct_b.astype(jnp.int32), mode='drop')
                replaced_all = replaced_all.at[scatter].set(replaced_b.astype(jnp.int32), mode='drop')
                uncapped_all = uncapped_all.at[scatter].set(
                    jnp.sum(mh_valid, axis=(1, 2)).astype(jnp.int32), mode='drop')

    valid_mask = from_all >= 0
    num_valid = jnp.sum(valid_mask, axis=1)
    if stats_out is not None:
        stats_out['direct_per_genome'] = direct_all
        stats_out['replaced_per_genome'] = replaced_all
        stats_out['uncapped_per_genome'] = uncapped_all
    if verbose or os.environ.get('DEBUG_PHASE2', '0') == '1':
        print(f"[H->H Discovery] COMPLETE (per-genome): total_h2h_conns={int(jnp.sum(num_valid))}, "
              f"mean_per_genome={float(jnp.mean(num_valid)):.1f}")
    return SparseHiddenConnections(
        from_indices=from_all, to_indices=to_all, weights=w_all, valid_mask=valid_mask, num_valid=num_valid,
    )


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
    discovery_population_chunk_size: Optional[int] = None,
) -> SparseHiddenConnections:
    """Population-chunked front for h->h discovery.

    The single-chunk pipeline materializes population-dense `(pop, A, A)`
    tensors over the active-set union -- recorded wall allocations of
    1000 x 692^2 x 4 and 1000 x 888^2 x 4 bytes. Per-genome work is
    independent once the union is fixed, so this wrapper (1) hoists the union
    computation so it stays population-global, then (2) slices the population
    through the unchanged pipeline and concatenates the sparse results.
    Chunking never changes values: with the union pinned, a slice of the full
    call and a call on the slice are the same computation.

    `discovery_population_chunk_size=None` sizes chunks from the measured
    working set (`execution.chunking.discovery_population_chunk`) and returns
    the whole population as one chunk whenever it fits, so every
    previously-feasible configuration runs the exact pre-wrapper path.
    """
    from emr_hyperneat.execution.chunking import (
        discovery_population_chunk,
    )

    pop_size = masks_A.shape[0]

    # Hoist the population-global union: computed here once, identical for
    # every chunk. (Inside the pipeline a None union is computed from the
    # masks it receives, which for a chunk would be the CHUNK's union --
    # smaller, and value-changing.)
    if global_union_active is None:
        global_union_active = jnp.any(masks_A, axis=0)
    num_active = int(jnp.sum(global_union_active))

    if discovery_population_chunk_size is None:
        chunk = discovery_population_chunk(pop_size, num_active)
    else:
        chunk = max(1, min(int(discovery_population_chunk_size), pop_size))
    if getattr(config, 'hh_discovery_mode', 'union') == 'per_genome' and not config.use_dense_discovery:
        # The per-genome route bounds its memory by the rows per bucket size, not by the union, so
        # the union-sized population split only multiplied its padded calls and changed the shapes
        # its eager steps see (bursts of recompiles in both modes on two RTX 2080 Ti: 36.6 s over
        # generations 2-29 at depth 5, population 1000). Whole population, one call; per-genome
        # results do not depend on how the population is sliced.
        chunk = pop_size

    if chunk >= pop_size:
        stats: Dict[str, Any] = {}
        result = _discover_sparse_hh_multi_hop_single_chunk(
            state, cppns_transformed, h_grid, masks_A,
            band_threshold, max_weight, config, cppn_forward,
            pop_chunk_size=pop_chunk_size, verbose=verbose,
            global_union_active=global_union_active, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled,
            num_cppn_outputs=num_cppn_outputs,
            stats_out=stats,
        )
        _record_discovery_stats(result, masks_A, num_active, config, [stats])
        return result

    if verbose:
        print(f"[H→H Discovery] population-chunked: pop={pop_size}, "
              f"chunk={chunk}, A={num_active}")

    parts = []
    chunk_stats: List[Dict[str, Any]] = []
    for start in range(0, pop_size, chunk):
        end = min(start + chunk, pop_size)
        chunk_cppns = tuple(arr[start:end] for arr in cppns_transformed)
        chunk_stats.append({})
        parts.append(_discover_sparse_hh_multi_hop_single_chunk(
            state, chunk_cppns, h_grid, masks_A[start:end],
            band_threshold, max_weight, config, cppn_forward,
            pop_chunk_size=pop_chunk_size, verbose=verbose and start == 0,
            global_union_active=global_union_active, device_id=device_id,
            geometry_seeding_enabled=geometry_seeding_enabled,
            num_cppn_outputs=num_cppn_outputs,
            stats_out=chunk_stats[-1],
        ))

    result = SparseHiddenConnections(
        from_indices=jnp.concatenate([p.from_indices for p in parts], axis=0),
        to_indices=jnp.concatenate([p.to_indices for p in parts], axis=0),
        weights=jnp.concatenate([p.weights for p in parts], axis=0),
        valid_mask=jnp.concatenate([p.valid_mask for p in parts], axis=0),
        num_valid=jnp.concatenate([p.num_valid for p in parts], axis=0),
    )
    _record_discovery_stats(result, masks_A, num_active, config, chunk_stats)
    return result


# ============================================================================
# Forward Pass (Three Modes)
# ============================================================================

class ForwardPassMode(Enum):
    """Forward pass execution modes."""
    DENSE_ONLY = "dense_only"       # Feedforward: W1 → tanh → W2 (no h→h)
    HYBRID_SPARSE_HH = "hybrid"     # Dense W1/W2 + sparse h→h
    # Not implemented: all-sparse would need sparse W1/W2. 0.1.0 accepted it and ran it with
    # no sparse h→h at all (every h→h gate in the generation loop tests
    # `== HYBRID_SPARSE_HH`), i.e. as DENSE_ONLY under a recurrent name. set_forward_mode and
    # forward_unified_extended refuse it; the member stays so the enum's values do not change.
    # Unreachable from config.
    FULL_SPARSE = "full_sparse"


def _explicit_activate_time(key: str, value: Any) -> int:
    """An explicit settling-step count where hidden-to-hidden connections are on: an integer of at least 1.

    One rule and one message for the configuration (``substrate.activate_time``, ``recurrence.activate_time``,
    `EMRHyperNEAT.set_recurrence_config`) and for `emr_hyperneat.substrate.SubstrateConfig`. A bool is refused,
    although Python counts it an int, and so is a float, whole or not.

    Args:
        key: The value's name in the message.
        value: The configured count.

    Returns:
        The count as a Python int.

    Raises:
        ValueError: A bool, a value that is not an integer, or a count below 1.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{key} must be an integer of at least 1 when hidden-to-hidden connections are on, "
                         f"got {value!r}")
    return int(value)


def check_recurrence_switch(enabled: Any, allow_hidden_to_hidden: Any) -> None:
    """Refuse ``enabled: false`` with ``allow_hidden_to_hidden: true``.

    The forward mode follows ``allow_hidden_to_hidden`` (`resolve_recurrence`), so such a block
    switches recurrence off and runs recurrent all the same. `EMRHyperNEAT._parse_extended_config`
    applies it to a recurrence block without a preset (a preset sets both fields consistently) and
    `EMRHyperNEAT.set_recurrence_config` to its config.

    Args:
        enabled: The recurrence switch.
        allow_hidden_to_hidden: Whether hidden-to-hidden connections are discovered.

    Raises:
        ValueError: ``enabled`` false while ``allow_hidden_to_hidden`` is true.
    """
    if allow_hidden_to_hidden and not enabled:
        raise ValueError(
            "recurrence enabled: false with allow_hidden_to_hidden: true is contradictory: the "
            "block switches recurrence off but turns hidden-to-hidden connections on, and the run "
            "would follow allow_hidden_to_hidden. Set enabled: true (or leave it out) for a "
            "recurrent run, or allow_hidden_to_hidden: false for a feed-forward one.")


def resolve_recurrence(config: Any, max_depth: Optional[int],
                       explicit: Dict[str, Any]) -> Tuple[ForwardPassMode, int, Optional[Any]]:
    """The forward mode, iteration level and settling steps of a recurrence configuration.

    The one rule of `EMRHyperNEAT._parse_extended_config` and `EMRHyperNEAT.set_recurrence_config`. Hidden-to-hidden
    connections on (``allow_hidden_to_hidden``) give the recurrent forward mode. Their settling steps are an
    explicit count (``explicit``, every value an integer of at least 1, all equal), else the configuration's own
    count (a preset's: ``full_recurrent`` carries 20), else ``2 (2**max_depth + 1)``. Without them nothing settles:
    any value is accepted, a count the configuration carries is kept as the attribute (as in 0.1.0), and none is
    used. Only ``allow_hidden_to_hidden``, ``iteration_level`` and ``activate_time`` are read, so another dataclass
    with these fields serves (the timing harness passes HMR's UnifiedExtendedConfig).

    Args:
        config: The recurrence configuration, or None for feed-forward.
        max_depth: The grid depth; None before `create_config`, when no count is derived.
        explicit: The counts set by name, None where a name is not set; their names are the messages' names.

    Returns:
        ``(forward_mode, iteration_level, activate_time)``; ``activate_time`` None leaves the current value.

    Raises:
        ValueError: With hidden-to-hidden connections on, an explicit count that is not an integer of at least 1,
            or two explicit counts that differ.
    """
    if config is None:
        return ForwardPassMode.DENSE_ONLY, 0, None
    level = config.iteration_level
    if not config.allow_hidden_to_hidden:
        return ForwardPassMode.DENSE_ONLY, level, config.activate_time
    given = {key: _explicit_activate_time(key, value) for key, value in explicit.items() if value is not None}
    if len(set(given.values())) > 1:
        raise ValueError(" and ".join(f"{key}={explicit[key]!r}" for key in given)
                         + " disagree: both set the settling steps, so give them the same value or set only one")
    if given:
        steps = next(iter(given.values()))
    elif config.activate_time is not None:
        steps = int(config.activate_time)
    elif max_depth is not None:
        steps = 2 * (2 ** max_depth + 1)
    else:
        steps = None
    return ForwardPassMode.HYBRID_SPARSE_HH, level, steps


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
        raise NotImplementedError(
            "ForwardPassMode.FULL_SPARSE is not implemented (all-sparse needs sparse W1/W2); "
            "use HYBRID_SPARSE_HH")

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
        # Precompute sparse h->h weight matrix once; each step is a BLAS matmul instead
        # of a per-step scatter-add (CPU bottleneck). Identical math (max abs diff ~5e-7).
        Whh = jnp.zeros((num_positions, num_positions)).at[safe_from, safe_to].add(effective_weights)

        def hh_step(hidden, _):
            h_delta = hidden @ Whh
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
    computation -- same peak memory (one chunk's lanes live at a time), one
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
    pop = arrays[0].shape[0]
    num_chunks = -(-pop // chunk)
    padded = num_chunks * chunk

    def stack(a: jnp.ndarray) -> jnp.ndarray:
        if padded != pop:
            pad_width = ((0, padded - pop),) + ((0, 0),) * (a.ndim - 1)
            a = jnp.pad(a, pad_width)
        return a.reshape((num_chunks, chunk) + a.shape[1:])

    stacked = tuple(stack(a) for a in arrays)
    mapped = jax.lax.map(lambda xs: jax.vmap(eval_fn)(*xs), stacked)
    return mapped.reshape(padded)[:pop]


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("start", "end"))
def _accumulate_level_slice_donated(
    acc: jnp.ndarray, block: jnp.ndarray, *, start: int, end: int
) -> jnp.ndarray:
    """Slice-set with the accumulator buffer donated.

    Streaming keeps W1/W2 fully device-resident and updates them per level with
    functional `.at[...].set`, which holds input and output alive at once --
    transiently doubling W1, the 786,432,000 B level-7 operand that kept forced
    streaming OOMing at depth 7 even after the query was chunked.
    Donating the accumulator lets XLA
    write in place. Donation aliases buffers; it never changes values. The
    caller must immediately rebind its reference, and does. Static slice bounds
    retrace once per level -- bounded by the depth.
    """
    return acc.at[:, :, start:end].set(block)


def _accumulate_level_slice(
    acc: jnp.ndarray, block: jnp.ndarray, start: int, end: int
) -> jnp.ndarray:
    """Backend-dispatching level-slice accumulate.

    CPU ignores donation and warns per trace, so the donated variant is used
    only where it buys memory (GPU). Both variants compute the same values.
    """
    if jax.default_backend() == "gpu":
        return _accumulate_level_slice_donated(acc, block, start=int(start), end=int(end))
    return acc.at[:, :, start:end].set(block)


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
    hh_activation_mode: str = 'every_iteration',
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
    ih_agg_indices: Optional[jnp.ndarray] = None,
    ih_global_agg_fn: Optional[Callable] = None,
    ih_per_node_aggregation: bool = False,
    ih_true_aggregation: bool = True,
    num_ih_aggregations: int = 4,
) -> jnp.ndarray:
    """Vmap-friendly hybrid forward pass with dynamic activation and aggregation.

    :func:`settle_hidden` with the input drive of :func:`input_drive` and the node function of
    :func:`node_activation`: without hidden-to-hidden connections this is
    :func:`dense_substrate_forward` with the same arguments.

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        from_indices: (max_sparse_conns,) source indices
        to_indices: (max_sparse_conns,) target indices
        hh_weights: (max_sparse_conns,) weights
        valid_mask: (max_sparse_conns,) validity mask
        activate_time: Number of settling steps
        act_indices: Optional (num_positions,) per-node activation indices.
        hidden_act_fn: The one hidden activation when there are no indices ('global'
            mode); tanh when None.
        output_act_fn: Output activation function. Defaults to sigmoid.
        num_activations: Number of activation functions for grouped forward.
        palette: Optional custom activation palette indices.
        hh_activation_mode: 'every_iteration' (the configured activation at every step) or
            'tanh' (tanh at every step, whatever is configured).
        hh_agg_indices: Optional (num_positions,) per-node H→H aggregation indices.
        hh_aggregation_mode: 'sum' (default) or 'dynamic' for per-node aggregation.
        num_hh_aggregations: Number of aggregation functions for dynamic H→H.
        ih_agg_indices: Optional (num_positions,) per-node input-to-hidden aggregation.
        ih_global_agg_fn: One input-to-hidden aggregation for every node.
        ih_per_node_aggregation: Use ``ih_agg_indices``.
        ih_true_aggregation: Exact per-node aggregation rather than the approximation.
        num_ih_aggregations: Number of input-to-hidden aggregation functions.

    Returns:
        (n_samples, num_outputs)
    """
    drive = input_drive(inputs, W1, ih_agg_indices, ih_global_agg_fn, ih_per_node_aggregation,
                        ih_true_aggregation, num_ih_aggregations)
    node_fn = node_activation(hh_activation_mode, act_indices, hidden_act_fn, palette,
                              num_activations)
    hidden = settle_hidden(drive, from_indices, to_indices, hh_weights, valid_mask,
                           activate_time, node_fn, hh_agg_indices, hh_aggregation_mode,
                           num_hh_aggregations)
    output_fn = output_act_fn if output_act_fn is not None else jax.nn.sigmoid
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
    hh_activation_mode: str = 'every_iteration',
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
    ih_agg_indices: Optional[jnp.ndarray] = None,
    ih_global_agg_fn: Optional[Callable] = None,
    ih_per_node_aggregation: bool = False,
    ih_true_aggregation: bool = True,
    num_ih_aggregations: int = 4,
    node_rule: str = 'paper',
) -> jnp.ndarray:
    """Vmap-friendly hybrid forward pass with TRUE neuromodulation.

    :func:`settle_hidden` whose node function is the neuromodulated node of the feed-forward
    evaluation (:func:`true_neuromodulation_node_fn`, under ``node_rule``): gain inside the
    activation and gate after
    it, at every settling step. Without hidden-to-hidden connections this is the feed-forward
    neuromodulated substrate exactly.

    Args:
        inputs: (n_samples, num_inputs)
        W1: (num_inputs, num_positions)
        W2: (num_positions, num_outputs)
        from_indices: (max_sparse_conns,) source indices
        to_indices: (max_sparse_conns,) target indices
        hh_weights: (max_sparse_conns,) weights
        valid_mask: (max_sparse_conns,) validity mask
        activate_time: Number of settling steps
        receptor_densities: (num_positions, num_nt) per-node receptor densities
        base_gains: (num_positions,) per-node base gains
        neurotransmitter: (num_nt,) NT vector for task
        mod_strength: Modulation strength factor
        modulation_mode: 'full', 'gating_only', or 'gain_bias_only'
        use_output_inversion: Whether NT4 controls output inversion
        num_nt: Number of neurotransmitter types
        act_indices: Optional (num_positions,) per-node activation indices.
        hidden_act_fn: The one hidden activation when there are no indices; tanh when None.
        output_act_fn: Output activation function. Defaults to sigmoid.
        num_activations: Number of activation functions for grouped forward.
        palette: Optional custom activation palette indices.
        hh_activation_mode: 'every_iteration' or 'tanh'.
        hh_agg_indices: Optional (num_positions,) per-node H→H aggregation indices.
        hh_aggregation_mode: 'sum' (default) or 'dynamic' for per-node aggregation.
        num_hh_aggregations: Number of aggregation functions for dynamic H→H.
        ih_agg_indices, ih_global_agg_fn, ih_per_node_aggregation, ih_true_aggregation,
            num_ih_aggregations: input-to-hidden aggregation, as in :func:`input_drive`.

    Returns:
        (n_samples, num_outputs)
    """
    act_fn = node_activation(hh_activation_mode, act_indices, hidden_act_fn, palette,
                             num_activations)
    drive = input_drive(inputs, W1, ih_agg_indices, ih_global_agg_fn, ih_per_node_aggregation,
                        ih_true_aggregation, num_ih_aggregations)
    hidden = settle_hidden(
        drive, from_indices, to_indices, hh_weights, valid_mask, activate_time,
        true_neuromodulation_node_fn(act_fn, receptor_densities, base_gains, neurotransmitter,
                                     mod_strength, num_nt, modulation_mode, node_rule),
        hh_agg_indices, hh_aggregation_mode, num_hh_aggregations,
    )
    output_fn = output_act_fn if output_act_fn is not None else jax.nn.sigmoid
    return invert_outputs(output_fn(hidden @ W2), neurotransmitter, use_output_inversion, num_nt,
                          node_rule)


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
    hh_activation_mode: str = 'every_iteration',
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
    fitness_mode: str = 'mse',
    node_rule: str = 'paper',
) -> float:
    """Fitness of one network under true neuromodulation (see
    :func:`forward_hybrid_vmapped_neuromodulated`)."""
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
        node_rule=node_rule,
    )
    return _eval_single_fitness(outputs, targets, fitness_mode)


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
    hh_activation_mode: str = 'every_iteration',
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
    fitness_mode: str = 'mse',
) -> float:
    """Fitness of one network with dynamic functions (see
    :func:`forward_hybrid_vmapped_dynamic`)."""
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
    return _eval_single_fitness(outputs, targets, fitness_mode)


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

#: What a strategy name means to EMRHyperNEAT, through the constructor and the configuration key
#: multi_gpu_strategy alike: the five primary strategies, then the aliases. In 0.1.0 the constructor
#: used the base class's normalization instead, which sent FULL_PIPELINE_PARALLEL, MULTI_GPU and
#: DATA_PARALLEL to PIPELINE_CHUNKED and CPPN_CHUNKED to POSITION_SHARDING_CHUNKED and kept the other
#: aliases, all strategies this class has no route for, so such a run took the one-device loop under
#: the requested name.
STRATEGY_NAME_MAP: Dict[str, MultiGPUStrategy] = {
    'SINGLE_GPU': MultiGPUStrategy.SINGLE_GPU,
    'FULL_PIPELINE_PARALLEL': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
    'STREAMING': MultiGPUStrategy.STREAMING,
    'EVAL_ONLY_PARALLEL': MultiGPUStrategy.EVAL_ONLY_PARALLEL,
    'POPULATION_PARALLEL_PROCESS': MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
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
          random_scope: 'genome'     # random modes: genome (per slot, per run) or shared (0.1.0's
                                     # draw: one assignment for every genome and run)
          index_decode: 'tanh'       # tanh (0.1.0's; a palette of 9+ never reaches its end entries)
                                     # or uniform (every entry, equal bins); aggregation has its own
          palette: 'sin_only'        # Optional: use specific activation subset
          hh_activation_mode: 'every_iteration'  # every_iteration or tanh (tanh on h→h)

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

    #: Accepted values for `dynamic_functions.mode`. Anything else used to fall
    #: through the activation dispatch and silently run the fixed tanh/sigmoid
    #: baseline, so `create_config` now rejects it. 'modular' is in the set because
    #: it parses, but it warns: only the frozen HMR snapshot dispatches it.
    DYNAMIC_FUNCTION_MODES = frozenset({
        'disabled',
        'global',
        'cppn_output',
        'weight_interpretation',
        'random_fixed',
        'random_generation',
        'modular',
    })
    #: Keys of the neuromodulation block EMR reads (_parse_neuromodulation_config). Any other key
    #: raises; 0.1.0 ignored it. `enabled` is read only to refuse a contradiction with the mode.
    NEUROMODULATION_KEYS = frozenset({
        'mode', 'preset', 'enabled', 'gate_threshold', 'gate_scaling', 'gate_hardness', 'context_dim',
        'context_influence', 'context_source', 'mod_neuron_ratio', 'mod_connection_type', 'mod_decay',
        'num_nt_types', 'modulation_strength', 'receptor_from_weight', 'receptor_derivation',
        'modulation_mode', 'use_output_inversion', 'branch_gating', 'use_self_connection_query',
        'node_rule',
    })
    #: The neuromodulation modes the evaluation dispatches, one level each (context gating carries
    #: its base gates, XdG-style).
    NEUROMODULATION_MODES = ('disabled', 'static_gating', 'context_gating', 'modulatory_neurons',
                             'true_neuromodulation')
    #: Keys of the multitask block (MultiTaskConfig). Any other key raises; 0.1.0 ignored it. Eight
    #: of these are accepted only at their default value, because nothing in EMR applies them.
    MULTITASK_KEYS = frozenset({
        'enabled', 'num_tasks', 'task_names', 'nt_per_task', 'fitness_aggregation', 'task_weights',
        'joint_evolution', 'orthogonality_bonus', 'specialization_bonus', 'hidden_activation',
        'per_task_activation', 'fitness_mode', 'modulation_mode', 'modulation_strength_override',
        'two_module_mode', 'generalist_bonus_type', 'generalist_bonus_weight', 'generalist_threshold',
        'softmin_temperature', 'modulation_penalty', 'specialization_bonus_weight',
        'confusion_eval_frequency', 'orthogonality_bonus_weight', 'orthogonality_metric',
        'branch_gating_mode',
    })
    #: Keys of the emr_hyperneat block EMR reads (the base class, this class and
    #: parse_execution_config). Any other key raises; 0.1.0 ignored it (the tests alone set three
    #: that nothing read: `extended`, so two "recurrence" tests ran feed-forward; `strategy`;
    #: `pop_size`).
    EMR_HYPERNEAT_KEYS = frozenset({
        # emrhyperneat_base.py
        'band_threshold', 'division_threshold', 'extra_randkey_split', 'fitness_mode',
        'fuse_w1_computation', 'initial_depth', 'iteration_level', 'max_depth', 'max_weight',
        'population_chunk_size', 'skip_unused_masks', 'sparse_forward_threshold', 'variance_rule',
        'variance_threshold', 'verbose',
        # this class
        'aggregation', 'dynamic_functions', 'fitness_rescale_variance', 'geometry_seeding',
        'locality_radius', 'multi_gpu_strategy', 'multitask', 'neuromodulation', 'recurrence',
        'streaming_verbose',
        # execution.parse_execution_config
        'execution', 'enable_streaming',
    })
    #: Keys of the recurrence block EMR reads (_parse_extended_config). Any other key raises, except
    #: `use_vectorized_discovery`, retired: it was stored and never consulted, so ignoring it changes
    #: no result, and older configurations still carry it (TestRetiredKnobs).
    RECURRENCE_KEYS = frozenset({
        'preset', 'enabled', 'allow_hidden_to_hidden', 'allow_backward', 'allow_lateral',
        'allow_self_loops', 'iteration_level', 'activate_time', 'max_connections', 'max_sparse_conns',
        'multi_hop_algorithm', 'hop_decay_factor', 'multi_hop_second_factor', 'hh_cache_enabled',
        'hh_refresh_interval', 'hh_mask_change_threshold', 'use_dense_discovery', 'hh_discovery_mode',
        'eval_chunk_size', 'use_vectorized_discovery',
    })
    #: Keys of the dynamic_functions block EMR reads (_parse_dynamic_functions_config). Any other key
    #: raises; 0.1.0 ignored it, and a run then lacked what its configuration asked for (HMR's
    #: aggregation runners nest `aggregation` here; others set `activation_palette`).
    #: `palette_evolution` is accepted only as {'enabled': False}, which the per-node paper's
    #: runners carry: palette evolution is not part of EMR.
    DYNAMIC_FUNCTIONS_KEYS = frozenset({
        'mode', 'random_seed', 'random_scope', 'hidden_activation', 'output_activation',
        'num_activations', 'hh_activation_mode', 'palette', 'interpretation', 'sparsity',
        'critical_periods', 'modular', 'cppn_output_indices', 'num_cppn_outputs', 'palette_evolution',
        'index_decode',
    })
    #: Keys of the blocks nested in dynamic_functions.
    CPPN_OUTPUT_INDICES_KEYS = frozenset({'weight', 'activation', 'aggregation'})
    SPARSITY_KEYS = frozenset({'level_0', 'level_1', 'level_2_plus', 'wta_k_percent'})
    CRITICAL_PERIODS_KEYS = frozenset({'enabled', 'phase1_end', 'phase2_end', 'min_plasticity'})
    MODULAR_KEYS = frozenset({'activation_method', 'sparsity_method', 'scaling_method',
                              'aggregation_method', 'sparsity'})
    #: Keys of the aggregation block EMR reads; any other key raises (0.1.0 ignored it).
    AGGREGATION_KEYS = frozenset({
        'mode', 'random_seed', 'random_scope', 'global_function', 'num_aggregations', 'interpretation',
        'use_true_aggregation', 'hh_aggregation_mode', 'index_decode',
    })
    #: Accepted values for `aggregation.mode`; anything else raises (0.1.0 ran sum).
    AGGREGATION_MODES = frozenset({
        'disabled', 'global', 'cppn_output', 'weight_interpretation', 'random_fixed', 'random_generation',
        'modular',
    })
    #: Keys of the geometry_seeding block EMR reads; any other key raises (0.1.0 ignored it).
    GEOMETRY_SEEDING_KEYS = frozenset({'enabled', 'seed_weight'})

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
            strategy: GPU execution strategy, a name of STRATEGY_NAME_MAP or SINGLE_GPU_PYTHON_LOOP
            position_config: Refused when given (a base-class strategy this class has no route for).
            island_config: Refused when given, as position_config.
            hybrid_config: Refused when given, as position_config.
            pmap_config: Refused when given, as position_config.

        Raises:
            ValueError: A strategy name without a route, or any of the four sharding configurations.
        """
        # The configuration key's meaning (STRATEGY_NAME_MAP); the Python-loop name keeps the
        # one-device loop. The base class is set up for one device, as when the configuration key
        # chooses the strategy.
        if strategy == MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP:
            routed = strategy
        elif strategy.name in STRATEGY_NAME_MAP:
            routed = STRATEGY_NAME_MAP[strategy.name]
        else:
            raise ValueError(
                f"strategy {strategy.name} has no route in EMRHyperNEAT; the names are "
                f"{sorted(STRATEGY_NAME_MAP)} (and SINGLE_GPU_PYTHON_LOOP)")
        # The four sharding configurations configure base-class strategies this class has no route
        # for; with a routed strategy 0.1.0 kept them and never read them.
        given = [key for key, value in (('position_config', position_config),
                                        ('island_config', island_config),
                                        ('hybrid_config', hybrid_config),
                                        ('pmap_config', pmap_config))
                 if value is not None]
        if given:
            raise ValueError(f"{', '.join(given)} configure base-class strategies EMRHyperNEAT has no "
                             f"route for; {routed.name} reads none of them")
        super().__init__(
            name=name,
            implementation=implementation,
            strategy=MultiGPUStrategy.SINGLE_GPU,
        )
        self.strategy = routed

        # Extended configuration
        self.extended_config: Optional[EMRConfig] = None
        # An explicit substrate.activate_time from the last create_config (None: the key was absent), which the
        # settling steps honor there and in a later set_recurrence_config (resolve_recurrence)
        self._substrate_activate_time: Any = None

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
        self.hh_activation_mode: str = 'every_iteration'  # every_iteration or tanh

        # Aggregation function settings
        self.agg_mode: str = 'disabled'  # disabled, global, cppn_output, weight_interpretation
        self.agg_global_function: str = 'sum'
        self.agg_num_aggregations: int = 4
        self.agg_use_true_aggregation: bool = True  # True = accurate, False = approximation
        self.agg_interpretation: str = 'magnitude_bio'
        # How channels and weight-interpretation signals become indices (INDEX_DECODES)
        self.df_index_decode: str = 'tanh'
        self.agg_index_decode: str = 'tanh'
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
        # The last generation's champion as evaluated, for evaluate_on_data.
        self._last_eval: Optional[Dict[str, Any]] = None

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

        # ====================================================================
        # Execution mode (memory residency)
        # ====================================================================
        # Where a generation's intermediate arrays live: all on device
        # (standard), one level at a time (streaming), or assembled in host RAM
        # and moved to the device chunk by chunk (depth_offload). Defaults to
        # standard, so behaviour is unchanged unless a config asks otherwise.
        #
        # Orthogonal to MultiGPUStrategy, which is a device-parallelism axis.
        self.execution_config: ExecutionConfig = ExecutionConfig()
        self._apply_execution_config()

        # Depth-offload runtime counters, read back through get_memory_stats().
        self._cpu_memory_peak_gb: float = 0.0
        self._gpu_transfers_count: int = 0


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
        # A missing block is warned about once, in the base create_config that
        # `super()` reaches below; warning here too would double it.
        _refuse_unknown_keys('emr_hyperneat', hmr_config, self.EMR_HYPERNEAT_KEYS)
        if hmr_config.get('iteration_level', 1) != 1:
            raise ValueError(
                "emr_hyperneat.iteration_level is not read (the multi-hop level is "
                "recurrence.iteration_level); 0.1.0 ignored a value other than 1, with a "
                "deprecation warning")

        # Opt-in reproduction flag (default False -> EMR's existing behavior is UNCHANGED).
        # When True, EMR performs HMR's extra per-generation pre-ask jax.random.split.
        # HMR's default (extra_randkey_split=True) produced the published results, so this
        # lets reproduction benchmarks recover HMR's exact per-seed trajectories in EMR.
        # NOTE: stored under a DISTINCT name so the base class's own `extra_randkey_split`
        # attribute (which defaults True but is unused by EMR's eval) cannot leak in.
        self._repro_extra_randkey_split = hmr_config.get('extra_randkey_split', False)

        # Check if geometry seeding is enabled - need to set CPPN num_inputs to 7
        geo_config = hmr_config.get('geometry_seeding', {})
        _refuse_unknown_keys('geometry_seeding', geo_config, self.GEOMETRY_SEEDING_KEYS)
        _refuse_unknown_keys('execution', hmr_config.get('execution') or {}, EXECUTION_KEYS)
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
        if rescale_var is not None and not rescale_var > 0:
            # 0.1.0 ignored a variance that is not positive, so the run scored 1-MSE
            raise ValueError(f"emr_hyperneat.fitness_rescale_variance must be a positive variance, got "
                             f"{rescale_var!r}")
        if rescale_var is not None and rescale_var > 0:
            self._fitness_rescale_variance = float(rescale_var)
            if self.verbose:
                print(f"[EMR-HyperNEAT] Fitness rescaling enabled: "
                      f"target_variance={rescale_var:.6f}, "
                      f"fitness = R² = 1 - MSE/var")

        # Now call parent to set up base configuration
        config = super().create_config(params)
        if self.fitness_mode not in FITNESS_MODES:
            raise ValueError(f"emr_hyperneat.fitness_mode must be one of {FITNESS_MODES}, got "
                             f"{self.fitness_mode!r}; 0.1.0 scored an unknown name as MSE")
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
            strategy_map = STRATEGY_NAME_MAP
            if strategy_str_upper in strategy_map:
                self.strategy = strategy_map[strategy_str_upper]
                if self.verbose:
                    print(f"[EMR-HyperNEAT] Strategy from config: {strategy_str} → {self.strategy}")
            else:
                # 0.1.0 printed a warning and ran the default strategy.
                raise ValueError(
                    f"Unknown multi_gpu_strategy {strategy_str!r}; the names are "
                    f"{sorted(strategy_map)}")

        # The base class read substrate.activate_time into activate_time; whether it was set decides the settling
        # steps of a recurrent configuration, here and in a later set_recurrence_config
        self._substrate_activate_time = algo_params.get('substrate', {}).get('activate_time')
        self._parse_extended_config(recurrence_section)

        # Parse dynamic functions configuration
        self._parse_dynamic_functions_config(hmr_config)

        # Parse neuromodulation configuration (4 levels + multi-task)
        self._parse_neuromodulation_config(hmr_config)

        # Parse streaming configuration
        # Parse execution mode (memory residency). `enable_streaming: true` is
        # still accepted and means `mode: streaming`; an explicit `execution.mode`
        # wins over it.
        # Updates the existing object rather than replacing it, so options set
        # imperatively before create_config survive the parse.
        self.execution_config = parse_execution_config(
            hmr_config, base=self.execution_config
        )
        self._apply_execution_config()
        # The generation loop selects its build step from this flag, so resolve
        # AUTO here where max_depth is known. `run_until_threshold` also flips it
        # for the duration of a MultiGPUStrategy.STREAMING run and restores it in a
        # finally, which keeps working. (Not an OOM fallback, as this comment once
        # claimed: there is no OOM handler on this path, and nothing else in the
        # package writes the flag.)
        self.enable_streaming = (
            self.execution_config.resolve(self.max_depth) is ExecutionMode.STREAMING
        )
        self._streaming_verbose = hmr_config.get('streaming_verbose', False)
        # variance_rule applies on every execution mode: streaming and depth_offload take the
        # legacy level-0 rule through legacy_level0_variance. 0.1.0 ignored it there, warning only
        # when the mode was configured, not when it changed at run time.


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

        self._check_feature_combinations()
        return config

    def _check_feature_combinations(self) -> None:
        """Refuse configurations one evaluation path would silently drop part of.

        Raises:
            ValueError: multi-task evaluation with dynamic functions or aggregation (it applies
                one activation per task and sum aggregation), multi-task two-module
                evaluation with recurrence (it is feed-forward), or the depth-offload execution
                mode or the EVAL_ONLY_PARALLEL strategy with dynamic functions, aggregation or
                neuromodulation (both evaluate tanh nodes with sum aggregation and no
                modulation).
        """
        recurrent = (self.extended_config is not None
                     and self.extended_config.allow_hidden_to_hidden)
        dropped = self._features_a_tanh_sum_path_drops()
        per_node = [feature for feature in dropped if feature != "neuromodulation"]
        multitask = (self.multitask_config is not None
                     and getattr(self.multitask_config, 'enabled', False))
        # The configured hidden activation applies in mode 'global' only, and the output activation
        # everywhere except the plain evaluation (no per-node function, aggregation or
        # neuromodulation) and the multi-task one, which apply tanh and sigmoid. 0.1.0 dropped both
        # there without a word.
        if self.dynamic_functions_mode != 'global' and self.df_hidden_activation != 'tanh':
            raise ValueError(
                f"hidden_activation {self.df_hidden_activation!r} (dynamic_functions or substrate) "
                f"applies in dynamic_functions.mode 'global' only; mode "
                f"{self.dynamic_functions_mode!r} would ignore it. Set mode 'global', or drop the key.")
        modulated = self.neuromod_config is not None and self.neuromod_config.mode != 'disabled'
        plain = (self.dynamic_functions_mode in ('disabled', 'modular')
                 and self.agg_mode in ('disabled', 'modular') and not modulated)
        if (plain or multitask) and self.df_output_activation != 'sigmoid':
            raise ValueError(
                f"output_activation {self.df_output_activation!r} (dynamic_functions or substrate) "
                f"would be ignored: the {'multi-task' if multitask else 'plain'} evaluation applies "
                f"a sigmoid output. Set dynamic_functions.mode 'global' with hidden_activation "
                f"'tanh' to use another output activation.")
        if multitask and per_node:
            raise ValueError(
                "run_generation_multitask evaluates every task with one hidden activation "
                "(multitask.per_task_activation or multitask.hidden_activation) and sum "
                "aggregation, so it would ignore " + ", ".join(per_node) + ". Configure the "
                "activation through the multitask section instead."
            )
        if (multitask and recurrent
                and getattr(self.multitask_config, 'two_module_mode', 'none') != 'none'):
            raise ValueError(
                "multitask.two_module_mode evaluates feed-forward only and would ignore the "
                "hidden-to-hidden connections this configuration discovers"
            )
        if self.neuromod_config is not None:
            rule = self.neuromod_config.node_rule
            if rule not in NEUROMODULATION_NODE_RULES:
                raise ValueError(f"neuromodulation.node_rule must be one of "
                                 f"{NEUROMODULATION_NODE_RULES}, got {rule!r}")
            if (rule == 'hmr_full' and self.multitask_config is not None
                    and getattr(self.multitask_config, 'two_module_mode', 'none') != 'none'):
                raise ValueError("neuromodulation.node_rule 'hmr_full' is not applied by "
                                 "multitask.two_module_mode, whose modules compute the paper's rule")
        if self.execution_config.resolve(self.max_depth) is ExecutionMode.DEPTH_OFFLOAD:
            self.refuse_depth_offload()
        if self.strategy == MultiGPUStrategy.EVAL_ONLY_PARALLEL:
            dropped = dropped + self._features_offload_drops()
            if dropped:
                raise ValueError(
                    "multi_gpu_strategy EVAL_ONLY_PARALLEL evaluates tanh hidden nodes with sum "
                    "aggregation, no neuromodulation and 1-MSE on weights it builds itself, so it "
                    "would ignore " + ", ".join(dropped)
                )

    def _refuse_pmap_pipeline_gaps(self, strategy_name: str) -> None:
        """Refuse what the pmap pipelines cannot compute, at the moment one is entered.

        The FULL_PIPELINE_PARALLEL cores (`_full_pipeline_with_hh_core`,
        `_single_genome_pipeline_with_hh`) discover hidden-to-hidden connections inline at
        iteration level 1 only: they have no multi-hop expansion, so in 0.1.0 a level-2 preset
        ran there as level 1 while the record named level 2 (the level-2 fitness vector of the
        pmap route equals the standard path's level-1 vector, XOR depth 3 population 8 and
        30). They also evaluate tanh hidden nodes with sum aggregation and no
        neuromodulation, like EVAL_ONLY_PARALLEL. This runs at dispatch, not at configuration
        time, because the same configuration falls back to the standard path on one device or
        on a small dataset, where level 2 is computed.

        Raises:
            ValueError: `allow_hidden_to_hidden` with `iteration_level` above 1, or a dropped
                feature.
        """
        recurrent = (self.extended_config is not None
                     and self.extended_config.allow_hidden_to_hidden)
        level = self.extended_config.iteration_level if self.extended_config else 0
        if recurrent and level > 1:
            raise ValueError(
                f"multi_gpu_strategy {strategy_name} discovers hidden-to-hidden connections at "
                f"iteration level 1 only (no multi-hop expansion) and would run iteration_level "
                f"{level} as level 1. Set recurrence.iteration_level: 1, or use SINGLE_GPU or "
                f"EVAL_ONLY_PARALLEL, which run the standard discovery."
            )
        dropped = self._features_a_tanh_sum_path_drops() + self._features_offload_drops()
        if dropped:
            raise ValueError(
                f"multi_gpu_strategy {strategy_name} evaluates tanh hidden nodes with sum "
                "aggregation, no neuromodulation and 1-MSE on weights it builds itself, so it "
                "would ignore " + ", ".join(dropped)
            )

    def _features_a_tanh_sum_path_drops(self) -> List[str]:
        """The configured features an evaluation with tanh nodes, sum and no modulation drops."""
        dropped = []
        if self.dynamic_functions_mode != 'disabled':
            dropped.append(f"dynamic_functions.mode={self.dynamic_functions_mode!r}")
        if self.agg_mode != 'disabled':
            dropped.append(f"aggregation.mode={self.agg_mode!r}")
        if self.neuromod_config is not None and self.neuromod_config.mode != 'disabled':
            dropped.append("neuromodulation")
        return dropped

    def _features_offload_drops(self) -> List[str]:
        """Configured features the evaluations that build their own weights and score 1-MSE drop.

        Depth offload, POPULATION_PARALLEL_PROCESS's workers, EVAL_ONLY_PARALLEL and the
        FULL_PIPELINE_PARALLEL pipelines score 1-MSE without the fitness rescaling and build their
        weight matrices without the locality penalty. 0.1.0 dropped these settings there without a
        word.
        """
        dropped = []
        if getattr(self, 'fitness_mode', 'mse') != 'mse':
            dropped.append(f"fitness_mode={self.fitness_mode!r}")
        if getattr(self, 'locality_radius', None) is not None:
            dropped.append(f"locality_radius={self.locality_radius!r}")
        if getattr(self, '_fitness_rescale_variance', None) is not None:
            dropped.append(f"fitness_rescale_variance={self._fitness_rescale_variance!r}")
        return dropped

    def refuse_depth_offload(self) -> None:
        """Raise if the depth-offload path would ignore part of this configuration.

        Called by ``create_config`` for a configured or depth-selected ``depth_offload``, and by
        the offload generation itself, because on a GPU ``auto`` can also select it at run time
        from the population size and device memory.

        Raises:
            ValueError: Dynamic functions, aggregation, neuromodulation, a fitness mode other than
                mse, a locality radius or a fitness rescaling are configured.
        """
        dropped = self._features_a_tanh_sum_path_drops() + self._features_offload_drops()
        if dropped:
            raise ValueError(
                "execution mode 'depth_offload' evaluates tanh hidden nodes with sum "
                "aggregation, no neuromodulation and 1-MSE, without a locality penalty or a "
                "fitness rescaling, so it would ignore "
                + ", ".join(dropped)
                + ". Use execution.mode 'standard' or 'streaming' for these features."
            )

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
        # A run's fixed random draws are its own: an object reused for the next seed draws them
        # again (0.1.0's shared draw was the same in every run, so keeping it changed nothing).
        self._random_fixed_indices = None
        self._random_fixed_agg_indices = None
        # The generation count is the run's too: random_generation keys on it, and in 0.1.0 a
        # reused object continued it.
        self._current_generation = 0

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

    def substrate_config(self) -> Any:
        """The substrate configuration this model's generation loop uses.

        Read from the parsed configuration, so defaults and the loop's own choices
        (e.g. the plain tanh/sigmoid evaluation when neither per-node mode is on)
        are the ones EMR applies. See `emr_hyperneat.substrate`.

        Returns:
            An `emr_hyperneat.substrate.SubstrateConfig`.

        Raises:
            RuntimeError: create_config has not been called.
            ValueError: the configuration evaluates something the substrate API does not
                describe (neuromodulation, multi-task, swarm mode, the weight-interpretation or
                random function modes, a non-standard execution mode, or hidden-to-hidden
                connections other than per-genome discovery without the h->h cache).
        """
        # Lazy import: emr_hyperneat.substrate imports this module.
        from emr_hyperneat.substrate import SubstrateConfig

        if self.max_depth is None or not hasattr(self, 'dynamic_functions_mode'):
            raise RuntimeError("call create_config before substrate_config")

        refusals = []
        if self.dynamic_functions_mode not in ('disabled', 'global', 'cppn_output'):
            refusals.append(f"dynamic_functions.mode={self.dynamic_functions_mode!r}")
        if self.agg_mode not in ('disabled', 'global', 'cppn_output'):
            refusals.append(f"aggregation.mode={self.agg_mode!r}")
        recurrent = self.extended_config is not None and self.extended_config.allow_hidden_to_hidden
        if recurrent:
            # The API builds hidden-to-hidden connections as per-genome discovery does; what it
            # cannot reproduce is refused.
            ec = self.extended_config
            if ec.hh_discovery_mode != 'per_genome':
                refusals.append(f"hh_discovery_mode {ec.hh_discovery_mode!r} (the API builds per_genome discovery)")
            if ec.use_dense_discovery:
                refusals.append("use_dense_discovery")
            if ec.hh_cache_enabled:
                refusals.append("the h->h cache (it reuses connections discovered for other genomes)")
            if self.hh_aggregation_mode == 'dynamic' and not (
                    self.agg_mode == 'cppn_output' and self.num_cppn_outputs > 1):
                refusals.append("hh_aggregation_mode 'dynamic' without the aggregation channel "
                                "(the weight-interpretation decode)")
        if self.neuromod_config is not None and self.neuromod_config.mode != 'disabled':
            refusals.append(f"neuromodulation mode {self.neuromod_config.mode!r}")
        if self.multitask_config is not None and getattr(self.multitask_config, 'enabled', False):
            refusals.append("multi-task evaluation")
        if self.swarm_mode:
            refusals.append("swarm mode")
        resolved = self.execution_config.resolve(self.max_depth)
        if resolved is not ExecutionMode.STANDARD:
            refusals.append(f"execution mode {resolved.value!r} (set execution.mode: standard)")
        if refusals:
            raise ValueError(
                "substrate_config describes the substrate of the standard "
                "execution path; this configuration also uses " + ", ".join(refusals)
            )

        # The generation loop's plain branch runs when neither per-node mode is on, and
        # it applies tanh and sigmoid whatever output_activation says.
        if self.dynamic_functions_mode == 'disabled' and self.agg_mode == 'disabled':
            hidden, output = 'tanh', 'sigmoid'
        else:
            name = self.df_hidden_activation if self.dynamic_functions_mode == 'global' else 'tanh'
            hidden = name if name in ACTIVATION_FUNCTIONS else 'tanh'
            output = (self.df_output_activation
                      if self.df_output_activation in ACTIVATION_FUNCTIONS else 'sigmoid')
        global_agg = (self.agg_global_function
                      if self.agg_global_function in AGGREGATION_FUNCTIONS else 'sum')

        return SubstrateConfig(
            input_coords=tuple(tuple(float(c) for c in xy) for xy in self.substrate_input_coords),
            output_coords=tuple(tuple(float(c) for c in xy) for xy in self.substrate_output_coords),
            max_depth=int(self.max_depth),
            variance_threshold=float(self.variance_threshold),
            initial_depth=int(self.initial_depth),
            variance_rule=self.variance_rule,
            max_weight=float(self.max_weight),
            weight_threshold=WEIGHT_THRESHOLD,
            locality_radius=self.locality_radius,
            geometry_seeding=bool(self.geometry_seeding_enabled),
            num_cppn_outputs=int(self.num_cppn_outputs),
            activation_mode=self.dynamic_functions_mode,
            activation_channel=int(self._cppn_output_indices['activation']),
            num_activations=int(self.df_num_activations),
            palette=(tuple(int(i) for i in np.asarray(self.df_palette))
                     if self.df_palette is not None else None),
            hidden_activation=hidden,
            output_activation=output,
            aggregation_mode=self.agg_mode,
            aggregation_channel=int(self._cppn_output_indices['aggregation']),
            num_aggregations=int(self.agg_num_aggregations),
            global_aggregation=global_agg,
            true_aggregation=bool(self.agg_use_true_aggregation),
            **self._substrate_recurrence_fields(recurrent),
            activation_index_decode=self.df_index_decode,
            aggregation_index_decode=self.agg_index_decode,
        ).validate()

    def _substrate_recurrence_fields(self, recurrent: bool) -> Dict[str, Any]:
        """The SubstrateConfig recurrence fields of this configuration (none when feed-forward)."""
        if not recurrent:
            return {}
        ec = self.extended_config
        return dict(
            hidden_to_hidden=True,
            allow_backward=bool(ec.allow_backward),
            allow_lateral=bool(ec.allow_lateral),
            allow_self_loops=bool(ec.allow_self_loops),
            band_threshold=float(self.band_threshold),
            iteration_level=int(ec.iteration_level),
            hop_decay_factor=float(ec.hop_decay_factor),
            multi_hop_algorithm=ec.multi_hop_algorithm,
            max_sparse_conns=int(ec.max_sparse_conns),
            activate_time=int(self.activate_time),
            settling_rule='emr',
            hh_activation_mode=self.hh_activation_mode,
            hh_aggregation_mode=self.hh_aggregation_mode,
        )

    def population_substrates(self, state: Any, device_id: int = 0) -> Any:
        """The substrates of the population in ``state``, as the next generation builds them.

        With recurrence the hidden-to-hidden query keeps every genome's pair outputs until the connections are
        built: ``count**2 * num_cppn_outputs`` float32 values per genome for ``count`` active positions (8 MB per
        genome at 1,000 positions and two outputs), where the generation loop holds one bucket slice at a time.

        Args:
            state: Algorithm state from initialize or run_generation.
            device_id: Device for the feed-forward query; the hidden-to-hidden query runs on the default device.

        Returns:
            An `emr_hyperneat.substrate.Substrate` with one entry per genome.
        """
        from emr_hyperneat.substrate import (
            build_recurrent_substrate,
            build_substrate,
            query_recurrent_substrate,
            query_substrate,
        )

        config = self.substrate_config()
        population = self._compiled_ask(state)
        transformed = self._compiled_transform_batch(state, population)
        substrate = build_substrate(
            *query_substrate(self._jitted_cppn_forward, state, transformed, config,
                             device_id=device_id),
            config,
        )
        if config.hidden_to_hidden:
            # The public two-step build, with this model's CPPNs as the evaluator
            substrate = build_recurrent_substrate(
                substrate, query_recurrent_substrate(self._jitted_cppn_forward, state, transformed, substrate))
        return substrate

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

    def _hh_class_flags(self) -> Tuple[bool, bool, bool]:
        """(allow_backward, allow_lateral, allow_self_loops) for the pmap pipelines' inline discovery.

        Every class and no self loops when no recurrence config is set, which is what these pipelines
        did for every preset in 0.1.0.
        """
        cfg = self.extended_config
        if cfg is None:
            return (True, True, False)
        return (bool(cfg.allow_backward), bool(cfg.allow_lateral), bool(cfg.allow_self_loops))

    def _parse_extended_config(self, recurrence_section: Dict[str, Any]) -> None:
        """Parse recurrence section into extended config.

        Args:
            recurrence_section: The 'recurrence' subsection of emr_hyperneat config
        """
        _refuse_unknown_keys('recurrence', recurrence_section, self.RECURRENCE_KEYS)
        # Check for preset first
        preset_name = recurrence_section.get('preset', None)
        if preset_name:
            base_config = get_recurrence_preset(preset_name)
            # A preset decides whether recurrence is on; a block key that says otherwise was
            # ignored in 0.1.0, so the run silently disagreed with its configuration
            for key in ('enabled', 'allow_hidden_to_hidden'):
                if key in recurrence_section and (bool(recurrence_section[key])
                                                  != bool(getattr(base_config, key))):
                    raise ValueError(
                        f"recurrence.{key}={recurrence_section[key]!r} contradicts preset "
                        f"{preset_name!r} ({key}={getattr(base_config, key)!r}): the preset "
                        f"decides it. Remove the key, or choose a preset that agrees with it.")
            if self.verbose:
                print(f"[EMR-HyperNEAT] Using preset '{preset_name}'")

            # Apply explicit overrides from recurrence_section
            # The preset provides defaults, but explicit values take precedence
            override_fields = {}
            configurable_fields = [
                'iteration_level', 'multi_hop_algorithm', 'hop_decay_factor',
                'hh_cache_enabled', 'hh_refresh_interval', 'hh_mask_change_threshold',
                'allow_backward', 'allow_lateral', 'allow_self_loops',
                'activate_time', 'max_connections', 'max_sparse_conns', 'use_dense_discovery',
                'hh_discovery_mode', 'multi_hop_second_factor',
                'eval_chunk_size',  # For OOM prevention on large configs
            ]
            for field in configurable_fields:
                if field in recurrence_section:
                    override_fields[field] = recurrence_section[field]
            if 'hh_discovery_mode' in override_fields:
                override_fields['hh_discovery_mode'] = _validated_hh_discovery_mode(
                    override_fields['hh_discovery_mode'])
            elif 'multi_hop_second_factor' in override_fields:
                # The preset resolved its discovery mode for the masked factor at construction; a
                # factor override re-resolves it (the raw factor exists only on the union path).
                override_fields['hh_discovery_mode'] = None

            # Create new config with overrides
            if override_fields:
                from dataclasses import replace
                self.extended_config = replace(base_config, **override_fields)
                if self.verbose:
                    print(f"[EMR-HyperNEAT] Applied overrides: {override_fields}")
            else:
                self.extended_config = base_config
            if not base_config.allow_hidden_to_hidden:
                allowed = [key for key in ('allow_backward', 'allow_lateral', 'allow_self_loops')
                           if recurrence_section.get(key)]
                if allowed:
                    raise ValueError(
                        f"recurrence.{allowed[0]} is set but preset {preset_name!r} has no "
                        f"hidden-to-hidden connections, so the run would be feed-forward; choose a "
                        f"recurrent preset")
        elif recurrence_section.get('enabled', False) or recurrence_section.get('allow_hidden_to_hidden', False):
            check_recurrence_switch(recurrence_section.get('enabled', True),
                                    recurrence_section.get('allow_hidden_to_hidden', True))
            # Build config from individual settings
            self.extended_config = EMRConfig(
                enabled=recurrence_section.get('enabled', True),
                allow_hidden_to_hidden=recurrence_section.get('allow_hidden_to_hidden', True),
                allow_backward=recurrence_section.get('allow_backward', False),
                allow_lateral=recurrence_section.get('allow_lateral', False),
                allow_self_loops=recurrence_section.get('allow_self_loops', False),
                iteration_level=recurrence_section.get('iteration_level', 1),
                activate_time=recurrence_section.get('activate_time', None),
                max_connections=recurrence_section.get('max_connections', 10000),
                max_sparse_conns=recurrence_section.get('max_sparse_conns', 10000),
                multi_hop_algorithm=recurrence_section.get('multi_hop_algorithm', 'matrix_power'),
                hop_decay_factor=recurrence_section.get('hop_decay_factor', 0.8),
                multi_hop_second_factor=recurrence_section.get('multi_hop_second_factor', 'masked'),
                hh_cache_enabled=recurrence_section.get('hh_cache_enabled', False),
                hh_refresh_interval=recurrence_section.get('hh_refresh_interval', 5),
                hh_mask_change_threshold=recurrence_section.get('hh_mask_change_threshold', 0.1),
                use_dense_discovery=recurrence_section.get('use_dense_discovery', False),
                hh_discovery_mode=_validated_hh_discovery_mode(
                    recurrence_section.get('hh_discovery_mode')),
                eval_chunk_size=recurrence_section.get('eval_chunk_size', None),
            )
        elif any(recurrence_section.get(key) for key in ('allow_backward', 'allow_lateral', 'allow_self_loops')):
            allowed = [key for key in ('allow_backward', 'allow_lateral', 'allow_self_loops')
                       if recurrence_section.get(key)]
            raise ValueError(
                f"recurrence.{allowed[0]} is set but hidden-to-hidden connections are off (no preset, "
                f"enabled or allow_hidden_to_hidden), so the run would be feed-forward; set a preset or "
                f"allow_hidden_to_hidden: true")
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

        # Forward mode, iteration level and settling steps, by the rule set_recurrence_config also applies. Either
        # block sets the steps (in 0.1.0 a recurrent configuration overwrote substrate.activate_time without a
        # word); unset, a preset's own count or 2 (2**max_depth + 1)
        self._forward_mode, self.iteration_level, steps = resolve_recurrence(
            self.extended_config, self.max_depth,
            {'recurrence.activate_time': recurrence_section.get('activate_time'),
             'substrate.activate_time': self._substrate_activate_time})
        if steps is not None:
            self.activate_time = steps

        # Log configuration
        if self.extended_config is not None and self.verbose:
            mode = self._forward_mode.value
            print(f"[EMR-HyperNEAT] Mode: {mode}")
            if self.extended_config.allow_hidden_to_hidden:
                print(f"  - iteration_level: {self.extended_config.iteration_level}")
                print(f"  - multi_hop_algorithm: {self.extended_config.multi_hop_algorithm}")
                print(f"  - hop_decay_factor: {self.extended_config.hop_decay_factor}")
                print(f"  - allow_backward: {self.extended_config.allow_backward}")
                print(f"  - allow_lateral: {self.extended_config.allow_lateral}")
                print(f"  - allow_self_loops: {self.extended_config.allow_self_loops}")
                print(f"  - activate_time: {self.activate_time}")

    def _parse_dynamic_functions_config(self, hmr_config: Dict[str, Any]) -> None:
        """Parse dynamic functions and aggregation configuration.

        Args:
            hmr_config: The 'emr_hyperneat' section of config
        """
        dynamic_funcs = hmr_config.get('dynamic_functions', {})
        _refuse_unknown_keys('dynamic_functions', dynamic_funcs, self.DYNAMIC_FUNCTIONS_KEYS)
        aggregation_section = hmr_config.get('aggregation', {})
        _refuse_unknown_keys('aggregation', aggregation_section, self.AGGREGATION_KEYS)
        for name, block, known in (
                ('dynamic_functions.cppn_output_indices', dynamic_funcs.get('cppn_output_indices'),
                 self.CPPN_OUTPUT_INDICES_KEYS),
                ('dynamic_functions.sparsity', dynamic_funcs.get('sparsity'), self.SPARSITY_KEYS),
                ('dynamic_functions.critical_periods', dynamic_funcs.get('critical_periods'),
                 self.CRITICAL_PERIODS_KEYS),
                ('dynamic_functions.modular', dynamic_funcs.get('modular'), self.MODULAR_KEYS),
                ('dynamic_functions.modular.sparsity',
                 (dynamic_funcs.get('modular') or {}).get('sparsity'), self.SPARSITY_KEYS)):
            _refuse_unknown_keys(name, block, known)
        if 'palette_evolution' in dynamic_funcs and dynamic_funcs['palette_evolution'] != {'enabled': False}:
            raise ValueError(
                f"dynamic_functions.palette_evolution is not part of EMR (HMR's dynamic-functions "
                f"variant mutated a palette mask; the palette paper's strategies are in "
                f"papers/emr-dynamic-functions-bio-inspired/scripts/palette_strategies/); only "
                f"{{'enabled': False}} is accepted, got {dynamic_funcs['palette_evolution']!r}. Set "
                f"algo.df_palette from the runner instead.")

        # Mode selection:
        # - 'disabled': Original hardcoded tanh/sigmoid (baseline)
        # - 'global': All hidden nodes use same configurable activation
        # - 'cppn_output': CPPN outputs activation index per node
        # - 'weight_interpretation': Derive activation from weight patterns
        # - 'random_fixed': Random activation per node, fixed at initialization
        # - 'random_generation': Random re-assigned each generation
        # - 'modular': Orthogonal configuration
        self.dynamic_functions_mode = dynamic_funcs.get('mode', 'disabled')
        if self.dynamic_functions_mode not in self.DYNAMIC_FUNCTION_MODES:
            raise ValueError(
                f"dynamic_functions.mode must be one of "
                f"{sorted(self.DYNAMIC_FUNCTION_MODES)}, got "
                f"{self.dynamic_functions_mode!r}. Unknown values used to fall through "
                "the activation dispatch and run the fixed tanh/sigmoid baseline "
                "instead, which reports as a successful run of the mode you asked for."
            )
        if self.dynamic_functions_mode == 'modular':
            # Parsed just below, echoed under verbose, and never dispatched: the
            # activation-index branch that reads df_modular_config exists only in the
            # frozen HMR snapshot (_hmr_frozen/hmrhyperneat_neuromodulation_functions
            # .py, the `elif ... == 'modular'` arm), and EMRHyperNEAT does not inherit
            # from it. On this path the mode leaves act_indices and _hidden_act_fn both
            # None, so evaluation falls back to fixed tanh.
            import warnings
            warnings.warn(
                "dynamic_functions.mode='modular' is parsed but not implemented on the "
                "EMR path: no activation indices are computed and evaluation falls back "
                "to fixed tanh. It is implemented only in the frozen HMR snapshot. Use "
                "'cppn_output' or 'weight_interpretation' for per-node activations.",
                UserWarning,
                stacklevel=2,
            )

        # Two blocks nothing in EMR reads: 0.1.0 parsed them and only printed them
        if 'sparsity' in dynamic_funcs:
            raise ValueError(
                "dynamic_functions.sparsity is not implemented in EMR (0.1.0 parsed it and nothing read "
                "it); mode 'modular', which EMR does not implement either, takes "
                "dynamic_functions.modular.sparsity")
        if 'modular' in dynamic_funcs and self.dynamic_functions_mode != 'modular':
            raise ValueError(
                f"dynamic_functions.modular configures mode 'modular' only; with mode "
                f"{self.dynamic_functions_mode!r} 0.1.0 ignored it")

        # For random modes: the seed and the scope of the draw (random_function_indices)
        self._random_mode_seed = dynamic_funcs.get('random_seed', None)
        self._random_scope = dynamic_funcs.get('random_scope', 'genome')
        if self._random_scope not in RANDOM_SCOPES:
            raise ValueError(f"dynamic_functions.random_scope must be one of {RANDOM_SCOPES}, got "
                             f"{self._random_scope!r}")
        # How a channel or a weight-interpretation signal becomes an index (INDEX_DECODES)
        self.df_index_decode = dynamic_funcs.get('index_decode', 'tanh')
        if self.df_index_decode not in INDEX_DECODES:
            raise ValueError(f"dynamic_functions.index_decode must be one of {INDEX_DECODES}, got "
                             f"{self.df_index_decode!r}")

        # For 'global' mode: which activation to use. Absent here, both come from the substrate
        # section, as in HMR's dynamic-functions variant (0.1.0 read substrate.hidden_activation and
        # substrate.output_activation nowhere). An unknown name raises: 0.1.0 ran tanh (hidden) or
        # sigmoid (output) for it.
        self.df_hidden_activation = dynamic_funcs.get(
            'hidden_activation', getattr(self, 'hidden_activation', 'tanh'))
        self.df_output_activation = dynamic_funcs.get(
            'output_activation', getattr(self, 'output_activation', 'sigmoid'))
        for key, name in (('hidden_activation', self.df_hidden_activation),
                          ('output_activation', self.df_output_activation)):
            if name not in ACTIVATION_FUNCTIONS:
                raise ValueError(
                    f"{key} {name!r} (dynamic_functions.{key}, or substrate.{key} when that is "
                    f"absent) is not an activation function; the names are "
                    f"{sorted(ACTIVATION_FUNCTIONS)}")

        # For 'cppn_output' mode: number of activation functions available
        self.df_num_activations = dynamic_funcs.get('num_activations', 4)

        # CPPN output configuration for cppn_output mode
        # num_cppn_outputs: How many outputs the CPPN has (default from auto-compute)
        # When num_cppn_outputs=1, we derive activation from the same weight output
        # When num_cppn_outputs>1, each output can be dedicated to different purposes
        self.num_cppn_outputs = dynamic_funcs.get(
            'num_cppn_outputs', getattr(self, '_auto_num_cppn_outputs', 1)
        )

        # cppn_output_indices: Which CPPN output index to use for each purpose.
        # Channel 0 is the weight; the generation loop always reads it there.
        #
        # Aggregation defaults to the first channel nothing else reads. Before 0.2.0 a
        # two-output CPPN with aggregation (but not activation) in cppn_output mode
        # defaulted to channel 0, so aggregation was decoded from the weight and the
        # evolved second output was never read. Runs made that way are reproduced by
        # setting cppn_output_indices: {aggregation: 0}. With both modes on and an
        # explicit num_cppn_outputs of 2, channel 1 is the activation's and aggregation
        # still shares channel 0, as before.
        cppn_indices = dynamic_funcs.get('cppn_output_indices', {})
        if cppn_indices.get('weight', 0) != 0:
            raise ValueError(
                "dynamic_functions.cppn_output_indices.weight must be 0: the weight is read "
                "from CPPN output 0 on every path, and another value was accepted and ignored "
                "in 0.1.0"
            )
        if self.num_cppn_outputs <= 1:
            default_agg_channel = 0
        elif self.num_cppn_outputs == 2:
            default_agg_channel = 0 if self.dynamic_functions_mode == 'cppn_output' else 1
        else:
            default_agg_channel = 2
        self._cppn_output_indices = {
            'weight': cppn_indices.get('weight', 0),
            'activation': cppn_indices.get('activation', 0 if self.num_cppn_outputs == 1 else 1),
            'aggregation': cppn_indices.get('aggregation', default_agg_channel),
        }

        # Validate CPPN output indices
        for purpose, idx in self._cppn_output_indices.items():
            if idx >= self.num_cppn_outputs:
                raise ValueError(
                    f"CPPN output index for '{purpose}' ({idx}) >= num_cppn_outputs ({self.num_cppn_outputs}). "
                    f"Increase num_cppn_outputs or fix cppn_output_indices."
                )

        # H→H activation mode: how the settling steps of the recurrent path (forward mode
        # HYBRID_SPARSE_HH) activate hidden nodes; the feed-forward path is unaffected.
        # - 'every_iteration' (default): the configured activation, per-node indices or the
        #   one 'global' function, at every step.
        # - 'tanh': tanh at every step, whatever is configured.
        # 'initial_only' is rejected. In 0.1.0 it applied no configured activation on the
        # recurrent path at all (the forward pass took no initial activated step, the one step
        # its name describes), and a name that says the opposite of what runs is not kept.
        self.hh_activation_mode = dynamic_funcs.get('hh_activation_mode', 'every_iteration')
        if self.hh_activation_mode == 'initial_only':
            raise ValueError(
                "dynamic_functions.hh_activation_mode='initial_only' is no longer accepted. "
                "In emr-hyperneat 0.1.0 it applied no configured activation on the "
                "recurrent path: every settling step was tanh and per-node or global "
                "activations were computed and never used. Use 'every_iteration' (the "
                "default: the configured activation at every step) or 'tanh' (tanh at every "
                "step, which settles as 0.1.0 did when no input aggregation is configured)."
            )
        if self.hh_activation_mode not in HH_ACTIVATION_MODES:
            raise ValueError(
                f"dynamic_functions.hh_activation_mode must be one of {HH_ACTIVATION_MODES}, "
                f"got {self.hh_activation_mode!r}"
            )
        if 'global_activation' in dynamic_funcs:
            raise ValueError(
                "dynamic_functions.global_activation is not a setting; the one hidden "
                "activation of mode 'global' is dynamic_functions.hidden_activation"
            )

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
            # Explicit list of ACTIVATION_LIST indices or names. 0.1.0 accepted an index outside
            # ACTIVATION_LIST, and evaluation dispatches by index.
            if not (isinstance(palette_config, (list, tuple)) and palette_config):
                raise ValueError(
                    "dynamic_functions.palette must be a preset name or a non-empty list of "
                    f"ACTIVATION_LIST indices or names, got {palette_config!r}")
            indices = []
            for entry in palette_config:
                if isinstance(entry, str) and entry in ACTIVATION_LIST:
                    indices.append(ACTIVATION_LIST.index(entry))
                elif (isinstance(entry, (int, np.integer)) and not isinstance(entry, bool)
                      and 0 <= int(entry) < len(ACTIVATION_LIST)):
                    indices.append(int(entry))
                else:
                    raise ValueError(
                        f"dynamic_functions.palette entry {entry!r} is neither an ACTIVATION_LIST "
                        f"index (0 to {len(ACTIVATION_LIST) - 1}) nor one of its names {ACTIVATION_LIST}")
            self.df_palette = jnp.array(indices, dtype=jnp.int32)
            self.df_palette_name = 'custom'

        # For 'weight_interpretation' mode: interpretation method
        self.df_interpretation = dynamic_funcs.get('interpretation', 'magnitude_bio')
        if self.df_interpretation not in ACTIVATION_INTERPRETATIONS:
            raise ValueError(
                f"dynamic_functions.interpretation must be one of {ACTIVATION_INTERPRETATIONS}, got "
                f"{self.df_interpretation!r}; 0.1.0 ran magnitude_bio for another name")

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
        if self.df_critical_periods_enabled:
            raise ValueError(
                "dynamic_functions.critical_periods is not implemented in EMR (nothing changes a "
                "mutation rate); 0.1.0 accepted it and ran without it. The critical periods of the "
                "palette paper are part of its strategies "
                "(papers/emr-dynamic-functions-bio-inspired/scripts/palette_strategies/)")
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
        if 'global_aggregation' in aggregation_config:
            raise ValueError(
                "aggregation.global_aggregation is not a setting; the function of aggregation "
                "mode 'global' is aggregation.global_function (the misspelled key used to be "
                "ignored, so such a configuration ran 'sum')"
            )
        self.agg_mode = aggregation_config.get('mode', 'disabled')
        if self.agg_mode not in self.AGGREGATION_MODES:
            raise ValueError(
                f"aggregation.mode must be one of {sorted(self.AGGREGATION_MODES)}, got "
                f"{self.agg_mode!r}; 0.1.0 ran sum aggregation for an unknown mode")
        if self.agg_mode == 'modular':
            import warnings
            warnings.warn(
                "aggregation.mode='modular' is parsed but not implemented on the EMR path: "
                "evaluation uses sum aggregation. It is implemented only in the frozen HMR "
                "aggregation variant.",
                UserWarning, stacklevel=2)

        # For random modes: the seed and the scope of the draw (random_function_indices)
        self._random_agg_seed = aggregation_config.get('random_seed', None)
        self._random_agg_scope = aggregation_config.get('random_scope', 'genome')
        if self._random_agg_scope not in RANDOM_SCOPES:
            raise ValueError(f"aggregation.random_scope must be one of {RANDOM_SCOPES}, got "
                             f"{self._random_agg_scope!r}")
        self.agg_index_decode = aggregation_config.get('index_decode', 'tanh')
        if self.agg_index_decode not in INDEX_DECODES:
            raise ValueError(f"aggregation.index_decode must be one of {INDEX_DECODES}, got "
                             f"{self.agg_index_decode!r}")

        # For 'global' mode: which aggregation to use (0.1.0 ran sum for an unknown name)
        self.agg_global_function = aggregation_config.get('global_function', 'sum')
        if self.agg_global_function not in AGGREGATION_FUNCTIONS:
            raise ValueError(
                f"aggregation.global_function must be one of {sorted(AGGREGATION_FUNCTIONS)}, got "
                f"{self.agg_global_function!r}")

        # For 'cppn_output' mode: number of aggregation functions available
        self.agg_num_aggregations = aggregation_config.get('num_aggregations', 4)

        # For 'weight_interpretation' mode: interpretation method
        self.agg_interpretation = aggregation_config.get('interpretation', 'magnitude_bio')
        if self.agg_interpretation not in AGGREGATION_INTERPRETATIONS:
            raise ValueError(
                f"aggregation.interpretation must be one of {AGGREGATION_INTERPRETATIONS}, got "
                f"{self.agg_interpretation!r}; 0.1.0 ran magnitude_bio for another name")

        # Whether to use true per-node aggregation or approximation
        self.agg_use_true_aggregation = aggregation_config.get('use_true_aggregation', True)

        # H→H aggregation mode: controls how aggregations are applied during H→H iterations
        # - 'sum': Standard scatter-add (current behavior, efficient)
        # - 'dynamic': Per-node aggregation using segment operations (more complex)
        self.hh_aggregation_mode = aggregation_config.get('hh_aggregation_mode', 'sum')
        if self.hh_aggregation_mode not in ('sum', 'dynamic'):
            raise ValueError(f"aggregation.hh_aggregation_mode must be 'sum' or 'dynamic', got "
                             f"{self.hh_aggregation_mode!r}")

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
        _refuse_unknown_keys('neuromodulation', neuromod_config, self.NEUROMODULATION_KEYS)

        # Mode selection (default: disabled = original behavior). An unknown mode built a configuration
        # whose dispatch raised at the first generation in 0.1.0.
        mode = neuromod_config.get('mode', 'disabled')
        if mode not in self.NEUROMODULATION_MODES:
            raise ValueError(
                f"neuromodulation.mode must be one of {self.NEUROMODULATION_MODES}, got {mode!r}; HMR's "
                f"neuromodulation variant used preset names as modes (e.g. "
                f"'true_neuromodulation_4nt_option_a_tanh', 'xdg_style', 'full', 'custom'); EMR's "
                f"presets are neuromodulation.preset")
        if neuromod_config.get('branch_gating', False):
            raise ValueError("neuromodulation.branch_gating is not implemented (no evaluation applies "
                             "it); 0.1.0 accepted and ignored it")

        # Check for preset
        preset = neuromod_config.get('preset', None)
        if preset is not None and preset not in NEUROMODULATION_PRESETS:
            raise ValueError(f"neuromodulation.preset must be one of {sorted(NEUROMODULATION_PRESETS)}, "
                             f"got {preset!r}; 0.1.0 ignored an unknown preset")
        levels_named = (preset not in (None, 'disabled')) or mode != 'disabled'
        if neuromod_config.get('enabled', False) and not levels_named:
            raise ValueError("neuromodulation.enabled is true but no mode or preset is set, so "
                             "modulation would be off")
        if 'enabled' in neuromod_config and not neuromod_config['enabled'] and levels_named:
            raise ValueError("neuromodulation.enabled is false but a mode or preset is set; 0.1.0 ran "
                             "the mode whatever enabled said. Set mode 'disabled' or drop enabled")
        if preset:
            base = get_neuromodulation_config(preset)
            # The preset's level, as the mode the evaluation dispatches on. The presets set level
            # flags and no mode, so in 0.1.0 a preset left the mode 'disabled' and the run had no
            # modulation.
            levels = tuple(name for name in ('static_gating', 'context_gating', 'modulatory_neurons',
                                             'true_neuromodulation') if getattr(base, name))
            preset_mode = {(): 'disabled', ('static_gating',): 'static_gating',
                           ('static_gating', 'context_gating'): 'context_gating',
                           ('modulatory_neurons',): 'modulatory_neurons',
                           ('true_neuromodulation',): 'true_neuromodulation'}.get(levels)
            if preset_mode is None:
                raise ValueError(f"neuromodulation.preset {preset!r} combines the levels {levels}; EMR "
                                 f"evaluates one level")
            if 'mode' in neuromod_config and mode != preset_mode:
                raise ValueError(f"neuromodulation.mode {mode!r} contradicts preset {preset!r} (level "
                                 f"{preset_mode!r})")
            # Override with explicit settings
            self.neuromod_config = NeuromodulationConfig(
                enabled=preset_mode != 'disabled',
                static_gating=base.static_gating,
                context_gating=base.context_gating,
                modulatory_neurons=base.modulatory_neurons,
                true_neuromodulation=base.true_neuromodulation,
                mode=preset_mode,
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
                node_rule=neuromod_config.get('node_rule', base.node_rule),
                # Branch-specific gating
                branch_gating=neuromod_config.get('branch_gating', base.branch_gating),
                # 0.1.0 dropped it on the preset path
                use_self_connection_query=neuromod_config.get('use_self_connection_query', False),
            )
            if preset_mode == 'disabled':
                self.neuromod_config = None
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
                node_rule=neuromod_config.get('node_rule', 'paper'),
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
        _refuse_unknown_keys('multitask', multitask_config, self.MULTITASK_KEYS)
        # Settings nothing in EMR applies, or that cannot rank genomes, raise unless left at their
        # default. The two constant bonuses add one value to every genome (orthogonality_bonus from the
        # tasks' NT vectors, specialization_bonus from the population's mean fitness variance): it never
        # changes a rank within a generation, only the reported fitness (and with it stagnation and a
        # threshold stop). 0.1.0 ignored the other six.
        for key, inert, why in (
                ('orthogonality_bonus', 0.0, "adds one constant to every genome (computed from the "
                                             "tasks' NT vectors), so it never changes a rank within a "
                                             "generation"),
                ('specialization_bonus', 0.0, "adds one constant to every genome (computed from the "
                                              "population's mean fitness variance), so it never "
                                              "changes a rank within a generation"),
                ('orthogonality_bonus_weight', 0.0, "adds one constant to every genome (computed from "
                                                    "the tasks' NT vectors), so it never changes a rank "
                                                    "within a generation"),
                ('specialization_bonus_weight', 0.0, "is never read"),
                ('modulation_penalty', 0.0, "is never read"),
                ('confusion_eval_frequency', 0, "is never read"),
                ('branch_gating_mode', 'none', "is never read (branch gating is not implemented)"),
                ('orthogonality_metric', 'cosine_mean', "applies only with an orthogonality bonus")):
            if multitask_config.get(key, inert) != inert:
                raise ValueError(f"multitask.{key}={multitask_config[key]!r} {why}")
        if multitask_config and not multitask_config.get('enabled', False) and any(
                key != 'enabled' for key in multitask_config):
            raise ValueError("multitask settings are given but multitask.enabled is not true, so they "
                             "would be ignored")
        if multitask_config.get('enabled', False):
            for key, value, names in (
                    ('fitness_mode', multitask_config.get('fitness_mode', 'mse'), MULTITASK_FITNESS_MODES),
                    ('fitness_aggregation', multitask_config.get('fitness_aggregation', 'mean'),
                     MULTITASK_AGGREGATIONS),
                    ('generalist_bonus_type', multitask_config.get('generalist_bonus_type', 'none'),
                     GENERALIST_BONUS_TYPES),
                    ('two_module_mode', multitask_config.get('two_module_mode', 'none'), TWO_MODULE_MODES),
                    ('modulation_mode', multitask_config.get('modulation_mode', 'full'), MODULATION_MODES)):
                if value not in names:
                    raise ValueError(f"multitask.{key} must be one of {names}, got {value!r}")
            # The hidden activations the multi-task evaluation runs (0.1.0 ran tanh for an unknown name)
            per_task = multitask_config.get('per_task_activation') or {}
            for name in [multitask_config.get('hidden_activation', 'tanh'), *per_task.values()]:
                if name not in ACTIVATION_FUNCTIONS:
                    raise ValueError(f"unknown multi-task activation {name!r}; the names are "
                                     f"{sorted(ACTIVATION_FUNCTIONS)}")
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
                # Read by the multi-task evaluation; 0.1.0 dropped these keys, so a runner could set
                # them only on the MultiTaskConfig object
                fitness_mode=multitask_config.get('fitness_mode', 'mse'),
                modulation_mode=multitask_config.get('modulation_mode', 'full'),
                modulation_strength_override=multitask_config.get('modulation_strength_override', None),
                two_module_mode=multitask_config.get('two_module_mode', 'none'),
                generalist_bonus_type=multitask_config.get('generalist_bonus_type', 'none'),
                generalist_bonus_weight=multitask_config.get('generalist_bonus_weight', 0.0),
                generalist_threshold=multitask_config.get('generalist_threshold', 0.9),
                softmin_temperature=multitask_config.get('softmin_temperature', 0.1),
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
        """Function histograms over the hidden nodes of the last generation's champion.

        The champion is the one :meth:`evaluate_on_data` scores, and the histograms count the
        same nodes as its ``activation_counts``: a position the mask drops computes nothing.
        In 0.1.0 they counted every position of the argmax genome, a dropped one at
        the function a raw CPPN output of 0 decodes to (the palette's middle entry), and named
        a palette entry by its rank in the palette rather than its ``ACTIVATION_LIST`` position.

        Returns:
            An empty dict before the first generation and after the paths that record no
            champion (depth offload, multi-task generations, the multi-GPU strategies).
            Otherwise 'num_positions' (positions on the evaluation axis) and
            'num_hidden_nodes', plus 'activation_histogram' and 'aggregation_histogram'
            (function name -> number of hidden nodes) where functions are assigned per node.
        """
        record = self._last_eval
        if record is None:
            return {}
        mask = np.asarray(record['mask_eval'], dtype=bool)
        result = {'num_positions': int(mask.shape[0]), 'num_hidden_nodes': int(mask.sum())}

        def histogram(indices, names):
            values, numbers = np.unique(np.asarray(indices)[mask], return_counts=True)
            return {names[int(v)]: int(n) for v, n in zip(values, numbers)}

        if record['act_indices'] is not None:
            result['activation_histogram'] = histogram(record['act_indices'], ACTIVATION_LIST)
        if record['agg_indices'] is not None:
            result['aggregation_histogram'] = histogram(record['agg_indices'], AGGREGATION_LIST)
        return result

    def evaluate_best_all_modes(self) -> Optional[Dict[str, float]]:
        """All six fitness metrics of the last generation's champion on the training data.

        The champion's outputs come from the forward it was evaluated with (see
        :meth:`evaluate_on_data`), so every metric describes the evolved network. Must be
        called after a generation.
        """
        record = self._last_eval
        if record is None:
            return None
        inputs = self._cached_inputs
        targets = self._cached_targets
        forward_one = record['forward_one']
        modes = ('mse', 'accuracy', 'hybrid', 'bce', 'soft_accuracy')

        def all_modes(*args):
            outputs = forward_one(inputs, *args)
            return [_eval_single_fitness(outputs, targets, mode) for mode in modes]

        # Inside the vmap over the champion's two copies, as the generation scored it
        values = jax.vmap(all_modes)(*record['champion_args'])
        scored = {mode: float(value[0]) for mode, value in zip(modes, values)}
        # acc_mse is not an _eval_single_fitness mode: accuracy plus a small MSE tiebreak
        scored['acc_mse'] = scored['accuracy'] + 0.01 * scored['mse']
        return {mode: scored[mode]
                for mode in ('mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy')}

    # ========================================================================
    # Execution mode: streaming and depth offload
    # ========================================================================

    @property
    def _recurrence_config(self) -> Optional['EMRConfig']:
        """Alias for `extended_config`, the single source of recurrence truth.

        The depth-offload path used to keep its own `_recurrence_config`, set only
        by an imperative `set_recurrence_config` call that `create_config` never
        made. A YAML-configured recurrent run therefore read an unset config and
        silently degraded to feedforward -- which is why hidden-to-hidden was dead
        in every depth-offload run before this was unified. Aliasing leaves exactly
        one place for the answer to live.
        """
        return self.extended_config

    def set_recurrence_config(self, config: 'EMRConfig') -> None:
        """Enable recurrence imperatively, with the forward mode, iteration level and settling steps the
        configuration would give (`resolve_recurrence`).

        Kept because benchmark scripts drive recurrence this way. It now writes
        `extended_config`, the same field `create_config` writes, rather than a
        parallel one -- previously the two did not meet, so a run that configured
        recurrence through YAML never engaged it. In 0.1.0 it also left
        `activate_time` and `iteration_level` at their old values, so a generation
        settled the old count, and it chose the forward mode by ``enabled and
        iteration_level > 0`` where the configuration uses ``allow_hidden_to_hidden``.
        Called before `create_config`, it is replaced by the configuration's own
        recurrence settings, as before.

        Args:
            config: Recurrence settings; only ``allow_hidden_to_hidden``, ``iteration_level``
                and ``activate_time`` decide the mode, level and steps. A None ``activate_time``
                takes an explicit ``substrate.activate_time`` from the configuration, else
                ``2 (2**max_depth + 1)``; before `create_config`, when the depth is unknown,
                it derives nothing.

        Raises:
            ValueError: With hidden-to-hidden connections on, a count that is not an integer of
                at least 1, or one that differs from the configuration's
                ``substrate.activate_time``; ``enabled`` false with ``allow_hidden_to_hidden``
                true (`check_recurrence_switch`; an object without ``enabled`` counts as enabled).
        """
        check_recurrence_switch(getattr(config, 'enabled', True), config.allow_hidden_to_hidden)
        mode, level, steps = resolve_recurrence(
            config, self.max_depth,
            {"set_recurrence_config's activate_time": config.activate_time,
             'substrate.activate_time': self._substrate_activate_time})
        self.extended_config = config
        self._forward_mode = mode
        self.iteration_level = level
        if steps is not None:
            self.activate_time = steps

    def get_forward_mode(self) -> ForwardPassMode:
        """Current forward-pass mode."""
        return self._forward_mode

    def set_forward_mode(self, mode: ForwardPassMode) -> None:
        """Set the forward-pass mode directly.

        Args:
            mode: DENSE_ONLY for feedforward, HYBRID_SPARSE_HH for recurrent.

        Raises:
            NotImplementedError: For FULL_SPARSE, which 0.1.0 accepted and ran as DENSE_ONLY.
        """
        if mode is ForwardPassMode.FULL_SPARSE:
            raise NotImplementedError(
                "ForwardPassMode.FULL_SPARSE is not implemented: 0.1.0 accepted it and ran it "
                "with no h->h connections (as DENSE_ONLY). Use HYBRID_SPARSE_HH for recurrence.")
        self._forward_mode = mode

    def _apply_execution_config(self) -> None:
        """Mirror `execution_config` onto the flat attributes the strategies read.

        The strategy functions and the existing tests both address these by name,
        so the config object and the imperative setters write the same place rather
        than competing.
        """
        config = self.execution_config
        self._depth_offload_threshold = config.depth_threshold
        self._eval_chunk_levels = config.eval_chunk_levels
        self._eval_chunk_positions = config.eval_chunk_positions
        self._cppn_pos_chunk_size = config.cppn_pos_chunk_size
        # `max_lookahead_positions` is now a MEMORY budget, not a truncation trigger:
        # a level larger than it is computed in slabs, which bounds memory without
        # changing the substrate. Exceeding it only truncates -- zeroed variance,
        # subdivision halted at and below that level -- when explicitly asked via
        # `truncate_variance_lookahead`. The old "never truncate" encoding of
        # float('inf') is gone: it disabled truncation by also disabling the budget,
        # so the default materialized the whole next level and would OOM at depth 12
        # (67M positions per genome).
        self._max_lookahead_positions = config.max_lookahead_positions
        self._truncate_variance_lookahead = config.truncate_variance_lookahead
        self._skip_dead_levels = config.skip_dead_levels
        self._offload_verbose = config.verbose
        self._target_shard_gb = config.target_shard_gb
        self._target_eval_shard_gb = config.target_eval_shard_gb
        self._enable_disk_offload = config.enable_disk_offload
        self._disk_offload_path = config.disk_offload_path
        self._target_cpu_gb = config.target_cpu_gb
        self._max_population_chunk = config.max_population_chunk
        self._use_memmap = config.use_memmap
        self._memmap_path = config.memmap_path
        self._use_cpu_grid = config.use_cpu_grid
        self._recurrent_target_memory_gb = config.recurrent_target_memory_gb

    def configure_offloading(
        self,
        depth_threshold: int = 9,
        eval_chunk_levels: int = 3,
        eval_chunk_positions: int = 1_000_000,
        cppn_pos_chunk_size: int = 100_000,
        max_lookahead_positions: int = 5_000_000,
        verbose: bool = False,
        use_cpu_grid: bool = False,
    ) -> None:
        """Configure depth offloading imperatively.

        Kept alongside the `execution:` config block so existing code and tests
        keep working; both write `execution_config`.

        `max_lookahead_positions` is a memory budget. Levels larger than it are
        computed in slabs, which bounds memory and leaves the substrate unchanged,
        so asking for a limit no longer implies accepting a different substrate --
        it did before slab computation existed. Truncation (zeroed variance, and
        subdivision halted at and below that level) remains available through
        `truncate_variance_lookahead` on the config, and stays default off.

        Args:
            depth_threshold: Depth at or above which `auto` selects offloading.
            eval_chunk_levels: Levels loaded at once during evaluation.
            eval_chunk_positions: Maximum positions per device chunk.
            cppn_pos_chunk_size: Maximum positions per CPPN query.
            max_lookahead_positions: Levels larger than this skip variance lookahead.
            verbose: Per-level statistics.
            use_cpu_grid: Keep the grid in host RAM, needed from depth 12.
        """
        self.execution_config.depth_threshold = depth_threshold
        self.execution_config.eval_chunk_levels = eval_chunk_levels
        self.execution_config.eval_chunk_positions = eval_chunk_positions
        self.execution_config.cppn_pos_chunk_size = cppn_pos_chunk_size
        self.execution_config.max_lookahead_positions = max_lookahead_positions
        self.execution_config.verbose = verbose
        self.execution_config.use_cpu_grid = use_cpu_grid
        self._apply_execution_config()

    def configure_sharding(
        self,
        target_shard_gb: float = 0.5,
        target_eval_shard_gb: float = 1.0,
        enable_disk_offload: bool = False,
        disk_offload_path: str = '/tmp/emr_offload',
        max_population_chunk: Optional[int] = None,
        target_cpu_gb: float = 32.0,
        use_memmap: bool = True,
        memmap_path: str = '/tmp/emr_memmap',
    ) -> None:
        """Configure intra-level sharding and host residency imperatively.

        Args:
            target_shard_gb: Device memory per position shard during CPPN queries.
            target_eval_shard_gb: Device memory per shard during evaluation.
            enable_disk_offload: Spill to disk at extreme depth.
            disk_offload_path: Directory for disk offloading.
            max_population_chunk: Override the computed population chunk size.
            target_cpu_gb: Host RAM budget for the weight matrices.
            use_memmap: Back large chunks with memory-mapped files.
            memmap_path: Directory for memmap files.
        """
        self.execution_config.target_shard_gb = target_shard_gb
        self.execution_config.target_eval_shard_gb = target_eval_shard_gb
        self.execution_config.enable_disk_offload = enable_disk_offload
        self.execution_config.disk_offload_path = disk_offload_path
        self.execution_config.max_population_chunk = max_population_chunk
        self.execution_config.target_cpu_gb = target_cpu_gb
        self.execution_config.use_memmap = use_memmap
        self.execution_config.memmap_path = memmap_path
        self._apply_execution_config()

    def resolve_execution_mode(self) -> ExecutionMode:
        """Resolve the configured mode against `max_depth`. Never returns AUTO.

        On a GPU backend the resolution is memory-aware: AUTO escalates a
        STANDARD selection to DEPTH_OFFLOAD when the standard path's calibrated
        working-set estimate exceeds the device (the d7/pop1000 wall the depth
        thresholds alone could not see). On CPU, and whenever sizing is
        unavailable, resolution is exactly the static threshold logic.
        """
        from emr_hyperneat.execution import (  # noqa: E501
            resolve_mode,
        )
        population = getattr(self, 'population_size', None)
        device_bytes = None
        if population is not None and jax.default_backend() == 'gpu':
            try:
                stats = jax.local_devices()[0].memory_stats() or {}
                device_bytes = stats.get('bytes_limit')
            except Exception:
                device_bytes = None
        coords_in = getattr(self, '_cached_input_coords', None)
        coords_out = getattr(self, '_cached_output_coords', None)
        return resolve_mode(
            self.execution_config, self.max_depth,
            population=population,
            num_inputs=int(coords_in.shape[0]) if coords_in is not None else 3,
            num_outputs=int(coords_out.shape[0]) if coords_out is not None else 1,
            device_bytes=device_bytes,
        )

    def generation_execution_mode(self) -> str:
        """The execution mode a generation of `run_generation` runs: 'depth_offload' when the
        resolution (memory-aware on a GPU) says so, otherwise 'streaming' or 'standard' by the
        build step the generation loop selects (`enable_streaming`). `run_until_threshold`
        reports its route's own mode in its result.

        Returns:
            'standard', 'streaming' or 'depth_offload'.
        """
        if self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD:
            return 'depth_offload'
        return 'streaming' if self.enable_streaming else 'standard'

    def should_use_offloading(self) -> bool:
        """Whether this generation will take the depth-offload path."""
        return self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD

    def _calculate_optimal_pos_chunk_size(
        self,
        pop_size: int,
        num_sources: int,
        target_gb: Optional[float] = None,
        max_nodes: Optional[int] = None,
    ) -> int:
        """Positions per CPPN query chunk. See `execution.chunking`.

        `max_nodes` sizes the dominant forward-carry term; when the caller has
        no transformed-CPPN tuple in hand, the genome default is used.
        """
        from emr_hyperneat.emrhyperneat_base import (  # noqa: E501
            DEFAULT_MAX_NODES,
        )
        return optimal_position_chunk_size(
            pop_size=pop_size,
            num_sources=num_sources,
            target_gb=self._target_shard_gb if target_gb is None else target_gb,
            max_chunk=self._cppn_pos_chunk_size,
            max_nodes=DEFAULT_MAX_NODES if max_nodes is None else max_nodes,
        )

    def _calculate_population_chunk_size(
        self,
        pop_size: int,
        total_positions: int,
        num_inputs: int = 3,
        num_outputs: int = 1,
    ) -> int:
        """Genomes per chunk that fit the host RAM budget. See `execution.chunking`."""
        return population_chunk_size(
            pop_size=pop_size,
            total_positions=total_positions,
            target_cpu_gb=self._target_cpu_gb,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_population_chunk=self._max_population_chunk,
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Memory counters accumulated by the depth-offload path."""
        return {
            'cpu_memory_peak_gb': self._cpu_memory_peak_gb,
            'gpu_transfers': self._gpu_transfers_count,
            'depth_offload_enabled': self.should_use_offloading(),
            'depth_threshold': self._depth_offload_threshold,
            'execution_mode': self.resolve_execution_mode().value,
        }

    def run_generation(
        self,
        state: Any,
        problem: Any,
        skip_metrics: bool = False,
        verbose: bool = True,
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation, dispatching on execution mode and multi-task.

        Two independent dispatches happen here. Execution mode comes first because
        depth offload fuses matrix building with evaluation and so replaces the
        whole generation rather than a step inside it; streaming and standard both
        continue into the normal loop, which selects the build step itself.

        Args:
            state: Algorithm state
            problem: Problem instance (or MultiTaskProblem for multi-task)
            skip_metrics: Skip metrics computation
            verbose: Verbose output

        Returns:
            Tuple of (new_state, metrics)
        """
        if self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD:
            # Imported here rather than at module scope: the strategy modules
            # import from this one, so a top-level import would form a cycle.
            from emr_hyperneat.execution.offload import (  # noqa: E501
                run_generation_offload,
            )
            return run_generation_offload(self, state, problem, skip_metrics)

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

    def _build_forward(
        self,
        W1: jnp.ndarray,
        W2: jnp.ndarray,
        sparse_hh: Optional['SparseHiddenConnections'],
        neuromod_data: Optional[Dict[str, Any]],
        act_indices: Optional[jnp.ndarray],
        hh_agg_indices: Optional[jnp.ndarray],
        ih_agg_indices: Optional[jnp.ndarray],
        ih_global_agg_fn: Optional[Callable],
        use_dynamic_eval: bool,
        use_ih_aggregation: bool,
        pop_size: int,
        total_positions: int,
    ) -> Tuple[Callable, Tuple, bool]:
        """This generation's per-genome forward and the population arrays it runs on.

        One function per configuration serves the population's fitness and
        :meth:`evaluate_on_data`, so a held-out score describes the network that was evolved.
        Feed-forward and recurrent evaluation share every per-node transformation (input
        drive, activation, neuromodulation); the recurrent one settles it with
        :func:`settle_hidden`, so without hidden-to-hidden connections the two compute the
        same arithmetic (bit for bit on CPU but for the exception :func:`settle_hidden` names).

        Args:
            W1, W2: Population weight matrices as evaluated (compacted on the recurrent path).
            sparse_hh: Hidden-to-hidden connections, or None for feed-forward evaluation.
            neuromod_data: The generation's neuromodulation arrays, or None.
            act_indices: Per-node activation indices, or None.
            hh_agg_indices: Per-node hidden-to-hidden aggregation indices, or None.
            ih_agg_indices: Per-node input-to-hidden aggregation indices, or None.
            ih_global_agg_fn: One input-to-hidden aggregation for every node, or None.
            use_dynamic_eval: A dynamic-functions mode is on.
            use_ih_aggregation: An input-to-hidden aggregation mode is on.
            pop_size: Genomes evaluated.
            total_positions: Positions evaluated.

        Returns:
            ``(forward_one, genome_args, chunk_by_scan)``. ``forward_one(inputs, *args)`` is
            one genome's outputs; ``genome_args`` are the population arrays (population axis
            first, None where unused); ``chunk_by_scan`` selects the scan-based chunked
            evaluation, which the plain recurrent path has always used.
        """
        recurrent = sparse_hh is not None
        palette = self.df_palette
        num_activations = self.df_num_activations
        hidden_fn = self._hidden_act_fn if self._hidden_act_fn is not None else jnp.tanh
        output_fn = self._output_act_fn if self._output_act_fn is not None else jax.nn.sigmoid
        activate_time = self.activate_time
        hh_mode = self.hh_activation_mode
        hh_agg_mode = self.hh_aggregation_mode
        num_hh_agg = self.agg_num_aggregations
        per_node_agg = ih_agg_indices is not None
        agg_args = (ih_agg_indices if per_node_agg
                    else jnp.zeros((pop_size, total_positions), dtype=jnp.int32))
        drive_kwargs = dict(
            global_agg_fn=ih_global_agg_fn,
            per_node_aggregation=per_node_agg,
            true_aggregation=self.agg_use_true_aggregation,
            num_aggregations=self.agg_num_aggregations,
        )
        hh_args: Tuple = ()
        if recurrent:
            hh_args = (sparse_hh.from_indices, sparse_hh.to_indices, sparse_hh.weights,
                       sparse_hh.valid_mask, hh_agg_indices)

        def activation(act_idx):
            # A settling step applies hh_activation_mode's activation; a dense pass the
            # configured one.
            if recurrent:
                return node_activation(hh_mode, act_idx, hidden_fn, palette, num_activations)
            return functools.partial(hidden_activation, act_indices=act_idx,
                                     hidden_act_fn=hidden_fn, palette=palette,
                                     num_activations=num_activations)

        def hidden_state(inputs, W1_i, agg_idx, node_fn, hh):
            drive = input_drive(inputs, W1_i, agg_idx, **drive_kwargs)
            if not recurrent:
                return node_fn(drive)
            from_idx, to_idx, hh_w, valid, hh_agg_idx = hh
            return settle_hidden(drive, from_idx, to_idx, hh_w, valid, activate_time, node_fn,
                                 hh_agg_idx, hh_agg_mode, num_hh_agg)

        if neuromod_data is not None:
            mode = neuromod_data['mode']
            if mode == 'static_gating':
                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, gates, *hh):
                    act = activation(act_idx)
                    h = hidden_state(inputs, W1_i, agg_idx, lambda p: act(p) * gates, hh)
                    return output_fn(h @ W2_i)
                return forward_one, (W1, W2, act_indices, agg_args,
                                     neuromod_data['static_gates']) + hh_args, False

            if mode == 'context_gating':
                context_influence = neuromod_data['context_influence']
                context_idx = self._multitask_state.get('current_task_idx', 0)

                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, base_g, ctx_mod, *hh):
                    act = activation(act_idx)
                    gates = jax.nn.sigmoid(base_g + context_influence * ctx_mod[context_idx])
                    h = hidden_state(inputs, W1_i, agg_idx, lambda p: act(p) * gates, hh)
                    return output_fn(h @ W2_i)
                return forward_one, (W1, W2, act_indices, agg_args,
                                     neuromod_data['base_gates'],
                                     neuromod_data['context_modulation']) + hh_args, False

            if mode == 'modulatory_neurons':
                mod_type = neuromod_data['mod_connection_type']
                mod_decay = neuromod_data['mod_decay']

                def modulated(pre, act, mod_w, mask):
                    h = act(pre)
                    # The mean activity of this genome's own modulatory neurons
                    mod_activity = (jnp.sum(h * mask, axis=1, keepdims=True)
                                    / jnp.maximum(jnp.sum(mask), 1))
                    mod_signal = jnp.tanh(mod_activity) * mod_decay
                    if mod_type == 'multiplicative':
                        return h * (1.0 + mod_signal * mod_w)
                    if mod_type == 'additive':
                        return h + mod_signal * mod_w
                    return h * jax.nn.sigmoid(mod_signal * mod_w)

                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, mod_w, mask, *hh):
                    act = activation(act_idx)
                    h = hidden_state(inputs, W1_i, agg_idx,
                                     lambda p: modulated(p, act, mod_w, mask), hh)
                    return output_fn(h @ W2_i)
                return forward_one, (W1, W2, act_indices, agg_args,
                                     neuromod_data['mod_connection_weights'],
                                     neuromod_data['mod_mask']) + hh_args, False

            if mode == 'true_neuromodulation':
                neurotransmitter = neuromod_data['neurotransmitter']
                mod_strength = neuromod_data['mod_strength']
                modulation_mode = neuromod_data['modulation_mode']
                use_output_inversion = neuromod_data['use_output_inversion']
                num_nt = neuromod_data['num_nt']
                node_rule = neuromod_data['node_rule']

                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, receptors, base_g, *hh):
                    act = activation(act_idx)
                    node = true_neuromodulation_node_fn(act, receptors, base_g, neurotransmitter,
                                                        mod_strength, num_nt, modulation_mode,
                                                        node_rule)
                    h = hidden_state(inputs, W1_i, agg_idx, node, hh)
                    return invert_outputs(output_fn(h @ W2_i), neurotransmitter,
                                          use_output_inversion, num_nt, node_rule)
                return forward_one, (W1, W2, act_indices, agg_args,
                                     neuromod_data['receptor_densities'],
                                     neuromod_data['base_gains']) + hh_args, False

            raise ValueError(f"Unknown neuromodulation mode: {mode}")

        dynamic = ((use_dynamic_eval and (act_indices is not None
                                          or self._hidden_act_fn is not None))
                   or use_ih_aggregation or (recurrent and hh_agg_indices is not None))
        if dynamic and not recurrent:
            def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx):
                return dense_substrate_forward(
                    inputs, W1_i, W2_i, act_idx, agg_idx,
                    hidden_act_fn=hidden_fn, output_act_fn=output_fn, palette=palette,
                    num_activations=num_activations, **drive_kwargs,
                )
            return forward_one, (W1, W2, act_indices, agg_args), False
        if dynamic:
            def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, *hh):
                h = hidden_state(inputs, W1_i, agg_idx, activation(act_idx), hh)
                return output_fn(h @ W2_i)
            return forward_one, (W1, W2, act_indices, agg_args) + hh_args, False
        if not recurrent:
            def forward_one(inputs, W1_i, W2_i):
                # Fixed tanh hidden and sigmoid output, whatever output_activation says
                return dense_substrate_forward(inputs, W1_i, W2_i,
                                               hidden_act_fn=jnp.tanh,
                                               output_act_fn=jax.nn.sigmoid)
            return forward_one, (W1, W2), False

        def forward_one(inputs, W1_i, W2_i, from_idx, to_idx, hh_w, valid):
            return forward_hybrid_vmapped(inputs, W1_i, W2_i, from_idx, to_idx, hh_w, valid,
                                          activate_time)
        return forward_one, (W1, W2) + hh_args[:4], True

    def _record_evaluation(
        self,
        forward_one: Callable,
        genome_args: Tuple,
        fitnesses: jnp.ndarray,
        cppn_population: Tuple,
        masks_full: jnp.ndarray,
        masks_eval: jnp.ndarray,
        eval_positions: Optional[jnp.ndarray],
        act_indices: Optional[jnp.ndarray],
        ih_agg_indices: Optional[jnp.ndarray],
        next_state: Any,
    ) -> None:
        """Keep what :meth:`evaluate_on_data` needs to re-run this generation's champion.

        The champion is the genome ``next_state`` carries first: TensorNEAT places the best
        genome of a generation, unmutated, first in the next population. Under a tie for the
        best fitness its sorts keep the highest index of the top species, not the first
        index ``argmax`` returns, so a tie is resolved by comparing genomes.

        Only the champion's slices are kept, twice (a leading axis of two): XLA lowers a
        batch of one genome to other reduction kernels, which moved the fitness by one
        float32 ulp, while two copies lower like the population batch. On CPU the
        training-data score then reproduces the champion's fitness bit for bit.
        """
        best = np.asarray(fitnesses)
        tied = np.flatnonzero(best == best.max())
        champion = int(tied[0])
        if tied.size > 1:
            kept = self._compiled_ask(next_state)
            same = np.ones(tied.size, dtype=bool)
            for part, kept_part in zip(cppn_population[:2], kept[:2]):
                rows = np.asarray(part)[tied]
                one = np.asarray(kept_part[0])
                equal = (rows == one) | (np.isnan(rows) & np.isnan(one))
                same &= equal.reshape(tied.size, -1).all(axis=1)
            if same.any():
                champion = int(tied[np.argmax(same)])
        pair = jnp.array([champion, champion])
        # With a padded compact axis the per-position arrays run past the real positions; the
        # record describes the substrate, so it keeps the real ones (the champion's evaluation
        # arguments keep the padded shapes the evaluation used).
        n_real = None if eval_positions is None else len(eval_positions)
        self._last_eval = {
            'forward_one': forward_one,
            'champion_args': tuple(None if a is None else a[pair] for a in genome_args),
            'champion_index': champion,
            'champion_fitness': float(fitnesses[champion]),
            'generation': self._current_generation,
            'champion_genome': tuple(np.asarray(part[champion]) for part in cppn_population[:2]),
            'mask_full': np.asarray(masks_full[champion]),
            'mask_eval': np.asarray(masks_eval[champion])[:n_real],
            'eval_positions': None if eval_positions is None else np.asarray(eval_positions),
            'act_indices': (None if act_indices is None
                            else np.asarray(act_indices[champion])[:n_real]),
            'agg_indices': (None if ih_agg_indices is None
                            else np.asarray(ih_agg_indices[champion])[:n_real]),
            'W1_full': self._cached_W1[champion],
            'W2_full': self._cached_W2[champion],
        }

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
            masks_A = substrate_masks(
                all_weights_for_variance, h_grid, self.variance_threshold,
                initial_depth=self.initial_depth,
                legacy_level0_variance=self._legacy_level0_variance,
            )
            # Observable for substrate depth, mirroring the offload path: without
            # it there is no way to see which levels a mode actually reached.
            # Pure read; results untouched.
            masks_np_reach = np.asarray(masks_A)
            self._last_reached_per_level = [
                int(np.count_nonzero(
                    masks_np_reach[:, h_grid.level_offsets_static[lv]:h_grid.level_offsets_static[lv + 1]]
                ))
                for lv in range(h_grid.num_levels)
            ]
            del masks_np_reach
            step_times['step2_variance_masks'] = time.perf_counter() - start

            # === STEP 4: Build W1/W2 Matrices ===
            start = time.perf_counter()
            # tanh scaling, optional locality penalty, then mask and threshold.
            # W1: (pop, num_inputs, total_positions); W2: (pop, total_positions, num_outputs)
            W1, W2 = substrate_weight_matrices(
                input_all_weights, output_all_weights, masks_A, self.max_weight,
                weight_threshold=WEIGHT_THRESHOLD,
                locality_radius=self.locality_radius,
                input_coords=input_coords,
                output_coords=output_coords,
                positions=all_positions,
            )
            if self.locality_radius is not None and self.verbose:
                print(f"[Locality] Applied penalty (r={self.locality_radius}) to W1 and W2")

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
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )
            self._cached_self_conn = self_conn_outputs
            if self.verbose:
                print(f"[CPPN Self-Conn] shape={self_conn_outputs.shape}, "
                      f"indices={self._cppn_output_indices}")

        # Cache sparse H→H connections for multi-task evaluation
        # This enables run_generation_multitask() to use H→H connections
        self._cached_sparse_hh = sparse_hh

        # Multi-task is excluded for a different reason, and a sharper one.
        # run_generation_multitask calls this function to build the substrate and
        # then re-evaluates each task from self._cached_W1/_cached_W2, which are
        # cached *above* deliberately so they stay ceiling-sized. The per-position
        # neuromodulation state (_static_gates and friends) is derived *below*, so
        # compaction makes it compact while the cached weights stay full, and the
        # two no longer line up. That combination broke 165 tests across the
        # multi-task, two-module and per-task-activation paths, every one of them
        # "Incompatible shapes for broadcasting" at generation 0.
        #
        # Caching the compact matrices instead would make multi-task self-consistent
        # but changes what every other consumer of _cached_W1 receives, so it is a
        # deliberate follow-up rather than something to slip in here.
        _neuromod_mode = getattr(self.neuromod_config, 'mode', None) if self.neuromod_config else None
        _multitask_on = (self.multitask_config is not None
                         and getattr(self.multitask_config, 'enabled', False))
        _compaction_blocked = _neuromod_mode == 'modulatory_neurons' or _multitask_on

        # In-generation view of the self-connection query. Compaction narrows
        # this local; self._cached_self_conn stays ceiling-sized for multi-task.
        self_conn_eval = self._cached_self_conn
        ceiling_positions = W1.shape[2] if W1.ndim > 2 else W1.shape[1]
        # The ceiling-sized mask, and the ceiling position of every evaluated position (None:
        # the identity), for evaluate_on_data's substrate statistics.
        masks_full = masks_A
        eval_positions = None
        # Virtual positions appended to the compact axis (execution.compaction_bucket 'auto');
        # 0 without compaction or with 'exact'. Counts and records use the real positions only.
        compact_pad = 0
        if self._forward_mode != ForwardPassMode.DENSE_ONLY and not _compaction_blocked:
            from emr_hyperneat.execution.compaction import (
                bucket_active_set,
                compact_weights,
                find_active_positions,
                pad_positions,
                remap_sparse_hh,
            )

            _active = find_active_positions(masks_A)
            # The random per-node modes draw their indices with the position count as the shape,
            # and a longer draw is not the same draw: they keep exact shapes.
            _random_draws = (self.dynamic_functions_mode in ('random_fixed', 'random_generation')
                             or getattr(self, 'agg_mode', None) in ('random_fixed', 'random_generation'))
            if _active.num_active and _active.num_active < _active.total_positions:
                # Bucketed: the compact axis is padded to a bucketed length so its shape repeats
                # across generations. A count that lands exactly on a bucket length needs no padding
                # but still takes the host path: there the exact path's device gathers, and the later
                # ops on the gathered mask, compiled a separate set of executables (13 on a GPU) the
                # first time such a count occurred.
                bucketed = self.execution_config.compaction_bucket == 'auto' and not _random_draws
                if bucketed:
                    _active = bucket_active_set(_active)
                    compact_pad = _active.length - _active.num_active
                # W2 is (pop, positions, n_out) here, unlike the offload path's host
                # layout, so it is moved to positions-last for the shared helper and
                # back afterwards.
                W1, _W2_last = compact_weights(
                    W1, jnp.transpose(W2, (0, 2, 1)), _active
                )
                W2 = jnp.transpose(_W2_last, (0, 2, 1))
                if bucketed:
                    masks_A = pad_positions(masks_A, _active, axis=1, fill=False)
                else:
                    masks_A = masks_A[:, _active.indices]
                eval_positions = _active.indices
                sparse_hh = remap_sparse_hh(sparse_hh, _active)
                # The self-connection query above ran over h_grid.all_positions, so
                # it is still ceiling-sized while everything else is compact, and
                # the `cppn_output` modes read act/agg indices straight off it.
                #
                # Compacted into a LOCAL, never into self._cached_self_conn. That
                # distinction is load-bearing: multi-task evaluation consumes the
                # cached self-connections alongside the uncompacted _cached_W1, so
                # shrinking the cache in place mismatches them. Doing exactly that
                # broke 165 tests across the multi-task and two-module paths, all
                # with "Incompatible shapes for broadcasting" at generation 0.
                # Position is axis 1 here -- (pop, positions, cppn_outputs).
                if self_conn_eval is not None:
                    if bucketed:
                        # Finite raw outputs for the virtual nodes: they decode to an ordinary
                        # palette entry, finite at a zero pre-activation, and feed nothing.
                        self_conn_eval = pad_positions(self_conn_eval, _active, axis=1, fill=0.0)
                    else:
                        self_conn_eval = self_conn_eval[:, _active.indices, :]
                if self.verbose:
                    print(f"[Compact] recurrent evaluation on {_active.num_active:,} "
                          f"active of {_active.total_positions:,} positions "
                          f"({_active.reduction:,.0f}x reduction)"
                          + (f", padded to {_active.length:,}" if compact_pad else ""))


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
        # eval_length is the position axis of the arrays, padding included (the chunking heuristics
        # key on it: it is the compiled shape); total_positions counts the real positions, for the reports.
        eval_length = W1.shape[2] if len(W1.shape) > 2 else W1.shape[1]
        total_positions = eval_length - compact_pad

        # Auto-compute chunk size (same heuristics as CPPN queries)
        if eval_chunk_size == -1:
            eval_chunk_size = _auto_eval_chunk_size(eval_length, pop_size)

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
            if self.df_interpretation == 'variance':
                weight_var = jnp.var(W1, axis=1)  # (pop, num_positions)
                normalized = jnp.tanh(weight_var * 5)
            elif self.df_interpretation == 'sign':
                normalized = jnp.mean(jnp.sign(W1), axis=1)  # (pop, num_positions)
            else:
                # magnitude_bio, and the default: mean absolute incoming weight
                mean_abs = jnp.mean(jnp.abs(W1), axis=1)  # (pop, num_positions)
                normalized = -jnp.tanh(mean_abs * 2 - 1)
            # With a palette the decode picks a palette position and maps it to the function's
            # ACTIVATION_LIST index, as the cppn_output and random modes do. 0.1.0 left the position
            # unmapped, and evaluation (which expects palette members) gave every node whose
            # position was not also a palette member an output of 0.
            num_options = (len(self.df_palette) if self.df_palette is not None
                           else self.df_num_activations)
            act_indices = continuous_to_index(normalized, num_options, self.df_index_decode)
            if self.df_palette is not None:
                act_indices = self.df_palette[act_indices]
        elif self.dynamic_functions_mode in ('random_fixed', 'random_generation'):
            # Random per-node activation (a negative control): drawn at the first generation and
            # kept (random_fixed, per population slot) or drawn every generation
            # (random_generation); see random_function_indices for the two scopes.
            num_activations = (len(self.df_palette) if self.df_palette is not None
                               else self.df_num_activations)
            fixed = self.dynamic_functions_mode == 'random_fixed'
            if not fixed or self._random_fixed_indices is None:
                draw = random_function_indices(
                    self._random_scope, self._random_mode_seed, state.randkey,
                    9997 if fixed else 9999, None if fixed else self._current_generation,
                    num_activations, self.df_palette, pop_size, total_positions)
                if fixed:
                    self._random_fixed_indices = draw
            act_indices = self._random_fixed_indices if fixed else draw
        elif self.dynamic_functions_mode == 'cppn_output':
            # CPPN output mode: activation index from the dedicated CPPN output at each
            # node's self-connection, or from W1 for a one-output CPPN
            num_activations = len(self.df_palette) if self.df_palette is not None else self.df_num_activations
            act_indices = cppn_output_function_indices(
                self_conn_eval, masks_A, W1,
                channel=self._cppn_output_indices['activation'],
                num_options=num_activations,
                num_cppn_outputs=self.num_cppn_outputs,
                palette=self.df_palette,
                decode=self.df_index_decode,
            )

        # Step 4b: Compute H→H aggregation indices if dynamic mode enabled
        hh_agg_indices = None
        if self.hh_aggregation_mode == 'dynamic':
            if self.num_cppn_outputs > 1 and self_conn_eval is not None and self.agg_mode == 'cppn_output':
                # Use dedicated CPPN output for H→H aggregation (multi-output CPPN)
                agg_output_idx = self._cppn_output_indices['aggregation']
                agg_raw = self_conn_eval[:, :, agg_output_idx]  # (pop, positions)
                hh_agg_indices = continuous_to_index(agg_raw, self.agg_num_aggregations, self.agg_index_decode)
            else:
                # Fallback: the input side's weight interpretation. 0.1.0 re-implemented it here for
                # magnitude_bio and variance only, so sign_uniformity and fan_in ran magnitude_bio
                # on the hidden-to-hidden side while the input side honored them; the two it had are
                # the same arithmetic.
                hh_agg_indices = aggregation_from_weight_interpretation(
                    W1, self.agg_interpretation, self.agg_num_aggregations, self.agg_index_decode)

        # Step 4c: Compute I→H aggregation indices based on aggregation mode
        ih_agg_indices = None
        ih_global_agg_fn = None
        if self.agg_mode == 'global':
            ih_global_agg_fn = AGGREGATION_FUNCTIONS.get(
                self.agg_global_function, AGGREGATION_FUNCTIONS['sum']
            )
        elif self.agg_mode == 'cppn_output':
            # Dedicated CPPN output at the self-connection, or W1 for a one-output CPPN
            ih_agg_indices = cppn_output_function_indices(
                self_conn_eval, masks_A, W1,
                channel=self._cppn_output_indices['aggregation'],
                num_options=self.agg_num_aggregations,
                num_cppn_outputs=self.num_cppn_outputs,
                decode=self.agg_index_decode,
            )
        elif self.agg_mode == 'weight_interpretation':
            ih_agg_indices = aggregation_from_weight_interpretation(
                W1, self.agg_interpretation, self.agg_num_aggregations, self.agg_index_decode
            )
        elif self.agg_mode in ('random_fixed', 'random_generation'):
            # Random per-node aggregation, as the activation's random modes (HMR's salts 7777, 8888)
            fixed = self.agg_mode == 'random_fixed'
            if not fixed or self._random_fixed_agg_indices is None:
                draw = random_function_indices(
                    self._random_agg_scope, self._random_agg_seed, state.randkey,
                    7777 if fixed else 8888, None if fixed else self._current_generation,
                    self.agg_num_aggregations, None, pop_size, total_positions)
                if fixed:
                    self._random_fixed_agg_indices = draw
            ih_agg_indices = self._random_fixed_agg_indices if fixed else draw

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
                context_modulation = jnp.zeros((pop_size, context_dim, eval_length))

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

                # Each genome's modulatory neurons: mod_neuron_ratio of its own active positions (at least one),
                # those of highest input-weight variance
                weight_var = jnp.var(W1, axis=1)  # (pop, num_positions)
                mod_mask = modulatory_neuron_masks(weight_var, masks_A, mod_ratio)  # (pop, num_positions)

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
                            and self_conn_eval is not None):
                        # Exact HMR/paper reproduction: raw CPPN self-connection query at
                        # (x,y,x,y,bias); index 0 = weight output -- the SAME source HMR uses
                        # (weight_values = self_conn_outputs[:, :, 0]). Already bounded ~[-1, 1].
                        weight_normalized = self_conn_eval[:, :, 0]
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
                neuromod_data['node_rule'] = self.neuromod_config.node_rule
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

        # One per-genome forward for this configuration: the population's fitness here and
        # evaluate_on_data's held-out score both come from it (see _build_forward).
        recurrent = (sparse_hh is not None
                     and self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH)
        forward_one, genome_args, chunk_by_scan = self._build_forward(
            W1, W2, sparse_hh if recurrent else None, neuromod_data, act_indices,
            hh_agg_indices, ih_agg_indices, ih_global_agg_fn, use_dynamic_eval,
            use_ih_aggregation, pop_size, eval_length,
        )
        fitness_mode_val = self.fitness_mode

        def eval_one(*args):
            outputs = forward_one(inputs_batch, *args)
            return _eval_single_fitness(outputs, targets_batch, fitness_mode_val)

        if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
            if chunk_by_scan:
                # Scan-based: one compiled dispatch, not ceil(pop/chunk) launches -- the
                # loop form measured +116-126% at d5/chunk 10.
                fitnesses = chunked_vmap_eval(eval_one, eval_chunk_size, *genome_args)
            else:
                fitness_chunks = []
                for chunk_start in range(0, pop_size, eval_chunk_size):
                    chunk_end = min(chunk_start + eval_chunk_size, pop_size)
                    fitness_chunks.append(jax.vmap(eval_one)(*[
                        None if arg is None else arg[chunk_start:chunk_end]
                        for arg in genome_args
                    ]))
                fitnesses = jnp.concatenate(fitness_chunks)
        else:
            fitnesses = jax.vmap(eval_one)(*genome_args)

        step_times['step5_evaluation'] = time.perf_counter() - start

        # === STEP 5b: Fitness Rescaling (optional) ===
        # When target variance is small, 1-MSE fitness compresses the useful range.
        # Convert to R² = 1 - MSE/var to spread the signal for evolutionary selection.
        if self._fitness_rescale_variance is not None:
            mse_values = 1.0 - fitnesses  # Recover MSE from 1-MSE fitness
            r2_values = 1.0 - mse_values / self._fitness_rescale_variance
            fitnesses = jnp.maximum(0.0, r2_values)

        # NaN (an overflowing settling, a NaN target) becomes -inf before selection sees it.
        fitnesses, nan_fitness_count = sanitize_fitness(fitnesses)

        # Cache per-individual fitness for external access (e.g., Baldwin Effect)
        self._cached_fitnesses = fitnesses

        # === STEP 6: NEAT Evolution ===
        start = time.perf_counter()
        new_state = self._compiled_tell(state, fitnesses)
        step_times['step6_neat_evolution'] = time.perf_counter() - start

        self._record_evaluation(
            forward_one, genome_args, fitnesses, cppn_population,
            masks_full, masks_A, eval_positions, act_indices, ih_agg_indices, new_state,
        )

        # Update generation counter
        self._current_generation += 1

        # Cache fitnesses for NAS substrate extraction
        self._last_fitnesses = fitnesses

        # === Metrics ===
        total_time = time.perf_counter() - total_start

        # Compute fitness statistics (over the finite entries when NaN was sanitized)
        best_fitness, mean_fitness, min_fitness, std_fitness = fitness_statistics(
            fitnesses, nan_fitness_count)
        max_fitness = best_fitness

        custom_metrics = {
            'total_positions': total_positions,
            'step_timings': step_times,
            'method': 'unified_extended',
            'recurrence_mode': self.get_recurrence_mode(),
            'multi_hop_algorithm': self.extended_config.multi_hop_algorithm if self.extended_config else 'none',
            'iteration_level': self.extended_config.iteration_level if self.extended_config else 0,
            'cache_hit': cache_hit,
            'nan_fitness_count': nan_fitness_count,
        }

        if sparse_hh is not None:
            custom_metrics['total_hh_connections'] = int(jnp.sum(sparse_hh.num_valid))
            custom_metrics['avg_hh_connections'] = float(jnp.mean(sparse_hh.num_valid))
            truncated = _last_discovery_stats.get('truncated_per_genome')
            if (not cache_hit and truncated is not None
                    and truncated.shape[0] == sparse_hh.num_valid.shape[0]):
                # Genomes whose h->h set the max_sparse_conns cap cut this generation, and the
                # largest set.
                custom_metrics['hh_truncated_genomes'] = int(np.count_nonzero(truncated))
                custom_metrics['hh_max_uncapped'] = int(
                    _last_discovery_stats['uncapped_per_genome'].max())

        # Compute substrate efficiency metrics
        active_counts = masks_A.sum(axis=1)  # Per-genome active position counts
        active_positions_mean = float(jnp.mean(active_counts))
        active_positions_min = int(jnp.min(active_counts))
        active_positions_max = int(jnp.max(active_counts))
        # Against the substrate *ceiling*, not the compacted evaluation space; see the
        # compaction block above.
        position_utilization = (
            active_positions_mean / ceiling_positions if ceiling_positions > 0 else 0.0
        )

        # Add hidden node counts to custom_metrics for benchmark reporting
        custom_metrics['avg_hidden_nodes'] = active_positions_mean
        custom_metrics['min_hidden_nodes'] = active_positions_min
        custom_metrics['max_hidden_nodes'] = active_positions_max
        custom_metrics['position_utilization'] = position_utilization
        # HMR's keys, under the same names and meanings.
        custom_metrics['variance_threshold'] = self.variance_threshold
        custom_metrics['num_levels'] = h_grid.num_levels

        # The evaluated population's species (state is the state before tell) and topology.
        num_species, species_sizes, species_fitness = species_summary(state.species, fitnesses)
        topology = topology_summary(
            W1, W2, masks_A, sparse_hh.num_valid if sparse_hh is not None else None)

        # Compute weight matrix density
        w1_total = W1.size
        w2_total = W2.size
        if compact_pad:
            # The virtual positions are zero columns of W1 and zero rows of W2; they are not
            # part of the substrate, so they stay out of the denominators.
            w1_total = W1.shape[0] * W1.shape[1] * (W1.shape[2] - compact_pad)
            w2_total = W2.shape[0] * (W2.shape[1] - compact_pad) * W2.shape[2]
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
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=pop_size,
            time_elapsed=total_time,
            **topology,
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

        node_rule = self.neuromod_config.node_rule if self.neuromod_config else 'paper'
        use_output_inversion = (self.neuromod_config.use_output_inversion if self.neuromod_config
                                else True)

        def modulated_node(receptor_densities_single, base_gains_single, neurotransmitter,
                           modulation_mode, modulation_strength_override):
            """A node under neuromodulation, as a function of its pre-activation: the same rule
            as the single-task path (true_neuromodulation_node_fn, neuromodulation.node_rule). The
            feed-forward and the recurrent evaluation below both apply it."""
            strength = (modulation_strength_override if modulation_strength_override is not None
                        else mod_strength)
            return true_neuromodulation_node_fn(act_fn, receptor_densities_single,
                                                base_gains_single, neurotransmitter, strength,
                                                neurotransmitter.shape[-1], modulation_mode,
                                                node_rule)

        def finish(hidden, W2_single, neurotransmitter, targets, fitness_mode):
            # Output inversion by NT[3] (complementary tasks like NAND, NOR), only with four or more
            # channels and use_output_inversion (0.1.0's multi-task path read nt[3] whatever the
            # count, and ignored the flag)
            outputs = invert_outputs(jax.nn.sigmoid(hidden @ W2_single), neurotransmitter,
                                     use_output_inversion, neurotransmitter.shape[-1], node_rule)
            return compute_fitness(outputs, targets, fitness_mode)

        if has_hh and activate_time > 0:
            def eval_single_network_true_neuromodulated_hh(
                W1_single, W2_single, receptor_densities_single, base_gains_single,
                neurotransmitter, inputs, targets,
                from_idx, to_idx, hh_w, hh_mask,
                fitness_mode='mse', modulation_mode='full', modulation_strength_override=None
            ):
                """One network with TRUE neuromodulation and H→H connections, on one task.

                W1, W2, receptors, base gains and the H→H connections are fixed; only the
                neurotransmitter vector varies per task. The hidden layer settles by the
                recurrent path's one rule (:func:`settle_hidden`), with the node function of
                the feed-forward evaluation, so without H→H connections both agree.
                """
                node = modulated_node(receptor_densities_single, base_gains_single,
                                      neurotransmitter, modulation_mode,
                                      modulation_strength_override)
                hidden = settle_hidden(inputs @ W1_single, from_idx, to_idx, hh_w, hh_mask,
                                       activate_time, node)
                return finish(hidden, W2_single, neurotransmitter, targets, fitness_mode)

            # Vmap over population (axis 0 for all per-genome arrays)
            eval_population = jax.vmap(
                eval_single_network_true_neuromodulated_hh,
                in_axes=(0, 0, 0, 0, None, None, None, 0, 0, 0, 0, None, None, None)
            )

        else:
            def eval_single_network_true_neuromodulated(
                W1_single, W2_single, receptor_densities_single, base_gains_single,
                neurotransmitter, inputs, targets, fitness_mode='mse',
                modulation_mode='full', modulation_strength_override=None
            ):
                """One network with TRUE neuromodulation (feedforward), on one task.

                W1, W2, receptors and base gains are fixed; only the neurotransmitter vector
                varies per task, so one network produces different behaviors.
                """
                node = modulated_node(receptor_densities_single, base_gains_single,
                                      neurotransmitter, modulation_mode,
                                      modulation_strength_override)
                hidden = node(inputs @ W1_single)
                return finish(hidden, W2_single, neurotransmitter, targets, fitness_mode)

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

        use_output_inversion = (self.neuromod_config.use_output_inversion if self.neuromod_config
                                else True)

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

            # Output inversion by NT[3], only with four or more channels and use_output_inversion
            outputs = invert_outputs(outputs, neurotransmitter, use_output_inversion,
                                     neurotransmitter.shape[-1])

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

            # Output inversion by NT[3], only with four or more channels and use_output_inversion
            outputs = invert_outputs(outputs, neurotransmitter, use_output_inversion,
                                     neurotransmitter.shape[-1])

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
        if self.population_chunk_size > 0:
            # Manual override.
            pop_chunk_size = self.population_chunk_size
        elif self.population_chunk_size == -1:
            # AUTO: chunk only when the query's forward carry would not fit the
            # shard budget. Every configuration whose full-population carry fits
            # keeps its measured behavior byte-identical; the gate exists for the
            # depth-7 class of failure, where the unchunked carry alone was
            # a recorded 6.56 GB. Per-genome query
            # results are independent, so chunking never changes values.
            # Tuple order is (nodes, conns, node_mask, conn_mask); the padded node
            # axis is index 0. (Index 1 is max_connections, which EQUALS 2*max_nodes
            # at the defaults -- a silent trap for anyone touching this.)
            cppn_max_nodes = cppns_transformed[0].shape[1]
            largest_level = max(
                int(h_grid.level_offsets_static[lv + 1] - h_grid.level_offsets_static[lv])
                for lv in range(num_levels)
            )
            pop_chunk_size = population_chunk_for_query(
                pop_size, largest_level, cppn_max_nodes, self._target_shard_gb)
        else:
            # 0 = explicit no-chunking.
            pop_chunk_size = pop_size

        # A CPPN emits several outputs when dynamic_functions.mode='cppn_output'
        # appends a per-node activation index. That requires the multi-output query
        # path, so the count is read here and threaded through.
        num_cppn_outputs = getattr(self, 'num_cppn_outputs', 1)

        # Pre-allocate weight matrix accumulators
        W1_accumulated = jnp.zeros((pop_size, num_inputs, total_positions), dtype=jnp.float32)
        W2_accumulated = jnp.zeros((pop_size, num_outputs, total_positions), dtype=jnp.float32)

        # Track level variances and reached masks for subdivision
        level_variances = []
        level_reached = []
        level_reached.append(jnp.ones((pop_size, 4), dtype=bool))  # Level 0 always reached

        # Cache for next-level weights (for variance lookahead)
        cached_next_input_weights = None
        # variance_rule legacy: level 0's own variance, set at level 0 and read where level 1's
        # reach is derived
        legacy_v0 = None

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
                prev_variance = (legacy_v0 if (level == 1 and legacy_v0 is not None)
                                 else level_variances[level - 1])
                # Canonical ES-HyperNEAT subdivides on two clauses:
                #   (level < initial_depth) or (variance > threshold)
                # A parent at level-1 is forced when level <= initial_depth.
                if level <= self.initial_depth:
                    prev_subdivided = prev_reached
                else:
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
                    num_cppn_outputs=num_cppn_outputs,
                )

            if level == 0 and self._legacy_level0_variance:
                # Channel 0 of the first input, the standard path's variance source.
                legacy_v0 = legacy_level0_variance(
                    level_input_weights[:, 0, :, 0] if num_cppn_outputs > 1
                    else level_input_weights[:, 0, :])

            # Output weights for this level (always query fresh)
            level_output_weights = query_level_positions(
                state, cppns_transformed, output_coords, level_positions,
                False, self._jitted_cppn_forward, pop_chunk_size,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
                num_cppn_outputs=num_cppn_outputs,
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
                    num_cppn_outputs=num_cppn_outputs,
                )

                # Compute variance from children
                if num_cppn_outputs > 1:
                    # Multi-output: (pop, num_sources, num_positions, num_outputs).
                    # Channel 0 is always the weight; the rest are secondary
                    # channels such as the per-node activation index.
                    variance_weights = cached_next_input_weights[:, 0, :, 0]
                else:
                    # Single output: (pop, num_sources, num_positions)
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
            # level_input_weights shape: (pop, num_sources, num_positions), or
            # (pop, num_sources, num_positions, num_outputs) when the CPPN is
            # multi-output -- notably dynamic_functions.mode='cppn_output', which
            # appends a per-node activation index. Only channel 0 is a weight here;
            # the activation channel is consumed by the per-node function selection
            # later in the pipeline, not by matrix construction.
            if level_input_weights.ndim == 4:
                level_input_weights = level_input_weights[:, :, :, 0]
                level_output_weights = level_output_weights[:, :, :, 0]

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
            # Donated on GPU: in-place slice write, no transient second W1/W2
            # (the level-7 786 MB doubling operand).
            W1_accumulated = _accumulate_level_slice(
                W1_accumulated, W1_level, level_start, level_end)
            W2_accumulated = _accumulate_level_slice(
                W2_accumulated, W2_level, level_start, level_end)

            # Explicit memory cleanup
            del level_input_weights, level_output_weights, W1_raw, W2_raw
            del W1_level, W2_level

            if self._streaming_verbose:
                level_time = time.perf_counter() - t_level
                print(f"  Level {level}: {num_level_positions} positions, {level_time*1000:.1f}ms")

        # Build full masks_A from level_reached
        masks_A = jnp.concatenate(level_reached, axis=1)
        # Observable for substrate depth, mirroring the offload path. Pure read.
        self._last_reached_per_level = [
            int(np.count_nonzero(np.asarray(mask))) for mask in level_reached
        ]

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
            orthogonality_bonus: Refused above 0, as specialization_bonus (see Raises).
            specialization_bonus: Refused above 0.

        Returns:
            Tuple of (new_state, metrics) with multi-task metrics

        Raises:
            ValueError: If neuromodulation is disabled or mode not true_neuromodulation; a bonus above 0
                (the two arguments, or orthogonality_bonus_weight on a multi-task configuration built by
                hand): each adds one value to every genome of a generation, from the NT vectors or from the
                population's mean variance, so it never changes a rank within the generation, only the reported
                fitness (the configuration keys are refused as well).
        """
        import time as time_module

        total_start = time_module.perf_counter()

        configured = getattr(getattr(self, 'multitask_config', None), 'orthogonality_bonus_weight', 0.0) or 0.0
        for name, value in (('orthogonality_bonus', orthogonality_bonus),
                            ('specialization_bonus', specialization_bonus),
                            ('multitask_config.orthogonality_bonus_weight', configured)):
            if value:
                raise ValueError(f"{name}={value} would add one value to every genome of a generation, which "
                                 f"never changes a rank within it")

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

        # A mode set on the configuration object by hand is checked before the generation is built
        mode = getattr(self.multitask_config, 'fitness_mode', 'mse') if self.multitask_config else 'mse'
        if mode not in MULTITASK_FITNESS_MODES:
            raise ValueError(f"multitask fitness_mode must be one of {MULTITASK_FITNESS_MODES}, got {mode!r}")

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
        # Its metrics carry this population's topology, which no task changes.
        _, structure = self.run_generation_verbose(state, first_problem, skip_metrics=True)
        # That generation's champion is the first task's, not this generation's
        self._last_eval = None

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

        # Also checked by create_config; repeated here because a runner may set two_module_mode on
        # the MultiTaskConfig after create_config
        if cfg_two_module_mode != 'none' and has_hh:
            raise ValueError(
                "multitask.two_module_mode evaluates feed-forward only and would ignore the "
                "hidden-to-hidden connections this configuration discovers"
            )
        if (cfg_two_module_mode != 'none' and self.neuromod_config is not None
                and self.neuromod_config.node_rule == 'hmr_full'):
            raise ValueError("neuromodulation.node_rule 'hmr_full' is not applied by "
                             "multitask.two_module_mode, whose modules compute the paper's rule")

        # Create eval functions for unique activation functions (cache to avoid redundant JIT)
        unique_activations = set(per_task_act_names)
        eval_fn_cache = {}
        eval_fn_cache_two_module = {}  # For two-module mode

        # Get all_positions for two-module evaluation (if needed)
        all_positions = None
        if cfg_two_module_mode != 'none':
            # The grid itself: 0.1.0 read a cached grid that nothing set, so two-module mode always
            # fell back to one module (with a warning only under verbose). The multi-task path keeps
            # every position (no compaction), so the grid's positions line up with W1.
            all_positions = get_hierarchical_grid(self.max_depth).all_positions

        for act_name in unique_activations:
            # An unknown name (a MultiTaskConfig built by hand; create_config checks the keys) ran
            # tanh in 0.1.0
            if act_name not in ACTIVATION_FUNCTIONS:
                raise ValueError(f"unknown multi-task activation {act_name!r}; the names are "
                                 f"{sorted(ACTIVATION_FUNCTIONS)}")
            act_fn = ACTIVATION_FUNCTIONS[act_name]
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
        # NaN (an overflowing settling, a NaN target) becomes -inf before selection sees it.
        aggregated_fitness, nan_fitness_count = sanitize_fitness(aggregated_fitness)
        new_state = self._compiled_tell(state, aggregated_fitness)

        # The generation counter was advanced by the run_generation_verbose call above; a
        # second increment here made the H→H cache refresh and the random modes' per-
        # generation keys run twice as fast under multi-task evaluation.

        # Build multi-task metrics
        total_time = time_module.perf_counter() - total_start

        best_fitness, mean_fitness, min_fitness, std_fitness = fitness_statistics(
            aggregated_fitness, nan_fitness_count)

        # Find best generalist (single individual with highest aggregated fitness)
        best_gen_idx = int(jnp.argmax(aggregated_fitness))
        best_generalist_per_task = {
            f'task_{i}': float(per_task_fitnesses[i][best_gen_idx])
            for i in range(num_tasks)
        }

        custom_metrics = {
            'method': 'multitask',
            'num_tasks': num_tasks,
            'nan_fitness_count': nan_fitness_count,
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
        num_species, species_sizes, species_fitness = species_summary(state.species,
                                                                      aggregated_fitness)
        metrics = AlgorithmMetrics(
            generation=self._current_generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=best_fitness,
            std_fitness=std_fitness,
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=len(aggregated_fitness) * num_tasks,  # pop_size * num_tasks
            time_elapsed=total_time,
            custom_metrics=custom_metrics,
            hidden_nodes_mean=structure.hidden_nodes_mean,
            hidden_nodes_std=structure.hidden_nodes_std,
            connections_mean=structure.connections_mean,
            connections_std=structure.connections_std,
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
        if activation_name is not None and activation_name not in ACTIVATION_FUNCTIONS:
            # 0.1.0 ran tanh for an unknown name
            raise ValueError(f"unknown activation {activation_name!r}; the names are "
                             f"{sorted(ACTIVATION_FUNCTIONS)}")
        act_fn = ACTIVATION_FUNCTIONS[activation_name] if activation_name else jnp.tanh

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
    ) -> Dict[str, Any]:
        """Evaluate the last generation's champion on arbitrary input/target data.

        The champion is scored with the forward its generation evaluated it with: the same
        per-genome function, arrays and fitness mode, rescaling included (see
        ``_build_forward``). On the training data its fitness is therefore the fitness it was
        selected on (bit for bit on CPU), and a held-out score describes the network that was
        evolved, recurrent connections, per-node aggregation, global and random activation
        modes and neuromodulation included. In 0.1.0 this re-derived the champion as a
        dense tanh network with sum aggregation, which held for feed-forward per-node
        activation and for nothing else.

        The champion is recorded by :meth:`run_generation_verbose`, the generation of the
        standard and streaming execution modes on one task. Depth offload, multi-task
        generations and the multi-GPU strategies record none, and this raises after them.

        Args:
            state: The state the last generation returned. Its first genome is that
                generation's champion: TensorNEAT places the best genome, unmutated, first in
                the next population, and this is checked.
            inputs: Input data array of shape (n_samples, n_features)
            targets: Target data array of shape (n_samples, n_outputs)

        Returns:
            Dict with 'mse', 'fitness', 'n_samples', the error distribution, substrate
            statistics, 'champion_index', 'generation' and 'training_fitness', and, when a
            dynamic-functions mode is on, 'activation_counts' and
            'activation_counts_per_level' over the champion's hidden nodes.

        Raises:
            RuntimeError: The last generation recorded no champion.
            ValueError: ``state`` is not the one the last generation returned.
        """
        record = self._last_eval
        if record is None:
            raise RuntimeError(
                "evaluate_on_data scores the champion of the last generation this model ran "
                "with run_generation_verbose (standard or streaming execution, one task); run "
                "a generation first. Depth offload, multi-task generations and the multi-GPU "
                "strategies record no champion"
            )
        population = self._compiled_ask(state)
        champion_nodes, champion_conns = record['champion_genome']
        if not (np.array_equal(np.asarray(population[0][0]), champion_nodes, equal_nan=True)
                and np.array_equal(np.asarray(population[1][0]), champion_conns,
                                   equal_nan=True)):
            raise ValueError(
                "evaluate_on_data scores the champion of the last generation this model ran, "
                "and the state passed is not the one that generation returned"
            )

        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        targets = jnp.asarray(targets, dtype=jnp.float32)
        forward_one = record['forward_one']
        fitness_mode = self.fitness_mode

        def score(*args):
            outputs = forward_one(inputs, *args)
            return outputs, _eval_single_fitness(outputs, targets, fitness_mode)

        # vmapped over the champion's two copies, as the generation vmapped the population
        outputs, fitness = jax.vmap(score)(*record['champion_args'])
        if self._fitness_rescale_variance is not None:
            fitness = jnp.maximum(
                0.0, 1.0 - (1.0 - fitness) / self._fitness_rescale_variance
            )
        outputs = outputs[0]
        errors = (outputs - targets) ** 2
        sample_errors = jnp.mean(errors, axis=1)

        h_grid = get_hierarchical_grid(self.max_depth)
        mask_full = record['mask_full']
        W1_best, W2_best = record['W1_full'], record['W2_full']
        W1_nonzero = int(jnp.sum(jnp.abs(W1_best) > 0))
        W2_nonzero = int(jnp.sum(jnp.abs(W2_best) > 0))
        W1_flat, W2_flat = W1_best.flatten(), W2_best.flatten()
        W1_active = W1_flat[jnp.abs(W1_flat) > 0]
        W2_active = W2_flat[jnp.abs(W2_flat) > 0]
        result: Dict[str, Any] = {
            'mse': float(jnp.mean(errors)),
            'fitness': float(fitness[0]),
            'n_samples': int(inputs.shape[0]),
            'mse_std': float(jnp.std(sample_errors)),
            'mse_min': float(jnp.min(sample_errors)),
            'mse_max': float(jnp.max(sample_errors)),
            'mse_median': float(jnp.median(sample_errors)),
            'active_hidden_nodes': int(np.sum(mask_full)),
            'total_positions': h_grid.total_positions,
            'total_connections': W1_nonzero + W2_nonzero,
            'input_connections': W1_nonzero,
            'output_connections': W2_nonzero,
            'w1_mean': float(jnp.mean(W1_active)) if W1_active.size else 0.0,
            'w1_std': float(jnp.std(W1_active)) if W1_active.size else 0.0,
            'w2_mean': float(jnp.mean(W2_active)) if W2_active.size else 0.0,
            'w2_std': float(jnp.std(W2_active)) if W2_active.size else 0.0,
            '_mask': mask_full,
            '_W1': W1_best,
            '_W2': W2_best,
            '_h_grid': h_grid,
            'champion_index': record['champion_index'],
            'generation': record['generation'],
            'training_fitness': record['champion_fitness'],
            'dynamic_functions_mode': self.dynamic_functions_mode,
        }
        if self.dynamic_functions_mode != 'disabled':
            result.update(self._champion_activation_counts(record, h_grid))
        return result

    def _champion_activation_counts(self, record: Dict[str, Any], h_grid: Any) -> Dict[str, Any]:
        """How many of the champion's hidden nodes use each activation, overall and per level.

        Hidden nodes only: positions the mask drops compute nothing. In 0.1.0 the counts
        included them, decoded at a raw output of 0, which inflated the middle function.
        """
        mask = np.asarray(record['mask_eval'], dtype=bool)
        positions = (record['eval_positions'] if record['eval_positions'] is not None
                     else np.arange(mask.shape[0]))
        if record['act_indices'] is not None:
            names = np.array([ACTIVATION_LIST[i] for i in np.asarray(record['act_indices'])])
        else:
            one = (self.df_hidden_activation if self.dynamic_functions_mode == 'global'
                   else 'tanh')
            names = np.full(mask.shape[0], one)
        live_names, live_positions = names[mask], np.asarray(positions)[mask]

        def counts(selected):
            values, numbers = np.unique(selected, return_counts=True)
            return {str(v): int(n) for v, n in zip(values, numbers)}

        offsets = h_grid.level_offsets_static
        per_level = {}
        for level in range(len(offsets) - 1):
            in_level = (live_positions >= offsets[level]) & (live_positions < offsets[level + 1])
            per_level[f'level_{level}'] = {
                'num_positions': int(offsets[level + 1] - offsets[level]),
                'num_hidden_nodes': int(np.sum(in_level)),
                'counts': counts(live_names[in_level]),
            }
        overall = counts(live_names)
        return {
            'activation_counts': overall,
            'activation_counts_per_level': per_level,
            'num_activations_used': len(overall),
        }

    # ========================================================================
    # Multi-GPU run_until_threshold() Override
    # ========================================================================

    #: The strategies run_until_threshold runs on more than one device (each on one when only one is visible).
    MULTI_DEVICE_ROUTES = (MultiGPUStrategy.FULL_PIPELINE_PARALLEL, MultiGPUStrategy.EVAL_ONLY_PARALLEL,
                           MultiGPUStrategy.POPULATION_PARALLEL_PROCESS)

    def get_num_devices(self) -> int:
        """Devices the strategy's route in run_until_threshold uses.

        Every visible device for FULL_PIPELINE_PARALLEL, EVAL_ONLY_PARALLEL and POPULATION_PARALLEL_PROCESS; one for
        SINGLE_GPU, SINGLE_GPU_PYTHON_LOOP and STREAMING. run_until_threshold runs the three multi-device routes on one
        device when only one is visible, and FULL_PIPELINE_PARALLEL also on an exhaustive or small dataset; its result
        records the devices a run used. The base class counted devices by the name.

        Returns:
            The number of devices.
        """
        return len(jax.devices()) if self.strategy in self.MULTI_DEVICE_ROUTES else 1

    @staticmethod
    def _record_route(result: Dict[str, Any], route: str, devices: int) -> Dict[str, Any]:
        """Records the route run_until_threshold took, the devices it used and the execution mode it ran.

        A strategy name alone does not say that FULL_PIPELINE_PARALLEL runs one device on an exhaustive dataset such
        as XOR, so a run could carry a multi-device label it did not run. The one-device route reports its own mode (depth offload replaces the generation even under the
        STREAMING route); the multi-device routes run their own on-device pipeline, 'standard'.

        Args:
            result: The route's result dictionary.
            route: SINGLE_GPU, STREAMING, FULL_PIPELINE_PARALLEL, EVAL_ONLY_PARALLEL or POPULATION_PARALLEL_PROCESS.
            devices: The devices the route used.

        Returns:
            The same dictionary, with 'route', 'devices' and 'execution_mode'.
        """
        result['route'] = route
        result['devices'] = devices
        result.setdefault('execution_mode', 'standard')
        return result

    def _refuse_ignored_execution_mode(self, route: str) -> None:
        """A multi-device route runs its own on-device pipeline: an explicitly configured streaming or depth
        offload would be ignored, so it is refused. AUTO leaves the choice to the engine, and the route's record says
        'standard'.

        Raises:
            ValueError: execution.mode is streaming or depth_offload.
        """
        if self.execution_config.mode in (ExecutionMode.STREAMING, ExecutionMode.DEPTH_OFFLOAD):
            raise ValueError(
                f"{route} runs its own on-device pipeline and would ignore execution.mode "
                f"{self.execution_config.mode.value!r}; use SINGLE_GPU or STREAMING for a memory mode")

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
           Keeping discovery on one device is what lets the h→h cache survive
           Recommended for: Recurrent (h→h) modes
           Memory: ~balanced across GPUs

        5. POPULATION_PARALLEL_PROCESS:
           ProcessPoolExecutor-based parallel processing
           Each GPU runs in separate Python process with isolated JAX runtime
           NOT RECOMMENDED: the least exercised path here, see MultiGPUStrategy

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
            Dict with 'generations', 'best_fitness', 'state', 'history' (optional), 'route' (the route taken:
            SINGLE_GPU, STREAMING, FULL_PIPELINE_PARALLEL, EVAL_ONLY_PARALLEL or POPULATION_PARALLEL_PROCESS),
            'devices' (the devices it used) and 'execution_mode' (the mode it ran: standard, streaming or
            depth_offload)

        Raises:
            ValueError: a multi-device route with an explicit execution.mode streaming or depth_offload, which
                it would ignore; a strategy without a route on two or more devices (assigned to
                ``self.strategy`` by hand); POPULATION_PARALLEL_PROCESS with features its workers drop.
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

        # Only run_generation_verbose records a champion for evaluate_on_data; the
        # multi-GPU loops below record none, and an earlier record would be stale
        self._last_eval = None

        # In 0.1.0 a strategy with no route below ran the one-device loop under its own name, on
        # any number of devices. The constructor and the configuration key map every name through
        # STRATEGY_NAME_MAP and raise for a name without a route, so this guard fires only for a
        # strategy assigned to self.strategy by hand.
        routed = (MultiGPUStrategy.FULL_PIPELINE_PARALLEL, MultiGPUStrategy.EVAL_ONLY_PARALLEL,
                  MultiGPUStrategy.POPULATION_PARALLEL_PROCESS, MultiGPUStrategy.STREAMING)
        one_device = (MultiGPUStrategy.SINGLE_GPU, MultiGPUStrategy.BASELINE,
                      MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP)
        if num_devices >= 2 and self.strategy not in routed + one_device:
            raise ValueError(
                f"strategy {self.strategy.name} has no route in run_until_threshold, which runs "
                f"SINGLE_GPU, STREAMING, FULL_PIPELINE_PARALLEL, EVAL_ONLY_PARALLEL and "
                f"POPULATION_PARALLEL_PROCESS; on {num_devices} devices it would run on one. A name "
                f"given to the constructor or to emr_hyperneat.multi_gpu_strategy maps to one of these "
                f"(STRATEGY_NAME_MAP), so this strategy was assigned by hand.")

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
                    self._refuse_ignored_execution_mode('FULL_PIPELINE_PARALLEL')
                    return self._record_route(self._run_until_threshold_data_parallel_full(
                        state, problem, target_fitness, max_generations, collect_history
                    ), 'FULL_PIPELINE_PARALLEL', num_devices)
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
                    return self._record_route(self._run_until_threshold_single_gpu(
                        state, problem, target_fitness, max_generations, collect_history
                    ), 'SINGLE_GPU', 1)
            elif self.strategy == MultiGPUStrategy.POPULATION_PARALLEL_PROCESS:
                # The worker processes score tanh nodes and sum aggregation (multi_gpu_worker.py); in
                # 0.1.0 they did so silently whatever the configuration asked for.
                dropped = self._features_a_tanh_sum_path_drops() + self._features_offload_drops()
                if dropped:
                    raise ValueError(
                        "POPULATION_PARALLEL_PROCESS evaluates tanh hidden nodes with sum aggregation "
                        "and 1-MSE, so it would ignore " + ", ".join(dropped) + ". Use SINGLE_GPU.")
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

                self._refuse_ignored_execution_mode('POPULATION_PARALLEL_PROCESS')
                return self._record_route(self._run_until_threshold_device_parallel_process(
                    state, problem, target_fitness, max_generations, collect_history
                ), 'POPULATION_PARALLEL_PROCESS', num_devices)

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

                self._refuse_ignored_execution_mode('EVAL_ONLY_PARALLEL')
                return self._record_route(self._run_until_threshold_pmap(
                    state, problem, target_fitness, max_generations, collect_history
                ), 'EVAL_ONLY_PARALLEL', num_devices)

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
                    return self._record_route(self._run_until_threshold_single_gpu(
                        state, problem, target_fitness, max_generations, collect_history
                    ), 'STREAMING', 1)
                finally:
                    self.enable_streaming = original_streaming

        # Fallback for single GPU or when multi-GPU strategies run on single device
        # Note: Legacy aliases (MULTI_GPU, DATA_PARALLEL, etc.) are mapped to the
        # primary strategies by the constructor and the configuration key alike
        # (STRATEGY_NAME_MAP), so only the primary 5 are checked here
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
                return self._record_route(self._run_until_threshold_single_gpu(
                    state, problem, target_fitness, max_generations, collect_history
                ), 'STREAMING', 1)
            finally:
                self.enable_streaming = original_streaming

        elif self.strategy == MultiGPUStrategy.SINGLE_GPU:
            print(f"[Strategy] SINGLE_GPU: Using 1 GPU")
            print(f"  - All computation on primary GPU")
        elif self.strategy not in one_device:
            print(f"[Strategy] {self.strategy.name} requested on {num_devices} device: "
                  f"running the one-device loop")

        # For single GPU, CPU, or fallback: use run_generation_verbose() loop
        return self._record_route(self._run_until_threshold_single_gpu(
            state, problem, target_fitness, max_generations, collect_history
        ), 'SINGLE_GPU', 1)

    def _run_until_threshold_single_gpu(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Single GPU implementation using the existing per-generation code.

        This preserves all unified extended features (h→h discovery, multi-hop,
        caching) by using the existing per-generation implementation.

        Execution mode is honoured here, the same way `run_generation` honours it:
        `streaming` through the `enable_streaming` flag that `create_config` resolves,
        and `depth_offload` through the explicit dispatch below. Without that dispatch
        a config asking for `depth_offload` silently ran the standard on-device path,
        which is exactly the out-of-memory failure the mode exists to avoid.

        The dispatch lives here rather than inside `run_generation_verbose` on purpose:
        the offload path does not populate `_cached_W1` / `_cached_W2`, and
        `run_generation_multitask` and the neuromodulation runner read those straight
        after calling that method.

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
            print(f"[EMR-HyperNEAT] Single GPU mode: max_gens={max_generations}, target={target_fitness}")

        # Resolved once: max_depth and the config do not change inside the loop.
        _use_offload = self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD
        if _use_offload:
            # Imported here rather than at module scope: the strategy modules import
            # from this one, so a top-level import would form a cycle.
            from emr_hyperneat.execution.offload import (  # noqa: E501
                run_generation_offload,
            )
            if self.enable_streaming:
                # Reachable through the MultiGPUStrategy.STREAMING arms above, which
                # force enable_streaming before calling this helper. The two are
                # mutually exclusive memory strategies, and offload wins here for the
                # same reason it wins in `run_generation`: it replaces the whole
                # generation rather than one step inside it. Say so instead of letting
                # the strategy banner claim a path that will not run.
                import warnings
                warnings.warn(
                    "Streaming was requested (MultiGPUStrategy.STREAMING) but the "
                    "execution mode resolves to depth_offload; offload takes "
                    "precedence and streaming will not run. Set one memory strategy, "
                    "not both.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.verbose:
                print("[EMR-HyperNEAT] Execution mode: depth_offload")

        while best_so_far < target_fitness and generation < max_generations:
            if _use_offload:
                state, metrics = run_generation_offload(self, state, problem)
            else:
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
            # The mode this loop ran (offload replaces the generation, even under the STREAMING route)
            'execution_mode': ('depth_offload' if _use_offload
                               else 'streaming' if self.enable_streaming else 'standard'),
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
                    level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid, legacy_level0_variance=self._legacy_level0_variance)
                    masks_A = compute_subdivision_masks_batch(
                        level_variances, self.variance_threshold, h_grid, return_all_masks=False,
                        initial_depth=self.initial_depth,
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
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid, legacy_level0_variance=self._legacy_level0_variance)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False,
                initial_depth=self.initial_depth,
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
                    # The worker builds a TensorNEAT CPPN from this and takes every setting it
                    # computes with from PersistentWorkerConfig. 0.1.0 also passed activate_time,
                    # extended_config and neat.species_size here, which nothing read; EMR refuses
                    # keys it does not read.
                    'emr_hyperneat': {
                        'initial_depth': self.initial_depth,
                        'max_depth': self.max_depth,
                        'variance_threshold': self.variance_threshold,
                        'band_threshold': self.band_threshold,
                        'max_weight': self.max_weight,
                    },
                    'substrate': {
                        'input_coords': input_coords_np.tolist(),
                        'output_coords': output_coords_np.tolist(),
                    },
                    'neat': {
                        'pop_size': pop_size,
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
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid, legacy_level0_variance=self._legacy_level0_variance)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False,
                initial_depth=self.initial_depth,
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
                        # Population total, the basis the single-device path reports
                        # (custom_metrics['total_hh_connections']); genome 0 alone
                        # understated it by a factor of the population size.
                        total_hh_count = int(jnp.sum(hh_valid))
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
                    total_hh_count = int(jnp.sum(sparse_hh.num_valid))  # Population total

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
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid, legacy_level0_variance=self._legacy_level0_variance)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False,
                initial_depth=self.initial_depth,
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
           - H→H discovery (direct connections; iteration level 1 only, no multi-hop expansion)
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

        self._refuse_pmap_pipeline_gaps('FULL_PIPELINE_PARALLEL')
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
                self.initial_depth,          # STATIC
                self._legacy_level0_variance,  # STATIC (24)
                *self._hh_class_flags(),  # STATIC (25-27): allow_backward, allow_lateral, allow_self_loops
            )

            # Aggregate: average fitness across data shards
            # Each GPU computed fitness on different samples, so average them
            fitnesses = jnp.mean(partial_fitnesses, axis=0)  # (pop_size,)

            # Aggregate h→h counts (same population on each GPU, so take first)
            # Each GPU computes h→h for the full population, they should be identical
            gen_hh_count = int(hh_counts_sharded[0])  # Take from first GPU
            self._last_pmap_hh_count = gen_hh_count  # for records and tests
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
            # The pmap pipelines discover at level 1 only (levels above 1 are refused on entry).
            effective_iteration_level=1 if allow_hh else 0,
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

        self._refuse_pmap_pipeline_gaps('POPULATION_PARALLEL_SEQUENTIAL')
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
                            self.initial_depth,
                            *self._hh_class_flags(),  # STATIC (24-26): allow_backward, allow_lateral, allow_self_loops
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
                            self.initial_depth,
                            self._legacy_level0_variance,  # STATIC (24)
                            *self._hh_class_flags(),  # STATIC (25-27): allow_backward, allow_lateral, allow_self_loops
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
                self._last_pmap_hh_count = gen_hh_count  # for records and tests

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
                        self.initial_depth,
                        *self._hh_class_flags(),  # STATIC (24-26): allow_backward, allow_lateral, allow_self_loops
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
                        self.initial_depth,
                        self._legacy_level0_variance,  # STATIC (24)
                        *self._hh_class_flags(),  # STATIC (25-27): allow_backward, allow_lateral, allow_self_loops
                    )

                # Gather: concatenate fitnesses and remove padding
                fitnesses = fitnesses_sharded.reshape(-1)[:pop_size]
                gen_hh_count = int(jnp.sum(hh_counts_sharded))
                self._last_pmap_hh_count = gen_hh_count  # for records and tests

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
            # The pmap pipelines discover at level 1 only (levels above 1 are refused on entry).
            effective_iteration_level=1 if allow_hh else 0,
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



def _hh_class_mask_rows(y_all: jnp.ndarray, source_indices: jnp.ndarray, target_indices: jnp.ndarray,
                        allow_backward: bool, allow_lateral: bool, allow_self_loops: bool) -> jnp.ndarray:
    """Connection-class mask of the pmap pipelines' inline h->h discovery.

    The rule of ``_get_constraint_mask_fn`` applied to ``source_indices`` rows against
    ``target_indices`` columns of the hierarchical grid: without ``allow_backward`` a connection
    needs y_source <= y_target, without ``allow_lateral`` y_source != y_target; a connection from a
    position to itself is allowed only with ``allow_self_loops``. The flags are Python booleans
    (static under jit and pmap). In 0.1.0 these pipelines applied no class at all and removed
    self connections unconditionally, so every preset ran as full recurrence without self loops.
    """
    y_src = y_all[source_indices]
    y_tgt = y_all[target_indices]
    valid = jnp.ones((source_indices.shape[0], target_indices.shape[0]), dtype=bool)
    if not allow_backward:
        valid = valid & (y_src[:, None] <= y_tgt[None, :])
    if not allow_lateral:
        valid = valid & (y_src[:, None] != y_tgt[None, :])
    same = source_indices[:, None] == target_indices[None, :]
    return (valid | same) if allow_self_loops else (valid & ~same)


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
    initial_depth: int = 0,
    legacy_level0_variance: bool = False,
    allow_backward: bool = True,
    allow_lateral: bool = True,
    allow_self_loops: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full pipeline with h→h discovery - runs on single GPU inside pmap.

    This function replicates the parent class's _full_pipeline_single_gpu logic
    but adds Phase 2 h→h discovery of direct connections (iteration level 1; no multi-hop
    expansion, which is why levels above 1 are refused on entry to this pipeline).

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
        legacy_level0_variance=legacy_level0_variance,
    )

    # STEP 3: Compute subdivision masks
    masks_A, _, _ = compute_subdivision_masks_batch_jit(
        level_variances,
        variance_threshold=variance_threshold,
        initial_depth=initial_depth,
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
            _all_idx = jnp.arange(total_positions)
            class_mask = _hh_class_mask_rows(all_positions[:, 1], _all_idx, _all_idx,
                                             allow_backward, allow_lateral, allow_self_loops)

            def get_hh_for_genome(cppn_tuple, mask):
                # Query all→all
                hh_matrix = jax.vmap(
                    lambda src_idx: get_hh_weights(cppn_tuple, src_idx)
                )(active_indices)

                # Apply thresholds
                hh_matrix = jnp.tanh(hh_matrix) * max_weight
                valid = (jnp.abs(hh_matrix) > band_threshold) & mask[None, :] & mask[:, None]

                # Connection class (backward, lateral, self loops) and no self connection unless allowed
                valid = valid & class_mask

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

                    # Connection class (backward, lateral, self loops) for this chunk's source rows

                    target_indices = jnp.arange(total_positions)

                    valid_chunk = valid_chunk & _hh_class_mask_rows(all_positions[:, 1], chunk_source_indices, target_indices,

                                                                    allow_backward, allow_lateral, allow_self_loops)

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
    initial_depth: int = 0,
    legacy_level0_variance: bool = False,
    allow_backward: bool = True,
    allow_lateral: bool = True,
    allow_self_loops: bool = False,
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
        legacy_level0_variance=legacy_level0_variance,
    )

    # STEP 3: Compute subdivision masks
    masks_batch, _, _ = compute_subdivision_masks_batch_jit(
        level_variances,
        variance_threshold=variance_threshold,
        initial_depth=initial_depth,
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

            # Connection class (backward, lateral, self loops) for this chunk's source rows

            target_indices = jnp.arange(total_positions)

            valid_chunk = valid_chunk & _hh_class_mask_rows(all_positions[:, 1], chunk_source_indices, target_indices,

                                                            allow_backward, allow_lateral, allow_self_loops)

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
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26),
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
    initial_depth,        # STATIC (23) - forces subdivision through this level
    allow_backward=True,  # STATIC (24) - connection class
    allow_lateral=True,   # STATIC (25)
    allow_self_loops=False,  # STATIC (26)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequential population-parallel: process genomes one at a time inside pmap.

    The key insight: Python for-loops inside a pmap-traced function are NOT
    traced by XLA. Each iteration executes and deallocates before the next,
    avoiding the memory explosion from fori_loop unrolling.

    Ideal scaling for 2 devices is ~2x; the actual figure is not measured in this
    repository.

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
            geometry_seeding_enabled, initial_depth,
            allow_backward=allow_backward, allow_lateral=allow_lateral, allow_self_loops=allow_self_loops,
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
    in_axes=(None, None, None, None, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27),
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
    initial_depth,        # STATIC (23) - forces subdivision through this level
    legacy_level0_variance,  # STATIC (24) - variance_rule: legacy level-0 rule
    allow_backward=True,  # STATIC (25) - connection class
    allow_lateral=True,   # STATIC (26)
    allow_self_loops=False,  # STATIC (27)
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
        geometry_seeding_enabled, initial_depth, legacy_level0_variance,
        allow_backward, allow_lateral, allow_self_loops,
    )


# POPULATION_PARALLEL: Split population across GPUs, replicate full dataset
# in_axes: (0=sharded) for cppns
#          (None=replicated) for everything else
# NOTE: parent_indices (index 14) contains JAX arrays so NOT static
@functools.partial(
    jax.pmap,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27),
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
    initial_depth,        # STATIC (23) - forces subdivision through this level
    legacy_level0_variance,  # STATIC (24) - variance_rule: legacy level-0 rule
    allow_backward=True,  # STATIC (25) - connection class
    allow_lateral=True,   # STATIC (26)
    allow_self_loops=False,  # STATIC (27)
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
        geometry_seeding_enabled, initial_depth, legacy_level0_variance,
        allow_backward, allow_lateral, allow_self_loops,
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
