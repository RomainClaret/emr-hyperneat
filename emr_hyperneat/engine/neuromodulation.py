"""Neuromodulation: its configuration and presets, the neurotransmitter presets and receptor derivation,
HMR's mode names and their translation, the CPPN channels that drive it, the modulatory options and
masks, and the neuromodulated node rules."""
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.engine.settling import _node_rule


#: The modulatory-neuron options of mode modulatory_neurons and their values. Each changes one thing
#: the two sources differ in: which neurons (selection), how many (count rule), the per-case signal,
#: the per-node weight on it, which nodes it acts on (scope); then where a CPPN output is read,
#: whether the CPPN carries both modulatory outputs whatever reads them, and the matched-gain
#: controls.
MODULATORY_OPTIONS: Dict[str, Tuple[str, ...]] = {
    'modulatory_selection': ('variance', 'cppn'),
    'modulatory_count_rule': ('floor', 'ceil', 'fixed'),
    'modulatory_signal': ('mean_decay', 'sum_factor'),
    'modulatory_node_weights': ('variance', 'none'),
    'modulatory_scope': ('all', 'standard'),
    'modulatory_cppn_read': ('input_mean', 'self'),
    'modulatory_cppn_outputs': ('read', 'always'),
    'modulatory_matched_gain': ('none', 'mean', 'shift'),
}
#: modulatory_source as a bundle of the first five options: an option left unset takes its source's
#: value, so either source alone computes what it computed before the options.
MODULATORY_BUNDLES: Dict[str, Dict[str, str]] = {
    'weights': {
        'modulatory_selection': 'variance',
        'modulatory_count_rule': 'floor',
        'modulatory_signal': 'mean_decay',
        'modulatory_node_weights': 'variance',
        'modulatory_scope': 'all',
    },
    'cppn': {
        'modulatory_selection': 'cppn',
        'modulatory_count_rule': 'ceil',
        'modulatory_signal': 'sum_factor',
        'modulatory_node_weights': 'none',
        'modulatory_scope': 'standard',
    },
}


def resolve_modulatory_options(block: Dict[str, Any]) -> Dict[str, Any]:
    """The modulatory options of a neuromodulation block: each key the block sets, else its default
    (the bundle of the block's modulatory_source for the first five, MODULATORY_OPTIONS' first value
    for the others). Values are not checked here; the neuromodulation parse refuses unknown ones.

    Args:
        block: The neuromodulation block (HMR's names already translated).

    Returns:
        Option name -> value, for every key of MODULATORY_OPTIONS.
    """
    bundle = MODULATORY_BUNDLES.get(
        block.get('modulatory_source', 'weights'), MODULATORY_BUNDLES['weights']
    )
    return {
        key: block.get(key, bundle.get(key, values[0]))
        for key, values in MODULATORY_OPTIONS.items()
    }


def modulatory_cppn_reads(options: Dict[str, Any]) -> Tuple[str, ...]:
    """The modulatory CPPN outputs resolved options read: 'node_type' for CPPN selection,
    'mod_factor' for the summed signal."""
    return tuple(
        name
        for name, read in (
            ('node_type', options['modulatory_selection'] == 'cppn'),
            ('mod_factor', options['modulatory_signal'] == 'sum_factor'),
        )
        if read
    )


# Neuromodulation Configuration
# Modular neuromodulation system with four progressive levels; a run evaluates one.

@dataclass
class NeuromodulationConfig:
    """Configuration for neuromodulation features.

    Four levels; a run evaluates one, chosen by `mode` or a preset (a preset that combines
    static and context gating runs as context gating, and other combinations raise):
    - Level 1: Static Gating - one gate per hidden node, derived from its own input weights, or per
      connection from a dedicated CPPN output (gate_source 'cppn')
    - Level 2: Context Gating - Task context modulates gates dynamically
    - Level 3: Modulatory Neurons - Two neuron types (standard/modulatory)
    - Level 4: TRUE Neuromodulation - NT vectors + receptor densities

    Attributes:
        enabled: Master switch for neuromodulation features

        # Level 1: Static Gating (weight-derived by default)
        static_gating: Enable one gate per hidden node, from the node's own input weights
            (gate_source 'weights', the default), or one per connection from a dedicated CPPN
            output (gate_source 'cppn')
        gate_threshold: Threshold for binary/soft_threshold scaling (default 0.5)
        gate_scaling: How a raw gate (tanh(2 * mean |w|) of a node's inputs, or the CPPN gate
            output) becomes a gate value
            - 'sigmoid': Continuous gates in [0, 1]
            - 'binary': Hard threshold to 0 or 1
            - 'soft_threshold': Soft ramp from threshold

        # Level 2: Context-Dependent Gating (XdG-style)
        context_gating: Enable context-dependent gate modulation
        context_dim: Dimensionality of context vector (default 4)
        context_influence: How much context modulates base gates [0, 1] (default 0.5)
        context_source: 'task_id', the only value accepted: context gating takes its context
            from the current task index, 0 unless set_task_neurotransmitter sets another
            ('input' and 'learned' raise; 0.1.0 accepted and ignored them)

        # Level 3: Modulatory Neurons (activity modulation: some of a genome's own nodes set a
        # per-case gain for the others; Soltoggio's modulatory neurons gate plasticity, which EMR
        # does not have)
        modulatory_neurons: Enable modulatory neuron types
        mod_neuron_ratio: Fraction of a genome's active hidden positions that are modulatory, in
            (0, 1] (default 0.1); counted by modulatory_count_rule
        mod_connection_type: How modulatory signal affects targets
            - 'multiplicative': target = activation * (1 + mod_signal)
            - 'additive': target = activation + mod_signal
            - 'gated': target = activation * sigmoid(mod_signal)
        mod_decay: Scale of the signal of modulatory_signal 'mean_decay' and of the per-node
            weights of modulatory_node_weights 'variance' (default 0.9); not a decay over time,
            there is no accumulator
        modulatory_source: The bundle of the modulatory options, 'weights' or 'cppn'
            (MODULATORY_BUNDLES)
        modulatory_selection, modulatory_count_rule, modulatory_signal, modulatory_node_weights,
            modulatory_scope: The options that each change one thing between the two bundles;
            None takes the bundle's value
        mod_neuron_count: The modulatory count of modulatory_count_rule 'fixed', at least 1
        modulatory_cppn_read: Where a CPPN output the options read is taken: 'input_mean' (its
            mean over the input-to-node queries, HMR's read) or 'self' (the self-connection query)
        modulatory_cppn_outputs: 'read' (the CPPN carries the outputs the options read) or
            'always' (both)
        modulatory_matched_gain: 'none', 'mean' (each case's signal replaced by its mean over the
            evaluation's cases) or 'shift' (each case receives the previous case's signal,
            cyclically)

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
    context_source: str = 'task_id'  # the context is the current task index; no other value

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

    # The level the evaluation runs. Without a preset, config parsing sets the flags above from it;
    # with one, it is the preset's level (static plus context gating runs as context_gating)
    mode: str = 'disabled'  # 'disabled', 'static_gating', 'context_gating', 'modulatory_neurons', 'true_neuromodulation'

    # Where the gates and the modulatory neurons come from: 'weights' (EMR's derivations from the
    # incoming weights) or 'cppn' (HMR's: a gate per connection from a dedicated CPPN output; the
    # modulatory node type and factor from two dedicated CPPN outputs)
    gate_source: str = 'weights'
    modulatory_source: str = 'weights'

    # Modulatory-neuron options, each changing one thing the two sources differ in; None takes the
    # value of modulatory_source's bundle (resolved in __post_init__, so a config always holds
    # concrete values)
    modulatory_selection: Optional[str] = None  # 'variance', 'cppn'
    modulatory_count_rule: Optional[str] = None  # 'floor', 'ceil', 'fixed'
    modulatory_signal: Optional[str] = None  # 'mean_decay', 'sum_factor'
    modulatory_node_weights: Optional[str] = None  # 'variance', 'none'
    modulatory_scope: Optional[str] = None  # 'all', 'standard'
    mod_neuron_count: Optional[int] = None  # modulatory_count_rule 'fixed'
    modulatory_cppn_read: str = 'input_mean'  # 'input_mean', 'self'
    modulatory_cppn_outputs: str = 'read'  # 'read', 'always'
    modulatory_matched_gain: str = 'none'  # 'none', 'mean', 'shift'

    # Additional parameters
    gate_hardness: float = 10.0  # For soft_threshold gate scaling
    modulation_mode: str = 'full'  # 'full', 'gating_only', 'gain_bias_only'
    use_output_inversion: bool = True  # Whether NT4 controls output inversion
    # The neuromodulated node: 'paper' (the published rule, on every path), 'hmr_full' (HMR FULL's
    # rule, which 0.1.0's single-task path computed) or 'anti' (HMR ANTI's); see
    # true_neuromodulation_node_rule.
    node_rule: str = 'paper'
    branch_gating: bool = False  # Branch-specific gating (e.g., IH vs HH)

    # Self-connection query for receptor densities (proper CPPN-based method)
    # When True, uses batch_query_population_self_connections() to get receptor
    # densities from CPPN output at (x,y,x,y,bias) instead of deriving from weights
    use_self_connection_query: bool = False  # Default: derive from weights (current behavior)

    def __post_init__(self) -> None:
        # The modulatory options left unset take modulatory_source's bundle (an unknown source is
        # left for the parse to refuse)
        bundle = MODULATORY_BUNDLES.get(self.modulatory_source)
        if bundle is not None:
            for key, value in bundle.items():
                if getattr(self, key) is None:
                    setattr(self, key, value)


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


# TRUE Neuromodulation: Neurotransmitter Presets (Level 4)

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
    'mnist': jnp.array([0.6, 0.4, 0.7, 1.0]),
    'iris': jnp.array([0.5, 0.5, 0.5, 1.0]),
}

# 5 NT types: one-hot encoding for perfect task separation
NT_PRESETS_5: Dict[str, jnp.ndarray] = {
    'xor':  jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    'and':  jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    'or':   jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    'nand': jnp.array([0.0, 0.0, 0.0, 1.0, 0.0]),
    'nor':  jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    'parity4': jnp.array([0.8, 0.2, 0.0, 0.0, 0.0]),
    'identity': jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    'mnist': jnp.array([0.6, 0.4, 0.0, 0.0, 0.0]),
    'iris': jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
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
    'mnist': jnp.array([0.7, 0.4, 0.6]),
    'iris': jnp.array([0.5, 0.5, 0.5]),
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
    'mnist': jnp.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    'iris': jnp.array([0.17, 0.17, 0.17, 0.17, 0.17, 0.17]),
}

# HMR's name for the 2-type table, its default: the same object, so a runner that edits one edits
# both
NT_PRESETS: Dict[str, jnp.ndarray] = NT_PRESETS_2

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
MODULATION_MODES = ('full', 'gating_only', 'gain_bias_only')


def apply_gate_scaling(
    gate_raw: jnp.ndarray,
    scaling: str,
    threshold: float = 0.5,
    hardness: float = 10.0,
) -> jnp.ndarray:
    """Convert a raw gate value to a gate value in [0, 1].

    Args:
        gate_raw: Raw gate values: tanh(2 * mean |w|) of each hidden node's input weights, in
            [0, 1), for the weight-derived gates (gate_source 'weights', the default), or each
            connection's raw dedicated CPPN output (gate_source 'cppn')
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


#: HMR's neuromodulation variant used preset names as modes. Its Option A names (receptor densities
#: derived from the weight channel) are EMR's true_neuromodulation with the name's NT count and
#: receptor derivation, read at the node's self-connection as HMR read them
#: (use_self_connection_query): name -> (num_nt_types, derivation). Its Option B name (receptors
#: from dedicated CPPN outputs) is in HMR_OPTION_B_MODES. Two of its names are EMR's own, with EMR's
#: meaning: 'true_neuromodulation' (HMR: 2 NTs, Option B) and 'static_gating' (HMR: a CPPN-output
#: gate per connection; EMR derives the gate from the incoming weights); a run that reproduces an
#: HMR run with those names sets EMR's settings explicitly.
HMR_NEUROMODULATION_MODES: Dict[str, Tuple[int, str]] = {
    'true_neuromodulation_option_a_abs': (2, 'abs'),
    'true_neuromodulation_option_a_normalized': (2, 'normalized'),
    'true_neuromodulation_option_a_tanh': (2, 'tanh'),
    'true_neuromodulation_2nt_option_a_tanh': (2, 'tanh'),
    'true_neuromodulation_3nt_option_a_tanh': (3, 'tanh'),
    'true_neuromodulation_4nt_option_a_abs': (4, 'abs'),
    'true_neuromodulation_4nt_option_a_normalized': (4, 'normalized'),
    'true_neuromodulation_4nt_option_a_tanh': (4, 'tanh'),
    'true_neuromodulation_4nt_option_a_fourier': (4, 'fourier'),
    'true_neuromodulation_5nt_option_a_tanh': (5, 'tanh'),
    'true_neuromodulation_6nt_option_a_tanh': (6, 'tanh'),
}
HMR_OPTION_B_MODES = ('true_neuromodulation_4nt',)
#: HMR's level presets used as modes. HMR's evaluation took one level, in the order true
#: neuromodulation, modulatory neurons, context gating, static gating, so 'xdg_style' (static and
#: context gating) ran Level 2 and 'modulatory_only' Level 3, both from dedicated CPPN outputs:
#: EMR's mode with the CPPN source. 'full' (all three) ran Level 3 while its CPPN also carried a
#: gate output nothing read, so it is refused rather than translated to a CPPN with one output
#: fewer. HMR's 'static_gating' is also EMR's own mode name (gates from the weights by default): a
#: run that reproduces HMR's sets gate_source 'cppn'.
HMR_LEVEL_MODES: Dict[str, Tuple[str, Dict[str, str]]] = {
    'xdg_style': ('context_gating', {'gate_source': 'cppn'}),
    'modulatory_only': ('modulatory_neurons', {'modulatory_source': 'cppn'}),
}


#: HMR's 'custom' level flags (other than true_neuromodulation alone) as the level HMR evaluated and
#: its CPPN source: its gating read the CPPN's gate output, its modulatory neurons the CPPN's factor
#: and node-type outputs, and static with context gating ran context gating.
CUSTOM_LEVEL_TRANSLATIONS: Dict[Tuple[str, ...], Tuple[str, Dict[str, str]]] = {
    ('static_gating',): ('static_gating', {'gate_source': 'cppn'}),
    ('context_gating',): ('context_gating', {'gate_source': 'cppn'}),
    ('static_gating', 'context_gating'): ('context_gating', {'gate_source': 'cppn'}),
    ('modulatory_neurons',): ('modulatory_neurons', {'modulatory_source': 'cppn'}),
}


#: The levels each HMR mode name sets (its preset's flags); Option A and B names set true
#: neuromodulation.
HMR_NAMED_MODE_LEVELS: Dict[str, Tuple[str, ...]] = {
    'xdg_style': ('static_gating', 'context_gating'),
    'modulatory_only': ('modulatory_neurons',),
    **{
        name: ('true_neuromodulation',)
        for name in (*HMR_NEUROMODULATION_MODES, *HMR_OPTION_B_MODES)
    },
}


def _refuse_level_flags_against_named_mode(block: Dict[str, Any], mode: str) -> None:
    """Refuse a level flag that changes the levels an HMR mode name sets.

    HMR merged a block's level flags into the preset its mode named (get_neuromodulation_config
    updates the preset's values with the block's) and evaluated the highest level set, so
    'xdg_style' with modulatory_neurons true ran modulatory neurons. The translation of HMR's names
    dropped the flags without a word. A flag that repeats the preset's value changes nothing and is
    accepted.

    Raises:
        ValueError: A flag whose value differs from the named preset's.
    """
    levels = HMR_NAMED_MODE_LEVELS[mode]
    for flag in ('static_gating', 'context_gating', 'modulatory_neurons', 'true_neuromodulation'):
        if flag in block and bool(block[flag]) != (flag in levels):
            raise ValueError(
                f"neuromodulation.{flag}={block[flag]!r} changes the levels HMR's mode {mode!r} "
                f"sets {levels}: HMR merged the flags into its preset and evaluated the highest "
                "level set, which this translation cannot follow; set mode 'custom' with the level "
                "flags you mean")


def _translate_hmr_neuromodulation(block: Dict[str, Any]) -> Dict[str, Any]:
    """An HMR-named neuromodulation block as the equivalent EMR block.

    An Option A mode name, or 'custom' with HMR's level flags and true_neuromodulation alone,
    becomes EMR's true_neuromodulation with the name's (or the block's) NT count and receptor
    derivation and HMR's receptor source (use_self_connection_query true unless the block says
    otherwise); the block's other keys pass through. Any other block is returned as is.

    Raises:
        ValueError: A setting contradicting the name, 'custom' with another level or with Option B.
    """
    mode = block.get('mode') if isinstance(block, dict) else None
    if mode in HMR_NAMED_MODE_LEVELS:
        _refuse_level_flags_against_named_mode(block, mode)
    if mode in HMR_LEVEL_MODES:
        level, source = HMR_LEVEL_MODES[mode]
        flags = ('static_gating', 'context_gating', 'modulatory_neurons', 'true_neuromodulation')
        translated = {
            key: value for key, value in block.items() if key not in flags + ('enabled', 'mode')
        }
        for key, value in source.items():
            if translated.get(key, value) != value:
                raise ValueError(
                    f"neuromodulation.{key}={translated[key]!r} contradicts HMR's mode {mode!r} "
                    f"(its {level} came from dedicated CPPN outputs)"
                )
        translated.update(mode=level, **source)
        return translated
    if mode == 'full':
        raise ValueError(
            "neuromodulation.mode 'full' is HMR's three-level preset, which evaluated Level 3 only "
            "(its evaluation takes one level) while its CPPN also evolved a gate output nothing "
            "read; use mode modulatory_neurons with modulatory_source 'cppn' (a CPPN with one "
            "output fewer)"
        )
    if mode in HMR_OPTION_B_MODES:
        # HMR's Option B name: receptors from dedicated CPPN outputs, four NT types
        if block.get('num_nt_types', 4) != 4 or block.get('receptor_from_weight', False):
            raise ValueError(
                f"neuromodulation.mode {mode!r} is HMR's Option B with four NT types; the block "
                f"contradicts it"
            )
        # The level flags repeat the name's level (checked above), so they go as the other names' do
        translated = {
            key: value
            for key, value in block.items()
            if key
            not in (
                'enabled',
                'mode',
                'static_gating',
                'context_gating',
                'modulatory_neurons',
                'true_neuromodulation',
            )
        }
        translated.update(mode='true_neuromodulation', num_nt_types=4, receptor_from_weight=False)
        return translated
    if mode not in HMR_NEUROMODULATION_MODES and mode != 'custom':
        return block
    level_flags = ('static_gating', 'context_gating', 'modulatory_neurons', 'true_neuromodulation')
    if mode == 'custom':
        levels = tuple(flag for flag in level_flags if block.get(flag, False))
        if levels in CUSTOM_LEVEL_TRANSLATIONS:
            level, source = CUSTOM_LEVEL_TRANSLATIONS[levels]
            translated = {
                key: value
                for key, value in block.items()
                if key not in level_flags + ('enabled', 'mode')
            }
            for key, value in source.items():
                if translated.get(key, value) != value:
                    raise ValueError(
                        f"neuromodulation.{key}={translated[key]!r} contradicts HMR's 'custom' "
                        f"with {levels} (its {level} came from dedicated CPPN outputs)"
                    )
            translated.update(mode=level, **source)
            return translated
        if not levels:
            raise ValueError(
                "neuromodulation.mode 'custom' with no level flag: HMR evaluated no "
                "neuromodulation; leave the block out"
            )
        if levels != ('true_neuromodulation',):
            evaluated = levels[-1]
            raise ValueError(
                f"neuromodulation.mode 'custom' with the levels {levels}: HMR evaluated "
                f"{evaluated} alone (its evaluation takes one level) while its CPPN also evolved "
                f"the outputs of the others, which nothing read; configure {evaluated} alone"
            )
        if not block.get('receptor_from_weight', True):
            # HMR's Option B through 'custom': receptors from dedicated CPPN outputs
            translated = {
                key: value
                for key, value in block.items()
                if key not in level_flags + ('enabled', 'mode')
            }
            translated.update(
                mode='true_neuromodulation',
                num_nt_types=block.get('num_nt_types', 2),
                receptor_from_weight=False,
            )
            return translated
        num_nt, derivation = block.get('num_nt_types', 2), block.get('receptor_derivation', 'tanh')
    else:
        num_nt, derivation = HMR_NEUROMODULATION_MODES[mode]
        for key, value in (('num_nt_types', num_nt), ('receptor_derivation', derivation),
                           ('receptor_from_weight', True)):
            if key in block and block[key] != value:
                raise ValueError(
                    f"neuromodulation.{key}={block[key]!r} contradicts HMR's mode {mode!r} "
                    f"({key}={value!r})"
                )
    translated = {
        key: value for key, value in block.items() if key not in level_flags + ('enabled', 'mode')
    }
    translated.update(mode='true_neuromodulation', num_nt_types=num_nt, receptor_from_weight=True,
                      receptor_derivation=derivation,
                      use_self_connection_query=block.get('use_self_connection_query', True))
    return translated


#: Where neuromodulation's gates and modulatory neurons come from (neuromodulation.gate_source,
#: modulatory_source).
NEUROMODULATION_SOURCES = ('weights', 'cppn')


def neuromodulation_cppn_channels(block: Dict[str, Any]) -> List[str]:
    """The dedicated CPPN outputs a neuromodulation block reads, in channel order, after every other
    channel (HMR's order): Option B's receptors (one per NT type, 'receptor'), the per-connection
    gate ('gate', gate_source 'cppn', HMR's gate_from_cppn), or the modulatory factor and node type
    ('mod_factor' when the summed signal reads it, 'node_type' when CPPN selection does, both under
    modulatory_cppn_outputs 'always'; modulatory_source 'cppn' reads both); none otherwise. Resolved
    as the neuromodulation parse resolves the block, because the CPPN's output count
    is fixed before that parse runs."""
    block = _translate_hmr_neuromodulation(block or {})
    if not isinstance(block, dict):
        return []
    preset = block.get('preset')
    base = get_neuromodulation_config(preset) if preset in NEUROMODULATION_PRESETS else None
    mode = block.get('mode')
    if mode is None and base is not None:
        mode = ('true_neuromodulation' if base.true_neuromodulation else
                'modulatory_neurons' if base.modulatory_neurons else
                'context_gating' if base.context_gating else
                'static_gating' if base.static_gating else 'disabled')
    if mode == 'true_neuromodulation' and option_b_receptor_outputs(block):
        return ['receptor'] * option_b_receptor_outputs(block)
    gate_source = block.get('gate_source', 'cppn' if block.get('gate_from_cppn') else 'weights')
    if mode in ('static_gating', 'context_gating') and gate_source == 'cppn':
        return ['gate']
    if mode == 'modulatory_neurons':
        options = resolve_modulatory_options(block)
        reads = modulatory_cppn_reads(options)
        always = options['modulatory_cppn_outputs'] == 'always'
        # Today's order for the CPPN source: the factor, then the node type
        return [name for name in ('mod_factor', 'node_type') if always or name in reads]
    return []


def _neuromodulation_block_of(neuromod_config: Any) -> Dict[str, Any]:
    """A neuromodulation configuration block from a block (returned as it is) or a
    NeuromodulationConfig (its fields, the unset ones left out), for neuromodulation_cppn_channels
    and option_b_receptor_outputs."""
    if neuromod_config is None:
        return {}
    if isinstance(neuromod_config, dict):
        return neuromod_config
    return {key: value for key, value in vars(neuromod_config).items() if value is not None}


def get_cppn_num_outputs(dynamic_functions_mode: str, neuromod_config: Any = None) -> int:
    """The CPPN outputs this engine builds for a dynamic-functions mode and a neuromodulation
    configuration (a block or a NeuromodulationConfig): the weight, the activation channel under
    'cppn_output', and the dedicated outputs of neuromodulation_cppn_channels.

    HMR's name (its neuromodulation variant). HMR counted a gate output whenever gating was on and
    two modulatory outputs whenever modulatory neurons were; this engine builds the outputs a
    configuration reads, so a gate derived from the weights takes none. Like HMR's helper, it does
    not count the aggregation and LEO channels, which create_config adds."""
    outputs = 1 + int(dynamic_functions_mode == 'cppn_output')
    return outputs + len(neuromodulation_cppn_channels(_neuromodulation_block_of(neuromod_config)))


def get_cppn_receptor_base_idx(dynamic_functions_mode: str, neuromod_config: Any = None) -> int:
    """The first CPPN output of Option B's receptor densities in get_cppn_num_outputs' layout
    (HMR's name): the receptors are the last outputs, so without them it is the output count."""
    block = _neuromodulation_block_of(neuromod_config)
    return get_cppn_num_outputs(dynamic_functions_mode, block) - option_b_receptor_outputs(block)


def exact_ceil_counts(counts: np.ndarray, ratio: float) -> np.ndarray:
    """ceil(ratio x count) for each count, exactly: integer arithmetic on the ratio as the decimal a
    configuration writes, the shortest repr of its float (0.3 is 3/10, not the binary
    0.299999999999999988898...). Neither float width is exact: float32 gives 16 for 0.3 x 50,
    float64 gives 8 for 0.07 x 100 (7.000000000000001). The products are Python integers, so no
    ratio's numerator overflows them.

    Args:
        counts: (n,) non-negative integer counts.
        ratio: A finite number.

    Returns:
        (n,) int64 counts.

    Raises:
        ValueError: A ratio that is not finite (NaN or infinite).
    """
    fraction = Fraction(repr(float(ratio)))
    products = np.asarray(counts, dtype=np.int64).astype(object) * fraction.numerator
    return (-(-products // fraction.denominator)).astype(np.int64)


def cppn_modulatory_masks(node_type: jnp.ndarray, active: jnp.ndarray, ratio: float) -> jnp.ndarray:
    """Each genome's modulatory neurons from the CPPN's node-type output: the ceil(ratio x active
    count) of its active positions with the highest output (HMR took the outputs above a percentile
    over every position, inactive ones included). A genome with no active position has none.

    The count is exact on the host (:func:`exact_ceil_counts`): computed in float32 on the device,
    it would count one more where ratio x count lands just above an integer (0.3 x 50 would give 16,
    at 677 of the counts 1 to 20,000 for ratio 0.3 and 336 for 0.15; 0.05, 0.1, 0.2 and 0.25 match
    the exact count to 2^24).

    Args:
        node_type: (pop, positions) the node-type output.
        active: (pop, positions) the active positions (concrete: the counts are taken on the host).
        ratio: The modulatory share of a genome's active positions.

    Returns:
        (pop, positions) boolean mask.
    """
    return modulatory_masks(node_type, active, 'ceil', ratio)


def modulatory_masks(
    score: jnp.ndarray,
    active: jnp.ndarray,
    count_rule: str,
    ratio: Optional[float] = None,
    count: Optional[int] = None,
) -> jnp.ndarray:
    """Each genome's modulatory neurons (mode modulatory_neurons): its active positions of highest
    score, as many as the count rule (modulatory_count_rule) gives for its active count n:

    - 'floor': max(1, floor(n x ratio)) in float64 on the host, the weights mode's count, kept bit
      for bit. Float64 counts one less than the exact count where n x ratio lands just below an
      integer (0.7 x 90 gives 62, the exact count is 63); it is exact at 0.05, 0.1, 0.15, 0.2, 0.25
      and 0.3 for every n to 2^24.
    - 'ceil': ceil(n x ratio), exact (:func:`exact_ceil_counts`), the CPPN source's count.
    - 'fixed': min(count, n).

    The rank is by descending score within the genome, a tie going to the lower position. An
    inactive position is never modulatory, so a genome with no active position has none whatever the
    rule.

    Args:
        score: (pop, positions) the selection score: input-weight variance or the node-type output.
        active: (pop, positions) the active positions (concrete: the counts are taken on the host).
        count_rule: 'floor', 'ceil' or 'fixed'.
        ratio: mod_neuron_ratio, read by 'floor' and 'ceil'.
        count: mod_neuron_count, read by 'fixed'.

    Returns:
        (pop, positions) boolean mask.
    """
    ranked = jnp.where(active, score, -jnp.inf)
    rank = jnp.argsort(jnp.argsort(-ranked, axis=1), axis=1)
    active_count = np.asarray(jnp.sum(active, axis=1))
    if count_rule == 'floor':
        counts = np.maximum(1, np.floor(active_count * float(ratio)))
    elif count_rule == 'ceil':
        counts = exact_ceil_counts(active_count, ratio)
    elif count_rule == 'fixed':
        counts = np.minimum(int(count), active_count)
    else:
        raise ValueError(f"unknown modulatory count rule {count_rule!r}")
    return (rank < jnp.asarray(counts.astype(np.int32))[:, None]) & active


def check_modulatory_options(config: 'NeuromodulationConfig', block: Dict[str, Any]) -> None:
    """Refuse modulatory-neuron options (mode modulatory_neurons) that cannot apply, or that a
    setting would make the run ignore.

    Args:
        config: The parsed configuration, the options resolved to their bundle where the block
            leaves them unset.
        block: The neuromodulation block as given (HMR's names translated), for the keys it sets.

    Raises:
        ValueError: An unknown value. ``mod_neuron_ratio`` outside (0, 1]: above 1 every active
            position is modulatory, and at 0 or below no position is (one under the weights mode's
            floor). Count rule ``'fixed'`` without ``mod_neuron_count``, or with
            ``mod_neuron_ratio`` set, which it does not read; ``mod_neuron_count`` under another
            rule, or not an integer of at least 1. ``mod_decay`` set where neither the signal
            (``'mean_decay'``) nor the node weights (``'variance'``) read it, as with the CPPN
            source. ``modulatory_cppn_read`` set where no option reads a CPPN output.
    """
    for key, values in MODULATORY_OPTIONS.items():
        value = getattr(config, key)
        if value not in values:
            raise ValueError(f"neuromodulation.{key} must be one of {values}, got {value!r}")
    rule = config.modulatory_count_rule
    if rule == 'fixed':
        if 'mod_neuron_count' not in block:
            raise ValueError(
                "neuromodulation.modulatory_count_rule 'fixed' counts mod_neuron_count modulatory "
                "neurons per genome; set it"
            )
        if 'mod_neuron_ratio' in block:
            raise ValueError(
                "neuromodulation.mod_neuron_ratio is not read under modulatory_count_rule 'fixed', "
                "which counts mod_neuron_count; remove it"
            )
        count = config.mod_neuron_count
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError(
                f"neuromodulation.mod_neuron_count must be an integer of at least 1, got {count!r}"
            )
    else:
        if 'mod_neuron_count' in block:
            raise ValueError(
                "neuromodulation.mod_neuron_count is read by modulatory_count_rule 'fixed' only, "
                f"not {rule!r}; remove it or set the rule"
            )
        ratio = config.mod_neuron_ratio
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float, np.integer, np.floating))
                or not 0 < ratio <= 1):
            raise ValueError(
                f"neuromodulation.mod_neuron_ratio must be in (0, 1], got {ratio!r}: above 1 every "
                "active position is modulatory, and at 0 or below none is (one under the 'floor' "
                "rule)"
            )
    if ('mod_decay' in block and config.modulatory_signal != 'mean_decay'
            and config.modulatory_node_weights != 'variance'):
        # HMR's Level 3 ignored it; only a key the block sets is refused
        raise ValueError(
            "neuromodulation.mod_decay scales the signal of modulatory_signal 'mean_decay' and the "
            "per-node weights of modulatory_node_weights 'variance'; with signal "
            f"{config.modulatory_signal!r} and node weights {config.modulatory_node_weights!r} "
            "(the CPPN source has no decay) nothing reads it, so the setting would be ignored: "
            "remove mod_decay"
        )
    if 'modulatory_cppn_read' in block and not modulatory_cppn_reads(vars(config)):
        raise ValueError(
            "neuromodulation.modulatory_cppn_read says where a CPPN output is read, and no option "
            "reads one (selection 'variance', signal 'mean_decay'), so the setting would be "
            "ignored: remove it"
        )


def option_b_receptor_outputs(block: Dict[str, Any]) -> int:
    """The CPPN outputs Option B's receptor densities take: one per NT type when true
    neuromodulation reads its receptors from dedicated outputs (receptor_from_weight false), else 0.
    Resolved as the neuromodulation parse resolves the block (HMR's names translated, a preset's
    values under the block's), because the CPPN's output count
    is fixed before that parse runs."""
    block = _translate_hmr_neuromodulation(block or {})
    if not isinstance(block, dict):
        return 0
    preset = block.get('preset')
    base = get_neuromodulation_config(preset) if preset in NEUROMODULATION_PRESETS else None
    true_neuromodulation = (block.get('mode') == 'true_neuromodulation'
                            or (base is not None and base.true_neuromodulation))
    from_weight = block.get(
        'receptor_from_weight', base.receptor_from_weight if base is not None else True
    )
    if not true_neuromodulation or from_weight:
        return 0
    return int(block.get('num_nt_types', base.num_nt_types if base is not None else 4))


#: The neuromodulated node's rules (neuromodulation.node_rule): the paper's, HMR FULL's and HMR
#: ANTI's (true_neuromodulation_node_rule).
NEUROMODULATION_NODE_RULES = ('paper', 'hmr_full', 'anti')


def modulatory_neuron_masks(weight_var: jnp.ndarray, active: jnp.ndarray, ratio: float) -> jnp.ndarray:
    """Each genome's modulatory neurons (neuromodulation mode modulatory_neurons): `ratio` of its own active
    positions, at least one, those of highest input-weight variance.

    The positions are each genome's own, so a genome's network does not depend on the rest of the
    population or use positions it does not have; and they are a fraction of its active positions,
    not of every grid position, which at depths 3 to 5 would make nearly every genome's whole active
    set modulatory.

    The count is :func:`modulatory_masks`' 'floor' rule, in float64 on the host, as Python's
    int(n * ratio) is. Float64 is the deviation where it differs from the exact count: 0.7 of 90
    active positions is 63 exactly (and in float32), 62 in float64, which this count keeps (the
    count since 0.2.0).

    Args:
        weight_var: (pop_size, positions) variance of each position's input weights (any selection
            score).
        active: (pop_size, positions) active-position mask (concrete: the counts are taken on the host).
        ratio: mod_neuron_ratio, the modulatory fraction of a genome's active positions.

    Returns:
        (pop_size, positions) boolean mask.
    """
    return modulatory_masks(weight_var, active, 'floor', ratio)


#: A node rule (:func:`_node_rule`) and the arrays it reads after the pre-activation.
NodeRule = Tuple[Callable, Tuple[jnp.ndarray, ...]]


def _neuromodulated_rule(form: str, act_rule: Callable) -> Callable:
    """A neuromodulated node over the node's activation rule, made once per form and activation.

    The rule's operands are the form's per-node terms, then the activation rule's operands:

    - ``'gate'``: ``(gates,)``, ``act(pre) * gates``;
    - ``'gain'``: ``(gains,)``, ``act(gains * pre)``;
    - ``'gain_gate'``: ``(gains, gates)``, ``act(gains * pre) * gates``;
    - ``'gain_bias'``: ``(gains, bias)``, ``act(gains * pre + bias)``;
    - ``'gain_bias_gate'``: ``(gains, bias, gates)``, ``act(gains * pre + bias) * gates``.
    """
    def make():
        if form == 'gate':
            def rule(pre, gates, *act_operands):
                return act_rule(pre, *act_operands) * gates
        elif form == 'gain':
            def rule(pre, gains, *act_operands):
                return act_rule(gains * pre, *act_operands)
        elif form == 'gain_gate':
            def rule(pre, gains, gates, *act_operands):
                return act_rule(gains * pre, *act_operands) * gates
        elif form == 'gain_bias':
            def rule(pre, gains, bias, *act_operands):
                return act_rule(gains * pre + bias, *act_operands)
        elif form == 'gain_bias_gate':
            def rule(pre, gains, bias, gates, *act_operands):
                return act_rule(gains * pre + bias, *act_operands) * gates
        else:
            raise ValueError(f"unknown neuromodulated form {form!r}")
        return rule

    return _node_rule(('neuromodulated', form, act_rule), make, act_rule)


def gated_node_rule(act: NodeRule, gates: jnp.ndarray) -> NodeRule:
    """A node gated after its activation, ``act(pre) * gates`` (neuromodulation modes
    static_gating and context_gating), as a node rule and its operands.

    Args:
        act: The node's activation as a node rule and its operands (:func:`node_activation_rule`).
        gates: (num_positions,) gates.
    """
    act_rule, act_operands = act
    return _neuromodulated_rule('gate', act_rule), (gates,) + tuple(act_operands)


def modulatory_node_rule(
    act: NodeRule,
    arrays: Tuple[jnp.ndarray, ...],
    mod_connection_type: str,
    signal: str = 'mean_decay',
    node_weights: str = 'variance',
    scope: str = 'all',
    mod_decay: Optional[float] = None,
    matched_gain: str = 'none',
) -> NodeRule:
    """A node under the genome's own modulatory neurons (neuromodulation mode modulatory_neurons),
    as a node rule made once per setting and activation, and its operands (one rule for every
    combination of the modulatory options). The key holds the static values (signal, node weights,
    scope, connection type, ``mod_decay`` where the signal reads it, matched gain, activation) and
    the arrays ride as operands, so a run makes the rule once and no generation compiles again.

    On each case the node's activity is ``h = act(pre)`` and the signal ``s`` is

    - ``'mean_decay'``: the mean activity of the modulatory neurons, squashed by tanh and scaled by
      ``mod_decay`` (the weights mode's signal);
    - ``'sum_factor'``: tanh of the sum of each modulatory neuron's activity times its factor (the
      CPPN source's).

    ``matched_gain`` ``'mean'`` replaces each case's ``s`` by its mean over the cases the rule is
    given (the evaluation's cases); ``'shift'`` gives case ``c`` the signal of case ``c - 1`` and
    case 0 the last case's. With ``node_weights`` ``'variance'`` the signal is scaled per node by
    the weights ``w`` (``'none'``: ``w = 1``), and the node outputs ``h * (1 + s w)``
    (multiplicative), ``h + s w`` (additive) or ``h * sigmoid(s w)`` (gated): every node under scope
    ``'all'``, the standard nodes only under ``'standard'``, where a modulatory neuron outputs
    ``h``. Each bundle's expression keeps its operation order, so the two sources compute what they
    did.

    Args:
        act: The node's activation as a node rule and its operands (:func:`node_activation_rule`).
        arrays: The genome's arrays, each (num_positions,), in this order: the per-node weights
            (node weights ``'variance'``), the factors (signal ``'sum_factor'``), then the
            modulatory neurons' mask.
        mod_connection_type: ``'multiplicative'``, ``'additive'`` or ``'gated'``.
        signal: ``'mean_decay'`` or ``'sum_factor'``.
        node_weights: ``'variance'`` or ``'none'``.
        scope: ``'all'`` or ``'standard'``.
        mod_decay: The signal's scale under ``'mean_decay'``.
        matched_gain: ``'none'``, ``'mean'`` or ``'shift'``.
    """
    act_rule, act_operands = act
    weighted = node_weights == 'variance'
    summed = signal == 'sum_factor'
    num_arrays = 1 + weighted + summed
    if len(arrays) != num_arrays:
        raise ValueError(
            f"modulatory_node_rule takes {num_arrays} arrays for node weights {node_weights!r} and "
            f"signal {signal!r}, got {len(arrays)}"
        )
    decay = None if summed else mod_decay

    def make():
        def rule(pre, *operands):
            weights = operands[0] if weighted else None
            factor = operands[int(weighted)] if summed else None
            mask = operands[num_arrays - 1]
            h = act_rule(pre, *operands[num_arrays:])
            if summed:
                s = jnp.tanh(jnp.sum(h * jnp.where(mask, factor, 0.0), axis=1, keepdims=True))
            else:
                # The mean activity of this genome's modulatory neurons (a per-genome mask)
                mod_activity = (jnp.sum(h * mask, axis=1, keepdims=True)
                                / jnp.maximum(jnp.sum(mask), 1))
                s = jnp.tanh(mod_activity) * decay
            if matched_gain == 'mean':
                s = jnp.mean(s, axis=0, keepdims=True)
            elif matched_gain == 'shift':
                s = jnp.roll(s, 1, axis=0)
            if weighted:
                s = s * weights
            if mod_connection_type == 'multiplicative':
                changed = h * (1.0 + s)
            elif mod_connection_type == 'additive':
                changed = h + s
            else:
                changed = h * jax.nn.sigmoid(s)
            return jnp.where(mask, h, changed) if scope == 'standard' else changed
        return rule

    rule = _node_rule(
        (
            'modulatory_neurons',
            signal,
            node_weights,
            scope,
            mod_connection_type,
            decay,
            matched_gain,
            act_rule,
        ),
        make,
        act_rule,
    )
    return rule, tuple(arrays) + tuple(act_operands)


def true_neuromodulation_node_rule(
    act: NodeRule,
    receptors: jnp.ndarray,
    base_gains: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    mod_strength: float,
    num_nt: int,
    modulation_mode: str,
    rule: str = 'paper',
    modulation: Optional[jnp.ndarray] = None,
) -> NodeRule:
    """A node under true neuromodulation, as a node rule and its operands.

    The first three neurotransmitter channels select among a node's receptors,
    m = receptors[:, :3] @ nt[:3]; the fourth only inverts outputs (:func:`invert_outputs`).

    'paper' (the published rule; HMR's neuromodulation variant, and the rule of every EMR path):
    gating_only act(pre) * sigmoid(s m); otherwise gains clip(bg + s m, 0.1, 5) (NaN to 1) and bias
    s m, gain_bias_only act(gains pre + bias), full act(gains pre + bias) * sigmoid(m). 'hmr_full'
    (HMR FULL's rule, which 0.1.0's single-task path computed): gains bg + s m, gate sigmoid(5 m);
    gating_only act(pre) * gate, gain_bias_only act(gains pre), full act(gains pre) * gate. 'anti'
    (HMR ANTI's rule, its anti-correlated-receptors variant): m over every channel, m = receptors @
    nt, gains clip(bg + s m, 0.1, 5), act(gains pre) * sigmoid(m); modulation_mode 'full' only, no
    output inversion and no branch gating (the configuration checks refuse the others).

    ``modulation``, when given, is m: branch gating routes one receptor channel per region of the
    grid (:func:`compute_modulation_with_branch_gating`), and either rule applies to it.

    The gains, bias and gates are computed here, once per genome and task; the rule
    (:func:`_neuromodulated_rule`) is made once per form and activation, so a settling step on it
    is made once and carries them (:func:`settle_hidden`).

    Args:
        act: The node's activation as a node rule and its operands (:func:`node_activation_rule`;
            a function of the pre-activation alone is ``(fn, ())``).
        receptors: (num_positions, num_nt) receptor densities.
        base_gains: (num_positions,) base gains.
        neurotransmitter: The task's neurotransmitter vector.
        mod_strength: Modulation strength s.
        num_nt: Number of neurotransmitter channels.
        modulation_mode: 'full', 'gating_only' or 'gain_bias_only'.
        rule: 'paper', 'hmr_full' or 'anti'.
        modulation: m when given (branch gating), else computed from the receptors.

    Returns:
        ``(node_rule, operands)``: the node is ``node_rule(pre, *operands)``.
    """
    act_rule, act_operands = act
    if modulation is None:
        k = num_nt if rule == 'anti' else min(3, num_nt)
        modulation = receptors[:, :k] @ neurotransmitter[:k]
    if rule == 'anti':
        form, terms = 'gain_gate', (jnp.clip(base_gains + mod_strength * modulation, 0.1, 5.0),
                                    jax.nn.sigmoid(modulation))
    elif rule == 'hmr_full':
        gains, gates = base_gains + mod_strength * modulation, jax.nn.sigmoid(modulation * 5.0)
        if modulation_mode == 'gating_only':
            form, terms = 'gate', (gates,)
        elif modulation_mode == 'gain_bias_only':
            form, terms = 'gain', (gains,)
        else:
            form, terms = 'gain_gate', (gains, gates)
    elif modulation_mode == 'gating_only':
        form, terms = 'gate', (jax.nn.sigmoid(modulation * mod_strength),)
    else:
        gains = jnp.clip(base_gains + mod_strength * modulation, 0.1, 5.0)
        gains = jnp.where(jnp.isnan(gains), 1.0, gains)
        bias = modulation * mod_strength
        if modulation_mode == 'gain_bias_only':
            form, terms = 'gain_bias', (gains, bias)
        else:
            form, terms = 'gain_bias_gate', (gains, bias, jax.nn.sigmoid(modulation))
    return _neuromodulated_rule(form, act_rule), terms + tuple(act_operands)


def true_neuromodulation_node_fn(
    act_fn: Callable[[jnp.ndarray], jnp.ndarray],
    receptors: jnp.ndarray,
    base_gains: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    mod_strength: float,
    num_nt: int,
    modulation_mode: str,
    rule: str = 'paper',
    modulation: Optional[jnp.ndarray] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """A node under true neuromodulation, as a function of its pre-activation.

    :func:`true_neuromodulation_node_rule` (the rules are described there) with its operands
    bound: a new function on every call, so a settling step on it is made per call; the
    evaluations settle through the rule and its operands instead.

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
    node_rule, operands = true_neuromodulation_node_rule(
        (act_fn, ()),
        receptors,
        base_gains,
        neurotransmitter,
        mod_strength,
        num_nt,
        modulation_mode,
        rule,
        modulation,
    )
    return lambda pre: node_rule(pre, *operands)


def invert_outputs(
    outputs: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    use_output_inversion: bool,
    num_nt: int,
    rule: str = 'paper',
) -> jnp.ndarray:
    """The fourth neurotransmitter channel's output inversion, clipped to [0, 1] under the paper's rule.

    Only a vector of four or more channels has an inversion channel. JAX clamps an out-of-range
    index to the last element, so reading nt[3] whatever the count would let a modulation channel
    invert the outputs when there are two or three neurotransmitters.
    """
    if use_output_inversion and num_nt >= 4:
        invert = jnp.clip(neurotransmitter[3], 0.0, 1.0) if rule == 'paper' else neurotransmitter[3]
        return invert * outputs + (1 - invert) * (1 - outputs)
    return outputs
