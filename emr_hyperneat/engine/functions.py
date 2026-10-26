"""What a node computes: the activation and aggregation registries and their index decodes, the
per-node assignment of functions (CPPN outputs, weight interpretation, random), palettes and their
evolution, the segment operations of hidden-to-hidden aggregation, and the forward primitives
(the input drive, the hidden activation, the dense forward)."""
import numbers
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.engine.config import _refuse_unknown_keys


#: Weight interpretations of dynamic_functions mode 'weight_interpretation'. The last five are HMR's
#: (its dynamic-functions variant); 0.1.0 ran them as magnitude_bio.
ACTIVATION_INTERPRETATIONS = ('magnitude_bio', 'variance', 'sign', 'magnitude', 'combined',
                              'hierarchical_sparsity', 'sparsity_threshold', 'stp_inspired')
#: Aggregation interpretations (aggregation mode 'weight_interpretation' and
#: modular.aggregation_method).
AGGREGATION_INTERPRETATIONS = ('magnitude_bio', 'variance', 'sign_uniformity', 'fan_in')
#: The three layers of dynamic_functions mode 'modular' (HMR's names), and its aggregation method.
MODULAR_ACTIVATION_METHODS = ('none', 'magnitude_bio', 'magnitude', 'sign', 'variance')
MODULAR_SPARSITY_METHODS = ('none', 'hierarchical', 'threshold', 'wta')
MODULAR_SCALING_METHODS = ('none', 'stp_inspired')
MODULAR_AGGREGATION_METHODS = ('none',) + AGGREGATION_INTERPRETATIONS


def activation_interpretation_signal(weights: jnp.ndarray, interpretation: str) -> jnp.ndarray:
    """The per-node value a weight interpretation decodes into an activation index.

    The formulas are HMR's (dynamic-functions variant). The three sparsity and scaling
    interpretations select the activation as magnitude_bio does; what else they do is applied at
    evaluation.

    Args:
        weights: Incoming weights, shape (pop_size, num_inputs, num_nodes).
        interpretation: One of ACTIVATION_INTERPRETATIONS.

    Returns:
        Values in [-1, 1] for :func:`continuous_to_index`, shape (pop_size, num_nodes).
    """
    if interpretation == 'sign':
        return jnp.mean(jnp.sign(weights), axis=1)
    if interpretation == 'variance':
        return jnp.tanh(jnp.var(weights, axis=1) * 5)
    mean_abs = jnp.mean(jnp.abs(weights), axis=1)
    if interpretation == 'magnitude':
        # High magnitude to a high index (HMR's "computational" mapping)
        return jnp.tanh(mean_abs * 2 - 1)
    if interpretation == 'combined':
        weight_var = jnp.var(weights, axis=1)
        pos_ratio = jnp.mean((weights > 0).astype(jnp.float32), axis=1)
        return (
            0.5 * jnp.tanh(mean_abs * 2)
            + 0.3 * jnp.tanh(weight_var * 5)
            + 0.2 * (pos_ratio - 0.5) * 2
        )
    if interpretation in (
        'magnitude_bio',
        'hierarchical_sparsity',
        'sparsity_threshold',
        'stp_inspired',
    ):
        # High magnitude to a low index (bounded functions first: divisive normalization)
        return -jnp.tanh(mean_abs * 2 - 1)
    raise ValueError(f"unknown activation interpretation {interpretation!r}; the names are "
                     f"{ACTIVATION_INTERPRETATIONS}")


def level_sparsity_quantiles(
    level_offsets: Tuple[int, ...], sparsity: Dict[str, float]
) -> jnp.ndarray:
    """Per-position quantile under which a hidden value is zeroed, by the position's grid level
    (HMR's hierarchical sparsity): level 0 keeps the top ``level_0`` fraction of each sample's
    active values, level 1 ``level_1``, deeper levels ``level_2_plus``.

    Args:
        level_offsets: The grid's cumulative level offsets, (0, 4, 20, 84, ...).
        sparsity: 'level_0', 'level_1' and 'level_2_plus', each the fraction of values kept.

    Returns:
        Quantiles in [0, 1], shape (total_positions,).
    """
    levels = np.repeat(np.arange(len(level_offsets) - 1), np.diff(np.asarray(level_offsets)))
    kept = np.where(levels == 0, sparsity['level_0'],
                    np.where(levels == 1, sparsity['level_1'], sparsity['level_2_plus']))
    return jnp.asarray(1.0 - kept, dtype=jnp.float32)


def node_sparsity_quantiles(weights: jnp.ndarray) -> jnp.ndarray:
    """Per-node quantile from the incoming weight magnitude (HMR's sparsity threshold): a weak node
    keeps less of its values. In [0.5, 0.95], shape (pop_size, num_nodes)."""
    mean_abs = jnp.mean(jnp.abs(weights), axis=1)
    return jax.nn.sigmoid(-mean_abs * 2 + 1) * 0.45 + 0.5


def stp_depression(weights: jnp.ndarray) -> jnp.ndarray:
    """Per-node factor that scales a node's incoming weights (HMR's static short-term-plasticity
    approximation): a strong node, with a high release probability U, is depressed more. In
    [0.6, 0.95], shape (pop_size, num_nodes)."""
    mean_abs = jnp.mean(jnp.abs(weights), axis=1)
    release = jax.nn.sigmoid(mean_abs * 2) * 0.7 + 0.1
    return 1.0 - release * 0.5


def sparsify_by_quantile(
    hidden: jnp.ndarray, quantiles: jnp.ndarray, node_mask: jnp.ndarray
) -> jnp.ndarray:
    """Zero each hidden value below its node's quantile of the sample's absolute values over the
    active nodes.

    HMR took the quantile over every position of the evaluation axis, so a genome's inactive
    positions (and the padding of its sparse path) entered the order statistics with the value their
    function gives at zero input (0 for tanh, 0.5 for sigmoid, 1 for gauss), and the kept fraction
    depended on how many positions were inactive. Here only the genome's active nodes count.

    Args:
        hidden: One genome's hidden values, shape (num_samples, num_nodes).
        quantiles: Per-node quantile in [0, 1], shape (num_nodes,).
        node_mask: The genome's active nodes, shape (num_nodes,).

    Returns:
        ``hidden`` with the values below their node's threshold set to 0.
    """
    magnitude = jnp.abs(hidden)
    ordered = jnp.sort(jnp.where(node_mask[None, :], magnitude, jnp.inf), axis=-1)
    active = jnp.sum(node_mask.astype(jnp.int32))
    rank = jnp.clip(
        jnp.floor(quantiles * (active - 1)).astype(jnp.int32), 0, jnp.maximum(active - 1, 0)
    )
    threshold = jnp.take(ordered, rank, axis=-1)          # (num_samples, num_nodes)
    return jnp.where(magnitude >= threshold, hidden, 0.0)


def sparsify_winner_take_all(
    hidden: jnp.ndarray, k_percent: float, node_mask: jnp.ndarray
) -> jnp.ndarray:
    """Keep the ``k_percent`` largest absolute values of each sample among the active nodes (at
    least one).

    HMR counted the k from every position of the evaluation axis, active or not; here it is a
    fraction of the genome's active nodes.
    """
    magnitude = jnp.abs(hidden)
    ordered = jnp.sort(jnp.where(node_mask[None, :], magnitude, -jnp.inf), axis=-1)
    active = jnp.sum(node_mask.astype(jnp.int32))
    keep = jnp.maximum(1, jnp.floor(active * k_percent).astype(jnp.int32))
    threshold = jnp.take(ordered, hidden.shape[-1] - keep, axis=-1)[:, None]
    return jnp.where(magnitude >= threshold, hidden, 0.0)


def parse_function_palette(entries: Any, names: List[str], where: str) -> jnp.ndarray:
    """A palette given as indices of ``names`` or as names, checked; an int32 array of indices.

    Args:
        entries: A non-empty list of indices or names.
        names: ACTIVATION_LIST or AGGREGATION_LIST.
        where: The setting, for the error message.

    Raises:
        ValueError: An entry is neither an index of ``names`` nor one of them, or the list is empty.
    """
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError(
            f"{where} must be a non-empty list of indices or names of {names}, got {entries!r}"
        )
    indices = []
    for entry in entries:
        if isinstance(entry, str) and entry in names:
            indices.append(names.index(entry))
        elif (isinstance(entry, (int, np.integer)) and not isinstance(entry, bool)
              and 0 <= int(entry) < len(names)):
            indices.append(int(entry))
        else:
            raise ValueError(
                f"{where} entry {entry!r} is neither an index (0 to {len(names) - 1}) nor one of "
                f"the names {names}"
            )
    return jnp.array(indices, dtype=jnp.int32)


#: dynamic_functions.palette_evolution's settings (from HMR's dynamic-functions variant).
PALETTE_EVOLUTION_KEYS = frozenset({
    'enabled', 'mutation_rate', 'min_active', 'initial_mask', 'stagnation_generations', 'decode',
})
#: How the per-node decode reads an evolving palette. 'stable': the channel is binned over the whole
#: ACTIVATION_LIST and a disabled function snaps to its nearest enabled one, so toggling one
#: function changes only the nodes of its bin. 'rebin': HMR's, the channel binned over the enabled
#: functions, so every toggle moves every bin and remaps a quarter to a half of every genome's
#: nodes, the champion's included, with no change to any genome.
PALETTE_EVOLUTION_DECODES = ('stable', 'rebin')


def parse_palette_evolution(block: Dict[str, Any]) -> Dict[str, Any]:
    """dynamic_functions.palette_evolution, checked (HMR's fell back silently: an unknown
    initial_mask ran 'full', out-of-range indices were dropped, min_active was never checked against
    the initial mask).

    Returns:
        mutation_rate, min_active, stagnation_generations, decode, and initial_mask (a boolean mask
        over ACTIVATION_LIST).

    Raises:
        ValueError: an unknown key or value.
    """
    _refuse_unknown_keys('dynamic_functions.palette_evolution', block, PALETTE_EVOLUTION_KEYS)
    mutation_rate = float(block.get('mutation_rate', 0.1))
    if not 0.0 < mutation_rate <= 1.0:
        raise ValueError(
            "dynamic_functions.palette_evolution.mutation_rate must be in (0, 1], got "
            f"{mutation_rate}"
        )
    initial = block.get('initial_mask', 'full')
    if isinstance(initial, str):
        if initial not in ('full', 'default'):
            raise ValueError(
                "dynamic_functions.palette_evolution.initial_mask must be 'full', 'default' or a "
                f"list of functions, got {initial!r}"
            )
        indices = (
            list(range(len(ACTIVATION_LIST))) if initial == 'full' else PALETTE_CONFIGS['default']
        )
    else:
        indices = [int(i) for i in parse_function_palette(
            initial, ACTIVATION_LIST, 'dynamic_functions.palette_evolution.initial_mask')]
    mask = np.zeros(len(ACTIVATION_LIST), dtype=bool)
    mask[indices] = True
    min_active = int(block.get('min_active', 2))
    if not 1 <= min_active <= int(mask.sum()):
        raise ValueError(
            "dynamic_functions.palette_evolution.min_active must be between 1 and the initial "
            f"mask's {int(mask.sum())} functions, got {min_active}"
        )
    stagnation = int(block.get('stagnation_generations', 5))
    if stagnation < 1:
        raise ValueError(
            f"dynamic_functions.palette_evolution.stagnation_generations must be at least 1, got "
            f"{stagnation}"
        )
    decode = block.get('decode', 'stable')
    if decode not in PALETTE_EVOLUTION_DECODES:
        raise ValueError(
            "dynamic_functions.palette_evolution.decode must be one of "
            f"{PALETTE_EVOLUTION_DECODES}, "
            f"got {decode!r}"
        )
    return dict(
        mutation_rate=mutation_rate,
        min_active=min_active,
        stagnation_generations=stagnation,
        decode=decode,
        initial_mask=mask,
    )


#: dynamic_functions.live_strategy's settings (HMR's live palette strategy).
LIVE_STRATEGY_KEYS = frozenset(
    {'enabled', 'perturbation_rate', 'perturbation_strength', 'perturbation_decay'}
)
#: Folded into state.randkey to draw the genomes the live strategy perturbs: the run's own key
#: stream is left as it is.
LIVE_STRATEGY_KEY = 0x11FE


def parse_live_strategy(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """dynamic_functions.live_strategy, checked (HMR read its settings unchecked).

    Returns:
        None when the block is absent or disabled; otherwise perturbation_rate (the share of the
        population perturbed when the palette grows, in [0, 1]), perturbation_strength (added to the
        bias of the CPPN's activation output node) and perturbation_decay (the rate falls by decay *
        generation / 100, floored at 0).

    Raises:
        ValueError: an unknown key, a wrong value, or settings in a disabled block (they would be
            ignored).
    """
    if not isinstance(block, dict):
        raise ValueError(f"dynamic_functions.live_strategy must be a mapping, got {block!r}")
    _refuse_unknown_keys('dynamic_functions.live_strategy', block, LIVE_STRATEGY_KEYS)
    enabled = block.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError(f"dynamic_functions.live_strategy.enabled must be a bool, got {enabled!r}")
    if not enabled:
        if set(block) - {'enabled'}:
            raise ValueError(
                "dynamic_functions.live_strategy settings are given but live_strategy.enabled is "
                "not true, so they would be ignored"
            )
        return None
    settings = {}
    for key, default, low, high in (('perturbation_rate', 0.15, 0.0, 1.0),
                                    ('perturbation_strength', 0.3, -np.inf, np.inf),
                                    ('perturbation_decay', 0.0, 0.0, np.inf)):
        value = block.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not low <= value <= high
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"dynamic_functions.live_strategy.{key} must be a finite number in [{low}, "
                f"{high}], got {value!r}"
            )
        settings[key] = float(value)
    return settings


def palette_snap_table(mask: np.ndarray) -> np.ndarray:
    """The stable decode's map: for every ACTIVATION_LIST position, the nearest enabled function
    (ties to the lower).

    Args:
        mask: (len(ACTIVATION_LIST),) bool, the enabled functions (at least one).

    Returns:
        (len(ACTIVATION_LIST),) int32 ACTIVATION_LIST indices.
    """
    enabled = np.flatnonzero(mask)
    positions = np.arange(mask.shape[0])
    return enabled[np.argmin(np.abs(positions[:, None] - enabled[None, :]), axis=1)].astype(
        np.int32
    )


def mutate_palette_mask(
    key: jnp.ndarray, mask: np.ndarray, mutation_rate: float, min_active: int
) -> np.ndarray:
    """HMR's palette mutation: every function's bit flips with probability ``mutation_rate``; a
    result with fewer than ``min_active`` functions keeps the old mask. The draw is HMR's
    (``jax.random.uniform`` over the list).
    """
    flip = np.asarray(jax.random.uniform(key, (mask.shape[0],))) < mutation_rate
    new = np.where(flip, ~mask, mask)
    return new if int(new.sum()) >= min_active else mask


# Dynamic Activation Functions Support
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
    # last axis = output units). Only meaningful as an output activation on multi-output
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

#: The number of functions in ACTIVATION_LIST: 18 (HMR's name; its dynamic-functions variant's list
#: holds the same 18, although its comment says 13)
NUM_ACTIVATIONS = len(ACTIVATION_LIST)


#: Where cppn_output reads a node's function channel ('self': at the node's self-connection, EMR's
#: read; 'input_mean': averaged over the input-to-node queries, HMR's dynamic-functions and
#: aggregation variants) and what the weight interpretations read ('w1': EMR's W1, the weight
#: channel scaled by tanh times max_weight and thresholded; 'raw': the weight channel itself,
#: unmasked, as HMR read it). Opt-ins, for runs that reproduce HMR's.
HMR_READS = {'cppn_read': ('self', 'input_mean'), 'weight_source': ('w1', 'raw')}


def _parse_hmr_reads(section: str, block: Dict[str, Any]) -> Tuple[str, str]:
    """The block's cppn_read and weight_source, checked against HMR_READS."""
    values = []
    for key, names in HMR_READS.items():
        value = block.get(key, names[0])
        if value not in names:
            raise ValueError(f"{section}.{key} must be one of {names}, got {value!r}")
        values.append(value)
    return values[0], values[1]


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
        raw: Continuous values, in [-1, 1] for every caller in the engine (CPPN channels and
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


# Palette Support for Custom Activation Function Sets
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
    'phase4_all': list(range(13, 18)),       # osc_adapt, gain_mod, receptive, band_pass, integrate
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
    # the unstack, and a traced index cannot select a function, so every recurrent palette
    # configuration would fail under hh_activation_mode 'every_iteration'.
    for act_idx in np.asarray(palette).tolist():
        func = ACTIVATION_FUNCTIONS[ACTIVATION_LIST[act_idx]]
        mask = (act_indices == act_idx)
        activated = func(pre_activation)
        result = jnp.where(mask[None, :] if pre_activation.ndim == 2 else mask, activated, result)
    return result


# Dynamic Aggregation Functions Support
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

    Hypothesis: Aggregation should relate to weight distribution.
    - High variance -> max/min (select dominant input)
    - Low variance -> mean (average inputs equally)
    - Mixed signs -> sum (preserve cancellation effects)
    - Uniform signs -> product (AND-like behavior)

    Args:
        weights: Weight matrix, shape (pop_size, num_inputs, num_nodes)
        interpretation: Method name: 'magnitude_bio', 'variance', 'sign_uniformity'
        num_aggregations: Number of aggregation functions available
        decode: 'tanh' or 'uniform' (INDEX_DECODES)

    Returns:
        Aggregation indices, shape (pop_size, num_nodes)
    """
    if interpretation == 'magnitude_bio':
        # High magnitude -> sum (preserve scale), low -> mean (normalize)
        mean_abs = jnp.mean(jnp.abs(weights), axis=1)  # (pop_size, num_nodes)
        normalized = -jnp.tanh(mean_abs * 2 - 1)  # Inverted like activation
        return continuous_to_index(normalized, num_aggregations, decode)

    elif interpretation == 'variance':
        # High variance -> max (winner-take-all), low -> mean (equal contribution)
        weight_var = jnp.var(weights, axis=1)  # (pop_size, num_nodes)
        normalized = jnp.tanh(weight_var * 5)
        return continuous_to_index(normalized, num_aggregations, decode)

    elif interpretation == 'sign_uniformity':
        # All same sign -> product (AND-gate), mixed -> sum (preserve cancellation)
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

    # mean = index 1 in AGGREGATION_LIST -> divide by num_inputs
    mean_idx = 1  # AGGREGATION_LIST.index('mean')
    factors = jnp.where(agg_indices == mean_idx, 1.0 / num_inputs, factors)

    return factors


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


# H->H Aggregation Helper Functions (Segment Operations)

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
    """Aggregate H->H contributions with per-node aggregation function.

    This function replaces simple scatter_add with per-node aggregation,
    enabling different aggregation modes (sum, mean, max, min, etc.) for
    different hidden nodes in H->H connections.

    Args:
        contributions: H->H contributions, shape (n_samples, max_sparse_conns)
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
