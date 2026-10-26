"""Configuration vocabulary: unknown-key refusal, the recurrence settings (`EMRConfig`) and their presets,
the forward-pass modes and the sparse hidden-to-hidden connection record."""
import dataclasses
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np


#: Fitness modes the single-task evaluation computes (emr_hyperneat.fitness_mode, _eval_single_fitness).
#: create_config raises on any other name, so an unknown mode is never scored as MSE.
FITNESS_MODES = ('mse', 'accuracy', 'hybrid', 'bce', 'soft_accuracy', 'ce', 'acc_mse', 'accuracy_confident')


#: How a recurrent substrate settles (recurrence.settling_rule). 'emr', the default:
#: :func:`settle_hidden`. 'hmr': :func:`settle_hidden_hmr`, the rule of HMR's dynamic-functions
#: variants, which the cross-problem control (GECCO Table 3), the per-node paper's recurrent tables
#: and the PPSN topology check ran; an opt-in for runs that reproduce them.
SETTLING_RULES = ('emr', 'hmr')

#: recurrence.network_model. 'grid', the default: EMR's multi-resolution grid. 'node_list': HMR's
#: node-list recurrent model (node_list.py), whose networks live on the leaves of a complete
#: quadtree.
NETWORK_MODELS = ('grid', 'node_list')


#: What refreshes the h->h cache (recurrence.hh_cache_trigger): the interval or a mask change, or
#: the interval only (HMR's unified model).
HH_CACHE_TRIGGERS = ('time_or_mask', 'time')


#: Why a key is not a setting, for the keys configurations are known to use. Every other unknown key
#: gets the list of the block's settings.
_UNKNOWN_KEY_HINTS: Dict[Tuple[str, str], str] = {
    ('dynamic_functions', 'aggregation'): (
        "aggregation is its own block next to dynamic_functions in emr_hyperneat (HMR's "
        "aggregation variant read it nested); nested, it was ignored and the run used sum "
        "aggregation with a one-output CPPN"
    ),
    ('dynamic_functions', 'activation_palette'): (
        "the palette is dynamic_functions.palette, a preset name or a list of ACTIVATION_LIST "
        "indices or names; this key was ignored, so the run had no palette"
    ),
    ('dynamic_functions', 'global_activation'): (
        "the one hidden activation of mode 'global' is dynamic_functions.hidden_activation"
    ),
    ('geometry_seeding', 'use_7d_inputs'): (
        "geometry seeding always queries 7-D rows (x1, y1, x2, y2, delta_x, delta_y, bias) and 5-D "
        "rows without it; no "
        "engine read this key, so use_7d_inputs false ran 7-D inputs"),
    ('dynamic_functions', 'interp_method'): "the setting is dynamic_functions.interpretation",
    ('dynamic_functions', 'weight_interpretation_method'): (
        "the setting is dynamic_functions.interpretation"
    ),
    ('dynamic_functions', 'method'): "the setting is dynamic_functions.interpretation",
    ('dynamic_functions', 'num_outputs'): (
        "the CPPN's output count follows from the configuration; "
        "dynamic_functions.num_cppn_outputs "
        "forces it"
    ),
    (
        'dynamic_functions',
        'global_function',
    ): "the global aggregation is aggregation.global_function",
    ('aggregation', 'global_aggregation'): (
        "the function of aggregation mode 'global' is aggregation.global_function (the misspelled "
        "key used to be ignored, so such a configuration ran 'sum')"
    ),
    (
        'aggregation',
        'aggregation_func',
    ): "the function of aggregation mode 'global' is aggregation.global_function",
    ('emr_hyperneat', 'extended'): (
        "the recurrence settings are emr_hyperneat.recurrence; this block was ignored, so the run "
        "was feed-forward"
    ),
    ('emr_hyperneat', 'strategy'): "the setting is emr_hyperneat.multi_gpu_strategy",
    ('emr_hyperneat', 'num_cppn_outputs'): (
        "the CPPN's output count follows from leo, dynamic_functions, aggregation and "
        "neuromodulation's dedicated outputs; dynamic_functions.num_cppn_outputs forces it"
    ),
    ('emr_hyperneat', 'hmr_hyperneat'): (
        "this is HMR's block name; EMR reads algorithm_params.emrhyperneat.emr_hyperneat"
    ),
    ('emr_hyperneat', 'activate_time'): (
        "the settling steps are recurrence.activate_time or substrate.activate_time"
    ),
    ('emr_hyperneat', 'pop_size'): (
        "the population size is algorithm_params.emrhyperneat.population_size (or neat.pop_size)"
    ),
    (
        'recurrence',
        'recurrence_preset',
    ): "the preset is recurrence.preset (HMR's extended block named it so)",
    (
        'neuromodulation',
        'static_gating',
    ): "a level is chosen with neuromodulation.mode or neuromodulation.preset",
    ('neuromodulation', 'context_derivation'): (
        "no engine ever read it (only tests set it); context gating takes its context from the "
        "current task index, and neuromodulation.context_source accepts only 'task_id'"
    ),
    (
        'neuromodulation',
        'context_gating',
    ): "a level is chosen with neuromodulation.mode or neuromodulation.preset",
    (
        'neuromodulation',
        'modulatory_neurons',
    ): "a level is chosen with neuromodulation.mode or neuromodulation.preset",
    ('neuromodulation', 'true_neuromodulation'): (
        "a level is chosen with neuromodulation.mode or neuromodulation.preset"
    ),
    ('neuromodulation', 'num_nt'): "the setting is neuromodulation.num_nt_types",
    ('recurrence', 'cache_enabled'): "the setting is recurrence.hh_cache_enabled",
    ('recurrence', 'use_cache'): "the setting is recurrence.hh_cache_enabled",
    # The node-list model takes no block of its own, so HMR's connection cap has no key
    ('recurrence', 'node_list'): (
        "the node-list model (recurrence.network_model: node_list) has no settings block: it "
        "applies no connection cap (HMR's cap of 500 kept the first connections in Python set "
        "order)"
    ),
    # The levels outside emr_hyperneat (_refuse_unknown_outer_keys). The section '' is the top level
    # of the configuration.
    ('algorithm_params', 'hmrhyperneat'): (
        "this is the block of the HMR engine (release 0.1.0); EMR reads "
        "algorithm_params.emrhyperneat"
    ),
    ('algorithm_params', 'generations'): (
        "EMR reads no generation count from the configuration; it is run_until_threshold's "
        "max_generations"
    ),
    ('algorithm_params.emrhyperneat', 'hmr_hyperneat'): (
        "this is HMR's block name; EMR reads emr_hyperneat"
    ),
    ('', 'hmr_hyperneat'): "this is HMR's block name; EMR reads emr_hyperneat",
    ('algorithm_params.emrhyperneat', 'pop_size'): (
        "the population size is population_size here, or neat.pop_size"
    ),
    ('', 'pop_size'): "the population size is population_size, or neat.pop_size",
    ('neat', 'species_threshold'): (
        "the speciation threshold is neat_species.compatibility_threshold"
    ),
    # A block of algorithm_params.emrhyperneat one level too high.
    **{
        (section, key): f"EMR reads algorithm_params.emrhyperneat.{key}"
        for section in ('', 'algorithm_params')
        for key in (
            'emr_hyperneat',
            'substrate',
            'neat',
            'population_size',
            'cppn',
            'swarm',
            'neat_mutation',
            'neat_species',
            'neat_selection',
        )
    },
    # NEAT operator settings in neat, where nothing reads them: they sit in three blocks beside it.
    **{
        ('neat', key): f"the setting is {block}.{key} (a block beside neat)"
        for block, keys in (
            (
                'neat_mutation',
                ('conn_add_prob', 'conn_delete_prob', 'node_add_prob', 'node_delete_prob'),
            ),
            ('neat_species', ('compatibility_threshold', 'max_stagnation', 'species_elitism')),
            ('neat_selection', ('genome_elitism', 'survival_threshold')),
        )
        for key in keys
    },
}


def _unknown_keys_message(section: str, block: Any, known: frozenset) -> Optional[str]:
    """What `_refuse_unknown_keys` raises, or None.

    The message names the right key when it is known (``_UNKNOWN_KEY_HINTS``), and the block's
    settings otherwise. The section '' is the top level of the configuration.

    Args:
        section: The block's name, as the message prints it.
        block: The block; anything but a dict is left to the parser.
        known: The keys EMR reads in this block.

    Returns:
        The message for the first unknown key, naming the others, or None when there is none.
    """
    if not isinstance(block, dict):
        return None
    unknown = sorted((k for k in block if k not in known), key=str)
    if not unknown:
        return None
    key = unknown[0]
    where = f"{section}.{key}" if section else key
    hint = _UNKNOWN_KEY_HINTS.get(
        (section, key), f"the settings of {section or 'the top level'} are {sorted(known)}")
    more = f" (also not settings: {unknown[1:]})" if len(unknown) > 1 else ""
    return f"{where} is not a setting; {hint}{more}"


def _refuse_unknown_keys(section: str, block: Any, known: frozenset) -> None:
    """Raise on a key of a configuration block that EMR does not read.

    Ignoring such a key would let a misspelled or misplaced setting run the default without a word.

    Args:
        section: The block's name, as the message prints it.
        block: The block; anything but a dict is left to the parser.
        known: The keys EMR reads in this block.

    Raises:
        ValueError: The block has a key outside ``known``.
    """
    message = _unknown_keys_message(section, block, known)
    if message is not None:
        raise ValueError(message)


# Dataclasses

@dataclass
class SparseHiddenConnections:
    """Sparse representation of hidden->hidden connections for a population.

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

        # Multi-hop settings
        multi_hop_algorithm: Algorithm for multi-hop expansion ("matrix_power" or "fori_loop")
        hop_decay_factor: Weight decay per hop (prevents exploding weights)

        # Caching settings
        hh_cache_enabled: Enable/disable h->h caching
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

    # (There is no use_vectorized_discovery field: nothing read it. Configurations
    # that still carry the key are accepted and the key is ignored; see
    # RECURRENCE_KEYS.)
    max_sparse_conns: int = 10000
    # What discovery does with a genome whose set exceeds max_sparse_conns (SPARSE_CAP_POLICIES):
    # 'refuse' raises SparseCapExceeded, naming the value that holds every set; 'truncate' keeps the
    # genome's first max_sparse_conns pairs in (source, target) position order, as 0.1.0 did, with a
    # SparseCapWarning; 'grow' raises the cap (grown_sparse_cap) and discovers again, so no genome
    # is truncated.
    sparse_cap_policy: str = 'refuse'
    # The most 'grow' may raise max_sparse_conns to (None: no ceiling; the padded h->h arrays take
    # 13 bytes per genome per slot); a set beyond it raises SparseCapExceeded. Refused under the
    # other policies, where nothing grows.
    max_sparse_conns_ceiling: Optional[int] = None

    # Multi-hop settings
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
    # What refreshes the h->h cache: 'time_or_mask' (the default) refreshes after
    # hh_refresh_interval generations or when the variance mask changes by more than
    # hh_mask_change_threshold; 'time' after hh_refresh_interval generations only, as HMR's unified
    # model's cache did
    hh_cache_trigger: str = 'time_or_mask'

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
                "multi_hop_second_factor must be 'masked' or 'raw', got "
                f"{self.multi_hop_second_factor!r}"
            )
        if self.hh_discovery_mode == 'per_genome' and self.multi_hop_second_factor == 'raw':
            raise ValueError(
                "multi_hop_second_factor 'raw' multiplies by the population union's weights and "
                "cannot be combined with hh_discovery_mode 'per_genome'"
            )
        check_sparse_cap_settings(self.max_sparse_conns, self.sparse_cap_policy,
                                  self.max_sparse_conns_ceiling, self.hh_cache_enabled)


#: What discovery may do with a genome whose h->h set exceeds max_sparse_conns (EMRConfig). Defined
#: before RECURRENCE_PRESETS, whose configurations check it as the module loads.
SPARSE_CAP_POLICIES = ('refuse', 'truncate', 'grow')


def check_sparse_cap_settings(max_sparse_conns: Any, policy: Any, ceiling: Any,
                              hh_cache_enabled: bool = False) -> None:
    """Refuse a cap, policy or ceiling discovery could not apply: the rule of `EMRConfig` and of
    `emr_hyperneat.substrate.SubstrateConfig`.

    Args:
        max_sparse_conns: An integer of at least 1 (a bool is refused).
        policy: One of SPARSE_CAP_POLICIES.
        ceiling: None, or under 'grow' an integer of at least ``max_sparse_conns``.
        hh_cache_enabled: The h->h cache keeps each set at the width it was found in, so 'grow'
            refuses it.

    Raises:
        ValueError: On any of the above.
    """
    def count(value: Any) -> bool:
        return isinstance(value, numbers.Integral) and not isinstance(value, bool)

    if not count(max_sparse_conns) or max_sparse_conns < 1:
        raise ValueError(
            f"max_sparse_conns must be an integer of at least 1, got {max_sparse_conns!r}")
    if policy not in SPARSE_CAP_POLICIES:
        raise ValueError(
            f"sparse_cap_policy must be one of {SPARSE_CAP_POLICIES}, got {policy!r}")
    if ceiling is not None:
        if policy != 'grow':
            raise ValueError(
                f"max_sparse_conns_ceiling bounds sparse_cap_policy 'grow' only; under "
                f"{policy!r} nothing grows, so drop it")
        if not count(ceiling) or ceiling < max_sparse_conns:
            raise ValueError(
                f"max_sparse_conns_ceiling must be an integer of at least max_sparse_conns "
                f"({max_sparse_conns}), got {ceiling!r}")
    if policy == 'grow' and hh_cache_enabled:
        raise ValueError(
            "sparse_cap_policy 'grow' cannot be combined with hh_cache_enabled: the cache keeps "
            "each set at the width it was found in; set max_sparse_conns instead")


# Presets

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


# Forward Pass Modes

class ForwardPassMode(Enum):
    """Forward pass execution modes."""
    DENSE_ONLY = "dense_only"       # Feedforward: W1 -> tanh -> W2 (no h->h)
    HYBRID_SPARSE_HH = "hybrid"     # Dense W1/W2 + sparse h->h
    # Not implemented: all-sparse would need sparse W1/W2, and every h->h gate in the generation
    # loop tests `== HYBRID_SPARSE_HH`, so the mode would run with no sparse h->h at all, as
    # DENSE_ONLY under a recurrent name. set_forward_mode refuses it; the member stays so the
    # enum's values do not change. Unreachable from config.
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
    any value is accepted, a count the configuration carries is kept as the attribute, and none is
    used. Only ``allow_hidden_to_hidden``, ``iteration_level`` and ``activate_time`` are read, so another dataclass
    with these fields serves (a caller may pass HMR's UnifiedExtendedConfig).

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
                         + " disagree: both set the settling steps, so give them the same value or "
                           "set only one")
    if given:
        steps = next(iter(given.values()))
    elif config.activate_time is not None:
        steps = int(config.activate_time)
    elif max_depth is not None:
        steps = 2 * (2 ** max_depth + 1)
    else:
        steps = None
    return ForwardPassMode.HYBRID_SPARSE_HH, level, steps
