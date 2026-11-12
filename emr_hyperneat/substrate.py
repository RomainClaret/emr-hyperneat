"""EMR-HyperNEAT's substrate, outside the generation loop.

Before a generation can score a CPPN, EMR-HyperNEAT turns it into a substrate in four
steps: query the CPPN at every position of a fixed multi-resolution grid; keep the
positions where the weight pattern varies, which become the hidden nodes; turn the
weight outputs into input-to-hidden and hidden-to-output matrices; and, in
``cppn_output`` mode, read each hidden node's activation and aggregation function from
extra CPPN outputs at the node's self-connection ``(x, y, x, y, 1)``. This module
exposes those steps, for tools that need the substrate itself: to draw it, to render
what it computes, or to run it on inputs of their own.

The generation loop builds and evaluates its substrate through the same functions
(`substrate_masks`, `substrate_weight_matrices`, `cppn_output_function_indices`,
`dense_substrate_forward` and the query-row builders in the engine modules), so what
this module returns is the substrate EMR evaluates, not a second implementation of it.
The equivalence is tested bit for bit against the generation loop.

Scope: the substrate of the standard execution path, with the activation modes
``disabled``, ``global`` and ``cppn_output`` and the aggregation modes of the same names (an
aggregation palette included), feed-forward or with hidden-to-hidden connections under per-genome
discovery, and true neuromodulation with receptors from each position's first input weight
(`substrate_neuromodulation`), of a feed-forward substrate or of a recurrent one under either
settling rule. The other neuromodulation levels and sources, multi-task evaluation, swarm mode,
the weight-interpretation and random function modes, the union h->h discovery and the h->h cache
are not covered, and `EMRHyperNEAT.substrate_config` refuses a configuration that uses them rather
than return a substrate EMR would not evaluate. `SUBSTRATE_CAPABILITIES` names what is covered
beyond the first release of this API.

Hidden-to-hidden connections are built in two more steps, the caller evaluating the CPPN
again: `hh_query_rows` gives each genome's rows (every pair of its active positions), and
`build_recurrent_substrate` turns the outputs into the genome's connections, with the
generation loop's own functions on the generation loop's shapes, so they are the loop's bit
for bit and do not depend on the other genomes. `substrate_forward` then settles the hidden
layer as the loop does, each genome on its own active positions:

    substrate = build_substrate(*queries.split(raw[None]), cfg)
    hh_rows = hh_query_rows(substrate)                      # one array per genome
    hh_raw = [my_cppn(rows) for rows in hh_rows]           # (count**2, num_cppn_outputs)
    substrate = build_recurrent_substrate(substrate, hh_raw)
    outputs, states = substrate_forward(substrate, inputs, return_states=True)

The recurrent forward matches the generation loop's fitness within float32 rounding, not bit
for bit: the loop evaluates a recurrent population on the union of its active positions. Under true
neuromodulation it composes the loop's own node rule and settling functions, so the same holds. A
genome whose settling is chaotic carries that rounding to order 1; `substrate_forward` says which
evaluations can then differ. A global ``'softmax'`` hidden activation is the softmax of the whole
grid in both (`substrate_forward`).

A feed-forward substrate can also be evaluated as the recurrent forward evaluates its genomes,
compiled and each on its own mask positions (``substrate_forward(..., compiled=True)``): within
float32 rounding of the dense forward, not bit for bit (`substrate_forward` names the exceptions),
at the cost of a genome's own positions rather than the whole grid's, its temporaries bounded by
the input chunk (the outputs grow with the inputs).

Two ways in:

    # A population the model is evolving:
    algo = EMRHyperNEAT()
    config = algo.create_config(params)
    state = algo.initialize(config, problem)
    substrate = algo.population_substrates(state)
    outputs = substrate_forward(substrate, inputs)       # (pop, samples, outputs)

    # CPPNs evaluated by your own code, e.g. a renderer:
    cfg = algo.substrate_config()
    queries = substrate_queries(cfg)
    raw = my_cppn(queries.stacked())                     # (rows, num_cppn_outputs)
    substrate = build_substrate(*queries.split(raw[None]), cfg)

The second form needs an evaluator that computes exactly what the model's CPPN
computes, output transform included: `build_substrate` trusts its inputs.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.emrhyperneat import (
    ACTIVATION_FUNCTIONS,
    AGGREGATION_FUNCTIONS,
    AGGREGATION_LIST,
    MODULATION_MODES,
    NEUROMODULATION_NODE_RULES,
    SETTLING_RULES,
    WEIGHT_THRESHOLD,
    _bucket_size_for,
    _explicit_activate_time,
    _get_extract_sparse_fn,
    _per_genome_pair_outputs,
    _per_genome_rows_for,
    batch_query_population_multi_source_multi_output,
    batch_query_population_self_connections,
    INDEX_DECODES,
    SPARSE_CAP_POLICIES,
    SparseCapExceeded,
    SparseCapWarning,
    continuous_to_index,
    cppn_output_function_indices,
    dense_substrate_forward,
    derive_receptor_from_weight,
    hh_pair_weights,
    hidden_activation,
    input_drive,
    invert_outputs,
    node_activation,
    node_activation_rule,
    per_genome_pair_connections,
    self_connection_rows,
    settle_hidden,
    settle_hidden_additive,
    settle_hidden_drive_once,
    full_grid_softmax,
    softmax_axis_multiplicity,
    softmax_missing_virtual,
    apply_sparse_cap_policy,
    check_sparse_cap_settings,
    sparse_cap_refusal,
    substrate_masks,
    substrate_weight_matrices,
    true_neuromodulation_node_rule,
)
from emr_hyperneat.emrhyperneat_base import (
    batch_query_population_multi_source_chunked,
    compute_hierarchical_variances_batch,
    cppn_query_rows,
    get_hierarchical_grid,
)

__all__ = [
    "ACTIVATION_MODES",
    "AGGREGATION_MODES",
    "HH_CONNECTION_CLASSES",
    "NeuromodulationSettings",
    "SPARSE_CAP_POLICIES",
    "SUBSTRATE_CAPABILITIES",
    "SparseCapExceeded",
    "SparseCapWarning",
    "Substrate",
    "SubstrateConfig",
    "SubstrateNeuromodulation",
    "SubstrateQueries",
    "WEIGHT_THRESHOLD",
    "build_recurrent_substrate",
    "build_substrate",
    "compute_hierarchical_variances_batch",
    "cppn_output_function_indices",
    "cppn_query_rows",
    "default_neurotransmitter",
    "derive_receptor_from_weight",
    "get_hierarchical_grid",
    "hh_connection_classes",
    "hh_pair_weights",
    "hh_query_rows",
    "query_recurrent_substrate",
    "query_substrate",
    "self_connection_rows",
    "substrate_forward",
    "substrate_neuromodulation",
    "substrate_queries",
    "substrate_weight_matrices",
]

# WEIGHT_THRESHOLD is the engine's default pruning threshold: input-to-hidden and hidden-to-output
# weights at or below it are pruned. emr_hyperneat.weight_threshold sets another;
# SubstrateConfig.weight_threshold carries the run's value for building.

#: Activation modes a substrate can be built for.
ACTIVATION_MODES = ("disabled", "global", "cppn_output")
#: Aggregation modes a substrate can be built for.
AGGREGATION_MODES = ("disabled", "global", "cppn_output")
#: The classes of a hidden-to-hidden connection (`Substrate.hh_class` codes index this tuple), by
#: the y coordinates of its ends as in the recurrence configuration: forward when the source lies
#: below the target, lateral on the same y, backward above, self a position to itself.
HH_CONNECTION_CLASSES = ("forward", "lateral", "backward", "self")
#: What the substrate API covers beyond its first release, for a caller that checks before it asks:
#: the connection cap's policies (``SubstrateConfig.sparse_cap_policy``), true neuromodulation of a
#: feed-forward substrate (`substrate_neuromodulation`, ``substrate_forward(..., neuromodulation=)``),
#: an aggregation palette (``SubstrateConfig.aggregation_palette``), the steps the substrate is
#: built from as public functions with stable signatures (`get_hierarchical_grid`,
#: `cppn_query_rows`, `cppn_output_function_indices`, `substrate_weight_matrices`,
#: `self_connection_rows`, `compute_hierarchical_variances_batch`), true neuromodulation of a
#: recurrent substrate (the same calls on a substrate with hidden-to-hidden connections), and the
#: compiled feed-forward forward (``substrate_forward(..., compiled=True, chunk_size=)``, each
#: genome on its own mask positions).
SUBSTRATE_CAPABILITIES = frozenset({
    "sparse_cap_policy",
    "neuromodulation",
    "aggregation_palette",
    "substrate_steps",
    "recurrent_neuromodulation",
    "compiled_forward",
})


class NeuromodulationSettings(NamedTuple):
    """True neuromodulation (level 4) as the generation loop applies it to a substrate,
    feed-forward or recurrent.

    `EMRHyperNEAT.substrate_config` fills it in from the parsed configuration
    (``SubstrateConfig.neuromodulation``). Receptor densities come from each hidden position's
    first input weight, the engine's default source. On a recurrent substrate the node rule applies
    at every settling step, and what the loop refuses there is refused (`SubstrateConfig.validate`),
    as are a strength and a receptor derivation the forward cannot use.

    Attributes:
        num_nt_types: Neurotransmitter channels, 2 to 6; at most 4 under node rule ``'anti'`` on a
            recurrent substrate (the rule reads every channel of the loop's single-task vector).
        modulation_strength: The modulation strength s; on a recurrent substrate, and in the
            compiled feed-forward forward, a number that stays finite in float32, the precision
            the forward reads it at.
        receptor_derivation: How a weight becomes receptor densities
            (`derive_receptor_from_weight`); one of that function's methods on a recurrent
            substrate.
        modulation_mode: ``'full'``, ``'gating_only'`` or ``'gain_bias_only'``
            (``MODULATION_MODES``); only ``'full'`` under node rule ``'anti'`` on a recurrent
            substrate.
        node_rule: The neuromodulated node's rule (``NEUROMODULATION_NODE_RULES``).
        output_inversion: Whether the fourth channel inverts the outputs (four or more channels);
            False under node rule ``'anti'`` on a recurrent substrate (the rule of the prototype's
            ANTI variant inverts no output).
    """

    num_nt_types: int = 4
    modulation_strength: float = 2.0
    receptor_derivation: str = "tanh"
    modulation_mode: str = "full"
    node_rule: str = "paper"
    output_inversion: bool = True


class SubstrateConfig(NamedTuple):
    """Everything that decides how a CPPN becomes a substrate, feed-forward or recurrent.

    Take it from a configured model with `EMRHyperNEAT.substrate_config`, which reads
    EMR's parsed configuration, defaults included. Filling one in by hand is possible
    but loses that guarantee. ``_replace`` gives a variant, e.g. other coordinates.

    Attributes:
        input_coords: Substrate input node positions, ``((x, y), ...)``.
        output_coords: Substrate output node positions.
        max_depth: Deepest grid level; the grid has ``sum(4**(l+1))`` positions.
        variance_threshold: A cell subdivides when its children's variance exceeds it.
        initial_depth: Levels subdivided unconditionally.
        variance_rule: ``'canonical'`` or ``'legacy'`` (the original standard-path level-0 rule).
        max_weight: Weight scale; a weight is ``tanh(cppn) * max_weight``.
        weight_threshold: Weights at or below this magnitude are pruned.
        locality_radius: Geometry-seeding distance penalty radius, or None.
        geometry_seeding: 7-D CPPN inputs with the signed deltas.
        num_cppn_outputs: The CPPN's output count. Channel 0 is always the weight.
        activation_mode: ``'disabled'``, ``'global'`` or ``'cppn_output'``.
        activation_channel: CPPN output carrying the activation choice.
        num_activations: Leading ACTIVATION_LIST entries available without a palette.
        palette: ACTIVATION_LIST indices available, or None.
        hidden_activation: Hidden function when activations are not per node; this is
            the function the generation loop actually applies, not the configured name.
        output_activation: Output function, likewise the effective one.
        aggregation_mode: ``'disabled'``, ``'global'`` or ``'cppn_output'``.
        aggregation_channel: CPPN output carrying the aggregation choice.
        num_aggregations: Leading AGGREGATION_LIST entries available.
        global_aggregation: Aggregation function in ``'global'`` mode.
        true_aggregation: Exact per-node aggregation, or the matmul approximation.
        hidden_to_hidden: The substrate has hidden-to-hidden connections (False: feed-forward,
            whatever the fields below say).
        allow_backward, allow_lateral, allow_self_loops: The connection classes discovery keeps
            (forward connections always).
        band_threshold: A pair is a connection when its weight's magnitude exceeds it.
        iteration_level: Hops composed from the genome's own direct connections (1, the default
            and the engine's: direct only; 2 composes two hops).
        hop_decay_factor: Decay per hop of a composite.
        multi_hop_algorithm: ``'matrix_power'`` or ``'fori_loop'``.
        max_sparse_conns: Connections kept per genome, the first in ascending (source, target).
        sparse_cap_policy: What `build_recurrent_substrate` does with a genome whose set exceeds
            ``max_sparse_conns``: ``'refuse'`` (raise SparseCapExceeded), ``'truncate'`` (keep the
            first pairs) or ``'grow'`` (build at a cap that holds the largest set, which the
            returned substrate's configuration records).
        max_sparse_conns_ceiling: The most ``'grow'`` may reach (None: no ceiling).
        activate_time: Settling steps, an integer of at least 1 (a bool is refused); None is the loop's
            automatic count, ``2 (2**max_depth + 1)``. No effect without hidden-to-hidden connections.
        settling_rule: ``'emr'`` (:func:`settle_hidden`) or ``'drive_once'``
            (:func:`settle_hidden_drive_once`, the prototype engine's rule; under true
            neuromodulation the additive rule of the prototype's FULL variant,
            :func:`settle_hidden_additive`).
        hh_activation_mode: The activation of a settling step, ``'every_iteration'`` or ``'tanh'``,
            or ``'initial_only'`` under the ``'drive_once'`` rule. Under ``'drive_once'`` with true
            neuromodulation it sets the first step's activation only (tanh under ``'tanh'``, else
            the configured one) and every update step applies tanh, so ``'every_iteration'`` is
            refused there unless the activation is plain tanh.
        hh_aggregation_mode: ``'sum'``, or ``'dynamic'`` (per-node aggregation of the recurrent
            input, read from the aggregation channel at the self-connection).
        activation_index_decode: How the activation channel becomes a function index,
            ``'tanh'`` (the default: a palette of 9 or more never reaches its end entries) or
            ``'uniform'`` (every entry, equal bins); `continuous_to_index`.
        aggregation_index_decode: The same for the aggregation channel, including the recurrent
            input's dynamic aggregation.
        aggregation_palette: AGGREGATION_LIST indices the aggregation channel chooses from, any
            subset in any order, or None for the leading ``num_aggregations`` entries.
        neuromodulation: True neuromodulation of the substrate, feed-forward or recurrent
            (`NeuromodulationSettings`), or None.
    """

    input_coords: Tuple[Tuple[float, float], ...]
    output_coords: Tuple[Tuple[float, float], ...]
    max_depth: int
    variance_threshold: float = 0.03
    initial_depth: int = 0
    variance_rule: str = "canonical"
    max_weight: float = 8.0
    weight_threshold: float = WEIGHT_THRESHOLD
    locality_radius: Optional[float] = None
    geometry_seeding: bool = False
    num_cppn_outputs: int = 1
    activation_mode: str = "disabled"
    activation_channel: int = 1
    num_activations: int = 4
    palette: Optional[Tuple[int, ...]] = None
    hidden_activation: str = "tanh"
    output_activation: str = "sigmoid"
    aggregation_mode: str = "disabled"
    aggregation_channel: int = 2
    num_aggregations: int = 4
    global_aggregation: str = "sum"
    true_aggregation: bool = True
    hidden_to_hidden: bool = False
    allow_backward: bool = True
    allow_lateral: bool = True
    allow_self_loops: bool = True
    band_threshold: float = 0.3
    iteration_level: int = 1  # the engine's default (EMRConfig.iteration_level)
    hop_decay_factor: float = 0.8
    multi_hop_algorithm: str = "matrix_power"
    max_sparse_conns: int = 10000
    activate_time: Optional[int] = None
    settling_rule: str = "emr"
    hh_activation_mode: str = "every_iteration"
    hh_aggregation_mode: str = "sum"
    # Appended last, so a configuration written without them reads the tanh decode
    activation_index_decode: str = "tanh"
    aggregation_index_decode: str = "tanh"
    # Appended last: what build_recurrent_substrate does with a genome whose set exceeds
    # max_sparse_conns, and how far 'grow' may go
    sparse_cap_policy: str = "refuse"
    max_sparse_conns_ceiling: Optional[int] = None
    # Appended last: the aggregation palette and true neuromodulation, of a feed-forward or a
    # recurrent substrate
    aggregation_palette: Optional[Tuple[int, ...]] = None
    neuromodulation: Optional[NeuromodulationSettings] = None

    def validate(self) -> "SubstrateConfig":
        """Raise ValueError if the configuration cannot describe an EMR substrate.

        Returns:
            The configuration itself, so calls can chain.
        """
        if not self.input_coords or not self.output_coords:
            raise ValueError("input_coords and output_coords must both be non-empty")
        if self.activation_mode not in ACTIVATION_MODES:
            raise ValueError(f"activation_mode must be one of {ACTIVATION_MODES}, "
                             f"got {self.activation_mode!r}")
        if self.aggregation_mode not in AGGREGATION_MODES:
            raise ValueError(f"aggregation_mode must be one of {AGGREGATION_MODES}, "
                             f"got {self.aggregation_mode!r}")
        if self.variance_rule not in ("canonical", "legacy"):
            raise ValueError(f"variance_rule must be 'canonical' or 'legacy', "
                             f"got {self.variance_rule!r}")
        for name in (self.hidden_activation, self.output_activation):
            if name not in ACTIVATION_FUNCTIONS:
                raise ValueError(f"unknown activation {name!r}")
        if self.global_aggregation not in AGGREGATION_FUNCTIONS:
            raise ValueError(f"unknown aggregation {self.global_aggregation!r}")
        for name, decode in (("activation_index_decode", self.activation_index_decode),
                             ("aggregation_index_decode", self.aggregation_index_decode)):
            if decode not in INDEX_DECODES:
                raise ValueError(f"{name} must be one of {INDEX_DECODES}, got {decode!r}")
        for mode, channel, purpose in (
            (self.activation_mode, self.activation_channel, "activation"),
            (self.aggregation_mode, self.aggregation_channel, "aggregation"),
        ):
            if mode == "cppn_output" and self.num_cppn_outputs > 1 and not (
                0 <= channel < self.num_cppn_outputs
            ):
                raise ValueError(f"{purpose}_channel {channel} is not an output of a "
                                 f"{self.num_cppn_outputs}-output CPPN")
        if self.hidden_to_hidden:
            if self.settling_rule not in SETTLING_RULES:
                raise ValueError(
                    f"settling_rule must be one of {SETTLING_RULES}, got {self.settling_rule!r}"
                )
            modes = ("every_iteration", "tanh") + (
                ("initial_only",) if self.settling_rule == "drive_once" else ()
            )
            if self.hh_activation_mode not in modes:
                raise ValueError(f"hh_activation_mode must be one of {modes} under the "
                                 f"{self.settling_rule!r} rule, got {self.hh_activation_mode!r}")
            if self.hh_aggregation_mode not in ("sum", "dynamic"):
                raise ValueError(f"hh_aggregation_mode must be 'sum' or 'dynamic', got {self.hh_aggregation_mode!r}")
            if self.hh_aggregation_mode == "dynamic" and not (
                    self.aggregation_mode == "cppn_output" and self.num_cppn_outputs > 1):
                raise ValueError("hh_aggregation_mode 'dynamic' reads the aggregation channel at the "
                                 "self-connection: it needs aggregation_mode 'cppn_output' and a "
                                 "multi-output CPPN")
            if self.multi_hop_algorithm not in ("matrix_power", "fori_loop"):
                raise ValueError(f"multi_hop_algorithm must be 'matrix_power' or 'fori_loop', "
                                 f"got {self.multi_hop_algorithm!r}")
            check_sparse_cap_settings(self.max_sparse_conns, self.sparse_cap_policy,
                                      self.max_sparse_conns_ceiling)
            if self.activate_time is not None:
                _explicit_activate_time("activate_time", self.activate_time)
        if self.aggregation_palette is not None:
            entries = tuple(self.aggregation_palette)
            if not entries or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                                  or not 0 <= int(i) < len(AGGREGATION_LIST) for i in entries):
                raise ValueError(f"aggregation_palette must hold AGGREGATION_LIST indices (0 to "
                                 f"{len(AGGREGATION_LIST) - 1}), got {self.aggregation_palette!r}")
        nm = self.neuromodulation
        if nm is not None:
            if not 2 <= int(nm.num_nt_types) <= 6:
                raise ValueError(f"neuromodulation.num_nt_types must be 2 to 6, got "
                                 f"{nm.num_nt_types!r}")
            if nm.modulation_mode not in MODULATION_MODES:
                raise ValueError(f"neuromodulation.modulation_mode must be one of "
                                 f"{MODULATION_MODES}, got {nm.modulation_mode!r}")
            if nm.node_rule not in NEUROMODULATION_NODE_RULES:
                raise ValueError(f"neuromodulation.node_rule must be one of "
                                 f"{NEUROMODULATION_NODE_RULES}, got {nm.node_rule!r}")
            if self.hidden_to_hidden:
                # What the generation loop refuses with true neuromodulation of a recurrent
                # substrate is refused here, and at forward time for a caller-built one
                refusal = _recurrent_neuromodulation_refusal(self, nm)
                if refusal is not None:
                    raise ValueError(refusal)
        return self


#: The methods `derive_receptor_from_weight` knows (its branches; it raises on any other name).
_RECEPTOR_DERIVATIONS = ("abs", "normalized", "tanh", "fourier", "softmax", "orthogonal",
                         "phase_shifted")


def _finite_number(value: Any) -> bool:
    """Whether ``value`` is a real number that stays finite in float32, the precision the recurrent
    forward reads it at: an integer or floating-point scalar (Python, numpy, JAX, or a
    zero-dimensional array), not a bool, a string or a complex number. A finite float64 past
    float32's largest value (about 3.4e38) is refused: the forward would read it as an infinity."""
    if isinstance(value, int) and not isinstance(value, bool):
        # A Python int past int64 is an object array to numpy; the forward reads it as a float
        try:
            value = float(value)
        except OverflowError:
            return False
    try:
        number = np.asarray(value)
    except (TypeError, ValueError):
        return False
    # Neither integer nor floating: numpy's bool, string and complex dtypes (jnp.issubdtype also
    # counts JAX's bfloat16 as floating, which np.issubdtype does not)
    if number.ndim != 0 or not (jnp.issubdtype(number.dtype, jnp.integer)
                                or jnp.issubdtype(number.dtype, jnp.floating)):
        return False
    # The forward casts the strength to float32 (_recurrent_forward), where 1e40 is already inf
    with np.errstate(over="ignore"):
        return bool(np.isfinite(number.astype(np.float32)))


def _recurrent_neuromodulation_refusal(config: SubstrateConfig,
                                       settings: NeuromodulationSettings) -> Optional[str]:
    """Why true neuromodulation with these settings cannot be evaluated on a recurrent substrate,
    or None.

    Two values the forward cannot use, refused on this path (the dense feed-forward forward checks
    neither; the compiled one adds `_checked_neuromodulation`'s check of the receptors' channels and
    refuses the strength too, `_compiled_neuromodulation_refusal`): a modulation strength that is
    not a finite number in float32, the forward's precision (NaN would reach the outputs, and 1e40
    is inf in float32), and a receptor derivation `derive_receptor_from_weight` does not know. Then
    the generation loop's own refusals in SubstrateConfig terms: under the ``'drive_once'`` settling
    rule, the additive rule of the prototype's FULL variant (`settle_hidden_additive`) sums the
    hidden-to-hidden input and updates through tanh, so it refuses dynamic h->h aggregation and an
    ``'every_iteration'`` activation other than plain tanh (`_check_additive_settling`); node rule
    ``'anti'`` (the rule of the prototype's ANTI variant) has the form ``'full'`` only, inverts no
    output, and reads every channel of the single-task vector, which has four. Under the ``'emr'``
    rule nothing else is refused: the loop settles every combination it accepts with the
    neuromodulated node.

    Args:
        config: A recurrent substrate's configuration.
        settings: The neuromodulation settings, configured or a caller-built
            `SubstrateNeuromodulation`'s.

    Returns:
        The refusal's message, or None when the loop evaluates the combination.
    """
    if not _finite_number(settings.modulation_strength):
        return (f"neuromodulation.modulation_strength must be a finite number on a recurrent "
                f"substrate, at most {float(np.finfo(np.float32).max):.3g} in magnitude (the "
                f"forward reads it in float32), got {settings.modulation_strength!r}")
    if not isinstance(settings.receptor_derivation, str) or (
            settings.receptor_derivation not in _RECEPTOR_DERIVATIONS):
        return (f"neuromodulation.receptor_derivation must be one of {_RECEPTOR_DERIVATIONS} "
                f"(derive_receptor_from_weight's methods), got {settings.receptor_derivation!r}")
    if config.settling_rule == "drive_once":
        if config.hh_aggregation_mode == "dynamic":
            return ("settling_rule 'drive_once' with neuromodulation sums the hidden-to-hidden "
                    "input, as the prototype's FULL variant did; hh_aggregation_mode 'dynamic' "
                    "(per-node hidden-to-hidden aggregation) would be ignored")
        # The loop's plain tanh (dynamic_functions 'disabled', or 'global' with tanh): no per-node
        # activation and a tanh hidden activation
        plain_tanh = (config.activation_mode != "cppn_output"
                      and config.hidden_activation == "tanh")
        if config.hh_activation_mode == "every_iteration" and not plain_tanh:
            return ("settling_rule 'drive_once' with neuromodulation applies tanh at every step "
                    "after the first, as the prototype's FULL variant did; set hh_activation_mode "
                    "'initial_only'")
    if settings.node_rule == "anti":
        if settings.modulation_mode != "full":
            return (f"neuromodulation.node_rule 'anti' gates and scales every node (the "
                    f"prototype's ANTI variant had no other form); modulation_mode "
                    f"{settings.modulation_mode!r} is not one of its forms")
        if int(settings.num_nt_types) > 4:
            return (f"neuromodulation.node_rule 'anti' reads every neurotransmitter channel and "
                    f"the loop's single-task vector has four; num_nt_types "
                    f"{settings.num_nt_types} is refused on a recurrent substrate (at most 4)")
        if settings.output_inversion:
            return ("neuromodulation.output_inversion: node_rule 'anti' (the rule of the "
                    "prototype's ANTI variant) inverts no output, and the loop turns the inversion "
                    "off; set output_inversion False")
    return None


class SubstrateQueries(NamedTuple):
    """The CPPN input rows a substrate is built from.

    Row layout ``[x1, y1, x2, y2, 1]`` (7 columns with geometry seeding): see
    `emr_hyperneat.emrhyperneat_base.cppn_query_rows`.

    Attributes:
        input_rows: (num_inputs, total_positions, input_dim), substrate input ->
            grid position.
        output_rows: (num_outputs, total_positions, input_dim), grid position ->
            substrate output.
        self_rows: (total_positions, input_dim), each position to itself.
    """

    input_rows: np.ndarray
    output_rows: np.ndarray
    self_rows: np.ndarray

    def stacked(self) -> np.ndarray:
        """Every row in one ``(rows, input_dim)`` array, for a single evaluator call."""
        dim = self.self_rows.shape[-1]
        return np.concatenate([
            self.input_rows.reshape(-1, dim),
            self.output_rows.reshape(-1, dim),
            self.self_rows,
        ])

    def split(self, outputs: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Cut evaluator outputs for `stacked` rows back into the three blocks.

        Args:
            outputs: ``(pop_size, rows, num_cppn_outputs)``, the CPPN outputs for
                `stacked` rows, one leading entry per genome.

        Returns:
            ``(input_outputs, output_outputs, self_outputs)`` with shapes
            ``(pop, num_inputs, P, C)``, ``(pop, num_outputs, P, C)`` and
            ``(pop, P, C)``, ready for `build_substrate`.
        """
        outputs = jnp.asarray(outputs)
        if outputs.ndim != 3:
            raise ValueError(f"expected (pop_size, rows, num_cppn_outputs), got {outputs.shape}")
        pop = outputs.shape[0]
        num_in, positions = self.input_rows.shape[:2]
        num_out = self.output_rows.shape[0]
        a = num_in * positions
        b = a + num_out * positions
        if outputs.shape[1] != b + positions:
            raise ValueError(f"expected {b + positions} rows, got {outputs.shape[1]}")
        channels = outputs.shape[2]
        return (
            outputs[:, :a].reshape(pop, num_in, positions, channels),
            outputs[:, a:b].reshape(pop, num_out, positions, channels),
            outputs[:, b:],
        )


class Substrate(NamedTuple):
    """A population of substrates, as the generation loop evaluates them.

    Arrays keep every grid position; a position the mask drops has zero weights in and
    out. Indices are None when the corresponding mode is not ``cppn_output``. The ``hh_``
    fields are None until `build_recurrent_substrate` fills them (and stay None for a
    feed-forward configuration); their per-connection arrays have the generation loop's
    layout: ``max_sparse_conns`` slots per genome, the connections first in ascending
    (source, target) order, then padding (``-1`` indices, weight 0, not valid).

    Attributes:
        positions: (total_positions, 2) grid positions, coarsest level first.
        levels: (total_positions,) grid level of each position.
        mask: (pop_size, total_positions) bool, the hidden nodes each substrate keeps:
            every position its subdivision reached, at every level. The recurrent forward evaluates
            only these positions: a position outside the mask loses its W1 and W2 columns and every
            hidden-to-hidden connection touching it, so clearing a mask entry knocks the node out.
            Under a global ``'softmax'`` hidden activation, which normalizes over every grid
            position (`full_grid_softmax`), a position outside the mask still enters the softmax's
            sum as a node with zero weights in and out, under true neuromodulation with the
            receptors and base gain given there.
            A genome whose mask is cleared entirely has no hidden node: no hidden-to-hidden rows or
            connections, settling states of width 0, and outputs equal to the output activation of
            zero, as the feed-forward forward gives with every column zeroed.
            The feed-forward forward reads W1 and W2 as given (a built substrate's columns outside the
            mask are zero), so there a node is removed by zeroing its columns; its compiled form
            (``substrate_forward(..., compiled=True)``) evaluates only the mask's positions, as the
            recurrent forward does.
        W1: (pop_size, num_inputs, total_positions) input -> hidden weights.
        W2: (pop_size, total_positions, num_outputs) hidden -> output weights.
        act_indices: (pop_size, total_positions) ACTIVATION_LIST indices, or None.
        agg_indices: (pop_size, total_positions) AGGREGATION_LIST indices, or None.
        config: The configuration the substrates were built with.
        hh_from, hh_to: (pop_size, max_sparse_conns) source and target grid positions.
        hh_weights: (pop_size, max_sparse_conns) weights.
        hh_valid: (pop_size, max_sparse_conns) which slots hold a connection.
        hh_class: (pop_size, max_sparse_conns) int8 codes into HH_CONNECTION_CLASSES, -1 on padding.
        hh_direct: (pop_size, max_sparse_conns) True for a direct connection at its own weight,
            False for a composite of several hops and for a direct pair a composite of larger
            magnitude replaced (the expansion keeps the larger magnitude).
        hh_uncapped: (pop_size,) connections before the ``max_sparse_conns`` cut; more than
            ``hh_valid`` holds means the cap removed the last ones in (source, target) order.
        hh_agg_indices: (pop_size, total_positions) AGGREGATION_LIST indices of each node's
            recurrent-input aggregation (``hh_aggregation_mode='dynamic'``), or None.
    """

    positions: jnp.ndarray
    levels: np.ndarray
    mask: jnp.ndarray
    W1: jnp.ndarray
    W2: jnp.ndarray
    act_indices: Optional[jnp.ndarray]
    agg_indices: Optional[jnp.ndarray]
    config: SubstrateConfig
    hh_from: Optional[jnp.ndarray] = None
    hh_to: Optional[jnp.ndarray] = None
    hh_weights: Optional[jnp.ndarray] = None
    hh_valid: Optional[jnp.ndarray] = None
    hh_class: Optional[np.ndarray] = None
    hh_direct: Optional[np.ndarray] = None
    hh_uncapped: Optional[np.ndarray] = None
    hh_agg_indices: Optional[jnp.ndarray] = None


def _coords(coords: Sequence[Sequence[float]]) -> np.ndarray:
    """Coordinates as float32, the precision the generation loop queries at."""
    return np.asarray(coords, dtype=np.float32).reshape(-1, 2)


def _levels(grid: Any) -> np.ndarray:
    offsets = grid.level_offsets_static
    return np.concatenate([
        np.full(offsets[lv + 1] - offsets[lv], lv, dtype=np.int32)
        for lv in range(grid.num_levels)
    ])


def _needs_self_connection(config: SubstrateConfig) -> bool:
    """Whether a function index is read from a dedicated CPPN output."""
    return config.num_cppn_outputs > 1 and (
        config.activation_mode == "cppn_output" or config.aggregation_mode == "cppn_output"
    )


def substrate_queries(config: SubstrateConfig) -> SubstrateQueries:
    """The rows at which to evaluate a CPPN to build its substrate.

    Args:
        config: Substrate configuration.

    Returns:
        The input->grid, grid->output and self-connection rows, built by the same
        functions the generation loop's query helpers use.
    """
    config.validate()
    grid = get_hierarchical_grid(config.max_depth)
    positions = grid.all_positions
    input_rows = np.stack([
        cppn_query_rows(src, positions, True, config.geometry_seeding)
        for src in _coords(config.input_coords)
    ])
    output_rows = np.stack([
        cppn_query_rows(dst, positions, False, config.geometry_seeding)
        for dst in _coords(config.output_coords)
    ])
    self_rows = np.asarray(self_connection_rows(positions, config.geometry_seeding))
    return SubstrateQueries(input_rows, output_rows, self_rows)


def query_substrate(
    cppn_forward: Any,
    state: Any,
    cppns_transformed: Tuple,
    config: SubstrateConfig,
    device_id: int = 0,
) -> Tuple[Any, Any, Optional[jnp.ndarray]]:
    """Evaluate a population of TensorNEAT CPPNs with the generation loop's query helpers.

    Args:
        cppn_forward: The genome forward, ``cppn_forward(state, transformed, x)``;
            the model's is ``EMRHyperNEAT._jitted_cppn_forward``.
        state: TensorNEAT state.
        cppns_transformed: Transformed population, a tuple of four arrays with a
            leading population axis.
        config: Substrate configuration.
        device_id: Device to run on.

    Returns:
        ``(input_outputs, output_outputs, self_outputs)`` for `build_substrate`;
        ``self_outputs`` is None when no function index comes from a CPPN output.
    """
    config.validate()
    grid = get_hierarchical_grid(config.max_depth)
    positions = grid.all_positions
    input_coords = jnp.array(config.input_coords, dtype=jnp.float32)
    output_coords = jnp.array(config.output_coords, dtype=jnp.float32)
    if config.num_cppn_outputs > 1:
        query = batch_query_population_multi_source_multi_output
    else:
        query = batch_query_population_multi_source_chunked
    input_outputs = query(
        state, cppns_transformed, input_coords, positions, True, cppn_forward,
        device_id=device_id, geometry_seeding_enabled=config.geometry_seeding,
    )
    output_outputs = query(
        state, cppns_transformed, output_coords, positions, False, cppn_forward,
        device_id=device_id, geometry_seeding_enabled=config.geometry_seeding,
    )
    self_outputs = None
    if _needs_self_connection(config):
        self_outputs = batch_query_population_self_connections(
            state, cppns_transformed, positions, cppn_forward,
            num_cppn_outputs=config.num_cppn_outputs,
            geometry_seeding_enabled=config.geometry_seeding,
            device_id=device_id,
        )
    return input_outputs, output_outputs, self_outputs


def build_substrate(
    input_outputs: Any,
    output_outputs: Any,
    self_outputs: Optional[Any],
    config: SubstrateConfig,
) -> Substrate:
    """Build substrates from CPPN outputs at the `substrate_queries` rows.

    Args:
        input_outputs: CPPN outputs at ``input_rows``, ``(pop, num_inputs, P)`` for a
            one-output CPPN or ``(pop, num_inputs, P, num_cppn_outputs)``.
        output_outputs: CPPN outputs at ``output_rows``, same convention.
        self_outputs: ``(pop, P, num_cppn_outputs)`` outputs at ``self_rows``; needed
            only when a function index comes from a dedicated CPPN output.
        config: Substrate configuration.

    Returns:
        The substrates, one per leading entry.
    """
    config.validate()
    grid = get_hierarchical_grid(config.max_depth)
    input_weights = input_outputs[..., 0] if np.ndim(input_outputs) == 4 else input_outputs
    output_weights = output_outputs[..., 0] if np.ndim(output_outputs) == 4 else output_outputs
    if _needs_self_connection(config) and self_outputs is None:
        raise ValueError("this configuration reads function indices from the "
                         "self-connection query; pass self_outputs")
    if not _needs_self_connection(config):
        self_outputs = None

    masks = substrate_masks(
        input_weights[:, 0, :], grid, config.variance_threshold,
        initial_depth=config.initial_depth,
        legacy_level0_variance=config.variance_rule == "legacy",
    )
    W1, W2 = substrate_weight_matrices(
        input_weights, output_weights, masks, config.max_weight,
        weight_threshold=config.weight_threshold,
        locality_radius=config.locality_radius,
        input_coords=jnp.array(config.input_coords, dtype=jnp.float32),
        output_coords=jnp.array(config.output_coords, dtype=jnp.float32),
        positions=grid.all_positions,
    )

    act_indices = None
    if config.activation_mode == "cppn_output":
        palette = _palette(config)
        act_indices = cppn_output_function_indices(
            self_outputs, masks, W1,
            channel=config.activation_channel,
            num_options=len(config.palette) if palette is not None else config.num_activations,
            num_cppn_outputs=config.num_cppn_outputs,
            palette=palette,
            decode=config.activation_index_decode,
        )
    agg_indices = None
    agg_palette = _aggregation_palette(config)
    agg_options = (len(config.aggregation_palette) if agg_palette is not None
                   else config.num_aggregations)
    if config.aggregation_mode == "cppn_output":
        agg_indices = cppn_output_function_indices(
            self_outputs, masks, W1,
            channel=config.aggregation_channel,
            num_options=agg_options,
            num_cppn_outputs=config.num_cppn_outputs,
            palette=agg_palette,
            decode=config.aggregation_index_decode,
        )
    hh_agg_indices = None
    if config.hidden_to_hidden and config.hh_aggregation_mode == "dynamic":
        # The generation loop's decode of the recurrent-input aggregation: the aggregation channel
        # at the self-connection, unmasked (validate() guarantees the channel exists), a palette
        # position mapped to its AGGREGATION_LIST index as the loop maps it.
        hh_agg_indices = continuous_to_index(
            jnp.asarray(self_outputs)[:, :, config.aggregation_channel], agg_options,
            config.aggregation_index_decode)
        if agg_palette is not None:
            hh_agg_indices = agg_palette[hh_agg_indices]
    return Substrate(
        positions=grid.all_positions,
        levels=_levels(grid),
        mask=masks,
        W1=W1,
        W2=W2,
        act_indices=act_indices,
        agg_indices=agg_indices,
        config=config,
        hh_agg_indices=hh_agg_indices,
    )


def _aggregation_palette(config: SubstrateConfig) -> Optional[jnp.ndarray]:
    if config.aggregation_palette is None:
        return None
    return jnp.array(config.aggregation_palette, dtype=jnp.int32)


def _evaluated_aggregations(config: SubstrateConfig) -> int:
    """How many aggregations evaluation dispatches over: the whole list under a palette (its entries
    are AGGREGATION_LIST indices), else the leading ``num_aggregations``, as in the generation loop."""
    if config.aggregation_palette is not None:
        return len(AGGREGATION_LIST)
    return int(config.num_aggregations)


def _palette(config: SubstrateConfig) -> Optional[jnp.ndarray]:
    if config.palette is None:
        return None
    return jnp.array(config.palette, dtype=jnp.int32)


def substrate_forward(substrate: Substrate, inputs: Any, return_states: bool = False, *,
                      neuromodulation: Optional["SubstrateNeuromodulation"] = None,
                      compiled: bool = False, chunk_size: Optional[int] = None) -> Any:
    """Run every substrate on the same inputs.

    A feed-forward substrate runs on the whole grid, as the generation loop does (bit for bit).
    A recurrent one settles its hidden layer for ``activate_time`` steps, each genome on its own
    active positions (padded to a bucket size, so compiled shapes repeat), which matches the
    loop's fitness within float32 rounding. A global ``'softmax'`` hidden activation normalizes
    across the hidden positions, and its definition is the whole grid's at the configured depth,
    as the feed-forward forward evaluates it. On a genome's own positions (and in the loop, on a
    compacted generation's union) each grid position left out has zero weights in and out, so it
    adds ``exp(c - m)`` to the denominator, ``c`` being the input such a position's softmax
    receives at that step and ``m`` the largest input; the bucket's padding adds nothing, and the
    outputs equal the whole grid's within float32 rounding (`full_grid_softmax`).

    ``compiled=True`` evaluates a feed-forward substrate as the recurrent forward evaluates its
    genomes: each genome on its own mask positions in ascending order, its W1 and W2 columns and
    rows, function indices and, under neuromodulation, receptors and base gains gathered there and
    zero-padded to the recurrent forward's power-of-two bucket (8, 16, 32, ...), the genomes of a
    bucket in fixed slots of one jitted call, the inputs in chunks of ``chunk_size`` rows rounded
    up to a multiple of 8 (the last one padded, so a bucket size has one compiled shape, and whole
    groups of eight rows, so a row's outputs do not depend on its place in the chunk under
    single-threaded Eigen; under XLA's default multi-threaded Eigen a Xeon E5-2683 v4 still put 28
    of 12,780 three-output values up to 1.2e-7 apart by their place in the chunk). Its operands and
    outputs are float32, as in the recurrent forward; under JAX_ENABLE_X64 an intermediate can still
    promote to float64 inside the program (exact per-node aggregation's accumulator among them), and
    the dense forward keeps float64 weights. It computes the dense forward's function on the mask's
    positions, so on a substrate as built (zero outside the mask) it equals the dense forward within
    float32 rounding, not bit for bit: the hidden-to-output sum runs over the genome's positions
    instead of the grid's, in another order, and XLA fuses the compiled program's operations (a
    multiply and an add rounded once, where the dense forward, op by op, rounds twice: exact and
    global sum and mean aggregation, the paper rule's gains * pre + bias, the gain bg + s m at a
    strength s that is not a power of two, among others). The difference grows with the magnitudes
    of the terms a sum adds, not with the sum, so product aggregation and other strengths and
    vectors under neuromodulation can widen it far past its size at the settings of the measured
    populations (the aggregations sum, mean, max and min; under neuromodulation, strength 2); the
    tests also cover product aggregation and, under neuromodulation, strength 5, within their bound
    (16 to 64 ulps of max(1, a genome's largest |h @ W2|), by output function; an ulp, a unit in the
    last place, is the gap between adjacent float32 values). Neither forward is the more accurate.
    That rounding can decide an overflow: a value within the two forwards' rounding difference of a
    float32 overflow threshold can overflow in one forward only, namely a softplus output whose
    argument, h @ W2, the two forwards sum to either side of ln(FLT_MAX) = 88.72 (FLT_MAX: float32's
    largest finite value, about 3.4e38; on any substrate, without neuromodulation, inputs in
    [-1, 1]), and a softplus node whose pre-activation comes from a fused step; its genome's outputs
    then differ in finiteness or by up to the output function's range. For finite inputs, a finite
    strength and vector, and s m finite at every position, away from such a threshold, NaN and inf
    fall where the dense forward has them. An infinite input makes every dense output NaN (0 * inf
    at the zero weights outside the mask), while a compiled genome with no zero weight from that
    input, at its positions or in its padding, can stay finite; a strength and a vector, each
    finite, whose product s m overflows at the positions outside a genome's mask can put NaN in the
    dense forward only. A strength, or an entry of the vector the node rule or the inversion reads,
    that is not finite in float32 is refused: the zero padding is inert only at a finite strength
    and finite entries m reads (at a padded position s m = inf * 0 is NaN, which the zero W2 rows
    carry to every output of the genome, while the dense forward's positions outside the mask
    follow another rule), and the inversion's entry, which acts after the output function in both
    forwards alike, is refused with them for one rule over the entries read. Two chunk sizes are two
    programs, which can round the hidden-to-output product apart (with three outputs, chunks of 512
    rows or more gave the default's bits at buckets up to 512, and of 3,000 or more at bucket 2,048;
    smaller ones did not always). A position its mask clears is dropped with its weights, where the
    dense forward still sums whatever weights it keeps. It costs a genome its own positions rather
    than the whole grid; its temporaries are bounded by ``chunk_size``, while the outputs grow with
    the inputs. Each bucket size compiles once per process for a combination of the settings it
    reads, the chunk size and the operand shapes (the numbers of inputs and outputs, the receptor
    channels, the vector's length), while the function cache keeps it. The cache holds 64 entries,
    one per combination of those settings, each a jitted function that keeps one program per
    operand shapes (the bucket size and the chunk rows among them), and evicts the least recently
    used: a workload that cycles through more than 64 combinations of the settings recompiles them
    on every pass, and bucket and chunk sizes never evict one another. A caller warms the buckets
    it expects with one call per bucket size and per output count it serves (two per bucket size
    for a caller that serves one and three outputs). A global ``'softmax'`` hidden activation,
    which normalizes over the grid's positions, is refused (the dense forward evaluates it on the
    whole grid). On a recurrent substrate ``compiled`` changes nothing: that forward is compiled per
    genome already.
    docs/architecture.md has the accuracy as measured.

    Under true neuromodulation each hidden node follows the generation loop's neuromodulated node
    rule and the outputs its inversion: with the substrate's own receptors and gains
    (`substrate_neuromodulation`) when its configuration has neuromodulation settings, or with
    ``neuromodulation`` when given, on any substrate. Feed-forward, bit for bit. Recurrent, the rule
    applies at every settling step as in the loop (`settle_hidden` under the ``'emr'`` rule; under
    ``'drive_once'``, the additive rule of the prototype's FULL variant, `settle_hidden_additive`:
    the node of the input drive, then ``activate_time`` update steps), with each genome's receptors
    and gains at its own active positions, and matches the loop's fitness within float32 rounding;
    the neurotransmitter vector and the strength are traced operands, so changing their values
    compiles nothing new (a vector of another length compiles anew). On the recurrent path under
    neuromodulation the configuration goes through `SubstrateConfig.validate` first, so one built
    by ``_replace`` or field by field is refused as validate refuses it (validate only reads it, so
    a valid configuration evaluates as it would without the call), and a given
    ``neuromodulation``'s own settings are held to the same checks, with validate's messages.

    Rounding. A genome whose settling is chaotic carries float32 rounding to order 1, so two
    evaluations of it that differ only in rounding can give outputs that differ by order 1: eager
    against compiled, two compiled programs, two processors, two groupings of the input rows, or
    two settings of XLA's ``--xla_cpu_multi_thread_eigen`` on one processor (on an Apple M4 Max up
    to 1.0 with the 18-function palette under ``'steep_gate'`` on XOR's four rows). One compiled
    program on one CPU, under one XLA_FLAGS setting, is deterministic. On a GPU a scatter-add
    accumulates in no fixed order, so one program can differ from itself where the settling uses
    one: dynamic h->h aggregation sums each node's recurrent input with it, and the
    ``'drive_once'`` rule scatter-adds its recurrent input at every step whatever the aggregation.
    On an NVIDIA T4 and an NVIDIA L4 (jax 0.6.1, JAX_DEFAULT_MATMUL_PRECISION=highest) one program
    of the recurrent forward under ``'drive_once'`` with neuromodulation did not repeat bit for bit
    (outputs up to 1.74e-4 apart on the L4). Two chunk sizes of the compiled feed-forward forward,
    two programs, were up to 1.45e-4 apart on the T4 and 8.4e-5 on the L4, on outputs of up to 158
    (their value on the CPU). On a GPU from Ampere on, the float32-rounding statements here assume
    JAX_DEFAULT_MATMUL_PRECISION=highest (JAX's default matmul precision there is TF32,
    TensorFloat-32: float32's range with a 10-bit mantissa). Through the same rounding the results
    depend on how many input rows one call evaluates. One-row calls against the same rows in one
    call of 576 (six genomes at depth 3, an Apple M4 Max under single-threaded Eigen) gave outputs
    2.4e-7 apart without neuromodulation and 2.8e-4 apart under ``'drive_once'`` with
    neuromodulation; with the 18-function palette under node rule ``'steep_gate'``, outputs 6.3e-7
    apart, but the states of two genomes whose states grow to 3.6e3 and 4.6e5 up to 38% and 16% of
    that apart, and the outputs of the program that also emits the states up to 1.0 apart. A caller
    comparing two evaluations groups the rows the same way.

    Args:
        substrate: Output of `build_substrate`, and of `build_recurrent_substrate` when the
            configuration has hidden-to-hidden connections.
        inputs: (n_samples, num_inputs).
        return_states: Also return the hidden state after each settling step (recurrent only).
            The states take activate_time x n_samples x count float32 values per genome, all held
            at once (34 steps x 16,384 rows x 444 positions is about 1 GB); the recurrent forward
            does not chunk its inputs, so a caller bounding that memory evaluates the rows in
            groups itself, which differs from one call only by rounding (the row count is part of
            the compiled program), carried to order 1 in a genome whose settling is chaotic (above).
        neuromodulation: A `SubstrateNeuromodulation` to apply instead of the configured one.
        compiled: Evaluate a feed-forward substrate genome by genome on its mask positions, compiled
            (above); False is the dense forward.
        chunk_size: Input rows per call of the compiled feed-forward forward, a positive integer,
            rounded up to a multiple of 8; None is 4,096. Every call takes a whole chunk (the last
            one zero-padded) and each chunk size compiles its own programs, so a call site keeps one
            chunk size (the next power of two of its usual row count, for example) rather than
            passing each call's row count. A call's temporaries are up to 3 times max(128, bucket) *
            chunk_size float32 values, by configuration and output count (XLA's own accounting: 2
            with four exact aggregations and three outputs, 4 MiB up to bucket 128 at 4,096 rows and
            64 MiB at bucket 2,048; 3 with six evaluated aggregations and three outputs, 96 MiB
            there), and nothing caps them: chunk_size=65,536 at bucket 2,048 holds up to 1.5 GiB a
            call (1 GiB with four exact aggregations). Refused without ``compiled=True`` and on a
            recurrent substrate.

    Returns:
        (pop_size, n_samples, num_outputs); with ``return_states`` the pair ``(outputs,
        states)``, states a list with one array per genome, (activate_time, n_samples, count)
        over the genome's own active positions in ascending order (the order of
        `hh_query_rows`, so the substrate's mask maps it back to the grid). Under neuromodulation
        and the ``'drive_once'`` rule they are the states after the ``activate_time`` update steps;
        the node of the input drive that starts them is left out, so the shape is the same under
        both rules. Under neuromodulation of a recurrent substrate the outputs come from the
        compiled program of the call without ``return_states`` and the states from a second
        program that also emits them, so on the CPU the outputs are that call's bit for bit (the
        second program rounds one settling step apart from the first with the 18-function palette
        under ``'steep_gate'``, which a chaotic genome carried from fitness 0.750 to 0.504).
        ``states[-1]`` through W2, the output activation and the inversion then reproduces the
        outputs, within rounding, for a genome that one ulp of its weights does not move; for a
        chaotic genome the two programs can differ. Without neuromodulation the outputs come from
        the program that returns the states. The compiled feed-forward forward returns float32
        outputs of the same shape, in the genomes' order.

    Raises:
        ValueError: On a feed-forward substrate: ``return_states``; ``chunk_size`` without
            ``compiled=True``, or not a positive integer; with ``compiled=True``, a global
            ``'softmax'`` hidden activation, inputs that are not ``(n_samples, num_inputs)``, or
            under neuromodulation a strength, or an entry of the vector the node rule or the
            inversion reads, that is not finite in float32. Under neuromodulation there, an unknown
            modulation mode or node rule, receptors or base gains that do not cover the substrate's
            genomes and positions, a neurotransmitter vector shorter than the channels the node
            rule and the inversion read, and, with the configuration's settings, a channel count
            outside 2 to 6 or an unknown receptor derivation (`derive_receptor_from_weight`); in
            the compiled forward, also receptors without the channels the node rule reads. On a
            recurrent substrate: ``chunk_size``; hidden-to-hidden connections not built; under
            neuromodulation, what `SubstrateConfig.validate` refuses in its configuration and in
            the settings of a given ``neuromodulation`` (a channel count outside 2 to 6, an unknown
            modulation mode, node rule or receptor derivation, a strength that is not a finite
            number in float32, and what the generation loop refuses there), receptors or base gains
            that do not cover the substrate's genomes and positions, receptors without the channels
            the node rule reads, and a neurotransmitter vector shorter than the channels the node
            rule and the inversion read.
    """
    config = substrate.config
    if config.hidden_to_hidden:
        if chunk_size is not None:
            raise ValueError("chunk_size applies to the compiled feed-forward forward; a recurrent "
                             "substrate's forward takes its inputs whole (compiled=True changes "
                             "nothing there)")
        if substrate.hh_from is None:
            raise ValueError("this substrate has hidden-to-hidden connections to settle: build them with "
                             "build_recurrent_substrate first")
        if neuromodulation is not None or config.neuromodulation is not None:
            # True neuromodulation of a recurrent substrate: a configuration that bypassed validate
            # is refused here as everywhere else (without the call, activate_time 0 would settle
            # under 'drive_once', a settling_rule typo would run 'emr', and 'initial_only' under
            # 'emr' would settle through tanh). The other paths do not validate: the plain recurrent
            # forward and the dense feed-forward forward run the configuration as given, and the
            # compiled feed-forward forward adds to the dense one's checks
            # `_checked_neuromodulation`'s check of the receptors' channels and
            # `_compiled_neuromodulation_refusal`'s of the strength and the vector.
            config.validate()
        if neuromodulation is None and config.neuromodulation is not None:
            neuromodulation = substrate_neuromodulation(substrate)
        return _recurrent_forward(substrate, inputs, return_states, neuromodulation)
    if return_states:
        raise ValueError("return_states needs a recurrent substrate: a feed-forward one does not "
                         "settle")
    if compiled:
        return _compiled_feedforward(substrate, inputs, neuromodulation, chunk_size)
    if chunk_size is not None:
        raise ValueError("chunk_size applies to the compiled feed-forward forward: pass "
                         "compiled=True (the dense forward evaluates the inputs whole)")
    if neuromodulation is None and config.neuromodulation is not None:
        neuromodulation = substrate_neuromodulation(substrate)
    if neuromodulation is not None:
        return _neuromodulated_forward(substrate, inputs, neuromodulation)
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    per_node_aggregation = substrate.agg_indices is not None
    agg_indices = (substrate.agg_indices if per_node_aggregation
                   else jnp.zeros(substrate.mask.shape, dtype=jnp.int32))
    global_agg_fn = (AGGREGATION_FUNCTIONS[config.global_aggregation]
                     if config.aggregation_mode == "global" else None)
    palette = _palette(config)
    hidden_act_fn = ACTIVATION_FUNCTIONS[config.hidden_activation]
    output_act_fn = ACTIVATION_FUNCTIONS[config.output_activation]

    def one(W1, W2, act_idx, agg_idx):
        return dense_substrate_forward(
            inputs, W1, W2, act_idx, agg_idx,
            hidden_act_fn=hidden_act_fn,
            output_act_fn=output_act_fn,
            palette=palette,
            num_activations=config.num_activations,
            global_agg_fn=global_agg_fn,
            per_node_aggregation=per_node_aggregation,
            true_aggregation=config.true_aggregation,
            num_aggregations=_evaluated_aggregations(config),
        )

    return jax.vmap(one)(substrate.W1, substrate.W2, substrate.act_indices, agg_indices)


# True neuromodulation of a substrate, feed-forward or recurrent


class SubstrateNeuromodulation(NamedTuple):
    """What each hidden node of a substrate reads besides its drive under true neuromodulation, and
    how it reads it.

    `substrate_neuromodulation` computes it as the generation loop does. A caller may build one
    itself, for example with receptor densities from its own summary of a node's weights through
    `derive_receptor_from_weight`; `substrate_forward` applies the generation loop's node rule to
    whatever it is given. The arrays cover every grid position; on a recurrent substrate and in the
    compiled feed-forward forward (``substrate_forward(..., compiled=True)``) each genome reads them
    at its own active positions, so a position its mask clears loses its receptors and gain with
    its connections. Under a global ``'softmax'`` hidden activation the recurrent forward reads
    them outside the mask too: the softmax counts every grid position (`full_grid_softmax`), and a
    position the mask clears enters it as a zero-weight node, whose softmax input is the node
    rule's value at a zero drive under the receptors and gain given there.

    Attributes:
        receptors: (pop_size, num_positions, num_nt_types) receptor densities in [-1, 1].
        base_gains: (pop_size, num_positions) base gains.
        neurotransmitter: The neurotransmitter vector every node reads.
        settings: The `NeuromodulationSettings` the node rule follows.
    """

    receptors: Any
    base_gains: Any
    neurotransmitter: Any
    settings: NeuromodulationSettings


def default_neurotransmitter(num_nt_types: int) -> jnp.ndarray:
    """The generation loop's neurotransmitter vector for a single-task run: 0.5 on the modulating
    channels and 1.0 on the inverting one (``[0.5, 0.5, 0.5, 1.0]`` from four channels on)."""
    return jnp.array([0.5] * min(3, num_nt_types) + [1.0] if num_nt_types >= 4
                     else [0.5] * num_nt_types)


def substrate_neuromodulation(substrate: Substrate,
                              neurotransmitter: Any = None) -> SubstrateNeuromodulation:
    """The receptor densities and base gains of every hidden position, as the generation loop
    computes them from the substrate's own weights.

    Each position's first input weight, divided by ``max_weight``, gives its receptor densities
    (`derive_receptor_from_weight`, rescaled to [-1, 1]) and its base gain (``|w| / 2 + 1/2``).
    These are the weights the substrate carries, pruned and locality-scaled, so the weight
    threshold and the locality radius change them as they do in the loop. They are derived at every
    grid position, each from its own weight alone; on a recurrent substrate the forward reads each
    genome's at its active positions, which equals the loop's derivation on its compacted axis.
    Under a global ``'softmax'`` hidden activation it also reads them at the positions outside the
    mask, which the softmax counts as zero-weight nodes (`full_grid_softmax`): a built substrate's
    weights are zero there, so they are the receptors and gain of a zero weight, as the loop
    derives them for the positions its compacted axis leaves out.

    Args:
        substrate: A substrate, feed-forward or recurrent, whose configuration has neuromodulation
            settings.
        neurotransmitter: The vector every node reads; None is the loop's single-task default
            (`default_neurotransmitter`).

    Returns:
        A `SubstrateNeuromodulation` for `substrate_forward`.
    """
    settings = substrate.config.neuromodulation
    if settings is None:
        raise ValueError("this substrate's configuration has no neuromodulation settings "
                         "(SubstrateConfig.neuromodulation); substrate_config fills them in when "
                         "the model has true neuromodulation")
    weight = substrate.W1[:, 0, :] / substrate.config.max_weight
    receptors = derive_receptor_from_weight(
        weight=weight, method=settings.receptor_derivation, num_nt_types=settings.num_nt_types
    ) * 2.0 - 1.0
    base_gains = jnp.abs(weight) * 0.5 + 0.5
    nt = (default_neurotransmitter(settings.num_nt_types) if neurotransmitter is None
          else jnp.asarray(neurotransmitter, dtype=jnp.float32))
    return SubstrateNeuromodulation(receptors, base_gains, nt, settings)


def _checked_neuromodulation(substrate: Substrate, nm: SubstrateNeuromodulation, *,
                             compiled: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """``nm``'s receptors and base gains, after the checks both forwards make before converting
    the inputs (the neurotransmitter vector is `_check_neurotransmitter`'s).

    On a recurrent substrate ``nm`` is held to `SubstrateConfig.validate`'s checks of its own
    settings, with validate's messages, so a caller-built one on a configuration without
    neuromodulation settings is checked as validate checks a configured one; the feed-forward path
    keeps its own message for an unknown mode or node rule.

    Args:
        substrate: The substrate the forward evaluates.
        nm: The neuromodulation it applies, configured or caller-built.
        compiled: The compiled feed-forward forward asks: its receptors are checked for the
            channels the node rule reads, as the recurrent forward's are; the dense feed-forward
            forward does not check them (it fails inside the computation).

    Returns:
        ``(receptors, base_gains)`` as arrays.

    Raises:
        ValueError: On a feed-forward substrate, an unknown modulation mode or node rule. On a
            recurrent one, what validate refuses in these settings (a channel count outside 2 to
            6, an unknown mode or node rule, `_recurrent_neuromodulation_refusal`'s refusals). On a
            recurrent substrate and in the compiled feed-forward forward, receptors without the
            channels the node rule reads. On either, arrays that do not cover the substrate's
            genomes and positions.
    """
    settings = nm.settings
    recurrent = substrate.config.hidden_to_hidden
    if recurrent:
        # validate on the configuration with these settings: its checks and its words, whether
        # the settings are the configured ones or a caller's (the configuration's other fields
        # were validated by substrate_forward)
        substrate.config._replace(neuromodulation=settings).validate()
    elif (settings.modulation_mode not in MODULATION_MODES
            or settings.node_rule not in NEUROMODULATION_NODE_RULES):
        raise ValueError(f"unknown modulation mode {settings.modulation_mode!r} or node rule "
                         f"{settings.node_rule!r}")
    receptors, base_gains = jnp.asarray(nm.receptors), jnp.asarray(nm.base_gains)
    if receptors.shape[:2] != substrate.mask.shape or base_gains.shape != substrate.mask.shape:
        raise ValueError(f"receptors {receptors.shape} and base gains {base_gains.shape} must cover "
                         f"the substrate's {substrate.mask.shape} (genomes, positions)")
    if recurrent or compiled:
        # The node rule reads receptors[:, :k], k every channel under 'anti' and otherwise the first
        # three: fewer, or no channel axis, would fail inside the compiled forward with a JAX
        # TypeError or IndexError. The dense feed-forward forward reads the receptors as given, so
        # there such receptors raise JAX's own error.
        num_nt = int(settings.num_nt_types)
        read = num_nt if settings.node_rule == "anti" else min(3, num_nt)
        if receptors.ndim != 3 or receptors.shape[2] < read:
            raise ValueError(f"receptors {tuple(receptors.shape)} must be "
                             f"{tuple(substrate.mask.shape) + (read,)} or have more channels: node "
                             f"rule {settings.node_rule!r} with {num_nt} channels reads {read} "
                             f"receptor channels")
    return receptors, base_gains


def _check_neurotransmitter(settings: NeuromodulationSettings,
                            neurotransmitter: jnp.ndarray) -> None:
    """Refuse a neurotransmitter vector shorter than the channels the node rule and the inversion
    read.

    Args:
        settings: The neuromodulation settings the forward follows.
        neurotransmitter: The vector, as an array.

    Raises:
        ValueError: The vector is not one-dimensional or has too few entries.
    """
    num_nt = int(settings.num_nt_types)
    # The channels the rule and the inversion read: every one under 'anti', otherwise the first
    # three and, when it inverts the outputs, the fourth. JAX clamps an index past the end, so a
    # shorter vector would run silently.
    needed = num_nt if settings.node_rule == "anti" else min(3, num_nt)
    if settings.output_inversion and num_nt >= 4:
        needed = max(needed, 4)
    if neurotransmitter.ndim != 1 or neurotransmitter.shape[0] < needed:
        raise ValueError(
            f"the neurotransmitter vector has shape {tuple(neurotransmitter.shape)}; node rule "
            f"{settings.node_rule!r} with {num_nt} channels reads {needed} entries "
            f"(default_neurotransmitter gives {min(num_nt, 4)})"
        )


def _neuromodulated_forward(substrate: Substrate, inputs: Any,
                            nm: SubstrateNeuromodulation) -> Any:
    """Every feed-forward substrate under true neuromodulation: the generation loop's per-genome
    forward for this mode, operation for operation.

    The checks run in this order: the settings and the arrays (`_checked_neuromodulation`), the
    inputs' conversion, the configured functions' lookup, then the vector
    (`_check_neurotransmitter`), so a call with two wrong arguments raises the earlier check's
    error.
    """
    config = substrate.config
    settings = nm.settings
    receptors, base_gains = _checked_neuromodulation(substrate, nm)
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    neurotransmitter = jnp.asarray(nm.neurotransmitter)
    per_node_aggregation = substrate.agg_indices is not None
    agg_indices = (substrate.agg_indices if per_node_aggregation
                   else jnp.zeros(substrate.mask.shape, dtype=jnp.int32))
    global_agg_fn = (AGGREGATION_FUNCTIONS[config.global_aggregation]
                     if config.aggregation_mode == "global" else None)
    palette = _palette(config)
    hidden_act_fn = ACTIVATION_FUNCTIONS[config.hidden_activation]
    output_act_fn = ACTIVATION_FUNCTIONS[config.output_activation]
    num_nt = int(settings.num_nt_types)
    _check_neurotransmitter(settings, neurotransmitter)

    def one(W1, W2, act_idx, agg_idx, receptors_i, base_gains_i):
        node_fn, operands = true_neuromodulation_node_rule(
            node_activation_rule('every_iteration', act_idx, hidden_act_fn, palette,
                                 config.num_activations),
            receptors_i, base_gains_i, neurotransmitter, settings.modulation_strength, num_nt,
            settings.modulation_mode, settings.node_rule)
        drive = input_drive(inputs, W1, agg_idx, global_agg_fn, per_node_aggregation,
                            config.true_aggregation, _evaluated_aggregations(config))
        hidden = node_fn(drive, *operands)
        return invert_outputs(output_act_fn(hidden @ W2), neurotransmitter,
                              settings.output_inversion, num_nt, settings.node_rule)

    return jax.vmap(one)(substrate.W1, substrate.W2, substrate.act_indices, agg_indices,
                         receptors, base_gains)


# The compiled feed-forward forward

#: Input rows per call of the compiled feed-forward forward when ``chunk_size`` is None:
#: the chunk a caller that draws pictures uses for its settling forward, so a 128-pixel picture
#: (16,384 rows) is four calls.
_COMPILED_CHUNK_ROWS = 4096

#: A chunk's rows are a multiple of this (``chunk_size`` rounded up, the extra rows zero and
#: dropped), so a row's outputs do not depend on its place in the chunk (under single-threaded
#: Eigen: under XLA's default multi-threaded Eigen a Xeon E5-2683 v4 still put 28 of 12,780
#: three-output values up to 1.2e-7 apart by their place in the chunk). With two or more outputs,
#: XLA:CPU computed the hidden-to-output product of a chunk whose rows leave a remainder of 4 to 7
#: by 8 with the rows 8 * (rows // 8) to + 3 on another kernel path, rounded apart from the same
#: row elsewhere in the chunk (an Apple M4 Max, jax 0.6.1: at 4,095 rows, rows 4,088 to
#: 4,091); remainders of 0 to 3, and one output, did not.
_CHUNK_ROW_MULTIPLE = 8

#: Grid positions per input row that one call of the compiled feed-forward forward holds over its
#: genome slots: a bucket of size S has max(1, this // S) slots (16 at 8, 1 from 128 on). The slot
#: count is fixed per bucket size, so a bucket size has one compiled shape whatever the number of
#: genomes it holds; with the genome count in the shape, every count a bucket met would compile
#: anew (up to the population size per bucket size). Unused slots are zero and cost at most this
#: many positions a row besides the genomes' own; a bucket with more genomes than slots takes
#: several calls.
_COMPILED_SLOT_POSITIONS = 128


def _compiled_slots(size: int) -> int:
    """The genome slots of one call of the compiled feed-forward forward at a bucket size."""
    return max(1, _COMPILED_SLOT_POSITIONS // size)


def _chunk_rows(chunk_size: Optional[int]) -> int:
    """The compiled feed-forward forward's input rows per call: ``chunk_size``, checked and rounded
    up to a multiple of `_CHUNK_ROW_MULTIPLE` (so 4,095 and 4,096 are one program), or the default.

    Raises:
        ValueError: ``chunk_size`` is not a positive integer (a bool is refused).
    """
    if chunk_size is None:
        return _COMPILED_CHUNK_ROWS
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer))
            or chunk_size < 1):
        raise ValueError(f"chunk_size must be a positive integer (input rows per call), got "
                         f"{chunk_size!r}")
    return -(-int(chunk_size) // _CHUNK_ROW_MULTIPLE) * _CHUNK_ROW_MULTIPLE


def _feedforward_key(config: SubstrateConfig, has_act: bool, has_agg: bool,
                     neuromodulation: Optional[NeuromodulationSettings] = None) -> Tuple[Any, ...]:
    """`_compiled_feedforward_forward`'s arguments: the settings the forward reads, in the
    combination given.

    Read and keyed: the output activation; the hidden activation without per-node activations, and
    with them the palette, or the number of activations without a palette; the global aggregation
    function under the 'global' mode; with per-node aggregation outside it, exact or approximated
    aggregation, and under exact aggregation the aggregations evaluated (approximated aggregation
    reads the indices alone); whether the substrate has activation and aggregation indices; under
    true neuromodulation the channel count, the node rule, the modulation mode (unread under
    'anti') and the inversion (unread below four channels). A setting the forward does not read in
    that combination is None in the key (False for the inversion), so changing it compiles nothing:
    a caller may still write its hidden activation with per-node activations on. The key holds no
    shape: the operands' shapes (the bucket size and the chunk rows, the input and output counts,
    the receptor channels, the vector's length) select among the programs one entry's jitted
    function keeps (`_feedforward_forward_fn`), so a bucket size or a chunk of rows never takes an
    entry of its own.
    """
    global_function = config.global_aggregation if config.aggregation_mode == "global" else None
    per_node_aggregation = has_agg and global_function is None
    # Approximated aggregation reads only the indices (compute_aggregation_correction_factors
    # ignores the count of aggregations), so the count is keyed under exact aggregation alone
    exact_aggregation = per_node_aggregation and bool(config.true_aggregation)
    key = (None if not has_act or config.palette is None
           else tuple(int(i) for i in config.palette),
           None if has_act else config.hidden_activation,
           config.output_activation, global_function,
           bool(config.true_aggregation) if per_node_aggregation else None,
           _evaluated_aggregations(config) if exact_aggregation else None,
           int(config.num_activations) if has_act and config.palette is None else None,
           bool(has_act), bool(has_agg))
    if neuromodulation is None:
        return key
    num_nt = int(neuromodulation.num_nt_types)
    node_rule = str(neuromodulation.node_rule)
    return key + ((num_nt, None if node_rule == "anti" else str(neuromodulation.modulation_mode),
                   node_rule, bool(neuromodulation.output_inversion) and num_nt >= 4),)


def _feedforward_forward_fn(config: SubstrateConfig, has_act: bool, has_agg: bool,
                            neuromodulation: Optional[NeuromodulationSettings] = None) -> Any:
    """The compiled feed-forward forward of every bucket size and chunk of rows.

    Keyed on the settings the dense forward reads, by combination (`_feedforward_key` lists them;
    unlike `_recurrent_forward_fn`'s key, it leaves out a setting the combination does not read);
    the bucket size and the rows are its operands' shapes; under true neuromodulation the
    receptors, the base gains, the neurotransmitter vector and the strength are operands, so other
    values of the vector or the strength compile nothing new. A configuration that differs
    elsewhere (thresholds, the receptor derivation, coordinates) gets the same jitted function,
    which compiles one program per operand shapes: a bucket size or a chunk of rows once each,
    other coordinates that keep the input and output counts nothing, while another number of
    inputs or outputs, receptors with another channel count or a vector of another length compile
    each bucket's program once more.
    """
    return _compiled_feedforward_forward(
        *_feedforward_key(config, has_act, has_agg, neuromodulation))


@functools.lru_cache(maxsize=64)
def _compiled_feedforward_forward(palette_entries: Optional[Tuple[int, ...]],
                                  hidden_activation_name: Optional[str], output_activation: str,
                                  global_aggregation: Optional[str],
                                  true_aggregation: Optional[bool],
                                  num_aggregations: Optional[int], num_activations: Optional[int],
                                  has_act: bool, has_agg: bool,
                                  neuromodulation: Optional[Tuple[int, Optional[str], str, bool]]
                                  = None) -> Any:
    """The jitted forward of a bucket's genome slots for one combination of the settings
    `_feedforward_key` reads.

    Each slot computes the dense forward's per-genome function (`dense_substrate_forward`, or the
    neuromodulated node rule and inversion of `_neuromodulated_forward`) on the slot's positions.
    An entry is one jitted function, which keeps one program per operand shapes: one per bucket
    size and chunk of rows, and once more for another input or output count, receptor channel count
    or vector length. So the cache's bound of 64 counts combinations of the settings, not programs
    and not buckets. The cache evicts the least recently used entry, so a workload that cycles
    through more than 64 combinations of the settings recompiles them on every pass. A setting the
    forward does not read is None and reaches no function that would read it.
    ``global_aggregation`` is the global function, None outside the 'global' mode.
    ``neuromodulation`` is ``(num_nt_types, modulation_mode, node_rule, output_inversion)``, which
    adds four operands (receptors, base gains, neurotransmitter vector, strength); None gives the
    forward without it.
    """
    # The palette concrete outside the trace: it selects Python functions while tracing
    palette = None if palette_entries is None else jnp.array(palette_entries, dtype=jnp.int32)
    hidden_fn = (None if hidden_activation_name is None
                 else ACTIVATION_FUNCTIONS[hidden_activation_name])
    output_fn = ACTIVATION_FUNCTIONS[output_activation]
    global_agg_fn = (None if global_aggregation is None
                     else AGGREGATION_FUNCTIONS[global_aggregation])
    act_axis = 0 if has_act else None

    if neuromodulation is None:
        def one(inputs, W1, W2, act_idx, agg_idx):
            return dense_substrate_forward(
                inputs, W1, W2, act_idx, agg_idx,
                hidden_act_fn=hidden_fn, output_act_fn=output_fn, palette=palette,
                num_activations=num_activations, global_agg_fn=global_agg_fn,
                per_node_aggregation=has_agg, true_aggregation=true_aggregation,
                num_aggregations=num_aggregations)

        def forward(inputs, W1, W2, act_idx, agg_idx):
            return jax.vmap(one, in_axes=(None, 0, 0, act_axis, 0))(inputs, W1, W2, act_idx,
                                                                    agg_idx)

        return jax.jit(forward)
    num_nt, modulation_mode, node_rule, output_inversion = neuromodulation

    def one_neuromodulated(inputs, W1, W2, act_idx, agg_idx, receptors, base_gains,
                           neurotransmitter, strength):
        node_fn, operands = true_neuromodulation_node_rule(
            node_activation_rule("every_iteration", act_idx, hidden_fn, palette, num_activations),
            receptors, base_gains, neurotransmitter, strength, num_nt, modulation_mode, node_rule)
        drive = input_drive(inputs, W1, agg_idx, global_agg_fn, has_agg, true_aggregation,
                            num_aggregations)
        hidden = node_fn(drive, *operands)
        return invert_outputs(output_fn(hidden @ W2), neurotransmitter, output_inversion, num_nt,
                              node_rule)

    def neuromodulated(inputs, W1, W2, act_idx, agg_idx, receptors, base_gains, neurotransmitter,
                       strength):
        return jax.vmap(one_neuromodulated, in_axes=(None, 0, 0, act_axis, 0, 0, 0, None, None))(
            inputs, W1, W2, act_idx, agg_idx, receptors, base_gains, neurotransmitter, strength)

    return jax.jit(neuromodulated)


def _compiled_neuromodulation_refusal(settings: NeuromodulationSettings,
                                      neurotransmitter: jnp.ndarray) -> Optional[str]:
    """Why the compiled feed-forward forward cannot use this strength or vector, or None (the dense
    forward checks neither, only the vector's length, `_check_neurotransmitter`).

    The forward zero-pads each genome's receptors and base gains past its count, which adds exact
    zeros only at a finite strength and vector: at a padded position m = receptors @ nt = 0, so an
    infinite strength makes s m = inf * 0 = NaN in the node rule's bias, gate or gain, and an
    infinite or NaN entry of the vector makes m itself NaN; the padded W2 rows are zero, and
    NaN * 0 carries it to every output of the genome. The dense forward's positions outside the
    mask carry the receptors of a zero weight instead and follow another rule, so the two put NaN
    at different genomes. That holds for the strength and for the entries m reads (the first three,
    every one under ``'anti'``). The inversion's entry (the fourth) acts after the output function,
    as in the dense forward, so the padding does not touch it; it is refused with the others for one
    rule over the entries `_check_neurotransmitter` counts. Refused: a strength `_finite_number`
    refuses (as on a recurrent substrate, whose message this follows) and a vector with an entry the
    node rule or the inversion reads that is not finite in float32, the precision the forward reads
    both at.
    """
    if not _finite_number(settings.modulation_strength):
        return (f"neuromodulation.modulation_strength must be a finite number with compiled=True, "
                f"at most {float(np.finfo(np.float32).max):.3g} in magnitude (the forward reads it "
                f"in float32), got {settings.modulation_strength!r}")
    num_nt = int(settings.num_nt_types)
    read = num_nt if settings.node_rule == "anti" else min(3, num_nt)
    if settings.output_inversion and num_nt >= 4:
        read = max(read, 4)
    with np.errstate(over="ignore", invalid="ignore"):
        entries = np.asarray(neurotransmitter).astype(np.float32)[:read]
    if not np.isfinite(entries).all():
        return (f"the neurotransmitter vector's entries the node rule and the inversion read (the "
                f"first {read}) must be finite in float32 with compiled=True (the forward reads "
                f"them in float32), got {np.asarray(neurotransmitter).tolist()}")
    return None


def _compiled_feedforward(substrate: Substrate, inputs: Any,
                          neuromodulation: Optional[SubstrateNeuromodulation],
                          chunk_size: Optional[int]) -> jnp.ndarray:
    """Every feed-forward substrate genome by genome on its mask positions, compiled
    (``substrate_forward(..., compiled=True)``).

    A genome's positions are its mask's in ascending order (the order of `hh_query_rows`), its
    count's bucket the recurrent forward's (`_bucket_size_for`), the positions past the count zero
    in W1, W2, the indices, the receptors and the gains: every function of the palette and every
    node rule is finite at a zero drive, so for finite inputs, strength and vector they add exact
    zeros to the output sum (a strength or a vector entry that is not finite is refused,
    `_compiled_neuromodulation_refusal`; an infinite input meets the padding's zero weights).
    The genomes of a bucket go in its fixed slots (`_compiled_slots`), the unused ones zero; the
    inputs in chunks of ``chunk_size`` rows rounded up to a multiple of `_CHUNK_ROW_MULTIPLE`
    (`_chunk_rows`), the last one zero-padded and its padding dropped. The checks are the dense
    forward's, in its order (the neuromodulation's arrays, the inputs, the configured functions,
    the vector), then this forward's own: the strength and the vector's entries finite in float32.
    """
    config = substrate.config
    if substrate.act_indices is None and config.hidden_activation == "softmax":
        # The recurrent forward adds the positions outside a genome's mask (full_grid_softmax);
        # this forward's genome slots carry no virtual position, so it refuses rather than
        # normalize over the slot's positions alone
        raise ValueError("compiled=True evaluates each genome on its own positions, and the "
                         "'softmax' hidden activation normalizes over every position of the grid, "
                         "which this forward does not add; use compiled=False, which evaluates the "
                         "whole grid")
    rows = _chunk_rows(chunk_size)
    if neuromodulation is None and config.neuromodulation is not None:
        neuromodulation = substrate_neuromodulation(substrate)
    settings = nm_arrays = nm_traced = None
    if neuromodulation is not None:
        settings = neuromodulation.settings
        receptors, base_gains = _checked_neuromodulation(substrate, neuromodulation, compiled=True)
        nm_arrays = (np.asarray(receptors, dtype=np.float32),
                     np.asarray(base_gains, dtype=np.float32))
    inputs = np.asarray(inputs, dtype=np.float32)
    if neuromodulation is not None:
        neurotransmitter = jnp.asarray(neuromodulation.neurotransmitter)
    # The configured functions' lookup (a KeyError for an unknown name), between the inputs'
    # conversion and the vector's check as in the dense forward
    if config.aggregation_mode == "global":
        AGGREGATION_FUNCTIONS[config.global_aggregation]  # noqa: B018
    ACTIVATION_FUNCTIONS[config.hidden_activation]  # noqa: B018
    ACTIVATION_FUNCTIONS[config.output_activation]  # noqa: B018
    if neuromodulation is not None:
        _check_neurotransmitter(settings, neurotransmitter)
        # The zero padding is inert only at a finite strength and vector
        refusal = _compiled_neuromodulation_refusal(settings, neurotransmitter)
        if refusal is not None:
            raise ValueError(refusal)
        # In float32, the recurrent path's precision; the vector and the strength are traced
        nm_traced = (jnp.asarray(neurotransmitter, dtype=jnp.float32),
                     jnp.asarray(settings.modulation_strength, dtype=jnp.float32))
    W1 = np.asarray(substrate.W1, dtype=np.float32)
    W2 = np.asarray(substrate.W2, dtype=np.float32)
    num_inputs, num_outputs = W1.shape[1], W2.shape[2]
    if inputs.ndim != 2 or inputs.shape[1] != num_inputs:
        # The dense forward fails inside the computation on a wrong input count, and on a single
        # row or a leading batch axis under the 'global' aggregation mode or exact per-node
        # aggregation; elsewhere it runs them. This forward takes (n_samples, num_inputs) only.
        raise ValueError(f"inputs must be (n_samples, {num_inputs}) for this substrate's "
                         f"{num_inputs} inputs, got {inputs.shape}")
    masks = np.asarray(substrate.mask, dtype=bool)
    pop = masks.shape[0]
    act = (None if substrate.act_indices is None
           else np.asarray(substrate.act_indices, dtype=np.int32))
    agg = (None if substrate.agg_indices is None
           else np.asarray(substrate.agg_indices, dtype=np.int32))
    total = inputs.shape[0]
    num_chunks = -(-total // rows)
    padded = np.zeros((num_chunks * rows, num_inputs), dtype=np.float32)
    padded[:total] = inputs
    chunks = [jnp.asarray(padded[i * rows:(i + 1) * rows]) for i in range(num_chunks)]
    counts = masks.sum(axis=1)
    sizes = np.array([_bucket_size_for(int(c)) for c in counts], dtype=np.int64)
    outputs = np.zeros((pop, total, num_outputs), dtype=np.float32)
    for size in sorted(set(sizes.tolist())):
        members = np.flatnonzero(sizes == size)
        slots = _compiled_slots(size)
        forward = _feedforward_forward_fn(config, act is not None, agg is not None, settings)
        for start in range(0, members.size, slots):
            real = members[start:start + slots]
            w1 = np.zeros((slots, num_inputs, size), dtype=np.float32)
            w2 = np.zeros((slots, size, num_outputs), dtype=np.float32)
            act_idx = None if act is None else np.zeros((slots, size), dtype=np.int32)
            agg_idx = np.zeros((slots, size), dtype=np.int32)
            if nm_arrays is not None:
                receptors_local = np.zeros((slots, size) + nm_arrays[0].shape[2:], dtype=np.float32)
                gains_local = np.zeros((slots, size), dtype=np.float32)
            for slot, genome in enumerate(real):
                active = np.flatnonzero(masks[genome])
                count = active.size
                w1[slot, :, :count] = W1[genome][:, active]
                w2[slot, :count] = W2[genome][active]
                if act_idx is not None:
                    act_idx[slot, :count] = act[genome][active]
                if agg is not None:
                    agg_idx[slot, :count] = agg[genome][active]
                if nm_arrays is not None:
                    receptors_local[slot, :count] = nm_arrays[0][genome][active]
                    gains_local[slot, :count] = nm_arrays[1][genome][active]
            operands = (jnp.asarray(w1), jnp.asarray(w2),
                        None if act_idx is None else jnp.asarray(act_idx), jnp.asarray(agg_idx))
            if nm_arrays is not None:
                operands += (jnp.asarray(receptors_local), jnp.asarray(gains_local)) + nm_traced
            for index, chunk in enumerate(chunks):
                first, last = index * rows, min((index + 1) * rows, total)
                # Read before the next call, so at most one call's temporaries are alive: with
                # every call dispatched first, the CPU's asynchronous dispatch ran several calls at
                # once and held their temporaries together (whether two calls in flight would be
                # faster on a 2-core host is not measured)
                result = np.asarray(forward(chunk, *operands))
                outputs[real, first:last] = result[:real.size, :last - first]
    return jnp.asarray(outputs)


# Hidden-to-hidden connections

def _bucket_slices(counts: np.ndarray) -> Iterator[Tuple[int, List[Tuple[np.ndarray, np.ndarray]]]]:
    """The generation loop's grouping of genomes for per-genome discovery.

    Genomes go to the smallest bucket size (8, 16, 32, ...) holding their active count, and a
    bucket is processed in slices of a fixed number of genomes for its size, the last slice
    padded by repeating its first genome. So a genome's arrays have the same shapes whatever the
    other genomes are, and the same shapes as in the loop.

    Yields:
        ``(size, [(real, gather), ...])``: the genomes of each slice, and the slice's rows with
        the padding.
    """
    sizes = np.array([_bucket_size_for(int(c)) for c in counts])
    for size in sorted(set(int(s) for s in sizes)):
        members = np.nonzero(sizes == size)[0]
        rows = _per_genome_rows_for(size)
        slices = []
        for start in range(0, members.shape[0], rows):
            real = members[start:start + rows]
            gather = np.concatenate([real, np.full(rows - real.shape[0], real[0])])
            slices.append((real, gather))
        yield size, slices


def _slice_positions(masks: np.ndarray, gather: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Each slice row's active positions in ascending order, padded with -1, and the validity."""
    idx = np.full((gather.shape[0], size), -1, dtype=np.int32)
    for row, genome in enumerate(gather):
        active = np.flatnonzero(masks[genome])
        idx[row, :active.size] = active
    return idx, idx >= 0


def hh_query_rows(substrate: Substrate) -> List[np.ndarray]:
    """The CPPN rows of each genome's hidden-to-hidden discovery.

    Every ordered pair of the genome's active positions: row ``i * count + j`` queries position
    ``active[i]`` to position ``active[j]``, the active positions in ascending index order (the
    order of the generation loop's per-genome discovery). Layout ``[x1, y1, x2, y2, 1]``, or
    ``[x1, y1, x2, y2, x2 - x1, y2 - y1, 1]`` with geometry seeding, as `cppn_query_rows`.

    Args:
        substrate: Output of `build_substrate` with a recurrent configuration.

    Returns:
        One float32 array ``(count**2, input_dim)`` per genome, ``count`` its active positions
        (``(0, input_dim)`` for a genome whose mask is empty).
    """
    config = substrate.config
    if not config.hidden_to_hidden:
        raise ValueError("this configuration has no hidden-to-hidden connections")
    positions = np.asarray(substrate.positions, dtype=np.float32)
    rows = []
    for mask in np.asarray(substrate.mask):
        coords = positions[np.flatnonzero(mask)]
        count = coords.shape[0]
        src = np.broadcast_to(coords[:, None, :], (count, count, 2))
        tgt = np.broadcast_to(coords[None, :, :], (count, count, 2))
        parts = [src, tgt]
        if config.geometry_seeding:
            parts += [tgt[..., 0:1] - src[..., 0:1], tgt[..., 1:2] - src[..., 1:2]]
        parts.append(np.ones((count, count, 1), dtype=np.float32))
        pairs = np.concatenate(parts, axis=-1)
        # The width is explicit: -1 cannot be inferred from an empty mask's size-0 array
        rows.append(pairs.reshape(count * count, pairs.shape[-1]))
    return rows


_pair_outputs_fns: Dict[Tuple, Any] = {}


def query_recurrent_substrate(
    cppn_forward: Any,
    state: Any,
    cppns_transformed: Tuple,
    substrate: Substrate,
) -> List[jnp.ndarray]:
    """Evaluate a population of TensorNEAT CPPNs at `hh_query_rows`, with the loop's own query.

    Args:
        cppn_forward: The genome forward, ``cppn_forward(state, transformed, x)``.
        state: TensorNEAT state.
        cppns_transformed: Transformed population (four arrays, population axis first).
        substrate: Output of `build_substrate` for the same population.

    Returns:
        One ``(count**2, num_cppn_outputs)`` array per genome (``(0, num_cppn_outputs)`` for a genome
        whose mask is empty), for `build_recurrent_substrate`.
    """
    config = substrate.config
    masks = np.asarray(substrate.mask)
    positions = jnp.asarray(substrate.positions)
    counts = masks.sum(axis=1)
    outputs: List[Any] = [None] * masks.shape[0]
    for size, slices in _bucket_slices(counts):
        key = (id(cppn_forward), size, bool(config.geometry_seeding))
        if key not in _pair_outputs_fns:
            _pair_outputs_fns[key] = jax.jit(functools.partial(
                _per_genome_pair_outputs, cppn_forward=cppn_forward,
                geometry_seeding_enabled=bool(config.geometry_seeding)))
        query = _pair_outputs_fns[key]
        for real, gather in slices:
            idx, _ = _slice_positions(masks, gather, size)
            coords = positions[np.where(idx >= 0, idx, 0)]
            cppns = tuple(arr[jnp.asarray(gather)] for arr in cppns_transformed)
            out = np.asarray(query(state, *cppns, coords))
            for row, genome in enumerate(real):
                count = int(counts[genome])
                pairs = out[row, :count, :count]
                outputs[genome] = jnp.asarray(pairs.reshape(count * count, out.shape[-1]))
    return outputs


_pair_connection_fns: Dict[Tuple, Any] = {}


def _pair_connections_fn(size: int, config: SubstrateConfig) -> Any:
    """The loop's post-query arithmetic, compiled for a bucket size and a configuration."""
    key = (size, float(config.band_threshold), float(config.max_weight), bool(config.allow_backward),
           bool(config.allow_lateral), bool(config.allow_self_loops), int(config.iteration_level),
           float(config.hop_decay_factor), config.multi_hop_algorithm)
    if key not in _pair_connection_fns:
        _pair_connection_fns[key] = jax.jit(functools.partial(
            per_genome_pair_connections,
            band_threshold=float(config.band_threshold), max_weight=float(config.max_weight),
            allow_backward=bool(config.allow_backward), allow_lateral=bool(config.allow_lateral),
            allow_self_loops=bool(config.allow_self_loops), iteration_level=int(config.iteration_level),
            hop_decay_factor=float(config.hop_decay_factor),
            use_fori=config.multi_hop_algorithm != "matrix_power"))
    return _pair_connection_fns[key]


def hh_connection_classes(positions: Any, from_indices: Any, to_indices: Any) -> np.ndarray:
    """Classes of hidden-to-hidden connections, as codes into HH_CONNECTION_CLASSES.

    Args:
        positions: (total_positions, 2) grid positions.
        from_indices, to_indices: Source and target positions, any shape; -1 marks padding.

    Returns:
        int8 codes of the same shape, -1 where either index is -1.
    """
    y = np.asarray(positions)[:, 1]
    src, tgt = np.asarray(from_indices), np.asarray(to_indices)
    ok = (src >= 0) & (tgt >= 0)
    src_safe, tgt_safe = np.where(ok, src, 0), np.where(ok, tgt, 0)
    y_src, y_tgt = y[src_safe], y[tgt_safe]
    codes = np.where(src_safe == tgt_safe, 3, np.where(y_src < y_tgt, 0, np.where(y_src == y_tgt, 1, 2)))
    return np.where(ok, codes, -1).astype(np.int8)


def build_recurrent_substrate(substrate: Substrate, hh_outputs: Sequence[Any]) -> Substrate:
    """Fill a substrate's hidden-to-hidden connections from CPPN outputs at `hh_query_rows`.

    Each genome's outputs are placed in the generation loop's bucket shapes and run through
    the loop's arithmetic (`per_genome_pair_connections`: weight and band, classes, the
    multi-hop expansion) and its extraction (``max_sparse_conns`` slots in ascending (source,
    target) order), so the connections are the loop's per-genome discovery bit for bit, and a
    genome's do not depend on the other genomes in the call. A genome whose set exceeds
    ``max_sparse_conns`` raises, as the loop's discovery does.

    Args:
        substrate: Output of `build_substrate` with a recurrent configuration.
        hh_outputs: One array per genome, the CPPN outputs at its `hh_query_rows`:
            ``(count**2, num_cppn_outputs)`` (or ``(count**2,)``); channel 0 is the weight.

    Returns:
        The substrate with its ``hh_`` fields filled.

    Raises:
        ValueError: A genome's connections exceed ``max_sparse_conns`` (``sparse_cap_refusal``).
    """
    config = substrate.config
    if not config.hidden_to_hidden:
        raise ValueError("this configuration has no hidden-to-hidden connections")
    masks = np.asarray(substrate.mask)
    pop = masks.shape[0]
    if len(hh_outputs) != pop:
        raise ValueError(f"expected one output array per genome ({pop}), got {len(hh_outputs)}")
    positions = jnp.asarray(substrate.positions)
    counts = masks.sum(axis=1)
    cap = int(config.max_sparse_conns)
    hh_from = np.full((pop, cap), -1, dtype=np.int32)
    hh_to = np.full((pop, cap), -1, dtype=np.int32)
    hh_weights = np.zeros((pop, cap), dtype=np.float32)
    hh_direct = np.zeros((pop, cap), dtype=bool)
    hh_uncapped = np.zeros(pop, dtype=np.int32)
    weight_channel = []
    for genome in range(pop):
        count = int(counts[genome])
        values = np.asarray(hh_outputs[genome], dtype=np.float32)
        if values.shape[:1] != (count * count,):
            raise ValueError(f"genome {genome}: expected {count * count} rows (its hh_query_rows), "
                             f"got an array of shape {values.shape}")
        weight_channel.append(values.reshape(count * count, -1)[:, 0].reshape(count, count)
                              if count else np.zeros((0, 0), dtype=np.float32))
    for size, slices in _bucket_slices(counts):
        connections = _pair_connections_fn(size, config)
        extract = _get_extract_sparse_fn(0, size, cap)
        for real, gather in slices:
            idx, valid = _slice_positions(masks, gather, size)
            idx_safe = np.where(valid, idx, 0).astype(np.int32)
            out = np.zeros((gather.shape[0], size, size, 1), dtype=np.float32)
            for row, genome in enumerate(gather):
                count = int(counts[genome])
                out[row, :count, :count, 0] = weight_channel[genome]
            mh_weights, mh_valid, direct_valid, direct_weights = connections(
                jnp.asarray(out), positions[idx_safe], jnp.asarray(valid))
            from_b, to_b, w_b = (np.asarray(a) for a in extract(mh_valid, mh_weights, jnp.asarray(idx_safe)))
            mh_valid, direct_valid, direct_weights = (
                np.asarray(a) for a in (mh_valid, direct_valid, direct_weights))
            for row, genome in enumerate(real):
                hh_from[genome], hh_to[genome], hh_weights[genome] = from_b[row], to_b[row], w_b[row]
                hh_uncapped[genome] = int(mh_valid[row].sum())
                kept = from_b[row] >= 0
                active = idx[row][valid[row]]
                i = np.searchsorted(active, from_b[row][kept])
                j = np.searchsorted(active, to_b[row][kept])
                hh_direct[genome, kept] = direct_valid[row, i, j] & (w_b[row][kept] == direct_weights[row, i, j])
    grown = apply_sparse_cap_policy(hh_uncapped, config)
    if grown is not None:
        # sparse_cap_policy 'grow': the substrate again at a cap that holds its largest set, which
        # its configuration then records
        return build_recurrent_substrate(
            substrate._replace(config=config._replace(max_sparse_conns=grown)), hh_outputs)
    return substrate._replace(
        hh_from=jnp.asarray(hh_from),
        hh_to=jnp.asarray(hh_to),
        hh_weights=jnp.asarray(hh_weights),
        hh_valid=jnp.asarray(hh_from >= 0),
        hh_class=hh_connection_classes(substrate.positions, hh_from, hh_to),
        hh_direct=hh_direct,
        hh_uncapped=hh_uncapped,
    )


def _activate_time(config: SubstrateConfig) -> int:
    """The loop's settling steps: configured, or ``2 (2**max_depth + 1)``."""
    if config.activate_time is not None:
        return int(config.activate_time)
    return ((2 ** config.max_depth) + 1) * 2


def _recurrent_forward_fn(config: SubstrateConfig, size: int, has_act: bool, has_hh_agg: bool,
                          return_states: bool,
                          neuromodulation: Optional[NeuromodulationSettings] = None,
                          complete: bool = False) -> Any:
    """One genome's recurrent forward on ``size`` positions: the generation loop's `_build_forward`.

    Compiled once per combination of the settings the forward reads, so a configuration that
    differs elsewhere (coordinates, thresholds, the connection cap) reuses it; at most 64 are kept.

    ``neuromodulation`` gives the forward under true neuromodulation: the settings its node rule
    and the inversion read join the key (channel count, modulation mode, node rule, inversion),
    while the receptors, the base gains, the neurotransmitter vector and the strength are its
    operands, so other values of the vector or the strength compile nothing new. It defaults to
    None, so a call without it builds the forward without neuromodulation, under the key that
    forward has. ``complete`` gives the forward of a global ``'softmax'`` hidden activation, which
    settles the virtual position of `full_grid_softmax` after the genome's positions and takes its
    operands last (`_recurrent_forward`); a key without it builds the forward without it.
    """
    key = (
        _activate_time(config),
        None if config.palette is None else tuple(int(i) for i in config.palette),
        config.hidden_activation,
        config.output_activation,
        config.activation_mode,
        config.aggregation_mode,
        config.global_aggregation,
        bool(config.true_aggregation),
        _evaluated_aggregations(config),
        int(config.num_activations),
        config.hh_activation_mode,
        config.hh_aggregation_mode,
        config.settling_rule,
        size,
        has_act,
        has_hh_agg,
        return_states,
    )
    settings = None if neuromodulation is None else (
        int(neuromodulation.num_nt_types),
        str(neuromodulation.modulation_mode),
        str(neuromodulation.node_rule),
        bool(neuromodulation.output_inversion),
    )
    if complete:
        return _compiled_recurrent_forward(*key, settings, True)
    if neuromodulation is None:
        return _compiled_recurrent_forward(*key)
    return _compiled_recurrent_forward(*key, settings)


@functools.lru_cache(maxsize=64)
def _compiled_recurrent_forward(
    steps: int,
    palette_entries: Optional[Tuple[int, ...]],
    hidden_activation_name: str,
    output_activation: str,
    activation_mode: str,
    aggregation_mode: str,
    global_aggregation: str,
    true_aggregation: bool,
    num_aggregations: int,
    num_activations: int,
    hh_mode: str,
    hh_aggregation_mode: str,
    settling_rule: str,
    size: int,
    has_act: bool,
    has_hh_agg: bool,
    return_states: bool,
    neuromodulation: Optional[Tuple[int, str, str, bool]] = None,
    complete: bool = False,
) -> Any:
    """The compiled forward for one combination of the settings `_recurrent_forward_fn` reads.

    ``neuromodulation`` is ``(num_nt_types, modulation_mode, node_rule, output_inversion)`` for the
    forward under true neuromodulation, which takes four more operands (receptors, base gains,
    neurotransmitter vector, strength); None gives the forward without it. ``complete`` gives the
    forward of a global ``'softmax'`` hidden activation (`_completed_recurrent_forward`); False
    builds the forwards above.
    """
    palette = None if palette_entries is None else jnp.array(palette_entries, dtype=jnp.int32)
    hidden_fn = ACTIVATION_FUNCTIONS[hidden_activation_name]
    output_fn = ACTIVATION_FUNCTIONS[output_activation]
    plain = activation_mode == "disabled" and aggregation_mode == "disabled" and not has_hh_agg
    drive_kwargs = dict(
        global_agg_fn=AGGREGATION_FUNCTIONS[global_aggregation] if aggregation_mode == "global" else None,
        per_node_aggregation=aggregation_mode == "cppn_output",
        true_aggregation=true_aggregation,
        num_aggregations=num_aggregations,
    )
    drive_once = settling_rule == "drive_once"
    if complete:
        return _completed_recurrent_forward(steps, output_fn, drive_kwargs, hh_mode,
                                            hh_aggregation_mode, num_aggregations, drive_once,
                                            return_states, neuromodulation)

    def forward(inputs, W1, W2, act_idx, agg_idx, hh_agg_idx, from_idx, to_idx, weights, valid):
        if plain:
            # The loop's plain recurrent path: input matmul, tanh nodes, sigmoid outputs
            drive = inputs @ W1
            if drive_once:
                hidden = settle_hidden_drive_once(
                    drive,
                    from_idx,
                    to_idx,
                    weights,
                    valid,
                    steps,
                    jnp.tanh,
                    jnp.tanh,
                    return_states=return_states,
                )
            else:
                hidden = settle_hidden(drive, from_idx, to_idx, weights, valid, steps, jnp.tanh,
                                       return_states=return_states)
            output = jax.nn.sigmoid
        else:
            drive = input_drive(inputs, W1, agg_idx, **drive_kwargs)
            node_fn = (jnp.tanh if hh_mode == "initial_only"
                       else node_activation(hh_mode, act_idx, hidden_fn, palette, num_activations))
            if drive_once:
                first_fn = (
                    jnp.tanh
                    if hh_mode == "tanh"
                    else functools.partial(
                        hidden_activation,
                        act_indices=act_idx,
                        hidden_act_fn=hidden_fn,
                        palette=palette,
                        num_activations=num_activations,
                    )
                )
                hidden = settle_hidden_drive_once(
                    drive,
                    from_idx,
                    to_idx,
                    weights,
                    valid,
                    steps,
                    first_fn,
                    node_fn,
                    hh_agg_idx,
                    hh_aggregation_mode,
                    num_aggregations,
                    return_states=return_states,
                )
            else:
                hidden = settle_hidden(
                    drive,
                    from_idx,
                    to_idx,
                    weights,
                    valid,
                    steps,
                    node_fn,
                    hh_agg_idx,
                    hh_aggregation_mode,
                    num_aggregations,
                    return_states=return_states,
                )
            output = output_fn
        if return_states:
            hidden, states = hidden
            return output(hidden @ W2), states
        return output(hidden @ W2)

    if neuromodulation is None:
        return jax.jit(forward)
    num_nt, modulation_mode, node_rule, output_inversion = neuromodulation

    def activation(act_idx):
        # The loop's settling activation as a node rule (_build_forward): tanh under 'initial_only'
        # (the 'drive_once' rule's later steps), else hh_activation_mode's
        if hh_mode == "initial_only":
            return jnp.tanh, ()
        return node_activation_rule(hh_mode, act_idx, hidden_fn, palette, num_activations)

    def first_activation(act_idx):
        # The 'drive_once' rule's first step applies the configured activation (tanh under
        # 'tanh'), as in the loop
        if hh_mode == "tanh":
            return jnp.tanh, ()
        return node_activation_rule("every_iteration", act_idx, hidden_fn, palette, num_activations)

    def neuromodulated(inputs, W1, W2, act_idx, agg_idx, hh_agg_idx, from_idx, to_idx, weights,
                       valid, receptors, base_gains, neurotransmitter, strength):
        # The loop's forward under true neuromodulation (_build_forward, mode
        # 'true_neuromodulation', recurrent) with its functions unchanged; never the plain branch
        # above, whose tanh and sigmoid the loop does not apply here
        def node(act):
            return true_neuromodulation_node_rule(act, receptors, base_gains, neurotransmitter,
                                                  strength, num_nt, modulation_mode, node_rule)

        drive = input_drive(inputs, W1, agg_idx, **drive_kwargs)
        if drive_once:
            # The additive rule of the prototype's FULL variant: the node of the input drive with
            # the configured activation, then activate_time steps of h + node(recurrent(h)), the
            # update through tanh
            first_fn, first_operands = node(first_activation(act_idx))
            update_fn, update_operands = node((jnp.tanh, ()))
            hidden = settle_hidden_additive(first_fn(drive, *first_operands), from_idx, to_idx,
                                            weights, valid, steps, update_fn, update_operands,
                                            return_states=return_states)
            if return_states:
                # The rule's states start with the first value; the forward returns the
                # activate_time states after the update steps, the documented shape, the last being
                # the settled layer
                hidden = hidden[0], hidden[1][1:]
        else:
            node_fn, node_operands = node(activation(act_idx))
            hidden = settle_hidden(drive, from_idx, to_idx, weights, valid, steps, node_fn,
                                   hh_agg_idx, hh_aggregation_mode, num_aggregations,
                                   return_states=return_states, node_operands=node_operands)
        if return_states:
            hidden, states = hidden
            return invert_outputs(output_fn(hidden @ W2), neurotransmitter, output_inversion,
                                  num_nt, node_rule), states
        return invert_outputs(output_fn(hidden @ W2), neurotransmitter, output_inversion, num_nt,
                              node_rule)

    return jax.jit(neuromodulated)


def _completed_recurrent_forward(steps: int, output_fn: Any, drive_kwargs: Dict[str, Any],
                                 hh_mode: str, hh_aggregation_mode: str, num_aggregations: int,
                                 drive_once: bool, return_states: bool,
                                 neuromodulation: Optional[Tuple[int, str, str, bool]]) -> Any:
    """`_compiled_recurrent_forward` under a global ``'softmax'`` hidden activation: the loop's
    forward with the whole grid's softmax on the genome's positions (`full_grid_softmax`).

    The genome's positions (padded to its bucket) carry one more, the virtual position of
    `full_grid_softmax`, settled by the same rule with a zero drive and no connection and dropped
    before the outputs and the states; it stands for the grid positions outside the genome's mask.
    The forwards take the multiplicity (bucket + 1,) last, and under true neuromodulation the
    virtual position's receptor densities and base gain after it (`_recurrent_forward`).
    """
    def nodes(multiplicity):
        # The settling activation and the 'drive_once' rule's first step, as in _build_forward:
        # tanh after the first step under 'initial_only', the whole grid's softmax wherever the
        # configured activation applies ('tanh' never reaches here)
        softmax = (full_grid_softmax, (multiplicity,))
        return ((jnp.tanh, ()) if hh_mode == "initial_only" else softmax), softmax

    def extended(drive, hh_agg_idx):
        drive = jnp.concatenate([drive, jnp.zeros_like(drive[:, :1])], axis=1)
        if hh_agg_idx is not None:
            hh_agg_idx = jnp.concatenate([hh_agg_idx, jnp.zeros_like(hh_agg_idx[:1])])
        return drive, hh_agg_idx

    def dropped(settled):
        if return_states:
            hidden, states = settled
            return hidden[:, :-1], states[..., :-1]
        return settled[:, :-1], None

    if neuromodulation is None:
        def forward(inputs, W1, W2, act_idx, agg_idx, hh_agg_idx, from_idx, to_idx, weights, valid,
                    multiplicity):
            drive, hh_agg = extended(input_drive(inputs, W1, agg_idx, **drive_kwargs), hh_agg_idx)
            (node_fn, node_operands), (first_fn, first_operands) = nodes(multiplicity)
            if drive_once:
                settled = settle_hidden_drive_once(drive, from_idx, to_idx, weights, valid, steps,
                                                   first_fn, node_fn, hh_agg, hh_aggregation_mode,
                                                   num_aggregations, return_states=return_states,
                                                   first_operands=first_operands,
                                                   step_operands=node_operands)
            else:
                settled = settle_hidden(drive, from_idx, to_idx, weights, valid, steps, node_fn,
                                        hh_agg, hh_aggregation_mode, num_aggregations,
                                        return_states=return_states, node_operands=node_operands)
            hidden, states = dropped(settled)
            outputs = output_fn(hidden @ W2)
            return (outputs, states) if return_states else outputs
        return jax.jit(forward)
    num_nt, modulation_mode, node_rule, output_inversion = neuromodulation

    def neuromodulated(inputs, W1, W2, act_idx, agg_idx, hh_agg_idx, from_idx, to_idx, weights,
                       valid, receptors, base_gains, neurotransmitter, strength, multiplicity,
                       receptor_v, gain_v):
        receptors = jnp.concatenate([receptors, receptor_v[None]], axis=0)
        base_gains = jnp.concatenate([base_gains, gain_v[None]])

        def node(act):
            return true_neuromodulation_node_rule(act, receptors, base_gains, neurotransmitter,
                                                  strength, num_nt, modulation_mode, node_rule)

        drive, hh_agg = extended(input_drive(inputs, W1, agg_idx, **drive_kwargs), hh_agg_idx)
        later, first = nodes(multiplicity)
        if drive_once:
            first_fn, first_operands = node(first)
            update_fn, update_operands = node((jnp.tanh, ()))
            settled = settle_hidden_additive(first_fn(drive, *first_operands), from_idx, to_idx,
                                             weights, valid, steps, update_fn, update_operands,
                                             return_states=return_states)
            if return_states:
                settled = settled[0], settled[1][1:]
        else:
            node_fn, node_operands = node(later)
            settled = settle_hidden(drive, from_idx, to_idx, weights, valid, steps, node_fn, hh_agg,
                                    hh_aggregation_mode, num_aggregations,
                                    return_states=return_states, node_operands=node_operands)
        hidden, states = dropped(settled)
        outputs = invert_outputs(output_fn(hidden @ W2), neurotransmitter, output_inversion, num_nt,
                                 node_rule)
        return (outputs, states) if return_states else outputs

    return jax.jit(neuromodulated)


def _recurrent_forward(substrate: Substrate, inputs: Any, return_states: bool,
                       neuromodulation: Optional[SubstrateNeuromodulation] = None) -> Any:
    """Each genome settled on its own active positions (padded to its bucket size).

    Under true neuromodulation (``neuromodulation``) each genome's receptors and base gains are
    gathered at its active positions with its W1 and W2 columns and zero-padded like them, so a
    position its mask clears loses its receptors and gain with its connections, and a padding
    position stays inert (no weight in or out, no connection). With ``return_states`` there, a
    genome's outputs come from the program without states and its states from the program with
    them (`substrate_forward`, Returns).

    Under a global ``'softmax'`` hidden activation a genome's positions carry the virtual position
    of `full_grid_softmax` after the padding, which stands for the grid positions outside its mask:
    their count without neuromodulation, and under it the receptors and base gain the arrays hold
    at the one with the largest softmax input at a zero drive, standing for their weighted count
    (`softmax_missing_virtual`). The padding is left out of the softmax.
    """
    config = substrate.config
    nm_receptors = nm_gains = nm_traced = None
    if neuromodulation is not None:
        receptors, base_gains = _checked_neuromodulation(substrate, neuromodulation)
        neurotransmitter = jnp.asarray(neuromodulation.neurotransmitter)
        _check_neurotransmitter(neuromodulation.settings, neurotransmitter)
        # In float32, the recurrent path's precision; the vector and the strength are traced
        # operands
        nm_receptors = np.asarray(receptors, dtype=np.float32)
        nm_gains = np.asarray(base_gains, dtype=np.float32)
        nm_traced = (jnp.asarray(neurotransmitter, dtype=jnp.float32),
                     jnp.asarray(neuromodulation.settings.modulation_strength, dtype=jnp.float32))
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    masks = np.asarray(substrate.mask)
    pop, total = masks.shape
    W1, W2 = np.asarray(substrate.W1), np.asarray(substrate.W2)
    act = None if substrate.act_indices is None else np.asarray(substrate.act_indices)
    agg = None if substrate.agg_indices is None else np.asarray(substrate.agg_indices)
    hh_agg = None if substrate.hh_agg_indices is None else np.asarray(substrate.hh_agg_indices)
    hh_from, hh_to = np.asarray(substrate.hh_from), np.asarray(substrate.hh_to)
    hh_weights, hh_valid = np.asarray(substrate.hh_weights), np.asarray(substrate.hh_valid)
    outputs = np.zeros((pop, inputs.shape[0], W2.shape[-1]), dtype=np.float32)
    # Per genome over its own positions: a dense (pop, steps, samples, grid) array costs the grid, not the genome
    # (251 MB for a 96 x 96 raster at depth 4 and five steps, against 4 MB for a genome of 20 positions)
    states: Optional[List[jnp.ndarray]] = [] if return_states else None

    def padded(values, size, dtype):
        out = np.zeros((size,) + values.shape[1:], dtype=dtype)
        out[:values.shape[0]] = values
        return out

    # The global 'softmax' hidden activation where the settling applies it: the whole grid's
    # softmax on each genome's positions (_completed_recurrent_forward); the plain forward and
    # 'tanh' settling never apply it
    plain = (config.activation_mode == "disabled" and config.aggregation_mode == "disabled"
             and hh_agg is None)
    complete = (act is None and config.hidden_activation == "softmax"
                and config.hh_activation_mode != "tanh"
                and (neuromodulation is not None or not plain))

    def softmax_completion(genome, count, size):
        """The completed forward's last operands: the multiplicity, and under true neuromodulation
        the virtual position's receptors and base gain."""
        missing = np.flatnonzero(~masks[genome].astype(bool))
        if nm_traced is None:
            return (softmax_axis_multiplicity(count, size, float(missing.size)),)
        if missing.size == 0:
            # The genome holds every grid position: the virtual position stands for none
            return (softmax_axis_multiplicity(count, size, 0.0),
                    jnp.zeros(nm_receptors.shape[-1], jnp.float32), jnp.zeros((), jnp.float32))
        settings = neuromodulation.settings
        receptor_v, gain_v, count_v, _ = softmax_missing_virtual(
            jnp.asarray(nm_receptors[genome][missing]), jnp.asarray(nm_gains[genome][missing]),
            jnp.ones((missing.size,), jnp.float32), nm_traced[0], nm_traced[1],
            int(settings.num_nt_types), settings.modulation_mode, settings.node_rule)
        return softmax_axis_multiplicity(count, size, count_v), receptor_v, gain_v

    for genome in range(pop):
        active = np.flatnonzero(masks[genome])
        count = active.size
        size = _bucket_size_for(count)
        local = np.zeros(total, dtype=np.int32)
        local[active] = np.arange(count, dtype=np.int32)
        # A position outside the mask does not exist here: its W1 and W2 columns are left out below, and so is every
        # connection touching it. A built substrate has none; an edited mask (a knocked-out node) does, and its
        # connections would otherwise land on the genome's first active position.
        valid = (hh_valid[genome] & masks[genome][np.where(hh_valid[genome], hh_from[genome], 0)]
                 & masks[genome][np.where(hh_valid[genome], hh_to[genome], 0)])
        from_local = np.where(valid, local[np.where(valid, hh_from[genome], 0)], 0).astype(np.int32)
        to_local = np.where(valid, local[np.where(valid, hh_to[genome], 0)], 0).astype(np.int32)
        w1 = np.zeros((W1.shape[1], size), dtype=np.float32)
        w1[:, :count] = W1[genome][:, active]
        w2 = padded(W2[genome][active], size, np.float32)
        act_idx = None if act is None else jnp.asarray(padded(act[genome][active], size, np.int32))
        agg_idx = jnp.asarray(padded(agg[genome][active], size, np.int32) if agg is not None
                              else np.zeros(size, dtype=np.int32))
        hh_agg_idx = None if hh_agg is None else jnp.asarray(padded(hh_agg[genome][active], size, np.int32))
        settings = None if neuromodulation is None else neuromodulation.settings
        forward = _recurrent_forward_fn(config, size, act is not None, hh_agg is not None,
                                        return_states, settings, complete=complete)
        operands = (inputs, jnp.asarray(w1), jnp.asarray(w2), act_idx, agg_idx, hh_agg_idx,
                    jnp.asarray(from_local), jnp.asarray(to_local),
                    jnp.asarray(hh_weights[genome]), jnp.asarray(valid))
        if nm_traced is not None:
            receptors_local = padded(nm_receptors[genome][active], size, np.float32)
            gains_local = padded(nm_gains[genome][active], size, np.float32)
            operands += (jnp.asarray(receptors_local), jnp.asarray(gains_local)) + nm_traced
        if complete:
            operands += softmax_completion(genome, count, size)
        result = forward(*operands)
        if return_states:
            out, settled = result
            states.append(settled[:, :, :count])
            if settings is not None:
                # Under true neuromodulation the outputs are the program's without states (the
                # cached program return_states=False runs): the program that also emits the states
                # rounds a settling step 1 to 2 ulps apart from it with the 18-function palette
                # under 'steep_gate', even under single-threaded Eigen, and a chaotic genome
                # carried that from fitness 0.750 to 0.504. Without neuromodulation the outputs
                # stay those of the program with states.
                out = _recurrent_forward_fn(config, size, act is not None, hh_agg is not None,
                                            False, settings, complete=complete)(*operands)
        else:
            out = result
        outputs[genome] = np.asarray(out)
    if return_states:
        return jnp.asarray(outputs), states
    return jnp.asarray(outputs)
