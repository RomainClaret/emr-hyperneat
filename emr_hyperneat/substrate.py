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
``disabled``, ``global`` and ``cppn_output`` and the aggregation modes of the same names,
feed-forward or with hidden-to-hidden connections under per-genome discovery. Neuromodulation,
multi-task evaluation, swarm mode, the weight-interpretation and random function modes, the
union h->h discovery and the h->h cache are not covered, and `EMRHyperNEAT.substrate_config`
refuses a configuration that uses them rather than return a substrate EMR would not evaluate.

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
for bit: the loop evaluates a recurrent population on the union of its active positions.

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
    WEIGHT_THRESHOLD,
    _bucket_size_for,
    _explicit_activate_time,
    _get_extract_sparse_fn,
    _per_genome_pair_outputs,
    _per_genome_rows_for,
    batch_query_population_multi_source_multi_output,
    batch_query_population_self_connections,
    INDEX_DECODES,
    continuous_to_index,
    cppn_output_function_indices,
    dense_substrate_forward,
    hh_pair_weights,
    input_drive,
    node_activation,
    per_genome_pair_connections,
    self_connection_rows,
    settle_hidden,
    substrate_masks,
    substrate_weight_matrices,
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
    "Substrate",
    "SubstrateConfig",
    "SubstrateQueries",
    "WEIGHT_THRESHOLD",
    "build_recurrent_substrate",
    "build_substrate",
    "compute_hierarchical_variances_batch",
    "cppn_output_function_indices",
    "cppn_query_rows",
    "get_hierarchical_grid",
    "hh_connection_classes",
    "hh_pair_weights",
    "hh_query_rows",
    "query_recurrent_substrate",
    "query_substrate",
    "self_connection_rows",
    "substrate_forward",
    "substrate_queries",
    "substrate_weight_matrices",
]

# WEIGHT_THRESHOLD is a constant of the engine, not a setting: input-to-hidden and hidden-to-output
# weights at or below it are pruned (SubstrateConfig.weight_threshold carries it for building).

#: Activation modes a substrate can be built for.
ACTIVATION_MODES = ("disabled", "global", "cppn_output")
#: Aggregation modes a substrate can be built for.
AGGREGATION_MODES = ("disabled", "global", "cppn_output")
#: The classes of a hidden-to-hidden connection (`Substrate.hh_class` codes index this tuple), by
#: the y coordinates of its ends as in the recurrence configuration: forward when the source lies
#: below the target, lateral on the same y, backward above, self a position to itself.
HH_CONNECTION_CLASSES = ("forward", "lateral", "backward", "self")


class SubstrateConfig(NamedTuple):
    """Everything that decides how a CPPN becomes a feed-forward substrate.

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
        activate_time: Settling steps, an integer of at least 1 (a bool is refused); None is the loop's
            automatic count, ``2 (2**max_depth + 1)``. No effect without hidden-to-hidden connections.
        settling_rule: ``'emr'``, the package's one settling rule (:func:`settle_hidden`).
        hh_activation_mode: The activation of a settling step, ``'every_iteration'`` or ``'tanh'``.
        hh_aggregation_mode: ``'sum'``, or ``'dynamic'`` (per-node aggregation of the recurrent
            input, read from the aggregation channel at the self-connection).
        activation_index_decode: How the activation channel becomes a function index,
            ``'tanh'`` (the default: a palette of 9 or more never reaches its end entries) or
            ``'uniform'`` (every entry, equal bins); `continuous_to_index`.
        aggregation_index_decode: The same for the aggregation channel, including the recurrent
            input's dynamic aggregation.
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
            if self.settling_rule != "emr":
                raise ValueError(f"settling_rule must be 'emr', the package's one settling rule, got "
                                 f"{self.settling_rule!r}")
            if self.hh_activation_mode not in ("every_iteration", "tanh"):
                raise ValueError(f"hh_activation_mode must be 'every_iteration' or 'tanh', got "
                                 f"{self.hh_activation_mode!r}")
            if self.hh_aggregation_mode not in ("sum", "dynamic"):
                raise ValueError(f"hh_aggregation_mode must be 'sum' or 'dynamic', got {self.hh_aggregation_mode!r}")
            if self.hh_aggregation_mode == "dynamic" and not (
                    self.aggregation_mode == "cppn_output" and self.num_cppn_outputs > 1):
                raise ValueError("hh_aggregation_mode 'dynamic' reads the aggregation channel at the "
                                 "self-connection: it needs aggregation_mode 'cppn_output' and a multi-output CPPN")
            if self.multi_hop_algorithm not in ("matrix_power", "fori_loop"):
                raise ValueError(f"multi_hop_algorithm must be 'matrix_power' or 'fori_loop', "
                                 f"got {self.multi_hop_algorithm!r}")
            if self.max_sparse_conns < 1:
                raise ValueError(f"max_sparse_conns must be at least 1, got {self.max_sparse_conns}")
            if self.activate_time is not None:
                _explicit_activate_time("activate_time", self.activate_time)
        return self


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
            A genome whose mask is cleared entirely has no hidden node: no hidden-to-hidden rows or
            connections, settling states of width 0, and outputs equal to the output activation of
            zero, as the feed-forward forward gives with every column zeroed.
            The feed-forward forward reads W1 and W2 as given (a built substrate's columns outside the
            mask are zero), so there a node is removed by zeroing its columns.
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
    if config.aggregation_mode == "cppn_output":
        agg_indices = cppn_output_function_indices(
            self_outputs, masks, W1,
            channel=config.aggregation_channel,
            num_options=config.num_aggregations,
            num_cppn_outputs=config.num_cppn_outputs,
            decode=config.aggregation_index_decode,
        )
    hh_agg_indices = None
    if config.hidden_to_hidden and config.hh_aggregation_mode == "dynamic":
        # The generation loop's decode of the recurrent-input aggregation: the aggregation channel
        # at the self-connection, unmasked (validate() guarantees the channel exists).
        hh_agg_indices = continuous_to_index(
            jnp.asarray(self_outputs)[:, :, config.aggregation_channel], config.num_aggregations,
            config.aggregation_index_decode)
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


def _palette(config: SubstrateConfig) -> Optional[jnp.ndarray]:
    if config.palette is None:
        return None
    return jnp.array(config.palette, dtype=jnp.int32)


def substrate_forward(substrate: Substrate, inputs: Any, return_states: bool = False) -> Any:
    """Run every substrate on the same inputs.

    A feed-forward substrate runs on the whole grid, as the generation loop does (bit for bit).
    A recurrent one settles its hidden layer for ``activate_time`` steps, each genome on its own
    active positions (padded to a bucket size, so compiled shapes repeat), which matches the
    loop's fitness within float32 rounding.

    Args:
        substrate: Output of `build_substrate`, and of `build_recurrent_substrate` when the
            configuration has hidden-to-hidden connections.
        inputs: (n_samples, num_inputs).
        return_states: Also return the hidden state after each settling step (recurrent only).

    Returns:
        (pop_size, n_samples, num_outputs); with ``return_states`` the pair ``(outputs,
        states)``, states a list with one array per genome, (activate_time, n_samples, count)
        over the genome's own active positions in ascending order (the order of
        `hh_query_rows`, so the substrate's mask maps it back to the grid).
    """
    config = substrate.config
    if config.hidden_to_hidden:
        if substrate.hh_from is None:
            raise ValueError("this substrate has hidden-to-hidden connections to settle: build them with "
                             "build_recurrent_substrate first")
        return _recurrent_forward(substrate, inputs, return_states)
    if return_states:
        raise ValueError("return_states needs a recurrent substrate: a feed-forward one does not settle")
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
            num_aggregations=config.num_aggregations,
        )

    return jax.vmap(one)(substrate.W1, substrate.W2, substrate.act_indices, agg_indices)


# ---------------------------------------------------------------------------------------------------
# Hidden-to-hidden connections
# ---------------------------------------------------------------------------------------------------

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
    multi-hop expansion) and its extraction (the first ``max_sparse_conns`` connections in
    ascending (source, target) order), so the connections are the loop's per-genome discovery
    bit for bit, and a genome's do not depend on the other genomes in the call.

    Args:
        substrate: Output of `build_substrate` with a recurrent configuration.
        hh_outputs: One array per genome, the CPPN outputs at its `hh_query_rows`:
            ``(count**2, num_cppn_outputs)`` (or ``(count**2,)``); channel 0 is the weight.

    Returns:
        The substrate with its ``hh_`` fields filled.
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
                          return_states: bool) -> Any:
    """One genome's recurrent forward on ``size`` positions: the generation loop's `_build_forward`.

    Compiled once per combination of the settings the forward reads, so a configuration that differs elsewhere
    (coordinates, thresholds, the connection cap) reuses it; at most 64 are kept.
    """
    return _compiled_recurrent_forward(
        _activate_time(config), None if config.palette is None else tuple(int(i) for i in config.palette),
        config.hidden_activation, config.output_activation, config.activation_mode, config.aggregation_mode,
        config.global_aggregation, bool(config.true_aggregation), int(config.num_aggregations),
        int(config.num_activations), config.hh_activation_mode, config.hh_aggregation_mode,
        size, has_act, has_hh_agg, return_states)


@functools.lru_cache(maxsize=64)
def _compiled_recurrent_forward(steps: int, palette_entries: Optional[Tuple[int, ...]], hidden_activation: str,
                                output_activation: str, activation_mode: str, aggregation_mode: str,
                                global_aggregation: str, true_aggregation: bool, num_aggregations: int,
                                num_activations: int, hh_mode: str, hh_aggregation_mode: str, size: int,
                                has_act: bool, has_hh_agg: bool, return_states: bool) -> Any:
    """The compiled forward for one combination of the settings `_recurrent_forward_fn` reads."""
    palette = None if palette_entries is None else jnp.array(palette_entries, dtype=jnp.int32)
    hidden_fn = ACTIVATION_FUNCTIONS[hidden_activation]
    output_fn = ACTIVATION_FUNCTIONS[output_activation]
    plain = activation_mode == "disabled" and aggregation_mode == "disabled" and not has_hh_agg
    drive_kwargs = dict(
        global_agg_fn=AGGREGATION_FUNCTIONS[global_aggregation] if aggregation_mode == "global" else None,
        per_node_aggregation=aggregation_mode == "cppn_output",
        true_aggregation=true_aggregation,
        num_aggregations=num_aggregations,
    )

    def forward(inputs, W1, W2, act_idx, agg_idx, hh_agg_idx, from_idx, to_idx, weights, valid):
        if plain:
            # The loop's plain recurrent path: input matmul, tanh nodes, sigmoid outputs
            drive = inputs @ W1
            hidden = settle_hidden(drive, from_idx, to_idx, weights, valid, steps, jnp.tanh,
                                   return_states=return_states)
            output = jax.nn.sigmoid
        else:
            drive = input_drive(inputs, W1, agg_idx, **drive_kwargs)
            node_fn = node_activation(hh_mode, act_idx, hidden_fn, palette, num_activations)
            hidden = settle_hidden(drive, from_idx, to_idx, weights, valid, steps, node_fn, hh_agg_idx,
                                   hh_aggregation_mode, num_aggregations, return_states=return_states)
            output = output_fn
        if return_states:
            hidden, states = hidden
            return output(hidden @ W2), states
        return output(hidden @ W2)

    return jax.jit(forward)


def _recurrent_forward(substrate: Substrate, inputs: Any, return_states: bool) -> Any:
    """Each genome settled on its own active positions (padded to its bucket size)."""
    config = substrate.config
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    masks = np.asarray(substrate.mask)
    pop, total = masks.shape
    W1, W2 = np.asarray(substrate.W1), np.asarray(substrate.W2)
    act = None if substrate.act_indices is None else np.asarray(substrate.act_indices)
    agg = None if substrate.agg_indices is None else np.asarray(substrate.agg_indices)
    hh_agg = None if substrate.hh_agg_indices is None else np.asarray(substrate.hh_agg_indices)
    hh_from, hh_to = np.asarray(substrate.hh_from), np.asarray(substrate.hh_to)
    hh_weights, hh_valid = np.asarray(substrate.hh_weights), np.asarray(substrate.hh_valid)
    steps = _activate_time(config)
    outputs = np.zeros((pop, inputs.shape[0], W2.shape[-1]), dtype=np.float32)
    # Per genome over its own positions: a dense (pop, steps, samples, grid) array costs the grid, not the genome
    # (251 MB for a 96 x 96 raster at depth 4 and five steps, against 4 MB for a genome of 20 positions)
    states: Optional[List[jnp.ndarray]] = [] if return_states else None

    def padded(values, size, dtype):
        out = np.zeros((size,) + values.shape[1:], dtype=dtype)
        out[:values.shape[0]] = values
        return out

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
        forward = _recurrent_forward_fn(config, size, act is not None, hh_agg is not None, return_states)
        result = forward(inputs, jnp.asarray(w1), jnp.asarray(w2), act_idx, agg_idx, hh_agg_idx,
                         jnp.asarray(from_local), jnp.asarray(to_local),
                         jnp.asarray(hh_weights[genome]), jnp.asarray(valid))
        if return_states:
            out, settled = result
            states.append(settled[:, :, :count])
        else:
            out = result
        outputs[genome] = np.asarray(out)
    if return_states:
        return jnp.asarray(outputs), states
    return jnp.asarray(outputs)
