"""HMR's node-list recurrent model on the EMR engine.

The model of HMR's recurrent-network variant (RN below), opt-in with
``recurrence.network_model: node_list``. Each genome's network is discovered on the leaves of a
complete quadtree at depth ``max_depth`` and settled as a list of nodes. RN made every leaf a
candidate (its active-leaf mask was all true, so no variance decided anything) and kept a
connection where the band test passed:

1. phase 1 connects each input to the leaves that pass;
2. phase 2 explores from the nodes found, ``iteration_level`` rounds, along the configured
   connection classes;
3. phase 3 connects the leaves that pass to each output;
4. cleaning keeps the connections that lie on a path from an input to an output.

RN ran this per source and per genome in host loops. Here a whole population is one compiled
call on fixed shapes. Every query is the CPPN at (source, lattice point), the lattice being the
leaves and their band neighbours, so each pair is queried once and the leaf and neighbour weights
are gathered from it. Phase 2's rounds are boolean steps over a leaves x leaves candidate matrix,
and cleaning is a bounded fixpoint. A genome's network is three dense blocks, inputs x leaves,
leaves x leaves and leaves x outputs, zero where there is no connection. On RN's own populations
the networks equal RN's bit for bit on CPU wherever RN's defects do not act. Those defects are
fixed here, not reproduced (docs/configuration.md):

- the band compares weights scaled by ``max_weight`` at both ends; RN scaled the neighbours by 8;
- a genome whose phases find nothing is empty; RN invented connections at weight 0.5;
- every connection counts; RN evaluated the first 500 in Python set order;
- depths above `MAX_DEPTH` are refused; from depth 6 RN dropped every hidden connection;
- cleaning runs to its fixpoint; RN stopped after 50 iterations;
- sums run in a fixed order; RN's ran in Python set order, so its fitness depended on hash order.

The forward (the engine's `_node_list_forward`) settles the hidden leaves with the engine's
settling step and reads the outputs one step behind them, as RN's synchronous update did:
``h_t = tanh(x W_in + h_(t-1) W_hh)`` for ``t = 1 .. T-1`` and ``out = sigmoid(h_(T-1) W_out)``.
"""
import threading
from functools import lru_cache
from typing import Any, Callable, Dict, NamedTuple, Sequence, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from emr_hyperneat.emrhyperneat_base import (
    NEIGHBOR_OFFSETS,
    cppn_query_rows,
    get_quadtree_structure,
    vectorized_weight_sparsification,
)

#: The deepest quadtree the node list accepts. From depth 6 RN's own build dropped every hidden
#: connection, so there is no reference, and a genome's search is 4^12 = 16.7M leaf pairs.
MAX_DEPTH = 5

#: RN's threshold for "the same coordinate" in its connection constraints.
SAME_COORDINATE = 1e-6

#: RN's sparsification threshold, PUREPLES's. The compiled core takes it as an argument, not a
#: constant: XLA rewrites a division by the constant 1 - 0.2 as a multiplication by its rounded
#: reciprocal, which moved one weight in ten by a float32 ulp against RN's eager division.
SPARSIFY_THRESHOLD = 0.2

#: RN evaluated at most this many connections per genome, the first in Python set order.
#: Reported per generation (``node_list_over_rn_cap``) so a run that reproduces an RN run can see
#: where RN would have truncated; never applied.
RN_CONNECTION_CAP = 500

#: CPPN query rows per compiled step: the budgets of EMR's per-genome h->h discovery
#: (``_PER_GENOME_ROW_BUDGET`` and ``_PER_GENOME_BLOCK_PAIRS`` in the engine), at 450-650 bytes of
#: XLA temporaries per query.
QUERY_ROW_BUDGET = 1 << 16
QUERY_BLOCK_BUDGET = 1 << 18

#: The per-genome counts `NodeListNetworks.counts` holds, in this order.
COUNT_NAMES = ('input_candidates', 'hidden_candidates', 'output_candidates', 'input_connections',
               'hidden_connections', 'output_connections', 'discovered_nodes', 'hidden_nodes',
               'cleaning_iterations')


def leaf_depth(max_depth: int) -> int:
    """The quadtree depth of the leaves for ``emr_hyperneat.max_depth``.

    RN's meaning, the quadtree depth itself, so an RN configuration carries over with the same
    number. EMR's grid meaning would put the leaves on the finest grid level, ``max_depth + 1``.
    """
    return int(max_depth)


class NodeListSettings(NamedTuple):
    """What a node-list configuration decides, as the engine parsed it.

    Attributes:
        depth: Quadtree depth of the leaves (`leaf_depth`), 1 to `MAX_DEPTH`.
        rounds: Phase-2 rounds (``recurrence.iteration_level``); 0 without hidden-to-hidden
            connections.
        classes: ``(allow_backward, allow_lateral, allow_self_loops)`` of phase 2.
        band_threshold: The band test's threshold.
        max_weight: Weight scale of the sparsification.
        activate_time: Steps T of the forward, at least 2.
        population_chunk: ``emr_hyperneat.population_chunk_size``: -1 the query budgets, 0 the
            whole population per compiled step, a positive value that many genomes.
    """
    depth: int
    rounds: int
    classes: Tuple[bool, bool, bool]
    band_threshold: float
    max_weight: float
    activate_time: int
    population_chunk: int


class NodeListLattice(NamedTuple):
    """The fixed shapes of a node-list configuration: leaves, query points and constraint masks.

    Attributes:
        depth: Quadtree depth of the leaves.
        leaves: (L, 2) float32 leaf centres, sorted by x then y (the order of RN's hidden
            nodes).
        points: (M, 2) float32 query points: the leaves and their band neighbours, sorted.
        leaf_index: (L,) int32, each leaf's row in ``points``.
        neighbor_index: (L, 4) int32, each leaf's left, right, top and bottom neighbours in
            ``points``.
        rows: ((I + L + O) * M, 5) float32 CPPN rows: each input and each leaf to every point
            (outgoing), then every point to each output (incoming), in ``cppn_query_rows``'s
            layout.
        input_mask: (I, L) phase 1's constraint.
        output_mask: (L, O) phase 3's constraint.
        input_coords: (I, 2) float32.
        output_coords: (O, 2) float32.
    """
    depth: int
    leaves: np.ndarray
    points: np.ndarray
    leaf_index: np.ndarray
    neighbor_index: np.ndarray
    rows: np.ndarray
    input_mask: np.ndarray
    output_mask: np.ndarray
    input_coords: np.ndarray
    output_coords: np.ndarray


class NodeListNetworks(NamedTuple):
    """A population's discovered networks, three dense blocks per genome.

    Attributes:
        W_in: (pop, I, L) input-to-leaf weights.
        W_hh: (pop, L, L) leaf-to-leaf weights.
        W_out: (pop, L, O) leaf-to-output weights.
        hidden: (pop, L) the hidden nodes after cleaning.
        counts: (pop, len(COUNT_NAMES)) int32.
    """
    W_in: jnp.ndarray
    W_hh: jnp.ndarray
    W_out: jnp.ndarray
    hidden: jnp.ndarray
    counts: jnp.ndarray


@lru_cache(maxsize=None)
def _sorted_leaves(depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """The leaves of RN's complete quadtree and their band neighbours, sorted by x then y.

    The quadtree is the base module's copy of RN's construction (`precompute_quadtree_structure`);
    the neighbours are RN's own float32 expression, exact for these binary fractions, so the CPPN
    sees RN's coordinates bit for bit.
    """
    tree = get_quadtree_structure(depth)
    leaves = np.asarray(tree.leaf_positions)
    neighbours = np.asarray(tree.leaf_positions[:, None, :]
                            + NEIGHBOR_OFFSETS[None, :, :] * tree.leaf_widths[:, None, None])
    order = np.lexsort((leaves[:, 1], leaves[:, 0]))
    return leaves[order], neighbours[order]


@lru_cache(maxsize=None)
def node_list_lattice(depth: int, input_coords: Tuple[Tuple[float, float], ...],
                      output_coords: Tuple[Tuple[float, float], ...]) -> NodeListLattice:
    """The lattice, rows and constraint masks of a depth and a substrate's coordinates.

    Args:
        depth: Quadtree depth of the leaves, 1 to `MAX_DEPTH`.
        input_coords: The substrate's input coordinates.
        output_coords: The substrate's output coordinates.

    Returns:
        A `NodeListLattice`.
    """
    leaves, neighbours = _sorted_leaves(depth)
    num_leaves = leaves.shape[0]
    points, inverse = np.unique(np.concatenate([leaves, neighbours.reshape(-1, 2)]), axis=0,
                                return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    inputs = np.asarray(input_coords, dtype=np.float32).reshape(-1, 2)
    outputs = np.asarray(output_coords, dtype=np.float32).reshape(-1, 2)
    rows = np.concatenate(
        [cppn_query_rows(source, points, True) for source in np.concatenate([inputs, leaves])]
        + [cppn_query_rows(target, points, False) for target in outputs]
    ).astype(np.float32)
    same = np.float32(SAME_COORDINATE)
    # Phase 1, RN's feed-forward constraint: the leaf above the input or level with it, not on
    # it
    y_in, x_in = inputs[:, None, 1], inputs[:, None, 0]
    y_leaf, x_leaf = leaves[None, :, 1], leaves[None, :, 0]
    level = np.abs(y_in - y_leaf) < same
    input_mask = ((y_in < y_leaf) | level) & ~((np.abs(x_in - x_leaf) < same) & level)
    # Phase 3, RN's incoming constraint, with its self mask: the leaf at or below the output, not
    # on it
    y_out, x_out = outputs[None, :, 1], outputs[None, :, 0]
    y_src, x_src = leaves[:, None, 1], leaves[:, None, 0]
    on_output = (np.abs(x_out - x_src) < same) & (np.abs(y_out - y_src) < same)
    output_mask = (y_src <= y_out) & ~on_output
    return NodeListLattice(
        depth=depth, leaves=leaves, points=points, leaf_index=inverse[:num_leaves].astype(np.int32),
        neighbor_index=inverse[num_leaves:].reshape(num_leaves, 4).astype(np.int32), rows=rows,
        input_mask=input_mask, output_mask=output_mask, input_coords=inputs, output_coords=outputs)


@lru_cache(maxsize=None)
def hidden_class_mask(depth: int, allow_backward: bool, allow_lateral: bool,
                      allow_self_loops: bool) -> np.ndarray:
    """Phase 2's constraint between leaves, RN's, (L, L) over the sorted leaves.

    Forward connections (source below target) always; lateral ones (level, not the same leaf),
    backward ones and self-loops as configured.
    """
    leaves, _ = _sorted_leaves(depth)
    same = np.float32(SAME_COORDINATE)
    y_src, y_tgt = leaves[:, None, 1], leaves[None, :, 1]
    level = np.abs(y_src - y_tgt) < same
    is_self = (np.abs(leaves[:, None, 0] - leaves[None, :, 0]) < same) & level
    valid = y_src < y_tgt
    if allow_lateral:
        valid = valid | (level & ~is_self)
    if allow_backward:
        valid = valid | (y_src > y_tgt)
    if allow_self_loops:
        valid = valid | is_self
    else:
        valid = valid & ~is_self
    return valid


def coincident_coordinates(depth: int, input_coords: Sequence[Sequence[float]],
                           output_coords: Sequence[Sequence[float]]
                           ) -> Tuple[Tuple[float, float], ...]:
    """Input or output coordinates on a leaf or on each other, which would merge nodes.

    RN matched coordinates within an L1 distance of 1e-4; the same test here. The engine refuses
    a configuration where this is not empty.
    """
    leaves, _ = _sorted_leaves(depth)
    coords = np.asarray(list(input_coords) + list(output_coords), dtype=np.float32).reshape(-1, 2)
    clash = []
    for i, coord in enumerate(coords):
        on_leaf = np.any(np.sum(np.abs(leaves - coord), axis=1) < 1e-4)
        on_other = np.any(np.sum(np.abs(np.delete(coords, i, axis=0) - coord), axis=1) < 1e-4)
        if on_leaf or on_other:
            clash.append((float(coord[0]), float(coord[1])))
    return tuple(clash)


def explore(c1: jnp.ndarray, m_hh: jnp.ndarray, rounds: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Phase 2's rounds over one genome's candidates, as RN explored them.

    Each round explores every leaf of the frontier: its candidate connections join the network
    and the targets not found before form the next frontier. RN's unexplored set also re-visits
    earlier leaves from round 3 on, which adds nothing: a source's discovery is a function of the
    genome and the source alone.

    Args:
        c1: (I, L) phase 1's connections; their targets are the first frontier.
        m_hh: (L, L) phase 2's candidate connections between leaves.
        rounds: ``iteration_level`` (static).

    Returns:
        ``(found, explored)``, (L,) each: the leaves discovered, and those explored, whose rows of
        ``m_hh`` are connections.
    """
    def step(_, carry):
        found, frontier, explored = carry
        explored = explored | frontier
        reached = jnp.any(frontier[:, None] & m_hh, axis=0)
        new = reached & ~found
        return found | new, new, explored

    found0 = jnp.any(c1, axis=0)
    found, _, explored = lax.fori_loop(0, rounds, step,
                                       (found0, found0, jnp.zeros_like(found0)))
    return found, explored


def clean(c1: jnp.ndarray, c2: jnp.ndarray, c3: jnp.ndarray,
          found: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    """Cleaning, RN's rule: keep a connection whose source an input reaches and whose target
    reaches an output.

    Discovery reaches every found leaf from an input, so only the backward set needs a fixpoint.
    It runs until nothing changes, bounded by the leaf count since every productive iteration adds
    a leaf; RN stopped after 50 iterations.

    Args:
        c1: (I, L) input-to-leaf connections.
        c2: (L, L) leaf-to-leaf connections (explored rows only).
        c3: (L, O) leaf-to-output connections, from any leaf.
        found: (L,) the leaves discovery reached.

    Returns:
        ``(k1, k2, k3, hidden, iterations)``: the kept blocks, the hidden leaves and the
        fixpoint's iterations.
    """
    leaves = found.shape[0]

    def unsettled(carry):
        _, changed, iteration = carry
        return changed & (iteration <= leaves)

    def spread(carry):
        reach, _, iteration = carry
        grown = reach | jnp.any(c2 & reach[None, :], axis=1)
        return grown, jnp.any(grown != reach), iteration + 1

    reach0 = jnp.any(c3, axis=1)
    reach, _, iterations = lax.while_loop(unsettled, spread,
                                          (reach0, jnp.array(True), jnp.array(0)))
    return (c1 & reach[None, :], c2 & reach[None, :], c3 & found[:, None], found & reach,
            iterations)


def discover_genome(raw: jnp.ndarray, sparsify_threshold: jnp.ndarray, leaf_index: jnp.ndarray,
                    neighbor_index: jnp.ndarray, input_mask: jnp.ndarray, hh_mask: jnp.ndarray,
                    output_mask: jnp.ndarray, num_inputs: int, num_leaves: int,
                    max_weight: float, band_threshold: float,
                    rounds: int) -> Tuple[jnp.ndarray, ...]:
    """One genome's network from its CPPN outputs on the lattice.

    Args:
        raw: (I + L + O, M) CPPN weight outputs: inputs and leaves to every point, every point to
            the outputs.
        sparsify_threshold: `SPARSIFY_THRESHOLD` as a float32 array (an argument of the compiled
            core).
        leaf_index, neighbor_index: The lattice's gathers.
        input_mask, hh_mask, output_mask: The three constraints, (I, L), (L, L), (L, O).
        num_inputs, num_leaves: I and L.
        max_weight: Weight scale.
        band_threshold: Band threshold.
        rounds: Phase-2 rounds.

    Returns:
        ``(W_in, W_hh, W_out, hidden, counts)`` as `NodeListNetworks` holds them for one genome.
    """
    inputs, leaves = num_inputs, num_leaves
    # RN's sparsification; leaves and neighbours on one scale (RN scaled the neighbours by 8)
    weights = vectorized_weight_sparsification(raw, threshold=sparsify_threshold,
                                               max_weight=max_weight)
    weight = weights[:, leaf_index]
    neighbour = weights[:, neighbor_index]
    # RN's band: max(min(top, bottom), min(left, right)) of |w - neighbour|, and a nonzero
    # weight
    diff = jnp.abs(weight[..., None] - neighbour)
    band = jnp.maximum(jnp.minimum(diff[..., 2], diff[..., 3]),
                       jnp.minimum(diff[..., 0], diff[..., 1]))
    candidate = (band > band_threshold) & (weight != 0.0)
    c1 = candidate[:inputs] & input_mask
    m_hh = candidate[inputs:inputs + leaves] & hh_mask
    c3 = jnp.transpose(candidate[inputs + leaves:]) & output_mask
    found, explored = explore(c1, m_hh, rounds)
    c2 = m_hh & explored[:, None]
    k1, k2, k3, hidden, iterations = clean(c1, c2, c3, found)
    W_in = jnp.where(k1, weight[:inputs], 0.0)
    W_hh = jnp.where(k2, weight[inputs:inputs + leaves], 0.0)
    W_out = jnp.where(k3, jnp.transpose(weight[inputs + leaves:]), 0.0)
    counts = jnp.stack([jnp.sum(c1), jnp.sum(c2), jnp.sum(c3), jnp.sum(k1), jnp.sum(k2),
                        jnp.sum(k3), jnp.sum(found), jnp.sum(hidden), iterations])
    return W_in, W_hh, W_out, hidden, counts.astype(jnp.int32)


def call_plan(queries_per_genome: int, population: int,
              population_chunk: int) -> Tuple[int, int, int, int]:
    """The static shape of a population's compiled discovery.

    Args:
        queries_per_genome: (I + L + O) * M.
        population: Genomes per generation.
        population_chunk: -1 the query budgets, 0 the whole population, a positive value that many
            genomes.

    Returns:
        ``(genomes per slice, slices, query rows per block, blocks)``. Slices run in sequence
        (``lax.map``), each vmapped over its genomes; the last slice repeats a genome, dropped
        afterwards. Blocks split a genome's rows when a slice would hold more than
        `QUERY_BLOCK_BUDGET` queries.
    """
    if population_chunk > 0:
        rows = min(population_chunk, population)
    elif population_chunk == 0:
        rows = population
    else:
        rows = max(1, min(population, QUERY_ROW_BUDGET // queries_per_genome))
    slices = -(-population // rows)
    if rows * queries_per_genome <= QUERY_BLOCK_BUDGET:
        return rows, slices, queries_per_genome, 1
    block = max(1, QUERY_BLOCK_BUDGET // rows)
    return rows, slices, block, -(-queries_per_genome // block)


class NodeListModel:
    """The compiled discovery of one node-list configuration and population size.

    Made once per run (`node_list_model`) and reused every generation: the core is one
    ``jax.jit`` over the whole population, with the lattice's arrays as arguments rather than
    constants.
    """

    def __init__(self, cppn_forward: Callable, settings: NodeListSettings,
                 input_coords: Tuple[Tuple[float, float], ...],
                 output_coords: Tuple[Tuple[float, float], ...], population: int):
        self.settings = settings
        self.lattice = node_list_lattice(settings.depth, input_coords, output_coords)
        lattice = self.lattice
        num_inputs = lattice.input_coords.shape[0]
        num_outputs = lattice.output_coords.shape[0]
        num_leaves, num_points = lattice.leaves.shape[0], lattice.points.shape[0]
        sources = num_inputs + num_leaves + num_outputs
        queries = sources * num_points
        rows, slices, block, blocks = call_plan(queries, population, settings.population_chunk)
        self.plan = (rows, slices, block, blocks)
        padding = np.repeat(lattice.rows[-1:], block * blocks - queries, axis=0)
        hh_mask = hidden_class_mask(settings.depth, *settings.classes)
        self._arrays = tuple(jnp.asarray(a) for a in (
            np.concatenate([lattice.rows, padding]), np.float32(SPARSIFY_THRESHOLD),
            lattice.leaf_index, lattice.neighbor_index, lattice.input_mask, hh_mask,
            lattice.output_mask))

        def genome(state, nodes, conns, conn_attrs, node_attrs, query_rows, sparsify_threshold,
                   leaf_index, neighbor_index, input_mask, hidden_mask, output_mask):
            cppn = (nodes, conns, conn_attrs, node_attrs)

            def evaluate(block_rows):
                return jax.vmap(lambda row: cppn_forward(state, cppn, row))(block_rows)

            if blocks == 1:
                outputs = evaluate(query_rows)
            else:
                outputs = lax.map(evaluate, query_rows.reshape(blocks, block, -1))
                outputs = outputs.reshape(blocks * block, -1)
            raw = outputs[:queries, 0].reshape(sources, num_points)
            return discover_genome(raw, sparsify_threshold, leaf_index, neighbor_index,
                                   input_mask, hidden_mask, output_mask, num_inputs, num_leaves,
                                   settings.max_weight, settings.band_threshold, settings.rounds)

        per_genome = jax.vmap(genome, in_axes=(None, 0, 0, 0, 0) + (None,) * 7)

        def core(state, nodes, conns, conn_attrs, node_attrs, *arrays):
            take = jnp.minimum(jnp.arange(slices * rows), population - 1)
            parts = tuple(part[take].reshape((slices, rows) + part.shape[1:])
                          for part in (nodes, conns, conn_attrs, node_attrs))
            if slices == 1:
                out = per_genome(state, *(part[0] for part in parts), *arrays)
            else:
                out = lax.map(lambda slice_parts: per_genome(state, *slice_parts, *arrays), parts)
                out = tuple(x.reshape((slices * rows,) + x.shape[2:]) for x in out)
            return tuple(x[:population] for x in out)

        self._core = jax.jit(core)

    def discover(self, state: Any, cppns_transformed: Sequence[jnp.ndarray]) -> NodeListNetworks:
        """The population's networks.

        Args:
            state: TensorNEAT state, for the CPPN forward.
            cppns_transformed: The four transformed CPPN arrays, population first.

        Returns:
            `NodeListNetworks`.
        """
        return NodeListNetworks(*self._core(state, *cppns_transformed, *self._arrays))


_MODELS: Dict[Tuple[Any, ...], NodeListModel] = {}
_MODELS_LOCK = threading.Lock()


def node_list_model(cppn_forward: Callable, settings: NodeListSettings,
                    input_coords: Sequence[Sequence[float]],
                    output_coords: Sequence[Sequence[float]], population: int) -> NodeListModel:
    """The `NodeListModel` of a CPPN forward, configuration and population size, made once.

    Keyed like the engine's per-genome discovery cores (the forward's identity and every static
    value), so a run compiles its discovery once.
    """
    inputs = tuple(tuple(float(c) for c in xy) for xy in input_coords)
    outputs = tuple(tuple(float(c) for c in xy) for xy in output_coords)
    key = (id(cppn_forward), settings, inputs, outputs, int(population))
    with _MODELS_LOCK:
        model = _MODELS.get(key)
        if model is None:
            model = NodeListModel(cppn_forward, settings, inputs, outputs, int(population))
            _MODELS[key] = model
    return model


def genome_connections(lattice: NodeListLattice, W_in: Any, W_hh: Any,
                       W_out: Any) -> Set[Tuple[np.float32, ...]]:
    """One genome's connections as RN listed them: ``(x1, y1, x2, y2, weight)`` float32 tuples.

    Args:
        lattice: The configuration's lattice.
        W_in, W_hh, W_out: The genome's three blocks, (I, L), (L, L) and (L, O).

    Returns:
        The set of connections, one per nonzero entry.
    """
    leaves = lattice.leaves
    blocks = ((lattice.input_coords, leaves, np.asarray(W_in)),
              (leaves, leaves, np.asarray(W_hh)),
              (leaves, lattice.output_coords, np.asarray(W_out)))
    out = set()
    for sources, targets, weights in blocks:
        for s, t in zip(*np.nonzero(weights)):
            out.add(tuple(np.float32(v) for v in (*sources[s], *targets[t], weights[s, t])))
    return out
