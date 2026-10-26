"""HMR's node-list recurrent model as recurrence.network_model 'node_list'.

HMR's recurrent-network variant (RN) discovered each genome's network on the leaves of a complete
quadtree and settled it as a node list, per source and per genome in Python. node_list.py computes
the same networks for a whole population in one compiled call. The tests here check the frontier
against RN's phase-2 rule written out on index sets, the forward without hidden connections
against the grid model's arithmetic, the engine's generation against its pieces (the networks,
slicing, the champion record, the compiles), the switch itself (the grid model scores another
population) and every refusal. The networks and the fitness are compared with RN's own
populations, the cleaning and the forward with RN's own functions, and the recipe of GR, RN's
GPU-resident variant, with GR, on HMR's values recorded in the research repository
(hmr_reference.hmr_value, CPU only): this package does not include the HMR engine. The port runs
on states it rebuilds itself (EMR's initialize of a configuration's twin is RN's generation-0
state, EMR's tell with RN's fitness vector RN's next one), and each rebuilt population is checked
against the digest RN recorded for it.
"""

import copy
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import emr_hyperneat.emrhyperneat as E
from conftest import GPU_TRAJECTORY_RTOL, EMRHyperNEAT, XORProblem, exact_backend
from emr_hyperneat import node_list
from emr_hyperneat.emrhyperneat_base import MultiGPUStrategy
from hmr_reference import digest, hmr_value, live  # type: ignore[import-not-found]

INPUTS = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]
OUTPUTS = [(0.0, 1.0)]
CLASSES = {
    'hidden_only': (False, False, False),
    'with_backward': (True, False, False),
    'with_lateral': (False, True, False),
    'with_self': (False, False, True),
    'full_recurrent': (True, True, True),
}
#: Fitness against RN on order-stable genomes: RN summed in Python set order, the port in a fixed
#: one (measured at most 2.4e-7)
FITNESS_TOLERANCE = 1e-6
#: A genome whose RN fitness moves by more than this when its connection list is shuffled depends
#: on RN's set order
ORDER_STABLE = 1e-6
#: The seed of every RN and GR run
SEED = 42


def _classes(name):
    return dict(zip(('allow_backward', 'allow_lateral', 'allow_self_loops'), CLASSES[name]))


def _rn_config(depth, pop, preset='full_recurrent', level=2, steps=None, max_weight=8.0,
               band=0.3):
    """An RN configuration (HMR-shaped); level 0 is RN's node list without phase 2."""
    recurrence = {'enabled': True, 'allow_hidden_to_hidden': True, 'iteration_level': level,
                  **_classes(preset)}
    if steps is not None:
        recurrence['activate_time'] = steps
    return {'algorithm_params': {'hmrhyperneat': {
        'population_size': pop,
        'substrate': {'input_coords': INPUTS, 'output_coords': OUTPUTS},
        'hmr_hyperneat': {'max_depth': depth, 'band_threshold': band, 'max_weight': max_weight,
                          'recurrence': recurrence},
    }}}


def node_list_twin(rn_config):
    """The node-list twin of an RN configuration (the recipe of emr-configuration-checks.md)."""
    params = copy.deepcopy(rn_config['algorithm_params'])
    block = params.pop('hmrhyperneat')
    emr = block.pop('hmr_hyperneat')
    recurrence = dict(emr['recurrence'], network_model='node_list')
    if recurrence['iteration_level'] == 0:
        recurrence = {'network_model': 'node_list'}
    emr['recurrence'] = recurrence
    block['emr_hyperneat'] = emr
    params['emrhyperneat'] = block
    return {'algorithm_params': params}


def _node_list_config(depth=2, pop=12, recurrence=None, **emr):
    """A node-list configuration in EMR's shape."""
    block = {'max_depth': depth, 'band_threshold': 0.3, 'max_weight': 8.0,
             'recurrence': {'network_model': 'node_list', 'preset': 'full_recurrent',
                            'iteration_level': 2, **(recurrence or {})}}
    block.update(emr)
    return {'algorithm_params': {'emrhyperneat': {
        'population_size': pop,
        'substrate': {'input_coords': INPUTS, 'output_coords': OUTPUTS},
        'emr_hyperneat': block,
    }}}


def _evolving(config, elitism=2):
    """``config`` with neat_selection.genome_elitism set to ``elitism``. At the default 15 a
    population of 15 or fewer is copied unmutated, so its later generations repeat generation 0's
    genomes."""
    config['algorithm_params']['emrhyperneat']['neat_selection'] = {'genome_elitism': elitism}
    return config


def connection_digest(connections):
    """The sha256 of a connection set as its canonical array: (x1, y1, x2, y2, weight) rows in
    float32, -0.0 as 0.0, sorted by column. hmr_live.py hashes RN's sets with its twin of this
    function."""
    rows = np.asarray(list(connections), dtype=np.float32).reshape(-1, 5) + np.float32(0.0)
    rows = rows[np.lexsort(rows.T[::-1])]
    return hashlib.sha256(rows.tobytes()).digest()


def _population(state):
    """What the port reads of a state: the population it asks (pop_nodes, pop_conns)."""
    return np.asarray(state.pop_nodes), np.asarray(state.pop_conns)


def rn_run(config, generations, neighbour_scale='rn'):
    """RN's generations through the seam (hmr_live.rn_node_list_run), with the arguments they come
    from. ``neighbour_scale`` 'configured' patches RN to sparsify the band's neighbours at the
    configured max_weight (R2's control)."""
    args = (config, XORProblem(), SEED, generations, neighbour_scale, ORDER_STABLE)
    return args, hmr_value('node_list.rn_run', lambda: live('rn_node_list_run', *args), inputs=args)


def _port_generation(twin, problem):
    """The node-list engine's first generation on the twin's own generation-0 state, which is RN's
    (research checks it against RN's own states)."""
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(twin), problem, seed=SEED)
    algorithm.run_generation(state, problem)
    return algorithm


def _port_graph(lattice, networks, genome):
    return node_list.genome_connections(lattice, networks.W_in[genome], networks.W_hh[genome],
                                        networks.W_out[genome])


def _compare(args, rn, twin):
    """Every RN generation against the port, on the states rebuilt from the twin's initialize and
    RN's fitness; returns the counts the callers assert on.

    Each rebuilt population must be the one RN evaluated (its digest). Off the CPU, which the
    recorded values do not cover, the networks are compared with a tolerance on RN's sets
    themselves, from research's live run."""
    problem = args[1]
    counts = dict(compared=0, invented=0, invented_nonempty=0, fitness_compared=0,
                  order_unstable=0, capped=0)
    depth = twin['algorithm_params']['emrhyperneat']['emr_hyperneat']['max_depth']
    lattice = node_list.node_list_lattice(depth, tuple(INPUTS), tuple(OUTPUTS))
    rn_graphs = None if exact_backend() else live('rn_node_list_graphs', *args)
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(twin), problem, seed=SEED)
    for generation, population in enumerate(rn['populations']):
        assert digest(_population(state)) == population, (
            f"generation {generation}: the rebuilt population is not the one RN evaluated")
        algorithm.run_generation(state, problem)
        networks = algorithm._cached_node_list
        port_fitness = np.asarray(algorithm._cached_fitnesses)
        rn_fitness = rn['fitness'][generation]
        for g in range(rn_fitness.shape[0]):
            port_graph = _port_graph(lattice, networks, g)
            size = int(rn['connections'][generation][g])
            if rn['invented'][generation][g]:
                counts['invented'] += 1
                counts['invented_nonempty'] += size > 0
                assert not port_graph, f"genome {g}: RN invented connections, the port's is empty"
                continue
            counts['compared'] += 1
            if rn_graphs is None:
                assert connection_digest(port_graph) == bytes(rn['graphs'][generation][g]), (
                    f"genome {g}: the port's {len(port_graph)} connections are not RN's {size}")
            else:
                rn_graph = rn_graphs[generation][g]
                assert len(port_graph ^ rn_graph) <= max(1, len(rn_graph) // 1000)
            if size > node_list.RN_CONNECTION_CAP:
                counts['capped'] += 1
                continue
            if not rn['stable'][generation][g]:
                counts['order_unstable'] += 1
                continue
            counts['fitness_compared'] += 1
            assert abs(port_fitness[g] - rn_fitness[g]) <= FITNESS_TOLERANCE, g
        state = algorithm._compiled_tell(state, jnp.asarray(rn_fitness))
    return counts


class TestAgainstLiveRN:
    """RN's populations, genome by genome: networks bit for bit, fitness within
    FITNESS_TOLERANCE."""

    @pytest.mark.parametrize('depth, preset, level, steps', [
        (1, 'with_lateral', 2, None),
        (2, 'hidden_only', 2, None),
        (2, 'full_recurrent', 2, 20),
        (2, 'with_self', 2, None),
        (2, 'with_backward', 2, None),
        (2, 'full_recurrent', 1, 20),
        (2, 'full_recurrent', 3, 20),
        (2, 'hidden_only', 0, None),
    ])
    def test_rn_populations(self, depth, preset, level, steps):
        config = _rn_config(depth, 20, preset, level, steps)
        args, rn = rn_run(config, 2)
        counts = _compare(args, rn, node_list_twin(config))
        assert counts['compared'] > 0 and counts['fitness_compared'] > 0, counts

    def test_invented_connections_are_not_ported(self):
        """R3's negative control. At band 3 RN finds no phase-1 or phase-3 connection for most
        genomes and invents them: (0, 0) fed by every input at 0.5 (RN:3502-3511) and the nearest
        node to the output at 0.5 (RN:3573-3584). The port leaves those genomes empty, the
        constant 0.5 network, so their fitness differs."""
        config = _rn_config(2, 20, 'full_recurrent', 2, 20, band=3.0)
        args, rn = rn_run(config, 1)
        counts = _compare(args, rn, node_list_twin(config))
        assert counts['invented'] >= 10 and counts['invented_nonempty'] > 0, counts
        port = np.asarray(_port_generation(node_list_twin(config), args[1])._cached_fitnesses)
        invented = np.flatnonzero(rn['invented'][0])
        assert np.all(port[invented] == np.float32(0.75))
        assert np.any(rn['fitness'][0][invented] != port[invented])

    def test_neighbour_scale(self):
        """R2's negative control. RN scaled the band's neighbour weights by 8 whatever max_weight
        said (RN:2334): at max_weight 3 its networks differ from the port's. Patching RN's
        neighbour sparsification to the configured max_weight makes them equal, which isolates
        the cause."""
        config = _rn_config(2, 20, 'full_recurrent', 2, 20, max_weight=3.0)
        twin = node_list_twin(config)
        args, rn = rn_run(config, 1)
        networks = _port_generation(twin, args[1])._cached_node_list
        lattice = node_list.node_list_lattice(2, tuple(INPUTS), tuple(OUTPUTS))
        differ = sum(connection_digest(_port_graph(lattice, networks, g)) != bytes(digest_g)
                     for g, digest_g in enumerate(rn['graphs'][0]))
        assert differ >= 15
        args, fixed = rn_run(config, 1, neighbour_scale='configured')
        counts = _compare(args, fixed, twin)
        assert counts['compared'] > 0, counts

    @pytest.mark.slow
    @pytest.mark.parametrize('preset, steps', [('hidden_only', None), ('full_recurrent', 20)])
    def test_depth_three(self, preset, steps):
        """Depth 3, where RN's 500 cap truncates most genomes (R4's negative control): the networks
        are compared before the cap, and some genome RN evaluated on 500 of its connections."""
        config = _rn_config(3, 20, preset, 2, steps)
        args, rn = rn_run(config, 2)
        counts = _compare(args, rn, node_list_twin(config))
        assert counts['compared'] > 0 and counts['capped'] > 0, counts
        assert any(np.any(reported > node_list.RN_CONNECTION_CAP) for reported in rn['reported'])


class TestPieces:
    """The frontier, the cleaning and the forward against RN's rules on built graphs."""

    @staticmethod
    def _rn_rounds(h1, m_hh, rounds):
        """RN's phase 2 on index sets: its unexplored set after a round is the found set minus the
        set it explored."""
        hidden, unexplored, connections = set(h1), set(h1), set()
        for _ in range(rounds):
            new = set()
            for source in unexplored:
                for target in np.nonzero(m_hh[source])[0]:
                    connections.add((source, int(target)))
                    if int(target) not in hidden:
                        new.add(int(target))
            hidden |= new
            unexplored = hidden - unexplored
        return hidden, connections

    @pytest.mark.parametrize('seed', range(6))
    @pytest.mark.parametrize('rounds', [0, 1, 2, 3, 5])
    def test_frontier_is_rns_rounds(self, seed, rounds):
        rng = np.random.default_rng(seed)
        leaves = 24
        m_hh = rng.random((leaves, leaves)) < 0.08
        c1 = rng.random((3, leaves)) < 0.1
        found, explored = node_list.explore(jnp.asarray(c1), jnp.asarray(m_hh), rounds)
        h1 = set(np.nonzero(c1.any(axis=0))[0].tolist())
        hidden, connections = self._rn_rounds(h1, m_hh, rounds)
        assert set(np.nonzero(np.asarray(found))[0].tolist()) == hidden
        ours = zip(*np.nonzero(m_hh & np.asarray(explored)[:, None]))
        assert {(int(s), int(t)) for s, t in ours} == connections

    @staticmethod
    def _as_coordinates(leaves, c1, c2, c3):
        rows = [[*INPUTS[i], *leaves[j], 1.0] for i, j in zip(*np.nonzero(c1))]
        rows += [[*leaves[s], *leaves[t], 1.0] for s, t in zip(*np.nonzero(c2))]
        rows += [[*leaves[j], *OUTPUTS[o], 1.0] for j, o in zip(*np.nonzero(c3))]
        return np.asarray(rows, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def _rn_cleaning(rows):
        """The rows RN's cleaning keeps (hmr_live.rn_clean_connections), through the seam."""
        args = (rows, np.asarray(INPUTS, np.float32), np.asarray(OUTPUTS, np.float32))
        return hmr_value('node_list.rn_cleaning', lambda: live('rn_clean_connections', *args),
                         inputs=args)

    @pytest.mark.parametrize('seed', range(8))
    def test_cleaning_is_rns(self, seed):
        """On graphs RN cleans within its 50 iterations, the port keeps exactly RN's
        connections."""
        leaves = node_list.node_list_lattice(2, tuple(INPUTS), tuple(OUTPUTS)).leaves
        rng = np.random.default_rng(seed)
        c1 = rng.random((3, 16)) < 0.15
        m_hh = rng.random((16, 16)) < 0.1
        found, explored = node_list.explore(jnp.asarray(c1), jnp.asarray(m_hh), 2)
        c2 = m_hh & np.asarray(explored)[:, None]
        c3 = rng.random((16, 1)) < 0.3
        k1, k2, k3, _, _ = node_list.clean(jnp.asarray(c1), jnp.asarray(c2), jnp.asarray(c3),
                                           found)
        kept = self._rn_cleaning(self._as_coordinates(leaves, c1, c2, c3))
        ours = self._as_coordinates(leaves, np.asarray(k1), np.asarray(k2), np.asarray(k3))
        assert {tuple(r) for r in ours} == {tuple(r) for r in kept}

    def test_cleaning_reaches_its_fixpoint(self):
        """R6's negative control: a 60-leaf chain from the input to the output. RN's cleaning stops
        after 50 iterations each way (RN:2829) and keeps only a dangling middle of it; the port
        keeps the chain."""
        leaves = node_list.node_list_lattice(4, tuple(INPUTS), tuple(OUTPUTS)).leaves
        chain = 60
        c1 = np.zeros((3, 256), bool)
        c1[0, 0] = True
        c2 = np.zeros((256, 256), bool)
        c2[np.arange(chain - 1), np.arange(1, chain)] = True
        c3 = np.zeros((256, 1), bool)
        c3[chain - 1, 0] = True
        found = np.zeros(256, bool)
        found[:chain] = True
        k1, k2, k3, _, _ = node_list.clean(*(jnp.asarray(a) for a in (c1, c2, c3, found)))
        assert int(np.sum(k1) + np.sum(k2) + np.sum(k3)) == chain + 1
        kept = self._rn_cleaning(self._as_coordinates(leaves, c1, c2, c3))
        assert len(kept) < chain + 1
        assert not np.any(np.all(np.isclose(kept[:, :2], INPUTS[0]), axis=1))

    def test_forward_is_rns_rule(self):
        """The forward against a float64 trace of RN's synchronous update, and RN's own forward,
        on a network with a self-loop, a lateral pair and a backward connection."""
        W_in = np.zeros((3, 4), np.float32)
        W_in[0, 0], W_in[2, 1] = 1.5, -2.0
        W_hh = np.zeros((4, 4), np.float32)
        W_hh[0, 0], W_hh[0, 1], W_hh[1, 0] = 0.7, 1.2, -0.9
        W_hh[1, 2], W_hh[2, 3], W_hh[3, 1] = 2.5, -1.1, 0.4
        W_out = np.zeros((4, 1), np.float32)
        W_out[3, 0], W_out[2, 0] = 1.7, -0.6
        inputs = np.asarray([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)
        steps = 9
        blocks = tuple(jnp.asarray(w)[None] for w in (W_in, W_hh, W_out))
        ours = np.asarray(jax.vmap(E._node_list_forward(steps), in_axes=(None, 0, 0, 0))(
            jnp.asarray(inputs), *blocks))[0]
        # RN's rule in float64: all nodes at once from the last step, inputs clamped, outputs
        # linear
        values = np.zeros((4, 3 + 4 + 1))
        values[:, :3] = inputs
        for _ in range(steps):
            new = np.zeros_like(values)
            new[:, :3] = inputs
            new[:, 3:7] = np.tanh(values[:, :3] @ W_in + values[:, 3:7] @ W_hh)
            new[:, 7:] = values[:, 3:7] @ W_out
            values = new
        np.testing.assert_allclose(ours, 1.0 / (1.0 + np.exp(-values[:, 7:])), atol=1e-6)
        conns = [[s, 3 + t, W_in[s, t]] for s, t in zip(*np.nonzero(W_in))]
        conns += [[3 + s, 3 + t, W_hh[s, t]] for s, t in zip(*np.nonzero(W_hh))]
        conns += [[3 + s, 7, W_out[s, 0]] for s in np.nonzero(W_out[:, 0])[0]]
        padded = np.full((20, 3), np.nan, np.float32)
        padded[:len(conns)] = conns
        args = (np.arange(8.0, dtype=np.float32)[:, None], padded, np.arange(20) < len(conns),
                inputs, 3, 1, steps)
        theirs = hmr_value('node_list.rn_forward', lambda: live('rn_forward_padded', *args),
                           inputs=args)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_forward_without_hidden_connections_is_the_grids(self):
        """Without leaf-to-leaf weights the forward computes the grid model's feed-forward
        arithmetic (bit for bit on CPU), whatever the steps."""
        rng = np.random.default_rng(3)
        W_in = jnp.asarray(rng.normal(size=(3, 16)).astype(np.float32))
        W_out = jnp.asarray(rng.normal(size=(16, 1)).astype(np.float32))
        inputs = jnp.asarray(rng.random((4, 3)).astype(np.float32))
        grid = np.asarray(jax.vmap(lambda a, b: E.dense_substrate_forward(inputs, a, b))(
            W_in[None], W_out[None]))
        for steps in (2, 5, 20):
            ours = np.asarray(jax.vmap(E._node_list_forward(steps), in_axes=(None, 0, 0, 0))(
                inputs, W_in[None], jnp.zeros((1, 16, 16)), W_out[None]))
            if exact_backend():
                np.testing.assert_array_equal(ours, grid)
            else:
                np.testing.assert_allclose(ours, grid, rtol=GPU_TRAJECTORY_RTOL)


class TestGeneration:
    """The engine's node-list generation: its pieces, its record, its compiles."""

    def _run(self, config, generations, seed=5):
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=seed)
        trajectory = []
        for _ in range(generations):
            state, metrics = algorithm.run_generation(state, problem)
            trajectory.append(np.asarray(algorithm._cached_fitnesses))
        return algorithm, state, metrics, trajectory

    def test_generation_is_the_pipeline(self):
        """run_generation's fitness is node_list_model's networks scored by _node_list_forward,
        bit for bit on CPU; the grid's caches are cleared."""
        config = _node_list_config(depth=2, pop=12)
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=5)
        cppns = algorithm._compiled_transform_batch(state, algorithm._compiled_ask(state))
        algorithm.run_generation(state, problem)
        settings = algorithm._node_list_settings
        model = node_list.node_list_model(algorithm._jitted_cppn_forward, settings, INPUTS,
                                          OUTPUTS, 12)
        networks = model.discover(state, cppns)
        forward = E._node_list_forward(settings.activate_time)
        expected = jax.vmap(lambda *w: E._eval_single_fitness(
            forward(algorithm._cached_inputs, *w), algorithm._cached_targets, 'mse'))(
            networks.W_in, networks.W_hh, networks.W_out)
        np.testing.assert_array_equal(np.asarray(algorithm._cached_fitnesses),
                                      np.asarray(expected))
        assert algorithm._cached_W1 is None and algorithm._cached_sparse_hh is None

    def test_slicing_changes_nothing(self):
        """The population in one call, in slices of 5 and by the query budget: the same networks
        and fitness. Elitism 2, so that generation 2 scores mutated genomes."""
        results = []
        for chunk in (0, 5, -1):
            config = _evolving(_node_list_config(depth=2, pop=12, population_chunk_size=chunk))
            algorithm, _, _, trajectory = self._run(config, 2)
            results.append((trajectory, algorithm._cached_node_list))
        for trajectory, networks in results[1:]:
            for ours, theirs in zip(trajectory, results[0][0]):
                np.testing.assert_array_equal(ours, theirs)
            for ours, theirs in zip(networks, results[0][1]):
                np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))

    def test_record_and_evaluate_on_data(self):
        """evaluate_on_data scores the champion with its generation's forward: its training
        fitness, bit for bit on CPU; the substrate statistics count the leaves and the three
        blocks. Elitism 2, so that the champion is found among mutated genomes, not in slot 0 of an
        unmutated copy of generation 0."""
        algorithm, state, metrics, _ = self._run(_evolving(_node_list_config(depth=2, pop=12)), 2)
        result = algorithm.evaluate_on_data(state, algorithm._cached_inputs,
                                            algorithm._cached_targets)
        assert result['fitness'] == result['training_fitness']
        assert result['total_positions'] == 16 and result['network_model'] == 'node_list'
        assert result['total_connections'] == (result['input_connections']
                                               + result['output_connections']
                                               + result['hidden_connections'])
        assert metrics.custom_metrics['method'] == 'node_list'
        assert algorithm.get_recurrence_mode() == 'node_list'
        assert algorithm.evaluate_best_all_modes()['mse'] == result['fitness']

    def test_run_until_threshold(self):
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        config = algorithm.create_config(_node_list_config(depth=1, pop=10))
        state = algorithm.initialize(config, problem, seed=1)
        result = algorithm.run_until_threshold(state, problem, target_fitness=2.0,
                                               max_generations=2)
        assert result['generations'] == 2 and result['route'] == 'SINGLE_GPU'

    def test_compiles_once(self):
        """From generation 2 on a node-list generation compiles nothing: the core is kept per
        configuration, the forward per step count and its scan step in the engine's registry.
        (Generation 1 compiles again once, as on the grid path: the state's argument types change
        after the first tell.) Elitism 2, so that the networks change from generation to generation:
        at the default 15 a population of 12 is copied unmutated and nothing could trigger a
        compile."""
        compiles = {'count': 0, 'active': False}

        def listener(event, duration, **kwargs):
            if compiles['active'] and event == '/jax/core/compile/backend_compile_duration':
                compiles['count'] += 1

        jax.monitoring.register_event_duration_secs_listener(listener)
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        config = algorithm.create_config(_evolving(_node_list_config(depth=2, pop=12)))
        state = algorithm.initialize(config, problem, seed=2)
        per_generation = []
        for _ in range(4):
            compiles['count'], compiles['active'] = 0, True
            state, _ = algorithm.run_generation(state, problem)
            compiles['active'] = False
            per_generation.append(compiles['count'])
        assert per_generation[2:] == [0, 0], per_generation

    def test_the_switch_is_live(self):
        """Negative control: the same configuration on the grid model scores another
        population."""
        _, _, _, node_list_trajectory = self._run(_node_list_config(depth=2, pop=12), 1)
        grid = _node_list_config(depth=2, pop=12)
        del grid['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['network_model']
        _, _, _, grid_trajectory = self._run(grid, 1)
        assert not np.array_equal(node_list_trajectory[0], grid_trajectory[0])


class TestChecks:
    """What the node list cannot apply raises at create_config (or at the call)."""

    @pytest.mark.parametrize(
        'change, match',
        [
            ({'variance_threshold': 0.03}, 'emr_hyperneat.variance_threshold'),
            ({'division_threshold': 0.5}, 'emr_hyperneat.division_threshold'),
            ({'initial_depth': 1}, 'emr_hyperneat.initial_depth'),
            ({'variance_rule': 'legacy'}, 'emr_hyperneat.variance_rule'),
            ({'weight_threshold': 0.1}, 'emr_hyperneat.weight_threshold'),
            ({'max_depth': 0}, 'max_depth 0'),
            ({'max_depth': 6}, 'max_depth 6'),
            ({'max_weight': -1.0}, 'max_weight'),
            ({'population_chunk_size': -3}, 'population_chunk_size'),
            (
                {'dynamic_functions': {'mode': 'global', 'hidden_activation': 'sin'}},
                'dynamic_functions.mode',
            ),
            ({'aggregation': {'mode': 'global', 'global_function': 'max'}}, 'aggregation.mode'),
            ({'neuromodulation': {'mode': 'static_gating'}}, 'neuromodulation.mode'),
            ({'leo': {'enabled': True}}, 'LEO'),
            ({'geometry_seeding': {'enabled': True}}, 'geometry seeding'),
            ({'locality_radius': 0.5}, 'locality_radius'),
            ({'execution': {'mode': 'streaming'}}, 'execution block'),
            ({'execution': {'compaction_bucket': 'auto'}}, 'execution block'),
            (
                {'execution': {'generation_loop': 'while_loop'}},
                "execution.generation_loop 'while_loop'",
            ),
            ({'enable_streaming': True}, 'enable_streaming'),
            ({'multi_gpu_strategy': 'EVAL_ONLY_PARALLEL'}, 'strategy EVAL_ONLY_PARALLEL'),
        ],
    )
    def test_emr_block(self, change, match):
        config = _node_list_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat'].update(change)
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('change, match', [
        *[({key: value}, f'recurrence.{key}') for key, value in (
            ('max_connections', 500), ('max_sparse_conns', 1000),
            ('multi_hop_algorithm', 'fori_loop'), ('hop_decay_factor', 0.5),
            ('multi_hop_second_factor', 'masked'), ('hh_refresh_interval', 3),
            ('use_dense_discovery', False), ('hh_discovery_mode', 'per_genome'))],
        ({'hh_cache_enabled': True}, 'hh_cache_enabled'),
        ({'settling_rule': 'hmr'}, "settling_rule 'hmr'"),
        ({'iteration_level': 0}, 'iteration_level 0 with hidden-to-hidden'),
        ({'activate_time': 1}, 'activate_time'),
        ({'network_model': 'quadtree'}, 'network_model must be one of'),
        ({'node_list': {'max_connections': 500}}, 'has no settings block'),
    ])
    def test_recurrence_block(self, change, match):
        config = _node_list_config(recurrence=change)
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_rounds_without_hidden_connections(self):
        config = _node_list_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = {
            'network_model': 'node_list', 'iteration_level': 2}
        with pytest.raises(ValueError, match='without hidden-to-hidden connections'):
            EMRHyperNEAT().create_config(config)

    def test_accepted_values(self):
        """What describes the node list's own behavior is accepted."""
        config = _node_list_config(recurrence={'hh_cache_enabled': False, 'settling_rule': 'emr',
                                               'eval_chunk_size': 4},
                                   execution={'mode': 'standard', 'generation_loop': 'python'},
                                   population_chunk_size=0)
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        settings = algorithm._node_list_settings
        assert settings.rounds == 2 and settings.classes == (True,) * 3
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['recurrence'] = {'network_model': 'node_list'}
        algorithm.create_config(config)
        assert algorithm._node_list_settings.rounds == 0

    def test_cppn_shape(self):
        config = _node_list_config()
        config['algorithm_params']['emrhyperneat']['cppn'] = {'genome': {'num_outputs': 2}}
        with pytest.raises(ValueError, match='5 inputs and 1 output'):
            EMRHyperNEAT().create_config(config)

    def test_coordinates_on_a_leaf(self):
        config = _node_list_config(depth=2)
        substrate = config['algorithm_params']['emrhyperneat']['substrate']
        substrate['input_coords'] = [(0.25, -0.75), (1.0, -1.0)]
        with pytest.raises(ValueError, match='on a leaf or on each other'):
            EMRHyperNEAT().create_config(config)

    def test_constructor_strategy(self):
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL)
        with pytest.raises(ValueError, match='strategy'):
            algorithm.create_config(_node_list_config())

    def test_calls_that_cannot_apply(self):
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        config = algorithm.create_config(_node_list_config(depth=1, pop=8))
        state = algorithm.initialize(config, problem, seed=0)
        with pytest.raises(ValueError, match="network_model 'node_list'"):
            algorithm.substrate_config()
        with pytest.raises(ValueError, match="network_model 'node_list'"):
            algorithm.set_recurrence_config(E.EMRConfig())
        with pytest.raises(ValueError, match="network_model 'node_list'"):
            algorithm.run_generation_multitask(state, [problem])
        algorithm.strategy = MultiGPUStrategy.STREAMING
        with pytest.raises(ValueError, match='strategy STREAMING'):
            algorithm.run_until_threshold(state, problem, target_fitness=2.0, max_generations=1)
        algorithm.strategy = MultiGPUStrategy.SINGLE_GPU
        algorithm.execution_config.mode = E.ExecutionMode.DEPTH_OFFLOAD
        with pytest.raises(ValueError, match='execution mode other than standard'):
            algorithm.run_generation(state, problem)

    def test_depth_six_keys(self):
        """R5's negative control: at depth 6 RN's padded build keeps none of a chain's connections
        (the cleaned hidden nodes are keyed at 5 decimals, the connections at 6); the node list
        refuses depth 6."""
        args = (INPUTS, OUTPUTS, [5, 6])
        kept = hmr_value('node_list.rn_chain_keys', lambda: live('rn_chain_after_keys', *args),
                         inputs=args)
        assert kept == [3, 0]
        with pytest.raises(ValueError, match='max_depth 6'):
            EMRHyperNEAT().create_config(_node_list_config(depth=6))


GR_THRESHOLD_BAND = float(np.float32(np.tanh(np.float32(0.1))) * np.float32(8.0))


def _gr_config(depth, variance, pop, preset='full_recurrent'):
    return {'algorithm_params': {'hmrhyperneat': {
        'population_size': pop,
        'substrate': {'input_coords': INPUTS, 'output_coords': OUTPUTS},
        'hmr_hyperneat': {'initial_depth': 0, 'max_depth': depth, 'variance_threshold': variance,
                          'band_threshold': 0.3, 'max_weight': 8.0,
                          'recurrence': {'preset': preset}},
    }}}


def gr_recipe(gr_config, band_threshold=GR_THRESHOLD_BAND, max_sparse_conns=None):
    """GR's dense model at depth <= 4 on EMR's default model (the recipe of
    emr-configuration-checks.md): per-genome discovery at level 1 with GR's threshold on the raw
    CPPN output (|raw| > 0.1, as band_threshold max_weight * tanh(0.1)), no h->h cap below P^2,
    HMR's level-0 variance rule and GR's pre-ask split."""
    params = copy.deepcopy(gr_config['algorithm_params'])
    block = params.pop('hmrhyperneat')
    emr = block.pop('hmr_hyperneat')
    positions = (4 ** (emr['max_depth'] + 2) - 4) // 3
    cap = max_sparse_conns or max(10000, positions * positions)
    emr.update(band_threshold=band_threshold, variance_rule='legacy', extra_randkey_split=True,
               recurrence={'enabled': True, 'allow_hidden_to_hidden': True,
                           **_classes('full_recurrent'), 'iteration_level': 1,
                           'activate_time': 20, 'max_sparse_conns': cap})
    block['emr_hyperneat'] = emr
    params['emrhyperneat'] = block
    return {'algorithm_params': params}


def _gr_args(config, generations):
    return config, XORProblem(), SEED, generations


def _gr_fitness(config, generations):
    """GR's fitness vector of each generation as its tell received it (hmr_live.gr_dense_fitness),
    through the seam; every generation ran GR's dense model."""
    args = _gr_args(config, generations)
    gr = hmr_value('node_list.gr_dense', lambda: live('gr_dense_fitness', *args), inputs=args)
    assert gr['methods'] == ['dense_recurrent'] * generations
    return gr['fitness']


def _emr_fitness(config, generations):
    algorithm = EMRHyperNEAT()
    problem = XORProblem()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=SEED)
    out = []
    for _ in range(generations):
        state, _ = algorithm.run_generation(state, problem)
        out.append(np.asarray(algorithm._cached_fitnesses))
    return out


def _agree(ours, theirs):
    for mine, gr in zip(ours, theirs):
        if exact_backend():
            np.testing.assert_array_equal(mine, gr)
        else:
            np.testing.assert_allclose(mine, gr, rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


class TestGRRecipe:
    """GR at depth <= 4 ran a dense W1/W2/W3 grid model, not a node list: its recipe is EMR's
    default model. Each test asks for GR's side first: where no value is recorded (the release on a
    GPU) it skips before EMR's run, which at depth 4 asks more than a 16 GB T4 holds."""

    @pytest.mark.parametrize('variance', [0.0, 0.03])
    def test_depth_two(self, variance):
        config = _gr_config(2, variance, 20)
        theirs = _gr_fitness(config, 3)
        _agree(_emr_fitness(gr_recipe(config), 3), theirs)

    # Population 20: GR hard-codes genome_elitism 15, and a population of 15 or fewer is copied
    # unmutated, so generation 2 would score generation 1's genomes again.
    @pytest.mark.slow
    @pytest.mark.parametrize('depth, variance, pop', [(3, 0.03, 20), (3, 0.0, 20),
                                                      (4, 0.03, 20)])
    def test_depths_three_and_four(self, depth, variance, pop):
        config = _gr_config(depth, variance, pop)
        theirs = _gr_fitness(config, 2)
        _agree(_emr_fitness(gr_recipe(config), 2), theirs)

    def test_configured_band_is_not_grs(self):
        """Negative control: GR ignored band_threshold on W3 (it kept |raw| > 0.1); EMR with GR's
        configured 0.3 keeps more hidden-to-hidden connections and scores another population."""
        config = _gr_config(2, 0.0, 20)
        theirs = _gr_fitness(config, 2)
        ours = _emr_fitness(gr_recipe(config, band_threshold=0.3), 2)
        assert any(not np.array_equal(a, b) for a, b in zip(ours, theirs))

    @pytest.mark.slow
    def test_emr_cap_is_not_grs(self):
        """Negative control: at depth 3 with every position active a genome exceeds EMR's default
        max_sparse_conns (10,000), which GR does not have, and EMR refuses."""
        config = _gr_config(3, 0.0, 10)
        with pytest.raises(ValueError, match=r'recurrence\.max_sparse_conns=10000 cannot hold'):
            _emr_fitness(gr_recipe(config, max_sparse_conns=10000), 2)

    def test_gr_fails_without_full_recurrent(self):
        """GR's dense path takes its steps from the preset's activate_time, None for every preset
        but full_recurrent (its line 5193), and fails there: the recipe needs an explicit
        count."""
        args = _gr_args(_gr_config(2, 0.03, 8, preset='hidden_only'), 1)
        failure = hmr_value('node_list.gr_failure', lambda: live('gr_failure', *args), inputs=args)
        assert failure is not None and failure['error'] == 'ValueError', failure
        assert 'scan got no values' in failure['message']
