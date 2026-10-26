"""The continual-learning API, from HMR's dynamic-functions variant.

The champion of the last generation, putting a genome into a population, the topology freeze, a
genome's fitness on another problem, and a fresh state for another problem. HMR's versions had
defects these tests guard: the champion read from the wrong population, injection over the
champion's slot, a freeze its own tests could not see (they counted every gene slot), and an
evaluator that did not reproduce the training fitness (no variance mask, no per-node functions or
recurrence, interleaved CPPN channels).
"""

import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    ANDProblem,
    DEFAULT_SEED,
    ParityProblem,
    XORProblem,
    create_base_config,
    create_config_with_dynamic_functions,
    create_config_with_neuromodulation,
    create_config_with_recurrence,
)
from emr_hyperneat import EMRHyperNEAT

POP = 20
DEPTH = 2


def _feed_forward():
    return create_base_config(max_depth=DEPTH, population_size=POP)


def _per_node():
    return create_config_with_dynamic_functions(
        "cppn_output_4", max_depth=DEPTH, population_size=POP
    )


def _recurrent():
    return create_config_with_recurrence("full_recurrent", max_depth=DEPTH, population_size=POP,
                                         iteration_level=1, hh_cache_enabled=False)


def _setup(config=None):
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(
        algo.create_config(config or _feed_forward()), problem, seed=DEFAULT_SEED
    )
    return algo, problem, state


def _topologies(state):
    """Each genome's topology: its node keys and connection (in, out) pairs (TensorNEAT marks empty
    slots NaN)."""
    found = []
    for nodes, conns in zip(np.asarray(state.pop_nodes), np.asarray(state.pop_conns)):
        keys = frozenset(nodes[~np.isnan(nodes[:, 0]), 0].tolist())
        pairs = frozenset(map(tuple, conns[~np.isnan(conns[:, 0]), :2].tolist()))
        found.append((keys, pairs))
    return found


class TestChampion:
    def test_the_champion_is_the_recorded_one(self):
        algo, problem, state = _setup()
        with pytest.raises(RuntimeError, match="no evaluated generation"):
            algo.get_best_genome_with_fitness()
        state, metrics = algo.run_generation(state, problem)
        nodes, conns, fitness, index = algo.get_best_genome_with_fitness()
        assert fitness == float(metrics.best_fitness)
        assert fitness == float(np.asarray(algo._cached_fitnesses)[index])
        # TensorNEAT keeps the champion first and unmutated in the next population
        np.testing.assert_array_equal(nodes, np.asarray(state.pop_nodes[0]))
        np.testing.assert_array_equal(conns, np.asarray(state.pop_conns[0]))


    def test_get_best_genome_is_the_champion(self):
        """0.1.0 returned what TensorNEAT's ask returns first, the population's node array."""
        algo, problem, state = _setup()
        with pytest.raises(RuntimeError, match="no evaluated generation"):
            algo.get_best_genome(state)
        state, _ = algo.run_generation(state, problem)
        nodes, conns = algo.get_best_genome(state)
        champion_nodes, champion_conns, _, _ = algo.get_best_genome_with_fitness()
        np.testing.assert_array_equal(np.asarray(nodes), np.asarray(champion_nodes))
        np.testing.assert_array_equal(np.asarray(conns), np.asarray(champion_conns))
        assert np.asarray(nodes).shape == np.asarray(state.pop_nodes[0]).shape


class TestInjection:
    def test_default_slots_are_the_last_and_keep_the_champion(self):
        algo, problem, state = _setup()
        state, _ = algo.run_generation(state, problem)
        donor = (state.pop_nodes[5], state.pop_conns[5])
        injected = algo.inject_genome_into_population(state, *donor, num_copies=2)
        before, after = np.asarray(state.pop_nodes), np.asarray(injected.pop_nodes)
        np.testing.assert_array_equal(after[:POP - 2], before[:POP - 2])
        for slot in (POP - 2, POP - 1):
            np.testing.assert_array_equal(after[slot], np.asarray(donor[0]))
            np.testing.assert_array_equal(
                np.asarray(injected.pop_conns[slot]), np.asarray(donor[1])
            )

    def test_hmr_slots_on_request(self):
        algo, problem, state = _setup()
        donor = (state.pop_nodes[5], state.pop_conns[5])
        injected = algo.inject_genome_into_population(state, *donor, slots=range(1))
        np.testing.assert_array_equal(np.asarray(injected.pop_nodes[0]), np.asarray(donor[0]))

    def test_an_injected_population_scores_the_genome(self):
        """Every slot holding the champion: the next generation scores it, every copy at its
        recorded fitness."""
        algo, problem, state = _setup()
        state, _ = algo.run_generation(state, problem)
        nodes, conns, fitness, _ = algo.get_best_genome_with_fitness()
        state = algo.inject_genome_into_population(state, nodes, conns, slots=range(POP))
        algo.run_generation(state, problem)
        np.testing.assert_array_equal(
            np.asarray(algo._cached_fitnesses), np.full(POP, fitness, dtype=np.float32)
        )

    def test_checks(self):
        algo, problem, state = _setup()
        nodes, conns = state.pop_nodes[3], state.pop_conns[3]
        with pytest.raises(ValueError, match="shaped"):
            algo.inject_genome_into_population(state, nodes[:-1], conns)
        with pytest.raises(ValueError, match="node keys"):
            algo.inject_genome_into_population(state, nodes.at[0, 0].set(99.0), conns)
        with pytest.raises(ValueError, match="slots must be distinct"):
            algo.inject_genome_into_population(state, nodes, conns, slots=[POP])
        with pytest.raises(ValueError, match="slots must be distinct"):
            algo.inject_genome_into_population(state, nodes, conns, slots=[1, 1])
        with pytest.raises(ValueError, match="num_copies"):
            algo.inject_genome_into_population(state, nodes, conns, num_copies=POP + 1)

    def test_the_hh_cache_is_refused(self):
        config = create_config_with_recurrence("hidden_only", max_depth=DEPTH, population_size=POP,
                                               hh_cache_enabled=True)
        algo, problem, state = _setup(config)
        with pytest.raises(ValueError, match="h->h cache"):
            algo.inject_genome_into_population(state, state.pop_nodes[1], state.pop_conns[1])


class TestFreeze:
    def test_frozen_topologies_stay_among_the_frozen_ones(self):
        algo, problem, state = _setup()
        state, _ = algo.run_generation(state, problem)
        state = algo.freeze_topology(state)
        assert algo.is_topology_frozen(state)
        allowed = set(_topologies(state))
        for _ in range(4):
            state, _ = algo.run_generation(state, problem)
            assert set(_topologies(state)) <= allowed

    def test_unfrozen_control_grows_new_topologies(self):
        algo, problem, state = _setup()
        state, _ = algo.run_generation(state, problem)
        allowed = set(_topologies(state))
        grown = False
        for _ in range(4):
            state, _ = algo.run_generation(state, problem)
            grown = grown or not set(_topologies(state)) <= allowed
        assert grown

    def test_unfreeze_restores_the_rates_and_growth(self):
        algo, problem, state = _setup()
        mutation = algo.neat_algo.genome.mutation
        rates = {name: getattr(mutation, name) for name in algo.STRUCTURAL_MUTATIONS}
        state, _ = algo.run_generation(state, problem)
        algo.freeze_topology(state)
        assert all(getattr(mutation, name) == 0 for name in algo.STRUCTURAL_MUTATIONS)
        algo.unfreeze_topology(state)
        assert not algo.is_topology_frozen(state)
        assert {name: getattr(mutation, name) for name in algo.STRUCTURAL_MUTATIONS} == rates
        allowed = set(_topologies(state))
        grown = False
        for _ in range(4):
            state, _ = algo.run_generation(state, problem)
            grown = grown or not set(_topologies(state)) <= allowed
        assert grown

    def test_create_config_ends_a_freeze(self):
        algo, problem, state = _setup()
        algo.freeze_topology(state)
        algo.create_config(_feed_forward())
        assert not algo.is_topology_frozen()
        assert all(
            getattr(algo.neat_algo.genome.mutation, name) > 0 for name in algo.STRUCTURAL_MUTATIONS
        )


class TestEvaluateGenome:

    @pytest.mark.parametrize(
        "make, exact", [(_feed_forward, True), (_per_node, True), (_recurrent, False)]
    )
    def test_the_training_problem_gives_the_recorded_fitness(self, make, exact):
        algo, problem, state = _setup(make())
        state, _ = algo.run_generation(state, problem)
        nodes, conns, fitness, _ = algo.get_best_genome_with_fitness()
        scored = algo.evaluate_genome_on_problem(state, nodes, conns, problem)
        if exact:
            assert scored == fitness
        else:  # the loop evaluates a recurrent population on its union of positions
            assert scored == pytest.approx(fitness, abs=1e-6)

    def test_another_problem(self):
        algo, problem, state = _setup()
        state, _ = algo.run_generation(state, problem)
        nodes, conns, fitness, _ = algo.get_best_genome_with_fitness()
        other = algo.evaluate_genome_on_problem(state, nodes, conns, ANDProblem())
        assert np.isfinite(other) and other != fitness

    def test_another_input_count_needs_coordinates(self):
        algo, problem, state = _setup()
        nodes, conns = state.pop_nodes[0], state.pop_conns[0]
        parity = ParityProblem(n_bits=3)
        with pytest.raises(ValueError, match="pass input_coords"):
            algo.evaluate_genome_on_problem(state, nodes, conns, parity)
        coords = [(-1.0, -1.0), (-0.33, -1.0), (0.33, -1.0), (1.0, -1.0)]
        assert np.isfinite(
            algo.evaluate_genome_on_problem(state, nodes, conns, parity, input_coords=coords)
        )

    def test_what_the_substrate_api_does_not_describe_is_refused(self):
        config = create_config_with_neuromodulation(
            "true_neuromodulation_4nt",
            recurrence_preset="feedforward",
            max_depth=DEPTH,
            population_size=POP,
        )
        algo, problem, state = _setup(config)
        with pytest.raises(ValueError, match="neuromodulation"):
            algo.evaluate_genome_on_problem(state, state.pop_nodes[0], state.pop_conns[0], problem)


class TestNewProblem:
    def test_a_fresh_state_with_the_inherited_genome(self):
        algo, problem, state = _setup()
        state, _ = algo.run_generation(state, problem)
        genome = algo.get_best_genome_with_fitness()[:2]
        fresh = algo.create_state_for_new_problem(ANDProblem(), seed=7, inherit_genome=genome)
        np.testing.assert_array_equal(np.asarray(fresh.pop_nodes[-1]), genome[0])
        np.testing.assert_array_equal(
            np.asarray(algo._cached_targets).ravel(), [0.0, 0.0, 0.0, 1.0]
        )
        algo.run_generation(fresh, ANDProblem())
        assert np.all(np.isfinite(np.asarray(algo._cached_fitnesses)))
