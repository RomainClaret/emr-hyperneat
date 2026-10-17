"""The per-generation metrics report what the run did: species, topology and fitness statistics.

0.1.0 reported ``num_species=0`` on the standard path and one species holding the whole
population on the multi-task and depth-offload paths, although TensorNEAT speciates the CPPN
population every generation; ``hidden_nodes_*`` and ``connections_*`` were never filled; depth
offload averaged a sanitized -inf into the mean fitness. The expected values here come from
TensorNEAT's own species state and the weight matrices, not from the helpers under test.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (
    DEFAULT_SEED,
    EMRHyperNEAT,
    MultiTaskProblem,
    XORProblem,
    create_base_config,
    create_config_with_multi_task,
)
from emr_hyperneat.emrhyperneat import (
    NT_TASK_PRESETS,
    fitness_statistics,
    species_summary,
    topology_summary,
)

NAN = float('nan')
# A low compatibility threshold makes TensorNEAT split the population within a few generations.
SPLITTING_SPECIES = {'compatibility_threshold': 0.5}


def _expected_species(state, fitnesses):
    """Species of the population ``state`` holds, read from its genome-to-species map."""
    members = np.asarray(state.species.idx2species)
    fitness = np.asarray(fitnesses)
    keys = sorted(set(members.tolist()))
    return (len(keys), sorted(int(np.sum(members == k)) for k in keys),
            sorted(float(np.max(fitness[members == k])) for k in keys))


def _assert_species(metrics, state, fitnesses, pop_size):
    count, sizes, best = _expected_species(state, fitnesses)
    assert metrics.num_species == count
    assert sorted(metrics.species_sizes) == sizes
    assert sum(metrics.species_sizes) == pop_size
    assert sorted(metrics.species_fitness) == pytest.approx(best)
    assert max(metrics.species_fitness) == pytest.approx(metrics.best_fitness)


def _splitting_config(**kwargs):
    config = create_base_config(**kwargs)
    config['algorithm_params']['emrhyperneat']['neat_species'] = dict(SPLITTING_SPECIES)
    return config


class TestHelpers:
    """The three helpers on hand-made inputs."""

    def test_species_summary(self):
        species = SimpleNamespace(
            member_count=jnp.array([4.0, NAN, 2.0, NAN]),
            species_keys=jnp.array([0.0, NAN, 5.0, NAN]),
            idx2species=jnp.array([0.0, 0.0, 5.0, 0.0, 5.0, 0.0]),
        )
        fitnesses = jnp.array([0.1, 0.9, 0.3, 0.2, 0.7, 0.4])
        assert species_summary(species, fitnesses) == (
            2, [4, 2], [pytest.approx(0.9), pytest.approx(0.7)])

    def test_species_summary_first_generation(self):
        """TensorNEAT's setup puts every genome in species 0."""
        species = SimpleNamespace(
            member_count=jnp.array([3.0, NAN]), species_keys=jnp.array([0.0, NAN]),
            idx2species=jnp.zeros(3))
        assert species_summary(species, jnp.array([0.2, -jnp.inf, 0.5])) == (
            1, [3], [pytest.approx(0.5)])

    def test_topology_summary(self):
        W1 = jnp.zeros((2, 2, 3)).at[0, 0, 0].set(1.0).at[0, 1, 0].set(-2.0).at[0, 1, 1].set(0.5)
        W1 = W1.at[1, 0, 0].set(1.0)
        W2 = jnp.zeros((2, 3, 1)).at[0, 0, 0].set(1.0).at[0, 1, 0].set(1.0)
        masks = jnp.array([[True, True, False], [True, False, False]])
        assert topology_summary(W1, W2, masks) == {
            'hidden_nodes_mean': 1.5, 'hidden_nodes_std': 0.5,
            'connections_mean': 3.0, 'connections_std': 2.0}
        with_hh = topology_summary(W1, W2, masks, jnp.array([1, 4]))
        assert (with_hh['connections_mean'], with_hh['connections_std']) == (5.5, 0.5)

    def test_fitness_statistics(self):
        best, mean, low, std = fitness_statistics(jnp.array([0.5, -jnp.inf, 0.9]), 1)
        assert (best, mean, low, std) == pytest.approx((0.9, 0.7, 0.5, 0.2))
        assert fitness_statistics(jnp.array([0.5, 0.9]), 0) == pytest.approx((0.9, 0.7, 0.5, 0.2))
        best, *rest = fitness_statistics(jnp.array([-jnp.inf, -jnp.inf]), 2)
        assert best == -np.inf and all(np.isnan(rest))


class TestStandardPath:
    """run_generation on the standard path, feed-forward and recurrent."""

    @pytest.mark.parametrize('preset', ['feedforward', 'hidden_only'])
    def test_species_and_topology(self, preset):
        config = _splitting_config(max_depth=2, population_size=40)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = {
            'preset': preset}
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
        most_species = 0
        for _ in range(5):
            evaluated = state
            state, metrics = algorithm.run_generation(state, problem)
            _assert_species(metrics, evaluated, algorithm._cached_fitnesses, 40)
            most_species = max(most_species, metrics.num_species)

            # The weight matrices are masked to the active positions before any compaction,
            # so their nonzero entries are the evaluated connections.
            W1, W2 = algorithm._cached_W1, algorithm._cached_W2
            connections = np.asarray(jnp.sum(W1 != 0, axis=(1, 2)) + jnp.sum(W2 != 0, axis=(1, 2)))
            if preset == 'hidden_only':
                connections = connections + np.asarray(algorithm._cached_sparse_hh.num_valid)
            assert metrics.connections_mean == pytest.approx(float(np.mean(connections)))
            assert metrics.connections_std == pytest.approx(float(np.std(connections)))
            assert metrics.hidden_nodes_mean == pytest.approx(
                metrics.custom_metrics['avg_hidden_nodes'])
            assert metrics.custom_metrics['num_levels'] == 3
            assert metrics.custom_metrics['variance_threshold'] == pytest.approx(0.03)
        # Otherwise the species checks above only ever saw the single species of generation 0.
        assert most_species > 1


class TestMultiTaskPath:
    def test_species_and_topology(self):
        task_names = ['xor', 'and']
        config = create_config_with_multi_task(
            task_names=task_names, fitness_aggregation='mean', population_size=40)
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['recurrence'] = {'preset': 'feedforward'}
        config['algorithm_params']['emrhyperneat']['neat_species'] = dict(SPLITTING_SPECIES)
        problems = [MultiTaskProblem(name) for name in task_names]
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), problems[0],
                                     seed=DEFAULT_SEED)
        most_species = 0
        for _ in range(5):
            evaluated = state
            state, metrics = algorithm.run_generation_multitask(
                state, problems=problems,
                neurotransmitters=[jnp.array(NT_TASK_PRESETS[name]) for name in task_names],
                aggregation_method='mean')
            # The aggregated fitness the population was told with is not cached; its maximum is.
            count, sizes, _ = _expected_species(evaluated, np.zeros(40))
            assert (metrics.num_species, sorted(metrics.species_sizes)) == (count, sizes)
            assert max(metrics.species_fitness) == pytest.approx(metrics.best_fitness)
            assert metrics.hidden_nodes_mean is not None and metrics.connections_mean is not None
            most_species = max(most_species, metrics.num_species)
        assert most_species > 1


class TestDepthOffloadPath:
    def test_species_and_statistics(self):
        config = _splitting_config(max_depth=2, population_size=30)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
            'mode': 'depth_offload'}
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
        most_species = 0
        for _ in range(5):
            evaluated = state
            state, metrics = algorithm.run_generation(state, problem)
            assert metrics.custom_metrics['mode'] == 'feedforward_depth_offload'
            count, sizes, _ = _expected_species(evaluated, np.zeros(30))
            assert (metrics.num_species, sorted(metrics.species_sizes)) == (count, sizes)
            assert max(metrics.species_fitness) == pytest.approx(metrics.best_fitness)
            assert metrics.custom_metrics['nan_fitness_count'] == 0
            assert np.isfinite(metrics.mean_fitness)
            most_species = max(most_species, metrics.num_species)
        assert most_species > 1
