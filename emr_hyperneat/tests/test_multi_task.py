"""Tests for multi-task evolution in EMR-HyperNEAT.

Tests cover:
- Multi-task configuration
- 6 fitness aggregation methods (mean, min, weighted, product, softmin, harmonic)
- Task-specific neurotransmitter vectors
- Multi-task metrics tracking
- Per-task fitness evaluation
"""

import pytest
import jax.numpy as jnp
from conftest import (
    EMRHyperNEAT,
    create_config_with_multi_task,
    run_multitask_evolution,
    assert_no_errors,
    assert_positive_fitness,
    MultiTaskProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
)


# Fitness aggregation methods to test
AGGREGATION_METHODS = ['mean', 'min', 'weighted', 'product', 'softmin', 'harmonic']

# Task configurations
TASK_NAMES = ['xor', 'and', 'or', 'nand', 'nor']


class TestMultiTaskConfiguration:
    """Tests for multi-task configuration."""

    def test_multi_task_config_creation(self):
        """Verify multi-task configuration is created correctly."""
        config = create_config_with_multi_task(
            task_names=['xor', 'and', 'or'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='min'
        )

        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr['multitask']['enabled'] == True
        assert hmr['multitask']['num_tasks'] == 3
        assert hmr['multitask']['task_names'] == ['xor', 'and', 'or']
        assert hmr['multitask']['fitness_aggregation'] == 'min'

    def test_multi_task_initialization(self, algorithm, xor_problem):
        """Verify multi-task algorithm initializes."""
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            neuromod_mode='true_neuromodulation_4nt'
        )
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("num_tasks", [2, 3, 4, 5])
    def test_different_task_counts(self, algorithm, xor_problem, num_tasks):
        """Verify different numbers of tasks work."""
        task_names = TASK_NAMES[:num_tasks]
        config = create_config_with_multi_task(
            task_names=task_names,
            neuromod_mode='true_neuromodulation_4nt'
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)


class TestFitnessAggregationMethods:
    """Tests for different fitness aggregation methods.

    Every method configures and initializes, and each one evolves for a few generations on
    the two-task fixture, one problem per task.
    """

    @pytest.mark.parametrize("method", AGGREGATION_METHODS)
    def test_aggregation_method_configures(self, algorithm, xor_problem, method):
        """Verify each aggregation method configures correctly."""
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation=method
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_mean_aggregation(self, algorithm, two_task_problems):
        """Verify mean aggregation works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_min_aggregation(self, algorithm, two_task_problems):
        """Verify min aggregation works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='min'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_weighted_aggregation(self, algorithm, two_task_problems):
        """Verify weighted aggregation works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_product_aggregation(self, algorithm, two_task_problems):
        """Verify product aggregation works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='product'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='product',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_softmin_aggregation(self, algorithm, two_task_problems):
        """Verify softmin aggregation works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_harmonic_aggregation(self, algorithm, two_task_problems):
        """Verify harmonic aggregation works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='harmonic'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='harmonic',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestNTVectorAssignment:
    """Tests for neurotransmitter vector assignment."""

    def test_nt_vectors_per_task(self, algorithm, xor_problem):
        """Verify NT vectors are assigned per task."""
        config = create_config_with_multi_task(
            task_names=['xor', 'and', 'or'],
            neuromod_mode='true_neuromodulation_4nt'
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Multi-task state should be initialized
        assert algorithm.multitask_config is not None
        assert algorithm.multitask_config.enabled == True

    def test_set_task_neurotransmitter(self, algorithm, xor_problem):
        """Verify task NT vectors can be set."""
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            neuromod_mode='true_neuromodulation_4nt'
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Should be able to set NT vector
        nt_vector = jnp.array([0.9, 0.1, 0.8, 1.0])
        algorithm.set_task_neurotransmitter(0, nt_vector)


class TestMultiTaskEvolution:
    """Tests for multi-task evolution execution."""

    def test_multi_task_runs_generation(self, algorithm, two_task_problems):
        """Verify multi-task evolution can run generations."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_5_task_evolution(self, algorithm, five_task_problems):
        """Verify 5-task evolution works."""
        config = create_config_with_multi_task(
            task_names=five_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='min'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=five_task_problems['problems'],
            nt_vectors=five_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestMultiTaskWithFeatures:
    """Tests for multi-task with other features enabled."""

    def test_multi_task_refuses_dynamic_functions(self, algorithm, two_task_problems):
        """Multi-task evaluation applies one activation per task, so per-node ones are refused.

        Accepting the combination would ignore the per-node activations, and a test that only
        checks for no error and a positive fitness would pass either way.
        """
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}

        with pytest.raises(ValueError, match="run_generation_multitask evaluates every task"):
            algorithm.create_config(config)

    def test_multi_task_with_full_recurrence(self, algorithm, two_task_problems):
        """Verify multi-task works with full recurrence."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'full_recurrent',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'allow_backward': True,
            'allow_lateral': True,
            'allow_self_loops': True,
            'iteration_level': 2,
        }

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestMultiTaskNTTypes:
    """Tests for different NT type counts in multi-task."""

    @pytest.mark.parametrize("num_nt", [2, 4, 5, 6])
    def test_nt_type_counts(self, algorithm, two_task_problems, num_nt):
        """Verify different NT type counts work with multi-task."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['neuromodulation']['num_nt_types'] = num_nt

        # Generate NT vectors matching num_nt (base vectors truncated/extended)
        # Task 1 (xor): high DA, low 5HT pattern
        # Task 2 (and): low DA, high 5HT pattern
        base_nt1 = [0.95, 0.05, 0.95, 1.0, 0.5, 0.5][:num_nt]
        base_nt2 = [0.10, 0.90, 0.10, 1.0, 0.5, 0.5][:num_nt]
        nt_vectors = [jnp.array(base_nt1), jnp.array(base_nt2)]

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=nt_vectors,
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"NT type count {num_nt} failed")


class TestMultiTaskAggregationBehavior:
    """Tests verifying aggregation methods produce distinct behavior."""

    def test_different_aggregations_produce_results(self, algorithm, two_task_problems):
        """Verify different aggregations all produce valid results."""
        results = {}
        for method in ['mean', 'min', 'product']:
            config = create_config_with_multi_task(
                task_names=two_task_problems['task_names'],
                fitness_aggregation=method
            )
            result = run_multitask_evolution(
                algorithm, config,
                problems=two_task_problems['problems'],
                nt_vectors=two_task_problems['nt_vectors'],
                aggregation_method=method,
                generations=QUICK_GENERATIONS
            )
            results[method] = result.best_fitness

        # All should produce valid (non-negative) fitness
        for method, fitness in results.items():
            assert fitness >= 0, f"Aggregation {method} has invalid fitness"


class TestComprehensiveMultiTask:
    """Slow runs: every aggregation method, all features on five tasks, and several task counts."""

    @pytest.mark.slow
    @pytest.mark.parametrize("method", AGGREGATION_METHODS)
    def test_aggregation_methods_evolve(self, algorithm, two_task_problems, method):
        """Verify all aggregation methods support evolution."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation=method
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method=method,
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        # Should achieve some fitness
        assert result.best_fitness >= 0

    @pytest.mark.slow
    def test_5_task_with_all_features(self, algorithm, five_task_problems):
        """5-task evolution with every feature multi-task evaluation applies.

        True neuromodulation and full recurrence. Per-node activation and aggregation are
        refused under multi-task evaluation (see test_multi_task_refuses_dynamic_functions).
        """
        config = create_config_with_multi_task(
            task_names=five_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {'preset': 'full_recurrent'}

        result = run_multitask_evolution(
            algorithm, config,
            problems=five_task_problems['problems'],
            nt_vectors=five_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.slow
    @pytest.mark.parametrize("num_tasks", [2, 3, 5])
    def test_different_task_counts_evolve(self, algorithm, two_task_problems, three_task_problems, five_task_problems, num_tasks):
        """Verify different task counts support evolution."""
        # Select the appropriate fixture based on num_tasks
        if num_tasks == 2:
            task_bundle = two_task_problems
        elif num_tasks == 3:
            task_bundle = three_task_problems
        else:  # num_tasks == 5
            task_bundle = five_task_problems

        config = create_config_with_multi_task(
            task_names=task_bundle['task_names'],
            fitness_aggregation='min'
        )
        result = run_multitask_evolution(
            algorithm, config,
            problems=task_bundle['problems'],
            nt_vectors=task_bundle['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"{num_tasks} tasks failed")


# Multi-Task Gap Tests

class TestTaskWeights:
    """Tests for task_weights parameter in weighted aggregation."""

    def test_weighted_aggregation_with_uniform_weights(self, algorithm, two_task_problems):
        """Verify weighted aggregation works with uniform weights."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.5, 0.5]  # Uniform

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_weighted_aggregation_with_skewed_weights(self, algorithm, two_task_problems):
        """Verify weighted aggregation with skewed weights (prioritize first task)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.8, 0.2]  # Skewed

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_weighted_aggregation_three_tasks(self, algorithm, three_task_problems):
        """Verify weighted aggregation works with 3 tasks."""
        config = create_config_with_multi_task(
            task_names=three_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.5, 0.3, 0.2]

        result = run_multitask_evolution(
            algorithm, config,
            problems=three_task_problems['problems'],
            nt_vectors=three_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("weights", [
        [1.0, 0.0],   # Full weight on first task
        [0.0, 1.0],   # Full weight on second task
        [0.9, 0.1],   # Nearly all first
        [0.1, 0.9],   # Nearly all second
    ])
    def test_extreme_weight_distributions(self, algorithm, two_task_problems, weights):
        """Verify extreme weight distributions are handled correctly."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = weights

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestJointEvolution:
    """Tests for joint_evolution toggle."""

    def test_joint_evolution_enabled(self, algorithm, two_task_problems):
        """Verify joint_evolution=True works (default behavior)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['joint_evolution'] = True

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_joint_evolution_disabled(self, algorithm, two_task_problems):
        """Verify joint_evolution=False works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['joint_evolution'] = False

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_joint_evolution_with_min_aggregation(self, algorithm, two_task_problems):
        """Verify joint_evolution=False with min aggregation."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['joint_evolution'] = False

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestConstantBonusesRaise:
    """The orthogonality and specialization bonuses add one value to every genome of a generation (computed from the
    tasks' NT vectors and from the population's mean fitness variance), which never changes a rank within it: a
    non-zero value raises, as a configuration key, as an argument of run_generation_multitask, and as
    orthogonality_bonus_weight on a multi-task configuration built by hand. They replace TestOrthogonalityBonus,
    TestSpecializationBonus and TestCombinedBonuses, which asserted only that nothing raised."""

    @pytest.mark.parametrize('key, value', [('orthogonality_bonus', 0.3), ('specialization_bonus', 0.2)])
    def test_the_configuration_key_raises(self, two_task_problems, key, value):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'][key] = value
        with pytest.raises(ValueError, match=f'multitask.{key}=.* never changes a rank'):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('argument, value', [('orthogonality_bonus', 0.3), ('specialization_bonus', 0.2)])
    def test_the_argument_raises(self, algorithm, two_task_problems, argument, value):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        state = algorithm.initialize(algorithm.create_config(config), two_task_problems['problems'][0],
                                     seed=DEFAULT_SEED)
        with pytest.raises(ValueError, match=f'{argument}=.* one value to every genome'):
            algorithm.run_generation_multitask(state, problems=two_task_problems['problems'],
                                               neurotransmitters=two_task_problems['nt_vectors'],
                                               **{argument: value})

    @pytest.mark.parametrize('key', ['orthogonality_bonus', 'specialization_bonus'])
    def test_zero_is_accepted(self, algorithm, two_task_problems, key):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'][key] = 0.0
        result = run_multitask_evolution(algorithm, config, problems=two_task_problems['problems'],
                                         nt_vectors=two_task_problems['nt_vectors'], aggregation_method='mean',
                                         generations=1)
        assert_no_errors(result)


class TestSoftminTemperature:
    """Tests for softmin_temperature parameter in multi-task fitness aggregation.

    The softmin_temperature controls the "sharpness" of the softmin aggregation:
    - Low temperature (0.1): Behaves like hard min (focuses on worst task)
    - High temperature (2.0): Smoother averaging across tasks
    """

    def test_softmin_temperature_default(self, algorithm, two_task_problems):
        """Verify softmin uses default temperature when not specified."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        # Default softmin_temperature should be 0.1
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr['multitask'].get('softmin_temperature', 0.1) == 0.1

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_softmin_temperature_values(self, algorithm, two_task_problems, temperature):
        """Verify different softmin_temperature values work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = temperature

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"softmin_temperature={temperature} failed")

    def test_softmin_low_temperature_like_min(self, algorithm, two_task_problems):
        """Verify low temperature (0.1) behaves like hard min."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = 0.1  # Sharp, like hard min

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_softmin_high_temperature_smooth(self, algorithm, two_task_problems):
        """Verify high temperature (2.0) provides smoother aggregation."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = 2.0  # Smooth, more like mean

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_softmin_with_three_tasks(self, algorithm, three_task_problems):
        """Verify softmin_temperature works with 3 tasks."""
        config = create_config_with_multi_task(
            task_names=three_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = 0.5

        result = run_multitask_evolution(
            algorithm, config,
            problems=three_task_problems['problems'],
            nt_vectors=three_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


# Multi-Task Edge Case Tests

class TestSoftminExtremes:
    """Tests for extreme softmin_temperature values."""

    def test_softmin_very_low_temperature(self, algorithm, two_task_problems):
        """Verify very low softmin_temperature (<0.1) works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = 0.01  # Very low

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_softmin_very_high_temperature(self, algorithm, two_task_problems):
        """Verify very high softmin_temperature (>5.0) works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = 10.0  # Very high

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("temp", [0.01, 0.05, 5.0, 10.0, 50.0])
    def test_extreme_temperature_values(self, algorithm, two_task_problems, temp):
        """Verify a range of extreme temperature values work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = temp

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"softmin_temperature={temp} failed")

    def test_softmin_extremes_with_bonuses(self, algorithm, two_task_problems):
        """Verify an extreme temperature works with a per-genome bonus.

        The constant orthogonality and specialization bonuses raise (they add one constant to
        every genome); the generalist bonus is applied per genome.
        """
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = 0.01
        hmr['multitask']['generalist_bonus_type'] = 'min_bonus'
        hmr['multitask']['generalist_bonus_weight'] = 0.1

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestManyTasks:
    """Tests for scaling to many tasks (6-10).

    Note: Uses MultiTaskProblem('xor') instead of XORProblem because
    create_config_with_multi_task uses 2-input substrate (no bias) to match
    MultiTaskProblem's input_shape=(2,). The variation comes from different
    NT vectors, not different problem targets.
    """

    def test_six_tasks(self, algorithm):
        """Verify 6 tasks work (XOR variants with different NT vectors)."""
        # Create 6 task variations - all XOR but with different NT vectors
        task_names = ['XOR_v1', 'XOR_v2', 'XOR_v3', 'XOR_v4', 'XOR_v5', 'XOR_v6']
        problems = [MultiTaskProblem('xor') for _ in range(6)]  # 2-input problems
        nt_vectors = [
            jnp.array([0.95, 0.05, 0.95, 1.0]),  # XOR v1
            jnp.array([0.10, 0.90, 0.10, 1.0]),  # XOR v2
            jnp.array([0.50, 0.50, 0.50, 0.5]),  # XOR v3
            jnp.array([0.80, 0.20, 0.60, 0.8]),  # XOR v4
            jnp.array([0.30, 0.70, 0.40, 0.6]),  # XOR v5
            jnp.array([0.60, 0.40, 0.80, 0.9]),  # XOR v6
        ]

        config = create_config_with_multi_task(
            task_names=task_names,
            fitness_aggregation='mean'
        )

        result = run_multitask_evolution(
            algorithm, config,
            problems=problems,
            nt_vectors=nt_vectors,
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_many_tasks_with_softmin(self, algorithm):
        """Verify many tasks (6) work with softmin aggregation."""
        task_names = [f'XOR_v{i}' for i in range(6)]
        problems = [MultiTaskProblem('xor') for _ in range(6)]  # 2-input problems
        nt_vectors = [
            jnp.array([0.95, 0.05, 0.95, 1.0]),
            jnp.array([0.10, 0.90, 0.10, 1.0]),
            jnp.array([0.50, 0.50, 0.50, 0.5]),
            jnp.array([0.80, 0.20, 0.60, 0.8]),
            jnp.array([0.30, 0.70, 0.40, 0.6]),
            jnp.array([0.60, 0.40, 0.80, 0.9]),
        ]

        config = create_config_with_multi_task(
            task_names=task_names,
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = 0.5

        result = run_multitask_evolution(
            algorithm, config,
            problems=problems,
            nt_vectors=nt_vectors,
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_many_tasks_with_weighted(self, algorithm):
        """Verify many tasks (6) work with weighted aggregation."""
        task_names = [f'XOR_v{i}' for i in range(6)]
        problems = [MultiTaskProblem('xor') for _ in range(6)]  # 2-input problems
        nt_vectors = [
            jnp.array([0.95, 0.05, 0.95, 1.0]),
            jnp.array([0.10, 0.90, 0.10, 1.0]),
            jnp.array([0.50, 0.50, 0.50, 0.5]),
            jnp.array([0.80, 0.20, 0.60, 0.8]),
            jnp.array([0.30, 0.70, 0.40, 0.6]),
            jnp.array([0.60, 0.40, 0.80, 0.9]),
        ]

        config = create_config_with_multi_task(
            task_names=task_names,
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]

        result = run_multitask_evolution(
            algorithm, config,
            problems=problems,
            nt_vectors=nt_vectors,
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestWeightNormalization:
    """Tests for task weight normalization behavior."""

    def test_unnormalized_weights_summing_to_one(self, algorithm, two_task_problems):
        """Verify weights that naturally sum to 1.0 work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.5, 0.5]  # Sum = 1.0

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_unnormalized_weights_summing_less_than_one(self, algorithm, two_task_problems):
        """Verify weights summing to less than 1.0 work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.3, 0.3]  # Sum = 0.6

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_unnormalized_weights_summing_greater_than_one(self, algorithm, two_task_problems):
        """Verify weights summing to greater than 1.0 work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.8, 0.8]  # Sum = 1.6

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_large_weight_values(self, algorithm, two_task_problems):
        """Verify large weight values work (should be normalized internally)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [10.0, 5.0]  # Large values

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("weights", [
        [0.1, 0.1],   # Sum = 0.2 (low)
        [1.0, 1.0],   # Sum = 2.0 (high)
        [5.0, 5.0],   # Sum = 10.0 (very high)
        [0.001, 0.999],  # Extreme ratio
    ])
    def test_various_weight_scales(self, algorithm, two_task_problems, weights):
        """Verify various weight scale combinations work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = weights

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"Weights {weights} failed")


# Extended MultiTaskConfig Feature Tests

class TestFitnessMode:
    """Tests for fitness_mode parameter in multi-task evaluation.

    The fitness_mode controls how fitness is calculated:
    - 'mse': Mean Squared Error (default)
    - 'accuracy': Classification accuracy (0 or 1 per sample)
    - 'acc_mse': Combined accuracy + MSE
    - 'hybrid': Hybrid fitness metric
    - 'bce': Binary Cross-Entropy loss
    - 'soft_accuracy': Soft accuracy with distance consideration
    """

    @pytest.mark.parametrize("mode", ['mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy'])
    def test_fitness_mode_options(self, algorithm, two_task_problems, mode):
        """Verify all fitness_mode options work with multi-task evolution."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['fitness_mode'] = mode

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"fitness_mode='{mode}' failed")

    def test_fitness_mode_mse_default(self, algorithm, two_task_problems):
        """Verify MSE is the default fitness mode."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        # Check default
        assert hmr['multitask'].get('fitness_mode', 'mse') == 'mse'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_fitness_mode_with_min_aggregation(self, algorithm, two_task_problems):
        """Verify fitness_mode works with min aggregation."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['fitness_mode'] = 'accuracy'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestNoEffectMultiTaskSettings:
    """Multi-task settings nothing in EMR applies raise unless left at their default: the orthogonality
    and specialization bonuses add one constant to every genome, so selection never sees them;
    specialization_bonus_weight, modulation_penalty, confusion_eval_frequency and branch_gating_mode
    are never read; orthogonality_metric applies only with an orthogonality bonus."""

    @pytest.mark.parametrize(
        'key, value',
        [
            ('orthogonality_bonus', 0.3),
            ('specialization_bonus', 0.2),
            ('confusion_eval_frequency', 5),
        ],
    )
    def test_a_setting_without_effect_raises(self, two_task_problems, key, value):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'][key] = value
        with pytest.raises(
            ValueError, match=f'multitask.{key}=.* (never changes a rank|is never read)'
        ):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize(
        'argument, value', [('orthogonality_bonus', 0.3), ('specialization_bonus', 0.2)]
    )
    def test_a_bonus_argument_raises(self, algorithm, two_task_problems, argument, value):
        """run_generation_multitask's arguments of the same names add the same constant, so they
        raise."""
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        state = algorithm.initialize(
            algorithm.create_config(config), two_task_problems['problems'][0], seed=DEFAULT_SEED
        )
        with pytest.raises(ValueError, match=f'{argument}=.* one value to every genome'):
            algorithm.run_generation_multitask(state, problems=two_task_problems['problems'],
                                               neurotransmitters=two_task_problems['nt_vectors'],
                                               **{argument: value})

    @pytest.mark.parametrize('key, value', [
        ('orthogonality_bonus_weight', 0.0), ('specialization_bonus_weight', 0.0),
        ('modulation_penalty', 0.0), ('confusion_eval_frequency', 0), ('branch_gating_mode', 'none'),
        ('orthogonality_metric', 'cosine_mean'),
    ])
    def test_the_default_is_accepted(self, algorithm, two_task_problems, key, value):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'][key] = value
        result = run_multitask_evolution(algorithm, config, problems=two_task_problems['problems'],
                                         nt_vectors=two_task_problems['nt_vectors'],
                                         aggregation_method='mean', generations=1)
        assert_no_errors(result)


class TestGeneralistBonus:
    """Tests for generalist_bonus_type, generalist_bonus_weight, and generalist_threshold.

    These parameters control bonuses for generalist networks:
    - 'none': No bonus (default)
    - 'min_bonus': Bonus based on minimum task performance
    - 'variance_penalty': Penalty for high variance across tasks
    - 'threshold_bonus': Bonus for exceeding threshold on all tasks
    """

    @pytest.mark.parametrize("bonus_type", ['none', 'min_bonus', 'variance_penalty', 'threshold_bonus'])
    def test_generalist_bonus_types(self, algorithm, two_task_problems, bonus_type):
        """Verify all generalist_bonus_type options work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = bonus_type
        hmr['multitask']['generalist_bonus_weight'] = 0.1 if bonus_type != 'none' else 0.0

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"generalist_bonus_type='{bonus_type}' failed")

    @pytest.mark.parametrize("weight", [0.0, 0.1, 0.3, 0.5])
    def test_generalist_bonus_weights(self, algorithm, two_task_problems, weight):
        """Verify different generalist_bonus_weight values work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = 'min_bonus'
        hmr['multitask']['generalist_bonus_weight'] = weight

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"generalist_bonus_weight={weight} failed")

    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9, 0.95])
    def test_generalist_threshold_values(self, algorithm, two_task_problems, threshold):
        """Verify different generalist_threshold values with threshold_bonus."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = 'threshold_bonus'
        hmr['multitask']['generalist_bonus_weight'] = 0.2
        hmr['multitask']['generalist_threshold'] = threshold

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"generalist_threshold={threshold} failed")

    def test_variance_penalty_with_three_tasks(self, algorithm, three_task_problems):
        """Verify variance_penalty works with 3 tasks."""
        config = create_config_with_multi_task(
            task_names=three_task_problems['task_names'],
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = 'variance_penalty'
        hmr['multitask']['generalist_bonus_weight'] = 0.1

        result = run_multitask_evolution(
            algorithm, config,
            problems=three_task_problems['problems'],
            nt_vectors=three_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestMultiTaskModulationMode:
    """Tests for modulation_mode at MultiTaskConfig level.

    Controls how neuromodulation affects activations:
    - 'full': Full modulation (gating + gain/bias)
    - 'gating_only': Only apply gating
    - 'gain_bias_only': Only apply gain/bias modulation
    """

    @pytest.mark.parametrize("mode", ['full', 'gating_only', 'gain_bias_only'])
    def test_multitask_modulation_modes(self, algorithm, two_task_problems, mode):
        """Verify all modulation_mode options work at MultiTaskConfig level."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['modulation_mode'] = mode

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"modulation_mode='{mode}' failed")

    def test_modulation_mode_with_softmin(self, algorithm, two_task_problems):
        """Verify modulation_mode works with softmin aggregation."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['modulation_mode'] = 'gating_only'
        hmr['multitask']['softmin_temperature'] = 0.3

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestModulationStrengthOverride:
    """Tests for modulation_strength_override parameter.

    Allows overriding the default modulation strength from NeuromodulationConfig.
    None = use default; float value = override.
    """

    def test_modulation_strength_override_none(self, algorithm, two_task_problems):
        """Verify modulation_strength_override=None uses default."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['modulation_strength_override'] = None  # Use default

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("strength", [0.5, 1.0, 2.0, 5.0])
    def test_modulation_strength_override_values(self, algorithm, two_task_problems, strength):
        """Verify different modulation_strength_override values work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['modulation_strength_override'] = strength

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"modulation_strength_override={strength} failed")


class TestTwoModuleMode:
    """two_module_mode (Liu & Wang 2024 PFC/Sensorimotor separation), feed-forward only.

    The configuration key reaches the evaluation and changes the fitness of the same population;
    these tests run feed-forward, since two-module mode refuses recurrence.
    """

    @pytest.mark.parametrize("mode", ['none', 'parallel', 'sequential'])
    def test_two_module_mode_options(self, algorithm, two_task_problems, mode):
        """Verify all two_module_mode options work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['two_module_mode'] = mode

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"two_module_mode='{mode}' failed")

    @staticmethod
    def _generation(two_task_problems, mode, **multitask):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean',
                                               recurrence_preset='feedforward', population_size=20)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'].update(
            two_module_mode=mode, **multitask)
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), two_task_problems['problems'][0],
                                     seed=DEFAULT_SEED)
        _, metrics = algorithm.run_generation_multitask(
            state, problems=two_task_problems['problems'],
            neurotransmitters=two_task_problems['nt_vectors'], aggregation_method='mean')
        return metrics

    @pytest.mark.parametrize("mode", ['parallel', 'sequential'])
    def test_two_modules_run(self, two_task_problems, mode):
        """The mode reaches the evaluation and changes the fitness of the same population."""
        one = self._generation(two_task_problems, 'none')
        two = self._generation(two_task_problems, mode)
        assert one.custom_metrics['two_module_mode'] is None
        assert two.custom_metrics['two_module_mode'] == mode
        assert two.custom_metrics['per_task_mean'] != one.custom_metrics['per_task_mean']

    def test_two_module_with_generalist_bonus(self, two_task_problems):
        """two_module_mode with the generalist bonus, both from the configuration."""
        metrics = self._generation(two_task_problems, 'parallel', generalist_bonus_type='min_bonus',
                                   generalist_bonus_weight=0.1)
        assert metrics.custom_metrics['two_module_mode'] == 'parallel'
        assert metrics.custom_metrics['generalist_bonus_type'] == 'min_bonus'

    def test_recurrence_is_refused(self, two_task_problems):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask']['two_module_mode'] = (
            'parallel')
        with pytest.raises(ValueError, match='two_module_mode evaluates feed-forward only'):
            EMRHyperNEAT().create_config(config)


class TestPerTaskActivation:
    """Tests for per-task activation function override.

    Tests hidden_activation (global) and per_task_activation (per-task override).
    """

    def test_hidden_activation_global(self, algorithm, two_task_problems):
        """Verify global hidden_activation works for all tasks."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['hidden_activation'] = 'sin'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("activation", ['tanh', 'sigmoid', 'relu', 'sin', 'identity'])
    def test_hidden_activation_options(self, algorithm, two_task_problems, activation):
        """Verify different hidden_activation options work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['hidden_activation'] = activation

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"hidden_activation='{activation}' failed")

    def test_per_task_activation_override(self, algorithm, two_task_problems):
        """Verify per_task_activation can override per-task activations."""
        task_names = two_task_problems['task_names']
        config = create_config_with_multi_task(
            task_names=task_names,
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['hidden_activation'] = 'tanh'  # Global default
        hmr['multitask']['per_task_activation'] = {
            task_names[0]: 'sin',   # Override first task
            task_names[1]: 'tanh',  # Keep second task same
        }

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_per_task_activation_with_three_tasks(self, algorithm, three_task_problems):
        """Verify per_task_activation works with 3 tasks."""
        task_names = three_task_problems['task_names']
        config = create_config_with_multi_task(
            task_names=task_names,
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['per_task_activation'] = {
            task_names[0]: 'sin',
            task_names[1]: 'tanh',
            task_names[2]: 'relu',
        }

        result = run_multitask_evolution(
            algorithm, config,
            problems=three_task_problems['problems'],
            nt_vectors=three_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestMultiTaskFitnessModes:
    """compute_fitness, the multi-task evaluators' fitness, computes 'accuracy_confident' rather
    than MSE, and refuses 'ce': categorical cross-entropy on these evaluators' sigmoid outputs ranks
    a constant output above a solver."""

    def test_accuracy_confident(self):
        import numpy as np
        from emr_hyperneat.emrhyperneat import compute_fitness
        outputs = jnp.array([[0.9], [0.2], [0.6], [0.4]])
        targets = jnp.array([[1.0], [0.0], [0.0], [1.0]])
        expected = 0.5 + 0.1 * float(np.mean(np.abs(np.array([0.9, 0.2, 0.6, 0.4]) - 0.5) * 2.0))
        confident = float(compute_fitness(outputs, targets, 'accuracy_confident'))
        assert confident == pytest.approx(expected, abs=1e-6)
        assert confident != pytest.approx(float(compute_fitness(outputs, targets, 'mse')), abs=1e-3)

    def test_ce_is_refused(self, algorithm, two_task_problems):
        from emr_hyperneat.emrhyperneat import compute_fitness
        with pytest.raises(ValueError, match="'ce' is categorical cross-entropy"):
            compute_fitness(jnp.full((4, 1), 0.9999), jnp.array([[0.0], [1.0], [1.0], [0.0]]), 'ce')
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        state = algorithm.initialize(algorithm.create_config(config), two_task_problems['problems'][0],
                                     seed=DEFAULT_SEED)
        algorithm.multitask_config.fitness_mode = 'ce'
        generation = algorithm._current_generation
        with pytest.raises(ValueError, match="fitness_mode must be one of"):
            algorithm.run_generation_multitask(state, problems=two_task_problems['problems'],
                                               neurotransmitters=two_task_problems['nt_vectors'])
        # Refused before the generation is built
        assert algorithm._current_generation == generation

    @pytest.mark.parametrize('mode', ['mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy',
                                      'accuracy_confident'])
    def test_the_configuration_key_sets_the_mode(self, algorithm, two_task_problems, mode):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask']['fitness_mode'] = mode
        algorithm.create_config(config)
        assert algorithm.multitask_config.fitness_mode == mode

    @pytest.mark.parametrize('mode', ['ce', 'no_such_mode'])
    def test_the_configuration_key_refuses_other_modes(self, algorithm, two_task_problems, mode):
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'],
                                               fitness_aggregation='mean')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask']['fitness_mode'] = mode
        with pytest.raises(ValueError, match="multitask.fitness_mode must be one of"):
            algorithm.create_config(config)

    @pytest.mark.parametrize('targets_1d', [False, True])
    def test_a_constant_output_never_outscores_a_solver(self, targets_1d):
        from emr_hyperneat.emrhyperneat import MULTITASK_FITNESS_MODES, compute_fitness
        targets = jnp.array([[0.0], [1.0], [1.0], [0.0]])
        if targets_1d:
            targets = targets[:, 0]
        solver = jnp.array([[0.01], [0.99], [0.99], [0.01]])
        constant = jnp.full((4, 1), 0.9999)
        for mode in MULTITASK_FITNESS_MODES:
            assert float(compute_fitness(solver, targets, mode)) > float(compute_fitness(constant, targets, mode)), mode

    def test_an_unknown_mode_raises(self):
        from emr_hyperneat.emrhyperneat import compute_fitness
        with pytest.raises(ValueError, match='unknown fitness mode'):
            compute_fitness(jnp.zeros((2, 1)), jnp.zeros((2, 1)), 'f1')


class TestHmrCallForm:
    """HMR's call form: run_generation_multitask(state, MultiTaskConfig(tasks={name: (inputs,
    targets)}, aggregation=..., ...)) returns MultiTaskMetrics keyed by task name. 47 constructions
    in 39 HMR runner files use it; 0.1.0's MultiTaskConfig had no aggregation, weights or nt_vectors
    field, so each raised TypeError.
    """

    @staticmethod
    def _setup(seed=DEFAULT_SEED):
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
            population_size=20,
        )
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(config), MultiTaskProblem('xor'), seed=seed
        )
        return algorithm, state

    @staticmethod
    def _tasks():
        return {
            name: (MultiTaskProblem(name).inputs, MultiTaskProblem(name).targets)
            for name in ('xor', 'and')
        }

    def test_metrics_by_task_name(self):
        from emr_hyperneat.emrhyperneat import (
            MultiTaskConfig, MultiTaskMetrics,
        )
        algorithm, state = self._setup()
        new_state, metrics = algorithm.run_generation_multitask(
            state, MultiTaskConfig(tasks=self._tasks(), aggregation='mean'))
        assert isinstance(metrics, MultiTaskMetrics)
        assert set(metrics.per_task_best) == {'xor', 'and'} == set(metrics.best_generalist_per_task)
        generalist = metrics.best_generalist_per_task
        assert metrics.aggregated_fitness == pytest.approx(
            (generalist['xor'] + generalist['and']) / 2, abs=1e-6
        )
        assert new_state is not state

    def test_the_same_generation_as_the_list_form(self):
        """The call form is the list form with the tasks' preset NT vectors: same fitnesses, same
        next population."""
        import numpy as np
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        from emr_hyperneat.emrhyperneat import (
            get_nt_for_task,
        )
        first, state = self._setup()
        new_dict, metrics = first.run_generation_multitask(
            state, MultiTaskConfig(tasks=self._tasks(), aggregation='mean')
        )
        second, state = self._setup()
        new_list, list_metrics = second.run_generation_multitask(
            state,
            problems=[MultiTaskProblem('xor'), MultiTaskProblem('and')],
            neurotransmitters=[get_nt_for_task(name, 4) for name in ('xor', 'and')],
            aggregation_method='mean',
        )
        assert [
            metrics.per_task_best['xor'],
            metrics.per_task_best['and'],
        ] == list_metrics.custom_metrics['per_task_best']
        np.testing.assert_array_equal(
            np.asarray(new_dict.pop_nodes), np.asarray(new_list.pop_nodes)
        )

    def test_hmr_product_is_the_plain_product(self):
        import numpy as np
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        algorithm, state = self._setup()
        algorithm.run_generation_multitask(
            state, MultiTaskConfig(tasks=self._tasks(), aggregation='product')
        )
        record = algorithm._last_multitask
        np.testing.assert_array_equal(np.asarray(record['aggregated']),
                                      np.asarray(record['per_task'][0] * record['per_task'][1]))

    @pytest.mark.parametrize(
        'settings, reason',
        [
            (dict(branch_gating_mode='diagonal'), 'branch_gating_mode must be one of'),
            (
                dict(branch_gating_mode='spatial', two_module_mode='parallel'),
                'both set the modulation by region',
            ),
            (dict(modulation_penalty=-0.05), 'modulation_penalty must be a number at least 0'),
            (dict(orthogonality_metric='cosine_max'), 'orthogonality_metric applies only'),
            (dict(confusion_eval_frequency=5), 'confusion_eval_frequency'),
            (dict(aggregation='weighted'), "'weighted' needs weights"),
            (
                dict(aggregation='mean', weights={'xor': 2.0}),
                "weights is read by aggregation 'weighted' only",
            ),
            (dict(nt_vectors={'xor': [1.0, 0.0]}), 'NT vector of task'),
        ],
    )
    def test_refusals(self, settings, reason):
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        algorithm, state = self._setup()
        with pytest.raises(ValueError, match=reason):
            algorithm.run_generation_multitask(
                state, MultiTaskConfig(tasks=self._tasks(), **settings)
            )

    def test_an_unknown_task_activation_raises(self):
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        algorithm, state = self._setup()
        with pytest.raises(ValueError, match="unknown multi-task activation 'sine'"):
            algorithm.run_generation_multitask(
                state, MultiTaskConfig(tasks=self._tasks(), per_task_activation={'xor': 'sine'})
            )

    def test_one_setting_one_name(self):
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        with pytest.raises(ValueError, match='name one setting'):
            MultiTaskConfig(aggregation='min', fitness_aggregation='product')
        assert MultiTaskConfig(aggregation='min').fitness_aggregation == 'min'

    def test_hmr_evaluators(self):
        import numpy as np
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        from emr_hyperneat.emrhyperneat import (
            get_nt_for_task,
        )
        algorithm, state = self._setup()
        algorithm.run_generation_multitask(
            state, MultiTaskConfig(tasks=self._tasks(), aggregation='mean')
        )
        X, Y = self._tasks()['and']
        nt = get_nt_for_task('and', 4)
        reference = np.asarray(algorithm.evaluate_with_neurotransmitter(nt, X, Y))
        np.testing.assert_array_equal(
            np.asarray(algorithm.evaluate_with_neurotransmitter('and', X, Y)), reference
        )
        np.testing.assert_array_equal(
            np.asarray(algorithm.evaluate_with_custom_nt(nt, X, Y)), reference
        )
        direct = algorithm._eval_population_true_neuromodulated(
            algorithm._cached_W1,
            algorithm._cached_W2,
            algorithm._neuromod_true['receptor_densities'],
            algorithm._neuromod_true['base_gains'],
            jnp.asarray(nt),
            jnp.asarray(X),
            jnp.asarray(Y),
            'mse',
            'full',
            None,
        )
        np.testing.assert_array_equal(np.asarray(direct), reference)
        product = algorithm.evaluate_multi_task_true_neuromodulated(self._tasks(), 'product')
        xor = np.asarray(algorithm.evaluate_with_neurotransmitter('xor', *self._tasks()['xor']))
        np.testing.assert_array_equal(np.asarray(product), xor * reference)
        with pytest.raises(ValueError, match='NT vector shape'):
            algorithm.evaluate_with_custom_nt([1.0, 0.0], X, Y)
        with pytest.raises(ValueError, match="unknown activation 'sine'"):
            algorithm.evaluate_with_neurotransmitter(nt, X, Y, activation_name='sine')


class TestPerGenomeBonuses:
    """HMR's per-genome orthogonality and specialization bonuses. EMR's constant versions add one
    value to every genome (the tasks' NT vectors; the population's mean fitness variance) and raise;
    HMR's defects are fixed: the fingerprints count each genome's active positions only, and with
    two tasks the specialization compares with the other task's vector (HMR's (t+2)%2 was the right
    one)."""

    @staticmethod
    def _run(**multitask):
        from emr_hyperneat.emrhyperneat import (
            get_nt_for_task,
        )
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
            population_size=20,
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'].update(multitask)
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(config), MultiTaskProblem('xor'), seed=DEFAULT_SEED
        )
        problems = [MultiTaskProblem('xor'), MultiTaskProblem('and')]
        nts = [get_nt_for_task(p.name, 4) for p in problems]
        algorithm.run_generation_multitask(
            state, problems=problems, neurotransmitters=nts, aggregation_method='mean'
        )
        return algorithm, problems, nts

    def test_orthogonality_is_per_genome_at_active_positions(self):
        import numpy as np
        from emr_hyperneat.emrhyperneat import (
            compute_population_orthogonality,
        )
        algorithm, problems, nts = self._run(orthogonality_bonus_weight=0.3)
        record = algorithm._last_multitask
        per_task = np.stack([np.asarray(f) for f in record['per_task']])
        orthogonality = np.asarray(record['orthogonality'])
        np.testing.assert_allclose(
            np.asarray(record['aggregated']),
            per_task.mean(axis=0) + 0.3 * orthogonality,
            rtol=0,
            atol=1e-6,
        )
        assert np.std(orthogonality) > 0, "one value per genome, not one constant"
        evaluator = algorithm._create_eval_population_true_neuromodulated(return_hidden=True)
        mask = np.asarray(algorithm._cached_masks_A, dtype=np.float32)[:, None, :]
        hidden, outside = {}, 0.0
        for t, (problem, nt) in enumerate(zip(problems, nts)):
            _, h = evaluator(
                algorithm._cached_W1,
                algorithm._cached_W2,
                algorithm._neuromod_true['receptor_densities'],
                algorithm._neuromod_true['base_gains'],
                jnp.asarray(nt),
                jnp.asarray(problem.inputs),
                jnp.asarray(problem.targets),
                'mse',
                'full',
                None,
            )
            hidden[f'task_{t}'] = np.asarray(h) * mask
            outside = max(outside, float(np.max(np.abs(np.asarray(h) * (1.0 - mask)))))
        assert outside > 0, "the activity at inactive positions the fingerprints leave out"
        np.testing.assert_allclose(
            orthogonality, np.asarray(compute_population_orthogonality(hidden)), rtol=0, atol=1e-6
        )

    def test_specialization_with_two_tasks(self):
        import numpy as np
        algorithm, problems, nts = self._run(specialization_bonus_weight=0.5)
        record = algorithm._last_multitask
        per_task = np.stack([np.asarray(f) for f in record['per_task']])
        wrong = np.stack(
            [
                np.asarray(
                    algorithm.evaluate_with_neurotransmitter(
                        nts[1], problems[0].inputs, problems[0].targets
                    )
                ),
                np.asarray(
                    algorithm.evaluate_with_neurotransmitter(
                        nts[0], problems[1].inputs, problems[1].targets
                    )
                ),
            ]
        )
        expected = per_task.mean(axis=0) - wrong.mean(axis=0)
        np.testing.assert_allclose(
            np.asarray(record['specialization']), expected, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(record['aggregated']),
            per_task.mean(axis=0) + 0.5 * expected,
            rtol=0,
            atol=1e-6,
        )

    def test_the_call_form_carries_them(self):
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        algorithm, state = TestHmrCallForm._setup()
        _, metrics = algorithm.run_generation_multitask(state, MultiTaskConfig(
            tasks=TestHmrCallForm._tasks(), aggregation='mean', orthogonality_bonus_weight=0.2,
            orthogonality_metric='cosine_max', specialization_bonus_weight=0.1))
        assert algorithm._last_multitask['orthogonality'] is not None
        assert algorithm._last_multitask['specialization'] is not None
        assert metrics.aggregated_fitness > 0

    @pytest.mark.parametrize(
        'settings, reason',
        [
            (
                dict(orthogonality_metric='cosine_max'),
                'applies only with multitask.orthogonality_bonus_weight',
            ),
            (
                dict(orthogonality_bonus_weight=0.2, orthogonality_metric='angle'),
                'orthogonality_metric must be one of',
            ),
            (
                dict(orthogonality_bonus_weight=-0.2),
                'orthogonality_bonus_weight must be a number at least 0',
            ),
            (
                dict(specialization_bonus_weight='high'),
                'specialization_bonus_weight must be a number',
            ),
        ],
    )
    def test_settings_are_checked(self, settings, reason):
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
            population_size=20,
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'].update(settings)
        with pytest.raises(ValueError, match=reason):
            EMRHyperNEAT().create_config(config)

    def test_specialization_needs_two_tasks(self):
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        algorithm, state = TestHmrCallForm._setup()
        tasks = {'xor': TestHmrCallForm._tasks()['xor']}
        with pytest.raises(ValueError, match='needs two or more tasks'):
            algorithm.run_generation_multitask(
                state, MultiTaskConfig(tasks=tasks, specialization_bonus_weight=0.5)
            )


class TestBranchGatingAndModulationPenalty:
    """HMR's multi-task branch gating and modulation penalty (0.1.0 accepted both and applied
    neither).

    Branch gating gives each region of the grid one receptor channel
    (compute_modulation_with_branch_gating); the modulation penalty, a bonus despite its name, adds
    penalty times the variance over tasks of a genome's mean modulation. HMR's defects are fixed:
    its penalty averaged the receptors over every position of the grid and multiplied them with
    every channel, the inversion channel included, whatever the branch mode; its branch-gated
    evaluator was feed-forward and tanh only, and two-module mode silently took priority over it."""

    @staticmethod
    def _numpy_fitness(algorithm, nt, inputs, targets, branch_mode):
        """A task's fitness recomputed in NumPy from the generation's cached arrays: the paper's
        rule, 'full' modulation, tanh hidden nodes, sigmoid outputs, inversion by nt[3], 1 - MSE."""
        import numpy as np
        from emr_hyperneat.emrhyperneat import (
            get_hierarchical_grid,
        )

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        W1 = np.asarray(algorithm._cached_W1, dtype=np.float64)
        W2 = np.asarray(algorithm._cached_W2, dtype=np.float64)
        rd = np.asarray(algorithm._neuromod_true['receptor_densities'], dtype=np.float64)
        bg = np.asarray(algorithm._neuromod_true['base_gains'], dtype=np.float64)
        nt = np.asarray(nt, dtype=np.float64)
        positions = np.asarray(
            get_hierarchical_grid(algorithm.max_depth).all_positions, dtype=np.float64
        )
        x, y = positions[:, 0], positions[:, 1]
        if branch_mode == 'spatial':
            m = np.where(
                (x < 0) & (y < 0),
                rd[..., 0] * nt[0],
                np.where(
                    (x >= 0) & (y < 0),
                    rd[..., 1] * nt[1],
                    np.where((x < 0) & (y >= 0), rd[..., 2] * nt[2], rd[..., :3] @ nt[:3]),
                ),
            )
        elif branch_mode == 'hierarchical':
            m = np.where(
                y < -0.33,
                rd[..., 0] * nt[0],
                np.where(y < 0.33, rd[..., 1] * nt[1], rd[..., 2] * nt[2]),
            )
        else:
            m = rd[..., :3] @ nt[:3]
        strength = algorithm.neuromod_config.modulation_strength
        gains = np.clip(bg + strength * m, 0.1, 5.0)
        pre = np.einsum('ci,pih->pch', np.asarray(inputs, dtype=np.float64), W1)
        hidden = (
            np.tanh(gains[:, None, :] * pre + strength * m[:, None, :]) * sigmoid(m)[:, None, :]
        )
        outputs = sigmoid(np.einsum('pch,pho->pco', hidden, W2))
        invert = np.clip(nt[3], 0.0, 1.0)
        outputs = invert * outputs + (1.0 - invert) * (1.0 - outputs)
        return 1.0 - np.mean(
            (outputs - np.asarray(targets, dtype=np.float64)[None]) ** 2, axis=(1, 2)
        )

    @staticmethod
    def _mean_modulation_variance(algorithm, nts, branch_mode='none'):
        """The penalty's variance recomputed in NumPy: per genome and task, the mean modulation over
        the genome's active positions; the variance over tasks."""
        import numpy as np
        from emr_hyperneat.emrhyperneat import (
            compute_modulation_with_branch_gating, get_hierarchical_grid,
        )
        rd = np.asarray(algorithm._neuromod_true['receptor_densities'], dtype=np.float64)
        active = np.asarray(algorithm._cached_masks_A, dtype=bool)
        positions = get_hierarchical_grid(algorithm.max_depth).all_positions
        means = []
        for nt in nts:
            if branch_mode == 'none':
                m = rd[..., :3] @ np.asarray(nt[:3], dtype=np.float64)
            else:
                m = np.stack(
                    [
                        np.asarray(
                            compute_modulation_with_branch_gating(
                                jnp.asarray(r, dtype=jnp.float32),
                                jnp.asarray(nt),
                                positions,
                                branch_mode,
                            )
                        )
                        for r in rd
                    ]
                )
            means.append(np.where(active, m, 0.0).sum(axis=1) / np.maximum(active.sum(axis=1), 1))
        return np.var(np.stack(means), axis=0)

    def test_the_routing(self):
        import numpy as np
        from emr_hyperneat.emrhyperneat import (
            compute_modulation_with_branch_gating,
        )
        positions = jnp.asarray([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.0]])
        receptors = jnp.tile(jnp.asarray([[1.0, 2.0, 3.0, 4.0]]), (5, 1))
        nt = jnp.asarray([0.1, 0.2, 0.3, 0.9])
        for mode, expected in (('none', [1.4] * 5), ('spatial', [0.1, 0.4, 0.9, 1.4, 1.4]),
                               ('hierarchical', [0.1, 0.1, 0.9, 0.9, 0.4])):
            np.testing.assert_allclose(
                np.asarray(compute_modulation_with_branch_gating(receptors, nt, positions, mode)),
                expected,
                rtol=1e-6,
                err_msg=mode,
            )

    @pytest.mark.parametrize('branch_mode', ['spatial', 'hierarchical'])
    def test_branch_gated_fitness_recomputed(self, branch_mode):
        import numpy as np
        algorithm, problems, nts = TestPerGenomeBonuses._run(branch_gating_mode=branch_mode)
        record = algorithm._last_multitask
        routed_differs = False
        for t, (problem, nt) in enumerate(zip(problems, nts)):
            expected = self._numpy_fitness(
                algorithm, nt, problem.inputs, problem.targets, branch_mode
            )
            np.testing.assert_allclose(
                np.asarray(record['per_task'][t]), expected, rtol=0, atol=1e-5
            )
            unrouted = self._numpy_fitness(algorithm, nt, problem.inputs, problem.targets, 'none')
            routed_differs = routed_differs or bool(np.max(np.abs(expected - unrouted)) > 1e-4)
        assert routed_differs, "the routing changes some genome's fitness"
        assert algorithm._last_multitask['modulation_variance'] is None

    def test_evaluate_with_neurotransmitter_applies_the_routing(self):
        import numpy as np
        algorithm, problems, nts = TestPerGenomeBonuses._run(branch_gating_mode='spatial')
        X, Y = problems[1].inputs, problems[1].targets
        np.testing.assert_array_equal(
            np.asarray(
                algorithm.evaluate_with_neurotransmitter(nts[1], X, Y, branch_gating_mode='spatial')
            ),
            np.asarray(algorithm._last_multitask['per_task'][1]),
        )
        with pytest.raises(ValueError, match='branch_gating_mode must be one of'):
            algorithm.evaluate_with_neurotransmitter(nts[1], X, Y, branch_gating_mode='diagonal')

    def test_recurrent_settling_takes_the_routed_node(self):
        """With hidden-to-hidden connections the routed modulation reaches every settling step: the
        generation's fitness equals settle_hidden with the paper's node on the routed modulation,
        and differs from the unrouted one."""
        import jax
        import numpy as np
        from emr_hyperneat.emrhyperneat import (
            compute_fitness,
            compute_modulation_with_branch_gating,
            get_hierarchical_grid,
            get_nt_for_task,
            invert_outputs,
            settle_hidden,
            true_neuromodulation_node_fn,
        )

        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            fitness_aggregation='mean',
            recurrence_preset='hidden_only',
            population_size=20,
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'][
            'branch_gating_mode'
        ] = 'spatial'
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(config), MultiTaskProblem('xor'), seed=DEFAULT_SEED
        )
        problems = [MultiTaskProblem('xor'), MultiTaskProblem('and')]
        nts = [get_nt_for_task(p.name, 4) for p in problems]
        algorithm.run_generation_multitask(
            state, problems=problems, neurotransmitters=nts, aggregation_method='mean'
        )
        hh = algorithm._cached_sparse_hh
        assert hh is not None and int(np.sum(np.asarray(hh.valid_mask))) > 0
        positions = get_hierarchical_grid(algorithm.max_depth).all_positions
        strength = algorithm.neuromod_config.modulation_strength

        def fitness(routed, nt, X, Y):
            def one(W1, W2, rd, bg, src, dst, w, valid):
                modulation = (
                    compute_modulation_with_branch_gating(rd, nt, positions, 'spatial')
                    if routed
                    else None
                )
                node = true_neuromodulation_node_fn(
                    jnp.tanh, rd, bg, nt, strength, 4, 'full', 'paper', modulation
                )
                hidden = settle_hidden(X @ W1, src, dst, w, valid, algorithm.activate_time, node)
                outputs = invert_outputs(jax.nn.sigmoid(hidden @ W2), nt, True, 4)
                return compute_fitness(outputs, Y, 'mse')
            return np.asarray(
                jax.vmap(one)(
                    algorithm._cached_W1,
                    algorithm._cached_W2,
                    algorithm._neuromod_true['receptor_densities'],
                    algorithm._neuromod_true['base_gains'],
                    hh.from_indices,
                    hh.to_indices,
                    hh.weights,
                    hh.valid_mask,
                )
            )

        differs = False
        for t, (problem, nt) in enumerate(zip(problems, nts)):
            X, Y = jnp.asarray(problem.inputs), jnp.asarray(problem.targets)
            routed = fitness(True, nt, X, Y)
            np.testing.assert_allclose(
                np.asarray(algorithm._last_multitask['per_task'][t]), routed, rtol=0, atol=1e-6
            )
            differs = differs or bool(np.max(np.abs(routed - fitness(False, nt, X, Y))) > 1e-4)
        assert differs

    @pytest.mark.parametrize('branch_mode', ['none', 'spatial'])
    def test_modulation_penalty_is_per_genome_at_active_positions(self, branch_mode):
        import numpy as np
        algorithm, problems, nts = TestPerGenomeBonuses._run(
            modulation_penalty=0.05, branch_gating_mode=branch_mode
        )
        record = algorithm._last_multitask
        variance = np.asarray(record['modulation_variance'])
        np.testing.assert_allclose(
            variance,
            self._mean_modulation_variance(algorithm, nts, branch_mode),
            rtol=1e-5,
            atol=1e-8,
        )
        assert np.std(variance) > 0, "one value per genome, not one constant"
        per_task = np.stack([np.asarray(f) for f in record['per_task']])
        np.testing.assert_allclose(
            np.asarray(record['aggregated']),
            per_task.mean(axis=0) + 0.05 * variance,
            rtol=0,
            atol=1e-6,
        )
        # HMR's version (every position, every channel of the NT vectors) gives other values
        rd = np.asarray(algorithm._neuromod_true['receptor_densities'], dtype=np.float64)
        hmr = np.var(
            np.stack([rd.mean(axis=1) @ np.asarray(nt, dtype=np.float64) for nt in nts]), axis=0
        )
        assert np.max(np.abs(hmr - variance)) > 1e-4

    def test_the_call_form_carries_them(self):
        import numpy as np
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        algorithm, state = TestHmrCallForm._setup()
        _, metrics = algorithm.run_generation_multitask(
            state,
            MultiTaskConfig(
                tasks=TestHmrCallForm._tasks(),
                aggregation='product',
                branch_gating_mode='hierarchical',
                modulation_penalty=0.05,
            ),
        )
        record = algorithm._last_multitask
        np.testing.assert_allclose(
            np.asarray(record['aggregated']),
            np.asarray(record['per_task'][0] * record['per_task'][1])
            + 0.05 * np.asarray(record['modulation_variance']),
            rtol=0,
            atol=1e-6,
        )
        assert (
            algorithm.multitask_config.branch_gating_mode == 'none'
        ), "the call's configuration is the call's"
        assert metrics.aggregated_fitness > 0

    def test_the_configuration_carries_them(self):
        algorithm, problems, nts = TestPerGenomeBonuses._run(
            branch_gating_mode='hierarchical', modulation_penalty=0.1
        )
        assert algorithm.multitask_config.branch_gating_mode == 'hierarchical'
        assert algorithm.multitask_config.modulation_penalty == 0.1
        assert algorithm._last_multitask['modulation_variance'] is not None

    @pytest.mark.parametrize(
        'settings, reason',
        [
            (dict(branch_gating_mode='diagonal'), 'branch_gating_mode must be one of'),
            (
                dict(branch_gating_mode='spatial', two_module_mode='parallel'),
                'both set the modulation by region',
            ),
            (dict(modulation_penalty=-0.1), 'modulation_penalty must be a number at least 0'),
            (dict(modulation_penalty='high'), 'modulation_penalty must be a number at least 0'),
        ],
    )
    def test_settings_are_checked(self, settings, reason):
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
            population_size=20,
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'].update(settings)
        with pytest.raises(ValueError, match=reason):
            EMRHyperNEAT().create_config(config)

    def test_branch_gating_needs_three_nt_types(self):
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
            population_size=20,
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['neuromodulation']['num_nt_types'] = 2
        hmr['multitask']['branch_gating_mode'] = 'spatial'
        with pytest.raises(ValueError, match='routes receptor channels 0 to 2'):
            EMRHyperNEAT().create_config(config)

    def test_the_penalty_needs_two_tasks(self):
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        config = create_config_with_multi_task(task_names=['xor'], fitness_aggregation='mean',
                                               recurrence_preset='feedforward', population_size=20)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multitask'][
            'modulation_penalty'
        ] = 0.05
        with pytest.raises(ValueError, match='needs two or more tasks'):
            EMRHyperNEAT().create_config(config)
        algorithm, state = TestHmrCallForm._setup()
        tasks = {'xor': TestHmrCallForm._tasks()['xor']}
        generation = algorithm._current_generation
        with pytest.raises(ValueError, match='needs two or more tasks'):
            algorithm.run_generation_multitask(
                state, MultiTaskConfig(tasks=tasks, modulation_penalty=0.05)
            )
        assert algorithm._current_generation == generation, "refused before the generation ran"
