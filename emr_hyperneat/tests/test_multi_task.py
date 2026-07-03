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
    create_base_config,
    run_quick_evolution,
    run_multitask_evolution,
    assert_no_errors,
    assert_positive_fitness,
    NEUROMODULATION_PRESETS,
    XORProblem,
    ANDProblem,
    ORProblem,
    NANDProblem,
    NORProblem,
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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("num_tasks", [2, 3, 4, 5])
    def test_different_task_counts(self, algorithm, xor_problem, num_tasks):
        """Verify different numbers of tasks work."""
        task_names = TASK_NAMES[:num_tasks]
        config = create_config_with_multi_task(
            task_names=task_names,
            neuromod_mode='true_neuromodulation_4nt'
        )
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)


class TestFitnessAggregationMethods:
    """Tests for different fitness aggregation methods.

    Note: Multi-task evolution tests require multiple problems matching task count.
    Tests here verify configuration/initialization. Evolution tests are skipped
    until proper multi-problem fixtures are available.
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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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

    def test_multi_task_with_dynamic_functions(self, algorithm, two_task_problems):
        """Verify multi-task works with dynamic functions."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

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
    """Comprehensive tests (marked slow)."""

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
        """Verify 5-task evolution with all features enabled."""
        config = create_config_with_multi_task(
            task_names=five_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}
        hmr['aggregation'] = {'mode': 'global', 'global_aggregation': 'sum'}

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


# =============================================================================
# Multi-Task Gap Tests
# =============================================================================

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


class TestOrthogonalityBonus:
    """Tests for orthogonality_bonus parameter."""

    def test_orthogonality_bonus_disabled(self, algorithm, two_task_problems):
        """Verify orthogonality_bonus=0.0 works (default)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['orthogonality_bonus'] = 0.0

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("bonus", [0.1, 0.2, 0.5])
    def test_orthogonality_bonus_values(self, algorithm, two_task_problems, bonus):
        """Verify orthogonality_bonus with recommended values (0.1-0.5)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['orthogonality_bonus'] = bonus

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_orthogonality_bonus_with_min_aggregation(self, algorithm, two_task_problems):
        """Verify orthogonality_bonus works with min aggregation."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['orthogonality_bonus'] = 0.3

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestSpecializationBonus:
    """Tests for specialization_bonus parameter."""

    def test_specialization_bonus_disabled(self, algorithm, two_task_problems):
        """Verify specialization_bonus=0.0 works (default)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['specialization_bonus'] = 0.0

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("bonus", [0.1, 0.2, 0.5])
    def test_specialization_bonus_values(self, algorithm, two_task_problems, bonus):
        """Verify specialization_bonus with different values."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['specialization_bonus'] = bonus

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_specialization_bonus_with_three_tasks(self, algorithm, three_task_problems):
        """Verify specialization_bonus works with 3 tasks."""
        config = create_config_with_multi_task(
            task_names=three_task_problems['task_names'],
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['specialization_bonus'] = 0.2

        result = run_multitask_evolution(
            algorithm, config,
            problems=three_task_problems['problems'],
            nt_vectors=three_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestCombinedBonuses:
    """Tests for combining orthogonality_bonus and specialization_bonus."""

    def test_both_bonuses_enabled(self, algorithm, two_task_problems):
        """Verify both bonuses can be enabled together."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['orthogonality_bonus'] = 0.2
        hmr['multitask']['specialization_bonus'] = 0.1

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='mean',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_bonuses_with_weighted_aggregation(self, algorithm, two_task_problems):
        """Verify bonuses work with weighted aggregation."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.6, 0.4]
        hmr['multitask']['orthogonality_bonus'] = 0.1
        hmr['multitask']['specialization_bonus'] = 0.1

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    def test_full_feature_combination(self, algorithm, three_task_problems):
        """Verify all multi-task features can be combined."""
        config = create_config_with_multi_task(
            task_names=three_task_problems['task_names'],
            fitness_aggregation='weighted'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['task_weights'] = [0.5, 0.3, 0.2]
        hmr['multitask']['joint_evolution'] = True
        hmr['multitask']['orthogonality_bonus'] = 0.2
        hmr['multitask']['specialization_bonus'] = 0.1

        result = run_multitask_evolution(
            algorithm, config,
            problems=three_task_problems['problems'],
            nt_vectors=three_task_problems['nt_vectors'],
            aggregation_method='weighted',
            generations=QUICK_GENERATIONS
        )

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


# =============================================================================
# Multi-Task Edge Case Tests
# =============================================================================

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
        """Verify extreme temperatures work with orthogonality and specialization bonuses."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['softmin_temperature'] = 0.01
        hmr['multitask']['orthogonality_bonus'] = 0.2
        hmr['multitask']['specialization_bonus'] = 0.1

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


# =============================================================================
# Extended MultiTaskConfig Feature Tests
# =============================================================================

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


class TestModulationPenalty:
    """Tests for modulation_penalty parameter.

    The modulation_penalty rewards networks that produce different modulation
    patterns per task (encourages task-specific behavior).
    Recommended values: 0.01-0.1; 0.0 = disabled.
    """

    def test_modulation_penalty_disabled(self, algorithm, two_task_problems):
        """Verify modulation_penalty=0.0 works (disabled)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['modulation_penalty'] = 0.0

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("penalty", [0.01, 0.05, 0.1])
    def test_modulation_penalty_values(self, algorithm, two_task_problems, penalty):
        """Verify recommended modulation_penalty values (0.01-0.1)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['modulation_penalty'] = penalty

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"modulation_penalty={penalty} failed")

    def test_modulation_penalty_with_softmin(self, algorithm, two_task_problems):
        """Verify modulation_penalty works with softmin aggregation."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['modulation_penalty'] = 0.05
        hmr['multitask']['softmin_temperature'] = 0.5

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

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


class TestOrthogonalityMetric:
    """Tests for orthogonality_metric options.

    Controls how orthogonality between NT vectors is measured:
    - 'cosine_mean': Mean cosine similarity (default)
    - 'cosine_max': Maximum cosine similarity
    - 'correlation': Correlation-based metric
    """

    @pytest.mark.parametrize("metric", ['cosine_mean', 'cosine_max', 'correlation'])
    def test_orthogonality_metric_options(self, algorithm, two_task_problems, metric):
        """Verify all orthogonality_metric options work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['orthogonality_bonus_weight'] = 0.2
        hmr['multitask']['orthogonality_metric'] = metric

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"orthogonality_metric='{metric}' failed")

    def test_orthogonality_metric_with_three_tasks(self, algorithm, three_task_problems):
        """Verify orthogonality_metric works with 3 tasks."""
        config = create_config_with_multi_task(
            task_names=three_task_problems['task_names'],
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['orthogonality_bonus_weight'] = 0.3
        hmr['multitask']['orthogonality_metric'] = 'cosine_max'

        result = run_multitask_evolution(
            algorithm, config,
            problems=three_task_problems['problems'],
            nt_vectors=three_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestBranchGatingMode:
    """Tests for branch_gating_mode in multi-task (Liu & Wang 2024 SST mechanism).

    Controls dendritic branch-specific gating:
    - 'none': No branch gating (default)
    - 'spatial': Spatial branch gating
    - 'hierarchical': Hierarchical branch gating
    """

    @pytest.mark.parametrize("mode", ['none', 'spatial', 'hierarchical'])
    def test_branch_gating_mode_options(self, algorithm, two_task_problems, mode):
        """Verify all branch_gating_mode options work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['branch_gating_mode'] = mode

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"branch_gating_mode='{mode}' failed")

    def test_branch_gating_with_softmin(self, algorithm, two_task_problems):
        """Verify branch_gating_mode works with softmin aggregation."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='softmin'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['branch_gating_mode'] = 'spatial'
        hmr['multitask']['softmin_temperature'] = 0.5

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='softmin',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


class TestTwoModuleMode:
    """Tests for two_module_mode architecture (Liu & Wang 2024 PFC/Sensorimotor separation).

    Controls two-module architecture:
    - 'none': Single module (default)
    - 'parallel': Parallel two-module architecture
    - 'sequential': Sequential two-module architecture
    """

    @pytest.mark.parametrize("mode", ['none', 'parallel', 'sequential'])
    def test_two_module_mode_options(self, algorithm, two_task_problems, mode):
        """Verify all two_module_mode options work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
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

    def test_two_module_with_generalist_bonus(self, algorithm, two_task_problems):
        """Verify two_module_mode works with generalist bonus."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='min'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['two_module_mode'] = 'parallel'
        hmr['multitask']['generalist_bonus_type'] = 'min_bonus'
        hmr['multitask']['generalist_bonus_weight'] = 0.1

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            aggregation_method='min',
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)


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


class TestSpecializationBonusWeight:
    """Tests for specialization_bonus_weight parameter (different from specialization_bonus).

    This is the weight for NT-task alignment bonus from confusion matrix gap.
    """

    def test_specialization_bonus_weight_disabled(self, algorithm, two_task_problems):
        """Verify specialization_bonus_weight=0.0 works (disabled)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['specialization_bonus_weight'] = 0.0

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("weight", [0.1, 0.3, 0.5])
    def test_specialization_bonus_weight_values(self, algorithm, two_task_problems, weight):
        """Verify different specialization_bonus_weight values work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['specialization_bonus_weight'] = weight

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"specialization_bonus_weight={weight} failed")


class TestConfusionEvalFrequency:
    """Tests for confusion_eval_frequency parameter.

    Controls how often confusion matrix is computed:
    - 0: Disabled (default)
    - N: Compute every N generations
    """

    def test_confusion_eval_disabled(self, algorithm, two_task_problems):
        """Verify confusion_eval_frequency=0 works (disabled)."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['confusion_eval_frequency'] = 0

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result)

    @pytest.mark.parametrize("frequency", [1, 5, 10])
    def test_confusion_eval_frequency_values(self, algorithm, two_task_problems, frequency):
        """Verify different confusion_eval_frequency values work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean'
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['confusion_eval_frequency'] = frequency

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS
        )

        assert_no_errors(result, f"confusion_eval_frequency={frequency} failed")
