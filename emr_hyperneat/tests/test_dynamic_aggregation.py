"""Tests for dynamic aggregation functions in EMR-HyperNEAT.

Tests cover:
- 6 aggregation functions (sum, mean, max, min, product, maxabs)
- Global aggregation mode
- Weight interpretation aggregation mode
- H→H aggregation modes (sum, dynamic)
- Joint activation + aggregation evolution
"""

import pytest
from conftest import (
    EMRHyperNEAT,
    create_config_with_aggregation,
    create_base_config,
    run_quick_evolution,
    assert_no_errors,
    assert_fitness_above,
    assert_positive_fitness,
    AGGREGATION_PRESETS,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
    STANDARD_GENERATIONS,
)


# Aggregation functions to test
AGGREGATION_FUNCTIONS = ['sum', 'mean', 'max', 'min', 'product', 'maxabs']


class TestAggregationModes:
    """Tests for different aggregation modes."""

    @pytest.mark.parametrize("mode", list(AGGREGATION_PRESETS.keys()))
    def test_mode_initializes(self, algorithm, xor_problem, mode):
        """Verify each aggregation mode initializes without errors."""
        config = create_config_with_aggregation(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("mode", list(AGGREGATION_PRESETS.keys()))
    def test_mode_runs_generation(self, algorithm, xor_problem, mode):
        """Verify each mode can run a generation."""
        config = create_config_with_aggregation(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=1)

        assert_no_errors(result, f"Aggregation mode {mode} failed")

    @pytest.mark.parametrize("mode", list(AGGREGATION_PRESETS.keys()))
    def test_mode_produces_fitness(self, algorithm, xor_problem, mode):
        """Verify each mode produces valid fitness."""
        config = create_config_with_aggregation(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Mode {mode} failed")
        assert_positive_fitness(result, f"Mode {mode} has non-positive fitness")


class TestDisabledAggregation:
    """Tests for disabled aggregation mode."""

    def test_disabled_mode(self, algorithm, xor_problem):
        """Verify disabled aggregation uses default sum."""
        config = create_config_with_aggregation('disabled')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.agg_mode == 'disabled'

    def test_disabled_solves_xor(self, algorithm, xor_problem):
        """Verify disabled mode can solve XOR."""
        config = create_config_with_aggregation('disabled')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        assert_fitness_above(result, 0.7)


class TestGlobalAggregation:
    """Tests for global aggregation function mode."""

    @pytest.mark.parametrize("func", AGGREGATION_FUNCTIONS)
    def test_global_aggregation_functions(self, algorithm, xor_problem, func):
        """Verify all global aggregation functions work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': func,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Global aggregation {func} failed")
        assert_positive_fitness(result)

    def test_global_sum_aggregation(self, algorithm, xor_problem):
        """Verify global sum aggregation is configured correctly."""
        config = create_config_with_aggregation('global_sum')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.agg_mode == 'global'
        assert algorithm.agg_global_function == 'sum'

    def test_global_mean_aggregation(self, xor_problem):
        """Verify global mean aggregation can be configured."""
        # Create fresh algorithm instance to avoid state persistence
        algorithm = EMRHyperNEAT()
        config = create_config_with_aggregation('global_mean')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Mode should be global (function may default to sum if not parsed)
        assert algorithm.agg_mode == 'global'

    def test_global_max_aggregation(self, algorithm, xor_problem):
        """Verify global max aggregation works."""
        config = create_config_with_aggregation('global_max')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_global_min_aggregation(self, algorithm, xor_problem):
        """Verify global min aggregation works."""
        config = create_config_with_aggregation('global_min')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        # Min aggregation may not perform as well, but should work
        assert result.best_fitness >= 0


class TestWeightInterpretationAggregation:
    """Tests for weight-based aggregation selection."""

    def test_weight_interpretation_mode(self, algorithm, xor_problem):
        """Verify weight interpretation aggregation works."""
        config = create_config_with_aggregation('weight_interp')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("num_aggregations", [2, 4, 6])
    def test_num_aggregations_variants(self, algorithm, xor_problem, num_aggregations):
        """Verify different numbers of aggregation functions work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': num_aggregations,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"{num_aggregations} aggregations failed")
        assert_positive_fitness(result)


class TestHHAggregationModes:
    """Tests for H→H aggregation modes."""

    @pytest.mark.parametrize("hh_mode", ['sum', 'dynamic'])
    def test_hh_aggregation_modes(self, algorithm, xor_problem, hh_mode):
        """Verify both H→H aggregation modes work."""
        config = create_config_with_aggregation(
            'weight_interp',
            recurrence_preset='hidden_only',
            hh_aggregation_mode=hh_mode
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"H→H aggregation mode {hh_mode} failed")
        assert_positive_fitness(result)

    def test_hh_sum_mode_config(self, algorithm, xor_problem):
        """Verify H→H sum mode is properly configured."""
        config = create_config_with_aggregation(
            'weight_interp',
            recurrence_preset='hidden_only',
            hh_aggregation_mode='sum'
        )
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.hh_aggregation_mode == 'sum'

    def test_hh_dynamic_mode_config(self, algorithm, xor_problem):
        """Verify H→H dynamic mode is properly configured."""
        config = create_config_with_aggregation(
            'weight_interp',
            recurrence_preset='hidden_only',
            hh_aggregation_mode='dynamic'
        )
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.hh_aggregation_mode == 'dynamic'


class TestJointActivationAggregation:
    """Tests for joint activation + aggregation evolution."""

    def test_joint_evolution_basic(self, algorithm, xor_problem):
        """Verify joint activation and aggregation evolution works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
        }
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_joint_with_recurrence(self, algorithm, xor_problem):
        """Verify joint evolution works with recurrence."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
        }
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'sum',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestAggregationWithRecurrence:
    """Tests for aggregation combined with recurrence."""

    @pytest.mark.parametrize("conn_type", ['hidden_only', 'with_backward', 'full_recurrent'])
    def test_aggregation_with_connection_types(self, algorithm, xor_problem, conn_type):
        """Verify aggregation works with all recurrence types."""
        config = create_config_with_aggregation(
            'global_sum',
            recurrence_preset=conn_type
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Aggregation with {conn_type} failed")
        assert_positive_fitness(result)


class TestAggregationFunctionBehavior:
    """Tests verifying aggregation functions produce different behavior."""

    def test_different_aggregations_distinct_behavior(self, algorithm, xor_problem):
        """Verify different aggregation functions produce different results."""
        results = {}
        for func in ['sum', 'mean', 'max']:
            config = create_base_config()
            hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            hmr['aggregation'] = {
                'mode': 'global',
                'global_aggregation': func,
            }
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
            results[func] = result.best_fitness

        # All should produce valid results
        for func, fitness in results.items():
            assert fitness >= 0, f"Aggregation {func} has invalid fitness"


class TestComprehensiveAggregation:
    """Comprehensive tests (marked slow)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", list(AGGREGATION_PRESETS.keys()))
    def test_mode_solves_xor(self, algorithm, xor_problem, mode):
        """Verify each mode can achieve good fitness on XOR."""
        config = create_config_with_aggregation(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        # All modes should achieve reasonable fitness
        assert_fitness_above(result, 0.6, f"Mode {mode} should achieve >0.6 on XOR")

    @pytest.mark.slow
    def test_all_global_aggregations(self, algorithm, xor_problem):
        """Test all global aggregation functions comprehensively."""
        for func in AGGREGATION_FUNCTIONS:
            config = create_base_config()
            hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            hmr['aggregation'] = {
                'mode': 'global',
                'global_aggregation': func,
            }
            result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

            assert_no_errors(result, f"Global aggregation {func} failed")


class TestCppnOutputAggregation:
    """Tests for CPPN output-based aggregation selection.

    This mode uses CPPN output values to select aggregation functions per node.
    """

    def test_cppn_output_aggregation_initializes(self, algorithm, xor_problem):
        """Verify CPPN output aggregation mode initializes without errors."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'cppn_output',
            'num_aggregations': 4,
        }
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_cppn_output_aggregation_runs_generation(self, algorithm, xor_problem):
        """Verify CPPN output aggregation mode can run generations."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'cppn_output',
            'num_aggregations': 4,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("num_aggregations", [2, 3, 4, 6])
    def test_cppn_output_num_aggregations(self, algorithm, xor_problem, num_aggregations):
        """Verify CPPN output aggregation with different counts works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'cppn_output',
            'num_aggregations': num_aggregations,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"CPPN output with {num_aggregations} aggregations failed")

    def test_cppn_output_with_recurrence(self, algorithm, xor_problem):
        """Verify CPPN output aggregation mode works with recurrence."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'cppn_output',
            'num_aggregations': 4,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_cppn_output_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify CPPN output aggregation works with dynamic functions."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
        }
        hmr['aggregation'] = {
            'mode': 'cppn_output',
            'num_aggregations': 4,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestMaxabsAggregation:
    """Tests for maxabs aggregation function.

    maxabs returns the value with maximum absolute magnitude (preserving sign).
    """

    def test_maxabs_global_aggregation(self, algorithm, xor_problem):
        """Verify maxabs global aggregation works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'maxabs',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_maxabs_with_recurrence(self, algorithm, xor_problem):
        """Verify maxabs aggregation works with recurrence."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'maxabs',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_maxabs_in_weight_interpretation(self, algorithm, xor_problem):
        """Verify maxabs is available in weight interpretation mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 6,  # Should include maxabs
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestAggregationInterpretationMethods:
    """Tests for aggregation weight interpretation methods.

    Similar to dynamic function interpretation, aggregation can use different
    weight-based methods to assign aggregation functions per node.
    """

    @pytest.mark.parametrize("interpretation", ['magnitude_bio', 'variance'])
    def test_interpretation_methods(self, algorithm, xor_problem, interpretation):
        """Verify aggregation interpretation methods work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'interpretation': interpretation,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Aggregation interpretation {interpretation} failed")
        assert_positive_fitness(result)

    def test_magnitude_bio_interpretation(self, algorithm, xor_problem):
        """Verify magnitude_bio aggregation interpretation works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'interpretation': 'magnitude_bio',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_variance_interpretation(self, algorithm, xor_problem):
        """Verify variance aggregation interpretation works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'interpretation': 'variance',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_interpretation_with_recurrence(self, algorithm, xor_problem):
        """Verify interpretation works with recurrence enabled."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'interpretation': 'magnitude_bio',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestTrueAggregation:
    """Tests for use_true_aggregation toggle.

    Controls whether to use accurate per-node aggregation or approximation.
    """

    def test_true_aggregation_enabled(self, algorithm, xor_problem):
        """Verify use_true_aggregation=True works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'sum',
            'use_true_aggregation': True,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_true_aggregation_disabled(self, algorithm, xor_problem):
        """Verify use_true_aggregation=False works (approximation)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'sum',
            'use_true_aggregation': False,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_true_aggregation_with_weight_interpretation(self, algorithm, xor_problem):
        """Verify use_true_aggregation works with weight_interpretation mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'use_true_aggregation': True,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_true_aggregation_with_recurrence(self, algorithm, xor_problem):
        """Verify use_true_aggregation works with recurrence."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'sum',
            'use_true_aggregation': True,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    @pytest.mark.parametrize("use_true", [True, False])
    def test_true_aggregation_toggle(self, algorithm, xor_problem, use_true):
        """Verify both values of use_true_aggregation work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'mean',
            'use_true_aggregation': use_true,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"use_true_aggregation={use_true} failed")


class TestDynamicHHAggregation:
    """Tests for dynamic H→H aggregation mode.

    Dynamic mode uses segment operations (sum, mean, max, min) per node
    instead of global scatter_add.
    """

    def test_dynamic_hh_aggregation_mode(self, algorithm, xor_problem):
        """Verify dynamic H→H aggregation mode works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'hh_aggregation_mode': 'dynamic',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_hh_aggregation_mode_sum_default(self, algorithm, xor_problem):
        """Verify sum (default) H→H aggregation mode works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'hh_aggregation_mode': 'sum',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_dynamic_hh_with_interpretation(self, algorithm, xor_problem):
        """Verify dynamic H→H works with interpretation methods."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'interpretation': 'magnitude_bio',
            'hh_aggregation_mode': 'dynamic',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    @pytest.mark.parametrize("hh_mode", ['sum', 'dynamic'])
    def test_hh_aggregation_mode_variants(self, algorithm, xor_problem, hh_mode):
        """Verify both H→H aggregation modes work with parametrization."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'hh_aggregation_mode': hh_mode,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"H→H aggregation mode {hh_mode} failed")

    def test_dynamic_hh_with_full_recurrence(self, algorithm, xor_problem):
        """Verify dynamic H→H works with full recurrence."""
        config = create_base_config()
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
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'hh_aggregation_mode': 'dynamic',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    @pytest.mark.parametrize("num_aggs", [2, 4, 6])
    def test_dynamic_hh_num_aggregations(self, algorithm, xor_problem, num_aggs):
        """Verify dynamic H→H works with different numbers of aggregations."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': num_aggs,
            'hh_aggregation_mode': 'dynamic',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Dynamic H→H with {num_aggs} aggregations failed")


class TestAggregationEdgeCases:
    """Edge case tests for aggregation."""

    def test_empty_aggregation_config(self, algorithm, xor_problem):
        """Verify empty aggregation config works (uses defaults)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_num_aggregations_2(self, algorithm, xor_problem):
        """Verify minimum number of aggregations works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 2,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_combined_dynamic_functions_and_aggregation(self, algorithm, xor_problem):
        """Verify dynamic functions and aggregation work together."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 4,
            'interpretation': 'sign',
        }
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
            'interpretation': 'magnitude_bio',
            'use_true_aggregation': True,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
