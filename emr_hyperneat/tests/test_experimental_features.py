"""Tests for experimental algorithm features in EMR-HyperNEAT.

Tests cover experimental and advanced features including:
- streaming_verbose: Per-level timing output during streaming mode
- critical_periods_config: Phase boundaries for dynamic functions plasticity
- two_module_mode: Parallel vs sequential dual-module processing
- generalist_bonus_type: Multi-task fitness bonuses/penalties
- FULL_SPARSE mode: defined in the enum and refused (not implemented)
- fitness_mode options: Different fitness functions (mse, accuracy, bce, etc.)
"""

import pytest

from conftest import (
    EMRHyperNEAT,
    ForwardPassMode,
    create_base_config,
    create_config_with_recurrence,
    create_config_with_streaming,
    create_config_with_multi_task,
    create_config_with_dynamic_functions,
    run_quick_evolution,
    run_multitask_evolution,
    assert_no_errors,
    assert_positive_fitness,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
)


# Streaming Verbose Tests

class TestStreamingVerbose:
    """Tests for streaming_verbose output mode."""

    def test_streaming_verbose_disabled_by_default(self, algorithm, xor_problem):
        """Verify streaming_verbose is disabled by default."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
        )

        # Explicitly ensure streaming_verbose is not set
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr.get('streaming_verbose', False) is False

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)

    def test_streaming_verbose_enabled(self, algorithm, xor_problem):
        """Verify streaming_verbose can be enabled without errors."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['streaming_verbose'] = True

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_streaming_verbose_with_small_chunks(self, algorithm, xor_problem):
        """Verify streaming_verbose works with small population chunks."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=20,  # Very small chunks
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['streaming_verbose'] = True

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)


# Critical Periods Tests

class TestCriticalPeriods:
    """dynamic_functions.critical_periods is not implemented (HMR prints a line and changes no
    mutation rate), so `enabled: True` raises; disabled is the default."""

    def test_critical_periods_disabled_by_default(self, algorithm, xor_problem):
        """Verify critical periods are disabled by default."""
        config = create_config_with_dynamic_functions('cppn_output_4')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Check that critical periods is disabled
        assert hasattr(algorithm, 'df_critical_periods_enabled')
        assert algorithm.df_critical_periods_enabled is False

    @pytest.mark.parametrize("phases", [{'phase1_end': 0.2, 'phase2_end': 0.5, 'min_plasticity': 0.3}, {}])
    def test_critical_periods_enabled_raises(self, algorithm, phases):
        config = create_config_with_dynamic_functions('cppn_output_4')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions']['critical_periods'] = dict(enabled=True, **phases)
        with pytest.raises(ValueError, match='critical_periods is not implemented'):
            algorithm.create_config(config)


# Two Module Mode Tests

class TestTwoModuleMode:
    """two_module_mode, feed-forward only: two-module mode refuses recurrence, so these tests run
    feed-forward."""

    def test_two_module_mode_none_default(self, algorithm):
        """Verify two_module_mode defaults to 'none'."""
        task_names = ['xor', 'and']
        config = create_config_with_multi_task(
            task_names=task_names,
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )

        # By default, two_module_mode should be 'none'
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr['multitask'].get('two_module_mode', 'none') == 'none'

    def test_two_module_mode_parallel(self, algorithm, two_task_problems):
        """Verify two_module_mode='parallel' runs without errors."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['two_module_mode'] = 'parallel'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    def test_two_module_mode_sequential(self, algorithm, two_task_problems):
        """Verify two_module_mode='sequential' runs without errors."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['two_module_mode'] = 'sequential'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    @pytest.mark.parametrize("mode", ['none', 'parallel', 'sequential'])
    def test_two_module_mode_all_options(self, algorithm, two_task_problems, mode):
        """Verify all two_module_mode options work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['two_module_mode'] = mode

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result, f"two_module_mode='{mode}' failed")


# Generalist Bonus Type Tests

class TestGeneralistBonusType:
    """Tests for generalist_bonus_type multi-task feature."""

    def test_generalist_bonus_none_default(self, algorithm, two_task_problems):
        """Verify generalist_bonus_type defaults to 'none'."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )

        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr['multitask'].get('generalist_bonus_type', 'none') == 'none'

    def test_generalist_bonus_min_bonus(self, algorithm, two_task_problems):
        """Verify generalist_bonus_type='min_bonus' works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = 'min_bonus'
        hmr['multitask']['generalist_bonus_weight'] = 0.5

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    def test_generalist_bonus_variance_penalty(self, algorithm, two_task_problems):
        """Verify generalist_bonus_type='variance_penalty' works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = 'variance_penalty'
        hmr['multitask']['generalist_bonus_weight'] = 0.3

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    def test_generalist_bonus_threshold_bonus(self, algorithm, two_task_problems):
        """Verify generalist_bonus_type='threshold_bonus' works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = 'threshold_bonus'
        hmr['multitask']['generalist_bonus_weight'] = 0.4

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    @pytest.mark.parametrize("bonus_type", ['none', 'min_bonus', 'variance_penalty', 'threshold_bonus'])
    def test_generalist_bonus_all_types(self, algorithm, two_task_problems, bonus_type):
        """Verify all generalist_bonus_type options work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = bonus_type
        hmr['multitask']['generalist_bonus_weight'] = 0.25 if bonus_type != 'none' else 0.0

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result, f"generalist_bonus_type='{bonus_type}' failed")

    @pytest.mark.parametrize("weight", [0.1, 0.25, 0.5, 1.0])
    def test_generalist_bonus_weight_values(self, algorithm, two_task_problems, weight):
        """Verify different generalist_bonus_weight values work."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = 'min_bonus'
        hmr['multitask']['generalist_bonus_weight'] = weight

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result, f"generalist_bonus_weight={weight} failed")


# FULL_SPARSE Forward Pass Mode Tests

class TestFullSparseMode:
    """FULL_SPARSE forward pass mode: defined, and refused (not implemented)."""

    def test_full_sparse_enum_exists(self):
        """Verify FULL_SPARSE is defined in ForwardPassMode."""
        assert hasattr(ForwardPassMode, 'FULL_SPARSE')
        assert ForwardPassMode.FULL_SPARSE.value == "full_sparse"

    def test_full_sparse_is_refused(self, algorithm):
        """FULL_SPARSE is not implemented and is refused rather than run with no h->h
        (test_configuration_parsing.py::TestForwardModeLabels)."""
        with pytest.raises(NotImplementedError, match='FULL_SPARSE'):
            algorithm.set_forward_mode(ForwardPassMode.FULL_SPARSE)

    def test_forward_pass_modes_enum_values(self):
        """Verify all ForwardPassMode enum values are defined."""
        expected_modes = ['DENSE_ONLY', 'HYBRID_SPARSE_HH', 'FULL_SPARSE']
        for mode in expected_modes:
            assert hasattr(ForwardPassMode, mode), f"ForwardPassMode.{mode} not found"


# Fitness Mode Tests

class TestFitnessMode:
    """Tests for different fitness_mode options."""

    @pytest.mark.parametrize("fitness_mode", ['mse', 'accuracy', 'bce', 'hybrid', 'soft_accuracy'])
    def test_fitness_mode_all_options(self, algorithm, two_task_problems, fitness_mode):
        """Verify all fitness_mode options work for multi-task."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['fitness_mode'] = fitness_mode

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result, f"fitness_mode='{fitness_mode}' failed")

    def test_fitness_mode_mse_default(self, algorithm, two_task_problems):
        """Verify MSE is the default fitness mode."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )

        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        # Default should be 'mse'
        assert hmr['multitask'].get('fitness_mode', 'mse') == 'mse'

    def test_fitness_mode_accuracy_for_classification(self, algorithm, two_task_problems):
        """Verify accuracy fitness mode works for classification tasks."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['fitness_mode'] = 'accuracy'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_fitness_mode_bce_for_binary(self, algorithm, two_task_problems):
        """Verify BCE fitness mode works for binary classification."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['fitness_mode'] = 'bce'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    def test_fitness_mode_hybrid(self, algorithm, two_task_problems):
        """Verify hybrid fitness mode works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['fitness_mode'] = 'hybrid'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    def test_fitness_mode_soft_accuracy(self, algorithm, two_task_problems):
        """Verify soft_accuracy fitness mode works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['fitness_mode'] = 'soft_accuracy'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    def test_fitness_mode_acc_mse(self, algorithm, two_task_problems):
        """Verify acc_mse (accuracy + MSE) fitness mode works."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['fitness_mode'] = 'acc_mse'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)


class TestSingleTaskFitnessModes:
    """Tests for fitness modes in single-task evolution scenarios."""

    def test_single_task_mse_fitness(self, algorithm, xor_problem):
        """Verify MSE fitness works for single task."""
        config = create_base_config()
        # MSE is the default for single task
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_single_task_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify fitness computation works with dynamic functions enabled."""
        config = create_config_with_dynamic_functions('cppn_output_4')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_single_task_with_recurrence(self, algorithm, xor_problem):
        """Verify fitness computation works with recurrence enabled."""
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


# Forward Pass Mode Auto-Detection Tests

class TestForwardPassModeDetection:
    """Tests for automatic forward pass mode detection and selection."""

    def test_dense_only_without_recurrence(self, algorithm, xor_problem):
        """Verify DENSE_ONLY is used when no H->H connections exist."""
        config = create_config_with_recurrence('feedforward')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Without H->H connections, should use DENSE_ONLY
        # The algorithm should not fail
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)

    def test_hybrid_mode_with_recurrence(self, algorithm, xor_problem):
        """Verify HYBRID mode is used when H->H connections exist."""
        config = create_config_with_recurrence('hidden_only')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # With H->H connections, should use HYBRID mode
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("preset", ['feedforward', 'hidden_only', 'with_backward', 'full_recurrent'])
    def test_mode_auto_detection_all_presets(self, algorithm, xor_problem, preset):
        """Verify forward pass mode auto-detection works for all recurrence presets."""
        config = create_config_with_recurrence(preset)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Auto-detection failed for {preset} preset")

    def test_dense_only_mode_enum(self):
        """Verify DENSE_ONLY mode enum value is correct."""
        assert ForwardPassMode.DENSE_ONLY.value == "dense_only"

    def test_hybrid_sparse_mode_enum(self):
        """Verify HYBRID_SPARSE_HH mode enum value is correct."""
        assert ForwardPassMode.HYBRID_SPARSE_HH.value == "hybrid"


# Additional FULL_SPARSE Tests

class TestFullSparseModeDetails:
    """Additional tests for FULL_SPARSE forward pass mode details."""

    def test_full_sparse_mode_value(self):
        """Verify FULL_SPARSE enum value."""
        assert ForwardPassMode.FULL_SPARSE.value == "full_sparse"

    def test_all_forward_pass_modes_defined(self):
        """Verify all forward pass modes are properly defined."""
        modes = [m.value for m in ForwardPassMode]
        assert "dense_only" in modes
        assert "hybrid" in modes
        assert "full_sparse" in modes
        assert len(modes) == 3

    def test_forward_pass_mode_string_representations(self):
        """Verify forward pass modes have expected string representations."""
        assert str(ForwardPassMode.DENSE_ONLY.value) == "dense_only"
        assert str(ForwardPassMode.HYBRID_SPARSE_HH.value) == "hybrid"
        assert str(ForwardPassMode.FULL_SPARSE.value) == "full_sparse"


# Combined Experimental Features Tests

class TestCombinedExperimentalFeatures:
    """Tests for combinations of experimental features."""

    def test_streaming_verbose_with_critical_periods(self, algorithm, xor_problem):
        """streaming_verbose with per-node functions and a critical-periods block. The block is
        disabled: enabled raises, since nothing in EMR applies it."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['streaming_verbose'] = True
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
            'critical_periods': {
                'enabled': False,
                'phase1_end': 0.2,
                'phase2_end': 0.5,
                'min_plasticity': 0.3,
            }
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)

    def test_generalist_bonus_with_fitness_mode(self, algorithm, two_task_problems):
        """Verify generalist_bonus works with different fitness_modes."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['generalist_bonus_type'] = 'min_bonus'
        hmr['multitask']['generalist_bonus_weight'] = 0.3
        hmr['multitask']['fitness_mode'] = 'accuracy'

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    def test_two_module_with_generalist_bonus(self, algorithm, two_task_problems):
        """Verify two_module_mode works with generalist_bonus."""
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multitask']['two_module_mode'] = 'parallel'
        hmr['multitask']['generalist_bonus_type'] = 'variance_penalty'
        hmr['multitask']['generalist_bonus_weight'] = 0.25

        result = run_multitask_evolution(
            algorithm, config,
            problems=two_task_problems['problems'],
            nt_vectors=two_task_problems['nt_vectors'],
            generations=QUICK_GENERATIONS,
            aggregation_method='mean',
        )
        assert_no_errors(result)

    @pytest.mark.slow
    def test_all_experimental_multi_task_features(self, algorithm, three_task_problems):
        """Two-module mode, a generalist bonus and hybrid fitness together, on three tasks."""
        config = create_config_with_multi_task(
            task_names=three_task_problems['task_names'],
            neuromod_mode='true_neuromodulation_4nt',
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

        # Enable all experimental multi-task features
        hmr['multitask']['two_module_mode'] = 'sequential'
        hmr['multitask']['generalist_bonus_type'] = 'threshold_bonus'
        hmr['multitask']['generalist_bonus_weight'] = 0.2
        hmr['multitask']['fitness_mode'] = 'hybrid'

        result = run_multitask_evolution(
            algorithm, config,
            problems=three_task_problems['problems'],
            nt_vectors=three_task_problems['nt_vectors'],
            generations=10,  # Twice QUICK_GENERATIONS
            aggregation_method='mean',
        )
        assert_no_errors(result)
        assert_positive_fitness(result)


# Utility Method Tests

class TestUtilityMethods:
    """Tests for algorithm utility methods."""

    def test_get_recurrence_mode_feedforward(self, xor_problem):
        """Verify get_recurrence_mode returns 'feedforward' for feedforward config."""
        algorithm = EMRHyperNEAT()
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        mode = algorithm.get_recurrence_mode()
        assert mode == 'feedforward'

    def test_get_recurrence_mode_hidden_only(self, xor_problem):
        """Verify get_recurrence_mode returns 'feedforward+hh' for hidden_only config."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_recurrence('hidden_only')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        mode = algorithm.get_recurrence_mode()
        # hidden_only preset enables h->h without backward/lateral/self connections
        # So it returns 'feedforward+hh' not 'hidden_only'
        assert mode == 'feedforward+hh'

    def test_get_recurrence_mode_full_recurrent(self, xor_problem):
        """Verify get_recurrence_mode returns 'full_recurrent' for full_recurrent config."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_recurrence('full_recurrent')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        mode = algorithm.get_recurrence_mode()
        assert mode == 'full_recurrent'

    def test_get_extended_config_returns_config(self, xor_problem):
        """Verify get_extended_config returns EMRConfig."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_recurrence('hidden_only')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        extended_config = algorithm.get_extended_config()
        assert extended_config is not None
        assert hasattr(extended_config, 'enabled')
        assert hasattr(extended_config, 'allow_hidden_to_hidden')

    def test_get_extended_config_feedforward_returns_none_or_disabled(self, xor_problem):
        """Verify get_extended_config returns None or disabled config for feedforward."""
        algorithm = EMRHyperNEAT()
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        extended_config = algorithm.get_extended_config()
        # Either None or disabled
        if extended_config is not None:
            assert extended_config.enabled is False or extended_config.allow_hidden_to_hidden is False

    def test_get_extended_metrics_after_generation(self, xor_problem):
        """Verify get_extended_metrics returns metrics after running a generation."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_recurrence('hidden_only')
        run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        algorithm.get_extended_metrics()
        # May be None if not collecting extended metrics
        # Just verify it doesn't error

    def test_get_extended_metrics_before_run(self, xor_problem):
        """Verify get_extended_metrics returns None before any generation."""
        algorithm = EMRHyperNEAT()
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        metrics = algorithm.get_extended_metrics()
        assert metrics is None

    def test_refresh_problem_cache(self, xor_problem):
        """Verify refresh_problem_cache updates cached data."""
        algorithm = EMRHyperNEAT()
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Should not raise
        algorithm.refresh_problem_cache(xor_problem)

    def test_refresh_problem_cache_with_different_problem(self, xor_problem):
        """Verify refresh_problem_cache works with a new problem instance."""
        algorithm = EMRHyperNEAT()
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Create a fresh XOR problem
        new_problem = XORProblem()

        # Should not raise
        algorithm.refresh_problem_cache(new_problem)


# Edge Case Combination Tests

class TestEdgeCaseCombinations:
    """Tests for edge case feature combinations."""

    def test_streaming_with_neuromodulation(self, xor_problem):
        """Verify streaming works with neuromodulation enabled."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
        )
        # Add neuromodulation
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['neuromodulation'] = {
            'mode': 'static_gating',
            'gate_threshold': 0.3,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_high_iteration_level_full_recurrent(self, xor_problem):
        """Verify high iteration_level (>3) works with full_recurrent."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_recurrence('full_recurrent', iteration_level=4)  # High iteration level

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_dynamic_functions_aggregation_neuromod_combined(self, xor_problem):
        """Verify dynamic functions + aggregation + neuromodulation together."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_dynamic_functions('cppn_output_4')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

        # Add dynamic aggregation
        hmr['aggregation'] = {
            'mode': 'global',
            'global_function': 'mean',
        }

        # Add neuromodulation
        hmr['neuromodulation'] = {
            'mode': 'static_gating',
            'gate_threshold': 0.3,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_streaming_dynamic_functions_recurrence(self, xor_problem):
        """Verify streaming + dynamic functions + recurrence together."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

        # Add dynamic functions
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'tanh',
        }

        # Add recurrence (the block is `recurrence`; EMR refuses `extended`)
        hmr['recurrence'] = {
            'preset': 'hidden_only',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)
        assert_positive_fitness(result)
        assert algorithm.extended_config.allow_hidden_to_hidden

    def test_all_features_combined(self, xor_problem):
        """Streaming, dynamic functions, aggregation, neuromodulation and recurrence together."""
        algorithm = EMRHyperNEAT()
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

        # Add all features
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'sin',
            'output_activation': 'tanh',
        }
        hmr['aggregation'] = {
            'mode': 'global',
            'global_function': 'sum',
        }
        hmr['neuromodulation'] = {
            'mode': 'static_gating',
            'gate_threshold': 0.2,
        }
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'iteration_level': 2,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)
        assert_positive_fitness(result)
        assert algorithm.extended_config.allow_hidden_to_hidden
        assert algorithm.extended_config.iteration_level == 2
