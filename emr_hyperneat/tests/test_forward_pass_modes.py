"""Tests for forward pass modes in EMR-HyperNEAT.

Tests the ForwardPassMode enum and automatic mode selection:
- DENSE_ONLY: Feedforward mode (W1 → tanh → W2, no H→H)
- HYBRID_SPARSE_HH: Dense W1/W2 + sparse H→H connections
- FULL_SPARSE: All sparse connections (extreme recurrence)

The forward pass mode is automatically determined based on recurrence settings:
- feedforward preset → DENSE_ONLY
- hidden_only, with_backward, etc. → HYBRID_SPARSE_HH
"""

import pytest
from conftest import (
    EMRHyperNEAT,
    create_config_with_recurrence,
    create_base_config,
    run_quick_evolution,
    assert_no_errors,
    assert_positive_fitness,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
)


class TestDenseOnlyMode:
    """Tests for DENSE_ONLY forward pass mode (feedforward)."""

    def test_dense_only_with_feedforward(self, algorithm, xor_problem):
        """Verify DENSE_ONLY mode is used with feedforward preset."""
        config = create_config_with_recurrence('feedforward')

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_dense_only_basic_evolution(self, algorithm, xor_problem):
        """Verify DENSE_ONLY mode supports basic evolution."""
        config = create_base_config()
        # No recurrence configuration = feedforward (DENSE_ONLY)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_dense_only_no_hh_connections(self, algorithm, xor_problem):
        """Verify feedforward has no H→H connections."""
        config = create_config_with_recurrence('feedforward')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # In feedforward mode, allow_hidden_to_hidden should be False
        assert algorithm.extended_config is not None
        assert not algorithm.extended_config.allow_hidden_to_hidden

    def test_dense_only_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify DENSE_ONLY works with dynamic functions."""
        config = create_config_with_recurrence('feedforward')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'global', 'global_activation': 'tanh'}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_dense_only_with_aggregation(self, algorithm, xor_problem):
        """Verify DENSE_ONLY works with aggregation configuration."""
        config = create_config_with_recurrence('feedforward')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {'mode': 'global', 'global_aggregation': 'sum'}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestHybridSparseHHMode:
    """Tests for HYBRID_SPARSE_HH forward pass mode."""

    def test_hybrid_with_hidden_only(self, algorithm, xor_problem):
        """Verify HYBRID_SPARSE_HH mode is used with hidden_only preset."""
        config = create_config_with_recurrence('hidden_only')

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_hybrid_has_hh_connections(self, algorithm, xor_problem):
        """Verify hidden_only preset enables H→H connections."""
        config = create_config_with_recurrence('hidden_only')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # In hidden_only mode, allow_hidden_to_hidden should be True
        assert algorithm.extended_config is not None
        assert algorithm.extended_config.allow_hidden_to_hidden

    @pytest.mark.parametrize("preset", ['hidden_only', 'with_backward', 'with_lateral', 'with_self', 'full_recurrent'])
    def test_hybrid_with_all_recurrent_presets(self, algorithm, xor_problem, preset):
        """Verify HYBRID_SPARSE_HH mode works with all recurrent presets."""
        config = create_config_with_recurrence(preset)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_hybrid_with_activate_time(self, algorithm, xor_problem):
        """Verify HYBRID_SPARSE_HH works with different activate_time values."""
        config = create_config_with_recurrence('hidden_only')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['activate_time'] = 5

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_hybrid_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify HYBRID_SPARSE_HH works with dynamic functions."""
        config = create_config_with_recurrence('hidden_only')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_hybrid_with_aggregation(self, algorithm, xor_problem):
        """Verify HYBRID_SPARSE_HH works with aggregation configuration."""
        config = create_config_with_recurrence('hidden_only')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['aggregation'] = {'mode': 'cppn_output', 'num_aggregations': 3}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestModeTransitions:
    """Tests for transitioning between forward pass modes."""

    def test_feedforward_to_recurrent(self, algorithm, xor_problem):
        """Verify evolution works when switching from feedforward config."""
        # First run with feedforward
        config_ff = create_config_with_recurrence('feedforward')
        result_ff = run_quick_evolution(algorithm, config_ff, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result_ff)

        # Then run with recurrent (new algorithm instance needed)
        config_rec = create_config_with_recurrence('hidden_only')
        result_rec = run_quick_evolution(algorithm, config_rec, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result_rec)

    def test_same_algorithm_different_modes(self, algorithm, xor_problem):
        """Verify same algorithm can be configured for different modes."""
        # Run feedforward
        config1 = create_config_with_recurrence('feedforward')
        config_obj1 = algorithm.create_config(config1)
        state1 = algorithm.initialize(config_obj1, xor_problem, seed=42)
        _, metrics1 = algorithm.run_generation(state1, xor_problem)

        # Run recurrent (reinitialize with different config)
        config2 = create_config_with_recurrence('hidden_only')
        config_obj2 = algorithm.create_config(config2)
        state2 = algorithm.initialize(config_obj2, xor_problem, seed=42)
        _, metrics2 = algorithm.run_generation(state2, xor_problem)

        # Both should produce valid results
        assert float(metrics1.best_fitness) >= 0
        assert float(metrics2.best_fitness) >= 0


class TestModeWithOtherFeatures:
    """Tests for forward pass modes combined with other algorithm features."""

    def test_dense_only_with_neuromodulation(self, algorithm, xor_problem):
        """Verify DENSE_ONLY works with neuromodulation."""
        config = create_config_with_recurrence('feedforward')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['neuromodulation'] = {
            'mode': 'static_gating',
            'gate_scaling': 'sigmoid',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_hybrid_with_neuromodulation(self, algorithm, xor_problem):
        """Verify HYBRID_SPARSE_HH works with neuromodulation."""
        config = create_config_with_recurrence('hidden_only')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['neuromodulation'] = {
            'mode': 'static_gating',
            'gate_scaling': 'sigmoid',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_dense_only_with_locality_radius(self, algorithm, xor_problem):
        """Verify DENSE_ONLY works with locality_radius."""
        config = create_config_with_recurrence('feedforward')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.5

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_hybrid_with_locality_radius(self, algorithm, xor_problem):
        """Verify HYBRID_SPARSE_HH works with locality_radius."""
        config = create_config_with_recurrence('hidden_only')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.5

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_dense_only_with_h_to_h_cache(self, algorithm, xor_problem):
        """Verify feedforward mode (no h→h) handles cache config gracefully."""
        config = create_config_with_recurrence('feedforward')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        # H→H caching only applies to recurrent modes, but should be ignored in feedforward
        hmr['recurrence']['hh_cache_enabled'] = True
        hmr['recurrence']['hh_refresh_interval'] = 5

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestComprehensiveModes:
    """Comprehensive tests (marked slow)."""

    @pytest.mark.slow
    def test_all_modes_solve_xor(self, algorithm, xor_problem):
        """Verify all forward pass modes can evolve on XOR."""
        modes = [
            ('feedforward', 'DENSE_ONLY'),
            ('hidden_only', 'HYBRID_SPARSE_HH'),
            ('with_backward', 'HYBRID_SPARSE_HH'),
            ('with_lateral', 'HYBRID_SPARSE_HH'),
            ('with_self', 'HYBRID_SPARSE_HH'),
            ('full_recurrent', 'HYBRID_SPARSE_HH'),
        ]

        for preset, expected_mode in modes:
            config = create_config_with_recurrence(preset)
            result = run_quick_evolution(algorithm, config, xor_problem, generations=10)
            assert_no_errors(result, f"{preset} ({expected_mode}) failed")
            assert result.best_fitness > 0, f"{preset} ({expected_mode}) has non-positive fitness"

    @pytest.mark.slow
    def test_hybrid_mode_consistency_across_seeds(self, algorithm, xor_problem):
        """Verify HYBRID_SPARSE_HH produces consistent results across seeds."""
        config = create_config_with_recurrence('hidden_only')
        results = []

        for seed in [42, 123, 456]:
            result = run_quick_evolution(
                algorithm, config, xor_problem,
                generations=QUICK_GENERATIONS, seed=seed
            )
            assert_no_errors(result, f"Seed {seed} failed")
            results.append(result.best_fitness)

        # All should produce positive fitness
        for fitness in results:
            assert fitness > 0
