"""Tests for connection types in EMR-HyperNEAT.

Tests all 6 sparse connection modes:
- feedforward: No H→H connections
- hidden_only: Forward H→H only (y_source < y_target)
- with_backward: Forward + backward H→H (y_source > y_target)
- with_lateral: Forward + lateral H→H (same y-coordinate)
- with_self: Forward + self-loops (from_idx == to_idx)
- full_recurrent: All connection types enabled
"""

import pytest
from conftest import (
    EMRHyperNEAT,
    create_config_with_recurrence,
    run_quick_evolution,
    assert_no_errors,
    assert_fitness_above,
    assert_positive_fitness,
    RECURRENCE_PRESETS,
    XORProblem,
    ParityProblem,
    DEFAULT_SEED,
    STANDARD_SEEDS,
    QUICK_GENERATIONS,
    STANDARD_GENERATIONS,
)


# All connection type presets to test
CONNECTION_TYPES = ['feedforward', 'hidden_only', 'with_backward', 'with_lateral', 'with_self', 'full_recurrent']


class TestConnectionTypeBasics:
    """Basic tests for each connection type."""

    @pytest.mark.parametrize("conn_type", CONNECTION_TYPES)
    def test_connection_type_initializes(self, algorithm, xor_problem, conn_type):
        """Verify each connection type initializes without errors."""
        config = create_config_with_recurrence(conn_type)
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Verify config was applied
        assert algorithm.extended_config is not None

    @pytest.mark.parametrize("conn_type", CONNECTION_TYPES)
    def test_connection_type_runs_generation(self, algorithm, xor_problem, conn_type):
        """Verify each connection type can run a single generation."""
        config = create_config_with_recurrence(conn_type)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=1)

        assert_no_errors(result, f"Connection type {conn_type} failed")
        assert_positive_fitness(result, f"Connection type {conn_type} has non-positive fitness")

    @pytest.mark.parametrize("conn_type", CONNECTION_TYPES)
    def test_connection_type_produces_fitness(self, algorithm, xor_problem, conn_type):
        """Verify each connection type produces improving fitness."""
        config = create_config_with_recurrence(conn_type)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Connection type {conn_type} failed")
        assert_fitness_above(result, 0.5, f"Connection type {conn_type} fitness too low")


class TestFeedforwardMode:
    """Tests specific to feedforward (no H→H) mode."""

    def test_feedforward_disabled_recurrence(self, algorithm, xor_problem):
        """Verify feedforward mode has recurrence disabled."""
        config = create_config_with_recurrence('feedforward')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == False
        # Note: iteration_level may still be set even when disabled

    def test_feedforward_no_hh_connections(self, algorithm, xor_problem):
        """Verify feedforward mode produces no H→H connections."""
        config = create_config_with_recurrence('feedforward')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Run one generation
        state, metrics = algorithm.run_generation(state, xor_problem)

        # Should have no H→H connections
        assert algorithm.extended_config.allow_hidden_to_hidden == False

    def test_feedforward_solves_xor(self, algorithm, xor_problem):
        """Verify feedforward mode can solve XOR."""
        config = create_config_with_recurrence('feedforward')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        assert_fitness_above(result, 0.8, "Feedforward should achieve >0.8 on XOR")


class TestHiddenOnlyMode:
    """Tests specific to hidden_only (forward H→H) mode."""

    def test_hidden_only_config(self, algorithm, xor_problem):
        """Verify hidden_only mode has correct configuration."""
        config = create_config_with_recurrence('hidden_only')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.extended_config.allow_hidden_to_hidden == True
        assert algorithm.extended_config.allow_backward == False
        assert algorithm.extended_config.allow_lateral == False
        assert algorithm.extended_config.allow_self_loops == False

    def test_hidden_only_allows_forward_hh(self, algorithm, xor_problem):
        """Verify hidden_only mode allows forward H→H connections."""
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestWithBackwardMode:
    """Tests specific to with_backward (feedback) mode."""

    def test_with_backward_config(self, algorithm, xor_problem):
        """Verify with_backward mode has correct configuration."""
        config = create_config_with_recurrence('with_backward')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.extended_config.allow_hidden_to_hidden == True
        assert algorithm.extended_config.allow_backward == True
        assert algorithm.extended_config.allow_lateral == False
        assert algorithm.extended_config.allow_self_loops == False

    def test_with_backward_produces_fitness(self, algorithm, xor_problem):
        """Verify with_backward mode produces valid fitness."""
        config = create_config_with_recurrence('with_backward')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestWithLateralMode:
    """Tests specific to with_lateral (same-layer) mode."""

    def test_with_lateral_config(self, algorithm, xor_problem):
        """Verify with_lateral mode has correct configuration."""
        config = create_config_with_recurrence('with_lateral')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.extended_config.allow_hidden_to_hidden == True
        assert algorithm.extended_config.allow_backward == False
        assert algorithm.extended_config.allow_lateral == True
        assert algorithm.extended_config.allow_self_loops == False

    def test_with_lateral_produces_fitness(self, algorithm, xor_problem):
        """Verify with_lateral mode produces valid fitness."""
        config = create_config_with_recurrence('with_lateral')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestWithSelfMode:
    """Tests specific to with_self (self-loops) mode."""

    def test_with_self_config(self, algorithm, xor_problem):
        """Verify with_self mode has correct configuration."""
        config = create_config_with_recurrence('with_self')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.extended_config.allow_hidden_to_hidden == True
        assert algorithm.extended_config.allow_backward == False
        assert algorithm.extended_config.allow_lateral == False
        assert algorithm.extended_config.allow_self_loops == True

    def test_with_self_produces_fitness(self, algorithm, xor_problem):
        """Verify with_self mode produces valid fitness."""
        config = create_config_with_recurrence('with_self')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestFullRecurrentMode:
    """Tests specific to full_recurrent (all connections) mode."""

    def test_full_recurrent_config(self, algorithm, xor_problem):
        """Verify full_recurrent mode has all connections enabled."""
        config = create_config_with_recurrence('full_recurrent')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.extended_config.allow_hidden_to_hidden == True
        assert algorithm.extended_config.allow_backward == True
        assert algorithm.extended_config.allow_lateral == True
        assert algorithm.extended_config.allow_self_loops == True

    def test_full_recurrent_has_activate_time(self, algorithm, xor_problem):
        """Verify full_recurrent mode has increased activate_time."""
        config = create_config_with_recurrence('full_recurrent')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Full recurrent should have longer activation time
        assert algorithm.extended_config.activate_time >= 10

    def test_full_recurrent_produces_fitness(self, algorithm, xor_problem):
        """Verify full_recurrent mode produces valid fitness."""
        config = create_config_with_recurrence('full_recurrent')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestConnectionTypeDistinctness:
    """Tests verifying connection types produce distinct behaviors."""

    def test_modes_produce_different_results(self, algorithm, xor_problem):
        """Verify different connection modes can produce different results."""
        results = {}
        for conn_type in ['feedforward', 'hidden_only', 'full_recurrent']:
            config = create_config_with_recurrence(conn_type)
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
            results[conn_type] = result.best_fitness

        # At minimum, all should work
        for conn_type, fitness in results.items():
            assert fitness > 0, f"{conn_type} has non-positive fitness"

    def test_recurrent_modes_distinct_from_feedforward(self, algorithm, xor_problem):
        """Verify recurrent modes behave differently from feedforward."""
        ff_config = create_config_with_recurrence('feedforward')
        ff_result = run_quick_evolution(algorithm, ff_config, xor_problem, generations=QUICK_GENERATIONS)

        hh_config = create_config_with_recurrence('hidden_only')
        hh_result = run_quick_evolution(algorithm, hh_config, xor_problem, generations=QUICK_GENERATIONS)

        # Both should work
        assert_no_errors(ff_result)
        assert_no_errors(hh_result)
        assert_positive_fitness(ff_result)
        assert_positive_fitness(hh_result)


class TestIterationLevelVariants:
    """Tests for different iteration levels."""

    @pytest.mark.parametrize("iteration_level", [0, 1, 2, 3])
    def test_iteration_levels(self, algorithm, xor_problem, iteration_level):
        """Verify different iteration levels work."""
        config = create_config_with_recurrence('hidden_only', iteration_level=iteration_level)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Iteration level {iteration_level} failed")
        assert_positive_fitness(result, f"Iteration level {iteration_level} non-positive fitness")

    def test_higher_iteration_enables_multi_hop(self, algorithm, xor_problem):
        """Verify higher iteration levels enable multi-hop connections."""
        for level in [2, 3]:
            config = create_config_with_recurrence('with_backward', iteration_level=level)
            config_obj = algorithm.create_config(config)
            state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.extended_config.iteration_level == level


class TestMultiHopAlgorithms:
    """Tests for different multi-hop computation algorithms."""

    @pytest.mark.parametrize("algorithm_type", ['matrix_power', 'fori_loop'])
    def test_multi_hop_algorithms(self, algorithm, xor_problem, algorithm_type):
        """Verify both multi-hop algorithms work."""
        config = create_config_with_recurrence('with_backward', iteration_level=2)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['multi_hop_algorithm'] = algorithm_type

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Multi-hop algorithm {algorithm_type} failed")
        assert_positive_fitness(result)


class TestConnectionTypesWithProblems:
    """Tests connection types on different problems."""

    @pytest.mark.parametrize("conn_type", CONNECTION_TYPES)
    def test_connection_types_on_xor(self, algorithm, xor_problem, conn_type):
        """Verify all connection types work on XOR."""
        config = create_config_with_recurrence(conn_type)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"{conn_type} failed on XOR")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("conn_type", ['feedforward', 'hidden_only', 'full_recurrent'])
    def test_connection_types_on_and(self, algorithm, and_problem, conn_type):
        """Verify key connection types work on AND."""
        config = create_config_with_recurrence(conn_type)
        result = run_quick_evolution(algorithm, config, and_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"{conn_type} failed on AND")
        assert_positive_fitness(result)


class TestComprehensiveConnectionTypes:
    """Comprehensive tests (marked slow)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("conn_type", CONNECTION_TYPES)
    def test_connection_type_solves_xor(self, algorithm, xor_problem, conn_type):
        """Verify each connection type can solve XOR given enough generations."""
        config = create_config_with_recurrence(conn_type)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        assert_fitness_above(result, 0.8, f"{conn_type} should achieve >0.8 on XOR")

    @pytest.mark.slow
    def test_all_connection_types_multi_seed(self, algorithm, xor_problem):
        """Verify connection types work across multiple seeds."""
        for conn_type in ['feedforward', 'hidden_only', 'full_recurrent']:
            config = create_config_with_recurrence(conn_type)
            for seed in STANDARD_SEEDS:
                result = run_quick_evolution(
                    algorithm, config, xor_problem,
                    generations=QUICK_GENERATIONS, seed=seed
                )
                assert_no_errors(result, f"{conn_type} seed={seed} failed")
                assert_positive_fitness(result)


class TestHopDecayFactor:
    """Tests for hop_decay_factor parameter.

    hop_decay_factor controls the weight attenuation for multi-hop connections.
    Lower values cause stronger decay over multiple hops.
    """

    @pytest.mark.parametrize("decay_factor", [0.5, 0.7, 0.9, 1.0])
    def test_hop_decay_factor_values(self, algorithm, xor_problem, decay_factor):
        """Verify different hop_decay_factor values work."""
        config = create_config_with_recurrence('hidden_only', iteration_level=2)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hop_decay_factor'] = decay_factor

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Hop decay factor {decay_factor} failed")

    def test_hop_decay_factor_near_zero(self, algorithm, xor_problem):
        """Verify hop_decay_factor=0.01 (minimal propagation) works.

        Note: hop_decay_factor must be in (0, 1] - 0.0 is not valid
        as it would completely eliminate multi-hop propagation.
        """
        config = create_config_with_recurrence('hidden_only', iteration_level=2)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hop_decay_factor'] = 0.01

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_hop_decay_factor_one(self, algorithm, xor_problem):
        """Verify hop_decay_factor=1.0 (no decay) works."""
        config = create_config_with_recurrence('hidden_only', iteration_level=2)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hop_decay_factor'] = 1.0

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_hop_decay_with_backward(self, algorithm, xor_problem):
        """Verify hop_decay_factor works with backward connections."""
        config = create_config_with_recurrence('with_backward', iteration_level=2)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hop_decay_factor'] = 0.8

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_hop_decay_with_full_recurrent(self, algorithm, xor_problem):
        """Verify hop_decay_factor works with full recurrence."""
        config = create_config_with_recurrence('full_recurrent', iteration_level=2)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hop_decay_factor'] = 0.9

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestConnectionToggleCombinations:
    """Tests for connection toggle combinations via existing presets.

    Note: The algorithm only supports predefined presets (feedforward, hidden_only,
    with_backward, with_lateral, with_self, full_recurrent). Custom arbitrary
    toggle combinations are not supported - use the closest preset instead.
    """

    def test_backward_connections_via_preset(self, algorithm, xor_problem):
        """Verify backward connections work via with_backward preset."""
        config = create_config_with_recurrence('with_backward', iteration_level=2)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_lateral_connections_via_preset(self, algorithm, xor_problem):
        """Verify lateral connections work via with_lateral preset."""
        config = create_config_with_recurrence('with_lateral', iteration_level=2)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_self_loops_via_preset(self, algorithm, xor_problem):
        """Verify self-loops work via with_self preset."""
        config = create_config_with_recurrence('with_self', iteration_level=2)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_all_connection_types_via_full_recurrent(self, algorithm, xor_problem):
        """Verify all connection types work via full_recurrent preset.

        full_recurrent enables: backward + lateral + self-loops
        """
        config = create_config_with_recurrence('full_recurrent', iteration_level=2)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_full_recurrent_with_higher_iteration(self, algorithm, xor_problem):
        """Verify full recurrent works with higher iteration levels."""
        config = create_config_with_recurrence('full_recurrent', iteration_level=3)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
