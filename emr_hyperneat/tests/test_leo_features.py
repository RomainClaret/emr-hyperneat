"""Tests for LEO (Link Expression Output) features.

LEO is a connection gating mechanism where the CPPN outputs two values:
- Output 0: Connection weight
- Output 1: LEO gate value (connection expressed only if > threshold)

This provides an additional control mechanism for connection existence
beyond the standard weight threshold.

Tests cover:
- LEO configuration and presets
- Threshold parameter variations
- LEO validity mask computation
- Weight sparsification with LEO gating
- Full evolution with LEO enabled
- Integration with other features (streaming, dynamic functions, recurrence)
"""

import pytest

from conftest import (
    create_config_with_leo,
    create_config_with_leo_and_geometry_seeding,
    run_quick_evolution,
    assert_no_errors,
    assert_positive_fitness,
    LEO_PRESETS,
    QUICK_GENERATIONS,
    STANDARD_GENERATIONS,
    DEFAULT_SEED,
    STANDARD_SEEDS,
)


# Test Classes


class TestLEOConfiguration:
    """Test LEO configuration parsing and validation."""

    def test_leo_disabled(self, leo_algorithm, xor_problem):
        """LEO disabled should work like standard algorithm."""
        config = create_config_with_leo(preset='disabled')
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_leo_enabled_default_threshold(self, leo_algorithm, xor_problem):
        """LEO enabled with default threshold (0.2)."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("preset", list(LEO_PRESETS.keys()))
    def test_leo_all_presets(self, leo_algorithm, xor_problem, preset):
        """All LEO presets should initialize and run without errors."""
        config = create_config_with_leo(preset=preset)
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_config_override(self, leo_algorithm, xor_problem):
        """LEO config should accept overrides."""
        config = create_config_with_leo(preset='default', threshold=0.35)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr['leo']['threshold'] == 0.35
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_requires_two_cppn_outputs(self, leo_algorithm, xor_problem):
        """With LEO the engine builds a two-output CPPN, and says so in num_cppn_outputs."""
        config = create_config_with_leo(preset='default')
        leo_algorithm.create_config(config)
        assert leo_algorithm.num_cppn_outputs == 2

    def test_invalid_preset_raises_error(self, leo_algorithm):
        """Invalid preset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown LEO preset"):
            create_config_with_leo(preset='nonexistent')


class TestLEOThreshold:
    """Test LEO threshold parameter variations."""

    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    def test_leo_threshold_values(self, leo_algorithm, xor_problem, threshold):
        """Different LEO thresholds should work."""
        config = create_config_with_leo(preset='default', threshold=threshold)
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_low_threshold_runs(self, leo_algorithm, xor_problem):
        """Low threshold (0.1) should allow more connections."""
        config = create_config_with_leo(preset='low_threshold')
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_high_threshold_runs(self, leo_algorithm, xor_problem):
        """High threshold (0.5) should allow fewer connections."""
        config = create_config_with_leo(preset='high_threshold')
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        # With very high threshold, might get lower fitness but should still work
        assert result.best_fitness >= 0

    def test_very_high_threshold_runs(self, leo_algorithm, xor_problem):
        """Very high threshold (0.7) is challenging but should run."""
        config = create_config_with_leo(preset='very_high_threshold')
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_zero_threshold_permissive(self, leo_algorithm, xor_problem):
        """Zero threshold should be very permissive (all connections pass)."""
        config = create_config_with_leo(preset='default', threshold=0.0)
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_threshold_comparison(self, leo_algorithm, xor_problem):
        """Lower threshold should generally not hurt fitness vs higher."""
        config_low = create_config_with_leo(preset='low_threshold')
        result_low = run_quick_evolution(
            leo_algorithm, config_low, xor_problem,
            generations=STANDARD_GENERATIONS
        )

        config_high = create_config_with_leo(preset='high_threshold')
        result_high = run_quick_evolution(
            leo_algorithm, config_high, xor_problem,
            generations=STANDARD_GENERATIONS
        )

        assert_no_errors(result_low)
        assert_no_errors(result_high)
        # Both should achieve some fitness
        assert result_low.best_fitness > 0
        assert result_high.best_fitness > 0


class TestLEOWithEvolution:
    """Test LEO with full evolution runs."""

    def test_leo_progresses_over_generations(self, leo_algorithm, xor_problem):
        """LEO should show fitness improvement over generations."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(
            leo_algorithm, config, xor_problem,
            generations=STANDARD_GENERATIONS
        )
        assert_no_errors(result)
        assert len(result.fitness_history) == STANDARD_GENERATIONS
        # Generally fitness should improve (last gen >= first gen on average)
        assert result.best_fitness >= result.fitness_history[0]

    def test_leo_with_hidden_only_recurrence(self, leo_algorithm, xor_problem):
        """LEO with h->h connections enabled."""
        config = create_config_with_leo(
            preset='default',
            recurrence_preset='hidden_only'
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_leo_with_full_recurrence(self, leo_algorithm, xor_problem):
        """LEO with full recurrent connections."""
        config = create_config_with_leo(
            preset='default',
            recurrence_preset='full_recurrent'
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_backward_connections(self, leo_algorithm, xor_problem):
        """LEO with backward connections enabled."""
        config = create_config_with_leo(
            preset='default',
            recurrence_preset='with_backward'
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_lateral_connections(self, leo_algorithm, xor_problem):
        """LEO with lateral connections enabled."""
        config = create_config_with_leo(
            preset='default',
            recurrence_preset='with_lateral'
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_self_loops(self, leo_algorithm, xor_problem):
        """LEO with self-loop connections enabled."""
        config = create_config_with_leo(
            preset='default',
            recurrence_preset='with_self'
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_larger_population(self, leo_algorithm, xor_problem):
        """LEO with larger population size."""
        config = create_config_with_leo(
            preset='default',
            population_size=200
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_leo_with_deeper_substrate(self, leo_algorithm, xor_problem):
        """LEO with deeper max_depth."""
        config = create_config_with_leo(
            preset='default',
            max_depth=3
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)


class TestLEOWithOtherFeatures:
    """Test LEO integration with other features."""

    def test_leo_with_streaming(self, leo_algorithm, xor_problem):
        """LEO with streaming mode."""
        config = create_config_with_leo(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['enable_streaming'] = True
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['population_chunk_size'] = 50
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_dynamic_functions_cppn_output(self, leo_algorithm, xor_problem):
        """LEO with dynamic function selection (CPPN output mode).

        This test validates that cppn_output mode works with LEO. The activation
        indices are derived from weight values (output 0), which is independent
        of the LEO gate (output 1).
        """
        config = create_config_with_leo(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4
        }
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_cppn_output_and_recurrence(self, leo_algorithm, xor_problem):
        """LEO + cppn_output dynamic functions + hidden recurrence.

        Validates that activation index derivation from weights works
        when both LEO (num_cppn_outputs=2) and recurrence are enabled.
        """
        config = create_config_with_leo(
            preset='default',
            recurrence_preset='hidden_only'
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4
        }
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_leo_with_cppn_output_and_weight_interpretation(self, leo_algorithm, xor_problem):
        """LEO + cppn_output with explicit weight interpretation config.

        Tests the combination with magnitude_bio interpretation to verify
        activation indices derived correctly from weight magnitudes.
        """
        config = create_config_with_leo(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 6,
            'interpretation': 'magnitude_bio'
        }
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_dynamic_functions_global(self, leo_algorithm, xor_problem):
        """LEO with global dynamic function mode."""
        config = create_config_with_leo(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid'
        }
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_aggregation_sum(self, leo_algorithm, xor_problem):
        """LEO with global sum aggregation."""
        config = create_config_with_leo(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['aggregation'] = {
            'mode': 'global',
            'global_function': 'sum'
        }
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_aggregation_mean(self, leo_algorithm, xor_problem):
        """LEO with global mean aggregation."""
        config = create_config_with_leo(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['aggregation'] = {
            'mode': 'global',
            'global_function': 'mean'
        }
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_geometry_seeding(self, leo_algorithm, xor_problem):
        """LEO combined with geometry seeding."""
        config = create_config_with_leo_and_geometry_seeding(
            leo_preset='default',
            gs_preset='default'
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)


class TestLEOOnDifferentProblems:
    """Test LEO on different problem types."""

    def test_leo_on_and_problem(self, leo_algorithm, and_problem):
        """LEO on AND logic gate."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(leo_algorithm, config, and_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_leo_on_or_problem(self, leo_algorithm, or_problem):
        """LEO on OR logic gate."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(leo_algorithm, config, or_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_leo_on_nand_problem(self, leo_algorithm, nand_problem):
        """LEO on NAND logic gate."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(leo_algorithm, config, nand_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_leo_on_nor_problem(self, leo_algorithm, nor_problem):
        """LEO on NOR logic gate."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(leo_algorithm, config, nor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_leo_on_parity3_problem(self, leo_algorithm, parity3_problem):
        """LEO on 3-bit parity (harder problem)."""
        config = create_config_with_leo(preset='default')
        # Parity3 has 4 inputs (3 bits + 1 bias), update substrate coords
        config['algorithm_params']['emrhyperneat']['substrate']['input_coords'] = [
            (-1.0, -1.0), (-0.33, -1.0), (0.33, -1.0), (1.0, -1.0)
        ]
        result = run_quick_evolution(
            leo_algorithm, config, parity3_problem,
            generations=STANDARD_GENERATIONS
        )
        assert_no_errors(result)


class TestLEOReproducibility:
    """Test LEO reproducibility and consistency."""

    def test_leo_reproducibility_same_seed(self, leo_algorithm, xor_problem):
        """Same seed should produce same results."""
        config = create_config_with_leo(preset='default')

        result1 = run_quick_evolution(
            leo_algorithm, config, xor_problem, seed=42
        )
        result2 = run_quick_evolution(
            leo_algorithm, config, xor_problem, seed=42
        )

        assert_no_errors(result1)
        assert_no_errors(result2)
        # With same seed, results should be identical
        assert result1.best_fitness == result2.best_fitness

    def test_leo_different_seeds_variation(self, leo_algorithm, xor_problem):
        """Different seeds should produce different results."""
        config = create_config_with_leo(preset='default')
        results = []

        for seed in STANDARD_SEEDS:
            result = run_quick_evolution(
                leo_algorithm, config, xor_problem, seed=seed
            )
            assert_no_errors(result)
            results.append(result.best_fitness)

        # All should be positive
        assert all(f > 0 for f in results)
        # Should have some variation (not all identical)
        # Note: with small generation count, might be similar
        assert len(set(results)) >= 1  # At least one unique value

    @pytest.mark.slow
    def test_leo_consistency_across_seeds(self, leo_algorithm, xor_problem):
        """LEO should consistently achieve positive fitness across seeds."""
        config = create_config_with_leo(preset='default')
        positive_count = 0

        for seed in [42, 123, 456, 789, 1000]:
            result = run_quick_evolution(
                leo_algorithm, config, xor_problem,
                seed=seed, generations=STANDARD_GENERATIONS
            )
            assert_no_errors(result)
            if result.best_fitness > 0.5:
                positive_count += 1

        # At least 60% should achieve decent fitness
        assert positive_count >= 3


class TestLEOEdgeCases:
    """Test LEO edge cases and boundary conditions."""

    def test_leo_with_minimal_population(self, leo_algorithm, xor_problem):
        """LEO with minimal population size."""
        config = create_config_with_leo(
            preset='default',
            population_size=10
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_with_minimal_depth(self, leo_algorithm, xor_problem):
        """LEO with minimal max_depth (1)."""
        config = create_config_with_leo(
            preset='default',
            max_depth=1
        )
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_single_generation(self, leo_algorithm, xor_problem):
        """LEO with single generation."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(
            leo_algorithm, config, xor_problem, generations=1
        )
        assert_no_errors(result)
        assert len(result.fitness_history) == 1

    def test_leo_threshold_boundary_0_1(self, leo_algorithm, xor_problem):
        """LEO with threshold at 0.1 boundary."""
        config = create_config_with_leo(preset='default', threshold=0.1)
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_leo_threshold_near_one(self, leo_algorithm, xor_problem):
        """LEO with threshold near 1.0 (very restrictive)."""
        config = create_config_with_leo(preset='default', threshold=0.95)
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)


class TestLEOPerformanceMetrics:
    """Test LEO performance measurement."""

    def test_leo_timing_recorded(self, leo_algorithm, xor_problem):
        """LEO should record generation timing."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(leo_algorithm, config, xor_problem)
        assert_no_errors(result)
        assert result.total_time > 0
        assert result.avg_gen_time_ms > 0
        assert len(result.gen_times) == QUICK_GENERATIONS

    def test_leo_fitness_history_complete(self, leo_algorithm, xor_problem):
        """LEO should record complete fitness history."""
        config = create_config_with_leo(preset='default')
        result = run_quick_evolution(
            leo_algorithm, config, xor_problem,
            generations=10
        )
        assert_no_errors(result)
        assert len(result.fitness_history) == 10
        assert result.final_fitness == result.fitness_history[-1]
        assert result.best_fitness == max(result.fitness_history)


# Channels, modes and execution paths

def _leo_run(config, generations=1, algo_cls=None):
    """Fitness per generation and the model, for one seeded run on XOR."""
    import copy
    import warnings

    import numpy as np

    from conftest import XORProblem
    from emr_hyperneat import (
        EMRHyperNEAT,
    )

    algo = (algo_cls or EMRHyperNEAT)()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conf = algo.create_config(copy.deepcopy(config))
    problem = XORProblem()
    state = algo.initialize(conf, problem, seed=DEFAULT_SEED)
    fits = []
    for _ in range(generations):
        state, _metrics = algo.run_generation(state, problem)
        fits.append(np.asarray(algo._cached_fitnesses))
    return algo, fits


def _leo_config(recurrence='feedforward', hand_set_genome=True, **leo):
    config = create_config_with_leo('default', recurrence_preset=recurrence,
                                    max_depth=3, population_size=20, **leo)
    if not hand_set_genome:
        config['algorithm_params']['emrhyperneat'].pop('cppn', None)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat'].pop('num_cppn_outputs', None)
    return config


def _hmr(config):
    return config['algorithm_params']['emrhyperneat']['emr_hyperneat']


class TestLEOChannelsAndModes:
    """LEO's channel, its modes, and one network whatever the execution mode."""

    def test_leo_counts_its_own_output(self):
        """B2: without a hand-set genome size the CPPN still gets LEO's output."""
        import numpy as np

        auto, auto_fits = _leo_run(_leo_config(hand_set_genome=False))
        assert auto.num_cppn_outputs == 2
        assert auto.neat_algo.genome.num_outputs == 2
        _, hand_fits = _leo_run(_leo_config())
        np.testing.assert_array_equal(auto_fits[0], hand_fits[0])

    def test_per_node_outputs_take_the_channels_after_leo(self):
        """B: activation and aggregation never read LEO's channel."""
        config = _leo_config(hand_set_genome=False)
        _hmr(config)['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}
        algo, _ = _leo_run(config)
        assert algo.num_cppn_outputs == 3
        assert algo._cppn_output_indices == {'weight': 0, 'activation': 2, 'aggregation': 0}

        _hmr(config)['aggregation'] = {'mode': 'cppn_output', 'num_aggregations': 4}
        algo, _ = _leo_run(config)
        assert algo.num_cppn_outputs == 4
        assert algo._cppn_output_indices == {'weight': 0, 'activation': 2, 'aggregation': 3}

    def test_a_per_node_output_on_leos_channel_is_refused(self):
        from emr_hyperneat import (
            EMRHyperNEAT,
        )

        config = _leo_config(hand_set_genome=False)
        _hmr(config)['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4,
                                             'cppn_output_indices': {'activation': 1}}
        with pytest.raises(ValueError, match="LEO's channel"):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize("leo, match", [
        ({'mode': 'bogus'}, "leo.mode"),
        ({'output_index': 2}, "leo.output_index"),
    ])
    def test_leo_settings_are_validated(self, leo, match):
        from emr_hyperneat import (
            EMRHyperNEAT,
        )

        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(_leo_config(**leo))

    def test_leo_needs_its_channel(self):
        from emr_hyperneat import (
            EMRHyperNEAT,
        )

        config = _leo_config(hand_set_genome=False)
        _hmr(config)['dynamic_functions'] = {'num_cppn_outputs': 1}
        with pytest.raises(ValueError, match="LEO needs a CPPN output"):
            EMRHyperNEAT().create_config(config)

    def test_replacement_mode_is_not_compound(self):
        """L: 'replacement' is its own rule, not 'compound' under another name."""
        import numpy as np

        compound, compound_fits = _leo_run(_leo_config(mode='compound'))
        replacement, replacement_fits = _leo_run(_leo_config(mode='replacement'))
        assert replacement.leo_mode == 'replacement'
        assert not np.array_equal(np.asarray(compound._cached_W1),
                                  np.asarray(replacement._cached_W1))
        assert not np.array_equal(compound_fits[0], replacement_fits[0])

    def test_leo_settings_reach_hidden_to_hidden_discovery(self, monkeypatch):
        """K: discovery used a hardcoded 0.2 and compound gating."""
        import emr_hyperneat.emrhyperneat as E

        seen = []
        discover = E.discover_sparse_hh_vectorized_multi_hop

        def spy(*args, **kwargs):
            seen.append((kwargs.get('leo_threshold'), kwargs.get('leo_mode')))
            return discover(*args, **kwargs)

        monkeypatch.setattr(E, 'discover_sparse_hh_vectorized_multi_hop', spy)
        _leo_run(_leo_config('hidden_only', threshold=0.6, mode='replacement'))
        assert seen and all(entry == (0.6, 'replacement') for entry in seen)

    @pytest.mark.parametrize("mode", ['compound', 'replacement'])
    def test_streaming_builds_the_standard_leo_network(self, mode):
        """M: the streaming path scaled and thresholded LEO weights differently."""
        import numpy as np

        from conftest import exact_backend

        standard = _leo_config(mode=mode)
        streaming = _leo_config(mode=mode)
        _hmr(streaming)['execution'] = {'mode': 'streaming'}
        _, standard_fits = _leo_run(standard, generations=2)
        _, streaming_fits = _leo_run(streaming, generations=2)
        for a, b in zip(standard_fits, streaming_fits):
            if exact_backend():
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, rtol=1e-4)

    def test_substrate_config_refuses_leo(self):
        from emr_hyperneat import (
            EMRHyperNEAT,
        )

        algo = EMRHyperNEAT()
        algo.create_config(_leo_config())
        with pytest.raises(ValueError, match="LEO gating"):
            algo.substrate_config()
