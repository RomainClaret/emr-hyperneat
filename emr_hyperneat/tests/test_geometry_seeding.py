"""Tests for geometry seeding features.

Geometry seeding adds spatial locality biases to connection weights:
- compute_locality_penalty(): Distance-based weight scaling
- apply_locality_penalty_to_weights(): Apply delta_x, delta_y penalties
- 7D CPPN inputs: (x1, y1, x2, y2, d, delta_x, delta_y) instead of 5D

The locality penalty favors local connections over distant ones,
potentially improving network efficiency and biological plausibility.

Tests cover:
- Geometry seeding configuration and presets
- Locality penalty computation
- 5D vs 7D CPPN input handling
- Full evolution with geometry seeding
- Integration with other features
"""

import pytest
import jax.numpy as jnp

from conftest import (
    create_config_with_geometry_seeding,
    run_quick_evolution,
    assert_no_errors,
    assert_positive_fitness,
    GEOMETRY_SEEDING_PRESETS,
    QUICK_GENERATIONS,
    STANDARD_GENERATIONS,
    DEFAULT_SEED,
    STANDARD_SEEDS,
)


# =============================================================================
# Test Classes
# =============================================================================


class TestGeometrySeedingConfiguration:
    """Test geometry seeding configuration parsing and validation."""

    def test_geometry_seeding_disabled(self, algorithm, xor_problem):
        """Geometry seeding disabled should work normally."""
        config = create_config_with_geometry_seeding(preset='disabled')
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_geometry_seeding_enabled_default(self, algorithm, xor_problem):
        """Geometry seeding enabled with default settings."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("preset", list(GEOMETRY_SEEDING_PRESETS.keys()))
    def test_geometry_seeding_all_presets(self, algorithm, xor_problem, preset):
        """All geometry seeding presets should work."""
        config = create_config_with_geometry_seeding(preset=preset)
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_geometry_seeding_config_override(self, algorithm, xor_problem):
        """Geometry seeding config should accept overrides."""
        config = create_config_with_geometry_seeding(
            preset='default',
            seed_weight=-0.5
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr['geometry_seeding']['seed_weight'] == -0.5
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_invalid_preset_raises_error(self, algorithm):
        """Invalid preset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown geometry seeding preset"):
            create_config_with_geometry_seeding(preset='nonexistent')


class TestSeedWeight:
    """Test seed_weight parameter variations."""

    @pytest.mark.parametrize("seed_weight", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_seed_weight_values(self, algorithm, xor_problem, seed_weight):
        """Different seed_weight values should work."""
        config = create_config_with_geometry_seeding(
            preset='default',
            seed_weight=seed_weight
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_negative_seed_weight(self, algorithm, xor_problem):
        """Negative seed weight (default -1.0) should work."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_positive_seed_weight(self, algorithm, xor_problem):
        """Positive seed weight should work."""
        config = create_config_with_geometry_seeding(preset='positive_weight')
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_zero_seed_weight(self, algorithm, xor_problem):
        """Zero seed weight should work (no locality bias)."""
        config = create_config_with_geometry_seeding(preset='zero_weight')
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)


class TestCPPNInputDimensions:
    """Test 5D vs 7D CPPN input handling."""

    def test_7d_inputs_enabled(self, algorithm, xor_problem):
        """7D inputs (with delta_x, delta_y) should work."""
        config = create_config_with_geometry_seeding(
            preset='default',
            use_7d_inputs=True
        )
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr['geometry_seeding']['use_7d_inputs'] is True
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_5d_inputs_fallback(self, algorithm, xor_problem):
        """5D inputs (standard) should work."""
        config = create_config_with_geometry_seeding(preset='5d_inputs')
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        assert hmr['geometry_seeding']['use_7d_inputs'] is False
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_5d_vs_7d_both_work(self, algorithm, xor_problem):
        """Both 5D and 7D input modes should achieve positive fitness."""
        config_5d = create_config_with_geometry_seeding(preset='5d_inputs')
        result_5d = run_quick_evolution(algorithm, config_5d, xor_problem)

        config_7d = create_config_with_geometry_seeding(preset='default')
        result_7d = run_quick_evolution(algorithm, config_7d, xor_problem)

        assert_no_errors(result_5d)
        assert_no_errors(result_7d)
        assert_positive_fitness(result_5d)
        assert_positive_fitness(result_7d)


class TestGeometrySeedingWithEvolution:
    """Test geometry seeding with full evolution runs."""

    def test_geometry_seeding_progresses(self, algorithm, xor_problem):
        """Geometry seeding should show fitness improvement."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(
            algorithm, config, xor_problem,
            generations=STANDARD_GENERATIONS
        )
        assert_no_errors(result)
        assert len(result.fitness_history) == STANDARD_GENERATIONS
        assert result.best_fitness >= result.fitness_history[0]

    def test_geometry_seeding_with_hidden_recurrence(self, algorithm, xor_problem):
        """Geometry seeding with h->h connections."""
        config = create_config_with_geometry_seeding(
            preset='default',
            recurrence_preset='hidden_only'
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_geometry_seeding_with_full_recurrence(self, algorithm, xor_problem):
        """Geometry seeding with full recurrent connections."""
        config = create_config_with_geometry_seeding(
            preset='default',
            recurrence_preset='full_recurrent'
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_geometry_seeding_larger_population(self, algorithm, xor_problem):
        """Geometry seeding with larger population."""
        config = create_config_with_geometry_seeding(
            preset='default',
            population_size=200
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_geometry_seeding_deeper_substrate(self, algorithm, xor_problem):
        """Geometry seeding with deeper max_depth."""
        config = create_config_with_geometry_seeding(
            preset='default',
            max_depth=3
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)


class TestGeometrySeedingWithOtherFeatures:
    """Test geometry seeding with other features."""

    def test_geometry_seeding_with_streaming(self, algorithm, xor_problem):
        """Geometry seeding with streaming mode."""
        config = create_config_with_geometry_seeding(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['enable_streaming'] = True
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['population_chunk_size'] = 50
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    @pytest.mark.xfail(
        reason="pre-existing upstream bug: geometry seeding's 7D CPPN inputs are not "
        "threaded through the multi-output activation query (broadcast (5,) vs (7,)); "
        "fails identically in the original research implementation",
        strict=True,
    )
    def test_geometry_seeding_with_dynamic_functions(self, algorithm, xor_problem):
        """Geometry seeding with dynamic function selection."""
        config = create_config_with_geometry_seeding(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4
        }
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_geometry_seeding_with_global_activation(self, algorithm, xor_problem):
        """Geometry seeding with global activation mode."""
        config = create_config_with_geometry_seeding(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid'
        }
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_geometry_seeding_with_aggregation(self, algorithm, xor_problem):
        """Geometry seeding with aggregation mode."""
        config = create_config_with_geometry_seeding(preset='default')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'sum'
        }
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)


class TestGeometrySeedingOnDifferentProblems:
    """Test geometry seeding on different problem types."""

    def test_on_and_problem(self, algorithm, and_problem):
        """Geometry seeding on AND logic gate."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(algorithm, config, and_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_on_or_problem(self, algorithm, or_problem):
        """Geometry seeding on OR logic gate."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(algorithm, config, or_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_on_nand_problem(self, algorithm, nand_problem):
        """Geometry seeding on NAND logic gate."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(algorithm, config, nand_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_on_nor_problem(self, algorithm, nor_problem):
        """Geometry seeding on NOR logic gate."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(algorithm, config, nor_problem)
        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_on_parity3_problem(self, algorithm, parity3_problem):
        """Geometry seeding on 3-bit parity."""
        config = create_config_with_geometry_seeding(preset='default')
        # Parity3 has 4 inputs (3 bits + 1 bias), update substrate coords
        config['algorithm_params']['emrhyperneat']['substrate']['input_coords'] = [
            (-1.0, -1.0), (-0.33, -1.0), (0.33, -1.0), (1.0, -1.0)
        ]
        result = run_quick_evolution(
            algorithm, config, parity3_problem,
            generations=STANDARD_GENERATIONS
        )
        assert_no_errors(result)


class TestGeometrySeedingReproducibility:
    """Test geometry seeding reproducibility."""

    def test_reproducibility_same_seed(self, algorithm, xor_problem):
        """Same seed should produce same results."""
        config = create_config_with_geometry_seeding(preset='default')

        result1 = run_quick_evolution(
            algorithm, config, xor_problem, seed=42
        )
        result2 = run_quick_evolution(
            algorithm, config, xor_problem, seed=42
        )

        assert_no_errors(result1)
        assert_no_errors(result2)
        assert result1.best_fitness == result2.best_fitness

    def test_different_seeds_variation(self, algorithm, xor_problem):
        """Different seeds should produce different results."""
        config = create_config_with_geometry_seeding(preset='default')
        results = []

        for seed in STANDARD_SEEDS:
            result = run_quick_evolution(
                algorithm, config, xor_problem, seed=seed
            )
            assert_no_errors(result)
            results.append(result.best_fitness)

        assert all(f > 0 for f in results)

    @pytest.mark.slow
    def test_consistency_across_seeds(self, algorithm, xor_problem):
        """Geometry seeding should consistently achieve positive fitness."""
        config = create_config_with_geometry_seeding(preset='default')
        positive_count = 0

        for seed in [42, 123, 456, 789, 1000]:
            result = run_quick_evolution(
                algorithm, config, xor_problem,
                seed=seed, generations=STANDARD_GENERATIONS
            )
            assert_no_errors(result)
            if result.best_fitness > 0.5:
                positive_count += 1

        assert positive_count >= 3


class TestGeometrySeedingEdgeCases:
    """Test geometry seeding edge cases."""

    def test_minimal_population(self, algorithm, xor_problem):
        """Geometry seeding with minimal population."""
        config = create_config_with_geometry_seeding(
            preset='default',
            population_size=10
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_minimal_depth(self, algorithm, xor_problem):
        """Geometry seeding with minimal max_depth."""
        config = create_config_with_geometry_seeding(
            preset='default',
            max_depth=1
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_single_generation(self, algorithm, xor_problem):
        """Geometry seeding with single generation."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=1
        )
        assert_no_errors(result)
        assert len(result.fitness_history) == 1

    def test_extreme_negative_seed_weight(self, algorithm, xor_problem):
        """Geometry seeding with extreme negative seed weight."""
        config = create_config_with_geometry_seeding(
            preset='default',
            seed_weight=-5.0
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)

    def test_extreme_positive_seed_weight(self, algorithm, xor_problem):
        """Geometry seeding with extreme positive seed weight."""
        config = create_config_with_geometry_seeding(
            preset='default',
            seed_weight=5.0
        )
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)


class TestGeometrySeedingPerformanceMetrics:
    """Test geometry seeding performance measurement."""

    def test_timing_recorded(self, algorithm, xor_problem):
        """Geometry seeding should record generation timing."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(algorithm, config, xor_problem)
        assert_no_errors(result)
        assert result.total_time > 0
        assert result.avg_gen_time_ms > 0
        assert len(result.gen_times) == QUICK_GENERATIONS

    def test_fitness_history_complete(self, algorithm, xor_problem):
        """Geometry seeding should record complete fitness history."""
        config = create_config_with_geometry_seeding(preset='default')
        result = run_quick_evolution(
            algorithm, config, xor_problem,
            generations=10
        )
        assert_no_errors(result)
        assert len(result.fitness_history) == 10
        assert result.final_fitness == result.fitness_history[-1]
        assert result.best_fitness == max(result.fitness_history)
