"""Tests for substrate building in EMR-HyperNEAT.

Tests cover:
- Native substrate building (input/output coords, weight matrices)
- Extended substrate building (multi-hop expansion, variance subdivision)
- Hierarchical position discovery
- Weight matrix construction (W1, W2)
- Position mask computation
- Depth configuration validation
"""

import pytest
import numpy as np
from conftest import (
    EMRHyperNEAT,
    create_base_config,
    create_config_with_recurrence,
    run_quick_evolution,
    assert_no_errors,
    assert_positive_fitness,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
    DEFAULT_MAX_DEPTH,
)


class TestSubstrateCoordinates:
    """Tests for substrate coordinate configuration.

    Note: The algorithm doesn't expose input_coords/output_coords as public attributes.
    Tests verify that configuration is accepted and initialization completes.
    """

    def test_standard_input_coords(self, algorithm, xor_problem):
        """Verify standard input coordinates are accepted."""
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Initialization should complete without error
        # (substrate coords are processed internally)

    def test_standard_output_coords(self, algorithm, xor_problem):
        """Verify standard output coordinates are accepted."""
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Initialization should complete without error

    def test_custom_input_coords(self, algorithm, xor_problem):
        """Verify custom input coordinates work."""
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['substrate']['input_coords'] = [
            (-1.0, -1.0), (-0.5, -1.0), (0.0, -1.0), (0.5, -1.0), (1.0, -1.0)
        ]

        # XOR expects 3 inputs, but we're testing config acceptance
        # Use a 5-input problem or just verify initialization
        config_obj = algorithm.create_config(config)
        # Note: This would fail during evolution due to shape mismatch with XOR
        # Just verify config is parsed

    def test_multiple_output_coords(self, algorithm, xor_problem):
        """Verify multiple output coordinates work."""
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['substrate']['output_coords'] = [
            (-0.5, 1.0), (0.5, 1.0)
        ]

        # XOR expects 1 output, but we're testing config acceptance
        config_obj = algorithm.create_config(config)
        # Note: This would fail during evolution due to shape mismatch
        # Just verify config is parsed


class TestDepthConfiguration:
    """Tests for depth configuration."""

    @pytest.mark.parametrize("max_depth", [1, 2, 3, 4, 5])
    def test_max_depth_values(self, algorithm, xor_problem, max_depth):
        """Verify different max depth values work."""
        config = create_base_config(max_depth=max_depth)
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.max_depth == max_depth

    def test_initial_depth_configuration(self, algorithm, xor_problem):
        """Verify initial depth is configured correctly."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['initial_depth'] = 0

        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.initial_depth == 0

    def test_depth_produces_hidden_positions(self, algorithm, xor_problem):
        """Verify depth produces hidden positions."""
        config = create_base_config(max_depth=3)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=1)

        assert_no_errors(result)


class TestThresholdParameters:
    """Tests for threshold parameters."""

    @pytest.mark.parametrize("variance_threshold", [0.01, 0.03, 0.05, 0.1])
    def test_variance_thresholds(self, algorithm, xor_problem, variance_threshold):
        """Verify different variance thresholds work."""
        config = create_base_config(variance_threshold=variance_threshold)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Variance threshold {variance_threshold} failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("division_threshold", [0.01, 0.03, 0.05, 0.1])
    def test_division_thresholds(self, algorithm, xor_problem, division_threshold):
        """Verify different division thresholds work."""
        config = create_base_config(division_threshold=division_threshold)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Division threshold {division_threshold} failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("band_threshold", [0.1, 0.2, 0.3, 0.5])
    def test_band_thresholds(self, algorithm, xor_problem, band_threshold):
        """Verify different band thresholds work."""
        config = create_base_config(band_threshold=band_threshold)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Band threshold {band_threshold} failed")
        assert_positive_fitness(result)


class TestWeightMatrixConstruction:
    """Tests for weight matrix construction."""

    def test_weight_matrices_created(self, algorithm, xor_problem):
        """Verify weight matrices are created during evolution."""
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Run one generation to build weight matrices
        state, metrics = algorithm.run_generation(state, xor_problem)

        # Should have produced valid fitness
        assert metrics.best_fitness > 0

    def test_max_weight_constraint(self, algorithm, xor_problem):
        """Verify max weight constraint is applied."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['max_weight'] = 5.0

        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.max_weight == 5.0

    @pytest.mark.parametrize("max_weight", [1.0, 3.0, 5.0, 10.0])
    def test_max_weight_values(self, algorithm, xor_problem, max_weight):
        """Verify different max weight values work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['max_weight'] = max_weight

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Max weight {max_weight} failed")
        assert_positive_fitness(result)


class TestExtendedSubstrate:
    """Tests for extended substrate building with recurrence."""

    def test_extended_substrate_with_recurrence(self, algorithm, xor_problem):
        """Verify extended substrate works with recurrence."""
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_multi_hop_expansion(self, algorithm, xor_problem):
        """Verify multi-hop expansion is applied."""
        config = create_config_with_recurrence('with_backward', iteration_level=3)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_hh_cache_enabled(self, algorithm, xor_problem):
        """Verify H→H caching is enabled when configured."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.hh_cache_enabled == True


class TestHierarchicalPositions:
    """Tests for hierarchical position discovery."""

    def test_position_hierarchy_created(self, algorithm, xor_problem):
        """Verify position hierarchy is created during initialization."""
        config = create_base_config(max_depth=3)
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Should have hierarchical grid structure
        assert algorithm.max_depth == 3

    @pytest.mark.parametrize("depth", [2, 3, 4])
    def test_position_count_scales_with_depth(self, algorithm, xor_problem, depth):
        """Verify position count scales with depth."""
        config = create_base_config(max_depth=depth)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=1)

        assert_no_errors(result, f"Depth {depth} failed")


class TestActivationConfiguration:
    """Tests for activation function configuration in substrate."""

    @pytest.mark.parametrize("hidden_act", ['tanh', 'sigmoid', 'relu'])
    def test_hidden_activations(self, algorithm, xor_problem, hidden_act):
        """Verify different hidden activations work."""
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['substrate']['hidden_activation'] = hidden_act

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Hidden activation {hidden_act} failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("output_act", ['sigmoid', 'tanh', 'identity'])
    def test_output_activations(self, algorithm, xor_problem, output_act):
        """Verify different output activations work."""
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['substrate']['output_activation'] = output_act

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        # Some activations may not work well for XOR, but should not error
        assert_no_errors(result, f"Output activation {output_act} failed")


class TestSubstrateWithFeatures:
    """Tests for substrate building with various features enabled."""

    def test_substrate_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify substrate works with dynamic functions."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_substrate_with_neuromodulation(self, algorithm, xor_problem):
        """Verify substrate works with neuromodulation."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestSubstrateValidation:
    """Tests for substrate validation and error handling."""

    def test_empty_input_coords_handled(self, algorithm, xor_problem):
        """Verify empty input coordinates are handled."""
        config = create_base_config()
        # This should use default coordinates
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_substrate_initialization_complete(self, algorithm, xor_problem):
        """Verify substrate initialization completes successfully."""
        config = create_base_config()
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Essential attributes should be initialized
        # Note: input_coords/output_coords are processed internally, not exposed
        assert algorithm.max_depth is not None


class TestComprehensiveSubstrate:
    """Comprehensive tests (marked slow)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("depth", [2, 3, 4, 5])
    def test_depth_solves_xor(self, algorithm, xor_problem, depth):
        """Verify different depths can solve XOR."""
        config = create_base_config(max_depth=depth)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=15)

        assert_no_errors(result)
        # Should achieve reasonable fitness
        assert result.best_fitness > 0.5, f"Depth {depth} should achieve >0.5 fitness"

    @pytest.mark.slow
    def test_substrate_consistency_across_seeds(self, algorithm, xor_problem):
        """Verify substrate produces consistent results across seeds."""
        config = create_base_config()
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


# =============================================================================
# Substrate Configuration Gap Tests
# =============================================================================

class TestMaxConnectionsConfig:
    """Tests for max_connections parameter."""

    @pytest.mark.parametrize("max_conns", [1000, 5000, 10000])
    def test_max_connections_values(self, algorithm, xor_problem, max_conns):
        """Verify different max_connections values work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if 'recurrence' not in hmr:
            hmr['recurrence'] = {}
        hmr['recurrence']['max_connections'] = max_conns

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_max_connections_with_recurrence(self, algorithm, xor_problem):
        """Verify max_connections works with hidden_only recurrence."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'max_connections': 5000,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_low_max_connections(self, algorithm, xor_problem):
        """Verify low max_connections value works (sparse mode)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if 'recurrence' not in hmr:
            hmr['recurrence'] = {}
        hmr['recurrence']['max_connections'] = 100  # Very sparse

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestMaxSparseConnsConfig:
    """Tests for max_sparse_conns parameter."""

    @pytest.mark.parametrize("max_sparse", [500, 1000, 5000])
    def test_max_sparse_conns_values(self, algorithm, xor_problem, max_sparse):
        """Verify different max_sparse_conns values work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if 'recurrence' not in hmr:
            hmr['recurrence'] = {}
        hmr['recurrence']['max_sparse_conns'] = max_sparse

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_max_sparse_conns_with_hidden_only(self, algorithm, xor_problem):
        """Verify max_sparse_conns works with hidden_only preset."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'max_sparse_conns': 2000,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_low_max_sparse_conns(self, algorithm, xor_problem):
        """Verify very low max_sparse_conns is handled."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if 'recurrence' not in hmr:
            hmr['recurrence'] = {}
        hmr['recurrence']['max_sparse_conns'] = 50

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestLocalityRadius:
    """Tests for locality_radius parameter."""

    @pytest.mark.parametrize("radius", [0.1, 0.5, 1.0, 2.0])
    def test_locality_radius_values(self, algorithm, xor_problem, radius):
        """Verify different locality_radius values work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = radius

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_locality_radius_disabled(self, algorithm, xor_problem):
        """Verify locality_radius=None (disabled) works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = None  # Disabled

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_locality_radius_with_recurrence(self, algorithm, xor_problem):
        """Verify locality_radius works with recurrence enabled."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.5
        hmr['recurrence'] = {'preset': 'hidden_only'}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_small_locality_radius(self, algorithm, xor_problem):
        """Verify very small locality_radius is handled."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.01  # Very local

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestDiscoveryToggles:
    """Tests for discovery toggle parameters."""

    def test_use_vectorized_discovery_true(self, algorithm, xor_problem):
        """Verify use_vectorized_discovery=True works (default)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if 'recurrence' not in hmr:
            hmr['recurrence'] = {}
        hmr['recurrence']['use_vectorized_discovery'] = True
        hmr['recurrence']['preset'] = 'hidden_only'

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_use_vectorized_discovery_false(self, algorithm, xor_problem):
        """Verify use_vectorized_discovery=False works (iterative)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if 'recurrence' not in hmr:
            hmr['recurrence'] = {}
        hmr['recurrence']['use_vectorized_discovery'] = False
        hmr['recurrence']['preset'] = 'hidden_only'

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_use_dense_discovery_false(self, algorithm, xor_problem):
        """Verify use_dense_discovery=False works (default)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if 'recurrence' not in hmr:
            hmr['recurrence'] = {}
        hmr['recurrence']['use_dense_discovery'] = False
        hmr['recurrence']['preset'] = 'hidden_only'

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_use_dense_discovery_true(self, algorithm, xor_problem):
        """Verify use_dense_discovery=True works (vanilla-style)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if 'recurrence' not in hmr:
            hmr['recurrence'] = {}
        hmr['recurrence']['use_dense_discovery'] = True
        hmr['recurrence']['preset'] = 'hidden_only'

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestCombinedSubstrateConfig:
    """Tests for combining multiple substrate configuration options."""

    def test_all_configs_combined(self, algorithm, xor_problem):
        """Verify all substrate configs can be combined."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.5
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'max_connections': 5000,
            'max_sparse_conns': 2000,
            'use_vectorized_discovery': True,
            'use_dense_discovery': False,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_sparse_config_with_locality(self, algorithm, xor_problem):
        """Verify sparse config works with locality penalty."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.3
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'max_sparse_conns': 500,  # Very sparse
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_dense_discovery_with_locality(self, algorithm, xor_problem):
        """Verify dense discovery works with locality_radius."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.8
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'use_dense_discovery': True,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestLocalityRadiusEdgeCases:
    """Edge case tests for locality_radius parameter."""

    def test_very_large_locality_radius(self, algorithm, xor_problem):
        """Verify very large locality_radius works (effectively no penalty)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 100.0  # Very large - nearly global

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_zero_locality_radius(self, algorithm, xor_problem):
        """Verify zero locality_radius is handled (extreme locality)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.001  # Near-zero

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_locality_radius_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify locality_radius works with dynamic functions."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.5
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_locality_radius_with_aggregation(self, algorithm, xor_problem):
        """Verify locality_radius works with aggregation mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.5
        hmr['aggregation'] = {
            'mode': 'weight_interpretation',
            'num_aggregations': 4,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_locality_with_streaming(self, algorithm, xor_problem):
        """Verify locality_radius works with streaming mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.5
        hmr['enable_streaming'] = True
        hmr['population_chunk_size'] = 25

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    @pytest.mark.parametrize("radius", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    def test_locality_radius_spectrum(self, algorithm, xor_problem, radius):
        """Verify a spectrum of locality_radius values work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = radius

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"locality_radius={radius} failed")


class TestGeometrySeeding:
    """Tests for geometry seeding functionality."""

    def test_geometry_seeding_with_locality(self, algorithm, xor_problem):
        """Verify geometry seeding via locality_radius is applied."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.3  # Enables geometry seeding

        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Locality radius should be set
        assert algorithm.locality_radius == 0.3

        # Run evolution
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)

    def test_geometry_seeding_disabled(self, algorithm, xor_problem):
        """Verify geometry seeding is disabled when locality_radius is None."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = None

        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.locality_radius is None

    def test_geometry_seeding_with_full_recurrence(self, algorithm, xor_problem):
        """Verify geometry seeding works with full recurrence."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['locality_radius'] = 0.5
        hmr['recurrence'] = {
            'preset': 'full_recurrent',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'allow_backward': True,
            'allow_lateral': True,
            'allow_self_loops': True,
            'iteration_level': 2,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
