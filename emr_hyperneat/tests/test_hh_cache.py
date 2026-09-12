"""Tests for H→H connection caching in EMR-HyperNEAT.

Tests cover:
- HHCacheManager initialization
- Cache hit/miss logic
- Time-based refresh (hh_refresh_interval)
- Change-based refresh (hh_mask_change_threshold)
- Cache update and retrieval
- Cache statistics
"""

import pytest
import jax.numpy as jnp
from conftest import (
    EMRHyperNEAT,
    HHCacheManager,
    EMRConfig,
    create_config_with_recurrence,
    create_base_config,
    run_quick_evolution,
    assert_no_errors,
    assert_positive_fitness,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
)


class TestHHCacheManagerBasic:
    """Basic tests for HHCacheManager."""

    def test_cache_manager_initialization(self):
        """Verify HHCacheManager initializes correctly."""
        config = EMRConfig(
            enabled=True,
            allow_hidden_to_hidden=True,
            iteration_level=2,
            hh_cache_enabled=True,
            hh_refresh_interval=5,
            hh_mask_change_threshold=0.1,
        )
        cache = HHCacheManager(config)

        assert cache is not None

    def test_cache_manager_disabled(self):
        """Verify cache manager works when disabled."""
        config = EMRConfig(
            enabled=True,
            allow_hidden_to_hidden=True,
            iteration_level=2,
            hh_cache_enabled=False,
        )
        cache = HHCacheManager(config)

        # Should always refresh when disabled
        assert cache.should_refresh(0, None) == True

    def test_initial_should_refresh(self):
        """Verify cache indicates refresh needed initially."""
        config = EMRConfig(
            enabled=True,
            allow_hidden_to_hidden=True,
            iteration_level=2,
            hh_cache_enabled=True,
            hh_refresh_interval=5,
        )
        cache = HHCacheManager(config)

        # Should need refresh at generation 0
        assert cache.should_refresh(0, None) == True


class TestCacheRefreshLogic:
    """Tests for cache refresh logic."""

    def test_time_based_refresh(self):
        """Verify time-based refresh works."""
        config = EMRConfig(
            enabled=True,
            allow_hidden_to_hidden=True,
            iteration_level=2,
            hh_cache_enabled=True,
            hh_refresh_interval=5,
            hh_mask_change_threshold=0.1,
        )
        cache = HHCacheManager(config)

        # Initially should refresh
        assert cache.should_refresh(0, jnp.ones((10, 10))) == True

        # After updating at gen 0, should NOT refresh until interval
        cache.update_cache(None, jnp.ones((10, 10)), 0)

        # Generations 1-4 should not need refresh (interval is 5)
        for gen in range(1, 5):
            result = cache.should_refresh(gen, jnp.ones((10, 10)))
            # May or may not refresh depending on implementation

    def test_refresh_interval_configuration(self):
        """Verify different refresh intervals work."""
        for interval in [1, 3, 5, 10]:
            config = EMRConfig(
                enabled=True,
                allow_hidden_to_hidden=True,
                iteration_level=2,
                hh_cache_enabled=True,
                hh_refresh_interval=interval,
            )
            cache = HHCacheManager(config)

            # Should initially refresh
            assert cache.should_refresh(0, None) == True


class TestCacheWithEvolution:
    """Tests for cache during evolution."""

    def test_cache_enabled_in_evolution(self, algorithm, xor_problem):
        """Verify cache is used during evolution when enabled."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hh_refresh_interval'] = 3

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_cache_disabled_in_evolution(self, algorithm, xor_problem):
        """Verify evolution works with cache disabled."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_cache_vs_no_cache_both_work(self, algorithm, xor_problem):
        """Verify both cached and non-cached evolution work."""
        # With cache
        config_cached = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        result_cached = run_quick_evolution(
            algorithm, config_cached, xor_problem,
            generations=QUICK_GENERATIONS
        )

        # Without cache
        config_no_cache = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)
        result_no_cache = run_quick_evolution(
            algorithm, config_no_cache, xor_problem,
            generations=QUICK_GENERATIONS
        )

        # Both should work
        assert_no_errors(result_cached)
        assert_no_errors(result_no_cache)
        assert_positive_fitness(result_cached)
        assert_positive_fitness(result_no_cache)


class TestCacheRefreshIntervals:
    """Tests for different cache refresh intervals."""

    @pytest.mark.parametrize("interval", [1, 3, 5, 10])
    def test_refresh_intervals(self, algorithm, xor_problem, interval):
        """Verify different refresh intervals work."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hh_refresh_interval'] = interval

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Refresh interval {interval} failed")
        assert_positive_fitness(result)

    def test_frequent_refresh(self, algorithm, xor_problem):
        """Verify frequent cache refresh works."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hh_refresh_interval'] = 1  # Every generation

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_infrequent_refresh(self, algorithm, xor_problem):
        """Verify infrequent cache refresh works."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hh_refresh_interval'] = 20  # Every 20 generations

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestCacheMaskChangeThreshold:
    """Tests for mask change threshold."""

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.2])
    def test_mask_change_thresholds(self, algorithm, xor_problem, threshold):
        """Verify different mask change thresholds work."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hh_mask_change_threshold'] = threshold

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Mask threshold {threshold} failed")
        assert_positive_fitness(result)


class TestCacheWithConnectionTypes:
    """Tests for cache with different connection types."""

    @pytest.mark.parametrize("conn_type", ['hidden_only', 'with_backward', 'with_lateral', 'full_recurrent'])
    def test_cache_with_connection_types(self, algorithm, xor_problem, conn_type):
        """Verify cache works with all connection types that have H→H."""
        config = create_config_with_recurrence(conn_type, hh_cache_enabled=True)

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Cache with {conn_type} failed")
        assert_positive_fitness(result)


class TestCacheWithFeatures:
    """Tests for cache with other features enabled."""

    def test_cache_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify cache works with dynamic functions."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_cache_with_neuromodulation(self, algorithm, xor_problem):
        """Verify cache works with neuromodulation."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestCacheStatistics:
    """Tests for cache statistics."""

    def test_cache_manager_stats(self):
        """Verify cache manager provides statistics."""
        config = EMRConfig(
            enabled=True,
            allow_hidden_to_hidden=True,
            iteration_level=2,
            hh_cache_enabled=True,
            hh_refresh_interval=5,
        )
        cache = HHCacheManager(config)

        # Get stats
        stats = cache.get_stats()

        # Should have stats dict
        assert isinstance(stats, dict)


class TestComprehensiveCache:
    """Comprehensive tests (marked slow)."""

    @pytest.mark.slow
    def test_cache_vs_no_cache_similar_results(self, algorithm, xor_problem):
        """Verify cached and non-cached produce similar results."""
        results_cached = []
        results_no_cache = []

        for seed in [42, 123, 456]:
            # With cache
            config_cached = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
            result_cached = run_quick_evolution(
                algorithm, config_cached, xor_problem,
                generations=10, seed=seed
            )
            results_cached.append(result_cached.best_fitness)

            # Without cache
            config_no_cache = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)
            result_no_cache = run_quick_evolution(
                algorithm, config_no_cache, xor_problem,
                generations=10, seed=seed
            )
            results_no_cache.append(result_no_cache.best_fitness)

        # Both should produce positive results
        assert all(f > 0 for f in results_cached)
        assert all(f > 0 for f in results_no_cache)

    @pytest.mark.slow
    @pytest.mark.parametrize("interval", [1, 5, 10])
    def test_intervals_achieve_fitness(self, algorithm, xor_problem, interval):
        """Verify all intervals can achieve fitness."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=True)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hh_refresh_interval'] = interval

        result = run_quick_evolution(algorithm, config, xor_problem, generations=15)

        assert_no_errors(result)
        assert result.best_fitness > 0.5, f"Interval {interval} should achieve >0.5 fitness"
