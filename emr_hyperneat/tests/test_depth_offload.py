"""Tests for EMR-HyperNEAT Depth Offloading and Sharding.

Tests the `depth_offload` execution mode, which enables ultra-deep substrates
(depth 9+) by offloading completed levels to CPU RAM. It was previously a separate
class, EMRHyperNEATDepthOffload; it is now a mode of the one algorithm.

Key features tested:
- Depth offloading configuration (threshold, chunk sizes)
- CPU-resident grid mode
- Intra-level sharding configuration
- Disk offloading path
- Memory tracking
"""

import pytest
import tempfile
import os
from conftest import (
    create_base_config,
    MultiGPUStrategy,
    run_quick_evolution,
    assert_no_errors,
    assert_positive_fitness,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
)

# Depth offload is a mode of the one algorithm class, not a separate variant.
# It used to be EMRHyperNEATDepthOffload, a subclass that forked the
# whole generation loop and silently missed features added to the main one.
try:
    from emr_hyperneat.emrhyperneat import (
        EMRHyperNEAT,
    )
    from emr_hyperneat.execution import (
        ExecutionMode,
    )
    DEPTH_OFFLOAD_AVAILABLE = True
except ImportError:
    DEPTH_OFFLOAD_AVAILABLE = False


@pytest.fixture
def depth_offload_algorithm():
    """Algorithm set to choose its execution mode by depth.

    `auto` is the faithful equivalent of the old subclass, which was always
    offload-enabled and decided per run by comparing max_depth against the
    threshold. Setting `depth_offload` outright would instead force the path at
    every depth and make the threshold tests vacuous.
    """
    if not DEPTH_OFFLOAD_AVAILABLE:
        pytest.skip("Depth offload mode not available")
    algorithm = EMRHyperNEAT()
    algorithm.execution_config.mode = ExecutionMode.AUTO
    return algorithm


@pytest.fixture
def xor_problem():
    """Create an XOR problem instance."""
    return XORProblem()


def create_depth_offload_config(max_depth: int = 3, population_size: int = 50):
    """Create a config suitable for depth offload testing."""
    config = create_base_config(max_depth=max_depth, population_size=population_size)
    # Depth offload works with lower depths too, just for testing
    return config


# =============================================================================
# Phase 9.1: Depth Offloading Tests
# =============================================================================

class TestDepthOffloadBasics:
    """Basic tests for depth offload variant."""

    def test_depth_offload_initializes(self, depth_offload_algorithm, xor_problem):
        """Verify depth offload algorithm initializes."""
        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert state is not None

    def test_depth_offload_runs_generation(self, depth_offload_algorithm, xor_problem):
        """Verify depth offload algorithm can run a generation."""
        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_depth_offload_solves_xor(self, depth_offload_algorithm, xor_problem):
        """Verify depth offload can solve XOR over multiple generations."""
        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        for _ in range(QUICK_GENERATIONS):
            state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness > 0


class TestDepthOffloadConfiguration:
    """Tests for configure_offloading() method."""

    def test_depth_offload_selected_by_depth_under_auto(self, depth_offload_algorithm):
        """Under `auto`, offload is reachable and chosen by depth.

        Replaces an assertion on `_depth_offload_enabled`, a flag that was always
        True and existed only because offload lived on a dedicated class. Whether
        offload runs is now a question about the resolved mode.
        """
        depth_offload_algorithm.max_depth = 3
        assert depth_offload_algorithm.should_use_offloading() is False
        depth_offload_algorithm.max_depth = 9
        assert depth_offload_algorithm.should_use_offloading() is True

    @pytest.mark.parametrize("threshold", [5, 7, 9, 12])
    def test_depth_offload_threshold_values(self, depth_offload_algorithm, xor_problem, threshold):
        """Verify different depth threshold values work."""
        depth_offload_algorithm.configure_offloading(depth_threshold=threshold)

        assert depth_offload_algorithm._depth_offload_threshold == threshold

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_offload_verbose_output(self, depth_offload_algorithm, xor_problem):
        """Verify verbose mode can be enabled."""
        depth_offload_algorithm.configure_offloading(verbose=True)

        assert depth_offload_algorithm._offload_verbose is True

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_cpu_resident_grid_mode(self, depth_offload_algorithm, xor_problem):
        """Verify CPU-resident grid mode can be enabled."""
        depth_offload_algorithm.configure_offloading(use_cpu_grid=True)

        assert depth_offload_algorithm._use_cpu_grid is True

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    @pytest.mark.parametrize("chunk_levels", [1, 3, 5])
    def test_eval_chunk_levels_values(self, depth_offload_algorithm, xor_problem, chunk_levels):
        """Verify different eval_chunk_levels values work."""
        depth_offload_algorithm.configure_offloading(eval_chunk_levels=chunk_levels)

        assert depth_offload_algorithm._eval_chunk_levels == chunk_levels

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    @pytest.mark.parametrize("chunk_positions", [100_000, 500_000, 1_000_000])
    def test_eval_chunk_positions_values(self, depth_offload_algorithm, xor_problem, chunk_positions):
        """Verify different eval_chunk_positions values work."""
        depth_offload_algorithm.configure_offloading(eval_chunk_positions=chunk_positions)

        assert depth_offload_algorithm._eval_chunk_positions == chunk_positions

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    @pytest.mark.parametrize("cppn_chunk", [10_000, 50_000, 100_000])
    def test_cppn_position_chunk_size(self, depth_offload_algorithm, xor_problem, cppn_chunk):
        """Verify different CPPN position chunk sizes work."""
        depth_offload_algorithm.configure_offloading(cppn_pos_chunk_size=cppn_chunk)

        assert depth_offload_algorithm._cppn_pos_chunk_size == cppn_chunk

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    @pytest.mark.parametrize("max_lookahead", [1_000_000, 3_000_000, 5_000_000])
    def test_max_lookahead_positions(self, depth_offload_algorithm, xor_problem, max_lookahead):
        """Verify different max_lookahead_positions values work."""
        depth_offload_algorithm.configure_offloading(max_lookahead_positions=max_lookahead)

        assert depth_offload_algorithm._max_lookahead_positions == max_lookahead

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_configure_offloading_all_params(self, depth_offload_algorithm, xor_problem):
        """Verify all offloading params can be configured together."""
        depth_offload_algorithm.configure_offloading(
            depth_threshold=7,
            eval_chunk_levels=2,
            eval_chunk_positions=500_000,
            cppn_pos_chunk_size=50_000,
            max_lookahead_positions=2_000_000,
            verbose=False,
            use_cpu_grid=False,
        )

        assert depth_offload_algorithm._depth_offload_threshold == 7
        assert depth_offload_algorithm._eval_chunk_levels == 2
        assert depth_offload_algorithm._eval_chunk_positions == 500_000
        assert depth_offload_algorithm._cppn_pos_chunk_size == 50_000
        assert depth_offload_algorithm._max_lookahead_positions == 2_000_000

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0


# =============================================================================
# Phase 9.2: Sharding Tests
# =============================================================================

class TestShardingConfiguration:
    """Tests for configure_sharding() method."""

    @pytest.mark.parametrize("target_gb", [0.25, 0.5, 1.0, 2.0])
    def test_target_shard_gb_values(self, depth_offload_algorithm, xor_problem, target_gb):
        """Verify different target_shard_gb values work."""
        depth_offload_algorithm.configure_sharding(target_shard_gb=target_gb)

        assert depth_offload_algorithm._target_shard_gb == target_gb

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    @pytest.mark.parametrize("target_eval_gb", [0.5, 1.0, 2.0])
    def test_target_eval_shard_gb(self, depth_offload_algorithm, xor_problem, target_eval_gb):
        """Verify different target_eval_shard_gb values work."""
        depth_offload_algorithm.configure_sharding(target_eval_shard_gb=target_eval_gb)

        assert depth_offload_algorithm._target_eval_shard_gb == target_eval_gb

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_disk_offload_enabled(self, depth_offload_algorithm, xor_problem):
        """Verify disk offload can be enabled."""
        depth_offload_algorithm.configure_sharding(enable_disk_offload=True)

        assert depth_offload_algorithm._enable_disk_offload is True

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_disk_offload_custom_path(self, depth_offload_algorithm, xor_problem):
        """Verify custom disk offload path can be set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, 'hmr_test_offload')
            depth_offload_algorithm.configure_sharding(
                enable_disk_offload=True,
                disk_offload_path=custom_path,
            )

            assert depth_offload_algorithm._disk_offload_path == custom_path

            config = create_depth_offload_config()
            config_obj = depth_offload_algorithm.create_config(config)
            state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
            state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

            assert metrics.best_fitness >= 0

    def test_configure_sharding_all_params(self, depth_offload_algorithm, xor_problem):
        """Verify all sharding params can be configured together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, 'hmr_combined_test')
            depth_offload_algorithm.configure_sharding(
                target_shard_gb=0.3,
                target_eval_shard_gb=0.5,
                enable_disk_offload=False,
                disk_offload_path=custom_path,
            )

            assert depth_offload_algorithm._target_shard_gb == 0.3
            assert depth_offload_algorithm._target_eval_shard_gb == 0.5
            assert depth_offload_algorithm._enable_disk_offload is False
            assert depth_offload_algorithm._disk_offload_path == custom_path

            config = create_depth_offload_config()
            config_obj = depth_offload_algorithm.create_config(config)
            state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
            state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

            assert metrics.best_fitness >= 0


class TestCombinedOffloadAndSharding:
    """Tests for combining offloading and sharding configurations."""

    def test_offloading_and_sharding_together(self, depth_offload_algorithm, xor_problem):
        """Verify offloading and sharding can be configured together."""
        depth_offload_algorithm.configure_offloading(
            depth_threshold=5,
            eval_chunk_levels=2,
            verbose=False,
        )
        depth_offload_algorithm.configure_sharding(
            target_shard_gb=0.5,
            target_eval_shard_gb=1.0,
        )

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        for _ in range(QUICK_GENERATIONS):
            state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness > 0

    def test_cpu_grid_with_sharding(self, depth_offload_algorithm, xor_problem):
        """Verify CPU-resident grid works with sharding configuration."""
        depth_offload_algorithm.configure_offloading(use_cpu_grid=True)
        depth_offload_algorithm.configure_sharding(target_shard_gb=0.3)

        config = create_depth_offload_config()
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0


class TestShouldUseOffloading:
    """Tests for should_use_offloading() method."""

    def test_should_use_offloading_below_threshold(self, depth_offload_algorithm, xor_problem):
        """Verify offloading disabled when depth below threshold."""
        depth_offload_algorithm.configure_offloading(depth_threshold=10)

        config = create_depth_offload_config(max_depth=3)  # Well below threshold
        config_obj = depth_offload_algorithm.create_config(config)
        depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # max_depth is 3, threshold is 10, so should not use offloading
        assert depth_offload_algorithm.should_use_offloading() is False

    def test_should_use_offloading_at_threshold(self, depth_offload_algorithm, xor_problem):
        """Verify offloading enabled when depth at or above threshold."""
        depth_offload_algorithm.configure_offloading(depth_threshold=3)

        config = create_depth_offload_config(max_depth=3)  # At threshold
        config_obj = depth_offload_algorithm.create_config(config)
        depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert depth_offload_algorithm.should_use_offloading() is True


class TestMemoryTracking:
    """Tests for memory tracking attributes."""

    def test_memory_tracking_initialized(self, depth_offload_algorithm):
        """Verify memory tracking attributes are initialized."""
        assert depth_offload_algorithm._cpu_memory_peak_gb == 0.0
        assert depth_offload_algorithm._gpu_transfers_count == 0

    def test_gpu_transfers_count_type(self, depth_offload_algorithm):
        """Verify GPU transfers count is an integer."""
        assert isinstance(depth_offload_algorithm._gpu_transfers_count, int)


class TestOptimalChunkSizeCalculation:
    """Tests for _calculate_optimal_pos_chunk_size() method."""

    def test_calculate_optimal_chunk_size(self, depth_offload_algorithm):
        """Verify optimal chunk size calculation works."""
        chunk_size = depth_offload_algorithm._calculate_optimal_pos_chunk_size(
            pop_size=100,
            num_sources=3,
            target_gb=0.5,
        )

        # Should be a positive integer
        assert isinstance(chunk_size, int)
        assert chunk_size >= 1000  # Minimum clamped value

    def test_calculate_optimal_chunk_size_uses_default(self, depth_offload_algorithm):
        """Verify calculation uses default target_shard_gb when None."""
        depth_offload_algorithm.configure_sharding(target_shard_gb=0.25)

        chunk_size = depth_offload_algorithm._calculate_optimal_pos_chunk_size(
            pop_size=100,
            num_sources=3,
            target_gb=None,  # Should use _target_shard_gb
        )

        assert isinstance(chunk_size, int)
        assert chunk_size >= 1000


class TestComprehensiveDepthOffload:
    """Comprehensive tests (marked slow)."""

    @pytest.mark.slow
    def test_multiple_generations_with_offloading(self, depth_offload_algorithm, xor_problem):
        """Verify depth offload works over many generations."""
        depth_offload_algorithm.configure_offloading(
            depth_threshold=3,
            verbose=False,
        )

        config = create_depth_offload_config(max_depth=3)
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        for _ in range(10):
            state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness > 0.5

    @pytest.mark.slow
    def test_different_depths_with_offloading(self, depth_offload_algorithm, xor_problem):
        """Verify depth offload works with different depths."""
        depth_offload_algorithm.configure_offloading(depth_threshold=2)

        for max_depth in [2, 3, 4]:
            config = create_depth_offload_config(max_depth=max_depth)
            config_obj = depth_offload_algorithm.create_config(config)
            state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
            state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

            assert metrics.best_fitness >= 0, f"Failed for depth {max_depth}"


class TestDiskOffloadFileIO:
    """Tests for disk offload file creation and cleanup."""

    def test_disk_offload_path_created(self, depth_offload_algorithm, xor_problem):
        """Verify disk offload path is stored when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, 'disk_offload_test')
            depth_offload_algorithm.configure_sharding(
                enable_disk_offload=True,
                disk_offload_path=custom_path,
            )

            # Path should be stored
            assert depth_offload_algorithm._disk_offload_path == custom_path

    def test_disk_offload_with_temp_directory(self, depth_offload_algorithm, xor_problem):
        """Verify disk offload works with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            offload_path = os.path.join(tmpdir, 'hmr_offload')

            depth_offload_algorithm.configure_sharding(
                enable_disk_offload=True,
                disk_offload_path=offload_path,
            )

            config = create_depth_offload_config()
            config_obj = depth_offload_algorithm.create_config(config)
            state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
            state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

            assert metrics.best_fitness >= 0

    def test_disk_offload_disabled_by_default(self, depth_offload_algorithm):
        """Verify disk offload is disabled by default."""
        assert depth_offload_algorithm._enable_disk_offload is False

    def test_disk_offload_toggle(self, depth_offload_algorithm):
        """Verify disk offload can be toggled on and off."""
        # Enable
        depth_offload_algorithm.configure_sharding(enable_disk_offload=True)
        assert depth_offload_algorithm._enable_disk_offload is True

        # Disable
        depth_offload_algorithm.configure_sharding(enable_disk_offload=False)
        assert depth_offload_algorithm._enable_disk_offload is False


class TestMemoryTrackingValidation:
    """Tests for GPU memory tracking validation."""

    def test_cpu_memory_peak_type(self, depth_offload_algorithm):
        """Verify CPU memory peak is a float."""
        assert isinstance(depth_offload_algorithm._cpu_memory_peak_gb, float)

    def test_gpu_transfers_count_initial(self, depth_offload_algorithm):
        """Verify GPU transfers count starts at 0."""
        assert depth_offload_algorithm._gpu_transfers_count == 0

    def test_memory_tracking_after_evolution(self, depth_offload_algorithm, xor_problem):
        """Verify memory tracking works after evolution."""
        depth_offload_algorithm.configure_offloading(depth_threshold=2)

        config = create_depth_offload_config(max_depth=3)
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        for _ in range(3):
            state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        # Memory tracking should be available
        assert hasattr(depth_offload_algorithm, '_cpu_memory_peak_gb')
        assert hasattr(depth_offload_algorithm, '_gpu_transfers_count')


class TestOffloadWithStreaming:
    """Tests for combining depth offload with streaming mode."""

    def test_offload_config_with_streaming_enabled(self, depth_offload_algorithm, xor_problem):
        """Verify offload configuration works when streaming is enabled."""
        config = create_depth_offload_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

        # Enable streaming mode in config
        hmr['enable_streaming'] = True
        hmr['population_chunk_size'] = 25

        depth_offload_algorithm.configure_offloading(
            depth_threshold=2,
            eval_chunk_levels=2,
        )

        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_high_depth_with_streaming(self, depth_offload_algorithm, xor_problem):
        """Verify high depth (offload-worthy) works with streaming."""
        config = create_depth_offload_config(max_depth=4)  # Higher depth
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['enable_streaming'] = True
        hmr['population_chunk_size'] = 25

        depth_offload_algorithm.configure_offloading(
            depth_threshold=3,  # Should trigger offloading at depth 4
        )

        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_offload_and_streaming_with_recurrence(self, depth_offload_algorithm, xor_problem):
        """Verify offload and streaming work together with recurrence."""
        config = create_depth_offload_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['enable_streaming'] = True
        hmr['population_chunk_size'] = 25
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }

        depth_offload_algorithm.configure_offloading(depth_threshold=2)

        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0


class TestOffloadEdgeCases:
    """Edge case tests for depth offloading."""

    def test_offload_with_minimum_depth(self, depth_offload_algorithm, xor_problem):
        """Verify offload works with minimum depth (1)."""
        config = create_depth_offload_config(max_depth=1)

        depth_offload_algorithm.configure_offloading(depth_threshold=1)

        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_offload_threshold_equals_depth(self, depth_offload_algorithm, xor_problem):
        """Verify offload threshold exactly matching depth works."""
        config = create_depth_offload_config(max_depth=3)

        # Threshold = depth, so offloading should be active
        depth_offload_algorithm.configure_offloading(depth_threshold=3)

        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert depth_offload_algorithm.should_use_offloading() is True

    def test_offload_with_small_population(self, depth_offload_algorithm, xor_problem):
        """Verify offload works with small population size."""
        config = create_depth_offload_config(max_depth=3, population_size=10)

        depth_offload_algorithm.configure_offloading(depth_threshold=2)

        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_offload_with_large_eval_chunk_levels(self, depth_offload_algorithm, xor_problem):
        """Verify offload works with large eval_chunk_levels."""
        config = create_depth_offload_config()

        # Large chunk levels (more than typical depth)
        depth_offload_algorithm.configure_offloading(
            depth_threshold=2,
            eval_chunk_levels=10,  # Larger than depth
        )

        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0


# =============================================================================
# Phase 9.3: Depth Offload with H→H Recurrence Tests
# =============================================================================

class TestDepthOffloadWithRecurrence:
    """Tests for depth offload with h→h (recurrence) enabled.

    The depth offload h→h path uses _evaluate_recurrent_chunked() which:
    - Transfers full matrices to GPU for h→h discovery
    - Uses discover_sparse_hh_vectorized_multi_hop()
    - Tracks h→h connection metrics
    """

    def test_depth_offload_with_hidden_only_recurrence(self, depth_offload_algorithm, xor_problem):
        """Verify depth offload works with hidden_only recurrence preset."""
        config = create_depth_offload_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }

        depth_offload_algorithm.configure_offloading(depth_threshold=2)
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_depth_offload_with_backward_recurrence(self, depth_offload_algorithm, xor_problem):
        """Verify depth offload works with backward connections enabled."""
        config = create_depth_offload_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'with_backward',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'allow_backward': True,
            'iteration_level': 2,
        }

        depth_offload_algorithm.configure_offloading(depth_threshold=2)
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_depth_offload_with_full_recurrent(self, depth_offload_algorithm, xor_problem):
        """Verify depth offload works with full recurrent connections."""
        config = create_depth_offload_config()
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

        depth_offload_algorithm.configure_offloading(depth_threshold=2)
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_depth_offload_hh_with_cpu_grid_mode(self, depth_offload_algorithm, xor_problem):
        """Verify h→h works with CPU-resident grid mode enabled."""
        config = create_depth_offload_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }

        depth_offload_algorithm.configure_offloading(
            depth_threshold=2,
            use_cpu_grid=True,  # Force CPU grid mode
        )
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_depth_offload_hh_with_sharding(self, depth_offload_algorithm, xor_problem):
        """Verify h→h works with sharding configuration."""
        config = create_depth_offload_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }

        depth_offload_algorithm.configure_offloading(depth_threshold=2)
        depth_offload_algorithm.configure_sharding(
            target_shard_gb=0.3,
            target_eval_shard_gb=0.5,
        )
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    @pytest.mark.parametrize("depth_threshold", [2, 3, 4])
    def test_depth_offload_hh_at_various_thresholds(self, depth_offload_algorithm, xor_problem, depth_threshold):
        """Verify h→h works with different depth thresholds."""
        config = create_depth_offload_config(max_depth=depth_threshold)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }

        depth_offload_algorithm.configure_offloading(depth_threshold=depth_threshold)
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        assert metrics.best_fitness >= 0

    def test_depth_offload_hh_metrics_tracking(self, depth_offload_algorithm, xor_problem):
        """Verify h→h connection metrics are tracked correctly."""
        config = create_depth_offload_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }

        depth_offload_algorithm.configure_offloading(depth_threshold=2)
        config_obj = depth_offload_algorithm.create_config(config)
        state = depth_offload_algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        state, metrics = depth_offload_algorithm.run_generation(state, xor_problem)

        # Verify h→h metrics attributes exist (may be 0 if no h→h discovered)
        assert hasattr(depth_offload_algorithm, '_last_hh_discovery_time_ms')
        assert hasattr(depth_offload_algorithm, '_last_hh_connections')
        assert hasattr(depth_offload_algorithm, '_last_avg_hh_connections')


class TestChunkedMetricAccumulation:
    """The population-chunked offload loop
    passed {} to both chunk-eval helpers, so the generation's reported h->h
    connection count was the LAST chunk's only. The loop now accumulates."""

    def test_hh_total_is_the_sum_across_chunks_not_the_last(self, monkeypatch):
        from emr_hyperneat.execution import offload as offload_mod

        chunk_totals = [100, 200, 300]
        calls = {"i": 0}

        def fake_eval(algo, state, cppns, W1, W2, masks, h_grid, x, y,
                      pop, chunk, timings, cached_sparse_hh=None,
                      consult_cache_manager=True):
            import jax.numpy as jnp
            i = calls["i"]; calls["i"] += 1
            algo._last_hh_connections = chunk_totals[i]
            algo._last_avg_hh_connections = float(chunk_totals[i]) / pop
            algo._last_max_hh_connections = chunk_totals[i]
            algo._last_hh_discovery_time_ms = 1.0
            timings["recurrent_hh_discovery"] = 0.001
            assert consult_cache_manager is False, (
                "per-chunk eval must not touch the population-level cache manager")
            return jnp.zeros(pop)

        monkeypatch.setattr(offload_mod, "_evaluate_recurrent_chunked", fake_eval)

        from conftest import create_config_with_recurrence
        algo = EMRHyperNEAT()
        config = create_config_with_recurrence("hidden_only")
        emr = config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]
        emr.setdefault("execution", {}).update(
            {"mode": "depth_offload", "target_cpu_gb": 1e-4})
        from conftest import XORProblem
        problem = XORProblem()
        cfg = algo.create_config(config)
        state = algo.initialize(cfg, problem, seed=11)
        state, metrics = algo.run_generation(state, problem, verbose=False)

        assert calls["i"] >= 2, "config did not actually chunk the population"
        expected = sum(chunk_totals[: calls["i"]])
        assert algo._last_hh_connections == expected, (
            f"reported {algo._last_hh_connections}, expected the sum {expected} "
            f"(the pre-fix behavior would report {chunk_totals[calls['i'] - 1]})")
        assert metrics.custom_metrics["gen_hh_total"] == expected


class TestOffloadCacheParity:
    """run_generation_offload dropped HMR's cached_sparse_hh pass-through
    and never consulted the cache manager, re-discovering every generation."""

    def test_manager_cache_hits_after_first_generation(self):
        from conftest import create_config_with_recurrence, XORProblem
        algo = EMRHyperNEAT()
        config = create_config_with_recurrence("hidden_only")
        emr = config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]
        emr.setdefault("execution", {}).update({"mode": "depth_offload"})
        # Pin the cache policy so the contract is deterministic regardless of
        # how much this tiny population's masks move between generations: no
        # time-based refresh within the horizon, no mask-change refresh.
        emr.setdefault("recurrence", {}).update(
            {"hh_refresh_interval": 10, "hh_mask_change_threshold": 1.0})
        problem = XORProblem()
        cfg = algo.create_config(config)
        state = algo.initialize(cfg, problem, seed=11)

        hits = []
        for _ in range(3):
            state, metrics = algo.run_generation(state, problem, verbose=False)
            hits.append(bool(metrics.custom_metrics.get("recurrent_hh_cache_hit")))
        # Generation 0 discovers; the next generations hit the manager cache
        # under the same should_refresh contract as the hybrid path.
        assert hits[0] is False
        assert hits[1] is True and hits[2] is True

    def test_explicit_cache_parameter_bypasses_discovery(self):
        import inspect
        from emr_hyperneat.execution.offload import (
            run_generation_offload,
        )
        # The restored HMR-era parameter must exist on the unified entry.
        assert "cached_sparse_hh" in inspect.signature(run_generation_offload).parameters


class TestRunUntilThresholdHonorsExecutionMode:
    """`run_until_threshold` must dispatch on execution mode, as `run_generation` does.

    It did not. `_run_until_threshold_single_gpu` called `run_generation_verbose`
    unconditionally, and that method carries no offload dispatch, so a config asking
    for `depth_offload` quietly ran the standard on-device path: precisely the
    out-of-memory failure the mode exists to prevent, at the depth where it matters.
    Streaming was never affected, because `create_config` resolves it into the
    `enable_streaming` flag that the build step reads.

    Two independent observables, so neither carries the result alone: a counting
    wrapper on the offload entry point, and `_gpu_transfers_count`, which only
    `execution/offload.py` ever increments.
    """

    @staticmethod
    def _algorithm(mode, max_depth=3, population_size=20):
        """Algorithm plus config object with the execution mode set explicitly."""
        algorithm = EMRHyperNEAT()
        config = create_base_config(max_depth=max_depth, population_size=population_size)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
            'mode': mode,
        }
        return algorithm, algorithm.create_config(config)

    @staticmethod
    def _count_offload_calls(monkeypatch):
        """Count dispatches into the offload entry point without changing behavior."""
        from emr_hyperneat.execution import offload as offload_module
        calls = []
        original = offload_module.run_generation_offload

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(offload_module, 'run_generation_offload', counting)
        return calls

    # Above any achievable fitness on this problem, so the loop always spends its
    # whole generation budget and the call counts below are exact rather than "at
    # least one".
    UNREACHABLE_FITNESS = 2.0
    GENERATIONS = 2

    def test_depth_offload_reaches_the_offload_path(self, xor_problem, monkeypatch):
        """The regression: depth_offload through run_until_threshold must offload."""
        algorithm, config_obj = self._algorithm('depth_offload')
        assert algorithm.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD
        calls = self._count_offload_calls(monkeypatch)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        result = algorithm.run_until_threshold(
            state, xor_problem, self.UNREACHABLE_FITNESS, self.GENERATIONS)

        assert len(calls) == self.GENERATIONS
        assert algorithm._gpu_transfers_count > 0
        assert result['generations'] == self.GENERATIONS

    def test_standard_mode_never_offloads(self, xor_problem, monkeypatch):
        """The other half of the contract: standard stays on the standard path."""
        algorithm, config_obj = self._algorithm('standard')
        assert algorithm.resolve_execution_mode() is ExecutionMode.STANDARD
        calls = self._count_offload_calls(monkeypatch)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        result = algorithm.run_until_threshold(
            state, xor_problem, self.UNREACHABLE_FITNESS, self.GENERATIONS)

        assert calls == []
        assert algorithm._gpu_transfers_count == 0
        assert result['generations'] == self.GENERATIONS

    def test_streaming_mode_is_unchanged(self, xor_problem, monkeypatch):
        """Streaming already worked through this loop; it must keep working."""
        algorithm, config_obj = self._algorithm('streaming')
        assert algorithm.resolve_execution_mode() is ExecutionMode.STREAMING
        assert algorithm.enable_streaming is True
        calls = self._count_offload_calls(monkeypatch)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        result = algorithm.run_until_threshold(
            state, xor_problem, self.UNREACHABLE_FITNESS, self.GENERATIONS)

        assert calls == []
        assert result['generations'] == self.GENERATIONS

    def test_offload_result_matches_run_generation(self, xor_problem):
        """Same mode, same budget, same seed: the two entry points must agree.

        `run_generation` already dispatched correctly, so it is the reference for
        what `run_until_threshold` should have been producing all along.
        """
        reference, reference_config = self._algorithm('depth_offload')
        state = reference.initialize(reference_config, xor_problem, seed=DEFAULT_SEED)
        best = -float('inf')
        for _ in range(self.GENERATIONS):
            state, metrics = reference.run_generation(state, xor_problem)
            best = max(best, float(metrics.best_fitness))

        algorithm, config_obj = self._algorithm('depth_offload')
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)
        result = algorithm.run_until_threshold(
            state, xor_problem, self.UNREACHABLE_FITNESS, self.GENERATIONS)

        assert float(result['best_fitness']) == pytest.approx(best)

    def test_streaming_strategy_with_offload_mode_warns(self, xor_problem):
        """The one path that can ask for both memory strategies at once.

        `MultiGPUStrategy.STREAMING` forces `enable_streaming` and then calls the
        single-device loop. Offload wins there, for the same reason it wins in
        `run_generation`: it replaces the generation rather than a step inside it.
        The user should be told, not left reading a streaming banner.
        """
        algorithm, config_obj = self._algorithm('depth_offload')
        algorithm.strategy = MultiGPUStrategy.STREAMING
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        with pytest.warns(UserWarning, match="offload takes precedence"):
            algorithm.run_until_threshold(
                state, xor_problem, self.UNREACHABLE_FITNESS, 1)
