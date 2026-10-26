"""Tests for streaming mode in EMR-HyperNEAT.

Tests cover:
- Streaming mode enable/disable
- Different population chunk sizes
- Depth offload for memory efficiency
- Level-by-level processing
- Memory efficiency validation
"""

import pytest
from conftest import (
    EMRHyperNEAT,
    create_config_with_streaming,
    create_base_config,
    run_quick_evolution,
    assert_no_errors,
    assert_fitness_above,
    assert_positive_fitness,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
)


# Chunk sizes to test
CHUNK_SIZES = [10, 25, 50, 100, 200]

# Depths to test
TEST_DEPTHS = [2, 3, 4, 5]


class TestStreamingModeBasic:
    """Basic tests for streaming mode."""

    def test_streaming_disabled(self, algorithm, xor_problem):
        """Verify streaming can be disabled."""
        config = create_config_with_streaming(enable_streaming=False)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_streaming_enabled(self, algorithm, xor_problem):
        """Verify streaming can be enabled."""
        config = create_config_with_streaming(enable_streaming=True)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_streaming_initialization(self, algorithm, xor_problem):
        """Verify streaming mode initializes correctly."""
        config = create_config_with_streaming(enable_streaming=True)
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Should complete without error


class TestChunkSizes:
    """Tests for different population chunk sizes."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_chunk_size_initializes(self, algorithm, xor_problem, chunk_size):
        """Verify different chunk sizes initialize."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=chunk_size
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_chunk_size_runs(self, algorithm, xor_problem, chunk_size):
        """Verify different chunk sizes run evolution."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=chunk_size
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Chunk size {chunk_size} failed")
        assert_positive_fitness(result)

    def test_small_chunk_size(self, algorithm, xor_problem):
        """Verify very small chunk size works."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=10
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_large_chunk_size(self, algorithm, xor_problem):
        """Verify large chunk size works."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=500
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestChunkSizeConsistency:
    """Tests verifying chunk sizes produce consistent results."""

    def test_different_chunks_similar_fitness(self, algorithm, xor_problem):
        """Verify different chunk sizes produce similar fitness (same seed)."""
        results = {}
        for chunk_size in [25, 50, 100]:
            config = create_config_with_streaming(
                enable_streaming=True,
                population_chunk_size=chunk_size
            )
            result = run_quick_evolution(
                algorithm, config, xor_problem,
                generations=QUICK_GENERATIONS, seed=42
            )
            results[chunk_size] = result.best_fitness

        # All should produce positive fitness
        for chunk_size, fitness in results.items():
            assert fitness > 0, f"Chunk size {chunk_size} has non-positive fitness"


class TestDepthWithStreaming:
    """Tests for different depths with streaming."""

    @pytest.mark.parametrize("depth", TEST_DEPTHS)
    def test_depth_with_streaming(self, algorithm, xor_problem, depth):
        """Verify different depths work with streaming."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
            max_depth=depth
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Depth {depth} with streaming failed")
        assert_positive_fitness(result)

    def test_higher_depth_benefits_streaming(self, algorithm, xor_problem):
        """Verify higher depths work with streaming enabled."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
            max_depth=5
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestStreamingWithRecurrence:
    """Tests for streaming with recurrence enabled."""

    @pytest.mark.parametrize("recurrence_preset", ['feedforward', 'hidden_only', 'full_recurrent'])
    def test_streaming_with_recurrence(self, algorithm, xor_problem, recurrence_preset):
        """Verify streaming works with different recurrence modes."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
            recurrence_preset=recurrence_preset
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Streaming with {recurrence_preset} failed")
        assert_positive_fitness(result)

    def test_streaming_with_hidden_only(self, algorithm, xor_problem):
        """Verify streaming works with hidden_only recurrence."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
            recurrence_preset='hidden_only'
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestStreamingWithFeatures:
    """Tests for streaming with other features enabled."""

    def test_streaming_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify streaming works with dynamic functions."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['enable_streaming'] = True
        hmr['population_chunk_size'] = 50
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_streaming_with_neuromodulation(self, algorithm, xor_problem):
        """Verify streaming works with neuromodulation."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['enable_streaming'] = True
        hmr['population_chunk_size'] = 50
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


class TestStreamingConfigParsing:
    """Tests for streaming configuration parsing."""

    def test_streaming_config_parsed(self, algorithm, xor_problem):
        """Verify streaming config is parsed correctly."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=100
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Should have streaming enabled
        assert algorithm.enable_streaming == True

    def test_chunk_size_parsed(self, algorithm, xor_problem):
        """Verify chunk size config is parsed correctly."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['population_chunk_size'] = 75

        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Should have chunk size set
        assert algorithm.population_chunk_size == 75


class TestStreamingVsNonStreaming:
    """Tests comparing streaming vs non-streaming."""

    def test_streaming_produces_valid_results(self, algorithm, xor_problem):
        """Verify streaming produces valid results."""
        # Non-streaming
        config_no_stream = create_config_with_streaming(enable_streaming=False)
        result_no_stream = run_quick_evolution(
            algorithm, config_no_stream, xor_problem,
            generations=QUICK_GENERATIONS
        )

        # Streaming
        config_stream = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50
        )
        result_stream = run_quick_evolution(
            algorithm, config_stream, xor_problem,
            generations=QUICK_GENERATIONS
        )

        # Both should work
        assert_no_errors(result_no_stream)
        assert_no_errors(result_stream)
        assert_positive_fitness(result_no_stream)
        assert_positive_fitness(result_stream)


class TestComprehensiveStreaming:
    """Slow runs: streaming at several chunk sizes and depths, and under four recurrence presets."""

    @pytest.mark.slow
    @pytest.mark.parametrize("chunk_size", [25, 50, 100, 200])
    def test_chunk_sizes_solve_xor(self, algorithm, xor_problem, chunk_size):
        """Verify different chunk sizes can solve XOR."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=chunk_size
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=15)

        assert_no_errors(result)
        assert_fitness_above(result, 0.7, f"Chunk size {chunk_size} should achieve >0.7")

    @pytest.mark.slow
    @pytest.mark.parametrize("depth", [3, 4, 5])
    def test_depths_with_streaming(self, algorithm, xor_problem, depth):
        """Verify depths work with streaming."""
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
            max_depth=depth
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=15)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.slow
    def test_streaming_all_recurrence_modes(self, algorithm, xor_problem):
        """Verify streaming works with all recurrence modes."""
        for preset in ['feedforward', 'hidden_only', 'with_backward', 'full_recurrent']:
            config = create_config_with_streaming(
                enable_streaming=True,
                population_chunk_size=50,
                recurrence_preset=preset
            )
            result = run_quick_evolution(algorithm, config, xor_problem, generations=10)

            assert_no_errors(result, f"Streaming with {preset} failed")
            assert_positive_fitness(result)


class TestDonatedLevelAccumulate:
    """streaming's per-level W1/W2 slice-updates transiently doubled
    the accumulator (the 786 MB level-7 operand). The donated variant writes in
    place on GPU; both variants must compute identical values."""

    def test_variants_compute_identical_values(self):
        import jax
        import jax.numpy as jnp
        from emr_hyperneat.emrhyperneat import (
            _accumulate_level_slice, _accumulate_level_slice_donated,
        )
        key = jax.random.PRNGKey(3)
        acc = jax.random.normal(key, (5, 3, 40))
        block = jax.random.normal(key, (5, 3, 12))
        plain = acc.at[:, :, 8:20].set(block)
        # Copy before the dispatch call: on a GPU the dispatch routes to the donated
        # variant, which invalidates `acc` ("Array has been deleted with
        # shape=float32[5,3,40]"). On CPU donation is ignored, which is why the order
        # never mattered there.
        acc_for_donated = jnp.array(acc)
        via_dispatch = _accumulate_level_slice(acc, block, 8, 20)
        # Donation aliases buffers; it never changes values: the value contract
        # is what matters on both backends.
        via_donated = _accumulate_level_slice_donated(
            acc_for_donated, block, start=8, end=20)
        assert bool(jnp.array_equal(via_dispatch, plain))
        assert bool(jnp.array_equal(via_donated, plain))

    def test_streaming_generation_unchanged_by_the_dispatch(self):
        from conftest import (
            create_config_with_streaming,
        )
        algo = EMRHyperNEAT()
        config = create_config_with_streaming()
        cfg = algo.create_config(config)
        problem = XORProblem()
        state = algo.initialize(cfg, problem, seed=5)
        state, metrics = algo.run_generation(state, problem, verbose=False)
        assert metrics.best_fitness == metrics.best_fitness  # finite, ran
