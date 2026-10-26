"""Tests for multi-GPU strategies in EMR-HyperNEAT.

Tests cover the multi-GPU strategies and several alias names:
- SINGLE_GPU: Baseline single GPU execution
- FULL_PIPELINE_PARALLEL: Full pipeline on each GPU (recommended for feedforward)
- STREAMING: One device, CPPN queries one level at a time (the streaming execution mode)
- EVAL_ONLY_PARALLEL: Only eval parallelized, h->h discovery on GPU 0 (recommended for h->h when
  the sample count is a multiple of the device count)
- POPULATION_PARALLEL_PROCESS: Population sharding via ProcessPoolExecutor
- POPULATION_PMAP, POSITION_SHARDING, HYBRID: the split routes, SINGLE_GPU's generation with its
  CPPN query and evaluation spread over devices (test_device_split.py, test_multi_device_parity.py)
- CPPN_CHUNKED: Alias of POSITION_SHARDING (STRATEGY_NAME_MAP)

Note: Multi-GPU tests skip gracefully when only 1 GPU is available.
Tests marked with @pytest.mark.skipif(get_num_gpu_devices() < 2) require 2+ GPUs.
GPU detection uses jax.devices() filtered by platform == 'gpu'.
"""

import os

import pytest
import jax
from conftest import (
    EMRHyperNEAT,
    MultiGPUStrategy,
    create_base_config,
    create_config_with_recurrence,
    create_config_with_dynamic_functions,
    create_config_with_aggregation,
    create_config_with_neuromodulation,
    create_config_with_streaming,
    run_quick_evolution,
    run_quick_evolution_multi_gpu,
    assert_no_errors,
    assert_positive_fitness,
    assert_trajectories_agree,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
)


def get_num_devices():
    """Get number of available devices."""
    return len(jax.devices())


def get_gpu_devices():
    """Get list of GPU devices."""
    return [d for d in jax.devices() if d.platform == 'gpu']


def get_num_gpu_devices():
    """Get number of GPU devices."""
    return len(get_gpu_devices())


def has_gpu():
    """Check if at least 1 GPU is available."""
    return get_num_gpu_devices() >= 1


class TestSingleGPUStrategy:
    """Tests for SINGLE_GPU strategy (baseline)."""

    def test_single_gpu_initializes(self, xor_problem):
        """Verify SINGLE_GPU strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.SINGLE_GPU
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_single_gpu_runs_generation(self, xor_problem):
        """Verify SINGLE_GPU strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.SINGLE_GPU
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_single_gpu_with_recurrence(self, xor_problem):
        """Verify SINGLE_GPU works with recurrence enabled."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.SINGLE_GPU
        )
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestSingleGPUPythonLoopStrategy:
    """Tests for SINGLE_GPU_PYTHON_LOOP strategy (debugging mode)."""

    def test_single_gpu_python_loop_initializes(self, xor_problem):
        """Verify SINGLE_GPU_PYTHON_LOOP strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_single_gpu_python_loop_runs_generation(self, xor_problem):
        """Verify SINGLE_GPU_PYTHON_LOOP strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_single_gpu_python_loop_with_recurrence(self, xor_problem):
        """Verify SINGLE_GPU_PYTHON_LOOP works with recurrence enabled."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP
        )
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestStreamingStrategy:
    """Tests for STREAMING strategy."""

    def test_streaming_initializes(self, xor_problem):
        """Verify STREAMING strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.STREAMING
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_streaming_runs_generation(self, xor_problem):
        """Verify STREAMING strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.STREAMING
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_streaming_with_recurrence(self, xor_problem):
        """Verify STREAMING works with recurrence enabled."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.STREAMING
        )
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestFullPipelineParallelStrategy:
    """Tests for FULL_PIPELINE_PARALLEL strategy."""

    def test_full_pipeline_initializes(self, xor_problem):
        """Verify FULL_PIPELINE_PARALLEL strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_full_pipeline_runs_generation(self, xor_problem):
        """Verify FULL_PIPELINE_PARALLEL strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_full_pipeline_on_xor_runs_one_device(self, xor_problem):
        """FULL_PIPELINE_PARALLEL shards the data, so on XOR (four samples) it runs the one-device
        loop, and the result says so. The test asserts the route and the device count: a run that
        took the one-device loop for another reason would pass a check of no errors alone."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_base_config()
        result = run_quick_evolution_multi_gpu(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert algorithm.strategy == MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        assert_no_errors(result)
        assert (result.route, result.devices) == ('SINGLE_GPU', 1)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_full_pipeline_shards_a_large_dataset(self):
        """On a dataset with at least 16 samples per device FULL_PIPELINE_PARALLEL runs on every GPU."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_base_config()
        result = run_quick_evolution_multi_gpu(algorithm, config, ShardableProblem(), generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert (result.route, result.devices) == ('FULL_PIPELINE_PARALLEL', len(jax.devices()))

    def test_full_pipeline_parallel_with_recurrence(self, xor_problem):
        """A FULL_PIPELINE_PARALLEL instance runs h->h recurrence through run_generation,
        which does not read the strategy."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestEvalOnlyParallelStrategy:
    """Tests for EVAL_ONLY_PARALLEL strategy."""

    def test_eval_only_initializes(self, xor_problem):
        """Verify EVAL_ONLY_PARALLEL strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_eval_only_runs_generation(self, xor_problem):
        """Verify EVAL_ONLY_PARALLEL strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_eval_only_multi_device(self, xor_problem):
        """Verify EVAL_ONLY_PARALLEL uses multiple devices when available.

        Uses run_quick_evolution_multi_gpu which calls run_until_threshold()
        to trigger actual multi-GPU routing via pmap.
        """
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL
        )
        config = create_base_config()
        result = run_quick_evolution_multi_gpu(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_eval_only_reports_the_same_hh_count_as_a_single_device(self, xor_problem):
        """Both paths must report the h->h count on the same basis: the population total.

        The single-device path reports `jnp.sum(num_valid)` over the population; reporting
        `num_valid[0]`, genome 0 alone, would understate it by a factor of the population
        size and read as a far sparser substrate, as in the multi-GPU measurements ("8 to
        29 against thousands"). Discovery runs on device 0 from the same masks in both
        paths, so the counts must agree exactly. The other tests in this class assert only
        that fitness is positive, which such a defect leaves untouched.
        """
        config = create_config_with_recurrence('hidden_only')

        single = EMRHyperNEAT(strategy=MultiGPUStrategy.SINGLE_GPU)
        state = single.initialize(single.create_config(config), xor_problem, seed=DEFAULT_SEED)
        single.run_generation(state, xor_problem)
        # run_generation's AlgorithmMetrics carries the count only in custom_metrics; the
        # extended metrics expose it under the name the pmap result reports it as
        single_count = int(single.get_extended_metrics().hidden_to_hidden_connections)
        assert single_count > 0, "no h->h discovered on a single device; nothing to compare"

        parallel = EMRHyperNEAT(strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL)
        state = parallel.initialize(parallel.create_config(config), xor_problem, seed=DEFAULT_SEED)
        result = parallel.run_until_threshold(state, xor_problem, 2.0, 1)
        parallel_count = int(result['sparse_hh_count'])

        assert parallel_count == single_count, (
            f"eval_only_parallel reports {parallel_count} h->h connections where a single "
            f"device reports {single_count}; the two paths are counting on different bases"
        )


class TestPopulationParallelProcessStrategy:
    """Tests for POPULATION_PARALLEL_PROCESS strategy."""

    def test_population_parallel_initializes(self, xor_problem):
        """Verify POPULATION_PARALLEL_PROCESS strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_PROCESS
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_population_parallel_runs_generation(self, xor_problem):
        """Verify POPULATION_PARALLEL_PROCESS strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_PROCESS
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    @pytest.mark.xfail(
        os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '').lower() != 'false',
        reason="ProcessPoolExecutor child processes fail to initialize cuBLAS when "
               "GPU memory is already allocated by parent test process. This is a "
               "known limitation of JAX with ProcessPoolExecutor - see CUDA BLAS "
               "initialization errors. Works in isolation but fails in test suite. "
               "With XLA_PYTHON_CLIENT_PREALLOCATE=false the parent holds no pool and the test "
               "passes, so it is expected to fail only when the parent preallocates.",
        strict=False,
    )
    def test_population_parallel_multi_device(self, xor_problem):
        """Verify POPULATION_PARALLEL_PROCESS shards across devices.

        Uses run_quick_evolution_multi_gpu which calls run_until_threshold()
        to trigger actual multi-GPU routing via ProcessPoolExecutor.
        """
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_PROCESS
        )
        config = create_base_config()
        result = run_quick_evolution_multi_gpu(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_population_parallel_process_with_recurrence(self, xor_problem):
        """A POPULATION_PARALLEL_PROCESS instance runs h->h recurrence through run_generation,
        which does not read the strategy (the process route scores 0 with h->h)."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_PROCESS
        )
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestCPPNChunkedStrategy:
    """Tests for CPPN_CHUNKED, an alias of POSITION_SHARDING (STRATEGY_NAME_MAP)."""

    def test_cppn_chunked_initializes(self, xor_problem):
        """Verify CPPN_CHUNKED strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.CPPN_CHUNKED
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_cppn_chunked_runs_generation(self, xor_problem):
        """Verify CPPN_CHUNKED strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.CPPN_CHUNKED
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_cppn_chunked_runs_the_position_split(self, xor_problem):
        """CPPN_CHUNKED is POSITION_SHARDING (STRATEGY_NAME_MAP): the CPPN query split by grid
        position over every device; on one device, the one-device loop."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.CPPN_CHUNKED
        )
        config = create_base_config()
        result = run_quick_evolution_multi_gpu(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert algorithm.strategy == MultiGPUStrategy.POSITION_SHARDING
        assert_no_errors(result)
        devices = len(jax.devices())
        assert (result.route, result.devices) == (
            ('POSITION_SHARDING', devices) if devices >= 2 else ('SINGLE_GPU', 1))

    def test_cppn_chunked_with_recurrence(self, xor_problem):
        """A CPPN_CHUNKED instance runs h->h recurrence through run_generation, which does not
        read the strategy."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.CPPN_CHUNKED
        )
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestStrategyConsistency:
    """Tests verifying strategy consistency."""

    def test_strategies_produce_valid_fitness(self, xor_problem):
        """Verify all strategies produce valid fitness values."""
        for strategy in [MultiGPUStrategy.SINGLE_GPU, MultiGPUStrategy.STREAMING]:
            algorithm = EMRHyperNEAT(
                strategy=strategy
            )
            config = create_base_config()
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

            assert_no_errors(result, f"Strategy {strategy} failed")
            assert_positive_fitness(result, f"Strategy {strategy} non-positive fitness")

    def test_same_seed_same_result_single_gpu(self, xor_problem):
        """Verify same seed produces same result with SINGLE_GPU."""
        results = []
        for _ in range(2):
            algorithm = EMRHyperNEAT(
                strategy=MultiGPUStrategy.SINGLE_GPU
            )
            config = create_base_config()
            result = run_quick_evolution(
                algorithm, config, xor_problem,
                generations=QUICK_GENERATIONS, seed=42
            )
            results.append(result.best_fitness)

        # Same seed should produce similar results (may have small numerical differences)
        assert abs(results[0] - results[1]) < 0.1, "Same seed should produce similar results"


class TestStrategyWithFeatures:
    """Tests for strategies with various features enabled."""

    @pytest.mark.parametrize("strategy", [MultiGPUStrategy.SINGLE_GPU, MultiGPUStrategy.STREAMING])
    def test_strategy_with_dynamic_functions(self, xor_problem, strategy):
        """Verify strategies work with dynamic functions."""
        algorithm = EMRHyperNEAT(
            strategy=strategy
        )
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Strategy {strategy} with dynamic functions failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("strategy", [MultiGPUStrategy.SINGLE_GPU, MultiGPUStrategy.STREAMING])
    def test_strategy_with_neuromodulation(self, xor_problem, strategy):
        """Verify strategies work with neuromodulation."""
        algorithm = EMRHyperNEAT(
            strategy=strategy
        )
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

        assert_no_errors(result, f"Strategy {strategy} with neuromodulation failed")
        assert_positive_fitness(result)


class TestStrategyStringParsing:
    """Tests for strategy string parsing."""

    def test_strategy_string_single_gpu(self, xor_problem):
        """Verify 'SINGLE_GPU' string is parsed correctly: `multi_gpu_strategy` overrides the
        constructor's STREAMING (`emr_hyperneat.strategy` is refused)."""
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.STREAMING)
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multi_gpu_strategy'] = 'SINGLE_GPU'

        config_obj = algorithm.create_config(config)
        assert algorithm.strategy == MultiGPUStrategy.SINGLE_GPU
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_strategy_string_streaming(self, xor_problem):
        """Verify 'STREAMING' string is parsed correctly from multi_gpu_strategy (`strategy` is
        refused)."""
        algorithm = EMRHyperNEAT()
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multi_gpu_strategy'] = 'STREAMING'

        config_obj = algorithm.create_config(config)
        assert algorithm.strategy == MultiGPUStrategy.STREAMING
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)


class TestComprehensiveStrategies:
    """Slow runs: SINGLE_GPU and STREAMING on XOR, and three strategies for a few generations."""

    @pytest.mark.slow
    @pytest.mark.parametrize("strategy", [MultiGPUStrategy.SINGLE_GPU, MultiGPUStrategy.STREAMING])
    def test_strategy_solves_xor(self, xor_problem, strategy):
        """Verify each strategy can achieve good fitness on XOR."""
        algorithm = EMRHyperNEAT(
            strategy=strategy
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=15)

        assert_no_errors(result)
        assert result.best_fitness > 0.7, f"Strategy {strategy} should achieve >0.7 on XOR"

    @pytest.mark.slow
    def test_all_strategies_no_crash(self, xor_problem):
        """Verify all strategies run without crashing."""
        for strategy in [MultiGPUStrategy.SINGLE_GPU, MultiGPUStrategy.STREAMING,
                        MultiGPUStrategy.FULL_PIPELINE_PARALLEL]:
            algorithm = EMRHyperNEAT(
                strategy=strategy
            )
            config = create_base_config()
            result = run_quick_evolution(algorithm, config, xor_problem, generations=3)

            assert_no_errors(result, f"Strategy {strategy} crashed")


class TestPopulationParallelSequentialStrategy:
    """Tests for POPULATION_PARALLEL_SEQUENTIAL strategy.

    The name is an alias of POPULATION_PARALLEL_PROCESS (STRATEGY_NAME_MAP), which splits the
    population across worker processes, one per device.
    """

    def test_population_parallel_sequential_initializes(self, xor_problem):
        """Verify POPULATION_PARALLEL_SEQUENTIAL strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_population_parallel_sequential_runs_generation(self, xor_problem):
        """Verify POPULATION_PARALLEL_SEQUENTIAL strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_population_parallel_sequential_with_recurrence(self, xor_problem):
        """A POPULATION_PARALLEL_SEQUENTIAL instance runs h->h recurrence through
        run_generation, which does not read the strategy."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL
        )
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_population_parallel_sequential_multi_device(self, xor_problem):
        """POPULATION_PARALLEL_SEQUENTIAL through the constructor on two or more devices runs as the configuration
        key's POPULATION_PARALLEL_PROCESS (STRATEGY_NAME_MAP), not the one-device loop that a name
        without a route would take."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL
        )
        config = create_base_config()
        result = run_quick_evolution_multi_gpu(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert algorithm.strategy == MultiGPUStrategy.POPULATION_PARALLEL_PROCESS
        assert_no_errors(result)
        assert_positive_fitness(result)


class TestPersistentParallelStrategy:
    """Tests for PERSISTENT_PARALLEL strategy.

    The name is an alias of POPULATION_PARALLEL_PROCESS (STRATEGY_NAME_MAP): the persistent
    workers it once named never ran, and the package no longer has them.
    """

    def test_persistent_parallel_initializes(self, xor_problem):
        """Verify PERSISTENT_PARALLEL strategy initializes."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.PERSISTENT_PARALLEL
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_persistent_parallel_runs_generation(self, xor_problem):
        """Verify PERSISTENT_PARALLEL strategy can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.PERSISTENT_PARALLEL
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_persistent_parallel_with_recurrence(self, xor_problem):
        """A PERSISTENT_PARALLEL instance runs h->h recurrence through run_generation, which
        does not read the strategy."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.PERSISTENT_PARALLEL
        )
        config = create_config_with_recurrence('hidden_only')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_persistent_parallel_multi_device(self, xor_problem):
        """PERSISTENT_PARALLEL through the constructor on two or more devices runs as the configuration
        key's POPULATION_PARALLEL_PROCESS (STRATEGY_NAME_MAP), not the one-device loop that a name
        without a route would take."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.PERSISTENT_PARALLEL
        )
        config = create_base_config()
        result = run_quick_evolution_multi_gpu(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert algorithm.strategy == MultiGPUStrategy.POPULATION_PARALLEL_PROCESS
        assert_no_errors(result)
        assert_positive_fitness(result)


class TestStrategyAliases:
    """Tests for strategy alias resolution.

    Verifies that legacy strategy names resolve to their current equivalents.
    """

    def test_baseline_alias_initializes(self, xor_problem):
        """Verify BASELINE alias initializes (should resolve to SINGLE_GPU)."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.BASELINE
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_baseline_alias_runs_generation(self, xor_problem):
        """Verify BASELINE alias can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.BASELINE
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_data_parallel_alias_initializes(self, xor_problem):
        """Verify DATA_PARALLEL alias initializes (should resolve to FULL_PIPELINE_PARALLEL)."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.DATA_PARALLEL
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_multi_gpu_alias_initializes(self, xor_problem):
        """Verify MULTI_GPU alias initializes (should resolve to FULL_PIPELINE_PARALLEL)."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.MULTI_GPU
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_pmap_parallel_alias_initializes(self, xor_problem):
        """Verify PMAP_PARALLEL alias initializes (should resolve to EVAL_ONLY_PARALLEL)."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.PMAP_PARALLEL
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_pipeline_chunked_alias_initializes(self, xor_problem):
        """Verify PIPELINE_CHUNKED alias initializes (should resolve to FULL_PIPELINE_PARALLEL)."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.PIPELINE_CHUNKED
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("alias,expected_type", [
        (MultiGPUStrategy.BASELINE, MultiGPUStrategy),
        (MultiGPUStrategy.DATA_PARALLEL, MultiGPUStrategy),
        (MultiGPUStrategy.MULTI_GPU, MultiGPUStrategy),
        (MultiGPUStrategy.PMAP_PARALLEL, MultiGPUStrategy),
        (MultiGPUStrategy.PIPELINE_CHUNKED, MultiGPUStrategy),
    ])
    def test_alias_enum_membership(self, alias, expected_type):
        """Verify all aliases are valid MultiGPUStrategy enum members."""
        assert isinstance(alias, expected_type)

    def test_cppn_chunked_alias_initializes(self, xor_problem):
        """Verify CPPN_CHUNKED alias initializes (it resolves to POSITION_SHARDING)."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.CPPN_CHUNKED
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_cppn_chunked_alias_runs_generation(self, xor_problem):
        """Verify CPPN_CHUNKED alias can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.CPPN_CHUNKED
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_position_sharding_chunked_alias_initializes(self, xor_problem):
        """Verify POSITION_SHARDING_CHUNKED alias initializes (it resolves to POSITION_SHARDING)."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POSITION_SHARDING_CHUNKED
        )
        config = create_base_config()
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_position_sharding_chunked_alias_runs_generation(self, xor_problem):
        """Verify POSITION_SHARDING_CHUNKED alias can run generations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POSITION_SHARDING_CHUNKED
        )
        config = create_base_config()
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


# Multi-GPU tests: device utilization, consistency, and integration

class TestMultiGPUDeviceUtilization:
    """The route each multi-device strategy takes and the devices it uses, as run_until_threshold
    records them: run_generation never reads the strategy, and a test that only checks that two
    GPUs are visible cannot fail on a one-device run.
    """

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_pmap_full_pipeline_uses_all_devices(self):
        """FULL_PIPELINE_PARALLEL runs the pmap pipeline on every GPU for a dataset it can shard."""
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL)
        result = run_quick_evolution_multi_gpu(algorithm, create_base_config(), ShardableProblem(),
                                               generations=2)

        assert_no_errors(result)
        assert (result.route, result.devices) == ('FULL_PIPELINE_PARALLEL', len(jax.devices()))

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_eval_only_parallel_uses_all_devices(self, xor_problem):
        """EVAL_ONLY_PARALLEL evaluates on every GPU."""
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL)
        result = run_quick_evolution_multi_gpu(algorithm, create_base_config(), xor_problem,
                                               generations=3)

        assert_no_errors(result)
        assert (result.route, result.devices) == ('EVAL_ONLY_PARALLEL', len(jax.devices()))

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    @pytest.mark.parametrize('name', ['POPULATION_PMAP', 'POSITION_SHARDING'])
    def test_split_route_uses_all_devices(self, xor_problem, name):
        """A split route spreads the CPPN query and the evaluation over every GPU."""
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy[name])
        result = run_quick_evolution_multi_gpu(
            algorithm, create_base_config(), xor_problem, generations=2
        )

        assert_no_errors(result)
        assert (result.route, result.devices) == (name, len(jax.devices()))

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 4, reason="Requires 4+ GPUs")
    def test_hybrid_uses_its_grid(self, xor_problem):
        """HYBRID's 2 x 2 grid: the genomes over two devices, the grid positions over two."""
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.HYBRID)
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['parallel'] = {
            'population_devices': 2, 'position_devices': 2}
        result = run_quick_evolution_multi_gpu(algorithm, config, xor_problem, generations=2)

        assert_no_errors(result)
        assert (result.route, result.devices) == ('HYBRID', 4)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    @pytest.mark.xfail(
        os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '').lower() != 'false',
        reason="A worker that fails raises since 0.2.0, and its process may find too little GPU "
               "memory left when the parent test process preallocates (see "
               "test_population_parallel_multi_device). In 0.1.0 this test passed whatever "
               "the workers did: their errors left zeros and raised nothing.",
        strict=False,
    )
    def test_population_parallel_shards_correctly(self, xor_problem):
        """POPULATION_PARALLEL_SEQUENTIAL is POPULATION_PARALLEL_PROCESS, one process per GPU."""
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL)
        result = run_quick_evolution_multi_gpu(
            algorithm, create_base_config(population_size=100), xor_problem, generations=1)

        assert_no_errors(result)
        assert (result.route, result.devices) == ('POPULATION_PARALLEL_PROCESS', len(jax.devices()))


class TestMultiGPUStrategyConsistency:
    """Tests to verify strategies produce consistent and valid results.

    These tests verify that different multi-GPU strategies produce
    similar fitness values for the same problem, indicating correctness.
    """

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_multi_gpu_vs_single_gpu_fitness_consistency(self, xor_problem):
        """Multi-GPU strategies should produce similar fitness to SINGLE_GPU.

        While exact fitness values may differ due to parallelization effects,
        all strategies should be able to make progress on XOR.
        """
        # Run single GPU baseline
        single_alg = EMRHyperNEAT(
            strategy=MultiGPUStrategy.SINGLE_GPU
        )
        config = create_base_config(population_size=100)
        single_result = run_quick_evolution(
            single_alg, config, xor_problem, generations=10, seed=42
        )
        assert_no_errors(single_result)

        # Run multi-GPU strategy
        multi_alg = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        multi_result = run_quick_evolution(
            multi_alg, config, xor_problem, generations=10, seed=42
        )
        assert_no_errors(multi_result)

        # Both should make progress (positive fitness)
        assert single_result.best_fitness > 0
        assert multi_result.best_fitness > 0

        # Fitness should be in similar ballpark (within 2x)
        ratio = max(single_result.best_fitness, multi_result.best_fitness) / \
                max(min(single_result.best_fitness, multi_result.best_fitness), 0.01)
        assert ratio < 3.0, f"Fitness ratio {ratio:.2f} is too different"

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_all_multi_gpu_strategies_produce_valid_fitness(self, xor_problem):
        """All multi-GPU strategies should produce positive fitness on XOR.

        This is a smoke test ensuring no strategy is completely broken.
        """
        multi_gpu_strategies = [
            MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
            MultiGPUStrategy.EVAL_ONLY_PARALLEL,
            MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL,
            MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
        ]

        config = create_base_config()
        results = {}

        for strategy in multi_gpu_strategies:
            algorithm = EMRHyperNEAT(
                strategy=strategy
            )
            result = run_quick_evolution(
                algorithm, config, xor_problem, generations=5, seed=42
            )
            results[strategy.name] = result

            # Each strategy should produce valid results
            assert_no_errors(result, f"Strategy {strategy.name}")
            assert_positive_fitness(result, f"Strategy {strategy.name}")

        # All strategies should have made some progress
        for name, result in results.items():
            assert result.best_fitness > 0, f"{name} had zero fitness"

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_same_seed_reproducibility_multi_gpu(self, xor_problem):
        """Same seed should produce consistent results on multi-GPU.

        Note: Due to parallel execution ordering, exact reproducibility
        is not guaranteed, but results should be qualitatively similar.
        """
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_base_config()

        # Run twice with same seed
        result1 = run_quick_evolution(
            algorithm, config, xor_problem, generations=5, seed=42
        )
        result2 = run_quick_evolution(
            algorithm, config, xor_problem, generations=5, seed=42
        )

        assert_no_errors(result1)
        assert_no_errors(result2)

        # Both runs should make progress
        assert result1.best_fitness > 0
        assert result2.best_fitness > 0


class TestSplitRoutesFollowSingleGPU:
    """The split routes run SINGLE_GPU's generation: on GPUs their history follows SINGLE_GPU's to
    float32 rounding (assert_trajectories_agree); on forced CPU devices it is SINGLE_GPU's bit for
    bit (test_multi_device_parity.py)."""

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    @pytest.mark.parametrize('recurrence', [None, 'hidden_only'])
    def test_split_routes_follow_single_gpu(self, xor_problem, recurrence):
        histories = {}
        for name in ('SINGLE_GPU', 'POPULATION_PMAP', 'POSITION_SHARDING'):
            if recurrence is None:
                config = create_base_config(population_size=22)
            else:
                config = create_config_with_recurrence(
                    recurrence, population_size=22, iteration_level=2, hh_cache_enabled=False
                )
            config['algorithm_params']['emrhyperneat']['neat_selection'] = {'genome_elitism': 2}
            algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy[name])
            result = run_quick_evolution_multi_gpu(
                algorithm, config, xor_problem, generations=5, target_fitness=2.0
            )
            assert_no_errors(result)
            assert result.route == name
            histories[name] = [float(best) for best in result.fitness_history]
        assert_trajectories_agree(
            histories, f'split routes against SINGLE_GPU ({recurrence or "feed-forward"})'
        )


class TestMultiGPUWithFeatures:
    """Tests for multi-GPU strategies combined with various features.

    These tests verify that multi-GPU execution works correctly when
    combined with dynamic functions, neuromodulation, and recurrence.
    """

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_full_pipeline_with_dynamic_functions(self, xor_problem):
        """FULL_PIPELINE_PARALLEL should work with dynamic functions enabled."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_config_with_dynamic_functions(
            mode='cppn_output_4',  # Valid mode: CPPN determines activation function
            recurrence_preset='feedforward',
        )
        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=5
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_eval_only_with_hh_caching(self, xor_problem):
        """An EVAL_ONLY_PARALLEL instance runs 10 generations with the h->h cache on through
        run_generation, which does not read the strategy."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.EVAL_ONLY_PARALLEL
        )
        config = create_config_with_recurrence('hidden_only')

        # Run multiple generations to exercise caching
        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=10
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_population_parallel_with_recurrence(self, xor_problem):
        """A POPULATION_PARALLEL_SEQUENTIAL instance runs recurrence through run_generation,
        which does not read the strategy."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL
        )
        config = create_config_with_recurrence('with_backward')
        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=5
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_full_pipeline_with_aggregation(self, xor_problem):
        """FULL_PIPELINE_PARALLEL should work with aggregation enabled."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_config_with_aggregation(mode='weight_interp')  # Valid mode: weight interpretation
        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=5
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_multi_gpu_with_neuromodulation(self, xor_problem):
        """Multi-GPU strategies should work with neuromodulation enabled."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_config_with_neuromodulation(
            mode='true_neuromodulation_4nt',
            recurrence_preset='hidden_only',
        )
        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=5
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_streaming_with_large_population(self, xor_problem):
        """STREAMING strategy should work with larger populations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.STREAMING
        )
        config = create_config_with_streaming(
            enable_streaming=True,
            population_chunk_size=50,
            population_size=200,
        )
        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=5
        )

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("strategy,recurrence", [
        (MultiGPUStrategy.FULL_PIPELINE_PARALLEL, 'hidden_only'),
        (MultiGPUStrategy.FULL_PIPELINE_PARALLEL, 'with_backward'),
        (MultiGPUStrategy.POPULATION_PARALLEL_PROCESS, 'hidden_only'),
        (MultiGPUStrategy.CPPN_CHUNKED, 'hidden_only'),
    ])
    def test_strategy_with_hh_recurrence(self, xor_problem, strategy, recurrence):
        """Each strategy's instance runs h->h recurrence through run_generation, which does
        not read the strategy."""
        algorithm = EMRHyperNEAT(strategy=strategy)
        config = create_config_with_recurrence(recurrence)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result)


class TestMultiGPUErrorHandling:
    """Tests for error handling in multi-GPU scenarios."""

    def test_graceful_fallback_single_gpu(self, xor_problem):
        """Multi-GPU strategies should work gracefully on single GPU.

        When run on a single GPU, multi-GPU strategies should still
        function correctly, just without the parallelization benefits.
        """
        # This test runs regardless of GPU count to verify fallback
        strategies_to_test = [
            MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
            MultiGPUStrategy.EVAL_ONLY_PARALLEL,
            MultiGPUStrategy.POPULATION_PARALLEL_SEQUENTIAL,
        ]

        config = create_base_config()

        for strategy in strategies_to_test:
            algorithm = EMRHyperNEAT(
                strategy=strategy
            )
            # Should not raise even on single GPU
            try:
                result = run_quick_evolution(
                    algorithm, config, xor_problem, generations=3
                )
                # If we get here, it either worked or gracefully handled the fallback
                if result.error is None:
                    assert result.best_fitness > 0
            except Exception as e:
                # Some strategies may not support fallback, which is acceptable
                # as long as they raise a clear error
                assert "single" in str(e).lower() or "gpu" in str(e).lower() or \
                       "device" in str(e).lower() or isinstance(e, ValueError)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_invalid_population_size_handling(self, xor_problem):
        """Multi-GPU strategies should handle edge case population sizes.

        Test with population sizes that don't divide evenly by GPU count.
        """
        num_gpus = get_num_gpu_devices()

        # Population size that doesn't divide evenly by GPU count
        odd_population = num_gpus * 10 + 1  # e.g., 21 for 2 GPUs

        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        config = create_base_config(population_size=odd_population)

        # Should handle uneven population gracefully
        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=3
        )

        # Either it works or raises a clear error about population size
        if result.error:
            assert "population" in result.error.lower() or "divide" in result.error.lower()
        else:
            assert result.best_fitness > 0

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(get_num_gpu_devices() < 2, reason="Requires 2+ GPUs")
    def test_small_population_multi_gpu(self, xor_problem):
        """Multi-GPU should handle very small populations."""
        algorithm = EMRHyperNEAT(
            strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        )
        # Small population - fewer than typical GPU parallelism
        config = create_base_config(population_size=10)

        result = run_quick_evolution(
            algorithm, config, xor_problem, generations=3
        )

        # Should work even with small populations
        if result.error is None:
            assert result.best_fitness > 0


class TestGPUDeviceDetection:
    """Tests for GPU device detection utilities."""

    def test_get_num_devices_returns_int(self):
        """Verify get_num_devices returns an integer."""
        num = get_num_devices()
        assert isinstance(num, int)
        assert num >= 0

    def test_get_gpu_devices_returns_list(self):
        """Verify get_gpu_devices returns a list."""
        devices = get_gpu_devices()
        assert isinstance(devices, list)

    def test_get_num_gpu_devices_returns_int(self):
        """Verify get_num_gpu_devices returns an integer."""
        num = get_num_gpu_devices()
        assert isinstance(num, int)
        assert num >= 0

    def test_has_gpu_returns_bool(self):
        """Verify has_gpu returns a boolean."""
        result = has_gpu()
        assert isinstance(result, bool)

    def test_gpu_detection_consistency(self):
        """Verify GPU detection functions are consistent."""
        gpu_list = get_gpu_devices()
        gpu_count = get_num_gpu_devices()

        assert len(gpu_list) == gpu_count
        if gpu_count > 0:
            assert has_gpu() is True
        else:
            assert has_gpu() is False


class ShardableProblem:
    """64 samples and not exhaustive: FULL_PIPELINE_PARALLEL shards it on up to four devices (16 samples each)."""
    input_shape = (3,)
    output_shape = (1,)
    jitable = True
    use_bias = True
    is_exhaustive_dataset = False
    fitness_threshold = 0.95

    def __init__(self):
        import numpy as np
        rng = np.random.default_rng(7)
        points = rng.uniform(-1.0, 1.0, size=(64, 2))
        self.inputs = [[float(x), float(y), 1.0] for x, y in points]
        self.targets = [[float(x * y > 0)] for x, y in points]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


def _two_host_devices_env():
    """The environment of a subprocess with two CPU devices. Platform selectors a runner may set (a GPU machine's
    profile can set JAX_PLATFORM_NAME and JAX_BACKEND_TARGET to gpu) are removed: inherited, they make JAX refuse to
    start on the CPU."""
    env = {key: value for key, value in os.environ.items()
           if key not in ('JAX_PLATFORM_NAME', 'JAX_BACKEND_TARGET')}
    env.update(JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=2')
    return env


class TestRouteRecords:
    """run_until_threshold records the route it took, the devices it used and the execution mode it ran; a strategy
    name alone does not say that FULL_PIPELINE_PARALLEL runs one device on an exhaustive dataset."""

    ROUTES_SCRIPT = r'''
import json
import sys
sys.path.insert(0, {tests!r})
import jax
import warnings
from conftest import EMRHyperNEAT, MultiGPUStrategy, XORProblem, create_base_config
from test_multi_gpu_strategies import ShardableProblem
from emr_hyperneat.execution import ExecutionMode
assert len(jax.devices()) == 2, jax.devices()
warnings.simplefilter('ignore', UserWarning)
found = {{}}
for label, key, constructed, problem, mode in (
        ('full_pipeline_xor', 'FULL_PIPELINE_PARALLEL', None, XORProblem(), None),
        ('full_pipeline_64', 'FULL_PIPELINE_PARALLEL', None, ShardableProblem(), None),
        ('constructor_full_pipeline_64', None, 'FULL_PIPELINE_PARALLEL', ShardableProblem(), None),
        ('single', 'SINGLE_GPU', None, XORProblem(), None),
        ('streaming', 'STREAMING', None, XORProblem(), None),
        ('full_pipeline_xor_offload', 'FULL_PIPELINE_PARALLEL', None, XORProblem(), 'depth_offload'),
        ('full_pipeline_64_offload', 'FULL_PIPELINE_PARALLEL', None, ShardableProblem(), 'depth_offload'),
        ('full_pipeline_64_auto', 'FULL_PIPELINE_PARALLEL', None, ShardableProblem(), 'auto')):
    algorithm = EMRHyperNEAT() if constructed is None else EMRHyperNEAT(strategy=MultiGPUStrategy[constructed])
    config = create_base_config(population_size=8)
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    if key is not None:
        hmr['multi_gpu_strategy'] = key
    if mode is not None:
        hmr['execution'] = {{'mode': mode}}
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
    if mode == 'auto':
        # AUTO escalated to depth offload (as on an 11 GB GPU at depth 7, population 1000): the multi-device
        # pipeline still runs on the devices, and says so
        type(algorithm).resolve_execution_mode = lambda self: ExecutionMode.DEPTH_OFFLOAD
    try:
        result = algorithm.run_until_threshold(state, problem, target_fitness=2.0, max_generations=1)
    except ValueError as error:
        found[label] = 'REFUSED' if 'would ignore execution.mode' in str(error) else str(error)
        continue
    found[label] = [result['route'], result['devices'], algorithm.get_num_devices(), result['execution_mode']]
print('ROUTES', json.dumps(found))
'''

    def test_routes_and_devices_on_two_devices(self):
        """On two (host) devices: XOR falls back to one device under FULL_PIPELINE_PARALLEL (honoring a configured
        depth offload there), a 64-sample dataset is sharded over both (an explicit depth offload is refused, an
        escalated AUTO still runs the on-device pipeline and records 'standard'), the constructor's
        FULL_PIPELINE_PARALLEL shards it as the configuration key's does, and SINGLE_GPU and STREAMING use one.
        get_num_devices says the route's count."""
        import json
        import subprocess
        import sys
        tests = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run([sys.executable, '-c', self.ROUTES_SCRIPT.format(tests=tests)],
                                env=_two_host_devices_env(), capture_output=True, text=True, timeout=900, cwd=tests)
        assert result.returncode == 0, result.stderr[-3000:]
        line = [row for row in result.stdout.splitlines() if row.startswith('ROUTES ')][-1]
        assert json.loads(line[len('ROUTES '):]) == {
            'full_pipeline_xor': ['SINGLE_GPU', 1, 2, 'standard'],
            'full_pipeline_64': ['FULL_PIPELINE_PARALLEL', 2, 2, 'standard'],
            'constructor_full_pipeline_64': ['FULL_PIPELINE_PARALLEL', 2, 2, 'standard'],
            'single': ['SINGLE_GPU', 1, 1, 'standard'],
            'streaming': ['STREAMING', 1, 1, 'streaming'],
            'full_pipeline_xor_offload': ['SINGLE_GPU', 1, 2, 'depth_offload'],
            'full_pipeline_64_offload': 'REFUSED',
            'full_pipeline_64_auto': ['FULL_PIPELINE_PARALLEL', 2, 2, 'standard'],
        }

    def test_streaming_route_reports_the_offload_that_ran(self):
        """Depth offload replaces the generation even under the STREAMING route (a warning says so); the record says
        depth_offload, not streaming."""
        import warnings
        algorithm = EMRHyperNEAT()
        config = create_base_config(population_size=8)
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['multi_gpu_strategy'] = 'STREAMING'
        hmr['execution'] = {'mode': 'depth_offload'}
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result = algorithm.run_until_threshold(state, problem, target_fitness=2.0, max_generations=1)
        assert (result['route'], result['devices'], result['execution_mode']) == ('STREAMING', 1, 'depth_offload')
        assert algorithm.generation_execution_mode() == 'depth_offload'

    def test_a_one_device_run_records_its_route(self):
        algorithm = EMRHyperNEAT()
        result = run_quick_evolution_multi_gpu(algorithm, create_base_config(population_size=8), XORProblem(),
                                               generations=1)
        assert_no_errors(result)
        assert (result.route, result.devices) == ('SINGLE_GPU', 1)
        assert algorithm.get_num_devices() == 1


class TestUnroutedStrategy:
    """A strategy name means one route: the constructor maps names as the configuration key does
    (STRATEGY_NAME_MAP), and every name has a route, the split routes and the island model the last
    added. run_until_threshold also refuses an unrouted strategy on two or more devices, which only
    an
    assignment by hand can reach."""

    SCRIPT = r'''
import sys
sys.path.insert(0, {tests!r})
import jax
from conftest import EMRHyperNEAT, MultiGPUStrategy, XORProblem, create_base_config
assert len(jax.devices()) == 2, jax.devices()
algorithm = EMRHyperNEAT()
problem = XORProblem()
state = algorithm.initialize(algorithm.create_config(create_base_config(population_size=8)), problem, seed=42)
algorithm.strategy = MultiGPUStrategy.PIPELINE_CHUNKED
try:
    algorithm.run_until_threshold(state, problem, target_fitness=2.0, max_generations=1)
except ValueError as error:
    print('REFUSED', error)
else:
    print('RAN')
'''

    def test_constructor_names_follow_the_configuration(self):
        from emr_hyperneat.emrhyperneat import STRATEGY_NAME_MAP
        for name, routed in STRATEGY_NAME_MAP.items():
            assert EMRHyperNEAT(strategy=MultiGPUStrategy[name]).strategy == routed, name
            algorithm = EMRHyperNEAT()
            config = create_base_config()
            config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multi_gpu_strategy'] = name
            if routed == MultiGPUStrategy.HYBRID:
                # HYBRID needs its grid (emr_hyperneat.parallel)
                config['algorithm_params']['emrhyperneat']['emr_hyperneat']['parallel'] = {
                    'population_devices': 1, 'position_devices': 1}
            if routed == MultiGPUStrategy.ISLAND_MODEL:
                # The island model needs its islands (emr_hyperneat.islands)
                config['algorithm_params']['emrhyperneat']['emr_hyperneat']['islands'] = {
                    'count': 2, 'migration_interval': 0}
            algorithm.create_config(config)
            assert algorithm.strategy == routed, name
        assert EMRHyperNEAT().strategy == MultiGPUStrategy.SINGLE_GPU
        assert (EMRHyperNEAT(strategy=MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP).strategy
                == MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP)

    def test_every_name_has_a_route(self):
        """Every MultiGPUStrategy member has a route, ISLAND_MODEL and ISLAND_MODEL_V2 included;
        the constructor raises for a name STRATEGY_NAME_MAP lacks."""
        for strategy in MultiGPUStrategy:
            EMRHyperNEAT(strategy=strategy)

    def test_unrouted_strategy_refused_on_two_devices(self):
        import subprocess
        import sys
        tests = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run([sys.executable, '-c', self.SCRIPT.format(tests=tests)],
                                env=_two_host_devices_env(), capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr[-2000:]
        assert 'REFUSED' in result.stdout and 'has no route in run_until_threshold' in result.stdout

    ROUTES_SCRIPT = r'''
import json
import sys
sys.path.insert(0, {tests!r})
import jax
from conftest import EMRHyperNEAT, MultiGPUStrategy, XORProblem, create_base_config
import warnings
from test_multi_gpu_strategies import ShardableProblem
from emr_hyperneat.execution import ExecutionMode
assert len(jax.devices()) == 2, jax.devices()
warnings.simplefilter('ignore', UserWarning)
found = {{}}
for label, strategy, problem, mode in (
        ('full_pipeline_xor', 'FULL_PIPELINE_PARALLEL', XORProblem(), None),
        ('full_pipeline_64', 'FULL_PIPELINE_PARALLEL', ShardableProblem(), None),
        ('single', 'SINGLE_GPU', XORProblem(), None),
        ('streaming', 'STREAMING', XORProblem(), None),
        ('population_split', 'POPULATION_PMAP', XORProblem(), None),
        ('position_split', 'CPPN_CHUNKED', XORProblem(), None),
        ('position_split_offload', 'POSITION_SHARDING', XORProblem(), 'depth_offload'),
        ('full_pipeline_xor_offload', 'FULL_PIPELINE_PARALLEL', XORProblem(), 'depth_offload'),
        ('full_pipeline_64_offload', 'FULL_PIPELINE_PARALLEL', ShardableProblem(), 'depth_offload'),
        ('full_pipeline_64_auto', 'FULL_PIPELINE_PARALLEL', ShardableProblem(), 'auto')):
    algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy[strategy])
    config = create_base_config(population_size=8)
    if mode is not None:
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {{'mode': mode}}
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
    if mode == 'auto':
        # AUTO escalated to depth offload (as on an 11 GB GPU at depth 7, population 1000): the multi-device
        # pipeline still runs on the devices, and says so
        type(algorithm).resolve_execution_mode = lambda self: ExecutionMode.DEPTH_OFFLOAD
    try:
        result = algorithm.run_until_threshold(state, problem, target_fitness=2.0, max_generations=1)
    except ValueError as error:
        found[label] = 'REFUSED' if 'would ignore execution.mode' in str(error) else str(error)
        continue
    found[label] = [result['route'], result['devices'], algorithm.get_num_devices(), result['execution_mode']]
print('ROUTES', json.dumps(found))
'''

    def test_routes_and_devices_on_two_devices(self):
        """On two (host) devices run_until_threshold records the route it took, the devices it used
        and the mode it ran: XOR falls back to one device under FULL_PIPELINE_PARALLEL (honoring a
        configured depth offload there), a 64-sample dataset is sharded over both (an explicit depth
        offload is refused, an escalated AUTO still runs the on-device pipeline and records
        'standard'), SINGLE_GPU and STREAMING use one, and the split routes both (CPPN_CHUNKED is
        POSITION_SHARDING), refusing an explicit depth offload. get_num_devices says the route's
        count."""
        import json
        import subprocess
        import sys
        env = _two_host_devices_env()
        tests = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run([sys.executable, '-c', self.ROUTES_SCRIPT.format(tests=tests)],
                                env=env, capture_output=True, text=True, timeout=900, cwd=tests)
        assert result.returncode == 0, result.stderr[-3000:]
        line = [row for row in result.stdout.splitlines() if row.startswith('ROUTES ')][-1]
        assert json.loads(line[len('ROUTES '):]) == {
            'full_pipeline_xor': ['SINGLE_GPU', 1, 2, 'standard'],
            'full_pipeline_64': ['FULL_PIPELINE_PARALLEL', 2, 2, 'standard'],
            'single': ['SINGLE_GPU', 1, 1, 'standard'],
            'streaming': ['STREAMING', 1, 1, 'streaming'],
            'population_split': ['POPULATION_PMAP', 2, 2, 'standard'],
            'position_split': ['POSITION_SHARDING', 2, 2, 'standard'],
            'position_split_offload': 'REFUSED',
            'full_pipeline_xor_offload': ['SINGLE_GPU', 1, 2, 'depth_offload'],
            'full_pipeline_64_offload': 'REFUSED',
            'full_pipeline_64_auto': ['FULL_PIPELINE_PARALLEL', 2, 2, 'standard'],
        }

    def test_streaming_route_reports_the_offload_that_ran(self):
        """Depth offload replaces the generation even under the STREAMING route (a warning says so);
        the record says depth_offload, not streaming."""
        import warnings
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.STREAMING)
        config = create_base_config(population_size=8)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
            'mode': 'depth_offload'
        }
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result = algorithm.run_until_threshold(
                state, problem, target_fitness=2.0, max_generations=1
            )
        assert (result['route'], result['devices'], result['execution_mode']) == (
            'STREAMING',
            1,
            'depth_offload',
        )
        assert algorithm.generation_execution_mode() == 'depth_offload'

    @pytest.mark.parametrize('key', ['position_config', 'island_config', 'hybrid_config', 'pmap_config'])
    def test_sharding_configurations_are_refused(self, key):
        """They configure the base class's own implementations of these strategies, which this class
        does not run (its split routes read emr_hyperneat.parallel), so they raise rather than being
        kept and never read."""
        with pytest.raises(ValueError, match=f'{key} configure base-class strategies'):
            EMRHyperNEAT(**{key: object()})

    @pytest.mark.skipif(len(jax.devices()) >= 2, reason='the one-device fallback needs one device')
    def test_one_device_runs_and_says_so(self, capsys):
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.DATA_PARALLEL)
        assert algorithm.strategy == MultiGPUStrategy.FULL_PIPELINE_PARALLEL
        problem = XORProblem()
        state = algorithm.initialize(
            algorithm.create_config(create_base_config(population_size=8)), problem, seed=DEFAULT_SEED)
        algorithm.run_until_threshold(state, problem, target_fitness=2.0, max_generations=1)
        assert 'FULL_PIPELINE_PARALLEL requested but only 1 GPU available' in capsys.readouterr().out


class TestFixedHiddenToHiddenRunnerIsRefused:
    """The fixed h->h runner would discover each population slot's hidden-to-hidden connections
    once, from the generation-0 population, so after the first reproduction a genome would be
    scored with the connections discovered for another genome. The runner and the argument that
    configures it are refused; recurrence is the configuration's recurrence block."""

    def test_recurrence_config_is_refused(self):
        from emr_hyperneat.emrhyperneat_base import EMRHyperNEATMultiGPU, RecurrenceConfig
        with pytest.raises(ValueError, match="recurrence_config configured only "
                                             "run_until_threshold_with_fixed_hh"):
            EMRHyperNEATMultiGPU(recurrence_config=RecurrenceConfig())
        EMRHyperNEATMultiGPU()

    def test_the_fixed_runner_is_refused(self):
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        config = create_config_with_recurrence('hidden_only', population_size=8)
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
        with pytest.raises(NotImplementedError,
                           match="run_until_threshold_with_fixed_hh is refused"):
            algorithm.run_until_threshold_with_fixed_hh(state, problem, target_fitness=0.9,
                                                        max_generations=2)


class TestBaseOnlyRunnersAreRefused:
    """The eleven runners the base class had, which skipped EMRHyperNEAT's evaluation and routes
    (run_until_threshold_data_parallel also weighted a padded dataset's last sample several times):
    each name is refused on EMRHyperNEAT, and the base class no longer has them."""

    @pytest.mark.parametrize('name', [
        'run_until_threshold_multi_gpu', 'run_until_threshold_dual_gpu_eval',
        'run_until_threshold_population_parallel', 'run_until_threshold_periodic_sync',
        'run_until_threshold_overlapped', 'run_until_threshold_periodic_overlapped',
        'run_until_threshold_gpu_resident_multi_gpu', 'run_until_threshold_gpu_resident_shardmap',
        'run_until_threshold_batched_multi_gpu', 'run_until_threshold_data_parallel',
        'run_until_threshold_streaming'])
    def test_refused(self, name):
        from emr_hyperneat.emrhyperneat_base import EMRHyperNEATMultiGPU
        with pytest.raises(NotImplementedError, match=f"{name} is refused on EMRHyperNEAT"):
            getattr(EMRHyperNEAT(), name)(None, None, target_fitness=0.9, max_generations=1)
        assert name in vars(EMRHyperNEAT) and not hasattr(EMRHyperNEATMultiGPU, name)
