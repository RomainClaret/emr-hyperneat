"""The pmap pipelines' inline h->h discovery applies the connection class.

In 0.1.0, `_full_pipeline_with_hh_core` (FULL_PIPELINE_PARALLEL, data- and population-parallel) and
`_single_genome_pipeline_with_hh` (the sequential per-genome variant) discovered hidden-to-hidden
connections as ``|w| > band & active & active & not self`` with no `allow_backward`, `allow_lateral`
or `allow_self_loops`, so every preset ran as full recurrence without self loops there. They now take
the three flags as static arguments and apply `_hh_class_mask_rows`, the rule of
`_get_constraint_mask_fn`. On one CPU device the population-parallel path runs as a one-shard pmap,
so its per-generation connection count and fitness can be checked against the standard path.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (
    EMRHyperNEAT,
    XORProblem,
    create_config_with_recurrence,
    DEFAULT_SEED,
)
from emr_hyperneat import emrhyperneat as engine


def _standard(preset, level=1):
    cfg = create_config_with_recurrence(preset, max_depth=3, population_size=8, iteration_level=level,
                                        hh_cache_enabled=False)
    algo = EMRHyperNEAT()
    state = algo.initialize(algo.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
    state, _m = algo.run_generation(state, XORProblem(), verbose=False)
    return int(engine._last_discovery_stats['total_per_genome'].sum()), np.asarray(algo._cached_fitnesses)


def _pmap_pop_parallel(preset, level=1):
    cfg = create_config_with_recurrence(preset, max_depth=3, population_size=8, iteration_level=level,
                                        hh_cache_enabled=False)
    algo = EMRHyperNEAT()
    state = algo.initialize(algo.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
    result = algo._run_until_threshold_pop_parallel(state, XORProblem(), target_fitness=2.0,
                                                    max_generations=1, collect_history=True)
    return int(algo._last_pmap_hh_count), float(result['best_fitness'])


class TestClassMaskRows:
    def test_rules(self):
        y = jnp.asarray([0.0, 0.5, 1.0, 1.0])
        idx = jnp.arange(4)
        fwd = np.asarray(engine._hh_class_mask_rows(y, idx, idx, False, False, False))
        assert fwd[0, 1] and fwd[1, 2] and not fwd[2, 1] and not fwd[2, 3] and not fwd[0, 0]
        lat = np.asarray(engine._hh_class_mask_rows(y, idx, idx, False, True, False))
        assert lat[2, 3] and not lat[3, 3] and not lat[2, 0]
        full = np.asarray(engine._hh_class_mask_rows(y, idx, idx, True, True, True))
        assert full.all()
        back = np.asarray(engine._hh_class_mask_rows(y, idx, idx, True, False, False))
        assert back[2, 0] and not back[2, 3] and not back[1, 1]

    def test_row_subset_matches_full_matrix(self):
        y = jnp.asarray([0.0, 0.5, 0.5, 1.0])
        idx = jnp.arange(4)
        full = np.asarray(engine._hh_class_mask_rows(y, idx, idx, False, True, False))
        rows = np.asarray(engine._hh_class_mask_rows(y, jnp.asarray([1, 3]), idx, False, True, False))
        assert np.array_equal(rows, full[[1, 3]])


class TestPmapPipelineClassConstraint:
    @pytest.mark.parametrize('preset', ['hidden_only', 'with_backward', 'full_recurrent'])
    def test_pop_parallel_matches_standard_path_at_level_one(self, preset):
        std_count, std_fit = _standard(preset)
        pmap_count, pmap_best = _pmap_pop_parallel(preset)
        # Same population (same seed, generation 0), same discovery rule: same connection count.
        assert pmap_count == std_count, (preset, pmap_count, std_count)
        assert abs(pmap_best - float(std_fit.max())) < 1e-4, (preset, pmap_best, float(std_fit.max()))

    def test_hidden_only_is_not_full_recurrence(self):
        # Before the fix every preset gave the full-recurrence count here.
        h_count, _ = _pmap_pop_parallel('hidden_only')
        f_count, _ = _pmap_pop_parallel('full_recurrent')
        assert h_count < f_count


def _data_parallel_full(preset, level=1):
    cfg = create_config_with_recurrence(preset, max_depth=3, population_size=8, iteration_level=level,
                                        hh_cache_enabled=False)
    algo = EMRHyperNEAT()
    state = algo.initialize(algo.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
    result = algo._run_until_threshold_data_parallel_full(state, XORProblem(), target_fitness=2.0,
                                                          max_generations=1, collect_history=True)
    return int(algo._last_pmap_hh_count), float(result['best_fitness'])


class TestPmapPipelineRefusesWhatItCannotCompute:
    """The pmap pipelines discover at iteration level 1 only (no multi-hop expansion). In 0.1.0
    a level-2 preset ran there as level 1 while the record named level 2: the pmap
    route's level-2 fitness vector equaled the standard path's level-1 vector (XOR depth 3,
    populations 8 and 30). Levels above 1 are refused on entry; the effective level is recorded."""

    @pytest.mark.parametrize('runner', ['pop_parallel', 'data_parallel_full'])
    def test_level_two_is_refused(self, runner):
        run = _pmap_pop_parallel if runner == 'pop_parallel' else _data_parallel_full
        with pytest.raises(ValueError, match='iteration_level'):
            run('hidden_only', level=2)

    def test_level_one_runs_and_records_the_effective_level(self):
        cfg = create_config_with_recurrence('hidden_only', max_depth=3, population_size=8,
                                            iteration_level=1, hh_cache_enabled=False)
        algo = EMRHyperNEAT()
        state = algo.initialize(algo.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
        algo._run_until_threshold_pop_parallel(state, XORProblem(), target_fitness=2.0,
                                               max_generations=1, collect_history=True)
        assert algo._extended_metrics.effective_iteration_level == 1

    def test_strategy_key_on_one_device_falls_back_to_the_standard_path_at_level_two(self):
        """With one device the FULL_PIPELINE_PARALLEL key takes the standard path, where level 2
        is computed: the refusal must not fire there, and the count is the standard level-2 one."""
        std_l1, _ = _standard('hidden_only', level=1)
        std_l2, _ = _standard('hidden_only', level=2)
        assert std_l2 != std_l1
        cfg = create_config_with_recurrence('hidden_only', max_depth=3, population_size=8,
                                            iteration_level=2, hh_cache_enabled=False)
        cfg['algorithm_params']['emrhyperneat']['emr_hyperneat']['multi_gpu_strategy'] = 'FULL_PIPELINE_PARALLEL'
        algo = EMRHyperNEAT()
        state = algo.initialize(algo.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
        algo.run_until_threshold(state, XORProblem(), target_fitness=2.0, max_generations=1,
                                 collect_history=True)
        assert int(engine._last_discovery_stats['total_per_genome'].sum()) == std_l2
