"""The pmap pipeline's h->h connections follow the connection class.

`_full_pipeline_with_hh_core` (FULL_PIPELINE_PARALLEL) takes its hidden-to-hidden connections from
the bounded discovery outside the pmap (`_pmap_pipeline_hh`), which applies `allow_backward`,
`allow_lateral` and `allow_self_loops` as the standard path does, rather than discovering them as
``|w| > band & active & active & not self``, which runs every preset as full recurrence without
self loops. On one CPU device the pipeline runs as a one-shard pmap, so its per-generation
connection count and fitness can be checked against the standard path, at iteration levels 0 and
1: level 0 discovers the direct connections, as level 1 does, on the pmap pipeline too.
"""

import jax
import numpy as np
import pytest

from conftest import (
    EMRHyperNEAT,
    XORProblem,
    assert_trajectories_agree,
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


def _data_parallel_full(preset, level=1):
    cfg = create_config_with_recurrence(preset, max_depth=3, population_size=8, iteration_level=level,
                                        hh_cache_enabled=False)
    algo = EMRHyperNEAT()
    state = algo.initialize(algo.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
    result = algo._run_until_threshold_data_parallel_full(state, XORProblem(), target_fitness=2.0,
                                                          max_generations=1, collect_history=True)
    return int(algo._last_pmap_hh_count), float(result['best_fitness'])


class TestPmapPipelineClassConstraint:
    @pytest.mark.parametrize('level', [0, 1])
    @pytest.mark.parametrize('preset', ['hidden_only', 'with_backward', 'full_recurrent'])
    def test_pipeline_matches_standard_path(self, preset, level):
        """Level 0 discovers the direct connections, as level 1 does, on the standard path and on
        the pmap pipeline alike: a pmap run at level 0 is not feed-forward (no h->h connection,
        the dense evaluation)."""
        std_count, std_fit = _standard(preset, level)
        pmap_count, pmap_best = _data_parallel_full(preset, level)
        # Same population (same seed, generation 0), same discovery rule: same connection count.
        assert pmap_count == std_count, (preset, level, pmap_count, std_count)
        assert abs(pmap_best - float(std_fit.max())) < 1e-4, (preset, level, pmap_best,
                                                              float(std_fit.max()))

    def test_hidden_only_is_not_full_recurrence(self):
        # Without the class constraint every preset would give the full-recurrence count here.
        h_count, _ = _data_parallel_full('hidden_only')
        f_count, _ = _data_parallel_full('full_recurrent')
        assert h_count < f_count


class TestPmapPipelineRefusesWhatItCannotCompute:
    """The pmap pipeline discovers at iteration level 1 only (no multi-hop expansion), so levels
    above 1 are refused on entry and the effective level is recorded. Without the refusal a level-2
    preset would run there as level 1 while the record named level 2: the pmap route's level-2
    fitness vector equaled the standard path's level-1 vector (XOR depth 3, populations 8
    and 30)."""

    def test_level_two_is_refused(self):
        with pytest.raises(ValueError, match='iteration_level'):
            _data_parallel_full('hidden_only', level=2)

    @pytest.mark.parametrize('level', [0, 1])
    def test_levels_zero_and_one_record_the_level_computed(self, level):
        """It records `effective_iteration_level` 1 with the direct connections discovered, at
        level 0 as at level 1."""
        cfg = create_config_with_recurrence('hidden_only', max_depth=3, population_size=8,
                                            iteration_level=level, hh_cache_enabled=False)
        algo = EMRHyperNEAT()
        state = algo.initialize(algo.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
        algo._run_until_threshold_data_parallel_full(state, XORProblem(), target_fitness=2.0,
                                                     max_generations=1, collect_history=True)
        assert algo._extended_metrics.effective_iteration_level == 1
        assert algo._extended_metrics.hidden_to_hidden_connections == algo._last_pmap_hh_count > 0

    def test_level_zero_runs_as_level_one(self):
        """Level 0 reaches the pipeline as level 1: the same connections, and the same fitness (bit
        for bit on a CPU)."""
        (count0, best0), (count1, best1) = (_data_parallel_full('with_backward', level=level)
                                            for level in (0, 1))
        assert count0 == count1 > 0, (count0, count1)
        assert_trajectories_agree({'level 0': [best0], 'level 1': [best1]},
                                  'data_parallel_full: level 0 against level 1')

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

    @pytest.mark.skipif(len(jax.devices()) < 2, reason='needs two or more devices')
    def test_strategy_key_on_several_devices_falls_back_for_an_exhaustive_dataset(self):
        """On two or more devices the key still takes the standard path for XOR, an exhaustive
        dataset (every device would need all four samples), so level 2 is computed there too."""
        self._fallback_computes_level_two()

    def _fallback_computes_level_two(self):
        std_l1, _ = _standard('hidden_only', level=1)
        std_l2, _ = _standard('hidden_only', level=2)
        assert std_l2 != std_l1
        cfg = create_config_with_recurrence('hidden_only', max_depth=3, population_size=8,
                                            iteration_level=2, hh_cache_enabled=False)
        cfg['algorithm_params']['emrhyperneat']['emr_hyperneat'][
            'multi_gpu_strategy'
        ] = 'FULL_PIPELINE_PARALLEL'
        algo = EMRHyperNEAT()
        state = algo.initialize(algo.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
        algo.run_until_threshold(state, XORProblem(), target_fitness=2.0, max_generations=1,
                                 collect_history=True)
        assert int(engine._last_discovery_stats['total_per_genome'].sum()) == std_l2
