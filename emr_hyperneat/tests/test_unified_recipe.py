"""HMR's unified model (UNI) as an EMR recipe.

UNI (HMR's unified model) discovered direct hidden-to-hidden connections and settled them by HMR's
older rule, with a cache refreshed on a timer. Nothing in it needs a new model: EMR runs it with

emr_hyperneat: weight_threshold = band_threshold, variance_rule 'legacy', extra_randkey_split true
recurrence: iteration_level 1, use_dense_discovery true, settling_rule 'hmr', hh_cache_trigger
'time' (with the cache on, as UNI's default was)

and, for UNI's default block (allow_hidden_to_hidden alone, enabled false), enabled true with
allow_lateral: UNI's connection classes there are forward and lateral pairs without self-loops, and
EMR refuses enabled false with hidden-to-hidden connections on. Three UNI defects are not
reproduced: padded active indices entered discovery as position 0, discovery ran over the
population's union of active positions instead of each genome's own, and a per-chunk cap truncated
the pairs. The recipe equals UNI where none of them acts (every position active, 84 positions, so
one chunk below the cap), checked against UNI as recorded in the research repository
(hmr_reference.py). Two additions make the recipe possible: the weight_threshold key (UNI pruned W1
and W2 at band_threshold, EMR at a fixed 0.1) and the 'time' trigger. initialize starts a new h->h
cache (0.1.0, like UNI, scored a reused object's first generations with the last run's connections).
"""

import copy
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (GPU_TRAJECTORY_RTOL, EMRHyperNEAT, XORProblem, create_base_config,
                      create_config_with_recurrence, exact_backend)
from emr_hyperneat.emrhyperneat import (
    WEIGHT_THRESHOLD,
    EMRConfig,
    EMRRecurrenceMetrics,
    HHCacheManager,
)
from hmr_reference import hmr_value, live  # type: ignore[import-not-found]

DEPTH, POP, SEED = 2, 20, 42


class _TwoInputXOR:
    """XOR on two inputs, no bias, at the configuration's coordinates."""
    jitable = True
    input_shape = (2,)
    output_shape = (1,)
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]
    input_coords = [[-1.0, 0.0], [1.0, 0.0]]
    output_coords = [[0.0, 1.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


def _uni_config(recurrence, variance_threshold=0.0, population=POP, **settings):
    """A UNI configuration; variance 0 makes every position active, where UNI's three defects do not
    act."""
    return {
        'algorithm_params': {
            'hmrhyperneat': {
                'population_size': population,
                'substrate': {
                    'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                    'output_coords': [[0.0, 1.0]],
                },
                'hmr_hyperneat': {
                    'initial_depth': 0,
                    'max_depth': DEPTH,
                    'variance_threshold': variance_threshold,
                    'recurrence': dict(recurrence),
                    **settings,
                },
                'neat_species': {'compatibility_threshold': 2.5, 'max_stagnation': 40},
            }
        }
    }


def uni_recipe(config, emr_overrides=None, recurrence_overrides=None):
    """The EMR configuration of a UNI configuration (the recipe in the module docstring)."""
    params = copy.deepcopy(config['algorithm_params'])
    block = params.pop('hmrhyperneat')
    emr = block.pop('hmr_hyperneat')
    recurrence = emr['recurrence']
    if recurrence.get('allow_hidden_to_hidden') and not recurrence.get('enabled', False):
        recurrence.update(enabled=True, allow_lateral=True)
    cache_on = recurrence.get('hh_cache_enabled', True)
    recurrence.update(
        iteration_level=1, use_dense_discovery=True, settling_rule='hmr', hh_cache_enabled=cache_on
    )
    if cache_on:
        recurrence['hh_cache_trigger'] = 'time'
    recurrence.update(recurrence_overrides or {})
    emr.update(variance_rule='legacy', extra_randkey_split=True,
               weight_threshold=emr.get('band_threshold', 0.3))
    emr.update(emr_overrides or {})
    block['emr_hyperneat'] = emr
    params['emrhyperneat'] = block
    return {**config, 'algorithm_params': params}


def _run_uni(config, generations):
    """UNI's fitness vector, h->h count and cache status of each generation (hmr_reference.py)."""
    args = (config, _TwoInputXOR(), SEED, generations)
    return hmr_value('unified_recipe.run_uni', lambda: live('uni_run', *args), inputs=args)


def _generations(algorithm, state, problem, generations):
    fitnesses, records = [], []
    for _ in range(generations):
        state, metrics = algorithm.run_generation(state, problem)
        fitnesses.append(np.asarray(algorithm._cached_fitnesses))
        records.append((metrics.custom_metrics['sparse_hh_count'],
                        metrics.custom_metrics['step_timings']['hh_cache_status']))
    return fitnesses, records


def _run_emr(config, generations, problem=None):
    algorithm = EMRHyperNEAT()
    problem = problem or _TwoInputXOR()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=SEED)
    return _generations(algorithm, state, problem, generations)


def _assert_same(ours, theirs):
    for mine, uni in zip(ours, theirs):
        if exact_backend():
            np.testing.assert_array_equal(mine, uni)
        else:
            np.testing.assert_allclose(mine, uni, rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


H_TO_H = {'allow_hidden_to_hidden': True}


class TestRecipeAgainstLiveUni:
    """The recipe against UNI itself, same seed: every generation's fitness vector, h->h count and
    cache status."""

    def test_without_the_cache(self):
        config = _uni_config({**H_TO_H, 'hh_cache_enabled': False})
        theirs, their_records = _run_uni(config, 4)
        ours, our_records = _run_emr(uni_recipe(config), 4)
        _assert_same(ours, theirs)
        assert our_records == their_records

    def test_with_the_timed_cache(self):
        config = _uni_config({**H_TO_H, 'hh_cache_enabled': True, 'hh_refresh_interval': 2})
        theirs, their_records = _run_uni(config, 5)
        ours, our_records = _run_emr(uni_recipe(config), 5)
        _assert_same(ours, theirs)
        assert our_records == their_records
        assert [status for _, status in our_records] == [
            'refresh',
            'cached',
            'refresh',
            'cached',
            'refresh',
        ]

    # The key-split control and its positive case run at population 40: UNI keeps 15 elites
    # unmutated, so the split shows only through generation 1's other genomes, 5 of them at
    # population 20 and 25 at 40.
    @pytest.mark.parametrize('emr_overrides, recurrence_overrides, population', [
        # the whole recipe (the controls' positive case)
        (None, None, POP),
        # the same at the key-split control's population
        (None, None, 40),
        # W1 and W2 pruned at EMR's 0.1, not at band_threshold
        ({'weight_threshold': WEIGHT_THRESHOLD}, {}, POP),
        ({}, {'settling_rule': 'emr'}, POP),                 # EMR's settling rule
        # no pre-ask key split (shows from the second generation)
        ({'extra_randkey_split': False}, {}, 40),
        ({}, {'allow_lateral': False}, POP),                 # forward pairs only
    ])
    def test_each_part_of_the_recipe_matters(self, emr_overrides, recurrence_overrides, population):
        """Negative controls, at band_threshold 1.0: at UNI's default 0.3 the weights pruning at 0.1
        keeps in addition are small enough to move the fitness by 6e-6 only, inside the tolerance of
        an inexact backend. The whole recipe equals UNI there too; without any one part, the
        generations are not UNI's."""
        config = _uni_config(
            {**H_TO_H, 'hh_cache_enabled': False}, band_threshold=1.0, population=population
        )
        theirs, their_records = _run_uni(config, 2)
        ours, our_records = _run_emr(uni_recipe(config, emr_overrides, recurrence_overrides), 2)
        if emr_overrides is None:
            _assert_same(ours, theirs)
            assert our_records == their_records
            return
        largest = max(float(np.max(np.abs(mine - uni))) for mine, uni in zip(ours, theirs))
        assert largest > 1e-3 or our_records != their_records


class TestCacheTrigger:
    """recurrence.hh_cache_trigger: 'time' refreshes every hh_refresh_interval generations and never
    on a mask change; the default 'time_or_mask' also refreshes when the mask changes by more than
    hh_mask_change_threshold.
    """

    @staticmethod
    def _cache(trigger):
        cache = HHCacheManager(
            EMRConfig(
                hh_cache_enabled=True,
                hh_refresh_interval=3,
                hh_mask_change_threshold=0.1,
                hh_cache_trigger=trigger,
            )
        )
        mask = jnp.zeros((4, 10), dtype=bool)
        cache.update_cache(object(), mask, generation=0)
        return cache, mask

    @pytest.mark.parametrize('trigger, refreshes', [('time', False), ('time_or_mask', True)])
    def test_a_changed_mask(self, trigger, refreshes):
        cache, mask = self._cache(trigger)
        assert cache.should_refresh(1, ~mask) is refreshes
        assert cache.should_refresh(1, mask) is False

    @pytest.mark.parametrize('trigger', ['time', 'time_or_mask'])
    def test_the_interval(self, trigger):
        cache, mask = self._cache(trigger)
        assert [cache.should_refresh(generation, mask) for generation in (1, 2, 3)] == [
            False,
            False,
            True,
        ]

    def test_an_unknown_trigger(self):
        with pytest.raises(ValueError, match='hh_cache_trigger must be one of'):
            HHCacheManager(EMRConfig(hh_cache_enabled=True, hh_cache_trigger='mask'))


def _recurrent_config(cache, **recurrence):
    config = create_config_with_recurrence(
        'hidden_only', max_depth=DEPTH, population_size=POP, hh_cache_enabled=cache
    )
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'].update(recurrence)
    return config


def _emr(config):
    return config['algorithm_params']['emrhyperneat']['emr_hyperneat']


class TestInitializeStartsANewCache:
    """Only create_config built the h->h cache, so an object reused for another run through
    initialize alone (one configuration, several seeds) served the new run the last run's
    connections until its generation count passed the last refresh by hh_refresh_interval, or the
    mask changed: initialize now builds a new cache."""

    @staticmethod
    def _reused(keep_the_cache=False):
        """Three generations, then a new run of two on the same object and configuration."""
        algorithm = EMRHyperNEAT()
        config = algorithm.create_config(
            _recurrent_config(True, hh_refresh_interval=2, hh_cache_trigger='time')
        )
        problem = XORProblem()
        _generations(algorithm, algorithm.initialize(config, problem, seed=SEED), problem, 3)
        kept = algorithm._hh_cache
        state = algorithm.initialize(config, problem, seed=SEED)
        if keep_the_cache:
            algorithm._hh_cache = kept    # 0.1.0's behavior
        return _generations(algorithm, state, problem, 2)

    @staticmethod
    def _fresh():
        return _run_emr(
            _recurrent_config(True, hh_refresh_interval=2, hh_cache_trigger='time'), 2, XORProblem()
        )

    def test_a_second_run_equals_a_fresh_one(self):
        again, again_records = self._reused()
        fresh, fresh_records = self._fresh()
        _assert_same(again, fresh)
        assert again_records == fresh_records
        assert [status for _, status in again_records] == ['refresh', 'cached']

    def test_negative_control_the_kept_cache(self):
        """With the last run's cache put back, the new run's first generation is served the
        connections of the last run's third population and scores otherwise than a fresh run."""
        again, again_records = self._reused(keep_the_cache=True)
        fresh, _ = self._fresh()
        assert again_records[0][1] == 'cached'
        assert float(np.max(np.abs(again[0] - fresh[0]))) > 1e-4


class TestWeightThreshold:
    """emr_hyperneat.weight_threshold prunes the input-to-hidden and hidden-to-output weights
    (WEIGHT_THRESHOLD, 0.1, unless set)."""

    @staticmethod
    def _weights(threshold=None, **emr):
        config = create_base_config(population_size=POP, max_depth=DEPTH)
        _emr(config).update(emr)
        if threshold is not None:
            _emr(config)['weight_threshold'] = threshold
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=SEED)
        algorithm.run_generation(state, XORProblem())
        weights = np.concatenate([np.abs(np.asarray(algorithm._cached_W1)).ravel(),
                                  np.abs(np.asarray(algorithm._cached_W2)).ravel()])
        return algorithm, weights[weights > 0], np.asarray(algorithm._cached_fitnesses)

    def test_the_default(self):
        algorithm, kept, _ = self._weights()
        assert algorithm.weight_threshold == WEIGHT_THRESHOLD
        assert algorithm.substrate_config().weight_threshold == WEIGHT_THRESHOLD
        assert kept.min() > WEIGHT_THRESHOLD
        assert np.any(kept <= 0.3)   # the negative control of the next test

    def test_a_set_threshold(self):
        algorithm, kept, _ = self._weights(0.3)
        assert algorithm.substrate_config().weight_threshold == 0.3
        assert kept.min() > 0.3

    def test_streaming_reads_it(self):
        _, _, standard = self._weights(0.3)
        _, _, streaming = self._weights(0.3, execution={'mode': 'streaming'})
        _, _, default = self._weights(None, execution={'mode': 'streaming'})
        _assert_same([streaming], [standard])
        assert float(np.max(np.abs(streaming - default))) > 1e-4

    @pytest.mark.parametrize('value', [-0.1, 8.0, True, '0.3'])
    def test_wrong_values(self, value):
        config = create_base_config(population_size=POP)
        _emr(config)['weight_threshold'] = value
        with pytest.raises(ValueError, match='weight_threshold must be a number'):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('extra, match', [
        ({'leo': {'enabled': True}}, 'LEO gates the weights'),
        ({'execution': {'mode': 'depth_offload'}}, 'would ignore weight_threshold=0.3'),
        ({'multi_gpu_strategy': 'EVAL_ONLY_PARALLEL'}, 'would ignore weight_threshold=0.3'),
    ])
    def test_refused_where_the_weights_are_built_at_0_1(self, extra, match):
        config = create_base_config(population_size=POP)
        _emr(config).update(extra, weight_threshold=0.3)
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)


class TestRefusals:

    @pytest.mark.parametrize(
        'cache, recurrence, match',
        [
            # the cache is off
            (False, {'hh_cache_trigger': 'time'}, 'hh_cache_trigger would be ignored'),
            (
                True,
                {'hh_cache_trigger': 'time', 'hh_mask_change_threshold': 0.2},
                'hh_mask_change_threshold would be',
            ),
            (True, {'hh_cache_trigger': 'mask'}, 'hh_cache_trigger must be one of'),
        ],
    )
    def test_cache_trigger(self, cache, recurrence, match):
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(_recurrent_config(cache, **recurrence))

    def test_cache_trigger_on_a_feed_forward_block(self):
        config = create_base_config(population_size=POP)
        _emr(config)['recurrence'] = {'hh_cache_trigger': 'time'}
        with pytest.raises(ValueError, match='hh_cache_trigger would be ignored'):
            EMRHyperNEAT().create_config(config)


class TestUniNames:
    """UNI's names: _use_sparse_hh, _use_full_recurrent, get_recurrence_metrics, and the
    generation's sparse_hh_count and step_timings['hh_cache_status']."""

    @pytest.mark.parametrize('recurrence, flags', [
        (None, (False, False)),
        ({'preset': 'hidden_only'}, (True, False)),
        ({'allow_hidden_to_hidden': True, 'allow_lateral': True}, (True, True)),
    ])
    def test_mode_flags(self, recurrence, flags):
        config = create_base_config(population_size=POP)
        if recurrence is not None:
            _emr(config)['recurrence'] = recurrence
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert (algorithm._use_sparse_hh, algorithm._use_full_recurrent) == flags

    @pytest.mark.parametrize('recurrence, flags', [
        ({'enabled': True, 'allow_hidden_to_hidden': True, 'allow_lateral': True}, (True, True)),
        ({'allow_hidden_to_hidden': True}, (True, False)),
    ])
    def test_uni_sets_the_same_rule(self, recurrence, flags):
        """UNI's own flags by the same rule. Its default block (enabled false) discovers forward and
        lateral pairs and reads (True, False); the recipe for it has lateral pairs enabled and reads
        (True, True) here.
        """
        config = _uni_config(recurrence)
        assert (
            hmr_value(
                'unified_recipe.uni_flags', lambda: live('uni_flags', config), inputs=(config,)
            )
            == flags
        )

    def test_generation_names(self):
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(_recurrent_config(True)), XORProblem(), seed=SEED
        )
        _, metrics = algorithm.run_generation(state, XORProblem())
        assert (
            metrics.custom_metrics['sparse_hh_count']
            == metrics.custom_metrics['total_hh_connections']
            > 0
        )
        assert metrics.custom_metrics['step_timings']['hh_cache_status'] == 'refresh'
        assert algorithm.get_recurrence_metrics() is algorithm.get_extended_metrics()
        assert algorithm.get_recurrence_metrics().hidden_to_hidden_connections == \
            metrics.custom_metrics['sparse_hh_count']

    def test_feed_forward_has_neither(self):
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(create_base_config(population_size=POP)),
            XORProblem(),
            seed=SEED,
        )
        _, metrics = algorithm.run_generation(state, XORProblem())
        assert 'sparse_hh_count' not in metrics.custom_metrics
        assert 'hh_cache_status' not in metrics.custom_metrics['step_timings']

    def test_the_record_carries_unis_fields(self):
        ours = {field.name for field in dataclasses.fields(EMRRecurrenceMetrics)}
        assert (
            hmr_value('unified_recipe.uni_record_fields', lambda: live('uni_record_fields')) <= ours
        )
