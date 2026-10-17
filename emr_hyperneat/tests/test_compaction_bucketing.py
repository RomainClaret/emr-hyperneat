"""Bucketed active-set compaction (`execution.compaction_bucket`).

The recurrent evaluation runs on the population's union of reached positions. Once that union
leaves part of the grid unused (depth 3 and deeper) its size changes between generations, and under
exact shapes every step after compaction meets a new shape and recompiles: 18 to 54 executables per
generation on two RTX 2080 Ti at depth 3, population 1000. `auto` pads the compact axis to a
bucketed length with virtual positions after the real ones (zero weights in and out, no h->h
connection, mask False). These tests pin that the padding changes no result and that a bucket size
compiles once. The padding changes no result bit for bit on an arm64 CPU; on x86 a sum over the
padded length can group its terms differently, and in other configurations than these cells a
genome's fitness moved by a float32 ulp or two (XOR, depth 3, population 20, seeds 42 and 1).
"""

import dataclasses

import jax
import numpy as np
import pytest

from conftest import (
    EMRHyperNEAT,
    XORProblem,
    create_config_with_recurrence,
    DEFAULT_SEED,
)
from emr_hyperneat import emrhyperneat as engine
from emr_hyperneat.execution import compaction

_ON_CPU = jax.default_backend() == 'cpu'


class TestBucketRule:
    def test_bounds_and_number_of_sizes(self):
        total = 87380  # the depth-7 ceiling
        sizes = set()
        for n in range(1, 20001):
            size = compaction.bucket_size(n, total)
            step = max(64, 1 << max(0, n.bit_length() - 4))
            assert n <= size < n + step, (n, size, step)
            sizes.add(size)
        # steps of 64 up to 512, then about eight sizes per doubling
        assert len(sizes) <= 80, len(sizes)

    def test_capped_at_the_ceiling(self):
        assert compaction.bucket_size(300, 320) == 320
        assert compaction.bucket_size(320, 320) == 320
        assert compaction.bucket_size(0, 320) == 0


class TestPaddedCompaction:
    def _active(self, seed, total=500, n=37):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=n, replace=False))
        return compaction.ActiveSet(indices=idx, num_active=n, total_positions=total), rng

    def test_weights_pad_with_zeros(self):
        exact, rng = self._active(0)
        padded = compaction.bucket_active_set(exact)
        assert padded.length > exact.num_active
        W1 = rng.standard_normal((3, 2, 500)).astype(np.float32)
        W2 = rng.standard_normal((3, 1, 500)).astype(np.float32)
        e1, e2 = (np.asarray(a) for a in compaction.compact_weights(W1, W2, exact))
        p1, p2 = (np.asarray(a) for a in compaction.compact_weights(W1, W2, padded))
        n = exact.num_active
        assert p1.shape[2] == p2.shape[2] == padded.length
        assert np.array_equal(p1[:, :, :n], e1) and np.array_equal(p2[:, :, :n], e2)
        assert not p1[:, :, n:].any() and not p2[:, :, n:].any()

    def test_remap_equals_the_exact_remap(self):
        exact, rng = self._active(1)
        padded = compaction.bucket_active_set(exact)
        # valid entries on real positions; invalid ones at -1, 0, inactive positions and past the grid
        values = np.concatenate([rng.choice(exact.indices, size=40),
                                 np.array([-1, 0, 499, 250, 10 ** 6])])
        pairs = np.stack([values, values[::-1]]).astype(np.int32)
        valid = np.zeros(pairs.shape, dtype=bool)
        valid[:, :40] = True
        hh = engine.SparseHiddenConnections(
            from_indices=pairs, to_indices=pairs[:, ::-1], weights=np.ones(pairs.shape, np.float32),
            valid_mask=valid, num_valid=valid.sum(axis=1))
        a = compaction.remap_sparse_hh(hh, exact)
        b = compaction.remap_sparse_hh(hh, padded)
        assert np.array_equal(np.asarray(a.from_indices), np.asarray(b.from_indices))
        assert np.array_equal(np.asarray(a.to_indices), np.asarray(b.to_indices))

    def test_pad_positions(self):
        exact, rng = self._active(2)
        padded = compaction.bucket_active_set(exact)
        masks = rng.random((4, 500)) < 0.3
        out = np.asarray(compaction.pad_positions(masks, padded, axis=1, fill=False))
        assert out.shape == (4, padded.length)
        assert np.array_equal(out[:, :exact.num_active], masks[:, exact.indices])
        assert not out[:, exact.num_active:].any()


_METRIC_FIELDS = ('total_positions', 'active_positions_mean', 'position_utilization', 'w1_density',
                  'w2_density', 'hidden_to_hidden_connections')


def _run(bucket, preset, level, act_mode='disabled', discovery='union', depth=3, population=30,
         generations=2):
    cfg = create_config_with_recurrence(preset, max_depth=depth, population_size=population,
                                        iteration_level=level, hh_cache_enabled=False)
    hmr = cfg['algorithm_params']['emrhyperneat']['emr_hyperneat']
    hmr['recurrence']['hh_discovery_mode'] = discovery
    hmr.setdefault('execution', {})['compaction_bucket'] = bucket
    if act_mode == 'cppn_output':
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}
    elif act_mode == 'random_generation':
        hmr['dynamic_functions'] = {'mode': 'random_generation', 'num_activations': 4}
    elif act_mode == 'agg_cppn4':
        hmr['aggregation'] = {'mode': 'cppn_output', 'num_aggregations': 4}
    sizes = []
    original = compaction.bucket_active_set

    def spy(active):
        padded = original(active)
        sizes.append((active.num_active, padded.length))
        return padded

    compaction.bucket_active_set = spy
    try:
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
        assert algorithm.execution_config.compaction_bucket == bucket
        out = {'fitness': [], 'hh': [], 'metrics': [], 'record': [], 'active': []}
        for _ in range(generations):
            state, _m = algorithm.run_generation(state, XORProblem(), verbose=False)
            out['fitness'].append(np.asarray(algorithm._cached_fitnesses))
            hh = algorithm._cached_sparse_hh
            out['hh'].append(tuple(np.asarray(x) for x in (hh.from_indices, hh.to_indices, hh.weights,
                                                            hh.valid_mask)))
            m = dataclasses.asdict(algorithm._extended_metrics)
            out['metrics'].append({k: m[k] for k in _METRIC_FIELDS})
            rec = algorithm._last_eval
            out['record'].append({k: None if rec[k] is None else np.asarray(rec[k])
                                  for k in ('mask_eval', 'eval_positions', 'act_indices')})
            active = getattr(algorithm, '_cached_active_indices', None)
            out['active'].append(None if active is None else np.asarray(active))
    finally:
        compaction.bucket_active_set = original
    return out, sizes


def _same(x, y):
    if x is None or y is None:
        return x is None and y is None
    return np.array_equal(x, y)


class TestAutoMatchesExact:
    @pytest.mark.parametrize('preset,level,act_mode,discovery', [
        ('hidden_only', 2, 'disabled', 'union'),
        ('full_recurrent', 1, 'disabled', 'per_genome'),
        ('with_backward', 2, 'cppn_output', 'union'),
        ('with_lateral', 2, 'agg_cppn4', 'per_genome'),
        ('hidden_only', 1, 'random_generation', 'union'),
    ])
    def test_same_results(self, preset, level, act_mode, discovery):
        exact, _ = _run('exact', preset, level, act_mode, discovery)
        auto, sizes = _run('auto', preset, level, act_mode, discovery)
        if act_mode == 'random_generation':
            # Random per-node draws depend on the array length: this mode keeps exact shapes.
            assert not sizes, sizes
        else:
            # The cell exercises the padding: compaction ran and padded at least once.
            assert sizes and any(length > n for n, length in sizes), sizes
        for gen in range(len(exact['fitness'])):
            fe, fa = exact['fitness'][gen], auto['fitness'][gen]
            if _ON_CPU:
                assert np.array_equal(fe, fa), f'generation {gen}: fitness differs by {np.max(np.abs(fe - fa))}'
            else:
                assert np.allclose(fe, fa, rtol=0, atol=1e-6)
            for a, b in zip(exact['hh'][gen], auto['hh'][gen]):
                assert np.array_equal(a, b)
            assert exact['metrics'][gen] == auto['metrics'][gen], (exact['metrics'][gen], auto['metrics'][gen])
            for key in exact['record'][gen]:
                assert _same(exact['record'][gen][key], auto['record'][gen][key]), key
            assert _same(exact['active'][gen], auto['active'][gen])


_COMPILES = {'active': False, 'count': 0, 'registered': False}


def _watch():
    if not _COMPILES['registered']:
        def listener(event, duration, **kwargs):
            if _COMPILES['active'] and event == '/jax/core/compile/backend_compile_duration':
                _COMPILES['count'] += 1
        jax.monitoring.register_event_duration_secs_listener(listener)
        _COMPILES['registered'] = True


class TestAutoCompilesOncePerBucket:
    def test_no_recompile_without_a_new_size(self):
        """Depth 3, population 100, per-genome discovery (its cores compile once per bucket size):
        from generation 2 on, a generation whose compact length and per-genome bucket sizes were all
        seen before compiles nothing. That includes the evaluation's settling scan, whose step is
        made once for tanh nodes (`settle_hidden`); a step made per call compiled the scan again,
        one executable, every generation.

        A size counts as seen from generation 1 on. Generation 0 compiles before the generation-1
        retrace (the argument types change once after the first generation), so a size seen only at
        generation 0 may compile again when it comes back: 2 executables at generation 2 on a GPU
        for a length last seen at generation 0."""
        _watch()
        cfg = create_config_with_recurrence('hidden_only', max_depth=3, population_size=100,
                                            iteration_level=1, hh_cache_enabled=False)
        hmr = cfg['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence']['hh_discovery_mode'] = 'per_genome'
        hmr.setdefault('execution', {})['compaction_bucket'] = 'auto'
        lengths, seen, new_size, compiles = [], set(), [], []
        original_bucket = compaction.bucket_active_set
        original_discover = engine.discover_sparse_hh_vectorized_multi_hop
        per_generation = {}

        def spy_bucket(active):
            padded = original_bucket(active)
            per_generation['length'] = padded.length
            return padded

        def spy_discover(state, cppns, h_grid, masks_A, *args, **kwargs):
            counts = np.asarray(masks_A).sum(axis=1)
            per_generation['buckets'] = {engine._bucket_size_for(int(c)) for c in counts if c > 0}
            return original_discover(state, cppns, h_grid, masks_A, *args, **kwargs)

        compaction.bucket_active_set = spy_bucket
        engine.discover_sparse_hh_vectorized_multi_hop = spy_discover
        try:
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(algorithm.create_config(cfg), XORProblem(), seed=DEFAULT_SEED)
            for generation in range(6):
                per_generation.clear()
                _COMPILES['count'] = 0
                _COMPILES['active'] = True
                try:
                    state, _m = algorithm.run_generation(state, XORProblem(), verbose=False)
                    jax.block_until_ready(algorithm._cached_fitnesses)
                finally:
                    _COMPILES['active'] = False
                keys = {('length', per_generation.get('length'))}
                keys |= {('bucket', b) for b in per_generation.get('buckets', set())}
                new_size.append(bool(keys - seen))
                if generation >= 1:
                    seen |= keys
                compiles.append(_COMPILES['count'])
                lengths.append(per_generation.get('length'))
        finally:
            compaction.bucket_active_set = original_bucket
            engine.discover_sparse_hh_vectorized_multi_hop = original_discover
        assert any(length is not None for length in lengths), 'compaction never ran at depth 3'
        checked = 0
        for generation in range(2, 6):
            if not new_size[generation]:
                checked += 1
                assert compiles[generation] == 0, (
                    f'generation {generation} compiled {compiles[generation]} executables with no new '
                    f'size (compiles {compiles}, lengths {lengths})')
        assert checked >= 1, f'every generation had a new size; the test checked nothing ({lengths})'


class TestSetting:
    def test_default_is_auto_and_exact_is_honored(self):
        from emr_hyperneat.execution import parse_execution_config
        assert parse_execution_config({}).compaction_bucket == 'auto'
        assert parse_execution_config({'execution': {'compaction_bucket': 'exact'}}).compaction_bucket == 'exact'

    def test_unknown_value_is_refused(self):
        from emr_hyperneat.execution import parse_execution_config
        with pytest.raises(ValueError, match='compaction_bucket'):
            parse_execution_config({'execution': {'compaction_bucket': 'pow2'}})


class TestBucketBoundary:
    def test_host_path_also_without_padding(self, monkeypatch):
        """A real count exactly on a bucket length needs no padding; the bucketed path must still build the
        compact arrays on the host. Taking the exact path's device gathers there compiled a set of other
        executables once (13 at a count of 128 or 320 on two RTX 2080 Ti). Forcing every count onto a
        bucket length (bucket_size = n) makes every generation that case."""
        monkeypatch.setattr(compaction, 'bucket_size', lambda n, total: int(n))
        calls = {'pad': 0}
        original = compaction.pad_positions

        def spy(values, active, axis, fill=0):
            calls['pad'] += 1
            return original(values, active, axis, fill)

        monkeypatch.setattr(compaction, 'pad_positions', spy)
        auto, sizes = _run('auto', 'hidden_only', 1)
        assert sizes and all(length == n for n, length in sizes), sizes
        assert calls['pad'] >= len(sizes), (calls, sizes)
        exact, _ = _run('exact', 'hidden_only', 1)
        for gen in range(len(exact['fitness'])):
            if _ON_CPU:
                assert np.array_equal(exact['fitness'][gen], auto['fitness'][gen])
            else:
                assert np.allclose(exact['fitness'][gen], auto['fitness'][gen], rtol=0, atol=1e-6)


class TestEvalChunkKeyedToLength:
    """The auto evaluation chunk (eval_chunk_size -1) follows the evaluated length, padding included.

    Keyed to the real count, the 1024 bucket (961 to 1,024 real positions) would evaluate counts below 1,000 whole
    and counts from 1,000 in chunks, compiling the chunked path at a length already seen.
    """

    def test_padded_length_decides(self):
        # Every count in the 1024 bucket evaluates at length 1024 and so gets the one chunk that length gets.
        assert engine._auto_eval_chunk_size(1024, 1000) == 500
        assert engine._auto_eval_chunk_size(999, 1000) is None

    @pytest.mark.parametrize('length, pop, chunk', [
        (999, 1000, None), (1000, 1000, 500), (1024, 300, 300), (5999, 1000, 500), (6000, 1000, 50),
        (49999, 1000, 50), (50000, 1000, 200), (199999, 150, 150), (200000, 1000, 50)])
    def test_thresholds(self, length, pop, chunk):
        assert engine._auto_eval_chunk_size(length, pop) == chunk
