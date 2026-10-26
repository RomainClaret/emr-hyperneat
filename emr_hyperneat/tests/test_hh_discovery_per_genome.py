"""The two h->h discovery queries and the multi-hop second factor.

`recurrence.hh_discovery_mode` chooses the query shape: `union` (one CPPN query over the
population's union of active positions, each genome then restricted to its own positions) or
`per_genome` (each genome over its own positions, in static buckets). `recurrence.multi_hop_second_factor`
chooses the algebra of the multi-hop expansion at `iteration_level` >= 2: `masked` (the default)
composes the genome's existing direct connections, hop h scaled by decay^(h-1), with a max-magnitude
merge; `raw`, which reproduces 0.1.0's expansion, multiplies by the raw weights over the
whole union and carries the decayed product into the next hop, so a genome's composite connections
can end on positions it does not have active, ignore its connection class and depend on the other
genomes.

What must hold:
- level 1: `union` and `per_genome` give every genome the same connections in the same order, the
  same weights (within one float32 ulp) and the same fitness;
- level 2 with `masked`: the same connections, weights within a few float32 ulps (the products run
  over different array shapes; over 36 configurations one weight of 4,708 was 3 ulps off, in one of them),
  and both match a numpy implementation of the documented formula (a validity bit may differ only for
  a pair whose weight sits at the band threshold);
- `raw` reproduces the leak (connections outside the genome's active set, class violations under
  hidden_only) and `masked` has none; `raw` with `per_genome` is refused;
- level 2 composites respect the preset's connection class in both paths:
  with_backward carries no lateral or self-loop composite, with_lateral no backward or self-loop one;
- the corrected decay: on a 4-node chain at level 3 the 3-hop term carries decay^2, not decay^3.
"""

import jax
import numpy as np
import pytest

from conftest import (
    EMRHyperNEAT,
    XORProblem,
    create_config_with_recurrence,
    DEFAULT_SEED,
    create_config_with_leo,
)
from emr_hyperneat import emrhyperneat as engine

RECURRENT_PRESETS = ['hidden_only', 'with_backward', 'with_lateral', 'with_self', 'full_recurrent']

# Tolerances by backend (two RTX 2080 Ti, jax 0.4.38): the CPU gives the union and
# the per-genome path bit-identical weights at level 1, the GPU 1 to 6 float32 ulps apart at level 1 and
# up to about 28 ulps against the reference formula at level 3, with identical connection sets and
# fitness: XLA fuses the same per-pair arithmetic differently for different batch shapes. The criterion
# stays "same connections, weights within a few ulps"; the numbers below are the observed maxima with
# headroom.
_ON_CPU = jax.default_backend() == 'cpu'
_ULPS_DIRECT = 1 if _ON_CPU else 8        # union against per_genome, level 1 and level 2 (x4 there)
_ULPS_REFERENCE = 4 if _ON_CPU else 64    # against the numpy reference of the documented formula, per weight
# The GPU's differences are absolute, not relative to the final weight: they arise in intermediate
# sums whose scale is set by max_weight, so a small weight (-0.3617, ulp 3e-8) can sit 11 of its own
# ulps away while its absolute difference (3.3e-7) is under 2 ulps of max_weight (at most 7.7e-7 at
# level 1 and 5.7e-6 at level 2 on two RTX 2080 Ti). Off the CPU a weight passes within either bound;
# connection sets stay exact everywhere.
# Against the numpy reference, a weight passes within _ULPS_REFERENCE of its ulps plus _REFERENCE_SAFETY times the
# float32 error bound of the sums behind it, which _reference_multi_hop computes per pair. An ulp count alone does
# not hold: each side sums the products in its own order. On an M4 Max (arm64) XLA and numpy's OpenBLAS both add
# them in index order with fused multiply-adds and agree bit for bit; numpy's OpenBLAS on x86 keeps two interleaved
# accumulators, which put the reference up to 17 ulps of the weight from the engine on a Xeon E5-2683 v4 (jax 0.6.1,
# numpy 1.26.4) and 2 to 7 ulps of max_weight on a Xeon 6542Y (1.7e-6 at most). The bound assumes float32 products,
# which conftest's pin of jax_default_matmul_precision to 'highest' keeps on GPUs that have TF32 (Ampere and later).
_U32 = float(np.finfo(np.float32).eps) / 2   # float32 unit roundoff, 2**-24
# The bound is derived for correctly rounded float32 operations on the same direct weights. The factor 2 is the
# margin for what it does not model: the reference composes the level-1 run's direct weights and the engine the
# level-k run's, two compiled programs that may give them an ulp apart on some processor (identical on the M4 Max).
# The allowance leaves the check its power (the 8 cases of test_matches_documented_formula, CPU): the x86
# order, emulated, uses at most 4.9% of it, and dropping the smallest product of an unclipped composite moves the
# composite by at least 800 times it.
_REFERENCE_SAFETY = 2.0
# Fitness off the CPU: up to 3.1e-5 apart at generation 0 on two RTX 2080 Ti.
_FIT_ATOL = 0.0 if _ON_CPU else 1e-4


def _atol(settings, level):
    """Absolute weight tolerance: 0 on the CPU, else 8 (level 1) or 64 (level >= 2) ulps of max_weight."""
    if _ON_CPU:
        return 0.0
    return (8 if level == 1 else 64) * float(np.spacing(np.float32(settings['max_weight'])))


def _run(mode, preset, iteration_level, act_mode, depth, population, generations=2, second_factor=None,
         max_sparse_conns=None):
    """Run `generations` generations; return the last sparse h->h set, the fitness per generation,
    the population's active masks and the grid positions of the last discovery call, and settings."""
    config = create_config_with_recurrence(
        preset, max_depth=depth, population_size=population,
        iteration_level=iteration_level, hh_cache_enabled=False,
    )
    hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    hmr['recurrence']['hh_discovery_mode'] = mode
    if second_factor is not None:
        hmr['recurrence']['multi_hop_second_factor'] = second_factor
    if max_sparse_conns is not None:
        hmr['recurrence']['max_sparse_conns'] = max_sparse_conns
    if act_mode == 'cppn_output':
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}
    captured = {}
    original = engine.discover_sparse_hh_vectorized_multi_hop

    def spy(state, cppns_transformed, h_grid, masks_A, *args, **kwargs):
        captured['masks_A'] = np.asarray(masks_A)
        captured['positions'] = np.asarray(h_grid.all_positions)
        return original(state, cppns_transformed, h_grid, masks_A, *args, **kwargs)

    engine.discover_sparse_hh_vectorized_multi_hop = spy
    try:
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
        # The preset branch of the parser must honor the keys, or both runs take the default path.
        assert algorithm.extended_config.hh_discovery_mode == mode
        if second_factor is not None:
            assert algorithm.extended_config.multi_hop_second_factor == second_factor
        fitnesses = []
        for _ in range(generations):
            state, _metrics = algorithm.run_generation(state, XORProblem(), verbose=False)
            fitnesses.append(np.asarray(algorithm._cached_fitnesses))
    finally:
        engine.discover_sparse_hh_vectorized_multi_hop = original
    settings = {
        'band_threshold': float(hmr['band_threshold']),
        'max_weight': float(hmr['max_weight']),
        'hop_decay_factor': float(algorithm.extended_config.hop_decay_factor),
        'custom_metrics': getattr(_metrics, 'custom_metrics', None),
    }
    return algorithm._cached_sparse_hh, fitnesses, captured['masks_A'], captured['positions'], settings


def _connection_lists(hh):
    fr = np.asarray(hh.from_indices)
    to = np.asarray(hh.to_indices)
    w = np.asarray(hh.weights)
    valid = np.asarray(hh.valid_mask)
    return [(fr[g][valid[g]], to[g][valid[g]], w[g][valid[g]]) for g in range(fr.shape[0])]


def _assert_same_connections(hh_a, hh_b, label='a against b', ulps=_ULPS_DIRECT, atol=0.0):
    lists_a = _connection_lists(hh_a)
    lists_b = _connection_lists(hh_b)
    assert len(lists_a) == len(lists_b)
    total = 0
    for g, ((fa, ta, wa), (fb, tb, wb)) in enumerate(zip(lists_a, lists_b)):
        assert fa.shape == fb.shape, f'{label}, genome {g}: {fa.shape[0]} against {fb.shape[0]} connections'
        assert np.array_equal(fa, fb) and np.array_equal(ta, tb), f'{label}, genome {g}: connection sets differ'
        ulp = np.spacing(np.abs(wa).astype(np.float32))
        assert np.all(np.abs(wa - wb) <= np.maximum(ulps * ulp, atol)), (
            f'{label}, genome {g}: a weight differs by more than {ulps} ulp and {atol:.2e} '
            f'(max {np.max(np.abs(wa - wb))})'
        )
        total += fa.shape[0]
    return total


def _leak_counts(hh, masks_A, positions, forward_only):
    """Connections whose target is not active for the genome, and forward-only violations."""
    fr, to, valid = (np.asarray(hh.from_indices), np.asarray(hh.to_indices), np.asarray(hh.valid_mask))
    outside = violations = 0
    for g in range(fr.shape[0]):
        sel = valid[g]
        outside += int((~masks_A[g][to[g][sel]]).sum())
        if forward_only:
            violations += int((positions[fr[g][sel], 1] >= positions[to[g][sel], 1]).sum())
    return outside, violations


def _reference_multi_hop(direct, band_threshold, hop_decay_factor, max_weight, level):
    """The documented expansion on one genome's direct connections, in numpy float32, with an error bound.

    A_total starts as A; for hop in 1..level-1: P = A_current @ A over the genome's own positions,
    P thresholded at the band, the decayed term decay^hop * P merged by larger magnitude into
    A_total, and the undecayed P carried to the next hop; clip to max_weight; valid where
    |A_total| > band.

    The bound is how far a correct float32 implementation can be from this reference, per pair. A
    product may sum in any order (numpy, XLA's CPU kernels and the GPU's all differ), and each side is
    then within gamma_n * (|X| @ |A|) of the exact product (the standard dot-product bound, gamma_n =
    n u / (1 - n u), n the number of positions, u = 2**-24), so two implementations are within twice
    that. A difference carried from the previous hop propagates through |A|; the decay multiplication
    adds one rounding and the decay factor itself one ulp (the fori_loop twin computes it as a traced
    float32 power). Where the merge compares two values closer than their bounds, either may win and
    the bound covers both. A wrong formula (the raw second factor, the decayed carry) is off by
    1e-2 or more; this bound is about 1e-5.
    """
    fr, to, w = direct
    idx = np.unique(np.concatenate([fr, to]))
    local = {p: i for i, p in enumerate(idx)}
    n = idx.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    A[[local[p] for p in fr], [local[p] for p in to]] = w.astype(np.float32)
    gamma = n * _U32 / (1 - n * _U32)
    abs_A = np.abs(A).astype(np.float64)
    current, total = A.copy(), A.copy()
    err_current = np.zeros((n, n))
    err_total = np.zeros((n, n))
    for hop in range(1, level):
        power = (current @ A).astype(np.float32)
        err_power = err_current @ abs_A + 2 * gamma * (np.abs(current).astype(np.float64) @ abs_A)
        kept = np.abs(power) > band_threshold
        power = np.where(kept, power, np.float32(0.0))
        err_power = np.where(kept, err_power, 0.0)   # a pair kept on one side only is a borderline pair
        factor = np.float32(hop_decay_factor ** hop)
        decayed = (power * factor).astype(np.float32)
        err_decayed = (err_power * float(factor) + np.spacing(np.abs(decayed)).astype(np.float64)
                       + np.abs(power).astype(np.float64) * float(np.spacing(factor)))
        take = np.abs(decayed) > np.abs(total)
        close = np.abs(np.abs(decayed).astype(np.float64) - np.abs(total)) <= err_decayed + err_total
        err_total = np.where(close, np.abs(decayed.astype(np.float64) - total) + np.maximum(err_decayed, err_total),
                             np.where(take, err_decayed, err_total))
        total = np.where(take, decayed, total)
        current = power
        err_current = err_power
    total = np.clip(total, -max_weight, max_weight)
    valid = np.abs(total) > band_threshold
    return idx, total, valid, err_total


def _assert_matches_reference(direct_lists, expanded_lists, settings, level, label):
    band = settings['band_threshold']
    checked = borderline = 0
    for g, ((fr, to, w), (fr2, to2, w2)) in enumerate(zip(direct_lists, expanded_lists)):
        if fr.shape[0] == 0:
            assert fr2.shape[0] == 0
            continue
        idx, total, valid, err = _reference_multi_hop(
            (fr, to, w), band, settings['hop_decay_factor'], settings['max_weight'], level,
        )
        local = {int(p): i for i, p in enumerate(idx)}
        ref = {(int(idx[i]), int(idx[j])): float(total[i, j]) for i, j in zip(*np.nonzero(valid))}
        got = {(int(a), int(b)): float(c) for a, b, c in zip(fr2, to2, w2)}
        for pair in set(ref) ^ set(got):
            value = ref.get(pair, got.get(pair))
            assert abs(abs(value) - band) < 1e-4, f'{label}, genome {g}: pair {pair} weight {value} differs'
            borderline += 1
        for pair in set(ref) & set(got):
            bound = (_ULPS_REFERENCE * float(np.spacing(np.float32(abs(ref[pair]))))
                     + _REFERENCE_SAFETY * err[local[pair[0]], local[pair[1]]])
            assert abs(ref[pair] - got[pair]) <= bound, (
                f'{label}, genome {g}: pair {pair} reference {ref[pair]} against {got[pair]} '
                f'(bound {bound:.3g})'
            )
            checked += 1
    assert checked > 0
    assert borderline <= max(1, checked // 100), f'{label}: {borderline} borderline pairs against {checked} checked'


class TestLevelOneMatchesUnion:
    """iteration_level 1: same connections, same weights, same fitness as the union path."""

    @pytest.mark.parametrize('act_mode', ['disabled', 'cppn_output'])
    @pytest.mark.parametrize('preset', RECURRENT_PRESETS)
    def test_depth3_population30(self, preset, act_mode):
        hh_u, fit_u, _m, _p, settings = _run('union', preset, 1, act_mode, depth=3, population=30)
        engine._per_genome_discovery_cache.clear()
        hh_p, fit_p, _m, _p, _ = _run('per_genome', preset, 1, act_mode, depth=3, population=30)
        # Witness that the per-genome path ran: it compiled at least one bucket core.
        assert len(engine._per_genome_discovery_cache) >= 1, 'per-genome core never compiled'
        total = _assert_same_connections(hh_u, hh_p, 'union against per_genome', atol=_atol(settings, 1))
        assert total > 0, 'the configuration discovered no h->h connection, the test is vacuous'
        for gen, (a, b) in enumerate(zip(fit_u, fit_p)):
            assert np.allclose(a, b, rtol=0, atol=_FIT_ATOL), (
                f'generation {gen}: fitness differs by {np.max(np.abs(a - b))}')

    def test_depth4_population100_mixed_buckets(self):
        """A population whose active counts span several bucket sizes."""
        hh_u, fit_u, _m, _p, settings = _run('union', 'hidden_only', 1, 'disabled', depth=4, population=100)
        engine._per_genome_discovery_cache.clear()
        hh_p, fit_p, _m, _p, _ = _run('per_genome', 'hidden_only', 1, 'disabled', depth=4, population=100)
        assert len(engine._per_genome_discovery_cache) >= 2, 'expected more than one bucket size'
        total = _assert_same_connections(hh_u, hh_p, 'union against per_genome', atol=_atol(settings, 1))
        assert total > 0
        assert np.array_equal(np.asarray(hh_u.num_valid), np.asarray(hh_p.num_valid))
        for gen, (a, b) in enumerate(zip(fit_u, fit_p)):
            assert np.allclose(a, b, rtol=0, atol=_FIT_ATOL), (
                f'generation {gen}: fitness differs by {np.max(np.abs(a - b))}')


class TestMultiHopSemantics:
    """iteration_level >= 2 under the masked second factor, and the raw factor's leak."""

    @pytest.mark.parametrize('preset', ['hidden_only', 'with_backward', 'full_recurrent'])
    def test_union_masked_equals_per_genome_at_level_two(self, preset):
        hh_u, fit_u, _m, _p, settings = _run('union', preset, 2, 'disabled', 3, 30, second_factor='masked')
        hh_p, fit_p, _m, _p, _ = _run('per_genome', preset, 2, 'disabled', 3, 30)
        # (U, U) against (S, S) products: the same terms summed in another order; 3 ulps observed once.
        total = _assert_same_connections(hh_u, hh_p, 'union masked against per_genome', ulps=4 * _ULPS_DIRECT,
                                         atol=_atol(settings, 2))
        assert total > 0
        for gen, (a, b) in enumerate(zip(fit_u, fit_p)):
            # The products run over different shapes ((U, U) against (S, S)); identical on this CPU,
            # allowed to differ by float order elsewhere.
            # The CPU keeps its 1e-6 at level 2; off the CPU the bound of the other fitness checks
            # (3.08e-5 apart at generation 0 on two RTX 2080 Ti, with_backward).
            assert np.allclose(a, b, rtol=0, atol=1e-6 if _ON_CPU else _FIT_ATOL), (
                f'generation {gen}: fitness differs by {np.max(np.abs(a - b))}')

    @pytest.mark.parametrize('level', [2, 3])
    @pytest.mark.parametrize('mode', ['per_genome', 'union'])
    @pytest.mark.parametrize('preset', ['hidden_only', 'full_recurrent'])
    def test_matches_documented_formula(self, preset, mode, level):
        # Generation 1 evaluates the seeded initial population in both runs, so the direct
        # connections of the level-1 run are the higher level's 1-hop matrix.
        hh_1, _f, masks_1, _p, settings = _run(mode, preset, 1, 'disabled', 3, 30, generations=1)
        hh_k, _f, masks_k, _p, _ = _run(mode, preset, level, 'disabled', 3, 30, generations=1)
        assert np.array_equal(masks_1, masks_k)
        _assert_matches_reference(_connection_lists(hh_1), _connection_lists(hh_k), settings, level,
                                  f'{mode} level {level} {preset}')

    def test_the_reference_check_sees_one_wrong_product(self):
        """The rounding bound leaves the check its power: an engine that dropped the smallest product of one
        unclipped 2-hop composite fails it."""
        hh_1, _f, _m, _p, settings = _run('per_genome', 'full_recurrent', 1, 'disabled', 3, 30, generations=1)
        hh_2, _f, _m, _p, _ = _run('per_genome', 'full_recurrent', 2, 'disabled', 3, 30, generations=1)
        direct, expanded = _connection_lists(hh_1), _connection_lists(hh_2)
        _assert_matches_reference(direct, expanded, settings, 2, 'as computed')
        decay = np.float32(settings['hop_decay_factor'])
        for g, ((fr, to, w), (fr2, to2, w2)) in enumerate(zip(direct, expanded)):
            weight = {(a, b): c for a, b, c in zip(fr.tolist(), to.tolist(), w.tolist())}
            for k, (a, b) in enumerate(zip(fr2.tolist(), to2.tolist())):
                if abs(w2[k]) >= settings['max_weight'] or weight.get((a, b)) == float(w2[k]):
                    continue    # clipped, or the direct weight kept
                products = [weight[(a, j)] * weight[(j, b)] for (src, j) in weight if src == a and (j, b) in weight]
                if len(products) < 2:
                    continue
                wrong = [(f, t, x.copy()) for f, t, x in expanded]
                wrong[g][2][k] = w2[k] - decay * np.float32(min(products, key=abs))
                with pytest.raises(AssertionError, match=r'reference .* against .* \(bound'):
                    _assert_matches_reference(direct, wrong, settings, 2, 'one product dropped')
                return
        pytest.fail('no unclipped composite of two products or more: the check is not exercised')

    def test_raw_second_factor_leaks_and_masked_does_not(self):
        hh_raw, _f, m_raw, pos, _ = _run('union', 'hidden_only', 2, 'disabled', 3, 30, generations=1,
                                          second_factor='raw')
        hh_msk, _f, m_msk, pos, _ = _run('union', 'hidden_only', 2, 'disabled', 3, 30, generations=1,
                                          second_factor='masked')
        hh_pg, _f, m_pg, pos, _ = _run('per_genome', 'hidden_only', 2, 'disabled', 3, 30, generations=1)
        assert np.array_equal(m_raw, m_msk) and np.array_equal(m_msk, m_pg)
        outside_raw, violations_raw = _leak_counts(hh_raw, m_raw, pos, forward_only=True)
        # On this configuration the raw factor gave 14,949 of 19,110 connections outside the genome's
        # active set and 7,486 backward or lateral ones under hidden_only.
        assert outside_raw > 0 and violations_raw > 0
        assert _leak_counts(hh_msk, m_msk, pos, forward_only=True) == (0, 0)
        assert _leak_counts(hh_pg, m_pg, pos, forward_only=True) == (0, 0)
        assert int(np.asarray(hh_raw.valid_mask).sum()) > int(np.asarray(hh_msk.valid_mask).sum())

    @pytest.mark.parametrize('preset', ['with_backward', 'with_lateral'])
    @pytest.mark.parametrize('mode', ['union', 'per_genome'])
    def test_masked_composites_respect_the_class(self, preset, mode):
        """A forward edge composed with a backward one lands lateral or on its source; both are
        dropped, so the preset's classes hold at level 2."""
        hh, _f, masks, pos, _ = _run(mode, preset, 2, 'disabled', 3, 30, generations=1, second_factor=None)
        fr, to, valid = (np.asarray(hh.from_indices), np.asarray(hh.to_indices), np.asarray(hh.valid_mask))
        lateral = self_loops = backward = total = 0
        for g in range(fr.shape[0]):
            sel = valid[g]
            f, t = fr[g][sel], to[g][sel]
            total += int(sel.sum())
            self_loops += int(np.sum(f == t))
            lateral += int(np.sum((pos[f, 1] == pos[t, 1]) & (f != t)))
            backward += int(np.sum(pos[f, 1] > pos[t, 1]))
        assert total > 0
        if preset == 'with_backward':
            assert lateral == 0 and self_loops == 0 and backward > 0
        else:
            assert backward == 0 and self_loops == 0 and lateral > 0

    @pytest.mark.parametrize('level', [1, 2, 3])
    @pytest.mark.parametrize('mode', ['union', 'per_genome'])
    def test_no_repeated_pair_and_both_endpoints_active(self, mode, level):
        """Every (from, to) pair occurs at most once per genome and both endpoints are active for
        that genome. The per-genome buckets pad a genome's active list with position 0, and the
        padded slots do not surface at level >= 2 (both factors of every product are zero on padded
        rows and columns); this pins it for both paths at every level, which the reference check
        cannot because it merges pairs into a dict."""
        hh, _f, masks, _pos, _ = _run(mode, 'hidden_only', level, 'disabled', 3, 30, generations=1)
        fr, to, valid = (np.asarray(hh.from_indices), np.asarray(hh.to_indices), np.asarray(hh.valid_mask))
        total = 0
        for g in range(fr.shape[0]):
            sel = valid[g]
            pairs = np.stack([fr[g][sel], to[g][sel]], axis=1)
            assert np.unique(pairs, axis=0).shape[0] == pairs.shape[0], f'genome {g}: a repeated pair'
            assert masks[g][fr[g][sel]].all(), f'genome {g}: a source that is not active'
            assert masks[g][to[g][sel]].all(), f'genome {g}: a target that is not active'
            total += pairs.shape[0]
        assert total > 0

    @pytest.mark.parametrize('mode', ['union', 'per_genome'])
    def test_leo_replacement_keeps_direct_connections_at_level_two(self, mode):
        """Under LEO replacement mode a direct connection can carry |w| <= band (the LEO gate
        replaced the weight gate). The expansion's final rule `|A_total| > band` dropped such
        connections at level 2 (63 of 2,415 on this cell); the expansion never removes a direct
        connection."""
        sets = {}
        for level in (1, 2):
            config = create_config_with_leo('default', recurrence_preset='hidden_only', max_depth=3,
                                            population_size=30, mode='replacement')
            hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            hmr['recurrence']['iteration_level'] = level
            hmr['recurrence']['hh_cache_enabled'] = False
            hmr['recurrence']['hh_discovery_mode'] = mode
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(
                algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED
            )
            assert algorithm.use_leo and algorithm.leo_mode == 'replacement'
            assert algorithm.extended_config.hh_discovery_mode == mode
            algorithm.run_generation(state, XORProblem(), verbose=False)
            sets[level] = [set(zip(f.tolist(), t.tolist()))
                           for f, t, _w in _connection_lists(algorithm._cached_sparse_hh)]
            band = float(hmr['band_threshold'])
        weak = sum(int(np.sum(np.abs(w) <= band)) for _f, _t, w in
                   _connection_lists(algorithm._cached_sparse_hh))
        assert len(sets[1]) == len(sets[2]) == 30
        lost = sum(len(direct - expanded) for direct, expanded in zip(sets[1], sets[2]))
        assert lost == 0, f'{lost} direct connections lost at level 2'
        assert sum(len(x) for x in sets[2]) >= sum(len(x) for x in sets[1])
        # The cell exercises the case: LEO expresses connections below the band.
        assert weak > 0

    def test_chain_decay_exponent(self):
        """On a 4-node chain the 3-hop term carries decay^2 under masked, decay^3 under raw."""
        import jax.numpy as jnp
        A = np.zeros((4, 4), dtype=np.float32)
        A[0, 1] = A[1, 2] = A[2, 3] = 1.0
        valid = A > 0
        for fn in (engine.compute_multi_hop_connections_matrix_power,
                   engine.compute_multi_hop_connections_fori_loop):
            masked, valid_m = fn(jnp.asarray(A), jnp.asarray(valid), 0.1, 3, 0.5, 8.0, 'masked')
            raw, valid_r = fn(jnp.asarray(A), jnp.asarray(valid), 0.1, 3, 0.5, 8.0, 'raw')
            masked, raw = np.asarray(masked), np.asarray(raw)
            assert np.isclose(masked[0, 2], 0.5, atol=1e-6) and np.isclose(raw[0, 2], 0.5, atol=1e-6)
            assert np.isclose(masked[0, 3], 0.25, atol=1e-6), fn.__name__
            assert np.isclose(raw[0, 3], 0.125, atol=1e-6), fn.__name__
            assert bool(valid_m[0, 3]) and bool(valid_r[0, 3])
            # Direct connections are never replaced by weaker composites (max-magnitude merge).
            assert np.isclose(masked[0, 1], 1.0) and np.isclose(masked[1, 2], 1.0)

    def test_chain_masked_ignores_edges_outside_the_direct_set(self):
        """A raw weight without a valid direct connection never enters a composite under masked."""
        import jax.numpy as jnp
        W = np.zeros((3, 3), dtype=np.float32)
        W[0, 1] = 1.0
        W[1, 2] = 1.0   # present in the raw weights ...
        valid = np.zeros((3, 3), dtype=bool)
        valid[0, 1] = True  # ... but not a valid direct connection
        masked, _ = engine.compute_multi_hop_connections_matrix_power(
            jnp.asarray(W), jnp.asarray(valid), 0.1, 2, 0.5, 8.0, 'masked')
        raw, _ = engine.compute_multi_hop_connections_matrix_power(
            jnp.asarray(W), jnp.asarray(valid), 0.1, 2, 0.5, 8.0, 'raw')
        assert float(masked[0, 2]) == 0.0
        assert np.isclose(float(raw[0, 2]), 0.5, atol=1e-6)


class TestDiscoveryModeConfiguration:
    def test_default_is_per_genome_and_masked(self, algorithm, xor_problem):
        """The defaults: per_genome discovery (faster than union at every depth 1-7 on a GPU, same
        results) and the masked factor."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)
        algorithm.initialize(algorithm.create_config(config), xor_problem, seed=DEFAULT_SEED)
        assert algorithm.extended_config.hh_discovery_mode == 'per_genome'
        assert algorithm.extended_config.multi_hop_second_factor == 'masked'
        assert engine.EMRConfig().hh_discovery_mode == 'per_genome'

    def test_raw_defaults_to_union(self, algorithm, xor_problem):
        """The raw factor exists only on the union path, so an unset discovery mode resolves to union under
        it: through the preset branch, the explicit branch and direct construction."""
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['multi_hop_second_factor'] = 'raw'
        algorithm.create_config(config)
        assert algorithm.extended_config.hh_discovery_mode == 'union'
        explicit = {'enabled': True, 'allow_hidden_to_hidden': True, 'iteration_level': 2,
                    'multi_hop_second_factor': 'raw'}
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = explicit
        algorithm.create_config(config)
        assert algorithm.extended_config.hh_discovery_mode == 'union'
        assert engine.EMRConfig(multi_hop_second_factor='raw').hh_discovery_mode == 'union'
        # An explicit union stays union under the masked factor.
        assert engine.EMRConfig(hh_discovery_mode='union').hh_discovery_mode == 'union'

    def test_unknown_mode_is_refused(self, algorithm):
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['hh_discovery_mode'] = 'cached'
        with pytest.raises(ValueError):
            algorithm.create_config(config)
        # Every construction path refuses it, not only the parser (EMRConfig.__post_init__).
        with pytest.raises(ValueError):
            engine.EMRConfig(hh_discovery_mode='cached')

    def test_unknown_second_factor_is_refused(self, algorithm):
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['multi_hop_second_factor'] = 'sum'
        with pytest.raises(ValueError):
            algorithm.create_config(config)

    def test_raw_with_per_genome_is_refused(self, algorithm):
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)
        rec = config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']
        rec['hh_discovery_mode'] = 'per_genome'
        rec['multi_hop_second_factor'] = 'raw'
        with pytest.raises(ValueError):
            algorithm.create_config(config)
        with pytest.raises(ValueError):
            engine.EMRConfig(hh_discovery_mode='per_genome', multi_hop_second_factor='raw')

    def test_negative_band_threshold_is_refused(self, algorithm):
        """A negative band makes every zero cell a connection: the padded slots of the per-genome
        buckets and the positions outside a genome's active set (1,544 zero-weight duplicates
        per_genome and 129,684 outside-set connections union at band -0.01, XOR depth 3
        population 30)."""
        config = create_config_with_recurrence('hidden_only', max_depth=3, population_size=30,
                                               iteration_level=2, hh_cache_enabled=False)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['band_threshold'] = -0.01
        with pytest.raises(ValueError, match='band_threshold'):
            algorithm.create_config(config)

    def test_get_recurrence_preset_copies_every_field(self):
        """A preset keeps every field through get_recurrence_preset, the discovery settings included."""
        import dataclasses
        probe = dataclasses.replace(engine.RECURRENCE_PRESETS['hidden_only'], hh_discovery_mode='per_genome',
                                    multi_hop_second_factor='masked', use_dense_discovery=True)
        engine.RECURRENCE_PRESETS['_probe_copy'] = probe
        try:
            copy = engine.get_recurrence_preset('_probe_copy')
        finally:
            del engine.RECURRENCE_PRESETS['_probe_copy']
        assert copy is not probe
        assert dataclasses.asdict(copy) == dataclasses.asdict(probe)

    def test_raw_is_honored_through_the_preset_branch(self, algorithm, xor_problem):
        config = create_config_with_recurrence('hidden_only', hh_cache_enabled=False)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['multi_hop_second_factor'] = 'raw'
        algorithm.initialize(algorithm.create_config(config), xor_problem, seed=DEFAULT_SEED)
        assert algorithm.extended_config.multi_hop_second_factor == 'raw'


_COMPILE_WATCH = {'active': False, 'count': 0, 'registered': False}


def _watch_compiles():
    """Count XLA backend compiles while `_COMPILE_WATCH['active']` is set (jax.monitoring)."""
    if not _COMPILE_WATCH['registered']:
        def listener(event, duration, **kwargs):
            if _COMPILE_WATCH['active'] and event == '/jax/core/compile/backend_compile_duration':
                _COMPILE_WATCH['count'] += 1
        jax.monitoring.register_event_duration_secs_listener(listener)
        _COMPILE_WATCH['registered'] = True


class TestPerGenomeCompileStability:
    def test_no_compiles_inside_discovery_once_every_bucket_size_has_run(self):
        """A bucket size compiles once per run: every slice has the static shape
        (`_per_genome_rows_for(size)`, size, ...). Without fixed rows the member count of a slice
        changes almost every generation and 150 to 275 executables recompile per generation, 94-99%
        of the per-genome discovery time on a CPU (XOR, depth 4, populations 100 and 1000).
        TensorNEAT's species state flips the weak type of four leaves between generations 0 and 1,
        which retraces once, so the window starts at generation 2; a bucket size that first appears
        later compiles once in that generation and is allowed."""
        _watch_compiles()
        config = create_config_with_recurrence('hidden_only', max_depth=4, population_size=100,
                                               iteration_level=1, hh_cache_enabled=False)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['hh_discovery_mode'] = 'per_genome'
        original = engine.discover_sparse_hh_vectorized_multi_hop
        compiles, sizes_seen, new_sizes = [], set(), []

        def counted(state, cppns_transformed, h_grid, masks_A, *args, **kwargs):
            counts = np.asarray(masks_A).sum(axis=1)
            sizes = {engine._bucket_size_for(int(c)) for c in counts if c > 0}
            new_sizes.append(bool(sizes - sizes_seen))
            sizes_seen.update(sizes)
            _COMPILE_WATCH['count'] = 0
            _COMPILE_WATCH['active'] = True
            try:
                result = original(state, cppns_transformed, h_grid, masks_A, *args, **kwargs)
                jax.block_until_ready(result.weights)
            finally:
                _COMPILE_WATCH['active'] = False
            compiles.append(_COMPILE_WATCH['count'])
            return result

        engine.discover_sparse_hh_vectorized_multi_hop = counted
        try:
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
            assert algorithm.extended_config.hh_discovery_mode == 'per_genome'
            for _ in range(5):
                state, _metrics = algorithm.run_generation(state, XORProblem(), verbose=False)
        finally:
            engine.discover_sparse_hh_vectorized_multi_hop = original
        assert len(compiles) == 5 and compiles[0] > 0
        # Off the CPU one small executable inside discovery at generation 2 with no new bucket size is
        # allowed (two RTX 2080 Ti compiled one); the bursts this test guards against are tens to
        # hundreds per generation.
        allowed = 0 if _ON_CPU else 1
        for generation in range(2, 5):
            if not new_sizes[generation]:
                assert compiles[generation] <= allowed, (
                    f'generation {generation} compiled {compiles[generation]} executables inside '
                    f'discovery with no new bucket size (all generations: {compiles})'
                )


class TestDiscoveryStats:
    """`engine._last_discovery_stats` carries per-genome counts of the last discovery."""

    @pytest.mark.parametrize('mode', ['union', 'per_genome'])
    def test_counts_are_consistent(self, mode):
        hh, _f, masks, _p, _ = _run(mode, 'hidden_only', 2, 'disabled', 3, 30, generations=1)
        stats = engine._last_discovery_stats
        assert stats['hh_discovery_mode'] == mode and stats['iteration_level'] == 2
        total = stats['total_per_genome']; direct = stats['direct_per_genome']; active = stats['active_per_genome']
        assert total.shape == direct.shape == active.shape == (30,)
        assert np.array_equal(total, np.asarray(hh.num_valid))
        assert np.array_equal(active, masks.sum(axis=1))
        assert np.array_equal(stats['active_masks'], masks)
        assert stats['union_active'] == int(masks.any(axis=0).sum())
        assert np.all(direct <= total) and int(direct.sum()) > 0
        # A direct connection joins two of the genome's active positions.
        assert np.all(direct <= active * active)
        replaced = stats['replaced_per_genome']
        assert replaced.shape == (30,) and np.all(replaced <= direct)
        # Under the masked factor at level 2 the merge replaces most direct weights.
        assert int(replaced.sum()) > 0

    @pytest.mark.parametrize('mode', ['union', 'per_genome'])
    def test_a_genome_over_the_cap_refuses_after_the_record(self, mode):
        """A genome whose set exceeds max_sparse_conns raises (0.1.0 kept its first pairs in
        position order without a word). The record is written first, and a cap equal to the
        largest set runs, with no truncated genome in the metrics."""
        _run(mode, 'hidden_only', 1, 'disabled', 3, 30, generations=1)
        stats = engine._last_discovery_stats
        uncapped = stats['uncapped_per_genome'].copy()
        assert np.array_equal(uncapped, stats['total_per_genome'])
        assert int(stats['truncated_per_genome'].max()) == 0 and stats['max_sparse_conns'] == 10000
        cap = max(1, int(np.median(uncapped[uncapped > 0])) // 2)
        over = int(np.count_nonzero(uncapped > cap))
        largest = int(uncapped.max())
        assert over > 0
        with pytest.raises(ValueError, match=(
                rf'recurrence\.max_sparse_conns={cap} cannot hold the hidden-to-hidden connections '
                rf'discovered for {over} of 30 genomes \(largest set {largest}\)'
                rf'.*at least {largest}')):
            _run(mode, 'hidden_only', 1, 'disabled', 3, 30, generations=1, max_sparse_conns=cap)
        capped = engine._last_discovery_stats
        # Same initial population: the same sets before the cap, recorded before the refusal.
        assert np.array_equal(capped['uncapped_per_genome'], uncapped)
        assert np.array_equal(capped['truncated_per_genome'], np.maximum(uncapped - cap, 0))
        assert capped['max_sparse_conns'] == cap
        _h, _f, _m, _p, settings = _run(mode, 'hidden_only', 1, 'disabled', 3, 30,
                                        generations=1, max_sparse_conns=largest)
        at_cap = engine._last_discovery_stats
        assert np.array_equal(at_cap['total_per_genome'], uncapped)
        assert int(at_cap['truncated_per_genome'].max()) == 0
        assert settings['custom_metrics']['hh_truncated_genomes'] == 0
        assert settings['custom_metrics']['hh_max_uncapped'] == largest

    def test_level_one_direct_equals_total(self):
        hh, _f, _m, _p, _ = _run('union', 'hidden_only', 1, 'disabled', 3, 30, generations=1)
        stats = engine._last_discovery_stats
        assert np.array_equal(stats['direct_per_genome'], stats['total_per_genome'])
        assert int(stats['replaced_per_genome'].sum()) == 0


class TestRowBlocking:
    """Above `_PER_GENOME_BLOCK_PAIRS` pairs per call the per-genome core queries blocks of source positions in
    sequence (bucket size 1024 and up at the default budget). The values must not change: bit for bit on the CPU.
    A budget of 64 pairs forces blocking on the small buckets of a depth-3 run."""

    def test_block_rule(self):
        for size in [8 << k for k in range(11)]:  # 8 .. 8192
            rows = engine._per_genome_rows_for(size)
            block = engine._per_genome_block_for(size, rows)
            assert size % block == 0, (size, block)
            if rows * size * size <= engine._PER_GENOME_BLOCK_PAIRS:
                assert block == size, (size, rows, block)
            else:
                assert block < size and (rows * block * size <= engine._PER_GENOME_BLOCK_PAIRS or block == 1)
        assert engine._per_genome_block_for(512, 1) == 512
        assert engine._per_genome_block_for(1024, 1) == 256
        assert engine._per_genome_block_for(2048, 1) == 128
        assert engine._per_genome_block_for(4096, 1) == 64

    @pytest.mark.parametrize('preset,level,act_mode', [
        ('hidden_only', 1, 'disabled'),
        ('full_recurrent', 2, 'cppn_output'),
    ])
    def test_blocked_equals_unblocked(self, preset, level, act_mode, monkeypatch):
        engine._per_genome_discovery_cache.clear()
        hh_a, fit_a, _m, _p, settings = _run('per_genome', preset, level, act_mode, depth=3, population=30)
        blocks = []
        original = engine._per_genome_block_for

        def spy(size, rows):
            block = original(size, rows)
            blocks.append((size, rows, block))
            return block

        engine._per_genome_discovery_cache.clear()
        monkeypatch.setattr(engine, '_PER_GENOME_BLOCK_PAIRS', 64)
        monkeypatch.setattr(engine, '_per_genome_block_for', spy)
        try:
            hh_b, fit_b, _m, _p, _ = _run('per_genome', preset, level, act_mode, depth=3, population=30)
        finally:
            engine._per_genome_discovery_cache.clear()
        # Witness: the forced budget made the cores query in blocks.
        assert any(block < size for size, _rows, block in blocks), blocks
        total = _assert_same_connections(hh_a, hh_b, 'unblocked against blocked', ulps=0,
                                         atol=_atol(settings, level))
        assert total > 0
        for gen, (a, b) in enumerate(zip(fit_a, fit_b)):
            assert np.allclose(a, b, rtol=0, atol=_FIT_ATOL), (
                f'generation {gen}: fitness differs by {np.max(np.abs(a - b))}')
