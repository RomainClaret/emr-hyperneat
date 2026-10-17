"""Substrate depth controls: initial_depth forcing, lookahead budget, dead levels.

Substrate depth used to be governed solely by variance_threshold: initial_depth
was parsed and never read, and the variance lookahead substituted zeros for
levels larger than its position budget, silently truncating the substrate.
These tests pin the repaired contracts:

- initial_depth=0 (the default) reproduces the variance-only substrate exactly,
  on every execution mode;
- initial_depth=k forces full occupancy, 4^(l+1) x population positions, on
  every level l <= k -- on STANDARD, STREAMING, and DEPTH_OFFLOAD alike, and
  the three modes agree on reach and fitness while forcing (reach exactly;
  fitness bitwise on CPU and to float32 rounding on a GPU);
- max_lookahead_positions is a MEMORY budget on the offload path: any budget
  yields the same substrate (oversized levels are computed in parent-row
  slabs), unless truncate_variance_lookahead=True explicitly requests the old
  zeroing;
- skip_dead_levels changes cost, never reach or fitness;
- division_threshold warns: nothing on the eager multi-resolution path reads it.

Assertions are on OCCUPANCY (_last_reached_per_level), not the deepest live
level: forcing fills levels completely while variance still prunes immediately
below, so the deepest live level is not monotonic in initial_depth; occupancy
is, and occupancy is what forcing controls.
"""

import warnings

import pytest

from conftest import DEFAULT_SEED, XORProblem, assert_trajectories_agree
from emr_hyperneat.emrhyperneat import (  # noqa: E501
    EMRHyperNEAT,
)
from emr_hyperneat.execution import (  # noqa: E501
    ExecutionMode,
)

MAX_DEPTH = 5
POPULATION = 20
MODES = ['standard', 'streaming', 'depth_offload']


def _make_config(mode='depth_offload', variance_threshold=0.03, initial_depth=None,
                 division_threshold=None, execution_extra=None):
    """Config with only the knobs under test; execution mode explicit."""
    hmr = {
        'max_depth': MAX_DEPTH,
        'variance_threshold': variance_threshold,
        'band_threshold': 0.3,
        'max_weight': 3.0,
        'verbose': False,
        'recurrence': {},
        'dynamic_functions': {},
        'aggregation': {},
        'neuromodulation': {},
        'execution': {'mode': mode, **(execution_extra or {})},
    }
    if initial_depth is not None:
        hmr['initial_depth'] = initial_depth
    if division_threshold is not None:
        hmr['division_threshold'] = division_threshold
    return {
        'algorithm_params': {
            'emrhyperneat': {
                'population_size': POPULATION,
                'emr_hyperneat': hmr,
                'substrate': {
                    'input_coords': [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],
                    'output_coords': [(0.0, 1.0)],
                },
            }
        }
    }


def _reach(config, expected_mode):
    """One generation; returns (per-level reach counts, best fitness)."""
    algo = EMRHyperNEAT()
    cfg = algo.create_config(config)
    assert algo.resolve_execution_mode() is expected_mode
    problem = XORProblem()
    state = algo.initialize(cfg, problem, seed=DEFAULT_SEED)
    state, metrics = algo.run_generation(state, problem)
    return list(algo._last_reached_per_level), float(metrics.best_fitness)


_MODE_ENUM = {
    'standard': ExecutionMode.STANDARD,
    'streaming': ExecutionMode.STREAMING,
    'depth_offload': ExecutionMode.DEPTH_OFFLOAD,
}


class TestInitialDepthForcing:
    """initial_depth forces subdivision uniformly; the default cannot fire."""

    @pytest.mark.parametrize("mode", MODES)
    def test_default_equals_absent_and_variance_only(self, mode):
        """initial_depth=0 == key absent: level 0 full, forcing inert."""
        reach_absent, fit_absent = _reach(_make_config(mode=mode), _MODE_ENUM[mode])
        reach_zero, fit_zero = _reach(
            _make_config(mode=mode, initial_depth=0), _MODE_ENUM[mode])
        assert reach_absent == reach_zero
        assert fit_absent == fit_zero
        # Level 0 is always fully reached: 4 positions per genome.
        assert reach_absent[0] == 4 * POPULATION

    @pytest.mark.parametrize("mode", MODES)
    def test_initial_depth_forces_full_occupancy(self, mode):
        """Forced levels hold exactly 4^(l+1) x population positions, every mode."""
        reach, _ = _reach(_make_config(mode=mode, initial_depth=2), _MODE_ENUM[mode])
        for level in range(3):  # levels 0..2 forced by initial_depth=2
            assert reach[level] == (4 ** (level + 1)) * POPULATION, (
                f"[{mode}] level {level}: {reach[level]} != "
                f"{(4 ** (level + 1)) * POPULATION}"
            )

    def test_forcing_agrees_across_modes(self):
        """With forcing live everywhere, the modes still produce one result."""
        results = {
            mode: _reach(_make_config(mode=mode, initial_depth=2), _MODE_ENUM[mode])
            for mode in MODES
        }
        assert len({tuple(reach) for reach, _ in results.values()}) == 1, results
        # reach is a count and agrees exactly on every backend; the fitness agrees
        # bitwise on CPU and to float32 rounding on a GPU (1 ulp on an RTX 2080 Ti)
        assert_trajectories_agree(
            {mode: [fit] for mode, (_, fit) in results.items()}, "forcing across modes")

    def test_forcing_is_live_not_vacuous(self):
        """initial_depth=2 must strictly grow occupancy over the default."""
        reach_default, _ = _reach(_make_config(), ExecutionMode.DEPTH_OFFLOAD)
        reach_forced, _ = _reach(
            _make_config(initial_depth=2), ExecutionMode.DEPTH_OFFLOAD)
        assert sum(reach_forced) > sum(reach_default)


class TestLookaheadBudget:
    """max_lookahead_positions bounds memory on the offload path; it must not
    change the substrate."""

    def test_budget_is_a_memory_knob(self):
        """Tiny and huge budgets produce the identical substrate and fitness.

        variance_threshold=0.0 keeps every level live so the tiny budget's slab
        path genuinely engages -- at the production threshold subdivision dies
        at level 2-3 and the invariance would hold vacuously.
        """
        tiny = _reach(_make_config(
            variance_threshold=0.0,
            execution_extra={'max_lookahead_positions': 1000}),
            ExecutionMode.DEPTH_OFFLOAD)
        huge = _reach(_make_config(
            variance_threshold=0.0,
            execution_extra={'max_lookahead_positions': 10_000_000}),
            ExecutionMode.DEPTH_OFFLOAD)
        assert tiny[0] == huge[0], (tiny, huge)
        assert_trajectories_agree(
            {'tiny': [tiny[1]], 'huge': [huge[1]]}, "lookahead budget")

    def test_truncate_restores_zeroing(self):
        """truncate_variance_lookahead=True is the old cap: a smaller substrate."""
        slabbed, _ = _reach(_make_config(
            variance_threshold=0.0,
            execution_extra={'max_lookahead_positions': 1000}),
            ExecutionMode.DEPTH_OFFLOAD)
        truncated, _ = _reach(_make_config(
            variance_threshold=0.0,
            execution_extra={'max_lookahead_positions': 1000,
                             'truncate_variance_lookahead': True}),
            ExecutionMode.DEPTH_OFFLOAD)
        assert sum(truncated) < sum(slabbed)


class TestSkipDeadLevels:
    """skip_dead_levels changes cost, never results."""

    def test_noop_on_reach_and_fitness(self):
        # Explicit on both sides: the default is True and the reported runtimes
        # were measured with it off, so neither arm relies on the default.
        base = _reach(_make_config(
            execution_extra={'skip_dead_levels': False}),
            ExecutionMode.DEPTH_OFFLOAD)
        skipped = _reach(_make_config(
            execution_extra={'skip_dead_levels': True}),
            ExecutionMode.DEPTH_OFFLOAD)
        assert base[0] == skipped[0], (base, skipped)
        assert_trajectories_agree(
            {'base': [base[1]], 'skipped': [skipped[1]]}, "skip_dead_levels")


class TestVarianceRule:
    """variance_rule: 'canonical' (default) unifies the modes; 'legacy' reproduces
    the original standard path."""

    def _cfg(self, rule=None, mode='standard'):
        cfg = _make_config(mode=mode)
        if rule is not None:
            cfg['algorithm_params']['emrhyperneat']['emr_hyperneat']['variance_rule'] = rule
        return cfg

    def test_default_is_canonical_and_equals_streaming(self):
        std = _reach(self._cfg(), ExecutionMode.STANDARD)
        strm = _reach(self._cfg(mode='streaming'), ExecutionMode.STREAMING)
        assert std[0] == strm[0], (std, strm)
        assert_trajectories_agree(
            {'standard': [std[1]], 'streaming': [strm[1]]},
            "canonical standard vs streaming")

    def test_legacy_is_all_or_nothing_at_level_one(self):
        """Under the legacy rule every genome has 0 or all 16 level-1 positions,
        so the level-1 total is a multiple of 16 -- an integer property that holds
        on every platform (the exact counts are platform-conditional)."""
        reach, _ = _reach(self._cfg(rule='legacy'), ExecutionMode.STANDARD)
        assert reach[1] % 16 == 0
        assert reach[1] > 0

    @pytest.mark.parametrize('mode, chunked', [('streaming', False), ('depth_offload', False),
                                               ('depth_offload', True)])
    def test_legacy_is_the_same_substrate_on_every_mode(self, mode, chunked, monkeypatch):
        """Streaming and depth_offload apply the legacy level-0 rule too; in 0.1.0 they ran the
        canonical rule under `legacy`. `chunked` sends depth_offload through its
        population-chunked loop (max_population_chunk below the population), the third place the
        rule is read; the spy shows that loop ran."""
        std = _reach(self._cfg(rule='legacy'), ExecutionMode.STANDARD)
        cfg = self._cfg(rule='legacy', mode=mode)
        chunked_calls = []
        if chunked:
            cfg['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'][
                'max_population_chunk'] = POPULATION // 2
            from emr_hyperneat.execution import offload
            original = offload._run_generation_population_chunked

            def spy(*args, **kwargs):
                chunked_calls.append(1)
                return original(*args, **kwargs)
            monkeypatch.setattr(offload, '_run_generation_population_chunked', spy)
        other = _reach(cfg, _MODE_ENUM[mode])
        assert len(chunked_calls) == int(chunked)
        assert std[0] == other[0], (std, other)
        assert other[0][1] % 16 == 0
        assert_trajectories_agree(
            {'standard': [std[1]], mode: [other[1]]},
            f"legacy standard vs {mode} (chunked={chunked})")

    def test_legacy_differs_from_canonical(self):
        legacy = _reach(self._cfg(rule='legacy'), ExecutionMode.STANDARD)
        canonical = _reach(self._cfg(rule='canonical'), ExecutionMode.STANDARD)
        assert legacy != canonical

    def test_invalid_rule_raises(self):
        algo = EMRHyperNEAT()
        with pytest.raises(ValueError, match='variance_rule'):
            algo.create_config(self._cfg(rule='bogus'))


class TestDivisionThresholdWarning:
    """division_threshold does nothing on the eager path; supplying it warns."""

    def test_supplied_warns(self):
        algo = EMRHyperNEAT()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            algo.create_config(_make_config(division_threshold=0.03))
        assert any(
            issubclass(w.category, DeprecationWarning)
            and 'division_threshold' in str(w.message)
            for w in caught
        )

    def test_absent_stays_silent(self):
        algo = EMRHyperNEAT()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            algo.create_config(_make_config())
        assert not any(
            'division_threshold' in str(w.message) for w in caught
        )
