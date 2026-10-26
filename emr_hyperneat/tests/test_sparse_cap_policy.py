"""The h->h cap's policies, its setter and its record.

A genome whose discovered hidden-to-hidden set exceeds ``recurrence.max_sparse_conns`` raises SparseCapExceeded under
``sparse_cap_policy`` 'refuse' (the default; a ValueError), keeps its first ``max_sparse_conns`` pairs in (source,
target) order under 'truncate' (as 0.1.0 did, which recorded the papers' recurrent runs), with a SparseCapWarning, and
under 'grow' the cap rises to hold the largest set and the population is discovered again, so the generation is the
one a run started at the grown cap computes. The substrate API follows the same policy, `set_sparse_cap` changes it
between generations, and `run_until_threshold` records the cap a run ends with.

The configuration is test_substrate_recurrent's: full recurrence at level 1, depth 3, 20 genomes, whose initial
population discovers sets of more than 40 connections.
"""
import pickle
import warnings

import numpy as np
import pytest
from conftest import (  # type: ignore[import-not-found]
    GPU_TRAJECTORY_RTOL,
    XORProblem,
    create_config_with_recurrence,
    exact_backend,
)
from emr_hyperneat import (
    SPARSE_CAP_POLICIES,
    EMRHyperNEAT,
    SparseCapExceeded,
    SparseCapWarning,
    substrate_forward,
)
from emr_hyperneat import emrhyperneat as engine

DEPTH = 3
POP = 20
SEED = 42
CAP = 40


def _hmr(config):
    return config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]


def _recurrent(**recurrence):
    config = create_config_with_recurrence("full_recurrent", max_depth=DEPTH, population_size=POP,
                                           iteration_level=1, hh_cache_enabled=False)
    _hmr(config)["recurrence"].update(recurrence)
    return config


def _setup(config):
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=SEED)
    return algo, problem, state


@pytest.fixture(autouse=True)
def _fresh_truncation_warnings():
    """'truncate' warns once per cap and largest set in a process; each test starts without that memory."""
    engine._sparse_cap_truncation_warned.clear()
    yield
    engine._sparse_cap_truncation_warned.clear()


@pytest.fixture(scope="module")
def full_sets():
    """The initial population's sets, uncapped: per-genome counts, and the arrays at a cap that holds them."""
    algo, problem, state = _setup(_recurrent())
    uncapped = np.asarray(algo.population_substrates(state).hh_uncapped)
    largest = int(uncapped.max())
    assert np.any(uncapped > CAP) and largest <= 10000
    exact, exact_problem, exact_state = _setup(_recurrent(max_sparse_conns=largest))
    _state, metrics = exact.run_generation(exact_state, exact_problem)
    return uncapped, largest, exact._cached_sparse_hh, metrics


def _same(actual, expected):
    if exact_backend():
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=GPU_TRAJECTORY_RTOL)


class TestSettings:
    def test_the_default_refuses_at_ten_thousand(self):
        algo, _problem, _state = _setup(_recurrent())
        config = algo.extended_config
        assert (config.max_sparse_conns, config.sparse_cap_policy, config.max_sparse_conns_ceiling) == (
            10000, "refuse", None)
        assert SPARSE_CAP_POLICIES == ("refuse", "truncate", "grow")
        assert issubclass(SparseCapExceeded, ValueError) and issubclass(SparseCapWarning, UserWarning)

    @pytest.mark.parametrize("recurrence", [
        {"sparse_cap_policy": "truncate", "max_sparse_conns": 500},
        {"sparse_cap_policy": "grow", "max_sparse_conns": 500, "max_sparse_conns_ceiling": 2000},
    ])
    def test_the_configuration_sets_them(self, recurrence):
        algo, _problem, _state = _setup(_recurrent(**recurrence))
        for key, value in recurrence.items():
            assert getattr(algo.extended_config, key) == value
        substrate_config = algo.substrate_config()
        assert substrate_config.sparse_cap_policy == recurrence["sparse_cap_policy"]
        assert substrate_config.max_sparse_conns_ceiling == recurrence.get("max_sparse_conns_ceiling")

    @pytest.mark.parametrize("recurrence, message", [
        ({"sparse_cap_policy": "drop"}, r"sparse_cap_policy must be one of"),
        ({"max_sparse_conns_ceiling": 50000}, r"max_sparse_conns_ceiling bounds sparse_cap_policy 'grow' only"),
        ({"sparse_cap_policy": "truncate", "max_sparse_conns_ceiling": 50000},
         r"max_sparse_conns_ceiling bounds sparse_cap_policy 'grow' only"),
        ({"sparse_cap_policy": "grow", "max_sparse_conns_ceiling": 5000},
         r"max_sparse_conns_ceiling must be an integer of at least max_sparse_conns \(10000\)"),
        ({"max_sparse_conns": 0}, r"max_sparse_conns must be an integer of at least 1"),
        ({"max_sparse_conns": True}, r"max_sparse_conns must be an integer of at least 1"),
        ({"sparse_cap_policy": "grow", "hh_cache_enabled": True},
         r"sparse_cap_policy 'grow' cannot be combined with hh_cache_enabled"),
    ])
    def test_what_discovery_cannot_apply_is_refused(self, recurrence, message):
        with pytest.raises(ValueError, match=message):
            _setup(_recurrent(**recurrence))

    def test_grown_cap_rule(self):
        assert engine.grown_sparse_cap(41, None) == 1024
        assert engine.grown_sparse_cap(10001, None) == 13312          # 12,501 rounded up to 1,024
        assert engine.grown_sparse_cap(10001, 12000) == 12000         # at most the ceiling
        assert engine.grown_sparse_cap(12001, 12000) is None          # beyond it: refused
        assert engine.grown_sparse_cap(4096, 4096) == 4096


class TestDiscoveryPolicies:
    def test_refuse_names_the_value_that_holds_every_set(self, full_sets):
        uncapped, largest, _full, _metrics = full_sets
        algo, problem, state = _setup(_recurrent(max_sparse_conns=CAP))
        with pytest.raises(SparseCapExceeded, match=rf"max_sparse_conns={CAP} cannot hold .*at least {largest}") as err:
            algo.run_generation(state, problem)
        assert (err.value.cap, err.value.largest) == (CAP, largest)
        # Which genomes, so a caller can leave them out: every genome's count, and the indices over the cap
        np.testing.assert_array_equal(err.value.uncapped, uncapped)
        np.testing.assert_array_equal(err.value.over, np.flatnonzero(uncapped > CAP))
        assert algo.extended_config.max_sparse_conns == CAP
        # It crosses a process boundary whole
        copy = pickle.loads(pickle.dumps(err.value))
        assert isinstance(copy, SparseCapExceeded) and str(copy) == str(err.value)
        np.testing.assert_array_equal(copy.over, err.value.over)

    def test_truncate_keeps_each_genomes_first_pairs(self, full_sets):
        uncapped, largest, full, _metrics = full_sets
        algo, problem, state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="truncate"))
        with pytest.warns(SparseCapWarning, match=rf"max_sparse_conns={CAP} truncated .*largest set {largest}"):
            _state, metrics = algo.run_generation(state, problem)
        kept = algo._cached_sparse_hh
        assert kept.from_indices.shape == (POP, CAP)
        np.testing.assert_array_equal(np.asarray(kept.num_valid), np.minimum(uncapped, CAP))
        # The first CAP slots of the whole set, in the extraction's (source, target) order
        np.testing.assert_array_equal(np.asarray(kept.from_indices), np.asarray(full.from_indices)[:, :CAP])
        np.testing.assert_array_equal(np.asarray(kept.to_indices), np.asarray(full.to_indices)[:, :CAP])
        _same(kept.weights, np.asarray(full.weights)[:, :CAP])
        custom = metrics.custom_metrics
        assert custom["hh_truncated_genomes"] == int(np.count_nonzero(uncapped > CAP))
        assert (custom["hh_max_uncapped"], custom["hh_max_sparse_conns"]) == (largest, CAP)
        assert algo.extended_config.max_sparse_conns == CAP

    def test_truncate_warns_again_only_for_a_larger_set(self, full_sets):
        algo, problem, state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="truncate"))
        with pytest.warns(SparseCapWarning):
            algo.run_generation(state, problem)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseCapWarning)
            # The same population again: no set larger than the one already reported
            algo.run_generation(state, problem)

    def test_grow_runs_the_generation_a_run_at_the_grown_cap_runs(self, full_sets):
        uncapped, largest, full, full_metrics = full_sets
        algo, problem, state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="grow"))
        grown = engine.grown_sparse_cap(largest, None)
        with pytest.warns(SparseCapWarning, match=rf"max_sparse_conns grew from {CAP} to {grown}"):
            _state, metrics = algo.run_generation(state, problem)
        assert algo.extended_config.max_sparse_conns == grown
        kept = algo._cached_sparse_hh
        assert kept.from_indices.shape == (POP, grown)
        np.testing.assert_array_equal(np.asarray(kept.num_valid), uncapped)
        np.testing.assert_array_equal(np.asarray(kept.from_indices)[:, :largest], np.asarray(full.from_indices))
        np.testing.assert_array_equal(np.asarray(kept.to_indices)[:, :largest], np.asarray(full.to_indices))
        assert np.all(np.asarray(kept.from_indices)[:, largest:] == -1)
        _same(np.asarray(kept.weights)[:, :largest], full.weights)
        # The width beyond the largest set is padding: the population scores as at the cap that just holds it
        _same([metrics.best_fitness, metrics.mean_fitness], [full_metrics.best_fitness, full_metrics.mean_fitness])
        custom = metrics.custom_metrics
        assert (custom["hh_truncated_genomes"], custom["hh_max_sparse_conns"]) == (0, grown)
        assert engine._last_discovery_stats["sparse_cap_grown_from"] == CAP

    def test_grow_stops_at_its_ceiling(self, full_sets):
        _uncapped, largest, _full, _metrics = full_sets
        algo, problem, state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="grow",
                                                 max_sparse_conns_ceiling=largest - 1))
        with pytest.raises(SparseCapExceeded, match=rf"stops at max_sparse_conns_ceiling={largest - 1}"):
            algo.run_generation(state, problem)
        assert algo.extended_config.max_sparse_conns == CAP

    def test_grow_reaches_a_ceiling_that_holds_the_largest_set(self, full_sets):
        _uncapped, largest, _full, _metrics = full_sets
        algo, problem, state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="grow",
                                                 max_sparse_conns_ceiling=largest))
        with pytest.warns(SparseCapWarning, match=rf"grew from {CAP} to {largest}"):
            algo.run_generation(state, problem)
        assert algo.extended_config.max_sparse_conns == largest

    @pytest.mark.parametrize("mode", ["streaming", "depth_offload"])
    def test_every_execution_mode_applies_the_policy(self, full_sets, mode):
        uncapped, largest, _full, _metrics = full_sets
        config = _recurrent(max_sparse_conns=CAP, sparse_cap_policy="grow")
        _hmr(config)["execution"] = {"mode": mode}
        algo, problem, state = _setup(config)
        with pytest.warns(SparseCapWarning, match=r"max_sparse_conns grew"):
            algo.run_generation(state, problem)
        assert algo.extended_config.max_sparse_conns == engine.grown_sparse_cap(largest, None)
        refusing = _recurrent(max_sparse_conns=CAP)
        _hmr(refusing)["execution"] = {"mode": mode}
        algo, problem, state = _setup(refusing)
        with pytest.raises(SparseCapExceeded):
            algo.run_generation(state, problem)


class TestSetter:
    def test_it_changes_the_next_discovery(self, full_sets):
        uncapped, largest, _full, _metrics = full_sets
        algo, problem, state = _setup(_recurrent())
        algo.set_sparse_cap(max_sparse_conns=CAP, sparse_cap_policy="truncate")
        with pytest.warns(SparseCapWarning):
            _state, metrics = algo.run_generation(state, problem)
        assert metrics.custom_metrics["hh_max_sparse_conns"] == CAP
        algo.set_sparse_cap(sparse_cap_policy="grow", max_sparse_conns_ceiling=largest)
        with pytest.warns(SparseCapWarning, match=rf"grew from {CAP} to {largest}"):
            algo.run_generation(state, problem)
        assert algo.extended_config.max_sparse_conns == largest

    def test_it_refuses_what_the_configuration_refuses_and_changes_nothing(self):
        algo, _problem, _state = _setup(_recurrent(sparse_cap_policy="grow", max_sparse_conns_ceiling=20000))
        with pytest.raises(ValueError, match=r"max_sparse_conns_ceiling bounds sparse_cap_policy 'grow' only"):
            algo.set_sparse_cap(sparse_cap_policy="refuse")
        assert algo.extended_config.sparse_cap_policy == "grow"
        algo.set_sparse_cap(sparse_cap_policy="refuse", max_sparse_conns_ceiling=None)
        assert (algo.extended_config.sparse_cap_policy, algo.extended_config.max_sparse_conns_ceiling) == (
            "refuse", None)
        with pytest.raises(ValueError, match=r"give max_sparse_conns, sparse_cap_policy or max_sparse_conns_ceiling"):
            algo.set_sparse_cap()
        with pytest.raises(ValueError, match=r"max_sparse_conns must be an integer of at least 1"):
            algo.set_sparse_cap(max_sparse_conns=0)
        assert algo.extended_config.max_sparse_conns == 10000

    def test_a_feed_forward_configuration_has_no_cap_to_set(self):
        config = create_config_with_recurrence("full_recurrent", max_depth=DEPTH, population_size=POP,
                                               iteration_level=1, hh_cache_enabled=False)
        _hmr(config)["recurrence"] = {"preset": "feedforward", "iteration_level": 0}
        algo, _problem, _state = _setup(config)
        with pytest.raises(ValueError, match=r"set_sparse_cap needs a configuration with hidden-to-hidden"):
            algo.set_sparse_cap(max_sparse_conns=100)


class TestRunRecord:
    def test_grow_records_where_the_cap_went(self, full_sets):
        _uncapped, largest, _full, _metrics = full_sets
        algo, problem, state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="grow"))
        with pytest.warns(SparseCapWarning):
            result = algo.run_until_threshold(state, problem, target_fitness=2.0, max_generations=2)
        record = result["sparse_cap"]
        grown = record["max_sparse_conns"]
        assert grown == algo.extended_config.max_sparse_conns >= largest
        assert record["configured_max_sparse_conns"] == CAP and record["sparse_cap_policy"] == "grow"
        assert record["grown"][0] == [CAP, engine.grown_sparse_cap(largest, None)]
        assert record["largest_set"] >= largest and record["truncated_discoveries"] == 0

    def test_truncate_records_its_truncations(self, full_sets):
        algo, problem, state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="truncate"))
        with pytest.warns(SparseCapWarning):
            result = algo.run_until_threshold(state, problem, target_fitness=2.0, max_generations=2)
        record = result["sparse_cap"]
        assert record["max_sparse_conns"] == CAP and record["grown"] == []
        assert record["truncated_discoveries"] >= 1 and record["largest_set"] > CAP

    def test_a_feed_forward_run_has_no_record(self):
        config = create_config_with_recurrence("full_recurrent", max_depth=DEPTH, population_size=POP,
                                               iteration_level=1, hh_cache_enabled=False)
        _hmr(config)["recurrence"] = {"preset": "feedforward", "iteration_level": 0}
        algo, problem, state = _setup(config)
        result = algo.run_until_threshold(state, problem, target_fitness=2.0, max_generations=1)
        assert "sparse_cap" not in result


class TestSubstrateAPI:
    def test_the_three_policies(self, full_sets):
        uncapped, largest, full, _metrics = full_sets
        refusing, _p, refusing_state = _setup(_recurrent(max_sparse_conns=CAP))
        with pytest.raises(SparseCapExceeded, match=rf"at least {largest}") as err:
            refusing.population_substrates(refusing_state)
        np.testing.assert_array_equal(err.value.uncapped, uncapped)
        np.testing.assert_array_equal(err.value.over, np.flatnonzero(uncapped > CAP))

        truncating, _p, truncating_state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="truncate"))
        with pytest.warns(SparseCapWarning):
            truncated = truncating.population_substrates(truncating_state)
        assert truncated.config.max_sparse_conns == CAP
        np.testing.assert_array_equal(np.asarray(truncated.hh_uncapped), uncapped)
        np.testing.assert_array_equal(np.asarray(truncated.hh_valid).sum(axis=1), np.minimum(uncapped, CAP))
        np.testing.assert_array_equal(np.asarray(truncated.hh_from), np.asarray(full.from_indices)[:, :CAP])

        growing, _p, growing_state = _setup(_recurrent(max_sparse_conns=CAP, sparse_cap_policy="grow"))
        with pytest.warns(SparseCapWarning, match=r"grew"):
            grown = growing.population_substrates(growing_state)
        # The substrate's configuration records the cap it was built at; the engine's is the loop's to raise
        assert grown.config.max_sparse_conns == engine.grown_sparse_cap(largest, None)
        assert growing.extended_config.max_sparse_conns == CAP
        np.testing.assert_array_equal(np.asarray(grown.hh_valid).sum(axis=1), uncapped)
        np.testing.assert_array_equal(np.asarray(grown.hh_from)[:, :largest], np.asarray(full.from_indices))
        # The width beyond the largest set is padding: the grown substrate's forward is the one at the cap that just
        # holds every set (emr-picbreeder found widths 10,000 and 131,072 draw the same pictures)
        exact_algo, _p, exact_state = _setup(_recurrent(max_sparse_conns=largest))
        exact = exact_algo.population_substrates(exact_state)
        assert exact.config.max_sparse_conns == largest < grown.config.max_sparse_conns
        inputs = growing._cached_inputs
        _same(substrate_forward(grown, inputs), substrate_forward(exact, inputs))
