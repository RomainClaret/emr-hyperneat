"""Regression tests that EMR's recurrence actually engages and stays consistent.

These tests guard against recurrence that is configured but never engages: a recurrent
path that reads a `_recurrence_config` which `create_config` never set would silently
degrade a YAML-configured recurrent run to feedforward, depth offload included, and
`hasattr` checks pass just as happily on a value that is never populated.

So the assertions here are deliberately about *effects*, not presence: that the
forward mode resolves, that recurrent and feedforward produce different answers,
that discovered connections are non-zero, and that the three execution modes agree
with each other. A test that only checks an attribute exists cannot distinguish
working recurrence from a dataclass default.

The compaction equivalence tests are here rather than in test_compaction.py because
they are about the recurrent path as a whole, across the feature combinations that
contribute per-position arrays to evaluation.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    XORProblem,
    assert_trajectories_agree,
    create_base_config,
    create_config_with_aggregation,
    create_config_with_dynamic_functions,
    create_config_with_neuromodulation,
    create_config_with_recurrence,
)
from emr_hyperneat import (
    EMRHyperNEAT,
)
from emr_hyperneat.execution import ExecutionMode
from emr_hyperneat.emrhyperneat import (
    ForwardPassMode,
)
from emr_hyperneat.execution import (
    compaction as compaction_module,
    offload as offload_module,
)

MODES = [ExecutionMode.STANDARD, ExecutionMode.STREAMING, ExecutionMode.DEPTH_OFFLOAD]
SEED = 7
DEPTH = 3
POPULATION = 8
# At the default genome_elitism of 15 a population of 8 is copied unmutated, so every later
# generation would repeat generation 0's genomes; with 2 elites, 6 genomes mutate.
ELITISM = 2


def evolving(config, elitism: int = ELITISM):
    """``config`` with neat_selection.genome_elitism set, so that later generations mutate."""
    config["algorithm_params"]["emrhyperneat"]["neat_selection"] = {"genome_elitism": elitism}
    return config


def feedforward_config(depth: int = DEPTH, population: int = POPULATION):
    """Base config with recurrence explicitly off, and elitism 2 (ELITISM)."""
    config = evolving(create_base_config(max_depth=depth, population_size=population))
    emr = config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]
    emr["recurrence"] = {"preset": "feedforward", "iteration_level": 0}
    return config


def recurrent_config(depth: int = DEPTH, population: int = POPULATION):
    """hidden_only recurrence at iteration level 2, and elitism 2 (ELITISM)."""
    return evolving(create_config_with_recurrence(
        "hidden_only", max_depth=depth, population_size=population, iteration_level=2
    ))


def run_generations(config, mode, generations: int = 2, seed: int = SEED):
    """Run a few generations and return (algorithm, per-generation best fitness)."""
    algorithm = EMRHyperNEAT()
    algorithm.execution_config.mode = mode
    problem = XORProblem()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=seed)
    fitness = []
    for _ in range(generations):
        state, metrics = algorithm.run_generation(state, problem)
        fitness.append(round(float(metrics.best_fitness), 9))
    return algorithm, fitness


def told_fitness(config, mode, generations: int = 2, seed: int = SEED):
    """Every generation's whole fitness vector, as tell received it, in one list.

    The runs use exact compaction shapes (execution.compaction_bucket 'exact'): on x86 a
    bucket-padded sum can round a genome's fitness by a float32 ulp (test_depth_offload.py), and
    on CPU this comparison is bitwise.
    """
    config = copy.deepcopy(config)
    emr = config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]
    emr.setdefault("execution", {})["compaction_bucket"] = "exact"
    algorithm = EMRHyperNEAT()
    algorithm.execution_config.mode = mode
    problem = XORProblem()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=seed)
    told, tell = [], algorithm._compiled_tell

    def spy_tell(state_, fitnesses):
        told.append(np.asarray(fitnesses))
        return tell(state_, fitnesses)

    algorithm._compiled_tell = spy_tell
    for _ in range(generations):
        state, _ = algorithm.run_generation(state, problem)
    assert len(told) == generations, f"{mode.value}: tell ran {len(told)} times"
    return [float(value) for fitnesses in told for value in fitnesses]


class TestRecurrenceEngages:
    """The forward mode must follow the config, in every execution mode."""

    @pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
    def test_recurrent_config_selects_the_hybrid_forward(self, mode):
        algorithm, _ = run_generations(recurrent_config(), mode, generations=1)
        assert algorithm.get_forward_mode() == ForwardPassMode.HYBRID_SPARSE_HH

    @pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
    def test_feedforward_config_selects_the_dense_forward(self, mode):
        algorithm, _ = run_generations(feedforward_config(), mode, generations=1)
        assert algorithm.get_forward_mode() == ForwardPassMode.DENSE_ONLY

    @pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
    def test_recurrent_changes_the_answer(self, mode):
        """The assertion a `hasattr` check cannot make.

        If recurrence silently degraded to feedforward, these would be equal.
        """
        _, recurrent = run_generations(recurrent_config(), mode)
        _, feedforward = run_generations(feedforward_config(), mode)
        assert recurrent != feedforward, (
            f"{mode.value}: recurrent produced the feedforward trajectory, so h->h "
            f"contributed nothing"
        )

    def test_activate_time_is_the_derived_value_not_the_raw_config(self):
        """EMRConfig.activate_time is Optional and defaults to None.

        create_config derives it; reading the raw field gave None and raised
        TypeError on the offload path for exactly this reason.
        """
        recurrent, _ = run_generations(recurrent_config(), ExecutionMode.STANDARD, 1)
        feedforward, _ = run_generations(feedforward_config(), ExecutionMode.STANDARD, 1)
        assert recurrent.activate_time is not None
        assert recurrent.activate_time > feedforward.activate_time

    def test_discovered_connections_are_non_zero(self):
        """The count itself, not `hasattr`.

        Only the offload path records this counter, which is why the mode is pinned
        here rather than parametrized.
        """
        algorithm, _ = run_generations(
            recurrent_config(), ExecutionMode.DEPTH_OFFLOAD, generations=1
        )
        assert algorithm._last_hh_connections > 0, (
            "no hidden-to-hidden connections were discovered, so recurrence was "
            "structurally absent even though the forward mode reported hybrid"
        )


class TestModesAgree:
    """Execution mode is a memory-residency choice and must not change results.

    Each generation's whole fitness vector (told_fitness), with elitism 2 so that the modes must
    also breed the same offspring: bitwise on CPU; on a GPU, to float32 rounding
    (conftest.assert_trajectories_agree): the standard path and the level-by-level paths reduce
    over different extents, and GPU kernels round per shape. On an RTX 2080 Ti the depth-3
    recurrent case differed by 2 ulp at generation 1 (best fitness, elitism 15). In 0.1.0
    depth_offload split the random key before every tell, which the standard path does
    only under extra_randkey_split, and bred another population from generation 1 on.
    """

    def test_feedforward_identical_across_modes(self):
        results = {m.value: told_fitness(feedforward_config(), m) for m in MODES}
        assert_trajectories_agree(results, "feedforward across execution modes")

    def test_recurrent_identical_across_modes_at_depth_two(self):
        results = {m.value: told_fitness(recurrent_config(depth=2), m) for m in MODES}
        assert_trajectories_agree(results, "recurrent depth 2 across execution modes")

    def test_streaming_and_offload_agree_at_depth_three(self):
        """Streaming and depth_offload are one computation at d3: a regression guard.

        The offload path increments _current_generation and consults the h->h cache
        manager; with neither, offload repeats its generation-0 fitness and the modes
        split at generation 1. The two modes produce one trajectory at depth 3, pop 8,
        seed 7 (the assertion compares the modes, not recorded values).
        """
        results = {
            m.value: told_fitness(recurrent_config(depth=3), m)
            for m in (ExecutionMode.STREAMING, ExecutionMode.DEPTH_OFFLOAD)
        }
        assert_trajectories_agree(results, "streaming vs depth_offload at depth 3")

    def test_recurrent_identical_across_modes_at_depth_three(self):
        """All three modes are one computation under the default variance rule.

        Under the default variance_rule 'canonical', the standard family computes its
        level-0 subdivision variance per parent over the four level-1 children, as
        streaming and depth_offload do, and reaches the same substrate ([32, 40, 12, 0])
        and the same fitness trajectory at depth 3, pop 8, seed 7 on Apple Silicon.
        'legacy', kept for reproducing published standard-path runs (TestVarianceRule),
        computes it from level 0's own four weights as a single per-genome scalar
        (all-or-nothing subdivision of the whole level).
        """
        results = {m.value: told_fitness(recurrent_config(depth=3), m) for m in MODES}
        assert_trajectories_agree(results, "recurrent depth 3 across execution modes")

    @pytest.mark.parametrize("build", [feedforward_config, recurrent_config],
                             ids=["feedforward", "recurrent"])
    def test_population_chunked_offload_identical_to_standard(self, build, monkeypatch):
        """depth_offload's population-chunked loop (execution.max_population_chunk below the
        population) breeds what the standard path breeds; the spy shows the loop ran. The h->h
        cache is off (EMR's default; conftest's recurrent preset turns it on): the chunked loop
        never consults the population-level cache manager, so with the cache on it rediscovers
        where the other routes reuse."""
        config = build()
        if build is recurrent_config:
            config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["recurrence"][
                "hh_cache_enabled"] = False
        chunked = copy.deepcopy(config)
        chunked["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["execution"] = {
            "max_population_chunk": POPULATION // 2}
        calls = []
        original = offload_module._run_generation_population_chunked

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(offload_module, "_run_generation_population_chunked", spy)
        results = {
            "standard": told_fitness(config, ExecutionMode.STANDARD),
            "population_chunked": told_fitness(chunked, ExecutionMode.DEPTH_OFFLOAD),
        }
        assert len(calls) == 2, "the population-chunked loop did not run"
        assert_trajectories_agree(results, "standard vs depth_offload's population-chunked loop")

    def test_population_chunked_offload_refuses_the_hh_cache(self):
        """With the h->h cache on (conftest's recurrent preset), depth_offload's population-chunked
        loop raises at the route's entry, before generation 0 asks for its population: it never
        consults the population-level cache manager, so it would rediscover every generation and
        silently differ from the unchunked loop. The unchunked loop still runs with the cache on."""
        config = recurrent_config()
        emr = config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]
        assert emr["recurrence"]["hh_cache_enabled"] is True
        chunked = copy.deepcopy(config)
        chunked["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["execution"] = {
            "max_population_chunk": POPULATION // 2}
        algorithm = EMRHyperNEAT()
        algorithm.execution_config.mode = ExecutionMode.DEPTH_OFFLOAD
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(chunked), problem, seed=SEED)
        asks, ask = [], algorithm._compiled_ask

        def counted_ask(state_):
            asks.append(1)
            return ask(state_)

        algorithm._compiled_ask = counted_ask
        with pytest.raises(ValueError, match="population-level"):
            algorithm.run_generation(state, problem)
        assert not asks, "the refusal came after generation 0's ask"
        assert len(told_fitness(config, ExecutionMode.DEPTH_OFFLOAD, generations=1)) == POPULATION


class TestCompactionEquivalence:
    """Compaction must not move results, under any feature that adds per-position state.

    The standard recurrent path derives act_indices, aggregation indices and every
    per-position neuromodulation array from W1 and masks_A. Compacting the sources
    means all of them are built on the compact axis; if any were not, results would
    be silently misaligned rather than raising.

    Bitwise on CPU. On a GPU the compact and full-axis reductions are lowered per
    extent and round differently: on an RTX 2080 Ti the three neuromodulation presets
    differ by up to 3.4e-6 relative (context gating) while the plain, dynamic-function
    and aggregation cases agree bitwise; a misaligned array moves a fitness by 1e-2.
    """

    FEATURES = [
        ("plain", lambda d, p: create_config_with_recurrence(
            "hidden_only", max_depth=d, population_size=p, iteration_level=2)),
        ("dynamic_cppn_output", lambda d, p: create_config_with_dynamic_functions(
            "cppn_output_4", recurrence_preset="hidden_only", max_depth=d,
            population_size=p)),
        ("dynamic_global_sin", lambda d, p: create_config_with_dynamic_functions(
            "global_sin", recurrence_preset="hidden_only", max_depth=d,
            population_size=p)),
        ("aggregation_weight_interp", lambda d, p: create_config_with_aggregation(
            "weight_interp", recurrence_preset="hidden_only", max_depth=d,
            population_size=p)),
        ("neuromod_static", lambda d, p: create_config_with_neuromodulation(
            "static_gating", recurrence_preset="hidden_only", max_depth=d,
            population_size=p)),
        ("neuromod_context", lambda d, p: create_config_with_neuromodulation(
            "context_gating", recurrence_preset="hidden_only", max_depth=d,
            population_size=p)),
        ("neuromod_4nt", lambda d, p: create_config_with_neuromodulation(
            "true_neuromodulation_4nt", recurrence_preset="hidden_only", max_depth=d,
            population_size=p)),
    ]

    @staticmethod
    def _all_positions(masks):
        """Neutralize compaction: every position stays active."""
        total = int(np.asarray(masks).shape[1])
        return compaction_module.ActiveSet(
            indices=np.arange(total), num_active=total, total_positions=total
        )

    @pytest.mark.parametrize("label,build", FEATURES, ids=[f[0] for f in FEATURES])
    def test_compaction_preserves_results(self, label, build, monkeypatch):
        # Elitism 2 (ELITISM): mutated genomes move the active set, which compaction must follow.
        config = evolving(build(DEPTH, POPULATION))
        compacted = run_generations(config, ExecutionMode.STANDARD)[1]

        monkeypatch.setattr(
            compaction_module, "find_active_positions", self._all_positions
        )
        uncompacted = run_generations(config, ExecutionMode.STANDARD)[1]

        assert_trajectories_agree(
            {"compacted": compacted, "uncompacted": uncompacted},
            f"{label}: compaction moved the result, so some per-position array was "
            f"left on the full axis",
        )

    def test_compaction_actually_reduces(self, monkeypatch):
        """Guards the tests above against passing because nothing was compacted."""
        seen = {}
        original = compaction_module.find_active_positions

        def recording(masks):
            result = original(masks)
            seen["active"] = result.num_active
            seen["total"] = result.total_positions
            return result

        monkeypatch.setattr(compaction_module, "find_active_positions", recording)
        run_generations(recurrent_config(), ExecutionMode.STANDARD, generations=1)

        assert seen, "compaction never ran on the standard recurrent path"
        assert seen["active"] < seen["total"], (
            f"compaction saved nothing: {seen['active']} of {seen['total']}"
        )

    def test_multitask_is_excluded_from_compaction(self):
        """Multi-task re-evaluates from ceiling-sized cached weights.

        run_generation_multitask calls run_generation_verbose to build the substrate
        and then evaluates each task from self._cached_W1/_cached_W2, which are
        cached before compaction so they stay ceiling-sized. Compacting would leave
        the per-position neuromodulation state narrower than those weights; doing it
        broke 165 tests, all "Incompatible shapes for broadcasting" at generation 0.
        """
        from conftest import create_config_with_multi_task  # noqa: PLC0415

        config = create_config_with_multi_task(
            task_names=["xor", "and"], neuromod_mode="true_neuromodulation_4nt",
            fitness_aggregation="mean",
        )
        emr = config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]
        emr["recurrence"] = {"preset": "hidden_only", "enabled": True,
                             "allow_hidden_to_hidden": True, "iteration_level": 2}

        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm.multitask_config is not None
        assert getattr(algorithm.multitask_config, "enabled", False), (
            "the fixture must actually enable multi-task for this guard to mean "
            "anything"
        )

    def test_modulatory_neurons_is_excluded_from_compaction(self):
        """Its modulatory count is a fraction of whatever space it is given.

        `int(total_positions * mod_ratio)` means compacting would change how many
        neurons are modulatory, and a ceiling-derived count can exceed the compact
        space outright. Excluded deliberately rather than silently changed.
        """
        config = create_config_with_neuromodulation(
            "modulatory_neurons", recurrence_preset="hidden_only",
            max_depth=DEPTH, population_size=POPULATION,
        )
        seen = []
        original = compaction_module.find_active_positions

        def recording(masks):
            seen.append(1)
            return original(masks)

        # Not monkeypatch: the guard must hold for the real call path.
        compaction_module.find_active_positions = recording
        try:
            run_generations(config, ExecutionMode.STANDARD, generations=1)
        finally:
            compaction_module.find_active_positions = original
        assert not seen, "modulatory_neurons must not be compacted"


class TestRecurrentReachesDepthFive:
    """The capability the compaction exists to deliver."""

    @pytest.mark.slow
    def test_depth_five_recurrent_runs(self):
        """Dense Whh here would be 5,460^2 x 4 x 100 = 11.9 GB and cannot allocate.

        Compaction takes it to a few MB. Reaching a fitness at all is the assertion;
        the value depends on the seed.
        """
        config = create_config_with_recurrence(
            "hidden_only", max_depth=5, population_size=100, iteration_level=2
        )
        algorithm, fitness = run_generations(
            config, ExecutionMode.STANDARD, generations=1
        )
        assert algorithm.get_forward_mode() == ForwardPassMode.HYBRID_SPARSE_HH
        assert np.isfinite(fitness[0])
