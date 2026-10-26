"""Built-in palette evolution, from HMR's dynamic-functions variant.

HMR's random walk on a palette mask: after a number of generations without a new best fitness, every
function's bit flips with a probability. Fixed rather than reproduced: the decode (HMR rebinned the
channel over the enabled functions, so every toggle remapped a quarter to a half of every genome's
nodes; the stable decode bins over the whole list and snaps a disabled function to its nearest
enabled one, and HMR's is the 'rebin' opt-in), the best fitness under skip_metrics, the state an
object kept across initialize, the palette metric, and the silent configuration.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    DEFAULT_SEED,
    XORProblem,
    create_config_with_dynamic_functions,
    create_config_with_multi_task,
)
from emr_hyperneat import EMRHyperNEAT
from emr_hyperneat.emrhyperneat import (
    ACTIVATION_LIST,
    continuous_to_index,
    palette_snap_table,
)

POP = 20
DEPTH = 2


def _config(**evolution):
    config = create_config_with_dynamic_functions(
        "cppn_output_4", max_depth=DEPTH, population_size=POP
    )
    block = dict(enabled=True, **evolution)
    config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["dynamic_functions"][
        "palette_evolution"
    ] = block
    return config


def _run(config, generations, seed=DEFAULT_SEED):
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=seed)
    records = []
    for _ in range(generations):
        state, metrics = algo.run_generation(state, problem)
        records.append(dict(metrics.custom_metrics))
    return algo, state, records


class TestConfiguration:
    @pytest.mark.parametrize("change, reason", [
        (dict(mutation_rate=0.0), "mutation_rate"),
        (dict(initial_mask="all"), "initial_mask"),
        (dict(initial_mask=[0, 99]), "initial_mask entry"),
        (dict(min_active=5, initial_mask=[0, 1, 2]), "min_active"),
        (dict(stagnation_generations=0), "stagnation_generations"),
        (dict(decode="nearest"), "decode"),
        (dict(rate=0.2), "palette_evolution"),
    ])
    def test_wrong_settings_raise(self, change, reason):
        with pytest.raises(ValueError, match=reason):
            EMRHyperNEAT().create_config(_config(**change))

    def test_needs_the_cppn_decode_and_owns_the_palette(self):
        config = create_config_with_dynamic_functions(
            "global_sin", max_depth=DEPTH, population_size=POP
        )
        config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["dynamic_functions"][
            "palette_evolution"
        ] = {"enabled": True}
        with pytest.raises(ValueError, match="needs mode 'cppn_output'"):
            EMRHyperNEAT().create_config(config)
        config = _config()
        config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["dynamic_functions"][
            "palette"
        ] = "oscillatory"
        with pytest.raises(ValueError, match="initial_mask"):
            EMRHyperNEAT().create_config(config)

    def test_initial_masks(self):
        for initial, expected in (
            ("full", list(range(len(ACTIVATION_LIST)))),
            ("default", [0, 1, 2, 3]),
            (["sin", 0], [0, 4]),
        ):
            algo = EMRHyperNEAT()
            algo.create_config(_config(initial_mask=initial))
            assert [int(i) for i in np.asarray(algo.df_palette)] == expected


class TestTheWalk:
    def test_metrics_and_the_mask(self):
        algo, _, records = _run(
            _config(stagnation_generations=1, mutation_rate=0.5, initial_mask="default"), 6
        )
        palette = [0, 1, 2, 3]
        changes = 0
        for record in records:
            assert record["palette"] == palette, "the palette the generation used"
            assert record["palette_names"] == [ACTIVATION_LIST[i] for i in palette]
            if record["palette_changed"]:
                changes += 1
                palette = record["next_palette"]
                assert len(palette) >= 2
            else:
                assert "next_palette" not in record
        assert changes > 0
        assert [int(i) for i in np.asarray(algo.df_palette)] == palette

    def test_a_seed_gives_one_walk_and_initialize_starts_it_again(self):
        config = _config(stagnation_generations=1, mutation_rate=0.5)
        _, _, first = _run(config, 4)
        _, _, second = _run(config, 4)
        assert [r["palette"] for r in first] == [r["palette"] for r in second]
        algo = EMRHyperNEAT()
        problem = XORProblem()
        state = algo.initialize(algo.create_config(config), problem, seed=DEFAULT_SEED)
        for _ in range(4):
            state, _ = algo.run_generation(state, problem)
        algo.initialize(algo.neat_algo, problem, seed=DEFAULT_SEED)
        assert [int(i) for i in np.asarray(algo.df_palette)] == list(range(len(ACTIVATION_LIST)))
        assert algo._palette_best is None and algo._palette_stagnation == 0

    def test_hmr_counts_the_first_generation_as_stagnant(self):
        """With stagnation_generations 1 the mask is mutated after the first generation already, as
        in HMR."""
        algo, _, records = _run(
            _config(stagnation_generations=1, mutation_rate=1.0, initial_mask="default"), 1
        )
        assert records[0]["palette_changed"]

    def test_a_runner_cannot_change_an_owned_palette(self):
        algo = EMRHyperNEAT()
        problem = XORProblem()
        state = algo.initialize(algo.create_config(_config()), problem, seed=DEFAULT_SEED)
        algo.df_palette = jnp.array([0, 4], dtype=jnp.int32)
        with pytest.raises(ValueError, match="owns df_palette"):
            algo.run_generation(state, problem)


class TestStableDecode:
    def test_removing_an_unused_function_changes_no_node(self):
        """Stable: toggling a function no node decodes to leaves every node's function; rebin moves
        the bins."""
        raw = jnp.asarray(np.random.default_rng(0).uniform(-0.999, 0.999, 500).astype(np.float32))
        full = np.ones(len(ACTIVATION_LIST), dtype=bool)
        stable_full = palette_snap_table(full)[
            np.asarray(continuous_to_index(raw, len(ACTIVATION_LIST)))
        ]
        unused = [f for f in range(len(ACTIVATION_LIST)) if f not in set(stable_full.tolist())]
        assert unused, "the double tanh leaves the outer functions unreachable"
        reduced = full.copy()
        reduced[unused[0]] = False
        stable_reduced = palette_snap_table(reduced)[
            np.asarray(continuous_to_index(raw, len(ACTIVATION_LIST)))
        ]
        np.testing.assert_array_equal(stable_reduced, stable_full)
        enabled = np.flatnonzero(reduced)
        rebinned = enabled[np.asarray(continuous_to_index(raw, enabled.size))]
        assert np.mean(rebinned != stable_full) > 0.2

    def test_snap_table(self):
        mask = np.zeros(len(ACTIVATION_LIST), dtype=bool)
        mask[[2, 6]] = True
        table = palette_snap_table(mask)
        assert table.tolist()[:9] == [2, 2, 2, 2, 2, 6, 6, 6, 6]
        assert set(table.tolist()) == {2, 6}


class TestPathsWithoutTheStep:
    def test_depth_offload_is_refused(self):
        """Offload evaluates tanh nodes, so it refuses the per-node decode, and palette evolution
        with it, at configuration (the run-time guard covers AUTO escalating to offload)."""
        config = _config()
        config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["execution"] = {
            "mode": "depth_offload"
        }
        with pytest.raises(ValueError, match="depth_offload"):
            EMRHyperNEAT().create_config(config)

    def test_multi_task_is_refused(self, two_task_problems):
        """The multi-task evaluators apply one activation per task, so they refuse the per-node
        decode, and palette evolution with it, at configuration (run_generation_multitask also
        refuses it at run time).
        """
        config = create_config_with_multi_task(
            task_names=two_task_problems["task_names"],
            fitness_aggregation="mean",
            recurrence_preset="feedforward",
        )
        config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["dynamic_functions"] = {
            "mode": "cppn_output", "num_activations": 4, "palette_evolution": {"enabled": True}}
        with pytest.raises(ValueError, match="would ignore dynamic_functions.mode='cppn_output'"):
            EMRHyperNEAT().create_config(config)

    def test_the_substrate_api_is_refused(self):
        algo = EMRHyperNEAT()
        algo.create_config(_config())
        with pytest.raises(ValueError, match="palette evolution"):
            algo.substrate_config()
