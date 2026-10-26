"""The public substrate API against the generation loop it is extracted from.

`emr_hyperneat.substrate` promises the substrate EMR evaluates, not a second
implementation of it. These tests hold it to that bit for bit: the substrates it
builds for a population equal the arrays the generation loop cached for the same
population, and its forward pass reproduces the loop's fitness exactly.

Also covered: the CPPN query-row layout, the configurations `substrate_config`
refuses, the aggregation channel of an aggregation-only CPPN, and recurrent configurations
with per-node or global activations, which the recurrent path applies without a warning.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    XORProblem,
    create_config_with_combined_recurrence,
    create_config_with_dynamic_functions,
    create_config_with_geometry_seeding,
    create_config_with_neuromodulation,
    create_config_with_recurrence,
)
from emr_hyperneat import (
    EMRHyperNEAT,
    SubstrateConfig,
    build_substrate,
    substrate_forward,
    substrate_queries,
)
from emr_hyperneat.emrhyperneat import (
    _eval_single_fitness,
    continuous_to_index,
    query_cppn_multi_output,
)
from emr_hyperneat.emrhyperneat_base import (
    _query_population_vmap_device_aware,
    get_hierarchical_grid,
)

DEPTH = 3
POP = 20
SEED = 42


def _hmr(config):
    return config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]


def _agg_only():
    config = create_config_with_recurrence("feedforward", max_depth=DEPTH, population_size=POP)
    _hmr(config)["aggregation"] = {"mode": "cppn_output", "num_aggregations": 4}
    return config


def _agg_global_max():
    config = create_config_with_recurrence("feedforward", max_depth=DEPTH, population_size=POP)
    _hmr(config)["aggregation"] = {"mode": "global", "global_function": "max"}
    return config


def _combined(**agg):
    config = create_config_with_combined_recurrence(
        "feedforward", max_depth=DEPTH, population_size=POP)
    _hmr(config)["aggregation"].update(agg)
    return config


def _palette_full():
    return create_config_with_dynamic_functions(
        "cppn_output_4", num_activations=18, palette="full",
        max_depth=DEPTH, population_size=POP)


def _uniform_full18():
    config = _palette_full()
    _hmr(config)["dynamic_functions"]["index_decode"] = "uniform"
    return config


def _uniform_combined():
    config = _combined(index_decode="uniform")
    _hmr(config)["dynamic_functions"]["index_decode"] = "uniform"
    return config


def _locality():
    config = create_config_with_dynamic_functions(
        "cppn_output_6", max_depth=DEPTH, population_size=POP)
    _hmr(config)["locality_radius"] = 0.5
    return config


def _geometry_seeded():
    config = create_config_with_geometry_seeding(
        "default", max_depth=DEPTH, population_size=POP)
    _hmr(config)["dynamic_functions"] = {"mode": "cppn_output", "num_activations": 6}
    return config


CONFIGS = {
    "plain": lambda: create_config_with_recurrence(
        "feedforward", max_depth=DEPTH, population_size=POP),
    "global_sin": lambda: create_config_with_dynamic_functions(
        "global_sin", max_depth=DEPTH, population_size=POP),
    "cppn_output": lambda: create_config_with_dynamic_functions(
        "cppn_output_4", max_depth=DEPTH, population_size=POP),
    "cppn_output_palette": lambda: create_config_with_dynamic_functions(
        "cppn_output_4", palette="oscillatory", max_depth=DEPTH, population_size=POP),
    "cppn_output_full18": _palette_full,
    "cppn_output_full18_uniform": _uniform_full18,
    "aggregation_only": _agg_only,
    "aggregation_global_max": _agg_global_max,
    "combined_true": _combined,
    "combined_approx": lambda: _combined(use_true_aggregation=False),
    "combined_uniform": _uniform_combined,
    "locality": _locality,
    "geometry_seeding": _geometry_seeded,
}


def _setup(config):
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=SEED)
    return algo, problem, state


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_substrates_and_fitness_match_the_generation_loop(name):
    algo, problem, state = _setup(CONFIGS[name]())
    substrate = algo.population_substrates(state)
    algo.run_generation(state, problem)

    assert _same(substrate.W1, algo._cached_W1), "W1"
    assert _same(substrate.W2, algo._cached_W2), "W2"
    assert _same(substrate.act_indices, algo._cached_act_indices), "activation indices"
    assert _same(substrate.agg_indices, algo._cached_ih_agg_indices), "aggregation indices"

    grid = get_hierarchical_grid(DEPTH)
    mask = np.asarray(substrate.mask)
    offsets = grid.level_offsets_static
    reached = [int(np.count_nonzero(mask[:, offsets[lv]:offsets[lv + 1]]))
               for lv in range(grid.num_levels)]
    assert reached == algo._last_reached_per_level

    outputs = substrate_forward(substrate, algo._cached_inputs)
    fitness = jax.vmap(
        lambda out: _eval_single_fitness(out, algo._cached_targets, algo.fitness_mode)
    )(outputs)
    np.testing.assert_array_equal(np.asarray(fitness), np.asarray(algo._cached_fitnesses))


@pytest.mark.parametrize("name", ["plain", "combined_true", "geometry_seeding"])
def test_query_rows_are_the_rows_the_generation_loop_queries(name):
    """Evaluating the CPPNs at substrate_queries rows reproduces the loop's own queries."""
    algo, _, state = _setup(CONFIGS[name]())
    config = algo.substrate_config()
    queries = substrate_queries(config)
    transformed = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    forward = algo._jitted_cppn_forward
    reference = algo.population_substrates(state)

    if config.num_cppn_outputs > 1:
        def evaluate(rows):
            return np.asarray(query_cppn_multi_output(forward, state, rows, transformed))
    else:
        def evaluate(rows):
            return np.asarray(_query_population_vmap_device_aware(
                forward, state, rows, transformed, 0))[..., None]

    input_out = np.stack([evaluate(rows) for rows in queries.input_rows], axis=1)
    output_out = np.stack([evaluate(rows) for rows in queries.output_rows], axis=1)
    self_out = evaluate(queries.self_rows)
    rebuilt = build_substrate(input_out, output_out, self_out, config)

    assert _same(rebuilt.W1, reference.W1)
    assert _same(rebuilt.W2, reference.W2)
    assert _same(rebuilt.act_indices, reference.act_indices)
    assert _same(rebuilt.agg_indices, reference.agg_indices)


def test_query_row_layout():
    config = SubstrateConfig(input_coords=((-1.0, -1.0), (1.0, -1.0)),
                             output_coords=((0.0, 1.0),), max_depth=1)
    queries = substrate_queries(config)
    positions = np.asarray(get_hierarchical_grid(1).all_positions)
    assert queries.input_rows.shape == (2, 20, 5)
    assert queries.output_rows.shape == (1, 20, 5)
    assert queries.self_rows.shape == (20, 5)
    # Rows are [from, to, 1]: input->grid rows start with the input node, grid->output
    # rows with the grid position.
    np.testing.assert_array_equal(queries.input_rows[1, :, :2], np.tile([1.0, -1.0], (20, 1)))
    np.testing.assert_array_equal(queries.input_rows[1, :, 2:4], positions)
    np.testing.assert_array_equal(queries.output_rows[0, :, :2], positions)
    np.testing.assert_array_equal(queries.output_rows[0, :, 2:4], np.tile([0.0, 1.0], (20, 1)))
    np.testing.assert_array_equal(queries.self_rows[:, :2], queries.self_rows[:, 2:4])
    for rows in (queries.input_rows, queries.output_rows, queries.self_rows):
        assert np.all(rows[..., -1] == 1.0)

    seeded = substrate_queries(config._replace(geometry_seeding=True))
    rows = seeded.input_rows[0]
    assert rows.shape == (20, 7)
    np.testing.assert_array_equal(rows[:, 4:6], rows[:, 2:4] - rows[:, 0:2])
    assert np.all(rows[:, -1] == 1.0)
    assert np.all(seeded.self_rows[:, 4:6] == 0.0)


def test_stacked_split_round_trip():
    config = SubstrateConfig(input_coords=((-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)),
                             output_coords=((0.0, 1.0),), max_depth=2)
    queries = substrate_queries(config)
    stacked = queries.stacked()
    fake = np.stack([stacked[:, :3], stacked[:, 2:5]])  # (2 genomes, rows, 3 channels)
    input_out, output_out, self_out = queries.split(fake)
    np.testing.assert_array_equal(input_out[1, 2, :, :], queries.input_rows[2][:, 2:5])
    np.testing.assert_array_equal(output_out[0, 0, :, :], queries.output_rows[0][:, :3])
    np.testing.assert_array_equal(self_out[1], queries.self_rows[:, 2:5])


def _recurrent_cppn_output():
    return create_config_with_dynamic_functions(
        "cppn_output_4", recurrence_preset="hidden_only", max_depth=DEPTH, population_size=POP)


def _recurrent_union():
    # Recurrence itself is described (test_substrate_recurrent.py); the union discovery is not
    config = _recurrent_cppn_output()
    _hmr(config)["recurrence"]["hh_discovery_mode"] = "union"
    return config


def _neuromodulated():
    return create_config_with_neuromodulation(
        "true_neuromodulation_4nt", recurrence_preset="feedforward",
        max_depth=DEPTH, population_size=POP)


def _streaming():
    config = create_config_with_recurrence("feedforward", max_depth=DEPTH, population_size=POP)
    _hmr(config)["execution"] = {"mode": "streaming"}
    return config


@pytest.mark.parametrize("make, reason", [
    (_recurrent_union, "hh_discovery_mode"),
    (_neuromodulated, "neuromodulation"),
    (lambda: create_config_with_dynamic_functions(
        "weight_interp_variance", max_depth=DEPTH, population_size=POP), "weight_interpretation"),
    (lambda: create_config_with_dynamic_functions(
        "random_fixed", max_depth=DEPTH, population_size=POP), "random_fixed"),
    (_streaming, "streaming"),
])
def test_substrate_config_refuses_what_it_cannot_describe(make, reason):
    algo = EMRHyperNEAT()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        algo.create_config(make())
    with pytest.raises(ValueError, match=reason):
        algo.substrate_config()


def test_substrate_config_before_create_config():
    with pytest.raises(RuntimeError):
        EMRHyperNEAT().substrate_config()


def test_aggregation_only_reads_its_own_channel():
    """The second output of an aggregation-only CPPN is read, not the weight."""
    algo, problem, state = _setup(_agg_only())
    assert algo.num_cppn_outputs == 2
    assert algo._cppn_output_indices["aggregation"] == 1
    assert algo.substrate_config().aggregation_channel == 1
    mask = algo.population_substrates(state).mask
    algo.run_generation(state, problem)

    self_conn = jnp.asarray(algo._cached_self_conn)
    from_own_channel = continuous_to_index(jnp.where(mask, self_conn[:, :, 1], 0.0), 4)
    from_weight = continuous_to_index(jnp.where(mask, self_conn[:, :, 0], 0.0), 4)
    np.testing.assert_array_equal(np.asarray(algo._cached_ih_agg_indices),
                                  np.asarray(from_own_channel))
    assert not np.array_equal(np.asarray(from_own_channel), np.asarray(from_weight))


def test_aggregation_channel_zero_reproduces_pre_0_2_runs():
    config = _agg_only()
    _hmr(config)["dynamic_functions"] = {"cppn_output_indices": {"aggregation": 0}}
    algo = EMRHyperNEAT()
    algo.create_config(config)
    assert algo.num_cppn_outputs == 2
    assert algo._cppn_output_indices["aggregation"] == 0


def test_two_output_ablation_keeps_sharing_the_weight_channel():
    """Both modes with an explicit two-output CPPN: channel 1 is the activation's."""
    config = create_config_with_combined_recurrence(
        "feedforward", max_depth=DEPTH, population_size=POP)
    _hmr(config)["dynamic_functions"]["num_cppn_outputs"] = 2
    algo = EMRHyperNEAT()
    algo.create_config(config)
    assert algo._cppn_output_indices == {"weight": 0, "activation": 1, "aggregation": 0}


@pytest.mark.parametrize("make", [
    _recurrent_cppn_output,
    lambda: create_config_with_dynamic_functions(
        "global_sin", recurrence_preset="hidden_only", max_depth=DEPTH, population_size=POP),
])
def test_the_recurrent_path_applies_activations_without_a_warning(make):
    """The recurrent path applies these activations (`every_iteration`) and create_config warns about none."""
    algo = EMRHyperNEAT()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        algo.create_config(make())
    assert algo.hh_activation_mode == "every_iteration"
    assert not [w for w in record if "hidden-to-hidden" in str(w.message)]


def test_uniform_decode_reaches_every_palette_entry():
    """With an 18-function palette, the tanh decode assigns positions 2 to 15 only; uniform assigns all 18 in the
    substrate API, as in the generation loop."""
    reached = {}
    for decode in ("tanh", "uniform"):
        config = create_config_with_dynamic_functions(
            "cppn_output_4", num_activations=18, palette="full", max_depth=4, population_size=100)
        _hmr(config)["dynamic_functions"]["index_decode"] = decode
        algo, _, state = _setup(config)
        reached[decode] = set(np.unique(np.asarray(algo.population_substrates(state).act_indices)).tolist())
    assert reached["tanh"] <= set(range(2, 16))
    assert reached["uniform"] == set(range(18))


def test_an_unknown_index_decode_is_refused():
    algo, _, _ = _setup(_palette_full())
    config = algo.substrate_config()
    with pytest.raises(ValueError, match="activation_index_decode must be one of"):
        config._replace(activation_index_decode="linear").validate()
    with pytest.raises(ValueError, match="aggregation_index_decode must be one of"):
        config._replace(aggregation_index_decode="linear").validate()
