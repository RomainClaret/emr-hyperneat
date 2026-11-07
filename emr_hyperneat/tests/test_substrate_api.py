"""The public substrate API against the generation loop it is extracted from.

`emr_hyperneat.substrate` promises the substrate EMR evaluates, not a second
implementation of it. These tests hold it to that bit for bit: the substrates it
builds for a population equal the arrays the generation loop cached for the same
population, and its forward pass reproduces the loop's fitness exactly.

Also covered: the CPPN query-row layout, the configurations `substrate_config`
refuses, the aggregation channel of an aggregation-only CPPN, recurrent configurations
with per-node or global activations, which the recurrent path applies without a warning,
true neuromodulation of a feed-forward substrate and the aggregation palette (bit for bit
against the loop), and the API's capabilities and public steps.
"""
import inspect

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
from emr_hyperneat import substrate as api
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


def _emr(config):
    return config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]


def _agg_only():
    config = create_config_with_recurrence("feedforward", max_depth=DEPTH, population_size=POP)
    _emr(config)["aggregation"] = {"mode": "cppn_output", "num_aggregations": 4}
    return config


def _agg_global_max():
    config = create_config_with_recurrence("feedforward", max_depth=DEPTH, population_size=POP)
    _emr(config)["aggregation"] = {"mode": "global", "global_function": "max"}
    return config


def _combined(**agg):
    config = create_config_with_combined_recurrence(
        "feedforward", max_depth=DEPTH, population_size=POP)
    _emr(config)["aggregation"].update(agg)
    return config


def _palette_full():
    return create_config_with_dynamic_functions(
        "cppn_output_4", num_activations=18, palette="full",
        max_depth=DEPTH, population_size=POP)


def _uniform_full18():
    config = _palette_full()
    _emr(config)["dynamic_functions"]["index_decode"] = "uniform"
    return config


def _uniform_combined():
    config = _combined(index_decode="uniform")
    _emr(config)["dynamic_functions"]["index_decode"] = "uniform"
    return config


def _locality():
    config = create_config_with_dynamic_functions(
        "cppn_output_6", max_depth=DEPTH, population_size=POP)
    _emr(config)["locality_radius"] = 0.5
    return config


def _geometry_seeded():
    config = create_config_with_geometry_seeding(
        "default", max_depth=DEPTH, population_size=POP)
    _emr(config)["dynamic_functions"] = {"mode": "cppn_output", "num_activations": 6}
    return config


def _agg_palette():
    """An aggregation palette that is not a leading run of AGGREGATION_LIST: max, product, mean."""
    config = create_config_with_recurrence("feedforward", max_depth=DEPTH, population_size=POP)
    _emr(config)["aggregation"] = {"mode": "cppn_output", "palette": ["max", "product", "mean"]}
    return config


CONFIGS = {
    "aggregation_palette": _agg_palette,
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
    _emr(config)["recurrence"]["hh_discovery_mode"] = "union"
    return config


def _neuromodulated(**settings):
    return create_config_with_neuromodulation(
        "true_neuromodulation_4nt", recurrence_preset="feedforward",
        max_depth=DEPTH, population_size=POP, **settings)


def _neuromodulated_output(name):
    config = _neuromodulated()
    config["algorithm_params"]["emrhyperneat"]["substrate"]["output_activation"] = name
    return config


def _neuromodulated_with(block, value):
    config = _neuromodulated()
    _emr(config)[block] = value
    return config


#: True neuromodulation of a feed-forward substrate: the engine's defaults (receptors from the first input
#: weight, the paper's rule, mode full, inversion, four channels, strength 2), then one setting at a time.
NM_CONFIGS = {
    "default": _neuromodulated,
    "rule_steep_gate": lambda: _neuromodulated(node_rule="steep_gate"),
    "rule_steep_gate_gating": lambda: _neuromodulated(node_rule="steep_gate", modulation_mode="gating_only"),
    "rule_anti": lambda: _neuromodulated(node_rule="anti", use_output_inversion=False),
    "mode_gating_only": lambda: _neuromodulated(modulation_mode="gating_only"),
    "mode_gain_bias_only": lambda: _neuromodulated(modulation_mode="gain_bias_only"),
    "nt_2": lambda: _neuromodulated(num_nt_types=2),
    "nt_6": lambda: _neuromodulated(num_nt_types=6),
    "no_inversion": lambda: _neuromodulated(use_output_inversion=False),
    "strength_5": lambda: _neuromodulated(modulation_strength=5.0),
    "output_relu": lambda: _neuromodulated_output("relu"),
    "cppn_activations": lambda: _neuromodulated_with(
        "dynamic_functions", {"mode": "cppn_output", "num_activations": 4}),
    "aggregation_palette": lambda: _neuromodulated_with(
        "aggregation", {"mode": "cppn_output", "palette": ["max", "product", "mean"]}),
    **{f"derivation_{method}": (lambda method=method: _neuromodulated(receptor_derivation=method))
       for method in ("abs", "normalized", "fourier", "softmax", "orthogonal", "phase_shifted")},
}


@pytest.mark.parametrize("name", sorted(NM_CONFIGS))
def test_neuromodulated_substrates_match_the_generation_loop(name):
    """The receptors, base gains and fitness of a neuromodulated feed-forward substrate are the loop's."""
    algo, problem, state = _setup(NM_CONFIGS[name]())
    substrate = algo.population_substrates(state)
    algo.run_generation(state, problem)
    assert substrate.config.neuromodulation is not None
    assert _same(substrate.W1, algo._cached_W1), "W1"
    nm = api.substrate_neuromodulation(substrate)
    assert _same(nm.receptors, algo._neuromod_true["receptor_densities"]), "receptors"
    assert _same(nm.base_gains, algo._neuromod_true["base_gains"]), "base gains"
    outputs = substrate_forward(substrate, algo._cached_inputs)
    fitness = jax.vmap(
        lambda out: _eval_single_fitness(out, algo._cached_targets, algo.fitness_mode)
    )(outputs)
    np.testing.assert_array_equal(np.asarray(fitness), np.asarray(algo._cached_fitnesses))


def test_neuromodulation_settings_follow_the_configuration():
    algo, _, _ = _setup(_neuromodulated(node_rule="steep_gate", modulation_strength=5.0, num_nt_types=6))
    settings = algo.substrate_config().neuromodulation
    assert settings == api.NeuromodulationSettings(
        num_nt_types=6, modulation_strength=5.0, receptor_derivation="tanh", modulation_mode="full",
        node_rule="steep_gate", output_inversion=True)


def test_the_output_activation_under_neuromodulation_is_the_one_the_loop_applies():
    """With neither per-node mode on, the loop's plain branch (tanh and sigmoid) does not run under
    neuromodulation, so the configured output activation is the one evaluated, and reported."""
    algo, _, _ = _setup(_neuromodulated_output("relu"))
    assert algo.substrate_config().output_activation == "relu"


def test_a_given_neuromodulation_replaces_the_configured_one():
    """The forward applies the SubstrateNeuromodulation it is given: the computed one gives the same
    outputs; other receptors give others; and a substrate built without neuromodulation takes one."""
    algo, problem, state = _setup(_neuromodulated())
    substrate = algo.population_substrates(state)
    inputs = jnp.asarray(problem.inputs, dtype=jnp.float32)
    nm = api.substrate_neuromodulation(substrate)
    configured = substrate_forward(substrate, inputs)
    np.testing.assert_array_equal(np.asarray(substrate_forward(substrate, inputs, neuromodulation=nm)),
                                  np.asarray(configured))
    flipped = nm._replace(receptors=-nm.receptors)
    assert not np.array_equal(np.asarray(substrate_forward(substrate, inputs, neuromodulation=flipped)),
                              np.asarray(configured))
    plain = substrate._replace(config=substrate.config._replace(neuromodulation=None))
    np.testing.assert_array_equal(np.asarray(substrate_forward(plain, inputs, neuromodulation=nm)),
                                  np.asarray(configured))
    with pytest.raises(ValueError, match="must cover the substrate"):
        substrate_forward(plain, inputs, neuromodulation=nm._replace(base_gains=nm.base_gains[:1]))



def test_a_short_neurotransmitter_vector_is_refused():
    """JAX clamps an index past the end, so a vector shorter than the channels the rule and the
    inversion read ran silently; it is refused."""
    algo, problem, state = _setup(_neuromodulated())
    substrate = algo.population_substrates(state)
    inputs = jnp.asarray(problem.inputs, dtype=jnp.float32)
    short = api.substrate_neuromodulation(substrate, neurotransmitter=[0.5, 0.5, 0.2])
    with pytest.raises(ValueError, match="reads 4 entries"):
        substrate_forward(substrate, inputs, neuromodulation=short)


def test_the_anti_rule_reads_every_channel():
    """Under 'anti' five channels need a five-entry vector: the four-entry default is refused, and
    five entries run."""
    algo, problem, state = _setup(_neuromodulated(node_rule="anti", use_output_inversion=False))
    substrate = algo.population_substrates(state)
    inputs = jnp.asarray(problem.inputs, dtype=jnp.float32)
    nm = api.substrate_neuromodulation(substrate)
    weight = substrate.W1[:, 0, :] / substrate.config.max_weight
    five = nm._replace(receptors=api.derive_receptor_from_weight(weight, "tanh", 5) * 2.0 - 1.0,
                       settings=nm.settings._replace(num_nt_types=5))
    with pytest.raises(ValueError, match="reads 5 entries"):
        substrate_forward(substrate, inputs, neuromodulation=five)
    outputs = substrate_forward(substrate, inputs, neuromodulation=five._replace(
        neurotransmitter=jnp.asarray([0.5, 0.5, 0.5, 1.0, 0.5])))
    assert np.all(np.isfinite(np.asarray(outputs)))

def test_substrate_neuromodulation_needs_its_settings():
    algo, _, state = _setup(create_config_with_recurrence("feedforward", max_depth=DEPTH, population_size=POP))
    with pytest.raises(ValueError, match="no neuromodulation settings"):
        api.substrate_neuromodulation(algo.population_substrates(state))


def test_the_default_neurotransmitter_is_the_loops():
    np.testing.assert_array_equal(np.asarray(api.default_neurotransmitter(4)), [0.5, 0.5, 0.5, 1.0])
    np.testing.assert_array_equal(np.asarray(api.default_neurotransmitter(6)), [0.5, 0.5, 0.5, 1.0])
    np.testing.assert_array_equal(np.asarray(api.default_neurotransmitter(2)), [0.5, 0.5])


def test_capabilities_and_public_steps():
    """SUBSTRATE_CAPABILITIES names what the API covers, and the substrate's steps are the engine's own
    functions, with the keyword names a caller may rely on."""
    from emr_hyperneat import emrhyperneat as engine
    from emr_hyperneat import emrhyperneat_base as base
    assert api.SUBSTRATE_CAPABILITIES == frozenset(
        {"sparse_cap_policy", "neuromodulation", "aggregation_palette", "substrate_steps"})
    assert api.get_hierarchical_grid is base.get_hierarchical_grid
    assert api.cppn_query_rows is base.cppn_query_rows
    assert api.compute_hierarchical_variances_batch is base.compute_hierarchical_variances_batch
    assert api.cppn_output_function_indices is engine.cppn_output_function_indices
    assert api.substrate_weight_matrices is engine.substrate_weight_matrices
    assert api.self_connection_rows is engine.self_connection_rows
    assert api.derive_receptor_from_weight is engine.derive_receptor_from_weight
    assert {name: _parameters(getattr(api, name)) for name in PUBLIC_STEP_PARAMETERS} == (
        PUBLIC_STEP_PARAMETERS)
    assert _parameters(api.derive_receptor_from_weight) == (
        ("weight", REQUIRED), ("method", "tanh"), ("num_nt_types", 4))


REQUIRED = "required"

#: The parameters (name, default) of the substrate's public steps, which the API keeps: a change to
#: one of these functions keeps them, or the API wraps the new function in the old parameters.
PUBLIC_STEP_PARAMETERS = {
    "get_hierarchical_grid": (("max_depth", REQUIRED),),
    "cppn_query_rows": (("source_coord", REQUIRED), ("target_positions", REQUIRED),
                        ("outgoing", REQUIRED), ("geometry_seeding_enabled", False),
                        ("agent_coord", None)),
    "compute_hierarchical_variances_batch": (("all_weights", REQUIRED), ("grid", REQUIRED),
                                             ("legacy_level0_variance", False)),
    "cppn_output_function_indices": (("self_conn", REQUIRED), ("masks", REQUIRED), ("W1", REQUIRED),
                                     ("channel", REQUIRED), ("num_options", REQUIRED),
                                     ("num_cppn_outputs", REQUIRED), ("palette", None),
                                     ("decode", "tanh")),
    "substrate_weight_matrices": (("input_weights", REQUIRED), ("output_weights", REQUIRED),
                                  ("masks", REQUIRED), ("max_weight", REQUIRED),
                                  ("weight_threshold", 0.1), ("locality_radius", None),
                                  ("input_coords", None), ("output_coords", None),
                                  ("positions", None)),
    "self_connection_rows": (("positions", REQUIRED), ("geometry_seeding_enabled", False)),
}


def _parameters(function):
    return tuple((parameter.name,
                  REQUIRED if parameter.default is inspect.Parameter.empty else parameter.default)
                 for parameter in inspect.signature(function).parameters.values())


def _streaming():
    config = create_config_with_recurrence("feedforward", max_depth=DEPTH, population_size=POP)
    _emr(config)["execution"] = {"mode": "streaming"}
    return config


def _neuromodulated_recurrent():
    return create_config_with_neuromodulation(
        "true_neuromodulation_4nt", recurrence_preset="hidden_only",
        max_depth=DEPTH, population_size=POP)


@pytest.mark.parametrize("make, reason", [
    (_recurrent_union, "hh_discovery_mode"),
    (_neuromodulated_recurrent, "true neuromodulation with hidden-to-hidden connections"),
    (lambda: _neuromodulated(receptor_from_weight=False), "receptor_from_weight false"),
    (lambda: _neuromodulated(use_self_connection_query=True), "use_self_connection_query"),
    (lambda: create_config_with_neuromodulation(
        "static_gating", recurrence_preset="feedforward", max_depth=DEPTH, population_size=POP),
     "neuromodulation mode 'static_gating'"),
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
    _emr(config)["dynamic_functions"] = {"cppn_output_indices": {"aggregation": 0}}
    algo = EMRHyperNEAT()
    algo.create_config(config)
    assert algo.num_cppn_outputs == 2
    assert algo._cppn_output_indices["aggregation"] == 0


def test_two_output_ablation_keeps_sharing_the_weight_channel():
    """Both modes with an explicit two-output CPPN: channel 1 is the activation's."""
    config = create_config_with_combined_recurrence(
        "feedforward", max_depth=DEPTH, population_size=POP)
    _emr(config)["dynamic_functions"]["num_cppn_outputs"] = 2
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
        _emr(config)["dynamic_functions"]["index_decode"] = decode
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
