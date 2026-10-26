"""The recurrent substrate API against the generation loop it is built from.

`build_recurrent_substrate` promises the generation loop's per-genome hidden-to-hidden discovery bit for bit, with a
genome's connections independent of the other genomes in the call (an explorer can rebuild one genome as a batch of
one). `substrate_forward` settles each genome on its own active positions and matches the loop's fitness
within float32 rounding, since the loop evaluates a recurrent population on the union of its active positions. Also
covered: the query rows, the connection classes, the direct flags against the loop's own counters, the cap report,
the settling states, a genome with no active position, the recurrence fields of `substrate_config` and what it still
refuses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    GPU_TRAJECTORY_RTOL,
    XORProblem,
    create_config_with_combined_recurrence,
    create_config_with_dynamic_functions,
    create_config_with_geometry_seeding,
    create_config_with_recurrence,
    exact_backend,
)
from emr_hyperneat import (
    HH_CONNECTION_CLASSES,
    SubstrateConfig,
    EMRHyperNEAT,
    build_recurrent_substrate,
    hh_connection_classes,
    hh_pair_weights,
    hh_query_rows,
    query_recurrent_substrate,
    substrate_forward,
)
from emr_hyperneat import (
    emrhyperneat as engine,
)

DEPTH = 3
POP = 20
SEED = 42
HH_FIELDS = ("hh_from", "hh_to", "hh_weights", "hh_valid", "hh_class", "hh_direct", "hh_uncapped")

# The loop evaluates a recurrent population on the union of its active positions and the API each genome on its own,
# so on a GPU the two hand XLA the same arithmetic in different layouts and differ in float32 rounding, which a
# genome's settling can amplify. On an RTX 2080 Ti (jax 0.6.1) the two fitnesses came out at most 4.8e-7 apart in every
# genome of these configurations but one, hh_aggregation's genome 6 (3.9e-3), which settles chaotically. On the CPU,
# where the two agree within 1e-6 (up to about 1.2e-7), one ulp in the weights moves two genomes of the 140 in these
# configurations by more than 1e-4, both in hh_aggregation (5e-2 and 1.2e-4), and every other genome by at most 7e-7.
# The defects the forward test guards move more genomes, by more: dropping the h->h connections moves 5 to 19 of 20
# genomes past 1e-3 in every configuration (largest 0.2 to 0.74).
GPU_FORWARD_ATOL = 1e-3
#: A genome whose fitness moves by more than this under a one-ulp perturbation of its weights settles chaotically; on a
#: GPU its loop and API fitness may differ by anything, so the comparison leaves it out (at most a quarter of them).
ULP_SENSITIVE = 1e-4
# Two forwards of one substrate agree bit for bit on the CPU. On a GPU, two compilations of the forward (with and
# without return_states) may round differently, and with dynamic h->h aggregation one compilation differs from itself:
# the settling sums each node's h->h inputs with a scatter-add, which a GPU accumulates in no fixed order. On an RTX
# 2080 Ti (jax 0.6.1) the forwards of the six other configurations were the same bit for bit every time;
# hh_aggregation's, run five times on the same arrays, moved genome 6 by up to 0.17 and genome 19 by 4e-4, whose
# outputs one ulp in the weights moves by 0.14 and 4.9e-4, and every other genome by at most 1.1e-6.
#: On a GPU, two forwards of one substrate are compared in the genomes whose outputs one ulp in the weights moves by at
#: most this, relative to them; the others may differ by anything, so the comparison leaves them out (at most a
#: quarter of them).
ULP_OUTPUT_SENSITIVE = GPU_TRAJECTORY_RTOL


def _hmr(config):
    return config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]


def _recurrent(preset, level=1, **recurrence):
    config = create_config_with_recurrence(preset, max_depth=DEPTH, population_size=POP,
                                           iteration_level=level, hh_cache_enabled=False)
    _hmr(config)["recurrence"].update(recurrence)
    return config


def _per_node(preset="full_recurrent", level=1, **recurrence):
    config = create_config_with_dynamic_functions(
        "cppn_output_4", recurrence_preset=preset, max_depth=DEPTH, population_size=POP)
    _hmr(config)["recurrence"].update(iteration_level=level, hh_cache_enabled=False, **recurrence)
    return config


def _hh_aggregation():
    config = create_config_with_combined_recurrence("full_recurrent", max_depth=DEPTH, population_size=POP)
    _hmr(config)["recurrence"].update(iteration_level=1, hh_cache_enabled=False)
    _hmr(config)["aggregation"]["hh_aggregation_mode"] = "dynamic"
    return config


def _hh_aggregation_uniform():
    config = _hh_aggregation()
    _hmr(config)["dynamic_functions"]["index_decode"] = "uniform"
    _hmr(config)["aggregation"]["index_decode"] = "uniform"
    return config


CONFIGS = {
    "hidden_only_L1": lambda: _recurrent("hidden_only", 1),
    "with_backward_L1": lambda: _recurrent("with_backward", 1),
    "full_recurrent_L1": lambda: _recurrent("full_recurrent", 1),
    "full_recurrent_L2": lambda: _recurrent("full_recurrent", 2),
    "hidden_only_L3_fori": lambda: _recurrent("hidden_only", 3, multi_hop_algorithm="fori_loop"),
    "per_node_L2": lambda: _per_node("full_recurrent", 2),
    "hh_aggregation": _hh_aggregation,
    "hh_aggregation_uniform": _hh_aggregation_uniform,
}


def _setup(config):
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=SEED)
    return algo, problem, state


def _genome(substrate, genome):
    """One genome's substrate, as a batch of one, without its h->h connections."""
    def one(array):
        return None if array is None else array[genome:genome + 1]
    return substrate._replace(
        mask=one(substrate.mask), W1=one(substrate.W1), W2=one(substrate.W2),
        act_indices=one(substrate.act_indices), agg_indices=one(substrate.agg_indices),
        hh_agg_indices=one(substrate.hh_agg_indices),
        **{field: None for field in HH_FIELDS},
    )


def _fitness(algo, outputs):
    return np.asarray(jax.vmap(
        lambda out: engine._eval_single_fitness(out, algo._cached_targets, algo.fitness_mode))(outputs))


def _ulp_response(algo, substrate, fitness, draws=3):
    """Per genome, the largest fitness change under a one-ulp relative perturbation of W1, W2 and the h->h weights."""
    rng = np.random.default_rng(0)
    response = np.zeros_like(fitness)

    def jitter(array):
        array = np.asarray(array)
        return jnp.asarray((array * (1 + 2.0 ** -23 * rng.choice([-1.0, 1.0], size=array.shape))).astype(np.float32))

    for _ in range(draws):
        perturbed = substrate._replace(W1=jitter(substrate.W1), W2=jitter(substrate.W2),
                                       hh_weights=jitter(substrate.hh_weights))
        response = np.maximum(response, np.abs(_fitness(algo, substrate_forward(perturbed, algo._cached_inputs))
                                               - fitness))
    return response


def _assert_same_program_outputs(actual, desired, atol_cpu=0.0):
    """Two compilations of one computation: equal on the CPU (to ``atol_cpu``), to float32 rounding on a GPU."""
    if exact_backend():
        np.testing.assert_allclose(np.asarray(actual), np.asarray(desired), rtol=0, atol=atol_cpu)
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(desired), rtol=GPU_TRAJECTORY_RTOL,
                                   atol=atol_cpu)


def _steady_genomes(substrate, inputs, draws=3):
    """Per genome, whether one ulp in its weights (the draws of `_ulp_response`) moves each of its outputs by at most
    ULP_OUTPUT_SENSITIVE relative to it. Every genome on the CPU, where nothing consults it."""
    if exact_backend():
        return np.ones(np.asarray(substrate.mask).shape[0], dtype=bool)
    rng = np.random.default_rng(0)
    outputs = np.asarray(substrate_forward(substrate, inputs), dtype=np.float64)
    moved = np.zeros(outputs.shape[0])

    def jitter(array):
        array = np.asarray(array)
        return jnp.asarray((array * (1 + 2.0 ** -23 * rng.choice([-1.0, 1.0], size=array.shape))).astype(np.float32))

    for _ in range(draws):
        perturbed = substrate._replace(W1=jitter(substrate.W1), W2=jitter(substrate.W2),
                                       hh_weights=jitter(substrate.hh_weights))
        change = np.abs(np.asarray(substrate_forward(perturbed, inputs), dtype=np.float64) - outputs)
        relative = np.divide(change, np.abs(outputs), out=np.where(change > 0, np.inf, 0.0), where=outputs != 0)
        moved = np.maximum(moved, relative.reshape(len(moved), -1).max(axis=1))
    steady = moved <= ULP_OUTPUT_SENSITIVE
    assert steady.mean() >= 0.75, f"{int((~steady).sum())} of {steady.size} genomes' outputs sensitive to one ulp"
    return steady


def _assert_same_forward(actual, desired, steady):
    """Two forwards of one substrate, genome by genome: bit for bit on the CPU; on a GPU, to GPU_TRAJECTORY_RTOL in the
    ``steady`` genomes (`_steady_genomes`), the others left out."""
    if exact_backend():
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(desired))
    else:
        np.testing.assert_allclose(np.asarray(actual)[steady], np.asarray(desired)[steady],
                                   rtol=GPU_TRAJECTORY_RTOL, atol=0.0)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_connections_match_the_generation_loop(name):
    """The public two-step build gives the loop's per-genome discovery bit for bit, and its weight matrices."""
    algo, problem, state = _setup(CONFIGS[name]())
    substrate = algo.population_substrates(state)
    algo.run_generation(state, problem)
    loop = algo._cached_sparse_hh
    for mine, theirs in (("hh_from", "from_indices"), ("hh_to", "to_indices"), ("hh_weights", "weights"),
                         ("hh_valid", "valid_mask")):
        np.testing.assert_array_equal(np.asarray(getattr(substrate, mine)), np.asarray(getattr(loop, theirs)),
                                      err_msg=mine)
    assert int(np.asarray(substrate.hh_valid).sum()) > 0, "the configuration discovers connections"
    np.testing.assert_array_equal(np.asarray(substrate.W1), np.asarray(algo._cached_W1))
    np.testing.assert_array_equal(np.asarray(substrate.W2), np.asarray(algo._cached_W2))
    # The direct flags and the cap report against the loop's own counters (the module's record of the last
    # discovery, the generation's)
    stats = engine._last_discovery_stats
    kept_direct = np.asarray(stats["direct_per_genome"]) - np.asarray(stats["replaced_per_genome"])
    np.testing.assert_array_equal(np.asarray(substrate.hh_direct).sum(axis=1), kept_direct)
    np.testing.assert_array_equal(np.asarray(substrate.hh_uncapped),
                                  np.asarray(stats["uncapped_per_genome"]))


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_forward_matches_the_loop_fitness(name):
    """Each genome settled on its own positions scores the loop's fitness: within 1e-6 on the CPU, within
    GPU_FORWARD_ATOL on a GPU for every genome that does not settle chaotically."""
    algo, problem, state = _setup(CONFIGS[name]())
    substrate = algo.population_substrates(state)
    algo.run_generation(state, problem)
    fitness = _fitness(algo, substrate_forward(substrate, algo._cached_inputs))
    loop = np.asarray(algo._cached_fitnesses)
    if exact_backend():
        np.testing.assert_allclose(fitness, loop, rtol=0, atol=1e-6)
        return
    sensitive = _ulp_response(algo, substrate, fitness) > ULP_SENSITIVE
    assert sensitive.mean() <= 0.25, f"{int(sensitive.sum())} of {sensitive.size} genomes sensitive to one ulp"
    np.testing.assert_allclose(fitness[~sensitive], loop[~sensitive], rtol=0, atol=GPU_FORWARD_ATOL)


@pytest.mark.parametrize("name", ["full_recurrent_L2", "per_node_L2", "hh_aggregation"])
def test_one_genome_is_built_and_run_as_in_the_population(name):
    """Nothing depends on the other genomes: a batch of one gives the population's arrays bit for bit, and its outputs
    as `_assert_same_forward` compares two forwards (bit for bit on the CPU)."""
    algo, _, state = _setup(CONFIGS[name]())
    population = algo.population_substrates(state)
    transformed = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    outputs = query_recurrent_substrate(algo._jitted_cppn_forward, state, transformed, population)
    inputs = algo._cached_inputs
    everyone = np.asarray(substrate_forward(population, inputs))
    steady = _steady_genomes(population, inputs)
    for genome in (0, 7, POP - 1):
        alone = build_recurrent_substrate(_genome(population, genome), [outputs[genome]])
        for field in HH_FIELDS:
            np.testing.assert_array_equal(np.asarray(getattr(alone, field))[0],
                                          np.asarray(getattr(population, field))[genome], err_msg=field)
        _assert_same_forward(np.asarray(substrate_forward(alone, inputs)), everyone[genome:genome + 1],
                             steady[genome:genome + 1])


def test_query_rows_classes_and_the_band():
    """Rows are every ordered pair of a genome's active positions, source-major; classes follow y; every direct
    connection passes the band at the weight hh_pair_weights gives."""
    algo, _, state = _setup(CONFIGS["full_recurrent_L1"]())
    substrate = algo.population_substrates(state)
    positions = np.asarray(substrate.positions)
    assert len({tuple(p) for p in positions}) == len(positions)  # a self-loop is a position to itself
    rows = hh_query_rows(substrate)
    transformed = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    outputs = query_recurrent_substrate(algo._jitted_cppn_forward, state, transformed, substrate)
    config = substrate.config
    y = positions[:, 1]
    for genome, genome_rows in enumerate(rows):
        active = np.flatnonzero(np.asarray(substrate.mask[genome]))
        count = active.size
        assert genome_rows.shape == (count * count, 5)
        if not count:
            continue
        np.testing.assert_array_equal(genome_rows[:, :2], np.repeat(positions[active], count, axis=0))
        np.testing.assert_array_equal(genome_rows[:, 2:4], np.tile(positions[active], (count, 1)))
        np.testing.assert_array_equal(genome_rows[:, 4], 1.0)
        weights, passes = (np.asarray(a).reshape(count, count)
                           for a in hh_pair_weights(outputs[genome], config.max_weight, config.band_threshold))
        valid = np.asarray(substrate.hh_valid[genome])
        src, tgt = np.asarray(substrate.hh_from[genome])[valid], np.asarray(substrate.hh_to[genome])[valid]
        i, j = np.searchsorted(active, src), np.searchsorted(active, tgt)
        assert np.all(passes[i, j])
        np.testing.assert_array_equal(np.asarray(substrate.hh_weights[genome])[valid], weights[i, j])
        codes = np.asarray(substrate.hh_class[genome])
        expected = np.where(src == tgt, 3, np.where(y[src] < y[tgt], 0, np.where(y[src] == y[tgt], 1, 2)))
        np.testing.assert_array_equal(codes[valid], expected)
        assert np.all(codes[~valid] == -1)
    assert HH_CONNECTION_CLASSES == ("forward", "lateral", "backward", "self")
    present = set(np.unique(np.asarray(substrate.hh_class)[np.asarray(substrate.hh_valid)]))
    assert present == {0, 1, 2, 3}, "full recurrence discovers all four classes here"
    assert np.all(np.asarray(substrate.hh_direct)[np.asarray(substrate.hh_valid)]), "level 1 is direct only"


def test_forward_only_preset_keeps_forward_connections():
    algo, _, state = _setup(CONFIGS["hidden_only_L1"]())
    substrate = algo.population_substrates(state)
    assert set(np.unique(np.asarray(substrate.hh_class)[np.asarray(substrate.hh_valid)])) == {0}


def test_a_genome_over_the_cap_refuses_in_the_api_and_the_loop():
    """A genome over max_sparse_conns raises in build_recurrent_substrate and in the generation's
    discovery; at a cap equal to the largest set both keep every connection, the same ones."""
    algo, problem, state = _setup(_recurrent("full_recurrent", 1))
    uncapped = np.asarray(algo.population_substrates(state).hh_uncapped)
    largest = int(uncapped.max())
    assert np.any(uncapped > 40)
    refusal = rf"recurrence\.max_sparse_conns=40 cannot hold .*at least {largest}"
    capped, capped_problem, capped_state = _setup(
        _recurrent("full_recurrent", 1, max_sparse_conns=40))
    with pytest.raises(ValueError, match=refusal):
        capped.population_substrates(capped_state)
    with pytest.raises(ValueError, match=refusal):
        capped.run_generation(capped_state, capped_problem)
    exact, exact_problem, exact_state = _setup(
        _recurrent("full_recurrent", 1, max_sparse_conns=largest))
    substrate = exact.population_substrates(exact_state)
    np.testing.assert_array_equal(np.asarray(substrate.hh_valid).sum(axis=1), uncapped)
    np.testing.assert_array_equal(np.asarray(substrate.hh_uncapped), uncapped)
    exact.run_generation(exact_state, exact_problem)
    np.testing.assert_array_equal(np.asarray(substrate.hh_from),
                                  np.asarray(exact._cached_sparse_hh.from_indices))


@pytest.mark.parametrize("name", ["per_node_L2", "hh_aggregation"])
def test_states_end_on_the_settled_layer(name):
    """return_states gives every settling step, per genome over its own positions; the outputs are those without
    it, and the last state through W2 gives them."""
    from emr_hyperneat.emrhyperneat import (
        ACTIVATION_FUNCTIONS,
    )
    algo, _, state = _setup(CONFIGS[name]())
    substrate = algo.population_substrates(state)
    inputs = algo._cached_inputs
    outputs, states = substrate_forward(substrate, inputs, return_states=True)
    steps = substrate.config.activate_time
    assert steps == algo.activate_time
    _assert_same_forward(outputs, substrate_forward(substrate, inputs), _steady_genomes(substrate, inputs))
    output_fn = ACTIVATION_FUNCTIONS[substrate.config.output_activation]
    assert len(states) == POP
    for genome, settled in enumerate(states):
        active = np.flatnonzero(np.asarray(substrate.mask[genome]))
        assert settled.shape == (steps, inputs.shape[0], active.size)
        last = np.asarray(output_fn(settled[-1] @ np.asarray(substrate.W2[genome])[active]))
        _assert_same_program_outputs(last, outputs[genome], atol_cpu=1e-6)


@pytest.mark.parametrize("route", ["substrate key", "set_recurrence_config"])
def test_the_generation_settles_the_configured_steps(route):
    """A step count set through `substrate.activate_time` or `set_recurrence_config` is the count the generation
    settles, not the derived or feed-forward count: the settling states have that many rows, and the loop's
    fitness is the API's at that count, not the API's at the count derived from the depth."""
    steps, derived = 3, 2 * (2 ** DEPTH + 1)
    problem = XORProblem()
    algo = EMRHyperNEAT()
    if route == "substrate key":
        config = _recurrent("hidden_only", 1)
        config["algorithm_params"]["emrhyperneat"]["substrate"]["activate_time"] = steps
        configured = algo.create_config(config)
    else:
        configured = algo.create_config(_recurrent("feedforward", 0))
        algo.set_recurrence_config(engine.EMRConfig(allow_backward=False, allow_lateral=False,
                                                    allow_self_loops=False, iteration_level=1,
                                                    activate_time=steps))
    state = algo.initialize(configured, problem, seed=SEED)
    assert algo.activate_time == steps
    substrate = algo.population_substrates(state)
    assert substrate.config.activate_time == steps
    algo.run_generation(state, problem)
    loop = np.asarray(algo._cached_fitnesses)
    inputs = algo._cached_inputs
    outputs, states = substrate_forward(substrate, inputs, return_states=True)
    assert all(settled.shape[0] == steps for settled in states)
    fitness = _fitness(algo, outputs)
    at_derived = _fitness(algo, substrate_forward(
        substrate._replace(config=substrate.config._replace(activate_time=derived)), inputs))
    assert np.any(np.abs(at_derived - loop) > 1e-4), "the step count must move some genome's fitness"
    if exact_backend():
        np.testing.assert_allclose(fitness, loop, rtol=0, atol=1e-6)
        return
    sensitive = _ulp_response(algo, substrate, fitness) > ULP_SENSITIVE
    np.testing.assert_allclose(fitness[~sensitive], loop[~sensitive], rtol=0, atol=GPU_FORWARD_ATOL)


def test_substrate_config_carries_the_recurrence():
    algo = EMRHyperNEAT()
    algo.create_config(CONFIGS["full_recurrent_L2"]())
    config = algo.substrate_config()
    assert config.hidden_to_hidden and config.iteration_level == 2
    assert (config.allow_backward, config.allow_lateral, config.allow_self_loops) == (True, True, True)
    assert config.band_threshold == algo.band_threshold and config.activate_time == algo.activate_time
    assert config.settling_rule == "emr" and config.multi_hop_algorithm == "matrix_power"
    feedforward = EMRHyperNEAT()
    feedforward.create_config(_recurrent("feedforward", 0))
    assert not feedforward.substrate_config().hidden_to_hidden
    # A hand-built configuration keeps its feed-forward meaning (a caller's _replace calls)
    assert SubstrateConfig(input_coords=((0.0, -1.0),), output_coords=((0.0, 1.0),), max_depth=2).hidden_to_hidden \
        is False


@pytest.mark.parametrize("change, reason", [
    (dict(hh_cache_enabled=True), "h->h cache"),
    (dict(hh_discovery_mode="union"), "hh_discovery_mode"),
    (dict(multi_hop_second_factor="raw"), "hh_discovery_mode"),
    (dict(use_dense_discovery=True), "use_dense_discovery"),
])
def test_substrate_config_refuses_what_the_api_cannot_build(change, reason):
    algo = EMRHyperNEAT()
    algo.create_config(_recurrent("full_recurrent", 2, **change))
    with pytest.raises(ValueError, match=reason):
        algo.substrate_config()


@pytest.mark.parametrize("fields, reason", [
    (dict(hh_aggregation_mode="dynamic"), "aggregation channel"),
    (dict(hh_activation_mode="initial_only"), "hh_activation_mode"),
    (dict(settling_rule="other"), "settling_rule"),
    (dict(multi_hop_algorithm="other"), "multi_hop_algorithm"),
])
def test_recurrent_fields_are_validated(fields, reason):
    config = SubstrateConfig(input_coords=((0.0, -1.0),), output_coords=((0.0, 1.0),), max_depth=2,
                             hidden_to_hidden=True, **fields)
    with pytest.raises(ValueError, match=reason):
        config.validate()


def test_a_feed_forward_substrate_does_not_settle():
    algo, _, state = _setup(_recurrent("feedforward", 0))
    substrate = algo.population_substrates(state)
    with pytest.raises(ValueError, match="return_states needs a recurrent substrate"):
        substrate_forward(substrate, algo._cached_inputs, return_states=True)
    with pytest.raises(ValueError, match="no hidden-to-hidden connections"):
        hh_query_rows(substrate)


def test_a_cleared_mask_entry_drops_the_node_and_its_connections():
    """A knocked-out node's connections are dropped, not moved onto another node."""
    algo, _, state = _setup(CONFIGS["full_recurrent_L1"]())
    substrate = algo.population_substrates(state)
    inputs = algo._cached_inputs
    valid = np.asarray(substrate.hh_valid)
    genome = int(np.argmax(valid.sum(axis=1)))
    sources = np.asarray(substrate.hh_from)[genome][valid[genome]]
    node = int(np.bincount(sources).argmax())  # the position with the most outgoing connections
    mask = np.asarray(substrate.mask).copy()
    mask[genome, node] = False
    knocked = substrate._replace(mask=jnp.asarray(mask))
    touching = (np.asarray(substrate.hh_from)[genome] == node) | (np.asarray(substrate.hh_to)[genome] == node)
    assert (touching & valid[genome]).any()
    dropped_valid = valid.copy()
    dropped_valid[genome] &= ~touching
    dropped = knocked._replace(hh_valid=jnp.asarray(dropped_valid))
    np.testing.assert_array_equal(np.asarray(substrate_forward(knocked, inputs))[genome],
                                  np.asarray(substrate_forward(dropped, inputs))[genome])


EMPTY = (0, 7, POP - 1)


def _geometry_7d():
    config = create_config_with_geometry_seeding("default", recurrence_preset="full_recurrent", max_depth=DEPTH,
                                                 population_size=POP)
    _hmr(config)["recurrence"].update(iteration_level=1, hh_cache_enabled=False)
    return config


EMPTY_MASK_CONFIGS = {name: CONFIGS[name] for name in ("full_recurrent_L1", "full_recurrent_L2", "per_node_L2",
                                                       "hh_aggregation")}
EMPTY_MASK_CONFIGS["geometry_7d"] = _geometry_7d


def _cleared(substrate, genomes):
    """The substrate with these genomes' masks cleared: no active position."""
    mask = np.asarray(substrate.mask).copy()
    mask[list(genomes)] = False
    return substrate._replace(mask=jnp.asarray(mask))


def _feed_forward(substrate):
    """The same genomes as a feed-forward substrate: no h->h connections, the columns outside the mask zeroed."""
    mask = np.asarray(substrate.mask)
    return substrate._replace(
        config=substrate.config._replace(hidden_to_hidden=False),
        W1=jnp.where(mask[:, None, :], substrate.W1, 0.0), W2=jnp.where(mask[:, :, None], substrate.W2, 0.0),
        hh_agg_indices=None, **{field: None for field in HH_FIELDS})


@pytest.mark.parametrize("name", sorted(EMPTY_MASK_CONFIGS))
def test_a_genome_with_no_active_position(name):
    """A genome whose mask is cleared has no rows, no CPPN outputs and no connections; it settles to states of width 0
    and gives the feed-forward forward's outputs (the output activation of zero), in a population and alone. The
    other genomes' arrays are the unedited population's, bit for bit, and so are their forwards and settling states on
    the CPU (on a GPU, their forwards as `_assert_same_forward` compares two)."""
    from emr_hyperneat.substrate import _bucket_slices
    algo, _, state = _setup(EMPTY_MASK_CONFIGS[name]())
    population = algo.population_substrates(state)
    config = population.config
    transformed = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    inputs = algo._cached_inputs
    rows_before = hh_query_rows(population)
    outputs_before = query_recurrent_substrate(algo._jitted_cppn_forward, state, transformed, population)
    forward_before, states_before = substrate_forward(population, inputs, return_states=True)

    edited = _cleared(population, EMPTY)
    counts = np.asarray(edited.mask).sum(axis=1)
    buckets = {genome: size for size, slices in _bucket_slices(counts) for real, _ in slices for genome in real}
    assert all(buckets[genome] == 8 for genome in EMPTY), "an empty genome takes the smallest bucket"
    rows = hh_query_rows(edited)
    outputs = query_recurrent_substrate(algo._jitted_cppn_forward, state, transformed, edited)
    built = build_recurrent_substrate(edited, outputs)
    forward, states = substrate_forward(built, inputs, return_states=True)
    steady = _steady_genomes(built, inputs)
    _assert_same_forward(substrate_forward(built, inputs), forward, steady)
    feed_forward = np.asarray(substrate_forward(_feed_forward(built), inputs))
    # The edited mask with the population's connections: the forward drops every connection of a cleared node
    stale = np.asarray(substrate_forward(edited, inputs, return_states=True)[0])
    output_fn = engine.ACTIVATION_FUNCTIONS[config.output_activation]
    width = 7 if config.geometry_seeding else 5
    for genome in range(POP):
        if genome in EMPTY:
            assert rows[genome].shape == (0, width) and rows[genome].dtype == np.float32
            assert np.asarray(outputs[genome]).shape == (0, config.num_cppn_outputs)
            weights, passes = hh_pair_weights(outputs[genome], config.max_weight, config.band_threshold)
            assert weights.shape == passes.shape == (0,)
            assert not np.asarray(built.hh_valid)[genome].any() and int(np.asarray(built.hh_uncapped)[genome]) == 0
            assert np.all(np.asarray(built.hh_from)[genome] == -1) and np.all(np.asarray(built.hh_to)[genome] == -1)
            assert np.all(np.asarray(built.hh_class)[genome] == -1) and not np.asarray(built.hh_direct)[genome].any()
            np.testing.assert_array_equal(np.asarray(forward)[genome], feed_forward[genome])
            np.testing.assert_array_equal(np.asarray(forward)[genome],
                                          np.asarray(output_fn(jnp.zeros_like(forward[genome]))))
            np.testing.assert_array_equal(stale[genome], np.asarray(forward)[genome])
            assert states[genome].shape == (config.activate_time, inputs.shape[0], 0)
            continue
        np.testing.assert_array_equal(rows[genome], rows_before[genome])
        np.testing.assert_array_equal(np.asarray(outputs[genome]), np.asarray(outputs_before[genome]))
        for field in HH_FIELDS:
            np.testing.assert_array_equal(np.asarray(getattr(built, field))[genome],
                                          np.asarray(getattr(population, field))[genome], err_msg=field)
        pick = slice(genome, genome + 1)
        _assert_same_forward(np.asarray(forward)[pick], np.asarray(forward_before)[pick], steady[pick])
        _assert_same_forward(stale[pick], np.asarray(forward_before)[pick], steady[pick])
        if exact_backend():
            # On a GPU a settled state near zero has no useful relative tolerance; the outputs above stand for it
            np.testing.assert_array_equal(np.asarray(states[genome]), np.asarray(states_before[genome]))

    # An empty genome alone, as a batch of one
    genome = EMPTY[1]
    one = _genome(edited, genome)
    assert hh_query_rows(one)[0].shape == (0, width)
    alone = build_recurrent_substrate(one, [outputs[genome]])
    assert not np.asarray(alone.hh_valid).any()
    alone_forward, alone_states = substrate_forward(alone, inputs, return_states=True)
    np.testing.assert_array_equal(np.asarray(alone_forward)[0], np.asarray(forward)[genome])
    assert alone_states[0].shape == (config.activate_time, inputs.shape[0], 0)


def test_a_population_with_no_active_position():
    """Every genome cleared: the whole path runs, every genome gives the feed-forward forward's outputs, and the
    connection classes of the padding are -1."""
    algo, _, state = _setup(CONFIGS["full_recurrent_L1"]())
    population = _cleared(algo.population_substrates(state), range(POP))
    transformed = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    inputs = algo._cached_inputs
    assert all(rows.shape == (0, 5) for rows in hh_query_rows(population))
    outputs = query_recurrent_substrate(algo._jitted_cppn_forward, state, transformed, population)
    assert all(np.asarray(out).shape == (0, 1) for out in outputs)
    built = build_recurrent_substrate(population, outputs)
    assert not np.asarray(built.hh_valid).any()
    np.testing.assert_array_equal(np.asarray(hh_connection_classes(built.positions, built.hh_from, built.hh_to)), -1)
    forward, states = substrate_forward(built, inputs, return_states=True)
    np.testing.assert_array_equal(np.asarray(forward), np.asarray(substrate_forward(_feed_forward(built), inputs)))
    assert all(settled.shape == (built.config.activate_time, inputs.shape[0], 0) for settled in states)
    # Any empty array stands for an empty genome's outputs
    one = build_recurrent_substrate(_genome(population, 0), [np.zeros(0, dtype=np.float32)])
    np.testing.assert_array_equal(np.asarray(substrate_forward(one, inputs))[0], np.asarray(forward)[0])


def test_outputs_of_the_wrong_length_are_refused():
    """A genome's outputs must have a row per pair of its active positions; an empty array for a genome that has
    positions is refused with the count it needs."""
    algo, _, state = _setup(CONFIGS["full_recurrent_L1"]())
    population = algo.population_substrates(state)
    transformed = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    outputs = list(query_recurrent_substrate(algo._jitted_cppn_forward, state, transformed, population))
    count = int(np.asarray(population.mask)[2].sum())
    for wrong in (np.zeros((0, 1), dtype=np.float32), np.asarray(outputs[2])[1:], np.float32(0.5)):
        bad = outputs[:2] + [wrong] + outputs[3:]
        with pytest.raises(ValueError, match=rf"genome 2: expected {count * count} rows"):
            build_recurrent_substrate(population, bad)
    # An empty genome's outputs must be empty
    bad = list(outputs)
    bad[2] = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="genome 2: expected 0 rows"):
        build_recurrent_substrate(_cleared(population, [2]), bad)


def test_the_compiled_forward_depends_only_on_what_it_reads():
    """Coordinates, thresholds and the cap do not recompile the forward, and list coordinates are accepted."""
    from emr_hyperneat.substrate import _recurrent_forward_fn
    algo, _, state = _setup(CONFIGS["hidden_only_L1"]())
    config = algo.population_substrates(state).config
    first = _recurrent_forward_fn(config, 16, False, False, False)
    other = config._replace(output_coords=((0.5, 1.0),), variance_threshold=0.05, max_sparse_conns=123,
                            input_coords=[list(c) for c in config.input_coords])
    assert _recurrent_forward_fn(other, 16, False, False, False) is first
    assert _recurrent_forward_fn(config._replace(activate_time=3), 16, False, False, False) is not first



def test_the_aggregation_decode_reaches_the_recurrent_input():
    """aggregation.index_decode reaches the recurrent input's dynamic aggregation: uniform and tanh give some nodes
    different functions (the loop and the substrate API agree under each, test_connections_match_the_generation_loop
    and test_forward_matches_the_loop_fitness)."""
    indices = {}
    for name in ("hh_aggregation", "hh_aggregation_uniform"):
        algo, _, state = _setup(CONFIGS[name]())
        indices[name] = np.asarray(algo.population_substrates(state).hh_agg_indices)
    assert indices["hh_aggregation"].shape == indices["hh_aggregation_uniform"].shape
    assert (indices["hh_aggregation"] != indices["hh_aggregation_uniform"]).any()
