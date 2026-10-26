"""The recurrent path's one settling rule, the champion record, and the refusals.

Every recurrent evaluation settles by one rule that reduces to the feed-forward evaluation
without hidden-to-hidden connections, whatever the evaluator (neuromodulated, multi-task or
data-parallel), and applies the input-to-hidden aggregation and, under the default
`hh_activation_mode: every_iteration`, the configured hidden activation at every step;
`initial_only`, which applied none, is refused.

`evaluate_on_data` reproduces the fitness the champion was selected on rather than re-deriving
the champion as a dense tanh network, NaN fitness does not reach selection, and what a path
cannot apply is refused rather than accepted and partly ignored.
"""

import copy
import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import emr_hyperneat.emrhyperneat as E
from conftest import (  # type: ignore[import-not-found]
    GPU_TRAJECTORY_RTOL,
    MultiTaskProblem,
    XORProblem,
    create_config_with_aggregation,
    create_config_with_dynamic_functions,
    create_config_with_multi_task,
    create_config_with_neuromodulation,
    create_config_with_recurrence,
    exact_backend,
)
from emr_hyperneat import EMRHyperNEAT
from hmr_reference import hmr_value, live  # type: ignore[import-not-found]

DEPTH, POP, SEED = 3, 20, 42
KW = dict(max_depth=DEPTH, population_size=POP)
EPS = float(np.finfo(np.float32).eps)


def _agree(actual, expected, exact=True):
    """Bit for bit on CPU where ``exact``; otherwise to float32 rounding.

    GPU kernels are chosen by shape, so on a GPU equal arithmetic agrees to rounding only
    (see conftest.exact_backend). The defects these tests guard moved fitness by 1e-2 or more.
    """
    if exact and exact_backend():
        np.testing.assert_array_equal(actual, expected)
    else:
        rtol = 4 * EPS if exact_backend() else GPU_TRAJECTORY_RTOL
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)


def _hmr(config):
    return config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]


def _run(config, generations=1, problem=None):
    algo = EMRHyperNEAT()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conf = algo.create_config(copy.deepcopy(config))
    problem = problem or XORProblem()
    states = [algo.initialize(conf, problem, seed=SEED)]
    fits, metrics = [], None
    for _ in range(generations):
        state, metrics = algo.run_generation(states[-1], problem)
        states.append(state)
        fits.append(np.asarray(algo._cached_fitnesses))
    return algo, states, fits, metrics


def _df(preset, rec="feedforward", **fields):
    config = create_config_with_dynamic_functions(preset, recurrence_preset=rec, **KW)
    _hmr(config)["dynamic_functions"].update(fields)
    return config


# the settling rule

def test_default_is_every_iteration_and_differs_from_tanh():
    default = _run(_df("cppn_output_4", "hidden_only"))[2][0]
    every = _run(_df("cppn_output_4", "hidden_only", hh_activation_mode="every_iteration"))[2][0]
    tanh = _run(_df("cppn_output_4", "hidden_only", hh_activation_mode="tanh"))[2][0]
    _agree(default, every)
    assert not np.allclose(every, tanh), "per-node activations must change the recurrent path"


def test_global_activation_applies_on_the_recurrent_path():
    sin = _run(_df("global_sin", "hidden_only"))[2][0]
    tanh = _run(_df("global_tanh", "hidden_only"))[2][0]
    assert not np.allclose(sin, tanh)


def test_input_aggregation_applies_on_the_recurrent_path():
    configs = {}
    for fn in ("sum", "max"):
        config = create_config_with_recurrence("hidden_only", **KW)
        _hmr(config)["aggregation"] = {"mode": "global", "global_function": fn}
        configs[fn] = _run(config)[2][0]
    assert not np.allclose(configs["sum"], configs["max"])


def _without_hh(monkeypatch):
    """Make discovery return no valid hidden-to-hidden connection."""
    discover = E.discover_sparse_hh_vectorized_multi_hop

    def none_valid(*args, **kwargs):
        sparse = discover(*args, **kwargs)
        return dataclasses.replace(sparse, valid_mask=jnp.zeros_like(sparse.valid_mask),
                                   num_valid=jnp.zeros_like(sparse.num_valid))

    monkeypatch.setattr(E, "discover_sparse_hh_vectorized_multi_hop", none_valid)


def _neuromod(mode, rec):
    return create_config_with_neuromodulation(mode, recurrence_preset=rec, **KW)


def _agg_cppn(rec):
    config = create_config_with_recurrence(rec, **KW)
    _hmr(config)["aggregation"] = {"mode": "cppn_output", "num_aggregations": 4}
    return config


REDUCTION = {
    "disabled": lambda rec: create_config_with_recurrence(rec, **KW),
    "global_sin": lambda rec: _df("global_sin", rec),
    "cppn_output": lambda rec: _df("cppn_output_4", rec),
    # sin, burst, gauss, tanh
    "cppn_output_palette": lambda rec: _df("cppn_output_4", rec, palette=[4, 11, 5, 0]),
    # sin, burst, resonator, gauss
    "cppn_output_oscillatory": lambda rec: _df("cppn_output_4", rec, palette="oscillatory"),
    "aggregation_cppn_output": _agg_cppn,
    "true_neuromodulation": lambda rec: _neuromod("true_neuromodulation_4nt", rec),
    "static_gating": lambda rec: _neuromod("static_gating", rec),
    "context_gating": lambda rec: _neuromod("context_gating", rec),
}
# resonator (sin(x) * exp(-|x| / 3)) is lowered differently by XLA inside the settling loop
# than in a dense pass: the same arithmetic, one float32 ulp apart on some nodes.
# true_neuromodulation under the published node rule (node_rule 'paper', with its additive bias):
# 2 of 20 fitnesses one float32 ulp apart (5.96e-08) at seed 42 on an M4 Max. The node function
# alone gives identical values in a dense pass and inside a scan, so the ulp comes from how XLA
# lowers the surrounding evaluation. The tolerance (4 eps) still catches the defects these tests
# guard (1e-2 or more).
ROUNDING = {"cppn_output_oscillatory", "true_neuromodulation"}


@pytest.mark.parametrize("name", sorted(REDUCTION))
def test_no_hidden_to_hidden_reduces_to_feed_forward(name, monkeypatch):
    """With no h->h connection every settling step is the feed-forward node function."""
    feed_forward = _run(REDUCTION[name]("feedforward"))[2][0]
    _without_hh(monkeypatch)
    recurrent = _run(REDUCTION[name]("hidden_only"))[2][0]
    _agree(recurrent, feed_forward, exact=name not in ROUNDING)


def _random_network(seed=1, n_in=3, n_pos=40, n_out=1, n_conn=60):
    rng = np.random.default_rng(seed)
    f32 = lambda a: jnp.asarray(a.astype(np.float32))  # noqa: E731
    return dict(
        inputs=f32(rng.normal(size=(8, n_in))),
        targets=f32(rng.uniform(size=(8, n_out))),
        W1=f32(rng.normal(size=(n_in, n_pos))),
        W2=f32(rng.normal(size=(n_pos, n_out))),
        frm=jnp.asarray(rng.integers(0, n_pos, n_conn).astype(np.int32)),
        to=jnp.asarray(rng.integers(0, n_pos, n_conn).astype(np.int32)),
        w=f32(rng.normal(size=n_conn)),
        valid=jnp.ones(n_conn, dtype=bool),
        n_pos=n_pos,
    )


def test_data_parallel_evaluator_uses_the_same_rule():
    """The EVAL_ONLY_PARALLEL evaluator settles by the same rule as the standard path."""
    net = _random_network()
    sse = E._pmap_eval_single_hybrid(net["W1"], net["W2"], net["frm"], net["to"], net["w"],
                                     net["valid"], net["inputs"], net["targets"], 5,
                                     net["n_pos"])
    outputs = E.forward_hybrid_vmapped(net["inputs"], net["W1"], net["W2"], net["frm"],
                                       net["to"], net["w"], net["valid"], 5)
    _agree(float(sse), float(jnp.sum((outputs - net["targets"]) ** 2)))


def test_multitask_recurrent_evaluator_reduces_to_its_feed_forward_one():
    algo = EMRHyperNEAT()
    algo.create_config(_neuromod("true_neuromodulation_4nt", "hidden_only"))
    net = _random_network(n_in=2)
    rng = np.random.default_rng(2)
    receptors = jnp.asarray(rng.uniform(-1, 1, size=(1, net["n_pos"], 4)).astype(np.float32))
    gains = jnp.asarray(rng.uniform(0.5, 1, size=(1, net["n_pos"])).astype(np.float32))
    nt = jnp.array([0.9, 0.1, 0.5, 1.0])
    W1, W2 = net["W1"][None], net["W2"][None]
    feed_forward = algo._create_eval_population_true_neuromodulated(has_hh=False)(
        W1, W2, receptors, gains, nt, net["inputs"], net["targets"], "mse", "full", None)
    none_valid = jnp.zeros_like(net["valid"])[None]
    recurrent = algo._create_eval_population_true_neuromodulated(has_hh=True, activate_time=5)(
        W1, W2, receptors, gains, nt, net["inputs"], net["targets"], net["frm"][None],
        net["to"][None], net["w"][None], none_valid, "mse", "full", None)
    _agree(recurrent, feed_forward)


def test_initial_only_is_rejected_with_an_explanation():
    with pytest.raises(ValueError, match="no longer accepted"):
        EMRHyperNEAT().create_config(
            _df("cppn_output_4", "hidden_only", hh_activation_mode="initial_only"))


# NaN, keys, channels

class _NaNTargetXOR(XORProblem):
    def __init__(self):
        super().__init__()
        self.targets = [[0.0], [float("nan")], [1.0], [0.0]]


def test_nan_fitness_becomes_minus_infinity():
    algo, states, fits, metrics = _run(create_config_with_recurrence("feedforward", **KW),
                                       generations=2, problem=_NaNTargetXOR())
    assert np.all(np.isneginf(fits[0])) and np.all(np.isneginf(fits[1]))
    assert metrics.custom_metrics["nan_fitness_count"] == POP
    assert np.isneginf(metrics.best_fitness)


def test_global_max_is_not_sum():
    s = _run(create_config_with_aggregation("global_sum", **KW))
    m = _run(create_config_with_aggregation("global_max", **KW))
    assert m[0].agg_global_function == "max"
    assert not np.allclose(s[2][0], m[2][0])


@pytest.mark.parametrize("section, key", [("aggregation", "global_aggregation"),
                                          ("dynamic_functions", "global_activation")])
def test_misspelled_keys_are_rejected(section, key):
    config = create_config_with_recurrence("feedforward", **KW)
    _hmr(config)[section] = {"mode": "global", key: "max" if section == "aggregation" else "sin"}
    with pytest.raises(ValueError, match="is not a setting"):
        EMRHyperNEAT().create_config(config)


def test_weight_channel_must_be_zero():
    config = _df("cppn_output_4", cppn_output_indices={"weight": 1})
    with pytest.raises(ValueError, match="weight must be 0"):
        EMRHyperNEAT().create_config(config)


# refusals

def test_depth_offload_refuses_per_node_features():
    config = _df("cppn_output_4")
    _hmr(config)["execution"] = {"mode": "depth_offload"}
    with pytest.raises(ValueError, match="depth_offload"):
        EMRHyperNEAT().create_config(config)


def test_eval_only_parallel_refuses_per_node_features():
    config = _df("cppn_output_4")
    _hmr(config)["multi_gpu_strategy"] = "EVAL_ONLY_PARALLEL"
    with pytest.raises(ValueError, match="EVAL_ONLY_PARALLEL"):
        EMRHyperNEAT().create_config(config)


# evaluate_on_data

def _rescaled(variance=0.5):
    config = _df("cppn_output_4")
    _hmr(config)["fitness_rescale_variance"] = variance
    return config


def _accuracy():
    config = _df("cppn_output_4")
    _hmr(config)["fitness_mode"] = "accuracy"
    return config


def _agg_max(rec="feedforward"):
    config = _df("cppn_output_4", rec)
    _hmr(config)["aggregation"] = {"mode": "global", "global_function": "max"}
    return config


CHAMPION = {
    "plain": lambda: create_config_with_recurrence("feedforward", **KW),
    "cppn_output": lambda: _df("cppn_output_4"),
    "cppn_output_palette": lambda: _df("cppn_output_4", palette="oscillatory"),
    "global_sin": lambda: _df("global_sin"),
    "random_fixed": lambda: _df("random_fixed"),
    "weight_interpretation": lambda: _df("weight_interp_variance"),
    "aggregation_max": _agg_max,
    "aggregation_cppn_output": lambda: _agg_cppn("feedforward"),
    "recurrent_plain": lambda: create_config_with_recurrence("hidden_only", **KW),
    "recurrent_cppn_output": lambda: _df("cppn_output_4", "hidden_only"),
    "recurrent_palette": lambda: _df("cppn_output_4", "hidden_only", palette="oscillatory"),
    "recurrent_aggregation_max": lambda: _agg_max("hidden_only"),
    "neuromodulation_true": lambda: _neuromod("true_neuromodulation_4nt", "feedforward"),
    "neuromodulation_true_recurrent": lambda: _neuromod("true_neuromodulation_4nt",
                                                         "hidden_only"),
    "neuromodulation_static_recurrent": lambda: _neuromod("static_gating", "hidden_only"),
    "rescaled": _rescaled,
    "accuracy": _accuracy,
}


@pytest.mark.parametrize("name", sorted(CHAMPION))
def test_evaluate_on_data_reproduces_the_champion_fitness(name):
    """On the training data the held-out scorer returns the fitness the champion was selected on."""
    algo, states, fits, _ = _run(CHAMPION[name]())
    result = algo.evaluate_on_data(states[1], algo._cached_inputs, algo._cached_targets)
    _agree(result["fitness"], float(np.max(fits[0])))
    assert result["training_fitness"] == float(np.max(fits[0]))
    assert fits[0][result["champion_index"]] == np.max(fits[0])


def test_a_tied_champion_is_the_genome_the_state_keeps():
    """Under a tie TensorNEAT keeps the highest index of the top species, not argmax's."""
    algo, states, fits, _ = _run(_rescaled(variance=0.25))  # every genome clamps to 0
    assert np.all(fits[0] == fits[0][0])
    result = algo.evaluate_on_data(states[1], algo._cached_inputs, algo._cached_targets)
    assert result["champion_index"] != 0
    _agree(result["fitness"], float(fits[0][0]))


def test_evaluate_on_data_refuses_another_state():
    algo, states, _, _ = _run(_df("cppn_output_4"))
    with pytest.raises(ValueError, match="not the one that generation returned"):
        algo.evaluate_on_data(states[0], algo._cached_inputs, algo._cached_targets)


def test_evaluate_on_data_needs_a_generation():
    algo = EMRHyperNEAT()
    config = algo.create_config(_df("cppn_output_4"))
    state = algo.initialize(config, XORProblem(), seed=SEED)
    with pytest.raises(RuntimeError, match="run a generation first"):
        algo.evaluate_on_data(state, algo._cached_inputs, algo._cached_targets)


def test_depth_offload_records_no_champion():
    config = create_config_with_recurrence("feedforward", **KW)
    _hmr(config)["execution"] = {"mode": "depth_offload"}
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=SEED)
    state, _ = algo.run_generation(state, problem)
    with pytest.raises(RuntimeError, match="record no champion"):
        algo.evaluate_on_data(state, algo._cached_inputs, algo._cached_targets)


def test_a_multitask_generation_records_no_champion():
    tasks = ["xor", "and"]
    algo = EMRHyperNEAT()
    config = algo.create_config(create_config_with_multi_task(tasks, **KW))
    problems = [MultiTaskProblem(t) for t in tasks]
    state = algo.initialize(config, problems[0], seed=SEED)
    state, _ = algo.run_generation_multitask(
        state, problems=problems, aggregation_method="mean",
        neurotransmitters=[jnp.array([0.95, 0.05, 0.95, 1.0]),
                           jnp.array([0.10, 0.90, 0.10, 1.0])])
    with pytest.raises(RuntimeError, match="record no champion"):
        algo.evaluate_on_data(state, problems[0].inputs, problems[0].targets)


def test_activation_counts_cover_hidden_nodes_only():
    algo, states, _, _ = _run(_df("cppn_output_4"))
    result = algo.evaluate_on_data(states[1], algo._cached_inputs, algo._cached_targets)
    assert sum(result["activation_counts"].values()) == result["active_hidden_nodes"]
    per_level = sum(level["num_hidden_nodes"]
                    for level in result["activation_counts_per_level"].values())
    assert per_level == result["active_hidden_nodes"]


HISTOGRAM_CONFIGS = {
    "cppn_output": lambda: _df("cppn_output_4"),
    "palette": lambda: _df("cppn_output_4", palette=[4, 11, 5, 0]),
    "aggregation": lambda: _agg_cppn("feedforward"),
    "recurrent": lambda: _df("cppn_output_4", "hidden_only"),
}


@pytest.mark.parametrize("name", list(HISTOGRAM_CONFIGS))
def test_best_activation_indices_count_the_champions_hidden_nodes(name):
    """The histograms cover the nodes `evaluate_on_data` counts, named by ACTIVATION_LIST.

    A position the argmax genome dropped is not counted (at the palette's middle entry or
    elsewhere), and a palette entry is named by its function, not by its rank in the palette.
    """
    algo, states, _, _ = _run(HISTOGRAM_CONFIGS[name]())
    info = algo.get_best_activation_indices()
    result = algo.evaluate_on_data(states[1], algo._cached_inputs, algo._cached_targets)
    assert info["num_hidden_nodes"] == result["active_hidden_nodes"]
    assert info["num_positions"] > info["num_hidden_nodes"]
    histograms = [key for key in ("activation_histogram", "aggregation_histogram") if key in info]
    assert histograms
    for key in histograms:
        assert sum(info[key].values()) == info["num_hidden_nodes"]
    if "activation_histogram" in info:
        assert info["activation_histogram"] == result["activation_counts"]
    if name == "palette":
        assert set(info["activation_histogram"]) <= {E.ACTIVATION_LIST[i] for i in (4, 11, 5, 0)}


def test_best_activation_indices_are_empty_without_a_champion():
    assert EMRHyperNEAT().get_best_activation_indices() == {}


def test_evaluate_best_all_modes_uses_the_evolved_forward():
    algo, _, fits, _ = _run(_agg_max())
    modes = algo.evaluate_best_all_modes()
    _agree(modes["mse"], float(np.max(fits[0])))
    assert list(modes) == ["mse", "accuracy", "acc_mse", "hybrid", "bce", "soft_accuracy"]


def _hmr_settled(settling_rule, dynamic_functions=None):
    """One generation of hidden_only (depth 3, population 30, cache off) under the given rule, and
    the fitness HMR's own forward (the frozen dynamic-functions variant, through hmr_reference.py)
    gives the same networks.
    """
    config = create_config_with_recurrence(
        'hidden_only', max_depth=3, population_size=30, hh_cache_enabled=False
    )
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['recurrence']['settling_rule'] = settling_rule
    if dynamic_functions is not None:
        emr['dynamic_functions'] = dynamic_functions
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=42)
    algorithm.run_generation(state, XORProblem())
    W1, W2 = jnp.asarray(algorithm._cached_W1), jnp.asarray(algorithm._cached_W2)
    hh = algorithm._cached_sparse_hh
    inputs, targets, steps = (
        algorithm._cached_inputs,
        algorithm._cached_targets,
        algorithm.activate_time,
    )
    networks = (
        W1,
        W2,
        hh.from_indices,
        hh.to_indices,
        hh.weights,
        hh.valid_mask,
        inputs,
        targets,
        steps,
    )
    functions = {}
    if dynamic_functions is not None:
        # The cached indices are on the compact axis; the networks are not
        compact = np.asarray(algorithm._cached_act_indices)
        active = np.asarray(algorithm._last_eval['eval_positions'])
        full = np.zeros((compact.shape[0], W1.shape[2]), dtype=np.int32)
        full[:, active] = compact[:, :active.shape[0]]
        functions = dict(dynamic_functions=dynamic_functions, act_indices=full,
                         num_activations=algorithm.df_num_activations, palette=algorithm.df_palette)
    hmr_fitness = hmr_value(
        'recurrent_rule.hmr_settled',
        lambda: live('settled_fitness', *networks, **functions),
        inputs=(networks, functions),
    )
    return np.asarray(algorithm._cached_fitnesses), hmr_fitness


class TestHmrSettlingRule:
    """recurrence.settling_rule 'hmr': HMR's dynamic-functions variants settle h = f(x W1), then
    activate_time - 1 steps of h = g(h + recurrent(h)), where EMR settles h = f(x W1 + recurrent(h))
    from zero. The cross-problem control (GECCO Table 3), the per-node paper's recurrent tables and
    the PPSN topology check ran HMR's rule; this opt-in reproduces it for runs that reproduce them.
    Checked against HMR's own forward on the same networks, as recorded in the research repository
    (hmr_reference.py), and the configurations it refuses.
    """

    @pytest.mark.parametrize('dynamic_functions', [
        None,
        {'mode': 'cppn_output', 'num_activations': 4, 'hh_activation_mode': 'initial_only'},
        {'mode': 'cppn_output', 'num_activations': 4, 'hh_activation_mode': 'every_iteration'},
    ])
    def test_matches_hmrs_forward(self, dynamic_functions):
        emr_fitness, hmr_fitness = _hmr_settled('hmr', dynamic_functions)
        if exact_backend():
            np.testing.assert_array_equal(emr_fitness, hmr_fitness)
        else:
            np.testing.assert_allclose(
                emr_fitness, hmr_fitness, rtol=GPU_TRAJECTORY_RTOL, atol=1e-6
            )

    def test_emrs_rule_is_a_different_network(self):
        """The negative control: the same check fails under EMR's rule."""
        emr_fitness, hmr_fitness = _hmr_settled('emr')
        assert np.max(np.abs(emr_fitness - hmr_fitness)) > 1e-2

    @pytest.mark.parametrize(
        'recurrence, extra, match',
        [
            (
                {'preset': 'feedforward', 'settling_rule': 'hmr'},
                {},
                'needs hidden-to-hidden recurrence',
            ),
            # Neuromodulation runs HMR FULL's additive rule (test_additive_settling.py); modulatory
            # neurons stay refused there
            (
                {'preset': 'hidden_only', 'settling_rule': 'hmr'},
                {'neuromodulation': {'mode': 'modulatory_neurons'}},
                "'hmr' with modulatory neurons is refused",
            ),
            (
                {'preset': 'hidden_only', 'settling_rule': 'hmr'},
                {'execution': {'mode': 'depth_offload'}},
                "would ignore .*settling_rule='hmr'",
            ),
            (
                {'preset': 'hidden_only', 'settling_rule': 'frozen'},
                {},
                'settling_rule must be one of',
            ),
            (
                {'preset': 'hidden_only'},
                {
                    'dynamic_functions': {
                        'mode': 'cppn_output',
                        'hh_activation_mode': 'initial_only',
                    }
                },
                'is no longer accepted',
            ),
        ],
    )
    def test_refusals(self, recurrence, extra, match):
        config = create_config_with_recurrence('hidden_only', max_depth=3, population_size=20)
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['recurrence'] = recurrence
        emr.update(extra)
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)


# the settling scan is compiled once per shape

_COMPILES = {"active": False, "count": 0, "registered": False}


def _watch_compiles():
    if not _COMPILES["registered"]:
        def listener(event, duration, **kwargs):
            if _COMPILES["active"] and event == "/jax/core/compile/backend_compile_duration":
                _COMPILES["count"] += 1
        jax.monitoring.register_event_duration_secs_listener(listener)
        _COMPILES["registered"] = True


def _settle_compiles(node_fn, net, **kwargs):
    """Executables compiled by one settling under eager vmap, as the evaluation runs it."""
    drive = (net["inputs"] @ net["W1"])[None]
    _COMPILES["count"] = 0
    _COMPILES["active"] = True
    try:
        jax.block_until_ready(jax.vmap(lambda d: E.settle_hidden(
            d, net["frm"], net["to"], net["w"], net["valid"], 5, node_fn, **kwargs))(drive))
    finally:
        _COMPILES["active"] = False
    return _COMPILES["count"]


@pytest.mark.parametrize("dynamic", [False, True])
def test_a_registry_node_function_compiles_its_settling_once(dynamic):
    """The scan traces its step through a cache keyed on the step's function object, so a step
    made per call compiled the scan again on every call. A registry function's step is made once:
    a second settling with the same shapes compiles nothing. A node function made per call (here a
    lambda) still compiles every call, which also shows the count sees the scan."""
    _watch_compiles()
    net = _random_network(seed=5)
    kwargs = {}
    if dynamic:
        agg = np.random.default_rng(6).integers(0, 4, net["n_pos"]).astype(np.int32)
        kwargs = dict(hh_agg_indices=jnp.asarray(agg), hh_aggregation_mode="dynamic")
    _settle_compiles(jnp.tanh, net, **kwargs)
    assert _settle_compiles(jnp.tanh, net, **kwargs) == 0
    assert _settle_compiles(lambda x: jnp.tanh(x), net, **kwargs) > 0


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("return_states", [False, True])
def test_the_reused_step_settles_as_a_step_made_per_call(dynamic, return_states):
    """The reused step carries the drive and the h->h operands; a step made per call closes over
    them. Same operations in the same order, so the same values (bit for bit on CPU)."""
    net = _random_network(seed=7)
    kwargs = dict(return_states=return_states)
    if dynamic:
        agg = np.random.default_rng(8).integers(0, 4, net["n_pos"]).astype(np.int32)
        kwargs.update(hh_agg_indices=jnp.asarray(agg), hh_aggregation_mode="dynamic")
    args = (net["inputs"] @ net["W1"], net["frm"], net["to"], net["w"], net["valid"], 5)
    reused = E.settle_hidden(*args, jnp.tanh, **kwargs)
    per_call = E.settle_hidden(*args, lambda x: jnp.tanh(x), **kwargs)
    for a, b in zip(jax.tree_util.tree_leaves(reused), jax.tree_util.tree_leaves(per_call)):
        _agree(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("palette", [None, [4, 11, 5, 0], [12, 13]])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("return_states", [False, True])
def test_a_node_rule_settles_as_the_partial_it_replaces(palette, dynamic, return_states):
    """Per-node activation settles through a node rule made once, which the reused step gives the
    genome's indices from its carry; the partial closes over them. Under the evaluation's vmap,
    the same values (bit for bit on CPU), resonator (12) and osc_adapt (13) included."""
    rng = np.random.default_rng(11)
    pop, n_pos = 4, 40
    nets = [_random_network(seed=20 + g) for g in range(pop)]
    stack = lambda key: jnp.stack([net[key] for net in nets])  # noqa: E731
    drive = jnp.stack([net["inputs"] @ net["W1"] for net in nets])
    options = palette or list(range(6))
    indices = jnp.asarray(np.asarray(options)[rng.integers(0, len(options), (pop, n_pos))],
                          dtype=jnp.int32)
    hh_agg = jnp.asarray(rng.integers(0, 4, (pop, n_pos)).astype(np.int32))
    pal = None if palette is None else jnp.array(palette, dtype=jnp.int32)

    def settle(d, frm, to, w, valid, idx, agg, as_rule):
        kwargs = dict(return_states=return_states)
        if dynamic:
            kwargs.update(hh_agg_indices=agg, hh_aggregation_mode="dynamic")
        if as_rule:
            rule, operands = E.node_activation_rule("every_iteration", idx, jnp.tanh, pal, 6)
            return E.settle_hidden(d, frm, to, w, valid, 9, rule, node_operands=operands, **kwargs)
        return E.settle_hidden(d, frm, to, w, valid, 9,
                               E.node_activation("every_iteration", idx, jnp.tanh, pal, 6), **kwargs)

    args = (drive, stack("frm"), stack("to"), stack("w"), stack("valid"), indices, hh_agg)
    with_rule = jax.vmap(lambda *a: settle(*a, True))(*args)
    with_partial = jax.vmap(lambda *a: settle(*a, False))(*args)
    for a, b in zip(jax.tree_util.tree_leaves(with_rule), jax.tree_util.tree_leaves(with_partial)):
        _agree(np.asarray(a), np.asarray(b))


_TRUE_NEUROMODULATION = {"enabled": True, "mode": "true_neuromodulation", "num_nt_types": 4,
                         "receptor_from_weight": True, "receptor_derivation": "tanh",
                         "modulation_strength": 2.0, "modulation_mode": "full"}

#: Recurrent configurations whose node function settles through a step made once: a registry
#: function (tanh nodes, a global activation (sin), input-to-hidden aggregation alone, tanh nodes
#: through the dynamic path) or a node rule (per-node activation, from a CPPN output with or
#: without a palette, from the weights, and with per-node h->h aggregation; each neuromodulation
#: mode, over tanh nodes or per-node activations).
EVALUATION_CONFIGS = {
    "plain": {},
    "global_sin": {"dynamic_functions": {"mode": "global", "hidden_activation": "sin",
                                         "output_activation": "sigmoid"}},
    "aggregation_only": {"aggregation": {"mode": "cppn_output", "num_aggregations": 4}},
    "per_node_cppn_output": {"dynamic_functions": {"mode": "cppn_output", "num_activations": 4}},
    "per_node_palette": {"dynamic_functions": {"mode": "cppn_output", "num_activations": 4,
                                               "palette": [4, 11, 5, 0]}},
    "per_node_weight_interpretation": {"dynamic_functions": {
        "mode": "weight_interpretation", "num_activations": 6, "interpretation": "variance"}},
    "per_node_hh_aggregation": {
        "dynamic_functions": {"mode": "cppn_output", "num_activations": 4},
        "aggregation": {"mode": "cppn_output", "num_aggregations": 4,
                        "hh_aggregation_mode": "dynamic"}},
    "static_gating": {"neuromodulation": {"enabled": True, "mode": "static_gating",
                                          "gate_scaling": "sigmoid"}},
    "context_gating": {"neuromodulation": {"enabled": True, "mode": "context_gating",
                                           "context_dim": 4, "context_influence": 0.5}},
    "modulatory_neurons": {"neuromodulation": {
        "enabled": True, "mode": "modulatory_neurons", "mod_neuron_ratio": 0.1,
        "mod_connection_type": "multiplicative", "mod_decay": 0.9}},
    "true_neuromodulation": {"neuromodulation": _TRUE_NEUROMODULATION},
    "true_neuromodulation_hmr_full_gating": {"neuromodulation": dict(
        _TRUE_NEUROMODULATION, node_rule="hmr_full", modulation_mode="gating_only")},
    "static_gating_per_node": {
        "neuromodulation": {"enabled": True, "mode": "static_gating", "gate_scaling": "sigmoid"},
        "dynamic_functions": {"mode": "cppn_output", "num_activations": 4}},
    # eval_chunk_size below the population: the scan-chunked plain path, and the chunk loop of the
    # others
    "plain_chunked": {"recurrence": {"eval_chunk_size": 7}},
    "plain_chunked_accuracy": {"recurrence": {"eval_chunk_size": 7}, "fitness_mode": "accuracy"},
    "per_node_chunked": {"recurrence": {"eval_chunk_size": 7},
                         "dynamic_functions": {"mode": "cppn_output", "num_activations": 4}},
    "true_neuromodulation_chunked": {"recurrence": {"eval_chunk_size": 7},
                                     "neuromodulation": _TRUE_NEUROMODULATION},
}


@pytest.mark.parametrize("name", sorted(EVALUATION_CONFIGS))
def test_a_recurrent_generation_compiles_nothing_in_its_evaluation(name, monkeypatch):
    """hidden_only at iteration level 1 without the h->h cache: a generation whose evaluation
    shapes were seen before compiles nothing from `_build_forward`'s return to
    `sanitize_fitness`, the evaluation stage. A step made per call would compile the settling
    scan, one executable, every generation, so each path settles through a step made once: the
    plain path, a global activation and aggregation alone (`node_activation` returns the function
    itself rather than a partial), per-node activation (a node rule that carries the indices),
    every neuromodulation mode (a rule that carries its terms), and the scan-chunked evaluation
    (`eval_chunk_size`, whose chunk loop runs in a cached `jit`). A shape counts as seen from
    generation 1 on: the argument types change once after the first generation.
    """
    _watch_compiles()
    build_forward = E.EMRHyperNEAT._build_forward
    sanitize = E.sanitize_fitness
    shapes = {}

    def spy_build_forward(self, *args, **kwargs):
        forward_one, genome_args, chunk_by_scan = build_forward(self, *args, **kwargs)
        shapes["evaluated"] = tuple(None if a is None else a.shape for a in genome_args)
        _COMPILES["count"] = 0
        _COMPILES["active"] = True
        return forward_one, genome_args, chunk_by_scan

    def spy_sanitize(*args, **kwargs):
        _COMPILES["active"] = False
        return sanitize(*args, **kwargs)

    monkeypatch.setattr(E.EMRHyperNEAT, "_build_forward", spy_build_forward)
    monkeypatch.setattr(E, "sanitize_fitness", spy_sanitize)
    config = create_config_with_recurrence("hidden_only", max_depth=2, population_size=POP,
                                           iteration_level=1, hh_cache_enabled=False)
    overrides = copy.deepcopy(EVALUATION_CONFIGS[name])
    _hmr(config)["recurrence"].update(overrides.pop("recurrence", {}))
    _hmr(config).update(overrides)
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=SEED)
    seen, compiles = set(), []
    try:
        for generation in range(8):
            state, _ = algo.run_generation(state, problem)
            jax.block_until_ready(algo._cached_fitnesses)
            if generation >= 2 and shapes["evaluated"] in seen:
                compiles.append((generation, _COMPILES["count"]))
            if generation >= 1:
                seen.add(shapes["evaluated"])
    finally:
        _COMPILES["active"] = False
    assert len(compiles) >= 2, f"the evaluation shapes changed every generation; checked {compiles}"
    assert all(count == 0 for _, count in compiles), compiles


def _generation_compiles(config, generations=6):
    """Executables compiled by each whole `run_generation` call, generation by generation."""
    _watch_compiles()
    algo = EMRHyperNEAT()
    problem = XORProblem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = algo.initialize(algo.create_config(copy.deepcopy(config)), problem, seed=SEED)
    counts = []
    for _ in range(generations):
        _COMPILES["count"] = 0
        _COMPILES["active"] = True
        try:
            state, _ = algo.run_generation(state, problem)
            jax.block_until_ready(algo._cached_fitnesses)
        finally:
            _COMPILES["active"] = False
        counts.append(_COMPILES["count"])
    return counts


#: Feed-forward configurations that query each genome's CPPN at its self-connections: the
#: activation or the aggregation from a dedicated CPPN output, and the neuromodulation decode
#: from the self-connection query. Feed-forward shapes never change, so every generation after
#: the first two can be checked.
SELF_CONNECTION_CONFIGS = {
    "activation_cppn_output": lambda: _df("cppn_output_4"),
    "aggregation_cppn_output": lambda: _agg_cppn("feedforward"),
    "neuromodulation_self_connection_query": lambda: create_config_with_neuromodulation(
        "true_neuromodulation_4nt", recurrence_preset="feedforward",
        use_self_connection_query=True, **KW),
}


@pytest.mark.parametrize("name", sorted(SELF_CONNECTION_CONFIGS))
def test_the_self_connection_query_compiles_once(name):
    """A whole generation compiles nothing once its shapes were seen. The self-connection query
    runs through the weight queries' cached population query: run eagerly, the CPPN forward takes
    the state as a static argument, which hashes by identity, so every generation's new state
    would compile it again, one executable per generation."""
    counts = _generation_compiles(SELF_CONNECTION_CONFIGS[name]())
    assert all(count == 0 for count in counts[2:]), counts


@pytest.mark.parametrize("chunk", [0, 7])
def test_the_self_connection_query_equals_its_eager_form(chunk):
    """The cached population query, which traces the state, equals the eager form, the CPPN
    forward vmapped with the state static: same outputs, bit for bit on CPU, chunked or not."""
    algo, states, _, _ = _run(_df("cppn_output_4", "hidden_only"), generations=2)
    positions = E.get_hierarchical_grid(algo.max_depth).all_positions
    rows = E.self_connection_rows(positions)
    forward = algo._jitted_cppn_forward
    for state in states:
        cppns = algo._compiled_transform_batch(state, algo._compiled_ask(state))
        eager = jax.vmap(lambda c: jax.vmap(lambda x: forward(state, c, x))(rows),  # noqa: B023
                         in_axes=((0, 0, 0, 0),))(tuple(cppns))
        cached = E.batch_query_population_self_connections(state, cppns, positions, forward, 2,
                                                           pop_chunk_size=chunk)
        assert cached.shape == eager.shape == (POP, positions.shape[0], 2)
        _agree(np.asarray(cached), np.asarray(eager))


def test_only_functions_made_once_enter_the_step_cache():
    """Registry functions and node rules, made once per setting, enter the step cache, and a
    second run of a configuration adds nothing to it. A node function made per call (a lambda, a
    partial) does not: caching its step would keep one entry per call alive."""
    _run(_df("cppn_output_4", "hidden_only", palette=[4, 11, 5, 0]), generations=2)
    steps, rules = len(E._SETTLE_STEPS), len(E._NODE_RULES)
    _run(_df("cppn_output_4", "hidden_only", palette=[4, 11, 5, 0]), generations=3)
    assert (len(E._SETTLE_STEPS), len(E._NODE_RULES)) == (steps, rules)
    assert all(E._is_stable_node_fn(key[0]) for key in E._SETTLE_STEPS)
    net = _random_network(seed=9)
    args = (net["inputs"] @ net["W1"], net["frm"], net["to"], net["w"], net["valid"], 5)
    E.settle_hidden(*args, lambda x: jnp.tanh(x))
    indices = jnp.asarray(np.arange(net["n_pos"]) % 4, dtype=jnp.int32)
    E.settle_hidden(*args, E.node_activation("every_iteration", indices))
    assert len(E._SETTLE_STEPS) == steps


def test_without_indices_the_node_function_is_the_configured_function():
    """`hidden_activation` applies `hidden_act_fn` alone when there are no indices, so
    `node_activation` returns that function itself: the same object on every call."""
    from functools import partial
    sin = E.ACTIVATION_FUNCTIONS["sin"]
    assert E.node_activation("every_iteration", None, sin) is sin
    assert E.node_activation("every_iteration") is jnp.tanh
    assert E.node_activation("tanh", None, sin) is jnp.tanh
    pre = jnp.linspace(-3.0, 3.0, 24, dtype=jnp.float32).reshape(4, 6)
    _agree(np.asarray(E.node_activation("every_iteration", None, sin)(pre)),
           np.asarray(E.hidden_activation(pre, None, sin)))
    indices = jnp.array([0, 1, 2, 3, 0, 1])
    assert isinstance(E.node_activation("every_iteration", indices, sin), partial)


def test_a_node_rule_is_made_once_per_setting():
    """`node_activation_rule` gives the function itself without indices (or under 'tanh'), and
    with indices the per-node activation made once per palette and number of activations, the
    indices as its operand: the same rule for every genome and generation, so its settling step
    is made once."""
    sin = E.ACTIVATION_FUNCTIONS["sin"]
    indices = jnp.array([0, 1, 2, 3, 0, 1])
    assert E.node_activation_rule("every_iteration", None, sin) == (sin, ())
    assert E.node_activation_rule("tanh", indices, sin) == (jnp.tanh, ())
    rule, operands = E.node_activation_rule("every_iteration", indices, sin, None, 4)
    assert len(operands) == 1 and operands[0] is indices and E._is_stable_node_fn(rule)
    assert E.node_activation_rule("every_iteration", indices + 0, jnp.tanh, None, 4)[0] is rule
    palette = jnp.array([4, 11, 5, 0], dtype=jnp.int32)
    on_palette = E.node_activation_rule("every_iteration", indices, sin, palette, 4)[0]
    assert on_palette is not rule
    assert E.node_activation_rule("every_iteration", indices, sin, [4, 11, 5, 0], 4)[0] is on_palette
    pre = jnp.linspace(-3.0, 3.0, 24, dtype=jnp.float32).reshape(4, 6)
    _agree(np.asarray(rule(pre, indices)),
           np.asarray(E.node_activation("every_iteration", indices, sin, None, 4)(pre)))
    with pytest.raises(ValueError, match="hh_activation_mode"):
        E.node_activation_rule("initial_only", indices)


def _tnm_node_closure(act_fn, receptors, base_gains, nt, s, num_nt, mode, rule):
    """The true-neuromodulation node as a closure over its terms: the reference."""
    k = min(3, num_nt)
    m = receptors[:, :k] @ nt[:k]
    if rule == "hmr_full":
        gains, gates = base_gains + s * m, jax.nn.sigmoid(m * 5.0)
        if mode == "gating_only":
            return lambda pre: act_fn(pre) * gates
        if mode == "gain_bias_only":
            return lambda pre: act_fn(gains * pre)
        return lambda pre: act_fn(gains * pre) * gates
    if mode == "gating_only":
        gates = jax.nn.sigmoid(m * s)
        return lambda pre: act_fn(pre) * gates
    gains = jnp.clip(base_gains + s * m, 0.1, 5.0)
    gains = jnp.where(jnp.isnan(gains), 1.0, gains)
    bias = m * s
    if mode == "gain_bias_only":
        return lambda pre: act_fn(gains * pre + bias)
    gates = jax.nn.sigmoid(m)
    return lambda pre: act_fn(gains * pre + bias) * gates


def _modulated_closure(act_fn, mod_w, mask, mod_type, mod_decay):
    """The modulatory-neuron node of `_build_forward` as a closure: the reference."""
    def node(pre):
        h = act_fn(pre)
        mod_activity = jnp.sum(h * mask, axis=1, keepdims=True) / jnp.maximum(jnp.sum(mask), 1)
        mod_signal = jnp.tanh(mod_activity) * mod_decay
        if mod_type == "multiplicative":
            return h * (1.0 + mod_signal * mod_w)
        if mod_type == "additive":
            return h + mod_signal * mod_w
        return h * jax.nn.sigmoid(mod_signal * mod_w)
    return node


def _modulated_options_closure(
    act_fn, weights, factor, mask, signal, node_weights, scope, mod_type, mod_decay, gain
):
    """Any combination of the modulatory options as a closure: the reference of the one rule that
    serves them all."""
    def node(pre):
        h = act_fn(pre)
        if signal == "sum_factor":
            s = jnp.tanh(jnp.sum(h * jnp.where(mask, factor, 0.0), axis=1, keepdims=True))
        else:
            s = (
                jnp.tanh(jnp.sum(h * mask, axis=1, keepdims=True) / jnp.maximum(jnp.sum(mask), 1))
                * mod_decay
            )
        if gain == "mean":
            s = jnp.mean(s, axis=0, keepdims=True)
        elif gain == "shift":
            s = jnp.roll(s, 1, axis=0)
        if node_weights == "variance":
            s = s * weights
        if mod_type == "multiplicative":
            changed = h * (1.0 + s)
        elif mod_type == "additive":
            changed = h + s
        else:
            changed = h * jax.nn.sigmoid(s)
        return jnp.where(mask, h, changed) if scope == "standard" else changed
    return node


NEUROMODULATED_CASES = (["gated"] + [f"modulatory_{t}" for t in ("multiplicative", "additive", "gated")]
                        + [f"tnm_{rule}_{mode}" for rule in ("paper", "hmr_full")
                           for mode in ("full", "gating_only", "gain_bias_only")])


@pytest.mark.parametrize("per_node", [False, True])
@pytest.mark.parametrize("case", NEUROMODULATED_CASES)
def test_a_neuromodulated_rule_settles_as_the_closure_it_replaces(case, per_node):
    """Each neuromodulation mode settles through a node rule made once, whose terms (gates, gains,
    bias, modulatory weights and neurons) and activation indices the reused step carries; the
    closures above, the form these rules replaced, close a new function over them on every call.
    Under the evaluation's vmap, the same values (bit for bit on CPU), over tanh nodes and over
    per-node activations with resonator."""
    rng = np.random.default_rng(31)
    pop, n_pos = 4, 40
    nets = [_random_network(seed=40 + g) for g in range(pop)]
    stack = lambda key: jnp.stack([net[key] for net in nets])  # noqa: E731
    f32 = lambda a: jnp.asarray(a.astype(np.float32))  # noqa: E731
    palette = jnp.array([4, 11, 12, 5], dtype=jnp.int32)
    per_genome = dict(
        drive=jnp.stack([net["inputs"] @ net["W1"] for net in nets]),
        frm=stack("frm"), to=stack("to"), w=stack("w"), valid=stack("valid"),
        idx=jnp.asarray(np.asarray(palette)[rng.integers(0, 4, (pop, n_pos))], dtype=jnp.int32),
        gates=f32(rng.uniform(size=(pop, n_pos))),
        mod_w=f32(rng.normal(size=(pop, n_pos))),
        mask=jnp.asarray(rng.uniform(size=(pop, n_pos)) < 0.2),
        receptors=f32(rng.uniform(-1, 1, size=(pop, n_pos, 4))),
        gains=f32(rng.uniform(0.5, 1.0, size=(pop, n_pos))),
    )
    nt = jnp.array([0.9, 0.1, 0.5, 1.0])

    def settle(a, as_rule):
        idx = a["idx"] if per_node else None
        if as_rule:
            act = E.node_activation_rule("every_iteration", idx, jnp.tanh, palette, 4)
            if case == "gated":
                node_fn, operands = E.gated_node_rule(act, a["gates"])
            elif case.startswith("modulatory"):
                node_fn, operands = E.modulatory_node_rule(act, (a["mod_w"], a["mask"]),
                                                           case.split("_")[1], mod_decay=0.9)
            else:
                rule, mode = case[4:].split("_", 1) if case.startswith("tnm_paper") else \
                    ("hmr_full", case[len("tnm_hmr_full_"):])
                node_fn, operands = E.true_neuromodulation_node_rule(
                    act, a["receptors"], a["gains"], nt, 2.0, 4, mode, rule)
        else:
            act_fn = E.node_activation("every_iteration", idx, jnp.tanh, palette, 4)
            operands = ()
            if case == "gated":
                node_fn = lambda p: act_fn(p) * a["gates"]  # noqa: E731
            elif case.startswith("modulatory"):
                node_fn = _modulated_closure(act_fn, a["mod_w"], a["mask"], case.split("_")[1], 0.9)
            else:
                rule, mode = case[4:].split("_", 1) if case.startswith("tnm_paper") else \
                    ("hmr_full", case[len("tnm_hmr_full_"):])
                node_fn = _tnm_node_closure(act_fn, a["receptors"], a["gains"], nt, 2.0, 4, mode,
                                            rule)
        return E.settle_hidden(a["drive"], a["frm"], a["to"], a["w"], a["valid"], 9, node_fn,
                               node_operands=operands)

    with_rule = jax.vmap(lambda a: settle(a, True))(per_genome)
    with_closure = jax.vmap(lambda a: settle(a, False))(per_genome)
    _agree(np.asarray(with_rule), np.asarray(with_closure))


#: (signal, node weights, scope, matched gain): the two bundles, mixed combinations, both controls.
MODULATORY_OPTION_CASES = [
    ("mean_decay", "variance", "all", "none"),
    ("sum_factor", "none", "standard", "none"),
    ("sum_factor", "variance", "all", "none"),
    ("mean_decay", "none", "standard", "none"),
    ("sum_factor", "variance", "standard", "mean"),
    ("mean_decay", "variance", "all", "shift"),
    ("mean_decay", "none", "all", "mean"),
    ("sum_factor", "none", "all", "shift"),
]


@pytest.mark.parametrize("per_node", [False, True])
@pytest.mark.parametrize("mod_type", ["multiplicative", "gated"])
@pytest.mark.parametrize("signal, node_weights, scope, gain", MODULATORY_OPTION_CASES)
def test_the_modulatory_rule_settles_as_its_closure(
    signal, node_weights, scope, gain, mod_type, per_node
):
    """Every combination of the modulatory options settles through the one rule, whose arrays
    (per-node weights, factors, mask, as the options need them) and activation indices the reused
    step carries: under the evaluation's vmap, the same values as a closure over them (bit for bit
    on CPU), over tanh nodes and over per-node activations with resonator."""
    rng = np.random.default_rng(37)
    pop, n_pos = 4, 40
    nets = [_random_network(seed=60 + g) for g in range(pop)]
    stack = lambda key: jnp.stack([net[key] for net in nets])  # noqa: E731
    f32 = lambda a: jnp.asarray(a.astype(np.float32))  # noqa: E731
    palette = jnp.array([4, 11, 12, 5], dtype=jnp.int32)
    per_genome = dict(
        drive=jnp.stack([net["inputs"] @ net["W1"] for net in nets]),
        frm=stack("frm"), to=stack("to"), w=stack("w"), valid=stack("valid"),
        idx=jnp.asarray(np.asarray(palette)[rng.integers(0, 4, (pop, n_pos))], dtype=jnp.int32),
        weights=f32(rng.normal(size=(pop, n_pos))),
        factor=f32(rng.uniform(-1, 1, size=(pop, n_pos))),
        mask=jnp.asarray(rng.uniform(size=(pop, n_pos)) < 0.2),
    )

    def settle(a, as_rule):
        idx = a["idx"] if per_node else None
        if as_rule:
            act = E.node_activation_rule("every_iteration", idx, jnp.tanh, palette, 4)
            arrays = (((a["weights"],) if node_weights == "variance" else ())
                      + ((a["factor"],) if signal == "sum_factor" else ()) + (a["mask"],))
            node_fn, operands = E.modulatory_node_rule(
                act, arrays, mod_type, signal, node_weights, scope, 0.9, gain
            )
        else:
            act_fn = E.node_activation("every_iteration", idx, jnp.tanh, palette, 4)
            node_fn = _modulated_options_closure(
                act_fn,
                a["weights"],
                a["factor"],
                a["mask"],
                signal,
                node_weights,
                scope,
                mod_type,
                0.9,
                gain,
            )
            operands = ()
        return E.settle_hidden(a["drive"], a["frm"], a["to"], a["w"], a["valid"], 9, node_fn,
                               node_operands=operands)

    with_rule = jax.vmap(lambda a: settle(a, True))(per_genome)
    with_closure = jax.vmap(lambda a: settle(a, False))(per_genome)
    _agree(np.asarray(with_rule), np.asarray(with_closure))


def test_a_neuromodulated_rule_is_made_once_per_setting():
    """A neuromodulated rule is kept per form and activation (and for modulatory neurons per
    connection type and decay), so every genome and generation shares it; over an activation
    made per call (a partial) it is made per call and not kept."""
    tanh = (jnp.tanh, ())
    gates = jnp.ones(6)
    assert E.gated_node_rule(tanh, gates)[0] is E.gated_node_rule(tanh, gates * 2)[0]
    mod = E.modulatory_node_rule(tanh, (gates, gates > 0), "additive", mod_decay=0.9)[0]
    assert mod is E.modulatory_node_rule(tanh, (gates, gates > 0), "additive", mod_decay=0.9)[0]
    assert mod is not E.modulatory_node_rule(tanh, (gates, gates > 0), "additive", mod_decay=0.7)[0]
    rules = len(E._NODE_RULES)
    partial = E.node_activation("every_iteration", jnp.arange(6) % 4)
    assert E.gated_node_rule((partial, ()), gates)[0] is not E.gated_node_rule((partial, ()), gates)[0]
    assert len(E._NODE_RULES) == rules


@pytest.mark.parametrize("per_task_activation", [False, True])
def test_a_multitask_generation_compiles_nothing(per_task_activation):
    """A recurrent multi-task generation (XOR and AND, true neuromodulation) compiles nothing once
    its shapes were seen; multi-task evaluation keeps every position, so they repeat from the
    first generations on. New evaluators every generation would compile their settling again for
    the first task's single-task evaluation and for each task: 1 + T executables."""
    _watch_compiles()
    tasks = ["xor", "and"]
    config = create_config_with_multi_task(tasks, max_depth=2, population_size=POP)
    rec = _hmr(config)["recurrence"]
    rec.update(iteration_level=1, hh_cache_enabled=False)
    if per_task_activation:
        _hmr(config)["multitask"]["per_task_activation"] = {"xor": "sin", "and": "tanh"}
    algo = EMRHyperNEAT()
    problems = [MultiTaskProblem(t) for t in tasks]
    nts = [jnp.array(E.NT_TASK_PRESETS[t]) for t in tasks]
    state = algo.initialize(algo.create_config(config), problems[0], seed=SEED)
    counts = []
    for _ in range(6):
        _COMPILES["count"] = 0
        _COMPILES["active"] = True
        try:
            state, _ = algo.run_generation_multitask(state, problems=problems, neurotransmitters=nts,
                                                     aggregation_method="min")
            jax.block_until_ready(state)
        finally:
            _COMPILES["active"] = False
        counts.append(_COMPILES["count"])
    assert all(count == 0 for count in counts[2:]), counts


def test_the_multitask_evaluators_are_made_once_per_setting():
    """The multi-task evaluator is kept per setting it closes over; an activation made per call
    gives a new evaluator on every call, which is not kept."""
    algo = EMRHyperNEAT()
    algo.create_config(_neuromod("true_neuromodulation_4nt", "hidden_only"))
    make = algo._create_eval_population_true_neuromodulated
    sin = E.ACTIVATION_FUNCTIONS["sin"]
    assert make(has_hh=True, activate_time=5, hidden_act_fn=sin) is make(
        has_hh=True, activate_time=5, hidden_act_fn=sin)
    assert make(has_hh=True, activate_time=5, hidden_act_fn=sin) is not make(
        has_hh=True, activate_time=6, hidden_act_fn=sin)
    assert make(has_hh=False) is make(has_hh=True, activate_time=0)
    kept = len(E._MULTITASK_EVALUATORS)
    assert make(has_hh=True, activate_time=5, hidden_act_fn=lambda x: jnp.sin(x)) is not make(
        has_hh=True, activate_time=5, hidden_act_fn=lambda x: jnp.sin(x))
    assert len(E._MULTITASK_EVALUATORS) == kept


def test_the_union_multi_hop_expansion_is_kept_per_setting():
    """The union discovery's multi-hop expansion is one `jit(vmap(...))` per device and setting,
    a setting keyed with its type."""
    device = jax.devices()[0]
    get = E._get_union_multi_hop_fn
    expand = get(device, "matrix_power", 0.3, 2, 0.8, 3.0, "masked")
    assert get(device, "matrix_power", 0.3, 2, 0.8, 3.0, "masked") is expand
    assert get(device, "fori_loop", 0.3, 2, 0.8, 3.0, "masked") is not expand
    assert get(device, "matrix_power", 0.3, 3, 0.8, 3.0, "masked") is not expand
    assert get(device, "matrix_power", 0.3, 2, 0.8, 3.0, "raw") is not expand
    assert get(device, "matrix_power", 0.3, 2, 0.8, 3, "masked") is not expand


@pytest.mark.parametrize("second_factor", ["masked", "raw"])
def test_a_union_discovery_compiles_nothing_for_a_union_seen_before(second_factor, monkeypatch):
    """`hh_discovery_mode: union` at iteration level 2 (with `raw`, the setting that reproduces a
    0.1.0 recurrent run): a discovery over a union of active positions whose size was seen before
    compiles nothing, so the multi-hop expansion, `jit(vmap(expand_single_genome))`, is not built
    and compiled again on every discovery."""
    _watch_compiles()
    discover = E.discover_sparse_hh_vectorized_multi_hop
    calls = []

    def spy(state, cppns, h_grid, masks_A, *args, **kwargs):
        union = int(np.asarray(masks_A).any(axis=0).sum())
        _COMPILES["count"] = 0
        _COMPILES["active"] = True
        try:
            result = discover(state, cppns, h_grid, masks_A, *args, **kwargs)
            jax.block_until_ready(result.weights)
        finally:
            _COMPILES["active"] = False
        calls.append((union, _COMPILES["count"]))
        return result

    monkeypatch.setattr(E, "discover_sparse_hh_vectorized_multi_hop", spy)
    config = create_config_with_recurrence("hidden_only", max_depth=3, population_size=POP,
                                           iteration_level=2, hh_cache_enabled=False)
    _hmr(config)["recurrence"].update(hh_discovery_mode="union",
                                      multi_hop_second_factor=second_factor)
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=SEED)
    for _ in range(10):
        state, _ = algo.run_generation(state, problem)
    seen, checked = set(), []
    for call, (union, count) in enumerate(calls):
        if call >= 2 and union in seen:
            checked.append(count)
        if call >= 1:
            seen.add(union)
    assert len(checked) >= 2, f"the union size changed at every discovery: {calls}"
    assert all(count == 0 for count in checked), calls
