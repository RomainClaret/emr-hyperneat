"""The recurrent path's one settling rule, the champion record, and the refusals added in 0.2.0.

In 0.1.0 the recurrent path applied no configured hidden activation under its default
(`hh_activation_mode: initial_only`), ignored input-to-hidden aggregation, and had three other
update rules in the neuromodulated, multi-task and data-parallel evaluators;
`evaluate_on_data` re-derived the champion as a dense tanh network; NaN fitness reached
selection; and several configurations were accepted and partly ignored. These tests hold the
fixed behavior: every recurrent evaluation settles by one rule that reduces to the
feed-forward evaluation without hidden-to-hidden connections, `evaluate_on_data` reproduces
the fitness the champion was selected on, and what a path cannot apply is refused.
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


# -- the settling rule ------------------------------------------------------------------------

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
    """The EVAL_ONLY_PARALLEL evaluator used an older rule in 0.1.0."""
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


# -- NaN, keys, channels ----------------------------------------------------------------------

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


# -- refusals -----------------------------------------------------------------------------------

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


# -- evaluate_on_data -------------------------------------------------------------------------

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

    In 0.1.0 they covered every position of the argmax genome, a dropped one at the
    palette's middle entry, and named a palette entry by its rank in the palette.
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


# -- the settling scan is compiled once per shape ---------------------------------------------

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


#: Recurrent configurations whose node function is a registry function: tanh nodes, a global
#: activation (sin) and input-to-hidden aggregation alone (tanh nodes through the dynamic path).
REGISTRY_NODE_FUNCTION_CONFIGS = {
    "plain": {},
    "global_sin": {"dynamic_functions": {"mode": "global", "hidden_activation": "sin",
                                         "output_activation": "sigmoid"}},
    "aggregation_only": {"aggregation": {"mode": "cppn_output", "num_aggregations": 4}},
}


@pytest.mark.parametrize("name", sorted(REGISTRY_NODE_FUNCTION_CONFIGS))
def test_a_recurrent_generation_compiles_nothing_in_its_evaluation(name, monkeypatch):
    """hidden_only at iteration level 1 without the h->h cache, with a registry node function: a
    generation whose evaluation shapes were seen before compiles nothing from `_build_forward`'s
    return to `sanitize_fitness`, the evaluation stage. With a step made per call it compiled the
    settling scan, one executable, every generation; a global activation and aggregation alone
    did so until `node_activation` returned the function itself rather than a partial. A shape
    counts as seen from generation 1 on: the argument types change once after the first
    generation."""
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
    _hmr(config).update(copy.deepcopy(REGISTRY_NODE_FUNCTION_CONFIGS[name]))
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


def test_a_node_function_made_per_call_is_not_cached():
    """Per-node activation settles through a partial made on every call: caching its step would
    keep one entry per call alive. Only registry functions enter the cache."""
    before = len(E._SETTLE_STEPS)
    _run(_df("cppn_output_4", "hidden_only"), generations=3)
    assert len(E._SETTLE_STEPS) == before
    assert all(any(key[0] is fn for fn in E._STABLE_NODE_FNS) for key in E._SETTLE_STEPS)


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
