"""A global 'softmax' hidden activation on an axis shorter than the grid.

Global softmax (ACTIVATION_FUNCTIONS['softmax'], not one of ACTIVATION_LIST's per-node functions)
normalizes across the hidden positions. Its definition is the uncompacted one, every position of the
grid at the configured depth, as the feed-forward evaluation and every uncompacted recurrent
generation evaluate it. Two evaluations run on a shorter axis: a compacted recurrent generation, on
the union of its population's active positions (padded to a bucket), and the substrate API, on each
genome's own positions. Normalized over that axis alone, a genome's fitness in the loop would depend
on the rest of its population, and the API would disagree with the loop. Both add the positions
their axis leaves out (`full_grid_softmax`): each has zero weights in and out, so it adds exp(c - m)
to the denominator, c being the input a zero-weight position's softmax receives at that step, and
the bucket's padding adds nothing. evaluate_with_neurotransmitter under 'softmax' after a compacted
generation does the same.

The tests:
- one genome in two populations whose compacted unions differ (execution.compaction_bucket 'exact',
  the second population the first with some genomes replaced, the unions read from the runs) gets
  the same fitness and outputs, with and without true neuromodulation, under both settling rules;
- a compacted generation equals the same population evaluated uncompacted (a generation kept on the
  ceiling as the list form of run_generation_multitask keeps its inner one), for each receptor
  source (Option A from the first input weight or the self-connection query, Option B), for the
  gating modes and with bucket padding; and so does evaluate_with_neurotransmitter, with and without
  branch gating;
- the API equals the loop on the same genomes, plain and under true neuromodulation, for the node
  rules it supports;
- uncompacted controls (a feed-forward generation, a recurrent one whose union is the whole grid,
  h->h 'tanh', which never applies softmax) build the forward without the completion and evaluate as
  the registry's softmax written out does (two programs, bit for bit under single-threaded Eigen).
test_the_comparison_refuses_the_compacted_axis_alone shows that the comparisons see an evaluation on
the compacted axis alone, and test_the_api_counts_the_grid_for_a_genome_alone (a genome evaluated
alone is its row of the population) guards the count of each genome's own missing positions.

How they compare. Two evaluations on axes of different lengths are two programs: on the CPU they
round apart, by a few ulps under single-threaded Eigen and more under XLA's default multi-threaded
Eigen, and a genome whose settling is chaotic carries that rounding to any size
(test_substrate_recurrent_neuromodulation.py). So they are compared as two programs are (conftest's
assert_close_across_programs) over the genomes whose outputs one ulp of their weights moves by at
most STEADY_OUTPUTS, at least a quarter of the population. Bit for bit only on the CPU: where one
program runs (a genome alone against its row of the population), and under single-threaded Eigen
(exact_backend() and not cpu_multi_threaded_eigen()) the uncompacted controls against the
registry softmax written out, two programs that an Apple M4 Max computed alike, natively and as
x86-64 code under Rosetta 2. An evaluation on the compacted axis alone is 1e-4 to 0.69 from the
references, and each comparison refuses it (test_the_comparison_refuses_the_compacted_axis_alone).
Off the CPU the comparisons take allowances set for other comparisons (GPU_ATOL, conftest's
GPU_TRAJECTORY_RTOL), not measured on these cases, and no CPU comparison is loosened for them.
"""

import copy
import functools
import inspect
import warnings
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    GPU_TRAJECTORY_RTOL,
    RECURRENCE_PRESETS,
    XORProblem,
    assert_close_across_programs,
    cpu_multi_threaded_eigen,
    create_base_config,
    create_config_with_neuromodulation,
    exact_backend,
)
from emr_hyperneat import substrate_forward
from emr_hyperneat import emrhyperneat as engine
from emr_hyperneat import substrate as api

DEPTH = 2
POP = 30
SEED = 42
#: A genome whose outputs one ulp of its weights moves by more than this (about eight ulps of an
#: output near 1) is left out of a comparison of two programs: they round apart by a few ulps, and
#: its settling would carry that past conftest's allowance of 128 ulps.
STEADY_OUTPUTS = 1e-6
#: The seeds tried, in order, until a generation compacts (its union below the grid): which seed
#: does moves with the PRNG default (jax_threefry_partitionable), so none is assumed
SEEDS = (SEED,) + tuple(range(16))
#: Off the CPU, the steady genomes' fitness within this
#: (test_substrate_recurrent_neuromodulation.py's GPU_FORWARD_ATOL): a GPU rounds the two layouts
#: apart and the settling can amplify that. Borrowed, not measured on these cases.
GPU_ATOL = 1e-3
GLOBAL_SOFTMAX = {"mode": "global", "hidden_activation": "softmax", "output_activation": "sigmoid"}


def _emr(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]


def _config(neuromodulation: Optional[Dict[str, Any]] = None, settling: str = "emr",
            hh_mode: Optional[str] = None, preset: str = "full_recurrent", bucket: str = "exact",
            hidden: str = "softmax", pop: int = POP) -> Dict[str, Any]:
    """XOR at depth 2 with a global hidden activation (softmax by default), recurrent at iteration
    level 1 without the h->h cache; ``neuromodulation`` the true_neuromodulation_4nt preset's
    overrides (Option A from the first input weight by default), None for none; ``preset`` None for
    feed-forward."""
    if neuromodulation is not None:
        config = create_config_with_neuromodulation(
            "true_neuromodulation_4nt", recurrence_preset=preset or "feedforward", max_depth=DEPTH,
            population_size=pop, **neuromodulation)
    else:
        config = create_base_config(max_depth=DEPTH, population_size=pop)
        _emr(config)["recurrence"] = (RECURRENCE_PRESETS[preset].copy() if preset is not None
                                      else {"preset": "feedforward", "iteration_level": 0})
    emr = _emr(config)
    if preset is not None:
        emr["recurrence"].update(iteration_level=1, hh_cache_enabled=False, settling_rule=settling)
    emr["dynamic_functions"] = dict(GLOBAL_SOFTMAX, hidden_activation=hidden)
    if hh_mode is not None:
        emr["dynamic_functions"]["hh_activation_mode"] = hh_mode
    config["algorithm_params"]["emrhyperneat"]["substrate"]["hidden_activation"] = hidden
    emr.setdefault("execution", {})["compaction_bucket"] = bucket
    return config


#: name -> configuration: without neuromodulation and under true neuromodulation (the paper's rule
#: and the steep_gate rule of the prototype's FULL variant), under both settling rules
#: ('drive_once' with 'every_iteration' carries each missing position's own state from step to
#: step; 'initial_only' applies softmax at the first step)
RULES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "plain_emr": lambda: _config(),
    "plain_drive_once_every_iteration": lambda: _config(settling="drive_once"),
    "plain_drive_once_initial_only": lambda: _config(settling="drive_once", hh_mode="initial_only"),
    "paper_emr": lambda: _config({}),
    "steep_gate_drive_once_initial_only": lambda: _config({"node_rule": "steep_gate"},
                                                          settling="drive_once",
                                                          hh_mode="initial_only"),
}
#: Beyond RULES, for the compacted generation against the uncompacted one: the other receptor
#: sources, whose missing positions each derive their own receptors (Option A from the
#: self-connection query, Option B from dedicated CPPN outputs), a modulation mode without the bias,
#: and 'anti' (the gating modes and bucket padding have tests of their own)
SOURCES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "paper_emr_self_query": lambda: _config({"use_self_connection_query": True}),
    "paper_emr_option_b": lambda: _config({"receptor_from_weight": False}),
    "paper_drive_once_option_b": lambda: _config({"receptor_from_weight": False},
                                                 settling="drive_once", hh_mode="initial_only"),
    "paper_emr_gain_bias_only": lambda: _config({"modulation_mode": "gain_bias_only"}),
    "anti_emr": lambda: _config({"node_rule": "anti", "use_output_inversion": False}),
}


class Generation(NamedTuple):
    """One generation as the loop evaluated it: its forward and arguments (`_build_forward`, spied),
    the outputs and fitness they give, and the union of active positions it compacted to (None:
    none)."""

    algo: Any
    state: Any
    forward_one: Callable
    genome_args: Tuple
    completion: Any
    outputs: np.ndarray
    fitness: np.ndarray
    union: Optional[int]


def _initialized(config: Dict[str, Any], seed: int = SEED) -> Tuple[Any, Any]:
    algo = engine.EMRHyperNEAT()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        state = algo.initialize(algo.create_config(copy.deepcopy(config)), XORProblem(), seed=seed)
    return algo, state


def _generation(algo: Any, state: Any, uncompacted: bool = False) -> Generation:
    """The loop's generation on ``state``; ``uncompacted`` keeps it on the ceiling, as the list form
    of run_generation_multitask keeps its inner generation (test_multitask_after_compaction.py)."""
    build = engine.EMRHyperNEAT._build_forward
    kept = {}

    def spy(self, *args, **kwargs):
        kept["completion"] = kwargs.get("softmax_completion")
        kept["built"] = build(self, *args, **kwargs)
        return kept["built"]

    algo._multitask_inner_generation = uncompacted
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(engine.EMRHyperNEAT, "_build_forward", spy)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                algo.run_generation(state, XORProblem())
    finally:
        algo._multitask_inner_generation = False
    forward_one, genome_args, _ = kept["built"]
    inputs = algo._cached_inputs
    outputs = np.asarray(jax.vmap(lambda *args: forward_one(inputs, *args))(*genome_args))
    indices = algo._cached_active_indices
    return Generation(algo, state, forward_one, genome_args, kept.get("completion"), outputs,
                      np.asarray(algo._cached_fitnesses), None if indices is None else len(indices))


def _compacting(config: Dict[str, Any]) -> Tuple[Any, Generation, int]:
    """The state, generation and seed of the first of SEEDS whose generation compacts."""
    for seed in SEEDS:
        algo, state = _initialized(config, seed)
        generation = _generation(algo, state)
        if generation.union is not None:
            return state, generation, seed
    pytest.fail(f"no seed of {SEEDS} gives a generation that compacts")


def _response(generation: Generation, draws: int = 3) -> np.ndarray:
    """Per genome, the largest output change when every entry of its W1 and W2 moves by one ulp
    (random signs, ``draws`` draws), the loop's forward otherwise unchanged."""
    rng = np.random.default_rng(0)
    inputs = generation.algo._cached_inputs
    response = np.zeros(generation.outputs.shape[0])
    for _ in range(draws):
        args = list(generation.genome_args)
        for i in (0, 1):
            array = np.asarray(args[i])
            signs = rng.choice([-1.0, 1.0], size=array.shape)
            args[i] = jnp.asarray((array * (1 + 2.0 ** -23 * signs)).astype(array.dtype))
        moved = np.asarray(jax.vmap(lambda *a: generation.forward_one(inputs, *a))(*args))
        diff = np.abs(moved - generation.outputs).reshape(response.shape[0], -1)
        response = np.maximum(response, np.where(np.isnan(diff), np.inf, diff).max(axis=1))
    return response


def _assert_same_evaluation(outputs: np.ndarray, fitness: np.ndarray, reference: Generation,
                            genomes: Optional[np.ndarray] = None) -> None:
    """Two programs' outputs and fitness of the same genomes against ``reference``'s: NaN at the
    same entries, then the genomes steady at one ulp (at least a quarter of ``genomes``, the floor
    of test_substrate_recurrent_neuromodulation.py) as two programs are compared on the CPU
    (assert_close_across_programs), within GPU_ATOL in fitness off it."""
    genomes = np.arange(reference.outputs.shape[0]) if genomes is None else genomes
    np.testing.assert_array_equal(np.isnan(outputs[genomes]), np.isnan(reference.outputs[genomes]))
    steady = genomes[_response(reference)[genomes] <= STEADY_OUTPUTS]
    floor = max(1, genomes.size // 4)
    assert steady.size >= floor, (f"{steady.size} of {genomes.size} genomes steady at one ulp, "
                                  f"fewer than {floor}")
    if exact_backend():
        assert_close_across_programs(outputs[steady], reference.outputs[steady])
        assert_close_across_programs(fitness[steady], reference.fitness[steady])
    else:
        np.testing.assert_allclose(fitness[steady], reference.fitness[steady], rtol=0,
                                   atol=GPU_ATOL)


def _sanitized(fitness: Any) -> np.ndarray:
    """Fitness as the loop records it (NaN to -inf)."""
    return np.asarray(engine.sanitize_fitness(jnp.asarray(fitness))[0])


# One genome in two populations

def _with_replaced(state: Any, replaced: np.ndarray, donor: int) -> Any:
    """``state`` with the genomes ``replaced`` copies of genome ``donor``."""
    nodes, conns = np.asarray(state.pop_nodes).copy(), np.asarray(state.pop_conns).copy()
    nodes[replaced], conns[replaced] = nodes[donor], conns[donor]
    return state.update(pop_nodes=jnp.asarray(nodes), pop_conns=jnp.asarray(conns))


@functools.lru_cache(maxsize=None)
def _two_populations(name: str) -> Tuple[Generation, Generation, np.ndarray]:
    """The case's first generation (a seed's population that compacts and has a genome other than
    genome 0 that alone reaches some position) and the same generation with those genomes replaced
    by copies of genome 0, which shrinks the union; the genomes kept."""
    config = RULES[name]()
    for seed in SEEDS:
        algo, state = _initialized(config, seed)
        first = _generation(algo, state)
        masks = np.asarray(algo._cached_masks_A, dtype=bool)
        reached = masks.sum(axis=0)
        owners = np.flatnonzero((masks & (reached == 1)[None, :]).any(axis=1))
        owners = owners[owners != 0][: POP // 2]
        if first.union is not None and owners.size:
            break
    else:
        pytest.fail(f"no seed of {SEEDS} compacts with a genome alone at some position")
    kept = np.setdiff1d(np.arange(POP), owners)
    algo_b, _ = _initialized(config, seed)
    second = _generation(algo_b, _with_replaced(state, owners, 0))
    return first, second, kept


@pytest.mark.parametrize("name", sorted(RULES))
def test_one_genome_in_two_populations_evaluates_alike(name):
    """The genomes two populations share get the same outputs and fitness, although the two
    compacted unions differ (read from the runs): softmax counts every grid position in both.
    Normalized over its own union, a genome's fitness would depend on the genomes beside it."""
    first, second, kept = _two_populations(name)
    assert first.union is not None and second.union is not None, "both generations compacted"
    assert second.union < first.union, (first.union, second.union)
    _assert_same_evaluation(second.outputs, second.fitness, first, kept)
    assert first.completion is not None and second.completion is not None


# A compacted generation against the same population uncompacted

@functools.lru_cache(maxsize=None)
def _compacted_and_not(name: str) -> Tuple[Generation, Generation]:
    config = {**RULES, **SOURCES}[name]()
    state, compacted, seed = _compacting(config)
    algo_u, _ = _initialized(config, seed)
    return compacted, _generation(algo_u, state, uncompacted=True)


@pytest.mark.parametrize("name", sorted({**RULES, **SOURCES}))
def test_a_compacted_generation_equals_the_uncompacted_one(name):
    """The compacted generation's outputs and fitness are those of the same population evaluated on
    the whole grid, for each rule, receptor source (the missing positions' own receptors under the
    self-connection query and Option B) and mode."""
    compacted, whole = _compacted_and_not(name)
    ceiling = whole.algo._cached_W1.shape[-1]
    assert whole.union is None and whole.genome_args[0].shape[-1] == ceiling
    assert compacted.union is not None and compacted.union < ceiling, "the generation compacted"
    _assert_same_evaluation(compacted.outputs, compacted.fitness, whole)
    assert compacted.completion is not None and whole.completion is None


#: Bucket padding: the configurations a padded axis is tested under (compaction_bucket 'auto'), and
#: the seeds tried, in order, until a generation's union is below its bucket's length and the bucket
#: below the grid, so the axis holds padding and leaves grid positions out (which seed does moves
#: with the PRNG default, so none is assumed)
PADDED: Dict[str, Dict[str, Any]] = {
    "plain_emr": {},
    "plain_drive_once_every_iteration": {"settling": "drive_once"},
    "paper_emr": {"neuromodulation": {}},
    "paper_emr_option_b": {"neuromodulation": {"receptor_from_weight": False}},
}
PADDING_SEEDS = tuple(range(16))
#: A smaller population, whose unions fall below the bucket of 64 under either PRNG default
PADDING_POP = 10


@pytest.mark.parametrize("name", sorted(PADDED))
def test_bucket_padding_counts_once(name):
    """Under compaction_bucket 'auto' the compact axis is padded to its bucket with positions that
    have zero weights and no connection and are no grid position: the softmax leaves them out (their
    output 0) and counts every grid position outside the union once, through the virtual position,
    so a generation that pads equals the same population uncompacted. Entered as positions of
    their own (a zero drive without neuromodulation; under Option B the receptors of a zero
    self-connection query, not those of any grid position), the padding would stand in for the
    positions outside the axis exactly only where a bucket reaches the grid's length, so the seed
    is one whose bucket is below it."""
    for seed in PADDING_SEEDS:
        config = _config(**PADDED[name], bucket="auto", pop=PADDING_POP)
        algo, state = _initialized(config, seed)
        padded = _generation(algo, state)
        length, ceiling = padded.genome_args[0].shape[-1], algo._cached_W1.shape[-1]
        if padded.union is not None and padded.union < length < ceiling:
            break
    else:
        pytest.fail(f"no seed of {PADDING_SEEDS} pads the union of {name} below the grid's length")
    algo_u, _ = _initialized(config, seed)
    whole = _generation(algo_u, state, uncompacted=True)
    _assert_same_evaluation(padded.outputs, padded.fitness, whole)
    completion = padded.completion
    assert completion is not None
    ceiling = whole.algo._cached_W1.shape[-1]
    assert completion.real == padded.union < completion.length
    assert completion.missing == ceiling - padded.union


@pytest.mark.parametrize("mode", ["static_gating", "context_gating"])
def test_the_gating_modes_on_a_compacted_axis(mode):
    """Static and context gating gate the activation after it, so a missing position's softmax input
    is its zero drive: the compacted generation equals the uncompacted one."""
    config = _config()
    _emr(config)["neuromodulation"] = {"enabled": True, "mode": mode, "gate_scaling": "sigmoid",
                                       "context_dim": 4, "context_influence": 0.5}
    state, compacted, seed = _compacting(config)
    algo_u, _ = _initialized(config, seed)
    whole = _generation(algo_u, state, uncompacted=True)
    _assert_same_evaluation(compacted.outputs, compacted.fitness, whole)
    assert compacted.completion is not None


def test_the_comparison_refuses_the_compacted_axis_alone():
    """The comparisons have the power to see an evaluation on the compacted axis alone: the
    compacted generation's forward built without the missing positions (softmax over the union
    alone) is refused against the uncompacted one, by 1e-4 or more."""
    compacted, whole = _compacted_and_not("plain_emr")
    built = {}
    build = engine.EMRHyperNEAT._build_forward

    def without_completion(self, *args, **kwargs):
        # An engine without the completion has none to drop
        if "softmax_completion" in inspect.signature(build).parameters:
            kwargs = dict(kwargs, softmax_completion=None)
        built["forward"] = build(self, *args, **kwargs)
        return built["forward"]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine.EMRHyperNEAT, "_build_forward", without_completion)
        algo_old, _ = _initialized(RULES["plain_emr"](), compacted.algo._run_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            algo_old.run_generation(compacted.state, XORProblem())
    forward_one, genome_args, _ = built["forward"]
    inputs = algo_old._cached_inputs
    outputs = np.asarray(jax.vmap(lambda *args: forward_one(inputs, *args))(*genome_args))
    fitness = np.asarray(algo_old._cached_fitnesses)
    assert float(np.nanmax(np.abs(fitness - whole.fitness))) > 1e-4
    with pytest.raises(AssertionError):
        _assert_same_evaluation(outputs, fitness, whole)


# evaluate_with_neurotransmitter

CALL_NT = (0.9, 0.1, 0.7, 0.6)


@pytest.mark.parametrize("name, branch", [("paper_emr", "none"), ("paper_emr", "spatial"),
                                          ("paper_emr_option_b", "hierarchical"),
                                          ("steep_gate_drive_once_initial_only", "none")])
def test_the_call_under_softmax_after_a_compacted_generation(name, branch):
    """evaluate_with_neurotransmitter('softmax') after a compacted generation gives what it gives
    after the same population's uncompacted generation: each missing position with its receptors,
    base gain and, under branch gating, its region."""
    compacted, whole = _compacted_and_not(name)
    nt = jnp.asarray(CALL_NT, dtype=jnp.float32)

    def call(generation):
        algo = generation.algo
        return np.asarray(algo.evaluate_with_neurotransmitter(
            nt, algo._cached_inputs, algo._cached_targets, activation_name="softmax",
            branch_gating_mode=branch))

    after_compaction, after_whole = call(compacted), call(whole)
    assert compacted.algo._cached_active_indices is not None
    np.testing.assert_array_equal(np.isfinite(after_compaction), np.isfinite(after_whole))
    if exact_backend():
        assert_close_across_programs(after_compaction, after_whole)
    else:
        np.testing.assert_allclose(after_compaction, after_whole, rtol=0, atol=GPU_ATOL)


# The substrate API against the loop

#: The API's cases: without neuromodulation under both rules, and under true neuromodulation with
#: Option A from the first input weight (the receptor source the API evaluates) for each node rule
API_CASES: Dict[str, Callable[[], Dict[str, Any]]] = {
    **{name: RULES[name] for name in ("plain_emr", "plain_drive_once_every_iteration",
                                      "plain_drive_once_initial_only", "paper_emr",
                                      "steep_gate_drive_once_initial_only")},
    "anti_emr": SOURCES["anti_emr"],
    "paper_emr_gain_bias_only": SOURCES["paper_emr_gain_bias_only"],
    "paper_drive_once_initial_only": lambda: _config({}, settling="drive_once",
                                                     hh_mode="initial_only"),
}


@pytest.mark.parametrize("name", sorted(API_CASES))
def test_the_api_equals_the_loop(name):
    """substrate_forward evaluates each genome on its own positions and the loop on the compacted
    union; both count every grid position, so the outputs and fitness agree (two programs)."""
    state, loop, seed = _compacting(API_CASES[name]())
    algo = loop.algo
    substrate = _initialized(API_CASES[name](), seed)[0].population_substrates(state)
    config = substrate.config
    neuromodulation = (None if config.neuromodulation is None
                       else api.substrate_neuromodulation(substrate))
    outputs = np.asarray(substrate_forward(substrate, algo._cached_inputs,
                                           neuromodulation=neuromodulation))
    fitness = np.asarray(jax.vmap(lambda out: engine._eval_single_fitness(
        out, algo._cached_targets, algo.fitness_mode))(outputs))
    _assert_same_evaluation(outputs, _sanitized(fitness), loop)
    assert loop.completion is not None
    # The states come from the same forward, over each genome's own positions
    _, states = substrate_forward(substrate, algo._cached_inputs, return_states=True,
                                  neuromodulation=neuromodulation)
    masks = np.asarray(substrate.mask)
    for genome, settled in enumerate(states):
        assert settled.shape == (algo.activate_time, algo._cached_inputs.shape[0],
                                 int(masks[genome].sum()))


def test_the_api_counts_the_grid_for_a_genome_alone():
    """A genome evaluated alone (a batch of one) is its row of the population: its missing positions
    are its own, whatever the other genomes hold."""
    algo, state = _initialized(RULES["paper_emr"]())
    substrate = algo.population_substrates(state)
    nm = api.substrate_neuromodulation(substrate)
    whole = np.asarray(substrate_forward(substrate, algo._cached_inputs, neuromodulation=nm))
    for genome in (0, POP - 1):
        def one(array):
            return None if array is None else array[genome:genome + 1]
        alone = substrate._replace(**{field: one(getattr(substrate, field)) for field in (
            "mask", "W1", "W2", "act_indices", "agg_indices", "hh_agg_indices", "hh_from", "hh_to",
            "hh_weights", "hh_valid", "hh_class", "hh_direct", "hh_uncapped")})
        nm_alone = nm._replace(receptors=nm.receptors[genome:genome + 1],
                               base_gains=nm.base_gains[genome:genome + 1])
        out = np.asarray(substrate_forward(alone, algo._cached_inputs, neuromodulation=nm_alone))
        # One program on the same operands: bit for bit on the CPU
        if exact_backend():
            np.testing.assert_array_equal(out[0], whole[genome])
        else:
            np.testing.assert_allclose(out[0], whole[genome], rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


# Uncompacted controls

def _registry_softmax_fitness(generation: Generation) -> np.ndarray:
    """The generation's fitness from the forward without the completion, written out: the input
    drive, the settling by the configured rule with ACTIVATION_FUNCTIONS['softmax'] (or the dense
    pass), the sigmoid output, the fitness. For the controls' configurations: no neuromodulation,
    no aggregation."""
    algo = generation.algo
    softmax = engine.ACTIVATION_FUNCTIONS["softmax"]
    hh = algo._cached_eval_sparse_hh
    recurrent = hh is not None and algo._forward_mode == engine.ForwardPassMode.HYBRID_SPARSE_HH
    steps, drive_once = algo.activate_time, algo._settling_rule == "drive_once"
    node = jnp.tanh if algo.hh_activation_mode == "tanh" else softmax

    def forward(W1, W2, *connections):
        drive = algo._cached_inputs @ W1
        if not recurrent:
            hidden = softmax(drive)
        elif drive_once:
            hidden = engine.settle_hidden_drive_once(drive, *connections, steps, node, node)
        else:
            hidden = engine.settle_hidden(drive, *connections, steps, node)
        return engine._eval_single_fitness(jax.nn.sigmoid(hidden @ W2), algo._cached_targets,
                                           algo.fitness_mode)

    arrays = (algo._cached_eval_W1, algo._cached_eval_W2)
    if recurrent:
        arrays += (hh.from_indices, hh.to_indices, hh.weights, hh.valid_mask)
    return _sanitized(jax.vmap(forward)(*arrays))


#: The seeds the whole-grid control runs, in order, until a generation's union is the whole grid
UNION_SEEDS = tuple(range(12))


def _whole_grid_generation() -> Generation:
    """A recurrent softmax generation whose union is the whole grid, so it did not compact: the
    first of UNION_SEEDS that gives one (which seed moves with the PRNG default, so none is
    assumed)."""
    config = _config(preset="full_recurrent", pop=50)
    for seed in UNION_SEEDS:
        algo, state = _initialized(config, seed)
        generation = _generation(algo, state)
        if bool(np.asarray(algo._cached_masks_A, dtype=bool).any(axis=0).all()):
            return generation
    pytest.fail(f"no seed of {UNION_SEEDS} gives a generation whose union is the whole grid")


@pytest.mark.parametrize("control", ["feedforward", "whole_grid", "hh_tanh"])
def test_uncompacted_controls_evaluate_as_before(control):
    """A generation that does not compact, or whose settling never applies softmax, builds the
    forward without the completion (no extra argument) and gives the registry softmax's fitness,
    written out and run eagerly: two programs, bit for bit on the CPU under single-threaded Eigen
    (measured so on an Apple M4 Max), as two programs under multi-threaded Eigen."""
    if control == "feedforward":
        algo, state = _initialized(_config(preset=None))
        generation = _generation(algo, state)
    elif control == "whole_grid":
        generation = _whole_grid_generation()
    else:
        algo, state = _initialized(_config(hh_mode="tanh"))
        generation = _generation(algo, state)
    assert generation.completion is None
    assert all(np.ndim(arg) == 0 or np.shape(arg)[-1] != generation.genome_args[0].shape[-1] + 1
               for arg in generation.genome_args if arg is not None)
    expected = _registry_softmax_fitness(generation)
    if exact_backend() and not cpu_multi_threaded_eigen():
        np.testing.assert_array_equal(generation.fitness, expected)
    elif exact_backend():
        assert_close_across_programs(generation.fitness, expected)
    else:
        np.testing.assert_allclose(generation.fitness, expected, rtol=GPU_TRAJECTORY_RTOL,
                                   atol=1e-6)


def test_the_completion_reads_the_missing_positions_receptors():
    """Under the self-connection query and Option B each missing position derives its own receptors
    from the ceiling-sized query, one entry per position; Option A from the first input weight
    derives the zero weight's for all of them, one entry standing for every missing position."""
    for name, per_position in (("paper_emr", False), ("paper_emr_self_query", True),
                               ("paper_emr_option_b", True)):
        compacted, _ = _compacted_and_not(name)
        completion = compacted.completion
        assert completion is not None, "the compacted generation counts the missing positions"
        algo = compacted.algo
        ceiling = algo._cached_W1.shape[-1]
        assert completion.missing == ceiling - compacted.union
        entries = completion.multiplicity.shape[0]
        if per_position:
            assert entries == completion.missing
            np.testing.assert_array_equal(np.asarray(completion.multiplicity), 1.0)
        else:
            assert entries == 1 and float(completion.multiplicity[0]) == completion.missing
            # The receptors of a zero weight, as the generation derives them
            zero = jnp.zeros((POP, algo._cached_W1.shape[1], 1), algo._cached_W1.dtype)
            receptors, gains = algo._true_neuromodulation_receptors(zero, None)
            np.testing.assert_array_equal(np.asarray(completion.receptors), np.asarray(receptors))
            np.testing.assert_array_equal(np.asarray(completion.base_gains), np.asarray(gains))
