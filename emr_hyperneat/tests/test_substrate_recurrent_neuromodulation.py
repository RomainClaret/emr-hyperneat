"""True neuromodulation of a recurrent substrate in the substrate API, against the generation loop.

The loop evaluates true neuromodulation with hidden-to-hidden connections in `_build_forward` (mode
'true_neuromodulation'): the neuromodulated node rule (`true_neuromodulation_node_rule` over the
settling activation) at every step of `settle_hidden` under settling_rule 'emr', the additive rule
of the prototype's FULL variant (`settle_hidden_additive`) under 'drive_once' (the node of the
input drive, then activate_time update steps), then the output inversion. `substrate_forward`
composes the same functions on each genome's own active positions, the receptors and base gains
gathered there with W1 and W2, so its outputs and fitness match the loop's within float32 rounding
(the loop evaluates the population on the union of its active positions), within the bounds
measured below. The reference is the loop's own evaluation of the same genomes: its forward and
arguments as `_build_forward` returns them (the spy of test_recurrent_rule.py), and its recorded
fitness.

Also covered: the receptors against the loop's, the settling states in every case (the outputs
with them equal those without, bit for bit on the CPU), a caller-built SubstrateNeuromodulation
(with and without the configuration's settings), the inversion with a neurotransmitter vector
whose fourth channel inverts, an edited mask, an empty genome, a batch of one, what the loop
refuses and what it accepts (built and settled), a configuration that bypassed validate (evaluated
as the validated one when valid, refused at forward time with validate's message when not), the
compile key, evaluate_genome_on_problem, and conftest's comparison helpers.
"""

import copy
import functools
import inspect
import re
import warnings
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    GPU_TRAJECTORY_RTOL,
    XORProblem,
    assert_close_across_programs,
    cpu_multi_threaded_eigen,
    create_config_with_neuromodulation,
    exact_backend,
)
from emr_hyperneat import (
    EMRHyperNEAT,
    SubstrateConfig,
    build_recurrent_substrate,
    query_recurrent_substrate,
    substrate_forward,
)
from emr_hyperneat import emrhyperneat as engine
from emr_hyperneat import substrate as api

DEPTH = 3
POP = 20
SEED = 42
HH_FIELDS = ("hh_from", "hh_to", "hh_weights", "hh_valid", "hh_class", "hh_direct", "hh_uncapped")

# The API against the loop on the CPU (an Apple M4 Max, jax 0.6.1,
# XLA_FLAGS=--xla_cpu_multi_thread_eigen=false), over the 22 cases below (seed 42, 20 genomes
# each, XOR's four rows), by node rule and settling rule: the largest output difference; the
# largest fitness one.
#   paper      'emr',        9 cases: 3.8e-6, one ulp of a relu output of 42.1 (3.0e-7 among
#                                     bounded outputs); 6.0e-8
#   paper      'drive_once', 3 cases: 9.5e-7, 16 ulps of an output of 0.56; 3.0e-7
#   steep_gate 'emr',        3 cases: 1.2e-7; 6.0e-8
#   steep_gate 'drive_once', 4 cases: 6.0e-8; 0
#   anti       'emr',        2 cases: 1.8e-7; 6.0e-8
#   anti       'drive_once', 1 case:  6.0e-8; 0
# The two are separate programs (the loop's eager vmap over the union of the population's
# positions, the API's compiled forward on each genome's own), so the outputs are compared as two
# programs are (conftest's CPU_CROSS_PROGRAM_ULPS: 128 ulps of their largest finite magnitude,
# 1.5e-5 for outputs up to 1, 128 times the relu case's difference and 16 times the 'drive_once'
# one), and the fitness within 1e-6, the recurrent API's fitness bound
# (test_substrate_recurrent.py), 3.3 times the largest seen. The defects these comparisons guard
# move outputs by 1e-2 or more (test_the_comparison_sees_a_wrong_forward: neuromodulation dropped,
# the other settling rule, the inversion ignored).
# These numbers are measured on these populations under single-threaded Eigen, not a bound of the
# forward: a genome whose settling is chaotic carries the two programs' rounding to any size (on a
# 32 x 32 grid of (x, y, 1) rows one genome's outputs are 0.91 from the loop's eager evaluation, as
# far as the loop's own compiled evaluation is from it; an Apple M4 Max, single-threaded Eigen),
# and another seed, or XLA's default multi-threaded Eigen (on a Xeon E5-2683 v4, an AVX2 CPU, two
# programs 1 to 20 ulps apart from the start), may hold one. So on the CPU, when a genome is past
# them, the case is compared as two programs that round apart are (ROUNDED_APART_ATOL). Under
# single-threaded Eigen no genome of these populations is past them, on an Apple M4 Max and on a
# Xeon E5-2683 v4. Under XLA's default multi-threaded Eigen two cases are on the M4 Max:
# global_sin up to 0.14 in outputs and 0.041 in fitness, and full18_uniform up to 1.0 and 0.25,
# compared on their 8 and 16 genomes steady at one ulp (the loop's outputs are the same under both
# settings, the API's compiled forward moves); and one case is on the Xeon: global_sin put 12 of
# its 20 genomes more than 1e-7 apart, up to 0.94 in outputs and 0.105 in fitness; the other 21
# cases stayed within the bounds measured on the M4 Max.
CPU_FITNESS_ATOL = 1e-6
#: The fewest genomes a comparison may keep once the sensitive ones are left out: a quarter of the
#: population, so a case still compares several genomes against the loop.
MIN_STEADY = POP // 4
#: A genome whose fitness moves by more than this under a one-ulp perturbation of its weights
#: settles chaotically, and its loop and API fitness may differ by anything.
ULP_SENSITIVE = 1e-4
# Off the CPU the two layouts round differently and a settling can amplify that
# (test_substrate_recurrent.py): the fitness of every genome that is not sensitive to one ulp,
# within GPU_FORWARD_ATOL.
GPU_FORWARD_ATOL = 1e-3
# A CPU whose two programs round apart (XLA's default multi-threaded Eigen: a Xeon E5-2683 v4, and
# an Apple M4 Max on two cases) is compared as a GPU is once the measured bounds fail: the genomes
# that are not sensitive to one ulp, at least MIN_STEADY of them, their outputs and fitness within
# ROUNDED_APART_ATOL. On the Xeon under XLA's defaults global_sin's 8 steady genomes are up to
# 3.2e-5 apart in outputs and 1.3e-5 in fitness, 30 and 75 times under it, and the 12 others are
# sensitive; under single-threaded Eigen all 20 are within 6.0e-8. Each wrong forward of
# test_the_comparison_sees_a_wrong_forward moves outputs by 1e-2 or more, and the whole comparison
# refuses it (the same test).
ROUNDED_APART_ATOL = GPU_FORWARD_ATOL
# On a GPU a scatter-add accumulates in no fixed order, and the 'drive_once' rule's recurrent input
# is one at every step (_drive_once_recurrent_delta), so there one program does not repeat bit for
# bit and a settling amplifies the difference. On an NVIDIA T4 (4 of these tests failed of 146) and
# an NVIDIA L4 (7 of 146; jax 0.6.1, JAX_DEFAULT_MATMUL_PRECISION=highest) every failure was a
# steep_gate_drive_once case: a repeat's outputs up to 1.74e-4 apart (L4) and its states up to
# 1.5e-8 (both), the states of two settling lengths up to 2.9e-6 (L4), the last state through W2
# against the outputs up to 3.2e-5 (T4), and a configuration built without validate against the
# validated one up to 6.5e-5 (T4). Under 'drive_once' on a GPU those comparisons allow
# GPU_SCATTER_ATOL, 5.7 times the largest and ten times below the 1e-2 a wrong forward moves outputs
# by (test_the_comparison_sees_a_wrong_forward). A settling carries that rounding further in a
# state than in the outputs (under steep_gate with the inversion an NVIDIA T4 put one steady
# genome's states 5.9e-3 apart on a repeat in one run and 2.3e-3 in another, its outputs within
# GPU_SCATTER_ATOL; one ulp of its weights moves its fitness by 6.0e-7 on an Apple M4 Max), so a
# repeat on a GPU under 'drive_once' compares the steady genomes' outputs and the states' shapes.
# The CPU branches of these comparisons do not read GPU_SCATTER_ATOL.
GPU_SCATTER_ATOL = 1e-3
#: A neurotransmitter vector whose fourth channel inverts the outputs (the loop's single-task
#: default, 1.0, does not)
INVERTING = (0.9, 0.1, 0.5, 0.3)


def _emr(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]


def _config(rule: str = "paper", settling: str = "emr", preset: str = "full_recurrent",
            level: int = 1, activations: Optional[Dict[str, Any]] = None,
            aggregation: Optional[Dict[str, Any]] = None, hh_mode: Optional[str] = None,
            substrate: Optional[Dict[str, Any]] = None,
            recurrence: Optional[Dict[str, Any]] = None, **neuromodulation: Any) -> Dict[str, Any]:
    """True neuromodulation on a recurrent substrate: full_recurrent at iteration level 1 (every
    connection class) by default, the h->h cache off, one setting at a time on top."""
    config = create_config_with_neuromodulation("true_neuromodulation_4nt",
                                                recurrence_preset=preset, max_depth=DEPTH,
                                                population_size=POP, node_rule=rule,
                                                **neuromodulation)
    emr = _emr(config)
    emr["recurrence"].update(iteration_level=level, hh_cache_enabled=False, settling_rule=settling)
    emr["recurrence"].update(recurrence or {})
    if activations is not None:
        emr["dynamic_functions"] = dict(activations)
        if activations.get("mode") == "global":
            config["algorithm_params"]["emrhyperneat"]["substrate"]["hidden_activation"] = (
                activations["hidden_activation"])
    if aggregation is not None:
        emr["aggregation"] = dict(aggregation)
    if hh_mode is not None:
        emr["dynamic_functions"]["hh_activation_mode"] = hh_mode
    if substrate is not None:
        config["algorithm_params"]["emrhyperneat"]["substrate"].update(substrate)
    return config


PER_NODE = {"mode": "cppn_output", "num_activations": 4}
PER_NODE_6 = {"mode": "cppn_output", "num_activations": 6}
FULL18_UNIFORM = {"mode": "cppn_output", "num_activations": 18, "palette": "full",
                  "index_decode": "uniform"}
GLOBAL_SIN = {"mode": "global", "hidden_activation": "sin", "output_activation": "sigmoid"}
GLOBAL_SOFTMAX = {"mode": "global", "hidden_activation": "softmax", "output_activation": "sigmoid"}
HH_DYNAMIC = {"mode": "cppn_output", "num_aggregations": 4, "use_true_aggregation": True,
              "hh_aggregation_mode": "dynamic"}

#: Each case: the configuration and the neurotransmitter vector the loop and the API both read
#: (None: the loop's single-task default). Node rule x settling rule first, then what else each
#: rule accepts.
CASES = {
    "paper_emr": (lambda: _config("paper", "emr"), None),
    "steep_gate_emr": (lambda: _config("steep_gate", "emr"), None),
    "anti_emr": (lambda: _config("anti", "emr", use_output_inversion=False), None),
    "paper_drive_once": (lambda: _config("paper", "drive_once", hh_mode="initial_only"), None),
    "steep_gate_drive_once": (lambda: _config("steep_gate", "drive_once", hh_mode="initial_only"),
                              None),
    "anti_drive_once": (lambda: _config("anti", "drive_once", hh_mode="initial_only",
                                        use_output_inversion=False), None),
    # 'drive_once' with neuromodulation: 'every_iteration' with plain tanh, 'tanh', and per-node
    # functions at the first step
    "paper_drive_once_every_iteration": (lambda: _config("paper", "drive_once"), None),
    "paper_drive_once_tanh_per_node": (lambda: _config("paper", "drive_once",
                                                       activations=PER_NODE, hh_mode="tanh"),
                                       None),
    "steep_gate_drive_once_per_node": (lambda: _config("steep_gate", "drive_once",
                                                       activations=PER_NODE,
                                                       hh_mode="initial_only"), None),
    # 'emr' with what a caller that draws pictures uses: per-node functions (a palette, the uniform
    # decode), a global activation, dynamic h->h aggregation, iteration level 2, the other presets,
    # modes, channel counts and a strength
    "per_node_L2": (lambda: _config("paper", "emr", level=2, activations=PER_NODE), None),
    "full18_uniform": (lambda: _config("steep_gate", "emr", activations=FULL18_UNIFORM), None),
    "global_sin": (lambda: _config("paper", "emr", activations=GLOBAL_SIN), None),
    "hh_dynamic": (lambda: _config("paper", "emr", activations=PER_NODE_6,
                                   aggregation=HH_DYNAMIC), None),
    "hidden_only_L2_tanh": (lambda: _config("paper", "emr", preset="hidden_only", level=2,
                                            activations=PER_NODE, hh_mode="tanh"), None),
    "with_lateral_L3": (lambda: _config("anti", "emr", preset="with_lateral", level=3,
                                        use_output_inversion=False), None),
    "gating_only": (lambda: _config("steep_gate", "emr", modulation_mode="gating_only"), None),
    "gain_bias_only_nt_6_strength_5": (lambda: _config("paper", "emr",
                                                       modulation_mode="gain_bias_only",
                                                       num_nt_types=6, modulation_strength=5.0),
                                       None),
    "output_relu_activate_time_5": (lambda: _config("paper", "emr",
                                                    substrate={"output_activation": "relu"},
                                                    recurrence={"activate_time": 5}), None),
    # The inversion: a fourth channel below 1, with the inversion on and off, under both rules
    "paper_emr_inverted": (lambda: _config("paper", "emr"), INVERTING),
    "paper_emr_not_inverted": (lambda: _config("paper", "emr", use_output_inversion=False),
                               INVERTING),
    "steep_gate_drive_once_inverted": (lambda: _config("steep_gate", "drive_once",
                                                       hh_mode="initial_only"), INVERTING),
    "steep_gate_drive_once_not_inverted": (lambda: _config("steep_gate", "drive_once",
                                                           hh_mode="initial_only",
                                                           use_output_inversion=False),
                                           INVERTING),
}


class Run(NamedTuple):
    """A case's population as the API builds it, and as the loop evaluated it in one generation."""

    algo: Any
    state: Any
    substrate: Any
    inputs: Any
    neuromodulation: Any
    outputs: np.ndarray
    fitness: np.ndarray
    loop_outputs: np.ndarray
    loop_fitness: np.ndarray


def _fitness(algo: Any, outputs: Any) -> np.ndarray:
    def one(out):
        return engine._eval_single_fitness(out, algo._cached_targets, algo.fitness_mode)
    return np.asarray(jax.vmap(one)(outputs))


def _setup(config: Dict[str, Any]) -> Tuple[Any, Any]:
    algo = EMRHyperNEAT()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        state = algo.initialize(algo.create_config(copy.deepcopy(config)), XORProblem(), seed=SEED)
    return algo, state


def _loop_run(config: Dict[str, Any], neurotransmitter: Optional[Tuple[float, ...]]
              ) -> Tuple[Any, Any, Any, Any, np.ndarray]:
    """The loop's generation on ``config``, its forward and arguments kept; with
    ``neurotransmitter``, the loop's evaluation reads that vector in place of its single-task
    default."""
    algo, state = _setup(config)
    substrate = algo.population_substrates(state)
    build = engine.EMRHyperNEAT._build_forward
    kept = {}

    def spy(self, W1, W2, sparse_hh, neuromod_data, *args, **kwargs):
        if neurotransmitter is not None:
            neuromod_data = dict(neuromod_data,
                                 neurotransmitter=jnp.asarray(neurotransmitter, dtype=jnp.float32))
        kept["built"] = build(self, W1, W2, sparse_hh, neuromod_data, *args, **kwargs)
        return kept["built"]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine.EMRHyperNEAT, "_build_forward", spy)
        algo.run_generation(state, XORProblem())
    forward_one, genome_args, _ = kept["built"]
    inputs = algo._cached_inputs
    # The loop's evaluation is one vmap of this forward over the population
    # (evaluate_population_on_device)
    loop_outputs = np.asarray(jax.vmap(lambda *args: forward_one(inputs, *args))(*genome_args))
    return algo, state, substrate, inputs, loop_outputs


@functools.lru_cache(maxsize=None)
def _run(name: str) -> Run:
    """One generation of a case, shared by the tests that read it (the arrays are not modified)."""
    make, neurotransmitter = CASES[name]
    algo, state, substrate, inputs, loop_outputs = _loop_run(make(), neurotransmitter)
    neuromodulation = api.substrate_neuromodulation(substrate, neurotransmitter=neurotransmitter)
    outputs = np.asarray(substrate_forward(substrate, inputs, neuromodulation=neuromodulation))
    return Run(algo, state, substrate, inputs, neuromodulation, outputs, _fitness(algo, outputs),
               loop_outputs, np.asarray(algo._cached_fitnesses))


def _ulp_response(run: Run, draws: int = 3) -> np.ndarray:
    """Per genome, the largest fitness change under a one-ulp relative perturbation of W1, W2 and
    the h->h weights."""
    rng = np.random.default_rng(0)
    response = np.zeros_like(run.fitness)

    def jitter(array):
        array = np.asarray(array)
        signs = rng.choice([-1.0, 1.0], size=array.shape)
        return jnp.asarray((array * (1 + 2.0 ** -23 * signs)).astype(np.float32))

    for _ in range(draws):
        perturbed = run.substrate._replace(W1=jitter(run.substrate.W1),
                                           W2=jitter(run.substrate.W2),
                                           hh_weights=jitter(run.substrate.hh_weights))
        outputs = substrate_forward(perturbed, run.inputs, neuromodulation=run.neuromodulation)
        response = np.maximum(response, np.abs(_fitness(run.algo, outputs) - run.fitness))
    return response


def _assert_across_programs(actual: Any, desired: Any, drive_once: bool = False) -> None:
    """Two compiled programs of one forward: CPU_CROSS_PROGRAM_ULPS on the CPU (conftest),
    GPU_TRAJECTORY_RTOL off it, and GPU_SCATTER_ATOL there under the 'drive_once' rule
    (``drive_once``)."""
    if exact_backend():
        assert_close_across_programs(actual, desired)
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(desired),
                                   rtol=GPU_TRAJECTORY_RTOL,
                                   atol=GPU_SCATTER_ATOL if drive_once else 1e-6)


def _assert_same_program(actual: Any, desired: Any, drive_once: bool = False) -> None:
    """One compiled program on the same operands: bit for bit on the CPU, to GPU_TRAJECTORY_RTOL
    off it, and GPU_SCATTER_ATOL there under the 'drive_once' rule (``drive_once``), whose
    scatter-add a GPU accumulates in no fixed order."""
    if exact_backend():
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(desired))
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(desired),
                                   rtol=GPU_TRAJECTORY_RTOL,
                                   atol=GPU_SCATTER_ATOL if drive_once else 1e-6)


def _assert_repeat(actual: Any, desired: Any, drive_once: bool) -> None:
    """One compiled program repeated on the same operands: bit for bit on the CPU, and on a GPU
    under the 'emr' rule (an NVIDIA T4 and an NVIDIA L4 repeated rule 'paper' so; an RTX 2080 Ti
    the forward without neuromodulation, test_substrate_recurrent.py); under 'drive_once' on a GPU
    as `_assert_same_program` compares (GPU_SCATTER_ATOL)."""
    if exact_backend() or not drive_once:
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(desired))
    else:
        _assert_same_program(actual, desired, drive_once=True)


def _genome(substrate: Any, genome: int) -> Any:
    """One genome's substrate, as a batch of one, without its h->h connections."""
    def one(array):
        return None if array is None else array[genome:genome + 1]
    return substrate._replace(
        mask=one(substrate.mask), W1=one(substrate.W1), W2=one(substrate.W2),
        act_indices=one(substrate.act_indices), agg_indices=one(substrate.agg_indices),
        hh_agg_indices=one(substrate.hh_agg_indices),
        **{field: None for field in HH_FIELDS},
    )


# Against the loop

def _assert_matches_the_loop_on_the_cpu(run: Run, fitness: np.ndarray, finite: np.ndarray,
                                        genomes: np.ndarray) -> None:
    """The CPU comparison over ``genomes``: the finite outputs as two programs are compared, the
    fitness within CPU_FITNESS_ATOL."""
    kept = finite[genomes]
    assert_close_across_programs(run.outputs[genomes][kept], run.loop_outputs[genomes][kept])
    np.testing.assert_allclose(fitness[genomes], run.loop_fitness[genomes], rtol=0,
                               atol=CPU_FITNESS_ATOL)


def _check_against_the_loop(run: Run, sensitive: Optional[np.ndarray] = None) -> None:
    """Every genome's outputs and fitness against the loop's: within the measured bounds on the
    CPU. Where a genome is past them (a CPU whose two programs round apart), and off the CPU, the
    comparison of two programs that round apart: the genomes that one ulp of their weights moves by
    at most ULP_SENSITIVE in fitness, at least MIN_STEADY of them, within ROUNDED_APART_ATOL, their
    outputs (on the CPU) and their fitness.

    Args:
        run: The case, its API outputs and fitness against the loop's.
        sensitive: The genomes sensitive to one ulp, when the caller has them (a run whose outputs
            it replaced keeps the sensitivity of its substrate); None measures them.
    """
    finite = ~np.isnan(run.loop_outputs)
    assert finite.reshape(POP, -1).all(axis=1).mean() >= 0.75, (
        "most genomes settle to finite outputs")
    np.testing.assert_array_equal(np.isnan(run.outputs), ~finite)
    fitness = np.asarray(engine.sanitize_fitness(jnp.asarray(run.fitness))[0])
    if exact_backend():
        try:
            _assert_matches_the_loop_on_the_cpu(run, fitness, finite, np.ones(POP, dtype=bool))
            return
        except AssertionError:
            # The bounds are measured, not a bound of the forward (the comment on CPU_FITNESS_ATOL)
            pass
    if sensitive is None:
        sensitive = _ulp_response(run) > ULP_SENSITIVE
    # A floor on the steady genomes, not a cap on the sensitive ones: the sensitivity is the
    # population's, so a chaotic case (global_sin) keeps its steady genomes compared
    steady = ~(sensitive | ~np.isfinite(fitness))
    assert int(steady.sum()) >= MIN_STEADY, (f"{int(steady.sum())} of {steady.size} genomes "
                                             f"steady at one ulp, fewer than {MIN_STEADY}")
    if exact_backend():
        kept = finite[steady]
        np.testing.assert_allclose(run.outputs[steady][kept], run.loop_outputs[steady][kept],
                                   rtol=0, atol=ROUNDED_APART_ATOL)
    np.testing.assert_allclose(fitness[steady], run.loop_fitness[steady], rtol=0,
                               atol=ROUNDED_APART_ATOL)


@pytest.mark.parametrize("name", sorted(CASES))
def test_outputs_and_fitness_match_the_loop(name):
    """Every genome's outputs and fitness are the loop's within the measured bounds (CPU), or, for
    the genomes that do not settle chaotically, within ROUNDED_APART_ATOL (a GPU, or a CPU whose two
    programs round apart: _check_against_the_loop). A genome
    whose settling overflows gives NaN in both, at the same outputs, and the loop records its
    fitness as -inf (sanitize_fitness), which the API's NaN fitness becomes the same way: in
    full18_uniform two of the 20 genomes overflow at a softplus node (the engine's softplus,
    log1p(exp(x)), is infinite once x passes about 88.7), under node rules 'paper' and 'steep_gate'
    alike, and three (those two and one more) without neuromodulation."""
    run = _run(name)
    config = run.substrate.config
    assert config.hidden_to_hidden and config.neuromodulation is not None
    assert int(np.asarray(run.substrate.hh_valid).sum()) > 0, "the case discovers connections"
    _check_against_the_loop(run)


def test_the_comparison_sees_a_wrong_forward():
    """The bounds leave the comparison its power: without neuromodulation, under the other settling
    rule, or with the inversion ignored, the API is 1e-2 or more from the loop, and the whole
    comparison refuses it, the one for two programs that round apart included (every genome kept
    steady, so that comparison is the one that judges)."""
    def wrong(run, substrate, neuromodulation=None):
        outputs = np.asarray(substrate_forward(substrate, run.inputs,
                                               neuromodulation=neuromodulation))
        assert float(np.max(np.abs(outputs - run.loop_outputs))) > 1e-2
        doctored = run._replace(outputs=outputs, fitness=_fitness(run.algo, outputs))
        with pytest.raises(AssertionError):
            _check_against_the_loop(doctored, sensitive=np.zeros(POP, dtype=bool))

    run = _run("paper_emr")
    wrong(run, run.substrate._replace(config=run.substrate.config._replace(neuromodulation=None)))
    run = _run("paper_drive_once")
    config = run.substrate.config._replace(settling_rule="emr",
                                           hh_activation_mode="every_iteration")
    wrong(run, run.substrate._replace(config=config), run.neuromodulation)
    run = _run("paper_emr_inverted")
    ignored = run.neuromodulation._replace(
        settings=run.neuromodulation.settings._replace(output_inversion=False))
    wrong(run, run.substrate, ignored)


@pytest.mark.parametrize("name", ["paper_emr", "gain_bias_only_nt_6_strength_5", "with_lateral_L3"])
def test_receptors_and_gains_are_the_loops_at_each_genomes_positions(name):
    """Derived at every grid position, each from its own first input weight, they equal the loop's
    on its compacted axis at the positions it evaluates."""
    run = _run(name)
    # The positions the loop evaluated, from its generation's record (None when it did not compact)
    active = run.algo._cached_active_indices
    receptors = np.asarray(run.algo._neuromod_true["receptor_densities"])
    gains = np.asarray(run.algo._neuromod_true["base_gains"])
    ours_receptors = np.asarray(run.neuromodulation.receptors)
    ours_gains = np.asarray(run.neuromodulation.base_gains)
    if active is not None:
        active = np.asarray(active)
        ours_receptors, ours_gains = ours_receptors[:, active], ours_gains[:, active]
        receptors, gains = receptors[:, :active.size], gains[:, :active.size]
    _assert_same_program(ours_receptors, receptors)
    _assert_same_program(ours_gains, gains)


# The settling states

def _steady(run: Run) -> np.ndarray:
    """The genomes that one ulp of their weights moves by at most ULP_SENSITIVE in fitness and whose
    fitness is finite, at least MIN_STEADY of them."""
    fitness = np.asarray(engine.sanitize_fitness(jnp.asarray(run.fitness))[0])
    steady = ~((_ulp_response(run) > ULP_SENSITIVE) | ~np.isfinite(fitness))
    assert int(steady.sum()) >= MIN_STEADY, (f"{int(steady.sum())} of {steady.size} genomes "
                                             f"steady at one ulp, fewer than {MIN_STEADY}")
    return steady


@pytest.mark.parametrize("name", sorted(CASES))
def test_states_end_on_the_settled_layer(name):
    """return_states gives activate_time states per genome over its own positions under both rules,
    and the outputs of the call without it: one compiled program gives both, so they are equal bit
    for bit on the CPU in every case (on a GPU every genome within GPU_TRAJECTORY_RTOL, the steady
    ones only where the settling sums with a scatter-add: dynamic h->h aggregation, and the
    'drive_once' rule's recurrent input at every step, under 'drive_once' within
    GPU_SCATTER_ATOL). The program that also emits the states
    rounds a settling step 1 to 2 ulps apart from it with the 18-function palette under steep_gate
    (full18_uniform), even under single-threaded Eigen, and chaotic genomes carry that to order 1
    (genome 11: fitness 0.750 without states, 0.504 from that program). The last state through W2,
    the output activation and the inversion gives the outputs for every genome that one ulp of its
    weights does not move (at least MIN_STEADY of them), within 8 ulps of the genome's outputs or
    1e-6 under single-threaded Eigen and as two programs that round apart are compared under
    multi-threaded Eigen; for a chaotic genome the two programs can differ."""
    run = _run(name)
    substrate, inputs, nm = run.substrate, run.inputs, run.neuromodulation
    outputs, states = substrate_forward(substrate, inputs, return_states=True, neuromodulation=nm)
    outputs = np.asarray(outputs)
    steps = substrate.config.activate_time
    assert steps == run.algo.activate_time
    # The last-state check below compares two programs, so it keeps to the steady genomes
    steady = _steady(run)
    if exact_backend() or (substrate.hh_agg_indices is None
                           and substrate.config.settling_rule != "drive_once"):
        # One program: bit for bit on the CPU; a GPU repeated the 'emr' forward without dynamic h->h
        # aggregation bit for bit (test_substrate_recurrent.py, the RTX 2080 Ti), so every genome
        _assert_same_program(outputs, run.outputs)
    else:
        # On a GPU a scatter-add accumulates in no fixed order (dynamic h->h aggregation, and the
        # 'drive_once' rule's recurrent input at every step, _drive_once_recurrent_delta; the
        # RTX 2080 Ti repeats covered the 'emr' rule only), so one program differs from itself in a
        # chaotic genome, and under 'drive_once' by up to 1.74e-4 in a steady one (an NVIDIA L4,
        # GPU_SCATTER_ATOL)
        drive_once = substrate.config.settling_rule == "drive_once"
        np.testing.assert_allclose(outputs[steady], run.outputs[steady], rtol=GPU_TRAJECTORY_RTOL,
                                   atol=GPU_SCATTER_ATOL if drive_once else 1e-6)
    output_fn = engine.ACTIVATION_FUNCTIONS[substrate.config.output_activation]
    settings = nm.settings
    assert len(states) == POP
    for genome, settled in enumerate(states):
        active = np.flatnonzero(np.asarray(substrate.mask[genome]))
        assert settled.shape == (steps, inputs.shape[0], active.size)
        if not steady[genome]:
            continue
        layer = output_fn(settled[-1] @ np.asarray(substrate.W2[genome])[active])
        last = engine.invert_outputs(layer, jnp.asarray(nm.neurotransmitter),
                                     settings.output_inversion, settings.num_nt_types,
                                     settings.node_rule)
        # The last state comes from the program that emits the states, the outputs from the one
        # without them: two programs, so the allowance scales with the outputs (a relu output of
        # 42.1 has an ulp of 3.8e-6), and under multi-threaded Eigen it is the one for two programs
        if exact_backend() and cpu_multi_threaded_eigen():
            assert_close_across_programs(np.asarray(last), outputs[genome])
        else:
            scale = float(np.max(np.abs(outputs[genome]))) if outputs[genome].size else 0.0
            ulp = float(np.spacing(np.float32(scale)))
            # On a GPU 1e-5, and GPU_SCATTER_ATOL under 'drive_once' (3.2e-5 on an NVIDIA T4)
            gpu = (GPU_SCATTER_ATOL if substrate.config.settling_rule == "drive_once" else 1e-5)
            np.testing.assert_allclose(np.asarray(last), outputs[genome], rtol=0,
                                       atol=max(1e-6 if exact_backend() else gpu, 8 * ulp))


@pytest.mark.parametrize("name", ["paper_emr", "steep_gate_drive_once"])
def test_each_state_is_the_layer_settled_that_many_steps(name):
    """State t is the layer settled for t + 1 steps, so under 'drive_once' they are the states
    after the update steps (the node of the input drive that starts the rule is left out) and under
    'emr' the states after each step."""
    run = _run(name)
    steps = 4
    substrate = run.substrate._replace(config=run.substrate.config._replace(activate_time=steps))
    _, states = substrate_forward(substrate, run.inputs, return_states=True,
                                  neuromodulation=run.neuromodulation)
    for count in range(1, steps + 1):
        shorter = substrate._replace(config=substrate.config._replace(activate_time=count))
        _, prefix = substrate_forward(shorter, run.inputs, return_states=True,
                                      neuromodulation=run.neuromodulation)
        for genome in range(POP):
            assert prefix[genome].shape[0] == count
            _assert_across_programs(prefix[genome], states[genome][:count],
                                    drive_once=substrate.config.settling_rule == "drive_once")
    # The first state is the first update's, not the node of the input drive: they differ wherever
    # a genome settles
    if substrate.config.settling_rule == "drive_once":
        settings = run.neuromodulation.settings
        moved = 0.0
        for genome in range(POP):
            active = np.flatnonzero(np.asarray(substrate.mask[genome]))
            if not active.size:
                continue
            receptors = jnp.asarray(np.asarray(run.neuromodulation.receptors)[genome][active])
            gains = jnp.asarray(np.asarray(run.neuromodulation.base_gains)[genome][active])
            node, operands = engine.true_neuromodulation_node_rule(
                (jnp.tanh, ()), receptors, gains, jnp.asarray(run.neuromodulation.neurotransmitter),
                settings.modulation_strength, settings.num_nt_types, settings.modulation_mode,
                settings.node_rule)
            drive = run.inputs @ jnp.asarray(np.asarray(substrate.W1[genome])[:, active])
            first = np.asarray(node(drive, *operands))
            moved = max(moved, float(np.max(np.abs(np.asarray(states[genome][0]) - first))))
        assert moved > 1e-2


# A given neuromodulation

def test_a_given_neuromodulation_replaces_the_configured_one():
    """The computed one gives the configured outputs (one program, bit for bit on the CPU), on a
    configuration with neuromodulation settings and on one without; other receptors give other
    outputs; wrong shapes and a short vector are refused as on a feed-forward substrate, and so are
    receptors without the channels the node rule reads, with their shape."""
    run = _run("paper_emr")
    substrate, inputs = run.substrate, run.inputs
    nm = api.substrate_neuromodulation(substrate)
    configured = np.asarray(substrate_forward(substrate, inputs))
    _assert_same_program(substrate_forward(substrate, inputs, neuromodulation=nm), configured)
    _assert_same_program(configured, run.outputs)
    flipped = np.asarray(substrate_forward(substrate, inputs,
                                           neuromodulation=nm._replace(receptors=-nm.receptors)))
    assert float(np.max(np.abs(flipped - configured))) > 1e-3
    plain = substrate._replace(config=substrate.config._replace(neuromodulation=None))
    _assert_same_program(substrate_forward(plain, inputs, neuromodulation=nm), configured)
    # Without one, the recurrent forward is the unmodulated one
    assert float(np.max(np.abs(np.asarray(substrate_forward(plain, inputs)) - configured))) > 1e-3
    with pytest.raises(ValueError, match="must cover the substrate"):
        substrate_forward(plain, inputs, neuromodulation=nm._replace(base_gains=nm.base_gains[:1]))
    short = nm._replace(neurotransmitter=jnp.asarray([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="reads 4 entries"):
        substrate_forward(substrate, inputs, neuromodulation=short)
    # The rule reads three receptor channels here ('paper', four channels) and every channel under
    # 'anti'; fewer, or none, are refused before the compiled forward, which would fail on them
    # with a JAX TypeError
    for receptors, shape in ((nm.receptors[..., :2], r"\(20, \d+, 2\) must be \(20, \d+, 3\)"),
                             (nm.receptors[..., 0], r"\(20, \d+\) must be \(20, \d+, 3\)")):
        with pytest.raises(ValueError, match=shape):
            substrate_forward(substrate, inputs, neuromodulation=nm._replace(receptors=receptors))
    anti = _run("anti_emr").neuromodulation
    with pytest.raises(ValueError, match=r"must be \(20, \d+, 4\) or have more channels: node rule "
                                         r"'anti' with 4 channels reads 4"):
        substrate_forward(_run("anti_emr").substrate, inputs,
                          neuromodulation=anti._replace(receptors=anti.receptors[..., :3]))


_COMPILES = {"active": False, "count": 0, "registered": False}


def _watch_compiles() -> None:
    """Count backend compilations while _COMPILES['active'] (the listener of
    test_recurrent_rule.py)."""
    if not _COMPILES["registered"]:
        def listener(event, duration, **kwargs):
            if _COMPILES["active"] and event == "/jax/core/compile/backend_compile_duration":
                _COMPILES["count"] += 1
        jax.monitoring.register_event_duration_secs_listener(listener)
        _COMPILES["registered"] = True


def test_the_vector_and_the_strength_compile_nothing_new():
    """The neurotransmitter vector and the strength are operands of the compiled forward: once a
    shape has run, other values compile nothing, and they change the outputs."""
    _watch_compiles()
    run = _run("paper_emr")
    substrate, inputs = run.substrate, run.inputs
    nm = api.substrate_neuromodulation(substrate)
    first = np.asarray(substrate_forward(substrate, inputs, neuromodulation=nm))
    other = nm._replace(neurotransmitter=jnp.asarray([0.2, 0.8, 0.3, 0.6], dtype=jnp.float32),
                        settings=nm.settings._replace(modulation_strength=3.5))
    _COMPILES["count"] = 0
    _COMPILES["active"] = True
    try:
        second = np.asarray(substrate_forward(substrate, inputs, neuromodulation=other))
    finally:
        _COMPILES["active"] = False
    assert _COMPILES["count"] == 0, f"{_COMPILES['count']} compiled for another vector and strength"
    assert float(np.max(np.abs(second - first))) > 1e-3


def test_the_compile_key():
    """The settings the node rule and the inversion read are in the key, the strength and the
    receptor derivation are not, and a call with neuromodulation None gets the forward that the
    call without it gets."""
    _recurrent_forward_fn = api._recurrent_forward_fn
    config = _run("paper_emr").substrate.config
    settings = config.neuromodulation
    plain = _recurrent_forward_fn(config, 16, False, False, False)
    assert _recurrent_forward_fn(config, 16, False, False, False, None) is plain
    modulated = _recurrent_forward_fn(config, 16, False, False, False, settings)
    assert modulated is not plain
    same_key = settings._replace(modulation_strength=5.0, receptor_derivation="abs")
    assert _recurrent_forward_fn(config, 16, False, False, False, same_key) is modulated
    for change in (dict(node_rule="steep_gate"), dict(modulation_mode="gating_only"),
                   dict(num_nt_types=3), dict(output_inversion=False)):
        other_key = settings._replace(**change)
        assert _recurrent_forward_fn(config, 16, False, False, False, other_key) is not modulated


# Edited masks, empty genomes, a batch of one

def test_a_cleared_mask_entry_drops_the_nodes_receptors_and_gain():
    """A knocked-out node loses its receptors and gain with its connections: NaN in them changes
    nothing, while NaN at a node it keeps reaches the outputs; the forward equals dropping the
    node's connections by hand."""
    run = _run("paper_emr")
    substrate, inputs = run.substrate, run.inputs
    valid = np.asarray(substrate.hh_valid)
    genome = int(np.argmax(valid.sum(axis=1)))
    sources = np.asarray(substrate.hh_from)[genome][valid[genome]]
    node = int(np.bincount(sources).argmax())  # the position with the most outgoing connections
    mask = np.asarray(substrate.mask).copy()
    mask[genome, node] = False
    knocked = substrate._replace(mask=jnp.asarray(mask))
    nm = api.substrate_neuromodulation(substrate)
    reference = np.asarray(substrate_forward(knocked, inputs, neuromodulation=nm))
    # Relative: this genome's outputs saturate near 0 (5e-19, against 1.6e-19 knocked out)
    assert not np.allclose(reference[genome], run.outputs[genome], rtol=1e-2, atol=0), (
        "the knockout must matter")

    def poisoned(position):
        receptors, gains = np.asarray(nm.receptors).copy(), np.asarray(nm.base_gains).copy()
        receptors[genome, position], gains[genome, position] = np.nan, np.nan
        return nm._replace(receptors=jnp.asarray(receptors), base_gains=jnp.asarray(gains))

    np.testing.assert_array_equal(
        np.asarray(substrate_forward(knocked, inputs, neuromodulation=poisoned(node))), reference)
    kept = [int(p) for p in np.flatnonzero(mask[genome])
            if np.any(np.asarray(substrate.W2[genome])[p] != 0)]
    assert kept, "the genome has an output-weighted node"
    poisoned_outputs = substrate_forward(knocked, inputs, neuromodulation=poisoned(kept[0]))
    assert np.isnan(np.asarray(poisoned_outputs)[genome]).any()
    # The knocked node's connections dropped by hand: the same program and operands
    touching = ((np.asarray(substrate.hh_from)[genome] == node)
                | (np.asarray(substrate.hh_to)[genome] == node))
    assert (touching & valid[genome]).any()
    dropped_valid = valid.copy()
    dropped_valid[genome] &= ~touching
    dropped = knocked._replace(hh_valid=jnp.asarray(dropped_valid))
    dropped_outputs = np.asarray(substrate_forward(dropped, inputs, neuromodulation=nm))
    np.testing.assert_array_equal(dropped_outputs[genome], reference[genome])
    # The configured receptors of the edited substrate are the same (W1 is unchanged)
    np.testing.assert_array_equal(np.asarray(substrate_forward(knocked, inputs)), reference)


EMPTY = (0, 7, POP - 1)


@pytest.mark.parametrize("output", ["sigmoid", "relu"])
@pytest.mark.parametrize("name", ["paper_emr_inverted", "steep_gate_drive_once_inverted"])
def test_a_genome_with_no_active_position(name, output):
    """A genome whose mask is cleared has no hidden node: settling states of width 0 and the outputs
    of a zero hidden layer, the output activation of zero through the inversion, as the
    feed-forward forward gives with its columns zeroed; the other genomes keep their outputs (one
    program, bit for bit on the CPU); alone, as a batch of one, the same. The inversion leaves
    sigmoid's 0.5 where it is; under relu it moves 0 to 1 - nt[3]."""
    run = _run(name)
    substrate, inputs, nm = run.substrate, run.inputs, run.neuromodulation
    substrate = substrate._replace(config=substrate.config._replace(output_activation=output))
    mask = np.asarray(substrate.mask).copy()
    mask[list(EMPTY)] = False
    edited = substrate._replace(mask=jnp.asarray(mask))
    forward, states = substrate_forward(edited, inputs, return_states=True, neuromodulation=nm)
    forward_before, states_before = substrate_forward(substrate, inputs, return_states=True,
                                                      neuromodulation=nm)
    forward_before = np.asarray(forward_before)
    config, settings = substrate.config, nm.settings
    output_fn = engine.ACTIVATION_FUNCTIONS[config.output_activation]
    zero_layer = output_fn(jnp.zeros((inputs.shape[0], len(config.output_coords))))
    expected = np.asarray(engine.invert_outputs(zero_layer, jnp.asarray(nm.neurotransmitter),
                                                settings.output_inversion, settings.num_nt_types,
                                                settings.node_rule))
    feed_forward = substrate._replace(
        config=config._replace(hidden_to_hidden=False), mask=jnp.asarray(mask),
        W1=jnp.where(mask[:, None, :], substrate.W1, 0.0),
        W2=jnp.where(mask[:, :, None], substrate.W2, 0.0),
        hh_agg_indices=None, **{field: None for field in HH_FIELDS})
    feed_forward_outputs = np.asarray(substrate_forward(feed_forward, inputs, neuromodulation=nm))
    # One program repeated: bit for bit, every genome, outputs and states, on the CPU. Under
    # 'drive_once' on a GPU, whose scatter-add accumulates in no fixed order, a chaotic genome's
    # outputs differ from themselves, so there the steady genomes' outputs, within GPU_SCATTER_ATOL
    # (_assert_repeat), as test_states_end_on_the_settled_layer compares; a settling state can
    # differ in a steady genome too (the T4's figures are at GPU_SCATTER_ATOL), so there the states
    # are compared by shape
    drive_once = config.settling_rule == "drive_once"
    gpu_drive_once = drive_once and not exact_backend()
    compared = _steady(run) if gpu_drive_once else np.ones(POP, dtype=bool)
    for genome in range(POP):
        if genome in EMPTY:
            np.testing.assert_array_equal(np.asarray(forward)[genome], expected)
            np.testing.assert_array_equal(np.asarray(forward)[genome], feed_forward_outputs[genome])
            assert states[genome].shape == (config.activate_time, inputs.shape[0], 0)
            continue
        assert states[genome].shape == states_before[genome].shape
        if not compared[genome]:
            continue
        _assert_repeat(np.asarray(forward)[genome], forward_before[genome], drive_once)
        if not gpu_drive_once:
            _assert_repeat(np.asarray(states[genome]), np.asarray(states_before[genome]),
                           drive_once)
    if output == "relu":
        np.testing.assert_allclose(expected, 1.0 - INVERTING[3], rtol=1e-6)
    lone = EMPTY[1]
    alone = _genome(edited, lone)._replace(**{field: getattr(edited, field)[lone:lone + 1]
                                              for field in HH_FIELDS
                                              if getattr(edited, field) is not None})
    one = nm._replace(receptors=nm.receptors[lone:lone + 1],
                      base_gains=nm.base_gains[lone:lone + 1])
    alone_forward, alone_states = substrate_forward(alone, inputs, return_states=True,
                                                    neuromodulation=one)
    np.testing.assert_array_equal(np.asarray(alone_forward)[0], expected)
    assert alone_states[0].shape == (config.activate_time, inputs.shape[0], 0)


@pytest.mark.parametrize("name", ["per_node_L2", "steep_gate_drive_once", "hh_dynamic"])
def test_one_genome_alone_is_its_row_in_the_population(name):
    """Built and run as a batch of one, a genome gives its population's connections and outputs
    (one program, bit for bit on the CPU), its receptors and gains derived from its own weights."""
    run = _run(name)
    population, inputs = run.substrate, run.inputs
    transformed = run.algo._compiled_transform_batch(run.state, run.algo._compiled_ask(run.state))
    outputs = query_recurrent_substrate(run.algo._jitted_cppn_forward, run.state, transformed,
                                        population)
    everyone = np.asarray(substrate_forward(population, inputs))
    for genome in (0, 7, POP - 1):
        alone = build_recurrent_substrate(_genome(population, genome), [outputs[genome]])
        for field in HH_FIELDS:
            np.testing.assert_array_equal(np.asarray(getattr(alone, field))[0],
                                          np.asarray(getattr(population, field))[genome],
                                          err_msg=field)
        _assert_same_program(np.asarray(substrate_forward(alone, inputs))[0], everyone[genome])


# What the loop refuses and accepts

def _recurrent_substrate_config(**fields: Any) -> SubstrateConfig:
    """A hand-built recurrent configuration under true neuromodulation (the engine's defaults), with
    ``fields``."""
    neuromodulation = fields.pop("neuromodulation", api.NeuromodulationSettings())
    return SubstrateConfig(input_coords=((-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)),
                           output_coords=((0.0, 1.0),), max_depth=2, hidden_to_hidden=True,
                           neuromodulation=neuromodulation, **fields)


_ANTI = api.NeuromodulationSettings(node_rule="anti", output_inversion=False)
_DRIVE_ONCE_INITIAL = dict(settling_rule="drive_once", hh_activation_mode="initial_only")

#: What the loop refuses with true neuromodulation of a recurrent substrate, each as the loop's
#: configuration (its create_config raises, with the loop's words) and as a SubstrateConfig
#: (validate raises, with the API's).
REFUSED = {
    "drive_once_hh_dynamic": (
        lambda: _config("paper", "drive_once", activations=PER_NODE, aggregation=HH_DYNAMIC,
                        hh_mode="initial_only"),
        "per-node hidden-to-hidden aggregation would be ignored",
        dict(_DRIVE_ONCE_INITIAL, activation_mode="cppn_output", aggregation_mode="cppn_output",
             num_cppn_outputs=3, hh_aggregation_mode="dynamic"),
        "hh_aggregation_mode 'dynamic'"),
    "drive_once_every_iteration_per_node": (
        lambda: _config("paper", "drive_once", activations=PER_NODE),
        "applies tanh at every step after the first",
        dict(settling_rule="drive_once", activation_mode="cppn_output", num_cppn_outputs=2),
        "applies tanh at every step after the first"),
    "drive_once_every_iteration_global_sin": (
        lambda: _config("paper", "drive_once", activations=GLOBAL_SIN),
        "applies tanh at every step after the first",
        dict(settling_rule="drive_once", activation_mode="global", hidden_activation="sin"),
        "applies tanh at every step after the first"),
    "anti_gating_only": (
        lambda: _config("anti", "emr", modulation_mode="gating_only", use_output_inversion=False),
        "is not one of its forms",
        dict(neuromodulation=_ANTI._replace(modulation_mode="gating_only")),
        "is not one of its forms"),
    "anti_inversion": (
        lambda: _config("anti", "emr", use_output_inversion=True),
        "inverts no output",
        dict(neuromodulation=_ANTI._replace(output_inversion=True)),
        "inverts no output"),
    "anti_5_channels": (
        lambda: _config("anti", "emr", num_nt_types=5, use_output_inversion=False),
        "at most 4 channels",
        dict(neuromodulation=_ANTI._replace(num_nt_types=5)),
        "at most 4"),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_what_the_loop_refuses_is_refused(name):
    """The loop's create_config refuses the combination, and so does SubstrateConfig.validate, each
    saying which."""
    make, loop_reason, fields, api_reason = REFUSED[name]
    with pytest.raises(ValueError, match=loop_reason):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            EMRHyperNEAT().create_config(make())
    with pytest.raises(ValueError, match=api_reason):
        _recurrent_substrate_config(**fields).validate()


def _classes(backward: bool, lateral: bool, self_loops: bool, level: int) -> Callable[[], Dict]:
    """A class combination no preset names, from hidden_only with the classes set."""
    return lambda: _config("paper", "emr", preset="hidden_only", level=level,
                           recurrence={"allow_backward": backward, "allow_lateral": lateral,
                                       "allow_self_loops": self_loops})


#: What the loop accepts with true neuromodulation of a recurrent substrate, and substrate_config
#: with it: under 'emr' every combination a caller that draws pictures may ask for (per-node and
#: global activations, a palette and the uniform decode, h->h aggregation 'sum' and 'dynamic', every
#: preset, the three class combinations no preset names, levels up to 4, the cap policies, an
#: explicit step count, both h->h activation modes, and that caller's mix of them); under
#: 'drive_once' what its additive rule applies; 'anti' in its form.
ACCEPTED = {
    "emr_full18_uniform": lambda: _config("paper", "emr", activations=FULL18_UNIFORM),
    "emr_global_sin": lambda: _config("steep_gate", "emr", activations=GLOBAL_SIN),
    "emr_hh_dynamic": lambda: _config("paper", "emr", activations=PER_NODE_6,
                                      aggregation=HH_DYNAMIC),
    "emr_hh_dynamic_tanh": lambda: _config("paper", "emr", activations=PER_NODE_6,
                                           aggregation=HH_DYNAMIC, hh_mode="tanh"),
    **{f"emr_{preset}_L{level}": (lambda preset=preset, level=level: _config(
        "paper", "emr", preset=preset, level=level))
       for preset in ("hidden_only", "with_backward", "with_lateral", "with_self", "full_recurrent")
       for level in (1, 2, 3, 4)},
    **{f"emr_classes_b{int(b)}l{int(la)}s{int(s)}_L{level}": _classes(b, la, s, level)
       for b, la, s in ((True, True, False), (True, False, True), (False, True, True))
       for level in (1, 4)},
    "emr_fori_loop": lambda: _config("paper", "emr", level=3,
                                     recurrence={"multi_hop_algorithm": "fori_loop"}),
    "emr_cap_truncate": lambda: _config("paper", "emr", recurrence={"sparse_cap_policy": "truncate",
                                                                    "max_sparse_conns": 40}),
    "emr_cap_grow": lambda: _config("paper", "emr", recurrence={
        "sparse_cap_policy": "grow", "max_sparse_conns": 40, "max_sparse_conns_ceiling": 4000}),
    "emr_activate_time": lambda: _config("paper", "emr", recurrence={"activate_time": 3}),
    "emr_gating_nt_2": lambda: _config("steep_gate", "emr", modulation_mode="gating_only",
                                       num_nt_types=2),
    "emr_nt_6_no_inversion": lambda: _config("paper", "emr", num_nt_types=6,
                                             use_output_inversion=False),
    # That caller's mix: the 18-function palette, dynamic h->h aggregation, level 4, the cap grown
    # and an explicit step count; a global tanh with h->h 'tanh' and the cap truncating
    "emr_mix_L4_grow_dynamic_palette": lambda: _config(
        "steep_gate", "emr", level=4, activations=FULL18_UNIFORM, aggregation=HH_DYNAMIC,
        recurrence={"sparse_cap_policy": "grow", "max_sparse_conns": 40,
                    "max_sparse_conns_ceiling": 4000, "activate_time": 7}),
    "emr_global_tanh_hh_tanh_truncate": lambda: _config(
        "paper", "emr", activations={"mode": "global", "hidden_activation": "tanh",
                                     "output_activation": "tanh"},
        hh_mode="tanh", recurrence={"sparse_cap_policy": "truncate", "max_sparse_conns": 20}),
    # A global softmax under h->h 'tanh', which never applies it (test_global_softmax_axis.py
    # has the settings that do, against the loop)
    "emr_global_softmax_hh_tanh": lambda: _config("paper", "emr", activations=GLOBAL_SOFTMAX,
                                                  hh_mode="tanh"),
    "drive_once_global_softmax_hh_tanh": lambda: _config("steep_gate", "drive_once",
                                                         activations=GLOBAL_SOFTMAX,
                                                         hh_mode="tanh"),
    "drive_once_initial_only_per_node": lambda: _config("paper", "drive_once",
                                                        activations=FULL18_UNIFORM,
                                                        hh_mode="initial_only"),
    "drive_once_tanh_per_node": lambda: _config("steep_gate", "drive_once", activations=PER_NODE,
                                                hh_mode="tanh"),
    "drive_once_every_iteration_plain": lambda: _config("paper", "drive_once"),
    "drive_once_every_iteration_global_tanh": lambda: _config("paper", "drive_once", activations={
        "mode": "global", "hidden_activation": "tanh", "output_activation": "sigmoid"}),
    **{f"anti_nt_{n}": (lambda n=n: _config("anti", "emr", num_nt_types=n,
                                            use_output_inversion=False))
       for n in (2, 3, 4)},
    "anti_drive_once": lambda: _config("anti", "drive_once", hh_mode="initial_only",
                                       use_output_inversion=False),
}


@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_what_the_loop_accepts_is_described(name):
    """The loop's configuration accepts the combination, substrate_config describes it, and the
    population's substrates settle under it: outputs for every genome and, per genome, the loop's
    activate_time states over its own positions."""
    algo, state = _setup(ACCEPTED[name]())
    config = algo.substrate_config()
    assert config.hidden_to_hidden and config.neuromodulation is not None
    assert config.neuromodulation.node_rule == algo.neuromod_config.node_rule
    assert config.settling_rule == algo._settling_rule
    with warnings.catch_warnings():
        # The cap policies 'truncate' and 'grow' say what they did (SparseCapWarning)
        warnings.simplefilter("ignore", UserWarning)
        substrate = algo.population_substrates(state)
    inputs = algo._cached_inputs
    outputs, states = substrate_forward(substrate, inputs, return_states=True)
    assert np.asarray(outputs).shape == (POP, inputs.shape[0], len(config.output_coords))
    masks = np.asarray(substrate.mask)
    assert len(states) == POP
    for genome, settled in enumerate(states):
        assert settled.shape == (algo.activate_time, inputs.shape[0], int(masks[genome].sum()))


def test_a_given_neuromodulation_is_held_to_the_loops_rules():
    """A caller-built SubstrateNeuromodulation on a recurrent substrate is checked at forward time
    from its own settings, on a configuration with neuromodulation settings or without, as validate
    checks a configured one and with validate's message, an unknown modulation mode or node rule
    included; on a feed-forward substrate an unknown one keeps 0.3.0's message."""
    run = _run("steep_gate_drive_once_per_node")  # 'drive_once', per-node functions, 'initial_only'
    substrate, inputs, nm = run.substrate, run.inputs, run.neuromodulation
    without = substrate._replace(config=substrate.config._replace(neuromodulation=None))
    _assert_same_program(substrate_forward(without, inputs, neuromodulation=nm), run.outputs)
    every_iteration = without._replace(
        config=without.config._replace(hh_activation_mode="every_iteration"))
    with pytest.raises(ValueError, match="applies tanh at every step after the first"):
        substrate_forward(every_iteration, inputs, neuromodulation=nm)
    weight = np.asarray(substrate.W1)[:, 0, :] / substrate.config.max_weight
    for settings, reason in (
            (nm.settings._replace(node_rule="anti", output_inversion=True), "inverts no output"),
            (nm.settings._replace(node_rule="anti", output_inversion=False,
                                  modulation_mode="gain_bias_only"), "is not one of its forms"),
            (nm.settings._replace(node_rule="anti", output_inversion=False, num_nt_types=5),
             "at most 4"),
            (nm.settings._replace(num_nt_types=7), "must be 2 to 6"),
            (nm.settings._replace(node_rule="other"), "neuromodulation.node_rule must be one of"),
            (nm.settings._replace(modulation_mode="other"),
             "neuromodulation.modulation_mode must be one of"),
            (nm.settings._replace(modulation_strength=float("inf")), "must be a finite number"),
            # finite in float64, inf in float32, the precision the forward reads it at
            (nm.settings._replace(modulation_strength=1e40), "must be a finite number"),
            (nm.settings._replace(receptor_derivation="other"),
             "receptor_derivation must be one of")):
        channels = min(max(int(settings.num_nt_types), 2), 6)
        receptors = jnp.asarray(api.derive_receptor_from_weight(weight, "tanh", channels)) * 2 - 1
        given = nm._replace(settings=settings, receptors=receptors,
                            neurotransmitter=jnp.full(max(int(settings.num_nt_types), 4), 0.5))
        with pytest.raises(ValueError, match=reason) as refused:
            substrate_forward(without, inputs, neuromodulation=given)
        with pytest.raises(ValueError) as by_validate:
            without.config._replace(neuromodulation=settings).validate()
        assert str(refused.value) == str(by_validate.value)
    # The feed-forward path keeps 0.3.0's message for an unknown mode or node rule
    feed_forward = substrate._replace(config=substrate.config._replace(hidden_to_hidden=False),
                                      hh_agg_indices=None, **{field: None for field in HH_FIELDS})
    unknown = nm._replace(settings=nm.settings._replace(node_rule="other"))
    with pytest.raises(ValueError, match="unknown modulation mode 'full' or node rule 'other'"):
        substrate_forward(feed_forward, inputs, neuromodulation=unknown)
    dynamic = _run("hh_dynamic")
    drive_once_dynamic = dynamic.substrate._replace(config=dynamic.substrate.config._replace(
        settling_rule="drive_once", hh_activation_mode="initial_only", neuromodulation=None))
    with pytest.raises(ValueError, match="hh_aggregation_mode 'dynamic'"):
        substrate_forward(drive_once_dynamic, dynamic.inputs,
                          neuromodulation=dynamic.neuromodulation)


def test_the_strength_is_judged_in_float32():
    """The recurrent path takes a strength of any integer or floating type that stays finite in
    float32, the precision its forward reads it at, and evaluates it as the same float (bit for
    bit on the CPU); it refuses the rest. 1e40 is finite in float64 but inf in float32, and is
    refused."""
    largest = float(np.finfo(np.float32).max)
    for strength in (2, 2.0, np.int64(2), np.float32(2.0), jnp.bfloat16(2.0), jnp.asarray(2.0),
                     np.asarray(2.0), 10 ** 30, largest, -largest):
        assert api._finite_number(strength), strength
    # 3.4028236e38 rounds to inf in float32; 10 ** 400 does not fit a float64
    for strength in (float("nan"), float("inf"), -float("inf"), 1e40, -1e40, 3.4028236e38,
                     10 ** 400, True, np.bool_(False), "2.0", None, np.array([2.0]), 2 + 0j):
        assert not api._finite_number(strength), strength
    run = _run("paper_emr")
    nm = run.neuromodulation
    assert nm.settings.modulation_strength == 2.0
    for strength in (2, np.float32(2.0), jnp.bfloat16(2.0)):
        given = nm._replace(settings=nm.settings._replace(modulation_strength=strength))
        _assert_same_program(substrate_forward(run.substrate, run.inputs, neuromodulation=given),
                             run.outputs)


def test_the_receptor_derivations_are_the_engines():
    """The methods the recurrent path accepts are derive_receptor_from_weight's branches: each
    derives receptors there, and another name raises there."""
    source = inspect.getsource(engine.derive_receptor_from_weight)
    assert tuple(re.findall(r"method == '(\w+)'", source)) == api._RECEPTOR_DERIVATIONS
    weight = jnp.linspace(-1.0, 1.0, 5)
    for method in api._RECEPTOR_DERIVATIONS:
        assert api.derive_receptor_from_weight(weight, method, 4).shape == (5, 4)
    with pytest.raises(ValueError, match="Unknown receptor derivation method"):
        api.derive_receptor_from_weight(weight, "other", 4)


def test_the_feed_forward_checks_stay_as_in_0_3_0():
    """The checks of the recurrent path (a finite strength, a known receptor derivation) leave a
    feed-forward configuration's validate as in 0.3.0, which accepts both; the same settings on a
    recurrent configuration are refused."""
    settings = api.NeuromodulationSettings(modulation_strength=float("nan"),
                                           receptor_derivation="bogus")
    config = SubstrateConfig(input_coords=((0.0, -1.0),), output_coords=((0.0, 1.0),),
                             max_depth=2, neuromodulation=settings)
    assert config.validate() is config
    with pytest.raises(ValueError, match="modulation_strength must be a finite number"):
        config._replace(hidden_to_hidden=True).validate()


# A configuration that bypassed validate

@pytest.mark.parametrize("name", ["paper_emr", "steep_gate_drive_once_per_node"])
def test_a_configuration_built_without_validate_evaluates_as_the_validated_one(name):
    """A recurrent configuration with neuromodulation that never went through validate, built by
    _replace from a feed-forward one or field by field, evaluates bit for bit as the one
    substrate_config validated (on a GPU under the 'drive_once' rule within GPU_SCATTER_ATOL),
    configured and with a caller-built neuromodulation, states included: the forward's validate
    only reads it."""
    run = _run(name)
    substrate, inputs, nm = run.substrate, run.inputs, run.neuromodulation
    config = substrate.config
    feed_forward = SubstrateConfig(input_coords=config.input_coords,
                                   output_coords=config.output_coords, max_depth=config.max_depth)
    rebuilt = {"_replace": feed_forward._replace(**config._asdict()),
               "fields": SubstrateConfig(**config._asdict())}
    outputs, states = substrate_forward(substrate, inputs, return_states=True)
    drive_once = config.settling_rule == "drive_once"
    for how, other_config in rebuilt.items():
        assert other_config == config and other_config.hidden_to_hidden, how
        other = substrate._replace(config=other_config)
        other_outputs, other_states = substrate_forward(other, inputs, return_states=True)
        _assert_same_program(other_outputs, outputs, drive_once)
        for mine, theirs in zip(other_states, states):
            _assert_same_program(mine, theirs, drive_once)
        _assert_same_program(substrate_forward(other, inputs, neuromodulation=nm), run.outputs,
                             drive_once)


#: What validate refuses on a recurrent configuration with neuromodulation, each set by _replace
#: (which skips validate), with a phrase of validate's message: a change of the neuromodulation
#: settings, of the configuration, or of both. Without its call of validate the forward would run
#: a NaN strength (NaN outputs), a strength of 1e40 as inf (which it is in float32), activate_time
#: 0 under 'drive_once' (the node of the input drive, states of shape (0, samples, count)), the
#: typo (as 'emr', without the 'drive_once' refusals) and 'initial_only' under 'emr' (settling
#: through tanh).
UNVALIDATED = {
    "anti_with_inversion": (dict(node_rule="anti", output_inversion=True), {},
                            "inverts no output"),
    "strength_nan": (dict(modulation_strength=float("nan")), {},
                     "modulation_strength must be a finite number"),
    "strength_inf_in_float32": (dict(modulation_strength=1e40), {},
                                "modulation_strength must be a finite number"),
    "receptor_derivation_unknown": (dict(receptor_derivation="bogus"), {},
                                    "receptor_derivation must be one of"),
    "activate_time_0": ({}, dict(activate_time=0),
                        "activate_time must be an integer of at least 1"),
    "activate_time_0_drive_once": ({}, dict(_DRIVE_ONCE_INITIAL, activate_time=0),
                                   "activate_time must be an integer of at least 1"),
    "settling_rule_typo": ({}, dict(settling_rule="DRIVE_ONCE", hh_activation_mode="initial_only"),
                           "settling_rule must be one of"),
    "initial_only_under_emr": ({}, dict(hh_activation_mode="initial_only"),
                               "hh_activation_mode must be one of"),
}


@pytest.mark.parametrize("name", sorted(UNVALIDATED))
def test_a_configuration_that_bypassed_validate_is_refused_at_forward_time(name):
    """The forward of true neuromodulation on a recurrent substrate validates the configuration
    first, so a setting validate refuses, set by _replace, is refused there with validate's
    message: configured, with and without return_states, and given by the caller on the
    configuration without neuromodulation settings."""
    run = _run("paper_emr")
    substrate, inputs, nm = run.substrate, run.inputs, run.neuromodulation
    settings_change, config_change, phrase = UNVALIDATED[name]
    settings = nm.settings._replace(**settings_change)
    config = substrate.config._replace(neuromodulation=settings, **config_change)
    with pytest.raises(ValueError, match=phrase) as by_validate:
        config.validate()
    message = str(by_validate.value)
    configured = substrate._replace(config=config)
    for return_states in (False, True):
        with pytest.raises(ValueError) as refused:
            substrate_forward(configured, inputs, return_states=return_states)
        assert str(refused.value) == message
    given = substrate._replace(config=config._replace(neuromodulation=None))
    with pytest.raises(ValueError) as refused:
        substrate_forward(given, inputs, neuromodulation=nm._replace(settings=settings))
    assert str(refused.value) == message


def test_substrate_config_still_refuses_what_the_api_cannot_build():
    """With true neuromodulation of a recurrent substrate, substrate_config refuses receptors from
    dedicated CPPN outputs, the self-connection query and the h->h cache."""
    for overrides, reason in ((dict(receptor_from_weight=False), "receptor_from_weight false"),
                              (dict(use_self_connection_query=True), "use_self_connection_query")):
        algo = EMRHyperNEAT()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            algo.create_config(_config("paper", "emr", **overrides))
        with pytest.raises(ValueError, match=reason):
            algo.substrate_config()
    algo = EMRHyperNEAT()
    algo.create_config(_config("paper", "emr", recurrence={"hh_cache_enabled": True}))
    with pytest.raises(ValueError, match="h->h cache"):
        algo.substrate_config()


# evaluate_genome_on_problem

@pytest.mark.parametrize("name", ["paper_emr", "steep_gate_drive_once"])
def test_evaluate_genome_on_problem_gives_the_recorded_fitness(name):
    """The champion scored through the substrate API reproduces the fitness it was selected on,
    within the bound of the comparison against the loop."""
    make, _ = CASES[name]
    algo, state = _setup(make())
    problem = XORProblem()
    state, _ = algo.run_generation(state, problem)
    nodes, conns, fitness, _ = algo.get_best_genome_with_fitness()
    scored = algo.evaluate_genome_on_problem(state, nodes, conns, problem)
    bound = CPU_FITNESS_ATOL if exact_backend() else GPU_FORWARD_ATOL
    assert scored == pytest.approx(fitness, abs=bound)


# The comparison helpers (conftest)

def test_assert_close_across_programs_with_nan():
    """The tolerance comes from the largest finite magnitude, NaN must sit at the same entries in
    both, and an infinity must be the same infinity: a population with an overflowing genome keeps
    its tolerance (a scale taken over NaN would make the tolerance NaN and fail every
    comparison)."""
    ulp = float(np.spacing(np.float32(1.0)))
    desired = np.array([[0.5, 1.0, np.nan], [np.inf, -0.25, np.nan]], dtype=np.float32)
    assert_close_across_programs(desired, desired)
    close = desired.copy()
    close[0, 0] += np.float32(100 * ulp)  # within 128 ulps of the finite scale, 1.0
    assert_close_across_programs(close, desired)
    assert_close_across_programs(np.full(3, np.nan), np.full(3, np.nan))
    far = desired.copy()
    far[0, 1] += np.float32(200 * ulp)
    with pytest.raises(AssertionError, match="ulps of 1"):
        assert_close_across_programs(far, desired)
    for entry, value, reason in (((0, 2), 0.0, "NaN at different entries"),
                                 ((1, 1), np.nan, "NaN at different entries"),
                                 ((1, 0), -np.inf, "ulps of 1"),
                                 ((1, 0), 1.0, "ulps of 1")):
        wrong = desired.copy()
        wrong[entry] = value
        with pytest.raises(AssertionError, match=reason):
            assert_close_across_programs(wrong, desired)


@pytest.mark.parametrize("flags, threaded", [
    (None, True),
    ("", True),
    ("--xla_cpu_multi_thread_eigen=false", False),
    ("--xla_cpu_multi_thread_eigen=0", False),
    ("--xla_cpu_multi_thread_eigen=False", False),
    ("--xla_cpu_multi_thread_eigen=FALSE", False),
    ('--xla_cpu_multi_thread_eigen="false"', False),
    ("--xla_cpu_multi_thread_eigen='false'", False),
    ('--xla_cpu_multi_thread_eigen="0"', False),
    ("--xla_cpu_multi_thread_eigen=true", True),
    ("--xla_cpu_multi_thread_eigen", True),
    ("--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false", False),
    ("--xla_cpu_multi_thread_eigen=false --xla_cpu_multi_thread_eigen=true", True),
    ("--xla_cpu_multi_thread_eigen=true --xla_cpu_multi_thread_eigen=false", False),
    ("--xla_cpu_multi_thread_eigen_other=false", True),
])
def test_cpu_multi_threaded_eigen_reads_xla_flags(monkeypatch, flags, threaded):
    """True on the CPU unless XLA_FLAGS sets --xla_cpu_multi_thread_eigen to false or 0, quoted or
    not, the last setting winning; never off the CPU. For the settings of the flag itself that is
    XLA's own reading, probed on an Apple M4 Max (jax 0.6.1, one process each, unset included):
    the false ones ran single-threaded and the others multi-threaded. The entry with a longer flag
    name (`--xla_cpu_multi_thread_eigen_other=false`) only checks that the helper does not match
    it; XLA itself refuses an unknown flag."""
    if flags is None:
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    else:
        monkeypatch.setenv("XLA_FLAGS", flags)
    assert cpu_multi_threaded_eigen() is (threaded and exact_backend())
