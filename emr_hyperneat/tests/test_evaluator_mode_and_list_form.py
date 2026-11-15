"""evaluate_with_neurotransmitter's default modes and the list form's inner generation.

The default modes. In 0.3.1 evaluate_with_neurotransmitter scored with modulation_mode 'full' and
fitness_mode 'mse' unless the call named others, whatever the last generation's selection had
scored with. After a single-task generation under neuromodulation.modulation_mode 'gating_only' or
'gain_bias_only', or under an emr_hyperneat.fitness_mode other than 'mse', a call without modes gave
another fitness than the generation's: here (XOR, depth 2, population 20, seed 42, one generation)
every genome, by up to 0.105 feed-forward under 'gating_only', 0.109 recurrent, 0.376 under 'bce'
and 0.193 under 'accuracy'. The list form of run_generation_multitask selects on
multitask.modulation_mode and multitask.fitness_mode ('full' and 'mse' without a multitask block),
whatever the single-task settings say. Both arguments default to None, the modes the last
generation's selection scored with, recorded with the receptor densities that generation derived;
a call that names its modes scores as in 0.3.1. evaluate_with_custom_nt and
evaluate_multi_task_true_neuromodulated keep the prototype's 'full' and 'mse', as their docstrings
say. The multi-task evaluators do not compute 'ce' (compute_fitness refuses it), so a call without
a fitness mode after a single-task generation under 'ce' raises and asks for one; 0.3.1 returned
1 - MSE.

The list form's inner generation. In 0.3.1 the list form's one inner generation
(run_generation_verbose under _multitask_inner_generation) built the substrate, then evaluated the
first problem under the single-task modes and the first configured task's vector (the single-task
vector without an enabled multitask block) and told NEAT; the list form discarded that state and
told the input state the tasks' aggregated fitness. The inner evaluation reached nothing selection
reads, but it stayed in the records: _cached_fitnesses, _last_fitnesses and the extended metrics'
fitness fields held the first problem's fitness under the inner network. The inner generation builds
the substrate, the caches and the receptors and keeps the counters and the topology metrics, and
builds no forward, evaluates nothing, tells nothing and records no champion; the list form records
the aggregated fitness it tells. The stop has its own flag, _multitask_structure_only, which the
list form sets beside _multitask_inner_generation: a generation under _multitask_inner_generation
alone, the whole-grid reference of test_global_softmax_axis.py, is scored as any generation. The
records hold the array the list form's tell receives, whatever the aggregation and the bonuses
(TOLD_NOT_MEAN, where it is not the tasks' mean). A list form that raises once its inner generation
has run keeps that generation's arrays, receptors and counter and records no fitness or champion
(None, and NaN in the extended metrics), and the default modes are its own: the previous
generation's records are not restored, because they describe another population than the arrays the
raised generation left. Its modes are recorded right after the inner generation, before anything
that can raise, a setting of the wrong type on a configuration object set by hand included
(RAISE_TRIGGERS). The list form sets both flags around its inner generation and restores both
however it ends (LIST_FORM_ENDS): a _multitask_structure_only left set would stop every later
single-task generation of the instance before its forward.

The references. For the default modes, the evaluator the call runs (the list form's, tanh nodes) on
the arrays the generation evaluated, under the modes named: one program on the same operands, bit
for bit on the CPU; and, where the call was measured against the loop (the CPU under
single-threaded Eigen, test_multitask_after_compaction.py), the generation's own fitness. For the
list form's selection, the list form recomposed without it on a second engine from the same seed: a
single-task generation of the first problem that keeps its arrays on the ceiling (what the inner
generation was in 0.3.1, its told state dropped), each task under its vector with the list form's
evaluator, the mean, NaN to -inf and NEAT's tell of the input state. Two computations, so no
substrate value is pinned.
"""

import collections
import dataclasses
import functools
import hashlib
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    DEFAULT_SEED,
    GPU_TRAJECTORY_RTOL,
    LOGIC_GATE_INPUTS,
    LOGIC_GATE_TARGETS,
    NT_TASK_PRESETS,
    MultiTaskProblem,
    XORProblem,
    cpu_multi_threaded_eigen,
    create_config_with_multi_task,
    create_config_with_neuromodulation,
    exact_backend,
)
from emr_hyperneat import emrhyperneat as engine

#: The single-task generation's NT vector for four NT types (run_generation_verbose)
DEFAULT_NT = (0.5, 0.5, 0.5, 1.0)
#: A vector whose fourth channel is below 1, so the output inversion is not the identity
CUSTOM_NT = (0.9, 0.1, 0.7, 0.6)
DEPTH = 2
POPULATION = 20

# name -> recurrence preset, neuromodulation.modulation_mode, emr_hyperneat.fitness_mode. The
# 'full' and 'mse' rows are the controls: 0.3.1's defaults, which a call without modes gave there
# as well.
SINGLE_TASK: Dict[str, Tuple[str, str, str]] = {
    'feedforward_full_mse': ('feedforward', 'full', 'mse'),
    'feedforward_gating_only_mse': ('feedforward', 'gating_only', 'mse'),
    'feedforward_gain_bias_only_mse': ('feedforward', 'gain_bias_only', 'mse'),
    'feedforward_full_bce': ('feedforward', 'full', 'bce'),
    'feedforward_full_accuracy': ('feedforward', 'full', 'accuracy'),
    'hidden_only_full_mse': ('hidden_only', 'full', 'mse'),
    'hidden_only_gating_only_mse': ('hidden_only', 'gating_only', 'mse'),
    'hidden_only_gain_bias_only_bce': ('hidden_only', 'gain_bias_only', 'bce'),
}
CONTROLS = ('feedforward_full_mse', 'hidden_only_full_mse')

# name -> the multitask block's settings (None: no block), neuromodulation.modulation_mode and
# emr_hyperneat.fitness_mode, for the list form on XOR and AND. The first case selects on modes
# 0.3.1's defaults are not; the controls select on 'full' and 'mse' under single-task settings that
# say otherwise, so they fail if a call reads the single-task settings (or the inner generation's
# record of them) after the list form.
LIST_FORM_MODES: Dict[str, Tuple[Optional[Dict[str, str]], str, str]] = {
    'block_gating_only_bce': (
        {'modulation_mode': 'gating_only', 'fitness_mode': 'bce'}, 'full', 'mse'),
    'block_defaults_under_single_task_gating_only_accuracy': ({}, 'gating_only', 'accuracy'),
    'no_block_under_single_task_gain_bias_only_bce': (None, 'gain_bias_only', 'bce'),
}

#: The list form's tasks, population and generations
TASKS = ('xor', 'and', 'or')
LIST_POPULATION = 30
GENERATIONS = 3
# name -> recurrence preset, whether the configuration has a multitask block (without one, the
# inner generation's compaction is blocked by _multitask_inner_generation alone)
LIST_FORM: Dict[str, Tuple[str, bool]] = {
    'hidden_only_block': ('hidden_only', True),
    'hidden_only_no_block': ('hidden_only', False),
    'feedforward_block': ('feedforward', True),
}
#: What the inner generation does not run
INNER_STEPS = ('_build_forward', '_evaluate_population', '_compiled_tell', '_record_evaluation')
#: Multitask settings under which the told fitness is neither the tasks' mean nor their aggregate
#: before the bonus: aggregation 'min' with the generalist 'min_bonus' at weight 0.1 tells 1.1 times
#: each genome's lowest task fitness
TOLD_NOT_MEAN = {'fitness_aggregation': 'min', 'generalist_bonus_type': 'min_bonus',
                 'generalist_bonus_weight': 0.1}
# name -> the single-task neuromodulation.modulation_mode and emr_hyperneat.fitness_mode under a
# list form whose multitask block selects on 'gating_only' and 'bce', for the list form that raises
# after its inner generation
RAISED_UNDER: Dict[str, Tuple[str, str]] = {
    'single_task_gain_bias_only_accuracy': ('gain_bias_only', 'accuracy'),
    'single_task_full_ce': ('full', 'ce'),
}
# name -> what makes that list form raise once its inner generation has run: the call's
# aggregation method, settings set by hand on the configuration object after a first list form,
# and the error. An unknown aggregation method is refused after the tasks are evaluated; a
# per_task_activation that is not a mapping and an orthogonality_bonus_weight of None raise where
# the settings are first read, right after the modes are recorded.
RAISE_TRIGGERS: Dict[str, Tuple[str, Dict[str, Any], type, str]] = {
    'unknown_aggregation_method': ('median', {}, ValueError, 'Unknown aggregation method'),
    'per_task_activation_not_a_mapping': (
        'mean', {'per_task_activation': ['tanh', 'tanh']}, AttributeError, "no attribute 'get'"),
    'orthogonality_bonus_weight_none': (
        'mean', {'orthogonality_bonus_weight': None}, TypeError, "'>' not supported"),
}
#: How a list form ends before a single-task generation on the same instance: it returns; it raises
#: once its inner generation has run (an unknown aggregation method); or its inner generation raises
#: (its host side, made to fail here), the case the list form's finally is for
LIST_FORM_ENDS = ('returned', 'raised_after_its_inner_generation',
                  'raised_inside_its_inner_generation')


def _emr(config: Dict[str, Any]) -> Dict[str, Any]:
    return config['algorithm_params']['emrhyperneat']['emr_hyperneat']


def _nt(vector: Any) -> jnp.ndarray:
    return jnp.asarray(vector, dtype=jnp.float32)


def _same_program(actual: Any, desired: Any) -> None:
    """One evaluator on the same operands: bit for bit on the CPU, to GPU_TRAJECTORY_RTOL off it."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    if exact_backend():
        np.testing.assert_array_equal(actual, desired)
    else:
        np.testing.assert_allclose(actual, desired, rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


def _bit_for_bit() -> bool:
    """Where the call was measured against the loop: the CPU under single-threaded Eigen."""
    return exact_backend() and not cpu_multi_threaded_eigen()


def _digest(value: Any) -> str:
    """A sha256 of the arrays in ``value``, each by dtype, shape and bytes: a pytree's leaves, a
    dataclass's fields (SparseHiddenConnections is not a pytree), and a marker for None."""
    h = hashlib.sha256()

    def feed(item: Any) -> None:
        if item is None:
            h.update(b'None')
        elif dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                feed(getattr(item, field.name))
        else:
            leaves = jax.tree_util.tree_leaves(item)
            if len(leaves) == 1 and leaves[0] is item:
                array = np.asarray(item)
                h.update(str(array.dtype).encode() + str(array.shape).encode())
                h.update(np.ascontiguousarray(array).tobytes())
            else:
                for leaf in leaves:
                    feed(leaf)

    feed(value)
    return h.hexdigest()


def _evaluate(algo: Any, nt: Any, inputs: Any, targets: Any, modulation_mode: str,
              fitness_mode: str) -> np.ndarray:
    """The evaluator evaluate_with_neurotransmitter runs (tanh nodes, no branch gating) on the
    arrays the last generation evaluated (_cached_eval_*) with its receptors, under the given
    modes."""
    hh = algo._cached_eval_sparse_hh
    has_hh = hh is not None and algo._forward_mode == engine.ForwardPassMode.HYBRID_SPARSE_HH
    evaluator = algo._create_eval_population_true_neuromodulated(
        has_hh=has_hh, activate_time=algo.activate_time if has_hh else 0, hidden_act_fn=jnp.tanh)
    args = (algo._cached_eval_W1, algo._cached_eval_W2, algo._neuromod_true['receptor_densities'],
            algo._neuromod_true['base_gains'], _nt(nt), jnp.asarray(inputs), jnp.asarray(targets))
    if has_hh:
        args += (hh.from_indices, hh.to_indices, hh.weights, hh.valid_mask)
    return np.asarray(evaluator(*args, fitness_mode, modulation_mode, None))


def _single_task_engine(preset: str, modulation_mode: str, fitness_mode: str) -> Any:
    """An engine after one single-task generation on XOR under true neuromodulation with the given
    modes."""
    config = create_config_with_neuromodulation('true_neuromodulation_4nt',
                                                recurrence_preset=preset, max_depth=DEPTH,
                                                population_size=POPULATION,
                                                modulation_mode=modulation_mode)
    _emr(config)['fitness_mode'] = fitness_mode
    algo = engine.EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=DEFAULT_SEED)
    algo.run_generation(state, problem)
    return algo


@functools.lru_cache(maxsize=None)
def _single_task(name: str) -> Any:
    return _single_task_engine(*SINGLE_TASK[name])


# A call without modes after a single-task generation

@pytest.mark.parametrize('name', sorted(SINGLE_TASK))
def test_a_call_without_modes_scores_as_the_single_task_selection(name):
    """After a single-task generation a call without modes scores with
    neuromodulation.modulation_mode and emr_hyperneat.fitness_mode, the modes selection used: what
    the call that names them gives, and on the backend where the call was measured against the loop
    the generation's own fitness of every genome. A call that names its modes scores as in 0.3.1
    (the evaluator under those modes). In 0.3.1 the call without modes scored 'full' and 'mse',
    which moves every genome of each case but the controls."""
    _, modulation_mode, fitness_mode = SINGLE_TASK[name]
    algo = _single_task(name)
    inputs, targets = algo._cached_inputs, algo._cached_targets
    named = algo.evaluate_with_neurotransmitter(_nt(DEFAULT_NT), inputs, targets,
                                                modulation_mode=modulation_mode,
                                                fitness_mode=fitness_mode)
    _same_program(named, _evaluate(algo, DEFAULT_NT, inputs, targets, modulation_mode,
                                   fitness_mode))
    if name not in CONTROLS:
        old = algo.evaluate_with_neurotransmitter(_nt(DEFAULT_NT), inputs, targets,
                                                  modulation_mode='full', fitness_mode='mse')
        assert not np.array_equal(np.asarray(old), np.asarray(named)), (
            "0.3.1's defaults score otherwise")
    _same_program(algo.evaluate_with_neurotransmitter(_nt(DEFAULT_NT), inputs, targets), named)
    if _bit_for_bit():
        np.testing.assert_array_equal(np.asarray(engine.sanitize_fitness(jnp.asarray(named))[0]),
                                      np.asarray(algo._cached_fitnesses))


def test_each_mode_defaults_on_its_own():
    """A call that names one mode takes the other from the generation's selection."""
    algo = _single_task('hidden_only_gain_bias_only_bce')
    inputs, targets = algo._cached_inputs, algo._cached_targets
    for given, expected in (({'modulation_mode': 'full'}, ('full', 'bce')),
                            ({'fitness_mode': 'mse'}, ('gain_bias_only', 'mse'))):
        _same_program(algo.evaluate_with_neurotransmitter(_nt(CUSTOM_NT), inputs, targets, **given),
                      _evaluate(algo, CUSTOM_NT, inputs, targets, *expected))


def test_the_prototype_wrappers_keep_full_modulation_and_mse():
    """The control: evaluate_with_custom_nt and evaluate_multi_task_true_neuromodulated score the
    prototype's 'full' and 'mse' whatever the generation ran, as their docstrings say (they forward
    the call with both modes named; without them they would score the generation's modes). The case
    is one where those score otherwise."""
    algo = _single_task('hidden_only_gain_bias_only_bce')
    inputs, targets = algo._cached_inputs, algo._cached_targets
    prototype = algo.evaluate_with_neurotransmitter(_nt(CUSTOM_NT), inputs, targets,
                                                    modulation_mode='full', fitness_mode='mse')
    generation = algo.evaluate_with_neurotransmitter(_nt(CUSTOM_NT), inputs, targets,
                                                     modulation_mode='gain_bias_only',
                                                     fitness_mode='bce')
    assert not np.array_equal(np.asarray(prototype), np.asarray(generation)), (
        "the generation's modes score otherwise")
    _same_program(algo.evaluate_with_custom_nt(_nt(CUSTOM_NT), inputs, targets), prototype)
    and_targets = jnp.asarray([[0.0], [0.0], [0.0], [1.0]], dtype=jnp.float32)
    tasks = {'xor': (inputs, targets), 'and': (inputs, and_targets)}
    expected = jnp.mean(jnp.stack([jnp.asarray(algo.evaluate_with_neurotransmitter(
        name, task_inputs, task_targets, modulation_mode='full', fitness_mode='mse'))
        for name, (task_inputs, task_targets) in tasks.items()], axis=0), axis=0)
    _same_program(algo.evaluate_multi_task_true_neuromodulated(tasks, 'mean'), expected)


def test_a_call_without_a_fitness_mode_after_a_ce_generation_refuses():
    """The multi-task evaluators do not compute 'ce' (compute_fitness refuses it): after a
    single-task generation under emr_hyperneat.fitness_mode 'ce' a call without a fitness mode
    cannot score as selection did and asks for one (0.3.1 returned 1 - MSE). A call that names one
    scores as in 0.3.1."""
    algo = _single_task_engine('feedforward', 'full', 'ce')
    inputs, targets = algo._cached_inputs, algo._cached_targets
    with pytest.raises(ValueError, match=r"fitness mode 'ce'.*pass fitness_mode"):
        algo.evaluate_with_neurotransmitter(_nt(DEFAULT_NT), inputs, targets)
    _same_program(algo.evaluate_with_neurotransmitter(_nt(DEFAULT_NT), inputs, targets,
                                                      fitness_mode='mse'),
                  _evaluate(algo, DEFAULT_NT, inputs, targets, 'full', 'mse'))


# A call without modes after the list form

@pytest.mark.parametrize('name', sorted(LIST_FORM_MODES))
def test_after_the_list_form_a_call_without_modes_scores_as_its_selection(name):
    """After the list form a call without modes scores each task as the list form's selection did,
    with multitask.modulation_mode and multitask.fitness_mode ('full' and 'mse' without a block),
    not the single-task settings: each task's fitness under its vector. The first case fails on
    0.3.1; the controls pass there and fail if the call reads the single-task settings after the
    list form."""
    block, modulation_mode, fitness_mode = LIST_FORM_MODES[name]
    names = ['xor', 'and']
    config = create_config_with_multi_task(task_names=names, fitness_aggregation='mean',
                                           max_depth=DEPTH, population_size=POPULATION,
                                           recurrence_preset='feedforward')
    emr = _emr(config)
    emr['neuromodulation']['modulation_mode'] = modulation_mode
    emr['fitness_mode'] = fitness_mode
    if block is None:
        emr.pop('multitask')
    else:
        emr['multitask'].update(block)
    algo = engine.EMRHyperNEAT()
    problems = [MultiTaskProblem(task) for task in names]
    state = algo.initialize(algo.create_config(config), problems[0], seed=DEFAULT_SEED)
    nts = [_nt(NT_TASK_PRESETS[task]) for task in names]
    algo.run_generation_multitask(state, problems, nts, aggregation_method='mean')
    selected = ((block or {}).get('modulation_mode', 'full'),
                (block or {}).get('fitness_mode', 'mse'))
    for task, (problem, nt) in enumerate(zip(problems, nts)):
        told = algo._last_multitask['per_task'][task]
        _same_program(_evaluate(algo, nt, problem.inputs, problem.targets, *selected), told)
        _same_program(algo.evaluate_with_neurotransmitter(nt, problem.inputs, problem.targets),
                      told)


def test_after_the_call_form_a_call_without_modes_scores_as_its_selection():
    """The prototype's call form runs the list form with the call's configuration, and the modes it
    selected on stay the default after the call has restored the model's own multitask
    configuration."""
    names = ['xor', 'and']
    config = create_config_with_multi_task(task_names=names, fitness_aggregation='mean',
                                           max_depth=DEPTH, population_size=POPULATION,
                                           recurrence_preset='feedforward')
    algo = engine.EMRHyperNEAT()
    state = algo.initialize(algo.create_config(config), MultiTaskProblem(names[0]),
                            seed=DEFAULT_SEED)
    tasks = {task: (np.asarray(LOGIC_GATE_INPUTS, dtype=np.float32),
                    np.asarray(LOGIC_GATE_TARGETS[task], dtype=np.float32)) for task in names}
    algo.run_generation_multitask(state, engine.MultiTaskConfig(tasks=tasks,
                                                                fitness_aggregation='mean',
                                                                modulation_mode='gain_bias_only',
                                                                fitness_mode='bce'))
    assert algo.multitask_config.modulation_mode == 'full', "the model's own configuration is back"
    for task, name in enumerate(names):
        _same_program(algo.evaluate_with_neurotransmitter(name, *tasks[name]),
                      algo._last_multitask['per_task'][task])


# The list form's inner generation

def _list_form_config(case: str) -> Dict[str, Any]:
    preset, block = LIST_FORM[case]
    config = create_config_with_multi_task(task_names=list(TASKS), fitness_aggregation='mean',
                                           population_size=LIST_POPULATION,
                                           recurrence_preset=preset)
    if not block:
        _emr(config).pop('multitask')
    return config


def _list_form_engine(config: Dict[str, Any]) -> Tuple[Any, List[MultiTaskProblem], Any]:
    algo = engine.EMRHyperNEAT()
    problems = [MultiTaskProblem(task) for task in TASKS]
    state = algo.initialize(algo.create_config(config), problems[0], seed=DEFAULT_SEED)
    return algo, problems, state


def _vectors() -> List[jnp.ndarray]:
    return [_nt(NT_TASK_PRESETS[task]) for task in TASKS]


@pytest.mark.parametrize('case', sorted(LIST_FORM))
def test_the_inner_generation_scores_and_tells_nothing(case, monkeypatch):
    """The list form's inner generation builds no forward, evaluates nothing, tells nothing and
    records no champion (in 0.3.1 each ran once per generation); the list form tells once per
    generation. The controls: the generation counter advances by one per generation, and no
    champion is recorded, as in 0.3.1."""
    algo, problems, state = _list_form_engine(_list_form_config(case))
    calls: collections.Counter = collections.Counter()

    def counted(step: str, function: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            calls[step, algo._multitask_inner_generation] += 1
            return function(*args, **kwargs)
        return call

    for step in INNER_STEPS:
        monkeypatch.setattr(algo, step, counted(step, getattr(algo, step)))
    for generation in range(GENERATIONS):
        state, _ = algo.run_generation_multitask(state, problems, _vectors(),
                                                 aggregation_method='mean')
        assert algo._current_generation == generation + 1
        assert algo._last_eval is None
    assert {step: calls[step, True] for step in INNER_STEPS} == {
        step: 0 for step in INNER_STEPS}, dict(calls)
    assert calls['_compiled_tell', False] == GENERATIONS, dict(calls)


def test_a_generation_kept_on_the_ceiling_alone_is_scored(monkeypatch):
    """The control: a single-task generation run under _multitask_inner_generation alone, as the
    whole-grid reference of test_global_softmax_axis.py runs one, keeps its arrays on the ceiling
    and is built, scored, told and recorded as any generation; only the list form, which also sets
    _multitask_structure_only, stops before the forward. It passes on 0.3.1 as well, and fails if
    the stop is keyed on _multitask_inner_generation (that reference then builds no forward and
    records no fitness)."""
    config = create_config_with_neuromodulation('true_neuromodulation_4nt',
                                                recurrence_preset='hidden_only', max_depth=DEPTH,
                                                population_size=POPULATION)
    algo = engine.EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=DEFAULT_SEED)
    calls: collections.Counter = collections.Counter()

    def counted(step: str, function: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            calls[step] += 1
            return function(*args, **kwargs)
        return call

    for step in INNER_STEPS:
        monkeypatch.setattr(algo, step, counted(step, getattr(algo, step)))
    algo._multitask_inner_generation = True
    try:
        _, metrics = algo.run_generation(state, problem)
    finally:
        algo._multitask_inner_generation = False
    assert dict(calls) == {step: 1 for step in INNER_STEPS}, dict(calls)
    assert algo._cached_active_indices is None, "kept on the ceiling"
    assert algo._last_eval is not None and algo._last_eval['generation'] == 0
    np.testing.assert_array_equal(np.asarray(algo._last_fitnesses),
                                  np.asarray(algo._cached_fitnesses))
    best = engine.fitness_statistics(algo._cached_fitnesses,
                                     metrics.custom_metrics['nan_fitness_count'])[0]
    assert metrics.best_fitness == best == algo.get_extended_metrics().fitness


@pytest.mark.parametrize('case', sorted(LIST_FORM))
def test_the_list_form_records_the_fitness_it_told(case):
    """After each list-form generation _cached_fitnesses and _last_fitnesses hold the aggregated
    fitness NEAT was told, and the extended metrics' fitness fields its statistics, the outer
    metrics'. In 0.3.1 they held the first problem's fitness under the inner network."""
    algo, problems, state = _list_form_engine(_list_form_config(case))
    for _ in range(GENERATIONS):
        state, metrics = algo.run_generation_multitask(state, problems, _vectors(),
                                                       aggregation_method='mean')
        told = np.asarray(algo._last_multitask['aggregated'])
        np.testing.assert_array_equal(np.asarray(algo._cached_fitnesses), told)
        np.testing.assert_array_equal(np.asarray(algo._last_fitnesses), told)
        best, _, low, spread = engine.fitness_statistics(
            jnp.asarray(told), metrics.custom_metrics['nan_fitness_count'])
        extended = algo.get_extended_metrics()
        assert extended is algo.get_recurrence_metrics()
        assert (extended.fitness, extended.min_fitness, extended.max_fitness,
                extended.std_fitness) == (best, low, best, spread)
        assert (metrics.best_fitness, metrics.min_fitness, metrics.max_fitness,
                metrics.std_fitness) == (best, low, best, spread)


def test_the_records_hold_the_told_fitness_not_the_tasks_mean(monkeypatch):
    """The records hold the array the list form's tell receives, whatever the aggregation and the
    bonuses. Under 'mean' without a bonus (the cases above) that array and the tasks' plain mean
    have the same bits, so records of the mean would pass there; here (TOLD_NOT_MEAN) the told
    fitness is 1.1 times the tasks' minimum. In 0.3.1 the records held the first problem's fitness
    under the inner network."""
    config = _list_form_config('hidden_only_block')
    _emr(config)['multitask'].update(TOLD_NOT_MEAN)
    algo, problems, state = _list_form_engine(config)
    told: List[Any] = []
    tell = algo._compiled_tell

    def spy(tell_state: Any, fitnesses: Any) -> Any:
        told.append(fitnesses)
        return tell(tell_state, fitnesses)

    monkeypatch.setattr(algo, '_compiled_tell', spy)
    for _ in range(GENERATIONS):
        state, metrics = algo.run_generation_multitask(
            state, problems, _vectors(), aggregation_method=TOLD_NOT_MEAN['fitness_aggregation'])
        fitness = np.asarray(told[-1])  # the list form's tell, the generation's last
        per_task = np.stack([np.asarray(task) for task in algo._last_multitask['per_task']])
        assert not np.array_equal(fitness, per_task.mean(axis=0)), (
            "the told fitness is not the tasks' mean")
        assert not np.array_equal(fitness, per_task.min(axis=0)), (
            "nor their aggregate before the bonus")
        np.testing.assert_array_equal(np.asarray(algo._cached_fitnesses), fitness)
        np.testing.assert_array_equal(np.asarray(algo._last_fitnesses), fitness)
        best, _, low, spread = engine.fitness_statistics(
            jnp.asarray(fitness), metrics.custom_metrics['nan_fitness_count'])
        extended = algo.get_extended_metrics()
        assert (extended.fitness, extended.min_fitness, extended.max_fitness,
                extended.std_fitness) == (best, low, best, spread)


class Recomposed(NamedTuple):
    """The list form's selection, recomposed without it."""
    state: Any  # NEAT's tell of the input state
    per_task: List[jnp.ndarray]
    aggregated: jnp.ndarray
    nan_count: int
    structure: Any  # the single-task generation's metrics (its topology)


def _recomposed(reference: Any, state: Any, problems: List[MultiTaskProblem]) -> Recomposed:
    """The list form's selection on ``state`` without the list form: a single-task generation of the
    first problem that keeps its arrays on the ceiling (an enabled multitask configuration blocks
    the compaction), its told state dropped, as the inner generation was in 0.3.1; then each task
    under its vector with the list form's evaluator (tanh nodes, 'mse', 'full') on the cached arrays
    and receptors, the mean, NaN to -inf, and NEAT's tell of the input state."""
    _, structure = reference.run_generation_verbose(state, problems[0], skip_metrics=True)
    hh = reference._cached_sparse_hh
    has_hh = hh is not None and reference._forward_mode == engine.ForwardPassMode.HYBRID_SPARSE_HH
    evaluator = reference._create_eval_population_true_neuromodulated(
        has_hh=has_hh, activate_time=reference.activate_time if has_hh else 0,
        hidden_act_fn=engine.ACTIVATION_FUNCTIONS['tanh'])
    per_task = []
    for problem, nt in zip(problems, _vectors()):
        args = (reference._cached_W1, reference._cached_W2,
                reference._neuromod_true['receptor_densities'],
                reference._neuromod_true['base_gains'], nt, jnp.array(problem.inputs),
                jnp.array(problem.targets))
        if has_hh:
            args += (hh.from_indices, hh.to_indices, hh.weights, hh.valid_mask)
        per_task.append(evaluator(*args, 'mse', 'full', None))
    aggregated, nan_count = engine.sanitize_fitness(jnp.mean(jnp.stack(per_task, axis=0), axis=0))
    return Recomposed(reference._compiled_tell(state, aggregated), per_task, aggregated, nan_count,
                      structure)


def _substrate_fields(metrics: Any) -> Dict[str, Any]:
    """The extended metrics' fields that describe the substrate, the connections and the counters:
    all but the fitness fields and the step timings."""
    return {key: value for key, value in dataclasses.asdict(metrics).items()
            if key not in ('fitness', 'min_fitness', 'max_fitness', 'std_fitness')
            and not key.endswith('_ms')}


@pytest.mark.parametrize('case', sorted(LIST_FORM))
def test_selection_is_unchanged(case):
    """Selection is the list form's recomposed on a second engine from the same seed, generation by
    generation: the per-task and aggregated fitness, the told state, the outer metrics (fitness
    statistics, species, generation, topology), the arrays the tasks read and the extended metrics'
    substrate fields and counters. Bit for bit on the CPU, which also compares the states and the
    arrays; off it the fitness to GPU_TRAJECTORY_RTOL. It holds on 0.3.1 too (the inner evaluation
    reached nothing selection reads), and holds without that evaluation."""
    config = _list_form_config(case)
    algo, problems, state = _list_form_engine(config)
    reference, _, reference_state = _list_form_engine(config)
    assert _digest(state) == _digest(reference_state), "one initial population"
    if reference.multitask_config is None:
        # What the inner flag does for the list form without a block: keep the arrays on the
        # ceiling
        reference.multitask_config = engine.MultiTaskConfig(enabled=True)
    topology = ('hidden_nodes_mean', 'hidden_nodes_std', 'connections_mean', 'connections_std')
    for _ in range(GENERATIONS):
        new_state, metrics = algo.run_generation_multitask(state, problems, _vectors(),
                                                           aggregation_method='mean')
        expected = _recomposed(reference, state, problems)
        assert len(algo._last_multitask['per_task']) == len(expected.per_task)
        for told, recomposed in zip(algo._last_multitask['per_task'], expected.per_task):
            _same_program(told, recomposed)
        _same_program(algo._last_multitask['aggregated'], expected.aggregated)
        assert metrics.generation == algo._current_generation == reference._current_generation
        assert [getattr(metrics, key) for key in topology] == [
            getattr(expected.structure, key) for key in topology]
        assert _substrate_fields(algo.get_extended_metrics()) == _substrate_fields(
            reference.get_extended_metrics())
        if exact_backend():
            assert _digest(new_state) == _digest(expected.state), "the told state"
            arrays = ('_cached_W1', '_cached_W2', '_cached_masks_A', '_cached_sparse_hh')
            assert [_digest(getattr(algo, name)) for name in arrays] == [
                _digest(getattr(reference, name)) for name in arrays]
            assert _digest((algo._neuromod_true['receptor_densities'],
                            algo._neuromod_true['base_gains'])) == _digest(
                (reference._neuromod_true['receptor_densities'],
                 reference._neuromod_true['base_gains']))
            best, mean, low, spread = engine.fitness_statistics(expected.aggregated,
                                                                expected.nan_count)
            species = engine.species_summary(state.species, expected.aggregated)
            assert (metrics.best_fitness, metrics.mean_fitness, metrics.min_fitness,
                    metrics.max_fitness, metrics.std_fitness) == (best, mean, low, best, spread)
            assert (metrics.num_species, metrics.species_sizes, metrics.species_fitness) == species
        state = new_state


# A list form that raises after its inner generation

@pytest.mark.parametrize('trigger', sorted(RAISE_TRIGGERS))
@pytest.mark.parametrize('name', sorted(RAISED_UNDER))
def test_a_list_form_that_raises_after_its_inner_generation_records_no_selection(name, trigger):
    """A list form that raises once its inner generation has run (RAISE_TRIGGERS: an unknown
    aggregation method, refused after the tasks are evaluated, or a hand-set setting of the wrong
    type, read right after the inner generation) keeps that generation's arrays, receptors and
    counter, as in 0.3.1, and records no selection: no fitness (None, NaN in the extended metrics)
    and no champion, rather than another population's. In 0.3.1 the records held the first
    problem's fitness under the inner network; restoring the previous generation's would pair its
    fitness with this generation's arrays. A call without modes scores with this call's modes (the
    block's), not the single-task ones the inner generation records with the receptors, under which
    a single-task 'ce' would refuse the call: the list form records its modes right after its inner
    generation (the two hand-set settings raise where the settings are first read, so a record
    placed after them would miss them)."""
    modulation_mode, fitness_mode = RAISED_UNDER[name]
    aggregation_method, hand_set, error, message = RAISE_TRIGGERS[trigger]
    names = ['xor', 'and']
    config = create_config_with_multi_task(task_names=names, fitness_aggregation='mean',
                                           max_depth=DEPTH, population_size=POPULATION,
                                           recurrence_preset='feedforward')
    emr = _emr(config)
    emr['neuromodulation']['modulation_mode'] = modulation_mode
    emr['fitness_mode'] = fitness_mode
    emr['multitask'].update(modulation_mode='gating_only', fitness_mode='bce')
    algo = engine.EMRHyperNEAT()
    problems = [MultiTaskProblem(task) for task in names]
    state = algo.initialize(algo.create_config(config), problems[0], seed=DEFAULT_SEED)
    nts = [_nt(NT_TASK_PRESETS[task]) for task in names]
    state, _ = algo.run_generation_multitask(state, problems, nts, aggregation_method='mean')
    assert algo._cached_fitnesses is not None and algo._current_generation == 1
    if hand_set:
        algo.multitask_config = dataclasses.replace(algo.multitask_config, **hand_set)
    with pytest.raises(error, match=message):
        algo.run_generation_multitask(state, problems, nts, aggregation_method=aggregation_method)
    assert algo._current_generation == 2, "the inner generation ran"
    assert (algo._neuromod_true['selection_modulation_mode'],
            algo._neuromod_true['selection_fitness_mode']) == ('gating_only', 'bce'), (
        "the modes this call scores with")
    assert algo._cached_fitnesses is None, (
        "no fitness is recorded for a generation that selected nothing")
    assert algo._last_fitnesses is None
    assert algo._last_eval is None
    extended = algo.get_extended_metrics()
    assert np.isnan([extended.fitness, extended.min_fitness, extended.max_fitness,
                     extended.std_fitness]).all()
    for problem, nt in zip(problems, nts):
        _same_program(algo.evaluate_with_neurotransmitter(nt, problem.inputs, problem.targets),
                      _evaluate(algo, nt, problem.inputs, problem.targets, 'gating_only', 'bce'))


# The list form's two flags

@pytest.mark.parametrize('end', LIST_FORM_ENDS)
def test_the_list_form_restores_both_flags(end, monkeypatch):
    """The list form sets _multitask_inner_generation and _multitask_structure_only around its inner
    generation and restores both however it ends (LIST_FORM_ENDS), so the next single-task
    generation on the same instance is scored, told and recorded (a _multitask_structure_only left
    set stops it before its forward: no tell, no fitness, no champion) and compacts its arrays (a
    _multitask_inner_generation left set keeps them on the ceiling; without a multitask block
    nothing else blocks the compaction)."""
    algo, problems, state = _list_form_engine(_list_form_config('hidden_only_no_block'))
    if end == 'returned':
        state, _ = algo.run_generation_multitask(state, problems, _vectors(),
                                                 aggregation_method='mean')
    elif end == 'raised_after_its_inner_generation':
        with pytest.raises(ValueError, match='Unknown aggregation method'):
            algo.run_generation_multitask(state, problems, _vectors(), aggregation_method='median')
        assert algo._current_generation == 1, "the inner generation ran"
    else:
        flags: List[Tuple[bool, bool]] = []

        def fail(*args: Any, **kwargs: Any) -> Any:
            flags.append((algo._multitask_inner_generation, algo._multitask_structure_only))
            raise RuntimeError('the inner generation fails')

        # the structure-only stop's last step, over the method on this instance
        algo._finish_generation = fail
        try:
            with pytest.raises(RuntimeError, match='the inner generation fails'):
                algo.run_generation_multitask(state, problems, _vectors(),
                                              aggregation_method='mean')
        finally:
            del algo._finish_generation
        assert flags == [(True, True)], "it raised inside the inner generation, under both flags"
    assert (algo._multitask_inner_generation, algo._multitask_structure_only) == (False, False)
    told: List[Any] = []
    tell = algo._compiled_tell

    def spy(tell_state: Any, fitnesses: Any) -> Any:
        told.append(fitnesses)
        return tell(tell_state, fitnesses)

    monkeypatch.setattr(algo, '_compiled_tell', spy)
    algo.run_generation(state, problems[0])
    assert len(told) == 1, "the single-task generation tells NEAT"
    assert algo._last_eval is not None and algo._cached_fitnesses is not None
    assert algo._cached_active_indices is not None, "the single-task generation compacts its arrays"
