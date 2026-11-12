"""evaluate_with_neurotransmitter and the list form of run_generation_multitask after a compacted
single-task generation, and the call after an STP-scaled one and after one without true
neuromodulation.

A recurrent generation under true neuromodulation (forward mode HYBRID_SPARSE_HH; the feed-forward
DENSE_ONLY never compacts) gathers its active positions to a bucket no longer than the ceiling and
derives the receptor densities and base gains on that compact axis, while _cached_W1, _cached_W2 and
_cached_sparse_hh stay ceiling-sized. Pairing the two is what these tests guard against:
- in evaluate_with_neurotransmitter, and evaluate_with_custom_nt and
  evaluate_multi_task_true_neuromodulated through it, it gives fitness off by up to 0.497 where the
  bucket equals the ceiling, a broadcasting error where it is smaller;
- in the list form of run_generation_multitask without a multitask block, or with a hand-set
  MultiTaskConfig whose enabled is False, an inner generation that compacted would pair them in the
  fitness it tells NEAT, by the same amounts.
A generation under STP scaling (dynamic_functions 'stp_inspired') evaluates W1 times the STP
depression, caches it as _cached_eval_W1 and derives Option A's receptors from it, while _cached_W1
stays unscaled; the call reads the _cached_eval_* arrays whenever a generation set them.

The references. For the call, the loop's own fitness of the same genomes (_cached_fitnesses) under
the configured modulation mode and the loop's single-task vector: the configurations below (buckets
at the ceiling and below it), Option B with self connections, the self-connection query and the STP
cases. Under branch gating, which the loop does not apply, the routed evaluator on the arrays the
generation evaluated at those positions' grid coordinates, and on the ceiling arrays with receptors
derived there by the loop's formula for each receptor source.
For the list form, the call form on the same state, whose configuration blocks compaction. The
controls (a generation whose union is the whole grid, a feed-forward one, a blocked generation after
a compacted one) evaluate the ceiling arrays. Which seed gives a union of the whole grid moves with
the PRNG default, so the whole-grid control runs seeds until one does instead of assuming one.

How the call and the loop compare: they are two programs, bit for bit on the CPU under
single-threaded Eigen, where they were measured so (an Apple M4 Max, natively and as x86-64 code
under Rosetta 2); otherwise (multi-threaded Eigen, a GPU) as two programs are compared (conftest's
assert_close_across_programs), over the genomes that one ulp of their weights moves by at most
ULP_SENSITIVE in fitness, at least a quarter of the population (the floor of
test_substrate_recurrent_neuromodulation.py).
"""

import dataclasses
import functools
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    GPU_TRAJECTORY_RTOL,
    XORProblem,
    assert_close_across_programs,
    cpu_multi_threaded_eigen,
    create_base_config,
    create_config_with_neuromodulation,
    exact_backend,
)
from emr_hyperneat import emrhyperneat as engine

#: The loop's single-task vector for four NT types (run_generation_verbose)
DEFAULT_NT = (0.5, 0.5, 0.5, 1.0)
#: A vector whose fourth channel is below 1, so the output inversion is not the identity
CUSTOM_NT = (0.9, 0.1, 0.7, 0.6)
#: A genome whose fitness one ulp of its weights moves by more than this settles chaotically, and
#: two programs may give it any fitness (test_substrate_recurrent_neuromodulation.py)
ULP_SENSITIVE = 1e-4

# name -> the case: depth, population, seed, recurrence preset, and the settings on top. What
# pairing the ceiling weights with the compact receptors gives is read from each case's generation
# (_paired), since which seed gives which moves with the substrate: a wrong value where the bucket
# is the ceiling, a broadcasting error where it is below it; the cases cover both
# (test_the_cases_cover_both_pairings). Seed 2 is the first seed whose generation-0 union lands in
# 65..83 of the 84 positions at depth 2, under jax's default PRNG (jax_threefry_partitionable on).
CASES: Dict[str, Dict[str, Any]] = {
    'd2_hidden_only_pop50_s42': dict(
        depth=2, pop=50, seed=42, preset='hidden_only'
    ),
    'd2_full_recurrent_pop50_s2': dict(
        depth=2, pop=50, seed=2, preset='full_recurrent'
    ),
    # The drive-once settling rule
    'd2_drive_once_pop50_s2': dict(
        depth=2,
        pop=50,
        seed=2,
        preset='hidden_only',
        recurrence=dict(settling_rule='drive_once', hh_cache_enabled=False),
        dynamic_functions=dict(hh_activation_mode='initial_only'),
    ),
    # Option B: the receptors from dedicated CPPN outputs at the self-connection
    'd2_option_b_pop50_s2': dict(
        depth=2,
        pop=50,
        seed=2,
        preset='hidden_only',
        nm=dict(receptor_from_weight=False),
    ),
    # Option B with self connections in the h->h set
    'd2_option_b_with_self_pop20_s42': dict(
        depth=2,
        pop=20,
        seed=42,
        preset='with_self',
        nm=dict(receptor_from_weight=False),
    ),
    # Option A from the self-connection query's weight output (the prototype's source)
    'd2_self_query_pop20_s42': dict(
        depth=2,
        pop=20,
        seed=42,
        preset='hidden_only',
        nm=dict(use_self_connection_query=True),
    ),
    'd2_steep_gate_gating_pop50_s2': dict(
        depth=2,
        pop=50,
        seed=2,
        preset='hidden_only',
        nm=dict(node_rule='steep_gate', modulation_mode='gating_only'),
    ),
    'd2_streaming_pop50_s42': dict(
        depth=2, pop=50, seed=42, preset='hidden_only', streaming=True
    ),
    'd2_hidden_only_pop20_s42': dict(
        depth=2, pop=20, seed=42, preset='hidden_only'
    ),
    'd3_hidden_only_pop50_s42': dict(
        depth=3, pop=50, seed=42, preset='hidden_only'
    ),
    # The union without padding (execution.compaction_bucket 'exact')
    'd2_exact_bucket_pop50_s42': dict(
        depth=2,
        pop=50,
        seed=42,
        preset='hidden_only',
        execution=dict(compaction_bucket='exact'),
    ),
}
#: The seeds the whole-grid control runs, in order, until a generation's union is the whole grid
UNION_SEEDS = tuple(range(8))
# Generations that do not compact: a union of the whole grid (the first of UNION_SEEDS that gives
# one); feed-forward
CONTROLS: Dict[str, Dict[str, Any]] = {
    'ctrl_d2_union_full_pop50': dict(depth=2, pop=50, seeds=UNION_SEEDS, preset='hidden_only'),
    'ctrl_d2_feedforward_pop50_s42': dict(depth=2, pop=50, seed=42, preset='feedforward'),
}
#: STP scaling on the modular layers with one activation on every node, which the call can reproduce
#: (weight_interpretation's 'stp_inspired' adds per-node activations, which the call does not apply)
STP = dict(mode='modular', modular=dict(activation_method='none', scaling_method='stp_inspired'))
#: Feed-forward, so they never compact: the scaling alone sets _cached_eval_W1 apart from
#: _cached_W1. Option A takes its receptors from the scaled weight, Option B from the
#: self-connection query.
STP_CASES: Dict[str, Dict[str, Any]] = {
    'stp_d2_feedforward_pop50_s42': dict(
        depth=2, pop=50, seed=42, preset='feedforward', dynamic_functions=STP
    ),
    'stp_d2_option_b_feedforward_pop50_s42': dict(
        depth=2,
        pop=50,
        seed=42,
        preset='feedforward',
        dynamic_functions=STP,
        nm=dict(receptor_from_weight=False),
    ),
}
#: The case the other tests build on: a generation that compacts
BASE = 'd2_hidden_only_pop50_s42'


def _emr(config: Dict[str, Any]) -> Dict[str, Any]:
    return config['algorithm_params']['emrhyperneat']['emr_hyperneat']


def _config(case: Dict[str, Any]) -> Dict[str, Any]:
    """True neuromodulation with four NT types (Option A from the weight by default) on XOR."""
    config = create_config_with_neuromodulation(
        'true_neuromodulation_4nt',
        recurrence_preset=case['preset'],
        max_depth=case['depth'],
        population_size=case['pop'],
        **case.get('nm', {}),
    )
    emr = _emr(config)
    emr['recurrence'].update(case.get('recurrence', {}))
    emr.setdefault('dynamic_functions', {}).update(case.get('dynamic_functions', {}))
    emr.setdefault('execution', {}).update(case.get('execution', {}))
    if case.get('streaming'):
        emr['enable_streaming'] = True
        emr['population_chunk_size'] = 25
    return config


def _initialized(case: Dict[str, Any]) -> Tuple[Any, XORProblem, Any]:
    algo = engine.EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(_config(case)), problem, seed=case['seed'])
    return algo, problem, state


def _union_is_the_grid(algo: Any) -> bool:
    """Whether the last generation's union of active positions (compaction's
    find_active_positions, over the ceiling-sized masks) is the whole grid, so that it did not
    compact."""
    return bool(np.asarray(algo._cached_masks_A, dtype=bool).any(axis=0).all())


class Run(NamedTuple):
    """One single-task generation and what the loop scored."""

    algo: Any
    inputs: jnp.ndarray
    targets: jnp.ndarray
    loop: np.ndarray  # the generation's fitness, _cached_fitnesses
    mode: str  # neuromodulation.modulation_mode


@functools.lru_cache(maxsize=None)
def _run(name: str) -> Run:
    """The case's generation; for a case with seeds, the first seed's whose union is the whole
    grid."""
    case = {**CASES, **CONTROLS, **STP_CASES}[name]
    for seed in case.get('seeds', (case.get('seed'),)):
        algo, problem, state = _initialized({**case, 'seed': seed})
        algo.run_generation(state, problem)
        if 'seeds' not in case or _union_is_the_grid(algo):
            return Run(
                algo,
                algo._cached_inputs,
                algo._cached_targets,
                np.asarray(algo._cached_fitnesses),
                algo.neuromod_config.modulation_mode,
            )
    pytest.fail(
        f"{name}: no seed of {case['seeds']} gives a generation whose union is the whole grid"
    )


def _paired(name: str) -> str:
    """What pairing the ceiling weights with the case's compact receptors gives, read from its
    generation: 'silent' where the bucket is the ceiling (a wrong value), 'raised' where it is
    below it (a broadcasting error)."""
    algo = _run(name).algo
    assert algo._cached_active_indices is not None, f"{name}: the generation did not compact"
    return 'raised' if algo._cached_eval_W1.shape[-1] < algo._cached_W1.shape[-1] else 'silent'


def _nt(vector: Tuple[float, ...]) -> jnp.ndarray:
    return jnp.asarray(vector, dtype=jnp.float32)


def _evaluated_arrays(algo: Any) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
    """W1, W2 and the h->h connections the generation evaluated, _cached_eval_*: compact after a
    compaction, W1 scaled under STP scaling, otherwise the ceiling arrays themselves."""
    return algo._cached_eval_W1, algo._cached_eval_W2, algo._cached_eval_sparse_hh


def _evaluate(
    algo: Any,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    hh: Any,
    nt: Tuple[float, ...],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    mode: str = 'full',
    receptors: Optional[jnp.ndarray] = None,
    gains: Optional[jnp.ndarray] = None,
    branch_mode: str = 'none',
    positions: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """The site's evaluator (tanh nodes, MSE) on the given arrays, the cached receptors by
    default."""
    has_hh = hh is not None and algo._forward_mode == engine.ForwardPassMode.HYBRID_SPARSE_HH
    evaluator = algo._create_eval_population_true_neuromodulated(
        has_hh=has_hh,
        activate_time=algo.activate_time if has_hh else 0,
        hidden_act_fn=jnp.tanh,
        branch_mode=branch_mode,
        positions=positions,
    )
    rd = algo._neuromod_true['receptor_densities'] if receptors is None else receptors
    bg = algo._neuromod_true['base_gains'] if gains is None else gains
    if has_hh:
        fitness = evaluator(
            W1,
            W2,
            rd,
            bg,
            _nt(nt),
            inputs,
            targets,
            hh.from_indices,
            hh.to_indices,
            hh.weights,
            hh.valid_mask,
            'mse',
            mode,
            None,
        )
    else:
        fitness = evaluator(W1, W2, rd, bg, _nt(nt), inputs, targets, 'mse', mode, None)
    return np.asarray(fitness)


def _ceiling_receptors(algo: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The receptor densities and base gains of every grid position, from the ceiling-sized
    caches by the loop's formula for the configured source: Option A from the first input weight
    or from the self-connection query's weight output
    (neuromodulation.use_self_connection_query), or Option B from the dedicated CPPN outputs at
    the self-connection. Not for an STP-scaled generation, whose Option A weight is the scaled
    one."""
    cfg = algo.neuromod_config
    assert getattr(algo, '_df_scaling_method', 'none') == 'none'
    if cfg.receptor_from_weight:
        if getattr(cfg, 'use_self_connection_query', False):
            weight = algo._cached_self_conn[:, :, 0]
        else:
            weight = algo._cached_W1[:, 0, :] / algo.max_weight
        receptors = (
            engine.derive_receptor_from_weight(
                weight=weight, method=cfg.receptor_derivation, num_nt_types=cfg.num_nt_types
            )
            * 2.0
            - 1.0
        )
        return receptors, jnp.abs(weight) * 0.5 + 0.5
    base = algo._cppn_output_indices['receptor']
    query = algo._cached_self_conn
    receptors = jnp.tanh(jnp.clip(query[:, :, base : base + cfg.num_nt_types], -10.0, 10.0))
    gains = jnp.abs(jnp.clip(query[:, :, 0], -10.0, 10.0)) * 0.5 + 0.5
    return jnp.where(jnp.isnan(receptors), 0.0, receptors), jnp.where(jnp.isnan(gains), 0.75, gains)


def _branch_positions(algo: Any) -> jnp.ndarray:
    """The grid coordinates of the evaluated positions: the whole grid, or after a compaction the
    active positions' coordinates padded with zeros to the bucket."""
    grid = engine.get_hierarchical_grid(algo.max_depth).all_positions
    indices = algo._cached_active_indices
    if indices is None:
        return grid
    n, length = len(indices), algo._cached_eval_W1.shape[-1]
    return jnp.concatenate(
        [grid[np.asarray(indices)], jnp.zeros((length - n, grid.shape[1]), grid.dtype)], axis=0
    )


# Comparisons


def _bit_for_bit() -> bool:
    """Where the site and the loop were measured bit for bit: the CPU under single-threaded
    Eigen."""
    return exact_backend() and not cpu_multi_threaded_eigen()


def _same_program(actual: Any, desired: Any) -> None:
    """One evaluator on the same operands: bit for bit on the CPU, to GPU_TRAJECTORY_RTOL off it."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    if exact_backend():
        np.testing.assert_array_equal(actual, desired)
    else:
        np.testing.assert_allclose(actual, desired, rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


def _ulp_sensitive(run: Run, draws: int = 4) -> np.ndarray:
    """The genomes whose fitness moves by more than ULP_SENSITIVE, or becomes non-finite, when every
    entry of their evaluated weights moves by one ulp (random signs, ``draws`` draws), under the
    loop's vector.
    """
    algo = run.algo
    W1, W2, hh = _evaluated_arrays(algo)
    base = _evaluate(algo, W1, W2, hh, DEFAULT_NT, run.inputs, run.targets, run.mode)
    rng = np.random.default_rng(0)

    def jitter(array: Any) -> jnp.ndarray:
        array = np.asarray(array)
        signs = rng.choice([-1.0, 1.0], size=array.shape)
        return jnp.asarray((array * (1 + 2.0**-23 * signs)).astype(np.float32))

    response = np.zeros(base.shape)
    for _ in range(draws):
        moved = None if hh is None else dataclasses.replace(hh, weights=jitter(hh.weights))
        fitness = _evaluate(
            algo, jitter(W1), jitter(W2), moved, DEFAULT_NT, run.inputs, run.targets, run.mode
        )
        response = np.maximum(response, np.abs(fitness - base))
    return ~np.isfinite(response) | (response > ULP_SENSITIVE)


def _like_the_loop(run: Run, fitness: Any, bit_for_bit: Optional[bool] = None) -> None:
    """A fitness of the generation's genomes under the loop's vector against the loop's own: bit for
    bit where they were measured so (_bit_for_bit); otherwise as two programs, over the genomes not
    sensitive to one ulp, at least a quarter of the population.

    Args:
        run: The generation.
        fitness: The fitness to check, before sanitize_fitness (the loop's NaN is -inf).
        bit_for_bit: Which comparison; None decides by the backend.
    """
    fitness = np.asarray(engine.sanitize_fitness(jnp.asarray(fitness))[0])
    if _bit_for_bit() if bit_for_bit is None else bit_for_bit:
        np.testing.assert_array_equal(fitness, run.loop)
        return
    steady = ~(_ulp_sensitive(run) | ~np.isfinite(run.loop))
    floor = max(1, run.loop.size // 4)
    assert (
        int(steady.sum()) >= floor
    ), f"{int(steady.sum())} of {steady.size} genomes steady, fewer than {floor}"
    assert_close_across_programs(fitness[steady], run.loop[steady])


# evaluate_with_neurotransmitter after a compacted generation


@pytest.mark.parametrize('name', sorted(CASES))
def test_the_call_evaluates_what_the_generation_evaluated(name):
    """After a compacted generation the call evaluates the arrays its receptors sit on: under the
    configured modulation mode and the loop's vector it gives the loop's fitness of every genome,
    where pairing the ceiling arrays with those receptors gives another fitness (the bucket at the
    ceiling) or a broadcasting error (the bucket below it; _paired reads which from the run);
    evaluate_with_custom_nt, which forwards to it with mode 'full', too.
    """
    run = _run(name)
    algo = run.algo
    assert algo._cached_active_indices is not None, "the generation compacted"
    bucket = algo._cached_eval_W1.shape[-1]
    assert bucket <= algo._cached_W1.shape[-1]
    assert algo._neuromod_true['receptor_densities'].shape[1] == bucket
    site = algo.evaluate_with_neurotransmitter(
        _nt(DEFAULT_NT), run.inputs, run.targets, modulation_mode=run.mode
    )
    _like_the_loop(run, site)
    if run.mode == 'full':
        _like_the_loop(run, algo.evaluate_with_custom_nt(_nt(DEFAULT_NT), run.inputs, run.targets))


def test_the_cases_cover_both_pairings():
    """The cases cover both outcomes of pairing the ceiling weights with the compact receptors,
    read from their generations: a bucket at the ceiling (a wrong value) and one below it (a
    broadcasting error)."""
    pairings = {name: _paired(name) for name in CASES}
    assert set(pairings.values()) == {'silent', 'raised'}, pairings


def test_the_wrappers_evaluate_the_same_arrays():
    """evaluate_with_custom_nt and evaluate_multi_task_true_neuromodulated (each task under its
    preset NT vector) evaluate the arrays the compacted generation evaluated."""
    run = _run(BASE)
    algo = run.algo
    W1, W2, hh = _evaluated_arrays(algo)
    _same_program(
        algo.evaluate_with_custom_nt(_nt(CUSTOM_NT), run.inputs, run.targets),
        _evaluate(algo, W1, W2, hh, CUSTOM_NT, run.inputs, run.targets),
    )
    and_targets = jnp.asarray([[0.0], [0.0], [0.0], [1.0]], dtype=jnp.float32)
    tasks = {'xor': (run.inputs, run.targets), 'and': (run.inputs, and_targets)}
    expected = [
        _evaluate(
            algo,
            W1,
            W2,
            hh,
            tuple(np.asarray(engine.get_nt_for_task(name, 4)).tolist()),
            inputs,
            targets,
        )
        for name, (inputs, targets) in tasks.items()
    ]
    _same_program(
        algo.evaluate_multi_task_true_neuromodulated(tasks, 'mean'),
        np.asarray(jnp.mean(jnp.stack([jnp.asarray(f) for f in expected], axis=0), axis=0)),
    )


@pytest.mark.parametrize('mode', ['spatial', 'hierarchical'])
@pytest.mark.parametrize(
    'name',
    [
        BASE,
        'd3_hidden_only_pop50_s42',
        'd2_option_b_with_self_pop20_s42',
        'd2_self_query_pop20_s42',
        'ctrl_d2_union_full_pop50',
    ],
)
def test_branch_gating_routes_the_evaluated_positions(name, mode):
    """Under branch gating the routing reads the grid coordinates of the positions the generation
    evaluated: the call equals the routed evaluator on those arrays and coordinates (bit for bit,
    one evaluator), and the same evaluation on the ceiling arrays with receptors derived at every
    grid position by the source's formula (another layout of the same arithmetic, the independent
    check: virtual positions and the order of the compact axis do not enter it). The cases: a
    bucket at the ceiling and below it (depth 3; Option B with self connections; the
    self-connection query), and the uncompacted control. After the compacted generation at the
    ceiling, routing the whole grid's coordinates against the compact receptors puts 5 and 4
    genomes of 50 off by more than 1e-6, by up to 0.022 and 0.012; below the ceiling it raises a
    broadcasting error."""
    run = _run(name)
    algo = run.algo
    site = np.asarray(
        algo.evaluate_with_neurotransmitter(
            _nt(CUSTOM_NT), run.inputs, run.targets, branch_gating_mode=mode
        )
    )
    W1, W2, hh = _evaluated_arrays(algo)
    _same_program(
        site,
        _evaluate(
            algo,
            W1,
            W2,
            hh,
            CUSTOM_NT,
            run.inputs,
            run.targets,
            branch_mode=mode,
            positions=_branch_positions(algo),
        ),
    )
    receptors, gains = _ceiling_receptors(algo)
    ceiling = _evaluate(
        algo,
        algo._cached_W1,
        algo._cached_W2,
        algo._cached_sparse_hh,
        CUSTOM_NT,
        run.inputs,
        run.targets,
        receptors=receptors,
        gains=gains,
        branch_mode=mode,
        positions=engine.get_hierarchical_grid(algo.max_depth).all_positions,
    )
    if exact_backend():
        assert_close_across_programs(site, ceiling)
    else:
        np.testing.assert_allclose(site, ceiling, rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


def test_receptors_on_another_axis_than_the_weights_raise_a_clear_error(monkeypatch):
    """Cached receptors and weights of different lengths (left by different generations) are refused
    with both lengths named, as the run has them, before the evaluator would fail on them with a
    broadcasting error.
    """
    run = _run(BASE)
    algo = run.algo
    for key in ('receptor_densities', 'base_gains'):
        monkeypatch.setitem(algo._neuromod_true, key, algo._neuromod_true[key][:, :-4])
    receptors, weights = (
        algo._neuromod_true['receptor_densities'].shape[1],
        algo._cached_eval_W1.shape[-1],
    )
    assert receptors != weights
    with pytest.raises(
        ValueError,
        match=rf"receptor densities cover {receptors} positions and the weights {weights}:",
    ):
        algo.evaluate_with_neurotransmitter(_nt(DEFAULT_NT), run.inputs, run.targets)


def test_the_comparison_off_the_bit_for_bit_path_sees_the_defect():
    """The comparison as two programs keeps its power: it accepts the evaluation of the arrays
    the generation evaluated and refuses the pairing of the ceiling weights and h->h connections
    with the compact receptors, on BASE or, where its bucket is below the ceiling, the first case
    whose bucket is the ceiling (BASE: 7 genomes of 50 off by more than 1e-6, by up to 0.046)."""
    name = next((case for case in (BASE, *sorted(CASES)) if _paired(case) == 'silent'), None)
    if name is None:
        pytest.fail("no case's bucket is the ceiling (test_the_cases_cover_both_pairings)")
    run = _run(name)
    algo = run.algo
    W1, W2, hh = _evaluated_arrays(algo)
    _like_the_loop(
        run, _evaluate(algo, W1, W2, hh, DEFAULT_NT, run.inputs, run.targets), bit_for_bit=False
    )
    paired = _evaluate(
        algo,
        algo._cached_W1,
        algo._cached_W2,
        algo._cached_sparse_hh,
        DEFAULT_NT,
        run.inputs,
        run.targets,
    )
    with pytest.raises(AssertionError):
        _like_the_loop(run, paired, bit_for_bit=False)


# evaluate_with_neurotransmitter after an STP-scaled generation


@pytest.mark.parametrize('name', sorted(STP_CASES))
def test_an_stp_scaled_generation_is_evaluated_on_its_scaled_weights(name):
    """After an STP-scaled generation that did not compact, the call evaluates the scaled W1 the
    generation evaluated (_cached_eval_W1) with the receptors it derived: the loop's fitness. The
    unscaled _cached_W1 gives another (Option A, whose receptors come from the scaled weight: 48
    genomes of 50 off by more than 1e-6, by up to 0.173; Option B: 44 of 50, by up to 0.127)."""
    run = _run(name)
    algo = run.algo
    assert algo._cached_active_indices is None, "the generation did not compact"
    assert not np.array_equal(
        np.asarray(algo._cached_eval_W1), np.asarray(algo._cached_W1)
    ), "STP scaled W1"
    site = algo.evaluate_with_neurotransmitter(
        _nt(DEFAULT_NT), run.inputs, run.targets, modulation_mode=run.mode
    )
    _like_the_loop(run, site)


def test_a_route_without_evaluation_arrays_leaves_the_ceiling_ones(monkeypatch):
    """Without the evaluation arrays (removed here: every generation of the package sets them
    with _cached_W1) the call evaluates the ceiling-sized _cached_W1, _cached_W2 and
    _cached_sparse_hh: on the STP generation, the unscaled W1."""
    run = _run('stp_d2_feedforward_pop50_s42')
    algo = run.algo
    for name in ('_cached_eval_W1', '_cached_eval_W2', '_cached_eval_sparse_hh'):
        monkeypatch.setattr(algo, name, None)
    _same_program(
        algo.evaluate_with_neurotransmitter(_nt(CUSTOM_NT), run.inputs, run.targets),
        _evaluate(
            algo,
            algo._cached_W1,
            algo._cached_W2,
            algo._cached_sparse_hh,
            CUSTOM_NT,
            run.inputs,
            run.targets,
        ),
    )


# The controls


@pytest.mark.parametrize('name', sorted(CONTROLS))
def test_an_uncompacted_generation_is_evaluated_as_before(name):
    """Without a compaction or STP scaling the generation evaluated the ceiling arrays
    themselves, and the call evaluates them with the receptors: equal to that evaluation (one
    evaluator) and to the loop's fitness. The whole-grid control is the run whose union is the
    whole grid, whatever the seed."""
    run = _run(name)
    algo = run.algo
    assert algo._cached_active_indices is None, "the generation did not compact"
    if 'seeds' in CONTROLS[name]:
        assert _union_is_the_grid(algo), "its union is the whole grid"
    ceiling_arrays = (algo._cached_W1, algo._cached_W2, algo._cached_sparse_hh)
    assert all(
        evaluated is ceiling for evaluated, ceiling in zip(_evaluated_arrays(algo), ceiling_arrays)
    )
    site = algo.evaluate_with_neurotransmitter(
        _nt(DEFAULT_NT), run.inputs, run.targets, modulation_mode=run.mode
    )
    _same_program(
        site, _evaluate(algo, *ceiling_arrays, DEFAULT_NT, run.inputs, run.targets, run.mode)
    )
    _like_the_loop(run, site)


def test_a_blocked_generation_after_a_compacted_one_evaluates_the_ceiling():
    """A generation whose compaction is blocked (a multitask block set by hand) clears the
    previous generation's active indices before its compaction decision, so the call then
    evaluates its ceiling arrays, which the loop evaluated too."""
    algo, problem, state = _initialized(CASES[BASE])
    state, _ = algo.run_generation(state, problem)
    assert algo._cached_active_indices is not None, "the first generation compacted"
    algo.multitask_config = engine.MultiTaskConfig(enabled=True)
    algo.run_generation_verbose(state, problem, skip_metrics=True)
    assert algo._cached_active_indices is None
    assert algo._neuromod_true['receptor_densities'].shape[1] == algo._cached_W1.shape[-1]
    run = Run(
        algo,
        algo._cached_inputs,
        algo._cached_targets,
        np.asarray(algo._cached_fitnesses),
        algo.neuromod_config.modulation_mode,
    )
    _like_the_loop(
        run, algo.evaluate_with_neurotransmitter(_nt(DEFAULT_NT), run.inputs, run.targets)
    )


@pytest.mark.parametrize('neuromodulation', ['none', 'static_gating'])
def test_the_call_after_a_generation_without_true_neuromodulation_says_so(neuromodulation):
    """After a generation without true neuromodulation (none, or another mode) the receptor
    densities are None, and the call raises its own error, for a vector and for a task name
    alike: its guard reads the densities (_neuromod_true is a dict from construction on) before
    the call would fail on the settings of no neuromodulation (AttributeError) or, under another
    mode, inside the evaluator on the densities (TypeError: 'NoneType' object is not
    subscriptable)."""
    if neuromodulation == 'none':
        config = create_base_config(max_depth=2, population_size=20)
    else:
        config = create_config_with_neuromodulation(
            neuromodulation, recurrence_preset='hidden_only', max_depth=2, population_size=20
        )
    algo = engine.EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=42)
    algo.run_generation(state, problem)
    assert algo._neuromod_true['receptor_densities'] is None
    for vector in (_nt(DEFAULT_NT), 'xor'):
        with pytest.raises(RuntimeError, match='TRUE neuromodulation data not available'):
            algo.evaluate_with_neurotransmitter(vector, algo._cached_inputs, algo._cached_targets)


# run_generation_multitask's list form


@functools.lru_cache(maxsize=None)
def _told_by_the_call_form(name: str) -> np.ndarray:
    """What the call form tells NEAT on the case's initial state (its configuration blocks
    compaction)."""
    algo, problem, state = _initialized(CASES[name])
    tasks = {
        'xor': (
            np.asarray(problem.inputs, dtype=np.float32),
            np.asarray(problem.targets, dtype=np.float32),
        )
    }
    algo.run_generation_multitask(
        state,
        engine.MultiTaskConfig(
            tasks=tasks, nt_vectors={'xor': DEFAULT_NT}, fitness_aggregation='mean'
        ),
    )
    return np.asarray(algo._last_multitask['aggregated'])


@pytest.mark.parametrize('configured', ['no multitask block', 'MultiTaskConfig(enabled=False)'])
def test_the_list_form_tells_what_the_call_form_tells(configured):
    """The list form with one task under the loop's vector, on a configuration whose single-task
    generation compacts, tells NEAT what the call form tells it on the same state and key: its
    inner generation keeps the arrays on the ceiling, so the receptors share _cached_W1's axis
    (an inner generation that compacted would put 7 genomes of 50 off by more than 1e-6, by up to
    0.046). evaluate_with_neurotransmitter then evaluates the same arrays: the task's fitness."""
    assert (
        _run(BASE).algo._cached_active_indices is not None
    ), "the state's single-task generation compacts"
    algo, problem, state = _initialized(CASES[BASE])
    if configured != 'no multitask block':
        algo.multitask_config = engine.MultiTaskConfig(enabled=False)
    algo.run_generation_multitask(state, [problem], [_nt(DEFAULT_NT)], aggregation_method='mean')
    _same_program(algo._last_multitask['aggregated'], _told_by_the_call_form(BASE))
    assert algo._cached_active_indices is None
    assert algo._neuromod_true['receptor_densities'].shape[1] == algo._cached_W1.shape[-1]
    _same_program(
        algo.evaluate_with_neurotransmitter(
            _nt(DEFAULT_NT), algo._cached_inputs, algo._cached_targets
        ),
        algo._last_multitask['per_task'][0],
    )
