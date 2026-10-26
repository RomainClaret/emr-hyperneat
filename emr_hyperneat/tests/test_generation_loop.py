"""The plain feed-forward generation as one traceable function, and HMR's GPU while_loop over it.

8a: run_generation_verbose runs the configurations of HMR's GPU loop's scope (a one-output CPPN, no
hidden-to-hidden connections, activation and aggregation 'disabled' or 'global') through
`_feedforward_generation`, which reads the configuration and writes nothing to the engine. Run op by
op it must equal the general path, which every other configuration runs, bit for bit on XLA:CPU:
every fitness array, the weights, masks and records, the metrics and the final key. The same holds
for its parts: the masks are traceable and equal the plain functions', the query keeps its outputs
on the device with the numpy path's values, and the device NaN count equals sanitize_fitness's.

8b: execution.generation_loop 'while_loop' compiles that function inside lax.while_loop (see the
section below and docs/configuration.md): performance only, up to rounding.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (
    GPU_TRAJECTORY_RTOL,
    RECURRENCE_PRESETS,
    EMRHyperNEAT,
    XORProblem,
    assert_trajectories_agree,
    create_base_config,
    create_config_with_aggregation,
    create_config_with_dynamic_functions,
    create_config_with_geometry_seeding,
    create_config_with_leo,
    create_config_with_neuromodulation,
    create_config_with_recurrence,
    exact_backend,
)
from emr_hyperneat.emrhyperneat_base import (
    batch_query_population_multi_source_chunked,
    compute_hierarchical_variances_batch,
    compute_subdivision_masks_batch,
    get_hierarchical_grid,
)
from emr_hyperneat.emrhyperneat import (
    query_population_on_device,
    sanitize_fitness,
    sanitize_fitness_on_device,
    substrate_masks,
    substrate_query_rows,
)
from hmr_reference import hmr_value, live  # type: ignore[import-not-found]

SEED = 42
GENERATIONS = 3


def _plain(population: int = 30, depth: int = 3) -> dict:
    config = create_base_config(max_depth=depth, population_size=population)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = dict(
        RECURRENCE_PRESETS['feedforward']
    )
    return config


def _with(config: dict, **settings) -> dict:
    config['algorithm_params']['emrhyperneat']['emr_hyperneat'].update(settings)
    return config


def _with_recurrence(config: dict, **settings) -> dict:
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'].update(settings)
    return config


#: Configurations in the core's scope: each setting the core applies on its own
CORE_CONFIGS = {
    'plain': lambda: _plain(),
    'aggregation_global_max': lambda: create_config_with_aggregation(
        'global_max', max_depth=3, population_size=30
    ),
    'activation_global_sin': lambda: create_config_with_dynamic_functions(
        'global_sin', max_depth=3, population_size=30
    ),
    'eval_chunk_7': lambda: _with_recurrence(_plain(), eval_chunk_size=7),
    'fitness_accuracy': lambda: _with(_plain(), fitness_mode='accuracy'),
    'fitness_rescale': lambda: _with(_plain(), fitness_rescale_variance=0.25),
    'geometry_locality': lambda: _with(
        create_config_with_geometry_seeding('default', max_depth=3, population_size=30),
        locality_radius=0.5,
    ),
    'hmr_compatible': lambda: _with(
        _plain(), variance_rule='legacy', extra_randkey_split=True, initial_depth=1
    ),
    'population_150': lambda: _plain(population=150),
}

#: Configurations the core does not take, and the blocker each must name
GENERAL_CONFIGS = {
    'hidden_only': (
        lambda: create_config_with_recurrence(
            'hidden_only', max_depth=3, population_size=30, hh_cache_enabled=False
        ),
        'hidden-to-hidden connections',
    ),
    'activation_cppn_output': (
        lambda: create_config_with_dynamic_functions(
            'cppn_output_4', max_depth=3, population_size=30
        ),
        "dynamic_functions.mode 'cppn_output'",
    ),
    'activation_weight_interpretation': (
        lambda: create_config_with_dynamic_functions(
            'weight_interp_sign', max_depth=3, population_size=30
        ),
        "dynamic_functions.mode 'weight_interpretation'",
    ),
    'aggregation_weight_interpretation': (
        lambda: create_config_with_aggregation('weight_interp', max_depth=3, population_size=30),
        "aggregation.mode 'weight_interpretation'",
    ),
    'leo': (lambda: create_config_with_leo('default', max_depth=3, population_size=30), 'LEO'),
    'neuromodulation': (
        lambda: create_config_with_neuromodulation(
            'static_gating', recurrence_preset='feedforward', max_depth=3, population_size=30
        ),
        'neuromodulation',
    ),
    'streaming': (
        lambda: _with(_plain(), execution={'mode': 'streaming'}),
        "execution.mode 'streaming'",
    ),
}


def _engine(config: dict, general: bool = False):
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(
        algorithm.create_config(copy.deepcopy(config)), XORProblem(), seed=SEED
    )
    if general:
        # The general path, as every configuration outside the core's scope runs it
        algorithm._feedforward_core_blockers = lambda: ['forced general path']
    return algorithm, state


def _run(config: dict, general: bool = False, generations: int = GENERATIONS):
    """Each generation's records and metrics, and the final state."""
    algorithm, state = _engine(config, general)
    records = []
    for _ in range(generations):
        state, metrics = algorithm.run_generation(state, XORProblem())
        records.append(
            {
                'fitnesses': np.asarray(algorithm._cached_fitnesses),
                'W1': np.asarray(algorithm._cached_W1),
                'W2': np.asarray(algorithm._cached_W2),
                'masks': np.asarray(algorithm._cached_masks_A),
                'reached': list(algorithm._last_reached_per_level),
                'stats': (
                    metrics.best_fitness,
                    metrics.mean_fitness,
                    metrics.min_fitness,
                    metrics.std_fitness,
                ),
                'species': (metrics.num_species, list(metrics.species_sizes)),
                'custom_keys': sorted(k for k in metrics.custom_metrics if k != 'step_timings'),
                'champion': algorithm._last_eval['champion_index'],
                'none_records': tuple(
                    getattr(algorithm, name) is None
                    for name in (
                        '_cached_self_conn',
                        '_cached_sparse_hh',
                        '_cached_act_indices',
                        '_cached_hh_agg_indices',
                        '_cached_ih_agg_indices',
                        '_cppn_neuromodulation',
                    )
                ),
            }
        )
    return algorithm, state, records


def _arrays_agree(a: np.ndarray, b: np.ndarray, what: str) -> None:
    if exact_backend() or a.dtype == bool:
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), what
    else:
        np.testing.assert_allclose(a, b, rtol=GPU_TRAJECTORY_RTOL, atol=0.0, err_msg=what)


def _records_agree(core: list, general: list) -> None:
    for generation, (c, g) in enumerate(zip(core, general)):
        for name in ('fitnesses', 'W1', 'W2', 'masks'):
            _arrays_agree(c[name], g[name], f'generation {generation}: {name}')
        for name in ('reached', 'species', 'custom_keys', 'champion', 'none_records'):
            assert c[name] == g[name], f'generation {generation}: {name}: {c[name]} vs {g[name]}'
        if exact_backend():
            assert c['stats'] == g['stats'], f'generation {generation}: statistics'
        else:
            np.testing.assert_allclose(c['stats'], g['stats'], rtol=GPU_TRAJECTORY_RTOL)


class TestFeedforwardCore:
    @pytest.mark.parametrize('name', sorted(CORE_CONFIGS))
    def test_the_default_path_takes_the_core(self, name):
        algorithm, state = _engine(CORE_CONFIGS[name]())
        assert algorithm._feedforward_core_blockers() == []
        taken = []
        original = algorithm._run_generation_feedforward

        def spy(*args, **kwargs):
            taken.append(True)
            return original(*args, **kwargs)

        algorithm._run_generation_feedforward = spy
        algorithm.run_generation(state, XORProblem())
        assert taken == [True]

    @pytest.mark.parametrize('name', sorted(GENERAL_CONFIGS))
    def test_other_configurations_take_the_general_path(self, name):
        make, blocker = GENERAL_CONFIGS[name]
        algorithm, state = _engine(make())
        assert any(blocker in reason for reason in algorithm._feedforward_core_blockers()), \
            algorithm._feedforward_core_blockers()
        algorithm._run_generation_feedforward = lambda *args, **kwargs: pytest.fail('took the core')
        algorithm.run_generation(state, XORProblem())

    def test_verbose_takes_the_general_path(self, capsys):
        config = _plain()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['verbose'] = True
        algorithm, state = _engine(config)
        algorithm._run_generation_feedforward = lambda *args, **kwargs: pytest.fail('took the core')
        algorithm.run_generation(state, XORProblem())
        assert '[Feedforward Diag]' in capsys.readouterr().out

    @pytest.mark.parametrize('name', sorted(CORE_CONFIGS))
    def test_the_core_equals_the_general_path(self, name):
        _, core_state, core = _run(CORE_CONFIGS[name]())
        _, general_state, general = _run(CORE_CONFIGS[name](), general=True)
        _records_agree(core, general)
        # The key is integer arithmetic: equal on every backend
        np.testing.assert_array_equal(
            np.asarray(core_state.randkey), np.asarray(general_state.randkey)
        )

    def test_a_one_ulp_change_fails_the_comparison(self):
        """Negative control: the comparison above sees one float32 ulp in one genome."""
        if not exact_backend():
            pytest.skip('the GPU comparison is to rounding by design')
        _, _, general = _run(_plain(), general=True, generations=1)
        algorithm, state = _engine(_plain())
        rescale = algorithm._rescale_fitness
        algorithm._rescale_fitness = lambda f: rescale(f).at[3].set(jnp.nextafter(f[3], jnp.inf))
        algorithm.run_generation(state, XORProblem())
        nudged = [{**general[0], 'fitnesses': np.asarray(algorithm._cached_fitnesses)}]
        with pytest.raises(AssertionError, match='fitnesses'):
            _records_agree(nudged, general)

    def test_the_core_writes_nothing_to_the_engine(self):
        algorithm, state = _engine(_plain())
        before = {name: id(value) for name, value in vars(algorithm).items()}
        gen = algorithm._feedforward_generation(
            state,
            algorithm._substrate_query_rows(),
            algorithm._cached_inputs,
            algorithm._cached_targets,
        )
        after = {name: id(value) for name, value in vars(algorithm).items()}
        assert before == after
        assert gen.fitnesses.shape == (30,) and int(gen.nan_count) == 0


class TestTraceableParts:

    @pytest.mark.parametrize(
        'depth,initial_depth,legacy', [(2, 0, False), (3, 1, True), (4, 0, True)]
    )
    def test_masks_are_traceable_and_equal_the_plain_functions(self, depth, initial_depth, legacy):
        grid = get_hierarchical_grid(depth)
        weights = jax.random.normal(jax.random.PRNGKey(depth), (12, grid.total_positions)) * 0.3
        eager = substrate_masks(
            weights, grid, 0.03, initial_depth=initial_depth, legacy_level0_variance=legacy
        )
        traced = jax.jit(lambda w: substrate_masks(w, grid, 0.03, initial_depth=initial_depth,
                                                   legacy_level0_variance=legacy))(weights)
        plain = compute_subdivision_masks_batch(
            compute_hierarchical_variances_batch(weights, grid, legacy_level0_variance=legacy),
            0.03,
            grid,
            return_all_masks=False,
            initial_depth=initial_depth,
        )
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(plain))
        np.testing.assert_array_equal(np.asarray(traced), np.asarray(plain))
        # Negative control: a different threshold gives a different mask
        assert not np.array_equal(
            np.asarray(substrate_masks(weights, grid, 0.3)), np.asarray(plain)
        )

    @pytest.mark.parametrize('population,geometry', [(30, False), (150, False), (150, True)])
    def test_the_device_query_equals_the_numpy_path(self, population, geometry):
        config = (
            create_config_with_geometry_seeding('default', max_depth=3, population_size=population)
            if geometry
            else _plain(population=population)
        )
        algorithm, state = _engine(config)
        cppns = algorithm._compiled_transform_batch(state, algorithm._compiled_ask(state))
        positions = get_hierarchical_grid(3).all_positions
        for coords, outgoing in (
            (algorithm._cached_input_coords, True),
            (algorithm._cached_output_coords, False),
        ):
            rows = substrate_query_rows(coords, positions, outgoing, geometry)
            on_device = query_population_on_device(
                state, cppns, rows, algorithm._jitted_cppn_forward
            )
            through_numpy = batch_query_population_multi_source_chunked(
                state, cppns, coords, positions, outgoing, algorithm._jitted_cppn_forward,
                geometry_seeding_enabled=geometry)
            assert isinstance(on_device, jax.Array)
            # Data movement only: equal on every backend
            assert np.asarray(on_device).tobytes() == np.asarray(through_numpy).tobytes()

    def test_the_device_nan_count(self):
        clean = jnp.array([0.5, 0.25, 0.75], dtype=jnp.float32)
        dirty = jnp.array([0.5, jnp.nan, 0.75, jnp.nan], dtype=jnp.float32)
        for fitnesses in (clean, dirty):
            expected, count = sanitize_fitness(fitnesses)
            values, device_count = jax.jit(sanitize_fitness_on_device)(fitnesses)
            assert np.asarray(values).tobytes() == np.asarray(expected).tobytes()
            assert int(device_count) == count and device_count.dtype == jnp.int32


# 8b: execution.generation_loop 'while_loop' The loop compiles _feedforward_generation inside
# lax.while_loop. Its criterion (the design note's section 4): bit for bit the same arithmetic as
# the core compiled once per generation; against the default Python loop, fitness to rounding (XLA
# fuses the compiled evaluation, one float32 ulp on CPU in the probes), the same generation count
# and stop generation, and the same final key on every backend (threefry is integer arithmetic).

LOOP_GENERATIONS = 4
NO_STOP = 2.0  # above any 1-MSE or accuracy fitness: every run takes max_generations

#: Configurations the loop runs, each a setting it applies
LOOP_CONFIGS = {name: CORE_CONFIGS[name] for name in (
    'plain', 'aggregation_global_max', 'activation_global_sin', 'eval_chunk_7', 'fitness_accuracy',
    'fitness_rescale', 'geometry_locality', 'hmr_compatible', 'population_150')}


def _looped(config: dict) -> dict:
    config = copy.deepcopy(config)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['execution'] = {**emr.get('execution', {}), 'generation_loop': 'while_loop'}
    return config


def _threshold_run(config: dict, target: float = NO_STOP, generations: int = LOOP_GENERATIONS):
    algorithm, state = _engine(config)
    result = algorithm.run_until_threshold(
        state,
        XORProblem(),
        target_fitness=target,
        max_generations=generations,
        collect_history=True,
    )
    return algorithm, result


def _jit_core_run(config: dict, generations: int = LOOP_GENERATIONS):
    """The core compiled once per generation, in a Python loop: the loop's arithmetic without the
    loop."""
    algorithm, state = _engine(config)
    rows_in, rows_out = algorithm._substrate_query_rows()

    @jax.jit
    def step(current, inputs, targets):
        gen = algorithm._feedforward_generation(current, (rows_in, rows_out), inputs, targets)
        return gen.new_state, jnp.max(gen.fitnesses)

    history = []
    for _ in range(generations):
        state, best = step(state, algorithm._cached_inputs, algorithm._cached_targets)
        history.append(float(best))
    return history, state


def _histories_agree(loop: list, python: list, what: str) -> None:
    assert len(loop) == len(python), f'{what}: {len(loop)} vs {len(python)} generations'
    if exact_backend():
        np.testing.assert_allclose(loop, python, rtol=1e-6, atol=0.0, err_msg=what)
    else:
        assert_trajectories_agree({'while_loop': loop, 'python': python}, what)


def _keys_equal(a, b) -> bool:
    return np.asarray(a.randkey).tobytes() == np.asarray(b.randkey).tobytes()


class TestGenerationLoopKey:
    def test_python_is_the_default(self):
        algorithm, _ = _engine(_plain())
        assert algorithm.execution_config.generation_loop == 'python'

    @pytest.mark.parametrize('value', ['python', 'while_loop'])
    def test_the_values(self, value):
        config = _plain()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
            'generation_loop': value
        }
        algorithm, _ = _engine(config)
        assert algorithm.execution_config.generation_loop == value

    def test_another_value_raises(self):
        config = _plain()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
            'generation_loop': 'scan'
        }
        with pytest.raises(ValueError, match="generation_loop must be 'python' or 'while_loop'"):
            EMRHyperNEAT().create_config(config)

    def test_the_key_is_an_execution_key(self):
        from emr_hyperneat.execution import (
            EXECUTION_KEYS,
        )
        assert 'generation_loop' in EXECUTION_KEYS


class TestWhileLoopEquality:

    @pytest.mark.parametrize(
        'name', ['plain', 'aggregation_global_max', 'eval_chunk_7', 'hmr_compatible']
    )
    def test_equals_the_core_compiled_per_generation(self, name):
        _, result = _threshold_run(_looped(LOOP_CONFIGS[name]()))
        history, state = _jit_core_run(LOOP_CONFIGS[name]())
        assert (
            result['generation_loop'] == 'while_loop' and result['generations'] == LOOP_GENERATIONS
        )
        if exact_backend():
            assert result['history'] == history
            for leaf_loop, leaf_jit in zip(jax.tree_util.tree_leaves(result['state']),
                                           jax.tree_util.tree_leaves(state)):
                assert np.asarray(leaf_loop).tobytes() == np.asarray(leaf_jit).tobytes()
        else:
            assert_trajectories_agree({'while_loop': result['history'], 'jit_core': history}, name)
        assert _keys_equal(result['state'], state)

    @pytest.mark.parametrize('name', sorted(LOOP_CONFIGS))
    def test_equals_the_python_loop_to_rounding(self, name):
        _, python = _threshold_run(LOOP_CONFIGS[name]())
        algorithm, loop = _threshold_run(_looped(LOOP_CONFIGS[name]()))
        _histories_agree(loop['history'], python['history'], name)
        assert loop['generations'] == python['generations'] == LOOP_GENERATIONS
        assert _keys_equal(loop['state'], python['state']), f'{name}: final randkey differs'
        for key in ('converged', 'solved', 'execution_mode', 'route', 'devices'):
            assert loop[key] == python[key], key
        np.testing.assert_allclose(
            loop['final_fitness'], python['final_fitness'], rtol=GPU_TRAJECTORY_RTOL
        )
        assert loop['sparse_hh_stats'] is None and loop['nan_fitness_count'] == 0
        assert algorithm._current_generation == LOOP_GENERATIONS

    def test_the_stop_generation(self):
        """A target the first generation reaches stops both loops there; the best then stays
        reached."""
        for target in (0.0, 0.5):
            _, python = _threshold_run(_plain(), target=target, generations=10)
            _, loop = _threshold_run(_looped(_plain()), target=target, generations=10)
            assert loop['generations'] == python['generations'], target
            assert loop['converged'] == python['converged'] is True

    def test_zero_generations(self):
        algorithm, loop = _threshold_run(_looped(_plain()), generations=0)
        assert (
            loop['generations'] == 0
            and loop['history'] == []
            and loop['best_fitness'] == float('-inf')
        )
        assert algorithm._current_generation == 0

    def test_the_float32_threshold(self):
        """For a float32 best b, b >= float32_threshold(t) exactly when float(b) >= t: the Python
        loop's rule."""
        from emr_hyperneat.emrhyperneat import (
            float32_threshold,
        )
        for target in (
            0.95,
            0.9,
            0.99,
            0.75,
            1.0,
            0.0,
            -0.1,
            1e-30,
            1e39,
            float('inf'),
            float('-inf'),
        ):
            threshold = float32_threshold(target)
            assert threshold.dtype == np.float32
            nearest = (
                np.float32(target) if abs(target) < 1e38 or np.isinf(target) else np.float32(np.inf)
            )
            for best in (np.nextafter(nearest, np.float32(-np.inf)), nearest,
                         np.nextafter(nearest, np.float32(np.inf))):
                assert bool(best >= threshold) == (float(best) >= target), (target, best)
        # 0.95 is below its float32 rounding: HMR's float32 cast would stop a generation early there
        assert float(np.float32(0.95)) < 0.95 and float32_threshold(0.95) > np.float32(0.95)
        assert np.isnan(float32_threshold(float('nan')))

    def test_nan_fitness_is_counted(self):
        """A NaN target makes every fitness NaN, told as -inf: counted per genome and generation."""
        class NaNTarget(XORProblem):
            def __init__(self):
                super().__init__()
                self.targets = [[0.0], [1.0], [float('nan')], [0.0]]

        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(_looped(_plain())), NaNTarget(), seed=SEED
        )
        result = algorithm.run_until_threshold(
            state, NaNTarget(), target_fitness=NO_STOP, max_generations=2, collect_history=True
        )
        assert result['nan_fitness_count'] == 2 * 30
        assert result['history'] == [float('-inf')] * 2 and result['best_fitness'] == float('-inf')


class TestWhileLoopNegativeControls:
    def test_the_base_key_flag_fails_the_key_check(self):
        """The base class's extra_randkey_split is True on this engine; a loop reading it would
        split every generation and diverge from the second generation on."""
        _, python = _threshold_run(_plain())
        algorithm, state = _engine(_looped(_plain()))
        assert (
            algorithm.extra_randkey_split is True and algorithm._repro_extra_randkey_split is False
        )
        algorithm._repro_extra_randkey_split = algorithm.extra_randkey_split
        loop = algorithm.run_until_threshold(state, XORProblem(), target_fitness=NO_STOP,
                                             max_generations=LOOP_GENERATIONS, collect_history=True)
        assert not _keys_equal(loop['state'], python['state'])
        # the split precedes the first tell
        assert loop['history'][0] == pytest.approx(python['history'][0])
        with pytest.raises(AssertionError):
            _histories_agree(loop['history'], python['history'], 'base key flag')

    def test_a_stale_tell_fails_the_trajectory_check(self):
        """A body that tells a constant fitness, not the generation's, evolves another population.
        """
        _, python = _threshold_run(_plain())
        algorithm, state = _engine(_looped(_plain()))
        tell = algorithm._compiled_tell
        algorithm._compiled_tell = lambda current, fitnesses: tell(
            current, jnp.zeros_like(fitnesses)
        )
        loop = algorithm.run_until_threshold(state, XORProblem(), target_fitness=NO_STOP,
                                             max_generations=LOOP_GENERATIONS, collect_history=True)
        with pytest.raises(AssertionError):
            _histories_agree(loop['history'], python['history'], 'stale tell')


_COMPILES = {'active': False, 'count': 0, 'registered': False}


def _watch_compiles():
    if not _COMPILES['registered']:
        def listener(event, duration, **kwargs):
            if _COMPILES['active'] and event == '/jax/core/compile/backend_compile_duration':
                _COMPILES['count'] += 1
        jax.monitoring.register_event_duration_secs_listener(listener)
        _COMPILES['registered'] = True


def _contains_tracer(value, depth: int = 0) -> bool:
    if isinstance(value, jax.core.Tracer):
        return True
    if depth < 3 and isinstance(value, dict):
        return any(_contains_tracer(item, depth + 1) for item in value.values())
    if depth < 3 and isinstance(value, (list, tuple)):
        return any(_contains_tracer(item, depth + 1) for item in value)
    return False


class TestWhileLoopRuns:
    def test_a_second_call_compiles_nothing(self):
        _watch_compiles()
        algorithm, state = _engine(_looped(_plain()))
        counts = []
        for _ in range(2):
            _COMPILES['count'], _COMPILES['active'] = 0, True
            try:
                result = algorithm.run_until_threshold(state, XORProblem(), target_fitness=NO_STOP,
                                                       max_generations=LOOP_GENERATIONS)
                jax.block_until_ready(result['state'])
            finally:
                _COMPILES['active'] = False
            counts.append(_COMPILES['count'])
        assert counts[0] > 0 and counts[1] == 0, counts
        # Negative control: create_config clears the cached loop, which then compiles again
        state = algorithm.initialize(
            algorithm.create_config(_looped(_plain())), XORProblem(), seed=SEED
        )
        _COMPILES['count'], _COMPILES['active'] = 0, True
        try:
            jax.block_until_ready(
                algorithm.run_until_threshold(
                    state, XORProblem(), target_fitness=NO_STOP, max_generations=LOOP_GENERATIONS
                )['state']
            )
        finally:
            _COMPILES['active'] = False
        assert _COMPILES['count'] > 0

    def test_no_tracer_is_left_and_no_record_kept(self):
        algorithm, state = _engine(_looped(_plain()))
        # A generation of the Python path first: its records must not survive the loop
        algorithm.execution_config.generation_loop = 'python'
        state, _ = algorithm.run_generation(state, XORProblem())
        assert algorithm._cached_fitnesses is not None and algorithm._last_eval is not None
        algorithm.execution_config.generation_loop = 'while_loop'
        result = algorithm.run_until_threshold(state, XORProblem(), target_fitness=NO_STOP,
                                               max_generations=LOOP_GENERATIONS)
        assert not any(_contains_tracer(value) for value in vars(algorithm).values())
        for name in (
            '_cached_fitnesses',
            '_last_fitnesses',
            '_cached_W1',
            '_cached_masks_A',
            '_last_eval',
            '_extended_metrics',
            '_last_reached_per_level',
        ):
            assert getattr(algorithm, name) is None, name
        assert algorithm._current_generation == 1 + LOOP_GENERATIONS
        with pytest.raises(RuntimeError, match='while_loop'):
            algorithm.evaluate_on_data(
                result['state'], np.zeros((4, 3), np.float32), np.zeros((4, 1), np.float32)
            )


def _node_list() -> dict:
    """HMR's node-list model at depth 2 (test_node_list.py's shape: no variance settings)."""
    return {
        'algorithm_params': {
            'emrhyperneat': {
                'population_size': 12,
                'substrate': {
                    'input_coords': [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],
                    'output_coords': [(0.0, 1.0)],
                },
                'emr_hyperneat': {
                    'max_depth': 2,
                    'band_threshold': 0.3,
                    'max_weight': 8.0,
                    'recurrence': {
                        'network_model': 'node_list',
                        'preset': 'full_recurrent',
                        'iteration_level': 2,
                    },
                },
            }
        }
    }


#: Configurations the loop refuses, and the name the refusal must give
REFUSED = {
    'hidden_only': (
        lambda: create_config_with_recurrence(
            'hidden_only', max_depth=3, population_size=30, hh_cache_enabled=False
        ),
        'hidden-to-hidden',
    ),
    'activation_cppn_output': (GENERAL_CONFIGS['activation_cppn_output'][0], 'cppn_output'),
    'activation_weight_interpretation': (
        GENERAL_CONFIGS['activation_weight_interpretation'][0],
        'weight_interpretation',
    ),
    'aggregation_weight_interpretation': (
        GENERAL_CONFIGS['aggregation_weight_interpretation'][0],
        "aggregation.mode 'weight_interpretation'",
    ),
    'leo': (GENERAL_CONFIGS['leo'][0], 'LEO'),
    'neuromodulation': (GENERAL_CONFIGS['neuromodulation'][0], 'neuromodulation'),
    'streaming': (lambda: _with(_plain(), execution={'mode': 'streaming'}), 'streaming'),
    'depth_offload': (
        lambda: _with(_plain(), execution={'mode': 'depth_offload'}),
        'depth_offload',
    ),
    'palette_evolution': (
        lambda: create_config_with_dynamic_functions(
            'cppn_output_4', max_depth=3, population_size=30, palette_evolution={'enabled': True}
        ),
        'palette_evolution',
    ),
}


class TestWhileLoopRefusals:
    @pytest.mark.parametrize('name', sorted(REFUSED))
    def test_refused_and_named(self, name):
        make, named = REFUSED[name]
        algorithm, state = _engine(_looped(make()))
        with pytest.raises(ValueError, match="generation_loop 'while_loop'") as raised:
            algorithm.run_until_threshold(
                state, XORProblem(), target_fitness=NO_STOP, max_generations=2
            )
        assert named in str(raised.value), str(raised.value)
        # Negative control: the same configuration runs on the Python loop
        algorithm, state = _engine(make())
        algorithm.run_until_threshold(
            state, XORProblem(), target_fitness=NO_STOP, max_generations=1
        )

    def test_the_live_palette_strategy(self):
        """A host step between generations: refused and named (with its cppn_output mode)."""
        class Strategy:
            def get_active_palette(self):
                return [0, 4, 5, 13]

        make = lambda: create_config_with_dynamic_functions(
            'cppn_output_4', max_depth=3, population_size=30, live_strategy={'enabled': True}
        )
        algorithm, state = _engine(_looped(make()))
        algorithm.set_palette_strategy(Strategy())
        with pytest.raises(ValueError, match="generation_loop 'while_loop'") as raised:
            algorithm.run_until_threshold(
                state, XORProblem(), target_fitness=NO_STOP, max_generations=2
            )
        assert 'live palette strategy' in str(raised.value) and 'cppn_output' in str(raised.value)
        # Negative control: the Python loop runs it
        algorithm, state = _engine(make())
        algorithm.set_palette_strategy(Strategy())
        algorithm.run_until_threshold(
            state, XORProblem(), target_fitness=NO_STOP, max_generations=1
        )

    def test_the_node_list_model(self):
        """Refused when configured (create_config names the key) and when set after it (the node
        list's guard)."""
        with pytest.raises(ValueError, match="execution.generation_loop 'while_loop'"):
            EMRHyperNEAT().create_config(_looped(_node_list()))
        algorithm, state = _engine(_node_list())
        algorithm.execution_config.generation_loop = 'while_loop'
        with pytest.raises(
            ValueError, match="node_list' cannot apply.*generation_loop 'while_loop'"
        ):
            algorithm.run_until_threshold(
                state, XORProblem(), target_fitness=NO_STOP, max_generations=1
            )
        # The loop's own refusal names the model too (the guard aside)
        assert any('node_list' in reason for reason in algorithm._generation_loop_refusals())
        # Negative control: the Python loop runs it
        algorithm.execution_config.generation_loop = 'python'
        algorithm.run_until_threshold(
            state, XORProblem(), target_fitness=NO_STOP, max_generations=1
        )

    def test_a_multi_device_route_refuses_it(self, monkeypatch):
        config = _looped(_plain())
        config['algorithm_params']['emrhyperneat']['emr_hyperneat'][
            'multi_gpu_strategy'
        ] = 'eval_only_parallel'
        algorithm, state = _engine(config)
        device = jax.devices()[0]
        monkeypatch.setattr(jax, 'devices', lambda *args, **kwargs: [device, device])
        with pytest.raises(ValueError, match="generation_loop 'while_loop' runs one device"):
            algorithm.run_until_threshold(
                state, XORProblem(), target_fitness=NO_STOP, max_generations=2
            )

    def test_verbose_warns(self):
        config = _looped(_plain())
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['verbose'] = True
        algorithm, state = _engine(config)
        with pytest.warns(UserWarning, match='prints nothing per generation'):
            result = algorithm.run_until_threshold(
                state, XORProblem(), target_fitness=NO_STOP, max_generations=2
            )
        assert result['generations'] == 2


# HMR's base engine's GPU loop through the seam (the HMR engine, release 0.1.0). With
# HMR's level-0 rule and pre-ask key split (as the release's frozen-HMR test sets them), HMR's
# while_loop and EMR's give the same generations and final key, and the same best of every
# generation to rounding (bit for bit in the design probe, depth 2, population 60). HMR's side comes
# through hmr_reference.py: live here, recorded on CPU for the public release, so on another backend
# the comparison runs in a CPU subprocess.

def _base_loop_config(algo_key: str, section_key: str, **section) -> dict:
    return {
        'algorithm_params': {
            algo_key: {
                'population_size': 60,
                'substrate': {
                    'input_coords': [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],
                    'output_coords': [(0.0, 1.0)],
                    'output_activation': 'sigmoid',
                    'hidden_activation': 'tanh',
                },
                section_key: {
                    'initial_depth': 0,
                    'max_depth': 2,
                    'variance_threshold': 0.03,
                    **section,
                },
            }
        }
    }


def _on_cpu(test_id: str) -> None:
    """Run one test of this file in a fresh interpreter pinned to XLA:CPU (JAX reads JAX_PLATFORMS
    when it starts)."""
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    # An inherited JAX_PLATFORM_NAME or JAX_BACKEND_TARGET of gpu asks for a backend
    # JAX_PLATFORMS=cpu never loads
    env.pop('JAX_PLATFORM_NAME', None)
    env.pop('JAX_BACKEND_TARGET', None)
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-q',
            '-p',
            'no:cacheprovider',
            f'{os.path.basename(__file__)}::{test_id}',
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=here,
        timeout=900,
    )
    assert (
        proc.returncode == 0
    ), f'CPU subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}'


@pytest.mark.golden_cpu
def test_the_loop_equals_hmrs_base_loop():
    if jax.default_backend() != 'cpu':
        _on_cpu('test_the_loop_equals_hmrs_base_loop')
        return
    args = (
        _base_loop_config('hmrhyperneat', 'hmr_hyperneat'),
        XORProblem(),
        SEED,
        LOOP_GENERATIONS,
    )
    h_history, h_generations, h_key = hmr_value('generation_loop.hmr_base_while_loop',
                                                lambda: live('base_while_loop', *args), inputs=args)
    algorithm = EMRHyperNEAT()
    config = _base_loop_config(
        'emrhyperneat',
        'emr_hyperneat',
        variance_rule='legacy',
        extra_randkey_split=True,
        execution={'generation_loop': 'while_loop'},
    )
    state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=SEED)
    result = algorithm.run_until_threshold(state, XORProblem(), target_fitness=NO_STOP,
                                           max_generations=LOOP_GENERATIONS, collect_history=True)
    assert result['generation_loop'] == 'while_loop'
    assert result['generations'] == h_generations == LOOP_GENERATIONS
    np.testing.assert_allclose(
        result['history'], np.asarray(h_history, dtype=np.float64), rtol=1e-6, atol=0.0
    )
    np.testing.assert_array_equal(np.asarray(result['state'].randkey), h_key)
    # Negative control: EMR's default key handling (no pre-ask split) ends on another key
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['extra_randkey_split'] = False
    state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=SEED)
    other = algorithm.run_until_threshold(state, XORProblem(), target_fitness=NO_STOP,
                                          max_generations=LOOP_GENERATIONS)
    assert np.asarray(other['state'].randkey).tobytes() != np.asarray(h_key).tobytes()
