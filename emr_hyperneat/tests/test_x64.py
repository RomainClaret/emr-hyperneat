"""EMR under JAX's x64 mode.

With ``JAX_ENABLE_X64=1`` EMR stopped at its first generation on every platform: TensorNEAT's
species update traces two ``lax.cond`` calls whose branches differ in dtype (``check_stagnation``:
float32 fitness and generation against the species state's float64 ``best_fitness`` and
``last_improved``; ``count_members``: ``jnp.nan`` as float64 against a float32 count), and tracing
raised "cond branches must have equal output types". EMR's ``create_config`` now
gives the species module a dtype-unifying ``lax.cond`` when x64 is on (``x64_compat.py``).

Under x64 the plain feed-forward fitness would be float32, on the traceable core and on the general
path alike: TensorNEAT's CPPN forward returns weakly typed float64, the on-device query keeps the
weak type (a numpy round trip made the weights strong float64), and a weak float64 weight meeting
the float32 inputs computes in float32. ``substrate_weight_matrices`` takes the CPPN outputs as
strong default floats (``strong_default_float``), and the while_loop generation loop carries its
fitness in the default float. These tests run x64 in subprocesses, since x64 is a process-wide
setting.
"""
import os
import subprocess
import sys
import textwrap

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from emr_hyperneat import x64_compat
from emr_hyperneat import emrhyperneat as engine

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_x64(script: str, timeout: int = 900) -> str:
    """Run ``script`` on the CPU with x64 on, in a subprocess; return its stdout."""
    env = {key: value for key, value in os.environ.items()
           if key not in ('JAX_PLATFORM_NAME', 'JAX_BACKEND_TARGET')}
    env.update(JAX_ENABLE_X64='1', JAX_PLATFORMS='cpu')
    source = f'import sys\nsys.path.insert(0, {TESTS_DIR!r})\n' + textwrap.dedent(script)
    result = subprocess.run([sys.executable, '-c', source], env=env, capture_output=True,
                            text=True, timeout=timeout, cwd=TESTS_DIR)
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


_GENERATIONS = '''
    import jax, jax.numpy as jnp
    from conftest import XORProblem, create_base_config, EMRHyperNEAT
    assert jax.config.x64_enabled
    for preset in {presets!r}:
        config = create_base_config(population_size=30, max_depth=3)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = {{
            'preset': preset}}
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
        for _ in range(3):
            state, _ = algorithm.run_generation(state, problem)
        fitness = algorithm._cached_fitnesses
        species = state.species
        print(preset, bool(jnp.all(jnp.isfinite(fitness))), float(jnp.max(fitness)) > 0.5,
              str(species.best_fitness.dtype), str(species.member_count.dtype))
'''

_PRESETS = ('feedforward', 'hidden_only', 'full_recurrent')


class TestX64Generations:
    def test_three_generations_complete(self):
        """Feed-forward and two recurrent presets run three generations under x64; the species
        state is float64."""
        out = _run_x64(_GENERATIONS.format(presets=_PRESETS))
        lines = [line.split() for line in out.splitlines()
                 if line.split()[:1] and line.split()[0] in _PRESETS]
        assert [line[0] for line in lines] == list(_PRESETS)
        for _, finite, positive, best_dtype, count_dtype in lines:
            assert (finite, positive, best_dtype, count_dtype) == ('True', 'True', 'float64',
                                                                   'float64')

    def test_negative_control_without_the_patch(self):
        """With the species module's real ``jax`` put back after create_config, the first tell
        raises the error 0.1.0 raised, so the generations above pass because of the patch."""
        out = _run_x64('''
            import jax
            from conftest import XORProblem, create_base_config, EMRHyperNEAT
            from tensorneat.algorithm.neat import species as species_module
            config = create_base_config(population_size=30, max_depth=3)
            algorithm = EMRHyperNEAT()
            problem = XORProblem()
            state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
            species_module.jax = jax
            try:
                algorithm.run_generation(state, problem)
                print('NO ERROR')
            except TypeError as exc:
                print('TypeError', 'cond branches must have equal output types' in str(exc))
        ''')
        assert 'TypeError True' in out


_CORE_AND_GENERAL = '''
    import copy, json
    import numpy as np
    from conftest import RECURRENCE_PRESETS, EMRHyperNEAT, XORProblem, create_base_config
    from emr_hyperneat import emrhyperneat as engine
    if {no_cast!r}:
        # The negative control: the weights keep TensorNEAT's weak type
        engine.strong_default_float = lambda x: x
    config = create_base_config(max_depth=3, population_size=30)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['recurrence'] = dict(RECURRENCE_PRESETS['feedforward'])

    def run(general):
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(copy.deepcopy(config)), XORProblem(),
                                     seed=42)
        if general:
            algorithm._feedforward_core_blockers = lambda: ['forced general path']
        records = []
        for _ in range(3):
            state, _ = algorithm.run_generation(state, XORProblem())
            records.append({{name: np.asarray(getattr(algorithm, attr)) for name, attr in (
                ('fitnesses', '_cached_fitnesses'), ('W1', '_cached_W1'), ('W2', '_cached_W2'),
                ('masks', '_cached_masks_A'))}})
        return records

    core, general = run(False), run(True)
    print('RESULT', json.dumps({{
        'dtypes': sorted({{str(r['fitnesses'].dtype) for r in core + general}}),
        'equal': all(c[k].dtype == g[k].dtype and c[k].tobytes() == g[k].tobytes()
                     for c, g in zip(core, general) for k in c),
        'best': [float(np.max(r['fitnesses'])) for r in core],
    }}))
'''


def _core_and_general(no_cast: bool) -> dict:
    out = _run_x64(_CORE_AND_GENERAL.format(no_cast=no_cast))
    return json.loads(
        next(line for line in out.splitlines() if line.startswith('RESULT'))[len('RESULT ') :]
    )


class TestFeedforwardFloat64:
    def test_core_and_general_path_score_in_float64_alike(self):
        """Under x64 the plain feed-forward core and the general path score in float64 and agree bit
        for bit over three generations (fitness, W1, W2, masks); some best is not a float32 value.
        """
        result = _core_and_general(no_cast=False)
        assert result['dtypes'] == ['float64'] and result['equal'], result
        assert any(float(np.float32(best)) != best for best in result['best']), result

    def test_negative_control_without_the_cast(self):
        """Without the cast the weights stay weakly typed and the fitness is float32, on both paths.
        """
        result = _core_and_general(no_cast=True)
        assert result['dtypes'] == ['float32'] and result['equal'], result

    def test_while_loop_carries_float64(self):
        """Item 8b's while_loop runs under x64 (its carries were float32) and gives the Python
        loop's float64 history."""
        out = _run_x64(
            '''
            import json
            from conftest import RECURRENCE_PRESETS, EMRHyperNEAT, XORProblem, create_base_config
            histories = {}
            for loop in ('python', 'while_loop'):
                config = create_base_config(max_depth=3, population_size=30)
                emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
                emr['recurrence'] = dict(RECURRENCE_PRESETS['feedforward'])
                emr['execution'] = {'generation_loop': loop}
                algorithm = EMRHyperNEAT()
                state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=42)
                result = algorithm.run_until_threshold(state, XORProblem(), target_fitness=2.0,
                                                       max_generations=3, collect_history=True)
                histories[loop] = [result['generations'], result['history']]
            print('LOOPS', json.dumps(histories))
        '''
        )
        loops = json.loads(
            next(line for line in out.splitlines() if line.startswith('LOOPS'))[len('LOOPS ') :]
        )
        (python_generations, python_history), (loop_generations, loop_history) = (
            loops['python'],
            loops['while_loop'],
        )
        assert python_generations == loop_generations == 3
        np.testing.assert_allclose(loop_history, python_history, rtol=1e-12, atol=0.0)
        assert any(float(np.float32(best)) != best for best in loop_history), loop_history


class TestFloat32Untouched:
    def test_nothing_is_patched_under_float32(self):
        """Under float32 (this process) ensure_x64_species_dtypes does nothing: the species module
        keeps the real jax, so float32 runs trace what they traced before."""
        assert not jax.config.x64_enabled
        assert x64_compat.ensure_x64_species_dtypes() is False
        assert x64_compat.species_module().jax is jax

    def test_strong_default_float_keeps_float32_values(self):
        """Under float32 the cast only drops the weak type: float32 in, the same bytes out."""
        weak = jnp.full((5,), 0.1)
        assert weak.dtype == jnp.float32 and weak.weak_type
        strong = engine.strong_default_float(weak)
        assert strong.dtype == jnp.float32 and not strong.weak_type
        assert np.asarray(strong).tobytes() == np.asarray(weak).tobytes()


class TestUnifyingCond:
    def test_promotes_mismatched_branches(self):
        """Branches of different dtypes, leaf by leaf, come out at the promoted dtype; plain
        lax.cond refuses them."""
        def true_fun():
            return jnp.float32(1.0), jnp.int32(2)

        def false_fun():
            return jnp.int32(3), jnp.float32(4.0)

        for pred, values in ((True, [1.0, 2.0]), (False, [3.0, 4.0])):
            out = jax.jit(lambda p: x64_compat.unifying_cond(p, true_fun, false_fun))(pred)
            assert [str(leaf.dtype) for leaf in out] == ['float32', 'float32']
            assert [float(leaf) for leaf in out] == values
        with pytest.raises(TypeError, match='cond branches must have equal output types'):
            jax.jit(lambda p: jax.lax.cond(p, true_fun, false_fun))(True)

    def test_equal_dtypes_are_left_as_they_are(self):
        """Matching branches give lax.cond's own result, value for value."""
        def branches(x):
            return (lambda: x * 2.0, lambda: x - 1.0)

        x = jnp.arange(4, dtype=jnp.float32)
        for pred in (True, False):
            ours = x64_compat.unifying_cond(pred, *branches(x))
            theirs = jax.lax.cond(pred, *branches(x))
            assert ours.dtype == theirs.dtype and bool(jnp.array_equal(ours, theirs))

    def test_idempotent_under_x64(self):
        out = _run_x64('''
            from emr_hyperneat import x64_compat
            assert x64_compat.ensure_x64_species_dtypes() is True
            first = x64_compat.species_module().jax
            assert x64_compat.ensure_x64_species_dtypes() is True
            print('SAME', x64_compat.species_module().jax is first)
        ''')
        assert 'SAME True' in out
