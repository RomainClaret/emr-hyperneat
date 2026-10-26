"""The pmap pipeline discovers its h->h connections outside the pmap.

In 0.1.0 the recurrent FULL_PIPELINE_PARALLEL route ran out of memory on GPUs
(``RESOURCE_EXHAUSTED ... 69.48GiB on device ordinal 0`` at depth 4 and population 100, on 2 and
4 devices alike) where SINGLE_GPU ran the same configuration on one 24 GB GPU: its pmapped core
queried every genome's CPPN at every pair of grid positions in one vmap, one buffer of about
P x N^2 x 401 bytes, 115 GiB of compiled temporaries in all at depth 4 and population 100 (XLA:CPU,
2 forced devices). The runner now finds the masks and the connections once, outside the pmap,
with the bounded discovery SINGLE_GPU and EVAL_ONLY_PARALLEL run, and passes them in, replicated
on every device. Each case runs in a subprocess with forced CPU devices.
"""
import json
import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import pytest

from emr_hyperneat import emrhyperneat as engine

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

_PROBLEM = '''
    import numpy as np

    class SampleProblem:
        input_shape = (3,)
        output_shape = (1,)
        jitable = True
        use_bias = True
        is_exhaustive_dataset = False
        fitness_threshold = 0.95

        def __init__(self, n_samples):
            rng = np.random.default_rng(7)
            points = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
            self.inputs = [[float(x), float(y), 1.0] for x, y in points]
            self.targets = [[float(x * y > 0)] for x, y in points]

        def get_data(self):
            return list(zip(self.inputs, self.targets))
'''


def _run(script: str, devices: int, timeout: int = 1800) -> str:
    """Run ``script`` after the sample problem, on ``devices`` forced CPU devices; its stdout."""
    env = {key: value for key, value in os.environ.items()
           if key not in ('JAX_PLATFORM_NAME', 'JAX_BACKEND_TARGET')}
    env.update(JAX_PLATFORMS='cpu', XLA_FLAGS=f'--xla_force_host_platform_device_count={devices}')
    source = (f'import sys\nsys.path.insert(0, {TESTS_DIR!r})\n' + textwrap.dedent(_PROBLEM)
              + textwrap.dedent(script))
    result = subprocess.run([sys.executable, '-c', source], env=env, capture_output=True,
                            text=True, timeout=timeout, cwd=TESTS_DIR)
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


_HISTORY = '''
    import json
    from conftest import EMRHyperNEAT, create_config_with_recurrence
    cfg = create_config_with_recurrence('hidden_only', max_depth=3, population_size={pop},
                                        iteration_level=1, hh_cache_enabled=False)
    emr = cfg['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['multi_gpu_strategy'] = {strategy!r}
    algo = EMRHyperNEAT()
    problem = SampleProblem(64)
    state = algo.initialize(algo.create_config(cfg), problem, seed=42)
    history, tell = [], algo._compiled_tell
    def recording(st, fitness):
        history.append([float(v) for v in np.asarray(fitness, dtype=np.float64)])
        return tell(st, fitness)
    algo._compiled_tell = recording
    route = algo.run_until_threshold(state, problem, target_fitness=2.0,
                                     max_generations=3)['route']
    print('HISTORY', json.dumps({{'route': route, 'history': history}}))
'''


def _history(strategy: str, devices: int, pop: int = 30):
    script = _HISTORY.format(strategy=strategy, pop=pop)
    out = _run(script, devices)
    line = next(line for line in out.splitlines() if line.startswith('HISTORY'))
    record = json.loads(line[len('HISTORY '):])
    return record['route'], [jnp.asarray(g) for g in record['history']]


def _gaps(history, reference):
    return [float(jnp.max(jnp.abs(a - b))) for a, b in zip(history, reference)]


@pytest.fixture(scope='module')
def single_gpu():
    route, history = _history('SINGLE_GPU', 1)
    assert route == 'SINGLE_GPU' and len(history) == 3
    return history


class TestRecurrentFullPipeline:
    @pytest.mark.parametrize('devices', [2, 4])
    def test_scores_as_single_gpu(self, single_gpu, devices):
        """Three generations of the recurrent FULL_PIPELINE_PARALLEL route on 2 and 4 devices
        against SINGLE_GPU: generation 0 within 2e-7 and every generation within 1e-5. The route
        sums its samples in another order (data shards), and the rounding grows over generations:
        at generation 1, 1.8e-6 on an Apple M4 and 5.25e-6 on an x86 Xeon. With the discovery
        inside the pmap it gave the same values, bit for bit, at these sizes."""
        route, history = _history('FULL_PIPELINE_PARALLEL', devices)
        assert route == 'FULL_PIPELINE_PARALLEL' and len(history) == 3
        gaps = _gaps(history, single_gpu)
        assert gaps[0] <= 2e-7, gaps
        assert max(gaps) <= 1e-5, gaps


_MEMORY = '''
    import jax
    from conftest import EMRHyperNEAT, create_config_with_recurrence
    from emr_hyperneat import emrhyperneat as engine

    class Stop(Exception):
        pass

    original = engine._pmap_data_parallel_full_hh

    def compiled_size(*args):
        print('TEMP', original.lower(*args).compile().memory_analysis().temp_size_in_bytes)
        raise Stop()

    engine._pmap_data_parallel_full_hh = compiled_size
    cfg = create_config_with_recurrence('hidden_only', max_depth=4, population_size=100,
                                        iteration_level=1, hh_cache_enabled=False)
    cfg['algorithm_params']['emrhyperneat']['emr_hyperneat']['multi_gpu_strategy'] = (
        'FULL_PIPELINE_PARALLEL')
    algo = EMRHyperNEAT()
    problem = SampleProblem(2048)
    state = algo.initialize(algo.create_config(cfg), problem, seed=42)
    try:
        algo._run_until_threshold_data_parallel_full(state, problem, target_fitness=2.0,
                                                     max_generations=1)
    except Stop:
        pass
'''


@pytest.mark.slow
def test_compiled_memory_is_bounded_and_does_not_grow_with_devices():
    """The pmapped core's compiled temporaries at depth 4, population 100, 2,048 samples: 115.0 and
    114.7 GiB before the fix on 2 and 4 devices; now under 4 GiB, and no larger on 4 devices than on
    2 (2.255 and 1.474 GiB on this Mac)."""
    sizes = {}
    for devices in (2, 4):
        out = _run(_MEMORY, devices)
        sizes[devices] = int(next(line for line in out.splitlines()
                                  if line.startswith('TEMP')).split()[1])
    assert sizes[2] < 4 * 2**30, sizes
    assert sizes[4] <= sizes[2] * 1.05, sizes


def test_core_refuses_hh_without_the_outside_discovery():
    """With h->h connections the core needs the discovery from outside the pmap; it does not
    discover them itself any more."""
    zeros = jnp.zeros((2, 1))
    with pytest.raises(ValueError, match='discovery outside the pmap'):
        engine._full_pipeline_with_hh_core(
            (zeros, zeros, zeros, zeros), jnp.zeros((5, 2)), jnp.zeros((2, 2)),
            jnp.zeros((1, 2)), jnp.zeros((4, 2)), jnp.zeros((4, 1)), 0.03, 0.3, 3.0, 0.1, 5,
            (4,), (0,), (2,), (), 1, 5, 10, 1, True, None, None, hh=None,
        )
