"""EMR against HMR's dynamic-functions variant, the engine of the published pre-migration results.

On the shared feed-forward path, with extra_randkey_split and the legacy variance rule, EMR
reproduces that variant bit for bit: the best fitness of every generation and the final randkey, for
two activation functions and one seed (a compact form of the 18-cell check of
papers/emr-dynamic-functions). HMR's side comes through hmr_reference.py: run live in the research
repository and recorded there on the CPU for this package, whose tests do not run HMR. The recorded
values are CPU values, so on another backend the comparison runs in a CPU subprocess.
"""
import os
import subprocess
import sys

import jax
import numpy as np
import pytest

from conftest import EMRHyperNEAT  # type: ignore[import-not-found]
from hmr_reference import decode, digest, encode, hmr_value, live  # type: ignore[import-not-found]


class _Parity3:
    """3-input parity, the problem family the published HMR results used."""

    input_shape = (3,)
    output_shape = (1,)
    jitable = True
    fitness_threshold = 0.95

    def __init__(self):
        self.inputs = [[float(b) for b in f'{i:03b}'] for i in range(8)]
        self.targets = [[float(sum(int(x) for x in f'{i:03b}') % 2)] for i in range(8)]
        self.input_coords = [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]]
        self.output_coords = [[0.0, 1.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


def _config(algo_key, section_key, activation):
    return {
        'algorithm_params': {
            algo_key: {
                'population_size': 60,
                'substrate': {
                    'input_coords': [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                    'output_coords': [[0.0, 1.0]],
                },
                section_key: {
                    'initial_depth': 0,
                    # HMR's variance rule is the original level-0 one; EMR is pinned to it to stay
                    # bit-equivalent
                    'variance_rule': 'legacy',
                    'max_depth': 2,
                    'variance_threshold': 0.03,
                    'dynamic_functions': {'mode': 'global', 'hidden_activation': activation},
                    # restores HMR's extra per-generation randkey split in EMR
                    'extra_randkey_split': True,
                },
                # No neat_species block: EMR reads it, while HMR keeps its own defaults
                # (compatibility 3.0), so passing one would make the engines differ
            },
        },
    }


def _trajectory(config, problem, seed, generations=3):
    algo = EMRHyperNEAT()
    state = algo.initialize(algo.create_config(config), problem, seed=seed)
    fits = []
    for _ in range(generations):
        state, metrics = algo.run_generation(state, problem)
        fits.append(float(metrics.best_fitness))
    return fits, np.asarray(state.randkey)


def _on_cpu(test_id):
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
@pytest.mark.parametrize('activation', ['tanh', 'sin'])
def test_emr_reproduces_hmrs_trajectory(activation):
    """EMR with extra_randkey_split equals HMR's dynamic-functions variant: every generation's best
    fitness and the final randkey."""
    if jax.default_backend() != 'cpu':
        _on_cpu(f'test_emr_reproduces_hmrs_trajectory[{activation}]')
        return
    problem = _Parity3()
    args = (_config('hmrhyperneat', 'hmr_hyperneat', activation), problem, 42, 3)
    h_fits, h_key = hmr_value(
        'hmr_reference.frozen_trajectory', lambda: live('frozen_trajectory', *args), inputs=args
    )
    e_fits, e_key = _trajectory(
        _config('emrhyperneat', 'emr_hyperneat', activation), problem, seed=42
    )
    assert h_fits == e_fits, f'{activation}: trajectories diverge\n  HMR: {h_fits}\n  EMR: {e_fits}'
    np.testing.assert_array_equal(e_key, h_key, err_msg=f'{activation}: final randkey differs')


def test_the_encoding_is_exact():
    """A value comes back bit for bit (NaN payloads, -0.0, float32 and Python floats, keys,
    containers), and its digest with it."""
    value = {
        'array': np.array([[np.nan, -0.0], [1.5, 0.1]], dtype=np.float32),
        'float': 0.1,
        'scalar': np.float32(0.3),
        'key': np.array([0, 42], dtype=np.uint32),
        ('pair', 1): [True, None, 'x'],
        'set': {'b', 'a'},
        'tuple': (1, 2.5),
    }
    back = decode(encode(value))
    assert back.keys() == value.keys()
    for name in ('array', 'key'):
        assert back[name].dtype == value[name].dtype
        assert back[name].tobytes() == value[name].tobytes()
    assert (
        back['float'] == 0.1
        and back['scalar'] == np.float32(0.3)
        and type(back['scalar']) is np.float32
    )
    assert (
        back[('pair', 1)] == [True, None, 'x']
        and back['set'] == {'a', 'b'}
        and back['tuple'] == (1, 2.5)
    )
    assert digest(back) == digest(value)
    # Negative control: one bit of the array changes the digest
    flipped = value['array'].copy()
    flipped.view(np.uint32)[1, 1] ^= 1
    assert digest({**value, 'array': flipped}) != digest(value)
