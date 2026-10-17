"""Golden-trajectory and determinism regression tests.

Two layers of protection:

1. ``test_same_seed_is_bit_identical``, running the identical config/seed twice
   must produce bit-identical fitness trajectories. Version-proof: holds on any
   platform/JAX version, so a failure always means a real nondeterminism or
   behavior regression.

2. ``test_golden_xor_trajectory``, the exact values the standalone port produced
   when it was verified bit-for-bit against the original research implementation
   (CPU, jax 0.6.1). Guards against silent numeric drift from future feature
   additions. Marked ``golden_cpu``: deselect it when intentionally changing JAX
   versions (then re-record the constants here). The mean is pinned per CPU
   architecture, see below. When the process's default backend is a GPU, the
   check runs in a CPU subprocess: the pins are CPU values, and GPU kernels round
   differently.
"""
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

import pytest

from emr_hyperneat import EMRHyperNEAT
from conftest import GPU_TRAJECTORY_RTOL  # type: ignore[import-not-found]

# Recorded on CPU with jax==0.6.1 / numpy 1.26.4 at the time the port was verified
# bit-identical to the research implementation (seed=42, pop=50, max_depth=3, XOR).
#
# Recorded under the default ``variance_rule: canonical``: ``best`` did not
# move (the best genome's substrate is unchanged), ``mean`` did. The 0.3.x values are
# still reproducible with ``variance_rule: legacy`` (arm64 mean 0.6244490743).
#
# ``best`` is a maximum over the population, so it is order-independent. ``mean`` sums 50
# float32 fitnesses, where the backend's summation order can shift the low bits; under the
# legacy rule the two architectures did differ there (the two literals below). Under the
# canonical rule they agree to the last digit, so this map currently holds one value twice.
# It is kept as a guard: if a genuine per-architecture regression appears, the 1e-9
# tolerance is already in place, without loosening it until a platform gap fits under it.
#
# The processor is NOT what drives substrate divergence between otherwise identical runs.
# JAX's ``jax_threefry_partitionable`` default was ``False`` through 0.4.38 and is ``True``
# from 0.5.0, and it changes the initial population drawn from a given seed. This package
# supports jax>=0.5,<0.7, where it is always ``True`` -- part of why the ``golden_cpu``
# marker gates on the pinned JAX. Comparing against numbers recorded on jax 0.4.x needs
# the flag set explicitly.
GOLDEN_GEN0_BEST = 0.7304084301
GOLDEN_GEN0_MEAN_BY_MACHINE = {
    "arm64": 0.6064587235450745,    # macOS, Apple Silicon (canonical)
    # x86-64 (Linux, jax 0.6.1, canonical): the same value to the last digit. Under the
    # legacy rule the two architectures differed (0.6244490743 vs 0.6244491338729858).
    "x86_64": 0.6064587235450745,
}
GOLDEN_TOL = 1e-9

GOLDEN_CONFIG = {
    "algorithm_params": {
        "emrhyperneat": {
            "population_size": 50,
            "substrate": {
                "input_coords": [[-1.0, 0.0], [1.0, 0.0]],
                "output_coords": [[0.0, 1.0]],
            },
            "emr_hyperneat": {
                "initial_depth": 0,
                "max_depth": 3,
                "variance_threshold": 0.03,
            },
            "neat_species": {
                "compatibility_threshold": 2.5,
                "max_stagnation": 40,
            },
        },
    },
}


class _XOR2:
    """2-input XOR without bias, the exact problem the goldens were recorded on."""

    input_shape = (2,)
    output_shape = (1,)
    jitable = True
    fitness_threshold = 0.95

    def __init__(self):
        self.inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        self.targets = [[0.0], [1.0], [1.0], [0.0]]
        self.input_coords = [[-1.0, 0.0], [1.0, 0.0]]
        self.output_coords = [[0.0, 1.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


def _run_trajectory(config, problem, n_generations=3, seed=42):
    algo = EMRHyperNEAT()
    cfg = algo.create_config(config)
    state = algo.initialize(cfg, problem, seed=seed)
    traj = []
    for _ in range(n_generations):
        state, metrics = algo.run_generation(state, problem)
        traj.append((float(metrics.best_fitness), float(metrics.mean_fitness)))
    return traj


def _golden_generation_zero():
    """(best, mean) of generation 0 for the golden configuration on this backend."""
    return _run_trajectory(GOLDEN_CONFIG, _XOR2(), n_generations=1, seed=42)[0]


def _golden_generation_zero_on_cpu():
    """``_golden_generation_zero`` in a fresh interpreter pinned to XLA:CPU.

    The goldens are CPU values. A process whose default backend is a GPU cannot
    reproduce them in-process: the package places its arrays on ``jax.devices()[0]``,
    and GPU kernels round differently (an RTX 2080 Ti puts the generation-0 best
    2.4e-7 below the pin). ``JAX_PLATFORMS`` must be set before JAX initializes, so
    the check runs in a subprocess, the way the paper-validation tests do.
    """
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "import test_golden\n"
        "print('GOLDEN_JSON=' + json.dumps(list(test_golden._golden_generation_zero())))\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "TF_CPP_MIN_LOG_LEVEL": "3"}
    # As in the paper-validation tests: an inherited JAX_PLATFORM_NAME or JAX_BACKEND_TARGET
    # of gpu asks JAX for a backend that JAX_PLATFORMS=cpu never loads.
    env.pop("JAX_PLATFORM_NAME", None)
    env.pop("JAX_BACKEND_TARGET", None)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900,
    )
    assert proc.returncode == 0, (
        f"CPU golden subprocess failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
    )
    for line in proc.stdout.splitlines():
        if line.startswith("GOLDEN_JSON="):
            best, mean = json.loads(line[len("GOLDEN_JSON="):])
            return best, mean
    raise AssertionError(
        f"no golden emitted.\nSTDOUT tail:\n{proc.stdout[-1500:]}\n"
        f"STDERR tail:\n{proc.stderr[-1500:]}"
    )


def test_same_seed_is_bit_identical(base_config, xor_problem):
    t1 = _run_trajectory(base_config, xor_problem)
    t2 = _run_trajectory(base_config, xor_problem)
    assert t1 == t2, f"non-deterministic trajectories:\n{t1}\nvs\n{t2}"
    assert all(math.isfinite(b) and math.isfinite(m) for b, m in t1)


@pytest.mark.golden_cpu
@pytest.mark.skipif(
    not jax.__version__.startswith("0.6"),
    reason=(
        "golden values were recorded on jax 0.6.x; install the pinned stack "
        "(pip install -e . -c requirements-lock.txt) to run them, or re-record "
        "the constants for a new JAX version"
    ),
)
def test_golden_xor_trajectory(xor_problem):
    machine = platform.machine()
    if machine not in GOLDEN_GEN0_MEAN_BY_MACHINE:
        pytest.skip(
            f"no golden mean recorded for CPU architecture {machine!r}; the mean is "
            f"summation-order dependent, so it must be recorded per architecture. "
            f"Run this test, take the reported value, and add it to "
            f"GOLDEN_GEN0_MEAN_BY_MACHINE."
        )
    golden_mean = GOLDEN_GEN0_MEAN_BY_MACHINE[machine]

    if jax.default_backend() == "cpu":
        best, mean = _golden_generation_zero()
    else:
        best, mean = _golden_generation_zero_on_cpu()
        # The pins are CPU values, so the exact check above runs on CPU. That alone
        # would make this test blind to the backend it is running on; also observe
        # the in-process value here, to float32 rounding (an RTX 2080 Ti puts the
        # generation-0 best 2.4e-7 below the pin, 3.3e-7 relative).
        device_best, device_mean = _golden_generation_zero()
        np.testing.assert_allclose(
            device_best, GOLDEN_GEN0_BEST, rtol=GPU_TRAJECTORY_RTOL, atol=0.0,
            equal_nan=False,
            err_msg=f"{jax.default_backend()} golden best beyond float32 rounding of the CPU pin")
        np.testing.assert_allclose(
            device_mean, golden_mean, rtol=GPU_TRAJECTORY_RTOL, atol=0.0,
            equal_nan=False,
            err_msg=f"{jax.default_backend()} golden mean beyond float32 rounding of the CPU pin")
    assert abs(best - GOLDEN_GEN0_BEST) < GOLDEN_TOL, (
        f"golden best drifted: {best!r} != {GOLDEN_GEN0_BEST!r}"
    )
    assert abs(mean - golden_mean) < GOLDEN_TOL, (
        f"golden mean drifted on {machine}: {mean!r} != {golden_mean!r}"
    )
