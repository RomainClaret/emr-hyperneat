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
   versions (then re-record the constants here).
"""
import math

import jax

import pytest

from emr_hyperneat import EMRHyperNEAT

# Recorded on CPU with jax==0.6.1 at the time the port was verified bit-identical
# to the research implementation (seed=42, pop=50, max_depth=3, XOR).
GOLDEN_GEN0_BEST = 0.7304084301
GOLDEN_GEN0_MEAN = 0.6244490743
GOLDEN_TOL = 1e-9


def _run_trajectory(config, problem, n_generations=3, seed=42):
    algo = EMRHyperNEAT()
    cfg = algo.create_config(config)
    state = algo.initialize(cfg, problem, seed=seed)
    traj = []
    for _ in range(n_generations):
        state, metrics = algo.run_generation(state, problem)
        traj.append((float(metrics.best_fitness), float(metrics.mean_fitness)))
    return traj


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
    config = {
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

    traj = _run_trajectory(config, _XOR2(), n_generations=1, seed=42)
    best, mean = traj[0]
    assert abs(best - GOLDEN_GEN0_BEST) < GOLDEN_TOL, (
        f"golden best drifted: {best!r} != {GOLDEN_GEN0_BEST!r}"
    )
    assert abs(mean - GOLDEN_GEN0_MEAN) < GOLDEN_TOL, (
        f"golden mean drifted: {mean!r} != {GOLDEN_GEN0_MEAN!r}"
    )
