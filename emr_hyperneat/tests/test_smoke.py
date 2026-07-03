"""Smoke test: the standalone EMR-HyperNEAT package runs evolution end-to-end.

Builds a minimal 2-input XOR substrate, runs a few generations, and asserts the
algorithm produces finite, in-range fitness. This validates the standalone port
(import + config + initialize + run_generation) without any geenns dependency.

Run directly:
    JAX_PLATFORMS=cpu python emr_hyperneat/tests/test_smoke.py

(requires `pip install -e .` and `pip install -e third_party/tensorneat`)
or via pytest.
"""
import math

import numpy as np

from emr_hyperneat import EMRHyperNEAT


class XORProblem:
    """Minimal 2-input XOR problem (no bias) on a 2->1 substrate."""

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


def _build_config(pop=50, max_depth=3):
    return {
        "algorithm_params": {
            "emrhyperneat": {
                "population_size": pop,
                "substrate": {
                    "input_coords": [[-1.0, 0.0], [1.0, 0.0]],
                    "output_coords": [[0.0, 1.0]],
                },
                "emr_hyperneat": {
                    "initial_depth": 0,
                    "max_depth": max_depth,
                    "variance_threshold": 0.03,
                },
                "neat_species": {
                    "compatibility_threshold": 2.5,
                    "max_stagnation": 40,
                },
            },
        },
    }


def _run(cfg_dict, n_generations=3, seed=42):
    algo = EMRHyperNEAT()
    cfg = algo.create_config(cfg_dict)
    problem = XORProblem()
    state = algo.initialize(cfg, problem, seed=seed)

    best = None
    for gen in range(n_generations):
        state, metrics = algo.run_generation(state, problem)
        best = float(metrics.best_fitness)
        assert math.isfinite(best), f"gen {gen}: non-finite best_fitness {best}"
        assert -1e-6 <= best <= 1.0 + 1e-6, f"gen {gen}: out-of-range best_fitness {best}"
        print(f"  gen {gen}: best_fitness={best:.4f} mean={float(metrics.mean_fitness):.4f}")
    assert best is not None
    return best


def test_emr_runs_xor(n_generations=3, seed=42):
    """Base EMR path (single CPPN output)."""
    return _run(_build_config(), n_generations, seed)


def test_emr_dynamic_functions(n_generations=3, seed=42):
    """Per-node dynamic activation functions, exercises the multi-output CPPN
    path (num_cppn_outputs > 1)."""
    cfg = _build_config()
    cfg["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["dynamic_functions"] = {
        "mode": "cppn_output",
        "num_activations": 6,
    }
    return _run(cfg, n_generations, seed)


if __name__ == "__main__":
    print("[base EMR / XOR]")
    bf = test_emr_runs_xor()
    print("[dynamic_functions / multi-output CPPN]")
    df = test_emr_dynamic_functions()
    print(f"\nSMOKE OK -> base best={bf:.4f} | dynamic_functions best={df:.4f}")
