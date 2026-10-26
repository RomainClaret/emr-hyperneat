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


def _dynamic_functions_config():
    cfg = _build_config()
    cfg["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["dynamic_functions"] = {
        "mode": "cppn_output",
        "num_activations": 6,
    }
    return cfg


def test_emr_runs_xor(n_generations=3, seed=42):
    """Base EMR path (single CPPN output)."""
    _run(_build_config(), n_generations, seed)


def test_emr_dynamic_functions(n_generations=3, seed=42):
    """Per-node dynamic activation functions, exercises the multi-output CPPN
    path (num_cppn_outputs > 1)."""
    _run(_dynamic_functions_config(), n_generations, seed)


def test_fitness_utils():
    """fitness_utils.py, the fitness helpers docs/architecture.md lists and the engine does not
    import: its docstring's example (outputs [0.4, 0.2, 0.2, 0.2] for class 0: MSE fitness 0.88,
    argmax accuracy 1), a binary case under every mode and its jitted form, and the threshold
    helper's example."""
    import jax.numpy as jnp
    import pytest

    from emr_hyperneat import fitness_utils

    outputs, targets = jnp.array([[0.4, 0.2, 0.2, 0.2]]), jnp.array([[1.0, 0.0, 0.0, 0.0]])
    assert math.isclose(float(fitness_utils.compute_fitness(outputs, targets, 'mse')), 0.88,
                        rel_tol=1e-6)
    assert float(fitness_utils.compute_fitness(outputs, targets, 'accuracy')) == 1.0
    assert float(fitness_utils.compute_classification_accuracy(outputs, targets)) == 1.0

    binary = jnp.array([[0.1], [0.9], [0.8], [0.3]])
    labels = jnp.array([[0.0], [1.0], [1.0], [0.0]])
    for mode in fitness_utils.VALID_FITNESS_MODES:
        value = float(fitness_utils.compute_fitness(binary, labels, mode))
        assert math.isfinite(value) and value > 0.0, (mode, value)
        jitted = float(fitness_utils.compute_fitness_jit(binary, labels, mode, False))
        assert math.isclose(jitted, value, rel_tol=1e-6), (mode, jitted, value)
    assert math.isclose(float(fitness_utils.compute_fitness(binary, labels, 'mse')), 0.9625,
                        rel_tol=1e-6)
    assert math.isclose(fitness_utils.threshold_from_relative_improvement(0.5, 0.8), 0.9)
    with pytest.raises(ValueError, match='Unknown fitness mode'):
        fitness_utils.compute_fitness(binary, labels, 'f1')


if __name__ == "__main__":
    print("[base EMR / XOR]")
    bf = _run(_build_config())
    print("[dynamic_functions / multi-output CPPN]")
    df = _run(_dynamic_functions_config())
    print(f"\nSMOKE OK -> base best={bf:.4f} | dynamic_functions best={df:.4f}")
