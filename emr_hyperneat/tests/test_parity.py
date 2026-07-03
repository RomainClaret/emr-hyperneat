"""Determinism harness: prints a high-precision per-generation fitness trajectory
for a fixed XOR config/seed.

EMR-HyperNEAT is deterministic for a fixed seed, so this trajectory is stable across
runs and machines. It was used to confirm the standalone port reproduces the original
research implementation bit-for-bit (best + mean fitness to 10 decimals); it now serves
as a regression/determinism check.

    JAX_PLATFORMS=cpu python emr_hyperneat/tests/test_parity.py

(requires `pip install -e .` and `pip install -e third_party/tensorneat`)
"""
from emr_hyperneat import EMRHyperNEAT

IMPL = "standalone"


class XORProblem:
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


CFG = {
    "algorithm_params": {
        "emrhyperneat": {
            "population_size": 50,
            "substrate": {
                "input_coords": [[-1.0, 0.0], [1.0, 0.0]],
                "output_coords": [[0.0, 1.0]],
            },
            "emr_hyperneat": {"initial_depth": 0, "max_depth": 3, "variance_threshold": 0.03},
            "neat_species": {"compatibility_threshold": 2.5, "max_stagnation": 40},
        }
    }
}


def main(n_generations=5, seed=42):
    algo = EMRHyperNEAT()
    cfg = algo.create_config(CFG)
    problem = XORProblem()
    state = algo.initialize(cfg, problem, seed=seed)
    print(f"IMPL={IMPL}")
    for gen in range(n_generations):
        state, metrics = algo.run_generation(state, problem)
        print(f"gen {gen}: best={float(metrics.best_fitness):.10f} "
              f"mean={float(metrics.mean_fitness):.10f}")


if __name__ == "__main__":
    main()
