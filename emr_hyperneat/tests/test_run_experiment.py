"""The base class's whole-run loop, `run_experiment`: it runs the generations, keeps each
generation's metrics in memory, in order, and returns the run's summary.
"""
from typing import Any, Dict, List

from conftest import DEFAULT_SEED, XORProblem, create_base_config  # type: ignore[import-not-found]
from emr_hyperneat import EMRHyperNEAT

GENERATIONS = 2
POPULATION = 50
#: A target fitness no run this short reaches on XOR
UNREACHED = 0.999


def _recording(algorithm: EMRHyperNEAT, monkeypatch: Any) -> List[Any]:
    """The metrics each `run_generation` call of ``algorithm`` returns, in call order."""
    returned: List[Any] = []
    run_generation = algorithm.run_generation

    def recording(state: Any, problem: Any) -> Any:
        state, metrics = run_generation(state, problem)
        returned.append(metrics)
        return state, metrics

    monkeypatch.setattr(algorithm, "run_generation", recording)
    return returned


def _run(algorithm: EMRHyperNEAT, generations: int) -> Dict[str, Any]:
    """`run_experiment` on XOR (population 50, depth 2) with the unreached target."""
    config = algorithm.create_config(create_base_config(population_size=POPULATION))
    return algorithm.run_experiment(config, XORProblem(), max_generations=generations,
                                    target_fitness=UNREACHED, seed=DEFAULT_SEED)


def test_the_summary_holds_every_generation(monkeypatch):
    """The summary counts the generations and holds each one's metrics as the generation returned
    them; its best fitness and total time come from them, and a target no generation reaches is
    not a success. `metrics_history`, `get_best_fitness` and `get_final_stats` read the same
    record."""
    algorithm = EMRHyperNEAT()
    returned = _recording(algorithm, monkeypatch)
    result = _run(algorithm, GENERATIONS)
    history = result["metrics_history"]
    assert result["generations"] == GENERATIONS
    assert len(history) == len(returned) == GENERATIONS
    assert all(kept is given for kept, given in zip(history, returned))
    assert all(kept is given for kept, given in zip(algorithm.metrics_history, returned))
    best = max(float(metrics.best_fitness) for metrics in returned)
    assert result["best_fitness"] == best and 0.0 < best < UNREACHED
    assert result["success"] is False
    assert result["total_time"] == sum(metrics.time_elapsed for metrics in returned)
    assert algorithm.get_best_fitness() == best
    assert algorithm.get_final_stats()["generations"] == GENERATIONS


def test_each_run_keeps_its_own_record(monkeypatch):
    """A second run on the same instance keeps only its own generations, and assigning an empty
    list to `metrics_history` clears the record."""
    algorithm = EMRHyperNEAT()
    returned = _recording(algorithm, monkeypatch)
    _run(algorithm, GENERATIONS)
    returned.clear()
    result = _run(algorithm, 1)
    assert len(result["metrics_history"]) == 1 and result["metrics_history"][0] is returned[0]
    assert len(algorithm.metrics_history) == 1 and algorithm.metrics_history[-1] is returned[0]
    algorithm.metrics_history = []
    assert len(algorithm.metrics_history) == 0 and not algorithm.metrics_history
