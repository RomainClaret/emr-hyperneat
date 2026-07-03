"""Frozen-HMR integrity: the frozen HMR module must stay bit-equivalent to EMR.

The frozen HMR modules under ``emr_hyperneat._hmr_frozen`` exist solely to
reproduce pre-migration published results. Their load-bearing property is that
on the shared feedforward path (with EMR's ``extra_randkey_split`` flag) they are
**bit-identical** to EMR. This is a compact, fast version of the full 18-cell
check in ``papers/emr-dynamic-functions/scripts/analysis/_validate_emr_port.py``:
2 activation functions x 1 seed, full trajectory + final randkey.
"""
import importlib

import numpy as np
import pytest

from emr_hyperneat import EMRHyperNEAT
from emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions import (
    HMRHyperNEAT as HMRDynamicFunctions,
)

FROZEN_MODULES = [
    "emr_hyperneat._hmr_frozen.hmrhyperneat",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions_aggregation",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_neuromodulation_functions",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended_dynamic_functions_full",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_pipeline_chunking_multi_gpus",
]


@pytest.mark.parametrize("module_name", FROZEN_MODULES)
def test_frozen_module_imports(module_name):
    importlib.import_module(module_name)


class _Parity3:
    """3-input parity, the problem family the published HMR results used."""

    input_shape = (3,)
    output_shape = (1,)
    jitable = True
    fitness_threshold = 0.95

    def __init__(self):
        self.inputs = [
            [float(b) for b in f"{i:03b}"] for i in range(8)
        ]
        self.targets = [[float(sum(int(x) for x in f"{i:03b}") % 2)] for i in range(8)]
        self.input_coords = [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]]
        self.output_coords = [[0.0, 1.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


@pytest.fixture
def parity3_problem():
    return _Parity3()


def _config(algo_key, section_key, activation):
    return {
        "algorithm_params": {
            algo_key: {
                "population_size": 60,
                "substrate": {
                    "input_coords": [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                    "output_coords": [[0.0, 1.0]],
                },
                section_key: {
                    "initial_depth": 0,
                    "max_depth": 2,
                    "variance_threshold": 0.03,
                    "dynamic_functions": {
                        "mode": "global",
                        "hidden_activation": activation,
                    },
                    # restores HMR's extra per-generation randkey split in EMR
                    "extra_randkey_split": True,
                    "pop_size": 60,
                },
                "neat_species": {
                    "compatibility_threshold": 2.5,
                    "max_stagnation": 40,
                },
            },
        },
    }


def _trajectory(algo_cls, algo_key, section_key, activation, problem, seed, gens=3):
    algo = algo_cls()
    cfg = algo.create_config(_config(algo_key, section_key, activation))
    state = algo.initialize(cfg, problem, seed=seed)
    fits = []
    for _ in range(gens):
        state, metrics = algo.run_generation(state, problem)
        fits.append(float(metrics.best_fitness))
    randkey = np.asarray(state.randkey) if hasattr(state, "randkey") else None
    return fits, randkey


@pytest.mark.parametrize("activation", ["tanh", "sin"])
def test_emr_bit_reproduces_frozen_hmr(activation, parity3_problem):
    """EMR (extra_randkey_split=True) == frozen HMR, full trajectory + randkey."""
    h_fits, h_key = _trajectory(
        HMRDynamicFunctions, "hmrhyperneat", "hmr_hyperneat", activation,
        parity3_problem, seed=42,
    )
    e_fits, e_key = _trajectory(
        EMRHyperNEAT, "emrhyperneat", "emr_hyperneat", activation,
        parity3_problem, seed=42,
    )
    assert h_fits == e_fits, (
        f"{activation}: trajectories diverge\n  HMR: {h_fits}\n  EMR: {e_fits}"
    )
    if h_key is not None and e_key is not None:
        assert np.array_equal(h_key, e_key), f"{activation}: final randkey differs"
