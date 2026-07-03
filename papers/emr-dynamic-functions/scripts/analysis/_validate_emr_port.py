#!/usr/bin/env python3
"""Reproducibility check: EMR bit-reproduces the HMR module on the feedforward
per-node-activation path used for the paper's E1/E2/E7/E8/E9 experiments.

Background
----------
Per-node activation function evolution runs on EMR-HyperNEAT (the framework going forward).
HMR-HyperNEAT is the frozen module that produced the published results. EMR and HMR are
identical in substrate discovery, RNG initialization, AND the W1/W2 feedforward evaluation
(global activation mode). The ONLY difference was one extra `jax.random.split` per generation
that HMR performs before the ask (HMR's `extra_randkey_split`, default True). The opt-in EMR
config flag `emr_hyperneat.extra_randkey_split` restores that split, giving bit-identical
per-seed trajectories. The feedforward runners also pin `pop_size = 150`, because HMR's
pop-size wiring fixed the population at 150 regardless of the nominal value, while EMR honors
the config.

Scope
-----
Reproduced bit-for-bit on EMR: the feedforward (global activation) path -> E1/E2/E7/E8/E9.
NOT reproduced on EMR (kept on the HMR module): E3 (cppn_output aggregation) and E4/E5
(recurrence). EMR reimplemented those forwards -- e.g. it derives the per-node activation
index from the node's self-connection while HMR aggregates the activation channel over the
input->position connections -- so they do not bit-reproduce. See ./REPRODUCIBILITY.md.

Run:
    JAX_PLATFORM_NAME=cpu python papers/emr-dynamic-functions/scripts/analysis/_validate_emr_port.py
"""

import sys

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classification_problems import ParityProblem  # noqa: E402

from emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions import (  # noqa: E402
    HMRHyperNEAT,
)
from emr_hyperneat.emrhyperneat import (  # noqa: E402
    EMRHyperNEAT,
)

N_BITS = 2          # XOR / Parity-2 (fast; the per-gen-split fix is task-independent)
POP = 150
GENS = 12
FUNCS = [('sin', 4), ('burst', 11), ('resonator', 12), ('gauss', 5), ('tanh', 0), ('sigmoid', 1)]
SEEDS = (42, 43, 44)


def _config(top, mid, fn, idx):
    problem = ParityProblem(n_bits=N_BITS)
    n_in = problem.input_shape[0]
    ic = [(-1.0 + 2.0 * i / max(n_in - 1, 1) if n_in > 1 else 0.0, -1.0) for i in range(n_in)]
    oc = [(0.0, 1.0)]
    return {'algorithm_params': {top: {
        mid: {'initial_depth': 0, 'max_depth': 2, 'variance_threshold': 0.03,
              # HMR default is True; EMR default is False, so we set it explicitly on both.
              'extra_randkey_split': True,
              'dynamic_functions': {'mode': 'global', 'hidden_activation': fn,
                                    'palette': [idx], 'palette_evolution': {'enabled': False}}},
        'substrate': {'input_coords': ic, 'output_coords': oc},
        'neat': {'pop_size': POP, 'species_size': 10},
    }}}


def _trajectory(AlgoClass, top, mid, fn, idx, seed):
    algo = AlgoClass()
    nc = algo.create_config(_config(top, mid, fn, idx))
    problem = ParityProblem(n_bits=N_BITS)
    state = algo.initialize(nc, problem, seed=seed)
    fits = []
    for _ in range(GENS):
        state, m = algo.run_generation(state, problem)
        fits.append(round(float(m.best_fitness), 6))
    return fits, np.asarray(state.state_dict['randkey'])


def main():
    print(f"EMR(extra_randkey_split=True) vs HMR module  |  Parity-{N_BITS}, pop={POP}, "
          f"{GENS} gens, {len(FUNCS)} funcs x {len(SEEDS)} seeds\n", flush=True)
    identical = 0
    total = 0
    for fn, idx in FUNCS:
        for seed in SEEDS:
            total += 1
            h, hrk = _trajectory(HMRHyperNEAT, 'hmrhyperneat', 'hmr_hyperneat', fn, idx, seed)
            e, erk = _trajectory(EMRHyperNEAT, 'emrhyperneat', 'emr_hyperneat', fn, idx, seed)
            ok = (h == e) and np.array_equal(hrk, erk)
            identical += ok
            print(f"  {fn:9s} seed{seed}: {'IDENTICAL' if ok else 'DIFFER'}"
                  f"{'' if ok else f'  HMR={h} EMR={e}'}", flush=True)
    print(f"\n==> {identical}/{total} cells bit-identical (full trajectory + randkey)", flush=True)
    sys.exit(0 if identical == total else 1)


if __name__ == "__main__":
    main()
