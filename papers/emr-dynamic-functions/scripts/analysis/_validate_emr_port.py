#!/usr/bin/env python3
"""Reproducibility check: EMR equals the HMR engine bit for bit on the feedforward
per-node-activation path used for the paper's E1/E2/E7/E8/E9 experiments.

Background
----------
Per-node activation function evolution runs on EMR-HyperNEAT, the engine this repository ships.
HMR-HyperNEAT is the engine that produced the published results; release 0.1.0 ships it as
a frozen module and 0.2.0 does not. EMR and HMR are identical in substrate discovery, RNG
initialization, AND the W1/W2 feedforward evaluation (global activation mode) once EMR runs HMR's
level-0 variance rule and its extra `jax.random.split` per generation before the ask
(`variance_rule: legacy`, `extra_randkey_split: True`), as the EMR versions of this paper's runners
pin them. The feedforward runners also pin `pop_size = 150`, because HMR's pop-size wiring fixed the
population at 150 regardless of the nominal value, while EMR honors the config.

What is compared
----------------
HMR's side comes from hmr_trajectories.json, next to this script: HMR's trajectories of the 18 cells
(six global activation functions x seeds 42-44, Parity-2, population 150, 12 generations), recorded
on release 0.1.0 by its own version of this script, with the command the file gives. The values are
kept per CPU architecture (arm64, x86_64), with the release, its commit, the jax, jaxlib, numpy and
Python versions and the XLA_FLAGS conditions they were recorded under; the two architectures differ
in the float rounding of one cell. A cell is IDENTICAL when every generation's best fitness and the
final random key equal HMR's bit for bit. On an architecture without recorded values the check
stops; under other versions or another XLA_FLAGS it says so and still compares.

Scope
-----
Reproduced bit-for-bit on EMR: the feedforward (global activation) path -> E1/E2/E7/E8/E9.
NOT reproduced: E3 (cppn_output aggregation) and E4/E5 (recurrence), whose EMR versions run EMR's
own forwards (e.g. EMR derives the per-node activation index from the node's self-connection while
HMR aggregates the activation channel over the input->position connections). See
../../REPRODUCIBILITY.md.

Writes results/validate_emr_port_emr/validation.json: per cell, both trajectories and whether they
are equal. --reference reads another recording (a negative control perturbs one value).

Run, from the paper folder:
    JAX_PLATFORMS=cpu python scripts/analysis/_validate_emr_port.py
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from classification_problems import ParityProblem  # noqa: E402

from emr_hyperneat.emrhyperneat import (  # noqa: E402
    EMRHyperNEAT,
)

N_BITS = 2          # XOR / Parity-2 (fast; the per-gen-split fix is task-independent)
POP = 150
GENS = 12
FUNCS = [('sin', 4), ('burst', 11), ('resonator', 12), ('gauss', 5), ('tanh', 0), ('sigmoid', 1)]
SEEDS = (42, 43, 44)
REFERENCE = HERE / 'hmr_trajectories.json'
OUTPUT_DIR = HERE.parents[1] / 'results' / 'validate_emr_port_emr'


def _config(fn, idx):
    problem = ParityProblem(n_bits=N_BITS)
    n_in = problem.input_shape[0]
    ic = [(-1.0 + 2.0 * i / max(n_in - 1, 1) if n_in > 1 else 0.0, -1.0) for i in range(n_in)]
    oc = [(0.0, 1.0)]
    return {'algorithm_params': {'emrhyperneat': {
        'emr_hyperneat': {'initial_depth': 0, 'max_depth': 2, 'variance_threshold': 0.03,
                          # HMR's split before each ask and its level-0 subdivision rule, which
                          # HMR hard-wired; EMR's defaults are no split and 'canonical'.
                          'extra_randkey_split': True,
                          'variance_rule': 'legacy',
                          'dynamic_functions': {'mode': 'global', 'hidden_activation': fn,
                                                'palette': [idx],
                                                'palette_evolution': {'enabled': False}}},
        'substrate': {'input_coords': ic, 'output_coords': oc},
        'neat': {'pop_size': POP, 'species_size': 10},
    }}}


def _trajectory(fn, idx, seed):
    """Every generation's best fitness, unrounded, and the final random key of one EMR run."""
    algo = EMRHyperNEAT()
    nc = algo.create_config(_config(fn, idx))
    problem = ParityProblem(n_bits=N_BITS)
    state = algo.initialize(nc, problem, seed=seed)
    fits = []
    for _ in range(GENS):
        state, m = algo.run_generation(state, problem)
        fits.append(float(m.best_fitness))
    return fits, [int(k) for k in np.asarray(state.state_dict['randkey']).tolist()]


def _architecture():
    machine = platform.machine().lower()
    return {'aarch64': 'arm64', 'amd64': 'x86_64'}.get(machine, machine)


def _record(reference, arch, xla_flags):
    """HMR's record for this architecture, the one made under this XLA_FLAGS when there is one."""
    records = reference['platforms'].get(arch)
    if not records:
        return None
    return next((r for r in records if xla_flags in r['xla_flags']), records[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reference', type=Path, default=REFERENCE,
                        help="HMR's recorded trajectories (default: hmr_trajectories.json)")
    args = parser.parse_args()
    import jax
    import jaxlib

    reference = json.loads(args.reference.read_text())
    arch = _architecture()
    xla_flags = os.environ.get('XLA_FLAGS', '')
    record = _record(reference, arch, xla_flags)
    if record is None:
        print(f"No HMR trajectories recorded for {arch} in {args.reference.name} (recorded: "
              f"{', '.join(sorted(reference['platforms']))}); record them with its command.")
        sys.exit(2)
    here = {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__}
    notes = [f"{name} {value} here, {record[name]} recorded" for name, value in here.items()
             if record[name] != value]
    if xla_flags not in record['xla_flags']:
        notes.append(f"XLA_FLAGS {xla_flags!r} here, recorded under {record['xla_flags']}")
    print(f"EMR (variance_rule legacy, extra_randkey_split) vs HMR recorded on release "
          f"{record['release']}, {arch}  |  Parity-{N_BITS}, pop={POP}, {GENS} gens, "
          f"{len(FUNCS)} funcs x {len(SEEDS)} seeds\n", flush=True)
    for note in notes:
        print(f"  note: {note}", flush=True)
    cells = []
    for fn, idx in FUNCS:
        for seed in SEEDS:
            hmr = record['cells'][f'{fn}/{seed}']
            h_fits = [float.fromhex(value) for value in hmr['best_fitness']]
            e_fits, e_key = _trajectory(fn, idx, seed)
            ok = e_fits == h_fits and e_key == hmr['randkey']
            cells.append({'function': fn, 'seed': seed, 'identical': ok,
                          'hmr_best_fitness': h_fits, 'emr_best_fitness': e_fits,
                          'hmr_randkey': hmr['randkey'], 'emr_randkey': e_key})
            first = next((g for g, (h, e) in enumerate(zip(h_fits, e_fits)) if h != e), None)
            detail = ('' if ok else f"  generation {first + 1}: HMR={h_fits[first]!r} "
                      f"EMR={e_fits[first]!r}" if first is not None else "  random key differs")
            print(f"  {fn:9s} seed{seed}: {'IDENTICAL' if ok else 'DIFFER'}{detail}", flush=True)
    identical = sum(cell['identical'] for cell in cells)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {'architecture': arch, 'xla_flags': xla_flags, 'versions': here, 'notes': notes,
              'reference': args.reference.name, 'recorded': {k: record[k] for k in record
                                                             if k != 'cells'},
              'identical': identical, 'cells': cells}
    with open(OUTPUT_DIR / 'validation.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(f"\n==> {identical}/{len(cells)} cells bit-identical to HMR's recorded trajectories "
          f"(every generation's best fitness + the final randkey)", flush=True)
    sys.exit(0 if identical == len(cells) else 1)


if __name__ == "__main__":
    main()
