#!/usr/bin/env python3
"""Extended-generation confirmation on the INDIRECT (EMR-HyperNEAT) encoding.

Reviewer 4 (ALIFE 2026) asked whether the uniform-tanh multi-task barrier survives a
10x generation budget. The HEADLINE 0/54 barrier in this paper is the *indirect*
EMR-HyperNEAT (CPPN-generated substrate) encoding -- NOT the direct-encoded MLP control
(which has a much weaker barrier: ~76.7% at 5-task). The longest prior indirect run is
the 200-generation component ablation (still 0/30, XOR plateaued at 0.75).

This script extends the indirect uniform-tanh 5-task condition to 1000 generations (a
literal 10x of the 100-generation baseline) using the SAME pipeline as the paper's main
results: multihead_palette_neuromodulation.run_multihead_palette_experiment with
palette_mode='uniform' (all tanh), aggregation='product', Schema-B NT, max_depth=4.

Run as a quick 10-seed signal first (~12 h). Expected: 0/10 converged with XOR stuck at
0.75 if the barrier holds. Reports the true outcome either way.

Per-seed JSON saves + resume (skip completed seeds). Run ONE process to avoid OOM.

Usage (from repo root):
    JAX_PLATFORM_NAME=cpu python -u \
        papers/emr-neuromodulation/scripts/runners/benchmark_extended_generations_indirect.py \
        --generations 1000 --seeds 10
    ... --summary
"""

import argparse
import json
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]  # papers/emr-neuromodulation/<file> -> repo root
# Make the EMR-HyperNEAT framework + indirect pipeline importable.
import sys
sys.path.insert(0, str(_REPO / 'experiments' / 'neuromodulation'))
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO))

from multihead_palette_neuromodulation import run_multihead_palette_experiment

TASKS = ['xor', 'and', 'or', 'nand', 'nor']
# 30 reproducible seeds.
# Avoids seed 71 (known OOM).
SEEDS = [42, 123, 456, 789, 1000, 111, 222, 333, 444, 555,
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
RESULTS_DIR = _HERE.parent / 'results' / 'extended_generations_indirect'


def run_one(seed: int, generations: int, population: int):
    t0 = time.time()
    result = run_multihead_palette_experiment(
        task_names=TASKS,
        palette_mode='uniform',      # all tanh -- the barrier condition
        blend_mode='fixed',
        aggregation='product',       # product fitness, as in the main 5-task benchmark
        seed=seed,
        generations=generations,
        population=population,
        max_depth=4,
        success_threshold=0.98,      # matches the indirect component ablation
        verbose=False,
    )
    ptf = {k: float(v) for k, v in result.per_task_fitness.items()}
    return {
        'experiment': 'extended_generations_indirect_10x',
        'encoding': 'EMR-HyperNEAT (indirect, CPPN substrate)',
        'condition': 'uniform_tanh',
        'task_set': TASKS,
        'n_tasks': 5,
        'seed': seed,
        'generations_max': generations,
        'population': population,
        'max_depth': 4,
        'success_threshold': 0.98,
        'nt_schema': 'B',
        'converged': bool(result.converged),
        'convergence_gen': result.convergence_gen,
        'individual_min_fitness': float(result.individual_min_fitness),
        'per_task_fitness': ptf,
        'total_generations': getattr(result, 'total_generations', generations),
        'runtime_seconds': time.time() - t0,
    }


def summarize(results_dir: Path, generations: int):
    files = sorted(results_dir.glob(f'uniform_tanh_5task_g{generations}_seed*.json'))
    if not files:
        print(f"  no results for g={generations}")
        return
    conv, xor_accs, gens = 0, [], []
    for f in files:
        d = json.load(open(f))
        if d.get('converged'):
            conv += 1
            if d.get('convergence_gen') is not None:
                gens.append(d['convergence_gen'])
        if d.get('per_task_fitness', {}).get('xor') is not None:
            xor_accs.append(d['per_task_fitness']['xor'])
    n = len(files)
    print(f"  INDIRECT uniform-tanh 5-task @ {generations} gens: {conv}/{n} converged "
          f"({100*conv/n:.1f}%)", end='')
    if gens:
        print(f" | conv gens {gens}", end='')
    if xor_accs:
        import statistics
        print(f" | XOR mean={statistics.mean(xor_accs):.3f} (min {min(xor_accs):.3f})", end='')
    print()


def main():
    p = argparse.ArgumentParser(description='Indirect (EMR-HyperNEAT) 10x extended-generation confirmation')
    p.add_argument('--generations', type=int, default=1000, help='Max generations (default 1000 = 10x)')
    p.add_argument('--seeds', type=int, default=10, help='Number of seeds (default 10)')
    p.add_argument('--population', type=int, default=750)
    p.add_argument('--summary', action='store_true')
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.summary:
        summarize(RESULTS_DIR, args.generations)
        return

    seeds = SEEDS[:args.seeds]
    print(f"{'='*68}")
    print(f"INDIRECT (EMR-HyperNEAT) uniform-tanh 5-task | {len(seeds)} seeds | {args.generations} gens")
    print(f"  Expect 0/{len(seeds)} converged, XOR plateau at 0.75 if barrier holds")
    print(f"{'='*68}")

    t_all = time.time()
    for seed in seeds:
        fname = RESULTS_DIR / f'uniform_tanh_5task_g{args.generations}_seed{seed}.json'
        if fname.exists() and fname.stat().st_size > 0:
            print(f"  skip existing: {fname.name}")
            continue
        print(f"\n  Seed {seed} ...", flush=True)
        try:
            rec = run_one(seed, args.generations, args.population)
        except Exception as e:
            print(f"  ERROR seed {seed}: {e}")
            continue
        with open(fname, 'w') as f:
            json.dump(rec, f, indent=2)
        xor = rec['per_task_fitness'].get('xor')
        status = f"CONVERGED gen {rec['convergence_gen']}" if rec['converged'] else "NOT CONVERGED"
        print(f"  -> {status} | min={rec['individual_min_fitness']:.3f} XOR={xor:.3f} "
              f"({rec['runtime_seconds']:.0f}s)", flush=True)

    print(f"\nTotal: {(time.time()-t_all)/3600:.2f} h")
    summarize(RESULTS_DIR, args.generations)


if __name__ == '__main__':
    main()
