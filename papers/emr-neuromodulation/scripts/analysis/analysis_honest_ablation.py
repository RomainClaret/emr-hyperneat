#!/usr/bin/env python3
"""Analysis script for the honest neuromodulation ablation.

Flat NT [0.5,0.5,0.5] + correct ACh polarity + per-task activation.
Expected: 0/30, 75% plateau on all threshold tasks.
"""

import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'strengthening/honest_neuromod_ablation'
TASKS = ['xor', 'and', 'or', 'nand', 'nor']


def main():
    files = sorted(RESULTS_DIR.glob('honest_ablation_seed*.json'))
    if not files:
        print("No results found yet.")
        return

    converged = 0
    gens = []
    per_task_all = {t: [] for t in TASKS}

    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d.get('converged', False):
            converged += 1
            if d.get('convergence_gen') is not None:
                gens.append(d['convergence_gen'])
        for t in TASKS:
            per_task_all[t].append(d.get('per_task_fitness', {}).get(t, 0))

    n = len(files)
    print(f"{'='*60}")
    print(f"HONEST NEUROMOD ABLATION: {n} seeds")
    print(f"Flat NT [0.5,0.5,0.5] + correct ACh + per-task activation")
    print(f"{'='*60}")
    print(f"\nConverged: {converged}/{n} ({100*converged/n:.1f}%)")

    if gens:
        print(f"Convergence gen: median {np.median(gens):.0f}, range [{min(gens)}, {max(gens)}]")

    print(f"\nPer-task fitness (mean ± std):")
    for t in TASKS:
        vals = per_task_all[t]
        print(f"  {t:5s}: {np.mean(vals):.3f} ± {np.std(vals):.3f} "
              f"(min={np.min(vals):.3f}, max={np.max(vals):.3f})")

    # Check if pattern is consistent: XOR=1.0, all threshold=0.75
    xor_100 = sum(1 for v in per_task_all['xor'] if v >= 0.99)
    threshold_75 = sum(1 for t in ['and', 'or', 'nand', 'nor']
                       for v in per_task_all[t] if abs(v - 0.75) < 0.01)
    print(f"\nPattern check:")
    print(f"  XOR ≥ 0.99: {xor_100}/{n}")
    print(f"  Threshold tasks = 0.75 (±0.01): {threshold_75}/{4*n}")

    # Comparison with original ablation
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH ORIGINAL ABLATION")
    print(f"{'='*60}")
    print(f"{'Condition':<35} {'XOR':>8} {'AND':>8} {'OR':>8} {'NAND':>8} {'NOR':>8}")
    print(f"{'-'*75}")
    print(f"{'Original (flat NT, ACh=1.0)':<35} {'1.000':>8} {'0.500':>8} {'0.500':>8} {'0.500':>8} {'0.500':>8}")
    if n > 0:
        honest_str = {t: f"{np.mean(per_task_all[t]):.3f}" for t in TASKS}
        print(f"{'Honest (flat NT, correct ACh)':<35} "
              f"{honest_str['xor']:>8} {honest_str['and']:>8} {honest_str['or']:>8} "
              f"{honest_str['nand']:>8} {honest_str['nor']:>8}")
    print(f"{'Full system (task-specific NT)':<35} {'1.000':>8} {'1.000':>8} {'1.000':>8} {'1.000':>8} {'1.000':>8}")


if __name__ == '__main__':
    main()
