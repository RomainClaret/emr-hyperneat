#!/usr/bin/env python3
"""Analysis script for strengthening experiments (Experiments 13-16).

Loads all result JSONs from papers/emr-neuromodulation/results/strengthening/,
computes per-condition statistics, and outputs a formatted summary.

Usage:
    python papers/emr-neuromodulation/analysis_strengthening.py
"""

import json
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'strengthening'


def load_condition(base_dir: Path, prefix: str) -> dict:
    """Load all results matching a prefix, return summary dict."""
    files = sorted(base_dir.glob(f'{prefix}_seed*.json'))
    if not files:
        # Try alternate pattern (e.g., random_no_inv_set0_seed0)
        files = sorted(base_dir.glob(f'{prefix}*.json'))

    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))

    converged = [r for r in results if r.get('converged', False)]
    gens = [r['convergence_gen'] for r in converged if r.get('convergence_gen') is not None]
    min_fs = [r.get('individual_min_fitness', 0) for r in results]

    summary = {
        'total': len(results),
        'converged': len(converged),
        'rate': len(converged) / len(results) if results else 0,
        'gens': sorted(gens),
        'avg_min_f': float(np.mean(min_fs)) if min_fs else 0,
    }

    if gens:
        summary['median_gen'] = float(np.median(gens))
        summary['mean_gen'] = float(np.mean(gens))
        summary['std_gen'] = float(np.std(gens, ddof=1)) if len(gens) > 1 else 0
        summary['range'] = (min(gens), max(gens))
        if len(gens) > 1:
            summary['ci_95'] = (
                float(np.percentile(gens, 2.5)),
                float(np.percentile(gens, 97.5)),
            )

    return summary


def fisher_exact_test(a_conv: int, a_total: int, b_conv: int, b_total: int) -> float:
    """Fisher's exact test for convergence rate comparison."""
    table = [
        [a_conv, a_total - a_conv],
        [b_conv, b_total - b_conv],
    ]
    _, p = scipy_stats.fisher_exact(table)
    return p


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_random_nt():
    """Experiment 13: Random NT vectors."""
    print_header("EXPERIMENT 13: Random NT Vectors (600 runs)")

    exp_dir = RESULTS_DIR / 'random_nt'

    # No inversion
    no_inv_total = 0
    no_inv_conv = 0
    for i in range(10):
        files = sorted(exp_dir.glob(f'random_no_inv_set{i}_seed*.json'))
        for f in files:
            d = json.load(open(f))
            no_inv_total += 1
            if d.get('converged', False):
                no_inv_conv += 1

    print(f"\n  No inversion (ACh=1.0 for all): {no_inv_conv}/{no_inv_total} "
          f"({100*no_inv_conv/no_inv_total:.1f}%)")

    # With inversion
    with_inv_total = 0
    with_inv_conv = 0
    per_set = {}
    all_gens = []

    for i in range(10):
        files = sorted(exp_dir.glob(f'random_with_inv_set{i}_seed*.json'))
        conv = 0
        set_gens = []
        for f in files:
            d = json.load(open(f))
            with_inv_total += 1
            if d.get('converged', False):
                conv += 1
                with_inv_conv += 1
                if d.get('convergence_gen') is not None:
                    set_gens.append(d['convergence_gen'])
                    all_gens.append(d['convergence_gen'])
        per_set[i] = {'conv': conv, 'total': len(files), 'gens': set_gens}
        rate = 100 * conv / len(files) if files else 0
        med = f", median {np.median(set_gens):.0f} gen" if set_gens else ""
        print(f"  With inversion set {i}: {conv}/{len(files)} ({rate:.1f}%){med}")

    rate = 100 * with_inv_conv / with_inv_total
    print(f"\n  With inversion total: {with_inv_conv}/{with_inv_total} ({rate:.1f}%)")
    if all_gens:
        print(f"  Converged median: {np.median(all_gens):.0f}, range [{min(all_gens)}, {max(all_gens)}]")

    # Fisher's exact: no_inv vs with_inv
    p = fisher_exact_test(no_inv_conv, no_inv_total, with_inv_conv, with_inv_total)
    print(f"\n  Fisher's exact (no_inv vs with_inv): p = {p:.4e}")

    # Summary
    sets_any = sum(1 for v in per_set.values() if v['conv'] > 0)
    sets_100 = sum(1 for v in per_set.values() if v['conv'] == v['total'])
    print(f"  Sets with any success: {sets_any}/10")
    print(f"  Sets with 100% success: {sets_100}/10")

    # Fisher's exact: hand-designed (30/30) vs best random with_inv (set 3: 30/30)
    p2 = fisher_exact_test(30, 30, with_inv_conv, with_inv_total)
    print(f"  Fisher's exact (hand-designed 30/30 vs random with_inv overall): p = {p2:.4e}")

    return {
        'no_inv': {'conv': no_inv_conv, 'total': no_inv_total},
        'with_inv': {'conv': with_inv_conv, 'total': with_inv_total},
        'per_set': per_set,
    }


def analyze_task_scaling():
    """Experiment 14: Task scaling to 7/8/10 tasks."""
    print_header("EXPERIMENT 14: Task Scaling (90 runs)")

    exp_dir = RESULTS_DIR / 'task_scaling'

    for n_tasks in [7, 8, 10]:
        files = sorted(exp_dir.glob(f'{n_tasks}task_seed*.json'))
        conv = 0
        min_fs = []
        per_task_avgs = defaultdict(list)

        for f in files:
            d = json.load(open(f))
            if d.get('converged', False):
                conv += 1
            min_fs.append(d.get('individual_min_fitness', 0))
            for task, fit in d.get('per_task_fitness', {}).items():
                per_task_avgs[task].append(fit)

        rate = 100 * conv / len(files) if files else 0
        avg_mf = np.mean(min_fs) if min_fs else 0
        print(f"\n  {n_tasks}-task: {conv}/{len(files)} ({rate:.1f}%), avg min_f={avg_mf:.3f}")

        # Per-task breakdown
        for task in sorted(per_task_avgs.keys()):
            avg = np.mean(per_task_avgs[task])
            perfect = sum(1 for v in per_task_avgs[task] if v >= 0.95)
            print(f"    {task}: avg={avg:.3f}, >=95%: {perfect}/{len(per_task_avgs[task])}")

    # Fisher's exact: 5-task (30/30) vs 7-task (0/30)
    p = fisher_exact_test(30, 30, 0, 30)
    print(f"\n  Fisher's exact (5-task 30/30 vs 7-task 0/30): p = {p:.4e}")


def analyze_sigma_sweep():
    """Experiment 15: Modulation strength sweep."""
    print_header("EXPERIMENT 15: Modulation Strength Sweep (180 runs)")

    exp_dir = RESULTS_DIR / 'sigma_sweep'
    strengths = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    results = {}
    for s in strengths:
        s_str = f"{s:.1f}".replace('.', 'p')
        files = sorted(exp_dir.glob(f's{s_str}_seed*.json'))
        conv = 0
        gens = []
        for f in files:
            d = json.load(open(f))
            if d.get('converged', False):
                conv += 1
                if d.get('convergence_gen') is not None:
                    gens.append(d['convergence_gen'])

        rate = 100 * conv / len(files) if files else 0
        med = f"{np.median(gens):.0f}" if gens else '-'
        ci = ''
        if len(gens) > 1:
            ci = f", 95% CI [{np.percentile(gens, 2.5):.1f}, {np.percentile(gens, 97.5):.1f}]"
        rng = f", range [{min(gens)}, {max(gens)}]" if gens else ''
        print(f"  s={s:5.1f}: {conv}/{len(files)} ({rate:.1f}%), median {med}{ci}{rng}")
        results[s] = {'conv': conv, 'total': len(files), 'gens': gens}

    # Fisher's exact: s=0.5 vs s=5.0 (default)
    p = fisher_exact_test(results[0.5]['conv'], results[0.5]['total'],
                          results[5.0]['conv'], results[5.0]['total'])
    print(f"\n  Fisher's exact (s=0.5 vs s=5.0): p = {p:.4f}")

    return results


def analyze_deeper_substrate():
    """Experiment 16: Deeper substrate."""
    print_header("EXPERIMENT 16: Deeper Substrate (120 runs)")

    exp_dir = RESULTS_DIR / 'deeper_substrate'

    for depth in [4, 5, 6]:
        files = sorted(exp_dir.glob(f'depth{depth}_uniform_seed*.json'))
        conv = 0
        min_fs = []
        for f in files:
            d = json.load(open(f))
            if d.get('converged', False):
                conv += 1
            min_fs.append(d.get('individual_min_fitness', 0))

        rate = 100 * conv / len(files) if files else 0
        avg_mf = np.mean(min_fs) if min_fs else 0
        print(f"  depth={depth} uniform tanh: {conv}/{len(files)} ({rate:.1f}%), avg min_f={avg_mf:.3f}")

    # Per-head control at depth 6
    files = sorted(exp_dir.glob('depth6_perhead_seed*.json'))
    if files:
        conv = 0
        gens = []
        for f in files:
            d = json.load(open(f))
            if d.get('converged', False):
                conv += 1
                if d.get('convergence_gen') is not None:
                    gens.append(d['convergence_gen'])

        rate = 100 * conv / len(files) if files else 0
        med = f", median {np.median(gens):.0f}" if gens else ''
        rng = f", range [{min(gens)}, {max(gens)}]" if gens else ''
        print(f"  depth=6 per-head (control): {conv}/{len(files)} ({rate:.1f}%){med}{rng}")

        if conv > 0 and len(files) == 30:
            # Fisher's exact: uniform d=6 vs per-head d=6
            unif_files = sorted(exp_dir.glob('depth6_uniform_seed*.json'))
            unif_conv = sum(1 for f in unif_files if json.load(open(f)).get('converged', False))
            p = fisher_exact_test(unif_conv, len(unif_files), conv, len(files))
            print(f"  Fisher's exact (d=6 uniform vs d=6 per-head): p = {p:.4e}")
    else:
        print("  depth=6 per-head: NOT YET COMPLETE")


def total_runs():
    """Count total runs across all experiments."""
    total = 0
    for exp in ['random_nt', 'task_scaling', 'sigma_sweep', 'deeper_substrate']:
        exp_dir = RESULTS_DIR / exp
        total += len(list(exp_dir.glob('*.json')))
    return total


def main():
    print("=" * 70)
    print("  STRENGTHENING EXPERIMENTS — NEUROMODULATION PAPER")
    print(f"  Results directory: {RESULTS_DIR}")
    print("=" * 70)

    analyze_random_nt()
    analyze_task_scaling()
    analyze_sigma_sweep()
    analyze_deeper_substrate()

    t = total_runs()
    print_header(f"TOTAL: {t} runs across 4 experiments")
    print(f"  random_nt:         600")
    print(f"  task_scaling:       90")
    print(f"  sigma_sweep:       180")
    print(f"  deeper_substrate:  120")
    print(f"  TOTAL:             990")
    print(f"  Files found:       {t}")
    print(f"\n  Previous total:  3,849")
    print(f"  New total:       {3849 + 990} = 4,839")


if __name__ == '__main__':
    main()
