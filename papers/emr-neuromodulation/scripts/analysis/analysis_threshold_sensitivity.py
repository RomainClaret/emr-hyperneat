#!/usr/bin/env python3
"""E-S13: Convergence Threshold Sensitivity Analysis.

Re-analyzes existing continuous 2D experiment results (E28, E29, E34, E35)
at multiple convergence thresholds to determine if the 0% convergence at ≥98%
is a threshold artifact or genuine failure.

No new experiments, analysis only.

Thresholds tested: {0.80, 0.85, 0.90, 0.95, 0.98}

Usage:
    python papers/emr-neuromodulation/analysis_threshold_sensitivity.py
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# ============================================================================
# Result directories to analyze
# ============================================================================

BASE = Path(__file__).resolve().parents[2] / 'results'

EXPERIMENTS = {
    'E28 Synthetic 2D (1-layer)': BASE / 'e28_synthetic_2d',
    'E29 Synthetic 2D (depth)': BASE / 'e29_synthetic_2d_depth',
    'E34 Gaussian XOR': BASE / 'e34_gaussian_xor',
    'E35 Synthetic 2D (Pop=1500)': BASE / 'e35_synthetic_2d_pop1500',
    'Continuous Analogue': BASE / 'continuous_analogue',
}

THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98]


# ============================================================================
# Analysis
# ============================================================================

def analyze_directory(results_dir: Path, thresholds: List[float]) -> Dict:
    """Analyze all results in a directory at multiple thresholds."""
    if not results_dir.exists():
        return None

    files = sorted(results_dir.glob('*.json'))
    if not files:
        return None

    # Group by condition
    conditions = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        condition = data.get('condition', 'unknown')
        if condition not in conditions:
            conditions[condition] = []

        min_acc = data.get('min_task_accuracy', 0.0)
        per_task = data.get('per_task_fitness', {})

        # Find best min_acc across fitness history
        best_min_acc = min_acc
        for entry in data.get('fitness_history', []):
            hist_min = entry.get('min_task_accuracy', 0.0)
            if hist_min > best_min_acc:
                best_min_acc = hist_min

        conditions[condition].append({
            'seed': data.get('seed'),
            'min_task_accuracy': min_acc,
            'best_min_accuracy_history': best_min_acc,
            'per_task_fitness': per_task,
        })

    # Compute convergence at each threshold
    results = {}
    for condition, runs in sorted(conditions.items()):
        threshold_results = {}
        for thresh in thresholds:
            converged = sum(1 for r in runs if r['min_task_accuracy'] >= thresh)
            total = len(runs)
            threshold_results[f'{thresh:.2f}'] = {
                'converged': converged,
                'total': total,
                'rate': converged / total if total > 0 else 0.0,
            }
        results[condition] = {
            'n_runs': len(runs),
            'thresholds': threshold_results,
            'min_acc_distribution': {
                'mean': np.mean([r['min_task_accuracy'] for r in runs]),
                'median': np.median([r['min_task_accuracy'] for r in runs]),
                'std': np.std([r['min_task_accuracy'] for r in runs]),
                'min': min(r['min_task_accuracy'] for r in runs),
                'max': max(r['min_task_accuracy'] for r in runs),
            },
        }

    return results


def print_results():
    """Print threshold sensitivity analysis."""
    print("=" * 90)
    print("E-S13: Convergence Threshold Sensitivity Analysis")
    print("=" * 90)

    all_results = {}

    for exp_name, results_dir in EXPERIMENTS.items():
        results = analyze_directory(results_dir, THRESHOLDS)
        if results is None:
            print(f"\n{exp_name}: NO RESULTS FOUND at {results_dir}")
            continue

        all_results[exp_name] = results

        print(f"\n{'─' * 90}")
        print(f"{exp_name} ({results_dir.name})")
        print(f"{'─' * 90}")

        # Header
        thresh_header = ''.join(f'  ≥{t:.2f}' for t in THRESHOLDS)
        print(f"  {'Condition':<35} {'N':>4} {thresh_header}   min_acc μ±σ")
        print(f"  {'─' * 85}")

        for condition, cond_data in sorted(results.items()):
            n = cond_data['n_runs']
            dist = cond_data['min_acc_distribution']

            rates = []
            for thresh in THRESHOLDS:
                t_data = cond_data['thresholds'][f'{thresh:.2f}']
                rate_str = f"{t_data['converged']:>2}/{t_data['total']:<2}"
                rates.append(rate_str)

            rates_str = '  '.join(rates)
            print(f"  {condition:<35} {n:>4} {rates_str}   "
                  f"{dist['mean']:.3f}±{dist['std']:.3f}")

    # Summary: aggregate across all experiments
    if all_results:
        print(f"\n{'=' * 90}")
        print("AGGREGATE SUMMARY: Convergence rates across all continuous experiments")
        print(f"{'=' * 90}")

        for thresh in THRESHOLDS:
            total_conv = 0
            total_runs = 0
            for exp_results in all_results.values():
                for cond_data in exp_results.values():
                    t_data = cond_data['thresholds'][f'{thresh:.2f}']
                    total_conv += t_data['converged']
                    total_runs += t_data['total']
            rate = total_conv / total_runs if total_runs > 0 else 0.0
            print(f"  Threshold ≥{thresh:.2f}: {total_conv}/{total_runs} "
                  f"({rate * 100:.1f}%)")

    # Save results to JSON
    output_path = BASE / 'threshold_sensitivity_analysis.json'
    output = {}
    for exp_name, results in all_results.items():
        # Convert numpy to float
        for condition, cond_data in results.items():
            dist = cond_data['min_acc_distribution']
            for key in dist:
                dist[key] = float(dist[key])
        output[exp_name] = results

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    print_results()
