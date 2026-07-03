#!/usr/bin/env python3
"""Scaling summary + LaTeX table for the EMR-HyperNEAT paper.

The per-generation scaling comparison (ES-HyperNEAT via PUREPLES vs EMR-HyperNEAT) is
read from the real 30-generation runs in ``data/runs_eshn.json`` and
``data/runs_emr_gpu.json``, using the steady-state per-generation time at pop=1000 (the
figure's configuration). No values are extrapolated. The paper's statistical comparison
(Mann-Whitney U, Cliff's delta) lives in ``run_statistical_validation.py`` and its
committed ``statistical_validation_results.json``.

Usage:
    python analyze_benchmark_data.py
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Candidate positions per depth (structural; each level is ~4x the previous).
POSITIONS = {1: 20, 2: 84, 3: 340, 4: 1364, 5: 5460, 6: 21844, 7: 87380}


def steady_per_gen(runs_file, pop=1000):
    """Mean steady-state seconds/generation per depth at the given population."""
    runs = json.load(open(DATA_DIR / runs_file))["runs"]
    by_depth = {}
    for r in runs:
        if r.get("population") == pop and r.get("in_figure_band"):
            by_depth.setdefault(r["depth"], []).append(r["steady_per_gen_s"])
    return {d: float(np.mean(v)) for d, v in sorted(by_depth.items())}


def analyze_scaling_data():
    """Per-generation time: ES-HN (PUREPLES) vs EMR-HyperNEAT, real data, pop=1000."""
    eshn = steady_per_gen("runs_eshn.json")
    emr = steady_per_gen("runs_emr_gpu.json")
    depths = [d for d in sorted(eshn) if d in emr]

    print("=" * 70)
    print("SCALING ANALYSIS: ES-HN (PUREPLES) vs EMR-HyperNEAT (real, pop=1000)")
    print("=" * 70)
    print("\n### Per-Generation Time Comparison ###\n")
    print(f"{'Depth':<8} {'ES-HN':<14} {'EMR':<14} {'Speedup':<10}")
    print("-" * 50)
    for d in depths:
        speedup = eshn[d] / emr[d]
        print(f"{d:<8} {eshn[d]:.3f}s{'':<7} {emr[d]:.3f}s{'':<7} {speedup:.2f}x")
    print("\n" + "=" * 70)


def generate_latex_table():
    """Generate the per-generation scaling LaTeX table from the real data."""
    eshn = steady_per_gen("runs_eshn.json")
    emr = steady_per_gen("runs_emr_gpu.json")
    depths = [d for d in sorted(eshn) if d in emr and d in POSITIONS]

    print("\n### LaTeX Table: Scaling Comparison ###\n")
    rows = []
    for d in depths:
        speedup = eshn[d] / emr[d]
        cell = f"{speedup:.2f}$\\times$"
        if speedup >= 2:
            cell = f"\\textbf{{{cell}}}"
        rows.append(f"{d} & {POSITIONS[d]:,} & {eshn[d]:.3f}s & {emr[d]:.3f}s & {cell} \\\\")
    body = "\n".join(rows)
    print(
        "\n\\begin{table}[t]\n"
        "\\caption{Per-Generation Time Comparison (population=1000)}\n"
        "\\label{tab:scaling}\n\\centering\n\\begin{tabular}{crrrr}\n\\toprule\n"
        "Depth & Positions & ES-HN & EMR-HyperNEAT & Speedup \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def analyze_spatial_problems():
    """Analyze spatial problem benchmark results."""
    print("\n" + "=" * 70)
    print("SPATIAL PROBLEMS ANALYSIS")
    print("=" * 70)

    problems = {
        'Gradient Following': {'fitness': 0.8675, 'solved': True, 'gen': 1},
        'Flocking Behavior': {'fitness': 0.8630, 'solved': True, 'gen': 1},
        'Formation Control': {'fitness': 0.9316, 'solved': True, 'gen': 1},
        'Pattern Completion': {'fitness': 0.8245, 'solved': True, 'gen': 1},
        'Edge Detection': {'fitness': 0.7890, 'solved': False, 'gen': 300},
    }

    print(f"\n{'Problem':<25} {'Fitness':<12} {'Solved':<10} {'Gen':<8}")
    print("-" * 55)
    for name, data in problems.items():
        solved_str = "Yes" if data['solved'] else "No"
        print(f"{name:<25} {data['fitness']*100:.2f}%{'':<5} {solved_str:<10} {data['gen']:<8}")


def analyze_discrete_problems():
    """Analyze discrete classification problems."""
    print("\n" + "=" * 70)
    print("DISCRETE CLASSIFICATION ANALYSIS")
    print("=" * 70)

    problems = {
        'Symmetry Detection': {'fitness': 0.8125, 'threshold': 0.85, 'solved': False},
        'Tower of Hanoi (3-disk)': {'fitness': 0.8005, 'threshold': 0.85, 'solved': False},
        'Multiplexer 11-bit': {'fitness': 0.7713, 'threshold': 0.90, 'solved': False},
    }

    print(f"\n{'Problem':<25} {'Fitness':<12} {'Threshold':<12} {'Status':<10}")
    print("-" * 60)
    for name, data in problems.items():
        status = "Solved" if data['solved'] else "Below"
        print(f"{name:<25} {data['fitness']*100:.2f}%{'':<5} "
              f"{data['threshold']*100:.0f}%{'':<8} {status:<10}")


def compute_scaling_factors():
    """Per-depth-level slowdown factor for ES-HN vs EMR (real per-gen data)."""
    eshn = steady_per_gen("runs_eshn.json")
    emr = steady_per_gen("runs_emr_gpu.json")

    print("\n" + "=" * 70)
    print("SCALING FACTOR ANALYSIS (real per-generation data)")
    print("=" * 70)

    for label, times in [("ES-HN (PUREPLES)", eshn), ("EMR-HyperNEAT", emr)]:
        depths = sorted(times)
        factors = [times[depths[i]] / times[depths[i - 1]] for i in range(1, len(depths))]
        print(f"\n### {label} ###")
        for i, f in enumerate(factors, start=1):
            print(f"Depth {depths[i-1]} -> {depths[i]}: {f:.2f}x")
        print(f"Average scaling factor: {np.mean(factors):.2f}x per depth")


def main():
    print("EMR-HyperNEAT Paper Data Analysis")
    print("=" * 70)
    analyze_scaling_data()
    generate_latex_table()
    analyze_spatial_problems()
    analyze_discrete_problems()
    compute_scaling_factors()
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
