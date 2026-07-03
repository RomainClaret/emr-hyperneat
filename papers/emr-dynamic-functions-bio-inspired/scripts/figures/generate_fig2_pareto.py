#!/usr/bin/env python3
"""Generate fig2_pareto (speed-reliability Pareto frontier) for the bio-inspired paper.

Extracted from the figure batch. Reads single-task Parity-4 results via
find_latest_result(results, 'stdp_hebbian_replication'); if that data is absent the figure
is skipped (see README).
"""
import json
import os
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def find_latest_result(results_dir: str, prefix: str) -> Optional[str]:
    """Find the most recent result file/dir matching prefix."""
    dirs = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*")))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if dirs:
        for fname in ['combined_results.json', 'cl_results_combined.json', 'cl_results.json']:
            path = os.path.join(dirs[-1], fname)
            if os.path.exists(path):
                return path
    files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json")))
    return files[-1] if files else None


def fig2_pareto_frontier(output_dir: str, results_dir: str):
    """Pareto frontier: solve rate vs median generations."""
    filepath = find_latest_result(results_dir, 'stdp_hebbian_replication')
    if not filepath:
        print("  Fig 2: SKIPPED (no single-task results)")
        return

    with open(filepath) as f:
        data = json.load(f)

    strategies = []
    for name, sdata in data.items():
        if name in ('metadata', 'summary') or not isinstance(sdata, dict):
            continue
        if 'results' not in sdata:
            continue
        trials = sdata['results']
        solved = [t for t in trials if t.get('solved', False)]
        gens = [t['solved_gen'] for t in solved if t.get('solved_gen') is not None]
        rate = len(solved) / len(trials) if trials else 0
        med = np.median(gens) if gens else 100
        strategies.append((name, rate, med))

    if not strategies:
        print("  Fig 2: SKIPPED (no valid strategy data)")
        return

    category_colors = {  # Okabe-Ito colorblind-safe palette
        'circadian_rhythm': '#0072B2',
        'critical_period': '#CC79A7',
        'baseline': '#4D4D4D',
        'stdp': '#D55E00',
        'hebbian': '#D55E00',
        'metaplastic': '#009E73',
        'predator_prey': '#E69F00',
        'adult_neurogenesis': '#CC79A7',
    }
    category_labels = {
        'circadian_rhythm': 'Oscillatory',
        'critical_period': 'Developmental',
        'baseline': 'Baseline',
        'stdp': 'Temporal Credit',
        'hebbian': 'Temporal Credit',
        'metaplastic': 'Homeostatic',
        'predator_prey': 'Ecological',
        'adult_neurogenesis': 'Developmental',
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    for name, rate, med in strategies:
        color = category_colors.get(name, '#333333')
        ax.scatter(med, rate * 100, c=color, s=120, zorder=3, edgecolors='white', linewidth=1.5)
        # Label with per-strategy offset
        label_offsets = {
            'hebbian': (5, -12),
            'circadian_rhythm': (5, 8),
            'critical_period': (5, -12),
        }
        offset = label_offsets.get(name, (5, 5))
        display = name.replace('_', ' ').title()
        if len(display) > 12:
            display = display[:12] + '.'
        ax.annotate(display, (med, rate * 100), textcoords="offset points",
                    xytext=offset, fontsize=9, color='#333333')

    # Pareto frontier line
    pareto_points = sorted(strategies, key=lambda x: x[2])  # sort by median gen
    pareto_x, pareto_y = [], []
    best_rate = 0
    for name, rate, med in pareto_points:
        if rate >= best_rate:
            pareto_x.append(med)
            pareto_y.append(rate * 100)
            best_rate = rate
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, '--', color='#BDC3C7', linewidth=1.5, zorder=1, label='Pareto frontier')

    ax.set_xlabel('Median Generations to Solve')
    ax.set_ylabel('Solve Rate (%)')
    ax.set_title('Speed-Reliability Tradeoff (Parity-4, N=30)')
    ax.set_xlim(0, 80)
    ax.set_ylim(45, 105)
    ax.grid(True, alpha=0.3)

    # Category legend
    legend_handles = []
    seen = set()
    for name, _, _ in strategies:
        cat = category_labels.get(name, 'Unknown')
        color = category_colors.get(name, '#333333')
        if cat not in seen:
            legend_handles.append(mpatches.Patch(color=color, label=cat))
            seen.add(cat)
    ax.legend(handles=legend_handles, loc='lower left', framealpha=0.9)

    plt.savefig(os.path.join(output_dir, 'fig2_pareto.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig2_pareto.png'))


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[2]
    results_dir = str(base / "results")
    output_dir = str(base / "figures")
    os.makedirs(output_dir, exist_ok=True)
    fig2_pareto_frontier(output_dir, results_dir)
