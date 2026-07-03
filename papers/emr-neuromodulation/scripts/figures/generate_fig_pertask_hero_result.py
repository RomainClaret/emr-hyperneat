"""Generate fig_pertask_hero_result from REAL experiment data.

Data source: results/palette_per_head_20260118_185*.json
5 seeds × product aggregation × per-task activation (sin for XOR, tanh for threshold)
All 5 seeds converge to 100% on all 5 tasks (gen 11, 7, 31, 6, 6).
Annotation shows 30-seed validation stats (median 14, range 4-36).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Real experiment data files (product aggregation, per-task activation, 5 seeds)
DATA_FILES = [
    'results/palette_per_head_20260118_185629.json',  # seed 42
    'results/palette_per_head_20260118_185654.json',  # seed 123
    'results/palette_per_head_20260118_185753.json',  # seed 456
    'results/palette_per_head_20260118_185818.json',  # seed 789
    'results/palette_per_head_20260118_185841.json',  # seed 1000
]

TASKS = ['xor', 'and', 'or', 'nand', 'nor']
TASK_LABELS = {'xor': 'XOR', 'and': 'AND', 'or': 'OR', 'nand': 'NAND', 'nor': 'NOR'}
TASK_COLORS = {  # Okabe-Ito colorblind-safe (verified: min CVD CIEDE2000 dE >= 12; see figures/verify_colorblind.py)
    'xor': '#0072B2',    # blue
    'and': '#D55E00',    # vermillion
    'or': '#009E73',     # bluish green
    'nand': '#CC79A7',   # reddish purple
    'nor': '#56B4E9',    # sky blue
}
# Line styles and markers for visual differentiation when lines overlap
# (AND/OR/NOR overlap at 100%, XOR/NAND overlap during convergence)
TASK_LINESTYLES = {
    'xor': '-',          # solid (main story - parity task)
    'and': '--',         # dashed
    'or': '-.',          # dash-dot
    'nand': ':',         # dotted
    'nor': (0, (3, 1, 1, 1)),  # dash-dot-dot (custom)
}
TASK_MARKERS = {
    'xor': 'o',          # circle
    'and': 's',          # square
    'or': '^',           # triangle up
    'nand': 'D',         # diamond
    'nor': 'v',          # triangle down
}
# Mark every N generations to avoid clutter
MARKER_INTERVAL = 5


def load_experiments():
    """Load all experiment data from JSON files."""
    experiments = []
    for fpath in DATA_FILES:
        with open(Path(__file__).resolve().parents[2] / fpath) as f:
            data = json.load(f)
        experiments.append(data)
    return experiments


def create_pertask_hero_figure():
    """Create the hero figure with mean ± std shaded bands across 5 seeds.

    Aggregates to mean±std per task (5 lines) instead of 25 overlapping seed lines.
    Much clearer visualization of threshold convergence versus XOR solving.
    """
    experiments = load_experiments()

    # Find max generation across all seeds
    max_gen = max(exp['convergence_gen'] for exp in experiments)
    # Extend slightly past max convergence for display
    display_gen = max_gen + 3
    gen_range = np.arange(0, display_gen + 1)

    # Build per-task arrays: [n_seeds × n_generations]
    # After a seed's last recorded generation, fill with final value (1.0 = 100%)
    task_data = {task: np.zeros((len(experiments), len(gen_range))) for task in TASKS}

    for exp_idx, exp in enumerate(experiments):
        history = exp['fitness_history']
        hist_gens = {h['generation']: h for h in history}
        for task in TASKS:
            for g_idx, g in enumerate(gen_range):
                if g in hist_gens:
                    task_data[task][exp_idx, g_idx] = hist_gens[g][task] * 100
                else:
                    # After convergence: fill with 100% (all seeds converge to 100%)
                    task_data[task][exp_idx, g_idx] = 100.0

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.52))  # tuned so the tight-cropped aspect matches Fig 2 (equal on-page height at \columnwidth)

    for task in TASKS:
        data = task_data[task]
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        color = TASK_COLORS[task]
        linestyle = TASK_LINESTYLES[task]
        marker = TASK_MARKERS[task]
        # Plot line with style differentiation
        ax.plot(gen_range, mean, color=color, lw=2.0, linestyle=linestyle,
                label=TASK_LABELS[task], zorder=3)
        # Add markers at intervals to distinguish overlapping lines
        marker_indices = np.arange(0, len(gen_range), MARKER_INTERVAL)
        ax.plot(gen_range[marker_indices], mean[marker_indices], color=color,
                marker=marker, markersize=4, linestyle='none', zorder=4)
        ax.fill_between(gen_range, np.clip(mean - std, 60, 102),
                        np.clip(mean + std, 60, 102), color=color, alpha=0.12, zorder=2)

    # Mark individual convergence generations as small ticks on the x-axis
    conv_gens = [exp['convergence_gen'] for exp in experiments]
    for g in conv_gens:
        ax.axvline(x=g, color='gray', alpha=0.15, lw=0.5, linestyle=':', zorder=1)

    # Reference lines
    ax.axhline(y=98, color='#4D4D4D', alpha=0.4, lw=0.5, linestyle='--', zorder=1)
    ax.text(display_gen, 97, '98%', fontsize=5.5, color='#4D4D4D', alpha=0.8,
            ha='right', va='top')

    ax.set_xlabel('Generation', fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_xlim(-0.5, display_gen)
    ax.set_ylim(65, 103)
    ax.tick_params(axis='both', labelsize=7)
    ax.legend(fontsize=6.5, loc='lower right', ncol=2, framealpha=0.8,
              handlelength=1.5, columnspacing=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 30-seed stats (figure plots 5 representative seeds; annotation reflects full validation)
    ax.text(0.02, 0.02, 'Median convergence: gen 14 (range 4\u201336, n=30)',
            transform=ax.transAxes, fontsize=6.5, color='#333333', fontstyle='italic')

    plt.tight_layout(pad=0.3)

    # Save
    outdir = Path(__file__).resolve().parents[2] / 'figures'
    plt.savefig(outdir / 'fig_pertask_hero_result.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(outdir / 'fig_pertask_hero_result.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved fig_pertask_hero_result.pdf and .png')

    # Print summary
    print(f'\nData summary:')
    for exp in experiments:
        print(f'  Seed {exp["seed"]}: converged at gen {exp["convergence_gen"]}, '
              f'all tasks 100%')
    gens = [exp['convergence_gen'] for exp in experiments]
    print(f'  Median convergence: gen {np.median(gens):.0f} '
          f'(range {min(gens)}-{max(gens)})')


if __name__ == '__main__':
    create_pertask_hero_figure()
