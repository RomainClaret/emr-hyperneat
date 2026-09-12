#!/usr/bin/env python3
"""
Generate fig_zero_crossing.pdf for ALIFE 2026 Paper 1.

Oscillatory Structure vs Parity-4 Solve Rate:
Scatter plot showing that local extrema count (oscillatory richness)
predicts solvability better than simple zero-crossing count.

Per-function ablation data from: per_function_ablation_n30/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'gray': '#999999',
    'black': '#000000',
}


def count_local_extrema(func, x_range=(-5, 5), n_points=100000):
    """Count local extrema (peaks + troughs) of a function in the given range.

    Uses derivative sign changes to find extrema, then filters by minimum
    amplitude to exclude numerical noise while keeping real oscillations.
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = func(x)

    # Compute numerical derivative
    dy = np.diff(y)

    # Find sign changes in derivative (extrema locations)
    signs = np.sign(dy)
    # Remove zero-derivative regions
    signs[signs == 0] = 1
    sign_changes = np.diff(signs)

    # Maxima: derivative goes from + to - (sign_changes == -2)
    maxima_idx = np.where(sign_changes == -2)[0] + 1
    # Minima: derivative goes from - to + (sign_changes == 2)
    minima_idx = np.where(sign_changes == 2)[0] + 1

    all_extrema_idx = np.sort(np.concatenate([maxima_idx, minima_idx]))

    if len(all_extrema_idx) < 2:
        return len(all_extrema_idx)

    # Filter: require minimum amplitude between consecutive extrema
    # This removes tiny numerical artifacts while keeping real oscillations
    min_amplitude = 0.02  # 2% of typical function range
    filtered = [all_extrema_idx[0]]
    for i in range(1, len(all_extrema_idx)):
        amplitude = abs(y[all_extrema_idx[i]] - y[all_extrema_idx[i-1]])
        if amplitude > min_amplitude:
            filtered.append(all_extrema_idx[i])

    return len(filtered)


def count_zero_crossings(func, x_range=(-5, 5), n_points=100000):
    """Count zero crossings (sign changes) of a function."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = func(x)
    signs = np.sign(y)
    sign_changes = np.abs(np.diff(signs))
    return int(np.sum(sign_changes == 2))


def get_function_implementations():
    """Define all 18 functions matching ACTIVATION_LIST exactly."""
    return {
        'tanh': lambda x: np.tanh(x),
        'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        'relu': lambda x: np.maximum(0, x),
        'identity': lambda x: x,
        'sin': lambda x: np.sin(x),
        'gauss': lambda x: np.exp(-x**2),
        'lelu': lambda x: np.where(x > 0, x, 0.01 * x),
        'softplus': lambda x: np.log(1 + np.exp(np.clip(x, -500, 500))),
        'rs_adapt': lambda x: np.tanh(x) * (1 - 0.3 * np.abs(x)),
        'fs_fast': lambda x: 2 * np.maximum(0, x),
        'lts_low': lambda x: 1.0 / (1.0 + np.exp(-np.clip(2*x - 0.5, -500, 500))),
        'burst': lambda x: np.tanh(x) + 0.5 * np.sin(3*x),
        'resonator': lambda x: np.sin(x) * np.exp(-np.abs(x) / 3),
        'osc_adapt': lambda x: np.sin(x) * (1 - 0.2 * np.abs(x)),
        'gain_mod': lambda x: x / (1 + np.abs(x)),
        'receptive': lambda x: np.exp(-x**2) * np.cos(2*x),
        'band_pass': lambda x: np.exp(-np.abs(x - 1)) - np.exp(-np.abs(x + 1)),
        'integrate': lambda x: np.tanh(x) * (1 + 0.2 * np.exp(-np.abs(x))),
    }


# Solve rates from N=30 per-function ablation (Table 1 in paper)
SOLVE_RATES = {
    'sin': 100.0,
    'osc_adapt': 100.0,
    'burst': 100.0,
    'band_pass': 100.0,
    'receptive': 80.0,
    'resonator': 66.7,
    'gauss': 33.3,
    'rs_adapt': 23.3,
    'lts_low': 6.7,
    'tanh': 0.0,
    'sigmoid': 0.0,
    'relu': 0.0,
    'identity': 0.0,
    'lelu': 0.0,
    'softplus': 0.0,
    'fs_fast': 0.0,
    'gain_mod': 0.0,
    'integrate': 0.0,
}

# Categories for coloring
CATEGORIES = {
    'sin': 'Multi-crossing',
    'osc_adapt': 'Multi-crossing',
    'burst': 'Multi-crossing',
    'band_pass': 'Multi-crossing',
    'receptive': 'Intermediate',
    'resonator': 'Intermediate',
    'gauss': 'Intermediate',
    'rs_adapt': 'Intermediate',
    'lts_low': 'Intermediate',
    'tanh': 'Monotonic',
    'sigmoid': 'Monotonic',
    'relu': 'Monotonic',
    'identity': 'Monotonic',
    'lelu': 'Monotonic',
    'softplus': 'Monotonic',
    'fs_fast': 'Monotonic',
    'gain_mod': 'Monotonic',
    'integrate': 'Monotonic',
}

CATEGORY_COLORS = {
    'Multi-crossing': COLORS['blue'],
    'Intermediate': COLORS['orange'],
    'Monotonic': COLORS['red'],
}

CATEGORY_MARKERS = {
    'Multi-crossing': 'o',
    'Intermediate': 's',
    'Monotonic': 'X',
}


def main():
    funcs = get_function_implementations()

    # Compute local extrema for each function
    extrema_counts = {}
    zero_crossings = {}
    for name, func in funcs.items():
        extrema_counts[name] = count_local_extrema(func)
        zero_crossings[name] = count_zero_crossings(func)

    # Print comparison table
    print(f"{'Function':<14} {'Extrema':<10} {'ZeroCross':<12} {'Solve%':<10} {'Category'}")
    print("-" * 60)
    for name in sorted(extrema_counts, key=lambda n: -extrema_counts[n]):
        print(f"{name:<14} {extrema_counts[name]:<10} {zero_crossings[name]:<12} "
              f"{SOLVE_RATES[name]:<10.1f} {CATEGORIES[name]}")

    # Create figure, use local extrema as primary metric
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Jitter for overlapping points at (0, 0)
    np.random.seed(42)

    # Plot by category for legend
    for cat in ['Multi-crossing', 'Intermediate', 'Monotonic']:
        cat_names = [n for n in extrema_counts if CATEGORIES[n] == cat]
        x_vals = []
        y_vals = []
        for n in cat_names:
            x = extrema_counts[n]
            y = SOLVE_RATES[n]
            # Add small jitter to separate overlapping points
            if x == 0 and y == 0:
                x += np.random.uniform(-0.15, 0.15)
                y += np.random.uniform(-2, 2)
            x_vals.append(x)
            y_vals.append(y)

        ax.scatter(x_vals, y_vals,
                   c=CATEGORY_COLORS[cat],
                   marker=CATEGORY_MARKERS[cat],
                   s=60, edgecolors=COLORS['black'], linewidths=0.5,
                   label=f'{cat} ({len(cat_names)})',
                   zorder=3)

    # Label notable points
    # Functions that need labeling (skip the 0,0 cluster)
    label_config = {
        'sin': (0.4, 0),
        'osc_adapt': (0.4, -6),
        'burst': (0.4, 3),
        'band_pass': (0.3, 5),
        'receptive': (0.3, 0),
        'resonator': (0.3, 5),
        'gauss': (0.3, 3),
        'rs_adapt': (0.3, 3),
        'lts_low': (0.3, 3),
    }

    for name, (dx, dy) in label_config.items():
        x = extrema_counts[name]
        y = SOLVE_RATES[name]
        label = name.replace('_', ' ')
        ax.annotate(label, xy=(x, y),
                    xytext=(x + dx, y + dy),
                    fontsize=7, ha='left',
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.4))

    # Label the 0-extrema, 0% cluster (placed to the right to avoid overlap)
    zero_mono = [n for n in extrema_counts
                 if extrema_counts[n] == 0 and SOLVE_RATES[n] == 0]
    if zero_mono:
        ax.annotate(f'{len(zero_mono)} monotonic\nfunctions',
                    xy=(0, 0), xytext=(3.5, 10),
                    fontsize=7, ha='center', color=COLORS['black'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.6))

    ax.set_xlabel('Local Extrema in $[-5, 5]$')
    ax.set_ylabel('Parity-4 Solve Rate (%)')
    ax.set_title('Oscillatory Structure Predicts Solvability', fontsize=9)

    ax.set_ylim(-8, 108)
    ax.set_xlim(-0.8, max(extrema_counts.values()) + 1.5)

    ax.axhline(y=0, color=COLORS['gray'], linestyle=':', linewidth=0.5, alpha=0.5)

    ax.legend(loc='center right', framealpha=0.9, edgecolor=COLORS['gray'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # N annotation removed per paper revision

    plt.tight_layout()

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')
    os.makedirs(script_dir, exist_ok=True)
    pdf_path = os.path.join(script_dir, 'fig_zero_crossing.pdf')
    png_path = os.path.join(script_dir, 'fig_zero_crossing.png')

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGenerated: {pdf_path}")
    print(f"Generated: {png_path}")


if __name__ == '__main__':
    main()
