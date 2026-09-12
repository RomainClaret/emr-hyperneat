#!/usr/bin/env python3
"""Generate fig_results_overview for the emr-neuromodulation paper.

Activation comparison on XOR: monotonic tanh (fails at sum=2) vs oscillatory sin (succeeds).
Self-contained illustrative figure (no data needed).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})

output_dir = Path(__file__).resolve().parents[2] / "figures"
output_dir.mkdir(exist_ok=True)


def create_results_overview_figure():
    """Figure: Activation function comparison (sin vs tanh on XOR).

    Single panel showing the activation curves. Task-count scaling, the XOR
    plateau, and per-task success are omitted because their data is already
    stated in the text and Table 8. The activation-curve panel is retained
    because it gives visual insight into why tanh fails and sin succeeds.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.6))

    x = np.linspace(-0.5, 2.5, 200)
    ax.plot(x, np.tanh(x), '-', color='#0072B2', linewidth=2, label='tanh (monotonic)')
    ax.plot(x, np.sin(np.pi * x / 2), '-', color='#D55E00', linewidth=2, label=r'sin($\pi$x/2) (oscillatory)')

    # XOR target points
    xor_sums = [0, 1, 1, 2]
    xor_targets = [0, 1, 1, 0]
    for s, t in zip(xor_sums, xor_targets):
        ax.plot(s, t, 'D', color='gray', markersize=8, alpha=0.6, zorder=4)

    # Mark tanh failure at sum=2
    ax.plot(2, np.tanh(2), 'X', color='#0072B2', markersize=10, markeredgecolor='black',
            zorder=5)
    ax.annotate('tanh stuck HIGH\n(target: LOW)',
                xy=(2, np.tanh(2)), xytext=(1.5, 0.2),
                arrowprops=dict(arrowstyle='->', color='#0072B2', lw=1),
                fontsize=8, color='#0072B2', ha='center')

    # Mark sin success at sum=2
    ax.plot(2, np.sin(np.pi * 2 / 2), 'o', color='#D55E00', markersize=10,
            markeredgecolor='black', zorder=5)
    ax.annotate('sin returns LOW',
                xy=(2, np.sin(np.pi * 2 / 2)), xytext=(1.0, -0.7),
                arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1),
                fontsize=8, color='#D55E00', ha='center')

    ax.set_xlabel('Input Sum ($x_1 + x_2$)')
    ax.set_ylabel('Output')
    ax.set_title('Activation Function Comparison on XOR')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['(0,0)', '(0,1)/(1,0)', '(1,1)'])

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_results_overview.{ext}')
    plt.close()
    print('Created: fig_results_overview')


if __name__ == '__main__':
    create_results_overview_figure()
