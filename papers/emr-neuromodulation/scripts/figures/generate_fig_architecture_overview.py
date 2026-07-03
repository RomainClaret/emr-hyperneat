"""Generate architecture figure for ALIFE 2026 paper.

Portrait layout designed for single-column width (~3.3").
figsize=(3.5, 5.0), compact enough for LaTeX [t] float placement.
Stacked panels A (network) + B (evaluation).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"


def fig_architecture_overview():
    """Two-panel architecture figure in portrait orientation.

    Top: Network topology (2 inputs -> ~10 hidden -> 1 output)
    Bottom: 5 evaluation modes with NT vectors and activations.
    """
    fig = plt.figure(figsize=(3.5, 4.4))

    gs = fig.add_gridspec(2, 1, height_ratios=[0.38, 0.62], hspace=0.22)
    ax_net = fig.add_subplot(gs[0])
    ax_eval = fig.add_subplot(gs[1])

    # ================================================================
    # TOP PANEL: Network Architecture
    # ================================================================
    ax_net.set_xlim(-0.2, 5.2)
    ax_net.set_ylim(-0.8, 5.8)
    ax_net.axis('off')
    ax_net.set_title('A) Network Architecture', fontsize=8, fontweight='bold', pad=4)

    input_color = '#3498db'
    hidden_color = '#2ecc71'
    output_color = '#e74c3c'
    weight_color = '#bdc3c7'

    # Input neurons (x=0.5)
    input_y = [1.8, 3.5]
    input_labels = ['$x_1$', '$x_2$']
    for y, label in zip(input_y, input_labels):
        circle = plt.Circle((0.5, y), 0.28, color=input_color, ec='black', lw=0.8, zorder=5)
        ax_net.add_patch(circle)
        ax_net.text(0.5, y, label, ha='center', va='center', fontsize=6,
                    fontweight='bold', color='white', zorder=6)

    # Hidden neurons (x=2.5) - show 5 representative
    hidden_ys = np.linspace(0.8, 4.5, 5)
    for i, y in enumerate(hidden_ys):
        circle = plt.Circle((2.5, y), 0.22, color=hidden_color, ec='black', lw=0.7, zorder=5)
        ax_net.add_patch(circle)
        ax_net.text(2.5, y, f'$h_{{{i+1}}}$', ha='center', va='center', fontsize=4.5,
                    fontweight='bold', color='white', zorder=6)

    # Output neuron (x=4.5)
    circle = plt.Circle((4.5, 2.65), 0.28, color=output_color, ec='black', lw=1.0, zorder=5)
    ax_net.add_patch(circle)
    ax_net.text(4.5, 2.65, 'out', ha='center', va='center', fontsize=5.5,
                fontweight='bold', color='white', zorder=6)

    # Connections: input -> hidden
    for iy in input_y:
        for hy in hidden_ys:
            ax_net.plot([0.78, 2.28], [iy, hy], color=weight_color, lw=0.3, alpha=0.45, zorder=1)

    # Connections: hidden -> output
    for hy in hidden_ys:
        ax_net.plot([2.72, 4.22], [hy, 2.65], color=weight_color, lw=0.4, alpha=0.5, zorder=1)

    # Layer labels
    ax_net.text(0.5, 0.15, 'Input (2)', ha='center', fontsize=5.5, fontweight='bold')
    ax_net.text(2.5, 0.15, 'Hidden (N)', ha='center', fontsize=5.5, fontweight='bold')
    ax_net.text(4.5, 0.15, 'Output (1)', ha='center', fontsize=5.5, fontweight='bold')

    # Shared weight labels
    ax_net.text(1.45, 5.0, '$W_1$ (shared)', fontsize=5, ha='center', color='#7f8c8d',
                style='italic')
    ax_net.text(3.55, 5.0, '$W_2$ (shared)', fontsize=5, ha='center', color='#7f8c8d',
                style='italic')

    # Receptor density box
    rect_r = mpatches.FancyBboxPatch((1.6, -0.65), 1.8, 0.5,
                                      boxstyle="round,pad=0.06",
                                      facecolor='#f5eef8', edgecolor='#8e44ad', lw=0.7)
    ax_net.add_patch(rect_r)
    ax_net.text(2.5, -0.4, '$R \\in \\mathbb{R}^{N \\times 4}$ (receptors)',
                fontsize=5, ha='center', va='center', color='#8e44ad', fontweight='bold')

    # NT vector + activation arrow
    ax_net.annotate('NT + $f$ (per task)',
                    xy=(2.5, 4.8), xytext=(2.5, 5.45),
                    fontsize=5.5, ha='center', va='bottom', color='#e67e22',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.2))

    # "1 neuron" emphasis
    rect_out = mpatches.FancyBboxPatch((3.95, 1.9), 1.1, 0.35,
                                        boxstyle="round,pad=0.03",
                                        facecolor='#fadbd8', edgecolor=output_color,
                                        lw=0.8, linestyle='--', alpha=0.7)
    ax_net.add_patch(rect_out)
    ax_net.text(4.5, 2.07, '1 neuron', fontsize=4.5, ha='center', va='center',
                color=output_color, fontweight='bold')

    # ================================================================
    # BOTTOM PANEL: 5 Evaluation Modes
    # ================================================================
    ax_eval.set_xlim(-0.2, 10)
    ax_eval.set_ylim(-0.3, 9.8)
    ax_eval.axis('off')
    ax_eval.set_title('B) Per-Task Evaluation (5 passes, same network)',
                       fontsize=7, fontweight='bold', pad=4)

    tasks = [
        ('XOR',  [0.95, 0.05, 0.95, 1.0], 'sin',  '#e74c3c'),
        ('AND',  [0.10, 0.90, 0.10, 1.0], 'tanh', '#3498db'),
        ('OR',   [0.50, 0.50, 0.50, 1.0], 'tanh', '#2ecc71'),
        ('NAND', [0.10, 0.90, 0.10, 0.0], 'tanh', '#9b59b6'),
        ('NOR',  [0.50, 0.50, 0.50, 0.0], 'tanh', '#f39c12'),
    ]

    y_start = 9.2
    row_height = 1.75

    for i, (name, nt, act_fn, color) in enumerate(tasks):
        y = y_start - i * row_height

        # Task name box
        rect = mpatches.FancyBboxPatch((0.05, y - 0.42), 1.2, 0.84,
                                        boxstyle="round,pad=0.06",
                                        facecolor=color, edgecolor='black',
                                        lw=0.8, alpha=0.15)
        ax_eval.add_patch(rect)
        ax_eval.text(0.65, y + 0.06, name, fontsize=7, fontweight='bold',
                     ha='center', va='center', color=color)

        # Arrow task -> config
        ax_eval.annotate('', xy=(1.55, y), xytext=(1.25, y),
                         arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

        # NT vector + activation
        nt_str = f'[{nt[0]:.1f},{nt[1]:.1f},{nt[2]:.1f},{nt[3]:.0f}]'
        ax_eval.text(3.2, y + 0.18, 'NT=' + nt_str, fontsize=4.5,
                     ha='center', va='center', family='monospace', color='#34495e')

        act_color = '#e74c3c' if act_fn == 'sin' else '#3498db'
        ax_eval.text(3.2, y - 0.2, f'$f$ = {act_fn}', fontsize=6,
                     ha='center', va='center', fontweight='bold', color=act_color)

        # Arrow config -> shared network box (at this row's y)
        ax_eval.annotate('', xy=(5.0, y), xytext=(4.6, y),
                         arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

        # Arrow shared network box -> output (at this row's y)
        ax_eval.annotate('', xy=(7.2, y), xytext=(6.6, y),
                         arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

        # Output
        if nt[3] == 0.0:
            ax_eval.text(8.5, y + 0.12, f'{name}($x_1$,$x_2$)', fontsize=5,
                         ha='center', va='center', color=color, fontweight='bold')
            ax_eval.text(8.5, y - 0.15, 'out = 1\u2212\u03c3(\u00b7)', fontsize=4.5,
                         ha='center', va='center', color='#7f8c8d', style='italic')
        else:
            ax_eval.text(8.5, y + 0.12, f'{name}($x_1$,$x_2$)', fontsize=5,
                         ha='center', va='center', color=color, fontweight='bold')
            ax_eval.text(8.5, y - 0.15, 'out = \u03c3(\u00b7)', fontsize=4.5,
                         ha='center', va='center', color='#7f8c8d', style='italic')

    # Single shared network box spanning all 5 rows
    y_top_row = y_start
    y_bot_row = y_start - 4 * row_height
    box_pad = 0.45
    net_rect = mpatches.FancyBboxPatch((5.0, y_bot_row - box_pad), 1.6,
                                        (y_top_row - y_bot_row) + 2 * box_pad,
                                        boxstyle="round,pad=0.08",
                                        facecolor='#ecf0f1', edgecolor='#2c3e50',
                                        lw=0.9, zorder=0)
    ax_eval.add_patch(net_rect)
    y_mid = (y_top_row + y_bot_row) / 2
    ax_eval.text(5.8, y_mid + 0.25, '2 \u2192 N \u2192 1', fontsize=7, ha='center', va='center',
                 fontweight='bold', color='#2c3e50', zorder=1)
    ax_eval.text(5.8, y_mid - 0.25, '(1 output neuron)', fontsize=4.5, ha='center', va='center',
                 color='#7f8c8d', style='italic', zorder=1)

    # "Single network" label below shared box
    ax_eval.text(5.8, y_bot_row - box_pad - 0.3, 'Single network',
                 fontsize=5, ha='center', va='center', color='#2c3e50',
                 style='italic', fontweight='bold')

    fig.subplots_adjust(bottom=0.02, top=0.96, left=0.01, right=0.99)

    for fmt in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f'fig_architecture_overview.{fmt}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_architecture_overview")


if __name__ == '__main__':
    print(f"Generating figures to {FIGURES_DIR}")
    fig_architecture_overview()
    print("Done!")
