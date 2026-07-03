#!/usr/bin/env python3
"""Generate fig_system_overview for the emr-dynamic-functions paper.

System pipeline: palette -> per-node activation selection -> CPPN/NEAT decode -> substrate ->
fitness -> selection loop. Self-contained diagram (extracted from the figure batch; no data).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Ellipse
import numpy as np
from pathlib import Path

COLORS = {
    'orange': '#E69F00', 'sky_blue': '#56B4E9', 'green': '#009E73', 'yellow': '#F0E442',
    'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7', 'black': '#000000', 'gray': '#999999',
}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_figure_1_system_overview():
    """
    Figure 1: System Overview -- vertical one-column pipeline (the version used in
    alife_main.tex). Top-down flow:
      Activation Palette (+ Activation Selection) -> selects per node
      -> CPPN Genome -> decode -> Substrate (per-node activations) -> evaluate
      -> Fitness, with a NEAT-selection loop returning to the CPPN.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 5.5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 14)
    ax.axis('off')

    cx = 3.0  # center x for all boxes

    # === 1. Activation Palette (top) ===
    pal_x, pal_y, pal_w, pal_h = 0.2, 11.8, 3.2, 1.0
    palette_box = FancyBboxPatch((pal_x, pal_y), pal_w, pal_h,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFF3E0', edgecolor=COLORS['orange'], linewidth=1.5)
    ax.add_patch(palette_box)
    ax.text(pal_x + pal_w / 2, pal_y + 0.6, 'Activation Palette',
            fontsize=7.5, ha='center', fontweight='bold')
    ax.text(pal_x + pal_w / 2, pal_y + 0.25, '(18 functions)',
            fontsize=6.5, ha='center', color=COLORS['gray'])

    # === 2. Activation-selection box (right of palette) ===
    meta_x, meta_y, meta_w, meta_h = 4.1, 11.7, 2.6, 1.2
    meta_box = FancyBboxPatch((meta_x, meta_y), meta_w, meta_h,
                              boxstyle="round,pad=0.1",
                              facecolor='#F3E5F5', edgecolor=COLORS['purple'], linewidth=1.2)
    ax.add_patch(meta_box)
    ax.text(meta_x + meta_w / 2, meta_y + 0.7, 'Activation Selection',
            fontsize=7, ha='center', fontweight='bold')
    ax.text(meta_x + meta_w / 2, meta_y + 0.25, '(per-node assignment)',
            fontsize=6, ha='center', color=COLORS['gray'])

    # Arrow: Activation Selection -> Palette (horizontal left)
    ax.annotate('', xy=(pal_x + pal_w, pal_y + pal_h / 2),
                xytext=(meta_x, meta_y + meta_h / 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.2))

    # === 3. Arrow: Palette -> CPPN (selects per node) ===
    pal_cx = pal_x + pal_w / 2
    ax.annotate('', xy=(pal_cx, 10.5), xytext=(pal_cx, 11.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5))
    ax.text(pal_cx + 0.15, 11.1, 'selects per node', fontsize=5.5, ha='left',
            color=COLORS['orange'], fontstyle='italic')

    # === 4. CPPN Genome box ===
    cppn_x, cppn_y, cppn_w, cppn_h = 0.8, 8.8, 4.4, 1.7
    cppn_box = FancyBboxPatch((cppn_x, cppn_y), cppn_w, cppn_h,
                              boxstyle="round,pad=0.15",
                              facecolor='#E6F3FF', edgecolor=COLORS['blue'], linewidth=1.5)
    ax.add_patch(cppn_box)
    ax.text(cppn_x + cppn_w / 2, cppn_y + 1.25, 'CPPN Genome',
            fontsize=8, ha='center', fontweight='bold')
    ax.text(cppn_x + cppn_w / 2, cppn_y + 0.75, '(NEAT)',
            fontsize=7, ha='center', color=COLORS['gray'])
    ax.text(cppn_x + cppn_w / 2, cppn_y + 0.25, r'$f(x_1,y_1,x_2,y_2)$',
            fontsize=6.5, ha='center')

    # === 5. Arrow: CPPN -> Substrate (decode) ===
    ax.annotate('', xy=(cx, 7.8), xytext=(cx, 8.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=2))
    ax.text(cx + 0.15, 8.25, 'decode', fontsize=6, ha='left', color=COLORS['blue'])

    # === 6. Substrate box (centerpiece) ===
    sub_x, sub_y, sub_w, sub_h = 0.5, 3.85, 5.0, 3.95
    substrate_box = FancyBboxPatch((sub_x, sub_y), sub_w, sub_h,
                                   boxstyle="round,pad=0.15",
                                   facecolor='#FFFFF0', edgecolor=COLORS['orange'], linewidth=2)
    ax.add_patch(substrate_box)
    ax.text(sub_x + sub_w / 2, sub_y + sub_h - 0.3, 'Substrate',
            fontsize=9, ha='center', fontweight='bold')

    # -- Input nodes (left column inside substrate) --
    input_x = 1.5
    input_ys = [6.6, 5.7, 4.8]
    for y in input_ys:
        ax.add_patch(Circle((input_x, y), 0.2, facecolor='#DDDDDD', edgecolor='black', lw=1))
    ax.text(input_x, 4.25, 'input', fontsize=5.5, ha='center', color=COLORS['gray'])

    # -- Hidden nodes (center -- per-node activations, 4 nodes) --
    hidden_colors = [COLORS['vermillion'], COLORS['green'], COLORS['blue'], COLORS['purple']]
    hidden_labels = ['sin', 'tanh', 'ReLU', 'gauss']
    hidden_positions = [(2.7, 6.7), (3.5, 6.7), (2.7, 5.2), (3.5, 5.2)]
    for (hx, hy), hcolor, hlabel in zip(hidden_positions, hidden_colors, hidden_labels):
        ax.add_patch(Circle((hx, hy), 0.22, facecolor=hcolor, edgecolor='black', lw=1.2, alpha=0.8))
        ax.text(hx, hy - 0.45, hlabel, fontsize=5, ha='center', color=hcolor, fontweight='bold')

    # -- Output nodes (right column inside substrate) --
    output_x = 4.5
    output_ys = [6.3, 5.3]
    for y in output_ys:
        ax.add_patch(Circle((output_x, y), 0.2, facecolor='#DDDDDD', edgecolor='black', lw=1))
    ax.text(output_x, 4.75, 'output', fontsize=5.5, ha='center', color=COLORS['gray'])

    # -- Connections: input -> hidden, hidden -> output --
    for iy in input_ys:
        for (hx, hy) in hidden_positions:
            ax.plot([input_x + 0.2, hx - 0.22], [iy, hy], 'k-', alpha=0.12, lw=0.5)
    for (hx, hy) in hidden_positions:
        for oy in output_ys:
            ax.plot([hx + 0.22, output_x - 0.2], [hy, oy], 'k-', alpha=0.12, lw=0.5)

    # -- Per-node highlight: dashed box around hidden nodes --
    novel_rect = FancyBboxPatch((2.25, 4.55), 1.72, 2.4, boxstyle="round,pad=0.1",
                                facecolor='none', edgecolor=COLORS['vermillion'],
                                linewidth=1.2, linestyle='--')
    ax.add_patch(novel_rect)
    ax.text(3.1, 4.25, 'per-node activation', fontsize=5,
            ha='center', color=COLORS['vermillion'], fontstyle='italic')

    # === 7. Arrow: Substrate -> Fitness (evaluate) ===
    ax.annotate('', xy=(cx, 2.2), xytext=(cx, 3.85),
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2))
    ax.text(cx + 0.15, 3.0, 'evaluate', fontsize=6, ha='left', color=COLORS['orange'])

    # === 8. Fitness box ===
    fit_x, fit_y, fit_w, fit_h = 0.8, 1.5, 4.4, 0.7
    fitness_box = FancyBboxPatch((fit_x, fit_y), fit_w, fit_h,
                                 boxstyle="round,pad=0.15",
                                 facecolor='#E8F5E9', edgecolor=COLORS['green'], linewidth=1.5)
    ax.add_patch(fitness_box)
    ax.text(fit_x + fit_w / 2, fit_y + fit_h / 2, 'Fitness',
            fontsize=8, ha='center', va='center', fontweight='bold')

    # === 9. NEAT selection loop (right-side return arrow: Fitness -> CPPN) ===
    loop_x = 6.0
    ax.plot([fit_x + fit_w + 0.25, loop_x], [fit_y + fit_h / 2, fit_y + fit_h / 2],
            color=COLORS['black'], lw=1.3, alpha=0.6)
    ax.plot([loop_x, loop_x], [fit_y + fit_h / 2, cppn_y + cppn_h / 2],
            color=COLORS['black'], lw=1.3, alpha=0.6)
    ax.annotate('', xy=(cppn_x + cppn_w, cppn_y + cppn_h / 2),
                xytext=(loop_x + 0.08, cppn_y + cppn_h / 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['black'], lw=1.3, alpha=0.6))
    ax.text(loop_x + 0.15, 5.5, 'NEAT\nselection', fontsize=5.5,
            ha='left', va='center', color=COLORS['black'], fontstyle='italic', rotation=90)

    plt.tight_layout()
    # Save cropped to the drawn content with EQUAL padding on all sides. A plain
    # bbox_inches='tight' also reserves space for the invisible axes spines/background,
    # which are not left/right symmetric and skew the right margin; cropping to the
    # content artists guarantees even margins.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    exts = []
    for art in ax.get_children():
        if art is ax.patch or type(art).__name__ == 'Spine' or not art.get_visible():
            continue
        try:
            bb = art.get_window_extent(renderer=rend)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            exts.append(bb)
    content = Bbox.union(exts)
    pad = 0.08 * fig.dpi
    content = Bbox.from_extents(content.x0 - pad, content.y0 - pad,
                               content.x1 + pad, content.y1 + pad)
    bbi = content.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(OUTPUT_DIR / 'fig_system_overview.pdf', bbox_inches=bbi)
    fig.savefig(OUTPUT_DIR / 'fig_system_overview.png', bbox_inches=bbi)
    plt.close()
    print("Generated: fig_system_overview.pdf")



if __name__ == '__main__':
    generate_figure_1_system_overview()
