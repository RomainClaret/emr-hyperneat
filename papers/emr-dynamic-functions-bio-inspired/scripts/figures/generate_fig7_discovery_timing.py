"""Generate Fig 7: Convergence speed box plot by bio-inspired strategy.

Horizontal box plot showing solved_gen distributions for Parity-4.
Strategies sorted by median convergence speed (fastest at top).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parents[2] / "results" / "single_task" / "combined_results.json"
OUT_DIR = Path(__file__).resolve().parents[2] / "figures"

STRATEGY_META = {
    "circadian_rhythm":   {"display": "Circadian",    "category": "Oscillatory"},
    "hebbian":            {"display": "Hebbian",       "category": "Temporal Credit"},
    "stdp":               {"display": "STDP",          "category": "Temporal Credit"},
    "critical_period":    {"display": "Crit. Period",  "category": "Developmental"},
    "metaplastic":        {"display": "Metaplastic",   "category": "Homeostatic"},
    "adult_neurogenesis": {"display": "Neurogenesis",  "category": "Developmental"},
    "baseline":           {"display": "Baseline",      "category": "Baseline"},
}

CATEGORY_COLORS = {  # Okabe-Ito colorblind-safe palette
    "Oscillatory":     "#0072B2",
    "Temporal Credit": "#D55E00",
    "Developmental":   "#CC79A7",
    "Homeostatic":     "#009E73",
    "Baseline":        "#4D4D4D",
}

# ---------------------------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------------------------

with open(DATA_PATH) as f:
    raw = json.load(f)

# Build per-strategy arrays of solved_gen (solved runs only), skip predator_prey
strategy_data = {}
for key, meta in STRATEGY_META.items():
    runs = raw[key]["results"]
    solved_gens = [r["solved_gen"] for r in runs if r["solved"]]
    strategy_data[key] = {
        "gens": solved_gens,
        "median": float(np.median(solved_gens)) if solved_gens else float("inf"),
        "mean": float(np.mean(solved_gens)) if solved_gens else float("inf"),
        "n_solved": len(solved_gens),
        "n_total": len(runs),
        "display": meta["display"],
        "category": meta["category"],
    }

# Sort by median solved_gen ascending (fastest at top in horizontal box plot
# means fastest = highest y position)
sorted_keys = sorted(strategy_data.keys(), key=lambda k: strategy_data[k]["median"])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

fig, ax = plt.subplots(figsize=(7, 3.8))

positions = list(range(len(sorted_keys)))
labels = []
box_data = []
colors = []

for i, key in enumerate(sorted_keys):
    sd = strategy_data[key]
    box_data.append(sd["gens"])
    labels.append(sd["display"])
    colors.append(CATEGORY_COLORS[sd["category"]])

bp = ax.boxplot(
    box_data,
    positions=positions,
    vert=False,
    patch_artist=True,
    widths=0.55,
    showfliers=True,
    flierprops=dict(marker="o", markersize=3, markerfacecolor="#555555",
                    markeredgecolor="#555555", alpha=0.5),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="#333333", linewidth=0.8),
    capprops=dict(color="#333333", linewidth=0.8),
    boxprops=dict(linewidth=0.8, edgecolor="#333333"),
)

# Color the boxes and add mean diamonds
for i, (patch, key) in enumerate(zip(bp["boxes"], sorted_keys)):
    patch.set_facecolor(colors[i])
    patch.set_alpha(0.90)
    sd = strategy_data[key]
    # Mean marker
    ax.plot(sd["mean"], i, marker="D", color="white", markeredgecolor="black",
            markersize=5, zorder=5, markeredgewidth=0.8)

# Force x-axis extent
x_max = max(max(sd["gens"]) for sd in strategy_data.values() if sd["gens"]) * 1.08
ax.set_xlim(-2, x_max)

ax.set_yticks(positions)
ax.set_yticklabels(labels)
ax.set_xlabel("Generation Solved")
ax.set_title("Convergence Speed by Strategy (Parity-4, N=30)", pad=10)
ax.invert_yaxis()  # fastest (lowest median) at top

# Legend for categories (de-duplicate)
from matplotlib.patches import Patch
seen = set()
legend_handles = []
# Order legend by the order categories appear top-to-bottom
for key in sorted_keys:
    cat = strategy_data[key]["category"]
    if cat not in seen:
        seen.add(cat)
        legend_handles.append(Patch(facecolor=CATEGORY_COLORS[cat], edgecolor="#333333",
                                    alpha=0.90, label=cat))
ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

ax.grid(axis="x", alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig7_discovery_timing.pdf")
fig.savefig(OUT_DIR / "fig7_discovery_timing.png")
plt.close(fig)

# Print summary
print("Convergence speed summary (sorted by median):")
print(f"{'Strategy':<16} {'Median':>7} {'Mean':>7} {'n':>4}")
print("-" * 38)
for key in sorted_keys:
    sd = strategy_data[key]
    print(f"{sd['display']:<16} {sd['median']:>7.1f} {sd['mean']:>7.1f} {sd['n_solved']:>4}")

print(f"\nSaved to {OUT_DIR / 'fig7_discovery_timing.pdf'}")
print(f"Saved to {OUT_DIR / 'fig7_discovery_timing.png'}")
