#!/usr/bin/env python3
"""Per-population breakdown of EMR GPU total runtime (real 30-generation runs).

The runtime figure's GPU band is an IQR across population sizes. This script reads
``data/runs_emr_gpu.json`` and shows, per depth, the median ``total_runtime_s`` at each
population (50-1000 for depths 1-7; 300/1000 for depths 8-13), so it is clear which
populations drive the spread. ``extract_gpu_iqr.py`` reports the collapsed per-depth IQR.

Output: a depth x population table of median total runtimes + the per-depth IQR.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GPU_JSON = DATA_DIR / "runs_emr_gpu.json"


def extract_gpu_iqr_extended():
    runs = json.load(open(GPU_JSON))["runs"]

    # (depth, population) -> list of total_runtime_s
    cells = defaultdict(list)
    by_depth = defaultdict(list)
    for r in runs:
        if not r.get("in_figure_band"):
            continue
        cells[(r["depth"], r["population"])].append(r["total_runtime_s"])
        by_depth[r["depth"]].append(r["total_runtime_s"])

    depths = sorted(by_depth)
    pops = sorted({p for (_, p) in cells})

    print("EMR GPU total runtime by depth x population (median seconds, real runs)")
    print("=" * 78)
    header = f"{'Depth':<6}" + "".join(f"{('pop' + str(p)):<12}" for p in pops)
    print(header)
    print("-" * 78)
    for d in depths:
        row = f"{d:<6}"
        for p in pops:
            vals = cells.get((d, p))
            row += f"{(f'{np.median(vals):.1f}' if vals else '-'):<12}"
        print(row)

    print()
    print("Per-depth IQR across populations (matches the figure's GPU band):")
    print(f"{'Depth':<6} {'Q1 (25%)':<14} {'Median':<14} {'Q3 (75%)':<14} {'N':<4}")
    print("-" * 60)
    for d in depths:
        q1, median, q3 = np.percentile(by_depth[d], [25, 50, 75])
        print(f"{d:<6} {q1:<14.1f} {median:<14.1f} {q3:<14.1f} {len(by_depth[d]):<4}")


if __name__ == "__main__":
    extract_gpu_iqr_extended()
