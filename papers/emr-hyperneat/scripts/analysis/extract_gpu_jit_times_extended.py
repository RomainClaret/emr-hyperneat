#!/usr/bin/env python3
"""Per-population breakdown of EMR GPU JIT / construction overhead (real runs).

Companion to ``extract_gpu_jit_times.py``: reads ``data/runs_emr_gpu.json`` and shows the
one-time ``construction_overhead_s`` per depth at each population size, so the growth of
JIT/construction cost with population and depth is visible. The collapsed per-depth JIT
IQR (the figure's dotted GPU JIT band) is printed at the end.

Output: a depth x population table of JIT/construction seconds + the per-depth JIT IQR.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GPU_JSON = DATA_DIR / "runs_emr_gpu.json"


def extract_jit_iqr():
    runs = json.load(open(GPU_JSON))["runs"]

    cells = defaultdict(list)
    by_depth = defaultdict(list)
    for r in runs:
        if not r.get("in_figure_band"):
            continue
        cells[(r["depth"], r["population"])].append(r["construction_overhead_s"])
        by_depth[r["depth"]].append(r["construction_overhead_s"])

    depths = sorted(by_depth)
    pops = sorted({p for (_, p) in cells})

    print("EMR GPU JIT / construction overhead by depth x population (median s, real runs)")
    print("=" * 78)
    print(f"{'Depth':<6}" + "".join(f"{('pop' + str(p)):<12}" for p in pops))
    print("-" * 78)
    for d in depths:
        row = f"{d:<6}"
        for p in pops:
            vals = cells.get((d, p))
            row += f"{(f'{np.median(vals):.1f}' if vals else '-'):<12}"
        print(row)

    print()
    print("Per-depth JIT IQR across populations (the figure's GPU JIT band):")
    print(f"{'Depth':<6} {'Q1 (25%)':<12} {'Median':<12} {'Q3 (75%)':<12} {'N':<4}")
    print("-" * 58)
    for d in depths:
        q1, median, q3 = np.percentile(by_depth[d], [25, 50, 75])
        print(f"{d:<6} {q1:<12.1f} {median:<12.1f} {q3:<12.1f} {len(by_depth[d]):<4}")


if __name__ == "__main__":
    extract_jit_iqr()
