#!/usr/bin/env python3
"""Extract EMR-HyperNEAT GPU JIT / construction overhead per depth.

Reads the real 30-generation measurements in ``data/runs_emr_gpu.json``. The one-time
JIT-compilation / substrate-construction cost of each run is ``construction_overhead_s``
(paid once, not per generation). Reports the per-depth IQR (the figure's dotted GPU JIT
band) and the pop=1000 mean (the JIT column of the per-generation timing table).

Output: per-depth JIT IQR + pop=1000 JIT means + TikZ coordinates for the GPU JIT band.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GPU_JSON = DATA_DIR / "runs_emr_gpu.json"


def extract_gpu_jit_times():
    runs = json.load(open(GPU_JSON))["runs"]

    by_depth, pop1000 = {}, {}
    for r in runs:
        if not r.get("in_figure_band"):
            continue
        by_depth.setdefault(r["depth"], []).append(r["construction_overhead_s"])
        if r.get("population") == 1000:
            pop1000.setdefault(r["depth"], []).append(r["construction_overhead_s"])

    print("EMR GPU JIT / construction overhead (real 30-generation runs):")
    print("=" * 66)
    print(f"{'Depth':<6} {'Q1 (25%)':<12} {'Q3 (75%)':<12} {'pop=1000 mean':<15} {'N':<4}")
    print("-" * 66)

    q1_coords, q3_coords = [], []
    for depth in sorted(by_depth):
        times = by_depth[depth]
        q1, q3 = np.percentile(times, [25, 75])
        p1000 = np.mean(pop1000[depth]) if depth in pop1000 else float("nan")
        q1_coords.append(f"({depth}, {q1:.1f})")
        q3_coords.append(f"({depth}, {q3:.1f})")
        print(f"{depth:<6} {q1:<12.1f} {q3:<12.1f} {p1000:<15.1f} {len(times):<4}")

    print()
    print("TikZ coordinates for GPU JIT band:")
    print("% GPU JIT IQR (25th-75th percentile across populations)")
    print("\\addplot[name path=gpujitmin, color=oiblue, thick, dotted] coordinates {")
    print(f"    {' '.join(q1_coords)}")
    print("};")
    print("\\addplot[name path=gpujitmax, color=oiblue, thick, dotted] coordinates {")
    print(f"    {' '.join(q3_coords)}")
    print("};")


if __name__ == "__main__":
    extract_gpu_jit_times()
