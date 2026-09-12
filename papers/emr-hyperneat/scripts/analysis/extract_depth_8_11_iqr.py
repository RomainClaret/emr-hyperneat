#!/usr/bin/env python3
"""Extract EMR GPU total-runtime IQR for depths 8-11 (deep-substrate detail).

Focused view of the deep-substrate region of the runtime figure, reading the real
30-generation measurements in ``data/runs_emr_gpu.json`` (populations 300 and 1000 at
these depths). ``extract_gpu_iqr.py`` reports the full depth 1-13 band; this script
isolates depths 8-11 with the individual per-run totals for inspection.

Output: per-depth IQR for depths 8-11 + the individual run totals + TikZ fragment.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GPU_JSON = DATA_DIR / "runs_emr_gpu.json"


def extract_iqr():
    runs = json.load(open(GPU_JSON))["runs"]

    by_depth = {}
    for r in runs:
        if r.get("in_figure_band") and 8 <= r["depth"] <= 11:
            by_depth.setdefault(r["depth"], []).append(r["total_runtime_s"])

    print("EMR GPU total-runtime IQR, depths 8-11 (real 30-generation runs)")
    print("=" * 60)
    print(f"{'Depth':<6} {'Q1 (25%)':<14} {'Median':<14} {'Q3 (75%)':<14} {'N':<4}")
    print("-" * 60)

    q1_coords, q3_coords = [], []
    for depth in sorted(by_depth):
        times = by_depth[depth]
        q1, median, q3 = np.percentile(times, [25, 50, 75])
        q1_coords.append(f"({depth}, {q1:.1f})")
        q3_coords.append(f"({depth}, {q3:.1f})")
        print(f"{depth:<6} {q1:<14.1f} {median:<14.1f} {q3:<14.1f} {len(times):<4}")

    print()
    print("Individual run totals (seconds):")
    print("-" * 60)
    for depth in sorted(by_depth):
        vals = sorted(by_depth[depth])
        print(f"Depth {depth}: {[f'{t:.1f}' for t in vals]}")

    print()
    print("TikZ coordinates to append to the GPU band (depths 8-11):")
    print("% gpumin extension:")
    print(f"    {' '.join(q1_coords)}")
    print("% gpumax extension:")
    print(f"    {' '.join(q3_coords)}")


if __name__ == "__main__":
    extract_iqr()
