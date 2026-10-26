#!/usr/bin/env python3
"""Extract EMR-HyperNEAT GPU total-runtime IQR per depth for the runtime figure.

Reads ``data/runs_emr_gpu.json``. Each run records ``depth``, ``population``, ``seed``,
``n_generations`` (= 30), ``total_runtime_s`` (the construction time plus 30 times the
steady per-generation time) and ``in_figure_band`` (whether the run belongs to the runtime-figure
band). The band spans 8 population sizes for depths 1-7, one value each without a seed, and
populations 300/1000 for depths 8-13, whose runs lasted 10 generations (6 for population 300,
seed 123, which solved XOR at depths 8 to 12). At depth 7, population 750 is interpolated and
population 1000's steady time is an estimate; depth 13's population-1000 seed 456 is projected
from depth 12 (see the paper README's timing provenance).

Output: per-depth IQR (25th/50th/75th percentile of ``total_runtime_s``)
plus TikZ coordinates for the GPU band in the runtime comparison figure.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GPU_JSON = DATA_DIR / "runs_emr_gpu.json"


def iqr_by_depth(path):
    """Return {depth: sorted total_runtime_s list} over the in-figure-band runs."""
    runs = json.load(open(path))["runs"]
    by_depth = {}
    for r in runs:
        if not r.get("in_figure_band"):
            continue
        by_depth.setdefault(r["depth"], []).append(r["total_runtime_s"])
    return {d: sorted(v) for d, v in sorted(by_depth.items())}


def extract_gpu_iqr():
    by_depth = iqr_by_depth(GPU_JSON)

    print("EMR GPU total-runtime IQR (real 30-generation runs):")
    print("=" * 62)
    print(f"{'Depth':<6} {'Q1 (25%)':<14} {'Median':<14} {'Q3 (75%)':<14} {'N':<4}")
    print("-" * 62)

    q1_coords, q3_coords = [], []
    for depth, times in by_depth.items():
        q1, median, q3 = np.percentile(times, [25, 50, 75])
        q1_coords.append(f"({depth}, {q1:.1f})")
        q3_coords.append(f"({depth}, {q3:.1f})")
        print(f"{depth:<6} {q1:<14.1f} {median:<14.1f} {q3:<14.1f} {len(times):<4}")

    print()
    print("TikZ coordinates for GPU band:")
    print("% GPU IQR (25th-75th percentile)")
    print("\\addplot[name path=gpumin, color=oiblue, thick, dashed] coordinates {")
    print(f"    {' '.join(q1_coords)}")
    print("};")
    print("\\addplot[name path=gpumax, color=oiblue, thick, dashed] coordinates {")
    print(f"    {' '.join(q3_coords)}")
    print("};")


if __name__ == "__main__":
    extract_gpu_iqr()
