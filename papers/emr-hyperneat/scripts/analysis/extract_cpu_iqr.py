#!/usr/bin/env python3
"""Extract EMR-HyperNEAT CPU total-runtime IQR per depth for the runtime figure.

Reads the real 30-generation measurements in ``data/runs_emr_cpu.json`` (8 population
sizes x 3 seeds per depth). Each run records ``depth``, ``population``, ``seed``,
``n_generations`` (= 30), ``total_runtime_s`` (measured wall-clock for the full
30-generation run), and ``in_figure_band``. The runtime figure plots the CPU band
through depth 7 (CPU is not the primary deep-substrate comparison; GPU is); deeper
depths present in the data are reported here for completeness.

Output: per-depth IQR (25th/50th/75th percentile of the real ``total_runtime_s``)
plus TikZ coordinates for the CPU band in the runtime comparison figure.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CPU_JSON = DATA_DIR / "runs_emr_cpu.json"


def iqr_by_depth(path):
    """Return {depth: sorted total_runtime_s list} over the in-figure-band runs."""
    runs = json.load(open(path))["runs"]
    by_depth = {}
    for r in runs:
        if not r.get("in_figure_band"):
            continue
        by_depth.setdefault(r["depth"], []).append(r["total_runtime_s"])
    return {d: sorted(v) for d, v in sorted(by_depth.items())}


def extract_cpu_iqr():
    by_depth = iqr_by_depth(CPU_JSON)

    print("EMR CPU total-runtime IQR (real 30-generation runs):")
    print("=" * 62)
    print(f"{'Depth':<6} {'Q1 (25%)':<14} {'Median':<14} {'Q3 (75%)':<14} {'N':<4}")
    print("-" * 62)

    q1_coords, q3_coords = [], []
    for depth, times in by_depth.items():
        q1, median, q3 = np.percentile(times, [25, 50, 75])
        # The figure's CPU band spans depths 1-7.
        if depth <= 7:
            q1_coords.append(f"({depth}, {q1:.1f})")
            q3_coords.append(f"({depth}, {q3:.1f})")
        print(f"{depth:<6} {q1:<14.1f} {median:<14.1f} {q3:<14.1f} {len(times):<4}")

    print()
    print("TikZ coordinates for CPU band (depths 1-7):")
    print("% CPU IQR (25th-75th percentile)")
    print("\\addplot[name path=cpumin, color=oigreen, thick, dashed] coordinates {")
    print(f"    {' '.join(q1_coords)}")
    print("};")
    print("\\addplot[name path=cpumax, color=oigreen, thick, dashed] coordinates {")
    print(f"    {' '.join(q3_coords)}")
    print("};")


if __name__ == "__main__":
    extract_cpu_iqr()
