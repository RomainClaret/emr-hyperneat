#!/usr/bin/env python3
"""Extract ES-HyperNEAT (PUREPLES) total-runtime IQR per depth for the runtime figure.

Reads the real 30-generation measurements in ``data/runs_eshn.json``. ES-HyperNEAT
runs on CPU (PUREPLES is CPU-only); each run records ``depth``, ``population``,
``seed``, ``n_generations`` (= 30), ``total_runtime_s`` (measured wall-clock for the
full 30-generation run), and ``in_figure_band``.

Output: per-depth IQR (25th/50th/75th percentile of the real ``total_runtime_s``)
plus TikZ coordinates for the ES-HN band in the runtime comparison figure.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ESHN_JSON = DATA_DIR / "runs_eshn.json"


def iqr_by_depth(path):
    """Return {depth: sorted total_runtime_s list} over the in-figure-band runs."""
    runs = json.load(open(path))["runs"]
    by_depth = {}
    for r in runs:
        if not r.get("in_figure_band"):
            continue
        by_depth.setdefault(r["depth"], []).append(r["total_runtime_s"])
    return {d: sorted(v) for d, v in sorted(by_depth.items())}


def extract_eshn_iqr():
    by_depth = iqr_by_depth(ESHN_JSON)

    print("ES-HN (PUREPLES) total-runtime IQR (real 30-generation runs):")
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
    print("TikZ coordinates for ES-HN band:")
    print("% ES-HN IQR (25th-75th percentile)")
    print("\\addplot[name path=eshnmin, color=oivermillion, thick, dotted] coordinates {")
    print(f"    {' '.join(q1_coords)}")
    print("};")
    print("\\addplot[name path=eshnmax, color=oivermillion, thick, dotted] coordinates {")
    print(f"    {' '.join(q3_coords)}")
    print("};")


if __name__ == "__main__":
    extract_eshn_iqr()
