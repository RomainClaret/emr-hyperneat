#!/usr/bin/env python3
"""EMR GPU deep-substrate tail (depths 8-13): total-runtime and JIT IQR from real runs.

Reads the real 30-generation measurements in ``data/runs_emr_gpu.json`` for the
memory-tiered regime: depths 8-12 use GPU<->RAM streaming (weights exceed VRAM) and
depth 13 uses memmap offloading. Every value is a measured 30-generation
``total_runtime_s`` / ``construction_overhead_s`` for populations 300 and 1000 -- there
is no projection or extrapolation. ``extract_gpu_iqr.py`` reports the full depth 1-13
band; this script isolates the deep tail and also emits the JIT band fragment.

Output: per-depth total-runtime IQR + JIT IQR for depths 8-13, plus TikZ fragments to
append to the figure's GPU total-runtime and GPU JIT bands.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GPU_JSON = DATA_DIR / "runs_emr_gpu.json"

DEEP_DEPTHS = range(8, 14)


def main():
    runs = json.load(open(GPU_JSON))["runs"]

    total_by_depth = defaultdict(list)
    jit_by_depth = defaultdict(list)
    pops_by_depth = defaultdict(set)
    for r in runs:
        d = r["depth"]
        if not r.get("in_figure_band") or d not in DEEP_DEPTHS:
            continue
        total_by_depth[d].append(r["total_runtime_s"])
        jit_by_depth[d].append(r["construction_overhead_s"])
        pops_by_depth[d].add(r["population"])

    depths = sorted(total_by_depth)

    print("=" * 78)
    print("EMR GPU deep-substrate tail (depths 8-13, real 30-generation runs)")
    print("=" * 78)
    print(f"{'Depth':>5} {'N':>3} {'Pops':>12} {'Q1 (s)':>14} {'Median (s)':>14} {'Q3 (s)':>14}")
    print("-" * 78)
    total_q1, total_q3 = {}, {}
    for d in depths:
        vals = total_by_depth[d]
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        total_q1[d], total_q3[d] = q1, q3
        pops = ",".join(str(p) for p in sorted(pops_by_depth[d]))
        print(f"{d:>5} {len(vals):>3} {pops:>12} {q1:>14.1f} {median:>14.1f} {q3:>14.1f}")

    print()
    print("GPU JIT / construction overhead (seconds):")
    print(f"{'Depth':>5} {'N':>3} {'Q1 (s)':>14} {'Q3 (s)':>14}")
    print("-" * 42)
    jit_q1, jit_q3 = {}, {}
    for d in depths:
        vals = jit_by_depth[d]
        q1, q3 = np.percentile(vals, [25, 75])
        jit_q1[d], jit_q3[d] = q1, q3
        print(f"{d:>5} {len(vals):>3} {q1:>14.1f} {q3:>14.1f}")

    def frag(mapping):
        return " ".join(f"({d}, {mapping[d]:.1f})" for d in depths)

    print()
    print("-" * 78)
    print("TikZ fragments to append to the depth 1-7 bands (real depths 8-13):")
    print("-" * 78)
    print("% gpumin (total runtime):")
    print(f"    {frag(total_q1)}")
    print("% gpumax (total runtime):")
    print(f"    {frag(total_q3)}")
    print("% gpujitmin:")
    print(f"    {frag(jit_q1)}")
    print("% gpujitmax:")
    print(f"    {frag(jit_q3)}")


if __name__ == "__main__":
    main()
