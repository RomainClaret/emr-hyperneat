#!/usr/bin/env python3
"""Compute EMR-vs-ES-HN total-runtime speedup ratios from the real 30-generation runs.

Reads ``data/runs_eshn.json``, ``data/runs_emr_gpu.json`` and ``data/runs_emr_cpu.json``
and computes, per depth, the IQR (25th/50th/75th percentile) of ``total_runtime_s`` over
the in-figure-band runs. Speedup = ES-HN / EMR at the matching depth.

Two medians are reported because the runtime figure describes an "IQR median ratio":
  * the IQR midpoint, (Q1 + Q3) / 2 -- what the figure band's center reads as;
  * the true median (50th percentile).
At depth 7 (the deepest with an ES-HN baseline) these bracket the GPU speedup at
roughly 59x (midpoint) to 81x (true median); the CPU speedup is about 11x. ES-HN and
EMR CPU are compared over depths 1-7; GPU extends further but ES-HN does not.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def iqr_by_depth(name):
    runs = json.load(open(DATA_DIR / name))["runs"]
    by_depth = {}
    for r in runs:
        if not r.get("in_figure_band"):
            continue
        by_depth.setdefault(r["depth"], []).append(r["total_runtime_s"])
    return {d: np.percentile(v, [25, 50, 75]) for d, v in sorted(by_depth.items())}


def compute_speedup():
    eshn = iqr_by_depth("runs_eshn.json")
    gpu = iqr_by_depth("runs_emr_gpu.json")
    cpu = iqr_by_depth("runs_emr_cpu.json")

    depths = sorted(set(eshn) & set(gpu) & set(cpu))  # depths with an ES-HN baseline

    print("EMR-vs-ES-HN total-runtime speedup (real 30-generation runs)")
    print("=" * 78)
    print("Speedup = ES-HN / EMR.  'mid' = IQR midpoint (Q1+Q3)/2;  'med' = true median.")
    print("-" * 78)
    print(f"{'Depth':<6} {'ES-HN mid':<12} {'GPU mid':<10} {'CPU mid':<10} "
          f"{'GPU x (mid/med)':<18} {'CPU x (mid/med)':<18}")
    print("-" * 78)

    for d in depths:
        e_q1, e_med, e_q3 = eshn[d]
        g_q1, g_med, g_q3 = gpu[d]
        c_q1, c_med, c_q3 = cpu[d]
        e_mid, g_mid, c_mid = (e_q1 + e_q3) / 2, (g_q1 + g_q3) / 2, (c_q1 + c_q3) / 2
        print(f"{d:<6} {e_mid:<12.1f} {g_mid:<10.1f} {c_mid:<10.1f} "
              f"{f'{e_mid/g_mid:.1f}x / {e_med/g_med:.1f}x':<18} "
              f"{f'{e_mid/c_mid:.1f}x / {e_med/c_med:.1f}x':<18}")

    d7 = max(depths)
    e_q1, e_med, e_q3 = eshn[d7]
    g_q1, g_med, g_q3 = gpu[d7]
    c_q1, c_med, c_q3 = cpu[d7]
    e_mid, g_mid, c_mid = (e_q1 + e_q3) / 2, (g_q1 + g_q3) / 2, (c_q1 + c_q3) / 2

    print()
    print(f"Summary at depth {d7} (deepest ES-HN baseline):")
    print("-" * 50)
    print(f"ES-HN: {e_mid:.0f}s midpoint / {e_med:.0f}s median ({e_med/3600:.1f}h)")
    print(f"GPU:   {g_mid:.0f}s midpoint / {g_med:.0f}s median ({g_med/60:.1f}min)")
    print(f"CPU:   {c_mid:.0f}s midpoint / {c_med:.0f}s median ({c_med/60:.1f}min)")
    print(f"ES-HN vs GPU speedup: {e_mid/g_mid:.0f}x (midpoint) to {e_med/g_med:.0f}x (true median)")
    print(f"ES-HN vs CPU speedup: {e_mid/c_mid:.0f}x (midpoint) to {e_med/c_med:.0f}x (true median)")


if __name__ == "__main__":
    compute_speedup()
