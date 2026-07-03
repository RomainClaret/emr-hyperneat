"""E3: Monotonic plateau fitness analysis.

Extract best_fitness statistics for monotonic functions from per-function ablation data.
Reports mean/std plateau fitness to show monotonic functions aren't helpless, they
solve ~88% of Parity-4 patterns but can't cross the 0.95 threshold.
"""

import json
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "per_function_ablation_n30")

# 9 strictly monotonic functions (0% solve rate on Parity-4 FF)
MONOTONIC_FUNCTIONS = [
    "tanh", "sigmoid", "relu", "identity", "lelu",
    "softplus", "fs_fast", "gain_mod", "integrate"
]

# All 18 functions for complete picture
ALL_FUNCTIONS = [
    "tanh", "sigmoid", "relu", "identity", "sin", "gauss",
    "lelu", "softplus", "rs_adapt", "fs_fast", "lts_low",
    "burst", "resonator", "osc_adapt", "gain_mod", "receptive",
    "band_pass", "integrate"
]


def load_function_data(func_name: str) -> list[dict]:
    """Load per-seed results for a given function."""
    path = os.path.join(RESULTS_DIR, f"{func_name}.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("results", data) if isinstance(data, dict) else data


def analyze_plateau():
    """Compute plateau fitness statistics for all functions."""
    print("=" * 70)
    print("E3: Monotonic Plateau Fitness Analysis")
    print("=" * 70)

    all_results = {}
    for func in ALL_FUNCTIONS:
        results = load_function_data(func)
        if not results:
            continue
        best_fitnesses = [r["best_fitness"] for r in results if not r.get("solved", False)]
        solved_count = sum(1 for r in results if r.get("solved", False))
        total = len(results)
        all_results[func] = {
            "best_fitnesses": best_fitnesses,
            "solved": solved_count,
            "total": total,
            "solve_rate": solved_count / total if total > 0 else 0,
        }

    # Report monotonic functions specifically
    print("\n--- Monotonic Functions (0% solve rate, feedforward) ---")
    print(f"{'Function':<12} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'~Patterns':>10}")
    print("-" * 62)

    mono_all_fitnesses = []
    for func in MONOTONIC_FUNCTIONS:
        if func not in all_results:
            continue
        fitnesses = all_results[func]["best_fitnesses"]
        if not fitnesses:
            continue
        arr = np.array(fitnesses)
        mono_all_fitnesses.extend(fitnesses)
        # Parity-4 has 16 patterns. fitness = correct/16.
        # So mean_fitness * 16 = approx correct patterns
        approx_correct = np.mean(arr) * 16
        print(f"{func:<12} {len(arr):>4} {np.mean(arr):>8.4f} {np.std(arr):>8.4f} "
              f"{np.min(arr):>8.4f} {np.max(arr):>8.4f} {approx_correct:>10.1f}/16")

    if mono_all_fitnesses:
        arr = np.array(mono_all_fitnesses)
        print("-" * 62)
        print(f"{'ALL MONO':<12} {len(arr):>4} {np.mean(arr):>8.4f} {np.std(arr):>8.4f} "
              f"{np.min(arr):>8.4f} {np.max(arr):>8.4f} {np.mean(arr)*16:>10.1f}/16")

    # Report all functions for complete picture
    print("\n--- All Functions Summary ---")
    print(f"{'Function':<12} {'Solve%':>7} {'N_unsolved':>10} {'Plateau Mean':>13} {'Plateau Std':>12}")
    print("-" * 56)
    for func in ALL_FUNCTIONS:
        if func not in all_results:
            continue
        r = all_results[func]
        fitnesses = r["best_fitnesses"]
        solve_pct = r["solve_rate"] * 100
        if fitnesses:
            arr = np.array(fitnesses)
            print(f"{func:<12} {solve_pct:>6.1f}% {len(arr):>10} {np.mean(arr):>13.4f} {np.std(arr):>12.4f}")
        else:
            print(f"{func:<12} {solve_pct:>6.1f}% {0:>10} {'N/A':>13} {'N/A':>12}")

    # Key stats for paper
    if mono_all_fitnesses:
        arr = np.array(mono_all_fitnesses)
        print("\n" + "=" * 70)
        print("KEY STATS FOR PAPER (copy these into Section 4.3):")
        print(f"  Monotonic mean plateau fitness: {np.mean(arr):.3f} +/- {np.std(arr):.3f}")
        print(f"  Approximate correct patterns: {np.mean(arr)*16:.1f}/16")
        print(f"  Range: [{np.min(arr):.3f}, {np.max(arr):.3f}]")
        print(f"  N seeds (total across 9 functions): {len(arr)}")
        print(f"  Threshold: 0.95 (15.2/16 correct)")
        print("=" * 70)


if __name__ == "__main__":
    analyze_plateau()
