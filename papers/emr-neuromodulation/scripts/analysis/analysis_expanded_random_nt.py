#!/usr/bin/env python3
"""E-S11 Expanded Random NT Analysis.

Combines original 10 random NT sets (sets 0-9) with expanded sets (10-29)
to assess whether hand-designed NT vectors are uniquely privileged.

Re-runnable: just execute to get updated numbers as more sets complete.

Results:
  - Original: results/strengthening/random_nt/random_with_inv_set*.json
  - Expanded: results/expanded_random_nt/random_set_*.json
"""

import json
import glob
import os
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
ORIGINAL_DIR = RESULTS_DIR / "strengthening" / "random_nt"
EXPANDED_DIR = RESULTS_DIR / "expanded_random_nt"


def load_original_sets():
    """Load original 10 random NT sets (with inversion only)."""
    sets = {}
    files = sorted(glob.glob(str(ORIGINAL_DIR / "random_with_inv_set*.json")))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        set_idx = d.get("random_set_idx", int(os.path.basename(f).split("set")[1].split("_")[0]))
        if set_idx not in sets:
            sets[set_idx] = {
                "seeds": [],
                "converged": 0,
                "total": 0,
                "nt_vectors": d.get("nt_vectors", {}),
            }
        sets[set_idx]["total"] += 1
        sets[set_idx]["seeds"].append(d)
        if d.get("converged", False):
            sets[set_idx]["converged"] += 1
    return sets


def load_expanded_sets():
    """Load expanded random NT sets (10-29)."""
    sets = {}
    files = sorted(glob.glob(str(EXPANDED_DIR / "random_set_*.json")))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        set_idx = d["set_index"]
        n_seeds = d["n_seeds"]
        converged = d["convergence_count"]
        sets[set_idx] = {
            "total": n_seeds,
            "converged": converged,
            "convergence_rate": d["convergence_rate"],
            "nt_vectors": d["nt_vectors"],
            "or_nor_distance": d["or_nor_distance"],
            "or_nor_direction_angle": d["or_nor_direction_angle"],
            "results": d.get("results", []),
        }
    return sets


def compute_nt_distance(nt_a, nt_b):
    """Euclidean distance between two NT vectors (first 3 components)."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(nt_a[:3], nt_b[:3])))


def compute_nt_angle(nt_a, nt_b):
    """Angle between two NT vectors (first 3 components) in degrees."""
    a = np.array(nt_a[:3])
    b = np.array(nt_b[:3])
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def binomial_ci(k, n, alpha=0.05):
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    from scipy import stats
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def main():
    print("=" * 70)
    print("E-S11: Expanded Random NT Analysis")
    print("=" * 70)

    # Load original sets
    original = load_original_sets()
    print(f"\n--- Original Sets (0-9) ---")
    total_orig = 0
    conv_orig = 0
    for s in sorted(original.keys()):
        info = original[s]
        total_orig += info["total"]
        conv_orig += info["converged"]
        rate = info["converged"] / info["total"] * 100 if info["total"] > 0 else 0
        marker = " *** 100% ***" if info["converged"] == info["total"] and info["total"] > 0 else ""
        print(f"  Set {s:2d}: {info['converged']:2d}/{info['total']:2d} ({rate:5.1f}%){marker}")

    print(f"  Original total: {conv_orig}/{total_orig} ({conv_orig/total_orig*100:.1f}%)")

    # Load expanded sets
    expanded = load_expanded_sets()
    print(f"\n--- Expanded Sets (10-29) ---")
    total_exp = 0
    conv_exp = 0
    if not expanded:
        print("  No expanded sets found yet.")
    else:
        for s in sorted(expanded.keys()):
            info = expanded[s]
            total_exp += info["total"]
            conv_exp += info["converged"]
            rate = info["converged"] / info["total"] * 100 if info["total"] > 0 else 0
            complete = "complete" if info["total"] == 30 else f"partial ({info['total']}/30)"
            or_nor_d = info["or_nor_distance"]
            or_nor_a = info["or_nor_direction_angle"]

            # Compute key NT distances
            nt = info["nt_vectors"]
            xor_and_dist = compute_nt_distance(nt["xor"][:3], nt["and"][:3])
            xor_or_dist = compute_nt_distance(nt["xor"][:3], nt["or"][:3])
            and_or_dist = compute_nt_distance(nt["and"][:3], nt["or"][:3])

            print(f"  Set {s:2d}: {info['converged']:2d}/{info['total']:2d} ({rate:5.1f}%) "
                  f"[{complete}] OR-NOR dist={or_nor_d:.3f}")
            print(f"         NT dists: XOR-AND={xor_and_dist:.3f} XOR-OR={xor_or_dist:.3f} "
                  f"AND-OR={and_or_dist:.3f}")

        exp_rate = f"{conv_exp/total_exp*100:.1f}%" if total_exp > 0 else "N/A"
        print(f"  Expanded total: {conv_exp}/{total_exp} ({exp_rate})")

    # Combined analysis
    total_all = total_orig + total_exp
    conv_all = conv_orig + conv_exp
    print(f"\n--- Combined Analysis ---")
    print(f"  Sets completed: {len(original)} original + {len(expanded)} expanded = {len(original) + len(expanded)}")
    print(f"  Seeds completed: {total_all}")
    print(f"  Converged: {conv_all}/{total_all} ({conv_all/total_all*100:.1f}%)")

    # Binomial CI
    try:
        rate, ci_lo, ci_hi = binomial_ci(conv_all, total_all)
        print(f"  95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
    except ImportError:
        print("  (scipy not available for CI computation)")

    # Sets achieving 100%
    full_success_sets = []
    for s in sorted(original.keys()):
        if original[s]["converged"] == 30:
            full_success_sets.append(s)
    for s in sorted(expanded.keys()):
        if expanded[s]["converged"] == 30:
            full_success_sets.append(s)
    print(f"  Sets with 100% convergence: {full_success_sets if full_success_sets else 'none'}")
    print(f"  Rate of full-success sets: {len(full_success_sets)}/{len(original) + len(expanded)}")

    # E-S5 prediction test: does geometric distance predict success?
    print(f"\n--- E-S5 Prediction Test ---")
    or_nor_dists = ', '.join(f'{expanded[s]["or_nor_distance"]:.3f}' for s in sorted(expanded.keys()))
    print(f"  OR-NOR distance for all expanded sets: {or_nor_dists}")
    print(f"  (All 0.0 = compositional design shares OR's vector for NOR)")
    print(f"  Prediction: geometric properties alone are insufficient (R²=0.31 from H4 analysis)")

    # Failure pattern analysis
    print(f"\n--- Failure Pattern Analysis ---")
    xor_failures = 0
    other_failures = 0
    for s in sorted(expanded.keys()):
        for result in expanded[s]["results"]:
            if not result["converged"]:
                ptf = result["per_task_fitness"]
                if ptf.get("xor", 1.0) < 0.95:
                    xor_failures += 1
                else:
                    other_failures += 1
    total_failures = xor_failures + other_failures
    if total_failures > 0:
        print(f"  XOR as bottleneck: {xor_failures}/{total_failures} failures ({xor_failures/total_failures*100:.1f}%)")
        print(f"  Other task failures: {other_failures}/{total_failures}")
    else:
        print(f"  No failures to analyze (all converged)")

    # Summary for paper text
    print(f"\n{'=' * 70}")
    print(f"PAPER TEXT SUMMARY (copy-paste ready)")
    print(f"{'=' * 70}")
    n_expanded_sets = len(expanded)
    n_expanded_complete = sum(1 for s in expanded.values() if s["total"] == 30)
    n_expanded_partial = n_expanded_sets - n_expanded_complete
    print(f"  Expanded sets completed: {n_expanded_complete} of 20 additional sets")
    if n_expanded_partial > 0:
        partial_info = ', '.join(
            f'set {s}={expanded[s]["total"]}/30'
            for s in sorted(expanded.keys()) if expanded[s]['total'] < 30
        )
        print(f"  Partial sets: {n_expanded_partial} (seeds: {partial_info})")
    print(f"  New runs: {total_exp}")
    print(f"  New converged: {conv_exp}")
    print(f"  Combined with original: {conv_all}/{total_all} ({conv_all/total_all*100:.1f}%)")
    print(f"  Grand total experiment count: 9,399 + {total_exp} = {9399 + total_exp}")


if __name__ == "__main__":
    main()
