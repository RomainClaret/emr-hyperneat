"""Statistics audit for the non-parity strengthening experiments (Section 5.8).

Loads all 6 strengthening result directories and computes every statistic
cited in Section 5.8 and Table 11 of main_alife.tex.

Usage:
    python analysis_strengthening.py

Result directories (all in results/):
    circles_per_function_n30/, 18 functions × N=30 on ConcentricCircles
    step_per_function_n30/, 18 functions × N=30 on StepFunction
    sine_regression_per_function_n30/, 18 functions × N=30 on SineRegression
    circles_assignment_n30/, 9 configs × N=30 on ConcentricCircles
    circles_palette_n30/, 5 sizes × N=30 on ConcentricCircles
    pop_parity6_n30/, 6 conditions × N=30 on Parity-6
"""

import json
import os
import glob
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Function definitions (from analysis_paper_statistics.py)
FUNCTION_DEFS = {
    "sin":        lambda x: np.sin(x),
    "osc_adapt":  lambda x: np.sin(x) * (1 - 0.2 * np.abs(x)),
    "burst":      lambda x: np.tanh(x) + 0.5 * np.sin(3 * x),
    "band_pass":  lambda x: np.exp(-np.abs(x - 1)) - np.exp(-np.abs(x + 1)),
    "receptive":  lambda x: np.exp(-x**2) * np.cos(2 * x),
    "resonator":  lambda x: np.sin(x) * np.exp(-np.abs(x) / 3),
    "gauss":      lambda x: np.exp(-x**2),
    "rs_adapt":   lambda x: np.tanh(x) * (1 - 0.3 * np.abs(x)),
    "lts_low":    lambda x: 1.0 / (1.0 + np.exp(-(2 * x - 0.5))),
    "tanh":       lambda x: np.tanh(x),
    "sigmoid":    lambda x: 1.0 / (1.0 + np.exp(-x)),
    "relu":       lambda x: np.maximum(0, x),
    "identity":   lambda x: x,
    "lelu":       lambda x: np.where(x > 0, x, 0.01 * x),
    "softplus":   lambda x: np.log1p(np.exp(x)),
    "fs_fast":    lambda x: 2 * np.maximum(0, x),
    "gain_mod":   lambda x: x / (1 + np.abs(x)),
    "integrate":  lambda x: np.tanh(x) * (1 + 0.2 * np.exp(-np.abs(x))),
}

# Categories from Table 1
PARITY4_TIERS = {
    "multi_crossing": ["sin", "osc_adapt", "burst", "band_pass"],
    "intermediate": ["receptive", "resonator", "gauss", "rs_adapt", "lts_low"],
    "monotonic": ["tanh", "sigmoid", "relu", "identity", "lelu", "softplus",
                  "fs_fast", "gain_mod", "integrate"],
}


def count_local_extrema(func, x_range=(-5, 5), n_points=10000):
    """Count local extrema (peaks + troughs) in the given range."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = func(x)
    dy = np.diff(y)
    sign_changes = np.diff(np.sign(dy))
    return int(np.sum(np.abs(sign_changes) >= 1.5))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_per_function_results(subdir):
    """Load all per-function results from a directory. Returns dict: func_name -> summary."""
    d = os.path.join(RESULTS_DIR, subdir)
    results = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        bn = os.path.basename(f).replace(".json", "")
        if bn == "combined_results":
            continue
        data = load_json(f)
        if "summary" not in data:
            continue
        results[bn] = {
            "n": len(data["results"]),
            "solve_rate": data["summary"]["solve_rate"],
            "solve_count": data["summary"].get("solve_count", 0),
            "avg_fitness": data["summary"]["avg_best_fitness"],
            "std_fitness": data["summary"].get("std_best_fitness", 0),
            "median_gen": data["summary"].get("median_gen_to_solve"),
            "avg_gen": data["summary"].get("avg_gen_to_solve"),
            "std_gen": data["summary"].get("std_gen_to_solve"),
        }
    return results


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_per_function_table(results, label=""):
    """Print a formatted per-function table."""
    print(f"\n  {'Function':<20s} {'N':>3s} {'Solve':>8s} {'Rate':>7s} {'Med Gen':>8s} {'Avg Fit':>8s}")
    print(f"  {'-'*20} {'-'*3} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    total_seeds = 0
    for fname in sorted(results.keys(), key=lambda x: -results[x]["solve_rate"]):
        r = results[fname]
        total_seeds += r["n"]
        mg = f"{r['median_gen']:.0f}" if r["median_gen"] is not None else "---"
        print(f"  {fname:<20s} {r['n']:>3d} {r['solve_count']:>3d}/{r['n']:<3d} "
              f"{r['solve_rate']*100:>6.1f}% {mg:>8s} {r['avg_fitness']:>8.4f}")
    print(f"\n  Total seeds: {total_seeds}")


# ============================================================================
# E-S1: Circles Per-Function (Section 5.8)
# ============================================================================
def compute_circles_per_function():
    print_header("CIRCLES PER-FUNCTION (ConcentricCircles, N=30)")
    results = load_per_function_results("circles_per_function_n30")
    print_per_function_table(results)

    # Three-tier analysis
    top = [f for f, r in results.items() if r["solve_rate"] == 1.0]
    intermediate = [f for f, r in results.items() if 0 < r["solve_rate"] < 1.0]
    bottom = [f for f, r in results.items() if r["solve_rate"] == 0.0]

    print(f"\n  Three-tier structure:")
    print(f"    Top (100%): {', '.join(top)} ({len(top)} functions)")
    int_strs = [f"{f} {results[f]['solve_rate']*100:.1f}%" for f in intermediate]
    print(f"    Intermediate: {', '.join(int_strs)} ({len(intermediate)} functions)")
    print(f"    Bottom (0%): {', '.join(bottom)} ({len(bottom)} functions)")

    # Compare monotonic vs oscillatory
    mono_solves = sum(results[f]["solve_count"] for f in PARITY4_TIERS["monotonic"] if f in results)
    mono_total = sum(results[f]["n"] for f in PARITY4_TIERS["monotonic"] if f in results)
    osc_solves = sum(results[f]["solve_count"] for f in PARITY4_TIERS["multi_crossing"] if f in results)
    osc_total = sum(results[f]["n"] for f in PARITY4_TIERS["multi_crossing"] if f in results)

    # Fisher's exact: monotonic vs multi-crossing
    table = [[osc_solves, osc_total - osc_solves],
             [mono_solves, mono_total - mono_solves]]
    _, fisher_p = stats.fisher_exact(table)
    print(f"\n  Fisher's exact (multi-crossing vs monotonic): p={fisher_p:.2e}")
    print(f"    Multi-crossing: {osc_solves}/{osc_total} ({osc_solves/osc_total*100:.1f}%)")
    print(f"    Monotonic: {mono_solves}/{mono_total} ({mono_solves/mono_total*100:.1f}%)")

    # Spearman: extrema count vs circles solve rate
    extrema = {f: count_local_extrema(FUNCTION_DEFS[f]) for f in results}
    ext_vals = [extrema[f] for f in results]
    sr_vals = [results[f]["solve_rate"] for f in results]
    rho, pval = stats.spearmanr(ext_vals, sr_vals)
    print(f"\n  Spearman (extrema vs circles solve rate): ρ={rho:.3f}, p={pval:.2e}")

    return results


# ============================================================================
# E-S2: Step Per-Function (Section 5.8)
# ============================================================================
def compute_step_per_function():
    print_header("STEP PER-FUNCTION (StepFunction, N=30)")
    results = load_per_function_results("step_per_function_n30")
    print_per_function_table(results)

    # Count by solve rate bins
    at_100 = sum(1 for r in results.values() if r["solve_rate"] == 1.0)
    above_80 = sum(1 for r in results.values() if r["solve_rate"] >= 0.80)
    below_50 = sum(1 for r in results.values() if r["solve_rate"] < 0.50)
    print(f"\n  Functions at 100%: {at_100}/18")
    print(f"  Functions ≥80%: {above_80}/18")
    print(f"  Functions <50%: {below_50}/18")

    return results


# ============================================================================
# E-S3: Sine Per-Function (Section 5.8)
# ============================================================================
def compute_sine_per_function():
    print_header("SINE REGRESSION PER-FUNCTION (N=30)")
    results = load_per_function_results("sine_regression_per_function_n30")
    print_per_function_table(results)

    at_100 = sum(1 for r in results.values() if r["solve_rate"] == 1.0)
    med_gens = [r["median_gen"] for r in results.values() if r["median_gen"] is not None]
    print(f"\n  Functions at 100%: {at_100}/18")
    print(f"  Convergence range: {min(med_gens):.0f}--{max(med_gens):.0f} median gens")
    print(f"  Speed ratio: {max(med_gens)/max(min(med_gens),1):.0f}×")

    return results


# ============================================================================
# E-S4: Circles Assignment (Section 5.8)
# ============================================================================
def compute_circles_assignment():
    print_header("CIRCLES ASSIGNMENT (ConcentricCircles, N=30)")
    results = load_per_function_results("circles_assignment_n30")
    print_per_function_table(results)

    # Fisher's exact: cppn_output_6 vs disabled
    if "cppn_output_6" in results and "disabled" in results:
        a = results["cppn_output_6"]
        b = results["disabled"]
        table = [[a["solve_count"], a["n"] - a["solve_count"]],
                 [b["solve_count"], b["n"] - b["solve_count"]]]
        _, p = stats.fisher_exact(table)
        print(f"\n  Fisher's exact (cppn_output_6 vs disabled): p={p:.2e}")

    return results


# ============================================================================
# E-S5: Circles Palette (Section 5.8)
# ============================================================================
def compute_circles_palette():
    print_header("CIRCLES PALETTE (ConcentricCircles, N=30)")
    results = load_per_function_results("circles_palette_n30")
    print_per_function_table(results)

    # Spearman: palette size vs solve rate
    size_map = {"gauss_only": 1, "size_3": 3, "size_6": 6, "size_8": 8, "size_13": 13}
    sizes = [size_map[k] for k in results if k in size_map]
    rates = [results[k]["solve_rate"] for k in results if k in size_map]
    if len(sizes) >= 3:
        rho, pval = stats.spearmanr(sizes, rates)
        print(f"\n  Spearman (palette size vs solve rate): ρ={rho:.3f}, p={pval:.4f}")

    return results


# ============================================================================
# E-S6: Pop Parity-6 (Section 5.3)
# ============================================================================
def compute_pop_parity6():
    print_header("POP PARITY-6 (N=30)")
    results = load_per_function_results("pop_parity6_n30")
    print_per_function_table(results)
    return results


# ============================================================================
# Cross-problem comparison (Table 11)
# ============================================================================
def compute_cross_problem_table(circles, step, sine):
    print_header("CROSS-PROBLEM COMPARISON (Table 11)")

    # Parity-4 data (from existing experiments, hardcoded for comparison)
    parity4 = {
        "sin": 100, "osc_adapt": 100, "burst": 100, "band_pass": 100,
        "receptive": 80, "resonator": 66.7, "gauss": 33.3, "rs_adapt": 23.3,
        "lts_low": 6.7, "tanh": 0, "sigmoid": 0, "relu": 0, "identity": 0,
        "lelu": 0, "softplus": 0, "fs_fast": 0, "gain_mod": 0, "integrate": 0,
    }

    print(f"\n  {'Function':<15s} {'Parity-4':>9s} {'Sine':>9s} {'Step':>9s} {'Circles':>9s}")
    print(f"  {'-'*15} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

    for fname in ["sin", "osc_adapt", "burst", "band_pass",
                   "receptive", "resonator", "gauss", "rs_adapt", "lts_low",
                   "tanh", "sigmoid", "relu", "identity", "lelu", "softplus",
                   "fs_fast", "gain_mod", "integrate"]:
        p4 = f"{parity4.get(fname, 0):.0f}%"
        sn = f"{sine[fname]['solve_rate']*100:.0f}%" if fname in sine else "---"
        st = f"{step[fname]['solve_rate']*100:.1f}%" if fname in step else "---"
        ci = f"{circles[fname]['solve_rate']*100:.1f}%" if fname in circles else "---"
        print(f"  {fname:<15s} {p4:>9s} {sn:>9s} {st:>9s} {ci:>9s}")

    # Cross-problem Spearman (parity-4 vs circles)
    common = [f for f in parity4 if f in circles]
    p4_vals = [parity4[f] for f in common]
    ci_vals = [circles[f]["solve_rate"] * 100 for f in common]
    rho, pval = stats.spearmanr(p4_vals, ci_vals)
    print(f"\n  Spearman (Parity-4 vs Circles solve rates): ρ={rho:.3f}, p={pval:.4f}")


# ============================================================================
# Total seed count
# ============================================================================
def compute_total_seeds():
    print_header("TOTAL STRENGTHENING SEEDS")
    dirs = {
        "circles_per_function_n30": "Circles per-function",
        "step_per_function_n30": "Step per-function",
        "sine_regression_per_function_n30": "Sine per-function",
        "circles_assignment_n30": "Circles assignment",
        "circles_palette_n30": "Circles palette",
        "pop_parity6_n30": "Pop Parity-6",
    }
    grand = 0
    for subdir, label in dirs.items():
        d = os.path.join(RESULTS_DIR, subdir)
        total = 0
        for f in glob.glob(os.path.join(d, "*.json")):
            if "combined" in os.path.basename(f):
                continue
            try:
                data = load_json(f)
                total += len(data.get("results", []))
            except Exception:
                pass
        grand += total
        print(f"  {label:<25s}: {total:>5d} seeds")
    print(f"  {'TOTAL':<25s}: {grand:>5d} seeds")
    return grand


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  STRENGTHENING EXPERIMENTS — FINAL STATISTICS AUDIT")
    print("  Paper: Per-Node Activation Function Evolution (ALIFE 2026)")
    print("  Section 5.8 + Table 11")
    print("=" * 70)

    circles = compute_circles_per_function()
    step = compute_step_per_function()
    sine = compute_sine_per_function()
    compute_circles_assignment()
    compute_circles_palette()
    compute_pop_parity6()
    compute_cross_problem_table(circles, step, sine)
    compute_total_seeds()

    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print("=" * 70)
