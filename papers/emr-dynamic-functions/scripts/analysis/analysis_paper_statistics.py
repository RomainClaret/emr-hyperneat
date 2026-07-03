"""Comprehensive statistics audit for the dynamic functions paper.

Loads ALL result JSON files and computes every statistic cited in the paper.
Outputs a structured report for paper verification.

Usage:
    python analysis_paper_statistics.py
"""

import json
import os
import glob
import numpy as np
from scipy import stats
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ============================================================================
# Function definitions for extrema counting (from analysis_extrema_correlation.py)
# ============================================================================
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

MONOTONIC_CATEGORY = {
    "sigmoid": "Monotonic/Bounded",
    "gauss": "Radial",
    "lts_low": "Monotonic/Bounded",
    "tanh": "Monotonic/Bounded",
    "relu": "Monotonic/Unbounded",
    "softplus": "Monotonic/Unbounded",
    "identity": "Linear",
    "lelu": "Monotonic/Unbounded",
}


def count_local_extrema(func, x_range=(-5, 5), n_points=10000):
    """Count local extrema (peaks + troughs) in the given range."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = func(x)
    dy = np.diff(y)
    sign_changes = np.diff(np.sign(dy))
    extrema = np.sum(np.abs(sign_changes) >= 1.5)
    return int(extrema)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================================
# E1: Per-function ablation (Table 1 / Section 4.3)
# ============================================================================
def compute_e1_per_function_ablation():
    """Per-function ablation on Parity-4 (N=30)."""
    report = {"experiment": "E1: Per-function ablation on Parity-4"}
    ablation_dir = os.path.join(RESULTS_DIR, "per_function_ablation_n30")

    results = {}
    for fname in FUNCTION_DEFS:
        fpath = os.path.join(ablation_dir, f"{fname}.json")
        if not os.path.exists(fpath):
            continue
        d = load_json(fpath)
        summary = d["summary"]
        results[fname] = {
            "solve_rate": summary["solve_rate"],
            "solve_count": summary["solve_count"],
            "total_runs": summary["total_runs"],
            "avg_gen": summary.get("avg_gen_to_solve"),
            "avg_fitness": summary.get("avg_best_fitness"),
        }

    report["functions"] = results
    report["n_functions"] = len(results)

    # Three-tier classification (solve_rate stored as fraction 0-1)
    tier_100 = [f for f, r in results.items() if r["solve_rate"] >= 0.999]
    tier_mid = [f for f, r in results.items() if 0.001 < r["solve_rate"] < 0.999]
    tier_0 = [f for f, r in results.items() if r["solve_rate"] <= 0.001]
    report["tiers"] = {
        "100%": {"count": len(tier_100), "functions": tier_100},
        "intermediate": {
            "count": len(tier_mid),
            "functions": tier_mid,
            "range": f"{min(r['solve_rate']*100 for f, r in results.items() if f in tier_mid):.1f}--{max(r['solve_rate']*100 for f, r in results.items() if f in tier_mid):.1f}%" if tier_mid else "N/A",
        },
        "0%": {"count": len(tier_0), "functions": tier_0},
    }
    return report


# ============================================================================
# E1b: Extrema vs solvability Spearman rho (Section 4.3)
# ============================================================================
def compute_e1b_extrema_correlation():
    """Spearman rho between extrema count and solve rate."""
    report = {"experiment": "E1b: Extrema-solvability Spearman correlation"}

    ablation_dir = os.path.join(RESULTS_DIR, "per_function_ablation_n30")
    names, solve_rates, extrema_counts = [], [], []

    for fname, func in FUNCTION_DEFS.items():
        fpath = os.path.join(ablation_dir, f"{fname}.json")
        if not os.path.exists(fpath):
            continue
        d = load_json(fpath)
        sr = d["summary"]["solve_rate"]
        ec = count_local_extrema(func)
        names.append(fname)
        solve_rates.append(sr)
        extrema_counts.append(ec)

    rho, p = stats.spearmanr(extrema_counts, solve_rates)
    report["spearman_rho"] = round(rho, 2)
    report["p_value"] = float(f"{p:.2e}")
    report["n_functions"] = len(names)
    report["paper_claims"] = {"rho": 0.81, "p": "4.4e-5"}
    return report


# ============================================================================
# E1c: Cross-task Spearman rho (Parity-4 vs XOR)
# ============================================================================
def compute_e1c_cross_task_correlation():
    """Spearman rho between Parity-4 and XOR solve rates."""
    report = {"experiment": "E1c: Cross-task Spearman correlation (Parity-4 vs XOR)"}

    parity_dir = os.path.join(RESULTS_DIR, "per_function_ablation_n30")
    xor_dir = os.path.join(RESULTS_DIR, "xor_per_function_n30")

    parity_rates, xor_rates, names = [], [], []
    for fname in FUNCTION_DEFS:
        p_path = os.path.join(parity_dir, f"{fname}.json")
        x_path = os.path.join(xor_dir, f"{fname}.json")
        if not os.path.exists(p_path) or not os.path.exists(x_path):
            continue
        p_data = load_json(p_path)
        x_data = load_json(x_path)
        parity_rates.append(p_data["summary"]["solve_rate"])
        xor_rates.append(x_data["summary"]["solve_rate"])
        names.append(fname)

    if len(names) >= 3:
        rho, p = stats.spearmanr(parity_rates, xor_rates)
        report["spearman_rho"] = round(rho, 2)
        report["p_value"] = float(f"{p:.2e}")
    report["n_functions"] = len(names)
    return report


# ============================================================================
# E2: Parity scaling sin/tanh (Table 3)
# ============================================================================
def compute_e2_parity_scaling():
    """Parity scaling: sin vs tanh, Parity-2 through Parity-8."""
    report = {"experiment": "E2: Parity scaling (sin vs tanh)"}

    # Sin scaling
    sin_dir = os.path.join(RESULTS_DIR, "parity_scaling_n30")
    sin_results = {}
    for n_bits in range(2, 9):
        fpath = os.path.join(sin_dir, f"parity_{n_bits}.json")
        if not os.path.exists(fpath):
            continue
        d = load_json(fpath)
        sin_results[n_bits] = {
            "solve_rate": d["summary"]["solve_rate"],
            "avg_gen": d["summary"].get("avg_gen_to_solve"),
            "N": d["summary"]["total_runs"],
        }
    report["sin_scaling"] = sin_results

    # Tanh scaling (monotonic)
    tanh_dir = os.path.join(RESULTS_DIR, "monotonic_parity_scaling_n30")
    tanh_results = {}
    for n_bits in range(2, 9):
        fpath = os.path.join(tanh_dir, f"parity_{n_bits}.json")
        if not os.path.exists(fpath):
            continue
        d = load_json(fpath)
        tanh_results[n_bits] = {
            "solve_rate": d["summary"]["solve_rate"],
            "avg_gen": d["summary"].get("avg_gen_to_solve"),
            "avg_fitness": d["summary"].get("avg_best_fitness"),
            "N": d["summary"]["total_runs"],
        }
    report["tanh_scaling"] = tanh_results
    return report


# ============================================================================
# E3: Population sensitivity (Table 4)
# ============================================================================
def compute_e3_pop_sensitivity():
    """Population sensitivity: sin and tanh across pop sizes."""
    report = {"experiment": "E3: Population sensitivity"}

    pop_dir = os.path.join(RESULTS_DIR, "pop_sensitivity_n30")
    results = {}
    for fpath in sorted(glob.glob(os.path.join(pop_dir, "*.json"))):
        d = load_json(fpath)
        act = d["activation"]
        pop = d["pop_size"]
        key = f"{act}_pop{pop}"
        results[key] = {
            "activation": act,
            "pop_size": pop,
            "solve_rate": d["summary"]["solve_rate"],
            "median_gen": d["summary"].get("median_gen"),
            "N": d["summary"]["n"],
        }
    report["results"] = results
    return report


# ============================================================================
# E5: Assignment method comparison (Table 5)
# ============================================================================
def compute_e5_assignment_methods():
    """Assignment method comparison on Two Spirals."""
    report = {"experiment": "E5: Assignment method comparison (Two Spirals)"}

    ts_combined = os.path.join(RESULTS_DIR, "two_spirals_n30", "combined_results.json")
    if not os.path.exists(ts_combined):
        report["error"] = "combined_results.json not found"
        return report

    d = load_json(ts_combined)
    results = {}
    cppn_fitnesses = {}
    baseline_fitnesses = {}

    for config_name, config_data in d.items():
        if config_name == "metadata":
            continue
        summary = config_data.get("overall_summary", {})
        results[config_name] = {
            "mean_fitness": summary.get("mean_fitness"),
            "std_fitness": summary.get("std_fitness"),
        }
        # Collect fitnesses per config for Mann-Whitney
        config_results = config_data.get("results", [])
        fitnesses = [r.get("best_fitness", 0) for r in config_results if isinstance(r, dict)]
        if config_name in ("cppn_output_4", "cppn_output_6"):
            cppn_fitnesses[config_name] = fitnesses
        elif config_name in ("disabled", "global_tanh", "global_sin",
                             "weight_interp_magnitude", "weight_interp_sign",
                             "weight_interp_variance"):
            baseline_fitnesses[config_name] = fitnesses

    # Mann-Whitney U test: cppn_output_4 vs disabled (paper's primary comparison)
    cppn4 = cppn_fitnesses.get("cppn_output_4", [])
    disabled = baseline_fitnesses.get("disabled", [])
    if cppn4 and disabled:
        u_stat, p_val = stats.mannwhitneyu(cppn4, disabled, alternative="greater")
        n1, n2 = len(cppn4), len(disabled)
        r = 1 - (2 * u_stat) / (n1 * n2)  # rank-biserial
        report["mann_whitney_cppn4_vs_disabled"] = {
            "U": float(u_stat),
            "p": float(f"{p_val:.2e}"),
            "n_cppn": n1,
            "n_disabled": n2,
            "rank_biserial_r": round(abs(r), 2),
        }
        report["paper_claims"] = {"p": "3.0e-11", "r": 1.0, "U": 900}

    report["configs"] = results
    return report


# ============================================================================
# E6: Palette size Spearman rho (Table 6)
# ============================================================================
def compute_e6_palette_size():
    """Palette size vs convergence speed, Spearman rho.

    NOTE: The palette size data is encoded in the paper tables (hardcoded),
    not in separate result files. We reproduce the table values here.
    """
    report = {"experiment": "E6: Palette size Spearman correlation"}

    # From Table 6 in the paper
    palette_sizes = [1, 3, 4, 5, 6, 7, 8, 13]
    avg_gens = [1.7, 3.4, 3.1, 3.2, 2.6, 3.3, 3.4, 9.3]

    rho, p = stats.spearmanr(palette_sizes, avg_gens)
    report["spearman_rho"] = round(rho, 2)
    report["p_value"] = round(p, 2)
    report["paper_claims"] = {"rho": 0.64, "p": 0.09}
    return report


# ============================================================================
# E4: Recurrent per-function (Table 7) + Parity-8 monotonic partial
# ============================================================================
def compute_e4_recurrent():
    """Recurrent per-function ablation + Parity-8 partial monotonic results."""
    report = {"experiment": "E4: Recurrent architectures"}

    # Parity-8 recurrent results (partial for monotonic)
    rec_dir = os.path.join(RESULTS_DIR, "recurrent_parity8_n30")
    if not os.path.exists(rec_dir):
        report["error"] = "recurrent_parity8_n30 directory not found"
        return report

    osc_funcs = {"sin", "burst", "resonator", "osc_adapt"}
    func_results = defaultdict(list)

    for fpath in sorted(glob.glob(os.path.join(rec_dir, "*_parity8_seed*.json"))):
        fname_base = os.path.basename(fpath)
        func = fname_base.split("_parity8_seed")[0]
        d = load_json(fpath)
        func_results[func].append(d)

    # Oscillatory functions (complete N=30)
    osc_report = {}
    for func in sorted(osc_funcs):
        seeds = func_results.get(func, [])
        n = len(seeds)
        solved = sum(1 for s in seeds if s.get("solved", False))
        solved_gens = [s["solved_gen"] for s in seeds if s.get("solved") and s.get("solved_gen")]
        avg_gen = round(np.mean(solved_gens), 1) if solved_gens else None
        osc_report[func] = {
            "N": n,
            "solved": solved,
            "rate": round(solved / n * 100, 1) if n > 0 else 0,
            "avg_gen": avg_gen,
        }
    report["oscillatory_parity8"] = osc_report

    # Monotonic functions (partial)
    mono_report = {}
    bounded_solved, bounded_total = 0, 0
    unbounded_solved, unbounded_total = 0, 0

    for func in sorted(set(func_results.keys()) - osc_funcs):
        seeds = func_results[func]
        n = len(seeds)
        solved = sum(1 for s in seeds if s.get("solved", False))
        rate = round(solved / n * 100, 1) if n > 0 else 0
        solved_gens = [s["solved_gen"] for s in seeds if s.get("solved") and s.get("solved_gen")]
        avg_gen = round(np.mean(solved_gens), 1) if solved_gens else None
        avg_fit = round(np.mean([s.get("best_fitness", 0) for s in seeds]), 3)

        cat = MONOTONIC_CATEGORY.get(func, seeds[0].get("category", "Unknown"))
        mono_report[func] = {
            "category": cat,
            "N": n,
            "solved": solved,
            "rate": rate,
            "avg_gen": avg_gen,
            "avg_fitness": avg_fit,
        }

        # Bounded vs unbounded classification
        if cat in ("Monotonic/Bounded", "Radial"):
            bounded_solved += solved
            bounded_total += n
        elif cat in ("Monotonic/Unbounded", "Linear"):
            unbounded_solved += solved
            unbounded_total += n

    report["monotonic_parity8_partial"] = mono_report

    # Fisher's exact test: bounded vs unbounded
    if bounded_total > 0 and unbounded_total > 0:
        table = [[bounded_solved, bounded_total - bounded_solved],
                 [unbounded_solved, unbounded_total - unbounded_solved]]
        odds, p = stats.fisher_exact(table, alternative="greater")
        report["fisher_bounded_vs_unbounded"] = {
            "bounded": f"{bounded_solved}/{bounded_total} ({round(bounded_solved/bounded_total*100, 1)}%)",
            "unbounded": f"{unbounded_solved}/{unbounded_total} ({round(unbounded_solved/unbounded_total*100, 1) if unbounded_total > 0 else 0}%)",
            "odds_ratio": round(odds, 2) if np.isfinite(odds) else "inf",
            "p_value": float(f"{p:.4e}"),
            "total_seeds": bounded_total + unbounded_total,
        }

    return report


# ============================================================================
# E5b: Recurrence type (Table 8)
# ============================================================================
def compute_e5b_recurrence_type():
    """Recurrence type comparison on Parity-4."""
    report = {"experiment": "E5b: Recurrence type comparison"}

    rt_dir = os.path.join(RESULTS_DIR, "recurrence_type_n30")
    results = {}
    for fpath in sorted(glob.glob(os.path.join(rt_dir, "*.json"))):
        fname = os.path.basename(fpath).replace(".json", "")
        if fname == "combined":
            continue
        d = load_json(fpath)
        summary = d.get("summary", {})
        results[fname] = {
            "solve_rate": summary.get("solve_rate", 0),
            "solve_count": summary.get("solve_count", 0),
            "total_runs": summary.get("total_runs", 0),
            "avg_gen": summary.get("avg_gen_to_solve"),
            "median_gen": summary.get("median_gen_to_solve"),
        }

    report["presets"] = results

    # Fisher's exact: feedforward vs hidden_only
    ff = results.get("feedforward", {})
    ho = results.get("hidden_only", {})
    if ff and ho:
        ff_solved = ff.get("solve_count", 0)
        ff_total = ff.get("total_runs", 30)
        ho_solved = ho.get("solve_count", 0)
        ho_total = ho.get("total_runs", 30)
        table = [[ho_solved, ho_total - ho_solved],
                 [ff_solved, ff_total - ff_solved]]
        odds, p = stats.fisher_exact(table, alternative="greater")
        report["fisher_hidden_vs_ff"] = {
            "hidden_only": f"{ho_solved}/{ho_total}",
            "feedforward": f"{ff_solved}/{ff_total}",
            "p_value": float(f"{p:.4e}"),
        }

    return report


# ============================================================================
# E6b: Gradient baseline (Table 10)
# ============================================================================
def compute_e6b_gradient_baseline():
    """Gradient baseline comparison."""
    report = {"experiment": "E6b: Gradient baseline"}

    gb_dir = os.path.join(RESULTS_DIR, "gradient_baseline_n30")
    results = {}
    for fpath in sorted(glob.glob(os.path.join(gb_dir, "parity*.json"))):
        fname = os.path.basename(fpath).replace(".json", "")
        d = load_json(fpath)
        if isinstance(d, list):
            n = len(d)
            solved = sum(1 for r in d if r.get("solved", False) or r.get("best_accuracy", 0) >= 1.0)
            rate = round(solved / n * 100, 1) if n > 0 else 0
            solved_epochs = [r.get("solved_epoch") for r in d
                             if (r.get("solved", False) or r.get("best_accuracy", 0) >= 1.0)
                             and r.get("solved_epoch") is not None]
            avg_epoch = round(np.mean(solved_epochs)) if solved_epochs else None
            results[fname] = {
                "N": n,
                "solved": solved,
                "rate": rate,
                "avg_epoch": avg_epoch,
            }

    report["conditions"] = results
    return report


# ============================================================================
# E8: Depth sensitivity
# ============================================================================
def compute_e8_depth_sensitivity():
    """Depth sensitivity: sin and tanh at depths 2, 4, 6."""
    report = {"experiment": "E8: Depth sensitivity"}

    ds_dir = os.path.join(RESULTS_DIR, "depth_sensitivity_n30")
    results = {}
    for fpath in sorted(glob.glob(os.path.join(ds_dir, "*.json"))):
        fname = os.path.basename(fpath).replace(".json", "")
        if fname == "depth_sensitivity_combined":
            continue
        d = load_json(fpath)
        summary = d.get("summary", {})
        results[fname] = {
            "N": summary.get("n", len(d.get("results", []))),
            "solved": summary.get("solved", 0),
            "solve_rate": summary.get("solve_rate", 0),
            "median_gen": summary.get("median_gen"),
            "activation": d.get("activation"),
            "depth": d.get("depth"),
        }

    report["conditions"] = results

    # Flag partial data
    for key, val in results.items():
        if val["N"] < 30:
            val["partial"] = True
            val["note"] = f"Only {val['N']}/30 seeds complete"

    return report


# ============================================================================
# E9: lts_low CI at N=60
# ============================================================================
def compute_e9_lts_low_extended():
    """lts_low extended to N=60 for tighter CI."""
    report = {"experiment": "E9: lts_low extended (N=60)"}

    # Original N=30
    lts_orig = os.path.join(RESULTS_DIR, "per_function_ablation_n30", "lts_low.json")
    lts_ext = os.path.join(RESULTS_DIR, "per_function_ablation_n30", "lts_low_extended.json")

    if os.path.exists(lts_orig):
        d = load_json(lts_orig)
        orig_solved = d["summary"]["solve_count"]
        orig_total = d["summary"]["total_runs"]
        report["original_n30"] = {
            "solved": orig_solved,
            "total": orig_total,
            "rate": round(orig_solved / orig_total * 100, 1),
        }

    if os.path.exists(lts_ext):
        d = load_json(lts_ext)
        ext_solved = d["summary"]["solve_count"]
        ext_total = d["summary"]["total_runs"]
        report["extended_n30"] = {
            "solved": ext_solved,
            "total": ext_total,
            "rate": round(ext_solved / ext_total * 100, 1),
        }

    # Combined N=60
    if os.path.exists(lts_orig) and os.path.exists(lts_ext):
        total_solved = orig_solved + ext_solved
        total_n = orig_total + ext_total
        rate = total_solved / total_n
        # Wilson CI
        from math import sqrt
        z = 1.96
        denom = 1 + z**2 / total_n
        center = (rate + z**2 / (2 * total_n)) / denom
        margin = z * sqrt(rate * (1 - rate) / total_n + z**2 / (4 * total_n**2)) / denom
        ci_low = max(0, center - margin)
        ci_high = min(1, center + margin)
        report["combined_n60"] = {
            "solved": total_solved,
            "total": total_n,
            "rate": round(rate * 100, 1),
            "ci_95": f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]",
        }
        report["paper_claims"] = {
            "rate": "5.0%",
            "ci": "[1.0%, 14.0%]",
        }

    return report


# ============================================================================
# Total experiment count
# ============================================================================
def compute_total_experiments():
    """Count total experimental runs across all result files."""
    report = {"experiment": "Total experiment count"}
    total = 0
    breakdown = {}

    # Per-function ablation: 18 functions × 30 seeds
    ablation_dir = os.path.join(RESULTS_DIR, "per_function_ablation_n30")
    ablation_count = 0
    for fpath in glob.glob(os.path.join(ablation_dir, "*.json")):
        fname = os.path.basename(fpath).replace(".json", "")
        if fname in ("combined_results", "lts_low_extended"):
            continue
        d = load_json(fpath)
        ablation_count += d["summary"]["total_runs"]
    # Add lts_low extended
    lts_ext = os.path.join(ablation_dir, "lts_low_extended.json")
    if os.path.exists(lts_ext):
        d = load_json(lts_ext)
        ablation_count += d["summary"]["total_runs"]
    breakdown["per_function_ablation"] = ablation_count
    total += ablation_count

    # XOR per-function: 18 × 30
    xor_dir = os.path.join(RESULTS_DIR, "xor_per_function_n30")
    xor_count = 0
    for fpath in glob.glob(os.path.join(xor_dir, "*.json")):
        fname = os.path.basename(fpath).replace(".json", "")
        if fname == "combined_results":
            continue
        d = load_json(fpath)
        xor_count += d["summary"]["total_runs"]
    breakdown["xor_per_function"] = xor_count
    total += xor_count

    # Parity scaling (sin): 7 sizes × 30
    sin_dir = os.path.join(RESULTS_DIR, "parity_scaling_n30")
    sin_count = 0
    for fpath in glob.glob(os.path.join(sin_dir, "parity_*.json")):
        d = load_json(fpath)
        sin_count += d["summary"]["total_runs"]
    breakdown["parity_scaling_sin"] = sin_count
    total += sin_count

    # Monotonic parity scaling (tanh): 7 sizes × 30
    tanh_dir = os.path.join(RESULTS_DIR, "monotonic_parity_scaling_n30")
    tanh_count = 0
    for fpath in glob.glob(os.path.join(tanh_dir, "parity_*.json")):
        d = load_json(fpath)
        tanh_count += d["summary"]["total_runs"]
    breakdown["parity_scaling_tanh"] = tanh_count
    total += tanh_count

    # Pop sensitivity: 10 conditions × 30
    pop_dir = os.path.join(RESULTS_DIR, "pop_sensitivity_n30")
    pop_count = 0
    for fpath in glob.glob(os.path.join(pop_dir, "*.json")):
        d = load_json(fpath)
        pop_count += d["summary"]["n"]
    breakdown["pop_sensitivity"] = pop_count
    total += pop_count

    # Two Spirals N=30: 9 configs × 30
    ts_combined = os.path.join(RESULTS_DIR, "two_spirals_n30", "combined_results.json")
    if os.path.exists(ts_combined):
        d = load_json(ts_combined)
        ts_count = 0
        for k, v in d.items():
            if k == "metadata":
                continue
            ts_count += len(v.get("results", []))
        breakdown["two_spirals_assignment"] = ts_count
        total += ts_count

    # Two Spirals per-function: 6 functions × 30
    ts_pf_dir = os.path.join(RESULTS_DIR, "two_spirals_per_function_n30")
    ts_pf_count = 0
    for fpath in glob.glob(os.path.join(ts_pf_dir, "*.json")):
        fname = os.path.basename(fpath).replace(".json", "")
        if fname == "combined_results":
            continue
        d = load_json(fpath)
        ts_pf_count += d["summary"]["total_runs"]
    breakdown["two_spirals_per_function"] = ts_pf_count
    total += ts_pf_count

    # Recurrence type: 6 presets × 30
    rt_dir = os.path.join(RESULTS_DIR, "recurrence_type_n30")
    rt_count = 0
    for fpath in glob.glob(os.path.join(rt_dir, "*.json")):
        fname = os.path.basename(fpath).replace(".json", "")
        if fname == "combined":
            continue
        d = load_json(fpath)
        rt_count += d["summary"]["total_runs"]
    breakdown["recurrence_type"] = rt_count
    total += rt_count

    # Recurrent benchmarks (strategy on Two Spirals): per-seed files
    rec_bench_dir = os.path.join(RESULTS_DIR, "recurrent_benchmarks", "strategy")
    if os.path.exists(rec_bench_dir):
        rec_count = len(glob.glob(os.path.join(rec_bench_dir, "*.json")))
        breakdown["recurrent_benchmarks_strategy"] = rec_count
        total += rec_count

    # Recurrent Parity-8: all per-seed files (partial for monotonic)
    rec_p8_dir = os.path.join(RESULTS_DIR, "recurrent_parity8_n30")
    rec_p8_count = len(glob.glob(os.path.join(rec_p8_dir, "*_seed*.json")))
    breakdown["recurrent_parity8"] = rec_p8_count
    total += rec_p8_count

    # Depth sensitivity
    ds_dir = os.path.join(RESULTS_DIR, "depth_sensitivity_n30")
    ds_count = 0
    for fpath in glob.glob(os.path.join(ds_dir, "*.json")):
        fname = os.path.basename(fpath).replace(".json", "")
        if fname == "depth_sensitivity_combined":
            continue
        d = load_json(fpath)
        ds_count += len(d.get("results", []))
    breakdown["depth_sensitivity"] = ds_count
    total += ds_count

    # Depth-6 monotonic
    d6_dir = os.path.join(RESULTS_DIR, "depth6_monotonic_n30")
    d6_count = 0
    for fpath in glob.glob(os.path.join(d6_dir, "*.json")):
        d = load_json(fpath)
        d6_count += d["summary"]["total_runs"]
    breakdown["depth6_monotonic"] = d6_count
    total += d6_count

    # Gradient baseline
    gb_dir = os.path.join(RESULTS_DIR, "gradient_baseline_n30")
    gb_count = 0
    for fpath in glob.glob(os.path.join(gb_dir, "parity*.json")):
        d = load_json(fpath)
        if isinstance(d, list):
            gb_count += len(d)
    breakdown["gradient_baseline"] = gb_count
    total += gb_count

    report["breakdown"] = breakdown
    report["total"] = total
    return report


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("PAPER STATISTICS AUDIT")
    print("Per-Node Activation Function Evolution in Indirect Encoding")
    print("=" * 70)

    all_reports = {}

    # Run all analyses
    analyses = [
        ("e1_per_function_ablation", compute_e1_per_function_ablation),
        ("e1b_extrema_correlation", compute_e1b_extrema_correlation),
        ("e1c_cross_task_correlation", compute_e1c_cross_task_correlation),
        ("e2_parity_scaling", compute_e2_parity_scaling),
        ("e3_pop_sensitivity", compute_e3_pop_sensitivity),
        ("e4_recurrent", compute_e4_recurrent),
        ("e5_assignment_methods", compute_e5_assignment_methods),
        ("e5b_recurrence_type", compute_e5b_recurrence_type),
        ("e6_palette_size", compute_e6_palette_size),
        ("e6b_gradient_baseline", compute_e6b_gradient_baseline),
        ("e8_depth_sensitivity", compute_e8_depth_sensitivity),
        ("e9_lts_low_extended", compute_e9_lts_low_extended),
        ("total_experiments", compute_total_experiments),
    ]

    for name, func in analyses:
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")
        try:
            report = func()
            all_reports[name] = report
            # Pretty print key results
            for k, v in report.items():
                if k == "experiment":
                    print(f"  {v}")
                elif isinstance(v, dict) and len(str(v)) > 200:
                    print(f"  {k}:")
                    for kk, vv in v.items():
                        print(f"    {kk}: {vv}")
                else:
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_reports[name] = {"error": str(e)}

    # Save full report
    output_path = os.path.join(RESULTS_DIR, "paper_statistics_report.json")
    with open(output_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Full report saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
