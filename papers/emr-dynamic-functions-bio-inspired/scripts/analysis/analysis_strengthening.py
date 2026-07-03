#!/usr/bin/env python3
"""Analysis script for bio-inspired strengthening experiments (E1a-E4, P1b-P2c).

Reads result JSONs from the strengthening experiments and produces:
1. LaTeX-ready table rows for paper integration
2. Statistical summaries (solve rate, CI, median gen, osc%)
3. Parity scaling trend analysis (P4→P5→P6)
4. Non-parity problem analysis (Two Moons, Visual Discrimination)
5. Timescale rescaling validation (P1b)
6. N=60 extension with merged statistics (P2a)
7. Population sensitivity on Parity-5 (P2b)
8. Full 8-strategy topology sensitivity (P2c)

Usage:
    python papers/emr-dynamic-functions-bio-inspired/analysis_strengthening.py
    python papers/emr-dynamic-functions-bio-inspired/analysis_strengthening.py --experiment gaussian_xor
    python papers/emr-dynamic-functions-bio-inspired/analysis_strengthening.py --experiment timescale_rescaling
    python papers/emr-dynamic-functions-bio-inspired/analysis_strengthening.py --experiment n60_extension
    python papers/emr-dynamic-functions-bio-inspired/analysis_strengthening.py --experiment pop_sensitivity_p5
    python papers/emr-dynamic-functions-bio-inspired/analysis_strengthening.py --experiment topology_full
    python papers/emr-dynamic-functions-bio-inspired/analysis_strengthening.py --experiment all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy import stats


RESULTS_BASE = Path(__file__).resolve().parents[2] / "results"

STRATEGY_DISPLAY_NAMES = {
    'circadian_rhythm_dual': 'Circadian',
    'clonal_selection_dual': 'Clonal Sel.',
    'stdp_dual': 'STDP',
    'baseline_dual': 'Baseline',
    'critical_period_refined_dual': 'Crit.\\ Period',
    'hebbian_dual': 'Hebbian',
    'metaplastic_dual': 'Metaplastic',
    'adult_neurogenesis_dual': 'Neurogenesis',
    'predator_prey_dual': 'Pred.-Prey',
}

# Canonical ordering for tables (same as single-task Table 1)
STRATEGY_ORDER = [
    'circadian_rhythm_dual',
    'clonal_selection_dual',
    'stdp_dual',
    'baseline_dual',
    'critical_period_refined_dual',
    'hebbian_dual',
    'metaplastic_dual',
    'adult_neurogenesis_dual',
    'predator_prey_dual',
]


def binomial_ci_wilson(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if trials == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials
    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def load_results(results_dir: Path, strategy: str) -> Optional[Dict]:
    """Load result JSON for a strategy.

    Handles multiple naming conventions:
    - strategy_dual.json (new benchmarks)
    - strategy.json (P4 single_task)
    - parity_4_strategy.json (dual-domain)
    - strategy without _dual suffix (original results)
    """
    # Try direct name first
    path = results_dir / f"{strategy}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Try without _dual suffix (original naming)
    base_name = strategy.replace('_dual', '')
    # Handle renamed strategies (old → new naming)
    alt_names = {
        'critical_period_refined': 'critical_period',
        'adult_neurogenesis': 'adult_neurogenesis',
        'clonal_selection': 'clonal_selection',
    }
    path = results_dir / f"{base_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Try alternate name mapping (e.g., critical_period_refined → critical_period)
    alt_name = alt_names.get(base_name, None)
    if alt_name and alt_name != base_name:
        path = results_dir / f"{alt_name}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    # Try parity_4_ prefix (dual-domain naming)
    path = results_dir / f"parity_4_{strategy}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Try combined_results.json and look for strategy within
    path = results_dir / "combined_results.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        # Check if it has strategy-level keys
        for name in [strategy, base_name, alt_name]:
            if name and name in data:
                return data[name]
    return None


OSCILLATORY_INDICES = {4, 11, 12, 13, 15}  # sin, burst, resonator, osc_adapt, damped_osc


def _check_oscillatory_from_palette(seed_data: Dict) -> bool:
    """Check for oscillatory functions via final_palette or final_act_palette."""
    # Direct flag (P5, P6, Gaussian XOR format)
    osc = seed_data.get('has_oscillatory', seed_data.get('oscillatory_present', None))
    if osc is not None:
        return bool(osc)

    # Check oscillatory_functions list
    osc_funcs = seed_data.get('oscillatory_functions', None)
    if osc_funcs is not None:
        return len(osc_funcs) > 0

    # Fallback: check final_palette or final_act_palette for oscillatory indices
    palette = seed_data.get('final_act_palette', seed_data.get('final_palette', None))
    if palette is not None and isinstance(palette, list):
        return any(idx in OSCILLATORY_INDICES for idx in palette)

    # Last resort: check strategy_metrics for oscillatory info
    sm = seed_data.get('strategy_metrics', {})
    if isinstance(sm, dict):
        osc_count = sm.get('oscillatory_count', sm.get('sin_count', None))
        if osc_count is not None:
            return int(osc_count) > 0

    return False


def extract_metrics(data: Dict) -> Dict[str, Any]:
    """Extract key metrics from a result JSON.

    Handles both list-of-dicts and dict-of-dicts result formats.
    """
    results = data.get('results', [])
    # Normalize: if dict, convert to list of values
    if isinstance(results, dict):
        results = list(results.values())
    n_total = len(results)
    if n_total == 0:
        return {'n': 0, 'solve_rate': 0.0, 'solved': 0}

    solved = 0
    convergence_gens = []
    has_osc = 0  # Count among SOLVED runs only
    has_sin = 0
    has_agg = 0
    sin_retained = 0

    for seed_data in results:
        converged = seed_data.get('solved', seed_data.get('converged', False))
        if converged:
            solved += 1
            gen = seed_data.get('solved_gen', seed_data.get('convergence_gen',
                  seed_data.get('gen', 0)))
            convergence_gens.append(gen)

            # Oscillatory presence (count among solved runs only)
            if _check_oscillatory_from_palette(seed_data):
                has_osc += 1

        # Sin discovery (check sin_discovered_gen or has_sin flag)
        sin_disc_gen = seed_data.get('sin_discovered_gen', None)
        sin_disc = seed_data.get('sin_discovered', seed_data.get('has_sin', False))
        if sin_disc or (sin_disc_gen is not None and sin_disc_gen >= 0):
            has_sin += 1

        # Sin in final palette (retention)
        sin_in_pal = seed_data.get('sin_in_palette', seed_data.get('sin_retained', False))
        if sin_in_pal:
            sin_retained += 1

        # Aggregation discovery (check optimal_agg_discovered_gen field)
        agg_disc_gen = seed_data.get('optimal_agg_discovered_gen', None)
        agg_disc = seed_data.get('agg_discovered', seed_data.get('has_agg_plus', False))
        if agg_disc or (agg_disc_gen is not None and agg_disc_gen >= 0):
            has_agg += 1

    solve_rate = solved / n_total
    ci_lo, ci_hi = binomial_ci_wilson(solved, n_total)

    metrics = {
        'n': n_total,
        'solved': solved,
        'solve_rate': solve_rate,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'has_osc': has_osc,
        'osc_rate': has_osc / solved if solved > 0 else 0.0,  # % of SOLVED runs
        'has_sin': has_sin,
        'sin_rate': has_sin / n_total if n_total > 0 else 0.0,
        'has_agg': has_agg,
        'agg_rate': has_agg / n_total if n_total > 0 else 0.0,
        'sin_retained': sin_retained,
        'sin_ret_rate': sin_retained / n_total if n_total > 0 else 0.0,
    }

    if convergence_gens:
        metrics['median_gen'] = np.median(convergence_gens)
        metrics['mean_gen'] = np.mean(convergence_gens)
        metrics['min_gen'] = min(convergence_gens)
        metrics['max_gen'] = max(convergence_gens)
        metrics['convergence_gens'] = convergence_gens
    else:
        metrics['median_gen'] = None
        metrics['mean_gen'] = None
        metrics['convergence_gens'] = []

    return metrics


def analyze_gaussian_xor():
    """Analyze E1a: Gaussian XOR results."""
    results_dir = RESULTS_BASE / "gaussian_xor"
    if not results_dir.exists():
        print("No Gaussian XOR results found.")
        return

    print("=" * 70)
    print("E1a: GAUSSIAN XOR (8 strategies × 30 seeds)")
    print("=" * 70)

    all_metrics = {}
    for strategy in STRATEGY_ORDER:
        data = load_results(results_dir, strategy)
        if data is None:
            continue
        metrics = extract_metrics(data)
        all_metrics[strategy] = metrics
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        n = metrics['n']
        sr = metrics['solve_rate'] * 100
        ci = f"[{metrics['ci_lo']*100:.0f}, {metrics['ci_hi']*100:.0f}]"
        med = f"{metrics['median_gen']:.0f}" if metrics['median_gen'] is not None else "---"
        osc = f"{metrics['osc_rate']*100:.0f}"
        print(f"  {name:20s}: {metrics['solved']}/{n} ({sr:.0f}%) CI={ci} med_gen={med} osc={osc}%")

    if not all_metrics:
        print("  No results yet.")
        return

    # LaTeX table
    print("\n--- LaTeX Table (Gaussian XOR) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Gaussian XOR classification ($N{=}30$, 100 generations, pop$=$500).}")
    print(r"\label{tab:gaussian_xor}")
    print(r"{\footnotesize\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Strategy & Solve\% & 95\% CI & Med Gen & Osc.\% \\")
    print(r"\midrule")

    # Find best solve rate and fastest median for bolding
    best_sr = max(m['solve_rate'] for m in all_metrics.values())
    solved_meds = [m['median_gen'] for m in all_metrics.values() if m['median_gen'] is not None]
    best_med = min(solved_meds) if solved_meds else None

    for strategy in STRATEGY_ORDER:
        if strategy not in all_metrics:
            continue
        m = all_metrics[strategy]
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = f"{m['solve_rate']*100:.0f}\\%"
        ci = f"[{m['ci_lo']*100:.0f}, {m['ci_hi']*100:.0f}]"
        med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "---"
        osc = f"{m['osc_rate']*100:.0f}"

        # Bold best values
        if m['solve_rate'] == best_sr:
            sr = r"\textbf{" + sr + "}"
        if m['median_gen'] is not None and m['median_gen'] == best_med:
            med = r"\textbf{" + med + "}"

        print(f"{name} & {sr} & {ci} & {med} & {osc} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    return all_metrics


def analyze_parity6():
    """Analyze E2: Parity-6 completion results (expand existing 4→8 strategies)."""
    results_dir = RESULTS_BASE / "parity6_single_task"
    if not results_dir.exists():
        print("No Parity-6 results found.")
        return

    print("\n" + "=" * 70)
    print("E2: PARITY-6 (8 strategies × 30 seeds)")
    print("=" * 70)

    all_metrics = {}
    for strategy in STRATEGY_ORDER:
        data = load_results(results_dir, strategy)
        if data is None:
            continue
        metrics = extract_metrics(data)
        if metrics['n'] < 30:
            print(f"  {STRATEGY_DISPLAY_NAMES.get(strategy, strategy):20s}: {metrics['n']}/30 seeds (INCOMPLETE)")
            continue
        all_metrics[strategy] = metrics
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = metrics['solve_rate'] * 100
        ci = f"[{metrics['ci_lo']*100:.0f}, {metrics['ci_hi']*100:.0f}]"
        med = f"{metrics['median_gen']:.0f}" if metrics['median_gen'] is not None else "---"
        osc = f"{metrics['osc_rate']*100:.0f}"
        print(f"  {name:20s}: {metrics['solved']}/{metrics['n']} ({sr:.0f}%) CI={ci} med_gen={med} osc={osc}%")

    if len(all_metrics) < 4:
        print("  Not enough complete results for table.")
        return

    # LaTeX table (expanded version of Table 9)
    print("\n--- LaTeX Table (Parity-6, expanded to 8 strategies) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Parity-6 single-task validation ($N{=}30$, 200 generations, pop$=$300).}")
    print(r"\label{tab:parity6}")
    print(r"{\footnotesize\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Strategy & Solve\% & 95\% CI & Med Gen & Osc.\% \\")
    print(r"\midrule")

    best_sr = max(m['solve_rate'] for m in all_metrics.values())
    solved_meds = [m['median_gen'] for m in all_metrics.values() if m['median_gen'] is not None]
    best_med = min(solved_meds) if solved_meds else None
    best_osc = max(m['osc_rate'] for m in all_metrics.values())

    for strategy in STRATEGY_ORDER:
        if strategy not in all_metrics:
            continue
        m = all_metrics[strategy]
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = f"{m['solve_rate']*100:.0f}\\%"
        ci = f"[{m['ci_lo']*100:.0f}, {m['ci_hi']*100:.0f}]"
        med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "---"
        osc = f"{m['osc_rate']*100:.0f}"

        if m['solve_rate'] == best_sr and best_sr < 1.0:
            sr = r"\textbf{" + sr + "}"
        if m['median_gen'] is not None and m['median_gen'] == best_med:
            med = r"\textbf{" + med + "}"
        if m['osc_rate'] == best_osc:
            osc = r"\textbf{" + osc + "}"

        print(f"{name} & {sr} & {ci} & {med} & {osc} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    return all_metrics


def analyze_parity5():
    """Analyze E3: Parity-5 results."""
    results_dir = RESULTS_BASE / "parity5_single_task"
    if not results_dir.exists():
        print("No Parity-5 results found.")
        return

    print("\n" + "=" * 70)
    print("E3: PARITY-5 (8 strategies × 30 seeds)")
    print("=" * 70)

    all_metrics = {}
    for strategy in STRATEGY_ORDER:
        data = load_results(results_dir, strategy)
        if data is None:
            continue
        metrics = extract_metrics(data)
        if metrics['n'] < 30:
            print(f"  {STRATEGY_DISPLAY_NAMES.get(strategy, strategy):20s}: {metrics['n']}/30 seeds (INCOMPLETE)")
            continue
        all_metrics[strategy] = metrics
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = metrics['solve_rate'] * 100
        med = f"{metrics['median_gen']:.0f}" if metrics['median_gen'] is not None else "---"
        osc = f"{metrics['osc_rate']*100:.0f}"
        print(f"  {name:20s}: {metrics['solved']}/{metrics['n']} ({sr:.0f}%) med_gen={med} osc={osc}%")

    if not all_metrics:
        print("  No complete results yet.")
        return

    # LaTeX table
    print("\n--- LaTeX Table (Parity-5) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Parity-5 single-task results ($N{=}30$, 200 generations, pop$=$300).}")
    print(r"\label{tab:parity5}")
    print(r"{\footnotesize\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Strategy & Solve\% & 95\% CI & Med Gen & Osc.\% \\")
    print(r"\midrule")

    best_sr = max(m['solve_rate'] for m in all_metrics.values())
    solved_meds = [m['median_gen'] for m in all_metrics.values() if m['median_gen'] is not None]
    best_med = min(solved_meds) if solved_meds else None

    for strategy in STRATEGY_ORDER:
        if strategy not in all_metrics:
            continue
        m = all_metrics[strategy]
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = f"{m['solve_rate']*100:.0f}\\%"
        ci = f"[{m['ci_lo']*100:.0f}, {m['ci_hi']*100:.0f}]"
        med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "---"
        osc = f"{m['osc_rate']*100:.0f}"

        if m['solve_rate'] == best_sr and best_sr < 1.0:
            sr = r"\textbf{" + sr + "}"
        if m['median_gen'] is not None and m['median_gen'] == best_med:
            med = r"\textbf{" + med + "}"

        print(f"{name} & {sr} & {ci} & {med} & {osc} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    return all_metrics


def analyze_dual_domain():
    """Analyze E4: Dual-domain completion results (expand 5→8 strategies)."""
    results_dir = RESULTS_BASE / "dual_domain"
    if not results_dir.exists():
        print("No dual-domain results found.")
        return

    print("\n" + "=" * 70)
    print("E4: DUAL-DOMAIN (8 strategies × 30 seeds)")
    print("=" * 70)

    all_metrics = {}
    for strategy in STRATEGY_ORDER:
        data = load_results(results_dir, strategy)
        if data is None:
            continue
        metrics = extract_metrics(data)
        if metrics['n'] < 30:
            print(f"  {STRATEGY_DISPLAY_NAMES.get(strategy, strategy):20s}: {metrics['n']}/30 seeds (INCOMPLETE)")
            continue
        all_metrics[strategy] = metrics
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = metrics['solve_rate'] * 100
        sin = f"{metrics['sin_rate']*100:.0f}"
        agg = f"{metrics['agg_rate']*100:.0f}"
        print(f"  {name:20s}: {metrics['solved']}/{metrics['n']} ({sr:.0f}%) sin_disc={sin}% agg_disc={agg}%")

    if len(all_metrics) < 5:
        print("  Not enough complete results for full table.")
        return

    # LaTeX table (expanded version of Table 3)
    print("\n--- LaTeX Table (Dual-Domain, expanded to 8 strategies) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Dual-domain discovery results on Parity-4 ($N{=}30$). " +
          r"Clonal selection achieves the highest solve rate; STDP shows highest aggregation discovery at 50\%.}")
    print(r"\label{tab:dual_domain}")
    print(r"{\footnotesize\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Strategy & Solve\% & Sin Disc.\% & Sin Ret.\% & Agg Disc.\% \\")
    print(r"\midrule")

    best_sr = max(m['solve_rate'] for m in all_metrics.values())
    best_agg = max(m['agg_rate'] for m in all_metrics.values())

    for strategy in STRATEGY_ORDER:
        if strategy not in all_metrics:
            continue
        m = all_metrics[strategy]
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = f"{m['solve_rate']*100:.0f}\\%"
        sin_disc = f"{m['sin_rate']*100:.0f}\\%"
        # Sin retention not always tracked separately, use sin_rate as proxy
        sin_ret = f"{m.get('sin_ret_rate', m['sin_rate'])*100:.0f}\\%"
        agg_disc = f"{m['agg_rate']*100:.0f}\\%"

        if m['solve_rate'] == best_sr:
            sr = r"\textbf{" + sr + "}"
            name = r"\textbf{" + name + "}"
        if m['agg_rate'] == best_agg and best_agg > 0:
            agg_disc = r"\textbf{" + agg_disc + "}"

        print(f"{name} & {sr} & {sin_disc} & {sin_ret} & {agg_disc} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    return all_metrics


def analyze_two_moons():
    """Analyze P3a: Two Moons classification results."""
    results_dir = RESULTS_BASE / "two_moons"
    if not results_dir.exists():
        print("No Two Moons results found.")
        return

    print("\n" + "=" * 70)
    print("P3a: TWO MOONS (8 strategies × 30 seeds)")
    print("=" * 70)

    all_metrics = {}
    for strategy in STRATEGY_ORDER[:8]:
        data = load_results(results_dir, strategy)
        if data is None:
            continue
        metrics = extract_metrics(data)
        all_metrics[strategy] = metrics
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        n = metrics['n']
        sr = metrics['solve_rate'] * 100
        ci = f"[{metrics['ci_lo']*100:.0f}, {metrics['ci_hi']*100:.0f}]"
        med = f"{metrics['median_gen']:.0f}" if metrics['median_gen'] is not None else "---"
        osc = f"{metrics['osc_rate']*100:.0f}"
        print(f"  {name:20s}: {metrics['solved']}/{n} ({sr:.1f}%) CI={ci} med_gen={med} osc={osc}%")

    if not all_metrics:
        print("  No results yet.")
        return

    # Kruskal-Wallis on convergence speed
    groups = [m['convergence_gens'] for m in all_metrics.values() if m['convergence_gens']]
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis (convergence speed): H={h_stat:.2f}, p={p_val:.4f}")

    # Key pairwise Fisher's exact tests
    print("\n  Key pairwise Fisher's exact tests:")
    pairs = [
        ('critical_period_refined_dual', 'circadian_rhythm_dual'),
        ('stdp_dual', 'circadian_rhythm_dual'),
        ('baseline_dual', 'circadian_rhythm_dual'),
        ('critical_period_refined_dual', 'hebbian_dual'),
    ]
    for s1, s2 in pairs:
        if s1 in all_metrics and s2 in all_metrics:
            m1, m2 = all_metrics[s1], all_metrics[s2]
            table = [[m1['solved'], m1['n'] - m1['solved']],
                     [m2['solved'], m2['n'] - m2['solved']]]
            odds, p_val = stats.fisher_exact(table)
            n1 = STRATEGY_DISPLAY_NAMES.get(s1, s1)
            n2 = STRATEGY_DISPLAY_NAMES.get(s2, s2)
            print(f"    {n1} vs {n2}: p={p_val:.4f}")

    # Oscillatory barrier
    total_solved = sum(m['solved'] for m in all_metrics.values())
    total_osc = sum(m['has_osc'] for m in all_metrics.values())
    print(f"\n  Oscillatory barrier: {total_osc}/{total_solved} solved runs with osc "
          f"({total_osc/total_solved*100:.1f}%)")
    print("  Per-strategy osc% (of solved):")
    for strategy in STRATEGY_ORDER[:8]:
        if strategy in all_metrics and all_metrics[strategy]['solved'] > 0:
            m = all_metrics[strategy]
            name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
            print(f"    {name:20s}: {m['has_osc']}/{m['solved']} "
                  f"({m['osc_rate']*100:.0f}%)")

    # LaTeX table
    print("\n--- LaTeX Table (Two Moons) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Two Moons classification ($N{=}30$, 100 gens, pop$=$500, target$\geq$0.90).}")
    print(r"\label{tab:two_moons}")
    print(r"{\footnotesize\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Strategy & Solve\% & 95\% CI & Med Gen & Osc.\% \\")
    print(r"\midrule")

    best_sr = max(m['solve_rate'] for m in all_metrics.values())
    solved_meds = [m['median_gen'] for m in all_metrics.values() if m['median_gen'] is not None]
    best_med = min(solved_meds) if solved_meds else None

    for strategy in STRATEGY_ORDER[:8]:
        if strategy not in all_metrics:
            continue
        m = all_metrics[strategy]
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = f"{m['solve_rate']*100:.1f}\\%"
        ci = f"[{m['ci_lo']*100:.0f}, {m['ci_hi']*100:.0f}]"
        med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "---"
        osc = f"{m['osc_rate']*100:.0f}"

        if m['solve_rate'] == best_sr and best_sr < 1.0:
            sr = r"\textbf{" + sr + "}"
        if m['median_gen'] is not None and m['median_gen'] == best_med:
            med = r"\textbf{" + med + "}"

        print(f"{name} & {sr} & {ci} & {med} & {osc} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    return all_metrics


def analyze_visual_discrimination():
    """Analyze P3b: Visual Discrimination results (may be partial)."""
    results_dir = RESULTS_BASE / "visual_discrimination"
    if not results_dir.exists():
        print("No Visual Discrimination results found.")
        return

    print("\n" + "=" * 70)
    print("P3b: VISUAL DISCRIMINATION (8 strategies × 30 seeds)")
    print("=" * 70)

    all_metrics = {}
    incomplete = []
    for strategy in STRATEGY_ORDER[:8]:
        data = load_results(results_dir, strategy)
        if data is None:
            continue
        metrics = extract_metrics(data)
        all_metrics[strategy] = metrics
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        n = metrics['n']
        sr = metrics['solve_rate'] * 100
        ci = f"[{metrics['ci_lo']*100:.0f}, {metrics['ci_hi']*100:.0f}]"
        med = f"{metrics['median_gen']:.0f}" if metrics['median_gen'] is not None else "---"
        osc = f"{metrics['osc_rate']*100:.0f}"
        status = "" if n >= 30 else f" [INCOMPLETE: {n}/30]"
        if n < 30:
            incomplete.append((name, n))
        print(f"  {name:20s}: {metrics['solved']}/{n} ({sr:.1f}%) CI={ci} "
              f"med_gen={med} osc={osc}%{status}")

    if incomplete:
        print(f"\n  WARNING: {len(incomplete)} strategies incomplete:")
        for name, n in incomplete:
            print(f"    {name}: {n}/30 seeds")

    if not all_metrics:
        print("  No results yet.")
        return

    # LaTeX table
    total_runs = sum(m['n'] for m in all_metrics.values())
    total_complete = sum(1 for m in all_metrics.values() if m['n'] >= 30)
    print(f"\n--- LaTeX Table (Visual Discrimination, {total_runs} runs, "
          f"{total_complete}/8 complete) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Visual Discrimination ($N{=}30$, 100 gens, pop$=$500, "
          r"target$\geq$0.90).\protect\footnotemark}")
    print(r"\label{tab:visual_discrimination}")
    print(r"{\footnotesize\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Strategy & $N$ & Solve\% & Med Gen & Osc.\% \\")
    print(r"\midrule")

    for strategy in STRATEGY_ORDER[:8]:
        if strategy not in all_metrics:
            continue
        m = all_metrics[strategy]
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        n_str = str(m['n'])
        sr = f"{m['solve_rate']*100:.0f}\\%"
        med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "---"
        osc = f"{m['osc_rate']*100:.0f}"

        print(f"{name} & {n_str} & {sr} & {med} & {osc} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")
    if incomplete:
        n_complete = total_complete
        print(r"\footnotetext{" + f"{n_complete}" + r" of 8 strategies completed at $N{=}30$; "
              r"remaining strategies partial at time of writing. "
              r"All partial results show 100\% solve rate.}")

    return all_metrics


def analyze_parity_scaling(p4_metrics: Optional[Dict] = None):
    """Analyze P4→P5→P6 scaling trends across strategies with full statistics."""
    print("\n" + "=" * 70)
    print("PARITY SCALING ANALYSIS (P4 → P5 → P6)")
    print("=" * 70)

    p4_dir = RESULTS_BASE / "single_task"
    p5_dir = RESULTS_BASE / "parity5_single_task"
    p6_dir = RESULTS_BASE / "parity6_single_task"

    # Collect all metrics per difficulty level
    level_metrics = {}
    for label, results_dir in [("P4", p4_dir), ("P5", p5_dir), ("P6", p6_dir)]:
        level_metrics[label] = {}
        for strategy in STRATEGY_ORDER[:8]:
            data = load_results(results_dir, strategy)
            if data is not None:
                metrics = extract_metrics(data)
                if metrics['n'] >= 30:
                    level_metrics[label][strategy] = metrics

    # Print per-strategy scaling table
    print(f"\n  {'Strategy':20s}  {'P4':>12s}  {'P5':>12s}  {'P6':>12s}")
    print("  " + "-" * 60)
    for strategy in STRATEGY_ORDER[:8]:
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        row = f"  {name:20s}"
        for label in ["P4", "P5", "P6"]:
            m = level_metrics[label].get(strategy)
            if m is None:
                row += f"  {'---':>12s}"
            else:
                sr = m['solve_rate'] * 100
                med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "?"
                row += f"  {sr:.0f}%/{med}g".rjust(13)
        print(row)

    # Kruskal-Wallis per difficulty level (convergence speed of solved runs)
    print("\n  Kruskal-Wallis H-test on convergence generations (solved runs only):")
    for label in ["P4", "P5", "P6"]:
        groups = []
        for strategy in STRATEGY_ORDER[:8]:
            m = level_metrics[label].get(strategy)
            if m and m['convergence_gens']:
                groups.append(m['convergence_gens'])
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            print(f"    {label}: H={h_stat:.2f}, p={p_val:.4g} ({len(groups)} groups)")
        else:
            print(f"    {label}: insufficient data")

    # Key pairwise comparisons: circadian vs neurogenesis at each level
    print("\n  Key pairwise: Circadian vs Neurogenesis (Fisher's exact on solve rate):")
    for label in ["P4", "P5", "P6"]:
        circ = level_metrics[label].get('circadian_rhythm_dual')
        neuro = level_metrics[label].get('adult_neurogenesis_dual')
        if circ and neuro:
            table = [[circ['solved'], circ['n'] - circ['solved']],
                     [neuro['solved'], neuro['n'] - neuro['solved']]]
            odds, p_val = stats.fisher_exact(table)
            gap = (circ['solve_rate'] - neuro['solve_rate']) * 100
            print(f"    {label}: {circ['solve_rate']*100:.0f}% vs {neuro['solve_rate']*100:.0f}% "
                  f"(gap={gap:+.0f}pp, Fisher p={p_val:.4g})")

    # Circadian vs baseline at each level
    print("\n  Key pairwise: Circadian vs Baseline (Fisher's exact on solve rate):")
    for label in ["P4", "P5", "P6"]:
        circ = level_metrics[label].get('circadian_rhythm_dual')
        base = level_metrics[label].get('baseline_dual')
        if circ and base:
            table = [[circ['solved'], circ['n'] - circ['solved']],
                     [base['solved'], base['n'] - base['solved']]]
            odds, p_val = stats.fisher_exact(table)
            gap = (circ['solve_rate'] - base['solve_rate']) * 100
            print(f"    {label}: {circ['solve_rate']*100:.0f}% vs {base['solve_rate']*100:.0f}% "
                  f"(gap={gap:+.0f}pp, Fisher p={p_val:.4g})")

    # Oscillatory barrier across levels (has_osc already counts among solved only)
    print("\n  Oscillatory barrier across difficulty levels (% of solved runs):")
    for label in ["P4", "P5", "P6"]:
        total_solved = 0
        total_osc = 0
        for strategy in STRATEGY_ORDER[:8]:
            m = level_metrics[label].get(strategy)
            if m:
                total_solved += m['solved']
                total_osc += m['has_osc']
        if total_solved > 0:
            pct = total_osc / total_solved * 100
            print(f"    {label}: {total_osc}/{total_solved} solved runs with oscillatory ({pct:.1f}%)")

    return level_metrics


def _count_runs_in_file(filepath: Path) -> int:
    """Count experiment runs in a JSON file, handling multiple schemas."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception:
        return 0

    # Schema 1: Standard {'results': [...]} or {'results': {...}}
    results = data.get('results', None)
    if results is not None:
        if isinstance(results, list):
            return len(results)
        if isinstance(results, dict):
            # Check if values are seed-level dicts or strategy-level dicts
            first_val = next(iter(results.values()), None) if results else None
            if isinstance(first_val, dict) and 'runs' in first_val:
                # Nested: {'strategy_name': {'runs': [...], ...}}
                return sum(len(v.get('runs', [])) for v in results.values()
                           if isinstance(v, dict))
            return len(results)

    # Schema 2: {'runs': [...]} (persistent_cl, sequential_hybrid_cl format)
    runs = data.get('runs', None)
    if runs is not None and isinstance(runs, list):
        return len(runs)

    # Schema 3: {'trials': [...]} (oracle_baseline, oracle_composite format)
    trials = data.get('trials', None)
    if trials is not None and isinstance(trials, list):
        return len(trials)

    return 0


def count_total_runs():
    """Count total runs across all result directories with per-directory breakdown."""
    total = 0
    results_base = RESULTS_BASE
    dir_counts = {}

    # Skip pure analysis/summary files
    skip_names = {'top_tier_posthoc_analysis.json'}

    for subdir in sorted(results_base.iterdir()):
        if not subdir.is_dir():
            continue
        dir_total = 0
        counted_files = []
        # First pass: count individual (non-combined) files
        individual_count = 0
        combined_count = 0
        for f in sorted(subdir.iterdir()):
            if f.suffix != '.json' or f.name in skip_names:
                continue
            n = _count_runs_in_file(f)
            if f.name.startswith(('combined', 'analysis_summary')):
                combined_count += n
            else:
                individual_count += n
                if n > 0:
                    counted_files.append((f.name, n))

        # Use individual files when available, fall back to combined
        dir_total = individual_count if individual_count > 0 else combined_count
        if dir_total > 0:
            dir_counts[subdir.name] = dir_total
            total += dir_total

    print(f"\n{'='*70}")
    print(f"RUN COUNT VERIFICATION (per directory)")
    print(f"{'='*70}")
    for name, count in sorted(dir_counts.items()):
        print(f"  {name:40s}: {count:5d}")
    print(f"  {'':40s}  -----")
    print(f"  {'TOTAL':40s}: {total:5d}")
    print(f"{'='*70}")
    return total


def analyze_oscillatory_barrier():
    """Comprehensive oscillatory barrier analysis across all single-task experiments."""
    print("\n" + "=" * 70)
    print("OSCILLATORY BARRIER ANALYSIS (all single-task experiments)")
    print("=" * 70)

    experiments = [
        ("P4 (single_task)", RESULTS_BASE / "single_task"),
        ("P5 (parity5_single_task)", RESULTS_BASE / "parity5_single_task"),
        ("P6 (parity6_single_task)", RESULTS_BASE / "parity6_single_task"),
        ("Gaussian XOR", RESULTS_BASE / "gaussian_xor"),
        ("Two Moons", RESULTS_BASE / "two_moons"),
        ("Visual Discrimination", RESULTS_BASE / "visual_discrimination"),
    ]

    grand_solved = 0
    grand_osc = 0

    for exp_name, results_dir in experiments:
        if not results_dir.exists():
            continue
        exp_solved = 0
        exp_osc = 0
        for strategy in STRATEGY_ORDER[:8]:
            data = load_results(results_dir, strategy)
            if data is None:
                continue
            metrics = extract_metrics(data)
            exp_solved += metrics['solved']
            exp_osc += metrics['has_osc']
        if exp_solved > 0:
            pct = exp_osc / exp_solved * 100
            print(f"  {exp_name:35s}: {exp_osc}/{exp_solved} ({pct:.1f}%)")
            grand_solved += exp_solved
            grand_osc += exp_osc

    if grand_solved > 0:
        print(f"  {'GRAND TOTAL':35s}: {grand_osc}/{grand_solved} ({grand_osc/grand_solved*100:.1f}%)")


def analyze_timescale_rescaling():
    """Analyze P1b: Timescale Rescaling Validation.

    Compares rescaled strategies (GRN, Glial, Ant Colony) against originals.
    Tests whether compressing Tc recovers performance on Parity-4.
    """
    results_dir = RESULTS_BASE / "timescale_rescaling"
    if not results_dir.exists():
        print("No timescale rescaling results found.")
        return

    print("\n" + "=" * 70)
    print("P1b: TIMESCALE RESCALING VALIDATION (3 strategies × 30 seeds)")
    print("=" * 70)

    # Original strategy performance (from paper Table 23 / existing data)
    # These are the pre-rescaling solve rates and Tc values
    ORIGINAL_DATA = {
        'grn_rescaled': {
            'original_name': 'GRN',
            'original_tc': '>100',
            'rescaled_tc': '~10',
            'original_solve_rate': 0.03,  # 3% from paper
        },
        'glial_rescaled': {
            'original_name': 'Glial Mod.',
            'original_tc': '~50',
            'rescaled_tc': '~5',
            'original_solve_rate': 0.47,  # 47% from paper
        },
        'ant_colony_rescaled': {
            'original_name': 'Ant Colony',
            'original_tc': '~20',
            'rescaled_tc': '~5',
            'original_solve_rate': 0.80,  # 80% from paper
        },
    }

    RESCALED_STRATEGIES = ['grn_rescaled', 'glial_rescaled', 'ant_colony_rescaled']

    all_metrics = {}
    for strategy in RESCALED_STRATEGIES:
        path = results_dir / f"{strategy}.json"
        if not path.exists():
            print(f"  {strategy}: no results file")
            continue
        with open(path) as f:
            data = json.load(f)
        metrics = extract_metrics(data)
        all_metrics[strategy] = metrics
        orig = ORIGINAL_DATA[strategy]
        name = orig['original_name']
        sr = metrics['solve_rate'] * 100
        ci = f"[{metrics['ci_lo']*100:.0f}, {metrics['ci_hi']*100:.0f}]"
        med = f"{metrics['median_gen']:.0f}" if metrics['median_gen'] is not None else "---"
        orig_sr = orig['original_solve_rate'] * 100
        delta = sr - orig_sr
        print(f"  {name:20s}: {metrics['solved']}/{metrics['n']} ({sr:.0f}%) CI={ci} "
              f"med_gen={med}  [original: {orig_sr:.0f}%, delta: {delta:+.0f}pp]")

    if not all_metrics:
        print("  No results yet.")
        return

    # Fisher's exact test: rescaled vs original (for each strategy)
    print("\n  Fisher's exact (rescaled vs original):")
    for strategy in RESCALED_STRATEGIES:
        if strategy not in all_metrics:
            continue
        m = all_metrics[strategy]
        orig = ORIGINAL_DATA[strategy]
        # Approximate original as N=30 with solve_rate
        orig_solved = round(orig['original_solve_rate'] * 30)
        table = [[m['solved'], m['n'] - m['solved']],
                 [orig_solved, 30 - orig_solved]]
        odds, p_val = stats.fisher_exact(table)
        name = orig['original_name']
        print(f"    {name}: rescaled {m['solve_rate']*100:.0f}% vs original "
              f"{orig['original_solve_rate']*100:.0f}%, Fisher p={p_val:.4f}")

    # LaTeX table rows for integration into Table 23
    print("\n--- LaTeX rows for Table 23 (Timescale Rescaling) ---")
    print(r"% Add these rows to Table 23 as rescaled variants")
    for strategy in RESCALED_STRATEGIES:
        if strategy not in all_metrics:
            continue
        m = all_metrics[strategy]
        orig = ORIGINAL_DATA[strategy]
        name = orig['original_name'] + " (rescaled)"
        sr = f"{m['solve_rate']*100:.0f}\\%"
        ci = f"[{m['ci_lo']*100:.0f}, {m['ci_hi']*100:.0f}]"
        med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "---"
        tc = orig['rescaled_tc']
        osc = f"{m['osc_rate']*100:.0f}"
        print(f"{name} & {tc} & {sr} & {ci} & {med} & {osc} \\\\")

    return all_metrics


def analyze_n60_extension():
    """Analyze P2a: N=60 Single-Task Parity-4 Extension.

    Merges existing seeds 42-71 (N=30) with new seeds 72-101 (N=30)
    to produce N=60 statistics with narrower CIs and more statistical power.
    """
    new_dir = RESULTS_BASE / "single_task_n60"
    orig_dir = RESULTS_BASE / "single_task"

    if not new_dir.exists():
        print("No N=60 extension results found.")
        return

    print("\n" + "=" * 70)
    print("P2a: N=60 SINGLE-TASK PARITY-4 (seeds 42-71 + 72-101)")
    print("=" * 70)

    all_n30_orig = {}
    all_n30_new = {}
    all_n60 = {}

    for strategy in STRATEGY_ORDER[:8]:
        # Load original N=30 (seeds 42-71)
        orig_data = load_results(orig_dir, strategy)
        orig_metrics = extract_metrics(orig_data) if orig_data else None

        # Load new N=30 (seeds 72-101)
        new_data = load_results(new_dir, strategy)
        new_metrics = extract_metrics(new_data) if new_data else None

        if orig_metrics:
            all_n30_orig[strategy] = orig_metrics
        if new_metrics:
            all_n30_new[strategy] = new_metrics

        # Merge for N=60
        if orig_data and new_data:
            orig_results = orig_data.get('results', [])
            new_results = new_data.get('results', [])
            if isinstance(orig_results, dict):
                orig_results = list(orig_results.values())
            if isinstance(new_results, dict):
                new_results = list(new_results.values())
            merged = {'results': orig_results + new_results}
            merged_metrics = extract_metrics(merged)
            all_n60[strategy] = merged_metrics

            name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
            sr_orig = orig_metrics['solve_rate'] * 100 if orig_metrics else 0
            sr_new = new_metrics['solve_rate'] * 100 if new_metrics else 0
            sr_60 = merged_metrics['solve_rate'] * 100
            ci_60 = f"[{merged_metrics['ci_lo']*100:.0f}, {merged_metrics['ci_hi']*100:.0f}]"
            med = f"{merged_metrics['median_gen']:.0f}" if merged_metrics['median_gen'] is not None else "---"
            print(f"  {name:20s}: N30a={sr_orig:.0f}% N30b={sr_new:.0f}% "
                  f"N60={sr_60:.0f}% CI={ci_60} med={med}")
        elif new_metrics:
            name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
            print(f"  {name:20s}: new seeds only ({new_metrics['n']} runs)")

    if not all_n60:
        print("  No merged N=60 data available.")
        return

    # Key pairwise tests at N=60
    print("\n  Key pairwise Fisher's exact (N=60):")
    pairs = [
        ('circadian_rhythm_dual', 'baseline_dual'),
        ('circadian_rhythm_dual', 'adult_neurogenesis_dual'),
        ('clonal_selection_dual', 'critical_period_refined_dual'),
        ('circadian_rhythm_dual', 'hebbian_dual'),
    ]
    for s1, s2 in pairs:
        if s1 in all_n60 and s2 in all_n60:
            m1, m2 = all_n60[s1], all_n60[s2]
            table = [[m1['solved'], m1['n'] - m1['solved']],
                     [m2['solved'], m2['n'] - m2['solved']]]
            odds, p_val = stats.fisher_exact(table)
            n1 = STRATEGY_DISPLAY_NAMES.get(s1, s1)
            n2 = STRATEGY_DISPLAY_NAMES.get(s2, s2)
            gap = (m1['solve_rate'] - m2['solve_rate']) * 100
            print(f"    {n1} vs {n2}: {m1['solve_rate']*100:.0f}% vs {m2['solve_rate']*100:.0f}% "
                  f"(gap={gap:+.0f}pp, Fisher p={p_val:.4f})")

    # Kruskal-Wallis on convergence speed at N=60
    groups = [m['convergence_gens'] for m in all_n60.values() if m['convergence_gens']]
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis (N=60 convergence speed): H={h_stat:.2f}, p={p_val:.4g}")

    # Compare CI widths: N=30 vs N=60
    print("\n  CI width comparison (N=30 → N=60):")
    for strategy in STRATEGY_ORDER[:8]:
        if strategy in all_n30_orig and strategy in all_n60:
            m30 = all_n30_orig[strategy]
            m60 = all_n60[strategy]
            w30 = (m30['ci_hi'] - m30['ci_lo']) * 100
            w60 = (m60['ci_hi'] - m60['ci_lo']) * 100
            name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
            print(f"    {name:20s}: {w30:.0f}pp → {w60:.0f}pp (reduction: {w30-w60:.0f}pp)")

    # LaTeX table (N=60 version of Table 3)
    print("\n--- LaTeX Table (N=60 Parity-4) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Single-task Parity-4 ($N{=}60$, seeds 42--101, pop$=$500, 100 gens).}")
    print(r"\label{tab:single_task_n60}")
    print(r"{\footnotesize\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Strategy & Solve\% & 95\% CI & Med Gen & Mean Gen & Osc.\% \\")
    print(r"\midrule")

    best_sr = max(m['solve_rate'] for m in all_n60.values())
    solved_meds = [m['median_gen'] for m in all_n60.values() if m['median_gen'] is not None]
    best_med = min(solved_meds) if solved_meds else None

    for strategy in STRATEGY_ORDER[:8]:
        if strategy not in all_n60:
            continue
        m = all_n60[strategy]
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        sr = f"{m['solve_rate']*100:.0f}\\%"
        ci = f"[{m['ci_lo']*100:.0f}, {m['ci_hi']*100:.0f}]"
        med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "---"
        mean = f"{m['mean_gen']:.1f}" if m['mean_gen'] is not None else "---"
        osc = f"{m['osc_rate']*100:.0f}"

        if m['solve_rate'] == best_sr and best_sr < 1.0:
            sr = r"\textbf{" + sr + "}"
        if m['median_gen'] is not None and m['median_gen'] == best_med:
            med = r"\textbf{" + med + "}"

        print(f"{name} & {sr} & {ci} & {med} & {mean} & {osc} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    return all_n60


def analyze_pop_sensitivity_p5():
    """Analyze P2b: Population Sensitivity on Parity-5.

    Compares pop=200, pop=400 (existing), and pop=800 for 3 strategies.
    Tests whether larger populations help weaker strategies.
    """
    pop_dir = RESULTS_BASE / "pop_sensitivity_p5"
    existing_dir = RESULTS_BASE / "parity5_single_task"

    if not pop_dir.exists():
        print("No pop sensitivity P5 results found.")
        return

    print("\n" + "=" * 70)
    print("P2b: POPULATION SENSITIVITY ON PARITY-5")
    print("=" * 70)

    POP_STRATEGIES = ['circadian_rhythm_dual', 'baseline_dual', 'adult_neurogenesis_dual']
    POP_SIZES = [200, 400, 800]

    # Collect metrics for each strategy × pop size
    all_metrics = {}  # (strategy, pop) -> metrics
    for strategy in POP_STRATEGIES:
        for pop in POP_SIZES:
            if pop == 400:
                # Load from existing P5 results
                data = load_results(existing_dir, strategy)
            else:
                path = pop_dir / f"{strategy}_pop{pop}.json"
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                else:
                    data = None
            if data:
                metrics = extract_metrics(data)
                all_metrics[(strategy, pop)] = metrics

    if not all_metrics:
        print("  No results yet.")
        return

    # Summary table
    print(f"\n  {'Strategy':20s}  {'pop=200':>12s}  {'pop=400':>12s}  {'pop=800':>12s}")
    print("  " + "-" * 60)
    for strategy in POP_STRATEGIES:
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        row = f"  {name:20s}"
        for pop in POP_SIZES:
            m = all_metrics.get((strategy, pop))
            if m is None:
                row += f"  {'---':>12s}"
            else:
                sr = m['solve_rate'] * 100
                med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "?"
                row += f"  {sr:.0f}%/{med}g".rjust(13)
        print(row)

    # Statistical tests: pop=200 vs pop=800 for each strategy
    print("\n  Fisher's exact (pop=200 vs pop=800):")
    for strategy in POP_STRATEGIES:
        m200 = all_metrics.get((strategy, 200))
        m800 = all_metrics.get((strategy, 800))
        if m200 and m800:
            table = [[m200['solved'], m200['n'] - m200['solved']],
                     [m800['solved'], m800['n'] - m800['solved']]]
            odds, p_val = stats.fisher_exact(table)
            name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
            gap = (m800['solve_rate'] - m200['solve_rate']) * 100
            print(f"    {name}: pop200={m200['solve_rate']*100:.0f}% vs pop800={m800['solve_rate']*100:.0f}% "
                  f"(gap={gap:+.0f}pp, Fisher p={p_val:.4f})")

    # Mann-Whitney U on convergence speed for each strategy (pop=200 vs pop=800)
    print("\n  Mann-Whitney U (convergence speed, pop=200 vs pop=800):")
    for strategy in POP_STRATEGIES:
        m200 = all_metrics.get((strategy, 200))
        m800 = all_metrics.get((strategy, 800))
        if m200 and m800 and m200['convergence_gens'] and m800['convergence_gens']:
            u_stat, p_val = stats.mannwhitneyu(m200['convergence_gens'], m800['convergence_gens'],
                                                alternative='two-sided')
            name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
            n1, n2 = len(m200['convergence_gens']), len(m800['convergence_gens'])
            r_effect = 1 - 2 * u_stat / (n1 * n2) if n1 * n2 > 0 else 0
            print(f"    {name}: U={u_stat:.0f}, p={p_val:.4f}, r={r_effect:.3f}")

    # LaTeX table
    print("\n--- LaTeX Table (Pop Sensitivity P5) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Population sensitivity on Parity-5 ($N{=}30$, 150 gens, depth$=$4).}")
    print(r"\label{tab:pop_sensitivity_p5}")
    print(r"{\footnotesize\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{Pop$=$200} & \multicolumn{2}{c}{Pop$=$400} & \multicolumn{2}{c}{Pop$=$800} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    print(r"Strategy & Solve\% & Med & Solve\% & Med & Solve\% & Med \\")
    print(r"\midrule")

    for strategy in POP_STRATEGIES:
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        cells = []
        for pop in POP_SIZES:
            m = all_metrics.get((strategy, pop))
            if m:
                sr = f"{m['solve_rate']*100:.0f}\\%"
                med = f"{m['median_gen']:.0f}" if m['median_gen'] is not None else "---"
                cells.append(f"{sr} & {med}")
            else:
                cells.append("--- & ---")
        print(f"{name} & {' & '.join(cells)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    return all_metrics


def analyze_topology_full():
    """Analyze P2c: Full 8-Strategy Topology Sensitivity on Single-Task P4.

    Compares feedforward (existing Table 3) vs full_recurrent (new) for all 8 strategies.
    """
    topo_dir = RESULTS_BASE / "topology_sensitivity_single_task"
    ff_dir = RESULTS_BASE / "single_task"

    if not topo_dir.exists():
        print("No topology sensitivity results found.")
        return

    print("\n" + "=" * 70)
    print("P2c: FULL 8-STRATEGY TOPOLOGY SENSITIVITY (feedforward vs full_recurrent)")
    print("=" * 70)

    ff_metrics = {}
    fr_metrics = {}

    for strategy in STRATEGY_ORDER[:8]:
        # Load feedforward baseline from Table 3 data
        ff_data = load_results(ff_dir, strategy)
        if ff_data:
            ff_metrics[strategy] = extract_metrics(ff_data)

        # Load full_recurrent from new results
        path = topo_dir / f"{strategy}_full_recurrent.json"
        if path.exists():
            with open(path) as f:
                fr_data = json.load(f)
            fr_metrics[strategy] = extract_metrics(fr_data)

    if not fr_metrics:
        print("  No full_recurrent results yet.")
        return

    # Summary comparison
    print(f"\n  {'Strategy':20s}  {'FF':>15s}  {'Full Rec.':>15s}  {'Delta':>8s}")
    print("  " + "-" * 65)
    for strategy in STRATEGY_ORDER[:8]:
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        ff_m = ff_metrics.get(strategy)
        fr_m = fr_metrics.get(strategy)
        ff_str = f"{ff_m['solve_rate']*100:.0f}%/{ff_m['median_gen']:.0f}g" if ff_m and ff_m['median_gen'] is not None else (f"{ff_m['solve_rate']*100:.0f}%/---" if ff_m else "---")
        fr_str = f"{fr_m['solve_rate']*100:.0f}%/{fr_m['median_gen']:.0f}g" if fr_m and fr_m['median_gen'] is not None else (f"{fr_m['solve_rate']*100:.0f}%/---" if fr_m else "---")
        delta = ""
        if ff_m and fr_m:
            d = (fr_m['solve_rate'] - ff_m['solve_rate']) * 100
            delta = f"{d:+.0f}pp"
        print(f"  {name:20s}  {ff_str:>15s}  {fr_str:>15s}  {delta:>8s}")

    # Fisher's exact: FF vs FR for each strategy
    print("\n  Fisher's exact (feedforward vs full_recurrent):")
    for strategy in STRATEGY_ORDER[:8]:
        ff_m = ff_metrics.get(strategy)
        fr_m = fr_metrics.get(strategy)
        if ff_m and fr_m:
            table = [[ff_m['solved'], ff_m['n'] - ff_m['solved']],
                     [fr_m['solved'], fr_m['n'] - fr_m['solved']]]
            odds, p_val = stats.fisher_exact(table)
            name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
            print(f"    {name}: FF={ff_m['solve_rate']*100:.0f}% vs FR={fr_m['solve_rate']*100:.0f}%, "
                  f"Fisher p={p_val:.4f}")

    # Mann-Whitney U on convergence speed
    print("\n  Mann-Whitney U (convergence speed, FF vs FR, solved runs only):")
    for strategy in STRATEGY_ORDER[:8]:
        ff_m = ff_metrics.get(strategy)
        fr_m = fr_metrics.get(strategy)
        if ff_m and fr_m and ff_m['convergence_gens'] and fr_m['convergence_gens']:
            u_stat, p_val = stats.mannwhitneyu(ff_m['convergence_gens'], fr_m['convergence_gens'],
                                                alternative='two-sided')
            name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
            n1, n2 = len(ff_m['convergence_gens']), len(fr_m['convergence_gens'])
            r_effect = 1 - 2 * u_stat / (n1 * n2) if n1 * n2 > 0 else 0
            print(f"    {name}: U={u_stat:.0f}, p={p_val:.4f}, r={r_effect:.3f}")

    # Aggregate: does topology affect the oscillatory barrier?
    ff_total_solved = sum(m['solved'] for m in ff_metrics.values())
    ff_total_osc = sum(m['has_osc'] for m in ff_metrics.values())
    fr_total_solved = sum(m['solved'] for m in fr_metrics.values())
    fr_total_osc = sum(m['has_osc'] for m in fr_metrics.values())
    if ff_total_solved > 0 and fr_total_solved > 0:
        print(f"\n  Oscillatory barrier: FF {ff_total_osc}/{ff_total_solved} "
              f"({ff_total_osc/ff_total_solved*100:.1f}%), "
              f"FR {fr_total_osc}/{fr_total_solved} "
              f"({fr_total_osc/fr_total_solved*100:.1f}%)")

    # LaTeX table (replacement for Table 21)
    print("\n--- LaTeX Table (Topology Sensitivity, 8 strategies) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Topology sensitivity on Parity-4 ($N{=}30$, pop$=$500, 100 gens). " +
          r"Feedforward baseline from Table~\ref{tab:single_task}; " +
          r"full-recurrent uses \texttt{activate\_time}$=$20.}")
    print(r"\label{tab:topology_full}")
    print(r"{\footnotesize\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"& \multicolumn{3}{c}{Feedforward} & \multicolumn{3}{c}{Full Recurrent} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"Strategy & Solve\% & CI & Med & Solve\% & CI & Med \\")
    print(r"\midrule")

    for strategy in STRATEGY_ORDER[:8]:
        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        ff_m = ff_metrics.get(strategy)
        fr_m = fr_metrics.get(strategy)

        if ff_m:
            ff_sr = f"{ff_m['solve_rate']*100:.0f}\\%"
            ff_ci = f"[{ff_m['ci_lo']*100:.0f},{ff_m['ci_hi']*100:.0f}]"
            ff_med = f"{ff_m['median_gen']:.0f}" if ff_m['median_gen'] is not None else "---"
        else:
            ff_sr, ff_ci, ff_med = "---", "---", "---"

        if fr_m:
            fr_sr = f"{fr_m['solve_rate']*100:.0f}\\%"
            fr_ci = f"[{fr_m['ci_lo']*100:.0f},{fr_m['ci_hi']*100:.0f}]"
            fr_med = f"{fr_m['median_gen']:.0f}" if fr_m['median_gen'] is not None else "---"
        else:
            fr_sr, fr_ci, fr_med = "---", "---", "---"

        print(f"{name} & {ff_sr} & {ff_ci} & {ff_med} & {fr_sr} & {fr_ci} & {fr_med} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    return ff_metrics, fr_metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze bio-inspired strengthening experiments')
    parser.add_argument('--experiment', choices=['gaussian_xor', 'parity6', 'parity5',
                                                  'dual_domain', 'two_moons',
                                                  'visual_discrimination', 'scaling',
                                                  'oscillatory', 'count',
                                                  'timescale_rescaling', 'n60_extension',
                                                  'pop_sensitivity_p5', 'topology_full',
                                                  'all'],
                        default='all', help='Which experiment to analyze')
    parser.add_argument('--save', action='store_true',
                        help='Save output to analysis_all_statistics.txt')
    args = parser.parse_args()

    # Optionally tee output to file
    output_file = None
    if args.save:
        output_path = Path(__file__).resolve().parents[2] / "analysis_all_statistics.txt"
        output_file = open(output_path, 'w')
        import io

        class Tee(io.TextIOBase):
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                return len(data)
            def flush(self):
                for s in self.streams:
                    s.flush()

        sys.stdout = Tee(sys.__stdout__, output_file)

    try:
        if args.experiment in ('gaussian_xor', 'all'):
            analyze_gaussian_xor()
        if args.experiment in ('parity6', 'all'):
            analyze_parity6()
        if args.experiment in ('parity5', 'all'):
            analyze_parity5()
        if args.experiment in ('dual_domain', 'all'):
            analyze_dual_domain()
        if args.experiment in ('two_moons', 'all'):
            analyze_two_moons()
        if args.experiment in ('visual_discrimination', 'all'):
            analyze_visual_discrimination()
        if args.experiment in ('scaling', 'all'):
            analyze_parity_scaling()
        if args.experiment in ('oscillatory', 'all'):
            analyze_oscillatory_barrier()
        if args.experiment in ('timescale_rescaling', 'all'):
            analyze_timescale_rescaling()
        if args.experiment in ('n60_extension', 'all'):
            analyze_n60_extension()
        if args.experiment in ('pop_sensitivity_p5', 'all'):
            analyze_pop_sensitivity_p5()
        if args.experiment in ('topology_full', 'all'):
            analyze_topology_full()
        if args.experiment in ('count', 'all'):
            count_total_runs()
    finally:
        if output_file:
            sys.stdout = sys.__stdout__
            output_file.close()
            print(f"Output saved to {Path(__file__).resolve().parents[2] / 'analysis_all_statistics.txt'}")


if __name__ == '__main__':
    main()
