#!/usr/bin/env python3
"""Experiment A3: Per-Strategy Oscillatory Function Composition Analysis.

Analyzes existing oscillatory analysis data to show that strategies discover
*different* oscillatory function portfolios, the oscillatory property matters,
not the exact function.

Key outputs:
- Normalized function share per strategy
- Shannon entropy (portfolio diversity)
- Chi-square test for independence (8×5 contingency table)
- LaTeX-ready numbers for paper integration

Data source:
    results/oscillatory_analysis/oscillatory_analysis_all_strategies.json

Usage:
    python papers/emr-dynamic-functions-bio-inspired/analysis_oscillatory_composition.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from scipy import stats

# Oscillatory function index → name mapping
OSC_NAMES = {
    4: 'sin',
    11: 'burst',
    12: 'resonator',
    13: 'osc_adapt',
    15: 'receptive',
}

OSC_INDICES = [4, 11, 12, 13, 15]


def load_data(results_dir: Path) -> Dict[str, Any]:
    """Load oscillatory analysis JSON."""
    filepath = results_dir / 'oscillatory_analysis' / 'oscillatory_analysis_all_strategies.json'
    with open(filepath) as f:
        return json.load(f)


def compute_normalized_shares(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Compute normalized function share per strategy (count / total_osc_count)."""
    shares = {}
    for strategy, info in data['per_strategy'].items():
        counts = info.get('osc_function_counts', {})
        total = sum(counts.values())
        if total == 0:
            shares[strategy] = {OSC_NAMES[idx]: 0.0 for idx in OSC_INDICES}
            continue
        share = {}
        for idx in OSC_INDICES:
            count = counts.get(str(idx), 0)
            share[OSC_NAMES[idx]] = count / total
        shares[strategy] = share
    return shares


def compute_shannon_entropy(data: Dict[str, Any]) -> Dict[str, float]:
    """Compute Shannon entropy of each strategy's oscillatory portfolio."""
    entropies = {}
    for strategy, info in data['per_strategy'].items():
        counts = info.get('osc_function_counts', {})
        total = sum(counts.values())
        if total == 0:
            entropies[strategy] = 0.0
            continue
        probs = []
        for idx in OSC_INDICES:
            count = counts.get(str(idx), 0)
            if count > 0:
                probs.append(count / total)
        # Shannon entropy: -sum(p * log2(p))
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        entropies[strategy] = entropy
    return entropies


def build_contingency_table(data: Dict[str, Any]) -> np.ndarray:
    """Build 8×5 contingency table (strategies × oscillatory functions)."""
    strategies = sorted(data['per_strategy'].keys())
    table = np.zeros((len(strategies), len(OSC_INDICES)), dtype=int)
    for i, strategy in enumerate(strategies):
        counts = data['per_strategy'][strategy].get('osc_function_counts', {})
        for j, idx in enumerate(OSC_INDICES):
            table[i, j] = counts.get(str(idx), 0)
    return table, strategies


def run_chi_square(table: np.ndarray) -> Dict[str, float]:
    """Run chi-square test for independence on contingency table."""
    chi2, p, dof, expected = stats.chi2_contingency(table)
    # Cramér's V effect size
    n = table.sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0
    return {
        'chi2': chi2,
        'p_value': p,
        'dof': dof,
        'cramers_v': cramers_v,
        'n': int(n),
    }


def compute_composite_only_rates(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Compute composite-only solve rates (solved without pure sin)."""
    rates = {}
    for strategy, info in data['per_strategy'].items():
        n_solved = info.get('n_solved', 0)
        composite_only = info.get('solved_composite_only', 0)
        sin_disc_pct = info.get('sin_disc_pct', 0)
        rates[strategy] = {
            'n_solved': n_solved,
            'composite_only': composite_only,
            'composite_only_pct': composite_only / n_solved * 100 if n_solved > 0 else 0,
            'sin_pct': sin_disc_pct,
        }
    return rates


def format_latex_table(shares: Dict, entropies: Dict, composite_rates: Dict,
                       data: Dict) -> str:
    """Format results as LaTeX-ready table."""
    max_entropy = np.log2(5)  # Maximum possible with 5 functions
    lines = []
    lines.append("% A3: Oscillatory Function Composition")
    lines.append("% Strategy & Sin% & Burst% & Reson.% & Osc.Ad.% & Recept.% & H & Comp.Only%")

    # Sort by solve rate descending
    strategy_order = sorted(
        data['per_strategy'].keys(),
        key=lambda s: data['per_strategy'][s].get('solve_rate', 0),
        reverse=True,
    )

    for strategy in strategy_order:
        s = shares.get(strategy, {})
        h = entropies.get(strategy, 0)
        cr = composite_rates.get(strategy, {})
        line = (f"% {strategy:<20s} "
                f"{s.get('sin', 0)*100:5.1f}  "
                f"{s.get('burst', 0)*100:5.1f}  "
                f"{s.get('resonator', 0)*100:5.1f}  "
                f"{s.get('osc_adapt', 0)*100:5.1f}  "
                f"{s.get('receptive', 0)*100:5.1f}  "
                f"{h:5.3f}/{max_entropy:.3f}  "
                f"{cr.get('composite_only_pct', 0):5.1f}%")
        lines.append(line)

    return '\n'.join(lines)


def main():
    results_dir = Path(__file__).resolve().parents[2] / 'results'

    print("=" * 70)
    print("A3: Per-Strategy Oscillatory Function Composition")
    print("=" * 70)

    # Load data
    data = load_data(results_dir)
    n_strategies = len(data['per_strategy'])
    print(f"\nLoaded data for {n_strategies} strategies")

    # 1. Normalized shares
    shares = compute_normalized_shares(data)
    print("\n--- Normalized Function Shares ---")
    print(f"{'Strategy':<20s} {'sin':>6s} {'burst':>6s} {'reson':>6s} {'osc_ad':>6s} {'recept':>6s}")
    print("-" * 56)
    for strategy in sorted(shares.keys()):
        s = shares[strategy]
        print(f"{strategy:<20s} {s['sin']*100:5.1f}% {s['burst']*100:5.1f}% "
              f"{s['resonator']*100:5.1f}% {s['osc_adapt']*100:5.1f}% "
              f"{s['receptive']*100:5.1f}%")

    # 2. Shannon entropy
    entropies = compute_shannon_entropy(data)
    max_h = np.log2(5)
    print(f"\n--- Shannon Entropy (max = {max_h:.3f} for 5 functions) ---")
    for strategy in sorted(entropies.keys(), key=lambda s: entropies[s], reverse=True):
        h = entropies[strategy]
        print(f"  {strategy:<20s} H = {h:.3f} ({h/max_h*100:.1f}% of max)")

    # 3. Chi-square test
    table, strategy_names = build_contingency_table(data)
    chi_sq_result = run_chi_square(table)
    print(f"\n--- Chi-Square Test for Independence ---")
    print(f"  Contingency table: {table.shape[0]} strategies × {table.shape[1]} functions")
    print(f"  χ² = {chi_sq_result['chi2']:.2f}")
    print(f"  df = {chi_sq_result['dof']}")
    print(f"  p = {chi_sq_result['p_value']:.2e}")
    print(f"  Cramér's V = {chi_sq_result['cramers_v']:.3f}")
    print(f"  N (total observations) = {chi_sq_result['n']}")

    if chi_sq_result['p_value'] < 0.001:
        print("  → SIGNIFICANT: Strategies discover different oscillatory portfolios (p < 0.001)")
    elif chi_sq_result['p_value'] < 0.05:
        print("  → SIGNIFICANT: Strategies discover different oscillatory portfolios (p < 0.05)")
    else:
        print("  → NOT significant: No evidence of portfolio differentiation")

    # 4. Composite-only rates
    composite_rates = compute_composite_only_rates(data)
    print(f"\n--- Composite-Only Solve Rates ---")
    print(f"{'Strategy':<20s} {'Solved':>6s} {'Comp.Only':>9s} {'Rate':>6s}")
    print("-" * 45)
    for strategy in sorted(composite_rates.keys()):
        cr = composite_rates[strategy]
        print(f"{strategy:<20s} {cr['n_solved']:>6d} {cr['composite_only']:>9d} "
              f"{cr['composite_only_pct']:>5.1f}%")

    # 5. Key narrative numbers
    print(f"\n--- Key Numbers for Paper ---")
    total_solved = data['aggregate']['total_solved']
    total_composite = data['aggregate']['total_composite_only']
    total_sin_only = data['aggregate']['total_solved_sin_only']
    composite_pct = total_composite / total_solved * 100
    sin_only_pct = total_sin_only / total_solved * 100
    print(f"  Total solved: {total_solved}")
    print(f"  Composite-only: {total_composite} ({composite_pct:.1f}%)")
    print(f"  Sin+composites: {total_sin_only} ({sin_only_pct:.1f}%)")
    print(f"  Predator-prey: 0% sin, 100% composite-only ({composite_rates.get('predator_prey', {}).get('composite_only', 0)}/{data['per_strategy'].get('predator_prey', {}).get('n_solved', 0)} solved)")

    # Entropy range
    entropy_vals = list(entropies.values())
    print(f"  Entropy range: {min(entropy_vals):.3f} – {max(entropy_vals):.3f} (max={max_h:.3f})")

    # 6. LaTeX table
    latex = format_latex_table(shares, entropies, composite_rates, data)
    print(f"\n--- LaTeX-Ready Numbers ---")
    print(latex)

    # Save analysis results
    output = {
        'normalized_shares': {k: {fk: round(fv, 4) for fk, fv in v.items()} for k, v in shares.items()},
        'shannon_entropy': {k: round(v, 4) for k, v in entropies.items()},
        'max_entropy': round(max_h, 4),
        'chi_square': {k: round(v, 6) if isinstance(v, float) else v for k, v in chi_sq_result.items()},
        'composite_only_rates': composite_rates,
        'aggregate': data['aggregate'],
    }

    output_file = results_dir / 'oscillatory_analysis' / 'oscillatory_composition_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
