#!/usr/bin/env python3
"""Statistical analysis of the N=30 benchmark results.

Reads N=30 benchmark JSON results and produces:
1. Statistical tests (Mann-Whitney U, Kruskal-Wallis, Spearman, etc.)
2. Effect sizes (rank-biserial, Cohen's d)
3. Confidence intervals (95% CI, exact binomial)

Usage:
    python analysis_round89.py            # defaults to this paper's results/ directory

The script auto-discovers the most recent JSON files for each experiment type.
"""

import argparse
import json
import os
import glob
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def find_latest_result(results_dir: str, prefix: str) -> str | None:
    """Find the most recent result file/dir matching prefix."""
    # Try directory pattern first (combined_results.json inside)
    dirs = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*")))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if dirs:
        combined = os.path.join(dirs[-1], "combined_results.json")
        if os.path.exists(combined):
            return combined
    # Try direct JSON file
    files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json")))
    if files:
        return files[-1]
    return None


def binomial_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple:
    """Exact Clopper-Pearson binomial confidence interval."""
    alpha = 1 - confidence
    if successes == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    if successes == trials:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower, upper)


def rank_biserial(group1, group2):
    """Rank-biserial correlation effect size for Mann-Whitney U."""
    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * u_stat) / (n1 * n2)
    return r


def analyze_xor_speedup(filepath: str) -> dict:
    """Analyze XOR speedup benchmark results."""
    with open(filepath) as f:
        data = json.load(f)

    results = {}

    # Handle nested structure: data[config_name]['results'] = [trials]
    # or flat structure: data['results'][config_name] = [trials]
    for config_name, config_data in data.items():
        if config_name in ('metadata', 'speedup_analysis', 'summary'):
            continue
        trials = None
        if isinstance(config_data, dict) and 'results' in config_data:
            trials = config_data['results']
        elif isinstance(config_data, list):
            trials = config_data
        if trials and isinstance(trials, list):
            solved = [t for t in trials if t.get('solved', False)]
            gens = [t['solved_gen'] for t in solved if t.get('solved_gen') is not None]
            n_total = len(trials)
            n_solved = len(solved)
            results[config_name] = {
                'n': n_total,
                'n_solved': n_solved,
                'solve_rate': n_solved / n_total if n_total > 0 else 0,
                'gens': gens,
                'avg_gen': np.mean(gens) if gens else None,
                'std_gen': np.std(gens) if gens else None,
                'ci': binomial_ci(n_solved, n_total),
            }

    print("\n" + "=" * 70)
    print("XOR SPEEDUP ANALYSIS")
    print("=" * 70)

    for name, r in results.items():
        ci_str = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        avg_str = f"{r['avg_gen']:.1f}" if r['avg_gen'] else "N/A"
        print(f"  {name}: {r['n_solved']}/{r['n']} ({r['solve_rate']*100:.1f}%) "
              f"avg_gen={avg_str} CI={ci_str}")

    # Mann-Whitney U: sin_only vs default
    if 'sin_only' in results and 'default' in results:
        sin_gens = results['sin_only']['gens']
        def_gens = results['default']['gens']
        if sin_gens and def_gens:
            u, p = stats.mannwhitneyu(sin_gens, def_gens, alternative='two-sided')
            r_rb = rank_biserial(sin_gens, def_gens)
            speedup = np.mean(def_gens) / np.mean(sin_gens) if np.mean(sin_gens) > 0 else float('inf')
            print(f"\n  Sin vs Default speedup: {speedup:.1f}×")
            print(f"  Mann-Whitney U={u:.1f}, p={p:.2e}, rank-biserial r={r_rb:.3f}")

    return results


def analyze_strategies(filepath: str) -> dict:
    """Analyze STDP/Hebbian strategy comparison results."""
    with open(filepath) as f:
        data = json.load(f)

    results = {}

    for strategy_name, strategy_data_item in data.items():
        if strategy_name in ('metadata', 'summary'):
            continue
        trials = None
        if isinstance(strategy_data_item, dict) and 'results' in strategy_data_item:
            trials = strategy_data_item['results']
        elif isinstance(strategy_data_item, list):
            trials = strategy_data_item
        if trials and isinstance(trials, list):
            solved = [t for t in trials if t.get('solved', False)]
            gens = [t['solved_gen'] for t in solved if t.get('solved_gen') is not None]
            n_total = len(trials)
            n_solved = len(solved)
            results[strategy_name] = {
                'n': n_total,
                'n_solved': n_solved,
                'solve_rate': n_solved / n_total if n_total > 0 else 0,
                'gens': gens,
                'avg_gen': np.mean(gens) if gens else None,
                'std_gen': np.std(gens) if gens else None,
                'ci': binomial_ci(n_solved, n_total),
            }

    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON ANALYSIS")
    print("=" * 70)

    for name, r in sorted(results.items(), key=lambda x: -(x[1]['solve_rate'])):
        ci_str = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        avg_str = f"{r['avg_gen']:.1f}" if r['avg_gen'] else "N/A"
        print(f"  {name}: {r['n_solved']}/{r['n']} ({r['solve_rate']*100:.1f}%) "
              f"avg_gen={avg_str} CI={ci_str}")

    # Kruskal-Wallis across strategies with solved runs
    solved_groups = {k: v['gens'] for k, v in results.items() if v['gens']}
    if len(solved_groups) >= 3:
        groups = list(solved_groups.values())
        h, p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis H={h:.2f}, p={p:.2e} (across {len(groups)} strategies)")

        # Post-hoc Dunn's test (manual pairwise Mann-Whitney with Bonferroni)
        names = list(solved_groups.keys())
        n_comparisons = len(names) * (len(names) - 1) // 2
        print(f"  Post-hoc pairwise comparisons (Bonferroni α={0.05/n_comparisons:.4f}):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                u, p_pair = stats.mannwhitneyu(
                    solved_groups[names[i]], solved_groups[names[j]],
                    alternative='two-sided'
                )
                sig = "***" if p_pair < 0.05 / n_comparisons else "ns"
                print(f"    {names[i]} vs {names[j]}: U={u:.1f}, p={p_pair:.4f} {sig}")

    return results


def analyze_palette_antagonism(filepath: str) -> dict:
    """Analyze palette size antagonism results."""
    with open(filepath) as f:
        data = json.load(f)

    summary = data.get('summary', {})
    results_raw = data.get('results', {})

    print("\n" + "=" * 70)
    print("PALETTE SIZE ANTAGONISM ANALYSIS")
    print("=" * 70)

    sizes = []
    avg_gens = []

    for size_key in sorted(summary.keys(), key=lambda x: int(x.split('_')[1])):
        s = summary[size_key]
        size = int(size_key.split('_')[1])
        n = s['n_runs']
        rate = s['success_rate']
        avg = s.get('avg_gen')
        std = s.get('std_gen')
        ci = binomial_ci(s['n_solved'], n)

        if avg is not None:
            sizes.append(size)
            avg_gens.append(avg)

        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        avg_str = f"{avg:.1f}" if avg else "N/A"
        std_str = f"±{std:.2f}" if std else ""
        print(f"  Size {size}: {rate}% success (N={n}) "
              f"avg_gen={avg_str}{std_str} CI={ci_str}")

    # Spearman rank correlation: palette size vs convergence speed
    if len(sizes) >= 3:
        rho, p = stats.spearmanr(sizes, avg_gens)
        print(f"\n  Spearman correlation (size vs avg_gen): ρ={rho:.3f}, p={p:.4f}")

    return summary


def analyze_hanoi(filepath: str) -> dict:
    """Analyze Hanoi verification results."""
    with open(filepath) as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("HANOI PLATEAU ANALYSIS")
    print("=" * 70)

    variant_fitness = {}

    for variant_name, variant_data in data.items():
        if variant_name in ('metadata', 'summary', 'cross_variant_analysis'):
            continue
        trials = None
        if isinstance(variant_data, dict) and 'results' in variant_data:
            trials = variant_data['results']
        elif isinstance(variant_data, list):
            trials = variant_data
        if trials and isinstance(trials, list):
            fitnesses = [t['best_fitness'] for t in trials]
            avg = np.mean(fitnesses)
            std = np.std(fitnesses)
            ci = stats.t.interval(0.95, len(fitnesses) - 1,
                                  loc=avg, scale=stats.sem(fitnesses))
            variant_fitness[variant_name] = fitnesses
            print(f"  {variant_name}: {avg*100:.1f}% ± {std*100:.1f}% "
                  f"95% CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")

    # One-sample Wilcoxon against 0.78 for state-reasoning variants
    state_variants = {k: v for k, v in variant_fitness.items()
                      if k != 'memorization' and len(v) >= 5}
    if state_variants:
        all_fitness = []
        for v in state_variants.values():
            all_fitness.extend(v)
        w, p = stats.wilcoxon([f - 0.78 for f in all_fitness])
        print(f"\n  Wilcoxon against 78%: W={w:.1f}, p={p:.4f}")

    return variant_fitness


def analyze_two_spirals(filepath: str) -> dict:
    """Analyze Two Spirals verification results."""
    with open(filepath) as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("TWO SPIRALS CEILING ANALYSIS")
    print("=" * 70)

    config_results = {}

    for config_name, config_data_item in data.items():
        if config_name in ('metadata', 'summary', 'ceiling_analysis'):
            continue
        # Two spirals has nested structure with depths
        trials = None
        if isinstance(config_data_item, dict) and 'results' in config_data_item:
            trials = config_data_item['results']
        elif isinstance(config_data_item, list):
            trials = config_data_item
        if trials and isinstance(trials, list):
            fitnesses = [t['best_fitness'] for t in trials]
            avg = np.mean(fitnesses)
            std = np.std(fitnesses)
            ci = stats.t.interval(0.95, len(fitnesses) - 1,
                                  loc=avg, scale=stats.sem(fitnesses)) if len(fitnesses) > 1 else (avg, avg)
            config_results[config_name] = {
                'fitnesses': fitnesses,
                'avg': avg,
                'std': std,
                'ci': ci,
            }
            print(f"  {config_name}: {avg*100:.1f}% ± {std*100:.1f}% "
                  f"95% CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")

    return config_results


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of the benchmark results')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory containing benchmark results (default: this paper''s results/)')
    args = parser.parse_args()

    from pathlib import Path
    results_dir = args.results_dir or str(Path(__file__).resolve().parents[2] / 'results')

    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print(f"Results directory: {results_dir}")

    # Find and analyze each experiment type
    xor_file = find_latest_result(results_dir, 'xor_speedup')
    if xor_file:
        print(f"\nXOR: {xor_file}")
        analyze_xor_speedup(xor_file)
    else:
        print("\nXOR: No results found")

    strategy_file = find_latest_result(results_dir, 'stdp_hebbian')
    if strategy_file:
        print(f"\nStrategies: {strategy_file}")
        analyze_strategies(strategy_file)
    else:
        print("\nStrategies: No results found")

    palette_file = find_latest_result(results_dir, 'palette_size_antagonism')
    if palette_file:
        print(f"\nPalette: {palette_file}")
        analyze_palette_antagonism(palette_file)
    else:
        print("\nPalette: No results found")

    hanoi_file = find_latest_result(results_dir, 'hanoi_verification')
    if hanoi_file:
        print(f"\nHanoi: {hanoi_file}")
        analyze_hanoi(hanoi_file)
    else:
        print("\nHanoi: No results found")

    spirals_file = find_latest_result(results_dir, 'two_spirals')
    if spirals_file:
        print(f"\nTwo Spirals: {spirals_file}")
        analyze_two_spirals(spirals_file)
    else:
        print("\nTwo Spirals: No results found")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
