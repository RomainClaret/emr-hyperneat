#!/usr/bin/env python3
"""Paper 2 Statistical Analysis, Bio-Inspired Strategies for Activation Function Discovery.

Reads N=30 benchmark JSON results from all workstreams and produces:
1. Statistical tests (Mann-Whitney U, Kruskal-Wallis, Dunn's)
2. Effect sizes (rank-biserial)
3. Confidence intervals (Wilson binomial, bootstrap)
4. LaTeX-ready table rows
5. Summary for BENCHMARK_RESULTS.md

Usage:
    python papers/emr-dynamic-functions-bio-inspired/analysis_paper2.py
    python analysis_paper2.py
"""

import argparse
import contextlib
import json
import os
import glob
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy import stats


class _TeeStream:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, file, stdout):
        self.file = file
        self.stdout = stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# =============================================================================
# Statistical utility functions (reused from analysis_round89.py)
# =============================================================================

def find_latest_result(results_dir: str, prefix: str) -> Optional[str]:
    """Find the most recent result file/dir matching prefix."""
    dirs = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*")))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if dirs:
        combined = os.path.join(dirs[-1], "combined_results.json")
        if os.path.exists(combined):
            return combined
        # Try cl_results files for continual learning
        for cl_name in ['cl_results_combined.json', 'cl_results.json']:
            cl_results = os.path.join(dirs[-1], cl_name)
            if os.path.exists(cl_results):
                return cl_results
    files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json")))
    if files:
        return files[-1]
    return None


def binomial_ci_wilson(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion (preferred for small samples)."""
    if trials == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials
    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def binomial_ci_clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
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


def bootstrap_ci(data: List[float], n_bootstrap: int = 10000, confidence: float = 0.95,
                 statistic=np.median) -> Tuple[float, float]:
    """Bootstrap confidence interval for any statistic."""
    if len(data) < 2:
        val = statistic(data) if data else 0.0
        return (val, val)
    rng = np.random.RandomState(42)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))
    alpha = (1 - confidence) / 2
    return (np.percentile(boot_stats, alpha * 100), np.percentile(boot_stats, (1 - alpha) * 100))


def rank_biserial(group1: List[float], group2: List[float]) -> float:
    """Rank-biserial correlation effect size for Mann-Whitney U."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    n1, n2 = len(group1), len(group2)
    return 1 - (2 * u_stat) / (n1 * n2)


def dunns_test(groups: Dict[str, List[float]], alpha: float = 0.05) -> List[Dict]:
    """Post-hoc Dunn's test with Bonferroni correction after Kruskal-Wallis."""
    names = list(groups.keys())
    n_comparisons = len(names) * (len(names) - 1) // 2
    bonferroni_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            g1, g2 = groups[names[i]], groups[names[j]]
            if len(g1) >= 2 and len(g2) >= 2:
                u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                r = rank_biserial(g1, g2)
                results.append({
                    'pair': f"{names[i]} vs {names[j]}",
                    'U': u, 'p': p,
                    'r': r,
                    'significant': p < bonferroni_alpha,
                    'bonferroni_alpha': bonferroni_alpha,
                })
    return results


# =============================================================================
# Workstream A: Single-Task 8-Strategy Analysis
# =============================================================================

def analyze_single_task(filepath: str) -> Dict:
    """Analyze Workstream A: single-task 8-strategy comparison."""
    with open(filepath) as f:
        data = json.load(f)

    results = {}
    for strategy_name, strategy_data in data.items():
        if strategy_name in ('metadata', 'summary'):
            continue
        if not isinstance(strategy_data, dict) or 'results' not in strategy_data:
            continue

        trials = strategy_data['results']
        n_total = len(trials)
        solved = [t for t in trials if t.get('solved', False)]
        n_solved = len(solved)
        gens = [t['solved_gen'] for t in solved if t.get('solved_gen') is not None]

        sin_disc = [t for t in trials if t.get('sin_discovered_gen') is not None]
        sin_retained = [t for t in trials if t.get('sin_retained', False)]

        ci_wilson = binomial_ci_wilson(n_solved, n_total)
        ci_cp = binomial_ci_clopper_pearson(n_solved, n_total)

        median_gen = float(np.median(gens)) if gens else None
        boot_ci = bootstrap_ci(gens) if len(gens) >= 2 else (median_gen, median_gen) if gens else (None, None)

        results[strategy_name] = {
            'n': n_total,
            'n_solved': n_solved,
            'solve_rate': n_solved / n_total if n_total > 0 else 0,
            'ci_wilson': ci_wilson,
            'ci_cp': ci_cp,
            'gens': gens,
            'median_gen': median_gen,
            'mean_gen': float(np.mean(gens)) if gens else None,
            'std_gen': float(np.std(gens)) if gens else None,
            'boot_ci': boot_ci,
            'sin_disc_rate': len(sin_disc) / n_total if n_total > 0 else 0,
            'sin_retained_rate': len(sin_retained) / n_total if n_total > 0 else 0,
        }

    # Print analysis
    print("\n" + "=" * 90)
    print("WORKSTREAM A: SINGLE-TASK 8-STRATEGY COMPARISON")
    print("=" * 90)

    sorted_results = sorted(results.items(), key=lambda x: (-x[1]['solve_rate'], x[1]['median_gen'] or 999))

    print(f"\n{'Strategy':<18} {'N':>3} {'Solved':>6} {'Rate':>6} {'95% CI Wilson':>18} "
          f"{'Med Gen':>8} {'95% CI Boot':>18} {'Sin%':>6}")
    print("-" * 90)

    for name, r in sorted_results:
        ci_str = f"[{r['ci_wilson'][0]:.3f}, {r['ci_wilson'][1]:.3f}]"
        med_str = f"{r['median_gen']:.0f}" if r['median_gen'] is not None else "N/A"
        boot_str = f"[{r['boot_ci'][0]:.1f}, {r['boot_ci'][1]:.1f}]" if r['boot_ci'][0] is not None else "N/A"
        print(f"{name:<18} {r['n']:>3} {r['n_solved']:>6} {r['solve_rate']*100:>5.1f}% "
              f"{ci_str:>18} {med_str:>8} {boot_str:>18} {r['sin_disc_rate']*100:>5.1f}%")

    # Kruskal-Wallis across solved groups
    solved_groups = {k: v['gens'] for k, v in results.items() if v['gens'] and len(v['gens']) >= 2}
    if len(solved_groups) >= 3:
        h, p = stats.kruskal(*solved_groups.values())
        print(f"\nKruskal-Wallis H={h:.2f}, p={p:.2e} (across {len(solved_groups)} strategies)")

        if p < 0.05:
            dunn_results = dunns_test(solved_groups)
            print(f"Post-hoc pairwise comparisons (Bonferroni α={dunn_results[0]['bonferroni_alpha']:.4f}):")
            for dr in dunn_results:
                sig = "***" if dr['significant'] else "ns"
                print(f"  {dr['pair']}: U={dr['U']:.1f}, p={dr['p']:.4f}, r={dr['r']:.3f} {sig}")

    # Generate LaTeX table rows
    print("\n--- LaTeX Table 1 Rows ---")
    category_map = {
        'circadian_rhythm': ('Circadian', 'Oscillatory'),
        'critical_period': ('Crit.\\ Period', 'Developmental'),
        'baseline': ('Baseline', 'Random 10\\% mut.'),
        'stdp': ('STDP', 'Temporal Credit'),
        'hebbian': ('Hebbian', 'Temporal Credit'),
        'metaplastic': ('Metaplastic', 'Homeostatic'),
        'predator_prey': ('Pred.-Prey', 'Ecological'),
        'adult_neurogenesis': ('Neurogenesis', 'Developmental'),
    }
    for name, r in sorted_results:
        display, cat = category_map.get(name, (name, '?'))
        rate = f"{r['solve_rate']*100:.0f}\\%"
        med = f"{r['median_gen']:.1f}" if r['median_gen'] is not None else "N/A"
        sin = f"{r['sin_disc_rate']*100:.0f}\\%"
        bold = "\\textbf{" if r['solve_rate'] == 1.0 and (r['median_gen'] or 999) < 20 else ""
        bold_end = "}" if bold else ""
        print(f"{bold}{display}{bold_end} & {cat} & {rate} & {med} & {sin} \\\\")

    return results


# =============================================================================
# Workstream B/C: Continual Learning Analysis
# =============================================================================

def analyze_continual_learning(filepath: str, label: str = "CL") -> Dict:
    """Analyze continual learning results from Workstreams B or C."""
    with open(filepath) as f:
        data = json.load(f)

    cl_results = data.get('results', {})
    config = data.get('config', {})

    print(f"\n{'=' * 90}")
    print(f"CONTINUAL LEARNING ANALYSIS: {label}")
    print(f"{'=' * 90}")
    print(f"Task sequence: {config.get('task_sequence', 'unknown')}")
    seeds = config.get('seeds', [])
    n_seeds = seeds if isinstance(seeds, int) else len(seeds)
    print(f"Seeds: {n_seeds}")

    strategy_summaries = {}

    for strategy_name, strat_data in cl_results.items():
        runs = strat_data.get('runs', [])
        valid_runs = [r for r in runs if 'metrics' in r]
        if not valid_runs:
            continue

        # Task completion rates
        task_completions = []
        for run in valid_runs:
            tasks = run.get('tasks', [])
            n_solved = sum(1 for t in tasks if t.get('solved', False))
            task_completions.append(n_solved / len(tasks) if tasks else 0)

        # Sin retention across all runs
        sin_retentions = [r['metrics']['sin_retention'] for r in valid_runs]

        # Average accuracy
        avg_accuracies = [r['metrics']['average_accuracy'] for r in valid_runs]

        # FWT estimates
        fwt_estimates = [r['metrics'].get('forward_transfer_estimate', 0) for r in valid_runs]

        # Total solved tasks
        total_solved = [r['metrics']['total_solved'] for r in valid_runs]

        n = len(valid_runs)
        # Infer n_tasks from config or from first run's task list
        task_seq = config.get('task_sequence', [])
        if task_seq:
            n_tasks = len(task_seq)
        elif valid_runs and 'tasks' in valid_runs[0]:
            n_tasks = len(valid_runs[0]['tasks'])
        else:
            n_tasks = 7  # default for 7-task CL
        mean_completion = np.mean(task_completions)
        ci_completion = bootstrap_ci(task_completions)

        strategy_summaries[strategy_name] = {
            'n': n,
            'task_completions': task_completions,
            'mean_completion': mean_completion,
            'ci_completion': ci_completion,
            'sin_retentions': sin_retentions,
            'mean_sin_retention': np.mean(sin_retentions),
            'avg_accuracies': avg_accuracies,
            'mean_accuracy': np.mean(avg_accuracies),
            'fwt_estimates': fwt_estimates,
            'mean_fwt': np.mean(fwt_estimates),
            'total_solved': total_solved,
            'mean_total_solved': np.mean(total_solved),
            'n_tasks': n_tasks,
        }

    # Print summary table
    sorted_strats = sorted(strategy_summaries.items(),
                           key=lambda x: -x[1]['mean_completion'])

    print(f"\n{'Strategy':<30} {'N':>3} {'Completion':>10} {'95% CI':>18} "
          f"{'Sin Ret':>8} {'Avg Acc':>8} {'FWT':>8} {'Total':>6}")
    print("-" * 90)

    for name, s in sorted_strats:
        ci_str = f"[{s['ci_completion'][0]:.3f}, {s['ci_completion'][1]:.3f}]"
        print(f"{name:<30} {s['n']:>3} {s['mean_completion']*100:>9.1f}% "
              f"{ci_str:>18} {s['mean_sin_retention']*100:>7.1f}% "
              f"{s['mean_accuracy']:>8.3f} {s['mean_fwt']:>7.3f} "
              f"{s['mean_total_solved']:>5.1f}/{s['n_tasks']}")

    # Between-strategy comparisons (task completion)
    completion_groups = {k: v['task_completions'] for k, v in strategy_summaries.items()
                         if len(v['task_completions']) >= 2}
    if len(completion_groups) >= 3:
        h, p = stats.kruskal(*completion_groups.values())
        print(f"\nKruskal-Wallis (task completion): H={h:.2f}, p={p:.2e}")
        if p < 0.05:
            dunn_results = dunns_test(completion_groups)
            for dr in dunn_results:
                sig = "***" if dr['significant'] else "ns"
                print(f"  {dr['pair']}: U={dr['U']:.1f}, p={dr['p']:.4f}, r={dr['r']:.3f} {sig}")

    return strategy_summaries


# =============================================================================
# Workstream D/E: Dual-Domain / Failed Mechanisms Analysis
# =============================================================================

def analyze_dual_domain(filepath: str, label: str = "Dual Domain") -> Dict:
    """Analyze dual-domain bio strategy results.

    Supports two formats:
    - Flat: {strategy_name: {results: [...]}, ...}
    - Nested: {problem_name: {strategy_name: {results: [...]}, ...}, ...}
    """
    with open(filepath) as f:
        data = json.load(f)

    results = {}

    for key, value in data.items():
        if key in ('metadata',):
            continue
        if not isinstance(value, dict):
            continue

        # Flat format: value has 'results' directly
        if 'results' in value and isinstance(value['results'], list):
            strategy_name = key
            problem_name = value.get('problem', 'parity_4')
            trials = value['results']
            n_total = len(trials)
            solved = [t for t in trials if t.get('solved', False)]
            n_solved = len(solved)
            gens = [t['solved_gen'] for t in solved if t.get('solved_gen') is not None]

            sin_disc = [t for t in trials if t.get('sin_discovered_gen') is not None]
            sin_retained = [t for t in trials if t.get('sin_retained', False)]
            agg_disc = [t for t in trials if t.get('optimal_agg_discovered_gen') is not None]

            ci = binomial_ci_wilson(n_solved, n_total)
            median_gen = float(np.median(gens)) if gens else None
            boot = bootstrap_ci(gens) if len(gens) >= 2 else (median_gen, median_gen)

            rkey = f"{problem_name}/{strategy_name}"
            results[rkey] = {
                'problem': problem_name,
                'strategy': strategy_name,
                'n': n_total,
                'n_solved': n_solved,
                'solve_rate': n_solved / n_total if n_total > 0 else 0,
                'ci_wilson': ci,
                'gens': gens,
                'median_gen': median_gen,
                'boot_ci': boot,
                'sin_disc_rate': len(sin_disc) / n_total if n_total > 0 else 0,
                'sin_retained_rate': len(sin_retained) / n_total if n_total > 0 else 0,
                'agg_disc_rate': len(agg_disc) / n_total if n_total > 0 else 0,
            }
        else:
            # Nested format: value is {strategy_name: {results: [...]}, ...}
            problem_name = key
            for strategy_name, strategy_data in value.items():
                if not isinstance(strategy_data, dict) or 'results' not in strategy_data:
                    continue

                trials = strategy_data['results']
                n_total = len(trials)
                solved = [t for t in trials if t.get('solved', False)]
                n_solved = len(solved)
                gens = [t['solved_gen'] for t in solved if t.get('solved_gen') is not None]

                sin_disc = [t for t in trials if t.get('sin_discovered_gen') is not None]
                sin_retained = [t for t in trials if t.get('sin_retained', False)]
                agg_disc = [t for t in trials if t.get('optimal_agg_discovered_gen') is not None]

                ci = binomial_ci_wilson(n_solved, n_total)
                median_gen = float(np.median(gens)) if gens else None
                boot = bootstrap_ci(gens) if len(gens) >= 2 else (median_gen, median_gen)

                rkey = f"{problem_name}/{strategy_name}"
                results[rkey] = {
                    'problem': problem_name,
                    'strategy': strategy_name,
                    'n': n_total,
                    'n_solved': n_solved,
                    'solve_rate': n_solved / n_total if n_total > 0 else 0,
                    'ci_wilson': ci,
                    'gens': gens,
                    'median_gen': median_gen,
                    'boot_ci': boot,
                    'sin_disc_rate': len(sin_disc) / n_total if n_total > 0 else 0,
                    'sin_retained_rate': len(sin_retained) / n_total if n_total > 0 else 0,
                    'agg_disc_rate': len(agg_disc) / n_total if n_total > 0 else 0,
                }

    print(f"\n{'=' * 90}")
    print(f"{label.upper()} ANALYSIS")
    print(f"{'=' * 90}")

    sorted_results = sorted(results.items(), key=lambda x: (-x[1]['solve_rate'], x[1]['median_gen'] or 999))

    print(f"\n{'Key':<40} {'N':>3} {'Rate':>6} {'95% CI':>18} "
          f"{'Med Gen':>8} {'Sin%':>6} {'SinRet%':>8} {'Agg4%':>6}")
    print("-" * 100)

    for key, r in sorted_results:
        ci_str = f"[{r['ci_wilson'][0]:.3f}, {r['ci_wilson'][1]:.3f}]"
        med_str = f"{r['median_gen']:.0f}" if r['median_gen'] is not None else "N/A"
        print(f"{key:<40} {r['n']:>3} {r['solve_rate']*100:>5.1f}% "
              f"{ci_str:>18} {med_str:>8} {r['sin_disc_rate']*100:>5.1f}% "
              f"{r['sin_retained_rate']*100:>7.1f}% {r['agg_disc_rate']*100:>5.1f}%")

    return results


# =============================================================================
# Combined Analysis: LaTeX Output
# =============================================================================

def generate_latex_tables(single_task: Dict, cl_elite: Dict, cl_hybrid: Dict,
                          dual_domain: Dict, failed: Dict):
    """Generate all LaTeX table content from combined results."""

    print("\n" + "=" * 90)
    print("LATEX TABLE GENERATION")
    print("=" * 90)

    # Table 1: Single-Task (already printed in analyze_single_task)
    print("\n--- Table 1: Single-Task Strategy Comparison (N=30) ---")
    print("(See Workstream A output above)")

    # Table 2: CL Comparison
    if cl_elite:
        print("\n--- Table 2: 7-Task CL Comparison (N=30) ---")
        for name, s in sorted(cl_elite.items(), key=lambda x: -x[1]['mean_completion']):
            comp = f"{s['mean_completion']*100:.1f}\\%"
            sinr = f"{s['mean_sin_retention']*100:.0f}\\%"
            aa = f"{s['mean_accuracy']:.3f}"
            print(f"{name:<30} & {comp} & {sinr} & {aa} \\\\")

    # Table 6: Dual-Domain
    if dual_domain:
        print("\n--- Table 6: Dual-Domain Discovery (N=30) ---")
        for key, r in sorted(dual_domain.items(), key=lambda x: -x[1]['solve_rate']):
            rate = f"{r['solve_rate']*100:.0f}\\%"
            sin = f"{r['sin_disc_rate']*100:.0f}\\%"
            sinr = f"{r['sin_retained_rate']*100:.0f}\\%"
            agg = f"{r['agg_disc_rate']*100:.0f}\\%"
            print(f"{r['strategy']:<30} & {rate} & {sin} & {sinr} & {agg} \\\\")

    # Table 7: Failed Mechanisms
    if failed:
        print("\n--- Table 7: Failed Mechanisms (N=30) ---")
        for key, r in sorted(failed.items(), key=lambda x: x[1]['solve_rate']):
            rate = f"{r['solve_rate']*100:.0f}\\%"
            ci_str = f"[{r['ci_wilson'][0]:.2f}, {r['ci_wilson'][1]:.2f}]"
            print(f"{r['strategy']:<30} & {rate} & {ci_str} & {r['n']} \\\\")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Paper 2 Statistical Analysis')
    parser.add_argument('--results-dir', type=str,
                        default='results',
                        help='Directory containing benchmark results')
    args = parser.parse_args()

    # Tee stdout to analysis_results.txt alongside console output
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_results.txt')
    output_file = open(output_path, 'w')
    tee = _TeeStream(output_file, sys.stdout)

    with contextlib.redirect_stdout(tee):
        _run_analysis(args)

    output_file.close()
    print(f"\nResults saved to {output_path}")


def _run_analysis(args):
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, args.results_dir)

    print("=" * 90)
    print("PAPER 2 STATISTICAL ANALYSIS — Bio-Inspired Strategies")
    print("=" * 90)
    print(f"Results directory: {results_dir}")

    # Workstream A: Single-task
    single_task_results = {}
    st_file = find_latest_result(results_dir, 'stdp_hebbian_replication')
    if st_file:
        print(f"\nWorkstream A: {st_file}")
        single_task_results = analyze_single_task(st_file)
    else:
        print("\nWorkstream A: No results found")

    # Workstream B: CL Elite
    cl_elite_results = {}
    cl_dirs = sorted(glob.glob(os.path.join(results_dir, 'continual_learning_*')))
    cl_dirs += sorted(glob.glob(os.path.join(base, 'results', 'continual_learning_*')))
    cl_dirs = [d for d in cl_dirs if os.path.isdir(d)]
    # Find the one with elite strategies
    for cl_dir in reversed(cl_dirs):
        for cl_name in ['cl_results_combined.json', 'cl_results.json']:
            cl_file = os.path.join(cl_dir, cl_name)
            if os.path.exists(cl_file):
                with open(cl_file) as f:
                    peek = json.load(f)
                strat_names = list(peek.get('results', {}).keys())
                if 'baseline_dual' in strat_names and 'circadian_rhythm_dual' in strat_names:
                    print(f"\nWorkstream B (CL Elite): {cl_file}")
                    cl_elite_results = analyze_continual_learning(cl_file, "CL Elite")
                    break
        if cl_elite_results:
            break
    if not cl_elite_results:
        print("\nWorkstream B: No results found")

    # Workstream C: CL Hybrid
    cl_hybrid_results = {}
    for cl_dir in reversed(cl_dirs):
        for cl_name in ['cl_results_combined.json', 'cl_results.json']:
            cl_file = os.path.join(cl_dir, cl_name)
            if os.path.exists(cl_file):
                with open(cl_file) as f:
                    peek = json.load(f)
                strat_names = list(peek.get('results', {}).keys())
                if 'stdp_consolidation_dual' in strat_names:
                    print(f"\nWorkstream C (CL Hybrid): {cl_file}")
                    cl_hybrid_results = analyze_continual_learning(cl_file, "CL Hybrid")
                    break
        if cl_hybrid_results:
            break
    if not cl_hybrid_results:
        print("\nWorkstream C: No results found")

    # Workstream D/E: Dual-domain and Failed mechanisms
    # These results live in results/ (project root), not scripts/.../results/
    dual_results_dir = os.path.join(base, 'results')
    dd_dirs = sorted(glob.glob(os.path.join(dual_results_dir, 'bio_dual_palette_*')))
    # Also check the scripts results dir in case they were saved there
    dd_dirs += sorted(glob.glob(os.path.join(results_dir, 'bio_dual_palette_*')))
    dd_dirs = [d for d in dd_dirs if os.path.isdir(d)]

    dual_domain_results = {}
    for dd_dir in reversed(dd_dirs):
        combined = os.path.join(dd_dir, 'combined_results.json')
        if os.path.exists(combined):
            with open(combined) as f:
                peek = json.load(f)
            top_keys = [k for k in peek.keys() if k != 'metadata']
            if 'clonal_selection_dual' in top_keys and 'circadian_rhythm_dual' in top_keys:
                print(f"\nWorkstream D (Dual-Domain): {combined}")
                dual_domain_results = analyze_dual_domain(combined, "Dual Domain")
                break
    if not dual_domain_results:
        print("\nWorkstream D: No results found")

    failed_results = {}
    for dd_dir in reversed(dd_dirs):
        combined = os.path.join(dd_dir, 'combined_results.json')
        if os.path.exists(combined):
            with open(combined) as f:
                peek = json.load(f)
            top_keys = [k for k in peek.keys() if k != 'metadata']
            if 'genetic_regulatory_network_dual' in top_keys and 'glial_modulation_dual' in top_keys:
                print(f"\nWorkstream E (Failed Mechanisms): {combined}")
                failed_results = analyze_dual_domain(combined, "Failed Mechanisms")
                break
    if not failed_results:
        print("\nWorkstream E: No results found")

    # Combined LaTeX output
    generate_latex_tables(single_task_results, cl_elite_results, cl_hybrid_results,
                          dual_domain_results, failed_results)

    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY FOR BENCHMARK_RESULTS.MD")
    print("=" * 90)

    total_experiments = 0
    if single_task_results:
        n = sum(r['n'] for r in single_task_results.values())
        total_experiments += n
        print(f"Workstream A: {n} experiments across {len(single_task_results)} strategies")
    if cl_elite_results:
        n = sum(r['n'] for r in cl_elite_results.values())
        print(f"Workstream B: {n} strategy-seed combinations ({len(cl_elite_results)} strategies)")
    if cl_hybrid_results:
        n = sum(r['n'] for r in cl_hybrid_results.values())
        print(f"Workstream C: {n} strategy-seed combinations ({len(cl_hybrid_results)} strategies)")
    if dual_domain_results:
        n = sum(r['n'] for r in dual_domain_results.values())
        total_experiments += n
        print(f"Workstream D: {n} experiments across {len(dual_domain_results)} strategy-problem pairs")
    if failed_results:
        n = sum(r['n'] for r in failed_results.values())
        total_experiments += n
        print(f"Workstream E: {n} experiments across {len(failed_results)} strategy-problem pairs")

    print(f"\nTotal unique experiments: {total_experiments}")

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
