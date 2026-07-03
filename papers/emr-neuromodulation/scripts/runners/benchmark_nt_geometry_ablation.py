#!/usr/bin/env python3
"""E-S5: NT Geometry Ablation, Distance vs Direction vs Output Polarity.

Tests whether same-class pair compatibility is determined by NT distance,
NT direction, or ACh inversion (output polarity mechanism).

Key finding from E-S3: for complementary pairs (4/4 truth table conflict),
NT distance predicts convergence under Schema A:
  OR+NOR:   dist=0.566 → 0%
  AND+NOR:  dist=0.894 → 16.7%
  AND+NAND: dist=1.200 → 100%

This experiment disentangles the variables:
  Part A: Reduce AND+NAND distance → does rate drop? (distance matters)
  Part B: Increase OR+NOR distance → does rate rise? (distance matters)
  Part C: Rotate OR+NOR direction at fixed distance → stays 0%? (direction irrelevant)
  Part D: Add ACh inversion to OR+NOR → jumps to ~100%? (polarity mechanism)

Conditions (ordered by priority):
  D1: OR+NOR  ACh=0.0   dist=0.566  (inversion mechanism)
  A1: AND+NAND          dist=0.600  (reduced from 1.200)
  A2: AND+NAND          dist=0.300  (further reduced)
  B1: OR+NOR            dist=0.707  (increased from 0.566)
  A3: AND+NAND          dist=0.566  (matching OR+NOR baseline)
  C1: OR+NOR  dir=[+DA,-NE]  dist=0.566  (mirror direction)
  C2: OR+NOR  dir=[-5HT,+NE] dist=0.566  (rotated direction)

Total: 7 conditions × 30 seeds = 210 runs

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_nt_geometry_ablation.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_nt_geometry_ablation.py --summary
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'experiments' / 'neuromodulation'))

from multihead_palette_neuromodulation import run_multihead_palette_experiment

# Lazy JAX import to avoid startup cost during --summary
_jnp = None
def jnp():
    global _jnp
    if _jnp is None:
        import jax.numpy as _j
        _jnp = _j
    return _jnp


# ============================================================================
# Schema A baseline NT vectors (for reference)
# ============================================================================
# AND  = [0.10, 0.90, 0.10, 1.0]
# OR   = [0.50, 0.50, 0.50, 1.0]
# NAND = [0.90, 0.10, 0.50, 1.0]  dist from AND = 1.200
# NOR  = [0.10, 0.50, 0.90, 1.0]  dist from OR  = 0.566


# ============================================================================
# Experiment conditions
# ============================================================================

def make_profiles(task1: str, nt1: List[float],
                  task2: str, nt2: List[float]) -> Dict:
    """Create NT profile dict for a 2-task experiment."""
    return {
        task1: jnp().array(nt1),
        task2: jnp().array(nt2),
    }


def get_conditions() -> List[Dict]:
    """Return all experiment conditions in priority order."""
    return [
        # D1: ACh inversion mechanism (expected ~100%, fastest to run)
        {
            'name': 'D1_or_nor_ach_inversion',
            'pair': ('or', 'nor'),
            'description': 'OR+NOR with ACh=0.0 for NOR (output inversion)',
            'profiles_fn': lambda: make_profiles(
                'or',  [0.50, 0.50, 0.50, 1.0],
                'nor', [0.10, 0.50, 0.90, 0.0],  # ACh=0.0 → inversion
            ),
            'custom_nt': {'or': [0.50, 0.50, 0.50, 1.0],
                          'nor': [0.10, 0.50, 0.90, 0.0]},
            'euclidean_dist': 0.566,
            'variable': 'ACh inversion (polarity mechanism)',
        },
        # A1: AND+NAND distance reduction to 0.600
        {
            'name': 'A1_and_nand_dist0.600',
            'pair': ('and', 'nand'),
            'description': 'AND+NAND with NAND NT at dist=0.600 (reduced from 1.200)',
            'profiles_fn': lambda: make_profiles(
                'and',  [0.10, 0.90, 0.10, 1.0],
                'nand', [0.50, 0.50, 0.30, 1.0],  # interpolated, dist=0.600
            ),
            'custom_nt': {'and': [0.10, 0.90, 0.10, 1.0],
                          'nand': [0.50, 0.50, 0.30, 1.0]},
            'euclidean_dist': 0.600,
            'variable': 'distance reduction (half of original 1.200)',
        },
        # A2: AND+NAND distance reduction to 0.300
        {
            'name': 'A2_and_nand_dist0.300',
            'pair': ('and', 'nand'),
            'description': 'AND+NAND with NAND NT at dist=0.300 (quarter of original)',
            'profiles_fn': lambda: make_profiles(
                'and',  [0.10, 0.90, 0.10, 1.0],
                'nand', [0.30, 0.70, 0.20, 1.0],  # interpolated, dist=0.300
            ),
            'custom_nt': {'and': [0.10, 0.90, 0.10, 1.0],
                          'nand': [0.30, 0.70, 0.20, 1.0]},
            'euclidean_dist': 0.300,
            'variable': 'distance reduction (quarter of original 1.200)',
        },
        # B1: OR+NOR distance increase to 0.707 (max in this direction)
        {
            'name': 'B1_or_nor_dist0.707',
            'pair': ('or', 'nor'),
            'description': 'OR+NOR with NOR NT at dist=0.707 (increased from 0.566)',
            'profiles_fn': lambda: make_profiles(
                'or',  [0.50, 0.50, 0.50, 1.0],
                'nor', [0.00, 0.50, 1.00, 1.0],  # pushed to boundary
            ),
            'custom_nt': {'or': [0.50, 0.50, 0.50, 1.0],
                          'nor': [0.00, 0.50, 1.00, 1.0]},
            'euclidean_dist': 0.707,
            'variable': 'distance increase (25% more than 0.566)',
        },
        # A3: AND+NAND at OR+NOR's distance (0.566), direct comparison
        {
            'name': 'A3_and_nand_dist0.566',
            'pair': ('and', 'nand'),
            'description': 'AND+NAND with NAND NT at dist=0.566 (matching OR+NOR)',
            'profiles_fn': lambda: make_profiles(
                'and',  [0.10, 0.90, 0.10, 1.0],
                'nand', [0.477, 0.523, 0.289, 1.0],  # dist=0.566 along AND→NAND
            ),
            'custom_nt': {'and': [0.10, 0.90, 0.10, 1.0],
                          'nand': [0.477, 0.523, 0.289, 1.0]},
            'euclidean_dist': 0.566,
            'variable': 'distance matching OR+NOR (0.566 vs original 1.200)',
        },
        # C1: OR+NOR direction mirror at same distance
        {
            'name': 'C1_or_nor_dir_mirror',
            'pair': ('or', 'nor'),
            'description': 'OR+NOR at dist=0.566, direction [+DA,-NE] (mirror of original)',
            'profiles_fn': lambda: make_profiles(
                'or',  [0.50, 0.50, 0.50, 1.0],
                'nor', [0.90, 0.50, 0.10, 1.0],  # mirror: +DA, -NE
            ),
            'custom_nt': {'or': [0.50, 0.50, 0.50, 1.0],
                          'nor': [0.90, 0.50, 0.10, 1.0]},
            'euclidean_dist': 0.566,
            'variable': 'direction mirror at fixed distance',
        },
        # C2: OR+NOR direction rotated at same distance
        {
            'name': 'C2_or_nor_dir_rotated',
            'pair': ('or', 'nor'),
            'description': 'OR+NOR at dist=0.566, direction [-5HT,+NE] (rotated)',
            'profiles_fn': lambda: make_profiles(
                'or',  [0.50, 0.50, 0.50, 1.0],
                'nor', [0.50, 0.10, 0.90, 1.0],  # rotated: -5HT, +NE
            ),
            'custom_nt': {'or': [0.50, 0.50, 0.50, 1.0],
                          'nor': [0.50, 0.10, 0.90, 1.0]},
            'euclidean_dist': 0.566,
            'variable': 'direction rotation at fixed distance',
        },
    ]


# ============================================================================
# Runner
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'nt_geometry_ablation'
SEEDS = list(range(1, 31))


def run_condition(condition: Dict, seeds: List[int] = SEEDS,
                  verbose: bool = True) -> Dict:
    """Run a single ablation condition across all seeds."""
    name = condition['name']
    pair = condition['pair']
    profiles = condition['profiles_fn']()

    result_file = RESULTS_DIR / f'{name}.json'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing_seeds = set()
    existing_results = []
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)
        existing_seeds = {r['seed'] for r in existing.get('results', [])}
        existing_results = existing.get('results', [])
        if verbose:
            print(f"  Resuming {name}: {len(existing_seeds)}/{len(seeds)} done")

    results = list(existing_results)

    for seed in seeds:
        if seed in existing_seeds:
            continue

        if verbose:
            print(f"  [{name}] seed={seed}", end=' ', flush=True)

        result = run_multihead_palette_experiment(
            task_names=list(pair),
            palette_mode='uniform',
            blend_mode='fixed',
            aggregation='product',
            seed=seed,
            generations=100,
            population=750,
            max_depth=4,
            success_threshold=0.98,
            verbose=False,
            nt_profiles=profiles,
        )

        result_dict = {
            'pair': '+'.join(pair),
            'seed': seed,
            'converged': result.converged,
            'convergence_gen': result.convergence_gen,
            'individual_min_fitness': result.individual_min_fitness,
            'per_task_fitness': result.per_task_fitness,
            'runtime_seconds': result.runtime_seconds,
        }
        results.append(result_dict)

        if verbose:
            status = "\u2713" if result.converged else "\u2717"
            gen_str = (f"gen {result.convergence_gen}" if result.converged
                       else f"min={result.individual_min_fitness:.3f}")
            print(f"{status} {gen_str} ({result.runtime_seconds:.0f}s)")

        # Incremental save
        _save_condition(result_file, condition, results)

    return _summarize_condition(condition, results)


def _save_condition(path: Path, condition: Dict, results: List[Dict]):
    """Save condition results to JSON."""
    n = len(results)
    conv = sum(1 for r in results if r['converged'])
    with open(path, 'w') as f:
        json.dump({
            'condition': condition['name'],
            'pair': '+'.join(condition['pair']),
            'description': condition['description'],
            'custom_nt': condition['custom_nt'],
            'euclidean_dist': condition['euclidean_dist'],
            'variable': condition['variable'],
            'n_seeds': n,
            'convergence_count': conv,
            'convergence_rate': conv / n if n > 0 else 0.0,
            'results': results,
        }, f, indent=2)


def _summarize_condition(condition: Dict, results: List[Dict]) -> Dict:
    """Summarize a condition's results."""
    n = len(results)
    conv = sum(1 for r in results if r['converged'])
    conv_gens = [r['convergence_gen'] for r in results if r['converged']]
    return {
        'name': condition['name'],
        'pair': '+'.join(condition['pair']),
        'euclidean_dist': condition['euclidean_dist'],
        'variable': condition['variable'],
        'n_seeds': n,
        'convergence_count': conv,
        'convergence_rate': conv / n if n > 0 else 0.0,
        'median_gen': float(np.median(conv_gens)) if conv_gens else None,
    }


def print_summary():
    """Print summary of all completed conditions."""
    print("=" * 75)
    print("E-S5: NT Geometry Ablation — Results Summary")
    print("=" * 75)

    # Baselines from existing data
    print("\nBaselines (from Schema A uniform tanh):")
    print(f"  {'Pair':<15} {'Dist':>6} {'Rate':>8} {'Source'}")
    print(f"  {'-'*50}")
    baselines = [
        ('or+nor',   0.566, '0.0%',   'Schema A'),
        ('and+nor',  0.894, '16.7%',  'Schema A'),
        ('and+nand', 1.200, '100.0%', 'Schema A'),
    ]
    for pair, dist, rate, src in baselines:
        print(f"  {pair:<15} {dist:>6.3f} {rate:>8} {src}")

    print(f"\nAblation conditions:")
    print(f"  {'Condition':<30} {'Pair':<12} {'Dist':>6} {'Rate':>8} {'N':>4} {'Med Gen':>8} {'Variable'}")
    print(f"  {'-'*100}")

    conditions = get_conditions()
    for cond in conditions:
        result_file = RESULTS_DIR / f"{cond['name']}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            n = data['n_seeds']
            conv = data['convergence_count']
            rate = f"{conv/n*100:.1f}%" if n > 0 else 'N/A'
            conv_gens = [r['convergence_gen'] for r in data['results'] if r['converged']]
            med = f"{np.median(conv_gens):.0f}" if conv_gens else '-'
            pair = '+'.join(cond['pair'])
            print(f"  {cond['name']:<30} {pair:<12} {cond['euclidean_dist']:>6.3f} "
                  f"{rate:>8} {n:>4} {med:>8} {cond['variable']}")
        else:
            pair = '+'.join(cond['pair'])
            print(f"  {cond['name']:<30} {pair:<12} {cond['euclidean_dist']:>6.3f} "
                  f"{'--':>8} {'--':>4} {'--':>8} {cond['variable']}")

    total_done = sum(
        1 for c in conditions
        if (RESULTS_DIR / f"{c['name']}.json").exists()
    )
    total_runs = sum(
        json.load(open(RESULTS_DIR / f"{c['name']}.json"))['n_seeds']
        for c in conditions
        if (RESULTS_DIR / f"{c['name']}.json").exists()
    )
    print(f"\n  Conditions complete: {total_done}/{len(conditions)}")
    print(f"  Total runs: {total_runs}")


def main():
    parser = argparse.ArgumentParser(description='E-S5: NT Geometry Ablation')
    parser.add_argument('--summary', action='store_true', help='Print summary only')
    parser.add_argument('--condition', type=str, help='Run specific condition by name')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    conditions = get_conditions()

    if args.condition:
        conditions = [c for c in conditions if c['name'] == args.condition]
        if not conditions:
            print(f"Unknown condition: {args.condition}")
            print("Available:", [c['name'] for c in get_conditions()])
            return

    # Verify distances
    print("Verifying NT distances...")
    for cond in conditions:
        profiles = cond['custom_nt']
        tasks = list(cond['pair'])
        v1 = np.array(profiles[tasks[0]][:3])
        v2 = np.array(profiles[tasks[1]][:3])
        actual_dist = np.linalg.norm(v1 - v2)
        expected = cond['euclidean_dist']
        assert abs(actual_dist - expected) < 0.01, \
            f"{cond['name']}: expected dist={expected}, got {actual_dist:.3f}"
        print(f"  {cond['name']}: dist={actual_dist:.3f} \u2713")

    print(f"\n{'=' * 70}")
    print(f"E-S5: NT Geometry Ablation")
    print(f"Conditions: {len(conditions)}, Seeds: {len(SEEDS)}")
    print(f"Total runs: {len(conditions) * len(SEEDS)}")
    print(f"{'=' * 70}")

    start = time.time()
    summaries = []

    for i, cond in enumerate(conditions):
        print(f"\n{'#' * 60}")
        print(f"# [{i+1}/{len(conditions)}] {cond['name']}")
        print(f"# {cond['description']}")
        print(f"# Distance: {cond['euclidean_dist']:.3f}")
        print(f"{'#' * 60}")

        summary = run_condition(cond)
        summaries.append(summary)

        rate = summary['convergence_rate'] * 100
        med = summary['median_gen']
        med_str = f"median gen {med:.0f}" if med is not None else "no convergence"
        print(f"\n  >>> {cond['name']}: {summary['convergence_count']}/{summary['n_seeds']} "
              f"({rate:.1f}%), {med_str}")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"ALL CONDITIONS COMPLETE — {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"{'=' * 70}")
    print_summary()


if __name__ == '__main__':
    main()
