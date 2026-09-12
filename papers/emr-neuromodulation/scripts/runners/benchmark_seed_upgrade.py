#!/usr/bin/env python3
"""E-S1: Benchmark Seed Upgrade to n=30.

Upgrades the original benchmark (n=3 per condition) to n=30 for statistical rigor.
Uses the same CPPN/NEAT pipeline as the original benchmark, with uniform tanh,
feedforward topology, Pop=750, 100 generations.

Conditions to run:
  - 5 single-task (xor, and, or, nand, nor) × 30 seeds = 150 runs
  - 10 three-task combos × 30 seeds = 300 runs
  - 5 four-task combos × 30 seeds = 150 runs
  - 1 five-task × 30 seeds = 30 runs
  Total: 630 runs

Note: 2-task data already exists at n=30 from ablation_schema_a/b_uniform experiments.

Results saved to: papers/emr-neuromodulation/results/benchmark_n30/

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_seed_upgrade.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_seed_upgrade.py \
        --conditions 1task_xor 1task_and  # specific conditions
    python papers/emr-neuromodulation/scripts/runners/benchmark_seed_upgrade.py \
        --seeds 3  # smoke test
    python papers/emr-neuromodulation/scripts/runners/benchmark_seed_upgrade.py --summary
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path('.') / 'src'))

from experiments.neuromodulation.multihead_palette_neuromodulation import (
    ALL_TASKS,
    run_multihead_palette_experiment,
)


# ============================================================================
# Constants
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'benchmark_n30'

# All task subsets to test
SINGLE_TASKS = [[t] for t in ALL_TASKS]  # 5 conditions

THREE_TASK_COMBOS = [list(c) for c in combinations(ALL_TASKS, 3)]  # 10 conditions

FOUR_TASK_COMBOS = [list(c) for c in combinations(ALL_TASKS, 4)]  # 5 conditions

FIVE_TASK = [ALL_TASKS]  # 1 condition

# Build condition name → task list mapping
ALL_CONDITIONS = {}
for tasks in SINGLE_TASKS:
    name = f"1task_{'_'.join(tasks)}"
    ALL_CONDITIONS[name] = tasks
for tasks in THREE_TASK_COMBOS:
    name = f"3task_{'_'.join(tasks)}"
    ALL_CONDITIONS[name] = tasks
for tasks in FOUR_TASK_COMBOS:
    name = f"4task_{'_'.join(tasks)}"
    ALL_CONDITIONS[name] = tasks
for tasks in FIVE_TASK:
    name = f"5task_{'_'.join(tasks)}"
    ALL_CONDITIONS[name] = tasks


# ============================================================================
# Utilities
# ============================================================================

def result_exists(filepath: Path) -> bool:
    """Check if result file exists and is non-empty."""
    return filepath.exists() and filepath.stat().st_size > 0


def save_result(result_dict: Dict, filepath: Path):
    """Save result to JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        return str(obj)

    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2, default=convert)


def summarize_condition(results_dir: Path, prefix: str):
    """Summarize results for a condition."""
    files = sorted(results_dir.glob(f'{prefix}_seed*.json'))
    if not files:
        print(f"  {prefix}: no results")
        return

    converged = 0
    gens = []
    min_accs = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if data.get('converged', False):
            converged += 1
            cg = data.get('convergence_gen')
            if cg is not None:
                gens.append(cg)
        min_accs.append(data.get('individual_min_fitness', 0.0))

    total = len(files)
    rate = 100 * converged / total if total > 0 else 0
    print(f"  {prefix}: {converged}/{total} ({rate:.1f}%)", end='')
    if gens:
        print(f" | median gen {np.median(gens):.0f} [{min(gens)}-{max(gens)}]", end='')
    if min_accs:
        print(f" | avg min_fit {np.mean(min_accs):.4f}", end='')
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S1: Benchmark seed upgrade to n=30')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds per condition (default: 30)')
    parser.add_argument('--conditions', nargs='+', default=None,
                        help='Specific conditions to run (default: all)')
    parser.add_argument('--pop-size', type=int, default=750,
                        help='Population size (default: 750)')
    parser.add_argument('--generations', type=int, default=100,
                        help='Max generations (default: 100)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of existing results only')
    parser.add_argument('--group', type=str, default=None,
                        choices=['1task', '3task', '4task', '5task'],
                        help='Run only a specific task-count group')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter conditions
    if args.conditions:
        conditions = {k: v for k, v in ALL_CONDITIONS.items()
                      if k in args.conditions}
    elif args.group:
        conditions = {k: v for k, v in ALL_CONDITIONS.items()
                      if k.startswith(args.group)}
    else:
        conditions = ALL_CONDITIONS

    if args.summary:
        print("\n=== E-S1 Benchmark Seed Upgrade Summary ===\n")
        for group in ['1task', '3task', '4task', '5task']:
            group_conds = {k: v for k, v in ALL_CONDITIONS.items()
                          if k.startswith(group)}
            if group_conds:
                print(f"\n--- {group.upper()} ---")
                for cond in group_conds:
                    summarize_condition(RESULTS_DIR, cond)

                # Aggregate for this group
                total_files = 0
                total_converged = 0
                for cond in group_conds:
                    files = sorted(RESULTS_DIR.glob(f'{cond}_seed*.json'))
                    for f in files:
                        total_files += 1
                        with open(f) as fh:
                            data = json.load(fh)
                        if data.get('converged', False):
                            total_converged += 1
                if total_files > 0:
                    agg_rate = 100 * total_converged / total_files
                    print(f"  AGGREGATE: {total_converged}/{total_files} ({agg_rate:.1f}%)")
        return

    total_start = time.time()
    total_runs = 0
    total_skipped = 0

    for cond_name, task_list in conditions.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (tasks: {task_list})")
        print(f"  Pop={args.pop_size}, Gens={args.generations}, Seeds={args.seeds}")
        print(f"  Uniform tanh, product aggregation")
        print(f"{'='*60}")

        for seed in range(args.seeds):
            # Skip seed 71 (known OOM trigger)
            if seed == 71:
                print(f"  Skip seed 71 (known OOM)")
                continue

            fname = RESULTS_DIR / f'{cond_name}_seed{seed}.json'
            if result_exists(fname):
                print(f"  Skip existing: {fname.name}")
                total_skipped += 1
                continue

            print(f"\n  Seed {seed}:")
            try:
                result = run_multihead_palette_experiment(
                    task_names=task_list,
                    palette_mode='uniform',  # all tanh
                    blend_mode='fixed',
                    aggregation='product',
                    seed=seed,
                    generations=args.generations,
                    population=args.pop_size,
                    success_threshold=0.90,  # match original benchmark
                    verbose=True,
                )

                # Convert to dict for JSON
                result_dict = asdict(result)
                result_dict['condition'] = cond_name
                result_dict['benchmark_upgrade'] = True

                save_result(result_dict, fname)
                total_runs += 1

                status = (f"gen {result.convergence_gen}"
                          if result.converged else "NOT CONVERGED")
                print(f"  -> {status} (min_fit={result.individual_min_fitness:.4f}, "
                      f"{result.runtime_seconds:.1f}s)")

            except Exception as e:
                print(f"  ERROR on seed {seed}: {e}")
                continue

        print(f"\n--- Summary for {cond_name} ---")
        summarize_condition(RESULTS_DIR, cond_name)

    total_runtime = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total: {total_runs} new runs, {total_skipped} skipped")
    print(f"Runtime: {total_runtime:.0f}s ({total_runtime/3600:.1f} hours)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
