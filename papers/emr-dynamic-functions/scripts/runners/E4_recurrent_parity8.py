#!/usr/bin/env python3
"""Recurrent ablation on Parity-8: 9 monotonic functions + sin under full recurrence.

ALIFE 2026 Paper 1 (E4): Tests whether recurrence collapse (monotonic functions
matching oscillatory under recurrence) is robust to problem scale. Parity-8 is
much harder than Parity-4 -- recurrent monotonic functions needed 64-115 gens
on Parity-4 and may fail entirely on Parity-8.

Usage:
    python papers/emr-dynamic-functions/scripts/runners/E4_recurrent_parity8.py
    python papers/emr-dynamic-functions/scripts/runners/E4_recurrent_parity8.py --function tanh
    python papers/emr-dynamic-functions/scripts/runners/E4_recurrent_parity8.py --list
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification_problems import ParityProblem

from emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended_dynamic_functions_full import (
    HMRHyperNEATUnifiedExtendedDynamicFunctions as UnifiedAlgo,
    RECURRENCE_PRESETS,
    ACTIVATION_LIST,
)

# ============================================================================
# Constants
# ============================================================================

OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "results" / "recurrent_parity8_n30")

SEEDS = list(range(42, 72))  # 30 seeds

RECURRENCE_PRESET = 'full_recurrent'

# 9 monotonic + sin (control) = 10 total
FUNCTIONS_TO_TEST = [
    'sin',       # control (oscillatory)
    'tanh',      # monotonic/bounded
    'sigmoid',   # monotonic/bounded
    'relu',      # monotonic/unbounded
    'identity',  # linear
    'lelu',      # monotonic/unbounded
    'softplus',  # monotonic/unbounded
    'fs_fast',   # monotonic/unbounded
    'lts_low',   # monotonic/bounded
    'integrate', # monotonic/bounded
]

FUNCTION_CATEGORIES = {
    'sin':       'Oscillatory',
    'tanh':      'Monotonic/Bounded',
    'sigmoid':   'Monotonic/Bounded',
    'relu':      'Monotonic/Unbounded',
    'identity':  'Linear',
    'lelu':      'Monotonic/Unbounded',
    'softplus':  'Monotonic/Unbounded',
    'fs_fast':   'Monotonic/Unbounded',
    'lts_low':   'Monotonic/Bounded',
    'integrate': 'Monotonic/Bounded',
}

# Experiment parameters
POP_SIZE = 500
MAX_GENERATIONS = 300
MAX_DEPTH = 2
TARGET_FITNESS = 0.95
N_BITS = 8


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class TrialResult:
    """Metrics for a single Parity-8 recurrent ablation trial."""
    function_name: str
    function_index: int
    category: str
    recurrence_preset: str
    seed: int
    solved: bool
    solved_gen: Optional[int]
    best_fitness: float
    generations_run: int
    elapsed_seconds: float
    pop_size: int
    max_depth: int
    target_fitness: float
    n_bits: int


# ============================================================================
# Core trial runner
# ============================================================================


def run_trial(
    function_name: str,
    function_index: int,
    seed: int,
) -> TrialResult:
    """Run a single Parity-8 trial with given activation under full recurrence."""

    problem = ParityProblem(n_bits=N_BITS)

    # Parity-8 with bias: input_shape=(9,), output_shape=(1,)
    n_inputs = problem.input_shape[0]  # 9
    n_outputs = problem.output_shape[0]  # 1

    # Spread inputs evenly from -1.0 to 1.0 on y=-1.0
    input_coords = []
    for i in range(n_inputs):
        x = -1.0 + 2.0 * i / max(n_inputs - 1, 1)
        input_coords.append((x, -1.0))

    output_coords = [(0.0, 1.0)]

    algo_config = {
        'algorithm_params': {
            'hmrhyperneat': {
                'hmr_hyperneat': {
                    'initial_depth': 0,
                    'max_depth': MAX_DEPTH,
                    'variance_threshold': 0.03,
                    'extra_randkey_split': True,
                    'recurrence': {
                        'preset': RECURRENCE_PRESET,
                    },
                    'dynamic_functions': {
                        'mode': 'global',
                        'hidden_activation': function_name,
                        'palette': [function_index],
                        'palette_evolution': {'enabled': False},
                    },
                },
                'substrate': {
                    'input_coords': input_coords,
                    'output_coords': output_coords,
                },
                'neat': {
                    'pop_size': POP_SIZE,
                    'species_size': 10,
                },
            }
        }
    }

    start_time = time.time()

    algo = UnifiedAlgo()
    neat_config = algo.create_config(algo_config)
    state = algo.initialize(neat_config, problem, seed=seed)

    best_fitness = 0.0
    solved_gen = None

    for gen in range(MAX_GENERATIONS):
        state, metrics = algo.run_generation(state, problem)
        best_fitness = max(best_fitness, metrics.best_fitness)

        if metrics.best_fitness >= TARGET_FITNESS and solved_gen is None:
            solved_gen = gen + 1
            break

    elapsed = time.time() - start_time

    return TrialResult(
        function_name=function_name,
        function_index=function_index,
        category=FUNCTION_CATEGORIES.get(function_name, 'Unknown'),
        recurrence_preset=RECURRENCE_PRESET,
        seed=seed,
        solved=solved_gen is not None,
        solved_gen=solved_gen,
        best_fitness=float(best_fitness),
        generations_run=gen + 1,
        elapsed_seconds=elapsed,
        pop_size=POP_SIZE,
        max_depth=MAX_DEPTH,
        target_fitness=TARGET_FITNESS,
        n_bits=N_BITS,
    )


# ============================================================================
# File I/O and resume logic
# ============================================================================


def result_filename(function_name: str, seed: int) -> str:
    """Compute per-seed result filename."""
    return f"{function_name}_parity8_seed{seed}.json"


def combined_filename(function_name: str) -> str:
    """Compute per-function combined result filename."""
    return f"{function_name}_parity8.json"


def load_completed_seeds(function_name: str) -> set:
    """Return set of seeds already completed for a function."""
    completed = set()
    for seed in SEEDS:
        filepath = os.path.join(OUTPUT_DIR, result_filename(function_name, seed))
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                if 'seed' in data and 'solved' in data:
                    completed.add(data['seed'])
            except (json.JSONDecodeError, KeyError):
                pass  # corrupted file: not counted as completed
    return completed


def save_trial(result: TrialResult) -> None:
    """Save a single trial result to its per-seed JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, result_filename(result.function_name, result.seed))
    with open(filepath, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)


def load_trial(function_name: str, seed: int) -> Optional[dict]:
    """Load a previously saved trial result."""
    filepath = os.path.join(OUTPUT_DIR, result_filename(function_name, seed))
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def save_combined(function_name: str, results: List[dict], summary: dict) -> None:
    """Save combined per-function results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, combined_filename(function_name))
    combined = {
        'function_name': function_name,
        'function_index': ACTIVATION_LIST.index(function_name),
        'category': FUNCTION_CATEGORIES.get(function_name, 'Unknown'),
        'recurrence_preset': RECURRENCE_PRESET,
        'n_bits': N_BITS,
        'pop_size': POP_SIZE,
        'max_generations': MAX_GENERATIONS,
        'max_depth': MAX_DEPTH,
        'target_fitness': TARGET_FITNESS,
        'seeds': SEEDS,
        'summary': summary,
        'results': results,
    }
    with open(filepath, 'w') as f:
        json.dump(combined, f, indent=2, default=str)


# ============================================================================
# Status display
# ============================================================================


def show_status() -> None:
    """Show completion status for all functions."""
    print("=" * 70)
    print("E4: RECURRENT PARITY-8 ABLATION STATUS")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Seeds: {len(SEEDS)} ({SEEDS[0]}-{SEEDS[-1]})")
    print(f"Functions: {len(FUNCTIONS_TO_TEST)}")
    print(f"Total expected: {len(FUNCTIONS_TO_TEST) * len(SEEDS)} runs")
    print()
    print(f"{'Function':<14} {'Category':<22} {'Done':<8} {'Solved':<8} {'Rate':<8}")
    print("-" * 60)

    total_done = 0
    total_solved = 0

    for func_name in FUNCTIONS_TO_TEST:
        completed = load_completed_seeds(func_name)
        done = len(completed)
        total_done += done

        # Count solved from loaded results
        solved = 0
        for seed in completed:
            result = load_trial(func_name, seed)
            if result and result.get('solved', False):
                solved += 1
        total_solved += solved

        rate = f"{solved}/{done}" if done > 0 else "N/A"
        category = FUNCTION_CATEGORIES.get(func_name, 'Unknown')
        status = "DONE" if done == len(SEEDS) else f"{done}/{len(SEEDS)}"
        print(f"{func_name:<14} {category:<22} {status:<8} {rate:<8}")

    print("-" * 60)
    pct = total_done / (len(FUNCTIONS_TO_TEST) * len(SEEDS)) * 100
    print(f"{'TOTAL':<14} {'':22} {total_done}/{len(FUNCTIONS_TO_TEST) * len(SEEDS)} ({pct:.0f}%)")


# ============================================================================
# Main experiment runner
# ============================================================================


def run_function(function_name: str) -> dict:
    """Run all seeds for a single function. Returns summary dict."""

    if function_name not in ACTIVATION_LIST:
        print(f"ERROR: Unknown function '{function_name}'. Available: {ACTIVATION_LIST}")
        sys.exit(1)

    function_index = ACTIVATION_LIST.index(function_name)
    category = FUNCTION_CATEGORIES.get(function_name, 'Unknown')
    completed_seeds = load_completed_seeds(function_name)

    print(f"\n{'='*60}")
    print(f"FUNCTION: {function_name} (index {function_index}, {category})")
    print(f"{'='*60}")
    print(f"  Recurrence: {RECURRENCE_PRESET}")
    print(f"  Problem: Parity-{N_BITS} (256 samples, 9 inputs + bias)")
    print(f"  Pop: {POP_SIZE}, Max Gen: {MAX_GENERATIONS}, Depth: {MAX_DEPTH}, Target: {TARGET_FITNESS}")

    if completed_seeds:
        print(f"  Resuming: {len(completed_seeds)}/{len(SEEDS)} seeds already complete")

    all_results = []

    for seed in SEEDS:
        if seed in completed_seeds:
            existing = load_trial(function_name, seed)
            if existing:
                all_results.append(existing)
                status = "SOLVED" if existing['solved'] else f"FAIL (fit={existing['best_fitness']:.4f})"
                print(f"  Seed {seed}: SKIP ({status})")
                continue

        print(f"  Seed {seed}: ", end='', flush=True)

        trial = run_trial(function_name, function_index, seed)
        result_dict = asdict(trial)
        all_results.append(result_dict)

        # Save immediately (incremental)
        save_trial(trial)

        if trial.solved:
            print(f"SOLVED @ gen {trial.solved_gen} ({trial.elapsed_seconds:.1f}s)")
        else:
            print(f"FAIL (fit={trial.best_fitness:.4f}, {trial.generations_run} gens, "
                  f"{trial.elapsed_seconds:.1f}s)")

        gc.collect()

    # Compute summary
    solve_count = sum(1 for r in all_results if r['solved'])
    solve_rate = solve_count / len(all_results) if all_results else 0.0
    solved_gens = [r['solved_gen'] for r in all_results if r['solved']]
    best_fitnesses = [r['best_fitness'] for r in all_results]
    elapsed_times = [r['elapsed_seconds'] for r in all_results]

    summary = {
        'function_name': function_name,
        'function_index': function_index,
        'category': category,
        'solve_rate': solve_rate,
        'solve_count': solve_count,
        'total_runs': len(all_results),
        'avg_gen_to_solve': float(np.mean(solved_gens)) if solved_gens else None,
        'median_gen_to_solve': float(np.median(solved_gens)) if solved_gens else None,
        'std_gen_to_solve': float(np.std(solved_gens)) if len(solved_gens) > 1 else None,
        'min_gen_to_solve': int(min(solved_gens)) if solved_gens else None,
        'max_gen_to_solve': int(max(solved_gens)) if solved_gens else None,
        'avg_best_fitness': float(np.mean(best_fitnesses)),
        'median_best_fitness': float(np.median(best_fitnesses)),
        'avg_elapsed_seconds': float(np.mean(elapsed_times)),
        'total_elapsed_seconds': float(np.sum(elapsed_times)),
    }

    # Save combined result for this function
    save_combined(function_name, all_results, summary)

    print(f"\n  => {function_name}: {solve_rate*100:.0f}% ({solve_count}/{len(all_results)}) | "
          f"Med Gen: {summary['median_gen_to_solve'] if summary['median_gen_to_solve'] else 'Never'} | "
          f"Avg Fit: {summary['avg_best_fitness']:.4f}")

    return summary


def print_summary_table(summaries: Dict[str, dict]) -> None:
    """Print formatted summary table across all functions."""
    print("\n" + "=" * 90)
    print("E4: RECURRENT PARITY-8 ABLATION SUMMARY")
    print("=" * 90)
    print(f"{'Function':<14} {'Category':<22} {'Solve%':<10} {'Med Gen':<10} "
          f"{'Avg Fit':<10} {'Avg Time':<10}")
    print("-" * 76)

    # Sort: highest solve rate first, then lowest median gen
    sorted_funcs = sorted(
        summaries.items(),
        key=lambda x: (-x[1]['solve_rate'],
                       x[1]['median_gen_to_solve'] or 999)
    )

    for func_name, s in sorted_funcs:
        category = s['category']
        solve_pct = f"{s['solve_rate']*100:.0f}%"
        med_gen = f"{s['median_gen_to_solve']:.0f}" if s['median_gen_to_solve'] else "Never"
        avg_fit = f"{s['avg_best_fitness']:.4f}"
        avg_time = f"{s['avg_elapsed_seconds']:.0f}s"
        print(f"{func_name:<14} {category:<22} {solve_pct:<10} {med_gen:<10} "
              f"{avg_fit:<10} {avg_time:<10}")

    # Category breakdown
    osc_rates = [s['solve_rate'] for n, s in summaries.items() if s['category'] == 'Oscillatory']
    mono_rates = [s['solve_rate'] for n, s in summaries.items()
                  if 'Monotonic' in s['category'] or s['category'] == 'Linear']

    print(f"\n--- CATEGORY SUMMARY ---")
    if osc_rates:
        print(f"Oscillatory ({len(osc_rates)} func): avg {np.mean(osc_rates)*100:.0f}% solve rate")
    if mono_rates:
        print(f"Monotonic/Linear ({len(mono_rates)} funcs): avg {np.mean(mono_rates)*100:.0f}% solve rate")

    # Key comparison
    sin_summary = summaries.get('sin')
    if sin_summary:
        sin_rate = sin_summary['solve_rate']
        any_mono_solves = any(s['solve_rate'] > 0 for n, s in summaries.items()
                             if s['category'] != 'Oscillatory')
        if any_mono_solves:
            print(f"\nRecurrence collapse: PRESENT (monotonic functions solve Parity-8 under recurrence)")
        else:
            print(f"\nRecurrence collapse: ABSENT (monotonic 0% vs sin {sin_rate*100:.0f}% - "
                  f"oscillatory barrier holds at Parity-8)")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='E4: Recurrent Ablation on Parity-8 (ALIFE 2026 Paper 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 10 functions (300 total runs)
  python papers/emr-dynamic-functions/scripts/runners/E4_recurrent_parity8.py

  # Run a single function (safe for OOM recovery)
  python papers/emr-dynamic-functions/scripts/runners/E4_recurrent_parity8.py --function tanh

  # Check completion status
  python papers/emr-dynamic-functions/scripts/runners/E4_recurrent_parity8.py --list
        """,
    )

    parser.add_argument('--function', type=str, default=None,
                        help='Run a single function only (e.g. tanh, sin). '
                             'Recommended: run ONE function per process to avoid OOM.')
    parser.add_argument('--list', action='store_true',
                        help='Show completion status and exit.')

    args = parser.parse_args()

    # --list: show status and exit
    if args.list:
        show_status()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("E4: RECURRENT ABLATION ON PARITY-8")
    print("=" * 70)
    print(f"Problem: Parity-{N_BITS} ({2**N_BITS} samples, {N_BITS}+1 inputs with bias)")
    print(f"Recurrence: {RECURRENCE_PRESET}")
    print(f"Pop: {POP_SIZE}, Max Gen: {MAX_GENERATIONS}, Depth: {MAX_DEPTH}, Target: {TARGET_FITNESS}")
    print(f"Seeds: {len(SEEDS)} ({SEEDS[0]}-{SEEDS[-1]})")
    print(f"Output: {OUTPUT_DIR}")

    start_time = time.time()

    # Determine which functions to run
    if args.function:
        if args.function not in FUNCTIONS_TO_TEST:
            if args.function in ACTIVATION_LIST:
                print(f"WARNING: '{args.function}' is not in the planned 10 functions but exists "
                      f"in ACTIVATION_LIST. Running anyway.")
                functions = [args.function]
            else:
                print(f"ERROR: Unknown function '{args.function}'.")
                print(f"Available: {FUNCTIONS_TO_TEST}")
                sys.exit(1)
        else:
            functions = [args.function]
    else:
        functions = FUNCTIONS_TO_TEST

    total_runs = len(functions) * len(SEEDS)
    print(f"Functions to run: {functions}")
    print(f"Total runs: {total_runs}")
    print()

    # Run
    all_summaries = {}
    for func_name in functions:
        summary = run_function(func_name)
        all_summaries[func_name] = summary
        gc.collect()

    # Print summary table
    print_summary_table(all_summaries)

    # Save master summary
    total_time = time.time() - start_time
    master = {
        'metadata': {
            'experiment': 'E4_recurrent_parity8_ablation',
            'paper': 'ALIFE 2026 Paper 1 (Per-Node Activation Function Evolution)',
            'recurrence_preset': RECURRENCE_PRESET,
            'problem': f'parity_{N_BITS}',
            'n_bits': N_BITS,
            'n_samples': 2 ** N_BITS,
            'pop_size': POP_SIZE,
            'max_generations': MAX_GENERATIONS,
            'max_depth': MAX_DEPTH,
            'target_fitness': TARGET_FITNESS,
            'seeds': SEEDS,
            'functions_tested': functions,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'total_runtime_seconds': total_time,
        },
        'summaries': all_summaries,
    }

    master_path = os.path.join(OUTPUT_DIR, 'master_summary.json')
    with open(master_path, 'w') as f:
        json.dump(master, f, indent=2, default=str)

    print(f"\nTotal runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
