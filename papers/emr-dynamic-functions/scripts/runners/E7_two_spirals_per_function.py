#!/usr/bin/env python3
"""Per-function ablation on Two Spirals: 6 representative functions.

ALIFE 2026 Paper 1 (E7): Tests bidirectional substrate matching. For non-periodic
problems, function choice should be largely irrelevant, all cluster near ~75%.
Confirms that the oscillatory advantage is task-dependent, not universal.

Usage:
    python papers/emr-dynamic-functions/scripts/runners/E7_two_spirals_per_function.py
    python papers/emr-dynamic-functions/scripts/runners/E7_two_spirals_per_function.py --function sin
    python papers/emr-dynamic-functions/scripts/runners/E7_two_spirals_per_function.py --list
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
from classification_problems import TwoSpiralsProblem

from emr_hyperneat.emrhyperneat import (
    EMRHyperNEAT,
    ACTIVATION_LIST,
)


# ============================================================================
# Constants
# ============================================================================

OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "results" / "two_spirals_per_function_n30")
SEEDS = list(range(42, 72))  # 30 seeds
FUNCTIONS_TO_TEST = ['sin', 'tanh', 'gauss', 'burst', 'relu', 'sigmoid']

FUNCTION_CATEGORIES = {
    'sin': 'Oscillatory',
    'tanh': 'Monotonic/Bounded',
    'gauss': 'Radial',
    'burst': 'Oscillatory',
    'relu': 'Monotonic/Unbounded',
    'sigmoid': 'Monotonic/Bounded',
}


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class TrialMetrics:
    """Metrics for a single trial."""
    function_name: str
    function_index: int
    seed: int
    solved: bool
    solved_gen: Optional[int]
    best_fitness: float
    generations_run: int
    elapsed_seconds: float


# ============================================================================
# Trial runner
# ============================================================================


def run_single_trial(
    function_name: str,
    function_index: int,
    seed: int,
    max_generations: int = 100,
    pop_size: int = 500,
    max_depth: int = 3,
    target_fitness: float = 0.95,
) -> TrialMetrics:
    """Run single Two Spirals trial with a specific activation function.

    Args:
        function_name: Name from ACTIVATION_LIST.
        function_index: Index in ACTIVATION_LIST.
        seed: Random seed for reproducibility.
        max_generations: Generation budget.
        pop_size: NEAT population size.
        max_depth: EMR quadtree max depth (3 for Two Spirals).
        target_fitness: Fitness threshold for declaring solved.

    Returns:
        TrialMetrics with convergence information.
    """
    problem = TwoSpiralsProblem()
    start_time = time.time()

    # Substrate coordinates: 3 inputs (2 coords + bias) spread on y=-1.0
    n_inputs = problem.input_shape[0]  # 3
    input_coords = []
    for i in range(n_inputs):
        x = -1.0 + 2.0 * i / max(n_inputs - 1, 1) if n_inputs > 1 else 0.0
        input_coords.append((x, -1.0))

    output_coords = [(0.0, 1.0)]

    algo_config = {
        'algorithm_params': {
            'emrhyperneat': {
                'emr_hyperneat': {
                    'extra_randkey_split': True,  # reproduce HMR (paper) per-seed results in EMR
                    'initial_depth': 0,
                    'max_depth': max_depth,
                    'variance_threshold': 0.03,
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
                    'pop_size': 150,  # repro: HMR pop-bug ran the paper at 150 regardless of nominal; EMR honors config
                    'species_size': 10,
                },
            }
        }
    }

    algo = EMRHyperNEAT()
    neat_config = algo.create_config(algo_config)
    state = algo.initialize(neat_config, problem, seed=seed)

    best_fitness = 0.0
    solved_gen = None

    for gen in range(max_generations):
        state, metrics = algo.run_generation(state, problem)
        best_fitness = max(best_fitness, metrics.best_fitness)

        if metrics.best_fitness >= target_fitness and solved_gen is None:
            solved_gen = gen + 1
            break

    elapsed = time.time() - start_time

    return TrialMetrics(
        function_name=function_name,
        function_index=function_index,
        seed=seed,
        solved=solved_gen is not None,
        solved_gen=solved_gen,
        best_fitness=float(best_fitness),
        generations_run=gen + 1,
        elapsed_seconds=elapsed,
    )


# ============================================================================
# Resume logic
# ============================================================================


def load_existing_results(output_dir: str, function_name: str) -> Dict[str, Any]:
    """Load existing results for a function, if any.

    Returns:
        Dict with 'results' list and 'summary', or empty dict if no file.
    """
    filepath = os.path.join(output_dir, f"{function_name}.json")
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def get_completed_seeds(existing: Dict[str, Any]) -> set:
    """Extract completed seed numbers from existing results."""
    if not existing or 'results' not in existing:
        return set()
    return {r['seed'] for r in existing['results']}


# ============================================================================
# Per-function experiment
# ============================================================================


def run_function_experiment(
    function_name: str,
    function_index: int,
    seeds: List[int],
    output_dir: str,
    max_generations: int = 100,
    pop_size: int = 500,
    max_depth: int = 3,
    target_fitness: float = 0.95,
) -> Dict[str, Any]:
    """Run experiment for a single activation function with skip-resume.

    Saves incrementally after each seed. Skips seeds already completed.

    Args:
        function_name: Activation function name.
        function_index: Index in ACTIVATION_LIST.
        seeds: List of seed values.
        output_dir: Directory for JSON output.
        max_generations: Generation budget per seed.
        pop_size: NEAT population size.
        max_depth: EMR max depth.
        target_fitness: Convergence threshold.

    Returns:
        Combined results dict for this function.
    """
    print(f"\n{'='*60}")
    print(f"FUNCTION: {function_name} (index {function_index})")
    print(f"{'='*60}")

    # Load existing results for resume
    existing = load_existing_results(output_dir, function_name)
    completed_seeds = get_completed_seeds(existing)
    results = list(existing.get('results', []))

    for seed in seeds:
        if seed in completed_seeds:
            print(f"  Seed {seed}: SKIP (already completed)")
            continue

        print(f"  Seed {seed}: ", end='', flush=True)

        trial = run_single_trial(
            function_name, function_index, seed,
            max_generations, pop_size, max_depth, target_fitness,
        )
        results.append(asdict(trial))

        if trial.solved:
            print(f"SOLVED @ gen {trial.solved_gen} ({trial.elapsed_seconds:.1f}s)")
        else:
            print(f"FAIL (fit={trial.best_fitness:.4f}, {trial.elapsed_seconds:.1f}s)")

        # Save incrementally after each seed
        func_data = _build_function_result(function_name, function_index, results)
        filepath = os.path.join(output_dir, f"{function_name}.json")
        with open(filepath, 'w') as f:
            json.dump(func_data, f, indent=2, default=str)

    # Final summary
    func_data = _build_function_result(function_name, function_index, results)
    s = func_data['summary']
    print(f"  -> {function_name}: {s['solve_rate']*100:.0f}% solve, "
          f"avg fit={s['avg_best_fitness']:.4f}")

    return func_data


def _build_function_result(
    function_name: str,
    function_index: int,
    results: List[Dict],
) -> Dict[str, Any]:
    """Build the complete result dict with summary statistics for a function."""
    solve_count = sum(1 for r in results if r['solved'])
    solve_rate = solve_count / len(results) if results else 0.0

    solved_gens = [r['solved_gen'] for r in results if r['solved']]
    avg_gen = float(np.mean(solved_gens)) if solved_gens else None
    std_gen = float(np.std(solved_gens)) if len(solved_gens) > 1 else 0.0
    median_gen = float(np.median(solved_gens)) if solved_gens else None

    best_fitnesses = [r['best_fitness'] for r in results]

    return {
        'function_name': function_name,
        'function_index': function_index,
        'results': results,
        'summary': {
            'solve_rate': solve_rate,
            'solve_count': solve_count,
            'total_runs': len(results),
            'avg_gen_to_solve': avg_gen,
            'std_gen_to_solve': std_gen if avg_gen is not None else None,
            'median_gen_to_solve': median_gen,
            'avg_best_fitness': float(np.mean(best_fitnesses)) if best_fitnesses else 0.0,
            'std_best_fitness': float(np.std(best_fitnesses)) if best_fitnesses else 0.0,
        },
    }


# ============================================================================
# Status listing
# ============================================================================


def print_status(output_dir: str, functions: List[str], seeds: List[int]):
    """Print current completion status for all functions."""
    print(f"\n{'='*70}")
    print(f"STATUS: {output_dir}")
    print(f"{'='*70}")
    print(f"{'Function':<12} {'Completed':<14} {'Remaining':<12} {'Solve%':<10}")
    print("-" * 48)

    total_completed = 0
    total_expected = len(functions) * len(seeds)

    for func_name in functions:
        existing = load_existing_results(output_dir, func_name)
        completed = get_completed_seeds(existing)
        remaining = len(seeds) - len(completed)
        total_completed += len(completed)

        if existing and 'summary' in existing:
            solve_pct = f"{existing['summary']['solve_rate']*100:.0f}%"
        else:
            solve_pct = "N/A"

        status = "DONE" if remaining == 0 else f"{len(completed)}/{len(seeds)}"
        print(f"{func_name:<12} {status:<14} {remaining:<12} {solve_pct:<10}")

    print(f"\nTotal: {total_completed}/{total_expected} "
          f"({total_completed/total_expected*100:.0f}% complete)")


# ============================================================================
# Summary table
# ============================================================================


def print_summary_table(all_results: Dict[str, Dict]):
    """Print final summary table sorted by solve rate."""
    print(f"\n{'='*80}")
    print("E7: PER-FUNCTION ON TWO SPIRALS (N=30, pop=500, depth=3, 100 gen)")
    print(f"{'='*80}")
    print(f"{'Function':<12} {'Category':<22} {'Solve%':<10} {'Med Gen':<10} "
          f"{'Avg Gen':<10} {'Avg Fit':<10}")
    print("-" * 74)

    sorted_funcs = sorted(
        all_results.items(),
        key=lambda x: (-x[1]['summary']['solve_rate'],
                       x[1]['summary']['avg_gen_to_solve'] or 999)
    )

    for func_name, result in sorted_funcs:
        s = result['summary']
        category = FUNCTION_CATEGORIES.get(func_name, 'Unknown')
        avg_gen_str = f"{s['avg_gen_to_solve']:.1f}" if s['avg_gen_to_solve'] else "Never"
        med_gen_str = (f"{s['median_gen_to_solve']:.0f}"
                       if s.get('median_gen_to_solve') else "Never")
        print(f"{func_name:<12} {category:<22} {s['solve_rate']*100:<10.0f} "
              f"{med_gen_str:<10} {avg_gen_str:<10} {s['avg_best_fitness']:<10.4f}")

    # Category summary
    osc_funcs = [n for n in FUNCTIONS_TO_TEST
                 if FUNCTION_CATEGORIES.get(n, '') in ('Oscillatory',)]
    mono_funcs = [n for n in FUNCTIONS_TO_TEST
                  if FUNCTION_CATEGORIES.get(n, '') in
                  ('Monotonic/Bounded', 'Monotonic/Unbounded', 'Radial')]

    osc_rates = [all_results[f]['summary']['solve_rate']
                 for f in osc_funcs if f in all_results]
    mono_rates = [all_results[f]['summary']['solve_rate']
                  for f in mono_funcs if f in all_results]

    print(f"\n--- CATEGORY AVERAGES ---")
    if osc_rates:
        print(f"Oscillatory ({', '.join(osc_funcs)}): "
              f"avg {np.mean(osc_rates)*100:.1f}% solve rate")
    if mono_rates:
        print(f"Non-oscillatory ({', '.join(mono_funcs)}): "
              f"avg {np.mean(mono_rates)*100:.1f}% solve rate")

    if osc_rates and mono_rates:
        diff = np.mean(osc_rates) - np.mean(mono_rates)
        print(f"Difference: {diff*100:+.1f}pp "
              f"({'oscillatory advantage' if diff > 0 else 'no oscillatory advantage'})")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='E7: Per-Function on Two Spirals (ALIFE 2026 Paper 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 6 functions (default)
  python %(prog)s

  # Run a single function
  python %(prog)s --function sin

  # Check status
  python %(prog)s --list
        """,
    )
    parser.add_argument('--function', type=str, default=None,
                        help='Test only a specific function name')
    parser.add_argument('--list', action='store_true',
                        help='Show completion status and exit')
    parser.add_argument('--max-gens', type=int, default=100,
                        help='Maximum generations (default: 100)')
    parser.add_argument('--pop-size', type=int, default=500,
                        help='Population size (default: 500)')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='EMR max depth (default: 3 for Two Spirals)')
    parser.add_argument('--target-fitness', type=float, default=0.95,
                        help='Fitness threshold for solved (default: 0.95)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    output_dir = args.output_dir
    seeds = SEEDS

    # Determine which functions to run
    if args.function:
        if args.function not in ACTIVATION_LIST:
            print(f"ERROR: Unknown function '{args.function}'. "
                  f"Available: {ACTIVATION_LIST}")
            sys.exit(1)
        if args.function not in FUNCTIONS_TO_TEST:
            print(f"WARNING: '{args.function}' is not in the standard E7 set "
                  f"{FUNCTIONS_TO_TEST}, running anyway.")
        functions = [args.function]
    else:
        functions = FUNCTIONS_TO_TEST

    # Status mode
    if args.list:
        print_status(output_dir, functions, seeds)
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Header
    n_samples = len(TwoSpiralsProblem().get_data())
    total_runs = len(functions) * len(seeds)

    print("=" * 70)
    print("ALIFE 2026 Paper 1 (E7): Per-Function on Two Spirals")
    print("=" * 70)
    print(f"Functions: {functions}")
    print(f"Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
    print(f"Max generations: {args.max_gens}")
    print(f"Population: {args.pop_size}")
    print(f"Max depth: {args.max_depth}")
    print(f"Target fitness: {args.target_fitness}")
    print(f"Problem: Two Spirals ({n_samples} samples)")
    print(f"Total runs: {total_runs}")
    print(f"Output: {output_dir}")
    print()

    start_time = time.time()
    all_results = {}

    for func_name in functions:
        func_index = ACTIVATION_LIST.index(func_name)

        result = run_function_experiment(
            func_name, func_index, seeds, output_dir,
            args.max_gens, args.pop_size, args.max_depth, args.target_fitness,
        )
        all_results[func_name] = result

        # GC between functions to avoid OOM
        gc.collect()

    # Print summary table
    print_summary_table(all_results)

    # Save combined results
    total_time = time.time() - start_time
    combined = {
        'metadata': {
            'experiment': 'two_spirals_per_function',
            'experiment_id': 'E7',
            'purpose': 'ALIFE 2026 Paper 1 — per-function characterization on Two Spirals',
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'seeds': seeds,
            'functions': functions,
            'max_generations': args.max_gens,
            'pop_size': 150,  # repro: HMR pop-bug ran the paper at 150 regardless of nominal; EMR honors config
            'max_depth': args.max_depth,
            'target_fitness': args.target_fitness,
            'problem': 'two_spirals',
            'n_samples': n_samples,
            'activation_mode': 'global',
            'total_runs': sum(r['summary']['total_runs'] for r in all_results.values()),
            'total_runtime_seconds': total_time,
            'function_categories': FUNCTION_CATEGORIES,
        },
        'results': all_results,
    }

    combined_path = os.path.join(output_dir, 'combined_results.json')
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
