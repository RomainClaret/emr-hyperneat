#!/usr/bin/env python3
"""Per-function ablation on XOR (Parity-2): each of 18 activation functions individually.

ALIFE 2026 Paper 1 (E2): Tests whether the three-tier characterization from Parity-4
is general or task-specific. XOR is easier -- more functions should succeed.

Usage:
    python papers/emr-dynamic-functions/scripts/runners/E2_xor_per_function.py
    python papers/emr-dynamic-functions/scripts/runners/E2_xor_per_function.py --function sin
    python papers/emr-dynamic-functions/scripts/runners/E2_xor_per_function.py --list
"""

import argparse
import json
import os
from pathlib import Path
import time
import sys
import gc
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification_problems import ParityProblem

from emr_hyperneat.emrhyperneat import (
    EMRHyperNEAT,
    ACTIVATION_LIST,
    ACTIVATION_FUNCTIONS,
)


OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "results" / "xor_per_function_n30")
SEEDS = list(range(42, 72))  # 30 seeds

# Function categorization for the paper
FUNCTION_CATEGORIES = {
    'tanh': 'Monotonic/Bounded',
    'sigmoid': 'Monotonic/Bounded',
    'relu': 'Monotonic/Unbounded',
    'identity': 'Linear',
    'sin': 'Oscillatory',
    'gauss': 'Radial',
    'lelu': 'Monotonic/Unbounded',
    'softplus': 'Monotonic/Unbounded',
    'rs_adapt': 'Adaptive',
    'fs_fast': 'Monotonic/Unbounded',
    'lts_low': 'Monotonic/Bounded',
    'burst': 'Oscillatory',
    'resonator': 'Oscillatory',
    'osc_adapt': 'Oscillatory',
    'gain_mod': 'Adaptive',
    'receptive': 'Quasi-Oscillatory',
    'band_pass': 'Adaptive',
    'integrate': 'Monotonic/Bounded',
}


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


def run_single_trial(
    function_name: str,
    function_index: int,
    seed: int,
    max_generations: int = 300,
    pop_size: int = 500,
    max_depth: int = 2,
    target_fitness: float = 0.95,
) -> TrialMetrics:
    """Run single XOR (Parity-2) trial with a specific activation function."""

    problem = ParityProblem(n_bits=2)
    start_time = time.time()

    n_inputs = problem.input_shape[0]  # 3: 2 bits + bias

    input_coords = []
    for i in range(n_inputs):
        x = -1.0 + 2.0 * i / max(n_inputs - 1, 1) if n_inputs > 1 else 0.0
        input_coords.append((x, -1.0))

    output_coords = [(0.0, 1.0)]

    # Use global activation mode with this specific function
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


def load_existing_results(filepath: str) -> Dict[str, Any]:
    """Load existing results file for resume support.

    Returns:
        Dict with 'results' list and 'completed_seeds' set.
    """
    results = []
    completed_seeds = set()

    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
        if 'results' in existing:
            results = existing['results']
            completed_seeds = {r['seed'] for r in results}

    return {'results': results, 'completed_seeds': completed_seeds}


def save_function_results(
    filepath: str,
    function_name: str,
    function_index: int,
    results: List[Dict],
) -> None:
    """Save function results incrementally."""
    solve_count = sum(1 for r in results if r['solved'])
    total = len(results)
    solve_rate = solve_count / total if total > 0 else 0.0

    solved_gens = [r['solved_gen'] for r in results if r['solved']]
    avg_gen = float(np.mean(solved_gens)) if solved_gens else None
    std_gen = float(np.std(solved_gens)) if len(solved_gens) > 1 else 0.0

    best_fitnesses = [r['best_fitness'] for r in results]

    output = {
        'function_name': function_name,
        'function_index': function_index,
        'results': results,
        'summary': {
            'solve_rate': solve_rate,
            'solve_count': solve_count,
            'total_runs': total,
            'avg_gen_to_solve': avg_gen,
            'std_gen_to_solve': float(std_gen) if avg_gen is not None else None,
            'avg_best_fitness': float(np.mean(best_fitnesses)) if best_fitnesses else None,
            'std_best_fitness': float(np.std(best_fitnesses)) if len(best_fitnesses) > 1 else 0.0,
        },
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def run_function_experiment(
    function_name: str,
    function_index: int,
    seeds: List[int],
    output_dir: str,
    max_generations: int = 300,
    pop_size: int = 500,
    max_depth: int = 2,
) -> Dict[str, Any]:
    """Run experiment for a single activation function with per-seed resume."""

    filepath = os.path.join(output_dir, f"{function_name}.json")

    # Load existing results for resume
    existing = load_existing_results(filepath)
    results = existing['results']
    completed_seeds = existing['completed_seeds']

    if completed_seeds:
        remaining = [s for s in seeds if s not in completed_seeds]
        print(f"\n{'='*60}")
        print(f"FUNCTION: {function_name} (index {function_index})")
        print(f"  Resuming: {len(completed_seeds)}/{len(seeds)} seeds done, "
              f"{len(remaining)} remaining")
        print(f"{'='*60}")
        if not remaining:
            print(f"  All seeds complete. Skipping.")
            # Recompute summary from existing results
            save_function_results(filepath, function_name, function_index, results)
            return _build_summary(function_name, function_index, results)
    else:
        remaining = seeds
        print(f"\n{'='*60}")
        print(f"FUNCTION: {function_name} (index {function_index})")
        print(f"{'='*60}")

    for seed in remaining:
        print(f"  Seed {seed}: ", end='', flush=True)

        trial = run_single_trial(
            function_name, function_index, seed,
            max_generations, pop_size, max_depth,
        )

        results.append(asdict(trial))

        if trial.solved:
            print(f"SOLVED @ gen {trial.solved_gen} ({trial.elapsed_seconds:.1f}s)")
        else:
            print(f"FAIL (fit={trial.best_fitness:.4f}, {trial.elapsed_seconds:.1f}s)")

        # Save after EVERY seed for safe resume
        save_function_results(filepath, function_name, function_index, results)

    summary = _build_summary(function_name, function_index, results)

    solve_rate = summary['summary']['solve_rate']
    avg_fit = summary['summary']['avg_best_fitness']
    print(f"  -> {function_name}: {solve_rate*100:.0f}% solve, avg fit={avg_fit:.4f}")

    return summary


def _build_summary(
    function_name: str,
    function_index: int,
    results: List[Dict],
) -> Dict[str, Any]:
    """Build summary dict from results list."""
    solve_count = sum(1 for r in results if r['solved'])
    total = len(results)
    solve_rate = solve_count / total if total > 0 else 0.0

    solved_gens = [r['solved_gen'] for r in results if r['solved']]
    avg_gen = float(np.mean(solved_gens)) if solved_gens else None
    std_gen = float(np.std(solved_gens)) if len(solved_gens) > 1 else 0.0

    best_fitnesses = [r['best_fitness'] for r in results]

    return {
        'function_name': function_name,
        'function_index': function_index,
        'results': results,
        'summary': {
            'solve_rate': solve_rate,
            'solve_count': solve_count,
            'total_runs': total,
            'avg_gen_to_solve': avg_gen,
            'std_gen_to_solve': float(std_gen) if avg_gen is not None else None,
            'avg_best_fitness': float(np.mean(best_fitnesses)) if best_fitnesses else None,
            'std_best_fitness': float(np.std(best_fitnesses)) if len(best_fitnesses) > 1 else 0.0,
        },
    }


def show_status(output_dir: str, seeds: List[int]) -> None:
    """Show current completion status for all functions."""
    print(f"\n{'='*80}")
    print(f"STATUS: XOR Per-Function Ablation (N={len(seeds)})")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    print(f"{'Function':<14} {'Category':<22} {'Done':<8} {'Solve%':<10} {'Avg Gen':<10}")
    print("-" * 64)

    total_done = 0
    total_expected = 0

    for i, func_name in enumerate(ACTIVATION_LIST):
        filepath = os.path.join(output_dir, f"{func_name}.json")
        category = FUNCTION_CATEGORIES.get(func_name, 'Unknown')
        total_expected += len(seeds)

        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
            n_done = len(data.get('results', []))
            total_done += n_done
            summary = data.get('summary', {})
            solve_rate = summary.get('solve_rate', 0)
            avg_gen = summary.get('avg_gen_to_solve')
            avg_gen_str = f"{avg_gen:.1f}" if avg_gen is not None else "Never"
            status = f"{n_done}/{len(seeds)}"
            print(f"{func_name:<14} {category:<22} {status:<8} {solve_rate*100:<10.0f} {avg_gen_str:<10}")
        else:
            print(f"{func_name:<14} {category:<22} {'0/' + str(len(seeds)):<8} {'--':<10} {'--':<10}")

    print(f"\nTotal: {total_done}/{total_expected} runs "
          f"({total_done/total_expected*100:.0f}% complete)")


def main():
    parser = argparse.ArgumentParser(
        description='E2: Per-Function Ablation on XOR (ALIFE 2026 Paper 1)')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds per function (default: 30)')
    parser.add_argument('--max-gens', type=int, default=300,
                        help='Maximum generations (default: 300)')
    parser.add_argument('--pop-size', type=int, default=500,
                        help='Population size (default: 500)')
    parser.add_argument('--function', type=str, default=None,
                        help='Test only specific function name')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='EMR max depth (default: 2)')
    parser.add_argument('--list', action='store_true',
                        help='Show completion status and exit')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seeds = list(range(42, 42 + args.seeds))

    # Status mode
    if args.list:
        show_status(OUTPUT_DIR, seeds)
        return

    # Select functions to test
    if args.function:
        if args.function not in ACTIVATION_LIST:
            print(f"ERROR: Unknown function '{args.function}'")
            print(f"Available: {', '.join(ACTIVATION_LIST)}")
            sys.exit(1)
        functions_to_test = [(ACTIVATION_LIST.index(args.function), args.function)]
    else:
        functions_to_test = [(i, name) for i, name in enumerate(ACTIVATION_LIST)]

    n_samples = 4  # 2^2 = 4 XOR samples
    print("=" * 70)
    print("ALIFE 2026 Paper 1 (E2): Per-Function Ablation on XOR (Parity-2)")
    print("=" * 70)
    print(f"Functions: {len(functions_to_test)}")
    print(f"Seeds: {args.seeds} ({seeds[0]}-{seeds[-1]})")
    print(f"Max generations: {args.max_gens}")
    print(f"Population: {args.pop_size}")
    print(f"Problem: XOR / Parity-2 ({n_samples} samples)")
    print(f"Target fitness: 0.95")
    print(f"Output: {OUTPUT_DIR}")

    start_time = time.time()
    all_results = {}

    for func_index, func_name in functions_to_test:
        result = run_function_experiment(
            func_name, func_index, seeds, OUTPUT_DIR,
            args.max_gens, args.pop_size, args.max_depth,
        )
        all_results[func_name] = result

        # Force garbage collection between functions to prevent OOM
        gc.collect()

    # Summary table
    print("\n" + "=" * 80)
    print("PER-FUNCTION ABLATION SUMMARY (XOR / Parity-2, 4 samples)")
    print("=" * 80)
    print(f"{'Function':<14} {'Category':<22} {'Solve%':<10} {'Avg Gen':<10} {'Avg Fit':<10}")
    print("-" * 66)

    # Sort by solve rate descending, then avg gen ascending
    sorted_funcs = sorted(
        all_results.items(),
        key=lambda x: (-x[1]['summary']['solve_rate'],
                       x[1]['summary']['avg_gen_to_solve'] or 999)
    )

    for func_name, result in sorted_funcs:
        s = result['summary']
        category = FUNCTION_CATEGORIES.get(func_name, 'Unknown')
        avg_gen_str = f"{s['avg_gen_to_solve']:.1f}" if s['avg_gen_to_solve'] else "Never"
        print(f"{func_name:<14} {category:<22} {s['solve_rate']*100:<10.0f} "
              f"{avg_gen_str:<10} {s['avg_best_fitness']:<10.4f}")

    total_time = time.time() - start_time

    # Save combined results
    combined = {
        'metadata': {
            'experiment': 'xor_per_function_ablation',
            'purpose': 'ALIFE 2026 Paper 1 (E2) -- per-function characterization on XOR',
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'seeds': seeds,
            'max_generations': args.max_gens,
            'pop_size': 150,  # repro: HMR pop-bug ran the paper at 150 regardless of nominal; EMR honors config
            'max_depth': args.max_depth,
            'problem': 'parity_2',
            'n_samples': n_samples,
            'activation_mode': 'global',
            'total_runtime_seconds': total_time,
            'function_categories': FUNCTION_CATEGORIES,
        },
        'results': {
            name: {
                'function_name': data['function_name'],
                'function_index': data['function_index'],
                'summary': data['summary'],
            }
            for name, data in all_results.items()
        },
    }

    combined_path = os.path.join(OUTPUT_DIR, "combined_results.json")
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    # Oscillatory vs monotonic summary
    osc_funcs = [n for n, c in FUNCTION_CATEGORIES.items() if 'Oscillatory' in c]
    mono_funcs = [n for n, c in FUNCTION_CATEGORIES.items()
                  if 'Monotonic' in c or c in ('Linear', 'Radial')]
    adaptive_funcs = [n for n, c in FUNCTION_CATEGORIES.items() if c == 'Adaptive']

    osc_rates = [all_results[f]['summary']['solve_rate'] for f in osc_funcs if f in all_results]
    mono_rates = [all_results[f]['summary']['solve_rate'] for f in mono_funcs if f in all_results]
    adapt_rates = [all_results[f]['summary']['solve_rate'] for f in adaptive_funcs if f in all_results]

    print(f"\n--- CATEGORY BREAKDOWN ---")
    if osc_rates:
        print(f"Oscillatory ({len(osc_rates)} funcs): avg {np.mean(osc_rates)*100:.0f}% solve rate")
    if mono_rates:
        print(f"Monotonic   ({len(mono_rates)} funcs): avg {np.mean(mono_rates)*100:.0f}% solve rate")
    if adapt_rates:
        print(f"Adaptive    ({len(adapt_rates)} funcs): avg {np.mean(adapt_rates)*100:.0f}% solve rate")

    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
