#!/usr/bin/env python3
"""Monotonic parity scaling: tanh activation on Parity-2 through Parity-8 at N=30.

ALIFE 2026 Paper 1 (E1): Mirror of Table 5 sin scaling with tanh.
Expected: 0% from Parity-3+ (tanh cannot produce oscillatory patterns needed for parity).

Usage:
    python papers/emr-dynamic-functions/scripts/runners/E1_parity_scaling.py
    python papers/emr-dynamic-functions/scripts/runners/E1_parity_scaling.py --n-bits 4
    python papers/emr-dynamic-functions/scripts/runners/E1_parity_scaling.py --list
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification_problems import ParityProblem
from emr_hyperneat.emrhyperneat import (
    ACTIVATION_LIST,
    EMRHyperNEAT,
)

OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "results" / "monotonic_parity_scaling_n30")
SEEDS = list(range(42, 72))  # 30 seeds: 42-71
PARITY_SIZES = [2, 3, 4, 5, 6, 7, 8]
TANH_INDEX = ACTIVATION_LIST.index('tanh')  # Should be 0

MAX_GENERATIONS = 100
POP_SIZE = 300
MAX_DEPTH = 2
TARGET_FITNESS = 0.95


def load_partial_results(filepath: str) -> Optional[dict]:
    """Load existing partial results from JSON file."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


def get_completed_seeds(data: Optional[dict]) -> set[int]:
    """Extract set of completed seed numbers from result data."""
    if data is None:
        return set()
    return {r['seed'] for r in data.get('results', [])}


def build_result_dict(n_bits: int, results_list: list[dict]) -> dict[str, Any]:
    """Build the standard result dict with summary statistics."""
    solve_count = sum(1 for r in results_list if r['solved'])
    total = len(results_list)
    solve_rate = solve_count / total if total > 0 else 0.0

    solved_gens = [r['solved_gen'] for r in results_list if r['solved']]
    avg_gen = float(np.mean(solved_gens)) if solved_gens else None
    std_gen = float(np.std(solved_gens)) if len(solved_gens) > 1 else (0.0 if solved_gens else None)

    best_fitnesses = [r['best_fitness'] for r in results_list]

    return {
        'n_bits': n_bits,
        'n_samples': 2 ** n_bits,
        'function_name': 'tanh',
        'function_index': TANH_INDEX,
        'results': sorted(results_list, key=lambda r: r['seed']),
        'summary': {
            'solve_rate': solve_rate,
            'solve_count': solve_count,
            'total_runs': total,
            'avg_gen_to_solve': avg_gen,
            'std_gen_to_solve': std_gen,
            'avg_best_fitness': float(np.mean(best_fitnesses)) if best_fitnesses else None,
            'std_best_fitness': float(np.std(best_fitnesses)) if len(best_fitnesses) > 1 else None,
        },
    }


def make_input_coords(n_inputs: int) -> list[tuple[float, float]]:
    """Generate evenly-spaced input coordinates on y=-1.0."""
    if n_inputs == 1:
        return [(0.0, -1.0)]
    return [(-1.0 + 2.0 * i / (n_inputs - 1), -1.0) for i in range(n_inputs)]


def run_single_seed(n_bits: int, seed: int) -> dict[str, Any]:
    """Run a single trial of tanh on Parity-{n_bits} with given seed."""
    problem = ParityProblem(n_bits=n_bits)
    n_inputs = problem.input_shape[0]
    input_coords = make_input_coords(n_inputs)
    output_coords = [(0.0, 1.0)]

    algo_config = {
        'algorithm_params': {
            'emrhyperneat': {
                'emr_hyperneat': {
                    'extra_randkey_split': True,  # reproduce HMR (paper) per-seed results in EMR
                    'initial_depth': 0,
                    'max_depth': MAX_DEPTH,
                    'variance_threshold': 0.03,
                    'dynamic_functions': {
                        'mode': 'global',
                        'hidden_activation': 'tanh',
                        'palette': [TANH_INDEX],
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
    solved_gen: Optional[int] = None
    start_time = time.time()
    gens_run = 0

    for gen in range(MAX_GENERATIONS):
        state, metrics = algo.run_generation(state, problem)
        best_fitness = max(best_fitness, metrics.best_fitness)
        gens_run = gen + 1

        if metrics.best_fitness >= TARGET_FITNESS and solved_gen is None:
            solved_gen = gen + 1
            break

    elapsed = time.time() - start_time

    return {
        'function_name': 'tanh',
        'function_index': TANH_INDEX,
        'seed': seed,
        'n_bits': n_bits,
        'solved': solved_gen is not None,
        'solved_gen': solved_gen,
        'best_fitness': float(best_fitness),
        'generations_run': gens_run,
        'elapsed_seconds': elapsed,
    }


def run_parity_size(n_bits: int) -> dict[str, Any]:
    """Run tanh activation on Parity-{n_bits} with incremental saves."""
    filepath = os.path.join(OUTPUT_DIR, f"parity_{n_bits}.json")

    existing = load_partial_results(filepath)
    completed_seeds = get_completed_seeds(existing)
    remaining_seeds = [s for s in SEEDS if s not in completed_seeds]

    if not remaining_seeds:
        print(f"  Parity-{n_bits}: all 30 seeds complete, skipping")
        return existing

    print(f"\n{'='*60}")
    print(f"PARITY-{n_bits} ({2**n_bits} samples) — tanh activation")
    print(f"  Completed: {len(completed_seeds)}/30, remaining: {len(remaining_seeds)}")
    print(f"{'='*60}")

    results_list: list[dict] = existing['results'] if existing is not None else []

    for seed in remaining_seeds:
        print(f"  Seed {seed}: ", end='', flush=True)

        trial_result = run_single_seed(n_bits, seed)
        results_list.append(trial_result)

        if trial_result['solved']:
            print(f"SOLVED @ gen {trial_result['solved_gen']} ({trial_result['elapsed_seconds']:.1f}s)")
        else:
            print(f"FAIL (fit={trial_result['best_fitness']:.4f}, {trial_result['elapsed_seconds']:.1f}s)")

        # Save after EVERY seed for safe resume
        data = build_result_dict(n_bits, results_list)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    return build_result_dict(n_bits, results_list)


def update_combined_results() -> None:
    """Rebuild combined.json from all individual parity size JSONs."""
    combined_path = os.path.join(OUTPUT_DIR, "combined.json")

    all_results: dict[str, Any] = {}
    for n_bits in PARITY_SIZES:
        filepath = os.path.join(OUTPUT_DIR, f"parity_{n_bits}.json")
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
            all_results[f"parity_{n_bits}"] = data

    combined = {
        'metadata': {
            'experiment': 'monotonic_parity_scaling_n30',
            'purpose': 'ALIFE 2026 Paper 1 (E1) — Parity scaling with tanh activation at N=30',
            'seeds': SEEDS,
            'max_generations': MAX_GENERATIONS,
            'pop_size': 150,  # repro: HMR pop-bug ran the paper at 150 regardless of nominal; EMR honors config
            'max_depth': MAX_DEPTH,
            'activation': 'tanh',
            'activation_mode': 'global',
            'parity_sizes': PARITY_SIZES,
        },
        'results': all_results,
    }

    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined results saved to: {combined_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("MONOTONIC PARITY SCALING SUMMARY (tanh activation, N=30)")
    print(f"{'='*60}")
    print(f"{'Problem':<12} {'Samples':<10} {'Solve%':<10} {'Avg Gen':<10} {'Std':<10}")
    print("-" * 52)
    for n_bits in PARITY_SIZES:
        key = f"parity_{n_bits}"
        if key in all_results:
            s = all_results[key]['summary']
            avg_str = f"{s['avg_gen_to_solve']:.1f}" if s['avg_gen_to_solve'] else "Never"
            std_str = f"{s['std_gen_to_solve']:.1f}" if s['std_gen_to_solve'] else "---"
            print(f"Parity-{n_bits:<5} {2**n_bits:<10} {s['solve_rate']*100:<10.0f} {avg_str:<10} {std_str:<10}")


def show_status() -> None:
    """Print completion status for all parity sizes."""
    print("Monotonic parity scaling status (tanh, N=30 target):")
    print(f"{'Problem':<14} {'Seeds':<10} {'Solve%':<10} {'Status'}")
    print("-" * 50)
    for n_bits in PARITY_SIZES:
        filepath = os.path.join(OUTPUT_DIR, f"parity_{n_bits}.json")
        existing = load_partial_results(filepath)
        n_done = len(get_completed_seeds(existing))
        status = "COMPLETE" if n_done >= 30 else f"{n_done}/30"

        solve_str = "---"
        if existing and 'summary' in existing:
            solve_str = f"{existing['summary']['solve_rate']*100:.0f}%"

        print(f"Parity-{n_bits:<7} {n_done:<10} {solve_str:<10} {status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Monotonic parity scaling with tanh activation at N=30 (Paper 1 E1)'
    )
    parser.add_argument('--n-bits', type=int, default=None,
                        help='Run specific parity size only (e.g., 4)')
    parser.add_argument('--list', action='store_true',
                        help='Show completion status and exit')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.list:
        show_status()
        return

    if args.n_bits:
        if args.n_bits < 2 or args.n_bits > 8:
            print(f"Error: --n-bits must be between 2 and 8, got {args.n_bits}")
            sys.exit(1)
        sizes = [args.n_bits]
    else:
        sizes = PARITY_SIZES

    print("=" * 70)
    print("Paper 1 E1: Monotonic Parity Scaling — N=30 with Tanh Activation")
    print("=" * 70)
    print(f"Sizes: {sizes}")
    print(f"Seeds: 30 (42-71)")
    print(f"Config: pop={POP_SIZE}, depth={MAX_DEPTH}, tanh activation, {MAX_GENERATIONS} gens max")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Resume: automatic (completed seeds skipped)")
    print(f"Expected: 0% solve rate from Parity-3+ (monotonic cannot do parity)")

    start_time = time.time()

    for n_bits in sizes:
        run_parity_size(n_bits)

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")

    update_combined_results()


if __name__ == '__main__':
    main()
