#!/usr/bin/env python3
"""Depth sensitivity: sin and tanh on Parity-4 at EMR depths 2, 4, 6.

ALIFE 2026 Paper 1 (E8): Tests whether the oscillatory divide is depth-invariant.
Expected: sin 100% at all depths, tanh 0% at all depths. The divide is
representational, not substrate capacity.

Note: Depth 2 data may already exist from per_function_ablation (Parity-4).
This script generates all conditions for consistent methodology.

Usage:
    python papers/emr-dynamic-functions/scripts/runners/E8_depth_sensitivity.py
    python papers/emr-dynamic-functions/scripts/runners/E8_depth_sensitivity.py --activation sin --depth 4
    python papers/emr-dynamic-functions/scripts/runners/E8_depth_sensitivity.py --list
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
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
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "results" / "depth_sensitivity_n30")
SEEDS = list(range(42, 72))  # 30 seeds
ACTIVATIONS = ['sin', 'tanh']
DEPTHS = [2, 4, 6]

POP_SIZE = 500
MAX_GENERATIONS = 300
TARGET_FITNESS = 0.95
SPECIES_SIZE = 10
VARIANCE_THRESHOLD = 0.03


@dataclass
class TrialResult:
    """Result of a single (activation, depth, seed) trial."""
    activation: str
    depth: int
    seed: int
    solved: bool
    solved_gen: Optional[int]
    best_fitness: float
    elapsed_seconds: float
    error: Optional[str] = None


def make_input_coords(n_inputs: int) -> List[tuple]:
    """Generate evenly-spaced input coordinates on y=-1.0."""
    coords = []
    for i in range(n_inputs):
        x = -1.0 + 2.0 * i / max(n_inputs - 1, 1) if n_inputs > 1 else 0.0
        coords.append((x, -1.0))
    return coords


def run_single_trial(
    activation_name: str,
    depth: int,
    seed: int,
) -> TrialResult:
    """Run a single Parity-4 trial with one activation at one depth.

    Args:
        activation_name: Activation function name (must be in ACTIVATION_LIST).
        depth: EMR max_depth (2, 4, or 6).
        seed: Random seed for NEAT initialization.

    Returns:
        TrialResult with solve status, generation, fitness, and timing.
    """
    start_time = time.time()

    problem = ParityProblem(n_bits=4)
    n_inputs = problem.input_shape[0]  # 5 (4 bits + bias)
    input_coords = make_input_coords(n_inputs)

    func_index = ACTIVATION_LIST.index(activation_name)

    algo_config = {
        'algorithm_params': {
            'emrhyperneat': {
                'emr_hyperneat': {
                    'extra_randkey_split': True,  # reproduce HMR (paper) per-seed results in EMR
                    'initial_depth': 0,
                    'max_depth': depth,
                    'variance_threshold': VARIANCE_THRESHOLD,
                    'dynamic_functions': {
                        'mode': 'global',
                        'hidden_activation': activation_name,
                        'palette': [func_index],
                        'palette_evolution': {'enabled': False},
                    },
                },
                'substrate': {
                    'input_coords': input_coords,
                    'output_coords': [(0.0, 1.0)],
                },
                'neat': {
                    'pop_size': 150,  # repro: HMR pop-bug ran the paper at 150 regardless of nominal; EMR honors config
                    'species_size': SPECIES_SIZE,
                },
            }
        }
    }

    algo = EMRHyperNEAT()
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
        activation=activation_name,
        depth=depth,
        seed=seed,
        solved=solved_gen is not None,
        solved_gen=solved_gen,
        best_fitness=float(best_fitness),
        elapsed_seconds=elapsed,
    )


def result_filepath(activation: str, depth: int) -> str:
    """Return the JSON file path for a given (activation, depth) condition."""
    return os.path.join(OUTPUT_DIR, f"{activation}_depth{depth}.json")


def load_existing_results(filepath: str) -> List[Dict[str, Any]]:
    """Load previously saved results from JSON, or return empty list."""
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        data = json.load(f)
    return data.get('results', [])


def completed_seeds(results: List[Dict[str, Any]]) -> set:
    """Extract set of completed seeds from results list."""
    return {r['seed'] for r in results if 'seed' in r}


def save_results(
    filepath: str,
    activation: str,
    depth: int,
    results: List[Dict[str, Any]],
) -> None:
    """Save results incrementally to JSON."""
    valid = [r for r in results if r.get('error') is None]
    solved = sum(1 for r in valid if r.get('solved', False))
    n = len(valid)

    solved_gens = [r['solved_gen'] for r in valid if r.get('solved', False)]

    summary = {
        'n': n,
        'n_errors': len(results) - n,
        'solved': solved,
        'solve_rate': solved / n if n > 0 else 0.0,
    }
    if solved_gens:
        summary['median_gen'] = float(np.median(solved_gens))
        summary['mean_gen'] = float(np.mean(solved_gens))
        summary['min_gen'] = int(min(solved_gens))
        summary['max_gen'] = int(max(solved_gens))

    output = {
        'activation': activation,
        'depth': depth,
        'config': {
            'problem': 'Parity-4',
            'n_inputs': 5,
            'pop_size': 150,  # repro: HMR pop-bug ran the paper at 150 regardless of nominal; EMR honors config
            'max_generations': MAX_GENERATIONS,
            'target_fitness': TARGET_FITNESS,
            'variance_threshold': VARIANCE_THRESHOLD,
            'dynamic_functions_mode': 'global',
        },
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'results': results,
    }
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def run_condition(activation: str, depth: int, seeds: List[int]) -> None:
    """Run all seeds for one (activation, depth) condition with resume logic."""
    filepath = result_filepath(activation, depth)

    # Load existing and determine remaining seeds
    results = load_existing_results(filepath)
    done = completed_seeds(results)
    remaining = [s for s in seeds if s not in done]

    if not remaining:
        valid = [r for r in results if r.get('error') is None]
        solved = sum(1 for r in valid if r.get('solved', False))
        print(f"  [{activation} depth={depth}] All {len(seeds)} seeds complete "
              f"({solved}/{len(valid)} solved). Skipping.")
        return

    if done:
        print(f"  [{activation} depth={depth}] Resuming: {len(done)} done, "
              f"{len(remaining)} remaining")
    else:
        print(f"  [{activation} depth={depth}] Starting fresh: {len(remaining)} seeds")

    condition_start = time.time()

    for seed in remaining:
        print(f"    Seed {seed}: ", end='', flush=True)
        try:
            trial = run_single_trial(activation, depth, seed)
            results.append(asdict(trial))

            if trial.solved:
                print(f"SOLVED gen {trial.solved_gen} ({trial.elapsed_seconds:.1f}s)")
            else:
                print(f"FAIL fitness={trial.best_fitness:.4f} ({trial.elapsed_seconds:.1f}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'activation': activation,
                'depth': depth,
                'seed': seed,
                'solved': False,
                'solved_gen': None,
                'best_fitness': 0.0,
                'elapsed_seconds': 0.0,
                'error': str(e),
            })

        # Incremental save after each seed
        save_results(filepath, activation, depth, results)

    # Condition summary
    valid = [r for r in results if r.get('error') is None]
    solved = sum(1 for r in valid if r.get('solved', False))
    solved_gens = [r['solved_gen'] for r in valid if r.get('solved', False)]
    elapsed = time.time() - condition_start

    print(f"  [{activation} depth={depth}] Done: {solved}/{len(valid)} solved "
          f"({solved/len(valid)*100:.1f}%)", end='')
    if solved_gens:
        print(f", median gen {np.median(solved_gens):.1f}", end='')
    print(f" [{elapsed/60:.1f} min]")


def show_status() -> None:
    """Print completion status for all conditions."""
    print(f"\n{'='*70}")
    print("DEPTH SENSITIVITY STATUS")
    print(f"{'='*70}")
    print(f"{'Activation':<12} {'Depth':<8} {'Done':<8} {'Solved':<10} {'Rate':<10} {'File'}")
    print(f"{'-'*70}")

    total_done = 0
    total_solved = 0
    total_expected = len(ACTIVATIONS) * len(DEPTHS) * len(SEEDS)

    for activation in ACTIVATIONS:
        for depth in DEPTHS:
            filepath = result_filepath(activation, depth)
            if os.path.exists(filepath):
                results = load_existing_results(filepath)
                valid = [r for r in results if r.get('error') is None]
                n = len(valid)
                solved = sum(1 for r in valid if r.get('solved', False))
                rate = f"{solved/n*100:.1f}%" if n > 0 else "N/A"
                total_done += n
                total_solved += solved
                exists = "YES"
            else:
                n = 0
                solved = 0
                rate = "N/A"
                exists = "NO"

            print(f"{activation:<12} {depth:<8} {n:<8} {solved:<10} {rate:<10} {exists}")

    print(f"{'-'*70}")
    print(f"Total: {total_done}/{total_expected} runs complete, "
          f"{total_solved} solved")
    print()


def print_summary_table(activations: List[str], depths: List[int]) -> None:
    """Print a summary table across all conditions."""
    print(f"\n{'='*70}")
    print("DEPTH SENSITIVITY SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Activation':<12}", end='')
    for depth in depths:
        print(f"{'Depth '+str(depth):>18}", end='')
    print()
    print(f"{'-'*70}")

    for activation in activations:
        print(f"{activation:<12}", end='')
        for depth in depths:
            filepath = result_filepath(activation, depth)
            if os.path.exists(filepath):
                results = load_existing_results(filepath)
                valid = [r for r in results if r.get('error') is None]
                n = len(valid)
                solved = sum(1 for r in valid if r.get('solved', False))
                solved_gens = [r['solved_gen'] for r in valid if r.get('solved', False)]
                rate = f"{solved}/{n}" if n > 0 else "---"
                med = f"med={np.median(solved_gens):.0f}" if solved_gens else ""
                cell = f"{rate} {med}"
            else:
                cell = "---"
            print(f"{cell:>18}", end='')
        print()
    print()


def save_combined_results(activations: List[str], depths: List[int]) -> None:
    """Merge all condition files into a single combined JSON."""
    combined = {
        'experiment': 'E8_depth_sensitivity',
        'problem': 'Parity-4',
        'config': {
            'activations': activations,
            'depths': depths,
            'seeds': SEEDS,
            'pop_size': 150,  # repro: HMR pop-bug ran the paper at 150 regardless of nominal; EMR honors config
            'max_generations': MAX_GENERATIONS,
            'target_fitness': TARGET_FITNESS,
        },
        'timestamp': datetime.now().isoformat(),
        'conditions': {},
    }

    for activation in activations:
        for depth in depths:
            key = f"{activation}_depth{depth}"
            filepath = result_filepath(activation, depth)
            if os.path.exists(filepath):
                with open(filepath) as f:
                    data = json.load(f)
                combined['conditions'][key] = {
                    'summary': data.get('summary', {}),
                    'results': data.get('results', []),
                }

    combined_path = os.path.join(OUTPUT_DIR, "depth_sensitivity_combined.json")
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Combined results saved: {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description='E8: Depth Sensitivity — sin and tanh on Parity-4 at depths 2, 4, 6'
    )
    parser.add_argument('--activation', type=str, default=None,
                        choices=ACTIVATIONS,
                        help='Run a specific activation only')
    parser.add_argument('--depth', type=int, default=None,
                        choices=DEPTHS,
                        help='Run a specific depth only')
    parser.add_argument('--list', action='store_true',
                        help='Show completion status and exit')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --list: show status and exit
    if args.list:
        show_status()
        return

    # Determine which conditions to run
    activations = [args.activation] if args.activation else ACTIVATIONS
    depths = [args.depth] if args.depth else DEPTHS

    total_conditions = len(activations) * len(depths)
    total_runs = total_conditions * len(SEEDS)

    print("=" * 70)
    print("E8: DEPTH SENSITIVITY — OSCILLATORY DIVIDE VS SUBSTRATE CAPACITY")
    print("=" * 70)
    print(f"Problem:      Parity-4 (5 inputs, 1 output)")
    print(f"Activations:  {activations}")
    print(f"Depths:       {depths}")
    print(f"Seeds:        {len(SEEDS)} ({SEEDS[0]}-{SEEDS[-1]})")
    print(f"Pop size:     {POP_SIZE}")
    print(f"Max gens:     {MAX_GENERATIONS}")
    print(f"Target:       {TARGET_FITNESS}")
    print(f"Conditions:   {total_conditions} ({total_runs} max runs)")
    print(f"Output:       {OUTPUT_DIR}")
    print()

    experiment_start = time.time()

    for activation in activations:
        for depth in depths:
            print(f"\n{'─'*60}")
            print(f"CONDITION: {activation} depth={depth}")
            print(f"{'─'*60}")

            run_condition(activation, depth, SEEDS)

            # Release memory between conditions
            gc.collect()

    # Print summary table
    print_summary_table(activations, depths)

    # Save combined results
    save_combined_results(activations, depths)

    total_elapsed = time.time() - experiment_start
    print(f"Total experiment time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")
    print("=" * 70)


if __name__ == '__main__':
    main()
