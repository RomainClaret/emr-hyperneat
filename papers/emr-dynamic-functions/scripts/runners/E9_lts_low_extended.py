#!/usr/bin/env python3
"""lts_low extended seeds (72-101): Tighten CI from N=30 to N=60.

ALIFE 2026 Paper 1 (E9): The lts_low function has the widest CI in the
per-function ablation [0.8%, 22.1%]. Adding 30 seeds (72-101) narrows this
to approximately [3%, 15%].

Uses the same configuration as per_function_ablation.py with lts_low.

Usage:
    python papers/emr-dynamic-functions/scripts/runners/E9_lts_low_extended.py
    python papers/emr-dynamic-functions/scripts/runners/E9_lts_low_extended.py --list
"""

import argparse
import json
import os
from pathlib import Path
import time
import sys
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification_problems import ParityProblem

from emr_hyperneat.emrhyperneat import (
    EMRHyperNEAT,
    ACTIVATION_LIST,
)

OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "results" / "per_function_ablation_n30")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lts_low_extended.json")
SEEDS = list(range(72, 102))  # 30 additional seeds: 72-101
LTS_LOW_INDEX = ACTIVATION_LIST.index('lts_low')


def load_partial_results(filepath: str) -> Optional[dict]:
    """Load existing partial results."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


def get_completed_seeds(data: Optional[dict]) -> set:
    """Extract completed seed numbers."""
    if data is None:
        return set()
    return {r['seed'] for r in data.get('results', [])}


def build_result_dict(results_list: list) -> dict:
    """Build standard result dict from results list."""
    solve_count = sum(1 for r in results_list if r['solved'])
    total = len(results_list)
    solve_rate = solve_count / total if total > 0 else 0.0

    solved_gens = [r['solved_gen'] for r in results_list if r['solved']]
    avg_gen = float(np.mean(solved_gens)) if solved_gens else None
    std_gen = float(np.std(solved_gens)) if len(solved_gens) > 1 else (0.0 if solved_gens else None)

    best_fitnesses = [r['best_fitness'] for r in results_list]

    return {
        'function_name': 'lts_low',
        'function_index': LTS_LOW_INDEX,
        'seeds_range': '72-101',
        'purpose': 'E9: Extend lts_low from N=30 to N=60 to tighten CI',
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


def run_single_seed(seed: int) -> dict:
    """Run single Parity-4 trial with lts_low activation."""
    problem = ParityProblem(n_bits=4)
    start_time = time.time()

    n_inputs = problem.input_shape[0]
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
                    'max_depth': 2,
                    'variance_threshold': 0.03,
                    'dynamic_functions': {
                        'mode': 'global',
                        'hidden_activation': 'lts_low',
                        'palette': [LTS_LOW_INDEX],
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
    max_generations = 300
    target_fitness = 0.95

    for gen in range(max_generations):
        state, metrics = algo.run_generation(state, problem)
        best_fitness = max(best_fitness, metrics.best_fitness)

        if metrics.best_fitness >= target_fitness and solved_gen is None:
            solved_gen = gen + 1
            break

    elapsed = time.time() - start_time

    return {
        'function_name': 'lts_low',
        'function_index': LTS_LOW_INDEX,
        'seed': seed,
        'solved': solved_gen is not None,
        'solved_gen': solved_gen,
        'best_fitness': float(best_fitness),
        'generations_run': gen + 1,
        'elapsed_seconds': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description='lts_low extended seeds (E9): N=30 additional seeds (72-101)'
    )
    parser.add_argument('--list', action='store_true',
                        help='Show status and exit')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = load_partial_results(OUTPUT_FILE)
    completed_seeds = get_completed_seeds(existing)

    if args.list:
        n_done = len(completed_seeds)
        print(f"lts_low extended (seeds 72-101): {n_done}/30")
        if existing:
            s = existing['summary']
            print(f"  Solve rate: {s['solve_rate']*100:.1f}%")
            if s['avg_gen_to_solve']:
                print(f"  Avg gen: {s['avg_gen_to_solve']:.1f}")
        return

    remaining_seeds = [s for s in SEEDS if s not in completed_seeds]

    if not remaining_seeds:
        print("All 30 extended seeds complete.")
        return

    print("=" * 60)
    print("E9: lts_low Extended Seeds (72-101) on Parity-4")
    print("=" * 60)
    print(f"Completed: {len(completed_seeds)}/30, remaining: {len(remaining_seeds)}")
    print(f"Config: pop=500, depth=2, 300 gens max, target=0.95")
    print(f"Output: {OUTPUT_FILE}")

    results_list = existing['results'] if existing is not None else []

    for seed in remaining_seeds:
        print(f"  Seed {seed}: ", end='', flush=True)

        trial = run_single_seed(seed)
        results_list.append(trial)

        if trial['solved']:
            print(f"SOLVED @ gen {trial['solved_gen']} ({trial['elapsed_seconds']:.1f}s)")
        else:
            print(f"FAIL (fit={trial['best_fitness']:.4f}, {trial['elapsed_seconds']:.1f}s)")

        # Save after every seed
        data = build_result_dict(results_list)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    # Final summary
    data = build_result_dict(results_list)
    print(f"\n{'='*60}")
    print("lts_low EXTENDED SUMMARY (seeds 72-101)")
    print(f"{'='*60}")
    s = data['summary']
    print(f"Solve rate: {s['solve_rate']*100:.1f}% ({s['solve_count']}/{s['total_runs']})")
    if s['avg_gen_to_solve']:
        print(f"Avg gen to solve: {s['avg_gen_to_solve']:.1f}")
    print(f"Avg best fitness: {s['avg_best_fitness']:.4f}")


if __name__ == '__main__':
    main()
