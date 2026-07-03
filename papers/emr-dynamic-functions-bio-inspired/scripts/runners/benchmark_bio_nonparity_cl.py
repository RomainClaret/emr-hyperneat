#!/usr/bin/env python3
"""Non-Parity CL: P4 → Two Moons → P4 (and reverse).

Tests whether the CL advantage generalizes beyond the Parity-4/Concentric Circles
problem pair. This is the last "Open" item in the paper's RQ2 conclusion.

Design:
  Forward: Parity-4 → Two Moons → Parity-4 (3 stages, fresh NEAT per task)
  Reverse: Two Moons → Parity-4 → Two Moons (3 stages, fresh NEAT per task)
  Seed continuity: seed_next = base + gens_to_solve * 7 + (1 if solved else 0) * 1000
  7 strategies × 30 seeds × 2 sequences = 420 runs

Usage:
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_nonparity_cl.py --sequence forward
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_nonparity_cl.py --sequence reverse
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_nonparity_cl.py --sequence forward --strategy circadian_rhythm_dual
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_nonparity_cl.py --sequence forward --pilot
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification_problems import XORProblem
from emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions_aggregation import (
    HMRHyperNEAT,
)
from classification_problems import ParityProblem, TwoMoonsProblem

from palette_strategies.strategy_47_baseline_dual import BaselineDualStrategy
from palette_strategies.strategy_52_circadian_rhythm_dual import CircadianRhythmDualStrategy
from palette_strategies.strategy_16_stdp_dual import STDPDualStrategy
from palette_strategies.strategy_64_critical_period_refined_dual import CriticalPeriodRefinedDualStrategy
from palette_strategies.strategy_29_clonal_selection_dual import ClonalSelectionDualStrategy
from palette_strategies.strategy_8_hebbian_dual import HebbianDualStrategy

OSCILLATORY_INDICES = {4, 11, 12, 13, 15}

STRATEGIES = {
    'baseline_dual': BaselineDualStrategy,
    'circadian_rhythm_dual': CircadianRhythmDualStrategy,
    'stdp_dual': STDPDualStrategy,
    'critical_period_refined_dual': CriticalPeriodRefinedDualStrategy,
    'clonal_selection_dual': ClonalSelectionDualStrategy,
    'hebbian_dual': HebbianDualStrategy,
}

# Exclude strategies that are passive controls or variants
# consolidation_window_dual, sin_default_dual, baseline_30pct_dual not needed

SEQUENCES = {
    'forward': [
        {'name': 'parity_4', 'target': 0.95, 'max_gens': 100},
        {'name': 'two_moons', 'target': 0.90, 'max_gens': 100},
        {'name': 'parity_4', 'target': 0.95, 'max_gens': 100},
    ],
    'reverse': [
        {'name': 'two_moons', 'target': 0.90, 'max_gens': 100},
        {'name': 'parity_4', 'target': 0.95, 'max_gens': 100},
        {'name': 'two_moons', 'target': 0.90, 'max_gens': 100},
    ],
}

PROBLEMS = {
    'parity_4': lambda: ParityProblem(n_bits=4),
    'two_moons': lambda: TwoMoonsProblem(n_samples=200, noise=0.10),
}


def make_substrate_coords(problem):
    """Generate substrate input/output coordinates."""
    n_inputs = problem.input_shape[0]
    n_outputs = problem.output_shape[0]

    input_coords = []
    for i in range(n_inputs):
        x = -1.0 + 2.0 * i / max(n_inputs - 1, 1) if n_inputs > 1 else 0.0
        input_coords.append((x, -1.0))

    output_coords = []
    for i in range(n_outputs):
        x = -1.0 + 2.0 * i / max(n_outputs - 1, 1) if n_outputs > 1 else 0.0
        output_coords.append((x, 1.0))

    return input_coords, output_coords


def run_nonparity_cl(
    strategy_name: str,
    seed: int,
    sequence_name: str,
    pop_size: int = 500,
    max_depth: int = 4,
) -> Dict[str, Any]:
    """Run non-parity cross-problem CL with seed continuity."""
    tasks = SEQUENCES[sequence_name]
    strategy_cls = STRATEGIES[strategy_name]
    strategy = strategy_cls()
    strategy_state = strategy.initialize({
        'initial_act_palette': [0, 1, 2, 3],
        'initial_agg_palette': [0, 1],
    }, seed)

    results = {
        'strategy': strategy_name,
        'seed': seed,
        'sequence': sequence_name,
        'tasks': [],
    }

    current_seed = seed
    total_gen = 0

    for task in tasks:
        task_start = time.time()
        task_name = task['name']
        target = task['target']
        max_gens = task['max_gens']

        problem = PROBLEMS[task_name]()
        input_coords, output_coords = make_substrate_coords(problem)
        current_act_palette = strategy.get_active_palette(strategy_state)
        current_agg_palette = strategy.get_active_agg_palette(strategy_state)

        algo_config = {
            'algorithm_params': {
                'hmrhyperneat': {
                    'hmr_hyperneat': {
                        'initial_depth': 0,
                        'max_depth': max_depth,
                        'variance_threshold': 0.03,
                        'dynamic_functions': {
                            'mode': 'cppn_output',
                            'palette': current_act_palette,
                            'palette_evolution': {'enabled': False},
                            'aggregation': {
                                'mode': 'cppn_output',
                                'num_aggregations': 6,
                            },
                        },
                    },
                    'substrate': {
                        'input_coords': input_coords,
                        'output_coords': output_coords,
                    },
                    'neat': {
                        'pop_size': pop_size,
                        'species_size': 10,
                    },
                }
            }
        }

        algo = HMRHyperNEAT()
        neat_config = algo.create_config(algo_config)
        algo_state = algo.initialize(neat_config, problem, seed=current_seed)

        solved = False
        gens_to_solve = max_gens
        best_fitness = 0.0
        prev_fitness = 0.0

        for gen in range(max_gens):
            current_act_palette = strategy.get_active_palette(strategy_state)
            algo.df_palette = jnp.array(current_act_palette)

            if hasattr(strategy, 'get_active_agg_palette'):
                current_agg_palette = strategy.get_active_agg_palette(strategy_state)
                algo.agg_palette = jnp.array(current_agg_palette)

            algo_state, metrics = algo.run_generation(algo_state, problem)
            gen_fitness = float(metrics.best_fitness)
            best_fitness = max(best_fitness, gen_fitness)

            strategy_state, _ = strategy.post_generation_update(
                strategy_state, total_gen, gen_fitness, prev_fitness,
                population_data={'fitnesses': [gen_fitness]}
            )
            prev_fitness = gen_fitness
            total_gen += 1

            if gen_fitness >= target and not solved:
                solved = True
                gens_to_solve = gen + 1
                break

        # Seed continuity: breaks determinism
        current_seed = seed + gens_to_solve * 7 + (1 if solved else 0) * 1000

        task_time = time.time() - task_start
        final_palette = list(strategy.get_active_palette(strategy_state))

        task_result = {
            'task': task_name,
            'solved': solved,
            'gens_to_solve': gens_to_solve if solved else None,
            'best_fitness': best_fitness,
            'final_act_palette': final_palette,
            'has_sin': 4 in final_palette,
            'has_oscillatory': any(i in OSCILLATORY_INDICES for i in final_palette),
            'next_seed': current_seed,
            'time_seconds': task_time,
        }
        results['tasks'].append(task_result)

        status = "SOLVED" if solved else f"FAIL ({best_fitness:.3f})"
        print(f"      {task_name} (target={target}): {status} in {gens_to_solve} gens")

    tasks_solved = sum(1 for t in results['tasks'] if t['solved'])
    results['summary'] = {
        'tasks_solved': tasks_solved,
        'total_tasks': len(tasks),
        'all_solved': tasks_solved == len(tasks),
        'stage1_solved': results['tasks'][0]['solved'],
        'stage2_solved': results['tasks'][1]['solved'],
        'stage3_solved': results['tasks'][2]['solved'],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Non-Parity CL Benchmark')
    parser.add_argument('--sequence', choices=['forward', 'reverse'], required=True)
    parser.add_argument('--strategy', type=str, default=None,
                        help='Single strategy to run (default: all)')
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--pilot', action='store_true', help='Run N=3 pilot')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[2] / 'results' / 'non_parity_cl' / args.sequence
    os.makedirs(output_dir, exist_ok=True)

    strategies = [args.strategy] if args.strategy else list(STRATEGIES.keys())

    if args.pilot:
        seeds = [42, 43, 44]
        print(f"PILOT MODE: {args.sequence}, N=3")
    else:
        seeds = [42 + i for i in range(args.seeds)]

    for strategy_name in strategies:
        result_file = output_dir / f'{strategy_name}.json'

        # Resume support
        results_list = []
        existing_seeds = set()
        if result_file.exists():
            with open(result_file) as f:
                existing_data = json.load(f)
            results_list = existing_data.get('results', [])
            existing_seeds = {r['seed'] for r in results_list}
            remaining = [s for s in seeds if s not in existing_seeds]
            if not remaining:
                print(f"All {len(seeds)} seeds completed for {strategy_name} ({args.sequence}). Skipping.")
                continue
            print(f"Resuming: {len(existing_seeds)} done, {len(remaining)} remaining")
            seeds_to_run = remaining
        else:
            print(f"Starting fresh: {strategy_name} ({args.sequence}), N={len(seeds)}")
            seeds_to_run = seeds

        print(f"\n{'='*60}")
        print(f"NON-PARITY CL: {strategy_name} ({args.sequence})")
        print(f"Sequence: {' → '.join(t['name'] for t in SEQUENCES[args.sequence])}")
        print(f"Seeds: {len(seeds_to_run)} ({seeds_to_run[0]}-{seeds_to_run[-1]})")
        print(f"{'='*60}")

        for seed in seeds_to_run:
            print(f"\n  Seed {seed}:", end='')
            try:
                result = run_nonparity_cl(strategy_name, seed, args.sequence)
                results_list.append(result)

                all_solved = result['summary']['all_solved']
                status = "ALL" if all_solved else f"{result['summary']['tasks_solved']}/3"
                print(f"  → {status}")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            # Incremental save
            save_data = {
                'strategy': strategy_name,
                'sequence': args.sequence,
                'experiment': 'nonparity_cl',
                'results': results_list,
                'summary': {
                    'n': len(results_list),
                    'all_solved': sum(1 for r in results_list if r['summary']['all_solved']),
                    'all_solved_rate': sum(1 for r in results_list if r['summary']['all_solved']) / len(results_list),
                },
            }
            with open(result_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)

        # Final summary
        n = len(results_list)
        all_solved = sum(1 for r in results_list if r['summary']['all_solved'])
        s1 = sum(1 for r in results_list if r['summary']['stage1_solved'])
        s2 = sum(1 for r in results_list if r['summary']['stage2_solved'])
        s3 = sum(1 for r in results_list if r['summary']['stage3_solved'])
        print(f"\n{'='*60}")
        print(f"SUMMARY: {strategy_name} ({args.sequence})")
        print(f"  All-3 solved: {all_solved}/{n} ({100*all_solved/n:.1f}%)")
        print(f"  Stage 1: {s1}/{n} ({100*s1/n:.1f}%)")
        print(f"  Stage 2: {s2}/{n} ({100*s2/n:.1f}%)")
        print(f"  Stage 3: {s3}/{n} ({100*s3/n:.1f}%)")
        print(f"  Results: {result_file}")


if __name__ == '__main__':
    main()
