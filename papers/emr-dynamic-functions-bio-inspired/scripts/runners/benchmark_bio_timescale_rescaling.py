#!/usr/bin/env python3
"""P1b: Timescale Rescaling Validation for Bio-Inspired Strategies.

Tests whether compressing the characteristic timescale (Tc) of failed/borderline
strategies recovers their performance on Parity-4. This is the key causal test
for RQ3 (timescale compatibility hypothesis).

Strategies and their rescaled parameters:
    GRN (Tc >>100 -> ~10):
        expression_decay: 0.9 -> 0.3
        regulation_learning_rate: 0.08 -> 0.8
        regulation_decay: 0.98 -> 0.80
        hill_coefficient: 2.0 -> 4.0

    Glial Modulation (Tc ~50 -> ~5):
        support_decay: 0.99 -> 0.90
        support_learning_rate: 0.12 -> 1.2
        energy_regen_rate: 15 -> 1.5
        base_function_cost: 1.5 -> 15.0

    Ant Colony (Tc ~20 -> ~5):
        pheromone_decay: 0.85 -> 0.22
        pheromone_deposit: 0.3 -> 1.5
        elite_bonus: 2.0 -> 10.0

Design:
    - Problem: Parity-4 (pop=500, 100 gens, depth=2, feedforward)
    - Seeds: 42-71 (N=30 each)
    - Target fitness: 0.95
    - Run ONE strategy per invocation (OOM safety)

Usage:
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_timescale_rescaling.py --strategy grn_rescaled
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_timescale_rescaling.py --strategy glial_rescaled
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_timescale_rescaling.py --strategy ant_colony_rescaled
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_timescale_rescaling.py --strategy grn_rescaled --pilot
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification_problems import ParityProblem
from emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions_aggregation import (
    HMRHyperNEAT,
)
from palette_strategies.base_strategy import NUM_ACTIVATIONS, ACTIVATION_NAMES

# Import original strategy classes
from palette_strategies.strategy_78_grn_agg_discovery_dual import GRNAggDiscoveryDualStrategy
from palette_strategies.strategy_38_glial_modulation import GlialModulationStrategy
from palette_strategies.strategy_32_ant_colony_pheromone import AntColonyPheromoneStrategy

OSCILLATORY_INDICES = {4, 11, 12, 13, 15}


def create_rescaled_strategies() -> Dict[str, Any]:
    """Create strategy instances with compressed timescale parameters."""
    return {
        'grn_rescaled': {
            'class': GRNAggDiscoveryDualStrategy,
            'kwargs': {
                'expression_decay': 0.3,           # was 0.9
                'regulation_learning_rate': 0.8,    # was 0.08
                'regulation_decay': 0.80,           # was 0.98
                'hill_coefficient': 4.0,            # was 2.0
            },
            'is_dual': True,
            'description': 'GRN with compressed Tc (~10 gens, was >>100)',
        },
        'glial_rescaled': {
            'class': GlialModulationStrategy,
            'kwargs': {
                'support_decay': 0.90,              # was 0.99
                'support_learning_rate': 1.2,        # was 0.12
                'energy_regen_rate': 1.5,            # was 15.0
                'base_function_cost': 15.0,          # was 1.5
            },
            'is_dual': False,
            'description': 'Glial with compressed Tc (~5 gens, was ~50)',
        },
        'ant_colony_rescaled': {
            'class': AntColonyPheromoneStrategy,
            'kwargs': {
                'pheromone_decay': 0.22,             # was 0.85
                'pheromone_deposit': 1.5,            # was 0.3
                'elite_bonus': 10.0,                 # was 2.0
            },
            'is_dual': False,
            'description': 'Ant Colony with compressed Tc (~5 gens, was ~20)',
        },
    }


STRATEGIES = create_rescaled_strategies()


def run_single_trial(
    strategy_name: str,
    seed: int,
    pop_size: int = 500,
    max_depth: int = 2,
    max_gens: int = 100,
    target_fitness: float = 0.95,
) -> Dict[str, Any]:
    """Run single Parity-4 trial with rescaled strategy."""
    start_time = time.time()

    config = STRATEGIES[strategy_name]
    strategy = config['class'](**config['kwargs'])
    is_dual = config['is_dual']

    problem = ParityProblem(n_bits=4)

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

    # Initialize strategy
    if is_dual:
        initial_act_palette = [0, 1, 2, 3]
        initial_agg_palette = [0, 1]
        strategy_state = strategy.initialize({
            'initial_act_palette': initial_act_palette,
            'initial_agg_palette': initial_agg_palette,
        }, seed)
    else:
        initial_act_palette = [0, 1, 2, 3]
        strategy_state = strategy.initialize({
            'initial_palette': initial_act_palette,
        }, seed)

    current_act_palette = strategy.get_active_palette(strategy_state)

    # Build algo config
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
    state = algo.initialize(neat_config, problem, seed=seed)

    best_fitness = 0.0
    prev_best_fitness = 0.0
    solved_gen = None
    sin_discovered_gen = None

    for gen in range(max_gens):
        current_act_palette = strategy.get_active_palette(strategy_state)
        algo.df_palette = jnp.array(current_act_palette)

        state, metrics = algo.run_generation(state, problem)
        prev_best_fitness = best_fitness
        best_fitness = max(best_fitness, metrics.best_fitness)

        if sin_discovered_gen is None and 4 in current_act_palette:
            sin_discovered_gen = gen + 1

        strategy_state, strategy_metrics = strategy.post_generation_update(
            strategy_state, gen,
            metrics.best_fitness, prev_best_fitness,
            population_data={'fitnesses': [metrics.best_fitness]}
        )

        if metrics.best_fitness >= target_fitness and solved_gen is None:
            solved_gen = gen + 1

    final_palette = strategy.get_active_palette(strategy_state)
    final_agg_palette = (strategy.get_active_agg_palette(strategy_state)
                         if is_dual and hasattr(strategy, 'get_active_agg_palette')
                         else [0, 1])
    osc_funcs = [idx for idx in final_palette if idx in OSCILLATORY_INDICES]

    elapsed = time.time() - start_time

    return {
        'strategy': strategy_name,
        'problem': 'parity_4',
        'seed': seed,
        'solved': solved_gen is not None,
        'solved_gen': solved_gen,
        'best_fitness': float(best_fitness),
        'sin_discovered_gen': sin_discovered_gen,
        'sin_in_palette': 4 in final_palette,
        'has_oscillatory': len(osc_funcs) > 0,
        'oscillatory_functions': osc_funcs,
        'final_act_palette': list(final_palette),
        'final_agg_palette': list(final_agg_palette),
        'elapsed_seconds': elapsed,
        'rescaled_params': STRATEGIES[strategy_name]['kwargs'],
    }


def _save_results(filepath: Path, strategy: str, results: list):
    """Save results incrementally."""
    valid = [r for r in results if 'solved' in r]
    solved = sum(1 for r in valid if r['solved'])
    n = len(valid)
    output = {
        'strategy': strategy,
        'problem': 'parity_4',
        'experiment': 'P1b_timescale_rescaling',
        'description': STRATEGIES[strategy]['description'],
        'rescaled_params': STRATEGIES[strategy]['kwargs'],
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'n': n,
            'solved': solved,
            'solve_rate': solved / n if n > 0 else 0,
        },
        'results': results,
    }
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description='P1b: Timescale Rescaling Validation'
    )
    parser.add_argument('--strategy', type=str, required=True,
                        choices=list(STRATEGIES.keys()),
                        help='Rescaled strategy to run')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds (default: 30)')
    parser.add_argument('--pilot', action='store_true',
                        help='Pilot test (N=3)')
    parser.add_argument('--pop-size', type=int, default=500,
                        help='Population size (default: 500)')
    parser.add_argument('--max-gens', type=int, default=100,
                        help='Max generations (default: 100)')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Max EMR depth (default: 2)')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[2] / 'results' / 'timescale_rescaling'
    os.makedirs(output_dir, exist_ok=True)

    result_file = output_dir / f'{args.strategy}.json'

    if args.pilot:
        seeds = [42, 43, 44]
        print(f"PILOT MODE: {args.strategy}, N=3")
    else:
        seeds = [42 + i for i in range(args.seeds)]

    # Resume support
    results = []
    existing_seeds = set()
    if result_file.exists():
        with open(result_file) as f:
            existing_data = json.load(f)
        results = existing_data.get('results', [])
        existing_seeds = {r['seed'] for r in results}
        remaining = [s for s in seeds if s not in existing_seeds]
        if not remaining:
            print(f"All {len(seeds)} seeds completed for {args.strategy}. Skipping.")
            return
        print(f"Resuming: {len(existing_seeds)} done, {len(remaining)} remaining")
        seeds = remaining
    else:
        print(f"Starting fresh: {args.strategy}, N={len(seeds)}")

    config = STRATEGIES[args.strategy]
    print(f"\n{'='*60}")
    print(f"TIMESCALE RESCALING: {args.strategy}")
    print(f"Description: {config['description']}")
    print(f"Rescaled params: {config['kwargs']}")
    print(f"Problem: Parity-4, pop={args.pop_size}, depth={args.max_depth}, gens={args.max_gens}")
    print(f"Target fitness: 0.95")
    print(f"Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
    print(f"{'='*60}\n")

    start_time = time.time()

    for seed in seeds:
        print(f"  Seed {seed}: ", end='', flush=True)
        try:
            result = run_single_trial(
                strategy_name=args.strategy,
                seed=seed,
                pop_size=args.pop_size,
                max_depth=args.max_depth,
                max_gens=args.max_gens,
            )
            results.append(result)

            status = f"gen {result['solved_gen']}" if result['solved'] else f"FAIL ({result['best_fitness']:.3f})"
            osc = "OSC" if result['has_oscillatory'] else "no-osc"
            sin = "SIN" if result['sin_in_palette'] else "no-sin"
            print(f"{status} [{osc},{sin}] ({result['elapsed_seconds']:.1f}s)")

            _save_results(result_file, args.strategy, results)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'strategy': args.strategy,
                'seed': seed,
                'error': str(e),
            })

    # Summary
    valid = [r for r in results if 'solved' in r]
    n = len(valid)
    solved = sum(1 for r in valid if r['solved'])
    solved_gens = [r['solved_gen'] for r in valid if r['solved']]
    osc_count = sum(1 for r in valid if r.get('has_oscillatory', False))
    sin_count = sum(1 for r in valid if r.get('sin_in_palette', False))

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.strategy} on Parity-4 (RESCALED)")
    print(f"  Solve rate: {solved}/{n} ({solved/n*100:.1f}%)")
    if solved_gens:
        print(f"  Median gen: {np.median(solved_gens):.1f}")
        print(f"  Mean gen: {np.mean(solved_gens):.1f}")
        print(f"  Range: {min(solved_gens)}-{max(solved_gens)}")
    print(f"  Sin discovery: {sin_count}/{n} ({sin_count/n*100:.1f}%)")
    print(f"  Oscillatory presence: {osc_count}/{n} ({osc_count/n*100:.1f}%)")
    print(f"  Total runtime: {(time.time() - start_time)/60:.1f} min")
    print(f"  Results: {result_file}")


if __name__ == '__main__':
    main()
