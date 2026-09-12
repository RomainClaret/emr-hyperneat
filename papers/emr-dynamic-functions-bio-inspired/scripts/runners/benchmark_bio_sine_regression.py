#!/usr/bin/env python3
"""Sine Regression Benchmark: oscillatory functions are GEOMETRICALLY required.

Tests 8 bio-inspired strategies on sine approximation. Parity requires oscillatory
functions MATHEMATICALLY (symmetry of the decision function); sine regression
requires them GEOMETRICALLY (the target itself is oscillatory).

Design:
    - 8 dual strategies (same as single-task table)
    - Problem: approximate sin(x) over [0, 2π] with 20 training points
    - Input: 1D (normalized x), Output: 1D (normalized sin(x))
    - Config: pop=300, depth=3, max_gens=200, target_fitness=0.95 (R²)
    - Seeds: 42-71 (N=30)

Usage:
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_sine_regression.py --strategy circadian_rhythm_dual
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_sine_regression.py --strategy baseline_dual --pilot
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions_aggregation import (
    HMRHyperNEAT,
)
from palette_strategies.strategy_52_circadian_rhythm_dual import CircadianRhythmDualStrategy
from palette_strategies.strategy_29_clonal_selection_dual import ClonalSelectionDualStrategy
from palette_strategies.strategy_16_stdp_dual import STDPDualStrategy
from palette_strategies.strategy_47_baseline_dual import BaselineDualStrategy
from palette_strategies.strategy_64_critical_period_refined_dual import CriticalPeriodRefinedDualStrategy
from palette_strategies.strategy_8_hebbian_dual import HebbianDualStrategy
from palette_strategies.strategy_15_metaplastic_dual import MetaplasticDualStrategy
from palette_strategies.strategy_63_adult_neurogenesis_dual import AdultNeurogenesisDualStrategy

OSCILLATORY_INDICES = {4, 11, 12, 13, 15}

STRATEGIES = {
    'circadian_rhythm_dual': CircadianRhythmDualStrategy,
    'clonal_selection_dual': ClonalSelectionDualStrategy,
    'stdp_dual': STDPDualStrategy,
    'baseline_dual': BaselineDualStrategy,
    'critical_period_refined_dual': CriticalPeriodRefinedDualStrategy,
    'hebbian_dual': HebbianDualStrategy,
    'metaplastic_dual': MetaplasticDualStrategy,
    'adult_neurogenesis_dual': AdultNeurogenesisDualStrategy,
}


class SineRegressionProblem:
    """Approximate sin(x) over [0, 2π], normalized for 1 - MSE fitness.

    Input: normalized x in [0, 1] (from original x in [0, 2π])
    Target: normalized (sin(x) + 1) / 2 in [0, 1] (from sin(x) in [-1, 1])

    This normalization ensures 1 - MSE fitness is in [0, 1] and compatible
    with the 0.95 target threshold used for classification problems.
    """
    jitable = True
    use_bias = True
    fitness_threshold = 0.95

    def __init__(self, n_points: int = 20):
        self.n_points = n_points
        # Training data
        x_raw = np.linspace(0, 2 * np.pi, n_points)
        y_raw = np.sin(x_raw)

        # Normalize to [0, 1]
        x_norm = (x_raw / (2 * np.pi)).astype(np.float32)
        y_norm = ((y_raw + 1.0) / 2.0).astype(np.float32)

        # Build inputs with bias (1D x + bias)
        self.inputs = [[float(x_norm[i]), 1.0] for i in range(n_points)]
        self.targets = [[float(y_norm[i])] for i in range(n_points)]

        self.input_shape = (2,)  # 1 input + 1 bias
        self.output_shape = (1,)

    def get_data(self):
        return list(zip(self.inputs, self.targets))


def run_single_trial(
    strategy_class,
    seed: int,
    pop_size: int = 300,
    max_depth: int = 3,
    max_gens: int = 200,
    target_fitness: float = 0.95,
) -> Dict[str, Any]:
    """Run single sine regression trial with given strategy."""
    start_time = time.time()

    strategy = strategy_class()
    problem = SineRegressionProblem(n_points=20)

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

    initial_act_palette = [0, 1, 2, 3]
    initial_agg_palette = [0, 1]
    strategy_state = strategy.initialize({
        'initial_act_palette': initial_act_palette,
        'initial_agg_palette': initial_agg_palette,
    }, seed)

    current_act_palette = strategy.get_active_palette(strategy_state)

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

        strategy_state, _ = strategy.post_generation_update(
            strategy_state, gen, metrics.best_fitness, prev_best_fitness,
            population_data={'fitnesses': [metrics.best_fitness]}
        )

        if metrics.best_fitness >= target_fitness and solved_gen is None:
            solved_gen = gen + 1

    final_palette = strategy.get_active_palette(strategy_state)
    final_agg_palette = strategy.get_active_agg_palette(strategy_state)
    osc_funcs = [idx for idx in final_palette if idx in OSCILLATORY_INDICES]

    elapsed = time.time() - start_time

    return {
        'strategy': strategy.name if hasattr(strategy, 'name') else strategy_class.__name__,
        'problem': 'sine_regression',
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
    }


def _save_results(filepath: Path, strategy: str, results: list):
    valid = [r for r in results if 'solved' in r]
    solved = sum(1 for r in valid if r['solved'])
    n = len(valid)
    output = {
        'strategy': strategy,
        'problem': 'sine_regression',
        'experiment': 'sine_regression',
        'description': 'Sine approximation: geometric oscillatory requirement',
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
    parser = argparse.ArgumentParser(description='Sine Regression Benchmark')
    parser.add_argument('--strategy', type=str, required=True, choices=list(STRATEGIES.keys()))
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--pop-size', type=int, default=300)
    parser.add_argument('--max-gens', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--target-fitness', type=float, default=0.95)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[2] / 'results' / 'sine_regression'
    os.makedirs(output_dir, exist_ok=True)

    result_file = output_dir / f'{args.strategy}.json'

    if args.pilot:
        seeds = [42, 43, 44]
        print(f"PILOT MODE: {args.strategy} on sine_regression, N=3")
    else:
        seeds = [42 + i for i in range(args.seeds)]

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
        print(f"Starting fresh: {args.strategy} on sine_regression, N={len(seeds)}")

    print(f"\n{'='*60}")
    print(f"SINE REGRESSION: {args.strategy}")
    print(f"Problem: sin(x) on [0, 2π], 20 points, pop={args.pop_size}, depth={args.max_depth}, gens={args.max_gens}")
    print(f"Target fitness: {args.target_fitness}")
    print(f"Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
    print(f"{'='*60}\n")

    start_time = time.time()

    for seed in seeds:
        print(f"  Seed {seed}: ", end='', flush=True)
        try:
            result = run_single_trial(
                strategy_class=STRATEGIES[args.strategy],
                seed=seed,
                pop_size=args.pop_size,
                max_depth=args.max_depth,
                max_gens=args.max_gens,
                target_fitness=args.target_fitness,
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
            results.append({'strategy': args.strategy, 'seed': seed, 'error': str(e)})

    valid = [r for r in results if 'solved' in r]
    n = len(valid)
    solved = sum(1 for r in valid if r['solved'])
    solved_gens = [r['solved_gen'] for r in valid if r['solved']]
    osc_count = sum(1 for r in valid if r.get('has_oscillatory', False))

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.strategy} on sine_regression")
    print(f"  Solve rate: {solved}/{n} ({solved/n*100:.1f}%)")
    if solved_gens:
        print(f"  Median gen: {np.median(solved_gens):.1f}")
    print(f"  Oscillatory presence: {osc_count}/{n} ({osc_count/n*100:.1f}%)")
    print(f"  Total runtime: {(time.time() - start_time)/60:.1f} min")
    print(f"  Results: {result_file}")


if __name__ == '__main__':
    main()
