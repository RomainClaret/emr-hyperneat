#!/usr/bin/env python3
"""P2a: N=60 Single-Task Parity-4 Extension.

Extends the core Table 3 comparison from N=30 to N=60 by running seeds 72-101.
Provides independent replication with genuinely new seeds (not subject to
seed determinism concerns). Merging with existing seeds 42-71 gives N=60.

Expected improvements at N=60:
    - Circadian vs baseline significance: p=0.19 -> ~p=0.06
    - Wilson CI for 97% narrows from [83,99] to [89,99]
    - Pairwise tests gain statistical power (addresses Limitation L3)

Design:
    - Strategies: 8 core dual strategies
    - Problem: Parity-4 (pop=500, 100 gens, depth=2, feedforward)
    - Seeds: 72-101 (N=30, NEW seeds only, merge with existing 42-71 for N=60)
    - Target fitness: 0.95
    - Run ONE strategy per invocation (OOM safety)

Usage:
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_p4_n60_extension.py --strategy circadian_rhythm_dual
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_p4_n60_extension.py --strategy baseline_dual
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_p4_n60_extension.py --strategy circadian_rhythm_dual --pilot
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

from classification_problems import ParityProblem
from emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions_aggregation import (
    HMRHyperNEAT,
)
from palette_strategies.base_strategy import NUM_ACTIVATIONS, ACTIVATION_NAMES
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


def run_single_trial(
    strategy_class,
    seed: int,
    pop_size: int = 500,
    max_depth: int = 2,
    max_gens: int = 100,
    target_fitness: float = 0.95,
) -> Dict[str, Any]:
    """Run single Parity-4 trial with given strategy."""
    start_time = time.time()

    strategy = strategy_class()
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

        strategy_state, strategy_metrics = strategy.post_generation_update(
            strategy_state, gen,
            metrics.best_fitness, prev_best_fitness,
            population_data={'fitnesses': [metrics.best_fitness]}
        )

        if metrics.best_fitness >= target_fitness and solved_gen is None:
            solved_gen = gen + 1

    final_palette = strategy.get_active_palette(strategy_state)
    final_agg_palette = strategy.get_active_agg_palette(strategy_state)
    osc_funcs = [idx for idx in final_palette if idx in OSCILLATORY_INDICES]

    elapsed = time.time() - start_time

    return {
        'strategy': strategy.name,
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
    }


def _save_results(filepath: Path, strategy: str, results: list):
    """Save results incrementally."""
    valid = [r for r in results if 'solved' in r]
    solved = sum(1 for r in valid if r['solved'])
    n = len(valid)
    output = {
        'strategy': strategy,
        'problem': 'parity_4',
        'experiment': 'P2a_n60_extension',
        'seed_range': '72-101',
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
        description='P2a: N=60 Single-Task Parity-4 Extension (seeds 72-101)'
    )
    parser.add_argument('--strategy', type=str, required=True,
                        choices=list(STRATEGIES.keys()),
                        help='Strategy to run')
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
    output_dir = Path(__file__).resolve().parents[2] / 'results' / 'single_task_n60'
    os.makedirs(output_dir, exist_ok=True)

    result_file = output_dir / f'{args.strategy}.json'

    if args.pilot:
        seeds = [72, 73, 74]
        print(f"PILOT MODE: {args.strategy}, N=3 (seeds 72-74)")
    else:
        # Seeds 72-101 (genuinely new, independent from existing 42-71)
        seeds = [72 + i for i in range(args.seeds)]

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

    print(f"\n{'='*60}")
    print(f"N=60 EXTENSION: {args.strategy}")
    print(f"Problem: Parity-4, pop={args.pop_size}, depth={args.max_depth}, gens={args.max_gens}")
    print(f"Target fitness: 0.95")
    print(f"Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
    print(f"NOTE: Merge with existing seeds 42-71 for N=60 total")
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
    print(f"SUMMARY: {args.strategy} on Parity-4 (seeds 72-101)")
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
