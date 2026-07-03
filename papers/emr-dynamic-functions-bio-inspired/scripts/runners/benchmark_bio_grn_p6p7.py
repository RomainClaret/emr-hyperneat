#!/usr/bin/env python3
"""GRN-Rescaled on Parity-6/Parity-7: Does the 0% oscillatory finding scale further?

P4: 30% (9/30), 0% oscillatory
P5: 50% (15/30), 0% oscillatory
P6: ?
P7: ?

Tests whether the non-oscillatory band_pass+integrate pathway has a difficulty ceiling.

Usage:
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_grn_p6p7.py --problem parity_6
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_grn_p6p7.py --problem parity_7 --pilot
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
from palette_strategies.strategy_78_grn_agg_discovery_dual import GRNAggDiscoveryDualStrategy

OSCILLATORY_INDICES = {4, 11, 12, 13, 15}

GRN_RESCALED_KWARGS = {
    'expression_decay': 0.3,
    'regulation_learning_rate': 0.8,
    'regulation_decay': 0.80,
    'hill_coefficient': 4.0,
}


def run_single_trial(
    n_bits: int,
    seed: int,
    pop_size: int = 300,
    max_depth: int = 3,
    max_gens: int = 200,
    target_fitness: float = 0.80,
) -> Dict[str, Any]:
    """Run single Parity-N trial with GRN-rescaled strategy."""
    start_time = time.time()

    strategy = GRNAggDiscoveryDualStrategy(**GRN_RESCALED_KWARGS)
    problem = ParityProblem(n_bits=n_bits)

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
    final_agg_palette = (strategy.get_active_agg_palette(strategy_state)
                         if hasattr(strategy, 'get_active_agg_palette')
                         else [0, 1])
    osc_funcs = [idx for idx in final_palette if idx in OSCILLATORY_INDICES]

    elapsed = time.time() - start_time

    return {
        'strategy': 'grn_rescaled',
        'problem': f'parity_{n_bits}',
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
        'rescaled_params': GRN_RESCALED_KWARGS,
    }


def _save_results(filepath: Path, results: list, n_bits: int, config: dict):
    valid = [r for r in results if 'solved' in r]
    solved = sum(1 for r in valid if r['solved'])
    n = len(valid)
    output = {
        'strategy': 'grn_rescaled',
        'problem': f'parity_{n_bits}',
        'experiment': f'grn_rescaled_parity{n_bits}',
        'description': f'GRN-rescaled (Tc ~10) on Parity-{n_bits}: test non-osc pathway scaling',
        'rescaled_params': GRN_RESCALED_KWARGS,
        'config': config,
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
    parser = argparse.ArgumentParser(description='GRN-Rescaled on Parity-6/7')
    parser.add_argument('--problem', choices=['parity_4', 'parity_5', 'parity_6', 'parity_7', 'parity_8'], required=True)
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--pop-size', type=int, default=300)
    parser.add_argument('--max-gens', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--target-fitness', type=float, default=0.80)
    args = parser.parse_args()

    n_bits = int(args.problem.split('_')[1])
    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[2] / 'results' / 'timescale_rescaling_p6p7'
    os.makedirs(output_dir, exist_ok=True)

    # P4 with rich config gets a distinct filename to avoid overwriting original P4 results
    suffix = f'_richconfig' if n_bits == 4 else ''
    result_file = output_dir / f'grn_rescaled_p{n_bits}{suffix}.json'

    if args.pilot:
        seeds = [42, 43, 44]
        print(f"PILOT MODE: grn_rescaled on {args.problem}, N=3")
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
            print(f"All {len(seeds)} seeds completed. Skipping.")
            return
        print(f"Resuming: {len(existing_seeds)} done, {len(remaining)} remaining")
        seeds = remaining
    else:
        print(f"Starting fresh: grn_rescaled on {args.problem}, N={len(seeds)}")

    config = {
        'pop_size': args.pop_size,
        'max_depth': args.max_depth,
        'max_gens': args.max_gens,
        'target_fitness': args.target_fitness,
        'n_bits': n_bits,
    }

    print(f"\n{'='*60}")
    print(f"GRN-RESCALED ON {args.problem.upper()}")
    print(f"Rescaled params: {GRN_RESCALED_KWARGS}")
    print(f"Config: pop={args.pop_size}, depth={args.max_depth}, gens={args.max_gens}, target={args.target_fitness}")
    print(f"Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
    print(f"{'='*60}\n")

    start_time = time.time()

    for seed in seeds:
        print(f"  Seed {seed}: ", end='', flush=True)
        try:
            result = run_single_trial(
                n_bits=n_bits,
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
            _save_results(result_file, results, n_bits, config)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'strategy': 'grn_rescaled', 'seed': seed, 'error': str(e)})

    valid = [r for r in results if 'solved' in r]
    n = len(valid)
    solved = sum(1 for r in valid if r['solved'])
    solved_gens = [r['solved_gen'] for r in valid if r['solved']]
    osc_count = sum(1 for r in valid if r.get('has_oscillatory', False))

    print(f"\n{'='*60}")
    print(f"SUMMARY: grn_rescaled on {args.problem}")
    print(f"  Solve rate: {solved}/{n} ({solved/n*100:.1f}%)")
    if solved_gens:
        print(f"  Median gen: {np.median(solved_gens):.1f}")
        print(f"  Range: {min(solved_gens)}-{max(solved_gens)}")
    print(f"  Oscillatory presence: {osc_count}/{n} ({osc_count/n*100:.1f}%)")
    print(f"  Total runtime: {(time.time() - start_time)/60:.1f} min")
    print(f"  Results: {result_file}")


if __name__ == '__main__':
    main()
