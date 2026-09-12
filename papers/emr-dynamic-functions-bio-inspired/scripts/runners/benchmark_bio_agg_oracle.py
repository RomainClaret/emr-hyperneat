#!/usr/bin/env python3
"""Aggregation Oracle: Can monotonic activations + min aggregation solve parity without oscillatory?

Tests whether the aggregation taxonomy finding (min critical for XOR: 20%->100%) can
substitute for oscillatory activations on parity tasks. If monotonic activations +
min aggregation solves Parity-4, the "universal oscillatory requirement" has a loophole.

Conditions:
    1. monotonic_min: [tanh, sigmoid, relu, identity] + min aggregation (global)
    2. bandpass_integrate_min: [band_pass, integrate] + min aggregation (global)

Both use FIXED palettes (no palette evolution, no aggregation evolution).

Design:
    - Problem: Parity-4 (pop=500, 100 gens, depth=2, feedforward)
    - Seeds: 42-71 (N=30 each)
    - Target fitness: 0.95
    - Aggregation: global 'min' (all nodes use min, no evolution)
    - Activation: fixed palette via cppn_output mode (no palette evolution)

Usage:
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_agg_oracle.py --condition monotonic_min
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_agg_oracle.py --condition bandpass_integrate_min
    python papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_agg_oracle.py --condition monotonic_min --pilot
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
from palette_strategies.base_strategy import ACTIVATION_NAMES, AGGREGATION_NAMES

OSCILLATORY_INDICES = {4, 11, 12, 13, 15}

# Condition configurations
# Activation indices: tanh=0, sigmoid=1, relu=2, identity=3, band_pass=16, integrate=17
# Aggregation: 'min' = index 3 in AGGREGATION_LIST ['sum','mean','max','min','product','maxabs']
CONDITIONS = {
    'monotonic_min': {
        'act_palette': [0, 1, 2, 3],   # tanh, sigmoid, relu, identity
        'agg_function': 'min',
        'description': 'All monotonic activations + global min aggregation',
        'output_filename': 'monotonic_min.json',
    },
    'bandpass_integrate_min': {
        'act_palette': [16, 17],        # band_pass, integrate
        'agg_function': 'min',
        'description': 'band_pass + integrate activations + global min aggregation',
        'output_filename': 'bandpass_integrate_min.json',
    },
}


def run_single_trial(
    condition_name: str,
    seed: int,
    pop_size: int = 500,
    max_depth: int = 2,
    max_gens: int = 100,
    target_fitness: float = 0.95,
) -> Dict[str, Any]:
    """Run single Parity-4 trial with fixed palette and global min aggregation."""
    start_time = time.time()

    config = CONDITIONS[condition_name]
    act_palette = config['act_palette']
    agg_function = config['agg_function']

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

    # Build algo config: fixed activation palette + global min aggregation
    algo_config = {
        'algorithm_params': {
            'hmrhyperneat': {
                'hmr_hyperneat': {
                    'initial_depth': 0,
                    'max_depth': max_depth,
                    'variance_threshold': 0.03,
                    'dynamic_functions': {
                        'mode': 'cppn_output',
                        'palette': act_palette,
                        'palette_evolution': {'enabled': False},
                        'aggregation': {
                            'mode': 'global',
                            'global_function': agg_function,
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
    solved_gen = None
    fitness_history = []

    for gen in range(max_gens):
        state, metrics = algo.run_generation(state, problem)
        best_fitness = max(best_fitness, metrics.best_fitness)
        fitness_history.append(float(metrics.best_fitness))

        if metrics.best_fitness >= target_fitness and solved_gen is None:
            solved_gen = gen + 1

    elapsed = time.time() - start_time

    # Check if any oscillatory functions are in the palette
    osc_in_palette = [idx for idx in act_palette if idx in OSCILLATORY_INDICES]

    return {
        'condition': condition_name,
        'problem': 'parity_4',
        'seed': seed,
        'solved': solved_gen is not None,
        'solved_gen': solved_gen,
        'best_fitness': float(best_fitness),
        'has_oscillatory': len(osc_in_palette) > 0,
        'oscillatory_in_palette': osc_in_palette,
        'act_palette': act_palette,
        'act_palette_names': [ACTIVATION_NAMES[i] for i in act_palette],
        'agg_function': agg_function,
        'elapsed_seconds': elapsed,
        'fitness_history': fitness_history,
    }


def _save_results(filepath: Path, condition_name: str, results: list):
    """Save results incrementally."""
    config = CONDITIONS[condition_name]
    valid = [r for r in results if 'solved' in r]
    solved = sum(1 for r in valid if r['solved'])
    n = len(valid)

    # Wilson binomial CI
    if n > 0:
        z = 1.96
        p_hat = solved / n
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
        ci_low = max(0, center - margin)
        ci_high = min(1, center + margin)
    else:
        ci_low, ci_high = 0, 0

    solved_gens = [r['solved_gen'] for r in valid if r['solved']]

    output = {
        'condition': condition_name,
        'problem': 'parity_4',
        'experiment': 'agg_oracle',
        'description': config['description'],
        'act_palette': config['act_palette'],
        'act_palette_names': [ACTIVATION_NAMES[i] for i in config['act_palette']],
        'agg_function': config['agg_function'],
        'config': {
            'pop_size': 500,
            'max_depth': 2,
            'max_gens': 100,
            'target_fitness': 0.95,
            'n_bits': 4,
            'palette_evolution': False,
            'aggregation_evolution': False,
        },
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'n': n,
            'solved': solved,
            'solve_rate': solved / n if n > 0 else 0,
            'ci_95': [ci_low, ci_high],
            'median_gen': float(np.median(solved_gens)) if solved_gens else None,
            'mean_gen': float(np.mean(solved_gens)) if solved_gens else None,
            'min_gen': int(min(solved_gens)) if solved_gens else None,
            'max_gen': int(max(solved_gens)) if solved_gens else None,
        },
        'results': results,
    }
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregation Oracle: monotonic activations + min aggregation on Parity-4'
    )
    parser.add_argument('--condition', type=str, required=True,
                        choices=list(CONDITIONS.keys()),
                        help='Condition to run')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds (default: 30)')
    parser.add_argument('--pilot', action='store_true',
                        help='Pilot test (N=3, seeds 42-44)')
    parser.add_argument('--pop-size', type=int, default=500,
                        help='Population size (default: 500)')
    parser.add_argument('--max-gens', type=int, default=100,
                        help='Max generations (default: 100)')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Max EMR depth (default: 2)')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[2] / 'results' / 'agg_oracle'
    os.makedirs(output_dir, exist_ok=True)

    condition_config = CONDITIONS[args.condition]
    result_file = output_dir / condition_config['output_filename']

    if args.pilot:
        seeds = [42, 43, 44]
        print(f"PILOT MODE: {args.condition}, N=3")
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
            print(f"All {len(seeds)} seeds completed for {args.condition}. Skipping.")
            return
        print(f"Resuming: {len(existing_seeds)} done, {len(remaining)} remaining")
        seeds = remaining
    else:
        print(f"Starting fresh: {args.condition}, N={len(seeds)}")

    act_names = [ACTIVATION_NAMES[i] for i in condition_config['act_palette']]

    print(f"\n{'='*60}")
    print(f"AGGREGATION ORACLE: {args.condition}")
    print(f"Description: {condition_config['description']}")
    print(f"Activation palette: {condition_config['act_palette']} = {act_names}")
    print(f"Aggregation: global '{condition_config['agg_function']}' (fixed, no evolution)")
    print(f"Problem: Parity-4, pop={args.pop_size}, depth={args.max_depth}, gens={args.max_gens}")
    print(f"Target fitness: 0.95")
    print(f"Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
    print(f"{'='*60}\n")

    start_time = time.time()

    for seed in seeds:
        print(f"  Seed {seed}: ", end='', flush=True)
        try:
            result = run_single_trial(
                condition_name=args.condition,
                seed=seed,
                pop_size=args.pop_size,
                max_depth=args.max_depth,
                max_gens=args.max_gens,
            )
            results.append(result)

            status = f"gen {result['solved_gen']}" if result['solved'] else f"FAIL ({result['best_fitness']:.3f})"
            print(f"{status} ({result['elapsed_seconds']:.1f}s)")

            _save_results(result_file, args.condition, results)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'condition': args.condition,
                'seed': seed,
                'error': str(e),
            })

    # Summary
    valid = [r for r in results if 'solved' in r]
    n = len(valid)
    solved = sum(1 for r in valid if r['solved'])
    solved_gens = [r['solved_gen'] for r in valid if r['solved']]

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.condition} on Parity-4")
    print(f"  Activations: {act_names}")
    print(f"  Aggregation: {condition_config['agg_function']} (global, fixed)")
    print(f"  Solve rate: {solved}/{n} ({solved/n*100:.1f}%)")
    if solved_gens:
        print(f"  Median gen: {np.median(solved_gens):.1f}")
        print(f"  Mean gen: {np.mean(solved_gens):.1f}")
        print(f"  Range: {min(solved_gens)}-{max(solved_gens)}")
    print(f"  Total runtime: {(time.time() - start_time)/60:.1f} min")
    print(f"  Results: {result_file}")


if __name__ == '__main__':
    main()
