#!/usr/bin/env python3
"""
Run statistical validation experiments for the EMR-HyperNEAT paper (GECCO 2026).

This script runs 30+ seeds at key depths (3, 4, 5) to generate
statistically valid performance data for the paper.

From release 0.2.0 this runner runs EMR-HyperNEAT. The published runs used the HMR engine's base
module, which release 0.1.0 ships: from the repository root,
`python scripts/replicate.py run emr-hyperneat --release 0.1.0 --` followed by a command of
Usage without its `papers/emr-hyperneat/` prefix runs the original on 0.1.0. The published file,
data/statistical_validation_results.json, comes from --quick (depths 3 and 4, five seeds). The
EMR version runs the experiment as it ran, on EMR's defaults:
    - Population 1000, feed-forward XOR with a bias input, tanh hidden and sigmoid output nodes,
      up to 100 generations, threshold 0.98, initial_depth 0 (the HMR module ignored
      initial_depth; 0 forces nothing).
    - HMR's level-0 variance rule and random key stream, as the published runs had them:
      USE_HMR_RULE_AND_SPLIT in emr_settings.py pins variance_rule legacy and
      extra_randkey_split (False gives EMR's defaults).
    - run_xor_experiment is the single run (run_hmr_xor_experiment in release 0.1.0).
    - Results go to results/statistical_validation_emr/, apart from the published file in data/
      (the data release).

Usage:
    python papers/emr-hyperneat/scripts/runners/run_statistical_validation.py

Output:
    papers/emr-hyperneat/results/statistical_validation_emr/statistical_validation_results.json
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Configure JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import numpy as np

# Add paths

import jax
import jax.numpy as jnp

from emr_hyperneat.emrhyperneat import EMRHyperNEAT
from emr_settings import hmr_rule_and_split


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    depth: int
    seed: int
    time_per_gen: float
    generations_to_solve: int
    best_fitness: float
    solved: bool
    total_time: float
    jit_warmup_time: float


class XORProblem:
    """Simple XOR problem wrapper."""
    def __init__(self):
        self.input_shape = (3,)  # x1, x2, bias
        self.output_shape = (1,)
        self.fitness_threshold = 0.98
        self._data = [
            ([0.0, 0.0, 1.0], [0.0]),
            ([0.0, 1.0, 1.0], [1.0]),
            ([1.0, 0.0, 1.0], [1.0]),
            ([1.0, 1.0, 1.0], [0.0]),
        ]

    def get_data(self):
        return self._data


def run_xor_experiment(
    depth: int,
    seed: int,
    population: int = 1000,
    max_generations: int = 100,
    fitness_threshold: float = 0.98
) -> ExperimentResult:
    """Run single EMR-HyperNEAT XOR experiment."""

    # XOR problem definition
    input_coords = [(-1.0, -1.0), (1.0, -1.0), (0.0, -0.5)]  # x1, x2, bias
    output_coords = [(0.0, 1.0)]

    # EMR-HyperNEAT config, with HMR's level-0 rule and key split, which the published runs used
    # and emr_settings pins.
    emr_config = {
        'initial_depth': 0,
        'max_depth': depth,
        'variance_threshold': 0.03,
        **hmr_rule_and_split(),
    }

    config_dict = {
        'algorithm_params': {
            'emrhyperneat': {
                'population_size': population,
                'emr_hyperneat': emr_config,
                'substrate': {
                    'input_coords': input_coords,
                    'output_coords': output_coords,
                    'output_activation': 'sigmoid',
                    'hidden_activation': 'tanh',
                },
            }
        }
    }

    # Create problem
    problem = XORProblem()

    # Warmup run for JIT compilation
    warmup_algo = EMRHyperNEAT()
    warmup_config = warmup_algo.create_config(config_dict)
    warmup_start = time.time()
    warmup_state = warmup_algo.initialize(warmup_config, problem, seed=seed)
    warmup_state, _ = warmup_algo.run_generation_verbose(warmup_state, problem)
    jit_warmup_time = time.time() - warmup_start

    # Actual experiment
    algo = EMRHyperNEAT()
    config = algo.create_config(config_dict)
    state = algo.initialize(config, problem, seed=seed)

    start_time = time.time()
    gen_times = []
    best_fitness = 0.0

    for gen in range(1, max_generations + 1):
        gen_start = time.time()
        state, metrics = algo.run_generation_verbose(state, problem)
        gen_times.append(time.time() - gen_start)

        current_fitness = float(metrics.best_fitness)
        if current_fitness > best_fitness:
            best_fitness = current_fitness

        if best_fitness >= fitness_threshold:
            total_time = time.time() - start_time
            return ExperimentResult(
                depth=depth,
                seed=seed,
                time_per_gen=np.mean(gen_times[1:]) if len(gen_times) > 1 else gen_times[0],
                generations_to_solve=gen,
                best_fitness=best_fitness,
                solved=True,
                total_time=total_time,
                jit_warmup_time=jit_warmup_time
            )

    total_time = time.time() - start_time
    return ExperimentResult(
        depth=depth,
        seed=seed,
        time_per_gen=np.mean(gen_times[1:]) if len(gen_times) > 1 else gen_times[0],
        generations_to_solve=max_generations,
        best_fitness=best_fitness,
        solved=False,
        total_time=total_time,
        jit_warmup_time=jit_warmup_time
    )


def run_validation_suite(
    depths: List[int] = [3, 4, 5],
    n_seeds: int = 30,
    population: int = 1000,
    output_dir: str = None
) -> Dict[str, Any]:
    """Run full statistical validation suite."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "results" / "statistical_validation_emr"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EMR-HyperNEAT Statistical Validation Suite (EMR version)")
    print("=" * 70)
    print(f"Depths: {depths}")
    print(f"Seeds per depth: {n_seeds}")
    print(f"Population: {population}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 70)

    all_results = {
        'config': {
            'depths': depths,
            'n_seeds': n_seeds,
            'population': population,
            'timestamp': datetime.now().isoformat(),
            'devices': [str(d) for d in jax.devices()]
        },
        'results': [],
        'summary': {}
    }

    for depth in depths:
        print(f"\n### Depth {depth} ({n_seeds} seeds) ###")
        depth_results = []

        for i, seed in enumerate(range(42, 42 + n_seeds)):
            print(f"  [{i+1}/{n_seeds}] Seed {seed}...", end=" ", flush=True)

            try:
                result = run_xor_experiment(
                    depth=depth,
                    seed=seed,
                    population=population
                )
                depth_results.append(asdict(result))

                status = "SOLVED" if result.solved else "UNSOLVED"
                print(f"{status} gen={result.generations_to_solve}, "
                      f"fitness={result.best_fitness:.4f}, "
                      f"time/gen={result.time_per_gen:.3f}s")
            except Exception as e:
                print(f"ERROR: {e}")
                depth_results.append({
                    'depth': depth,
                    'seed': seed,
                    'error': str(e)
                })

        all_results['results'].extend(depth_results)

        # Compute summary statistics for this depth
        valid_results = [r for r in depth_results if 'error' not in r]
        if valid_results:
            times = [r['time_per_gen'] for r in valid_results]
            gens = [r['generations_to_solve'] for r in valid_results]
            solved = sum(1 for r in valid_results if r['solved'])

            all_results['summary'][f'depth_{depth}'] = {
                'n_runs': len(valid_results),
                'solve_rate': solved / len(valid_results),
                'time_per_gen_mean': np.mean(times),
                'time_per_gen_std': np.std(times),
                'time_per_gen_median': np.median(times),
                'generations_mean': np.mean(gens),
                'generations_std': np.std(gens),
            }

            print(f"\n  Summary: {solved}/{len(valid_results)} solved, "
                  f"time/gen={np.mean(times):.3f}±{np.std(times):.3f}s")

    # Save results
    output_file = output_dir / "statistical_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return all_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run EMR-HyperNEAT statistical validation')
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 4, 5],
                       help='Depths to test (default: 3 4 5)')
    parser.add_argument('--seeds', type=int, default=30,
                       help='Number of seeds per depth (default: 30)')
    parser.add_argument('--population', type=int, default=1000,
                       help='Population size (default: 1000)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 5 seeds')

    args = parser.parse_args()

    if args.quick:
        args.seeds = 5
        args.depths = [3, 4]

    run_validation_suite(
        depths=args.depths,
        n_seeds=args.seeds,
        population=args.population
    )


if __name__ == '__main__':
    main()
