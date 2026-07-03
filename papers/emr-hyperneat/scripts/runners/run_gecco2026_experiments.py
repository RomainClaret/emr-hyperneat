#!/usr/bin/env python3
"""
Run all experiments for GECCO 2026 HMR-HyperNEAT Unified Extended paper.

This script runs:
1. Caching ablation study (no cache, JAX cache, H→H cache, both)
2. Multi-hop expansion study (iteration_level 1-4)
3. Recurrence preset comparison (feedforward through full_recurrent)
4. Depth scaling study (depths 3-6)

Usage:
    python papers/emr-hyperneat/scripts/runners/run_gecco2026_experiments.py --experiment all
    python papers/emr-hyperneat/scripts/runners/run_gecco2026_experiments.py --experiment caching
    python papers/emr-hyperneat/scripts/runners/run_gecco2026_experiments.py --experiment presets

Output:
    papers/hmr-hyperneat/data/gecco2026_results.json
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

# Configure environment BEFORE importing JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import numpy as np

# Add paths

import jax
import jax.numpy as jnp


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_type: str
    config: Dict[str, Any]
    seed: int
    time_per_gen: float
    generations_to_solve: int
    best_fitness: float
    solved: bool
    total_time: float
    warmup_time: float
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


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


def clear_jax_cache():
    """Clear JAX compilation cache."""
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '/tmp/jax_cache')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"  Cleared JAX cache at {cache_dir}")


def run_single_experiment(
    preset: str,
    depth: int,
    seed: int,
    population: int = 500,
    max_generations: int = 50,
    fitness_threshold: float = 0.98,
    hh_cache_enabled: bool = True,
    iteration_level: int = 2,
    use_jax_cache: bool = True,
) -> ExperimentResult:
    """Run a single experiment with specified configuration."""

    from emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended import (
        HMRHyperNEATUnifiedExtended,
    )

    # XOR problem definition
    input_coords = [(-1.0, -1.0), (1.0, -1.0), (0.0, -0.5)]  # x1, x2, bias
    output_coords = [(0.0, 1.0)]

    # HMR-HyperNEAT config with recurrence settings
    hmr_config = {
        'initial_depth': 0,
        'max_depth': depth,
        'variance_threshold': 0.03,
        'recurrence': {
            'preset': preset,
            'iteration_level': iteration_level,
            'hh_cache_enabled': hh_cache_enabled,
            'hh_refresh_interval': 5,
            'hh_mask_change_threshold': 0.1,
        },
    }

    config_dict = {
        'algorithm_params': {
            'hmrhyperneat': {
                'population_size': population,
                'hmr_hyperneat': hmr_config,
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
    warmup_start = time.time()
    warmup_algo = HMRHyperNEATUnifiedExtended()
    warmup_config = warmup_algo.create_config(config_dict)
    warmup_state = warmup_algo.initialize(warmup_config, problem, seed=seed)
    warmup_state, _ = warmup_algo.run_generation_verbose(warmup_state, problem)
    warmup_time = time.time() - warmup_start

    # Actual experiment
    algo = HMRHyperNEATUnifiedExtended()
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
                experiment_type="xor",
                config={
                    'preset': preset,
                    'depth': depth,
                    'population': population,
                    'hh_cache_enabled': hh_cache_enabled,
                    'iteration_level': iteration_level,
                    'use_jax_cache': use_jax_cache,
                },
                seed=seed,
                time_per_gen=np.mean(gen_times[1:]) if len(gen_times) > 1 else gen_times[0],
                generations_to_solve=gen,
                best_fitness=best_fitness,
                solved=True,
                total_time=total_time,
                warmup_time=warmup_time,
                extra_metrics={
                    'cache_refresh_count': getattr(metrics, 'cache_refresh_count', 0),
                }
            )

    total_time = time.time() - start_time
    return ExperimentResult(
        experiment_type="xor",
        config={
            'preset': preset,
            'depth': depth,
            'population': population,
            'hh_cache_enabled': hh_cache_enabled,
            'iteration_level': iteration_level,
            'use_jax_cache': use_jax_cache,
        },
        seed=seed,
        time_per_gen=np.mean(gen_times[1:]) if len(gen_times) > 1 else gen_times[0],
        generations_to_solve=max_generations,
        best_fitness=best_fitness,
        solved=False,
        total_time=total_time,
        warmup_time=warmup_time,
        extra_metrics={
            'cache_refresh_count': 0,
        }
    )


def run_caching_ablation(
    depth: int = 4,
    population: int = 500,
    n_seeds: int = 5,
    output_dir: Path = None
) -> List[ExperimentResult]:
    """Run caching ablation study."""
    print("\n" + "="*70)
    print("CACHING ABLATION STUDY")
    print("="*70)

    results = []

    # Test configurations: (use_jax_cache, hh_cache_enabled, label)
    configs = [
        (False, False, "no_cache"),
        (True, False, "jax_only"),
        (False, True, "hh_only"),
        (True, True, "both"),
    ]

    for preset in ['feedforward', 'hidden_only', 'full_recurrent']:
        print(f"\n### Preset: {preset} ###")

        for use_jax, use_hh, label in configs:
            print(f"\n  Config: {label}")

            # Clear JAX cache if not using it
            if not use_jax:
                clear_jax_cache()

            for seed in range(42, 42 + n_seeds):
                print(f"    Seed {seed}...", end=" ", flush=True)

                try:
                    result = run_single_experiment(
                        preset=preset,
                        depth=depth,
                        seed=seed,
                        population=population,
                        hh_cache_enabled=use_hh,
                        use_jax_cache=use_jax,
                    )
                    result.config['cache_config'] = label
                    results.append(result)

                    status = "SOLVED" if result.solved else "UNSOLVED"
                    print(f"{status} gen={result.generations_to_solve}, "
                          f"time/gen={result.time_per_gen:.3f}s")

                except Exception as e:
                    print(f"ERROR: {e}")

    return results


def run_multihop_study(
    depth: int = 4,
    population: int = 500,
    n_seeds: int = 5,
) -> List[ExperimentResult]:
    """Run multi-hop expansion study."""
    print("\n" + "="*70)
    print("MULTI-HOP EXPANSION STUDY")
    print("="*70)

    results = []

    for iteration_level in [1, 2, 3, 4]:
        print(f"\n### iteration_level: {iteration_level} ###")

        for seed in range(42, 42 + n_seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)

            try:
                result = run_single_experiment(
                    preset='hidden_only',
                    depth=depth,
                    seed=seed,
                    population=population,
                    iteration_level=iteration_level,
                )
                results.append(result)

                status = "SOLVED" if result.solved else "UNSOLVED"
                print(f"{status} gen={result.generations_to_solve}, "
                      f"fitness={result.best_fitness:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")

    return results


def run_preset_comparison(
    depth: int = 4,
    population: int = 500,
    n_seeds: int = 10,
) -> List[ExperimentResult]:
    """Run recurrence preset comparison."""
    print("\n" + "="*70)
    print("RECURRENCE PRESET COMPARISON")
    print("="*70)

    results = []
    presets = ['feedforward', 'hidden_only', 'with_backward', 'with_lateral',
               'with_self', 'full_recurrent']

    for preset in presets:
        print(f"\n### Preset: {preset} ###")

        for seed in range(42, 42 + n_seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)

            try:
                result = run_single_experiment(
                    preset=preset,
                    depth=depth,
                    seed=seed,
                    population=population,
                )
                results.append(result)

                status = "SOLVED" if result.solved else "UNSOLVED"
                print(f"{status} gen={result.generations_to_solve}, "
                      f"time/gen={result.time_per_gen:.3f}s")

            except Exception as e:
                print(f"ERROR: {e}")

    return results


def run_depth_scaling(
    population: int = 500,
    n_seeds: int = 5,
) -> List[ExperimentResult]:
    """Run depth scaling study."""
    print("\n" + "="*70)
    print("DEPTH SCALING STUDY")
    print("="*70)

    results = []

    for depth in [3, 4, 5, 6]:
        print(f"\n### Depth: {depth} ###")

        for seed in range(42, 42 + n_seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)

            try:
                result = run_single_experiment(
                    preset='hidden_only',
                    depth=depth,
                    seed=seed,
                    population=population,
                )
                results.append(result)

                status = "SOLVED" if result.solved else "UNSOLVED"
                print(f"{status} time/gen={result.time_per_gen:.3f}s, "
                      f"fitness={result.best_fitness:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")

    return results


def compute_summary_statistics(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Compute summary statistics from results."""
    summary = {}

    # Group by experiment type and key config values
    groups = {}
    for r in results:
        key = (r.config.get('preset', 'unknown'),
               r.config.get('depth', 0),
               r.config.get('cache_config', 'default'),
               r.config.get('iteration_level', 2))
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for key, group_results in groups.items():
        preset, depth, cache_config, iter_level = key

        times = [r.time_per_gen for r in group_results]
        gens = [r.generations_to_solve for r in group_results]
        solved = sum(1 for r in group_results if r.solved)

        summary[f"{preset}_d{depth}_{cache_config}_i{iter_level}"] = {
            'n_runs': len(group_results),
            'solve_rate': solved / len(group_results) if group_results else 0,
            'time_per_gen_mean': np.mean(times) if times else 0,
            'time_per_gen_std': np.std(times) if times else 0,
            'generations_mean': np.mean(gens) if gens else 0,
            'generations_std': np.std(gens) if gens else 0,
        }

    return summary


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run GECCO 2026 experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'caching', 'multihop', 'presets', 'depth'],
                       help='Which experiment to run')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of seeds per configuration')
    parser.add_argument('--population', type=int, default=500,
                       help='Population size')
    parser.add_argument('--depth', type=int, default=4,
                       help='Default depth for experiments')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 2 seeds')

    args = parser.parse_args()

    if args.quick:
        args.seeds = 2

    output_dir = Path(__file__).resolve().parents[2] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GECCO 2026 HMR-HyperNEAT Unified Extended Experiments")
    print("="*70)
    print(f"JAX devices: {jax.devices()}")
    print(f"Experiment: {args.experiment}")
    print(f"Seeds: {args.seeds}")
    print(f"Population: {args.population}")
    print("="*70)

    all_results = []

    if args.experiment in ['all', 'presets']:
        results = run_preset_comparison(
            depth=args.depth,
            population=args.population,
            n_seeds=args.seeds,
        )
        all_results.extend(results)

    if args.experiment in ['all', 'caching']:
        results = run_caching_ablation(
            depth=args.depth,
            population=args.population,
            n_seeds=args.seeds,
        )
        all_results.extend(results)

    if args.experiment in ['all', 'multihop']:
        results = run_multihop_study(
            depth=args.depth,
            population=args.population,
            n_seeds=args.seeds,
        )
        all_results.extend(results)

    if args.experiment in ['all', 'depth']:
        results = run_depth_scaling(
            population=args.population,
            n_seeds=args.seeds,
        )
        all_results.extend(results)

    # Compute summary
    summary = compute_summary_statistics(all_results)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'experiment': args.experiment,
            'seeds': args.seeds,
            'population': args.population,
            'depth': args.depth,
            'devices': [str(d) for d in jax.devices()],
        },
        'results': [asdict(r) for r in all_results],
        'summary': summary,
    }

    output_file = output_dir / f"gecco2026_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"Total experiments: {len(all_results)}")
    print("="*70)

    # Print summary table
    print("\n### Summary ###")
    for key, stats in summary.items():
        print(f"  {key}: solve_rate={stats['solve_rate']:.1%}, "
              f"time/gen={stats['time_per_gen_mean']:.3f}±{stats['time_per_gen_std']:.3f}s")


if __name__ == '__main__':
    main()
