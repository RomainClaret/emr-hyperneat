#!/usr/bin/env python3
"""Recurrence type decomposition: tanh on Parity-4 with each of 6 recurrence presets.

ALIFE 2026 Paper 1 (E5): Decomposes which type of recurrence is sufficient to
collapse the oscillatory divide. Tests feedforward (0% baseline), hidden_only,
with_backward, with_lateral, with_self, and full_recurrent.

Expected: Some subset is sufficient. Possibly hidden_only alone collapses the divide.

Usage:
    python papers/emr-dynamic-functions/scripts/runners/E5_recurrence_type.py
    python papers/emr-dynamic-functions/scripts/runners/E5_recurrence_type.py --preset hidden_only
    python papers/emr-dynamic-functions/scripts/runners/E5_recurrence_type.py --list
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification_problems import ParityProblem

from emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended_dynamic_functions_full import (
    HMRHyperNEATUnifiedExtendedDynamicFunctions as UnifiedAlgo,
    RECURRENCE_PRESETS,
    ACTIVATION_LIST,
)

# ============================================================================
# Constants
# ============================================================================

OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "results" / "recurrence_type_n30")
SEEDS = list(range(42, 72))  # 30 seeds: 42-71
PRESETS = [
    'feedforward',
    'hidden_only',
    'with_backward',
    'with_lateral',
    'with_self',
    'full_recurrent',
]
TANH_INDEX = ACTIVATION_LIST.index('tanh')

# Problem configuration
N_BITS = 4
POP_SIZE = 500
MAX_GENERATIONS = 300
MAX_DEPTH = 2
TARGET_FITNESS = 0.95


# ============================================================================
# Substrate coordinates
# ============================================================================


def make_input_coords(n_inputs: int) -> List[tuple]:
    """Generate evenly-spaced input coordinates on y=-1.0."""
    if n_inputs == 1:
        return [(0.0, -1.0)]
    return [(-1.0 + 2.0 * i / (n_inputs - 1), -1.0) for i in range(n_inputs)]


# ============================================================================
# Result I/O
# ============================================================================


def result_filepath(preset_name: str) -> str:
    """Return the JSON filepath for a given preset."""
    return os.path.join(OUTPUT_DIR, f"{preset_name}.json")


def load_partial_results(filepath: str) -> Optional[dict]:
    """Load existing partial results from JSON file."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


def get_completed_seeds(data: Optional[dict]) -> set:
    """Extract set of completed seed numbers from result data."""
    if data is None:
        return set()
    return {r['seed'] for r in data.get('results', [])}


def build_result_dict(preset_name: str, results_list: List[dict]) -> Dict[str, Any]:
    """Build the standard result dict with summary statistics."""
    solve_count = sum(1 for r in results_list if r['solved'])
    total = len(results_list)
    solve_rate = solve_count / total if total > 0 else 0.0

    solved_gens = [r['solved_gen'] for r in results_list if r['solved']]
    avg_gen = float(np.mean(solved_gens)) if solved_gens else None
    std_gen = float(np.std(solved_gens)) if len(solved_gens) > 1 else (0.0 if solved_gens else None)
    median_gen = float(np.median(solved_gens)) if solved_gens else None

    best_fitnesses = [r['best_fitness'] for r in results_list]

    return {
        'preset_name': preset_name,
        'n_bits': N_BITS,
        'activation': 'tanh',
        'activation_index': TANH_INDEX,
        'results': sorted(results_list, key=lambda r: r['seed']),
        'summary': {
            'solve_rate': solve_rate,
            'solve_count': solve_count,
            'total_runs': total,
            'avg_gen_to_solve': avg_gen,
            'std_gen_to_solve': std_gen,
            'median_gen_to_solve': median_gen,
            'avg_best_fitness': float(np.mean(best_fitnesses)) if best_fitnesses else None,
            'std_best_fitness': float(np.std(best_fitnesses)) if len(best_fitnesses) > 1 else None,
        },
    }


# ============================================================================
# Single trial runner
# ============================================================================


def run_single_seed(preset_name: str, seed: int) -> Dict[str, Any]:
    """Run a single trial of tanh on Parity-4 with a given recurrence preset."""
    problem = ParityProblem(n_bits=N_BITS)
    n_inputs = problem.input_shape[0]  # 5: 4 bits + bias
    input_coords = make_input_coords(n_inputs)
    output_coords = [(0.0, 1.0)]

    algo_config = {
        'algorithm_params': {
            'hmrhyperneat': {
                'hmr_hyperneat': {
                    'initial_depth': 0,
                    'max_depth': MAX_DEPTH,
                    'variance_threshold': 0.03,
                    'extra_randkey_split': True,
                    'recurrence': {
                        'preset': preset_name,
                    },
                    'dynamic_functions': {
                        'mode': 'global',
                        'hidden_activation': 'tanh',
                        'palette': [TANH_INDEX],
                        'palette_evolution': {'enabled': False},
                    },
                },
                'substrate': {
                    'input_coords': input_coords,
                    'output_coords': output_coords,
                },
                'neat': {
                    'pop_size': POP_SIZE,
                    'species_size': 10,
                },
            }
        }
    }

    algo = UnifiedAlgo()
    neat_config = algo.create_config(algo_config)
    state = algo.initialize(neat_config, problem, seed=seed)

    best_fitness = 0.0
    solved_gen: Optional[int] = None
    start_time = time.time()
    gens_run = 0

    for gen in range(MAX_GENERATIONS):
        state, metrics = algo.run_generation(state, problem)
        best_fitness = max(best_fitness, metrics.best_fitness)
        gens_run = gen + 1

        if metrics.best_fitness >= TARGET_FITNESS and solved_gen is None:
            solved_gen = gen + 1
            break

    elapsed = time.time() - start_time

    return {
        'preset_name': preset_name,
        'activation': 'tanh',
        'activation_index': TANH_INDEX,
        'seed': seed,
        'n_bits': N_BITS,
        'solved': solved_gen is not None,
        'solved_gen': solved_gen,
        'best_fitness': float(best_fitness),
        'generations_run': gens_run,
        'elapsed_seconds': elapsed,
    }


# ============================================================================
# Preset runner with incremental saves
# ============================================================================


def run_preset(preset_name: str) -> Dict[str, Any]:
    """Run all seeds for a single recurrence preset with incremental saves."""
    filepath = result_filepath(preset_name)

    existing = load_partial_results(filepath)
    completed_seeds = get_completed_seeds(existing)
    remaining_seeds = [s for s in SEEDS if s not in completed_seeds]

    if not remaining_seeds:
        print(f"  {preset_name}: all {len(SEEDS)} seeds complete, skipping")
        return existing

    print(f"\n{'='*60}")
    print(f"PRESET: {preset_name}")
    print(f"  Completed: {len(completed_seeds)}/{len(SEEDS)}, remaining: {len(remaining_seeds)}")
    print(f"{'='*60}")

    results_list: List[dict] = existing['results'] if existing is not None else []

    for seed in remaining_seeds:
        print(f"  Seed {seed}: ", end='', flush=True)

        trial_result = run_single_seed(preset_name, seed)
        results_list.append(trial_result)

        if trial_result['solved']:
            print(f"SOLVED @ gen {trial_result['solved_gen']} ({trial_result['elapsed_seconds']:.1f}s)")
        else:
            print(f"FAIL (fit={trial_result['best_fitness']:.4f}, {trial_result['elapsed_seconds']:.1f}s)")

        # Save after EVERY seed for safe resume
        data = build_result_dict(preset_name, results_list)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    return build_result_dict(preset_name, results_list)


# ============================================================================
# Combined results and summary
# ============================================================================


def update_combined_results() -> Dict[str, Any]:
    """Rebuild combined.json from all individual preset JSONs."""
    combined_path = os.path.join(OUTPUT_DIR, "combined.json")

    all_results: Dict[str, Any] = {}
    for preset_name in PRESETS:
        filepath = result_filepath(preset_name)
        existing = load_partial_results(filepath)
        if existing is not None:
            all_results[preset_name] = existing

    combined = {
        'metadata': {
            'experiment': 'recurrence_type_decomposition_n30',
            'purpose': 'ALIFE 2026 Paper 1 (E5) — Recurrence type decomposition with tanh on Parity-4',
            'seeds': SEEDS,
            'max_generations': MAX_GENERATIONS,
            'pop_size': POP_SIZE,
            'max_depth': MAX_DEPTH,
            'activation': 'tanh',
            'activation_mode': 'global',
            'n_bits': N_BITS,
            'target_fitness': TARGET_FITNESS,
            'presets': PRESETS,
        },
        'results': all_results,
    }

    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)

    return combined


def print_summary_table() -> None:
    """Print summary table across all presets."""
    print(f"\n{'='*70}")
    print("RECURRENCE TYPE DECOMPOSITION SUMMARY (tanh on Parity-4, N=30)")
    print(f"{'='*70}")
    print(f"{'Preset':<18} {'Seeds':<8} {'Solve%':<10} {'Median Gen':<12} {'Avg Gen':<10} {'Avg Fit':<10}")
    print("-" * 68)

    for preset_name in PRESETS:
        filepath = result_filepath(preset_name)
        existing = load_partial_results(filepath)
        n_done = len(get_completed_seeds(existing))

        if existing and 'summary' in existing:
            s = existing['summary']
            solve_str = f"{s['solve_rate'] * 100:.0f}%"
            median_str = f"{s['median_gen_to_solve']:.1f}" if s['median_gen_to_solve'] else "Never"
            avg_str = f"{s['avg_gen_to_solve']:.1f}" if s['avg_gen_to_solve'] else "Never"
            fit_str = f"{s['avg_best_fitness']:.4f}" if s['avg_best_fitness'] else "---"
        else:
            solve_str = "---"
            median_str = "---"
            avg_str = "---"
            fit_str = "---"

        status = f"{n_done}/{len(SEEDS)}"
        print(f"{preset_name:<18} {status:<8} {solve_str:<10} {median_str:<12} {avg_str:<10} {fit_str:<10}")


def show_status() -> None:
    """Print completion status for all presets."""
    print("Recurrence type decomposition status (tanh on Parity-4, N=30 target):")
    print(f"{'Preset':<18} {'Seeds':<10} {'Solve%':<10} {'Status'}")
    print("-" * 50)

    total_done = 0
    total_target = len(PRESETS) * len(SEEDS)

    for preset_name in PRESETS:
        filepath = result_filepath(preset_name)
        existing = load_partial_results(filepath)
        n_done = len(get_completed_seeds(existing))
        total_done += n_done
        status = "COMPLETE" if n_done >= len(SEEDS) else f"{n_done}/{len(SEEDS)}"

        solve_str = "---"
        if existing and 'summary' in existing:
            solve_str = f"{existing['summary']['solve_rate'] * 100:.0f}%"

        print(f"{preset_name:<18} {n_done:<10} {solve_str:<10} {status}")

    print(f"\nTotal: {total_done}/{total_target} seeds ({total_done / total_target * 100:.0f}%)")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='E5: Recurrence type decomposition — tanh on Parity-4 with 6 presets (Paper 1, ALIFE 2026)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 6 presets (30 seeds each, 180 total runs)
  python papers/emr-dynamic-functions/scripts/runners/E5_recurrence_type.py

  # Run a specific preset only
  python papers/emr-dynamic-functions/scripts/runners/E5_recurrence_type.py --preset hidden_only

  # Check completion status
  python papers/emr-dynamic-functions/scripts/runners/E5_recurrence_type.py --list
        """,
    )

    parser.add_argument('--preset', type=str, default=None,
                        choices=PRESETS,
                        help='Run a specific preset only (default: all 6)')
    parser.add_argument('--list', action='store_true',
                        help='Show completion status and exit')

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.list:
        show_status()
        return

    # Determine which presets to run
    if args.preset:
        presets_to_run = [args.preset]
    else:
        presets_to_run = PRESETS

    total_runs = len(presets_to_run) * len(SEEDS)

    print("=" * 70)
    print("E5: RECURRENCE TYPE DECOMPOSITION")
    print("=" * 70)
    print(f"Problem:        Parity-{N_BITS} ({2 ** N_BITS} samples, {N_BITS + 1} inputs incl. bias)")
    print(f"Activation:     tanh (index {TANH_INDEX})")
    print(f"Presets:        {presets_to_run}")
    print(f"Seeds:          {len(SEEDS)} ({SEEDS[0]}-{SEEDS[-1]})")
    print(f"Max generations: {MAX_GENERATIONS}")
    print(f"Population:     {POP_SIZE}")
    print(f"Max depth:      {MAX_DEPTH}")
    print(f"Target fitness: {TARGET_FITNESS}")
    print(f"Output:         {OUTPUT_DIR}")
    print(f"Total runs:     {total_runs} (minus already completed)")

    start_time = time.time()

    for preset_name in presets_to_run:
        run_preset(preset_name)
        gc.collect()

    total_time = time.time() - start_time

    # Rebuild combined results and print summary
    update_combined_results()
    print_summary_table()

    print(f"\nTotal runtime: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
