#!/usr/bin/env python3
"""E-S11: Expanded Random NT Sets, Tighten CI on random vector success rate.

Current data: 10 random NT sets (with inversion) give ~10% set-level success
with wide 95% CI [0.3%, 44.5%]. This experiment adds 20 new random sets
(indices 10-29) to bring the total to 30 for a tighter binomial CI.

Design:
  - 20 new random NT sets (indices 10-29), each with 30 seeds = 600 runs
  - XOR NT always fixed at [0.95, 0.05, 0.95, 1.0]
  - AND and OR: DA, 5HT, NE drawn from U(0,1), ACh=1.0
  - NAND = same DA/5HT/NE as AND, ACh=0.0 (Schema B inversion)
  - NOR = same DA/5HT/NE as OR, ACh=0.0 (Schema B inversion)
  - Deterministic seeding: np.random.RandomState(42 + set_index)
  - CPPN-based architecture: Pop=750, 100 gen, product agg, >=98% threshold

Post-hoc: compute OR-NOR Euclidean distance and direction angle for each set.

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_expanded_random_nt.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_expanded_random_nt.py --summary
    python papers/emr-neuromodulation/scripts/runners/benchmark_expanded_random_nt.py --set-indices 10 11 12
    python papers/emr-neuromodulation/scripts/runners/benchmark_expanded_random_nt.py --seeds 1 2 3
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'experiments' / 'neuromodulation'))

from multihead_palette_neuromodulation import run_multihead_palette_experiment

# Lazy JAX import to avoid startup cost during --summary
_jnp = None
def jnp():
    global _jnp
    if _jnp is None:
        import jax.numpy as _j
        _jnp = _j
    return _jnp


# ============================================================================
# Constants
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'expanded_random_nt'
OLD_RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'strengthening' / 'random_nt'
SEEDS = list(range(1, 31))
NEW_SET_INDICES = list(range(10, 30))

# Fixed XOR NT vector (same across all sets)
XOR_NT = [0.95, 0.05, 0.95, 1.0]


# ============================================================================
# NT vector generation
# ============================================================================

def generate_random_nt_set(set_index: int) -> Dict[str, List[float]]:
    """Generate a random NT vector set deterministically.

    XOR is always fixed. AND and OR get random DA/5HT/NE from U(0,1).
    NAND mirrors AND with ACh=0.0, NOR mirrors OR with ACh=0.0 (Schema B).

    Args:
        set_index: Index for deterministic seeding (seed = 42 + set_index).

    Returns:
        Dict mapping task names to 4D NT vectors [DA, 5HT, NE, ACh].
    """
    rng = np.random.RandomState(42 + set_index)

    # Random AND vector: DA, 5HT, NE from U(0,1), ACh=1.0
    and_da, and_5ht, and_ne = rng.uniform(0, 1, 3).tolist()

    # Random OR vector: DA, 5HT, NE from U(0,1), ACh=1.0
    or_da, or_5ht, or_ne = rng.uniform(0, 1, 3).tolist()

    return {
        'xor':  XOR_NT,
        'and':  [and_da,  and_5ht,  and_ne,  1.0],
        'or':   [or_da,   or_5ht,   or_ne,   1.0],
        'nand': [and_da,  and_5ht,  and_ne,  0.0],  # AND + ACh inversion
        'nor':  [or_da,   or_5ht,   or_ne,   0.0],  # OR + ACh inversion
    }


def compute_or_nor_geometry(nt_vectors: Dict[str, List[float]]) -> Tuple[float, float]:
    """Compute Euclidean distance and direction angle between OR and NOR vectors.

    Distance is computed on DA/5HT/NE dimensions only (first 3).
    Direction angle is the angle of the OR->NOR difference vector in the
    DA-NE plane (most informative for Schema B geometry).

    Args:
        nt_vectors: Dict mapping task names to 4D NT vectors.

    Returns:
        Tuple of (euclidean_distance, direction_angle_degrees).
    """
    or_vec = np.array(nt_vectors['or'][:3])
    nor_vec = np.array(nt_vectors['nor'][:3])
    diff = nor_vec - or_vec
    distance = float(np.linalg.norm(diff))

    # Direction angle in DA-NE plane (atan2 of NE_diff vs DA_diff)
    angle_rad = np.arctan2(diff[2], diff[0])  # NE vs DA
    angle_deg = float(np.degrees(angle_rad))

    return distance, angle_deg


# ============================================================================
# Runner
# ============================================================================

def make_profiles(nt_vectors: Dict[str, List[float]]) -> Dict:
    """Convert raw NT vectors to JAX arrays for the experiment runner."""
    return {task: jnp().array(vec) for task, vec in nt_vectors.items()}


def run_set(set_index: int, seeds: List[int] = SEEDS,
            verbose: bool = True) -> Dict:
    """Run a single random NT set across all seeds."""
    nt_vectors = generate_random_nt_set(set_index)
    profiles = make_profiles(nt_vectors)
    or_nor_dist, or_nor_angle = compute_or_nor_geometry(nt_vectors)

    result_file = RESULTS_DIR / f'random_set_{set_index}.json'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing_seeds = set()
    existing_results = []
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)
        existing_seeds = {r['seed'] for r in existing.get('results', [])}
        existing_results = existing.get('results', [])
        if verbose:
            print(f"  Resuming set {set_index}: {len(existing_seeds)}/{len(seeds)} done")

    results = list(existing_results)

    for seed in seeds:
        if seed in existing_seeds:
            continue

        if verbose:
            print(f"  [set_{set_index}] seed={seed}", end=' ', flush=True)

        result = run_multihead_palette_experiment(
            task_names=['xor', 'and', 'or', 'nand', 'nor'],
            palette_mode='uniform',
            blend_mode='fixed',
            aggregation='product',
            seed=seed,
            generations=100,
            population=750,
            max_depth=4,
            success_threshold=0.98,
            verbose=False,
            nt_profiles=profiles,
        )

        result_dict = {
            'seed': seed,
            'converged': result.converged,
            'convergence_gen': result.convergence_gen,
            'individual_min_fitness': result.individual_min_fitness,
            'per_task_fitness': result.per_task_fitness,
            'runtime_seconds': result.runtime_seconds,
        }
        results.append(result_dict)

        if verbose:
            status = "\u2713" if result.converged else "\u2717"
            gen_str = (f"gen {result.convergence_gen}" if result.converged
                       else f"min={result.individual_min_fitness:.3f}")
            print(f"{status} {gen_str} ({result.runtime_seconds:.0f}s)")

        # Incremental save
        _save_set(result_file, set_index, nt_vectors, or_nor_dist,
                  or_nor_angle, results)

    return _summarize_set(set_index, nt_vectors, or_nor_dist,
                          or_nor_angle, results)


def _save_set(path: Path, set_index: int, nt_vectors: Dict,
              or_nor_dist: float, or_nor_angle: float,
              results: List[Dict]):
    """Save set results to JSON."""
    n = len(results)
    conv = sum(1 for r in results if r['converged'])
    with open(path, 'w') as f:
        json.dump({
            'set_index': set_index,
            'nt_vectors': nt_vectors,
            'or_nor_distance': round(or_nor_dist, 4),
            'or_nor_direction_angle': round(or_nor_angle, 1),
            'n_seeds': n,
            'convergence_count': conv,
            'convergence_rate': round(conv / n, 4) if n > 0 else 0.0,
            'results': results,
        }, f, indent=2)


def _summarize_set(set_index: int, nt_vectors: Dict,
                   or_nor_dist: float, or_nor_angle: float,
                   results: List[Dict]) -> Dict:
    """Summarize a set's results."""
    n = len(results)
    conv = sum(1 for r in results if r['converged'])
    conv_gens = [r['convergence_gen'] for r in results if r['converged']]
    return {
        'set_index': set_index,
        'or_nor_distance': or_nor_dist,
        'or_nor_direction_angle': or_nor_angle,
        'n_seeds': n,
        'convergence_count': conv,
        'convergence_rate': conv / n if n > 0 else 0.0,
        'median_gen': float(np.median(conv_gens)) if conv_gens else None,
    }


# ============================================================================
# Summary (loads existing data for sets 0-9 from old results)
# ============================================================================

def load_old_set_results(set_index: int) -> Optional[Dict]:
    """Load results for sets 0-9 from the old strengthening experiment.

    Old format: individual per-seed JSONs named random_with_inv_set{i}_seed{s}.json
    """
    files = sorted(OLD_RESULTS_DIR.glob(f'random_with_inv_set{set_index}_seed*.json'))
    if not files:
        return None

    results = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        results.append({
            'seed': d.get('seed', 0),
            'converged': d.get('converged', False),
            'convergence_gen': d.get('convergence_gen'),
            'individual_min_fitness': d.get('individual_min_fitness', 0),
            'per_task_fitness': d.get('per_task_fitness', {}),
            'runtime_seconds': d.get('runtime_seconds', 0),
        })

    # Extract NT vectors from first file
    nt_vectors = None
    with open(files[0]) as fh:
        first = json.load(fh)
    if 'nt_vectors' in first:
        nt_vectors = first['nt_vectors']

    n = len(results)
    conv = sum(1 for r in results if r['converged'])
    conv_gens = [r['convergence_gen'] for r in results if r['converged']]

    or_nor_dist = None
    or_nor_angle = None
    if nt_vectors:
        or_nor_dist, or_nor_angle = compute_or_nor_geometry(nt_vectors)

    return {
        'set_index': set_index,
        'source': 'old_strengthening',
        'nt_vectors': nt_vectors,
        'or_nor_distance': or_nor_dist,
        'or_nor_direction_angle': or_nor_angle,
        'n_seeds': n,
        'convergence_count': conv,
        'convergence_rate': conv / n if n > 0 else 0.0,
        'median_gen': float(np.median(conv_gens)) if conv_gens else None,
    }


def load_new_set_results(set_index: int) -> Optional[Dict]:
    """Load results for sets 10-29 from the new expanded experiment."""
    result_file = RESULTS_DIR / f'random_set_{set_index}.json'
    if not result_file.exists():
        return None

    with open(result_file) as f:
        data = json.load(f)

    conv_gens = [r['convergence_gen'] for r in data['results'] if r['converged']]

    return {
        'set_index': data['set_index'],
        'source': 'expanded',
        'nt_vectors': data.get('nt_vectors'),
        'or_nor_distance': data.get('or_nor_distance'),
        'or_nor_direction_angle': data.get('or_nor_direction_angle'),
        'n_seeds': data['n_seeds'],
        'convergence_count': data['convergence_count'],
        'convergence_rate': data['convergence_rate'],
        'median_gen': float(np.median(conv_gens)) if conv_gens else None,
    }


def print_summary():
    """Print summary of all 30 sets (10 old + 20 new)."""
    print("=" * 90)
    print("E-S11: Expanded Random NT Sets — Results Summary")
    print("  Old sets 0-9: from strengthening/random_nt/ (with inversion)")
    print("  New sets 10-29: from expanded_random_nt/ (Schema B, XOR fixed)")
    print("=" * 90)

    all_sets = []

    # Load old sets 0-9
    for i in range(10):
        data = load_old_set_results(i)
        if data:
            all_sets.append(data)

    # Load new sets 10-29
    for i in range(10, 30):
        data = load_new_set_results(i)
        if data:
            all_sets.append(data)

    if not all_sets:
        print("\n  No results found.")
        return

    print(f"\n  {'Set':>4} {'Source':<8} {'Rate':>10} {'Conv':>6} {'N':>4} "
          f"{'Med Gen':>8} {'OR-NOR dist':>12} {'OR-NOR angle':>13}")
    print(f"  {'-' * 80}")

    total_conv = 0
    total_n = 0
    set_level_success = 0
    set_level_total = 0

    for data in sorted(all_sets, key=lambda x: x['set_index']):
        i = data['set_index']
        src = data.get('source', '?')[:7]
        n = data['n_seeds']
        conv = data['convergence_count']
        rate = f"{conv/n*100:.1f}%" if n > 0 else 'N/A'
        med = f"{data['median_gen']:.0f}" if data['median_gen'] is not None else '-'
        dist = f"{data['or_nor_distance']:.3f}" if data['or_nor_distance'] is not None else '?'
        angle = f"{data['or_nor_direction_angle']:.1f}" if data['or_nor_direction_angle'] is not None else '?'

        marker = ""
        if n >= 30 and conv == n:
            marker = " ***"
        elif n >= 30 and conv == 0:
            marker = ""

        print(f"  {i:>4} {src:<8} {rate:>10} {conv:>4}/{n:<4} "
              f"{med:>8} {dist:>12} {angle:>12}{marker}")

        total_conv += conv
        total_n += n
        if n >= 30:
            set_level_total += 1
            if conv > 0:
                set_level_success += 1

    # Overall statistics
    print(f"\n  {'='*80}")
    overall_rate = total_conv / total_n * 100 if total_n > 0 else 0
    print(f"  Overall seed-level: {total_conv}/{total_n} ({overall_rate:.1f}%)")

    if set_level_total > 0:
        set_rate = set_level_success / set_level_total * 100
        print(f"  Set-level success (any convergence): {set_level_success}/{set_level_total} "
              f"({set_rate:.1f}%)")

        # Wilson score interval for set-level success
        from scipy import stats as scipy_stats
        n_s = set_level_total
        k_s = set_level_success
        p_hat = k_s / n_s
        z = 1.96  # 95% CI
        denom = 1 + z**2 / n_s
        center = (p_hat + z**2 / (2 * n_s)) / denom
        width = z * np.sqrt(p_hat * (1 - p_hat) / n_s + z**2 / (4 * n_s**2)) / denom
        ci_low = max(0, center - width) * 100
        ci_high = min(1, center + width) * 100
        print(f"  Set-level 95% Wilson CI: [{ci_low:.1f}%, {ci_high:.1f}%]")

        # Count 100% sets
        perfect_sets = sum(1 for d in all_sets
                          if d['n_seeds'] >= 30 and d['convergence_count'] == d['n_seeds'])
        print(f"  Sets with 100% success: {perfect_sets}/{set_level_total}")

    # Geometry correlation analysis
    complete_sets = [d for d in all_sets
                     if d['n_seeds'] >= 30 and d.get('or_nor_distance') is not None]
    if len(complete_sets) >= 5:
        dists = [d['or_nor_distance'] for d in complete_sets]
        rates = [d['convergence_rate'] for d in complete_sets]
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(dists, rates)
        print(f"\n  OR-NOR distance vs convergence rate:")
        print(f"    Spearman rho = {rho:.3f}, p = {p_val:.4f}")

    print(f"\n  Total runs across all sets: {total_n}")
    new_runs = sum(d['n_seeds'] for d in all_sets if d.get('source') == 'expanded')
    old_runs = sum(d['n_seeds'] for d in all_sets if d.get('source') == 'old_strengthening')
    print(f"    Old (sets 0-9): {old_runs}")
    print(f"    New (sets 10-29): {new_runs}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S11: Expanded Random NT Sets')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of all 30 sets (no JAX import)')
    parser.add_argument('--set-indices', type=int, nargs='+',
                        help='Run specific set indices (default: 10-29)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Run specific seeds (default: 1-30)')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    set_indices = args.set_indices if args.set_indices else NEW_SET_INDICES
    seeds = args.seeds if args.seeds else SEEDS

    # Preview NT vectors for all requested sets
    print("Previewing NT vectors for requested sets...")
    for idx in set_indices:
        nt = generate_random_nt_set(idx)
        dist, angle = compute_or_nor_geometry(nt)
        print(f"  Set {idx}: AND=[{nt['and'][0]:.3f}, {nt['and'][1]:.3f}, {nt['and'][2]:.3f}]  "
              f"OR=[{nt['or'][0]:.3f}, {nt['or'][1]:.3f}, {nt['or'][2]:.3f}]  "
              f"OR-NOR dist={dist:.3f}  angle={angle:.1f}")

    total_runs = len(set_indices) * len(seeds)
    print(f"\n{'=' * 70}")
    print(f"E-S11: Expanded Random NT Sets")
    print(f"Sets: {len(set_indices)} (indices {min(set_indices)}-{max(set_indices)})")
    print(f"Seeds: {len(seeds)} ({min(seeds)}-{max(seeds)})")
    print(f"Total runs: {total_runs}")
    print(f"Architecture: CPPN-based, Pop=750, 100 gen, product agg, >=98%")
    print(f"XOR NT fixed: {XOR_NT}")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'=' * 70}")

    start = time.time()
    summaries = []

    for i, idx in enumerate(set_indices):
        print(f"\n{'#' * 60}")
        print(f"# [{i+1}/{len(set_indices)}] Random NT Set {idx}")
        nt = generate_random_nt_set(idx)
        dist, angle = compute_or_nor_geometry(nt)
        print(f"# OR-NOR distance: {dist:.3f}, angle: {angle:.1f} deg")
        print(f"{'#' * 60}")

        summary = run_set(idx, seeds=seeds)
        summaries.append(summary)

        rate = summary['convergence_rate'] * 100
        med = summary['median_gen']
        med_str = f"median gen {med:.0f}" if med is not None else "no convergence"
        print(f"\n  >>> Set {idx}: {summary['convergence_count']}/{summary['n_seeds']} "
              f"({rate:.1f}%), {med_str}")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"ALL SETS COMPLETE — {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"{'=' * 70}")
    print_summary()


if __name__ == '__main__':
    main()
