#!/usr/bin/env python3
"""E-S25: Pop Scaling Ceiling at 3-Input (Uniform Tanh).

Extends E-S10 (Pop=100-1500) with Pop=2000/3000 to test whether the
3-input barrier under uniform tanh has a ceiling or is purely search-limited.

E-S10 results: Pop=100→6.7%, Pop=300→10%, Pop=750→20%, Pop=1500→50%.
Question: Does it continue to ~100% at Pop=3000, or plateau?

Conditions: uniform tanh ONLY
Pop = {2000, 3000} × 30 seeds = 60 runs

Architecture: MLP + (mu+lambda)-ES (matching E-S10 exactly)
3 tasks (Parity-3, AND-3, OR-3), N_IN=3, N_HIDDEN=15, 200 gens.

Results saved to: papers/emr-neuromodulation/results/pop_scaling_3input_ceiling/

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_pop_scaling_3input_ceiling.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_pop_scaling_3input_ceiling.py \
        --seeds 3 --pop-sizes 2000  # smoke test
    python papers/emr-neuromodulation/scripts/runners/benchmark_pop_scaling_3input_ceiling.py --summary
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple
from itertools import product as iterproduct

import numpy as np

import jax
import jax.numpy as jnp

# ============================================================================
# Constants, MUST match E-S10 exactly
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'pop_scaling_3input_ceiling'

N_IN = 3
N_HIDDEN = 15  # Matching E-S10
N_OUT = 1
MODULATION_STRENGTH = 5.0

INPUTS = jnp.array(list(iterproduct([0.0, 1.0], repeat=3)))

TASK_NAMES = ['parity3', 'and3', 'or3']  # 3 tasks, matching E-S10


def _parity3_targets():
    return jnp.array([float(sum(bits) % 2) for bits in iterproduct([0.0, 1.0], repeat=3)])

def _and3_targets():
    return jnp.array([float(all(b == 1.0 for b in bits)) for bits in iterproduct([0.0, 1.0], repeat=3)])

def _or3_targets():
    return jnp.array([float(any(b == 1.0 for b in bits)) for bits in iterproduct([0.0, 1.0], repeat=3)])


TARGETS = {
    'parity3': _parity3_targets(),
    'and3': _and3_targets(),
    'or3': _or3_targets(),
}

NT_VECTORS = {
    'parity3': jnp.array([0.95, 0.05, 0.95, 1.0]),
    'and3':    jnp.array([0.10, 0.90, 0.10, 1.0]),
    'or3':     jnp.array([0.50, 0.50, 0.50, 1.0]),
}

DEFAULT_POP_SIZES = [2000, 3000]


# ============================================================================
# Parameter initialization
# ============================================================================

def init_params(key: jax.Array) -> Dict[str, jnp.ndarray]:
    params = {}
    scale = 0.5
    keys = jax.random.split(key, 6)
    params['W1'] = jax.random.normal(keys[0], (N_IN, N_HIDDEN)) * scale
    params['b1'] = jax.random.normal(keys[1], (N_HIDDEN,)) * 0.1
    params['R1'] = jax.random.normal(keys[2], (N_HIDDEN, 3)) * 0.3
    params['g1'] = jnp.ones((N_HIDDEN,)) + jax.random.normal(keys[3], (N_HIDDEN,)) * 0.1
    params['W_out'] = jax.random.normal(keys[4], (N_HIDDEN, N_OUT)) * scale
    params['b_out'] = jax.random.normal(keys[5], (N_OUT,)) * 0.1
    return params


def init_population(key: jax.Array, pop_size: int) -> Dict[str, jnp.ndarray]:
    keys = jax.random.split(key, pop_size)
    return jax.vmap(init_params)(keys)


# ============================================================================
# Forward pass
# ============================================================================

def _forward_neuromod(params: Dict, inputs: jnp.ndarray, nt: jnp.ndarray,
                      activation_fn, s: float = MODULATION_STRENGTH) -> jnp.ndarray:
    W1 = params['W1']
    b1 = params['b1']
    R = params['R1']
    g = params['g1']

    mod = R @ nt[:3]
    g_eff = g + s * mod
    g_eff = jnp.clip(g_eff, 0.1, 5.0)
    gates = jax.nn.sigmoid(mod)
    mod_bias = mod * s

    pre_h = inputs @ W1 + b1
    h = activation_fn(g_eff * pre_h + mod_bias) * gates

    output = jax.nn.sigmoid(h @ params['W_out'] + params['b_out'])
    invert = nt[3]
    output = invert * output + (1.0 - invert) * (1.0 - output)
    return output


@jax.jit
def forward_tanh(params, inputs, nt):
    return _forward_neuromod(params, inputs, nt, jnp.tanh)


# ============================================================================
# Evaluation
# ============================================================================

def eval_single_multitask(params: Dict) -> Tuple[float, Dict[str, float]]:
    per_task = {}
    for task in TASK_NAMES:
        nt = NT_VECTORS[task]
        targets = TARGETS[task]

        output = forward_tanh(params, INPUTS, nt)
        output = output.squeeze(-1)

        preds = (output > 0.5).astype(jnp.float32)
        acc = float(jnp.mean(jnp.equal(preds, targets)))
        per_task[task] = acc

    product_fitness = 1.0
    for v in per_task.values():
        product_fitness *= v
    return product_fitness, per_task


# ============================================================================
# (mu+lambda)-ES Optimizer
# ============================================================================

def run_es(
    pop_size: int,
    seed: int,
    generations: int = 200,
    mu_frac: float = 0.1,
    sigma: float = 0.3,
    success_threshold: float = 0.95,
    verbose: bool = True,
) -> Dict:
    start_time = time.time()
    mu = max(1, int(pop_size * mu_frac))

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    pop = init_population(init_key, pop_size)

    converged = False
    convergence_gen = None
    best_per_task = None
    best_fitness = 0.0
    fitness_history = []

    for gen in range(generations):
        fitnesses = []
        all_per_task = []

        for i in range(pop_size):
            ind_params = jax.tree.map(lambda x: x[i], pop)
            prod_fit, pt = eval_single_multitask(ind_params)
            fitnesses.append(prod_fit)
            all_per_task.append(pt)

        fitnesses = jnp.array(fitnesses)

        best_idx = int(jnp.argmax(fitnesses))
        best_fitness = float(fitnesses[best_idx])
        best_per_task = all_per_task[best_idx]
        min_acc = min(best_per_task.values())

        fitness_history.append({
            'generation': gen,
            'best_product_fitness': best_fitness,
            'min_task_accuracy': min_acc,
            **best_per_task,
        })

        if all(v >= success_threshold for v in best_per_task.values()):
            converged = True
            convergence_gen = gen
            if verbose:
                print(f"  *** CONVERGED at gen {gen} (min_acc={min_acc:.4f}) ***")
            break

        if verbose and gen % 10 == 0:
            task_str = ' '.join(f"{t}:{best_per_task[t]:.2f}" for t in TASK_NAMES)
            print(f"  Gen {gen:3d}: prod={best_fitness:.4f} min={min_acc:.4f} | {task_str}")

        top_indices = jnp.argsort(fitnesses)[-mu:]
        parents = jax.tree.map(lambda x: x[top_indices], pop)

        offspring_per_parent = pop_size // mu
        key, mut_key = jax.random.split(key)

        def replicate_and_mutate(parent_params, rng):
            keys = jax.random.split(rng, offspring_per_parent)
            def make_offspring(k):
                return jax.tree.map(
                    lambda p: p + sigma * jax.random.normal(k, p.shape),
                    parent_params
                )
            return jax.vmap(make_offspring)(keys)

        parent_keys = jax.random.split(mut_key, mu)
        all_offspring = jax.vmap(replicate_and_mutate)(parents, parent_keys)

        pop = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:])[:pop_size],
            all_offspring
        )

    runtime = time.time() - start_time

    sample_params = init_params(jax.random.PRNGKey(0))
    n_params = sum(p.size for p in jax.tree.leaves(sample_params))

    return {
        'condition': f'tanh_pop{pop_size}',
        'activation_mode': 'uniform_tanh',
        'n_tasks': 3,
        'tasks': TASK_NAMES,
        'n_inputs': N_IN,
        'n_patterns': 8,
        'seed': seed,
        'pop_size': pop_size,
        'mu': mu,
        'sigma': sigma,
        'generations_max': generations,
        'n_hidden': N_HIDDEN,
        'n_params': n_params,
        'modulation_strength': MODULATION_STRENGTH,
        'converged': converged,
        'convergence_gen': convergence_gen,
        'best_product_fitness': best_fitness,
        'min_task_accuracy': min(best_per_task.values()) if best_per_task else 0.0,
        'per_task_fitness': best_per_task,
        'runtime_seconds': runtime,
        'fitness_history': fitness_history,
    }


# ============================================================================
# Utilities
# ============================================================================

def result_exists(filepath: Path) -> bool:
    return filepath.exists() and filepath.stat().st_size > 0


def save_result(result: Dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, default=convert)


def summarize_pop(results_dir: Path, pop_size: int):
    prefix = f'tanh_pop{pop_size}'
    files = sorted(results_dir.glob(f'{prefix}_seed*.json'))
    if not files:
        print(f"  Pop={pop_size}: no results")
        return

    converged = 0
    gens = []
    parity_accs = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if data.get('converged', False):
            converged += 1
            if data.get('convergence_gen') is not None:
                gens.append(data['convergence_gen'])
        pt = data.get('per_task_fitness', {})
        if 'parity3' in pt:
            parity_accs.append(pt['parity3'])

    total = len(files)
    rate = 100 * converged / total if total > 0 else 0
    print(f"  Pop={pop_size:>5}: {converged}/{total} ({rate:.1f}%)", end='')
    if gens:
        print(f" | median gen {np.median(gens):.0f} [{min(gens)}-{max(gens)}]", end='')
    if parity_accs:
        par_solved = sum(1 for a in parity_accs if a >= 0.95)
        print(f" | Parity-3: {par_solved}/{len(parity_accs)}", end='')
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S25: Pop scaling ceiling at 3-input (uniform tanh)')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds (default: 30)')
    parser.add_argument('--pop-sizes', nargs='+', type=int, default=None,
                        help='Population sizes (default: 2000 3000)')
    parser.add_argument('--generations', type=int, default=200,
                        help='Max generations (default: 200)')
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='ES mutation sigma (default: 0.3)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of existing results')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pop_sizes = args.pop_sizes if args.pop_sizes else DEFAULT_POP_SIZES

    if args.summary:
        print("\n=== E-S25 Pop Scaling Ceiling 3-Input Summary ===\n")
        for ps in DEFAULT_POP_SIZES:
            summarize_pop(RESULTS_DIR, ps)
        return

    total_start = time.time()

    for pop_size in pop_sizes:
        print(f"\n{'='*60}")
        print(f"Pop={pop_size}, uniform tanh ({args.seeds} seeds)")
        print(f"  Tasks: {TASK_NAMES} (3-input, 8 patterns each)")
        print(f"  N_HIDDEN: {N_HIDDEN}, mu: {max(1, int(pop_size * 0.1))}")
        print(f"{'='*60}")

        for seed in range(args.seeds):
            if seed == 71:
                print(f"  Skip seed 71 (known OOM)")
                continue

            fname = RESULTS_DIR / f'tanh_pop{pop_size}_seed{seed}.json'
            if result_exists(fname):
                print(f"  Skip existing: {fname.name}")
                continue

            print(f"\n  Seed {seed}:")
            try:
                result = run_es(
                    pop_size=pop_size,
                    seed=seed,
                    generations=args.generations,
                    sigma=args.sigma,
                    verbose=True,
                )

                save_result(result, fname)
                status = (f"gen {result['convergence_gen']}"
                          if result['converged'] else "NOT CONVERGED")
                par_acc = result['per_task_fitness'].get('parity3', 0.0)
                print(f"  -> {status} (min_acc={result['min_task_accuracy']:.4f}, "
                      f"Parity-3={par_acc:.4f}, {result['runtime_seconds']:.1f}s)")
            except Exception as e:
                print(f"  ERROR on seed {seed}: {e}")
                continue

        print(f"\n--- Summary for Pop={pop_size} ---")
        summarize_pop(RESULTS_DIR, pop_size)

    total_runtime = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/3600:.1f} hours)")
    print(f"{'='*60}")

    print("\n=== Final Summary ===\n")
    for ps in DEFAULT_POP_SIZES:
        summarize_pop(RESULTS_DIR, ps)


if __name__ == '__main__':
    main()
