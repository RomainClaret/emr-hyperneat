#!/usr/bin/env python3
"""E-S2: Parity-3 Multi-Task Neuromodulation.

Bridges the arity gap between 2-input (100%) and 4-input (13.3%) to
understand whether scaling is gradual or cliff-like.

Tasks (3-input, 8 patterns each):
  - Parity-3: output 1 if odd number of 1s
  - AND-3: output 1 iff all 3 inputs = 1
  - OR-3: output 1 iff any input = 1

Conditions (4 total, 120 runs):
  1. 1layer_3task_uniform_tanh, baseline (Parity-3 should fail)
  2. 1layer_3task_pertask, sin for Parity-3, tanh for AND-3/OR-3
  3. 2layer_3task_uniform_tanh, depth control
  4. 1layer_3task_uniform_sin, oscillatory control

Architecture: MLP + (mu+lambda)-ES (consistent with E22/E25 higher-arity)
N=30 seeds, 200 generations, Pop=750, N_IN=3, N_HIDDEN=15.

Results saved to: papers/emr-neuromodulation/results/parity3_multitask/

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_parity3_multitask.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_parity3_multitask.py \
        --seeds 3 --conditions 1layer_3task_uniform_tanh 1layer_3task_pertask  # smoke test
    python papers/emr-neuromodulation/scripts/runners/benchmark_parity3_multitask.py --summary
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from itertools import product as iterproduct

# ============================================================================
# Constants
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'parity3_multitask'

N_IN = 3
N_HIDDEN = 15  # Scaled between 10 (2-input) and 20 (4-input)
N_OUT = 1
MODULATION_STRENGTH = 5.0

# Generate all 3-input patterns (8 rows)
INPUTS = jnp.array(list(iterproduct([0.0, 1.0], repeat=3)))  # (8, 3)

TASK_NAMES = ['parity3', 'and3', 'or3']


# Truth tables for 3-input tasks
def _parity3_targets():
    """Parity-3: output 1 if odd number of 1s among 3 inputs."""
    targets = []
    for bits in iterproduct([0.0, 1.0], repeat=3):
        targets.append(float(sum(bits) % 2))
    return jnp.array(targets)


def _and3_targets():
    """AND-3: output 1 iff all 3 inputs = 1."""
    targets = []
    for bits in iterproduct([0.0, 1.0], repeat=3):
        targets.append(float(all(b == 1.0 for b in bits)))
    return jnp.array(targets)


def _or3_targets():
    """OR-3: output 1 iff any input = 1."""
    targets = []
    for bits in iterproduct([0.0, 1.0], repeat=3):
        targets.append(float(any(b == 1.0 for b in bits)))
    return jnp.array(targets)


TARGETS = {
    'parity3': _parity3_targets(),
    'and3': _and3_targets(),
    'or3': _or3_targets(),
}

# NT vectors: reuse Schema B profiles mapped to 3-input task types
NT_VECTORS = {
    'parity3': jnp.array([0.95, 0.05, 0.95, 1.0]),  # XOR/parity profile
    'and3':    jnp.array([0.10, 0.90, 0.10, 1.0]),   # AND profile
    'or3':     jnp.array([0.50, 0.50, 0.50, 1.0]),   # OR profile
}

# Per-task activation: parity-3 needs oscillatory
PERTASK_ACTIVATIONS = {
    'parity3': 'sin',
    'and3': 'tanh',
    'or3': 'tanh',
}

ALL_CONDITIONS = [
    '1layer_3task_uniform_tanh',
    '1layer_3task_pertask',
    '2layer_3task_uniform_tanh',
    '1layer_3task_uniform_sin',
]


# ============================================================================
# Parameter initialization
# ============================================================================

def _count_layers(condition: str) -> int:
    return int(condition[0])


def _get_activation_mode(condition: str) -> str:
    if 'pertask' in condition:
        return 'pertask'
    elif 'uniform_sin' in condition:
        return 'uniform_sin'
    else:
        return 'uniform_tanh'


def init_params(key: jax.Array, n_layers: int) -> Dict[str, jnp.ndarray]:
    """Initialize parameters for neuromodulated MLP with 3-input."""
    params = {}
    scale = 0.5

    n_keys = 4 * n_layers + 2
    keys = jax.random.split(key, n_keys)
    ki = 0

    for l in range(n_layers):
        in_dim = N_IN if l == 0 else N_HIDDEN
        params[f'W{l+1}'] = jax.random.normal(keys[ki], (in_dim, N_HIDDEN)) * scale
        ki += 1
        params[f'b{l+1}'] = jax.random.normal(keys[ki], (N_HIDDEN,)) * 0.1
        ki += 1
        params[f'R{l+1}'] = jax.random.normal(keys[ki], (N_HIDDEN, 3)) * 0.3
        ki += 1
        params[f'g{l+1}'] = jnp.ones((N_HIDDEN,)) + jax.random.normal(keys[ki], (N_HIDDEN,)) * 0.1
        ki += 1

    params['W_out'] = jax.random.normal(keys[ki], (N_HIDDEN, N_OUT)) * scale
    ki += 1
    params['b_out'] = jax.random.normal(keys[ki], (N_OUT,)) * 0.1

    return params


def init_population(key: jax.Array, pop_size: int, n_layers: int) -> Dict[str, jnp.ndarray]:
    keys = jax.random.split(key, pop_size)
    return jax.vmap(lambda k: init_params(k, n_layers))(keys)


# ============================================================================
# Forward pass
# ============================================================================

def _forward_neuromod(params: Dict, inputs: jnp.ndarray, nt: jnp.ndarray,
                      n_layers: int, activation_fn,
                      s: float = MODULATION_STRENGTH) -> jnp.ndarray:
    h = inputs

    for l in range(n_layers):
        W = params[f'W{l+1}']
        b = params[f'b{l+1}']
        R = params[f'R{l+1}']
        g = params[f'g{l+1}']

        mod = R @ nt[:3]
        g_eff = g + s * mod
        g_eff = jnp.clip(g_eff, 0.1, 5.0)
        gates = jax.nn.sigmoid(mod)
        mod_bias = mod * s

        pre_h = h @ W + b
        h = activation_fn(g_eff * pre_h + mod_bias) * gates

    output = jax.nn.sigmoid(h @ params['W_out'] + params['b_out'])
    invert = nt[3]
    output = invert * output + (1.0 - invert) * (1.0 - output)

    return output


_FORWARD_CACHE = {}


def get_forward_fn(n_layers: int, activation_name: str):
    key = (n_layers, activation_name)
    if key not in _FORWARD_CACHE:
        act_fn = {'sin': jnp.sin, 'tanh': jnp.tanh}[activation_name]
        @jax.jit
        def _forward(params, inputs, nt):
            return _forward_neuromod(params, inputs, nt, n_layers, act_fn)
        _FORWARD_CACHE[key] = _forward
    return _FORWARD_CACHE[key]


# ============================================================================
# Evaluation
# ============================================================================

def eval_single_multitask(params: Dict, n_layers: int,
                          activation_mode: str) -> Tuple[float, Dict[str, float]]:
    per_task = {}
    for task in TASK_NAMES:
        nt = NT_VECTORS[task]
        targets = TARGETS[task]

        if activation_mode == 'pertask':
            act_name = PERTASK_ACTIVATIONS[task]
        elif activation_mode == 'uniform_sin':
            act_name = 'sin'
        else:
            act_name = 'tanh'

        forward_fn = get_forward_fn(n_layers, act_name)
        output = forward_fn(params, INPUTS, nt)
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
    condition: str,
    seed: int,
    pop_size: int = 750,
    generations: int = 200,
    mu_frac: float = 0.1,
    sigma: float = 0.3,
    success_threshold: float = 0.95,
    verbose: bool = True,
) -> Dict:
    start_time = time.time()
    n_layers = _count_layers(condition)
    activation_mode = _get_activation_mode(condition)
    mu = max(1, int(pop_size * mu_frac))

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    pop = init_population(init_key, pop_size, n_layers)

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
            prod_fit, pt = eval_single_multitask(ind_params, n_layers, activation_mode)
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

        # Selection
        top_indices = jnp.argsort(fitnesses)[-mu:]
        parents = jax.tree.map(lambda x: x[top_indices], pop)

        # Reproduction
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

    sample_params = init_params(jax.random.PRNGKey(0), n_layers)
    n_params = sum(p.size for p in jax.tree.leaves(sample_params))

    return {
        'condition': condition,
        'n_layers': n_layers,
        'activation_mode': activation_mode,
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


def summarize_condition(results_dir: Path, prefix: str):
    files = sorted(results_dir.glob(f'{prefix}_seed*.json'))
    if not files:
        print(f"  {prefix}: no results")
        return

    converged = 0
    gens = []
    min_accs = []
    parity_accs = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if data.get('converged', False):
            converged += 1
            if data.get('convergence_gen') is not None:
                gens.append(data['convergence_gen'])
        min_accs.append(data.get('min_task_accuracy', 0.0))
        pt = data.get('per_task_fitness', {})
        if 'parity3' in pt:
            parity_accs.append(pt['parity3'])

    total = len(files)
    rate = 100 * converged / total if total > 0 else 0
    print(f"  {prefix}: {converged}/{total} ({rate:.1f}%)", end='')
    if gens:
        print(f" | median gen {np.median(gens):.0f} [{min(gens)}-{max(gens)}]", end='')
    if parity_accs:
        print(f" | Parity-3 solved: {sum(1 for a in parity_accs if a >= 0.95)}/{len(parity_accs)}", end='')
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S2: Parity-3 multi-task neuromodulation')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds (default: 30)')
    parser.add_argument('--conditions', nargs='+', default=None,
                        choices=ALL_CONDITIONS,
                        help='Conditions to run (default: all)')
    parser.add_argument('--pop-size', type=int, default=750,
                        help='ES population size (default: 750)')
    parser.add_argument('--generations', type=int, default=200,
                        help='Max generations (default: 200)')
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='ES mutation sigma (default: 0.3)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of existing results')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    conditions = args.conditions if args.conditions else ALL_CONDITIONS

    if args.summary:
        print("\n=== E-S2 Parity-3 Results Summary ===\n")
        for cond in ALL_CONDITIONS:
            summarize_condition(RESULTS_DIR, cond)
        return

    total_start = time.time()

    for condition in conditions:
        n_layers = _count_layers(condition)
        act_mode = _get_activation_mode(condition)

        print(f"\n{'='*60}")
        print(f"Condition: {condition} ({args.seeds} seeds)")
        print(f"  Layers: {n_layers}, Activation: {act_mode}")
        print(f"  Tasks: {TASK_NAMES} (3-input, 8 patterns each)")
        print(f"  N_HIDDEN: {N_HIDDEN}, Pop: {args.pop_size}")
        print(f"{'='*60}")

        for seed in range(args.seeds):
            if seed == 71:
                print(f"  Skip seed 71 (known OOM)")
                continue

            fname = RESULTS_DIR / f'{condition}_seed{seed}.json'
            if result_exists(fname):
                print(f"  Skip existing: {fname.name}")
                continue

            print(f"\n  Seed {seed}:")
            try:
                result = run_es(
                    condition=condition,
                    seed=seed,
                    pop_size=args.pop_size,
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

        print(f"\n--- Summary for {condition} ---")
        summarize_condition(RESULTS_DIR, condition)

    total_runtime = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    print(f"{'='*60}")

    print("\n=== Final Summary ===\n")
    for cond in ALL_CONDITIONS:
        summarize_condition(RESULTS_DIR, cond)


if __name__ == '__main__':
    main()
