#!/usr/bin/env python3
"""E-S20: Topology x Arity Extended (4-Input and 5-Input).

Extends E-S12 (topology × arity at 3-input) to 4-input and 5-input arities.
E-S12 showed feedforward 20% vs full_recurrent 80% at 3-input with tanh.
This tests whether the recurrence advantage persists/widens at higher arity.

Tasks (N-input):
  - Parity-N: output 1 if odd number of 1s
  - AND-N: output 1 iff all N inputs = 1
  - OR-N: output 1 iff any input = 1

Conditions (4 total, 120 runs):
  1. feedforward_4input, FF, 4-input, uniform tanh
  2. full_recurrent_4input, Full recurrent, 4-input, uniform tanh
  3. feedforward_5input, FF, 5-input, uniform tanh
  4. full_recurrent_5input, Full recurrent, 5-input, uniform tanh

All use uniform tanh activation (the suboptimal condition where topology matters).
N=30 seeds, 200 generations, Pop=750, N_HIDDEN=20.

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_topology_arity_extended.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_topology_arity_extended.py \
        --seeds 3 --conditions feedforward_4input full_recurrent_4input  # smoke test
    python papers/emr-neuromodulation/scripts/runners/benchmark_topology_arity_extended.py --summary

Results saved to: papers/emr-neuromodulation/results/topology_arity_extended/
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'topology_arity_extended'

N_HIDDEN = 20
N_OUT = 1
MODULATION_STRENGTH = 5.0
N_RECURRENT_STEPS = 3

ALL_CONDITIONS = [
    'feedforward_4input',
    'full_recurrent_4input',
    'feedforward_5input',
    'full_recurrent_5input',
]


# ============================================================================
# Condition parsing
# ============================================================================

def _parse_condition(condition: str):
    """Parse condition into (topology, n_in)."""
    if 'feedforward' in condition:
        topology = 'feedforward'
    else:
        topology = 'full_recurrent'

    if '4input' in condition:
        n_in = 4
    else:
        n_in = 5

    return topology, n_in


# ============================================================================
# Task generation (dynamic per arity)
# ============================================================================

def get_task_data(n_in: int):
    """Generate inputs, targets, NT vectors, and task names for given arity."""
    inputs = jnp.array(list(iterproduct([0.0, 1.0], repeat=n_in)))

    def parity_targets():
        targets = []
        for bits in iterproduct([0.0, 1.0], repeat=n_in):
            targets.append(float(sum(bits) % 2))
        return jnp.array(targets)

    def and_targets():
        targets = []
        for bits in iterproduct([0.0, 1.0], repeat=n_in):
            targets.append(float(all(b == 1.0 for b in bits)))
        return jnp.array(targets)

    def or_targets():
        targets = []
        for bits in iterproduct([0.0, 1.0], repeat=n_in):
            targets.append(float(any(b == 1.0 for b in bits)))
        return jnp.array(targets)

    task_names = [f'parity{n_in}', f'and{n_in}', f'or{n_in}']
    targets = {
        f'parity{n_in}': parity_targets(),
        f'and{n_in}': and_targets(),
        f'or{n_in}': or_targets(),
    }
    nt_vectors = {
        f'parity{n_in}': jnp.array([0.95, 0.05, 0.95, 1.0]),
        f'and{n_in}':    jnp.array([0.10, 0.90, 0.10, 1.0]),
        f'or{n_in}':     jnp.array([0.50, 0.50, 0.50, 1.0]),
    }

    return inputs, targets, nt_vectors, task_names


# ============================================================================
# Parameter initialization
# ============================================================================

def init_params(key: jax.Array, topology: str, n_in: int) -> Dict[str, jnp.ndarray]:
    params = {}
    scale = 0.5

    n_keys_needed = 7 if topology == 'full_recurrent' else 6
    keys = jax.random.split(key, n_keys_needed)
    ki = 0

    params['W1'] = jax.random.normal(keys[ki], (n_in, N_HIDDEN)) * scale
    ki += 1
    params['b1'] = jax.random.normal(keys[ki], (N_HIDDEN,)) * 0.1
    ki += 1
    params['R1'] = jax.random.normal(keys[ki], (N_HIDDEN, 3)) * 0.3
    ki += 1
    params['g1'] = jnp.ones((N_HIDDEN,)) + jax.random.normal(keys[ki], (N_HIDDEN,)) * 0.1
    ki += 1

    if topology == 'full_recurrent':
        params['W_lateral'] = jax.random.normal(keys[ki], (N_HIDDEN, N_HIDDEN)) * 0.1
        ki += 1

    params['W_out'] = jax.random.normal(keys[ki], (N_HIDDEN, N_OUT)) * scale
    ki += 1
    params['b_out'] = jax.random.normal(keys[ki], (N_OUT,)) * 0.1

    return params


def init_population(key: jax.Array, pop_size: int, topology: str, n_in: int) -> Dict[str, jnp.ndarray]:
    keys = jax.random.split(key, pop_size)
    return jax.vmap(lambda k: init_params(k, topology, n_in))(keys)


# ============================================================================
# Forward pass
# ============================================================================

def _forward_neuromod(params: Dict, inputs: jnp.ndarray, nt: jnp.ndarray,
                      topology: str,
                      s: float = MODULATION_STRENGTH) -> jnp.ndarray:
    W1 = params['W1']
    b1 = params['b1']
    R1 = params['R1']
    g1 = params['g1']

    mod = R1 @ nt[:3]
    g_eff = g1 + s * mod
    g_eff = jnp.clip(g_eff, 0.1, 5.0)
    gates = jax.nn.sigmoid(mod)
    mod_bias = mod * s

    if topology == 'feedforward':
        pre_h = inputs @ W1 + b1
        h = jnp.tanh(g_eff * pre_h + mod_bias) * gates
    else:
        W_lateral = params['W_lateral']
        pre_h = inputs @ W1 + b1
        h = jnp.tanh(g_eff * pre_h + mod_bias) * gates
        for _ in range(N_RECURRENT_STEPS):
            pre_h = inputs @ W1 + h @ W_lateral + b1
            h = jnp.tanh(g_eff * pre_h + mod_bias) * gates

    output = jax.nn.sigmoid(h @ params['W_out'] + params['b_out'])

    invert = nt[3]
    output = invert * output + (1.0 - invert) * (1.0 - output)

    return output


_FORWARD_CACHE = {}

def get_forward_fn(topology: str):
    if topology not in _FORWARD_CACHE:
        if topology == 'feedforward':
            @jax.jit
            def _forward(params, inputs, nt):
                return _forward_neuromod(params, inputs, nt, 'feedforward')
        else:
            @jax.jit
            def _forward(params, inputs, nt):
                return _forward_neuromod(params, inputs, nt, 'full_recurrent')
        _FORWARD_CACHE[topology] = _forward
    return _FORWARD_CACHE[topology]


# ============================================================================
# Evaluation
# ============================================================================

def eval_single_multitask(params: Dict, topology: str, inputs: jnp.ndarray,
                          targets: Dict, nt_vectors: Dict,
                          task_names: List[str]) -> Tuple[float, Dict[str, float]]:
    per_task = {}
    forward_fn = get_forward_fn(topology)

    for task in task_names:
        nt = nt_vectors[task]
        tgt = targets[task]

        output = forward_fn(params, inputs, nt)
        output = output.squeeze(-1)

        preds = (output > 0.5).astype(jnp.float32)
        acc = float(jnp.mean(jnp.equal(preds, tgt)))
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
    topology, n_in = _parse_condition(condition)
    mu = max(1, int(pop_size * mu_frac))

    inputs, targets, nt_vectors, task_names = get_task_data(n_in)
    n_patterns = len(inputs)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    pop = init_population(init_key, pop_size, topology, n_in)

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
            prod_fit, pt = eval_single_multitask(
                ind_params, topology, inputs, targets, nt_vectors, task_names)
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
            task_str = ' '.join(f"{t}:{best_per_task[t]:.2f}" for t in task_names)
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

    sample_params = init_params(jax.random.PRNGKey(0), topology, n_in)
    n_params = sum(p.size for p in jax.tree.leaves(sample_params))

    return {
        'experiment': 'E-S20_topology_arity_extended',
        'condition': condition,
        'topology': topology,
        'activation': 'tanh',
        'n_tasks': 3,
        'tasks': task_names,
        'n_inputs': n_in,
        'n_patterns': n_patterns,
        'domain': f'boolean_{n_in}input',
        'seed': seed,
        'pop_size': pop_size,
        'mu': mu,
        'sigma': sigma,
        'generations_max': generations,
        'n_hidden': N_HIDDEN,
        'n_params': n_params,
        'n_recurrent_steps': N_RECURRENT_STEPS if topology == 'full_recurrent' else 0,
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
    runtimes = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if data.get('converged', False):
            converged += 1
            if data.get('convergence_gen') is not None:
                gens.append(data['convergence_gen'])
        min_accs.append(data.get('min_task_accuracy', 0.0))
        runtimes.append(data.get('runtime_seconds', 0.0))

    total = len(files)
    rate = 100 * converged / total if total > 0 else 0
    n_params = None
    if files:
        with open(files[0]) as fh:
            n_params = json.load(fh).get('n_params')

    print(f"  {prefix}: {converged}/{total} ({rate:.1f}%)", end='')
    if n_params is not None:
        print(f" | {n_params} params", end='')
    if gens:
        print(f" | median gen {np.median(gens):.0f} [{min(gens)}-{max(gens)}]", end='')
    if runtimes:
        print(f" | median runtime {np.median(runtimes):.0f}s", end='')
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S20: Topology x Arity Extended (4-input and 5-input)')
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
        print("\n=== E-S20 Topology x Arity Extended Results Summary ===\n")
        for cond in ALL_CONDITIONS:
            summarize_condition(RESULTS_DIR, cond)
        return

    total_start = time.time()

    for condition in conditions:
        topology, n_in = _parse_condition(condition)

        print(f"\n{'='*60}")
        print(f"Condition: {condition} ({args.seeds} seeds)")
        print(f"  Topology: {topology}, Arity: {n_in}-input, Activation: uniform tanh")
        print(f"  Tasks: Parity-{n_in} + AND-{n_in} + OR-{n_in} ({2**n_in} patterns)")
        print(f"  N_HIDDEN: {N_HIDDEN}, Pop: {args.pop_size}")
        if topology == 'full_recurrent':
            print(f"  Recurrent steps: {N_RECURRENT_STEPS}")
        sample_params = init_params(jax.random.PRNGKey(0), topology, n_in)
        n_params = sum(p.size for p in jax.tree.leaves(sample_params))
        print(f"  Parameters: {n_params}")
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
                parity_key = f'parity{n_in}'
                par_acc = result['per_task_fitness'].get(parity_key, 0.0)
                print(f"  -> {status} (min_acc={result['min_task_accuracy']:.4f}, "
                      f"Parity-{n_in}={par_acc:.4f}, {result['runtime_seconds']:.1f}s)")
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
