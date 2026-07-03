#!/usr/bin/env python3
"""E-S8: Arity Capacity Match, resolving the 4-input vs 5-input anomaly.

The E22/E33 results show an anomaly: 4-input tasks (20 hidden neurons) achieve
only 13.3% multi-task success, while 5-input tasks (25 hidden neurons) reach
83.3%. This experiment disentangles arity from capacity by cross-matching:

  E-S8a: 4-input tasks with N_HIDDEN=25 (5-input capacity)
  E-S8b: 5-input tasks with N_HIDDEN=20 (4-input capacity)

If the bottleneck is capacity (hidden neurons), then:
  - E-S8a (4-input, 25h) should improve over baseline 13.3%
  - E-S8b (5-input, 20h) should degrade from baseline 83.3%

If the bottleneck is arity (input dimensionality), then:
  - E-S8a should stay near 13.3% despite extra capacity
  - E-S8b should stay near 83.3% despite reduced capacity

Tasks:
  4-input: Parity-4 (odd parity), AND-4 (all 1s), OR-4 (any 1), 16 patterns
  5-input: Parity-5 (odd parity), AND-5 (all 1s), OR-5 (any 1), 32 patterns

Conditions (8 total, 4 per sub-experiment):
  4input_25h_uniform_tanh, baseline activation, swapped capacity
  4input_25h_pertask, sin for parity, tanh for rest, swapped capacity
  4input_25h_uniform_sin, all sin, swapped capacity
  4input_25h_2layer_tanh, 2 hidden layers of 25, uniform tanh
  5input_20h_uniform_tanh, baseline activation, swapped capacity
  5input_20h_pertask, sin for parity, tanh for rest, swapped capacity
  5input_20h_uniform_sin, all sin, swapped capacity
  5input_20h_2layer_tanh, 2 hidden layers of 20, uniform tanh

Pop=750, 200 gen, sigma=0.3, success_threshold=0.95
30 seeds per condition = 240 runs total

Results saved to: papers/emr-neuromodulation/results/arity_capacity_match/

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_arity_capacity_match.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_arity_capacity_match.py \
        --seeds 3 --conditions 4input_25h_pertask 5input_20h_pertask  # smoke test
    python papers/emr-neuromodulation/scripts/runners/benchmark_arity_capacity_match.py --summary
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from itertools import product as iterproduct

# ============================================================================
# Constants
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'arity_capacity_match'

POP_SIZE = 750
GENERATIONS = 200
SIGMA = 0.3
MU_FRAC = 0.1
SUCCESS_THRESHOLD = 0.95
MODULATION_STRENGTH = 5.0
N_OUT = 1

ALL_CONDITIONS = [
    '4input_25h_uniform_tanh',
    '4input_25h_pertask',
    '4input_25h_uniform_sin',
    '4input_25h_2layer_tanh',
    '5input_20h_uniform_tanh',
    '5input_20h_pertask',
    '5input_20h_uniform_sin',
    '5input_20h_2layer_tanh',
]

# ============================================================================
# Task definitions, generated dynamically from arity
# ============================================================================

def make_inputs(n_in: int) -> jnp.ndarray:
    """Generate all 2^n_in binary input patterns."""
    return jnp.array(list(iterproduct([0.0, 1.0], repeat=n_in)))


def make_targets(n_in: int) -> Dict[str, jnp.ndarray]:
    """Generate truth tables for parity, AND, OR at given arity."""
    parity_t, and_t, or_t = [], [], []
    for bits in iterproduct([0.0, 1.0], repeat=n_in):
        parity_t.append(float(sum(bits) % 2))
        and_t.append(float(all(b == 1.0 for b in bits)))
        or_t.append(float(any(b == 1.0 for b in bits)))
    return {
        f'parity{n_in}': jnp.array(parity_t),
        f'and{n_in}':    jnp.array(and_t),
        f'or{n_in}':     jnp.array(or_t),
    }


INPUTS_4 = make_inputs(4)   # (16, 4)
INPUTS_5 = make_inputs(5)   # (32, 5)
TARGETS_4 = make_targets(4)
TARGETS_5 = make_targets(5)

TASK_NAMES_4 = ['parity4', 'and4', 'or4']
TASK_NAMES_5 = ['parity5', 'and5', 'or5']

# NT vectors, parity gets XOR profile, and gets AND profile, or gets OR profile
NT_VECTORS_4 = {
    'parity4': jnp.array([0.95, 0.05, 0.95, 1.0]),
    'and4':    jnp.array([0.10, 0.90, 0.10, 1.0]),
    'or4':     jnp.array([0.50, 0.50, 0.50, 1.0]),
}
NT_VECTORS_5 = {
    'parity5': jnp.array([0.95, 0.05, 0.95, 1.0]),
    'and5':    jnp.array([0.10, 0.90, 0.10, 1.0]),
    'or5':     jnp.array([0.50, 0.50, 0.50, 1.0]),
}

PERTASK_ACTIVATIONS_4 = {
    'parity4': 'sin',
    'and4': 'tanh',
    'or4': 'tanh',
}
PERTASK_ACTIVATIONS_5 = {
    'parity5': 'sin',
    'and5': 'tanh',
    'or5': 'tanh',
}


# ============================================================================
# Condition parsing
# ============================================================================

def parse_condition(condition: str) -> Dict:
    """Parse condition string into experiment parameters.

    Returns dict with keys: n_in, n_hidden, n_layers, activation_mode, task_names,
    inputs, targets, nt_vectors, pertask_activations, n_patterns.
    """
    parts = condition.split('_')

    # Parse n_in from first token
    if parts[0] == '4input':
        n_in = 4
        task_names = TASK_NAMES_4
        inputs = INPUTS_4
        targets = TARGETS_4
        nt_vectors = NT_VECTORS_4
        pertask_acts = PERTASK_ACTIVATIONS_4
        n_patterns = 16
    elif parts[0] == '5input':
        n_in = 5
        task_names = TASK_NAMES_5
        inputs = INPUTS_5
        targets = TARGETS_5
        nt_vectors = NT_VECTORS_5
        pertask_acts = PERTASK_ACTIVATIONS_5
        n_patterns = 32
    else:
        raise ValueError(f"Unknown arity prefix in condition: {condition}")

    # Parse n_hidden from second token (e.g., '25h' or '20h')
    n_hidden = int(parts[1].replace('h', ''))

    # Parse activation mode and layer count from remaining tokens
    # Possible suffixes: uniform_tanh, pertask, uniform_sin, 2layer_tanh
    suffix = '_'.join(parts[2:])
    if suffix == 'uniform_tanh':
        n_layers = 1
        activation_mode = 'uniform_tanh'
    elif suffix == 'pertask':
        n_layers = 1
        activation_mode = 'pertask'
    elif suffix == 'uniform_sin':
        n_layers = 1
        activation_mode = 'uniform_sin'
    elif suffix == '2layer_tanh':
        n_layers = 2
        activation_mode = 'uniform_tanh'
    else:
        raise ValueError(f"Unknown suffix in condition: {condition}")

    return {
        'n_in': n_in,
        'n_hidden': n_hidden,
        'n_layers': n_layers,
        'activation_mode': activation_mode,
        'task_names': task_names,
        'inputs': inputs,
        'targets': targets,
        'nt_vectors': nt_vectors,
        'pertask_activations': pertask_acts,
        'n_patterns': n_patterns,
    }


# ============================================================================
# Parameter initialization
# ============================================================================

def init_params(key: jax.Array, n_in: int, n_hidden: int,
                n_layers: int = 1) -> Dict[str, jnp.ndarray]:
    params = {}
    scale = 0.5
    n_keys = 4 * n_layers + 2
    keys = jax.random.split(key, n_keys)
    ki = 0

    for l in range(n_layers):
        in_dim = n_in if l == 0 else n_hidden
        params[f'W{l+1}'] = jax.random.normal(keys[ki], (in_dim, n_hidden)) * scale
        ki += 1
        params[f'b{l+1}'] = jax.random.normal(keys[ki], (n_hidden,)) * 0.1
        ki += 1
        params[f'R{l+1}'] = jax.random.normal(keys[ki], (n_hidden, 3)) * 0.3
        ki += 1
        params[f'g{l+1}'] = jnp.ones((n_hidden,)) + jax.random.normal(keys[ki], (n_hidden,)) * 0.1
        ki += 1

    params['W_out'] = jax.random.normal(keys[ki], (n_hidden, N_OUT)) * scale
    ki += 1
    params['b_out'] = jax.random.normal(keys[ki], (N_OUT,)) * 0.1
    return params


def init_population(key: jax.Array, pop_size: int, n_in: int, n_hidden: int,
                    n_layers: int = 1) -> Dict[str, jnp.ndarray]:
    keys = jax.random.split(key, pop_size)
    return jax.vmap(lambda k: init_params(k, n_in, n_hidden, n_layers))(keys)


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

def eval_single_multitask(params: Dict, cfg: Dict) -> Tuple[float, Dict[str, float]]:
    """Evaluate one individual on all tasks for the given condition config."""
    n_layers = cfg['n_layers']
    activation_mode = cfg['activation_mode']
    task_names = cfg['task_names']
    inputs = cfg['inputs']
    targets = cfg['targets']
    nt_vectors = cfg['nt_vectors']
    pertask_acts = cfg['pertask_activations']

    per_task = {}
    for task in task_names:
        nt = nt_vectors[task]
        tgt = targets[task]

        # Determine activation for this task
        if activation_mode == 'uniform_tanh':
            act_name = 'tanh'
        elif activation_mode == 'uniform_sin':
            act_name = 'sin'
        elif activation_mode == 'pertask':
            act_name = pertask_acts[task]
        else:
            raise ValueError(f"Unknown activation_mode: {activation_mode}")

        forward_fn = get_forward_fn(n_layers, act_name)
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
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    mu_frac: float = MU_FRAC,
    sigma: float = SIGMA,
    success_threshold: float = SUCCESS_THRESHOLD,
    verbose: bool = True,
) -> Dict:
    start_time = time.time()
    cfg = parse_condition(condition)
    n_in = cfg['n_in']
    n_hidden = cfg['n_hidden']
    n_layers = cfg['n_layers']
    task_names = cfg['task_names']
    mu = max(1, int(pop_size * mu_frac))

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    pop = init_population(init_key, pop_size, n_in, n_hidden, n_layers)

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
            prod_fit, pt = eval_single_multitask(ind_params, cfg)
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

        # Selection and reproduction
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

    sample_params = init_params(jax.random.PRNGKey(0), n_in, n_hidden, n_layers)
    n_params = sum(p.size for p in jax.tree.leaves(sample_params))

    # Identify parity task name for this condition
    parity_task = f'parity{n_in}'

    return {
        'condition': condition,
        'n_layers': n_layers,
        'activation_mode': cfg['activation_mode'],
        'n_tasks': 3,
        'tasks': task_names,
        'n_inputs': n_in,
        'n_hidden': n_hidden,
        'n_patterns': cfg['n_patterns'],
        'seed': seed,
        'pop_size': pop_size,
        'mu': mu,
        'sigma': sigma,
        'generations_max': generations,
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


def summarize_condition(results_dir: Path, condition: str):
    """Print summary statistics for a single condition."""
    files = sorted(results_dir.glob(f'{condition}_seed*.json'))
    if not files:
        print(f"  {condition}: no results")
        return

    converged = 0
    gens = []
    parity_accs = []
    n_in = 4 if condition.startswith('4input') else 5
    parity_task = f'parity{n_in}'

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if data.get('converged', False):
            converged += 1
            if data.get('convergence_gen') is not None:
                gens.append(data['convergence_gen'])
        pt = data.get('per_task_fitness', {})
        if parity_task in pt:
            parity_accs.append(pt[parity_task])

    total = len(files)
    rate = 100 * converged / total if total > 0 else 0
    print(f"  {condition:>30s}: {converged}/{total} ({rate:5.1f}%)", end='')
    if gens:
        print(f" | median gen {np.median(gens):5.0f} [{min(gens)}-{max(gens)}]", end='')
    if parity_accs:
        parity_solved = sum(1 for a in parity_accs if a >= SUCCESS_THRESHOLD)
        print(f" | Parity-{n_in} solved: {parity_solved}/{len(parity_accs)}", end='')
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S8: Arity Capacity Match — 4-input vs 5-input anomaly')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds (default: 30)')
    parser.add_argument('--conditions', nargs='+', default=None,
                        help=f'Conditions to run (default: all). '
                             f'Options: {", ".join(ALL_CONDITIONS)}')
    parser.add_argument('--generations', type=int, default=GENERATIONS,
                        help=f'Max generations (default: {GENERATIONS})')
    parser.add_argument('--sigma', type=float, default=SIGMA,
                        help=f'ES mutation sigma (default: {SIGMA})')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of existing results')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    conditions = args.conditions if args.conditions else ALL_CONDITIONS

    # Validate condition names
    for c in conditions:
        if c not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition '{c}'")
            print(f"Valid conditions: {ALL_CONDITIONS}")
            sys.exit(1)

    if args.summary:
        print("\n=== E-S8 Arity Capacity Match Summary ===")
        print("\n--- E-S8a: 4-input tasks with 25 hidden (5-input capacity) ---\n")
        for c in ALL_CONDITIONS[:4]:
            summarize_condition(RESULTS_DIR, c)
        print("\n--- E-S8b: 5-input tasks with 20 hidden (4-input capacity) ---\n")
        for c in ALL_CONDITIONS[4:]:
            summarize_condition(RESULTS_DIR, c)
        return

    total_start = time.time()

    for condition in conditions:
        cfg = parse_condition(condition)
        n_in = cfg['n_in']
        n_hidden = cfg['n_hidden']
        n_layers = cfg['n_layers']
        task_names = cfg['task_names']
        mu = max(1, int(POP_SIZE * MU_FRAC))

        print(f"\n{'='*70}")
        print(f"Condition: {condition}")
        print(f"  {n_in}-input tasks, {n_hidden} hidden, {n_layers} layer(s), "
              f"activation={cfg['activation_mode']}")
        print(f"  Tasks: {task_names} ({cfg['n_patterns']} patterns each)")
        print(f"  Pop={POP_SIZE}, mu={mu}, sigma={args.sigma}, "
              f"max_gen={args.generations}")
        print(f"  Seeds: {args.seeds}")
        print(f"{'='*70}")

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
                    pop_size=POP_SIZE,
                    generations=args.generations,
                    sigma=args.sigma,
                    verbose=True,
                )

                save_result(result, fname)
                parity_task = f'parity{n_in}'
                status = (f"gen {result['convergence_gen']}"
                          if result['converged'] else "NOT CONVERGED")
                par_acc = result['per_task_fitness'].get(parity_task, 0.0)
                print(f"  -> {status} (min_acc={result['min_task_accuracy']:.4f}, "
                      f"Parity-{n_in}={par_acc:.4f}, {result['runtime_seconds']:.1f}s)")
            except Exception as e:
                print(f"  ERROR on seed {seed}: {e}")
                continue

        print(f"\n--- Summary for {condition} ---")
        summarize_condition(RESULTS_DIR, condition)

    total_runtime = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/3600:.1f} hours)")
    print(f"{'='*70}")

    print("\n=== Final Summary ===")
    print("\n--- E-S8a: 4-input tasks with 25 hidden (5-input capacity) ---\n")
    for c in ALL_CONDITIONS[:4]:
        summarize_condition(RESULTS_DIR, c)
    print("\n--- E-S8b: 5-input tasks with 20 hidden (4-input capacity) ---\n")
    for c in ALL_CONDITIONS[4:]:
        summarize_condition(RESULTS_DIR, c)


if __name__ == '__main__':
    main()
