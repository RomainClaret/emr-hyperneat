#!/usr/bin/env python3
"""E-S7: Adam Gradient Descent on Neuromodulated Architecture.

Proves the multi-task barrier is STRUCTURAL (optimizer-independent), not ES-specific.
The neuromodulated forward pass is fully differentiable, if the barrier were merely
a search limitation, Adam should overcome it.

Conditions (4 × 30 seeds = 120 runs):
  1. adam_tanh_lr0.01, 1L, 10 hidden, uniform tanh, Adam lr=0.01, 2000 steps
  2. adam_tanh_lr0.001, 1L, 10 hidden, uniform tanh, Adam lr=0.001, 2000 steps
  3. adam_pertask_lr0.01, 1L, 10 hidden, per-task activation, Adam lr=0.01, 2000 steps
  4. es_tanh_replication, 1L, 10 hidden, uniform tanh, ES Pop=750, 100 gen (control)

Expected: Adam+tanh 0/30 (structural impossibility), Adam+per-task 30/30, ES+tanh 0/30.

Architecture: Direct-encoded MLP, 5 tasks, 2-input, Schema B.
Loss: sum of per-task MSE (differentiable surrogate). Convergence: accuracy ≥ 0.95.

Results saved to: papers/emr-neuromodulation/results/adam_neuromod/

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_adam_neuromod.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_adam_neuromod.py \
        --seeds 3 --conditions adam_tanh_lr0.01 adam_pertask_lr0.01  # smoke test
    python papers/emr-neuromodulation/scripts/runners/benchmark_adam_neuromod.py --summary
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

import jax
import jax.numpy as jnp
import optax

# ============================================================================
# Constants
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'adam_neuromod'

N_IN = 2
N_HIDDEN = 10
N_OUT = 1
MODULATION_STRENGTH = 5.0

INPUTS = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

TASK_NAMES = ['xor', 'and', 'or', 'nand', 'nor']

TRUTH_TABLES = {
    'xor':  jnp.array([0.0, 1.0, 1.0, 0.0]),
    'and':  jnp.array([0.0, 0.0, 0.0, 1.0]),
    'or':   jnp.array([0.0, 1.0, 1.0, 1.0]),
    'nand': jnp.array([1.0, 1.0, 1.0, 0.0]),
    'nor':  jnp.array([1.0, 0.0, 0.0, 0.0]),
}

# Schema B NT vectors
NT_VECTORS = {
    'xor':  jnp.array([0.95, 0.05, 0.95, 1.0]),
    'and':  jnp.array([0.10, 0.90, 0.10, 1.0]),
    'or':   jnp.array([0.50, 0.50, 0.50, 1.0]),
    'nand': jnp.array([0.10, 0.90, 0.10, 0.0]),
    'nor':  jnp.array([0.50, 0.50, 0.50, 0.0]),
}

PERTASK_ACTIVATIONS = {
    'xor': 'sin',
    'and': 'tanh',
    'or': 'tanh',
    'nand': 'tanh',
    'nor': 'tanh',
}

ALL_CONDITIONS = [
    'adam_tanh_lr0.01',
    'adam_tanh_lr0.001',
    'adam_pertask_lr0.01',
    'es_tanh_replication',
]


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
# Forward pass (differentiable)
# ============================================================================

def forward_neuromod(params: Dict, inputs: jnp.ndarray, nt: jnp.ndarray,
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


# ============================================================================
# Loss functions (differentiable)
# ============================================================================

def multitask_mse_loss_tanh(params: Dict) -> jnp.ndarray:
    """Sum of MSE across all 5 tasks, uniform tanh activation."""
    total_loss = 0.0
    for task in TASK_NAMES:
        nt = NT_VECTORS[task]
        targets = TRUTH_TABLES[task]
        output = forward_neuromod(params, INPUTS, nt, jnp.tanh)
        output = output.squeeze(-1)
        total_loss = total_loss + jnp.mean((output - targets) ** 2)
    return total_loss


def multitask_mse_loss_pertask(params: Dict) -> jnp.ndarray:
    """Sum of MSE across all 5 tasks, per-task activation."""
    act_map = {'sin': jnp.sin, 'tanh': jnp.tanh}
    total_loss = 0.0
    for task in TASK_NAMES:
        nt = NT_VECTORS[task]
        targets = TRUTH_TABLES[task]
        act_fn = act_map[PERTASK_ACTIVATIONS[task]]
        output = forward_neuromod(params, INPUTS, nt, act_fn)
        output = output.squeeze(-1)
        total_loss = total_loss + jnp.mean((output - targets) ** 2)
    return total_loss


# ============================================================================
# Accuracy evaluation (non-differentiable)
# ============================================================================

def eval_accuracy(params: Dict, activation_mode: str) -> Tuple[float, Dict[str, float]]:
    act_map = {'sin': jnp.sin, 'tanh': jnp.tanh}
    per_task = {}
    for task in TASK_NAMES:
        nt = NT_VECTORS[task]
        targets = TRUTH_TABLES[task]
        if activation_mode == 'pertask':
            act_fn = act_map[PERTASK_ACTIVATIONS[task]]
        else:
            act_fn = jnp.tanh
        output = forward_neuromod(params, INPUTS, nt, act_fn)
        output = output.squeeze(-1)
        preds = (output > 0.5).astype(jnp.float32)
        acc = float(jnp.mean(jnp.equal(preds, targets)))
        per_task[task] = acc

    product_fitness = 1.0
    for v in per_task.values():
        product_fitness *= v
    return product_fitness, per_task


# ============================================================================
# Adam Optimizer
# ============================================================================

def run_adam(
    condition: str,
    seed: int,
    lr: float = 0.01,
    steps: int = 2000,
    success_threshold: float = 0.95,
    verbose: bool = True,
) -> Dict:
    start_time = time.time()

    is_pertask = 'pertask' in condition
    activation_mode = 'pertask' if is_pertask else 'uniform_tanh'
    loss_fn = multitask_mse_loss_pertask if is_pertask else multitask_mse_loss_tanh

    key = jax.random.PRNGKey(seed)
    params = init_params(key)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    converged = False
    convergence_step = None
    best_per_task = None
    best_fitness = 0.0
    fitness_history = []

    for step in range(steps):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Check accuracy every 10 steps (or last step)
        if step % 10 == 0 or step == steps - 1:
            prod_fit, pt = eval_accuracy(params, activation_mode)
            loss_val = float(loss_jit(params))
            min_acc = min(pt.values())

            if prod_fit > best_fitness:
                best_fitness = prod_fit
                best_per_task = pt

            fitness_history.append({
                'step': step,
                'loss': loss_val,
                'best_product_fitness': prod_fit,
                'min_task_accuracy': min_acc,
                **pt,
            })

            if not converged and all(v >= success_threshold for v in pt.values()):
                converged = True
                convergence_step = step
                if verbose:
                    print(f"  *** CONVERGED at step {step} (loss={loss_val:.4f}, "
                          f"min_acc={min_acc:.4f}) ***")

            if verbose and step % 100 == 0:
                task_str = ' '.join(f"{t}:{pt[t]:.2f}" for t in TASK_NAMES)
                print(f"  Step {step:4d}: loss={loss_val:.4f} prod={prod_fit:.4f} "
                      f"min={min_acc:.4f} | {task_str}")

    runtime = time.time() - start_time

    sample_params = init_params(jax.random.PRNGKey(0))
    n_params = sum(p.size for p in jax.tree.leaves(sample_params))

    return {
        'condition': condition,
        'optimizer': 'adam',
        'learning_rate': lr,
        'steps': steps,
        'activation_mode': activation_mode,
        'n_tasks': 5,
        'tasks': TASK_NAMES,
        'n_inputs': N_IN,
        'n_patterns': 4,
        'seed': seed,
        'n_hidden': N_HIDDEN,
        'n_params': n_params,
        'modulation_strength': MODULATION_STRENGTH,
        'nt_schema': 'B',
        'converged': converged,
        'convergence_step': convergence_step,
        'best_product_fitness': best_fitness,
        'min_task_accuracy': min(best_per_task.values()) if best_per_task else 0.0,
        'per_task_fitness': best_per_task,
        'runtime_seconds': runtime,
        'fitness_history': fitness_history,
    }


# ============================================================================
# (mu+lambda)-ES Optimizer (for ES control condition)
# ============================================================================

def run_es(
    condition: str,
    seed: int,
    pop_size: int = 750,
    generations: int = 100,
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
            prod_fit, pt = eval_accuracy(ind_params, 'uniform_tanh')
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
        'condition': condition,
        'optimizer': 'es',
        'activation_mode': 'uniform_tanh',
        'n_tasks': 5,
        'tasks': TASK_NAMES,
        'n_inputs': N_IN,
        'n_patterns': 4,
        'seed': seed,
        'pop_size': pop_size,
        'mu': mu,
        'sigma': sigma,
        'generations_max': generations,
        'n_hidden': N_HIDDEN,
        'n_params': n_params,
        'modulation_strength': MODULATION_STRENGTH,
        'nt_schema': 'B',
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
    files = sorted(results_dir.glob(f'{condition}_seed*.json'))
    if not files:
        print(f"  {condition}: no results")
        return

    converged = 0
    convergence_vals = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if data.get('converged', False):
            converged += 1
            val = data.get('convergence_step') or data.get('convergence_gen')
            if val is not None:
                convergence_vals.append(val)

    total = len(files)
    rate = 100 * converged / total if total > 0 else 0
    unit = 'step' if 'adam' in condition else 'gen'
    print(f"  {condition:<25}: {converged}/{total} ({rate:.1f}%)", end='')
    if convergence_vals:
        print(f" | median {unit} {np.median(convergence_vals):.0f} "
              f"[{min(convergence_vals)}-{max(convergence_vals)}]", end='')
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S7: Adam gradient descent on neuromodulated architecture (RQ2)')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds (default: 30)')
    parser.add_argument('--conditions', nargs='+', default=None,
                        choices=ALL_CONDITIONS,
                        help='Conditions to run (default: all)')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Adam optimization steps (default: 2000)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of existing results')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    conditions = args.conditions if args.conditions else ALL_CONDITIONS

    if args.summary:
        print("\n=== E-S7 Adam Neuromod — Summary ===\n")
        for cond in ALL_CONDITIONS:
            summarize_condition(RESULTS_DIR, cond)
        return

    total_start = time.time()

    for condition in conditions:
        is_es = condition.startswith('es_')

        print(f"\n{'='*60}")
        print(f"Condition: {condition} ({args.seeds} seeds)")
        if is_es:
            print(f"  Optimizer: ES, Pop=750, 100 gen")
        else:
            lr = 0.001 if 'lr0.001' in condition else 0.01
            act_mode = 'per-task' if 'pertask' in condition else 'uniform tanh'
            print(f"  Optimizer: Adam, lr={lr}, {args.steps} steps")
            print(f"  Activation: {act_mode}")
        print(f"  Tasks: {TASK_NAMES} (2-input)")
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
                if is_es:
                    result = run_es(
                        condition=condition,
                        seed=seed,
                        verbose=True,
                    )
                else:
                    lr = 0.001 if 'lr0.001' in condition else 0.01
                    result = run_adam(
                        condition=condition,
                        seed=seed,
                        lr=lr,
                        steps=args.steps,
                        verbose=True,
                    )

                save_result(result, fname)
                if result.get('converged'):
                    val = result.get('convergence_step') or result.get('convergence_gen')
                    unit = 'step' if not is_es else 'gen'
                    print(f"  -> CONVERGED at {unit} {val} ({result['runtime_seconds']:.1f}s)")
                else:
                    print(f"  -> NOT CONVERGED (min_acc={result['min_task_accuracy']:.4f}, "
                          f"{result['runtime_seconds']:.1f}s)")
            except Exception as e:
                print(f"  ERROR on seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n--- Summary for {condition} ---")
        summarize_condition(RESULTS_DIR, condition)

    total_runtime = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/3600:.1f} hours)")
    print(f"{'='*60}")

    print("\n=== Final Summary ===\n")
    for cond in ALL_CONDITIONS:
        summarize_condition(RESULTS_DIR, cond)


if __name__ == '__main__':
    main()
