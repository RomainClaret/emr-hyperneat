#!/usr/bin/env python3
"""E-S26: Adam Optimizer Independence at 3-Input.

Extends the crown jewel optimizer independence test (E-S7, 2-input) to 3-input.
E-S25 revealed a search-budget ceiling at 3-input: Pop=2000/3000 don't improve
over Pop=1500 (50%). This tests whether Adam bypasses the ceiling, confirming
the 3-input barrier is also evolutionary-search-specific.

Conditions (2 × 30 seeds = 60 runs):
  1. adam_tanh_lr0.01, 1L, 20 hidden, uniform tanh, Adam lr=0.01, 2000 steps
  2. adam_pertask_lr0.01, 1L, 20 hidden, per-task activation, Adam lr=0.01, 2000 steps

Architecture: Direct-encoded MLP, 5 tasks (Parity-3, AND-3, OR-3, NAND-3, NOR-3),
3-input (8 patterns), N_HIDDEN=20, Schema B NT vectors.
Loss: sum of per-task MSE. Convergence: accuracy ≥ 0.95.

Results saved to: papers/emr-neuromodulation/results/adam_3input/

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_adam_3input.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_adam_3input.py \
        --seeds 3 --conditions adam_tanh_lr0.01  # smoke test
    python papers/emr-neuromodulation/scripts/runners/benchmark_adam_3input.py --summary
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
import optax

# ============================================================================
# Constants
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'adam_3input'

N_IN = 3
N_HIDDEN = 20  # Matching E-S9 3-input experiments
N_OUT = 1
MODULATION_STRENGTH = 5.0

INPUTS = jnp.array(list(iterproduct([0.0, 1.0], repeat=3)))  # (8, 3)

TASK_NAMES = ['parity3', 'and3', 'or3', 'nand3', 'nor3']


def _parity3_targets():
    return jnp.array([float(sum(bits) % 2) for bits in iterproduct([0.0, 1.0], repeat=3)])

def _and3_targets():
    return jnp.array([float(all(b == 1.0 for b in bits)) for bits in iterproduct([0.0, 1.0], repeat=3)])

def _or3_targets():
    return jnp.array([float(any(b == 1.0 for b in bits)) for bits in iterproduct([0.0, 1.0], repeat=3)])

def _nand3_targets():
    return jnp.array([float(not all(b == 1.0 for b in bits)) for bits in iterproduct([0.0, 1.0], repeat=3)])

def _nor3_targets():
    return jnp.array([float(not any(b == 1.0 for b in bits)) for bits in iterproduct([0.0, 1.0], repeat=3)])


TRUTH_TABLES = {
    'parity3': _parity3_targets(),
    'and3': _and3_targets(),
    'or3': _or3_targets(),
    'nand3': _nand3_targets(),
    'nor3': _nor3_targets(),
}

# Schema B NT vectors with ACh inversion
NT_VECTORS = {
    'parity3': jnp.array([0.95, 0.05, 0.95, 1.0]),
    'and3':    jnp.array([0.10, 0.90, 0.10, 1.0]),
    'or3':     jnp.array([0.50, 0.50, 0.50, 1.0]),
    'nand3':   jnp.array([0.10, 0.90, 0.10, 0.0]),
    'nor3':    jnp.array([0.50, 0.50, 0.50, 0.0]),
}

PERTASK_ACTIVATIONS = {
    'parity3': 'sin',
    'and3': 'tanh',
    'or3': 'tanh',
    'nand3': 'tanh',
    'nor3': 'tanh',
}

ALL_CONDITIONS = [
    'adam_tanh_lr0.01',
    'adam_pertask_lr0.01',
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
    total_loss = 0.0
    for task in TASK_NAMES:
        nt = NT_VECTORS[task]
        targets = TRUTH_TABLES[task]
        output = forward_neuromod(params, INPUTS, nt, jnp.tanh)
        output = output.squeeze(-1)
        total_loss = total_loss + jnp.mean((output - targets) ** 2)
    return total_loss


def multitask_mse_loss_pertask(params: Dict) -> jnp.ndarray:
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
        'n_patterns': 8,
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
            val = data.get('convergence_step')
            if val is not None:
                convergence_vals.append(val)

    total = len(files)
    rate = 100 * converged / total if total > 0 else 0
    print(f"  {condition:<25}: {converged}/{total} ({rate:.1f}%)", end='')
    if convergence_vals:
        print(f" | median step {np.median(convergence_vals):.0f} "
              f"[{min(convergence_vals)}-{max(convergence_vals)}]", end='')
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S26: Adam optimizer independence at 3-input')
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
        print("\n=== E-S26 Adam 3-Input — Summary ===\n")
        for cond in ALL_CONDITIONS:
            summarize_condition(RESULTS_DIR, cond)
        return

    total_start = time.time()

    for condition in conditions:
        lr = 0.01
        act_mode = 'per-task' if 'pertask' in condition else 'uniform tanh'

        print(f"\n{'='*60}")
        print(f"Condition: {condition} ({args.seeds} seeds)")
        print(f"  Optimizer: Adam, lr={lr}, {args.steps} steps")
        print(f"  Activation: {act_mode}")
        print(f"  Tasks: {TASK_NAMES} (3-input, 8 patterns)")
        print(f"  N_HIDDEN: {N_HIDDEN}")
        print(f"{'='*60}")

        for seed in range(args.seeds):
            fname = RESULTS_DIR / f'{condition}_seed{seed}.json'
            if result_exists(fname):
                print(f"  Skip existing: {fname.name}")
                continue

            print(f"\n  Seed {seed}:")
            try:
                result = run_adam(
                    condition=condition,
                    seed=seed,
                    lr=lr,
                    steps=args.steps,
                    verbose=True,
                )

                save_result(result, fname)
                if result.get('converged'):
                    print(f"  -> CONVERGED at step {result['convergence_step']} "
                          f"({result['runtime_seconds']:.1f}s)")
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
