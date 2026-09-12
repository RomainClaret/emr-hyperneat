#!/usr/bin/env python3
"""E-S12: Topology x Arity Interaction.

Tests whether topology remains neutral at 3-input (where tanh=20%).
The E-S5 finding (topology strictly neutral on 2-input Boolean tasks)
may not hold when the task is harder and activation is suboptimal.

Tasks (3-input, 8 patterns each):
  - Parity-3: output 1 if odd number of 1s
  - AND-3: output 1 iff all 3 inputs = 1
  - OR-3: output 1 iff any input = 1

Conditions (2 total, 60 runs):
  1. feedforward      -- standard MLP, single pass
  2. full_recurrent   -- hidden layer has lateral connections, 3 recurrent steps

Both use uniform tanh activation (the transition zone where Parity-3 ~20%).
Architecture: MLP + (mu+lambda)-ES (consistent with E-S2/E22/E25).
N=30 seeds, 200 generations, Pop=750, N_IN=3, N_HIDDEN=15.

Results saved to: papers/emr-neuromodulation/results/topology_3input/

Usage:
    python papers/emr-neuromodulation/scripts/runners/benchmark_topology_3input.py
    python papers/emr-neuromodulation/scripts/runners/benchmark_topology_3input.py \
        --seeds 3 --conditions feedforward full_recurrent  # smoke test
    python papers/emr-neuromodulation/scripts/runners/benchmark_topology_3input.py --summary
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'topology_3input'

N_IN = 3
N_HIDDEN = 15
N_OUT = 1
MODULATION_STRENGTH = 5.0
N_RECURRENT_STEPS = 3

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

ALL_CONDITIONS = ['feedforward', 'full_recurrent']


# ============================================================================
# Parameter initialization
# ============================================================================

def init_params(key: jax.Array, topology: str) -> Dict[str, jnp.ndarray]:
    """Initialize parameters for neuromodulated MLP with optional lateral connections.

    Args:
        key: PRNG key.
        topology: 'feedforward' or 'full_recurrent'.

    Returns:
        Parameter dict. full_recurrent adds W_lateral (N_HIDDEN, N_HIDDEN).
    """
    params = {}
    scale = 0.5

    n_keys_needed = 7 if topology == 'full_recurrent' else 6
    keys = jax.random.split(key, n_keys_needed)
    ki = 0

    # Hidden layer: W1 (N_IN, N_HIDDEN), b1, R1, g1
    params['W1'] = jax.random.normal(keys[ki], (N_IN, N_HIDDEN)) * scale
    ki += 1
    params['b1'] = jax.random.normal(keys[ki], (N_HIDDEN,)) * 0.1
    ki += 1
    params['R1'] = jax.random.normal(keys[ki], (N_HIDDEN, 3)) * 0.3
    ki += 1
    params['g1'] = jnp.ones((N_HIDDEN,)) + jax.random.normal(keys[ki], (N_HIDDEN,)) * 0.1
    ki += 1

    # Lateral connections for full_recurrent
    if topology == 'full_recurrent':
        params['W_lateral'] = jax.random.normal(keys[ki], (N_HIDDEN, N_HIDDEN)) * 0.1
        ki += 1

    # Output layer
    params['W_out'] = jax.random.normal(keys[ki], (N_HIDDEN, N_OUT)) * scale
    ki += 1
    params['b_out'] = jax.random.normal(keys[ki], (N_OUT,)) * 0.1

    return params


def init_population(key: jax.Array, pop_size: int, topology: str) -> Dict[str, jnp.ndarray]:
    """Initialize a population of parameter sets."""
    keys = jax.random.split(key, pop_size)
    return jax.vmap(lambda k: init_params(k, topology))(keys)


# ============================================================================
# Forward pass
# ============================================================================

def _forward_neuromod(params: Dict, inputs: jnp.ndarray, nt: jnp.ndarray,
                      topology: str,
                      s: float = MODULATION_STRENGTH) -> jnp.ndarray:
    """Neuromodulated forward pass with topology selection.

    Args:
        params: Parameter dict (includes W_lateral for full_recurrent).
        inputs: Input patterns (N_patterns, N_IN).
        nt: 4D neurotransmitter vector [DA, 5HT, NE, ACh].
        topology: 'feedforward' or 'full_recurrent'.
        s: Modulation strength.

    Returns:
        Output predictions (N_patterns, N_OUT).
    """
    W1 = params['W1']
    b1 = params['b1']
    R1 = params['R1']
    g1 = params['g1']

    # Compute neuromodulation components (shared across topologies)
    mod = R1 @ nt[:3]
    g_eff = g1 + s * mod
    g_eff = jnp.clip(g_eff, 0.1, 5.0)
    gates = jax.nn.sigmoid(mod)
    mod_bias = mod * s

    if topology == 'feedforward':
        # Standard single-pass MLP
        pre_h = inputs @ W1 + b1
        h = jnp.tanh(g_eff * pre_h + mod_bias) * gates
    else:
        # full_recurrent: lateral connections with recurrent steps
        W_lateral = params['W_lateral']

        # Initial hidden state
        pre_h = inputs @ W1 + b1
        h = jnp.tanh(g_eff * pre_h + mod_bias) * gates

        # Recurrent updates: re-apply neuromodulation at each step
        for _ in range(N_RECURRENT_STEPS):
            pre_h = inputs @ W1 + h @ W_lateral + b1
            h = jnp.tanh(g_eff * pre_h + mod_bias) * gates

    # Output layer
    output = jax.nn.sigmoid(h @ params['W_out'] + params['b_out'])

    # ACh inversion gate
    invert = nt[3]
    output = invert * output + (1.0 - invert) * (1.0 - output)

    return output


# Cache JIT-compiled forward functions per topology
_FORWARD_CACHE = {}


def get_forward_fn(topology: str):
    """Get a JIT-compiled forward function for the given topology."""
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

def eval_single_multitask(params: Dict, topology: str) -> Tuple[float, Dict[str, float]]:
    """Evaluate one individual on all 3 tasks.

    Returns:
        (product_fitness, per_task_accuracy_dict)
    """
    per_task = {}
    forward_fn = get_forward_fn(topology)

    for task in TASK_NAMES:
        nt = NT_VECTORS[task]
        targets = TARGETS[task]

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
    """Run (mu+lambda)-ES for a single condition and seed.

    Args:
        condition: Topology name ('feedforward' or 'full_recurrent').
        seed: Random seed.
        pop_size: Population size.
        generations: Max generations.
        mu_frac: Fraction of population for parent selection.
        sigma: Mutation standard deviation.
        success_threshold: Per-task accuracy threshold for convergence.
        verbose: Print progress.

    Returns:
        Result dict with all experiment metadata and fitness history.
    """
    start_time = time.time()
    topology = condition  # condition IS the topology for this experiment
    mu = max(1, int(pop_size * mu_frac))

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    pop = init_population(init_key, pop_size, topology)

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
            prod_fit, pt = eval_single_multitask(ind_params, topology)
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

        # Selection: top mu individuals
        top_indices = jnp.argsort(fitnesses)[-mu:]
        parents = jax.tree.map(lambda x: x[top_indices], pop)

        # Reproduction: mutate parents
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

    # Count parameters
    sample_params = init_params(jax.random.PRNGKey(0), topology)
    n_params = sum(p.size for p in jax.tree.leaves(sample_params))

    return {
        'experiment': 'E-S12_topology_3input',
        'condition': condition,
        'topology': topology,
        'activation': 'tanh',
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
    """Check if a result file already exists and is non-empty."""
    return filepath.exists() and filepath.stat().st_size > 0


def save_result(result: Dict, filepath: Path):
    """Save result dict to JSON with incremental save safety."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, default=convert)


def summarize_condition(results_dir: Path, prefix: str):
    """Print summary statistics for a condition."""
    files = sorted(results_dir.glob(f'{prefix}_seed*.json'))
    if not files:
        print(f"  {prefix}: no results")
        return

    converged = 0
    gens = []
    min_accs = []
    parity_accs = []
    runtimes = []

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
    if parity_accs:
        par_solved = sum(1 for a in parity_accs if a >= 0.95)
        print(f" | Parity-3 solved: {par_solved}/{len(parity_accs)}", end='')
    if runtimes:
        print(f" | median runtime {np.median(runtimes):.0f}s", end='')
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='E-S12: Topology x Arity Interaction (3-input, uniform tanh)')
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
        print("\n=== E-S12 Topology x 3-Input Results Summary ===\n")
        for cond in ALL_CONDITIONS:
            summarize_condition(RESULTS_DIR, cond)
        return

    total_start = time.time()

    for condition in conditions:
        topology = condition

        print(f"\n{'='*60}")
        print(f"Condition: {condition} ({args.seeds} seeds)")
        print(f"  Topology: {topology}, Activation: uniform tanh")
        print(f"  Tasks: {TASK_NAMES} (3-input, 8 patterns each)")
        print(f"  N_HIDDEN: {N_HIDDEN}, Pop: {args.pop_size}")
        if topology == 'full_recurrent':
            print(f"  Recurrent steps: {N_RECURRENT_STEPS}")
        sample_params = init_params(jax.random.PRNGKey(0), topology)
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
