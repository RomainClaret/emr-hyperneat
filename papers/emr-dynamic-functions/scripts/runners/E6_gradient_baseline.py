"""Gradient-trained MLP baseline for parity tasks.

Tests whether standard MLPs with gradient descent (Adam) can solve parity
with sin vs tanh activation. This addresses the killer objection: if a
gradient-trained MLP with sin also trivially solves parity, then the paper's
contribution narrows to "sin is good for parity" rather than anything
specific to neuroevolution.

Design:
- MLP: Linear -> activation -> Linear -> sigmoid output
- Hidden sizes: [8, 16, 32, 64] (default), configurable via --hidden-sizes
- Activations: sin, tanh
- Problems: Parity-4, Parity-8 (default), configurable via --parity-sizes
- N=30 seeds, 5000 epochs max, learning rates [0.01, 0.001]
- Success criterion: accuracy >= 0.95

E6 extension: Run with --parity-sizes 5,6,7 --hidden-sizes 16,32 --learning-rates 0.01
to test gradient baseline at intermediate parity sizes.

Usage:
    # Original experiment (Parity-4, Parity-8)
    python papers/emr-dynamic-functions/scripts/runners/E6_gradient_baseline.py

    # E6: Intermediate parity sizes
    python papers/emr-dynamic-functions/scripts/runners/E6_gradient_baseline.py \\
        --parity-sizes 5,6,7 --hidden-sizes 16,32 --learning-rates 0.01 \\
        --output-dir results/gradient_baseline_n30

Uses JAX for consistency with the rest of the codebase.
"""

import argparse
import json
import os
from pathlib import Path
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax


def generate_parity_data(n_bits: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate all 2^n parity patterns."""
    n_samples = 2 ** n_bits
    X = jnp.array([[int(b) for b in format(i, f'0{n_bits}b')]
                    for i in range(n_samples)], dtype=jnp.float32)
    y = jnp.sum(X, axis=1) % 2  # parity = sum mod 2
    return X, y.astype(jnp.float32)


def init_mlp_params(key: jax.random.PRNGKey, n_inputs: int,
                    n_hidden: int, n_outputs: int = 1) -> dict:
    """Initialize MLP parameters with Xavier initialization."""
    k1, k2 = jax.random.split(key)
    w1_std = jnp.sqrt(2.0 / (n_inputs + n_hidden))
    w2_std = jnp.sqrt(2.0 / (n_hidden + n_outputs))
    return {
        'w1': jax.random.normal(k1, (n_inputs, n_hidden)) * w1_std,
        'b1': jnp.zeros(n_hidden),
        'w2': jax.random.normal(k2, (n_hidden, n_outputs)) * w2_std,
        'b2': jnp.zeros(n_outputs),
    }


def forward_mlp(params: dict, x: jnp.ndarray,
                activation: str) -> jnp.ndarray:
    """Forward pass through single-hidden-layer MLP."""
    h = x @ params['w1'] + params['b1']
    if activation == 'sin':
        h = jnp.sin(h)
    elif activation == 'tanh':
        h = jnp.tanh(h)
    elif activation == 'relu':
        h = jax.nn.relu(h)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    out = h @ params['w2'] + params['b2']
    return jax.nn.sigmoid(out.squeeze(-1))


def loss_fn(params: dict, x: jnp.ndarray, y: jnp.ndarray,
            activation: str) -> jnp.ndarray:
    """Binary cross-entropy loss."""
    pred = forward_mlp(params, x, activation)
    eps = 1e-7
    pred = jnp.clip(pred, eps, 1 - eps)
    return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))


def compute_accuracy(params: dict, x: jnp.ndarray, y: jnp.ndarray,
                     activation: str) -> float:
    """Compute classification accuracy."""
    pred = forward_mlp(params, x, activation)
    pred_labels = (pred >= 0.5).astype(jnp.float32)
    return float(jnp.mean(pred_labels == y))


def train_mlp(seed: int, n_bits: int, n_hidden: int,
              activation: str, lr: float,
              max_epochs: int = 5000,
              target_acc: float = 0.95) -> dict:
    """Train a single MLP and return results."""
    key = jax.random.PRNGKey(seed)
    X, y = generate_parity_data(n_bits)
    params = init_mlp_params(key, n_bits, n_hidden)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    grad_fn = jax.grad(loss_fn)

    t0 = time.time()
    solved = False
    solved_epoch = None
    best_acc = 0.0

    for epoch in range(max_epochs):
        grads = grad_fn(params, X, y, activation)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            acc = compute_accuracy(params, X, y, activation)
            best_acc = max(best_acc, acc)
            if acc >= target_acc and not solved:
                solved = True
                solved_epoch = epoch + 1
                break

    if not solved:
        acc = compute_accuracy(params, X, y, activation)
        best_acc = max(best_acc, acc)

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'n_bits': n_bits,
        'n_hidden': n_hidden,
        'activation': activation,
        'learning_rate': lr,
        'solved': solved,
        'solved_epoch': solved_epoch,
        'best_accuracy': float(best_acc),
        'elapsed_seconds': elapsed,
    }


def load_config_results(filepath: str) -> list:
    """Load existing per-config results for resume."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return []


def get_completed_seeds(config_results: list) -> set:
    """Extract completed seed numbers from config results."""
    return {r['seed'] for r in config_results}


def main():
    """Run gradient baseline experiments."""
    parser = argparse.ArgumentParser(
        description='Gradient-trained MLP baseline for parity tasks'
    )
    parser.add_argument('--parity-sizes', type=str, default='4,8',
                        help='Comma-separated parity sizes (default: 4,8)')
    parser.add_argument('--hidden-sizes', type=str, default='8,16,32,64',
                        help='Comma-separated hidden sizes (default: 8,16,32,64)')
    parser.add_argument('--learning-rates', type=str, default='0.01,0.001',
                        help='Comma-separated learning rates (default: 0.01,0.001)')
    parser.add_argument('--activations', type=str, default='sin,tanh',
                        help='Comma-separated activations (default: sin,tanh)')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds (default: 30)')
    parser.add_argument('--max-epochs', type=int, default=5000,
                        help='Maximum training epochs (default: 5000)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-timestamped)')
    parser.add_argument('--list', action='store_true',
                        help='Show status and exit')
    args = parser.parse_args()

    parity_sizes = [int(x) for x in args.parity_sizes.split(',')]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]
    learning_rates = [float(x) for x in args.learning_rates.split(',')]
    activations = [x.strip() for x in args.activations.split(',')]

    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = os.path.join(
            str(Path(__file__).resolve().parents[2] / "results"),
            f"gradient_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(results_dir, exist_ok=True)

    n_seeds = args.seeds
    seeds = list(range(42, 42 + n_seeds))

    if args.list:
        print("Gradient baseline status:")
        print(f"{'Config':<40} {'Seeds':<10} {'Status'}")
        print("-" * 60)
        for n_bits in parity_sizes:
            for activation in activations:
                for n_hidden in hidden_sizes:
                    for lr in learning_rates:
                        config_name = (f"parity{n_bits}_{activation}_"
                                       f"h{n_hidden}_lr{lr}")
                        config_path = os.path.join(
                            results_dir, f"{config_name}.json"
                        )
                        existing = load_config_results(config_path)
                        n_done = len(existing)
                        status = ("COMPLETE" if n_done >= n_seeds
                                  else f"{n_done}/{n_seeds}")
                        print(f"{config_name:<40} {n_done:<10} {status}")
        return

    total_configs = (len(parity_sizes) * len(activations) *
                     len(hidden_sizes) * len(learning_rates))
    print(f"Running {total_configs} configurations x {n_seeds} seeds = "
          f"{total_configs * n_seeds} total runs")
    print(f"Output: {results_dir}")
    print(f"Resume: automatic (completed seeds skipped)")

    all_results = []
    config_idx = 0

    for n_bits in parity_sizes:
        for activation in activations:
            for n_hidden in hidden_sizes:
                for lr in learning_rates:
                    config_idx += 1
                    config_name = (f"parity{n_bits}_{activation}_"
                                   f"h{n_hidden}_lr{lr}")
                    config_path = os.path.join(
                        results_dir, f"{config_name}.json"
                    )

                    # Load existing results for resume
                    config_results = load_config_results(config_path)
                    completed_seeds = get_completed_seeds(config_results)
                    remaining_seeds = [s for s in seeds
                                       if s not in completed_seeds]

                    if not remaining_seeds:
                        print(f"\n[{config_idx}/{total_configs}] "
                              f"{config_name} — all {n_seeds} seeds "
                              f"complete, skipping")
                        all_results.extend(config_results)
                        continue

                    print(f"\n[{config_idx}/{total_configs}] {config_name} "
                          f"({len(completed_seeds)}/{n_seeds} done)")

                    for seed in remaining_seeds:
                        result = train_mlp(
                            seed=seed, n_bits=n_bits,
                            n_hidden=n_hidden, activation=activation,
                            lr=lr, max_epochs=args.max_epochs,
                        )
                        config_results.append(result)

                        # Save incrementally after each seed
                        with open(config_path, 'w') as f:
                            json.dump(config_results, f, indent=2)

                    # Summary
                    solved_count = sum(
                        1 for r in config_results if r['solved']
                    )
                    solve_rate = solved_count / len(config_results) * 100
                    solved_epochs = [
                        r['solved_epoch'] for r in config_results
                        if r['solved']
                    ]
                    mean_epoch = (np.mean(solved_epochs)
                                  if solved_epochs else None)
                    best_accs = [r['best_accuracy'] for r in config_results]

                    print(f"  Solve rate: {solve_rate:.1f}% "
                          f"({solved_count}/{len(config_results)})")
                    if solved_epochs:
                        print(f"  Mean epoch to solve: {mean_epoch:.1f}")
                    print(f"  Mean best accuracy: "
                          f"{np.mean(best_accs):.4f} +/- "
                          f"{np.std(best_accs):.4f}")

                    all_results.extend(config_results)

    # Save combined results
    combined_path = os.path.join(results_dir, "combined_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 80)
    print(f"{'Problem':<12} {'Activation':<6} {'Hidden':<7} {'LR':<7} "
          f"{'Solve%':>7} {'Epoch':>7} {'BestAcc':>8}")
    print("-" * 60)

    for n_bits in parity_sizes:
        for activation in activations:
            for n_hidden in hidden_sizes:
                for lr in learning_rates:
                    subset = [r for r in all_results
                              if r['n_bits'] == n_bits
                              and r['activation'] == activation
                              and r['n_hidden'] == n_hidden
                              and r['learning_rate'] == lr]
                    if not subset:
                        continue
                    solved = sum(1 for r in subset if r['solved'])
                    solve_pct = solved / len(subset) * 100
                    epochs = [r['solved_epoch'] for r in subset
                              if r['solved']]
                    mean_ep = f"{np.mean(epochs):.0f}" if epochs else "---"
                    mean_acc = np.mean(
                        [r['best_accuracy'] for r in subset]
                    )
                    print(f"Parity-{n_bits:<4} {activation:<6} "
                          f"{n_hidden:<7} {lr:<7.3f} "
                          f"{solve_pct:>6.1f}% {mean_ep:>7} "
                          f"{mean_acc:>8.4f}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
