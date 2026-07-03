"""E5: Extrema count vs solve rate Spearman correlation.

Compute local extrema count for each of 18 activation functions in [-5, 5],
then compute Spearman rho between extrema count and solve rate.
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "per_function_ablation_n30")

# Function definitions and their solve rates from the paper
# Format: (name, solve_rate, function_lambda)
FUNCTIONS = [
    ("sin",        100.0, lambda x: np.sin(x)),
    ("osc_adapt",  100.0, lambda x: np.sin(x) * (1 - 0.2 * np.abs(x))),
    ("burst",      100.0, lambda x: np.tanh(x) + 0.5 * np.sin(3 * x)),
    ("band_pass",  100.0, lambda x: np.exp(-np.abs(x - 1)) - np.exp(-np.abs(x + 1))),
    ("receptive",   80.0, lambda x: np.exp(-x**2) * np.cos(2 * x)),
    ("resonator",   66.7, lambda x: np.sin(x) * np.exp(-np.abs(x) / 3)),
    ("gauss",       33.3, lambda x: np.exp(-x**2)),
    ("rs_adapt",    23.3, lambda x: np.tanh(x) * (1 - 0.3 * np.abs(x))),
    ("lts_low",      6.7, lambda x: 1.0 / (1.0 + np.exp(-(2 * x - 0.5)))),
    ("tanh",         0.0, lambda x: np.tanh(x)),
    ("sigmoid",      0.0, lambda x: 1.0 / (1.0 + np.exp(-x))),
    ("relu",         0.0, lambda x: np.maximum(0, x)),
    ("identity",     0.0, lambda x: x),
    ("lelu",         0.0, lambda x: np.where(x > 0, x, 0.01 * x)),
    ("softplus",     0.0, lambda x: np.log1p(np.exp(x))),
    ("fs_fast",      0.0, lambda x: 2 * np.maximum(0, x)),
    ("gain_mod",     0.0, lambda x: x / (1 + np.abs(x))),
    ("integrate",    0.0, lambda x: np.tanh(x) * (1 + 0.2 * np.exp(-np.abs(x)))),
]


def count_local_extrema(func, x_range=(-5, 5), n_points=10000) -> int:
    """Count local extrema (peaks + troughs) in the given range."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = func(x)

    # Find local maxima and minima using sign changes in derivative
    dy = np.diff(y)
    # Sign changes in derivative indicate extrema
    sign_changes = np.diff(np.sign(dy))
    # Local maxima: derivative goes from positive to negative (sign change = -2)
    # Local minima: derivative goes from negative to positive (sign change = +2)
    extrema = np.sum(np.abs(sign_changes) >= 1.5)  # threshold to avoid noise
    return int(extrema)


def main():
    print("=" * 70)
    print("E5: Extrema Count vs Solve Rate — Spearman Correlation")
    print("=" * 70)

    names = []
    solve_rates = []
    extrema_counts = []

    print(f"\n{'Function':<12} {'Extrema':>8} {'Solve%':>8}")
    print("-" * 30)

    for name, solve_rate, func in FUNCTIONS:
        n_extrema = count_local_extrema(func)
        names.append(name)
        solve_rates.append(solve_rate)
        extrema_counts.append(n_extrema)
        print(f"{name:<12} {n_extrema:>8} {solve_rate:>7.1f}%")

    extrema_arr = np.array(extrema_counts)
    solve_arr = np.array(solve_rates)

    # Spearman rank correlation
    rho, p_value = stats.spearmanr(extrema_arr, solve_arr)

    # Bootstrap 95% CI for Spearman rho
    n_bootstrap = 10000
    n = len(extrema_arr)
    rng = np.random.RandomState(42)
    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(extrema_arr[idx], solve_arr[idx])
        if not np.isnan(r):
            bootstrap_rhos.append(r)
    bootstrap_rhos = np.array(bootstrap_rhos)
    ci_low = np.percentile(bootstrap_rhos, 2.5)
    ci_high = np.percentile(bootstrap_rhos, 97.5)

    print("\n" + "=" * 70)
    print("RESULTS:")
    print(f"  Spearman rho = {rho:.3f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  95% CI (bootstrap, 10000 resamples): [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  N functions = {n}")
    print("=" * 70)
    print("\nKEY STAT FOR PAPER:")
    print(f"  Local extrema count predicts solvability (Spearman rho={rho:.2f}, "
          f"p={p_value:.1e}, 95% CI [{ci_low:.2f}, {ci_high:.2f}], N={n})")
    print("=" * 70)


if __name__ == "__main__":
    main()
