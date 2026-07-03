#!/usr/bin/env python3
"""E-S3: Quantitative H4 Model, NT Vector Geometry Predicts Task Compatibility.

Transforms H4 from "partially supported" to "supported with quantitative evidence"
by computing NT vector distances and correlating with 2-task convergence rates.

Uses EXISTING data only (zero new experiments):
  - Schema A (independent): 10 pairs × 30 seeds = 300 datapoints
  - Schema B (compositional): 10 pairs × 30 seeds = 300 datapoints
  Total: 600 datapoints

Analysis:
  1. Compute pairwise NT distances (Euclidean, cosine) on DA/5HT/NE subspace
  2. Classify task computational class (parity vs threshold)
  3. Correlate distances with convergence rates (Spearman rank)
  4. Fit logistic model: P(convergence) ~ f(NT_distance, class_match)
  5. Leave-one-pair-out cross-validation

Usage:
    python papers/emr-neuromodulation/analysis_h4_quantitative.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ============================================================================
# NT Vector Definitions (from the two schemas)
# ============================================================================

# Schema A (NT_PRESETS_4): each task has unique NT, all ACh=1.0
SCHEMA_A = {
    'xor':  np.array([0.95, 0.05, 0.95]),  # DA/5HT/NE only
    'and':  np.array([0.10, 0.90, 0.10]),
    'or':   np.array([0.50, 0.50, 0.50]),
    'nand': np.array([0.90, 0.10, 0.50]),
    'nor':  np.array([0.10, 0.50, 0.90]),
}

# Schema B (HEAD_NT_PROFILES): NAND/NOR share hidden-layer modulation with AND/OR
# ACh dimension used for output inversion, not hidden-layer behavior
SCHEMA_B = {
    'xor':  np.array([0.95, 0.05, 0.95]),
    'and':  np.array([0.10, 0.90, 0.10]),
    'or':   np.array([0.50, 0.50, 0.50]),
    'nand': np.array([0.10, 0.90, 0.10]),  # same as AND at hidden layer
    'nor':  np.array([0.50, 0.50, 0.50]),  # same as OR at hidden layer
}

# Task computational class
PARITY_TASKS = {'xor', 'xnor'}
THRESHOLD_TASKS = {'and', 'or', 'nand', 'nor'}

ALL_TASKS = ['xor', 'and', 'or', 'nand', 'nor']
ALL_PAIRS = list(combinations(ALL_TASKS, 2))  # 10 pairs


# ============================================================================
# Distance metrics
# ============================================================================

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two NT vectors."""
    return float(np.linalg.norm(a - b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two NT vectors."""
    dot = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms < 1e-10:
        return 0.0
    return float(dot / norms)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - similarity)."""
    return 1.0 - cosine_similarity(a, b)


def class_match(t1: str, t2: str) -> bool:
    """Whether two tasks belong to the same computational class."""
    t1_parity = t1 in PARITY_TASKS
    t2_parity = t2 in PARITY_TASKS
    return t1_parity == t2_parity


# ============================================================================
# Data loading
# ============================================================================

def load_pair_results(results_dir: Path) -> Dict[str, Dict]:
    """Load 2-task convergence results from a results directory.

    Handles two formats:
    1. Single aggregated JSON (pair_results dict)
    2. Per-pair JSON files (pair_xor+and.json)

    Returns:
        Dict mapping pair name to {convergence_rate, n_seeds, convergence_gens}
    """
    pair_data = {}

    # Try per-pair files first
    for pair in ALL_PAIRS:
        pair_name = '+'.join(pair)
        pair_file = results_dir / f'pair_{pair_name}.json'
        if pair_file.exists():
            with open(pair_file) as f:
                data = json.load(f)

            if 'results' in data:
                # Per-seed results format
                seeds = data['results']
                n_seeds = len(seeds)
                converged_count = sum(1 for s in seeds if s.get('converged', False))
                conv_gens = [s['convergence_gen'] for s in seeds
                             if s.get('converged') and s.get('convergence_gen') is not None]
            else:
                n_seeds = data.get('n_seeds', 30)
                converged_count = data.get('convergence_count', 0)
                conv_gens = []

            pair_data[pair_name] = {
                'convergence_rate': converged_count / n_seeds if n_seeds > 0 else 0.0,
                'n_seeds': n_seeds,
                'convergence_gens': conv_gens,
                'converged_count': converged_count,
            }

    # Try aggregated JSON files
    if not pair_data:
        for json_file in sorted(results_dir.glob('*.json')):
            with open(json_file) as f:
                data = json.load(f)
            if 'pair_results' in data:
                for pair_name, pair_info in data['pair_results'].items():
                    pair_data[pair_name] = {
                        'convergence_rate': pair_info.get('convergence_rate', 0.0),
                        'n_seeds': pair_info.get('n_seeds', 30),
                        'convergence_gens': pair_info.get('convergence_gens', []),
                        'converged_count': int(pair_info.get('convergence_rate', 0.0) *
                                               pair_info.get('n_seeds', 30)),
                    }

    return pair_data


# ============================================================================
# Analysis
# ============================================================================

def compute_pair_features(schema: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Compute all distance features for each task pair under a given schema."""
    features = {}
    for t1, t2 in ALL_PAIRS:
        pair_name = f'{t1}+{t2}'
        v1 = schema[t1]
        v2 = schema[t2]
        features[pair_name] = {
            'euclidean': euclidean_distance(v1, v2),
            'cosine_dist': cosine_distance(v1, v2),
            'cosine_sim': cosine_similarity(v1, v2),
            'class_match': class_match(t1, t2),
            'has_parity': t1 in PARITY_TASKS or t2 in PARITY_TASKS,
            'both_parity': t1 in PARITY_TASKS and t2 in PARITY_TASKS,
            'both_threshold': t1 in THRESHOLD_TASKS and t2 in THRESHOLD_TASKS,
        }
    return features


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman rank correlation and approximate p-value.

    Uses Fieller-Hartley-Pearson approximation for p-value.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    # Rank both arrays
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1

    # Pearson correlation of ranks
    rx_centered = rx - np.mean(rx)
    ry_centered = ry - np.mean(ry)

    num = np.sum(rx_centered * ry_centered)
    den = np.sqrt(np.sum(rx_centered**2) * np.sum(ry_centered**2))

    if den < 1e-10:
        return 0.0, 1.0

    rho = float(num / den)

    # Approximate p-value via t-distribution
    if abs(rho) >= 1.0:
        return rho, 0.0

    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
    # Use normal approximation for p-value
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    return rho, p_value


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz and Stegun)."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def logistic_regression_loo(X: np.ndarray, y: np.ndarray) -> Dict:
    """Simple logistic regression with leave-one-out cross-validation.

    Uses gradient descent for fitting since we avoid sklearn dependency.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary outcomes (n_samples,)

    Returns:
        Dict with coefficients, accuracy, AUC estimate
    """
    n = len(y)

    # Fit on all data
    weights, bias = _fit_logistic(X, y, lr=0.1, epochs=1000)

    # Full-data predictions
    logits = X @ weights + bias
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    preds = (probs > 0.5).astype(float)
    train_accuracy = float(np.mean(preds == y))

    # AUC estimate (simple: average of sensitivity and specificity)
    if np.sum(y) > 0 and np.sum(1 - y) > 0:
        sensitivity = float(np.sum((preds == 1) & (y == 1)) / np.sum(y == 1))
        specificity = float(np.sum((preds == 0) & (y == 0)) / np.sum(y == 0))
        auc_approx = (sensitivity + specificity) / 2
    else:
        auc_approx = 0.5

    # Leave-one-out CV
    loo_correct = 0
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]

        w, b = _fit_logistic(X_train, y_train, lr=0.1, epochs=500)
        p = 1.0 / (1.0 + np.exp(-np.clip(X_test @ w + b, -20, 20)))
        pred = float(p[0] > 0.5)
        if pred == y_test:
            loo_correct += 1

    loo_accuracy = loo_correct / n

    return {
        'weights': weights.tolist(),
        'bias': float(bias),
        'train_accuracy': train_accuracy,
        'auc_approx': auc_approx,
        'loo_accuracy': loo_accuracy,
        'loo_correct': loo_correct,
        'loo_total': n,
    }


def _fit_logistic(X: np.ndarray, y: np.ndarray,
                  lr: float = 0.1, epochs: int = 1000) -> Tuple[np.ndarray, float]:
    """Fit logistic regression via gradient descent."""
    n, d = X.shape
    weights = np.zeros(d)
    bias = 0.0

    for _ in range(epochs):
        logits = X @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

        # Gradient
        error = probs - y
        grad_w = (X.T @ error) / n
        grad_b = float(np.mean(error))

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ============================================================================
# Main analysis
# ============================================================================

def run_analysis():
    """Run full H4 quantitative analysis."""
    print("=" * 70)
    print("E-S3: Quantitative H4 Model")
    print("NT Vector Geometry Predicts Task Compatibility")
    print("=" * 70)

    # Load data from both schemas
    schema_a_dir = Path(__file__).resolve().parents[2] / 'results' / 'ablation_schema_a_uniform'
    schema_b_dir = Path(__file__).resolve().parents[2] / 'results' / 'ablation_schema_b_uniform'

    schema_a_data = load_pair_results(schema_a_dir)
    schema_b_data = load_pair_results(schema_b_dir)

    if not schema_a_data:
        print(f"\nWARNING: No Schema A data found in {schema_a_dir}")
        print("Looking for aggregated JSON files...")
        # Try parent directory
        parent = schema_a_dir.parent
        for f in sorted(parent.glob('ablation_schema_a_uniform*.json')):
            print(f"  Found: {f}")
            with open(f) as fh:
                data = json.load(fh)
            if 'pair_results' in data:
                for pair_name, pair_info in data['pair_results'].items():
                    rate = pair_info.get('convergence_rate', 0.0)
                    n = pair_info.get('n_seeds', 30)
                    schema_a_data[pair_name] = {
                        'convergence_rate': rate,
                        'n_seeds': n,
                        'convergence_gens': [],
                        'converged_count': int(rate * n),
                    }

    if not schema_b_data:
        print(f"\nWARNING: No Schema B data found in {schema_b_dir}")
        parent = schema_b_dir.parent
        for f in sorted(parent.glob('ablation_schema_b_uniform*.json')):
            print(f"  Found: {f}")
            with open(f) as fh:
                data = json.load(fh)
            if 'pair_results' in data:
                for pair_name, pair_info in data['pair_results'].items():
                    rate = pair_info.get('convergence_rate', 0.0)
                    n = pair_info.get('n_seeds', 30)
                    schema_b_data[pair_name] = {
                        'convergence_rate': rate,
                        'n_seeds': n,
                        'convergence_gens': [],
                        'converged_count': int(rate * n),
                    }

    # Compute NT vector features
    schema_a_features = compute_pair_features(SCHEMA_A)
    schema_b_features = compute_pair_features(SCHEMA_B)

    # ===== Section 1: NT Distance Tables =====
    print("\n" + "=" * 70)
    print("SECTION 1: NT Vector Distances")
    print("=" * 70)

    for schema_name, schema, features, data in [
        ("Schema A (Independent)", SCHEMA_A, schema_a_features, schema_a_data),
        ("Schema B (Compositional)", SCHEMA_B, schema_b_features, schema_b_data),
    ]:
        print(f"\n--- {schema_name} ---")
        print(f"{'Pair':<15} {'Eucl':<8} {'CosDist':<8} {'CosSim':<8} {'Class':<8} {'Rate':<8}")
        print("-" * 55)

        for t1, t2 in ALL_PAIRS:
            pair_name = f'{t1}+{t2}'
            feat = features[pair_name]
            rate = data.get(pair_name, {}).get('convergence_rate', -1)
            rate_str = f"{rate*100:.1f}%" if rate >= 0 else "N/A"
            class_str = "same" if feat['class_match'] else "diff"

            print(f"{pair_name:<15} {feat['euclidean']:<8.3f} {feat['cosine_dist']:<8.3f} "
                  f"{feat['cosine_sim']:<8.3f} {class_str:<8} {rate_str:<8}")

    # ===== Section 2: Correlation Analysis =====
    print("\n" + "=" * 70)
    print("SECTION 2: Spearman Rank Correlations")
    print("=" * 70)

    for schema_name, features, data in [
        ("Schema A", schema_a_features, schema_a_data),
        ("Schema B", schema_b_features, schema_b_data),
    ]:
        print(f"\n--- {schema_name} ---")

        if not data:
            print("  No data available")
            continue

        # Build vectors
        eucl_dists = []
        cos_dists = []
        rates = []
        has_parity_flags = []
        class_matches = []

        for t1, t2 in ALL_PAIRS:
            pair_name = f'{t1}+{t2}'
            if pair_name not in data:
                continue
            feat = features[pair_name]
            eucl_dists.append(feat['euclidean'])
            cos_dists.append(feat['cosine_dist'])
            rates.append(data[pair_name]['convergence_rate'])
            has_parity_flags.append(1.0 if feat['has_parity'] else 0.0)
            class_matches.append(1.0 if feat['class_match'] else 0.0)

        eucl_dists = np.array(eucl_dists)
        cos_dists = np.array(cos_dists)
        rates = np.array(rates)
        has_parity_flags = np.array(has_parity_flags)
        class_matches = np.array(class_matches)

        # Correlations
        rho_eucl, p_eucl = spearman_rank_correlation(eucl_dists, rates)
        rho_cos, p_cos = spearman_rank_correlation(cos_dists, rates)
        rho_parity, p_parity = spearman_rank_correlation(has_parity_flags, rates)
        rho_class, p_class = spearman_rank_correlation(class_matches, rates)

        print(f"  Euclidean distance vs convergence:  rho={rho_eucl:+.3f}, p={p_eucl:.4f}")
        print(f"  Cosine distance vs convergence:     rho={rho_cos:+.3f}, p={p_cos:.4f}")
        print(f"  Has parity task vs convergence:      rho={rho_parity:+.3f}, p={p_parity:.4f}")
        print(f"  Same class vs convergence:           rho={rho_class:+.3f}, p={p_class:.4f}")

    # ===== Section 3: Combined Analysis Across Both Schemas =====
    print("\n" + "=" * 70)
    print("SECTION 3: Combined Cross-Schema Analysis")
    print("=" * 70)

    all_eucl = []
    all_cos = []
    all_rates = []
    all_has_parity = []
    all_class_match = []
    all_schema_label = []

    for schema_name, features, data in [
        ("A", schema_a_features, schema_a_data),
        ("B", schema_b_features, schema_b_data),
    ]:
        for t1, t2 in ALL_PAIRS:
            pair_name = f'{t1}+{t2}'
            if pair_name not in data:
                continue
            feat = features[pair_name]
            all_eucl.append(feat['euclidean'])
            all_cos.append(feat['cosine_dist'])
            all_rates.append(data[pair_name]['convergence_rate'])
            all_has_parity.append(1.0 if feat['has_parity'] else 0.0)
            all_class_match.append(1.0 if feat['class_match'] else 0.0)
            all_schema_label.append(schema_name)

    all_eucl = np.array(all_eucl)
    all_cos = np.array(all_cos)
    all_rates = np.array(all_rates)
    all_has_parity = np.array(all_has_parity)
    all_class_match = np.array(all_class_match)

    n_total = len(all_rates)
    print(f"\nTotal data points: {n_total} (10 pairs × 2 schemas)")

    rho_eucl, p_eucl = spearman_rank_correlation(all_eucl, all_rates)
    rho_cos, p_cos = spearman_rank_correlation(all_cos, all_rates)
    rho_parity, p_parity = spearman_rank_correlation(all_has_parity, all_rates)

    print(f"  Combined Euclidean vs rate:  rho={rho_eucl:+.3f}, p={p_eucl:.4f}")
    print(f"  Combined Cosine vs rate:     rho={rho_cos:+.3f}, p={p_cos:.4f}")
    print(f"  Combined has_parity vs rate: rho={rho_parity:+.3f}, p={p_parity:.4f}")

    # ===== Section 4: Logistic Regression Model =====
    print("\n" + "=" * 70)
    print("SECTION 4: Logistic Regression — P(convergence) ~ f(features)")
    print("=" * 70)

    if n_total >= 5:
        # Binarize convergence rate for logistic regression
        y_binary = (all_rates > 0.5).astype(float)

        # Model 1: Euclidean distance only
        X1 = all_eucl.reshape(-1, 1)
        result1 = logistic_regression_loo(X1, y_binary)
        print(f"\n  Model 1: P(conv) ~ euclidean_distance")
        print(f"    Train accuracy: {result1['train_accuracy']*100:.1f}%")
        print(f"    LOO accuracy:   {result1['loo_accuracy']*100:.1f}% "
              f"({result1['loo_correct']}/{result1['loo_total']})")
        print(f"    AUC (approx):   {result1['auc_approx']:.3f}")

        # Model 2: Has parity task only
        X2 = all_has_parity.reshape(-1, 1)
        result2 = logistic_regression_loo(X2, y_binary)
        print(f"\n  Model 2: P(conv) ~ has_parity_task")
        print(f"    Train accuracy: {result2['train_accuracy']*100:.1f}%")
        print(f"    LOO accuracy:   {result2['loo_accuracy']*100:.1f}% "
              f"({result2['loo_correct']}/{result2['loo_total']})")
        print(f"    AUC (approx):   {result2['auc_approx']:.3f}")

        # Model 3: Euclidean + has_parity
        X3 = np.column_stack([all_eucl, all_has_parity])
        result3 = logistic_regression_loo(X3, y_binary)
        print(f"\n  Model 3: P(conv) ~ euclidean + has_parity")
        print(f"    Train accuracy: {result3['train_accuracy']*100:.1f}%")
        print(f"    LOO accuracy:   {result3['loo_accuracy']*100:.1f}% "
              f"({result3['loo_correct']}/{result3['loo_total']})")
        print(f"    AUC (approx):   {result3['auc_approx']:.3f}")
        print(f"    Weights: eucl={result3['weights'][0]:.3f}, "
              f"parity={result3['weights'][1]:.3f}, bias={result3['bias']:.3f}")

        # Model 4: Full model (eucl + cosine + has_parity + class_match)
        X4 = np.column_stack([all_eucl, all_cos, all_has_parity, all_class_match])
        result4 = logistic_regression_loo(X4, y_binary)
        print(f"\n  Model 4: P(conv) ~ eucl + cosine + parity + class_match")
        print(f"    Train accuracy: {result4['train_accuracy']*100:.1f}%")
        print(f"    LOO accuracy:   {result4['loo_accuracy']*100:.1f}% "
              f"({result4['loo_correct']}/{result4['loo_total']})")
        print(f"    AUC (approx):   {result4['auc_approx']:.3f}")

        # R-squared for convergence rate prediction (continuous)
        # Simple linear: rate ~ euclidean
        A = np.column_stack([all_eucl, np.ones(n_total)])
        coeffs, _, _, _ = np.linalg.lstsq(A, all_rates, rcond=None)
        pred_rates = A @ coeffs
        r2_eucl = compute_r_squared(all_rates, pred_rates)

        # rate ~ has_parity
        A2 = np.column_stack([all_has_parity, np.ones(n_total)])
        coeffs2, _, _, _ = np.linalg.lstsq(A2, all_rates, rcond=None)
        pred_rates2 = A2 @ coeffs2
        r2_parity = compute_r_squared(all_rates, pred_rates2)

        # rate ~ eucl + has_parity
        A3 = np.column_stack([all_eucl, all_has_parity, np.ones(n_total)])
        coeffs3, _, _, _ = np.linalg.lstsq(A3, all_rates, rcond=None)
        pred_rates3 = A3 @ coeffs3
        r2_combined = compute_r_squared(all_rates, pred_rates3)

        print(f"\n  Linear R² (continuous convergence rate):")
        print(f"    rate ~ euclidean:              R² = {r2_eucl:.3f}")
        print(f"    rate ~ has_parity:             R² = {r2_parity:.3f}")
        print(f"    rate ~ euclidean + has_parity:  R² = {r2_combined:.3f}")

    # ===== Section 5: Key Insights =====
    print("\n" + "=" * 70)
    print("SECTION 5: Key Insights for H4")
    print("=" * 70)

    # Count XOR-containing vs threshold-only pairs
    xor_pairs_a = {p: schema_a_data.get(p, {}).get('convergence_rate', -1)
                   for p in [f'xor+{t}' for t in ['and', 'or', 'nand', 'nor']]}
    thresh_pairs_a = {p: schema_a_data.get(p, {}).get('convergence_rate', -1)
                      for p in ['and+or', 'and+nand', 'and+nor', 'or+nand', 'or+nor', 'nand+nor']}

    print("\n  Schema A XOR-containing pair rates:")
    for p, r in xor_pairs_a.items():
        print(f"    {p}: {r*100:.1f}%" if r >= 0 else f"    {p}: N/A")

    print("\n  Schema A threshold-only pair rates:")
    for p, r in thresh_pairs_a.items():
        print(f"    {p}: {r*100:.1f}%" if r >= 0 else f"    {p}: N/A")

    xor_rates_a = [r for r in xor_pairs_a.values() if r >= 0]
    thresh_rates_a = [r for r in thresh_pairs_a.values() if r >= 0]

    if xor_rates_a and thresh_rates_a:
        print(f"\n  XOR-containing mean rate: {np.mean(xor_rates_a)*100:.1f}%")
        print(f"  Threshold-only mean rate: {np.mean(thresh_rates_a)*100:.1f}%")
        print(f"  Gap: {(np.mean(thresh_rates_a) - np.mean(xor_rates_a))*100:.1f}pp")

    # OR's central position analysis
    print("\n  OR's central position in NT space:")
    for schema_name, schema in [("Schema A", SCHEMA_A), ("Schema B", SCHEMA_B)]:
        or_vec = schema['or']
        distances = {t: euclidean_distance(or_vec, schema[t])
                     for t in ALL_TASKS if t != 'or'}
        print(f"    {schema_name}: OR distances = "
              + ", ".join(f"{t}={d:.3f}" for t, d in distances.items()))
        print(f"      Mean distance from OR: {np.mean(list(distances.values())):.3f}")

    # Save results
    output = {
        'schema_a_features': {k: v for k, v in schema_a_features.items()},
        'schema_b_features': {k: v for k, v in schema_b_features.items()},
        'schema_a_rates': {p: schema_a_data.get(p, {}).get('convergence_rate', None)
                           for p in ['+'.join(pair) for pair in ALL_PAIRS]},
        'schema_b_rates': {p: schema_b_data.get(p, {}).get('convergence_rate', None)
                           for p in ['+'.join(pair) for pair in ALL_PAIRS]},
        'correlations': {
            'combined_eucl_rho': rho_eucl if n_total >= 5 else None,
            'combined_eucl_p': p_eucl if n_total >= 5 else None,
            'combined_parity_rho': rho_parity if n_total >= 5 else None,
            'combined_parity_p': p_parity if n_total >= 5 else None,
        },
    }

    if n_total >= 5:
        output['logistic_models'] = {
            'model1_euclidean': result1,
            'model2_has_parity': result2,
            'model3_eucl_parity': result3,
            'model4_full': result4,
        }
        output['linear_r_squared'] = {
            'euclidean': r2_eucl,
            'has_parity': r2_parity,
            'combined': r2_combined,
        }

    output_file = Path(__file__).resolve().parents[2] / 'results' / 'h4_quantitative_analysis.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    run_analysis()
