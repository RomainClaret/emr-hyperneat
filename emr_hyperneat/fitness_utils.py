"""Configurable fitness functions for HMR-HyperNEAT benchmarks.

This module provides fitness computation functions that support both:
1. Binary classification tasks (e.g., XOR, NAND)
2. Multi-class classification tasks (e.g., symmetry detection with 4 classes)
3. Regression tasks (e.g., morphogen gradient interpolation)

The key insight motivating this module:
- CPPNs naturally produce soft/smooth outputs
- MSE penalizes soft outputs even when argmax classification is correct
- Different metrics reveal different aspects of network capability

Example:
    A network outputting [0.84, 0.05, 0.05, 0.05] for class 0:
    - MSE fitness: 0.8125 (penalized for not being [1, 0, 0, 0])
    - Argmax accuracy: 100% (correctly identifies class 0)
    - The network IS solving the task, but MSE says it isn't
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional


# Valid fitness modes
VALID_FITNESS_MODES = ['mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy']


def compute_fitness(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
    mode: str = 'mse',
    is_multiclass: bool = False,
) -> float:
    """Compute fitness using specified mode.

    Supports both binary and multi-class classification, as well as regression.

    Different fitness functions have different properties:
    - MSE: Penalizes soft outputs, good for regression
    - Accuracy: Pure classification metric, ignores confidence
    - Acc+MSE: Accuracy primary, MSE for gradient signal
    - Hybrid: Weighted combination of accuracy and MSE
    - BCE: Cross-entropy loss (binary or categorical)
    - Soft accuracy: Continuous approximation of accuracy

    Args:
        outputs: Network outputs, shape (num_samples, num_outputs)
        targets: Target values, same shape as outputs
        mode: Fitness mode from VALID_FITNESS_MODES
        is_multiclass: If True, treat as multi-class (argmax over last dim)
                      If False, treat as binary (threshold at 0.5)

    Returns:
        Fitness value in [0, 1] range (higher = better)
    """
    # Ensure 2D shape
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    # Auto-detect multiclass if more than 1 output per sample
    if outputs.shape[-1] > 1:
        is_multiclass = True

    if mode == 'mse':
        return _compute_mse_fitness(outputs, targets)

    elif mode == 'accuracy':
        return _compute_accuracy_fitness(outputs, targets, is_multiclass)

    elif mode == 'acc_mse':
        return _compute_acc_mse_fitness(outputs, targets, is_multiclass)

    elif mode == 'hybrid':
        return _compute_hybrid_fitness(outputs, targets, is_multiclass)

    elif mode == 'bce':
        return _compute_bce_fitness(outputs, targets, is_multiclass)

    elif mode == 'soft_accuracy':
        return _compute_soft_accuracy_fitness(outputs, targets, is_multiclass)

    else:
        raise ValueError(
            f"Unknown fitness mode: {mode}. Valid modes: {VALID_FITNESS_MODES}"
        )


def _compute_mse_fitness(outputs: jnp.ndarray, targets: jnp.ndarray) -> float:
    """MSE-based fitness. Good for regression, penalizes soft outputs."""
    mse = jnp.mean((outputs - targets) ** 2)
    return 1.0 - mse


def _compute_accuracy_fitness(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
    is_multiclass: bool,
) -> float:
    """Pure classification accuracy. Ignores output confidence."""
    if is_multiclass:
        # Multi-class: compare argmax
        predictions = jnp.argmax(outputs, axis=-1)
        labels = jnp.argmax(targets, axis=-1)
        return jnp.mean(predictions == labels)
    else:
        # Binary: threshold at 0.5
        predictions = (outputs > 0.5).astype(jnp.float32)
        return jnp.mean(predictions == targets)


def _compute_acc_mse_fitness(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
    is_multiclass: bool,
) -> float:
    """Accuracy + small MSE component for gradient signal."""
    accuracy = _compute_accuracy_fitness(outputs, targets, is_multiclass)
    mse = jnp.mean((outputs - targets) ** 2)
    # Scale MSE to be a small tiebreaker (max contribution ~0.01)
    return accuracy + 0.01 * (1.0 - mse)


def _compute_hybrid_fitness(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
    is_multiclass: bool,
    accuracy_weight: float = 0.8,
) -> float:
    """Weighted hybrid: accuracy_weight * accuracy + (1-w) * (1-MSE)."""
    accuracy = _compute_accuracy_fitness(outputs, targets, is_multiclass)
    mse = jnp.mean((outputs - targets) ** 2)
    return accuracy_weight * accuracy + (1.0 - accuracy_weight) * (1.0 - mse)


def _compute_bce_fitness(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
    is_multiclass: bool,
) -> float:
    """Cross-entropy fitness. Categorical for multiclass, binary otherwise."""
    eps = 1e-7
    outputs_safe = jnp.clip(outputs, eps, 1.0 - eps)

    if is_multiclass:
        # Categorical cross-entropy
        # Normalize outputs to sum to 1 (softmax-like)
        outputs_normalized = outputs_safe / jnp.sum(outputs_safe, axis=-1, keepdims=True)
        ce = -jnp.mean(jnp.sum(targets * jnp.log(outputs_normalized + eps), axis=-1))
    else:
        # Binary cross-entropy
        ce = -jnp.mean(
            targets * jnp.log(outputs_safe) +
            (1.0 - targets) * jnp.log(1.0 - outputs_safe)
        )

    # Convert to fitness: perfect = 1.0
    return 1.0 / (1.0 + ce)


def _compute_soft_accuracy_fitness(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
    is_multiclass: bool,
    temperature: float = 10.0,
) -> float:
    """Soft accuracy with temperature for gradient signal.

    For multiclass: Uses softmax with temperature to sharpen outputs,
    then computes expected accuracy.
    For binary: Uses sigmoid-based soft thresholding.
    """
    if is_multiclass:
        # Sharpen outputs using temperature
        # Higher temp = more like hard argmax
        outputs_sharpened = jax.nn.softmax(outputs * temperature, axis=-1)
        # Soft accuracy: expected probability of correct class
        return jnp.mean(jnp.sum(outputs_sharpened * targets, axis=-1))
    else:
        # Binary soft accuracy
        T = temperature
        soft_pred = jax.nn.sigmoid((outputs - 0.5) * T)
        return jnp.mean(soft_pred * targets + (1.0 - soft_pred) * (1.0 - targets))


def compute_classification_accuracy(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> float:
    """Compute argmax classification accuracy.

    Always uses argmax for multiclass, threshold for single output.
    Use this to track actual classification performance regardless of
    which fitness metric is used for evolution.

    Args:
        outputs: Network outputs, shape (num_samples, num_outputs)
        targets: Target values, same shape as outputs

    Returns:
        Classification accuracy in [0, 1]
    """
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    if outputs.shape[-1] > 1:
        predictions = jnp.argmax(outputs, axis=-1)
        labels = jnp.argmax(targets, axis=-1)
        return jnp.mean(predictions == labels)
    else:
        predictions = (outputs > 0.5).astype(jnp.float32)
        return jnp.mean(predictions == targets)


def compute_output_statistics(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> Dict[str, float]:
    """Compute detailed output distribution statistics.

    Useful for understanding WHY MSE fitness differs from accuracy.

    Args:
        outputs: Network outputs, shape (num_samples, num_outputs)
        targets: Target values, same shape as outputs

    Returns:
        Dictionary with:
        - confidence_correct: Mean output for correct class
        - confidence_gap: Mean gap between correct and max incorrect
        - entropy: Mean output entropy (measure of uncertainty)
        - mse_fitness: MSE-based fitness
        - accuracy: Argmax accuracy
    """
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    is_multiclass = outputs.shape[-1] > 1

    # Basic metrics
    mse_fitness = float(_compute_mse_fitness(outputs, targets))
    accuracy = float(compute_classification_accuracy(outputs, targets))

    if is_multiclass:
        # Get correct class indices
        correct_class = jnp.argmax(targets, axis=-1)

        # Confidence for correct class
        batch_indices = jnp.arange(outputs.shape[0])
        confidence_correct = float(jnp.mean(outputs[batch_indices, correct_class]))

        # Confidence gap: correct - max(incorrect)
        # Create mask for incorrect classes
        mask = 1.0 - targets  # 1 where not correct class
        incorrect_outputs = outputs * mask - (1 - mask) * 1e9  # -inf for correct
        max_incorrect = jnp.max(incorrect_outputs, axis=-1)
        confidence_gap = float(jnp.mean(
            outputs[batch_indices, correct_class] - max_incorrect
        ))

        # Entropy
        eps = 1e-7
        outputs_safe = jnp.clip(outputs, eps, 1.0)
        outputs_normalized = outputs_safe / jnp.sum(outputs_safe, axis=-1, keepdims=True)
        entropy = float(-jnp.mean(
            jnp.sum(outputs_normalized * jnp.log(outputs_normalized), axis=-1)
        ))
    else:
        # Binary case
        confidence_correct = float(jnp.mean(
            outputs * targets + (1 - outputs) * (1 - targets)
        ))
        confidence_gap = float(jnp.mean(jnp.abs(outputs - 0.5)))  # Distance from boundary
        eps = 1e-7
        outputs_safe = jnp.clip(outputs, eps, 1.0 - eps)
        entropy = float(-jnp.mean(
            outputs_safe * jnp.log(outputs_safe) +
            (1 - outputs_safe) * jnp.log(1 - outputs_safe)
        ))

    return {
        'confidence_correct': confidence_correct,
        'confidence_gap': confidence_gap,
        'entropy': entropy,
        'mse_fitness': mse_fitness,
        'accuracy': accuracy,
    }


def compute_trivial_baseline(
    targets: jnp.ndarray,
    strategy: str = 'mean',
) -> float:
    """Compute trivial baseline fitness for a problem.

    A trivial baseline outputs a constant value for all inputs.
    This helps establish how much "real learning" is required.

    Args:
        targets: Target values, shape (num_samples, num_outputs)
        strategy: 'mean' outputs mean of targets
                  'zeros' outputs all zeros
                  'ones' outputs all ones
                  'uniform' outputs 1/num_classes for each class

    Returns:
        MSE fitness of trivial constant output
    """
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    num_samples, num_outputs = targets.shape

    if strategy == 'mean':
        trivial_output = jnp.tile(jnp.mean(targets, axis=0, keepdims=True), (num_samples, 1))
    elif strategy == 'zeros':
        trivial_output = jnp.zeros_like(targets)
    elif strategy == 'ones':
        trivial_output = jnp.ones_like(targets)
    elif strategy == 'uniform':
        trivial_output = jnp.ones_like(targets) / num_outputs
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return float(_compute_mse_fitness(trivial_output, targets))


def compute_relative_improvement(
    fitness: float,
    trivial_baseline: float,
    max_fitness: float = 1.0,
) -> float:
    """Compute relative improvement from trivial baseline.

    This normalizes difficulty across problems with different baselines.

    Args:
        fitness: Achieved fitness
        trivial_baseline: Fitness of trivial constant output
        max_fitness: Maximum achievable fitness (typically 1.0)

    Returns:
        Relative improvement in [0, 1] where:
        - 0 = no better than trivial
        - 1 = achieved maximum fitness

    Example:
        If trivial=0.5 and fitness=0.75, relative improvement = 0.5
        (achieved 50% of possible improvement over trivial)
    """
    max_improvement = max_fitness - trivial_baseline
    if max_improvement <= 0:
        return 0.0
    achieved_improvement = fitness - trivial_baseline
    return max(0.0, min(1.0, achieved_improvement / max_improvement))


def threshold_from_relative_improvement(
    trivial_baseline: float,
    required_improvement: float = 0.80,
    max_fitness: float = 1.0,
) -> float:
    """Compute fitness threshold requiring specified relative improvement.

    Use this to set difficulty-normalized thresholds across problems.

    Args:
        trivial_baseline: Fitness of trivial constant output
        required_improvement: Required relative improvement (e.g., 0.80 = 80%)
        max_fitness: Maximum achievable fitness (typically 1.0)

    Returns:
        Fitness threshold that requires specified improvement over trivial

    Example:
        If trivial=0.5 and required_improvement=0.80:
        threshold = 0.5 + 0.80 * (1.0 - 0.5) = 0.9
    """
    max_improvement = max_fitness - trivial_baseline
    return trivial_baseline + required_improvement * max_improvement


# JIT-compiled versions for performance
compute_fitness_jit = jax.jit(compute_fitness, static_argnums=(2, 3))
compute_classification_accuracy_jit = jax.jit(compute_classification_accuracy)
