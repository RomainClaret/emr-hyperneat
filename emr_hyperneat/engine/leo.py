"""Link-expression output (LEO): the validity masks a CPPN's LEO channel draws and the sparsification of
the weights it gates."""
import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def vectorized_weight_sparsification_with_leo(
    weights: jnp.ndarray,
    leo_outputs: jnp.ndarray,
    weight_threshold: float = 0.2,
    leo_threshold: float = 0.2,
    max_weight: float = 8.0,
) -> jnp.ndarray:
    """Apply weight sparsification with LEO (Link Expression Output) gating.

    LEO enables the CPPN to separately express "should this connection exist?"
    (leo_output) from "how strong should it be?" (weight). This is critical for
    modular problems where strong inhibitory connections (-0.8) should exist but
    would be pruned by weight-magnitude-only thresholding.

    Args:
        weights: Raw weight values from CPPN output 0
            Shape: (N,) or (batch, N)
        leo_outputs: LEO expression values from CPPN output 1
            Shape: (N,) or (batch, N)
        weight_threshold: Minimum |weight| magnitude to keep (default 0.2)
        leo_threshold: Minimum LEO value to express connection (default 0.2)
        max_weight: Maximum allowed weight magnitude (default 8.0)

    Returns:
        Sparsified and scaled weights where connections exist only if
        BOTH weight threshold AND LEO threshold are exceeded.
        Shape: same as input weights
    """
    # Handle NaN/Inf values
    weights = jnp.where(jnp.isnan(weights) | jnp.isinf(weights), 0.0, weights)
    leo_outputs = jnp.where(jnp.isnan(leo_outputs) | jnp.isinf(leo_outputs), 0.0, leo_outputs)

    # Compute validity masks
    abs_weights = jnp.abs(weights)
    weight_valid = abs_weights > weight_threshold
    leo_valid = leo_outputs > leo_threshold

    # Connection exists only if BOTH conditions are met
    connection_valid = weight_valid & leo_valid

    # Scale weights to range [-max_weight, max_weight]
    # Using linear scaling from threshold to 1.0
    scaled_positive = (weights - weight_threshold) / (1.0 - weight_threshold)
    scaled_negative = (weights + weight_threshold) / (1.0 - weight_threshold)
    scaled = jnp.where(weights > 0, scaled_positive, scaled_negative)
    scaled = jnp.clip(scaled * max_weight, -max_weight, max_weight)

    # Apply LEO gating - zero out connections that don't pass both thresholds
    return jnp.where(connection_valid, scaled, 0.0)


def compute_leo_validity_mask(
    weights: jnp.ndarray,
    leo_outputs: jnp.ndarray,
    weight_threshold: float = 0.2,
    leo_threshold: float = 0.2,
) -> jnp.ndarray:
    """Compute boolean validity mask using LEO gating.

    This is the LEO-aware version of simple weight thresholding.
    Used where only the validity mask is needed (not the scaled weights).

    Args:
        weights: Raw weight values from CPPN
        leo_outputs: LEO expression values from CPPN
        weight_threshold: Minimum |weight| magnitude
        leo_threshold: Minimum LEO value to express connection

    Returns:
        Boolean mask where True = connection should exist
    """
    # Handle NaN/Inf
    weights = jnp.where(jnp.isnan(weights) | jnp.isinf(weights), 0.0, weights)
    leo_outputs = jnp.where(jnp.isnan(leo_outputs) | jnp.isinf(leo_outputs), 0.0, leo_outputs)

    # Both conditions must be met
    weight_valid = jnp.abs(weights) > weight_threshold
    leo_valid = leo_outputs > leo_threshold

    return weight_valid & leo_valid


#: The CPPN output LEO is read from. The weight is output 0; per-node activation and
#: aggregation outputs, when configured, take the channels after LEO's.
LEO_CHANNEL = 1

#: `leo.mode` values. 'compound': a connection needs |weight| above the weight threshold
#: and LEO above the LEO threshold. 'replacement' (Verbancsics & Stanley 2011): LEO alone
#: decides, and the weight is used unthresholded. 'hmr_compound': HMR's LEO gate
#: (:func:`leo_sparsification_hmr_compound`).
LEO_MODES = ('compound', 'replacement', 'hmr_compound')


def compute_leo_validity_mask_replacement(
    leo_outputs: jnp.ndarray,
    leo_threshold: float = 0.2,
) -> jnp.ndarray:
    """LEO replacement mode: a connection exists if LEO exceeds the threshold, alone.

    No weight magnitude check: the original LEO design (Verbancsics & Stanley 2011), where
    the LEO output decides topology and the weight output only strength.

    Args:
        leo_outputs: LEO expression values from the CPPN.
        leo_threshold: Minimum LEO value to express a connection.

    Returns:
        Boolean mask, True where a connection exists.
    """
    leo_outputs = jnp.where(jnp.isnan(leo_outputs) | jnp.isinf(leo_outputs), 0.0, leo_outputs)
    return leo_outputs > leo_threshold


@functools.partial(jax.jit, static_argnums=(2, 3))
def vectorized_weight_sparsification_leo_replacement(
    weights: jnp.ndarray,
    leo_outputs: jnp.ndarray,
    leo_threshold: float = 0.2,
    max_weight: float = 8.0,
) -> jnp.ndarray:
    """LEO replacement mode: the gate replaces the weight threshold.

    The weight (tanh-normalized) is scaled by ``max_weight`` and clipped, with no
    magnitude filter, and kept where LEO exceeds its threshold.

    Args:
        weights: tanh-normalized weights from CPPN output 0.
        leo_outputs: tanh-normalized LEO values from CPPN output LEO_CHANNEL.
        leo_threshold: Minimum LEO value to express a connection.
        max_weight: Maximum weight magnitude.

    Returns:
        Weights where LEO expresses the connection, zero elsewhere.
    """
    weights = jnp.where(jnp.isnan(weights) | jnp.isinf(weights), 0.0, weights)
    leo_outputs = jnp.where(jnp.isnan(leo_outputs) | jnp.isinf(leo_outputs), 0.0, leo_outputs)
    leo_valid = leo_outputs > leo_threshold
    clipped = jnp.clip(weights * max_weight, -max_weight, max_weight)
    return jnp.where(leo_valid, clipped, 0.0)


def leo_sparsification_hmr_compound(
    weights: jnp.ndarray,
    leo_outputs: jnp.ndarray,
    weight_threshold: float,
    leo_threshold: float,
    max_weight: float,
) -> jnp.ndarray:
    """HMR's LEO gate (its LEO variant): the weight scaled, W = tanh(w) * max_weight, kept where |W|
    > weight_threshold and the LEO value exceeds its threshold. EMR's 'compound' thresholds the
    tanh-normalized weight and rescales the kept range linearly onto [0, max_weight].

    Args:
        weights: tanh-normalized weights (any shape).
        leo_outputs: tanh-normalized LEO values, same shape.
        weight_threshold: The threshold on |W| (HMR's 0.1).
        leo_threshold: The LEO threshold.
        max_weight: Maximum weight magnitude.

    Returns:
        The gated weights.
    """
    scaled = weights * max_weight
    return jnp.where(
        (jnp.abs(scaled) > weight_threshold) & (leo_outputs > leo_threshold), scaled, 0.0
    )


def leo_sparsify(
    weights: jnp.ndarray,
    leo_outputs: jnp.ndarray,
    leo_mode: str,
    weight_threshold: float,
    leo_threshold: float,
    max_weight: float,
) -> jnp.ndarray:
    """LEO-gated weights in the configured mode; the one rule of every execution mode.

    Args:
        weights: tanh-normalized weights (any shape).
        leo_outputs: tanh-normalized LEO values, same shape.
        leo_mode: 'compound', 'replacement' or 'hmr_compound' (see LEO_MODES).
        weight_threshold: The weight threshold of 'compound' mode.
        leo_threshold: The LEO threshold.
        max_weight: Maximum weight magnitude.

    Returns:
        The gated, scaled weights.
    """
    if leo_mode == 'replacement':
        return vectorized_weight_sparsification_leo_replacement(
            weights, leo_outputs, leo_threshold, max_weight)
    if leo_mode == 'hmr_compound':
        return leo_sparsification_hmr_compound(
            weights, leo_outputs, weight_threshold, leo_threshold, max_weight
        )
    return vectorized_weight_sparsification_with_leo(
        weights, leo_outputs, weight_threshold, leo_threshold, max_weight)
