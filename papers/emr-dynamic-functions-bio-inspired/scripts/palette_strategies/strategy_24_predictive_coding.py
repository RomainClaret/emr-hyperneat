"""Strategy 24: Predictive Coding (Prediction-Error-Driven Learning).

Implements cortical predictive coding where learning is driven by prediction
errors (surprise), not raw correlations.

Biological Basis:
- Cortical hierarchies constantly predict incoming signals
- Learning is driven by prediction ERRORS, not raw activity
- Surprise (unexpected outcomes) teaches more than expected outcomes
- This is accumulating evidence as THE computational principle of the brain

Key Insight:
- Current Hebbian strategies: Learn from correlation (active + successful = good)
- Predictive coding: Learn from prediction error (surprising outcomes teach more)
- Functions that produce unexpected fitness improvements get strongest updates

Learning Rule:
    prediction[i] = expected_usefulness[i] (from previous affinity)
    actual[i] = fitness_contribution if function[i] active else 0
    prediction_error[i] = actual[i] - prediction[i]
    affinity[i] += lr * prediction_error[i]

    Global surprise = mean(|prediction_error|)
    If surprise > threshold: explore more (confused state)
    If surprise < threshold: exploit more (confident state)

Expected improvements:
- Natural novelty detection (unexpected functions get attention)
- Automatic difficulty adaptation
- Meta-learning capability (learns to learn)
- Avoids getting stuck on over-predicted patterns
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    DEFAULT_PALETTE_INDICES,
)


class PredictiveCodingStrategy(PaletteEvolutionStrategy):
    """Learn by prediction error (surprise), not correlation.

    Predictive coding suggests that the brain constantly generates
    predictions and learns primarily from prediction errors. Functions
    that produce UNEXPECTED fitness improvements teach the most.
    """

    name = "predictive_coding"
    description = "Prediction-error-driven learning (surprise teaches)"

    def __init__(
        self,
        # Prediction parameters
        prediction_lr: float = 0.15,              # Learning rate for predictions
        error_sensitivity: float = 1.5,           # How much errors affect affinity
        error_momentum: float = 0.4,              # Smoothing for prediction errors
        # Surprise-based exploration
        surprise_threshold: float = 0.15,         # Threshold for "confused" state
        surprise_exploration_boost: float = 1.5,  # Explore more when surprised
        surprise_exploit_factor: float = 0.7,     # Exploit more when confident
        # Affinity parameters
        affinity_baseline: float = 0.5,           # Starting affinity
        affinity_protection_threshold: float = 0.65,
        # Mutation rates
        base_activate_rate: float = 0.10,
        base_deactivate_rate: float = 0.06,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Predictive Coding strategy.

        Args:
            prediction_lr: How fast predictions update based on actual outcomes
            error_sensitivity: Scale factor for prediction error → affinity update
            error_momentum: EMA momentum for smoothing prediction errors
            surprise_threshold: Above this, system is "confused" and explores
            surprise_exploration_boost: Exploration multiplier when surprised
            surprise_exploit_factor: Exploration multiplier when confident
            affinity_baseline: Initial affinity for all functions
            affinity_protection_threshold: Threshold for protection from deactivation
        """
        # Prediction
        self.prediction_lr = prediction_lr
        self.error_sensitivity = error_sensitivity
        self.error_momentum = error_momentum

        # Surprise-based modulation
        self.surprise_threshold = surprise_threshold
        self.surprise_exploration_boost = surprise_exploration_boost
        self.surprise_exploit_factor = surprise_exploit_factor

        # Affinity
        self.affinity_baseline = affinity_baseline
        self.affinity_protection_threshold = affinity_protection_threshold

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with prediction system."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Predictions: expected usefulness of each function
        predictions = jnp.ones(NUM_ACTIVATIONS) * self.affinity_baseline

        # Prediction errors: actual - predicted (smoothed)
        prediction_errors = jnp.zeros(NUM_ACTIVATIONS)

        # Affinity: learned value of each function
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * self.affinity_baseline

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 242424),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Prediction system
            'predictions': predictions,
            'prediction_errors': prediction_errors,
            'function_affinity': function_affinity,
            # Global state
            'surprise': 0.0,  # Mean absolute prediction error
            'fitness_baseline': 0.5,  # Expected fitness (for computing actual)
            # Tracking
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_actual_contributions(
        self,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
        fitness_baseline: float,
    ) -> jnp.ndarray:
        """Compute actual fitness contribution for each function.

        Active functions get credit proportional to fitness improvement.
        Inactive functions get zero contribution.
        """
        # Fitness improvement relative to baseline
        if fitness_baseline > 0.01:
            fitness_signal = (fitness - fitness_baseline) / fitness_baseline
        else:
            fitness_signal = fitness - fitness_baseline

        # Clip to reasonable range
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Active functions share the credit
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)

        # Each active function gets proportional contribution
        # (simplified - in reality we'd want per-function contribution estimation)
        per_function_contribution = fitness_signal / n_active

        # Only active functions get non-zero actual contribution
        actual = active * per_function_contribution

        return actual

    def _update_predictions(
        self,
        predictions: jnp.ndarray,
        actuals: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update predictions based on actual outcomes.

        Predictions move toward actuals for active functions.
        Inactive function predictions decay toward baseline.

        Returns:
            (new_predictions, prediction_errors)
        """
        # Compute raw prediction error
        raw_error = actuals - predictions

        # Only update predictions for active functions
        active = (mask > 0.5).astype(jnp.float32)

        # Active: move prediction toward actual
        # Inactive: decay toward baseline
        baseline = jnp.ones(NUM_ACTIVATIONS) * self.affinity_baseline
        inactive_target = 0.95 * predictions + 0.05 * baseline

        new_predictions = jnp.where(
            active > 0.5,
            predictions + self.prediction_lr * raw_error,
            inactive_target
        )

        # Clip predictions to valid range
        new_predictions = jnp.clip(new_predictions, 0.0, 1.0)

        return new_predictions, raw_error

    def _update_affinity_from_errors(
        self,
        affinity: jnp.ndarray,
        prediction_errors: jnp.ndarray,
        smoothed_errors: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update affinity based on prediction errors.

        Key insight: Surprising outcomes (large prediction errors) drive
        the strongest learning. Expected outcomes don't change much.

        Returns:
            (new_affinity, new_smoothed_errors)
        """
        # Smooth prediction errors (EMA)
        new_smoothed = (
            self.error_momentum * smoothed_errors +
            (1 - self.error_momentum) * prediction_errors
        )

        # Affinity update driven by smoothed prediction error
        # Positive error (better than expected) → increase affinity
        # Negative error (worse than expected) → decrease affinity
        delta = self.error_sensitivity * new_smoothed

        new_affinity = affinity + delta

        # Clip to valid range
        new_affinity = jnp.clip(new_affinity, 0.05, 0.95)

        return new_affinity, new_smoothed

    def _compute_surprise(self, prediction_errors: jnp.ndarray) -> float:
        """Compute global surprise as mean absolute prediction error.

        High surprise = system is confused, predictions don't match reality
        Low surprise = system is confident, predictions are accurate
        """
        return float(jnp.mean(jnp.abs(prediction_errors)))

    def _surprise_modulated_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        surprise: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with surprise-modulated exploration.

        High surprise → explore more (confused, need to try things)
        Low surprise → exploit more (confident, stick with what works)
        """
        key1, key2 = jax.random.split(key)

        # Surprise modulates exploration
        if surprise > self.surprise_threshold:
            # Confused: explore more
            exploration_factor = self.surprise_exploration_boost
        else:
            # Confident: exploit more
            exploration_factor = self.surprise_exploit_factor

        effective_activate_rate = self.base_activate_rate * exploration_factor
        effective_deactivate_rate = self.base_deactivate_rate * exploration_factor

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            aff = float(affinity[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                # Higher affinity → more likely to activate
                rate = effective_activate_rate * (0.5 + 0.5 * aff)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                if aff >= self.affinity_protection_threshold:
                    # Protected: very low deactivation
                    rate = effective_deactivate_rate * 0.1
                else:
                    # Unprotected: inversely proportional to affinity
                    rate = effective_deactivate_rate * (1.0 - aff)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'exploration_factor': exploration_factor,
            'effective_activate_rate': effective_activate_rate,
            'effective_deactivate_rate': effective_deactivate_rate,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with predictive coding learning."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update fitness baseline (EMA)
        new_fitness_baseline = (
            0.9 * state['fitness_baseline'] +
            0.1 * best_fitness
        )

        # Step 1: Compute actual contributions for each function
        actuals = self._compute_actual_contributions(
            state['mask'],
            best_fitness,
            prev_best_fitness,
            state['fitness_baseline'],
        )

        # Step 2: Update predictions and get prediction errors
        new_predictions, prediction_errors = self._update_predictions(
            state['predictions'],
            actuals,
            state['mask'],
        )

        # Step 3: Update affinity from prediction errors
        new_affinity, new_smoothed_errors = self._update_affinity_from_errors(
            state['function_affinity'],
            prediction_errors,
            state['prediction_errors'],
        )

        # Step 4: Compute global surprise
        surprise = self._compute_surprise(new_smoothed_errors)

        # Step 5: Surprise-modulated mutation
        new_mask, mutation_info = self._surprise_modulated_mutation(
            subkey,
            state['mask'],
            new_affinity,
            surprise,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update fitness history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Prediction system
            'predictions': new_predictions,
            'prediction_errors': new_smoothed_errors,
            'function_affinity': new_affinity,
            # Global state
            'surprise': surprise,
            'fitness_baseline': new_fitness_baseline,
            # Tracking
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)
        protected = [
            i for i in active_palette
            if new_affinity[i] >= self.affinity_protection_threshold
        ]

        # Top functions by prediction error magnitude (most surprising)
        error_magnitudes = jnp.abs(new_smoothed_errors)
        top_surprise_idx = jnp.argsort(error_magnitudes)[-3:][::-1]
        top_surprising = [
            (int(i), float(new_smoothed_errors[i]))
            for i in top_surprise_idx
        ]

        # Top functions by affinity
        top_aff_idx = jnp.argsort(new_affinity)[-3:][::-1]
        top_affinity = [(int(i), float(new_affinity[i])) for i in top_aff_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Prediction
            'surprise': surprise,
            'avg_prediction': float(jnp.mean(new_predictions)),
            'max_prediction_error': float(jnp.max(jnp.abs(new_smoothed_errors))),
            'top_surprising_functions': top_surprising,
            # Affinity
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'top_affinity_functions': top_affinity,
            'sin_affinity': float(new_affinity[4]),
            'sin_prediction_error': float(new_smoothed_errors[4]),
            # Protection
            'n_protected': len(protected),
            'protected_functions': protected,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with predictive coding stats."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']
        predictions = state['predictions']
        errors = state['prediction_errors']

        # Top functions by affinity
        top_aff_idx = jnp.argsort(affinity)[-5:][::-1]
        top_affinities = [(int(i), float(affinity[i])) for i in top_aff_idx]

        # Most surprising (highest absolute error)
        error_mags = jnp.abs(errors)
        top_err_idx = jnp.argsort(error_mags)[-5:][::-1]
        top_errors = [(int(i), float(errors[i])) for i in top_err_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Prediction
            'surprise': state['surprise'],
            'avg_prediction': float(jnp.mean(predictions)),
            'prediction_accuracy': 1.0 - state['surprise'],  # Higher = more accurate
            # Top functions
            'top_affinity_functions': top_affinities,
            'most_surprising_functions': top_errors,
            # Sin-specific
            'sin_affinity': float(affinity[4]),
            'sin_prediction': float(predictions[4]),
            'sin_prediction_error': float(errors[4]),
            # Global
            'fitness_baseline': state['fitness_baseline'],
        }
