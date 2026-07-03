"""Strategy 25: Intrinsic Plasticity (Per-Function Threshold/Gain Adaptation).

Implements intrinsic plasticity - neurons adapting their own excitability,
not just synaptic weights. This is orthogonal to all existing strategies
which only modify function selection.

Biological Basis:
- Neurons regulate their own firing threshold and gain based on activity
- Too much activity → increase threshold, decrease gain (dampening)
- Too little activity → decrease threshold, increase gain (amplifying)
- Maintains neurons in a homeostatic operating range
- Complements synaptic plasticity with intrinsic changes

Key Insight:
- All previous strategies (1-24) evolve WHICH functions to include
- Intrinsic plasticity evolves HOW RESPONSIVE each function is
- A function that's "too hot" (saturating) gets dampened
- A function that's "too cold" (rarely active) gets amplified

This strategy COMBINES intrinsic plasticity with Hebbian palette selection:
- Base layer: Hebbian co-occurrence learning for function selection
- Intrinsic layer: Threshold/gain adaptation for function tuning

Learning Rules:
    # For each function i:
    activity[i] = mean activation when function i is used
    error[i] = activity[i] - target_activity

    # Threshold adaptation (shift activation curve)
    threshold[i] -= threshold_lr * error[i]  # High activity → higher threshold

    # Gain adaptation (scale response magnitude)
    gain[i] *= (1 - gain_lr * error[i])  # High activity → lower gain

Expected improvements:
- Better function utilization (no over/under-active functions)
- Complementary to palette selection (tune what's selected)
- Homeostatic stability during evolution
- Faster convergence (optimal function responses)
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


class IntrinsicPlasticityStrategy(PaletteEvolutionStrategy):
    """Per-function threshold and gain adaptation with Hebbian selection.

    Intrinsic plasticity tunes how responsive each function is, while
    Hebbian learning determines which functions to include in the palette.
    """

    name = "intrinsic_plasticity"
    description = "Per-function threshold/gain tuning with Hebbian selection"

    def __init__(
        self,
        # Intrinsic plasticity parameters
        threshold_lr: float = 0.08,           # How fast thresholds adapt
        gain_lr: float = 0.04,                # How fast gains adapt
        target_activity: float = 0.5,         # Homeostatic target activity
        threshold_bounds: Tuple[float, float] = (-0.5, 0.5),  # Threshold range
        gain_bounds: Tuple[float, float] = (0.5, 2.0),        # Gain range
        # Hebbian parameters (for palette selection)
        hebbian_lr: float = 0.12,
        hebbian_decay: float = 0.02,
        affinity_protection: float = 0.6,
        # Mutation rates
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.05,
        # Activity tracking
        activity_momentum: float = 0.7,       # EMA for activity estimation
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Intrinsic Plasticity strategy.

        Args:
            threshold_lr: Learning rate for threshold adaptation
            gain_lr: Learning rate for gain adaptation
            target_activity: Homeostatic target for activity level
            threshold_bounds: Min/max for thresholds
            gain_bounds: Min/max for gains
            hebbian_lr: Learning rate for Hebbian co-occurrence
            hebbian_decay: Decay rate for unused functions
            affinity_protection: Affinity threshold for deactivation protection
        """
        # Intrinsic plasticity
        self.threshold_lr = threshold_lr
        self.gain_lr = gain_lr
        self.target_activity = target_activity
        self.threshold_bounds = threshold_bounds
        self.gain_bounds = gain_bounds

        # Hebbian
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.affinity_protection = affinity_protection

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # Activity
        self.activity_momentum = activity_momentum

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with intrinsic plasticity parameters."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Intrinsic plasticity parameters
        thresholds = jnp.zeros(NUM_ACTIVATIONS)  # Start at 0 (no bias)
        gains = jnp.ones(NUM_ACTIVATIONS)        # Start at 1 (no scaling)

        # Activity tracking (estimated activity for each function)
        activity_estimates = jnp.ones(NUM_ACTIVATIONS) * self.target_activity

        # Hebbian affinity for palette selection
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 252525),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Intrinsic plasticity state
            'thresholds': thresholds,
            'gains': gains,
            'activity_estimates': activity_estimates,
            # Hebbian state
            'function_affinity': function_affinity,
            # Tracking
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def get_intrinsic_params(self, state: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Return current intrinsic plasticity parameters.

        These can be used to modify how activation functions behave:
        output = gain[i] * activation_function(input - threshold[i])
        """
        return {
            'thresholds': state['thresholds'],
            'gains': state['gains'],
        }

    def _estimate_activity(
        self,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
    ) -> jnp.ndarray:
        """Estimate activity level for each function based on fitness contribution.

        Simplified model: active functions that contribute to fitness improvement
        are considered "high activity", others are "low activity".
        """
        # Fitness improvement signal
        fitness_delta = fitness - prev_fitness
        fitness_signal = 1.0 / (1.0 + jnp.exp(-fitness_delta * 10))  # Sigmoid

        # Active functions get activity proportional to fitness contribution
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)

        # Each active function gets a portion of the fitness signal
        # Functions in high-fitness runs are "more active"
        activity = active * (0.3 + 0.7 * fitness_signal)

        # Inactive functions have zero activity
        return activity

    def _update_intrinsic_params(
        self,
        thresholds: jnp.ndarray,
        gains: jnp.ndarray,
        activity_estimates: jnp.ndarray,
        new_activity: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update intrinsic plasticity parameters based on activity.

        Homeostatic adaptation:
        - High activity → increase threshold, decrease gain (dampen)
        - Low activity → decrease threshold, increase gain (amplify)
        """
        # Update activity estimates (EMA)
        active = (mask > 0.5).astype(jnp.float32)
        new_estimates = jnp.where(
            active > 0.5,
            self.activity_momentum * activity_estimates +
            (1 - self.activity_momentum) * new_activity,
            activity_estimates * 0.99  # Slow decay for inactive
        )

        # Compute activity error
        error = new_estimates - self.target_activity

        # Update thresholds (only for active functions)
        # High activity → positive error → increase threshold
        new_thresholds = thresholds - self.threshold_lr * error * active
        new_thresholds = jnp.clip(
            new_thresholds, self.threshold_bounds[0], self.threshold_bounds[1]
        )

        # Update gains (only for active functions)
        # High activity → positive error → decrease gain (multiply by < 1)
        gain_update = 1.0 - self.gain_lr * error
        new_gains = gains * jnp.where(active > 0.5, gain_update, 1.0)
        new_gains = jnp.clip(new_gains, self.gain_bounds[0], self.gain_bounds[1])

        return new_thresholds, new_gains, new_estimates

    def _update_hebbian_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_improved: bool,
        gains: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update Hebbian affinity based on co-occurrence with success.

        Novelty: Weight updates by intrinsic gain - functions that are
        well-tuned (gain near 1.0) get stronger affinity updates.
        """
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_improved:
            # Strengthen active functions, weighted by how well-tuned they are
            # Gain near 1.0 = well-tuned, stronger update
            tune_quality = 1.0 - jnp.abs(gains - 1.0)  # 1.0 when gain=1
            signal = self.hebbian_lr * active * (0.5 + 0.5 * tune_quality)
        else:
            # Weaken active functions slightly
            signal = -self.hebbian_lr * 0.3 * active

        new_affinity = affinity + signal

        # Apply decay to inactive functions
        inactive = 1.0 - active
        new_affinity = new_affinity - self.hebbian_decay * inactive * affinity

        return jnp.clip(new_affinity, 0.05, 0.95)

    def _apply_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        gains: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with affinity and gain-based biasing.

        Functions with well-tuned gains (close to 1.0) are more likely
        to be activated - they require less adaptation.
        """
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_ACTIVATIONS):
            aff = float(affinity[i])
            gain = float(gains[i])
            tune_quality = 1.0 - abs(gain - 1.0)  # How well-tuned is this function

            if mask[i] < 0.5:
                # Inactive: maybe activate
                # Higher affinity and better tune quality → more likely
                rate = self.base_activate_rate * (0.5 + 0.5 * aff) * (0.7 + 0.3 * tune_quality)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                if aff >= self.affinity_protection:
                    rate = self.base_deactivate_rate * 0.1
                else:
                    # Also consider how well-tuned - poorly tuned more likely to deactivate
                    rate = self.base_deactivate_rate * (1.0 - aff) * (2.0 - tune_quality)
                    rate = min(rate, self.base_deactivate_rate * 2)  # Cap rate

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update intrinsic plasticity and Hebbian affinity."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Estimate activity from fitness
        new_activity = self._estimate_activity(
            state['mask'], best_fitness, prev_best_fitness
        )

        # Step 2: Update intrinsic plasticity parameters
        new_thresholds, new_gains, new_estimates = self._update_intrinsic_params(
            state['thresholds'],
            state['gains'],
            state['activity_estimates'],
            new_activity,
            state['mask'],
        )

        # Step 3: Update Hebbian affinity
        new_affinity = self._update_hebbian_affinity(
            state['function_affinity'],
            state['mask'],
            improved,
            new_gains,
        )

        # Step 4: Apply mutation
        new_mask, mutation_info = self._apply_mutation(
            subkey, state['mask'], new_affinity, new_gains
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track fitness
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
            # Intrinsic plasticity
            'thresholds': new_thresholds,
            'gains': new_gains,
            'activity_estimates': new_estimates,
            # Hebbian
            'function_affinity': new_affinity,
            # Tracking
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)
        protected = [
            i for i in active_palette
            if new_affinity[i] >= self.affinity_protection
        ]

        # Functions with extreme gains (need tuning)
        extreme_gains = [
            (int(i), float(new_gains[i]))
            for i in range(NUM_ACTIVATIONS)
            if abs(float(new_gains[i]) - 1.0) > 0.3
        ]

        # Top affinity functions
        top_aff_idx = jnp.argsort(new_affinity)[-3:][::-1]
        top_affinity = [(int(i), float(new_affinity[i])) for i in top_aff_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Intrinsic plasticity
            'mean_threshold': float(jnp.mean(new_thresholds)),
            'mean_gain': float(jnp.mean(new_gains)),
            'threshold_std': float(jnp.std(new_thresholds)),
            'gain_std': float(jnp.std(new_gains)),
            'extreme_gain_functions': extreme_gains[:5],
            'sin_threshold': float(new_thresholds[4]),
            'sin_gain': float(new_gains[4]),
            # Hebbian
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'top_affinity_functions': top_affinity,
            'sin_affinity': float(new_affinity[4]),
            # Protection
            'n_protected': len(protected),
            'protected_functions': protected,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including intrinsic parameters."""
        palette = self.get_active_palette(state)
        thresholds = state['thresholds']
        gains = state['gains']
        affinity = state['function_affinity']

        # Top functions by affinity
        top_aff_idx = jnp.argsort(affinity)[-5:][::-1]
        top_affinities = [(int(i), float(affinity[i])) for i in top_aff_idx]

        # Functions with most gain adaptation
        gain_deviation = jnp.abs(gains - 1.0)
        top_gain_idx = jnp.argsort(gain_deviation)[-5:][::-1]
        most_adapted = [
            (int(i), float(gains[i]), float(thresholds[i]))
            for i in top_gain_idx
        ]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Intrinsic plasticity
            'intrinsic_params': {
                'mean_threshold': float(jnp.mean(thresholds)),
                'mean_gain': float(jnp.mean(gains)),
                'most_adapted_functions': most_adapted,
            },
            # Hebbian
            'top_affinity_functions': top_affinities,
            # Sin-specific
            'sin_affinity': float(affinity[4]),
            'sin_threshold': float(thresholds[4]),
            'sin_gain': float(gains[4]),
        }
