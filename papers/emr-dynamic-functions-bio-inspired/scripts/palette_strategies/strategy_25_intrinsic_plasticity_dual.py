"""Strategy 25D: Intrinsic Plasticity Dual (Threshold/Gain Adaptation for Both Palettes).

Extends Intrinsic Plasticity to jointly evolve activation AND aggregation function
palettes with per-function threshold and gain adaptation for both domains.

Key mechanisms:
1. Threshold adaptation: Shift activation curves based on activity
2. Gain adaptation: Scale response magnitude based on activity
3. Homeostatic target: Maintain functions in optimal operating range
4. Cross-domain learning: Share intrinsic params between domains

Biological basis:
- Neurons regulate their own firing threshold and gain
- Too much activity → increase threshold, decrease gain
- Too little activity → decrease threshold, increase gain
- Applies to both computational and integrative functions
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

NUM_AGGREGATIONS = 6
DEFAULT_AGGREGATION_INDICES = [0, 1]


def create_initial_agg_mask(indices: List[int]) -> jnp.ndarray:
    mask = jnp.zeros(NUM_AGGREGATIONS)
    for idx in indices:
        if 0 <= idx < NUM_AGGREGATIONS:
            mask = mask.at[idx].set(1.0)
    return mask


def agg_mask_to_indices(mask: jnp.ndarray) -> List[int]:
    return [int(i) for i in range(NUM_AGGREGATIONS) if mask[i] > 0.5]


class IntrinsicPlasticityDualStrategy(PaletteEvolutionStrategy):
    """Per-function threshold/gain adaptation for both activation and aggregation palettes."""

    name = "intrinsic_plasticity_dual"
    description = "Intrinsic plasticity for dual palette evolution"

    def __init__(
        self,
        # Intrinsic plasticity parameters
        threshold_lr: float = 0.08,
        gain_lr: float = 0.04,
        target_activity: float = 0.5,
        threshold_bounds: Tuple[float, float] = (-0.5, 0.5),
        gain_bounds: Tuple[float, float] = (0.5, 2.0),
        # Hebbian parameters
        hebbian_lr: float = 0.12,
        hebbian_decay: float = 0.02,
        act_affinity_protection: float = 0.6,
        agg_affinity_protection: float = 0.6,
        # Mutation rates
        act_base_activate_rate: float = 0.12,
        act_base_deactivate_rate: float = 0.05,
        agg_base_activate_rate: float = 0.10,
        agg_base_deactivate_rate: float = 0.04,
        # Activity tracking
        activity_momentum: float = 0.7,
        # Cross-domain
        cross_learning_rate: float = 0.06,
        # Constraints
        max_active_act: int = 6,
        max_active_agg: int = 4,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        # General
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        # Intrinsic plasticity
        self.threshold_lr = threshold_lr
        self.gain_lr = gain_lr
        self.target_activity = target_activity
        self.threshold_bounds = threshold_bounds
        self.gain_bounds = gain_bounds

        # Hebbian
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.act_affinity_protection = act_affinity_protection
        self.agg_affinity_protection = agg_affinity_protection

        # Mutation
        self.act_base_activate_rate = act_base_activate_rate
        self.act_base_deactivate_rate = act_base_deactivate_rate
        self.agg_base_activate_rate = agg_base_activate_rate
        self.agg_base_deactivate_rate = agg_base_deactivate_rate

        # Activity
        self.activity_momentum = activity_momentum

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        # Constraints
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        # Activation
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_thresholds = jnp.zeros(NUM_ACTIVATIONS)
        act_gains = jnp.ones(NUM_ACTIVATIONS)
        act_activity_estimates = jnp.ones(NUM_ACTIVATIONS) * self.target_activity
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Aggregation
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_mask(initial_agg)
        agg_thresholds = jnp.zeros(NUM_AGGREGATIONS)
        agg_gains = jnp.ones(NUM_AGGREGATIONS)
        agg_activity_estimates = jnp.ones(NUM_AGGREGATIONS) * self.target_activity
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation state
            'act_mask': act_mask,
            'act_thresholds': act_thresholds,
            'act_gains': act_gains,
            'act_activity_estimates': act_activity_estimates,
            'act_affinity': act_affinity,
            # Aggregation state
            'agg_mask': agg_mask,
            'agg_thresholds': agg_thresholds,
            'agg_gains': agg_gains,
            'agg_activity_estimates': agg_activity_estimates,
            'agg_affinity': agg_affinity,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # General
            'rng_key': jax.random.PRNGKey(seed + 252525),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            'mask': act_mask,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return agg_mask_to_indices(state['agg_mask'])

    def get_intrinsic_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return current intrinsic plasticity parameters for both domains."""
        return {
            'act_thresholds': state['act_thresholds'],
            'act_gains': state['act_gains'],
            'agg_thresholds': state['agg_thresholds'],
            'agg_gains': state['agg_gains'],
        }

    def _estimate_activity(
        self,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
        num_funcs: int,
    ) -> jnp.ndarray:
        """Estimate activity level based on fitness contribution."""
        fitness_delta = fitness - prev_fitness
        fitness_signal = 1.0 / (1.0 + jnp.exp(-fitness_delta * 10))

        active = (mask > 0.5).astype(jnp.float32)
        activity = active * (0.3 + 0.7 * fitness_signal)

        return activity

    def _update_intrinsic_params(
        self,
        thresholds: jnp.ndarray,
        gains: jnp.ndarray,
        activity_estimates: jnp.ndarray,
        new_activity: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update intrinsic plasticity parameters based on activity."""
        active = (mask > 0.5).astype(jnp.float32)

        # Update activity estimates (EMA)
        new_estimates = jnp.where(
            active > 0.5,
            self.activity_momentum * activity_estimates + (1 - self.activity_momentum) * new_activity,
            activity_estimates * 0.99
        )

        # Compute activity error
        error = new_estimates - self.target_activity

        # Update thresholds
        new_thresholds = thresholds - self.threshold_lr * error * active
        new_thresholds = jnp.clip(new_thresholds, self.threshold_bounds[0], self.threshold_bounds[1])

        # Update gains
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
        """Update Hebbian affinity weighted by intrinsic gain."""
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_improved:
            tune_quality = 1.0 - jnp.abs(gains - 1.0)
            signal = self.hebbian_lr * active * (0.5 + 0.5 * tune_quality)
        else:
            signal = -self.hebbian_lr * 0.3 * active

        new_affinity = affinity + signal

        # Decay inactive
        inactive = 1.0 - active
        new_affinity = new_affinity - self.hebbian_decay * inactive * affinity

        return jnp.clip(new_affinity, 0.05, 0.95)

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_improved: bool,
        act_gains: jnp.ndarray,
        agg_gains: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update cross-domain affinity weighted by both gains."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)

        # Weight by average tune quality
        act_tune = 1.0 - jnp.abs(act_gains - 1.0)
        agg_tune = 1.0 - jnp.abs(agg_gains - 1.0)
        tune_weight = jnp.outer(act_tune, agg_tune)

        if fitness_improved:
            delta = self.cross_learning_rate * cross_active * (0.5 + 0.5 * tune_weight)
        else:
            delta = -self.cross_learning_rate * 0.3 * cross_active

        return jnp.clip(cross_affinity + delta, 0.0, 1.0)

    def _compute_protection_scores_act(
        self,
        affinity: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_agg = max(jnp.sum(agg_active), 1)
        cross_score = jnp.dot(cross_affinity, agg_active) / n_agg

        return 0.70 * affinity + 0.30 * cross_score

    def _compute_protection_scores_agg(
        self,
        affinity: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        act_active = (act_mask > 0.5).astype(jnp.float32)
        n_act = max(jnp.sum(act_active), 1)
        cross_score = jnp.dot(cross_affinity.T, act_active) / n_act

        return 0.70 * affinity + 0.30 * cross_score

    def _apply_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        gains: jnp.ndarray,
        is_activation: bool,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with affinity and gain-based biasing."""
        key1, key2 = jax.random.split(key)

        if is_activation:
            num_funcs = NUM_ACTIVATIONS
            max_active = self.max_active_act
            min_active = self.min_active_act
            protection_threshold = self.act_affinity_protection
            base_activate = self.act_base_activate_rate
            base_deactivate = self.act_base_deactivate_rate
        else:
            num_funcs = NUM_AGGREGATIONS
            max_active = self.max_active_agg
            min_active = self.min_active_agg
            protection_threshold = self.agg_affinity_protection
            base_activate = self.agg_base_activate_rate
            base_deactivate = self.agg_base_deactivate_rate

        activate_probs = jax.random.uniform(key1, (num_funcs,))
        deactivate_probs = jax.random.uniform(key2, (num_funcs,))

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(num_funcs):
            aff = float(affinity[i])
            gain = float(gains[i])
            tune_quality = 1.0 - abs(gain - 1.0)

            if mask[i] < 0.5:
                current_active = int(jnp.sum(mask > 0.5))
                if current_active >= max_active:
                    continue

                rate = base_activate * (0.5 + 0.5 * aff) * (0.7 + 0.3 * tune_quality)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if aff >= protection_threshold:
                    rate = base_deactivate * 0.1
                else:
                    rate = base_deactivate * (1.0 - aff) * (2.0 - tune_quality)
                    rate = min(rate, base_deactivate * 2)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
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
        key, subkey1, subkey2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Estimate activity for both domains
        act_activity = self._estimate_activity(
            state['act_mask'], best_fitness, prev_best_fitness, NUM_ACTIVATIONS
        )
        agg_activity = self._estimate_activity(
            state['agg_mask'], best_fitness, prev_best_fitness, NUM_AGGREGATIONS
        )

        # Update intrinsic params
        new_act_thresholds, new_act_gains, new_act_estimates = self._update_intrinsic_params(
            state['act_thresholds'], state['act_gains'], state['act_activity_estimates'],
            act_activity, state['act_mask']
        )
        new_agg_thresholds, new_agg_gains, new_agg_estimates = self._update_intrinsic_params(
            state['agg_thresholds'], state['agg_gains'], state['agg_activity_estimates'],
            agg_activity, state['agg_mask']
        )

        # Update Hebbian affinity
        new_act_affinity = self._update_hebbian_affinity(
            state['act_affinity'], state['act_mask'], improved, new_act_gains
        )
        new_agg_affinity = self._update_hebbian_affinity(
            state['agg_affinity'], state['agg_mask'], improved, new_agg_gains
        )

        # Update cross-domain
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'],
            improved, new_act_gains, new_agg_gains
        )

        # Apply mutations
        new_act_mask, act_mut = self._apply_mutation(
            subkey1, state['act_mask'], new_act_affinity, new_act_gains, True
        )
        new_agg_mask, agg_mut = self._apply_mutation(
            subkey2, state['agg_mask'], new_agg_affinity, new_agg_gains, False
        )

        # Track fitness
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_thresholds': new_act_thresholds,
            'act_gains': new_act_gains,
            'act_activity_estimates': new_act_estimates,
            'act_affinity': new_act_affinity,
            'agg_mask': new_agg_mask,
            'agg_thresholds': new_agg_thresholds,
            'agg_gains': new_agg_gains,
            'agg_activity_estimates': new_agg_estimates,
            'agg_affinity': new_agg_affinity,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'mask': new_act_mask,
        }

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': agg_mask_to_indices(new_agg_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Activation intrinsic stats
            'act_mean_threshold': float(jnp.mean(new_act_thresholds)),
            'act_mean_gain': float(jnp.mean(new_act_gains)),
            'sin_threshold': float(new_act_thresholds[4]),
            'sin_gain': float(new_act_gains[4]),
            # Aggregation intrinsic stats
            'agg_mean_threshold': float(jnp.mean(new_agg_thresholds)),
            'agg_mean_gain': float(jnp.mean(new_agg_gains)),
            # Affinity stats
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'sin_affinity': float(new_act_affinity[4]),
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Mutations
            'act_activated': act_mut['activated'],
            'act_deactivated': act_mut['deactivated'],
            'agg_activated': agg_mut['activated'],
            'agg_deactivated': agg_mut['deactivated'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Intrinsic params
            'act_mean_threshold': float(jnp.mean(state['act_thresholds'])),
            'act_mean_gain': float(jnp.mean(state['act_gains'])),
            'agg_mean_threshold': float(jnp.mean(state['agg_thresholds'])),
            'agg_mean_gain': float(jnp.mean(state['agg_gains'])),
            # Affinity
            'sin_affinity': float(state['act_affinity'][4]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
        }
