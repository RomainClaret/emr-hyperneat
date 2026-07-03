"""Strategy 24 Dual: Predictive Coding for Both Activation AND Aggregation.

Extends PredictiveCoding to jointly evolve both activation and aggregation palettes
with prediction-error-driven learning in both domains.

Key mechanisms extended to dual:
1. Separate prediction systems for activations and aggregations
2. Surprise computed for each domain independently
3. Cross-domain predictions track expected act-agg combinations
4. Meta-learning adapts exploration based on combined surprise
5. Max_active constraints prevent antagonism (max_act=6, max_agg=4)

Biological rationale:
- Predictive coding operates throughout the cortical hierarchy
- Different modalities have independent prediction systems
- Global surprise integrates across domains

Expected improvement:
- Natural novelty detection in BOTH domains
- Automatic difficulty adaptation in each domain
- Cross-domain surprise-based exploration
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
)


class PredictiveCodingDualStrategy(PaletteEvolutionStrategy):
    """Prediction-error-driven learning with dual palette evolution.

    Surprise-based learning in both activation and aggregation domains.
    """

    name = "predictive_coding_dual"
    description = "Dual palette prediction-error-driven learning"

    def __init__(
        self,
        # Prediction parameters
        prediction_lr: float = 0.15,
        error_sensitivity: float = 1.5,
        error_momentum: float = 0.4,
        # Surprise-based exploration
        surprise_threshold: float = 0.15,
        surprise_exploration_boost: float = 1.5,
        surprise_exploit_factor: float = 0.7,
        # Cross-domain
        cross_learning_rate: float = 0.12,
        cross_influence: float = 0.25,
        # Affinity
        affinity_baseline: float = 0.5,
        affinity_protection_threshold: float = 0.65,
        # Mutation
        base_activate_rate: float = 0.10,
        base_deactivate_rate: float = 0.06,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.prediction_lr = prediction_lr
        self.error_sensitivity = error_sensitivity
        self.error_momentum = error_momentum
        self.surprise_threshold = surprise_threshold
        self.surprise_exploration_boost = surprise_exploration_boost
        self.surprise_exploit_factor = surprise_exploit_factor
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        self.affinity_baseline = affinity_baseline
        self.affinity_protection_threshold = affinity_protection_threshold
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        act_mask = create_initial_palette_mask(config.get('initial_act_palette', self.initial_act_palette))
        agg_mask = create_initial_agg_palette_mask(config.get('initial_agg_palette', self.initial_agg_palette))

        return {
            'act_mask': act_mask,
            'act_predictions': jnp.ones(NUM_ACTIVATIONS) * self.affinity_baseline,
            'act_errors': jnp.zeros(NUM_ACTIVATIONS),
            'act_affinity': jnp.ones(NUM_ACTIVATIONS) * self.affinity_baseline,
            'agg_mask': agg_mask,
            'agg_predictions': jnp.ones(NUM_AGGREGATIONS) * self.affinity_baseline,
            'agg_errors': jnp.zeros(NUM_AGGREGATIONS),
            'agg_affinity': jnp.ones(NUM_AGGREGATIONS) * self.affinity_baseline,
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,
            'rng_key': jax.random.PRNGKey(seed + 242425),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'act_surprise': 0.0,
            'agg_surprise': 0.0,
            'fitness_baseline': 0.5,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_actuals(self, mask: jnp.ndarray, fitness: float, baseline: float) -> jnp.ndarray:
        if baseline > 0.01:
            fs = (fitness - baseline) / baseline
        else:
            fs = fitness - baseline
        fs = max(-1.0, min(1.0, fs))
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)
        return active * (fs / n_active)

    def _update_predictions(
        self, preds: jnp.ndarray, actuals: jnp.ndarray, mask: jnp.ndarray, n_funcs: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raw_error = actuals - preds
        active = (mask > 0.5).astype(jnp.float32)
        baseline = jnp.ones(n_funcs) * self.affinity_baseline
        inactive_target = 0.95 * preds + 0.05 * baseline
        new_preds = jnp.where(active > 0.5, preds + self.prediction_lr * raw_error, inactive_target)
        return jnp.clip(new_preds, 0.0, 1.0), raw_error

    def _update_affinity(
        self, aff: jnp.ndarray, errors: jnp.ndarray, smoothed: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        new_smoothed = self.error_momentum * smoothed + (1 - self.error_momentum) * errors
        delta = self.error_sensitivity * new_smoothed
        return jnp.clip(aff + delta, 0.05, 0.95), new_smoothed

    def _compute_surprise(self, errors: jnp.ndarray) -> float:
        return float(jnp.mean(jnp.abs(errors)))

    def _mutate_palette(
        self, key: jax.random.PRNGKey, mask: jnp.ndarray, aff: jnp.ndarray,
        surprise: float, cross: jnp.ndarray, other_mask: jnp.ndarray,
        n_funcs: int, min_active: int, max_active: int, is_act: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []

        if surprise > self.surprise_threshold:
            exp_factor = self.surprise_exploration_boost
        else:
            exp_factor = self.surprise_exploit_factor

        eff_act = self.base_activate_rate * exp_factor
        eff_deact = self.base_deactivate_rate * exp_factor

        # Add cross-domain influence
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)
        if is_act:
            cross_score = jnp.dot(cross, other_active) / n_other
        else:
            cross_score = jnp.dot(cross.T, other_active) / n_other

        effective_aff = 0.8 * aff + 0.2 * cross_score * self.cross_influence

        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))
        current = int(jnp.sum(mask > 0.5))

        for i in range(n_funcs):
            a = float(effective_aff[i])
            if mask[i] < 0.5:
                if current + len(activated) >= max_active:
                    continue
                rate = eff_act * (0.5 + 0.5 * a)
                if act_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if a >= self.affinity_protection_threshold:
                    rate = eff_deact * 0.1
                else:
                    rate = eff_deact * (1.0 - a)
                if deact_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            return mask, {'activated': [], 'deactivated': [], 'exploration_factor': exp_factor}

        prefix = 'act_' if is_act else 'agg_'
        return new_mask, {f'{prefix}activated': activated, f'{prefix}deactivated': deactivated, f'{prefix}exp_factor': exp_factor}

    def post_generation_update(
        self, state: Dict[str, Any], generation: int, best_fitness: float,
        prev_best_fitness: float, population_data: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        new_fitness_baseline = 0.9 * state['fitness_baseline'] + 0.1 * best_fitness

        # Compute actuals
        act_actuals = self._compute_actuals(state['act_mask'], best_fitness, state['fitness_baseline'])
        agg_actuals = self._compute_actuals(state['agg_mask'], best_fitness, state['fitness_baseline'])

        # Update predictions
        new_act_preds, act_errors = self._update_predictions(state['act_predictions'], act_actuals, state['act_mask'], NUM_ACTIVATIONS)
        new_agg_preds, agg_errors = self._update_predictions(state['agg_predictions'], agg_actuals, state['agg_mask'], NUM_AGGREGATIONS)

        # Update affinities
        new_act_aff, new_act_errors = self._update_affinity(state['act_affinity'], act_errors, state['act_errors'])
        new_agg_aff, new_agg_errors = self._update_affinity(state['agg_affinity'], agg_errors, state['agg_errors'])

        # Cross-domain
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        fs = max(-1.0, min(1.0, (best_fitness - new_fitness_baseline) / max(0.1, new_fitness_baseline)))
        cross_delta = self.cross_learning_rate * fs * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Compute surprise
        act_surprise = self._compute_surprise(new_act_errors)
        agg_surprise = self._compute_surprise(new_agg_errors)
        combined_surprise = 0.6 * act_surprise + 0.4 * agg_surprise

        # Mutations
        new_act_mask, act_mut = self._mutate_palette(k_act, state['act_mask'], new_act_aff, combined_surprise, new_cross, state['agg_mask'], NUM_ACTIVATIONS, self.min_active_act, self.max_active_act, True)
        new_agg_mask, agg_mut = self._mutate_palette(k_agg, state['agg_mask'], new_agg_aff, combined_surprise, new_cross, state['act_mask'], NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg, False)

        fh = state['fitness_history'] + [best_fitness]
        if len(fh) > 20:
            fh = fh[-20:]

        new_state = {
            'act_mask': new_act_mask, 'act_predictions': new_act_preds, 'act_errors': new_act_errors, 'act_affinity': new_act_aff,
            'agg_mask': new_agg_mask, 'agg_predictions': new_agg_preds, 'agg_errors': new_agg_errors, 'agg_affinity': new_agg_aff,
            'cross_affinity': new_cross, 'rng_key': key, 'generation': generation + 1, 'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best, 'strategy_name': self.name, 'act_surprise': act_surprise, 'agg_surprise': agg_surprise,
            'fitness_baseline': new_fitness_baseline, 'fitness_history': fh,
        }

        metrics = {
            'act_palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'act_surprise': act_surprise, 'agg_surprise': agg_surprise, 'combined_surprise': combined_surprise,
            'sin_affinity': float(new_act_aff[4]) if 4 < len(new_act_aff) else 0.0,
            'sin_prediction_error': float(new_act_errors[4]) if 4 < len(new_act_errors) else 0.0,
        }
        metrics.update(act_mut)
        metrics.update(agg_mut)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'strategy': self.name,
            'active_act_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'generation': state['generation'],
            'act_surprise': state['act_surprise'],
            'agg_surprise': state['agg_surprise'],
            'sin_affinity': float(state['act_affinity'][4]),
        }
