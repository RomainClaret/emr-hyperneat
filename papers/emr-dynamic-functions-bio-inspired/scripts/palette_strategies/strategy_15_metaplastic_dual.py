"""Strategy 15D: Metaplastic Dual (Adaptive Learning Rates for Both Palettes).

Extends Metaplastic strategy to jointly evolve activation AND aggregation
function palettes with adaptive learning rates for both domains.

Key mechanisms:
1. Stagnation-boosted LR: If stuck, increase learning rate
2. Success-reduced LR: If improving, decrease learning rate
3. Sliding threshold: Protection threshold adapts to fitness distribution
4. Cross-domain learning with shared metaplastic state

Biological basis:
- Metaplasticity: plasticity is itself plastic
- BCM theory: sliding threshold for LTP/LTD boundary
- Applies to both computational (activation) and integrative (aggregation) learning
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


class CriticalPeriodPhase:
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


def create_initial_agg_mask(indices: List[int]) -> jnp.ndarray:
    mask = jnp.zeros(NUM_AGGREGATIONS)
    for idx in indices:
        if 0 <= idx < NUM_AGGREGATIONS:
            mask = mask.at[idx].set(1.0)
    return mask


def agg_mask_to_indices(mask: jnp.ndarray) -> List[int]:
    return [int(i) for i in range(NUM_AGGREGATIONS) if mask[i] > 0.5]


class MetaplasticDualStrategy(PaletteEvolutionStrategy):
    """Metaplastic dual strategy with adaptive learning rates for both palettes."""

    name = "metaplastic_dual"
    description = "Metaplastic adaptive learning for dual palette evolution"

    def __init__(
        self,
        # Critical period timing
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # Activation phase rates
        act_exploration_activate: float = 0.35,
        act_exploration_deactivate: float = 0.02,
        act_confirmation_activate: float = 0.10,
        act_confirmation_deactivate_max: float = 0.15,
        act_confirmation_deactivate_min: float = 0.01,
        act_consolidation_activate: float = 0.02,
        act_consolidation_deactivate: float = 0.01,
        # Aggregation phase rates
        agg_exploration_activate: float = 0.30,
        agg_exploration_deactivate: float = 0.02,
        agg_confirmation_activate: float = 0.08,
        agg_confirmation_deactivate_max: float = 0.12,
        agg_confirmation_deactivate_min: float = 0.01,
        agg_consolidation_activate: float = 0.02,
        agg_consolidation_deactivate: float = 0.01,
        # Base Hebbian parameters
        base_learning_rate: float = 0.20,
        base_anti_hebbian_rate: float = 0.05,
        act_base_protection_threshold: float = 0.55,
        agg_base_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Metaplastic parameters
        stagnation_lr_boost: float = 1.5,
        stagnation_threshold_gens: int = 10,
        max_stagnation_boost: float = 3.0,
        success_lr_reduction: float = 0.7,
        success_window: int = 5,
        success_threshold: float = 0.8,
        threshold_adaptation_rate: float = 0.05,
        threshold_min: float = 0.40,
        threshold_max: float = 0.70,
        threshold_percentile: float = 0.70,
        # Cross-domain
        cross_learning_rate: float = 0.08,
        # Constraints
        max_active_act: int = 6,
        max_active_agg: int = 4,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        # Other
        early_consolidation_threshold: float = 0.95,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

        # Activation rates
        self.act_exploration_activate = act_exploration_activate
        self.act_exploration_deactivate = act_exploration_deactivate
        self.act_confirmation_activate = act_confirmation_activate
        self.act_confirmation_deactivate_max = act_confirmation_deactivate_max
        self.act_confirmation_deactivate_min = act_confirmation_deactivate_min
        self.act_consolidation_activate = act_consolidation_activate
        self.act_consolidation_deactivate = act_consolidation_deactivate

        # Aggregation rates
        self.agg_exploration_activate = agg_exploration_activate
        self.agg_exploration_deactivate = agg_exploration_deactivate
        self.agg_confirmation_activate = agg_confirmation_activate
        self.agg_confirmation_deactivate_max = agg_confirmation_deactivate_max
        self.agg_confirmation_deactivate_min = agg_confirmation_deactivate_min
        self.agg_consolidation_activate = agg_consolidation_activate
        self.agg_consolidation_deactivate = agg_consolidation_deactivate

        # Hebbian
        self.base_learning_rate = base_learning_rate
        self.base_anti_hebbian_rate = base_anti_hebbian_rate
        self.act_base_protection_threshold = act_base_protection_threshold
        self.agg_base_protection_threshold = agg_base_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Metaplastic
        self.stagnation_lr_boost = stagnation_lr_boost
        self.stagnation_threshold_gens = stagnation_threshold_gens
        self.max_stagnation_boost = max_stagnation_boost
        self.success_lr_reduction = success_lr_reduction
        self.success_window = success_window
        self.success_threshold = success_threshold
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_percentile = threshold_percentile

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        # Constraints
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        self.early_consolidation_threshold = early_consolidation_threshold
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION
        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        # Activation
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Aggregation
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_mask(initial_agg)
        agg_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            'act_mask': act_mask,
            'act_weights': act_weights,
            'act_affinity': act_affinity,
            'agg_mask': agg_mask,
            'agg_weights': agg_weights,
            'agg_affinity': agg_affinity,
            'cross_affinity': cross_affinity,
            'rng_key': jax.random.PRNGKey(seed + 151515),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'fitness_ema': 0.5,
            # Metaplastic state (shared between domains)
            'lr_multiplier': 1.0,
            'act_protection_threshold': self.act_base_protection_threshold,
            'agg_protection_threshold': self.agg_base_protection_threshold,
            'improvement_history': [],
            'lr_history': [],
            'mask': act_mask,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return agg_mask_to_indices(state['agg_mask'])

    def _compute_lr_multiplier(
        self,
        stagnation_count: int,
        improvement_history: List[bool],
        phase: str,
    ) -> float:
        """Compute metaplastic learning rate multiplier."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            phase_mult = self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            phase_mult = self.confirmation_lr_multiplier
        else:
            phase_mult = 0.1

        # Stagnation boost
        if stagnation_count >= self.stagnation_threshold_gens:
            stagnation_factor = stagnation_count / self.stagnation_threshold_gens
            stagnation_boost = min(
                self.max_stagnation_boost,
                1.0 + (self.stagnation_lr_boost - 1.0) * stagnation_factor
            )
        else:
            stagnation_boost = 1.0

        # Success reduction
        if len(improvement_history) >= self.success_window:
            recent = improvement_history[-self.success_window:]
            improvement_rate = sum(recent) / len(recent)
            if improvement_rate >= self.success_threshold:
                success_mult = self.success_lr_reduction
            else:
                success_mult = 1.0
        else:
            success_mult = 1.0

        return phase_mult * stagnation_boost * success_mult

    def _update_protection_threshold(
        self,
        current_threshold: float,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        num_funcs: int,
    ) -> float:
        """Update protection threshold using sliding percentile."""
        valid_affinities = [float(affinity[i]) for i in range(num_funcs) if mask[i] > 0.5]

        if len(valid_affinities) < 2:
            return current_threshold

        target = float(np.percentile(valid_affinities, self.threshold_percentile * 100))
        target = max(self.threshold_min, min(self.threshold_max, target))

        return (1 - self.threshold_adaptation_rate) * current_threshold + self.threshold_adaptation_rate * target

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        lr_multiplier: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update Hebbian weights with metaplastic LR."""
        lr = self.base_learning_rate * lr_multiplier
        anti_lr = self.base_anti_hebbian_rate * lr_multiplier

        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        if fitness_signal >= 0:
            weight_delta = lr * fitness_signal * co_active
            affinity_delta = lr * fitness_signal * active
        else:
            weight_delta = anti_lr * fitness_signal * co_active
            affinity_delta = anti_lr * fitness_signal * active

        new_weights = jnp.clip(weights + weight_delta, 0.0, 1.0)
        new_affinity = jnp.clip(affinity + affinity_delta, 0.0, 1.0)

        return new_weights, new_affinity

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_signal: float,
        lr_multiplier: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity with metaplastic LR."""
        lr = self.cross_learning_rate * lr_multiplier

        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)

        if fitness_signal >= 0:
            delta = lr * fitness_signal * cross_active
        else:
            delta = lr * 0.3 * fitness_signal * cross_active

        return jnp.clip(cross_affinity + delta, 0.0, 1.0)

    def _compute_protection_scores_act(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        pairwise_score = jnp.dot(weights, active) / n_active

        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_agg = max(jnp.sum(agg_active), 1)
        cross_score = jnp.dot(cross_affinity, agg_active) / n_agg

        return 0.60 * affinity + 0.25 * pairwise_score + 0.15 * cross_score

    def _compute_protection_scores_agg(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        pairwise_score = jnp.dot(weights, active) / n_active

        act_active = (act_mask > 0.5).astype(jnp.float32)
        n_act = max(jnp.sum(act_active), 1)
        cross_score = jnp.dot(cross_affinity.T, act_active) / n_act

        return 0.60 * affinity + 0.25 * pairwise_score + 0.15 * cross_score

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
        protection_threshold: float,
        is_activation: bool,
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key)

        if is_activation:
            num_funcs = NUM_ACTIVATIONS
            max_active = self.max_active_act
            min_active = self.min_active_act
            rates = {
                CriticalPeriodPhase.EXPLORATION: (self.act_exploration_activate, self.act_exploration_deactivate),
                CriticalPeriodPhase.CONFIRMATION: (self.act_confirmation_activate, None),
                CriticalPeriodPhase.CONSOLIDATION: (self.act_consolidation_activate, self.act_consolidation_deactivate),
            }
            deact_range = (self.act_confirmation_deactivate_min, self.act_confirmation_deactivate_max)
        else:
            num_funcs = NUM_AGGREGATIONS
            max_active = self.max_active_agg
            min_active = self.min_active_agg
            rates = {
                CriticalPeriodPhase.EXPLORATION: (self.agg_exploration_activate, self.agg_exploration_deactivate),
                CriticalPeriodPhase.CONFIRMATION: (self.agg_confirmation_activate, None),
                CriticalPeriodPhase.CONSOLIDATION: (self.agg_consolidation_activate, self.agg_consolidation_deactivate),
            }
            deact_range = (self.agg_confirmation_deactivate_min, self.agg_confirmation_deactivate_max)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (num_funcs,))
        deactivate_probs = jax.random.uniform(key2, (num_funcs,))

        activate_rate, deactivate_rate = rates[phase]

        for i in range(num_funcs):
            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                current_active = int(jnp.sum(mask > 0.5))
                if current_active >= max_active:
                    continue

                if phase == CriticalPeriodPhase.CONFIRMATION:
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= protection_threshold:
                        continue
                    deact_rate = deactivate_rate
                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= protection_threshold:
                        deact_rate = deact_range[0]
                    else:
                        t = protection / protection_threshold
                        deact_rate = deact_range[1] * (1 - t) + deact_range[0] * t
                else:
                    deact_rate = deactivate_rate

                if deactivate_probs[i] < deact_rate:
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

        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']

        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Update improvement history
        improvement_history = state['improvement_history'] + [improved]
        if len(improvement_history) > self.success_window * 2:
            improvement_history = improvement_history[-self.success_window * 2:]

        # Compute shared metaplastic LR
        lr_multiplier = self._compute_lr_multiplier(new_stagnation, improvement_history, phase)

        # Update protection thresholds
        new_act_threshold = self._update_protection_threshold(
            state['act_protection_threshold'], state['act_affinity'], state['act_mask'], NUM_ACTIVATIONS
        )
        new_agg_threshold = self._update_protection_threshold(
            state['agg_protection_threshold'], state['agg_affinity'], state['agg_mask'], NUM_AGGREGATIONS
        )

        # Hebbian updates with shared LR
        new_act_weights, new_act_affinity = self._hebbian_update(
            state['act_weights'], state['act_affinity'], state['act_mask'], fitness_signal, lr_multiplier
        )
        new_agg_weights, new_agg_affinity = self._hebbian_update(
            state['agg_weights'], state['agg_affinity'], state['agg_mask'], fitness_signal, lr_multiplier
        )

        # Cross-domain update
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], fitness_signal, lr_multiplier
        )

        # Protection scores
        act_protection = self._compute_protection_scores_act(
            new_act_affinity, new_act_weights, state['act_mask'], new_cross, state['agg_mask']
        )
        agg_protection = self._compute_protection_scores_agg(
            new_agg_affinity, new_agg_weights, state['agg_mask'], new_cross, state['act_mask']
        )

        # Mutations with adaptive thresholds
        new_act_mask, act_mut = self._mutate_palette(
            subkey1, state['act_mask'], phase, act_protection, new_act_threshold, True
        )
        new_agg_mask, agg_mut = self._mutate_palette(
            subkey2, state['agg_mask'], phase, agg_protection, new_agg_threshold, False
        )

        # Track LR history
        lr_history = state['lr_history'] + [lr_multiplier]
        if len(lr_history) > 50:
            lr_history = lr_history[-50:]

        new_state = {
            'act_mask': new_act_mask,
            'act_weights': new_act_weights,
            'act_affinity': new_act_affinity,
            'agg_mask': new_agg_mask,
            'agg_weights': new_agg_weights,
            'agg_affinity': new_agg_affinity,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'fitness_ema': new_fitness_ema,
            'lr_multiplier': lr_multiplier,
            'act_protection_threshold': new_act_threshold,
            'agg_protection_threshold': new_agg_threshold,
            'improvement_history': improvement_history,
            'lr_history': lr_history,
            'mask': new_act_mask,
        }

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': agg_mask_to_indices(new_agg_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'sin_affinity': float(new_act_affinity[4]),
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'lr_multiplier': lr_multiplier,
            'act_protection_threshold': new_act_threshold,
            'agg_protection_threshold': new_agg_threshold,
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
            'phase': state['phase'],
            'generation': state['generation'],
            'sin_affinity': float(state['act_affinity'][4]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
            'lr_multiplier': state['lr_multiplier'],
            'act_protection_threshold': state['act_protection_threshold'],
            'agg_protection_threshold': state['agg_protection_threshold'],
        }
