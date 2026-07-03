"""Strategy 14D: Decaying Hebbian Dual (Use-It-Or-Lose-It for Both Palettes).

Extends DecayingHebbian to jointly evolve activation AND aggregation function
palettes with activity-dependent decay for both domains.

Key mechanisms:
1. Hebbian learning for both domains
2. Activity-dependent decay: inactive functions lose affinity
3. Recent use tracking: recently used functions decay slower
4. Cross-domain learning: successful act-agg combinations persist

Biological basis:
- Synaptic proteins have finite half-lives
- Without activity, synapses weaken (structural plasticity)
- Applies to both computational choice (activation) and integration (aggregation)
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

# Aggregation constants
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


class DecayingHebbianDualStrategy(PaletteEvolutionStrategy):
    """Decaying Hebbian with use-it-or-lose-it for both palettes."""

    name = "decaying_hebbian_dual"
    description = "Decaying Hebbian learning for dual palette evolution"

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
        # Hebbian parameters
        learning_rate: float = 0.20,
        anti_hebbian_rate: float = 0.05,
        act_affinity_protection_threshold: float = 0.55,
        agg_affinity_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Decay parameters
        base_decay_rate: float = 0.02,
        decay_target: float = 0.5,
        activity_decay_reduction: float = 0.8,
        recent_use_window: int = 10,
        recent_use_decay_reduction: float = 0.5,
        exploration_decay_multiplier: float = 0.5,
        consolidation_decay_multiplier: float = 0.2,
        # Cross-domain
        cross_learning_rate: float = 0.08,
        cross_decay_rate: float = 0.01,
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
        self.learning_rate = learning_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.act_affinity_protection_threshold = act_affinity_protection_threshold
        self.agg_affinity_protection_threshold = agg_affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Decay
        self.base_decay_rate = base_decay_rate
        self.decay_target = decay_target
        self.activity_decay_reduction = activity_decay_reduction
        self.recent_use_window = recent_use_window
        self.recent_use_decay_reduction = recent_use_decay_reduction
        self.exploration_decay_multiplier = exploration_decay_multiplier
        self.consolidation_decay_multiplier = consolidation_decay_multiplier

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_decay_rate = cross_decay_rate

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
        act_recent_activity = {i: [] for i in range(NUM_ACTIVATIONS)}

        # Aggregation
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_mask(initial_agg)
        agg_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_recent_activity = {i: [] for i in range(NUM_AGGREGATIONS)}

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            'act_mask': act_mask,
            'act_weights': act_weights,
            'act_affinity': act_affinity,
            'act_recent_activity': act_recent_activity,
            'agg_mask': agg_mask,
            'agg_weights': agg_weights,
            'agg_affinity': agg_affinity,
            'agg_recent_activity': agg_recent_activity,
            'cross_affinity': cross_affinity,
            'rng_key': jax.random.PRNGKey(seed + 141414),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'fitness_ema': 0.5,
            'act_decay_applied': 0.0,
            'agg_decay_applied': 0.0,
            'mask': act_mask,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return agg_mask_to_indices(state['agg_mask'])

    def _apply_decay(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        generation: int,
        recent_activity: Dict[int, List[int]],
        phase: str,
        num_funcs: int,
    ) -> Tuple[jnp.ndarray, float]:
        """Apply activity-dependent decay."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            decay_multiplier = self.exploration_decay_multiplier
        elif phase == CriticalPeriodPhase.CONSOLIDATION:
            decay_multiplier = self.consolidation_decay_multiplier
        else:
            decay_multiplier = 1.0

        effective_decay_rate = self.base_decay_rate * decay_multiplier
        new_affinity = affinity.copy()
        total_decay = 0.0
        active = (mask > 0.5).astype(jnp.float32)

        for i in range(num_funcs):
            current_affinity = float(affinity[i])
            distance = current_affinity - self.decay_target

            if abs(distance) < 0.01:
                continue

            decay = effective_decay_rate * distance

            if active[i] > 0.5:
                decay *= (1.0 - self.activity_decay_reduction)

            recent_gens = recent_activity.get(i, [])
            recent_use_count = sum(1 for g in recent_gens if generation - g <= self.recent_use_window)
            if recent_use_count > 0:
                recent_factor = min(1.0, recent_use_count / self.recent_use_window)
                decay *= (1.0 - self.recent_use_decay_reduction * recent_factor)

            new_value = max(0.05, min(0.95, current_affinity - decay))
            new_affinity = new_affinity.at[i].set(new_value)
            total_decay += abs(decay)

        return new_affinity, total_decay

    def _update_recent_activity(
        self,
        recent_activity: Dict[int, List[int]],
        mask: jnp.ndarray,
        generation: int,
        num_funcs: int,
    ) -> Dict[int, List[int]]:
        """Update recent activity tracking."""
        new_activity = {}
        for i in range(num_funcs):
            history = recent_activity.get(i, [])
            history = [g for g in history if generation - g <= self.recent_use_window * 2]
            if mask[i] > 0.5:
                history.append(generation)
            new_activity[i] = history
        return new_activity

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update Hebbian weights."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr = self.learning_rate * self.exploration_lr_multiplier
            anti_lr = self.anti_hebbian_rate * self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr = self.learning_rate * self.confirmation_lr_multiplier
            anti_lr = self.anti_hebbian_rate * self.confirmation_lr_multiplier
        else:
            lr = self.learning_rate * 0.1
            anti_lr = self.anti_hebbian_rate * 0.1

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
        phase: str,
    ) -> jnp.ndarray:
        """Update cross-domain affinity."""
        if phase == CriticalPeriodPhase.CONSOLIDATION:
            lr = self.cross_learning_rate * 0.1
        else:
            lr = self.cross_learning_rate

        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)

        if fitness_signal >= 0:
            delta = lr * fitness_signal * cross_active
        else:
            delta = lr * 0.3 * fitness_signal * cross_active

        new_cross = jnp.clip(cross_affinity + delta, 0.0, 1.0)

        # Apply decay to inactive combinations
        inactive = 1.0 - cross_active
        new_cross = new_cross - self.cross_decay_rate * inactive * (new_cross - 0.5)

        return new_cross

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
        is_activation: bool,
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key)

        if is_activation:
            num_funcs = NUM_ACTIVATIONS
            max_active = self.max_active_act
            min_active = self.min_active_act
            protection_threshold = self.act_affinity_protection_threshold
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
            protection_threshold = self.agg_affinity_protection_threshold
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

        # Update recent activity
        new_act_recent = self._update_recent_activity(
            state['act_recent_activity'], state['act_mask'], generation, NUM_ACTIVATIONS
        )
        new_agg_recent = self._update_recent_activity(
            state['agg_recent_activity'], state['agg_mask'], generation, NUM_AGGREGATIONS
        )

        # Hebbian updates
        new_act_weights, new_act_affinity = self._hebbian_update(
            state['act_weights'], state['act_affinity'], state['act_mask'], fitness_signal, phase
        )
        new_agg_weights, new_agg_affinity = self._hebbian_update(
            state['agg_weights'], state['agg_affinity'], state['agg_mask'], fitness_signal, phase
        )

        # Cross-domain update
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], fitness_signal, phase
        )

        # Apply decay
        new_act_affinity, act_decay = self._apply_decay(
            new_act_affinity, state['act_mask'], generation, new_act_recent, phase, NUM_ACTIVATIONS
        )
        new_agg_affinity, agg_decay = self._apply_decay(
            new_agg_affinity, state['agg_mask'], generation, new_agg_recent, phase, NUM_AGGREGATIONS
        )

        # Protection scores
        act_protection = self._compute_protection_scores_act(
            new_act_affinity, new_act_weights, state['act_mask'], new_cross, state['agg_mask']
        )
        agg_protection = self._compute_protection_scores_agg(
            new_agg_affinity, new_agg_weights, state['agg_mask'], new_cross, state['act_mask']
        )

        # Mutations
        new_act_mask, act_mut = self._mutate_palette(
            subkey1, state['act_mask'], phase, act_protection, True
        )
        new_agg_mask, agg_mut = self._mutate_palette(
            subkey2, state['agg_mask'], phase, agg_protection, False
        )

        new_state = {
            'act_mask': new_act_mask,
            'act_weights': new_act_weights,
            'act_affinity': new_act_affinity,
            'act_recent_activity': new_act_recent,
            'agg_mask': new_agg_mask,
            'agg_weights': new_agg_weights,
            'agg_affinity': new_agg_affinity,
            'agg_recent_activity': new_agg_recent,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'fitness_ema': new_fitness_ema,
            'act_decay_applied': state['act_decay_applied'] + act_decay,
            'agg_decay_applied': state['agg_decay_applied'] + agg_decay,
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
            'act_decay': act_decay,
            'agg_decay': agg_decay,
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
            'total_act_decay': state['act_decay_applied'],
            'total_agg_decay': state['agg_decay_applied'],
        }
