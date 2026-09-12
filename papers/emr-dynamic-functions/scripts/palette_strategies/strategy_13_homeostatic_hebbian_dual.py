"""Strategy 13 Dual: Homeostatic Hebbian for Both Activation AND Aggregation.

Extends HomeostaticHebbian to jointly evolve both activation and aggregation palettes
with homeostatic regulation to prevent affinity saturation in both domains.

Key mechanisms extended to dual:
1. Synaptic scaling for both activation and aggregation affinities
2. Soft bounds prevent saturation in both domains
3. Cross-domain affinity matrix tracks act-agg combinations
4. Max_active constraints prevent antagonism (max_act=6, max_agg=4)

Biological rationale:
- Homeostatic plasticity operates across ALL synapses, not just one type
- The brain maintains activity within functional ranges for all circuits
- Different modalities (like visual and motor) have separate but coordinated scaling

Expected improvement:
- Fewer false protections in BOTH domains
- Better adaptation when conditions change
- More robust to early luck in either domain
- Cross-domain homeostasis prevents one domain from dominating
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


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class HomeostaticHebbianDualStrategy(PaletteEvolutionStrategy):
    """Homeostatic Hebbian with dual palette evolution.

    Extends homeostatic regulation to both activation AND aggregation palettes,
    with cross-domain learning to find optimal combinations.
    """

    name = "homeostatic_hebbian_dual"
    description = "Dual palette evolution with homeostatic regulation for both domains"

    def __init__(
        self,
        # Critical period timing
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # Phase-specific base rates (activation)
        exploration_activate: float = 0.35,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate_max: float = 0.15,
        confirmation_deactivate_min: float = 0.01,
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # Phase-specific rates for aggregation
        agg_exploration_activate: float = 0.30,
        agg_exploration_deactivate: float = 0.02,
        agg_confirmation_deactivate_max: float = 0.12,
        agg_confirmation_deactivate_min: float = 0.01,
        # Hebbian parameters
        learning_rate: float = 0.20,
        anti_hebbian_rate: float = 0.05,
        affinity_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Cross-domain learning
        cross_learning_rate: float = 0.15,
        cross_influence: float = 0.3,
        # HOMEOSTATIC PARAMETERS
        target_mean_affinity: float = 0.5,
        scaling_rate: float = 0.08,
        scaling_threshold: float = 0.15,
        affinity_ceiling: float = 0.95,
        affinity_floor: float = 0.05,
        soft_ceiling_start: float = 0.80,
        soft_floor_start: float = 0.20,
        # Constraints
        early_consolidation_threshold: float = 0.95,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize dual homeostatic Hebbian strategy."""
        # Critical period timing
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

        # Activation rates
        self.exploration_activate = exploration_activate
        self.exploration_deactivate = exploration_deactivate
        self.confirmation_activate = confirmation_activate
        self.confirmation_deactivate_max = confirmation_deactivate_max
        self.confirmation_deactivate_min = confirmation_deactivate_min
        self.consolidation_activate = consolidation_activate
        self.consolidation_deactivate = consolidation_deactivate

        # Aggregation rates
        self.agg_exploration_activate = agg_exploration_activate
        self.agg_exploration_deactivate = agg_exploration_deactivate
        self.agg_confirmation_deactivate_max = agg_confirmation_deactivate_max
        self.agg_confirmation_deactivate_min = agg_confirmation_deactivate_min

        # Hebbian parameters
        self.learning_rate = learning_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.affinity_protection_threshold = affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence

        # Homeostatic parameters
        self.target_mean_affinity = target_mean_affinity
        self.scaling_rate = scaling_rate
        self.scaling_threshold = scaling_threshold
        self.affinity_ceiling = affinity_ceiling
        self.affinity_floor = affinity_floor
        self.soft_ceiling_start = soft_ceiling_start
        self.soft_floor_start = soft_floor_start

        # Other
        self.early_consolidation_threshold = early_consolidation_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase with early consolidation."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual homeostatic tracking."""
        # Activation palette
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        # Aggregation palette
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Activation domain state
        act_hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Aggregation domain state
        agg_hebbian_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation state
            'act_mask': act_mask,
            'act_hebbian_weights': act_hebbian_weights,
            'act_affinity': act_affinity,
            # Aggregation state
            'agg_mask': agg_mask,
            'agg_hebbian_weights': agg_hebbian_weights,
            'agg_affinity': agg_affinity,
            # Cross-domain state
            'cross_affinity': cross_affinity,
            # Common state
            'rng_key': jax.random.PRNGKey(seed + 131314),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'fitness_history': [],
            'fitness_ema': 0.5,
            # Homeostatic tracking
            'act_scaling_count': 0,
            'agg_scaling_count': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _apply_soft_bounds(
        self, affinity: jnp.ndarray, delta: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply soft bounds to prevent saturation."""
        positive_mask = delta > 0
        dist_to_ceiling = self.affinity_ceiling - affinity
        ceiling_factor = jnp.where(
            affinity > self.soft_ceiling_start,
            dist_to_ceiling / (self.affinity_ceiling - self.soft_ceiling_start),
            jnp.ones_like(affinity)
        )
        ceiling_factor = jnp.clip(ceiling_factor, 0.0, 1.0)

        dist_to_floor = affinity - self.affinity_floor
        floor_factor = jnp.where(
            affinity < self.soft_floor_start,
            dist_to_floor / (self.soft_floor_start - self.affinity_floor),
            jnp.ones_like(affinity)
        )
        floor_factor = jnp.clip(floor_factor, 0.0, 1.0)

        scaled_delta = jnp.where(positive_mask, delta * ceiling_factor, delta * floor_factor)
        new_affinity = affinity + scaled_delta
        return jnp.clip(new_affinity, self.affinity_floor, self.affinity_ceiling)

    def _apply_synaptic_scaling(
        self, affinity: jnp.ndarray, mask: jnp.ndarray
    ) -> Tuple[jnp.ndarray, bool]:
        """Apply synaptic scaling to maintain homeostasis."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = jnp.sum(active)

        active_affinity_sum = jnp.sum(affinity * active)
        mean_active_affinity = jnp.where(
            n_active > 0,
            active_affinity_sum / n_active,
            self.target_mean_affinity
        )

        drift = mean_active_affinity - self.target_mean_affinity
        scaling_needed = jnp.abs(drift) > self.scaling_threshold

        if scaling_needed:
            if mean_active_affinity > self.target_mean_affinity:
                scale = 1.0 - self.scaling_rate * (drift / mean_active_affinity)
            else:
                scale = 1.0 - self.scaling_rate * (drift / (1.0 - mean_active_affinity + 0.01))

            scale = jnp.clip(scale, 0.9, 1.1)
            scaled_affinity = jnp.where(
                active > 0.5,
                self.target_mean_affinity + (affinity - self.target_mean_affinity) * scale,
                affinity
            )
            return jnp.clip(scaled_affinity, self.affinity_floor, self.affinity_ceiling), True

        return affinity, False

    def _get_phase_lr(self, phase: str) -> float:
        """Get learning rate multiplier for phase."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            return self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            return self.confirmation_lr_multiplier
        else:
            return 0.1

    def _hebbian_update_dual(
        self,
        act_weights: jnp.ndarray,
        act_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_weights: jnp.ndarray,
        agg_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, bool, jnp.ndarray, jnp.ndarray, bool, jnp.ndarray]:
        """Update Hebbian weights with homeostatic regulation for both domains.

        Returns: (new_act_weights, new_act_affinity, act_scaled,
                  new_agg_weights, new_agg_affinity, agg_scaled,
                  new_cross_affinity)
        """
        lr_mult = self._get_phase_lr(phase)
        lr = self.learning_rate * lr_mult
        anti_lr = self.anti_hebbian_rate * lr_mult
        cross_lr = self.cross_learning_rate * lr_mult

        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)

        # --- Activation domain update ---
        act_co_active = jnp.outer(act_active, act_active)
        if fitness_signal >= 0:
            act_weight_delta = lr * fitness_signal * act_co_active
            act_affinity_delta = lr * fitness_signal * act_active
        else:
            act_weight_delta = anti_lr * fitness_signal * act_co_active
            act_affinity_delta = anti_lr * fitness_signal * act_active

        new_act_weights = jnp.clip(act_weights + act_weight_delta, 0.0, 1.0)
        new_act_affinity = self._apply_soft_bounds(act_affinity, act_affinity_delta)
        new_act_affinity, act_scaled = self._apply_synaptic_scaling(new_act_affinity, act_mask)

        # --- Aggregation domain update ---
        agg_co_active = jnp.outer(agg_active, agg_active)
        if fitness_signal >= 0:
            agg_weight_delta = lr * fitness_signal * agg_co_active
            agg_affinity_delta = lr * fitness_signal * agg_active
        else:
            agg_weight_delta = anti_lr * fitness_signal * agg_co_active
            agg_affinity_delta = anti_lr * fitness_signal * agg_active

        new_agg_weights = jnp.clip(agg_weights + agg_weight_delta, 0.0, 1.0)
        new_agg_affinity = self._apply_soft_bounds(agg_affinity, agg_affinity_delta)
        new_agg_affinity, agg_scaled = self._apply_synaptic_scaling(new_agg_affinity, agg_mask)

        # --- Cross-domain update ---
        cross_active = jnp.outer(act_active, agg_active)
        if fitness_signal >= 0:
            cross_delta = cross_lr * fitness_signal * cross_active
        else:
            cross_delta = (anti_lr * 0.5) * fitness_signal * cross_active
        new_cross_affinity = jnp.clip(cross_affinity + cross_delta, 0.0, 1.0)

        return (new_act_weights, new_act_affinity, act_scaled,
                new_agg_weights, new_agg_affinity, agg_scaled,
                new_cross_affinity)

    def _compute_protection_scores_act(
        self,
        act_affinity: jnp.ndarray,
        act_weights: jnp.ndarray,
        act_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection for activations with cross-domain influence."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_act_active = max(jnp.sum(act_active), 1)
        n_agg_active = max(jnp.sum(agg_active), 1)

        pairwise_score = jnp.dot(act_weights, act_active) / n_act_active
        cross_score = jnp.dot(cross_affinity, agg_active) / n_agg_active

        protection = (
            0.55 * act_affinity +
            0.25 * pairwise_score +
            0.20 * cross_score * self.cross_influence
        )
        return protection

    def _compute_protection_scores_agg(
        self,
        agg_affinity: jnp.ndarray,
        agg_weights: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection for aggregations with cross-domain influence."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_act_active = max(jnp.sum(act_active), 1)
        n_agg_active = max(jnp.sum(agg_active), 1)

        pairwise_score = jnp.dot(agg_weights, agg_active) / n_agg_active
        cross_score = jnp.dot(cross_affinity.T, act_active) / n_act_active

        protection = (
            0.55 * agg_affinity +
            0.25 * pairwise_score +
            0.20 * cross_score * self.cross_influence
        )
        return protection

    def _mutate_activation_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation to activation palette."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = self.exploration_activate
            use_protection = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = self.confirmation_activate
            use_protection = True
        else:
            activate_rate = self.consolidation_activate
            use_protection = True

        current_active = int(jnp.sum(mask > 0.5))

        for i in range(NUM_ACTIVATIONS):
            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                if current_active + len(activated) >= self.max_active_act:
                    continue

                if use_protection and phase == CriticalPeriodPhase.CONFIRMATION:
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= self.affinity_protection_threshold:
                        continue
                    deact_rate = self.consolidation_deactivate
                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= self.affinity_protection_threshold:
                        deact_rate = self.confirmation_deactivate_min
                    else:
                        t = protection / self.affinity_protection_threshold
                        deact_rate = (
                            self.confirmation_deactivate_max * (1 - t) +
                            self.confirmation_deactivate_min * t
                        )
                else:
                    deact_rate = self.exploration_deactivate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < self.min_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'act_activated': activated, 'act_deactivated': deactivated}

    def _mutate_aggregation_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation to aggregation palette."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_AGGREGATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_AGGREGATIONS,))

        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = self.agg_exploration_activate
            use_protection = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = self.confirmation_activate
            use_protection = True
        else:
            activate_rate = self.consolidation_activate
            use_protection = True

        current_active = int(jnp.sum(mask > 0.5))

        for i in range(NUM_AGGREGATIONS):
            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                if current_active + len(activated) >= self.max_active_agg:
                    continue

                if use_protection and phase == CriticalPeriodPhase.CONFIRMATION:
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= self.affinity_protection_threshold:
                        continue
                    deact_rate = self.consolidation_deactivate
                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= self.affinity_protection_threshold:
                        deact_rate = self.agg_confirmation_deactivate_min
                    else:
                        t = protection / self.affinity_protection_threshold
                        deact_rate = (
                            self.agg_confirmation_deactivate_max * (1 - t) +
                            self.agg_confirmation_deactivate_min * t
                        )
                else:
                    deact_rate = self.agg_exploration_deactivate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < self.min_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'agg_activated': activated, 'agg_deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual homeostatic Hebbian learning."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']

        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness

        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Dual Hebbian update with homeostasis
        (new_act_weights, new_act_affinity, act_scaled,
         new_agg_weights, new_agg_affinity, agg_scaled,
         new_cross_affinity) = self._hebbian_update_dual(
            state['act_hebbian_weights'],
            state['act_affinity'],
            state['act_mask'],
            state['agg_hebbian_weights'],
            state['agg_affinity'],
            state['agg_mask'],
            state['cross_affinity'],
            fitness_signal,
            phase,
        )

        # Compute protection scores
        act_protection = self._compute_protection_scores_act(
            new_act_affinity, new_act_weights, state['act_mask'],
            new_cross_affinity, state['agg_mask']
        )
        agg_protection = self._compute_protection_scores_agg(
            new_agg_affinity, new_agg_weights, state['agg_mask'],
            new_cross_affinity, state['act_mask']
        )

        # Apply mutations
        new_act_mask, act_mutation = self._mutate_activation_palette(
            key_act, state['act_mask'], phase, act_protection
        )
        new_agg_mask, agg_mutation = self._mutate_aggregation_palette(
            key_agg, state['agg_mask'], phase, agg_protection
        )

        act_palette_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_palette_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        act_scaling_count = state['act_scaling_count'] + (1 if act_scaled else 0)
        agg_scaling_count = state['agg_scaling_count'] + (1 if agg_scaled else 0)

        new_state = {
            'act_mask': new_act_mask,
            'act_hebbian_weights': new_act_weights,
            'act_affinity': new_act_affinity,
            'agg_mask': new_agg_mask,
            'agg_hebbian_weights': new_agg_weights,
            'agg_affinity': new_agg_affinity,
            'cross_affinity': new_cross_affinity,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'act_scaling_count': act_scaling_count,
            'agg_scaling_count': agg_scaling_count,
        }

        active_act_palette = mask_to_indices(new_act_mask)
        active_agg_palette = mask_to_indices(new_agg_mask)
        sin_affinity = float(new_act_affinity[4]) if 4 < len(new_act_affinity) else 0.0

        metrics = {
            'act_palette_changed': act_palette_changed,
            'agg_palette_changed': agg_palette_changed,
            'current_act_palette': active_act_palette,
            'current_agg_palette': active_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            'fitness_signal': fitness_signal,
            # Activation stats
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'act_max_affinity': float(jnp.max(new_act_affinity)),
            'sin_affinity': sin_affinity,
            'act_scaled': act_scaled,
            'act_scaling_count': act_scaling_count,
            # Aggregation stats
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'agg_max_affinity': float(jnp.max(new_agg_affinity)),
            'agg_scaled': agg_scaled,
            'agg_scaling_count': agg_scaling_count,
            # Cross-domain stats
            'cross_avg_affinity': float(jnp.mean(new_cross_affinity)),
            'cross_max_affinity': float(jnp.max(new_cross_affinity)),
        }
        metrics.update(act_mutation)
        metrics.update(agg_mutation)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual homeostatic stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        act_affinity = state['act_affinity']
        agg_affinity = state['agg_affinity']

        top_act_indices = jnp.argsort(act_affinity)[-3:][::-1]
        top_act = [(int(i), float(act_affinity[i])) for i in top_act_indices]

        top_agg_indices = jnp.argsort(agg_affinity)[-3:][::-1]
        top_agg = [(int(i), float(agg_affinity[i])) for i in top_agg_indices]

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'act_palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'phase': state['phase'],
            'generation': state['generation'],
            'top_act_affinity': top_act,
            'top_agg_affinity': top_agg,
            'sin_affinity': float(act_affinity[4]),
            'stagnation_count': state['stagnation_count'],
            'act_scaling_count': state['act_scaling_count'],
            'agg_scaling_count': state['agg_scaling_count'],
        }
