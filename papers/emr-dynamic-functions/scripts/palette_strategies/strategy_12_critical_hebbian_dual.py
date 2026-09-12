"""Strategy 12 Dual: Critical Period + Hebbian Learning for Both Activation AND Aggregation.

Extends CriticalHebbian to jointly evolve both activation and aggregation palettes
using the same bio-inspired mechanisms:
1. Hebbian learning for both domains
2. Critical periods for developmental phases
3. Cross-domain learning (activation-aggregation combinations)

Key innovation: Learn which activation-aggregation COMBINATIONS work together,
not just which functions work individually.

Biological analogy:
- Critical periods exist for multiple modalities (vision, hearing, language)
- Hebbian plasticity strengthens connections in all circuits
- Cross-modal learning links features that co-occur
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


class CriticalHebbianDualStrategy(PaletteEvolutionStrategy):
    """Critical periods with Hebbian-learned protection for BOTH palettes.

    Extends CriticalHebbian to jointly learn:
    1. Which activations are valuable (existing)
    2. Which aggregations are valuable (NEW)
    3. Which activation-aggregation combinations are synergistic (NEW)

    NO HARD-CODED FUNCTION LISTS - learns what to protect from experience.
    """

    name = "critical_hebbian_dual"
    description = "Dual palette evolution with cross-domain Hebbian learning"

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
        # Phase-specific rates for aggregation (can be different)
        agg_exploration_activate: float = 0.30,
        agg_exploration_deactivate: float = 0.02,
        agg_confirmation_deactivate_max: float = 0.12,
        agg_confirmation_deactivate_min: float = 0.01,
        # Hebbian learning parameters
        learning_rate: float = 0.20,
        anti_hebbian_rate: float = 0.05,
        affinity_protection_threshold: float = 0.55,
        # Cross-domain learning
        cross_learning_rate: float = 0.15,
        cross_influence: float = 0.3,  # How much cross-domain affects protection
        # Phase-specific Hebbian modulation
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Early consolidation
        early_consolidation_threshold: float = 0.95,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,   # Optimal is 6, >6 causes antagonism
        max_active_agg: int = 4,   # Optimal is 4 for parity
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize dual palette strategy."""
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
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

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
        """Initialize state with dual Hebbian matrices and cross-domain tracking."""
        # Activation palette
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        # Aggregation palette
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Hebbian weight matrices - activation domain
        act_hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Hebbian weight matrices - aggregation domain
        agg_hebbian_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Cross-domain affinity matrix (NEW)
        # Tracks which activation-aggregation combinations succeed together
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
            'rng_key': jax.random.PRNGKey(seed + 121213),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'fitness_history': [],
            'fitness_ema': 0.5,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _get_phase_lr(self, phase: str) -> float:
        """Get learning rate multiplier for phase."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            return self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            return self.confirmation_lr_multiplier
        else:
            return 0.1  # Very slow in consolidation

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update Hebbian weights for both domains and cross-domain.

        Returns:
            Tuple of (new_act_weights, new_act_affinity,
                     new_agg_weights, new_agg_affinity,
                     new_cross_affinity)
        """
        lr_mult = self._get_phase_lr(phase)
        lr = self.learning_rate * lr_mult
        anti_lr = self.anti_hebbian_rate * lr_mult
        cross_lr = self.cross_learning_rate * lr_mult

        # Active masks
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
        new_act_affinity = jnp.clip(act_affinity + act_affinity_delta, 0.0, 1.0)

        # --- Aggregation domain update ---
        agg_co_active = jnp.outer(agg_active, agg_active)
        if fitness_signal >= 0:
            agg_weight_delta = lr * fitness_signal * agg_co_active
            agg_affinity_delta = lr * fitness_signal * agg_active
        else:
            agg_weight_delta = anti_lr * fitness_signal * agg_co_active
            agg_affinity_delta = anti_lr * fitness_signal * agg_active

        new_agg_weights = jnp.clip(agg_weights + agg_weight_delta, 0.0, 1.0)
        new_agg_affinity = jnp.clip(agg_affinity + agg_affinity_delta, 0.0, 1.0)

        # --- Cross-domain update (NEW) ---
        # Learn which act-agg combinations correlate with success
        cross_active = jnp.outer(act_active, agg_active)
        if fitness_signal >= 0:
            cross_delta = cross_lr * fitness_signal * cross_active
        else:
            cross_delta = (anti_lr * 0.5) * fitness_signal * cross_active  # Less anti-learning for cross

        new_cross_affinity = jnp.clip(cross_affinity + cross_delta, 0.0, 1.0)

        return (new_act_weights, new_act_affinity,
                new_agg_weights, new_agg_affinity,
                new_cross_affinity)

    def _compute_protection_scores_act(
        self,
        act_affinity: jnp.ndarray,
        act_weights: jnp.ndarray,
        act_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection score for each activation (0 to 1).

        Includes cross-domain influence from aggregation.
        """
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_act_active = max(jnp.sum(act_active), 1)
        n_agg_active = max(jnp.sum(agg_active), 1)

        # Pairwise affinity with current active activations
        pairwise_score = jnp.dot(act_weights, act_active) / n_act_active

        # Cross-domain score: how well does this activation work with active aggregations
        cross_score = jnp.dot(cross_affinity, agg_active) / n_agg_active

        # Combine: individual (60%) + pairwise (25%) + cross-domain (15%)
        protection = (
            0.60 * act_affinity +
            0.25 * pairwise_score +
            0.15 * cross_score * self.cross_influence
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
        """Compute protection score for each aggregation (0 to 1).

        Includes cross-domain influence from activation.
        """
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_act_active = max(jnp.sum(act_active), 1)
        n_agg_active = max(jnp.sum(agg_active), 1)

        # Pairwise affinity with current active aggregations
        pairwise_score = jnp.dot(agg_weights, agg_active) / n_agg_active

        # Cross-domain score: how well does this aggregation work with active activations
        cross_score = jnp.dot(cross_affinity.T, act_active) / n_act_active

        # Combine: individual (60%) + pairwise (25%) + cross-domain (15%)
        protection = (
            0.60 * agg_affinity +
            0.25 * pairwise_score +
            0.15 * cross_score * self.cross_influence
        )

        return protection

    def _mutate_activation_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation to activation palette.

        Key constraint: max_active_act prevents antagonism (>6 activations hurts).
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        # Get phase-specific rates
        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = self.exploration_activate
            use_protection = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = self.confirmation_activate
            use_protection = True
        else:  # CONSOLIDATION
            activate_rate = self.consolidation_activate
            use_protection = True

        # Track current active count for max constraint
        current_active = int(jnp.sum(mask > 0.5))

        for i in range(NUM_ACTIVATIONS):
            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                # Inactive - might activate
                # CRITICAL: Skip if already at max
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
                # Active - might deactivate
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

        # Ensure minimum active
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
        """Apply phase-appropriate mutation to aggregation palette.

        Key constraint: max_active_agg prevents overload (too many aggs hurts).
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_AGGREGATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_AGGREGATIONS,))

        # Get phase-specific rates for aggregation
        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = self.agg_exploration_activate
            use_protection = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = self.confirmation_activate  # Same as activation
            use_protection = True
        else:  # CONSOLIDATION
            activate_rate = self.consolidation_activate
            use_protection = True

        # Track current active count for max constraint
        current_active = int(jnp.sum(mask > 0.5))

        for i in range(NUM_AGGREGATIONS):
            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                # Inactive - might activate
                # CRITICAL: Skip if already at max
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
                # Active - might deactivate
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

        # Ensure minimum active
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
        """Update with dual palette Hebbian learning."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Determine phase
        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']

        # Update fitness EMA
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness

        # Compute fitness signal
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Dual Hebbian update
        (new_act_weights, new_act_affinity,
         new_agg_weights, new_agg_affinity,
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

        # Apply mutations to both palettes
        new_act_mask, act_mutation = self._mutate_activation_palette(
            key_act, state['act_mask'], phase, act_protection
        )
        new_agg_mask, agg_mutation = self._mutate_aggregation_palette(
            key_agg, state['agg_mask'], phase, agg_protection
        )

        act_palette_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_palette_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Track fitness history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

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
        }

        # Compute stats
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
            # Aggregation stats
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'agg_max_affinity': float(jnp.max(new_agg_affinity)),
            # Cross-domain stats
            'cross_avg_affinity': float(jnp.mean(new_cross_affinity)),
            'cross_max_affinity': float(jnp.max(new_cross_affinity)),
        }
        metrics.update(act_mutation)
        metrics.update(agg_mutation)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual palette info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        act_affinity = state['act_affinity']
        agg_affinity = state['agg_affinity']

        # Find highest affinity functions
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
            # Learned affinities
            'top_act_affinity': top_act,
            'top_agg_affinity': top_agg,
            'sin_affinity': float(act_affinity[4]),
            'stagnation_count': state['stagnation_count'],
        }
