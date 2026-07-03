"""Strategy 16D: STDP Dual (Spike-Timing-Dependent Plasticity for Both Palettes).

Extends STDP to jointly evolve activation AND aggregation function palettes
with temporal credit assignment for both domains.

Key mechanisms:
1. STDP for activations: Track which activations PRECEDE fitness improvement
2. STDP for aggregations: Same temporal credit for aggregation functions
3. Cross-domain STDP: Learn which act-agg combinations precede success
4. Separate temporal histories for each domain

Biological basis:
- Temporal causality matters for both what computation (activation) and
  how inputs combine (aggregation)
- Cross-domain interactions can reveal synergistic combinations
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
DEFAULT_AGGREGATION_INDICES = [0, 1]  # Start with sum, mean


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


def create_initial_agg_mask(indices: List[int]) -> jnp.ndarray:
    """Create initial aggregation palette mask."""
    mask = jnp.zeros(NUM_AGGREGATIONS)
    for idx in indices:
        if 0 <= idx < NUM_AGGREGATIONS:
            mask = mask.at[idx].set(1.0)
    return mask


def agg_mask_to_indices(mask: jnp.ndarray) -> List[int]:
    """Convert aggregation mask to list of indices."""
    return [int(i) for i in range(NUM_AGGREGATIONS) if mask[i] > 0.5]


class STDPDualStrategy(PaletteEvolutionStrategy):
    """STDP-based dual palette evolution with temporal credit assignment.

    Learns which activation AND aggregation functions PRECEDE fitness improvements.
    """

    name = "stdp_dual"
    description = "Spike-timing-dependent plasticity for dual palette evolution"

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
        # Aggregation phase rates (slightly different)
        agg_exploration_activate: float = 0.30,
        agg_exploration_deactivate: float = 0.02,
        agg_confirmation_activate: float = 0.08,
        agg_confirmation_deactivate_max: float = 0.12,
        agg_confirmation_deactivate_min: float = 0.01,
        agg_consolidation_activate: float = 0.02,
        agg_consolidation_deactivate: float = 0.01,
        # STDP window parameters
        ltp_window: int = 5,
        ltd_window: int = 3,
        history_length: int = 10,
        # STDP learning rates
        ltp_rate: float = 0.25,
        ltd_rate: float = 0.10,
        # Cross-domain STDP
        cross_ltp_rate: float = 0.15,
        cross_ltd_rate: float = 0.05,
        # Temporal weighting
        temporal_decay: float = 0.7,
        # Protection
        act_affinity_protection_threshold: float = 0.55,
        agg_affinity_protection_threshold: float = 0.55,
        # Phase modulation
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
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
        """Initialize STDP dual strategy."""
        # Critical period timing
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

        # STDP parameters
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.history_length = history_length
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.cross_ltp_rate = cross_ltp_rate
        self.cross_ltd_rate = cross_ltd_rate
        self.temporal_decay = temporal_decay

        # Protection
        self.act_affinity_protection_threshold = act_affinity_protection_threshold
        self.agg_affinity_protection_threshold = agg_affinity_protection_threshold

        # Phase modulation
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Constraints
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        # Other
        self.early_consolidation_threshold = early_consolidation_threshold
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

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
        """Initialize state with dual STDP tracking."""
        # Activation palette
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        act_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5

        # Aggregation palette
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_mask(initial_agg)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation state
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_weights': act_weights,
            'act_history': [],  # (gen, mask, fitness) tuples

            # Aggregation state
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_weights': agg_weights,
            'agg_history': [],

            # Cross-domain
            'cross_affinity': cross_affinity,

            # General state
            'rng_key': jax.random.PRNGKey(seed + 161616),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'fitness_ema': 0.5,

            # STDP tracking
            'ltp_events': 0,
            'ltd_events': 0,

            # Legacy compatibility
            'mask': act_mask,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return agg_mask_to_indices(state['agg_mask'])

    def _compute_temporal_weight(self, generations_from_improvement: int) -> float:
        """Compute temporal weight - closer = stronger effect."""
        return self.temporal_decay ** abs(generations_from_improvement)

    def _stdp_update_domain(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        history: List[Tuple[int, jnp.ndarray, float]],
        current_gen: int,
        fitness_improved: bool,
        improvement_magnitude: float,
        lr_mult: float,
        is_activation: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Apply STDP update for one domain."""
        num_funcs = NUM_ACTIVATIONS if is_activation else NUM_AGGREGATIONS

        new_affinity = affinity.copy()
        new_weights = weights.copy()

        ltp_functions = []
        ltd_functions = []

        if fitness_improved and len(history) >= 2:
            for hist_gen, hist_mask, hist_fitness in history:
                gens_before = current_gen - hist_gen

                if 1 <= gens_before <= self.ltp_window:
                    temporal_weight = self._compute_temporal_weight(gens_before)
                    ltp_delta = (
                        self.ltp_rate * lr_mult *
                        temporal_weight *
                        improvement_magnitude *
                        (hist_mask > 0.5).astype(jnp.float32)
                    )
                    new_affinity = jnp.clip(new_affinity + ltp_delta, 0.0, 1.0)

                    ltp_indices = [i for i in range(num_funcs) if hist_mask[i] > 0.5]
                    ltp_functions.extend(ltp_indices)

                    active = (hist_mask > 0.5).astype(jnp.float32)
                    co_active = jnp.outer(active, active)
                    weight_delta = self.ltp_rate * lr_mult * temporal_weight * improvement_magnitude * co_active
                    new_weights = jnp.clip(new_weights + weight_delta, 0.0, 1.0)

        # LTD for stagnation
        if not fitness_improved and len(history) >= self.ltd_window:
            recent_mask = history[-1][1] if history else jnp.zeros(num_funcs)
            active = (recent_mask > 0.5).astype(jnp.float32)

            ltd_delta = self.ltd_rate * lr_mult * 0.5 * active
            new_affinity = jnp.clip(new_affinity - ltd_delta, 0.0, 1.0)

            ltd_indices = [i for i in range(num_funcs) if recent_mask[i] > 0.5]
            ltd_functions.extend(ltd_indices)

        return new_affinity, new_weights, {
            'ltp_functions': list(set(ltp_functions)),
            'ltd_functions': list(set(ltd_functions)),
        }

    def _stdp_update_cross_domain(
        self,
        cross_affinity: jnp.ndarray,
        act_history: List,
        agg_history: List,
        current_gen: int,
        fitness_improved: bool,
        improvement_magnitude: float,
        lr_mult: float,
    ) -> jnp.ndarray:
        """Apply STDP to cross-domain affinities."""
        new_cross = cross_affinity.copy()

        if fitness_improved and len(act_history) >= 2 and len(agg_history) >= 2:
            for act_gen, act_mask, _ in act_history:
                for agg_gen, agg_mask, _ in agg_history:
                    gens_before = current_gen - max(act_gen, agg_gen)

                    if 1 <= gens_before <= self.ltp_window:
                        temporal_weight = self._compute_temporal_weight(gens_before)

                        act_active = (act_mask > 0.5).astype(jnp.float32)
                        agg_active = (agg_mask > 0.5).astype(jnp.float32)
                        cross_active = jnp.outer(act_active, agg_active)

                        delta = (
                            self.cross_ltp_rate * lr_mult *
                            temporal_weight * improvement_magnitude * cross_active
                        )
                        new_cross = jnp.clip(new_cross + delta, 0.0, 1.0)

        # Cross-domain LTD
        if not fitness_improved and len(act_history) >= 1 and len(agg_history) >= 1:
            act_mask = act_history[-1][1]
            agg_mask = agg_history[-1][1]

            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            cross_active = jnp.outer(act_active, agg_active)

            delta = self.cross_ltd_rate * lr_mult * 0.3 * cross_active
            new_cross = jnp.clip(new_cross - delta, 0.0, 1.0)

        return new_cross

    def _compute_protection_scores_act(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute activation protection scores including cross-domain."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        pairwise_score = jnp.dot(weights, active) / n_active

        # Cross-domain score
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
        """Compute aggregation protection scores including cross-domain."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        pairwise_score = jnp.dot(weights, active) / n_active

        # Cross-domain score
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
        """Apply phase-appropriate mutation."""
        key1, key2 = jax.random.split(key)

        if is_activation:
            num_funcs = NUM_ACTIVATIONS
            max_active = self.max_active_act
            min_active = self.min_active_act
            protection_threshold = self.act_affinity_protection_threshold

            if phase == CriticalPeriodPhase.EXPLORATION:
                activate_rate = self.act_exploration_activate
                deactivate_rate = self.act_exploration_deactivate
            elif phase == CriticalPeriodPhase.CONFIRMATION:
                activate_rate = self.act_confirmation_activate
                deactivate_max = self.act_confirmation_deactivate_max
                deactivate_min = self.act_confirmation_deactivate_min
            else:
                activate_rate = self.act_consolidation_activate
                deactivate_rate = self.act_consolidation_deactivate
        else:
            num_funcs = NUM_AGGREGATIONS
            max_active = self.max_active_agg
            min_active = self.min_active_agg
            protection_threshold = self.agg_affinity_protection_threshold

            if phase == CriticalPeriodPhase.EXPLORATION:
                activate_rate = self.agg_exploration_activate
                deactivate_rate = self.agg_exploration_deactivate
            elif phase == CriticalPeriodPhase.CONFIRMATION:
                activate_rate = self.agg_confirmation_activate
                deactivate_max = self.agg_confirmation_deactivate_max
                deactivate_min = self.agg_confirmation_deactivate_min
            else:
                activate_rate = self.agg_consolidation_activate
                deactivate_rate = self.agg_consolidation_deactivate

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (num_funcs,))
        deactivate_probs = jax.random.uniform(key2, (num_funcs,))

        for i in range(num_funcs):
            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                # Inactive - might activate
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
                # Active - might deactivate
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= protection_threshold:
                        continue
                    deact_rate = deactivate_rate

                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= protection_threshold:
                        deact_rate = deactivate_min
                    else:
                        t = protection / protection_threshold
                        deact_rate = deactivate_max * (1 - t) + deactivate_min * t
                else:
                    deact_rate = deactivate_rate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
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
        """Update with dual STDP learning."""
        key, subkey1, subkey2 = jax.random.split(state['rng_key'], 3)

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

        # Improvement magnitude
        improvement_magnitude = (best_fitness - prev_best_fitness) / max(0.1, prev_best_fitness)
        improvement_magnitude = max(0.0, min(1.0, improvement_magnitude))

        # Phase-specific LR
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr_mult = self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr_mult = self.confirmation_lr_multiplier
        else:
            lr_mult = 0.1

        # Update histories
        act_history = state['act_history'] + [(generation, state['act_mask'].copy(), best_fitness)]
        if len(act_history) > self.history_length:
            act_history = act_history[-self.history_length:]

        agg_history = state['agg_history'] + [(generation, state['agg_mask'].copy(), best_fitness)]
        if len(agg_history) > self.history_length:
            agg_history = agg_history[-self.history_length:]

        # STDP updates for both domains
        new_act_affinity, new_act_weights, act_info = self._stdp_update_domain(
            state['act_affinity'], state['act_weights'], act_history,
            generation, improved, improvement_magnitude, lr_mult, True
        )

        new_agg_affinity, new_agg_weights, agg_info = self._stdp_update_domain(
            state['agg_affinity'], state['agg_weights'], agg_history,
            generation, improved, improvement_magnitude, lr_mult, False
        )

        # Cross-domain STDP
        new_cross = self._stdp_update_cross_domain(
            state['cross_affinity'], act_history, agg_history,
            generation, improved, improvement_magnitude, lr_mult
        )

        # Compute protection scores
        act_protection = self._compute_protection_scores_act(
            new_act_affinity, new_act_weights, state['act_mask'],
            new_cross, state['agg_mask']
        )
        agg_protection = self._compute_protection_scores_agg(
            new_agg_affinity, new_agg_weights, state['agg_mask'],
            new_cross, state['act_mask']
        )

        # Apply mutations
        new_act_mask, act_mutation = self._mutate_palette(
            subkey1, state['act_mask'], phase, act_protection, True
        )
        new_agg_mask, agg_mutation = self._mutate_palette(
            subkey2, state['agg_mask'], phase, agg_protection, False
        )

        # Track STDP events
        ltp_events = state['ltp_events'] + (1 if improved else 0)
        ltd_events = state['ltd_events'] + (0 if improved else 1)

        new_state = {
            'act_mask': new_act_mask,
            'act_affinity': new_act_affinity,
            'act_weights': new_act_weights,
            'act_history': act_history,
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_affinity,
            'agg_weights': new_agg_weights,
            'agg_history': agg_history,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'fitness_ema': new_fitness_ema,
            'ltp_events': ltp_events,
            'ltd_events': ltd_events,
            'mask': new_act_mask,
        }

        # Metrics
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = agg_mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            # Activation stats
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'sin_affinity': float(new_act_affinity[4]),
            'act_activated': act_mutation['activated'],
            'act_deactivated': act_mutation['deactivated'],
            # Aggregation stats
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'agg4_affinity': float(new_agg_affinity[3]) if NUM_AGGREGATIONS > 3 else 0.0,
            'agg_activated': agg_mutation['activated'],
            'agg_deactivated': agg_mutation['deactivated'],
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # STDP stats
            'total_ltp_events': ltp_events,
            'total_ltd_events': ltd_events,
            'improvement_magnitude': improvement_magnitude,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual STDP stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'has_agg4': 3 in agg_palette,
            'phase': state['phase'],
            'generation': state['generation'],
            'sin_affinity': float(state['act_affinity'][4]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
            'stagnation_count': state['stagnation_count'],
            'total_ltp_events': state['ltp_events'],
            'total_ltd_events': state['ltd_events'],
        }
