"""Strategy 13 Symmetric: Homeostatic Hebbian for Activation AND Aggregation Discovery.

Extends HomeostaticHebbian to symmetric palette evolution with homeostatic
regulation to prevent affinity saturation in both domains, combined with
memory cells and affinity floors for guaranteed retention.

Key mechanisms:
1. Synaptic scaling for both activation and aggregation affinities
2. Soft bounds prevent saturation in both domains
3. Cross-domain affinity matrix tracks act-agg combinations
4. Memory cells crystallize high-value functions
5. Protected indices for sin/extreme aggregations
6. Affinity floors for guaranteed retention

Biological rationale:
- Homeostatic plasticity operates across ALL synapses, not just one type
- The brain maintains activity within functional ranges for all circuits
- Different modalities have separate but coordinated scaling
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


# Critical indices for guaranteed retention
SIN_IDX = 4  # Sin activation - critical for parity problems
CORE_EXTREME_AGGS = [2, 3]  # max (idx 2), min (idx 3) - critical for aggregation


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class HomeostaticHebbianSymmetricStrategy(PaletteEvolutionStrategy):
    """Homeostatic Hebbian with symmetric palette evolution.

    Extends homeostatic regulation to both activation AND aggregation palettes,
    with memory cells and affinity floors for guaranteed retention.
    """

    name = "homeostatic_hebbian_symmetric"
    description = "Symmetric homeostatic Hebbian with memory cells"

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
        affinity_floor_base: float = 0.05,
        soft_ceiling_start: float = 0.80,
        soft_floor_start: float = 0.20,
        # Memory cell parameters (from winning patterns)
        memory_threshold: float = 0.75,
        memory_sustain_generations: int = 8,
        memory_decay_rate: float = 0.05,
        # Affinity floors (CRITICAL for retention)
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Constraints
        early_consolidation_threshold: float = 0.95,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize symmetric homeostatic Hebbian strategy."""
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
        self.affinity_floor_base = affinity_floor_base
        self.soft_ceiling_start = soft_ceiling_start
        self.soft_floor_start = soft_floor_start

        # Memory cell parameters
        self.memory_threshold = memory_threshold
        self.memory_sustain_generations = memory_sustain_generations
        self.memory_decay_rate = memory_decay_rate

        # Affinity floors for critical functions
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

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
        """Initialize state with homeostatic tracking and memory cells."""
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

        # Memory cells (symmetric pattern)
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        sin_discovered = SIN_IDX in initial_act
        sin_discovery_gen = 0 if sin_discovered else -1
        extreme_agg_discovered = any(idx in initial_agg for idx in CORE_EXTREME_AGGS)
        extreme_agg_discovery_gen = 0 if extreme_agg_discovered else -1

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
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_counts': agg_memory_counts,
            'agg_memory_cells': agg_memory_cells,
            # Discovery tracking
            'sin_discovered': sin_discovered,
            'sin_discovery_gen': sin_discovery_gen,
            'extreme_agg_discovered': extreme_agg_discovered,
            'extreme_agg_discovery_gen': extreme_agg_discovery_gen,
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
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
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

        dist_to_floor = affinity - self.affinity_floor_base
        floor_factor = jnp.where(
            affinity < self.soft_floor_start,
            dist_to_floor / (self.soft_floor_start - self.affinity_floor_base),
            jnp.ones_like(affinity)
        )
        floor_factor = jnp.clip(floor_factor, 0.0, 1.0)

        scaled_delta = jnp.where(positive_mask, delta * ceiling_factor, delta * floor_factor)
        new_affinity = affinity + scaled_delta
        return jnp.clip(new_affinity, self.affinity_floor_base, self.affinity_ceiling)

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
            return jnp.clip(scaled_affinity, self.affinity_floor_base, self.affinity_ceiling), True

        return affinity, False

    def _apply_affinity_floors(
        self, act_affinity: jnp.ndarray, agg_affinity: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for sin and extreme aggregations."""
        new_act = act_affinity.copy()
        new_agg = agg_affinity.copy()

        # Sin activation floor
        new_act = new_act.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation floors
        for idx in CORE_EXTREME_AGGS:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(agg_affinity[idx], self.extreme_agg_affinity_floor)
            )

        return new_act, new_agg

    def _update_memory_cells(
        self, affinity: jnp.ndarray, memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray, mask: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell counts and crystallize sustained high-affinity functions."""
        above_threshold = affinity >= self.memory_threshold
        active = mask > 0.5

        new_counts = jnp.where(
            above_threshold & active,
            memory_counts + 1,
            jnp.zeros_like(memory_counts)
        )

        newly_memory = new_counts >= self.memory_sustain_generations
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

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
        """Update Hebbian weights with homeostatic regulation."""
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

    def _compute_protection_scores(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        other_mask: jnp.ndarray,
        is_act: bool,
        memory_cells: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection including memory cell status."""
        active = (mask > 0.5).astype(jnp.float32)
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        n_other = max(jnp.sum(other_active), 1)

        pairwise_score = jnp.dot(weights, active) / n_active

        if is_act:
            cross_score = jnp.dot(cross_affinity, other_active) / n_other
        else:
            cross_score = jnp.dot(cross_affinity.T, other_active) / n_other

        base_prot = (
            0.50 * affinity +
            0.25 * pairwise_score +
            0.15 * cross_score * self.cross_influence
        )

        # Memory cells get maximum protection
        memory_boost = memory_cells.astype(jnp.float32) * 0.10

        return jnp.clip(base_prot + memory_boost, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
        num_funcs: int,
        min_active: int,
        max_active: int,
        is_act: bool,
        memory_cells: jnp.ndarray,
        protected_indices: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with protected indices."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (num_funcs,))
        deactivate_probs = jax.random.uniform(key2, (num_funcs,))

        protected_set = set(protected_indices or [])

        # Get phase-specific rates
        if is_act:
            if phase == CriticalPeriodPhase.EXPLORATION:
                activate_rate = self.exploration_activate
                use_protection = False
            elif phase == CriticalPeriodPhase.CONFIRMATION:
                activate_rate = self.confirmation_activate
                use_protection = True
            else:
                activate_rate = self.consolidation_activate
                use_protection = True
            deact_max = self.confirmation_deactivate_max
            deact_min = self.confirmation_deactivate_min
        else:
            if phase == CriticalPeriodPhase.EXPLORATION:
                activate_rate = self.agg_exploration_activate
                use_protection = False
            elif phase == CriticalPeriodPhase.CONFIRMATION:
                activate_rate = self.confirmation_activate
                use_protection = True
            else:
                activate_rate = self.consolidation_activate
                use_protection = True
            deact_max = self.agg_confirmation_deactivate_max
            deact_min = self.agg_confirmation_deactivate_min

        current_active = int(jnp.sum(mask > 0.5))

        for i in range(num_funcs):
            protection = float(protection_scores[i])
            is_memory = bool(memory_cells[i])
            is_protected = i in protected_set

            if mask[i] < 0.5:
                if current_active + len(activated) >= max_active:
                    continue

                if is_protected:
                    effective_rate = activate_rate * 2.0
                elif use_protection and phase == CriticalPeriodPhase.CONFIRMATION:
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Memory cells never deactivate
                if is_memory:
                    continue

                # Protected indices almost never deactivate
                if is_protected:
                    if deactivate_probs[i] < 0.001:
                        new_mask = new_mask.at[i].set(0.0)
                        deactivated.append(i)
                    continue

                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= self.affinity_protection_threshold:
                        continue
                    deact_rate = self.consolidation_deactivate
                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= self.affinity_protection_threshold:
                        deact_rate = deact_min
                    else:
                        t = protection / self.affinity_protection_threshold
                        deact_rate = deact_max * (1 - t) + deact_min * t
                else:
                    deact_rate = self.exploration_deactivate if is_act else self.agg_exploration_deactivate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        prefix = 'act_' if is_act else 'agg_'
        return new_mask, {f'{prefix}activated': activated, f'{prefix}deactivated': deactivated}

    def _update_discovery(
        self, state: Dict[str, Any], generation: int
    ) -> Dict[str, Any]:
        """Track discovery of sin and extreme aggregations."""
        updates = {}

        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        if not state['sin_discovered'] and SIN_IDX in act_palette:
            updates['sin_discovered'] = True
            updates['sin_discovery_gen'] = generation

        if not state['extreme_agg_discovered']:
            if any(idx in agg_palette for idx in CORE_EXTREME_AGGS):
                updates['extreme_agg_discovered'] = True
                updates['extreme_agg_discovery_gen'] = generation

        return updates

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with homeostatic Hebbian learning and memory cells."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

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

        # Apply affinity floors AFTER homeostatic scaling
        new_act_affinity, new_agg_affinity = self._apply_affinity_floors(
            new_act_affinity, new_agg_affinity
        )

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_affinity, state['act_memory_counts'],
            state['act_memory_cells'], state['act_mask'])
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_affinity, state['agg_memory_counts'],
            state['agg_memory_cells'], state['agg_mask'])

        # Compute protection scores
        act_protection = self._compute_protection_scores(
            new_act_affinity, new_act_weights, state['act_mask'],
            new_cross_affinity, state['agg_mask'], True, new_act_mem_cells
        )
        agg_protection = self._compute_protection_scores(
            new_agg_affinity, new_agg_weights, state['agg_mask'],
            new_cross_affinity, state['act_mask'], False, new_agg_mem_cells
        )

        # Apply mutations with protected indices
        new_act_mask, act_mutation = self._mutate_palette(
            key_act, state['act_mask'], phase, act_protection,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            True, new_act_mem_cells, protected_indices=[SIN_IDX]
        )
        new_agg_mask, agg_mutation = self._mutate_palette(
            key_agg, state['agg_mask'], phase, agg_protection,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            False, new_agg_mem_cells, protected_indices=CORE_EXTREME_AGGS
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
            # Memory cells
            'act_memory_counts': new_act_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_counts': new_agg_mem_counts,
            'agg_memory_cells': new_agg_mem_cells,
            # Discovery tracking
            'sin_discovered': state['sin_discovered'],
            'sin_discovery_gen': state['sin_discovery_gen'],
            'extreme_agg_discovered': state['extreme_agg_discovered'],
            'extreme_agg_discovery_gen': state['extreme_agg_discovery_gen'],
            # Common state
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

        # Update discovery tracking
        discovery_updates = self._update_discovery(new_state, generation)
        new_state.update(discovery_updates)

        # Compute metrics
        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))
        active_act_palette = mask_to_indices(new_act_mask)
        active_agg_palette = mask_to_indices(new_agg_mask)

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
            'sin_affinity': float(new_act_affinity[SIN_IDX]),
            'act_scaled': act_scaled,
            'act_scaling_count': act_scaling_count,
            'act_memory_cells': act_mem_count,
            # Aggregation stats
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'agg_max_affinity': float(jnp.max(new_agg_affinity)),
            'max_agg_affinity': float(new_agg_affinity[2]),
            'min_agg_affinity': float(new_agg_affinity[3]),
            'agg_scaled': agg_scaled,
            'agg_scaling_count': agg_scaling_count,
            'agg_memory_cells': agg_mem_count,
            # Cross-domain stats
            'cross_avg_affinity': float(jnp.mean(new_cross_affinity)),
            'cross_max_affinity': float(jnp.max(new_cross_affinity)),
            # Memory cell totals
            'total_memory_cells': act_mem_count + agg_mem_count,
            'sin_is_memory': bool(new_act_mem_cells[SIN_IDX]),
            # Discovery
            'sin_discovered': new_state['sin_discovered'],
            'sin_discovery_gen': new_state['sin_discovery_gen'],
            'extreme_agg_discovered': new_state['extreme_agg_discovered'],
            'extreme_agg_discovery_gen': new_state['extreme_agg_discovery_gen'],
        }
        metrics.update(act_mutation)
        metrics.update(agg_mutation)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with homeostatic and memory stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        act_mem_count = int(jnp.sum(state['act_memory_cells']))
        agg_mem_count = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'act_palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': SIN_IDX in act_palette,
            'has_extreme_aggs': any(idx in agg_palette for idx in CORE_EXTREME_AGGS),
            'phase': state['phase'],
            'generation': state['generation'],
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            'act_scaling_count': state['act_scaling_count'],
            'agg_scaling_count': state['agg_scaling_count'],
            'sin_discovered': state['sin_discovered'],
            'sin_discovery_gen': state['sin_discovery_gen'],
            'extreme_agg_discovered': state['extreme_agg_discovered'],
            'extreme_agg_discovery_gen': state['extreme_agg_discovery_gen'],
            'stagnation_count': state['stagnation_count'],
        }
