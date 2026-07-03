"""Strategy 103 Symmetric: Cross-Domain Reinforcement v2 for Activation AND Aggregation.

Extends cross_domain_v2_dual to symmetric palette evolution with multiplicative
cross-domain dynamics applied to both activation and aggregation discovery.

Key mechanisms:
1. MULTIPLICATIVE (not additive) cross-domain affinity updates
2. Sin-extreme multiplier (1.5x) creates robust pairings
3. Mutual capture bonus when both domains discover together
4. Memory cells crystallize high-value functions
5. Protected indices for sin/extreme aggregations
6. Affinity floors for guaranteed retention

Bio inspiration: Hebbian plasticity in cortical circuits shows multiplicative
rather than additive dynamics. LTP/LTD effects multiply existing synaptic strength,
creating stronger "winner-take-all" dynamics for successful pairings.

Expected: Sin-extreme pairings survive domain shifts through robust multiplicative coupling.
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


# Critical indices for guaranteed retention
SIN_IDX = 4  # Sin activation - critical for parity problems


class CrossDomainV2SymmetricStrategy(PaletteEvolutionStrategy):
    """Cross-domain reinforcement v2 with symmetric palette evolution.

    Multiplicative cross-domain dynamics for both activation and aggregation
    discovery with memory cells and affinity floors for guaranteed retention.
    """

    name = "cross_domain_v2_symmetric"
    description = "Symmetric multiplicative cross-domain reinforcement"

    def __init__(
        self,
        # === MULTIPLICATIVE CROSS-DOMAIN PARAMETERS ===
        multiplicative_update: bool = True,
        cross_affinity_growth_factor: float = 1.15,   # Multiply on success
        cross_affinity_decay_factor: float = 0.98,    # Multiply on no success
        sin_extreme_multiplier: float = 1.5,          # Extra boost for sin-extreme
        mutual_capture_bonus: float = 0.4,            # Bonus when both domains capture
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        base_cross_learning_rate: float = 0.10,
        # === Tagging parameters ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        # === Memory cell parameters ===
        memory_threshold: float = 0.75,
        memory_sustain_generations: int = 8,
        # === Affinity floors (CRITICAL for retention) ===
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # === Mutation parameters ===
        act_base_activate_rate: float = 0.12,
        act_base_deactivate_rate: float = 0.05,
        agg_base_activate_rate: float = 0.10,
        agg_base_deactivate_rate: float = 0.04,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Cross-Domain v2 symmetric strategy."""
        # Multiplicative cross-domain
        self.multiplicative_update = multiplicative_update
        self.cross_affinity_growth_factor = cross_affinity_growth_factor
        self.cross_affinity_decay_factor = cross_affinity_decay_factor
        self.sin_extreme_multiplier = sin_extreme_multiplier
        self.mutual_capture_bonus = mutual_capture_bonus

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.base_cross_learning_rate = base_cross_learning_rate

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # Memory cells
        self.memory_threshold = memory_threshold
        self.memory_sustain_generations = memory_sustain_generations

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Mutation
        self.act_base_activate_rate = act_base_activate_rate
        self.act_base_deactivate_rate = act_base_deactivate_rate
        self.agg_base_activate_rate = agg_base_activate_rate
        self.agg_base_deactivate_rate = agg_base_deactivate_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with cross-domain affinity tracking and memory cells."""
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity (starts higher for sin-extreme)
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for j in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[SIN_IDX, j].set(0.6)

        # Memory cells
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
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain (core of this strategy)
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
            # Stats
            'capture_events': 0,
            'mutual_capture_events': 0,
            'multiplicative_growth_events': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1030303),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

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

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        cross_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with cross-affinity boost."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == SIN_IDX:
                    tag_strength *= self.extreme_tag_boost
                # Cross-affinity boost
                max_cross = float(jnp.max(cross_affinity[i, :]))
                tag_strength *= (1 + 0.2 * max_cross)
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                max_cross = float(jnp.max(cross_affinity[:, j]))
                tag_strength *= (1 + 0.2 * max_cross)
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture_with_mutual_bonus(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]:
        """Attempt capture with mutual capture bonus."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        new_cross = cross_affinity.copy()
        capture_count = 0
        mutual_capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, new_cross, 0, 0

        act_captured_this_round = []
        agg_captured_this_round = []

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1
                        act_captured_this_round.append(i)

                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1
                        agg_captured_this_round.append(j)

        # MUTUAL CAPTURE BONUS
        for i in act_captured_this_round:
            for j in agg_captured_this_round:
                current = float(new_cross[i, j])
                bonus_factor = 1.0 + self.mutual_capture_bonus
                if i == SIN_IDX and j in CORE_EXTREME_AGGS:
                    bonus_factor *= self.sin_extreme_multiplier
                new_cross = new_cross.at[i, j].set(
                    min(1.0, current * bonus_factor)
                )
                mutual_capture_count += 1

        return new_act_captured, new_agg_captured, new_cross, capture_count, mutual_capture_count

    def _update_affinities_multiplicative(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Update affinities with MULTIPLICATIVE cross-domain dynamics."""
        growth_events = 0

        # Decay individual affinities
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        # Learning for individual domains
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * fitness_delta)
                    )
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta)
                    )

        # MULTIPLICATIVE CROSS-DOMAIN UPDATE
        new_cross = cross_affinity.copy()

        if self.multiplicative_update:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    current = float(new_cross[i, j])

                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        if fitness_delta > 0:
                            # GROWTH: Multiply by growth factor
                            growth_factor = self.cross_affinity_growth_factor

                            # Extra boost for sin-extreme pairs
                            if i == SIN_IDX and j in CORE_EXTREME_AGGS:
                                growth_factor *= self.sin_extreme_multiplier

                            new_value = current * growth_factor
                            new_cross = new_cross.at[i, j].set(min(1.0, new_value))
                            growth_events += 1
                        else:
                            # DECAY: Multiply by decay factor (floor at 0.2)
                            new_value = max(0.2, current * self.cross_affinity_decay_factor)
                            new_cross = new_cross.at[i, j].set(new_value)
                    else:
                        # Inactive pairs decay slowly
                        new_value = max(0.1, current * 0.995)
                        new_cross = new_cross.at[i, j].set(new_value)
        else:
            # Fallback to additive update
            if fitness_delta > 0:
                act_active = (act_mask > 0.5).astype(jnp.float32)
                agg_active = (agg_mask > 0.5).astype(jnp.float32)
                co_active = jnp.outer(act_active, agg_active)
                delta = self.base_cross_learning_rate * fitness_delta * co_active
                new_cross = jnp.clip(new_cross + delta, 0.0, 1.0)

        return new_act_aff, new_agg_aff, new_cross, growth_events

    def _apply_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        other_mask: jnp.ndarray,
        is_activation: bool,
        memory_cells: jnp.ndarray,
        protected_indices: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with protected indices for guaranteed retention."""
        key1, key2 = jax.random.split(key)

        if is_activation:
            num_funcs = NUM_ACTIVATIONS
            max_active = self.max_active_act
            min_active = self.min_active_act
            base_activate = self.act_base_activate_rate
            base_deactivate = self.act_base_deactivate_rate
        else:
            num_funcs = NUM_AGGREGATIONS
            max_active = self.max_active_agg
            min_active = self.min_active_agg
            base_activate = self.agg_base_activate_rate
            base_deactivate = self.agg_base_deactivate_rate

        protected_set = set(protected_indices or [])

        activate_probs = jax.random.uniform(key1, (num_funcs,))
        deactivate_probs = jax.random.uniform(key2, (num_funcs,))

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(num_funcs):
            aff = float(affinity[i])
            cap = float(captured[i])
            is_memory = bool(memory_cells[i])
            is_protected = i in protected_set

            # Cross-domain influence
            if is_activation:
                cross_score = float(jnp.max(cross_affinity[i, :] * (other_mask > 0.5)))
            else:
                cross_score = float(jnp.max(cross_affinity[:, i] * (other_mask > 0.5)))

            if mask[i] < 0.5:
                # Activation logic
                current_active = int(jnp.sum(mask > 0.5))
                if current_active >= max_active:
                    continue

                # Protected indices get activation boost
                if is_protected:
                    rate = base_activate * 2.0
                else:
                    rate = base_activate * (0.5 + 0.3 * aff + 0.2 * cross_score)

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Deactivation logic

                # Memory cells never deactivate
                if is_memory:
                    continue

                # Protected indices almost never deactivate (0.1% chance)
                if is_protected:
                    if deactivate_probs[i] < 0.001:
                        new_mask = new_mask.at[i].set(0.0)
                        deactivated.append(i)
                    continue

                # Captured functions have reduced deactivation
                if cap > 0.5:
                    rate = base_deactivate * (1 - self.captured_protection)
                else:
                    rate = base_deactivate * (1.0 - 0.6 * aff - 0.3 * cross_score)
                    rate = max(0.001, min(rate, base_deactivate * 2))

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        prefix = 'act_' if is_activation else 'agg_'
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
        """Update with multiplicative cross-domain dynamics and memory cells."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags'],
            state['cross_affinity']
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE WITH MUTUAL BONUS ===
        (new_act_captured, new_agg_captured, post_capture_cross,
         capture_count, mutual_capture_count) = self._attempt_capture_with_mutual_bonus(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            state['cross_affinity'],
            new_tag_history, generation, improved
        )

        # === MULTIPLICATIVE AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross_affinity, growth_events = \
            self._update_affinities_multiplicative(
                state['act_affinities'], state['agg_affinities'],
                state['act_mask'], state['agg_mask'],
                new_act_captured, new_agg_captured,
                post_capture_cross, fitness_delta
            )

        # === APPLY AFFINITY FLOORS ===
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # === UPDATE MEMORY CELLS ===
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_aff, state['act_memory_counts'],
            state['act_memory_cells'], state['act_mask'])
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_aff, state['agg_memory_counts'],
            state['agg_memory_cells'], state['agg_mask'])

        # === APPLY MUTATIONS WITH PROTECTED INDICES ===
        new_act_mask, act_mut = self._apply_mutation(
            k1, state['act_mask'], new_act_aff, new_act_captured,
            new_cross_affinity, state['agg_mask'], True,
            new_act_mem_cells, protected_indices=[SIN_IDX]
        )
        new_agg_mask, agg_mut = self._apply_mutation(
            k2, state['agg_mask'], new_agg_aff, new_agg_captured,
            new_cross_affinity, new_act_mask, False,
            new_agg_mem_cells, protected_indices=list(CORE_EXTREME_AGGS)
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
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
            # Stats
            'capture_events': state['capture_events'] + capture_count,
            'mutual_capture_events': state['mutual_capture_events'] + mutual_capture_count,
            'multiplicative_growth_events': state['multiplicative_growth_events'] + growth_events,
            # General
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        # Update discovery tracking
        discovery_updates = self._update_discovery(new_state, generation)
        new_state.update(discovery_updates)

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Compute memory cell counts
        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_act_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Cross-domain metrics (KEY)
            'sin_max_cross': float(new_cross_affinity[SIN_IDX, 2]),
            'sin_min_cross': float(new_cross_affinity[SIN_IDX, 3]),
            'max_cross_affinity': float(jnp.max(new_cross_affinity)),
            'mean_cross_affinity': float(jnp.mean(new_cross_affinity)),
            'multiplicative_growth_events': new_state['multiplicative_growth_events'],
            'mutual_capture_events': new_state['mutual_capture_events'],
            # Affinity metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Tagging metrics
            'sin_tag': float(new_act_tags[SIN_IDX]),
            'sin_captured': bool(new_act_captured[SIN_IDX] > 0.5),
            'capture_events': new_state['capture_events'],
            # Status
            'has_sin': SIN_IDX in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Discovery
            'sin_discovered': new_state['sin_discovered'],
            'sin_discovery_gen': new_state['sin_discovery_gen'],
            'extreme_agg_discovered': new_state['extreme_agg_discovered'],
            'extreme_agg_discovery_gen': new_state['extreme_agg_discovery_gen'],
            # Mutations
            **act_mut,
            **agg_mut,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with cross-domain and memory status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        act_mem_count = int(jnp.sum(state['act_memory_cells']))
        agg_mem_count = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': SIN_IDX in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Cross-domain (key metric)
            'sin_max_cross': float(state['cross_affinity'][SIN_IDX, 2]),
            'sin_min_cross': float(state['cross_affinity'][SIN_IDX, 3]),
            'max_cross_affinity': float(jnp.max(state['cross_affinity'])),
            'multiplicative_growth_events': state['multiplicative_growth_events'],
            'mutual_capture_events': state['mutual_capture_events'],
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Affinity
            'sin_captured': bool(state['act_captured'][SIN_IDX] > 0.5),
            'capture_events': state['capture_events'],
            # Discovery
            'sin_discovered': state['sin_discovered'],
            'sin_discovery_gen': state['sin_discovery_gen'],
            'extreme_agg_discovered': state['extreme_agg_discovered'],
            'extreme_agg_discovery_gen': state['extreme_agg_discovery_gen'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
