"""Strategy 114: Cross-Reactive Memory + Aggregation Dual.

Combines immune memory cross-reactivity (strategy 44) with aggregation-led
dual discovery (strategy 101) for bidirectional protection.

Key Innovation:
- Sin forms MEMORY that CROSS-PROTECTS extreme aggregations
- Extreme aggregations form memory that cross-protects sin
- Bidirectional cross-reactivity creates mutual protection
- Memory cells provide long-lived retention across domain shifts

Bio inspiration: In the immune system, B-cell and T-cell memory provides
rapid response to previously encountered antigens. Cross-reactivity
allows memory cells to recognize similar (but not identical) threats.
Similarly, sin's "memory" of success can protect related extremes.

Expected: Bidirectional cross-protection makes sin-extreme forgetting
nearly impossible through mutual memory-based retention.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import jax
import jax.numpy as jnp

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


class CrossReactiveAggDualStrategy(PaletteEvolutionStrategy):
    """Immune cross-reactivity with aggregation-led discovery.

    Sin and extremes cross-protect each other
    through immune memory-style bidirectional protection.

    Critical innovation: Sin's memory protects extremes, and extreme's
    memory protects sin, creating redundant retention.
    """

    name = "cross_reactive_agg_dual"
    description = "Dual: Sin-extreme bidirectional cross-protection via immune memory"

    def __init__(
        self,
        # === CROSS-REACTIVITY PARAMETERS (KEY INNOVATION) ===
        memory_formation_threshold: float = 0.7,       # Fitness to form memory
        sin_extreme_cross_radius: int = 1,             # Cross-protection radius
        cross_protection_strength: float = 0.5,        # Protection level
        mutual_memory_boost: float = 0.4,              # Boost when both form memory
        memory_cell_lifespan: int = 40,                # Memory duration
        cross_decay_rate: float = 0.05,                # Cross-protection decay
        # === Aggregation-led parameters (from strategy 101) ===
        agg_exploration_rate: float = 0.25,            # Aggressive agg exploration
        agg_capture_priority: float = 1.2,             # Prioritize agg capture
        sin_extreme_capture_bonus: float = 0.4,        # Bonus for sin-extreme capture
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
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Cross-Reactive Memory + Aggregation strategy."""
        # Cross-reactivity (KEY INNOVATION)
        self.memory_formation_threshold = memory_formation_threshold
        self.sin_extreme_cross_radius = sin_extreme_cross_radius
        self.cross_protection_strength = cross_protection_strength
        self.mutual_memory_boost = mutual_memory_boost
        self.memory_cell_lifespan = memory_cell_lifespan
        self.cross_decay_rate = cross_decay_rate

        # Aggregation-led
        self.agg_exploration_rate = agg_exploration_rate
        self.agg_capture_priority = agg_capture_priority
        self.sin_extreme_capture_bonus = sin_extreme_capture_bonus

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
        """Initialize state with memory + aggregation tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
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

        # Tagging
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # MEMORY CELLS: {func_idx: {'formation_gen': int, 'protection': float}}
        # Separate memory for activations and aggregations
        # CROSS-PROTECTION: per-function protection from other domain's memory
        act_cross_protection = jnp.zeros(NUM_ACTIVATIONS)
        agg_cross_protection = jnp.zeros(NUM_AGGREGATIONS)

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
            # Cross-domain affinity
            'cross_affinity': cross_affinity,
            # MEMORY (KEY)
            'act_memory_cells': {},  # {idx: {'formation_gen', 'protection'}}
            'agg_memory_cells': {},
            'act_cross_protection': act_cross_protection,
            'agg_cross_protection': agg_cross_protection,
            # Stats
            'capture_events': 0,
            'memory_formation_events': 0,
            'cross_protection_events': 0,
            'mutual_memory_events': 0,
            'diversity_rescues': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1140000),
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

    def _update_memory_cells(
        self,
        act_memory: Dict[int, Dict],
        agg_memory: Dict[int, Dict],
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        best_fitness: float,
        generation: int,
    ) -> Tuple[Dict, Dict, int, int]:
        """Update memory cells - form new, decay old, track mutual formation."""
        new_act_memory = dict(act_memory)
        new_agg_memory = dict(agg_memory)
        memory_formed = 0
        mutual_memory = 0

        # Track what forms memory this round
        act_formed = []
        agg_formed = []

        # Form memory for successful palette
        if best_fitness >= self.memory_formation_threshold:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    if i not in new_act_memory:
                        new_act_memory[i] = {
                            'formation_gen': generation,
                            'protection': 0.9,
                        }
                        memory_formed += 1
                        act_formed.append(i)
                    else:
                        # Boost existing memory
                        new_act_memory[i]['protection'] = min(
                            1.0, new_act_memory[i]['protection'] + 0.1
                        )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    if j not in new_agg_memory:
                        new_agg_memory[j] = {
                            'formation_gen': generation,
                            'protection': 0.9,
                        }
                        memory_formed += 1
                        agg_formed.append(j)
                    else:
                        new_agg_memory[j]['protection'] = min(
                            1.0, new_agg_memory[j]['protection'] + 0.1
                        )

        # MUTUAL MEMORY: sin and extremes forming memory together
        if 4 in act_formed:
            for j in CORE_EXTREME_AGGS:
                if j in agg_formed:
                    # Mutual boost
                    new_act_memory[4]['protection'] = min(
                        1.0, new_act_memory[4]['protection'] + self.mutual_memory_boost
                    )
                    new_agg_memory[j]['protection'] = min(
                        1.0, new_agg_memory[j]['protection'] + self.mutual_memory_boost
                    )
                    mutual_memory += 1

        # Decay old memories
        expired_act = []
        expired_agg = []
        for i, info in new_act_memory.items():
            age = generation - info['formation_gen']
            if age > self.memory_cell_lifespan:
                expired_act.append(i)
            elif age > self.memory_cell_lifespan * 0.7:
                decay_factor = (self.memory_cell_lifespan - age) / (self.memory_cell_lifespan * 0.3)
                new_act_memory[i]['protection'] *= decay_factor

        for j, info in new_agg_memory.items():
            age = generation - info['formation_gen']
            if age > self.memory_cell_lifespan:
                expired_agg.append(j)
            elif age > self.memory_cell_lifespan * 0.7:
                decay_factor = (self.memory_cell_lifespan - age) / (self.memory_cell_lifespan * 0.3)
                new_agg_memory[j]['protection'] *= decay_factor

        for i in expired_act:
            del new_act_memory[i]
        for j in expired_agg:
            del new_agg_memory[j]

        return new_act_memory, new_agg_memory, memory_formed, mutual_memory

    def _update_cross_protection(
        self,
        act_cross_protection: jnp.ndarray,
        agg_cross_protection: jnp.ndarray,
        act_memory: Dict[int, Dict],
        agg_memory: Dict[int, Dict],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Update bidirectional cross-protection.

        KEY INNOVATION: Sin's memory protects extreme aggregations.
        Extreme aggregation's memory protects sin.
        """
        cross_protection_events = 0

        # Decay existing cross-protection
        new_act_cross = act_cross_protection * (1 - self.cross_decay_rate)
        new_agg_cross = agg_cross_protection * (1 - self.cross_decay_rate)

        # SIN → EXTREMES: Sin's memory protects extreme aggregations
        if 4 in act_memory:
            sin_protection = act_memory[4]['protection']
            for j in CORE_EXTREME_AGGS:
                current = float(new_agg_cross[j])
                boost = sin_protection * self.cross_protection_strength
                new_agg_cross = new_agg_cross.at[j].set(
                    min(1.0, current + boost)
                )
                cross_protection_events += 1

        # EXTREMES → SIN: Extreme memory protects sin
        for j in CORE_EXTREME_AGGS:
            if j in agg_memory:
                extreme_protection = agg_memory[j]['protection']
                current = float(new_act_cross[4])
                boost = extreme_protection * self.cross_protection_strength
                new_act_cross = new_act_cross.at[4].set(
                    min(1.0, current + boost)
                )
                cross_protection_events += 1

        return new_act_cross, new_agg_cross, cross_protection_events

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_memory: Dict,
        agg_memory: Dict,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with memory boost."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:
                    tag_strength *= self.extreme_tag_boost
                # Memory boost
                if i in act_memory:
                    tag_strength *= (1 + 0.3 * act_memory[i]['protection'])
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = self.agg_capture_priority  # Aggregation-led priority
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                if j in agg_memory:
                    tag_strength *= (1 + 0.3 * agg_memory[j]['protection'])
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Attempt capture with sin-extreme bonus."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

        # Track if sin captures
        sin_capturing = False

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1
                        if i == 4:
                            sin_capturing = True

                for j in range(NUM_AGGREGATIONS):
                    threshold = self.agg_tag_threshold
                    # Lower threshold for extremes when sin is capturing
                    if sin_capturing and j in CORE_EXTREME_AGGS:
                        threshold *= (1 - self.sin_extreme_capture_bonus)
                    if hist_agg_tags[j] > threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

        return new_act_captured, new_agg_captured, capture_count

    def _update_affinities(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_cross_protection: jnp.ndarray,
        agg_cross_protection: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update affinities with cross-protection influence."""
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        # Cross-protection adds to affinity (makes harder to remove)
        new_act_aff = new_act_aff + act_cross_protection * 0.1
        new_agg_aff = new_agg_aff + agg_cross_protection * 0.1
        new_act_aff = jnp.clip(new_act_aff, 0.0, 1.0)
        new_agg_aff = jnp.clip(new_agg_aff, 0.0, 1.0)

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

        # Cross-domain update
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(act_active, agg_active)
            delta = self.base_cross_learning_rate * fitness_delta * co_active
            new_cross = jnp.clip(new_cross + delta, 0.0, 1.0)

        return new_act_aff, new_agg_aff, new_cross

    def _explore_new_functions(
        self,
        mask: jnp.ndarray,
        n_funcs: int,
        stagnation: int,
        generation: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
        is_agg: bool = False,
    ) -> jnp.ndarray:
        """Add exploration of new functions - KEY FIX for discovery."""
        k1, k2 = jax.random.split(key)
        new_mask = mask.copy()

        # Exploration rate scales with stagnation
        exploration_prob = self.agg_exploration_rate * (1 + stagnation * 0.1)
        exploration_prob = min(exploration_prob, 0.6)

        # Always try to add preferred indices (sin/extremes) if not present
        if prefer_indices:
            for idx in prefer_indices:
                if 0 <= idx < n_funcs and new_mask[idx] < 0.5:
                    # Higher probability for preferred indices
                    if float(jax.random.uniform(k1)) < exploration_prob * 1.5:
                        new_mask = new_mask.at[idx].set(1.0)
                    k1, _ = jax.random.split(k1)

        # Additional random exploration on stagnation
        if stagnation > 3 or generation % 5 == 0:
            inactive = [i for i in range(n_funcs) if new_mask[i] < 0.5]
            if inactive and float(jax.random.uniform(k2)) < exploration_prob:
                # Prefer sin/extremes for random exploration
                priority = [i for i in inactive if i in (prefer_indices or [])]
                pool = priority if priority else inactive
                to_add = int(jax.random.choice(k2, jnp.array(pool)))
                new_mask = new_mask.at[to_add].set(1.0)

        return new_mask

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        cross_protection: jnp.ndarray,
        memory: Dict,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
        stagnation: int = 0,
        generation: int = 0,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette with memory + cross-protection influence + exploration."""
        k1, k2 = jax.random.split(key)

        # Score: affinity + capture + tag + cross-protection + memory
        score = affinities + captured * 0.25 + tags * 0.15 + cross_protection * 0.2

        # Memory influence
        for i in range(n_funcs):
            if i in memory:
                mem_boost = memory[i]['protection'] * 0.3
                score = score.at[i].set(score[i] + mem_boost)

        # Strong preference for sin/extremes
        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + 0.8)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Exploration phase - add new functions to discover
        mask = self._explore_new_functions(
            mask, n_funcs, stagnation, generation, k1,
            prefer_indices=prefer_indices, is_agg=(n_funcs == NUM_AGGREGATIONS)
        )

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < min_diversity:
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k2, jnp.array(inactive), shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with memory + cross-protection dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === MEMORY UPDATE ===
        new_act_memory, new_agg_memory, memory_formed, mutual_memory = \
            self._update_memory_cells(
                state['act_memory_cells'], state['agg_memory_cells'],
                state['act_mask'], state['agg_mask'],
                best_fitness, generation
            )

        # === CROSS-PROTECTION UPDATE ===
        new_act_cross, new_agg_cross, cross_events = self._update_cross_protection(
            state['act_cross_protection'], state['agg_cross_protection'],
            new_act_memory, new_agg_memory
        )

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags'],
            new_act_memory, new_agg_memory
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE ===
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            state['act_mask'], state['agg_mask'],
            new_tag_history, generation, improved
        )

        # === AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross_affinity = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], state['agg_mask'],
            new_act_cross, new_agg_cross,
            state['cross_affinity'], fitness_delta
        )

        # === PALETTE SELECTION (with exploration) ===
        new_act_mask, act_diversity_rescue = self._select_palette(
            new_act_aff, new_act_captured, new_act_tags, new_act_cross, new_act_memory,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            self.min_diversity_act, k1, prefer_indices=[4],
            stagnation=new_stagnation, generation=generation
        )
        new_agg_mask, agg_diversity_rescue = self._select_palette(
            new_agg_aff, new_agg_captured, new_agg_tags, new_agg_cross, new_agg_memory,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            self.min_diversity_agg, k2, prefer_indices=list(CORE_EXTREME_AGGS),
            stagnation=new_stagnation, generation=generation
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
            'act_memory_cells': new_act_memory,
            'agg_memory_cells': new_agg_memory,
            'act_cross_protection': new_act_cross,
            'agg_cross_protection': new_agg_cross,
            'capture_events': state['capture_events'] + capture_count,
            'memory_formation_events': state['memory_formation_events'] + memory_formed,
            'cross_protection_events': state['cross_protection_events'] + cross_events,
            'mutual_memory_events': state['mutual_memory_events'] + mutual_memory,
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue + agg_diversity_rescue,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Memory metrics (KEY)
            'n_act_memory': len(new_act_memory),
            'n_agg_memory': len(new_agg_memory),
            'sin_has_memory': 4 in new_act_memory,
            'max_has_memory': 2 in new_agg_memory,
            'min_has_memory': 3 in new_agg_memory,
            'memory_formation_events': new_state['memory_formation_events'],
            'mutual_memory_events': new_state['mutual_memory_events'],
            # Cross-protection (KEY)
            'sin_cross_protection': float(new_act_cross[4]),
            'max_cross_protection': float(new_agg_cross[2]),
            'min_cross_protection': float(new_agg_cross[3]),
            'cross_protection_events': new_state['cross_protection_events'],
            # Affinity metrics
            'sin_affinity': float(new_act_aff[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with memory + cross-protection status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Memory status
            'n_act_memory': len(state['act_memory_cells']),
            'n_agg_memory': len(state['agg_memory_cells']),
            'sin_has_memory': 4 in state['act_memory_cells'],
            'memory_formation_events': state['memory_formation_events'],
            'mutual_memory_events': state['mutual_memory_events'],
            # Cross-protection
            'sin_cross_protection': float(state['act_cross_protection'][4]),
            'cross_protection_events': state['cross_protection_events'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
