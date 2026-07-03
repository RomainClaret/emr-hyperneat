"""Strategy 87: Triple Hybrid Dual.

Combines ALL THREE winners:
- Strategy 81 (Synaptic Tagging): Tag-and-capture consolidation (100% sin retention)
- Strategy 82 (Homeostatic): Balance + discovery bonus (100% sin retention)
- Strategy 74 (Cross-Domain Reinforcement): 2x learning when both domains change (67% Parity-5)

Key innovation: Three-layer protection system with coordinated priority:
1. Tagging every generation (marks candidates)
2. Affinity updates with reinforcement (when BOTH domains change)
3. Capture on improvement (requires tag + affinity threshold)
4. Homeostatic correction on stagnation (maintains balance)
5. Triple protection for sin-extreme pairs (all mechanisms reinforce)

Bio inspiration: Neural circuits use multiple overlapping plasticity mechanisms -
short-term tagging, long-term consolidation, homeostatic scaling, and modulatory
reinforcement all work together to create stable, adaptive memory.

Expected: 100% sin retention + improved Parity-5 solving through synergistic mechanisms.
"""

from typing import Dict, Any, List, Optional, Tuple
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


class TripleHybridDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution combining tagging, homeostatic, and reinforcement mechanisms.

    Triple Hybrid combining:
    - Synaptic tagging (81): Two-stage tag-and-capture consolidation
    - Homeostatic plasticity (82): Balance between averaging/extreme aggregations
    - Cross-domain reinforcement (74): 2x learning when both domains improve

    Priority order for mechanism coordination:
    1. Tag active functions every generation
    2. Update affinity with reinforcement (2x when BOTH domains change)
    3. Capture on improvement (requires tag + affinity threshold)
    4. Homeostatic correction on stagnation
    5. Triple protection for sin-extreme pairs
    """

    name = "triple_hybrid_dual"
    description = "Dual: Tag + Homeostatic + Reinforcement triple hybrid"

    def __init__(
        self,
        # === Tagging parameters (from strategy 81) ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        fitness_delta_threshold: float = 0.01,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        # === Homeostatic parameters (from strategy 82) ===
        target_extreme_ratio: float = 0.60,
        imbalance_threshold: float = 0.15,
        correction_strength: float = 1.8,
        discovery_bonus: float = 0.5,
        extreme_protection: float = 0.6,
        sin_protection: float = 0.5,
        # === Reinforcement parameters (from strategy 74) ===
        base_cross_learning_rate: float = 0.15,
        reinforcement_multiplier: float = 2.0,
        affinity_protection_threshold: float = 0.6,
        affinity_protection_strength: float = 0.5,
        # === Triple protection parameters ===
        triple_protection_threshold: float = 0.7,  # All 3 mechanisms agree
        triple_protection_strength: float = 0.9,  # Near-total protection
        capture_affinity_threshold: float = 0.6,  # Affinity needed for capture
        # === Mutation parameters ===
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        stagnation_threshold: int = 5,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Triple Hybrid strategy."""
        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.fitness_delta_threshold = fitness_delta_threshold
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.imbalance_threshold = imbalance_threshold
        self.correction_strength = correction_strength
        self.discovery_bonus = discovery_bonus
        self.extreme_protection = extreme_protection
        self.sin_protection = sin_protection

        # Reinforcement
        self.base_cross_learning_rate = base_cross_learning_rate
        self.reinforcement_multiplier = reinforcement_multiplier
        self.affinity_protection_threshold = affinity_protection_threshold
        self.affinity_protection_strength = affinity_protection_strength

        # Triple protection
        self.triple_protection_threshold = triple_protection_threshold
        self.triple_protection_strength = triple_protection_strength
        self.capture_affinity_threshold = capture_affinity_threshold

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.stagnation_threshold = stagnation_threshold

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with all three mechanisms."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Tagging state (from 81)
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)
        tag_history = []

        # Cross-domain affinity (from 74)
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Track previous state (for reinforcement detection)
        prev_act_mask = act_mask.copy()
        prev_agg_mask = agg_mask.copy()

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'prev_act_mask': prev_act_mask,
            'prev_agg_mask': prev_agg_mask,
            # Tagging (from 81)
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Reinforcement (from 74)
            'cross_affinity': cross_affinity,
            # Tracking
            'capture_events': 0,
            'cross_capture_events': 0,
            'homeostatic_corrections': 0,
            'discovery_bonuses_applied': 0,
            'reinforcement_events': 0,
            'triple_protection_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 870000),
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

    # === MECHANISM 1: TAGGING (from 81) ===
    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags for active functions."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:  # sin
                    tag_strength *= self.extreme_tag_boost
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    # === MECHANISM 2: REINFORCEMENT (from 74) ===
    def _update_cross_affinity_reinforced(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        prev_act_mask: jnp.ndarray,
        prev_agg_mask: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, int]:
        """Update cross-domain affinity with reinforcement multiplier."""
        new_affinity = cross_affinity.copy()
        reinforcement_events = 0

        if improvement > 0:
            act_changed = not jnp.allclose(act_mask, prev_act_mask)
            agg_changed = not jnp.allclose(agg_mask, prev_agg_mask)
            both_changed = act_changed and agg_changed

            lr = self.base_cross_learning_rate
            if both_changed:
                lr *= self.reinforcement_multiplier
                reinforcement_events += 1

            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    for j in range(NUM_AGGREGATIONS):
                        if agg_mask[j] > 0.5:
                            current = cross_affinity[i, j]
                            boost = lr * improvement
                            if i == 4 and j in CORE_EXTREME_AGGS:
                                boost *= 1.3
                            new_affinity = new_affinity.at[i, j].set(
                                min(1.0, current + boost)
                            )

        return new_affinity, reinforcement_events

    # === MECHANISM 3: CAPTURE (requires tag + affinity) ===
    def _attempt_capture_combined(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Capture with combined tag + affinity requirements."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0
        cross_capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0, 0

        # Standard tagging capture
        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

        # CROSS-CAPTURE: Requires BOTH tag AND high affinity
        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    for j in range(NUM_AGGREGATIONS):
                        act_tagged = hist_act_tags[i] > self.tag_threshold * 0.8
                        agg_tagged = hist_agg_tags[j] > self.agg_tag_threshold * 0.8
                        high_affinity = cross_affinity[i, j] > self.capture_affinity_threshold

                        if act_tagged and agg_tagged and high_affinity:
                            if new_act_captured[i] < 0.5:
                                new_act_captured = new_act_captured.at[i].set(1.0)
                                cross_capture_count += 1
                            if new_agg_captured[j] < 0.5:
                                new_agg_captured = new_agg_captured.at[j].set(1.0)
                                cross_capture_count += 1

        return new_act_captured, new_agg_captured, capture_count, cross_capture_count

    # === MECHANISM 4: HOMEOSTATIC BALANCE (from 82) ===
    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute current extreme/averaging ratio."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    # === TRIPLE PROTECTION CHECK ===
    def _check_triple_protection(
        self,
        idx: int,
        is_act: bool,
        captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> bool:
        """Check if a function qualifies for triple protection."""
        is_captured = captured[idx] > 0.5

        if is_act:
            max_affinity = float(jnp.max(cross_affinity[idx, :]))
            # For sin, check affinity with extreme aggs
            if idx == 4:
                max_affinity = max(float(cross_affinity[4, 2]), float(cross_affinity[4, 3]))
        else:
            max_affinity = float(jnp.max(cross_affinity[:, idx]))

        high_affinity = max_affinity > self.triple_protection_threshold

        # Special case for sin-extreme pairs
        is_critical = (is_act and idx == 4) or (not is_act and idx in CORE_EXTREME_AGGS)

        # Triple protection: captured + high affinity + critical
        return is_captured and high_affinity and is_critical

    # === COMBINED MUTATION ===
    def _mutate_act_palette_triple(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Mutate activation palette with all three protection mechanisms."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        triple_protection_count = 0

        for i in range(NUM_ACTIVATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            max_affinity = float(jnp.max(cross_affinity[i, :]))
            has_triple_protection = self._check_triple_protection(i, True, captured, cross_affinity, mask)

            if mask[i] < 0.5:  # Inactive
                activate_rate = self.base_activate_rate
                # Discovery bonus for sin (homeostatic)
                if i == 4:
                    activate_rate += self.discovery_bonus
                # Affinity boost (reinforcement)
                if max_affinity > self.affinity_protection_threshold:
                    activate_rate *= 1.5
                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                deactivate_rate = self.base_deactivate_rate

                # MECHANISM 1: Tag protection
                if captured[i] > 0.5:
                    deactivate_rate *= (1 - self.captured_protection)

                # MECHANISM 2: Homeostatic sin protection
                if i == 4:
                    deactivate_rate *= (1 - self.sin_protection)

                # MECHANISM 3: Affinity protection (reinforcement)
                if max_affinity > self.affinity_protection_threshold:
                    deactivate_rate *= (1 - self.affinity_protection_strength)

                # TRIPLE PROTECTION: All three mechanisms agree
                if has_triple_protection:
                    deactivate_rate *= (1 - self.triple_protection_strength)
                    triple_protection_count += 1

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_act or active_count > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}, triple_protection_count

    def _mutate_agg_palette_triple(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        extreme_ratio: float,
    ) -> Tuple[jnp.ndarray, Dict, int, int, int]:
        """Mutate aggregation palette with all three protection mechanisms."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        homeostatic_corrections = 0
        discovery_bonuses = 0
        triple_protection_count = 0

        needs_more_extreme = extreme_ratio < self.target_extreme_ratio - self.imbalance_threshold
        needs_more_averaging = extreme_ratio > self.target_extreme_ratio + self.imbalance_threshold

        for j in range(NUM_AGGREGATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            sin_affinity = float(cross_affinity[4, j]) if act_mask[4] > 0.5 else 0.5
            max_affinity = float(jnp.max(cross_affinity[:, j]))
            is_extreme = j in EXTREME_AGGS
            is_core_extreme = j in CORE_EXTREME_AGGS
            has_triple_protection = self._check_triple_protection(j, False, captured, cross_affinity, mask)

            if mask[j] < 0.5:  # Inactive
                activate_rate = self.base_activate_rate

                # HOMEOSTATIC: Correction for balance
                if needs_more_extreme and is_core_extreme:
                    activate_rate *= self.correction_strength
                    homeostatic_corrections += 1
                elif needs_more_averaging and not is_extreme:
                    activate_rate *= self.correction_strength
                    homeostatic_corrections += 1

                # HOMEOSTATIC: Discovery bonus
                if is_core_extreme and extreme_ratio < 0.5:
                    activate_rate += self.discovery_bonus
                    discovery_bonuses += 1

                # REINFORCEMENT: Affinity boost
                if max_affinity > self.affinity_protection_threshold:
                    activate_rate *= 1.5

                if p < activate_rate:
                    new_mask = new_mask.at[j].set(1.0)
                    activated.append(j)
            else:  # Active
                deactivate_rate = self.base_deactivate_rate

                # MECHANISM 1: Tag protection
                if captured[j] > 0.5:
                    deactivate_rate *= (1 - self.captured_protection)

                # MECHANISM 2: Homeostatic extreme protection
                if is_core_extreme:
                    deactivate_rate *= (1 - self.extreme_protection)

                # MECHANISM 2: Homeostatic balance protection
                if needs_more_extreme and is_extreme:
                    deactivate_rate *= 0.5
                elif needs_more_averaging and not is_extreme:
                    deactivate_rate *= 0.5

                # MECHANISM 3: Affinity protection (reinforcement)
                if max_affinity > self.affinity_protection_threshold:
                    deactivate_rate *= (1 - self.affinity_protection_strength)

                # TRIPLE PROTECTION: All three mechanisms agree
                if has_triple_protection:
                    deactivate_rate *= (1 - self.triple_protection_strength)
                    triple_protection_count += 1

                if p < deactivate_rate:
                    new_mask = new_mask.at[j].set(0.0)
                    deactivated.append(j)

        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}, homeostatic_corrections, discovery_bonuses, triple_protection_count

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with all three mechanisms in priority order."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === PRIORITY 1: TAGGING (every generation) ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags']
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === PRIORITY 2: REINFORCEMENT (affinity updates) ===
        new_cross_affinity, reinforcement_events = self._update_cross_affinity_reinforced(
            state['cross_affinity'],
            state['act_mask'], state['agg_mask'],
            state['prev_act_mask'], state['prev_agg_mask'],
            improvement
        )

        # === PRIORITY 3: CAPTURE (on improvement) ===
        new_act_captured, new_agg_captured, capture_count, cross_capture_count = self._attempt_capture_combined(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            new_cross_affinity, new_tag_history, generation, improved
        )

        # === PRIORITY 4: HOMEOSTATIC BALANCE ===
        extreme_ratio = self._compute_extreme_ratio(state['agg_mask'])

        # === PRIORITY 5: MUTATIONS (with triple protection) ===
        should_mutate = new_stagnation >= self.stagnation_threshold
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        total_homeostatic = 0
        total_discovery = 0
        total_triple_protection = 0

        if should_mutate:
            new_act_mask, _, triple_prot_act = self._mutate_act_palette_triple(
                k_act, state['act_mask'], new_act_captured, new_cross_affinity
            )
            new_agg_mask, _, homeostatic_corrections, discovery_bonuses, triple_prot_agg = self._mutate_agg_palette_triple(
                k_agg, state['agg_mask'], new_agg_captured, new_cross_affinity,
                state['act_mask'], extreme_ratio
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            total_homeostatic = homeostatic_corrections
            total_discovery = discovery_bonuses
            total_triple_protection = triple_prot_act + triple_prot_agg
            new_stagnation = 0

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'prev_act_mask': state['act_mask'],
            'prev_agg_mask': state['agg_mask'],
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            'capture_events': state['capture_events'] + capture_count,
            'cross_capture_events': state['cross_capture_events'] + cross_capture_count,
            'homeostatic_corrections': state['homeostatic_corrections'] + total_homeostatic,
            'discovery_bonuses_applied': state['discovery_bonuses_applied'] + total_discovery,
            'reinforcement_events': state['reinforcement_events'] + reinforcement_events,
            'triple_protection_events': state['triple_protection_events'] + total_triple_protection,
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
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'max_tag': float(new_agg_tags[2]),
            'min_tag': float(new_agg_tags[3]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': state['capture_events'] + capture_count,
            'cross_capture_events': state['cross_capture_events'] + cross_capture_count,
            # Homeostatic metrics
            'extreme_ratio': extreme_ratio,
            'homeostatic_corrections': state['homeostatic_corrections'] + total_homeostatic,
            'discovery_bonuses_applied': state['discovery_bonuses_applied'] + total_discovery,
            # Reinforcement metrics
            'reinforcement_events': state['reinforcement_events'] + reinforcement_events,
            'sin_max_affinity': float(new_cross_affinity[4, 2]),
            'sin_min_affinity': float(new_cross_affinity[4, 3]),
            # Triple protection
            'triple_protection_events': state['triple_protection_events'] + total_triple_protection,
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with all mechanism status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Tagging
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'capture_events': state['capture_events'],
            'cross_capture_events': state['cross_capture_events'],
            # Homeostatic
            'homeostatic_corrections': state['homeostatic_corrections'],
            'discovery_bonuses_applied': state['discovery_bonuses_applied'],
            # Reinforcement
            'reinforcement_events': state['reinforcement_events'],
            # Triple protection
            'triple_protection_events': state['triple_protection_events'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
