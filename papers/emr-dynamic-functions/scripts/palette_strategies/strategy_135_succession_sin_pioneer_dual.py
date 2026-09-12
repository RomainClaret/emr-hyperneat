"""Strategy 135: Succession-Sin-Pioneer Dual.

Fixes the 0% sin retention bug in succession strategies by giving sin
automatic pioneer status with founder protection.

Key Fix:
- Sin has AUTOMATIC pioneer status from generation 0
- Sin has very high founder protection (95%)
- Extreme aggregations also get pioneer bonus
- Lower threshold for sin to become a founder

succession_immune_pioneer_dual treated all
functions equally - sin had to earn pioneer status like others. Evolution
found non-sin solutions before sin could become a founder.

Solution: Sin starts as a GUARANTEED pioneer with path to founder status.
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
    CORE_EXTREME_AGGS,
)


class SuccessionSinPioneerDualStrategy(PaletteEvolutionStrategy):
    """Ecological succession with sin as guaranteed pioneer.

    Ensures sin retention by ensuring sin starts
    as a pioneer with very high protection and easier path to founder.
    """

    name = "succession_sin_pioneer_dual"
    description = "Dual: Ecological succession with sin as guaranteed pioneer"

    def __init__(
        self,
        # === SIN PIONEER FIX (CRITICAL) ===
        sin_pioneer_status: bool = True,
        sin_founder_protection: float = 0.95,
        extreme_agg_pioneer_bonus: float = 0.3,
        sin_founder_threshold: float = 0.4,
        regular_founder_threshold: float = 0.6,
        # === SUCCESSION PHASES ===
        pioneer_phase_until: int = 15,
        intermediate_phase_until: int = 40,
        pioneer_mutation_rate: float = 0.25,
        climax_protection: float = 0.9,
        # === IMMUNE MEMORY ===
        memory_capacity: int = 5,
        memory_lifespan: int = 50,
        founder_memory_bonus: float = 0.5,
        # === AFFINITY ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_learning_rate: float = 0.12,
        sin_extreme_coupling: float = 0.5,
        # === TAGGING ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.40,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        extreme_tag_boost: float = 1.5,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 3,
        min_diversity_agg: int = 2,
        # === INITIAL ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Succession-Sin-Pioneer strategy."""
        # SIN PIONEER (CRITICAL)
        self.sin_pioneer_status = sin_pioneer_status
        self.sin_founder_protection = sin_founder_protection
        self.extreme_agg_pioneer_bonus = extreme_agg_pioneer_bonus
        self.sin_founder_threshold = sin_founder_threshold
        self.regular_founder_threshold = regular_founder_threshold

        # Succession
        self.pioneer_phase_until = pioneer_phase_until
        self.intermediate_phase_until = intermediate_phase_until
        self.pioneer_mutation_rate = pioneer_mutation_rate
        self.climax_protection = climax_protection

        # Immune memory
        self.memory_capacity = memory_capacity
        self.memory_lifespan = memory_lifespan
        self.founder_memory_bonus = founder_memory_bonus

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_coupling = sin_extreme_coupling

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.extreme_tag_boost = extreme_tag_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial - include sin and extreme aggs
        default_act = list(DEFAULT_PALETTE_INDICES)
        if 4 not in default_act:
            default_act.append(4)
        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        for agg in CORE_EXTREME_AGGS:
            if agg not in default_agg:
                default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with sin as pioneer."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        initial_act = list(initial_act)
        if 4 not in initial_act:
            initial_act.append(4)

        initial_agg = list(initial_agg)
        for agg in CORE_EXTREME_AGGS:
            if agg not in initial_agg:
                initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        act_affinities = act_affinities.at[4].set(0.8)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(0.75)

        # Tagging
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, agg].set(0.7)

        # CRITICAL: Sin as pioneer
        act_pioneers: Set[int] = {4} if self.sin_pioneer_status else set()
        act_founders: Set[int] = set()
        agg_pioneers: Set[int] = set(CORE_EXTREME_AGGS)
        agg_founders: Set[int] = set()

        # Contribution tracking for founder promotion
        act_contribution = jnp.zeros(NUM_ACTIVATIONS)
        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        act_contribution = act_contribution.at[4].set(0.3)  # Sin starts with credit
        for agg in CORE_EXTREME_AGGS:
            agg_contribution = agg_contribution.at[agg].set(0.25)

        # Immune memory
        memory: List[Tuple[int, int, int]] = []  # (type, idx, birth_gen)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            'cross_affinity': cross_affinity,
            # Succession state
            'act_pioneers': act_pioneers,
            'act_founders': act_founders,
            'agg_pioneers': agg_pioneers,
            'agg_founders': agg_founders,
            'act_contribution': act_contribution,
            'agg_contribution': agg_contribution,
            # Immune memory
            'memory': memory,
            # Stats
            'capture_events': 0,
            'pioneer_to_founder': 0,
            'rng_key': jax.random.PRNGKey(seed + 1350000),
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

    def _get_phase(self, generation: int) -> str:
        if generation < self.pioneer_phase_until:
            return 'pioneer'
        elif generation < self.intermediate_phase_until:
            return 'intermediate'
        else:
            return 'climax'

    def _update_contributions(
        self,
        act_contribution: jnp.ndarray,
        agg_contribution: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update contribution tracking."""
        new_act = act_contribution * 0.95
        new_agg = agg_contribution * 0.95

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                boost = 0.05 if improved else 0.01
                if i == 4:  # Sin gets extra
                    boost *= 1.5
                new_act = new_act.at[i].set(min(1.0, new_act[i] + boost))

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                boost = 0.05 if improved else 0.01
                if j in CORE_EXTREME_AGGS:
                    boost *= 1.3
                new_agg = new_agg.at[j].set(min(1.0, new_agg[j] + boost))

        return new_act, new_agg

    def _promote_pioneers(
        self,
        act_pioneers: Set[int],
        act_founders: Set[int],
        agg_pioneers: Set[int],
        agg_founders: Set[int],
        act_contribution: jnp.ndarray,
        agg_contribution: jnp.ndarray,
        memory: List[Tuple],
        generation: int,
    ) -> Tuple[Set[int], Set[int], Set[int], Set[int], List[Tuple], int]:
        """Promote pioneers to founders based on contribution."""
        new_act_pioneers = set(act_pioneers)
        new_act_founders = set(act_founders)
        new_agg_pioneers = set(agg_pioneers)
        new_agg_founders = set(agg_founders)
        new_memory = list(memory)
        promotions = 0

        # Activation pioneers
        for i in list(new_act_pioneers):
            threshold = self.sin_founder_threshold if i == 4 else self.regular_founder_threshold
            if act_contribution[i] > threshold:
                new_act_pioneers.discard(i)
                new_act_founders.add(i)
                # Add to immune memory
                if len(new_memory) < self.memory_capacity:
                    new_memory.append(('act', i, generation))
                promotions += 1

        # Aggregation pioneers
        for j in list(new_agg_pioneers):
            threshold = self.regular_founder_threshold - (
                self.extreme_agg_pioneer_bonus if j in CORE_EXTREME_AGGS else 0
            )
            if agg_contribution[j] > threshold:
                new_agg_pioneers.discard(j)
                new_agg_founders.add(j)
                if len(new_memory) < self.memory_capacity:
                    new_memory.append(('agg', j, generation))
                promotions += 1

        # Expire old memories
        new_memory = [
            (t, idx, gen) for t, idx, gen in new_memory
            if generation - gen < self.memory_lifespan
        ]

        return (new_act_pioneers, new_act_founders, new_agg_pioneers,
                new_agg_founders, new_memory, promotions)

    def _get_protection(
        self,
        idx: int,
        domain: str,
        state: Dict[str, Any],
        phase: str,
    ) -> float:
        """Get protection level for a function."""
        if domain == 'act':
            if idx == 4:  # Sin ALWAYS high protection
                return self.sin_founder_protection
            if idx in state['act_founders']:
                return self.climax_protection if phase == 'climax' else 0.7
            if idx in state['act_pioneers']:
                return 0.5
        else:
            if idx in CORE_EXTREME_AGGS:
                return 0.85
            if idx in state['agg_founders']:
                return self.climax_protection if phase == 'climax' else 0.7
            if idx in state['agg_pioneers']:
                return 0.5 + self.extreme_agg_pioneer_bonus if idx in CORE_EXTREME_AGGS else 0.5
        return 0.0

    def _select_palettes(
        self,
        state: Dict[str, Any],
        phase: str,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select palettes with succession-based protection."""
        k1, k2 = jax.random.split(key)

        # Activation selection
        act_score = (
            state['act_affinities'] +
            state['act_captured'] * 0.3 +
            state['act_tags'] * 0.2 +
            state['act_contribution'] * 0.3
        )

        # Add protection bonus
        for i in range(NUM_ACTIVATIONS):
            prot = self._get_protection(i, 'act', state, phase)
            act_score = act_score.at[i].set(act_score[i] + prot * 0.5)

        # Sin gets extra
        act_score = act_score.at[4].set(act_score[4] + 1.0)

        target_act = min(self.max_active_act, max(self.min_diversity_act, self.min_active_act))
        top_act = jnp.argsort(act_score)[-target_act:]
        act_mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_act:
            act_mask = act_mask.at[int(idx)].set(1.0)
        # ALWAYS include sin
        act_mask = act_mask.at[4].set(1.0)

        # Aggregation selection
        agg_score = (
            state['agg_affinities'] +
            state['agg_captured'] * 0.3 +
            state['agg_tags'] * 0.2 +
            state['agg_contribution'] * 0.3
        )

        for j in range(NUM_AGGREGATIONS):
            prot = self._get_protection(j, 'agg', state, phase)
            agg_score = agg_score.at[j].set(agg_score[j] + prot * 0.5)

        for agg in CORE_EXTREME_AGGS:
            agg_score = agg_score.at[agg].set(agg_score[agg] + 0.5)

        target_agg = min(self.max_active_agg, max(self.min_diversity_agg, self.min_active_agg))
        top_agg = jnp.argsort(agg_score)[-target_agg:]
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for idx in top_agg:
            agg_mask = agg_mask.at[int(idx)].set(1.0)
        # Always include at least one extreme
        if not any(agg_mask[agg] > 0.5 for agg in CORE_EXTREME_AGGS):
            agg_mask = agg_mask.at[2].set(1.0)

        return act_mask, agg_mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with succession dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        phase = self._get_phase(generation)

        # Update contributions
        new_act_contrib, new_agg_contrib = self._update_contributions(
            state['act_contribution'], state['agg_contribution'],
            state['act_mask'], state['agg_mask'], improved
        )

        # Promote pioneers to founders
        (new_act_pioneers, new_act_founders, new_agg_pioneers,
         new_agg_founders, new_memory, promotions) = self._promote_pioneers(
            state['act_pioneers'], state['act_founders'],
            state['agg_pioneers'], state['agg_founders'],
            new_act_contrib, new_agg_contrib,
            state['memory'], generation
        )

        # Update tags
        new_act_tags = state['act_tags'] * self.tag_decay
        new_agg_tags = state['agg_tags'] * self.tag_decay

        for j in range(NUM_AGGREGATIONS):
            if state['agg_mask'][j] > 0.5:
                boost = self.extreme_tag_boost if j in CORE_EXTREME_AGGS else 1.0
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + 0.35 * boost)
                )

        for i in range(NUM_ACTIVATIONS):
            if state['act_mask'][i] > 0.5:
                boost = self.extreme_tag_boost if i == 4 else 1.0
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + 0.3 * boost)
                )

        # Update affinities
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay

        if fitness_delta > 0:
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    bonus = 1.6 if j in CORE_EXTREME_AGGS else 1.0
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta * bonus)
                    )
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    bonus = 1.5 if i == 4 else 1.0
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * fitness_delta * bonus)
                    )

        # Affinity floors
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Capture
        new_act_captured = state['act_captured'].copy()
        new_agg_captured = state['agg_captured'].copy()
        capture_count = 0

        if improved:
            for j in range(NUM_AGGREGATIONS):
                if state['agg_tags'][j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                    new_agg_captured = new_agg_captured.at[j].set(1.0)
                    capture_count += 1
            for i in range(NUM_ACTIVATIONS):
                if state['act_tags'][i] > self.tag_threshold and new_act_captured[i] < 0.5:
                    new_act_captured = new_act_captured.at[i].set(1.0)
                    capture_count += 1

        # Update cross affinity
        new_cross = state['cross_affinity'].copy()
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if state['act_mask'][i] > 0.5 and state['agg_mask'][j] > 0.5:
                        delta = self.cross_learning_rate * fitness_delta
                        if i == 4 and j in CORE_EXTREME_AGGS:
                            delta *= (1 + self.sin_extreme_coupling)
                        new_cross = new_cross.at[i, j].set(min(1.0, new_cross[i, j] + delta))

        # Build state for selection
        temp_state = {
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_contribution': new_act_contrib,
            'agg_contribution': new_agg_contrib,
            'act_pioneers': new_act_pioneers,
            'act_founders': new_act_founders,
            'agg_pioneers': new_agg_pioneers,
            'agg_founders': new_agg_founders,
        }

        new_act_mask, new_agg_mask = self._select_palettes(temp_state, phase, k1)

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': state['tag_history'],
            'cross_affinity': new_cross,
            'act_pioneers': new_act_pioneers,
            'act_founders': new_act_founders,
            'agg_pioneers': new_agg_pioneers,
            'agg_founders': new_agg_founders,
            'act_contribution': new_act_contrib,
            'agg_contribution': new_agg_contrib,
            'memory': new_memory,
            'capture_events': state['capture_events'] + capture_count,
            'pioneer_to_founder': state['pioneer_to_founder'] + promotions,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'n_act_pioneers': len(new_act_pioneers),
            'n_act_founders': len(new_act_founders),
            'n_agg_pioneers': len(new_agg_pioneers),
            'n_agg_founders': len(new_agg_founders),
            'sin_is_founder': 4 in new_act_founders,
            'sin_contribution': float(new_act_contrib[4]),
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'sin_is_founder': 4 in state['act_founders'],
            'n_founders': len(state['act_founders']) + len(state['agg_founders']),
            'generation': state['generation'],
        }
