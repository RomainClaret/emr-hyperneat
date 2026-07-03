"""Strategy 136: Succession-Sin-Extreme-Founders Dual.

Sin and extreme aggregations form a "founding coalition" with mutual protection.

Key Innovation:
- Sin and max/min form a founding coalition from generation 0
- Coalition members get 1.5x protection multiplier
- All coalition members must be removed together (or none)
- Slight fitness bonus when all coalition members are active

This creates an "all-or-nothing" dynamic where evolution can't easily
remove sin without also losing the beneficial extreme aggregations.
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


class SuccessionSinExtremeFoundersDualStrategy(PaletteEvolutionStrategy):
    """Ecological succession with sin-extreme founding coalition.

    Sin and extreme aggs form a protected coalition
    that's difficult to break apart.
    """

    name = "succession_sin_extreme_founders_dual"
    description = "Dual: Sin and extreme aggs as founding coalition"

    def __init__(
        self,
        # === COALITION PARAMETERS (CRITICAL) ===
        sin_extreme_founding_coalition: bool = True,
        coalition_members_act: List[int] = None,  # [4] = sin
        coalition_members_agg: List[int] = None,  # [2, 3] = max, min
        coalition_protection_multiplier: float = 1.5,
        coalition_fitness_bonus: float = 0.1,
        coalition_break_penalty: float = 0.3,
        # === SUCCESSION PHASES ===
        pioneer_phase_until: int = 15,
        intermediate_phase_until: int = 40,
        climax_protection: float = 0.9,
        # === AFFINITY ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_learning_rate: float = 0.12,
        sin_extreme_coupling: float = 0.6,
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
        """Initialize Succession-Sin-Extreme-Founders strategy."""
        # COALITION (CRITICAL)
        self.sin_extreme_founding_coalition = sin_extreme_founding_coalition
        self.coalition_members_act = coalition_members_act or [4]
        self.coalition_members_agg = coalition_members_agg or list(CORE_EXTREME_AGGS)
        self.coalition_protection_multiplier = coalition_protection_multiplier
        self.coalition_fitness_bonus = coalition_fitness_bonus
        self.coalition_break_penalty = coalition_break_penalty

        # Succession
        self.pioneer_phase_until = pioneer_phase_until
        self.intermediate_phase_until = intermediate_phase_until
        self.climax_protection = climax_protection

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

        # Initial - include coalition members
        default_act = list(DEFAULT_PALETTE_INDICES)
        for m in self.coalition_members_act:
            if m not in default_act:
                default_act.append(m)

        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        for m in self.coalition_members_agg:
            if m not in default_agg:
                default_agg.append(m)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with coalition."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        initial_act = list(initial_act)
        for m in self.coalition_members_act:
            if m not in initial_act:
                initial_act.append(m)

        initial_agg = list(initial_agg)
        for m in self.coalition_members_agg:
            if m not in initial_agg:
                initial_agg.append(m)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities - coalition members start high
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)

        for m in self.coalition_members_act:
            act_affinities = act_affinities.at[m].set(0.85)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)

        for m in self.coalition_members_agg:
            agg_affinities = agg_affinities.at[m].set(0.8)

        # Tagging
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Coalition members are pre-captured
        for m in self.coalition_members_act:
            act_captured = act_captured.at[m].set(1.0)
        for m in self.coalition_members_agg:
            agg_captured = agg_captured.at[m].set(1.0)

        # Cross-domain - strong sin-extreme coupling
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for act_m in self.coalition_members_act:
            for agg_m in self.coalition_members_agg:
                cross_affinity = cross_affinity.at[act_m, agg_m].set(0.85)

        # Coalition integrity tracking
        coalition_intact = True
        coalition_strength = 1.0

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
            'coalition_intact': coalition_intact,
            'coalition_strength': coalition_strength,
            'capture_events': 0,
            'rng_key': jax.random.PRNGKey(seed + 1360000),
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

    def _check_coalition_integrity(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> Tuple[bool, float]:
        """Check if coalition is intact and compute strength."""
        act_present = sum(1 for m in self.coalition_members_act if act_mask[m] > 0.5)
        agg_present = sum(1 for m in self.coalition_members_agg if agg_mask[m] > 0.5)

        total_members = len(self.coalition_members_act) + len(self.coalition_members_agg)
        present = act_present + agg_present

        intact = (present == total_members)
        strength = present / total_members

        return intact, strength

    def _get_coalition_protection(
        self,
        idx: int,
        domain: str,
        coalition_strength: float,
        phase: str,
    ) -> float:
        """Get protection level for coalition member."""
        is_coalition = (
            (domain == 'act' and idx in self.coalition_members_act) or
            (domain == 'agg' and idx in self.coalition_members_agg)
        )

        if not is_coalition:
            return 0.0

        base_protection = 0.8 if phase == 'climax' else 0.6
        coalition_bonus = coalition_strength * self.coalition_protection_multiplier * 0.2

        return min(0.95, base_protection + coalition_bonus)

    def _select_palettes(
        self,
        state: Dict[str, Any],
        phase: str,
        coalition_strength: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select palettes with coalition protection."""
        k1, k2 = jax.random.split(key)

        # Activation selection
        act_score = (
            state['act_affinities'] +
            state['act_captured'] * 0.3 +
            state['act_tags'] * 0.2
        )

        # Coalition protection
        for i in range(NUM_ACTIVATIONS):
            prot = self._get_coalition_protection(i, 'act', coalition_strength, phase)
            act_score = act_score.at[i].set(act_score[i] + prot * 0.6)

        target_act = min(self.max_active_act, max(self.min_diversity_act, self.min_active_act))
        top_act = jnp.argsort(act_score)[-target_act:]
        act_mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_act:
            act_mask = act_mask.at[int(idx)].set(1.0)

        # ALWAYS include coalition act members
        for m in self.coalition_members_act:
            act_mask = act_mask.at[m].set(1.0)

        # Aggregation selection
        agg_score = (
            state['agg_affinities'] +
            state['agg_captured'] * 0.3 +
            state['agg_tags'] * 0.2
        )

        for j in range(NUM_AGGREGATIONS):
            prot = self._get_coalition_protection(j, 'agg', coalition_strength, phase)
            agg_score = agg_score.at[j].set(agg_score[j] + prot * 0.6)

        target_agg = min(self.max_active_agg, max(self.min_diversity_agg, self.min_active_agg))
        top_agg = jnp.argsort(agg_score)[-target_agg:]
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for idx in top_agg:
            agg_mask = agg_mask.at[int(idx)].set(1.0)

        # ALWAYS include coalition agg members
        for m in self.coalition_members_agg:
            agg_mask = agg_mask.at[m].set(1.0)

        return act_mask, agg_mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with coalition dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        phase = self._get_phase(generation)

        # Check coalition integrity
        coalition_intact, coalition_strength = self._check_coalition_integrity(
            state['act_mask'], state['agg_mask']
        )

        # Update tags
        new_act_tags = state['act_tags'] * self.tag_decay
        new_agg_tags = state['agg_tags'] * self.tag_decay

        for j in range(NUM_AGGREGATIONS):
            if state['agg_mask'][j] > 0.5:
                boost = self.extreme_tag_boost if j in self.coalition_members_agg else 1.0
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + 0.35 * boost)
                )

        for i in range(NUM_ACTIVATIONS):
            if state['act_mask'][i] > 0.5:
                boost = self.extreme_tag_boost if i in self.coalition_members_act else 1.0
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + 0.3 * boost)
                )

        # Update affinities
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay

        if fitness_delta > 0:
            # Coalition bonus when all active
            bonus_mult = 1.0 + (self.coalition_fitness_bonus if coalition_intact else 0)

            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    coalition_boost = 1.6 if j in self.coalition_members_agg else 1.0
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta * coalition_boost * bonus_mult)
                    )
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    coalition_boost = 1.5 if i in self.coalition_members_act else 1.0
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * fitness_delta * coalition_boost * bonus_mult)
                    )

        # Affinity floors for coalition members
        for m in self.coalition_members_act:
            new_act_aff = new_act_aff.at[m].set(max(0.6, float(new_act_aff[m])))
        for m in self.coalition_members_agg:
            new_agg_aff = new_agg_aff.at[m].set(max(0.55, float(new_agg_aff[m])))

        # Update cross affinity with strong coalition coupling
        new_cross = state['cross_affinity'].copy()
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if state['act_mask'][i] > 0.5 and state['agg_mask'][j] > 0.5:
                        delta = self.cross_learning_rate * fitness_delta
                        # Strong coupling for coalition pairs
                        if i in self.coalition_members_act and j in self.coalition_members_agg:
                            delta *= (1 + self.sin_extreme_coupling)
                        new_cross = new_cross.at[i, j].set(min(1.0, new_cross[i, j] + delta))

        # Build state for selection
        temp_state = {
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_captured': state['act_captured'],
            'agg_captured': state['agg_captured'],
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
        }

        new_act_mask, new_agg_mask = self._select_palettes(
            temp_state, phase, coalition_strength, k1
        )

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': state['act_captured'],
            'agg_captured': state['agg_captured'],
            'tag_history': state['tag_history'],
            'cross_affinity': new_cross,
            'coalition_intact': coalition_intact,
            'coalition_strength': coalition_strength,
            'capture_events': state['capture_events'],
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
            'coalition_intact': coalition_intact,
            'coalition_strength': coalition_strength,
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
            'coalition_intact': state['coalition_intact'],
            'coalition_strength': state['coalition_strength'],
            'generation': state['generation'],
        }
