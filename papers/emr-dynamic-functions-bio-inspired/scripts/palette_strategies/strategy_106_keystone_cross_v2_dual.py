"""Strategy 106: Keystone + Cross-Domain v2 Dual.

Combines keystone ecology (strategy 46) with multiplicative cross-domain
coupling (strategy 103) for robust sin-extreme pairings.

Key Innovation:
- Keystone functions gain AMPLIFIED multiplicative coupling
- When sin achieves keystone status, its cross-affinity growth is boosted
- Extreme aggregations that become keystones receive extra protection
- Keystone-to-keystone pairings get multiplicative super-boost

Bio inspiration: In ecosystems, keystone species not only protect others
but also form especially strong mutualistic relationships. When two
keystone species interact, their relationship becomes foundational to
the entire ecosystem.

Expected: Sin-extreme pairings become near-permanent through combined
keystone protection and amplified multiplicative coupling.
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


class KeystoneCrossV2DualStrategy(PaletteEvolutionStrategy):
    """Keystone ecology with amplified multiplicative cross-domain coupling.

    Keystone status amplifies the multiplicative
    cross-domain dynamics, creating super-robust pairings.

    Critical innovation: Keystone × keystone pairings get multiplicative
    super-boost, creating near-permanent sin-extreme coupling.
    """

    name = "keystone_cross_v2_dual"
    description = "Dual: Keystone status amplifies multiplicative cross-domain coupling"

    def __init__(
        self,
        # === KEYSTONE-BOOSTED MULTIPLICATIVE PARAMETERS (KEY INNOVATION) ===
        base_cross_growth_factor: float = 1.15,       # Base multiplicative growth
        keystone_cross_boost: float = 1.25,           # Boosted when keystone active
        keystone_keystone_multiplier: float = 1.5,    # Keystone-keystone super-boost
        sin_keystone_priority: float = 0.7,           # Sin gets keystone faster
        # === Keystone parameters (from strategy 46) ===
        keystone_threshold: float = 0.5,              # Fitness to become keystone
        keystone_facilitation: float = 0.4,           # Keystone helps neighbors
        keystone_protection: float = 0.7,             # Keystone protection strength
        keystone_decay: float = 0.95,                 # Keystone status decay
        # === Multiplicative cross-domain (from strategy 103) ===
        cross_affinity_decay_factor: float = 0.98,
        mutual_capture_bonus: float = 0.4,
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
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
        """Initialize Keystone + Cross-Domain v2 strategy."""
        # Keystone-boosted multiplicative (KEY INNOVATION)
        self.base_cross_growth_factor = base_cross_growth_factor
        self.keystone_cross_boost = keystone_cross_boost
        self.keystone_keystone_multiplier = keystone_keystone_multiplier
        self.sin_keystone_priority = sin_keystone_priority

        # Keystone
        self.keystone_threshold = keystone_threshold
        self.keystone_facilitation = keystone_facilitation
        self.keystone_protection = keystone_protection
        self.keystone_decay = keystone_decay

        # Multiplicative cross-domain
        self.cross_affinity_decay_factor = cross_affinity_decay_factor
        self.mutual_capture_bonus = mutual_capture_bonus

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

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
        """Initialize state with keystone + cross-domain tracking."""
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

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Keystone status (from strategy 46)
        act_keystone = jnp.zeros(NUM_ACTIVATIONS)
        agg_keystone = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for j in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, j].set(0.6)

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
            # Keystone status (KEY)
            'act_keystone': act_keystone,
            'agg_keystone': agg_keystone,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'capture_events': 0,
            'keystone_boost_events': 0,
            'keystone_keystone_events': 0,
            'diversity_rescues': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1060000),
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

    def _update_keystone_status(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_keystone: jnp.ndarray,
        agg_keystone: jnp.ndarray,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update keystone status based on fitness contribution."""
        new_act_keystone = act_keystone * self.keystone_decay
        new_agg_keystone = agg_keystone * self.keystone_decay

        if improved:
            # Active functions with high affinity become keystones
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5 and act_affinities[i] > self.keystone_threshold:
                    keystone_strength = act_affinities[i]
                    # Sin gets priority for keystone status
                    if i == 4:
                        keystone_strength *= (1 + self.sin_keystone_priority)
                    new_act_keystone = new_act_keystone.at[i].set(
                        min(1.0, new_act_keystone[i] + keystone_strength * 0.3)
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5 and agg_affinities[j] > self.keystone_threshold:
                    keystone_strength = agg_affinities[j]
                    # Extremes get priority
                    if j in CORE_EXTREME_AGGS:
                        keystone_strength *= 1.3
                    new_agg_keystone = new_agg_keystone.at[j].set(
                        min(1.0, new_agg_keystone[j] + keystone_strength * 0.3)
                    )

        return new_act_keystone, new_agg_keystone

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_keystone: jnp.ndarray,
        agg_keystone: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with keystone facilitation boost."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:
                    tag_strength *= self.extreme_tag_boost
                # Keystone facilitation: keystones boost neighbors
                keystone_boost = float(act_keystone[i]) * self.keystone_facilitation
                tag_strength *= (1 + keystone_boost)
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                keystone_boost = float(agg_keystone[j]) * self.keystone_facilitation
                tag_strength *= (1 + keystone_boost)
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
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Attempt capture for tagged functions."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

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

        return new_act_captured, new_agg_captured, capture_count

    def _update_affinities_keystone_boosted(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_keystone: jnp.ndarray,
        agg_keystone: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]:
        """Update affinities with KEYSTONE-BOOSTED multiplicative dynamics.

        KEY INNOVATION: Keystone status amplifies cross-affinity growth factor.
        Keystone-keystone pairings get multiplicative super-boost.
        """
        keystone_boost_events = 0
        keystone_keystone_events = 0

        # Decay individual affinities
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

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

        # KEYSTONE-BOOSTED MULTIPLICATIVE CROSS-DOMAIN UPDATE
        new_cross = cross_affinity.copy()

        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                current = float(new_cross[i, j])

                if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                    if fitness_delta > 0:
                        # Determine growth factor based on keystone status
                        growth_factor = self.base_cross_growth_factor

                        # Keystone boost: if either is keystone, boost growth
                        act_is_keystone = act_keystone[i] > self.keystone_threshold
                        agg_is_keystone = agg_keystone[j] > self.keystone_threshold

                        if act_is_keystone or agg_is_keystone:
                            growth_factor = self.keystone_cross_boost
                            keystone_boost_events += 1

                            # Keystone-keystone super-boost
                            if act_is_keystone and agg_is_keystone:
                                growth_factor *= self.keystone_keystone_multiplier
                                keystone_keystone_events += 1

                        new_value = current * growth_factor
                        new_cross = new_cross.at[i, j].set(min(1.0, new_value))
                    else:
                        # Keystones resist decay
                        decay_factor = self.cross_affinity_decay_factor
                        if act_keystone[i] > 0.3 or agg_keystone[j] > 0.3:
                            decay_factor = 0.995  # Slower decay for keystone pairs
                        new_value = max(0.2, current * decay_factor)
                        new_cross = new_cross.at[i, j].set(new_value)
                else:
                    # Inactive pairs decay
                    new_value = max(0.1, current * 0.995)
                    new_cross = new_cross.at[i, j].set(new_value)

        return new_act_aff, new_agg_aff, new_cross, keystone_boost_events, keystone_keystone_events

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        keystone: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        other_mask: jnp.ndarray,
        n_funcs: int,
        is_act: bool,
        min_active: int,
        max_active: int,
        min_diversity: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette with keystone protection."""
        # Score: affinity + capture + tag + keystone + cross-domain
        score = affinities + captured * 0.3 + tags * 0.2 + keystone * self.keystone_protection

        # Cross-domain influence
        for i in range(n_funcs):
            if is_act:
                cross_influence = 0.0
                for j in range(NUM_AGGREGATIONS):
                    if other_mask[j] > 0.5:
                        cross_influence = max(cross_influence, float(cross_affinity[i, j]))
            else:
                cross_influence = 0.0
                for i_act in range(NUM_ACTIVATIONS):
                    if other_mask[i_act] > 0.5:
                        cross_influence = max(cross_influence, float(cross_affinity[i_act, i]))
            score = score.at[i].set(score[i] + cross_influence * 0.25)

        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + 0.5)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < min_diversity:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive), shape=(min(needed, len(inactive)),), replace=False)
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
        """Update with keystone-boosted multiplicative dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === KEYSTONE UPDATE ===
        new_act_keystone, new_agg_keystone = self._update_keystone_status(
            state['act_mask'], state['agg_mask'],
            state['act_keystone'], state['agg_keystone'],
            state['act_affinities'], state['agg_affinities'],
            improved
        )

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags'],
            new_act_keystone, new_agg_keystone
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE ===
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            new_tag_history, generation, improved
        )

        # === KEYSTONE-BOOSTED MULTIPLICATIVE UPDATE ===
        (new_act_aff, new_agg_aff, new_cross_affinity,
         keystone_boost_events, keystone_keystone_events) = \
            self._update_affinities_keystone_boosted(
                state['act_affinities'], state['agg_affinities'],
                state['act_mask'], state['agg_mask'],
                new_act_keystone, new_agg_keystone,
                state['cross_affinity'], fitness_delta
            )

        # === PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_palette(
            new_act_aff, new_act_captured, new_act_tags, new_act_keystone,
            new_cross_affinity, state['agg_mask'],
            NUM_ACTIVATIONS, True,
            self.min_active_act, self.max_active_act, self.min_diversity_act,
            k1, prefer_indices=[4]
        )
        new_agg_mask, agg_diversity_rescue = self._select_palette(
            new_agg_aff, new_agg_captured, new_agg_tags, new_agg_keystone,
            new_cross_affinity, new_act_mask,
            NUM_AGGREGATIONS, False,
            self.min_active_agg, self.max_active_agg, self.min_diversity_agg,
            k2, prefer_indices=list(CORE_EXTREME_AGGS)
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
            'act_keystone': new_act_keystone,
            'agg_keystone': new_agg_keystone,
            'cross_affinity': new_cross_affinity,
            'capture_events': state['capture_events'] + capture_count,
            'keystone_boost_events': state['keystone_boost_events'] + keystone_boost_events,
            'keystone_keystone_events': state['keystone_keystone_events'] + keystone_keystone_events,
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
            # Keystone metrics (KEY)
            'sin_keystone': float(new_act_keystone[4]),
            'max_keystone': float(new_agg_keystone[2]),
            'min_keystone': float(new_agg_keystone[3]),
            'keystone_boost_events': new_state['keystone_boost_events'],
            'keystone_keystone_events': new_state['keystone_keystone_events'],
            # Cross-domain metrics
            'sin_max_cross': float(new_cross_affinity[4, 2]),
            'sin_min_cross': float(new_cross_affinity[4, 3]),
            'max_cross_affinity': float(jnp.max(new_cross_affinity)),
            'mean_cross_affinity': float(jnp.mean(new_cross_affinity)),
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
        """Return state summary with keystone + cross-domain status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Keystone status
            'sin_keystone': float(state['act_keystone'][4]),
            'max_keystone': float(state['agg_keystone'][2]),
            'min_keystone': float(state['agg_keystone'][3]),
            'keystone_boost_events': state['keystone_boost_events'],
            'keystone_keystone_events': state['keystone_keystone_events'],
            # Cross-domain
            'sin_max_cross': float(state['cross_affinity'][4, 2]),
            'max_cross_affinity': float(jnp.max(state['cross_affinity'])),
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
