"""Strategy 105: Keystone Aggregation-Led Dual.

Combines aggregation_led_dual (#101) with ecosystem_dynamics (#46) keystone
mechanics. Extreme aggregations become "keystone functions" that protect sin
through ecological facilitation networks.

Problem with pure aggregation-led:
- Extreme aggs discovered but no ecological protection mechanism
- Sin-extreme pairings rely only on cross-domain affinity
- No structural protection against domain-shift pressure

Key Innovation:
- Extreme aggregations (max/min) gain keystone status when stable
- Keystones create facilitation networks that protect sin
- Keystone-protected sin has enhanced capture resistance
- Ecological dynamics create redundant protection layer

Bio inspiration: Keystone species in ecosystems have disproportionate impact.
When max/min aggregations become keystones, they actively facilitate sin's
survival through ecological dependency networks.

Expected: Near-permanent sin retention through keystone protection.
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


class KeystoneAggLedDualStrategy(PaletteEvolutionStrategy):
    """Keystone-enhanced aggregation-led dual palette evolution.

    Combines aggressive aggregation exploration with
    keystone function dynamics. Extreme aggregations become keystones that
    create facilitation networks protecting sin.

    Critical innovation: Keystone status creates ecological protection
    beyond simple affinity - keystones actively prevent sin removal.
    """

    name = "keystone_agg_led_dual"
    description = "Dual: Aggregation-led with keystone facilitation for sin protection"

    def __init__(
        self,
        # === KEYSTONE PARAMETERS (NEW from ecosystem_dynamics) ===
        keystone_threshold: float = 0.5,          # Stability to become keystone
        keystone_facilitation: float = 0.4,       # Support to protected functions
        keystone_protection: float = 0.8,         # Protection strength
        sin_extreme_facilitation_boost: float = 0.6,  # Extra facilitation for sin
        keystone_stability_required: int = 5,     # Gens stable to become keystone
        # === AGGREGATION-LED PARAMETERS (from #101) ===
        agg_exploration_rate: float = 0.25,       # HIGH agg exploration
        agg_discovery_bonus: float = 0.6,         # Bonus for finding extreme aggs
        act_exploration_rate: float = 0.08,       # LOW act exploration
        agg_led_activation_boost: float = 0.4,    # Act boost when extreme aggs present
        agg_stability_threshold: int = 5,         # Gens before agg is "stable"
        # === Affinity parameters ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_learning_rate: float = 0.12,
        sin_extreme_coupling: float = 0.5,
        # === Tagging parameters ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.40,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.85,
        extreme_tag_boost: float = 1.5,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 3,
        min_diversity_agg: int = 2,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Keystone Aggregation-Led strategy."""
        # Keystone parameters (NEW)
        self.keystone_threshold = keystone_threshold
        self.keystone_facilitation = keystone_facilitation
        self.keystone_protection = keystone_protection
        self.sin_extreme_facilitation_boost = sin_extreme_facilitation_boost
        self.keystone_stability_required = keystone_stability_required

        # Aggregation-led parameters
        self.agg_exploration_rate = agg_exploration_rate
        self.agg_discovery_bonus = agg_discovery_bonus
        self.act_exploration_rate = act_exploration_rate
        self.agg_led_activation_boost = agg_led_activation_boost
        self.agg_stability_threshold = agg_stability_threshold

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
        """Initialize state with keystone tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)

        # Extra boost for extreme aggs
        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(
                agg_affinities[i] + self.agg_discovery_bonus * 0.3
            )

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Aggregation stability tracking
        agg_stability_counts = jnp.zeros(NUM_AGGREGATIONS)

        # Keystone tracking (NEW)
        keystone_aggs: Set[int] = set()
        facilitated_acts: Dict[int, int] = {}  # {act: keystone_agg protecting it}

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
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Aggregation-led tracking
            'agg_stability_counts': agg_stability_counts,
            'extreme_agg_discovered': False,
            'extreme_agg_discovery_gen': None,
            'activation_boost_active': False,
            # Keystone tracking (NEW)
            'keystone_aggs': keystone_aggs,
            'facilitated_acts': facilitated_acts,
            'keystone_history': [],
            # Stats
            'capture_events': 0,
            'agg_exploration_events': 0,
            'act_boosted_events': 0,
            'keystone_promotions': 0,
            'facilitation_protections': 0,
            'diversity_rescues': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1050000),
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
        agg_mask: jnp.ndarray,
        agg_stability_counts: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        current_keystones: Set[int],
    ) -> Tuple[Set[int], int]:
        """Update keystone status for aggregations.

        Extreme aggregations become keystones when:
        1. They are stable (active for keystone_stability_required gens)
        2. Their affinity exceeds keystone_threshold

        Returns: (new_keystones, n_new_promotions)
        """
        new_keystones = set(current_keystones)
        n_promotions = 0

        for j in CORE_EXTREME_AGGS:
            if agg_mask[j] > 0.5:
                is_stable = agg_stability_counts[j] >= self.keystone_stability_required
                has_affinity = agg_affinities[j] >= self.keystone_threshold

                if is_stable and has_affinity and j not in new_keystones:
                    new_keystones.add(j)
                    n_promotions += 1
            else:
                # Lost from palette - lose keystone status
                new_keystones.discard(j)

        return new_keystones, n_promotions

    def _compute_facilitation(
        self,
        keystone_aggs: Set[int],
        act_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
    ) -> Dict[int, int]:
        """Compute which activations are facilitated by keystone aggs.

        Keystones facilitate activations with high cross-affinity.
        Sin (index 4) gets extra facilitation from extreme agg keystones.
        """
        facilitated = {}

        for keystone in keystone_aggs:
            # Find activations with high cross-affinity to this keystone
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    cross_aff = float(cross_affinity[i, keystone])
                    threshold = self.keystone_threshold * 0.8

                    # Sin gets easier facilitation from extreme aggs
                    if i == 4 and keystone in CORE_EXTREME_AGGS:
                        threshold *= 0.5  # Much easier

                    if cross_aff >= threshold:
                        facilitated[i] = keystone

        return facilitated

    def _update_agg_stability(
        self,
        agg_mask: jnp.ndarray,
        agg_stability_counts: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, bool]:
        """Update stability counts for aggregations."""
        new_counts = agg_stability_counts.copy()

        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                new_counts = new_counts.at[i].set(new_counts[i] + 1)
            else:
                new_counts = new_counts.at[i].set(0)

        has_stable_extreme = any(
            new_counts[i] >= self.agg_stability_threshold
            for i in CORE_EXTREME_AGGS
        )

        return new_counts, has_stable_extreme

    def _update_tags_with_facilitation(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        extreme_agg_discovered: bool,
        facilitated_acts: Dict[int, int],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with keystone facilitation boost."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        # Aggregation tagging (prioritized)
        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.35)
                )

        # Activation tagging with facilitation boost
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:  # Sin
                    tag_strength *= self.extreme_tag_boost

                # Extreme agg boost
                if extreme_agg_discovered:
                    tag_strength *= (1 + self.agg_led_activation_boost)

                # KEYSTONE FACILITATION BOOST (NEW)
                if i in facilitated_acts:
                    facilitation_boost = self.keystone_facilitation
                    if i == 4:  # Sin gets extra
                        facilitation_boost *= (1 + self.sin_extreme_facilitation_boost)
                    tag_strength *= (1 + facilitation_boost)

                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
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
        """Attempt capture."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                # Capture aggregations
                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

                # Capture activations
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

        return new_act_captured, new_agg_captured, capture_count

    def _update_affinities(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
        extreme_agg_discovered: bool,
        keystone_aggs: Set[int],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update affinities with keystone bonus."""
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if fitness_delta > 0:
            # Aggregation affinity update
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus += self.agg_discovery_bonus
                    # Keystone bonus
                    if j in keystone_aggs:
                        bonus *= 1.3
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta * bonus)
                    )

            # Activation affinity update
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    lr = self.act_affinity_lr
                    if extreme_agg_discovered:
                        lr *= (1 + self.agg_led_activation_boost)
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + lr * fitness_delta)
                    )

        # Cross-domain affinity update with sin-extreme coupling
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(act_active, agg_active)
            delta = self.cross_learning_rate * fitness_delta * co_active

            # Strong sin-extreme coupling with keystone boost
            for i in [4]:  # sin
                for j in CORE_EXTREME_AGGS:
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        coupling = self.sin_extreme_coupling
                        if j in keystone_aggs:
                            coupling *= (1 + self.keystone_facilitation)
                        delta = delta.at[i, j].set(delta[i, j] * (1 + coupling))

            new_cross = jnp.clip(new_cross + delta, 0.0, 1.0)

        return new_act_aff, new_agg_aff, new_cross

    def _explore_aggregations(
        self,
        key: jax.random.PRNGKey,
        agg_mask: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        agg_captured: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, int]:
        """Aggressive aggregation exploration."""
        new_mask = agg_mask.copy()
        n_explorations = 0

        explore_probs = jax.random.uniform(key, (NUM_AGGREGATIONS,))

        for j in range(NUM_AGGREGATIONS):
            effective_rate = self.agg_exploration_rate
            if j in CORE_EXTREME_AGGS:
                effective_rate *= 1.5

            if agg_mask[j] < 0.5:
                if explore_probs[j] < effective_rate:
                    new_mask = new_mask.at[j].set(1.0)
                    n_explorations += 1

        if int(jnp.sum(new_mask > 0.5)) > self.max_active_agg:
            new_mask = agg_mask
            n_explorations = 0

        return new_mask, n_explorations

    def _select_act_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
        extreme_agg_discovered: bool,
        facilitated_acts: Dict[int, int],
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int, int]:
        """Select activation palette with facilitation protection."""
        score = affinities + captured * 0.3 + tags * 0.2

        # Cross-domain influence
        for i in range(NUM_ACTIVATIONS):
            cross_influence = 0.0
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    cross_influence = max(cross_influence, float(cross_affinity[i, j]))
            score = score.at[i].set(score[i] + cross_influence * 0.25)

        # Sin preference boost
        sin_boost = self.discovery_bonus if hasattr(self, 'discovery_bonus') else 0.5
        if extreme_agg_discovered:
            sin_boost *= (1 + self.agg_led_activation_boost)
        score = score.at[4].set(score[4] + sin_boost)

        # FACILITATION PROTECTION (NEW) - facilitated functions get score boost
        facilitation_protection_count = 0
        for i in facilitated_acts:
            score = score.at[i].set(score[i] + self.keystone_protection)
            if i == 4:
                score = score.at[i].set(score[i] + self.sin_extreme_facilitation_boost)
            facilitation_protection_count += 1

        # Select top-k
        target_size = min(max(self.min_diversity_act, self.min_active_act), self.max_active_act)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < self.min_diversity_act:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(NUM_ACTIVATIONS) if mask[i] < 0.5]
            needed = self.min_diversity_act - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive), shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue, facilitation_protection_count

    def _select_agg_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        keystone_aggs: Set[int],
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int]:
        """Select aggregation palette with keystone protection."""
        score = affinities + captured * 0.3 + tags * 0.2

        # Extreme preference
        for j in CORE_EXTREME_AGGS:
            score = score.at[j].set(score[j] + self.agg_discovery_bonus)

        # Keystone protection (NEW)
        for j in keystone_aggs:
            score = score.at[j].set(score[j] + self.keystone_protection)

        target_size = min(max(self.min_diversity_agg, self.min_active_agg), self.max_active_agg)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(NUM_AGGREGATIONS)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < self.min_diversity_agg:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(NUM_AGGREGATIONS) if mask[i] < 0.5]
            needed = self.min_diversity_agg - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive), shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute extreme/averaging ratio."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with keystone-enhanced aggregation-led mechanics."""
        key, k1, k2, k3 = jax.random.split(state['rng_key'], 4)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === AGGREGATION STABILITY TRACKING ===
        new_agg_stability, has_stable_extreme = self._update_agg_stability(
            state['agg_mask'], state['agg_stability_counts']
        )

        # Track extreme agg discovery
        new_extreme_discovered = state['extreme_agg_discovered']
        new_extreme_discovery_gen = state['extreme_agg_discovery_gen']
        if has_stable_extreme and not state['extreme_agg_discovered']:
            new_extreme_discovered = True
            new_extreme_discovery_gen = generation

        # === KEYSTONE STATUS UPDATE (NEW) ===
        new_keystones, n_promotions = self._update_keystone_status(
            state['agg_mask'], new_agg_stability,
            state['agg_affinities'], state['keystone_aggs']
        )

        # Update keystone history
        keystone_history = list(state['keystone_history'])
        if new_keystones != state['keystone_aggs']:
            keystone_history.append((generation, list(new_keystones)))
            if len(keystone_history) > 20:
                keystone_history = keystone_history[-20:]

        # === FACILITATION NETWORK (NEW) ===
        new_facilitated = self._compute_facilitation(
            new_keystones, state['act_mask'], state['cross_affinity']
        )

        # === AGGREGATION EXPLORATION ===
        new_agg_mask, n_agg_explorations = self._explore_aggregations(
            k1, state['agg_mask'], state['agg_affinities'], state['agg_captured']
        )

        # === TAGGING WITH FACILITATION ===
        new_act_tags, new_agg_tags = self._update_tags_with_facilitation(
            state['act_mask'], new_agg_mask,
            state['act_tags'], state['agg_tags'],
            new_extreme_discovered, new_facilitated
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

        # === AFFINITY UPDATE WITH KEYSTONE BONUS ===
        new_act_aff, new_agg_aff, new_cross_affinity = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], new_agg_mask,
            state['cross_affinity'], fitness_delta,
            new_extreme_discovered, new_keystones
        )

        # === PALETTE SELECTION ===
        # Aggregation palette with keystone protection
        new_agg_mask_selected, agg_diversity_rescue = self._select_agg_palette(
            new_agg_aff, new_agg_captured, new_agg_tags, new_keystones, k2
        )
        final_agg_mask = jnp.maximum(new_agg_mask, new_agg_mask_selected)
        if int(jnp.sum(final_agg_mask > 0.5)) > self.max_active_agg:
            final_agg_mask = new_agg_mask_selected

        # Activation palette with facilitation protection
        new_act_mask, act_diversity_rescue, facilitation_protections = self._select_act_palette(
            new_act_aff, new_act_captured, new_act_tags,
            new_cross_affinity, final_agg_mask,
            new_extreme_discovered, new_facilitated, k3
        )

        activation_boost_active = new_extreme_discovered

        extreme_ratio = self._compute_extreme_ratio(final_agg_mask)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], final_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': final_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            'agg_stability_counts': new_agg_stability,
            'extreme_agg_discovered': new_extreme_discovered,
            'extreme_agg_discovery_gen': new_extreme_discovery_gen,
            'activation_boost_active': activation_boost_active,
            # Keystone state (NEW)
            'keystone_aggs': new_keystones,
            'facilitated_acts': new_facilitated,
            'keystone_history': keystone_history,
            # Stats
            'capture_events': state['capture_events'] + capture_count,
            'agg_exploration_events': state['agg_exploration_events'] + n_agg_explorations,
            'act_boosted_events': state['act_boosted_events'] + (1 if activation_boost_active and not state['activation_boost_active'] else 0),
            'keystone_promotions': state['keystone_promotions'] + n_promotions,
            'facilitation_protections': state['facilitation_protections'] + facilitation_protections,
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue + agg_diversity_rescue,
            # General
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(final_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Keystone metrics (NEW)
            'n_keystones': len(new_keystones),
            'keystones': list(new_keystones),
            'n_facilitated': len(new_facilitated),
            'facilitated_acts': list(new_facilitated.keys()),
            'sin_facilitated': 4 in new_facilitated,
            'keystone_promotions': new_state['keystone_promotions'],
            'facilitation_protections': facilitation_protections,
            # Aggregation-led metrics
            'extreme_agg_discovered': new_extreme_discovered,
            'extreme_agg_discovery_gen': new_extreme_discovery_gen,
            'activation_boost_active': activation_boost_active,
            'agg_exploration_events': new_state['agg_exploration_events'],
            'has_stable_extreme': has_stable_extreme,
            # Affinity metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'sin_affinity': float(new_act_aff[4]),
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Homeostatic
            'extreme_ratio': extreme_ratio,
            # Cross-domain
            'sin_max_affinity': float(new_cross_affinity[4, 2]),
            'sin_min_affinity': float(new_cross_affinity[4, 3]),
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with keystone info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Keystone (NEW)
            'n_keystones': len(state['keystone_aggs']),
            'keystones': list(state['keystone_aggs']),
            'n_facilitated': len(state['facilitated_acts']),
            'sin_facilitated': 4 in state['facilitated_acts'],
            'keystone_promotions': state['keystone_promotions'],
            # Aggregation-led
            'extreme_agg_discovered': state['extreme_agg_discovered'],
            'activation_boost_active': state['activation_boost_active'],
            # Affinity
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinities'])),
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'capture_events': state['capture_events'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
