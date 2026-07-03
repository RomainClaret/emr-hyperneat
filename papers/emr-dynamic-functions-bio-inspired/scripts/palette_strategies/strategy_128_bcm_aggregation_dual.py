"""Strategy 128: BCM-Aggregation Self-Regulating Discovery Dual.

Combines BCM Metaplasticity (#15) with Aggregation-led (#101) mechanics.
BCM sliding threshold auto-regulates exploration based on discovery rate.

Key Innovation:
- BCM sliding threshold tracks recent discovery success
- High discovery rate -> raise threshold (be more selective)
- Low discovery rate -> lower threshold (explore more)
- Aggregation-first: extreme aggs discovered first, activations follow
- Extreme aggs have LOWER BCM threshold (automatic extreme preference)

Biological basis: BCM theory proposes a sliding modification threshold where
the LTP/LTD boundary depends on recent activity. Here, recent discovery
success modulates exploration rate - successful discovery creates confidence
to be more selective, while stagnation opens up exploration.

Expected: Self-regulating exploration that adapts to task difficulty.
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


class BCMAggregationDualStrategy(PaletteEvolutionStrategy):
    """BCM-regulated aggregation-led dual palette evolution.

    Combines BCM sliding threshold with
    aggregation-first discovery. The BCM threshold self-regulates
    based on discovery rate.

    Critical innovation: BCM threshold adapts to discovery success.
    When discoveries happen, threshold rises (be selective). When
    stagnating, threshold falls (explore more). Extreme aggs have
    inherently lower threshold for automatic preference.
    """

    name = "bcm_aggregation_dual"
    description = "Dual: BCM self-regulating threshold with aggregation-led discovery"

    def __init__(
        self,
        # === BCM PARAMETERS (NEW from #15) ===
        bcm_threshold_lr: float = 0.1,        # Learning rate for threshold
        bcm_window: int = 10,                 # Window to track discovery rate
        bcm_min_threshold: float = 0.3,       # Min protection threshold
        bcm_max_threshold: float = 0.8,       # Max protection threshold
        bcm_target_discovery_rate: float = 0.15,  # Target discovery per gen
        bcm_threshold_decay: float = 0.95,    # Threshold decay toward baseline
        # === EXTREME PREFERENCE (NEW) ===
        extreme_bcm_discount: float = 0.3,    # Threshold reduction for extremes
        sin_bcm_discount: float = 0.25,       # Threshold reduction for sin
        # === AGGREGATION-LED PARAMETERS (from #101) ===
        agg_exploration_rate: float = 0.25,
        agg_discovery_bonus: float = 0.6,
        act_exploration_rate: float = 0.08,
        agg_led_activation_boost: float = 0.4,
        agg_stability_threshold: int = 5,
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
        """Initialize BCM-Aggregation strategy."""
        # BCM parameters (NEW)
        self.bcm_threshold_lr = bcm_threshold_lr
        self.bcm_window = bcm_window
        self.bcm_min_threshold = bcm_min_threshold
        self.bcm_max_threshold = bcm_max_threshold
        self.bcm_target_discovery_rate = bcm_target_discovery_rate
        self.bcm_threshold_decay = bcm_threshold_decay

        # Extreme preference (NEW)
        self.extreme_bcm_discount = extreme_bcm_discount
        self.sin_bcm_discount = sin_bcm_discount

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
        """Initialize state with BCM and aggregation-led tracking."""
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

        # BCM state (NEW)
        bcm_threshold = 0.55  # Start at middle
        discovery_history: List[bool] = []  # Track discoveries per gen

        # Aggregation stability tracking
        agg_stability_counts = jnp.zeros(NUM_AGGREGATIONS)

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
            # BCM state (NEW)
            'bcm_threshold': bcm_threshold,
            'discovery_history': discovery_history,
            # Aggregation-led tracking
            'agg_stability_counts': agg_stability_counts,
            'extreme_agg_discovered': False,
            'extreme_agg_discovery_gen': None,
            'activation_boost_active': False,
            # Discovered functions (for BCM)
            'discovered_acts': set(initial_act),
            'discovered_aggs': set(initial_agg),
            # Stats
            'capture_events': 0,
            'agg_exploration_events': 0,
            'act_boosted_events': 0,
            'diversity_rescues': 0,
            'bcm_threshold_history': [bcm_threshold],
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1280000),
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

    def _update_bcm_threshold(
        self,
        current_threshold: float,
        discovery_history: List[bool],
        new_discoveries: int,
    ) -> Tuple[float, List[bool]]:
        """Update BCM sliding threshold based on discovery rate.

        KEY INNOVATION: High discovery -> raise threshold (be selective)
                        Low discovery -> lower threshold (explore more)
        """
        # Update history
        new_history = discovery_history + [new_discoveries > 0]
        if len(new_history) > self.bcm_window:
            new_history = new_history[-self.bcm_window:]

        # Compute recent discovery rate
        if len(new_history) >= 3:
            discovery_rate = sum(new_history) / len(new_history)

            # Compare to target rate
            error = discovery_rate - self.bcm_target_discovery_rate

            # Adjust threshold
            # If discovery_rate > target: raise threshold (be more selective)
            # If discovery_rate < target: lower threshold (explore more)
            adjustment = self.bcm_threshold_lr * error

            new_threshold = current_threshold + adjustment
        else:
            new_threshold = current_threshold

        # Apply decay toward baseline (0.55)
        baseline = 0.55
        new_threshold = new_threshold * self.bcm_threshold_decay + baseline * (1 - self.bcm_threshold_decay)

        # Clamp
        new_threshold = max(self.bcm_min_threshold, min(self.bcm_max_threshold, new_threshold))

        return new_threshold, new_history

    def _get_effective_threshold(
        self,
        base_threshold: float,
        func_idx: int,
        is_activation: bool,
    ) -> float:
        """Get function-specific BCM threshold.

        KEY INNOVATION: Extreme aggs and sin have lower threshold.
        """
        threshold = base_threshold

        if is_activation:
            if func_idx == 4:  # sin
                threshold *= (1 - self.sin_bcm_discount)
        else:
            if func_idx in CORE_EXTREME_AGGS:
                threshold *= (1 - self.extreme_bcm_discount)

        return threshold

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

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        extreme_agg_discovered: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.35)
                )

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:
                    tag_strength *= self.extreme_tag_boost
                if extreme_agg_discovered:
                    tag_strength *= (1 + self.agg_led_activation_boost)
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
                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update affinities."""
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if fitness_delta > 0:
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus += self.agg_discovery_bonus
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta * bonus)
                    )

            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    lr = self.act_affinity_lr
                    if extreme_agg_discovered:
                        lr *= (1 + self.agg_led_activation_boost)
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + lr * fitness_delta)
                    )

        # Cross-domain
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            for i in [4]:
                for j in CORE_EXTREME_AGGS:
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        delta = self.cross_learning_rate * fitness_delta * (1 + self.sin_extreme_coupling)
                        new_cross = new_cross.at[i, j].set(
                            min(1.0, new_cross[i, j] + delta)
                        )

        return new_act_aff, new_agg_aff, new_cross

    def _explore_aggregations_bcm(
        self,
        key: jax.random.PRNGKey,
        agg_mask: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        bcm_threshold: float,
        discovered_aggs: Set[int],
    ) -> Tuple[jnp.ndarray, int, Set[int]]:
        """Explore aggregations with BCM-regulated threshold.

        KEY INNOVATION: Exploration rate modulated by BCM threshold.
        Lower threshold = more exploration allowed.
        """
        new_mask = agg_mask.copy()
        n_explorations = 0
        new_discovered = set(discovered_aggs)

        explore_probs = jax.random.uniform(key, (NUM_AGGREGATIONS,))

        # BCM modulates exploration rate
        # Lower threshold -> explore more
        bcm_exploration_mult = (self.bcm_max_threshold - bcm_threshold) / (
            self.bcm_max_threshold - self.bcm_min_threshold
        )
        bcm_exploration_mult = 0.5 + bcm_exploration_mult  # Range [0.5, 1.5]

        for j in range(NUM_AGGREGATIONS):
            effective_threshold = self._get_effective_threshold(bcm_threshold, j, False)

            # Exploration rate modulated by BCM
            effective_rate = self.agg_exploration_rate * bcm_exploration_mult

            # Extra boost for extreme aggs
            if j in CORE_EXTREME_AGGS:
                effective_rate *= 1.5

            if agg_mask[j] < 0.5:
                if explore_probs[j] < effective_rate:
                    new_mask = new_mask.at[j].set(1.0)
                    n_explorations += 1
                    if j not in new_discovered:
                        new_discovered.add(j)

        # Don't exceed max
        if int(jnp.sum(new_mask > 0.5)) > self.max_active_agg:
            new_mask = agg_mask
            n_explorations = 0
            new_discovered = discovered_aggs

        return new_mask, n_explorations, new_discovered

    def _select_act_palette_bcm(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
        extreme_agg_discovered: bool,
        bcm_threshold: float,
        discovered_acts: Set[int],
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int, Set[int]]:
        """Select activation palette with BCM-modulated threshold."""
        score = affinities + captured * 0.3 + tags * 0.2
        new_discovered = set(discovered_acts)

        for i in range(NUM_ACTIVATIONS):
            cross_influence = 0.0
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    cross_influence = max(cross_influence, float(cross_affinity[i, j]))
            score = score.at[i].set(score[i] + cross_influence * 0.25)

        # Sin preference (BCM-boosted)
        sin_boost = 0.5
        if extreme_agg_discovered:
            sin_boost *= (1 + self.agg_led_activation_boost)
        # Lower BCM threshold = more sin preference
        bcm_sin_boost = (self.bcm_max_threshold - bcm_threshold) / (
            self.bcm_max_threshold - self.bcm_min_threshold
        ) * 0.3
        score = score.at[4].set(score[4] + sin_boost + bcm_sin_boost)

        target_size = min(max(self.min_diversity_act, self.min_active_act), self.max_active_act)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)
            if int(idx) not in new_discovered:
                new_discovered.add(int(idx))

        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < self.min_diversity_act:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(NUM_ACTIVATIONS) if mask[i] < 0.5]
            needed = self.min_diversity_act - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive),
                                          shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue, new_discovered

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
        """Update with BCM-regulated aggregation-led mechanics."""
        key, k1, k2, k3 = jax.random.split(state['rng_key'], 4)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === AGG STABILITY TRACKING ===
        new_agg_stability, has_stable_extreme = self._update_agg_stability(
            state['agg_mask'], state['agg_stability_counts']
        )

        new_extreme_discovered = state['extreme_agg_discovered']
        new_extreme_discovery_gen = state['extreme_agg_discovery_gen']
        if has_stable_extreme and not state['extreme_agg_discovered']:
            new_extreme_discovered = True
            new_extreme_discovery_gen = generation

        activation_boost_active = new_extreme_discovered

        # === AGGREGATION EXPLORATION WITH BCM ===
        new_agg_mask, n_agg_explorations, new_discovered_aggs = self._explore_aggregations_bcm(
            k1, state['agg_mask'], state['agg_affinities'],
            state['bcm_threshold'], state['discovered_aggs']
        )

        # Count new discoveries
        new_agg_discoveries = len(new_discovered_aggs) - len(state['discovered_aggs'])

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], new_agg_mask,
            state['act_tags'], state['agg_tags'],
            new_extreme_discovered
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

        # === AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross_affinity = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], new_agg_mask,
            state['cross_affinity'], fitness_delta,
            new_extreme_discovered
        )

        # === ACTIVATION PALETTE SELECTION WITH BCM ===
        new_act_mask, act_diversity_rescue, new_discovered_acts = self._select_act_palette_bcm(
            new_act_aff, new_act_captured, new_act_tags,
            new_cross_affinity, new_agg_mask,
            new_extreme_discovered, state['bcm_threshold'],
            state['discovered_acts'], k2
        )

        # Count new act discoveries
        new_act_discoveries = len(new_discovered_acts) - len(state['discovered_acts'])
        total_discoveries = new_agg_discoveries + new_act_discoveries

        # === BCM THRESHOLD UPDATE ===
        new_bcm_threshold, new_discovery_history = self._update_bcm_threshold(
            state['bcm_threshold'], state['discovery_history'], total_discoveries
        )

        # Track boost events
        act_boosted = 0
        if activation_boost_active and not state['activation_boost_active']:
            act_boosted = 1

        extreme_ratio = self._compute_extreme_ratio(new_agg_mask)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        bcm_threshold_history = state['bcm_threshold_history'] + [new_bcm_threshold]
        if len(bcm_threshold_history) > 50:
            bcm_threshold_history = bcm_threshold_history[-50:]

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
            # BCM state
            'bcm_threshold': new_bcm_threshold,
            'discovery_history': new_discovery_history,
            # Aggregation-led state
            'agg_stability_counts': new_agg_stability,
            'extreme_agg_discovered': new_extreme_discovered,
            'extreme_agg_discovery_gen': new_extreme_discovery_gen,
            'activation_boost_active': activation_boost_active,
            # Discovery tracking
            'discovered_acts': new_discovered_acts,
            'discovered_aggs': new_discovered_aggs,
            # Stats
            'capture_events': state['capture_events'] + capture_count,
            'agg_exploration_events': state['agg_exploration_events'] + n_agg_explorations,
            'act_boosted_events': state['act_boosted_events'] + act_boosted,
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue,
            'bcm_threshold_history': bcm_threshold_history,
            # General
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Compute discovery rate
        if len(new_discovery_history) > 0:
            recent_discovery_rate = sum(new_discovery_history) / len(new_discovery_history)
        else:
            recent_discovery_rate = 0.0

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # BCM metrics (NEW)
            'bcm_threshold': new_bcm_threshold,
            'recent_discovery_rate': recent_discovery_rate,
            'new_discoveries_this_gen': total_discoveries,
            'total_discovered_acts': len(new_discovered_acts),
            'total_discovered_aggs': len(new_discovered_aggs),
            # Aggregation-led metrics
            'extreme_agg_discovered': new_extreme_discovered,
            'extreme_agg_discovery_gen': new_extreme_discovery_gen,
            'activation_boost_active': activation_boost_active,
            'agg_exploration_events': new_state['agg_exploration_events'],
            'act_boosted_events': new_state['act_boosted_events'],
            'has_stable_extreme': has_stable_extreme,
            # Affinity metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'sin_affinity': float(new_act_aff[4]),
            'max_affinity': float(new_agg_aff[2]),
            'min_affinity': float(new_agg_aff[3]),
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'max_captured': bool(new_agg_captured[2] > 0.5),
            'min_captured': bool(new_agg_captured[3] > 0.5),
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
        """Return state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        if len(state['discovery_history']) > 0:
            recent_discovery_rate = sum(state['discovery_history']) / len(state['discovery_history'])
        else:
            recent_discovery_rate = 0.0

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # BCM
            'bcm_threshold': state['bcm_threshold'],
            'recent_discovery_rate': recent_discovery_rate,
            'total_discovered_acts': len(state['discovered_acts']),
            'total_discovered_aggs': len(state['discovered_aggs']),
            # Aggregation-led
            'extreme_agg_discovered': state['extreme_agg_discovered'],
            'activation_boost_active': state['activation_boost_active'],
            'agg_exploration_events': state['agg_exploration_events'],
            # Affinity
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinities'])),
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'capture_events': state['capture_events'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
