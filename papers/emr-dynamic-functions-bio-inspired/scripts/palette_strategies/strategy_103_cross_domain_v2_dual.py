"""Strategy 103: Cross-Domain Reinforcement v2 Dual.

Strengthened version of cross-domain reinforcement with multiplicative updates.

Problem with original (Strategy 74):
- Additive cross-domain updates are too weak
- Sin-extreme pairings can still break under domain shifts
- Cross-affinity decay can undo valuable pairings

Key Innovation:
- MULTIPLICATIVE (not additive) cross-domain affinity updates
- When sin+extreme co-activate with improvement: multiply affinity by growth factor
- Sin-extreme multiplier creates especially robust pairings
- Mutual capture bonus when both capture together

Bio inspiration: Hebbian plasticity in cortical circuits often shows multiplicative
rather than additive dynamics. LTP/LTD effects multiply existing synaptic strength,
creating stronger "winner-take-all" dynamics for successful pairings.

Expected: Sin-extreme pairings survive domain shifts through robust multiplicative coupling.
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


class CrossDomainV2DualStrategy(PaletteEvolutionStrategy):
    """Cross-domain reinforcement with multiplicative dynamics.

    Stronger sin-extreme coupling through multiplicative
    rather than additive cross-domain affinity updates.

    Critical innovation: Multiplicative updates create winner-take-all dynamics.
    Successful pairings grow exponentially stronger relative to unsuccessful ones.
    """

    name = "cross_domain_v2_dual"
    description = "Dual: Multiplicative cross-domain reinforcement for robust sin-extreme coupling"

    def __init__(
        self,
        # === MULTIPLICATIVE CROSS-DOMAIN PARAMETERS (NEW) ===
        multiplicative_update: bool = True,
        cross_affinity_growth_factor: float = 1.15,   # Multiply on success
        cross_affinity_decay_factor: float = 0.98,    # Multiply on no success
        sin_extreme_multiplier: float = 1.5,          # Extra boost for sin-extreme
        mutual_capture_bonus: float = 0.4,            # Bonus when both domains capture
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        base_cross_learning_rate: float = 0.10,       # Base additive rate (fallback)
        # === Tagging parameters ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        # === Homeostatic parameters ===
        target_extreme_ratio: float = 0.60,
        discovery_bonus: float = 0.5,
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
        """Initialize Cross-Domain v2 strategy."""
        # Multiplicative cross-domain (NEW)
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

        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.discovery_bonus = discovery_bonus

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
        """Initialize state with cross-domain affinity tracking."""
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

        # Cross-domain affinity (starts higher for sin-extreme)
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        # Initialize sin-extreme pairs higher
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
            # Cross-domain (core of this strategy)
            'cross_affinity': cross_affinity,
            # Stats
            'capture_events': 0,
            'mutual_capture_events': 0,
            'multiplicative_growth_events': 0,
            'diversity_rescues': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1030000),
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
                if i == 4:
                    tag_strength *= self.extreme_tag_boost
                # Cross-affinity boost: high cross-affinity = faster tagging
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
        """Attempt capture with mutual capture bonus.

        KEY INNOVATION: When both an activation and aggregation capture in the
        same window, their cross-affinity gets a multiplicative bonus.

        Returns: (new_act_captured, new_agg_captured, new_cross_affinity,
                  capture_count, mutual_capture_count)
        """
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        new_cross = cross_affinity.copy()
        capture_count = 0
        mutual_capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, new_cross, 0, 0

        # Track what captures this round
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

        # MUTUAL CAPTURE BONUS: Boost cross-affinity for pairs that captured together
        for i in act_captured_this_round:
            for j in agg_captured_this_round:
                current = float(new_cross[i, j])
                bonus_factor = 1.0 + self.mutual_capture_bonus
                # Extra bonus for sin-extreme
                if i == 4 and j in CORE_EXTREME_AGGS:
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
        """Update affinities with MULTIPLICATIVE cross-domain dynamics.

        KEY INNOVATION: Cross-affinity updates are multiplicative, not additive.
        Successful pairs grow exponentially stronger relative to unsuccessful ones.
        """
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
                            if i == 4 and j in CORE_EXTREME_AGGS:
                                growth_factor *= self.sin_extreme_multiplier

                            new_value = current * growth_factor
                            new_cross = new_cross.at[i, j].set(min(1.0, new_value))
                            growth_events += 1
                        else:
                            # DECAY: Multiply by decay factor (but floor at 0.2)
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

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
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
        """Select palette with cross-affinity influence."""
        # Score: affinity + capture + tag + cross-domain influence
        score = affinities + captured * 0.3 + tags * 0.2

        # Cross-domain influence: boost functions that pair well with active in other domain
        for i in range(n_funcs):
            if is_act:
                # For activations: look at cross-affinity with active aggregations
                cross_influence = 0.0
                for j in range(NUM_AGGREGATIONS):
                    if other_mask[j] > 0.5:
                        cross_influence = max(cross_influence, float(cross_affinity[i, j]))
            else:
                # For aggregations: look at cross-affinity with active activations
                cross_influence = 0.0
                for i_act in range(NUM_ACTIVATIONS):
                    if other_mask[i_act] > 0.5:
                        cross_influence = max(cross_influence, float(cross_affinity[i_act, i]))
            score = score.at[i].set(score[i] + cross_influence * 0.25)

        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + self.discovery_bonus)

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
        """Update with multiplicative cross-domain dynamics."""
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

        # === PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_palette(
            new_act_aff, new_act_captured, new_act_tags,
            new_cross_affinity, state['agg_mask'],
            NUM_ACTIVATIONS, True,
            self.min_active_act, self.max_active_act, self.min_diversity_act,
            k1, prefer_indices=[4]
        )
        new_agg_mask, agg_diversity_rescue = self._select_palette(
            new_agg_aff, new_agg_captured, new_agg_tags,
            new_cross_affinity, new_act_mask,
            NUM_AGGREGATIONS, False,
            self.min_active_agg, self.max_active_agg, self.min_diversity_agg,
            k2, prefer_indices=list(CORE_EXTREME_AGGS)
        )

        extreme_ratio = self._compute_extreme_ratio(new_agg_mask)

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
            'capture_events': state['capture_events'] + capture_count,
            'mutual_capture_events': state['mutual_capture_events'] + mutual_capture_count,
            'multiplicative_growth_events': state['multiplicative_growth_events'] + growth_events,
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
            # Cross-domain metrics (KEY)
            'sin_max_cross': float(new_cross_affinity[4, 2]),
            'sin_min_cross': float(new_cross_affinity[4, 3]),
            'max_cross_affinity': float(jnp.max(new_cross_affinity)),
            'mean_cross_affinity': float(jnp.mean(new_cross_affinity)),
            'multiplicative_growth_events': new_state['multiplicative_growth_events'],
            'mutual_capture_events': new_state['mutual_capture_events'],
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
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with cross-domain status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Cross-domain (key metric)
            'sin_max_cross': float(state['cross_affinity'][4, 2]),
            'sin_min_cross': float(state['cross_affinity'][4, 3]),
            'max_cross_affinity': float(jnp.max(state['cross_affinity'])),
            'multiplicative_growth_events': state['multiplicative_growth_events'],
            'mutual_capture_events': state['mutual_capture_events'],
            # Affinity
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'capture_events': state['capture_events'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
