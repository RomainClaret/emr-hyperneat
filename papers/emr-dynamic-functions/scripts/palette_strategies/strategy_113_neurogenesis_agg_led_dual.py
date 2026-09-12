"""Strategy 113: Neurogenesis Aggregation-Led Dual.

Combines aggregation_led_dual (#101) with adult_neurogenesis (#41) mechanics.
Controlled aggregation birth with survival criteria creates efficient
extreme discovery while preventing palette bloat.

Problem with pure aggregation-led:
- Aggressive exploration can add functions that don't contribute
- No survival-based pruning mechanism
- Palette can grow with unused functions

Key Innovation:
- New aggregations go through maturation period (young aggs)
- Young aggs have higher plasticity (faster learning)
- Only aggs that contribute to fitness survive integration
- Sin gets survival bonus when coupled with mature extreme aggs

Bio inspiration: Adult neurogenesis in hippocampus generates new neurons
that must form useful connections to survive. Similarly, new aggregations
must prove their worth during maturation to join the stable palette.

Expected: Efficient extreme discovery with automatic pruning of failures.
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


class NeurogenesisAggLedDualStrategy(PaletteEvolutionStrategy):
    """Neurogenesis-enhanced aggregation-led dual palette evolution.

    Combines aggressive aggregation exploration with
    neurogenesis-style birth/maturation/survival mechanics.

    Critical innovation: New aggregations must survive a maturation period
    by contributing to fitness. Sin gets survival bonus when coupled with
    mature extreme aggregations.
    """

    name = "neurogenesis_agg_led_dual"
    description = "Dual: Aggregation-led with neurogenesis birth/survival mechanics"

    def __init__(
        self,
        # === NEUROGENESIS PARAMETERS (NEW from adult_neurogenesis) ===
        agg_neurogenesis_rate: float = 0.10,       # Birth rate for new aggs
        agg_maturation_period: int = 8,            # Gens before survival decision
        young_agg_plasticity: float = 2.0,         # Learning multiplier for young aggs
        agg_survival_threshold: float = 0.15,      # Minimum contribution to survive
        max_young_aggs: int = 2,                   # Max concurrent young aggs
        survival_sin_coupling_bonus: float = 0.3,  # Survival boost from sin coupling
        # === AGGREGATION-LED PARAMETERS (from #101) ===
        agg_exploration_rate: float = 0.25,        # Still used for initial discovery
        agg_discovery_bonus: float = 0.6,
        act_exploration_rate: float = 0.08,
        agg_led_activation_boost: float = 0.4,
        agg_stability_threshold: int = 5,
        # === Contribution tracking ===
        contribution_decay: float = 0.9,
        contribution_boost_on_improvement: float = 0.25,
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
        max_stable_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Neurogenesis Aggregation-Led strategy."""
        # Neurogenesis parameters (NEW)
        self.agg_neurogenesis_rate = agg_neurogenesis_rate
        self.agg_maturation_period = agg_maturation_period
        self.young_agg_plasticity = young_agg_plasticity
        self.agg_survival_threshold = agg_survival_threshold
        self.max_young_aggs = max_young_aggs
        self.survival_sin_coupling_bonus = survival_sin_coupling_bonus

        # Aggregation-led parameters
        self.agg_exploration_rate = agg_exploration_rate
        self.agg_discovery_bonus = agg_discovery_bonus
        self.act_exploration_rate = act_exploration_rate
        self.agg_led_activation_boost = agg_led_activation_boost
        self.agg_stability_threshold = agg_stability_threshold

        # Contribution
        self.contribution_decay = contribution_decay
        self.contribution_boost_on_improvement = contribution_boost_on_improvement

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
        self.max_stable_agg = max_stable_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neurogenesis tracking."""
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

        # Neurogenesis state (NEW)
        stable_aggs: Set[int] = set(initial_agg)
        young_aggs: Dict[int, Dict] = {}  # {agg_idx: {'birth_gen': int}}
        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.4)

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
            # Neurogenesis state (NEW)
            'stable_aggs': stable_aggs,
            'young_aggs': young_aggs,
            'agg_contribution': agg_contribution,
            'agg_births': [],      # List of (gen, agg_idx) tuples
            'agg_survivals': [],   # Aggs that survived maturation
            'agg_prunings': [],    # Aggs that were pruned
            # Stats
            'capture_events': 0,
            'agg_exploration_events': 0,
            'act_boosted_events': 0,
            'total_agg_births': 0,
            'total_agg_survivals': 0,
            'total_agg_prunings': 0,
            'diversity_rescues': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1130000),
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

    def _maybe_birth_agg(
        self,
        stable_aggs: Set[int],
        young_aggs: Dict[int, Dict],
        key: jax.random.PRNGKey,
        generation: int,
    ) -> Tuple[Set[int], Dict[int, Dict], Optional[int]]:
        """Possibly birth a new aggregation with preference for extremes."""
        key1, key2 = jax.random.split(key)
        new_young = dict(young_aggs)
        born = None

        if (len(new_young) < self.max_young_aggs and
            jax.random.uniform(key1) < self.agg_neurogenesis_rate):

            # Find available aggs (not in stable or young)
            available = [j for j in range(NUM_AGGREGATIONS)
                        if j not in stable_aggs and j not in new_young]

            if available:
                # Prefer extreme aggs
                extreme_available = [j for j in available if j in CORE_EXTREME_AGGS]

                if extreme_available and jax.random.uniform(key2) < 0.7:
                    # 70% chance to try extreme first
                    idx = int(jax.random.randint(key2, (), 0, len(extreme_available)))
                    new_agg = extreme_available[idx]
                else:
                    idx = int(jax.random.randint(key2, (), 0, len(available)))
                    new_agg = available[idx]

                new_young[new_agg] = {'birth_gen': generation}
                born = new_agg

        return stable_aggs, new_young, born

    def _mature_aggs(
        self,
        stable_aggs: Set[int],
        young_aggs: Dict[int, Dict],
        agg_contribution: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        generation: int,
    ) -> Tuple[Set[int], Dict[int, Dict], List[int], List[int]]:
        """Process aggregation maturation - survive or prune.

        KEY INNOVATION: Sin coupling provides survival bonus.
        Extreme aggs with high sin affinity are more likely to survive.
        """
        new_stable = set(stable_aggs)
        new_young = {}
        survived = []
        pruned = []

        # Check sin activity
        sin_active = act_mask[4] > 0.5

        for agg_idx, info in young_aggs.items():
            age = generation - info['birth_gen']

            if age >= self.agg_maturation_period:
                # Time to decide survival
                base_contribution = float(agg_contribution[agg_idx])

                # Sin coupling bonus (if sin is active and has high affinity)
                survival_threshold = self.agg_survival_threshold
                if sin_active and agg_idx in CORE_EXTREME_AGGS:
                    sin_aff = float(cross_affinity[4, agg_idx])
                    if sin_aff > 0.5:
                        # Lower threshold = easier survival
                        survival_threshold *= (1 - self.survival_sin_coupling_bonus)
                        base_contribution += sin_aff * 0.1  # Bonus contribution

                if base_contribution >= survival_threshold:
                    # Survived! Integrate into stable palette
                    if len(new_stable) < self.max_stable_agg:
                        new_stable.add(agg_idx)
                        survived.append(agg_idx)
                    else:
                        # Stable palette full, still prune
                        pruned.append(agg_idx)
                else:
                    # Failed to contribute enough, prune
                    pruned.append(agg_idx)
            else:
                # Still young
                new_young[agg_idx] = info

        return new_stable, new_young, survived, pruned

    def _update_agg_contribution(
        self,
        agg_contribution: jnp.ndarray,
        agg_mask: jnp.ndarray,
        young_aggs: Dict[int, Dict],
        improved: bool,
    ) -> jnp.ndarray:
        """Update aggregation contribution tracking."""
        new_contribution = agg_contribution * self.contribution_decay

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                current = float(new_contribution[j])
                if improved:
                    boost = self.contribution_boost_on_improvement
                    # Young aggs get extra plasticity
                    if j in young_aggs:
                        boost *= self.young_agg_plasticity
                    # Extreme aggs get bonus
                    if j in CORE_EXTREME_AGGS:
                        boost *= 1.3
                    new_contribution = new_contribution.at[j].set(current + boost)
                else:
                    # Small boost for being active
                    new_contribution = new_contribution.at[j].set(current + 0.01)

        return jnp.clip(new_contribution, 0, 2.0)

    def _create_agg_mask(
        self,
        stable_aggs: Set[int],
        young_aggs: Dict[int, Dict],
    ) -> jnp.ndarray:
        """Create agg mask from stable and young aggs."""
        mask = jnp.zeros(NUM_AGGREGATIONS)

        for j in stable_aggs:
            if 0 <= j < NUM_AGGREGATIONS:
                mask = mask.at[j].set(1.0)

        for j in young_aggs.keys():
            if 0 <= j < NUM_AGGREGATIONS:
                mask = mask.at[j].set(1.0)

        return mask

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
        young_aggs: Dict[int, Dict],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with young agg plasticity boost."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        # Aggregation tagging (prioritized, with young plasticity)
        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                # Young aggs tag faster (more plastic)
                if j in young_aggs:
                    tag_strength *= self.young_agg_plasticity
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.35)
                )

        # Activation tagging
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:  # Sin
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
        young_aggs: Dict[int, Dict],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update affinities with young agg plasticity."""
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if fitness_delta > 0:
            # Aggregation affinity update (with young plasticity)
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    lr = self.agg_affinity_lr
                    if j in young_aggs:
                        lr *= self.young_agg_plasticity
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus += self.agg_discovery_bonus
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + lr * fitness_delta * bonus)
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

        # Cross-domain affinity update
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(act_active, agg_active)

            # Base cross-learning with young agg boost
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        lr = self.cross_learning_rate
                        if j in young_aggs:
                            lr *= self.young_agg_plasticity

                        delta = lr * fitness_delta

                        # Strong sin-extreme coupling
                        if i == 4 and j in CORE_EXTREME_AGGS:
                            delta *= (1 + self.sin_extreme_coupling)

                        new_cross = new_cross.at[i, j].set(
                            min(1.0, new_cross[i, j] + delta)
                        )

        return new_act_aff, new_agg_aff, new_cross

    def _select_act_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
        extreme_agg_discovered: bool,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int]:
        """Select activation palette."""
        score = affinities + captured * 0.3 + tags * 0.2

        # Cross-domain influence
        for i in range(NUM_ACTIVATIONS):
            cross_influence = 0.0
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    cross_influence = max(cross_influence, float(cross_affinity[i, j]))
            score = score.at[i].set(score[i] + cross_influence * 0.25)

        # Sin preference
        sin_boost = 0.5
        if extreme_agg_discovered:
            sin_boost *= (1 + self.agg_led_activation_boost)
        score = score.at[4].set(score[4] + sin_boost)

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
        """Update with neurogenesis-enhanced aggregation-led mechanics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        stable_aggs = set(state['stable_aggs'])
        young_aggs = dict(state['young_aggs'])

        # === AGG CONTRIBUTION UPDATE (NEW) ===
        new_agg_contribution = self._update_agg_contribution(
            state['agg_contribution'],
            state['agg_mask'],
            young_aggs,
            improved
        )

        # === MATURATION PROCESSING (NEW) ===
        stable_aggs, young_aggs, survived, pruned = self._mature_aggs(
            stable_aggs, young_aggs,
            new_agg_contribution,
            state['cross_affinity'],
            state['act_mask'],
            generation
        )

        # === MAYBE BIRTH NEW AGG (NEW) ===
        stable_aggs, young_aggs, born = self._maybe_birth_agg(
            stable_aggs, young_aggs, k1, generation
        )

        # === CREATE AGG MASK FROM NEUROGENESIS STATE ===
        new_agg_mask = self._create_agg_mask(stable_aggs, young_aggs)

        # === AGGREGATION STABILITY TRACKING ===
        new_agg_stability, has_stable_extreme = self._update_agg_stability(
            new_agg_mask, state['agg_stability_counts']
        )

        # Track extreme agg discovery
        new_extreme_discovered = state['extreme_agg_discovered']
        new_extreme_discovery_gen = state['extreme_agg_discovery_gen']
        if has_stable_extreme and not state['extreme_agg_discovered']:
            new_extreme_discovered = True
            new_extreme_discovery_gen = generation

        activation_boost_active = new_extreme_discovered

        # === TAGGING WITH YOUNG PLASTICITY ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], new_agg_mask,
            state['act_tags'], state['agg_tags'],
            new_extreme_discovered, young_aggs
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

        # === AFFINITY UPDATE WITH YOUNG PLASTICITY ===
        new_act_aff, new_agg_aff, new_cross_affinity = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], new_agg_mask,
            state['cross_affinity'], fitness_delta,
            new_extreme_discovered, young_aggs
        )

        # === ACTIVATION PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_act_palette(
            new_act_aff, new_act_captured, new_act_tags,
            new_cross_affinity, new_agg_mask,
            new_extreme_discovered, k2
        )

        extreme_ratio = self._compute_extreme_ratio(new_agg_mask)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Update history
        agg_births = list(state['agg_births'])
        if born is not None:
            agg_births.append((generation, born))
            if len(agg_births) > 50:
                agg_births = agg_births[-50:]

        agg_survivals = list(state['agg_survivals']) + survived
        if len(agg_survivals) > 50:
            agg_survivals = agg_survivals[-50:]

        agg_prunings = list(state['agg_prunings']) + pruned
        if len(agg_prunings) > 50:
            agg_prunings = agg_prunings[-50:]

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
            'agg_stability_counts': new_agg_stability,
            'extreme_agg_discovered': new_extreme_discovered,
            'extreme_agg_discovery_gen': new_extreme_discovery_gen,
            'activation_boost_active': activation_boost_active,
            # Neurogenesis state (NEW)
            'stable_aggs': stable_aggs,
            'young_aggs': young_aggs,
            'agg_contribution': new_agg_contribution,
            'agg_births': agg_births,
            'agg_survivals': agg_survivals,
            'agg_prunings': agg_prunings,
            # Stats
            'capture_events': state['capture_events'] + capture_count,
            'agg_exploration_events': state['agg_exploration_events'],
            'act_boosted_events': state['act_boosted_events'] + (1 if activation_boost_active and not state['activation_boost_active'] else 0),
            'total_agg_births': state['total_agg_births'] + (1 if born else 0),
            'total_agg_survivals': state['total_agg_survivals'] + len(survived),
            'total_agg_prunings': state['total_agg_prunings'] + len(pruned),
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue,
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

        # Survival rate
        survival_rate = (new_state['total_agg_survivals'] /
                        max(new_state['total_agg_births'], 1)) * 100

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neurogenesis metrics (NEW)
            'n_stable_aggs': len(stable_aggs),
            'n_young_aggs': len(young_aggs),
            'stable_aggs': list(stable_aggs),
            'young_aggs': list(young_aggs.keys()),
            'born_this_gen': born,
            'survived_this_gen': survived,
            'pruned_this_gen': pruned,
            'total_agg_births': new_state['total_agg_births'],
            'total_agg_survivals': new_state['total_agg_survivals'],
            'total_agg_prunings': new_state['total_agg_prunings'],
            'agg_survival_rate': survival_rate,
            # Aggregation-led metrics
            'extreme_agg_discovered': new_extreme_discovered,
            'extreme_agg_discovery_gen': new_extreme_discovery_gen,
            'activation_boost_active': activation_boost_active,
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
        """Return state summary with neurogenesis info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        survival_rate = (state['total_agg_survivals'] /
                        max(state['total_agg_births'], 1)) * 100

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Neurogenesis (NEW)
            'n_stable_aggs': len(state['stable_aggs']),
            'n_young_aggs': len(state['young_aggs']),
            'stable_aggs': list(state['stable_aggs']),
            'total_agg_births': state['total_agg_births'],
            'total_agg_survivals': state['total_agg_survivals'],
            'total_agg_prunings': state['total_agg_prunings'],
            'agg_survival_rate': survival_rate,
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
