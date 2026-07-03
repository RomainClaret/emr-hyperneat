"""Strategy 84: Tag+Homeostatic Dual Hybrid.

Combines two 100% sin retention winners:
- Strategy 81 (Synaptic Tagging): Tag-and-capture consolidation
- Strategy 82 (Homeostatic): Balance + discovery bonus

Key innovation: Captured functions are protected from homeostatic deactivation,
combining the best of both mechanisms for robust sin retention.

Bio inspiration: Synaptic tagging captures important connections while
homeostatic plasticity maintains overall network balance - the two mechanisms
work together in biological neural networks.

Expected: 100% sin retention with improved stability.
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


class TagHomeostaticDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution combining tagging and homeostatic mechanisms.

    Hybrid combining:
    - Synaptic tagging (81): Two-stage tag-and-capture consolidation
    - Homeostatic plasticity (82): Balance between averaging/extreme aggregations

    Critical interaction: Captured functions get protection from homeostatic
    deactivation, ensuring sin retention while maintaining exploration.
    """

    name = "tag_homeostatic_dual"
    description = "Dual: Tag-and-capture + homeostatic balance hybrid"

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
        # === Cross-domain parameters ===
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # === Mutation parameters ===
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Tag+Homeostatic hybrid strategy."""
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

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with both tagging and homeostatic tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)
        tag_history = []

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Tagging (from 81)
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Tracking
            'capture_events': 0,
            'homeostatic_corrections': 0,
            'discovery_bonuses_applied': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 840000),
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags for active functions (from strategy 81)."""
        # Decay existing tags
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        # Tag active functions
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                # Boost for sin (idx 4)
                if i == 4:
                    tag_strength *= self.extreme_tag_boost
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                # Boost for extreme aggregations
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
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
        """Attempt to capture tagged functions on improvement (from strategy 81)."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

        # Check tag history within capture window
        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                # Capture activations with strong historical tags
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

                # Capture aggregations with strong historical tags
                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

        return new_act_captured, new_agg_captured, capture_count

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute current extreme/averaging ratio (from strategy 82)."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity matrix."""
        new_affinity = cross_affinity.copy()

        if improvement > 0:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    for j in range(NUM_AGGREGATIONS):
                        if agg_mask[j] > 0.5:
                            current = cross_affinity[i, j]
                            boost = self.cross_learning_rate * improvement
                            # Extra boost for sin-extreme pairs
                            if i == 4 and j in CORE_EXTREME_AGGS:
                                boost *= (1 + self.sin_extreme_affinity_boost)
                            new_affinity = new_affinity.at[i, j].set(
                                min(1.0, current + boost)
                            )

        return new_affinity

    def _mutate_act_palette_hybrid(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Mutate activation palette with tag protection and homeostatic influence."""
        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_ACTIVATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            # Cross-domain influence
            affinity_boost = float(jnp.max(cross_affinity[i, :]))

            if mask[i] < 0.5:  # Inactive
                # Activation rate boosted by affinity
                activate_rate = self.base_activate_rate * (1 + affinity_boost * 0.5)
                # Discovery bonus for sin (from homeostatic)
                if i == 4:
                    activate_rate += self.discovery_bonus
                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                # Base deactivation rate
                deactivate_rate = self.base_deactivate_rate

                # TAG PROTECTION: Captured functions get reduced deactivation
                if captured[i] > 0.5:
                    deactivate_rate *= (1 - self.captured_protection)

                # SIN PROTECTION (homeostatic)
                if i == 4:
                    deactivate_rate *= (1 - self.sin_protection)

                # Affinity protection
                if affinity_boost > 0.6:
                    deactivate_rate *= 0.7

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_act or active_count > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _mutate_agg_palette_hybrid(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        extreme_ratio: float,
    ) -> Tuple[jnp.ndarray, Dict, int, int]:
        """Mutate aggregation palette with tag protection and homeostatic balance."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        homeostatic_corrections = 0
        discovery_bonuses = 0

        # Check if we need homeostatic correction
        needs_more_extreme = extreme_ratio < self.target_extreme_ratio - self.imbalance_threshold
        needs_more_averaging = extreme_ratio > self.target_extreme_ratio + self.imbalance_threshold

        for j in range(NUM_AGGREGATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            # Cross-domain influence (from sin if active)
            sin_affinity = float(cross_affinity[4, j]) if act_mask[4] > 0.5 else 0.5

            is_extreme = j in EXTREME_AGGS
            is_core_extreme = j in CORE_EXTREME_AGGS

            if mask[j] < 0.5:  # Inactive
                activate_rate = self.base_activate_rate

                # Homeostatic correction
                if needs_more_extreme and is_core_extreme:
                    activate_rate *= self.correction_strength
                    homeostatic_corrections += 1
                elif needs_more_averaging and not is_extreme:
                    activate_rate *= self.correction_strength
                    homeostatic_corrections += 1

                # Discovery bonus for underrepresented categories
                if is_core_extreme and extreme_ratio < 0.5:
                    activate_rate += self.discovery_bonus
                    discovery_bonuses += 1

                # Sin affinity boost
                if sin_affinity > 0.6:
                    activate_rate *= 1.3

                if p < activate_rate:
                    new_mask = new_mask.at[j].set(1.0)
                    activated.append(j)
            else:  # Active
                deactivate_rate = self.base_deactivate_rate

                # TAG PROTECTION: Captured functions protected
                if captured[j] > 0.5:
                    deactivate_rate *= (1 - self.captured_protection)

                # EXTREME PROTECTION (homeostatic)
                if is_core_extreme:
                    deactivate_rate *= (1 - self.extreme_protection)

                # Homeostatic protection based on balance
                if needs_more_extreme and is_extreme:
                    deactivate_rate *= 0.5
                elif needs_more_averaging and not is_extreme:
                    deactivate_rate *= 0.5

                # Sin affinity protection
                if sin_affinity > 0.6:
                    deactivate_rate *= 0.7

                if p < deactivate_rate:
                    new_mask = new_mask.at[j].set(0.0)
                    deactivated.append(j)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}, homeostatic_corrections, discovery_bonuses

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with combined tag+homeostatic mechanisms."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === TAGGING MECHANISM (from 81) ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags']
        )

        # Update tag history
        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # Attempt capture on improvement
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            new_tag_history, generation, improved
        )

        # === HOMEOSTATIC MECHANISM (from 82) ===
        extreme_ratio = self._compute_extreme_ratio(state['agg_mask'])

        # === CROSS-DOMAIN AFFINITY ===
        new_cross_affinity = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improvement
        )

        # === MUTATIONS ===
        new_act_mask, act_mut_info = self._mutate_act_palette_hybrid(
            k_act, state['act_mask'], new_act_captured, new_cross_affinity
        )
        new_agg_mask, agg_mut_info, homeostatic_corrections, discovery_bonuses = self._mutate_agg_palette_hybrid(
            k_agg, state['agg_mask'], new_agg_captured, new_cross_affinity,
            state['act_mask'], extreme_ratio
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            'capture_events': state['capture_events'] + capture_count,
            'homeostatic_corrections': state['homeostatic_corrections'] + homeostatic_corrections,
            'discovery_bonuses_applied': state['discovery_bonuses_applied'] + discovery_bonuses,
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
            'max_captured': bool(new_agg_captured[2] > 0.5),
            'min_captured': bool(new_agg_captured[3] > 0.5),
            'capture_events': state['capture_events'] + capture_count,
            # Homeostatic metrics
            'extreme_ratio': extreme_ratio,
            'homeostatic_corrections': state['homeostatic_corrections'] + homeostatic_corrections,
            'discovery_bonuses_applied': state['discovery_bonuses_applied'] + discovery_bonuses,
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
        """Return state summary with tag+homeostatic status."""
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
            'max_captured': bool(state['agg_captured'][2] > 0.5),
            'min_captured': bool(state['agg_captured'][3] > 0.5),
            'capture_events': state['capture_events'],
            # Homeostatic
            'homeostatic_corrections': state['homeostatic_corrections'],
            'discovery_bonuses_applied': state['discovery_bonuses_applied'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
