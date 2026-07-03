"""Strategy 96: Clonal BCM Dual.

Combines clonal_hybrid_dual (91) with BCM metaplasticity (15):
- Base: Strategy 91 (Clonal+Tag+Homeostatic) - 100% Parity-5, 100% sin retention
- Extension: Strategy 15 (Metaplastic) - BCM sliding threshold

Key innovation: Capture threshold adapts based on capture frequency. Many
captures → threshold rises (harder to capture). Few captures → threshold
drops (easier to capture). Self-regulating protection that adapts to problem
difficulty.

Bio inspiration: BCM (Bienenstock-Cooper-Munro) theory proposes that the
threshold for LTP/LTD slides based on recent postsynaptic activity. Here,
the capture threshold slides based on capture frequency.

Expected: Self-regulating protection that prevents both over-capture (saturation)
and under-capture (losing discoveries).
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


class ClonalBCMDualStrategy(PaletteEvolutionStrategy):
    """Clonal hybrid base with BCM metaplastic capture threshold.

    Hybrid combining:
    - Clonal Hybrid (91): Tag+Homeostatic+Clonal
    - BCM Metaplasticity (15): Sliding threshold based on activity

    Critical interaction: Capture threshold rises with many captures, drops
    with few. Creates self-regulating protection.
    """

    name = "clonal_bcm_dual"
    description = "Dual: Clonal hybrid with BCM sliding capture threshold"

    def __init__(
        self,
        # === BCM metaplasticity parameters (from strategy 15) ===
        sliding_threshold_min: float = 0.35,      # Minimum capture threshold
        sliding_threshold_max: float = 0.70,      # Maximum capture threshold
        threshold_adaptation_rate: float = 0.05,  # How fast threshold adapts
        bcm_affinity_coupling: float = 0.3,       # How affinity affects threshold
        capture_rate_window: int = 10,            # Window for capture rate calc
        target_capture_rate: float = 0.1,         # Target: 1 capture per 10 gens
        # === Clonal selection parameters (from 91) ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        proliferation_rate: float = 0.25,
        hypermutation_rate: float = 0.08,
        hypermutation_strength: float = 0.2,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Tagging parameters (from 91) ===
        tag_threshold: float = 0.5,  # This becomes the initial BCM threshold
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        captured_hypermutation_protection: float = 0.9,
        # === Homeostatic parameters (from 91) ===
        target_extreme_ratio: float = 0.60,
        imbalance_threshold: float = 0.15,
        discovery_bonus: float = 0.5,
        extreme_protection: float = 0.6,
        sin_protection: float = 0.5,
        # === Cross-domain parameters ===
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Clonal+BCM hybrid strategy."""
        # BCM parameters
        self.sliding_threshold_min = sliding_threshold_min
        self.sliding_threshold_max = sliding_threshold_max
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.bcm_affinity_coupling = bcm_affinity_coupling
        self.capture_rate_window = capture_rate_window
        self.target_capture_rate = target_capture_rate

        # Clonal selection
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.proliferation_rate = proliferation_rate
        self.hypermutation_rate = hypermutation_rate
        self.hypermutation_strength = hypermutation_strength
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Tagging (initial threshold is BCM starting point)
        self.initial_tag_threshold = tag_threshold
        self.initial_agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost
        self.captured_hypermutation_protection = captured_hypermutation_protection

        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.imbalance_threshold = imbalance_threshold
        self.discovery_bonus = discovery_bonus
        self.extreme_protection = extreme_protection
        self.sin_protection = sin_protection

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with clonal + BCM sliding threshold."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Clonal selection state: affinities
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

        # BCM sliding threshold state
        act_capture_threshold = self.initial_tag_threshold
        agg_capture_threshold = self.initial_agg_tag_threshold
        capture_history = []  # Recent capture events (0 or 1 per gen)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'generation': 0,
            'best_fitness': 0.0,
            'stagnation_counter': 0,
            # Clonal state
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging state
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'act_tag_gens': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'agg_tag_gens': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
            # BCM state
            'act_capture_threshold': act_capture_threshold,
            'agg_capture_threshold': agg_capture_threshold,
            'capture_history': capture_history,
        }

    def _update_bcm_threshold(
        self,
        current_threshold: float,
        capture_history: List[int],
        domain: str = 'act'
    ) -> float:
        """Update BCM sliding threshold based on capture rate.

        If capturing too much → raise threshold (harder to capture)
        If capturing too little → lower threshold (easier to capture)
        """
        if len(capture_history) < 3:
            return current_threshold

        # Calculate recent capture rate
        recent = capture_history[-self.capture_rate_window:]
        capture_rate = sum(recent) / len(recent)

        # Compare to target
        rate_diff = capture_rate - self.target_capture_rate

        # Adjust threshold
        if rate_diff > 0:
            # Too many captures → raise threshold
            new_threshold = current_threshold + self.threshold_adaptation_rate * rate_diff * 2
        else:
            # Too few captures → lower threshold
            new_threshold = current_threshold + self.threshold_adaptation_rate * rate_diff

        # Clamp to bounds
        min_thresh = self.sliding_threshold_min
        max_thresh = self.sliding_threshold_max
        new_threshold = max(min_thresh, min(max_thresh, new_threshold))

        return new_threshold

    def update(
        self,
        state: Dict[str, Any],
        fitness_scores: jnp.ndarray,
        function_usage: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update with clonal + BCM sliding threshold."""
        generation = state['generation']
        best_fitness = float(jnp.max(fitness_scores))
        mean_fitness = float(jnp.mean(fitness_scores))

        # Update fitness tracking
        improved = best_fitness > state['best_fitness'] + 1e-6
        state['best_fitness'] = max(state['best_fitness'], best_fitness)
        state['stagnation_counter'] = 0 if improved else state['stagnation_counter'] + 1

        # Update affinities from fitness
        act_affinities = state['act_affinities'] * self.affinity_decay
        agg_affinities = state['agg_affinities'] * self.affinity_decay

        act_indices = mask_to_indices(state['act_mask'])
        agg_indices = mask_to_indices(state['agg_mask'])
        fitness_delta = best_fitness - state.get('prev_best_fitness', 0.0)

        if improved and fitness_delta > 0:
            for i in act_indices:
                if 0 <= i < NUM_ACTIVATIONS:
                    boost = self.act_affinity_lr * fitness_delta * 10
                    act_affinities = act_affinities.at[i].set(
                        min(1.0, float(act_affinities[i]) + boost)
                    )
            for i in agg_indices:
                if 0 <= i < NUM_AGGREGATIONS:
                    boost = self.agg_affinity_lr * fitness_delta * 10
                    agg_affinities = agg_affinities.at[i].set(
                        min(1.0, float(agg_affinities[i]) + boost)
                    )

        state['act_affinities'] = act_affinities
        state['agg_affinities'] = agg_affinities
        state['prev_best_fitness'] = best_fitness

        # Decay tags
        act_tags = state['act_tags'] * self.tag_decay
        agg_tags = state['agg_tags'] * self.tag_decay

        # Get current BCM thresholds
        act_threshold = state['act_capture_threshold']
        agg_threshold = state['agg_capture_threshold']

        # Update tags using BCM-adjusted thresholds
        for i in range(NUM_ACTIVATIONS):
            # BCM coupling: high affinity can lower effective threshold
            effective_threshold = act_threshold * (1 - self.bcm_affinity_coupling * float(act_affinities[i]))
            effective_threshold = max(self.sliding_threshold_min, effective_threshold)

            if float(act_affinities[i]) > effective_threshold:
                act_tags = act_tags.at[i].set(
                    min(1.0, float(act_tags[i]) + 0.2)
                )

        for i in range(NUM_AGGREGATIONS):
            effective_threshold = agg_threshold * (1 - self.bcm_affinity_coupling * float(agg_affinities[i]))
            if i in CORE_EXTREME_AGGS:
                effective_threshold *= 0.8  # Easier for max/min
            effective_threshold = max(self.sliding_threshold_min * 0.8, effective_threshold)

            if float(agg_affinities[i]) > effective_threshold:
                boost = 0.2 * (self.extreme_tag_boost if i in CORE_EXTREME_AGGS else 1.0)
                agg_tags = agg_tags.at[i].set(min(1.0, float(agg_tags[i]) + boost))

        # Track tag duration
        act_tag_gens = state['act_tag_gens']
        agg_tag_gens = state['agg_tag_gens']
        for i in range(NUM_ACTIVATIONS):
            if float(act_tags[i]) > 0.3:
                act_tag_gens = act_tag_gens.at[i].set(int(act_tag_gens[i]) + 1)
            else:
                act_tag_gens = act_tag_gens.at[i].set(0)
        for i in range(NUM_AGGREGATIONS):
            if float(agg_tags[i]) > 0.3:
                agg_tag_gens = agg_tag_gens.at[i].set(int(agg_tag_gens[i]) + 1)
            else:
                agg_tag_gens = agg_tag_gens.at[i].set(0)

        # Capture mechanism
        act_captured = state['act_captured']
        agg_captured = state['agg_captured']
        capture_count = 0

        for i in range(NUM_ACTIVATIONS):
            if float(act_captured[i]) < 0.5 and int(act_tag_gens[i]) >= self.capture_window:
                act_captured = act_captured.at[i].set(1.0)
                capture_count += 1

        for i in range(NUM_AGGREGATIONS):
            window = self.capture_window - 1 if i in CORE_EXTREME_AGGS else self.capture_window
            if float(agg_captured[i]) < 0.5 and int(agg_tag_gens[i]) >= window:
                agg_captured = agg_captured.at[i].set(1.0)
                capture_count += 1

        # Update capture history for BCM
        capture_history = list(state.get('capture_history', []))
        capture_history.append(1 if capture_count > 0 else 0)
        if len(capture_history) > self.capture_rate_window * 2:
            capture_history = capture_history[-self.capture_rate_window * 2:]

        # Update BCM sliding thresholds
        act_threshold = self._update_bcm_threshold(act_threshold, capture_history, 'act')
        agg_threshold = self._update_bcm_threshold(agg_threshold, capture_history, 'agg')

        state['act_tags'] = act_tags
        state['agg_tags'] = agg_tags
        state['act_tag_gens'] = act_tag_gens
        state['agg_tag_gens'] = agg_tag_gens
        state['act_captured'] = act_captured
        state['agg_captured'] = agg_captured
        state['act_capture_threshold'] = act_threshold
        state['agg_capture_threshold'] = agg_threshold
        state['capture_history'] = capture_history
        state['generation'] = generation + 1

        return state

    def mutate(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any],
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], jax.random.PRNGKey]:
        """Mutate with clonal mechanisms - captured protected from hypermutation."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        act_captured = state['act_captured']
        agg_captured = state['agg_captured']

        rng_key, k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 7)

        act_indices = mask_to_indices(act_mask)
        agg_indices = mask_to_indices(agg_mask)

        # Activation mutations
        if jax.random.uniform(k1) < self.hypermutation_rate:
            inactive_acts = [i for i in range(NUM_ACTIVATIONS)
                           if float(act_mask[i]) < 0.5]
            if inactive_acts:
                weights = [float(act_affinities[i]) + 0.1 for i in inactive_acts]
                total = sum(weights)
                weights = [w / total for w in weights]
                idx = jax.random.choice(k2, jnp.array(inactive_acts),
                                       p=jnp.array(weights))
                act_mask = act_mask.at[int(idx)].set(1.0)

        # Remove low-affinity (but not captured)
        if jax.random.uniform(k3) < self.hypermutation_rate and len(act_indices) > self.min_active_act:
            candidates = [i for i in act_indices
                         if float(act_captured[i]) < self.captured_hypermutation_protection
                         and i != 4]
            if candidates:
                affs = [(i, float(act_affinities[i])) for i in candidates]
                affs.sort(key=lambda x: x[1])
                to_remove = affs[0][0]
                act_mask = act_mask.at[to_remove].set(0.0)

        # Aggregation mutations
        if jax.random.uniform(k4) < self.hypermutation_rate:
            inactive_aggs = [i for i in range(NUM_AGGREGATIONS)
                           if float(agg_mask[i]) < 0.5]
            if inactive_aggs:
                weights = []
                for i in inactive_aggs:
                    w = float(agg_affinities[i]) + 0.1
                    if i in CORE_EXTREME_AGGS:
                        w *= 2.0
                    weights.append(w)
                total = sum(weights)
                weights = [w / total for w in weights]
                idx = jax.random.choice(k5, jnp.array(inactive_aggs),
                                       p=jnp.array(weights))
                agg_mask = agg_mask.at[int(idx)].set(1.0)

        # Remove low-affinity agg (but not captured)
        if jax.random.uniform(k6) < self.hypermutation_rate and len(agg_indices) > self.min_active_agg:
            candidates = [i for i in agg_indices
                         if float(agg_captured[i]) < self.captured_hypermutation_protection
                         and i not in CORE_EXTREME_AGGS]
            if candidates:
                affs = [(i, float(agg_affinities[i])) for i in candidates]
                affs.sort(key=lambda x: x[1])
                to_remove = affs[0][0]
                agg_mask = agg_mask.at[to_remove].set(0.0)

        # Ensure constraints
        act_active = int(jnp.sum(act_mask))
        agg_active = int(jnp.sum(agg_mask))

        if act_active < self.min_active_act:
            inactive = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if inactive:
                rng_key, k = jax.random.split(rng_key)
                idx = jax.random.choice(k, jnp.array(inactive))
                act_mask = act_mask.at[int(idx)].set(1.0)

        if agg_active < self.min_active_agg:
            inactive = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if inactive:
                rng_key, k = jax.random.split(rng_key)
                idx = jax.random.choice(k, jnp.array(inactive))
                agg_mask = agg_mask.at[int(idx)].set(1.0)

        state['act_mask'] = act_mask
        state['agg_mask'] = agg_mask

        return state, rng_key

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Get current activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Get current aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update after each generation - wrapper for compatibility."""
        fitness_scores = jnp.array([best_fitness])
        state = self.update(state, fitness_scores, {}, {})
        rng_key = state.get('rng_key', jax.random.PRNGKey(generation))
        state, _ = self.mutate(state, {}, rng_key)
        metrics = self.get_diagnostics(state)
        return state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return summary for logging."""
        return self.get_diagnostics(state)

    def get_diagnostics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy diagnostics."""
        act_indices = mask_to_indices(state['act_mask'])
        agg_indices = mask_to_indices(state['agg_mask'])

        act_captured_list = [i for i in range(NUM_ACTIVATIONS)
                           if float(state['act_captured'][i]) > 0.5]
        agg_captured_list = [i for i in range(NUM_AGGREGATIONS)
                           if float(state['agg_captured'][i]) > 0.5]

        capture_history = state.get('capture_history', [])
        recent_capture_rate = (
            sum(capture_history[-10:]) / len(capture_history[-10:])
            if capture_history else 0.0
        )

        return {
            'generation': state['generation'],
            'act_palette_size': len(act_indices),
            'agg_palette_size': len(agg_indices),
            'has_sin': 4 in act_indices,
            'has_max': 2 in agg_indices,
            'has_min': 3 in agg_indices,
            'act_captured': act_captured_list,
            'agg_captured': agg_captured_list,
            'sin_captured': 4 in act_captured_list,
            'max_captured': 2 in agg_captured_list,
            'min_captured': 3 in agg_captured_list,
            'sin_affinity': float(state['act_affinities'][4]),
            'max_affinity': float(state['agg_affinities'][2]),
            # BCM-specific
            'act_capture_threshold': state['act_capture_threshold'],
            'agg_capture_threshold': state['agg_capture_threshold'],
            'recent_capture_rate': recent_capture_rate,
        }
