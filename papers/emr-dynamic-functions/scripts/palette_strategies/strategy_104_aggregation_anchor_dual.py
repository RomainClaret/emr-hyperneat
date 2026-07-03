"""Strategy 104: Aggregation Anchor Dual.

Aggregation-first approach with near-permanent aggregation anchoring:
- Problem: Even with good retention mechanisms, extreme aggs (max/min) can still
  be lost during task transitions, preventing later tasks from benefiting
- Hypothesis: Lock aggregations permanently once max/min are discovered, allowing
  ONLY activations to adapt. Creates a stable "landing zone" for sin.

Key mechanisms:
- Aggregation anchor protection: Once max/min captured, 95% protection (almost immune)
- Aggregation hypermutation immunity: Anchored aggs cannot be removed
- Activation freedom: Full mutation flexibility for activations
- Anchor-sin discovery boost: When anchors exist, sin has higher discovery bonus

Bio inspiration: Some neural circuits lock in during development and never change.
The basic sensory pathways (edge detection, motion detection) are permanent, while
higher-level circuits remain plastic. Extreme aggregations are like basic operations.

Expected: Stable max/min anchors create a consistent environment for sin to thrive.
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


class AggregationAnchorDualStrategy(PaletteEvolutionStrategy):
    """Aggregation-first with near-permanent anchoring of extreme aggs.

    Strategy:
    - Once max/min discovered and captured, they become permanent anchors
    - Activations retain full plasticity
    - Anchored extreme aggs create stable environment for sin discovery
    """

    name = "aggregation_anchor_dual"
    description = "Dual: Permanent aggregation anchors with activation plasticity"

    def __init__(
        self,
        # === Aggregation anchor parameters ===
        agg_anchor_protection: float = 0.95,          # Protection for anchored aggs
        agg_anchor_hypermutation_immunity: bool = True,  # Anchored = immune to removal
        anchor_sin_discovery_boost: float = 0.6,      # Boost sin discovery when anchors exist
        anchor_threshold: float = 0.65,               # Affinity threshold to become anchor
        anchor_capture_window: int = 3,               # Faster capture for anchors
        # === Aggregation discovery parameters ===
        agg_exploration_rate: float = 0.20,           # High agg exploration
        act_exploration_rate: float = 0.10,           # Normal activation exploration
        extreme_discovery_bonus: float = 0.4,         # Bonus for discovering max/min
        # === Tag-and-capture parameters ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.40,              # Lower for easier agg capture
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.5,               # Higher for extreme aggs
        # === Affinity learning ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.15,                # Higher agg learning
        affinity_decay: float = 0.98,
        # === Cross-domain parameters ===
        cross_learning_rate: float = 0.10,
        sin_extreme_affinity_boost: float = 0.35,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Aggregation Anchor strategy."""
        # Anchor parameters
        self.agg_anchor_protection = agg_anchor_protection
        self.agg_anchor_hypermutation_immunity = agg_anchor_hypermutation_immunity
        self.anchor_sin_discovery_boost = anchor_sin_discovery_boost
        self.anchor_threshold = anchor_threshold
        self.anchor_capture_window = anchor_capture_window

        # Discovery parameters
        self.agg_exploration_rate = agg_exploration_rate
        self.act_exploration_rate = act_exploration_rate
        self.extreme_discovery_bonus = extreme_discovery_bonus

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

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
        """Initialize state with anchor tracking."""
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

        # Anchor state - anchored aggs are PERMANENTLY protected
        agg_anchored = jnp.zeros(NUM_AGGREGATIONS)  # 1.0 = anchored (permanent)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'generation': 0,
            'best_fitness': 0.0,
            'stagnation_counter': 0,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging state
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'act_tag_gens': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'agg_tag_gens': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
            # Anchor state
            'agg_anchored': agg_anchored,
        }

    def update(
        self,
        state: Dict[str, Any],
        fitness_scores: jnp.ndarray,
        function_usage: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update with aggregation anchoring."""
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
                    # Extra boost for sin if anchors exist
                    if i == 4:  # sin
                        has_anchors = any(
                            float(state['agg_anchored'][j]) > 0.5
                            for j in CORE_EXTREME_AGGS
                        )
                        if has_anchors:
                            boost *= (1 + self.anchor_sin_discovery_boost)
                    act_affinities = act_affinities.at[i].set(
                        min(1.0, float(act_affinities[i]) + boost)
                    )

            for i in agg_indices:
                if 0 <= i < NUM_AGGREGATIONS:
                    boost = self.agg_affinity_lr * fitness_delta * 10
                    # Extra boost for extreme aggs
                    if i in CORE_EXTREME_AGGS:
                        boost *= (1 + self.extreme_discovery_bonus)
                    agg_affinities = agg_affinities.at[i].set(
                        min(1.0, float(agg_affinities[i]) + boost)
                    )

        state['act_affinities'] = act_affinities
        state['agg_affinities'] = agg_affinities
        state['prev_best_fitness'] = best_fitness

        # Decay tags
        act_tags = state['act_tags'] * self.tag_decay
        agg_tags = state['agg_tags'] * self.tag_decay

        # Update tags from high-affinity functions
        for i in range(NUM_ACTIVATIONS):
            if float(act_affinities[i]) > self.tag_threshold:
                act_tags = act_tags.at[i].set(
                    min(1.0, float(act_tags[i]) + 0.2)
                )
        for i in range(NUM_AGGREGATIONS):
            threshold = self.agg_tag_threshold
            if i in CORE_EXTREME_AGGS:
                threshold *= 0.7  # Even easier for extreme aggs
            if float(agg_affinities[i]) > threshold:
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
        agg_anchored = state['agg_anchored']

        for i in range(NUM_ACTIVATIONS):
            if float(act_captured[i]) < 0.5 and int(act_tag_gens[i]) >= self.capture_window:
                act_captured = act_captured.at[i].set(1.0)

        for i in range(NUM_AGGREGATIONS):
            # Faster capture for extreme aggs
            window = self.anchor_capture_window if i in CORE_EXTREME_AGGS else self.capture_window
            if float(agg_captured[i]) < 0.5 and int(agg_tag_gens[i]) >= window:
                agg_captured = agg_captured.at[i].set(1.0)
                # If extreme agg, also anchor it
                if i in CORE_EXTREME_AGGS:
                    agg_anchored = agg_anchored.at[i].set(1.0)

        # Additional anchoring check: high affinity + in palette = anchor
        for i in CORE_EXTREME_AGGS:
            if float(agg_anchored[i]) < 0.5:  # Not yet anchored
                if float(agg_affinities[i]) > self.anchor_threshold and float(state['agg_mask'][i]) > 0.5:
                    agg_anchored = agg_anchored.at[i].set(1.0)
                    agg_captured = agg_captured.at[i].set(1.0)

        state['act_tags'] = act_tags
        state['agg_tags'] = agg_tags
        state['act_tag_gens'] = act_tag_gens
        state['agg_tag_gens'] = agg_tag_gens
        state['act_captured'] = act_captured
        state['agg_captured'] = agg_captured
        state['agg_anchored'] = agg_anchored
        state['generation'] = generation + 1

        return state

    def mutate(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any],
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], jax.random.PRNGKey]:
        """Mutate with aggregation anchoring - anchored aggs are immune."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        act_captured = state['act_captured']
        agg_captured = state['agg_captured']
        agg_anchored = state['agg_anchored']

        rng_key, k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 7)

        act_indices = mask_to_indices(act_mask)
        agg_indices = mask_to_indices(agg_mask)

        # Check if anchors exist for sin discovery boost
        has_anchors = any(float(agg_anchored[i]) > 0.5 for i in CORE_EXTREME_AGGS)

        # === ACTIVATION MUTATIONS (full flexibility) ===
        if jax.random.uniform(k1) < self.act_exploration_rate:
            inactive_acts = [i for i in range(NUM_ACTIVATIONS)
                           if float(act_mask[i]) < 0.5]
            if inactive_acts:
                weights = []
                for i in inactive_acts:
                    w = float(act_affinities[i]) + 0.1
                    # Sin discovery boost when anchors exist
                    if i == 4 and has_anchors:
                        w *= (1 + self.anchor_sin_discovery_boost)
                    weights.append(w)
                total = sum(weights)
                weights = [w / total for w in weights]
                idx = jax.random.choice(k2, jnp.array(inactive_acts),
                                       p=jnp.array(weights))
                act_mask = act_mask.at[int(idx)].set(1.0)

        # Remove low-affinity act (not captured sin)
        if jax.random.uniform(k3) < self.act_exploration_rate and len(act_indices) > self.min_active_act:
            candidates = [i for i in act_indices
                         if float(act_captured[i]) < 0.5
                         and i != 4]  # Never remove sin
            if candidates:
                affs = [(i, float(act_affinities[i])) for i in candidates]
                affs.sort(key=lambda x: x[1])
                to_remove = affs[0][0]
                act_mask = act_mask.at[to_remove].set(0.0)

        # === AGGREGATION MUTATIONS (anchors are immune) ===
        if jax.random.uniform(k4) < self.agg_exploration_rate:
            inactive_aggs = [i for i in range(NUM_AGGREGATIONS)
                           if float(agg_mask[i]) < 0.5]
            if inactive_aggs:
                weights = []
                for i in inactive_aggs:
                    w = float(agg_affinities[i]) + 0.1
                    # Strong bias toward extreme aggs
                    if i in CORE_EXTREME_AGGS:
                        w *= 3.0
                    weights.append(w)
                total = sum(weights)
                weights = [w / total for w in weights]
                idx = jax.random.choice(k5, jnp.array(inactive_aggs),
                                       p=jnp.array(weights))
                agg_mask = agg_mask.at[int(idx)].set(1.0)

        # Remove low-affinity agg (NEVER remove anchored)
        if jax.random.uniform(k6) < self.agg_exploration_rate and len(agg_indices) > self.min_active_agg:
            candidates = []
            for i in agg_indices:
                if self.agg_anchor_hypermutation_immunity and float(agg_anchored[i]) > 0.5:
                    continue  # Anchored = immune
                if float(agg_captured[i]) > self.agg_anchor_protection:
                    continue  # High capture protection
                if i in CORE_EXTREME_AGGS:
                    continue  # Never remove extreme aggs
                candidates.append(i)

            if candidates:
                affs = [(i, float(agg_affinities[i])) for i in candidates]
                affs.sort(key=lambda x: x[1])
                to_remove = affs[0][0]
                agg_mask = agg_mask.at[to_remove].set(0.0)

        # Ensure anchored aggs remain in palette
        for i in CORE_EXTREME_AGGS:
            if float(agg_anchored[i]) > 0.5:
                agg_mask = agg_mask.at[i].set(1.0)

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
        agg_anchored_list = [i for i in range(NUM_AGGREGATIONS)
                           if float(state['agg_anchored'][i]) > 0.5]

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
            # Anchor diagnostics
            'agg_anchored': agg_anchored_list,
            'max_anchored': 2 in agg_anchored_list,
            'min_anchored': 3 in agg_anchored_list,
        }
