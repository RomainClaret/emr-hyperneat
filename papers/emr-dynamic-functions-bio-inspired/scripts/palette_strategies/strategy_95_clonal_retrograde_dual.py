"""Strategy 95: Clonal Retrograde Dual.

Combines clonal_hybrid_dual (91) with retrograde signaling (37):
- Base: Strategy 91 (Clonal+Tag+Homeostatic) - 100% Parity-5, 100% sin retention
- Extension: Strategy 37 (Retrograde Signaling) - Backward credit assignment

Key innovation: When a function gets captured, retrograde signals boost the
"recently co-active" functions, creating capture cascades. This builds stable
"discovery pathways" that can be reactivated when similar patterns recur.

Bio inspiration: Retrograde signaling (endocannabinoids, nitric oxide) allows
postsynaptic neurons to influence presynaptic activity. Combined with clonal
capture, this creates temporal credit chains for function discovery.

Expected: Stable discovery pathways that create coordinated capture groups.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque
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


class ClonalRetrogradeDualStrategy(PaletteEvolutionStrategy):
    """Clonal hybrid base with retrograde signaling extension.

    Hybrid combining:
    - Clonal Hybrid (91): Tag+Homeostatic+Clonal
    - Retrograde Signaling (37): Backward credit through co-activation chains

    Critical interaction: When function captured, retrograde signals boost
    co-active functions, creating cascade captures.
    """

    name = "clonal_retrograde_dual"
    description = "Dual: Clonal hybrid with retrograde capture cascades"

    def __init__(
        self,
        # === Retrograde signaling parameters (from strategy 37) ===
        trace_decay: float = 0.7,               # Decay per step in chain
        capture_retrograde_boost: float = 0.3,  # Tag boost from capture cascade
        cascade_capture_window: int = 3,        # Generations to look back for cascade
        retrograde_affinity_boost: float = 0.15,  # Affinity boost from retrograde
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
        tag_threshold: float = 0.5,
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
        """Initialize Clonal+Retrograde hybrid strategy."""
        # Retrograde signaling
        self.trace_decay = trace_decay
        self.capture_retrograde_boost = capture_retrograde_boost
        self.cascade_capture_window = cascade_capture_window
        self.retrograde_affinity_boost = retrograde_affinity_boost

        # Clonal selection
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.proliferation_rate = proliferation_rate
        self.hypermutation_rate = hypermutation_rate
        self.hypermutation_strength = hypermutation_strength
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
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
        """Initialize state with clonal + retrograde tracking."""
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

        # Retrograde signaling state: history of active palettes
        # Store as list of (act_indices, agg_indices) tuples
        act_history = []  # List of active act indices per generation
        agg_history = []  # List of active agg indices per generation

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
            # Retrograde state
            'act_history': act_history,
            'agg_history': agg_history,
        }

    def update(
        self,
        state: Dict[str, Any],
        fitness_scores: jnp.ndarray,
        function_usage: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update with clonal + retrograde mechanisms."""
        generation = state['generation']
        best_fitness = float(jnp.max(fitness_scores))
        mean_fitness = float(jnp.mean(fitness_scores))

        # Update fitness tracking
        improved = best_fitness > state['best_fitness'] + 1e-6
        state['best_fitness'] = max(state['best_fitness'], best_fitness)
        state['stagnation_counter'] = 0 if improved else state['stagnation_counter'] + 1

        # Get current active functions
        act_indices = mask_to_indices(state['act_mask'])
        agg_indices = mask_to_indices(state['agg_mask'])

        # Store in retrograde history
        act_history = list(state.get('act_history', []))
        agg_history = list(state.get('agg_history', []))
        act_history.append(list(act_indices))
        agg_history.append(list(agg_indices))
        # Keep limited history
        if len(act_history) > self.cascade_capture_window + 2:
            act_history = act_history[-(self.cascade_capture_window + 2):]
            agg_history = agg_history[-(self.cascade_capture_window + 2):]

        # Update affinities from fitness
        act_affinities = state['act_affinities'] * self.affinity_decay
        agg_affinities = state['agg_affinities'] * self.affinity_decay

        fitness_delta = best_fitness - state.get('prev_best_fitness', 0.0)

        if improved and fitness_delta > 0:
            # Boost active functions
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

        # Update tags from high-affinity functions
        for i in range(NUM_ACTIVATIONS):
            if float(act_affinities[i]) > self.tag_threshold:
                act_tags = act_tags.at[i].set(
                    min(1.0, float(act_tags[i]) + 0.2)
                )
        for i in range(NUM_AGGREGATIONS):
            aff = float(agg_affinities[i])
            threshold = self.agg_tag_threshold
            if i in CORE_EXTREME_AGGS:
                threshold *= 0.8  # Easier capture for max/min
            if aff > threshold:
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

        # Capture mechanism with retrograde cascade
        act_captured = state['act_captured']
        agg_captured = state['agg_captured']
        new_act_captures = []
        new_agg_captures = []

        for i in range(NUM_ACTIVATIONS):
            if float(act_captured[i]) < 0.5 and int(act_tag_gens[i]) >= self.capture_window:
                act_captured = act_captured.at[i].set(1.0)
                new_act_captures.append(i)

        for i in range(NUM_AGGREGATIONS):
            window = self.capture_window - 1 if i in CORE_EXTREME_AGGS else self.capture_window
            if float(agg_captured[i]) < 0.5 and int(agg_tag_gens[i]) >= window:
                agg_captured = agg_captured.at[i].set(1.0)
                new_agg_captures.append(i)

        # === RETROGRADE CASCADE ===
        # When a function is captured, boost co-active functions from recent history
        if new_act_captures or new_agg_captures:
            # Look back through history
            for t, (hist_acts, hist_aggs) in enumerate(zip(
                reversed(act_history[:-1]), reversed(agg_history[:-1])
            )):
                if t >= self.cascade_capture_window:
                    break
                temporal_weight = self.trace_decay ** (t + 1)

                # Boost tags of co-active functions
                for captured_act in new_act_captures:
                    for hist_act in hist_acts:
                        if hist_act != captured_act and 0 <= hist_act < NUM_ACTIVATIONS:
                            boost = self.capture_retrograde_boost * temporal_weight
                            act_tags = act_tags.at[hist_act].set(
                                min(1.0, float(act_tags[hist_act]) + boost)
                            )
                            # Also boost affinity
                            aff_boost = self.retrograde_affinity_boost * temporal_weight
                            act_affinities = act_affinities.at[hist_act].set(
                                min(1.0, float(act_affinities[hist_act]) + aff_boost)
                            )

                for captured_agg in new_agg_captures:
                    for hist_agg in hist_aggs:
                        if hist_agg != captured_agg and 0 <= hist_agg < NUM_AGGREGATIONS:
                            boost = self.capture_retrograde_boost * temporal_weight
                            agg_tags = agg_tags.at[hist_agg].set(
                                min(1.0, float(agg_tags[hist_agg]) + boost)
                            )
                            aff_boost = self.retrograde_affinity_boost * temporal_weight
                            agg_affinities = agg_affinities.at[hist_agg].set(
                                min(1.0, float(agg_affinities[hist_agg]) + aff_boost)
                            )

        state['act_tags'] = act_tags
        state['agg_tags'] = agg_tags
        state['act_tag_gens'] = act_tag_gens
        state['agg_tag_gens'] = agg_tag_gens
        state['act_captured'] = act_captured
        state['agg_captured'] = agg_captured
        state['act_affinities'] = act_affinities
        state['agg_affinities'] = agg_affinities
        state['act_history'] = act_history
        state['agg_history'] = agg_history
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

        # Proliferation: high-affinity functions more likely to stay
        act_indices = mask_to_indices(act_mask)
        agg_indices = mask_to_indices(agg_mask)

        # Activation mutations
        if jax.random.uniform(k1) < self.hypermutation_rate:
            inactive_acts = [i for i in range(NUM_ACTIVATIONS)
                           if float(act_mask[i]) < 0.5]
            if inactive_acts:
                # Weight by affinity
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
                         and i != 4]  # Never remove sin if captured
            if candidates:
                # Remove lowest affinity
                affs = [(i, float(act_affinities[i])) for i in candidates]
                affs.sort(key=lambda x: x[1])
                to_remove = affs[0][0]
                act_mask = act_mask.at[to_remove].set(0.0)

        # Aggregation mutations
        if jax.random.uniform(k4) < self.hypermutation_rate:
            inactive_aggs = [i for i in range(NUM_AGGREGATIONS)
                           if float(agg_mask[i]) < 0.5]
            if inactive_aggs:
                # Bias toward extreme aggs
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
        # Create fitness scores array from best_fitness
        fitness_scores = jnp.array([best_fitness])

        # Call internal update
        state = self.update(state, fitness_scores, {}, {})

        # Call internal mutate
        rng_key = state.get('rng_key', jax.random.PRNGKey(generation))
        state, _ = self.mutate(state, {}, rng_key)

        # Return state and metrics
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
            'history_length': len(state.get('act_history', [])),
        }
