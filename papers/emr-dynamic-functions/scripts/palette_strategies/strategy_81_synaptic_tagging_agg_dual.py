"""Strategy 81: Synaptic Tagging Aggregation Dual (Cross-Domain Tag-and-Capture).

Bio inspiration: Synaptic tagging and capture - tags mark activated synapses,
plasticity-related proteins capture tags for consolidation. Two-stage learning
decouples discovery from immediate reward.

Key innovation:
- Cross-domain tagging: both activation and aggregation must be tagged
- Extreme aggregation tag boost (1.3x)
- Capture requires both domains tagged for cross-domain consolidation

Expected: Better cross-domain synergy through coordinated tag-and-capture.
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


class SynapticTaggingAggDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with cross-domain synaptic tagging.

    Uses two-stage learning where functions are tagged when active,
    then captured (consolidated) when fitness improves significantly.
    Cross-domain capture requires both activation and aggregation tagged.
    """

    name = "synaptic_tagging_agg_dual"
    description = "Dual: Cross-domain synaptic tagging with extreme aggregation boost"

    def __init__(
        self,
        # Tagging parameters
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,  # Lower for aggregations
        tag_decay: float = 0.9,
        agg_tag_decay: float = 0.85,  # Slower decay for aggregations
        # Capture parameters
        capture_window: int = 5,
        agg_capture_window: int = 8,  # Wider for aggregations
        fitness_delta_threshold: float = 0.01,
        # Extreme aggregation boost
        extreme_tag_boost: float = 1.3,
        # Mutation parameters
        base_activate_rate: float = 0.15,
        base_deactivate_rate: float = 0.10,
        captured_protection: float = 0.8,  # 80% reduction if captured
        # Stagnation
        stagnation_threshold: int = 5,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
        # Alias for compatibility with test scripts
        initial_palette: List[int] = None,
    ):
        """Initialize Synaptic Tagging Aggregation Dual strategy."""
        # Handle initial_palette alias
        if initial_palette is not None and initial_act_palette is None:
            initial_act_palette = initial_palette

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.agg_tag_decay = agg_tag_decay

        # Capture
        self.capture_window = capture_window
        self.agg_capture_window = agg_capture_window
        self.fitness_delta_threshold = fitness_delta_threshold

        # Extreme boost
        self.extreme_tag_boost = extreme_tag_boost

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.captured_protection = captured_protection

        # Stagnation
        self.stagnation_threshold = stagnation_threshold

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with tagging and capture tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Tags (strength 0-1)
            'act_tags': jnp.zeros(NUM_ACTIVATIONS),
            'agg_tags': jnp.zeros(NUM_AGGREGATIONS),
            # Tag generation (when tagged)
            'act_tag_gen': jnp.full(NUM_ACTIVATIONS, -1),
            'agg_tag_gen': jnp.full(NUM_AGGREGATIONS, -1),
            # Captured (consolidated)
            'act_captured': jnp.zeros(NUM_ACTIVATIONS),
            'agg_captured': jnp.zeros(NUM_AGGREGATIONS),
            # Cross-domain captured pairs
            'cross_captured': jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS)),
            # Capture events
            'capture_events': 0,
            'cross_capture_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 810000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_tags(
        self,
        tags: jnp.ndarray,
        tag_gen: jnp.ndarray,
        mask: jnp.ndarray,
        generation: int,
        decay: float,
        threshold: float,
        is_agg: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags based on current activity."""
        # Decay existing tags
        new_tags = tags * decay

        # Tag active functions
        new_tag_gen = tag_gen.copy()
        for i in range(len(mask)):
            if mask[i] > 0.5:
                # Boost for extreme aggregations
                if is_agg and i in CORE_EXTREME_AGGS:
                    tag_strength = threshold * self.extreme_tag_boost
                else:
                    tag_strength = threshold

                # Set tag if not already tagged or refresh
                if new_tags[i] < tag_strength:
                    new_tags = new_tags.at[i].set(tag_strength)
                    new_tag_gen = new_tag_gen.at[i].set(generation)

        return jnp.clip(new_tags, 0.0, 1.0), new_tag_gen

    def _attempt_capture(
        self,
        tags: jnp.ndarray,
        tag_gen: jnp.ndarray,
        captured: jnp.ndarray,
        generation: int,
        capture_window: int,
        threshold: float,
    ) -> Tuple[jnp.ndarray, List[int]]:
        """Attempt to capture tagged functions within window."""
        new_captured = captured.copy()
        captured_indices = []

        for i in range(len(tags)):
            if captured[i] < 0.5:  # Not already captured
                tag_age = generation - tag_gen[i]
                if tags[i] >= threshold and 0 <= tag_age <= capture_window:
                    new_captured = new_captured.at[i].set(1.0)
                    captured_indices.append(i)

        return new_captured, captured_indices

    def _attempt_cross_capture(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_tag_gen: jnp.ndarray,
        agg_tag_gen: jnp.ndarray,
        cross_captured: jnp.ndarray,
        generation: int,
    ) -> Tuple[jnp.ndarray, int]:
        """Attempt cross-domain capture requiring both domains tagged."""
        new_cross = cross_captured.copy()
        n_captured = 0

        for i in range(NUM_ACTIVATIONS):
            act_tag_age = generation - act_tag_gen[i]
            act_valid = (
                act_tags[i] >= self.tag_threshold and
                0 <= act_tag_age <= self.capture_window
            )

            if not act_valid:
                continue

            for j in range(NUM_AGGREGATIONS):
                if cross_captured[i, j] > 0.5:
                    continue

                agg_tag_age = generation - agg_tag_gen[j]
                agg_valid = (
                    agg_tags[j] >= self.agg_tag_threshold and
                    0 <= agg_tag_age <= self.agg_capture_window
                )

                if agg_valid:
                    new_cross = new_cross.at[i, j].set(1.0)
                    n_captured += 1

        return new_cross, n_captured

    def _mutate_palette_tagging(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        captured: jnp.ndarray,
        cross_captured: jnp.ndarray,
        is_act: bool,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with capture protection."""
        n_funcs = NUM_ACTIVATIONS if is_act else NUM_AGGREGATIONS
        min_active = self.min_active_act if is_act else self.min_active_agg
        max_active = self.max_active_act if is_act else self.max_active_agg

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(n_funcs):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            # Check protection
            is_captured = captured[i] > 0.5
            if is_act:
                has_cross_capture = jnp.any(cross_captured[i, :] > 0.5)
            else:
                has_cross_capture = jnp.any(cross_captured[:, i] > 0.5)

            is_protected = is_captured or has_cross_capture

            if mask[i] < 0.5:  # Inactive
                if p < self.base_activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                if is_protected:
                    deactivate_rate = self.base_deactivate_rate * (1 - self.captured_protection)
                else:
                    deactivate_rate = self.base_deactivate_rate

                # Extra protection for sin and extreme aggs
                if is_act and i == 4:
                    deactivate_rate *= 0.5
                elif not is_act and i in CORE_EXTREME_AGGS:
                    deactivate_rate *= 0.5

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < min_active or active_count > max_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with tagging and capture mechanism."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update tags
        new_act_tags, new_act_tag_gen = self._update_tags(
            state['act_tags'], state['act_tag_gen'], state['act_mask'],
            generation, self.tag_decay, self.tag_threshold, is_agg=False
        )
        new_agg_tags, new_agg_tag_gen = self._update_tags(
            state['agg_tags'], state['agg_tag_gen'], state['agg_mask'],
            generation, self.agg_tag_decay, self.agg_tag_threshold, is_agg=True
        )

        # Attempt capture on significant fitness improvement
        new_act_captured = state['act_captured']
        new_agg_captured = state['agg_captured']
        new_cross_captured = state['cross_captured']
        act_captured_indices = []
        agg_captured_indices = []
        n_cross_captured = 0
        new_capture_events = state['capture_events']
        new_cross_capture_events = state['cross_capture_events']

        if fitness_delta >= self.fitness_delta_threshold:
            new_act_captured, act_captured_indices = self._attempt_capture(
                new_act_tags, new_act_tag_gen, state['act_captured'],
                generation, self.capture_window, self.tag_threshold
            )
            new_agg_captured, agg_captured_indices = self._attempt_capture(
                new_agg_tags, new_agg_tag_gen, state['agg_captured'],
                generation, self.agg_capture_window, self.agg_tag_threshold
            )
            new_cross_captured, n_cross_captured = self._attempt_cross_capture(
                new_act_tags, new_agg_tags,
                new_act_tag_gen, new_agg_tag_gen,
                state['cross_captured'], generation
            )

            new_capture_events += len(act_captured_indices) + len(agg_captured_indices)
            new_cross_capture_events += n_cross_captured

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_palette_tagging(
                k_act, state['act_mask'], new_act_captured, new_cross_captured, True
            )
            new_agg_mask, agg_mutation_info = self._mutate_palette_tagging(
                k_agg, state['agg_mask'], new_agg_captured, new_cross_captured, False
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            new_stagnation = 0

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_tag_gen': new_act_tag_gen,
            'agg_tag_gen': new_agg_tag_gen,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'cross_captured': new_cross_captured,
            'capture_events': new_capture_events,
            'cross_capture_events': new_cross_capture_events,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
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
            'mutation_triggered': act_mutation_info is not None,
            # Tagging metrics
            'act_n_tagged': int(jnp.sum(new_act_tags >= self.tag_threshold)),
            'agg_n_tagged': int(jnp.sum(new_agg_tags >= self.agg_tag_threshold)),
            'sin_tag_strength': float(new_act_tags[4]),
            'max_tag_strength': float(new_agg_tags[2]),
            'min_tag_strength': float(new_agg_tags[3]),
            # Capture metrics
            'act_n_captured': int(jnp.sum(new_act_captured > 0.5)),
            'agg_n_captured': int(jnp.sum(new_agg_captured > 0.5)),
            'cross_n_captured': int(jnp.sum(new_cross_captured > 0.5)),
            'capture_events': new_capture_events,
            'cross_capture_events': new_cross_capture_events,
            'act_captured_this_gen': act_captured_indices,
            'agg_captured_this_gen': agg_captured_indices,
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_captured': new_act_captured[4] > 0.5,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'max_captured': new_agg_captured[2] > 0.5,
            'min_captured': new_agg_captured[3] > 0.5,
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with tagging status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'sin_tag_strength': float(state['act_tags'][4]),
            'max_tag_strength': float(state['agg_tags'][2]),
            'min_tag_strength': float(state['agg_tags'][3]),
            'sin_captured': state['act_captured'][4] > 0.5,
            'max_captured': state['agg_captured'][2] > 0.5,
            'min_captured': state['agg_captured'][3] > 0.5,
            'act_n_captured': int(jnp.sum(state['act_captured'] > 0.5)),
            'agg_n_captured': int(jnp.sum(state['agg_captured'] > 0.5)),
            'cross_n_captured': int(jnp.sum(state['cross_captured'] > 0.5)),
            'capture_events': state['capture_events'],
            'cross_capture_events': state['cross_capture_events'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
