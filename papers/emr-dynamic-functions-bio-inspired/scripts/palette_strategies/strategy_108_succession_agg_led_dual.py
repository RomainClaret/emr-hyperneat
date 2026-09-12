"""Strategy 108: Succession + Aggregation-Led Dual.

Combines ecological succession (strategy 26) with aggregation-led dual discovery
(strategy 101) for phased exploration with extreme-led discovery.

Key Innovation:
- PIONEER phase: Explores broadly, discovers extremes aggressively
- INTERMEDIATE phase: Balanced exploration, captures sin-extreme pairings
- CLIMAX phase: Protects discovered sin-extreme, minimal exploration
- Each phase has different exploration rates and protection levels

Bio inspiration: Ecological succession from pioneer → climax communities.
Early evolution explores broadly, late evolution protects what works.
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
    CORE_EXTREME_AGGS,
)


class SuccessionAggLedDualStrategy(PaletteEvolutionStrategy):
    """Phased discovery with aggregation-led progression.

    Ecological succession phases guide exploration.
    Pioneer explores, intermediate discovers sin-extreme, climax protects.
    """

    name = "succession_agg_led_dual"
    description = "Dual: Pioneer→Intermediate→Climax phases with aggregation-led discovery"

    def __init__(
        self,
        # === SUCCESSION PHASE PARAMETERS (KEY INNOVATION) ===
        pioneer_end: int = 10,                    # End of pioneer phase
        intermediate_end: int = 30,               # End of intermediate phase
        # === Pioneer phase ===
        pioneer_agg_exploration: float = 0.30,    # High aggregation exploration
        pioneer_act_exploration: float = 0.25,
        pioneer_sin_bias: float = 0.3,            # Moderate sin preference in pioneer
        # === Intermediate phase ===
        intermediate_exploration: float = 0.15,
        intermediate_capture_boost: float = 1.3,   # Boost capture during intermediate
        # === Climax phase ===
        climax_extreme_protection: float = 0.90,  # Strong protection for discovered
        climax_exploration: float = 0.05,         # Minimal exploration
        succession_sin_delay: int = 5,            # Generations before sin bias kicks in
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Tagging parameters ===
        tag_threshold: float = 0.5,
        tag_decay: float = 0.9,
        capture_window: int = 5,
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
        """Initialize Succession Aggregation-Led Dual strategy."""
        # Succession phases (KEY)
        self.pioneer_end = pioneer_end
        self.intermediate_end = intermediate_end
        self.pioneer_agg_exploration = pioneer_agg_exploration
        self.pioneer_act_exploration = pioneer_act_exploration
        self.pioneer_sin_bias = pioneer_sin_bias
        self.intermediate_exploration = intermediate_exploration
        self.intermediate_capture_boost = intermediate_capture_boost
        self.climax_extreme_protection = climax_extreme_protection
        self.climax_exploration = climax_exploration
        self.succession_sin_delay = succession_sin_delay

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Tagging
        self.tag_threshold = tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window

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
        """Initialize state with succession tracking."""
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

        # Tagging
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

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
            # Succession phase tracking
            'current_phase': 'pioneer',
            'discovered_in_intermediate': set(),
            # Stats
            'phase_transitions': 0,
            'intermediate_captures': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1080000),
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

    def _get_phase(self, generation: int) -> str:
        """Determine current succession phase."""
        if generation < self.pioneer_end:
            return 'pioneer'
        elif generation < self.intermediate_end:
            return 'intermediate'
        else:
            return 'climax'

    def _get_phase_params(self, generation: int) -> Dict[str, Any]:
        """Get parameters for current succession phase."""
        phase = self._get_phase(generation)

        if phase == 'pioneer':
            return {
                'phase': 'pioneer',
                'agg_exploration': self.pioneer_agg_exploration,
                'act_exploration': self.pioneer_act_exploration,
                'sin_bias': self.pioneer_sin_bias if generation >= self.succession_sin_delay else 0.1,
                'protection': 0.3,
                'capture_boost': 1.0,
            }
        elif phase == 'intermediate':
            progress = (generation - self.pioneer_end) / (self.intermediate_end - self.pioneer_end)
            return {
                'phase': 'intermediate',
                'agg_exploration': self.intermediate_exploration,
                'act_exploration': self.intermediate_exploration,
                'sin_bias': 0.6 + 0.2 * progress,  # Increasing sin bias
                'protection': 0.5 + 0.2 * progress,
                'capture_boost': self.intermediate_capture_boost,
            }
        else:  # climax
            return {
                'phase': 'climax',
                'agg_exploration': self.climax_exploration,
                'act_exploration': self.climax_exploration,
                'sin_bias': 0.9,
                'protection': self.climax_extreme_protection,
                'capture_boost': 1.0,
            }

    def _update_tags(
        self,
        mask: jnp.ndarray,
        tags: jnp.ndarray,
        phase_params: Dict,
    ) -> jnp.ndarray:
        """Update tags with phase-dependent strength."""
        new_tags = tags * self.tag_decay
        n_funcs = len(tags)

        for i in range(n_funcs):
            if mask[i] > 0.5:
                tag_strength = phase_params['capture_boost']
                new_tags = new_tags.at[i].set(min(1.0, new_tags[i] + tag_strength * 0.3))

        return new_tags

    def _attempt_capture(
        self,
        tags: jnp.ndarray,
        captured: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
        phase: str,
    ) -> Tuple[jnp.ndarray, int]:
        """Attempt capture with phase-dependent thresholds."""
        new_captured = captured.copy()
        capture_count = 0

        if not improved:
            return new_captured, 0

        # Intermediate phase has lower capture threshold
        threshold = self.tag_threshold * (0.8 if phase == 'intermediate' else 1.0)

        for hist_gen, hist_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(len(tags)):
                    if hist_tags[i] > threshold and new_captured[i] < 0.5:
                        new_captured = new_captured.at[i].set(1.0)
                        capture_count += 1

        return new_captured, capture_count

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        phase_params: Dict,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
        is_agg: bool = False,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette with phase-dependent exploration."""
        k1, k2 = jax.random.split(key)

        # Score: affinity + capture + tags
        score = affinities + captured * 0.3 + tags * 0.2

        # Phase-dependent protection for captured
        if phase_params['phase'] == 'climax':
            for i in range(n_funcs):
                if captured[i] > 0.5:
                    score = score.at[i].set(score[i] + phase_params['protection'])

        # Preference boost
        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    # Sin gets special bias based on phase
                    boost = phase_params.get('sin_bias', 0.5) if i == 4 else 0.5
                    score = score.at[i].set(score[i] + boost)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # EXPLORATION: Phase-dependent rate
        exploration = phase_params['agg_exploration'] if is_agg else phase_params['act_exploration']

        if prefer_indices:
            for idx in prefer_indices:
                if 0 <= idx < n_funcs and mask[idx] < 0.5:
                    if float(jax.random.uniform(k1)) < exploration * 1.5:
                        mask = mask.at[idx].set(1.0)
                    k1, _ = jax.random.split(k1)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < min_diversity:
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k2, jnp.array(inactive),
                                          shape=(min(needed, len(inactive)),), replace=False)
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
        """Update with succession-based dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Get phase parameters
        phase_params = self._get_phase_params(generation)
        old_phase = state['current_phase']
        new_phase = phase_params['phase']
        phase_transition = old_phase != new_phase

        # === TAGGING ===
        new_act_tags = self._update_tags(state['act_mask'], state['act_tags'], phase_params)
        new_agg_tags = self._update_tags(state['agg_mask'], state['agg_tags'], phase_params)

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE ===
        new_act_captured, act_captures = self._attempt_capture(
            new_act_tags, state['act_captured'], new_tag_history,
            generation, improved, new_phase
        )
        new_agg_captured, agg_captures = self._attempt_capture(
            new_agg_tags, state['agg_captured'],
            [(g, t) for g, t in state['tag_history'][-5:]],
            generation, improved, new_phase
        )

        # Track intermediate captures
        int_captures = 0
        if new_phase == 'intermediate':
            int_captures = act_captures + agg_captures

        # === AFFINITY UPDATE ===
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * fitness_delta)
                    )
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta)
                    )

        # === PALETTE SELECTION ===
        new_act_mask, _ = self._select_palette(
            new_act_aff, new_act_captured, new_act_tags,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            self.min_diversity_act, phase_params, k1,
            prefer_indices=[4], is_agg=False
        )
        new_agg_mask, _ = self._select_palette(
            new_agg_aff, new_agg_captured, new_agg_tags,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            self.min_diversity_agg, phase_params, k2,
            prefer_indices=list(CORE_EXTREME_AGGS), is_agg=True
        )

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
            'current_phase': new_phase,
            'discovered_in_intermediate': state['discovered_in_intermediate'],
            'phase_transitions': state['phase_transitions'] + (1 if phase_transition else 0),
            'intermediate_captures': state['intermediate_captures'] + int_captures,
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
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Succession metrics (KEY)
            'phase': new_phase,
            'phase_transition': phase_transition,
            'phase_transitions': new_state['phase_transitions'],
            'intermediate_captures': new_state['intermediate_captures'],
            # Capture metrics
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'max_captured': bool(new_agg_captured[2] > 0.5),
            'min_captured': bool(new_agg_captured[3] > 0.5),
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

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'current_phase': state['current_phase'],
            'phase_transitions': state['phase_transitions'],
            'intermediate_captures': state['intermediate_captures'],
            'generation': state['generation'],
        }
