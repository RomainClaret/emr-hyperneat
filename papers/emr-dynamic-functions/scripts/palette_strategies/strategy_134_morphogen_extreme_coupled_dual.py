"""Strategy 134: Morphogen-Extreme-Coupled Dual.

Morphogen gradients with sin-extreme coupling: sin source strengthens
when extreme aggregations (max/min) are active.

Key Innovation:
- Sin's morphogen source gets +50% boost when max/min are active
- Creates positive feedback: extreme aggs → stronger sin → sin discovery
- Minimum sin source floor prevents complete loss
- Coupling decay rate allows adaptation

This addresses the root cause: sin needs SYNERGISTIC support from
extreme aggregations, not just independent discovery.
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


class MorphogenExtremeCoupledDualStrategy(PaletteEvolutionStrategy):
    """Morphogen gradients with sin-extreme coupling.

    Sin source strengthens when extreme aggs are
    active, creating synergistic discovery dynamics.
    """

    name = "morphogen_extreme_coupled_dual"
    description = "Dual: Morphogen with sin-extreme coupling boost"

    def __init__(
        self,
        # === COUPLING PARAMETERS (CRITICAL FIX) ===
        sin_extreme_coupling_boost: float = 0.5,
        coupling_decay_rate: float = 0.02,
        min_sin_source_strength: float = 1.0,
        base_sin_source: float = 1.5,
        extreme_agg_source: float = 1.8,
        # === MORPHOGEN PARAMETERS ===
        morphogen_diffusion_rate: float = 0.3,
        morphogen_decay_rate: float = 0.05,
        gradient_threshold: float = 0.4,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_learning_rate: float = 0.12,
        sin_extreme_affinity_coupling: float = 0.5,
        # === TAGGING ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.40,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        extreme_tag_boost: float = 1.5,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 3,
        min_diversity_agg: int = 2,
        # === INITIAL PALETTES ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Morphogen-Extreme-Coupled strategy."""
        # COUPLING (CRITICAL)
        self.sin_extreme_coupling_boost = sin_extreme_coupling_boost
        self.coupling_decay_rate = coupling_decay_rate
        self.min_sin_source_strength = min_sin_source_strength
        self.base_sin_source = base_sin_source
        self.extreme_agg_source = extreme_agg_source

        # Morphogen
        self.morphogen_diffusion_rate = morphogen_diffusion_rate
        self.morphogen_decay_rate = morphogen_decay_rate
        self.gradient_threshold = gradient_threshold

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_coupling = sin_extreme_affinity_coupling

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.extreme_tag_boost = extreme_tag_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial - include sin and extreme aggs
        default_act = list(DEFAULT_PALETTE_INDICES)
        if 4 not in default_act:
            default_act.append(4)
        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        for agg in CORE_EXTREME_AGGS:
            if agg not in default_agg:
                default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with coupled morphogen sources."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        initial_act = list(initial_act)
        if 4 not in initial_act:
            initial_act.append(4)

        initial_agg = list(initial_agg)
        for agg in CORE_EXTREME_AGGS:
            if agg not in initial_agg:
                initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        act_affinities = act_affinities.at[4].set(0.7)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(0.7)

        # Morphogen fields with initial coupling
        act_morphogen = jnp.zeros(NUM_ACTIVATIONS)
        agg_morphogen = jnp.zeros(NUM_AGGREGATIONS)

        act_morphogen = act_morphogen.at[4].set(self.base_sin_source)
        for agg in CORE_EXTREME_AGGS:
            agg_morphogen = agg_morphogen.at[agg].set(self.extreme_agg_source)

        # Tagging
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, agg].set(0.7)

        # Coupling strength (dynamic)
        coupling_strength = 1.0

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_morphogen': act_morphogen,
            'agg_morphogen': agg_morphogen,
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            'cross_affinity': cross_affinity,
            'coupling_strength': coupling_strength,
            'capture_events': 0,
            'rng_key': jax.random.PRNGKey(seed + 1340000),
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

    def _count_active_extremes(self, agg_mask: jnp.ndarray) -> int:
        """Count active extreme aggregations."""
        return sum(1 for agg in CORE_EXTREME_AGGS if agg_mask[agg] > 0.5)

    def _update_morphogen_with_coupling(
        self,
        act_morphogen: jnp.ndarray,
        agg_morphogen: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        coupling_strength: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Update morphogen with sin-extreme coupling."""
        # Decay
        new_act = act_morphogen * (1.0 - self.morphogen_decay_rate)
        new_agg = agg_morphogen * (1.0 - self.morphogen_decay_rate)

        # CRITICAL COUPLING: Sin source boosted by active extremes
        n_active_extremes = self._count_active_extremes(agg_mask)
        if n_active_extremes > 0:
            coupling_boost = self.sin_extreme_coupling_boost * n_active_extremes * coupling_strength
            new_act = new_act.at[4].set(
                min(3.0, new_act[4] + coupling_boost)
            )
            # Also boost extreme agg sources when sin is active
            if act_mask[4] > 0.5:
                for agg in CORE_EXTREME_AGGS:
                    if agg_mask[agg] > 0.5:
                        new_agg = new_agg.at[agg].set(
                            min(3.0, new_agg[agg] + coupling_boost * 0.5)
                        )

        # Decay coupling slightly if not used
        new_coupling = coupling_strength * (1 - self.coupling_decay_rate)
        # But boost coupling when both sin and extremes are active
        if act_mask[4] > 0.5 and n_active_extremes > 0:
            new_coupling = min(1.5, coupling_strength + 0.05)

        # Floor for sin source
        new_act = new_act.at[4].set(max(self.min_sin_source_strength, float(new_act[4])))

        # Active functions contribute
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                new_act = new_act.at[i].set(min(3.0, new_act[i] + 0.1))

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                boost = 0.15 if j in CORE_EXTREME_AGGS else 0.1
                new_agg = new_agg.at[j].set(min(3.0, new_agg[j] + boost))

        return jnp.clip(new_act, 0, 3.0), jnp.clip(new_agg, 0, 3.0), new_coupling

    def _select_palettes(
        self,
        state: Dict[str, Any],
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select palettes with coupling-aware scores."""
        k1, k2 = jax.random.split(key)

        # Activation selection
        act_score = (
            state['act_affinities'] +
            state['act_captured'] * 0.3 +
            state['act_tags'] * 0.2 +
            state['act_morphogen'] * 0.4
        )
        # Sin gets extra boost based on coupling
        act_score = act_score.at[4].set(
            act_score[4] + 0.5 + state['coupling_strength'] * 0.3
        )

        target_act = min(self.max_active_act, max(self.min_diversity_act, self.min_active_act))
        top_act = jnp.argsort(act_score)[-target_act:]
        act_mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_act:
            act_mask = act_mask.at[int(idx)].set(1.0)
        # Always include sin
        act_mask = act_mask.at[4].set(1.0)

        # Aggregation selection
        agg_score = (
            state['agg_affinities'] +
            state['agg_captured'] * 0.3 +
            state['agg_tags'] * 0.2 +
            state['agg_morphogen'] * 0.4
        )
        for agg in CORE_EXTREME_AGGS:
            agg_score = agg_score.at[agg].set(agg_score[agg] + 0.4)

        target_agg = min(self.max_active_agg, max(self.min_diversity_agg, self.min_active_agg))
        top_agg = jnp.argsort(agg_score)[-target_agg:]
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for idx in top_agg:
            agg_mask = agg_mask.at[int(idx)].set(1.0)
        # Always include at least one extreme
        if not any(agg_mask[agg] > 0.5 for agg in CORE_EXTREME_AGGS):
            agg_mask = agg_mask.at[2].set(1.0)

        return act_mask, agg_mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with coupling dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Update morphogen with coupling
        new_act_morph, new_agg_morph, new_coupling = self._update_morphogen_with_coupling(
            state['act_morphogen'], state['agg_morphogen'],
            state['act_mask'], state['agg_mask'],
            state['coupling_strength']
        )

        # Update tags
        new_act_tags = state['act_tags'] * self.tag_decay
        new_agg_tags = state['agg_tags'] * self.tag_decay

        for j in range(NUM_AGGREGATIONS):
            if state['agg_mask'][j] > 0.5:
                boost = self.extreme_tag_boost if j in CORE_EXTREME_AGGS else 1.0
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + 0.35 * boost)
                )

        for i in range(NUM_ACTIVATIONS):
            if state['act_mask'][i] > 0.5:
                boost = self.extreme_tag_boost if i == 4 else 1.0
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + 0.3 * boost)
                )

        # Update affinities
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay

        if fitness_delta > 0:
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    bonus = 1.6 if j in CORE_EXTREME_AGGS else 1.0
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta * bonus)
                    )
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    bonus = 1.5 if i == 4 else 1.0
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * fitness_delta * bonus)
                    )

        # Affinity floors
        new_act_aff = new_act_aff.at[4].set(max(0.5, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.4, float(new_agg_aff[agg])))

        # Capture
        new_act_captured = state['act_captured'].copy()
        new_agg_captured = state['agg_captured'].copy()
        capture_count = 0

        if improved:
            for j in range(NUM_AGGREGATIONS):
                if state['agg_tags'][j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                    new_agg_captured = new_agg_captured.at[j].set(1.0)
                    capture_count += 1
            for i in range(NUM_ACTIVATIONS):
                if state['act_tags'][i] > self.tag_threshold and new_act_captured[i] < 0.5:
                    new_act_captured = new_act_captured.at[i].set(1.0)
                    capture_count += 1

        # Update cross affinity
        new_cross = state['cross_affinity'].copy()
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if state['act_mask'][i] > 0.5 and state['agg_mask'][j] > 0.5:
                        delta = self.cross_learning_rate * fitness_delta
                        if i == 4 and j in CORE_EXTREME_AGGS:
                            delta *= (1 + self.sin_extreme_affinity_coupling)
                        new_cross = new_cross.at[i, j].set(min(1.0, new_cross[i, j] + delta))

        # Build temporary state for selection
        temp_state = {
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_morphogen': new_act_morph,
            'agg_morphogen': new_agg_morph,
            'coupling_strength': new_coupling,
        }

        new_act_mask, new_agg_mask = self._select_palettes(temp_state, k1)

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_morphogen': new_act_morph,
            'agg_morphogen': new_agg_morph,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': state['tag_history'],
            'cross_affinity': new_cross,
            'coupling_strength': new_coupling,
            'capture_events': state['capture_events'] + capture_count,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
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
            'coupling_strength': new_coupling,
            'sin_morphogen': float(new_act_morph[4]),
            'n_active_extremes': self._count_active_extremes(new_agg_mask),
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'coupling_strength': state['coupling_strength'],
            'generation': state['generation'],
        }
