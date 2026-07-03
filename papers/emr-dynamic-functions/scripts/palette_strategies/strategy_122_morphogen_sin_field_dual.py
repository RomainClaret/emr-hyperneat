"""Strategy 122: Morphogen-Sin-Field Attractor Dual.

Combines Morphogen Gradients (#31) with Critical Period (#9) and sin-preference.
Sin has a special morphogen source active only during exploration window.

Key Innovation:
- Sin has a strong morphogen source during exploration phase
- This creates spatial bias toward sin discovery early
- Morphogen concentration influences mutation probability
- After sin_source_active_until, sin source deactivates

Biological basis: In development, morphogen gradients guide cell differentiation.
Certain cell types are favored in specific regions. Here, sin is favored early.

Expected: Reliable early sin discovery through gradient-biased mutations.
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
    CORE_EXTREME_AGGS,
)


class MorphogenSinFieldDualStrategy(PaletteEvolutionStrategy):
    """Morphogen-guided sin-field attractor for dual palette evolution.

    Sin has a special morphogen source that creates
    strong attraction during exploration. Gradient fields bias which
    functions get added.

    Critical innovation: Time-limited sin-specific morphogen source.
    """

    name = "morphogen_sin_field_dual"
    description = "Dual: Sin-specific morphogen field attracts discovery during exploration"

    def __init__(
        self,
        # === Sin morphogen source parameters ===
        sin_source_strength: float = 2.0,
        sin_source_active_until: int = 30,
        sin_position_attraction: float = 0.15,
        sin_idx: int = 4,
        # === General morphogen ===
        base_diffusion_rate: float = 0.1,
        morphogen_decay: float = 0.95,
        # === Critical period phases ===
        exploration_phase_end: int = 30,
        confirmation_phase_end: int = 60,
        exploration_mutation_rate: float = 0.15,
        confirmation_mutation_rate: float = 0.08,
        consolidation_mutation_rate: float = 0.03,
        # === General parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        tag_threshold: float = 0.5,
        tag_decay: float = 0.9,
        captured_protection: float = 0.85,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Morphogen-Sin-Field strategy."""
        # Sin morphogen
        self.sin_source_strength = sin_source_strength
        self.sin_source_active_until = sin_source_active_until
        self.sin_position_attraction = sin_position_attraction
        self.sin_idx = sin_idx

        # General morphogen
        self.base_diffusion_rate = base_diffusion_rate
        self.morphogen_decay = morphogen_decay

        # Critical period
        self.exploration_phase_end = exploration_phase_end
        self.confirmation_phase_end = confirmation_phase_end
        self.exploration_mutation_rate = exploration_mutation_rate
        self.confirmation_mutation_rate = confirmation_mutation_rate
        self.consolidation_mutation_rate = consolidation_mutation_rate

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay
        self.tag_threshold = tag_threshold
        self.tag_decay = tag_decay
        self.captured_protection = captured_protection

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with morphogen gradients."""
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

        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(agg_affinities[i] + 0.2)

        # Morphogen concentration fields
        act_morphogen = jnp.zeros(NUM_ACTIVATIONS)
        agg_morphogen = jnp.zeros(NUM_AGGREGATIONS)

        # Initialize sin with high concentration
        act_morphogen = act_morphogen.at[self.sin_idx].set(self.sin_source_strength)

        # Initialize extreme aggs with moderate concentration
        for idx in CORE_EXTREME_AGGS:
            agg_morphogen = agg_morphogen.at[idx].set(1.0)

        # Tagging state
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
            # Morphogen fields
            'act_morphogen': act_morphogen,
            'agg_morphogen': agg_morphogen,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            # Stats
            'sin_discoveries': 0,
            'morphogen_influenced_adds': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1220000),
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
        if generation < self.exploration_phase_end:
            return "exploration"
        elif generation < self.confirmation_phase_end:
            return "confirmation"
        else:
            return "consolidation"

    def _get_mutation_rate(self, generation: int) -> float:
        phase = self._get_phase(generation)
        if phase == "exploration":
            return self.exploration_mutation_rate
        elif phase == "confirmation":
            return self.confirmation_mutation_rate
        else:
            return self.consolidation_mutation_rate

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with morphogen-guided mutations."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        phase = self._get_phase(generation)
        mutation_rate = self._get_mutation_rate(generation)

        # Update morphogen fields
        act_morphogen = state['act_morphogen'] * self.morphogen_decay
        agg_morphogen = state['agg_morphogen'] * self.morphogen_decay

        # Sin source active during exploration
        if generation < self.sin_source_active_until:
            sin_strength = self.sin_source_strength * (1 - generation / self.sin_source_active_until)
            act_morphogen = act_morphogen.at[self.sin_idx].add(sin_strength)

        # Extreme agg sources always slightly active
        for idx in CORE_EXTREME_AGGS:
            agg_morphogen = agg_morphogen.at[idx].add(0.3)

        # === ACTIVATION UPDATE ===
        act_mask = state['act_mask']
        act_affinities = state['act_affinities']
        act_tags = state['act_tags'] * self.tag_decay
        act_captured = state['act_captured']
        sin_discoveries = state['sin_discoveries']
        morphogen_adds = state['morphogen_influenced_adds']

        # During exploration, sin has strong attraction
        if phase == "exploration" and float(act_mask[self.sin_idx]) < 0.5:
            sin_prob = float(act_morphogen[self.sin_idx]) * self.sin_position_attraction
            if jax.random.uniform(k1) < sin_prob:
                act_mask = act_mask.at[self.sin_idx].set(1.0)
                sin_discoveries += 1
                morphogen_adds += 1

        # Normal morphogen-weighted mutation
        if jax.random.uniform(k2) < mutation_rate:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                weights = jnp.array([float(act_morphogen[i]) + 0.1 for i in candidates])
                probs = weights / weights.sum()
                new_idx = int(jax.random.choice(k3, jnp.array(candidates), p=probs))
                act_mask = act_mask.at[new_idx].set(1.0)
                morphogen_adds += 1

        # === AGGREGATION UPDATE ===
        agg_mask = state['agg_mask']
        agg_affinities = state['agg_affinities']
        agg_tags = state['agg_tags'] * self.tag_decay
        agg_captured = state['agg_captured']

        # Extreme aggs have strong attraction during exploration
        if phase == "exploration":
            for extreme_idx in CORE_EXTREME_AGGS:
                if float(agg_mask[extreme_idx]) < 0.5:
                    prob = float(agg_morphogen[extreme_idx]) * 0.1
                    if jax.random.uniform(k4) < prob:
                        agg_mask = agg_mask.at[extreme_idx].set(1.0)
                        break

        # Normal mutation
        if jax.random.uniform(k4) < mutation_rate:
            candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                weights = jnp.array([float(agg_morphogen[i]) + 0.1 for i in candidates])
                probs = weights / weights.sum()
                new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        # Update affinities on improvement
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
                act_morphogen = act_morphogen.at[a].add(self.base_diffusion_rate)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)
                agg_morphogen = agg_morphogen.at[g].add(self.base_diffusion_rate)

        # Clamp
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)

        # Ensure minimum diversity
        if sum(float(act_mask[i]) for i in range(NUM_ACTIVATIONS)) < self.min_active_act:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k1, jnp.array(candidates)))
                act_mask = act_mask.at[new_idx].set(1.0)

        if sum(float(agg_mask[i]) for i in range(NUM_AGGREGATIONS)) < self.min_active_agg:
            candidates = [i for i in CORE_EXTREME_AGGS if float(agg_mask[i]) < 0.5]
            if not candidates:
                candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k3, jnp.array(candidates)))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        new_state = {
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
            'sin_discoveries': sin_discoveries,
            'morphogen_influenced_adds': morphogen_adds,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'phase': phase,
            'sin_discovered': sin_discoveries > state['sin_discoveries'],
            'morphogen_adds': morphogen_adds - state['morphogen_influenced_adds'],
        }

        return new_state, metrics
