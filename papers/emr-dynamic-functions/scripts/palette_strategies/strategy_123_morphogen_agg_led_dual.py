"""Strategy 123: Morphogen-Agg-Led Spatial Dual.

Combines Morphogen gradients (#31) with Critical Period (#9) and Aggregation-led (#101).
Agg morphogen sources develop first, act sources follow toward them.

Key Innovation:
- Aggregations develop morphogen sources FIRST (agg-lead duration)
- Activation mutations are attracted toward active agg source positions
- Creates spatial coupling between act/agg development
- Sin is attracted toward extreme agg sources

Biological basis: In development, some cell types differentiate first and then
guide the development of other cell types through morphogen signaling.

Expected: Coordinated act-agg discovery through spatial guidance.
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


class MorphogenAggLedDualStrategy(PaletteEvolutionStrategy):
    """Morphogen-guided agg-led spatial development for dual palette evolution.

    Aggregations develop morphogen sources first,
    which then attract activation mutations.

    Critical innovation: Spatial coupling through temporal leadership.
    """

    name = "morphogen_agg_led_dual"
    description = "Dual: Agg morphogen sources lead, activations follow the gradient"

    def __init__(
        self,
        # === Agg-led timing ===
        agg_lead_duration: int = 20,
        agg_source_strength: float = 1.5,
        # === Act-agg attraction ===
        act_agg_attraction: float = 0.2,
        sin_extreme_attraction: float = 0.35,
        sin_idx: int = 4,
        # === Morphogen parameters ===
        morphogen_decay: float = 0.92,
        diffusion_rate: float = 0.15,
        # === Mutation rates ===
        agg_lead_mutation_rate: float = 0.15,
        act_follow_mutation_rate: float = 0.10,
        normal_mutation_rate: float = 0.08,
        # === General parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Morphogen-Agg-Led strategy."""
        # Agg-led timing
        self.agg_lead_duration = agg_lead_duration
        self.agg_source_strength = agg_source_strength

        # Act-agg attraction
        self.act_agg_attraction = act_agg_attraction
        self.sin_extreme_attraction = sin_extreme_attraction
        self.sin_idx = sin_idx

        # Morphogen
        self.morphogen_decay = morphogen_decay
        self.diffusion_rate = diffusion_rate

        # Mutation rates
        self.agg_lead_mutation_rate = agg_lead_mutation_rate
        self.act_follow_mutation_rate = act_follow_mutation_rate
        self.normal_mutation_rate = normal_mutation_rate

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with morphogen fields."""
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

        # Morphogen concentrations
        act_morphogen = jnp.zeros(NUM_ACTIVATIONS)
        agg_morphogen = jnp.zeros(NUM_AGGREGATIONS)

        # Initialize extreme agg sources
        for idx in CORE_EXTREME_AGGS:
            agg_morphogen = agg_morphogen.at[idx].set(self.agg_source_strength)

        # Act-agg attraction field (how much each act is attracted to aggs)
        # This represents spatial coupling
        act_agg_coupling = jnp.zeros(NUM_ACTIVATIONS)
        # Sin starts with high coupling to extremes
        act_agg_coupling = act_agg_coupling.at[self.sin_idx].set(0.5)

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
            # Spatial coupling
            'act_agg_coupling': act_agg_coupling,
            # Stats
            'agg_led_mutations': 0,
            'coupled_discoveries': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1230000),
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

    def _is_agg_lead_phase(self, generation: int) -> bool:
        return generation < self.agg_lead_duration

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with agg-led morphogen guidance."""
        key, k1, k2, k3, k4, k5 = jax.random.split(state['rng_key'], 6)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        agg_lead_phase = self._is_agg_lead_phase(generation)

        # === UPDATE MORPHOGEN FIELDS ===
        act_morphogen = state['act_morphogen'] * self.morphogen_decay
        agg_morphogen = state['agg_morphogen'] * self.morphogen_decay

        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        act_agg_coupling = state['act_agg_coupling']
        agg_led_mutations = state['agg_led_mutations']
        coupled_discoveries = state['coupled_discoveries']

        # Active aggs emit morphogen
        for i in range(NUM_AGGREGATIONS):
            if float(agg_mask[i]) > 0.5:
                strength = self.agg_source_strength
                if i in CORE_EXTREME_AGGS:
                    strength *= 1.5
                agg_morphogen = agg_morphogen.at[i].add(strength * 0.3)

        # === AGGREGATION UPDATE (leads during agg-lead phase) ===
        if agg_lead_phase:
            agg_mutation_rate = self.agg_lead_mutation_rate
        else:
            agg_mutation_rate = self.normal_mutation_rate

        if jax.random.uniform(k1) < agg_mutation_rate:
            candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                # Weight by morphogen concentration
                weights = jnp.array([float(agg_morphogen[i]) + 0.1 for i in candidates])
                # Boost extremes
                for j, i in enumerate(candidates):
                    if i in CORE_EXTREME_AGGS:
                        weights = weights.at[j].set(weights[j] + 0.5)
                probs = weights / weights.sum()
                new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs))
                agg_mask = agg_mask.at[new_idx].set(1.0)
                agg_led_mutations += 1

        # === ACTIVATION UPDATE (follows agg gradients) ===
        if agg_lead_phase:
            act_mutation_rate = self.act_follow_mutation_rate
        else:
            act_mutation_rate = self.normal_mutation_rate

        if jax.random.uniform(k3) < act_mutation_rate:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                # Weight by coupling to active aggs
                weights = []
                for i in candidates:
                    w = 0.1 + float(act_morphogen[i])
                    # Add attraction from active extreme aggs
                    extreme_active = sum(1 for e in CORE_EXTREME_AGGS if float(agg_mask[e]) > 0.5)
                    if i == self.sin_idx and extreme_active > 0:
                        w += self.sin_extreme_attraction * extreme_active
                    # General coupling
                    w += float(act_agg_coupling[i]) * self.act_agg_attraction
                    weights.append(w)

                probs = jnp.array(weights)
                probs = probs / probs.sum()
                new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs))
                act_mask = act_mask.at[new_idx].set(1.0)

                if new_idx == self.sin_idx:
                    coupled_discoveries += 1

        # Update coupling based on co-activation success
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
                # Increase coupling for acts that succeed with extremes
                if any(g in CORE_EXTREME_AGGS for g in active_aggs):
                    act_agg_coupling = act_agg_coupling.at[a].add(self.diffusion_rate)

            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)

        # Clamp
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)
        act_agg_coupling = jnp.clip(act_agg_coupling, 0.0, 1.0)

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
                new_idx = int(jax.random.choice(k5, jnp.array(candidates)))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        new_state = {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_morphogen': act_morphogen,
            'agg_morphogen': agg_morphogen,
            'act_agg_coupling': act_agg_coupling,
            'agg_led_mutations': agg_led_mutations,
            'coupled_discoveries': coupled_discoveries,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'agg_lead_phase': agg_lead_phase,
            'agg_led_mutations': agg_led_mutations,
            'coupled_discoveries': coupled_discoveries,
            'mean_coupling': float(act_agg_coupling.mean()),
        }

        return new_state, metrics
