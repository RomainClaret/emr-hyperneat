"""Strategy 124: Morphogen-Gradient Plasticity Dual.

Combines Morphogen gradients (#31) with Critical Period plasticity (#9).
Gradient steepness determines function plasticity (high gradient = high learning).

Key Innovation:
- Functions at morphogen concentration edges have higher plasticity
- Edge positions = transition regions = high gradient
- High plasticity = faster affinity updates, more likely to be mutated
- Creates dynamic exploration at the boundaries of active regions

Biological basis: In development, cells at morphogen gradient edges often
have higher plasticity and can differentiate in multiple ways. We apply
this to palette evolution: edge functions are more malleable.

Expected: More dynamic exploration at palette boundaries.
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


class MorphogenGradientPlasticityDualStrategy(PaletteEvolutionStrategy):
    """Morphogen gradient-based plasticity for dual palette evolution.

    Functions at gradient edges have higher plasticity,
    leading to more dynamic exploration at palette boundaries.

    Critical innovation: Edge plasticity creates adaptive exploration zones.
    """

    name = "morphogen_gradient_plasticity_dual"
    description = "Dual: Gradient edges have high plasticity for dynamic exploration"

    def __init__(
        self,
        # === Gradient plasticity ===
        gradient_plasticity_scale: float = 2.0,
        gradient_decay_per_gen: float = 0.95,
        edge_plasticity_bonus: float = 1.5,
        edge_threshold: float = 0.3,
        # === Morphogen parameters ===
        morphogen_emission_rate: float = 0.3,
        morphogen_decay: float = 0.90,
        # === Sin preference ===
        sin_idx: int = 4,
        sin_edge_attraction: float = 0.25,
        # === Mutation parameters ===
        base_mutation_rate: float = 0.10,
        plasticity_mutation_boost: float = 0.15,
        # === General parameters ===
        base_affinity_lr: float = 0.10,
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
        """Initialize Morphogen-Gradient Plasticity strategy."""
        # Gradient plasticity
        self.gradient_plasticity_scale = gradient_plasticity_scale
        self.gradient_decay_per_gen = gradient_decay_per_gen
        self.edge_plasticity_bonus = edge_plasticity_bonus
        self.edge_threshold = edge_threshold

        # Morphogen
        self.morphogen_emission_rate = morphogen_emission_rate
        self.morphogen_decay = morphogen_decay

        # Sin
        self.sin_idx = sin_idx
        self.sin_edge_attraction = sin_edge_attraction

        # Mutation
        self.base_mutation_rate = base_mutation_rate
        self.plasticity_mutation_boost = plasticity_mutation_boost

        # General
        self.base_affinity_lr = base_affinity_lr
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
        """Initialize state with morphogen and plasticity fields."""
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

        # Initialize morphogen from initial palette
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_morphogen = act_morphogen.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_morphogen = agg_morphogen.at[i].set(0.5)

        # Plasticity per function (starts uniform)
        act_plasticity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_plasticity = jnp.ones(NUM_AGGREGATIONS) * 0.5

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
            # Plasticity
            'act_plasticity': act_plasticity,
            'agg_plasticity': agg_plasticity,
            # Stats
            'edge_mutations': 0,
            'high_plasticity_updates': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1240000),
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

    def _calculate_gradient(self, morphogen: jnp.ndarray, idx: int, n_funcs: int) -> float:
        """Calculate local gradient magnitude at position idx."""
        left_val = float(morphogen[max(0, idx - 1)])
        right_val = float(morphogen[min(n_funcs - 1, idx + 1)])
        center_val = float(morphogen[idx])
        # Gradient is the maximum absolute difference
        gradient = max(abs(center_val - left_val), abs(center_val - right_val))
        return gradient

    def _is_edge_position(self, morphogen: jnp.ndarray, idx: int, n_funcs: int) -> bool:
        """Check if position is at a gradient edge."""
        gradient = self._calculate_gradient(morphogen, idx, n_funcs)
        return gradient > self.edge_threshold

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with gradient-based plasticity."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === UPDATE MORPHOGEN FIELDS ===
        act_morphogen = state['act_morphogen'] * self.morphogen_decay
        agg_morphogen = state['agg_morphogen'] * self.morphogen_decay

        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        act_plasticity = state['act_plasticity'] * self.gradient_decay_per_gen
        agg_plasticity = state['agg_plasticity'] * self.gradient_decay_per_gen
        edge_mutations = state['edge_mutations']
        high_plasticity_updates = state['high_plasticity_updates']

        # Active functions emit morphogen
        for i in range(NUM_ACTIVATIONS):
            if float(act_mask[i]) > 0.5:
                act_morphogen = act_morphogen.at[i].add(self.morphogen_emission_rate)
        for i in range(NUM_AGGREGATIONS):
            if float(agg_mask[i]) > 0.5:
                agg_morphogen = agg_morphogen.at[i].add(self.morphogen_emission_rate)

        # === UPDATE PLASTICITY BASED ON GRADIENT ===
        for i in range(NUM_ACTIVATIONS):
            gradient = self._calculate_gradient(act_morphogen, i, NUM_ACTIVATIONS)
            # High gradient = high plasticity
            plasticity_boost = gradient * self.gradient_plasticity_scale
            if self._is_edge_position(act_morphogen, i, NUM_ACTIVATIONS):
                plasticity_boost += self.edge_plasticity_bonus
            act_plasticity = act_plasticity.at[i].add(plasticity_boost * 0.1)

        for i in range(NUM_AGGREGATIONS):
            gradient = self._calculate_gradient(agg_morphogen, i, NUM_AGGREGATIONS)
            plasticity_boost = gradient * self.gradient_plasticity_scale
            if self._is_edge_position(agg_morphogen, i, NUM_AGGREGATIONS):
                plasticity_boost += self.edge_plasticity_bonus
            agg_plasticity = agg_plasticity.at[i].add(plasticity_boost * 0.1)

        # Clamp plasticity
        act_plasticity = jnp.clip(act_plasticity, 0.1, 2.0)
        agg_plasticity = jnp.clip(agg_plasticity, 0.1, 2.0)

        # === ACTIVATION MUTATION (plasticity-weighted) ===
        candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
        if candidates:
            # Mutation probability based on plasticity
            weights = []
            for i in candidates:
                w = 0.1 + float(act_plasticity[i]) * self.plasticity_mutation_boost
                if i == self.sin_idx:
                    w += self.sin_edge_attraction
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            mutation_rate = self.base_mutation_rate
            if jax.random.uniform(k1) < mutation_rate:
                new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs))
                act_mask = act_mask.at[new_idx].set(1.0)

                if self._is_edge_position(act_morphogen, new_idx, NUM_ACTIVATIONS):
                    edge_mutations += 1

        # === AGGREGATION MUTATION (plasticity-weighted) ===
        candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
        if candidates:
            weights = []
            for i in candidates:
                w = 0.1 + float(agg_plasticity[i]) * self.plasticity_mutation_boost
                if i in CORE_EXTREME_AGGS:
                    w += 0.3
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            if jax.random.uniform(k3) < mutation_rate:
                new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        # Update affinities (plasticity-scaled learning rate)
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                lr = self.base_affinity_lr * float(act_plasticity[a])
                act_affinities = act_affinities.at[a].add(lr)
                if float(act_plasticity[a]) > 1.0:
                    high_plasticity_updates += 1

            for g in active_aggs:
                lr = self.base_affinity_lr * float(agg_plasticity[g])
                agg_affinities = agg_affinities.at[g].add(lr)

        # Clamp affinities
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
            'act_plasticity': act_plasticity,
            'agg_plasticity': agg_plasticity,
            'edge_mutations': edge_mutations,
            'high_plasticity_updates': high_plasticity_updates,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'edge_mutations': edge_mutations,
            'high_plasticity_updates': high_plasticity_updates,
            'mean_act_plasticity': float(act_plasticity.mean()),
            'mean_agg_plasticity': float(agg_plasticity.mean()),
        }

        return new_state, metrics
