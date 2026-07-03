"""Strategy 122S: Morphogen Sin Field Symmetric.

Sin-centered morphogen field with extreme aggregation coupling.
Protected indices maintain high field concentration.
"""

from dataclasses import dataclass, field
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
)

SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3


@dataclass
class MorphogenSinFieldSymmetricState:
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    sin_field_strength: float
    act_field_values: jnp.ndarray
    agg_field_values: jnp.ndarray
    act_affinity: jnp.ndarray
    agg_affinity: jnp.ndarray
    cross_affinity: jnp.ndarray
    act_memory_cells: jnp.ndarray
    agg_memory_cells: jnp.ndarray
    act_memory_counts: jnp.ndarray
    agg_memory_counts: jnp.ndarray
    rng_key: jnp.ndarray
    generation: int
    stagnation_count: int
    best_fitness_seen: float
    fitness_history: List[float] = field(default_factory=list)


class MorphogenSinFieldSymmetricStrategy(PaletteEvolutionStrategy):
    """Sin-centered morphogen field for symmetric discovery."""

    name = "morphogen_sin_field_symmetric"
    description = "Symmetric: Sin-centered field with extreme agg coupling"

    def __init__(
        self,
        sin_field_radius: float = 0.4,
        field_decay: float = 2.5,
        protected_field_floor: float = 0.7,
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.sin_field_radius = sin_field_radius
        self.field_decay = field_decay
        self.protected_field_floor = protected_field_floor
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]

    def _apply_affinity_floors(self, act_aff, agg_aff):
        new_act = act_aff.at[SIN_IDX].set(jnp.maximum(act_aff[SIN_IDX], self.sin_affinity_floor))
        new_agg = agg_aff
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor))
        return new_act, new_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> MorphogenSinFieldSymmetricState:
        key = jax.random.PRNGKey(seed + 122200)
        act_mask = create_initial_palette_mask(self.initial_act_palette)
        agg_mask = create_initial_agg_palette_mask(self.initial_agg_palette)

        # Sin-centered field values
        act_field = jnp.ones(NUM_ACTIVATIONS) * 0.3
        act_field = act_field.at[SIN_IDX].set(1.0)  # Sin is center
        agg_field = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for idx in [MAX_IDX, MIN_IDX]:
            agg_field = agg_field.at[idx].set(0.8)  # Extremes coupled to sin

        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        act_affinity, agg_affinity = self._apply_affinity_floors(act_affinity, agg_affinity)

        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg_idx in [MAX_IDX, MIN_IDX]:
            cross_affinity = cross_affinity.at[SIN_IDX, agg_idx].set(0.8)

        return MorphogenSinFieldSymmetricState(
            act_mask=act_mask, agg_mask=agg_mask,
            sin_field_strength=1.0,
            act_field_values=act_field, agg_field_values=agg_field,
            act_affinity=act_affinity, agg_affinity=agg_affinity,
            cross_affinity=cross_affinity,
            act_memory_cells=jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_),
            agg_memory_cells=jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_),
            act_memory_counts=jnp.zeros(NUM_ACTIVATIONS),
            agg_memory_counts=jnp.zeros(NUM_AGGREGATIONS),
            rng_key=key, generation=0, stagnation_count=0, best_fitness_seen=0.0,
            fitness_history=[],
        )

    def get_active_palette(self, state) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _select_palette(self, scores, n_funcs, protected_indices, key):
        top_k = jnp.argsort(scores)[-6:]
        mask = jnp.zeros(n_funcs)
        for idx in top_k:
            mask = mask.at[int(idx)].set(1.0)
        key, subkey = jax.random.split(key)
        probs = jax.random.uniform(subkey, (n_funcs,))
        for idx in protected_indices:
            if 0 <= idx < n_funcs and mask[idx] < 0.5:
                if probs[idx] >= self.protected_deactivation_prob:
                    mask = mask.at[idx].set(1.0)
        return mask

    def post_generation_update(self, state, generation, best_fitness, prev_best_fitness, population_data=None):
        key = state.rng_key
        key, k_act, k_agg = jax.random.split(key, 3)

        improved = best_fitness > state.best_fitness_seen
        improvement = best_fitness - prev_best_fitness
        new_stagnation = 0 if improved else state.stagnation_count + 1
        new_best = best_fitness if improved else state.best_fitness_seen

        # Update sin field strength
        new_sin_strength = state.sin_field_strength
        if improved:
            new_sin_strength = jnp.minimum(new_sin_strength * 1.05, 2.0)
        else:
            new_sin_strength = jnp.maximum(new_sin_strength * 0.98, 0.5)

        # Update field values with protection
        new_act_field = state.act_field_values * 0.98
        new_agg_field = state.agg_field_values * 0.98
        if improvement > 0:
            new_act_field = new_act_field + 0.1 * improvement * state.act_mask
            new_agg_field = new_agg_field + 0.1 * improvement * state.agg_mask
        new_act_field = new_act_field.at[SIN_IDX].set(jnp.maximum(new_act_field[SIN_IDX], self.protected_field_floor))
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg_field = new_agg_field.at[idx].set(jnp.maximum(new_agg_field[idx], self.protected_field_floor))
        new_act_field = jnp.clip(new_act_field, 0.0, 1.0)
        new_agg_field = jnp.clip(new_agg_field, 0.0, 1.0)

        # Update affinities
        new_act_aff = state.act_affinity + (improvement if improvement > 0 else -0.01) * 0.1 * state.act_mask
        new_agg_aff = state.agg_affinity + (improvement if improvement > 0 else -0.01) * 0.1 * state.agg_mask
        new_act_aff = jnp.clip(new_act_aff * 0.99, 0.0, 1.0)
        new_agg_aff = jnp.clip(new_agg_aff * 0.99, 0.0, 1.0)
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Memory cells
        act_candidate = (state.act_mask > 0.5) & (new_act_aff >= self.memory_formation_threshold)
        agg_candidate = (state.agg_mask > 0.5) & (new_agg_aff >= self.memory_formation_threshold)
        new_act_mem_counts = jnp.where(act_candidate, state.act_memory_counts + 1, 0)
        new_agg_mem_counts = jnp.where(agg_candidate, state.agg_memory_counts + 1, 0)
        new_act_mem = jnp.logical_or(state.act_memory_cells, new_act_mem_counts >= self.memory_formation_count)
        new_agg_mem = jnp.logical_or(state.agg_memory_cells, new_agg_mem_counts >= self.memory_formation_count)

        # Select palettes
        act_scores = 0.4 * new_act_field + 0.6 * new_act_aff
        agg_scores = 0.4 * new_agg_field + 0.6 * new_agg_aff
        new_act_mask = self._select_palette(act_scores, NUM_ACTIVATIONS, [SIN_IDX], k_act)
        new_agg_mask = self._select_palette(agg_scores, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX], k_agg)
        new_act_mask = jnp.where(new_act_mem, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem, 1.0, new_agg_mask)

        new_state = MorphogenSinFieldSymmetricState(
            act_mask=new_act_mask, agg_mask=new_agg_mask,
            sin_field_strength=float(new_sin_strength),
            act_field_values=new_act_field, agg_field_values=new_agg_field,
            act_affinity=new_act_aff, agg_affinity=new_agg_aff,
            cross_affinity=state.cross_affinity,
            act_memory_cells=new_act_mem, agg_memory_cells=new_agg_mem,
            act_memory_counts=new_act_mem_counts, agg_memory_counts=new_agg_mem_counts,
            rng_key=key, generation=generation + 1,
            stagnation_count=new_stagnation, best_fitness_seen=new_best,
            fitness_history=state.fitness_history[-19:] + [best_fitness],
        )

        metrics = {
            'current_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'has_sin': SIN_IDX in mask_to_indices(new_act_mask),
            'has_max': MAX_IDX in mask_to_indices(new_agg_mask),
            'has_min': MIN_IDX in mask_to_indices(new_agg_mask),
            'sin_field_strength': float(new_sin_strength),
        }
        return new_state, metrics

    def get_state_summary(self, state) -> Dict[str, Any]:
        return {'strategy': self.name, 'generation': state.generation}
