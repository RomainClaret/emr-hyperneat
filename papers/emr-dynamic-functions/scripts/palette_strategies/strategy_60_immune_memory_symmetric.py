"""Strategy 60S: Immune Memory Symmetric.

Adaptive immunity with memory cell formation for sin/extremes.
Protected indices get automatic memory formation boost.
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
class ImmuneMemorySymmetricState:
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    act_affinity: jnp.ndarray
    agg_affinity: jnp.ndarray
    act_memory_cells: jnp.ndarray
    agg_memory_cells: jnp.ndarray
    act_memory_counts: jnp.ndarray
    agg_memory_counts: jnp.ndarray
    act_plasma_timer: jnp.ndarray
    agg_plasma_timer: jnp.ndarray
    rng_key: jnp.ndarray
    generation: int
    stagnation_count: int
    best_fitness_seen: float
    fitness_history: List[float] = field(default_factory=list)


class ImmuneMemorySymmetricStrategy(PaletteEvolutionStrategy):
    """Adaptive immunity with automatic memory for protected indices."""

    name = "immune_memory_symmetric"
    description = "Symmetric: Immune memory with protected index auto-formation"

    def __init__(
        self,
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        protected_memory_boost: float = 0.5,
        plasma_cell_duration: int = 5,
        plasma_cell_boost: float = 1.3,
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count
        self.protected_memory_boost = protected_memory_boost
        self.plasma_cell_duration = plasma_cell_duration
        self.plasma_cell_boost = plasma_cell_boost
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]

    def _apply_affinity_floors(self, act_aff, agg_aff):
        new_act = act_aff.at[SIN_IDX].set(jnp.maximum(act_aff[SIN_IDX], self.sin_affinity_floor))
        new_agg = agg_aff
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor))
        return new_act, new_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> ImmuneMemorySymmetricState:
        key = jax.random.PRNGKey(seed + 60100)
        act_mask = create_initial_palette_mask(self.initial_act_palette)
        agg_mask = create_initial_agg_palette_mask(self.initial_agg_palette)

        act_aff = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_aff = jnp.ones(NUM_AGGREGATIONS) * 0.5
        act_aff, agg_aff = self._apply_affinity_floors(act_aff, agg_aff)

        return ImmuneMemorySymmetricState(
            act_mask=act_mask, agg_mask=agg_mask,
            act_affinity=act_aff, agg_affinity=agg_aff,
            act_memory_cells=jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_),
            agg_memory_cells=jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_),
            act_memory_counts=jnp.zeros(NUM_ACTIVATIONS),
            agg_memory_counts=jnp.zeros(NUM_AGGREGATIONS),
            act_plasma_timer=jnp.zeros(NUM_ACTIVATIONS),
            agg_plasma_timer=jnp.zeros(NUM_AGGREGATIONS),
            rng_key=key, generation=0, stagnation_count=0, best_fitness_seen=0.0,
            fitness_history=[],
        )

    def get_active_palette(self, state) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _select_palette(self, affinity, plasma_timer, n_funcs, protected_indices, key):
        # Boost affinity for plasma cells
        effective_affinity = affinity + (plasma_timer > 0).astype(jnp.float32) * (self.plasma_cell_boost - 1) * affinity

        top_k = jnp.argsort(effective_affinity)[-6:]
        mask = jnp.zeros(n_funcs)
        for idx in top_k:
            mask = mask.at[int(idx)].set(1.0)

        # Protected indices
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

        # Update affinities
        delta = improvement if improvement > 0 else -0.01
        new_act_aff = state.act_affinity + delta * 0.1 * state.act_mask
        new_agg_aff = state.agg_affinity + delta * 0.1 * state.agg_mask
        new_act_aff = jnp.clip(new_act_aff * 0.99, 0.0, 1.0)
        new_agg_aff = jnp.clip(new_agg_aff * 0.99, 0.0, 1.0)
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Memory cell formation with protected boost
        act_threshold = self.memory_formation_threshold
        agg_threshold = self.memory_formation_threshold

        # Protected indices get easier threshold
        act_above = new_act_aff >= act_threshold
        agg_above = new_agg_aff >= agg_threshold
        # Boost protected indices
        act_above = act_above.at[SIN_IDX].set(act_above[SIN_IDX] | (new_act_aff[SIN_IDX] >= act_threshold - self.protected_memory_boost))
        for idx in [MAX_IDX, MIN_IDX]:
            agg_above = agg_above.at[idx].set(agg_above[idx] | (new_agg_aff[idx] >= agg_threshold - self.protected_memory_boost))

        act_candidate = (state.act_mask > 0.5) & act_above
        agg_candidate = (state.agg_mask > 0.5) & agg_above
        new_act_mem_counts = jnp.where(act_candidate, state.act_memory_counts + 1, 0)
        new_agg_mem_counts = jnp.where(agg_candidate, state.agg_memory_counts + 1, 0)
        new_act_mem = jnp.logical_or(state.act_memory_cells, new_act_mem_counts >= self.memory_formation_count)
        new_agg_mem = jnp.logical_or(state.agg_memory_cells, new_agg_mem_counts >= self.memory_formation_count)

        # Update plasma timers
        new_act_plasma = jnp.maximum(0, state.act_plasma_timer - 1)
        new_agg_plasma = jnp.maximum(0, state.agg_plasma_timer - 1)
        # Trigger plasma cells on improvement
        if improvement > 0:
            active_act = (state.act_mask > 0.5)
            active_agg = (state.agg_mask > 0.5)
            new_act_plasma = jnp.where(active_act, self.plasma_cell_duration, new_act_plasma)
            new_agg_plasma = jnp.where(active_agg, self.plasma_cell_duration, new_agg_plasma)

        # Select palettes
        new_act_mask = self._select_palette(new_act_aff, new_act_plasma, NUM_ACTIVATIONS, [SIN_IDX], k_act)
        new_agg_mask = self._select_palette(new_agg_aff, new_agg_plasma, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX], k_agg)
        new_act_mask = jnp.where(new_act_mem, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem, 1.0, new_agg_mask)

        new_state = ImmuneMemorySymmetricState(
            act_mask=new_act_mask, agg_mask=new_agg_mask,
            act_affinity=new_act_aff, agg_affinity=new_agg_aff,
            act_memory_cells=new_act_mem, agg_memory_cells=new_agg_mem,
            act_memory_counts=new_act_mem_counts, agg_memory_counts=new_agg_mem_counts,
            act_plasma_timer=new_act_plasma, agg_plasma_timer=new_agg_plasma,
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
            'act_memory_cells': int(jnp.sum(new_act_mem)),
            'agg_memory_cells': int(jnp.sum(new_agg_mem)),
        }
        return new_state, metrics

    def get_state_summary(self, state) -> Dict[str, Any]:
        return {'strategy': self.name, 'generation': state.generation}
