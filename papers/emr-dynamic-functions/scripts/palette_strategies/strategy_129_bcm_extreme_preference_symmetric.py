"""Strategy 129S: BCM Extreme Preference Symmetric.

BCM with explicit extreme preference and sin coupling.
Lower threshold for sin and extreme aggregations.
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
class BCMExtremePreferenceSymmetricState:
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    act_affinity: jnp.ndarray
    agg_affinity: jnp.ndarray
    act_threshold: jnp.ndarray
    agg_threshold: jnp.ndarray
    act_memory_cells: jnp.ndarray
    agg_memory_cells: jnp.ndarray
    act_memory_counts: jnp.ndarray
    agg_memory_counts: jnp.ndarray
    rng_key: jnp.ndarray
    generation: int
    stagnation_count: int
    best_fitness_seen: float
    fitness_history: List[float] = field(default_factory=list)


class BCMExtremePreferenceSymmetricStrategy(PaletteEvolutionStrategy):
    """BCM with per-function sliding thresholds."""

    name = "bcm_extreme_preference_symmetric"
    description = "Symmetric: BCM per-function thresholds with extreme preference"

    def __init__(
        self,
        base_threshold: float = 0.5,
        sin_threshold_discount: float = 0.35,
        extreme_threshold_discount: float = 0.30,
        threshold_lr: float = 0.05,
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.base_threshold = base_threshold
        self.sin_threshold_discount = sin_threshold_discount
        self.extreme_threshold_discount = extreme_threshold_discount
        self.threshold_lr = threshold_lr
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

    def initialize(self, config: Dict[str, Any], seed: int) -> BCMExtremePreferenceSymmetricState:
        key = jax.random.PRNGKey(seed + 129100)
        act_mask = create_initial_palette_mask(self.initial_act_palette)
        agg_mask = create_initial_agg_palette_mask(self.initial_agg_palette)

        act_aff = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_aff = jnp.ones(NUM_AGGREGATIONS) * 0.5
        act_aff, agg_aff = self._apply_affinity_floors(act_aff, agg_aff)

        # Per-function thresholds with discounts
        act_threshold = jnp.ones(NUM_ACTIVATIONS) * self.base_threshold
        act_threshold = act_threshold.at[SIN_IDX].set(self.base_threshold * (1 - self.sin_threshold_discount))
        agg_threshold = jnp.ones(NUM_AGGREGATIONS) * self.base_threshold
        for idx in [MAX_IDX, MIN_IDX]:
            agg_threshold = agg_threshold.at[idx].set(self.base_threshold * (1 - self.extreme_threshold_discount))

        return BCMExtremePreferenceSymmetricState(
            act_mask=act_mask, agg_mask=agg_mask,
            act_affinity=act_aff, agg_affinity=agg_aff,
            act_threshold=act_threshold, agg_threshold=agg_threshold,
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

    def _select_palette(self, affinity, threshold, n_funcs, protected_indices, key):
        mask = (affinity >= threshold).astype(jnp.float32)

        # Ensure minimum
        active_count = int(jnp.sum(mask > 0.5))
        if active_count < 2:
            top_k = jnp.argsort(affinity)[-4:]
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

        # Update thresholds (BCM dynamics)
        new_act_threshold = state.act_threshold + self.threshold_lr * (new_act_aff - state.act_threshold)
        new_agg_threshold = state.agg_threshold + self.threshold_lr * (new_agg_aff - state.agg_threshold)
        # Maintain discount floors for sin/extremes
        new_act_threshold = new_act_threshold.at[SIN_IDX].set(
            jnp.minimum(new_act_threshold[SIN_IDX], self.base_threshold * (1 - self.sin_threshold_discount + 0.1))
        )
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg_threshold = new_agg_threshold.at[idx].set(
                jnp.minimum(new_agg_threshold[idx], self.base_threshold * (1 - self.extreme_threshold_discount + 0.1))
            )

        # Memory cells
        act_candidate = (state.act_mask > 0.5) & (new_act_aff >= self.memory_formation_threshold)
        agg_candidate = (state.agg_mask > 0.5) & (new_agg_aff >= self.memory_formation_threshold)
        new_act_mem_counts = jnp.where(act_candidate, state.act_memory_counts + 1, 0)
        new_agg_mem_counts = jnp.where(agg_candidate, state.agg_memory_counts + 1, 0)
        new_act_mem = jnp.logical_or(state.act_memory_cells, new_act_mem_counts >= self.memory_formation_count)
        new_agg_mem = jnp.logical_or(state.agg_memory_cells, new_agg_mem_counts >= self.memory_formation_count)

        # Select palettes
        new_act_mask = self._select_palette(new_act_aff, new_act_threshold, NUM_ACTIVATIONS, [SIN_IDX], k_act)
        new_agg_mask = self._select_palette(new_agg_aff, new_agg_threshold, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX], k_agg)
        new_act_mask = jnp.where(new_act_mem, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem, 1.0, new_agg_mask)

        new_state = BCMExtremePreferenceSymmetricState(
            act_mask=new_act_mask, agg_mask=new_agg_mask,
            act_affinity=new_act_aff, agg_affinity=new_agg_aff,
            act_threshold=new_act_threshold, agg_threshold=new_agg_threshold,
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
            'sin_threshold': float(new_act_threshold[SIN_IDX]),
        }
        return new_state, metrics

    def get_state_summary(self, state) -> Dict[str, Any]:
        return {'strategy': self.name, 'generation': state.generation}
