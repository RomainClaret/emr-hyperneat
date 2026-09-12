"""Strategy 128S: BCM Aggregation Symmetric.

BCM sliding threshold with aggregation-led discovery.
Extreme aggs have lower threshold (automatic preference).
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
class BCMAggregationSymmetricState:
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    act_affinity: jnp.ndarray
    agg_affinity: jnp.ndarray
    bcm_threshold: float
    act_memory_cells: jnp.ndarray
    agg_memory_cells: jnp.ndarray
    act_memory_counts: jnp.ndarray
    agg_memory_counts: jnp.ndarray
    discovery_history: List[float]
    rng_key: jnp.ndarray
    generation: int
    stagnation_count: int
    best_fitness_seen: float
    fitness_history: List[float] = field(default_factory=list)


class BCMAggregationSymmetricStrategy(PaletteEvolutionStrategy):
    """BCM self-regulating threshold for symmetric discovery."""

    name = "bcm_aggregation_symmetric"
    description = "Symmetric: BCM sliding threshold with extreme preference"

    def __init__(
        self,
        bcm_min_threshold: float = 0.3,
        bcm_max_threshold: float = 0.8,
        bcm_threshold_lr: float = 0.1,
        extreme_bcm_discount: float = 0.3,
        sin_bcm_discount: float = 0.25,
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.bcm_min_threshold = bcm_min_threshold
        self.bcm_max_threshold = bcm_max_threshold
        self.bcm_threshold_lr = bcm_threshold_lr
        self.extreme_bcm_discount = extreme_bcm_discount
        self.sin_bcm_discount = sin_bcm_discount
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

    def _get_effective_threshold(self, base_threshold, idx, is_activation):
        if is_activation and idx == SIN_IDX:
            return base_threshold * (1 - self.sin_bcm_discount)
        elif not is_activation and idx in [MAX_IDX, MIN_IDX]:
            return base_threshold * (1 - self.extreme_bcm_discount)
        return base_threshold

    def initialize(self, config: Dict[str, Any], seed: int) -> BCMAggregationSymmetricState:
        key = jax.random.PRNGKey(seed + 128100)
        act_mask = create_initial_palette_mask(self.initial_act_palette)
        agg_mask = create_initial_agg_palette_mask(self.initial_agg_palette)

        act_aff = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_aff = jnp.ones(NUM_AGGREGATIONS) * 0.5
        act_aff, agg_aff = self._apply_affinity_floors(act_aff, agg_aff)

        return BCMAggregationSymmetricState(
            act_mask=act_mask, agg_mask=agg_mask,
            act_affinity=act_aff, agg_affinity=agg_aff,
            bcm_threshold=0.5,
            act_memory_cells=jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_),
            agg_memory_cells=jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_),
            act_memory_counts=jnp.zeros(NUM_ACTIVATIONS),
            agg_memory_counts=jnp.zeros(NUM_AGGREGATIONS),
            discovery_history=[],
            rng_key=key, generation=0, stagnation_count=0, best_fitness_seen=0.0,
            fitness_history=[],
        )

    def get_active_palette(self, state) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _select_palette(self, affinity, bcm_threshold, n_funcs, is_activation, protected_indices, key):
        mask = jnp.zeros(n_funcs)
        for i in range(n_funcs):
            eff_threshold = self._get_effective_threshold(bcm_threshold, i, is_activation)
            if affinity[i] >= eff_threshold:
                mask = mask.at[i].set(1.0)

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

        # Update BCM threshold
        discovery_rate = 1.0 if improved else 0.0
        discovery_history = state.discovery_history[-9:] + [discovery_rate]
        avg_discovery = sum(discovery_history) / len(discovery_history) if discovery_history else 0.0

        new_bcm_threshold = state.bcm_threshold
        if avg_discovery > 0.15:
            new_bcm_threshold = min(self.bcm_max_threshold, new_bcm_threshold + self.bcm_threshold_lr * 0.1)
        else:
            new_bcm_threshold = max(self.bcm_min_threshold, new_bcm_threshold - self.bcm_threshold_lr * 0.05)

        # Update affinities
        delta = improvement if improvement > 0 else -0.01
        new_act_aff = state.act_affinity + delta * 0.1 * state.act_mask
        new_agg_aff = state.agg_affinity + delta * 0.1 * state.agg_mask
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
        new_act_mask = self._select_palette(new_act_aff, new_bcm_threshold, NUM_ACTIVATIONS, True, [SIN_IDX], k_act)
        new_agg_mask = self._select_palette(new_agg_aff, new_bcm_threshold, NUM_AGGREGATIONS, False, [MAX_IDX, MIN_IDX], k_agg)
        new_act_mask = jnp.where(new_act_mem, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem, 1.0, new_agg_mask)

        new_state = BCMAggregationSymmetricState(
            act_mask=new_act_mask, agg_mask=new_agg_mask,
            act_affinity=new_act_aff, agg_affinity=new_agg_aff,
            bcm_threshold=new_bcm_threshold,
            act_memory_cells=new_act_mem, agg_memory_cells=new_agg_mem,
            act_memory_counts=new_act_mem_counts, agg_memory_counts=new_agg_mem_counts,
            discovery_history=discovery_history,
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
            'bcm_threshold': new_bcm_threshold,
        }
        return new_state, metrics

    def get_state_summary(self, state) -> Dict[str, Any]:
        return {'strategy': self.name, 'generation': state.generation}
