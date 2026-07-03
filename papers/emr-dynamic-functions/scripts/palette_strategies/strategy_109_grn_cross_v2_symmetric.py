"""Strategy 109S: GRN Cross-Domain v2 Symmetric.

Combines multiplicative cross-domain affinity with GRN attractor locking.
Sin-extreme pairings form stable regulatory attractors.
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
class GRNCrossV2SymmetricState:
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    act_expression: jnp.ndarray
    agg_expression: jnp.ndarray
    cross_regulation: jnp.ndarray
    cross_affinity: jnp.ndarray
    act_affinity: jnp.ndarray
    agg_affinity: jnp.ndarray
    act_memory_cells: jnp.ndarray
    agg_memory_cells: jnp.ndarray
    act_memory_counts: jnp.ndarray
    agg_memory_counts: jnp.ndarray
    attractor_lock_events: int
    rng_key: jnp.ndarray
    generation: int
    stagnation_count: int
    best_fitness_seen: float
    fitness_history: List[float] = field(default_factory=list)


class GRNCrossV2SymmetricStrategy(PaletteEvolutionStrategy):
    """Multiplicative cross-domain with GRN attractor locking."""

    name = "grn_cross_v2_symmetric"
    description = "Symmetric: GRN with multiplicative cross-domain affinity"

    def __init__(
        self,
        expression_floor: float = 0.7,
        cross_affinity_growth_factor: float = 1.15,
        cross_affinity_decay_factor: float = 0.98,
        sin_extreme_multiplier: float = 1.5,
        regulation_learning_rate: float = 0.08,
        regulation_decay: float = 0.98,
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.expression_floor = expression_floor
        self.cross_affinity_growth_factor = cross_affinity_growth_factor
        self.cross_affinity_decay_factor = cross_affinity_decay_factor
        self.sin_extreme_multiplier = sin_extreme_multiplier
        self.regulation_learning_rate = regulation_learning_rate
        self.regulation_decay = regulation_decay
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]

    def _apply_floors(self, act_expr, agg_expr, act_aff, agg_aff):
        new_act_expr = act_expr.at[SIN_IDX].set(jnp.maximum(act_expr[SIN_IDX], self.expression_floor))
        new_agg_expr = agg_expr
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg_expr = new_agg_expr.at[idx].set(jnp.maximum(new_agg_expr[idx], self.expression_floor))

        new_act_aff = act_aff.at[SIN_IDX].set(jnp.maximum(act_aff[SIN_IDX], self.sin_affinity_floor))
        new_agg_aff = agg_aff
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg_aff = new_agg_aff.at[idx].set(jnp.maximum(new_agg_aff[idx], self.extreme_agg_affinity_floor))

        return new_act_expr, new_agg_expr, new_act_aff, new_agg_aff

    def initialize(self, config: Dict[str, Any], seed: int) -> GRNCrossV2SymmetricState:
        key = jax.random.PRNGKey(seed + 109200)
        act_mask = create_initial_palette_mask(self.initial_act_palette)
        agg_mask = create_initial_agg_palette_mask(self.initial_agg_palette)

        act_expr = jnp.ones(NUM_ACTIVATIONS) * 0.4
        agg_expr = jnp.ones(NUM_AGGREGATIONS) * 0.4
        act_aff = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_aff = jnp.ones(NUM_AGGREGATIONS) * 0.5

        act_expr, agg_expr, act_aff, agg_aff = self._apply_floors(act_expr, agg_expr, act_aff, agg_aff)

        # Cross-domain affinity and regulation
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        cross_reg = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.3
        for agg_idx in [MAX_IDX, MIN_IDX]:
            cross_affinity = cross_affinity.at[SIN_IDX, agg_idx].set(0.7)
            cross_reg = cross_reg.at[SIN_IDX, agg_idx].set(0.6)

        return GRNCrossV2SymmetricState(
            act_mask=act_mask, agg_mask=agg_mask,
            act_expression=act_expr, agg_expression=agg_expr,
            cross_regulation=cross_reg, cross_affinity=cross_affinity,
            act_affinity=act_aff, agg_affinity=agg_aff,
            act_memory_cells=jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_),
            agg_memory_cells=jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_),
            act_memory_counts=jnp.zeros(NUM_ACTIVATIONS),
            agg_memory_counts=jnp.zeros(NUM_AGGREGATIONS),
            attractor_lock_events=0,
            rng_key=key, generation=0, stagnation_count=0, best_fitness_seen=0.0,
            fitness_history=[],
        )

    def get_active_palette(self, state) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _select_palette(self, expression, affinity, cross_scores, n_funcs, protected_indices, key):
        scores = 0.3 * expression + 0.4 * affinity + 0.3 * cross_scores
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

        # Expression update
        delta = improvement if improvement > 0 else -0.01
        new_act_expr = state.act_expression * 0.95 + delta * 0.1 * state.act_mask
        new_agg_expr = state.agg_expression * 0.95 + delta * 0.1 * state.agg_mask
        new_act_expr = jnp.clip(new_act_expr, 0.0, 1.0)
        new_agg_expr = jnp.clip(new_agg_expr, 0.0, 1.0)

        # Affinity update
        new_act_aff = state.act_affinity + delta * 0.1 * state.act_mask
        new_agg_aff = state.agg_affinity + delta * 0.1 * state.agg_mask
        new_act_aff = jnp.clip(new_act_aff * 0.99, 0.0, 1.0)
        new_agg_aff = jnp.clip(new_agg_aff * 0.99, 0.0, 1.0)

        # Apply floors
        new_act_expr, new_agg_expr, new_act_aff, new_agg_aff = self._apply_floors(
            new_act_expr, new_agg_expr, new_act_aff, new_agg_aff
        )

        # Multiplicative cross-affinity update
        new_cross_aff = state.cross_affinity.copy()
        new_cross_reg = state.cross_regulation * self.regulation_decay
        attractor_locks = state.attractor_lock_events

        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                current_aff = float(new_cross_aff[i, j])
                current_reg = float(new_cross_reg[i, j])

                if state.act_mask[i] > 0.5 and state.agg_mask[j] > 0.5:
                    if improvement > 0:
                        growth = self.cross_affinity_growth_factor
                        growth *= (1 + current_reg * 0.2)  # Regulation boost
                        if i == SIN_IDX and j in [MAX_IDX, MIN_IDX]:
                            growth *= self.sin_extreme_multiplier
                            if current_reg > 0.8:
                                attractor_locks += 1
                        new_cross_aff = new_cross_aff.at[i, j].set(jnp.minimum(1.0, current_aff * growth))
                        reg_delta = self.regulation_learning_rate * improvement
                        if i == SIN_IDX and j in [MAX_IDX, MIN_IDX]:
                            reg_delta *= self.sin_extreme_multiplier
                        new_cross_reg = new_cross_reg.at[i, j].set(jnp.minimum(1.5, current_reg + reg_delta))
                    else:
                        decay = self.cross_affinity_decay_factor
                        if current_reg > 0.5:
                            decay = decay ** 0.5  # Slower decay for regulated
                        new_cross_aff = new_cross_aff.at[i, j].set(jnp.maximum(0.2, current_aff * decay))
                else:
                    new_cross_aff = new_cross_aff.at[i, j].set(jnp.maximum(0.1, current_aff * 0.995))

        # Memory cells
        act_candidate = (state.act_mask > 0.5) & (new_act_aff >= self.memory_formation_threshold)
        agg_candidate = (state.agg_mask > 0.5) & (new_agg_aff >= self.memory_formation_threshold)
        new_act_mem_counts = jnp.where(act_candidate, state.act_memory_counts + 1, 0)
        new_agg_mem_counts = jnp.where(agg_candidate, state.agg_memory_counts + 1, 0)
        new_act_mem = jnp.logical_or(state.act_memory_cells, new_act_mem_counts >= self.memory_formation_count)
        new_agg_mem = jnp.logical_or(state.agg_memory_cells, new_agg_mem_counts >= self.memory_formation_count)

        # Compute cross-scores for selection
        act_cross_scores = jnp.sum(new_cross_aff * (state.agg_mask > 0.5).astype(jnp.float32), axis=1) / jnp.maximum(1, jnp.sum(state.agg_mask > 0.5))
        agg_cross_scores = jnp.sum(new_cross_aff.T * (state.act_mask > 0.5).astype(jnp.float32), axis=1) / jnp.maximum(1, jnp.sum(state.act_mask > 0.5))

        # Select palettes
        new_act_mask = self._select_palette(new_act_expr, new_act_aff, act_cross_scores, NUM_ACTIVATIONS, [SIN_IDX], k_act)
        new_agg_mask = self._select_palette(new_agg_expr, new_agg_aff, agg_cross_scores, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX], k_agg)
        new_act_mask = jnp.where(new_act_mem, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem, 1.0, new_agg_mask)

        new_state = GRNCrossV2SymmetricState(
            act_mask=new_act_mask, agg_mask=new_agg_mask,
            act_expression=new_act_expr, agg_expression=new_agg_expr,
            cross_regulation=new_cross_reg, cross_affinity=new_cross_aff,
            act_affinity=new_act_aff, agg_affinity=new_agg_aff,
            act_memory_cells=new_act_mem, agg_memory_cells=new_agg_mem,
            act_memory_counts=new_act_mem_counts, agg_memory_counts=new_agg_mem_counts,
            attractor_lock_events=attractor_locks,
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
            'sin_max_cross': float(new_cross_aff[SIN_IDX, MAX_IDX]),
            'sin_min_cross': float(new_cross_aff[SIN_IDX, MIN_IDX]),
            'attractor_lock_events': attractor_locks,
        }
        return new_state, metrics

    def get_state_summary(self, state) -> Dict[str, Any]:
        return {'strategy': self.name, 'generation': state.generation}
