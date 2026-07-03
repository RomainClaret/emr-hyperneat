"""Strategy 78S: GRN Aggregation Discovery Symmetric.

GRN with sin-extreme regulatory coupling for symmetric discovery.
Hardcoded sin->max and sin->min regulatory links.
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
class GRNAggDiscoverySymmetricState:
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    act_expression: jnp.ndarray
    agg_expression: jnp.ndarray
    cross_regulation: jnp.ndarray
    act_affinity: jnp.ndarray
    agg_affinity: jnp.ndarray
    act_memory_cells: jnp.ndarray
    agg_memory_cells: jnp.ndarray
    act_memory_counts: jnp.ndarray
    agg_memory_counts: jnp.ndarray
    sin_extreme_coupling_events: int
    rng_key: jnp.ndarray
    generation: int
    stagnation_count: int
    best_fitness_seen: float
    fitness_history: List[float] = field(default_factory=list)


class GRNAggDiscoverySymmetricStrategy(PaletteEvolutionStrategy):
    """GRN with sin-extreme coupling for symmetric discovery."""

    name = "grn_agg_discovery_symmetric"
    description = "Symmetric: GRN with sin-extreme regulatory coupling"

    def __init__(
        self,
        basal_expression: float = 0.2,
        expression_threshold: float = 0.4,
        expression_floor: float = 0.7,
        sin_to_max_coupling: float = 0.5,
        sin_to_min_coupling: float = 0.5,
        sin_extreme_lr_multiplier: float = 1.5,
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
        self.basal_expression = basal_expression
        self.expression_threshold = expression_threshold
        self.expression_floor = expression_floor
        self.sin_to_max_coupling = sin_to_max_coupling
        self.sin_to_min_coupling = sin_to_min_coupling
        self.sin_extreme_lr_multiplier = sin_extreme_lr_multiplier
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

    def initialize(self, config: Dict[str, Any], seed: int) -> GRNAggDiscoverySymmetricState:
        key = jax.random.PRNGKey(seed + 78100)
        act_mask = create_initial_palette_mask(self.initial_act_palette)
        agg_mask = create_initial_agg_palette_mask(self.initial_agg_palette)

        act_expr = jnp.ones(NUM_ACTIVATIONS) * self.basal_expression
        agg_expr = jnp.ones(NUM_AGGREGATIONS) * self.basal_expression
        act_aff = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_aff = jnp.ones(NUM_AGGREGATIONS) * 0.5

        act_expr, agg_expr, act_aff, agg_aff = self._apply_floors(act_expr, agg_expr, act_aff, agg_aff)

        # Cross-regulation with hardcoded sin-extreme links
        cross_reg = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.3
        cross_reg = cross_reg.at[SIN_IDX, MAX_IDX].set(self.sin_to_max_coupling)
        cross_reg = cross_reg.at[SIN_IDX, MIN_IDX].set(self.sin_to_min_coupling)

        return GRNAggDiscoverySymmetricState(
            act_mask=act_mask, agg_mask=agg_mask,
            act_expression=act_expr, agg_expression=agg_expr,
            cross_regulation=cross_reg,
            act_affinity=act_aff, agg_affinity=agg_aff,
            act_memory_cells=jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_),
            agg_memory_cells=jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_),
            act_memory_counts=jnp.zeros(NUM_ACTIVATIONS),
            agg_memory_counts=jnp.zeros(NUM_AGGREGATIONS),
            sin_extreme_coupling_events=0,
            rng_key=key, generation=0, stagnation_count=0, best_fitness_seen=0.0,
            fitness_history=[],
        )

    def get_active_palette(self, state) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _select_palette(self, expression, affinity, n_funcs, protected_indices, key):
        scores = 0.5 * expression + 0.5 * affinity
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

        # Expression update with sin-extreme coupling
        sin_expr = state.act_expression[SIN_IDX]
        delta = improvement if improvement > 0 else -0.01

        new_act_expr = state.act_expression * 0.95 + delta * 0.1 * state.act_mask
        new_agg_expr = state.agg_expression * 0.95 + delta * 0.1 * state.agg_mask

        # Boost extreme agg expression when sin is active
        for idx in [MAX_IDX, MIN_IDX]:
            coupling = self.sin_to_max_coupling if idx == MAX_IDX else self.sin_to_min_coupling
            new_agg_expr = new_agg_expr.at[idx].add(sin_expr * coupling * 0.1)

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

        # Update cross-regulation with boosted sin-extreme learning
        new_cross_reg = state.cross_regulation * self.regulation_decay
        coupling_events = state.sin_extreme_coupling_events
        if improvement > 0:
            active_act = (state.act_mask > 0.5).astype(jnp.float32)
            active_agg = (state.agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active_act, active_agg)
            new_cross_reg = new_cross_reg + self.regulation_learning_rate * improvement * co_active

            # Boosted sin-extreme learning
            if state.act_mask[SIN_IDX] > 0.5:
                for idx in [MAX_IDX, MIN_IDX]:
                    if state.agg_mask[idx] > 0.5:
                        boost = self.regulation_learning_rate * improvement * self.sin_extreme_lr_multiplier
                        new_cross_reg = new_cross_reg.at[SIN_IDX, idx].add(boost)
                        coupling_events += 1

        # Memory cells
        act_candidate = (state.act_mask > 0.5) & (new_act_aff >= self.memory_formation_threshold)
        agg_candidate = (state.agg_mask > 0.5) & (new_agg_aff >= self.memory_formation_threshold)
        new_act_mem_counts = jnp.where(act_candidate, state.act_memory_counts + 1, 0)
        new_agg_mem_counts = jnp.where(agg_candidate, state.agg_memory_counts + 1, 0)
        new_act_mem = jnp.logical_or(state.act_memory_cells, new_act_mem_counts >= self.memory_formation_count)
        new_agg_mem = jnp.logical_or(state.agg_memory_cells, new_agg_mem_counts >= self.memory_formation_count)

        # Select palettes
        new_act_mask = self._select_palette(new_act_expr, new_act_aff, NUM_ACTIVATIONS, [SIN_IDX], k_act)
        new_agg_mask = self._select_palette(new_agg_expr, new_agg_aff, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX], k_agg)
        new_act_mask = jnp.where(new_act_mem, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem, 1.0, new_agg_mask)

        new_state = GRNAggDiscoverySymmetricState(
            act_mask=new_act_mask, agg_mask=new_agg_mask,
            act_expression=new_act_expr, agg_expression=new_agg_expr,
            cross_regulation=new_cross_reg,
            act_affinity=new_act_aff, agg_affinity=new_agg_aff,
            act_memory_cells=new_act_mem, agg_memory_cells=new_agg_mem,
            act_memory_counts=new_act_mem_counts, agg_memory_counts=new_agg_mem_counts,
            sin_extreme_coupling_events=coupling_events,
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
            'sin_max_coupling': float(new_cross_reg[SIN_IDX, MAX_IDX]),
            'sin_min_coupling': float(new_cross_reg[SIN_IDX, MIN_IDX]),
        }
        return new_state, metrics

    def get_state_summary(self, state) -> Dict[str, Any]:
        return {'strategy': self.name, 'generation': state.generation}
