"""Strategy 80S: Hebbian Aggregation Discovery Symmetric (Cross-Domain Co-occurrence).

Biological Basis: Synaptic co-occurrence learning - "cells that fire together wire together."
Extended to cross-domain: activation-aggregation pairs that succeed together become linked.
In symmetric search, this extends to discovering BOTH activation AND aggregation functions
through unified Hebbian learning.

Key mechanisms for SYMMETRIC discovery:
1. 18x6 cross-domain Hebbian matrix tracking activation-aggregation pairs
2. PLUS within-domain Hebbian for both activation and aggregation discovery
3. Boosted learning rate for sin-extreme pairs (1.5x multiplier)
4. Cross-domain consolidation protects successful pairs from mutation
5. Protected indices (0.1% deactivation) for sin and extreme aggregations
6. Memory cell crystallization for sustained high-Hebbian functions
7. Affinity floors for guaranteed retention

Expected: Better sin-max/min synergy discovery through explicit bidirectional pair learning.
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)

# Critical indices for retention
SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3


class HebbianAggDiscoverySymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with cross-domain Hebbian learning.

    Discovers BOTH activation and aggregation functions through unified
    evolution guided by Hebbian co-occurrence learning. When pairs of
    functions succeed together, their association is strengthened.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Bidirectional Hebbian updates
    - Discovery tracking for research analysis
    """

    name = "hebbian_agg_discovery_symmetric"
    description = "Symmetric: Cross-domain Hebbian learning for bidirectional discovery"

    def __init__(
        self,
        # Hebbian parameters
        learning_rate: float = 0.10,
        cross_learning_rate: float = 0.12,
        decay_rate: float = 0.02,
        consolidation_threshold: float = 0.65,
        # Sin-extreme boost
        sin_extreme_lr_multiplier: float = 1.5,
        # Mutation parameters
        base_activate_rate: float = 0.15,
        base_deactivate_rate: float = 0.08,
        protection_reduction: float = 0.7,
        # Stagnation
        stagnation_threshold: int = 5,
        consolidation_gens: int = 3,
        # Affinity floors for guaranteed retention
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Memory cell parameters
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Initial palettes - INCLUDE critical functions for retention
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Hebbian Aggregation Discovery Symmetric strategy.

        Args:
            learning_rate: Within-domain Hebbian learning rate
            cross_learning_rate: Cross-domain Hebbian learning rate
            decay_rate: Hebbian weight decay per generation
            consolidation_threshold: Threshold for consolidation
            sin_extreme_lr_multiplier: Learning boost for sin-extreme pairs
            base_activate_rate: Base activation probability
            base_deactivate_rate: Base deactivation probability
            protection_reduction: Deactivation reduction for protected functions
            stagnation_threshold: Generations without improvement before mutation
            consolidation_gens: Generations to consolidate
            sin_affinity_floor: Minimum affinity for sin function
            extreme_agg_affinity_floor: Minimum affinity for max/min aggregations
            memory_formation_threshold: Hebbian threshold for memory candidacy
            memory_formation_count: Sustained generations to become memory cell
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            initial_act_palette: Starting activation palette indices (includes sin by default)
            initial_agg_palette: Starting aggregation palette indices (includes max/min by default)
        """
        # Hebbian
        self.learning_rate = learning_rate
        self.cross_learning_rate = cross_learning_rate
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold

        # Sin-extreme boost
        self.sin_extreme_lr_multiplier = sin_extreme_lr_multiplier

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.protection_reduction = protection_reduction

        # Stagnation
        self.stagnation_threshold = stagnation_threshold
        self.consolidation_gens = consolidation_gens

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Memory cells
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # CRITICAL: Include sin (idx 4) and extreme aggs (idx 2,3) in initial palettes
        # This ensures they start active and can be protected from the beginning
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]  # tanh, sigmoid, relu, identity, sin
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]  # sum, mean, max, min

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with Hebbian matrices and memory tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Hebbian weight matrices
        act_hebbian = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        agg_hebbian = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        cross_hebbian = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Individual affinities derived from Hebbian weights
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Apply affinity floors immediately
        act_affinity = act_affinity.at[SIN_IDX].set(
            max(self.sin_affinity_floor, float(act_affinity[SIN_IDX]))
        )
        for idx in [MAX_IDX, MIN_IDX]:
            agg_affinity = agg_affinity.at[idx].set(
                max(self.extreme_agg_affinity_floor, float(agg_affinity[idx]))
            )

        # Consolidation tracking
        act_consolidated = jnp.zeros(NUM_ACTIVATIONS)
        agg_consolidated = jnp.zeros(NUM_AGGREGATIONS)
        cross_consolidated = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        consolidation_count = jnp.zeros(NUM_ACTIVATIONS + NUM_AGGREGATIONS)

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        discovery_gen = {
            'sin': -1,
            'max': -1,
            'min': -1,
        }

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Hebbian
            'act_hebbian': act_hebbian,
            'agg_hebbian': agg_hebbian,
            'cross_hebbian': cross_hebbian,
            # Affinities
            'act_affinity': act_affinity,
            'agg_affinity': agg_affinity,
            # Consolidation
            'act_consolidated': act_consolidated,
            'agg_consolidated': agg_consolidated,
            'cross_consolidated': cross_consolidated,
            'consolidation_count': consolidation_count,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Discovery tracking
            'discovery_gen': discovery_gen,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 80000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply minimum affinity floors for critical functions."""
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )
        new_agg = agg_affinity
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )
        return new_act, new_agg

    def _update_memory_cells(
        self,
        affinity: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell status based on sustained high affinity."""
        active = mask > 0.5
        above_threshold = affinity >= self.memory_formation_threshold

        candidate = active & above_threshold
        new_counts = jnp.where(candidate, memory_counts + 1, 0)

        newly_memory = new_counts >= self.memory_formation_count
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _update_hebbian_weights(
        self,
        hebbian: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_delta: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update within-domain Hebbian weights."""
        new_hebbian = hebbian * (1 - self.decay_rate)

        if fitness_delta > 0:
            active = (mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active, active)
            co_active = co_active * (1 - jnp.eye(n_funcs))
            delta = self.learning_rate * fitness_delta * co_active
            new_hebbian = new_hebbian + delta

        return jnp.clip(new_hebbian, 0.0, 1.0)

    def _update_cross_hebbian(
        self,
        cross_hebbian: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain Hebbian weights with sin-extreme boost."""
        new_cross = cross_hebbian * (1 - self.decay_rate)

        if fitness_delta > 0:
            active_act = (act_mask > 0.5).astype(jnp.float32)
            active_agg = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active_act, active_agg)

            delta = self.cross_learning_rate * fitness_delta * co_active
            new_cross = new_cross + delta

            # BOOSTED sin-extreme learning
            sin_active = act_mask[SIN_IDX] > 0.5
            if sin_active:
                for agg_idx in CORE_EXTREME_AGGS:
                    if agg_mask[agg_idx] > 0.5:
                        boost = self.cross_learning_rate * fitness_delta * self.sin_extreme_lr_multiplier
                        current = new_cross[SIN_IDX, agg_idx]
                        new_cross = new_cross.at[SIN_IDX, agg_idx].set(current + boost)

        return jnp.clip(new_cross, 0.0, 1.0)

    def _update_affinities_from_hebbian(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
        act_hebbian: jnp.ndarray,
        agg_hebbian: jnp.ndarray,
        cross_hebbian: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update individual affinities based on Hebbian weights.

        Functions with strong Hebbian connections to active functions
        get affinity boosts.
        """
        # Activation affinity from within-domain and cross-domain
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)

        # Within-domain contribution
        within_act = jnp.sum(act_hebbian * active_act[None, :], axis=1) / max(1, int(jnp.sum(active_act)))

        # Cross-domain contribution (how well does each activation connect to active aggs?)
        cross_act = jnp.sum(cross_hebbian * active_agg[None, :], axis=1) / max(1, int(jnp.sum(active_agg)))

        # Combined affinity update
        new_act_aff = act_affinity * 0.95 + 0.025 * within_act + 0.025 * cross_act

        # Aggregation affinity
        within_agg = jnp.sum(agg_hebbian * active_agg[None, :], axis=1) / max(1, int(jnp.sum(active_agg)))
        cross_agg = jnp.sum(cross_hebbian.T * active_act[None, :], axis=1) / max(1, int(jnp.sum(active_act)))

        new_agg_aff = agg_affinity * 0.95 + 0.025 * within_agg + 0.025 * cross_agg

        return jnp.clip(new_act_aff, 0.0, 1.0), jnp.clip(new_agg_aff, 0.0, 1.0)

    def _update_consolidation(
        self,
        consolidated: jnp.ndarray,
        hebbian: jnp.ndarray,
        mask: jnp.ndarray,
        consolidation_count: jnp.ndarray,
        start_idx: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update consolidation status based on Hebbian weights."""
        new_consolidated = consolidated.copy()
        new_count = consolidation_count.copy()

        for i in range(n_funcs):
            if mask[i] > 0.5:
                other_active = jnp.where((mask > 0.5) & (jnp.arange(n_funcs) != i), 1.0, 0.0)
                if jnp.sum(other_active) > 0:
                    avg_weight = float(jnp.sum(hebbian[i] * other_active) / jnp.sum(other_active))
                    if avg_weight >= self.consolidation_threshold:
                        new_count = new_count.at[start_idx + i].set(
                            new_count[start_idx + i] + 1
                        )
                        if new_count[start_idx + i] >= self.consolidation_gens:
                            new_consolidated = new_consolidated.at[i].set(1.0)

        return new_consolidated, new_count

    def _update_cross_consolidation(
        self,
        cross_consolidated: jnp.ndarray,
        cross_hebbian: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update cross-domain consolidation."""
        new_consolidated = cross_consolidated.copy()

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                for j in range(NUM_AGGREGATIONS):
                    if agg_mask[j] > 0.5:
                        if cross_hebbian[i, j] >= self.consolidation_threshold:
                            new_consolidated = new_consolidated.at[i, j].set(1.0)

        return new_consolidated

    def _compute_protection_score(
        self,
        idx: int,
        is_act: bool,
        state: Dict[str, Any],
    ) -> float:
        """Compute protection score for a function."""
        if is_act:
            consolidated = state['act_consolidated'][idx] > 0.5
            cross_protected = jnp.any(state['cross_consolidated'][idx] > 0.5)
            memory_cell = state['act_memory_cells'][idx]
        else:
            consolidated = state['agg_consolidated'][idx] > 0.5
            cross_protected = jnp.any(state['cross_consolidated'][:, idx] > 0.5)
            memory_cell = state['agg_memory_cells'][idx]

        if memory_cell or consolidated or cross_protected:
            return 1.0
        return 0.0

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        hebbian: jnp.ndarray,
        is_act: bool,
        state: Dict[str, Any],
        protected_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply Hebbian-guided mutation with protection."""
        protected_indices = protected_indices or []
        n_funcs = NUM_ACTIVATIONS if is_act else NUM_AGGREGATIONS
        min_active = self.min_active_act if is_act else self.min_active_agg
        max_active = self.max_active_act if is_act else self.max_active_agg

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(n_funcs):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            protection = self._compute_protection_score(i, is_act, state)

            if mask[i] < 0.5:  # Inactive
                # Activate based on Hebbian affinity with active functions
                active_weights = hebbian[i] * (mask > 0.5).astype(jnp.float32)
                avg_affinity = float(jnp.mean(active_weights)) if jnp.any(mask > 0.5) else 0.5
                activate_rate = self.base_activate_rate * (1 + avg_affinity + float(affinity[i]))

                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                # Base deactivation rate
                if protection > 0.5:
                    deactivate_rate = self.base_deactivate_rate * (1 - self.protection_reduction)
                else:
                    deactivate_rate = self.base_deactivate_rate * (1 - float(affinity[i]) * 0.5)

                # Protected indices get very low deactivation (0.1%)
                if i in protected_indices:
                    deactivate_rate = 0.001

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < min_active or active_count > max_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'protected_count': len(protected_indices),
        }

    def _update_discovery_tracking(
        self,
        discovery_gen: Dict[str, int],
        act_palette: List[int],
        agg_palette: List[int],
        generation: int,
    ) -> Dict[str, int]:
        """Track when critical functions are first discovered."""
        new_discovery = discovery_gen.copy()

        if SIN_IDX in act_palette and new_discovery['sin'] < 0:
            new_discovery['sin'] = generation
        if MAX_IDX in agg_palette and new_discovery['max'] < 0:
            new_discovery['max'] = generation
        if MIN_IDX in agg_palette and new_discovery['min'] < 0:
            new_discovery['min'] = generation

        return new_discovery

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with Hebbian learning and cross-domain consolidation."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update Hebbian weights
        new_act_hebbian = self._update_hebbian_weights(
            state['act_hebbian'], state['act_mask'], fitness_delta, NUM_ACTIVATIONS
        )
        new_agg_hebbian = self._update_hebbian_weights(
            state['agg_hebbian'], state['agg_mask'], fitness_delta, NUM_AGGREGATIONS
        )
        new_cross_hebbian = self._update_cross_hebbian(
            state['cross_hebbian'], state['act_mask'], state['agg_mask'], fitness_delta
        )

        # Update affinities from Hebbian
        new_act_aff, new_agg_aff = self._update_affinities_from_hebbian(
            state['act_affinity'],
            state['agg_affinity'],
            new_act_hebbian,
            new_agg_hebbian,
            new_cross_hebbian,
            state['act_mask'],
            state['agg_mask'],
        )

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_aff, state['act_memory_counts'], state['act_memory_cells'], state['act_mask']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_aff, state['agg_memory_counts'], state['agg_memory_cells'], state['agg_mask']
        )

        # Update consolidation
        new_act_consolidated, new_count = self._update_consolidation(
            state['act_consolidated'], new_act_hebbian, state['act_mask'],
            state['consolidation_count'], 0, NUM_ACTIVATIONS
        )
        new_agg_consolidated, new_count = self._update_consolidation(
            state['agg_consolidated'], new_agg_hebbian, state['agg_mask'],
            new_count, NUM_ACTIVATIONS, NUM_AGGREGATIONS
        )
        new_cross_consolidated = self._update_cross_consolidation(
            state['cross_consolidated'], new_cross_hebbian,
            state['act_mask'], state['agg_mask']
        )

        # Create intermediate state for protection scoring
        intermediate_state = {
            **state,
            'act_consolidated': new_act_consolidated,
            'agg_consolidated': new_agg_consolidated,
            'cross_consolidated': new_cross_consolidated,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
        }

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None

        if new_stagnation >= self.stagnation_threshold:
            # Protected indices
            protected_act = [SIN_IDX]
            protected_agg = [MAX_IDX, MIN_IDX]

            new_act_mask, act_mutation_info = self._mutate_palette(
                k_act, state['act_mask'], new_act_aff, new_act_hebbian,
                True, intermediate_state, protected_indices=protected_act
            )
            new_agg_mask, agg_mutation_info = self._mutate_palette(
                k_agg, state['agg_mask'], new_agg_aff, new_agg_hebbian,
                False, intermediate_state, protected_indices=protected_agg
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            new_stagnation = 0

        # Get current palettes
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Update discovery tracking
        new_discovery = self._update_discovery_tracking(
            state['discovery_gen'], act_palette, agg_palette, generation
        )

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_hebbian': new_act_hebbian,
            'agg_hebbian': new_agg_hebbian,
            'cross_hebbian': new_cross_hebbian,
            'act_affinity': new_act_aff,
            'agg_affinity': new_agg_aff,
            'act_consolidated': new_act_consolidated,
            'agg_consolidated': new_agg_consolidated,
            'cross_consolidated': new_cross_consolidated,
            'consolidation_count': new_count,
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'discovery_gen': new_discovery,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        # Compute metrics
        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'mutation_triggered': act_mutation_info is not None,
            # Hebbian metrics
            'act_mean_hebbian': float(jnp.mean(new_act_hebbian)),
            'agg_mean_hebbian': float(jnp.mean(new_agg_hebbian)),
            'cross_mean_hebbian': float(jnp.mean(new_cross_hebbian)),
            # Sin-extreme cross hebbian
            'sin_max_hebbian': float(new_cross_hebbian[SIN_IDX, MAX_IDX]),
            'sin_min_hebbian': float(new_cross_hebbian[SIN_IDX, MIN_IDX]),
            # Consolidation
            'act_n_consolidated': int(jnp.sum(new_act_consolidated > 0.5)),
            'agg_n_consolidated': int(jnp.sum(new_agg_consolidated > 0.5)),
            'cross_n_consolidated': int(jnp.sum(new_cross_consolidated > 0.5)),
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Sin status
            'has_sin': SIN_IDX in act_palette,
            'sin_consolidated': new_act_consolidated[SIN_IDX] > 0.5,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            # Discovery tracking
            'sin_discovered_gen': new_discovery['sin'],
            'max_discovered_gen': new_discovery['max'],
            'min_discovered_gen': new_discovery['min'],
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with Hebbian status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        act_mem_count = int(jnp.sum(state['act_memory_cells']))
        agg_mem_count = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            # Hebbian
            'sin_max_hebbian': float(state['cross_hebbian'][SIN_IDX, MAX_IDX]),
            'sin_min_hebbian': float(state['cross_hebbian'][SIN_IDX, MIN_IDX]),
            # Affinities
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'max_affinity': float(state['agg_affinity'][MAX_IDX]),
            'min_affinity': float(state['agg_affinity'][MIN_IDX]),
            # Consolidation
            'act_n_consolidated': int(jnp.sum(state['act_consolidated'] > 0.5)),
            'agg_n_consolidated': int(jnp.sum(state['agg_consolidated'] > 0.5)),
            'cross_n_consolidated': int(jnp.sum(state['cross_consolidated'] > 0.5)),
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Discovery
            'discovery_gen': state['discovery_gen'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
        }
