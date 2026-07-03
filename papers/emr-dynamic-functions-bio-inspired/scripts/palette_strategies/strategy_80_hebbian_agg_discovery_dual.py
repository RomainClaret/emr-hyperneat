"""Strategy 80: Hebbian Aggregation Discovery Dual (Cross-Domain Co-occurrence).

Bio inspiration: Synaptic co-occurrence learning - "cells that fire together wire together."
Extended to cross-domain: activation-aggregation pairs that succeed together become linked.

Key innovation:
- 18x6 cross-domain Hebbian matrix tracking activation-aggregation pairs
- Boosted learning rate for sin-extreme pairs (1.5x multiplier)
- Cross-domain consolidation protects successful pairs from mutation

Expected: Better sin-max/min synergy discovery through explicit pair learning.
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


class HebbianAggDiscoveryDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with cross-domain Hebbian learning.

    Tracks which activation-aggregation pairs succeed together and uses
    this information to guide discovery and protection.
    """

    name = "hebbian_agg_discovery_dual"
    description = "Dual: Cross-domain Hebbian learning for act-agg pair discovery"

    def __init__(
        self,
        # Hebbian parameters
        learning_rate: float = 0.10,
        cross_learning_rate: float = 0.12,  # Higher for cross-domain
        decay_rate: float = 0.02,
        consolidation_threshold: float = 0.65,
        # Sin-extreme boost
        sin_extreme_lr_multiplier: float = 1.5,
        # Mutation parameters
        base_activate_rate: float = 0.15,
        base_deactivate_rate: float = 0.08,
        protection_reduction: float = 0.7,  # Reduce deactivation by 70% if protected
        # Stagnation
        stagnation_threshold: int = 5,
        consolidation_gens: int = 3,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Hebbian Aggregation Discovery Dual strategy."""
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

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with Hebbian matrices."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Hebbian weight matrices
        act_hebbian = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        agg_hebbian = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        cross_hebbian = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Consolidation tracking
        act_consolidated = jnp.zeros(NUM_ACTIVATIONS)
        agg_consolidated = jnp.zeros(NUM_AGGREGATIONS)
        cross_consolidated = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Hebbian
            'act_hebbian': act_hebbian,
            'agg_hebbian': agg_hebbian,
            'cross_hebbian': cross_hebbian,
            # Consolidation
            'act_consolidated': act_consolidated,
            'agg_consolidated': agg_consolidated,
            'cross_consolidated': cross_consolidated,
            'consolidation_count': jnp.zeros(NUM_ACTIVATIONS + NUM_AGGREGATIONS),
            # General state
            'rng_key': jax.random.PRNGKey(seed + 800000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_hebbian_weights(
        self,
        hebbian: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_delta: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update within-domain Hebbian weights."""
        # Decay
        new_hebbian = hebbian * (1 - self.decay_rate)

        # Strengthen co-active pairs on improvement
        if fitness_delta > 0:
            active = (mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active, active)
            # Remove self-connections
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
        # Decay
        new_cross = cross_hebbian * (1 - self.decay_rate)

        # Strengthen co-active pairs on improvement
        if fitness_delta > 0:
            active_act = (act_mask > 0.5).astype(jnp.float32)
            active_agg = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active_act, active_agg)

            # Base learning
            delta = self.cross_learning_rate * fitness_delta * co_active
            new_cross = new_cross + delta

            # BOOSTED sin-extreme learning
            sin_active = act_mask[4] > 0.5
            if sin_active:
                for agg_idx in CORE_EXTREME_AGGS:
                    if agg_mask[agg_idx] > 0.5:
                        boost = self.cross_learning_rate * fitness_delta * self.sin_extreme_lr_multiplier
                        current = new_cross[4, agg_idx]
                        new_cross = new_cross.at[4, agg_idx].set(current + boost)

        return jnp.clip(new_cross, 0.0, 1.0)

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
                # Average Hebbian weight with other active functions
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
        """Compute protection score for a function based on Hebbian/consolidation."""
        if is_act:
            consolidated = state['act_consolidated'][idx] > 0.5
            # Check cross-domain protection
            cross_protected = jnp.any(state['cross_consolidated'][idx] > 0.5)
        else:
            consolidated = state['agg_consolidated'][idx] > 0.5
            # Check cross-domain protection
            cross_protected = jnp.any(state['cross_consolidated'][:, idx] > 0.5)

        if consolidated or cross_protected:
            return 1.0
        return 0.0

    def _mutate_palette_hebbian(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        hebbian: jnp.ndarray,
        consolidated: jnp.ndarray,
        is_act: bool,
        state: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply Hebbian-guided mutation."""
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
                activate_rate = self.base_activate_rate * (1 + avg_affinity)

                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                # Reduce deactivation if protected
                if protection > 0.5:
                    deactivate_rate = self.base_deactivate_rate * (1 - self.protection_reduction)
                else:
                    deactivate_rate = self.base_deactivate_rate

                # Extra protection for sin (activation) and extreme (aggregation)
                if is_act and i == 4:  # sin
                    deactivate_rate *= 0.5
                elif not is_act and i in CORE_EXTREME_AGGS:
                    deactivate_rate *= 0.5

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < min_active or active_count > max_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

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
        }

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_palette_hebbian(
                k_act, state['act_mask'], new_act_hebbian, new_act_consolidated,
                True, intermediate_state
            )
            new_agg_mask, agg_mutation_info = self._mutate_palette_hebbian(
                k_agg, state['agg_mask'], new_agg_hebbian, new_agg_consolidated,
                False, intermediate_state
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            new_stagnation = 0

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_hebbian': new_act_hebbian,
            'agg_hebbian': new_agg_hebbian,
            'cross_hebbian': new_cross_hebbian,
            'act_consolidated': new_act_consolidated,
            'agg_consolidated': new_agg_consolidated,
            'cross_consolidated': new_cross_consolidated,
            'consolidation_count': new_count,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

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
            'sin_max_hebbian': float(new_cross_hebbian[4, 2]),
            'sin_min_hebbian': float(new_cross_hebbian[4, 3]),
            # Consolidation
            'act_n_consolidated': int(jnp.sum(new_act_consolidated > 0.5)),
            'agg_n_consolidated': int(jnp.sum(new_agg_consolidated > 0.5)),
            'cross_n_consolidated': int(jnp.sum(new_cross_consolidated > 0.5)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_consolidated': new_act_consolidated[4] > 0.5,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
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

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'sin_max_hebbian': float(state['cross_hebbian'][4, 2]),
            'sin_min_hebbian': float(state['cross_hebbian'][4, 3]),
            'act_n_consolidated': int(jnp.sum(state['act_consolidated'] > 0.5)),
            'agg_n_consolidated': int(jnp.sum(state['agg_consolidated'] > 0.5)),
            'cross_n_consolidated': int(jnp.sum(state['cross_consolidated'] > 0.5)),
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
        }
