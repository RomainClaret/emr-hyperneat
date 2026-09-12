"""Strategy 49D: Sin Default Dual (Fixed Palette with Sin for Both Domains).

Extends SinDefaultStrategy to include BOTH activation AND aggregation palettes.
Fixed palettes with no evolution - tests if having sin and full aggregation
available from the start is sufficient.

Key dual mechanisms:
1. Fixed activation palette including sin
2. Fixed aggregation palette with all common aggregations
3. Cross-domain affinity tracking (for analysis only)
4. No mutation - pure baseline reference

Expected: 100% solve with sin, baseline aggregation performance
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
)


class SinDefaultDualStrategy(PaletteEvolutionStrategy):
    """Fixed dual palette that includes sin and extended aggregations.

    No palette evolution - tests whether having sin and good aggregations
    available in the initial palette is sufficient for solving problems.
    """

    name = "sin_default_dual"
    description = "Fixed dual palette [act: 0,1,2,3,4] [agg: 0,1,2,3] - no evolution"

    def __init__(
        self,
        act_palette_indices: List[int] = None,
        agg_palette_indices: List[int] = None,
        cross_learning_rate: float = 0.05,
    ):
        """Initialize strategy.

        Args:
            act_palette_indices: Activation palette indices. Default [0,1,2,3,4]
            agg_palette_indices: Aggregation palette indices. Default [0,1,2,3]
            cross_learning_rate: Rate for cross-domain tracking (analysis only)
        """
        self.act_palette_indices = act_palette_indices or [0, 1, 2, 3, 4]  # default + sin
        self.agg_palette_indices = agg_palette_indices or [0, 1, 2, 3]  # sum, mean, max, min
        self.cross_learning_rate = cross_learning_rate

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize with fixed palettes.

        Args:
            config: Configuration (mostly ignored)
            seed: Random seed (for cross-affinity tracking only)

        Returns:
            State dict with fixed palettes
        """
        act_mask = create_initial_palette_mask(self.act_palette_indices)
        agg_mask = create_initial_agg_palette_mask(self.agg_palette_indices)

        # Cross-domain affinity for tracking only
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'cross_affinity': cross_affinity,
            'generation': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return fixed activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return fixed aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity (for analysis only)."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        delta = self.cross_learning_rate * fitness_delta * co_active
        new_cross = cross_affinity + delta
        return jnp.clip(new_cross, 0.0, 1.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """No-op update for palettes, but track cross-affinity.

        Args:
            state: Current state
            generation: Current generation
            best_fitness: Best fitness this gen
            prev_best_fitness: Previous best fitness
            population_data: Population data (ignored)

        Returns:
            Tuple of (unchanged palettes, metrics with cross-domain info)
        """
        fitness_delta = best_fitness - prev_best_fitness
        improved = best_fitness > state['best_fitness_seen']
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Update cross-affinity for analysis
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        new_state = {
            'act_mask': state['act_mask'],  # Fixed
            'agg_mask': state['agg_mask'],  # Fixed
            'cross_affinity': new_cross,
            'generation': generation + 1,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        metrics = {
            'palette_changed': False,
            'agg_palette_changed': False,
            'current_palette': self.get_active_palette(state),
            'current_agg_palette': self.get_active_agg_palette(state),
            'fitness_improved': improved,
            'mutation_event': None,
            # Cross-domain metrics
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # Sin status
            'has_sin': 4 in self.act_palette_indices,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'act_palette_size': len(self.act_palette_indices),
            'agg_palette_size': len(self.agg_palette_indices),
            'has_sin': 4 in self.act_palette_indices,
            'evolution_enabled': False,
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
