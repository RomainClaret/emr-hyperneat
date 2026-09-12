"""Strategy 4: Sin in Default Palette.

The simplest fix - just include sin in the starting palette.
No evolution needed. Tests if having sin available is sufficient.

Expected: 100% solve, <10 generations (similar to sin_only performance)
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
)


class SinDefaultStrategy(PaletteEvolutionStrategy):
    """Fixed palette that includes sin from the start.

    No palette evolution - just tests whether having sin available
    in the initial palette is sufficient for solving parity problems.
    """

    name = "sin_default"
    description = "Fixed palette [0,1,2,3,4] - default + sin, no evolution"

    def __init__(self, palette_indices: List[int] = None):
        """Initialize strategy.

        Args:
            palette_indices: Custom palette indices. Default is [0,1,2,3,4]
        """
        self.palette_indices = palette_indices or [0, 1, 2, 3, 4]

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize with fixed palette including sin.

        Args:
            config: Configuration (ignored for this strategy)
            seed: Random seed (not used since no mutation)

        Returns:
            State dict with fixed palette
        """
        mask = create_initial_palette_mask(self.palette_indices)

        return {
            'mask': mask,
            'generation': 0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return fixed palette indices.

        Args:
            state: Current state

        Returns:
            List of active indices (always same)
        """
        return mask_to_indices(state['mask'])

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """No-op update - palette is fixed.

        Args:
            state: Current state
            generation: Current generation
            best_fitness: Best fitness this gen
            prev_best_fitness: Previous best fitness
            population_data: Population data (ignored)

        Returns:
            Tuple of (unchanged state, empty metrics)
        """
        new_state = {
            **state,
            'generation': generation + 1,
        }

        metrics = {
            'palette_changed': False,
            'current_palette': self.get_active_palette(state),
            'mutation_event': None,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary.

        Args:
            state: Current state

        Returns:
            Summary dict
        """
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'palette_size': len(self.palette_indices),
            'has_sin': 4 in self.palette_indices,
            'evolution_enabled': False,
        }
