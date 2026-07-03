"""Strategy 3: Asymmetric Rates + Sticky Discovery.

Key improvements over baseline:
1. Higher rate (25%) to ACTIVATE inactive functions
2. Lower rate (5%) to DEACTIVATE active functions
3. Sticky oscillatory functions - once discovered, can't be deactivated
4. Mutations every generation (not just on stagnation)

Expected: 80%+ discovery rate, <30 generations
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    DEFAULT_PALETTE_INDICES,
)


class AsymmetricStickyStrategy(PaletteEvolutionStrategy):
    """Asymmetric mutation rates with sticky discoveries.

    Uses higher activation rate than deactivation rate, ensuring
    exploration of new functions while preserving useful discoveries.
    """

    name = "asymmetric_sticky"
    description = "25% activate, 5% deactivate, sticky oscillatory [4,11,12]"

    def __init__(
        self,
        activate_rate: float = 0.25,
        deactivate_rate: float = 0.05,
        sticky_functions: List[int] = None,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            activate_rate: Rate to activate inactive functions (default 25%)
            deactivate_rate: Rate to deactivate active functions (default 5%)
            sticky_functions: Functions that can't be deactivated once discovered
                            Default: [4, 11, 12] (sin, burst, resonator)
            min_active: Minimum number of active functions
            initial_palette: Starting palette indices
        """
        self.activate_rate = activate_rate
        self.deactivate_rate = deactivate_rate
        self.sticky_functions = sticky_functions or [4, 11, 12]
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with discovery tracking.

        Args:
            config: Configuration dict
            seed: Random seed

        Returns:
            Initial state with mask and discovery history
        """
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Track ever-discovered functions
        discovery_history = mask > 0.5

        return {
            'mask': mask,
            'discovery_history': discovery_history,
            'rng_key': jax.random.PRNGKey(seed + 33333),  # Offset for uniqueness
            'generation': 0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        discovery_history: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Apply asymmetric mutation with sticky discoveries.

        Args:
            key: JAX random key
            mask: Current palette mask
            discovery_history: Which functions have ever been discovered

        Returns:
            Tuple of (new_mask, new_discovery, mutation_info)
        """
        key1, key2 = jax.random.split(key)

        # Generate random values for each activation slot
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        new_mask = mask.copy()

        # Track what changed
        activated = []
        deactivated = []

        # Higher rate for activating INACTIVE functions (exploring)
        inactive_mask = mask < 0.5
        for i in range(NUM_ACTIVATIONS):
            if inactive_mask[i] and activate_probs[i] < self.activate_rate:
                new_mask = new_mask.at[i].set(1.0)
                activated.append(i)

        # Update discovery history
        new_discovery = discovery_history | (new_mask > 0.5)

        # Lower rate for deactivating ACTIVE functions (conservative)
        active_mask = mask > 0.5
        for i in range(NUM_ACTIVATIONS):
            if active_mask[i] and deactivate_probs[i] < self.deactivate_rate:
                # Check sticky constraint
                if i in self.sticky_functions and new_discovery[i]:
                    # Never deactivate sticky functions once discovered
                    continue
                new_mask = new_mask.at[i].set(0.0)
                deactivated.append(i)

        # Ensure minimum active constraint
        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active:
            # Revert to original mask if we'd go below minimum
            new_mask = mask
            new_discovery = discovery_history
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
            'sticky_protected': [i for i in self.sticky_functions if new_discovery[i]],
        }

        return new_mask, new_discovery, mutation_info

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update palette with asymmetric mutation EVERY generation.

        Unlike baseline, this mutates every generation, not just on stagnation.

        Args:
            state: Current state
            generation: Current generation
            best_fitness: Best fitness this gen
            prev_best_fitness: Previous best fitness
            population_data: Population data (not used)

        Returns:
            Tuple of (new_state, metrics)
        """
        key, subkey = jax.random.split(state['rng_key'])

        new_mask, new_discovery, mutation_info = self._mutate_palette(
            subkey,
            state['mask'],
            state['discovery_history'],
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        new_state = {
            'mask': new_mask,
            'discovery_history': new_discovery,
            'rng_key': key,
            'generation': generation + 1,
            'strategy_name': self.name,
        }

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': mask_to_indices(new_mask),
            'activated': mutation_info['activated'],
            'deactivated': mutation_info['deactivated'],
            'sticky_protected': mutation_info['sticky_protected'],
            'discovery_history': mask_to_indices(new_discovery),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including discovery status."""
        palette = self.get_active_palette(state)
        discovered = mask_to_indices(state['discovery_history'])

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'sin_ever_discovered': 4 in discovered,
            'discovered_functions': discovered,
            'sticky_active': [i for i in self.sticky_functions if i in discovered],
        }
