"""Strategy 1: Baseline (Current Implementation).

The existing palette evolution mechanism:
- Uniform 10% mutation rate per activation
- Stagnation-triggered (5 generations without improvement)
- No persistence guarantee
- No fitness guidance

Expected: 33% discovery rate, ~48 generations when successful
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


class BaselineStrategy(PaletteEvolutionStrategy):
    """Baseline palette evolution - current implementation.

    Uniform symmetric mutation triggered by fitness stagnation.
    This is the strategy we're trying to improve upon.
    """

    name = "baseline"
    description = "Current: 10% uniform, stagnation-triggered (5 gens)"

    def __init__(
        self,
        mutation_rate: float = 0.1,
        stagnation_threshold: int = 5,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            mutation_rate: Probability of flipping each activation (default 10%)
            stagnation_threshold: Generations without improvement before mutation
            min_active: Minimum number of active functions
            initial_palette: Starting palette indices
        """
        self.mutation_rate = mutation_rate
        self.stagnation_threshold = stagnation_threshold
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with stagnation tracking.

        Args:
            config: Configuration dict
            seed: Random seed

        Returns:
            Initial state
        """
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 11111),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _mutate_palette_uniform(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply uniform symmetric mutation.

        Each activation has mutation_rate chance to flip (on→off or off→on).

        Args:
            key: JAX random key
            mask: Current palette mask

        Returns:
            Tuple of (new_mask, mutation_info)
        """
        flip_probs = jax.random.uniform(key, (NUM_ACTIVATIONS,))
        flip_mask = flip_probs < self.mutation_rate

        # XOR flip: active becomes inactive, inactive becomes active
        new_mask = jnp.where(flip_mask, 1.0 - mask, mask)

        # Track changes
        flipped_indices = jnp.where(flip_mask)[0].tolist()
        activated = [i for i in flipped_indices if mask[i] < 0.5]
        deactivated = [i for i in flipped_indices if mask[i] > 0.5]

        # Ensure minimum active constraint
        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active:
            new_mask = mask  # Revert
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
            'flipped_count': len(flipped_indices),
        }

        return new_mask, mutation_info

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with stagnation-triggered mutation.

        Only mutates when fitness hasn't improved for stagnation_threshold gens.

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

        # Check if fitness improved
        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Check if we should mutate
        mutation_info = None
        new_mask = state['mask']
        palette_changed = False

        if new_stagnation >= self.stagnation_threshold:
            # Trigger mutation
            new_mask, mutation_info = self._mutate_palette_uniform(
                subkey, state['mask']
            )
            palette_changed = not jnp.allclose(state['mask'], new_mask)
            new_stagnation = 0  # Reset counter after mutation

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': mask_to_indices(new_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'mutation_triggered': mutation_info is not None,
        }

        if mutation_info:
            metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including stagnation status."""
        palette = self.get_active_palette(state)

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
        }
