"""Strategy 2: Fitness-Guided Mutation.

Track which activations appear in high-fitness networks and bias
mutations toward those functions.

Key improvements over baseline:
1. Track activation usage in top 10% of population
2. Use EMA to smooth usage statistics
3. Bias activation probability toward high-usage functions
4. Still uses stagnation trigger but with guided mutations

Expected: 60%+ discovery rate, <40 generations
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


class FitnessGuidedStrategy(PaletteEvolutionStrategy):
    """Fitness-guided palette evolution.

    Tracks which activations appear in elite networks and biases
    mutations to favor those functions. Uses EMA smoothing for
    stable usage estimates.
    """

    name = "fitness_guided"
    description = "Track elite usage, bias mutations toward high-fitness activations"

    def __init__(
        self,
        elite_percentile: float = 0.1,
        ema_alpha: float = 0.1,
        base_mutation_rate: float = 0.15,
        stagnation_threshold: int = 5,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            elite_percentile: Top fraction of population to track (default 10%)
            ema_alpha: EMA smoothing factor (default 0.1)
            base_mutation_rate: Base mutation probability (default 15%)
            stagnation_threshold: Generations without improvement before mutation
            min_active: Minimum number of active functions
            initial_palette: Starting palette indices
        """
        self.elite_percentile = elite_percentile
        self.ema_alpha = ema_alpha
        self.base_mutation_rate = base_mutation_rate
        self.stagnation_threshold = stagnation_threshold
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with usage tracking.

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
            'rng_key': jax.random.PRNGKey(seed + 22222),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Usage tracking
            'activation_usage_counts': jnp.zeros(NUM_ACTIVATIONS),
            'elite_activation_weights': jnp.ones(NUM_ACTIVATIONS) / NUM_ACTIVATIONS,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_usage_stats(
        self,
        state: Dict[str, Any],
        population_data: Optional[Dict],
    ) -> Dict[str, Any]:
        """Update activation usage statistics from population.

        Args:
            state: Current state
            population_data: Dict with 'fitnesses' and optionally 'activation_indices'

        Returns:
            Updated state with new usage statistics
        """
        if population_data is None:
            return state

        fitnesses = population_data.get('fitnesses')
        if fitnesses is None:
            return state

        # Compute elite threshold
        fitnesses_array = jnp.array(fitnesses)
        elite_threshold = jnp.percentile(
            fitnesses_array,
            (1 - self.elite_percentile) * 100
        )

        # If we have activation indices, use them
        activation_indices = population_data.get('activation_indices')
        if activation_indices is not None:
            elite_mask = fitnesses_array >= elite_threshold

            # Count activations in elite networks
            new_counts = jnp.zeros(NUM_ACTIVATIONS)
            for i, (act_idx, is_elite) in enumerate(zip(activation_indices, elite_mask)):
                if is_elite:
                    # act_idx might be array or list of indices used by this network
                    if hasattr(act_idx, '__iter__'):
                        for idx in act_idx:
                            if 0 <= idx < NUM_ACTIVATIONS:
                                new_counts = new_counts.at[int(idx)].add(1)
                    else:
                        if 0 <= act_idx < NUM_ACTIVATIONS:
                            new_counts = new_counts.at[int(act_idx)].add(1)

            # Update EMA
            total = jnp.sum(new_counts)
            if total > 0:
                normalized = new_counts / total
                new_weights = (
                    (1 - self.ema_alpha) * state['elite_activation_weights'] +
                    self.ema_alpha * normalized
                )
            else:
                new_weights = state['elite_activation_weights']
        else:
            # No activation data - use fitness to guide (proxy)
            # Higher fitness → current palette is good → slightly boost current
            if jnp.max(fitnesses_array) > state['best_fitness_seen']:
                # Current palette seems good, slightly boost current activations
                current_mask = state['mask'] > 0.5
                boost = current_mask.astype(jnp.float32) * 0.1
                new_weights = state['elite_activation_weights'] + boost
                new_weights = new_weights / jnp.sum(new_weights)
            else:
                new_weights = state['elite_activation_weights']

            new_counts = state['activation_usage_counts']

        new_state = {
            **state,
            'activation_usage_counts': new_counts,
            'elite_activation_weights': new_weights,
        }

        return new_state

    def _get_mutation_bias(self, state: Dict[str, Any]) -> jnp.ndarray:
        """Get mutation bias from elite weights.

        Returns probability of activating each function.
        Higher weight = more likely to be activated.
        """
        # Softmax with temperature to convert weights to probabilities
        weights = state['elite_activation_weights']
        temperature = 0.5  # Lower = more focused on high-weight
        probs = jax.nn.softmax(weights / temperature)
        return probs

    def _mutate_palette_guided(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        state: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply fitness-guided mutation.

        Args:
            key: JAX random key
            mask: Current palette mask
            state: Current state (for usage weights)

        Returns:
            Tuple of (new_mask, mutation_info)
        """
        key1, key2 = jax.random.split(key)

        # Get biased activation probabilities
        activation_bias = self._get_mutation_bias(state)

        # For inactive functions: use bias to decide activation
        inactive_mask = mask < 0.5
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))

        # Bias: higher weight functions more likely to activate
        # Scale base rate by activation bias (multiply by NUM_ACTIVATIONS to normalize)
        biased_rates = self.base_mutation_rate * activation_bias * NUM_ACTIVATIONS

        new_mask = mask.copy()
        activated = []
        deactivated = []

        # Activate based on biased rates
        for i in range(NUM_ACTIVATIONS):
            if inactive_mask[i] and activate_probs[i] < biased_rates[i]:
                new_mask = new_mask.at[i].set(1.0)
                activated.append(i)

        # Deactivate with lower rate (conservative)
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))
        deactivate_rate = self.base_mutation_rate * 0.3  # Lower rate for deactivation

        active_mask = mask > 0.5
        for i in range(NUM_ACTIVATIONS):
            if active_mask[i] and deactivate_probs[i] < deactivate_rate:
                # Less likely to deactivate high-weight functions
                if activation_bias[i] > 0.1:  # Above average weight
                    continue
                new_mask = new_mask.at[i].set(0.0)
                deactivated.append(i)

        # Ensure minimum active constraint
        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
            'activation_bias': activation_bias.tolist(),
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
        """Update with fitness-guided mutation.

        1. Update usage statistics from population
        2. Check stagnation
        3. If stagnating, mutate with fitness guidance

        Args:
            state: Current state
            generation: Current generation
            best_fitness: Best fitness this gen
            prev_best_fitness: Previous best fitness
            population_data: Dict with fitness and optional activation data

        Returns:
            Tuple of (new_state, metrics)
        """
        key, subkey = jax.random.split(state['rng_key'])

        # Update usage stats
        state = self._update_usage_stats(state, population_data)

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
            # Trigger guided mutation
            new_mask, mutation_info = self._mutate_palette_guided(
                subkey, state['mask'], state
            )
            palette_changed = not jnp.allclose(state['mask'], new_mask)
            new_stagnation = 0

        new_state = {
            **state,
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
        }

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': mask_to_indices(new_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'mutation_triggered': mutation_info is not None,
            'elite_weights': state['elite_activation_weights'].tolist(),
        }

        if mutation_info:
            metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including usage statistics."""
        palette = self.get_active_palette(state)
        weights = state['elite_activation_weights']

        # Find top weighted activations
        top_indices = jnp.argsort(weights)[-3:][::-1].tolist()

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'stagnation_count': state['stagnation_count'],
            'top_weighted_activations': top_indices,
            'sin_weight': float(weights[4]),
        }
