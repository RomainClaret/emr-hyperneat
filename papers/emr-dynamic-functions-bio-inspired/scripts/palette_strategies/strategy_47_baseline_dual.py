"""Strategy 47D: Baseline Dual (Uniform Mutation for Both Domains).

Extends BaselineStrategy to jointly evolve BOTH activation AND aggregation
function palettes using uniform symmetric mutation.

Key dual mechanisms:
1. Dual masks - separate masks for activation and aggregation
2. Dual stagnation tracking - triggers mutation in both domains
3. Cross-domain affinity - track which act-agg pairs work together
4. Coordinated mutation - mutations can trigger in both domains

Expected: Baseline performance for dual palette evolution comparison
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
)


class BaselineDualStrategy(PaletteEvolutionStrategy):
    """Baseline dual palette evolution - uniform symmetric mutation.

    Applies the same stagnation-triggered uniform mutation to both
    activation and aggregation palettes independently.
    """

    name = "baseline_dual"
    description = "Dual: 10% uniform, stagnation-triggered (5 gens) for both domains"

    def __init__(
        self,
        act_mutation_rate: float = 0.1,
        agg_mutation_rate: float = 0.1,
        stagnation_threshold: int = 5,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        cross_learning_rate: float = 0.05,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            act_mutation_rate: Probability of flipping each activation (default 10%)
            agg_mutation_rate: Probability of flipping each aggregation (default 10%)
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum number of active activation functions
            min_active_agg: Minimum number of active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            cross_learning_rate: Rate of cross-domain affinity learning
            initial_act_palette: Starting activation palette indices
            initial_agg_palette: Starting aggregation palette indices
        """
        self.act_mutation_rate = act_mutation_rate
        self.agg_mutation_rate = agg_mutation_rate
        self.stagnation_threshold = stagnation_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.cross_learning_rate = cross_learning_rate
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual masks and cross-domain tracking.

        Args:
            config: Configuration dict
            seed: Random seed

        Returns:
            Initial state
        """
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            # Aggregation domain
            'agg_mask': agg_mask,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 470000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _mutate_palette_uniform(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        mutation_rate: float,
        min_active: int,
        max_active: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply uniform symmetric mutation to a palette.

        Args:
            key: JAX random key
            mask: Current palette mask
            mutation_rate: Probability of flipping each function
            min_active: Minimum active functions
            max_active: Maximum active functions
            n_funcs: Number of functions in this domain

        Returns:
            Tuple of (new_mask, mutation_info)
        """
        flip_probs = jax.random.uniform(key, (n_funcs,))
        flip_mask = flip_probs < mutation_rate

        # XOR flip: active becomes inactive, inactive becomes active
        new_mask = jnp.where(flip_mask, 1.0 - mask, mask)

        # Track changes
        flipped_indices = jnp.where(flip_mask)[0].tolist()
        activated = [i for i in flipped_indices if mask[i] < 0.5]
        deactivated = [i for i in flipped_indices if mask[i] > 0.5]

        # Ensure constraints
        active_count = jnp.sum(new_mask > 0.5)
        if active_count < min_active or active_count > max_active:
            new_mask = mask  # Revert
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
            'flipped_count': len(flipped_indices),
        }

        return new_mask, mutation_info

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity based on co-activation success."""
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
        """Update with stagnation-triggered mutation for both domains.

        Args:
            state: Current state
            generation: Current generation
            best_fitness: Best fitness this gen
            prev_best_fitness: Previous best fitness
            population_data: Population data (not used)

        Returns:
            Tuple of (new_state, metrics)
        """
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        # Check if fitness improved
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Check if we should mutate
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None

        if new_stagnation >= self.stagnation_threshold:
            # Trigger mutation in both domains
            new_act_mask, act_mutation_info = self._mutate_palette_uniform(
                k_act, state['act_mask'],
                self.act_mutation_rate,
                self.min_active_act, self.max_active_act,
                NUM_ACTIVATIONS,
            )
            new_agg_mask, agg_mutation_info = self._mutate_palette_uniform(
                k_agg, state['agg_mask'],
                self.agg_mutation_rate,
                self.min_active_agg, self.max_active_agg,
                NUM_AGGREGATIONS,
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            new_stagnation = 0  # Reset counter after mutation

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'cross_affinity': new_cross,
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
            # Cross-domain metrics
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_affinity': float(new_cross[4].mean()) if len(act_palette) > 0 else 0.0,
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including dual palette status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
