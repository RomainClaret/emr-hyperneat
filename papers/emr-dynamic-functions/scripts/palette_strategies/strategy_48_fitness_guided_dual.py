"""Strategy 48D: Fitness-Guided Dual (Elite Usage for Both Domains).

Extends FitnessGuidedStrategy to jointly evolve BOTH activation AND aggregation
function palettes using elite population usage tracking.

Key dual mechanisms:
1. Dual usage tracking - EMA weights for both act and agg domains
2. Dual biased mutations - favor high-usage functions in both domains
3. Cross-domain affinity - track which act-agg pairs appear in elite networks
4. Coordinated exploration - elite success guides both domains

Expected: Improved discovery rates through fitness guidance in both domains
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


class FitnessGuidedDualStrategy(PaletteEvolutionStrategy):
    """Fitness-guided dual palette evolution.

    Tracks which activations and aggregations appear in elite networks
    and biases mutations to favor those functions in both domains.
    """

    name = "fitness_guided_dual"
    description = "Dual: Track elite usage, bias mutations in both domains"

    def __init__(
        self,
        elite_percentile: float = 0.1,
        ema_alpha: float = 0.1,
        base_mutation_rate: float = 0.15,
        stagnation_threshold: int = 5,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        cross_learning_rate: float = 0.08,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize strategy."""
        self.elite_percentile = elite_percentile
        self.ema_alpha = ema_alpha
        self.base_mutation_rate = base_mutation_rate
        self.stagnation_threshold = stagnation_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.cross_learning_rate = cross_learning_rate
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual usage tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_usage_counts': jnp.zeros(NUM_ACTIVATIONS),
            'act_elite_weights': jnp.ones(NUM_ACTIVATIONS) / NUM_ACTIVATIONS,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_usage_counts': jnp.zeros(NUM_AGGREGATIONS),
            'agg_elite_weights': jnp.ones(NUM_AGGREGATIONS) / NUM_AGGREGATIONS,
            # Cross-domain
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 480000),
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

    def _update_usage_stats(
        self,
        state: Dict[str, Any],
        population_data: Optional[Dict],
        best_fitness: float,
    ) -> Dict[str, Any]:
        """Update usage statistics from population for both domains."""
        if population_data is None:
            return state

        fitnesses = population_data.get('fitnesses')
        if fitnesses is None:
            return state

        new_act_weights = state['act_elite_weights']
        new_agg_weights = state['agg_elite_weights']

        # If fitness improved, boost current palette weights
        if best_fitness > state['best_fitness_seen']:
            # Boost activation weights for current palette
            act_boost = (state['act_mask'] > 0.5).astype(jnp.float32) * 0.15
            new_act_weights = state['act_elite_weights'] + act_boost
            new_act_weights = new_act_weights / jnp.sum(new_act_weights)

            # Boost aggregation weights for current palette
            agg_boost = (state['agg_mask'] > 0.5).astype(jnp.float32) * 0.15
            new_agg_weights = state['agg_elite_weights'] + agg_boost
            new_agg_weights = new_agg_weights / jnp.sum(new_agg_weights)

        return {
            **state,
            'act_elite_weights': new_act_weights,
            'agg_elite_weights': new_agg_weights,
        }

    def _get_mutation_bias(self, weights: jnp.ndarray, temperature: float = 0.5) -> jnp.ndarray:
        """Get mutation bias from elite weights."""
        return jax.nn.softmax(weights / temperature)

    def _mutate_palette_guided(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        elite_weights: jnp.ndarray,
        min_active: int,
        max_active: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply fitness-guided mutation."""
        key1, key2 = jax.random.split(key)

        # Get biased activation probabilities
        activation_bias = self._get_mutation_bias(elite_weights)

        # For inactive functions: use bias to decide activation
        inactive_mask = mask < 0.5
        activate_probs = jax.random.uniform(key1, (n_funcs,))

        # Bias: higher weight functions more likely to activate
        biased_rates = self.base_mutation_rate * activation_bias * n_funcs

        new_mask = mask.copy()
        activated = []
        deactivated = []

        current_active = int(jnp.sum(mask > 0.5))

        # Activate based on biased rates
        for i in range(n_funcs):
            if inactive_mask[i] and activate_probs[i] < biased_rates[i]:
                if current_active + len(activated) < max_active:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)

        # Deactivate with lower rate (conservative)
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))
        deactivate_rate = self.base_mutation_rate * 0.3

        active_mask = mask > 0.5
        for i in range(n_funcs):
            if active_mask[i] and deactivate_probs[i] < deactivate_rate:
                if activation_bias[i] > 0.1:
                    continue
                if jnp.sum(new_mask > 0.5) - len(deactivated) > min_active:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active constraint
        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
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
        """Update with fitness-guided mutation for both domains."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        # Update usage stats
        state = self._update_usage_stats(state, population_data, best_fitness)

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
            # Trigger guided mutation in both domains
            new_act_mask, act_mutation_info = self._mutate_palette_guided(
                k_act, state['act_mask'], state['act_elite_weights'],
                self.min_active_act, self.max_active_act, NUM_ACTIVATIONS,
            )
            new_agg_mask, agg_mutation_info = self._mutate_palette_guided(
                k_agg, state['agg_mask'], state['agg_elite_weights'],
                self.min_active_agg, self.max_active_agg, NUM_AGGREGATIONS,
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            new_stagnation = 0

        new_state = {
            'act_mask': new_act_mask,
            'act_usage_counts': state['act_usage_counts'],
            'act_elite_weights': state['act_elite_weights'],
            'agg_mask': new_agg_mask,
            'agg_usage_counts': state['agg_usage_counts'],
            'agg_elite_weights': state['agg_elite_weights'],
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
            # Weights
            'act_elite_weights': state['act_elite_weights'].tolist(),
            'agg_elite_weights': state['agg_elite_weights'].tolist(),
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_weight': float(state['act_elite_weights'][4]),
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including usage statistics."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        act_weights = state['act_elite_weights']
        agg_weights = state['agg_elite_weights']

        top_act = jnp.argsort(act_weights)[-3:][::-1].tolist()
        top_agg = jnp.argsort(agg_weights)[-2:][::-1].tolist()

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'stagnation_count': state['stagnation_count'],
            'top_weighted_activations': top_act,
            'top_weighted_aggregations': top_agg,
            'sin_weight': float(act_weights[4]),
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
