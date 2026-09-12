"""Strategy 76D: Extreme-Value Aggregation Bias Dual (Winner-Take-All Circuits).

Biological Basis: Winner-take-all circuits use max-pooling operations.
Cortical lateral inhibition creates competition where strongest signal wins.

Key mechanism: Explicitly bias toward extreme aggregations (max/min/maxabs).
Bias increases as fitness stagnates (harder problem signal).

Hypothesis: Hard parity problems (P5, P6) require extreme-value aggregations.
Tests H2.

Expected: Better discovery and retention of max/min aggregations.
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
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class ExtremeValueBiasDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with explicit extreme aggregation bias.

    Biases mutation toward extreme-value aggregations, especially as
    fitness stagnates (indicates harder problem).
    """

    name = "extreme_value_bias_dual"
    description = "Dual: Biases toward extreme aggregations (max/min) for hard problems"

    def __init__(
        self,
        # Mutation rates
        act_mutation_rate: float = 0.1,
        agg_mutation_rate: float = 0.1,
        stagnation_threshold: int = 5,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Extreme bias parameters
        base_extreme_bias: float = 1.2,    # Base activation boost for extreme aggs
        stagnation_bias_rate: float = 0.05,  # Bias increase per stagnation gen
        max_extreme_bias: float = 2.0,     # Maximum activation boost
        deactivation_protection: float = 0.5,  # Deactivation reduction for extreme
        # Cross-domain
        cross_learning_rate: float = 0.05,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Extreme-Value Aggregation Bias Dual strategy.

        Args:
            act_mutation_rate: Base activation mutation probability
            agg_mutation_rate: Base aggregation mutation probability
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            base_extreme_bias: Base activation boost for extreme aggregations
            stagnation_bias_rate: Bias increase per stagnation generation
            max_extreme_bias: Maximum activation boost
            deactivation_protection: Deactivation reduction for active extreme aggs
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
        self.base_extreme_bias = base_extreme_bias
        self.stagnation_bias_rate = stagnation_bias_rate
        self.max_extreme_bias = max_extreme_bias
        self.deactivation_protection = deactivation_protection
        self.cross_learning_rate = cross_learning_rate
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with extreme bias tracking."""
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
            # Tracking
            'extreme_discoveries': 0,
            'extreme_retained': 0,
            'cumulative_stagnation': 0,  # Total stagnation for bias calculation
            # General state
            'rng_key': jax.random.PRNGKey(seed + 760000),
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

    def _compute_extreme_bias(self, stagnation_count: int) -> float:
        """Compute extreme bias based on stagnation (problem difficulty signal)."""
        bias = self.base_extreme_bias + stagnation_count * self.stagnation_bias_rate
        return min(bias, self.max_extreme_bias)

    def _mutate_act_palette_uniform(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply uniform mutation to activation palette."""
        flip_probs = jax.random.uniform(key, (NUM_ACTIVATIONS,))
        flip_mask = flip_probs < self.act_mutation_rate
        new_mask = jnp.where(flip_mask, 1.0 - mask, mask)

        flipped_indices = jnp.where(flip_mask)[0].tolist()
        activated = [i for i in flipped_indices if mask[i] < 0.5]
        deactivated = [i for i in flipped_indices if mask[i] > 0.5]

        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active_act or active_count > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _mutate_agg_palette_extreme_biased(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        extreme_bias: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply extreme-biased mutation to aggregation palette.

        Extreme aggregations get boosted activation rate and protected deactivation.
        """
        new_mask = mask.copy()

        # Compute per-aggregation rates with extreme bias
        activation_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate
        deactivation_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate

        # Boost activation for extreme aggregations (especially max/min)
        for agg_idx in CORE_EXTREME_AGGS:  # max (2), min (3)
            if mask[agg_idx] < 0.5:  # Currently inactive
                activation_rates = activation_rates.at[agg_idx].set(
                    self.agg_mutation_rate * extreme_bias
                )
            else:  # Currently active - protect from deactivation
                deactivation_rates = deactivation_rates.at[agg_idx].set(
                    self.agg_mutation_rate * (1.0 - self.deactivation_protection)
                )

        # Slightly boost other extreme aggs (product, maxabs)
        for agg_idx in [4, 5]:  # product, maxabs
            if mask[agg_idx] < 0.5:
                activation_rates = activation_rates.at[agg_idx].set(
                    self.agg_mutation_rate * (extreme_bias * 0.7)
                )

        # Apply mutations
        flip_probs = jax.random.uniform(key, (NUM_AGGREGATIONS,))
        for i in range(NUM_AGGREGATIONS):
            if mask[i] > 0.5:  # Currently active
                if flip_probs[i] < deactivation_rates[i]:
                    new_mask = new_mask.at[i].set(0.0)
            else:  # Currently inactive
                if flip_probs[i] < activation_rates[i]:
                    new_mask = new_mask.at[i].set(1.0)

        # Track changes
        activated = []
        deactivated = []
        for i in range(NUM_AGGREGATIONS):
            if mask[i] < 0.5 and new_mask[i] > 0.5:
                activated.append(i)
            elif mask[i] > 0.5 and new_mask[i] < 0.5:
                deactivated.append(i)

        # Count extreme activations/deactivations
        extreme_activated = [a for a in activated if a in EXTREME_AGGS]
        extreme_deactivated = [a for a in deactivated if a in EXTREME_AGGS]

        # Ensure constraints
        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []
            extreme_activated = []
            extreme_deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'extreme_activated': extreme_activated,
            'extreme_deactivated': extreme_deactivated,
        }

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
        """Update with extreme aggregation bias."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
            new_cumulative_stag = 0  # Reset on improvement
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']
            new_cumulative_stag = state['cumulative_stagnation'] + 1

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Compute extreme bias based on cumulative stagnation
        extreme_bias = self._compute_extreme_bias(new_cumulative_stag)

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None
        new_extreme_discoveries = state['extreme_discoveries']

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_act_palette_uniform(
                k_act, state['act_mask']
            )
            new_agg_mask, agg_mutation_info = self._mutate_agg_palette_extreme_biased(
                k_agg, state['agg_mask'], extreme_bias
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

            # Track extreme discoveries
            if agg_mutation_info:
                new_extreme_discoveries += len(agg_mutation_info.get('extreme_activated', []))

            new_stagnation = 0

        # Track extreme retention
        agg_palette = mask_to_indices(new_agg_mask)
        extreme_count = sum(1 for a in agg_palette if a in EXTREME_AGGS)
        core_extreme_count = sum(1 for a in agg_palette if a in CORE_EXTREME_AGGS)

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'cross_affinity': new_cross,
            'extreme_discoveries': new_extreme_discoveries,
            'extreme_retained': extreme_count,
            'cumulative_stagnation': new_cumulative_stag,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        act_palette = mask_to_indices(new_act_mask)

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
            # Extreme bias metrics
            'extreme_bias': extreme_bias,
            'cumulative_stagnation': new_cumulative_stag,
            'extreme_count': extreme_count,
            'core_extreme_count': core_extreme_count,
            'extreme_discoveries': new_extreme_discoveries,
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'has_product': 4 in agg_palette,
            'has_maxabs': 5 in agg_palette,
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']
            metrics['extreme_activated'] = agg_mutation_info.get('extreme_activated', [])
            metrics['extreme_deactivated'] = agg_mutation_info.get('extreme_deactivated', [])

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with extreme bias status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        extreme_count = sum(1 for a in agg_palette if a in EXTREME_AGGS)
        extreme_bias = self._compute_extreme_bias(state['cumulative_stagnation'])

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'extreme_count': extreme_count,
            'extreme_bias': extreme_bias,
            'extreme_discoveries': state['extreme_discoveries'],
            'cumulative_stagnation': state['cumulative_stagnation'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
