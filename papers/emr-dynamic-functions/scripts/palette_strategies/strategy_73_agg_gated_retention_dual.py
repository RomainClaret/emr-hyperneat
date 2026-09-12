"""Strategy 73D: Aggregation-Gated Retention Dual (Thalamic Gating for Aggregations).

Biological Basis: Thalamic gates control information flow and protect important pathways.

Key mechanism: Track per-aggregation fitness attribution (ablation-style).
Aggregations that consistently contribute to fitness improvement get "locked"
with very low deactivation probability. Prevents loss of functionally important
aggregations during continual learning.

Expected: Higher aggregation retention rate by protecting proven aggregations.
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
)


class AggGatedRetentionDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with thalamic gating for aggregation retention.

    Tracks fitness attribution per aggregation and locks high-attribution
    aggregations to prevent loss during evolution.
    """

    name = "agg_gated_retention_dual"
    description = "Dual: Thalamic gating protects high-attribution aggregations"

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
        # Gating parameters
        attribution_increment: float = 0.1,  # Attribution gain per fitness improvement
        attribution_decay: float = 0.02,     # Attribution decay per generation
        lock_threshold: float = 0.3,         # Attribution level to lock
        lock_protection: float = 0.95,       # Deactivation probability reduction when locked
        # Cross-domain
        cross_learning_rate: float = 0.05,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Aggregation-Gated Retention Dual strategy.

        Args:
            act_mutation_rate: Base activation mutation probability
            agg_mutation_rate: Base aggregation mutation probability
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            attribution_increment: Attribution gain per fitness improvement
            attribution_decay: Attribution decay per generation
            lock_threshold: Attribution level to lock aggregation
            lock_protection: Deactivation probability reduction when locked (0.95 = 95% protected)
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
        self.attribution_increment = attribution_increment
        self.attribution_decay = attribution_decay
        self.lock_threshold = lock_threshold
        self.lock_protection = lock_protection
        self.cross_learning_rate = cross_learning_rate
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with attribution tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Per-aggregation attribution tracking
        agg_attribution = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Activation domain
            'act_mask': act_mask,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_attribution': agg_attribution,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Tracking
            'locked_count': 0,
            'total_locks_applied': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 730000),
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

    def _update_attribution(
        self,
        agg_attribution: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_improved: bool,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update per-aggregation attribution based on fitness.

        Active aggregations during fitness improvement gain attribution.
        All attributions decay slightly over time.
        """
        new_attribution = agg_attribution.copy()

        # Decay all attributions
        new_attribution = new_attribution * (1.0 - self.attribution_decay)

        # Attribute fitness improvement to active aggregations
        if fitness_improved and fitness_delta > 0:
            active_agg = (agg_mask > 0.5).astype(jnp.float32)
            increment = self.attribution_increment * min(fitness_delta * 10, 1.0)
            new_attribution = new_attribution + increment * active_agg

        return jnp.clip(new_attribution, 0.0, 1.0)

    def _get_locked_aggregations(self, agg_attribution: jnp.ndarray) -> List[int]:
        """Return list of locked aggregation indices."""
        return [i for i in range(NUM_AGGREGATIONS) if agg_attribution[i] >= self.lock_threshold]

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

    def _mutate_agg_palette_gated(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        agg_attribution: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply gated mutation to aggregation palette.

        High-attribution aggregations have reduced deactivation probability.
        """
        new_mask = mask.copy()

        # Compute per-aggregation deactivation rates
        deactivation_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate
        activation_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate

        # Locked aggregations have very low deactivation rate
        locked = agg_attribution >= self.lock_threshold
        deactivation_rates = jnp.where(
            locked,
            deactivation_rates * (1.0 - self.lock_protection),
            deactivation_rates
        )

        # For each aggregation, decide flip based on current state
        flip_probs = jax.random.uniform(key, (NUM_AGGREGATIONS,))

        # Apply appropriate rate based on current state
        for i in range(NUM_AGGREGATIONS):
            if mask[i] > 0.5:  # Currently active - use deactivation rate
                if flip_probs[i] < deactivation_rates[i]:
                    new_mask = new_mask.at[i].set(0.0)
            else:  # Currently inactive - use activation rate
                if flip_probs[i] < activation_rates[i]:
                    new_mask = new_mask.at[i].set(1.0)

        # Track changes
        activated = []
        deactivated = []
        protected_count = 0
        for i in range(NUM_AGGREGATIONS):
            if mask[i] < 0.5 and new_mask[i] > 0.5:
                activated.append(i)
            elif mask[i] > 0.5 and new_mask[i] < 0.5:
                deactivated.append(i)
        protected_count = int(jnp.sum(locked))

        # Ensure constraints
        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'protected_count': protected_count,
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
        """Update with gated aggregation protection."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update attribution
        new_attribution = self._update_attribution(
            state['agg_attribution'],
            state['agg_mask'],
            improved,
            fitness_delta,
        )

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None
        new_total_locks = state['total_locks_applied']

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_act_palette_uniform(
                k_act, state['act_mask']
            )
            new_agg_mask, agg_mutation_info = self._mutate_agg_palette_gated(
                k_agg, state['agg_mask'], new_attribution
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            if agg_mutation_info and agg_mutation_info['protected_count'] > 0:
                new_total_locks += 1
            new_stagnation = 0

        locked_aggs = self._get_locked_aggregations(new_attribution)
        new_locked_count = len(locked_aggs)

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'agg_attribution': new_attribution,
            'cross_affinity': new_cross,
            'locked_count': new_locked_count,
            'total_locks_applied': new_total_locks,
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
            # Gating metrics
            'locked_aggs': locked_aggs,
            'locked_count': new_locked_count,
            'total_locks_applied': new_total_locks,
            'max_attribution': float(jnp.max(new_attribution)),
            'mean_attribution': float(jnp.mean(new_attribution)),
            # Per-aggregation attribution for extreme aggs
            'max_agg_attribution': float(new_attribution[2]),
            'min_agg_attribution': float(new_attribution[3]),
            # Sin status
            'has_sin': 4 in act_palette,
            # Aggregation status
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
        """Return state summary with gating status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        locked_aggs = self._get_locked_aggregations(state['agg_attribution'])

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'locked_aggs': locked_aggs,
            'locked_count': len(locked_aggs),
            'max_attribution': float(jnp.max(state['agg_attribution'])),
            'total_locks_applied': state['total_locks_applied'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
