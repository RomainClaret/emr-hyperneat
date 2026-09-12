"""Strategy 72D: Aggregation Homeostasis Dual (Activity Balance for Aggregations).

Biological Basis: Homeostatic plasticity maintains stable activity balance.

Key mechanism: Track aggregation "activity balance" between averaging (sum/mean)
and extreme-value (max/min/product/maxabs) categories. If one category dominates,
boost exploration in the underrepresented category.

Target: 50% averaging, 50% extreme to maintain diverse aggregation capabilities.

Expected: Better aggregation retention through balanced exploration.
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
)


class AggHomeostasisDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with homeostatic aggregation balance.

    Maintains balance between averaging and extreme-value aggregations
    through activity tracking and biased mutation.
    """

    name = "agg_homeostasis_dual"
    description = "Dual: Homeostatic balance between averaging and extreme aggregations"

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
        # Homeostasis parameters
        target_extreme_ratio: float = 0.5,  # Target: 50% extreme
        imbalance_threshold: float = 0.2,   # Trigger correction if off by 20%
        correction_strength: float = 1.5,    # Mutation rate multiplier
        # Cross-domain
        cross_learning_rate: float = 0.05,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Aggregation Homeostasis Dual strategy.

        Args:
            act_mutation_rate: Base activation mutation probability
            agg_mutation_rate: Base aggregation mutation probability
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            target_extreme_ratio: Target ratio of extreme aggregations
            imbalance_threshold: Imbalance threshold to trigger correction
            correction_strength: Mutation rate multiplier for underrepresented
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
        self.target_extreme_ratio = target_extreme_ratio
        self.imbalance_threshold = imbalance_threshold
        self.correction_strength = correction_strength
        self.cross_learning_rate = cross_learning_rate
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with homeostasis tracking."""
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
            # Homeostasis tracking
            'extreme_ratio_history': [],
            'corrections_applied': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 720000),
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

    def _compute_extreme_ratio(self, agg_palette: List[int]) -> float:
        """Compute ratio of extreme aggregations in palette."""
        if len(agg_palette) == 0:
            return 0.0
        extreme_count = sum(1 for a in agg_palette if a in EXTREME_AGGS)
        return extreme_count / len(agg_palette)

    def _compute_imbalance(self, agg_palette: List[int]) -> Tuple[float, str]:
        """Compute imbalance and direction.

        Returns:
            Tuple of (imbalance_magnitude, direction)
            direction: 'extreme' if too many extreme, 'averaging' if too many averaging
        """
        extreme_ratio = self._compute_extreme_ratio(agg_palette)
        imbalance = extreme_ratio - self.target_extreme_ratio

        if imbalance > 0:
            return imbalance, 'extreme'
        else:
            return -imbalance, 'averaging'

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

    def _mutate_agg_palette_homeostatic(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        imbalance: float,
        direction: str,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply homeostatic mutation to aggregation palette.

        Biases mutation toward underrepresented category.
        """
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()

        # Compute per-aggregation mutation rates
        rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate

        if imbalance > self.imbalance_threshold:
            # Apply correction - boost underrepresented category
            if direction == 'extreme':
                # Too many extreme, boost averaging activation
                for i in AVERAGING_AGGS:
                    if mask[i] < 0.5:  # Currently inactive
                        rates = rates.at[i].set(self.agg_mutation_rate * self.correction_strength)
                # Boost extreme deactivation
                for i in EXTREME_AGGS:
                    if mask[i] > 0.5:  # Currently active
                        rates = rates.at[i].set(self.agg_mutation_rate * self.correction_strength)
            else:
                # Too many averaging, boost extreme activation
                for i in EXTREME_AGGS:
                    if mask[i] < 0.5:  # Currently inactive
                        rates = rates.at[i].set(self.agg_mutation_rate * self.correction_strength)
                # Boost averaging deactivation
                for i in AVERAGING_AGGS:
                    if mask[i] > 0.5:  # Currently active
                        rates = rates.at[i].set(self.agg_mutation_rate * self.correction_strength)

        flip_probs = jax.random.uniform(key1, (NUM_AGGREGATIONS,))
        flip_mask = flip_probs < rates
        new_mask = jnp.where(flip_mask, 1.0 - mask, mask)

        flipped_indices = jnp.where(flip_mask)[0].tolist()
        activated = [i for i in flipped_indices if mask[i] < 0.5]
        deactivated = [i for i in flipped_indices if mask[i] > 0.5]

        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'correction_applied': imbalance > self.imbalance_threshold,
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
        """Update with homeostatic aggregation balancing."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

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

        # Track homeostasis
        agg_palette = mask_to_indices(state['agg_mask'])
        imbalance, direction = self._compute_imbalance(agg_palette)
        extreme_ratio = self._compute_extreme_ratio(agg_palette)

        extreme_ratio_history = state['extreme_ratio_history'] + [extreme_ratio]
        if len(extreme_ratio_history) > 20:
            extreme_ratio_history = extreme_ratio_history[-20:]

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None
        correction_applied = False
        new_corrections = state['corrections_applied']

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_act_palette_uniform(
                k_act, state['act_mask']
            )
            new_agg_mask, agg_mutation_info = self._mutate_agg_palette_homeostatic(
                k_agg, state['agg_mask'], imbalance, direction
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            correction_applied = agg_mutation_info.get('correction_applied', False)
            if correction_applied:
                new_corrections += 1
            new_stagnation = 0

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'cross_affinity': new_cross,
            'extreme_ratio_history': extreme_ratio_history,
            'corrections_applied': new_corrections,
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
            # Homeostasis metrics
            'extreme_ratio': extreme_ratio,
            'imbalance': imbalance,
            'imbalance_direction': direction,
            'correction_applied': correction_applied,
            'total_corrections': new_corrections,
            # Sin status
            'has_sin': 4 in act_palette,
            # Aggregation category status
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
        """Return state summary with homeostasis status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        extreme_ratio = self._compute_extreme_ratio(agg_palette)
        imbalance, direction = self._compute_imbalance(agg_palette)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'extreme_ratio': extreme_ratio,
            'imbalance': imbalance,
            'imbalance_direction': direction,
            'total_corrections': state['corrections_applied'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
