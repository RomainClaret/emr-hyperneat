"""Strategy 74D: Cross-Domain Reinforcement Dual (Neuromodulatory Cross-Domain Learning).

Biological Basis: Neuromodulatory reinforcement of cross-modal associations.
Dopaminergic signals reinforce synapses that were active during reward.

Key mechanisms:
1. Double cross-domain learning rate when BOTH domains contribute to fitness
2. Use cross-affinity to guide aggregation selection
3. Sin-guided aggregation: When sin (idx 4) has high affinity with specific
   aggregations, protect those aggregations.

Expected: Better cross-domain synergy through enhanced affinity learning.
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


class CrossDomainReinforcementDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with neuromodulatory cross-domain reinforcement.

    Enhances cross-domain learning when both domains contribute to fitness,
    and uses affinity to guide mutation decisions.
    """

    name = "cross_domain_reinforcement_dual"
    description = "Dual: Neuromodulatory cross-domain affinity reinforcement"

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
        # Cross-domain parameters
        base_cross_learning_rate: float = 0.15,  # Higher base rate
        reinforcement_multiplier: float = 2.0,   # When both domains contribute
        affinity_protection_threshold: float = 0.6,  # Protect high-affinity pairs
        affinity_protection_strength: float = 0.5,   # Deactivation reduction
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Cross-Domain Reinforcement Dual strategy.

        Args:
            act_mutation_rate: Base activation mutation probability
            agg_mutation_rate: Base aggregation mutation probability
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            base_cross_learning_rate: Base cross-domain affinity learning rate
            reinforcement_multiplier: Multiplier when both domains change
            affinity_protection_threshold: Threshold to protect high-affinity pairs
            affinity_protection_strength: Deactivation reduction for protected
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
        self.base_cross_learning_rate = base_cross_learning_rate
        self.reinforcement_multiplier = reinforcement_multiplier
        self.affinity_protection_threshold = affinity_protection_threshold
        self.affinity_protection_strength = affinity_protection_strength
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with cross-domain tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'prev_act_mask': act_mask,  # Track previous for change detection
            # Aggregation domain
            'agg_mask': agg_mask,
            'prev_agg_mask': agg_mask,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Tracking
            'reinforcement_events': 0,
            'protection_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 740000),
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

    def _update_cross_affinity_reinforced(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        prev_act_mask: jnp.ndarray,
        prev_agg_mask: jnp.ndarray,
        fitness_improved: bool,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, bool]:
        """Update cross-domain affinity with reinforcement.

        Returns:
            Tuple of (new_cross_affinity, reinforcement_applied)
        """
        if not fitness_improved or fitness_delta <= 0:
            return cross_affinity, False

        # Detect domain changes
        act_changed = not jnp.allclose(act_mask, prev_act_mask)
        agg_changed = not jnp.allclose(agg_mask, prev_agg_mask)

        # Compute learning rate with potential reinforcement
        if act_changed and agg_changed:
            lr = self.base_cross_learning_rate * self.reinforcement_multiplier
            reinforced = True
        else:
            lr = self.base_cross_learning_rate
            reinforced = False

        # Compute co-activation
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        # Update affinity
        delta = lr * fitness_delta * co_active
        new_cross = cross_affinity + delta

        return jnp.clip(new_cross, 0.0, 1.0), reinforced

    def _get_sin_affinity_protection(
        self,
        cross_affinity: jnp.ndarray,
        act_palette: List[int],
    ) -> List[int]:
        """Get aggregations protected by sin affinity.

        If sin (idx 4) is active and has high affinity with certain aggregations,
        those aggregations get protection.
        """
        if 4 not in act_palette:
            return []

        sin_row = cross_affinity[4, :]
        protected = []
        for agg_idx in range(NUM_AGGREGATIONS):
            if sin_row[agg_idx] >= self.affinity_protection_threshold:
                protected.append(agg_idx)

        return protected

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

    def _mutate_agg_palette_affinity_guided(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_palette: List[int],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply affinity-guided mutation to aggregation palette.

        High-affinity aggregations (especially with sin) get protection.
        """
        new_mask = mask.copy()

        # Get protected aggregations
        protected = self._get_sin_affinity_protection(cross_affinity, act_palette)

        # Compute per-aggregation rates
        deactivation_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate
        activation_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate

        # Reduce deactivation for protected aggregations
        for p in protected:
            if mask[p] > 0.5:  # Currently active
                deactivation_rates = deactivation_rates.at[p].set(
                    self.agg_mutation_rate * (1.0 - self.affinity_protection_strength)
                )

        # Use average cross-affinity to boost activation of high-affinity aggregations
        avg_affinity = jnp.mean(cross_affinity, axis=0)  # Average over all activations
        for i in range(NUM_AGGREGATIONS):
            if mask[i] < 0.5 and avg_affinity[i] >= self.affinity_protection_threshold:
                activation_rates = activation_rates.at[i].set(
                    self.agg_mutation_rate * (1.0 + self.affinity_protection_strength)
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

        # Ensure constraints
        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'protected_count': len(protected),
            'protected': protected,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with cross-domain reinforcement."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update cross-domain affinity with reinforcement
        new_cross, reinforced = self._update_cross_affinity_reinforced(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            state['prev_act_mask'],
            state['prev_agg_mask'],
            improved,
            fitness_delta,
        )
        new_reinforcement_events = state['reinforcement_events'] + (1 if reinforced else 0)

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None
        new_protection_events = state['protection_events']

        if new_stagnation >= self.stagnation_threshold:
            act_palette = mask_to_indices(state['act_mask'])

            new_act_mask, act_mutation_info = self._mutate_act_palette_uniform(
                k_act, state['act_mask']
            )
            new_agg_mask, agg_mutation_info = self._mutate_agg_palette_affinity_guided(
                k_agg, state['agg_mask'], new_cross, act_palette
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            if agg_mutation_info and agg_mutation_info.get('protected_count', 0) > 0:
                new_protection_events += 1
            new_stagnation = 0

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'prev_act_mask': state['act_mask'],
            'prev_agg_mask': state['agg_mask'],
            'cross_affinity': new_cross,
            'reinforcement_events': new_reinforcement_events,
            'protection_events': new_protection_events,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Sin-specific affinity
        sin_affinity = float(new_cross[4].mean()) if len(act_palette) > 0 else 0.0
        sin_max_aff = float(new_cross[4, 2]) if 4 in act_palette else 0.0  # sin-max
        sin_min_aff = float(new_cross[4, 3]) if 4 in act_palette else 0.0  # sin-min

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
            # Reinforcement metrics
            'reinforced_this_gen': reinforced,
            'total_reinforcement_events': new_reinforcement_events,
            'total_protection_events': new_protection_events,
            # Sin-specific
            'sin_affinity': sin_affinity,
            'sin_max_affinity': sin_max_aff,
            'sin_min_affinity': sin_min_aff,
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']
            metrics['protected_this_gen'] = agg_mutation_info.get('protected', [])

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with cross-domain status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        protected = self._get_sin_affinity_protection(state['cross_affinity'], act_palette)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
            'sin_protected_aggs': protected if 4 in act_palette else [],
            'reinforcement_events': state['reinforcement_events'],
            'protection_events': state['protection_events'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
        }
