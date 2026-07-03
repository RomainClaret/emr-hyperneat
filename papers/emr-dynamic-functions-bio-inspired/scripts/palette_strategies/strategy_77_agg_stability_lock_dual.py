"""Strategy 77D: Aggregation Stability Lock Dual (Metaplasticity-Based Retention).

Biological Basis: Metaplasticity - stable synapses become harder to change.
Synapses that have been stable for long periods require stronger signals
to modify, providing natural protection for learned patterns.

Key mechanism: Track consecutive generations each aggregation is active.
Aggregations active 5+ consecutive generations become progressively harder
to remove. "If it works, don't change it" for aggregation domain.

Expected: Better aggregation retention through usage-based locking.
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


class AggStabilityLockDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with metaplasticity-based aggregation locking.

    Aggregations that remain active for consecutive generations become
    progressively harder to deactivate.
    """

    name = "agg_stability_lock_dual"
    description = "Dual: Metaplasticity locks long-active aggregations"

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
        # Stability lock parameters
        stability_threshold: int = 5,      # Generations to consider "stable"
        max_lock_strength: float = 0.8,    # Maximum deactivation reduction
        lock_growth_rate: float = 0.15,    # How fast lock strength grows
        # Cross-domain
        cross_learning_rate: float = 0.05,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Aggregation Stability Lock Dual strategy.

        Args:
            act_mutation_rate: Base activation mutation probability
            agg_mutation_rate: Base aggregation mutation probability
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            stability_threshold: Consecutive gens to trigger locking
            max_lock_strength: Maximum deactivation probability reduction
            lock_growth_rate: Rate at which lock strength increases
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
        self.stability_threshold = stability_threshold
        self.max_lock_strength = max_lock_strength
        self.lock_growth_rate = lock_growth_rate
        self.cross_learning_rate = cross_learning_rate
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with consecutive activity tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Track consecutive active generations for each aggregation
        agg_consecutive_active = jnp.zeros(NUM_AGGREGATIONS)
        # Initialize for starting palette
        for idx in initial_agg:
            agg_consecutive_active = agg_consecutive_active.at[idx].set(1.0)

        return {
            # Activation domain
            'act_mask': act_mask,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_consecutive_active': agg_consecutive_active,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Tracking
            'lock_events': 0,  # Times lock prevented deactivation
            'max_stability_seen': 0,  # Maximum consecutive active gens
            # General state
            'rng_key': jax.random.PRNGKey(seed + 770000),
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

    def _update_consecutive_active(
        self,
        agg_consecutive_active: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update consecutive active generations for each aggregation."""
        new_consecutive = agg_consecutive_active.copy()

        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                # Still active - increment counter
                new_consecutive = new_consecutive.at[i].set(agg_consecutive_active[i] + 1)
            else:
                # Deactivated - reset counter
                new_consecutive = new_consecutive.at[i].set(0.0)

        return new_consecutive

    def _compute_lock_strength(self, consecutive_active: float) -> float:
        """Compute lock strength based on consecutive active generations.

        Returns 0 if below threshold, then grows with generations above threshold.
        """
        if consecutive_active < self.stability_threshold:
            return 0.0

        # Sigmoid-like growth after threshold
        excess = consecutive_active - self.stability_threshold
        strength = 1.0 - (1.0 / (1.0 + excess * self.lock_growth_rate))
        return min(strength * self.max_lock_strength, self.max_lock_strength)

    def _get_locked_aggregations(
        self,
        agg_consecutive_active: jnp.ndarray,
    ) -> List[Tuple[int, float]]:
        """Return list of (aggregation_idx, lock_strength) for locked aggregations."""
        locked = []
        for i in range(NUM_AGGREGATIONS):
            strength = self._compute_lock_strength(float(agg_consecutive_active[i]))
            if strength > 0:
                locked.append((i, strength))
        return locked

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

    def _mutate_agg_palette_stability_locked(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        agg_consecutive_active: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply stability-locked mutation to aggregation palette.

        Long-active aggregations have reduced deactivation probability.
        """
        new_mask = mask.copy()

        # Compute per-aggregation deactivation rates with stability lock
        deactivation_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate
        activation_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_mutation_rate

        locked_info = []
        for i in range(NUM_AGGREGATIONS):
            lock_strength = self._compute_lock_strength(float(agg_consecutive_active[i]))
            if lock_strength > 0 and mask[i] > 0.5:
                # Reduce deactivation rate for locked aggregations
                deactivation_rates = deactivation_rates.at[i].set(
                    self.agg_mutation_rate * (1.0 - lock_strength)
                )
                locked_info.append((i, lock_strength))

        # Apply mutations
        flip_probs = jax.random.uniform(key, (NUM_AGGREGATIONS,))
        locks_applied = 0

        for i in range(NUM_AGGREGATIONS):
            if mask[i] > 0.5:  # Currently active
                would_deactivate = flip_probs[i] < self.agg_mutation_rate
                actually_deactivated = flip_probs[i] < deactivation_rates[i]
                if would_deactivate and not actually_deactivated:
                    locks_applied += 1  # Lock prevented deactivation
                if actually_deactivated:
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
            locks_applied = 0

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'locked_info': locked_info,
            'locks_applied': locks_applied,
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
        """Update with stability-based aggregation locking."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update consecutive active tracking
        new_consecutive = self._update_consecutive_active(
            state['agg_consecutive_active'],
            state['agg_mask'],
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
        new_lock_events = state['lock_events']

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_act_palette_uniform(
                k_act, state['act_mask']
            )
            new_agg_mask, agg_mutation_info = self._mutate_agg_palette_stability_locked(
                k_agg, state['agg_mask'], new_consecutive
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

            if agg_mutation_info:
                new_lock_events += agg_mutation_info.get('locks_applied', 0)

            # Update consecutive after mutation (reset deactivated ones)
            new_consecutive = self._update_consecutive_active(new_consecutive, new_agg_mask)

            new_stagnation = 0

        # Track max stability
        new_max_stability = max(state['max_stability_seen'], int(jnp.max(new_consecutive)))

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'agg_consecutive_active': new_consecutive,
            'cross_affinity': new_cross,
            'lock_events': new_lock_events,
            'max_stability_seen': new_max_stability,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Get locked aggregations
        locked_aggs = self._get_locked_aggregations(new_consecutive)

        # Track extreme aggregation stability
        max_stability = float(new_consecutive[2]) if len(agg_palette) > 0 else 0.0
        min_stability = float(new_consecutive[3]) if len(agg_palette) > 0 else 0.0

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
            # Stability lock metrics
            'locked_aggs': [(i, s) for i, s in locked_aggs],
            'locked_count': len(locked_aggs),
            'lock_events_total': new_lock_events,
            'max_stability_seen': new_max_stability,
            'max_agg_stability': max_stability,
            'min_agg_stability': min_stability,
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
            metrics['locks_applied_this_gen'] = agg_mutation_info.get('locks_applied', 0)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with stability lock status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        locked_aggs = self._get_locked_aggregations(state['agg_consecutive_active'])

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'locked_aggs': [(i, s) for i, s in locked_aggs],
            'locked_count': len(locked_aggs),
            'lock_events_total': state['lock_events'],
            'max_stability_seen': state['max_stability_seen'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
