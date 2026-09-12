"""Strategy 36: Burst Refractory (Refractory Period Gating).

Implements refractory period dynamics for palette evolution. Functions that
are continuously active for multiple generations enter a refractory period
with reduced mutability, creating natural attention windows and cycles.

Biological Basis:
- Neurons enter absolute/relative refractory periods after action potentials
- During refractory: reduced excitability, harder to activate
- Burst firing followed by silence is common neural pattern
- Prevents runaway excitation and allows information processing phases

Key Insight:
- Current strategies don't model temporal activity patterns
- Burst firing followed by refractory creates natural attention windows
- Functions cycle through active → burst → refractory → recovery
- Prevents getting stuck in local optima through forced exploration

Refractory Mechanism:
    # Track consecutive active generations
    if function f is active:
        consecutive_active[f] += 1
    else:
        consecutive_active[f] = 0

    # Enter refractory after burst threshold
    if consecutive_active[f] >= burst_threshold:
        refractory_countdown[f] = refractory_duration
        consecutive_active[f] = 0

    # During refractory: reduced selection probability
    if refractory_countdown[f] > 0:
        selection_weight[f] *= refractory_factor
        refractory_countdown[f] -= 1

Expected improvements:
- Natural attention windows (burst → silence cycles)
- Prevention of runaway function dominance
- Forced periodic exploration during refractory
- Self-organizing temporal patterns
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    DEFAULT_PALETTE_INDICES,
)


class BurstRefractoryStrategy(PaletteEvolutionStrategy):
    """Refractory period gating after burst activity.

    Functions that are continuously active enter refractory periods.
    During refractory, selection probability is reduced.
    Creates natural cycles of activity and forced exploration.
    """

    name = "burst_refractory"
    description = "Burst firing triggers refractory periods with reduced selection"

    def __init__(
        self,
        # Burst detection
        burst_threshold: int = 4,               # Consecutive gens to trigger refractory
        consecutive_boost: float = 0.05,        # Small weight boost per consecutive use
        # Refractory period
        refractory_duration: int = 5,           # Generations in refractory
        refractory_factor: float = 0.2,         # Selection weight multiplier during refractory
        refractory_recovery_boost: float = 0.3, # Fitness boost when exiting refractory
        # Relative refractory (partial recovery)
        relative_refractory_duration: int = 2,  # Partial refractory after absolute
        relative_refractory_factor: float = 0.6, # Less severe reduction
        # Base selection
        base_weight: float = 1.0,               # Starting selection weight
        fitness_weight_learning: float = 0.1,   # Weight adaptation from fitness
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Burst Refractory strategy.

        Args:
            burst_threshold: Consecutive active generations to trigger refractory
            consecutive_boost: Weight boost per consecutive use (pre-burst)
            refractory_duration: Length of absolute refractory period
            refractory_factor: Weight multiplier during absolute refractory
            refractory_recovery_boost: Fitness weight boost when exiting refractory
            relative_refractory_duration: Length of relative refractory (after absolute)
            relative_refractory_factor: Weight multiplier during relative refractory
            base_weight: Starting selection weight
            fitness_weight_learning: Rate of weight adaptation from fitness
            palette_size: Target number of active functions
        """
        # Burst detection
        self.burst_threshold = burst_threshold
        self.consecutive_boost = consecutive_boost

        # Refractory periods
        self.refractory_duration = refractory_duration
        self.refractory_factor = refractory_factor
        self.refractory_recovery_boost = refractory_recovery_boost
        self.relative_refractory_duration = relative_refractory_duration
        self.relative_refractory_factor = relative_refractory_factor

        # Selection
        self.base_weight = base_weight
        self.fitness_weight_learning = fitness_weight_learning

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with refractory tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 363636),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Burst/refractory state
            'consecutive_active': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'absolute_refractory': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),  # Countdown
            'relative_refractory': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),  # Countdown
            # Selection weights
            'selection_weights': jnp.ones(NUM_ACTIVATIONS) * self.base_weight,
            # Tracking
            'total_bursts': jnp.zeros(NUM_ACTIVATIONS),
            'refractory_entries': jnp.zeros(NUM_ACTIVATIONS),
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_consecutive_and_refractory(
        self,
        consecutive: jnp.ndarray,
        absolute_ref: jnp.ndarray,
        relative_ref: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update burst tracking and refractory states."""
        new_consecutive = consecutive.copy()
        new_absolute = absolute_ref.copy()
        new_relative = relative_ref.copy()
        new_bursts = jnp.zeros(NUM_ACTIVATIONS)
        refractory_exits = jnp.zeros(NUM_ACTIVATIONS)

        for i in range(NUM_ACTIVATIONS):
            # Decrement refractory countdowns
            if absolute_ref[i] > 0:
                new_absolute = new_absolute.at[i].set(int(absolute_ref[i]) - 1)
                # Check for transition to relative refractory
                if absolute_ref[i] == 1 and self.relative_refractory_duration > 0:
                    new_relative = new_relative.at[i].set(self.relative_refractory_duration)
            elif relative_ref[i] > 0:
                new_relative = new_relative.at[i].set(int(relative_ref[i]) - 1)
                # Check for exit from all refractory
                if relative_ref[i] == 1:
                    refractory_exits = refractory_exits.at[i].set(1)

            # Update consecutive active counter
            if mask[i] > 0.5:
                # Active: increment counter if not in refractory
                if absolute_ref[i] == 0 and relative_ref[i] == 0:
                    new_consecutive = new_consecutive.at[i].set(int(consecutive[i]) + 1)

                    # Check for burst threshold
                    if new_consecutive[i] >= self.burst_threshold:
                        # Enter absolute refractory
                        new_absolute = new_absolute.at[i].set(self.refractory_duration)
                        new_consecutive = new_consecutive.at[i].set(0)
                        new_bursts = new_bursts.at[i].set(1)
            else:
                # Inactive: reset consecutive counter
                new_consecutive = new_consecutive.at[i].set(0)

        return new_consecutive, new_absolute, new_relative, new_bursts, refractory_exits

    def _compute_effective_weights(
        self,
        base_weights: jnp.ndarray,
        consecutive: jnp.ndarray,
        absolute_ref: jnp.ndarray,
        relative_ref: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective selection weights with refractory modulation."""
        effective = base_weights.copy()

        for i in range(NUM_ACTIVATIONS):
            current = float(base_weights[i])

            if absolute_ref[i] > 0:
                # Absolute refractory: severe reduction
                effective = effective.at[i].set(current * self.refractory_factor)
            elif relative_ref[i] > 0:
                # Relative refractory: moderate reduction
                effective = effective.at[i].set(current * self.relative_refractory_factor)
            else:
                # Normal: boost for consecutive use (pre-burst reward)
                consec = int(consecutive[i])
                if consec > 0:
                    boost = 1.0 + self.consecutive_boost * min(consec, self.burst_threshold - 1)
                    effective = effective.at[i].set(current * boost)

        return jnp.maximum(effective, 0.01)

    def _update_base_weights(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        refractory_exits: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update base weights from fitness and refractory exit."""
        new_weights = weights.copy()

        if improved:
            # Boost active functions that contributed to improvement
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    current = float(weights[i])
                    new_weights = new_weights.at[i].set(
                        current + self.fitness_weight_learning
                    )

        # Recovery boost for functions exiting refractory
        for i in range(NUM_ACTIVATIONS):
            if refractory_exits[i] > 0:
                current = float(new_weights[i])
                new_weights = new_weights.at[i].set(
                    current + self.refractory_recovery_boost
                )

        # Slight decay to neutral
        new_weights = new_weights * 0.98 + self.base_weight * 0.02

        return jnp.clip(new_weights, 0.1, 3.0)

    def _select_palette(
        self,
        effective_weights: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette based on effective weights."""
        key1, key2 = jax.random.split(key)

        # Top-k selection with some randomness
        probs = jax.nn.softmax(effective_weights)

        # Deterministic top selections
        top_k = self.palette_size - 1
        top_indices = jnp.argsort(effective_weights)[-top_k:]

        new_mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            new_mask = new_mask.at[int(idx)].set(1.0)

        # One probabilistic selection from remainder
        available_probs = probs * (1 - new_mask)
        available_probs = available_probs / (jnp.sum(available_probs) + 1e-8)
        sample = jax.random.choice(key2, NUM_ACTIVATIONS, p=available_probs)
        new_mask = new_mask.at[int(sample)].set(1.0)

        return new_mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with burst-refractory dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update consecutive activity and refractory states
        (new_consecutive, new_absolute, new_relative,
         burst_events, refractory_exits) = self._update_consecutive_and_refractory(
            state['consecutive_active'],
            state['absolute_refractory'],
            state['relative_refractory'],
            state['mask'],
        )

        # Step 2: Update base weights
        new_base_weights = self._update_base_weights(
            state['selection_weights'],
            state['mask'],
            improved,
            refractory_exits,
        )

        # Step 3: Compute effective weights
        effective_weights = self._compute_effective_weights(
            new_base_weights,
            new_consecutive,
            new_absolute,
            new_relative,
        )

        # Step 4: Select new palette
        new_mask = self._select_palette(effective_weights, k1)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update tracking
        new_total_bursts = state['total_bursts'] + burst_events
        new_refractory_entries = state['refractory_entries'] + burst_events  # Same event

        # Track fitness
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Burst/refractory state
            'consecutive_active': new_consecutive,
            'absolute_refractory': new_absolute,
            'relative_refractory': new_relative,
            # Selection weights
            'selection_weights': new_base_weights,
            # Tracking
            'total_bursts': new_total_bursts,
            'refractory_entries': new_refractory_entries,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Functions in refractory
        in_absolute = [i for i in range(NUM_ACTIVATIONS) if new_absolute[i] > 0]
        in_relative = [i for i in range(NUM_ACTIVATIONS) if new_relative[i] > 0]

        # Burst events this generation
        burst_list = [i for i in range(NUM_ACTIVATIONS) if burst_events[i] > 0]

        # Consecutive activity
        max_consecutive = int(jnp.max(new_consecutive))
        highest_consecutive = [(int(i), int(new_consecutive[i]))
                               for i in jnp.argsort(new_consecutive)[-3:][::-1]
                               if new_consecutive[i] > 0]

        # Top weights
        top_weights_idx = jnp.argsort(effective_weights)[-5:][::-1]
        top_weights = [(int(i), float(effective_weights[i])) for i in top_weights_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Burst stats
            'burst_events_this_gen': burst_list,
            'n_bursts_this_gen': len(burst_list),
            'total_bursts': int(jnp.sum(new_total_bursts)),
            # Refractory stats
            'in_absolute_refractory': in_absolute,
            'in_relative_refractory': in_relative,
            'n_in_refractory': len(in_absolute) + len(in_relative),
            # Consecutive activity
            'max_consecutive': max_consecutive,
            'highest_consecutive': highest_consecutive,
            # Weights
            'top_effective_weights': top_weights,
            'mean_effective_weight': float(jnp.mean(effective_weights)),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_consecutive': int(new_consecutive[4]),
            'sin_in_refractory': new_absolute[4] > 0 or new_relative[4] > 0,
            'sin_total_bursts': int(new_total_bursts[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with burst/refractory status."""
        palette = self.get_active_palette(state)
        consecutive = state['consecutive_active']
        absolute = state['absolute_refractory']
        relative = state['relative_refractory']

        # Compute effective weights for summary
        effective = self._compute_effective_weights(
            state['selection_weights'],
            consecutive,
            absolute,
            relative,
        )

        # Functions in different states
        in_absolute = [i for i in range(NUM_ACTIVATIONS) if absolute[i] > 0]
        in_relative = [i for i in range(NUM_ACTIVATIONS) if relative[i] > 0]
        building_burst = [i for i in range(NUM_ACTIVATIONS)
                         if consecutive[i] > 0 and consecutive[i] < self.burst_threshold]

        # Top by effective weight
        top_eff = jnp.argsort(effective)[-5:][::-1]
        top_effective = [(int(i), float(effective[i])) for i in top_eff]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Refractory state
            'in_absolute_refractory': in_absolute,
            'in_relative_refractory': in_relative,
            'building_burst': building_burst,
            # Weights
            'top_effective': top_effective,
            # History
            'total_bursts': int(jnp.sum(state['total_bursts'])),
            # Sin-specific
            'sin_consecutive': int(consecutive[4]),
            'sin_in_refractory': absolute[4] > 0 or relative[4] > 0,
            'sin_total_bursts': int(state['total_bursts'][4]),
        }
