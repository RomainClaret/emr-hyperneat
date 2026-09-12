"""Strategy 42: Synaptic Homeostasis Hypothesis (Sleep-Inspired Consolidation).

Implements the Synaptic Homeostasis Hypothesis (SHY) for palette evolution.
Alternates between wake phases (learning and strengthening) and sleep phases
(global downscaling). Only the strongest functions survive scaling.

Biological Basis:
- During wake, synapses strengthen through learning
- During sleep, global synaptic downscaling occurs
- Only strongly potentiated synapses survive scaling
- This "sharpen and prune" cycle consolidates important learning
- Proposed by Giulio Tononi and Chiara Cirelli

Key Insight:
- Current strategies lack natural consolidation phases
- Wake/sleep cycles create distinct learn vs consolidate modes
- Global downscaling automatically prunes weak functions
- Strongest functions emerge without explicit ranking
- Prevents runaway strengthening (homeostatic regulation)

Sleep Mechanism:
    # Wake phase (normal operation)
    for gen in range(wake_cycle_length):
        weights += learning_rate * fitness_improvement
        # Normal exploration and strengthening

    # Sleep phase (consolidation)
    for gen in range(sleep_duration):
        weights *= sleep_scaling_factor  # Global downscaling
        for func in palette:
            if weights[func] < renormalization_threshold:
                remove_from_palette(func)  # Pruned during sleep

    # Cycle repeats

Expected improvements:
- Natural consolidation without explicit memory
- Global downscaling prevents bloat
- Sleep phases allow re-evaluation of palette
- Stronger functions preserved automatically
- Rhythmic exploration-consolidation pattern
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


class SynapticHomeostasisHypothesisStrategy(PaletteEvolutionStrategy):
    """Sleep-inspired consolidation with wake/sleep cycles.

    Alternates between wake phases (learning, strengthening) and sleep phases
    (global downscaling, pruning). Implements the Synaptic Homeostasis
    Hypothesis for natural consolidation of important functions.
    """

    name = "synaptic_homeostasis_hypothesis"
    description = "Wake/sleep cycles with global downscaling for consolidation"

    def __init__(
        self,
        # Wake/sleep cycle parameters
        wake_cycle_length: int = 15,               # Generations before sleep phase
        sleep_duration: int = 3,                    # Generations in sleep phase
        sleep_scaling_factor: float = 0.7,          # Global weight reduction during sleep
        renormalization_threshold: float = 0.3,     # Minimum weight to survive sleep
        # Wake phase learning
        wake_learning_rate: float = 0.15,           # Learning rate during wake
        wake_exploration_rate: float = 0.1,         # Exploration during wake
        # Sleep phase behavior
        sleep_exploration_rate: float = 0.02,       # Minimal exploration during sleep
        sleep_recovery_boost: float = 0.1,          # Small recovery for pruned candidates
        # Weight dynamics
        initial_weight: float = 1.0,
        max_weight: float = 3.0,
        min_weight: float = 0.05,
        weight_decay: float = 0.99,                 # Slow decay during wake
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Synaptic Homeostasis Hypothesis strategy.

        Args:
            wake_cycle_length: Generations before entering sleep
            sleep_duration: Generations in sleep phase
            sleep_scaling_factor: Weight multiplier during sleep (0-1)
            renormalization_threshold: Minimum weight to survive sleep
            wake_learning_rate: Learning rate during wake phase
            wake_exploration_rate: Exploration probability during wake
            sleep_exploration_rate: Minimal exploration during sleep
            sleep_recovery_boost: Recovery for functions near threshold
            initial_weight: Starting weight for all functions
            max_weight: Maximum weight cap
            min_weight: Minimum weight floor
            weight_decay: Slow decay during wake
            palette_size: Target palette size
        """
        # Cycle
        self.wake_cycle_length = wake_cycle_length
        self.sleep_duration = sleep_duration
        self.sleep_scaling_factor = sleep_scaling_factor
        self.renormalization_threshold = renormalization_threshold

        # Wake
        self.wake_learning_rate = wake_learning_rate
        self.wake_exploration_rate = wake_exploration_rate

        # Sleep
        self.sleep_exploration_rate = sleep_exploration_rate
        self.sleep_recovery_boost = sleep_recovery_boost

        # Weight dynamics
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.weight_decay = weight_decay

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with wake/sleep tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize weights
        weights = jnp.ones(NUM_ACTIVATIONS) * self.initial_weight
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                weights = weights.at[i].set(self.initial_weight * 1.2)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 424242),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Wake/sleep state
            'weights': weights,
            'is_sleeping': False,
            'cycle_position': 0,                    # Position in current cycle
            'sleep_gens_remaining': 0,              # Gens left in sleep phase
            # History
            'sleep_events': [],                     # List of (gen, pruned_funcs)
            'wake_events': [],                      # List of (gen, added_funcs)
            'total_sleep_cycles': 0,
            'previous_mask': mask,
            'fitness_history': [],
            # Stats
            'functions_pruned_in_sleep': 0,
            'functions_added_in_wake': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _enter_sleep(
        self,
        weights: jnp.ndarray,
        generation: int,
    ) -> Tuple[jnp.ndarray, int]:
        """Begin sleep phase with global downscaling."""
        # Apply global downscaling
        new_weights = weights * self.sleep_scaling_factor

        return new_weights, self.sleep_duration

    def _process_sleep_gen(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[int]]:
        """Process one generation of sleep."""
        new_weights = weights.copy()
        new_mask = mask.copy()
        pruned = []

        # Continue downscaling slightly
        new_weights = new_weights * (1 - 0.05)

        # Prune functions below threshold
        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5 and new_weights[i] < self.renormalization_threshold:
                new_mask = new_mask.at[i].set(0.0)
                pruned.append(i)

        # Ensure minimum palette
        active = mask_to_indices(new_mask)
        if len(active) < self.min_active:
            # Keep top functions by weight even if below threshold
            top_idx = jnp.argsort(new_weights)[-self.min_active:]
            for idx in top_idx:
                new_mask = new_mask.at[int(idx)].set(1.0)
                if int(idx) in pruned:
                    pruned.remove(int(idx))

        return new_weights, new_mask, pruned

    def _process_wake_gen(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        improvement: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[int]]:
        """Process one generation of wake."""
        key1, key2 = jax.random.split(key)
        new_weights = weights * self.weight_decay
        new_mask = mask.copy()
        added = []

        # Strengthen active functions if improving
        if improved:
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    current = float(new_weights[i])
                    delta = self.wake_learning_rate * max(improvement, 0.1)
                    new_weights = new_weights.at[i].set(
                        min(current + delta, self.max_weight)
                    )

        # Maybe explore new function
        if jax.random.uniform(key1) < self.wake_exploration_rate:
            # Find function not in palette with reasonable weight
            inactive = [i for i in range(NUM_ACTIVATIONS)
                       if mask[i] < 0.5 and weights[i] > self.min_weight]
            if inactive:
                idx = int(jax.random.randint(key2, (), 0, len(inactive)))
                new_func = inactive[idx]
                new_mask = new_mask.at[new_func].set(1.0)
                new_weights = new_weights.at[new_func].set(
                    float(new_weights[new_func]) + 0.3
                )
                added.append(new_func)

        new_weights = jnp.clip(new_weights, self.min_weight, self.max_weight)

        return new_weights, new_mask, added

    def _update_mask_from_weights(
        self,
        weights: jnp.ndarray,
        current_mask: jnp.ndarray,
        is_sleeping: bool,
    ) -> jnp.ndarray:
        """Update mask based on weights (for sleep phase mostly)."""
        if is_sleeping:
            # During sleep, mask follows threshold
            new_mask = jnp.where(weights >= self.renormalization_threshold, 1.0, 0.0)

            # Ensure minimum
            active = mask_to_indices(new_mask)
            if len(active) < self.min_active:
                top_idx = jnp.argsort(weights)[-self.min_active:]
                for idx in top_idx:
                    new_mask = new_mask.at[int(idx)].set(1.0)

            return new_mask
        else:
            # During wake, use top-k by weight
            top_idx = jnp.argsort(weights)[-self.palette_size:]
            new_mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_idx:
                new_mask = new_mask.at[int(idx)].set(1.0)
            return new_mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with wake/sleep dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        is_sleeping = state['is_sleeping']
        cycle_position = state['cycle_position']
        sleep_gens_remaining = state['sleep_gens_remaining']
        weights = state['weights']
        mask = state['mask']

        sleep_events = list(state['sleep_events'])
        wake_events = list(state['wake_events'])
        total_cycles = state['total_sleep_cycles']
        pruned_count = state['functions_pruned_in_sleep']
        added_count = state['functions_added_in_wake']

        pruned_this_gen = []
        added_this_gen = []

        if is_sleeping:
            # Process sleep generation
            weights, mask, pruned_this_gen = self._process_sleep_gen(weights, mask)
            sleep_gens_remaining -= 1
            pruned_count += len(pruned_this_gen)

            if pruned_this_gen:
                sleep_events.append((generation, pruned_this_gen))

            if sleep_gens_remaining <= 0:
                # Wake up
                is_sleeping = False
                cycle_position = 0
                total_cycles += 1
        else:
            # Process wake generation
            weights, mask, added_this_gen = self._process_wake_gen(
                weights, mask, improved, improvement, k1
            )
            cycle_position += 1
            added_count += len(added_this_gen)

            if added_this_gen:
                wake_events.append((generation, added_this_gen))

            if cycle_position >= self.wake_cycle_length:
                # Time to sleep
                is_sleeping = True
                weights, sleep_gens_remaining = self._enter_sleep(weights, generation)

        # Update mask based on weights
        new_mask = self._update_mask_from_weights(weights, mask, is_sleeping)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Trim history
        if len(sleep_events) > 20:
            sleep_events = sleep_events[-20:]
        if len(wake_events) > 20:
            wake_events = wake_events[-20:]

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
            # Wake/sleep state
            'weights': weights,
            'is_sleeping': is_sleeping,
            'cycle_position': cycle_position,
            'sleep_gens_remaining': sleep_gens_remaining,
            # History
            'sleep_events': sleep_events,
            'wake_events': wake_events,
            'total_sleep_cycles': total_cycles,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Stats
            'functions_pruned_in_sleep': pruned_count,
            'functions_added_in_wake': added_count,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Weight stats
        active_weights = [float(weights[i]) for i in active_palette]
        top_weight_idx = jnp.argsort(weights)[-5:][::-1]
        top_weights = [(int(i), float(weights[i])) for i in top_weight_idx]

        # Functions near threshold
        near_threshold = [i for i in range(NUM_ACTIVATIONS)
                         if abs(float(weights[i]) - self.renormalization_threshold) < 0.1]

        # Sleep status
        phase = "SLEEP" if is_sleeping else "WAKE"
        if is_sleeping:
            phase_progress = f"{self.sleep_duration - sleep_gens_remaining}/{self.sleep_duration}"
        else:
            phase_progress = f"{cycle_position}/{self.wake_cycle_length}"

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Phase info
            'phase': phase,
            'phase_progress': phase_progress,
            'is_sleeping': is_sleeping,
            'cycle_position': cycle_position,
            'total_sleep_cycles': total_cycles,
            # Weight stats
            'mean_weight': float(jnp.mean(weights)),
            'max_weight': float(jnp.max(weights)),
            'active_mean_weight': float(np.mean(active_weights)) if active_weights else 0.0,
            'top_weights': top_weights,
            'near_threshold': near_threshold,
            # Actions
            'pruned_this_gen': pruned_this_gen,
            'added_this_gen': added_this_gen,
            # Cumulative stats
            'total_pruned': pruned_count,
            'total_added': added_count,
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_weight': float(weights[4]),
            'sin_above_threshold': float(weights[4]) >= self.renormalization_threshold,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with wake/sleep status."""
        palette = self.get_active_palette(state)
        weights = state['weights']
        is_sleeping = state['is_sleeping']

        # Top by weight
        top_idx = jnp.argsort(weights)[-5:][::-1]
        top_weights = [(int(i), float(weights[i])) for i in top_idx]

        # Phase info
        phase = "SLEEP" if is_sleeping else "WAKE"
        if is_sleeping:
            remaining = state['sleep_gens_remaining']
            phase_info = f"{phase} ({remaining} gens left)"
        else:
            pos = state['cycle_position']
            until_sleep = self.wake_cycle_length - pos
            phase_info = f"{phase} ({until_sleep} gens until sleep)"

        # Above threshold count
        above_threshold = int(jnp.sum(weights >= self.renormalization_threshold))

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Phase
            'phase': phase,
            'phase_info': phase_info,
            'total_sleep_cycles': state['total_sleep_cycles'],
            # Weights
            'top_weights': top_weights,
            'above_threshold_count': above_threshold,
            # Stats
            'total_pruned': state['functions_pruned_in_sleep'],
            'total_added': state['functions_added_in_wake'],
            # Sin-specific
            'sin_weight': float(weights[4]),
            'sin_above_threshold': float(weights[4]) >= self.renormalization_threshold,
        }
