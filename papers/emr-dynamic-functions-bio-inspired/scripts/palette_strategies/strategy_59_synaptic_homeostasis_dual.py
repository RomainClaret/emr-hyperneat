"""Strategy 59D: Synaptic Homeostasis Dual (Sleep-Inspired Consolidation for Both Domains).

Extends SynapticHomeostasisHypothesisStrategy to jointly evolve BOTH activation AND
aggregation function palettes using wake/sleep cycle dynamics.

Key dual mechanisms:
1. Dual weight tracking - separate weights for act and agg functions
2. Synchronized sleep - both domains enter sleep phase together
3. Coordinated downscaling - global scaling affects both domains
4. Cross-domain consolidation - successful pairs survive sleep better

Expected: Natural consolidation cycles in both domains
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

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


class SynapticHomeostasisDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with wake/sleep consolidation cycles.

    Both domains alternate between wake (learning) and sleep (downscaling)
    phases. Global downscaling during sleep prunes weak functions while
    preserving strong ones in both domains.
    """

    name = "synaptic_homeostasis_dual"
    description = "Dual: Wake/sleep cycles with consolidation in both domains"

    def __init__(
        self,
        # Wake/sleep cycle
        wake_cycle_length: int = 15,
        sleep_duration: int = 3,
        sleep_scaling_factor: float = 0.7,
        renormalization_threshold: float = 0.3,
        # Wake phase
        wake_learning_rate: float = 0.15,
        wake_exploration_rate: float = 0.1,
        # Sleep phase
        sleep_exploration_rate: float = 0.02,
        sleep_recovery_boost: float = 0.1,
        # Weight dynamics
        initial_weight: float = 1.0,
        max_weight: float = 3.0,
        min_weight: float = 0.05,
        weight_decay: float = 0.99,
        # Cross-domain
        cross_consolidation_rate: float = 0.1,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Synaptic Homeostasis Dual strategy."""
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

        # Weight
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.weight_decay = weight_decay

        # Cross-domain
        self.cross_consolidation_rate = cross_consolidation_rate

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual weight tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize weights
        act_weights = jnp.ones(NUM_ACTIVATIONS) * self.initial_weight
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_weights = act_weights.at[i].set(self.initial_weight * 1.2)

        agg_weights = jnp.ones(NUM_AGGREGATIONS) * self.initial_weight
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_weights = agg_weights.at[i].set(self.initial_weight * 1.2)

        # Cross-domain consolidation strength
        cross_strength = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_weights': act_weights,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_weights': agg_weights,
            # Cross-domain
            'cross_strength': cross_strength,
            # Wake/sleep state
            'is_sleeping': False,
            'cycle_position': 0,
            'sleep_gens_remaining': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 595959),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Stats
            'total_sleep_cycles': 0,
            'act_pruned_in_sleep': 0,
            'agg_pruned_in_sleep': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _enter_sleep(
        self,
        act_weights: jnp.ndarray,
        agg_weights: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Begin sleep phase with global downscaling."""
        new_act = act_weights * self.sleep_scaling_factor
        new_agg = agg_weights * self.sleep_scaling_factor
        return new_act, new_agg, self.sleep_duration

    def _process_sleep_gen(
        self,
        act_weights: jnp.ndarray,
        agg_weights: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_strength: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[int], List[int]]:
        """Process one generation of sleep."""
        new_act_w = act_weights * 0.95
        new_agg_w = agg_weights * 0.95
        new_act_m = act_mask.copy()
        new_agg_m = agg_mask.copy()
        act_pruned = []
        agg_pruned = []

        # Cross-domain boost for co-active pairs
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                cross_boost = float(jnp.sum(cross_strength[i, :] * active_agg))
                threshold = self.renormalization_threshold * (1 - cross_boost * 0.2)
                if new_act_w[i] < threshold:
                    new_act_m = new_act_m.at[i].set(0.0)
                    act_pruned.append(i)

        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                cross_boost = float(jnp.sum(cross_strength[:, i] * active_act))
                threshold = self.renormalization_threshold * (1 - cross_boost * 0.2)
                if new_agg_w[i] < threshold:
                    new_agg_m = new_agg_m.at[i].set(0.0)
                    agg_pruned.append(i)

        # Ensure minimum palettes
        act_active = mask_to_indices(new_act_m)
        if len(act_active) < self.min_active_act:
            top_idx = jnp.argsort(new_act_w)[-self.min_active_act:]
            for idx in top_idx:
                new_act_m = new_act_m.at[int(idx)].set(1.0)
                if int(idx) in act_pruned:
                    act_pruned.remove(int(idx))

        agg_active = mask_to_indices(new_agg_m)
        if len(agg_active) < self.min_active_agg:
            top_idx = jnp.argsort(new_agg_w)[-self.min_active_agg:]
            for idx in top_idx:
                new_agg_m = new_agg_m.at[int(idx)].set(1.0)
                if int(idx) in agg_pruned:
                    agg_pruned.remove(int(idx))

        return new_act_w, new_agg_w, new_act_m, new_agg_m, act_pruned, agg_pruned

    def _process_wake_gen(
        self,
        act_weights: jnp.ndarray,
        agg_weights: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_strength: jnp.ndarray,
        improved: bool,
        improvement: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Process one generation of wake."""
        key1, key2, key3, key4 = jax.random.split(key, 4)

        new_act_w = act_weights * self.weight_decay
        new_agg_w = agg_weights * self.weight_decay
        new_act_m = act_mask.copy()
        new_agg_m = agg_mask.copy()
        new_cross = cross_strength.copy()

        if improved:
            # Strengthen active functions
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    delta = self.wake_learning_rate * max(improvement, 0.1)
                    new_act_w = new_act_w.at[i].set(
                        min(float(new_act_w[i]) + delta, self.max_weight)
                    )

            for i in range(NUM_AGGREGATIONS):
                if agg_mask[i] > 0.5:
                    delta = self.wake_learning_rate * max(improvement, 0.1)
                    new_agg_w = new_agg_w.at[i].set(
                        min(float(new_agg_w[i]) + delta, self.max_weight)
                    )

            # Strengthen cross-domain pairs
            active_act = (act_mask > 0.5).astype(jnp.float32)
            active_agg = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active_act, active_agg)
            new_cross = new_cross + self.cross_consolidation_rate * improvement * co_active
            new_cross = jnp.clip(new_cross, 0.0, 1.5)

        # Maybe explore new activation function
        if jax.random.uniform(key1) < self.wake_exploration_rate:
            inactive = [i for i in range(NUM_ACTIVATIONS)
                       if act_mask[i] < 0.5 and act_weights[i] > self.min_weight]
            if inactive:
                idx = int(jax.random.randint(key2, (), 0, len(inactive)))
                new_func = inactive[idx]
                new_act_m = new_act_m.at[new_func].set(1.0)
                new_act_w = new_act_w.at[new_func].set(float(new_act_w[new_func]) + 0.3)

        # Maybe explore new aggregation function
        if jax.random.uniform(key3) < self.wake_exploration_rate:
            inactive = [i for i in range(NUM_AGGREGATIONS)
                       if agg_mask[i] < 0.5 and agg_weights[i] > self.min_weight]
            if inactive:
                idx = int(jax.random.randint(key4, (), 0, len(inactive)))
                new_func = inactive[idx]
                new_agg_m = new_agg_m.at[new_func].set(1.0)
                new_agg_w = new_agg_w.at[new_func].set(float(new_agg_w[new_func]) + 0.3)

        new_act_w = jnp.clip(new_act_w, self.min_weight, self.max_weight)
        new_agg_w = jnp.clip(new_agg_w, self.min_weight, self.max_weight)

        return new_act_w, new_agg_w, new_act_m, new_agg_m, new_cross

    def _update_mask_from_weights(
        self,
        weights: jnp.ndarray,
        is_sleeping: bool,
        palette_size: int,
        min_active: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update mask based on weights."""
        if is_sleeping:
            new_mask = jnp.where(weights >= self.renormalization_threshold, 1.0, 0.0)
            active = mask_to_indices(new_mask)
            if len(active) < min_active:
                top_idx = jnp.argsort(weights)[-min_active:]
                for idx in top_idx:
                    new_mask = new_mask.at[int(idx)].set(1.0)
            return new_mask
        else:
            top_idx = jnp.argsort(weights)[-palette_size:]
            new_mask = jnp.zeros(n_funcs)
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
        """Update with dual wake/sleep dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

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
        act_weights = state['act_weights']
        agg_weights = state['agg_weights']
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        cross_strength = state['cross_strength']
        total_cycles = state['total_sleep_cycles']
        act_pruned_total = state['act_pruned_in_sleep']
        agg_pruned_total = state['agg_pruned_in_sleep']
        act_pruned_this = []
        agg_pruned_this = []

        if is_sleeping:
            act_weights, agg_weights, act_mask, agg_mask, act_pruned_this, agg_pruned_this = \
                self._process_sleep_gen(act_weights, agg_weights, act_mask, agg_mask, cross_strength)
            sleep_gens_remaining -= 1
            act_pruned_total += len(act_pruned_this)
            agg_pruned_total += len(agg_pruned_this)

            if sleep_gens_remaining <= 0:
                is_sleeping = False
                cycle_position = 0
                total_cycles += 1
        else:
            act_weights, agg_weights, act_mask, agg_mask, cross_strength = \
                self._process_wake_gen(
                    act_weights, agg_weights, act_mask, agg_mask, cross_strength,
                    improved, improvement, k1
                )
            cycle_position += 1

            if cycle_position >= self.wake_cycle_length:
                is_sleeping = True
                act_weights, agg_weights, sleep_gens_remaining = \
                    self._enter_sleep(act_weights, agg_weights)

        new_act_mask = self._update_mask_from_weights(
            act_weights, is_sleeping, self.act_palette_size, self.min_active_act, NUM_ACTIVATIONS
        )
        new_agg_mask = self._update_mask_from_weights(
            agg_weights, is_sleeping, self.agg_palette_size, self.min_active_agg, NUM_AGGREGATIONS
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_weights': act_weights,
            'agg_mask': new_agg_mask,
            'agg_weights': agg_weights,
            'cross_strength': cross_strength,
            'is_sleeping': is_sleeping,
            'cycle_position': cycle_position,
            'sleep_gens_remaining': sleep_gens_remaining,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'total_sleep_cycles': total_cycles,
            'act_pruned_in_sleep': act_pruned_total,
            'agg_pruned_in_sleep': agg_pruned_total,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        phase = "SLEEP" if is_sleeping else "WAKE"

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Phase
            'phase': phase,
            'is_sleeping': is_sleeping,
            'cycle_position': cycle_position,
            'total_sleep_cycles': total_cycles,
            # Weights
            'act_mean_weight': float(jnp.mean(act_weights)),
            'agg_mean_weight': float(jnp.mean(agg_weights)),
            'act_max_weight': float(jnp.max(act_weights)),
            'agg_max_weight': float(jnp.max(agg_weights)),
            # Cross-domain
            'cross_mean_strength': float(jnp.mean(cross_strength)),
            # Pruning
            'act_pruned_this_gen': act_pruned_this,
            'agg_pruned_this_gen': agg_pruned_this,
            'act_total_pruned': act_pruned_total,
            'agg_total_pruned': agg_pruned_total,
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_weight': float(act_weights[4]),
            'sin_above_threshold': float(act_weights[4]) >= self.renormalization_threshold,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual wake/sleep status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        is_sleeping = state['is_sleeping']

        phase = "SLEEP" if is_sleeping else "WAKE"

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'phase': phase,
            'total_sleep_cycles': state['total_sleep_cycles'],
            'act_mean_weight': float(jnp.mean(state['act_weights'])),
            'agg_mean_weight': float(jnp.mean(state['agg_weights'])),
            'cross_mean_strength': float(jnp.mean(state['cross_strength'])),
            'act_total_pruned': state['act_pruned_in_sleep'],
            'agg_total_pruned': state['agg_pruned_in_sleep'],
            'sin_weight': float(state['act_weights'][4]),
        }
