"""Strategy 35: Synaptic Fatigue (Temporal Fatigue Dynamics).

Implements synaptic fatigue principles for palette evolution. Functions that
are repeatedly used accumulate fatigue and become less effective, while
inactive functions recover. Creates natural exploration cycles.

Biological Basis:
- Synapses weaken with repeated activation (vesicle depletion)
- Neurotransmitter pools require time to replenish
- Short-term synaptic depression is ubiquitous in neural circuits
- Recovery during non-use restores synaptic efficacy

Key Insight:
- Current strategies don't account for overuse degradation
- Synaptic fatigue creates natural novelty detection
- Overused functions become less effective, forcing exploration
- Recovery allows rediscovery of previously fatigued functions

Fatigue Mechanism:
    # Active functions accumulate fatigue
    for f in active_palette:
        fatigue[f] += fatigue_rate * (1 - fatigue[f])

    # Inactive functions recover
    for f not in active_palette:
        fatigue[f] *= (1 - recovery_rate)

    # Effective weight is reduced by fatigue
    effective_weight[f] = base_weight[f] * (effectiveness_floor + (1 - effectiveness_floor) * (1 - fatigue[f]))

    # Selection uses effective weights
    selection_prob = softmax(effective_weight)

Expected improvements:
- Natural novelty detection (overuse triggers exploration)
- Automatic function rotation
- Prevention of getting stuck on suboptimal functions
- Self-correcting exploration dynamics
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


class SynapticFatigueStrategy(PaletteEvolutionStrategy):
    """Temporal fatigue dynamics forcing exploration.

    Functions accumulate fatigue with use and recover when inactive.
    Effective weight is reduced by fatigue, naturally rotating functions.
    Creates self-correcting exploration without explicit mechanisms.
    """

    name = "synaptic_fatigue"
    description = "Synaptic fatigue with use-dependent depression and recovery"

    def __init__(
        self,
        # Fatigue dynamics
        fatigue_rate: float = 0.15,             # How fast fatigue accumulates
        recovery_rate: float = 0.08,            # How fast inactive functions recover
        effectiveness_floor: float = 0.3,       # Minimum effectiveness when fully fatigued
        # Success-dependent modulation
        success_fatigue_reduction: float = 0.3, # Fatigue reduction on fitness improvement
        failure_fatigue_boost: float = 0.1,     # Extra fatigue on stagnation
        # Base weights (learned from fitness)
        base_weight_learning_rate: float = 0.1, # How fast base weights adapt
        base_weight_decay: float = 0.99,        # Slow decay toward neutral
        initial_base_weight: float = 1.0,       # Starting base weight
        # Selection parameters
        temperature: float = 0.5,               # Softmax temperature
        min_effective_weight: float = 0.1,      # Floor for selection probability
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Synaptic Fatigue strategy.

        Args:
            fatigue_rate: Rate of fatigue accumulation (0-1)
            recovery_rate: Rate of recovery when inactive (0-1)
            effectiveness_floor: Minimum effectiveness at max fatigue
            success_fatigue_reduction: Fatigue reduction on fitness improvement
            failure_fatigue_boost: Extra fatigue on stagnation
            base_weight_learning_rate: Rate of base weight adaptation
            base_weight_decay: Slow decay of base weights
            initial_base_weight: Starting weight for all functions
            temperature: Softmax temperature for selection
            min_effective_weight: Floor for selection probability
            palette_size: Target number of active functions
        """
        # Fatigue dynamics
        self.fatigue_rate = fatigue_rate
        self.recovery_rate = recovery_rate
        self.effectiveness_floor = effectiveness_floor

        # Success modulation
        self.success_fatigue_reduction = success_fatigue_reduction
        self.failure_fatigue_boost = failure_fatigue_boost

        # Base weights
        self.base_weight_learning_rate = base_weight_learning_rate
        self.base_weight_decay = base_weight_decay
        self.initial_base_weight = initial_base_weight

        # Selection
        self.temperature = temperature
        self.min_effective_weight = min_effective_weight

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with fatigue tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize fatigue (0 = fresh, 1 = fully fatigued)
        fatigue = jnp.zeros(NUM_ACTIVATIONS)

        # Initialize base weights (initial palette gets boost)
        base_weights = jnp.ones(NUM_ACTIVATIONS) * self.initial_base_weight
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                base_weights = base_weights.at[i].set(self.initial_base_weight * 1.2)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 353535),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Fatigue state
            'fatigue': fatigue,
            'base_weights': base_weights,
            # Tracking
            'total_fatigue_accumulated': jnp.zeros(NUM_ACTIVATIONS),
            'recovery_events': jnp.zeros(NUM_ACTIVATIONS),
            'previous_mask': mask,
            'fitness_history': [],
            # Rotation tracking
            'rotation_count': 0,  # How often functions rotated out due to fatigue
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_effective_weights(
        self,
        base_weights: jnp.ndarray,
        fatigue: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective weights accounting for fatigue."""
        # Effectiveness = floor + (1 - floor) * (1 - fatigue)
        effectiveness = self.effectiveness_floor + (1 - self.effectiveness_floor) * (1 - fatigue)
        effective_weights = base_weights * effectiveness
        return jnp.maximum(effective_weights, self.min_effective_weight)

    def _update_fatigue(
        self,
        fatigue: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        stagnation: int,
    ) -> jnp.ndarray:
        """Update fatigue levels based on activity."""
        new_fatigue = fatigue.copy()

        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                # Active: accumulate fatigue
                current = float(fatigue[i])
                delta = self.fatigue_rate * (1 - current)

                # Modulate by success
                if improved:
                    delta *= (1 - self.success_fatigue_reduction)
                elif stagnation > 3:
                    delta *= (1 + self.failure_fatigue_boost)

                new_fatigue = new_fatigue.at[i].set(min(current + delta, 1.0))
            else:
                # Inactive: recover
                current = float(fatigue[i])
                new_value = current * (1 - self.recovery_rate)
                new_fatigue = new_fatigue.at[i].set(new_value)

        return new_fatigue

    def _update_base_weights(
        self,
        base_weights: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        improvement_amount: float,
    ) -> jnp.ndarray:
        """Update base weights based on fitness."""
        new_weights = base_weights * self.base_weight_decay  # Slow decay

        if improved:
            # Boost weights of active functions
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    current = float(new_weights[i])
                    delta = self.base_weight_learning_rate * max(improvement_amount, 0.1)
                    new_weights = new_weights.at[i].set(current + delta)

        return jnp.clip(new_weights, 0.1, 3.0)

    def _select_palette(
        self,
        effective_weights: jnp.ndarray,
        current_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette based on effective weights."""
        key1, key2 = jax.random.split(key)

        # Compute selection probabilities
        probs = jax.nn.softmax(effective_weights / self.temperature)

        # Mix of top-k selection and probabilistic sampling
        top_k_count = max(self.min_active, self.palette_size - 2)

        # Get top-k by effective weight
        top_indices = jnp.argsort(effective_weights)[-top_k_count:]

        new_mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            new_mask = new_mask.at[int(idx)].set(1.0)

        # Sample remaining slots probabilistically
        remaining = self.palette_size - top_k_count
        if remaining > 0:
            # Zero out already selected
            available_probs = probs * (1 - new_mask)
            available_probs = available_probs / (jnp.sum(available_probs) + 1e-8)

            for _ in range(remaining):
                key2, subkey = jax.random.split(key2)
                sample = jax.random.choice(subkey, NUM_ACTIVATIONS, p=available_probs)
                new_mask = new_mask.at[int(sample)].set(1.0)
                available_probs = available_probs.at[int(sample)].set(0)
                available_probs = available_probs / (jnp.sum(available_probs) + 1e-8)

        # Count rotations (functions that left due to low effective weight)
        rotation_count = 0
        old_active = mask_to_indices(current_mask)
        new_active = mask_to_indices(new_mask)
        for i in old_active:
            if i not in new_active:
                rotation_count += 1

        return new_mask, rotation_count

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with synaptic fatigue dynamics."""
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

        # Step 1: Update fatigue levels
        new_fatigue = self._update_fatigue(
            state['fatigue'],
            state['mask'],
            improved,
            new_stagnation,
        )

        # Step 2: Update base weights
        new_base_weights = self._update_base_weights(
            state['base_weights'],
            state['mask'],
            improved,
            improvement,
        )

        # Step 3: Compute effective weights
        effective_weights = self._compute_effective_weights(new_base_weights, new_fatigue)

        # Step 4: Select new palette
        new_mask, rotation_count = self._select_palette(effective_weights, state['mask'], k1)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track fatigue accumulation
        fatigue_delta = jnp.maximum(new_fatigue - state['fatigue'], 0)
        new_total_fatigue = state['total_fatigue_accumulated'] + fatigue_delta

        # Track recovery events
        recovery_events = state['recovery_events'].copy()
        for i in range(NUM_ACTIVATIONS):
            if state['fatigue'][i] > 0.5 and new_fatigue[i] < 0.3:
                recovery_events = recovery_events.at[i].add(1)

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
            # Fatigue state
            'fatigue': new_fatigue,
            'base_weights': new_base_weights,
            # Tracking
            'total_fatigue_accumulated': new_total_fatigue,
            'recovery_events': recovery_events,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            'rotation_count': state['rotation_count'] + rotation_count,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Fatigue stats
        active_fatigue = [float(new_fatigue[i]) for i in active_palette]

        # Most fatigued functions
        top_fatigue_idx = jnp.argsort(new_fatigue)[-5:][::-1]
        top_fatigued = [(int(i), float(new_fatigue[i])) for i in top_fatigue_idx]

        # Most recovered
        recovery_idx = jnp.argsort(state['fatigue'] - new_fatigue)[-3:][::-1]
        most_recovered = [(int(i), float(state['fatigue'][i] - new_fatigue[i])) for i in recovery_idx]

        # Effective weights for active palette
        active_effective = [(int(i), float(effective_weights[i])) for i in active_palette]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Fatigue stats
            'mean_fatigue': float(jnp.mean(new_fatigue)),
            'max_fatigue': float(jnp.max(new_fatigue)),
            'active_mean_fatigue': float(np.mean(active_fatigue)) if active_fatigue else 0.0,
            'top_fatigued': top_fatigued,
            'most_recovered': most_recovered,
            # Effective weights
            'mean_effective_weight': float(jnp.mean(effective_weights)),
            'active_effective_weights': active_effective,
            # Rotation
            'rotations_this_gen': rotation_count,
            'total_rotations': state['rotation_count'] + rotation_count,
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_fatigue': float(new_fatigue[4]),
            'sin_base_weight': float(new_base_weights[4]),
            'sin_effective_weight': float(effective_weights[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with fatigue status."""
        palette = self.get_active_palette(state)
        fatigue = state['fatigue']
        base_weights = state['base_weights']
        effective = self._compute_effective_weights(base_weights, fatigue)

        # Top functions by effective weight
        top_eff = jnp.argsort(effective)[-5:][::-1]
        top_effective = [(int(i), float(effective[i])) for i in top_eff]

        # Most fatigued
        top_fat = jnp.argsort(fatigue)[-5:][::-1]
        most_fatigued = [(int(i), float(fatigue[i])) for i in top_fat]

        # Freshest (lowest fatigue)
        freshest_idx = jnp.argsort(fatigue)[:3]
        freshest = [(int(i), float(fatigue[i])) for i in freshest_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Fatigue
            'mean_fatigue': float(jnp.mean(fatigue)),
            'most_fatigued': most_fatigued,
            'freshest': freshest,
            # Weights
            'top_effective': top_effective,
            # Rotation
            'total_rotations': state['rotation_count'],
            # Sin-specific
            'sin_fatigue': float(fatigue[4]),
            'sin_effective_weight': float(effective[4]),
        }
