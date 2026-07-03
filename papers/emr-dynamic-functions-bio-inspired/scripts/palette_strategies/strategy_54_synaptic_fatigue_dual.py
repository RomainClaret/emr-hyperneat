"""Strategy 54D: Synaptic Fatigue Dual (Temporal Fatigue for Both Domains).

Extends SynapticFatigueStrategy to jointly evolve BOTH activation AND aggregation
function palettes using synaptic fatigue dynamics.

Key dual mechanisms:
1. Dual fatigue tracking - separate fatigue for act and agg functions
2. Dual base weights - independent learned weights in both domains
3. Cross-domain fatigue coupling - high fatigue in one domain can affect other
4. Coordinated rotation - natural rotation cycles in both domains

Expected: Use-dependent exploration in both domains
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


class SynapticFatigueDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with synaptic fatigue dynamics.

    Both activation and aggregation functions accumulate fatigue with use
    and recover when inactive. Creates natural rotation in both domains.
    """

    name = "synaptic_fatigue_dual"
    description = "Dual: Synaptic fatigue with use-dependent depression in both domains"

    def __init__(
        self,
        # Fatigue dynamics
        fatigue_rate: float = 0.15,
        recovery_rate: float = 0.08,
        effectiveness_floor: float = 0.3,
        # Success-dependent modulation
        success_fatigue_reduction: float = 0.3,
        failure_fatigue_boost: float = 0.1,
        # Base weights
        base_weight_learning_rate: float = 0.1,
        base_weight_decay: float = 0.99,
        initial_base_weight: float = 1.0,
        # Selection
        temperature: float = 0.5,
        min_effective_weight: float = 0.1,
        # Cross-domain coupling
        cross_fatigue_coupling: float = 0.1,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Synaptic Fatigue Dual strategy."""
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

        # Cross-domain
        self.cross_fatigue_coupling = cross_fatigue_coupling

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual fatigue tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize fatigue for both domains
        act_fatigue = jnp.zeros(NUM_ACTIVATIONS)
        agg_fatigue = jnp.zeros(NUM_AGGREGATIONS)

        # Initialize base weights
        act_weights = jnp.ones(NUM_ACTIVATIONS) * self.initial_base_weight
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_weights = act_weights.at[i].set(self.initial_base_weight * 1.2)

        agg_weights = jnp.ones(NUM_AGGREGATIONS) * self.initial_base_weight
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_weights = agg_weights.at[i].set(self.initial_base_weight * 1.2)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_fatigue': act_fatigue,
            'act_base_weights': act_weights,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_fatigue': agg_fatigue,
            'agg_base_weights': agg_weights,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 545454),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Rotation tracking
            'act_rotation_count': 0,
            'agg_rotation_count': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_effective_weights(
        self,
        base_weights: jnp.ndarray,
        fatigue: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective weights accounting for fatigue."""
        effectiveness = self.effectiveness_floor + (1 - self.effectiveness_floor) * (1 - fatigue)
        effective_weights = base_weights * effectiveness
        return jnp.maximum(effective_weights, self.min_effective_weight)

    def _update_fatigue(
        self,
        fatigue: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        stagnation: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update fatigue levels based on activity."""
        new_fatigue = fatigue.copy()

        for i in range(n_funcs):
            if mask[i] > 0.5:
                current = float(fatigue[i])
                delta = self.fatigue_rate * (1 - current)
                if improved:
                    delta *= (1 - self.success_fatigue_reduction)
                elif stagnation > 3:
                    delta *= (1 + self.failure_fatigue_boost)
                new_fatigue = new_fatigue.at[i].set(min(current + delta, 1.0))
            else:
                current = float(fatigue[i])
                new_value = current * (1 - self.recovery_rate)
                new_fatigue = new_fatigue.at[i].set(new_value)

        return new_fatigue

    def _update_base_weights(
        self,
        base_weights: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        improvement: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update base weights based on fitness."""
        new_weights = base_weights * self.base_weight_decay

        if improved:
            for i in range(n_funcs):
                if mask[i] > 0.5:
                    current = float(new_weights[i])
                    delta = self.base_weight_learning_rate * max(improvement, 0.1)
                    new_weights = new_weights.at[i].set(current + delta)

        return jnp.clip(new_weights, 0.1, 3.0)

    def _select_palette(
        self,
        effective_weights: jnp.ndarray,
        current_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
        palette_size: int,
        min_active: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette based on effective weights."""
        key1, key2 = jax.random.split(key)

        probs = jax.nn.softmax(effective_weights / self.temperature)
        top_k_count = max(min_active, palette_size - 2)
        top_indices = jnp.argsort(effective_weights)[-top_k_count:]

        new_mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            new_mask = new_mask.at[int(idx)].set(1.0)

        remaining = palette_size - top_k_count
        if remaining > 0:
            available_probs = probs * (1 - new_mask)
            available_probs = available_probs / (jnp.sum(available_probs) + 1e-8)
            for _ in range(remaining):
                key2, subkey = jax.random.split(key2)
                sample = jax.random.choice(subkey, n_funcs, p=available_probs)
                new_mask = new_mask.at[int(sample)].set(1.0)
                available_probs = available_probs.at[int(sample)].set(0)
                available_probs = available_probs / (jnp.sum(available_probs) + 1e-8)

        # Count rotations
        rotation_count = 0
        old_active = mask_to_indices(current_mask)
        new_active = mask_to_indices(new_mask)
        for i in old_active:
            if i not in new_active:
                rotation_count += 1

        return new_mask, rotation_count

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity based on success."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        delta = 0.1 * improvement * co_active
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
        """Update with dual synaptic fatigue dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update fatigue for both domains
        new_act_fatigue = self._update_fatigue(
            state['act_fatigue'], state['act_mask'],
            improved, new_stagnation, NUM_ACTIVATIONS
        )
        new_agg_fatigue = self._update_fatigue(
            state['agg_fatigue'], state['agg_mask'],
            improved, new_stagnation, NUM_AGGREGATIONS
        )

        # Cross-domain fatigue coupling
        act_mean_fatigue = float(jnp.mean(new_act_fatigue))
        agg_mean_fatigue = float(jnp.mean(new_agg_fatigue))
        if act_mean_fatigue > 0.6:
            new_agg_fatigue = new_agg_fatigue * (1 + self.cross_fatigue_coupling)
        if agg_mean_fatigue > 0.6:
            new_act_fatigue = new_act_fatigue * (1 + self.cross_fatigue_coupling)
        new_act_fatigue = jnp.clip(new_act_fatigue, 0, 1)
        new_agg_fatigue = jnp.clip(new_agg_fatigue, 0, 1)

        # Update base weights
        new_act_weights = self._update_base_weights(
            state['act_base_weights'], state['act_mask'],
            improved, improvement, NUM_ACTIVATIONS
        )
        new_agg_weights = self._update_base_weights(
            state['agg_base_weights'], state['agg_mask'],
            improved, improvement, NUM_AGGREGATIONS
        )

        # Compute effective weights
        act_eff_weights = self._compute_effective_weights(new_act_weights, new_act_fatigue)
        agg_eff_weights = self._compute_effective_weights(new_agg_weights, new_agg_fatigue)

        # Select new palettes
        new_act_mask, act_rotations = self._select_palette(
            act_eff_weights, state['act_mask'], k_act,
            self.act_palette_size, self.min_active_act, NUM_ACTIVATIONS
        )
        new_agg_mask, agg_rotations = self._select_palette(
            agg_eff_weights, state['agg_mask'], k_agg,
            self.agg_palette_size, self.min_active_agg, NUM_AGGREGATIONS
        )

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improvement
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_fatigue': new_act_fatigue,
            'act_base_weights': new_act_weights,
            'agg_mask': new_agg_mask,
            'agg_fatigue': new_agg_fatigue,
            'agg_base_weights': new_agg_weights,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'act_rotation_count': state['act_rotation_count'] + act_rotations,
            'agg_rotation_count': state['agg_rotation_count'] + agg_rotations,
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
            # Fatigue stats
            'act_mean_fatigue': float(jnp.mean(new_act_fatigue)),
            'agg_mean_fatigue': float(jnp.mean(new_agg_fatigue)),
            'act_max_fatigue': float(jnp.max(new_act_fatigue)),
            'agg_max_fatigue': float(jnp.max(new_agg_fatigue)),
            # Effective weights
            'act_mean_effective': float(jnp.mean(act_eff_weights)),
            'agg_mean_effective': float(jnp.mean(agg_eff_weights)),
            # Rotation
            'act_rotations_this_gen': act_rotations,
            'agg_rotations_this_gen': agg_rotations,
            'act_total_rotations': new_state['act_rotation_count'],
            'agg_total_rotations': new_state['agg_rotation_count'],
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_fatigue': float(new_act_fatigue[4]),
            'sin_effective_weight': float(act_eff_weights[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual fatigue status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        act_eff = self._compute_effective_weights(
            state['act_base_weights'], state['act_fatigue']
        )
        agg_eff = self._compute_effective_weights(
            state['agg_base_weights'], state['agg_fatigue']
        )

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_mean_fatigue': float(jnp.mean(state['act_fatigue'])),
            'agg_mean_fatigue': float(jnp.mean(state['agg_fatigue'])),
            'act_mean_effective': float(jnp.mean(act_eff)),
            'agg_mean_effective': float(jnp.mean(agg_eff)),
            'act_total_rotations': state['act_rotation_count'],
            'agg_total_rotations': state['agg_rotation_count'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
