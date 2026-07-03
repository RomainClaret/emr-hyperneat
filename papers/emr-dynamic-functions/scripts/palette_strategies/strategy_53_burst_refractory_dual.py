"""Strategy 53D: Burst Refractory Dual (Refractory Period Gating for Both Domains).

Extends BurstRefractoryStrategy to jointly evolve BOTH activation AND aggregation
function palettes using refractory period dynamics.

Key dual mechanisms:
1. Dual refractory tracking - separate burst/refractory states for act and agg
2. Independent burst thresholds - each domain can have different burst sensitivity
3. Cross-domain recovery - functions can boost recovery in the other domain
4. Coordinated selection - refractory in one domain can influence the other

Expected: Natural attention windows and forced exploration in both domains
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


class BurstRefractoryDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with refractory period gating.

    Functions in both domains that are continuously active enter refractory
    periods, creating natural attention windows and forced exploration cycles.
    """

    name = "burst_refractory_dual"
    description = "Dual: Burst firing triggers refractory in both domains"

    def __init__(
        self,
        # Burst detection
        burst_threshold: int = 4,
        consecutive_boost: float = 0.05,
        # Refractory period
        refractory_duration: int = 5,
        refractory_factor: float = 0.2,
        refractory_recovery_boost: float = 0.3,
        # Relative refractory
        relative_refractory_duration: int = 2,
        relative_refractory_factor: float = 0.6,
        # Base selection
        base_weight: float = 1.0,
        fitness_weight_learning: float = 0.1,
        # Cross-domain
        cross_learning_rate: float = 0.08,
        cross_recovery_bonus: float = 0.1,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Burst Refractory Dual strategy."""
        self.burst_threshold = burst_threshold
        self.consecutive_boost = consecutive_boost
        self.refractory_duration = refractory_duration
        self.refractory_factor = refractory_factor
        self.refractory_recovery_boost = refractory_recovery_boost
        self.relative_refractory_duration = relative_refractory_duration
        self.relative_refractory_factor = relative_refractory_factor
        self.base_weight = base_weight
        self.fitness_weight_learning = fitness_weight_learning
        self.cross_learning_rate = cross_learning_rate
        self.cross_recovery_bonus = cross_recovery_bonus
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual refractory tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_consecutive': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'act_absolute_ref': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'act_relative_ref': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'act_weights': jnp.ones(NUM_ACTIVATIONS) * self.base_weight,
            'act_total_bursts': jnp.zeros(NUM_ACTIVATIONS),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_consecutive': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
            'agg_absolute_ref': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
            'agg_relative_ref': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
            'agg_weights': jnp.ones(NUM_AGGREGATIONS) * self.base_weight,
            'agg_total_bursts': jnp.zeros(NUM_AGGREGATIONS),
            # Cross-domain
            'cross_affinity': cross_affinity,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 535353),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_refractory(
        self,
        consecutive: jnp.ndarray,
        absolute_ref: jnp.ndarray,
        relative_ref: jnp.ndarray,
        mask: jnp.ndarray,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update burst tracking and refractory states."""
        new_consecutive = consecutive.copy()
        new_absolute = absolute_ref.copy()
        new_relative = relative_ref.copy()
        new_bursts = jnp.zeros(n_funcs)
        refractory_exits = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            if absolute_ref[i] > 0:
                new_absolute = new_absolute.at[i].set(int(absolute_ref[i]) - 1)
                if absolute_ref[i] == 1 and self.relative_refractory_duration > 0:
                    new_relative = new_relative.at[i].set(self.relative_refractory_duration)
            elif relative_ref[i] > 0:
                new_relative = new_relative.at[i].set(int(relative_ref[i]) - 1)
                if relative_ref[i] == 1:
                    refractory_exits = refractory_exits.at[i].set(1)

            if mask[i] > 0.5:
                if absolute_ref[i] == 0 and relative_ref[i] == 0:
                    new_consecutive = new_consecutive.at[i].set(int(consecutive[i]) + 1)
                    if new_consecutive[i] >= self.burst_threshold:
                        new_absolute = new_absolute.at[i].set(self.refractory_duration)
                        new_consecutive = new_consecutive.at[i].set(0)
                        new_bursts = new_bursts.at[i].set(1)
            else:
                new_consecutive = new_consecutive.at[i].set(0)

        return new_consecutive, new_absolute, new_relative, new_bursts, refractory_exits

    def _compute_effective_weights(
        self,
        base_weights: jnp.ndarray,
        consecutive: jnp.ndarray,
        absolute_ref: jnp.ndarray,
        relative_ref: jnp.ndarray,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute effective selection weights with refractory modulation."""
        effective = base_weights.copy()

        for i in range(n_funcs):
            current = float(base_weights[i])
            if absolute_ref[i] > 0:
                effective = effective.at[i].set(current * self.refractory_factor)
            elif relative_ref[i] > 0:
                effective = effective.at[i].set(current * self.relative_refractory_factor)
            else:
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
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update base weights from fitness and refractory exit."""
        new_weights = weights.copy()

        if improved:
            for i in range(n_funcs):
                if mask[i] > 0.5:
                    new_weights = new_weights.at[i].set(float(weights[i]) + self.fitness_weight_learning)

        for i in range(n_funcs):
            if refractory_exits[i] > 0:
                new_weights = new_weights.at[i].set(float(new_weights[i]) + self.refractory_recovery_boost)

        new_weights = new_weights * 0.98 + self.base_weight * 0.02
        return jnp.clip(new_weights, 0.1, 3.0)

    def _select_palette(
        self,
        effective_weights: jnp.ndarray,
        key: jax.random.PRNGKey,
        palette_size: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Select palette based on effective weights."""
        key1, key2 = jax.random.split(key)

        probs = jax.nn.softmax(effective_weights)
        top_k = palette_size - 1
        top_indices = jnp.argsort(effective_weights)[-top_k:]

        new_mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            new_mask = new_mask.at[int(idx)].set(1.0)

        available_probs = probs * (1 - new_mask)
        available_probs = available_probs / (jnp.sum(available_probs) + 1e-8)
        sample = jax.random.choice(key2, n_funcs, p=available_probs)
        new_mask = new_mask.at[int(sample)].set(1.0)

        return new_mask

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        delta = self.cross_learning_rate * improvement * co_active
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
        """Update with dual burst-refractory dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update refractory states for activation domain
        (act_consec, act_abs, act_rel, act_bursts, act_exits) = self._update_refractory(
            state['act_consecutive'], state['act_absolute_ref'], state['act_relative_ref'],
            state['act_mask'], NUM_ACTIVATIONS
        )

        # Update refractory states for aggregation domain
        (agg_consec, agg_abs, agg_rel, agg_bursts, agg_exits) = self._update_refractory(
            state['agg_consecutive'], state['agg_absolute_ref'], state['agg_relative_ref'],
            state['agg_mask'], NUM_AGGREGATIONS
        )

        # Update cross-affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improvement
        )

        # Update base weights
        new_act_weights = self._update_base_weights(
            state['act_weights'], state['act_mask'], improved, act_exits, NUM_ACTIVATIONS
        )
        new_agg_weights = self._update_base_weights(
            state['agg_weights'], state['agg_mask'], improved, agg_exits, NUM_AGGREGATIONS
        )

        # Compute effective weights
        act_eff = self._compute_effective_weights(new_act_weights, act_consec, act_abs, act_rel, NUM_ACTIVATIONS)
        agg_eff = self._compute_effective_weights(new_agg_weights, agg_consec, agg_abs, agg_rel, NUM_AGGREGATIONS)

        # Select palettes
        new_act_mask = self._select_palette(act_eff, k_act, self.act_palette_size, NUM_ACTIVATIONS)
        new_agg_mask = self._select_palette(agg_eff, k_agg, self.agg_palette_size, NUM_AGGREGATIONS)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_consecutive': act_consec,
            'act_absolute_ref': act_abs,
            'act_relative_ref': act_rel,
            'act_weights': new_act_weights,
            'act_total_bursts': state['act_total_bursts'] + act_bursts,
            'agg_mask': new_agg_mask,
            'agg_consecutive': agg_consec,
            'agg_absolute_ref': agg_abs,
            'agg_relative_ref': agg_rel,
            'agg_weights': new_agg_weights,
            'agg_total_bursts': state['agg_total_bursts'] + agg_bursts,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        act_in_ref = [i for i in range(NUM_ACTIVATIONS) if act_abs[i] > 0 or act_rel[i] > 0]
        agg_in_ref = [i for i in range(NUM_AGGREGATIONS) if agg_abs[i] > 0 or agg_rel[i] > 0]

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'act_n_bursts_this_gen': int(jnp.sum(act_bursts)),
            'agg_n_bursts_this_gen': int(jnp.sum(agg_bursts)),
            'act_in_refractory': act_in_ref,
            'agg_in_refractory': agg_in_ref,
            'act_total_bursts': int(jnp.sum(new_state['act_total_bursts'])),
            'agg_total_bursts': int(jnp.sum(new_state['agg_total_bursts'])),
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'has_sin': 4 in act_palette,
            'sin_consecutive': int(act_consec[4]),
            'sin_in_refractory': act_abs[4] > 0 or act_rel[4] > 0,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual burst/refractory status."""
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'generation': state['generation'],
            'act_total_bursts': int(jnp.sum(state['act_total_bursts'])),
            'agg_total_bursts': int(jnp.sum(state['agg_total_bursts'])),
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
