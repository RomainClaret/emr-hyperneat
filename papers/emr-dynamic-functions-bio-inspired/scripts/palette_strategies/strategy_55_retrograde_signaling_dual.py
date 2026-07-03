"""Strategy 55D: Retrograde Signaling Dual (Backward Credit for Both Domains).

Extends RetrogradeSignalingStrategy to jointly evolve BOTH activation AND aggregation
function palettes using backward credit propagation.

Key dual mechanisms:
1. Dual eligibility traces - separate eligibility for act and agg functions
2. Dual activation history - track palette changes in both domains
3. Cross-domain credit - successful pairs receive coordinated credit
4. Temporal credit assignment in both domains

Expected: Temporal credit assignment guides both domains
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from collections import deque

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


class RetrogradeSignalingDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with backward credit propagation.

    Both activation and aggregation functions accumulate eligibility
    through backward credit flow. Temporal credit assignment enables
    discovery of important functions in both domains.
    """

    name = "retrograde_signaling_dual"
    description = "Dual: Backward credit propagation for both domains"

    def __init__(
        self,
        # Eligibility trace dynamics
        trace_decay: float = 0.7,
        eligibility_decay: float = 0.9,
        eligibility_max: float = 2.0,
        # Credit assignment
        credit_scale: float = 0.5,
        improvement_credit: float = 1.0,
        failure_penalty: float = 0.3,
        # History tracking
        history_length: int = 8,
        # Selection
        base_weight: float = 1.0,
        exploration_bonus: float = 0.2,
        # Cross-domain
        cross_credit_rate: float = 0.15,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Retrograde Signaling Dual strategy."""
        # Eligibility traces
        self.trace_decay = trace_decay
        self.eligibility_decay = eligibility_decay
        self.eligibility_max = eligibility_max

        # Credit assignment
        self.credit_scale = credit_scale
        self.improvement_credit = improvement_credit
        self.failure_penalty = failure_penalty

        # History
        self.history_length = history_length

        # Selection
        self.base_weight = base_weight
        self.exploration_bonus = exploration_bonus

        # Cross-domain
        self.cross_credit_rate = cross_credit_rate

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual eligibility tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize eligibility traces
        act_eligibility = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_eligibility = act_eligibility.at[i].set(0.3)

        agg_eligibility = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_eligibility = agg_eligibility.at[i].set(0.3)

        # Cross-domain credit matrix
        cross_credit = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_eligibility': act_eligibility,
            'act_history': [act_mask.tolist()],
            'act_total_credit': jnp.zeros(NUM_ACTIVATIONS),
            'act_total_penalty': jnp.zeros(NUM_ACTIVATIONS),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_eligibility': agg_eligibility,
            'agg_history': [agg_mask.tolist()],
            'agg_total_credit': jnp.zeros(NUM_AGGREGATIONS),
            'agg_total_penalty': jnp.zeros(NUM_AGGREGATIONS),
            # Cross-domain
            'cross_credit': cross_credit,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 555555),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            'propagation_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _propagate_credit_backward(
        self,
        eligibility: jnp.ndarray,
        activation_history: List[List[float]],
        credit: float,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Propagate credit backward through activation history."""
        new_eligibility = eligibility.copy()
        credit_assigned = jnp.zeros(n_funcs)

        for t, past_mask in enumerate(reversed(activation_history)):
            temporal_weight = self.trace_decay ** t
            for i in range(n_funcs):
                if past_mask[i] > 0.5:
                    delta = credit * temporal_weight
                    new_eligibility = new_eligibility.at[i].set(
                        min(float(new_eligibility[i]) + delta, self.eligibility_max)
                    )
                    credit_assigned = credit_assigned.at[i].add(delta)

        return new_eligibility, credit_assigned

    def _propagate_penalty_backward(
        self,
        eligibility: jnp.ndarray,
        activation_history: List[List[float]],
        penalty: float,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Propagate penalty backward through history."""
        new_eligibility = eligibility.copy()
        penalty_assigned = jnp.zeros(n_funcs)

        for t, past_mask in enumerate(reversed(activation_history)):
            temporal_weight = self.trace_decay ** t
            for i in range(n_funcs):
                if past_mask[i] > 0.5:
                    delta = penalty * temporal_weight
                    new_eligibility = new_eligibility.at[i].set(
                        max(float(new_eligibility[i]) - delta, -self.eligibility_max * 0.5)
                    )
                    penalty_assigned = penalty_assigned.at[i].add(delta)

        return new_eligibility, penalty_assigned

    def _compute_selection_weights(
        self,
        eligibility: jnp.ndarray,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute selection weights from eligibility."""
        weights = self.base_weight + eligibility * self.credit_scale

        for i in range(n_funcs):
            if eligibility[i] < 0.1:
                weights = weights.at[i].set(float(weights[i]) + self.exploration_bonus)

        return jnp.maximum(weights, 0.1)

    def _select_palette(
        self,
        weights: jnp.ndarray,
        key: jax.random.PRNGKey,
        palette_size: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Select palette based on weights."""
        key1, key2 = jax.random.split(key)

        probs = jax.nn.softmax(weights)
        n_top = palette_size - 1
        top_indices = jnp.argsort(weights)[-n_top:]

        new_mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            new_mask = new_mask.at[int(idx)].set(1.0)

        available_probs = probs * (1 - new_mask)
        available_probs = available_probs / (jnp.sum(available_probs) + 1e-8)
        sample = jax.random.choice(key2, n_funcs, p=available_probs)
        new_mask = new_mask.at[int(sample)].set(1.0)

        return new_mask

    def _update_cross_credit(
        self,
        cross_credit: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        credit: float,
    ) -> jnp.ndarray:
        """Update cross-domain credit for co-active pairs."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        delta = self.cross_credit_rate * credit * co_active
        new_cross = cross_credit + delta
        return jnp.clip(new_cross, -2.0, 2.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual retrograde credit assignment."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Apply eligibility decay
        new_act_elig = state['act_eligibility'] * self.eligibility_decay
        new_agg_elig = state['agg_eligibility'] * self.eligibility_decay

        # Backward propagation
        act_credit = jnp.zeros(NUM_ACTIVATIONS)
        agg_credit = jnp.zeros(NUM_AGGREGATIONS)
        act_penalty = jnp.zeros(NUM_ACTIVATIONS)
        agg_penalty = jnp.zeros(NUM_AGGREGATIONS)
        propagation_event = 0

        if delta > 0:
            credit = self.improvement_credit * delta
            new_act_elig, act_credit = self._propagate_credit_backward(
                new_act_elig, state['act_history'], credit, NUM_ACTIVATIONS
            )
            new_agg_elig, agg_credit = self._propagate_credit_backward(
                new_agg_elig, state['agg_history'], credit, NUM_AGGREGATIONS
            )
            propagation_event = 1
        elif delta < 0:
            penalty = self.failure_penalty * abs(delta)
            new_act_elig, act_penalty = self._propagate_penalty_backward(
                new_act_elig, state['act_history'], penalty, NUM_ACTIVATIONS
            )
            new_agg_elig, agg_penalty = self._propagate_penalty_backward(
                new_agg_elig, state['agg_history'], penalty, NUM_AGGREGATIONS
            )
            propagation_event = 1

        # Update cross-domain credit
        new_cross = self._update_cross_credit(
            state['cross_credit'], state['act_mask'], state['agg_mask'],
            delta if delta > 0 else 0
        )

        # Compute selection weights
        act_weights = self._compute_selection_weights(new_act_elig, NUM_ACTIVATIONS)
        agg_weights = self._compute_selection_weights(new_agg_elig, NUM_AGGREGATIONS)

        # Select new palettes
        new_act_mask = self._select_palette(act_weights, k_act, self.act_palette_size, NUM_ACTIVATIONS)
        new_agg_mask = self._select_palette(agg_weights, k_agg, self.agg_palette_size, NUM_AGGREGATIONS)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Update history
        act_history = list(state['act_history']) + [new_act_mask.tolist()]
        if len(act_history) > self.history_length:
            act_history = act_history[-self.history_length:]

        agg_history = list(state['agg_history']) + [new_agg_mask.tolist()]
        if len(agg_history) > self.history_length:
            agg_history = agg_history[-self.history_length:]

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_eligibility': new_act_elig,
            'act_history': act_history,
            'act_total_credit': state['act_total_credit'] + act_credit,
            'act_total_penalty': state['act_total_penalty'] + act_penalty,
            'agg_mask': new_agg_mask,
            'agg_eligibility': new_agg_elig,
            'agg_history': agg_history,
            'agg_total_credit': state['agg_total_credit'] + agg_credit,
            'agg_total_penalty': state['agg_total_penalty'] + agg_penalty,
            'cross_credit': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'propagation_events': state['propagation_events'] + propagation_event,
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
            'fitness_delta': delta,
            # Eligibility
            'act_mean_eligibility': float(jnp.mean(new_act_elig)),
            'agg_mean_eligibility': float(jnp.mean(new_agg_elig)),
            'act_max_eligibility': float(jnp.max(new_act_elig)),
            'agg_max_eligibility': float(jnp.max(new_agg_elig)),
            # Propagation
            'propagation_events_total': new_state['propagation_events'],
            'act_history_length': len(act_history),
            'agg_history_length': len(agg_history),
            # Cross-domain
            'cross_mean_credit': float(jnp.mean(new_cross)),
            'cross_max_credit': float(jnp.max(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_eligibility': float(new_act_elig[4]),
            'sin_total_credit': float(new_state['act_total_credit'][4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual eligibility status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        act_net_credit = state['act_total_credit'] - state['act_total_penalty']
        agg_net_credit = state['agg_total_credit'] - state['agg_total_penalty']

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_mean_eligibility': float(jnp.mean(state['act_eligibility'])),
            'agg_mean_eligibility': float(jnp.mean(state['agg_eligibility'])),
            'act_mean_net_credit': float(jnp.mean(act_net_credit)),
            'agg_mean_net_credit': float(jnp.mean(agg_net_credit)),
            'total_propagation_events': state['propagation_events'],
            'cross_mean_credit': float(jnp.mean(state['cross_credit'])),
        }
