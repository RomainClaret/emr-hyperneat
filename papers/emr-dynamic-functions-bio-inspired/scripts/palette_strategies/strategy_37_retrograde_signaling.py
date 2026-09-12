"""Strategy 37: Retrograde Signaling (Backward Credit Assignment).

Implements retrograde signaling principles for palette evolution. Credit from
fitness improvements flows backward through the sequence of recently active
functions, creating temporal credit assignment without explicit gradients.

Biological Basis:
- Postsynaptic neurons send retrograde signals (endocannabinoids, nitric oxide)
- These signals modulate presynaptic activity
- Enables local credit assignment and plasticity
- Creates temporal associations between recent activity and outcomes

Key Insight:
- Current strategies assign credit equally to all active functions
- Retrograde signaling creates temporal credit assignment
- Recent function additions get more credit than older ones
- Functions that preceded success get credit proportionally

Retrograde Mechanism:
    # Track activation sequence (palette history)
    activation_chains.append(current_palette)

    # On fitness improvement: propagate credit backward
    if fitness_improved:
        for t, palette in enumerate(reversed(activation_chains)):
            temporal_weight = trace_decay ** t
            for func in palette:
                eligibility[func] += fitness_delta * temporal_weight

    # Update selection weights based on accumulated eligibility
    selection_weight = base_weight + eligibility * credit_scale

    # Decay eligibility over time
    eligibility *= decay_rate

Expected improvements:
- Temporal credit assignment (recent → more credit)
- Functions that preceded success are strengthened
- No need for explicit gradient computation
- Self-organizing credit flow through activation history
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from collections import deque

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    DEFAULT_PALETTE_INDICES,
)


class RetrogradeSignalingStrategy(PaletteEvolutionStrategy):
    """Backward credit assignment through activation chains.

    Credit flows backward from fitness improvements through recent
    activation history. Creates temporal credit assignment without
    explicit gradients.
    """

    name = "retrograde_signaling"
    description = "Backward credit propagation through activation chains"

    def __init__(
        self,
        # Eligibility trace dynamics
        trace_decay: float = 0.7,               # Decay per generation in chain
        eligibility_decay: float = 0.9,         # Overall eligibility decay
        eligibility_max: float = 2.0,           # Max eligibility value
        # Credit assignment
        credit_scale: float = 0.5,              # How much eligibility affects selection
        improvement_credit: float = 1.0,        # Credit for fitness improvement
        failure_penalty: float = 0.3,           # Penalty for fitness decrease
        # History tracking
        history_length: int = 8,                # Generations to track
        # Selection
        base_weight: float = 1.0,               # Starting selection weight
        exploration_bonus: float = 0.2,         # Bonus for low-eligibility functions
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Retrograde Signaling strategy.

        Args:
            trace_decay: Temporal decay for backward credit (0.7 = 30% decay per step)
            eligibility_decay: Per-generation decay of accumulated eligibility
            eligibility_max: Maximum eligibility cap
            credit_scale: Scaling factor for eligibility → selection weight
            improvement_credit: Base credit assigned on improvement
            failure_penalty: Penalty magnitude on fitness decrease
            history_length: Number of generations to track for backward credit
            base_weight: Starting selection weight
            exploration_bonus: Weight bonus for functions with low eligibility
            palette_size: Target number of active functions
        """
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

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with eligibility tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize eligibility traces (start slightly positive for initial palette)
        eligibility = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                eligibility = eligibility.at[i].set(0.3)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 373737),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Eligibility state
            'eligibility': eligibility,
            # Activation history (list of palette masks for backward propagation)
            'activation_history': [mask.tolist()],
            'fitness_history': [],
            # Tracking
            'total_credit_received': jnp.zeros(NUM_ACTIVATIONS),
            'total_penalty_received': jnp.zeros(NUM_ACTIVATIONS),
            'backward_propagation_events': 0,
            'previous_mask': mask,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _propagate_credit_backward(
        self,
        eligibility: jnp.ndarray,
        activation_history: List[List[float]],
        credit: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Propagate credit backward through activation history."""
        new_eligibility = eligibility.copy()
        credit_assigned = jnp.zeros(NUM_ACTIVATIONS)

        # Iterate backward through history (most recent first)
        for t, past_mask in enumerate(reversed(activation_history)):
            temporal_weight = self.trace_decay ** t

            # Credit to functions that were active at time t
            for i in range(NUM_ACTIVATIONS):
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Propagate penalty backward (more recent = more penalty)."""
        new_eligibility = eligibility.copy()
        penalty_assigned = jnp.zeros(NUM_ACTIVATIONS)

        # Iterate backward (most recent gets more penalty)
        for t, past_mask in enumerate(reversed(activation_history)):
            temporal_weight = self.trace_decay ** t

            for i in range(NUM_ACTIVATIONS):
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
    ) -> jnp.ndarray:
        """Compute selection weights from eligibility."""
        # Base weight + scaled eligibility
        weights = self.base_weight + eligibility * self.credit_scale

        # Exploration bonus for low-eligibility functions
        for i in range(NUM_ACTIVATIONS):
            if eligibility[i] < 0.1:
                weights = weights.at[i].set(float(weights[i]) + self.exploration_bonus)

        return jnp.maximum(weights, 0.1)

    def _select_palette(
        self,
        weights: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette based on weights."""
        key1, key2 = jax.random.split(key)

        # Probabilistic selection with top-k bias
        probs = jax.nn.softmax(weights)

        # Top selections
        n_top = self.palette_size - 1
        top_indices = jnp.argsort(weights)[-n_top:]

        new_mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            new_mask = new_mask.at[int(idx)].set(1.0)

        # One probabilistic selection
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
        """Update with retrograde credit assignment."""
        key, k1 = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Apply eligibility decay
        new_eligibility = state['eligibility'] * self.eligibility_decay

        # Step 2: Backward credit propagation based on fitness change
        credit_assigned = jnp.zeros(NUM_ACTIVATIONS)
        penalty_assigned = jnp.zeros(NUM_ACTIVATIONS)
        propagation_event = 0

        if delta > 0:
            # Improvement: propagate credit backward
            credit = self.improvement_credit * delta
            new_eligibility, credit_assigned = self._propagate_credit_backward(
                new_eligibility,
                state['activation_history'],
                credit,
            )
            propagation_event = 1
        elif delta < 0:
            # Decrease: propagate penalty backward
            penalty = self.failure_penalty * abs(delta)
            new_eligibility, penalty_assigned = self._propagate_penalty_backward(
                new_eligibility,
                state['activation_history'],
                penalty,
            )
            propagation_event = 1

        # Step 3: Compute selection weights
        selection_weights = self._compute_selection_weights(new_eligibility)

        # Step 4: Select new palette
        new_mask = self._select_palette(selection_weights, k1)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update activation history
        activation_history = list(state['activation_history'])
        activation_history.append(new_mask.tolist())
        if len(activation_history) > self.history_length:
            activation_history = activation_history[-self.history_length:]

        # Update tracking
        new_total_credit = state['total_credit_received'] + credit_assigned
        new_total_penalty = state['total_penalty_received'] + penalty_assigned

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
            # Eligibility state
            'eligibility': new_eligibility,
            # History
            'activation_history': activation_history,
            'fitness_history': fitness_history,
            # Tracking
            'total_credit_received': new_total_credit,
            'total_penalty_received': new_total_penalty,
            'backward_propagation_events': state['backward_propagation_events'] + propagation_event,
            'previous_mask': state['mask'],
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Top by eligibility
        top_elig_idx = jnp.argsort(new_eligibility)[-5:][::-1]
        top_eligibility = [(int(i), float(new_eligibility[i])) for i in top_elig_idx]

        # Top credit receivers
        top_credit_idx = jnp.argsort(new_total_credit)[-3:][::-1]
        top_credit = [(int(i), float(new_total_credit[i])) for i in top_credit_idx]

        # Credit/penalty this generation
        credit_this_gen = [(int(i), float(credit_assigned[i]))
                          for i in range(NUM_ACTIVATIONS) if credit_assigned[i] > 0.01]
        penalty_this_gen = [(int(i), float(penalty_assigned[i]))
                           for i in range(NUM_ACTIVATIONS) if penalty_assigned[i] > 0.01]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'fitness_delta': delta,
            # Eligibility
            'mean_eligibility': float(jnp.mean(new_eligibility)),
            'max_eligibility': float(jnp.max(new_eligibility)),
            'top_eligibility': top_eligibility,
            # Credit assignment
            'credit_this_gen': credit_this_gen,
            'penalty_this_gen': penalty_this_gen,
            'top_total_credit': top_credit,
            # Propagation
            'propagation_events_total': state['backward_propagation_events'] + propagation_event,
            'history_length': len(activation_history),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_eligibility': float(new_eligibility[4]),
            'sin_total_credit': float(new_total_credit[4]),
            'sin_total_penalty': float(new_total_penalty[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with eligibility status."""
        palette = self.get_active_palette(state)
        eligibility = state['eligibility']
        total_credit = state['total_credit_received']
        total_penalty = state['total_penalty_received']

        # Top by eligibility
        top_elig = jnp.argsort(eligibility)[-5:][::-1]
        top_eligibility = [(int(i), float(eligibility[i])) for i in top_elig]

        # Net credit (credit - penalty)
        net_credit = total_credit - total_penalty
        top_net = jnp.argsort(net_credit)[-5:][::-1]
        top_net_credit = [(int(i), float(net_credit[i])) for i in top_net]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Eligibility
            'top_eligibility': top_eligibility,
            'mean_eligibility': float(jnp.mean(eligibility)),
            # Credit
            'top_net_credit': top_net_credit,
            'total_propagation_events': state['backward_propagation_events'],
            # Sin-specific
            'sin_eligibility': float(eligibility[4]),
            'sin_net_credit': float(net_credit[4]),
        }
