"""Strategy 27D: Quorum Sensing Dual (Population-Level Consensus for Both Domains).

Extends QuorumSensingStrategy to jointly evolve BOTH activation AND aggregation
function palettes using population-level voting and consensus mechanisms.

Cross-Domain Learning:
- Separate collective memory for activation and aggregation domains
- Cross-domain consensus: high activation quorum can support related aggregations
- Shared signal propagation with domain-specific thresholds

Key Dual Mechanisms:
1. Dual collective memory - separate population signals for both domains
2. Dual quorum thresholds - aggregations may need different consensus levels
3. Cross-domain stable promotion - functions supported by cross-domain success
4. Unified fitness weighting - same fitness affects both domain votes
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

# Aggregation domain constants
NUM_AGGREGATIONS = 6
DEFAULT_AGGREGATION_INDICES = [0, 1]  # sum, mean


class QuorumSensingDualStrategy(PaletteEvolutionStrategy):
    """Population-level consensus for dual activation+aggregation palettes.

    Functions in both domains accumulate "votes" from across the population.
    When a quorum is reached, functions are promoted to stable status.
    Cross-domain learning allows activation and aggregation discoveries
    to mutually reinforce each other.
    """

    name = "quorum_sensing_dual"
    description = "Population-level voting and consensus for both activation and aggregation"

    def __init__(
        self,
        # Activation quorum parameters
        act_quorum_threshold: float = 0.4,
        act_minority_threshold: float = 0.05,
        act_signal_decay: float = 0.85,
        # Aggregation quorum parameters
        agg_quorum_threshold: float = 0.35,  # Slightly lower for smaller domain
        agg_minority_threshold: float = 0.08,
        agg_signal_decay: float = 0.88,
        # Voting weights
        vote_weight_by_fitness: bool = True,
        fitness_weight_power: float = 2.0,
        # Cross-domain
        cross_learning_rate: float = 0.05,
        cross_influence: float = 0.15,  # How much cross-domain affects protection
        # Function states
        stable_promotion_gens: int = 5,
        unstable_after_gens: int = 10,
        # Mutation rates - activation
        act_stable_deactivate_rate: float = 0.01,
        act_normal_activate_rate: float = 0.12,
        act_normal_deactivate_rate: float = 0.06,
        # Mutation rates - aggregation
        agg_stable_deactivate_rate: float = 0.02,
        agg_normal_activate_rate: float = 0.10,
        agg_normal_deactivate_rate: float = 0.05,
        # Minority boost
        minority_activate_boost: float = 1.5,
        # Constraints
        min_active_act: int = 2,
        max_active_act: int = 6,
        min_active_agg: int = 1,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Quorum Sensing Dual strategy."""
        # Activation quorum
        self.act_quorum_threshold = act_quorum_threshold
        self.act_minority_threshold = act_minority_threshold
        self.act_signal_decay = act_signal_decay

        # Aggregation quorum
        self.agg_quorum_threshold = agg_quorum_threshold
        self.agg_minority_threshold = agg_minority_threshold
        self.agg_signal_decay = agg_signal_decay

        # Voting
        self.vote_weight_by_fitness = vote_weight_by_fitness
        self.fitness_weight_power = fitness_weight_power

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence

        # Stability
        self.stable_promotion_gens = stable_promotion_gens
        self.unstable_after_gens = unstable_after_gens

        # Mutation - activation
        self.act_stable_deactivate_rate = act_stable_deactivate_rate
        self.act_normal_activate_rate = act_normal_activate_rate
        self.act_normal_deactivate_rate = act_normal_deactivate_rate

        # Mutation - aggregation
        self.agg_stable_deactivate_rate = agg_stable_deactivate_rate
        self.agg_normal_activate_rate = agg_normal_activate_rate
        self.agg_normal_deactivate_rate = agg_normal_deactivate_rate

        # Minority
        self.minority_activate_boost = minority_activate_boost

        # Constraints
        self.min_active_act = min_active_act
        self.max_active_act = max_active_act
        self.min_active_agg = min_active_agg
        self.max_active_agg = max_active_agg

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual collective memories."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        act_collective = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_collective = act_collective.at[i].set(0.3)

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_mask = agg_mask.at[i].set(1.0)

        agg_collective = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_collective = agg_collective.at[i].set(0.3)

        # Cross-domain affinity matrix (which act-agg combinations work)
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_collective': act_collective,
            'act_stable': [],
            'act_above_quorum': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'act_below_quorum': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_collective': agg_collective,
            'agg_stable': [],
            'agg_above_quorum': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
            'agg_below_quorum': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
            # Cross-domain
            'cross_affinity': cross_affinity,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 272727),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'fitness_ema': 0.5,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return [i for i in range(NUM_AGGREGATIONS) if state['agg_mask'][i] > 0.5]

    def _compute_population_signal(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness: float,
        best_fitness: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute population signals for both domains."""
        # Weight by fitness
        if self.vote_weight_by_fitness and best_fitness > 0.01:
            relative_fitness = fitness / best_fitness
            vote_weight = relative_fitness ** self.fitness_weight_power
        else:
            vote_weight = 1.0

        # Activation signal
        active_act = (act_mask > 0.5).astype(jnp.float32)
        n_active_act = max(jnp.sum(active_act), 1.0)
        act_signal = active_act * vote_weight / n_active_act

        # Aggregation signal
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        n_active_agg = max(jnp.sum(active_agg), 1.0)
        agg_signal = active_agg * vote_weight / n_active_agg

        return act_signal, agg_signal

    def _update_collective_memory(
        self,
        act_collective: jnp.ndarray,
        agg_collective: jnp.ndarray,
        act_signal: jnp.ndarray,
        agg_signal: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update collective memories for both domains."""
        new_act = (
            self.act_signal_decay * act_collective +
            (1 - self.act_signal_decay) * act_signal
        )
        new_agg = (
            self.agg_signal_decay * agg_collective +
            (1 - self.agg_signal_decay) * agg_signal
        )
        return jnp.clip(new_act, 0.0, 1.0), jnp.clip(new_agg, 0.0, 1.0)

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

    def _update_quorum_counts(
        self,
        collective: jnp.ndarray,
        above_count: jnp.ndarray,
        below_count: jnp.ndarray,
        stable_list: List[int],
        quorum_threshold: float,
        n_functions: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[int]]:
        """Update quorum status counters for a domain."""
        new_above = above_count.copy()
        new_below = below_count.copy()
        new_stable = list(stable_list)

        for i in range(n_functions):
            signal = float(collective[i])

            if signal >= quorum_threshold:
                new_above = new_above.at[i].set(int(above_count[i]) + 1)
                new_below = new_below.at[i].set(0)

                if int(new_above[i]) >= self.stable_promotion_gens:
                    if i not in new_stable:
                        new_stable.append(i)
            else:
                new_below = new_below.at[i].set(int(below_count[i]) + 1)
                new_above = new_above.at[i].set(0)

                if i in new_stable:
                    if int(new_below[i]) >= self.unstable_after_gens:
                        new_stable.remove(i)

        return new_above, new_below, new_stable

    def _apply_quorum_mutation_act(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        collective: jnp.ndarray,
        stable_list: List[int],
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply quorum-based mutation to activation palette."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        # Cross-domain support (how much active aggs support each activation)
        cross_support = jnp.dot(cross_affinity, agg_mask > 0.5) / max(jnp.sum(agg_mask > 0.5), 1)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_ACTIVATIONS):
            signal = float(collective[i])
            support = float(cross_support[i])
            effective_signal = signal + self.cross_influence * support

            is_stable = i in stable_list
            at_quorum = effective_signal >= self.act_quorum_threshold
            at_minority = signal <= self.act_minority_threshold

            if mask[i] < 0.5:
                # Inactive: maybe activate
                if is_stable or at_quorum:
                    rate = self.act_normal_activate_rate * (1.0 + effective_signal)
                elif at_minority:
                    rate = self.act_normal_activate_rate * self.minority_activate_boost * 0.5
                else:
                    rate = self.act_normal_activate_rate * (0.3 + 0.7 * effective_signal)

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                if is_stable:
                    rate = self.act_stable_deactivate_rate
                elif at_quorum:
                    rate = self.act_normal_deactivate_rate * 0.2
                else:
                    rate = self.act_normal_deactivate_rate * (1.0 - effective_signal * 0.7)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        n_active = int(jnp.sum(new_mask > 0.5))
        if n_active < self.min_active_act or n_active > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'act_activated': activated, 'act_deactivated': deactivated}

    def _apply_quorum_mutation_agg(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        collective: jnp.ndarray,
        stable_list: List[int],
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply quorum-based mutation to aggregation palette."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (NUM_AGGREGATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_AGGREGATIONS,))

        # Cross-domain support from activations
        cross_support = jnp.dot(cross_affinity.T, act_mask > 0.5) / max(jnp.sum(act_mask > 0.5), 1)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_AGGREGATIONS):
            signal = float(collective[i])
            support = float(cross_support[i])
            effective_signal = signal + self.cross_influence * support

            is_stable = i in stable_list
            at_quorum = effective_signal >= self.agg_quorum_threshold
            at_minority = signal <= self.agg_minority_threshold

            if mask[i] < 0.5:
                if is_stable or at_quorum:
                    rate = self.agg_normal_activate_rate * (1.0 + effective_signal)
                elif at_minority:
                    rate = self.agg_normal_activate_rate * self.minority_activate_boost * 0.5
                else:
                    rate = self.agg_normal_activate_rate * (0.3 + 0.7 * effective_signal)

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if is_stable:
                    rate = self.agg_stable_deactivate_rate
                elif at_quorum:
                    rate = self.agg_normal_deactivate_rate * 0.2
                else:
                    rate = self.agg_normal_deactivate_rate * (1.0 - effective_signal * 0.7)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        n_active = int(jnp.sum(new_mask > 0.5))
        if n_active < self.min_active_agg or n_active > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'agg_activated': activated, 'agg_deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual quorum sensing dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Compute population signals for both domains
        act_signal, agg_signal = self._compute_population_signal(
            state['act_mask'],
            state['agg_mask'],
            best_fitness,
            new_best,
        )

        # Step 2: Update collective memories
        new_act_collective, new_agg_collective = self._update_collective_memory(
            state['act_collective'],
            state['agg_collective'],
            act_signal,
            agg_signal,
        )

        # Step 3: Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Step 4: Update quorum counts for both domains
        new_act_above, new_act_below, new_act_stable = self._update_quorum_counts(
            new_act_collective,
            state['act_above_quorum'],
            state['act_below_quorum'],
            state['act_stable'],
            self.act_quorum_threshold,
            NUM_ACTIVATIONS,
        )

        new_agg_above, new_agg_below, new_agg_stable = self._update_quorum_counts(
            new_agg_collective,
            state['agg_above_quorum'],
            state['agg_below_quorum'],
            state['agg_stable'],
            self.agg_quorum_threshold,
            NUM_AGGREGATIONS,
        )

        # Step 5: Apply mutations to both domains
        new_act_mask, act_mutation = self._apply_quorum_mutation_act(
            k1, state['act_mask'], new_act_collective, new_act_stable,
            new_cross, state['agg_mask']
        )
        new_agg_mask, agg_mutation = self._apply_quorum_mutation_agg(
            k2, state['agg_mask'], new_agg_collective, new_agg_stable,
            new_cross, state['act_mask']
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Update fitness EMA
        new_fitness_ema = 0.9 * state['fitness_ema'] + 0.1 * best_fitness

        new_state = {
            # Activation domain
            'act_mask': new_act_mask,
            'act_collective': new_act_collective,
            'act_stable': new_act_stable,
            'act_above_quorum': new_act_above,
            'act_below_quorum': new_act_below,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_collective': new_agg_collective,
            'agg_stable': new_agg_stable,
            'agg_above_quorum': new_agg_above,
            'agg_below_quorum': new_agg_below,
            # Cross-domain
            'cross_affinity': new_cross,
            # General state
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'fitness_ema': new_fitness_ema,
            'strategy_name': self.name,
            'fitness_history': (state['fitness_history'] + [best_fitness])[-20:],
        }

        # Compute metrics
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = self.get_active_agg_palette(new_state)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Activation quorum
            'act_n_stable': len(new_act_stable),
            'act_stable_functions': new_act_stable,
            'act_mean_collective': float(jnp.mean(new_act_collective)),
            # Aggregation quorum
            'agg_n_stable': len(new_agg_stable),
            'agg_stable_functions': new_agg_stable,
            'agg_mean_collective': float(jnp.mean(new_agg_collective)),
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_collective': float(new_act_collective[4]),
            'sin_stable': 4 in new_act_stable,
            # Agg4 status
            'has_agg4': len(agg_palette) >= 4,
        }
        metrics.update(act_mutation)
        metrics.update(agg_mutation)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual quorum status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Activation quorum
            'act_stable': state['act_stable'],
            'act_mean_collective': float(jnp.mean(state['act_collective'])),
            # Aggregation quorum
            'agg_stable': state['agg_stable'],
            'agg_mean_collective': float(jnp.mean(state['agg_collective'])),
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
            # Sin-specific
            'sin_collective': float(state['act_collective'][4]),
            'sin_stable': 4 in state['act_stable'],
        }
