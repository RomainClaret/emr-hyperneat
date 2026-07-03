"""Strategy 132: Predator-Prey-Clonal Cycles with Protection Dual.

Combines Predator-Prey dynamics (#30) with Clonal Selection protection (#91).
Boom-bust cycles with clonal capture protection during bust.

Key Innovation:
- Boom phase: aggressive exploration, high mutation rates
- Bust phase: selective pressure, functions must prove worth
- Clonal capture during boom protects functions during bust
- Sin and extremes have clonal capture priority

Biological basis: In ecology, predator-prey cycles create boom-bust dynamics.
In immune systems, successful clones are protected even during downturn.
Combining these: aggressive exploration with protection for winners.

Expected: Dynamic exploration with protection for valuable discoveries.
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
    CORE_EXTREME_AGGS,
)


class PredatorClonalDualStrategy(PaletteEvolutionStrategy):
    """Predator-prey-clonal cycles strategy for dual palette evolution.

    Boom-bust cycles with clonal protection.
    Aggressive exploration during boom, protection for winners during bust.

    Critical innovation: Dynamic cycles with selective protection.
    """

    name = "predator_clonal_dual"
    description = "Dual: Boom-bust cycles with clonal capture protection"

    def __init__(
        self,
        # === Cycle parameters ===
        cycle_period: int = 20,
        boom_fraction: float = 0.6,
        # === Boom parameters ===
        boom_mutation_rate: float = 0.18,
        boom_capture_threshold: float = 0.7,
        # === Bust parameters ===
        bust_mutation_rate: float = 0.04,
        bust_protection_strength: float = 0.9,
        bust_survival_threshold: float = 0.3,
        # === Clonal capture ===
        capture_affinity_threshold: float = 0.6,
        capture_decay: float = 0.98,
        # === Sin and extreme priority ===
        sin_idx: int = 4,
        sin_capture_bonus: float = 0.2,
        extreme_capture_bonus: float = 0.25,
        # === General parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Predator-Clonal strategy."""
        # Cycle
        self.cycle_period = cycle_period
        self.boom_fraction = boom_fraction

        # Boom
        self.boom_mutation_rate = boom_mutation_rate
        self.boom_capture_threshold = boom_capture_threshold

        # Bust
        self.bust_mutation_rate = bust_mutation_rate
        self.bust_protection_strength = bust_protection_strength
        self.bust_survival_threshold = bust_survival_threshold

        # Capture
        self.capture_affinity_threshold = capture_affinity_threshold
        self.capture_decay = capture_decay

        # Sin and extreme
        self.sin_idx = sin_idx
        self.sin_capture_bonus = sin_capture_bonus
        self.extreme_capture_bonus = extreme_capture_bonus

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with cycle and capture tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)

        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(agg_affinities[i] + 0.2)

        # Clonal capture protection
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Capture
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            # Cycle tracking
            'cycle_phase': 'boom',
            'phase_generation': 0,
            'total_captures': 0,
            'bust_survivals': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1320000),
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

    def _get_cycle_phase(self, generation: int) -> str:
        """Determine current cycle phase."""
        cycle_position = generation % self.cycle_period
        boom_duration = int(self.cycle_period * self.boom_fraction)
        return 'boom' if cycle_position < boom_duration else 'bust'

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with boom-bust cycle and clonal protection."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        cycle_phase = self._get_cycle_phase(generation)
        prev_phase = state['cycle_phase']
        phase_generation = state['phase_generation']

        # Track phase transitions
        if cycle_phase != prev_phase:
            phase_generation = 0
        else:
            phase_generation += 1

        # === UPDATE CAPTURE STATUS ===
        act_captured = state['act_captured'] * self.capture_decay
        agg_captured = state['agg_captured'] * self.capture_decay

        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        total_captures = state['total_captures']
        bust_survivals = state['bust_survivals']

        # Update affinities first
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)

        # === BOOM PHASE: Aggressive exploration + capture ===
        if cycle_phase == 'boom':
            mutation_rate = self.boom_mutation_rate

            # Capture high-affinity functions during boom
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    capture_thresh = self.capture_affinity_threshold
                    if i == self.sin_idx:
                        capture_thresh -= self.sin_capture_bonus
                    if float(act_affinities[i]) >= capture_thresh:
                        if float(act_captured[i]) < 0.5:
                            act_captured = act_captured.at[i].set(1.0)
                            total_captures += 1

            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    capture_thresh = self.capture_affinity_threshold
                    if i in CORE_EXTREME_AGGS:
                        capture_thresh -= self.extreme_capture_bonus
                    if float(agg_affinities[i]) >= capture_thresh:
                        if float(agg_captured[i]) < 0.5:
                            agg_captured = agg_captured.at[i].set(1.0)
                            total_captures += 1

        # === BUST PHASE: Low mutation, capture protection ===
        else:
            mutation_rate = self.bust_mutation_rate

            # During bust, captured functions are protected
            # Non-captured active functions face survival pressure
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5 and float(act_captured[i]) > 0.5:
                    # Captured = protected
                    bust_survivals += 1

            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5 and float(agg_captured[i]) > 0.5:
                    bust_survivals += 1

        # === ACTIVATION MUTATION ===
        candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k1) < mutation_rate:
            # During boom: explore widely
            # During bust: only protected or high-affinity
            weights = []
            for i in candidates:
                w = 0.1
                if cycle_phase == 'boom':
                    w += 0.3  # Boom = wide exploration
                    if i == self.sin_idx:
                        w += 0.3
                else:
                    # Bust = only if previously captured or high affinity
                    w += float(act_captured[i]) * self.bust_protection_strength
                    w += float(act_affinities[i]) * 0.2
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs))
            act_mask = act_mask.at[new_idx].set(1.0)

        # === AGGREGATION MUTATION ===
        candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k3) < mutation_rate:
            weights = []
            for i in candidates:
                w = 0.1
                if cycle_phase == 'boom':
                    w += 0.3
                    if i in CORE_EXTREME_AGGS:
                        w += 0.3
                else:
                    w += float(agg_captured[i]) * self.bust_protection_strength
                    w += float(agg_affinities[i]) * 0.2
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs))
            agg_mask = agg_mask.at[new_idx].set(1.0)

        # Clamp affinities
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)

        # Ensure minimum diversity
        if sum(float(act_mask[i]) for i in range(NUM_ACTIVATIONS)) < self.min_active_act:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k1, jnp.array(candidates)))
                act_mask = act_mask.at[new_idx].set(1.0)

        if sum(float(agg_mask[i]) for i in range(NUM_AGGREGATIONS)) < self.min_active_agg:
            candidates = [i for i in CORE_EXTREME_AGGS if float(agg_mask[i]) < 0.5]
            if not candidates:
                candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k3, jnp.array(candidates)))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        new_state = {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'cycle_phase': cycle_phase,
            'phase_generation': phase_generation,
            'total_captures': total_captures,
            'bust_survivals': bust_survivals,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'cycle_phase': cycle_phase,
            'phase_generation': phase_generation,
            'total_captures': total_captures,
            'bust_survivals': bust_survivals,
            'n_act_captured': int(sum(1 for i in range(NUM_ACTIVATIONS) if float(act_captured[i]) > 0.5)),
            'n_agg_captured': int(sum(1 for i in range(NUM_AGGREGATIONS) if float(agg_captured[i]) > 0.5)),
        }

        return new_state, metrics
