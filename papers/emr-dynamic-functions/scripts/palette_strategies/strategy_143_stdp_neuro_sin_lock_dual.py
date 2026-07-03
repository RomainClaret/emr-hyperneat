"""Strategy 143: STDP-Neuro-Sin-Lock Dual.

Combines the best working mechanism (STDP-Neurogenesis survival) with
the Sin-Extreme-Lock mechanism for coexistence-based protection.

Key Combination:
- From #117: STDP temporal credit for survival, neurogenesis, maturation periods
- From #139: Coexistence locking where sin+extreme pairs become locked

Rationale: STDP-Neurogenesis-Survival achieved 100% sin retention in CL benchmark.
Adding sin-extreme locking provides explicit protection for the sin+extreme pairs.

Biological Inspiration: Combining spike-timing dependent plasticity with
epigenetic locking creates robust memory of successful function combinations.
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


class STDPNeuroSinLockDualStrategy(PaletteEvolutionStrategy):
    """STDP-Neurogenesis with sin-extreme locking.

    Hybrid of best working mechanism with explicit
    sin-extreme protection through coexistence locking.

    Critical innovation: Combines temporal learning (STDP) with spatial
    locking (sin-extreme pairs) for robust retention.
    """

    name = "stdp_neuro_sin_lock_dual"
    description = "Dual: STDP-Neurogenesis + Sin-Extreme coexistence locking"

    def __init__(
        self,
        # === STDP PARAMETERS (FROM #117) ===
        stdp_ltp_window: int = 5,
        stdp_ltd_window: int = 8,
        stdp_learning_rate: float = 0.15,
        stdp_survival_multiplier: float = 0.7,
        # === NEUROGENESIS PARAMETERS (FROM #117) ===
        neurogenesis_rate: float = 0.10,
        maturation_period: int = 8,
        immature_protection: float = 0.3,
        # === LOCK PARAMETERS (FROM #139) ===
        coexistence_lock_threshold: int = 5,
        lock_protection_strength: float = 0.9,
        lock_pairs: List[Tuple[int, int]] = None,
        # === GUARANTEED INITIAL STATE ===
        sin_always_initial: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        # === PRUNING PARAMETERS ===
        prune_threshold_act: float = 0.2,
        prune_threshold_agg: float = 0.25,
        stagnation_prune_after: int = 6,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === INITIAL PALETTES ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize STDP-Neuro-Sin-Lock strategy."""
        # STDP parameters
        self.stdp_ltp_window = stdp_ltp_window
        self.stdp_ltd_window = stdp_ltd_window
        self.stdp_learning_rate = stdp_learning_rate
        self.stdp_survival_multiplier = stdp_survival_multiplier

        # Neurogenesis parameters
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.immature_protection = immature_protection

        # Lock parameters
        self.coexistence_lock_threshold = coexistence_lock_threshold
        self.lock_protection_strength = lock_protection_strength
        self.lock_pairs = lock_pairs or [(4, 2), (4, 3)]

        # Guaranteed state
        self.sin_always_initial = sin_always_initial
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Pruning
        self.prune_threshold_act = prune_threshold_act
        self.prune_threshold_agg = prune_threshold_agg
        self.stagnation_prune_after = stagnation_prune_after

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial palettes
        default_act = list(DEFAULT_PALETTE_INDICES)
        if self.sin_always_initial and 4 not in default_act:
            default_act.append(4)

        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        if self.extreme_always_initial:
            for agg in CORE_EXTREME_AGGS:
                if agg not in default_agg:
                    default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with STDP and lock tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # CRITICAL: Ensure sin and extreme in initial palette
        if self.sin_always_initial and 4 not in initial_act:
            initial_act = list(initial_act) + [4]

        initial_agg = list(initial_agg)
        if self.extreme_always_initial:
            for agg in CORE_EXTREME_AGGS:
                if agg not in initial_agg:
                    initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        act_affinities = act_affinities.at[4].set(0.8)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for agg in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[agg].set(0.75)

        # STDP traces
        act_trace = jnp.zeros(NUM_ACTIVATIONS)
        agg_trace = jnp.zeros(NUM_AGGREGATIONS)

        # Maturation tracking (generations since added)
        act_maturity = jnp.zeros(NUM_ACTIVATIONS)
        agg_maturity = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_maturity = act_maturity.at[i].set(self.maturation_period)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_maturity = agg_maturity.at[i].set(self.maturation_period)

        # Lock tracking
        coexistence_count = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        lock_status = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        # Initialize coexistence for pairs in initial palette
        for act_idx, agg_idx in self.lock_pairs:
            if 4 in initial_act and agg_idx in initial_agg:
                coexistence_count = coexistence_count.at[act_idx, agg_idx].set(
                    float(self.coexistence_lock_threshold)
                )
                lock_status = lock_status.at[act_idx, agg_idx].set(1.0)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # STDP traces
            'act_trace': act_trace,
            'agg_trace': agg_trace,
            # Maturity
            'act_maturity': act_maturity,
            'agg_maturity': agg_maturity,
            # Lock tracking
            'coexistence_count': coexistence_count,
            'lock_status': lock_status,
            # Fitness history for STDP
            'fitness_deltas': [],
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1430000),
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

    def _update_stdp_traces(
        self,
        act_trace: jnp.ndarray,
        agg_trace: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update STDP eligibility traces."""
        # Decay existing traces
        decay = 0.8
        new_act_trace = act_trace * decay
        new_agg_trace = agg_trace * decay

        # Add trace for active functions
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                new_act_trace = new_act_trace.at[i].set(new_act_trace[i] + 0.2)

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                new_agg_trace = new_agg_trace.at[j].set(new_agg_trace[j] + 0.2)

        # STDP-like modulation based on fitness timing
        if fitness_delta > 0:  # LTP - strengthen
            new_act_trace = new_act_trace * (1 + self.stdp_learning_rate)
            new_agg_trace = new_agg_trace * (1 + self.stdp_learning_rate)
        elif fitness_delta < -0.01:  # LTD - weaken
            new_act_trace = new_act_trace * (1 - self.stdp_learning_rate * 0.5)
            new_agg_trace = new_agg_trace * (1 - self.stdp_learning_rate * 0.5)

        return jnp.clip(new_act_trace, 0, 1), jnp.clip(new_agg_trace, 0, 1)

    def _update_coexistence(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        coexistence_count: jnp.ndarray,
        lock_status: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update coexistence counters and form locks."""
        new_coexistence = coexistence_count.copy()
        new_lock = lock_status.copy()

        for act_idx, agg_idx in self.lock_pairs:
            both_active = act_mask[act_idx] > 0.5 and agg_mask[agg_idx] > 0.5

            if both_active:
                new_coexistence = new_coexistence.at[act_idx, agg_idx].set(
                    new_coexistence[act_idx, agg_idx] + 1
                )

                if (new_coexistence[act_idx, agg_idx] >= self.coexistence_lock_threshold
                        and new_lock[act_idx, agg_idx] < 0.5):
                    new_lock = new_lock.at[act_idx, agg_idx].set(1.0)
            else:
                new_coexistence = new_coexistence.at[act_idx, agg_idx].set(0)

        return new_coexistence, new_lock

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update palette using STDP + locking."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        act_trace = state['act_trace']
        agg_trace = state['agg_trace']
        act_maturity = state['act_maturity']
        agg_maturity = state['agg_maturity']
        coexistence_count = state['coexistence_count']
        lock_status = state['lock_status']
        rng_key = state['rng_key']

        fitness_delta = best_fitness - prev_best_fitness
        improved = fitness_delta > 0.001
        stagnated = fitness_delta < 0.0001

        # Update stagnation
        stagnation_count = state['stagnation_count']
        if stagnated:
            stagnation_count += 1
        else:
            stagnation_count = max(0, stagnation_count - 1)

        # Update STDP traces
        act_trace, agg_trace = self._update_stdp_traces(
            act_trace, agg_trace, act_mask, agg_mask, fitness_delta
        )

        # Update maturity
        new_act_maturity = act_maturity.copy()
        new_agg_maturity = agg_maturity.copy()
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                new_act_maturity = new_act_maturity.at[i].set(
                    min(self.maturation_period, new_act_maturity[i] + 1)
                )
        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                new_agg_maturity = new_agg_maturity.at[j].set(
                    min(self.maturation_period, new_agg_maturity[j] + 1)
                )

        # Update locks
        coexistence_count, lock_status = self._update_coexistence(
            act_mask, agg_mask, coexistence_count, lock_status
        )

        # Calculate protection
        act_protection = jnp.zeros(NUM_ACTIVATIONS)
        agg_protection = jnp.zeros(NUM_AGGREGATIONS)

        for act_idx, agg_idx in self.lock_pairs:
            if lock_status[act_idx, agg_idx] > 0.5:
                act_protection = act_protection.at[act_idx].set(self.lock_protection_strength)
                agg_protection = agg_protection.at[agg_idx].set(self.lock_protection_strength)

        # STDP survival modulation
        act_protection = act_protection + act_trace * self.stdp_survival_multiplier
        agg_protection = agg_protection + agg_trace * self.stdp_survival_multiplier

        # Maturity protection
        for i in range(NUM_ACTIVATIONS):
            if new_act_maturity[i] < self.maturation_period:
                act_protection = act_protection.at[i].set(
                    max(float(act_protection[i]), self.immature_protection)
                )
        for j in range(NUM_AGGREGATIONS):
            if new_agg_maturity[j] < self.maturation_period:
                agg_protection = agg_protection.at[j].set(
                    max(float(agg_protection[j]), self.immature_protection)
                )

        # Update affinities with STDP modulation
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    bonus = 1.0 + float(act_trace[i]) * 0.5
                    if i == 4:
                        bonus *= 1.5
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * bonus)
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    bonus = 1.0 + float(agg_trace[j]) * 0.5
                    if j in CORE_EXTREME_AGGS:
                        bonus *= 1.4
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * bonus)
                    )

        # Affinity floors
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Pruning with STDP and lock protection
        new_act_mask = act_mask.copy()
        new_agg_mask = agg_mask.copy()

        rng_key, prune_key = jax.random.split(rng_key)
        prune_rand = jax.random.uniform(prune_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        if stagnation_count > self.stagnation_prune_after:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] > 0.5 and i != 4:
                    if new_act_aff[i] < self.prune_threshold_act:
                        if prune_rand[i] > act_protection[i]:
                            new_act_mask = new_act_mask.at[i].set(0.0)

            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5 and j not in CORE_EXTREME_AGGS:
                    if new_agg_aff[j] < self.prune_threshold_agg:
                        idx = NUM_ACTIVATIONS + j
                        if prune_rand[idx] > agg_protection[j]:
                            new_agg_mask = new_agg_mask.at[j].set(0.0)

        # CRITICAL: Ensure sin and extreme aggs are never removed
        new_act_mask = new_act_mask.at[4].set(1.0)
        for agg in CORE_EXTREME_AGGS:
            new_agg_mask = new_agg_mask.at[agg].set(1.0)

        # Neurogenesis
        rng_key, neuro_key = jax.random.split(rng_key)
        neuro_rand = jax.random.uniform(neuro_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        act_count = int(jnp.sum(new_act_mask))
        agg_count = int(jnp.sum(new_agg_mask))

        if act_count < self.max_active_act:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] < 0.5:
                    if neuro_rand[i] < self.neurogenesis_rate:
                        new_act_mask = new_act_mask.at[i].set(1.0)
                        new_act_maturity = new_act_maturity.at[i].set(0)
                        break

        if agg_count < self.max_active_agg:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] < 0.5:
                    idx = NUM_ACTIVATIONS + j
                    if neuro_rand[idx] < self.neurogenesis_rate:
                        new_agg_mask = new_agg_mask.at[j].set(1.0)
                        new_agg_maturity = new_agg_maturity.at[j].set(0)
                        break

        # Ensure minimums
        if int(jnp.sum(new_act_mask)) < self.min_active_act:
            new_act_mask = new_act_mask.at[0].set(1.0)
        if int(jnp.sum(new_agg_mask)) < self.min_active_agg:
            new_agg_mask = new_agg_mask.at[0].set(1.0)

        new_state = {
            **state,
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_trace': act_trace,
            'agg_trace': agg_trace,
            'act_maturity': new_act_maturity,
            'agg_maturity': new_agg_maturity,
            'coexistence_count': coexistence_count,
            'lock_status': lock_status,
            'rng_key': rng_key,
            'generation': generation + 1,
            'stagnation_count': stagnation_count,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        locked_pairs = sum(1 for a, j in self.lock_pairs if lock_status[a, j] > 0.5)
        metrics = {
            'locked_pairs': locked_pairs,
            'avg_act_trace': float(act_trace.mean()),
            'avg_agg_trace': float(agg_trace.mean()),
        }

        return new_state, metrics
