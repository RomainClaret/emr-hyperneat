"""Strategy 139: Sin-Extreme-Lock Dual.

Implements coexistence locking: once sin AND max (or min) co-exist for N
generations, both become "locked" - highly resistant to removal.

Key Mechanism:
- Track generations of sin+extreme coexistence
- After 5 generations of coexistence, pairs become "locked"
- Locked pairs have 0.9 protection strength
- Unlock requires 20% fitness drop (significant regression)

Biological Inspiration: Epigenetic locking in gene expression where stable
co-expression patterns become resistant to change through chromatin modification.
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


class SinExtremeLockDualStrategy(PaletteEvolutionStrategy):
    """Sin-Extreme lock with coexistence-based protection.

    Once sin and extreme aggs coexist for multiple
    generations, they become locked together - highly resistant to removal.

    Critical innovation: Temporal coexistence tracking with lock formation.
    """

    name = "sin_extreme_lock_dual"
    description = "Dual: Coexistence locking for sin and extreme aggs"

    def __init__(
        self,
        # === LOCK PARAMETERS (CORE) ===
        coexistence_lock_threshold: int = 5,  # Gens of coexistence to lock
        lock_protection_strength: float = 0.9,
        lock_pairs: List[Tuple[int, int]] = None,  # (sin, max), (sin, min)
        unlock_fitness_drop_required: float = 0.2,  # 20% fitness drop to unlock
        # === GUARANTEED INITIAL STATE ===
        sin_always_initial: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        # === EXPLORATION PARAMETERS ===
        exploration_rate_act: float = 0.08,
        exploration_rate_agg: float = 0.10,
        lock_exploration_boost: float = 0.3,  # Locked pairs boost exploration
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
        """Initialize Sin-Extreme-Lock strategy."""
        # Lock parameters
        self.coexistence_lock_threshold = coexistence_lock_threshold
        self.lock_protection_strength = lock_protection_strength
        self.lock_pairs = lock_pairs or [(4, 2), (4, 3)]  # (sin, max), (sin, min)
        self.unlock_fitness_drop_required = unlock_fitness_drop_required

        # Guaranteed state
        self.sin_always_initial = sin_always_initial
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Exploration
        self.exploration_rate_act = exploration_rate_act
        self.exploration_rate_agg = exploration_rate_agg
        self.lock_exploration_boost = lock_exploration_boost

        # Pruning
        self.prune_threshold_act = prune_threshold_act
        self.prune_threshold_agg = prune_threshold_agg
        self.stagnation_prune_after = stagnation_prune_after

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial palettes - ALWAYS include sin and extreme aggs
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
        """Initialize state with lock tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # CRITICAL: Ensure sin is ALWAYS in initial palette
        if self.sin_always_initial and 4 not in initial_act:
            initial_act = list(initial_act) + [4]

        # Ensure extreme aggs in initial palette
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
        act_affinities = act_affinities.at[4].set(0.8)  # Sin high

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for agg in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[agg].set(0.75)

        # Lock tracking
        # Coexistence counter: for each (act, agg) pair, count consecutive coexistence gens
        coexistence_count = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        # Lock status: 1.0 if locked, 0.0 if not
        lock_status = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        # Initialize coexistence count for pairs present in initial palette
        for act_idx, agg_idx in self.lock_pairs:
            if 4 in initial_act and agg_idx in initial_agg:
                # Start with threshold count since they're in initial palette
                coexistence_count = coexistence_count.at[act_idx, agg_idx].set(
                    float(self.coexistence_lock_threshold)
                )
                # Already locked!
                lock_status = lock_status.at[act_idx, agg_idx].set(1.0)

        # Peak fitness for unlock tracking
        peak_fitness = 0.0

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, agg].set(0.7)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Lock tracking (CORE)
            'coexistence_count': coexistence_count,
            'lock_status': lock_status,
            'peak_fitness': peak_fitness,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'lock_events': 0,
            'unlock_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1390000),
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

    def _update_coexistence(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        coexistence_count: jnp.ndarray,
        lock_status: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Update coexistence counters and form new locks."""
        new_coexistence = coexistence_count.copy()
        new_lock = lock_status.copy()
        new_lock_events = 0

        for act_idx, agg_idx in self.lock_pairs:
            both_active = act_mask[act_idx] > 0.5 and agg_mask[agg_idx] > 0.5

            if both_active:
                # Increment coexistence counter
                new_coexistence = new_coexistence.at[act_idx, agg_idx].set(
                    new_coexistence[act_idx, agg_idx] + 1
                )

                # Check for lock formation
                if (new_coexistence[act_idx, agg_idx] >= self.coexistence_lock_threshold
                        and new_lock[act_idx, agg_idx] < 0.5):
                    new_lock = new_lock.at[act_idx, agg_idx].set(1.0)
                    new_lock_events += 1
            else:
                # Reset coexistence counter (but don't unlock yet)
                new_coexistence = new_coexistence.at[act_idx, agg_idx].set(0)

        return new_coexistence, new_lock, new_lock_events

    def _check_unlock(
        self,
        lock_status: jnp.ndarray,
        current_fitness: float,
        peak_fitness: float,
    ) -> Tuple[jnp.ndarray, int]:
        """Check if any locks should be released due to fitness drop."""
        new_lock = lock_status.copy()
        unlock_events = 0

        # Only unlock if there's been a significant fitness drop
        if peak_fitness > 0.01:
            fitness_drop = (peak_fitness - current_fitness) / peak_fitness
            if fitness_drop >= self.unlock_fitness_drop_required:
                # Unlock all pairs (rare event)
                for act_idx, agg_idx in self.lock_pairs:
                    if new_lock[act_idx, agg_idx] > 0.5:
                        new_lock = new_lock.at[act_idx, agg_idx].set(0.0)
                        unlock_events += 1

        return new_lock, unlock_events

    def _calculate_protection(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        lock_status: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate protection based on lock status."""
        act_protection = jnp.zeros(NUM_ACTIVATIONS)
        agg_protection = jnp.zeros(NUM_AGGREGATIONS)

        for act_idx, agg_idx in self.lock_pairs:
            if lock_status[act_idx, agg_idx] > 0.5:
                act_protection = act_protection.at[act_idx].set(self.lock_protection_strength)
                agg_protection = agg_protection.at[agg_idx].set(self.lock_protection_strength)

        # CRITICAL: Sin always has base protection
        act_protection = act_protection.at[4].set(max(0.7, float(act_protection[4])))
        # Extreme aggs always have base protection
        for agg in CORE_EXTREME_AGGS:
            agg_protection = agg_protection.at[agg].set(max(0.6, float(agg_protection[agg])))

        return act_protection, agg_protection

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update palette using coexistence locking."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        coexistence_count = state['coexistence_count']
        lock_status = state['lock_status']
        peak_fitness = state['peak_fitness']
        cross_affinity = state['cross_affinity']
        rng_key = state['rng_key']

        fitness_delta = best_fitness - prev_best_fitness
        improved = fitness_delta > 0.001
        stagnated = fitness_delta < 0.0001

        # Update peak fitness
        new_peak = max(peak_fitness, best_fitness)

        # Update stagnation
        stagnation_count = state['stagnation_count']
        if stagnated:
            stagnation_count += 1
        else:
            stagnation_count = max(0, stagnation_count - 1)

        # Update coexistence and locks
        coexistence_count, lock_status, lock_events = self._update_coexistence(
            act_mask, agg_mask, coexistence_count, lock_status
        )

        # Check for unlock
        lock_status, unlock_events = self._check_unlock(
            lock_status, best_fitness, peak_fitness
        )

        # Calculate protection
        act_protection, agg_protection = self._calculate_protection(
            act_mask, agg_mask, lock_status
        )

        # Update affinities
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    bonus = 1.0
                    if i == 4:
                        bonus = 1.5
                    # Lock bonus
                    if act_protection[i] > 0.5:
                        bonus *= 1.3
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * bonus)
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus = 1.4
                    if agg_protection[j] > 0.5:
                        bonus *= 1.3
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * bonus)
                    )

        # Affinity floors
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Pruning (with lock protection)
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

        # Exploration (locked pairs boost exploration)
        rng_key, explore_key = jax.random.split(rng_key)
        explore_rand = jax.random.uniform(explore_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        act_count = int(jnp.sum(new_act_mask))
        agg_count = int(jnp.sum(new_agg_mask))

        # Calculate locked pair count
        locked_pairs = sum(1 for a, j in self.lock_pairs if lock_status[a, j] > 0.5)

        if act_count < self.max_active_act:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] < 0.5:
                    rate = self.exploration_rate_act
                    if locked_pairs > 0:
                        rate *= (1 + self.lock_exploration_boost)
                    if explore_rand[i] < rate:
                        new_act_mask = new_act_mask.at[i].set(1.0)
                        break

        if agg_count < self.max_active_agg:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] < 0.5:
                    rate = self.exploration_rate_agg
                    if locked_pairs > 0:
                        rate *= (1 + self.lock_exploration_boost)
                    idx = NUM_ACTIVATIONS + j
                    if explore_rand[idx] < rate:
                        new_agg_mask = new_agg_mask.at[j].set(1.0)
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
            'coexistence_count': coexistence_count,
            'lock_status': lock_status,
            'peak_fitness': new_peak,
            'cross_affinity': cross_affinity,
            'rng_key': rng_key,
            'generation': generation + 1,
            'stagnation_count': stagnation_count,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'lock_events': state['lock_events'] + lock_events,
            'unlock_events': state['unlock_events'] + unlock_events,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'lock_events': lock_events,
            'unlock_events': unlock_events,
            'locked_pairs': locked_pairs,
            'coexistence_sin_max': float(coexistence_count[4, 2]),
            'coexistence_sin_min': float(coexistence_count[4, 3]),
        }

        return new_state, metrics
