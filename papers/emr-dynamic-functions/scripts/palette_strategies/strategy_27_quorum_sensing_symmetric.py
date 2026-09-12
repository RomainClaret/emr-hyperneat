"""Strategy 27S: Quorum Sensing Symmetric (Population-Level Consensus).

Biological Basis: Bacterial quorum sensing where cells coordinate behavior based on
population-level chemical signaling. When enough bacteria are present (quorum),
collective behaviors like bioluminescence or virulence are activated.

Key mechanisms for SYMMETRIC discovery:
1. Dual collective memory for activation and aggregation domains
2. Population-level voting weighted by fitness
3. Quorum threshold for stable function promotion
4. Cross-domain consensus: successful activations support related aggregations
5. Protected indices (0.1% deactivation) for sin and extreme aggregations
6. Memory cell crystallization for stable, high-performing functions
7. Affinity floors for guaranteed retention

Expected: Population consensus drives discovery of robust function combinations.
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

# Critical indices for retention
SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3


class QuorumSensingSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with population-level quorum sensing.

    Functions in both domains accumulate "votes" from across the population.
    When a quorum is reached, functions are promoted to stable status with
    reduced deactivation rates.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Discovery tracking for research analysis
    """

    name = "quorum_sensing_symmetric"
    description = "Symmetric: Population-level voting and consensus for bidirectional discovery"

    def __init__(
        self,
        # Quorum parameters (both domains)
        quorum_threshold: float = 0.4,
        minority_threshold: float = 0.05,
        signal_decay: float = 0.85,
        # Voting weights
        vote_weight_by_fitness: bool = True,
        fitness_weight_power: float = 2.0,
        # Cross-domain influence
        cross_learning_rate: float = 0.05,
        cross_influence: float = 0.15,
        # Function states
        stable_promotion_gens: int = 5,
        unstable_after_gens: int = 10,
        # Mutation rates
        stable_deactivate_rate: float = 0.01,
        normal_activate_rate: float = 0.12,
        normal_deactivate_rate: float = 0.06,
        minority_activate_boost: float = 1.5,
        # Affinity floors for guaranteed retention
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Memory cell parameters
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Constraints
        min_active_act: int = 2,
        max_active_act: int = 6,
        min_active_agg: int = 1,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Quorum Sensing Symmetric strategy."""
        # Quorum
        self.quorum_threshold = quorum_threshold
        self.minority_threshold = minority_threshold
        self.signal_decay = signal_decay

        # Voting
        self.vote_weight_by_fitness = vote_weight_by_fitness
        self.fitness_weight_power = fitness_weight_power

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence

        # Stability
        self.stable_promotion_gens = stable_promotion_gens
        self.unstable_after_gens = unstable_after_gens

        # Mutation
        self.stable_deactivate_rate = stable_deactivate_rate
        self.normal_activate_rate = normal_activate_rate
        self.normal_deactivate_rate = normal_deactivate_rate
        self.minority_activate_boost = minority_activate_boost

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Memory cells
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Constraints
        self.min_active_act = min_active_act
        self.max_active_act = max_active_act
        self.min_active_agg = min_active_agg
        self.max_active_agg = max_active_agg

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual collective memories and tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Collective memory (signal strength) for each function
        act_collective = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_collective = act_collective.at[i].set(0.3)
        # Sin starts with higher signal
        act_collective = act_collective.at[SIN_IDX].set(self.sin_affinity_floor)

        agg_collective = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_collective = agg_collective.at[i].set(0.3)
        # Extreme aggs start with higher signal
        for idx in [MAX_IDX, MIN_IDX]:
            agg_collective = agg_collective.at[idx].set(self.extreme_agg_affinity_floor)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        discovery_gen = {'sin': -1, 'max': -1, 'min': -1}

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
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Discovery
            'discovery_gen': discovery_gen,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 27000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'fitness_ema': 0.5,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _apply_signal_floors(
        self,
        act_collective: jnp.ndarray,
        agg_collective: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply minimum signal floors for critical functions."""
        new_act = act_collective.at[SIN_IDX].set(
            jnp.maximum(act_collective[SIN_IDX], self.sin_affinity_floor)
        )
        new_agg = agg_collective
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )
        return new_act, new_agg

    def _update_memory_cells(
        self,
        collective: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell status based on sustained high collective signal."""
        active = mask > 0.5
        above_threshold = collective >= self.memory_formation_threshold

        candidate = active & above_threshold
        new_counts = jnp.where(candidate, memory_counts + 1, 0)

        newly_memory = new_counts >= self.memory_formation_count
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _compute_population_signal(
        self,
        mask: jnp.ndarray,
        fitness: float,
        best_fitness: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute population signal for a domain."""
        if self.vote_weight_by_fitness and best_fitness > 0.01:
            relative_fitness = fitness / best_fitness
            vote_weight = relative_fitness ** self.fitness_weight_power
        else:
            vote_weight = 1.0

        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)
        signal = active * vote_weight / n_active

        return signal

    def _update_collective_memory(
        self,
        collective: jnp.ndarray,
        signal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update collective memory with signal decay."""
        new_collective = (
            self.signal_decay * collective +
            (1 - self.signal_decay) * signal
        )
        return jnp.clip(new_collective, 0.0, 1.0)

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
        return jnp.clip(cross_affinity + delta, 0.0, 1.0)

    def _update_quorum_counts(
        self,
        collective: jnp.ndarray,
        above_count: jnp.ndarray,
        below_count: jnp.ndarray,
        stable_list: List[int],
        n_functions: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[int]]:
        """Update quorum status counters for a domain."""
        new_above = above_count.copy()
        new_below = below_count.copy()
        new_stable = list(stable_list)

        for i in range(n_functions):
            signal = float(collective[i])

            if signal >= self.quorum_threshold:
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

    def _apply_quorum_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        collective: jnp.ndarray,
        stable_list: List[int],
        memory_cells: jnp.ndarray,
        cross_support: jnp.ndarray,
        n_funcs: int,
        min_active: int,
        max_active: int,
        protected_indices: List[int],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply quorum-based mutation with protection."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (n_funcs,))
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(n_funcs):
            signal = float(collective[i])
            support = float(cross_support[i]) if i < len(cross_support) else 0.0
            effective_signal = signal + self.cross_influence * support

            is_stable = i in stable_list
            is_memory = bool(memory_cells[i])
            is_protected = i in protected_indices
            at_quorum = effective_signal >= self.quorum_threshold
            at_minority = signal <= self.minority_threshold

            if mask[i] < 0.5:
                # Inactive: maybe activate
                if is_stable or at_quorum:
                    rate = self.normal_activate_rate * (1.0 + effective_signal)
                elif at_minority:
                    rate = self.normal_activate_rate * self.minority_activate_boost * 0.5
                else:
                    rate = self.normal_activate_rate * (0.3 + 0.7 * effective_signal)

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                if is_protected:
                    rate = 0.001  # 0.1% for protected
                elif is_memory or is_stable:
                    rate = self.stable_deactivate_rate
                elif at_quorum:
                    rate = self.normal_deactivate_rate * 0.2
                else:
                    rate = self.normal_deactivate_rate * (1.0 - effective_signal * 0.7)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        n_active = int(jnp.sum(new_mask > 0.5))
        if n_active < min_active or n_active > max_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _update_discovery_tracking(
        self,
        discovery_gen: Dict[str, int],
        act_palette: List[int],
        agg_palette: List[int],
        generation: int,
    ) -> Dict[str, int]:
        """Track when critical functions are first discovered."""
        new_discovery = discovery_gen.copy()

        if SIN_IDX in act_palette and new_discovery['sin'] < 0:
            new_discovery['sin'] = generation
        if MAX_IDX in agg_palette and new_discovery['max'] < 0:
            new_discovery['max'] = generation
        if MIN_IDX in agg_palette and new_discovery['min'] < 0:
            new_discovery['min'] = generation

        return new_discovery

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

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Compute population signals
        act_signal = self._compute_population_signal(
            state['act_mask'], best_fitness, new_best, NUM_ACTIVATIONS
        )
        agg_signal = self._compute_population_signal(
            state['agg_mask'], best_fitness, new_best, NUM_AGGREGATIONS
        )

        # Update collective memories
        new_act_collective = self._update_collective_memory(state['act_collective'], act_signal)
        new_agg_collective = self._update_collective_memory(state['agg_collective'], agg_signal)

        # Apply signal floors
        new_act_collective, new_agg_collective = self._apply_signal_floors(
            new_act_collective, new_agg_collective
        )

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], fitness_delta
        )

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_collective, state['act_memory_counts'],
            state['act_memory_cells'], state['act_mask']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_collective, state['agg_memory_counts'],
            state['agg_memory_cells'], state['agg_mask']
        )

        # Update quorum counts
        new_act_above, new_act_below, new_act_stable = self._update_quorum_counts(
            new_act_collective, state['act_above_quorum'],
            state['act_below_quorum'], state['act_stable'], NUM_ACTIVATIONS
        )
        new_agg_above, new_agg_below, new_agg_stable = self._update_quorum_counts(
            new_agg_collective, state['agg_above_quorum'],
            state['agg_below_quorum'], state['agg_stable'], NUM_AGGREGATIONS
        )

        # Cross-domain support
        act_cross_support = jnp.dot(new_cross, state['agg_mask'] > 0.5) / max(jnp.sum(state['agg_mask'] > 0.5), 1)
        agg_cross_support = jnp.dot(new_cross.T, state['act_mask'] > 0.5) / max(jnp.sum(state['act_mask'] > 0.5), 1)

        # Apply mutations with protection
        new_act_mask, act_mutation = self._apply_quorum_mutation(
            k1, state['act_mask'], new_act_collective, new_act_stable,
            new_act_mem_cells, act_cross_support, NUM_ACTIVATIONS,
            self.min_active_act, self.max_active_act, [SIN_IDX]
        )
        new_agg_mask, agg_mutation = self._apply_quorum_mutation(
            k2, state['agg_mask'], new_agg_collective, new_agg_stable,
            new_agg_mem_cells, agg_cross_support, NUM_AGGREGATIONS,
            self.min_active_agg, self.max_active_agg, [MAX_IDX, MIN_IDX]
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Get palettes and update discovery
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)
        new_discovery = self._update_discovery_tracking(
            state['discovery_gen'], act_palette, agg_palette, generation
        )

        new_fitness_ema = 0.9 * state['fitness_ema'] + 0.1 * best_fitness

        new_state = {
            'act_mask': new_act_mask,
            'act_collective': new_act_collective,
            'act_stable': new_act_stable,
            'act_above_quorum': new_act_above,
            'act_below_quorum': new_act_below,
            'agg_mask': new_agg_mask,
            'agg_collective': new_agg_collective,
            'agg_stable': new_agg_stable,
            'agg_above_quorum': new_agg_above,
            'agg_below_quorum': new_agg_below,
            'cross_affinity': new_cross,
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'discovery_gen': new_discovery,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'fitness_ema': new_fitness_ema,
            'strategy_name': self.name,
        }

        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Quorum metrics
            'act_n_stable': len(new_act_stable),
            'agg_n_stable': len(new_agg_stable),
            'act_mean_collective': float(jnp.mean(new_act_collective)),
            'agg_mean_collective': float(jnp.mean(new_agg_collective)),
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Sin status
            'has_sin': SIN_IDX in act_palette,
            'sin_collective': float(new_act_collective[SIN_IDX]),
            'sin_stable': SIN_IDX in new_act_stable,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            # Discovery
            'sin_discovered_gen': new_discovery['sin'],
            'max_discovered_gen': new_discovery['max'],
            'min_discovered_gen': new_discovery['min'],
        }
        metrics.update({'act_' + k: v for k, v in act_mutation.items()})
        metrics.update({'agg_' + k: v for k, v in agg_mutation.items()})

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with quorum status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        act_mem_count = int(jnp.sum(state['act_memory_cells']))
        agg_mem_count = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'act_stable': state['act_stable'],
            'agg_stable': state['agg_stable'],
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'sin_collective': float(state['act_collective'][SIN_IDX]),
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
            'discovery_gen': state['discovery_gen'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
