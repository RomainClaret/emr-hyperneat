"""Strategy 138S: Sin-Extreme Codiscovery Symmetric (Cross-Domain Triggers).

Biological Basis: Gene co-expression networks where activation of one gene
triggers expression of related genes in the same pathway. Discovery of one
component creates a "discovery window" where related components are more
likely to be discovered.

Key mechanisms for SYMMETRIC discovery:
1. Discovery of sin triggers +40% exploration probability for extreme aggs
2. Discovery of extreme agg triggers +40% exploration for sin
3. Discovery window lasts for 10 generations after discovery
4. Initial sin exploration bias ensures early discovery
5. Protected indices (0.1% deactivation) for sin and extreme aggregations
6. Memory cell crystallization for stable, high-affinity functions
7. Affinity floors for guaranteed retention

Expected: Cross-domain discovery triggers accelerate finding of sin-extreme pairs.
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


class SinExtremeCodeiscoverySymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with cross-domain discovery triggers.

    Discovery of sin or extreme aggs creates a temporary boost window
    for discovering the complementary domain, encouraging rapid
    establishment of sin-extreme pairs.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations (0.1% deactivation)
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Discovery tracking for research analysis
    """

    name = "sin_extreme_codiscovery_symmetric"
    description = "Symmetric: Cross-domain discovery triggers between sin and extreme aggs"

    def __init__(
        self,
        # Codiscovery parameters
        codiscovery_boost: float = 0.4,
        discovery_window: int = 10,
        cross_domain_affinity_rate: float = 0.2,
        initial_sin_exploration_bias: float = 0.15,
        # Affinity parameters
        affinity_lr: float = 0.12,
        affinity_decay: float = 0.98,
        # Exploration parameters
        base_exploration_rate: float = 0.10,
        # Pruning parameters
        prune_threshold: float = 0.2,
        stagnation_prune_after: int = 6,
        # Affinity floors for guaranteed retention
        sin_affinity_floor: float = 0.7,
        extreme_agg_affinity_floor: float = 0.6,
        # Memory cell parameters
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Sin-Extreme Codiscovery Symmetric strategy."""
        # Codiscovery
        self.codiscovery_boost = codiscovery_boost
        self.discovery_window = discovery_window
        self.cross_domain_affinity_rate = cross_domain_affinity_rate
        self.initial_sin_exploration_bias = initial_sin_exploration_bias

        # Affinity
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # Exploration
        self.base_exploration_rate = base_exploration_rate

        # Pruning
        self.prune_threshold = prune_threshold
        self.stagnation_prune_after = stagnation_prune_after

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Memory cells
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial palettes - ALWAYS include sin and extreme aggs
        default_act = list(DEFAULT_PALETTE_INDICES)
        if SIN_IDX not in default_act:
            default_act.append(SIN_IDX)
        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        for agg in [MAX_IDX, MIN_IDX]:
            if agg not in default_agg:
                default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with codiscovery tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # Ensure sin and extreme aggs in initial palettes
        initial_act = list(initial_act)
        if SIN_IDX not in initial_act:
            initial_act.append(SIN_IDX)
        initial_agg = list(initial_agg)
        for agg in [MAX_IDX, MIN_IDX]:
            if agg not in initial_agg:
                initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinity = act_affinity.at[i].set(0.5)
        act_affinity = act_affinity.at[SIN_IDX].set(self.sin_affinity_floor)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinity = agg_affinity.at[i].set(0.55)
        for agg in [MAX_IDX, MIN_IDX]:
            agg_affinity = agg_affinity.at[agg].set(self.extreme_agg_affinity_floor)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in [MAX_IDX, MIN_IDX]:
            cross_affinity = cross_affinity.at[SIN_IDX, agg].set(0.7)

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking (all start in palette so discovered at gen 0)
        discovery_gen = {'sin': 0, 'max': 0, 'min': 0}

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinity': act_affinity,
            'agg_affinity': agg_affinity,
            'cross_affinity': cross_affinity,
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Exploration boost state
            'sin_discovery_boost_remaining': self.discovery_window,
            'extreme_discovery_boost_remaining': self.discovery_window,
            # Discovery tracking
            'discovery_gen': discovery_gen,
            'codiscovery_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 138000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply minimum affinity floors for critical functions."""
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )
        new_agg = agg_affinity
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )
        return new_act, new_agg

    def _update_memory_cells(
        self,
        affinity: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell status based on sustained high affinity."""
        active = mask > 0.5
        above_threshold = affinity >= self.memory_formation_threshold

        candidate = active & above_threshold
        new_counts = jnp.where(candidate, memory_counts + 1, 0)

        newly_memory = new_counts >= self.memory_formation_count
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _calculate_exploration_rates(
        self,
        sin_boost_remaining: int,
        extreme_boost_remaining: int,
        generation: int,
    ) -> Tuple[float, float]:
        """Calculate current exploration rates based on discovery boosts."""
        act_rate = self.base_exploration_rate
        agg_rate = self.base_exploration_rate

        # If extreme was recently discovered, boost sin exploration
        if extreme_boost_remaining > 0:
            act_rate *= (1 + self.codiscovery_boost)

        # If sin was recently discovered, boost extreme exploration
        if sin_boost_remaining > 0:
            agg_rate *= (1 + self.codiscovery_boost)

        # Early generation bias toward sin
        if generation < 20:
            act_rate += self.initial_sin_exploration_bias

        return act_rate, agg_rate

    def _check_new_discoveries(
        self,
        new_act_mask: jnp.ndarray,
        prev_act_mask: jnp.ndarray,
        new_agg_mask: jnp.ndarray,
        prev_agg_mask: jnp.ndarray,
    ) -> Tuple[bool, bool]:
        """Check if sin or extreme was just discovered (activated)."""
        sin_just_discovered = (
            new_act_mask[SIN_IDX] > 0.5 and
            prev_act_mask[SIN_IDX] < 0.5
        )

        extreme_just_discovered = any(
            new_agg_mask[idx] > 0.5 and prev_agg_mask[idx] < 0.5
            for idx in [MAX_IDX, MIN_IDX]
        )

        return sin_just_discovered, extreme_just_discovered

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
        """Update palette using codiscovery triggers."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        fitness_delta = best_fitness - prev_best_fitness
        improved = fitness_delta > 0.001

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Store previous masks
        prev_act_mask = state['act_mask'].copy()
        prev_agg_mask = state['agg_mask'].copy()

        # Get exploration rates
        act_explore_rate, agg_explore_rate = self._calculate_exploration_rates(
            state['sin_discovery_boost_remaining'],
            state['extreme_discovery_boost_remaining'],
            generation,
        )

        # Update affinities
        new_act_aff = state['act_affinity'] * self.affinity_decay
        new_agg_aff = state['agg_affinity'] * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    bonus = 1.2 if i == SIN_IDX else 1.0
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.affinity_lr * bonus)
                    )

            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    bonus = 1.2 if j in [MAX_IDX, MIN_IDX] else 1.0
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.affinity_lr * bonus)
                    )

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_aff, state['act_memory_counts'],
            state['act_memory_cells'], state['act_mask']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_aff, state['agg_memory_counts'],
            state['agg_memory_cells'], state['agg_mask']
        )

        # Pruning (protect sin and extreme)
        new_act_mask = state['act_mask'].copy()
        new_agg_mask = state['agg_mask'].copy()

        prune_rand = jax.random.uniform(k1, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        if new_stagnation > self.stagnation_prune_after:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] > 0.5 and i != SIN_IDX:  # Never prune sin
                    if new_act_aff[i] < self.prune_threshold:
                        if not new_act_mem_cells[i]:  # Don't prune memory cells
                            if prune_rand[i] > 0.5:
                                new_act_mask = new_act_mask.at[i].set(0.0)

            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5 and j not in [MAX_IDX, MIN_IDX]:  # Never prune extreme
                    if new_agg_aff[j] < self.prune_threshold:
                        if not new_agg_mem_cells[j]:
                            if prune_rand[NUM_ACTIVATIONS + j] > 0.5:
                                new_agg_mask = new_agg_mask.at[j].set(0.0)

        # Exploration with codiscovery boost
        explore_rand = jax.random.uniform(k2, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        act_count = int(jnp.sum(new_act_mask))
        agg_count = int(jnp.sum(new_agg_mask))
        codiscovery_events = 0

        if act_count < self.max_active_act:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] < 0.5:
                    rate = act_explore_rate
                    if i == SIN_IDX:
                        rate *= 1.5
                    if explore_rand[i] < rate:
                        new_act_mask = new_act_mask.at[i].set(1.0)
                        if i == SIN_IDX:
                            codiscovery_events += 1
                        break

        if agg_count < self.max_active_agg:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] < 0.5:
                    rate = agg_explore_rate
                    if j in [MAX_IDX, MIN_IDX]:
                        rate *= 1.5
                    if explore_rand[NUM_ACTIVATIONS + j] < rate:
                        new_agg_mask = new_agg_mask.at[j].set(1.0)
                        if j in [MAX_IDX, MIN_IDX]:
                            codiscovery_events += 1
                        break

        # CRITICAL: Ensure sin and extreme aggs are NEVER removed
        new_act_mask = new_act_mask.at[SIN_IDX].set(1.0)
        for agg in [MAX_IDX, MIN_IDX]:
            new_agg_mask = new_agg_mask.at[agg].set(1.0)

        # Check for new discoveries and update boost counters
        sin_just, extreme_just = self._check_new_discoveries(
            new_act_mask, prev_act_mask, new_agg_mask, prev_agg_mask
        )

        new_sin_boost = state['sin_discovery_boost_remaining'] - 1
        new_extreme_boost = state['extreme_discovery_boost_remaining'] - 1
        new_sin_boost = max(0, new_sin_boost)
        new_extreme_boost = max(0, new_extreme_boost)

        if sin_just:
            new_sin_boost = self.discovery_window
            codiscovery_events += 1
        if extreme_just:
            new_extreme_boost = self.discovery_window
            codiscovery_events += 1

        # Ensure minimums
        if int(jnp.sum(new_act_mask)) < self.min_active_act:
            new_act_mask = new_act_mask.at[0].set(1.0)
        if int(jnp.sum(new_agg_mask)) < self.min_active_agg:
            new_agg_mask = new_agg_mask.at[0].set(1.0)

        # Get palettes and update discovery tracking
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)
        new_discovery = self._update_discovery_tracking(
            state['discovery_gen'], act_palette, agg_palette, generation
        )

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinity': new_act_aff,
            'agg_affinity': new_agg_aff,
            'cross_affinity': state['cross_affinity'],
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'sin_discovery_boost_remaining': new_sin_boost,
            'extreme_discovery_boost_remaining': new_extreme_boost,
            'discovery_gen': new_discovery,
            'codiscovery_events': state['codiscovery_events'] + codiscovery_events,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Codiscovery
            'codiscovery_events': codiscovery_events,
            'sin_boost_remaining': new_sin_boost,
            'extreme_boost_remaining': new_extreme_boost,
            'act_explore_rate': act_explore_rate,
            'agg_explore_rate': agg_explore_rate,
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Status
            'has_sin': SIN_IDX in act_palette,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            # Discovery
            'sin_discovered_gen': new_discovery['sin'],
            'max_discovered_gen': new_discovery['max'],
            'min_discovered_gen': new_discovery['min'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with codiscovery status."""
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
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'sin_boost_remaining': state['sin_discovery_boost_remaining'],
            'extreme_boost_remaining': state['extreme_discovery_boost_remaining'],
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'codiscovery_events': state['codiscovery_events'],
            'discovery_gen': state['discovery_gen'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
