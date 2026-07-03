"""Strategy 137S: Sin-Extreme Symbiosis Symmetric (Obligate Mutualism).

Biological Basis: Obligate mutualism in nature (e.g., clownfish-anemone,
mycorrhizal fungi-plant roots) where both partners depend on each other.
When one partner is removed, the other becomes vulnerable.

Key mechanisms for SYMMETRIC discovery:
1. Sin and extreme_aggs (max/min) form symbiotic pairs
2. When partner is active, protection is maximized
3. Orphaned functions (without partners) are 2x more likely to be pruned
4. Symbiosis formation rate tracks how quickly new pairs form
5. Protected indices (0.1% deactivation) for sin and extreme aggregations
6. Memory cell crystallization for stable, high-affinity functions
7. Affinity floors for guaranteed retention

Expected: Symbiotic coupling between sin and extreme aggs ensures their co-retention.
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


class SinExtremeSymbiosisSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with sin-extreme symbiosis.

    Sin and extreme aggregations form obligate mutualistic pairs.
    They protect each other from deactivation, and orphaned functions
    (those without their partner) become vulnerable.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations (0.1% deactivation)
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Discovery tracking for research analysis
    """

    name = "sin_extreme_symbiosis_symmetric"
    description = "Symmetric: Obligate mutualism between sin and extreme aggs"

    def __init__(
        self,
        # Symbiosis parameters
        symbiosis_protection: float = 0.8,
        orphan_vulnerability: float = 2.0,
        symbiosis_formation_rate: float = 0.15,
        # Affinity parameters
        affinity_lr: float = 0.12,
        affinity_decay: float = 0.98,
        cross_affinity_lr: float = 0.12,
        # Pruning parameters
        prune_threshold: float = 0.2,
        stagnation_prune_boost: float = 0.15,
        # Exploration parameters
        exploration_rate: float = 0.10,
        symbiotic_exploration_boost: float = 0.5,
        # Affinity floors for guaranteed retention
        sin_affinity_floor: float = 0.7,  # Higher since symbiosis is key
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
        """Initialize Sin-Extreme Symbiosis Symmetric strategy."""
        # Symbiosis
        self.symbiosis_pairs = [(SIN_IDX, MAX_IDX), (SIN_IDX, MIN_IDX)]
        self.symbiosis_protection = symbiosis_protection
        self.orphan_vulnerability = orphan_vulnerability
        self.symbiosis_formation_rate = symbiosis_formation_rate

        # Affinity
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_affinity_lr = cross_affinity_lr

        # Pruning
        self.prune_threshold = prune_threshold
        self.stagnation_prune_boost = stagnation_prune_boost

        # Exploration
        self.exploration_rate = exploration_rate
        self.symbiotic_exploration_boost = symbiotic_exploration_boost

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
        """Initialize state with symbiotic pairs."""
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

        # Symbiosis strength matrix
        symbiosis_strength = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        for act_idx, agg_idx in self.symbiosis_pairs:
            symbiosis_strength = symbiosis_strength.at[act_idx, agg_idx].set(1.0)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in [MAX_IDX, MIN_IDX]:
            cross_affinity = cross_affinity.at[SIN_IDX, agg].set(0.7)

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        discovery_gen = {'sin': 0, 'max': 0, 'min': 0}  # All start in palette

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinity': act_affinity,
            'agg_affinity': agg_affinity,
            'symbiosis_strength': symbiosis_strength,
            'orphan_count_act': jnp.zeros(NUM_ACTIVATIONS),
            'orphan_count_agg': jnp.zeros(NUM_AGGREGATIONS),
            'cross_affinity': cross_affinity,
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            'discovery_gen': discovery_gen,
            'symbiosis_events': 0,
            'orphan_pruning_events': 0,
            'rng_key': jax.random.PRNGKey(seed + 137000),
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

    def _check_symbiosis_status(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Check which functions have active symbiotic partners."""
        has_partner_act = jnp.zeros(NUM_ACTIVATIONS)
        has_partner_agg = jnp.zeros(NUM_AGGREGATIONS)

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                for j in range(NUM_AGGREGATIONS):
                    if agg_mask[j] > 0.5 and symbiosis_strength[i, j] > 0.3:
                        has_partner_act = has_partner_act.at[i].set(1.0)
                        break

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                for i in range(NUM_ACTIVATIONS):
                    if act_mask[i] > 0.5 and symbiosis_strength[i, j] > 0.3:
                        has_partner_agg = has_partner_agg.at[j].set(1.0)
                        break

        return has_partner_act, has_partner_agg

    def _update_symbiosis(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
        fitness_delta: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int]:
        """Update symbiotic relationships."""
        new_symbiosis = symbiosis_strength.copy()
        events = 0

        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        if new_symbiosis[i, j] > 0.1:
                            new_symbiosis = new_symbiosis.at[i, j].set(
                                min(1.0, new_symbiosis[i, j] + 0.1)
                            )
                        elif new_symbiosis[i, j] < 0.1:
                            if jax.random.uniform(key) < self.symbiosis_formation_rate:
                                new_symbiosis = new_symbiosis.at[i, j].set(0.3)
                                events += 1

        # Decay bonds for inactive pairs
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if act_mask[i] < 0.5 or agg_mask[j] < 0.5:
                    new_symbiosis = new_symbiosis.at[i, j].set(
                        new_symbiosis[i, j] * 0.9
                    )

        # CRITICAL: Sin-extreme bonds never fully break
        for act_idx, agg_idx in self.symbiosis_pairs:
            new_symbiosis = new_symbiosis.at[act_idx, agg_idx].set(
                max(0.5, float(new_symbiosis[act_idx, agg_idx]))
            )

        return new_symbiosis, events

    def _calculate_protection(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        has_partner_act: jnp.ndarray,
        has_partner_agg: jnp.ndarray,
        act_memory_cells: jnp.ndarray,
        agg_memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate protection based on symbiotic status and memory cells."""
        act_protection = jnp.zeros(NUM_ACTIVATIONS)
        agg_protection = jnp.zeros(NUM_AGGREGATIONS)

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                if act_memory_cells[i]:
                    act_protection = act_protection.at[i].set(0.9)
                elif has_partner_act[i] > 0.5:
                    act_protection = act_protection.at[i].set(self.symbiosis_protection)
                else:
                    act_protection = act_protection.at[i].set(0.1)

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                if agg_memory_cells[j]:
                    agg_protection = agg_protection.at[j].set(0.9)
                elif has_partner_agg[j] > 0.5:
                    agg_protection = agg_protection.at[j].set(self.symbiosis_protection)
                else:
                    agg_protection = agg_protection.at[j].set(0.1)

        # CRITICAL: Sin always has high base protection
        act_protection = act_protection.at[SIN_IDX].set(
            max(0.95, float(act_protection[SIN_IDX]))
        )
        for agg in [MAX_IDX, MIN_IDX]:
            agg_protection = agg_protection.at[agg].set(
                max(0.9, float(agg_protection[agg]))
            )

        return act_protection, agg_protection

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
        """Update palette using symbiotic relationships."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        fitness_delta = best_fitness - prev_best_fitness
        improved = fitness_delta > 0.001

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Check symbiosis status
        has_partner_act, has_partner_agg = self._check_symbiosis_status(
            state['act_mask'], state['agg_mask'], state['symbiosis_strength']
        )

        # Update orphan counts
        new_orphan_act = state['orphan_count_act'].copy()
        new_orphan_agg = state['orphan_count_agg'].copy()

        for i in range(NUM_ACTIVATIONS):
            if state['act_mask'][i] > 0.5 and has_partner_act[i] < 0.5:
                new_orphan_act = new_orphan_act.at[i].set(new_orphan_act[i] + 1)
            else:
                new_orphan_act = new_orphan_act.at[i].set(0)

        for j in range(NUM_AGGREGATIONS):
            if state['agg_mask'][j] > 0.5 and has_partner_agg[j] < 0.5:
                new_orphan_agg = new_orphan_agg.at[j].set(new_orphan_agg[j] + 1)
            else:
                new_orphan_agg = new_orphan_agg.at[j].set(0)

        # Update symbiosis
        new_symbiosis, symbiosis_events = self._update_symbiosis(
            state['act_mask'], state['agg_mask'],
            state['symbiosis_strength'], fitness_delta, k1
        )

        # Update affinities
        new_act_aff = state['act_affinity'] * self.affinity_decay
        new_agg_aff = state['agg_affinity'] * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    bonus = 1.3 if has_partner_act[i] > 0.5 else 1.0
                    if i == SIN_IDX:
                        bonus *= 1.2
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.affinity_lr * bonus)
                    )

            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    bonus = 1.3 if has_partner_agg[j] > 0.5 else 1.0
                    if j in [MAX_IDX, MIN_IDX]:
                        bonus *= 1.2
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

        # Calculate protection
        act_protection, agg_protection = self._calculate_protection(
            state['act_mask'], state['agg_mask'],
            has_partner_act, has_partner_agg,
            new_act_mem_cells, new_agg_mem_cells
        )

        # Pruning with orphan vulnerability (but NEVER prune protected)
        new_act_mask = state['act_mask'].copy()
        new_agg_mask = state['agg_mask'].copy()
        orphan_pruning = 0

        prune_rand = jax.random.uniform(k2, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        if new_stagnation > 5:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] > 0.5 and i != SIN_IDX:  # Never prune sin
                    base_threshold = self.prune_threshold
                    if new_orphan_act[i] > 3:
                        base_threshold *= self.orphan_vulnerability
                        orphan_pruning += 1
                    if new_act_aff[i] < base_threshold:
                        if prune_rand[i] > act_protection[i]:
                            new_act_mask = new_act_mask.at[i].set(0.0)

            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5 and j not in [MAX_IDX, MIN_IDX]:  # Never prune extreme
                    base_threshold = self.prune_threshold
                    if new_orphan_agg[j] > 3:
                        base_threshold *= self.orphan_vulnerability
                        orphan_pruning += 1
                    if new_agg_aff[j] < base_threshold:
                        if prune_rand[NUM_ACTIVATIONS + j] > agg_protection[j]:
                            new_agg_mask = new_agg_mask.at[j].set(0.0)

        # CRITICAL: Ensure sin and extreme aggs are NEVER removed
        new_act_mask = new_act_mask.at[SIN_IDX].set(1.0)
        for agg in [MAX_IDX, MIN_IDX]:
            new_agg_mask = new_agg_mask.at[agg].set(1.0)

        # Exploration
        explore_rand = jax.random.uniform(key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        act_count = int(jnp.sum(new_act_mask))
        agg_count = int(jnp.sum(new_agg_mask))

        if act_count < self.max_active_act:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] < 0.5:
                    rate = self.exploration_rate
                    if has_partner_act.sum() > 0:
                        rate *= (1 + self.symbiotic_exploration_boost)
                    if explore_rand[i] < rate:
                        new_act_mask = new_act_mask.at[i].set(1.0)
                        break

        if agg_count < self.max_active_agg:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] < 0.5:
                    rate = self.exploration_rate
                    if has_partner_agg.sum() > 0:
                        rate *= (1 + self.symbiotic_exploration_boost)
                    if explore_rand[NUM_ACTIVATIONS + j] < rate:
                        new_agg_mask = new_agg_mask.at[j].set(1.0)
                        break

        # Ensure minimums
        if int(jnp.sum(new_act_mask)) < self.min_active_act:
            new_act_mask = new_act_mask.at[0].set(1.0)
        if int(jnp.sum(new_agg_mask)) < self.min_active_agg:
            new_agg_mask = new_agg_mask.at[0].set(1.0)

        # Get palettes and update discovery
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
            'symbiosis_strength': new_symbiosis,
            'orphan_count_act': new_orphan_act,
            'orphan_count_agg': new_orphan_agg,
            'cross_affinity': state['cross_affinity'],
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'discovery_gen': new_discovery,
            'symbiosis_events': state['symbiosis_events'] + symbiosis_events,
            'orphan_pruning_events': state['orphan_pruning_events'] + orphan_pruning,
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
            # Symbiosis
            'symbiosis_events': symbiosis_events,
            'orphan_pruning': orphan_pruning,
            'has_partners_act': int(has_partner_act.sum()),
            'has_partners_agg': int(has_partner_agg.sum()),
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
        """Return state summary with symbiosis status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        has_partner_act, has_partner_agg = self._check_symbiosis_status(
            state['act_mask'], state['agg_mask'], state['symbiosis_strength']
        )

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
            'has_partners_act': int(has_partner_act.sum()),
            'has_partners_agg': int(has_partner_agg.sum()),
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'symbiosis_events': state['symbiosis_events'],
            'orphan_pruning_events': state['orphan_pruning_events'],
            'discovery_gen': state['discovery_gen'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
