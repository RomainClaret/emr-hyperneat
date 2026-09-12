"""Strategy 30S: Predator-Prey Symmetric (Ecological Oscillation Dynamics).

Biological Basis: Lotka-Volterra predator-prey dynamics from ecology. Prey (generalist
functions) have exponential growth while predators (specialist functions) depend
on prey for survival. This creates natural oscillation patterns in population sizes.

Key mechanisms for SYMMETRIC discovery:
1. Separate prey/predator populations for both activation and aggregation domains
2. Cross-feeding: success in one domain provides energy for the other
3. Synchronized oscillation phases create coordinated exploration
4. Shared predator success memory reduces death rates on joint success
5. Protected indices (0.1% deactivation) for sin and extreme aggregations
6. Memory cell crystallization for stable, high-performing functions
7. Affinity floors for guaranteed retention

Sin is classified as a PREDATOR (specialist) while basic activations are PREY (generalists).
Max/min aggregations are predators while sum/mean are prey.

Expected: Ecological oscillations drive discovery of diverse function combinations.
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


class PredatorPreySymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with predator-prey dynamics.

    Both domains have generalist (prey) and specialist (predator) functions.
    Lotka-Volterra dynamics create natural exploration cycles.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations (0.1% deactivation)
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Discovery tracking for research analysis
    """

    name = "predator_prey_symmetric"
    description = "Symmetric: Lotka-Volterra oscillations for bidirectional discovery"

    # Activation function classifications
    ACT_PREY = [0, 1, 2, 3, 5, 6]  # identity, tanh, sigmoid, relu, step, leaky_relu
    ACT_PREDATORS = [4, 11, 12, 13, 15]  # sin, burst, resonator, osc_adapt, receptive
    ACT_NEUTRAL = [7, 8, 9, 10, 14, 16, 17]  # gaussian, softplus, elu, swish, etc.

    # Aggregation function classifications
    AGG_PREY = [0, 1]  # sum, mean - generalists
    AGG_PREDATORS = [2, 3]  # max, min - specialists
    AGG_NEUTRAL = [4, 5]  # prod, abs_max - context-dependent

    def __init__(
        self,
        # Lotka-Volterra parameters
        prey_growth: float = 0.12,
        predation_rate: float = 0.08,
        predator_death: float = 0.12,
        predator_conversion: float = 0.5,
        # Population bounds
        pop_min: float = 0.1,
        pop_max: float = 2.0,
        # Cross-domain coupling
        cross_feeding_rate: float = 0.05,
        cross_learning_rate: float = 0.05,
        # Fitness feedback
        fitness_death_modulation: float = 0.15,
        fitness_memory_decay: float = 0.9,
        # Carrying capacity
        prey_capacity: float = 1.5,
        predator_capacity: float = 1.0,
        # Palette composition
        base_prey_slots: int = 2,
        base_predator_slots: int = 1,
        max_palette: int = 6,
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
        """Initialize Predator-Prey Symmetric strategy."""
        # LV parameters
        self.prey_growth = prey_growth
        self.predation_rate = predation_rate
        self.predator_death = predator_death
        self.predator_conversion = predator_conversion

        # Bounds
        self.pop_min = pop_min
        self.pop_max = pop_max

        # Cross-domain
        self.cross_feeding_rate = cross_feeding_rate
        self.cross_learning_rate = cross_learning_rate

        # Fitness feedback
        self.fitness_death_modulation = fitness_death_modulation
        self.fitness_memory_decay = fitness_memory_decay

        # Carrying capacity
        self.prey_capacity = prey_capacity
        self.predator_capacity = predator_capacity

        # Palette composition
        self.base_prey_slots = base_prey_slots
        self.base_predator_slots = base_predator_slots
        self.max_palette = max_palette

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
        """Initialize state with predator-prey populations."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Activation populations
        act_prey_pops = jnp.ones(len(self.ACT_PREY)) * 0.8
        for i, func_idx in enumerate(self.ACT_PREY):
            if func_idx in initial_act:
                act_prey_pops = act_prey_pops.at[i].set(1.2)

        act_pred_pops = jnp.ones(len(self.ACT_PREDATORS)) * 0.3
        for i, func_idx in enumerate(self.ACT_PREDATORS):
            if func_idx in initial_act:
                act_pred_pops = act_pred_pops.at[i].set(0.6)

        # Aggregation populations
        agg_prey_pops = jnp.ones(len(self.AGG_PREY)) * 0.8
        for i, func_idx in enumerate(self.AGG_PREY):
            if func_idx in initial_agg:
                agg_prey_pops = agg_prey_pops.at[i].set(1.2)

        agg_pred_pops = jnp.ones(len(self.AGG_PREDATORS)) * 0.3
        for i, func_idx in enumerate(self.AGG_PREDATORS):
            if func_idx in initial_agg:
                agg_pred_pops = agg_pred_pops.at[i].set(0.6)

        # Neutral fitness
        act_neutral_fit = jnp.zeros(len(self.ACT_NEUTRAL))
        agg_neutral_fit = jnp.zeros(len(self.AGG_NEUTRAL))

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Function affinities (derived from population)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        act_affinity = act_affinity.at[SIN_IDX].set(self.sin_affinity_floor)
        for idx in [MAX_IDX, MIN_IDX]:
            agg_affinity = agg_affinity.at[idx].set(self.extreme_agg_affinity_floor)

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        discovery_gen = {'sin': -1, 'max': -1, 'min': -1}

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Populations
            'act_prey_pops': act_prey_pops,
            'act_pred_pops': act_pred_pops,
            'act_neutral_fit': act_neutral_fit,
            'agg_prey_pops': agg_prey_pops,
            'agg_pred_pops': agg_pred_pops,
            'agg_neutral_fit': agg_neutral_fit,
            # Affinities
            'act_affinity': act_affinity,
            'agg_affinity': agg_affinity,
            'cross_affinity': cross_affinity,
            'predator_success': 0.0,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Phases
            'act_phase': 'prey_rise',
            'agg_phase': 'prey_rise',
            # Discovery
            'discovery_gen': discovery_gen,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 30000),
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

    def _lotka_volterra_step(
        self,
        prey_pops: jnp.ndarray,
        pred_pops: jnp.ndarray,
        predator_success: float,
        cross_energy: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute one Lotka-Volterra step."""
        total_prey = jnp.sum(prey_pops) + 0.01
        total_pred = jnp.sum(pred_pops) + 0.01
        n_prey = len(prey_pops)
        n_pred = len(pred_pops)

        effective_death = self.predator_death * (1.0 - self.fitness_death_modulation * predator_success)

        new_prey = prey_pops.copy()
        new_pred = pred_pops.copy()

        for i in range(n_prey):
            pop = prey_pops[i]
            growth = self.prey_growth * pop * (1 - pop / self.prey_capacity)
            predation = self.predation_rate * pop * total_pred / n_prey
            delta = growth - predation
            new_prey = new_prey.at[i].set(pop + delta)

        for i in range(n_pred):
            pop = pred_pops[i]
            birth = self.predator_conversion * self.predation_rate * total_prey * pop / n_pred
            birth = birth + cross_energy * 0.1
            death = effective_death * pop * (1 + pop / self.predator_capacity)
            delta = birth - death
            new_pred = new_pred.at[i].set(pop + delta)

        return (
            jnp.clip(new_prey, self.pop_min, self.pop_max),
            jnp.clip(new_pred, self.pop_min, self.pop_max)
        )

    def _determine_phase(
        self,
        prey_pops: jnp.ndarray,
        pred_pops: jnp.ndarray,
        prey_list: List[int],
        pred_list: List[int],
    ) -> str:
        """Determine oscillation phase."""
        total_prey = float(jnp.sum(prey_pops))
        total_pred = float(jnp.sum(pred_pops))

        prey_threshold = 0.7 * len(prey_list)
        pred_threshold = 0.5 * len(pred_list)

        if total_prey > prey_threshold and total_pred < pred_threshold:
            return 'prey_high'
        elif total_pred > pred_threshold and total_prey < prey_threshold:
            return 'predator_high'
        elif total_prey > prey_threshold and total_pred > pred_threshold:
            return 'both_high'
        else:
            return 'both_low'

    def _select_palette_from_pops(
        self,
        prey_pops: jnp.ndarray,
        pred_pops: jnp.ndarray,
        neutral_fit: jnp.ndarray,
        prey_list: List[int],
        pred_list: List[int],
        neutral_list: List[int],
        memory_cells: jnp.ndarray,
        n_functions: int,
        max_size: int,
        key: jax.random.PRNGKey,
        protected_indices: List[int],
    ) -> jnp.ndarray:
        """Select palette based on population levels with protection."""
        selected = []

        # CRITICAL: Protected indices are always included
        for idx in protected_indices:
            if idx < n_functions:
                selected.append(idx)

        # Guaranteed prey slots
        prey_sorted = jnp.argsort(prey_pops)[::-1]
        for i in range(min(self.base_prey_slots, len(prey_sorted))):
            func_idx = prey_list[int(prey_sorted[i])]
            if func_idx not in selected:
                selected.append(func_idx)

        # Guaranteed predator slots
        pred_sorted = jnp.argsort(pred_pops)[::-1]
        for i in range(min(self.base_predator_slots, len(pred_sorted))):
            func_idx = pred_list[int(pred_sorted[i])]
            if func_idx not in selected:
                selected.append(func_idx)

        # Memory cells get priority
        for i in range(n_functions):
            if memory_cells[i] and i not in selected and len(selected) < max_size:
                selected.append(i)

        # Fill remaining
        remaining = max_size - len(selected)
        if remaining > 0:
            candidates = []

            for i, func_idx in enumerate(prey_list):
                if func_idx not in selected:
                    candidates.append((func_idx, float(prey_pops[i])))

            for i, func_idx in enumerate(pred_list):
                if func_idx not in selected:
                    candidates.append((func_idx, float(pred_pops[i])))

            for i, func_idx in enumerate(neutral_list):
                if func_idx not in selected and func_idx < n_functions:
                    candidates.append((func_idx, float(neutral_fit[i]) + 0.3))

            candidates.sort(key=lambda x: -x[1])
            for i in range(min(remaining, len(candidates))):
                selected.append(candidates[i][0])

        # Build mask
        mask = jnp.zeros(n_functions)
        for func_idx in selected:
            if 0 <= func_idx < n_functions:
                mask = mask.at[func_idx].set(1.0)

        return mask

    def _update_neutral_fitness(
        self,
        neutral_fit: jnp.ndarray,
        mask: jnp.ndarray,
        neutral_list: List[int],
        improvement: float,
    ) -> jnp.ndarray:
        """Update fitness memory for neutral functions."""
        new_fit = 0.9 * neutral_fit

        for i, func_idx in enumerate(neutral_list):
            if func_idx < len(mask) and mask[func_idx] > 0.5:
                new_fit = new_fit.at[i].add(improvement)

        return jnp.clip(new_fit, -1.0, 1.0)

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        delta = self.cross_learning_rate * fitness_delta * co_active
        return jnp.clip(cross_affinity + delta, 0.0, 1.0)

    def _update_affinities_from_pops(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
        act_prey_pops: jnp.ndarray,
        act_pred_pops: jnp.ndarray,
        agg_prey_pops: jnp.ndarray,
        agg_pred_pops: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update affinities based on population levels."""
        new_act_aff = act_affinity * 0.95

        for i, func_idx in enumerate(self.ACT_PREY):
            if func_idx < NUM_ACTIVATIONS:
                new_act_aff = new_act_aff.at[func_idx].add(0.05 * act_prey_pops[i])

        for i, func_idx in enumerate(self.ACT_PREDATORS):
            if func_idx < NUM_ACTIVATIONS:
                new_act_aff = new_act_aff.at[func_idx].add(0.05 * act_pred_pops[i])

        new_agg_aff = agg_affinity * 0.95

        for i, func_idx in enumerate(self.AGG_PREY):
            if func_idx < NUM_AGGREGATIONS:
                new_agg_aff = new_agg_aff.at[func_idx].add(0.05 * agg_prey_pops[i])

        for i, func_idx in enumerate(self.AGG_PREDATORS):
            if func_idx < NUM_AGGREGATIONS:
                new_agg_aff = new_agg_aff.at[func_idx].add(0.05 * agg_pred_pops[i])

        return jnp.clip(new_act_aff, 0.0, 1.0), jnp.clip(new_agg_aff, 0.0, 1.0)

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
        """Update with predator-prey dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Check predator activity
        act_pred_active = any(
            state['act_mask'][f] > 0.5 for f in self.ACT_PREDATORS if f < NUM_ACTIVATIONS
        )
        agg_pred_active = any(
            state['agg_mask'][f] > 0.5 for f in self.AGG_PREDATORS if f < NUM_AGGREGATIONS
        )

        # Update predator success
        if (act_pred_active or agg_pred_active) and improvement > 0:
            new_pred_success = (
                self.fitness_memory_decay * state['predator_success'] +
                (1 - self.fitness_memory_decay) * min(improvement * 2, 1.0)
            )
        else:
            new_pred_success = self.fitness_memory_decay * state['predator_success']

        # Cross-domain energy
        act_prey_energy = float(jnp.sum(state['act_prey_pops']))
        agg_prey_energy = float(jnp.sum(state['agg_prey_pops']))
        act_to_agg_energy = self.cross_feeding_rate * act_prey_energy
        agg_to_act_energy = self.cross_feeding_rate * agg_prey_energy

        # LV steps
        new_act_prey, new_act_pred = self._lotka_volterra_step(
            state['act_prey_pops'], state['act_pred_pops'],
            new_pred_success, agg_to_act_energy
        )
        new_agg_prey, new_agg_pred = self._lotka_volterra_step(
            state['agg_prey_pops'], state['agg_pred_pops'],
            new_pred_success, act_to_agg_energy
        )

        # Phases
        new_act_phase = self._determine_phase(
            new_act_prey, new_act_pred, self.ACT_PREY, self.ACT_PREDATORS
        )
        new_agg_phase = self._determine_phase(
            new_agg_prey, new_agg_pred, self.AGG_PREY, self.AGG_PREDATORS
        )

        # Neutral fitness
        new_act_neutral = self._update_neutral_fitness(
            state['act_neutral_fit'], state['act_mask'], self.ACT_NEUTRAL, improvement
        )
        new_agg_neutral = self._update_neutral_fitness(
            state['agg_neutral_fit'], state['agg_mask'], self.AGG_NEUTRAL, improvement
        )

        # Cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improvement
        )

        # Update affinities from populations
        new_act_aff, new_agg_aff = self._update_affinities_from_pops(
            state['act_affinity'], state['agg_affinity'],
            new_act_prey, new_act_pred, new_agg_prey, new_agg_pred
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

        # Select palettes with protection
        new_act_mask = self._select_palette_from_pops(
            new_act_prey, new_act_pred, new_act_neutral,
            self.ACT_PREY, self.ACT_PREDATORS, self.ACT_NEUTRAL,
            new_act_mem_cells, NUM_ACTIVATIONS, self.max_palette, k1, [SIN_IDX]
        )
        new_agg_mask = self._select_palette_from_pops(
            new_agg_prey, new_agg_pred, new_agg_neutral,
            self.AGG_PREY, self.AGG_PREDATORS, self.AGG_NEUTRAL,
            new_agg_mem_cells, NUM_AGGREGATIONS, self.max_active_agg, k2, [MAX_IDX, MIN_IDX]
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Get palettes and update discovery
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)
        new_discovery = self._update_discovery_tracking(
            state['discovery_gen'], act_palette, agg_palette, generation
        )

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_prey_pops': new_act_prey,
            'act_pred_pops': new_act_pred,
            'act_neutral_fit': new_act_neutral,
            'agg_prey_pops': new_agg_prey,
            'agg_pred_pops': new_agg_pred,
            'agg_neutral_fit': new_agg_neutral,
            'act_affinity': new_act_aff,
            'agg_affinity': new_agg_aff,
            'cross_affinity': new_cross,
            'predator_success': new_pred_success,
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'act_phase': new_act_phase,
            'agg_phase': new_agg_phase,
            'discovery_gen': new_discovery,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
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
            # Populations
            'act_total_prey': float(jnp.sum(new_act_prey)),
            'act_total_pred': float(jnp.sum(new_act_pred)),
            'agg_total_prey': float(jnp.sum(new_agg_prey)),
            'agg_total_pred': float(jnp.sum(new_agg_pred)),
            # Phases
            'act_phase': new_act_phase,
            'agg_phase': new_agg_phase,
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'predator_success': new_pred_success,
            # Sin status
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
        """Return state summary with predator-prey status."""
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
            'act_phase': state['act_phase'],
            'agg_phase': state['agg_phase'],
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'predator_success': state['predator_success'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
            'discovery_gen': state['discovery_gen'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
