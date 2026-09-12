"""Strategy 30: Predator-Prey (Ecological Oscillation Dynamics).

Implements Lotka-Volterra predator-prey dynamics for palette evolution.
Generalist functions are "prey" (always useful baseline), specialist
functions are "predators" (context-dependent, high-risk high-reward).

Biological Basis:
- Predator-prey populations oscillate: high prey → more predators →
  fewer prey → fewer predators → prey recovery → repeat
- Creates sustained cycling that prevents stable but suboptimal states
- Natural boom-bust dynamics for exploration

Key Insight:
- Evolution can get stuck in local optima with stable function sets
- Predator-prey oscillations prevent premature convergence
- Generalists provide baseline stability
- Specialists rise and fall, enabling exploration without catastrophe

Lotka-Volterra Dynamics:
    # Population dynamics
    d_prey = prey_growth * prey - predation_rate * prey * predator
    d_predator = predation_rate * prey * predator - predator_death * predator

    # Update populations
    prey += d_prey
    predator += d_predator

    # Palette cycles between:
    # 1. High prey (generalists dominate) → predators grow
    # 2. High predator (specialists dominate) → prey declines
    # 3. Low prey → predators decline
    # 4. Low predator → prey recovers
    # → Repeat

    # Fitness feedback modulates death rates
    if fitness improving with predators:
        predator_death *= 0.9  # Predators thrive longer

Expected improvements:
- Prevents premature convergence to suboptimal function sets
- Natural exploration cycles (specialists rise/fall)
- Stable baseline (generalists always present)
- Self-correcting dynamics (no manual tuning needed)
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


class PredatorPreyStrategy(PaletteEvolutionStrategy):
    """Ecological predator-prey oscillation dynamics for function selection.

    Generalist functions (prey) provide stable baseline. Specialist functions
    (predators) rise and fall through Lotka-Volterra dynamics. Fitness
    feedback modulates predator success, creating adaptive exploration.
    """

    name = "predator_prey"
    description = "Lotka-Volterra oscillations between generalist and specialist functions"

    # Function classifications
    DEFAULT_PREY = [0, 1, 2, 3, 5, 6]  # identity, tanh, sigmoid, relu, step, leaky_relu
    DEFAULT_PREDATORS = [4, 11, 12, 13, 15]  # sin, burst, resonator, osc_adapt, receptive
    DEFAULT_NEUTRAL = [7, 8, 9, 10, 14, 16, 17]  # gaussian, softplus, elu, swish, etc.

    def __init__(
        self,
        # Function classifications
        prey_functions: List[int] = None,
        predator_functions: List[int] = None,
        neutral_functions: List[int] = None,
        # Lotka-Volterra parameters
        prey_growth_rate: float = 0.4,       # Prey natural reproduction
        predation_rate: float = 0.5,         # Predator consumption of prey
        predator_death_rate: float = 0.3,    # Predator natural death
        predator_conversion: float = 0.5,    # Prey energy → predator offspring
        # Population bounds
        pop_min: float = 0.1,
        pop_max: float = 2.0,
        # Fitness feedback
        fitness_death_modulation: float = 0.15,  # How much fitness affects death rate
        fitness_memory_decay: float = 0.9,       # Memory of past fitness with predators
        # Carrying capacity
        prey_carrying_capacity: float = 1.5,
        predator_carrying_capacity: float = 1.0,
        # Palette composition
        base_prey_slots: int = 0,     # No guaranteed slots (proportional)
        base_predator_slots: int = 0,  # No guaranteed slots (proportional)
        max_palette_size: int = 6,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Predator-Prey strategy.

        Args:
            prey_functions: Generalist function indices (stable baseline)
            predator_functions: Specialist function indices (high variance)
            neutral_functions: Functions not in predator-prey dynamics
            prey_growth_rate: r in Lotka-Volterra
            predation_rate: α in Lotka-Volterra
            predator_death_rate: q in Lotka-Volterra
            predator_conversion: Efficiency of prey → predator
            pop_min: Minimum population (prevents extinction)
            pop_max: Maximum population (carrying capacity limit)
            fitness_death_modulation: How much success reduces predator death
            fitness_memory_decay: EMA decay for predator success memory
            prey_carrying_capacity: K for prey logistic growth
            predator_carrying_capacity: K for predator capacity
            base_prey_slots: Guaranteed prey in palette
            base_predator_slots: Guaranteed predator exploration
            max_palette_size: Total palette size
        """
        # Classifications
        self.prey_functions = prey_functions or self.DEFAULT_PREY
        self.predator_functions = predator_functions or self.DEFAULT_PREDATORS
        self.neutral_functions = neutral_functions or self.DEFAULT_NEUTRAL

        # Lotka-Volterra
        self.prey_growth_rate = prey_growth_rate
        self.predation_rate = predation_rate
        self.predator_death_rate = predator_death_rate
        self.predator_conversion = predator_conversion

        # Bounds
        self.pop_min = pop_min
        self.pop_max = pop_max

        # Fitness feedback
        self.fitness_death_modulation = fitness_death_modulation
        self.fitness_memory_decay = fitness_memory_decay

        # Carrying capacity
        self.prey_carrying_capacity = prey_carrying_capacity
        self.predator_carrying_capacity = predator_carrying_capacity

        # Palette
        self.base_prey_slots = base_prey_slots
        self.base_predator_slots = base_predator_slots
        self.max_palette_size = max_palette_size

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

        # Build function population arrays
        self._build_population_arrays()

    def _build_population_arrays(self):
        """Build arrays for individual function populations."""
        self.n_prey = len(self.prey_functions)
        self.n_predators = len(self.predator_functions)
        self.n_neutral = len(self.neutral_functions)

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with predator-prey populations."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize prey populations (per-function)
        prey_pops = jnp.ones(self.n_prey) * 0.8  # Start healthy
        for i, func_idx in enumerate(self.prey_functions):
            if func_idx in initial:
                prey_pops = prey_pops.at[i].set(1.2)

        # Initialize predator populations
        predator_pops = jnp.ones(self.n_predators) * 0.3  # Start low
        for i, func_idx in enumerate(self.predator_functions):
            if func_idx in initial:
                predator_pops = predator_pops.at[i].set(0.6)

        # Neutral function fitnesses (simple tracking)
        neutral_fitness = jnp.zeros(self.n_neutral)

        # Predator success memory (fitness improvement when predators active)
        predator_success = 0.0

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 303030),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Predator-prey state
            'prey_populations': prey_pops,
            'predator_populations': predator_pops,
            'neutral_fitness': neutral_fitness,
            'predator_success': predator_success,
            # Tracking
            'cycle_phase': 'prey_rise',  # Current oscillation phase
            'phase_generations': 0,
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_lotka_volterra_step(
        self,
        prey_pops: jnp.ndarray,
        predator_pops: jnp.ndarray,
        predator_success: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute one step of aggregate 2-species Lotka-Volterra dynamics.

        Classic LV on aggregate populations (mean prey P, mean predator Q):
            dP = rP - αPQ
            dQ = βαPQ - qQ

        Individual populations are then scaled proportionally from the
        aggregate change. No logistic growth (dampens oscillations).

        Fitness feedback modulates predator death rate.
        """
        # Aggregate populations (mean per-function)
        mean_prey = float(jnp.mean(prey_pops))
        mean_pred = float(jnp.mean(predator_pops))

        # Modify predator death based on success memory
        effective_death_rate = self.predator_death_rate * (
            1.0 - self.fitness_death_modulation * predator_success
        )

        # Classic Lotka-Volterra on aggregate populations
        d_prey = self.prey_growth_rate * mean_prey - self.predation_rate * mean_prey * mean_pred
        d_pred = (self.predator_conversion * self.predation_rate * mean_prey * mean_pred
                  - effective_death_rate * mean_pred)

        # Compute scale factors for individual populations
        if mean_prey > 1e-6:
            prey_scale = (mean_prey + d_prey) / mean_prey
        else:
            prey_scale = 1.0

        if mean_pred > 1e-6:
            pred_scale = (mean_pred + d_pred) / mean_pred
        else:
            pred_scale = 1.0

        # Scale individual populations proportionally
        new_prey = prey_pops * prey_scale
        new_predator = predator_pops * pred_scale

        # Clip to valid range
        new_prey = jnp.clip(new_prey, self.pop_min, self.pop_max)
        new_predator = jnp.clip(new_predator, self.pop_min, self.pop_max)

        return new_prey, new_predator

    def _determine_phase(
        self,
        prey_pops: jnp.ndarray,
        predator_pops: jnp.ndarray,
        prev_phase: str,
    ) -> str:
        """Determine current oscillation phase."""
        total_prey = float(jnp.sum(prey_pops))
        total_predator = float(jnp.sum(predator_pops))

        prey_threshold = 0.7 * len(self.prey_functions)
        predator_threshold = 0.5 * len(self.predator_functions)

        if total_prey > prey_threshold and total_predator < predator_threshold:
            return 'prey_high'  # Prey dominating
        elif total_predator > predator_threshold and total_prey < prey_threshold:
            return 'predator_high'  # Predator dominating
        elif total_prey > prey_threshold and total_predator > predator_threshold:
            return 'both_high'  # Both high (transition)
        else:
            return 'both_low'  # Both low (recovery)

    def _select_palette_from_populations(
        self,
        prey_pops: jnp.ndarray,
        predator_pops: jnp.ndarray,
        neutral_fitness: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette via proportional slot allocation from populations.

        Prey/predator share of palette slots is determined by their relative
        mean populations. Within each group, top-K by population are selected.
        No guaranteed slots, composition reflects actual LV dynamics.
        """
        mean_prey = float(jnp.mean(prey_pops))
        mean_pred = float(jnp.mean(predator_pops))

        total = mean_prey + mean_pred + 1e-8
        prey_share = mean_prey / total

        # Proportional allocation, clamped to [1, max-1] so both groups present
        prey_slots = max(1, min(self.max_palette_size - 1,
                                round(prey_share * self.max_palette_size)))
        pred_slots = self.max_palette_size - prey_slots

        selected = []

        # Fill prey slots with top-K prey by population
        prey_sorted = jnp.argsort(prey_pops)[::-1]
        for i in range(min(prey_slots, len(prey_sorted))):
            func_idx = self.prey_functions[int(prey_sorted[i])]
            if func_idx not in selected:
                selected.append(func_idx)

        # Fill predator slots with top-K predators by population
        predator_sorted = jnp.argsort(predator_pops)[::-1]
        for i in range(min(pred_slots, len(predator_sorted))):
            func_idx = self.predator_functions[int(predator_sorted[i])]
            if func_idx not in selected:
                selected.append(func_idx)

        # If we haven't filled max_palette_size (unlikely but safe), fill remaining
        remaining = self.max_palette_size - len(selected)
        if remaining > 0:
            # Add any remaining prey or predator by population
            all_candidates = []
            for i, func_idx in enumerate(self.prey_functions):
                if func_idx not in selected:
                    all_candidates.append((func_idx, float(prey_pops[i])))
            for i, func_idx in enumerate(self.predator_functions):
                if func_idx not in selected:
                    all_candidates.append((func_idx, float(predator_pops[i])))
            all_candidates.sort(key=lambda x: -x[1])
            for i in range(min(remaining, len(all_candidates))):
                selected.append(all_candidates[i][0])

        # Build mask
        mask = jnp.zeros(NUM_ACTIVATIONS)
        for func_idx in selected:
            mask = mask.at[func_idx].set(1.0)

        return mask

    def _update_neutral_fitness(
        self,
        neutral_fitness: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update fitness memory for neutral functions."""
        new_fitness = 0.9 * neutral_fitness

        for i, func_idx in enumerate(self.neutral_functions):
            if mask[func_idx] > 0.5:
                new_fitness = new_fitness.at[i].add(improvement)

        return jnp.clip(new_fitness, -1.0, 1.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with predator-prey dynamics."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Check if predators were active this generation
        predator_active = any(
            state['mask'][func_idx] > 0.5
            for func_idx in self.predator_functions
        )

        # Update predator success memory
        if predator_active and improvement > 0:
            new_predator_success = (
                self.fitness_memory_decay * state['predator_success'] +
                (1 - self.fitness_memory_decay) * min(improvement * 2, 1.0)
            )
        else:
            new_predator_success = self.fitness_memory_decay * state['predator_success']

        # Step 1: Lotka-Volterra dynamics
        new_prey, new_predator = self._compute_lotka_volterra_step(
            state['prey_populations'],
            state['predator_populations'],
            new_predator_success,
        )

        # Step 2: Determine oscillation phase
        new_phase = self._determine_phase(
            new_prey, new_predator, state['cycle_phase']
        )
        phase_changed = new_phase != state['cycle_phase']
        new_phase_gens = 0 if phase_changed else state['phase_generations'] + 1

        # Step 3: Update neutral function fitness
        new_neutral = self._update_neutral_fitness(
            state['neutral_fitness'],
            state['mask'],
            improvement,
        )

        # Step 4: Select palette based on populations
        new_mask = self._select_palette_from_populations(
            new_prey, new_predator, new_neutral, subkey
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

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
            # Predator-prey state
            'prey_populations': new_prey,
            'predator_populations': new_predator,
            'neutral_fitness': new_neutral,
            'predator_success': new_predator_success,
            # Tracking
            'cycle_phase': new_phase,
            'phase_generations': new_phase_gens,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Population stats
        prey_in_palette = sum(1 for f in self.prey_functions if f in active_palette)
        predator_in_palette = sum(1 for f in self.predator_functions if f in active_palette)
        neutral_in_palette = sum(1 for f in self.neutral_functions if f in active_palette)

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Populations
            'total_prey_pop': float(jnp.sum(new_prey)),
            'total_predator_pop': float(jnp.sum(new_predator)),
            'prey_pop_range': (float(jnp.min(new_prey)), float(jnp.max(new_prey))),
            'predator_pop_range': (float(jnp.min(new_predator)), float(jnp.max(new_predator))),
            # Composition
            'prey_in_palette': prey_in_palette,
            'predator_in_palette': predator_in_palette,
            'neutral_in_palette': neutral_in_palette,
            # Dynamics
            'cycle_phase': new_phase,
            'phase_changed': phase_changed,
            'predator_success': new_predator_success,
            # Sin status (sin is predator index 0)
            'has_sin': 4 in active_palette,
            'sin_population': float(new_predator[0]) if 4 in self.predator_functions else 0.0,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with predator-prey status."""
        palette = self.get_active_palette(state)
        prey_pops = state['prey_populations']
        predator_pops = state['predator_populations']

        # Top prey and predators
        top_prey = [(self.prey_functions[i], float(prey_pops[i]))
                    for i in jnp.argsort(prey_pops)[-3:][::-1]]
        top_predators = [(self.predator_functions[i], float(predator_pops[i]))
                         for i in jnp.argsort(predator_pops)[-3:][::-1]]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Populations
            'total_prey': float(jnp.sum(prey_pops)),
            'total_predator': float(jnp.sum(predator_pops)),
            'top_prey': top_prey,
            'top_predators': top_predators,
            # Dynamics
            'cycle_phase': state['cycle_phase'],
            'phase_generations': state['phase_generations'],
            'predator_success': state['predator_success'],
            # Sin-specific
            'sin_population': float(predator_pops[0]) if 4 in self.predator_functions else 0.0,
        }
