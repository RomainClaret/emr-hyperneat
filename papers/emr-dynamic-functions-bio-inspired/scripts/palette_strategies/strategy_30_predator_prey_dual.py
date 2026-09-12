"""Strategy 30D: Predator-Prey Dual (Ecological Oscillation for Both Domains).

Extends PredatorPreyStrategy to jointly evolve BOTH activation AND aggregation
function palettes using Lotka-Volterra predator-prey dynamics.

Cross-Domain Learning:
- Separate predator/prey populations for both domains
- Cross-domain energy transfer: prey from one domain feeds predators in other
- Coupled oscillations create coordinated exploration patterns
- Shared fitness feedback modulates death rates in both domains

Key Dual Mechanisms:
1. Dual population dynamics - separate prey/predator for act and agg
2. Cross-feeding - successful activations can boost aggregation predators
3. Synchronized oscillation phases - both domains cycle together
4. Shared predator success memory - joint success reduces death rates
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


class PredatorPreyDualStrategy(PaletteEvolutionStrategy):
    """Ecological predator-prey oscillations for dual palette evolution.

    Both activation and aggregation domains have generalist (prey) and
    specialist (predator) functions. Lotka-Volterra dynamics create
    natural exploration cycles. Cross-domain coupling allows
    successful discoveries in one domain to influence the other.
    """

    name = "predator_prey_dual"
    description = "Lotka-Volterra oscillations for both activation and aggregation domains"

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
        # Activation Lotka-Volterra
        act_prey_growth: float = 0.15,
        act_predation_rate: float = 0.08,
        act_predator_death: float = 0.12,
        act_predator_conversion: float = 0.6,
        # Aggregation Lotka-Volterra
        agg_prey_growth: float = 0.12,
        agg_predation_rate: float = 0.10,
        agg_predator_death: float = 0.15,
        agg_predator_conversion: float = 0.5,
        # Population bounds
        pop_min: float = 0.1,
        pop_max: float = 2.0,
        # Cross-domain coupling
        cross_feeding_rate: float = 0.05,  # How much one domain feeds other
        cross_learning_rate: float = 0.05,
        # Fitness feedback
        fitness_death_modulation: float = 0.15,
        fitness_memory_decay: float = 0.9,
        # Carrying capacity
        act_prey_capacity: float = 1.5,
        act_predator_capacity: float = 1.0,
        agg_prey_capacity: float = 1.2,
        agg_predator_capacity: float = 0.8,
        # Palette composition
        act_base_prey_slots: int = 2,
        act_base_predator_slots: int = 1,
        agg_base_prey_slots: int = 1,
        agg_base_predator_slots: int = 1,
        max_act_palette: int = 6,
        max_agg_palette: int = 4,
        # Constraints
        min_active_act: int = 2,
        max_active_act: int = 6,
        min_active_agg: int = 1,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Predator-Prey Dual strategy."""
        # Activation LV
        self.act_prey_growth = act_prey_growth
        self.act_predation_rate = act_predation_rate
        self.act_predator_death = act_predator_death
        self.act_predator_conversion = act_predator_conversion

        # Aggregation LV
        self.agg_prey_growth = agg_prey_growth
        self.agg_predation_rate = agg_predation_rate
        self.agg_predator_death = agg_predator_death
        self.agg_predator_conversion = agg_predator_conversion

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
        self.act_prey_capacity = act_prey_capacity
        self.act_predator_capacity = act_predator_capacity
        self.agg_prey_capacity = agg_prey_capacity
        self.agg_predator_capacity = agg_predator_capacity

        # Palette composition
        self.act_base_prey_slots = act_base_prey_slots
        self.act_base_predator_slots = act_base_predator_slots
        self.agg_base_prey_slots = agg_base_prey_slots
        self.agg_base_predator_slots = agg_base_predator_slots
        self.max_act_palette = max_act_palette
        self.max_agg_palette = max_agg_palette

        # Constraints
        self.min_active_act = min_active_act
        self.max_active_act = max_active_act
        self.min_active_agg = min_active_agg
        self.max_active_agg = max_active_agg

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual predator-prey populations."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        # Activation prey populations
        act_prey_pops = jnp.ones(len(self.ACT_PREY)) * 0.8
        for i, func_idx in enumerate(self.ACT_PREY):
            if func_idx in initial_act:
                act_prey_pops = act_prey_pops.at[i].set(1.2)

        # Activation predator populations
        act_pred_pops = jnp.ones(len(self.ACT_PREDATORS)) * 0.3
        for i, func_idx in enumerate(self.ACT_PREDATORS):
            if func_idx in initial_act:
                act_pred_pops = act_pred_pops.at[i].set(0.6)

        # Activation neutral fitness
        act_neutral_fit = jnp.zeros(len(self.ACT_NEUTRAL))

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_mask = agg_mask.at[i].set(1.0)

        # Aggregation prey populations
        agg_prey_pops = jnp.ones(len(self.AGG_PREY)) * 0.8
        for i, func_idx in enumerate(self.AGG_PREY):
            if func_idx in initial_agg:
                agg_prey_pops = agg_prey_pops.at[i].set(1.2)

        # Aggregation predator populations
        agg_pred_pops = jnp.ones(len(self.AGG_PREDATORS)) * 0.3
        for i, func_idx in enumerate(self.AGG_PREDATORS):
            if func_idx in initial_agg:
                agg_pred_pops = agg_pred_pops.at[i].set(0.6)

        # Aggregation neutral fitness
        agg_neutral_fit = jnp.zeros(len(self.AGG_NEUTRAL))

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation populations
            'act_mask': act_mask,
            'act_prey_pops': act_prey_pops,
            'act_pred_pops': act_pred_pops,
            'act_neutral_fit': act_neutral_fit,
            # Aggregation populations
            'agg_mask': agg_mask,
            'agg_prey_pops': agg_prey_pops,
            'agg_pred_pops': agg_pred_pops,
            'agg_neutral_fit': agg_neutral_fit,
            # Cross-domain
            'cross_affinity': cross_affinity,
            'predator_success': 0.0,  # Shared success memory
            # Tracking
            'act_phase': 'prey_rise',
            'agg_phase': 'prey_rise',
            # General state
            'rng_key': jax.random.PRNGKey(seed + 303030),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return [i for i in range(NUM_AGGREGATIONS) if state['agg_mask'][i] > 0.5]

    def _lotka_volterra_step(
        self,
        prey_pops: jnp.ndarray,
        pred_pops: jnp.ndarray,
        predator_success: float,
        prey_growth: float,
        predation_rate: float,
        pred_death: float,
        pred_conversion: float,
        prey_capacity: float,
        pred_capacity: float,
        cross_energy: float,  # Energy from other domain
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute one Lotka-Volterra step for a domain."""
        total_prey = jnp.sum(prey_pops) + 0.01
        total_pred = jnp.sum(pred_pops) + 0.01
        n_prey = len(prey_pops)
        n_pred = len(pred_pops)

        # Modify death rate based on success
        effective_death = pred_death * (1.0 - self.fitness_death_modulation * predator_success)

        new_prey = prey_pops.copy()
        new_pred = pred_pops.copy()

        # Prey dynamics
        for i in range(n_prey):
            pop = prey_pops[i]
            growth = prey_growth * pop * (1 - pop / prey_capacity)
            predation = predation_rate * pop * total_pred / n_prey
            delta = growth - predation
            new_prey = new_prey.at[i].set(pop + delta)

        # Predator dynamics (with cross-domain energy boost)
        for i in range(n_pred):
            pop = pred_pops[i]
            birth = pred_conversion * predation_rate * total_prey * pop / n_pred
            birth = birth + cross_energy * 0.1  # Cross-domain energy boost
            death = effective_death * pop * (1 + pop / pred_capacity)
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
        base_prey_slots: int,
        base_pred_slots: int,
        max_size: int,
        n_functions: int,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette based on population levels."""
        selected = []

        # Guaranteed prey slots
        prey_sorted = jnp.argsort(prey_pops)[::-1]
        for i in range(min(base_prey_slots, len(prey_sorted))):
            func_idx = prey_list[int(prey_sorted[i])]
            if func_idx not in selected:
                selected.append(func_idx)

        # Guaranteed predator slots
        pred_sorted = jnp.argsort(pred_pops)[::-1]
        for i in range(min(base_pred_slots, len(pred_sorted))):
            func_idx = pred_list[int(pred_sorted[i])]
            if func_idx not in selected:
                selected.append(func_idx)

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

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual predator-prey dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Check if predators active in either domain
        act_pred_active = any(
            state['act_mask'][f] > 0.5 for f in self.ACT_PREDATORS if f < NUM_ACTIVATIONS
        )
        agg_pred_active = any(
            state['agg_mask'][f] > 0.5 for f in self.AGG_PREDATORS if f < NUM_AGGREGATIONS
        )

        # Update shared predator success
        if (act_pred_active or agg_pred_active) and improvement > 0:
            new_pred_success = (
                self.fitness_memory_decay * state['predator_success'] +
                (1 - self.fitness_memory_decay) * min(improvement * 2, 1.0)
            )
        else:
            new_pred_success = self.fitness_memory_decay * state['predator_success']

        # Compute cross-domain energy transfer
        act_prey_energy = float(jnp.sum(state['act_prey_pops']))
        agg_prey_energy = float(jnp.sum(state['agg_prey_pops']))
        act_to_agg_energy = self.cross_feeding_rate * act_prey_energy
        agg_to_act_energy = self.cross_feeding_rate * agg_prey_energy

        # Step 1: Lotka-Volterra for activation domain
        new_act_prey, new_act_pred = self._lotka_volterra_step(
            state['act_prey_pops'], state['act_pred_pops'], new_pred_success,
            self.act_prey_growth, self.act_predation_rate,
            self.act_predator_death, self.act_predator_conversion,
            self.act_prey_capacity, self.act_predator_capacity,
            agg_to_act_energy
        )

        # Step 2: Lotka-Volterra for aggregation domain
        new_agg_prey, new_agg_pred = self._lotka_volterra_step(
            state['agg_prey_pops'], state['agg_pred_pops'], new_pred_success,
            self.agg_prey_growth, self.agg_predation_rate,
            self.agg_predator_death, self.agg_predator_conversion,
            self.agg_prey_capacity, self.agg_predator_capacity,
            act_to_agg_energy
        )

        # Step 3: Determine phases
        new_act_phase = self._determine_phase(
            new_act_prey, new_act_pred, self.ACT_PREY, self.ACT_PREDATORS
        )
        new_agg_phase = self._determine_phase(
            new_agg_prey, new_agg_pred, self.AGG_PREY, self.AGG_PREDATORS
        )

        # Step 4: Update neutral fitness
        new_act_neutral = self._update_neutral_fitness(
            state['act_neutral_fit'], state['act_mask'],
            self.ACT_NEUTRAL, improvement
        )
        new_agg_neutral = self._update_neutral_fitness(
            state['agg_neutral_fit'], state['agg_mask'],
            self.AGG_NEUTRAL, improvement
        )

        # Step 5: Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improvement
        )

        # Step 6: Select palettes
        new_act_mask = self._select_palette_from_pops(
            new_act_prey, new_act_pred, new_act_neutral,
            self.ACT_PREY, self.ACT_PREDATORS, self.ACT_NEUTRAL,
            self.act_base_prey_slots, self.act_base_predator_slots,
            self.max_act_palette, NUM_ACTIVATIONS, k1
        )
        new_agg_mask = self._select_palette_from_pops(
            new_agg_prey, new_agg_pred, new_agg_neutral,
            self.AGG_PREY, self.AGG_PREDATORS, self.AGG_NEUTRAL,
            self.agg_base_prey_slots, self.agg_base_predator_slots,
            self.max_agg_palette, NUM_AGGREGATIONS, k2
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        new_state = {
            # Activation populations
            'act_mask': new_act_mask,
            'act_prey_pops': new_act_prey,
            'act_pred_pops': new_act_pred,
            'act_neutral_fit': new_act_neutral,
            # Aggregation populations
            'agg_mask': new_agg_mask,
            'agg_prey_pops': new_agg_prey,
            'agg_pred_pops': new_agg_pred,
            'agg_neutral_fit': new_agg_neutral,
            # Cross-domain
            'cross_affinity': new_cross,
            'predator_success': new_pred_success,
            # Tracking
            'act_phase': new_act_phase,
            'agg_phase': new_agg_phase,
            # General state
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': (state['fitness_history'] + [best_fitness])[-20:],
        }

        # Compute metrics
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = self.get_active_agg_palette(new_state)

        # Composition stats
        act_prey_in = sum(1 for f in self.ACT_PREY if f in act_palette)
        act_pred_in = sum(1 for f in self.ACT_PREDATORS if f in act_palette)
        agg_prey_in = sum(1 for f in self.AGG_PREY if f in agg_palette)
        agg_pred_in = sum(1 for f in self.AGG_PREDATORS if f in agg_palette)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Activation populations
            'act_total_prey': float(jnp.sum(new_act_prey)),
            'act_total_pred': float(jnp.sum(new_act_pred)),
            'act_prey_in_palette': act_prey_in,
            'act_pred_in_palette': act_pred_in,
            'act_phase': new_act_phase,
            # Aggregation populations
            'agg_total_prey': float(jnp.sum(new_agg_prey)),
            'agg_total_pred': float(jnp.sum(new_agg_pred)),
            'agg_prey_in_palette': agg_prey_in,
            'agg_pred_in_palette': agg_pred_in,
            'agg_phase': new_agg_phase,
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'predator_success': new_pred_success,
            # Sin status (sin is predator)
            'has_sin': 4 in act_palette,
            'sin_population': float(new_act_pred[0]) if 4 in self.ACT_PREDATORS else 0.0,
            # Agg4 status
            'has_agg4': len(agg_palette) >= 4,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual predator-prey status."""
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
            # Activation
            'act_total_prey': float(jnp.sum(state['act_prey_pops'])),
            'act_total_pred': float(jnp.sum(state['act_pred_pops'])),
            'act_phase': state['act_phase'],
            # Aggregation
            'agg_total_prey': float(jnp.sum(state['agg_prey_pops'])),
            'agg_total_pred': float(jnp.sum(state['agg_pred_pops'])),
            'agg_phase': state['agg_phase'],
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
            'predator_success': state['predator_success'],
        }
