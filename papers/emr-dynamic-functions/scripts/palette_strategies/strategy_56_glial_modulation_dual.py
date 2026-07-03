"""Strategy 56D: Glial Modulation Dual (Energy Constraints for Both Domains).

Extends GlialModulationStrategy to jointly evolve BOTH activation AND aggregation
function palettes using astrocyte-inspired metabolic constraints.

Key dual mechanisms:
1. Shared energy pool - both domains compete for limited energy
2. Dual glial support - separate support tracking for act and agg
3. Cross-domain support transfer - success in one boosts support in other
4. Coordinated starvation response - both domains affected during energy crisis

Expected: Metabolic efficiency pressure in both domains
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
)


class GlialModulationDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with energy-based metabolic constraints.

    Both activation and aggregation functions consume energy and compete
    for glial support. Energy starvation triggers prioritization in both domains.
    """

    name = "glial_modulation_dual"
    description = "Dual: Metabolic constraints with astrocyte-like energy management"

    def __init__(
        self,
        # Energy pool dynamics
        initial_energy: float = 120.0,
        energy_max: float = 180.0,
        energy_regen_rate: float = 18.0,
        # Function costs
        act_base_cost: float = 1.5,
        agg_base_cost: float = 1.0,
        complexity_cost_factor: float = 0.05,
        # Starvation thresholds
        starvation_threshold: float = 25.0,
        critical_threshold: float = 10.0,
        # Glial support
        initial_support: float = 0.4,
        support_learning_rate: float = 0.12,
        support_decay: float = 0.99,
        support_max: float = 1.5,
        support_threshold: float = 0.4,
        # Cross-domain support transfer
        cross_support_rate: float = 0.1,
        # Selection
        base_selection_weight: float = 1.0,
        support_weight_factor: float = 0.5,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        restricted_act_size: int = 4,
        restricted_agg_size: int = 2,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Glial Modulation Dual strategy."""
        # Energy
        self.initial_energy = initial_energy
        self.energy_max = energy_max
        self.energy_regen_rate = energy_regen_rate

        # Costs
        self.act_base_cost = act_base_cost
        self.agg_base_cost = agg_base_cost
        self.complexity_cost_factor = complexity_cost_factor

        # Thresholds
        self.starvation_threshold = starvation_threshold
        self.critical_threshold = critical_threshold

        # Glial support
        self.initial_support = initial_support
        self.support_learning_rate = support_learning_rate
        self.support_decay = support_decay
        self.support_max = support_max
        self.support_threshold = support_threshold

        # Cross-domain
        self.cross_support_rate = cross_support_rate

        # Selection
        self.base_selection_weight = base_selection_weight
        self.support_weight_factor = support_weight_factor

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.restricted_act_size = restricted_act_size
        self.restricted_agg_size = restricted_agg_size
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, 4]
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with energy and dual glial support."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize glial support for both domains
        act_support = jnp.ones(NUM_ACTIVATIONS) * self.initial_support
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_support = act_support.at[i].set(self.initial_support * 1.5)

        agg_support = jnp.ones(NUM_AGGREGATIONS) * self.initial_support
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_support = agg_support.at[i].set(self.initial_support * 1.5)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_support': act_support,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_support': agg_support,
            # Energy state
            'energy_pool': self.initial_energy,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 565656),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Tracking
            'starvation_events': 0,
            'critical_events': 0,
            'total_energy_consumed': 0.0,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_total_cost(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> float:
        """Compute total energy cost for both palettes."""
        total = 0.0
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                total += self.act_base_cost + i * self.complexity_cost_factor
        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                total += self.agg_base_cost + i * self.complexity_cost_factor
        return total

    def _update_energy(
        self,
        energy: float,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> Tuple[float, float, str]:
        """Update shared energy pool."""
        consumed = self._compute_total_cost(act_mask, agg_mask)
        new_energy = energy - consumed + self.energy_regen_rate
        new_energy = min(max(new_energy, 0.0), self.energy_max)

        if new_energy < self.critical_threshold:
            state = 'critical'
        elif new_energy < self.starvation_threshold:
            state = 'starvation'
        else:
            state = 'normal'

        return new_energy, consumed, state

    def _update_support(
        self,
        support: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        improvement: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update glial support for a domain."""
        new_support = support * self.support_decay

        if improved:
            for i in range(n_funcs):
                if mask[i] > 0.5:
                    delta = self.support_learning_rate * max(improvement, 0.1)
                    new_support = new_support.at[i].set(
                        min(float(new_support[i]) + delta, self.support_max)
                    )

        return jnp.maximum(new_support, 0.01)

    def _apply_cross_support(
        self,
        act_support: jnp.ndarray,
        agg_support: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply cross-domain support transfer on success."""
        if improvement <= 0:
            return act_support, agg_support

        new_act_support = act_support.copy()
        new_agg_support = agg_support.copy()

        # Mean support of active functions in each domain
        act_active_support = sum(float(act_support[i]) for i in range(NUM_ACTIVATIONS) if act_mask[i] > 0.5)
        agg_active_support = sum(float(agg_support[i]) for i in range(NUM_AGGREGATIONS) if agg_mask[i] > 0.5)

        n_act_active = max(sum(1 for i in range(NUM_ACTIVATIONS) if act_mask[i] > 0.5), 1)
        n_agg_active = max(sum(1 for i in range(NUM_AGGREGATIONS) if agg_mask[i] > 0.5), 1)

        act_mean = act_active_support / n_act_active
        agg_mean = agg_active_support / n_agg_active

        # Transfer boost based on partner domain's support
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                delta = self.cross_support_rate * agg_mean * improvement
                new_act_support = new_act_support.at[i].set(
                    min(float(new_act_support[i]) + delta, self.support_max)
                )

        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                delta = self.cross_support_rate * act_mean * improvement
                new_agg_support = new_agg_support.at[i].set(
                    min(float(new_agg_support[i]) + delta, self.support_max)
                )

        return new_act_support, new_agg_support

    def _compute_selection_weights(
        self,
        support: jnp.ndarray,
        energy_state: str,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute selection weights based on support and energy."""
        weights = jnp.ones(n_funcs) * self.base_selection_weight
        weights = weights + support * self.support_weight_factor

        if energy_state in ['starvation', 'critical']:
            for i in range(n_funcs):
                if support[i] < self.support_threshold:
                    penalty_factor = 0.3 if energy_state == 'critical' else 0.5
                    weights = weights.at[i].multiply(penalty_factor)

        return jnp.maximum(weights, 0.05)

    def _select_palette(
        self,
        weights: jnp.ndarray,
        energy_state: str,
        target_size: int,
        restricted_size: int,
        min_active: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Select palette based on weights and energy state."""
        if energy_state == 'critical':
            size = max(min_active, restricted_size - 1)
        elif energy_state == 'starvation':
            size = restricted_size
        else:
            size = target_size

        top_indices = jnp.argsort(weights)[-size:]
        new_mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            new_mask = new_mask.at[int(idx)].set(1.0)

        return new_mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual glial modulation dynamics."""
        key = jax.random.split(state['rng_key'])[0]

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update energy
        new_energy, consumed, energy_state = self._update_energy(
            state['energy_pool'], state['act_mask'], state['agg_mask']
        )

        starvation_events = state['starvation_events']
        critical_events = state['critical_events']
        if energy_state == 'starvation':
            starvation_events += 1
        elif energy_state == 'critical':
            critical_events += 1

        # Update glial support for both domains
        new_act_support = self._update_support(
            state['act_support'], state['act_mask'],
            improved, improvement, NUM_ACTIVATIONS
        )
        new_agg_support = self._update_support(
            state['agg_support'], state['agg_mask'],
            improved, improvement, NUM_AGGREGATIONS
        )

        # Apply cross-domain support transfer
        new_act_support, new_agg_support = self._apply_cross_support(
            new_act_support, new_agg_support,
            state['act_mask'], state['agg_mask'],
            improvement
        )

        # Compute selection weights
        act_weights = self._compute_selection_weights(
            new_act_support, energy_state, NUM_ACTIVATIONS
        )
        agg_weights = self._compute_selection_weights(
            new_agg_support, energy_state, NUM_AGGREGATIONS
        )

        # Select palettes
        new_act_mask = self._select_palette(
            act_weights, energy_state,
            self.act_palette_size, self.restricted_act_size,
            self.min_active_act, NUM_ACTIVATIONS
        )
        new_agg_mask = self._select_palette(
            agg_weights, energy_state,
            self.agg_palette_size, self.restricted_agg_size,
            self.min_active_agg, NUM_AGGREGATIONS
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_support': new_act_support,
            'agg_mask': new_agg_mask,
            'agg_support': new_agg_support,
            'energy_pool': new_energy,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'starvation_events': starvation_events,
            'critical_events': critical_events,
            'total_energy_consumed': state['total_energy_consumed'] + consumed,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Energy
            'energy_pool': new_energy,
            'energy_consumed': consumed,
            'energy_state': energy_state,
            'is_starving': energy_state in ['starvation', 'critical'],
            # Support
            'act_mean_support': float(jnp.mean(new_act_support)),
            'agg_mean_support': float(jnp.mean(new_agg_support)),
            'act_max_support': float(jnp.max(new_act_support)),
            'agg_max_support': float(jnp.max(new_agg_support)),
            # Events
            'total_starvation_events': starvation_events,
            'total_critical_events': critical_events,
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_support': float(new_act_support[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual glial status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        energy = state['energy_pool']

        if energy < self.critical_threshold:
            energy_state = 'critical'
        elif energy < self.starvation_threshold:
            energy_state = 'starvation'
        else:
            energy_state = 'normal'

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'energy_pool': energy,
            'energy_state': energy_state,
            'act_mean_support': float(jnp.mean(state['act_support'])),
            'agg_mean_support': float(jnp.mean(state['agg_support'])),
            'total_starvation_events': state['starvation_events'],
            'total_critical_events': state['critical_events'],
            'total_energy_consumed': state['total_energy_consumed'],
        }
