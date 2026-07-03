"""Strategy 38: Glial Modulation (Energy-Based Constraints).

Implements astrocyte-inspired metabolic constraints for palette evolution.
Functions consume energy when active, and a global energy pool must regenerate.
When energy is low, only functions with high glial support remain active.

Biological Basis:
- Astrocytes regulate synaptic strength and energy supply
- Neural activity requires glucose/ATP, managed by glial cells
- High-demand neurons get more glial support
- Energy starvation forces prioritization of essential circuits

Key Insight:
- Current strategies don't model resource competition
- Glial modulation creates metabolic constraints
- Functions compete for limited energy resources
- Energy starvation naturally selects most important functions

Glial Mechanism:
    # Each active function consumes energy
    energy_consumed = sum(function_cost[f] for f in active_palette)
    energy_pool -= energy_consumed

    # Energy regenerates each generation
    energy_pool += regeneration_rate

    # Glial support grows for successful functions
    if fitness_improved:
        for f in active_palette:
            glial_support[f] += support_learning_rate

    # When energy is low, only supported functions work
    if energy_pool < starvation_threshold:
        available_functions = filter(f: glial_support[f] > support_threshold)

Expected improvements:
- Metabolic efficiency pressure (fewer, better functions)
- Natural priority queue based on glial support
- Energy starvation triggers essential-only mode
- Self-organizing resource allocation
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


class GlialModulationStrategy(PaletteEvolutionStrategy):
    """Energy-based constraints with astrocyte-like support.

    Functions consume energy from a global pool. Glial support determines
    which functions are prioritized when energy is scarce. Creates natural
    resource competition and prioritization.
    """

    name = "glial_modulation"
    description = "Metabolic constraints with astrocyte-like energy management"

    def __init__(
        self,
        # Energy pool dynamics
        initial_energy: float = 100.0,          # Starting energy pool
        energy_max: float = 150.0,              # Maximum energy cap
        energy_regen_rate: float = 15.0,        # Energy regenerated per gen (FIXED: was 8.0)
        # Function costs
        base_function_cost: float = 1.5,        # Base cost per active function (FIXED: was 2.5)
        complexity_cost_factor: float = 0.05,   # Extra cost based on function index (FIXED: was 0.1)
        # Starvation thresholds
        starvation_threshold: float = 20.0,     # Below this: restricted mode (FIXED: was 25.0)
        critical_threshold: float = 8.0,        # Below this: emergency mode (FIXED: was 10.0)
        # Glial support
        initial_support: float = 0.4,           # Starting glial support (FIXED: was 0.3)
        support_learning_rate: float = 0.12,    # How fast support grows (FIXED: was 0.08)
        support_decay: float = 0.99,            # Support decay per generation (FIXED: was 0.98)
        support_max: float = 1.5,               # Maximum support level
        support_threshold: float = 0.4,         # Min support for starvation survival (FIXED: was 0.5)
        # Selection
        base_selection_weight: float = 1.0,     # Base selection weight
        support_weight_factor: float = 0.5,     # How much support affects selection
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        restricted_palette_size: int = 4,       # Palette size during starvation
        initial_palette: List[int] = None,
    ):
        """Initialize Glial Modulation strategy.

        Args:
            initial_energy: Starting energy pool
            energy_max: Maximum energy cap
            energy_regen_rate: Energy regenerated each generation
            base_function_cost: Base energy cost per active function
            complexity_cost_factor: Additional cost scaling with function index
            starvation_threshold: Energy level triggering restricted mode
            critical_threshold: Energy level triggering emergency mode
            initial_support: Starting glial support for all functions
            support_learning_rate: Rate of support growth on success
            support_decay: Per-generation decay of support
            support_max: Maximum support cap
            support_threshold: Minimum support for survival during starvation
            base_selection_weight: Base weight for selection
            support_weight_factor: How much glial support affects selection
            palette_size: Normal target palette size
            restricted_palette_size: Palette size during energy restriction
        """
        # Energy
        self.initial_energy = initial_energy
        self.energy_max = energy_max
        self.energy_regen_rate = energy_regen_rate

        # Costs
        self.base_function_cost = base_function_cost
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

        # Selection
        self.base_selection_weight = base_selection_weight
        self.support_weight_factor = support_weight_factor

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.restricted_palette_size = restricted_palette_size
        # FIXED: Include sin (4) in initial palette for parity problems
        self.initial_palette = initial_palette or [0, 1, 2, 3, 4]  # tanh, sigmoid, relu, identity, sin

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with energy and glial support."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize glial support (initial palette gets boost)
        glial_support = jnp.ones(NUM_ACTIVATIONS) * self.initial_support
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                glial_support = glial_support.at[i].set(self.initial_support * 1.5)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 383838),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Energy state
            'energy_pool': self.initial_energy,
            # Glial support
            'glial_support': glial_support,
            # Tracking
            'starvation_events': 0,
            'critical_events': 0,
            'energy_history': [self.initial_energy],
            'total_energy_consumed': 0.0,
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_function_cost(self, func_idx: int) -> float:
        """Compute energy cost for a specific function."""
        return self.base_function_cost + func_idx * self.complexity_cost_factor

    def _compute_total_cost(self, mask: jnp.ndarray) -> float:
        """Compute total energy cost for current palette."""
        total = 0.0
        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                total += self._compute_function_cost(i)
        return total

    def _update_energy(
        self,
        energy: float,
        mask: jnp.ndarray,
    ) -> Tuple[float, float, str]:
        """Update energy pool based on consumption and regeneration."""
        # Consume energy for active functions
        consumed = self._compute_total_cost(mask)
        new_energy = energy - consumed

        # Regenerate
        new_energy += self.energy_regen_rate

        # Cap at maximum
        new_energy = min(new_energy, self.energy_max)

        # Determine state
        if new_energy < self.critical_threshold:
            state = 'critical'
        elif new_energy < self.starvation_threshold:
            state = 'starvation'
        else:
            state = 'normal'

        return max(new_energy, 0.0), consumed, state

    def _update_glial_support(
        self,
        support: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        improvement: float,
    ) -> jnp.ndarray:
        """Update glial support based on activity and fitness."""
        # Decay all support
        new_support = support * self.support_decay

        if improved:
            # Boost support for active functions
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    delta = self.support_learning_rate * max(improvement, 0.1)
                    new_support = new_support.at[i].set(
                        min(float(new_support[i]) + delta, self.support_max)
                    )

        return jnp.maximum(new_support, 0.01)

    def _compute_selection_weights(
        self,
        support: jnp.ndarray,
        energy_state: str,
    ) -> jnp.ndarray:
        """Compute selection weights based on support and energy state."""
        weights = jnp.ones(NUM_ACTIVATIONS) * self.base_selection_weight

        # Add support bonus
        weights = weights + support * self.support_weight_factor

        if energy_state == 'starvation' or energy_state == 'critical':
            # Penalize low-support functions during energy restriction
            for i in range(NUM_ACTIVATIONS):
                if support[i] < self.support_threshold:
                    penalty = 1.0 - (support[i] / self.support_threshold)
                    weights = weights.at[i].set(
                        float(weights[i]) * (0.3 if energy_state == 'critical' else 0.5)
                    )

        return jnp.maximum(weights, 0.05)

    def _select_palette(
        self,
        weights: jnp.ndarray,
        support: jnp.ndarray,
        energy_state: str,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette based on weights and energy state."""
        # Determine palette size based on energy state
        if energy_state == 'critical':
            target_size = max(self.min_active, self.restricted_palette_size - 1)
        elif energy_state == 'starvation':
            target_size = self.restricted_palette_size
        else:
            target_size = self.palette_size

        # Select top functions by weight
        top_indices = jnp.argsort(weights)[-target_size:]

        new_mask = jnp.zeros(NUM_ACTIVATIONS)
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
        """Update with glial modulation dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update energy pool
        new_energy, consumed, energy_state = self._update_energy(
            state['energy_pool'],
            state['mask'],
        )

        # Track starvation/critical events
        new_starvation_events = state['starvation_events']
        new_critical_events = state['critical_events']
        if energy_state == 'starvation':
            new_starvation_events += 1
        elif energy_state == 'critical':
            new_critical_events += 1

        # Step 2: Update glial support
        new_support = self._update_glial_support(
            state['glial_support'],
            state['mask'],
            improved,
            improvement,
        )

        # Step 3: Compute selection weights
        weights = self._compute_selection_weights(new_support, energy_state)

        # Step 4: Select new palette
        new_mask = self._select_palette(weights, new_support, energy_state, k1)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track energy history
        energy_history = state['energy_history'] + [new_energy]
        if len(energy_history) > 20:
            energy_history = energy_history[-20:]

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
            # Energy state
            'energy_pool': new_energy,
            # Glial support
            'glial_support': new_support,
            # Tracking
            'starvation_events': new_starvation_events,
            'critical_events': new_critical_events,
            'energy_history': energy_history,
            'total_energy_consumed': state['total_energy_consumed'] + consumed,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Top by glial support
        top_support_idx = jnp.argsort(new_support)[-5:][::-1]
        top_support = [(int(i), float(new_support[i])) for i in top_support_idx]

        # Functions above support threshold
        well_supported = [i for i in range(NUM_ACTIVATIONS)
                        if new_support[i] >= self.support_threshold]

        # Active function costs
        active_costs = [(int(i), self._compute_function_cost(i)) for i in active_palette]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Energy
            'energy_pool': new_energy,
            'energy_consumed_this_gen': consumed,
            'energy_state': energy_state,
            'is_starving': energy_state in ['starvation', 'critical'],
            # Starvation history
            'total_starvation_events': new_starvation_events,
            'total_critical_events': new_critical_events,
            # Glial support
            'mean_support': float(jnp.mean(new_support)),
            'max_support': float(jnp.max(new_support)),
            'top_support': top_support,
            'n_well_supported': len(well_supported),
            # Costs
            'active_function_costs': active_costs,
            'total_cost_this_gen': consumed,
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_support': float(new_support[4]),
            'sin_is_supported': 4 in well_supported,
            'sin_cost': self._compute_function_cost(4),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with energy and support status."""
        palette = self.get_active_palette(state)
        support = state['glial_support']
        energy = state['energy_pool']

        # Determine current energy state
        if energy < self.critical_threshold:
            energy_state = 'critical'
        elif energy < self.starvation_threshold:
            energy_state = 'starvation'
        else:
            energy_state = 'normal'

        # Top by support
        top_sup = jnp.argsort(support)[-5:][::-1]
        top_support = [(int(i), float(support[i])) for i in top_sup]

        # Well-supported functions
        well_supported = [i for i in range(NUM_ACTIVATIONS)
                        if support[i] >= self.support_threshold]

        # Energy efficiency (fitness per energy consumed)
        avg_fitness = np.mean(state['fitness_history']) if state['fitness_history'] else 0
        energy_efficiency = avg_fitness / (state['total_energy_consumed'] + 1)

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Energy
            'energy_pool': energy,
            'energy_state': energy_state,
            'total_starvation_events': state['starvation_events'],
            'total_critical_events': state['critical_events'],
            # Support
            'top_support': top_support,
            'n_well_supported': len(well_supported),
            # Efficiency
            'total_energy_consumed': state['total_energy_consumed'],
            'energy_efficiency': energy_efficiency,
            # Sin-specific
            'sin_support': float(support[4]),
            'sin_is_supported': 4 in well_supported,
        }
