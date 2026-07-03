"""Strategy 79: Critical Period Aggregation Dual (Offset Developmental Windows).

Bio inspiration: Different brain regions have different critical period timings.
Visual cortex closes before auditory cortex. Similarly, aggregation functions
should consolidate BEFORE activation functions to provide a stable foundation.

Key innovation:
- Aggregation critical period ends earlier (gen 12/28/30 vs 30/60/80 for activation)
- Aggregation locks first, then activation explores on stable foundation
- Extreme aggregations get extra protection during critical period

Expected: Better aggregation retention through early consolidation.
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class CriticalPeriodAggDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with offset critical periods.

    Aggregation functions consolidate earlier than activation functions,
    providing a stable foundation for activation exploration.
    """

    name = "critical_period_agg_dual"
    description = "Dual: Offset critical periods - aggregation consolidates first"

    def __init__(
        self,
        # Activation critical period phases
        act_exploration_end: int = 30,
        act_confirmation_end: int = 60,
        # Aggregation critical period phases (EARLIER)
        agg_exploration_end: int = 12,
        agg_confirmation_end: int = 28,
        agg_consolidation_gen: int = 30,
        # Activation mutation rates by phase
        act_exploration_activate: float = 0.25,
        act_exploration_deactivate: float = 0.05,
        act_confirmation_activate: float = 0.10,
        act_confirmation_deactivate: float = 0.12,
        act_consolidation_activate: float = 0.02,
        act_consolidation_deactivate: float = 0.02,
        # Aggregation mutation rates by phase (LOWER)
        agg_exploration_activate: float = 0.20,
        agg_exploration_deactivate: float = 0.05,
        agg_confirmation_activate: float = 0.08,
        agg_confirmation_deactivate: float = 0.10,
        agg_consolidation_activate: float = 0.01,
        agg_consolidation_deactivate: float = 0.01,
        # Extreme aggregation protection
        extreme_deactivate_multiplier: float = 0.3,  # 70% reduction in deactivation
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Stagnation
        stagnation_threshold: int = 5,
        # Cross-domain
        cross_learning_rate: float = 0.05,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Critical Period Aggregation Dual strategy."""
        # Activation phases
        self.act_exploration_end = act_exploration_end
        self.act_confirmation_end = act_confirmation_end

        # Aggregation phases (earlier)
        self.agg_exploration_end = agg_exploration_end
        self.agg_confirmation_end = agg_confirmation_end
        self.agg_consolidation_gen = agg_consolidation_gen

        # Activation rates
        self.act_exploration_activate = act_exploration_activate
        self.act_exploration_deactivate = act_exploration_deactivate
        self.act_confirmation_activate = act_confirmation_activate
        self.act_confirmation_deactivate = act_confirmation_deactivate
        self.act_consolidation_activate = act_consolidation_activate
        self.act_consolidation_deactivate = act_consolidation_deactivate

        # Aggregation rates
        self.agg_exploration_activate = agg_exploration_activate
        self.agg_exploration_deactivate = agg_exploration_deactivate
        self.agg_confirmation_activate = agg_confirmation_activate
        self.agg_confirmation_deactivate = agg_confirmation_deactivate
        self.agg_consolidation_activate = agg_consolidation_activate
        self.agg_consolidation_deactivate = agg_consolidation_deactivate

        # Protection
        self.extreme_deactivate_multiplier = extreme_deactivate_multiplier

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.stagnation_threshold = stagnation_threshold

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with phase tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Phase tracking
            'act_phase': 'exploration',
            'agg_phase': 'exploration',
            'agg_locked': False,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 790000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _get_act_phase(self, generation: int) -> str:
        """Determine activation critical period phase."""
        if generation < self.act_exploration_end:
            return 'exploration'
        elif generation < self.act_confirmation_end:
            return 'confirmation'
        else:
            return 'consolidation'

    def _get_agg_phase(self, generation: int) -> str:
        """Determine aggregation critical period phase (earlier)."""
        if generation < self.agg_exploration_end:
            return 'exploration'
        elif generation < self.agg_confirmation_end:
            return 'confirmation'
        else:
            return 'consolidation'

    def _get_act_rates(self, phase: str) -> Tuple[float, float]:
        """Get activation mutation rates for current phase."""
        if phase == 'exploration':
            return self.act_exploration_activate, self.act_exploration_deactivate
        elif phase == 'confirmation':
            return self.act_confirmation_activate, self.act_confirmation_deactivate
        else:
            return self.act_consolidation_activate, self.act_consolidation_deactivate

    def _get_agg_rates(self, phase: str) -> Tuple[float, float]:
        """Get aggregation mutation rates for current phase."""
        if phase == 'exploration':
            return self.agg_exploration_activate, self.agg_exploration_deactivate
        elif phase == 'confirmation':
            return self.agg_confirmation_activate, self.agg_confirmation_deactivate
        else:
            return self.agg_consolidation_activate, self.agg_consolidation_deactivate

    def _mutate_act_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        activate_rate: float,
        deactivate_rate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation to activation palette."""
        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_ACTIVATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            if mask[i] < 0.5:  # Inactive
                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_act or active_count > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _mutate_agg_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        activate_rate: float,
        deactivate_rate: float,
        locked: bool,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation to aggregation palette with extreme protection."""
        if locked:
            return mask, {'activated': [], 'deactivated': [], 'locked': True}

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_AGGREGATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            if mask[i] < 0.5:  # Inactive
                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                # Extreme aggregations get extra protection
                if i in CORE_EXTREME_AGGS:
                    effective_deactivate = deactivate_rate * self.extreme_deactivate_multiplier
                else:
                    effective_deactivate = deactivate_rate

                if p < effective_deactivate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated, 'locked': False}

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
        new_cross = cross_affinity + delta
        return jnp.clip(new_cross, 0.0, 1.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with offset critical periods."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Determine phases
        act_phase = self._get_act_phase(generation)
        agg_phase = self._get_agg_phase(generation)

        # Check if aggregation should be locked
        agg_locked = generation >= self.agg_consolidation_gen

        # Get mutation rates
        act_activate, act_deactivate = self._get_act_rates(act_phase)
        agg_activate, agg_deactivate = self._get_agg_rates(agg_phase)

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_act_palette(
                k_act, state['act_mask'], act_activate, act_deactivate
            )
            new_agg_mask, agg_mutation_info = self._mutate_agg_palette(
                k_agg, state['agg_mask'], agg_activate, agg_deactivate, agg_locked
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            new_stagnation = 0

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'cross_affinity': new_cross,
            'act_phase': act_phase,
            'agg_phase': agg_phase,
            'agg_locked': agg_locked,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
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
            'mutation_triggered': act_mutation_info is not None,
            # Phase info
            'act_phase': act_phase,
            'agg_phase': agg_phase,
            'agg_locked': agg_locked,
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with phase status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'act_phase': state['act_phase'],
            'agg_phase': state['agg_phase'],
            'agg_locked': state['agg_locked'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
        }
