"""Strategy 75D: Aggregation Critical Period Dual (Shorter Aggregation Critical Periods).

Biological Basis: Different modalities have different critical period timings.
Visual cortex consolidates before higher-order areas. Lower-level processing
stabilizes to provide consistent input to higher processing.

Key mechanism: SHORTER critical period for aggregations than activations.
Aggregation consolidates EARLY (gen 15) so activation can explore with
stable aggregation foundations.

Hypothesis: Stable aggregations enable more effective activation search.

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
)


class AggCriticalPeriodDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with offset critical periods.

    Aggregation has shorter critical period (consolidates early) while
    activation has longer critical period (continues exploration).
    """

    name = "agg_critical_period_dual"
    description = "Dual: Aggregation consolidates early, activation explores longer"

    def __init__(
        self,
        # Activation critical periods (longer)
        act_exploration_end: int = 30,
        act_confirmation_end: int = 60,
        # Aggregation critical periods (SHORTER)
        agg_exploration_end: int = 15,   # Half the exploration time
        agg_confirmation_end: int = 35,  # Earlier consolidation
        # Mutation rates
        exploration_mutation_rate: float = 0.15,
        confirmation_mutation_rate: float = 0.08,
        consolidation_mutation_rate: float = 0.02,
        stagnation_threshold: int = 5,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Cross-domain
        cross_learning_rate: float = 0.05,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Aggregation Critical Period Dual strategy.

        Args:
            act_exploration_end: End of activation exploration phase
            act_confirmation_end: End of activation confirmation phase
            agg_exploration_end: End of aggregation exploration phase (shorter)
            agg_confirmation_end: End of aggregation confirmation phase (earlier)
            exploration_mutation_rate: High mutation during exploration
            confirmation_mutation_rate: Medium mutation during confirmation
            consolidation_mutation_rate: Low mutation during consolidation
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            cross_learning_rate: Rate of cross-domain affinity learning
            initial_act_palette: Starting activation palette indices
            initial_agg_palette: Starting aggregation palette indices
        """
        # Activation periods
        self.act_exploration_end = act_exploration_end
        self.act_confirmation_end = act_confirmation_end

        # Aggregation periods (SHORTER)
        self.agg_exploration_end = agg_exploration_end
        self.agg_confirmation_end = agg_confirmation_end

        # Mutation rates
        self.exploration_mutation_rate = exploration_mutation_rate
        self.confirmation_mutation_rate = confirmation_mutation_rate
        self.consolidation_mutation_rate = consolidation_mutation_rate
        self.stagnation_threshold = stagnation_threshold

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        # Palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with critical period tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            # Aggregation domain
            'agg_mask': agg_mask,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Period tracking
            'act_phase': 'exploration',
            'agg_phase': 'exploration',
            # General state
            'rng_key': jax.random.PRNGKey(seed + 750000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _get_phase(
        self,
        generation: int,
        exploration_end: int,
        confirmation_end: int,
    ) -> str:
        """Determine current phase based on generation."""
        if generation < exploration_end:
            return 'exploration'
        elif generation < confirmation_end:
            return 'confirmation'
        else:
            return 'consolidation'

    def _get_mutation_rate(self, phase: str) -> float:
        """Get mutation rate for current phase."""
        if phase == 'exploration':
            return self.exploration_mutation_rate
        elif phase == 'confirmation':
            return self.confirmation_mutation_rate
        else:
            return self.consolidation_mutation_rate

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        mutation_rate: float,
        min_active: int,
        max_active: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation to palette."""
        flip_probs = jax.random.uniform(key, (n_funcs,))
        flip_mask = flip_probs < mutation_rate
        new_mask = jnp.where(flip_mask, 1.0 - mask, mask)

        flipped_indices = jnp.where(flip_mask)[0].tolist()
        activated = [i for i in flipped_indices if mask[i] < 0.5]
        deactivated = [i for i in flipped_indices if mask[i] > 0.5]

        active_count = jnp.sum(new_mask > 0.5)
        if active_count < min_active or active_count > max_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

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

        # Determine current phases
        act_phase = self._get_phase(generation, self.act_exploration_end, self.act_confirmation_end)
        agg_phase = self._get_phase(generation, self.agg_exploration_end, self.agg_confirmation_end)

        # Get mutation rates
        act_mutation_rate = self._get_mutation_rate(act_phase)
        agg_mutation_rate = self._get_mutation_rate(agg_phase)

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
            new_act_mask, act_mutation_info = self._mutate_palette(
                k_act, state['act_mask'],
                act_mutation_rate,
                self.min_active_act, self.max_active_act,
                NUM_ACTIVATIONS,
            )
            new_agg_mask, agg_mutation_info = self._mutate_palette(
                k_agg, state['agg_mask'],
                agg_mutation_rate,
                self.min_active_agg, self.max_active_agg,
                NUM_AGGREGATIONS,
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
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Check extreme aggregation status
        has_extreme = any(a in EXTREME_AGGS for a in agg_palette)
        extreme_count = sum(1 for a in agg_palette if a in EXTREME_AGGS)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'mutation_triggered': act_mutation_info is not None,
            # Cross-domain metrics
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # Phase metrics
            'act_phase': act_phase,
            'agg_phase': agg_phase,
            'act_mutation_rate': act_mutation_rate,
            'agg_mutation_rate': agg_mutation_rate,
            'phase_offset': self.act_exploration_end - self.agg_exploration_end,
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'has_extreme': has_extreme,
            'extreme_count': extreme_count,
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with critical period status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'act_phase': state['act_phase'],
            'agg_phase': state['agg_phase'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
