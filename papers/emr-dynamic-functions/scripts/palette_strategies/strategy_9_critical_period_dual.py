"""Strategy 9 Dual: Critical Period Discovery for Both Activation AND Aggregation.

Extends Critical Period strategy to jointly evolve both palettes with
phase-specific plasticity.

Bio-inspired developmental windows:
- Early life: High plasticity, rapid learning (EXPLORATION)
- Adolescence: Refinement and pruning (CONFIRMATION)
- Adulthood: Stable, consolidated circuits (CONSOLIDATION)

Three phases for EACH domain:
1. Exploration (gen 0-20): Aggressive discovery
2. Confirmation (gen 20-50): Prune unhelpful, confirm useful
3. Consolidation (gen 50+): Lock confirmed

Key innovation: Cross-domain confirmation - functions that persist together
across domains get stronger protection.

Biological analogy:
- Critical periods exist for multiple modalities (vision, hearing, language)
- Each modality has its own developmental timeline but they interact
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


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class CriticalPeriodDualStrategy(PaletteEvolutionStrategy):
    """Critical period palette evolution with developmental phases for BOTH domains.

    Three developmental phases applied to both activation and aggregation:
    1. EXPLORATION: High activation rate, minimal deactivation
    2. CONFIRMATION: Moderate activation, high deactivation (prune unhelpful)
    3. CONSOLIDATION: Minimal changes, lock in confirmed functions

    Cross-domain learning:
    - Tracks which act-agg combinations persist together
    - Functions confirmed in one domain can protect related functions in other
    """

    name = "critical_period_dual"
    description = "Dual palette with developmental phases and cross-domain confirmation"

    def __init__(
        self,
        # Phase boundaries (generations)
        exploration_end: int = 20,
        confirmation_end: int = 50,
        # Phase-specific rates for activations
        act_exploration_activate: float = 0.35,
        act_exploration_deactivate: float = 0.02,
        act_confirmation_activate: float = 0.10,
        act_confirmation_deactivate: float = 0.15,
        act_consolidation_activate: float = 0.02,
        act_consolidation_deactivate: float = 0.01,
        # Phase-specific rates for aggregations
        agg_exploration_activate: float = 0.30,
        agg_exploration_deactivate: float = 0.03,
        agg_confirmation_activate: float = 0.08,
        agg_confirmation_deactivate: float = 0.12,
        agg_consolidation_activate: float = 0.02,
        agg_consolidation_deactivate: float = 0.01,
        # Confirmation tracking
        confirmation_persistence: int = 5,  # Gens to confirm a function
        # Cross-domain learning
        cross_learning_rate: float = 0.1,
        cross_protection_boost: float = 0.3,  # Extra protection from cross-domain
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,   # Optimal is 6, >6 causes antagonism
        max_active_agg: int = 4,   # Optimal is 4 for parity
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize strategy."""
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

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

        self.confirmation_persistence = confirmation_persistence
        self.cross_learning_rate = cross_learning_rate
        self.cross_protection_boost = cross_protection_boost
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_phase(self, generation: int) -> str:
        """Determine current developmental phase."""
        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual confirmation tracking."""
        # Activation palette
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        # Aggregation palette
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Persistence tracking - how long each function has been active
        act_persistence = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                act_persistence = act_persistence.at[i].set(1)

        agg_persistence = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                agg_persistence = agg_persistence.at[i].set(1)

        # Confirmed functions (protected in consolidation)
        act_confirmed = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_confirmed = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Cross-domain co-persistence: how often act[i] and agg[j] are active together
        cross_copersistence = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS), dtype=jnp.int32)

        # Fitness tracking per function
        act_fitness_sum = jnp.zeros(NUM_ACTIVATIONS)
        act_fitness_count = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_fitness_sum = jnp.zeros(NUM_AGGREGATIONS)
        agg_fitness_count = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)

        return {
            # Activation state
            'act_mask': act_mask,
            'act_persistence': act_persistence,
            'act_confirmed': act_confirmed,
            'act_fitness_sum': act_fitness_sum,
            'act_fitness_count': act_fitness_count,
            # Aggregation state
            'agg_mask': agg_mask,
            'agg_persistence': agg_persistence,
            'agg_confirmed': agg_confirmed,
            'agg_fitness_sum': agg_fitness_sum,
            'agg_fitness_count': agg_fitness_count,
            # Cross-domain state
            'cross_copersistence': cross_copersistence,
            # Common state
            'rng_key': jax.random.PRNGKey(seed + 99999),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _update_persistence_and_confirmation(
        self,
        mask: jnp.ndarray,
        persistence: jnp.ndarray,
        confirmed: jnp.ndarray,
        best_fitness: float,
        fitness_sum: jnp.ndarray,
        fitness_count: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update persistence counters and confirm long-present functions."""
        active = mask > 0.5

        # Update persistence
        new_persistence = jnp.where(active, persistence + 1, jnp.zeros_like(persistence))

        # Update fitness tracking
        new_fitness_sum = jnp.where(active, fitness_sum + best_fitness, fitness_sum)
        new_fitness_count = jnp.where(active, fitness_count + 1, fitness_count)

        # Confirm functions present long enough
        long_enough = new_persistence >= self.confirmation_persistence
        new_confirmed = jnp.logical_or(confirmed, long_enough)

        return new_persistence, new_confirmed, new_fitness_sum, new_fitness_count

    def _update_cross_copersistence(
        self,
        copersistence: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update cross-domain co-persistence matrix."""
        act_active = (act_mask > 0.5).astype(jnp.int32)
        agg_active = (agg_mask > 0.5).astype(jnp.int32)

        # Increment for pairs that are both active
        co_active = jnp.outer(act_active, agg_active)
        new_copersistence = copersistence + co_active

        return new_copersistence

    def _compute_cross_protection(
        self,
        copersistence: jnp.ndarray,
        other_mask: jnp.ndarray,
        other_confirmed: jnp.ndarray,
        is_activation: bool = True,
    ) -> jnp.ndarray:
        """Compute cross-domain protection based on co-persistence with confirmed functions."""
        other_active = (other_mask > 0.5).astype(jnp.float32)
        other_confirmed_float = other_confirmed.astype(jnp.float32)

        # Weight by confirmation status of the other domain
        weighted_other = other_active * (0.5 + 0.5 * other_confirmed_float)

        if is_activation:
            # For activations: how much co-persistence with (confirmed) aggregations
            copersist_score = jnp.dot(copersistence.astype(jnp.float32), weighted_other)
        else:
            # For aggregations: how much co-persistence with (confirmed) activations
            copersist_score = jnp.dot(copersistence.T.astype(jnp.float32), weighted_other)

        # Normalize to [0, 1]
        max_score = jnp.max(copersist_score) + 1e-6
        normalized = copersist_score / max_score

        return normalized

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        confirmed: jnp.ndarray,
        cross_protection: jnp.ndarray,
        activate_rates: Tuple[float, float, float],
        deactivate_rates: Tuple[float, float, float],
        min_active: int,
        max_active: int,
        n_functions: int,
    ) -> Tuple[jnp.ndarray, List[int], List[int]]:
        """Apply phase-appropriate mutation to a palette.

        Key constraint: max_active prevents antagonism (too many functions hurts).
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        # Get phase-specific rates
        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = activate_rates[0]
            deactivate_rate = deactivate_rates[0]
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = activate_rates[1]
            deactivate_rate = deactivate_rates[1]
        else:
            activate_rate = activate_rates[2]
            deactivate_rate = deactivate_rates[2]

        activate_probs = jax.random.uniform(key1, (n_functions,))
        deactivate_probs = jax.random.uniform(key2, (n_functions,))

        # Track current active count for max constraint
        current_active = int(jnp.sum(mask > 0.5))

        for i in range(n_functions):
            cross_prot = float(cross_protection[i])

            if mask[i] < 0.5:
                # Inactive - might activate
                # CRITICAL: Skip if already at max
                if current_active + len(activated) >= max_active:
                    continue

                if activate_probs[i] < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                # In consolidation, confirmed functions are protected
                if phase == CriticalPeriodPhase.CONSOLIDATION and confirmed[i]:
                    continue

                # Cross-domain protection reduces deactivation rate
                protection_factor = 1.0 - self.cross_protection_boost * cross_prot
                effective_deact = deactivate_rate * max(0.1, protection_factor)

                if deactivate_probs[i] < effective_deact:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, activated, deactivated

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with critical period logic for both palettes."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        # Check improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Determine current phase
        phase = self._get_phase(generation)
        phase_changed = phase != state['phase']

        # --- Update activation domain ---
        new_act_persistence, new_act_confirmed, new_act_fitness_sum, new_act_fitness_count = \
            self._update_persistence_and_confirmation(
                state['act_mask'],
                state['act_persistence'],
                state['act_confirmed'],
                best_fitness,
                state['act_fitness_sum'],
                state['act_fitness_count'],
            )

        # --- Update aggregation domain ---
        new_agg_persistence, new_agg_confirmed, new_agg_fitness_sum, new_agg_fitness_count = \
            self._update_persistence_and_confirmation(
                state['agg_mask'],
                state['agg_persistence'],
                state['agg_confirmed'],
                best_fitness,
                state['agg_fitness_sum'],
                state['agg_fitness_count'],
            )

        # --- Update cross-domain co-persistence ---
        new_cross_copersistence = self._update_cross_copersistence(
            state['cross_copersistence'],
            state['act_mask'],
            state['agg_mask'],
        )

        # --- Compute cross-domain protection ---
        act_cross_protection = self._compute_cross_protection(
            new_cross_copersistence,
            state['agg_mask'],
            new_agg_confirmed,
            is_activation=True,
        )
        agg_cross_protection = self._compute_cross_protection(
            new_cross_copersistence,
            state['act_mask'],
            new_act_confirmed,
            is_activation=False,
        )

        # --- Apply mutations ---
        act_rates = (
            self.act_exploration_activate,
            self.act_confirmation_activate,
            self.act_consolidation_activate,
        )
        act_deact_rates = (
            self.act_exploration_deactivate,
            self.act_confirmation_deactivate,
            self.act_consolidation_deactivate,
        )
        new_act_mask, act_activated, act_deactivated = self._mutate_palette(
            key_act, state['act_mask'], phase, new_act_confirmed,
            act_cross_protection, act_rates, act_deact_rates,
            self.min_active_act, self.max_active_act, NUM_ACTIVATIONS,
        )

        agg_rates = (
            self.agg_exploration_activate,
            self.agg_confirmation_activate,
            self.agg_consolidation_activate,
        )
        agg_deact_rates = (
            self.agg_exploration_deactivate,
            self.agg_confirmation_deactivate,
            self.agg_consolidation_deactivate,
        )
        new_agg_mask, agg_activated, agg_deactivated = self._mutate_palette(
            key_agg, state['agg_mask'], phase, new_agg_confirmed,
            agg_cross_protection, agg_rates, agg_deact_rates,
            self.min_active_agg, self.max_active_agg, NUM_AGGREGATIONS,
        )

        act_palette_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_palette_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        new_state = {
            'act_mask': new_act_mask,
            'act_persistence': new_act_persistence,
            'act_confirmed': new_act_confirmed,
            'act_fitness_sum': new_act_fitness_sum,
            'act_fitness_count': new_act_fitness_count,
            'agg_mask': new_agg_mask,
            'agg_persistence': new_agg_persistence,
            'agg_confirmed': new_agg_confirmed,
            'agg_fitness_sum': new_agg_fitness_sum,
            'agg_fitness_count': new_agg_fitness_count,
            'cross_copersistence': new_cross_copersistence,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
        }

        # Compute stats
        n_act_confirmed = int(jnp.sum(new_act_confirmed))
        n_agg_confirmed = int(jnp.sum(new_agg_confirmed))
        avg_act_persistence = float(jnp.mean(
            new_act_persistence[state['act_mask'] > 0.5]
        )) if jnp.sum(state['act_mask'] > 0.5) > 0 else 0.0
        avg_agg_persistence = float(jnp.mean(
            new_agg_persistence[state['agg_mask'] > 0.5]
        )) if jnp.sum(state['agg_mask'] > 0.5) > 0 else 0.0

        # Find strongest cross-domain pairs
        max_copersist = float(jnp.max(new_cross_copersistence))
        max_idx = jnp.unravel_index(
            jnp.argmax(new_cross_copersistence), new_cross_copersistence.shape
        )

        metrics = {
            'act_palette_changed': act_palette_changed,
            'agg_palette_changed': agg_palette_changed,
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Critical period stats
            'phase': phase,
            'phase_changed': phase_changed,
            # Activation domain
            'n_act_confirmed': n_act_confirmed,
            'avg_act_persistence': avg_act_persistence,
            'act_activated': act_activated,
            'act_deactivated': act_deactivated,
            # Aggregation domain
            'n_agg_confirmed': n_agg_confirmed,
            'avg_agg_persistence': avg_agg_persistence,
            'agg_activated': agg_activated,
            'agg_deactivated': agg_deactivated,
            # Cross-domain
            'max_copersistence': max_copersist,
            'strongest_cross_pair': (int(max_idx[0]), int(max_idx[1])),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including phase and dual confirmation stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        # List confirmed functions
        act_confirmed_list = [i for i in range(NUM_ACTIVATIONS) if state['act_confirmed'][i]]
        agg_confirmed_list = [i for i in range(NUM_AGGREGATIONS) if state['agg_confirmed'][i]]

        # Find strongest cross-domain pair
        copersist = state['cross_copersistence']
        max_copersist = float(jnp.max(copersist))
        max_idx = jnp.unravel_index(jnp.argmax(copersist), copersist.shape)

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'act_palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'phase': state['phase'],
            'generation': state['generation'],
            # Confirmation stats
            'n_act_confirmed': len(act_confirmed_list),
            'act_confirmed': act_confirmed_list,
            'n_agg_confirmed': len(agg_confirmed_list),
            'agg_confirmed': agg_confirmed_list,
            # Cross-domain
            'strongest_cross_pair': (int(max_idx[0]), int(max_idx[1])),
            'max_copersistence': max_copersist,
            'stagnation_count': state['stagnation_count'],
        }
