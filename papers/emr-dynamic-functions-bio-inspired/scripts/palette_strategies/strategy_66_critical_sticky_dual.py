"""Strategy 66D: Critical Period + Sticky Dual (Both Domains).

Extends CriticalStickyStrategy to jointly evolve BOTH activation AND
aggregation function palettes with coordinated critical periods.

Key dual mechanisms:
1. Dual phase progression - synchronized phases for both domains
2. Cross-domain sticky protection - successful pairs protected together
3. Dual confirmation tracking - persistence for both domains
4. Coordinated early consolidation - if one domain solves, both stabilize

Expected: 100% discovery @ 2 gens, 60%+ retention, coordinated evolution
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


# Oscillatory activation indices that get sticky protection
OSCILLATORY_ACT_INDICES = [4, 11, 12, 13, 15]  # sin, burst, resonator, osc_adapt, receptive

# Aggregation indices that get sticky protection (those that complement oscillatory)
STICKY_AGG_INDICES = [0, 1, 2]  # sum, mean, max - core aggregations


class CriticalStickyDualStrategy(PaletteEvolutionStrategy):
    """Critical Period with sticky protection for both domains.

    Combines:
    - Critical Period phases (fast discovery in exploration phase)
    - Sticky oscillatory protection for activations
    - Sticky protection for core aggregations
    - Cross-domain affinity tracking

    Phases:
    1. EXPLORATION: High activation for both, minimal deactivation
       - Goal: Discover oscillatory functions and good aggregations
    2. CONFIRMATION: Moderate activation, selective pruning
       - Oscillatory activations protected with sticky rate
       - Core aggregations protected similarly
    3. CONSOLIDATION: Lock in palette, minimal changes
    """

    name = "critical_sticky_dual"
    description = "Dual critical periods with sticky protection for both domains"

    def __init__(
        self,
        # Phase boundaries
        exploration_end: int = 20,
        confirmation_end: int = 50,
        # Activation phase rates
        act_exploration_activate: float = 0.35,
        act_exploration_deactivate: float = 0.02,
        act_confirmation_activate: float = 0.10,
        act_confirmation_deactivate: float = 0.15,
        act_consolidation_activate: float = 0.02,
        act_consolidation_deactivate: float = 0.01,
        # Aggregation phase rates (slightly lower)
        agg_exploration_activate: float = 0.25,
        agg_exploration_deactivate: float = 0.02,
        agg_confirmation_activate: float = 0.08,
        agg_confirmation_deactivate: float = 0.12,
        agg_consolidation_activate: float = 0.02,
        agg_consolidation_deactivate: float = 0.01,
        # Sticky protection
        sticky_deactivate_rate: float = 0.01,
        # Cross-domain protection
        cross_domain_protection: float = 0.3,
        # Confirmation tracking
        confirmation_persistence: int = 5,
        # Early consolidation
        early_consolidation_threshold: float = 0.95,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
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

        self.sticky_deactivate_rate = sticky_deactivate_rate
        self.cross_domain_protection = cross_domain_protection
        self.confirmation_persistence = confirmation_persistence
        self.early_consolidation_threshold = early_consolidation_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase, with early consolidation if solved."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def _get_act_rates(self, phase: str) -> Tuple[float, float]:
        """Get activation rates for current phase."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            return self.act_exploration_activate, self.act_exploration_deactivate
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            return self.act_confirmation_activate, self.act_confirmation_deactivate
        else:
            return self.act_consolidation_activate, self.act_consolidation_deactivate

    def _get_agg_rates(self, phase: str) -> Tuple[float, float]:
        """Get aggregation rates for current phase."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            return self.agg_exploration_activate, self.agg_exploration_deactivate
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            return self.agg_confirmation_activate, self.agg_confirmation_deactivate
        else:
            return self.agg_consolidation_activate, self.agg_consolidation_deactivate

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual confirmation tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Activation persistence and confirmation
        act_persistence = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                act_persistence = act_persistence.at[i].set(1)
        act_confirmed = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)

        # Aggregation persistence and confirmation
        agg_persistence = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                agg_persistence = agg_persistence.at[i].set(1)
        agg_confirmed = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Cross-domain affinity
        affinity = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        return {
            # Activation state
            'act_mask': act_mask,
            'act_persistence': act_persistence,
            'act_confirmed': act_confirmed,
            # Aggregation state
            'agg_mask': agg_mask,
            'agg_persistence': agg_persistence,
            'agg_confirmed': agg_confirmed,
            # Cross-domain
            'affinity': affinity,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 666666),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            # Discovery tracking
            'oscillatory_discovery_gen': {i: None for i in OSCILLATORY_ACT_INDICES},
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette."""
        return mask_to_indices(state['agg_mask'])

    def _update_persistence_and_confirmation(
        self,
        mask: jnp.ndarray,
        persistence: jnp.ndarray,
        confirmed: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update persistence counters and confirm long-present functions."""
        active = mask > 0.5
        new_persistence = jnp.where(active, persistence + 1, jnp.zeros_like(persistence))
        long_enough = new_persistence >= self.confirmation_persistence
        new_confirmed = jnp.logical_or(confirmed, long_enough)
        return new_persistence, new_confirmed

    def _compute_cross_protection(
        self,
        affinity: jnp.ndarray,
        act_active: jnp.ndarray,
        agg_active: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute cross-domain protection based on affinity."""
        # Activations protected by their strongest aggregation partners
        act_protection = jnp.zeros(NUM_ACTIVATIONS)
        for i in range(NUM_ACTIVATIONS):
            if act_active[i]:
                max_aff = 0.0
                for j in range(NUM_AGGREGATIONS):
                    if agg_active[j]:
                        max_aff = max(max_aff, float(affinity[i, j]))
                act_protection = act_protection.at[i].set(max_aff * self.cross_domain_protection)

        # Aggregations protected by their strongest activation partners
        agg_protection = jnp.zeros(NUM_AGGREGATIONS)
        for j in range(NUM_AGGREGATIONS):
            if agg_active[j]:
                max_aff = 0.0
                for i in range(NUM_ACTIVATIONS):
                    if act_active[i]:
                        max_aff = max(max_aff, float(affinity[i, j]))
                agg_protection = agg_protection.at[j].set(max_aff * self.cross_domain_protection)

        return act_protection, agg_protection

    def _mutate_activation_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        confirmed: jnp.ndarray,
        cross_protection: jnp.ndarray,
        activate_rate: float,
        deactivate_rate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Mutate activation palette with sticky oscillatory protection."""
        key1, key2, key3 = jax.random.split(key, 3)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        protected = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))
        protection_rolls = jax.random.uniform(key3, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            if mask[i] < 0.5:
                if activate_probs[i] < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                is_oscillatory = i in OSCILLATORY_ACT_INDICES

                # Full protection in consolidation for confirmed
                if phase == CriticalPeriodPhase.CONSOLIDATION and confirmed[i]:
                    if is_oscillatory:
                        protected.append(i)
                    continue

                # Determine deactivation rate
                if is_oscillatory:
                    actual_rate = self.sticky_deactivate_rate
                    protected.append(i)
                else:
                    actual_rate = deactivate_rate

                # Cross-domain protection reduces deactivation chance
                protection = float(cross_protection[i])
                if protection_rolls[i] < protection:
                    protected.append(i)
                    continue

                if deactivate_probs[i] < actual_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < self.min_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'protected': protected,
        }

    def _mutate_aggregation_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        confirmed: jnp.ndarray,
        cross_protection: jnp.ndarray,
        activate_rate: float,
        deactivate_rate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Mutate aggregation palette with sticky protection."""
        key1, key2, key3 = jax.random.split(key, 3)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        protected = []

        activate_probs = jax.random.uniform(key1, (NUM_AGGREGATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_AGGREGATIONS,))
        protection_rolls = jax.random.uniform(key3, (NUM_AGGREGATIONS,))

        for i in range(NUM_AGGREGATIONS):
            if mask[i] < 0.5:
                if activate_probs[i] < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                is_sticky = i in STICKY_AGG_INDICES

                # Full protection in consolidation for confirmed
                if phase == CriticalPeriodPhase.CONSOLIDATION and confirmed[i]:
                    if is_sticky:
                        protected.append(i)
                    continue

                # Determine deactivation rate
                if is_sticky:
                    actual_rate = self.sticky_deactivate_rate
                    protected.append(i)
                else:
                    actual_rate = deactivate_rate

                # Cross-domain protection
                protection = float(cross_protection[i])
                if protection_rolls[i] < protection:
                    protected.append(i)
                    continue

                if deactivate_probs[i] < actual_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < self.min_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'protected': protected,
        }

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness: float,
        improved: bool,
    ) -> jnp.ndarray:
        """Update cross-domain affinity based on success."""
        if not improved or fitness <= 0:
            return affinity

        new_affinity = affinity.copy()
        act_active = act_mask > 0.5
        agg_active = agg_mask > 0.5

        learning_rate = 0.1 * fitness
        for i in range(NUM_ACTIVATIONS):
            if act_active[i]:
                for j in range(NUM_AGGREGATIONS):
                    if agg_active[j]:
                        current = float(new_affinity[i, j])
                        new_affinity = new_affinity.at[i, j].set(
                            min(1.0, current + learning_rate)
                        )

        return new_affinity

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual critical period + sticky logic."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Determine phase
        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']

        act_activate_rate, act_deactivate_rate = self._get_act_rates(phase)
        agg_activate_rate, agg_deactivate_rate = self._get_agg_rates(phase)

        # Update persistence and confirmation
        new_act_persistence, new_act_confirmed = self._update_persistence_and_confirmation(
            state['act_mask'], state['act_persistence'], state['act_confirmed']
        )
        new_agg_persistence, new_agg_confirmed = self._update_persistence_and_confirmation(
            state['agg_mask'], state['agg_persistence'], state['agg_confirmed']
        )

        # Update affinity
        new_affinity = self._update_affinity(
            state['affinity'], state['act_mask'], state['agg_mask'],
            best_fitness, improved
        )

        # Compute cross-domain protection
        act_protection, agg_protection = self._compute_cross_protection(
            new_affinity, state['act_mask'] > 0.5, state['agg_mask'] > 0.5
        )

        # Track oscillatory discovery
        osc_discovery = dict(state['oscillatory_discovery_gen'])
        current_act = self.get_active_palette(state)
        for i in OSCILLATORY_ACT_INDICES:
            if i in current_act and osc_discovery[i] is None:
                osc_discovery[i] = generation

        # Mutate both palettes
        new_act_mask, act_info = self._mutate_activation_palette(
            k_act, state['act_mask'], phase, new_act_confirmed,
            act_protection, act_activate_rate, act_deactivate_rate
        )
        new_agg_mask, agg_info = self._mutate_aggregation_palette(
            k_agg, state['agg_mask'], phase, new_agg_confirmed,
            agg_protection, agg_activate_rate, agg_deactivate_rate
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        new_state = {
            'act_mask': new_act_mask,
            'act_persistence': new_act_persistence,
            'act_confirmed': new_act_confirmed,
            'agg_mask': new_agg_mask,
            'agg_persistence': new_agg_persistence,
            'agg_confirmed': new_agg_confirmed,
            'affinity': new_affinity,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'oscillatory_discovery_gen': osc_discovery,
        }

        # Compute stats
        n_act_confirmed = int(jnp.sum(new_act_confirmed))
        n_agg_confirmed = int(jnp.sum(new_agg_confirmed))
        n_confirmed_osc = sum(1 for i in OSCILLATORY_ACT_INDICES if new_act_confirmed[i])
        active_osc = [i for i in OSCILLATORY_ACT_INDICES if new_act_mask[i] > 0.5]

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            # Confirmation
            'n_act_confirmed': n_act_confirmed,
            'n_agg_confirmed': n_agg_confirmed,
            'n_confirmed_oscillatory': n_confirmed_osc,
            'active_oscillatory': active_osc,
            # Activation info
            'act_activated': act_info['activated'],
            'act_deactivated': act_info['deactivated'],
            'act_protected': act_info['protected'],
            # Aggregation info
            'agg_activated': agg_info['activated'],
            'agg_deactivated': agg_info['deactivated'],
            'agg_protected': agg_info['protected'],
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_confirmed': bool(new_act_confirmed[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual oscillatory tracking."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        act_confirmed = state['act_confirmed']
        agg_confirmed = state['agg_confirmed']

        confirmed_act_list = [i for i in range(NUM_ACTIVATIONS) if act_confirmed[i]]
        confirmed_agg_list = [i for i in range(NUM_AGGREGATIONS) if agg_confirmed[i]]
        active_osc = [i for i in OSCILLATORY_ACT_INDICES if i in act_palette]
        confirmed_osc = [i for i in OSCILLATORY_ACT_INDICES if act_confirmed[i]]

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'phase': state['phase'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'n_act_confirmed': len(confirmed_act_list),
            'n_agg_confirmed': len(confirmed_agg_list),
            'confirmed_act': confirmed_act_list,
            'confirmed_agg': confirmed_agg_list,
            'active_oscillatory': active_osc,
            'confirmed_oscillatory': confirmed_osc,
            'oscillatory_discovery_gen': state['oscillatory_discovery_gen'],
        }
