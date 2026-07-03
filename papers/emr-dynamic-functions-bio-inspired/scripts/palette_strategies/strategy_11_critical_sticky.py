"""Strategy 11: Critical Period + Sticky Oscillatory Hybrid.

Fixes Critical Period's aggressive pruning by adding sticky protection for
oscillatory functions during the confirmation phase.

The original Critical Period (Strategy 9) had:
- 100% sin discovery at gen 2.0 (fastest!)
- BUT only 7% sin retention (aggressive confirmation phase pruning)

This hybrid adds:
- Oscillatory functions use sticky (0.01) deactivation rate instead of 0.15
- Confirmed oscillatory functions become fully protected
- Maintains fast discovery while preserving useful functions

Expected: 100% discovery @ 2 gens, 60%+ retention (vs 7%)
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


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


# Oscillatory indices that get sticky protection
OSCILLATORY_INDICES = [4, 11, 12, 13, 15]  # sin, burst, resonator, osc_adapt, receptive


class CriticalStickyStrategy(PaletteEvolutionStrategy):
    """Critical Period with sticky oscillatory protection.

    Combines:
    - Critical Period phases (fast discovery in exploration phase)
    - Sticky oscillatory protection (preserve discovered oscillatory functions)

    Key fix: During confirmation phase, oscillatory functions use sticky
    deactivation rate (0.01) instead of normal rate (0.15), preventing
    the aggressive pruning that caused only 7% sin retention.

    Phases:
    1. EXPLORATION: High activation (35%), minimal deactivation (2%)
       - Goal: Discover oscillatory functions quickly
    2. CONFIRMATION: Moderate activation (10%), pruning non-oscillatory (15%)
       - Oscillatory functions protected with sticky rate (1%)
    3. CONSOLIDATION: Lock in palette, minimal changes
    """

    name = "critical_sticky"
    description = "Critical periods with sticky oscillatory protection"

    def __init__(
        self,
        # Phase boundaries
        exploration_end: int = 20,
        confirmation_end: int = 50,
        # Phase rates (same as critical_period)
        exploration_activate: float = 0.35,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate: float = 0.15,
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # Sticky oscillatory protection (the key fix)
        sticky_deactivate_rate: float = 0.01,
        # Confirmation tracking
        confirmation_persistence: int = 5,
        # Early solve detection - if solved, skip to consolidation
        early_consolidation_threshold: float = 0.95,
        # Constraints
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            exploration_end: Generation when exploration phase ends
            confirmation_end: Generation when confirmation phase ends
            exploration_activate/deactivate: Rates during exploration
            confirmation_activate/deactivate: Rates during confirmation
            consolidation_activate/deactivate: Rates during consolidation
            sticky_deactivate_rate: Deactivation rate for oscillatory functions
            confirmation_persistence: Gens activation must be present to confirm
            early_consolidation_threshold: If fitness exceeds this, enter consolidation
            min_active: Minimum active functions
            initial_palette: Starting palette indices
        """
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

        self.exploration_activate = exploration_activate
        self.exploration_deactivate = exploration_deactivate
        self.confirmation_activate = confirmation_activate
        self.confirmation_deactivate = confirmation_deactivate
        self.consolidation_activate = consolidation_activate
        self.consolidation_deactivate = consolidation_deactivate

        self.sticky_deactivate_rate = sticky_deactivate_rate
        self.confirmation_persistence = confirmation_persistence
        self.early_consolidation_threshold = early_consolidation_threshold
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase, with early consolidation if solved."""
        # Early consolidation if problem is solved
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def _get_rates(self, phase: str) -> Tuple[float, float]:
        """Get base activation/deactivation rates for current phase."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            return self.exploration_activate, self.exploration_deactivate
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            return self.confirmation_activate, self.confirmation_deactivate
        else:
            return self.consolidation_activate, self.consolidation_deactivate

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with confirmation tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Track persistence for confirmation
        activation_persistence = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                activation_persistence = activation_persistence.at[i].set(1)

        # Confirmed activations
        confirmed = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 111111),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Critical period state
            'phase': CriticalPeriodPhase.EXPLORATION,
            'activation_persistence': activation_persistence,
            'confirmed': confirmed,
            # Track when oscillatory functions were discovered
            'oscillatory_discovery_gen': {i: None for i in OSCILLATORY_INDICES},
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_persistence_and_confirmation(
        self,
        mask: jnp.ndarray,
        persistence: jnp.ndarray,
        confirmed: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update persistence counters and confirm long-present activations."""
        active = mask > 0.5

        # Increment for active, reset for inactive
        new_persistence = jnp.where(
            active,
            persistence + 1,
            jnp.zeros_like(persistence)
        )

        # Confirm activations present long enough
        long_enough = new_persistence >= self.confirmation_persistence
        new_confirmed = jnp.logical_or(confirmed, long_enough)

        return new_persistence, new_confirmed

    def _mutate_palette_critical_sticky(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        confirmed: jnp.ndarray,
        activate_rate: float,
        deactivate_rate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation with sticky oscillatory protection.

        Key difference from critical_period:
        - Oscillatory functions use sticky_deactivate_rate during confirmation
        - Confirmed oscillatory functions are fully protected
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        protected_oscillatory = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            if mask[i] < 0.5:
                # Inactive - might activate
                if activate_probs[i] < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                is_oscillatory = i in OSCILLATORY_INDICES

                # Protection logic
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    # Full protection for confirmed activations
                    if confirmed[i]:
                        if is_oscillatory:
                            protected_oscillatory.append(i)
                        continue

                # Get deactivation rate based on type
                if is_oscillatory:
                    # STICKY: Oscillatory uses much lower rate
                    if phase == CriticalPeriodPhase.CONFIRMATION:
                        # Key fix: Don't aggressively prune oscillatory during confirmation
                        actual_deact_rate = self.sticky_deactivate_rate
                    else:
                        actual_deact_rate = self.sticky_deactivate_rate
                    protected_oscillatory.append(i)
                else:
                    actual_deact_rate = deactivate_rate

                if deactivate_probs[i] < actual_deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
            'protected_oscillatory': protected_oscillatory,
            'activate_rate': activate_rate,
            'deactivate_rate': deactivate_rate,
            'sticky_rate': self.sticky_deactivate_rate,
        }

        return new_mask, mutation_info

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with critical period + sticky logic."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Determine phase (with early consolidation if solved)
        phase = self._get_phase(generation, new_best)
        activate_rate, deactivate_rate = self._get_rates(phase)

        phase_changed = phase != state['phase']

        # Update persistence and confirmation
        new_persistence, new_confirmed = self._update_persistence_and_confirmation(
            state['mask'],
            state['activation_persistence'],
            state['confirmed'],
        )

        # Track oscillatory discovery
        osc_discovery = dict(state['oscillatory_discovery_gen'])
        current_palette = self.get_active_palette(state)
        for i in OSCILLATORY_INDICES:
            if i in current_palette and osc_discovery[i] is None:
                osc_discovery[i] = generation

        # Apply mutation with sticky protection
        new_mask, mutation_info = self._mutate_palette_critical_sticky(
            subkey, state['mask'], phase, new_confirmed,
            activate_rate, deactivate_rate
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'activation_persistence': new_persistence,
            'confirmed': new_confirmed,
            'oscillatory_discovery_gen': osc_discovery,
        }

        # Stats
        n_confirmed = int(jnp.sum(new_confirmed))
        n_confirmed_osc = sum(1 for i in OSCILLATORY_INDICES if new_confirmed[i])
        active_osc = [i for i in OSCILLATORY_INDICES if new_mask[i] > 0.5]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': mask_to_indices(new_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            'n_confirmed': n_confirmed,
            'n_confirmed_oscillatory': n_confirmed_osc,
            'active_oscillatory': active_osc,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with oscillatory tracking."""
        palette = self.get_active_palette(state)
        confirmed = state['confirmed']

        confirmed_list = [i for i in range(NUM_ACTIVATIONS) if confirmed[i]]
        active_osc = [i for i in OSCILLATORY_INDICES if i in palette]
        confirmed_osc = [i for i in OSCILLATORY_INDICES if confirmed[i]]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'has_osc_adapt': 13 in palette,
            'has_receptive': 15 in palette,
            'phase': state['phase'],
            'generation': state['generation'],
            'n_confirmed': len(confirmed_list),
            'confirmed_activations': confirmed_list,
            'active_oscillatory': active_osc,
            'confirmed_oscillatory': confirmed_osc,
            'oscillatory_discovery_gen': state['oscillatory_discovery_gen'],
            'stagnation_count': state['stagnation_count'],
        }
