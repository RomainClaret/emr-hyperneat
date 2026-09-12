"""Strategy 9: Critical Period Discovery.

Bio-inspired developmental windows with phase-specific plasticity.

Inspired by critical periods in neurodevelopment where:
- Early life: High plasticity, rapid learning
- Adolescence: Refinement and pruning
- Adulthood: Stable, consolidated circuits

Three phases:
1. Exploration (gen 0-20): 35% activate, 2% deactivate - aggressive discovery
2. Confirmation (gen 20-50): 10% activate, 15% deactivate - prune unhelpful
3. Consolidation (gen 50+): 2% activate, 1% deactivate - lock confirmed

Expected: 2-4 gen discovery (fastest), 80-90% solve
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
    """Enum-like for critical period phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class CriticalPeriodStrategy(PaletteEvolutionStrategy):
    """Critical period palette evolution with developmental phases.

    Three developmental phases:
    1. EXPLORATION: High activation rate, minimal deactivation
       - Goal: Discover as many useful activations as possible
       - Rates: 35% activate, 2% deactivate

    2. CONFIRMATION: Moderate activation, high deactivation
       - Goal: Prune unhelpful activations, confirm useful ones
       - Activations that consistently help survive
       - Rates: 10% activate, 15% deactivate

    3. CONSOLIDATION: Minimal changes, lock in confirmed activations
       - Goal: Stable palette for final optimization
       - Rates: 2% activate, 1% deactivate
    """

    name = "critical_period"
    description = "Developmental phases: explore → confirm → consolidate"

    def __init__(
        self,
        # Phase boundaries (generations)
        exploration_end: int = 20,
        confirmation_end: int = 50,
        # Phase-specific rates
        exploration_activate: float = 0.35,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate: float = 0.15,
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # Confirmation tracking
        confirmation_persistence: int = 5,  # Gens to confirm an activation
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
            confirmation_persistence: Gens activation must be present to confirm
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

        self.confirmation_persistence = confirmation_persistence
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _get_phase(self, generation: int) -> str:
        """Determine current developmental phase."""
        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def _get_rates(self, phase: str) -> Tuple[float, float]:
        """Get activation/deactivation rates for current phase."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            return self.exploration_activate, self.exploration_deactivate
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            return self.confirmation_activate, self.confirmation_deactivate
        else:  # CONSOLIDATION
            return self.consolidation_activate, self.consolidation_deactivate

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with confirmation tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Track how long each activation has been active (for confirmation)
        activation_persistence = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        # Set initial activations as already present
        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                activation_persistence = activation_persistence.at[i].set(1)

        # Confirmed activations (protected in consolidation)
        confirmed = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 99999),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Critical period state
            'phase': CriticalPeriodPhase.EXPLORATION,
            'activation_persistence': activation_persistence,
            'confirmed': confirmed,
            # Fitness tracking per activation
            'activation_fitness_sum': jnp.zeros(NUM_ACTIVATIONS),
            'activation_fitness_count': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_persistence_and_confirmation(
        self,
        mask: jnp.ndarray,
        persistence: jnp.ndarray,
        confirmed: jnp.ndarray,
        best_fitness: float,
        fitness_sum: jnp.ndarray,
        fitness_count: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update persistence counters and confirm long-present activations.

        Args:
            mask: Current active mask
            persistence: How long each activation has been active
            confirmed: Already confirmed activations
            best_fitness: This generation's best fitness
            fitness_sum/count: Accumulated fitness per activation

        Returns:
            Tuple of (new_persistence, new_confirmed, new_fitness_sum, new_count)
        """
        active = mask > 0.5

        # Update persistence: increment for active, reset for inactive
        new_persistence = jnp.where(
            active,
            persistence + 1,
            jnp.zeros_like(persistence)
        )

        # Update fitness tracking for active activations
        new_fitness_sum = jnp.where(active, fitness_sum + best_fitness, fitness_sum)
        new_fitness_count = jnp.where(active, fitness_count + 1, fitness_count)

        # Confirm activations that have been present long enough
        long_enough = new_persistence >= self.confirmation_persistence
        new_confirmed = jnp.logical_or(confirmed, long_enough)

        return new_persistence, new_confirmed, new_fitness_sum, new_fitness_count

    def _mutate_palette_critical(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        confirmed: jnp.ndarray,
        activate_rate: float,
        deactivate_rate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation.

        Args:
            key: JAX random key
            mask: Current palette mask
            phase: Current developmental phase
            confirmed: Confirmed (protected) activations
            activate_rate: Phase-specific activation rate
            deactivate_rate: Phase-specific deactivation rate

        Returns:
            Tuple of (new_mask, mutation_info)
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

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
                # In consolidation, confirmed activations are protected
                if phase == CriticalPeriodPhase.CONSOLIDATION and confirmed[i]:
                    continue

                if deactivate_probs[i] < deactivate_rate:
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
            'activate_rate': activate_rate,
            'deactivate_rate': deactivate_rate,
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
        """Update with critical period logic.

        1. Determine current phase
        2. Update persistence and confirmation tracking
        3. Apply phase-appropriate mutation
        """
        key, subkey = jax.random.split(state['rng_key'])

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
        activate_rate, deactivate_rate = self._get_rates(phase)

        # Track phase transition
        phase_changed = phase != state['phase']

        # Update persistence and confirmation
        new_persistence, new_confirmed, new_fitness_sum, new_fitness_count = \
            self._update_persistence_and_confirmation(
                state['mask'],
                state['activation_persistence'],
                state['confirmed'],
                best_fitness,
                state['activation_fitness_sum'],
                state['activation_fitness_count'],
            )

        # Apply phase-appropriate mutation
        new_mask, mutation_info = self._mutate_palette_critical(
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
            # Critical period state
            'phase': phase,
            'activation_persistence': new_persistence,
            'confirmed': new_confirmed,
            'activation_fitness_sum': new_fitness_sum,
            'activation_fitness_count': new_fitness_count,
        }

        # Compute stats
        n_confirmed = int(jnp.sum(new_confirmed))
        avg_persistence = float(jnp.mean(new_persistence[state['mask'] > 0.5]))

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': mask_to_indices(new_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Critical period stats
            'phase': phase,
            'phase_changed': phase_changed,
            'n_confirmed': n_confirmed,
            'avg_persistence': avg_persistence,
        }

        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including phase and confirmation stats."""
        palette = self.get_active_palette(state)
        confirmed = state['confirmed']
        persistence = state['activation_persistence']

        # List confirmed activations
        confirmed_list = [i for i in range(NUM_ACTIVATIONS) if confirmed[i]]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'phase': state['phase'],
            'generation': state['generation'],
            'n_confirmed': len(confirmed_list),
            'confirmed_activations': confirmed_list,
            'stagnation_count': state['stagnation_count'],
        }
