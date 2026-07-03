"""Strategy 22: Eligibility Trace (Three-Factor Learning).

Implements dopamine-gated eligibility traces for temporal credit assignment.

Biological Basis:
- Eligibility traces: Local activity creates decaying "eligibility" mark
- Dopamine: Global reward signal that gates learning
- Three-factor rule: Learning = eligibility × dopamine × direction

Key Insight:
- STDP (Strategy 16) uses fixed 5-10 generation windows
- Eligibility traces provide CONTINUOUS, decaying credit assignment
- Functions active before reward get credit proportional to recency

Learning Rule:
    eligibility[i] = decay * eligibility[i-1] + activity[i]
    dopamine = fitness_improvement - baseline (reward prediction error)
    affinity[i] += lr * dopamine * eligibility[i]

This extends temporal credit assignment beyond STDP's hard windows,
linking function discovery to multi-generation fitness improvements.

Expected improvements:
- Better temporal credit assignment over long timescales
- Smoother learning than window-based approaches
- Natural recency bias without hard cutoffs
- Dopamine-gated learning prevents noise from affecting stable states
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


class EligibilityTraceStrategy(PaletteEvolutionStrategy):
    """Three-factor learning: eligibility × dopamine → credit.

    Eligibility traces solve the temporal credit assignment problem by
    maintaining a decaying record of which functions were recently active.
    When dopamine (reward signal) arrives, functions with high eligibility
    receive the most credit.
    """

    name = "eligibility_trace"
    description = "Three-factor dopamine-gated eligibility trace learning"

    def __init__(
        self,
        # Eligibility trace parameters
        eligibility_decay: float = 0.85,           # Trace decay per generation
        eligibility_boost_active: float = 1.0,     # Boost for active functions
        eligibility_boost_changed: float = 0.5,    # Extra boost when function added
        # Dopamine (reward) parameters
        dopamine_baseline_momentum: float = 0.9,   # EMA for baseline fitness
        dopamine_sensitivity: float = 1.5,         # How much fitness diff matters
        dopamine_learning_rate: float = 0.2,       # How much DA affects affinity
        # Protection
        affinity_protection_threshold: float = 0.6,
        protection_decay: float = 0.98,            # Slow decay of protected status
        # Mutation rates
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        da_exploration_modulation: float = 0.3,    # DA reduces exploration
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Eligibility Trace strategy.

        Args:
            eligibility_decay: How fast eligibility traces decay (0.85 = ~5 gen half-life)
            eligibility_boost_active: Eligibility boost for active functions
            eligibility_boost_changed: Extra boost when function just activated
            dopamine_baseline_momentum: Momentum for baseline fitness tracking
            dopamine_sensitivity: Scale factor for reward prediction error
            dopamine_learning_rate: Learning rate for affinity updates
            affinity_protection_threshold: Threshold for protection
            base_activate_rate: Base rate for activating new functions
            base_deactivate_rate: Base rate for deactivating functions
            da_exploration_modulation: How much DA reduces exploration
        """
        # Eligibility
        self.eligibility_decay = eligibility_decay
        self.eligibility_boost_active = eligibility_boost_active
        self.eligibility_boost_changed = eligibility_boost_changed

        # Dopamine
        self.dopamine_baseline_momentum = dopamine_baseline_momentum
        self.dopamine_sensitivity = dopamine_sensitivity
        self.dopamine_learning_rate = dopamine_learning_rate

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold
        self.protection_decay = protection_decay

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.da_exploration_modulation = da_exploration_modulation

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with eligibility traces."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Eligibility traces (one per function)
        eligibility = jnp.zeros(NUM_ACTIVATIONS)

        # Function affinity (learned value)
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 222222),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Eligibility traces
            'eligibility': eligibility,
            # Dopamine system
            'dopamine_baseline': 0.5,  # Expected fitness (for RPE)
            'dopamine_signal': 0.0,    # Current DA level
            # Learning state
            'function_affinity': function_affinity,
            # Tracking
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_eligibility(
        self,
        eligibility: jnp.ndarray,
        mask: jnp.ndarray,
        previous_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update eligibility traces with decay and activity boost.

        Eligibility = decay × previous_eligibility + current_activity

        Functions that were just activated get an extra boost.
        """
        # Decay all eligibility traces
        new_eligibility = self.eligibility_decay * eligibility

        # Boost for currently active functions
        active = (mask > 0.5).astype(jnp.float32)
        new_eligibility = new_eligibility + self.eligibility_boost_active * active

        # Extra boost for newly activated functions
        was_inactive = (previous_mask < 0.5).astype(jnp.float32)
        just_activated = active * was_inactive
        new_eligibility = new_eligibility + self.eligibility_boost_changed * just_activated

        # Clip to reasonable range
        return jnp.clip(new_eligibility, 0.0, 3.0)

    def _compute_dopamine(
        self,
        fitness: float,
        baseline: float,
    ) -> Tuple[float, float]:
        """Compute dopamine signal as reward prediction error.

        DA = sensitivity × (actual - expected) / expected

        Returns:
            (dopamine_signal, new_baseline)
        """
        # Update baseline (expected fitness)
        new_baseline = (
            self.dopamine_baseline_momentum * baseline +
            (1 - self.dopamine_baseline_momentum) * fitness
        )

        # Compute reward prediction error
        if baseline > 0.01:
            prediction_error = (fitness - baseline) / baseline
        else:
            prediction_error = fitness - baseline

        # Scale by sensitivity
        dopamine = self.dopamine_sensitivity * prediction_error

        # Clip to reasonable range
        dopamine = max(-1.0, min(1.0, dopamine))

        return dopamine, new_baseline

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        eligibility: jnp.ndarray,
        dopamine: float,
    ) -> jnp.ndarray:
        """Update affinity using three-factor rule.

        affinity[i] += lr × dopamine × eligibility[i]

        Only functions with high eligibility receive credit/blame.
        """
        # Three-factor update: learning_rate × dopamine × eligibility
        delta = self.dopamine_learning_rate * dopamine * eligibility

        # Apply update
        new_affinity = affinity + delta

        # Clip to valid range
        return jnp.clip(new_affinity, 0.05, 0.95)

    def _compute_protection(
        self,
        affinity: jnp.ndarray,
        eligibility: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection based on affinity and eligibility.

        High affinity = learned value
        High eligibility = recent activity
        Protection = weighted combination
        """
        # Protection from learned value (affinity)
        affinity_protection = affinity

        # Recent activity contributes somewhat
        eligibility_contribution = 0.2 * jnp.clip(eligibility / 2.0, 0, 1)

        # Combine
        protection = 0.8 * affinity_protection + 0.2 * eligibility_contribution

        return jnp.clip(protection, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        dopamine: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with dopamine-modulated rates.

        High dopamine (success) → less exploration (exploit)
        Low/negative dopamine (failure) → more exploration
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        # Dopamine modulates exploration
        # High DA = less exploration, Low DA = more exploration
        da_factor = 1.0 - self.da_exploration_modulation * dopamine
        da_factor = max(0.5, min(1.5, da_factor))

        effective_activate_rate = self.base_activate_rate * da_factor
        effective_deactivate_rate = self.base_deactivate_rate * da_factor

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            prot = float(protection[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                # Higher affinity/protection = more likely to activate
                rate = effective_activate_rate * (0.5 + 0.5 * prot)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                if prot >= self.affinity_protection_threshold:
                    # Protected: very low deactivation rate
                    rate = effective_deactivate_rate * 0.1
                else:
                    # Not protected: higher deactivation rate
                    rate = effective_deactivate_rate * (1.0 - prot)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'effective_activate_rate': effective_activate_rate,
            'effective_deactivate_rate': effective_deactivate_rate,
            'da_factor': da_factor,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with eligibility trace learning."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Compute dopamine signal (reward prediction error)
        dopamine, new_baseline = self._compute_dopamine(
            best_fitness,
            state['dopamine_baseline']
        )

        # Step 2: Update eligibility traces
        new_eligibility = self._update_eligibility(
            state['eligibility'],
            state['mask'],
            state['previous_mask'],
        )

        # Step 3: Three-factor learning update to affinity
        new_affinity = self._update_affinity(
            state['function_affinity'],
            new_eligibility,
            dopamine,
        )

        # Step 4: Compute protection
        protection = self._compute_protection(new_affinity, new_eligibility)

        # Step 5: Apply mutation with DA modulation
        new_mask, mutation_info = self._mutate_palette(
            subkey,
            state['mask'],
            protection,
            dopamine,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update fitness history
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
            # Eligibility traces
            'eligibility': new_eligibility,
            # Dopamine system
            'dopamine_baseline': new_baseline,
            'dopamine_signal': dopamine,
            # Learning state
            'function_affinity': new_affinity,
            # Tracking
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)
        protected = [
            i for i in active_palette
            if protection[i] >= self.affinity_protection_threshold
        ]

        # Top eligibility functions
        top_elig_idx = jnp.argsort(new_eligibility)[-3:][::-1]
        top_eligibility = [(int(i), float(new_eligibility[i])) for i in top_elig_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Dopamine
            'dopamine_signal': dopamine,
            'dopamine_baseline': new_baseline,
            # Eligibility
            'avg_eligibility': float(jnp.mean(new_eligibility)),
            'max_eligibility': float(jnp.max(new_eligibility)),
            'top_eligibility_functions': top_eligibility,
            'sin_eligibility': float(new_eligibility[4]),
            # Affinity
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'sin_affinity': float(new_affinity[4]),
            # Protection
            'n_protected': len(protected),
            'protected_functions': protected,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with eligibility trace stats."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']
        eligibility = state['eligibility']

        # Top functions by affinity
        top_aff_idx = jnp.argsort(affinity)[-5:][::-1]
        top_affinities = [(int(i), float(affinity[i])) for i in top_aff_idx]

        # Top functions by eligibility
        top_elig_idx = jnp.argsort(eligibility)[-5:][::-1]
        top_eligibilities = [(int(i), float(eligibility[i])) for i in top_elig_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Affinity
            'top_affinity_functions': top_affinities,
            'sin_affinity': float(affinity[4]),
            'avg_affinity': float(jnp.mean(affinity)),
            # Eligibility
            'top_eligibility_functions': top_eligibilities,
            'sin_eligibility': float(eligibility[4]),
            'avg_eligibility': float(jnp.mean(eligibility)),
            # Dopamine
            'dopamine_signal': state['dopamine_signal'],
            'dopamine_baseline': state['dopamine_baseline'],
        }
