"""Strategy 12: Critical Period + Hebbian Learning (No Hard-Coding).

Combines critical period developmental phases with Hebbian learning to
discover what to protect WITHOUT hard-coding any function indices.

Key insight: Instead of hard-coding OSCILLATORY_INDICES, let Hebbian
learning discover which functions are valuable based on fitness signal.

Phases:
1. EXPLORATION (gen 0-20): High plasticity, rapid Hebbian weight updates
2. CONFIRMATION (gen 20-50): Prune based on Hebbian affinity (low = prune)
3. CONSOLIDATION (gen 50+): Lock in high-affinity functions

This should:
- Discover sin is important for parity (through fitness correlation)
- Discover tanh is important for classification (different problem)
- Generalize to ANY problem without domain knowledge

Biological analogy:
- Critical periods control WHEN to learn
- Hebbian plasticity learns WHAT is valuable
- Like a child learning which skills are useful through experience
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


class CriticalHebbianStrategy(PaletteEvolutionStrategy):
    """Critical periods with Hebbian-learned protection.

    NO HARD-CODED FUNCTION LISTS - learns what to protect from experience.

    How it works:
    1. Hebbian weights track which functions correlate with high fitness
    2. During confirmation, functions with HIGH Hebbian affinity are protected
    3. Functions with LOW affinity are aggressively pruned

    The key difference from CriticalSticky:
    - CriticalSticky: if i in [4,11,12,13,15]: protect (HARD-CODED)
    - CriticalHebbian: if hebbian_affinity[i] > threshold: protect (LEARNED)
    """

    name = "critical_hebbian"
    description = "Critical periods with Hebbian-learned protection (no hard-coding)"

    def __init__(
        self,
        # Critical period timing
        exploration_end: int = 30,  # Extended to allow more Hebbian learning
        confirmation_end: int = 60,
        # Phase-specific base rates
        exploration_activate: float = 0.35,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate_max: float = 0.15,  # Reduced from 0.20
        confirmation_deactivate_min: float = 0.01,  # Min deactivation for high-affinity
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # Hebbian learning parameters
        learning_rate: float = 0.20,  # Increased for faster affinity accumulation
        anti_hebbian_rate: float = 0.05,  # Reduced anti-learning
        affinity_protection_threshold: float = 0.55,  # Lowered threshold
        # Phase-specific Hebbian modulation
        exploration_lr_multiplier: float = 1.5,  # Learn faster during exploration
        confirmation_lr_multiplier: float = 0.5,  # Slower learning during confirmation
        # Early consolidation
        early_consolidation_threshold: float = 0.95,
        # Constraints
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            exploration_end: Generation when exploration ends
            confirmation_end: Generation when confirmation ends
            exploration_activate/deactivate: Rates during exploration
            confirmation_deactivate_max: Deactivation rate for LOW affinity functions
            confirmation_deactivate_min: Deactivation rate for HIGH affinity functions
            consolidation_activate/deactivate: Rates during consolidation
            learning_rate: Base Hebbian learning rate
            anti_hebbian_rate: Anti-Hebbian (failure) learning rate
            affinity_protection_threshold: Affinity above this gets protection
            exploration_lr_multiplier: Learning rate boost during exploration
            confirmation_lr_multiplier: Learning rate reduction during confirmation
            early_consolidation_threshold: Fitness threshold for early consolidation
            min_active: Minimum active functions
            initial_palette: Starting palette indices
        """
        # Critical period timing
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

        # Phase rates
        self.exploration_activate = exploration_activate
        self.exploration_deactivate = exploration_deactivate
        self.confirmation_activate = confirmation_activate
        self.confirmation_deactivate_max = confirmation_deactivate_max
        self.confirmation_deactivate_min = confirmation_deactivate_min
        self.consolidation_activate = consolidation_activate
        self.consolidation_deactivate = consolidation_deactivate

        # Hebbian parameters
        self.learning_rate = learning_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.affinity_protection_threshold = affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Other
        self.early_consolidation_threshold = early_consolidation_threshold
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase with early consolidation."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with Hebbian weight matrix."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Hebbian weight matrix - tracks pairwise function success
        # w[i,j] = how often i and j succeed together
        hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5

        # Individual function affinity - how often each function correlates with success
        # This is the KEY for protection decisions
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 121212),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Critical period
            'phase': CriticalPeriodPhase.EXPLORATION,
            # Hebbian state
            'hebbian_weights': hebbian_weights,
            'function_affinity': function_affinity,  # Per-function success correlation
            'fitness_history': [],
            'fitness_ema': 0.5,  # Exponential moving average for baseline
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update Hebbian weights and function affinity.

        Args:
            weights: Pairwise Hebbian weight matrix
            affinity: Per-function affinity scores
            mask: Active function mask
            fitness_signal: -1 to 1, positive = success
            phase: Current developmental phase

        Returns:
            Tuple of (new_weights, new_affinity)
        """
        # Phase-specific learning rate
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr = self.learning_rate * self.exploration_lr_multiplier
            anti_lr = self.anti_hebbian_rate * self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr = self.learning_rate * self.confirmation_lr_multiplier
            anti_lr = self.anti_hebbian_rate * self.confirmation_lr_multiplier
        else:  # CONSOLIDATION
            lr = self.learning_rate * 0.1  # Very slow learning
            anti_lr = self.anti_hebbian_rate * 0.1

        active = (mask > 0.5).astype(jnp.float32)

        # Update pairwise weights (which functions succeed TOGETHER)
        co_active = jnp.outer(active, active)
        if fitness_signal >= 0:
            weight_delta = lr * fitness_signal * co_active
        else:
            weight_delta = anti_lr * fitness_signal * co_active

        new_weights = jnp.clip(weights + weight_delta, 0.0, 1.0)

        # Update individual function affinity (which functions correlate with success)
        # This is the KEY metric for protection decisions
        if fitness_signal >= 0:
            affinity_delta = lr * fitness_signal * active
        else:
            affinity_delta = anti_lr * fitness_signal * active

        new_affinity = jnp.clip(affinity + affinity_delta, 0.0, 1.0)

        return new_weights, new_affinity

    def _compute_protection_scores(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection score for each function (0 to 1).

        High score = should be protected during confirmation.

        Combines:
        1. Individual affinity (how often this function correlates with success)
        2. Pairwise affinity (how well it works with other active functions)
        """
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        # Pairwise affinity with current active set
        pairwise_score = jnp.dot(weights, active) / n_active

        # Combine individual and pairwise (individual weighted more)
        protection = 0.7 * affinity + 0.3 * pairwise_score

        return protection

    def _mutate_palette_critical_hebbian(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation with Hebbian-learned protection.

        During CONFIRMATION:
        - High protection score → low deactivation rate (LEARNED, not hard-coded)
        - Low protection score → high deactivation rate
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        protection_info = {}

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        # Get phase-specific rates
        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = self.exploration_activate
            # During exploration, don't use protection - try everything
            use_protection = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = self.confirmation_activate
            use_protection = True
        else:  # CONSOLIDATION
            activate_rate = self.consolidation_activate
            use_protection = True

        for i in range(NUM_ACTIVATIONS):
            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                # Inactive - might activate
                # During confirmation, prefer activating high-affinity functions
                if use_protection and phase == CriticalPeriodPhase.CONFIRMATION:
                    # Boost activation rate for high-affinity functions
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    # In consolidation, protect everything above threshold
                    if protection >= self.affinity_protection_threshold:
                        protection_info[i] = f"protected (affinity={protection:.2f})"
                        continue
                    deact_rate = self.consolidation_deactivate

                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    # HEBBIAN-GUIDED PROTECTION (the key innovation)
                    # High affinity → low deactivation rate
                    # Low affinity → high deactivation rate
                    # Linear interpolation between min and max

                    if protection >= self.affinity_protection_threshold:
                        # High affinity - minimal pruning
                        deact_rate = self.confirmation_deactivate_min
                        protection_info[i] = f"protected (affinity={protection:.2f})"
                    else:
                        # Interpolate: lower affinity = higher deactivation
                        # affinity 0 → deactivate_max
                        # affinity threshold → deactivate_min
                        t = protection / self.affinity_protection_threshold
                        deact_rate = (
                            self.confirmation_deactivate_max * (1 - t) +
                            self.confirmation_deactivate_min * t
                        )
                        protection_info[i] = f"vulnerable (affinity={protection:.2f}, rate={deact_rate:.2f})"
                else:
                    # Exploration - minimal deactivation
                    deact_rate = self.exploration_deactivate

                if deactivate_probs[i] < deact_rate:
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
            'protection_info': protection_info,
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
        """Update with critical period phases and Hebbian learning."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
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

        # Update fitness EMA for baseline
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness

        # Compute fitness signal for Hebbian update
        # Positive = above recent average, Negative = below
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Hebbian update (phase-modulated learning rate)
        new_weights, new_affinity = self._hebbian_update(
            state['hebbian_weights'],
            state['function_affinity'],
            state['mask'],
            fitness_signal,
            phase,
        )

        # Compute protection scores for mutation
        protection_scores = self._compute_protection_scores(
            new_affinity, new_weights, state['mask']
        )

        # Apply mutation with Hebbian-learned protection
        new_mask, mutation_info = self._mutate_palette_critical_hebbian(
            subkey, state['mask'], phase, protection_scores
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track fitness history
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
            'phase': phase,
            'hebbian_weights': new_weights,
            'function_affinity': new_affinity,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
        }

        # Compute stats
        active_palette = mask_to_indices(new_mask)
        protected_functions = [
            i for i in active_palette
            if protection_scores[i] >= self.affinity_protection_threshold
        ]

        # Check if sin (index 4) has high affinity (for tracking, not hard-coded behavior)
        sin_affinity = float(new_affinity[4]) if 4 < len(new_affinity) else 0.0

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            'fitness_signal': fitness_signal,
            # Hebbian stats
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'sin_affinity': sin_affinity,  # For tracking (not used in logic)
            'n_protected': len(protected_functions),
            'protected_functions': protected_functions,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with learned affinities."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']

        # Find highest affinity functions (what the strategy LEARNED is important)
        top_indices = jnp.argsort(affinity)[-5:][::-1]  # Top 5
        top_affinities = [(int(i), float(affinity[i])) for i in top_indices]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'phase': state['phase'],
            'generation': state['generation'],
            # Learned affinities (no hard-coding!)
            'top_affinity_functions': top_affinities,
            'sin_affinity': float(affinity[4]),  # For comparison
            'avg_affinity': float(jnp.mean(affinity)),
            'stagnation_count': state['stagnation_count'],
        }
