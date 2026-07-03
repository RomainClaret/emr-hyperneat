"""Strategy 14: Decaying Hebbian.

Extends CriticalHebbian with activity-dependent decay.

Problem solved: Once sin reaches high affinity, it never decays even if unused.
This prevents adaptation when switching between problem types (e.g., parity → classification).

Solution: Use-it-or-lose-it decay - inactive functions slowly lose affinity.

Biological rationale:
- Synaptic proteins have finite half-lives
- Without activity, synapses weaken over time
- "What fires together, wires together" requires ongoing firing
- Long-term potentiation requires maintenance through activity

Key mechanisms:
1. Base decay: All affinities slowly decay toward neutral (0.5)
2. Activity reduction: Active functions decay slower
3. Recent use window: Functions used recently decay much slower

Expected improvement over CriticalHebbian:
- Better adaptation when switching problem types
- Prevents "locked in" functions that are no longer useful
- More dynamic palette that reflects current utility
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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


class DecayingHebbianStrategy(PaletteEvolutionStrategy):
    """CriticalHebbian with use-it-or-lose-it decay.

    Adds activity-dependent affinity decay to enable problem adaptation.
    """

    name = "decaying_hebbian"
    description = "CriticalHebbian with activity-dependent affinity decay"

    def __init__(
        self,
        # Critical period timing
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # Phase rates
        exploration_activate: float = 0.35,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate_max: float = 0.15,
        confirmation_deactivate_min: float = 0.01,
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # Hebbian parameters
        learning_rate: float = 0.20,
        anti_hebbian_rate: float = 0.05,
        affinity_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # DECAY PARAMETERS (NEW)
        base_decay_rate: float = 0.02,           # 2% decay per generation when inactive
        decay_target: float = 0.5,               # Decay toward neutral
        activity_decay_reduction: float = 0.8,   # Active functions decay 80% slower
        recent_use_window: int = 10,             # Generations to count as "recently used"
        recent_use_decay_reduction: float = 0.5, # Recently used functions decay 50% slower
        # Decay phase modulation
        exploration_decay_multiplier: float = 0.5,   # Less decay during exploration
        consolidation_decay_multiplier: float = 0.2, # Much less decay after consolidation
        # Other
        early_consolidation_threshold: float = 0.95,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy with decay parameters."""
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

        # Decay parameters
        self.base_decay_rate = base_decay_rate
        self.decay_target = decay_target
        self.activity_decay_reduction = activity_decay_reduction
        self.recent_use_window = recent_use_window
        self.recent_use_decay_reduction = recent_use_decay_reduction
        self.exploration_decay_multiplier = exploration_decay_multiplier
        self.consolidation_decay_multiplier = consolidation_decay_multiplier

        # Other
        self.early_consolidation_threshold = early_consolidation_threshold
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with decay tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Track recent usage for each function
        # List of generation numbers when each function was last active
        recent_activity = {i: [] for i in range(NUM_ACTIVATIONS)}

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 141414),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'hebbian_weights': hebbian_weights,
            'function_affinity': function_affinity,
            'fitness_history': [],
            'fitness_ema': 0.5,
            # Decay tracking
            'recent_activity': recent_activity,
            'decay_applied_total': 0.0,
            'affinities': function_affinity,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _apply_decay(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        generation: int,
        recent_activity: Dict[int, List[int]],
        phase: str,
    ) -> Tuple[jnp.ndarray, float]:
        """Apply activity-dependent decay.

        Returns: (new_affinity, total_decay_applied)
        """
        # Phase-modulated decay rate
        if phase == CriticalPeriodPhase.EXPLORATION:
            decay_multiplier = self.exploration_decay_multiplier
        elif phase == CriticalPeriodPhase.CONSOLIDATION:
            decay_multiplier = self.consolidation_decay_multiplier
        else:  # CONFIRMATION
            decay_multiplier = 1.0

        effective_decay_rate = self.base_decay_rate * decay_multiplier

        new_affinity = affinity.copy()
        total_decay = 0.0

        active = (mask > 0.5).astype(jnp.float32)

        for i in range(NUM_ACTIVATIONS):
            current_affinity = float(affinity[i])

            # Compute distance to target
            distance = current_affinity - self.decay_target

            # Only decay if not at target
            if abs(distance) < 0.01:
                continue

            # Base decay
            decay = effective_decay_rate * distance

            # Reduce decay if currently active
            if active[i] > 0.5:
                decay *= (1.0 - self.activity_decay_reduction)

            # Reduce decay if recently used
            recent_gens = recent_activity.get(i, [])
            recent_use_count = sum(1 for g in recent_gens if generation - g <= self.recent_use_window)
            if recent_use_count > 0:
                # More recent use = less decay
                recent_factor = min(1.0, recent_use_count / self.recent_use_window)
                decay *= (1.0 - self.recent_use_decay_reduction * recent_factor)

            # Apply decay
            new_value = current_affinity - decay
            new_value = max(0.05, min(0.95, new_value))  # Soft bounds

            new_affinity = new_affinity.at[i].set(new_value)
            total_decay += abs(decay)

        return new_affinity, total_decay

    def _update_recent_activity(
        self,
        recent_activity: Dict[int, List[int]],
        mask: jnp.ndarray,
        generation: int,
    ) -> Dict[int, List[int]]:
        """Update recent activity tracking."""
        new_activity = {}

        for i in range(NUM_ACTIVATIONS):
            # Get existing history
            history = recent_activity.get(i, [])

            # Prune old entries
            history = [g for g in history if generation - g <= self.recent_use_window * 2]

            # Add current generation if active
            if mask[i] > 0.5:
                history.append(generation)

            new_activity[i] = history

        return new_activity

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update Hebbian weights (same as CriticalHebbian)."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr = self.learning_rate * self.exploration_lr_multiplier
            anti_lr = self.anti_hebbian_rate * self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr = self.learning_rate * self.confirmation_lr_multiplier
            anti_lr = self.anti_hebbian_rate * self.confirmation_lr_multiplier
        else:
            lr = self.learning_rate * 0.1
            anti_lr = self.anti_hebbian_rate * 0.1

        active = (mask > 0.5).astype(jnp.float32)

        co_active = jnp.outer(active, active)
        if fitness_signal >= 0:
            weight_delta = lr * fitness_signal * co_active
        else:
            weight_delta = anti_lr * fitness_signal * co_active

        new_weights = jnp.clip(weights + weight_delta, 0.0, 1.0)

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
        """Compute protection score for each function."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        pairwise_score = jnp.dot(weights, active) / n_active
        protection = 0.7 * affinity + 0.3 * pairwise_score

        return protection

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        protection_info = {}

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = self.exploration_activate
            use_protection = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = self.confirmation_activate
            use_protection = True
        else:
            activate_rate = self.consolidation_activate
            use_protection = True

        for i in range(NUM_ACTIVATIONS):
            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                if use_protection and phase == CriticalPeriodPhase.CONFIRMATION:
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= self.affinity_protection_threshold:
                        protection_info[i] = f"protected (affinity={protection:.2f})"
                        continue
                    deact_rate = self.consolidation_deactivate

                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= self.affinity_protection_threshold:
                        deact_rate = self.confirmation_deactivate_min
                        protection_info[i] = f"protected (affinity={protection:.2f})"
                    else:
                        t = protection / self.affinity_protection_threshold
                        deact_rate = (
                            self.confirmation_deactivate_max * (1 - t) +
                            self.confirmation_deactivate_min * t
                        )
                        protection_info[i] = f"vulnerable (affinity={protection:.2f})"
                else:
                    deact_rate = self.exploration_deactivate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'protection_info': protection_info,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with Hebbian learning and decay."""
        key, subkey = jax.random.split(state['rng_key'])

        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']

        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness

        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Update recent activity BEFORE decay
        new_recent_activity = self._update_recent_activity(
            state['recent_activity'], state['mask'], generation
        )

        # Hebbian update
        new_weights, new_affinity = self._hebbian_update(
            state['hebbian_weights'],
            state['function_affinity'],
            state['mask'],
            fitness_signal,
            phase,
        )

        # Apply decay AFTER Hebbian (decay counteracts Hebbian over time)
        new_affinity, decay_applied = self._apply_decay(
            new_affinity,
            state['mask'],
            generation,
            new_recent_activity,
            phase,
        )

        protection_scores = self._compute_protection_scores(
            new_affinity, new_weights, state['mask']
        )

        new_mask, mutation_info = self._mutate_palette(
            subkey, state['mask'], phase, protection_scores
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        total_decay = state['decay_applied_total'] + decay_applied

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
            'recent_activity': new_recent_activity,
            'decay_applied_total': total_decay,
            'affinities': new_affinity,
        }

        active_palette = mask_to_indices(new_mask)
        protected_functions = [
            i for i in active_palette
            if protection_scores[i] >= self.affinity_protection_threshold
        ]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            'fitness_signal': fitness_signal,
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'sin_affinity': float(new_affinity[4]),
            'n_protected': len(protected_functions),
            'protected_functions': protected_functions,
            # Decay stats
            'decay_applied': decay_applied,
            'total_decay': total_decay,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with decay stats."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']

        top_indices = jnp.argsort(affinity)[-5:][::-1]
        top_affinities = [(int(i), float(affinity[i])) for i in top_indices]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'phase': state['phase'],
            'generation': state['generation'],
            'top_affinity_functions': top_affinities,
            'sin_affinity': float(affinity[4]),
            'avg_affinity': float(jnp.mean(affinity)),
            'stagnation_count': state['stagnation_count'],
            'total_decay': state['decay_applied_total'],
        }
