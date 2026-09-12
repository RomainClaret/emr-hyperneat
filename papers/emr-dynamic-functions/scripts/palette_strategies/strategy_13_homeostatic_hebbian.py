"""Strategy 13: Homeostatic Hebbian.

Extends CriticalHebbian with homeostatic regulation to prevent affinity saturation.

Problem solved: CriticalHebbian affinities can saturate at 1.0 or 0.0, making
subsequent learning impossible.

Solution: Synaptic scaling + soft bounds.

Biological rationale:
- Real synapses maintain activity within functional ranges through synaptic scaling
- The brain normalizes synaptic strengths to prevent runaway potentiation/depression
- Homeostasis allows continued learning throughout lifetime

Key mechanisms:
1. Synaptic scaling: If mean affinity drifts too high/low, scale back toward target
2. Soft bounds: Instead of hard 0.0/1.0 clipping, asymptotic approach to bounds
3. Activity-dependent regulation: More active functions get stronger scaling

Expected improvement over CriticalHebbian:
- Fewer false protections (affinities don't saturate at 1.0)
- Better adaptation (affinities can decrease from 1.0 to indicate problems)
- More robust to early luck (initial high fitness doesn't lock in bad functions)
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


class HomeostaticHebbianStrategy(PaletteEvolutionStrategy):
    """CriticalHebbian with homeostatic regulation.

    Adds synaptic scaling and soft bounds to prevent affinity saturation.
    """

    name = "homeostatic_hebbian"
    description = "CriticalHebbian with homeostatic regulation for affinity stability"

    def __init__(
        self,
        # Critical period timing (same as CriticalHebbian)
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # Phase-specific base rates
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
        # HOMEOSTATIC PARAMETERS (NEW)
        target_mean_affinity: float = 0.5,       # Target for synaptic scaling
        scaling_rate: float = 0.08,              # How fast to scale toward target
        scaling_threshold: float = 0.15,         # Trigger scaling if mean differs by this
        # Soft bounds (instead of hard 0.0/1.0)
        affinity_ceiling: float = 0.95,          # Can't go above this
        affinity_floor: float = 0.05,            # Can't go below this
        soft_ceiling_start: float = 0.80,        # Start slowing learning here
        soft_floor_start: float = 0.20,          # Start slowing anti-learning here
        # Other
        early_consolidation_threshold: float = 0.95,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy with homeostatic parameters."""
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

        # Homeostatic parameters
        self.target_mean_affinity = target_mean_affinity
        self.scaling_rate = scaling_rate
        self.scaling_threshold = scaling_threshold
        self.affinity_ceiling = affinity_ceiling
        self.affinity_floor = affinity_floor
        self.soft_ceiling_start = soft_ceiling_start
        self.soft_floor_start = soft_floor_start

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
        """Initialize state with Hebbian weights and homeostatic tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Hebbian weights
        hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 131313),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            # Hebbian state
            'hebbian_weights': hebbian_weights,
            'function_affinity': function_affinity,
            'fitness_history': [],
            'fitness_ema': 0.5,
            # Homeostatic tracking
            'scaling_applied_count': 0,
            'affinities': function_affinity,  # Alias for compatibility
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _apply_soft_bounds(self, affinity: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """Apply soft bounds to affinity updates.

        Instead of hard clipping at 0/1, gradually reduce learning rate
        as we approach bounds, preventing saturation.
        """
        # For positive delta (increasing affinity)
        positive_mask = delta > 0
        # Compute distance to ceiling
        dist_to_ceiling = self.affinity_ceiling - affinity
        # Scale down positive deltas as we approach ceiling
        ceiling_factor = jnp.where(
            affinity > self.soft_ceiling_start,
            dist_to_ceiling / (self.affinity_ceiling - self.soft_ceiling_start),
            jnp.ones_like(affinity)
        )
        ceiling_factor = jnp.clip(ceiling_factor, 0.0, 1.0)

        # For negative delta (decreasing affinity)
        # Compute distance to floor
        dist_to_floor = affinity - self.affinity_floor
        # Scale down negative deltas as we approach floor
        floor_factor = jnp.where(
            affinity < self.soft_floor_start,
            dist_to_floor / (self.soft_floor_start - self.affinity_floor),
            jnp.ones_like(affinity)
        )
        floor_factor = jnp.clip(floor_factor, 0.0, 1.0)

        # Apply appropriate factor based on delta sign
        scaled_delta = jnp.where(
            positive_mask,
            delta * ceiling_factor,
            delta * floor_factor
        )

        # Apply update and clip to hard bounds as safety net
        new_affinity = affinity + scaled_delta
        return jnp.clip(new_affinity, self.affinity_floor, self.affinity_ceiling)

    def _apply_synaptic_scaling(self, affinity: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Apply synaptic scaling to maintain homeostasis.

        If mean affinity of active functions drifts too far from target,
        scale all affinities back toward the target mean.
        """
        active = (mask > 0.5).astype(jnp.float32)
        n_active = jnp.sum(active)

        # Compute mean affinity of active functions
        active_affinity_sum = jnp.sum(affinity * active)
        mean_active_affinity = jnp.where(
            n_active > 0,
            active_affinity_sum / n_active,
            self.target_mean_affinity
        )

        # Check if scaling is needed
        drift = mean_active_affinity - self.target_mean_affinity
        scaling_needed = jnp.abs(drift) > self.scaling_threshold

        # Scale toward target (multiplicative scaling)
        if scaling_needed:
            # Scale factor to bring mean toward target
            if mean_active_affinity > self.target_mean_affinity:
                # Affinities too high - scale down
                scale = 1.0 - self.scaling_rate * (drift / mean_active_affinity)
            else:
                # Affinities too low - scale up
                scale = 1.0 - self.scaling_rate * (drift / (1.0 - mean_active_affinity + 0.01))

            scale = jnp.clip(scale, 0.9, 1.1)  # Limit scaling magnitude

            # Apply scaling only to active functions
            scaled_affinity = jnp.where(
                active > 0.5,
                self.target_mean_affinity + (affinity - self.target_mean_affinity) * scale,
                affinity
            )
            return jnp.clip(scaled_affinity, self.affinity_floor, self.affinity_ceiling)

        return affinity

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, bool]:
        """Update Hebbian weights with homeostatic regulation.

        Returns: (new_weights, new_affinity, scaling_applied)
        """
        # Phase-specific learning rate
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

        # Update pairwise weights
        co_active = jnp.outer(active, active)
        if fitness_signal >= 0:
            weight_delta = lr * fitness_signal * co_active
        else:
            weight_delta = anti_lr * fitness_signal * co_active

        new_weights = jnp.clip(weights + weight_delta, 0.0, 1.0)

        # Compute affinity delta
        if fitness_signal >= 0:
            affinity_delta = lr * fitness_signal * active
        else:
            affinity_delta = anti_lr * fitness_signal * active

        # Apply soft bounds (KEY HOMEOSTATIC MECHANISM)
        new_affinity = self._apply_soft_bounds(affinity, affinity_delta)

        # Apply synaptic scaling (SECOND HOMEOSTATIC MECHANISM)
        pre_scaling_mean = float(jnp.mean(new_affinity * active) / max(jnp.sum(active), 1))
        new_affinity = self._apply_synaptic_scaling(new_affinity, mask)
        post_scaling_mean = float(jnp.mean(new_affinity * active) / max(jnp.sum(active), 1))

        scaling_applied = abs(pre_scaling_mean - post_scaling_mean) > 0.001

        return new_weights, new_affinity, scaling_applied

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
        """Update with homeostatic Hebbian learning."""
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

        # Hebbian update with homeostatic regulation
        new_weights, new_affinity, scaling_applied = self._hebbian_update(
            state['hebbian_weights'],
            state['function_affinity'],
            state['mask'],
            fitness_signal,
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

        scaling_count = state['scaling_applied_count'] + (1 if scaling_applied else 0)

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
            'scaling_applied_count': scaling_count,
            'affinities': new_affinity,
        }

        active_palette = mask_to_indices(new_mask)
        protected_functions = [
            i for i in active_palette
            if protection_scores[i] >= self.affinity_protection_threshold
        ]

        # Compute homeostatic stats
        active_mask = (state['mask'] > 0.5).astype(jnp.float32)
        active_affinities = new_affinity * active_mask
        mean_active = float(jnp.sum(active_affinities) / max(jnp.sum(active_mask), 1))

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
            'min_affinity': float(jnp.min(new_affinity)),
            'sin_affinity': float(new_affinity[4]),
            'n_protected': len(protected_functions),
            'protected_functions': protected_functions,
            # Homeostatic stats
            'scaling_applied': scaling_applied,
            'total_scalings': scaling_count,
            'mean_active_affinity': mean_active,
            'affinity_range': float(jnp.max(new_affinity) - jnp.min(new_affinity)),
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with homeostatic stats."""
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
            'total_scalings': state['scaling_applied_count'],
        }
