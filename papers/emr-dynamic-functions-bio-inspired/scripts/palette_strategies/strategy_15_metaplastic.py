"""Strategy 15: Metaplastic Strategy.

Extends CriticalHebbian with adaptive learning rates based on evolutionary state.

Problem solved: CriticalHebbian has fixed learning rates that may be suboptimal
at different stages of evolution.

Solution: Metaplasticity - the brain learns how to learn.

Biological rationale:
- Metaplasticity: synaptic plasticity is itself plastic
- After critical periods, plasticity decreases
- Novelty increases plasticity
- Success stabilizes plasticity
- BCM (Bienenstock-Cooper-Munro) theory: sliding threshold

Key mechanisms:
1. Stagnation-boosted LR: If stuck, increase learning rate
2. Success-reduced LR: If improving, decrease learning rate
3. Sliding threshold: Protection threshold adapts to population fitness distribution
4. Phase-matched plasticity: Different plasticity rules for different phases

Expected improvement over CriticalHebbian:
- Faster escape from local optima (stagnation boost)
- More stable solutions (success reduction)
- Better adaptation to problem difficulty (sliding threshold)
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


class MetaplasticStrategy(PaletteEvolutionStrategy):
    """CriticalHebbian with adaptive learning rates (metaplasticity).

    Learning rate adapts to evolutionary state:
    - Stagnation → boost LR to escape local optima
    - Success → reduce LR to stabilize
    - Sliding threshold for protection decisions
    """

    name = "metaplastic"
    description = "CriticalHebbian with metaplastic adaptive learning rates"

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
        # Base Hebbian parameters
        base_learning_rate: float = 0.20,
        base_anti_hebbian_rate: float = 0.05,
        base_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # METAPLASTIC PARAMETERS (NEW)
        # Stagnation response
        stagnation_lr_boost: float = 1.5,       # Multiply LR by this when stagnating
        stagnation_threshold_gens: int = 10,    # Gens without improvement = stagnation
        max_stagnation_boost: float = 3.0,      # Cap the stagnation boost
        # Success response
        success_lr_reduction: float = 0.7,      # Multiply LR by this on improvement
        success_window: int = 5,                # Consider last N gens for success
        success_threshold: float = 0.8,         # % of recent gens with improvement
        # Sliding threshold
        threshold_adaptation_rate: float = 0.05, # How fast threshold adapts
        threshold_min: float = 0.40,             # Minimum protection threshold
        threshold_max: float = 0.70,             # Maximum protection threshold
        threshold_percentile: float = 0.70,      # Protect top 30% of affinities
        # Other
        early_consolidation_threshold: float = 0.95,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy with metaplastic parameters."""
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

        # Base Hebbian parameters
        self.base_learning_rate = base_learning_rate
        self.base_anti_hebbian_rate = base_anti_hebbian_rate
        self.base_protection_threshold = base_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Metaplastic parameters
        self.stagnation_lr_boost = stagnation_lr_boost
        self.stagnation_threshold_gens = stagnation_threshold_gens
        self.max_stagnation_boost = max_stagnation_boost
        self.success_lr_reduction = success_lr_reduction
        self.success_window = success_window
        self.success_threshold = success_threshold
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_percentile = threshold_percentile

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
        """Initialize state with metaplastic tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 151515),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'hebbian_weights': hebbian_weights,
            'function_affinity': function_affinity,
            'fitness_history': [],
            'fitness_ema': 0.5,
            # Metaplastic state
            'current_lr_multiplier': 1.0,        # Current effective LR multiplier
            'current_protection_threshold': self.base_protection_threshold,
            'improvement_history': [],            # Track improvements for success detection
            'lr_multiplier_history': [],         # Track LR changes
            'affinities': function_affinity,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_lr_multiplier(
        self,
        stagnation_count: int,
        improvement_history: List[bool],
        phase: str,
    ) -> float:
        """Compute metaplastic learning rate multiplier.

        Higher when stagnating, lower when succeeding.
        """
        # Phase base multiplier
        if phase == CriticalPeriodPhase.EXPLORATION:
            phase_mult = self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            phase_mult = self.confirmation_lr_multiplier
        else:
            phase_mult = 0.1  # Very low in consolidation

        # Stagnation boost
        if stagnation_count >= self.stagnation_threshold_gens:
            # More stagnation = higher boost (up to max)
            stagnation_factor = stagnation_count / self.stagnation_threshold_gens
            stagnation_boost = min(
                self.max_stagnation_boost,
                1.0 + (self.stagnation_lr_boost - 1.0) * stagnation_factor
            )
        else:
            stagnation_boost = 1.0

        # Success reduction
        if len(improvement_history) >= self.success_window:
            recent = improvement_history[-self.success_window:]
            improvement_rate = sum(recent) / len(recent)
            if improvement_rate >= self.success_threshold:
                success_mult = self.success_lr_reduction
            else:
                success_mult = 1.0
        else:
            success_mult = 1.0

        return phase_mult * stagnation_boost * success_mult

    def _update_protection_threshold(
        self,
        current_threshold: float,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> float:
        """Update protection threshold using sliding percentile.

        BCM-inspired: threshold tracks recent activity level.
        """
        active = (mask > 0.5).astype(jnp.float32)
        active_affinities = affinity * active

        # Get valid affinities (non-zero for active functions)
        valid_affinities = [float(affinity[i]) for i in range(NUM_ACTIVATIONS) if mask[i] > 0.5]

        if len(valid_affinities) < 2:
            return current_threshold

        # Compute target threshold as percentile of active affinities
        target = float(np.percentile(valid_affinities, self.threshold_percentile * 100))

        # Clip to bounds
        target = max(self.threshold_min, min(self.threshold_max, target))

        # Smooth adaptation
        new_threshold = (
            (1 - self.threshold_adaptation_rate) * current_threshold +
            self.threshold_adaptation_rate * target
        )

        return new_threshold

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        lr_multiplier: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update Hebbian weights with metaplastic learning rate."""
        lr = self.base_learning_rate * lr_multiplier
        anti_lr = self.base_anti_hebbian_rate * lr_multiplier

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
        protection_threshold: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation with adaptive threshold."""
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
                # Use adaptive threshold
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= protection_threshold:
                        protection_info[i] = f"protected (affinity={protection:.2f}, thresh={protection_threshold:.2f})"
                        continue
                    deact_rate = self.consolidation_deactivate

                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= protection_threshold:
                        deact_rate = self.confirmation_deactivate_min
                        protection_info[i] = f"protected (affinity={protection:.2f})"
                    else:
                        t = protection / protection_threshold
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
        """Update with metaplastic Hebbian learning."""
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

        # Update improvement history
        improvement_history = state['improvement_history'] + [improved]
        if len(improvement_history) > self.success_window * 2:
            improvement_history = improvement_history[-self.success_window * 2:]

        # Compute metaplastic LR multiplier
        lr_multiplier = self._compute_lr_multiplier(
            new_stagnation, improvement_history, phase
        )

        # Update protection threshold (sliding)
        new_protection_threshold = self._update_protection_threshold(
            state['current_protection_threshold'],
            state['function_affinity'],
            state['mask'],
        )

        # Hebbian update with adaptive LR
        new_weights, new_affinity = self._hebbian_update(
            state['hebbian_weights'],
            state['function_affinity'],
            state['mask'],
            fitness_signal,
            lr_multiplier,
        )

        protection_scores = self._compute_protection_scores(
            new_affinity, new_weights, state['mask']
        )

        # Mutation with adaptive threshold
        new_mask, mutation_info = self._mutate_palette(
            subkey, state['mask'], phase, protection_scores, new_protection_threshold
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        lr_history = state['lr_multiplier_history'] + [lr_multiplier]
        if len(lr_history) > 50:
            lr_history = lr_history[-50:]

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
            'current_lr_multiplier': lr_multiplier,
            'current_protection_threshold': new_protection_threshold,
            'improvement_history': improvement_history,
            'lr_multiplier_history': lr_history,
            'affinities': new_affinity,
        }

        active_palette = mask_to_indices(new_mask)
        protected_functions = [
            i for i in active_palette
            if protection_scores[i] >= new_protection_threshold
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
            # Metaplastic stats
            'lr_multiplier': lr_multiplier,
            'protection_threshold': new_protection_threshold,
            'avg_lr': float(np.mean(lr_history)) if lr_history else lr_multiplier,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with metaplastic stats."""
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
            'current_lr_multiplier': state['current_lr_multiplier'],
            'protection_threshold': state['current_protection_threshold'],
        }
