"""Strategy 17: Competitive Hebbian Learning.

Extends Hebbian learning with zero-sum resource competition.

Key insight: Standard Hebbian allows ALL successful functions to reach high
affinity simultaneously. This can lead to "god functions" that dominate and
prevent exploration of alternative solutions.

Biological Basis:
- Lateral inhibition in cortex: Winner-take-all dynamics
- Resource competition: Synapses compete for limited trophic factors
- Competitive learning networks: Soft WTA for unsupervised learning

For palette evolution:
- Total affinity across all functions is constrained (zero-sum)
- When one function gains affinity, others must lose
- Prevents all oscillatory functions from saturating at 1.0
- Forces specialization: discover the MOST important functions

Key mechanisms:
1. Affinity budget: Total affinity sum is maintained near target
2. Winner boost: Functions with highest fitness correlation get extra
3. Loser penalty: Functions with lowest correlation lose affinity
4. Redistribution: Affinity is redistributed, not created/destroyed

Expected improvement over standard Hebbian:
- More diverse palettes (no single dominant function)
- Better function ranking (clear winners emerge)
- Faster pruning of useless functions
- Prevents affinity saturation
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


class CompetitiveHebbianStrategy(PaletteEvolutionStrategy):
    """Competitive Hebbian learning with zero-sum affinity dynamics.

    Functions compete for limited affinity budget.
    """

    name = "competitive_hebbian"
    description = "Zero-sum Hebbian with lateral inhibition and resource competition"

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
        # Competition parameters
        target_affinity_sum: float = NUM_ACTIVATIONS * 0.5,  # Total affinity budget
        redistribution_rate: float = 0.15,        # How fast affinity redistributes
        winner_k: int = 3,                        # Top K functions to boost
        loser_k: int = 3,                         # Bottom K functions to penalize
        winner_boost: float = 1.3,                # Multiplicative boost for winners
        loser_penalty: float = 0.7,               # Multiplicative penalty for losers
        # Lateral inhibition
        inhibition_radius: float = 0.5,           # Similarity threshold for competition
        inhibition_strength: float = 0.1,         # Strength of lateral inhibition
        # Base Hebbian parameters
        learning_rate: float = 0.20,
        anti_hebbian_rate: float = 0.05,
        affinity_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Other
        early_consolidation_threshold: float = 0.95,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize competitive Hebbian strategy.

        Args:
            target_affinity_sum: Total affinity budget across all functions
            redistribution_rate: Speed of affinity redistribution
            winner_k: Number of top functions to boost
            loser_k: Number of bottom functions to penalize
            winner_boost: Multiplicative boost factor
            loser_penalty: Multiplicative penalty factor
            inhibition_radius: How similar functions must be to compete
            inhibition_strength: Strength of lateral inhibition
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

        # Competition parameters
        self.target_affinity_sum = target_affinity_sum
        self.redistribution_rate = redistribution_rate
        self.winner_k = winner_k
        self.loser_k = loser_k
        self.winner_boost = winner_boost
        self.loser_penalty = loser_penalty
        self.inhibition_radius = inhibition_radius
        self.inhibition_strength = inhibition_strength

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
        """Initialize state with competition tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize with uniform affinity summing to target
        initial_affinity = self.target_affinity_sum / NUM_ACTIVATIONS
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * initial_affinity

        # Pairwise weights
        hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5

        # Competition tracking: recent fitness contributions per function
        fitness_contributions = jnp.zeros(NUM_ACTIVATIONS)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 171717),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            # Hebbian state
            'function_affinity': function_affinity,
            'hebbian_weights': hebbian_weights,
            'fitness_history': [],
            'fitness_ema': 0.5,
            # Competition tracking
            'fitness_contributions': fitness_contributions,  # Per-function fitness credit
            'competition_events': 0,                         # Count of redistribution
            'winners': [],                                   # Recent winners
            'losers': [],                                    # Recent losers
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_fitness_contributions(
        self,
        mask: jnp.ndarray,
        fitness_signal: float,
        prev_contributions: jnp.ndarray,
        decay: float = 0.9,
    ) -> jnp.ndarray:
        """Compute per-function fitness contributions using credit assignment.

        Active functions share credit for fitness; contributions decay over time.
        """
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        # Decay previous contributions
        new_contributions = prev_contributions * decay

        # Add current generation's credit (split among active functions)
        credit_per_function = fitness_signal / n_active
        new_contributions = new_contributions + credit_per_function * active

        return new_contributions

    def _apply_competition(
        self,
        affinity: jnp.ndarray,
        fitness_contributions: jnp.ndarray,
        mask: jnp.ndarray,
        phase: str,
    ) -> Tuple[jnp.ndarray, List[int], List[int]]:
        """Apply competitive dynamics with winner boost and loser penalty.

        Returns: (new_affinity, winner_indices, loser_indices)
        """
        active = (mask > 0.5).astype(jnp.float32)
        active_indices = [i for i in range(NUM_ACTIVATIONS) if active[i] > 0.5]

        if len(active_indices) < self.winner_k + self.loser_k:
            # Not enough active functions for competition
            return affinity, [], []

        # Rank active functions by fitness contribution
        active_contributions = [(i, float(fitness_contributions[i])) for i in active_indices]
        active_contributions.sort(key=lambda x: x[1], reverse=True)

        # Identify winners and losers
        winners = [idx for idx, _ in active_contributions[:self.winner_k]]
        losers = [idx for idx, _ in active_contributions[-self.loser_k:]]

        # Phase-modulated competition strength
        if phase == CriticalPeriodPhase.EXPLORATION:
            competition_strength = 0.5  # Weaker during exploration
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            competition_strength = 1.0  # Full competition
        else:
            competition_strength = 0.3  # Mild during consolidation

        new_affinity = affinity.copy()

        # Boost winners
        for idx in winners:
            boost = 1.0 + (self.winner_boost - 1.0) * competition_strength * self.redistribution_rate
            new_affinity = new_affinity.at[idx].set(
                min(0.95, float(new_affinity[idx]) * boost)
            )

        # Penalize losers
        for idx in losers:
            penalty = 1.0 - (1.0 - self.loser_penalty) * competition_strength * self.redistribution_rate
            new_affinity = new_affinity.at[idx].set(
                max(0.05, float(new_affinity[idx]) * penalty)
            )

        return new_affinity, winners, losers

    def _apply_lateral_inhibition(
        self,
        affinity: jnp.ndarray,
        hebbian_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply lateral inhibition between similar functions.

        Functions with high co-occurrence (similar weights) inhibit each other.
        """
        new_affinity = affinity.copy()

        # Compute similarity matrix from hebbian weights
        # Functions that often co-occur are "similar" and should compete
        for i in range(NUM_ACTIVATIONS):
            for j in range(i + 1, NUM_ACTIVATIONS):
                similarity = float(hebbian_weights[i, j])

                if similarity > self.inhibition_radius:
                    # These functions compete - the weaker one gets inhibited
                    inhibition = self.inhibition_strength * (similarity - self.inhibition_radius)

                    if affinity[i] > affinity[j]:
                        # i is stronger, j gets inhibited
                        new_affinity = new_affinity.at[j].set(
                            max(0.05, float(new_affinity[j]) - inhibition)
                        )
                    else:
                        # j is stronger, i gets inhibited
                        new_affinity = new_affinity.at[i].set(
                            max(0.05, float(new_affinity[i]) - inhibition)
                        )

        return new_affinity

    def _normalize_affinity_sum(self, affinity: jnp.ndarray) -> jnp.ndarray:
        """Normalize affinity to maintain target sum (zero-sum constraint)."""
        current_sum = jnp.sum(affinity)

        if current_sum < 0.1:
            # Avoid division by zero
            return jnp.ones(NUM_ACTIVATIONS) * (self.target_affinity_sum / NUM_ACTIVATIONS)

        # Scale to target sum
        scale_factor = self.target_affinity_sum / current_sum
        new_affinity = affinity * scale_factor

        # Clip to valid range
        return jnp.clip(new_affinity, 0.05, 0.95)

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Standard Hebbian update (before competition)."""
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

        # Update individual affinity
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
        """Update with competitive Hebbian learning."""
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

        # Step 1: Update fitness contributions
        new_contributions = self._compute_fitness_contributions(
            state['mask'], fitness_signal, state['fitness_contributions']
        )

        # Step 2: Standard Hebbian update
        new_weights, new_affinity = self._hebbian_update(
            state['hebbian_weights'],
            state['function_affinity'],
            state['mask'],
            fitness_signal,
            phase,
        )

        # Step 3: Apply competition (winner boost, loser penalty)
        new_affinity, winners, losers = self._apply_competition(
            new_affinity, new_contributions, state['mask'], phase
        )

        # Step 4: Apply lateral inhibition
        new_affinity = self._apply_lateral_inhibition(new_affinity, new_weights)

        # Step 5: Normalize to maintain zero-sum constraint
        new_affinity = self._normalize_affinity_sum(new_affinity)

        # Compute protection scores
        protection_scores = self._compute_protection_scores(
            new_affinity, new_weights, state['mask']
        )

        # Apply mutation
        new_mask, mutation_info = self._mutate_palette(
            subkey, state['mask'], phase, protection_scores
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        competition_events = state['competition_events'] + (1 if winners else 0)

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'function_affinity': new_affinity,
            'hebbian_weights': new_weights,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'fitness_contributions': new_contributions,
            'competition_events': competition_events,
            'winners': winners,
            'losers': losers,
        }

        active_palette = mask_to_indices(new_mask)
        protected_functions = [
            i for i in active_palette
            if protection_scores[i] >= self.affinity_protection_threshold
        ]

        # Compute diversity metric (how spread out are affinities?)
        affinity_std = float(jnp.std(new_affinity))
        affinity_entropy = float(-jnp.sum(new_affinity * jnp.log(new_affinity + 1e-8)))

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
            # Competition stats
            'winners': winners,
            'losers': losers,
            'competition_events': competition_events,
            'affinity_sum': float(jnp.sum(new_affinity)),
            'affinity_std': affinity_std,
            'affinity_entropy': affinity_entropy,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with competition stats."""
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
            'competition_events': state['competition_events'],
            'recent_winners': state['winners'],
            'recent_losers': state['losers'],
            'affinity_sum': float(jnp.sum(affinity)),
        }
