"""Strategy 16: STDP (Spike-Timing-Dependent Plasticity).

Extends CriticalHebbian with temporal order-dependent learning.

Key insight: Current Hebbian strategies track co-occurrence but not SEQUENCE.
Functions that are active BEFORE a fitness improvement matter more than
those active during or after.

Biological Basis:
- In real neurons, the timing of pre- and post-synaptic activity matters
- Pre-before-post → Long-Term Potentiation (LTP, strengthening)
- Post-before-pre → Long-Term Depression (LTD, weakening)
- This captures causality: what CAUSED the improvement?

For palette evolution:
- Track which functions were active in generations BEFORE fitness improvement
- Functions consistently present before success get higher affinity
- Functions that appear after success get weaker boost

Key mechanisms:
1. Activation history: Track last N generations of active palettes
2. Temporal affinity: Compute affinity based on position relative to improvement
3. LTP window: Functions active 1-5 gens before improvement get boost
4. LTD window: Functions active 1-3 gens after (or never near) get penalty

Expected improvement over Hebbian:
- Discovers function SEQUENCES, not just sets
- Better credit assignment (what caused improvement vs what was along for the ride)
- More robust to spurious correlations
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque
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


class STDPStrategy(PaletteEvolutionStrategy):
    """STDP-based palette evolution with temporal credit assignment.

    Learns which functions PRECEDE fitness improvements (causality).
    """

    name = "stdp"
    description = "Spike-timing-dependent plasticity with temporal credit assignment"

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
        # STDP window parameters
        ltp_window: int = 5,              # Generations before improvement to boost
        ltd_window: int = 3,              # Generations after for depression
        history_length: int = 10,         # Total history to maintain
        # STDP learning rates
        ltp_rate: float = 0.25,           # LTP strengthening rate
        ltd_rate: float = 0.10,           # LTD weakening rate
        # Temporal weighting (closer to improvement = stronger effect)
        temporal_decay: float = 0.7,      # Decay factor per generation from improvement
        # Protection threshold
        affinity_protection_threshold: float = 0.55,
        # Phase modulation
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Other
        early_consolidation_threshold: float = 0.95,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize STDP strategy.

        Args:
            ltp_window: How many generations before improvement get LTP
            ltd_window: How many generations after improvement get LTD
            history_length: Total generations of palette history to track
            ltp_rate: Learning rate for potentiation
            ltd_rate: Learning rate for depression
            temporal_decay: Decay per generation (closer = stronger)
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

        # STDP parameters
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.history_length = history_length
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.temporal_decay = temporal_decay

        # Other
        self.affinity_protection_threshold = affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier
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
        """Initialize state with STDP tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Function affinity (what STDP learns)
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Pairwise weights for compatibility with protection score computation
        hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 161616),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            # STDP state
            'function_affinity': function_affinity,
            'hebbian_weights': hebbian_weights,
            # History tracking for STDP
            'palette_history': [],        # List of (generation, active_mask, fitness)
            'fitness_history': [],
            'fitness_ema': 0.5,
            # STDP-specific tracking
            'ltp_events': 0,              # Count of LTP updates
            'ltd_events': 0,              # Count of LTD updates
            'sequence_discoveries': [],   # Track discovered sequences
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_temporal_weight(self, generations_from_improvement: int) -> float:
        """Compute temporal weight based on distance from improvement event.

        Closer to improvement = stronger effect.
        """
        return self.temporal_decay ** abs(generations_from_improvement)

    def _stdp_update(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        palette_history: List[Tuple[int, jnp.ndarray, float]],
        current_gen: int,
        fitness_improved: bool,
        improvement_magnitude: float,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Apply STDP update based on temporal position relative to improvement.

        Args:
            affinity: Current function affinity
            weights: Current pairwise weights
            palette_history: History of (gen, mask, fitness) tuples
            current_gen: Current generation
            fitness_improved: Whether fitness improved this generation
            improvement_magnitude: How much fitness improved (normalized)
            phase: Current developmental phase

        Returns:
            Tuple of (new_affinity, new_weights, update_info)
        """
        # Phase-specific learning rate modulation
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr_mult = self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr_mult = self.confirmation_lr_multiplier
        else:
            lr_mult = 0.1  # Very slow in consolidation

        new_affinity = affinity.copy()
        new_weights = weights.copy()

        ltp_functions = []
        ltd_functions = []

        if fitness_improved and len(palette_history) >= 2:
            # LTP: Boost functions that were active BEFORE improvement
            # Look at history from ltp_window generations ago

            for hist_gen, hist_mask, hist_fitness in palette_history:
                gens_before = current_gen - hist_gen

                if 1 <= gens_before <= self.ltp_window:
                    # This generation preceded improvement - apply LTP
                    temporal_weight = self._compute_temporal_weight(gens_before)
                    ltp_delta = (
                        self.ltp_rate * lr_mult *
                        temporal_weight *
                        improvement_magnitude *
                        (hist_mask > 0.5).astype(jnp.float32)
                    )
                    new_affinity = jnp.clip(new_affinity + ltp_delta, 0.0, 1.0)

                    # Track which functions got LTP
                    ltp_indices = [i for i in range(NUM_ACTIVATIONS) if hist_mask[i] > 0.5]
                    ltp_functions.extend(ltp_indices)

                    # Update pairwise weights for functions that succeeded together
                    active = (hist_mask > 0.5).astype(jnp.float32)
                    co_active = jnp.outer(active, active)
                    weight_delta = self.ltp_rate * lr_mult * temporal_weight * improvement_magnitude * co_active
                    new_weights = jnp.clip(new_weights + weight_delta, 0.0, 1.0)

        # LTD: Weaken functions that were active but didn't precede improvement
        # This applies when we have history but no improvement
        if not fitness_improved and len(palette_history) >= self.ltd_window:
            # Functions that were active recently but no improvement came
            recent_mask = palette_history[-1][1] if palette_history else jnp.zeros(NUM_ACTIVATIONS)
            active = (recent_mask > 0.5).astype(jnp.float32)

            # Gentle depression for active functions that aren't producing results
            ltd_delta = self.ltd_rate * lr_mult * 0.5 * active
            new_affinity = jnp.clip(new_affinity - ltd_delta, 0.0, 1.0)

            ltd_indices = [i for i in range(NUM_ACTIVATIONS) if recent_mask[i] > 0.5]
            ltd_functions.extend(ltd_indices)

        update_info = {
            'ltp_functions': list(set(ltp_functions)),
            'ltd_functions': list(set(ltd_functions)),
            'ltp_applied': fitness_improved and len(palette_history) >= 2,
            'ltd_applied': not fitness_improved and len(palette_history) >= self.ltd_window,
        }

        return new_affinity, new_weights, update_info

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
        """Apply phase-appropriate mutation with STDP-learned protection."""
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
                # Inactive - might activate
                if use_protection and phase == CriticalPeriodPhase.CONFIRMATION:
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
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
        """Update with STDP learning."""
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

        # Update fitness EMA
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness

        # Compute improvement magnitude (normalized)
        improvement_magnitude = (best_fitness - prev_best_fitness) / max(0.1, prev_best_fitness)
        improvement_magnitude = max(0.0, min(1.0, improvement_magnitude))

        # Update palette history
        palette_history = state['palette_history'] + [(generation, state['mask'].copy(), best_fitness)]
        if len(palette_history) > self.history_length:
            palette_history = palette_history[-self.history_length:]

        # STDP update
        new_affinity, new_weights, stdp_info = self._stdp_update(
            state['function_affinity'],
            state['hebbian_weights'],
            palette_history,
            generation,
            improved,
            improvement_magnitude,
            phase,
        )

        # Compute protection scores
        protection_scores = self._compute_protection_scores(
            new_affinity, new_weights, state['mask']
        )

        # Apply mutation
        new_mask, mutation_info = self._mutate_palette(
            subkey, state['mask'], phase, protection_scores
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update fitness history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        # Track STDP events
        ltp_events = state['ltp_events'] + (1 if stdp_info['ltp_applied'] else 0)
        ltd_events = state['ltd_events'] + (1 if stdp_info['ltd_applied'] else 0)

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
            'palette_history': palette_history,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'ltp_events': ltp_events,
            'ltd_events': ltd_events,
            'sequence_discoveries': state['sequence_discoveries'],
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
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'sin_affinity': float(new_affinity[4]),
            'n_protected': len(protected_functions),
            'protected_functions': protected_functions,
            # STDP-specific stats
            'ltp_applied': stdp_info['ltp_applied'],
            'ltd_applied': stdp_info['ltd_applied'],
            'ltp_functions': stdp_info['ltp_functions'],
            'ltd_functions': stdp_info['ltd_functions'],
            'total_ltp_events': ltp_events,
            'total_ltd_events': ltd_events,
            'improvement_magnitude': improvement_magnitude,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with STDP stats."""
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
            'total_ltp_events': state['ltp_events'],
            'total_ltd_events': state['ltd_events'],
        }
