"""Strategy 19: Consolidation Window (Memory Consolidation).

Implements memory consolidation with periodic "sleep" phases where recent
important discoveries are strengthened without new learning interference.

Biological Basis (Sleep/Rest Consolidation):
- During wake: Active learning, exploration, memory encoding
- During sleep: Replay of important memories, consolidation
- Memory traces get strengthened during consolidation
- New learning is suppressed to prevent interference

For palette evolution:
- Active phases: Normal mutation, exploration, learning
- Consolidation phases: Replay high-affinity functions, suppress new mutations
- Long-term memory: Separate affinity store for consolidated knowledge
- Transfer: Gradually move working memory → long-term memory

Key mechanisms:
1. Periodic consolidation windows (every N generations)
2. During consolidation: Strengthen high-affinity, suppress mutation
3. Long-term memory: More stable affinity storage
4. Replay: High-affinity functions get additional boost during consolidation

Expected improvement over continuous learning:
- More stable retention of good discoveries
- Prevents oscillation from constant mutation
- Better protection of important functions
- Emergent memory hierarchy (short-term vs long-term)
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


class ConsolidationPhase:
    """Consolidation state phases."""
    ACTIVE = "active"              # Normal learning
    CONSOLIDATING = "consolidating"  # Sleep/replay phase


class ConsolidationWindowStrategy(PaletteEvolutionStrategy):
    """Memory consolidation with periodic consolidation windows.

    Alternates between active learning and consolidation phases.
    """

    name = "consolidation_window"
    description = "Memory consolidation with periodic consolidation windows"

    def __init__(
        self,
        # Consolidation timing
        consolidation_frequency: int = 10,      # Every N gens, consolidate
        consolidation_duration: int = 3,        # Consolidation window length
        # Consolidation parameters
        replay_strength: float = 1.5,           # Boost for high-affinity during consolidation
        replay_threshold: float = 0.6,          # Affinity threshold for replay
        transfer_rate: float = 0.1,             # Working → long-term transfer rate
        ltm_decay_rate: float = 0.02,           # Slow decay in long-term memory
        # Active phase parameters
        active_learning_rate: float = 0.15,     # Learning rate during active phase
        active_mutation_rate: float = 0.20,     # Mutation rate during active phase
        # Consolidation phase parameters
        consolidation_mutation_rate: float = 0.02,  # Minimal mutation during consolidation
        consolidation_learning_rate: float = 0.05,  # Slower learning during consolidation
        # Protection
        affinity_protection_threshold: float = 0.55,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Consolidation Window strategy.

        Args:
            consolidation_frequency: Generations between consolidation windows
            consolidation_duration: Length of consolidation window
            replay_strength: How much to boost high-affinity functions during consolidation
            replay_threshold: Minimum affinity to be replayed
            transfer_rate: Speed of transfer to long-term memory
            ltm_decay_rate: Slow decay rate for long-term memory
        """
        # Consolidation timing
        self.consolidation_frequency = consolidation_frequency
        self.consolidation_duration = consolidation_duration

        # Consolidation parameters
        self.replay_strength = replay_strength
        self.replay_threshold = replay_threshold
        self.transfer_rate = transfer_rate
        self.ltm_decay_rate = ltm_decay_rate

        # Active phase parameters
        self.active_learning_rate = active_learning_rate
        self.active_mutation_rate = active_mutation_rate

        # Consolidation phase parameters
        self.consolidation_mutation_rate = consolidation_mutation_rate
        self.consolidation_learning_rate = consolidation_learning_rate

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _get_consolidation_phase(
        self,
        generation: int,
        last_consolidation: int,
    ) -> Tuple[str, bool]:
        """Determine current phase and if we're starting consolidation.

        Returns:
            (phase, starting_new_consolidation)
        """
        gens_since_consolidation = generation - last_consolidation

        if gens_since_consolidation < self.consolidation_duration:
            # Currently in consolidation window
            return ConsolidationPhase.CONSOLIDATING, False
        elif gens_since_consolidation >= self.consolidation_frequency:
            # Time to start new consolidation
            return ConsolidationPhase.CONSOLIDATING, True
        else:
            # Active learning phase
            return ConsolidationPhase.ACTIVE, False

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with working and long-term memory."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Working memory affinity (changes frequently)
        working_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Long-term memory affinity (consolidated, stable)
        ltm_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Co-occurrence tracking for Hebbian learning
        co_occurrence = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS))

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 191919),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Memory systems
            'working_affinity': working_affinity,
            'ltm_affinity': ltm_affinity,
            'co_occurrence': co_occurrence,
            # Consolidation state
            'consolidation_phase': ConsolidationPhase.ACTIVE,
            'last_consolidation_gen': -self.consolidation_frequency,  # Start fresh
            'consolidations_completed': 0,
            # Tracking
            'fitness_history': [],
            'fitness_ema': 0.5,
            'replay_events': 0,
            'transfer_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_effective_affinity(
        self,
        working: jnp.ndarray,
        ltm: jnp.ndarray,
        phase: str,
    ) -> jnp.ndarray:
        """Compute effective affinity combining working and long-term memory.

        During consolidation, long-term memory dominates.
        During active, working memory has more influence.
        """
        if phase == ConsolidationPhase.CONSOLIDATING:
            # Long-term memory dominates during consolidation
            return 0.3 * working + 0.7 * ltm
        else:
            # Working memory dominates during active learning
            return 0.6 * working + 0.4 * ltm

    def _update_working_memory(
        self,
        working: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> jnp.ndarray:
        """Update working memory based on fitness signal."""
        if phase == ConsolidationPhase.CONSOLIDATING:
            lr = self.consolidation_learning_rate
        else:
            lr = self.active_learning_rate

        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            delta = lr * fitness_signal * active
        else:
            # Slower unlearning
            delta = lr * 0.3 * fitness_signal * active

        return jnp.clip(working + delta, 0.0, 1.0)

    def _perform_consolidation(
        self,
        working: jnp.ndarray,
        ltm: jnp.ndarray,
        mask: jnp.ndarray,
        co_occurrence: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Perform memory consolidation.

        1. Replay high-affinity functions (boost them)
        2. Transfer working → long-term memory
        3. Slight decay of unused long-term memory

        Returns:
            (new_working, new_ltm, n_replayed, n_transferred)
        """
        new_working = working.copy()
        new_ltm = ltm.copy()
        n_replayed = 0
        n_transferred = 0

        active = (mask > 0.5).astype(jnp.float32)

        # Step 1: Replay - boost high-affinity functions
        for i in range(NUM_ACTIVATIONS):
            if float(working[i]) >= self.replay_threshold:
                # Replay this function
                boost = self.replay_strength * (float(working[i]) - self.replay_threshold)
                new_working = new_working.at[i].set(
                    min(0.95, float(new_working[i]) + boost)
                )
                n_replayed += 1

        # Step 2: Transfer from working to long-term memory
        # Functions with high working affinity AND active → transfer
        for i in range(NUM_ACTIVATIONS):
            if float(working[i]) >= self.replay_threshold and float(active[i]) > 0.5:
                # Transfer toward working memory value
                diff = float(working[i]) - float(ltm[i])
                transfer = self.transfer_rate * diff
                new_ltm = new_ltm.at[i].set(
                    min(0.95, float(new_ltm[i]) + transfer)
                )
                if transfer > 0.01:
                    n_transferred += 1

        # Step 3: Slow decay of long-term memory for inactive functions
        for i in range(NUM_ACTIVATIONS):
            if float(active[i]) < 0.5:
                # Decay toward baseline
                decay = self.ltm_decay_rate * (float(new_ltm[i]) - 0.5)
                new_ltm = new_ltm.at[i].set(
                    max(0.05, float(new_ltm[i]) - decay)
                )

        return new_working, new_ltm, n_replayed, n_transferred

    def _update_co_occurrence(
        self,
        co_occurrence: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update co-occurrence matrix for Hebbian learning."""
        if fitness_signal <= 0:
            return co_occurrence

        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        # Strengthen co-occurrence for functions active during success
        delta = 0.1 * fitness_signal * co_active
        return jnp.clip(co_occurrence + delta, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        effective_affinity: jnp.ndarray,
        phase: str,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with phase-dependent rates."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        protection_info = {}

        if phase == ConsolidationPhase.CONSOLIDATING:
            mutation_rate = self.consolidation_mutation_rate
        else:
            mutation_rate = self.active_mutation_rate

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            affinity = float(effective_affinity[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                # Higher affinity → higher activation probability
                effective_rate = mutation_rate * (0.5 + affinity)
                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                if affinity >= self.affinity_protection_threshold:
                    # Protected by high affinity
                    deact_rate = mutation_rate * 0.1  # Very low
                    protection_info[i] = f"protected (affinity={affinity:.2f})"
                else:
                    # Vulnerable
                    deact_rate = mutation_rate * (1.0 - affinity)

                # Much lower deactivation during consolidation
                if phase == ConsolidationPhase.CONSOLIDATING:
                    deact_rate *= 0.2

                if deactivate_probs[i] < deact_rate:
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
        """Update with consolidation window mechanism."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Determine consolidation phase
        phase, starting_consolidation = self._get_consolidation_phase(
            generation,
            state['last_consolidation_gen'],
        )

        # Update last consolidation if starting new one
        last_consolidation_gen = state['last_consolidation_gen']
        consolidations_completed = state['consolidations_completed']
        if starting_consolidation:
            last_consolidation_gen = generation
            consolidations_completed += 1

        # Compute fitness signal
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Update working memory
        new_working = self._update_working_memory(
            state['working_affinity'],
            state['mask'],
            fitness_signal,
            phase,
        )

        # Update co-occurrence
        new_co_occurrence = self._update_co_occurrence(
            state['co_occurrence'],
            state['mask'],
            fitness_signal,
        )

        # During consolidation, perform memory consolidation
        new_ltm = state['ltm_affinity']
        n_replayed = 0
        n_transferred = 0

        if phase == ConsolidationPhase.CONSOLIDATING:
            new_working, new_ltm, n_replayed, n_transferred = self._perform_consolidation(
                new_working,
                state['ltm_affinity'],
                state['mask'],
                new_co_occurrence,
            )

        # Compute effective affinity
        effective_affinity = self._compute_effective_affinity(
            new_working, new_ltm, phase
        )

        # Apply mutation
        new_mask, mutation_info = self._mutate_palette(
            subkey, state['mask'], effective_affinity, phase
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update fitness history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        replay_events = state['replay_events'] + n_replayed
        transfer_events = state['transfer_events'] + n_transferred

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Memory systems
            'working_affinity': new_working,
            'ltm_affinity': new_ltm,
            'co_occurrence': new_co_occurrence,
            # Consolidation state
            'consolidation_phase': phase,
            'last_consolidation_gen': last_consolidation_gen,
            'consolidations_completed': consolidations_completed,
            # Tracking
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'replay_events': replay_events,
            'transfer_events': transfer_events,
        }

        active_palette = mask_to_indices(new_mask)
        protected = [
            i for i in active_palette
            if effective_affinity[i] >= self.affinity_protection_threshold
        ]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Phase info
            'consolidation_phase': phase,
            'starting_consolidation': starting_consolidation,
            'consolidations_completed': consolidations_completed,
            # Affinity stats
            'avg_working_affinity': float(jnp.mean(new_working)),
            'avg_ltm_affinity': float(jnp.mean(new_ltm)),
            'avg_effective_affinity': float(jnp.mean(effective_affinity)),
            'sin_working_affinity': float(new_working[4]),
            'sin_ltm_affinity': float(new_ltm[4]),
            # Consolidation stats
            'n_replayed': n_replayed,
            'n_transferred': n_transferred,
            'total_replay_events': replay_events,
            'total_transfer_events': transfer_events,
            # Protection
            'n_protected': len(protected),
            'protected_functions': protected,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with memory consolidation stats."""
        palette = self.get_active_palette(state)
        working = state['working_affinity']
        ltm = state['ltm_affinity']

        # Top functions by LTM affinity (most consolidated)
        top_indices = jnp.argsort(ltm)[-5:][::-1]
        top_ltm = [(int(i), float(ltm[i])) for i in top_indices]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'consolidation_phase': state['consolidation_phase'],
            'consolidations_completed': state['consolidations_completed'],
            'generation': state['generation'],
            'top_ltm_functions': top_ltm,
            'sin_working_affinity': float(working[4]),
            'sin_ltm_affinity': float(ltm[4]),
            'avg_working_affinity': float(jnp.mean(working)),
            'avg_ltm_affinity': float(jnp.mean(ltm)),
            'stagnation_count': state['stagnation_count'],
            'total_replay_events': state['replay_events'],
            'total_transfer_events': state['transfer_events'],
        }
