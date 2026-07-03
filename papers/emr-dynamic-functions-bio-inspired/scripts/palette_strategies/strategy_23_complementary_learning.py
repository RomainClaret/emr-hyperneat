"""Strategy 23: Complementary Learning Systems (Hippocampus-Cortex).

Implements dual memory system inspired by hippocampus-cortex interaction.

Biological Basis:
- Hippocampus: Fast learning, rapid encoding, but prone to forgetting
- Neocortex: Slow learning, stable storage, resistant to interference
- Sleep consolidation: Replay transfers hippocampal traces to cortex

Key Insight:
- Strategy 19 (ConsolidationWindow) has periodic consolidation
- Complementary Learning makes dual systems EXPLICIT with continuous operation
- Fast system rapidly discovers new functions
- Slow system retains proven functions across generations

Learning Rules:
    Fast system: fast_affinity = (1 - decay) * fast + lr_fast * signal
    Slow system: slow_affinity += lr_slow * signal

    Consolidation (every N gens):
        slow_affinity[successful] += consolidation_rate * fast_affinity[successful]
        fast_affinity *= 0.5  (partial reset after consolidation)

    Effective affinity = 0.7 * slow + 0.3 * fast

This solves the stability-plasticity dilemma:
- Fast system provides plasticity (rapid discovery)
- Slow system provides stability (long-term retention)
- Consolidation bridges them (transfer of proven knowledge)
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


class ComplementaryLearningStrategy(PaletteEvolutionStrategy):
    """Hippocampal-cortical dual memory system.

    Fast system (hippocampus-like) learns quickly but forgets.
    Slow system (cortex-like) learns slowly but remembers.
    Consolidation periodically transfers from fast to slow.
    """

    name = "complementary_learning"
    description = "Hippocampus-cortex dual memory with consolidation"

    def __init__(
        self,
        # Fast system (hippocampus-like)
        fast_learning_rate: float = 0.35,      # Rapid acquisition
        fast_decay: float = 0.15,              # Forgetful
        fast_weight: float = 0.3,              # Contribution to effective affinity
        # Slow system (cortex-like)
        slow_learning_rate: float = 0.06,      # Gradual learning
        slow_decay: float = 0.0,               # No decay (stable)
        slow_weight: float = 0.7,              # Contribution to effective affinity
        # Consolidation
        consolidation_interval: int = 12,      # Every N gens
        consolidation_rate: float = 0.25,      # Transfer rate fast → slow
        consolidation_threshold: float = 0.55, # Min fast affinity to transfer
        replay_boost: float = 1.3,             # Strengthen successful during consolidation
        fast_reset_factor: float = 0.5,        # How much to reset fast after consolidation
        # Protection
        affinity_protection_threshold: float = 0.6,
        # Mutation rates
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Complementary Learning strategy.

        Args:
            fast_learning_rate: Learning rate for fast (hippocampal) system
            fast_decay: Decay rate for fast system
            fast_weight: Weight of fast system in effective affinity
            slow_learning_rate: Learning rate for slow (cortical) system
            slow_decay: Decay rate for slow system (typically 0)
            slow_weight: Weight of slow system in effective affinity
            consolidation_interval: Generations between consolidation
            consolidation_rate: How much fast transfers to slow
            consolidation_threshold: Minimum fast affinity to consolidate
            replay_boost: Boost factor during consolidation
            fast_reset_factor: How much to reset fast after consolidation
        """
        # Fast system
        self.fast_learning_rate = fast_learning_rate
        self.fast_decay = fast_decay
        self.fast_weight = fast_weight

        # Slow system
        self.slow_learning_rate = slow_learning_rate
        self.slow_decay = slow_decay
        self.slow_weight = slow_weight

        # Consolidation
        self.consolidation_interval = consolidation_interval
        self.consolidation_rate = consolidation_rate
        self.consolidation_threshold = consolidation_threshold
        self.replay_boost = replay_boost
        self.fast_reset_factor = fast_reset_factor

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual memory systems."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Dual memory systems
        fast_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5  # Hippocampus
        slow_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5  # Cortex

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 232323),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Dual memory
            'fast_affinity': fast_affinity,
            'slow_affinity': slow_affinity,
            # Consolidation tracking
            'last_consolidation': 0,
            'consolidation_count': 0,
            # Tracking
            'fitness_history': [],
            'fitness_ema': 0.5,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _get_effective_affinity(
        self,
        fast_affinity: jnp.ndarray,
        slow_affinity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective affinity from dual systems.

        Effective = slow_weight × slow + fast_weight × fast
        """
        return (
            self.slow_weight * slow_affinity +
            self.fast_weight * fast_affinity
        )

    def _update_fast_system(
        self,
        fast_affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update fast (hippocampal) system.

        Fast system learns quickly from current experience and decays.
        """
        # Decay toward baseline
        new_fast = (1 - self.fast_decay) * fast_affinity + self.fast_decay * 0.5

        # Learn from current experience
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            # Positive: boost active functions
            delta = self.fast_learning_rate * fitness_signal * active
        else:
            # Negative: modest reduction for active functions
            delta = self.fast_learning_rate * 0.3 * fitness_signal * active

        new_fast = new_fast + delta

        return jnp.clip(new_fast, 0.05, 0.95)

    def _update_slow_system(
        self,
        slow_affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update slow (cortical) system.

        Slow system learns gradually and doesn't decay.
        """
        active = (mask > 0.5).astype(jnp.float32)

        # Slow, gradual learning
        if fitness_signal >= 0:
            delta = self.slow_learning_rate * fitness_signal * active
        else:
            # Very slow negative learning
            delta = self.slow_learning_rate * 0.2 * fitness_signal * active

        new_slow = slow_affinity + delta

        # Apply minimal decay if configured
        if self.slow_decay > 0:
            new_slow = (1 - self.slow_decay) * new_slow + self.slow_decay * 0.5

        return jnp.clip(new_slow, 0.05, 0.95)

    def _consolidate(
        self,
        fast_affinity: jnp.ndarray,
        slow_affinity: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Perform consolidation from fast to slow system.

        Transfer high-affinity fast traces to slow system.
        Apply replay boost to successful functions.
        Partial reset of fast system after transfer.
        """
        new_fast = fast_affinity.copy()
        new_slow = slow_affinity.copy()
        consolidated_indices = []

        active = (mask > 0.5)

        for i in range(NUM_ACTIVATIONS):
            fast_val = float(fast_affinity[i])

            # Only consolidate if above threshold
            if fast_val >= self.consolidation_threshold:
                # Transfer with replay boost for active successful functions
                if active[i] and fast_val > float(slow_affinity[i]):
                    transfer = self.consolidation_rate * fast_val * self.replay_boost
                else:
                    transfer = self.consolidation_rate * fast_val

                new_slow = new_slow.at[i].set(
                    min(0.95, float(new_slow[i]) + transfer)
                )
                consolidated_indices.append(i)

        # Partial reset of fast system (simulates "clearing" hippocampus)
        new_fast = new_fast * self.fast_reset_factor + 0.5 * (1 - self.fast_reset_factor)

        consolidation_metrics = {
            'consolidated_indices': consolidated_indices,
            'n_consolidated': len(consolidated_indices),
            'avg_transfer': float(jnp.mean(new_slow) - jnp.mean(slow_affinity)),
        }

        return new_fast, new_slow, consolidation_metrics

    def _compute_protection(
        self,
        effective_affinity: jnp.ndarray,
        slow_affinity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection from effective and slow affinity.

        Slow affinity provides stable protection.
        """
        # Primary protection from slow (stable) system
        protection = 0.7 * slow_affinity + 0.3 * effective_affinity

        return jnp.clip(protection, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        effective_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with affinity-guided rates."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            prot = float(protection[i])
            aff = float(effective_affinity[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                # Higher affinity = more likely to activate
                rate = self.base_activate_rate * (0.5 + 0.5 * aff)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                if prot >= self.affinity_protection_threshold:
                    # Protected by slow system: very low deactivation
                    rate = self.base_deactivate_rate * 0.1
                else:
                    # Not protected: higher deactivation
                    rate = self.base_deactivate_rate * (1.0 - prot)

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
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with complementary learning systems."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Compute fitness signal
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Step 1: Update fast system (hippocampus)
        new_fast = self._update_fast_system(
            state['fast_affinity'],
            state['mask'],
            fitness_signal,
        )

        # Step 2: Update slow system (cortex)
        new_slow = self._update_slow_system(
            state['slow_affinity'],
            state['mask'],
            fitness_signal,
        )

        # Step 3: Check for consolidation
        consolidation_metrics = {}
        gens_since_consolidation = generation - state['last_consolidation']
        did_consolidate = False

        if gens_since_consolidation >= self.consolidation_interval:
            new_fast, new_slow, consolidation_metrics = self._consolidate(
                new_fast, new_slow, state['mask']
            )
            last_consolidation = generation
            consolidation_count = state['consolidation_count'] + 1
            did_consolidate = True
        else:
            last_consolidation = state['last_consolidation']
            consolidation_count = state['consolidation_count']

        # Step 4: Compute effective affinity
        effective_affinity = self._get_effective_affinity(new_fast, new_slow)

        # Step 5: Compute protection (primarily from slow system)
        protection = self._compute_protection(effective_affinity, new_slow)

        # Step 6: Apply mutation
        new_mask, mutation_info = self._mutate_palette(
            subkey,
            state['mask'],
            protection,
            effective_affinity,
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
            # Dual memory
            'fast_affinity': new_fast,
            'slow_affinity': new_slow,
            # Consolidation tracking
            'last_consolidation': last_consolidation,
            'consolidation_count': consolidation_count,
            # Tracking
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)
        protected = [
            i for i in active_palette
            if protection[i] >= self.affinity_protection_threshold
        ]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Dual system stats
            'avg_fast_affinity': float(jnp.mean(new_fast)),
            'avg_slow_affinity': float(jnp.mean(new_slow)),
            'avg_effective_affinity': float(jnp.mean(effective_affinity)),
            'sin_fast_affinity': float(new_fast[4]),
            'sin_slow_affinity': float(new_slow[4]),
            'sin_effective_affinity': float(effective_affinity[4]),
            # Consolidation
            'did_consolidate': did_consolidate,
            'consolidation_count': consolidation_count,
            'gens_since_consolidation': gens_since_consolidation,
            # Protection
            'n_protected': len(protected),
            'protected_functions': protected,
        }
        metrics.update(mutation_info)
        metrics.update(consolidation_metrics)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual memory stats."""
        palette = self.get_active_palette(state)
        fast = state['fast_affinity']
        slow = state['slow_affinity']
        effective = self._get_effective_affinity(fast, slow)

        # Top functions by effective affinity
        top_idx = jnp.argsort(effective)[-5:][::-1]
        top_effective = [(int(i), float(effective[i])) for i in top_idx]

        # Top by slow (consolidated) affinity
        top_slow_idx = jnp.argsort(slow)[-5:][::-1]
        top_slow = [(int(i), float(slow[i])) for i in top_slow_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Affinity stats
            'top_effective_functions': top_effective,
            'top_slow_functions': top_slow,
            'sin_effective': float(effective[4]),
            'sin_slow': float(slow[4]),
            'sin_fast': float(fast[4]),
            # Averages
            'avg_fast': float(jnp.mean(fast)),
            'avg_slow': float(jnp.mean(slow)),
            'avg_effective': float(jnp.mean(effective)),
            # Consolidation
            'consolidation_count': state['consolidation_count'],
            'gens_since_consolidation': state['generation'] - state['last_consolidation'],
        }
