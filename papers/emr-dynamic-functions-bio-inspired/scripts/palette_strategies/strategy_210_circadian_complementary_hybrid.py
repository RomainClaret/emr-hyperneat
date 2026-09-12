"""Strategy 50: Circadian Rhythm + Complementary Learning Hybrid.

Combines the best features of:
1. Circadian Rhythm (strategy 34): Intrinsic oscillatory gating - temporal structure
2. Complementary Learning (strategy 23): Fast/slow memory systems - function tracking

Key Insight:
- Circadian gates WHEN functions are available (temporal cycling)
- Complementary Learning tracks WHICH functions work (memory systems)
- Together: temporal structure + learned function preferences

Mechanism:
1. Circadian clock advances each generation
2. Function activity depends on phase alignment AND affinity
3. Fast memory rapidly discovers useful functions
4. Slow memory retains proven functions
5. Consolidation transfers fast discoveries to slow memory
6. Palette selection combines circadian activity × effective affinity

Hybrid Activity Computation:
    circadian_activity = (1 - amplitude) + amplitude * (1 + cos(phase_diff)) / 2
    effective_affinity = slow_weight * slow + fast_weight * fast
    combined = circadian_activity * effective_affinity

This ensures:
- Functions peak at their circadian phase
- High-affinity functions are preferred at all times
- Both temporal diversity and learned preferences guide selection
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


class CircadianComplementaryHybridStrategy(PaletteEvolutionStrategy):
    """Hybrid combining circadian oscillations with dual memory systems.

    Circadian rhythm provides temporal gating structure.
    Complementary learning provides fast/slow memory for function tracking.
    Selection combines both: activity = circadian × affinity
    """

    name = "circadian_complementary_hybrid"
    description = "Circadian rhythm gating with fast/slow memory systems"

    def __init__(
        self,
        # Circadian parameters
        circadian_period: int = 25,              # Generations per full cycle
        initial_amplitude: float = 0.5,          # Default oscillation amplitude
        amplitude_min: float = 0.2,              # Minimum amplitude
        amplitude_max: float = 0.8,              # Maximum amplitude
        phase_learning_rate: float = 0.12,       # How fast phases adapt
        activity_threshold: float = 0.35,        # Loose circadian gate
        # Complementary learning parameters
        fast_learning_rate: float = 0.35,        # Rapid discovery
        fast_decay: float = 0.12,                # Moderate forgetting
        slow_learning_rate: float = 0.06,        # Gradual learning
        consolidation_interval: int = 12,        # 2x per circadian cycle
        consolidation_rate: float = 0.30,        # Aggressive transfer
        consolidation_threshold: float = 0.50,   # Lower threshold for transfer
        # Weighting
        fast_weight: float = 0.3,                # Fast system contribution
        slow_weight: float = 0.7,                # Slow system contribution
        circadian_weight: float = 0.5,           # How much circadian affects selection
        affinity_weight: float = 0.5,            # How much affinity affects selection
        # Palette
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize hybrid strategy.

        Args:
            circadian_period: Generations for one full circadian cycle
            initial_amplitude: Starting oscillation amplitude
            phase_learning_rate: How fast phases adapt to success
            activity_threshold: Minimum combined activity for palette
            fast_learning_rate: Learning rate for fast memory
            fast_decay: Decay rate for fast memory
            slow_learning_rate: Learning rate for slow memory
            consolidation_interval: Generations between consolidation
            consolidation_rate: Transfer rate fast → slow
            fast_weight: Weight of fast system in affinity
            slow_weight: Weight of slow system in affinity
            circadian_weight: Weight of circadian in combined activity
            affinity_weight: Weight of affinity in combined activity
            palette_size: Target palette size
        """
        # Circadian
        self.circadian_period = circadian_period
        self.initial_amplitude = initial_amplitude
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_learning_rate = phase_learning_rate
        self.activity_threshold = activity_threshold

        # Complementary learning
        self.fast_learning_rate = fast_learning_rate
        self.fast_decay = fast_decay
        self.slow_learning_rate = slow_learning_rate
        self.consolidation_interval = consolidation_interval
        self.consolidation_rate = consolidation_rate
        self.consolidation_threshold = consolidation_threshold

        # Weighting
        self.fast_weight = fast_weight
        self.slow_weight = slow_weight
        self.circadian_weight = circadian_weight
        self.affinity_weight = affinity_weight

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _initialize_phases(
        self,
        key: jax.random.PRNGKey,
        initial: List[int],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize function phases and amplitudes."""
        key1, key2 = jax.random.split(key)

        # Random initial phases
        phases = jax.random.uniform(key1, (NUM_ACTIVATIONS,)) * 2 * jnp.pi
        amplitudes = jnp.ones(NUM_ACTIVATIONS) * self.initial_amplitude

        # Initial palette functions start at phase 0 (peak)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                phases = phases.at[i].set(0.0)
                amplitudes = amplitudes.at[i].set(self.initial_amplitude * 0.8)

        # Small perturbations
        perturbations = jax.random.uniform(key2, (NUM_ACTIVATIONS,)) * 0.2 - 0.1
        phases = phases + perturbations

        return phases, amplitudes

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with circadian clock and dual memory systems."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        key = jax.random.PRNGKey(seed + 505050)
        key, subkey = jax.random.split(key)

        # Initialize circadian state
        phases, amplitudes = self._initialize_phases(subkey, initial)
        circadian_phase = 0.0

        # Initialize dual memory systems
        fast_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        slow_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Boost initial palette in slow memory
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                slow_affinity = slow_affinity.at[i].set(0.65)

        return {
            'mask': mask,
            'rng_key': key,
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Circadian state
            'circadian_phase': circadian_phase,
            'function_phases': phases,
            'function_amplitudes': amplitudes,
            # Dual memory
            'fast_affinity': fast_affinity,
            'slow_affinity': slow_affinity,
            # Consolidation
            'last_consolidation': 0,
            'consolidation_count': 0,
            'cycles_completed': 0,
            # Tracking
            'fitness_history': [],
            'fitness_ema': 0.5,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_circadian_activity(
        self,
        circadian_phase: float,
        function_phases: jnp.ndarray,
        amplitudes: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute circadian activity for each function."""
        activity = jnp.zeros(NUM_ACTIVATIONS)

        for i in range(NUM_ACTIVATIONS):
            phase_diff = circadian_phase - float(function_phases[i])
            amplitude = float(amplitudes[i])
            oscillation = (1 + jnp.cos(phase_diff)) / 2
            act = (1 - amplitude) + amplitude * oscillation
            activity = activity.at[i].set(act)

        return activity

    def _get_effective_affinity(
        self,
        fast_affinity: jnp.ndarray,
        slow_affinity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective affinity from dual systems."""
        return self.slow_weight * slow_affinity + self.fast_weight * fast_affinity

    def _compute_combined_activity(
        self,
        circadian_activity: jnp.ndarray,
        effective_affinity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Combine circadian gating with affinity scores.

        Combined = circadian_weight × circadian + affinity_weight × affinity
        Then normalize to [0, 1].
        """
        combined = (
            self.circadian_weight * circadian_activity +
            self.affinity_weight * effective_affinity
        )
        # Normalize
        combined = combined / (self.circadian_weight + self.affinity_weight)
        return combined

    def _update_fast_system(
        self,
        fast_affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update fast (hippocampal) memory system."""
        # Decay toward baseline
        new_fast = (1 - self.fast_decay) * fast_affinity + self.fast_decay * 0.5

        # Learn from current experience
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            delta = self.fast_learning_rate * fitness_signal * active
        else:
            delta = self.fast_learning_rate * 0.3 * fitness_signal * active

        new_fast = new_fast + delta
        return jnp.clip(new_fast, 0.05, 0.95)

    def _update_slow_system(
        self,
        slow_affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update slow (cortical) memory system."""
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            delta = self.slow_learning_rate * fitness_signal * active
        else:
            delta = self.slow_learning_rate * 0.2 * fitness_signal * active

        new_slow = slow_affinity + delta
        return jnp.clip(new_slow, 0.05, 0.95)

    def _consolidate(
        self,
        fast_affinity: jnp.ndarray,
        slow_affinity: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Perform consolidation from fast to slow system."""
        new_fast = fast_affinity.copy()
        new_slow = slow_affinity.copy()
        consolidated_indices = []

        active = (mask > 0.5)

        for i in range(NUM_ACTIVATIONS):
            fast_val = float(fast_affinity[i])

            if fast_val >= self.consolidation_threshold:
                # Active functions get boosted transfer
                if active[i] and fast_val > float(slow_affinity[i]):
                    transfer = self.consolidation_rate * fast_val * 1.3
                else:
                    transfer = self.consolidation_rate * fast_val

                new_slow = new_slow.at[i].set(
                    min(0.95, float(new_slow[i]) + transfer)
                )
                consolidated_indices.append(i)

        # Partial reset of fast system
        new_fast = new_fast * 0.5 + 0.5 * 0.5

        return new_fast, new_slow, {
            'consolidated_indices': consolidated_indices,
            'n_consolidated': len(consolidated_indices),
        }

    def _adapt_phases(
        self,
        phases: jnp.ndarray,
        mask: jnp.ndarray,
        effective_affinity: jnp.ndarray,
        circadian_phase: float,
        improvement: float,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Adapt phases based on fitness and affinity."""
        new_phases = phases.copy()

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (NUM_ACTIVATIONS,)) * 0.08

        if improvement > 0:
            # Successful functions with high affinity move toward current clock
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    affinity_boost = float(effective_affinity[i])
                    phase_diff = circadian_phase - float(phases[i])
                    adjustment = self.phase_learning_rate * improvement * affinity_boost * jnp.sin(phase_diff)
                    new_phases = new_phases.at[i].set(float(phases[i]) + adjustment)

        # Apply noise
        new_phases = new_phases + noise
        new_phases = jnp.mod(new_phases, 2 * jnp.pi)

        return new_phases

    def _select_palette(
        self,
        combined_activity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select palette based on combined activity scores."""
        above_threshold = combined_activity >= self.activity_threshold
        n_above = int(jnp.sum(above_threshold))

        if n_above >= self.min_active and n_above <= self.palette_size:
            mask = above_threshold.astype(jnp.float32)
        elif n_above < self.min_active:
            top_k = jnp.argsort(combined_activity)[-self.min_active:]
            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            top_k = jnp.argsort(combined_activity)[-self.palette_size:]
            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)

        return mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with both circadian and complementary learning dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Compute fitness signal for memory systems
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Step 1: Advance circadian clock
        phase_increment = 2 * jnp.pi / self.circadian_period
        new_circadian_phase = state['circadian_phase'] + phase_increment
        cycles_completed = state['cycles_completed']
        if new_circadian_phase >= 2 * jnp.pi:
            new_circadian_phase = new_circadian_phase % (2 * jnp.pi)
            cycles_completed += 1

        # Step 2: Update fast memory system
        new_fast = self._update_fast_system(
            state['fast_affinity'],
            state['mask'],
            fitness_signal,
        )

        # Step 3: Update slow memory system
        new_slow = self._update_slow_system(
            state['slow_affinity'],
            state['mask'],
            fitness_signal,
        )

        # Step 4: Check for consolidation
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

        # Step 5: Compute effective affinity
        effective_affinity = self._get_effective_affinity(new_fast, new_slow)

        # Step 6: Adapt circadian phases (now influenced by affinity)
        new_phases = self._adapt_phases(
            state['function_phases'],
            state['mask'],
            effective_affinity,
            new_circadian_phase,
            improvement,
            k1,
        )

        # Step 7: Compute circadian activity
        circadian_activity = self._compute_circadian_activity(
            new_circadian_phase,
            new_phases,
            state['function_amplitudes'],  # Amplitudes don't adapt in hybrid
        )

        # Step 8: Combine circadian activity with affinity
        combined_activity = self._compute_combined_activity(
            circadian_activity,
            effective_affinity,
        )

        # Step 9: Select palette based on combined activity
        new_mask = self._select_palette(combined_activity)

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
            # Circadian state
            'circadian_phase': float(new_circadian_phase),
            'function_phases': new_phases,
            'function_amplitudes': state['function_amplitudes'],
            # Dual memory
            'fast_affinity': new_fast,
            'slow_affinity': new_slow,
            # Consolidation
            'last_consolidation': last_consolidation,
            'consolidation_count': consolidation_count,
            'cycles_completed': cycles_completed,
            # Tracking
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)
        clock_position = float(new_circadian_phase) / (2 * jnp.pi)

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Clock
            'circadian_phase': float(new_circadian_phase),
            'clock_position': clock_position,
            'cycles_completed': cycles_completed,
            # Circadian activity
            'mean_circadian_activity': float(jnp.mean(circadian_activity)),
            # Affinity
            'avg_fast_affinity': float(jnp.mean(new_fast)),
            'avg_slow_affinity': float(jnp.mean(new_slow)),
            'avg_effective_affinity': float(jnp.mean(effective_affinity)),
            # Combined
            'mean_combined_activity': float(jnp.mean(combined_activity)),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_circadian': float(circadian_activity[4]),
            'sin_affinity': float(effective_affinity[4]),
            'sin_combined': float(combined_activity[4]),
            # Consolidation
            'did_consolidate': did_consolidate,
            'consolidation_count': consolidation_count,
        }
        metrics.update(consolidation_metrics)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with both circadian and memory stats."""
        palette = self.get_active_palette(state)
        fast = state['fast_affinity']
        slow = state['slow_affinity']
        effective = self._get_effective_affinity(fast, slow)

        circadian_activity = self._compute_circadian_activity(
            state['circadian_phase'],
            state['function_phases'],
            state['function_amplitudes'],
        )

        combined = self._compute_combined_activity(circadian_activity, effective)

        # Top functions by combined activity
        top_idx = jnp.argsort(combined)[-5:][::-1]
        top_combined = [(int(i), float(combined[i])) for i in top_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Clock
            'circadian_phase': float(state['circadian_phase']),
            'clock_position': float(state['circadian_phase']) / (2 * jnp.pi),
            'cycles_completed': state['cycles_completed'],
            # Top functions
            'top_combined': top_combined,
            # Sin
            'sin_combined': float(combined[4]),
            'sin_affinity': float(effective[4]),
            'sin_circadian': float(circadian_activity[4]),
            # Averages
            'avg_fast': float(jnp.mean(fast)),
            'avg_slow': float(jnp.mean(slow)),
            'mean_combined': float(jnp.mean(combined)),
            # Consolidation
            'consolidation_count': state['consolidation_count'],
        }
