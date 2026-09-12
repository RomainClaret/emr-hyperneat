"""Strategy 34: Circadian Rhythm (Intrinsic Oscillatory Gating).

Implements circadian rhythm principles for palette evolution. Functions have
intrinsic oscillation phases, and a master circadian clock gates which
functions are available at any given generation.

Biological Basis:
- Circadian rhythms are ~24-hour cycles in physiology/behavior
- Master clock in suprachiasmatic nucleus synchronizes peripheral clocks
- Gene expression, hormone levels, metabolism follow circadian patterns
- Rhythms persist even without external cues (intrinsic)
- Phase can be adjusted by environmental signals (entrainment)

Key Insight:
- Current strategies are purely reactive (respond to fitness)
- Circadian rhythms add intrinsic periodic structure
- Functions become available at different phases
- Prevents premature convergence through forced diversity cycling
- Phase adaptation allows learning of optimal timing

Circadian Mechanism:
    # Advance master clock
    circadian_phase += 2π / period

    # Compute activity for each function based on phase alignment
    for each function i:
        phase_diff = circadian_phase - function_phases[i]
        activity[i] = (1 - amplitude[i]) + amplitude[i] * (1 + cos(phase_diff)) / 2

    # Select palette based on current activity
    active_mask = activity > threshold

    # Adapt phases toward successful timing
    if fitness_improved:
        # Move successful function phases toward current clock phase
        function_phases[active] += learning_rate * sin(circadian_phase - phases[active])

Expected improvements:
- Natural exploration/exploitation cycles
- Forced diversity through phase cycling
- Phase adaptation learns optimal timing
- Prevents premature convergence
- Periodic revisiting of all functions
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


class CircadianRhythmStrategy(PaletteEvolutionStrategy):
    """Intrinsic oscillatory gating of function availability.

    Functions have individual phases and amplitudes. A master circadian clock
    advances each generation, and functions are active when their phase
    aligns with the clock. Phases adapt based on fitness feedback.
    """

    name = "circadian_rhythm"
    description = "Intrinsic circadian oscillations gating function availability"

    def __init__(
        self,
        # Master clock
        circadian_period: int = 20,              # Generations per full cycle
        # Per-function oscillation
        initial_amplitude: float = 0.6,          # Default oscillation amplitude
        amplitude_min: float = 0.2,              # Minimum amplitude (always some rhythm)
        amplitude_max: float = 0.9,              # Maximum amplitude (strong rhythm)
        amplitude_adaptation_rate: float = 0.05, # How fast amplitude adapts
        # Phase dynamics
        phase_learning_rate: float = 0.15,       # How fast phases adapt
        phase_noise: float = 0.1,                # Random phase drift
        # Activity threshold
        activity_threshold: float = 0.4,         # Minimum activity for palette
        # Entrainment (external influence)
        entrainment_strength: float = 0.3,       # How much fitness entrains phase
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Circadian Rhythm strategy.

        Args:
            circadian_period: Generations for one full phase cycle
            initial_amplitude: Starting oscillation amplitude
            amplitude_min: Minimum amplitude (prevents complete flatness)
            amplitude_max: Maximum amplitude
            amplitude_adaptation_rate: How fast amplitudes adapt
            phase_learning_rate: How fast phases adapt to success
            phase_noise: Random phase drift per generation
            activity_threshold: Minimum activity for palette inclusion
            entrainment_strength: How strongly fitness affects phase
            palette_size: Target palette size
        """
        # Clock
        self.circadian_period = circadian_period

        # Oscillation
        self.initial_amplitude = initial_amplitude
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.amplitude_adaptation_rate = amplitude_adaptation_rate

        # Phase
        self.phase_learning_rate = phase_learning_rate
        self.phase_noise = phase_noise

        # Activity
        self.activity_threshold = activity_threshold

        # Entrainment
        self.entrainment_strength = entrainment_strength

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

        # Random initial phases distributed around the cycle
        phases = jax.random.uniform(key1, (NUM_ACTIVATIONS,)) * 2 * jnp.pi

        # Initial amplitudes
        amplitudes = jnp.ones(NUM_ACTIVATIONS) * self.initial_amplitude

        # Group initial palette functions to peak together (phase 0)
        # This gives them a head start
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                phases = phases.at[i].set(0.0)  # All start at peak
                amplitudes = amplitudes.at[i].set(self.initial_amplitude * 0.8)  # Slightly less rhythmic

        # Add small random perturbations to initial phases
        key2, subkey = jax.random.split(key2)
        perturbations = jax.random.uniform(subkey, (NUM_ACTIVATIONS,)) * 0.3 - 0.15
        phases = phases + perturbations

        return phases, amplitudes

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with circadian clock and function phases."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        key = jax.random.PRNGKey(seed + 343434)
        key, subkey = jax.random.split(key)

        # Initialize function phases and amplitudes
        phases, amplitudes = self._initialize_phases(subkey, initial)

        # Master clock starts at phase 0
        circadian_phase = 0.0

        # Initial activity levels
        activity = self._compute_activity(circadian_phase, phases, amplitudes)

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
            'activity': activity,
            # Tracking
            'phase_changes': jnp.zeros(NUM_ACTIVATIONS),
            'cycles_completed': 0,
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_activity(
        self,
        circadian_phase: float,
        function_phases: jnp.ndarray,
        amplitudes: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute current activity level for each function.

        Activity follows cosine function:
        activity = (1 - amplitude) + amplitude * (1 + cos(phase_diff)) / 2

        This gives:
        - At phase alignment (phase_diff=0): activity = 1
        - At anti-phase (phase_diff=π): activity = 1 - amplitude
        - With amplitude=0: activity = 1 (always on)
        - With amplitude=1: activity oscillates 0 to 1
        """
        activity = jnp.zeros(NUM_ACTIVATIONS)

        for i in range(NUM_ACTIVATIONS):
            phase_diff = circadian_phase - float(function_phases[i])
            amplitude = float(amplitudes[i])

            # Cosine oscillation: peaks when phase_diff = 0
            oscillation = (1 + jnp.cos(phase_diff)) / 2  # Range [0, 1]

            # Activity: baseline + amplitude-scaled oscillation
            act = (1 - amplitude) + amplitude * oscillation

            activity = activity.at[i].set(act)

        return activity

    def _adapt_phases(
        self,
        phases: jnp.ndarray,
        mask: jnp.ndarray,
        circadian_phase: float,
        improvement: float,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Adapt function phases based on fitness feedback.

        Successful functions move their phase toward current clock phase.
        This is "entrainment" - external signal (fitness) synchronizes phases.
        """
        new_phases = phases.copy()

        # Add random drift (intrinsic instability)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (NUM_ACTIVATIONS,)) * self.phase_noise

        if improvement > 0:
            # Successful: move active function phases toward current clock
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    phase_diff = circadian_phase - float(phases[i])
                    # Use sin for smooth attraction (gradient of cos)
                    adjustment = self.phase_learning_rate * improvement * jnp.sin(phase_diff)
                    new_phases = new_phases.at[i].set(float(phases[i]) + adjustment)

            # Entrainment: push inactive phases away slightly
            for i in range(NUM_ACTIVATIONS):
                if mask[i] <= 0.5:
                    phase_diff = circadian_phase - float(phases[i])
                    # Slight push away (anti-entrainment for diversity)
                    adjustment = -self.entrainment_strength * 0.3 * jnp.sin(phase_diff)
                    new_phases = new_phases.at[i].set(float(phases[i]) + adjustment)
        else:
            # Failure: add more noise to explore
            noise = noise * 2

        # Apply noise
        new_phases = new_phases + noise

        # Wrap phases to [0, 2π]
        new_phases = jnp.mod(new_phases, 2 * jnp.pi)

        return new_phases

    def _adapt_amplitudes(
        self,
        amplitudes: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Adapt oscillation amplitudes based on fitness.

        Successful functions reduce amplitude (become more available).
        Unsuccessful increase amplitude (become more rhythmic/selective).
        """
        new_amplitudes = amplitudes.copy()

        if improvement > 0:
            # Success: reduce amplitude for active functions (more available)
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    delta = -self.amplitude_adaptation_rate * improvement
                    new_amplitudes = new_amplitudes.at[i].set(
                        float(amplitudes[i]) + delta
                    )
        else:
            # Failure: increase amplitude for active functions (more selective)
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    delta = self.amplitude_adaptation_rate * 0.5
                    new_amplitudes = new_amplitudes.at[i].set(
                        float(amplitudes[i]) + delta
                    )

        return jnp.clip(new_amplitudes, self.amplitude_min, self.amplitude_max)

    def _select_palette(
        self,
        activity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select palette based on current activity levels."""
        above_threshold = activity >= self.activity_threshold
        n_above = int(jnp.sum(above_threshold))

        if n_above >= self.min_active and n_above <= self.palette_size:
            mask = above_threshold.astype(jnp.float32)
        elif n_above < self.min_active:
            # Too few: take top by activity
            top_k = jnp.argsort(activity)[-self.min_active:]
            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            # Too many: take top by activity
            top_k = jnp.argsort(activity)[-self.palette_size:]
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
        """Update with circadian dynamics."""
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

        # Step 1: Advance master clock
        phase_increment = 2 * jnp.pi / self.circadian_period
        new_circadian_phase = state['circadian_phase'] + phase_increment
        cycles_completed = state['cycles_completed']
        if new_circadian_phase >= 2 * jnp.pi:
            new_circadian_phase = new_circadian_phase % (2 * jnp.pi)
            cycles_completed += 1

        # Step 2: Adapt function phases based on fitness (entrainment)
        new_phases = self._adapt_phases(
            state['function_phases'],
            state['mask'],
            new_circadian_phase,
            improvement,
            k1,
        )

        # Step 3: Adapt amplitudes
        new_amplitudes = self._adapt_amplitudes(
            state['function_amplitudes'],
            state['mask'],
            improvement,
        )

        # Step 4: Compute new activity levels
        new_activity = self._compute_activity(
            new_circadian_phase,
            new_phases,
            new_amplitudes,
        )

        # Step 5: Select palette based on activity
        new_mask = self._select_palette(new_activity)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track phase changes
        phase_changes = jnp.abs(new_phases - state['function_phases'])

        # Track fitness
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
            'function_amplitudes': new_amplitudes,
            'activity': new_activity,
            # Tracking
            'phase_changes': phase_changes,
            'cycles_completed': cycles_completed,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Clock position as fraction of cycle
        clock_position = float(new_circadian_phase) / (2 * jnp.pi)

        # Top active functions
        top_activity_idx = jnp.argsort(new_activity)[-5:][::-1]
        top_activity = [(int(i), float(new_activity[i])) for i in top_activity_idx]

        # Functions near peak (phase aligned with clock)
        phase_alignment = jnp.cos(new_circadian_phase - new_phases)
        near_peak_idx = jnp.argsort(phase_alignment)[-3:][::-1]
        near_peak = [(int(i), float(phase_alignment[i])) for i in near_peak_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Clock
            'circadian_phase': float(new_circadian_phase),
            'clock_position': clock_position,  # 0-1 fraction
            'cycles_completed': cycles_completed,
            # Activity
            'mean_activity': float(jnp.mean(new_activity)),
            'max_activity': float(jnp.max(new_activity)),
            'min_activity': float(jnp.min(new_activity)),
            'top_activity': top_activity,
            # Phase alignment
            'near_peak_functions': near_peak,
            'mean_phase_change': float(jnp.mean(phase_changes)),
            # Amplitudes
            'mean_amplitude': float(jnp.mean(new_amplitudes)),
            'amplitude_range': (float(jnp.min(new_amplitudes)), float(jnp.max(new_amplitudes))),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_activity': float(new_activity[4]),
            'sin_phase': float(new_phases[4]),
            'sin_amplitude': float(new_amplitudes[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with circadian status."""
        palette = self.get_active_palette(state)
        phases = state['function_phases']
        amplitudes = state['function_amplitudes']
        activity = state['activity']
        clock = state['circadian_phase']

        # Top functions by activity
        top_act = jnp.argsort(activity)[-5:][::-1]
        top_activity = [(int(i), float(activity[i])) for i in top_act]

        # Phase alignment with clock
        alignment = jnp.cos(clock - phases)
        most_aligned = jnp.argsort(alignment)[-3:][::-1]
        aligned_funcs = [(int(i), float(alignment[i])) for i in most_aligned]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Clock
            'circadian_phase': float(clock),
            'clock_position': float(clock) / (2 * jnp.pi),
            'cycles_completed': state['cycles_completed'],
            'period': self.circadian_period,
            # Activity
            'top_activity': top_activity,
            'mean_activity': float(jnp.mean(activity)),
            # Phase
            'most_aligned': aligned_funcs,
            'mean_amplitude': float(jnp.mean(amplitudes)),
            # Sin-specific
            'sin_activity': float(activity[4]),
            'sin_phase': float(phases[4]),
        }
