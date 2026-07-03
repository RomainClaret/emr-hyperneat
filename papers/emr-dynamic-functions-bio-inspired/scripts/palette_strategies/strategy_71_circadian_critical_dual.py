"""Strategy 71: Circadian + Critical Period Dual ("Rhythmic Development").

Hybrid combining Circadian rhythm oscillations with Critical Period developmental windows.

Key synergies:
1. Master clock (period=20) provides rhythmic exploration/exploitation cycles
2. Critical period provides developmental trajectory (open → closed)
3. NOVEL: Critical period MODULATES oscillation amplitude
   - Open period: high amplitude oscillations (wide exploration)
   - Closed period: damped oscillations (stable exploitation)
4. Phase entrainment strength also varies with openness

Biological basis:
- Circadian rhythms are stronger during development
- Critical periods may modulate oscillatory dynamics
- Sleep (circadian) is crucial during developmental sensitive periods
- Combines temporal cycles with developmental trajectory

Expected: Strong rhythmic exploration early, damped stable cycles later
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
)


class CircadianCriticalDualStrategy(PaletteEvolutionStrategy):
    """Critical period modulates circadian oscillation amplitude.

    During open critical periods, oscillations are strong (high amplitude).
    As periods close, oscillations dampen to favor stability.
    """

    name = "circadian_critical_dual"
    description = "Critical period modulates circadian oscillation amplitude"

    def __init__(
        self,
        # Circadian parameters
        circadian_period: int = 20,
        base_amplitude: float = 0.8,  # Maximum amplitude when fully open
        damped_amplitude: float = 0.2,  # Minimum amplitude when closed
        # Critical period timing
        critical_period_end: int = 60,
        closure_rate: float = 0.95,  # Per 10 gens
        min_openness: float = 0.1,
        # Phase dynamics
        phase_learning_rate: float = 0.15,
        phase_noise_open: float = 0.12,  # More noise when open
        phase_noise_closed: float = 0.03,  # Less noise when closed
        entrainment_strength: float = 0.25,
        # Activity threshold
        activity_threshold_open: float = 0.35,  # Lower threshold when open
        activity_threshold_closed: float = 0.55,  # Higher threshold when closed
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        # Cross-domain
        cross_learning_rate: float = 0.08,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
        # Alias for compatibility with test scripts
        initial_palette: List[int] = None,
    ):
        # Handle initial_palette alias
        if initial_palette is not None and initial_act_palette is None:
            initial_act_palette = initial_palette

        # Circadian
        self.circadian_period = circadian_period
        self.base_amplitude = base_amplitude
        self.damped_amplitude = damped_amplitude

        # Critical period
        self.critical_period_end = critical_period_end
        self.closure_rate = closure_rate
        self.min_openness = min_openness

        # Phase dynamics
        self.phase_learning_rate = phase_learning_rate
        self.phase_noise_open = phase_noise_open
        self.phase_noise_closed = phase_noise_closed
        self.entrainment_strength = entrainment_strength

        # Activity thresholds
        self.activity_threshold_open = activity_threshold_open
        self.activity_threshold_closed = activity_threshold_closed

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _compute_openness(self, generation: int) -> float:
        """Compute critical period openness."""
        if generation >= self.critical_period_end:
            return self.min_openness

        progress = generation / self.critical_period_end
        openness = (self.closure_rate ** (progress * 10))
        return max(self.min_openness, openness)

    def _compute_effective_amplitude(self, openness: float) -> float:
        """Compute amplitude modulated by critical period."""
        # Interpolate between damped and base amplitude
        amplitude_range = self.base_amplitude - self.damped_amplitude
        return self.damped_amplitude + amplitude_range * openness

    def _compute_effective_threshold(self, openness: float) -> float:
        """Compute activity threshold modulated by openness."""
        # Lower threshold when open (more inclusive), higher when closed
        threshold_range = self.activity_threshold_closed - self.activity_threshold_open
        return self.activity_threshold_open + threshold_range * (1 - openness)

    def _initialize_phases(
        self,
        key: jax.random.PRNGKey,
        initial: List[int],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Initialize function phases."""
        phases = jax.random.uniform(key, (n_funcs,)) * 2 * jnp.pi

        # Initial functions start at phase 0 (in sync with clock)
        for i in initial:
            if 0 <= i < n_funcs:
                phases = phases.at[i].set(0.0)

        return phases

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize with Circadian + Critical Period state."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 717171)
        key, k1, k2 = jax.random.split(key, 3)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        act_phases = self._initialize_phases(k1, initial_act, NUM_ACTIVATIONS)
        agg_phases = self._initialize_phases(k2, initial_agg, NUM_AGGREGATIONS)

        # Initial openness and derived values
        openness = 1.0
        effective_amplitude = self._compute_effective_amplitude(openness)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_phases': act_phases,

            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_phases': agg_phases,

            # Cross-domain
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,

            # Clock and developmental state
            'clock_phase': 0.0,
            'cycles_completed': 0,
            'openness': openness,
            'effective_amplitude': effective_amplitude,

            # General state
            'rng_key': key,
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_activity(
        self,
        clock_phase: float,
        function_phases: jnp.ndarray,
        amplitude: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute activity with amplitude-modulated oscillations."""
        activity = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            phase_diff = clock_phase - float(function_phases[i])
            # Cosine oscillation centered at 0.5, scaled by amplitude
            oscillation = (1 + jnp.cos(phase_diff)) / 2
            # Activity = base (1-amplitude) + oscillating component
            act = (1 - amplitude) + amplitude * oscillation
            activity = activity.at[i].set(act)

        return activity

    def _adapt_phases(
        self,
        phases: jnp.ndarray,
        mask: jnp.ndarray,
        clock_phase: float,
        improvement: float,
        openness: float,
        key: jax.random.PRNGKey,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Adapt phases with openness-modulated noise."""
        new_phases = phases.copy()

        # Noise level depends on openness
        noise_level = (
            self.phase_noise_open * openness +
            self.phase_noise_closed * (1 - openness)
        )
        noise = jax.random.normal(key, (n_funcs,)) * noise_level

        # Entrainment strength also modulated
        effective_entrainment = self.entrainment_strength * (0.5 + 0.5 * openness)

        for i in range(n_funcs):
            phase_diff = clock_phase - float(phases[i])

            if mask[i] > 0.5:
                if improvement > 0:
                    # Active and improving: entrain to clock
                    adjustment = self.phase_learning_rate * improvement * jnp.sin(phase_diff)
                else:
                    # Active but not improving: weaker entrainment
                    adjustment = effective_entrainment * 0.3 * jnp.sin(phase_diff)
            else:
                # Inactive: random drift with entrainment
                adjustment = -effective_entrainment * 0.2 * jnp.sin(phase_diff)

            new_phases = new_phases.at[i].set(float(phases[i]) + adjustment)

        new_phases = new_phases + noise
        return jnp.mod(new_phases, 2 * jnp.pi)

    def _select_palette(
        self,
        activity: jnp.ndarray,
        threshold: float,
        palette_size: int,
        min_active: int,
    ) -> jnp.ndarray:
        """Select palette based on activity and threshold."""
        n_funcs = len(activity)
        above_threshold = activity >= threshold
        n_above = int(jnp.sum(above_threshold))

        if n_above >= min_active and n_above <= palette_size:
            # Just use those above threshold
            mask = above_threshold.astype(jnp.float32)
        elif n_above < min_active:
            # Not enough above threshold - take top min_active
            top_k = jnp.argsort(activity)[-min_active:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            # Too many above threshold - take top palette_size
            top_k = jnp.argsort(activity)[-palette_size:]
            mask = jnp.zeros(n_funcs)
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
        """Update with Circadian + Critical Period dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Advance master clock
        phase_increment = 2 * jnp.pi / self.circadian_period
        new_clock_phase = state['clock_phase'] + phase_increment
        cycles_completed = state['cycles_completed']
        if new_clock_phase >= 2 * jnp.pi:
            new_clock_phase = new_clock_phase % (2 * jnp.pi)
            cycles_completed += 1

        # Compute current openness and derived values
        openness = self._compute_openness(generation)
        effective_amplitude = self._compute_effective_amplitude(openness)
        effective_threshold = self._compute_effective_threshold(openness)

        # Update cross-domain affinity
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_delta = self.cross_learning_rate * improvement * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Adapt phases (openness-modulated)
        new_act_phases = self._adapt_phases(
            state['act_phases'], state['act_mask'],
            new_clock_phase, improvement, openness, k_act, NUM_ACTIVATIONS
        )
        new_agg_phases = self._adapt_phases(
            state['agg_phases'], state['agg_mask'],
            new_clock_phase, improvement, openness, k_agg, NUM_AGGREGATIONS
        )

        # Compute activity with amplitude-modulated oscillations
        act_activity = self._compute_activity(
            new_clock_phase, new_act_phases, effective_amplitude, NUM_ACTIVATIONS
        )
        agg_activity = self._compute_activity(
            new_clock_phase, new_agg_phases, effective_amplitude, NUM_AGGREGATIONS
        )

        # Select palettes with openness-modulated threshold
        new_act_mask = self._select_palette(
            act_activity, effective_threshold, self.act_palette_size, self.min_active_act
        )
        new_agg_mask = self._select_palette(
            agg_activity, effective_threshold, self.agg_palette_size, self.min_active_agg
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_phases': new_act_phases,
            'agg_mask': new_agg_mask,
            'agg_phases': new_agg_phases,
            'cross_affinity': new_cross,
            'clock_phase': float(new_clock_phase),
            'cycles_completed': cycles_completed,
            'openness': openness,
            'effective_amplitude': effective_amplitude,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)
        clock_position = float(new_clock_phase) / (2 * jnp.pi)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Clock stats
            'clock_phase': float(new_clock_phase),
            'clock_position': clock_position,
            'cycles_completed': cycles_completed,
            # Critical period stats
            'openness': openness,
            'effective_amplitude': effective_amplitude,
            'effective_threshold': effective_threshold,
            # Activity stats
            'act_mean_activity': float(jnp.mean(act_activity)),
            'agg_mean_activity': float(jnp.mean(agg_activity)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_activity': float(act_activity[4]),
            'sin_phase': float(new_act_phases[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with Circadian + Critical Period stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'clock_phase': state['clock_phase'],
            'clock_position': state['clock_phase'] / (2 * jnp.pi),
            'cycles_completed': state['cycles_completed'],
            'openness': state['openness'],
            'effective_amplitude': state['effective_amplitude'],
            'generation': state['generation'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
