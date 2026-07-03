"""Strategy 52D: Circadian Rhythm Dual (Oscillatory Gating for Both Domains).

Extends CircadianRhythmStrategy to jointly evolve BOTH activation AND aggregation
function palettes using intrinsic circadian oscillation dynamics.

Key dual mechanisms:
1. Dual phase tracking - separate phases for act and agg functions
2. Shared master clock - single circadian clock synchronizes both domains
3. Cross-domain entrainment - successful act-agg pairs influence each other's phases
4. Independent amplitudes - each domain adapts oscillation strength independently

Expected: Natural exploration/exploitation cycles in both domains
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


class CircadianRhythmDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with intrinsic circadian oscillations.

    Both activation and aggregation functions have individual phases synchronized
    to a master clock. Functions are active when phases align with the clock.
    """

    name = "circadian_rhythm_dual"
    description = "Dual: Circadian oscillations gating both domains"

    def __init__(
        self,
        circadian_period: int = 20,
        initial_amplitude: float = 0.6,
        amplitude_min: float = 0.2,
        amplitude_max: float = 0.9,
        amplitude_adaptation_rate: float = 0.05,
        phase_learning_rate: float = 0.15,
        phase_noise: float = 0.1,
        activity_threshold: float = 0.4,
        entrainment_strength: float = 0.3,
        cross_entrainment: float = 0.1,
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Circadian Rhythm Dual strategy."""
        self.circadian_period = circadian_period
        self.initial_amplitude = initial_amplitude
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.amplitude_adaptation_rate = amplitude_adaptation_rate
        self.phase_learning_rate = phase_learning_rate
        self.phase_noise = phase_noise
        self.activity_threshold = activity_threshold
        self.entrainment_strength = entrainment_strength
        self.cross_entrainment = cross_entrainment
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _initialize_phases(
        self,
        key: jax.random.PRNGKey,
        initial: List[int],
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize function phases and amplitudes."""
        key1, key2 = jax.random.split(key)
        phases = jax.random.uniform(key1, (n_funcs,)) * 2 * jnp.pi
        amplitudes = jnp.ones(n_funcs) * self.initial_amplitude

        for i in initial:
            if 0 <= i < n_funcs:
                phases = phases.at[i].set(0.0)
                amplitudes = amplitudes.at[i].set(self.initial_amplitude * 0.8)

        perturbations = jax.random.uniform(key2, (n_funcs,)) * 0.3 - 0.15
        phases = phases + perturbations

        return phases, amplitudes

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual circadian tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 525252)
        key, k_act, k_agg = jax.random.split(key, 3)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        act_phases, act_amplitudes = self._initialize_phases(k_act, initial_act, NUM_ACTIVATIONS)
        agg_phases, agg_amplitudes = self._initialize_phases(k_agg, initial_agg, NUM_AGGREGATIONS)

        circadian_phase = 0.0
        act_activity = self._compute_activity(circadian_phase, act_phases, act_amplitudes)
        agg_activity = self._compute_activity(circadian_phase, agg_phases, agg_amplitudes)

        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            'act_mask': act_mask,
            'act_phases': act_phases,
            'act_amplitudes': act_amplitudes,
            'act_activity': act_activity,
            'agg_mask': agg_mask,
            'agg_phases': agg_phases,
            'agg_amplitudes': agg_amplitudes,
            'agg_activity': agg_activity,
            'cross_affinity': cross_affinity,
            'circadian_phase': circadian_phase,
            'cycles_completed': 0,
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
        circadian_phase: float,
        function_phases: jnp.ndarray,
        amplitudes: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute current activity level for each function."""
        n_funcs = len(function_phases)
        activity = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            phase_diff = circadian_phase - float(function_phases[i])
            amplitude = float(amplitudes[i])
            oscillation = (1 + jnp.cos(phase_diff)) / 2
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
        n_funcs: int,
    ) -> jnp.ndarray:
        """Adapt function phases based on fitness feedback."""
        new_phases = phases.copy()
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (n_funcs,)) * self.phase_noise

        if improvement > 0:
            for i in range(n_funcs):
                if mask[i] > 0.5:
                    phase_diff = circadian_phase - float(phases[i])
                    adjustment = self.phase_learning_rate * improvement * jnp.sin(phase_diff)
                    new_phases = new_phases.at[i].set(float(phases[i]) + adjustment)
                else:
                    phase_diff = circadian_phase - float(phases[i])
                    adjustment = -self.entrainment_strength * 0.3 * jnp.sin(phase_diff)
                    new_phases = new_phases.at[i].set(float(phases[i]) + adjustment)
        else:
            noise = noise * 2

        new_phases = new_phases + noise
        return jnp.mod(new_phases, 2 * jnp.pi)

    def _adapt_amplitudes(
        self,
        amplitudes: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Adapt oscillation amplitudes based on fitness."""
        new_amplitudes = amplitudes.copy()

        if improvement > 0:
            for i in range(n_funcs):
                if mask[i] > 0.5:
                    delta = -self.amplitude_adaptation_rate * improvement
                    new_amplitudes = new_amplitudes.at[i].set(float(amplitudes[i]) + delta)
        else:
            for i in range(n_funcs):
                if mask[i] > 0.5:
                    delta = self.amplitude_adaptation_rate * 0.5
                    new_amplitudes = new_amplitudes.at[i].set(float(amplitudes[i]) + delta)

        return jnp.clip(new_amplitudes, self.amplitude_min, self.amplitude_max)

    def _select_palette(
        self,
        activity: jnp.ndarray,
        palette_size: int,
        min_active: int,
    ) -> jnp.ndarray:
        """Select palette based on current activity levels."""
        n_funcs = len(activity)
        above_threshold = activity >= self.activity_threshold
        n_above = int(jnp.sum(above_threshold))

        if n_above >= min_active and n_above <= palette_size:
            mask = above_threshold.astype(jnp.float32)
        elif n_above < min_active:
            top_k = jnp.argsort(activity)[-min_active:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            top_k = jnp.argsort(activity)[-palette_size:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)

        return mask

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        delta = self.cross_entrainment * improvement * co_active
        new_cross = cross_affinity + delta
        return jnp.clip(new_cross, 0.0, 1.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual circadian dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Advance master clock
        phase_increment = 2 * jnp.pi / self.circadian_period
        new_circadian_phase = state['circadian_phase'] + phase_increment
        cycles_completed = state['cycles_completed']
        if new_circadian_phase >= 2 * jnp.pi:
            new_circadian_phase = new_circadian_phase % (2 * jnp.pi)
            cycles_completed += 1

        # Update cross-affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            improvement,
        )

        # Adapt phases for both domains
        new_act_phases = self._adapt_phases(
            state['act_phases'], state['act_mask'],
            new_circadian_phase, improvement, k_act, NUM_ACTIVATIONS
        )
        new_agg_phases = self._adapt_phases(
            state['agg_phases'], state['agg_mask'],
            new_circadian_phase, improvement, k_agg, NUM_AGGREGATIONS
        )

        # Adapt amplitudes
        new_act_amplitudes = self._adapt_amplitudes(
            state['act_amplitudes'], state['act_mask'], improvement, NUM_ACTIVATIONS
        )
        new_agg_amplitudes = self._adapt_amplitudes(
            state['agg_amplitudes'], state['agg_mask'], improvement, NUM_AGGREGATIONS
        )

        # Compute new activity levels
        new_act_activity = self._compute_activity(new_circadian_phase, new_act_phases, new_act_amplitudes)
        new_agg_activity = self._compute_activity(new_circadian_phase, new_agg_phases, new_agg_amplitudes)

        # Select palettes
        new_act_mask = self._select_palette(new_act_activity, self.act_palette_size, self.min_active_act)
        new_agg_mask = self._select_palette(new_agg_activity, self.agg_palette_size, self.min_active_agg)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_phases': new_act_phases,
            'act_amplitudes': new_act_amplitudes,
            'act_activity': new_act_activity,
            'agg_mask': new_agg_mask,
            'agg_phases': new_agg_phases,
            'agg_amplitudes': new_agg_amplitudes,
            'agg_activity': new_agg_activity,
            'cross_affinity': new_cross,
            'circadian_phase': float(new_circadian_phase),
            'cycles_completed': cycles_completed,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)
        clock_position = float(new_circadian_phase) / (2 * jnp.pi)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'circadian_phase': float(new_circadian_phase),
            'clock_position': clock_position,
            'cycles_completed': cycles_completed,
            'act_mean_activity': float(jnp.mean(new_act_activity)),
            'agg_mean_activity': float(jnp.mean(new_agg_activity)),
            'act_mean_amplitude': float(jnp.mean(new_act_amplitudes)),
            'agg_mean_amplitude': float(jnp.mean(new_agg_amplitudes)),
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'has_sin': 4 in act_palette,
            'sin_activity': float(new_act_activity[4]),
            'sin_phase': float(new_act_phases[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual circadian status."""
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'circadian_phase': state['circadian_phase'],
            'clock_position': state['circadian_phase'] / (2 * jnp.pi),
            'cycles_completed': state['cycles_completed'],
            'generation': state['generation'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
