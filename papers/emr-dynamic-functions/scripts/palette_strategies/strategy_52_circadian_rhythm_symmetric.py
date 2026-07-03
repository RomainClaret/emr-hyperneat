"""Strategy 52S: Circadian Rhythm Symmetric (Protected Oscillatory Gating).

Extends CircadianRhythmDualStrategy to symmetric mode with winning patterns:
- Protected indices get activity floor regardless of phase
- Affinity floors for sin and extreme aggregations
- Memory cell crystallization for sustained performance
- Cross-domain affinity matrix for coordinated discovery

Key symmetric mechanisms:
1. Protected activity floors - sin/max/min always have minimum activity
2. Phase-independent protection - critical functions never deactivated by phase
3. Affinity floor enforcement - applied every generation
4. Memory cell formation - 8 gens sustained → permanent protection

Circadian Insight: Protected indices act like "zeitgebers" (time-givers) that
anchor the circadian system, providing stable reference points while other
functions oscillate.
"""

from dataclasses import dataclass, field
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
)

# Critical indices for protection
SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3


@dataclass
class CircadianRhythmSymmetricState:
    """State for circadian rhythm symmetric strategy."""
    # Palette masks
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    # Phase and amplitude tracking
    act_phases: jnp.ndarray
    agg_phases: jnp.ndarray
    act_amplitudes: jnp.ndarray
    agg_amplitudes: jnp.ndarray
    act_activity: jnp.ndarray
    agg_activity: jnp.ndarray
    # Affinities
    act_affinity: jnp.ndarray
    agg_affinity: jnp.ndarray
    cross_affinity: jnp.ndarray
    # Memory cells
    act_memory_cells: jnp.ndarray
    agg_memory_cells: jnp.ndarray
    act_memory_counts: jnp.ndarray
    agg_memory_counts: jnp.ndarray
    # Circadian clock
    circadian_phase: float
    cycles_completed: int
    # General state
    rng_key: jnp.ndarray
    generation: int
    stagnation_count: int
    best_fitness_seen: float
    fitness_history: List[float] = field(default_factory=list)


class CircadianRhythmSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with circadian oscillations and protection.

    Applies winning patterns:
    - Protected indices (0.1% deactivation for sin/max/min)
    - Affinity floors (sin=0.6, extreme_agg=0.5)
    - Memory cell crystallization (8 gens sustained)
    - Activity floors for protected indices regardless of phase
    """

    name = "circadian_rhythm_symmetric"
    description = "Symmetric: Circadian oscillations with protected zeitgebers"

    def __init__(
        self,
        # Circadian parameters
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
        # Protection parameters (winning patterns)
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_activity_floor: float = 0.7,
        protected_deactivation_prob: float = 0.001,  # 0.1%
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 4,
        min_active_act: int = 3,
        min_active_agg: int = 2,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Circadian Rhythm Symmetric strategy."""
        # Circadian parameters
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

        # Protection parameters
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_activity_floor = protected_activity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Palette composition
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        # CRITICAL: Include sin and extreme aggregations in initial palettes
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]

    def _initialize_phases(
        self,
        key: jax.random.PRNGKey,
        initial: List[int],
        n_funcs: int,
        protected_indices: List[int],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize function phases and amplitudes.

        Protected indices start in-phase (phase=0) for immediate activity.
        """
        key1, key2 = jax.random.split(key)
        phases = jax.random.uniform(key1, (n_funcs,)) * 2 * jnp.pi
        amplitudes = jnp.ones(n_funcs) * self.initial_amplitude

        # Initial palette functions start in-phase
        for i in initial:
            if 0 <= i < n_funcs:
                phases = phases.at[i].set(0.0)
                amplitudes = amplitudes.at[i].set(self.initial_amplitude * 0.8)

        # Protected indices also start in-phase with lower amplitude (more stable)
        for i in protected_indices:
            if 0 <= i < n_funcs:
                phases = phases.at[i].set(0.0)
                amplitudes = amplitudes.at[i].set(self.amplitude_min)  # Low amplitude = stable

        # Small perturbations
        perturbations = jax.random.uniform(key2, (n_funcs,)) * 0.3 - 0.15
        phases = phases + perturbations

        return phases, amplitudes

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for protected indices."""
        # Sin gets affinity floor
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )

        # Max and min get affinity floors
        new_agg = agg_affinity
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )

        return new_act, new_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> CircadianRhythmSymmetricState:
        """Initialize state with circadian tracking and protection."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 525200)
        key, k_act, k_agg = jax.random.split(key, 3)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize phases with protection
        act_phases, act_amplitudes = self._initialize_phases(
            k_act, initial_act, NUM_ACTIVATIONS, [SIN_IDX]
        )
        agg_phases, agg_amplitudes = self._initialize_phases(
            k_agg, initial_agg, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX]
        )

        # Initialize activity
        circadian_phase = 0.0
        act_activity = self._compute_activity(
            circadian_phase, act_phases, act_amplitudes, [SIN_IDX]
        )
        agg_activity = self._compute_activity(
            circadian_phase, agg_phases, agg_amplitudes, [MAX_IDX, MIN_IDX]
        )

        # Initialize affinities with floors
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        act_affinity, agg_affinity = self._apply_affinity_floors(act_affinity, agg_affinity)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        # Boost sin-extreme pairs
        for agg_idx in [MAX_IDX, MIN_IDX]:
            cross_affinity = cross_affinity.at[SIN_IDX, agg_idx].set(0.7)

        # Memory cells
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS)

        return CircadianRhythmSymmetricState(
            act_mask=act_mask,
            agg_mask=agg_mask,
            act_phases=act_phases,
            agg_phases=agg_phases,
            act_amplitudes=act_amplitudes,
            agg_amplitudes=agg_amplitudes,
            act_activity=act_activity,
            agg_activity=agg_activity,
            act_affinity=act_affinity,
            agg_affinity=agg_affinity,
            cross_affinity=cross_affinity,
            act_memory_cells=act_memory_cells,
            agg_memory_cells=agg_memory_cells,
            act_memory_counts=act_memory_counts,
            agg_memory_counts=agg_memory_counts,
            circadian_phase=circadian_phase,
            cycles_completed=0,
            rng_key=key,
            generation=0,
            stagnation_count=0,
            best_fitness_seen=0.0,
            fitness_history=[],
        )

    def get_active_palette(self, state: CircadianRhythmSymmetricState) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state: CircadianRhythmSymmetricState) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _compute_activity(
        self,
        circadian_phase: float,
        function_phases: jnp.ndarray,
        amplitudes: jnp.ndarray,
        protected_indices: List[int],
    ) -> jnp.ndarray:
        """Compute current activity level for each function.

        Protected indices get activity floor regardless of phase.
        """
        n_funcs = len(function_phases)
        activity = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            phase_diff = circadian_phase - float(function_phases[i])
            amplitude = float(amplitudes[i])
            oscillation = (1 + jnp.cos(phase_diff)) / 2
            act = (1 - amplitude) + amplitude * oscillation
            activity = activity.at[i].set(act)

        # Protected indices get activity floor
        for idx in protected_indices:
            if 0 <= idx < n_funcs:
                activity = activity.at[idx].set(
                    jnp.maximum(activity[idx], self.protected_activity_floor)
                )

        return activity

    def _adapt_phases(
        self,
        phases: jnp.ndarray,
        mask: jnp.ndarray,
        circadian_phase: float,
        improvement: float,
        key: jax.random.PRNGKey,
        n_funcs: int,
        protected_indices: List[int],
    ) -> jnp.ndarray:
        """Adapt function phases based on fitness feedback.

        Protected indices have reduced phase noise for stability.
        """
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

        # Reduce noise for protected indices
        for idx in protected_indices:
            if 0 <= idx < n_funcs:
                noise = noise.at[idx].set(noise[idx] * 0.1)

        new_phases = new_phases + noise
        return jnp.mod(new_phases, 2 * jnp.pi)

    def _adapt_amplitudes(
        self,
        amplitudes: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
        n_funcs: int,
        protected_indices: List[int],
    ) -> jnp.ndarray:
        """Adapt oscillation amplitudes based on fitness.

        Protected indices keep low amplitude for stability.
        """
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

        new_amplitudes = jnp.clip(new_amplitudes, self.amplitude_min, self.amplitude_max)

        # Protected indices keep low amplitude for stability
        for idx in protected_indices:
            if 0 <= idx < n_funcs:
                new_amplitudes = new_amplitudes.at[idx].set(
                    jnp.minimum(new_amplitudes[idx], self.amplitude_min + 0.1)
                )

        return new_amplitudes

    def _select_palette_protected(
        self,
        activity: jnp.ndarray,
        affinity: jnp.ndarray,
        palette_size: int,
        min_active: int,
        protected_indices: List[int],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette based on activity levels with protected indices.

        Protected indices only deactivated with 0.1% probability.
        """
        n_funcs = len(activity)

        # Combine activity and affinity for selection score
        selection_score = 0.6 * activity + 0.4 * affinity

        # Start with all above threshold
        above_threshold = selection_score >= self.activity_threshold
        n_above = int(jnp.sum(above_threshold))

        if n_above >= min_active and n_above <= palette_size:
            mask = above_threshold.astype(jnp.float32)
        elif n_above < min_active:
            top_k = jnp.argsort(selection_score)[-min_active:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            top_k = jnp.argsort(selection_score)[-palette_size:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)

        # Protected indices: only 0.1% chance of deactivation
        key, subkey = jax.random.split(key)
        deactivate_probs = jax.random.uniform(subkey, (n_funcs,))

        for idx in protected_indices:
            if 0 <= idx < n_funcs:
                # If protected and would be deactivated, check probability
                if mask[idx] < 0.5:
                    if deactivate_probs[idx] >= self.protected_deactivation_prob:
                        # Keep protected index active
                        mask = mask.at[idx].set(1.0)

        return mask

    def _update_memory_cells(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell formation based on sustained performance."""
        active = mask > 0.5
        above_threshold = affinity >= self.memory_formation_threshold

        # Candidate for memory formation: active AND above threshold
        candidate = active & above_threshold

        # Increment count for candidates, reset for non-candidates
        new_counts = jnp.where(candidate, memory_counts + 1, 0)

        # Form memory cells when count reaches threshold
        newly_memory = new_counts >= self.memory_formation_count
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity based on co-activation success."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        delta = self.cross_entrainment * improvement * co_active
        new_cross = cross_affinity + delta
        return jnp.clip(new_cross, 0.0, 1.0)

    def post_generation_update(
        self,
        state: CircadianRhythmSymmetricState,
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[CircadianRhythmSymmetricState, Dict[str, Any]]:
        """Update with circadian dynamics and protection."""
        key = state.rng_key
        key, k_act, k_agg, k_sel_act, k_sel_agg = jax.random.split(key, 5)

        improved = best_fitness > state.best_fitness_seen
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state.stagnation_count + 1
            new_best = state.best_fitness_seen

        # Advance master clock
        phase_increment = 2 * jnp.pi / self.circadian_period
        new_circadian_phase = state.circadian_phase + phase_increment
        cycles_completed = state.cycles_completed
        if new_circadian_phase >= 2 * jnp.pi:
            new_circadian_phase = new_circadian_phase % (2 * jnp.pi)
            cycles_completed += 1

        # Update cross-affinity
        new_cross = self._update_cross_affinity(
            state.cross_affinity,
            state.act_mask,
            state.agg_mask,
            improvement,
        )

        # Update affinities based on fitness
        fitness_delta = improvement if improvement > 0 else -0.01
        new_act_affinity = state.act_affinity + fitness_delta * 0.1 * state.act_mask
        new_agg_affinity = state.agg_affinity + fitness_delta * 0.1 * state.agg_mask
        new_act_affinity = jnp.clip(new_act_affinity * 0.99, 0.0, 1.0)  # Slight decay
        new_agg_affinity = jnp.clip(new_agg_affinity * 0.99, 0.0, 1.0)

        # Apply affinity floors
        new_act_affinity, new_agg_affinity = self._apply_affinity_floors(
            new_act_affinity, new_agg_affinity
        )

        # Adapt phases for both domains
        new_act_phases = self._adapt_phases(
            state.act_phases, state.act_mask,
            new_circadian_phase, improvement, k_act, NUM_ACTIVATIONS, [SIN_IDX]
        )
        new_agg_phases = self._adapt_phases(
            state.agg_phases, state.agg_mask,
            new_circadian_phase, improvement, k_agg, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX]
        )

        # Adapt amplitudes
        new_act_amplitudes = self._adapt_amplitudes(
            state.act_amplitudes, state.act_mask, improvement, NUM_ACTIVATIONS, [SIN_IDX]
        )
        new_agg_amplitudes = self._adapt_amplitudes(
            state.agg_amplitudes, state.agg_mask, improvement, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX]
        )

        # Compute new activity levels
        new_act_activity = self._compute_activity(
            new_circadian_phase, new_act_phases, new_act_amplitudes, [SIN_IDX]
        )
        new_agg_activity = self._compute_activity(
            new_circadian_phase, new_agg_phases, new_agg_amplitudes, [MAX_IDX, MIN_IDX]
        )

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_affinity, state.act_mask,
            state.act_memory_counts, state.act_memory_cells
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_affinity, state.agg_mask,
            state.agg_memory_counts, state.agg_memory_cells
        )

        # Select palettes with protection
        new_act_mask = self._select_palette_protected(
            new_act_activity, new_act_affinity,
            self.act_palette_size, self.min_active_act, [SIN_IDX], k_sel_act
        )
        new_agg_mask = self._select_palette_protected(
            new_agg_activity, new_agg_affinity,
            self.agg_palette_size, self.min_active_agg, [MAX_IDX, MIN_IDX], k_sel_agg
        )

        # Memory cells are always active
        new_act_mask = jnp.where(new_act_mem_cells, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem_cells, 1.0, new_agg_mask)

        fitness_history = state.fitness_history + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = CircadianRhythmSymmetricState(
            act_mask=new_act_mask,
            agg_mask=new_agg_mask,
            act_phases=new_act_phases,
            agg_phases=new_agg_phases,
            act_amplitudes=new_act_amplitudes,
            agg_amplitudes=new_agg_amplitudes,
            act_activity=new_act_activity,
            agg_activity=new_agg_activity,
            act_affinity=new_act_affinity,
            agg_affinity=new_agg_affinity,
            cross_affinity=new_cross,
            act_memory_cells=new_act_mem_cells,
            agg_memory_cells=new_agg_mem_cells,
            act_memory_counts=new_act_mem_counts,
            agg_memory_counts=new_agg_mem_counts,
            circadian_phase=float(new_circadian_phase),
            cycles_completed=cycles_completed,
            rng_key=key,
            generation=generation + 1,
            stagnation_count=new_stagnation,
            best_fitness_seen=new_best,
            fitness_history=fitness_history,
        )

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': not jnp.allclose(state.act_mask, new_act_mask),
            'agg_palette_changed': not jnp.allclose(state.agg_mask, new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'circadian_phase': float(new_circadian_phase),
            'cycles_completed': cycles_completed,
            'act_memory_cells': int(jnp.sum(new_act_mem_cells)),
            'agg_memory_cells': int(jnp.sum(new_agg_mem_cells)),
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'sin_affinity': float(new_act_affinity[SIN_IDX]),
            'sin_activity': float(new_act_activity[SIN_IDX]),
        }

        return new_state, metrics

    def get_state_summary(self, state: CircadianRhythmSymmetricState) -> Dict[str, Any]:
        """Return state summary."""
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': SIN_IDX in self.get_active_palette(state),
            'has_max': MAX_IDX in self.get_active_agg_palette(state),
            'has_min': MIN_IDX in self.get_active_agg_palette(state),
            'circadian_phase': state.circadian_phase,
            'cycles_completed': state.cycles_completed,
            'generation': state.generation,
            'act_memory_cells': int(jnp.sum(state.act_memory_cells)),
            'agg_memory_cells': int(jnp.sum(state.agg_memory_cells)),
        }
