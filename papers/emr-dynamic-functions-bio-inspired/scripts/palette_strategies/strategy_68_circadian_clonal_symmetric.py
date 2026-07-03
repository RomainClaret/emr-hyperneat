"""Strategy 68S: Circadian Clonal Symmetric (Clone Priority for Protected Lineages).

Extends CircadianClonalDualStrategy to symmetric mode with winning patterns:
- Clone priority for sin/extreme lineages
- Affinity floors for sin and extreme aggregations
- Memory cell crystallization for sustained performance
- Cross-domain clonal coordination

Key symmetric mechanisms:
1. Lineage priority - sin/max/min lineages get clone priority
2. Circadian-gated cloning - clones inherit parent's phase relationship
3. Cross-domain lineage coupling - successful pairs clone together
4. Protected proliferation - critical functions always proliferate

Biological Insight: Like stem cell niches that prioritize certain lineages,
protected indices represent essential lineages that maintain priority
access to proliferation signals regardless of circadian phase.
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
class CircadianClonalSymmetricState:
    """State for circadian clonal symmetric strategy."""
    # Palette masks
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    # Circadian tracking
    act_phases: jnp.ndarray
    agg_phases: jnp.ndarray
    circadian_phase: float
    cycles_completed: int
    # Clonal lineage tracking
    act_lineage_strength: jnp.ndarray  # Proliferation potential
    agg_lineage_strength: jnp.ndarray
    act_clone_count: jnp.ndarray  # Number of times function was cloned
    agg_clone_count: jnp.ndarray
    # Affinities
    act_affinity: jnp.ndarray
    agg_affinity: jnp.ndarray
    cross_affinity: jnp.ndarray
    # Memory cells
    act_memory_cells: jnp.ndarray
    agg_memory_cells: jnp.ndarray
    act_memory_counts: jnp.ndarray
    agg_memory_counts: jnp.ndarray
    # General state
    rng_key: jnp.ndarray
    generation: int
    stagnation_count: int
    best_fitness_seen: float
    fitness_history: List[float] = field(default_factory=list)


class CircadianClonalSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with circadian-gated cloning and protection.

    Applies winning patterns:
    - Protected lineages get clone priority
    - Affinity floors (sin=0.6, extreme_agg=0.5)
    - Memory cell crystallization (8 gens sustained)
    - Cross-domain lineage coupling
    """

    name = "circadian_clonal_symmetric"
    description = "Symmetric: Circadian clonal with protected lineage priority"

    def __init__(
        self,
        # Circadian parameters
        circadian_period: int = 20,
        phase_noise: float = 0.1,
        activity_threshold: float = 0.4,
        # Clonal parameters
        clone_threshold: float = 0.6,
        lineage_boost_rate: float = 0.15,
        lineage_decay: float = 0.98,
        protected_lineage_priority: float = 1.5,  # 50% higher priority
        # Protection parameters (winning patterns)
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,  # 0.1%
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Cross-domain
        cross_lineage_coupling: float = 0.2,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 4,
        min_active_act: int = 3,
        min_active_agg: int = 2,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Circadian Clonal Symmetric strategy."""
        # Circadian parameters
        self.circadian_period = circadian_period
        self.phase_noise = phase_noise
        self.activity_threshold = activity_threshold

        # Clonal parameters
        self.clone_threshold = clone_threshold
        self.lineage_boost_rate = lineage_boost_rate
        self.lineage_decay = lineage_decay
        self.protected_lineage_priority = protected_lineage_priority

        # Protection parameters
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Cross-domain
        self.cross_lineage_coupling = cross_lineage_coupling

        # Palette composition
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        # CRITICAL: Include sin and extreme aggregations in initial palettes
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for protected indices."""
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )
        new_agg = agg_affinity
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )
        return new_act, new_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> CircadianClonalSymmetricState:
        """Initialize state with circadian-clonal tracking and protection."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 686800)
        key, k_act, k_agg = jax.random.split(key, 3)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize phases
        act_phases = jax.random.uniform(k_act, (NUM_ACTIVATIONS,)) * 2 * jnp.pi
        agg_phases = jax.random.uniform(k_agg, (NUM_AGGREGATIONS,)) * 2 * jnp.pi

        # Initial palette and protected indices start in-phase
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_phases = act_phases.at[i].set(0.0)
        act_phases = act_phases.at[SIN_IDX].set(0.0)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_phases = agg_phases.at[i].set(0.0)
        for idx in [MAX_IDX, MIN_IDX]:
            agg_phases = agg_phases.at[idx].set(0.0)

        # Initialize lineage strength (protected get priority)
        act_lineage = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_lineage = jnp.ones(NUM_AGGREGATIONS) * 0.5

        act_lineage = act_lineage.at[SIN_IDX].set(self.protected_lineage_priority)
        for idx in [MAX_IDX, MIN_IDX]:
            agg_lineage = agg_lineage.at[idx].set(self.protected_lineage_priority)

        # Initialize affinities with floors
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        act_affinity, agg_affinity = self._apply_affinity_floors(act_affinity, agg_affinity)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg_idx in [MAX_IDX, MIN_IDX]:
            cross_affinity = cross_affinity.at[SIN_IDX, agg_idx].set(0.7)

        # Memory cells
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS)

        return CircadianClonalSymmetricState(
            act_mask=act_mask,
            agg_mask=agg_mask,
            act_phases=act_phases,
            agg_phases=agg_phases,
            circadian_phase=0.0,
            cycles_completed=0,
            act_lineage_strength=act_lineage,
            agg_lineage_strength=agg_lineage,
            act_clone_count=jnp.zeros(NUM_ACTIVATIONS),
            agg_clone_count=jnp.zeros(NUM_AGGREGATIONS),
            act_affinity=act_affinity,
            agg_affinity=agg_affinity,
            cross_affinity=cross_affinity,
            act_memory_cells=act_memory_cells,
            agg_memory_cells=agg_memory_cells,
            act_memory_counts=act_memory_counts,
            agg_memory_counts=agg_memory_counts,
            rng_key=key,
            generation=0,
            stagnation_count=0,
            best_fitness_seen=0.0,
            fitness_history=[],
        )

    def get_active_palette(self, state: CircadianClonalSymmetricState) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state: CircadianClonalSymmetricState) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _compute_activity(
        self,
        circadian_phase: float,
        function_phases: jnp.ndarray,
        protected_indices: List[int],
    ) -> jnp.ndarray:
        """Compute circadian activity levels."""
        n_funcs = len(function_phases)
        activity = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            phase_diff = circadian_phase - float(function_phases[i])
            act = (1 + jnp.cos(phase_diff)) / 2
            activity = activity.at[i].set(act)

        # Protected indices always have high activity
        for idx in protected_indices:
            if 0 <= idx < n_funcs:
                activity = activity.at[idx].set(
                    jnp.maximum(activity[idx], 0.7)
                )

        return activity

    def _update_lineage_strength(
        self,
        lineage: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
        protected_indices: List[int],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update lineage strength based on success.

        Protected lineages maintain priority.
        """
        new_lineage = lineage * self.lineage_decay

        for i in range(n_funcs):
            if mask[i] > 0.5 and improvement > 0:
                new_lineage = new_lineage.at[i].set(
                    lineage[i] + self.lineage_boost_rate * improvement
                )

        # Protected indices maintain minimum lineage strength
        for idx in protected_indices:
            if 0 <= idx < n_funcs:
                new_lineage = new_lineage.at[idx].set(
                    jnp.maximum(new_lineage[idx], self.protected_lineage_priority * 0.8)
                )

        return jnp.clip(new_lineage, 0.0, 2.0)

    def _select_palette_clonal(
        self,
        activity: jnp.ndarray,
        lineage: jnp.ndarray,
        affinity: jnp.ndarray,
        palette_size: int,
        min_active: int,
        protected_indices: List[int],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette using clonal selection with protection."""
        n_funcs = len(activity)

        # Combine activity, lineage, and affinity
        scores = 0.3 * activity + 0.4 * lineage + 0.3 * affinity

        # Select top k
        top_k = jnp.argsort(scores)[-palette_size:]
        mask = jnp.zeros(n_funcs)
        for idx in top_k:
            mask = mask.at[int(idx)].set(1.0)

        # Ensure minimum
        n_active = int(jnp.sum(mask))
        if n_active < min_active:
            remaining = jnp.argsort(scores)[-(min_active):]
            for idx in remaining:
                mask = mask.at[int(idx)].set(1.0)

        # Protected indices: only 0.1% chance of deactivation
        key, subkey = jax.random.split(key)
        deactivate_probs = jax.random.uniform(subkey, (n_funcs,))

        for idx in protected_indices:
            if 0 <= idx < n_funcs:
                if mask[idx] < 0.5:
                    if deactivate_probs[idx] >= self.protected_deactivation_prob:
                        mask = mask.at[idx].set(1.0)

        return mask

    def _update_memory_cells(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell formation."""
        active = mask > 0.5
        above_threshold = affinity >= self.memory_formation_threshold
        candidate = active & above_threshold
        new_counts = jnp.where(candidate, memory_counts + 1, 0)
        newly_memory = new_counts >= self.memory_formation_count
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)
        return new_counts, new_memory_cells

    def post_generation_update(
        self,
        state: CircadianClonalSymmetricState,
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[CircadianClonalSymmetricState, Dict[str, Any]]:
        """Update with circadian-clonal dynamics and protection."""
        key = state.rng_key
        key, k_act, k_agg, k_phase = jax.random.split(key, 4)

        improved = best_fitness > state.best_fitness_seen
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state.stagnation_count + 1
            new_best = state.best_fitness_seen

        # Advance circadian clock
        phase_increment = 2 * jnp.pi / self.circadian_period
        new_circadian_phase = state.circadian_phase + phase_increment
        cycles_completed = state.cycles_completed
        if new_circadian_phase >= 2 * jnp.pi:
            new_circadian_phase = new_circadian_phase % (2 * jnp.pi)
            cycles_completed += 1

        # Update phases with noise
        phase_noise = jax.random.normal(k_phase, (NUM_ACTIVATIONS + NUM_AGGREGATIONS,)) * self.phase_noise
        new_act_phases = jnp.mod(state.act_phases + phase_noise[:NUM_ACTIVATIONS], 2 * jnp.pi)
        new_agg_phases = jnp.mod(state.agg_phases + phase_noise[NUM_ACTIVATIONS:], 2 * jnp.pi)

        # Protected indices stay in phase
        new_act_phases = new_act_phases.at[SIN_IDX].set(new_circadian_phase)
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg_phases = new_agg_phases.at[idx].set(new_circadian_phase)

        # Update lineage strength
        new_act_lineage = self._update_lineage_strength(
            state.act_lineage_strength, state.act_mask, improvement, [SIN_IDX], NUM_ACTIVATIONS
        )
        new_agg_lineage = self._update_lineage_strength(
            state.agg_lineage_strength, state.agg_mask, improvement, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS
        )

        # Update affinities
        fitness_delta = improvement if improvement > 0 else -0.01
        new_act_affinity = state.act_affinity + fitness_delta * 0.1 * state.act_mask
        new_agg_affinity = state.agg_affinity + fitness_delta * 0.1 * state.agg_mask
        new_act_affinity = jnp.clip(new_act_affinity * 0.99, 0.0, 1.0)
        new_agg_affinity = jnp.clip(new_agg_affinity * 0.99, 0.0, 1.0)
        new_act_affinity, new_agg_affinity = self._apply_affinity_floors(
            new_act_affinity, new_agg_affinity
        )

        # Update cross affinity
        active_act = (state.act_mask > 0.5).astype(jnp.float32)
        active_agg = (state.agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        new_cross = state.cross_affinity + self.cross_lineage_coupling * improvement * co_active
        new_cross = jnp.clip(new_cross, 0.0, 1.0)

        # Compute activity
        act_activity = self._compute_activity(new_circadian_phase, new_act_phases, [SIN_IDX])
        agg_activity = self._compute_activity(new_circadian_phase, new_agg_phases, [MAX_IDX, MIN_IDX])

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_affinity, state.act_mask,
            state.act_memory_counts, state.act_memory_cells
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_affinity, state.agg_mask,
            state.agg_memory_counts, state.agg_memory_cells
        )

        # Select palettes
        new_act_mask = self._select_palette_clonal(
            act_activity, new_act_lineage, new_act_affinity,
            self.act_palette_size, self.min_active_act, [SIN_IDX], k_act
        )
        new_agg_mask = self._select_palette_clonal(
            agg_activity, new_agg_lineage, new_agg_affinity,
            self.agg_palette_size, self.min_active_agg, [MAX_IDX, MIN_IDX], k_agg
        )

        # Memory cells always active
        new_act_mask = jnp.where(new_act_mem_cells, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem_cells, 1.0, new_agg_mask)

        fitness_history = state.fitness_history + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = CircadianClonalSymmetricState(
            act_mask=new_act_mask,
            agg_mask=new_agg_mask,
            act_phases=new_act_phases,
            agg_phases=new_agg_phases,
            circadian_phase=float(new_circadian_phase),
            cycles_completed=cycles_completed,
            act_lineage_strength=new_act_lineage,
            agg_lineage_strength=new_agg_lineage,
            act_clone_count=state.act_clone_count,
            agg_clone_count=state.agg_clone_count,
            act_affinity=new_act_affinity,
            agg_affinity=new_agg_affinity,
            cross_affinity=new_cross,
            act_memory_cells=new_act_mem_cells,
            agg_memory_cells=new_agg_mem_cells,
            act_memory_counts=new_act_mem_counts,
            agg_memory_counts=new_agg_mem_counts,
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
            'sin_lineage': float(new_act_lineage[SIN_IDX]),
        }

        return new_state, metrics

    def get_state_summary(self, state: CircadianClonalSymmetricState) -> Dict[str, Any]:
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
