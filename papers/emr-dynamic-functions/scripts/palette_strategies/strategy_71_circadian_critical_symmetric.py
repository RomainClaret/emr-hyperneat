"""Strategy 71S: Circadian Critical Symmetric (Critical Period Protection for Sin/Extremes).

Extends CircadianCriticalDualStrategy to symmetric mode with winning patterns:
- Critical period protection for sin and extreme aggregations
- Affinity floors applied every generation
- Memory cell crystallization for sustained performance
- Cross-domain critical period coordination

Key symmetric mechanisms:
1. Phase-independent protection - sin/max/min bypass critical period gates
2. Critical windows - non-protected functions have limited plasticity windows
3. Cross-domain synchronization - critical periods align across domains
4. Consolidated protection - functions crystallize after critical period success

Biological Insight: Like how critical periods in development protect essential
neural circuits from further modification, protected indices are immune to
critical period restrictions, maintaining plasticity when others cannot change.
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
class CircadianCriticalSymmetricState:
    """State for circadian critical symmetric strategy."""
    # Palette masks
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    # Critical period tracking
    act_critical_open: jnp.ndarray  # Whether critical period is open
    agg_critical_open: jnp.ndarray
    act_critical_age: jnp.ndarray  # Age since function became active
    agg_critical_age: jnp.ndarray
    act_consolidated: jnp.ndarray  # Whether function is consolidated
    agg_consolidated: jnp.ndarray
    # Circadian tracking
    circadian_phase: float
    cycles_completed: int
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


class CircadianCriticalSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with critical period dynamics and protection.

    Applies winning patterns:
    - Protected indices bypass critical period restrictions
    - Affinity floors (sin=0.6, extreme_agg=0.5)
    - Memory cell crystallization (8 gens sustained)
    - Consolidated functions resist change
    """

    name = "circadian_critical_symmetric"
    description = "Symmetric: Critical periods with protected bypass"

    def __init__(
        self,
        # Circadian parameters
        circadian_period: int = 20,
        # Critical period parameters
        critical_period_length: int = 15,
        consolidation_threshold: float = 0.7,
        consolidated_plasticity: float = 0.1,  # Reduced plasticity after consolidation
        # Protection parameters (winning patterns)
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,  # 0.1%
        protected_always_plastic: bool = True,  # Protected indices bypass critical periods
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Cross-domain
        cross_critical_coupling: float = 0.15,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 4,
        min_active_act: int = 3,
        min_active_agg: int = 2,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Circadian Critical Symmetric strategy."""
        # Circadian parameters
        self.circadian_period = circadian_period

        # Critical period parameters
        self.critical_period_length = critical_period_length
        self.consolidation_threshold = consolidation_threshold
        self.consolidated_plasticity = consolidated_plasticity

        # Protection parameters
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.protected_always_plastic = protected_always_plastic
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Cross-domain
        self.cross_critical_coupling = cross_critical_coupling

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

    def initialize(self, config: Dict[str, Any], seed: int) -> CircadianCriticalSymmetricState:
        """Initialize state with critical period tracking and protection."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 717100)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize critical period tracking
        act_critical_open = jnp.ones(NUM_ACTIVATIONS, dtype=jnp.bool_)  # All start open
        agg_critical_open = jnp.ones(NUM_AGGREGATIONS, dtype=jnp.bool_)
        act_critical_age = jnp.zeros(NUM_ACTIVATIONS)
        agg_critical_age = jnp.zeros(NUM_AGGREGATIONS)
        act_consolidated = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_consolidated = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

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

        return CircadianCriticalSymmetricState(
            act_mask=act_mask,
            agg_mask=agg_mask,
            act_critical_open=act_critical_open,
            agg_critical_open=agg_critical_open,
            act_critical_age=act_critical_age,
            agg_critical_age=agg_critical_age,
            act_consolidated=act_consolidated,
            agg_consolidated=agg_consolidated,
            circadian_phase=0.0,
            cycles_completed=0,
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

    def get_active_palette(self, state: CircadianCriticalSymmetricState) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state: CircadianCriticalSymmetricState) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _update_critical_periods(
        self,
        critical_open: jnp.ndarray,
        critical_age: jnp.ndarray,
        consolidated: jnp.ndarray,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        protected_indices: List[int],
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update critical period state.

        Protected indices always have open critical periods.
        """
        new_open = critical_open.copy()
        new_age = critical_age.copy()
        new_consolidated = consolidated.copy()

        for i in range(n_funcs):
            is_protected = i in protected_indices

            if mask[i] > 0.5:
                # Active - increment age
                new_age = new_age.at[i].set(critical_age[i] + 1)

                # Check if critical period should close
                if new_age[i] >= self.critical_period_length and not is_protected:
                    new_open = new_open.at[i].set(False)

                    # Consolidate if affinity is high enough
                    if affinity[i] >= self.consolidation_threshold:
                        new_consolidated = new_consolidated.at[i].set(True)
            else:
                # Inactive - reset age
                new_age = new_age.at[i].set(0.0)

            # Protected indices always have open critical periods
            if is_protected:
                new_open = new_open.at[i].set(True)

        return new_open, new_age, new_consolidated

    def _get_plasticity(
        self,
        critical_open: jnp.ndarray,
        consolidated: jnp.ndarray,
        protected_indices: List[int],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Get plasticity level for each function."""
        plasticity = jnp.ones(n_funcs)

        for i in range(n_funcs):
            is_protected = i in protected_indices

            if is_protected:
                # Protected indices always fully plastic
                plasticity = plasticity.at[i].set(1.0)
            elif consolidated[i]:
                # Consolidated functions have reduced plasticity
                plasticity = plasticity.at[i].set(self.consolidated_plasticity)
            elif not critical_open[i]:
                # Critical period closed, moderate plasticity
                plasticity = plasticity.at[i].set(0.3)

        return plasticity

    def _select_palette_protected(
        self,
        affinity: jnp.ndarray,
        plasticity: jnp.ndarray,
        palette_size: int,
        min_active: int,
        protected_indices: List[int],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette with protection."""
        n_funcs = len(affinity)

        # Score combines affinity and plasticity
        scores = 0.7 * affinity + 0.3 * plasticity

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
        state: CircadianCriticalSymmetricState,
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[CircadianCriticalSymmetricState, Dict[str, Any]]:
        """Update with critical period dynamics and protection."""
        key = state.rng_key
        key, k_act, k_agg = jax.random.split(key, 3)

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

        # Update critical periods
        new_act_open, new_act_age, new_act_consolidated = self._update_critical_periods(
            state.act_critical_open, state.act_critical_age, state.act_consolidated,
            state.act_mask, state.act_affinity, [SIN_IDX], NUM_ACTIVATIONS
        )
        new_agg_open, new_agg_age, new_agg_consolidated = self._update_critical_periods(
            state.agg_critical_open, state.agg_critical_age, state.agg_consolidated,
            state.agg_mask, state.agg_affinity, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS
        )

        # Get plasticity levels
        act_plasticity = self._get_plasticity(new_act_open, new_act_consolidated, [SIN_IDX], NUM_ACTIVATIONS)
        agg_plasticity = self._get_plasticity(new_agg_open, new_agg_consolidated, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS)

        # Update affinities (modulated by plasticity)
        fitness_delta = improvement if improvement > 0 else -0.01
        new_act_affinity = state.act_affinity + fitness_delta * 0.1 * state.act_mask * act_plasticity
        new_agg_affinity = state.agg_affinity + fitness_delta * 0.1 * state.agg_mask * agg_plasticity
        new_act_affinity = jnp.clip(new_act_affinity * 0.99, 0.0, 1.0)
        new_agg_affinity = jnp.clip(new_agg_affinity * 0.99, 0.0, 1.0)
        new_act_affinity, new_agg_affinity = self._apply_affinity_floors(
            new_act_affinity, new_agg_affinity
        )

        # Update cross affinity
        active_act = (state.act_mask > 0.5).astype(jnp.float32)
        active_agg = (state.agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        new_cross = state.cross_affinity + self.cross_critical_coupling * improvement * co_active
        new_cross = jnp.clip(new_cross, 0.0, 1.0)

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
        new_act_mask = self._select_palette_protected(
            new_act_affinity, act_plasticity,
            self.act_palette_size, self.min_active_act, [SIN_IDX], k_act
        )
        new_agg_mask = self._select_palette_protected(
            new_agg_affinity, agg_plasticity,
            self.agg_palette_size, self.min_active_agg, [MAX_IDX, MIN_IDX], k_agg
        )

        # Memory cells always active
        new_act_mask = jnp.where(new_act_mem_cells, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem_cells, 1.0, new_agg_mask)

        fitness_history = state.fitness_history + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = CircadianCriticalSymmetricState(
            act_mask=new_act_mask,
            agg_mask=new_agg_mask,
            act_critical_open=new_act_open,
            agg_critical_open=new_agg_open,
            act_critical_age=new_act_age,
            agg_critical_age=new_agg_age,
            act_consolidated=new_act_consolidated,
            agg_consolidated=new_agg_consolidated,
            circadian_phase=float(new_circadian_phase),
            cycles_completed=cycles_completed,
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
            'act_consolidated': int(jnp.sum(new_act_consolidated)),
            'agg_consolidated': int(jnp.sum(new_agg_consolidated)),
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'sin_affinity': float(new_act_affinity[SIN_IDX]),
        }

        return new_state, metrics

    def get_state_summary(self, state: CircadianCriticalSymmetricState) -> Dict[str, Any]:
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
            'act_consolidated': int(jnp.sum(state.act_consolidated)),
            'agg_consolidated': int(jnp.sum(state.agg_consolidated)),
        }
