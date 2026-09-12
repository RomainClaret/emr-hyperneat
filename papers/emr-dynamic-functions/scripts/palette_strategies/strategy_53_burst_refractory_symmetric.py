"""Strategy 53S: Burst Refractory Symmetric (Protected Refractory Dynamics).

Extends BurstRefractoryDualStrategy to symmetric mode with winning patterns:
- Protected indices skip refractory periods
- Affinity floors for sin and extreme aggregations
- Memory cell crystallization for sustained performance
- Cross-domain affinity matrix for coordinated discovery

Key symmetric mechanisms:
1. Protected refractory skip - sin/max/min never enter refractory period
2. Affinity floor enforcement - applied every generation
3. Memory cell formation - 8 gens sustained → permanent protection
4. Cross-domain burst coordination - synchronized burst windows

Biological Insight: Like certain pacemaker neurons that never enter full
refractory, protected indices maintain constant readiness while allowing
other functions to cycle through burst and recovery phases.
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
class BurstRefractorySymmetricState:
    """State for burst refractory symmetric strategy."""
    # Palette masks
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    # Burst/refractory tracking
    act_burst_timers: jnp.ndarray  # Time in current burst (0=refractory)
    agg_burst_timers: jnp.ndarray
    act_refractory_remaining: jnp.ndarray  # Time remaining in refractory
    agg_refractory_remaining: jnp.ndarray
    act_burst_strength: jnp.ndarray  # Accumulated burst effectiveness
    agg_burst_strength: jnp.ndarray
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


class BurstRefractorySymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with burst-refractory dynamics and protection.

    Applies winning patterns:
    - Protected indices skip refractory periods (0.1% deactivation)
    - Affinity floors (sin=0.6, extreme_agg=0.5)
    - Memory cell crystallization (8 gens sustained)
    - Burst coordination across domains
    """

    name = "burst_refractory_symmetric"
    description = "Symmetric: Burst-refractory with protected pacemakers"

    def __init__(
        self,
        # Burst dynamics
        burst_duration: int = 8,
        refractory_duration: int = 4,
        burst_boost: float = 0.3,
        fatigue_rate: float = 0.05,
        recovery_rate: float = 0.1,
        # Protection parameters (winning patterns)
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,  # 0.1%
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Selection
        selection_threshold: float = 0.4,
        cross_burst_coupling: float = 0.15,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 4,
        min_active_act: int = 3,
        min_active_agg: int = 2,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Burst Refractory Symmetric strategy."""
        # Burst dynamics
        self.burst_duration = burst_duration
        self.refractory_duration = refractory_duration
        self.burst_boost = burst_boost
        self.fatigue_rate = fatigue_rate
        self.recovery_rate = recovery_rate

        # Protection parameters
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Selection
        self.selection_threshold = selection_threshold
        self.cross_burst_coupling = cross_burst_coupling

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

    def initialize(self, config: Dict[str, Any], seed: int) -> BurstRefractorySymmetricState:
        """Initialize state with burst-refractory tracking and protection."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 535300)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize burst timers (protected indices start mid-burst)
        act_burst_timers = jnp.zeros(NUM_ACTIVATIONS)
        agg_burst_timers = jnp.zeros(NUM_AGGREGATIONS)

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_burst_timers = act_burst_timers.at[i].set(self.burst_duration // 2)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_burst_timers = agg_burst_timers.at[i].set(self.burst_duration // 2)

        # Protected indices always in burst mode
        act_burst_timers = act_burst_timers.at[SIN_IDX].set(self.burst_duration)
        for idx in [MAX_IDX, MIN_IDX]:
            agg_burst_timers = agg_burst_timers.at[idx].set(self.burst_duration)

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

        return BurstRefractorySymmetricState(
            act_mask=act_mask,
            agg_mask=agg_mask,
            act_burst_timers=act_burst_timers,
            agg_burst_timers=agg_burst_timers,
            act_refractory_remaining=jnp.zeros(NUM_ACTIVATIONS),
            agg_refractory_remaining=jnp.zeros(NUM_AGGREGATIONS),
            act_burst_strength=jnp.ones(NUM_ACTIVATIONS),
            agg_burst_strength=jnp.ones(NUM_AGGREGATIONS),
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

    def get_active_palette(self, state: BurstRefractorySymmetricState) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state: BurstRefractorySymmetricState) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _update_burst_refractory(
        self,
        burst_timers: jnp.ndarray,
        refractory_remaining: jnp.ndarray,
        burst_strength: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
        protected_indices: List[int],
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update burst and refractory state.

        Protected indices skip refractory periods.
        """
        new_burst = burst_timers.copy()
        new_refractory = refractory_remaining.copy()
        new_strength = burst_strength.copy()

        for i in range(n_funcs):
            is_protected = i in protected_indices

            if refractory_remaining[i] > 0 and not is_protected:
                # In refractory - recover strength
                new_refractory = new_refractory.at[i].set(refractory_remaining[i] - 1)
                new_strength = new_strength.at[i].set(
                    jnp.minimum(burst_strength[i] + self.recovery_rate, 1.0)
                )
            elif burst_timers[i] > 0:
                # In burst - decrement timer, apply fatigue
                new_burst = new_burst.at[i].set(burst_timers[i] - 1)
                if mask[i] > 0.5:
                    new_strength = new_strength.at[i].set(
                        jnp.maximum(burst_strength[i] - self.fatigue_rate, 0.3)
                    )

                # Check if burst ending
                if new_burst[i] <= 0 and not is_protected:
                    new_refractory = new_refractory.at[i].set(float(self.refractory_duration))
            else:
                # Neither bursting nor refractory - start new burst
                new_burst = new_burst.at[i].set(float(self.burst_duration))

            # Protected indices never enter refractory
            if is_protected:
                new_refractory = new_refractory.at[i].set(0.0)
                new_burst = new_burst.at[i].set(
                    jnp.maximum(new_burst[i], float(self.burst_duration) // 2)
                )

        return new_burst, new_refractory, new_strength

    def _compute_effective_scores(
        self,
        affinity: jnp.ndarray,
        burst_strength: jnp.ndarray,
        burst_timers: jnp.ndarray,
        refractory_remaining: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective selection scores."""
        # Base score from affinity
        scores = affinity.copy()

        n_funcs = len(affinity)
        for i in range(n_funcs):
            if burst_timers[i] > 0:
                # In burst - boost score
                scores = scores.at[i].set(scores[i] + self.burst_boost * burst_strength[i])
            elif refractory_remaining[i] > 0:
                # In refractory - reduce score
                scores = scores.at[i].set(scores[i] * 0.3)

        return jnp.clip(scores, 0.0, 1.0)

    def _select_palette_protected(
        self,
        scores: jnp.ndarray,
        palette_size: int,
        min_active: int,
        protected_indices: List[int],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette with protected indices."""
        n_funcs = len(scores)

        # Select top k by score
        top_k = jnp.argsort(scores)[-palette_size:]
        mask = jnp.zeros(n_funcs)
        for idx in top_k:
            mask = mask.at[int(idx)].set(1.0)

        # Ensure minimum active
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
        state: BurstRefractorySymmetricState,
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[BurstRefractorySymmetricState, Dict[str, Any]]:
        """Update with burst-refractory dynamics and protection."""
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

        # Update burst/refractory for both domains
        new_act_burst, new_act_refract, new_act_strength = self._update_burst_refractory(
            state.act_burst_timers, state.act_refractory_remaining,
            state.act_burst_strength, state.act_mask, improvement, [SIN_IDX], NUM_ACTIVATIONS
        )
        new_agg_burst, new_agg_refract, new_agg_strength = self._update_burst_refractory(
            state.agg_burst_timers, state.agg_refractory_remaining,
            state.agg_burst_strength, state.agg_mask, improvement, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS
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
        new_cross = state.cross_affinity + self.cross_burst_coupling * improvement * co_active
        new_cross = jnp.clip(new_cross, 0.0, 1.0)

        # Compute effective scores
        act_scores = self._compute_effective_scores(
            new_act_affinity, new_act_strength, new_act_burst, new_act_refract
        )
        agg_scores = self._compute_effective_scores(
            new_agg_affinity, new_agg_strength, new_agg_burst, new_agg_refract
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

        # Select palettes
        new_act_mask = self._select_palette_protected(
            act_scores, self.act_palette_size, self.min_active_act, [SIN_IDX], k_act
        )
        new_agg_mask = self._select_palette_protected(
            agg_scores, self.agg_palette_size, self.min_active_agg, [MAX_IDX, MIN_IDX], k_agg
        )

        # Memory cells always active
        new_act_mask = jnp.where(new_act_mem_cells, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem_cells, 1.0, new_agg_mask)

        fitness_history = state.fitness_history + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = BurstRefractorySymmetricState(
            act_mask=new_act_mask,
            agg_mask=new_agg_mask,
            act_burst_timers=new_act_burst,
            agg_burst_timers=new_agg_burst,
            act_refractory_remaining=new_act_refract,
            agg_refractory_remaining=new_agg_refract,
            act_burst_strength=new_act_strength,
            agg_burst_strength=new_agg_strength,
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
            'act_memory_cells': int(jnp.sum(new_act_mem_cells)),
            'agg_memory_cells': int(jnp.sum(new_agg_mem_cells)),
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'sin_affinity': float(new_act_affinity[SIN_IDX]),
            'act_bursting': int(jnp.sum(new_act_burst > 0)),
            'agg_bursting': int(jnp.sum(new_agg_burst > 0)),
        }

        return new_state, metrics

    def get_state_summary(self, state: BurstRefractorySymmetricState) -> Dict[str, Any]:
        """Return state summary."""
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': SIN_IDX in self.get_active_palette(state),
            'has_max': MAX_IDX in self.get_active_agg_palette(state),
            'has_min': MIN_IDX in self.get_active_agg_palette(state),
            'generation': state.generation,
            'act_memory_cells': int(jnp.sum(state.act_memory_cells)),
            'agg_memory_cells': int(jnp.sum(state.agg_memory_cells)),
        }
