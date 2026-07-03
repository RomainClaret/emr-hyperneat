"""Strategy 54S: Synaptic Fatigue Symmetric (Protected Fatigue Dynamics).

Extends SynapticFatigueDualStrategy to symmetric mode with winning patterns:
- Protected indices have fatigue floor preventing exhaustion
- Affinity floors for sin and extreme aggregations
- Memory cell crystallization for sustained performance
- Cross-domain fatigue coupling

Key symmetric mechanisms:
1. Protected fatigue floor - sin/max/min never fully fatigue
2. Use-dependent rotation in non-protected functions
3. Cross-domain fatigue coupling - high fatigue triggers rotation
4. Affinity floor enforcement every generation

Biological Insight: Like essential neurotransmitter systems that maintain
baseline activity even under heavy use, protected indices have reserve
capacity that prevents complete exhaustion.
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
class SynapticFatigueSymmetricState:
    """State for synaptic fatigue symmetric strategy."""
    # Palette masks
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    # Fatigue tracking
    act_fatigue: jnp.ndarray
    agg_fatigue: jnp.ndarray
    act_base_weights: jnp.ndarray
    agg_base_weights: jnp.ndarray
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
    act_rotation_count: int
    agg_rotation_count: int
    fitness_history: List[float] = field(default_factory=list)


class SynapticFatigueSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with fatigue dynamics and protection.

    Applies winning patterns:
    - Protected indices have fatigue floor (never fully exhausted)
    - Affinity floors (sin=0.6, extreme_agg=0.5)
    - Memory cell crystallization (8 gens sustained)
    - Cross-domain fatigue coupling
    """

    name = "synaptic_fatigue_symmetric"
    description = "Symmetric: Synaptic fatigue with protected reserve capacity"

    def __init__(
        self,
        # Fatigue dynamics
        fatigue_rate: float = 0.15,
        recovery_rate: float = 0.08,
        effectiveness_floor: float = 0.3,
        protected_fatigue_floor: float = 0.5,  # Protected functions never go below 50% effectiveness
        # Success-dependent modulation
        success_fatigue_reduction: float = 0.3,
        failure_fatigue_boost: float = 0.1,
        # Base weights
        base_weight_learning_rate: float = 0.1,
        base_weight_decay: float = 0.99,
        initial_base_weight: float = 1.0,
        # Protection parameters (winning patterns)
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,  # 0.1%
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Selection
        temperature: float = 0.5,
        min_effective_weight: float = 0.1,
        cross_fatigue_coupling: float = 0.1,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 4,
        min_active_act: int = 3,
        min_active_agg: int = 2,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Synaptic Fatigue Symmetric strategy."""
        # Fatigue dynamics
        self.fatigue_rate = fatigue_rate
        self.recovery_rate = recovery_rate
        self.effectiveness_floor = effectiveness_floor
        self.protected_fatigue_floor = protected_fatigue_floor

        # Success modulation
        self.success_fatigue_reduction = success_fatigue_reduction
        self.failure_fatigue_boost = failure_fatigue_boost

        # Base weights
        self.base_weight_learning_rate = base_weight_learning_rate
        self.base_weight_decay = base_weight_decay
        self.initial_base_weight = initial_base_weight

        # Protection parameters
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Selection
        self.temperature = temperature
        self.min_effective_weight = min_effective_weight
        self.cross_fatigue_coupling = cross_fatigue_coupling

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

    def initialize(self, config: Dict[str, Any], seed: int) -> SynapticFatigueSymmetricState:
        """Initialize state with fatigue tracking and protection."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 545400)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize fatigue (0 = no fatigue, 1 = fully fatigued)
        act_fatigue = jnp.zeros(NUM_ACTIVATIONS)
        agg_fatigue = jnp.zeros(NUM_AGGREGATIONS)

        # Initialize base weights
        act_weights = jnp.ones(NUM_ACTIVATIONS) * self.initial_base_weight
        agg_weights = jnp.ones(NUM_AGGREGATIONS) * self.initial_base_weight

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_weights = act_weights.at[i].set(self.initial_base_weight * 1.2)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_weights = agg_weights.at[i].set(self.initial_base_weight * 1.2)

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

        return SynapticFatigueSymmetricState(
            act_mask=act_mask,
            agg_mask=agg_mask,
            act_fatigue=act_fatigue,
            agg_fatigue=agg_fatigue,
            act_base_weights=act_weights,
            agg_base_weights=agg_weights,
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
            act_rotation_count=0,
            agg_rotation_count=0,
            fitness_history=[],
        )

    def get_active_palette(self, state: SynapticFatigueSymmetricState) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state: SynapticFatigueSymmetricState) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _update_fatigue(
        self,
        fatigue: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
        protected_indices: List[int],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update fatigue levels.

        Protected indices have fatigue floor (never fully exhausted).
        """
        new_fatigue = fatigue.copy()

        for i in range(n_funcs):
            is_protected = i in protected_indices

            if mask[i] > 0.5:
                # Active - accumulate fatigue
                if improvement > 0:
                    # Success reduces fatigue accumulation
                    delta = self.fatigue_rate * (1 - self.success_fatigue_reduction * improvement)
                else:
                    # Failure increases fatigue
                    delta = self.fatigue_rate * (1 + self.failure_fatigue_boost)

                new_val = fatigue[i] + delta
            else:
                # Inactive - recover
                new_val = fatigue[i] - self.recovery_rate

            # Apply fatigue floor for protected indices
            if is_protected:
                # Protected functions never go above 50% fatigue
                # (i.e., never below 50% effectiveness)
                max_fatigue = 1 - self.protected_fatigue_floor
                new_val = jnp.minimum(new_val, max_fatigue)

            new_fatigue = new_fatigue.at[i].set(jnp.clip(new_val, 0.0, 1.0))

        return new_fatigue

    def _compute_effective_weights(
        self,
        base_weights: jnp.ndarray,
        fatigue: jnp.ndarray,
        affinity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective weights considering fatigue and affinity."""
        # Effectiveness = 1 - fatigue (with floor)
        effectiveness = jnp.maximum(1 - fatigue, self.effectiveness_floor)

        # Effective weight = base * effectiveness * affinity
        effective = base_weights * effectiveness * (0.5 + 0.5 * affinity)

        return jnp.clip(effective, 0.0, 2.0)

    def _select_palette_protected(
        self,
        effective_weights: jnp.ndarray,
        palette_size: int,
        min_active: int,
        protected_indices: List[int],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette with protected indices."""
        n_funcs = len(effective_weights)

        # Select top k by effective weight
        top_k = jnp.argsort(effective_weights)[-palette_size:]
        mask = jnp.zeros(n_funcs)
        for idx in top_k:
            mask = mask.at[int(idx)].set(1.0)

        # Ensure minimum active
        n_active = int(jnp.sum(mask))
        if n_active < min_active:
            remaining = jnp.argsort(effective_weights)[-(min_active):]
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
        state: SynapticFatigueSymmetricState,
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[SynapticFatigueSymmetricState, Dict[str, Any]]:
        """Update with fatigue dynamics and protection."""
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

        # Update fatigue for both domains
        new_act_fatigue = self._update_fatigue(
            state.act_fatigue, state.act_mask, improvement, [SIN_IDX], NUM_ACTIVATIONS
        )
        new_agg_fatigue = self._update_fatigue(
            state.agg_fatigue, state.agg_mask, improvement, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS
        )

        # Update base weights
        new_act_weights = state.act_base_weights * self.base_weight_decay
        new_agg_weights = state.agg_base_weights * self.base_weight_decay

        if improvement > 0:
            # Boost weights for active functions on success
            new_act_weights = new_act_weights + self.base_weight_learning_rate * improvement * state.act_mask
            new_agg_weights = new_agg_weights + self.base_weight_learning_rate * improvement * state.agg_mask

        new_act_weights = jnp.clip(new_act_weights, 0.1, 2.0)
        new_agg_weights = jnp.clip(new_agg_weights, 0.1, 2.0)

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
        new_cross = state.cross_affinity + self.cross_fatigue_coupling * improvement * co_active
        new_cross = jnp.clip(new_cross, 0.0, 1.0)

        # Compute effective weights
        act_effective = self._compute_effective_weights(
            new_act_weights, new_act_fatigue, new_act_affinity
        )
        agg_effective = self._compute_effective_weights(
            new_agg_weights, new_agg_fatigue, new_agg_affinity
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
            act_effective, self.act_palette_size, self.min_active_act, [SIN_IDX], k_act
        )
        new_agg_mask = self._select_palette_protected(
            agg_effective, self.agg_palette_size, self.min_active_agg, [MAX_IDX, MIN_IDX], k_agg
        )

        # Memory cells always active
        new_act_mask = jnp.where(new_act_mem_cells, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem_cells, 1.0, new_agg_mask)

        # Track rotations
        act_rotation = int(not jnp.allclose(state.act_mask, new_act_mask))
        agg_rotation = int(not jnp.allclose(state.agg_mask, new_agg_mask))

        fitness_history = state.fitness_history + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = SynapticFatigueSymmetricState(
            act_mask=new_act_mask,
            agg_mask=new_agg_mask,
            act_fatigue=new_act_fatigue,
            agg_fatigue=new_agg_fatigue,
            act_base_weights=new_act_weights,
            agg_base_weights=new_agg_weights,
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
            act_rotation_count=state.act_rotation_count + act_rotation,
            agg_rotation_count=state.agg_rotation_count + agg_rotation,
            fitness_history=fitness_history,
        )

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': bool(act_rotation),
            'agg_palette_changed': bool(agg_rotation),
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
            'sin_fatigue': float(new_act_fatigue[SIN_IDX]),
            'avg_act_fatigue': float(jnp.mean(new_act_fatigue)),
            'avg_agg_fatigue': float(jnp.mean(new_agg_fatigue)),
        }

        return new_state, metrics

    def get_state_summary(self, state: SynapticFatigueSymmetricState) -> Dict[str, Any]:
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
            'act_rotations': state.act_rotation_count,
            'agg_rotations': state.agg_rotation_count,
        }
