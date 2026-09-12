"""Strategy 31S: Morphogen Gradient Symmetric (Protected Concentration Floors).

Extends MorphogenGradientDualStrategy to symmetric mode with winning patterns:
- Protected indices have concentration floor preventing deactivation
- Affinity floors for sin and extreme aggregations
- Memory cell crystallization for sustained performance
- Cross-domain morphogen coupling

Key symmetric mechanisms:
1. Concentration floor - sin/max/min always have minimum concentration
2. Spatial organization with protected positions
3. Cross-domain gradient influence
4. Memory cells crystallize high-affinity functions

Developmental Insight: Like how essential morphogens (e.g., Shh, BMP) maintain
minimum expression levels to ensure proper patterning, protected indices
maintain baseline concentration regardless of gradient dynamics.
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
class MorphogenGradientSymmetricState:
    """State for morphogen gradient symmetric strategy."""
    # Palette masks
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    # Morphogen tracking
    source_positions: jnp.ndarray  # (n_sources, 2)
    source_strengths: jnp.ndarray  # (n_sources,)
    source_velocities: jnp.ndarray  # (n_sources, 2)
    act_concentrations: jnp.ndarray
    agg_concentrations: jnp.ndarray
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


class MorphogenGradientSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with morphogen gradients and protection.

    Applies winning patterns:
    - Protected indices have concentration floor
    - Affinity floors (sin=0.6, extreme_agg=0.5)
    - Memory cell crystallization (8 gens sustained)
    - Cross-domain morphogen coupling
    """

    name = "morphogen_gradient_symmetric"
    description = "Symmetric: Morphogen gradients with protected concentration floors"

    # Activation positions
    ACT_POSITIONS = {
        0: (0.2, 0.8), 1: (0.3, 0.7), 2: (0.1, 0.7), 3: (0.2, 0.6),
        4: (0.8, 0.2), 5: (0.3, 0.6), 6: (0.1, 0.6), 7: (0.2, 0.2),
        8: (0.5, 0.5), 9: (0.6, 0.5), 10: (0.5, 0.4), 11: (0.7, 0.3),
        12: (0.9, 0.3), 13: (0.8, 0.4), 14: (0.3, 0.3), 15: (0.7, 0.2),
        16: (0.1, 0.3), 17: (0.2, 0.4),
    }

    # Aggregation positions
    AGG_POSITIONS = {
        0: (0.5, 0.8),  # sum
        1: (0.4, 0.7),  # mean
        2: (0.7, 0.6),  # max
        3: (0.8, 0.5),  # min
        4: (0.6, 0.3),  # product
        5: (0.9, 0.4),  # maxabs
    }

    def __init__(
        self,
        # Sources
        n_sources: int = 3,
        # Gradient parameters
        gradient_decay: float = 3.0,
        concentration_threshold: float = 0.35,
        protected_concentration_floor: float = 0.7,
        # Source dynamics
        source_learning_rate: float = 0.08,
        source_momentum: float = 0.7,
        # Protection parameters (winning patterns)
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        protected_deactivation_prob: float = 0.001,
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Cross-domain
        cross_source_influence: float = 0.8,
        # Palette composition
        max_act_palette: int = 8,
        min_act_palette: int = 3,
        max_agg_palette: int = 4,
        min_agg_palette: int = 2,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Morphogen Gradient Symmetric strategy."""
        self.n_sources = n_sources
        self.gradient_decay = gradient_decay
        self.concentration_threshold = concentration_threshold
        self.protected_concentration_floor = protected_concentration_floor
        self.source_learning_rate = source_learning_rate
        self.source_momentum = source_momentum
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.protected_deactivation_prob = protected_deactivation_prob
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count
        self.cross_source_influence = cross_source_influence
        self.max_act_palette = max_act_palette
        self.min_act_palette = min_act_palette
        self.max_agg_palette = max_agg_palette
        self.min_agg_palette = min_agg_palette

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

    def _compute_concentrations(
        self,
        source_positions: jnp.ndarray,
        source_strengths: jnp.ndarray,
        function_positions: Dict[int, Tuple[float, float]],
        protected_indices: List[int],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute morphogen concentrations at each function position."""
        concentrations = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            if i in function_positions:
                pos = function_positions[i]
                total_conc = 0.0

                for s in range(self.n_sources):
                    src_pos = source_positions[s]
                    dist = jnp.sqrt((pos[0] - src_pos[0])**2 + (pos[1] - src_pos[1])**2)
                    conc = float(source_strengths[s]) * jnp.exp(-self.gradient_decay * dist)
                    total_conc += conc

                concentrations = concentrations.at[i].set(total_conc)

        # Protected indices get concentration floor
        for idx in protected_indices:
            if 0 <= idx < n_funcs:
                concentrations = concentrations.at[idx].set(
                    jnp.maximum(concentrations[idx], self.protected_concentration_floor)
                )

        return jnp.clip(concentrations, 0.0, 2.0)

    def initialize(self, config: Dict[str, Any], seed: int) -> MorphogenGradientSymmetricState:
        """Initialize state with morphogen tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 313100)
        key, k_pos = jax.random.split(key)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize sources near protected function positions
        sin_pos = self.ACT_POSITIONS[SIN_IDX]
        max_pos = self.AGG_POSITIONS[MAX_IDX]
        min_pos = self.AGG_POSITIONS[MIN_IDX]

        source_positions = jnp.array([
            sin_pos,
            max_pos,
            min_pos,
        ])[:self.n_sources]

        # Add noise
        noise = jax.random.uniform(k_pos, (self.n_sources, 2)) * 0.1 - 0.05
        source_positions = jnp.clip(source_positions + noise, 0.05, 0.95)

        source_strengths = jnp.ones(self.n_sources)
        source_velocities = jnp.zeros((self.n_sources, 2))

        # Compute initial concentrations
        act_concentrations = self._compute_concentrations(
            source_positions, source_strengths, self.ACT_POSITIONS, [SIN_IDX], NUM_ACTIVATIONS
        )
        agg_concentrations = self._compute_concentrations(
            source_positions, source_strengths, self.AGG_POSITIONS, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS
        )

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

        return MorphogenGradientSymmetricState(
            act_mask=act_mask,
            agg_mask=agg_mask,
            source_positions=source_positions,
            source_strengths=source_strengths,
            source_velocities=source_velocities,
            act_concentrations=act_concentrations,
            agg_concentrations=agg_concentrations,
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

    def get_active_palette(self, state: MorphogenGradientSymmetricState) -> List[int]:
        return mask_to_indices(state.act_mask)

    def get_active_agg_palette(self, state: MorphogenGradientSymmetricState) -> List[int]:
        return mask_to_indices(state.agg_mask)

    def _update_sources(
        self,
        source_positions: jnp.ndarray,
        source_strengths: jnp.ndarray,
        source_velocities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update source positions based on active function success."""
        new_positions = source_positions.copy()
        new_strengths = source_strengths.copy()
        new_velocities = source_velocities.copy()

        # Move sources toward successful functions
        if improvement > 0:
            for s in range(self.n_sources):
                # Find closest active function
                src_pos = source_positions[s]
                best_target = None
                best_dist = float('inf')

                # Check activations
                for i in range(NUM_ACTIVATIONS):
                    if act_mask[i] > 0.5 and i in self.ACT_POSITIONS:
                        pos = self.ACT_POSITIONS[i]
                        dist = jnp.sqrt((pos[0] - src_pos[0])**2 + (pos[1] - src_pos[1])**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_target = pos

                if best_target is not None:
                    direction = jnp.array(best_target) - src_pos
                    velocity = self.source_momentum * source_velocities[s] + \
                               self.source_learning_rate * improvement * direction
                    new_velocities = new_velocities.at[s].set(velocity)
                    new_pos = src_pos + velocity
                    new_positions = new_positions.at[s].set(jnp.clip(new_pos, 0.05, 0.95))

                # Boost strength on success
                new_strengths = new_strengths.at[s].set(
                    jnp.clip(source_strengths[s] * 1.05, 0.3, 2.0)
                )
        else:
            # Decay strength on failure
            new_strengths = new_strengths * 0.98

        return new_positions, new_strengths, new_velocities

    def _select_palette_by_concentration(
        self,
        concentrations: jnp.ndarray,
        affinity: jnp.ndarray,
        max_palette: int,
        min_palette: int,
        protected_indices: List[int],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette based on concentration and affinity."""
        n_funcs = len(concentrations)

        # Combine concentration and affinity
        scores = 0.5 * concentrations + 0.5 * affinity

        # Above threshold
        above_threshold = scores >= self.concentration_threshold
        n_above = int(jnp.sum(above_threshold))

        if n_above >= min_palette and n_above <= max_palette:
            mask = above_threshold.astype(jnp.float32)
        elif n_above < min_palette:
            top_k = jnp.argsort(scores)[-min_palette:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            top_k = jnp.argsort(scores)[-max_palette:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
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
        state: MorphogenGradientSymmetricState,
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[MorphogenGradientSymmetricState, Dict[str, Any]]:
        """Update with morphogen dynamics and protection."""
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

        # Update sources
        new_positions, new_strengths, new_velocities = self._update_sources(
            state.source_positions, state.source_strengths, state.source_velocities,
            state.act_mask, state.agg_mask, improvement
        )

        # Compute new concentrations
        new_act_conc = self._compute_concentrations(
            new_positions, new_strengths, self.ACT_POSITIONS, [SIN_IDX], NUM_ACTIVATIONS
        )
        new_agg_conc = self._compute_concentrations(
            new_positions, new_strengths * self.cross_source_influence,
            self.AGG_POSITIONS, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS
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
        new_cross = state.cross_affinity + 0.1 * improvement * co_active
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
        new_act_mask = self._select_palette_by_concentration(
            new_act_conc, new_act_affinity,
            self.max_act_palette, self.min_act_palette, [SIN_IDX], k_act
        )
        new_agg_mask = self._select_palette_by_concentration(
            new_agg_conc, new_agg_affinity,
            self.max_agg_palette, self.min_agg_palette, [MAX_IDX, MIN_IDX], k_agg
        )

        # Memory cells always active
        new_act_mask = jnp.where(new_act_mem_cells, 1.0, new_act_mask)
        new_agg_mask = jnp.where(new_agg_mem_cells, 1.0, new_agg_mask)

        fitness_history = state.fitness_history + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = MorphogenGradientSymmetricState(
            act_mask=new_act_mask,
            agg_mask=new_agg_mask,
            source_positions=new_positions,
            source_strengths=new_strengths,
            source_velocities=new_velocities,
            act_concentrations=new_act_conc,
            agg_concentrations=new_agg_conc,
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
            'sin_concentration': float(new_act_conc[SIN_IDX]),
        }

        return new_state, metrics

    def get_state_summary(self, state: MorphogenGradientSymmetricState) -> Dict[str, Any]:
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
