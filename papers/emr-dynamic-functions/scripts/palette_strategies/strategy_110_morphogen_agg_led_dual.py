"""Strategy 110: Morphogen + Aggregation-Led Dual.

Combines morphogen gradient spatial organization (strategy 31) with
aggregation-led dual discovery (strategy 101) for spatial co-location.

Key Innovation:
- Sin and extreme aggregations are placed NEARBY in virtual space
- Morphogen sources naturally co-activate spatially close functions
- Spatial proximity creates implicit sin-extreme co-activation
- Sources evolve toward successful clusters

Bio inspiration: Morphogen gradients organize cell fate in embryos.
Functions near the same source activate together. Sin and extremes
are positioned close together so they naturally co-activate.
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
    CORE_EXTREME_AGGS,
)


class MorphogenAggLedDualStrategy(PaletteEvolutionStrategy):
    """Spatial co-location for sin-extreme co-activation.

    Morphogen gradients co-activate spatially
    nearby functions. Sin and extremes are positioned together.
    """

    name = "morphogen_agg_led_dual"
    description = "Dual: Morphogen gradients co-activate sin-extreme via spatial proximity"

    # Function positions in 2D virtual space
    # KEY: Sin and extremes are clustered together!
    FUNCTION_POSITIONS = {
        # Monotonic cluster (top-left)
        0: (0.2, 0.8),   # identity
        1: (0.3, 0.7),   # tanh
        2: (0.1, 0.7),   # sigmoid
        3: (0.2, 0.6),   # relu
        # SIN-EXTREME CLUSTER (bottom-right) - KEY INNOVATION
        4: (0.8, 0.2),   # sin - near extremes!
        # Other functions
        5: (0.3, 0.5),   # step
        6: (0.1, 0.5),   # leaky_relu
        7: (0.5, 0.5),   # gaussian
        8: (0.6, 0.6),   # softplus
        9: (0.4, 0.4),   # elu
        10: (0.5, 0.3),  # swish
        11: (0.7, 0.4),  # burst
        12: (0.6, 0.2),  # resonator
    }

    # Aggregation positions - extremes near sin!
    AGG_POSITIONS = {
        0: (0.3, 0.6),   # sum
        1: (0.4, 0.7),   # mean
        2: (0.7, 0.3),   # max - near sin!
        3: (0.9, 0.3),   # min - near sin!
        4: (0.4, 0.5),   # prod
        5: (0.5, 0.6),   # median
    }

    def __init__(
        self,
        # === SPATIAL CONFIGURATION (KEY INNOVATION) ===
        sin_position: Tuple[float, float] = (0.8, 0.2),
        max_position: Tuple[float, float] = (0.7, 0.3),
        min_position: Tuple[float, float] = (0.9, 0.3),
        n_sources: int = 3,
        # === Gradient parameters ===
        gradient_decay: float = 3.0,
        concentration_threshold: float = 0.35,
        # === Source dynamics ===
        source_learning_rate: float = 0.10,
        source_momentum: float = 0.7,
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Exploration ===
        exploration_rate: float = 0.20,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Morphogen Aggregation-Led Dual strategy."""
        # Update positions with custom sin/extreme locations
        self.function_positions = dict(self.FUNCTION_POSITIONS)
        self.function_positions[4] = sin_position
        self.agg_positions = dict(self.AGG_POSITIONS)
        self.agg_positions[2] = max_position
        self.agg_positions[3] = min_position

        self.n_sources = n_sources
        self.gradient_decay = gradient_decay
        self.concentration_threshold = concentration_threshold
        self.source_learning_rate = source_learning_rate
        self.source_momentum = source_momentum

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Exploration
        self.exploration_rate = exploration_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

        # Build position arrays
        self._build_position_arrays()

    def _build_position_arrays(self):
        """Build position arrays for functions."""
        self.act_positions = jnp.zeros((NUM_ACTIVATIONS, 2))
        for i, pos in self.function_positions.items():
            if i < NUM_ACTIVATIONS:
                self.act_positions = self.act_positions.at[i, 0].set(pos[0])
                self.act_positions = self.act_positions.at[i, 1].set(pos[1])

        self.agg_pos_array = jnp.zeros((NUM_AGGREGATIONS, 2))
        for i, pos in self.agg_positions.items():
            if i < NUM_AGGREGATIONS:
                self.agg_pos_array = self.agg_pos_array.at[i, 0].set(pos[0])
                self.agg_pos_array = self.agg_pos_array.at[i, 1].set(pos[1])

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with morphogen sources."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        key = jax.random.PRNGKey(seed + 1100000)
        key, k1 = jax.random.split(key)

        # Initialize sources - one near sin-extreme cluster
        source_positions = jnp.zeros((self.n_sources, 2))
        source_strengths = jnp.ones(self.n_sources)
        source_velocities = jnp.zeros((self.n_sources, 2))

        # Place first source near sin-extreme cluster
        sin_pos = jnp.array(self.function_positions.get(4, (0.8, 0.2)))
        source_positions = source_positions.at[0].set(sin_pos)

        # Other sources random
        for s in range(1, self.n_sources):
            key, subkey = jax.random.split(key)
            pos = jax.random.uniform(subkey, (2,))
            source_positions = source_positions.at[s].set(pos)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Morphogen state
            'source_positions': source_positions,
            'source_strengths': source_strengths,
            'source_velocities': source_velocities,
            # Concentrations
            'act_concentrations': jnp.zeros(NUM_ACTIVATIONS),
            'agg_concentrations': jnp.zeros(NUM_AGGREGATIONS),
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Success memory
            'act_success': jnp.zeros(NUM_ACTIVATIONS),
            'agg_success': jnp.zeros(NUM_AGGREGATIONS),
            # Stats
            'sin_extreme_co_activation': 0,
            # General
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

    def _compute_concentrations(
        self,
        source_positions: jnp.ndarray,
        source_strengths: jnp.ndarray,
        func_positions: jnp.ndarray,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute morphogen concentration at each function position."""
        concentrations = jnp.zeros(n_funcs)

        for func_idx in range(n_funcs):
            func_pos = func_positions[func_idx]
            total_conc = 0.0

            for s in range(self.n_sources):
                source_pos = source_positions[s]
                strength = source_strengths[s]
                distance = jnp.sqrt(jnp.sum((func_pos - source_pos) ** 2))
                contribution = strength * jnp.exp(-self.gradient_decay * distance)
                total_conc += contribution

            concentrations = concentrations.at[func_idx].set(total_conc)

        return concentrations

    def _update_sources(
        self,
        source_positions: jnp.ndarray,
        source_strengths: jnp.ndarray,
        source_velocities: jnp.ndarray,
        act_success: jnp.ndarray,
        agg_success: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update source positions toward successful function clusters."""
        gradients = jnp.zeros((self.n_sources, 2))

        for s in range(self.n_sources):
            source_pos = source_positions[s]
            weighted_direction = jnp.zeros(2)
            total_weight = 0.0

            # Pull toward successful activations
            for func_idx in range(NUM_ACTIVATIONS):
                weight = float(act_success[func_idx])
                if act_mask[func_idx] > 0.5:
                    weight *= 2.0
                if weight > 0.01:
                    func_pos = self.act_positions[func_idx]
                    direction = func_pos - source_pos
                    distance = jnp.sqrt(jnp.sum(direction ** 2)) + 0.01
                    weighted_direction += weight * direction / distance
                    total_weight += weight

            # Pull toward successful aggregations (in same space)
            for agg_idx in range(NUM_AGGREGATIONS):
                weight = float(agg_success[agg_idx])
                if agg_mask[agg_idx] > 0.5:
                    weight *= 2.0
                if weight > 0.01:
                    agg_pos = self.agg_pos_array[agg_idx]
                    direction = agg_pos - source_pos
                    distance = jnp.sqrt(jnp.sum(direction ** 2)) + 0.01
                    weighted_direction += weight * direction / distance
                    total_weight += weight

            if total_weight > 0.01:
                gradients = gradients.at[s].set(weighted_direction / total_weight)

        # Update velocities with momentum
        new_velocities = self.source_momentum * source_velocities + self.source_learning_rate * gradients
        new_positions = jnp.clip(source_positions + new_velocities, 0.0, 1.0)

        # Update strengths
        new_strengths = source_strengths * 0.98
        if improvement > 0:
            new_strengths = new_strengths + 0.05 * improvement
        new_strengths = jnp.clip(new_strengths, 0.3, 2.0)

        return new_positions, new_strengths, new_velocities

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        concentrations: jnp.ndarray,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        stagnation: int,
        generation: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette based on concentrations + affinity."""
        k1, k2 = jax.random.split(key)

        # Score: concentration + affinity
        score = concentrations + affinities * 0.5

        # Preference boost
        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + 0.6)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Exploration
        exploration_prob = self.exploration_rate * (1 + stagnation * 0.1)
        if prefer_indices:
            for idx in prefer_indices:
                if 0 <= idx < n_funcs and mask[idx] < 0.5:
                    if float(jax.random.uniform(k1)) < exploration_prob * 1.5:
                        mask = mask.at[idx].set(1.0)
                    k1, _ = jax.random.split(k1)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < min_diversity:
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k2, jnp.array(inactive),
                                          shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with morphogen gradient dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === SUCCESS MEMORY ===
        new_act_success = 0.9 * state['act_success']
        new_agg_success = 0.9 * state['agg_success']
        if improvement > 0:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    new_act_success = new_act_success.at[i].add(improvement)
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    new_agg_success = new_agg_success.at[j].add(improvement)
        new_act_success = jnp.clip(new_act_success, 0.0, 1.0)
        new_agg_success = jnp.clip(new_agg_success, 0.0, 1.0)

        # === UPDATE SOURCES ===
        new_positions, new_strengths, new_velocities = self._update_sources(
            state['source_positions'], state['source_strengths'],
            state['source_velocities'], new_act_success, new_agg_success,
            state['act_mask'], state['agg_mask'], improvement
        )

        # === COMPUTE CONCENTRATIONS ===
        new_act_conc = self._compute_concentrations(
            new_positions, new_strengths, self.act_positions, NUM_ACTIVATIONS
        )
        new_agg_conc = self._compute_concentrations(
            new_positions, new_strengths, self.agg_pos_array, NUM_AGGREGATIONS
        )

        # === AFFINITY UPDATE ===
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay
        if improvement > 0:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * improvement)
                    )
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * improvement)
                    )

        # === PALETTE SELECTION ===
        new_act_mask, _ = self._select_palette(
            new_act_aff, new_act_conc, NUM_ACTIVATIONS,
            self.min_active_act, self.max_active_act, self.min_diversity_act,
            new_stagnation, generation, k1, prefer_indices=[4]
        )
        new_agg_mask, _ = self._select_palette(
            new_agg_aff, new_agg_conc, NUM_AGGREGATIONS,
            self.min_active_agg, self.max_active_agg, self.min_diversity_agg,
            new_stagnation, generation, k2, prefer_indices=list(CORE_EXTREME_AGGS)
        )

        # Check sin-extreme co-activation
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)
        co_activation = 1 if (4 in act_palette and any(j in agg_palette for j in CORE_EXTREME_AGGS)) else 0

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'source_positions': new_positions,
            'source_strengths': new_strengths,
            'source_velocities': new_velocities,
            'act_concentrations': new_act_conc,
            'agg_concentrations': new_agg_conc,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_success': new_act_success,
            'agg_success': new_agg_success,
            'sin_extreme_co_activation': state['sin_extreme_co_activation'] + co_activation,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Morphogen metrics (KEY)
            'sin_concentration': float(new_act_conc[4]),
            'max_concentration': float(new_agg_conc[2]),
            'min_concentration': float(new_agg_conc[3]),
            'sin_extreme_co_activation': new_state['sin_extreme_co_activation'],
            'mean_source_strength': float(jnp.mean(new_strengths)),
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'sin_concentration': float(state['act_concentrations'][4]),
            'sin_extreme_co_activation': state['sin_extreme_co_activation'],
            'generation': state['generation'],
        }
