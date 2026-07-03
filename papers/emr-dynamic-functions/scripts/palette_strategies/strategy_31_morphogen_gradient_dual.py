"""Strategy 31D: Morphogen Gradient Dual (Spatial Fields for Both Palettes).

Extends Morphogen Gradient to jointly evolve activation AND aggregation
function palettes using diffusion-based spatial organization for both domains.

Key mechanisms:
1. Functions in 2D space for both domains
2. Morphogen sources diffuse gradients activating nearby functions
3. Sources evolve toward successful function clusters
4. Cross-domain: Shared sources influence both domains with different strengths

Developmental basis:
- Morphogens create concentration gradients in embryos
- Position along gradient determines cell fate (function activation)
- Spatial organization emerges from gradient interactions
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

NUM_AGGREGATIONS = 6
DEFAULT_AGGREGATION_INDICES = [0, 1]


def create_initial_agg_mask(indices: List[int]) -> jnp.ndarray:
    mask = jnp.zeros(NUM_AGGREGATIONS)
    for idx in indices:
        if 0 <= idx < NUM_AGGREGATIONS:
            mask = mask.at[idx].set(1.0)
    return mask


def agg_mask_to_indices(mask: jnp.ndarray) -> List[int]:
    return [int(i) for i in range(NUM_AGGREGATIONS) if mask[i] > 0.5]


class MorphogenGradientDualStrategy(PaletteEvolutionStrategy):
    """Morphogen gradient organization for both activation and aggregation palettes."""

    name = "morphogen_gradient_dual"
    description = "Morphogen gradients for dual palette spatial organization"

    # Activation positions (from original)
    ACT_POSITIONS = {
        0: (0.2, 0.8), 1: (0.3, 0.7), 2: (0.1, 0.7), 3: (0.2, 0.6),
        4: (0.8, 0.2), 5: (0.3, 0.6), 6: (0.1, 0.6), 7: (0.2, 0.2),
        8: (0.5, 0.5), 9: (0.6, 0.5), 10: (0.5, 0.4), 11: (0.7, 0.3),
        12: (0.9, 0.3), 13: (0.8, 0.4), 14: (0.3, 0.3), 15: (0.7, 0.2),
        16: (0.1, 0.3), 17: (0.2, 0.4),
    }

    # Aggregation positions in same space
    AGG_POSITIONS = {
        0: (0.5, 0.8),  # sum - central, versatile
        1: (0.4, 0.7),  # mean - near sum
        2: (0.7, 0.6),  # max - toward specialization
        3: (0.8, 0.5),  # min - specialist
        4: (0.6, 0.3),  # product - nonlinear area
        5: (0.9, 0.4),  # maxabs - edge specialist
    }

    def __init__(
        self,
        # Sources
        n_sources: int = 3,
        # Gradient parameters
        act_gradient_decay: float = 3.0,
        agg_gradient_decay: float = 3.5,  # Similar decay to activation (was 4.0)
        act_concentration_threshold: float = 0.35,
        agg_concentration_threshold: float = 0.30,  # Lower threshold for agg (was 0.40)
        # Source dynamics
        source_learning_rate: float = 0.08,
        source_momentum: float = 0.7,
        source_position_decay: float = 0.02,
        # Strengths
        initial_strength: float = 1.0,
        strength_learning_rate: float = 0.05,
        strength_decay: float = 0.98,
        strength_min: float = 0.3,
        strength_max: float = 2.0,
        # Cross-domain: sources affect both with different weights
        cross_source_influence: float = 0.8,  # How much sources affect agg (was 0.3)
        # Palette limits
        max_act_palette: int = 8,
        min_act_palette: int = 3,
        max_agg_palette: int = 4,
        min_agg_palette: int = 2,
        # General
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.n_sources = n_sources

        self.act_gradient_decay = act_gradient_decay
        self.agg_gradient_decay = agg_gradient_decay
        self.act_concentration_threshold = act_concentration_threshold
        self.agg_concentration_threshold = agg_concentration_threshold

        self.source_learning_rate = source_learning_rate
        self.source_momentum = source_momentum
        self.source_position_decay = source_position_decay

        self.initial_strength = initial_strength
        self.strength_learning_rate = strength_learning_rate
        self.strength_decay = strength_decay
        self.strength_min = strength_min
        self.strength_max = strength_max

        self.cross_source_influence = cross_source_influence

        self.max_act_palette = max_act_palette
        self.min_act_palette = min_act_palette
        self.max_agg_palette = max_agg_palette
        self.min_agg_palette = min_agg_palette

        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

        self._build_position_arrays()

    def _build_position_arrays(self):
        """Build position arrays for both domains."""
        self.act_positions = jnp.zeros((NUM_ACTIVATIONS, 2))
        for i, pos in self.ACT_POSITIONS.items():
            if i < NUM_ACTIVATIONS:
                self.act_positions = self.act_positions.at[i, 0].set(pos[0])
                self.act_positions = self.act_positions.at[i, 1].set(pos[1])

        self.agg_positions = jnp.zeros((NUM_AGGREGATIONS, 2))
        for i, pos in self.AGG_POSITIONS.items():
            if i < NUM_AGGREGATIONS:
                self.agg_positions = self.agg_positions.at[i, 0].set(pos[0])
                self.agg_positions = self.agg_positions.at[i, 1].set(pos[1])

    def _initialize_sources(self, key, initial_act, initial_agg):
        """Initialize sources near initial functions."""
        positions = jnp.zeros((self.n_sources, 2))
        strengths = jnp.ones(self.n_sources) * self.initial_strength

        all_initial_pos = []
        for i in initial_act:
            if i < NUM_ACTIVATIONS:
                all_initial_pos.append(self.ACT_POSITIONS.get(i, (0.5, 0.5)))
        for i in initial_agg:
            if i < NUM_AGGREGATIONS:
                all_initial_pos.append(self.AGG_POSITIONS.get(i, (0.5, 0.5)))

        if all_initial_pos:
            all_initial_pos = jnp.array(all_initial_pos)
            for s in range(self.n_sources):
                idx = s % len(all_initial_pos)
                key, subkey = jax.random.split(key)
                offset = jax.random.uniform(subkey, (2,), minval=-0.1, maxval=0.1)
                positions = positions.at[s].set(
                    jnp.clip(all_initial_pos[idx] + offset, 0.0, 1.0)
                )
        else:
            key, subkey = jax.random.split(key)
            positions = jax.random.uniform(subkey, (self.n_sources, 2))

        return positions, strengths, key

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_mask(initial_agg)

        key = jax.random.PRNGKey(seed + 313131)
        source_positions, source_strengths, key = self._initialize_sources(
            key, initial_act, initial_agg
        )
        source_velocities = jnp.zeros((self.n_sources, 2))

        act_success = jnp.zeros(NUM_ACTIVATIONS)
        agg_success = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_success = act_success.at[i].set(0.3)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_success = agg_success.at[i].set(0.3)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'source_positions': source_positions,
            'source_strengths': source_strengths,
            'source_velocities': source_velocities,
            'act_success': act_success,
            'agg_success': agg_success,
            'act_concentrations': jnp.zeros(NUM_ACTIVATIONS),
            'agg_concentrations': jnp.zeros(NUM_AGGREGATIONS),
            'rng_key': key,
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'mask': act_mask,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return agg_mask_to_indices(state['agg_mask'])

    def _compute_concentrations(
        self,
        source_positions: jnp.ndarray,
        source_strengths: jnp.ndarray,
        func_positions: jnp.ndarray,
        gradient_decay: float,
        cross_influence: float = 1.0,
    ) -> jnp.ndarray:
        """Compute morphogen concentration at each function position."""
        n_funcs = func_positions.shape[0]
        concentrations = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            func_pos = func_positions[i]
            total = 0.0

            for s in range(self.n_sources):
                source_pos = source_positions[s]
                strength = source_strengths[s] * cross_influence
                distance = jnp.sqrt(jnp.sum((func_pos - source_pos) ** 2))
                total += strength * jnp.exp(-gradient_decay * distance)

            concentrations = concentrations.at[i].set(total)

        return concentrations

    def _select_palette(
        self,
        concentrations: jnp.ndarray,
        threshold: float,
        min_size: int,
        max_size: int,
    ) -> jnp.ndarray:
        """Select palette based on concentrations."""
        n_funcs = concentrations.shape[0]
        above = concentrations >= threshold
        n_above = int(jnp.sum(above))

        if min_size <= n_above <= max_size:
            return above.astype(jnp.float32)
        elif n_above < min_size:
            top_k = jnp.argsort(concentrations)[-min_size:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
            return mask
        else:
            top_k = jnp.argsort(concentrations)[-max_size:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
            return mask

    def _update_success(
        self,
        success: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update success memory."""
        new_success = 0.9 * success
        n_funcs = success.shape[0]
        for i in range(n_funcs):
            if mask[i] > 0.5:
                new_success = new_success.at[i].add(max(0, improvement))
        return jnp.clip(new_success, 0.0, 1.0)

    def _compute_source_gradients(
        self,
        source_positions: jnp.ndarray,
        act_success: jnp.ndarray,
        agg_success: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute gradients for source movement toward successful functions."""
        gradients = jnp.zeros((self.n_sources, 2))

        for s in range(self.n_sources):
            source_pos = source_positions[s]
            total_weight = 0.0
            weighted_dir = jnp.zeros(2)

            # From activations
            for i in range(NUM_ACTIVATIONS):
                weight = float(act_success[i])
                if act_mask[i] > 0.5:
                    weight *= 2.0
                if weight > 0.01:
                    func_pos = self.act_positions[i]
                    direction = func_pos - source_pos
                    distance = jnp.sqrt(jnp.sum(direction ** 2)) + 0.01
                    weighted_dir += weight * direction / distance
                    total_weight += weight

            # From aggregations (with cross influence)
            for i in range(NUM_AGGREGATIONS):
                weight = float(agg_success[i]) * self.cross_source_influence
                if agg_mask[i] > 0.5:
                    weight *= 2.0
                if weight > 0.01:
                    func_pos = self.agg_positions[i]
                    direction = func_pos - source_pos
                    distance = jnp.sqrt(jnp.sum(direction ** 2)) + 0.01
                    weighted_dir += weight * direction / distance
                    total_weight += weight

            if total_weight > 0.01:
                gradients = gradients.at[s].set(weighted_dir / total_weight)

        return gradients

    def _update_sources(
        self,
        positions: jnp.ndarray,
        strengths: jnp.ndarray,
        velocities: jnp.ndarray,
        gradients: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update source positions and strengths."""
        new_velocities = self.source_momentum * velocities + self.source_learning_rate * gradients
        new_positions = positions + new_velocities

        center = jnp.array([0.5, 0.5])
        center_pull = self.source_position_decay * (center - new_positions)
        new_positions = jnp.clip(new_positions + center_pull, 0.0, 1.0)

        new_strengths = self.strength_decay * strengths
        if improvement > 0:
            new_strengths = new_strengths + self.strength_learning_rate * improvement
        new_strengths = jnp.clip(new_strengths, self.strength_min, self.strength_max)

        return new_positions, new_strengths, new_velocities

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key = state['rng_key']

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Update success
        new_act_success = self._update_success(state['act_success'], state['act_mask'], improvement)
        new_agg_success = self._update_success(state['agg_success'], state['agg_mask'], improvement)

        # Compute gradients
        gradients = self._compute_source_gradients(
            state['source_positions'], new_act_success, new_agg_success,
            state['act_mask'], state['agg_mask']
        )

        # Update sources
        new_positions, new_strengths, new_velocities = self._update_sources(
            state['source_positions'], state['source_strengths'],
            state['source_velocities'], gradients, improvement
        )

        # Compute concentrations for both domains
        new_act_conc = self._compute_concentrations(
            new_positions, new_strengths, self.act_positions, self.act_gradient_decay
        )
        new_agg_conc = self._compute_concentrations(
            new_positions, new_strengths, self.agg_positions,
            self.agg_gradient_decay, self.cross_source_influence
        )

        # Select palettes
        new_act_mask = self._select_palette(
            new_act_conc, self.act_concentration_threshold,
            self.min_act_palette, self.max_act_palette
        )
        new_agg_mask = self._select_palette(
            new_agg_conc, self.agg_concentration_threshold,
            self.min_agg_palette, self.max_agg_palette
        )

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'source_positions': new_positions,
            'source_strengths': new_strengths,
            'source_velocities': new_velocities,
            'act_success': new_act_success,
            'agg_success': new_agg_success,
            'act_concentrations': new_act_conc,
            'agg_concentrations': new_agg_conc,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'mask': new_act_mask,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = agg_mask_to_indices(new_agg_mask)

        source_pos_list = [
            (float(new_positions[s, 0]), float(new_positions[s, 1]))
            for s in range(self.n_sources)
        ]

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Concentration stats
            'act_mean_concentration': float(jnp.mean(new_act_conc)),
            'agg_mean_concentration': float(jnp.mean(new_agg_conc)),
            'sin_concentration': float(new_act_conc[4]),
            # Sources
            'source_positions': source_pos_list,
            'source_strengths': [float(s) for s in new_strengths],
            # Function status
            'has_sin': 4 in act_palette,
            'sin_success': float(new_act_success[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'generation': state['generation'],
            'act_mean_concentration': float(jnp.mean(state['act_concentrations'])),
            'agg_mean_concentration': float(jnp.mean(state['agg_concentrations'])),
            'sin_concentration': float(state['act_concentrations'][4]),
            'n_sources': self.n_sources,
            'source_strengths': [float(s) for s in state['source_strengths']],
        }
