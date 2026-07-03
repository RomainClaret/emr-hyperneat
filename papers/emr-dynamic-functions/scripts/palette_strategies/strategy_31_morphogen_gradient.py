"""Strategy 31: Morphogen Gradient (Developmental Spatial Fields).

Implements morphogen-based developmental patterning. Functions exist in a
virtual 2D space, and morphogen sources diffuse gradients that activate
nearby functions. Source positions evolve toward successful function clusters.

Biological Basis:
- Morphogens are signaling molecules in embryonic development
- They diffuse from sources, creating concentration gradients
- Position along gradient determines cell fate
- Examples: Sonic hedgehog, BMPs, Wnts in embryogenesis

Key Insight:
- Functions can be organized in virtual spatial fields
- Functions "close together" in morphogen space co-activate
- This creates emergent functional groupings
- Gradient sources evolve to find optimal function clusters

Morphogen Mechanism:
    # Compute morphogen concentration at each function position
    for function i at position p:
        concentration[i] = sum(
            strength[s] * exp(-decay * distance(p, source[s]))
            for s in sources
        )

    # Activate functions above threshold
    active_mask = concentration > threshold

    # Evolve: Move sources toward high-fitness function clusters
    for source s:
        gradient = weighted_direction_toward(successful_functions)
        source_positions[s] += learning_rate * gradient

Expected improvements:
- Emergent functional organization (similar functions cluster)
- Smooth activation transitions (gradient-based)
- Evolvable spatial structure (sources move)
- Natural neighborhood effects (nearby functions co-activate)
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


class MorphogenGradientStrategy(PaletteEvolutionStrategy):
    """Developmental organization via diffusion-like spatial fields.

    Functions are placed in a 2D virtual space. Morphogen sources diffuse
    concentration gradients. Functions activate when concentration exceeds
    threshold. Sources evolve toward successful function clusters.
    """

    name = "morphogen_gradient"
    description = "Developmental morphogen gradients for spatial function organization"

    # Default function positions in 2D space (organized by similarity)
    DEFAULT_POSITIONS = {
        # Monotonic cluster (top-left)
        0: (0.2, 0.8),   # identity
        1: (0.3, 0.7),   # tanh
        2: (0.1, 0.7),   # sigmoid
        3: (0.2, 0.6),   # relu
        5: (0.3, 0.6),   # step
        6: (0.1, 0.6),   # leaky_relu
        # Oscillatory cluster (bottom-right)
        4: (0.8, 0.2),   # sin
        11: (0.7, 0.3),  # burst
        12: (0.9, 0.3),  # resonator
        13: (0.8, 0.4),  # osc_adapt
        15: (0.7, 0.2),  # receptive
        # Spatial cluster (bottom-left)
        7: (0.2, 0.2),   # gaussian
        14: (0.3, 0.3),  # locality
        16: (0.1, 0.3),  # spatial_decay
        17: (0.2, 0.4),  # edge_detector
        # Nonlinear cluster (center)
        8: (0.5, 0.5),   # softplus
        9: (0.6, 0.5),   # elu
        10: (0.5, 0.4),  # swish
    }

    def __init__(
        self,
        # Spatial configuration
        function_positions: Dict[int, Tuple[float, float]] = None,
        n_sources: int = 3,
        # Gradient parameters
        gradient_decay: float = 3.0,          # Exponential decay with distance
        concentration_threshold: float = 0.35, # Min concentration to activate
        # Source dynamics
        source_learning_rate: float = 0.08,
        source_momentum: float = 0.7,
        source_position_decay: float = 0.02,  # Slight pull toward center
        # Morphogen strengths
        initial_strength: float = 1.0,
        strength_learning_rate: float = 0.05,
        strength_decay: float = 0.98,
        strength_min: float = 0.3,
        strength_max: float = 2.0,
        # Palette composition
        max_palette_size: int = 8,
        min_palette_size: int = 3,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Morphogen Gradient strategy.

        Args:
            function_positions: Map function index -> (x, y) position
            n_sources: Number of morphogen sources
            gradient_decay: How fast concentration drops with distance
            concentration_threshold: Minimum for activation
            source_learning_rate: How fast sources move
            source_momentum: Momentum for source movement
            source_position_decay: Pull toward center (prevents drift)
            initial_strength: Starting morphogen strength
            strength_learning_rate: How fast strengths adapt
            strength_decay: Passive strength decay
            strength_min: Minimum source strength
            strength_max: Maximum source strength
            max_palette_size: Upper limit on active functions
            min_palette_size: Lower limit on active functions
        """
        # Spatial
        self.function_positions = function_positions or self.DEFAULT_POSITIONS
        self.n_sources = n_sources

        # Gradient
        self.gradient_decay = gradient_decay
        self.concentration_threshold = concentration_threshold

        # Source dynamics
        self.source_learning_rate = source_learning_rate
        self.source_momentum = source_momentum
        self.source_position_decay = source_position_decay

        # Strengths
        self.initial_strength = initial_strength
        self.strength_learning_rate = strength_learning_rate
        self.strength_decay = strength_decay
        self.strength_min = strength_min
        self.strength_max = strength_max

        # Palette
        self.max_palette_size = max_palette_size
        self.min_palette_size = min_palette_size

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

        # Build position array
        self._build_position_array()

    def _build_position_array(self):
        """Build position array for all functions."""
        self.positions = jnp.zeros((NUM_ACTIVATIONS, 2))
        for func_idx, pos in self.function_positions.items():
            if func_idx < NUM_ACTIVATIONS:
                self.positions = self.positions.at[func_idx, 0].set(pos[0])
                self.positions = self.positions.at[func_idx, 1].set(pos[1])

    def _initialize_sources(self, key: jax.random.PRNGKey, initial: List[int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize source positions near initial palette functions."""
        positions = jnp.zeros((self.n_sources, 2))
        strengths = jnp.ones(self.n_sources) * self.initial_strength

        if initial:
            # Place sources near initial palette clusters
            initial_positions = jnp.array([
                self.function_positions.get(i, (0.5, 0.5))
                for i in initial if i < NUM_ACTIVATIONS
            ])

            if len(initial_positions) > 0:
                # Distribute sources near initial functions
                for s in range(self.n_sources):
                    idx = s % len(initial_positions)
                    # Add small random offset
                    key, subkey = jax.random.split(key)
                    offset = jax.random.uniform(subkey, (2,), minval=-0.1, maxval=0.1)
                    positions = positions.at[s].set(
                        jnp.clip(initial_positions[idx] + offset, 0.0, 1.0)
                    )
        else:
            # Random initialization
            key, subkey = jax.random.split(key)
            positions = jax.random.uniform(subkey, (self.n_sources, 2))

        return positions, strengths

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with morphogen sources."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        key = jax.random.PRNGKey(seed + 313131)
        key, subkey = jax.random.split(key)

        # Initialize sources
        source_positions, source_strengths = self._initialize_sources(subkey, initial)

        # Source velocities for momentum
        source_velocities = jnp.zeros((self.n_sources, 2))

        # Function success memory
        function_success = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                function_success = function_success.at[i].set(0.3)

        return {
            'mask': mask,
            'rng_key': key,
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Morphogen state
            'source_positions': source_positions,
            'source_strengths': source_strengths,
            'source_velocities': source_velocities,
            # Function state
            'function_success': function_success,
            'concentrations': jnp.zeros(NUM_ACTIVATIONS),
            # Tracking
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_concentrations(
        self,
        source_positions: jnp.ndarray,
        source_strengths: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute morphogen concentration at each function position.

        Concentration is sum of contributions from all sources,
        where each source contributes strength * exp(-decay * distance).
        """
        concentrations = jnp.zeros(NUM_ACTIVATIONS)

        for func_idx in range(NUM_ACTIVATIONS):
            func_pos = self.positions[func_idx]
            total_conc = 0.0

            for s in range(self.n_sources):
                source_pos = source_positions[s]
                strength = source_strengths[s]

                # Euclidean distance
                distance = jnp.sqrt(jnp.sum((func_pos - source_pos) ** 2))

                # Exponential decay with distance
                contribution = strength * jnp.exp(-self.gradient_decay * distance)
                total_conc += contribution

            concentrations = concentrations.at[func_idx].set(total_conc)

        return concentrations

    def _select_palette_from_concentrations(
        self,
        concentrations: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select palette based on morphogen concentrations."""
        # Functions above threshold
        above_threshold = concentrations >= self.concentration_threshold

        n_above = int(jnp.sum(above_threshold))

        if n_above >= self.min_palette_size and n_above <= self.max_palette_size:
            # Use threshold-based selection
            mask = above_threshold.astype(jnp.float32)
        elif n_above < self.min_palette_size:
            # Too few: take top by concentration
            top_k = jnp.argsort(concentrations)[-self.min_palette_size:]
            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            # Too many: take top by concentration
            top_k = jnp.argsort(concentrations)[-self.max_palette_size:]
            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)

        return mask

    def _update_function_success(
        self,
        success: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update success memory for active functions."""
        new_success = 0.9 * success

        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                new_success = new_success.at[i].add(max(0, improvement))

        return jnp.clip(new_success, 0.0, 1.0)

    def _compute_source_gradients(
        self,
        source_positions: jnp.ndarray,
        function_success: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute gradients for source movement toward successful functions."""
        gradients = jnp.zeros((self.n_sources, 2))

        for s in range(self.n_sources):
            source_pos = source_positions[s]
            total_weight = 0.0
            weighted_direction = jnp.zeros(2)

            for func_idx in range(NUM_ACTIVATIONS):
                # Weight by success and activity
                weight = float(function_success[func_idx])
                if mask[func_idx] > 0.5:
                    weight *= 2.0  # Boost active functions

                if weight > 0.01:
                    func_pos = self.positions[func_idx]
                    direction = func_pos - source_pos
                    distance = jnp.sqrt(jnp.sum(direction ** 2)) + 0.01

                    # Inverse distance weighting (closer = stronger pull)
                    weighted_direction += weight * direction / distance
                    total_weight += weight

            if total_weight > 0.01:
                gradients = gradients.at[s].set(weighted_direction / total_weight)

        return gradients

    def _update_sources(
        self,
        source_positions: jnp.ndarray,
        source_strengths: jnp.ndarray,
        source_velocities: jnp.ndarray,
        gradients: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update source positions and strengths."""
        # Update velocities with momentum
        new_velocities = (
            self.source_momentum * source_velocities +
            self.source_learning_rate * gradients
        )

        # Update positions
        new_positions = source_positions + new_velocities

        # Pull toward center (prevent drift to edges)
        center = jnp.array([0.5, 0.5])
        center_pull = self.source_position_decay * (center - new_positions)
        new_positions = new_positions + center_pull

        # Clip to valid range
        new_positions = jnp.clip(new_positions, 0.0, 1.0)

        # Update strengths based on improvement
        new_strengths = self.strength_decay * source_strengths
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
        """Update with morphogen gradient dynamics."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update function success memory
        new_success = self._update_function_success(
            state['function_success'],
            state['mask'],
            improvement,
        )

        # Step 2: Compute gradients for source movement
        gradients = self._compute_source_gradients(
            state['source_positions'],
            new_success,
            state['mask'],
        )

        # Step 3: Update sources
        new_positions, new_strengths, new_velocities = self._update_sources(
            state['source_positions'],
            state['source_strengths'],
            state['source_velocities'],
            gradients,
            improvement,
        )

        # Step 4: Compute new concentrations
        new_concentrations = self._compute_concentrations(
            new_positions,
            new_strengths,
        )

        # Step 5: Select palette from concentrations
        new_mask = self._select_palette_from_concentrations(new_concentrations)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track fitness
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Morphogen state
            'source_positions': new_positions,
            'source_strengths': new_strengths,
            'source_velocities': new_velocities,
            # Function state
            'function_success': new_success,
            'concentrations': new_concentrations,
            # Tracking
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Concentration stats
        active_concentrations = [float(new_concentrations[i]) for i in active_palette]

        # Source positions as list
        source_pos_list = [
            (float(new_positions[s, 0]), float(new_positions[s, 1]))
            for s in range(self.n_sources)
        ]

        # Top functions by concentration
        top_conc_idx = jnp.argsort(new_concentrations)[-5:][::-1]
        top_concentrations = [(int(i), float(new_concentrations[i])) for i in top_conc_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Concentration
            'mean_concentration': float(jnp.mean(new_concentrations)),
            'max_concentration': float(jnp.max(new_concentrations)),
            'active_mean_concentration': float(np.mean(active_concentrations)) if active_concentrations else 0.0,
            'top_concentrations': top_concentrations,
            # Sources
            'source_positions': source_pos_list,
            'source_strengths': [float(s) for s in new_strengths],
            'mean_source_velocity': float(jnp.mean(jnp.abs(new_velocities))),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_concentration': float(new_concentrations[4]),
            'sin_success': float(new_success[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with morphogen status."""
        palette = self.get_active_palette(state)
        concentrations = state['concentrations']
        positions = state['source_positions']
        strengths = state['source_strengths']

        # Top functions by concentration
        top_conc = jnp.argsort(concentrations)[-5:][::-1]
        top_functions = [(int(i), float(concentrations[i])) for i in top_conc]

        # Source info
        sources = [
            {
                'position': (float(positions[s, 0]), float(positions[s, 1])),
                'strength': float(strengths[s]),
            }
            for s in range(self.n_sources)
        ]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Concentration
            'top_concentrations': top_functions,
            'mean_concentration': float(jnp.mean(concentrations)),
            'threshold': self.concentration_threshold,
            # Sources
            'sources': sources,
            # Sin-specific
            'sin_concentration': float(concentrations[4]),
        }
