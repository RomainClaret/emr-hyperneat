"""Strategy 26: Ecological Succession (Developmental Phase Preferences).

Implements ecological succession - ecosystems develop through stages from
pioneer species to climax community. Applied to palette evolution, this
means different evolutionary phases prefer different function types.

Biological/Ecological Basis:
- Pioneer species: Hardy generalists that colonize first (grasses, lichens)
- Intermediate species: Bridge between pioneers and specialists
- Climax species: Specialists that dominate stable ecosystems (oak trees)
- Succession is directional: generalist → specialist

Key Insight:
- Early evolution needs stable, general-purpose functions (tanh, sigmoid)
- Late evolution can afford specialized, complex functions (sin, burst)
- Treating all generations equally wastes early exploration
- Explicit developmental phases guide function discovery timing

Phases:
    Pioneer (gen 0-10): High exploration, prefer generalists
        - generalist_functions = [identity, tanh, sigmoid, relu]
        - High mutation rate (explore broadly)

    Intermediate (gen 10-30): Balanced exploration/exploitation
        - All functions equally weighted
        - Medium mutation rate

    Climax (gen 30+): Low exploration, prefer discovered specialists
        - specialist_functions = [sin, burst, osc_adapt, ...]
        - Low mutation rate (exploit what works)
        - Protect functions discovered during intermediate phase

Expected improvements:
- Faster early convergence (generalists provide stable base)
- Better late-stage specialization (complex functions added when ready)
- Natural curriculum learning (simple → complex)
- Reduced random exploration in climax phase
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


class EcologicalSuccessionStrategy(PaletteEvolutionStrategy):
    """Developmental phases with different function preferences.

    Evolution progresses through pioneer → intermediate → climax phases,
    with each phase having different exploration rates and function biases.
    """

    name = "ecological_succession"
    description = "Developmental phases: pioneer generalists → climax specialists"

    # Function classifications (based on complexity and specialization)
    # Generalists: Simple, stable, widely useful
    GENERALIST_FUNCTIONS = [0, 1, 2, 5, 6]  # identity, tanh, sigmoid, relu, lrelu
    # Specialists: Complex, specific, powerful when appropriate
    SPECIALIST_FUNCTIONS = [4, 11, 12, 13, 15]  # sin, burst, osc_adapt, modulated, log_cosh
    # Neutral: Can work in any phase
    NEUTRAL_FUNCTIONS = [3, 7, 8, 9, 10, 14, 16, 17]  # step, elu, swish, gelu, etc.

    def __init__(
        self,
        # Phase boundaries
        pioneer_end: int = 10,                # End of pioneer phase
        intermediate_end: int = 30,           # End of intermediate phase
        transition_smoothness: float = 5.0,   # Generations for smooth transitions
        # Pioneer phase parameters
        pioneer_mutation_rate: float = 0.25,  # High exploration
        pioneer_generalist_bias: float = 2.0, # 2x more likely to activate generalists
        # Intermediate phase parameters
        intermediate_mutation_rate: float = 0.12,
        intermediate_bias: float = 1.0,       # Equal weighting
        # Climax phase parameters
        climax_mutation_rate: float = 0.04,   # Low exploration
        climax_specialist_bias: float = 1.5,  # Prefer specialists
        climax_discovery_protection: float = 0.8,  # Protect discovered specialists
        # Affinity parameters
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.01,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Ecological Succession strategy.

        Args:
            pioneer_end: Generation where pioneer phase ends
            intermediate_end: Generation where intermediate phase ends
            transition_smoothness: Generations for smooth phase transitions
            pioneer_mutation_rate: Base mutation rate during pioneer phase
            pioneer_generalist_bias: Multiplier for activating generalist functions
            intermediate_mutation_rate: Base mutation rate during intermediate
            climax_mutation_rate: Base mutation rate during climax phase
            climax_specialist_bias: Multiplier for activating specialist functions
            climax_discovery_protection: Affinity threshold for protection in climax
        """
        # Phase boundaries
        self.pioneer_end = pioneer_end
        self.intermediate_end = intermediate_end
        self.transition_smoothness = transition_smoothness

        # Pioneer
        self.pioneer_mutation_rate = pioneer_mutation_rate
        self.pioneer_generalist_bias = pioneer_generalist_bias

        # Intermediate
        self.intermediate_mutation_rate = intermediate_mutation_rate
        self.intermediate_bias = intermediate_bias

        # Climax
        self.climax_mutation_rate = climax_mutation_rate
        self.climax_specialist_bias = climax_specialist_bias
        self.climax_discovery_protection = climax_discovery_protection

        # Affinity
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

        # Build function type lookup
        self.function_type = {}
        for i in self.GENERALIST_FUNCTIONS:
            self.function_type[i] = 'generalist'
        for i in self.SPECIALIST_FUNCTIONS:
            self.function_type[i] = 'specialist'
        for i in self.NEUTRAL_FUNCTIONS:
            self.function_type[i] = 'neutral'

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with phase tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Function affinity (learned value)
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Track first discovery generation for each function
        discovery_generations = jnp.full(NUM_ACTIVATIONS, -1, dtype=jnp.float32)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                discovery_generations = discovery_generations.at[i].set(0.0)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 262626),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Phase tracking
            'current_phase': 'pioneer',
            'phase_progress': 0.0,  # 0-1 within current phase
            # Function tracking
            'function_affinity': function_affinity,
            'discovery_generations': discovery_generations,
            'discovered_specialists': [],  # Specialists found during intermediate
            # Tracking
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _get_phase_params(self, generation: int) -> Dict[str, Any]:
        """Get parameters for current developmental phase.

        Returns smoothly interpolated parameters during transitions.
        """
        if generation < self.pioneer_end:
            # Pioneer phase
            phase = 'pioneer'
            progress = generation / self.pioneer_end
            mutation_rate = self.pioneer_mutation_rate
            generalist_bias = self.pioneer_generalist_bias
            specialist_bias = 0.5  # Discourage specialists early
            protection_threshold = 0.8  # Only protect very high affinity

        elif generation < self.intermediate_end:
            # Intermediate phase
            phase = 'intermediate'
            gen_in_phase = generation - self.pioneer_end
            phase_length = self.intermediate_end - self.pioneer_end
            progress = gen_in_phase / phase_length

            # Smooth transition from pioneer
            transition_factor = min(1.0, gen_in_phase / self.transition_smoothness)

            # Mutation rate decreases through intermediate
            mutation_rate = (
                self.pioneer_mutation_rate * (1 - transition_factor) +
                self.intermediate_mutation_rate * transition_factor
            )

            # Biases become neutral
            generalist_bias = (
                self.pioneer_generalist_bias * (1 - progress) +
                self.intermediate_bias * progress
            )
            specialist_bias = (
                0.5 * (1 - progress) +
                self.intermediate_bias * progress
            )
            protection_threshold = 0.65  # Moderate protection

        else:
            # Climax phase
            phase = 'climax'
            gen_in_phase = generation - self.intermediate_end
            progress = min(1.0, gen_in_phase / 20)  # Cap at 20 gens into climax

            # Smooth transition from intermediate
            transition_factor = min(1.0, gen_in_phase / self.transition_smoothness)

            mutation_rate = (
                self.intermediate_mutation_rate * (1 - transition_factor) +
                self.climax_mutation_rate * transition_factor
            )

            # Specialist bias increases
            generalist_bias = 1.0  # Neutral for generalists
            specialist_bias = (
                self.intermediate_bias * (1 - transition_factor) +
                self.climax_specialist_bias * transition_factor
            )
            protection_threshold = self.climax_discovery_protection

        return {
            'phase': phase,
            'progress': progress,
            'mutation_rate': mutation_rate,
            'generalist_bias': generalist_bias,
            'specialist_bias': specialist_bias,
            'protection_threshold': protection_threshold,
        }

    def _get_function_bias(self, func_idx: int, phase_params: Dict) -> float:
        """Get activation bias for a function based on current phase."""
        func_type = self.function_type.get(func_idx, 'neutral')

        if func_type == 'generalist':
            return phase_params['generalist_bias']
        elif func_type == 'specialist':
            return phase_params['specialist_bias']
        else:
            return 1.0  # Neutral functions

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_improved: bool,
        phase: str,
    ) -> jnp.ndarray:
        """Update function affinity based on co-occurrence with success."""
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_improved:
            # Strengthen active functions
            signal = self.affinity_lr * active
        else:
            # Slight weakening
            signal = -self.affinity_lr * 0.3 * active

        new_affinity = affinity + signal

        # Decay inactive (stronger in climax to prune unused)
        decay_rate = self.affinity_decay * (1.5 if phase == 'climax' else 1.0)
        new_affinity = new_affinity - decay_rate * (1 - active) * affinity

        return jnp.clip(new_affinity, 0.05, 0.95)

    def _apply_succession_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        phase_params: Dict,
        discovered_specialists: List[int],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with phase-dependent biasing."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        base_rate = phase_params['mutation_rate']
        protection_threshold = phase_params['protection_threshold']
        phase = phase_params['phase']

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_ACTIVATIONS):
            aff = float(affinity[i])
            bias = self._get_function_bias(i, phase_params)

            if mask[i] < 0.5:
                # Inactive: maybe activate
                # Rate influenced by phase bias and affinity
                rate = base_rate * 0.5 * bias * (0.5 + 0.5 * aff)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                # Check protection
                is_protected = (
                    aff >= protection_threshold or
                    (phase == 'climax' and i in discovered_specialists)
                )

                if is_protected:
                    rate = base_rate * 0.05  # Very low for protected
                else:
                    # Higher deactivation for low affinity, scaled by inverse bias
                    inv_bias = 1.0 / max(bias, 0.5)
                    rate = base_rate * 0.4 * (1.0 - aff) * inv_bias

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'phase': phase,
            'mutation_rate': base_rate,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with ecological succession dynamics."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Get phase parameters
        phase_params = self._get_phase_params(generation)

        # Update affinity
        new_affinity = self._update_affinity(
            state['function_affinity'],
            state['mask'],
            improved,
            phase_params['phase'],
        )

        # Track discovered specialists during intermediate phase
        discovered_specialists = list(state['discovered_specialists'])
        if phase_params['phase'] == 'intermediate':
            active = mask_to_indices(state['mask'])
            for i in active:
                if i in self.SPECIALIST_FUNCTIONS and i not in discovered_specialists:
                    discovered_specialists.append(i)

        # Update discovery generations
        new_discovery = state['discovery_generations'].copy()
        for i in mask_to_indices(state['mask']):
            if new_discovery[i] < 0:
                new_discovery = new_discovery.at[i].set(float(generation))

        # Apply succession mutation
        new_mask, mutation_info = self._apply_succession_mutation(
            subkey,
            state['mask'],
            new_affinity,
            phase_params,
            discovered_specialists,
        )

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
            # Phase
            'current_phase': phase_params['phase'],
            'phase_progress': phase_params['progress'],
            # Functions
            'function_affinity': new_affinity,
            'discovery_generations': new_discovery,
            'discovered_specialists': discovered_specialists,
            # Tracking
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Count by type
        n_generalists = sum(1 for i in active_palette if i in self.GENERALIST_FUNCTIONS)
        n_specialists = sum(1 for i in active_palette if i in self.SPECIALIST_FUNCTIONS)
        n_neutral = len(active_palette) - n_generalists - n_specialists

        # Top affinity
        top_aff_idx = jnp.argsort(new_affinity)[-3:][::-1]
        top_affinity = [(int(i), float(new_affinity[i])) for i in top_aff_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Phase
            'phase': phase_params['phase'],
            'phase_progress': phase_params['progress'],
            'mutation_rate': phase_params['mutation_rate'],
            # Composition
            'n_generalists': n_generalists,
            'n_specialists': n_specialists,
            'n_neutral': n_neutral,
            'generalist_ratio': n_generalists / max(len(active_palette), 1),
            'specialist_ratio': n_specialists / max(len(active_palette), 1),
            # Specialists
            'discovered_specialists': discovered_specialists,
            'has_sin': 4 in active_palette,
            'sin_affinity': float(new_affinity[4]),
            # Affinity
            'avg_affinity': float(jnp.mean(new_affinity)),
            'top_affinity_functions': top_affinity,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with succession phase info."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']

        # Count by type
        n_generalists = sum(1 for i in palette if i in self.GENERALIST_FUNCTIONS)
        n_specialists = sum(1 for i in palette if i in self.SPECIALIST_FUNCTIONS)

        # Top affinity
        top_aff_idx = jnp.argsort(affinity)[-5:][::-1]
        top_affinities = [(int(i), float(affinity[i])) for i in top_aff_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Phase
            'current_phase': state['current_phase'],
            'phase_progress': state['phase_progress'],
            # Composition
            'n_generalists': n_generalists,
            'n_specialists': n_specialists,
            'generalist_ratio': n_generalists / max(len(palette), 1),
            'specialist_ratio': n_specialists / max(len(palette), 1),
            # Specialists
            'discovered_specialists': state['discovered_specialists'],
            # Affinity
            'top_affinity_functions': top_affinities,
            'sin_affinity': float(affinity[4]),
        }
