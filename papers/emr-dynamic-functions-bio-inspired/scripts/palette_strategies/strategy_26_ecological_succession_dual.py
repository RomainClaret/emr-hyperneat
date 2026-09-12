"""Strategy 26D: Ecological Succession Dual (Developmental Phases for Both Palettes).

Extends Ecological Succession to jointly evolve activation AND aggregation
function palettes with developmental phases for both domains.

Key mechanisms:
1. Pioneer phase: High exploration, generalist functions for both domains
2. Intermediate phase: Balanced exploration/exploitation
3. Climax phase: Low exploration, specialist functions protected
4. Cross-domain: Generalist act → generalist agg pairings in pioneer

Ecological basis:
- Pioneer species are hardy generalists (simple activations, basic aggregations)
- Climax species are specialists (sin, burst for activations; max, min for aggregations)
- Succession is directional: generalist → specialist in both domains
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


class EcologicalSuccessionDualStrategy(PaletteEvolutionStrategy):
    """Developmental phases with function preferences for both palettes."""

    name = "ecological_succession_dual"
    description = "Ecological succession: pioneer generalists → climax specialists for both domains"

    # Activation function classifications
    ACT_GENERALIST = [0, 1, 2, 5, 6]  # identity, tanh, sigmoid, relu, lrelu
    ACT_SPECIALIST = [4, 11, 12, 13, 15]  # sin, burst, osc_adapt, modulated, log_cosh
    ACT_NEUTRAL = [3, 7, 8, 9, 10, 14, 16, 17]

    # Aggregation function classifications
    AGG_GENERALIST = [0, 1]  # sum, mean - basic, stable
    AGG_SPECIALIST = [2, 3]  # max, min - powerful but specific
    AGG_NEUTRAL = [4, 5]  # product, maxabs

    def __init__(
        self,
        # Phase boundaries
        pioneer_end: int = 10,
        intermediate_end: int = 30,
        transition_smoothness: float = 5.0,
        # Pioneer phase
        pioneer_mutation_rate: float = 0.25,
        pioneer_generalist_bias: float = 2.0,
        # Intermediate phase
        intermediate_mutation_rate: float = 0.12,
        intermediate_bias: float = 1.0,
        # Climax phase
        climax_mutation_rate: float = 0.04,
        climax_specialist_bias: float = 1.5,
        climax_discovery_protection: float = 0.8,
        # Affinity parameters
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.01,
        # Cross-domain
        cross_learning_rate: float = 0.06,
        # Constraints
        max_active_act: int = 6,
        max_active_agg: int = 4,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.pioneer_end = pioneer_end
        self.intermediate_end = intermediate_end
        self.transition_smoothness = transition_smoothness

        self.pioneer_mutation_rate = pioneer_mutation_rate
        self.pioneer_generalist_bias = pioneer_generalist_bias

        self.intermediate_mutation_rate = intermediate_mutation_rate
        self.intermediate_bias = intermediate_bias

        self.climax_mutation_rate = climax_mutation_rate
        self.climax_specialist_bias = climax_specialist_bias
        self.climax_discovery_protection = climax_discovery_protection

        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_learning_rate = cross_learning_rate

        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

        # Build type lookups
        self.act_type = {}
        for i in self.ACT_GENERALIST:
            self.act_type[i] = 'generalist'
        for i in self.ACT_SPECIALIST:
            self.act_type[i] = 'specialist'
        for i in self.ACT_NEUTRAL:
            self.act_type[i] = 'neutral'

        self.agg_type = {}
        for i in self.AGG_GENERALIST:
            self.agg_type[i] = 'generalist'
        for i in self.AGG_SPECIALIST:
            self.agg_type[i] = 'specialist'
        for i in self.AGG_NEUTRAL:
            self.agg_type[i] = 'neutral'

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        # Activation
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Aggregation
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_mask(initial_agg)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Discovery tracking
        act_discovery = jnp.full(NUM_ACTIVATIONS, -1, dtype=jnp.float32)
        agg_discovery = jnp.full(NUM_AGGREGATIONS, -1, dtype=jnp.float32)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_discovery = act_discovery.at[i].set(0.0)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_discovery = agg_discovery.at[i].set(0.0)

        return {
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_discovery': act_discovery,
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_discovery': agg_discovery,
            'cross_affinity': cross_affinity,
            'rng_key': jax.random.PRNGKey(seed + 262626),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'current_phase': 'pioneer',
            'phase_progress': 0.0,
            'discovered_act_specialists': [],
            'discovered_agg_specialists': [],
            'mask': act_mask,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return agg_mask_to_indices(state['agg_mask'])

    def _get_phase_params(self, generation: int) -> Dict[str, Any]:
        """Get parameters for current developmental phase."""
        if generation < self.pioneer_end:
            phase = 'pioneer'
            progress = generation / self.pioneer_end
            mutation_rate = self.pioneer_mutation_rate
            generalist_bias = self.pioneer_generalist_bias
            specialist_bias = 0.5
            protection_threshold = 0.8

        elif generation < self.intermediate_end:
            phase = 'intermediate'
            gen_in_phase = generation - self.pioneer_end
            phase_length = self.intermediate_end - self.pioneer_end
            progress = gen_in_phase / phase_length
            transition_factor = min(1.0, gen_in_phase / self.transition_smoothness)

            mutation_rate = (
                self.pioneer_mutation_rate * (1 - transition_factor) +
                self.intermediate_mutation_rate * transition_factor
            )
            generalist_bias = (
                self.pioneer_generalist_bias * (1 - progress) +
                self.intermediate_bias * progress
            )
            specialist_bias = (
                0.5 * (1 - progress) +
                self.intermediate_bias * progress
            )
            protection_threshold = 0.65

        else:
            phase = 'climax'
            gen_in_phase = generation - self.intermediate_end
            progress = min(1.0, gen_in_phase / 20)
            transition_factor = min(1.0, gen_in_phase / self.transition_smoothness)

            mutation_rate = (
                self.intermediate_mutation_rate * (1 - transition_factor) +
                self.climax_mutation_rate * transition_factor
            )
            generalist_bias = 1.0
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

    def _get_function_bias(self, func_idx: int, phase_params: Dict, is_activation: bool) -> float:
        """Get activation bias for a function based on phase."""
        if is_activation:
            func_type = self.act_type.get(func_idx, 'neutral')
        else:
            func_type = self.agg_type.get(func_idx, 'neutral')

        if func_type == 'generalist':
            return phase_params['generalist_bias']
        elif func_type == 'specialist':
            return phase_params['specialist_bias']
        return 1.0

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_improved: bool,
        phase: str,
        num_funcs: int,
    ) -> jnp.ndarray:
        """Update function affinity."""
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_improved:
            signal = self.affinity_lr * active
        else:
            signal = -self.affinity_lr * 0.3 * active

        new_affinity = affinity + signal
        decay_rate = self.affinity_decay * (1.5 if phase == 'climax' else 1.0)
        new_affinity = new_affinity - decay_rate * (1 - active) * affinity

        return jnp.clip(new_affinity, 0.05, 0.95)

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_improved: bool,
        phase: str,
    ) -> jnp.ndarray:
        """Update cross-domain affinity."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)

        if fitness_improved:
            delta = self.cross_learning_rate * cross_active
        else:
            delta = -self.cross_learning_rate * 0.3 * cross_active

        new_cross = cross_affinity + delta
        decay_rate = self.affinity_decay * (1.5 if phase == 'climax' else 1.0)
        inactive = 1.0 - cross_active
        new_cross = new_cross - decay_rate * inactive * (cross_affinity - 0.5)

        return jnp.clip(new_cross, 0.0, 1.0)

    def _compute_protection_scores_act(
        self,
        affinity: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_agg = max(jnp.sum(agg_active), 1)
        cross_score = jnp.dot(cross_affinity, agg_active) / n_agg
        return 0.70 * affinity + 0.30 * cross_score

    def _compute_protection_scores_agg(
        self,
        affinity: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        act_active = (act_mask > 0.5).astype(jnp.float32)
        n_act = max(jnp.sum(act_active), 1)
        cross_score = jnp.dot(cross_affinity.T, act_active) / n_act
        return 0.70 * affinity + 0.30 * cross_score

    def _apply_succession_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        protection_scores: jnp.ndarray,
        phase_params: Dict,
        discovered_specialists: List[int],
        is_activation: bool,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with phase-dependent biasing."""
        key1, key2 = jax.random.split(key)

        if is_activation:
            num_funcs = NUM_ACTIVATIONS
            max_active = self.max_active_act
            min_active = self.min_active_act
            specialist_list = self.ACT_SPECIALIST
        else:
            num_funcs = NUM_AGGREGATIONS
            max_active = self.max_active_agg
            min_active = self.min_active_agg
            specialist_list = self.AGG_SPECIALIST

        base_rate = phase_params['mutation_rate']
        protection_threshold = phase_params['protection_threshold']
        phase = phase_params['phase']

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (num_funcs,))
        deactivate_probs = jax.random.uniform(key2, (num_funcs,))

        for i in range(num_funcs):
            prot = float(protection_scores[i])
            aff = float(affinity[i])
            bias = self._get_function_bias(i, phase_params, is_activation)

            if mask[i] < 0.5:
                current_active = int(jnp.sum(mask > 0.5))
                if current_active >= max_active:
                    continue

                rate = base_rate * 0.5 * bias * (0.5 + 0.5 * aff)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                is_protected = (
                    prot >= protection_threshold or
                    (phase == 'climax' and i in discovered_specialists)
                )

                if is_protected:
                    rate = base_rate * 0.05
                else:
                    inv_bias = 1.0 / max(bias, 0.5)
                    rate = base_rate * 0.4 * (1.0 - aff) * inv_bias

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key, subkey1, subkey2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        phase_params = self._get_phase_params(generation)
        phase = phase_params['phase']

        # Update affinities
        new_act_affinity = self._update_affinity(
            state['act_affinity'], state['act_mask'], improved, phase, NUM_ACTIVATIONS
        )
        new_agg_affinity = self._update_affinity(
            state['agg_affinity'], state['agg_mask'], improved, phase, NUM_AGGREGATIONS
        )
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improved, phase
        )

        # Track discovered specialists
        discovered_act = list(state['discovered_act_specialists'])
        discovered_agg = list(state['discovered_agg_specialists'])

        if phase == 'intermediate':
            for i in mask_to_indices(state['act_mask']):
                if i in self.ACT_SPECIALIST and i not in discovered_act:
                    discovered_act.append(i)
            for i in agg_mask_to_indices(state['agg_mask']):
                if i in self.AGG_SPECIALIST and i not in discovered_agg:
                    discovered_agg.append(i)

        # Protection scores
        act_protection = self._compute_protection_scores_act(
            new_act_affinity, new_cross, state['agg_mask']
        )
        agg_protection = self._compute_protection_scores_agg(
            new_agg_affinity, new_cross, state['act_mask']
        )

        # Apply mutations
        new_act_mask, act_mut = self._apply_succession_mutation(
            subkey1, state['act_mask'], new_act_affinity, act_protection,
            phase_params, discovered_act, True
        )
        new_agg_mask, agg_mut = self._apply_succession_mutation(
            subkey2, state['agg_mask'], new_agg_affinity, agg_protection,
            phase_params, discovered_agg, False
        )

        new_state = {
            'act_mask': new_act_mask,
            'act_affinity': new_act_affinity,
            'act_discovery': state['act_discovery'],
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_affinity,
            'agg_discovery': state['agg_discovery'],
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'current_phase': phase,
            'phase_progress': phase_params['progress'],
            'discovered_act_specialists': discovered_act,
            'discovered_agg_specialists': discovered_agg,
            'mask': new_act_mask,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = agg_mask_to_indices(new_agg_mask)

        n_act_gen = sum(1 for i in act_palette if i in self.ACT_GENERALIST)
        n_act_spec = sum(1 for i in act_palette if i in self.ACT_SPECIALIST)
        n_agg_gen = sum(1 for i in agg_palette if i in self.AGG_GENERALIST)
        n_agg_spec = sum(1 for i in agg_palette if i in self.AGG_SPECIALIST)

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_progress': phase_params['progress'],
            'mutation_rate': phase_params['mutation_rate'],
            # Composition
            'n_act_generalists': n_act_gen,
            'n_act_specialists': n_act_spec,
            'n_agg_generalists': n_agg_gen,
            'n_agg_specialists': n_agg_spec,
            # Affinity
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'sin_affinity': float(new_act_affinity[4]),
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Specialists
            'discovered_act_specialists': discovered_act,
            'discovered_agg_specialists': discovered_agg,
            'has_sin': 4 in act_palette,
            'act_activated': act_mut['activated'],
            'act_deactivated': act_mut['deactivated'],
            'agg_activated': agg_mut['activated'],
            'agg_deactivated': agg_mut['deactivated'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'phase': state['current_phase'],
            'phase_progress': state['phase_progress'],
            'generation': state['generation'],
            'sin_affinity': float(state['act_affinity'][4]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
            'discovered_act_specialists': state['discovered_act_specialists'],
            'discovered_agg_specialists': state['discovered_agg_specialists'],
        }
