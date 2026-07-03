"""Strategy 7 Dual: Neuromodulated Palette Evolution for Both Activation AND Aggregation.

Extends Neuromodulated strategy to jointly evolve both palettes using same
neuromodulatory signals:

Bio-inspired neuromodulation:
- Dopamine (DA): Reward signal → reduce exploration when improving
- Acetylcholine (ACh): Uncertainty signal → increase exploration when stagnating
- Norepinephrine (NE): Arousal signal → amplify plasticity early or when challenged

Both activation and aggregation palettes respond to the same global neuromodulator
state - just as in the brain, neuromodulators affect learning globally.

Key innovation: Cross-domain sticky patterns - oscillatory activations (sin, burst,
resonator) protected together with complementary aggregations.
"""

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
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
)


# Oscillatory activation indices that are "sticky"
OSCILLATORY_INDICES = [4, 11, 12]  # sin, burst, resonator

# Aggregation indices that pair well with oscillatory activations
OSCILLATORY_FRIENDLY_AGGS = [0, 1, 3]  # sum, mean, min (not product which explodes)


class NeuromodulatedDualStrategy(PaletteEvolutionStrategy):
    """Asymmetric+Sticky with neuromodulatory rate adjustment for BOTH palettes.

    Neuromodulation dynamically adjusts rates for both domains:
    - Dopamine (reward): When fitness improves → reduce exploration (exploit)
    - Acetylcholine (uncertainty): When stagnating → increase exploration
    - Norepinephrine (arousal): Early in evolution → high plasticity

    Cross-domain learning:
    - Tracks which activation-aggregation combinations succeed
    - Sticky oscillatory activations can protect complementary aggregations
    """

    name = "neuromodulated_dual"
    description = "Dual palette with shared DA/ACh/NE neuromodulation"

    def __init__(
        self,
        # Base rates for activations
        base_activate_rate: float = 0.25,
        base_deactivate_rate: float = 0.05,
        # Base rates for aggregations
        agg_base_activate_rate: float = 0.20,
        agg_base_deactivate_rate: float = 0.06,
        # Neuromodulation parameters
        dopamine_sensitivity: float = 0.5,
        acetylcholine_sensitivity: float = 0.3,
        norepinephrine_sensitivity: float = 0.2,
        modulation_ema_alpha: float = 0.3,
        # Stagnation
        stagnation_threshold: int = 3,
        # Sticky oscillatory
        deactivate_sticky_rate: float = 0.01,
        agg_deactivate_sticky_rate: float = 0.02,
        # Cross-domain learning
        cross_learning_rate: float = 0.1,
        cross_influence: float = 0.25,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,   # Optimal is 6, >6 causes antagonism
        max_active_agg: int = 4,   # Optimal is 4 for parity
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize strategy."""
        # Activation rates
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        # Aggregation rates
        self.agg_base_activate_rate = agg_base_activate_rate
        self.agg_base_deactivate_rate = agg_base_deactivate_rate
        # Neuromodulation
        self.dopamine_sensitivity = dopamine_sensitivity
        self.acetylcholine_sensitivity = acetylcholine_sensitivity
        self.norepinephrine_sensitivity = norepinephrine_sensitivity
        self.modulation_ema_alpha = modulation_ema_alpha
        self.stagnation_threshold = stagnation_threshold
        # Sticky
        self.deactivate_sticky_rate = deactivate_sticky_rate
        self.agg_deactivate_sticky_rate = agg_deactivate_sticky_rate
        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neuromodulator tracking and cross-domain affinity."""
        # Activation palette
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        # Aggregation palette
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Palette masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Common state
            'rng_key': jax.random.PRNGKey(seed + 77777),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'prev_fitness': 0.0,
            'strategy_name': self.name,
            # Neuromodulator levels (shared for both domains - like real brain)
            'dopamine': 0.5,
            'acetylcholine': 0.5,
            'norepinephrine': 1.0,  # Start high
            # Tracking
            'fitness_history': [],
            'improvement_rate_ema': 0.0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _update_neuromodulators(
        self,
        state: Dict[str, Any],
        best_fitness: float,
        prev_best_fitness: float,
        generation: int,
    ) -> Dict[str, float]:
        """Update neuromodulator levels based on learning progress."""
        alpha = self.modulation_ema_alpha

        # --- Dopamine: Reward signal ---
        improvement = best_fitness - prev_best_fitness
        if prev_best_fitness > 0:
            relative_improvement = improvement / prev_best_fitness
        else:
            relative_improvement = improvement

        da_signal = max(0, min(1, 0.5 + relative_improvement * 10))
        new_dopamine = (1 - alpha) * state['dopamine'] + alpha * da_signal

        # --- Acetylcholine: Uncertainty signal ---
        stagnation = state['stagnation_count'] / max(self.stagnation_threshold, 1)
        ach_signal = min(1.0, stagnation)
        new_acetylcholine = (1 - alpha) * state['acetylcholine'] + alpha * ach_signal

        # --- Norepinephrine: Arousal/Plasticity signal ---
        time_decay = max(0, 1.0 - generation / 50.0)
        challenge = 1.0 - best_fitness
        ne_signal = max(time_decay, challenge * 0.5)
        new_norepinephrine = (1 - alpha) * state['norepinephrine'] + alpha * ne_signal

        return {
            'dopamine': float(new_dopamine),
            'acetylcholine': float(new_acetylcholine),
            'norepinephrine': float(new_norepinephrine),
        }

    def _compute_effective_rates(
        self,
        dopamine: float,
        acetylcholine: float,
        norepinephrine: float,
        base_activate: float,
        base_deactivate: float,
    ) -> Tuple[float, float]:
        """Compute effective rates from neuromodulators."""
        da_factor = 1.0 - self.dopamine_sensitivity * (dopamine - 0.5)
        ach_factor = 1.0 + self.acetylcholine_sensitivity * (acetylcholine - 0.5)
        ne_factor = 0.5 + self.norepinephrine_sensitivity * norepinephrine

        effective_activate = base_activate * da_factor * ach_factor * ne_factor
        effective_deactivate = base_deactivate * (1.0 / max(da_factor, 0.5)) * ne_factor

        effective_activate = max(0.05, min(0.5, effective_activate))
        effective_deactivate = max(0.01, min(0.2, effective_deactivate))

        return effective_activate, effective_deactivate

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity based on fitness."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)

        lr = self.cross_learning_rate
        if fitness_signal >= 0:
            delta = lr * fitness_signal * cross_active
        else:
            delta = (lr * 0.5) * fitness_signal * cross_active

        return jnp.clip(cross_affinity + delta, 0.0, 1.0)

    def _mutate_act_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        effective_activate: float,
        effective_deactivate: float,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, List[int], List[int]]:
        """Apply neuromodulated mutation to activation palette.

        Key constraint: max_active_act prevents antagonism (>6 activations hurts).
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        # Compute cross-domain protection boost
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_agg_active = max(jnp.sum(agg_active), 1)
        cross_boost = jnp.dot(cross_affinity, agg_active) / n_agg_active

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        # Track current active count for max constraint
        current_active = int(jnp.sum(mask > 0.5))

        for i in range(NUM_ACTIVATIONS):
            if mask[i] < 0.5:
                # Inactive - might activate
                # CRITICAL: Skip if already at max
                if current_active + len(activated) >= self.max_active_act:
                    continue

                # Boost activation if cross-domain affinity is high
                boost = 1.0 + self.cross_influence * float(cross_boost[i])
                if activate_probs[i] < effective_activate * boost:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                # Use sticky rate for oscillatory functions
                if i in OSCILLATORY_INDICES:
                    deact_rate = self.deactivate_sticky_rate
                else:
                    # Reduce deactivation if high cross-domain affinity
                    protection = 1.0 - self.cross_influence * float(cross_boost[i])
                    deact_rate = effective_deactivate * max(0.2, protection)

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < self.min_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, activated, deactivated

    def _mutate_agg_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        effective_activate: float,
        effective_deactivate: float,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, List[int], List[int]]:
        """Apply neuromodulated mutation to aggregation palette.

        Key constraint: max_active_agg prevents overload.
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        # Compute cross-domain protection from activations
        act_active = (act_mask > 0.5).astype(jnp.float32)
        n_act_active = max(jnp.sum(act_active), 1)
        cross_boost = jnp.dot(cross_affinity.T, act_active) / n_act_active

        # Check if oscillatory activations are present (sticky protection for friendly aggs)
        has_oscillatory = any(act_mask[i] > 0.5 for i in OSCILLATORY_INDICES)

        activate_probs = jax.random.uniform(key1, (NUM_AGGREGATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_AGGREGATIONS,))

        # Track current active count for max constraint
        current_active = int(jnp.sum(mask > 0.5))

        for i in range(NUM_AGGREGATIONS):
            if mask[i] < 0.5:
                # Inactive - might activate
                # CRITICAL: Skip if already at max
                if current_active + len(activated) >= self.max_active_agg:
                    continue

                boost = 1.0 + self.cross_influence * float(cross_boost[i])
                if activate_probs[i] < effective_activate * boost:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                # Use sticky rate for oscillatory-friendly aggregations when oscillatory present
                if has_oscillatory and i in OSCILLATORY_FRIENDLY_AGGS:
                    deact_rate = self.agg_deactivate_sticky_rate
                else:
                    protection = 1.0 - self.cross_influence * float(cross_boost[i])
                    deact_rate = effective_deactivate * max(0.2, protection)

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < self.min_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, activated, deactivated

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with neuromodulated dual mutation."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        # Check if fitness improved
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update neuromodulator levels (SHARED for both domains)
        neuromodulators = self._update_neuromodulators(
            state, best_fitness, prev_best_fitness, generation
        )

        # Compute effective rates for each domain
        act_effective_activate, act_effective_deactivate = self._compute_effective_rates(
            neuromodulators['dopamine'],
            neuromodulators['acetylcholine'],
            neuromodulators['norepinephrine'],
            self.base_activate_rate,
            self.base_deactivate_rate,
        )
        agg_effective_activate, agg_effective_deactivate = self._compute_effective_rates(
            neuromodulators['dopamine'],
            neuromodulators['acetylcholine'],
            neuromodulators['norepinephrine'],
            self.agg_base_activate_rate,
            self.agg_base_deactivate_rate,
        )

        # Compute fitness signal for cross-domain learning
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 10:
            fitness_history = fitness_history[-10:]
        baseline = sum(fitness_history) / len(fitness_history)
        fitness_signal = (best_fitness - baseline) / max(0.1, baseline)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Update cross-domain affinity
        new_cross_affinity = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_signal,
        )

        # Apply neuromodulated mutation to both palettes
        new_act_mask, act_activated, act_deactivated = self._mutate_act_palette(
            key_act, state['act_mask'],
            act_effective_activate, act_effective_deactivate,
            new_cross_affinity, state['agg_mask'],
        )
        new_agg_mask, agg_activated, agg_deactivated = self._mutate_agg_palette(
            key_agg, state['agg_mask'],
            agg_effective_activate, agg_effective_deactivate,
            new_cross_affinity, state['act_mask'],
        )

        act_palette_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_palette_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'cross_affinity': new_cross_affinity,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'prev_fitness': best_fitness,
            'strategy_name': self.name,
            'dopamine': neuromodulators['dopamine'],
            'acetylcholine': neuromodulators['acetylcholine'],
            'norepinephrine': neuromodulators['norepinephrine'],
            'fitness_history': fitness_history,
            'improvement_rate_ema': state['improvement_rate_ema'],
        }

        metrics = {
            'act_palette_changed': act_palette_changed,
            'agg_palette_changed': agg_palette_changed,
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neuromodulator levels
            'dopamine': neuromodulators['dopamine'],
            'acetylcholine': neuromodulators['acetylcholine'],
            'norepinephrine': neuromodulators['norepinephrine'],
            # Effective rates
            'act_effective_activate': act_effective_activate,
            'act_effective_deactivate': act_effective_deactivate,
            'agg_effective_activate': agg_effective_activate,
            'agg_effective_deactivate': agg_effective_deactivate,
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross_affinity)),
            'cross_max_affinity': float(jnp.max(new_cross_affinity)),
            # Mutation events
            'act_activated': act_activated,
            'act_deactivated': act_deactivated,
            'agg_activated': agg_activated,
            'agg_deactivated': agg_deactivated,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including neuromodulator and dual palette info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        # Find strongest cross-domain pair
        cross_aff = state['cross_affinity']
        cross_strongest_val = float(jnp.max(cross_aff))
        cross_strongest_idx = jnp.unravel_index(jnp.argmax(cross_aff), cross_aff.shape)

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'act_palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'has_burst': 11 in act_palette,
            'has_resonator': 12 in act_palette,
            'stagnation_count': state['stagnation_count'],
            # Neuromodulator levels
            'dopamine': state['dopamine'],
            'acetylcholine': state['acetylcholine'],
            'norepinephrine': state['norepinephrine'],
            # Cross-domain
            'cross_strongest_pair': (int(cross_strongest_idx[0]), int(cross_strongest_idx[1])),
            'cross_strongest_weight': cross_strongest_val,
        }
