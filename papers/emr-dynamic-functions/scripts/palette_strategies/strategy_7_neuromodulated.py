"""Strategy 7: Neuromodulated Asymmetric+Sticky.

Extends Strategy 3 (asymmetric_sticky) with neuromodulatory signals that
dynamically adjust activation/deactivation rates based on learning progress.

Bio-inspired neuromodulation:
- Dopamine (DA): Reward signal → reduce exploration when improving
- Acetylcholine (ACh): Uncertainty signal → increase exploration when stagnating
- Norepinephrine (NE): Arousal signal → amplify plasticity early or when challenged

Expected: 100% discovery maintained, 75-85% solve (vs 67% asymmetric_sticky)
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


# Oscillatory activation indices that are "sticky"
OSCILLATORY_INDICES = [4, 11, 12]  # sin, burst, resonator


class NeuromodulatedStrategy(PaletteEvolutionStrategy):
    """Asymmetric+Sticky with neuromodulatory rate adjustment.

    Neuromodulation dynamically adjusts rates:
    - Dopamine (reward): When fitness improves → reduce exploration (exploit)
    - Acetylcholine (uncertainty): When stagnating → increase exploration
    - Norepinephrine (arousal): Early in evolution → high plasticity

    effective_activate = base_activate * da_factor * ach_factor * ne_factor
    effective_deactivate = base_deactivate / da_factor * ach_factor * ne_factor
    """

    name = "neuromodulated"
    description = "Asymmetric+Sticky with DA/ACh/NE neuromodulation"

    def __init__(
        self,
        # Base rates (from asymmetric_sticky)
        base_activate_rate: float = 0.25,
        base_deactivate_rate: float = 0.05,
        # Neuromodulation parameters
        dopamine_sensitivity: float = 0.5,
        acetylcholine_sensitivity: float = 0.3,
        norepinephrine_sensitivity: float = 0.2,
        modulation_ema_alpha: float = 0.3,
        # Stagnation
        stagnation_threshold: int = 3,
        # Sticky oscillatory
        deactivate_sticky_rate: float = 0.01,
        # Constraints
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            base_activate_rate: Base probability of activating inactive function
            base_deactivate_rate: Base probability of deactivating active function
            dopamine_sensitivity: How much reward reduces exploration (0-1)
            acetylcholine_sensitivity: How much uncertainty increases exploration (0-1)
            norepinephrine_sensitivity: How much arousal affects plasticity (0-1)
            modulation_ema_alpha: EMA smoothing for neuromodulator levels
            stagnation_threshold: Gens without improvement to trigger ACh boost
            deactivate_sticky_rate: Rate for oscillatory functions (much lower)
            min_active: Minimum active functions
            initial_palette: Starting palette indices
        """
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.dopamine_sensitivity = dopamine_sensitivity
        self.acetylcholine_sensitivity = acetylcholine_sensitivity
        self.norepinephrine_sensitivity = norepinephrine_sensitivity
        self.modulation_ema_alpha = modulation_ema_alpha
        self.stagnation_threshold = stagnation_threshold
        self.deactivate_sticky_rate = deactivate_sticky_rate
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neuromodulator tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 77777),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'prev_fitness': 0.0,
            'strategy_name': self.name,
            # Neuromodulator levels (0-1 scale)
            'dopamine': 0.5,  # Start neutral
            'acetylcholine': 0.5,
            'norepinephrine': 1.0,  # Start high (early exploration)
            # Tracking for modulation
            'fitness_history': [],
            'improvement_rate_ema': 0.0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_neuromodulators(
        self,
        state: Dict[str, Any],
        best_fitness: float,
        prev_best_fitness: float,
        generation: int,
    ) -> Dict[str, float]:
        """Update neuromodulator levels based on learning progress.

        Returns:
            Dict with updated neuromodulator levels
        """
        alpha = self.modulation_ema_alpha

        # --- Dopamine: Reward signal ---
        # High when fitness is improving, low when stagnating
        improvement = best_fitness - prev_best_fitness
        if prev_best_fitness > 0:
            relative_improvement = improvement / prev_best_fitness
        else:
            relative_improvement = improvement

        # Scale to 0-1 range
        da_signal = max(0, min(1, 0.5 + relative_improvement * 10))
        new_dopamine = (1 - alpha) * state['dopamine'] + alpha * da_signal

        # --- Acetylcholine: Uncertainty signal ---
        # High when stagnating, low when making progress
        stagnation = state['stagnation_count'] / max(self.stagnation_threshold, 1)
        ach_signal = min(1.0, stagnation)
        new_acetylcholine = (1 - alpha) * state['acetylcholine'] + alpha * ach_signal

        # --- Norepinephrine: Arousal/Plasticity signal ---
        # High early in evolution, decays over time
        # Also spikes when fitness is challenging (far from 1.0)
        time_decay = max(0, 1.0 - generation / 50.0)  # Decay over 50 gens
        challenge = 1.0 - best_fitness  # High when far from solving
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
    ) -> Tuple[float, float]:
        """Compute effective activation/deactivation rates from neuromodulators.

        Returns:
            Tuple of (effective_activate, effective_deactivate)
        """
        # Dopamine: High DA → exploit (reduce activation, reduce deactivation)
        da_factor = 1.0 - self.dopamine_sensitivity * (dopamine - 0.5)

        # Acetylcholine: High ACh → explore (increase activation)
        ach_factor = 1.0 + self.acetylcholine_sensitivity * (acetylcholine - 0.5)

        # Norepinephrine: High NE → high plasticity (amplify both rates)
        ne_factor = 0.5 + self.norepinephrine_sensitivity * norepinephrine

        # Effective activation: base * (explore more with ACh) * (less with DA) * NE
        effective_activate = self.base_activate_rate * da_factor * ach_factor * ne_factor

        # Effective deactivation: base * (more with ACh if exploring) / (less with DA) * NE
        # But when DA is high (reward), we don't want to lose good activations
        effective_deactivate = self.base_deactivate_rate * (1.0 / max(da_factor, 0.5)) * ne_factor

        # Clamp to reasonable ranges
        effective_activate = max(0.05, min(0.5, effective_activate))
        effective_deactivate = max(0.01, min(0.2, effective_deactivate))

        return effective_activate, effective_deactivate

    def _mutate_palette_neuromodulated(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        effective_activate: float,
        effective_deactivate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply neuromodulated asymmetric mutation.

        Args:
            key: JAX random key
            mask: Current palette mask
            effective_activate: Neuromodulated activation rate
            effective_deactivate: Neuromodulated deactivation rate

        Returns:
            Tuple of (new_mask, mutation_info)
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        # Process each activation
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            if mask[i] < 0.5:
                # Currently inactive - might activate
                if activate_probs[i] < effective_activate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Currently active - might deactivate
                # Use sticky rate for oscillatory functions
                if i in OSCILLATORY_INDICES:
                    deact_rate = self.deactivate_sticky_rate
                else:
                    deact_rate = effective_deactivate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active constraint
        active_count = jnp.sum(new_mask > 0.5)
        if active_count < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
            'effective_activate': effective_activate,
            'effective_deactivate': effective_deactivate,
        }

        return new_mask, mutation_info

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with neuromodulated mutation.

        1. Update neuromodulator levels based on learning progress
        2. Compute effective rates from neuromodulators
        3. Apply neuromodulated mutation every generation
        """
        key, subkey = jax.random.split(state['rng_key'])

        # Check if fitness improved
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update neuromodulator levels
        neuromodulators = self._update_neuromodulators(
            state, best_fitness, prev_best_fitness, generation
        )

        # Compute effective rates
        effective_activate, effective_deactivate = self._compute_effective_rates(
            neuromodulators['dopamine'],
            neuromodulators['acetylcholine'],
            neuromodulators['norepinephrine'],
        )

        # Apply neuromodulated mutation every generation (not just on stagnation)
        new_mask, mutation_info = self._mutate_palette_neuromodulated(
            subkey, state['mask'], effective_activate, effective_deactivate
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update fitness history for tracking
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'prev_fitness': best_fitness,
            'strategy_name': self.name,
            # Neuromodulator levels
            'dopamine': neuromodulators['dopamine'],
            'acetylcholine': neuromodulators['acetylcholine'],
            'norepinephrine': neuromodulators['norepinephrine'],
            # Tracking
            'fitness_history': fitness_history,
            'improvement_rate_ema': state['improvement_rate_ema'],
        }

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': mask_to_indices(new_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neuromodulator levels
            'dopamine': neuromodulators['dopamine'],
            'acetylcholine': neuromodulators['acetylcholine'],
            'norepinephrine': neuromodulators['norepinephrine'],
            # Effective rates
            'effective_activate': effective_activate,
            'effective_deactivate': effective_deactivate,
        }

        if mutation_info:
            metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including neuromodulator levels."""
        palette = self.get_active_palette(state)

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'has_burst': 11 in palette,
            'has_resonator': 12 in palette,
            'stagnation_count': state['stagnation_count'],
            # Neuromodulator levels
            'dopamine': state['dopamine'],
            'acetylcholine': state['acetylcholine'],
            'norepinephrine': state['norepinephrine'],
        }
