"""Strategy 10: NeuroHebbian Hybrid.

Combines the best mechanisms from:
- Neuromodulated (Strategy 7): Dynamic exploration/exploitation via DA/ACh/NE
- Hebbian (Strategy 8): Co-occurrence learning to find good activation pairs

Key innovation:
- Hebbian weights guide WHICH activations to try (affinity-based selection)
- Neuromodulation controls WHEN to explore vs exploit (rate modulation)
- Double protection: Hebbian consolidation + neuromodulated sticky oscillatory

Expected: 90%+ solve, 2-3 gen discovery, 80%+ retention
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


# Oscillatory indices - protected with sticky deactivation
OSCILLATORY_INDICES = [4, 11, 12, 13, 15]  # sin, burst, resonator, osc_adapt, receptive


class NeuroHebbianStrategy(PaletteEvolutionStrategy):
    """Hybrid strategy combining neuromodulation with Hebbian learning.

    Neuromodulation (from Strategy 7):
    - Dopamine: Reward signal, reduces exploration when improving
    - Acetylcholine: Uncertainty signal, increases exploration when stagnating
    - Norepinephrine: Arousal signal, high plasticity early on

    Hebbian learning (from Strategy 8):
    - Co-occurrence matrix tracks which pairs succeed together
    - High affinity functions are more likely to be activated
    - Consolidated pairs become protected

    Integration:
    - Neuromodulators modulate the effective learning rates
    - Hebbian affinities guide which functions to activate/deactivate
    - Both mechanisms contribute to protection of useful functions
    """

    name = "neuro_hebbian"
    description = "Neuromodulation + Hebbian co-occurrence learning hybrid"

    def __init__(
        self,
        # Neuromodulation parameters (from Strategy 7)
        base_activate_rate: float = 0.25,
        base_deactivate_rate: float = 0.05,
        dopamine_sensitivity: float = 0.5,
        acetylcholine_sensitivity: float = 0.3,
        norepinephrine_sensitivity: float = 0.2,
        modulation_ema_alpha: float = 0.3,
        stagnation_threshold: int = 3,
        deactivate_sticky_rate: float = 0.01,
        # Hebbian parameters (from Strategy 8)
        learning_rate: float = 0.1,
        anti_hebbian_rate: float = 0.05,
        consolidation_threshold: float = 0.7,
        consolidation_gens: int = 5,
        hebbian_influence: float = 0.5,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize hybrid strategy.

        Args:
            base_activate_rate: Base probability of activating inactive function
            base_deactivate_rate: Base probability of deactivating active function
            dopamine_sensitivity: How much reward reduces exploration (0-1)
            acetylcholine_sensitivity: How much uncertainty increases exploration (0-1)
            norepinephrine_sensitivity: How much arousal affects plasticity (0-1)
            modulation_ema_alpha: EMA smoothing for neuromodulator levels
            stagnation_threshold: Gens without improvement to trigger ACh boost
            deactivate_sticky_rate: Rate for oscillatory functions (much lower)
            learning_rate: Hebbian weight update rate
            anti_hebbian_rate: Anti-Hebbian (failure) update rate
            consolidation_threshold: Weight threshold for "strong" pairs
            consolidation_gens: Gens of high weight to consolidate
            hebbian_influence: How much associations affect mutation rates
            min_active: Minimum active functions
            initial_palette: Starting palette indices
        """
        # Neuromodulation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.dopamine_sensitivity = dopamine_sensitivity
        self.acetylcholine_sensitivity = acetylcholine_sensitivity
        self.norepinephrine_sensitivity = norepinephrine_sensitivity
        self.modulation_ema_alpha = modulation_ema_alpha
        self.stagnation_threshold = stagnation_threshold
        self.deactivate_sticky_rate = deactivate_sticky_rate
        # Hebbian
        self.learning_rate = learning_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_gens = consolidation_gens
        self.hebbian_influence = hebbian_influence
        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize combined state from both strategies."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Hebbian weight matrix (from Strategy 8)
        hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        consolidation_counts = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.int32)
        protected_pairs = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.bool_)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 101010),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'prev_fitness': 0.0,
            'strategy_name': self.name,
            # Neuromodulator levels (from Strategy 7)
            'dopamine': 0.5,
            'acetylcholine': 0.5,
            'norepinephrine': 1.0,
            # Hebbian state (from Strategy 8)
            'hebbian_weights': hebbian_weights,
            'consolidation_counts': consolidation_counts,
            'protected_pairs': protected_pairs,
            # Shared tracking
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
        """Update neuromodulator levels (from Strategy 7)."""
        alpha = self.modulation_ema_alpha

        # Dopamine: Reward signal
        improvement = best_fitness - prev_best_fitness
        if prev_best_fitness > 0:
            relative_improvement = improvement / prev_best_fitness
        else:
            relative_improvement = improvement
        da_signal = max(0, min(1, 0.5 + relative_improvement * 10))
        new_dopamine = (1 - alpha) * state['dopamine'] + alpha * da_signal

        # Acetylcholine: Uncertainty signal
        stagnation = state['stagnation_count'] / max(self.stagnation_threshold, 1)
        ach_signal = min(1.0, stagnation)
        new_acetylcholine = (1 - alpha) * state['acetylcholine'] + alpha * ach_signal

        # Norepinephrine: Arousal signal
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
    ) -> Tuple[float, float]:
        """Compute neuromodulated rates (from Strategy 7)."""
        da_factor = 1.0 - self.dopamine_sensitivity * (dopamine - 0.5)
        ach_factor = 1.0 + self.acetylcholine_sensitivity * (acetylcholine - 0.5)
        ne_factor = 0.5 + self.norepinephrine_sensitivity * norepinephrine

        effective_activate = self.base_activate_rate * da_factor * ach_factor * ne_factor
        effective_deactivate = self.base_deactivate_rate * (1.0 / max(da_factor, 0.5)) * ne_factor

        effective_activate = max(0.05, min(0.5, effective_activate))
        effective_deactivate = max(0.01, min(0.2, effective_deactivate))

        return effective_activate, effective_deactivate

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        norepinephrine: float,
    ) -> jnp.ndarray:
        """Apply Hebbian update, modulated by norepinephrine.

        NE modulates learning rate: High NE = faster learning (early exploration).
        """
        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        # NE modulates learning rate
        ne_modulated_lr = self.learning_rate * (0.5 + norepinephrine * 0.5)
        ne_modulated_anti = self.anti_hebbian_rate * (0.5 + norepinephrine * 0.5)

        if fitness_signal >= 0:
            delta = ne_modulated_lr * fitness_signal * co_active
        else:
            delta = ne_modulated_anti * fitness_signal * co_active

        new_weights = jnp.clip(weights + delta, 0.0, 1.0)
        return new_weights

    def _update_consolidation(
        self,
        weights: jnp.ndarray,
        consolidation_counts: jnp.ndarray,
        protected_pairs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update consolidation tracking (from Strategy 8)."""
        strong = weights >= self.consolidation_threshold
        new_counts = jnp.where(strong, consolidation_counts + 1, 0)
        newly_protected = new_counts >= self.consolidation_gens
        new_protected = jnp.logical_or(protected_pairs, newly_protected)
        return new_counts, new_protected

    def _compute_affinity_scores(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Hebbian affinity scores (from Strategy 8)."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        affinities = jnp.dot(weights, active) / n_active
        return affinities

    def _mutate_palette_neuro_hebbian(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        protected_pairs: jnp.ndarray,
        effective_activate: float,
        effective_deactivate: float,
        dopamine: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply combined neuromodulated + Hebbian mutation.

        - Neuromodulators set base rates
        - Hebbian affinities bias which functions to try
        - Both oscillatory (sticky) and Hebbian-protected pairs are protected
        - High dopamine (reward) protects the current palette
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            affinity = float(affinities[i])

            if mask[i] < 0.5:
                # Inactive - might activate
                # Combine neuromodulated rate with Hebbian affinity
                rate = effective_activate * (1 + self.hebbian_influence * (affinity - 0.5))
                rate = max(0.05, min(0.6, rate))

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                # Check protection from multiple sources

                # 1. Oscillatory sticky protection
                if i in OSCILLATORY_INDICES:
                    base_deact = self.deactivate_sticky_rate
                else:
                    base_deact = effective_deactivate

                # 2. Hebbian pair protection
                is_hebbian_protected = False
                for j in range(NUM_ACTIVATIONS):
                    if j != i and mask[j] > 0.5:
                        if protected_pairs[i, j] or protected_pairs[j, i]:
                            is_hebbian_protected = True
                            break

                if is_hebbian_protected:
                    continue

                # 3. Dopamine-based protection (high reward = don't change)
                da_protection = dopamine * 0.5  # High DA reduces deactivation
                rate = base_deact * (1 - da_protection)

                # Hebbian affinity also affects deactivation
                rate = rate * (1 + self.hebbian_influence * (0.5 - affinity))
                rate = max(0.005, min(0.3, rate))

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
            'avg_affinity': float(jnp.mean(affinities)),
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
        """Update combining both mechanisms.

        1. Update neuromodulators based on learning progress
        2. Compute Hebbian fitness signal and update weights
        3. Update consolidation
        4. Apply combined neuro-hebbian mutation
        """
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # 1. Update neuromodulators
        neuromodulators = self._update_neuromodulators(
            state, best_fitness, prev_best_fitness, generation
        )

        # 2. Compute effective rates
        effective_activate, effective_deactivate = self._compute_effective_rates(
            neuromodulators['dopamine'],
            neuromodulators['acetylcholine'],
            neuromodulators['norepinephrine'],
        )

        # 3. Hebbian fitness signal
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 10:
            fitness_history = fitness_history[-10:]

        baseline = sum(fitness_history) / len(fitness_history)
        fitness_signal = (best_fitness - baseline) / max(0.1, baseline)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # 4. Hebbian update (modulated by NE)
        new_weights = self._hebbian_update(
            state['hebbian_weights'],
            state['mask'],
            fitness_signal,
            neuromodulators['norepinephrine'],
        )

        # 5. Update consolidation
        new_counts, new_protected = self._update_consolidation(
            new_weights,
            state['consolidation_counts'],
            state['protected_pairs'],
        )

        # 6. Compute affinities
        affinities = self._compute_affinity_scores(new_weights, state['mask'])

        # 7. Apply combined mutation
        new_mask, mutation_info = self._mutate_palette_neuro_hebbian(
            subkey,
            state['mask'],
            affinities,
            new_protected,
            effective_activate,
            effective_deactivate,
            neuromodulators['dopamine'],
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'prev_fitness': best_fitness,
            'strategy_name': self.name,
            # Neuromodulators
            'dopamine': neuromodulators['dopamine'],
            'acetylcholine': neuromodulators['acetylcholine'],
            'norepinephrine': neuromodulators['norepinephrine'],
            # Hebbian
            'hebbian_weights': new_weights,
            'consolidation_counts': new_counts,
            'protected_pairs': new_protected,
            'fitness_history': fitness_history,
            'improvement_rate_ema': state['improvement_rate_ema'],
        }

        # Stats
        n_protected = int(jnp.sum(new_protected) / 2)
        avg_weight = float(jnp.mean(new_weights))

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': mask_to_indices(new_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neuromodulators
            'dopamine': neuromodulators['dopamine'],
            'acetylcholine': neuromodulators['acetylcholine'],
            'norepinephrine': neuromodulators['norepinephrine'],
            # Hebbian
            'fitness_signal': fitness_signal,
            'n_protected_pairs': n_protected,
            'avg_hebbian_weight': avg_weight,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return combined state summary."""
        palette = self.get_active_palette(state)
        weights = state['hebbian_weights']

        strongest_val = float(jnp.max(weights))
        strongest_idx = jnp.unravel_index(jnp.argmax(weights), weights.shape)

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'has_osc_adapt': 13 in palette,
            'has_receptive': 15 in palette,
            'stagnation_count': state['stagnation_count'],
            # Neuromodulators
            'dopamine': state['dopamine'],
            'acetylcholine': state['acetylcholine'],
            'norepinephrine': state['norepinephrine'],
            # Hebbian
            'strongest_pair': (int(strongest_idx[0]), int(strongest_idx[1])),
            'strongest_weight': strongest_val,
            'n_protected': int(jnp.sum(state['protected_pairs']) / 2),
        }
