"""Strategy 21: Multi-Neuromodulatory (Full Neuromodulator System).

Implements the full four-neuromodulator system with interactions.

Biological Basis:
- Acetylcholine (ACh): Attention, focus, precision of processing
- Dopamine (DA): Reward prediction, motivation, reinforcement learning
- Norepinephrine (NE): Arousal, urgency, fight-or-flight, exploration
- Serotonin (5-HT): Mood, patience, long-term stability, impulse control

Key Interactions:
- ACh ↔ DA synergy: Attention amplifies reward signals
- NE ↔ 5-HT opposition: Urgency vs patience trade-off
- DA → ACh: High reward increases attention
- 5-HT → DA: High serotonin dampens reward sensitivity

For palette evolution:
- ACh: Controls precision of affinity updates (low = broad, high = precise)
- DA: Controls learning rate based on reward prediction error
- NE: Controls exploration rate (high = more mutation)
- 5-HT: Controls retention rate (high = more stable, less change)

Key mechanisms:
1. Neuromodulator dynamics: Each updates based on evolutionary state
2. Interaction effects: Neuromodulators influence each other
3. Behavioral modulation: Combined levels control exploration/exploitation
4. State-dependent plasticity: Learning rules change based on neuromodulator balance

Expected improvement over Strategy 7 (simple neuromodulation):
- More nuanced exploration/exploitation balance
- Better adaptation to different evolutionary phases
- More stable long-term behavior (5-HT influence)
- Better attention mechanisms (ACh influence)
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


class MultiNeuromodulatoryStrategy(PaletteEvolutionStrategy):
    """Full neuromodulatory system: ACh, DA, NE, 5-HT with interactions.

    Models the four major neuromodulators and their effects on palette evolution.
    """

    name = "multi_neuromodulatory"
    description = "Full neuromodulatory system with ACh/DA/NE/5-HT interactions"

    def __init__(
        self,
        # Neuromodulator baseline levels
        ach_baseline: float = 0.5,      # Acetylcholine baseline
        da_baseline: float = 0.5,       # Dopamine baseline
        ne_baseline: float = 0.5,       # Norepinephrine baseline
        serotonin_baseline: float = 0.5,  # Serotonin (5-HT) baseline
        # Neuromodulator sensitivity
        ach_sensitivity: float = 0.3,   # How quickly ACh responds
        da_sensitivity: float = 0.4,    # How quickly DA responds
        ne_sensitivity: float = 0.35,   # How quickly NE responds
        serotonin_sensitivity: float = 0.2,  # 5-HT is slow to change
        # Interaction weights
        ach_da_synergy: float = 0.2,    # ACh amplifies DA effects
        ne_5ht_opposition: float = 0.3,  # NE and 5-HT oppose each other
        da_to_ach: float = 0.15,        # DA boosts ACh
        serotonin_to_da: float = -0.1,   # 5-HT dampens DA
        # Neuromodulator decay rates (toward baseline)
        ach_decay: float = 0.1,
        da_decay: float = 0.15,
        ne_decay: float = 0.12,
        serotonin_decay: float = 0.05,  # 5-HT is slow to decay
        # Behavioral effects
        base_mutation_rate: float = 0.15,
        base_learning_rate: float = 0.12,
        base_retention_rate: float = 0.5,
        # Protection
        affinity_protection_threshold: float = 0.55,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Multi-Neuromodulatory strategy.

        Args:
            ach_baseline: Baseline acetylcholine level
            da_baseline: Baseline dopamine level
            ne_baseline: Baseline norepinephrine level
            serotonin_baseline: Baseline serotonin level
            ach_sensitivity: ACh response speed
            da_sensitivity: DA response speed
            ne_sensitivity: NE response speed
            serotonin_sensitivity: 5-HT response speed
            ach_da_synergy: How much ACh amplifies DA effects
            ne_5ht_opposition: How much NE and 5-HT oppose
        """
        # Baselines
        self.ach_baseline = ach_baseline
        self.da_baseline = da_baseline
        self.ne_baseline = ne_baseline
        self.serotonin_baseline = serotonin_baseline

        # Sensitivities
        self.ach_sensitivity = ach_sensitivity
        self.da_sensitivity = da_sensitivity
        self.ne_sensitivity = ne_sensitivity
        self.serotonin_sensitivity = serotonin_sensitivity

        # Interactions
        self.ach_da_synergy = ach_da_synergy
        self.ne_5ht_opposition = ne_5ht_opposition
        self.da_to_ach = da_to_ach
        self.serotonin_to_da = serotonin_to_da

        # Decay rates
        self.ach_decay = ach_decay
        self.da_decay = da_decay
        self.ne_decay = ne_decay
        self.serotonin_decay = serotonin_decay

        # Behavioral effects
        self.base_mutation_rate = base_mutation_rate
        self.base_learning_rate = base_learning_rate
        self.base_retention_rate = base_retention_rate

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neuromodulator system."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Function affinity
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Co-occurrence tracking
        co_occurrence = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS))

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 212121),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Neuromodulator levels
            'acetylcholine': self.ach_baseline,
            'dopamine': self.da_baseline,
            'norepinephrine': self.ne_baseline,
            'serotonin': self.serotonin_baseline,
            # Learning state
            'function_affinity': function_affinity,
            'co_occurrence': co_occurrence,
            # Tracking
            'fitness_history': [],
            'fitness_ema': 0.5,
            'reward_prediction': 0.0,  # For TD-like DA computation
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_neuromodulators(
        self,
        ach: float,
        da: float,
        ne: float,
        serotonin: float,
        fitness_signal: float,
        stagnation: int,
        improved: bool,
    ) -> Tuple[float, float, float, float, Dict]:
        """Update neuromodulator levels based on evolutionary state.

        Returns:
            (new_ach, new_da, new_ne, new_serotonin, neuromod_metrics)
        """
        # Compute raw updates based on state

        # Acetylcholine: Increases with consistent improvement (attention)
        if improved:
            ach_delta = self.ach_sensitivity * 0.3
        else:
            ach_delta = -self.ach_sensitivity * 0.1

        # DA from ACh interaction (attention boosts reward sensitivity)
        ach_boost_to_da = self.da_to_ach * ach

        # Dopamine: Reward prediction error (TD-like)
        # High when fitness exceeds expectation
        da_delta = self.da_sensitivity * fitness_signal + ach_boost_to_da

        # Serotonin dampening effect on DA
        da_delta += self.serotonin_to_da * serotonin

        # Norepinephrine: Increases with stagnation (urgency/arousal)
        if stagnation > 5:
            ne_delta = self.ne_sensitivity * 0.4 * (stagnation / 20)
        elif improved:
            ne_delta = -self.ne_sensitivity * 0.2  # Calm down on success
        else:
            ne_delta = 0.0

        # Serotonin: Increases with long-term success (stability/patience)
        # High when things are generally good
        if fitness_signal > 0.2:
            serotonin_delta = self.serotonin_sensitivity * 0.2
        elif fitness_signal < -0.2:
            serotonin_delta = -self.serotonin_sensitivity * 0.1
        else:
            serotonin_delta = 0.0

        # Apply NE-5HT opposition
        ne_5ht_effect = self.ne_5ht_opposition * (ne - serotonin)
        ne_delta += ne_5ht_effect * 0.5
        serotonin_delta -= ne_5ht_effect * 0.5

        # Apply ACh-DA synergy
        ach_da_effect = self.ach_da_synergy * ach * da
        da_delta += ach_da_effect * 0.3

        # Apply decay toward baseline
        ach_decay_delta = self.ach_decay * (self.ach_baseline - ach)
        da_decay_delta = self.da_decay * (self.da_baseline - da)
        ne_decay_delta = self.ne_decay * (self.ne_baseline - ne)
        serotonin_decay_delta = self.serotonin_decay * (self.serotonin_baseline - serotonin)

        # Compute new levels
        new_ach = max(0.1, min(0.9, ach + ach_delta + ach_decay_delta))
        new_da = max(0.1, min(0.9, da + da_delta + da_decay_delta))
        new_ne = max(0.1, min(0.9, ne + ne_delta + ne_decay_delta))
        new_serotonin = max(0.1, min(0.9, serotonin + serotonin_delta + serotonin_decay_delta))

        metrics = {
            'ach_delta': ach_delta,
            'da_delta': da_delta,
            'ne_delta': ne_delta,
            'serotonin_delta': serotonin_delta,
            'ne_5ht_effect': ne_5ht_effect,
            'ach_da_synergy_effect': ach_da_effect,
        }

        return new_ach, new_da, new_ne, new_serotonin, metrics

    def _compute_behavioral_modulation(
        self,
        ach: float,
        da: float,
        ne: float,
        serotonin: float,
    ) -> Tuple[float, float, float]:
        """Compute behavioral parameters from neuromodulator levels.

        Returns:
            (exploration_rate, learning_rate, retention_rate)
        """
        # Exploration rate: Driven by NE, dampened by 5-HT
        # High NE = more exploration/mutation
        # High 5-HT = more patience/stability
        exploration_rate = self.base_mutation_rate * (
            1.0 + 0.5 * (ne - 0.5) - 0.3 * (serotonin - 0.5)
        )
        exploration_rate = max(0.05, min(0.4, exploration_rate))

        # Learning rate: Driven by DA (reward signal), modulated by ACh (attention)
        # High DA = more learning from success
        # High ACh = more precise learning
        learning_rate = self.base_learning_rate * (
            1.0 + 0.6 * (da - 0.5) + 0.3 * (ach - 0.5)
        )
        learning_rate = max(0.05, min(0.3, learning_rate))

        # Retention rate: Driven by 5-HT (stability), dampened by NE (change)
        # High 5-HT = more retention of current state
        # High NE = more willingness to change
        retention_rate = self.base_retention_rate * (
            1.0 + 0.4 * (serotonin - 0.5) - 0.2 * (ne - 0.5)
        )
        retention_rate = max(0.3, min(0.8, retention_rate))

        return exploration_rate, learning_rate, retention_rate

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        learning_rate: float,
        ach: float,  # For precision modulation
    ) -> jnp.ndarray:
        """Update affinity with neuromodulator-modulated learning.

        ACh modulates precision: high ACh = more focused updates.
        """
        new_affinity = affinity.copy()
        active = (mask > 0.5).astype(jnp.float32)

        # ACh modulates update precision
        # High ACh: Stronger updates for active functions, weaker for inactive
        precision = 0.5 + 0.5 * ach

        for i in range(NUM_ACTIVATIONS):
            if float(active[i]) > 0.5:
                if fitness_signal >= 0:
                    # Positive learning
                    delta = learning_rate * precision * fitness_signal
                else:
                    # Negative learning (slower)
                    delta = learning_rate * 0.3 * precision * fitness_signal

                new_affinity = new_affinity.at[i].set(
                    max(0.05, min(0.95, float(new_affinity[i]) + delta))
                )

        return new_affinity

    def _update_co_occurrence(
        self,
        co_occurrence: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        ach: float,
    ) -> jnp.ndarray:
        """Update co-occurrence with ACh-modulated learning."""
        if fitness_signal <= 0:
            return co_occurrence

        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        # ACh modulates co-occurrence learning precision
        lr = 0.1 * (0.5 + 0.5 * ach)
        delta = lr * fitness_signal * co_active

        return jnp.clip(co_occurrence + delta, 0.0, 1.0)

    def _compute_protection(
        self,
        affinity: jnp.ndarray,
        co_occurrence: jnp.ndarray,
        mask: jnp.ndarray,
        serotonin: float,
    ) -> jnp.ndarray:
        """Compute protection scores with 5-HT modulated stability.

        High 5-HT = more protective of established functions.
        """
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        # Pairwise score from co-occurrence
        pairwise_score = jnp.dot(co_occurrence, active) / n_active

        # Base protection
        protection = 0.6 * affinity + 0.4 * pairwise_score

        # 5-HT boosts protection for high-affinity functions
        # (patience/stability favors protecting what works)
        serotonin_boost = serotonin * 0.1
        boosted = jnp.where(
            affinity > self.affinity_protection_threshold,
            protection + serotonin_boost,
            protection
        )

        return jnp.clip(boosted, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        exploration_rate: float,
        retention_rate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with neuromodulator-controlled rates."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        protection_info = {}

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            prot = float(protection[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                # Exploration rate modulates activation
                effective_rate = exploration_rate * (0.5 + prot)
                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                # Retention rate modulates deactivation resistance
                if prot >= self.affinity_protection_threshold:
                    # Protected by high affinity
                    deact_rate = exploration_rate * (1.0 - retention_rate) * 0.2
                    protection_info[i] = f"protected (prot={prot:.2f})"
                else:
                    # Vulnerable
                    deact_rate = exploration_rate * (1.0 - retention_rate) * (1.0 - prot)

                if deactivate_probs[i] < deact_rate:
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
            'protection_info': protection_info,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with multi-neuromodulatory system."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Compute fitness signal (reward prediction error for DA)
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Step 1: Update neuromodulator levels
        new_ach, new_da, new_ne, new_serotonin, neuromod_metrics = self._update_neuromodulators(
            state['acetylcholine'],
            state['dopamine'],
            state['norepinephrine'],
            state['serotonin'],
            fitness_signal,
            new_stagnation,
            improved,
        )

        # Step 2: Compute behavioral modulation
        exploration_rate, learning_rate, retention_rate = self._compute_behavioral_modulation(
            new_ach, new_da, new_ne, new_serotonin
        )

        # Step 3: Update affinity with modulated learning
        new_affinity = self._update_affinity(
            state['function_affinity'],
            state['mask'],
            fitness_signal,
            learning_rate,
            new_ach,
        )

        # Step 4: Update co-occurrence
        new_co_occurrence = self._update_co_occurrence(
            state['co_occurrence'],
            state['mask'],
            fitness_signal,
            new_ach,
        )

        # Step 5: Compute protection with 5-HT modulation
        protection = self._compute_protection(
            new_affinity,
            new_co_occurrence,
            state['mask'],
            new_serotonin,
        )

        # Step 6: Apply mutation with modulated rates
        new_mask, mutation_info = self._mutate_palette(
            subkey,
            state['mask'],
            protection,
            exploration_rate,
            retention_rate,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update fitness history
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
            # Neuromodulator levels
            'acetylcholine': new_ach,
            'dopamine': new_da,
            'norepinephrine': new_ne,
            'serotonin': new_serotonin,
            # Learning state
            'function_affinity': new_affinity,
            'co_occurrence': new_co_occurrence,
            # Tracking
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'reward_prediction': new_fitness_ema,
        }

        active_palette = mask_to_indices(new_mask)
        protected = [
            i for i in active_palette
            if protection[i] >= self.affinity_protection_threshold
        ]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neuromodulator levels
            'acetylcholine': new_ach,
            'dopamine': new_da,
            'norepinephrine': new_ne,
            'serotonin': new_serotonin,
            # Behavioral modulation
            'exploration_rate': exploration_rate,
            'learning_rate': learning_rate,
            'retention_rate': retention_rate,
            # Affinity stats
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'sin_affinity': float(new_affinity[4]),
            # Protection
            'n_protected': len(protected),
            'protected_functions': protected,
            # Neuromodulator dynamics
            **{f'neuromod_{k}': v for k, v in neuromod_metrics.items()},
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with neuromodulator stats."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']

        # Top functions by affinity
        top_indices = jnp.argsort(affinity)[-5:][::-1]
        top_affinities = [(int(i), float(affinity[i])) for i in top_indices]

        # Compute current behavioral parameters
        exploration, learning, retention = self._compute_behavioral_modulation(
            state['acetylcholine'],
            state['dopamine'],
            state['norepinephrine'],
            state['serotonin'],
        )

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'top_affinity_functions': top_affinities,
            'sin_affinity': float(affinity[4]),
            'avg_affinity': float(jnp.mean(affinity)),
            'stagnation_count': state['stagnation_count'],
            # Neuromodulator state
            'acetylcholine': state['acetylcholine'],
            'dopamine': state['dopamine'],
            'norepinephrine': state['norepinephrine'],
            'serotonin': state['serotonin'],
            # Behavioral parameters
            'exploration_rate': exploration,
            'learning_rate': learning,
            'retention_rate': retention,
        }
