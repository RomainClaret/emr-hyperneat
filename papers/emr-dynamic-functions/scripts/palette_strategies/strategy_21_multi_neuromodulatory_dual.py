"""Strategy 21 Dual: Multi-Neuromodulatory for Both Activation AND Aggregation.

Extends MultiNeuromodulatory to jointly evolve both activation and aggregation palettes
using the full four-neuromodulator system with interactions.

Biological Basis:
- Acetylcholine (ACh): Attention, focus, precision of processing
- Dopamine (DA): Reward prediction, motivation, reinforcement learning
- Norepinephrine (NE): Arousal, urgency, fight-or-flight, exploration
- Serotonin (5-HT): Mood, patience, long-term stability, impulse control

Key Extension:
- Neuromodulators affect BOTH activation AND aggregation learning equally
- Cross-domain affinity learns which act-agg combinations succeed together
- Same neuromodulator state drives both domains (biologically realistic)
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


class MultiNeuromodulatoryDualStrategy(PaletteEvolutionStrategy):
    """Full neuromodulatory system for dual palette evolution.

    Extends MultiNeuromodulatory to jointly learn:
    1. Which activations are valuable
    2. Which aggregations are valuable
    3. Which activation-aggregation combinations are synergistic
    """

    name = "multi_neuromodulatory_dual"
    description = "Dual palette with 4-neuromodulator system (ACh/DA/NE/5-HT)"

    def __init__(
        self,
        # Neuromodulator baseline levels
        ach_baseline: float = 0.5,
        da_baseline: float = 0.5,
        ne_baseline: float = 0.5,
        serotonin_baseline: float = 0.5,
        # Neuromodulator sensitivity
        ach_sensitivity: float = 0.3,
        da_sensitivity: float = 0.4,
        ne_sensitivity: float = 0.35,
        serotonin_sensitivity: float = 0.2,
        # Interaction weights
        ach_da_synergy: float = 0.2,
        ne_5ht_opposition: float = 0.3,
        da_to_ach: float = 0.15,
        serotonin_to_da: float = -0.1,
        # Neuromodulator decay rates
        ach_decay: float = 0.1,
        da_decay: float = 0.15,
        ne_decay: float = 0.12,
        serotonin_decay: float = 0.05,
        # Behavioral effects
        base_mutation_rate: float = 0.15,
        base_learning_rate: float = 0.12,
        base_retention_rate: float = 0.5,
        # Cross-domain learning
        cross_learning_rate: float = 0.10,
        cross_influence: float = 0.3,
        # Protection
        affinity_protection_threshold: float = 0.55,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize dual Multi-Neuromodulatory strategy."""
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

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual neuromodulator system."""
        # Activation palette
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        act_co_occurrence = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS))

        # Aggregation palette
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_co_occurrence = jnp.zeros((NUM_AGGREGATIONS, NUM_AGGREGATIONS))

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation state
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_co_occurrence': act_co_occurrence,
            # Aggregation state
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_co_occurrence': agg_co_occurrence,
            # Cross-domain state
            'cross_affinity': cross_affinity,
            # Common state
            'rng_key': jax.random.PRNGKey(seed + 212121),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Neuromodulator levels (shared across domains)
            'acetylcholine': self.ach_baseline,
            'dopamine': self.da_baseline,
            'norepinephrine': self.ne_baseline,
            'serotonin': self.serotonin_baseline,
            # Tracking
            'fitness_history': [],
            'fitness_ema': 0.5,
            'reward_prediction': 0.0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

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
        """Update neuromodulator levels based on evolutionary state."""
        # ACh: Increases with consistent improvement
        if improved:
            ach_delta = self.ach_sensitivity * 0.3
        else:
            ach_delta = -self.ach_sensitivity * 0.1

        # DA from ACh interaction
        ach_boost_to_da = self.da_to_ach * ach

        # DA: Reward prediction error
        da_delta = self.da_sensitivity * fitness_signal + ach_boost_to_da
        da_delta += self.serotonin_to_da * serotonin

        # NE: Increases with stagnation
        if stagnation > 5:
            ne_delta = self.ne_sensitivity * 0.4 * (stagnation / 20)
        elif improved:
            ne_delta = -self.ne_sensitivity * 0.2
        else:
            ne_delta = 0.0

        # 5-HT: Long-term stability
        if fitness_signal > 0.2:
            serotonin_delta = self.serotonin_sensitivity * 0.2
        elif fitness_signal < -0.2:
            serotonin_delta = -self.serotonin_sensitivity * 0.1
        else:
            serotonin_delta = 0.0

        # NE-5HT opposition
        ne_5ht_effect = self.ne_5ht_opposition * (ne - serotonin)
        ne_delta += ne_5ht_effect * 0.5
        serotonin_delta -= ne_5ht_effect * 0.5

        # ACh-DA synergy
        ach_da_effect = self.ach_da_synergy * ach * da
        da_delta += ach_da_effect * 0.3

        # Decay toward baseline
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
        }

        return new_ach, new_da, new_ne, new_serotonin, metrics

    def _compute_behavioral_modulation(
        self,
        ach: float,
        da: float,
        ne: float,
        serotonin: float,
    ) -> Tuple[float, float, float]:
        """Compute behavioral parameters from neuromodulator levels."""
        # Exploration rate
        exploration_rate = self.base_mutation_rate * (
            1.0 + 0.5 * (ne - 0.5) - 0.3 * (serotonin - 0.5)
        )
        exploration_rate = max(0.05, min(0.4, exploration_rate))

        # Learning rate
        learning_rate = self.base_learning_rate * (
            1.0 + 0.6 * (da - 0.5) + 0.3 * (ach - 0.5)
        )
        learning_rate = max(0.05, min(0.3, learning_rate))

        # Retention rate
        retention_rate = self.base_retention_rate * (
            1.0 + 0.4 * (serotonin - 0.5) - 0.2 * (ne - 0.5)
        )
        retention_rate = max(0.3, min(0.8, retention_rate))

        return exploration_rate, learning_rate, retention_rate

    def _update_affinity_dual(
        self,
        act_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_signal: float,
        learning_rate: float,
        ach: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update affinities for both domains with cross-domain learning."""
        precision = 0.5 + 0.5 * ach
        cross_lr = self.cross_learning_rate * precision

        # Active masks
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)

        # Update activation affinity
        new_act_affinity = act_affinity.copy()
        for i in range(NUM_ACTIVATIONS):
            if float(act_active[i]) > 0.5:
                if fitness_signal >= 0:
                    delta = learning_rate * precision * fitness_signal
                else:
                    delta = learning_rate * 0.3 * precision * fitness_signal
                new_act_affinity = new_act_affinity.at[i].set(
                    max(0.05, min(0.95, float(new_act_affinity[i]) + delta))
                )

        # Update aggregation affinity
        new_agg_affinity = agg_affinity.copy()
        for i in range(NUM_AGGREGATIONS):
            if float(agg_active[i]) > 0.5:
                if fitness_signal >= 0:
                    delta = learning_rate * precision * fitness_signal
                else:
                    delta = learning_rate * 0.3 * precision * fitness_signal
                new_agg_affinity = new_agg_affinity.at[i].set(
                    max(0.05, min(0.95, float(new_agg_affinity[i]) + delta))
                )

        # Update cross-domain affinity
        cross_active = jnp.outer(act_active, agg_active)
        if fitness_signal >= 0:
            cross_delta = cross_lr * fitness_signal * cross_active
        else:
            cross_delta = cross_lr * 0.3 * fitness_signal * cross_active
        new_cross_affinity = jnp.clip(cross_affinity + cross_delta, 0.0, 1.0)

        return new_act_affinity, new_agg_affinity, new_cross_affinity

    def _update_co_occurrence_dual(
        self,
        act_co_occurrence: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_co_occurrence: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_signal: float,
        ach: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update co-occurrence for both domains."""
        if fitness_signal <= 0:
            return act_co_occurrence, agg_co_occurrence

        lr = 0.1 * (0.5 + 0.5 * ach)

        # Activation co-occurrence
        act_active = (act_mask > 0.5).astype(jnp.float32)
        act_co_active = jnp.outer(act_active, act_active)
        new_act_co = jnp.clip(act_co_occurrence + lr * fitness_signal * act_co_active, 0.0, 1.0)

        # Aggregation co-occurrence
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        agg_co_active = jnp.outer(agg_active, agg_active)
        new_agg_co = jnp.clip(agg_co_occurrence + lr * fitness_signal * agg_co_active, 0.0, 1.0)

        return new_act_co, new_agg_co

    def _compute_protection_act(
        self,
        act_affinity: jnp.ndarray,
        act_co_occurrence: jnp.ndarray,
        act_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
        serotonin: float,
    ) -> jnp.ndarray:
        """Compute protection scores for activations with cross-domain influence."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_act = max(jnp.sum(act_active), 1)
        n_agg = max(jnp.sum(agg_active), 1)

        pairwise_score = jnp.dot(act_co_occurrence, act_active) / n_act
        cross_score = jnp.dot(cross_affinity, agg_active) / n_agg

        protection = (
            0.55 * act_affinity +
            0.30 * pairwise_score +
            0.15 * cross_score * self.cross_influence
        )

        # 5-HT boosts protection for high-affinity functions
        serotonin_boost = serotonin * 0.1
        protection = jnp.where(
            act_affinity > self.affinity_protection_threshold,
            protection + serotonin_boost,
            protection
        )

        return jnp.clip(protection, 0.0, 1.0)

    def _compute_protection_agg(
        self,
        agg_affinity: jnp.ndarray,
        agg_co_occurrence: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        serotonin: float,
    ) -> jnp.ndarray:
        """Compute protection scores for aggregations with cross-domain influence."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        n_act = max(jnp.sum(act_active), 1)
        n_agg = max(jnp.sum(agg_active), 1)

        pairwise_score = jnp.dot(agg_co_occurrence, agg_active) / n_agg
        cross_score = jnp.dot(cross_affinity.T, act_active) / n_act

        protection = (
            0.55 * agg_affinity +
            0.30 * pairwise_score +
            0.15 * cross_score * self.cross_influence
        )

        # 5-HT boosts protection
        serotonin_boost = serotonin * 0.1
        protection = jnp.where(
            agg_affinity > self.affinity_protection_threshold,
            protection + serotonin_boost,
            protection
        )

        return jnp.clip(protection, 0.0, 1.0)

    def _mutate_activation_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        exploration_rate: float,
        retention_rate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation to activation palette with max constraint."""
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        current_active = int(jnp.sum(mask > 0.5))

        for i in range(NUM_ACTIVATIONS):
            prot = float(protection[i])

            if mask[i] < 0.5:
                # Skip if at max
                if current_active + len(activated) >= self.max_active_act:
                    continue
                effective_rate = exploration_rate * (0.5 + prot)
                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if prot >= self.affinity_protection_threshold:
                    deact_rate = exploration_rate * (1.0 - retention_rate) * 0.2
                else:
                    deact_rate = exploration_rate * (1.0 - retention_rate) * (1.0 - prot)

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum
        if jnp.sum(new_mask > 0.5) < self.min_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'act_activated': activated, 'act_deactivated': deactivated}

    def _mutate_aggregation_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        exploration_rate: float,
        retention_rate: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation to aggregation palette with max constraint."""
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_AGGREGATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_AGGREGATIONS,))

        current_active = int(jnp.sum(mask > 0.5))

        for i in range(NUM_AGGREGATIONS):
            prot = float(protection[i])

            if mask[i] < 0.5:
                # Skip if at max
                if current_active + len(activated) >= self.max_active_agg:
                    continue
                effective_rate = exploration_rate * (0.5 + prot)
                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if prot >= self.affinity_protection_threshold:
                    deact_rate = exploration_rate * (1.0 - retention_rate) * 0.2
                else:
                    deact_rate = exploration_rate * (1.0 - retention_rate) * (1.0 - prot)

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum
        if jnp.sum(new_mask > 0.5) < self.min_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'agg_activated': activated, 'agg_deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual neuromodulatory system."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Fitness signal
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Step 1: Update neuromodulators (shared across domains)
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

        # Step 3: Update affinities (both domains + cross)
        new_act_affinity, new_agg_affinity, new_cross_affinity = self._update_affinity_dual(
            state['act_affinity'],
            state['act_mask'],
            state['agg_affinity'],
            state['agg_mask'],
            state['cross_affinity'],
            fitness_signal,
            learning_rate,
            new_ach,
        )

        # Step 4: Update co-occurrence
        new_act_co, new_agg_co = self._update_co_occurrence_dual(
            state['act_co_occurrence'],
            state['act_mask'],
            state['agg_co_occurrence'],
            state['agg_mask'],
            fitness_signal,
            new_ach,
        )

        # Step 5: Compute protection (with cross-domain influence)
        act_protection = self._compute_protection_act(
            new_act_affinity, new_act_co, state['act_mask'],
            new_cross_affinity, state['agg_mask'], new_serotonin
        )
        agg_protection = self._compute_protection_agg(
            new_agg_affinity, new_agg_co, state['agg_mask'],
            new_cross_affinity, state['act_mask'], new_serotonin
        )

        # Step 6: Apply mutations to both palettes
        new_act_mask, act_mutation = self._mutate_activation_palette(
            key_act, state['act_mask'], act_protection, exploration_rate, retention_rate
        )
        new_agg_mask, agg_mutation = self._mutate_aggregation_palette(
            key_agg, state['agg_mask'], agg_protection, exploration_rate, retention_rate
        )

        act_palette_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_palette_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Update fitness history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_affinity': new_act_affinity,
            'act_co_occurrence': new_act_co,
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_affinity,
            'agg_co_occurrence': new_agg_co,
            'cross_affinity': new_cross_affinity,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'acetylcholine': new_ach,
            'dopamine': new_da,
            'norepinephrine': new_ne,
            'serotonin': new_serotonin,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'reward_prediction': new_fitness_ema,
        }

        active_act_palette = mask_to_indices(new_act_mask)
        active_agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'act_palette_changed': act_palette_changed,
            'agg_palette_changed': agg_palette_changed,
            'current_act_palette': active_act_palette,
            'current_agg_palette': active_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neuromodulators
            'acetylcholine': new_ach,
            'dopamine': new_da,
            'norepinephrine': new_ne,
            'serotonin': new_serotonin,
            # Behavioral
            'exploration_rate': exploration_rate,
            'learning_rate': learning_rate,
            'retention_rate': retention_rate,
            # Activation stats
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'act_max_affinity': float(jnp.max(new_act_affinity)),
            'sin_affinity': float(new_act_affinity[4]),
            # Aggregation stats
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'agg_max_affinity': float(jnp.max(new_agg_affinity)),
            # Cross-domain stats
            'cross_avg_affinity': float(jnp.mean(new_cross_affinity)),
            'cross_max_affinity': float(jnp.max(new_cross_affinity)),
        }
        metrics.update(act_mutation)
        metrics.update(agg_mutation)
        metrics.update({f'neuromod_{k}': v for k, v in neuromod_metrics.items()})

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual palette and neuromodulator info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        exploration, learning, retention = self._compute_behavioral_modulation(
            state['acetylcholine'],
            state['dopamine'],
            state['norepinephrine'],
            state['serotonin'],
        )

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'act_palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
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
            # Affinities
            'sin_affinity': float(state['act_affinity'][4]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
        }
