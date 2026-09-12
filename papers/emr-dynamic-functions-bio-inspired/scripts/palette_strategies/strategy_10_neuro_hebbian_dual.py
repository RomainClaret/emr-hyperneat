"""Strategy 10D: NeuroHebbian Dual (Neuromodulation + Hebbian for Both Domains).

Extends NeuroHebbianStrategy to jointly evolve BOTH activation AND aggregation
function palettes using combined neuromodulation and Hebbian learning.

Cross-Domain Learning:
- Shared neuromodulators (DA/ACh/NE) affect both domains
- Separate Hebbian weight matrices for each domain
- Cross-domain Hebbian: learn which act-agg pairs succeed together
- Double protection from both mechanisms in both domains

Key Dual Mechanisms:
1. Shared neuromodulation - same DA/ACh/NE for both domains
2. Dual Hebbian matrices - separate co-occurrence learning
3. Cross-domain Hebbian matrix - act-agg pair associations
4. Dual consolidation - protected pairs in both domains
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

# Aggregation domain constants
NUM_AGGREGATIONS = 6
DEFAULT_AGGREGATION_INDICES = [0, 1]  # sum, mean

# Oscillatory function indices
ACT_OSCILLATORY = [4, 11, 12, 13, 15]  # sin, burst, resonator, osc_adapt, receptive


class NeuroHebbianDualStrategy(PaletteEvolutionStrategy):
    """Hybrid neuromodulation + Hebbian learning for dual palette evolution.

    Combines neuromodulated exploration/exploitation control with Hebbian
    co-occurrence learning in both activation and aggregation domains,
    plus cross-domain Hebbian learning for act-agg combinations.
    """

    name = "neuro_hebbian_dual"
    description = "Neuromodulation + Hebbian for both activation and aggregation"

    def __init__(
        self,
        # Neuromodulation (shared)
        base_activate_rate: float = 0.25,
        base_deactivate_rate: float = 0.05,
        dopamine_sensitivity: float = 0.5,
        acetylcholine_sensitivity: float = 0.3,
        norepinephrine_sensitivity: float = 0.2,
        modulation_ema_alpha: float = 0.3,
        stagnation_threshold: int = 3,
        sticky_deactivate_rate: float = 0.01,
        # Hebbian - activation
        act_learning_rate: float = 0.1,
        act_anti_hebbian_rate: float = 0.05,
        act_consolidation_threshold: float = 0.7,
        # Hebbian - aggregation
        agg_learning_rate: float = 0.08,
        agg_anti_hebbian_rate: float = 0.04,
        agg_consolidation_threshold: float = 0.65,
        # Cross-domain Hebbian
        cross_learning_rate: float = 0.05,
        cross_consolidation_threshold: float = 0.6,
        # Hebbian general
        consolidation_gens: int = 5,
        hebbian_influence: float = 0.5,
        # Constraints
        min_active_act: int = 2,
        max_active_act: int = 6,
        min_active_agg: int = 1,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize NeuroHebbian Dual strategy."""
        # Neuromodulation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.dopamine_sensitivity = dopamine_sensitivity
        self.acetylcholine_sensitivity = acetylcholine_sensitivity
        self.norepinephrine_sensitivity = norepinephrine_sensitivity
        self.modulation_ema_alpha = modulation_ema_alpha
        self.stagnation_threshold = stagnation_threshold
        self.sticky_deactivate_rate = sticky_deactivate_rate

        # Hebbian - activation
        self.act_learning_rate = act_learning_rate
        self.act_anti_hebbian_rate = act_anti_hebbian_rate
        self.act_consolidation_threshold = act_consolidation_threshold

        # Hebbian - aggregation
        self.agg_learning_rate = agg_learning_rate
        self.agg_anti_hebbian_rate = agg_anti_hebbian_rate
        self.agg_consolidation_threshold = agg_consolidation_threshold

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_consolidation_threshold = cross_consolidation_threshold

        # General Hebbian
        self.consolidation_gens = consolidation_gens
        self.hebbian_influence = hebbian_influence

        # Constraints
        self.min_active_act = min_active_act
        self.max_active_act = max_active_act
        self.min_active_agg = min_active_agg
        self.max_active_agg = max_active_agg

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize dual NeuroHebbian state."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_mask = agg_mask.at[i].set(1.0)

        # Hebbian matrices
        act_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        agg_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        cross_weights = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Consolidation tracking
        act_consol_counts = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.int32)
        agg_consol_counts = jnp.zeros((NUM_AGGREGATIONS, NUM_AGGREGATIONS), dtype=jnp.int32)
        cross_consol_counts = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS), dtype=jnp.int32)

        act_protected = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.bool_)
        agg_protected = jnp.zeros((NUM_AGGREGATIONS, NUM_AGGREGATIONS), dtype=jnp.bool_)
        cross_protected = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS), dtype=jnp.bool_)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_weights': act_weights,
            'act_consol_counts': act_consol_counts,
            'act_protected': act_protected,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_weights': agg_weights,
            'agg_consol_counts': agg_consol_counts,
            'agg_protected': agg_protected,
            # Cross-domain
            'cross_weights': cross_weights,
            'cross_consol_counts': cross_consol_counts,
            'cross_protected': cross_protected,
            # Neuromodulators
            'dopamine': 0.5,
            'acetylcholine': 0.5,
            'norepinephrine': 1.0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 101010),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'fitness_history': [],
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return [i for i in range(NUM_AGGREGATIONS) if state['agg_mask'][i] > 0.5]

    def _update_neuromodulators(
        self,
        state: Dict[str, Any],
        best_fitness: float,
        prev_best_fitness: float,
        generation: int,
    ) -> Dict[str, float]:
        """Update shared neuromodulator levels."""
        alpha = self.modulation_ema_alpha

        # Dopamine: reward
        improvement = best_fitness - prev_best_fitness
        rel_improvement = improvement / max(prev_best_fitness, 0.01)
        da_signal = max(0, min(1, 0.5 + rel_improvement * 10))
        new_da = (1 - alpha) * state['dopamine'] + alpha * da_signal

        # Acetylcholine: uncertainty
        stagnation = state['stagnation_count'] / max(self.stagnation_threshold, 1)
        ach_signal = min(1.0, stagnation)
        new_ach = (1 - alpha) * state['acetylcholine'] + alpha * ach_signal

        # Norepinephrine: arousal
        time_decay = max(0, 1.0 - generation / 50.0)
        challenge = 1.0 - best_fitness
        ne_signal = max(time_decay, challenge * 0.5)
        new_ne = (1 - alpha) * state['norepinephrine'] + alpha * ne_signal

        return {'dopamine': float(new_da), 'acetylcholine': float(new_ach), 'norepinephrine': float(new_ne)}

    def _compute_effective_rates(
        self,
        dopamine: float,
        acetylcholine: float,
        norepinephrine: float,
    ) -> Tuple[float, float]:
        """Compute neuromodulated rates."""
        da_factor = 1.0 - self.dopamine_sensitivity * (dopamine - 0.5)
        ach_factor = 1.0 + self.acetylcholine_sensitivity * (acetylcholine - 0.5)
        ne_factor = 0.5 + self.norepinephrine_sensitivity * norepinephrine

        eff_activate = self.base_activate_rate * da_factor * ach_factor * ne_factor
        eff_deactivate = self.base_deactivate_rate * (1.0 / max(da_factor, 0.5)) * ne_factor

        return (
            max(0.05, min(0.5, eff_activate)),
            max(0.01, min(0.2, eff_deactivate))
        )

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        ne: float,
        lr: float,
        anti_lr: float,
    ) -> jnp.ndarray:
        """Apply Hebbian update modulated by norepinephrine."""
        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        ne_lr = lr * (0.5 + ne * 0.5)
        ne_anti = anti_lr * (0.5 + ne * 0.5)

        if fitness_signal >= 0:
            delta = ne_lr * fitness_signal * co_active
        else:
            delta = ne_anti * fitness_signal * co_active

        return jnp.clip(weights + delta, 0.0, 1.0)

    def _cross_hebbian_update(
        self,
        weights: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_signal: float,
        ne: float,
    ) -> jnp.ndarray:
        """Update cross-domain Hebbian weights."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        lr = self.cross_learning_rate * (0.5 + ne * 0.5)
        delta = lr * fitness_signal * co_active

        return jnp.clip(weights + delta, 0.0, 1.0)

    def _update_consolidation(
        self,
        weights: jnp.ndarray,
        counts: jnp.ndarray,
        protected: jnp.ndarray,
        threshold: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update consolidation tracking."""
        strong = weights >= threshold
        new_counts = jnp.where(strong, counts + 1, 0)
        newly_protected = new_counts >= self.consolidation_gens
        new_protected = jnp.logical_or(protected, newly_protected)
        return new_counts, new_protected

    def _compute_affinities(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Hebbian affinity scores."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        return jnp.dot(weights, active) / n_active

    def _mutate_act(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        protected: jnp.ndarray,
        cross_protected: jnp.ndarray,
        agg_mask: jnp.ndarray,
        eff_activate: float,
        eff_deactivate: float,
        dopamine: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Mutate activation palette with NeuroHebbian dynamics."""
        k1, k2 = jax.random.split(key)
        act_probs = jax.random.uniform(k1, (NUM_ACTIVATIONS,))
        deact_probs = jax.random.uniform(k2, (NUM_ACTIVATIONS,))

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_ACTIVATIONS):
            aff = float(affinities[i])

            if mask[i] < 0.5:
                rate = eff_activate * (1 + self.hebbian_influence * (aff - 0.5))
                rate = max(0.05, min(0.6, rate))
                if act_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Check protections
                if i in ACT_OSCILLATORY:
                    base = self.sticky_deactivate_rate
                else:
                    base = eff_deactivate

                # Hebbian protection
                is_protected = False
                for j in range(NUM_ACTIVATIONS):
                    if j != i and mask[j] > 0.5 and (protected[i, j] or protected[j, i]):
                        is_protected = True
                        break

                # Cross-domain protection
                if not is_protected:
                    for j in range(NUM_AGGREGATIONS):
                        if agg_mask[j] > 0.5 and cross_protected[i, j]:
                            is_protected = True
                            break

                if is_protected:
                    continue

                da_prot = dopamine * 0.5
                rate = base * (1 - da_prot) * (1 + self.hebbian_influence * (0.5 - aff))
                rate = max(0.005, min(0.3, rate))

                if deact_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Constraints
        n_active = int(jnp.sum(new_mask > 0.5))
        if n_active < self.min_active_act or n_active > self.max_active_act:
            return mask, {'act_activated': [], 'act_deactivated': []}

        return new_mask, {'act_activated': activated, 'act_deactivated': deactivated}

    def _mutate_agg(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        protected: jnp.ndarray,
        cross_protected: jnp.ndarray,
        act_mask: jnp.ndarray,
        eff_activate: float,
        eff_deactivate: float,
        dopamine: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Mutate aggregation palette with NeuroHebbian dynamics."""
        k1, k2 = jax.random.split(key)
        act_probs = jax.random.uniform(k1, (NUM_AGGREGATIONS,))
        deact_probs = jax.random.uniform(k2, (NUM_AGGREGATIONS,))

        # Scale rates for smaller domain
        agg_activate = eff_activate * 0.8
        agg_deactivate = eff_deactivate * 0.9

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_AGGREGATIONS):
            aff = float(affinities[i])

            if mask[i] < 0.5:
                rate = agg_activate * (1 + self.hebbian_influence * (aff - 0.5))
                rate = max(0.05, min(0.5, rate))
                if act_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                is_protected = False
                for j in range(NUM_AGGREGATIONS):
                    if j != i and mask[j] > 0.5 and (protected[i, j] or protected[j, i]):
                        is_protected = True
                        break

                if not is_protected:
                    for j in range(NUM_ACTIVATIONS):
                        if act_mask[j] > 0.5 and cross_protected[j, i]:
                            is_protected = True
                            break

                if is_protected:
                    continue

                da_prot = dopamine * 0.5
                rate = agg_deactivate * (1 - da_prot) * (1 + self.hebbian_influence * (0.5 - aff))
                rate = max(0.005, min(0.25, rate))

                if deact_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        n_active = int(jnp.sum(new_mask > 0.5))
        if n_active < self.min_active_agg or n_active > self.max_active_agg:
            return mask, {'agg_activated': [], 'agg_deactivated': []}

        return new_mask, {'agg_activated': activated, 'agg_deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual NeuroHebbian dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stag = 0
            new_best = best_fitness
        else:
            new_stag = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update neuromodulators
        neuro = self._update_neuromodulators(state, best_fitness, prev_best_fitness, generation)
        eff_activate, eff_deactivate = self._compute_effective_rates(
            neuro['dopamine'], neuro['acetylcholine'], neuro['norepinephrine']
        )

        # Fitness signal
        fitness_history = (state['fitness_history'] + [best_fitness])[-10:]
        baseline = sum(fitness_history) / len(fitness_history)
        fitness_signal = (best_fitness - baseline) / max(0.1, baseline)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Hebbian updates
        new_act_weights = self._hebbian_update(
            state['act_weights'], state['act_mask'], fitness_signal,
            neuro['norepinephrine'], self.act_learning_rate, self.act_anti_hebbian_rate
        )
        new_agg_weights = self._hebbian_update(
            state['agg_weights'], state['agg_mask'], fitness_signal,
            neuro['norepinephrine'], self.agg_learning_rate, self.agg_anti_hebbian_rate
        )
        new_cross_weights = self._cross_hebbian_update(
            state['cross_weights'], state['act_mask'], state['agg_mask'],
            fitness_signal, neuro['norepinephrine']
        )

        # Consolidation updates
        new_act_counts, new_act_prot = self._update_consolidation(
            new_act_weights, state['act_consol_counts'], state['act_protected'],
            self.act_consolidation_threshold
        )
        new_agg_counts, new_agg_prot = self._update_consolidation(
            new_agg_weights, state['agg_consol_counts'], state['agg_protected'],
            self.agg_consolidation_threshold
        )
        new_cross_counts, new_cross_prot = self._update_consolidation(
            new_cross_weights, state['cross_consol_counts'], state['cross_protected'],
            self.cross_consolidation_threshold
        )

        # Compute affinities
        act_aff = self._compute_affinities(new_act_weights, state['act_mask'])
        agg_aff = self._compute_affinities(new_agg_weights, state['agg_mask'])

        # Mutate palettes
        new_act_mask, act_info = self._mutate_act(
            k1, state['act_mask'], act_aff, new_act_prot, new_cross_prot,
            state['agg_mask'], eff_activate, eff_deactivate, neuro['dopamine']
        )
        new_agg_mask, agg_info = self._mutate_agg(
            k2, state['agg_mask'], agg_aff, new_agg_prot, new_cross_prot,
            state['act_mask'], eff_activate, eff_deactivate, neuro['dopamine']
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        new_state = {
            'act_mask': new_act_mask,
            'act_weights': new_act_weights,
            'act_consol_counts': new_act_counts,
            'act_protected': new_act_prot,
            'agg_mask': new_agg_mask,
            'agg_weights': new_agg_weights,
            'agg_consol_counts': new_agg_counts,
            'agg_protected': new_agg_prot,
            'cross_weights': new_cross_weights,
            'cross_consol_counts': new_cross_counts,
            'cross_protected': new_cross_prot,
            'dopamine': neuro['dopamine'],
            'acetylcholine': neuro['acetylcholine'],
            'norepinephrine': neuro['norepinephrine'],
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stag,
            'best_fitness_seen': new_best,
            'fitness_history': fitness_history,
            'strategy_name': self.name,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = self.get_active_agg_palette(new_state)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stag,
            'fitness_improved': improved,
            'dopamine': neuro['dopamine'],
            'acetylcholine': neuro['acetylcholine'],
            'norepinephrine': neuro['norepinephrine'],
            'act_avg_weight': float(jnp.mean(new_act_weights)),
            'agg_avg_weight': float(jnp.mean(new_agg_weights)),
            'cross_avg_weight': float(jnp.mean(new_cross_weights)),
            'act_n_protected': int(jnp.sum(new_act_prot) / 2),
            'agg_n_protected': int(jnp.sum(new_agg_prot) / 2),
            'cross_n_protected': int(jnp.sum(new_cross_prot)),
            'has_sin': 4 in act_palette,
            'has_agg4': len(agg_palette) >= 4,
        }
        metrics.update(act_info)
        metrics.update(agg_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return combined state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'stagnation_count': state['stagnation_count'],
            'dopamine': state['dopamine'],
            'acetylcholine': state['acetylcholine'],
            'norepinephrine': state['norepinephrine'],
            'act_n_protected': int(jnp.sum(state['act_protected']) / 2),
            'agg_n_protected': int(jnp.sum(state['agg_protected']) / 2),
            'cross_n_protected': int(jnp.sum(state['cross_protected'])),
        }
