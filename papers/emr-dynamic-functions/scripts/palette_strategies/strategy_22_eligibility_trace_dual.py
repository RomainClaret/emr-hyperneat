"""Strategy 22 Dual: Eligibility Trace for Both Activation AND Aggregation.

Extends EligibilityTrace to jointly evolve both activation and aggregation palettes
with three-factor learning in both domains.

Key mechanisms extended to dual:
1. Separate eligibility traces for activations and aggregations
2. Dopamine signal is SHARED (global reward signal)
3. Cross-domain eligibility tracks act-agg combinations
4. Three-factor rule applies to both domains
5. Max_active constraints prevent antagonism (max_act=6, max_agg=4)

Biological rationale:
- Eligibility traces exist at ALL synapses
- Dopamine is a global signal affecting all circuits equally
- Cross-modal associations have shared eligibility windows

Expected improvement:
- Better temporal credit assignment in BOTH domains
- Shared dopamine properly coordinates dual learning
- Cross-domain eligibility captures act-agg timing relationships
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


class EligibilityTraceDualStrategy(PaletteEvolutionStrategy):
    """Three-factor learning with dual palette evolution.

    Eligibility traces in both domains, shared dopamine signal.
    """

    name = "eligibility_trace_dual"
    description = "Dual palette dopamine-gated eligibility trace learning"

    def __init__(
        self,
        # Eligibility parameters
        eligibility_decay: float = 0.85,
        eligibility_boost_active: float = 1.0,
        eligibility_boost_changed: float = 0.5,
        # Dopamine parameters
        dopamine_baseline_momentum: float = 0.9,
        dopamine_sensitivity: float = 1.5,
        dopamine_learning_rate: float = 0.2,
        # Cross-domain
        cross_learning_rate: float = 0.15,
        cross_influence: float = 0.3,
        # Protection
        affinity_protection_threshold: float = 0.6,
        protection_decay: float = 0.98,
        # Mutation
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        da_exploration_modulation: float = 0.3,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.eligibility_decay = eligibility_decay
        self.eligibility_boost_active = eligibility_boost_active
        self.eligibility_boost_changed = eligibility_boost_changed
        self.dopamine_baseline_momentum = dopamine_baseline_momentum
        self.dopamine_sensitivity = dopamine_sensitivity
        self.dopamine_learning_rate = dopamine_learning_rate
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        self.affinity_protection_threshold = affinity_protection_threshold
        self.protection_decay = protection_decay
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.da_exploration_modulation = da_exploration_modulation
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        act_mask = create_initial_palette_mask(config.get('initial_act_palette', self.initial_act_palette))
        agg_mask = create_initial_agg_palette_mask(config.get('initial_agg_palette', self.initial_agg_palette))

        return {
            'act_mask': act_mask,
            'act_eligibility': jnp.zeros(NUM_ACTIVATIONS),
            'act_affinity': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'act_prev_mask': act_mask,
            'agg_mask': agg_mask,
            'agg_eligibility': jnp.zeros(NUM_AGGREGATIONS),
            'agg_affinity': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'agg_prev_mask': agg_mask,
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,
            'rng_key': jax.random.PRNGKey(seed + 222223),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'dopamine_baseline': 0.5,
            'dopamine_signal': 0.0,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_eligibility(
        self, elig: jnp.ndarray, mask: jnp.ndarray, prev_mask: jnp.ndarray
    ) -> jnp.ndarray:
        new_elig = self.eligibility_decay * elig
        active = (mask > 0.5).astype(jnp.float32)
        new_elig = new_elig + self.eligibility_boost_active * active
        was_inactive = (prev_mask < 0.5).astype(jnp.float32)
        just_activated = active * was_inactive
        new_elig = new_elig + self.eligibility_boost_changed * just_activated
        return jnp.clip(new_elig, 0.0, 3.0)

    def _compute_dopamine(self, fitness: float, baseline: float) -> Tuple[float, float]:
        new_baseline = self.dopamine_baseline_momentum * baseline + (1 - self.dopamine_baseline_momentum) * fitness
        if baseline > 0.01:
            pe = (fitness - baseline) / baseline
        else:
            pe = fitness - baseline
        da = max(-1.0, min(1.0, self.dopamine_sensitivity * pe))
        return da, new_baseline

    def _update_affinity(self, aff: jnp.ndarray, elig: jnp.ndarray, da: float) -> jnp.ndarray:
        delta = self.dopamine_learning_rate * da * elig
        return jnp.clip(aff + delta, 0.05, 0.95)

    def _compute_protection(
        self, aff: jnp.ndarray, elig: jnp.ndarray, cross: jnp.ndarray,
        other_mask: jnp.ndarray, is_act: bool
    ) -> jnp.ndarray:
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)
        if is_act:
            cross_score = jnp.dot(cross, other_active) / n_other
        else:
            cross_score = jnp.dot(cross.T, other_active) / n_other
        elig_contrib = 0.2 * jnp.clip(elig / 2.0, 0, 1)
        prot = 0.6 * aff + 0.2 * elig_contrib + 0.2 * cross_score * self.cross_influence
        return jnp.clip(prot, 0.0, 1.0)

    def _mutate_palette(
        self, key: jax.random.PRNGKey, mask: jnp.ndarray, protection: jnp.ndarray,
        da: float, n_funcs: int, min_active: int, max_active: int, is_act: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []

        da_factor = max(0.5, min(1.5, 1.0 - self.da_exploration_modulation * da))
        eff_act = self.base_activate_rate * da_factor
        eff_deact = self.base_deactivate_rate * da_factor

        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))

        current = int(jnp.sum(mask > 0.5))

        for i in range(n_funcs):
            prot = float(protection[i])
            if mask[i] < 0.5:
                if current + len(activated) >= max_active:
                    continue
                rate = eff_act * (0.5 + 0.5 * prot)
                if act_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if prot >= self.affinity_protection_threshold:
                    rate = eff_deact * 0.1
                else:
                    rate = eff_deact * (1.0 - prot)
                if deact_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            return mask, {'activated': [], 'deactivated': []}

        prefix = 'act_' if is_act else 'agg_'
        return new_mask, {f'{prefix}activated': activated, f'{prefix}deactivated': deactivated}

    def post_generation_update(
        self, state: Dict[str, Any], generation: int, best_fitness: float,
        prev_best_fitness: float, population_data: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Shared dopamine signal
        da, new_baseline = self._compute_dopamine(best_fitness, state['dopamine_baseline'])

        # Update eligibility
        new_act_elig = self._update_eligibility(state['act_eligibility'], state['act_mask'], state['act_prev_mask'])
        new_agg_elig = self._update_eligibility(state['agg_eligibility'], state['agg_mask'], state['agg_prev_mask'])

        # Three-factor update
        new_act_aff = self._update_affinity(state['act_affinity'], new_act_elig, da)
        new_agg_aff = self._update_affinity(state['agg_affinity'], new_agg_elig, da)

        # Cross-domain
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_delta = self.cross_learning_rate * da * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Protection
        act_prot = self._compute_protection(new_act_aff, new_act_elig, new_cross, state['agg_mask'], True)
        agg_prot = self._compute_protection(new_agg_aff, new_agg_elig, new_cross, state['act_mask'], False)

        # Mutations
        new_act_mask, act_mut = self._mutate_palette(k_act, state['act_mask'], act_prot, da, NUM_ACTIVATIONS, self.min_active_act, self.max_active_act, True)
        new_agg_mask, agg_mut = self._mutate_palette(k_agg, state['agg_mask'], agg_prot, da, NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg, False)

        fh = state['fitness_history'] + [best_fitness]
        if len(fh) > 20:
            fh = fh[-20:]

        new_state = {
            'act_mask': new_act_mask, 'act_eligibility': new_act_elig, 'act_affinity': new_act_aff, 'act_prev_mask': state['act_mask'],
            'agg_mask': new_agg_mask, 'agg_eligibility': new_agg_elig, 'agg_affinity': new_agg_aff, 'agg_prev_mask': state['agg_mask'],
            'cross_affinity': new_cross, 'rng_key': key, 'generation': generation + 1, 'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best, 'strategy_name': self.name, 'dopamine_baseline': new_baseline, 'dopamine_signal': da,
            'fitness_history': fh,
        }

        metrics = {
            'act_palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'dopamine_signal': da, 'dopamine_baseline': new_baseline,
            'sin_affinity': float(new_act_aff[4]) if 4 < len(new_act_aff) else 0.0,
            'sin_eligibility': float(new_act_elig[4]) if 4 < len(new_act_elig) else 0.0,
            'act_avg_elig': float(jnp.mean(new_act_elig)), 'agg_avg_elig': float(jnp.mean(new_agg_elig)),
        }
        metrics.update(act_mut)
        metrics.update(agg_mut)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'strategy': self.name,
            'active_act_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'generation': state['generation'],
            'dopamine_signal': state['dopamine_signal'],
            'sin_affinity': float(state['act_affinity'][4]),
        }
