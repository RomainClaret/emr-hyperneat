"""Strategy 23 Dual: Complementary Learning for Both Activation AND Aggregation.

Extends ComplementaryLearning to jointly evolve both activation and aggregation palettes
with fast/slow memory systems in both domains.

Key mechanisms extended to dual:
1. Fast (hippocampus-like) and slow (cortex-like) systems for BOTH domains
2. Consolidation transfers from fast to slow in both domains
3. Cross-domain fast/slow tracking for act-agg combinations
4. Stability-plasticity tradeoff managed in each domain
5. Max_active constraints prevent antagonism (max_act=6, max_agg=4)

Biological rationale:
- Hippocampus-cortex distinction applies to ALL modalities
- Visual, motor, and semantic memories all use CLS
- Cross-modal associations are consolidated together

Expected improvement:
- Solves stability-plasticity in BOTH domains
- Fast discovery + slow retention for both act and agg
- Cross-domain consolidation links successful combinations
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


class ComplementaryLearningDualStrategy(PaletteEvolutionStrategy):
    """Hippocampus-cortex dual memory with dual palette evolution.

    Fast/slow systems in both activation and aggregation domains.
    """

    name = "complementary_learning_dual"
    description = "Dual palette hippocampus-cortex memory with consolidation"

    def __init__(
        self,
        # Fast system
        fast_learning_rate: float = 0.35,
        fast_decay: float = 0.15,
        fast_weight: float = 0.3,
        # Slow system
        slow_learning_rate: float = 0.06,
        slow_decay: float = 0.0,
        slow_weight: float = 0.7,
        # Consolidation
        consolidation_interval: int = 12,
        consolidation_rate: float = 0.25,
        consolidation_threshold: float = 0.55,
        replay_boost: float = 1.3,
        fast_reset_factor: float = 0.5,
        # Cross-domain
        cross_learning_rate: float = 0.15,
        cross_influence: float = 0.3,
        # Protection
        affinity_protection_threshold: float = 0.6,
        # Mutation
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.fast_learning_rate = fast_learning_rate
        self.fast_decay = fast_decay
        self.fast_weight = fast_weight
        self.slow_learning_rate = slow_learning_rate
        self.slow_decay = slow_decay
        self.slow_weight = slow_weight
        self.consolidation_interval = consolidation_interval
        self.consolidation_rate = consolidation_rate
        self.consolidation_threshold = consolidation_threshold
        self.replay_boost = replay_boost
        self.fast_reset_factor = fast_reset_factor
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        self.affinity_protection_threshold = affinity_protection_threshold
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
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
            'act_fast': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'act_slow': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'agg_mask': agg_mask,
            'agg_fast': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'agg_slow': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,
            'rng_key': jax.random.PRNGKey(seed + 232324),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'last_consolidation': 0,
            'consolidation_count': 0,
            'fitness_history': [],
            'fitness_ema': 0.5,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _get_effective(self, fast: jnp.ndarray, slow: jnp.ndarray) -> jnp.ndarray:
        return self.slow_weight * slow + self.fast_weight * fast

    def _update_fast(self, fast: jnp.ndarray, mask: jnp.ndarray, fs: float) -> jnp.ndarray:
        new_fast = (1 - self.fast_decay) * fast + self.fast_decay * 0.5
        active = (mask > 0.5).astype(jnp.float32)
        delta = self.fast_learning_rate * fs * active if fs >= 0 else self.fast_learning_rate * 0.3 * fs * active
        return jnp.clip(new_fast + delta, 0.05, 0.95)

    def _update_slow(self, slow: jnp.ndarray, mask: jnp.ndarray, fs: float) -> jnp.ndarray:
        active = (mask > 0.5).astype(jnp.float32)
        delta = self.slow_learning_rate * fs * active if fs >= 0 else self.slow_learning_rate * 0.2 * fs * active
        new_slow = slow + delta
        if self.slow_decay > 0:
            new_slow = (1 - self.slow_decay) * new_slow + self.slow_decay * 0.5
        return jnp.clip(new_slow, 0.05, 0.95)

    def _consolidate(
        self, fast: jnp.ndarray, slow: jnp.ndarray, mask: jnp.ndarray, n_funcs: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        new_fast, new_slow = fast.copy(), slow.copy()
        consolidated = []
        active = (mask > 0.5)

        for i in range(n_funcs):
            fv = float(fast[i])
            if fv >= self.consolidation_threshold:
                if active[i] and fv > float(slow[i]):
                    transfer = self.consolidation_rate * fv * self.replay_boost
                else:
                    transfer = self.consolidation_rate * fv
                new_slow = new_slow.at[i].set(min(0.95, float(new_slow[i]) + transfer))
                consolidated.append(i)

        new_fast = new_fast * self.fast_reset_factor + 0.5 * (1 - self.fast_reset_factor)
        return new_fast, new_slow, {'consolidated': consolidated, 'n_consolidated': len(consolidated)}

    def _compute_protection(
        self, effective: jnp.ndarray, slow: jnp.ndarray, cross: jnp.ndarray,
        other_mask: jnp.ndarray, is_act: bool
    ) -> jnp.ndarray:
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)
        if is_act:
            cross_score = jnp.dot(cross, other_active) / n_other
        else:
            cross_score = jnp.dot(cross.T, other_active) / n_other
        return 0.5 * slow + 0.3 * effective + 0.2 * cross_score * self.cross_influence

    def _mutate_palette(
        self, key: jax.random.PRNGKey, mask: jnp.ndarray, protection: jnp.ndarray,
        effective: jnp.ndarray, n_funcs: int, min_active: int, max_active: int, is_act: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []

        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))
        current = int(jnp.sum(mask > 0.5))

        for i in range(n_funcs):
            prot = float(protection[i])
            aff = float(effective[i])
            if mask[i] < 0.5:
                if current + len(activated) >= max_active:
                    continue
                rate = self.base_activate_rate * (0.5 + 0.5 * aff)
                if act_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if prot >= self.affinity_protection_threshold:
                    rate = self.base_deactivate_rate * 0.1
                else:
                    rate = self.base_deactivate_rate * (1.0 - prot)
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

        alpha = 0.2
        new_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fs = max(-1.0, min(1.0, (best_fitness - new_ema) / max(0.1, new_ema)))

        # Update fast systems
        new_act_fast = self._update_fast(state['act_fast'], state['act_mask'], fs)
        new_agg_fast = self._update_fast(state['agg_fast'], state['agg_mask'], fs)

        # Update slow systems
        new_act_slow = self._update_slow(state['act_slow'], state['act_mask'], fs)
        new_agg_slow = self._update_slow(state['agg_slow'], state['agg_mask'], fs)

        # Cross-domain
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_delta = self.cross_learning_rate * fs * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Consolidation
        consol_metrics = {}
        did_consolidate = False
        gens_since = generation - state['last_consolidation']
        last_consol = state['last_consolidation']
        consol_count = state['consolidation_count']

        if gens_since >= self.consolidation_interval:
            new_act_fast, new_act_slow, act_consol = self._consolidate(new_act_fast, new_act_slow, state['act_mask'], NUM_ACTIVATIONS)
            new_agg_fast, new_agg_slow, agg_consol = self._consolidate(new_agg_fast, new_agg_slow, state['agg_mask'], NUM_AGGREGATIONS)
            last_consol = generation
            consol_count += 1
            did_consolidate = True
            consol_metrics = {'act_consolidated': act_consol['consolidated'], 'agg_consolidated': agg_consol['consolidated']}

        # Effective affinities
        act_eff = self._get_effective(new_act_fast, new_act_slow)
        agg_eff = self._get_effective(new_agg_fast, new_agg_slow)

        # Protection
        act_prot = self._compute_protection(act_eff, new_act_slow, new_cross, state['agg_mask'], True)
        agg_prot = self._compute_protection(agg_eff, new_agg_slow, new_cross, state['act_mask'], False)

        # Mutations
        new_act_mask, act_mut = self._mutate_palette(k_act, state['act_mask'], act_prot, act_eff, NUM_ACTIVATIONS, self.min_active_act, self.max_active_act, True)
        new_agg_mask, agg_mut = self._mutate_palette(k_agg, state['agg_mask'], agg_prot, agg_eff, NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg, False)

        fh = state['fitness_history'] + [best_fitness]
        if len(fh) > 20:
            fh = fh[-20:]

        new_state = {
            'act_mask': new_act_mask, 'act_fast': new_act_fast, 'act_slow': new_act_slow,
            'agg_mask': new_agg_mask, 'agg_fast': new_agg_fast, 'agg_slow': new_agg_slow,
            'cross_affinity': new_cross, 'rng_key': key, 'generation': generation + 1,
            'stagnation_count': new_stagnation, 'best_fitness_seen': new_best, 'strategy_name': self.name,
            'last_consolidation': last_consol, 'consolidation_count': consol_count,
            'fitness_history': fh, 'fitness_ema': new_ema,
        }

        metrics = {
            'act_palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'did_consolidate': did_consolidate, 'consolidation_count': consol_count,
            'sin_fast': float(new_act_fast[4]) if 4 < len(new_act_fast) else 0.0,
            'sin_slow': float(new_act_slow[4]) if 4 < len(new_act_slow) else 0.0,
            'sin_effective': float(act_eff[4]) if 4 < len(act_eff) else 0.0,
        }
        metrics.update(act_mut)
        metrics.update(agg_mut)
        metrics.update(consol_metrics)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        act_eff = self._get_effective(state['act_fast'], state['act_slow'])
        return {
            'strategy': self.name,
            'active_act_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'generation': state['generation'],
            'consolidation_count': state['consolidation_count'],
            'sin_effective': float(act_eff[4]),
        }
