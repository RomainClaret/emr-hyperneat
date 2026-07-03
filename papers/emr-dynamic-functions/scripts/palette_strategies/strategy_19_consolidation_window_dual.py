"""Strategy 19 Dual: Consolidation Window for Both Activation AND Aggregation.

Extends ConsolidationWindow to jointly evolve both activation and aggregation palettes
with periodic consolidation phases for both domains.

Key mechanisms extended to dual:
1. Working and long-term memory for both act and agg domains
2. Periodic consolidation windows apply to both domains
3. Replay and transfer operate independently in each domain
4. Cross-domain memory tracks successful act-agg combinations
5. Max_active constraints prevent antagonism (max_act=6, max_agg=4)

Biological rationale:
- Sleep consolidation affects ALL memory types (visual, motor, semantic)
- Different modalities have separate but synchronized consolidation
- Cross-modal associations are strengthened during replay

Expected improvement:
- More stable retention in BOTH domains
- Prevents oscillation from constant mutation in either domain
- Better protection of important functions in both domains
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


class ConsolidationPhase:
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"


class ConsolidationWindowDualStrategy(PaletteEvolutionStrategy):
    """Memory consolidation with dual palette evolution.

    Periodic consolidation windows for both activation and aggregation.
    """

    name = "consolidation_window_dual"
    description = "Dual palette memory consolidation with periodic windows"

    def __init__(
        self,
        # Consolidation timing
        consolidation_frequency: int = 10,
        consolidation_duration: int = 3,
        # Consolidation parameters
        replay_strength: float = 1.5,
        replay_threshold: float = 0.6,
        transfer_rate: float = 0.1,
        ltm_decay_rate: float = 0.02,
        # Active phase parameters
        active_learning_rate: float = 0.15,
        active_mutation_rate: float = 0.20,
        # Consolidation phase parameters
        consolidation_mutation_rate: float = 0.02,
        consolidation_learning_rate: float = 0.05,
        # Cross-domain
        cross_learning_rate: float = 0.12,
        cross_influence: float = 0.25,
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
        self.consolidation_frequency = consolidation_frequency
        self.consolidation_duration = consolidation_duration
        self.replay_strength = replay_strength
        self.replay_threshold = replay_threshold
        self.transfer_rate = transfer_rate
        self.ltm_decay_rate = ltm_decay_rate
        self.active_learning_rate = active_learning_rate
        self.active_mutation_rate = active_mutation_rate
        self.consolidation_mutation_rate = consolidation_mutation_rate
        self.consolidation_learning_rate = consolidation_learning_rate
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        self.affinity_protection_threshold = affinity_protection_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_consolidation_phase(self, generation: int, last_consol: int) -> Tuple[str, bool]:
        gens_since = generation - last_consol
        if gens_since < self.consolidation_duration:
            return ConsolidationPhase.CONSOLIDATING, False
        elif gens_since >= self.consolidation_frequency:
            return ConsolidationPhase.CONSOLIDATING, True
        return ConsolidationPhase.ACTIVE, False

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        act_mask = create_initial_palette_mask(config.get('initial_act_palette', self.initial_act_palette))
        agg_mask = create_initial_agg_palette_mask(config.get('initial_agg_palette', self.initial_agg_palette))

        return {
            'act_mask': act_mask,
            'act_working': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'act_ltm': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'agg_mask': agg_mask,
            'agg_working': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'agg_ltm': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,
            'rng_key': jax.random.PRNGKey(seed + 191920),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'consolidation_phase': ConsolidationPhase.ACTIVE,
            'last_consolidation': -self.consolidation_frequency,
            'consolidations_completed': 0,
            'fitness_history': [],
            'fitness_ema': 0.5,
            'act_replay_events': 0,
            'agg_replay_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_effective(self, working: jnp.ndarray, ltm: jnp.ndarray, phase: str) -> jnp.ndarray:
        if phase == ConsolidationPhase.CONSOLIDATING:
            return 0.3 * working + 0.7 * ltm
        return 0.6 * working + 0.4 * ltm

    def _update_working(self, working: jnp.ndarray, mask: jnp.ndarray, fs: float, phase: str) -> jnp.ndarray:
        lr = self.consolidation_learning_rate if phase == ConsolidationPhase.CONSOLIDATING else self.active_learning_rate
        active = (mask > 0.5).astype(jnp.float32)
        delta = lr * fs * active if fs >= 0 else lr * 0.3 * fs * active
        return jnp.clip(working + delta, 0.0, 1.0)

    def _consolidate(
        self, working: jnp.ndarray, ltm: jnp.ndarray, mask: jnp.ndarray, n_funcs: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        new_working = working.copy()
        new_ltm = ltm.copy()
        active = (mask > 0.5).astype(jnp.float32)
        n_replay, n_transfer = 0, 0

        for i in range(n_funcs):
            if float(working[i]) >= self.replay_threshold:
                boost = self.replay_strength * (float(working[i]) - self.replay_threshold)
                new_working = new_working.at[i].set(min(0.95, float(new_working[i]) + boost))
                n_replay += 1

            if float(working[i]) >= self.replay_threshold and float(active[i]) > 0.5:
                diff = float(working[i]) - float(ltm[i])
                transfer = self.transfer_rate * diff
                new_ltm = new_ltm.at[i].set(min(0.95, float(new_ltm[i]) + transfer))
                if transfer > 0.01:
                    n_transfer += 1

            if float(active[i]) < 0.5:
                decay = self.ltm_decay_rate * (float(new_ltm[i]) - 0.5)
                new_ltm = new_ltm.at[i].set(max(0.05, float(new_ltm[i]) - decay))

        return new_working, new_ltm, n_replay, n_transfer

    def _compute_protection(
        self, effective: jnp.ndarray, cross: jnp.ndarray, other_mask: jnp.ndarray, is_act: bool
    ) -> jnp.ndarray:
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)
        if is_act:
            cross_score = jnp.dot(cross, other_active) / n_other
        else:
            cross_score = jnp.dot(cross.T, other_active) / n_other
        return 0.7 * effective + 0.3 * cross_score * self.cross_influence

    def _mutate_palette(
        self, key: jax.random.PRNGKey, mask: jnp.ndarray, effective: jnp.ndarray,
        phase: str, n_funcs: int, min_active: int, max_active: int, is_act: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []

        mut_rate = self.consolidation_mutation_rate if phase == ConsolidationPhase.CONSOLIDATING else self.active_mutation_rate
        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))

        current = int(jnp.sum(mask > 0.5))

        for i in range(n_funcs):
            aff = float(effective[i])
            if mask[i] < 0.5:
                if current + len(activated) >= max_active:
                    continue
                eff_rate = mut_rate * (0.5 + aff)
                if act_probs[i] < eff_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if aff >= self.affinity_protection_threshold:
                    dr = mut_rate * 0.1
                else:
                    dr = mut_rate * (1.0 - aff)
                if phase == ConsolidationPhase.CONSOLIDATING:
                    dr *= 0.2
                if deact_probs[i] < dr:
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

        phase, starting = self._get_consolidation_phase(generation, state['last_consolidation'])
        last_consol = generation if starting else state['last_consolidation']
        consol_count = state['consolidations_completed'] + (1 if starting else 0)

        alpha = 0.2
        new_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fs = max(-1.0, min(1.0, (best_fitness - new_ema) / max(0.1, new_ema)))

        # Update working memory
        new_act_working = self._update_working(state['act_working'], state['act_mask'], fs, phase)
        new_agg_working = self._update_working(state['agg_working'], state['agg_mask'], fs, phase)

        # Update cross-domain
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_delta = self.cross_learning_rate * fs * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Consolidation
        new_act_ltm, new_agg_ltm = state['act_ltm'], state['agg_ltm']
        act_replay, agg_replay = 0, 0

        if phase == ConsolidationPhase.CONSOLIDATING:
            new_act_working, new_act_ltm, ar, _ = self._consolidate(new_act_working, state['act_ltm'], state['act_mask'], NUM_ACTIVATIONS)
            new_agg_working, new_agg_ltm, agr, _ = self._consolidate(new_agg_working, state['agg_ltm'], state['agg_mask'], NUM_AGGREGATIONS)
            act_replay, agg_replay = ar, agr

        # Effective affinity
        act_eff = self._compute_effective(new_act_working, new_act_ltm, phase)
        agg_eff = self._compute_effective(new_agg_working, new_agg_ltm, phase)

        # Protection
        act_prot = self._compute_protection(act_eff, new_cross, state['agg_mask'], True)
        agg_prot = self._compute_protection(agg_eff, new_cross, state['act_mask'], False)

        # Mutations
        new_act_mask, act_mut = self._mutate_palette(k_act, state['act_mask'], act_prot, phase, NUM_ACTIVATIONS, self.min_active_act, self.max_active_act, True)
        new_agg_mask, agg_mut = self._mutate_palette(k_agg, state['agg_mask'], agg_prot, phase, NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg, False)

        fh = state['fitness_history'] + [best_fitness]
        if len(fh) > 20:
            fh = fh[-20:]

        new_state = {
            'act_mask': new_act_mask, 'act_working': new_act_working, 'act_ltm': new_act_ltm,
            'agg_mask': new_agg_mask, 'agg_working': new_agg_working, 'agg_ltm': new_agg_ltm,
            'cross_affinity': new_cross, 'rng_key': key, 'generation': generation + 1,
            'stagnation_count': new_stagnation, 'best_fitness_seen': new_best, 'strategy_name': self.name,
            'consolidation_phase': phase, 'last_consolidation': last_consol, 'consolidations_completed': consol_count,
            'fitness_history': fh, 'fitness_ema': new_ema,
            'act_replay_events': state['act_replay_events'] + act_replay,
            'agg_replay_events': state['agg_replay_events'] + agg_replay,
        }

        metrics = {
            'act_palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'consolidation_phase': phase, 'starting_consolidation': starting,
            'sin_affinity': float(act_eff[4]) if 4 < len(act_eff) else 0.0,
            'sin_ltm': float(new_act_ltm[4]) if 4 < len(new_act_ltm) else 0.0,
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
            'consolidation_phase': state['consolidation_phase'],
            'consolidations_completed': state['consolidations_completed'],
            'generation': state['generation'],
        }
