"""Strategy 18 Dual: Neural Darwinism for Both Activation AND Aggregation.

Extends NeuralDarwinism to jointly evolve both activation and aggregation palettes
with cooperation/antagonism tracking in both domains.

Key mechanisms extended to dual:
1. Cooperation matrices for both act and agg domains
2. Antagonism matrices for both domains
3. Neuronal group detection in each domain
4. Selective stabilization with cross-domain awareness
5. Max_active constraints prevent antagonism (max_act=6, max_agg=4)

Biological rationale:
- Neural Darwinism applies to ALL neuronal populations
- Groups that cooperate (act-agg combinations) get stabilized together
- Cross-modal antagonism detection: certain act-agg pairs may conflict

Expected improvement:
- Discovers function pairs that CONFLICT in either domain
- Cross-domain antagonism detection (bad act-agg combinations)
- Better pruning based on conflict rather than just inactivity
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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


class CriticalPeriodPhase:
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class NeuralDarwinismDualStrategy(PaletteEvolutionStrategy):
    """Neural Darwinism with dual palette evolution.

    Tracks cooperation and antagonism in both activation and aggregation domains.
    """

    name = "neural_darwinism_dual"
    description = "Dual palette selective stabilization with cooperation/antagonism"

    def __init__(
        self,
        # Critical period timing
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # Phase rates
        exploration_activate: float = 0.35,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate_max: float = 0.15,
        confirmation_deactivate_min: float = 0.01,
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # Aggregation rates
        agg_exploration_activate: float = 0.30,
        agg_exploration_deactivate: float = 0.02,
        # Neural Darwinism parameters
        cooperation_threshold: float = 0.65,
        antagonism_threshold: float = 0.35,
        cooperation_rate: float = 0.15,
        antagonism_rate: float = 0.10,
        selection_pressure: float = 0.20,
        group_min_size: int = 2,
        antagonism_prune_threshold: float = 0.7,
        selective_death_rate: float = 0.1,
        # Cross-domain
        cross_learning_rate: float = 0.12,
        cross_influence: float = 0.25,
        # Base parameters
        learning_rate: float = 0.20,
        affinity_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        early_consolidation_threshold: float = 0.95,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end
        self.exploration_activate = exploration_activate
        self.exploration_deactivate = exploration_deactivate
        self.confirmation_activate = confirmation_activate
        self.confirmation_deactivate_max = confirmation_deactivate_max
        self.confirmation_deactivate_min = confirmation_deactivate_min
        self.consolidation_activate = consolidation_activate
        self.consolidation_deactivate = consolidation_deactivate
        self.agg_exploration_activate = agg_exploration_activate
        self.agg_exploration_deactivate = agg_exploration_deactivate
        self.cooperation_threshold = cooperation_threshold
        self.antagonism_threshold = antagonism_threshold
        self.cooperation_rate = cooperation_rate
        self.antagonism_rate = antagonism_rate
        self.selection_pressure = selection_pressure
        self.group_min_size = group_min_size
        self.antagonism_prune_threshold = antagonism_prune_threshold
        self.selective_death_rate = selective_death_rate
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        self.learning_rate = learning_rate
        self.affinity_protection_threshold = affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier
        self.early_consolidation_threshold = early_consolidation_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION
        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        act_mask = create_initial_palette_mask(config.get('initial_act_palette', self.initial_act_palette))
        agg_mask = create_initial_agg_palette_mask(config.get('initial_agg_palette', self.initial_agg_palette))

        return {
            'act_mask': act_mask,
            'act_affinity': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'act_cooperation': jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5,
            'act_antagonism': jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS)),
            'agg_mask': agg_mask,
            'agg_affinity': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'agg_cooperation': jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5,
            'agg_antagonism': jnp.zeros((NUM_AGGREGATIONS, NUM_AGGREGATIONS)),
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,
            'rng_key': jax.random.PRNGKey(seed + 181819),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'fitness_history': [],
            'fitness_ema': 0.5,
            'act_selection_events': 0,
            'agg_selection_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _get_phase_lr(self, phase: str) -> float:
        if phase == CriticalPeriodPhase.EXPLORATION:
            return self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            return self.confirmation_lr_multiplier
        return 0.1

    def _update_coop_antag(
        self, coop: jnp.ndarray, antag: jnp.ndarray, mask: jnp.ndarray,
        fitness_signal: float, lr: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        if fitness_signal > 0:
            coop_delta = self.cooperation_rate * lr * fitness_signal * co_active
            antag_delta = -self.antagonism_rate * lr * 0.3 * fitness_signal * co_active
        else:
            antag_delta = self.antagonism_rate * lr * abs(fitness_signal) * co_active
            coop_delta = -self.cooperation_rate * lr * 0.3 * abs(fitness_signal) * co_active

        return jnp.clip(coop + coop_delta, 0.0, 1.0), jnp.clip(antag + antag_delta, 0.0, 1.0)

    def _detect_groups(self, coop: jnp.ndarray, mask: jnp.ndarray, n_funcs: int) -> List[Set[int]]:
        active = [i for i in range(n_funcs) if mask[i] > 0.5]
        if len(active) < self.group_min_size:
            return []
        groups, visited = [], set()
        for start in active:
            if start in visited:
                continue
            group = {start}
            queue = [start]
            while queue:
                curr = queue.pop(0)
                for other in active:
                    if other not in group and coop[curr, other] > self.cooperation_threshold:
                        group.add(other)
                        queue.append(other)
            if len(group) >= self.group_min_size:
                groups.append(group)
            visited.update(group)
        return groups

    def _detect_antagonistic_pairs(self, antag: jnp.ndarray, mask: jnp.ndarray, n_funcs: int) -> List[Tuple[int, int]]:
        active = [i for i in range(n_funcs) if mask[i] > 0.5]
        pairs = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if antag[active[i], active[j]] > self.antagonism_prune_threshold:
                    pairs.append((active[i], active[j]))
        return pairs

    def _apply_selection(
        self, affinity: jnp.ndarray, coop: jnp.ndarray, antag: jnp.ndarray,
        groups: List[Set[int]], antag_pairs: List[Tuple[int, int]], phase: str, n_funcs: int
    ) -> Tuple[jnp.ndarray, List[int]]:
        if phase == CriticalPeriodPhase.EXPLORATION:
            pressure = self.selection_pressure * 0.5
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            pressure = self.selection_pressure
        else:
            pressure = self.selection_pressure * 0.3

        new_affinity = affinity.copy()
        to_prune = []

        for group in groups:
            if len(group) > 1:
                gc = sum(float(coop[i, j]) for i in group for j in group if i != j)
                avg_c = gc / (len(group) * (len(group) - 1))
                for idx in group:
                    boost = pressure * (avg_c - 0.5)
                    new_affinity = new_affinity.at[idx].set(min(0.95, float(new_affinity[idx]) + boost))

        antag_count = {i: 0 for i in range(n_funcs)}
        for i, j in antag_pairs:
            antag_count[i] += 1
            antag_count[j] += 1
            penalty = pressure * float(antag[i, j])
            new_affinity = new_affinity.at[i].set(max(0.05, float(new_affinity[i]) - penalty * 0.5))
            new_affinity = new_affinity.at[j].set(max(0.05, float(new_affinity[j]) - penalty * 0.5))

        for idx, count in antag_count.items():
            if count >= 2:
                to_prune.append(idx)

        return new_affinity, to_prune

    def _compute_protection(
        self, affinity: jnp.ndarray, coop: jnp.ndarray, mask: jnp.ndarray,
        cross: jnp.ndarray, other_mask: jnp.ndarray, is_act: bool
    ) -> jnp.ndarray:
        active = (mask > 0.5).astype(jnp.float32)
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        n_other = max(jnp.sum(other_active), 1)

        pairwise = jnp.dot(coop, active) / n_active
        if is_act:
            cross_score = jnp.dot(cross, other_active) / n_other
        else:
            cross_score = jnp.dot(cross.T, other_active) / n_other

        return 0.55 * affinity + 0.25 * pairwise + 0.20 * cross_score * self.cross_influence

    def _mutate_palette(
        self, key: jax.random.PRNGKey, mask: jnp.ndarray, phase: str,
        protection: jnp.ndarray, to_prune: List[int], n_funcs: int,
        min_active: int, max_active: int, is_act: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()
        activated, deactivated, killed = [], [], []

        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))
        death_probs = jax.random.uniform(key3, (n_funcs,))

        use_death = phase == CriticalPeriodPhase.CONFIRMATION

        if use_death:
            for idx in to_prune:
                if mask[idx] > 0.5 and death_probs[idx] < self.selective_death_rate:
                    new_mask = new_mask.at[idx].set(0.0)
                    killed.append(idx)

        if is_act:
            act_rate = self.exploration_activate if phase == CriticalPeriodPhase.EXPLORATION else self.confirmation_activate if phase == CriticalPeriodPhase.CONFIRMATION else self.consolidation_activate
            deact_max = self.confirmation_deactivate_max
            deact_min = self.confirmation_deactivate_min
        else:
            act_rate = self.agg_exploration_activate if phase == CriticalPeriodPhase.EXPLORATION else self.confirmation_activate if phase == CriticalPeriodPhase.CONFIRMATION else self.consolidation_activate
            deact_max = 0.12
            deact_min = 0.01

        current_active = int(jnp.sum(new_mask > 0.5))

        for i in range(n_funcs):
            if i in killed:
                continue
            prot = float(protection[i])
            if mask[i] < 0.5:
                if current_active + len(activated) >= max_active:
                    continue
                eff = act_rate * (0.5 + prot) if phase != CriticalPeriodPhase.EXPLORATION else act_rate
                if act_probs[i] < eff:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if phase == CriticalPeriodPhase.CONSOLIDATION and prot >= self.affinity_protection_threshold:
                    continue
                if prot >= self.affinity_protection_threshold:
                    dr = deact_min
                else:
                    t = prot / self.affinity_protection_threshold
                    dr = deact_max * (1 - t) + deact_min * t
                if phase == CriticalPeriodPhase.EXPLORATION:
                    dr = self.exploration_deactivate if is_act else self.agg_exploration_deactivate
                if deact_probs[i] < dr:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            return mask, {'activated': [], 'deactivated': [], 'killed': []}

        prefix = 'act_' if is_act else 'agg_'
        return new_mask, {f'{prefix}activated': activated, f'{prefix}deactivated': deactivated, f'{prefix}killed': killed}

    def post_generation_update(
        self, state: Dict[str, Any], generation: int, best_fitness: float,
        prev_best_fitness: float, population_data: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']
        phase = self._get_phase(generation, new_best)
        lr = self._get_phase_lr(phase)

        alpha = 0.2
        new_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fs = max(-1.0, min(1.0, (best_fitness - new_ema) / max(0.1, new_ema)))

        # Update cooperation/antagonism
        new_act_coop, new_act_antag = self._update_coop_antag(state['act_cooperation'], state['act_antagonism'], state['act_mask'], fs, lr)
        new_agg_coop, new_agg_antag = self._update_coop_antag(state['agg_cooperation'], state['agg_antagonism'], state['agg_mask'], fs, lr)

        # Update affinities
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        new_act_aff = state['act_affinity'] + self.learning_rate * lr * fs * act_active
        new_agg_aff = state['agg_affinity'] + self.learning_rate * lr * fs * agg_active
        new_act_aff = jnp.clip(new_act_aff, 0.0, 1.0)
        new_agg_aff = jnp.clip(new_agg_aff, 0.0, 1.0)

        # Cross-domain
        cross_active = jnp.outer(act_active, agg_active)
        cross_delta = self.cross_learning_rate * lr * fs * cross_active
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Detect groups and antagonism
        act_groups = self._detect_groups(new_act_coop, state['act_mask'], NUM_ACTIVATIONS)
        act_antag_pairs = self._detect_antagonistic_pairs(new_act_antag, state['act_mask'], NUM_ACTIVATIONS)
        agg_groups = self._detect_groups(new_agg_coop, state['agg_mask'], NUM_AGGREGATIONS)
        agg_antag_pairs = self._detect_antagonistic_pairs(new_agg_antag, state['agg_mask'], NUM_AGGREGATIONS)

        # Selection
        new_act_aff, act_prune = self._apply_selection(new_act_aff, new_act_coop, new_act_antag, act_groups, act_antag_pairs, phase, NUM_ACTIVATIONS)
        new_agg_aff, agg_prune = self._apply_selection(new_agg_aff, new_agg_coop, new_agg_antag, agg_groups, agg_antag_pairs, phase, NUM_AGGREGATIONS)

        # Protection
        act_prot = self._compute_protection(new_act_aff, new_act_coop, state['act_mask'], new_cross, state['agg_mask'], True)
        agg_prot = self._compute_protection(new_agg_aff, new_agg_coop, state['agg_mask'], new_cross, state['act_mask'], False)

        # Mutations
        new_act_mask, act_mut = self._mutate_palette(k_act, state['act_mask'], phase, act_prot, act_prune, NUM_ACTIVATIONS, self.min_active_act, self.max_active_act, True)
        new_agg_mask, agg_mut = self._mutate_palette(k_agg, state['agg_mask'], phase, agg_prot, agg_prune, NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg, False)

        fh = state['fitness_history'] + [best_fitness]
        if len(fh) > 20:
            fh = fh[-20:]

        new_state = {
            'act_mask': new_act_mask, 'act_affinity': new_act_aff, 'act_cooperation': new_act_coop, 'act_antagonism': new_act_antag,
            'agg_mask': new_agg_mask, 'agg_affinity': new_agg_aff, 'agg_cooperation': new_agg_coop, 'agg_antagonism': new_agg_antag,
            'cross_affinity': new_cross, 'rng_key': key, 'generation': generation + 1, 'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best, 'strategy_name': self.name, 'phase': phase, 'fitness_history': fh, 'fitness_ema': new_ema,
            'act_selection_events': state['act_selection_events'] + (1 if act_groups or act_antag_pairs else 0),
            'agg_selection_events': state['agg_selection_events'] + (1 if agg_groups or agg_antag_pairs else 0),
        }

        metrics = {
            'act_palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'phase': phase, 'fitness_signal': fs,
            'act_n_groups': len(act_groups), 'agg_n_groups': len(agg_groups),
            'act_n_antag_pairs': len(act_antag_pairs), 'agg_n_antag_pairs': len(agg_antag_pairs),
            'sin_affinity': float(new_act_aff[4]) if 4 < len(new_act_aff) else 0.0,
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
            'phase': state['phase'],
            'generation': state['generation'],
            'sin_affinity': float(state['act_affinity'][4]),
            'act_selection_events': state['act_selection_events'],
            'agg_selection_events': state['agg_selection_events'],
        }
