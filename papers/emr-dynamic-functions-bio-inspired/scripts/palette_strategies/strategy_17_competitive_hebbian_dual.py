"""Strategy 17 Dual: Competitive Hebbian for Both Activation AND Aggregation.

Extends CompetitiveHebbian to jointly evolve both activation and aggregation palettes
with zero-sum resource competition in both domains.

Key mechanisms extended to dual:
1. Separate affinity budgets for activations and aggregations
2. Competition occurs within each domain independently
3. Cross-domain learning tracks act-agg combinations
4. Lateral inhibition operates in both domains
5. Max_active constraints prevent antagonism (max_act=6, max_agg=4)

Biological rationale:
- Competition for resources occurs at all levels (visual, motor, etc.)
- Different circuit types have independent resource pools
- Cross-modal coordination still happens through shared rewards

Expected improvement:
- More diverse palettes in BOTH domains
- Clear winners emerge in each domain separately
- Cross-domain competition may help find synergistic combinations
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


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class CompetitiveHebbianDualStrategy(PaletteEvolutionStrategy):
    """Competitive Hebbian with dual palette evolution.

    Zero-sum competition in both activation and aggregation domains.
    """

    name = "competitive_hebbian_dual"
    description = "Dual palette zero-sum Hebbian with lateral inhibition"

    def __init__(
        self,
        # Critical period timing
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # Phase rates (activation)
        exploration_activate: float = 0.35,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate_max: float = 0.15,
        confirmation_deactivate_min: float = 0.01,
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # Phase rates (aggregation)
        agg_exploration_activate: float = 0.30,
        agg_exploration_deactivate: float = 0.02,
        # Competition parameters
        act_target_affinity_sum: float = NUM_ACTIVATIONS * 0.5,
        agg_target_affinity_sum: float = NUM_AGGREGATIONS * 0.5,
        redistribution_rate: float = 0.15,
        winner_k: int = 3,
        loser_k: int = 3,
        agg_winner_k: int = 2,
        agg_loser_k: int = 2,
        winner_boost: float = 1.3,
        loser_penalty: float = 0.7,
        # Lateral inhibition
        inhibition_radius: float = 0.5,
        inhibition_strength: float = 0.1,
        # Cross-domain
        cross_learning_rate: float = 0.15,
        cross_influence: float = 0.3,
        # Base Hebbian parameters
        learning_rate: float = 0.20,
        anti_hebbian_rate: float = 0.05,
        affinity_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Constraints
        early_consolidation_threshold: float = 0.95,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize dual competitive Hebbian strategy."""
        # Critical period timing
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

        # Activation rates
        self.exploration_activate = exploration_activate
        self.exploration_deactivate = exploration_deactivate
        self.confirmation_activate = confirmation_activate
        self.confirmation_deactivate_max = confirmation_deactivate_max
        self.confirmation_deactivate_min = confirmation_deactivate_min
        self.consolidation_activate = consolidation_activate
        self.consolidation_deactivate = consolidation_deactivate

        # Aggregation rates
        self.agg_exploration_activate = agg_exploration_activate
        self.agg_exploration_deactivate = agg_exploration_deactivate

        # Competition parameters
        self.act_target_affinity_sum = act_target_affinity_sum
        self.agg_target_affinity_sum = agg_target_affinity_sum
        self.redistribution_rate = redistribution_rate
        self.winner_k = winner_k
        self.loser_k = loser_k
        self.agg_winner_k = agg_winner_k
        self.agg_loser_k = agg_loser_k
        self.winner_boost = winner_boost
        self.loser_penalty = loser_penalty
        self.inhibition_radius = inhibition_radius
        self.inhibition_strength = inhibition_strength

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence

        # Hebbian parameters
        self.learning_rate = learning_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.affinity_protection_threshold = affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Other
        self.early_consolidation_threshold = early_consolidation_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION
        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual competition tracking."""
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Activation domain
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * (self.act_target_affinity_sum / NUM_ACTIVATIONS)
        act_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        act_contributions = jnp.zeros(NUM_ACTIVATIONS)

        # Aggregation domain
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * (self.agg_target_affinity_sum / NUM_AGGREGATIONS)
        agg_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        agg_contributions = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_hebbian_weights': act_weights,
            'act_contributions': act_contributions,
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_hebbian_weights': agg_weights,
            'agg_contributions': agg_contributions,
            'cross_affinity': cross_affinity,
            'rng_key': jax.random.PRNGKey(seed + 171718),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'fitness_history': [],
            'fitness_ema': 0.5,
            'act_competition_events': 0,
            'agg_competition_events': 0,
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
        else:
            return 0.1

    def _compute_contributions(
        self, mask: jnp.ndarray, fitness_signal: float, prev: jnp.ndarray, decay: float = 0.9
    ) -> jnp.ndarray:
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        new_contrib = prev * decay + (fitness_signal / n_active) * active
        return new_contrib

    def _apply_competition(
        self, affinity: jnp.ndarray, contributions: jnp.ndarray, mask: jnp.ndarray,
        phase: str, winner_k: int, loser_k: int
    ) -> Tuple[jnp.ndarray, List[int], List[int]]:
        active = (mask > 0.5).astype(jnp.float32)
        active_indices = [i for i in range(len(affinity)) if active[i] > 0.5]

        if len(active_indices) < winner_k + loser_k:
            return affinity, [], []

        active_contrib = [(i, float(contributions[i])) for i in active_indices]
        active_contrib.sort(key=lambda x: x[1], reverse=True)

        winners = [idx for idx, _ in active_contrib[:winner_k]]
        losers = [idx for idx, _ in active_contrib[-loser_k:]]

        if phase == CriticalPeriodPhase.EXPLORATION:
            strength = 0.5
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            strength = 1.0
        else:
            strength = 0.3

        new_affinity = affinity.copy()
        for idx in winners:
            boost = 1.0 + (self.winner_boost - 1.0) * strength * self.redistribution_rate
            new_affinity = new_affinity.at[idx].set(min(0.95, float(new_affinity[idx]) * boost))

        for idx in losers:
            penalty = 1.0 - (1.0 - self.loser_penalty) * strength * self.redistribution_rate
            new_affinity = new_affinity.at[idx].set(max(0.05, float(new_affinity[idx]) * penalty))

        return new_affinity, winners, losers

    def _normalize_affinity(self, affinity: jnp.ndarray, target_sum: float) -> jnp.ndarray:
        current_sum = jnp.sum(affinity)
        if current_sum < 0.1:
            return jnp.ones(len(affinity)) * (target_sum / len(affinity))
        return jnp.clip(affinity * (target_sum / current_sum), 0.05, 0.95)

    def _hebbian_update(
        self, weights: jnp.ndarray, affinity: jnp.ndarray, mask: jnp.ndarray,
        fitness_signal: float, lr: float, anti_lr: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        if fitness_signal >= 0:
            weight_delta = lr * fitness_signal * co_active
            affinity_delta = lr * fitness_signal * active
        else:
            weight_delta = anti_lr * fitness_signal * co_active
            affinity_delta = anti_lr * fitness_signal * active

        new_weights = jnp.clip(weights + weight_delta, 0.0, 1.0)
        new_affinity = jnp.clip(affinity + affinity_delta, 0.0, 1.0)
        return new_weights, new_affinity

    def _compute_protection(
        self, affinity: jnp.ndarray, weights: jnp.ndarray, mask: jnp.ndarray,
        cross_affinity: jnp.ndarray, other_mask: jnp.ndarray, is_act: bool
    ) -> jnp.ndarray:
        active = (mask > 0.5).astype(jnp.float32)
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        n_other = max(jnp.sum(other_active), 1)

        pairwise = jnp.dot(weights, active) / n_active
        if is_act:
            cross_score = jnp.dot(cross_affinity, other_active) / n_other
        else:
            cross_score = jnp.dot(cross_affinity.T, other_active) / n_other

        return 0.55 * affinity + 0.25 * pairwise + 0.20 * cross_score * self.cross_influence

    def _mutate_palette(
        self, key: jax.random.PRNGKey, mask: jnp.ndarray, phase: str,
        protection: jnp.ndarray, num_funcs: int, min_active: int, max_active: int,
        is_act: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []

        act_probs = jax.random.uniform(key1, (num_funcs,))
        deact_probs = jax.random.uniform(key2, (num_funcs,))

        if is_act:
            act_rate = self.exploration_activate if phase == CriticalPeriodPhase.EXPLORATION else self.confirmation_activate if phase == CriticalPeriodPhase.CONFIRMATION else self.consolidation_activate
            deact_rate_max = self.confirmation_deactivate_max
            deact_rate_min = self.confirmation_deactivate_min
        else:
            act_rate = self.agg_exploration_activate if phase == CriticalPeriodPhase.EXPLORATION else self.confirmation_activate if phase == CriticalPeriodPhase.CONFIRMATION else self.consolidation_activate
            deact_rate_max = 0.12
            deact_rate_min = 0.01

        current_active = int(jnp.sum(mask > 0.5))
        use_protection = phase != CriticalPeriodPhase.EXPLORATION

        for i in range(num_funcs):
            prot = float(protection[i])
            if mask[i] < 0.5:
                if current_active + len(activated) >= max_active:
                    continue
                effective = act_rate * (0.5 + prot) if use_protection else act_rate
                if act_probs[i] < effective:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if phase == CriticalPeriodPhase.CONSOLIDATION and prot >= self.affinity_protection_threshold:
                    continue
                if prot >= self.affinity_protection_threshold:
                    dr = deact_rate_min
                else:
                    t = prot / self.affinity_protection_threshold
                    dr = deact_rate_max * (1 - t) + deact_rate_min * t
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

        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']

        alpha = 0.2
        new_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = max(-1.0, min(1.0, (best_fitness - new_ema) / max(0.1, new_ema)))

        lr_mult = self._get_phase_lr(phase)
        lr = self.learning_rate * lr_mult
        anti_lr = self.anti_hebbian_rate * lr_mult
        cross_lr = self.cross_learning_rate * lr_mult

        # Update contributions
        new_act_contrib = self._compute_contributions(state['act_mask'], fitness_signal, state['act_contributions'])
        new_agg_contrib = self._compute_contributions(state['agg_mask'], fitness_signal, state['agg_contributions'])

        # Hebbian updates
        new_act_weights, new_act_affinity = self._hebbian_update(
            state['act_hebbian_weights'], state['act_affinity'], state['act_mask'], fitness_signal, lr, anti_lr)
        new_agg_weights, new_agg_affinity = self._hebbian_update(
            state['agg_hebbian_weights'], state['agg_affinity'], state['agg_mask'], fitness_signal, lr, anti_lr)

        # Cross-domain update
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)
        if fitness_signal >= 0:
            cross_delta = cross_lr * fitness_signal * cross_active
        else:
            cross_delta = (anti_lr * 0.5) * fitness_signal * cross_active
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Competition
        new_act_affinity, act_winners, act_losers = self._apply_competition(
            new_act_affinity, new_act_contrib, state['act_mask'], phase, self.winner_k, self.loser_k)
        new_agg_affinity, agg_winners, agg_losers = self._apply_competition(
            new_agg_affinity, new_agg_contrib, state['agg_mask'], phase, self.agg_winner_k, self.agg_loser_k)

        # Normalize
        new_act_affinity = self._normalize_affinity(new_act_affinity, self.act_target_affinity_sum)
        new_agg_affinity = self._normalize_affinity(new_agg_affinity, self.agg_target_affinity_sum)

        # Protection
        act_prot = self._compute_protection(new_act_affinity, new_act_weights, state['act_mask'], new_cross, state['agg_mask'], True)
        agg_prot = self._compute_protection(new_agg_affinity, new_agg_weights, state['agg_mask'], new_cross, state['act_mask'], False)

        # Mutations
        new_act_mask, act_mut = self._mutate_palette(k_act, state['act_mask'], phase, act_prot, NUM_ACTIVATIONS, self.min_active_act, self.max_active_act, True)
        new_agg_mask, agg_mut = self._mutate_palette(k_agg, state['agg_mask'], phase, agg_prot, NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg, False)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask, 'act_affinity': new_act_affinity, 'act_hebbian_weights': new_act_weights,
            'act_contributions': new_act_contrib, 'agg_mask': new_agg_mask, 'agg_affinity': new_agg_affinity,
            'agg_hebbian_weights': new_agg_weights, 'agg_contributions': new_agg_contrib,
            'cross_affinity': new_cross, 'rng_key': key, 'generation': generation + 1,
            'stagnation_count': new_stagnation, 'best_fitness_seen': new_best, 'strategy_name': self.name,
            'phase': phase, 'fitness_history': fitness_history, 'fitness_ema': new_ema,
            'act_competition_events': state['act_competition_events'] + (1 if act_winners else 0),
            'agg_competition_events': state['agg_competition_events'] + (1 if agg_winners else 0),
        }

        metrics = {
            'act_palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'phase': phase, 'phase_changed': phase_changed, 'fitness_signal': fitness_signal,
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'sin_affinity': float(new_act_affinity[4]) if 4 < len(new_act_affinity) else 0.0,
            'act_winners': act_winners, 'act_losers': act_losers,
            'agg_winners': agg_winners, 'agg_losers': agg_losers,
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
            'act_competition_events': state['act_competition_events'],
            'agg_competition_events': state['agg_competition_events'],
        }
