"""Strategy 58D: Anchor Inhibition Dual (Probabilistic Protection for Both Domains).

Extends AnchorInhibitionStrategy to jointly evolve BOTH activation AND aggregation
function palettes using fitness-contribution-based anchoring.

Key dual mechanisms:
1. Dual anchor strength - separate anchoring for act and agg functions
2. Dual contribution tracking - fitness contribution estimated per domain
3. Cross-domain anchor boost - success boosts anchors in partner domain
4. Coordinated protection - anchored pairs get extra protection

Expected: Fitness-driven protection in both domains
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


class AnchorInhibitionDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with fitness-based anchoring.

    Both activation and aggregation functions build anchor strength
    based on their contribution to fitness. Anchored functions are
    probabilistically protected from removal.
    """

    name = "anchor_inhibition_dual"
    description = "Dual: Probabilistic protection based on fitness contribution"

    def __init__(
        self,
        # Anchor dynamics
        anchor_learning_rate: float = 0.15,
        anchor_decay: float = 0.95,
        anchor_threshold: float = 0.6,
        anchor_max: float = 1.5,
        # Protection
        removal_resistance: float = 0.95,
        base_selection_prob: float = 0.7,
        # Contribution estimation
        contribution_window: int = 10,
        min_samples: int = 3,
        # Cross-domain
        cross_anchor_rate: float = 0.1,
        # Exploration
        exploration_rate: float = 0.1,
        stagnation_exploration: int = 8,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Anchor Inhibition Dual strategy."""
        # Anchor dynamics
        self.anchor_learning_rate = anchor_learning_rate
        self.anchor_decay = anchor_decay
        self.anchor_threshold = anchor_threshold
        self.anchor_max = anchor_max

        # Protection
        self.removal_resistance = removal_resistance
        self.base_selection_prob = base_selection_prob

        # Contribution
        self.contribution_window = contribution_window
        self.min_samples = min_samples

        # Cross-domain
        self.cross_anchor_rate = cross_anchor_rate

        # Exploration
        self.exploration_rate = exploration_rate
        self.stagnation_exploration = stagnation_exploration

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual anchor tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize anchor strengths
        act_anchor = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_anchor = act_anchor.at[i].set(0.3)

        agg_anchor = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_anchor = agg_anchor.at[i].set(0.3)

        # Cross-domain anchor affinity
        cross_anchor = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_anchor': act_anchor,
            'act_fitness_active': [[] for _ in range(NUM_ACTIVATIONS)],
            'act_fitness_inactive': [[] for _ in range(NUM_ACTIVATIONS)],
            'act_contribution': jnp.zeros(NUM_ACTIVATIONS),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_anchor': agg_anchor,
            'agg_fitness_active': [[] for _ in range(NUM_AGGREGATIONS)],
            'agg_fitness_inactive': [[] for _ in range(NUM_AGGREGATIONS)],
            'agg_contribution': jnp.zeros(NUM_AGGREGATIONS),
            # Cross-domain
            'cross_anchor': cross_anchor,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 585858),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Tracking
            'act_protection_count': jnp.zeros(NUM_ACTIVATIONS),
            'agg_protection_count': jnp.zeros(NUM_AGGREGATIONS),
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_contribution_tracking(
        self,
        fitness_active: List[List],
        fitness_inactive: List[List],
        mask: jnp.ndarray,
        generation: int,
        fitness: float,
        n_funcs: int,
    ) -> Tuple[List[List], List[List]]:
        """Update fitness tracking for contribution estimation."""
        new_active = [list(h) for h in fitness_active]
        new_inactive = [list(h) for h in fitness_inactive]

        for i in range(n_funcs):
            if mask[i] > 0.5:
                new_active[i].append((generation, fitness))
            else:
                new_inactive[i].append((generation, fitness))

            if len(new_active[i]) > self.contribution_window:
                new_active[i] = new_active[i][-self.contribution_window:]
            if len(new_inactive[i]) > self.contribution_window:
                new_inactive[i] = new_inactive[i][-self.contribution_window:]

        return new_active, new_inactive

    def _estimate_contributions(
        self,
        fitness_active: List[List],
        fitness_inactive: List[List],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Estimate contribution of each function."""
        contributions = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            active_samples = fitness_active[i]
            inactive_samples = fitness_inactive[i]

            if len(active_samples) >= self.min_samples and len(inactive_samples) >= self.min_samples:
                mean_active = np.mean([f for _, f in active_samples])
                mean_inactive = np.mean([f for _, f in inactive_samples])
                contribution = mean_active - mean_inactive
                contributions = contributions.at[i].set(contribution)
            elif len(active_samples) >= self.min_samples:
                mean_active = np.mean([f for _, f in active_samples])
                contributions = contributions.at[i].set(mean_active * 0.5)

        return contributions

    def _update_anchor_strength(
        self,
        anchor: jnp.ndarray,
        contributions: jnp.ndarray,
        mask: jnp.ndarray,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update anchor strengths based on contributions."""
        new_anchor = anchor.copy()

        for i in range(n_funcs):
            contribution = float(contributions[i])
            if contribution > 0 and mask[i] > 0.5:
                delta = self.anchor_learning_rate * contribution
                new_anchor = new_anchor.at[i].set(
                    min(float(anchor[i]) + delta, self.anchor_max)
                )
            elif contribution < 0:
                new_anchor = new_anchor.at[i].set(float(anchor[i]) * self.anchor_decay)
            else:
                new_anchor = new_anchor.at[i].set(float(anchor[i]) * 0.98)

        return new_anchor

    def _apply_cross_anchor_boost(
        self,
        act_anchor: jnp.ndarray,
        agg_anchor: jnp.ndarray,
        cross_anchor: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply cross-domain anchor boost on success."""
        if improvement <= 0:
            return act_anchor, agg_anchor, cross_anchor * 0.99

        new_act = act_anchor.copy()
        new_agg = agg_anchor.copy()

        # Update cross-domain matrix
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        new_cross = cross_anchor + self.cross_anchor_rate * improvement * co_active
        new_cross = jnp.clip(new_cross, 0.0, 1.5)

        # Boost from partner domain
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                partner_boost = float(jnp.mean(new_cross[i, :] * active_agg))
                new_act = new_act.at[i].set(
                    min(float(new_act[i]) + partner_boost * 0.1, self.anchor_max)
                )

        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                partner_boost = float(jnp.mean(new_cross[:, i] * active_act))
                new_agg = new_agg.at[i].set(
                    min(float(new_agg[i]) + partner_boost * 0.1, self.anchor_max)
                )

        return new_act, new_agg, new_cross

    def _select_palette_with_anchors(
        self,
        anchor: jnp.ndarray,
        contributions: jnp.ndarray,
        current_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
        stagnation: int,
        palette_size: int,
        min_active: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select palette with anchor protection."""
        key1, key2, key3 = jax.random.split(key, 3)

        new_mask = jnp.zeros(n_funcs)
        protection_events = jnp.zeros(n_funcs)

        current_palette = mask_to_indices(current_mask)

        for i in current_palette:
            if anchor[i] >= self.anchor_threshold:
                if jax.random.uniform(key1) < self.removal_resistance:
                    new_mask = new_mask.at[i].set(1.0)
                    protection_events = protection_events.at[i].add(1)
                key1, _ = jax.random.split(key1)
            else:
                if jax.random.uniform(key1) < self.base_selection_prob:
                    new_mask = new_mask.at[i].set(1.0)
                key1, _ = jax.random.split(key1)

        n_active = int(jnp.sum(new_mask))
        if n_active < min_active:
            priority = anchor + contributions * 0.5
            inactive = jnp.where(new_mask < 0.5)[0]
            sorted_inactive = sorted(inactive.tolist(), key=lambda x: float(priority[x]), reverse=True)
            for idx in sorted_inactive[:min_active - n_active]:
                new_mask = new_mask.at[idx].set(1.0)

        n_active = int(jnp.sum(new_mask))
        if n_active > palette_size:
            active = mask_to_indices(new_mask)
            sorted_active = sorted(active, key=lambda x: float(anchor[x]), reverse=True)
            new_mask = jnp.zeros(n_funcs)
            for idx in sorted_active[:palette_size]:
                new_mask = new_mask.at[idx].set(1.0)

        explore = (
            jax.random.uniform(key2) < self.exploration_rate or
            stagnation >= self.stagnation_exploration
        )
        if explore:
            inactive = [i for i in range(n_funcs) if new_mask[i] < 0.5]
            if inactive:
                new_idx = inactive[int(jax.random.randint(key3, (), 0, len(inactive)))]
                if int(jnp.sum(new_mask)) >= palette_size:
                    active = mask_to_indices(new_mask)
                    replaceable = [i for i in active if anchor[i] < self.anchor_threshold]
                    if replaceable:
                        replace_idx = min(replaceable, key=lambda x: float(anchor[x]))
                        new_mask = new_mask.at[replace_idx].set(0.0)
                        new_mask = new_mask.at[new_idx].set(1.0)
                else:
                    new_mask = new_mask.at[new_idx].set(1.0)

        return new_mask, protection_events

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual anchor-based protection."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update contribution tracking
        act_active, act_inactive = self._update_contribution_tracking(
            state['act_fitness_active'], state['act_fitness_inactive'],
            state['act_mask'], generation, best_fitness, NUM_ACTIVATIONS
        )
        agg_active, agg_inactive = self._update_contribution_tracking(
            state['agg_fitness_active'], state['agg_fitness_inactive'],
            state['agg_mask'], generation, best_fitness, NUM_AGGREGATIONS
        )

        # Estimate contributions
        act_contrib = self._estimate_contributions(act_active, act_inactive, NUM_ACTIVATIONS)
        agg_contrib = self._estimate_contributions(agg_active, agg_inactive, NUM_AGGREGATIONS)

        # Update anchor strengths
        new_act_anchor = self._update_anchor_strength(
            state['act_anchor'], act_contrib, state['act_mask'], NUM_ACTIVATIONS
        )
        new_agg_anchor = self._update_anchor_strength(
            state['agg_anchor'], agg_contrib, state['agg_mask'], NUM_AGGREGATIONS
        )

        # Apply cross-domain boost
        new_act_anchor, new_agg_anchor, new_cross = self._apply_cross_anchor_boost(
            new_act_anchor, new_agg_anchor, state['cross_anchor'],
            state['act_mask'], state['agg_mask'], improvement
        )

        # Select palettes with anchors
        new_act_mask, act_prot = self._select_palette_with_anchors(
            new_act_anchor, act_contrib, state['act_mask'], k_act,
            new_stagnation, self.act_palette_size, self.min_active_act, NUM_ACTIVATIONS
        )
        new_agg_mask, agg_prot = self._select_palette_with_anchors(
            new_agg_anchor, agg_contrib, state['agg_mask'], k_agg,
            new_stagnation, self.agg_palette_size, self.min_active_agg, NUM_AGGREGATIONS
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_anchor': new_act_anchor,
            'act_fitness_active': act_active,
            'act_fitness_inactive': act_inactive,
            'act_contribution': act_contrib,
            'agg_mask': new_agg_mask,
            'agg_anchor': new_agg_anchor,
            'agg_fitness_active': agg_active,
            'agg_fitness_inactive': agg_inactive,
            'agg_contribution': agg_contrib,
            'cross_anchor': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'act_protection_count': state['act_protection_count'] + act_prot,
            'agg_protection_count': state['agg_protection_count'] + agg_prot,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        act_anchored = [i for i in range(NUM_ACTIVATIONS) if new_act_anchor[i] >= self.anchor_threshold]
        agg_anchored = [i for i in range(NUM_AGGREGATIONS) if new_agg_anchor[i] >= self.anchor_threshold]

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Anchor stats
            'act_n_anchored': len(act_anchored),
            'agg_n_anchored': len(agg_anchored),
            'act_mean_anchor': float(jnp.mean(new_act_anchor)),
            'agg_mean_anchor': float(jnp.mean(new_agg_anchor)),
            # Contribution
            'act_mean_contribution': float(jnp.mean(act_contrib)),
            'agg_mean_contribution': float(jnp.mean(agg_contrib)),
            # Cross-domain
            'cross_mean_anchor': float(jnp.mean(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_anchor': float(new_act_anchor[4]),
            'sin_contribution': float(act_contrib[4]),
            'sin_is_anchored': 4 in act_anchored,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual anchor status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        act_anchored = [i for i in range(NUM_ACTIVATIONS) if state['act_anchor'][i] >= self.anchor_threshold]
        agg_anchored = [i for i in range(NUM_AGGREGATIONS) if state['agg_anchor'][i] >= self.anchor_threshold]

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_n_anchored': len(act_anchored),
            'agg_n_anchored': len(agg_anchored),
            'act_mean_anchor': float(jnp.mean(state['act_anchor'])),
            'agg_mean_anchor': float(jnp.mean(state['agg_anchor'])),
            'cross_mean_anchor': float(jnp.mean(state['cross_anchor'])),
            'sin_anchor': float(state['act_anchor'][4]),
            'sin_is_anchored': 4 in act_anchored,
        }
