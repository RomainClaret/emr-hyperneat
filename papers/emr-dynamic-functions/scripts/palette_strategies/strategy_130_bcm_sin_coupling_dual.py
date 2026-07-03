"""Strategy 130: BCM-Sin-Coupling Threshold Dual.

Combines BCM Metaplasticity (#15) with Aggregation-led (#101) and sin-extreme coupling.
Sin's BCM threshold is reduced when extreme aggs are active.

Key Innovation:
- Sin's BCM threshold is coupled to extreme agg presence
- When extreme aggs (max/min) are active, sin threshold drops
- Creates ecosystem-driven sin discovery
- Averaging aggs slightly INCREASE sin threshold (penalty)

Biological basis: BCM thresholds can be modulated by global network state.
We use extreme agg presence as a "neuromodulatory" signal that reduces
sin's threshold, making it easier to discover and retain.

Expected: Ecosystem-driven sin discovery through extreme-coupling.
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
    CORE_EXTREME_AGGS,
    AVERAGING_AGGS,
)


class BCMSinCouplingDualStrategy(PaletteEvolutionStrategy):
    """BCM-Sin-Coupling strategy for dual palette evolution.

    Sin's BCM threshold is modulated by extreme agg
    presence, creating ecosystem-driven discovery dynamics.

    Critical innovation: Cross-domain threshold coupling for sin-extreme synergy.
    """

    name = "bcm_sin_coupling_dual"
    description = "Dual: Sin threshold drops when extreme aggs are active"

    def __init__(
        self,
        # === BCM parameters ===
        bcm_threshold_lr: float = 0.1,
        bcm_min_threshold: float = 0.2,
        bcm_max_threshold: float = 0.9,
        bcm_initial_threshold: float = 0.5,
        # === Sin-extreme coupling ===
        extreme_sin_coupling_discount: float = 0.25,
        averaging_sin_penalty: float = 0.1,
        sin_idx: int = 4,
        # === General BCM ===
        extreme_base_discount: float = 0.2,
        # === Mutation rates ===
        base_mutation_rate: float = 0.10,
        coupling_mutation_boost: float = 0.15,
        # === General parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize BCM-Sin-Coupling strategy."""
        # BCM
        self.bcm_threshold_lr = bcm_threshold_lr
        self.bcm_min_threshold = bcm_min_threshold
        self.bcm_max_threshold = bcm_max_threshold
        self.bcm_initial_threshold = bcm_initial_threshold

        # Sin-extreme coupling
        self.extreme_sin_coupling_discount = extreme_sin_coupling_discount
        self.averaging_sin_penalty = averaging_sin_penalty
        self.sin_idx = sin_idx

        # General BCM
        self.extreme_base_discount = extreme_base_discount

        # Mutation
        self.base_mutation_rate = base_mutation_rate
        self.coupling_mutation_boost = coupling_mutation_boost

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with coupled BCM thresholds."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)

        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(agg_affinities[i] + 0.2)

        # BCM thresholds
        act_thresholds = jnp.ones(NUM_ACTIVATIONS) * self.bcm_initial_threshold
        agg_thresholds = jnp.ones(NUM_AGGREGATIONS) * self.bcm_initial_threshold

        # Apply extreme discount
        for idx in CORE_EXTREME_AGGS:
            agg_thresholds = agg_thresholds.at[idx].set(
                self.bcm_initial_threshold - self.extreme_base_discount
            )

        # Activity tracking
        act_activity = jnp.zeros(NUM_ACTIVATIONS)
        agg_activity = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # BCM thresholds
            'act_thresholds': act_thresholds,
            'agg_thresholds': agg_thresholds,
            # Activity
            'act_activity': act_activity,
            'agg_activity': agg_activity,
            # Stats
            'coupling_activations': 0,
            'sin_threshold_reductions': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1300000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _count_active_extremes(self, agg_mask: jnp.ndarray) -> int:
        """Count active extreme aggregations."""
        return sum(1 for idx in CORE_EXTREME_AGGS if float(agg_mask[idx]) > 0.5)

    def _count_active_averaging(self, agg_mask: jnp.ndarray) -> int:
        """Count active averaging aggregations."""
        return sum(1 for idx in AVERAGING_AGGS if float(agg_mask[idx]) > 0.5)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with sin-extreme coupling."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === UPDATE THRESHOLDS WITH COUPLING ===
        act_thresholds = state['act_thresholds']
        agg_thresholds = state['agg_thresholds']
        act_activity = state['act_activity']
        agg_activity = state['agg_activity']

        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        coupling_activations = state['coupling_activations']
        sin_threshold_reductions = state['sin_threshold_reductions']

        # Count active extremes and averaging
        n_extremes = self._count_active_extremes(agg_mask)
        n_averaging = self._count_active_averaging(agg_mask)

        # Update activity
        if improved:
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    act_activity = act_activity.at[i].set(act_activity[i] * 0.9 + 0.3)
            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    agg_activity = agg_activity.at[i].set(agg_activity[i] * 0.9 + 0.3)
        else:
            act_activity = act_activity * 0.95
            agg_activity = agg_activity * 0.95

        # Update BCM thresholds
        for i in range(NUM_ACTIVATIONS):
            current = float(act_thresholds[i])
            activity = float(act_activity[i])
            delta = self.bcm_threshold_lr * (activity - current)
            new_thresh = current + delta

            # Sin threshold coupling
            if i == self.sin_idx:
                # Extreme discount
                extreme_discount = n_extremes * self.extreme_sin_coupling_discount
                # Averaging penalty
                averaging_penalty = n_averaging * self.averaging_sin_penalty
                # Net effect
                new_thresh -= extreme_discount
                new_thresh += averaging_penalty

                if extreme_discount > 0:
                    sin_threshold_reductions += 1
                    coupling_activations += 1

            new_thresh = max(self.bcm_min_threshold, min(self.bcm_max_threshold, new_thresh))
            act_thresholds = act_thresholds.at[i].set(new_thresh)

        for i in range(NUM_AGGREGATIONS):
            current = float(agg_thresholds[i])
            activity = float(agg_activity[i])
            delta = self.bcm_threshold_lr * (activity - current)
            new_thresh = current + delta

            if i in CORE_EXTREME_AGGS:
                new_thresh -= self.extreme_base_discount

            new_thresh = max(self.bcm_min_threshold, min(self.bcm_max_threshold, new_thresh))
            agg_thresholds = agg_thresholds.at[i].set(new_thresh)

        # === ACTIVATION MUTATION ===
        candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
        if candidates:
            # Calculate mutation probability
            mutation_rate = self.base_mutation_rate
            if n_extremes > 0:
                # Boost mutation when extremes are active
                mutation_rate += self.coupling_mutation_boost

            if jax.random.uniform(k1) < mutation_rate:
                # Weight by inverse threshold
                weights = []
                for i in candidates:
                    w = 1.0 - float(act_thresholds[i]) + 0.1
                    if i == self.sin_idx:
                        # Extra boost if extremes are active
                        w += n_extremes * 0.2
                    weights.append(w)

                probs = jnp.array(weights)
                probs = probs / probs.sum()

                new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs))
                act_mask = act_mask.at[new_idx].set(1.0)

        # === AGGREGATION MUTATION ===
        candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k3) < self.base_mutation_rate:
            weights = []
            for i in candidates:
                w = 1.0 - float(agg_thresholds[i]) + 0.1
                if i in CORE_EXTREME_AGGS:
                    w += 0.3
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs))
            agg_mask = agg_mask.at[new_idx].set(1.0)

        # Update affinities
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)

        # Clamp
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)

        # Ensure minimum diversity
        if sum(float(act_mask[i]) for i in range(NUM_ACTIVATIONS)) < self.min_active_act:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k1, jnp.array(candidates)))
                act_mask = act_mask.at[new_idx].set(1.0)

        if sum(float(agg_mask[i]) for i in range(NUM_AGGREGATIONS)) < self.min_active_agg:
            candidates = [i for i in CORE_EXTREME_AGGS if float(agg_mask[i]) < 0.5]
            if not candidates:
                candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k3, jnp.array(candidates)))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        new_state = {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_thresholds': act_thresholds,
            'agg_thresholds': agg_thresholds,
            'act_activity': act_activity,
            'agg_activity': agg_activity,
            'coupling_activations': coupling_activations,
            'sin_threshold_reductions': sin_threshold_reductions,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'n_active_extremes': n_extremes,
            'sin_threshold': float(act_thresholds[self.sin_idx]),
            'coupling_activations': coupling_activations,
            'sin_threshold_reductions': sin_threshold_reductions,
        }

        return new_state, metrics
