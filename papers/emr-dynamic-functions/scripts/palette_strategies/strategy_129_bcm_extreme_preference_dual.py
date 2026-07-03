"""Strategy 129: BCM-Extreme Preference Dual.

Combines BCM Metaplasticity (#15) with Aggregation-led extreme preference (#101).
BCM threshold is LOWER for extreme aggs, creating automatic extreme bias.

Key Innovation:
- BCM sliding thresholds adapt based on function activity
- Extreme aggregations (max/min) have permanent threshold discount
- Sin also gets threshold discount for easier retention
- Threshold-based bias rather than explicit protection rules

Biological basis: BCM theory shows that synaptic modification thresholds
slide based on average activity. We bias these thresholds to favor extremes.

Expected: Automatic extreme preference through differential thresholds.
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
)


class BCMExtremePreferenceDualStrategy(PaletteEvolutionStrategy):
    """BCM-Extreme Preference Strategy for dual palette evolution.

    BCM sliding thresholds are biased to be lower for
    extreme aggregations (max/min), making them more likely to be retained.

    Critical innovation: Differential thresholds create automatic preference.
    """

    name = "bcm_extreme_preference_dual"
    description = "Dual: BCM thresholds biased lower for extremes and sin"

    def __init__(
        self,
        # === BCM parameters ===
        bcm_threshold_lr: float = 0.1,
        bcm_window: int = 10,
        bcm_min_threshold: float = 0.2,
        bcm_max_threshold: float = 0.9,
        bcm_initial_threshold: float = 0.5,
        # === Extreme preference ===
        extreme_bcm_discount: float = 0.3,
        discount_decay_per_capture: float = 0.1,
        # === Sin preference ===
        sin_idx: int = 4,
        sin_discount: float = 0.2,
        # === Agg-led timing ===
        agg_lead_generations: int = 15,
        agg_lead_act_mutation: float = 0.05,
        normal_act_mutation: float = 0.12,
        agg_lead_agg_mutation: float = 0.15,
        normal_agg_mutation: float = 0.08,
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
        """Initialize BCM-Extreme Preference strategy."""
        # BCM parameters
        self.bcm_threshold_lr = bcm_threshold_lr
        self.bcm_window = bcm_window
        self.bcm_min_threshold = bcm_min_threshold
        self.bcm_max_threshold = bcm_max_threshold
        self.bcm_initial_threshold = bcm_initial_threshold

        # Extreme preference
        self.extreme_bcm_discount = extreme_bcm_discount
        self.discount_decay_per_capture = discount_decay_per_capture

        # Sin preference
        self.sin_idx = sin_idx
        self.sin_discount = sin_discount

        # Agg-led timing
        self.agg_lead_generations = agg_lead_generations
        self.agg_lead_act_mutation = agg_lead_act_mutation
        self.normal_act_mutation = normal_act_mutation
        self.agg_lead_agg_mutation = agg_lead_agg_mutation
        self.normal_agg_mutation = normal_agg_mutation

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
        """Initialize state with BCM thresholds."""
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

        # BCM thresholds - start at initial, with discounts applied
        act_thresholds = jnp.ones(NUM_ACTIVATIONS) * self.bcm_initial_threshold
        agg_thresholds = jnp.ones(NUM_AGGREGATIONS) * self.bcm_initial_threshold

        # Apply extreme discount to aggregation thresholds
        for idx in CORE_EXTREME_AGGS:
            agg_thresholds = agg_thresholds.at[idx].set(
                self.bcm_initial_threshold - self.extreme_bcm_discount
            )

        # Apply sin discount
        act_thresholds = act_thresholds.at[self.sin_idx].set(
            self.bcm_initial_threshold - self.sin_discount
        )

        # Activity history (simplified - we'll use running averages)
        act_activity = jnp.zeros(NUM_ACTIVATIONS)
        agg_activity = jnp.zeros(NUM_AGGREGATIONS)

        # Track captured extremes
        captured_extremes = jnp.zeros(NUM_AGGREGATIONS)

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
            # Activity tracking
            'act_activity': act_activity,
            'agg_activity': agg_activity,
            # Captured tracking
            'captured_extremes': captured_extremes,
            # Stats
            'threshold_adjustments': 0,
            'extreme_captures': 0,
            'sin_captures': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1290000),
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

    def _get_mutation_rates(self, generation: int) -> Tuple[float, float]:
        """Get mutation rates based on agg-led timing."""
        if generation < self.agg_lead_generations:
            return self.agg_lead_act_mutation, self.agg_lead_agg_mutation
        else:
            return self.normal_act_mutation, self.normal_agg_mutation

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with BCM threshold-based selection."""
        key, k1, k2, k3, k4, k5, k6 = jax.random.split(state['rng_key'], 7)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        act_mutation_rate, agg_mutation_rate = self._get_mutation_rates(generation)

        # === UPDATE BCM THRESHOLDS ===
        act_thresholds = state['act_thresholds']
        agg_thresholds = state['agg_thresholds']
        act_activity = state['act_activity']
        agg_activity = state['agg_activity']
        captured_extremes = state['captured_extremes']
        threshold_adjustments = state['threshold_adjustments']

        # Simulate activity based on active functions (simplified)
        # Active functions accumulate activity when fitness improves
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']

        if improved:
            # Active functions get activity credit
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    act_activity = act_activity.at[i].set(
                        act_activity[i] * 0.9 + 0.3  # Running average
                    )
            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    agg_activity = agg_activity.at[i].set(
                        agg_activity[i] * 0.9 + 0.3
                    )
                    # Track captured extremes
                    if i in CORE_EXTREME_AGGS:
                        captured_extremes = captured_extremes.at[i].set(1.0)
        else:
            # Decay activity
            act_activity = act_activity * 0.95
            agg_activity = agg_activity * 0.95

        # Update thresholds toward mean activity (BCM sliding threshold)
        for i in range(NUM_ACTIVATIONS):
            current = float(act_thresholds[i])
            activity = float(act_activity[i])
            delta = self.bcm_threshold_lr * (activity - current)
            new_thresh = current + delta

            # Apply sin discount
            if i == self.sin_idx:
                new_thresh = max(self.bcm_min_threshold, new_thresh - self.sin_discount)

            new_thresh = max(self.bcm_min_threshold, min(self.bcm_max_threshold, new_thresh))
            act_thresholds = act_thresholds.at[i].set(new_thresh)

        for i in range(NUM_AGGREGATIONS):
            current = float(agg_thresholds[i])
            activity = float(agg_activity[i])
            delta = self.bcm_threshold_lr * (activity - current)
            new_thresh = current + delta

            # Apply extreme discount
            if i in CORE_EXTREME_AGGS:
                discount = self.extreme_bcm_discount
                # Decay discount if captured
                if float(captured_extremes[i]) > 0.5:
                    discount *= (1 - self.discount_decay_per_capture)
                new_thresh = max(self.bcm_min_threshold, new_thresh - discount)

            new_thresh = max(self.bcm_min_threshold, min(self.bcm_max_threshold, new_thresh))
            agg_thresholds = agg_thresholds.at[i].set(new_thresh)

        threshold_adjustments += 1

        # === ACTIVATION UPDATE ===
        act_affinities = state['act_affinities']
        sin_captures = state['sin_captures']

        # Mutation: add new function weighted by inverse threshold
        if jax.random.uniform(k1) < act_mutation_rate:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                # Sin has higher probability due to low threshold
                if self.sin_idx in candidates:
                    sin_prob = 0.3 + (1.0 - float(act_thresholds[self.sin_idx])) * 0.3
                    if jax.random.uniform(k2) < sin_prob:
                        act_mask = act_mask.at[self.sin_idx].set(1.0)
                        sin_captures += 1
                        candidates.remove(self.sin_idx)

                if candidates and jax.random.uniform(k3) < act_mutation_rate:
                    # Weight by inverse threshold
                    weights = jnp.array([1.0 - float(act_thresholds[i]) + 0.1 for i in candidates])
                    probs = weights / weights.sum()
                    new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs))
                    act_mask = act_mask.at[new_idx].set(1.0)

        # === AGGREGATION UPDATE ===
        agg_affinities = state['agg_affinities']
        extreme_captures = state['extreme_captures']

        # Mutation: extreme aggs have high probability due to low thresholds
        if jax.random.uniform(k5) < agg_mutation_rate:
            candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                # Check for missing extremes first
                missing_extremes = [i for i in CORE_EXTREME_AGGS if float(agg_mask[i]) < 0.5]
                if missing_extremes:
                    # High probability of adding extreme
                    extreme_prob = sum(1.0 - float(agg_thresholds[i]) for i in missing_extremes) / len(missing_extremes)
                    if jax.random.uniform(k6) < extreme_prob:
                        new_idx = int(jax.random.choice(k6, jnp.array(missing_extremes)))
                        agg_mask = agg_mask.at[new_idx].set(1.0)
                        extreme_captures += 1
                    else:
                        # Add from any candidate
                        weights = jnp.array([1.0 - float(agg_thresholds[i]) + 0.1 for i in candidates])
                        probs = weights / weights.sum()
                        new_idx = int(jax.random.choice(k6, jnp.array(candidates), p=probs))
                        agg_mask = agg_mask.at[new_idx].set(1.0)
                else:
                    weights = jnp.array([1.0 - float(agg_thresholds[i]) + 0.1 for i in candidates])
                    probs = weights / weights.sum()
                    new_idx = int(jax.random.choice(k6, jnp.array(candidates), p=probs))
                    agg_mask = agg_mask.at[new_idx].set(1.0)

        # Update affinities on improvement
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)

        # Clamp affinities
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
            'captured_extremes': captured_extremes,
            'threshold_adjustments': threshold_adjustments,
            'extreme_captures': extreme_captures,
            'sin_captures': sin_captures,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'agg_led_phase': generation < self.agg_lead_generations,
            'threshold_adjustments': threshold_adjustments,
            'mean_act_threshold': float(act_thresholds.mean()),
            'mean_agg_threshold': float(agg_thresholds.mean()),
        }

        return new_state, metrics
