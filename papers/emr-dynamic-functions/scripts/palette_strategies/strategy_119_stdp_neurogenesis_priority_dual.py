"""Strategy 119: STDP-Neurogenesis Birth Priority Dual.

Combines STDP temporal credit (#16) with neurogenesis birth mechanisms (#41).
High LTD (stagnation-correlated) regions have reduced birth rate.

Key Innovation:
- LTD tracking identifies functions correlated with stagnation
- Birth probability is REDUCED in regions with high LTD
- This prevents wasting mutations on failure-prone areas
- Successful regions (high LTP) get priority for new births

Biological basis: In the brain, regions associated with negative outcomes
have reduced neurogenesis. We apply this principle: avoid births in regions
that correlate with stagnation.

Expected: More efficient exploration by avoiding stagnation-prone regions.
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


class STDPNeurogenesisPriorityDualStrategy(PaletteEvolutionStrategy):
    """STDP-guided birth priority for dual palette evolution.

    LTD tracking identifies stagnation-prone regions.
    Birth probability is reduced in high-LTD areas, increased in high-LTP areas.

    Critical innovation: Failure avoidance through LTD-based birth inhibition.
    """

    name = "stdp_neurogenesis_priority_dual"
    description = "Dual: LTD-based birth inhibition avoids stagnation-prone regions"

    def __init__(
        self,
        # === STDP parameters ===
        ltp_window: int = 5,
        ltd_window: int = 3,
        ltp_rate: float = 0.20,
        ltd_rate: float = 0.15,
        stdp_decay: float = 0.85,
        # === Birth priority parameters ===
        ltd_birth_inhibition: float = 0.5,
        ltp_birth_boost: float = 0.3,
        ltd_birth_threshold: float = 0.3,
        base_birth_rate: float = 0.12,
        # === Sin preference ===
        sin_idx: int = 4,
        sin_birth_boost: float = 0.25,
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
        """Initialize STDP-Neurogenesis Birth Priority strategy."""
        # STDP
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.stdp_decay = stdp_decay

        # Birth priority
        self.ltd_birth_inhibition = ltd_birth_inhibition
        self.ltp_birth_boost = ltp_birth_boost
        self.ltd_birth_threshold = ltd_birth_threshold
        self.base_birth_rate = base_birth_rate

        # Sin
        self.sin_idx = sin_idx
        self.sin_birth_boost = sin_birth_boost

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
        """Initialize state with LTP/LTD tracking."""
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

        # LTP/LTD credit per function
        act_ltp = jnp.zeros(NUM_ACTIVATIONS)
        act_ltd = jnp.zeros(NUM_ACTIVATIONS)
        agg_ltp = jnp.zeros(NUM_AGGREGATIONS)
        agg_ltd = jnp.zeros(NUM_AGGREGATIONS)

        # Sin starts with slight LTP boost
        act_ltp = act_ltp.at[self.sin_idx].set(0.3)

        # Extremes start with slight LTP boost
        for idx in CORE_EXTREME_AGGS:
            agg_ltp = agg_ltp.at[idx].set(0.3)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # STDP credit
            'act_ltp': act_ltp,
            'act_ltd': act_ltd,
            'agg_ltp': agg_ltp,
            'agg_ltd': agg_ltd,
            # Stats
            'births_inhibited': 0,
            'ltp_guided_births': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1190000),
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

    def _calculate_birth_prob(self, ltp: float, ltd: float, is_sin: bool = False) -> float:
        """Calculate birth probability based on LTP/LTD credit."""
        base = self.base_birth_rate

        # LTD inhibition
        if ltd > self.ltd_birth_threshold:
            base *= (1.0 - self.ltd_birth_inhibition)

        # LTP boost
        base += ltp * self.ltp_birth_boost

        # Sin boost
        if is_sin:
            base += self.sin_birth_boost

        return min(1.0, max(0.01, base))

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with LTD-inhibited birth priority."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === UPDATE STDP CREDIT ===
        act_ltp = state['act_ltp'] * self.stdp_decay
        act_ltd = state['act_ltd'] * self.stdp_decay
        agg_ltp = state['agg_ltp'] * self.stdp_decay
        agg_ltd = state['agg_ltd'] * self.stdp_decay

        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        births_inhibited = state['births_inhibited']
        ltp_guided_births = state['ltp_guided_births']

        if improved:
            # Active functions get LTP credit
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    act_ltp = act_ltp.at[i].add(self.ltp_rate)
            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    agg_ltp = agg_ltp.at[i].add(self.ltp_rate)

            # Update affinities
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]
            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)
        else:
            # Stagnation: active functions get LTD credit
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    act_ltd = act_ltd.at[i].add(self.ltd_rate)
            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    agg_ltd = agg_ltd.at[i].add(self.ltd_rate)

        # Clamp
        act_ltp = jnp.clip(act_ltp, 0.0, 1.0)
        act_ltd = jnp.clip(act_ltd, 0.0, 1.0)
        agg_ltp = jnp.clip(agg_ltp, 0.0, 1.0)
        agg_ltd = jnp.clip(agg_ltd, 0.0, 1.0)
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)

        # === ACTIVATION BIRTH WITH PRIORITY ===
        candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
        if candidates:
            # Calculate birth probabilities
            probs = []
            for i in candidates:
                is_sin = (i == self.sin_idx)
                prob = self._calculate_birth_prob(
                    float(act_ltp[i]), float(act_ltd[i]), is_sin
                )
                probs.append(prob)

            # Weighted selection
            probs_arr = jnp.array(probs)
            probs_arr = probs_arr / probs_arr.sum()

            if jax.random.uniform(k1) < self.base_birth_rate:
                new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs_arr))

                # Check if this was LTD-inhibited
                if float(act_ltd[new_idx]) > self.ltd_birth_threshold:
                    births_inhibited += 1
                if float(act_ltp[new_idx]) > 0.2:
                    ltp_guided_births += 1

                act_mask = act_mask.at[new_idx].set(1.0)

        # === AGGREGATION BIRTH WITH PRIORITY ===
        candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
        if candidates:
            probs = []
            for i in candidates:
                prob = self._calculate_birth_prob(float(agg_ltp[i]), float(agg_ltd[i]))
                # Extreme boost
                if i in CORE_EXTREME_AGGS:
                    prob += 0.2
                probs.append(prob)

            probs_arr = jnp.array(probs)
            probs_arr = probs_arr / probs_arr.sum()

            if jax.random.uniform(k3) < self.base_birth_rate:
                new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs_arr))
                agg_mask = agg_mask.at[new_idx].set(1.0)

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
            'act_ltp': act_ltp,
            'act_ltd': act_ltd,
            'agg_ltp': agg_ltp,
            'agg_ltd': agg_ltd,
            'births_inhibited': births_inhibited,
            'ltp_guided_births': ltp_guided_births,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'births_inhibited': births_inhibited,
            'ltp_guided_births': ltp_guided_births,
            'mean_act_ltp': float(act_ltp.mean()),
            'mean_act_ltd': float(act_ltd.mean()),
        }

        return new_state, metrics
