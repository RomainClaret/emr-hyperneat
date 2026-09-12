"""Strategy 120: STDP-Neurogenesis Lineage Selection Dual.

Combines STDP lineage tracking (#16) with neurogenesis maturation (#113).
Surviving functions' LTP history influences birth probability of similar functions.

Key Innovation:
- Functions that survive and have high LTP credit create "successful lineages"
- Birth probability is increased for functions similar to successful survivors
- Similarity defined by index proximity (nearby functions in palette)
- Creates evolutionary momentum toward successful functional regions

Biological basis: In natural selection, successful organisms' offspring have
higher survival rates. Here, functions near successful survivors (in function
space) have increased birth probability.

Expected: Evolutionary momentum concentrating exploration near winners.
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


class STDPNeurogenesisLineageDualStrategy(PaletteEvolutionStrategy):
    """STDP-guided lineage selection for dual palette evolution.

    Surviving functions with high LTP create lineages
    that increase birth probability for nearby functions.

    Critical innovation: Evolutionary momentum through lineage inheritance.
    """

    name = "stdp_neurogenesis_lineage_dual"
    description = "Dual: Successful survivors boost birth probability of similar functions"

    def __init__(
        self,
        # === STDP parameters ===
        ltp_rate: float = 0.20,
        ltd_rate: float = 0.10,
        stdp_decay: float = 0.90,
        # === Lineage parameters ===
        lineage_inheritance: float = 0.3,
        similarity_radius: int = 2,
        lineage_survival_threshold: float = 0.4,
        lineage_decay: float = 0.95,
        # === Birth parameters ===
        base_birth_rate: float = 0.12,
        lineage_birth_boost: float = 0.25,
        # === Sin preference ===
        sin_idx: int = 4,
        sin_lineage_boost: float = 0.3,
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
        """Initialize STDP-Neurogenesis Lineage strategy."""
        # STDP
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.stdp_decay = stdp_decay

        # Lineage
        self.lineage_inheritance = lineage_inheritance
        self.similarity_radius = similarity_radius
        self.lineage_survival_threshold = lineage_survival_threshold
        self.lineage_decay = lineage_decay

        # Birth
        self.base_birth_rate = base_birth_rate
        self.lineage_birth_boost = lineage_birth_boost

        # Sin
        self.sin_idx = sin_idx
        self.sin_lineage_boost = sin_lineage_boost

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
        """Initialize state with lineage tracking."""
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

        # LTP credit
        act_ltp = jnp.zeros(NUM_ACTIVATIONS)
        agg_ltp = jnp.zeros(NUM_AGGREGATIONS)

        # Lineage strength (accumulated from successful survivors)
        act_lineage = jnp.zeros(NUM_ACTIVATIONS)
        agg_lineage = jnp.zeros(NUM_AGGREGATIONS)

        # Initial lineage for sin and extremes
        act_lineage = act_lineage.at[self.sin_idx].set(0.3)
        for idx in CORE_EXTREME_AGGS:
            agg_lineage = agg_lineage.at[idx].set(0.3)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # LTP credit
            'act_ltp': act_ltp,
            'agg_ltp': agg_ltp,
            # Lineage strength
            'act_lineage': act_lineage,
            'agg_lineage': agg_lineage,
            # Stats
            'lineage_propagations': 0,
            'lineage_births': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1200000),
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

    def _propagate_lineage(self, lineage: jnp.ndarray, successful_idx: int, n_funcs: int) -> jnp.ndarray:
        """Propagate lineage from successful survivor to nearby functions."""
        for offset in range(-self.similarity_radius, self.similarity_radius + 1):
            if offset == 0:
                continue
            neighbor = successful_idx + offset
            if 0 <= neighbor < n_funcs:
                # Lineage decreases with distance
                distance_factor = 1.0 - abs(offset) / (self.similarity_radius + 1)
                boost = self.lineage_inheritance * distance_factor
                lineage = lineage.at[neighbor].add(boost)
        return lineage

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with lineage-based birth priority."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === UPDATE LTP AND LINEAGE ===
        act_ltp = state['act_ltp'] * self.stdp_decay
        agg_ltp = state['agg_ltp'] * self.stdp_decay
        act_lineage = state['act_lineage'] * self.lineage_decay
        agg_lineage = state['agg_lineage'] * self.lineage_decay

        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        lineage_propagations = state['lineage_propagations']
        lineage_births = state['lineage_births']

        if improved:
            # Active functions get LTP credit
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    act_ltp = act_ltp.at[i].add(self.ltp_rate)
                    # High LTP survivors propagate lineage
                    if float(act_ltp[i]) > self.lineage_survival_threshold:
                        act_lineage = self._propagate_lineage(act_lineage, i, NUM_ACTIVATIONS)
                        lineage_propagations += 1

            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    agg_ltp = agg_ltp.at[i].add(self.ltp_rate)
                    if float(agg_ltp[i]) > self.lineage_survival_threshold:
                        agg_lineage = self._propagate_lineage(agg_lineage, i, NUM_AGGREGATIONS)
                        lineage_propagations += 1

            # Update affinities
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]
            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)

        # Clamp
        act_ltp = jnp.clip(act_ltp, 0.0, 1.0)
        agg_ltp = jnp.clip(agg_ltp, 0.0, 1.0)
        act_lineage = jnp.clip(act_lineage, 0.0, 1.0)
        agg_lineage = jnp.clip(agg_lineage, 0.0, 1.0)
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)

        # === ACTIVATION BIRTH WITH LINEAGE BOOST ===
        candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k1) < self.base_birth_rate:
            # Calculate birth probabilities based on lineage
            probs = []
            for i in candidates:
                prob = 0.1 + float(act_lineage[i]) * self.lineage_birth_boost
                if i == self.sin_idx:
                    prob += self.sin_lineage_boost
                probs.append(prob)

            probs_arr = jnp.array(probs)
            probs_arr = probs_arr / probs_arr.sum()

            new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs_arr))
            act_mask = act_mask.at[new_idx].set(1.0)

            if float(act_lineage[new_idx]) > 0.1:
                lineage_births += 1

        # === AGGREGATION BIRTH WITH LINEAGE BOOST ===
        candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k3) < self.base_birth_rate:
            probs = []
            for i in candidates:
                prob = 0.1 + float(agg_lineage[i]) * self.lineage_birth_boost
                if i in CORE_EXTREME_AGGS:
                    prob += 0.25
                probs.append(prob)

            probs_arr = jnp.array(probs)
            probs_arr = probs_arr / probs_arr.sum()

            new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs_arr))
            agg_mask = agg_mask.at[new_idx].set(1.0)

            if float(agg_lineage[new_idx]) > 0.1:
                lineage_births += 1

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
            'agg_ltp': agg_ltp,
            'act_lineage': act_lineage,
            'agg_lineage': agg_lineage,
            'lineage_propagations': lineage_propagations,
            'lineage_births': lineage_births,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'lineage_propagations': lineage_propagations,
            'lineage_births': lineage_births,
            'mean_act_lineage': float(act_lineage.mean()),
            'mean_agg_lineage': float(agg_lineage.mean()),
        }

        return new_state, metrics
