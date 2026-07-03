"""Strategy 144-L1: STDP-Neuro-Symbiosis SOFT.

Level 1 ablation: Remove hard mask forcing, keep affinity floors and initial seeding.

Test: Does STDP+Symbiosis work WITHOUT permanent mask guarantees?

Extensions:
- Independent aggregation STDP (separate temporal credit from activations)
- Cross-domain STDP (learn temporal relationships between act-agg pairs)
- Universal cross-STDP for all co-active pairs
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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
    EXTREME_AGGS,
    AVERAGING_AGGS,
    CORE_EXTREME_AGGS,
    CROSS_PAIR_CATEGORIES,
    SIN_IDX,
)


class STDPSoftL1Strategy(PaletteEvolutionStrategy):
    """STDP-Neurogenesis-Symbiosis with SOFT protection (no mask forcing).

    Tests if STDP+Symbiosis provides value
    when sin/extreme_aggs can actually be removed.
    """

    name = "stdp_soft_L1"
    description = "L1: STDP+Symbiosis WITHOUT mask forcing (can lose sin)"

    def __init__(
        self,
        # === STDP PARAMETERS ===
        ltp_window: int = 5,
        ltd_window: int = 3,
        ltp_rate: float = 0.25,
        ltd_rate: float = 0.10,
        stdp_history_length: int = 10,
        temporal_decay: float = 0.7,
        # === NEUROGENESIS PARAMETERS ===
        agg_neurogenesis_rate: float = 0.10,
        agg_maturation_period: int = 8,
        young_agg_plasticity: float = 2.0,
        base_survival_threshold: float = 0.15,
        max_young_aggs: int = 2,
        # === STDP-SURVIVAL ===
        stdp_survival_multiplier: float = 0.7,
        young_ltp_plasticity: float = 2.5,
        ltp_survival_threshold: float = 0.4,
        sin_extreme_ltp_boost: float = 0.5,
        # === SYMBIOSIS PARAMETERS ===
        symbiosis_pairs: List[Tuple[int, int]] = None,
        symbiosis_protection: float = 0.8,
        orphan_vulnerability: float = 2.0,
        symbiosis_formation_rate: float = 0.15,
        # === INITIAL SEEDING (KEPT) ===
        sin_always_initial: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_learning_rate: float = 0.12,
        sin_extreme_coupling: float = 0.5,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 3,
        min_diversity_agg: int = 2,
        max_stable_agg: int = 4,
        # === INITIAL PALETTES ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
        # === INDEPENDENT AGGREGATION STDP ===
        enable_independent_agg_stdp: bool = True,
        agg_ltp_window: int = 6,             # Separate window for agg LTP
        agg_ltd_window: int = 4,             # Separate window for agg LTD
        agg_ltp_rate: float = 0.30,          # Higher LTP rate for aggs
        agg_ltd_rate: float = 0.12,          # Higher LTD rate for aggs
        # === CROSS-DOMAIN STDP ===
        enable_cross_stdp: bool = True,
        cross_ltp_rate: float = 0.20,        # LTP for co-active pairs
        cross_ltd_rate: float = 0.08,        # LTD for separated pairs
        cross_temporal_decay: float = 0.75,  # How quickly cross credit decays
        category_ltp_multipliers: Dict[str, float] = None,  # Per-category LTP boost
    ):
        """Initialize STDP Soft L1 strategy."""
        # STDP parameters
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.stdp_history_length = stdp_history_length
        self.temporal_decay = temporal_decay

        # Neurogenesis
        self.agg_neurogenesis_rate = agg_neurogenesis_rate
        self.agg_maturation_period = agg_maturation_period
        self.young_agg_plasticity = young_agg_plasticity
        self.base_survival_threshold = base_survival_threshold
        self.max_young_aggs = max_young_aggs

        # STDP-survival
        self.stdp_survival_multiplier = stdp_survival_multiplier
        self.young_ltp_plasticity = young_ltp_plasticity
        self.ltp_survival_threshold = ltp_survival_threshold
        self.sin_extreme_ltp_boost = sin_extreme_ltp_boost

        # Symbiosis
        self.symbiosis_pairs = symbiosis_pairs or [(4, 2), (4, 3)]
        self.symbiosis_protection = symbiosis_protection
        self.orphan_vulnerability = orphan_vulnerability
        self.symbiosis_formation_rate = symbiosis_formation_rate

        # Initial seeding (KEPT for L1)
        self.sin_always_initial = sin_always_initial
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_coupling = sin_extreme_coupling

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg
        self.max_stable_agg = max_stable_agg

        # Initial palettes
        default_act = list(DEFAULT_PALETTE_INDICES)
        if self.sin_always_initial and 4 not in default_act:
            default_act.append(4)

        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        if self.extreme_always_initial:
            for agg in CORE_EXTREME_AGGS:
                if agg not in default_agg:
                    default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

        # Independent aggregation STDP
        self.enable_independent_agg_stdp = enable_independent_agg_stdp
        self.agg_ltp_window = agg_ltp_window
        self.agg_ltd_window = agg_ltd_window
        self.agg_ltp_rate = agg_ltp_rate
        self.agg_ltd_rate = agg_ltd_rate

        # Cross-domain STDP
        self.enable_cross_stdp = enable_cross_stdp
        self.cross_ltp_rate = cross_ltp_rate
        self.cross_ltd_rate = cross_ltd_rate
        self.cross_temporal_decay = cross_temporal_decay
        self.category_ltp_multipliers = category_ltp_multipliers or {
            'known_synergistic': 1.5,
            'oscillatory_extreme': 1.3,
            'smooth_averaging': 1.0,
            'rectified_extreme': 1.2,
            'periodic_averaging': 1.1,
        }

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # Initial seeding (KEPT for L1)
        if self.sin_always_initial and 4 not in initial_act:
            initial_act = list(initial_act) + [4]

        initial_agg = list(initial_agg)
        if self.extreme_always_initial:
            for agg in CORE_EXTREME_AGGS:
                if agg not in initial_agg:
                    initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        act_affinities = act_affinities.at[4].set(0.8)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for agg in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[agg].set(0.75)

        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, agg].set(0.7)

        # STDP credit
        act_ltp_credit = jnp.zeros(NUM_ACTIVATIONS)
        agg_ltp_credit = jnp.zeros(NUM_AGGREGATIONS)

        # Neurogenesis
        stable_aggs: Set[int] = set(initial_agg)
        young_aggs: Dict[int, Dict] = {}
        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.4)

        # Symbiosis
        symbiosis_strength = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        for act_idx, agg_idx in self.symbiosis_pairs:
            symbiosis_strength = symbiosis_strength.at[act_idx, agg_idx].set(1.0)

        # Cross-domain LTP credit (18 activations x 6 aggregations)
        cross_ltp_credit = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        # Initialize known synergistic pairs with small credit
        for agg in CORE_EXTREME_AGGS:
            cross_ltp_credit = cross_ltp_credit.at[SIN_IDX, agg].set(0.3)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'cross_affinity': cross_affinity,
            'act_ltp_credit': act_ltp_credit,
            'agg_ltp_credit': agg_ltp_credit,
            'stdp_history': [],
            'stable_aggs': stable_aggs,
            'young_aggs': young_aggs,
            'agg_contribution': agg_contribution,
            'symbiosis_strength': symbiosis_strength,
            'orphan_count_act': jnp.zeros(NUM_ACTIVATIONS),
            'orphan_count_agg': jnp.zeros(NUM_AGGREGATIONS),
            'rng_key': jax.random.PRNGKey(seed + 1441000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            'ltp_survival_events': 0,
            'symbiosis_events': 0,
            # Independent aggregation STDP
            'agg_ltd_credit': jnp.zeros(NUM_AGGREGATIONS),  # Separate LTD for aggs
            'agg_stdp_history': [],  # Separate history for agg STDP
            # Cross-domain STDP
            'cross_ltp_credit': cross_ltp_credit,
            'cross_ltd_credit': jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS)),
            'cross_stdp_updates': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_temporal_weight(self, gens_from_improvement: int) -> float:
        return self.temporal_decay ** abs(gens_from_improvement)

    def _check_symbiosis_status(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        has_partner_act = jnp.zeros(NUM_ACTIVATIONS)
        has_partner_agg = jnp.zeros(NUM_AGGREGATIONS)

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                for j in range(NUM_AGGREGATIONS):
                    if agg_mask[j] > 0.5 and symbiosis_strength[i, j] > 0.3:
                        has_partner_act = has_partner_act.at[i].set(1.0)
                        break

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                for i in range(NUM_ACTIVATIONS):
                    if act_mask[i] > 0.5 and symbiosis_strength[i, j] > 0.3:
                        has_partner_agg = has_partner_agg.at[j].set(1.0)
                        break

        return has_partner_act, has_partner_agg

    def _update_independent_agg_stdp(
        self,
        agg_mask: jnp.ndarray,
        agg_ltp_credit: jnp.ndarray,
        agg_ltd_credit: jnp.ndarray,
        agg_stdp_history: List,
        generation: int,
        improved: bool,
        stagnated: bool,
        young_aggs: Dict,
        has_partner_agg: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List]:
        """Update independent STDP credit for aggregations.

        Aggregations get their own temporal credit independent of activations.
        This allows aggregations to develop their own temporal associations.
        """
        if not self.enable_independent_agg_stdp:
            return agg_ltp_credit, agg_ltd_credit, agg_stdp_history

        # Decay existing credits
        new_ltp = agg_ltp_credit * 0.92
        new_ltd = agg_ltd_credit * 0.95

        # Update history
        new_history = agg_stdp_history + [(generation, agg_mask.copy())]
        if len(new_history) > self.stdp_history_length:
            new_history = new_history[-self.stdp_history_length:]

        # LTP: Aggregations active BEFORE improvement get credit
        if improved and len(new_history) >= 2:
            for hist_gen, hist_agg_mask in new_history:
                gens_before = generation - hist_gen
                if 1 <= gens_before <= self.agg_ltp_window:
                    temporal_weight = self.cross_temporal_decay ** gens_before
                    ltp_delta = self.agg_ltp_rate * temporal_weight

                    for j in range(NUM_AGGREGATIONS):
                        if hist_agg_mask[j] > 0.5:
                            bonus = 1.0
                            if j in young_aggs:
                                bonus *= 2.0  # Young aggs learn faster
                            if has_partner_agg[j] > 0.5:
                                bonus *= 1.3
                            if j in CORE_EXTREME_AGGS:
                                bonus *= 1.2  # Boost extremes
                            new_ltp = new_ltp.at[j].set(
                                min(1.0, new_ltp[j] + ltp_delta * bonus)
                            )

        # LTD: Aggregations active during stagnation lose credit
        if stagnated:
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    # Only penalize if not protected
                    if has_partner_agg[j] < 0.5:
                        new_ltd = new_ltd.at[j].set(
                            min(1.0, new_ltd[j] + self.agg_ltd_rate)
                        )

        return new_ltp, new_ltd, new_history

    def _update_cross_stdp(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_ltp_credit: jnp.ndarray,
        cross_ltd_credit: jnp.ndarray,
        stdp_history: List,
        generation: int,
        improved: bool,
        stagnated: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Update cross-domain STDP credit for act-agg pairs.

        Learn temporal relationships between activation-aggregation pairs.
        Pairs that are co-active before improvement get LTP credit.
        """
        if not self.enable_cross_stdp:
            return cross_ltp_credit, cross_ltd_credit, 0

        # Decay existing credits
        new_ltp = cross_ltp_credit * 0.93
        new_ltd = cross_ltd_credit * 0.95
        updates = 0

        # LTP: Co-active pairs before improvement get credit
        if improved and len(stdp_history) >= 2:
            for hist_gen, hist_act_mask, hist_agg_mask, hist_fitness in stdp_history:
                gens_before = generation - hist_gen
                if 1 <= gens_before <= self.ltp_window:
                    temporal_weight = self.cross_temporal_decay ** gens_before
                    base_ltp = self.cross_ltp_rate * temporal_weight

                    for i in range(NUM_ACTIVATIONS):
                        for j in range(NUM_AGGREGATIONS):
                            if hist_act_mask[i] > 0.5 and hist_agg_mask[j] > 0.5:
                                # Get category multiplier
                                multiplier = 1.0
                                for category, pairs in CROSS_PAIR_CATEGORIES.items():
                                    if (i, j) in pairs:
                                        multiplier = self.category_ltp_multipliers.get(category, 1.0)
                                        break

                                delta = base_ltp * multiplier
                                new_ltp = new_ltp.at[i, j].set(
                                    min(1.0, new_ltp[i, j] + delta)
                                )
                                updates += 1

        # LTD: Co-active pairs during stagnation lose credit
        if stagnated:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        new_ltd = new_ltd.at[i, j].set(
                            min(1.0, new_ltd[i, j] + self.cross_ltd_rate)
                        )

        # Ensure minimum credit for known synergistic pairs
        for agg in CORE_EXTREME_AGGS:
            new_ltp = new_ltp.at[SIN_IDX, agg].set(
                max(0.2, float(new_ltp[SIN_IDX, agg]))
            )

        return new_ltp, new_ltd, updates

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with STDP + symbiosis mechanics (L1: no mask forcing)."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        stable_aggs = set(state['stable_aggs'])
        young_aggs = dict(state['young_aggs'])

        # Check symbiosis status
        has_partner_act, has_partner_agg = self._check_symbiosis_status(
            state['act_mask'], state['agg_mask'], state['symbiosis_strength']
        )

        # Update STDP history
        stdp_history = state['stdp_history'] + [
            (generation, state['act_mask'].copy(), state['agg_mask'].copy(), best_fitness)
        ]
        if len(stdp_history) > self.stdp_history_length:
            stdp_history = stdp_history[-self.stdp_history_length:]

        # Update LTP credit
        new_act_ltp = state['act_ltp_credit'] * 0.95
        new_agg_ltp = state['agg_ltp_credit'] * 0.95

        if improved and len(stdp_history) >= 2:
            for hist_gen, hist_act_mask, hist_agg_mask, hist_fitness in stdp_history:
                gens_before = generation - hist_gen
                if 1 <= gens_before <= self.ltp_window:
                    temporal_weight = self._compute_temporal_weight(gens_before)
                    ltp_delta = self.ltp_rate * temporal_weight

                    for i in range(NUM_ACTIVATIONS):
                        if hist_act_mask[i] > 0.5:
                            bonus = 1.0
                            if has_partner_act[i] > 0.5:
                                bonus *= 1.3
                            new_act_ltp = new_act_ltp.at[i].set(
                                min(1.0, new_act_ltp[i] + ltp_delta * bonus)
                            )

                    for j in range(NUM_AGGREGATIONS):
                        if hist_agg_mask[j] > 0.5:
                            bonus = 1.0
                            if j in young_aggs:
                                bonus *= self.young_ltp_plasticity
                            if has_partner_agg[j] > 0.5:
                                bonus *= 1.3
                            new_agg_ltp = new_agg_ltp.at[j].set(
                                min(1.0, new_agg_ltp[j] + ltp_delta * bonus)
                            )

        # === Independent aggregation STDP ===
        stagnated = fitness_delta < 0.0001
        agg_ltd_credit = state.get('agg_ltd_credit', jnp.zeros(NUM_AGGREGATIONS))
        agg_stdp_history = state.get('agg_stdp_history', [])

        new_agg_ltp, new_agg_ltd, agg_stdp_history = self._update_independent_agg_stdp(
            state['agg_mask'],
            new_agg_ltp,
            agg_ltd_credit,
            agg_stdp_history,
            generation,
            improved,
            stagnated,
            young_aggs,
            has_partner_agg,
        )

        # === Cross-domain STDP ===
        cross_ltp_credit = state.get('cross_ltp_credit', jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS)))
        cross_ltd_credit = state.get('cross_ltd_credit', jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS)))

        new_cross_ltp, new_cross_ltd, cross_stdp_updates = self._update_cross_stdp(
            state['act_mask'],
            state['agg_mask'],
            cross_ltp_credit,
            cross_ltd_credit,
            stdp_history,
            generation,
            improved,
            stagnated,
        )

        # Update agg contribution
        new_agg_contribution = state['agg_contribution'] * 0.9
        for j in range(NUM_AGGREGATIONS):
            if state['agg_mask'][j] > 0.5:
                current = float(new_agg_contribution[j])
                if improved:
                    boost = 0.25
                    if j in young_aggs:
                        boost *= self.young_agg_plasticity
                    if j in CORE_EXTREME_AGGS:
                        boost *= 1.3
                    if has_partner_agg[j] > 0.5:
                        boost *= 1.2
                    new_agg_contribution = new_agg_contribution.at[j].set(current + boost)
                else:
                    new_agg_contribution = new_agg_contribution.at[j].set(current + 0.01)
        new_agg_contribution = jnp.clip(new_agg_contribution, 0, 2.0)

        # Maturation
        new_stable = set(stable_aggs)
        new_young = {}
        ltp_survive_count = 0

        for agg_idx, info in young_aggs.items():
            age = generation - info['birth_gen']
            if age >= self.agg_maturation_period:
                base_contribution = float(new_agg_contribution[agg_idx])
                survival_threshold = self.base_survival_threshold

                ltp_credit = float(new_agg_ltp[agg_idx])
                accumulated_ltp = info.get('ltp_accumulated', 0.0)
                total_ltp = (ltp_credit + accumulated_ltp) / 2

                if total_ltp > self.ltp_survival_threshold:
                    survival_threshold *= self.stdp_survival_multiplier
                    ltp_survive_count += 1

                if has_partner_agg[agg_idx] > 0.5:
                    survival_threshold *= 0.8

                if base_contribution >= survival_threshold:
                    if len(new_stable) < self.max_stable_agg:
                        new_stable.add(agg_idx)
            else:
                new_young[agg_idx] = info

        stable_aggs = new_stable
        young_aggs = new_young

        # Maybe birth new agg
        if (len(young_aggs) < self.max_young_aggs and
            jax.random.uniform(k1) < self.agg_neurogenesis_rate):
            available = [j for j in range(NUM_AGGREGATIONS)
                        if j not in stable_aggs and j not in young_aggs]
            if available:
                extreme_available = [j for j in available if j in CORE_EXTREME_AGGS]
                if extreme_available and jax.random.uniform(k2) < 0.7:
                    idx = int(jax.random.randint(k2, (), 0, len(extreme_available)))
                    new_agg = extreme_available[idx]
                else:
                    idx = int(jax.random.randint(k2, (), 0, len(available)))
                    new_agg = available[idx]
                young_aggs[new_agg] = {'birth_gen': generation, 'ltp_accumulated': 0.0}

        # Create agg mask
        new_agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for j in stable_aggs:
            if 0 <= j < NUM_AGGREGATIONS:
                new_agg_mask = new_agg_mask.at[j].set(1.0)
        for j in young_aggs.keys():
            if 0 <= j < NUM_AGGREGATIONS:
                new_agg_mask = new_agg_mask.at[j].set(1.0)

        # L1 CRITICAL CHANGE: NO HARD MASK FORCING for extreme aggs
        # Original code REMOVED:
        # for agg in CORE_EXTREME_AGGS:
        #     new_agg_mask = new_agg_mask.at[agg].set(1.0)
        #     if agg not in stable_aggs:
        #         stable_aggs.add(agg)

        # Update affinities
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay

        if fitness_delta > 0:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5:
                    lr = self.agg_affinity_lr * (1 + float(new_agg_ltp[j]) * 0.5)
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus += 0.6
                    if has_partner_agg[j] > 0.5:
                        bonus *= 1.2
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + lr * fitness_delta * bonus)
                    )

            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    lr = self.act_affinity_lr * (1 + float(new_act_ltp[i]) * 0.5)
                    bonus = 1.0
                    if i == 4:
                        bonus = 1.5
                    if has_partner_act[i] > 0.5:
                        bonus *= 1.2
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + lr * fitness_delta * bonus)
                    )

        # AFFINITY FLOORS (KEPT for L1)
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Select activation palette
        score = new_act_aff + new_act_ltp * 0.3 + has_partner_act * 0.15
        score = score.at[4].set(score[4] + 0.5)

        target_size = min(max(self.min_diversity_act, self.min_active_act), self.max_active_act)
        top_indices = jnp.argsort(score)[-target_size:]

        new_act_mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            new_act_mask = new_act_mask.at[int(idx)].set(1.0)

        # L1 CRITICAL CHANGE: NO HARD MASK FORCING for sin
        # Original code REMOVED:
        # new_act_mask = new_act_mask.at[4].set(1.0)

        # Ensure minimums
        if int(jnp.sum(new_act_mask)) < self.min_active_act:
            new_act_mask = new_act_mask.at[0].set(1.0)
        if int(jnp.sum(new_agg_mask)) < self.min_active_agg:
            new_agg_mask = new_agg_mask.at[0].set(1.0)

        new_state = {
            **state,
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_ltp_credit': new_act_ltp,
            'agg_ltp_credit': new_agg_ltp,
            'stdp_history': stdp_history,
            'stable_aggs': stable_aggs,
            'young_aggs': young_aggs,
            'agg_contribution': new_agg_contribution,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'fitness_history': state['fitness_history'] + [best_fitness],
            'ltp_survival_events': state['ltp_survival_events'] + ltp_survive_count,
            # Independent aggregation STDP
            'agg_ltd_credit': new_agg_ltd,
            'agg_stdp_history': agg_stdp_history,
            # Cross-domain STDP
            'cross_ltp_credit': new_cross_ltp,
            'cross_ltd_credit': new_cross_ltd,
            'cross_stdp_updates': state.get('cross_stdp_updates', 0) + cross_stdp_updates,
        }

        # Count high cross-LTP pairs for metrics
        high_cross_ltp_pairs = 0
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if new_cross_ltp[i, j] > 0.5:
                    high_cross_ltp_pairs += 1

        # Count non-sin-extreme high cross-LTP pairs
        non_sin_extreme_high_ltp = 0
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if new_cross_ltp[i, j] > 0.5:
                    if i != SIN_IDX or j not in CORE_EXTREME_AGGS:
                        non_sin_extreme_high_ltp += 1

        metrics = {
            'ltp_survival_events': ltp_survive_count,
            'has_partners_act': int(has_partner_act.sum()),
            'has_partners_agg': int(has_partner_agg.sum()),
            # metrics
            'cross_stdp_updates': cross_stdp_updates,
            'high_cross_ltp_pairs': high_cross_ltp_pairs,
            'non_sin_extreme_high_ltp': non_sin_extreme_high_ltp,
            'max_cross_ltp': float(jnp.max(new_cross_ltp)),
        }

        return new_state, metrics
