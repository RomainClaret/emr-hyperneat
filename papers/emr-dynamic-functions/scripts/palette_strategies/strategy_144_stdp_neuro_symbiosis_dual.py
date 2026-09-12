"""Strategy 144: STDP-Neuro-Symbiosis Dual.

Combines the best working mechanism (STDP-Neurogenesis-Survival #117) with
Sin-Extreme-Symbiosis (#137) for enhanced sin+extreme_agg retention.

Key Innovation:
- STDP temporal credit assignment for function survival from #117
- Neurogenesis birth/maturation/survival from #117
- Symbiotic protection between sin and extreme_aggs from #137
- Orphan vulnerability - functions without symbiotic partners are 2x more likely pruned

Biological Basis:
- STDP: Neurons that fire before reward signals get strengthened
- Neurogenesis: Newly-born neurons that are activated during learning survive
- Symbiosis: Clownfish-anemone style obligate mutualism where partners protect each other
- Combined: Functions that predict success AND have symbiotic partners have highest survival

Expected: Superior sin+extreme_agg retention by combining temporal credit with mutual protection.
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class STDPNeuroSymbiosisDualStrategy(PaletteEvolutionStrategy):
    """STDP-Neurogenesis with Sin-Extreme Symbiosis.

    Combines the proven STDP-Neurogenesis survival mechanism
    (#117, 100% sin retention) with symbiotic protection (#137) where sin and
    extreme_aggs protect each other.

    Critical innovation: Double protection - temporal credit AND symbiotic bond
    both contribute to survival probability.
    """

    name = "stdp_neuro_symbiosis_dual"
    description = "Dual: STDP-Neurogenesis + Sin-Extreme symbiotic protection"

    def __init__(
        self,
        # === STDP PARAMETERS (from #117) ===
        ltp_window: int = 5,
        ltd_window: int = 3,
        ltp_rate: float = 0.25,
        ltd_rate: float = 0.10,
        stdp_history_length: int = 10,
        temporal_decay: float = 0.7,
        # === NEUROGENESIS PARAMETERS (from #117) ===
        agg_neurogenesis_rate: float = 0.10,
        agg_maturation_period: int = 8,
        young_agg_plasticity: float = 2.0,
        base_survival_threshold: float = 0.15,
        max_young_aggs: int = 2,
        # === STDP-SURVIVAL INTEGRATION (from #117) ===
        stdp_survival_multiplier: float = 0.7,
        young_ltp_plasticity: float = 2.5,
        ltp_survival_threshold: float = 0.4,
        sin_extreme_ltp_boost: float = 0.5,
        # === SYMBIOSIS PARAMETERS (from #137) ===
        symbiosis_pairs: List[Tuple[int, int]] = None,  # (sin, max), (sin, min)
        symbiosis_protection: float = 0.8,
        orphan_vulnerability: float = 2.0,
        symbiosis_formation_rate: float = 0.15,
        # === GUARANTEED INITIAL STATE ===
        sin_always_initial: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_learning_rate: float = 0.12,
        sin_extreme_coupling: float = 0.5,
        # === TAGGING PARAMETERS ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.40,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.85,
        extreme_tag_boost: float = 1.5,
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
    ):
        """Initialize STDP-Neuro-Symbiosis strategy."""
        # STDP parameters
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.stdp_history_length = stdp_history_length
        self.temporal_decay = temporal_decay

        # Neurogenesis parameters
        self.agg_neurogenesis_rate = agg_neurogenesis_rate
        self.agg_maturation_period = agg_maturation_period
        self.young_agg_plasticity = young_agg_plasticity
        self.base_survival_threshold = base_survival_threshold
        self.max_young_aggs = max_young_aggs

        # STDP-survival integration
        self.stdp_survival_multiplier = stdp_survival_multiplier
        self.young_ltp_plasticity = young_ltp_plasticity
        self.ltp_survival_threshold = ltp_survival_threshold
        self.sin_extreme_ltp_boost = sin_extreme_ltp_boost

        # Symbiosis parameters (from #137)
        self.symbiosis_pairs = symbiosis_pairs or [(4, 2), (4, 3)]  # (sin, max), (sin, min)
        self.symbiosis_protection = symbiosis_protection
        self.orphan_vulnerability = orphan_vulnerability
        self.symbiosis_formation_rate = symbiosis_formation_rate

        # Guaranteed state
        self.sin_always_initial = sin_always_initial
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_coupling = sin_extreme_coupling

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg
        self.max_stable_agg = max_stable_agg

        # Initial palettes - ALWAYS include sin and extreme aggs
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

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with STDP, neurogenesis, and symbiosis tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # CRITICAL: Ensure sin is ALWAYS in initial palette
        if self.sin_always_initial and 4 not in initial_act:
            initial_act = list(initial_act) + [4]

        # Ensure extreme aggs in initial palette
        initial_agg = list(initial_agg)
        if self.extreme_always_initial:
            for agg in CORE_EXTREME_AGGS:
                if agg not in initial_agg:
                    initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities - sin and extreme aggs get high initial values
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        act_affinities = act_affinities.at[4].set(0.8)  # Sin very high

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for agg in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[agg].set(0.75)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, agg].set(0.7)  # Sin-extreme high

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # STDP credit tracking
        act_ltp_credit = jnp.zeros(NUM_ACTIVATIONS)
        agg_ltp_credit = jnp.zeros(NUM_AGGREGATIONS)

        # Neurogenesis state
        stable_aggs: Set[int] = set(initial_agg)
        young_aggs: Dict[int, Dict] = {}
        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.4)

        # Symbiosis state (from #137)
        symbiosis_strength = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        for act_idx, agg_idx in self.symbiosis_pairs:
            symbiosis_strength = symbiosis_strength.at[act_idx, agg_idx].set(1.0)

        orphan_count_act = jnp.zeros(NUM_ACTIVATIONS)
        orphan_count_agg = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # STDP state
            'act_ltp_credit': act_ltp_credit,
            'agg_ltp_credit': agg_ltp_credit,
            'stdp_history': [],
            # Neurogenesis state
            'stable_aggs': stable_aggs,
            'young_aggs': young_aggs,
            'agg_contribution': agg_contribution,
            'agg_births': [],
            'agg_survivals': [],
            'agg_prunings': [],
            # Symbiosis state (from #137)
            'symbiosis_strength': symbiosis_strength,
            'orphan_count_act': orphan_count_act,
            'orphan_count_agg': orphan_count_agg,
            # Stats
            'capture_events': 0,
            'total_agg_births': 0,
            'total_agg_survivals': 0,
            'total_agg_prunings': 0,
            'ltp_survival_events': 0,
            'symbiosis_events': 0,
            'orphan_pruning_events': 0,
            'diversity_rescues': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1440000),
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

    def _compute_temporal_weight(self, gens_from_improvement: int) -> float:
        """Compute STDP temporal weight - closer = stronger."""
        return self.temporal_decay ** abs(gens_from_improvement)

    def _check_symbiosis_status(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Check which functions have active symbiotic partners."""
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

    def _update_symbiosis(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
        fitness_delta: float,
        generation: int,
    ) -> Tuple[jnp.ndarray, int]:
        """Update symbiotic relationships."""
        new_symbiosis = symbiosis_strength.copy()
        events = 0

        # Strengthen existing bonds on improvement
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        if new_symbiosis[i, j] > 0.1:
                            new_symbiosis = new_symbiosis.at[i, j].set(
                                min(1.0, new_symbiosis[i, j] + 0.1)
                            )
                        elif new_symbiosis[i, j] < 0.1:
                            if jax.random.uniform(jax.random.PRNGKey(generation * 100 + i * 10 + j)) < self.symbiosis_formation_rate:
                                new_symbiosis = new_symbiosis.at[i, j].set(0.3)
                                events += 1

        # Decay bonds for inactive pairs
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if act_mask[i] < 0.5 or agg_mask[j] < 0.5:
                    new_symbiosis = new_symbiosis.at[i, j].set(
                        new_symbiosis[i, j] * 0.9
                    )

        # CRITICAL: Sin-extreme bonds never fully break
        for act_idx, agg_idx in self.symbiosis_pairs:
            new_symbiosis = new_symbiosis.at[act_idx, agg_idx].set(
                max(0.5, float(new_symbiosis[act_idx, agg_idx]))
            )

        return new_symbiosis, events

    def _update_ltp_credit(
        self,
        act_ltp_credit: jnp.ndarray,
        agg_ltp_credit: jnp.ndarray,
        stdp_history: List[Tuple],
        current_gen: int,
        improved: bool,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        young_aggs: Dict[int, Dict],
        has_partner_act: jnp.ndarray,
        has_partner_agg: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[int, Dict]]:
        """Update STDP LTP credit with symbiosis bonus."""
        new_act_ltp = act_ltp_credit * 0.95
        new_agg_ltp = agg_ltp_credit * 0.95
        new_young = {k: dict(v) for k, v in young_aggs.items()}

        if improved and len(stdp_history) >= 2:
            for hist_gen, hist_act_mask, hist_agg_mask, hist_fitness in stdp_history:
                gens_before = current_gen - hist_gen

                if 1 <= gens_before <= self.ltp_window:
                    temporal_weight = self._compute_temporal_weight(gens_before)
                    ltp_delta = self.ltp_rate * temporal_weight

                    # LTP for activations (symbiotic bonus)
                    for i in range(NUM_ACTIVATIONS):
                        if hist_act_mask[i] > 0.5:
                            bonus = 1.0
                            if has_partner_act[i] > 0.5:
                                bonus *= 1.3  # Symbiotic partner bonus
                            new_act_ltp = new_act_ltp.at[i].set(
                                min(1.0, new_act_ltp[i] + ltp_delta * bonus)
                            )

                    # LTP for aggregations (young + symbiotic bonus)
                    for j in range(NUM_AGGREGATIONS):
                        if hist_agg_mask[j] > 0.5:
                            bonus = 1.0
                            if j in new_young:
                                bonus *= self.young_ltp_plasticity
                            if has_partner_agg[j] > 0.5:
                                bonus *= 1.3  # Symbiotic partner bonus
                            new_agg_ltp = new_agg_ltp.at[j].set(
                                min(1.0, new_agg_ltp[j] + ltp_delta * bonus)
                            )
                            if j in new_young:
                                prev = new_young[j].get('ltp_accumulated', 0.0)
                                new_young[j]['ltp_accumulated'] = prev + ltp_delta * bonus

                    # Sin-extreme co-activation bonus
                    sin_active = hist_act_mask[4] > 0.5
                    for j in CORE_EXTREME_AGGS:
                        if sin_active and hist_agg_mask[j] > 0.5:
                            bonus = self.sin_extreme_ltp_boost * temporal_weight
                            new_act_ltp = new_act_ltp.at[4].set(
                                min(1.0, new_act_ltp[4] + bonus)
                            )
                            new_agg_ltp = new_agg_ltp.at[j].set(
                                min(1.0, new_agg_ltp[j] + bonus)
                            )

        # LTD for stagnation
        if not improved:
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    new_agg_ltp = new_agg_ltp.at[j].set(
                        max(0, new_agg_ltp[j] - self.ltd_rate * 0.3)
                    )

        return new_act_ltp, new_agg_ltp, new_young

    def _maybe_birth_agg(
        self,
        stable_aggs: Set[int],
        young_aggs: Dict[int, Dict],
        key: jax.random.PRNGKey,
        generation: int,
    ) -> Tuple[Set[int], Dict[int, Dict], Optional[int]]:
        """Possibly birth a new aggregation with preference for extremes."""
        key1, key2 = jax.random.split(key)
        new_young = dict(young_aggs)
        born = None

        if (len(new_young) < self.max_young_aggs and
            jax.random.uniform(key1) < self.agg_neurogenesis_rate):

            available = [j for j in range(NUM_AGGREGATIONS)
                        if j not in stable_aggs and j not in new_young]

            if available:
                extreme_available = [j for j in available if j in CORE_EXTREME_AGGS]

                if extreme_available and jax.random.uniform(key2) < 0.7:
                    idx = int(jax.random.randint(key2, (), 0, len(extreme_available)))
                    new_agg = extreme_available[idx]
                else:
                    idx = int(jax.random.randint(key2, (), 0, len(available)))
                    new_agg = available[idx]

                new_young[new_agg] = {'birth_gen': generation, 'ltp_accumulated': 0.0}
                born = new_agg

        return stable_aggs, new_young, born

    def _mature_aggs_with_stdp_symbiosis(
        self,
        stable_aggs: Set[int],
        young_aggs: Dict[int, Dict],
        agg_contribution: jnp.ndarray,
        agg_ltp_credit: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        has_partner_agg: jnp.ndarray,
        generation: int,
    ) -> Tuple[Set[int], Dict[int, Dict], List[int], List[int], int]:
        """Process aggregation maturation with STDP AND symbiosis-guided survival."""
        new_stable = set(stable_aggs)
        new_young = {}
        survived = []
        pruned = []
        ltp_survival_count = 0

        sin_active = act_mask[4] > 0.5

        for agg_idx, info in young_aggs.items():
            age = generation - info['birth_gen']

            if age >= self.agg_maturation_period:
                base_contribution = float(agg_contribution[agg_idx])
                survival_threshold = self.base_survival_threshold

                # STDP-SURVIVAL: LTP credit lowers threshold
                ltp_credit = float(agg_ltp_credit[agg_idx])
                accumulated_ltp = info.get('ltp_accumulated', 0.0)
                total_ltp = (ltp_credit + accumulated_ltp) / 2

                if total_ltp > self.ltp_survival_threshold:
                    survival_threshold *= self.stdp_survival_multiplier
                    ltp_survival_count += 1

                # SYMBIOSIS: Having partner lowers threshold
                if has_partner_agg[agg_idx] > 0.5:
                    survival_threshold *= 0.8  # 20% easier to survive with partner

                # Sin coupling bonus for extreme aggs
                if sin_active and agg_idx in CORE_EXTREME_AGGS:
                    sin_aff = float(cross_affinity[4, agg_idx])
                    if sin_aff > 0.5:
                        survival_threshold *= 0.8
                        base_contribution += sin_aff * 0.1

                if base_contribution >= survival_threshold:
                    if len(new_stable) < self.max_stable_agg:
                        new_stable.add(agg_idx)
                        survived.append(agg_idx)
                    else:
                        pruned.append(agg_idx)
                else:
                    pruned.append(agg_idx)
            else:
                new_young[agg_idx] = info

        return new_stable, new_young, survived, pruned, ltp_survival_count

    def _update_agg_contribution(
        self,
        agg_contribution: jnp.ndarray,
        agg_mask: jnp.ndarray,
        young_aggs: Dict[int, Dict],
        has_partner_agg: jnp.ndarray,
        improved: bool,
    ) -> jnp.ndarray:
        """Update aggregation contribution with symbiosis bonus."""
        new_contribution = agg_contribution * 0.9  # Base decay

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                current = float(new_contribution[j])
                if improved:
                    boost = 0.25
                    if j in young_aggs:
                        boost *= self.young_agg_plasticity
                    if j in CORE_EXTREME_AGGS:
                        boost *= 1.3
                    if has_partner_agg[j] > 0.5:
                        boost *= 1.2  # Symbiotic bonus
                    new_contribution = new_contribution.at[j].set(current + boost)
                else:
                    new_contribution = new_contribution.at[j].set(current + 0.01)

        return jnp.clip(new_contribution, 0, 2.0)

    def _create_agg_mask(
        self,
        stable_aggs: Set[int],
        young_aggs: Dict[int, Dict],
    ) -> jnp.ndarray:
        """Create agg mask from stable and young aggs."""
        mask = jnp.zeros(NUM_AGGREGATIONS)

        for j in stable_aggs:
            if 0 <= j < NUM_AGGREGATIONS:
                mask = mask.at[j].set(1.0)

        for j in young_aggs.keys():
            if 0 <= j < NUM_AGGREGATIONS:
                mask = mask.at[j].set(1.0)

        return mask

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_ltp_credit: jnp.ndarray,
        agg_ltp_credit: jnp.ndarray,
        has_partner_act: jnp.ndarray,
        has_partner_agg: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with LTP + symbiosis boost."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                tag_strength *= (1 + float(agg_ltp_credit[j]) * 0.5)
                if has_partner_agg[j] > 0.5:
                    tag_strength *= 1.2  # Symbiotic bonus
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.35)
                )

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:  # Sin
                    tag_strength *= self.extreme_tag_boost
                tag_strength *= (1 + float(act_ltp_credit[i]) * 0.5)
                if has_partner_act[i] > 0.5:
                    tag_strength *= 1.2  # Symbiotic bonus
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Attempt synaptic tag and capture."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

        return new_act_captured, new_agg_captured, capture_count

    def _update_affinities(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
        act_ltp_credit: jnp.ndarray,
        agg_ltp_credit: jnp.ndarray,
        has_partner_act: jnp.ndarray,
        has_partner_agg: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update affinities with LTP + symbiosis modulation."""
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if fitness_delta > 0:
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    lr = self.agg_affinity_lr * (1 + float(agg_ltp_credit[j]) * 0.5)
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus += 0.6
                    if has_partner_agg[j] > 0.5:
                        bonus *= 1.2  # Symbiotic bonus
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + lr * fitness_delta * bonus)
                    )

            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    lr = self.act_affinity_lr * (1 + float(act_ltp_credit[i]) * 0.5)
                    bonus = 1.0
                    if i == 4:  # Sin
                        bonus = 1.5
                    if has_partner_act[i] > 0.5:
                        bonus *= 1.2  # Symbiotic bonus
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + lr * fitness_delta * bonus)
                    )

        # Sin and extreme affinity floors
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Cross-domain update
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        delta = self.cross_learning_rate * fitness_delta
                        if i == 4 and j in CORE_EXTREME_AGGS:
                            delta *= (1 + self.sin_extreme_coupling)
                        new_cross = new_cross.at[i, j].set(
                            min(1.0, new_cross[i, j] + delta)
                        )

        return new_act_aff, new_agg_aff, new_cross

    def _select_act_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        ltp_credit: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
        has_partner: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int]:
        """Select activation palette with symbiosis consideration."""
        score = affinities + captured * 0.3 + tags * 0.2 + ltp_credit * 0.3

        # Symbiotic partner bonus
        score = score + has_partner * 0.15

        for i in range(NUM_ACTIVATIONS):
            cross_influence = 0.0
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    cross_influence = max(cross_influence, float(cross_affinity[i, j]))
            score = score.at[i].set(score[i] + cross_influence * 0.25)

        # Sin preference
        score = score.at[4].set(score[4] + 0.5)

        target_size = min(max(self.min_diversity_act, self.min_active_act), self.max_active_act)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # CRITICAL: Ensure sin is ALWAYS in palette
        mask = mask.at[4].set(1.0)

        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < self.min_diversity_act:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(NUM_ACTIVATIONS) if mask[i] < 0.5]
            needed = self.min_diversity_act - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive),
                                          shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute extreme/averaging ratio."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with STDP + neurogenesis + symbiosis mechanics."""
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

        # === CHECK SYMBIOSIS STATUS ===
        has_partner_act, has_partner_agg = self._check_symbiosis_status(
            state['act_mask'], state['agg_mask'], state['symbiosis_strength']
        )

        # === UPDATE SYMBIOSIS ===
        new_symbiosis, symbiosis_events = self._update_symbiosis(
            state['act_mask'], state['agg_mask'], state['symbiosis_strength'],
            fitness_delta, generation
        )

        # === UPDATE STDP HISTORY ===
        stdp_history = state['stdp_history'] + [
            (generation, state['act_mask'].copy(), state['agg_mask'].copy(), best_fitness)
        ]
        if len(stdp_history) > self.stdp_history_length:
            stdp_history = stdp_history[-self.stdp_history_length:]

        # === UPDATE LTP CREDIT (with symbiosis bonus) ===
        new_act_ltp, new_agg_ltp, young_aggs = self._update_ltp_credit(
            state['act_ltp_credit'], state['agg_ltp_credit'],
            stdp_history, generation, improved,
            state['act_mask'], state['agg_mask'], young_aggs,
            has_partner_act, has_partner_agg
        )

        # === AGG CONTRIBUTION UPDATE (with symbiosis bonus) ===
        new_agg_contribution = self._update_agg_contribution(
            state['agg_contribution'], state['agg_mask'], young_aggs,
            has_partner_agg, improved
        )

        # === MATURATION WITH STDP + SYMBIOSIS ===
        stable_aggs, young_aggs, survived, pruned, ltp_survive_count = self._mature_aggs_with_stdp_symbiosis(
            stable_aggs, young_aggs,
            new_agg_contribution, new_agg_ltp,
            state['cross_affinity'], state['act_mask'],
            has_partner_agg, generation
        )

        # === MAYBE BIRTH NEW AGG ===
        stable_aggs, young_aggs, born = self._maybe_birth_agg(
            stable_aggs, young_aggs, k1, generation
        )

        # === CREATE AGG MASK ===
        new_agg_mask = self._create_agg_mask(stable_aggs, young_aggs)

        # CRITICAL: Ensure extreme aggs are ALWAYS in mask
        for agg in CORE_EXTREME_AGGS:
            new_agg_mask = new_agg_mask.at[agg].set(1.0)
            if agg not in stable_aggs:
                stable_aggs.add(agg)

        # === UPDATE SYMBIOSIS STATUS FOR NEW MASK ===
        has_partner_act, has_partner_agg = self._check_symbiosis_status(
            state['act_mask'], new_agg_mask, new_symbiosis
        )

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], new_agg_mask,
            state['act_tags'], state['agg_tags'],
            new_act_ltp, new_agg_ltp,
            has_partner_act, has_partner_agg
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE ===
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            new_tag_history, generation, improved
        )

        # === AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross_affinity = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], new_agg_mask,
            state['cross_affinity'], fitness_delta,
            new_act_ltp, new_agg_ltp,
            has_partner_act, has_partner_agg
        )

        # === ACTIVATION PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_act_palette(
            new_act_aff, new_act_captured, new_act_tags, new_act_ltp,
            new_cross_affinity, new_agg_mask, has_partner_act, k2
        )

        extreme_ratio = self._compute_extreme_ratio(new_agg_mask)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Update history
        agg_births = list(state['agg_births'])
        if born is not None:
            agg_births.append((generation, born))
            if len(agg_births) > 50:
                agg_births = agg_births[-50:]

        agg_survivals = list(state['agg_survivals']) + survived
        if len(agg_survivals) > 50:
            agg_survivals = agg_survivals[-50:]

        agg_prunings = list(state['agg_prunings']) + pruned
        if len(agg_prunings) > 50:
            agg_prunings = agg_prunings[-50:]

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        # Update orphan counts
        new_orphan_act = state['orphan_count_act'].copy()
        new_orphan_agg = state['orphan_count_agg'].copy()
        for i in range(NUM_ACTIVATIONS):
            if new_act_mask[i] > 0.5 and has_partner_act[i] < 0.5:
                new_orphan_act = new_orphan_act.at[i].set(new_orphan_act[i] + 1)
            else:
                new_orphan_act = new_orphan_act.at[i].set(0)
        for j in range(NUM_AGGREGATIONS):
            if new_agg_mask[j] > 0.5 and has_partner_agg[j] < 0.5:
                new_orphan_agg = new_orphan_agg.at[j].set(new_orphan_agg[j] + 1)
            else:
                new_orphan_agg = new_orphan_agg.at[j].set(0)

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            # STDP state
            'act_ltp_credit': new_act_ltp,
            'agg_ltp_credit': new_agg_ltp,
            'stdp_history': stdp_history,
            # Neurogenesis state
            'stable_aggs': stable_aggs,
            'young_aggs': young_aggs,
            'agg_contribution': new_agg_contribution,
            'agg_births': agg_births,
            'agg_survivals': agg_survivals,
            'agg_prunings': agg_prunings,
            # Symbiosis state
            'symbiosis_strength': new_symbiosis,
            'orphan_count_act': new_orphan_act,
            'orphan_count_agg': new_orphan_agg,
            # Stats
            'capture_events': state['capture_events'] + capture_count,
            'total_agg_births': state['total_agg_births'] + (1 if born else 0),
            'total_agg_survivals': state['total_agg_survivals'] + len(survived),
            'total_agg_prunings': state['total_agg_prunings'] + len(pruned),
            'ltp_survival_events': state['ltp_survival_events'] + ltp_survive_count,
            'symbiosis_events': state['symbiosis_events'] + symbiosis_events,
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue,
            # General
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        survival_rate = (new_state['total_agg_survivals'] /
                        max(new_state['total_agg_births'], 1)) * 100

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neurogenesis metrics
            'n_stable_aggs': len(stable_aggs),
            'n_young_aggs': len(young_aggs),
            'born_this_gen': born,
            'survived_this_gen': survived,
            'pruned_this_gen': pruned,
            'total_agg_births': new_state['total_agg_births'],
            'total_agg_survivals': new_state['total_agg_survivals'],
            'agg_survival_rate': survival_rate,
            # STDP metrics
            'ltp_survival_events': new_state['ltp_survival_events'],
            'sin_ltp_credit': float(new_act_ltp[4]),
            'max_ltp_credit': float(new_agg_ltp[2]),
            'min_ltp_credit': float(new_agg_ltp[3]),
            # Symbiosis metrics
            'symbiosis_events': new_state['symbiosis_events'],
            'has_partners_act': int(has_partner_act.sum()),
            'has_partners_agg': int(has_partner_agg.sum()),
            # Affinity metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'sin_affinity': float(new_act_aff[4]),
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Homeostatic
            'extreme_ratio': extreme_ratio,
            # Cross-domain
            'sin_max_affinity': float(new_cross_affinity[4, 2]),
            'sin_min_affinity': float(new_cross_affinity[4, 3]),
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        survival_rate = (state['total_agg_survivals'] /
                        max(state['total_agg_births'], 1)) * 100

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'n_stable_aggs': len(state['stable_aggs']),
            'n_young_aggs': len(state['young_aggs']),
            'total_agg_births': state['total_agg_births'],
            'total_agg_survivals': state['total_agg_survivals'],
            'agg_survival_rate': survival_rate,
            'ltp_survival_events': state['ltp_survival_events'],
            'symbiosis_events': state['symbiosis_events'],
            'sin_ltp_credit': float(state['act_ltp_credit'][4]),
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinities'])),
            'capture_events': state['capture_events'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
