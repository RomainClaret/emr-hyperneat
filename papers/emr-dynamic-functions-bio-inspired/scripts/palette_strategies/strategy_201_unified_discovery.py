"""Strategy 201: Unified Activation-Aggregation Discovery.

Combines all bio-inspired mechanisms for symmetric act-agg discovery:
- Dynamic symbiosis pair discovery (from strategy 137)
- Independent aggregation STDP (from strategy 144)
- Cross-domain STDP (from strategy 144)
- Dynamic aggregation morphogen sources (from strategy 133)
- Cross-morphogen influence (from strategy 133)

Supports multiple discovery modes:
- balanced: Equal treatment of activations and aggregations
- activation_first: Prioritize activation discovery (existing approach)
- aggregation_first: Prioritize aggregation discovery (novel approach)
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
    CORE_EXTREME_AGGS,
    CROSS_PAIR_CATEGORIES,
    SIN_IDX,
    ACT_CATEGORIES,
    AGG_CATEGORIES,
)


class UnifiedDiscoveryStrategy(PaletteEvolutionStrategy):
    """Unified symmetric activation-aggregation discovery.

    Combines all bio-inspired mechanisms for both dimensions symmetrically.
    """

    name = "unified_discovery"
    description = "Symmetric act-agg discovery with all mechanisms"

    def __init__(
        self,
        # === DISCOVERY MODE ===
        discovery_mode: str = 'balanced',  # 'balanced', 'activation_first', 'aggregation_first'
        # === MECHANISM TOGGLES ===
        enable_dynamic_symbiosis: bool = True,
        enable_independent_agg_stdp: bool = True,
        enable_cross_stdp: bool = True,
        enable_agg_morphogen_discovery: bool = True,
        enable_cross_morphogen: bool = True,
        # === SYMBIOSIS PARAMETERS ===
        symbiosis_pairs: List[Tuple[int, int]] = None,
        symbiosis_protection: float = 0.8,
        dynamic_symbiosis_cooccurrence_threshold: int = 3,
        dynamic_symbiosis_affinity_threshold: float = 0.6,
        max_symbiosis_pairs: int = 12,
        # === STDP PARAMETERS ===
        ltp_window: int = 5,
        ltd_window: int = 3,
        ltp_rate: float = 0.25,
        ltd_rate: float = 0.10,
        temporal_decay: float = 0.7,
        agg_ltp_window: int = 6,
        agg_ltp_rate: float = 0.30,
        cross_ltp_rate: float = 0.20,
        # === MORPHOGEN PARAMETERS ===
        sin_source_strength: float = 2.5,
        extreme_agg_source_strength: float = 2.0,
        morphogen_decay_rate: float = 0.05,
        dynamic_source_strength: float = 1.5,
        max_agg_sources: int = 5,  # Increased from 4
        agg_source_threshold: float = 0.45,  # Lowered from 0.65
        agg_source_contribution_threshold: int = 2,  # Lowered from 3
        cross_morphogen_influence: float = 0.15,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_affinity_lr: float = 0.12,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 2,  # Increased from 1 to ensure agg exploration
        max_active_act: int = 8,
        max_active_agg: int = 5,  # Increased from 4
        # === INITIAL PALETTES ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
        # === CATEGORY MULTIPLIERS ===
        category_lr_multipliers: Dict[str, float] = None,
    ):
        """Initialize Unified Discovery strategy."""
        self.discovery_mode = discovery_mode

        # Mechanism toggles
        self.enable_dynamic_symbiosis = enable_dynamic_symbiosis
        self.enable_independent_agg_stdp = enable_independent_agg_stdp
        self.enable_cross_stdp = enable_cross_stdp
        self.enable_agg_morphogen_discovery = enable_agg_morphogen_discovery
        self.enable_cross_morphogen = enable_cross_morphogen

        # Symbiosis
        self.symbiosis_pairs = symbiosis_pairs or [(SIN_IDX, 2), (SIN_IDX, 3)]
        self.symbiosis_protection = symbiosis_protection
        self.dynamic_symbiosis_cooccurrence_threshold = dynamic_symbiosis_cooccurrence_threshold
        self.dynamic_symbiosis_affinity_threshold = dynamic_symbiosis_affinity_threshold
        self.max_symbiosis_pairs = max_symbiosis_pairs

        # STDP
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.temporal_decay = temporal_decay
        self.agg_ltp_window = agg_ltp_window
        self.agg_ltp_rate = agg_ltp_rate
        self.cross_ltp_rate = cross_ltp_rate

        # Morphogen
        self.sin_source_strength = sin_source_strength
        self.extreme_agg_source_strength = extreme_agg_source_strength
        self.morphogen_decay_rate = morphogen_decay_rate
        self.dynamic_source_strength = dynamic_source_strength
        self.max_agg_sources = max_agg_sources
        self.agg_source_threshold = agg_source_threshold
        self.agg_source_contribution_threshold = agg_source_contribution_threshold
        self.cross_morphogen_influence = cross_morphogen_influence

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_affinity_lr = cross_affinity_lr

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Mode-specific adjustments
        if discovery_mode == 'activation_first':
            # Boost activation learning, reduce aggregation learning
            self.act_affinity_lr *= 1.3
            self.agg_affinity_lr *= 0.8
        elif discovery_mode == 'aggregation_first':
            # Boost aggregation learning, reduce activation learning
            self.act_affinity_lr *= 0.8
            self.agg_affinity_lr *= 1.3
            self.agg_ltp_rate *= 1.2

        # Initial palettes
        default_act = list(DEFAULT_PALETTE_INDICES)
        if SIN_IDX not in default_act:
            default_act.append(SIN_IDX)

        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        for agg in CORE_EXTREME_AGGS:
            if agg not in default_agg:
                default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

        self.category_lr_multipliers = category_lr_multipliers or {
            'known_synergistic': 1.5,
            'oscillatory_extreme': 1.3,
            'smooth_averaging': 1.0,
            'rectified_extreme': 1.2,
            'periodic_averaging': 1.1,
        }

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize unified discovery state."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # Mode-specific initial seeding
        if self.discovery_mode == 'aggregation_first':
            # Start with more aggregations, fewer activations
            if len(initial_agg) < 4:
                for agg in range(NUM_AGGREGATIONS):
                    if agg not in initial_agg and len(initial_agg) < 4:
                        initial_agg = list(initial_agg) + [agg]
        else:
            # Standard initial seeding
            if SIN_IDX not in initial_act:
                initial_act = list(initial_act) + [SIN_IDX]
            for agg in CORE_EXTREME_AGGS:
                if agg not in initial_agg:
                    initial_agg = list(initial_agg) + [agg]

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        act_affinities = act_affinities.at[SIN_IDX].set(0.8)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for agg in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[agg].set(0.75)

        # Symbiosis state
        symbiosis_strength = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        for act_idx, agg_idx in self.symbiosis_pairs:
            symbiosis_strength = symbiosis_strength.at[act_idx, agg_idx].set(1.0)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[SIN_IDX, agg].set(0.7)

        # Morphogen fields
        act_morphogen = jnp.zeros(NUM_ACTIVATIONS)
        agg_morphogen = jnp.zeros(NUM_AGGREGATIONS)
        act_morphogen = act_morphogen.at[SIN_IDX].set(self.sin_source_strength)
        for agg in CORE_EXTREME_AGGS:
            agg_morphogen = agg_morphogen.at[agg].set(self.extreme_agg_source_strength)

        # Cross-morphogen
        cross_morphogen = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        for agg in CORE_EXTREME_AGGS:
            cross_morphogen = cross_morphogen.at[SIN_IDX, agg].set(0.5)

        # Cross-domain STDP credit
        cross_ltp_credit = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        for agg in CORE_EXTREME_AGGS:
            cross_ltp_credit = cross_ltp_credit.at[SIN_IDX, agg].set(0.3)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'symbiosis_strength': symbiosis_strength,
            'cross_affinity': cross_affinity,
            'act_morphogen': act_morphogen,
            'agg_morphogen': agg_morphogen,
            'cross_morphogen': cross_morphogen,
            'act_ltp_credit': jnp.zeros(NUM_ACTIVATIONS),
            'agg_ltp_credit': jnp.zeros(NUM_AGGREGATIONS),
            'agg_ltd_credit': jnp.zeros(NUM_AGGREGATIONS),
            'cross_ltp_credit': cross_ltp_credit,
            'cross_ltd_credit': jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS)),
            'cooccurrence_counts': jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS)),
            'agg_sources': set(CORE_EXTREME_AGGS),
            'agg_contribution_counts': jnp.zeros(NUM_AGGREGATIONS),
            'stdp_history': [],
            'agg_stdp_history': [],
            'rng_key': jax.random.PRNGKey(seed + 2010000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'discovery_mode': self.discovery_mode,
            'fitness_history': [],
            # Tracking
            'dynamic_pairs_formed': 0,
            'discovered_sources': 0,
            'cross_stdp_updates': 0,
            'cross_morphogen_updates': 0,
            # Discovery effectiveness
            'discovery_to_palette': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_temporal_weight(self, gens_before: int) -> float:
        return self.temporal_decay ** gens_before

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

    def _update_dynamic_symbiosis(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
        cooccurrence_counts: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Update cooccurrence and discover new symbiotic pairs."""
        if not self.enable_dynamic_symbiosis:
            return symbiosis_strength, cooccurrence_counts, 0

        new_cooccurrence = cooccurrence_counts.copy()
        new_symbiosis = symbiosis_strength.copy()
        new_pairs = 0

        # Update cooccurrence
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                    new_cooccurrence = new_cooccurrence.at[i, j].set(new_cooccurrence[i, j] + 1)
                else:
                    new_cooccurrence = new_cooccurrence.at[i, j].set(max(0, new_cooccurrence[i, j] * 0.8))

        # Count current pairs
        current_pairs = 0
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if new_symbiosis[i, j] > 0.3:
                    current_pairs += 1

        # Discover new pairs
        if fitness_delta > 0 and current_pairs < self.max_symbiosis_pairs:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if new_symbiosis[i, j] > 0.3:
                        continue
                    cooccurrence_ok = new_cooccurrence[i, j] >= self.dynamic_symbiosis_cooccurrence_threshold
                    affinity_ok = cross_affinity[i, j] >= self.dynamic_symbiosis_affinity_threshold
                    both_active = act_mask[i] > 0.5 and agg_mask[j] > 0.5
                    if cooccurrence_ok and affinity_ok and both_active:
                        new_symbiosis = new_symbiosis.at[i, j].set(0.5)
                        new_pairs += 1
                        current_pairs += 1
                        if current_pairs >= self.max_symbiosis_pairs:
                            break
                if current_pairs >= self.max_symbiosis_pairs:
                    break

        # Decay symbiosis for inactive pairs
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if act_mask[i] < 0.5 or agg_mask[j] < 0.5:
                    new_symbiosis = new_symbiosis.at[i, j].set(new_symbiosis[i, j] * 0.9)

        # Maintain floors for known pairs
        for act_idx, agg_idx in self.symbiosis_pairs:
            new_symbiosis = new_symbiosis.at[act_idx, agg_idx].set(
                max(0.5, float(new_symbiosis[act_idx, agg_idx]))
            )

        return new_symbiosis, new_cooccurrence, new_pairs

    def _update_stdp(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_ltp: jnp.ndarray,
        agg_ltp: jnp.ndarray,
        agg_ltd: jnp.ndarray,
        cross_ltp: jnp.ndarray,
        cross_ltd: jnp.ndarray,
        stdp_history: List,
        agg_stdp_history: List,
        generation: int,
        improved: bool,
        stagnated: bool,
        has_partner_act: jnp.ndarray,
        has_partner_agg: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List, List, int]:
        """Update all STDP credits."""
        # Decay
        new_act_ltp = act_ltp * 0.95
        new_agg_ltp = agg_ltp * 0.92
        new_agg_ltd = agg_ltd * 0.95
        new_cross_ltp = cross_ltp * 0.93
        new_cross_ltd = cross_ltd * 0.95
        cross_updates = 0

        # Update histories
        new_stdp_history = stdp_history + [(generation, act_mask.copy(), agg_mask.copy())]
        if len(new_stdp_history) > 10:
            new_stdp_history = new_stdp_history[-10:]

        new_agg_history = agg_stdp_history + [(generation, agg_mask.copy())]
        if len(new_agg_history) > 10:
            new_agg_history = new_agg_history[-10:]

        # LTP updates on improvement
        if improved and len(new_stdp_history) >= 2:
            for hist_gen, hist_act_mask, hist_agg_mask in new_stdp_history:
                gens_before = generation - hist_gen
                if 1 <= gens_before <= self.ltp_window:
                    temporal_weight = self._compute_temporal_weight(gens_before)

                    # Activation LTP
                    for i in range(NUM_ACTIVATIONS):
                        if hist_act_mask[i] > 0.5:
                            bonus = 1.3 if has_partner_act[i] > 0.5 else 1.0
                            delta = self.ltp_rate * temporal_weight * bonus
                            new_act_ltp = new_act_ltp.at[i].set(min(1.0, new_act_ltp[i] + delta))

                    # Aggregation LTP (independent)
                    if self.enable_independent_agg_stdp:
                        for j in range(NUM_AGGREGATIONS):
                            if hist_agg_mask[j] > 0.5:
                                bonus = 1.3 if has_partner_agg[j] > 0.5 else 1.0
                                if j in CORE_EXTREME_AGGS:
                                    bonus *= 1.2
                                delta = self.agg_ltp_rate * temporal_weight * bonus
                                new_agg_ltp = new_agg_ltp.at[j].set(min(1.0, new_agg_ltp[j] + delta))

                    # Cross-domain LTP
                    if self.enable_cross_stdp:
                        for i in range(NUM_ACTIVATIONS):
                            for j in range(NUM_AGGREGATIONS):
                                if hist_act_mask[i] > 0.5 and hist_agg_mask[j] > 0.5:
                                    multiplier = 1.0
                                    for category, pairs in CROSS_PAIR_CATEGORIES.items():
                                        if (i, j) in pairs:
                                            multiplier = self.category_lr_multipliers.get(category, 1.0)
                                            break
                                    delta = self.cross_ltp_rate * temporal_weight * multiplier
                                    new_cross_ltp = new_cross_ltp.at[i, j].set(min(1.0, new_cross_ltp[i, j] + delta))
                                    cross_updates += 1

        # LTD on stagnation
        if stagnated and self.enable_independent_agg_stdp:
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5 and has_partner_agg[j] < 0.5:
                    new_agg_ltd = new_agg_ltd.at[j].set(min(1.0, new_agg_ltd[j] + 0.1))

        # Maintain minimum cross-LTP for known pairs
        for agg in CORE_EXTREME_AGGS:
            new_cross_ltp = new_cross_ltp.at[SIN_IDX, agg].set(max(0.2, float(new_cross_ltp[SIN_IDX, agg])))

        return new_act_ltp, new_agg_ltp, new_agg_ltd, new_cross_ltp, new_cross_ltd, new_stdp_history, new_agg_history, cross_updates

    def _update_morphogen(
        self,
        act_morphogen: jnp.ndarray,
        agg_morphogen: jnp.ndarray,
        cross_morphogen: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        agg_sources: set,
        agg_contribution_counts: jnp.ndarray,
        fitness_delta: float,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, set, jnp.ndarray, int, int]:
        """Update morphogen fields and discover new sources."""
        # Decay
        new_act_morph = act_morphogen * (1.0 - self.morphogen_decay_rate)
        new_agg_morph = agg_morphogen * (1.0 - self.morphogen_decay_rate)
        new_cross_morph = cross_morphogen * 0.95
        new_cross_aff = cross_affinity.copy()

        # Maintain source strengths
        new_act_morph = new_act_morph.at[SIN_IDX].set(max(float(new_act_morph[SIN_IDX]), self.sin_source_strength * 0.9))
        for agg in CORE_EXTREME_AGGS:
            new_agg_morph = new_agg_morph.at[agg].set(max(float(new_agg_morph[agg]), self.extreme_agg_source_strength * 0.8))

        # Contribute from active functions
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                contrib = 0.3 if i == SIN_IDX else 0.1
                new_act_morph = new_act_morph.at[i].set(min(3.0, new_act_morph[i] + contrib))

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                contrib = 0.25 if j in CORE_EXTREME_AGGS else 0.15
                new_agg_morph = new_agg_morph.at[j].set(min(3.0, new_agg_morph[j] + contrib))

        # Dynamic source discovery
        new_sources = set(agg_sources)
        new_counts = agg_contribution_counts.copy()
        discovered = 0

        if self.enable_agg_morphogen_discovery:
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    if improved:
                        new_counts = new_counts.at[j].set(new_counts[j] + 1)
                    else:
                        new_counts = new_counts.at[j].set(max(0, new_counts[j] - 0.5))
                else:
                    new_counts = new_counts.at[j].set(max(0, new_counts[j] - 1))

            if len(new_sources) < self.max_agg_sources:
                for j in range(NUM_AGGREGATIONS):
                    if j not in new_sources:
                        affinity_ok = float(agg_affinities[j]) >= self.agg_source_threshold
                        contribution_ok = float(new_counts[j]) >= self.agg_source_contribution_threshold
                        is_active = agg_mask[j] > 0.5
                        if affinity_ok and contribution_ok and is_active:
                            new_sources.add(j)
                            new_agg_morph = new_agg_morph.at[j].set(max(float(new_agg_morph[j]), self.dynamic_source_strength))
                            discovered += 1
                            if len(new_sources) >= self.max_agg_sources:
                                break

        # Cross-morphogen influence
        cross_updates = 0
        if self.enable_cross_morphogen and fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        combined = (float(new_act_morph[i]) + float(new_agg_morph[j])) / 2
                        if combined > 0.5:
                            multiplier = 1.0
                            for category, pairs in CROSS_PAIR_CATEGORIES.items():
                                if (i, j) in pairs:
                                    multiplier = self.category_lr_multipliers.get(category, 1.0)
                                    break
                            delta = self.cross_morphogen_influence * combined * multiplier
                            new_cross_morph = new_cross_morph.at[i, j].set(min(1.0, new_cross_morph[i, j] + delta))
                            aff_delta = 0.1 * float(new_cross_morph[i, j])
                            new_cross_aff = new_cross_aff.at[i, j].set(min(1.0, new_cross_aff[i, j] + aff_delta))
                            cross_updates += 1

        # Minimum for known pairs
        for agg in CORE_EXTREME_AGGS:
            new_cross_morph = new_cross_morph.at[SIN_IDX, agg].set(max(0.3, float(new_cross_morph[SIN_IDX, agg])))

        return new_act_morph, new_agg_morph, new_cross_morph, new_cross_aff, new_sources, new_counts, discovered, cross_updates

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with all unified mechanisms."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness
        stagnated = fitness_delta < 0.0001

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Check symbiosis status
        has_partner_act, has_partner_agg = self._check_symbiosis_status(
            state['act_mask'], state['agg_mask'], state['symbiosis_strength']
        )

        # Update dynamic symbiosis
        new_symbiosis, new_cooccurrence, new_pairs = self._update_dynamic_symbiosis(
            state['act_mask'], state['agg_mask'], state['cross_affinity'],
            state['symbiosis_strength'], state['cooccurrence_counts'], fitness_delta
        )

        # Update STDP
        (new_act_ltp, new_agg_ltp, new_agg_ltd, new_cross_ltp, new_cross_ltd,
         new_stdp_history, new_agg_history, cross_stdp_updates) = self._update_stdp(
            state['act_mask'], state['agg_mask'],
            state['act_ltp_credit'], state['agg_ltp_credit'], state['agg_ltd_credit'],
            state['cross_ltp_credit'], state['cross_ltd_credit'],
            state['stdp_history'], state['agg_stdp_history'],
            generation, improved, stagnated, has_partner_act, has_partner_agg
        )

        # Update morphogen
        (new_act_morph, new_agg_morph, new_cross_morph, cross_aff_updated,
         new_agg_sources, new_agg_counts, discovered_sources, cross_morph_updates) = self._update_morphogen(
            state['act_morphogen'], state['agg_morphogen'], state['cross_morphogen'],
            state['cross_affinity'], state['act_mask'], state['agg_mask'],
            state['agg_affinities'], state['agg_sources'], state['agg_contribution_counts'],
            fitness_delta, improved
        )

        # Update affinities
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay

        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    lr = self.act_affinity_lr * (1 + float(new_act_ltp[i]) * 0.5)
                    bonus = 1.5 if i == SIN_IDX else 1.0
                    if has_partner_act[i] > 0.5:
                        bonus *= 1.2
                    new_act_aff = new_act_aff.at[i].set(min(1.0, new_act_aff[i] + lr * fitness_delta * bonus))

            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    lr = self.agg_affinity_lr * (1 + float(new_agg_ltp[j]) * 0.5)
                    bonus = 1.4 if j in CORE_EXTREME_AGGS else 1.0
                    if has_partner_agg[j] > 0.5:
                        bonus *= 1.2
                    new_agg_aff = new_agg_aff.at[j].set(min(1.0, new_agg_aff[j] + lr * fitness_delta * bonus))

        # Affinity floors
        new_act_aff = new_act_aff.at[SIN_IDX].set(max(0.6, float(new_act_aff[SIN_IDX])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Update cross-affinity from morphogen
        new_cross_aff = cross_aff_updated.copy()
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if state['act_mask'][i] > 0.5 and state['agg_mask'][j] > 0.5:
                        delta = self.cross_affinity_lr * fitness_delta
                        if i == SIN_IDX and j in CORE_EXTREME_AGGS:
                            delta *= 1.5
                        new_cross_aff = new_cross_aff.at[i, j].set(min(1.0, new_cross_aff[i, j] + delta))

        # === Discovery-to-Palette Bridge ===
        # Identify which aggregation sources are NEWLY discovered this generation
        newly_discovered_aggs = [j for j in new_agg_sources if j not in state['agg_sources']]
        discovery_to_palette = 0

        # Select palettes based on mode
        if self.discovery_mode == 'aggregation_first':
            # Select aggregation first, then activation
            agg_score = new_agg_aff + new_agg_morph * 0.4 + new_agg_ltp * 0.3 + has_partner_agg * 0.15
            for agg in CORE_EXTREME_AGGS:
                agg_score = agg_score.at[agg].set(agg_score[agg] + 0.5)
            # Give newly discovered agg sources a one-time boost
            for j in newly_discovered_aggs:
                agg_score = agg_score.at[j].set(agg_score[j] + 0.4)  # Discovery bonus
            target_agg = min(max(2, self.min_active_agg), self.max_active_agg)
            top_agg = jnp.argsort(agg_score)[-target_agg:]
            new_agg_mask = jnp.zeros(NUM_AGGREGATIONS)
            for idx in top_agg:
                new_agg_mask = new_agg_mask.at[int(idx)].set(1.0)
            # Ensure at least one newly discovered agg enters the palette
            if newly_discovered_aggs:
                best_new = max(newly_discovered_aggs, key=lambda j: float(new_agg_aff[j]))
                if new_agg_mask[best_new] < 0.5:  # Not already selected
                    new_agg_mask = new_agg_mask.at[best_new].set(1.0)
                    discovery_to_palette += 1

            # Then select activations based on cross-affinity with selected aggs
            act_score = new_act_aff + new_act_morph * 0.4 + new_act_ltp * 0.3 + has_partner_act * 0.15
            for i in range(NUM_ACTIVATIONS):
                cross_boost = 0.0
                for j in range(NUM_AGGREGATIONS):
                    if new_agg_mask[j] > 0.5:
                        cross_boost = max(cross_boost, float(new_cross_aff[i, j]) * 0.3)
                act_score = act_score.at[i].set(act_score[i] + cross_boost)
            act_score = act_score.at[SIN_IDX].set(act_score[SIN_IDX] + 0.5)
            target_act = min(max(3, self.min_active_act), self.max_active_act)
            top_act = jnp.argsort(act_score)[-target_act:]
            new_act_mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_act:
                new_act_mask = new_act_mask.at[int(idx)].set(1.0)
        else:
            # Balanced or activation_first
            act_score = new_act_aff + new_act_morph * 0.4 + new_act_ltp * 0.3 + has_partner_act * 0.15
            act_score = act_score.at[SIN_IDX].set(act_score[SIN_IDX] + 0.5)
            for i in range(NUM_ACTIVATIONS):
                cross_boost = 0.0
                for j in range(NUM_AGGREGATIONS):
                    if state['agg_mask'][j] > 0.5:
                        cross_boost = max(cross_boost, float(new_cross_aff[i, j]) * 0.3)
                act_score = act_score.at[i].set(act_score[i] + cross_boost)
            target_act = min(max(3, self.min_active_act), self.max_active_act)
            top_act = jnp.argsort(act_score)[-target_act:]
            new_act_mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_act:
                new_act_mask = new_act_mask.at[int(idx)].set(1.0)

            agg_score = new_agg_aff + new_agg_morph * 0.4 + new_agg_ltp * 0.3 + has_partner_agg * 0.15
            for agg in CORE_EXTREME_AGGS:
                agg_score = agg_score.at[agg].set(agg_score[agg] + 0.5)
            # Give newly discovered agg sources a one-time boost
            for j in newly_discovered_aggs:
                agg_score = agg_score.at[j].set(agg_score[j] + 0.4)  # Discovery bonus
            target_agg = min(max(2, self.min_active_agg), self.max_active_agg)
            top_agg = jnp.argsort(agg_score)[-target_agg:]
            new_agg_mask = jnp.zeros(NUM_AGGREGATIONS)
            for idx in top_agg:
                new_agg_mask = new_agg_mask.at[int(idx)].set(1.0)
            # Ensure at least one newly discovered agg enters the palette
            if newly_discovered_aggs:
                best_new = max(newly_discovered_aggs, key=lambda j: float(new_agg_aff[j]))
                if new_agg_mask[best_new] < 0.5:  # Not already selected
                    new_agg_mask = new_agg_mask.at[best_new].set(1.0)
                    discovery_to_palette += 1

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
            'symbiosis_strength': new_symbiosis,
            'cross_affinity': new_cross_aff,
            'act_morphogen': new_act_morph,
            'agg_morphogen': new_agg_morph,
            'cross_morphogen': new_cross_morph,
            'act_ltp_credit': new_act_ltp,
            'agg_ltp_credit': new_agg_ltp,
            'agg_ltd_credit': new_agg_ltd,
            'cross_ltp_credit': new_cross_ltp,
            'cross_ltd_credit': new_cross_ltd,
            'cooccurrence_counts': new_cooccurrence,
            'agg_sources': new_agg_sources,
            'agg_contribution_counts': new_agg_counts,
            'stdp_history': new_stdp_history,
            'agg_stdp_history': new_agg_history,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'fitness_history': state['fitness_history'] + [best_fitness],
            'dynamic_pairs_formed': state['dynamic_pairs_formed'] + new_pairs,
            'discovered_sources': state['discovered_sources'] + discovered_sources,
            'cross_stdp_updates': state['cross_stdp_updates'] + cross_stdp_updates,
            'cross_morphogen_updates': state['cross_morphogen_updates'] + cross_morph_updates,
            # Track discovery-to-palette effectiveness
            'discovery_to_palette': state.get('discovery_to_palette', 0) + discovery_to_palette,
        }

        # Count metrics
        total_symbiotic_pairs = sum(1 for i in range(NUM_ACTIVATIONS) for j in range(NUM_AGGREGATIONS) if new_symbiosis[i, j] > 0.3)
        non_sin_extreme_pairs = sum(1 for i in range(NUM_ACTIVATIONS) for j in range(NUM_AGGREGATIONS)
                                    if new_symbiosis[i, j] > 0.3 and (i != SIN_IDX or j not in CORE_EXTREME_AGGS))
        high_cross_ltp = sum(1 for i in range(NUM_ACTIVATIONS) for j in range(NUM_AGGREGATIONS) if new_cross_ltp[i, j] > 0.5)
        high_cross_morph = sum(1 for i in range(NUM_ACTIVATIONS) for j in range(NUM_AGGREGATIONS) if new_cross_morph[i, j] > 0.5)

        metrics = {
            'discovery_mode': self.discovery_mode,
            'dynamic_pairs_formed': new_pairs,
            'discovered_sources': discovered_sources,
            'cross_stdp_updates': cross_stdp_updates,
            'cross_morph_updates': cross_morph_updates,
            'total_symbiotic_pairs': total_symbiotic_pairs,
            'non_sin_extreme_pairs': non_sin_extreme_pairs,
            'high_cross_ltp': high_cross_ltp,
            'high_cross_morph': high_cross_morph,
            'total_agg_sources': len(new_agg_sources),
            'has_partners_act': int(has_partner_act.sum()),
            'has_partners_agg': int(has_partner_agg.sum()),
            # Discovery effectiveness
            'discovery_to_palette': discovery_to_palette,
            'newly_discovered_aggs': len(newly_discovered_aggs),
        }

        return new_state, metrics
