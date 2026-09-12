"""Strategy 118: STDP-Neurogenesis Cross-Domain Credit Dual.

Combines STDP cross-domain (#16) with Neurogenesis (#113).
(sin+max) pairs that co-activate before improvement both get survival bonus.
Cross-domain temporal correlation guides mutual survival.

Key Innovation:
- Cross-domain LTP tracks act-agg pairs that co-activate before improvement
- Pairs with high cross-LTP get mutual survival bonuses
- Sin+extreme pairs receive extra cross-LTP credit
- Creates strong pressure for functional pairings

Biological basis: In the brain, neural ensembles that co-activate before
reward get jointly strengthened. This strategy applies the same principle
across activation and aggregation domains.

Expected: Strong sin+extreme coupling through mutual survival credit.
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


class STDPNeurogenesisCrossDualStrategy(PaletteEvolutionStrategy):
    """STDP-guided cross-domain neurogenesis for dual palette evolution.

    Combines cross-domain temporal credit (STDP) with
    neurogenesis birth/maturation/survival. Act-agg pairs that co-activate
    before improvement get mutual survival bonuses.

    Critical innovation: Cross-domain LTP matrix tracks co-activation patterns.
    High cross-LTP pairs both receive reduced survival thresholds.
    """

    name = "stdp_neurogenesis_cross_dual"
    description = "Dual: Cross-domain STDP creates mutual survival bonuses for paired functions"

    def __init__(
        self,
        # === Cross-domain STDP parameters ===
        cross_ltp_rate: float = 0.15,        # Cross-LTP accumulation rate
        mutual_survival_bonus: float = 0.4,   # Threshold reduction for pairs
        cross_ltp_survival_mult: float = 0.6, # How much cross-LTP affects survival
        ltp_window: int = 5,                  # Gens before improvement to credit
        ltd_window: int = 3,                  # Gens for LTD
        temporal_decay: float = 0.7,
        # === Neurogenesis parameters ===
        neurogenesis_rate: float = 0.10,
        maturation_period: int = 10,
        base_survival_threshold: float = 0.15,
        max_young: int = 2,
        # === Sin-extreme boost ===
        sin_extreme_cross_boost: float = 0.5,
        sin_idx: int = 4,
        # === General parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        tag_threshold: float = 0.5,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.85,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize STDP-Neurogenesis Cross-Domain strategy."""
        # Cross-domain STDP
        self.cross_ltp_rate = cross_ltp_rate
        self.mutual_survival_bonus = mutual_survival_bonus
        self.cross_ltp_survival_mult = cross_ltp_survival_mult
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.temporal_decay = temporal_decay

        # Neurogenesis
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.base_survival_threshold = base_survival_threshold
        self.max_young = max_young

        # Sin-extreme boost
        self.sin_extreme_cross_boost = sin_extreme_cross_boost
        self.sin_idx = sin_idx

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay
        self.tag_threshold = tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with cross-domain STDP + neurogenesis tracking."""
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

        # Extra boost for extreme aggs
        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(agg_affinities[i] + 0.2)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain LTP matrix (NEW)
        cross_ltp_matrix = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        # Neurogenesis state
        young_acts: Dict[int, int] = {}  # {idx: birth_gen}
        young_aggs: Dict[int, int] = {}
        act_contribution = jnp.zeros(NUM_ACTIVATIONS)
        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_contribution = act_contribution.at[i].set(0.4)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.4)

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
            # Cross-domain LTP (NEW)
            'cross_ltp_matrix': cross_ltp_matrix,
            'stdp_history': [],  # (gen, act_mask, agg_mask, fitness) tuples
            # Neurogenesis
            'young_acts': young_acts,
            'young_aggs': young_aggs,
            'act_contribution': act_contribution,
            'agg_contribution': agg_contribution,
            # Stats
            'capture_events': 0,
            'cross_ltp_events': 0,
            'total_births': 0,
            'mutual_survival_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1180000),
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

    def _update_cross_ltp(
        self,
        cross_ltp_matrix: jnp.ndarray,
        stdp_history: List[Tuple],
        current_gen: int,
        improved: bool,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, int]:
        """Update cross-domain LTP based on co-activation before improvement."""
        events = 0

        if not improved or len(stdp_history) < 2:
            return cross_ltp_matrix, events

        # Credit co-active pairs in LTP window
        for hist_gen, hist_act, hist_agg, _ in stdp_history[-self.ltp_window:]:
            gens_before = current_gen - hist_gen
            if gens_before > 0 and gens_before <= self.ltp_window:
                weight = self.temporal_decay ** gens_before

                # Find active pairs at that time
                active_acts = jnp.where(hist_act > 0.5)[0]
                active_aggs = jnp.where(hist_agg > 0.5)[0]

                for a in active_acts:
                    for g in active_aggs:
                        credit = self.cross_ltp_rate * weight
                        # Extra boost for sin + extreme pairs
                        if int(a) == self.sin_idx and int(g) in CORE_EXTREME_AGGS:
                            credit += self.sin_extreme_cross_boost * weight
                        cross_ltp_matrix = cross_ltp_matrix.at[a, g].add(credit)
                        events += 1

        # Clamp to [0, 1]
        cross_ltp_matrix = jnp.clip(cross_ltp_matrix, 0.0, 1.0)

        return cross_ltp_matrix, events

    def _get_pair_survival_bonus(
        self,
        idx: int,
        is_activation: bool,
        cross_ltp_matrix: jnp.ndarray,
        other_mask: jnp.ndarray,
    ) -> float:
        """Get survival threshold reduction from cross-domain LTP."""
        bonus = 0.0

        if is_activation:
            # Sum LTP from active aggregations
            active_aggs = jnp.where(other_mask > 0.5)[0]
            for g in active_aggs:
                ltp = float(cross_ltp_matrix[idx, g])
                bonus += ltp * self.mutual_survival_bonus * self.cross_ltp_survival_mult
        else:
            # Sum LTP from active activations
            active_acts = jnp.where(other_mask > 0.5)[0]
            for a in active_acts:
                ltp = float(cross_ltp_matrix[a, idx])
                bonus += ltp * self.mutual_survival_bonus * self.cross_ltp_survival_mult

        return min(0.5, bonus)  # Cap bonus

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with cross-domain STDP-guided neurogenesis."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Track STDP history
        stdp_history = list(state['stdp_history'])
        stdp_history.append((generation, state['act_mask'], state['agg_mask'], best_fitness))
        if len(stdp_history) > 10:
            stdp_history = stdp_history[-10:]

        # Update cross-domain LTP
        cross_ltp_matrix, ltp_events = self._update_cross_ltp(
            state['cross_ltp_matrix'],
            stdp_history,
            generation,
            improved,
            state['act_mask'],
            state['agg_mask'],
        )

        # Decay cross-LTP
        cross_ltp_matrix = cross_ltp_matrix * self.affinity_decay

        # === ACTIVATION UPDATE ===
        act_mask = state['act_mask']
        act_affinities = state['act_affinities']
        act_tags = state['act_tags'] * self.tag_decay
        act_captured = state['act_captured']
        young_acts = dict(state['young_acts'])
        act_contribution = state['act_contribution'] * 0.95

        # Neurogenesis: maybe birth new activation
        if jax.random.uniform(k1) < self.neurogenesis_rate and len(young_acts) < self.max_young:
            candidates = [i for i in range(NUM_ACTIVATIONS)
                         if float(act_mask[i]) < 0.5 and i not in young_acts]
            if candidates:
                # Prefer sin if not active
                if self.sin_idx in candidates:
                    new_idx = self.sin_idx
                else:
                    new_idx = int(jax.random.choice(k2, jnp.array(candidates)))
                young_acts[new_idx] = generation
                act_mask = act_mask.at[new_idx].set(1.0)

        # Check maturation for young activations
        survived = []
        pruned = []
        for idx, birth_gen in list(young_acts.items()):
            age = generation - birth_gen
            if age >= self.maturation_period:
                # Get survival threshold with cross-domain bonus
                threshold = self.base_survival_threshold
                bonus = self._get_pair_survival_bonus(idx, True, cross_ltp_matrix, state['agg_mask'])
                threshold = max(0.05, threshold - bonus)

                contrib = float(act_contribution[idx])
                if contrib >= threshold:
                    survived.append(idx)
                else:
                    pruned.append(idx)
                    act_mask = act_mask.at[idx].set(0.0)
                del young_acts[idx]

        # === AGGREGATION UPDATE ===
        agg_mask = state['agg_mask']
        agg_affinities = state['agg_affinities']
        agg_tags = state['agg_tags'] * self.tag_decay
        agg_captured = state['agg_captured']
        young_aggs = dict(state['young_aggs'])
        agg_contribution = state['agg_contribution'] * 0.95

        # Neurogenesis: maybe birth new aggregation (prefer extremes)
        if jax.random.uniform(k3) < self.neurogenesis_rate and len(young_aggs) < self.max_young:
            # Prefer missing extreme aggs
            missing_extreme = [i for i in CORE_EXTREME_AGGS
                              if float(agg_mask[i]) < 0.5 and i not in young_aggs]
            if missing_extreme and jax.random.uniform(k4) < 0.7:
                new_idx = int(jax.random.choice(k4, jnp.array(missing_extreme)))
            else:
                candidates = [i for i in range(NUM_AGGREGATIONS)
                             if float(agg_mask[i]) < 0.5 and i not in young_aggs]
                if candidates:
                    new_idx = int(jax.random.choice(k4, jnp.array(candidates)))
                else:
                    new_idx = None

            if new_idx is not None:
                young_aggs[new_idx] = generation
                agg_mask = agg_mask.at[new_idx].set(1.0)

        # Check maturation for young aggregations
        agg_survived = []
        agg_pruned = []
        mutual_events = 0
        for idx, birth_gen in list(young_aggs.items()):
            age = generation - birth_gen
            if age >= self.maturation_period:
                threshold = self.base_survival_threshold
                bonus = self._get_pair_survival_bonus(idx, False, cross_ltp_matrix, state['act_mask'])
                if bonus > 0.1:
                    mutual_events += 1
                threshold = max(0.05, threshold - bonus)

                contrib = float(agg_contribution[idx])
                if contrib >= threshold:
                    agg_survived.append(idx)
                else:
                    agg_pruned.append(idx)
                    agg_mask = agg_mask.at[idx].set(0.0)
                del young_aggs[idx]

        # === Update affinities on improvement ===
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
                act_contribution = act_contribution.at[a].add(0.15)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)
                agg_contribution = agg_contribution.at[g].add(0.15)

        # Clamp affinities
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)
        act_contribution = jnp.clip(act_contribution, 0.0, 1.0)
        agg_contribution = jnp.clip(agg_contribution, 0.0, 1.0)

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

        # Build new state
        new_state = {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'cross_ltp_matrix': cross_ltp_matrix,
            'stdp_history': stdp_history,
            'young_acts': young_acts,
            'young_aggs': young_aggs,
            'act_contribution': act_contribution,
            'agg_contribution': agg_contribution,
            'capture_events': state['capture_events'],
            'cross_ltp_events': state['cross_ltp_events'] + ltp_events,
            'total_births': state['total_births'] + len(survived) + len(agg_survived),
            'mutual_survival_events': state['mutual_survival_events'] + mutual_events,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'act_survived': len(survived),
            'act_pruned': len(pruned),
            'agg_survived': len(agg_survived),
            'agg_pruned': len(agg_pruned),
            'cross_ltp_events': ltp_events,
            'mutual_survival_events': mutual_events,
        }

        return new_state, metrics
