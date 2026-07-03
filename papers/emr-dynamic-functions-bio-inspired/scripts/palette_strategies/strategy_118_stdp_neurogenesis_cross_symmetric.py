"""Strategy 118S: STDP-Neurogenesis Cross-Domain Credit Symmetric.

Biological Basis: Cross-domain temporal correlation guides mutual survival.
Act-agg pairs that co-activate before improvement both get survival bonuses.
Strong pressure for functional pairings through cross-domain LTP.

Key symmetric mechanisms:
1. Cross-domain LTP tracks act-agg pairs that co-activate before improvement
2. High cross-LTP pairs get mutual survival bonuses
3. Sin+extreme pairs receive extra cross-LTP credit
4. Protected indices for sin and extreme aggregations (0.1% deactivation)
5. Affinity floors for guaranteed retention
6. Memory cell crystallization for proven functions
7. Initial palettes include critical functions from start

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
    CORE_EXTREME_AGGS,
)

# Critical indices for retention
SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3


class STDPNeurogenesisCrossSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric cross-domain STDP-guided neurogenesis.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Initial palettes include sin and max/min from start
    - Cross-LTP matrix with sin-extreme bias
    """

    name = "stdp_neurogenesis_cross_symmetric"
    description = "Symmetric: Cross-domain STDP creates mutual survival bonuses"

    def __init__(
        self,
        # === Cross-domain STDP parameters ===
        cross_ltp_rate: float = 0.15,
        mutual_survival_bonus: float = 0.4,
        cross_ltp_survival_mult: float = 0.6,
        ltp_window: int = 5,
        temporal_decay: float = 0.7,
        # === Neurogenesis parameters ===
        neurogenesis_rate: float = 0.10,
        maturation_period: int = 10,
        base_survival_threshold: float = 0.15,
        max_young: int = 2,
        # === Sin-extreme boost ===
        sin_extreme_cross_boost: float = 0.5,
        # === Protected function settings ===
        protected_survival_discount: float = 0.5,
        protected_deactivation_prob: float = 0.001,
        # === Affinity floors ===
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # === Memory cell parameters ===
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
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
        """Initialize STDP-Neurogenesis Cross-Domain Symmetric strategy."""
        # Cross-domain STDP
        self.cross_ltp_rate = cross_ltp_rate
        self.mutual_survival_bonus = mutual_survival_bonus
        self.cross_ltp_survival_mult = cross_ltp_survival_mult
        self.ltp_window = ltp_window
        self.temporal_decay = temporal_decay

        # Neurogenesis
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.base_survival_threshold = base_survival_threshold
        self.max_young = max_young

        # Sin-extreme boost
        self.sin_extreme_cross_boost = sin_extreme_cross_boost

        # Protected function settings
        self.protected_survival_discount = protected_survival_discount
        self.protected_deactivation_prob = protected_deactivation_prob

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Memory cells
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # CRITICAL: Include sin and extreme aggregations in initial palettes
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with cross-domain STDP + neurogenesis tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # Ensure critical functions are included
        if SIN_IDX not in initial_act:
            initial_act = list(initial_act) + [SIN_IDX]
        if MAX_IDX not in initial_agg:
            initial_agg = list(initial_agg) + [MAX_IDX]
        if MIN_IDX not in initial_agg:
            initial_agg = list(initial_agg) + [MIN_IDX]

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize affinities with floors
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.4
        act_affinities = act_affinities.at[SIN_IDX].set(self.sin_affinity_floor)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(
                    max(0.5, float(act_affinities[i]))
                )

        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.45
        for idx in [MAX_IDX, MIN_IDX]:
            agg_affinities = agg_affinities.at[idx].set(self.extreme_agg_affinity_floor)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(
                    max(0.55, float(agg_affinities[i]))
                )

        # Cross-domain LTP matrix with sin-extreme initialization
        cross_ltp_matrix = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        for j in [MAX_IDX, MIN_IDX]:
            cross_ltp_matrix = cross_ltp_matrix.at[SIN_IDX, j].set(0.3)

        # Neurogenesis state
        young_acts: Dict[int, int] = {}
        young_aggs: Dict[int, int] = {}

        # Contribution tracking
        act_contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_contribution = act_contribution.at[i].set(0.5)
        act_contribution = act_contribution.at[SIN_IDX].set(0.7)

        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.5)
        for idx in [MAX_IDX, MIN_IDX]:
            agg_contribution = agg_contribution.at[idx].set(0.6)

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        discovery_gen = {
            'sin': 0 if SIN_IDX in initial_act else -1,
            'max': 0 if MAX_IDX in initial_agg else -1,
            'min': 0 if MIN_IDX in initial_agg else -1,
        }

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Cross-domain LTP
            'cross_ltp_matrix': cross_ltp_matrix,
            'stdp_history': [],
            # Neurogenesis
            'young_acts': young_acts,
            'young_aggs': young_aggs,
            'act_contribution': act_contribution,
            'agg_contribution': agg_contribution,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Discovery
            'discovery_gen': discovery_gen,
            # Stats
            'cross_ltp_events': 0,
            'mutual_survival_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1180001),
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

    def _apply_affinity_floors(
        self,
        act_aff: jnp.ndarray,
        agg_aff: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply minimum affinity floors for critical functions."""
        new_act = act_aff.at[SIN_IDX].set(
            jnp.maximum(act_aff[SIN_IDX], self.sin_affinity_floor)
        )
        new_agg = agg_aff
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )
        return new_act, new_agg

    def _update_memory_cells(
        self,
        affinity: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell status based on sustained high affinity."""
        active = mask > 0.5
        above_threshold = affinity >= self.memory_formation_threshold
        candidate = active & above_threshold
        new_counts = jnp.where(candidate, memory_counts + 1, 0)
        newly_memory = new_counts >= self.memory_formation_count
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)
        return new_counts, new_memory_cells

    def _is_protected(self, func_idx: int, is_activation: bool) -> bool:
        """Check if a function index is protected."""
        if is_activation:
            return func_idx == SIN_IDX
        else:
            return func_idx in [MAX_IDX, MIN_IDX]

    def _update_cross_ltp(
        self,
        cross_ltp_matrix: jnp.ndarray,
        stdp_history: List[Tuple],
        current_gen: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, int]:
        """Update cross-domain LTP based on co-activation before improvement."""
        events = 0

        if not improved or len(stdp_history) < 2:
            return cross_ltp_matrix, events

        for hist_gen, hist_act, hist_agg, _ in stdp_history[-self.ltp_window:]:
            gens_before = current_gen - hist_gen
            if gens_before > 0 and gens_before <= self.ltp_window:
                weight = self.temporal_decay ** gens_before

                active_acts = jnp.where(hist_act > 0.5)[0]
                active_aggs = jnp.where(hist_agg > 0.5)[0]

                for a in active_acts:
                    for g in active_aggs:
                        credit = self.cross_ltp_rate * weight
                        if int(a) == SIN_IDX and int(g) in [MAX_IDX, MIN_IDX]:
                            credit += self.sin_extreme_cross_boost * weight
                        cross_ltp_matrix = cross_ltp_matrix.at[a, g].add(credit)
                        events += 1

        return jnp.clip(cross_ltp_matrix, 0.0, 1.0), events

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
            active_aggs = jnp.where(other_mask > 0.5)[0]
            for g in active_aggs:
                ltp = float(cross_ltp_matrix[idx, g])
                bonus += ltp * self.mutual_survival_bonus * self.cross_ltp_survival_mult
        else:
            active_acts = jnp.where(other_mask > 0.5)[0]
            for a in active_acts:
                ltp = float(cross_ltp_matrix[a, idx])
                bonus += ltp * self.mutual_survival_bonus * self.cross_ltp_survival_mult

        return min(0.5, bonus)

    def _update_discovery_tracking(
        self,
        discovery_gen: Dict[str, int],
        act_palette: List[int],
        agg_palette: List[int],
        generation: int,
    ) -> Dict[str, int]:
        """Track when critical functions are first discovered."""
        new_discovery = discovery_gen.copy()
        if SIN_IDX in act_palette and new_discovery['sin'] < 0:
            new_discovery['sin'] = generation
        if MAX_IDX in agg_palette and new_discovery['max'] < 0:
            new_discovery['max'] = generation
        if MIN_IDX in agg_palette and new_discovery['min'] < 0:
            new_discovery['min'] = generation
        return new_discovery

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
        improvement = best_fitness - prev_best_fitness

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
            state['cross_ltp_matrix'], stdp_history, generation, improved
        )
        cross_ltp_matrix = cross_ltp_matrix * self.affinity_decay

        # === ACTIVATION UPDATE ===
        act_mask = state['act_mask'].copy()
        act_affinities = state['act_affinities'] * self.affinity_decay
        young_acts = dict(state['young_acts'])
        act_contribution = state['act_contribution'] * 0.95

        # Neurogenesis: maybe birth new activation (prefer sin)
        if jax.random.uniform(k1) < self.neurogenesis_rate and len(young_acts) < self.max_young:
            candidates = [i for i in range(NUM_ACTIVATIONS)
                         if float(act_mask[i]) < 0.5 and i not in young_acts]
            if candidates:
                if SIN_IDX in candidates and jax.random.uniform(k2) < 0.7:
                    new_idx = SIN_IDX
                else:
                    new_idx = int(jax.random.choice(k2, jnp.array(candidates)))
                young_acts[new_idx] = generation
                act_mask = act_mask.at[new_idx].set(1.0)

        # Check maturation
        survived_acts = []
        pruned_acts = []
        mutual_events = 0
        for idx, birth_gen in list(young_acts.items()):
            age = generation - birth_gen
            if age >= self.maturation_period:
                threshold = self.base_survival_threshold
                bonus = self._get_pair_survival_bonus(idx, True, cross_ltp_matrix, state['agg_mask'])
                if bonus > 0.1:
                    mutual_events += 1
                threshold = max(0.05, threshold - bonus)

                if self._is_protected(idx, True):
                    threshold *= self.protected_survival_discount
                    if jax.random.uniform(k1) < self.protected_deactivation_prob:
                        pruned_acts.append(idx)
                        act_mask = act_mask.at[idx].set(0.0)
                    else:
                        survived_acts.append(idx)
                elif float(act_contribution[idx]) >= threshold or state['act_memory_cells'][idx]:
                    survived_acts.append(idx)
                else:
                    pruned_acts.append(idx)
                    act_mask = act_mask.at[idx].set(0.0)
                del young_acts[idx]

        # === AGGREGATION UPDATE ===
        agg_mask = state['agg_mask'].copy()
        agg_affinities = state['agg_affinities'] * self.affinity_decay
        young_aggs = dict(state['young_aggs'])
        agg_contribution = state['agg_contribution'] * 0.95

        # Neurogenesis: maybe birth new aggregation (prefer extremes)
        if jax.random.uniform(k3) < self.neurogenesis_rate and len(young_aggs) < self.max_young:
            missing_extreme = [i for i in [MAX_IDX, MIN_IDX]
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

        # Check maturation
        survived_aggs = []
        pruned_aggs = []
        for idx, birth_gen in list(young_aggs.items()):
            age = generation - birth_gen
            if age >= self.maturation_period:
                threshold = self.base_survival_threshold
                bonus = self._get_pair_survival_bonus(idx, False, cross_ltp_matrix, state['act_mask'])
                if bonus > 0.1:
                    mutual_events += 1
                threshold = max(0.05, threshold - bonus)

                if self._is_protected(idx, False):
                    threshold *= self.protected_survival_discount
                    if jax.random.uniform(k3) < self.protected_deactivation_prob:
                        pruned_aggs.append(idx)
                        agg_mask = agg_mask.at[idx].set(0.0)
                    else:
                        survived_aggs.append(idx)
                elif float(agg_contribution[idx]) >= threshold or state['agg_memory_cells'][idx]:
                    survived_aggs.append(idx)
                else:
                    pruned_aggs.append(idx)
                    agg_mask = agg_mask.at[idx].set(0.0)
                del young_aggs[idx]

        # Ensure protected indices are present
        if float(act_mask[SIN_IDX]) < 0.5:
            act_mask = act_mask.at[SIN_IDX].set(1.0)
        for idx in [MAX_IDX, MIN_IDX]:
            if float(agg_mask[idx]) < 0.5:
                agg_mask = agg_mask.at[idx].set(1.0)

        # Update affinities on improvement
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]
            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
                act_contribution = act_contribution.at[a].add(0.15)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)
                agg_contribution = agg_contribution.at[g].add(0.15)

        # Clamp and apply floors
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)
        act_contribution = jnp.clip(act_contribution, 0.0, 1.0)
        agg_contribution = jnp.clip(agg_contribution, 0.0, 1.0)
        act_affinities, agg_affinities = self._apply_affinity_floors(act_affinities, agg_affinities)

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            act_affinities, state['act_memory_counts'], state['act_memory_cells'], act_mask
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            agg_affinities, state['agg_memory_counts'], state['agg_memory_cells'], agg_mask
        )

        # Get palettes and update discovery
        act_palette = mask_to_indices(act_mask)
        agg_palette = mask_to_indices(agg_mask)
        new_discovery = self._update_discovery_tracking(
            state['discovery_gen'], act_palette, agg_palette, generation
        )

        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))

        new_state = {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'cross_ltp_matrix': cross_ltp_matrix,
            'stdp_history': stdp_history,
            'young_acts': young_acts,
            'young_aggs': young_aggs,
            'act_contribution': act_contribution,
            'agg_contribution': agg_contribution,
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'discovery_gen': new_discovery,
            'cross_ltp_events': state['cross_ltp_events'] + ltp_events,
            'mutual_survival_events': state['mutual_survival_events'] + mutual_events,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'act_survived': len(survived_acts),
            'agg_survived': len(survived_aggs),
            'cross_ltp_events': ltp_events,
            'mutual_survival_events': mutual_events,
            'total_memory_cells': act_mem_count + agg_mem_count,
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'sin_affinity': float(act_affinities[SIN_IDX]),
            'sin_discovered_gen': new_discovery['sin'],
            'max_discovered_gen': new_discovery['max'],
            'min_discovered_gen': new_discovery['min'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        act_mem_count = int(jnp.sum(state['act_memory_cells']))
        agg_mem_count = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'cross_ltp_events': state['cross_ltp_events'],
            'mutual_survival_events': state['mutual_survival_events'],
            'discovery_gen': state['discovery_gen'],
            'generation': state['generation'],
        }
