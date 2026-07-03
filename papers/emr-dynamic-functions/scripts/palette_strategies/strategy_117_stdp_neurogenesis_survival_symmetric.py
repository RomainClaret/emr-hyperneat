"""Strategy 117S: STDP-Neurogenesis Guided Survival Symmetric.

Biological Basis: Combines STDP temporal credit with neurogenesis survival mechanics.
Young functions active BEFORE fitness improvement get survival credit boost.
Functions that PREDICT success survive maturation more easily.

Key symmetric mechanisms:
1. STDP LTP window tracks which functions PRECEDE improvement
2. Young functions with high LTP credit have lowered survival threshold
3. Cross-domain LTP for sin-extreme pairings
4. Protected indices for sin and extreme aggregations (0.1% deactivation)
5. Affinity floors for guaranteed retention
6. Memory cell crystallization for proven functions
7. Initial palettes include critical functions from start

Expected: More reliable sin-extreme pairing through temporal survival credit.
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


class STDPNeurogenesisSurvivalSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric STDP-guided neurogenesis survival for both domains.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations (0.1% deactivation)
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Initial palettes include sin and max/min from start
    - LTP credit floors for protected indices
    """

    name = "stdp_neurogenesis_survival_symmetric"
    description = "Symmetric: STDP temporal credit guides neurogenesis survival in both domains"

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
        neurogenesis_rate: float = 0.10,
        maturation_period: int = 8,
        young_plasticity: float = 2.0,
        base_survival_threshold: float = 0.15,
        max_young_act: int = 2,
        max_young_agg: int = 2,
        # === STDP-SURVIVAL INTEGRATION ===
        stdp_survival_multiplier: float = 0.7,
        young_ltp_plasticity: float = 2.5,
        ltp_survival_threshold: float = 0.4,
        sin_extreme_ltp_boost: float = 0.5,
        # === Protected function settings ===
        protected_survival_discount: float = 0.5,
        protected_deactivation_prob: float = 0.001,
        ltp_credit_floor: float = 0.3,
        # === Affinity floors ===
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # === Memory cell parameters ===
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # === Cross-domain ===
        cross_learning_rate: float = 0.05,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        max_stable_act: int = 6,
        max_stable_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize STDP-Neurogenesis Survival Symmetric strategy."""
        # STDP parameters
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.stdp_history_length = stdp_history_length
        self.temporal_decay = temporal_decay

        # Neurogenesis parameters
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.young_plasticity = young_plasticity
        self.base_survival_threshold = base_survival_threshold
        self.max_young_act = max_young_act
        self.max_young_agg = max_young_agg

        # STDP-survival integration
        self.stdp_survival_multiplier = stdp_survival_multiplier
        self.young_ltp_plasticity = young_ltp_plasticity
        self.ltp_survival_threshold = ltp_survival_threshold
        self.sin_extreme_ltp_boost = sin_extreme_ltp_boost

        # Protected function settings
        self.protected_survival_discount = protected_survival_discount
        self.protected_deactivation_prob = protected_deactivation_prob
        self.ltp_credit_floor = ltp_credit_floor

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Memory cells
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.max_stable_act = max_stable_act
        self.max_stable_agg = max_stable_agg

        # CRITICAL: Include sin and extreme aggregations in initial palettes
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with STDP + neurogenesis tracking for both domains."""
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

        # Initialize affinities with floors for protected indices
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

        # LTP credit - protected indices start with floor
        act_ltp_credit = jnp.zeros(NUM_ACTIVATIONS)
        act_ltp_credit = act_ltp_credit.at[SIN_IDX].set(self.ltp_credit_floor)
        agg_ltp_credit = jnp.zeros(NUM_AGGREGATIONS)
        for idx in [MAX_IDX, MIN_IDX]:
            agg_ltp_credit = agg_ltp_credit.at[idx].set(self.ltp_credit_floor)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Neurogenesis state - stable sets include protected indices
        stable_acts: Set[int] = set(initial_act)
        stable_aggs: Set[int] = set(initial_agg)
        young_acts: Dict[int, Dict] = {}
        young_aggs: Dict[int, Dict] = {}

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
            # LTP credit
            'act_ltp_credit': act_ltp_credit,
            'agg_ltp_credit': agg_ltp_credit,
            'stdp_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Neurogenesis state
            'stable_acts': stable_acts,
            'stable_aggs': stable_aggs,
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
            'total_act_births': 0,
            'total_agg_births': 0,
            'total_act_survivals': 0,
            'total_agg_survivals': 0,
            'ltp_survival_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1170001),
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

    def _apply_ltp_floors(
        self,
        act_ltp: jnp.ndarray,
        agg_ltp: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply LTP credit floors for protected indices."""
        new_act = act_ltp.at[SIN_IDX].set(
            jnp.maximum(act_ltp[SIN_IDX], self.ltp_credit_floor)
        )
        new_agg = agg_ltp
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.ltp_credit_floor)
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

    def _compute_temporal_weight(self, gens_from_improvement: int) -> float:
        """Compute STDP temporal weight - closer = stronger."""
        return self.temporal_decay ** abs(gens_from_improvement)

    def _update_ltp_credit(
        self,
        act_ltp_credit: jnp.ndarray,
        agg_ltp_credit: jnp.ndarray,
        stdp_history: List[Tuple],
        current_gen: int,
        improved: bool,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        young_acts: Dict[int, Dict],
        young_aggs: Dict[int, Dict],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[int, Dict], Dict[int, Dict]]:
        """Update STDP LTP credit based on temporal causality."""
        new_act_ltp = act_ltp_credit * 0.95
        new_agg_ltp = agg_ltp_credit * 0.95
        new_young_acts = {k: dict(v) for k, v in young_acts.items()}
        new_young_aggs = {k: dict(v) for k, v in young_aggs.items()}

        if improved and len(stdp_history) >= 2:
            for hist_gen, hist_act_mask, hist_agg_mask, _ in stdp_history:
                gens_before = current_gen - hist_gen

                if 1 <= gens_before <= self.ltp_window:
                    temporal_weight = self._compute_temporal_weight(gens_before)
                    ltp_delta = self.ltp_rate * temporal_weight

                    # LTP for activations
                    for i in range(NUM_ACTIVATIONS):
                        if hist_act_mask[i] > 0.5:
                            bonus = self.young_ltp_plasticity if i in new_young_acts else 1.0
                            new_act_ltp = new_act_ltp.at[i].set(
                                min(1.0, new_act_ltp[i] + ltp_delta * bonus)
                            )
                            if i in new_young_acts:
                                prev = new_young_acts[i].get('ltp_accumulated', 0.0)
                                new_young_acts[i]['ltp_accumulated'] = prev + ltp_delta * bonus

                    # LTP for aggregations
                    for j in range(NUM_AGGREGATIONS):
                        if hist_agg_mask[j] > 0.5:
                            bonus = self.young_ltp_plasticity if j in new_young_aggs else 1.0
                            new_agg_ltp = new_agg_ltp.at[j].set(
                                min(1.0, new_agg_ltp[j] + ltp_delta * bonus)
                            )
                            if j in new_young_aggs:
                                prev = new_young_aggs[j].get('ltp_accumulated', 0.0)
                                new_young_aggs[j]['ltp_accumulated'] = prev + ltp_delta * bonus

                    # Sin-extreme co-activation bonus
                    sin_active = hist_act_mask[SIN_IDX] > 0.5
                    for j in [MAX_IDX, MIN_IDX]:
                        if sin_active and hist_agg_mask[j] > 0.5:
                            boost = self.sin_extreme_ltp_boost * temporal_weight
                            new_act_ltp = new_act_ltp.at[SIN_IDX].set(
                                min(1.0, new_act_ltp[SIN_IDX] + boost)
                            )
                            new_agg_ltp = new_agg_ltp.at[j].set(
                                min(1.0, new_agg_ltp[j] + boost)
                            )

        # LTD for stagnation
        if not improved:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5 and not self._is_protected(i, True):
                    new_act_ltp = new_act_ltp.at[i].set(
                        max(0, new_act_ltp[i] - self.ltd_rate * 0.3)
                    )
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5 and not self._is_protected(j, False):
                    new_agg_ltp = new_agg_ltp.at[j].set(
                        max(0, new_agg_ltp[j] - self.ltd_rate * 0.3)
                    )

        return new_act_ltp, new_agg_ltp, new_young_acts, new_young_aggs

    def _maybe_birth(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        key: jax.random.PRNGKey,
        generation: int,
        max_young: int,
        n_funcs: int,
        protected_indices: List[int],
    ) -> Tuple[Set[int], Dict[int, Dict], Optional[int]]:
        """Possibly birth a new function, preferring protected indices."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_young = dict(young)
        born = None

        if len(new_young) >= max_young:
            return stable, new_young, born

        if jax.random.uniform(key1) < self.neurogenesis_rate:
            # Priority: protected indices not yet in palette
            priority_available = [
                i for i in protected_indices
                if i not in stable and i not in new_young and i < n_funcs
            ]

            if priority_available and jax.random.uniform(key3) < 0.7:
                idx = int(jax.random.randint(key2, (), 0, len(priority_available)))
                new_func = priority_available[idx]
                new_young[new_func] = {'birth_gen': generation, 'ltp_accumulated': 0.3}
                born = new_func
            else:
                available = [
                    i for i in range(n_funcs)
                    if i not in stable and i not in new_young
                ]
                if available:
                    idx = int(jax.random.randint(key2, (), 0, len(available)))
                    new_func = available[idx]
                    new_young[new_func] = {'birth_gen': generation, 'ltp_accumulated': 0.0}
                    born = new_func

        return stable, new_young, born

    def _mature_with_stdp(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        contribution: jnp.ndarray,
        ltp_credit: jnp.ndarray,
        memory_cells: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        other_mask: jnp.ndarray,
        generation: int,
        max_stable: int,
        is_activation: bool,
        key: jax.random.PRNGKey,
    ) -> Tuple[Set[int], Dict[int, Dict], List[int], List[int], int]:
        """Process maturation with STDP-guided survival."""
        new_stable = set(stable)
        new_young = {}
        survived = []
        pruned = []
        ltp_survival_count = 0

        keys = jax.random.split(key, len(young) + 1)
        key_idx = 0

        for func_idx, info in young.items():
            age = generation - info['birth_gen']

            if age >= self.maturation_period:
                base_contribution = float(contribution[func_idx])
                survival_threshold = self.base_survival_threshold

                # LTP credit lowers threshold
                ltp_credit_val = float(ltp_credit[func_idx])
                accumulated_ltp = info.get('ltp_accumulated', 0.0)
                total_ltp = (ltp_credit_val + accumulated_ltp) / 2

                if total_ltp > self.ltp_survival_threshold:
                    survival_threshold *= self.stdp_survival_multiplier
                    ltp_survival_count += 1

                # Cross-domain boost for sin-extreme
                if is_activation and func_idx == SIN_IDX:
                    for j in [MAX_IDX, MIN_IDX]:
                        if other_mask[j] > 0.5:
                            cross_aff = float(cross_affinity[SIN_IDX, j])
                            if cross_aff > 0.5:
                                survival_threshold *= 0.8
                                base_contribution += cross_aff * 0.1
                elif not is_activation and func_idx in [MAX_IDX, MIN_IDX]:
                    if other_mask[SIN_IDX] > 0.5:
                        cross_aff = float(cross_affinity[SIN_IDX, func_idx])
                        if cross_aff > 0.5:
                            survival_threshold *= 0.8
                            base_contribution += cross_aff * 0.1

                # Protected indices get additional discount
                if self._is_protected(func_idx, is_activation):
                    survival_threshold *= self.protected_survival_discount

                # Memory cells always survive
                if memory_cells[func_idx]:
                    if len(new_stable) < max_stable:
                        new_stable.add(func_idx)
                        survived.append(func_idx)
                    else:
                        new_young[func_idx] = info
                # Protected indices only pruned with very low probability
                elif self._is_protected(func_idx, is_activation):
                    if jax.random.uniform(keys[key_idx]) < self.protected_deactivation_prob:
                        pruned.append(func_idx)
                    else:
                        if len(new_stable) < max_stable:
                            new_stable.add(func_idx)
                            survived.append(func_idx)
                        else:
                            new_young[func_idx] = info
                    key_idx += 1
                # Regular maturation
                elif base_contribution >= survival_threshold:
                    if len(new_stable) < max_stable:
                        new_stable.add(func_idx)
                        survived.append(func_idx)
                    else:
                        pruned.append(func_idx)
                else:
                    pruned.append(func_idx)
            else:
                new_young[func_idx] = info

        return new_stable, new_young, survived, pruned, ltp_survival_count

    def _update_contribution(
        self,
        contribution: jnp.ndarray,
        mask: jnp.ndarray,
        young: Dict[int, Dict],
        improved: bool,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update contribution tracking."""
        new_contribution = contribution * 0.9

        for i in range(n_funcs):
            if mask[i] > 0.5:
                current = float(new_contribution[i])
                if improved:
                    boost = 0.25
                    if i in young:
                        boost *= self.young_plasticity
                    new_contribution = new_contribution.at[i].set(current + boost)
                else:
                    new_contribution = new_contribution.at[i].set(current + 0.01)

        return jnp.clip(new_contribution, 0, 2.0)

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        delta = self.cross_learning_rate * fitness_delta * co_active
        return jnp.clip(cross_affinity + delta, 0.0, 1.0)

    def _create_mask(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Create mask from stable and young."""
        mask = jnp.zeros(n_funcs)
        for i in stable:
            if 0 <= i < n_funcs:
                mask = mask.at[i].set(1.0)
        for i in young.keys():
            if 0 <= i < n_funcs:
                mask = mask.at[i].set(1.0)
        return mask

    def _ensure_protected_in_stable(
        self,
        stable: Set[int],
        protected_indices: List[int],
        n_funcs: int,
    ) -> Set[int]:
        """Ensure protected indices are in stable set."""
        new_stable = set(stable)
        for idx in protected_indices:
            if idx < n_funcs:
                new_stable.add(idx)
        return new_stable

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
        """Update with STDP-guided neurogenesis for both domains."""
        key, k_act, k_agg, k_mat_act, k_mat_agg = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update STDP history
        stdp_history = state['stdp_history'] + [
            (generation, state['act_mask'].copy(), state['agg_mask'].copy(), best_fitness)
        ]
        if len(stdp_history) > self.stdp_history_length:
            stdp_history = stdp_history[-self.stdp_history_length:]

        # Update LTP credit
        new_act_ltp, new_agg_ltp, young_acts, young_aggs = self._update_ltp_credit(
            state['act_ltp_credit'], state['agg_ltp_credit'],
            stdp_history, generation, improved,
            state['act_mask'], state['agg_mask'],
            state['young_acts'], state['young_aggs']
        )

        # Apply LTP floors for protected indices
        new_act_ltp, new_agg_ltp = self._apply_ltp_floors(new_act_ltp, new_agg_ltp)

        # Update contributions
        new_act_contrib = self._update_contribution(
            state['act_contribution'], state['act_mask'], young_acts, improved, NUM_ACTIVATIONS
        )
        new_agg_contrib = self._update_contribution(
            state['agg_contribution'], state['agg_mask'], young_aggs, improved, NUM_AGGREGATIONS
        )

        # Update affinities
        new_act_aff = state['act_affinities'] * 0.98
        new_agg_aff = state['agg_affinities'] * 0.98
        if improved:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    new_act_aff = new_act_aff.at[i].add(0.1)
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].add(0.1)
        new_act_aff = jnp.clip(new_act_aff, 0, 1)
        new_agg_aff = jnp.clip(new_agg_aff, 0, 1)

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improvement
        )

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_aff, state['act_memory_counts'],
            state['act_memory_cells'], state['act_mask']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_aff, state['agg_memory_counts'],
            state['agg_memory_cells'], state['agg_mask']
        )

        # Maturation with STDP
        stable_acts = set(state['stable_acts'])
        stable_aggs = set(state['stable_aggs'])

        stable_acts, young_acts, act_survived, act_pruned, act_ltp_survive = self._mature_with_stdp(
            stable_acts, young_acts, new_act_contrib, new_act_ltp,
            new_act_mem_cells, new_cross, state['agg_mask'],
            generation, self.max_stable_act, True, k_mat_act
        )
        stable_aggs, young_aggs, agg_survived, agg_pruned, agg_ltp_survive = self._mature_with_stdp(
            stable_aggs, young_aggs, new_agg_contrib, new_agg_ltp,
            new_agg_mem_cells, new_cross, state['act_mask'],
            generation, self.max_stable_agg, False, k_mat_agg
        )

        # Ensure protected indices in stable
        stable_acts = self._ensure_protected_in_stable(stable_acts, [SIN_IDX], NUM_ACTIVATIONS)
        stable_aggs = self._ensure_protected_in_stable(stable_aggs, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS)

        # Maybe birth new functions
        stable_acts, young_acts, act_born = self._maybe_birth(
            stable_acts, young_acts, k_act, generation,
            self.max_young_act, NUM_ACTIVATIONS, [SIN_IDX]
        )
        stable_aggs, young_aggs, agg_born = self._maybe_birth(
            stable_aggs, young_aggs, k_agg, generation,
            self.max_young_agg, NUM_AGGREGATIONS, [MAX_IDX, MIN_IDX]
        )

        # Create masks
        new_act_mask = self._create_mask(stable_acts, young_acts, NUM_ACTIVATIONS)
        new_agg_mask = self._create_mask(stable_aggs, young_aggs, NUM_AGGREGATIONS)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Get palettes and update discovery
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)
        new_discovery = self._update_discovery_tracking(
            state['discovery_gen'], act_palette, agg_palette, generation
        )

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_ltp_credit': new_act_ltp,
            'agg_ltp_credit': new_agg_ltp,
            'stdp_history': stdp_history,
            'cross_affinity': new_cross,
            'stable_acts': stable_acts,
            'stable_aggs': stable_aggs,
            'young_acts': young_acts,
            'young_aggs': young_aggs,
            'act_contribution': new_act_contrib,
            'agg_contribution': new_agg_contrib,
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'discovery_gen': new_discovery,
            'total_act_births': state['total_act_births'] + (1 if act_born else 0),
            'total_agg_births': state['total_agg_births'] + (1 if agg_born else 0),
            'total_act_survivals': state['total_act_survivals'] + len(act_survived),
            'total_agg_survivals': state['total_agg_survivals'] + len(agg_survived),
            'ltp_survival_events': state['ltp_survival_events'] + act_ltp_survive + agg_ltp_survive,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neurogenesis
            'n_stable_acts': len(stable_acts),
            'n_young_acts': len(young_acts),
            'n_stable_aggs': len(stable_aggs),
            'n_young_aggs': len(young_aggs),
            'act_born': act_born,
            'agg_born': agg_born,
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # LTP metrics
            'ltp_survival_events': new_state['ltp_survival_events'],
            'sin_ltp_credit': float(new_act_ltp[SIN_IDX]),
            'max_ltp_credit': float(new_agg_ltp[MAX_IDX]),
            'min_ltp_credit': float(new_agg_ltp[MIN_IDX]),
            # Status
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'max_affinity': float(new_agg_aff[MAX_IDX]),
            # Discovery
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
            'n_stable_acts': len(state['stable_acts']),
            'n_young_acts': len(state['young_acts']),
            'n_stable_aggs': len(state['stable_aggs']),
            'n_young_aggs': len(state['young_aggs']),
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'ltp_survival_events': state['ltp_survival_events'],
            'sin_ltp_credit': float(state['act_ltp_credit'][SIN_IDX]),
            'discovery_gen': state['discovery_gen'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
