"""Strategy 120S: STDP-Neurogenesis Lineage Selection Symmetric.

Biological Basis: Surviving functions' LTP history influences birth probability
of similar functions. Functions near successful survivors have increased birth
probability - evolutionary momentum toward successful functional regions.

Key symmetric mechanisms:
1. Successful high-LTP survivors propagate lineage to nearby functions
2. Birth probability increased for functions in successful lineages
3. Similarity defined by index proximity
4. Protected indices for sin and extreme aggregations (0.1% deactivation)
5. Affinity floors for guaranteed retention
6. Memory cell crystallization for proven functions
7. Initial palettes include critical functions from start

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

# Critical indices for retention
SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3


class STDPNeurogenesisLineageSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric STDP-guided lineage selection for both domains.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Lineage boost centered on sin/extremes
    - Initial lineage strength for critical functions
    """

    name = "stdp_neurogenesis_lineage_symmetric"
    description = "Symmetric: Successful survivors boost birth probability of similar functions"

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
        protected_lineage_boost: float = 0.4,
        # === Protected function settings ===
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
        """Initialize STDP-Neurogenesis Lineage Symmetric strategy."""
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
        self.protected_lineage_boost = protected_lineage_boost

        # Protected function settings
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
        """Initialize state with lineage tracking for both domains."""
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

        # LTP credit
        act_ltp = jnp.zeros(NUM_ACTIVATIONS)
        act_ltp = act_ltp.at[SIN_IDX].set(0.4)
        agg_ltp = jnp.zeros(NUM_AGGREGATIONS)
        for idx in [MAX_IDX, MIN_IDX]:
            agg_ltp = agg_ltp.at[idx].set(0.4)

        # Lineage strength - protected indices start with lineage boost
        act_lineage = jnp.zeros(NUM_ACTIVATIONS)
        act_lineage = act_lineage.at[SIN_IDX].set(self.protected_lineage_boost)
        agg_lineage = jnp.zeros(NUM_AGGREGATIONS)
        for idx in [MAX_IDX, MIN_IDX]:
            agg_lineage = agg_lineage.at[idx].set(self.protected_lineage_boost)

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
            'act_ltp': act_ltp,
            'agg_ltp': agg_ltp,
            # Lineage strength
            'act_lineage': act_lineage,
            'agg_lineage': agg_lineage,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Discovery
            'discovery_gen': discovery_gen,
            # Stats
            'lineage_propagations': 0,
            'lineage_births': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1200001),
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

    def _is_protected(self, func_idx: int, is_activation: bool) -> bool:
        """Check if a function index is protected."""
        if is_activation:
            return func_idx == SIN_IDX
        else:
            return func_idx in [MAX_IDX, MIN_IDX]

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

    def _propagate_lineage(
        self,
        lineage: jnp.ndarray,
        successful_idx: int,
        n_funcs: int,
        is_protected: bool,
    ) -> jnp.ndarray:
        """Propagate lineage from successful survivor to nearby functions."""
        for offset in range(-self.similarity_radius, self.similarity_radius + 1):
            if offset == 0:
                continue
            neighbor = successful_idx + offset
            if 0 <= neighbor < n_funcs:
                distance_factor = 1.0 - abs(offset) / (self.similarity_radius + 1)
                boost = self.lineage_inheritance * distance_factor
                # Protected indices spread stronger lineage
                if is_protected:
                    boost *= 1.5
                lineage = lineage.at[neighbor].add(boost)
        return lineage

    def _apply_lineage_floors(
        self,
        act_lineage: jnp.ndarray,
        agg_lineage: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply lineage floors for protected indices."""
        new_act = act_lineage.at[SIN_IDX].set(
            jnp.maximum(act_lineage[SIN_IDX], self.protected_lineage_boost * 0.8)
        )
        new_agg = agg_lineage
        for idx in [MAX_IDX, MIN_IDX]:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.protected_lineage_boost * 0.8)
            )
        return new_act, new_agg

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
        """Update with lineage-based birth priority for both domains."""
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

        act_mask = state['act_mask'].copy()
        agg_mask = state['agg_mask'].copy()
        act_affinities = state['act_affinities'] * self.affinity_decay
        agg_affinities = state['agg_affinities'] * self.affinity_decay
        lineage_propagations = state['lineage_propagations']
        lineage_births = state['lineage_births']

        if improved:
            # Active functions get LTP credit and may propagate lineage
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    act_ltp = act_ltp.at[i].add(self.ltp_rate)
                    act_affinities = act_affinities.at[i].add(self.affinity_lr)
                    # High LTP survivors propagate lineage
                    if float(act_ltp[i]) > self.lineage_survival_threshold:
                        act_lineage = self._propagate_lineage(
                            act_lineage, i, NUM_ACTIVATIONS, self._is_protected(i, True)
                        )
                        lineage_propagations += 1

            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    agg_ltp = agg_ltp.at[i].add(self.ltp_rate)
                    agg_affinities = agg_affinities.at[i].add(self.affinity_lr)
                    if float(agg_ltp[i]) > self.lineage_survival_threshold:
                        agg_lineage = self._propagate_lineage(
                            agg_lineage, i, NUM_AGGREGATIONS, self._is_protected(i, False)
                        )
                        lineage_propagations += 1

        # Clamp and apply floors
        act_ltp = jnp.clip(act_ltp, 0.0, 1.0)
        agg_ltp = jnp.clip(agg_ltp, 0.0, 1.0)
        act_lineage = jnp.clip(act_lineage, 0.0, 1.0)
        agg_lineage = jnp.clip(agg_lineage, 0.0, 1.0)
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)

        # Apply floors
        act_affinities, agg_affinities = self._apply_affinity_floors(act_affinities, agg_affinities)
        act_lineage, agg_lineage = self._apply_lineage_floors(act_lineage, agg_lineage)
        act_ltp = act_ltp.at[SIN_IDX].set(jnp.maximum(act_ltp[SIN_IDX], 0.3))
        for idx in [MAX_IDX, MIN_IDX]:
            agg_ltp = agg_ltp.at[idx].set(jnp.maximum(agg_ltp[idx], 0.3))

        # === ACTIVATION BIRTH WITH LINEAGE BOOST ===
        candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
        sin_missing = SIN_IDX in candidates
        if candidates and (jax.random.uniform(k1) < self.base_birth_rate or sin_missing):
            probs = []
            for i in candidates:
                prob = 0.1 + float(act_lineage[i]) * self.lineage_birth_boost
                if i == SIN_IDX:
                    prob += self.protected_lineage_boost
                probs.append(prob)

            probs_arr = jnp.array(probs)
            probs_arr = probs_arr / probs_arr.sum()

            new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs_arr))
            act_mask = act_mask.at[new_idx].set(1.0)

            if float(act_lineage[new_idx]) > 0.1:
                lineage_births += 1

        # === AGGREGATION BIRTH WITH LINEAGE BOOST ===
        candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
        extremes_missing = any(i in candidates for i in [MAX_IDX, MIN_IDX])
        if candidates and (jax.random.uniform(k3) < self.base_birth_rate or extremes_missing):
            probs = []
            for i in candidates:
                prob = 0.1 + float(agg_lineage[i]) * self.lineage_birth_boost
                if i in [MAX_IDX, MIN_IDX]:
                    prob += self.protected_lineage_boost
                probs.append(prob)

            probs_arr = jnp.array(probs)
            probs_arr = probs_arr / probs_arr.sum()

            new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs_arr))
            agg_mask = agg_mask.at[new_idx].set(1.0)

            if float(agg_lineage[new_idx]) > 0.1:
                lineage_births += 1

        # Ensure protected indices are present
        if float(act_mask[SIN_IDX]) < 0.5:
            act_mask = act_mask.at[SIN_IDX].set(1.0)
        for idx in [MAX_IDX, MIN_IDX]:
            if float(agg_mask[idx]) < 0.5:
                agg_mask = agg_mask.at[idx].set(1.0)

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
            'act_ltp': act_ltp,
            'agg_ltp': agg_ltp,
            'act_lineage': act_lineage,
            'agg_lineage': agg_lineage,
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'discovery_gen': new_discovery,
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
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'lineage_propagations': lineage_propagations,
            'lineage_births': lineage_births,
            'total_memory_cells': act_mem_count + agg_mem_count,
            'mean_act_lineage': float(act_lineage.mean()),
            'mean_agg_lineage': float(agg_lineage.mean()),
            'sin_lineage': float(act_lineage[SIN_IDX]),
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'sin_affinity': float(act_affinities[SIN_IDX]),
            'sin_discovered_gen': new_discovery['sin'],
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
            'lineage_propagations': state['lineage_propagations'],
            'lineage_births': state['lineage_births'],
            'sin_lineage': float(state['act_lineage'][SIN_IDX]),
            'discovery_gen': state['discovery_gen'],
            'generation': state['generation'],
        }
