"""Strategy 21 Symmetric: Multi-Neuromodulatory with Memory Cells and Affinity Floors.

Extends MultiNeuromodulatoryDual with winning patterns from eligibility_trace_symmetric:
- Four neuromodulators (ACh/DA/NE/5-HT) with biological interactions
- Memory cells from sustained high affinity
- Affinity floors for sin and extreme aggregations (CRITICAL for retention)
- Discovery tracking for both domains

Biological Basis:
- Acetylcholine (ACh): Attention, focus, precision of processing
- Dopamine (DA): Reward prediction, motivation, reinforcement learning
- Norepinephrine (NE): Arousal, urgency, fight-or-flight, exploration
- Serotonin (5-HT): Mood, patience, long-term stability, impulse control

Key neuromodulator interactions:
- ACh-DA synergy: Attention amplifies reward learning
- NE-5HT opposition: Urgency vs patience tradeoff
- DA from fitness: Reward prediction error drives learning
- 5-HT for stability: Protects valuable functions during stress

Additions:
- Memory cells: Functions maintaining high affinity for 8+ gens become permanent
- Affinity floors: Sin and extreme aggs never drop below threshold
- Discovery tracking: Track when new functions become valuable
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
)

# Critical function indices
SIN_IDX = 4
CORE_EXTREME_AGGS = [2, 3]  # max, min


class MultiNeuromodulatorySymmetricStrategy(PaletteEvolutionStrategy):
    """Four-neuromodulator system with memory cells for dual palette evolution.

    Combines the biological richness of four neuromodulators with
    the winning memory cell + affinity floor patterns from eligibility_trace.

    Key innovations:
    - Four interacting neuromodulators (not just dopamine)
    - Serotonin provides additional stability for memory cells
    - ACh-DA synergy boosts learning during focused attention
    - NE triggers exploration during stagnation
    - Memory cells become permanent via affinity thresholds
    """

    name = "multi_neuromodulatory_symmetric"
    description = "4-neuromodulator system with memory cells"

    def __init__(
        self,
        # Neuromodulator baseline levels
        ach_baseline: float = 0.5,
        da_baseline: float = 0.5,
        ne_baseline: float = 0.5,
        serotonin_baseline: float = 0.5,
        # Neuromodulator sensitivity
        ach_sensitivity: float = 0.3,
        da_sensitivity: float = 0.4,
        ne_sensitivity: float = 0.35,
        serotonin_sensitivity: float = 0.2,
        # Interaction weights
        ach_da_synergy: float = 0.2,
        ne_5ht_opposition: float = 0.3,
        da_to_ach: float = 0.15,
        serotonin_to_da: float = -0.1,
        # Neuromodulator decay rates
        ach_decay: float = 0.1,
        da_decay: float = 0.15,
        ne_decay: float = 0.12,
        serotonin_decay: float = 0.05,
        # Behavioral effects
        base_mutation_rate: float = 0.15,
        base_learning_rate: float = 0.12,
        base_retention_rate: float = 0.5,
        # Memory cell parameters
        memory_cell_affinity_threshold: float = 0.75,
        memory_cell_gens: int = 8,
        memory_cell_decay_rate: float = 0.05,
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Cross-domain learning
        cross_learning_rate: float = 0.10,
        cross_influence: float = 0.3,
        # Protection
        affinity_protection_threshold: float = 0.55,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Multi-Neuromodulatory Symmetric strategy."""
        # Neuromodulator baselines
        self.ach_baseline = ach_baseline
        self.da_baseline = da_baseline
        self.ne_baseline = ne_baseline
        self.serotonin_baseline = serotonin_baseline

        # Sensitivities
        self.ach_sensitivity = ach_sensitivity
        self.da_sensitivity = da_sensitivity
        self.ne_sensitivity = ne_sensitivity
        self.serotonin_sensitivity = serotonin_sensitivity

        # Interactions
        self.ach_da_synergy = ach_da_synergy
        self.ne_5ht_opposition = ne_5ht_opposition
        self.da_to_ach = da_to_ach
        self.serotonin_to_da = serotonin_to_da

        # Decay rates
        self.ach_decay = ach_decay
        self.da_decay = da_decay
        self.ne_decay = ne_decay
        self.serotonin_decay = serotonin_decay

        # Behavioral effects
        self.base_mutation_rate = base_mutation_rate
        self.base_learning_rate = base_learning_rate
        self.base_retention_rate = base_retention_rate

        # Memory cell parameters
        self.memory_cell_affinity_threshold = memory_cell_affinity_threshold
        self.memory_cell_gens = memory_cell_gens
        self.memory_cell_decay_rate = memory_cell_decay_rate

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Discovery
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neuromodulator system and memory cells."""
        # Activation palette
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        act_co_occurrence = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS))

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=bool)

        # Aggregation palette
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_co_occurrence = jnp.zeros((NUM_AGGREGATIONS, NUM_AGGREGATIONS))

        # Memory cell tracking for aggregations
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=bool)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation state
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_co_occurrence': act_co_occurrence,
            'act_memory_counts': act_memory_counts,
            'act_memory_cells': act_memory_cells,
            # Aggregation state
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_co_occurrence': agg_co_occurrence,
            'agg_memory_counts': agg_memory_counts,
            'agg_memory_cells': agg_memory_cells,
            # Cross-domain state
            'cross_affinity': cross_affinity,
            # Common state
            'rng_key': jax.random.PRNGKey(seed + 212121),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Neuromodulator levels (shared across domains)
            'acetylcholine': self.ach_baseline,
            'dopamine': self.da_baseline,
            'norepinephrine': self.ne_baseline,
            'serotonin': self.serotonin_baseline,
            # Tracking
            'fitness_history': [],
            'fitness_ema': 0.5,
            'reward_prediction': 0.0,
            # Discovery tracking
            'sin_discovered_gen': -1,
            'extreme_agg_discovered_gen': -1,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _update_neuromodulators(
        self,
        ach: float,
        da: float,
        ne: float,
        serotonin: float,
        fitness_signal: float,
        stagnation: int,
        improved: bool,
    ) -> Tuple[float, float, float, float, Dict]:
        """Update neuromodulator levels based on evolutionary state.

        This is the core neuromodulatory system with biological interactions.
        """
        # ACh: Increases with consistent improvement (focused attention)
        if improved:
            ach_delta = self.ach_sensitivity * 0.3
        else:
            ach_delta = -self.ach_sensitivity * 0.1

        # DA from ACh interaction (attention amplifies reward)
        ach_boost_to_da = self.da_to_ach * ach

        # DA: Reward prediction error (core learning signal)
        da_delta = self.da_sensitivity * fitness_signal + ach_boost_to_da
        da_delta += self.serotonin_to_da * serotonin  # 5-HT dampens DA

        # NE: Increases with stagnation (triggers exploration)
        if stagnation > 5:
            ne_delta = self.ne_sensitivity * 0.4 * (stagnation / 20)
        elif improved:
            ne_delta = -self.ne_sensitivity * 0.2
        else:
            ne_delta = 0.0

        # 5-HT: Long-term stability (patience, impulse control)
        if fitness_signal > 0.2:
            serotonin_delta = self.serotonin_sensitivity * 0.2
        elif fitness_signal < -0.2:
            serotonin_delta = -self.serotonin_sensitivity * 0.1
        else:
            serotonin_delta = 0.0

        # NE-5HT opposition (urgency vs patience tradeoff)
        ne_5ht_effect = self.ne_5ht_opposition * (ne - serotonin)
        ne_delta += ne_5ht_effect * 0.5
        serotonin_delta -= ne_5ht_effect * 0.5

        # ACh-DA synergy (attention × reward = enhanced learning)
        ach_da_effect = self.ach_da_synergy * ach * da
        da_delta += ach_da_effect * 0.3

        # Decay toward baseline
        ach_decay_delta = self.ach_decay * (self.ach_baseline - ach)
        da_decay_delta = self.da_decay * (self.da_baseline - da)
        ne_decay_delta = self.ne_decay * (self.ne_baseline - ne)
        serotonin_decay_delta = self.serotonin_decay * (self.serotonin_baseline - serotonin)

        # Compute new levels
        new_ach = max(0.1, min(0.9, ach + ach_delta + ach_decay_delta))
        new_da = max(0.1, min(0.9, da + da_delta + da_decay_delta))
        new_ne = max(0.1, min(0.9, ne + ne_delta + ne_decay_delta))
        new_serotonin = max(0.1, min(0.9, serotonin + serotonin_delta + serotonin_decay_delta))

        metrics = {
            'ach_delta': ach_delta,
            'da_delta': da_delta,
            'ne_delta': ne_delta,
            'serotonin_delta': serotonin_delta,
        }

        return new_ach, new_da, new_ne, new_serotonin, metrics

    def _compute_behavioral_modulation(
        self,
        ach: float,
        da: float,
        ne: float,
        serotonin: float,
    ) -> Tuple[float, float, float]:
        """Compute behavioral parameters from neuromodulator levels."""
        # Exploration rate: NE increases, 5-HT decreases
        exploration_rate = self.base_mutation_rate * (
            1.0 + 0.5 * (ne - 0.5) - 0.3 * (serotonin - 0.5)
        )
        exploration_rate = max(0.05, min(0.4, exploration_rate))

        # Learning rate: DA and ACh both increase
        learning_rate = self.base_learning_rate * (
            1.0 + 0.6 * (da - 0.5) + 0.3 * (ach - 0.5)
        )
        learning_rate = max(0.05, min(0.3, learning_rate))

        # Retention rate: 5-HT increases, NE decreases
        retention_rate = self.base_retention_rate * (
            1.0 + 0.4 * (serotonin - 0.5) - 0.2 * (ne - 0.5)
        )
        retention_rate = max(0.3, min(0.8, retention_rate))

        return exploration_rate, learning_rate, retention_rate

    def _update_memory_cells(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
        serotonin: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cells based on sustained high affinity.

        Memory cell formation enhanced by serotonin (stability).
        High serotonin → easier memory cell formation.
        """
        active = mask > 0.5
        # Serotonin lowers threshold (stability promotes memory)
        effective_threshold = self.memory_cell_affinity_threshold - 0.05 * (serotonin - 0.5)
        effective_threshold = max(0.6, min(0.85, effective_threshold))

        high_affinity = jnp.logical_and(affinity >= effective_threshold, active)

        # Increment counts for high-affinity active functions
        new_counts = jnp.where(high_affinity, memory_counts + 1, 0)

        # Functions become memory cells after sustained high affinity
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors to sin and extreme aggregations.

        This is CRITICAL for 100% retention.
        Sin and extreme aggregations never drop below their floors.
        """
        # Sin activation floor
        new_act_affinity = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation floors (max, min)
        new_agg_affinity = agg_affinity
        for idx in CORE_EXTREME_AGGS:
            new_agg_affinity = new_agg_affinity.at[idx].set(
                jnp.maximum(new_agg_affinity[idx], self.extreme_agg_affinity_floor)
            )

        return new_act_affinity, new_agg_affinity

    def _update_affinity_with_neuromodulators(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
        fitness_signal: float,
        learning_rate: float,
        ach: float,
        newly_discovered: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Update affinity using neuromodulator-gated learning.

        ACh modulates precision: high ACh → focused updates to active functions.
        Memory cells get protection from negative updates.
        """
        precision = 0.5 + 0.5 * ach
        active = (mask > 0.5).astype(jnp.float32)

        new_affinity = affinity.copy()
        for i in range(len(affinity)):
            if float(active[i]) > 0.5:
                # Positive fitness → full learning rate
                # Negative fitness → reduced learning rate
                if fitness_signal >= 0:
                    delta = learning_rate * precision * fitness_signal
                else:
                    # Memory cells resist negative changes
                    if bool(memory_cells[i]):
                        delta = learning_rate * 0.1 * precision * fitness_signal
                    else:
                        delta = learning_rate * 0.3 * precision * fitness_signal

                # Discovery boost for newly discovered functions
                if newly_discovered is not None and bool(newly_discovered[i]):
                    if fitness_signal > 0:
                        delta += self.discovery_boost

                new_affinity = new_affinity.at[i].set(
                    max(0.05, min(0.95, float(new_affinity[i]) + delta))
                )

        return new_affinity

    def _update_co_occurrence(
        self,
        co_occurrence: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        ach: float,
    ) -> jnp.ndarray:
        """Update co-occurrence matrix for pairwise function success."""
        if fitness_signal <= 0:
            return co_occurrence

        lr = 0.1 * (0.5 + 0.5 * ach)
        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)
        new_co = jnp.clip(co_occurrence + lr * fitness_signal * co_active, 0.0, 1.0)

        return new_co

    def _compute_protection(
        self,
        affinity: jnp.ndarray,
        co_occurrence: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        other_mask: jnp.ndarray,
        serotonin: float,
    ) -> jnp.ndarray:
        """Compute protection scores with memory cell bonus.

        Memory cells get significant protection boost.
        Cross-domain success also contributes to protection.
        """
        active = (mask > 0.5).astype(jnp.float32)
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        n_other = max(jnp.sum(other_active), 1)

        pairwise_score = jnp.dot(co_occurrence, active) / n_active
        cross_score = jnp.dot(cross_affinity, other_active) / n_other

        # Base protection
        protection = (
            0.50 * affinity +
            0.25 * pairwise_score +
            0.10 * cross_score * self.cross_influence
        )

        # Memory cell bonus
        memory_bonus = memory_cells.astype(jnp.float32) * 0.3
        protection = protection + memory_bonus

        # 5-HT boosts protection for high-affinity functions
        serotonin_boost = serotonin * 0.1
        protection = jnp.where(
            affinity > self.affinity_protection_threshold,
            protection + serotonin_boost,
            protection
        )

        return jnp.clip(protection, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        memory_cells: jnp.ndarray,
        exploration_rate: float,
        retention_rate: float,
        min_active: int,
        max_active: int,
        n_functions: int,
        protected_indices: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with memory cell protection.

        Args:
            protected_indices: Indices that should never be deactivated (e.g., sin, extreme aggs).
                              These get extremely low deactivation rates.
        """
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (n_functions,))
        deactivate_probs = jax.random.uniform(key2, (n_functions,))

        current_active = int(jnp.sum(mask > 0.5))
        protected_set = set(protected_indices) if protected_indices else set()

        for i in range(n_functions):
            prot = float(protection[i])
            is_memory = bool(memory_cells[i])
            is_protected = i in protected_set

            if mask[i] < 0.5:
                # Activate: skip if at max
                if current_active + len(activated) >= max_active:
                    continue
                effective_rate = exploration_rate * (0.5 + prot)
                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Protected indices (sin, extreme aggs) almost never deactivate
                if is_protected:
                    deact_rate = 0.001  # 0.1% chance - essentially never
                # Memory cells are highly protected
                elif is_memory:
                    deact_rate = exploration_rate * (1.0 - retention_rate) * 0.05
                elif prot >= self.affinity_protection_threshold:
                    deact_rate = exploration_rate * (1.0 - retention_rate) * 0.2
                else:
                    deact_rate = exploration_rate * (1.0 - retention_rate) * (1.0 - prot)

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum
        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _track_discoveries(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        generation: int,
        sin_discovered_gen: int,
        extreme_agg_discovered_gen: int,
    ) -> Tuple[int, int, jnp.ndarray, jnp.ndarray]:
        """Track when sin and extreme aggregations are discovered."""
        act_newly_discovered = jnp.zeros(NUM_ACTIVATIONS, dtype=bool)
        agg_newly_discovered = jnp.zeros(NUM_AGGREGATIONS, dtype=bool)

        new_sin_gen = sin_discovered_gen
        new_extreme_gen = extreme_agg_discovered_gen

        # Check sin discovery
        if sin_discovered_gen < 0 and act_mask[SIN_IDX] > 0.5:
            new_sin_gen = generation
            act_newly_discovered = act_newly_discovered.at[SIN_IDX].set(True)

        # Check extreme aggregation discovery
        has_extreme = any(agg_mask[idx] > 0.5 for idx in CORE_EXTREME_AGGS)
        if extreme_agg_discovered_gen < 0 and has_extreme:
            new_extreme_gen = generation
            for idx in CORE_EXTREME_AGGS:
                if agg_mask[idx] > 0.5:
                    agg_newly_discovered = agg_newly_discovered.at[idx].set(True)

        return new_sin_gen, new_extreme_gen, act_newly_discovered, agg_newly_discovered

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with four-neuromodulator system and memory cells."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Fitness signal (reward prediction error)
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Step 1: Update neuromodulators (shared across domains)
        new_ach, new_da, new_ne, new_serotonin, neuromod_metrics = self._update_neuromodulators(
            state['acetylcholine'],
            state['dopamine'],
            state['norepinephrine'],
            state['serotonin'],
            fitness_signal,
            new_stagnation,
            improved,
        )

        # Step 2: Compute behavioral modulation
        exploration_rate, learning_rate, retention_rate = self._compute_behavioral_modulation(
            new_ach, new_da, new_ne, new_serotonin
        )

        # Step 3: Track discoveries
        new_sin_gen, new_extreme_gen, act_newly_disc, agg_newly_disc = self._track_discoveries(
            state['act_mask'],
            state['agg_mask'],
            generation,
            state['sin_discovered_gen'],
            state['extreme_agg_discovered_gen'],
        )

        # Step 4: Update affinities (both domains)
        new_act_affinity = self._update_affinity_with_neuromodulators(
            state['act_affinity'],
            state['act_mask'],
            state['act_memory_cells'],
            fitness_signal,
            learning_rate,
            new_ach,
            act_newly_disc,
        )
        new_agg_affinity = self._update_affinity_with_neuromodulators(
            state['agg_affinity'],
            state['agg_mask'],
            state['agg_memory_cells'],
            fitness_signal,
            learning_rate,
            new_ach,
            agg_newly_disc,
        )

        # Step 5: Apply affinity floors (CRITICAL for retention)
        new_act_affinity, new_agg_affinity = self._apply_affinity_floors(
            new_act_affinity, new_agg_affinity
        )

        # Step 6: Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_affinity,
            state['act_mask'],
            state['act_memory_counts'],
            state['act_memory_cells'],
            new_serotonin,
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_affinity,
            state['agg_mask'],
            state['agg_memory_counts'],
            state['agg_memory_cells'],
            new_serotonin,
        )

        # Step 7: Update cross-domain affinity
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)
        if fitness_signal >= 0:
            cross_delta = self.cross_learning_rate * fitness_signal * cross_active
        else:
            cross_delta = self.cross_learning_rate * 0.3 * fitness_signal * cross_active
        new_cross_affinity = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Step 8: Update co-occurrence
        new_act_co = self._update_co_occurrence(
            state['act_co_occurrence'], state['act_mask'], fitness_signal, new_ach
        )
        new_agg_co = self._update_co_occurrence(
            state['agg_co_occurrence'], state['agg_mask'], fitness_signal, new_ach
        )

        # Step 9: Compute protection
        act_protection = self._compute_protection(
            new_act_affinity, new_act_co, state['act_mask'], new_act_mem_cells,
            new_cross_affinity, state['agg_mask'], new_serotonin
        )
        agg_protection = self._compute_protection(
            new_agg_affinity, new_agg_co, state['agg_mask'], new_agg_mem_cells,
            new_cross_affinity.T, state['act_mask'], new_serotonin
        )

        # Step 10: Apply mutations (with protected indices for sin and extreme aggs)
        new_act_mask, act_mutation = self._mutate_palette(
            key_act, state['act_mask'], act_protection, new_act_mem_cells,
            exploration_rate, retention_rate,
            self.min_active_act, self.max_active_act, NUM_ACTIVATIONS,
            protected_indices=[SIN_IDX],  # Sin is protected
        )
        new_agg_mask, agg_mutation = self._mutate_palette(
            key_agg, state['agg_mask'], agg_protection, new_agg_mem_cells,
            exploration_rate, retention_rate,
            self.min_active_agg, self.max_active_agg, NUM_AGGREGATIONS,
            protected_indices=CORE_EXTREME_AGGS,  # max, min are protected
        )

        act_palette_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_palette_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Update fitness history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_affinity': new_act_affinity,
            'act_co_occurrence': new_act_co,
            'act_memory_counts': new_act_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_affinity,
            'agg_co_occurrence': new_agg_co,
            'agg_memory_counts': new_agg_mem_counts,
            'agg_memory_cells': new_agg_mem_cells,
            'cross_affinity': new_cross_affinity,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'acetylcholine': new_ach,
            'dopamine': new_da,
            'norepinephrine': new_ne,
            'serotonin': new_serotonin,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'reward_prediction': new_fitness_ema,
            'sin_discovered_gen': new_sin_gen,
            'extreme_agg_discovered_gen': new_extreme_gen,
        }

        active_act_palette = mask_to_indices(new_act_mask)
        active_agg_palette = mask_to_indices(new_agg_mask)

        # Count memory cells
        n_act_mem = int(jnp.sum(new_act_mem_cells))
        n_agg_mem = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'act_palette_changed': act_palette_changed,
            'agg_palette_changed': agg_palette_changed,
            'current_act_palette': active_act_palette,
            'current_agg_palette': active_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neuromodulators
            'acetylcholine': new_ach,
            'dopamine': new_da,
            'norepinephrine': new_ne,
            'serotonin': new_serotonin,
            # Behavioral
            'exploration_rate': exploration_rate,
            'learning_rate': learning_rate,
            'retention_rate': retention_rate,
            # Memory cells
            'act_memory_cells': n_act_mem,
            'agg_memory_cells': n_agg_mem,
            'total_memory_cells': n_act_mem + n_agg_mem,
            # Discovery
            'sin_discovered_gen': new_sin_gen,
            'extreme_agg_discovered_gen': new_extreme_gen,
            # Activation stats
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'act_max_affinity': float(jnp.max(new_act_affinity)),
            'sin_affinity': float(new_act_affinity[SIN_IDX]),
            'sin_active': SIN_IDX in active_act_palette,
            # Aggregation stats
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'agg_max_affinity': float(jnp.max(new_agg_affinity)),
            'has_extreme_agg': any(idx in active_agg_palette for idx in CORE_EXTREME_AGGS),
            # Cross-domain stats
            'cross_avg_affinity': float(jnp.mean(new_cross_affinity)),
            'cross_max_affinity': float(jnp.max(new_cross_affinity)),
            # Mutations
            'act_activated': act_mutation['activated'],
            'act_deactivated': act_mutation['deactivated'],
            'agg_activated': agg_mutation['activated'],
            'agg_deactivated': agg_mutation['deactivated'],
        }
        metrics.update({f'neuromod_{k}': v for k, v in neuromod_metrics.items()})

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with neuromodulator and memory cell info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        exploration, learning, retention = self._compute_behavioral_modulation(
            state['acetylcholine'],
            state['dopamine'],
            state['norepinephrine'],
            state['serotonin'],
        )

        n_act_mem = int(jnp.sum(state['act_memory_cells']))
        n_agg_mem = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'act_palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': SIN_IDX in act_palette,
            'has_extreme_agg': any(idx in agg_palette for idx in CORE_EXTREME_AGGS),
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Neuromodulator state
            'acetylcholine': state['acetylcholine'],
            'dopamine': state['dopamine'],
            'norepinephrine': state['norepinephrine'],
            'serotonin': state['serotonin'],
            # Behavioral parameters
            'exploration_rate': exploration,
            'learning_rate': learning,
            'retention_rate': retention,
            # Memory cells
            'act_memory_cells': n_act_mem,
            'agg_memory_cells': n_agg_mem,
            'total_memory_cells': n_act_mem + n_agg_mem,
            # Discovery
            'sin_discovered_gen': state['sin_discovered_gen'],
            'extreme_agg_discovered_gen': state['extreme_agg_discovered_gen'],
            # Affinities
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
        }
