"""Strategy 10 Symmetric: NeuroHebbian with Memory Cells and Affinity Floors.

Extends NeuroHebbianDual with winning patterns from eligibility_trace_symmetric:
- Neuromodulation (DA/ACh/NE) for exploration/exploitation control
- Hebbian learning for pairwise co-occurrence associations
- Memory cells from sustained high affinity
- Affinity floors for sin and extreme aggregations (CRITICAL for retention)

Biological Basis:
- Dopamine (DA): Reward signal, stabilizes successful patterns
- Acetylcholine (ACh): Uncertainty signal, promotes exploration when stuck
- Norepinephrine (NE): Arousal, modulates learning rate
- Hebbian: "Cells that fire together wire together" - pairwise associations

Key additions:
- Memory cells: Functions maintaining high affinity become permanent
- Affinity floors: Sin and extreme aggs never drop below threshold
- Discovery tracking: Track when critical functions are found
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

# Oscillatory function indices for special handling
ACT_OSCILLATORY = [4, 11, 12, 13, 15]  # sin, burst, resonator, osc_adapt, receptive


class NeuroHebbianSymmetricStrategy(PaletteEvolutionStrategy):
    """Neuromodulation + Hebbian learning with memory cells for dual palette.

    Combines the proven Hebbian pairwise learning with memory cells
    and affinity floors from the winning eligibility_trace strategy.

    Key innovations:
    - Hebbian weights encode which pairs of functions work well together
    - Neuromodulators control exploration vs exploitation balance
    - Memory cells protect proven valuable functions
    - Affinity floors ensure sin and extreme aggs are never lost
    """

    name = "neuro_hebbian_symmetric"
    description = "Neuromodulation + Hebbian with memory cells"

    def __init__(
        self,
        # Neuromodulation (shared across domains)
        base_activate_rate: float = 0.25,
        base_deactivate_rate: float = 0.05,
        dopamine_sensitivity: float = 0.5,
        acetylcholine_sensitivity: float = 0.3,
        norepinephrine_sensitivity: float = 0.2,
        modulation_ema_alpha: float = 0.3,
        stagnation_threshold: int = 3,
        sticky_deactivate_rate: float = 0.01,
        # Hebbian parameters
        act_learning_rate: float = 0.1,
        act_anti_hebbian_rate: float = 0.05,
        agg_learning_rate: float = 0.08,
        agg_anti_hebbian_rate: float = 0.04,
        cross_learning_rate: float = 0.05,
        consolidation_threshold: float = 0.7,
        consolidation_gens: int = 5,
        hebbian_influence: float = 0.5,
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
        # Constraints
        min_active_act: int = 2,
        max_active_act: int = 6,
        min_active_agg: int = 1,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize NeuroHebbian Symmetric strategy."""
        # Neuromodulation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.dopamine_sensitivity = dopamine_sensitivity
        self.acetylcholine_sensitivity = acetylcholine_sensitivity
        self.norepinephrine_sensitivity = norepinephrine_sensitivity
        self.modulation_ema_alpha = modulation_ema_alpha
        self.stagnation_threshold = stagnation_threshold
        self.sticky_deactivate_rate = sticky_deactivate_rate

        # Hebbian
        self.act_learning_rate = act_learning_rate
        self.act_anti_hebbian_rate = act_anti_hebbian_rate
        self.agg_learning_rate = agg_learning_rate
        self.agg_anti_hebbian_rate = agg_anti_hebbian_rate
        self.cross_learning_rate = cross_learning_rate
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_gens = consolidation_gens
        self.hebbian_influence = hebbian_influence

        # Memory cells
        self.memory_cell_affinity_threshold = memory_cell_affinity_threshold
        self.memory_cell_gens = memory_cell_gens
        self.memory_cell_decay_rate = memory_cell_decay_rate

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Discovery
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        # Constraints
        self.min_active_act = min_active_act
        self.max_active_act = max_active_act
        self.min_active_agg = min_active_agg
        self.max_active_agg = max_active_agg

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with Hebbian matrices and memory cell tracking."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Hebbian weight matrices
        act_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        agg_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        cross_weights = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Consolidation tracking (for Hebbian pairs)
        act_consol_counts = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.int32)
        agg_consol_counts = jnp.zeros((NUM_AGGREGATIONS, NUM_AGGREGATIONS), dtype=jnp.int32)
        cross_consol_counts = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS), dtype=jnp.int32)

        act_protected_pairs = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.bool_)
        agg_protected_pairs = jnp.zeros((NUM_AGGREGATIONS, NUM_AGGREGATIONS), dtype=jnp.bool_)
        cross_protected_pairs = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS), dtype=jnp.bool_)

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=bool)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=bool)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_weights': act_weights,
            'act_consol_counts': act_consol_counts,
            'act_protected_pairs': act_protected_pairs,
            'act_memory_counts': act_memory_counts,
            'act_memory_cells': act_memory_cells,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_weights': agg_weights,
            'agg_consol_counts': agg_consol_counts,
            'agg_protected_pairs': agg_protected_pairs,
            'agg_memory_counts': agg_memory_counts,
            'agg_memory_cells': agg_memory_cells,
            # Cross-domain
            'cross_weights': cross_weights,
            'cross_consol_counts': cross_consol_counts,
            'cross_protected_pairs': cross_protected_pairs,
            # Neuromodulators
            'dopamine': 0.5,
            'acetylcholine': 0.5,
            'norepinephrine': 1.0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 101010),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'fitness_history': [],
            'fitness_ema': 0.5,
            'strategy_name': self.name,
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
        state: Dict[str, Any],
        best_fitness: float,
        prev_best_fitness: float,
        generation: int,
    ) -> Dict[str, float]:
        """Update shared neuromodulator levels."""
        alpha = self.modulation_ema_alpha

        # Dopamine: reward signal
        improvement = best_fitness - prev_best_fitness
        rel_improvement = improvement / max(prev_best_fitness, 0.01)
        da_signal = max(0, min(1, 0.5 + rel_improvement * 10))
        new_da = (1 - alpha) * state['dopamine'] + alpha * da_signal

        # Acetylcholine: uncertainty/exploration signal
        stagnation = state['stagnation_count'] / max(self.stagnation_threshold, 1)
        ach_signal = min(1.0, stagnation)
        new_ach = (1 - alpha) * state['acetylcholine'] + alpha * ach_signal

        # Norepinephrine: arousal
        time_decay = max(0, 1.0 - generation / 50.0)
        challenge = 1.0 - best_fitness
        ne_signal = max(time_decay, challenge * 0.5)
        new_ne = (1 - alpha) * state['norepinephrine'] + alpha * ne_signal

        return {
            'dopamine': float(new_da),
            'acetylcholine': float(new_ach),
            'norepinephrine': float(new_ne)
        }

    def _compute_effective_rates(
        self,
        dopamine: float,
        acetylcholine: float,
        norepinephrine: float,
    ) -> Tuple[float, float]:
        """Compute neuromodulated mutation rates."""
        da_factor = 1.0 - self.dopamine_sensitivity * (dopamine - 0.5)
        ach_factor = 1.0 + self.acetylcholine_sensitivity * (acetylcholine - 0.5)
        ne_factor = 0.5 + self.norepinephrine_sensitivity * norepinephrine

        eff_activate = self.base_activate_rate * da_factor * ach_factor * ne_factor
        eff_deactivate = self.base_deactivate_rate * (1.0 / max(da_factor, 0.5)) * ne_factor

        return (
            max(0.05, min(0.5, eff_activate)),
            max(0.01, min(0.2, eff_deactivate))
        )

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        ne: float,
        lr: float,
        anti_lr: float,
    ) -> jnp.ndarray:
        """Apply Hebbian learning update."""
        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        ne_lr = lr * (0.5 + ne * 0.5)
        ne_anti = anti_lr * (0.5 + ne * 0.5)

        if fitness_signal >= 0:
            delta = ne_lr * fitness_signal * co_active
        else:
            delta = ne_anti * fitness_signal * co_active

        return jnp.clip(weights + delta, 0.0, 1.0)

    def _cross_hebbian_update(
        self,
        weights: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_signal: float,
        ne: float,
    ) -> jnp.ndarray:
        """Update cross-domain Hebbian weights."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        lr = self.cross_learning_rate * (0.5 + ne * 0.5)
        delta = lr * fitness_signal * co_active

        return jnp.clip(weights + delta, 0.0, 1.0)

    def _update_consolidation(
        self,
        weights: jnp.ndarray,
        counts: jnp.ndarray,
        protected: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update consolidation tracking for Hebbian pairs."""
        strong = weights >= self.consolidation_threshold
        new_counts = jnp.where(strong, counts + 1, 0)
        newly_protected = new_counts >= self.consolidation_gens
        new_protected = jnp.logical_or(protected, newly_protected)
        return new_counts, new_protected

    def _update_memory_cells(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
        dopamine: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cells based on sustained high affinity.

        Dopamine enhances memory formation.
        """
        active = mask > 0.5
        # High dopamine lowers the threshold (reward promotes memory)
        effective_threshold = self.memory_cell_affinity_threshold - 0.05 * (dopamine - 0.5)
        effective_threshold = max(0.6, min(0.85, effective_threshold))

        high_affinity = jnp.logical_and(affinity >= effective_threshold, active)

        new_counts = jnp.where(high_affinity, memory_counts + 1, 0)
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors to sin and extreme aggregations.

        CRITICAL for 100% retention.
        """
        # Sin activation floor
        new_act_affinity = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation floors
        new_agg_affinity = agg_affinity
        for idx in CORE_EXTREME_AGGS:
            new_agg_affinity = new_agg_affinity.at[idx].set(
                jnp.maximum(new_agg_affinity[idx], self.extreme_agg_affinity_floor)
            )

        return new_act_affinity, new_agg_affinity

    def _update_affinity_from_hebbian(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
        fitness_signal: float,
        ne: float,
        lr: float,
    ) -> jnp.ndarray:
        """Update affinity based on Hebbian weights and fitness."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        # Compute Hebbian score for each function
        hebbian_score = jnp.dot(weights, active) / n_active

        # Learning modulated by NE
        effective_lr = lr * (0.5 + ne * 0.5)

        new_affinity = affinity.copy()
        for i in range(len(affinity)):
            if float(active[i]) > 0.5:
                # Combine fitness signal with Hebbian score
                heb_factor = float(hebbian_score[i]) - 0.5  # Center around 0
                combined_signal = fitness_signal + 0.3 * heb_factor

                if combined_signal >= 0:
                    delta = effective_lr * combined_signal
                else:
                    # Memory cells resist negative changes
                    if bool(memory_cells[i]):
                        delta = effective_lr * 0.1 * combined_signal
                    else:
                        delta = effective_lr * 0.3 * combined_signal

                new_affinity = new_affinity.at[i].set(
                    max(0.05, min(0.95, float(new_affinity[i]) + delta))
                )

        return new_affinity

    def _compute_protection(
        self,
        affinity: jnp.ndarray,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
        protected_pairs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection score combining affinity, Hebbian, and memory."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        # Hebbian score
        hebbian_score = jnp.dot(weights, active) / n_active

        # Base protection from affinity and Hebbian
        protection = 0.5 * affinity + 0.3 * hebbian_score

        # Memory cell bonus
        memory_bonus = memory_cells.astype(jnp.float32) * 0.3
        protection = protection + memory_bonus

        # Pair protection bonus (any protected pair)
        has_protected_pair = jnp.any(protected_pairs, axis=1).astype(jnp.float32) * 0.2
        protection = protection + has_protected_pair

        return jnp.clip(protection, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        memory_cells: jnp.ndarray,
        eff_activate: float,
        eff_deactivate: float,
        dopamine: float,
        min_active: int,
        max_active: int,
        n_functions: int,
        is_oscillatory: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with memory cell and Hebbian protection."""
        k1, k2 = jax.random.split(key)
        act_probs = jax.random.uniform(k1, (n_functions,))
        deact_probs = jax.random.uniform(k2, (n_functions,))

        new_mask = mask.copy()
        activated = []
        deactivated = []
        current_active = int(jnp.sum(mask > 0.5))

        for i in range(n_functions):
            prot = float(protection[i])
            is_memory = bool(memory_cells[i])

            if mask[i] < 0.5:
                # Activate: skip if at max
                if current_active + len(activated) >= max_active:
                    continue
                rate = eff_activate * (1 + self.hebbian_influence * (prot - 0.5))
                rate = max(0.05, min(0.6, rate))
                if act_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Base deactivation rate
                if is_oscillatory and i in is_oscillatory:
                    base = self.sticky_deactivate_rate
                else:
                    base = eff_deactivate

                # Memory cells are highly protected
                if is_memory:
                    rate = base * 0.05
                elif prot > 0.6:
                    da_prot = dopamine * 0.5
                    rate = base * (1 - da_prot) * 0.3
                else:
                    da_prot = dopamine * 0.5
                    rate = base * (1 - da_prot) * (1 + self.hebbian_influence * (0.5 - prot))

                rate = max(0.005, min(0.3, rate))
                if deact_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum
        if jnp.sum(new_mask > 0.5) < min_active:
            return mask, {'activated': [], 'deactivated': []}

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _track_discoveries(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        generation: int,
        sin_discovered_gen: int,
        extreme_agg_discovered_gen: int,
    ) -> Tuple[int, int]:
        """Track when sin and extreme aggregations are discovered."""
        new_sin_gen = sin_discovered_gen
        new_extreme_gen = extreme_agg_discovered_gen

        if sin_discovered_gen < 0 and act_mask[SIN_IDX] > 0.5:
            new_sin_gen = generation

        has_extreme = any(agg_mask[idx] > 0.5 for idx in CORE_EXTREME_AGGS)
        if extreme_agg_discovered_gen < 0 and has_extreme:
            new_extreme_gen = generation

        return new_sin_gen, new_extreme_gen

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with NeuroHebbian dynamics and memory cells."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stag = 0
            new_best = best_fitness
        else:
            new_stag = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update neuromodulators
        neuro = self._update_neuromodulators(state, best_fitness, prev_best_fitness, generation)
        eff_activate, eff_deactivate = self._compute_effective_rates(
            neuro['dopamine'], neuro['acetylcholine'], neuro['norepinephrine']
        )

        # Fitness signal
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Hebbian updates
        new_act_weights = self._hebbian_update(
            state['act_weights'], state['act_mask'], fitness_signal,
            neuro['norepinephrine'], self.act_learning_rate, self.act_anti_hebbian_rate
        )
        new_agg_weights = self._hebbian_update(
            state['agg_weights'], state['agg_mask'], fitness_signal,
            neuro['norepinephrine'], self.agg_learning_rate, self.agg_anti_hebbian_rate
        )
        new_cross_weights = self._cross_hebbian_update(
            state['cross_weights'], state['act_mask'], state['agg_mask'],
            fitness_signal, neuro['norepinephrine']
        )

        # Consolidation updates
        new_act_consol, new_act_prot = self._update_consolidation(
            new_act_weights, state['act_consol_counts'], state['act_protected_pairs']
        )
        new_agg_consol, new_agg_prot = self._update_consolidation(
            new_agg_weights, state['agg_consol_counts'], state['agg_protected_pairs']
        )
        new_cross_consol, new_cross_prot = self._update_consolidation(
            new_cross_weights, state['cross_consol_counts'], state['cross_protected_pairs']
        )

        # Update affinities from Hebbian
        new_act_affinity = self._update_affinity_from_hebbian(
            state['act_affinity'], new_act_weights, state['act_mask'],
            state['act_memory_cells'], fitness_signal, neuro['norepinephrine'],
            self.act_learning_rate
        )
        new_agg_affinity = self._update_affinity_from_hebbian(
            state['agg_affinity'], new_agg_weights, state['agg_mask'],
            state['agg_memory_cells'], fitness_signal, neuro['norepinephrine'],
            self.agg_learning_rate
        )

        # Apply affinity floors
        new_act_affinity, new_agg_affinity = self._apply_affinity_floors(
            new_act_affinity, new_agg_affinity
        )

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_affinity, state['act_mask'],
            state['act_memory_counts'], state['act_memory_cells'],
            neuro['dopamine']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_affinity, state['agg_mask'],
            state['agg_memory_counts'], state['agg_memory_cells'],
            neuro['dopamine']
        )

        # Compute protection
        act_protection = self._compute_protection(
            new_act_affinity, new_act_weights, state['act_mask'],
            new_act_mem_cells, new_act_prot
        )
        agg_protection = self._compute_protection(
            new_agg_affinity, new_agg_weights, state['agg_mask'],
            new_agg_mem_cells, new_agg_prot
        )

        # Mutate palettes
        new_act_mask, act_info = self._mutate_palette(
            k1, state['act_mask'], act_protection, new_act_mem_cells,
            eff_activate, eff_deactivate, neuro['dopamine'],
            self.min_active_act, self.max_active_act, NUM_ACTIVATIONS,
            ACT_OSCILLATORY
        )
        new_agg_mask, agg_info = self._mutate_palette(
            k2, state['agg_mask'], agg_protection, new_agg_mem_cells,
            eff_activate * 0.8, eff_deactivate * 0.9, neuro['dopamine'],
            self.min_active_agg, self.max_active_agg, NUM_AGGREGATIONS
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Track discoveries
        new_sin_gen, new_extreme_gen = self._track_discoveries(
            new_act_mask, new_agg_mask, generation,
            state['sin_discovered_gen'], state['extreme_agg_discovered_gen']
        )

        new_state = {
            'act_mask': new_act_mask,
            'act_affinity': new_act_affinity,
            'act_weights': new_act_weights,
            'act_consol_counts': new_act_consol,
            'act_protected_pairs': new_act_prot,
            'act_memory_counts': new_act_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_affinity,
            'agg_weights': new_agg_weights,
            'agg_consol_counts': new_agg_consol,
            'agg_protected_pairs': new_agg_prot,
            'agg_memory_counts': new_agg_mem_counts,
            'agg_memory_cells': new_agg_mem_cells,
            'cross_weights': new_cross_weights,
            'cross_consol_counts': new_cross_consol,
            'cross_protected_pairs': new_cross_prot,
            'dopamine': neuro['dopamine'],
            'acetylcholine': neuro['acetylcholine'],
            'norepinephrine': neuro['norepinephrine'],
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stag,
            'best_fitness_seen': new_best,
            'fitness_history': (state['fitness_history'] + [best_fitness])[-10:],
            'fitness_ema': new_fitness_ema,
            'strategy_name': self.name,
            'sin_discovered_gen': new_sin_gen,
            'extreme_agg_discovered_gen': new_extreme_gen,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        n_act_mem = int(jnp.sum(new_act_mem_cells))
        n_agg_mem = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'act_palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_act_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stag,
            'fitness_improved': improved,
            # Neuromodulators
            'dopamine': neuro['dopamine'],
            'acetylcholine': neuro['acetylcholine'],
            'norepinephrine': neuro['norepinephrine'],
            # Memory cells
            'act_memory_cells': n_act_mem,
            'agg_memory_cells': n_agg_mem,
            'total_memory_cells': n_act_mem + n_agg_mem,
            # Hebbian
            'act_avg_weight': float(jnp.mean(new_act_weights)),
            'agg_avg_weight': float(jnp.mean(new_agg_weights)),
            'cross_avg_weight': float(jnp.mean(new_cross_weights)),
            'act_n_protected_pairs': int(jnp.sum(new_act_prot) / 2),
            'agg_n_protected_pairs': int(jnp.sum(new_agg_prot) / 2),
            'cross_n_protected': int(jnp.sum(new_cross_prot)),
            # Discovery
            'sin_discovered_gen': new_sin_gen,
            'extreme_agg_discovered_gen': new_extreme_gen,
            # Stats
            'sin_active': SIN_IDX in act_palette,
            'sin_affinity': float(new_act_affinity[SIN_IDX]),
            'has_extreme_agg': any(idx in agg_palette for idx in CORE_EXTREME_AGGS),
            'act_activated': act_info['activated'],
            'act_deactivated': act_info['deactivated'],
            'agg_activated': agg_info['activated'],
            'agg_deactivated': agg_info['deactivated'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with Hebbian and memory cell info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

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
            # Neuromodulators
            'dopamine': state['dopamine'],
            'acetylcholine': state['acetylcholine'],
            'norepinephrine': state['norepinephrine'],
            # Memory cells
            'act_memory_cells': n_act_mem,
            'agg_memory_cells': n_agg_mem,
            'total_memory_cells': n_act_mem + n_agg_mem,
            # Discovery
            'sin_discovered_gen': state['sin_discovered_gen'],
            'extreme_agg_discovered_gen': state['extreme_agg_discovered_gen'],
            # Hebbian
            'act_n_protected_pairs': int(jnp.sum(state['act_protected_pairs']) / 2),
            'agg_n_protected_pairs': int(jnp.sum(state['agg_protected_pairs']) / 2),
            'cross_n_protected': int(jnp.sum(state['cross_protected_pairs'])),
            # Affinities
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
        }
