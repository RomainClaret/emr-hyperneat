"""Strategy 16 Symmetric: STDP.

Extends STDPStrategy with symmetric discovery features:
- Dual STDP tracking (separate for activation and aggregation)
- Cross-domain timing correlation (activation timing informs aggregation)
- Memory cells from sustained positive timing correlation
- Affinity floors and discovery tracking for both domains

Key mechanisms:
1. STDP temporal credit assignment: Pre-before-post = strengthen (LTP)
2. Dual timing scores per domain
3. Cross-domain timing: Act success → Agg timing boost
4. Memory cells for functions with sustained positive timing correlation

Biological rationale:
- Spike-timing-dependent plasticity captures causality
- Functions active BEFORE improvement caused that improvement
- Cross-modal timing: Visual cortex timing predicts motor responses
- Memory cells encode proven causal relationships
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque
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


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class STDPSymmetricStrategy(PaletteEvolutionStrategy):
    """STDP-based symmetric palette evolution with temporal credit assignment.

    Both activation and aggregation functions are learned through temporal
    correlation - functions active BEFORE fitness improvements get boosted.

    Key innovations:
    - Dual STDP per domain (activation and aggregation)
    - Cross-domain timing (success in one domain influences timing in other)
    - Memory cells from sustained positive timing correlation
    - Affinity floors prevent loss of critical functions
    """

    name = "stdp_symmetric"
    description = "Dual STDP with temporal credit assignment and memory cells"

    def __init__(
        self,
        # Critical period timing
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # Phase-specific base rates
        exploration_activate: float = 0.30,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate_max: float = 0.12,
        confirmation_deactivate_min: float = 0.01,
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # STDP window parameters
        ltp_window: int = 5,              # Generations before improvement to boost
        ltd_window: int = 3,              # Generations after for depression
        history_length: int = 10,         # Total history to maintain
        # STDP learning rates
        ltp_rate: float = 0.20,           # LTP strengthening rate
        ltd_rate: float = 0.08,           # LTD weakening rate
        # Temporal weighting (closer to improvement = stronger effect)
        temporal_decay: float = 0.7,      # Decay factor per generation from improvement
        # Cross-domain timing
        cross_timing_lr: float = 0.10,    # How much one domain's timing affects other
        cross_timing_weight: float = 0.25,
        # Memory cell parameters (sustained positive timing → memory)
        memory_cell_timing_threshold: float = 0.70,  # Timing score threshold
        memory_cell_gens: int = 8,        # Reduced from 10 for faster formation
        memory_cell_decay_rate: float = 0.05,
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Protection threshold
        affinity_protection_threshold: float = 0.55,
        # Phase modulation
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Other
        early_consolidation_threshold: float = 0.95,
        # Constraints
        act_min_active: int = 2,
        act_max_active: int = 6,
        agg_min_active: int = 1,
        agg_max_active: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize STDP Symmetric strategy."""
        # Critical period timing
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

        # Phase rates
        self.exploration_activate = exploration_activate
        self.exploration_deactivate = exploration_deactivate
        self.confirmation_activate = confirmation_activate
        self.confirmation_deactivate_max = confirmation_deactivate_max
        self.confirmation_deactivate_min = confirmation_deactivate_min
        self.consolidation_activate = consolidation_activate
        self.consolidation_deactivate = consolidation_deactivate

        # STDP parameters
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.history_length = history_length
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.temporal_decay = temporal_decay

        # Cross-domain timing
        self.cross_timing_lr = cross_timing_lr
        self.cross_timing_weight = cross_timing_weight

        # Memory cell parameters
        self.memory_cell_timing_threshold = memory_cell_timing_threshold
        self.memory_cell_gens = memory_cell_gens
        self.memory_cell_decay_rate = memory_cell_decay_rate

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Discovery
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        # Other
        self.affinity_protection_threshold = affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier
        self.early_consolidation_threshold = early_consolidation_threshold

        # Constraints
        self.act_min_active = act_min_active
        self.act_max_active = act_max_active
        self.agg_min_active = agg_min_active
        self.agg_max_active = agg_max_active
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase with early consolidation."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual STDP tracking."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinity = act_affinity.at[i].set(0.6)

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinity = agg_affinity.at[i].set(0.6)

        # STDP timing scores (replaces pairwise weights with per-function timing)
        act_timing_scores = jnp.zeros(NUM_ACTIVATIONS)
        agg_timing_scores = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain timing correlation
        cross_timing = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        # Memory cell tracking (based on sustained positive timing)
        act_timing_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_timing_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        act_ever_discovered = set(initial_act)
        agg_ever_discovered = set(initial_agg)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_timing_scores': act_timing_scores,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_timing_scores': agg_timing_scores,
            # Cross-domain
            'cross_timing': cross_timing,
            # Memory cells
            'act_timing_counts': act_timing_counts,
            'agg_timing_counts': agg_timing_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Discovery tracking
            'act_ever_discovered': act_ever_discovered,
            'agg_ever_discovered': agg_ever_discovered,
            'total_act_discoveries': 0,
            'total_agg_discoveries': 0,
            'discovery_to_palette': 0,
            # History tracking
            'act_history': [],
            'agg_history': [],
            'fitness_history': [],
            # General state
            'rng_key': jax.random.PRNGKey(seed + 161632),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            'ltp_events': 0,
            'ltd_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _compute_temporal_weight(self, generations_from_improvement: int) -> float:
        """Compute temporal weight based on distance from improvement event."""
        return self.temporal_decay ** abs(generations_from_improvement)

    def _stdp_update(
        self,
        affinity: jnp.ndarray,
        timing_scores: jnp.ndarray,
        history: List[Tuple[int, jnp.ndarray, float]],
        current_gen: int,
        fitness_improved: bool,
        improvement_magnitude: float,
        phase: str,
        cross_boost: jnp.ndarray,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Apply STDP update with memory cell protection and discovery boost.

        Args:
            affinity: Current function affinity
            timing_scores: Current per-function timing scores
            history: History of (gen, mask, fitness) tuples
            current_gen: Current generation
            fitness_improved: Whether fitness improved this generation
            improvement_magnitude: How much fitness improved (normalized)
            phase: Current developmental phase
            cross_boost: Cross-domain timing boost
            memory_cells: Current memory cell status
            newly_discovered: Indices of newly discovered functions

        Returns:
            Tuple of (new_affinity, new_timing_scores, update_info)
        """
        newly_discovered = newly_discovered or []

        # Phase-specific learning rate modulation
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr_mult = self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr_mult = self.confirmation_lr_multiplier
        else:
            lr_mult = 0.1  # Very slow in consolidation

        new_affinity = affinity.copy()
        new_timing = timing_scores.copy()

        ltp_functions = []
        ltd_functions = []

        if fitness_improved and len(history) >= 2:
            # LTP: Boost functions that were active BEFORE improvement
            for hist_gen, hist_mask, hist_fitness in history:
                gens_before = current_gen - hist_gen

                if 1 <= gens_before <= self.ltp_window:
                    # This generation preceded improvement - apply LTP
                    temporal_weight = self._compute_temporal_weight(gens_before)
                    active = (hist_mask > 0.5).astype(jnp.float32)

                    # Update affinity
                    ltp_delta = (
                        self.ltp_rate * lr_mult *
                        temporal_weight *
                        improvement_magnitude *
                        active
                    )
                    new_affinity = new_affinity + ltp_delta

                    # Update timing scores (positive timing correlation)
                    timing_delta = temporal_weight * improvement_magnitude * active * 0.2
                    new_timing = new_timing + timing_delta

                    # Track which functions got LTP
                    ltp_indices = [i for i in range(len(hist_mask)) if hist_mask[i] > 0.5]
                    ltp_functions.extend(ltp_indices)

        # LTD: Weaken functions that were active but didn't precede improvement
        if not fitness_improved and len(history) >= self.ltd_window:
            recent_mask = history[-1][1] if history else jnp.zeros(len(affinity))
            active = (recent_mask > 0.5).astype(jnp.float32)

            # Memory cells resist LTD
            ltd_mask = jnp.logical_and(active > 0.5, ~memory_cells)
            ltd_delta = self.ltd_rate * lr_mult * 0.5 * ltd_mask.astype(jnp.float32)
            new_affinity = new_affinity - ltd_delta

            # Negative timing for functions not producing results
            timing_delta = -0.1 * ltd_mask.astype(jnp.float32)
            new_timing = new_timing + timing_delta

            ltd_indices = [i for i in range(len(recent_mask)) if recent_mask[i] > 0.5 and not memory_cells[i]]
            ltd_functions.extend(ltd_indices)

        # Add cross-domain timing boost
        new_affinity = new_affinity + self.cross_timing_weight * cross_boost

        # Discovery boost
        for idx in newly_discovered:
            new_affinity = new_affinity.at[idx].set(
                new_affinity[idx] + self.discovery_boost
            )

        # Memory cells resist negative changes
        negative_change = new_affinity < affinity
        memory_protected = jnp.logical_and(negative_change, memory_cells)
        new_affinity = jnp.where(
            memory_protected,
            affinity * (1 - self.memory_cell_decay_rate),
            new_affinity
        )

        new_affinity = jnp.clip(new_affinity, 0.0, 1.0)
        new_timing = jnp.clip(new_timing, -1.0, 1.0)

        update_info = {
            'ltp_functions': list(set(ltp_functions)),
            'ltd_functions': list(set(ltd_functions)),
            'ltp_applied': fitness_improved and len(history) >= 2,
            'ltd_applied': not fitness_improved and len(history) >= self.ltd_window,
        }

        return new_affinity, new_timing, update_info

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for critical functions."""
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )
        new_agg = agg_affinity
        for idx in CORE_EXTREME_AGGS:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )
        return new_act, new_agg

    def _update_memory_cells_from_timing(
        self,
        timing_scores: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        timing_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cells based on sustained positive timing correlation.

        Memory cells form when EITHER:
        1. Positive timing score (>= threshold) for memory_cell_gens generations
        2. High affinity (>= threshold) for memory_cell_gens generations

        This hybrid approach matches clonal selection and predictive coding fixes.
        """
        active = mask > 0.5

        # Path 1: Positive timing correlation
        positive_timing = jnp.logical_and(
            timing_scores >= self.memory_cell_timing_threshold,
            active
        )

        # Path 2: High affinity (fallback)
        high_affinity = jnp.logical_and(
            affinity >= 0.75,
            active
        )

        # Either path counts toward memory cell status
        memory_candidate = jnp.logical_or(positive_timing, high_affinity)

        new_counts = jnp.where(memory_candidate, timing_counts + 1, 0)
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _update_cross_timing(
        self,
        cross_timing: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain timing correlation."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        delta = self.cross_timing_lr * fitness_delta * co_active
        return jnp.clip(cross_timing + delta, -1.0, 1.0)

    def _mutate_palette_stdp(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        phase: str,
        memory_cells: jnp.ndarray,
        min_active: int,
        max_active: int,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Apply phase-appropriate mutation with memory cell protection."""
        newly_discovered = newly_discovered or []
        key1, key2 = jax.random.split(key)
        n_funcs = len(mask)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        discovery_to_palette = 0

        activate_probs = jax.random.uniform(key1, (n_funcs,))
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))

        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = self.exploration_activate
            deactivate_rate = self.exploration_deactivate
            use_protection = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = self.confirmation_activate
            use_protection = True
        else:
            activate_rate = self.consolidation_activate
            deactivate_rate = self.consolidation_deactivate
            use_protection = True

        for i in range(n_funcs):
            aff = float(affinity[i])
            is_memory = bool(memory_cells[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                if use_protection:
                    effective_rate = activate_rate * (0.5 + 0.5 * aff)
                else:
                    effective_rate = activate_rate

                current_active = int(jnp.sum(new_mask > 0.5))
                if activate_probs[i] < effective_rate and current_active < max_active:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
                    if i in newly_discovered:
                        discovery_to_palette += 1
            else:
                # Active: maybe deactivate
                # Memory cells never deactivate
                if is_memory:
                    continue

                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if aff >= self.affinity_protection_threshold:
                        continue
                    deact_rate = self.consolidation_deactivate
                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if aff >= self.affinity_protection_threshold:
                        deact_rate = self.confirmation_deactivate_min
                    else:
                        t = aff / self.affinity_protection_threshold
                        deact_rate = (
                            self.confirmation_deactivate_max * (1 - t) +
                            self.confirmation_deactivate_min * t
                        )
                else:
                    deact_rate = self.exploration_deactivate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []
            discovery_to_palette = 0

        # Discovery slot guarantee
        if self.enable_discovery_slot and newly_discovered:
            current_active = int(jnp.sum(new_mask > 0.5))
            not_in_palette = [idx for idx in newly_discovered if new_mask[idx] < 0.5]
            if not_in_palette and current_active < max_active:
                best_new = max(not_in_palette, key=lambda j: float(affinity[j]))
                new_mask = new_mask.at[best_new].set(1.0)
                discovery_to_palette += 1

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
        }, discovery_to_palette

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual STDP and memory cells."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Determine phase
        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']

        # Compute improvement magnitude (normalized)
        improvement_magnitude = fitness_delta / max(0.1, prev_best_fitness)
        improvement_magnitude = max(0.0, min(1.0, improvement_magnitude))

        # Update histories
        act_history = state['act_history'] + [(generation, state['act_mask'].copy(), best_fitness)]
        agg_history = state['agg_history'] + [(generation, state['agg_mask'].copy(), best_fitness)]
        if len(act_history) > self.history_length:
            act_history = act_history[-self.history_length:]
        if len(agg_history) > self.history_length:
            agg_history = agg_history[-self.history_length:]

        # Identify new discovery candidates
        current_act_palette = set(mask_to_indices(state['act_mask']))
        current_agg_palette = set(mask_to_indices(state['agg_mask']))
        act_ever_discovered = state['act_ever_discovered'].copy()
        agg_ever_discovered = state['agg_ever_discovered'].copy()

        act_new_candidates = [
            i for i in range(NUM_ACTIVATIONS)
            if i not in act_ever_discovered and i not in current_act_palette
        ]
        agg_new_candidates = [
            i for i in range(NUM_AGGREGATIONS)
            if i not in agg_ever_discovered and i not in current_agg_palette
        ]

        # Update cross-domain timing
        new_cross_timing = self._update_cross_timing(
            state['cross_timing'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Compute cross-domain boosts
        act_cross_boost = jnp.dot(new_cross_timing, (state['agg_mask'] > 0.5).astype(jnp.float32))
        agg_cross_boost = jnp.dot(new_cross_timing.T, (state['act_mask'] > 0.5).astype(jnp.float32))

        # Normalize boosts
        n_agg = max(jnp.sum(state['agg_mask'] > 0.5), 1)
        n_act = max(jnp.sum(state['act_mask'] > 0.5), 1)
        act_cross_boost = act_cross_boost / n_agg
        agg_cross_boost = agg_cross_boost / n_act

        # Apply STDP update to both domains
        new_act_aff, new_act_timing, act_stdp_info = self._stdp_update(
            state['act_affinity'], state['act_timing_scores'], act_history,
            generation, improved, improvement_magnitude, phase,
            act_cross_boost, state['act_memory_cells'], act_new_candidates
        )
        new_agg_aff, new_agg_timing, agg_stdp_info = self._stdp_update(
            state['agg_affinity'], state['agg_timing_scores'], agg_history,
            generation, improved, improvement_magnitude, phase,
            agg_cross_boost, state['agg_memory_cells'], agg_new_candidates
        )

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update memory cells from sustained positive timing
        new_act_tc, new_act_mem_cells = self._update_memory_cells_from_timing(
            new_act_timing, new_act_aff, state['act_mask'],
            state['act_timing_counts'], state['act_memory_cells']
        )
        new_agg_tc, new_agg_mem_cells = self._update_memory_cells_from_timing(
            new_agg_timing, new_agg_aff, state['agg_mask'],
            state['agg_timing_counts'], state['agg_memory_cells']
        )

        # Apply mutation
        new_act_mask, act_mut_info, act_disc_to_pal = self._mutate_palette_stdp(
            k1, state['act_mask'], new_act_aff, phase,
            new_act_mem_cells, self.act_min_active, self.act_max_active,
            act_new_candidates
        )
        new_agg_mask, agg_mut_info, agg_disc_to_pal = self._mutate_palette_stdp(
            k2, state['agg_mask'], new_agg_aff, phase,
            new_agg_mem_cells, self.agg_min_active, self.agg_max_active,
            agg_new_candidates
        )

        # Update discovery tracking
        new_act_discoveries = 0
        new_agg_discoveries = 0
        final_act_palette = mask_to_indices(new_act_mask)
        final_agg_palette = mask_to_indices(new_agg_mask)

        for idx in final_act_palette:
            if idx not in act_ever_discovered:
                act_ever_discovered.add(idx)
                new_act_discoveries += 1
        for idx in final_agg_palette:
            if idx not in agg_ever_discovered:
                agg_ever_discovered.add(idx)
                new_agg_discoveries += 1

        # Update fitness history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Track STDP events
        ltp_events = state['ltp_events'] + (1 if act_stdp_info['ltp_applied'] or agg_stdp_info['ltp_applied'] else 0)
        ltd_events = state['ltd_events'] + (1 if act_stdp_info['ltd_applied'] or agg_stdp_info['ltd_applied'] else 0)

        new_state = {
            # Activation domain
            'act_mask': new_act_mask,
            'act_affinity': new_act_aff,
            'act_timing_scores': new_act_timing,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_aff,
            'agg_timing_scores': new_agg_timing,
            # Cross-domain
            'cross_timing': new_cross_timing,
            # Memory cells
            'act_timing_counts': new_act_tc,
            'agg_timing_counts': new_agg_tc,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            # Discovery tracking
            'act_ever_discovered': act_ever_discovered,
            'agg_ever_discovered': agg_ever_discovered,
            'total_act_discoveries': state['total_act_discoveries'] + new_act_discoveries,
            'total_agg_discoveries': state['total_agg_discoveries'] + new_agg_discoveries,
            'discovery_to_palette': state['discovery_to_palette'] + act_disc_to_pal + agg_disc_to_pal,
            # History tracking
            'act_history': act_history,
            'agg_history': agg_history,
            'fitness_history': fitness_history,
            # General state
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'ltp_events': ltp_events,
            'ltd_events': ltd_events,
        }

        # Check sin and extreme agg retention
        has_sin = SIN_IDX in final_act_palette
        has_extreme_agg = any(idx in final_agg_palette for idx in CORE_EXTREME_AGGS)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': final_act_palette,
            'current_agg_palette': final_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            # STDP stats
            'act_avg_affinity': float(jnp.mean(new_act_aff)),
            'agg_avg_affinity': float(jnp.mean(new_agg_aff)),
            'act_avg_timing': float(jnp.mean(new_act_timing)),
            'agg_avg_timing': float(jnp.mean(new_agg_timing)),
            'ltp_applied': act_stdp_info['ltp_applied'] or agg_stdp_info['ltp_applied'],
            'ltd_applied': act_stdp_info['ltd_applied'] or agg_stdp_info['ltd_applied'],
            'total_ltp_events': ltp_events,
            'total_ltd_events': ltd_events,
            # Sin and extreme agg
            'has_sin': has_sin,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'sin_timing': float(new_act_timing[SIN_IDX]),
            'has_extreme_agg': has_extreme_agg,
            'extreme_agg_affinities': [float(new_agg_aff[idx]) for idx in CORE_EXTREME_AGGS],
            # Discovery
            'new_act_discoveries': new_act_discoveries,
            'new_agg_discoveries': new_agg_discoveries,
            'total_act_discoveries': new_state['total_act_discoveries'],
            'total_agg_discoveries': new_state['total_agg_discoveries'],
            'discovery_to_palette': new_state['discovery_to_palette'],
            # Memory cells
            'act_memory_cell_count': int(jnp.sum(new_act_mem_cells)),
            'agg_memory_cell_count': int(jnp.sum(new_agg_mem_cells)),
        }
        metrics.update(act_mut_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': SIN_IDX in act_palette,
            'has_extreme_agg': any(idx in agg_palette for idx in CORE_EXTREME_AGGS),
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'phase': state['phase'],
            # STDP
            'act_avg_affinity': float(jnp.mean(state['act_affinity'])),
            'agg_avg_affinity': float(jnp.mean(state['agg_affinity'])),
            'act_avg_timing': float(jnp.mean(state['act_timing_scores'])),
            'agg_avg_timing': float(jnp.mean(state['agg_timing_scores'])),
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'sin_timing': float(state['act_timing_scores'][SIN_IDX]),
            'total_ltp_events': state['ltp_events'],
            'total_ltd_events': state['ltd_events'],
            # Discovery
            'total_act_discoveries': state['total_act_discoveries'],
            'total_agg_discoveries': state['total_agg_discoveries'],
            'discovery_to_palette': state['discovery_to_palette'],
            # Memory cells
            'act_memory_cell_count': int(jnp.sum(state['act_memory_cells'])),
            'agg_memory_cell_count': int(jnp.sum(state['agg_memory_cells'])),
            'act_memory_cell_indices': [
                i for i in range(NUM_ACTIVATIONS) if state['act_memory_cells'][i]
            ],
            'agg_memory_cell_indices': [
                i for i in range(NUM_AGGREGATIONS) if state['agg_memory_cells'][i]
            ],
        }
