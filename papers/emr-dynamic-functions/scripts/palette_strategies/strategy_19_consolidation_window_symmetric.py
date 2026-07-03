"""Strategy 19 Symmetric: Consolidation Window with Memory Cells and Affinity Floors.

Extends ConsolidationWindowDual with winning patterns from eligibility_trace_symmetric:
- Working memory and long-term memory (LTM) for both domains
- Periodic consolidation phases (sleep-like memory replay)
- Memory cells from sustained high affinity
- Affinity floors for sin and extreme aggregations (CRITICAL for retention)

Biological Basis:
- Sleep consolidation: Memory replay strengthens important traces
- Working memory: Short-term, volatile, decays quickly
- Long-term memory: Stable, resistant to interference
- Consolidation phase: Low mutation, high replay, transfer to LTM

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


class ConsolidationPhase:
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"


class ConsolidationWindowSymmetricStrategy(PaletteEvolutionStrategy):
    """Sleep-like consolidation with memory cells for dual palette evolution.

    Combines the periodic consolidation phases with memory cells
    and affinity floors from the winning eligibility_trace strategy.

    Key innovations:
    - Working memory decays quickly but learns fast during active phase
    - Long-term memory is stable and protected
    - Consolidation phases transfer proven patterns to LTM
    - Memory cells provide permanent protection beyond LTM
    - Affinity floors ensure sin and extreme aggs are never lost
    """

    name = "consolidation_window_symmetric"
    description = "Sleep-like consolidation with memory cells"

    def __init__(
        self,
        # Consolidation timing
        consolidation_frequency: int = 10,
        consolidation_duration: int = 3,
        # Consolidation parameters
        replay_strength: float = 1.5,
        replay_threshold: float = 0.6,
        transfer_rate: float = 0.1,
        ltm_decay_rate: float = 0.02,
        # Active phase parameters
        active_learning_rate: float = 0.15,
        active_mutation_rate: float = 0.20,
        # Consolidation phase parameters
        consolidation_mutation_rate: float = 0.02,
        consolidation_learning_rate: float = 0.05,
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
        # Cross-domain
        cross_learning_rate: float = 0.12,
        cross_influence: float = 0.25,
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
        """Initialize Consolidation Window Symmetric strategy."""
        # Consolidation timing
        self.consolidation_frequency = consolidation_frequency
        self.consolidation_duration = consolidation_duration

        # Consolidation parameters
        self.replay_strength = replay_strength
        self.replay_threshold = replay_threshold
        self.transfer_rate = transfer_rate
        self.ltm_decay_rate = ltm_decay_rate

        # Phase parameters
        self.active_learning_rate = active_learning_rate
        self.active_mutation_rate = active_mutation_rate
        self.consolidation_mutation_rate = consolidation_mutation_rate
        self.consolidation_learning_rate = consolidation_learning_rate

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

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_consolidation_phase(
        self, generation: int, last_consol: int
    ) -> Tuple[str, bool]:
        """Determine current consolidation phase."""
        gens_since = generation - last_consol
        if gens_since < self.consolidation_duration:
            return ConsolidationPhase.CONSOLIDATING, False
        elif gens_since >= self.consolidation_frequency:
            return ConsolidationPhase.CONSOLIDATING, True
        return ConsolidationPhase.ACTIVE, False

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with working/LTM and memory cell tracking."""
        # Activation domain
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_working = jnp.ones(NUM_ACTIVATIONS) * 0.5
        act_ltm = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=bool)

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_working = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_ltm = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Memory cell tracking
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=bool)

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_working': act_working,
            'act_ltm': act_ltm,
            'act_memory_counts': act_memory_counts,
            'act_memory_cells': act_memory_cells,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_working': agg_working,
            'agg_ltm': agg_ltm,
            'agg_memory_counts': agg_memory_counts,
            'agg_memory_cells': agg_memory_cells,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Common state
            'rng_key': jax.random.PRNGKey(seed + 191920),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Consolidation state
            'consolidation_phase': ConsolidationPhase.ACTIVE,
            'last_consolidation': -self.consolidation_frequency,
            'consolidations_completed': 0,
            # Tracking
            'fitness_history': [],
            'fitness_ema': 0.5,
            'act_replay_events': 0,
            'agg_replay_events': 0,
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

    def _compute_effective_affinity(
        self, working: jnp.ndarray, ltm: jnp.ndarray, phase: str
    ) -> jnp.ndarray:
        """Compute effective affinity from working + LTM.

        During consolidation: LTM dominates (70%)
        During active: Working dominates (60%)
        """
        if phase == ConsolidationPhase.CONSOLIDATING:
            return 0.3 * working + 0.7 * ltm
        return 0.6 * working + 0.4 * ltm

    def _update_working_memory(
        self,
        working: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> jnp.ndarray:
        """Update working memory based on fitness and phase."""
        lr = self.consolidation_learning_rate if phase == ConsolidationPhase.CONSOLIDATING else self.active_learning_rate
        active = (mask > 0.5).astype(jnp.float32)

        new_working = working.copy()
        for i in range(len(working)):
            if float(active[i]) > 0.5:
                if fitness_signal >= 0:
                    delta = lr * fitness_signal
                else:
                    # Memory cells resist negative changes
                    if bool(memory_cells[i]):
                        delta = lr * 0.1 * fitness_signal
                    else:
                        delta = lr * 0.3 * fitness_signal
                new_working = new_working.at[i].set(
                    max(0.0, min(1.0, float(new_working[i]) + delta))
                )

        return new_working

    def _consolidate_memory(
        self,
        working: jnp.ndarray,
        ltm: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Perform memory consolidation: replay and transfer.

        Memory cells get extra replay strength.
        """
        new_working = working.copy()
        new_ltm = ltm.copy()
        active = (mask > 0.5).astype(jnp.float32)
        n_replay, n_transfer = 0, 0

        for i in range(n_funcs):
            w_val = float(working[i])
            is_memory = bool(memory_cells[i])

            # Replay: boost working memory for high-value functions
            if w_val >= self.replay_threshold or is_memory:
                # Memory cells always get replayed
                boost_factor = 1.5 if is_memory else 1.0
                boost = self.replay_strength * boost_factor * (w_val - self.replay_threshold + 0.1)
                boost = max(0, boost)
                new_working = new_working.at[i].set(
                    min(0.95, float(new_working[i]) + boost)
                )
                n_replay += 1

            # Transfer: working → LTM for active high-value functions
            if (w_val >= self.replay_threshold or is_memory) and float(active[i]) > 0.5:
                diff = float(new_working[i]) - float(ltm[i])
                # Memory cells transfer faster
                transfer_factor = 1.5 if is_memory else 1.0
                transfer = self.transfer_rate * transfer_factor * diff
                new_ltm = new_ltm.at[i].set(
                    min(0.95, float(new_ltm[i]) + transfer)
                )
                if transfer > 0.01:
                    n_transfer += 1

            # Decay: inactive functions slowly decay in LTM
            if float(active[i]) < 0.5 and not is_memory:
                decay = self.ltm_decay_rate * (float(new_ltm[i]) - 0.5)
                new_ltm = new_ltm.at[i].set(
                    max(0.05, float(new_ltm[i]) - decay)
                )

        return new_working, new_ltm, n_replay, n_transfer

    def _update_memory_cells(
        self,
        effective_affinity: jnp.ndarray,
        mask: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cells based on sustained high affinity.

        Consolidation phase accelerates memory cell formation.
        """
        active = mask > 0.5

        # During consolidation, lower the threshold (easier to form memory)
        if phase == ConsolidationPhase.CONSOLIDATING:
            threshold = self.memory_cell_affinity_threshold - 0.05
        else:
            threshold = self.memory_cell_affinity_threshold

        high_affinity = jnp.logical_and(effective_affinity >= threshold, active)

        new_counts = jnp.where(high_affinity, memory_counts + 1, 0)
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _apply_affinity_floors(
        self,
        act_effective: jnp.ndarray,
        agg_effective: jnp.ndarray,
        act_ltm: jnp.ndarray,
        agg_ltm: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors to sin and extreme aggregations.

        CRITICAL for 100% retention.
        Floors apply to both effective affinity and LTM.
        """
        # Sin activation floor
        new_act_eff = act_effective.at[SIN_IDX].set(
            jnp.maximum(act_effective[SIN_IDX], self.sin_affinity_floor)
        )
        new_act_ltm = act_ltm.at[SIN_IDX].set(
            jnp.maximum(act_ltm[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation floors
        new_agg_eff = agg_effective
        new_agg_ltm = agg_ltm
        for idx in CORE_EXTREME_AGGS:
            new_agg_eff = new_agg_eff.at[idx].set(
                jnp.maximum(new_agg_eff[idx], self.extreme_agg_affinity_floor)
            )
            new_agg_ltm = new_agg_ltm.at[idx].set(
                jnp.maximum(new_agg_ltm[idx], self.extreme_agg_affinity_floor)
            )

        return new_act_eff, new_agg_eff, new_act_ltm, new_agg_ltm

    def _compute_protection(
        self,
        effective: jnp.ndarray,
        ltm: jnp.ndarray,
        memory_cells: jnp.ndarray,
        cross: jnp.ndarray,
        other_mask: jnp.ndarray,
        is_act: bool,
    ) -> jnp.ndarray:
        """Compute protection combining effective, LTM, memory cells, and cross-domain."""
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)

        if is_act:
            cross_score = jnp.dot(cross, other_active) / n_other
        else:
            cross_score = jnp.dot(cross.T, other_active) / n_other

        # Base protection from effective and LTM
        protection = 0.4 * effective + 0.3 * ltm

        # Memory cell bonus
        memory_bonus = memory_cells.astype(jnp.float32) * 0.3
        protection = protection + memory_bonus

        # Cross-domain bonus
        protection = protection + 0.1 * cross_score * self.cross_influence

        return jnp.clip(protection, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        memory_cells: jnp.ndarray,
        phase: str,
        n_funcs: int,
        min_active: int,
        max_active: int,
        protected_indices: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with phase-dependent rates and memory cell protection.

        Args:
            protected_indices: Indices that should never be deactivated (e.g., sin, extreme aggs).
                              These get extremely low deactivation rates.
        """
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []

        # Phase-dependent mutation rate
        mut_rate = self.consolidation_mutation_rate if phase == ConsolidationPhase.CONSOLIDATING else self.active_mutation_rate

        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))

        current = int(jnp.sum(mask > 0.5))
        protected_set = set(protected_indices) if protected_indices else set()

        for i in range(n_funcs):
            prot = float(protection[i])
            is_memory = bool(memory_cells[i])
            is_protected = i in protected_set

            if mask[i] < 0.5:
                # Activate
                if current + len(activated) >= max_active:
                    continue
                eff_rate = mut_rate * (0.5 + prot)
                if act_probs[i] < eff_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Protected indices (sin, extreme aggs) almost never deactivate
                if is_protected:
                    dr = 0.001  # 0.1% chance - essentially never
                # Memory cells are highly protected
                elif is_memory:
                    dr = mut_rate * 0.05
                elif prot >= self.affinity_protection_threshold:
                    dr = mut_rate * 0.1
                else:
                    dr = mut_rate * (1.0 - prot)

                # Consolidation phase reduces mutation
                if phase == ConsolidationPhase.CONSOLIDATING:
                    dr *= 0.2

                if deact_probs[i] < dr:
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
        """Update with consolidation windows and memory cells."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Determine consolidation phase
        phase, starting = self._get_consolidation_phase(generation, state['last_consolidation'])
        last_consol = generation if starting else state['last_consolidation']
        consol_count = state['consolidations_completed'] + (1 if starting else 0)

        # Fitness signal
        alpha = 0.2
        new_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fs = max(-1.0, min(1.0, (best_fitness - new_ema) / max(0.1, new_ema)))

        # Update working memory
        new_act_working = self._update_working_memory(
            state['act_working'], state['act_mask'], state['act_memory_cells'], fs, phase
        )
        new_agg_working = self._update_working_memory(
            state['agg_working'], state['agg_mask'], state['agg_memory_cells'], fs, phase
        )

        # Update cross-domain affinity
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_delta = self.cross_learning_rate * fs * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Consolidation (if in consolidating phase)
        new_act_ltm, new_agg_ltm = state['act_ltm'], state['agg_ltm']
        act_replay, agg_replay = 0, 0

        if phase == ConsolidationPhase.CONSOLIDATING:
            new_act_working, new_act_ltm, ar, _ = self._consolidate_memory(
                new_act_working, state['act_ltm'], state['act_mask'],
                state['act_memory_cells'], NUM_ACTIVATIONS
            )
            new_agg_working, new_agg_ltm, agr, _ = self._consolidate_memory(
                new_agg_working, state['agg_ltm'], state['agg_mask'],
                state['agg_memory_cells'], NUM_AGGREGATIONS
            )
            act_replay, agg_replay = ar, agr

        # Compute effective affinity
        act_eff = self._compute_effective_affinity(new_act_working, new_act_ltm, phase)
        agg_eff = self._compute_effective_affinity(new_agg_working, new_agg_ltm, phase)

        # Apply affinity floors
        act_eff, agg_eff, new_act_ltm, new_agg_ltm = self._apply_affinity_floors(
            act_eff, agg_eff, new_act_ltm, new_agg_ltm
        )

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            act_eff, state['act_mask'], state['act_memory_counts'],
            state['act_memory_cells'], phase
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            agg_eff, state['agg_mask'], state['agg_memory_counts'],
            state['agg_memory_cells'], phase
        )

        # Compute protection
        act_prot = self._compute_protection(
            act_eff, new_act_ltm, new_act_mem_cells, new_cross, state['agg_mask'], True
        )
        agg_prot = self._compute_protection(
            agg_eff, new_agg_ltm, new_agg_mem_cells, new_cross, state['act_mask'], False
        )

        # Mutate palettes (with protected indices for sin and extreme aggs)
        new_act_mask, act_mut = self._mutate_palette(
            k_act, state['act_mask'], act_prot, new_act_mem_cells, phase,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            protected_indices=[SIN_IDX],  # Sin is protected
        )
        new_agg_mask, agg_mut = self._mutate_palette(
            k_agg, state['agg_mask'], agg_prot, new_agg_mem_cells, phase,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            protected_indices=CORE_EXTREME_AGGS,  # max, min are protected
        )

        # Track discoveries
        new_sin_gen, new_extreme_gen = self._track_discoveries(
            new_act_mask, new_agg_mask, generation,
            state['sin_discovered_gen'], state['extreme_agg_discovered_gen']
        )

        # Update fitness history
        fh = state['fitness_history'] + [best_fitness]
        if len(fh) > 20:
            fh = fh[-20:]

        n_act_mem = int(jnp.sum(new_act_mem_cells))
        n_agg_mem = int(jnp.sum(new_agg_mem_cells))

        new_state = {
            # Activation domain
            'act_mask': new_act_mask,
            'act_working': new_act_working,
            'act_ltm': new_act_ltm,
            'act_memory_counts': new_act_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_working': new_agg_working,
            'agg_ltm': new_agg_ltm,
            'agg_memory_counts': new_agg_mem_counts,
            'agg_memory_cells': new_agg_mem_cells,
            # Cross-domain
            'cross_affinity': new_cross,
            # Common state
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Consolidation state
            'consolidation_phase': phase,
            'last_consolidation': last_consol,
            'consolidations_completed': consol_count,
            # Tracking
            'fitness_history': fh,
            'fitness_ema': new_ema,
            'act_replay_events': state['act_replay_events'] + act_replay,
            'agg_replay_events': state['agg_replay_events'] + agg_replay,
            # Discovery
            'sin_discovered_gen': new_sin_gen,
            'extreme_agg_discovered_gen': new_extreme_gen,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'act_palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Consolidation
            'consolidation_phase': phase,
            'starting_consolidation': starting,
            'consolidations_completed': consol_count,
            # Memory cells
            'act_memory_cells': n_act_mem,
            'agg_memory_cells': n_agg_mem,
            'total_memory_cells': n_act_mem + n_agg_mem,
            # Discovery
            'sin_discovered_gen': new_sin_gen,
            'extreme_agg_discovered_gen': new_extreme_gen,
            # Stats
            'sin_affinity': float(act_eff[SIN_IDX]),
            'sin_ltm': float(new_act_ltm[SIN_IDX]),
            'sin_active': SIN_IDX in act_palette,
            'has_extreme_agg': any(idx in agg_palette for idx in CORE_EXTREME_AGGS),
            # Replay events
            'act_replay_events': act_replay,
            'agg_replay_events': agg_replay,
            # Mutations
            'act_activated': act_mut['activated'],
            'act_deactivated': act_mut['deactivated'],
            'agg_activated': agg_mut['activated'],
            'agg_deactivated': agg_mut['deactivated'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with consolidation and memory cell info."""
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
            # Consolidation
            'consolidation_phase': state['consolidation_phase'],
            'consolidations_completed': state['consolidations_completed'],
            # Memory cells
            'act_memory_cells': n_act_mem,
            'agg_memory_cells': n_agg_mem,
            'total_memory_cells': n_act_mem + n_agg_mem,
            # Discovery
            'sin_discovered_gen': state['sin_discovered_gen'],
            'extreme_agg_discovered_gen': state['extreme_agg_discovered_gen'],
            # Affinities
            'sin_affinity': float(state['act_ltm'][SIN_IDX]),
            'avg_act_affinity': float(jnp.mean(state['act_ltm'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_ltm'])),
        }
