"""Strategy 18 Symmetric: Neural Darwinism.

Extends NeuralDarwinismStrategy with symmetric discovery features:
- Dual cooperation/antagonism matrices for activation and aggregation
- Cross-domain cooperation matrix (activation-aggregation synergies)
- Memory cell protection for proven groups
- Affinity floors for sin and extreme aggregations
- Discovery tracking and boost for both domains

Key mechanisms:
1. Dual cooperation/antagonism: Track success/conflict patterns per domain
2. Cross-domain groups: Identify activation-aggregation synergies (sin-max pairs)
3. Memory cells: Proven cooperative groups become memory cells
4. Selective death: Functions with multiple antagonistic relationships get pruned

Biological rationale:
- Edelman's Neural Darwinism (1987): Neuronal groups compete for survival
- Immunological memory: Successful groups become permanent (memory cells)
- Cross-modal cooperation: Different modalities cooperate for perception
- Selective stabilization: Cooperative groups survive, antagonists pruned
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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


class NeuralDarwinismSymmetricStrategy(PaletteEvolutionStrategy):
    """Neural Darwinism with dual cooperation/antagonism and memory cells.

    Extends activation-only Neural Darwinism to discover both activation AND
    aggregation functions while maintaining retention through memory cells
    and selective stabilization.

    Key innovations:
    - Dual cooperation/antagonism matrices per domain
    - Cross-domain cooperation tracking (sin-max synergy detection)
    - Memory cells for proven cooperative groups
    - Affinity floors prevent loss of critical functions
    """

    name = "neural_darwinism_symmetric"
    description = "Neural Darwinism with cross-domain groups and memory cells"

    def __init__(
        self,
        # Critical period timing - activation
        act_exploration_end: int = 30,
        act_confirmation_end: int = 60,
        # Critical period timing - aggregation (longer exploration)
        agg_exploration_end: int = 40,
        agg_confirmation_end: int = 70,
        # Phase rates - activation
        act_exploration_activate: float = 0.30,
        act_exploration_deactivate: float = 0.02,
        act_confirmation_activate: float = 0.10,
        act_confirmation_deactivate_max: float = 0.12,
        act_confirmation_deactivate_min: float = 0.01,
        act_consolidation_activate: float = 0.02,
        act_consolidation_deactivate: float = 0.01,
        # Phase rates - aggregation
        agg_exploration_activate: float = 0.35,
        agg_exploration_deactivate: float = 0.02,
        agg_confirmation_activate: float = 0.12,
        agg_confirmation_deactivate_max: float = 0.10,
        agg_confirmation_deactivate_min: float = 0.01,
        agg_consolidation_activate: float = 0.03,
        agg_consolidation_deactivate: float = 0.01,
        # Neural Darwinism parameters
        cooperation_threshold: float = 0.65,
        antagonism_threshold: float = 0.35,
        cooperation_rate: float = 0.15,
        antagonism_rate: float = 0.10,
        selection_pressure: float = 0.18,
        group_min_size: int = 2,
        antagonism_prune_threshold: float = 0.70,
        selective_death_rate: float = 0.08,
        # Cross-domain parameters
        cross_cooperation_rate: float = 0.12,
        cross_cooperation_threshold: float = 0.60,
        cross_synergy_boost: float = 0.15,
        # Memory cell parameters
        memory_cell_threshold: float = 0.75,
        memory_cell_gens: int = 10,
        memory_cell_decay_rate: float = 0.05,
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Base learning parameters
        learning_rate: float = 0.18,
        affinity_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Other
        early_consolidation_threshold: float = 0.95,
        act_min_active: int = 2,
        act_max_active: int = 6,
        agg_min_active: int = 1,
        agg_max_active: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Neural Darwinism Symmetric strategy."""
        # Critical period timing
        self.act_exploration_end = act_exploration_end
        self.act_confirmation_end = act_confirmation_end
        self.agg_exploration_end = agg_exploration_end
        self.agg_confirmation_end = agg_confirmation_end

        # Phase rates - activation
        self.act_exploration_activate = act_exploration_activate
        self.act_exploration_deactivate = act_exploration_deactivate
        self.act_confirmation_activate = act_confirmation_activate
        self.act_confirmation_deactivate_max = act_confirmation_deactivate_max
        self.act_confirmation_deactivate_min = act_confirmation_deactivate_min
        self.act_consolidation_activate = act_consolidation_activate
        self.act_consolidation_deactivate = act_consolidation_deactivate

        # Phase rates - aggregation
        self.agg_exploration_activate = agg_exploration_activate
        self.agg_exploration_deactivate = agg_exploration_deactivate
        self.agg_confirmation_activate = agg_confirmation_activate
        self.agg_confirmation_deactivate_max = agg_confirmation_deactivate_max
        self.agg_confirmation_deactivate_min = agg_confirmation_deactivate_min
        self.agg_consolidation_activate = agg_consolidation_activate
        self.agg_consolidation_deactivate = agg_consolidation_deactivate

        # Neural Darwinism
        self.cooperation_threshold = cooperation_threshold
        self.antagonism_threshold = antagonism_threshold
        self.cooperation_rate = cooperation_rate
        self.antagonism_rate = antagonism_rate
        self.selection_pressure = selection_pressure
        self.group_min_size = group_min_size
        self.antagonism_prune_threshold = antagonism_prune_threshold
        self.selective_death_rate = selective_death_rate

        # Cross-domain
        self.cross_cooperation_rate = cross_cooperation_rate
        self.cross_cooperation_threshold = cross_cooperation_threshold
        self.cross_synergy_boost = cross_synergy_boost

        # Memory cells
        self.memory_cell_threshold = memory_cell_threshold
        self.memory_cell_gens = memory_cell_gens
        self.memory_cell_decay_rate = memory_cell_decay_rate

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Discovery
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        # Base learning
        self.learning_rate = learning_rate
        self.affinity_protection_threshold = affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Other
        self.early_consolidation_threshold = early_consolidation_threshold
        self.act_min_active = act_min_active
        self.act_max_active = act_max_active
        self.agg_min_active = agg_min_active
        self.agg_max_active = agg_max_active
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_phase(
        self,
        generation: int,
        best_fitness: float,
        exploration_end: int,
        confirmation_end: int,
    ) -> str:
        """Determine current phase for a domain."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual cooperation/antagonism matrices."""
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

        # Cooperation/Antagonism matrices - Activation
        act_cooperation = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        act_antagonism = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS))

        # Cooperation/Antagonism matrices - Aggregation
        agg_cooperation = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        agg_antagonism = jnp.zeros((NUM_AGGREGATIONS, NUM_AGGREGATIONS))

        # Cross-domain cooperation matrix
        cross_cooperation = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        act_ever_discovered = set(initial_act)
        agg_ever_discovered = set(initial_agg)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_cooperation': act_cooperation,
            'act_antagonism': act_antagonism,
            'act_phase': CriticalPeriodPhase.EXPLORATION,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_cooperation': agg_cooperation,
            'agg_antagonism': agg_antagonism,
            'agg_phase': CriticalPeriodPhase.EXPLORATION,
            # Cross-domain
            'cross_cooperation': cross_cooperation,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Discovery tracking
            'act_ever_discovered': act_ever_discovered,
            'agg_ever_discovered': agg_ever_discovered,
            'total_act_discoveries': 0,
            'total_agg_discoveries': 0,
            'discovery_to_palette': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 181831),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            'fitness_ema': 0.5,
            # Tracking
            'act_neuronal_groups': [],
            'agg_neuronal_groups': [],
            'cross_synergy_pairs': [],
            'selection_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _update_cooperation_antagonism(
        self,
        cooperation: jnp.ndarray,
        antagonism: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update cooperation and antagonism matrices with memory cell protection."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr = self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr = self.confirmation_lr_multiplier
        else:
            lr = 0.1

        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        if fitness_signal > 0:
            coop_delta = self.cooperation_rate * lr * fitness_signal * co_active
            new_cooperation = jnp.clip(cooperation + coop_delta, 0.0, 1.0)
            antag_delta = -self.antagonism_rate * lr * 0.3 * fitness_signal * co_active
            new_antagonism = jnp.clip(antagonism + antag_delta, 0.0, 1.0)
        else:
            antag_delta = self.antagonism_rate * lr * abs(fitness_signal) * co_active
            new_antagonism = jnp.clip(antagonism + antag_delta, 0.0, 1.0)
            coop_delta = -self.cooperation_rate * lr * 0.3 * abs(fitness_signal) * co_active
            new_cooperation = jnp.clip(cooperation + coop_delta, 0.0, 1.0)

        # Memory cells maintain cooperation (don't decrease)
        memory_pairs = jnp.outer(memory_cells.astype(jnp.float32), memory_cells.astype(jnp.float32))
        new_cooperation = jnp.where(
            memory_pairs > 0,
            jnp.maximum(new_cooperation, cooperation * 0.95),
            new_cooperation
        )

        return new_cooperation, new_antagonism

    def _update_cross_cooperation(
        self,
        cross_cooperation: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update cross-domain cooperation matrix."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(act_active, agg_active)

        if fitness_signal > 0:
            delta = self.cross_cooperation_rate * fitness_signal * co_active
        else:
            delta = -self.cross_cooperation_rate * 0.3 * abs(fitness_signal) * co_active

        return jnp.clip(cross_cooperation + delta, 0.0, 1.0)

    def _detect_neuronal_groups(
        self,
        cooperation: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> List[Set[int]]:
        """Detect neuronal groups (clusters of cooperating functions)."""
        n_funcs = len(mask)
        active_indices = [i for i in range(n_funcs) if mask[i] > 0.5]

        if len(active_indices) < self.group_min_size:
            return []

        groups = []
        visited = set()

        def find_group(start: int) -> Set[int]:
            group = {start}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for other in active_indices:
                    if other not in group and cooperation[current, other] > self.cooperation_threshold:
                        group.add(other)
                        queue.append(other)
            return group

        for idx in active_indices:
            if idx not in visited:
                group = find_group(idx)
                if len(group) >= self.group_min_size:
                    groups.append(group)
                visited.update(group)

        return groups

    def _detect_cross_synergy_pairs(
        self,
        cross_cooperation: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> List[Tuple[int, int]]:
        """Detect activation-aggregation synergy pairs."""
        act_active = [i for i in range(NUM_ACTIVATIONS) if act_mask[i] > 0.5]
        agg_active = [i for i in range(NUM_AGGREGATIONS) if agg_mask[i] > 0.5]
        pairs = []

        for act_idx in act_active:
            for agg_idx in agg_active:
                if cross_cooperation[act_idx, agg_idx] > self.cross_cooperation_threshold:
                    pairs.append((act_idx, agg_idx))

        return pairs

    def _detect_antagonistic_pairs(
        self,
        antagonism: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> List[Tuple[int, int]]:
        """Detect pairs of functions with high antagonism."""
        n_funcs = len(mask)
        active_indices = [i for i in range(n_funcs) if mask[i] > 0.5]
        pairs = []

        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                idx_i = active_indices[i]
                idx_j = active_indices[j]
                if antagonism[idx_i, idx_j] > self.antagonism_prune_threshold:
                    pairs.append((idx_i, idx_j))

        return pairs

    def _update_memory_cells(
        self,
        affinities: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell tracking based on sustained high affinity."""
        above_threshold = affinities >= self.memory_cell_threshold
        new_counts = jnp.where(above_threshold, memory_counts + 1, 0)
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)
        return new_counts, new_memory_cells

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

    def _apply_selection(
        self,
        affinity: jnp.ndarray,
        cooperation: jnp.ndarray,
        antagonism: jnp.ndarray,
        mask: jnp.ndarray,
        groups: List[Set[int]],
        antagonistic_pairs: List[Tuple[int, int]],
        memory_cells: jnp.ndarray,
        phase: str,
        cross_boost_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, List[int]]:
        """Apply selective stabilization with memory cell protection."""
        cross_boost_indices = cross_boost_indices or []

        if phase == CriticalPeriodPhase.EXPLORATION:
            pressure = self.selection_pressure * 0.5
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            pressure = self.selection_pressure * 1.0
        else:
            pressure = self.selection_pressure * 0.3

        new_affinity = affinity.copy()
        functions_to_prune = []

        # Boost functions in cooperative groups
        for group in groups:
            group_cooperation = 0
            for i in group:
                for j in group:
                    if i != j:
                        group_cooperation += float(cooperation[i, j])

            if len(group) > 1:
                avg_cooperation = group_cooperation / (len(group) * (len(group) - 1))
                for idx in group:
                    boost = pressure * (avg_cooperation - 0.5)
                    new_affinity = new_affinity.at[idx].set(
                        min(0.95, float(new_affinity[idx]) + boost)
                    )

        # Cross-domain synergy boost
        for idx in cross_boost_indices:
            new_affinity = new_affinity.at[idx].set(
                min(0.95, float(new_affinity[idx]) + self.cross_synergy_boost)
            )

        # Track antagonism per function
        antagonism_count = {i: 0 for i in range(len(affinity))}
        for i, j in antagonistic_pairs:
            antagonism_count[i] += 1
            antagonism_count[j] += 1

            # Penalty (but memory cells resist)
            if not memory_cells[i]:
                penalty = pressure * float(antagonism[i, j])
                new_affinity = new_affinity.at[i].set(
                    max(0.05, float(new_affinity[i]) - penalty * 0.5)
                )
            if not memory_cells[j]:
                penalty = pressure * float(antagonism[i, j])
                new_affinity = new_affinity.at[j].set(
                    max(0.05, float(new_affinity[j]) - penalty * 0.5)
                )

        # Functions with multiple antagonistic relationships (non-memory) may be pruned
        for idx, count in antagonism_count.items():
            if count >= 2 and not memory_cells[idx]:
                functions_to_prune.append(idx)

        return new_affinity, functions_to_prune

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
    ) -> jnp.ndarray:
        """Update affinity with memory cell protection and discovery boost."""
        newly_discovered = newly_discovered or []

        if phase == CriticalPeriodPhase.EXPLORATION:
            lr = self.learning_rate * self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr = self.learning_rate * self.confirmation_lr_multiplier
        else:
            lr = self.learning_rate * 0.1

        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            delta = lr * fitness_signal * active
        else:
            delta = lr * 0.3 * fitness_signal * active

        new_affinity = affinity + delta

        # Memory cells resist negative changes
        negative_delta = delta < 0
        memory_protected = jnp.logical_and(negative_delta, memory_cells)
        new_affinity = jnp.where(
            memory_protected,
            affinity * (1 - self.memory_cell_decay_rate),
            new_affinity
        )

        # Discovery boost
        for idx in newly_discovered:
            new_affinity = new_affinity.at[idx].set(
                new_affinity[idx] + self.discovery_boost
            )

        return jnp.clip(new_affinity, 0.0, 1.0)

    def _compute_protection_scores(
        self,
        affinity: jnp.ndarray,
        cooperation: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection score with memory cell bonus."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        pairwise_score = jnp.dot(cooperation, active) / n_active
        protection = 0.6 * affinity + 0.3 * pairwise_score

        # Memory cells get protection bonus
        protection = jnp.where(memory_cells, protection + 0.2, protection)

        return jnp.clip(protection, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
        functions_to_prune: List[int],
        memory_cells: jnp.ndarray,
        # Phase rates
        exploration_activate: float,
        exploration_deactivate: float,
        confirmation_activate: float,
        confirmation_deactivate_max: float,
        confirmation_deactivate_min: float,
        consolidation_activate: float,
        consolidation_deactivate: float,
        min_active: int,
        max_active: int,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Apply mutation with selective death and memory cell protection."""
        newly_discovered = newly_discovered or []
        key1, key2, key3 = jax.random.split(key, 3)
        n_funcs = len(mask)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        selectively_killed = []
        discovery_to_palette = 0

        activate_probs = jax.random.uniform(key1, (n_funcs,))
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))
        death_probs = jax.random.uniform(key3, (n_funcs,))

        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = exploration_activate
            use_protection = False
            use_selective_death = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = confirmation_activate
            use_protection = True
            use_selective_death = True
        else:
            activate_rate = consolidation_activate
            use_protection = True
            use_selective_death = False

        # Selective death for antagonistic functions (not memory cells)
        if use_selective_death:
            for idx in functions_to_prune:
                if mask[idx] > 0.5 and not memory_cells[idx]:
                    if death_probs[idx] < self.selective_death_rate:
                        new_mask = new_mask.at[idx].set(0.0)
                        selectively_killed.append(idx)

        for i in range(n_funcs):
            if i in selectively_killed:
                continue

            protection = float(protection_scores[i])
            is_memory = bool(memory_cells[i])

            if mask[i] < 0.5:
                if use_protection and phase == CriticalPeriodPhase.CONFIRMATION:
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    current_active = int(jnp.sum(new_mask > 0.5))
                    if current_active < max_active:
                        new_mask = new_mask.at[i].set(1.0)
                        activated.append(i)
                        if i in newly_discovered:
                            discovery_to_palette += 1
            else:
                # Memory cells never deactivate
                if is_memory:
                    continue

                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= self.affinity_protection_threshold:
                        continue
                    deact_rate = consolidation_deactivate
                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= self.affinity_protection_threshold:
                        deact_rate = confirmation_deactivate_min
                    else:
                        t = protection / self.affinity_protection_threshold
                        deact_rate = (
                            confirmation_deactivate_max * (1 - t) +
                            confirmation_deactivate_min * t
                        )
                else:
                    deact_rate = exploration_deactivate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []
            selectively_killed = []
            discovery_to_palette = 0

        # Discovery slot guarantee
        if self.enable_discovery_slot and newly_discovered:
            current_active = int(jnp.sum(new_mask > 0.5))
            not_in_palette = [idx for idx in newly_discovered if new_mask[idx] < 0.5]
            if not_in_palette and current_active < max_active:
                best_new = max(not_in_palette, key=lambda j: float(protection_scores[j]))
                new_mask = new_mask.at[best_new].set(1.0)
                discovery_to_palette += 1

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'selectively_killed': selectively_killed,
        }, discovery_to_palette

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual Neural Darwinism and memory cells."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update phases
        act_phase = self._get_phase(
            generation, new_best,
            self.act_exploration_end, self.act_confirmation_end
        )
        agg_phase = self._get_phase(
            generation, new_best,
            self.agg_exploration_end, self.agg_confirmation_end
        )

        # Fitness signal
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

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

        # Update cooperation/antagonism matrices
        new_act_coop, new_act_antag = self._update_cooperation_antagonism(
            state['act_cooperation'], state['act_antagonism'],
            state['act_mask'], fitness_signal, act_phase, state['act_memory_cells']
        )
        new_agg_coop, new_agg_antag = self._update_cooperation_antagonism(
            state['agg_cooperation'], state['agg_antagonism'],
            state['agg_mask'], fitness_signal, agg_phase, state['agg_memory_cells']
        )

        # Update cross-domain cooperation
        new_cross_coop = self._update_cross_cooperation(
            state['cross_cooperation'], state['act_mask'], state['agg_mask'], fitness_signal
        )

        # Update affinities
        new_act_aff = self._update_affinity(
            state['act_affinity'], state['act_mask'], fitness_signal, act_phase,
            state['act_memory_cells'], act_new_candidates
        )
        new_agg_aff = self._update_affinity(
            state['agg_affinity'], state['agg_mask'], fitness_signal, agg_phase,
            state['agg_memory_cells'], agg_new_candidates
        )

        # Detect groups and pairs
        act_groups = self._detect_neuronal_groups(new_act_coop, state['act_mask'])
        agg_groups = self._detect_neuronal_groups(new_agg_coop, state['agg_mask'])
        cross_synergy = self._detect_cross_synergy_pairs(
            new_cross_coop, state['act_mask'], state['agg_mask']
        )
        act_antag_pairs = self._detect_antagonistic_pairs(new_act_antag, state['act_mask'])
        agg_antag_pairs = self._detect_antagonistic_pairs(new_agg_antag, state['agg_mask'])

        # Get cross-synergy boost indices
        act_synergy_boost = list(set(p[0] for p in cross_synergy))
        agg_synergy_boost = list(set(p[1] for p in cross_synergy))

        # Apply selection
        new_act_aff, act_to_prune = self._apply_selection(
            new_act_aff, new_act_coop, new_act_antag, state['act_mask'],
            act_groups, act_antag_pairs, state['act_memory_cells'], act_phase, act_synergy_boost
        )
        new_agg_aff, agg_to_prune = self._apply_selection(
            new_agg_aff, new_agg_coop, new_agg_antag, state['agg_mask'],
            agg_groups, agg_antag_pairs, state['agg_memory_cells'], agg_phase, agg_synergy_boost
        )

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_aff, state['act_memory_counts'], state['act_memory_cells']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_aff, state['agg_memory_counts'], state['agg_memory_cells']
        )

        # Compute protection scores
        act_protection = self._compute_protection_scores(
            new_act_aff, new_act_coop, state['act_mask'], new_act_mem_cells
        )
        agg_protection = self._compute_protection_scores(
            new_agg_aff, new_agg_coop, state['agg_mask'], new_agg_mem_cells
        )

        # Mutate palettes
        new_act_mask, act_mut_info, act_disc_to_pal = self._mutate_palette(
            k1, state['act_mask'], act_phase, act_protection, act_to_prune,
            new_act_mem_cells,
            self.act_exploration_activate, self.act_exploration_deactivate,
            self.act_confirmation_activate, self.act_confirmation_deactivate_max,
            self.act_confirmation_deactivate_min, self.act_consolidation_activate,
            self.act_consolidation_deactivate, self.act_min_active, self.act_max_active,
            act_new_candidates
        )
        new_agg_mask, agg_mut_info, agg_disc_to_pal = self._mutate_palette(
            k2, state['agg_mask'], agg_phase, agg_protection, agg_to_prune,
            new_agg_mem_cells,
            self.agg_exploration_activate, self.agg_exploration_deactivate,
            self.agg_confirmation_activate, self.agg_confirmation_deactivate_max,
            self.agg_confirmation_deactivate_min, self.agg_consolidation_activate,
            self.agg_consolidation_deactivate, self.agg_min_active, self.agg_max_active,
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
        selection_events = state['selection_events'] + (1 if act_groups or agg_groups else 0)

        new_state = {
            # Activation domain
            'act_mask': new_act_mask,
            'act_affinity': new_act_aff,
            'act_cooperation': new_act_coop,
            'act_antagonism': new_act_antag,
            'act_phase': act_phase,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_aff,
            'agg_cooperation': new_agg_coop,
            'agg_antagonism': new_agg_antag,
            'agg_phase': agg_phase,
            # Cross-domain
            'cross_cooperation': new_cross_coop,
            # Memory cells
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            # Discovery tracking
            'act_ever_discovered': act_ever_discovered,
            'agg_ever_discovered': agg_ever_discovered,
            'total_act_discoveries': state['total_act_discoveries'] + new_act_discoveries,
            'total_agg_discoveries': state['total_agg_discoveries'] + new_agg_discoveries,
            'discovery_to_palette': state['discovery_to_palette'] + act_disc_to_pal + agg_disc_to_pal,
            # General state
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            # Tracking
            'act_neuronal_groups': [list(g) for g in act_groups],
            'agg_neuronal_groups': [list(g) for g in agg_groups],
            'cross_synergy_pairs': cross_synergy,
            'selection_events': selection_events,
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
            # Phase info
            'act_phase': act_phase,
            'agg_phase': agg_phase,
            # Affinity stats
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            # Sin and extreme agg
            'has_sin': has_sin,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'has_extreme_agg': has_extreme_agg,
            'extreme_agg_affinities': [float(new_agg_aff[idx]) for idx in CORE_EXTREME_AGGS],
            # Neural Darwinism stats
            'n_act_groups': len(act_groups),
            'n_agg_groups': len(agg_groups),
            'n_cross_synergy_pairs': len(cross_synergy),
            'n_act_antagonistic': len(act_antag_pairs),
            'n_agg_antagonistic': len(agg_antag_pairs),
            'selection_events': selection_events,
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
            # Phases
            'act_phase': state['act_phase'],
            'agg_phase': state['agg_phase'],
            # Affinities
            'act_mean_affinity': float(jnp.mean(state['act_affinity'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinity'])),
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            # Groups
            'act_neuronal_groups': state['act_neuronal_groups'],
            'agg_neuronal_groups': state['agg_neuronal_groups'],
            'cross_synergy_pairs': state['cross_synergy_pairs'],
            'selection_events': state['selection_events'],
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
