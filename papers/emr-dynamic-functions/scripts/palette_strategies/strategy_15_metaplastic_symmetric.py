"""Strategy 15 Symmetric: Metaplastic Symmetric.

Extends MetaplasticStrategy with symmetric discovery features:
- Dual BCM sliding thresholds (separate for activation/aggregation)
- Memory cell mechanism from Clonal Selection Symmetric
- Cross-domain threshold coupling
- Sin and extreme aggregation affinity floors
- Discovery tracking and boost for both domains

Key mechanisms:
1. Adaptive learning rates: Stagnation → boost LR, Success → reduce LR
2. Dual BCM thresholds: Separate sliding thresholds for act/agg
3. Memory cells: Functions with sustained high affinity get permanent protection
4. Cross-domain coupling: High act threshold → exploration boost for agg

Biological rationale:
- Metaplasticity: synaptic plasticity is itself plastic
- BCM theory: sliding threshold adapts to activity distribution
- Immunological memory: proven functions become memory cells
- Cross-modal plasticity: success in one domain influences another
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


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class MetaplasticSymmetricStrategy(PaletteEvolutionStrategy):
    """Metaplastic strategy with dual BCM thresholds and memory cells.

    Extends activation-only Metaplastic to discover both activation AND
    aggregation functions while maintaining retention through memory cells.

    Key innovations:
    - Dual BCM sliding thresholds for each domain
    - Memory cell mechanism for permanent protection of proven functions
    - Cross-domain threshold coupling (success in one domain helps other)
    - Affinity floors prevent loss of critical functions
    """

    name = "metaplastic_symmetric"
    description = "Metaplastic BCM with dual thresholds and memory cells"

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
        # Base learning parameters
        base_learning_rate: float = 0.18,
        base_anti_hebbian_rate: float = 0.05,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Metaplastic parameters
        stagnation_lr_boost: float = 1.5,
        stagnation_threshold_gens: int = 10,
        max_stagnation_boost: float = 3.0,
        success_lr_reduction: float = 0.7,
        success_window: int = 5,
        success_threshold: float = 0.8,
        # Dual BCM sliding thresholds
        act_threshold_adaptation_rate: float = 0.05,
        act_threshold_min: float = 0.40,
        act_threshold_max: float = 0.70,
        act_threshold_percentile: float = 0.70,
        agg_threshold_adaptation_rate: float = 0.06,
        agg_threshold_min: float = 0.35,
        agg_threshold_max: float = 0.65,
        agg_threshold_percentile: float = 0.65,
        # Cross-domain coupling
        cross_threshold_coupling: float = 0.15,  # How much act threshold affects agg exploration
        cross_learning_rate: float = 0.05,
        # Memory cell parameters (from Clonal Selection)
        memory_cell_threshold: float = 0.75,
        memory_cell_gens: int = 10,
        memory_cell_decay_rate: float = 0.05,  # Only 5% decay for memory cells
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Other
        early_consolidation_threshold: float = 0.95,
        act_min_active: int = 2,
        act_max_active: int = 6,
        agg_min_active: int = 1,
        agg_max_active: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Metaplastic Symmetric strategy."""
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

        # Base learning
        self.base_learning_rate = base_learning_rate
        self.base_anti_hebbian_rate = base_anti_hebbian_rate
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Metaplastic
        self.stagnation_lr_boost = stagnation_lr_boost
        self.stagnation_threshold_gens = stagnation_threshold_gens
        self.max_stagnation_boost = max_stagnation_boost
        self.success_lr_reduction = success_lr_reduction
        self.success_window = success_window
        self.success_threshold = success_threshold

        # Dual BCM thresholds
        self.act_threshold_adaptation_rate = act_threshold_adaptation_rate
        self.act_threshold_min = act_threshold_min
        self.act_threshold_max = act_threshold_max
        self.act_threshold_percentile = act_threshold_percentile
        self.agg_threshold_adaptation_rate = agg_threshold_adaptation_rate
        self.agg_threshold_min = agg_threshold_min
        self.agg_threshold_max = agg_threshold_max
        self.agg_threshold_percentile = agg_threshold_percentile

        # Cross-domain
        self.cross_threshold_coupling = cross_threshold_coupling
        self.cross_learning_rate = cross_learning_rate

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
        """Initialize state with dual affinities and memory cell tracking."""
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

        # Hebbian weight matrices
        act_hebbian = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        agg_hebbian = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        cross_hebbian = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

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
            'act_hebbian': act_hebbian,
            'act_phase': CriticalPeriodPhase.EXPLORATION,
            'act_protection_threshold': 0.55,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_hebbian': agg_hebbian,
            'agg_phase': CriticalPeriodPhase.EXPLORATION,
            'agg_protection_threshold': 0.50,
            # Cross-domain
            'cross_hebbian': cross_hebbian,
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
            'rng_key': jax.random.PRNGKey(seed + 151531),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            'fitness_ema': 0.5,
            'improvement_history': [],
            'current_lr_multiplier': 1.0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _compute_lr_multiplier(
        self,
        stagnation_count: int,
        improvement_history: List[bool],
        phase: str,
    ) -> float:
        """Compute metaplastic learning rate multiplier."""
        # Phase base multiplier
        if phase == CriticalPeriodPhase.EXPLORATION:
            phase_mult = self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            phase_mult = self.confirmation_lr_multiplier
        else:
            phase_mult = 0.1

        # Stagnation boost
        if stagnation_count >= self.stagnation_threshold_gens:
            stagnation_factor = stagnation_count / self.stagnation_threshold_gens
            stagnation_boost = min(
                self.max_stagnation_boost,
                1.0 + (self.stagnation_lr_boost - 1.0) * stagnation_factor
            )
        else:
            stagnation_boost = 1.0

        # Success reduction
        if len(improvement_history) >= self.success_window:
            recent = improvement_history[-self.success_window:]
            improvement_rate = sum(recent) / len(recent)
            if improvement_rate >= self.success_threshold:
                success_mult = self.success_lr_reduction
            else:
                success_mult = 1.0
        else:
            success_mult = 1.0

        return phase_mult * stagnation_boost * success_mult

    def _update_bcm_threshold(
        self,
        current_threshold: float,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        adaptation_rate: float,
        threshold_min: float,
        threshold_max: float,
        threshold_percentile: float,
        cross_threshold_effect: float = 0.0,
    ) -> float:
        """Update BCM sliding threshold with cross-domain coupling."""
        valid_affinities = [float(affinity[i]) for i in range(len(affinity)) if mask[i] > 0.5]

        if len(valid_affinities) < 2:
            return current_threshold

        # Compute target as percentile
        target = float(np.percentile(valid_affinities, threshold_percentile * 100))

        # Apply cross-domain effect (high other-domain threshold → lower exploration barrier)
        target = target - cross_threshold_effect * self.cross_threshold_coupling

        # Clip to bounds
        target = max(threshold_min, min(threshold_max, target))

        # Smooth adaptation
        new_threshold = (1 - adaptation_rate) * current_threshold + adaptation_rate * target

        return new_threshold

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

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        lr_multiplier: float,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update Hebbian weights with memory cell protection and discovery boost."""
        newly_discovered = newly_discovered or []
        lr = self.base_learning_rate * lr_multiplier
        anti_lr = self.base_anti_hebbian_rate * lr_multiplier

        active = (mask > 0.5).astype(jnp.float32)

        # Weight update
        co_active = jnp.outer(active, active)
        if fitness_signal >= 0:
            weight_delta = lr * fitness_signal * co_active
        else:
            weight_delta = anti_lr * fitness_signal * co_active

        new_weights = jnp.clip(weights + weight_delta, 0.0, 1.0)

        # Affinity update
        if fitness_signal >= 0:
            affinity_delta = lr * fitness_signal * active
        else:
            affinity_delta = anti_lr * fitness_signal * active

        new_affinity = affinity + affinity_delta

        # Memory cells resist negative changes
        negative_delta = affinity_delta < 0
        memory_protected = jnp.logical_and(negative_delta, memory_cells)
        new_affinity = jnp.where(
            memory_protected,
            affinity * (1 - self.memory_cell_decay_rate),  # Only 5% decay
            new_affinity
        )

        # Discovery boost
        for idx in newly_discovered:
            new_affinity = new_affinity.at[idx].set(
                new_affinity[idx] + self.discovery_boost
            )

        return new_weights, jnp.clip(new_affinity, 0.0, 1.0)

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for critical functions."""
        # Sin affinity floor
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation floors
        new_agg = agg_affinity
        for idx in CORE_EXTREME_AGGS:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )

        return new_act, new_agg

    def _compute_protection_scores(
        self,
        affinity: jnp.ndarray,
        hebbian: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection score with memory cell bonus."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        pairwise_score = jnp.dot(hebbian, active) / n_active
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
        protection_threshold: float,
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
            activate_rate = exploration_activate
            use_protection = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = confirmation_activate
            use_protection = True
        else:
            activate_rate = consolidation_activate
            use_protection = True

        for i in range(n_funcs):
            protection = float(protection_scores[i])
            is_memory = bool(memory_cells[i])

            if mask[i] < 0.5:
                # Activation logic
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
                # Deactivation logic - memory cells are protected
                if is_memory:
                    continue  # Memory cells never deactivate

                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= protection_threshold:
                        continue
                    deact_rate = consolidation_deactivate

                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= protection_threshold:
                        deact_rate = confirmation_deactivate_min
                    else:
                        t = protection / protection_threshold
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
            discovery_to_palette = 0

        # Discovery slot guarantee
        if self.enable_discovery_slot and newly_discovered:
            current_active = int(jnp.sum(new_mask > 0.5))
            not_in_palette = [idx for idx in newly_discovered if new_mask[idx] < 0.5]
            if not_in_palette and current_active < max_active:
                # Add best discovery
                best_new = max(not_in_palette, key=lambda j: float(protection_scores[j]))
                new_mask = new_mask.at[best_new].set(1.0)
                discovery_to_palette += 1

        return new_mask, {'activated': activated, 'deactivated': deactivated}, discovery_to_palette

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual metaplastic learning and memory cells."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update phases for each domain
        act_phase = self._get_phase(
            generation, new_best,
            self.act_exploration_end, self.act_confirmation_end
        )
        agg_phase = self._get_phase(
            generation, new_best,
            self.agg_exploration_end, self.agg_confirmation_end
        )

        # Update improvement history
        improvement_history = state['improvement_history'] + [improved]
        if len(improvement_history) > self.success_window * 2:
            improvement_history = improvement_history[-self.success_window * 2:]

        # Fitness signal
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Compute LR multiplier (use most active phase)
        active_phase = act_phase if generation < self.act_confirmation_end else agg_phase
        lr_multiplier = self._compute_lr_multiplier(
            new_stagnation, improvement_history, active_phase
        )

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

        # Hebbian updates for both domains
        new_act_hebbian, new_act_affinity = self._hebbian_update(
            state['act_hebbian'], state['act_affinity'], state['act_mask'],
            fitness_signal, lr_multiplier, state['act_memory_cells'],
            act_new_candidates
        )
        new_agg_hebbian, new_agg_affinity = self._hebbian_update(
            state['agg_hebbian'], state['agg_affinity'], state['agg_mask'],
            fitness_signal, lr_multiplier, state['agg_memory_cells'],
            agg_new_candidates
        )

        # Cross-domain Hebbian update
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_delta = self.cross_learning_rate * fitness_signal * jnp.outer(act_active, agg_active)
        new_cross_hebbian = jnp.clip(state['cross_hebbian'] + cross_delta, 0.0, 1.0)

        # Apply affinity floors
        new_act_affinity, new_agg_affinity = self._apply_affinity_floors(
            new_act_affinity, new_agg_affinity
        )

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_affinity, state['act_memory_counts'], state['act_memory_cells']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_affinity, state['agg_memory_counts'], state['agg_memory_cells']
        )

        # Update BCM thresholds with cross-domain coupling
        act_threshold_effect = float(state['agg_protection_threshold'] - 0.5)
        new_act_threshold = self._update_bcm_threshold(
            state['act_protection_threshold'], new_act_affinity, state['act_mask'],
            self.act_threshold_adaptation_rate, self.act_threshold_min,
            self.act_threshold_max, self.act_threshold_percentile,
            act_threshold_effect
        )

        agg_threshold_effect = float(state['act_protection_threshold'] - 0.5)
        new_agg_threshold = self._update_bcm_threshold(
            state['agg_protection_threshold'], new_agg_affinity, state['agg_mask'],
            self.agg_threshold_adaptation_rate, self.agg_threshold_min,
            self.agg_threshold_max, self.agg_threshold_percentile,
            agg_threshold_effect
        )

        # Compute protection scores
        act_protection = self._compute_protection_scores(
            new_act_affinity, new_act_hebbian, state['act_mask'], new_act_mem_cells
        )
        agg_protection = self._compute_protection_scores(
            new_agg_affinity, new_agg_hebbian, state['agg_mask'], new_agg_mem_cells
        )

        # Mutate palettes
        new_act_mask, act_mut_info, act_disc_to_pal = self._mutate_palette(
            k1, state['act_mask'], act_phase, act_protection, new_act_threshold,
            new_act_mem_cells,
            self.act_exploration_activate, self.act_exploration_deactivate,
            self.act_confirmation_activate, self.act_confirmation_deactivate_max,
            self.act_confirmation_deactivate_min, self.act_consolidation_activate,
            self.act_consolidation_deactivate, self.act_min_active, self.act_max_active,
            act_new_candidates
        )

        new_agg_mask, agg_mut_info, agg_disc_to_pal = self._mutate_palette(
            k2, state['agg_mask'], agg_phase, agg_protection, new_agg_threshold,
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

        new_state = {
            # Activation domain
            'act_mask': new_act_mask,
            'act_affinity': new_act_affinity,
            'act_hebbian': new_act_hebbian,
            'act_phase': act_phase,
            'act_protection_threshold': new_act_threshold,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_affinity,
            'agg_hebbian': new_agg_hebbian,
            'agg_phase': agg_phase,
            'agg_protection_threshold': new_agg_threshold,
            # Cross-domain
            'cross_hebbian': new_cross_hebbian,
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
            'improvement_history': improvement_history,
            'current_lr_multiplier': lr_multiplier,
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
            'act_mean_affinity': float(jnp.mean(new_act_affinity)),
            'agg_mean_affinity': float(jnp.mean(new_agg_affinity)),
            # Threshold stats
            'act_protection_threshold': new_act_threshold,
            'agg_protection_threshold': new_agg_threshold,
            # Sin and extreme agg
            'has_sin': has_sin,
            'sin_affinity': float(new_act_affinity[SIN_IDX]),
            'has_extreme_agg': has_extreme_agg,
            'extreme_agg_affinities': [float(new_agg_affinity[idx]) for idx in CORE_EXTREME_AGGS],
            # Metaplastic
            'lr_multiplier': lr_multiplier,
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
            # Thresholds
            'act_protection_threshold': state['act_protection_threshold'],
            'agg_protection_threshold': state['agg_protection_threshold'],
            # Affinities
            'act_mean_affinity': float(jnp.mean(state['act_affinity'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinity'])),
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            # Metaplastic
            'current_lr_multiplier': state['current_lr_multiplier'],
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
