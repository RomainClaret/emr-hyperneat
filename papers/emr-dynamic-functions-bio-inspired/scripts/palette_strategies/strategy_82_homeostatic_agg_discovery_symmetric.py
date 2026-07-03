"""Strategy 82S: Homeostatic Aggregation Discovery Symmetric (Active Balance with Discovery).

Biological Basis: Homeostatic plasticity maintains stable activity balance while
allowing discovery of new functions. Like the brain maintains firing rate
homeostasis while learning, this strategy balances exploration with stability.

Key mechanisms for SYMMETRIC discovery (both activation AND aggregation):
1. Active balance between averaging and extreme aggregations
2. Discovery bonus for underrepresented categories in BOTH domains
3. Cross-domain affinity tracking for sin-extreme pairs
4. Bidirectional homeostatic correction
5. Protected indices (0.1% deactivation) for sin and extreme aggregations
6. Memory cell crystallization for stable high-affinity functions
7. Affinity floors for guaranteed retention

Expected: Better balanced discovery across both domains while maintaining stability.
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)

# Critical indices for retention
SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3

# Activation categories for homeostatic balance
# Periodic activations (sin, gaussian, absolute)
PERIODIC_ACTS = [4, 5, 6]  # sin, gaussian, absolute
# Linear/bounded activations
LINEAR_ACTS = [0, 1, 2, 3, 7, 8, 9]  # sigmoid, tanh, relu, linear, step, softsign, identity


class HomeostaticAggDiscoverySymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with homeostatic balance and active discovery.

    Discovers BOTH activation and aggregation functions through unified
    evolution with homeostatic balance. Maintains category balance while
    actively promoting discovery of underrepresented function types.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Bidirectional homeostatic correction (both domains)
    - Discovery tracking for research analysis
    """

    name = "homeostatic_agg_discovery_symmetric"
    description = "Symmetric: Homeostatic balance with bidirectional discovery"

    def __init__(
        self,
        # Balance parameters
        target_extreme_ratio: float = 0.60,
        target_periodic_ratio: float = 0.40,  # Target periodic activations
        imbalance_threshold: float = 0.15,
        correction_strength: float = 1.8,
        # Discovery parameters
        discovery_bonus: float = 0.5,
        exploration_rate: float = 0.20,
        # Mutation rates
        base_mutation_rate: float = 0.10,
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        # Protection
        extreme_protection: float = 0.6,
        sin_protection: float = 0.6,
        # Cross-domain
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # Affinity floors for guaranteed retention
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Memory cell parameters
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Stagnation
        stagnation_threshold: int = 5,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Homeostatic Aggregation Discovery Symmetric strategy.

        Args:
            target_extreme_ratio: Target ratio of extreme aggregations
            target_periodic_ratio: Target ratio of periodic activations
            imbalance_threshold: Threshold to trigger correction
            correction_strength: Strength of homeostatic correction
            discovery_bonus: Extra activation rate for underrepresented
            exploration_rate: Base exploration rate
            base_mutation_rate: Base mutation rate for activations
            base_activate_rate: Base activation probability for aggregations
            base_deactivate_rate: Base deactivation probability for aggregations
            extreme_protection: Deactivation reduction for extreme aggs
            sin_protection: Deactivation reduction for sin
            cross_learning_rate: Cross-domain affinity learning rate
            sin_extreme_affinity_boost: Initial sin-extreme affinity boost
            sin_affinity_floor: Minimum affinity for sin function
            extreme_agg_affinity_floor: Minimum affinity for max/min aggregations
            memory_formation_threshold: Affinity threshold for memory candidacy
            memory_formation_count: Sustained generations to become memory cell
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            initial_act_palette: Starting activation palette indices
            initial_agg_palette: Starting aggregation palette indices
        """
        # Balance
        self.target_extreme_ratio = target_extreme_ratio
        self.target_periodic_ratio = target_periodic_ratio
        self.imbalance_threshold = imbalance_threshold
        self.correction_strength = correction_strength

        # Discovery
        self.discovery_bonus = discovery_bonus
        self.exploration_rate = exploration_rate

        # Mutation
        self.base_mutation_rate = base_mutation_rate
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # Protection
        self.extreme_protection = extreme_protection
        self.sin_protection = sin_protection

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Memory cells
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Stagnation
        self.stagnation_threshold = stagnation_threshold

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial palettes
        # CRITICAL: Include sin (idx 4) and extreme aggs (idx 2,3) in initial palettes
        # This ensures they start active and can be protected from the beginning
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]  # tanh, sigmoid, relu, identity, sin
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]  # sum, mean, max, min

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with homeostasis and memory tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity with pre-boosted sin-extreme
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        cross_affinity = cross_affinity.at[SIN_IDX, MAX_IDX].set(0.5 + self.sin_extreme_affinity_boost)
        cross_affinity = cross_affinity.at[SIN_IDX, MIN_IDX].set(0.5 + self.sin_extreme_affinity_boost)

        # Individual affinities
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Apply affinity floors immediately
        act_affinity = act_affinity.at[SIN_IDX].set(
            max(self.sin_affinity_floor, float(act_affinity[SIN_IDX]))
        )
        for idx in [MAX_IDX, MIN_IDX]:
            agg_affinity = agg_affinity.at[idx].set(
                max(self.extreme_agg_affinity_floor, float(agg_affinity[idx]))
            )

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        discovery_gen = {
            'sin': -1,
            'max': -1,
            'min': -1,
        }

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinity': act_affinity,
            'agg_affinity': agg_affinity,
            'cross_affinity': cross_affinity,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Homeostasis tracking
            'extreme_ratio_history': [],
            'periodic_ratio_history': [],
            'corrections_applied': 0,
            'discoveries': {'averaging': 0, 'extreme': 0, 'periodic': 0, 'linear': 0},
            # Discovery tracking
            'discovery_gen': discovery_gen,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 82000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply minimum affinity floors for critical functions."""
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )
        new_agg = agg_affinity
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

    def _compute_extreme_ratio(self, agg_palette: List[int]) -> float:
        """Compute ratio of extreme aggregations in palette."""
        if len(agg_palette) == 0:
            return 0.0
        extreme_count = sum(1 for a in agg_palette if a in EXTREME_AGGS)
        return extreme_count / len(agg_palette)

    def _compute_periodic_ratio(self, act_palette: List[int]) -> float:
        """Compute ratio of periodic activations in palette."""
        if len(act_palette) == 0:
            return 0.0
        periodic_count = sum(1 for a in act_palette if a in PERIODIC_ACTS)
        return periodic_count / len(act_palette)

    def _compute_imbalance(
        self,
        act_palette: List[int],
        agg_palette: List[int],
    ) -> Tuple[float, str, float, str]:
        """Compute imbalance for both domains."""
        # Aggregation imbalance
        extreme_ratio = self._compute_extreme_ratio(agg_palette)
        agg_imbalance = extreme_ratio - self.target_extreme_ratio
        if agg_imbalance > 0:
            agg_direction = 'extreme'
            agg_imbalance = agg_imbalance
        else:
            agg_direction = 'averaging'
            agg_imbalance = -agg_imbalance

        # Activation imbalance
        periodic_ratio = self._compute_periodic_ratio(act_palette)
        act_imbalance = periodic_ratio - self.target_periodic_ratio
        if act_imbalance > 0:
            act_direction = 'periodic'
            act_imbalance = act_imbalance
        else:
            act_direction = 'linear'
            act_imbalance = -act_imbalance

        return agg_imbalance, agg_direction, act_imbalance, act_direction

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity with sin-extreme boost."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        delta = self.cross_learning_rate * fitness_delta * co_active
        new_cross = cross_affinity + delta

        # Extra sin-extreme learning
        if act_mask[SIN_IDX] > 0.5 and fitness_delta > 0:
            for agg_idx in CORE_EXTREME_AGGS:
                if agg_mask[agg_idx] > 0.5:
                    boost = self.cross_learning_rate * fitness_delta * 1.5
                    new_cross = new_cross.at[SIN_IDX, agg_idx].set(
                        new_cross[SIN_IDX, agg_idx] + boost
                    )

        return jnp.clip(new_cross, 0.0, 1.0)

    def _update_affinities(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update individual affinities based on cross-domain success."""
        if fitness_delta <= 0:
            return act_affinity, agg_affinity

        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)

        # Boost affinities for active functions on success
        lr = self.cross_learning_rate

        # Activation affinity from cross-domain success
        cross_boost_act = jnp.sum(cross_affinity * active_agg[None, :], axis=1)
        cross_boost_act = cross_boost_act / max(1.0, float(jnp.sum(active_agg)))
        new_act_aff = act_affinity + lr * fitness_delta * cross_boost_act * active_act

        # Aggregation affinity from cross-domain success
        cross_boost_agg = jnp.sum(cross_affinity.T * active_act[None, :], axis=1)
        cross_boost_agg = cross_boost_agg / max(1.0, float(jnp.sum(active_act)))
        new_agg_aff = agg_affinity + lr * fitness_delta * cross_boost_agg * active_agg

        return jnp.clip(new_act_aff, 0.0, 1.0), jnp.clip(new_agg_aff, 0.0, 1.0)

    def _mutate_act_palette_homeostatic(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        act_imbalance: float,
        act_direction: str,
        cross_affinity: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply homeostatic mutation to activation palette."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        correction_applied = False

        # Compute per-activation rates
        activate_rates = jnp.ones(NUM_ACTIVATIONS) * self.base_mutation_rate
        deactivate_rates = jnp.ones(NUM_ACTIVATIONS) * self.base_mutation_rate

        # Discovery bonus for underrepresented category
        if act_imbalance > self.imbalance_threshold:
            correction_applied = True
            if act_direction == 'periodic':
                # Too many periodic, boost linear discovery
                for i in LINEAR_ACTS:
                    if mask[i] < 0.5:
                        activate_rates = activate_rates.at[i].set(
                            self.base_mutation_rate + self.discovery_bonus
                        )
            else:
                # Too few periodic, boost periodic discovery
                for i in PERIODIC_ACTS:
                    if mask[i] < 0.5:
                        activate_rates = activate_rates.at[i].set(
                            self.base_mutation_rate + self.discovery_bonus
                        )

        # Protected: sin gets very low deactivation (0.1%)
        deactivate_rates = deactivate_rates.at[SIN_IDX].set(0.001)

        # Memory cells get reduced deactivation
        for i in range(NUM_ACTIVATIONS):
            if memory_cells[i] and mask[i] > 0.5:
                deactivate_rates = deactivate_rates.at[i].set(
                    deactivate_rates[i] * (1 - self.sin_protection)
                )

        # Affinity-based protection
        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                protection = float(affinity[i]) * self.sin_protection
                deactivate_rates = deactivate_rates.at[i].set(
                    deactivate_rates[i] * (1 - protection)
                )

        for i in range(NUM_ACTIVATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            if mask[i] < 0.5:  # Inactive
                if p < activate_rates[i]:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                if p < deactivate_rates[i]:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_act or active_count > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []
            correction_applied = False

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'correction_applied': correction_applied,
        }

    def _mutate_agg_palette_homeostatic(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        agg_imbalance: float,
        agg_direction: str,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply homeostatic mutation to aggregation palette."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        correction_applied = False

        # Compute per-aggregation rates
        activate_rates = jnp.ones(NUM_AGGREGATIONS) * self.base_activate_rate
        deactivate_rates = jnp.ones(NUM_AGGREGATIONS) * self.base_deactivate_rate

        # Discovery bonus for underrepresented category
        if agg_imbalance > self.imbalance_threshold:
            correction_applied = True
            if agg_direction == 'extreme':
                # Too many extreme, boost averaging discovery
                for i in AVERAGING_AGGS:
                    if mask[i] < 0.5:
                        activate_rates = activate_rates.at[i].set(
                            self.base_activate_rate + self.discovery_bonus
                        )
            else:
                # Too few extreme, boost extreme discovery
                for i in EXTREME_AGGS:
                    if mask[i] < 0.5:
                        activate_rates = activate_rates.at[i].set(
                            self.base_activate_rate + self.discovery_bonus
                        )

        # Protected: extreme aggs get very low deactivation (0.1%)
        for idx in [MAX_IDX, MIN_IDX]:
            deactivate_rates = deactivate_rates.at[idx].set(0.001)

        # Apply protection to extreme aggregations based on sin affinity
        sin_active = act_mask[SIN_IDX] > 0.5
        for i in CORE_EXTREME_AGGS:
            if mask[i] > 0.5:
                protection = self.extreme_protection
                if sin_active:
                    sin_aff = float(cross_affinity[SIN_IDX, i])
                    protection += sin_aff * 0.2
                # Already set to 0.001 for max/min, keep affinity protection for others
                if i not in [MAX_IDX, MIN_IDX]:
                    deactivate_rates = deactivate_rates.at[i].set(
                        self.base_deactivate_rate * (1 - protection)
                    )

        # Memory cells get reduced deactivation
        for i in range(NUM_AGGREGATIONS):
            if memory_cells[i] and mask[i] > 0.5 and i not in [MAX_IDX, MIN_IDX]:
                deactivate_rates = deactivate_rates.at[i].set(
                    deactivate_rates[i] * (1 - self.extreme_protection)
                )

        for i in range(NUM_AGGREGATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            if mask[i] < 0.5:
                if p < activate_rates[i]:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if p < deactivate_rates[i]:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []
            correction_applied = False

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'correction_applied': correction_applied,
        }

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
        """Update with homeostatic discovery mechanism."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Update individual affinities
        new_act_aff, new_agg_aff = self._update_affinities(
            state['act_affinity'],
            state['agg_affinity'],
            new_cross,
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_aff, state['act_memory_counts'], state['act_memory_cells'], state['act_mask']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_aff, state['agg_memory_counts'], state['agg_memory_cells'], state['agg_mask']
        )

        # Compute homeostasis metrics
        act_palette = mask_to_indices(state['act_mask'])
        agg_palette = mask_to_indices(state['agg_mask'])
        agg_imbalance, agg_direction, act_imbalance, act_direction = self._compute_imbalance(
            act_palette, agg_palette
        )
        extreme_ratio = self._compute_extreme_ratio(agg_palette)
        periodic_ratio = self._compute_periodic_ratio(act_palette)

        extreme_ratio_history = state['extreme_ratio_history'] + [extreme_ratio]
        if len(extreme_ratio_history) > 20:
            extreme_ratio_history = extreme_ratio_history[-20:]

        periodic_ratio_history = state['periodic_ratio_history'] + [periodic_ratio]
        if len(periodic_ratio_history) > 20:
            periodic_ratio_history = periodic_ratio_history[-20:]

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None
        act_correction_applied = False
        agg_correction_applied = False
        new_corrections = state['corrections_applied']
        new_discoveries = dict(state['discoveries'])

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_act_palette_homeostatic(
                k_act, state['act_mask'], new_act_aff,
                act_imbalance, act_direction, new_cross, new_act_mem_cells
            )
            new_agg_mask, agg_mutation_info = self._mutate_agg_palette_homeostatic(
                k_agg, state['agg_mask'], new_agg_aff,
                agg_imbalance, agg_direction, new_cross, state['act_mask'], new_agg_mem_cells
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            act_correction_applied = act_mutation_info.get('correction_applied', False)
            agg_correction_applied = agg_mutation_info.get('correction_applied', False)
            if act_correction_applied or agg_correction_applied:
                new_corrections += 1

            # Track discoveries
            for idx in act_mutation_info.get('activated', []):
                if idx in PERIODIC_ACTS:
                    new_discoveries['periodic'] += 1
                else:
                    new_discoveries['linear'] += 1
            for idx in agg_mutation_info.get('activated', []):
                if idx in EXTREME_AGGS:
                    new_discoveries['extreme'] += 1
                else:
                    new_discoveries['averaging'] += 1

            new_stagnation = 0

        # Get current palettes
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Update discovery tracking
        new_discovery = self._update_discovery_tracking(
            state['discovery_gen'], act_palette, agg_palette, generation
        )

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinity': new_act_aff,
            'agg_affinity': new_agg_aff,
            'cross_affinity': new_cross,
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'extreme_ratio_history': extreme_ratio_history,
            'periodic_ratio_history': periodic_ratio_history,
            'corrections_applied': new_corrections,
            'discoveries': new_discoveries,
            'discovery_gen': new_discovery,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        # Compute metrics
        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'mutation_triggered': act_mutation_info is not None,
            # Homeostasis metrics
            'extreme_ratio': extreme_ratio,
            'periodic_ratio': periodic_ratio,
            'agg_imbalance': agg_imbalance,
            'agg_imbalance_direction': agg_direction,
            'act_imbalance': act_imbalance,
            'act_imbalance_direction': act_direction,
            'act_correction_applied': act_correction_applied,
            'agg_correction_applied': agg_correction_applied,
            'total_corrections': new_corrections,
            # Discovery metrics
            'periodic_discoveries': new_discoveries['periodic'],
            'linear_discoveries': new_discoveries['linear'],
            'extreme_discoveries': new_discoveries['extreme'],
            'averaging_discoveries': new_discoveries['averaging'],
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Cross-domain metrics
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'sin_max_affinity': float(new_cross[SIN_IDX, MAX_IDX]),
            'sin_min_affinity': float(new_cross[SIN_IDX, MIN_IDX]),
            # Sin status
            'has_sin': SIN_IDX in act_palette,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            # Discovery tracking
            'sin_discovered_gen': new_discovery['sin'],
            'max_discovered_gen': new_discovery['max'],
            'min_discovered_gen': new_discovery['min'],
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with homeostasis status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        extreme_ratio = self._compute_extreme_ratio(agg_palette)
        periodic_ratio = self._compute_periodic_ratio(act_palette)
        agg_imbalance, agg_direction, act_imbalance, act_direction = self._compute_imbalance(
            act_palette, agg_palette
        )

        act_mem_count = int(jnp.sum(state['act_memory_cells']))
        agg_mem_count = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            # Affinities
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'max_affinity': float(state['agg_affinity'][MAX_IDX]),
            'min_affinity': float(state['agg_affinity'][MIN_IDX]),
            # Homeostasis
            'extreme_ratio': extreme_ratio,
            'periodic_ratio': periodic_ratio,
            'agg_imbalance': agg_imbalance,
            'act_imbalance': act_imbalance,
            'total_corrections': state['corrections_applied'],
            # Discoveries
            'discoveries': state['discoveries'],
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Cross-domain
            'sin_max_affinity': float(state['cross_affinity'][SIN_IDX, MAX_IDX]),
            'sin_min_affinity': float(state['cross_affinity'][SIN_IDX, MIN_IDX]),
            # Discovery tracking
            'discovery_gen': state['discovery_gen'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
