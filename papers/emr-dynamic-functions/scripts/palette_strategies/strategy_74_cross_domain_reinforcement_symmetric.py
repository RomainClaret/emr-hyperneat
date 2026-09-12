"""Strategy 74S: Cross-Domain Reinforcement Symmetric (Neuromodulatory Cross-Modal Learning).

Biological Basis: Neuromodulatory reinforcement of cross-modal associations.
Dopaminergic signals reinforce synapses that were active during reward.
Cross-modal learning in the brain allows one sensory modality to guide
associations in another (e.g., hearing-vision associations).

Key mechanisms for SYMMETRIC discovery (both activation AND aggregation):
1. Cross-affinity matrix tracks which activation-aggregation pairs succeed together
2. Double learning rate when BOTH domains improve simultaneously
3. Sin-guided protection: When sin (idx 4) pairs successfully with extreme aggs,
   BOTH get protected from deactivation
4. Bidirectional reinforcement: Aggregation success boosts activation affinity too
5. Protected indices (0.1% deactivation) for sin and extreme aggregations

Expected: Enhanced cross-domain synergy through bidirectional affinity learning.
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
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)

# Critical indices for retention
SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3


class CrossDomainReinforcementSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with neuromodulatory cross-domain reinforcement.

    Discovers BOTH activation and aggregation functions through unified evolution
    with bidirectional cross-domain reinforcement. When both domains contribute
    to fitness improvement, learning is enhanced in both directions.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations
    - Bidirectional affinity updates (not just act->agg)
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Discovery tracking for research analysis
    """

    name = "cross_domain_reinforcement_symmetric"
    description = "Symmetric: Bidirectional neuromodulatory cross-domain reinforcement"

    def __init__(
        self,
        # Mutation rates
        base_mutation_rate: float = 0.12,
        stagnation_threshold: int = 5,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Cross-domain parameters
        base_cross_learning_rate: float = 0.15,
        reinforcement_multiplier: float = 2.0,  # When both domains contribute
        affinity_protection_threshold: float = 0.6,
        affinity_protection_strength: float = 0.5,
        # Affinity floors for guaranteed retention
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Memory cell parameters
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,  # Generations to become memory cell
        # Initial palettes - INCLUDE critical functions for retention
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Cross-Domain Reinforcement Symmetric strategy.

        Args:
            base_mutation_rate: Unified mutation rate for both domains
            stagnation_threshold: Generations without improvement before mutation
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions
            max_active_agg: Maximum active aggregation functions
            base_cross_learning_rate: Base cross-domain affinity learning rate
            reinforcement_multiplier: Learning rate multiplier when both domains contribute
            affinity_protection_threshold: Threshold for affinity-based protection
            affinity_protection_strength: Deactivation reduction for protected functions
            sin_affinity_floor: Minimum affinity for sin function
            extreme_agg_affinity_floor: Minimum affinity for max/min aggregations
            memory_formation_threshold: Affinity threshold for memory candidacy
            memory_formation_count: Sustained generations to become memory cell
            initial_act_palette: Starting activation palette indices (includes sin by default)
            initial_agg_palette: Starting aggregation palette indices (includes max/min by default)
        """
        self.base_mutation_rate = base_mutation_rate
        self.stagnation_threshold = stagnation_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.base_cross_learning_rate = base_cross_learning_rate
        self.reinforcement_multiplier = reinforcement_multiplier
        self.affinity_protection_threshold = affinity_protection_threshold
        self.affinity_protection_strength = affinity_protection_strength
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count
        # CRITICAL: Include sin (idx 4) and extreme aggs (idx 2,3) in initial palettes
        # This ensures they start active and can be protected from the beginning
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]  # tanh, sigmoid, relu, identity, sin
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]  # sum, mean, max, min

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with bidirectional cross-domain tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity matrix (activation x aggregation)
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Individual affinities for each domain
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
            'sin': -1,  # Generation when sin was discovered
            'max': -1,  # Generation when max was discovered
            'min': -1,  # Generation when min was discovered
        }

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'prev_act_mask': act_mask,
            'act_memory_counts': act_memory_counts,
            'act_memory_cells': act_memory_cells,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'prev_agg_mask': agg_mask,
            'agg_memory_counts': agg_memory_counts,
            'agg_memory_cells': agg_memory_cells,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Tracking
            'reinforcement_events': 0,
            'protection_events': 0,
            'discovery_gen': discovery_gen,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 74000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply minimum affinity floors for critical functions."""
        # Sin floor
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation floors
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
        """Update memory cell status based on sustained high affinity.

        Functions that maintain high affinity while active for sustained
        periods become memory cells with permanent protection.
        """
        active = mask > 0.5
        above_threshold = affinity >= self.memory_formation_threshold

        # Increment count for active, high-affinity functions
        candidate = active & above_threshold
        new_counts = jnp.where(candidate, memory_counts + 1, 0)

        # Functions that sustained threshold become memory cells
        newly_memory = new_counts >= self.memory_formation_count
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _update_cross_affinity_bidirectional(
        self,
        cross_affinity: jnp.ndarray,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        prev_act_mask: jnp.ndarray,
        prev_agg_mask: jnp.ndarray,
        fitness_improved: bool,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
        """Update cross-domain affinity bidirectionally with reinforcement.

        When both domains contribute to fitness improvement, learning is
        enhanced in both directions. This is the key mechanism for
        symmetric discovery.

        Returns:
            Tuple of (new_cross_affinity, new_act_affinity, new_agg_affinity, reinforced)
        """
        if not fitness_improved or fitness_delta <= 0:
            return cross_affinity, act_affinity, agg_affinity, False

        # Detect domain changes
        act_changed = not jnp.allclose(act_mask, prev_act_mask)
        agg_changed = not jnp.allclose(agg_mask, prev_agg_mask)

        # Compute learning rate with potential reinforcement
        if act_changed and agg_changed:
            lr = self.base_cross_learning_rate * self.reinforcement_multiplier
            reinforced = True
        else:
            lr = self.base_cross_learning_rate
            reinforced = False

        # Compute co-activation matrix
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        # Update cross-affinity matrix
        cross_delta = lr * fitness_delta * co_active
        new_cross = cross_affinity + cross_delta

        # BIDIRECTIONAL: Update individual affinities based on cross-affinity success
        # Activation affinity boosted by aggregation partners
        act_cross_boost = jnp.mean(new_cross * active_agg[None, :], axis=1)
        new_act_aff = act_affinity + lr * 0.5 * act_cross_boost * active_act

        # Aggregation affinity boosted by activation partners
        agg_cross_boost = jnp.mean(new_cross * active_act[:, None], axis=0)
        new_agg_aff = agg_affinity + lr * 0.5 * agg_cross_boost * active_agg

        return (
            jnp.clip(new_cross, 0.0, 1.0),
            jnp.clip(new_act_aff, 0.0, 1.0),
            jnp.clip(new_agg_aff, 0.0, 1.0),
            reinforced,
        )

    def _get_protected_functions(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_memory_cells: jnp.ndarray,
        agg_memory_cells: jnp.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Get functions protected by high cross-affinity or memory cell status.

        Protection sources:
        1. Sin (idx 4) when active with high cross-affinity to extreme aggs
        2. Extreme aggs (idx 2,3) when active with high cross-affinity to sin
        3. Memory cells are always protected
        """
        protected_act = []
        protected_agg = []

        act_palette = mask_to_indices(act_mask)
        agg_palette = mask_to_indices(agg_mask)

        # Sin protection based on cross-affinity with extreme aggs
        if SIN_IDX in act_palette:
            sin_extreme_affinity = max(
                float(cross_affinity[SIN_IDX, MAX_IDX]),
                float(cross_affinity[SIN_IDX, MIN_IDX]),
            )
            if sin_extreme_affinity >= self.affinity_protection_threshold:
                protected_act.append(SIN_IDX)
                # Also protect the extreme aggs that sin pairs with
                if cross_affinity[SIN_IDX, MAX_IDX] >= self.affinity_protection_threshold:
                    protected_agg.append(MAX_IDX)
                if cross_affinity[SIN_IDX, MIN_IDX] >= self.affinity_protection_threshold:
                    protected_agg.append(MIN_IDX)

        # Memory cells are always protected
        for idx in range(NUM_ACTIVATIONS):
            if act_memory_cells[idx] and idx not in protected_act:
                protected_act.append(idx)
        for idx in range(NUM_AGGREGATIONS):
            if agg_memory_cells[idx] and idx not in protected_agg:
                protected_agg.append(idx)

        return protected_act, protected_agg

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        min_active: int,
        max_active: int,
        protected_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply affinity-guided mutation with protection.

        Higher affinity functions have lower deactivation probability.
        Protected indices get very low (0.1%) deactivation rate.
        """
        protected_indices = protected_indices or []
        new_mask = mask.copy()

        # Compute per-function rates based on affinity
        n_funcs = len(mask)
        deactivation_rates = jnp.ones(n_funcs) * self.base_mutation_rate
        activation_rates = jnp.ones(n_funcs) * self.base_mutation_rate

        # High affinity reduces deactivation
        for i in range(n_funcs):
            if mask[i] > 0.5:
                protection_factor = 1.0 - (affinity[i] * self.affinity_protection_strength)
                deactivation_rates = deactivation_rates.at[i].set(
                    self.base_mutation_rate * protection_factor
                )

        # Protected indices get very low deactivation rate (0.1%)
        for idx in protected_indices:
            if idx < n_funcs:
                deactivation_rates = deactivation_rates.at[idx].set(0.001)

        # High affinity increases activation probability for inactive functions
        for i in range(n_funcs):
            if mask[i] < 0.5 and affinity[i] >= self.affinity_protection_threshold:
                activation_rates = activation_rates.at[i].set(
                    self.base_mutation_rate * (1.0 + self.affinity_protection_strength)
                )

        # Apply mutations
        flip_probs = jax.random.uniform(key, (n_funcs,))
        for i in range(n_funcs):
            if mask[i] > 0.5:  # Currently active
                if flip_probs[i] < deactivation_rates[i]:
                    new_mask = new_mask.at[i].set(0.0)
            else:  # Currently inactive
                if flip_probs[i] < activation_rates[i]:
                    new_mask = new_mask.at[i].set(1.0)

        # Track changes
        activated = []
        deactivated = []
        for i in range(n_funcs):
            if mask[i] < 0.5 and new_mask[i] > 0.5:
                activated.append(i)
            elif mask[i] > 0.5 and new_mask[i] < 0.5:
                deactivated.append(i)

        # Ensure constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < min_active or active_count > max_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'protected_count': len(protected_indices),
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
        """Update with bidirectional cross-domain reinforcement."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update cross-domain affinity bidirectionally
        new_cross, new_act_aff, new_agg_aff, reinforced = self._update_cross_affinity_bidirectional(
            state['cross_affinity'],
            state['act_affinity'],
            state['agg_affinity'],
            state['act_mask'],
            state['agg_mask'],
            state['prev_act_mask'],
            state['prev_agg_mask'],
            improved,
            fitness_delta,
        )
        new_reinforcement_events = state['reinforcement_events'] + (1 if reinforced else 0)

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_aff,
            state['act_memory_counts'],
            state['act_memory_cells'],
            state['act_mask'],
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_aff,
            state['agg_memory_counts'],
            state['agg_memory_cells'],
            state['agg_mask'],
        )

        # Initialize mutation tracking
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None
        new_protection_events = state['protection_events']

        # Mutate if stagnating
        if new_stagnation >= self.stagnation_threshold:
            # Get protected functions
            protected_act, protected_agg = self._get_protected_functions(
                new_cross,
                state['act_mask'],
                state['agg_mask'],
                new_act_mem_cells,
                new_agg_mem_cells,
            )

            # Add critical indices to protection
            protected_act = list(set(protected_act + [SIN_IDX]))
            protected_agg = list(set(protected_agg + [MAX_IDX, MIN_IDX]))

            # Mutate both palettes with protection
            new_act_mask, act_mutation_info = self._mutate_palette(
                k_act,
                state['act_mask'],
                new_act_aff,
                self.min_active_act,
                self.max_active_act,
                protected_indices=protected_act,
            )
            new_agg_mask, agg_mutation_info = self._mutate_palette(
                k_agg,
                state['agg_mask'],
                new_agg_aff,
                self.min_active_agg,
                self.max_active_agg,
                protected_indices=protected_agg,
            )

            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

            if len(protected_act) > 0 or len(protected_agg) > 0:
                new_protection_events += 1

            new_stagnation = 0

        # Get current palettes for tracking
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Update discovery tracking
        new_discovery = self._update_discovery_tracking(
            state['discovery_gen'],
            act_palette,
            agg_palette,
            generation,
        )

        new_state = {
            'act_mask': new_act_mask,
            'act_affinity': new_act_aff,
            'prev_act_mask': state['act_mask'],
            'act_memory_counts': new_act_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_aff,
            'prev_agg_mask': state['agg_mask'],
            'agg_memory_counts': new_agg_mem_counts,
            'agg_memory_cells': new_agg_mem_cells,
            'cross_affinity': new_cross,
            'reinforcement_events': new_reinforcement_events,
            'protection_events': new_protection_events,
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

        # Sin-specific affinity metrics
        sin_affinity = float(new_act_aff[SIN_IDX])
        sin_max_cross = float(new_cross[SIN_IDX, MAX_IDX])
        sin_min_cross = float(new_cross[SIN_IDX, MIN_IDX])

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'mutation_triggered': act_mutation_info is not None,
            # Cross-domain metrics
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # Reinforcement metrics
            'reinforced_this_gen': reinforced,
            'total_reinforcement_events': new_reinforcement_events,
            'total_protection_events': new_protection_events,
            # Memory cell metrics
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Sin-specific
            'sin_affinity': sin_affinity,
            'sin_max_cross': sin_max_cross,
            'sin_min_cross': sin_min_cross,
            # Status
            'has_sin': SIN_IDX in act_palette,
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
        """Return state summary with cross-domain status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        protected_act, protected_agg = self._get_protected_functions(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            state['act_memory_cells'],
            state['agg_memory_cells'],
        )

        act_mem_count = int(jnp.sum(state['act_memory_cells']))
        agg_mem_count = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': SIN_IDX in act_palette,
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            # Affinity metrics
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'max_affinity': float(state['agg_affinity'][MAX_IDX]),
            'min_affinity': float(state['agg_affinity'][MIN_IDX]),
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
            # Protection
            'protected_act': protected_act,
            'protected_agg': protected_agg,
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Events
            'reinforcement_events': state['reinforcement_events'],
            'protection_events': state['protection_events'],
            # Discovery
            'discovery_gen': state['discovery_gen'],
            # General
            'stagnation_count': state['stagnation_count'],
            'best_fitness_seen': state['best_fitness_seen'],
        }
