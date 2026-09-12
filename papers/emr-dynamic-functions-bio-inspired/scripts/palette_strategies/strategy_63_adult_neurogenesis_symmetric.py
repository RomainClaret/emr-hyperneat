"""Strategy 63S: Adult Neurogenesis Symmetric (Birth-Maturation-Survival for Both Domains).

Biological Basis: Hippocampal neurogenesis - new neurons are born, mature through a
critical period, and survive or die based on contribution to task performance.
Young neurons have enhanced plasticity, while mature neurons provide stability.

Key symmetric mechanisms:
1. Dual stable/young tracking - separate pools per domain
2. Coordinated birth - new neurons can be born in both domains
3. Cross-domain maturation boost - success in partner domain helps survival
4. Protected indices for sin and extreme aggregations (0.1% deactivation)
5. Affinity floors for guaranteed retention
6. Memory cell crystallization for proven functions
7. Initial palettes include critical functions from start

Sin and extreme aggregations are treated specially:
- Lower survival threshold (easier to survive)
- Protected from random deactivation
- Affinity floors prevent loss

Expected: Controlled exploration with survival-based integration in both domains,
with guaranteed retention of critical functions.
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
    CORE_EXTREME_AGGS,
)

# Critical indices for retention
SIN_IDX = 4
MAX_IDX = 2
MIN_IDX = 3


class AdultNeurogenesisSymmetricStrategy(PaletteEvolutionStrategy):
    """Symmetric palette evolution with hippocampal-inspired neurogenesis.

    Both activation and aggregation functions can be born, mature,
    and survive or be pruned based on their contribution to fitness.

    Key improvements over dual version:
    - Protected indices for sin and extreme aggregations (0.1% deactivation)
    - Affinity floors for guaranteed retention
    - Memory cell crystallization for proven functions
    - Initial palettes include sin and max/min from start
    - Critical functions get survival threshold discount
    """

    name = "adult_neurogenesis_symmetric"
    description = "Symmetric: Neurogenesis with protected indices and memory cells"

    def __init__(
        self,
        # Neurogenesis
        neurogenesis_rate: float = 0.08,
        maturation_period: int = 10,
        young_plasticity: float = 2.0,
        survival_threshold: float = 0.1,
        max_young_act: int = 3,
        max_young_agg: int = 2,
        # Contribution
        contribution_decay: float = 0.9,
        contribution_boost: float = 0.3,
        # Cross-domain
        cross_survival_boost: float = 0.15,
        cross_learning_rate: float = 0.05,
        # Stable palette
        stable_mutation_rate: float = 0.02,
        max_stable_act: int = 8,
        max_stable_agg: int = 4,
        min_stable_act: int = 2,
        min_stable_agg: int = 1,
        # Protected function thresholds
        protected_survival_discount: float = 0.5,  # 50% lower threshold for sin/extremes
        protected_deactivation_prob: float = 0.001,  # 0.1% chance to deactivate
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Memory cell parameters
        memory_formation_threshold: float = 0.75,
        memory_formation_count: int = 8,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Adult Neurogenesis Symmetric strategy."""
        # Neurogenesis
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.young_plasticity = young_plasticity
        self.survival_threshold = survival_threshold
        self.max_young_act = max_young_act
        self.max_young_agg = max_young_agg

        # Contribution
        self.contribution_decay = contribution_decay
        self.contribution_boost = contribution_boost

        # Cross-domain
        self.cross_survival_boost = cross_survival_boost
        self.cross_learning_rate = cross_learning_rate

        # Stable
        self.stable_mutation_rate = stable_mutation_rate
        self.max_stable_act = max_stable_act
        self.max_stable_agg = max_stable_agg
        self.min_stable_act = min_stable_act
        self.min_stable_agg = min_stable_agg

        # Protected function settings
        self.protected_survival_discount = protected_survival_discount
        self.protected_deactivation_prob = protected_deactivation_prob

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Memory cells
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_formation_count = memory_formation_count

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        # CRITICAL: Include sin and extreme aggregations in initial palettes
        self.initial_act_palette = initial_act_palette or [0, 1, 2, 3, SIN_IDX]
        self.initial_agg_palette = initial_agg_palette or [0, 1, MAX_IDX, MIN_IDX]

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual neurogenesis tracking."""
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

        # Initialize contributions
        act_contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_contribution = act_contribution.at[i].set(0.5)
        # Give sin a head start
        act_contribution = act_contribution.at[SIN_IDX].set(0.7)

        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.5)
        # Give extremes a head start
        agg_contribution = agg_contribution.at[MAX_IDX].set(0.6)
        agg_contribution = agg_contribution.at[MIN_IDX].set(0.6)

        # Initialize affinities with floors
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        act_affinity = act_affinity.at[SIN_IDX].set(self.sin_affinity_floor)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_affinity = agg_affinity.at[MAX_IDX].set(self.extreme_agg_affinity_floor)
        agg_affinity = agg_affinity.at[MIN_IDX].set(self.extreme_agg_affinity_floor)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

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
            # Activation domain
            'act_mask': act_mask,
            'act_stable': set(initial_act),
            'act_young': {},
            'act_contribution': act_contribution,
            'act_affinity': act_affinity,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_stable': set(initial_agg),
            'agg_young': {},
            'agg_contribution': agg_contribution,
            'agg_affinity': agg_affinity,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Discovery
            'discovery_gen': discovery_gen,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 636363),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Stats
            'act_total_births': 0,
            'act_total_survivals': 0,
            'act_total_prunings': 0,
            'agg_total_births': 0,
            'agg_total_survivals': 0,
            'agg_total_prunings': 0,
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

    def _get_effective_survival_threshold(self, func_idx: int, is_activation: bool) -> float:
        """Get survival threshold with discount for protected indices."""
        base = self.survival_threshold

        if is_activation and func_idx == SIN_IDX:
            return base * self.protected_survival_discount
        elif not is_activation and func_idx in [MAX_IDX, MIN_IDX]:
            return base * self.protected_survival_discount

        return base

    def _maybe_birth_neuron(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        key: jax.random.PRNGKey,
        generation: int,
        max_young: int,
        n_funcs: int,
        protected_indices: List[int],
    ) -> Tuple[Set[int], Dict[int, Dict], Optional[int]]:
        """Possibly birth a new neuron, preferring protected indices."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_young = dict(young)
        born = None

        if len(new_young) >= max_young:
            return stable, new_young, born

        if jax.random.uniform(key1) < self.neurogenesis_rate:
            # First priority: protected indices not yet in palette
            priority_available = [
                i for i in protected_indices
                if i not in stable and i not in new_young and i < n_funcs
            ]

            if priority_available and jax.random.uniform(key3) < 0.7:
                # 70% chance to birth a protected function if available
                idx = int(jax.random.randint(key2, (), 0, len(priority_available)))
                new_func = priority_available[idx]
                new_young[new_func] = {'birth_gen': generation, 'contribution': 0.0}
                born = new_func
            else:
                # Regular birth from any available
                available = [
                    i for i in range(n_funcs)
                    if i not in stable and i not in new_young
                ]
                if available:
                    idx = int(jax.random.randint(key2, (), 0, len(available)))
                    new_func = available[idx]
                    new_young[new_func] = {'birth_gen': generation, 'contribution': 0.0}
                    born = new_func

        return stable, new_young, born

    def _mature_neurons(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        contribution: jnp.ndarray,
        affinity: jnp.ndarray,
        memory_cells: jnp.ndarray,
        partner_mean_contrib: float,
        generation: int,
        max_stable: int,
        is_activation: bool,
        key: jax.random.PRNGKey,
    ) -> Tuple[Set[int], Dict[int, Dict], List[int], List[int]]:
        """Process neuron maturation with protected index handling."""
        new_stable = set(stable)
        new_young = {}
        survived = []
        pruned = []

        keys = jax.random.split(key, len(young) + 1)
        key_idx = 0

        for func, info in young.items():
            age = generation - info['birth_gen']
            if age >= self.maturation_period:
                func_contrib = float(contribution[func])
                func_affinity = float(affinity[func])

                # Get effective threshold (lower for protected indices)
                effective_threshold = self._get_effective_survival_threshold(
                    func, is_activation
                )

                # Cross-domain boost
                effective_threshold -= partner_mean_contrib * self.cross_survival_boost

                # Memory cells always survive
                if memory_cells[func]:
                    if len(new_stable) < max_stable:
                        new_stable.add(func)
                        survived.append(func)
                    else:
                        new_young[func] = info  # Keep as young if can't stabilize
                # Protected indices get extra protection
                elif self._is_protected(func, is_activation):
                    # Only prune protected with very low probability
                    if jax.random.uniform(keys[key_idx]) < self.protected_deactivation_prob:
                        pruned.append(func)
                    else:
                        # Automatic survival for protected
                        if len(new_stable) < max_stable:
                            new_stable.add(func)
                            survived.append(func)
                        else:
                            new_young[func] = info
                    key_idx += 1
                # Regular maturation
                elif func_contrib >= effective_threshold or func_affinity >= 0.6:
                    if len(new_stable) < max_stable:
                        new_stable.add(func)
                        survived.append(func)
                    else:
                        pruned.append(func)
                else:
                    pruned.append(func)
            else:
                new_young[func] = info

        return new_stable, new_young, survived, pruned

    def _is_protected(self, func_idx: int, is_activation: bool) -> bool:
        """Check if a function index is protected."""
        if is_activation:
            return func_idx == SIN_IDX
        else:
            return func_idx in [MAX_IDX, MIN_IDX]

    def _update_contributions(
        self,
        contribution: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        young: Dict[int, Dict],
        improved: bool,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[int, Dict]]:
        """Update contribution and affinity tracking."""
        new_contribution = contribution * self.contribution_decay
        new_affinity = affinity * 0.95
        new_young = {}

        for i in range(n_funcs):
            if mask[i] > 0.5:
                current_contrib = float(new_contribution[i])
                current_aff = float(new_affinity[i])

                if improved:
                    boost = self.contribution_boost
                    if i in young:
                        boost *= self.young_plasticity
                    new_contribution = new_contribution.at[i].set(current_contrib + boost)
                    new_affinity = new_affinity.at[i].set(current_aff + 0.05)
                else:
                    new_contribution = new_contribution.at[i].set(current_contrib + 0.01)

        for func, info in young.items():
            new_info = dict(info)
            new_info['contribution'] = float(new_contribution[func])
            new_young[func] = new_info

        return (
            jnp.clip(new_contribution, 0, 2.0),
            jnp.clip(new_affinity, 0, 1.0),
            new_young
        )

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
        """Update with dual neurogenesis dynamics and protected indices."""
        key, k_act, k_agg, k_mature_act, k_mature_agg = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update contributions and affinities
        new_act_contrib, new_act_aff, act_young = self._update_contributions(
            state['act_contribution'], state['act_affinity'], state['act_mask'],
            state['act_young'], improved, NUM_ACTIVATIONS
        )
        new_agg_contrib, new_agg_aff, agg_young = self._update_contributions(
            state['agg_contribution'], state['agg_affinity'], state['agg_mask'],
            state['agg_young'], improved, NUM_AGGREGATIONS
        )

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

        # Compute mean contributions for cross-domain boost
        act_active = mask_to_indices(state['act_mask'])
        agg_active = mask_to_indices(state['agg_mask'])
        act_mean_contrib = float(np.mean([new_act_contrib[i] for i in act_active])) if act_active else 0
        agg_mean_contrib = float(np.mean([new_agg_contrib[i] for i in agg_active])) if agg_active else 0

        # Mature neurons with protected index handling
        act_stable, act_young, act_survived, act_pruned = self._mature_neurons(
            set(state['act_stable']), act_young, new_act_contrib,
            new_act_aff, new_act_mem_cells,
            agg_mean_contrib, generation, self.max_stable_act,
            is_activation=True, key=k_mature_act
        )
        agg_stable, agg_young, agg_survived, agg_pruned = self._mature_neurons(
            set(state['agg_stable']), agg_young, new_agg_contrib,
            new_agg_aff, new_agg_mem_cells,
            act_mean_contrib, generation, self.max_stable_agg,
            is_activation=False, key=k_mature_agg
        )

        # CRITICAL: Ensure protected indices are in stable sets
        act_stable = self._ensure_protected_in_stable(act_stable, [SIN_IDX], NUM_ACTIVATIONS)
        agg_stable = self._ensure_protected_in_stable(agg_stable, [MAX_IDX, MIN_IDX], NUM_AGGREGATIONS)

        # Birth new neurons (prefer protected indices if not present)
        act_protected = [SIN_IDX]
        agg_protected = [MAX_IDX, MIN_IDX]

        act_stable, act_young, act_born = self._maybe_birth_neuron(
            act_stable, act_young, k_act, generation,
            self.max_young_act, NUM_ACTIVATIONS, act_protected
        )
        agg_stable, agg_young, agg_born = self._maybe_birth_neuron(
            agg_stable, agg_young, k_agg, generation,
            self.max_young_agg, NUM_AGGREGATIONS, agg_protected
        )

        # Create masks
        new_act_mask = self._create_mask(act_stable, act_young, NUM_ACTIVATIONS)
        new_agg_mask = self._create_mask(agg_stable, agg_young, NUM_AGGREGATIONS)

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
            'act_stable': act_stable,
            'act_young': act_young,
            'act_contribution': new_act_contrib,
            'act_affinity': new_act_aff,
            'agg_mask': new_agg_mask,
            'agg_stable': agg_stable,
            'agg_young': agg_young,
            'agg_contribution': new_agg_contrib,
            'agg_affinity': new_agg_aff,
            'cross_affinity': new_cross,
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
            'discovery_gen': new_discovery,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'act_total_births': state['act_total_births'] + (1 if act_born else 0),
            'act_total_survivals': state['act_total_survivals'] + len(act_survived),
            'act_total_prunings': state['act_total_prunings'] + len(act_pruned),
            'agg_total_births': state['agg_total_births'] + (1 if agg_born else 0),
            'agg_total_survivals': state['agg_total_survivals'] + len(agg_survived),
            'agg_total_prunings': state['agg_total_prunings'] + len(agg_pruned),
        }

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neurogenesis
            'act_n_stable': len(act_stable),
            'act_n_young': len(act_young),
            'agg_n_stable': len(agg_stable),
            'agg_n_young': len(agg_young),
            'act_born': act_born,
            'agg_born': agg_born,
            'act_survived': act_survived,
            'agg_survived': agg_survived,
            'act_pruned': act_pruned,
            'agg_pruned': agg_pruned,
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Stats
            'act_survival_rate': (new_state['act_total_survivals'] / max(new_state['act_total_births'], 1)) * 100,
            'agg_survival_rate': (new_state['agg_total_survivals'] / max(new_state['agg_total_births'], 1)) * 100,
            # Sin/extreme status
            'has_sin': SIN_IDX in act_palette,
            'sin_is_stable': SIN_IDX in act_stable,
            'sin_is_young': SIN_IDX in act_young,
            'sin_contribution': float(new_act_contrib[SIN_IDX]),
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'has_max': MAX_IDX in agg_palette,
            'has_min': MIN_IDX in agg_palette,
            'max_affinity': float(new_agg_aff[MAX_IDX]),
            'min_affinity': float(new_agg_aff[MIN_IDX]),
            # Discovery
            'sin_discovered_gen': new_discovery['sin'],
            'max_discovered_gen': new_discovery['max'],
            'min_discovered_gen': new_discovery['min'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual neurogenesis status."""
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
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_n_stable': len(state['act_stable']),
            'act_n_young': len(state['act_young']),
            'agg_n_stable': len(state['agg_stable']),
            'agg_n_young': len(state['agg_young']),
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'act_total_births': state['act_total_births'],
            'agg_total_births': state['agg_total_births'],
            'sin_is_stable': SIN_IDX in state['act_stable'],
            'sin_contribution': float(state['act_contribution'][SIN_IDX]),
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
            'discovery_gen': state['discovery_gen'],
        }
