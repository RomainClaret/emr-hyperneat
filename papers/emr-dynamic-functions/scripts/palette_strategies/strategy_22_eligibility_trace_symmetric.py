"""Strategy 22 Symmetric: Eligibility Trace.

Extends EligibilityTraceStrategy with symmetric discovery features:
- Dual eligibility trace systems (separate for activation and aggregation)
- Cross-domain eligibility (activation success influences aggregation traces)
- Memory cells from sustained high eligibility + high dopamine
- Affinity floors and discovery tracking for both domains

Key mechanisms:
1. Eligibility traces: Decaying record of function activity
2. Dopamine: Global reward signal (fitness improvement)
3. Three-factor rule: affinity += lr × dopamine × eligibility
4. Memory cells: Functions with sustained high trace + reward become permanent

Biological rationale:
- Eligibility traces solve temporal credit assignment
- Dopamine gates which traces become permanent memory
- Cross-modal eligibility: Visual activity primes motor learning
- Memory cells encode proven causal relationships
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


class EligibilityTraceSymmetricStrategy(PaletteEvolutionStrategy):
    """Three-factor learning with dual eligibility traces and memory cells.

    Eligibility traces maintain decaying records of function activity.
    When dopamine (reward) arrives, functions with high eligibility
    receive credit and can become memory cells.

    Key innovations:
    - Dual eligibility per domain (activation and aggregation)
    - Cross-domain eligibility (success in one domain influences other)
    - Memory cells from sustained high eligibility + dopamine
    - Affinity floors prevent loss of critical functions
    """

    name = "eligibility_trace_symmetric"
    description = "Dual three-factor learning with memory cells"

    def __init__(
        self,
        # Eligibility trace parameters
        eligibility_decay: float = 0.85,           # Trace decay per generation
        eligibility_boost_active: float = 1.0,     # Boost for active functions
        eligibility_boost_changed: float = 0.5,    # Extra boost when function added
        # Cross-domain eligibility
        cross_eligibility_boost: float = 0.2,      # Boost from other domain's success
        cross_eligibility_decay: float = 0.9,      # How fast cross-domain influence decays
        # Dopamine (reward) parameters
        dopamine_baseline_momentum: float = 0.9,   # EMA for baseline fitness
        dopamine_sensitivity: float = 1.5,         # How much fitness diff matters
        dopamine_learning_rate: float = 0.15,      # How much DA affects affinity
        # Memory cell parameters (high trace + high DA → memory)
        memory_cell_eligibility_threshold: float = 0.5,  # Eligibility needed for memory
        memory_cell_dopamine_threshold: float = 0.3,     # DA needed for memory
        memory_cell_gens: int = 8,                       # Generations to maintain both
        memory_cell_decay_rate: float = 0.05,            # Slow decay for memory cells
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Protection
        affinity_protection_threshold: float = 0.6,
        # Mutation rates
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        da_exploration_modulation: float = 0.3,    # DA reduces exploration
        # Constraints
        act_min_active: int = 2,
        act_max_active: int = 6,
        agg_min_active: int = 1,
        agg_max_active: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Eligibility Trace Symmetric strategy."""
        # Eligibility
        self.eligibility_decay = eligibility_decay
        self.eligibility_boost_active = eligibility_boost_active
        self.eligibility_boost_changed = eligibility_boost_changed

        # Cross-domain eligibility
        self.cross_eligibility_boost = cross_eligibility_boost
        self.cross_eligibility_decay = cross_eligibility_decay

        # Dopamine
        self.dopamine_baseline_momentum = dopamine_baseline_momentum
        self.dopamine_sensitivity = dopamine_sensitivity
        self.dopamine_learning_rate = dopamine_learning_rate

        # Memory cells
        self.memory_cell_eligibility_threshold = memory_cell_eligibility_threshold
        self.memory_cell_dopamine_threshold = memory_cell_dopamine_threshold
        self.memory_cell_gens = memory_cell_gens
        self.memory_cell_decay_rate = memory_cell_decay_rate

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Discovery
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.da_exploration_modulation = da_exploration_modulation

        # Constraints
        self.act_min_active = act_min_active
        self.act_max_active = act_max_active
        self.agg_min_active = agg_min_active
        self.agg_max_active = agg_max_active
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual eligibility trace systems."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_eligibility = jnp.zeros(NUM_ACTIVATIONS)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_eligibility = jnp.zeros(NUM_AGGREGATIONS)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Memory cell tracking (high eligibility + high DA for sustained period)
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
            'act_eligibility': act_eligibility,
            'act_affinity': act_affinity,
            'act_previous_mask': act_mask,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_eligibility': agg_eligibility,
            'agg_affinity': agg_affinity,
            'agg_previous_mask': agg_mask,
            # Dopamine system (shared)
            'dopamine_baseline': 0.5,
            'dopamine_signal': 0.0,
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
            'rng_key': jax.random.PRNGKey(seed + 222232),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _update_eligibility(
        self,
        eligibility: jnp.ndarray,
        mask: jnp.ndarray,
        previous_mask: jnp.ndarray,
        cross_boost: float = 0.0,
    ) -> jnp.ndarray:
        """Update eligibility traces with decay, activity boost, and cross-domain boost.

        eligibility = decay × previous_eligibility + current_activity + cross_boost
        """
        # Decay all eligibility traces
        new_eligibility = self.eligibility_decay * eligibility

        # Boost for currently active functions
        active = (mask > 0.5).astype(jnp.float32)
        new_eligibility = new_eligibility + self.eligibility_boost_active * active

        # Extra boost for newly activated functions
        was_inactive = (previous_mask < 0.5).astype(jnp.float32)
        just_activated = active * was_inactive
        new_eligibility = new_eligibility + self.eligibility_boost_changed * just_activated

        # Cross-domain boost (if the other domain is doing well)
        new_eligibility = new_eligibility + cross_boost * active

        # Clip to reasonable range
        return jnp.clip(new_eligibility, 0.0, 3.0)

    def _compute_dopamine(
        self,
        fitness: float,
        baseline: float,
    ) -> Tuple[float, float]:
        """Compute dopamine signal as reward prediction error."""
        # Update baseline (expected fitness)
        new_baseline = (
            self.dopamine_baseline_momentum * baseline +
            (1 - self.dopamine_baseline_momentum) * fitness
        )

        # Compute reward prediction error
        if baseline > 0.01:
            prediction_error = (fitness - baseline) / baseline
        else:
            prediction_error = fitness - baseline

        # Scale by sensitivity
        dopamine = self.dopamine_sensitivity * prediction_error

        # Clip to reasonable range
        dopamine = max(-1.0, min(1.0, dopamine))

        return dopamine, new_baseline

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        eligibility: jnp.ndarray,
        dopamine: float,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
    ) -> jnp.ndarray:
        """Update affinity using three-factor rule with memory cell protection.

        affinity[i] += lr × dopamine × eligibility[i]
        Memory cells resist negative changes.
        """
        newly_discovered = newly_discovered or []

        # Three-factor update: learning_rate × dopamine × eligibility
        delta = self.dopamine_learning_rate * dopamine * eligibility

        # Apply update
        new_affinity = affinity + delta

        # Memory cells resist negative changes
        negative_delta = delta < 0
        memory_protected = jnp.logical_and(negative_delta, memory_cells)
        new_affinity = jnp.where(
            memory_protected,
            affinity * (1 - self.memory_cell_decay_rate),
            new_affinity
        )

        # Apply discovery boost
        for idx in newly_discovered:
            new_affinity = new_affinity.at[idx].set(
                new_affinity[idx] + self.discovery_boost
            )

        # Clip to valid range
        return jnp.clip(new_affinity, 0.05, 0.95)

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

    def _update_memory_cells_from_eligibility(
        self,
        eligibility: jnp.ndarray,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        dopamine: float,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cells based on sustained high eligibility + dopamine.

        Memory cells form when EITHER:
        1. High eligibility (>= threshold) + High dopamine for memory_cell_gens generations
        2. High affinity (>= 0.75) for memory_cell_gens generations (fallback)

        This hybrid approach ensures memory cells can form even when dopamine is variable.
        """
        active = mask > 0.5

        # Path 1: High eligibility + positive dopamine
        high_eligibility = eligibility >= self.memory_cell_eligibility_threshold
        positive_dopamine = dopamine >= self.memory_cell_dopamine_threshold
        eligibility_path = jnp.logical_and(
            jnp.logical_and(high_eligibility, positive_dopamine),
            active
        )

        # Path 2: High affinity (fallback)
        high_affinity = jnp.logical_and(
            affinity >= 0.75,
            active
        )

        # Either path counts toward memory cell status
        memory_candidate = jnp.logical_or(eligibility_path, high_affinity)

        new_counts = jnp.where(memory_candidate, memory_counts + 1, 0)
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        eligibility: jnp.ndarray,
        memory_cells: jnp.ndarray,
        dopamine: float,
        min_active: int,
        max_active: int,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Apply mutation with dopamine modulation and memory cell protection."""
        newly_discovered = newly_discovered or []
        key1, key2 = jax.random.split(key)
        n_funcs = len(mask)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        discovery_to_palette = 0

        # Dopamine modulates exploration
        da_factor = 1.0 - self.da_exploration_modulation * dopamine
        da_factor = max(0.5, min(1.5, da_factor))

        effective_activate_rate = self.base_activate_rate * da_factor
        effective_deactivate_rate = self.base_deactivate_rate * da_factor

        activate_probs = jax.random.uniform(key1, (n_funcs,))
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))

        for i in range(n_funcs):
            aff = float(affinity[i])
            elig = float(eligibility[i])
            is_memory = bool(memory_cells[i])

            # Combined protection score
            elig_contribution = 0.2 * min(elig / 2.0, 1.0)
            protection = 0.8 * aff + 0.2 * elig_contribution

            if mask[i] < 0.5:
                # Inactive: maybe activate
                rate = effective_activate_rate * (0.5 + 0.5 * protection)

                current_active = int(jnp.sum(new_mask > 0.5))
                if activate_probs[i] < rate and current_active < max_active:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
                    if i in newly_discovered:
                        discovery_to_palette += 1
            else:
                # Active: maybe deactivate
                # Memory cells never deactivate
                if is_memory:
                    continue

                if protection >= self.affinity_protection_threshold:
                    rate = effective_deactivate_rate * 0.1
                else:
                    rate = effective_deactivate_rate * (1.0 - protection)

                if deactivate_probs[i] < rate:
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
            'da_factor': da_factor,
        }, discovery_to_palette

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual eligibility traces and memory cells."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

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

        # Step 1: Compute dopamine signal (shared across domains)
        dopamine, new_baseline = self._compute_dopamine(
            best_fitness,
            state['dopamine_baseline']
        )

        # Step 2: Compute cross-domain eligibility boosts
        # If one domain has memory cells, boost eligibility in the other
        act_cross_boost = self.cross_eligibility_boost if jnp.any(state['agg_memory_cells']) else 0.0
        agg_cross_boost = self.cross_eligibility_boost if jnp.any(state['act_memory_cells']) else 0.0

        # Apply additional boost based on dopamine (success amplifies cross-domain learning)
        if dopamine > 0:
            act_cross_boost *= (1.0 + dopamine)
            agg_cross_boost *= (1.0 + dopamine)

        # Step 3: Update eligibility traces
        new_act_elig = self._update_eligibility(
            state['act_eligibility'], state['act_mask'],
            state['act_previous_mask'], act_cross_boost
        )
        new_agg_elig = self._update_eligibility(
            state['agg_eligibility'], state['agg_mask'],
            state['agg_previous_mask'], agg_cross_boost
        )

        # Step 4: Three-factor learning update to affinity
        new_act_aff = self._update_affinity(
            state['act_affinity'], new_act_elig, dopamine,
            state['act_memory_cells'], act_new_candidates
        )
        new_agg_aff = self._update_affinity(
            state['agg_affinity'], new_agg_elig, dopamine,
            state['agg_memory_cells'], agg_new_candidates
        )

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Step 5: Update memory cells
        new_act_mc, new_act_mem_cells = self._update_memory_cells_from_eligibility(
            new_act_elig, new_act_aff, state['act_mask'], dopamine,
            state['act_memory_counts'], state['act_memory_cells']
        )
        new_agg_mc, new_agg_mem_cells = self._update_memory_cells_from_eligibility(
            new_agg_elig, new_agg_aff, state['agg_mask'], dopamine,
            state['agg_memory_counts'], state['agg_memory_cells']
        )

        # Step 6: Apply mutation with DA modulation
        new_act_mask, act_mut_info, act_disc_to_pal = self._mutate_palette(
            k1, state['act_mask'], new_act_aff, new_act_elig, new_act_mem_cells,
            dopamine, self.act_min_active, self.act_max_active, act_new_candidates
        )
        new_agg_mask, agg_mut_info, agg_disc_to_pal = self._mutate_palette(
            k2, state['agg_mask'], new_agg_aff, new_agg_elig, new_agg_mem_cells,
            dopamine, self.agg_min_active, self.agg_max_active, agg_new_candidates
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
            'act_eligibility': new_act_elig,
            'act_affinity': new_act_aff,
            'act_previous_mask': state['act_mask'],
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_eligibility': new_agg_elig,
            'agg_affinity': new_agg_aff,
            'agg_previous_mask': state['agg_mask'],
            # Dopamine system
            'dopamine_baseline': new_baseline,
            'dopamine_signal': dopamine,
            # Memory cells
            'act_memory_counts': new_act_mc,
            'agg_memory_counts': new_agg_mc,
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
            # Dopamine
            'dopamine_signal': dopamine,
            'dopamine_baseline': new_baseline,
            # Eligibility
            'act_avg_eligibility': float(jnp.mean(new_act_elig)),
            'agg_avg_eligibility': float(jnp.mean(new_agg_elig)),
            'act_max_eligibility': float(jnp.max(new_act_elig)),
            'agg_max_eligibility': float(jnp.max(new_agg_elig)),
            'sin_eligibility': float(new_act_elig[SIN_IDX]),
            # Affinity
            'act_avg_affinity': float(jnp.mean(new_act_aff)),
            'agg_avg_affinity': float(jnp.mean(new_agg_aff)),
            # Sin and extreme agg
            'has_sin': has_sin,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
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
            # Eligibility
            'act_avg_eligibility': float(jnp.mean(state['act_eligibility'])),
            'agg_avg_eligibility': float(jnp.mean(state['agg_eligibility'])),
            'sin_eligibility': float(state['act_eligibility'][SIN_IDX]),
            # Affinity
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'act_avg_affinity': float(jnp.mean(state['act_affinity'])),
            'agg_avg_affinity': float(jnp.mean(state['agg_affinity'])),
            # Dopamine
            'dopamine_signal': state['dopamine_signal'],
            'dopamine_baseline': state['dopamine_baseline'],
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
