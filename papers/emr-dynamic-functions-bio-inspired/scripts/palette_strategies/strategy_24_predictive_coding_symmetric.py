"""Strategy 24 Symmetric: Predictive Coding.

COMPLETE REWRITE using winning patterns from eligibility_trace_symmetric:
- Shared dopamine signal (fitness-based, same for both domains)
- Surprise-modulated learning (like eligibility traces but using prediction error)
- Affinity floors for sin and extreme aggregations
- Memory cells from sustained high affinity

Key biological insight preserved:
- Cortical hierarchies predict incoming signals
- Learning is driven by prediction ERRORS (surprise)
- Unexpected outcomes teach more than expected outcomes

Changes:
- Replaced error-based affinity with fitness-based affinity (like eligibility_trace)
- Added shared dopamine signal across domains
- Fixed memory cell formation to use affinity-based path
- Kept predictive coding biology (surprise modulates learning rate)
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


class PredictiveCodingSymmetricStrategy(PaletteEvolutionStrategy):
    """Predictive coding with shared dopamine and memory cells.

    Uses winning patterns from eligibility_trace_symmetric.

    Key changes:
    - Shared dopamine signal (same for both domains) - CRITICAL for coordination
    - Fitness-based affinity updates (not prediction-error-based)
    - Surprise modulates learning rate (preserves predictive coding biology)
    - Memory cells from sustained high affinity (not low-surprise)

    Biological rationale:
    - Dopamine = reward prediction error (same as eligibility trace)
    - Surprise = how unexpected the outcome was (modulates learning)
    - High surprise → higher learning rate (pay attention to unexpected)
    - Low surprise → lower learning rate (already know this)
    """

    name = "predictive_coding_symmetric"
    description = "Predictive coding with shared dopamine and memory cells"

    def __init__(
        self,
        # Dopamine (shared reward signal) - COPIED FROM ELIGIBILITY TRACE
        dopamine_baseline_momentum: float = 0.9,
        dopamine_sensitivity: float = 1.5,
        dopamine_learning_rate: float = 0.15,
        # Prediction/surprise parameters (predictive coding biology)
        prediction_lr: float = 0.15,
        surprise_boost_factor: float = 1.3,  # High surprise → faster learning
        surprise_threshold: float = 0.2,  # Above this = surprising
        # Memory cell parameters - COPIED FROM ELIGIBILITY TRACE
        memory_cell_affinity_threshold: float = 0.75,
        memory_cell_gens: int = 8,
        memory_cell_decay_rate: float = 0.05,
        # Affinity floors - CRITICAL FOR RETENTION
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
        da_exploration_modulation: float = 0.3,
        # Constraints
        act_min_active: int = 2,
        act_max_active: int = 6,
        agg_min_active: int = 1,
        agg_max_active: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Predictive Coding Symmetric strategy."""
        # Dopamine (shared)
        self.dopamine_baseline_momentum = dopamine_baseline_momentum
        self.dopamine_sensitivity = dopamine_sensitivity
        self.dopamine_learning_rate = dopamine_learning_rate

        # Prediction/surprise
        self.prediction_lr = prediction_lr
        self.surprise_boost_factor = surprise_boost_factor
        self.surprise_threshold = surprise_threshold

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
        """Initialize state with shared dopamine system."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_predictions = jnp.ones(NUM_ACTIVATIONS) * 0.5
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_predictions = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Memory cell tracking (based on affinity, not surprise)
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
            'act_predictions': act_predictions,
            'act_affinity': act_affinity,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_predictions': agg_predictions,
            'agg_affinity': agg_affinity,
            # Shared dopamine system (like eligibility trace)
            'dopamine_baseline': 0.5,
            'dopamine_signal': 0.0,
            # Surprise tracking (per domain)
            'act_surprise': 0.0,
            'agg_surprise': 0.0,
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
            'rng_key': jax.random.PRNGKey(seed + 242433),
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

    def _compute_dopamine(
        self,
        fitness: float,
        baseline: float,
    ) -> Tuple[float, float]:
        """Compute dopamine signal as reward prediction error.

        COPIED FROM eligibility_trace_symmetric - this is the key to coordination.
        """
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

    def _update_predictions(
        self,
        predictions: jnp.ndarray,
        fitness: float,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, float]:
        """Update predictions and compute surprise (mean absolute error)."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)

        # Predict fitness contribution per active function
        per_function_contribution = fitness / n_active
        actual = active * per_function_contribution

        # Prediction error
        error = actual - predictions

        # Update predictions toward actual
        new_predictions = predictions + self.prediction_lr * error

        # Inactive functions decay toward baseline
        baseline = 0.5
        new_predictions = jnp.where(
            active > 0.5,
            new_predictions,
            0.95 * predictions + 0.05 * baseline
        )

        new_predictions = jnp.clip(new_predictions, 0.0, 1.0)

        # Compute surprise as mean absolute error
        surprise = float(jnp.mean(jnp.abs(error)))

        return new_predictions, surprise

    def _update_affinity_with_dopamine(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        dopamine: float,
        surprise: float,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
    ) -> jnp.ndarray:
        """Update affinity using dopamine (like eligibility trace) with surprise modulation.

        Use fitness-based dopamine, not prediction errors.
        Surprise modulates the learning rate (predictive coding biology).

        affinity[i] += lr × dopamine × (1 + surprise_boost if surprising)
        """
        newly_discovered = newly_discovered or []

        active = (mask > 0.5).astype(jnp.float32)

        # Surprise modulates learning rate (predictive coding biology)
        if surprise > self.surprise_threshold:
            lr_boost = self.surprise_boost_factor
        else:
            lr_boost = 1.0

        effective_lr = self.dopamine_learning_rate * lr_boost

        # Three-factor-style update: lr × dopamine × activity
        delta = effective_lr * dopamine * active

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

        return jnp.clip(new_affinity, 0.05, 0.95)

    def _apply_affinity_floors(
        self,
        act_affinity: jnp.ndarray,
        agg_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for critical functions - CRITICAL FOR RETENTION."""
        new_act = act_affinity.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )
        new_agg = agg_affinity
        for idx in CORE_EXTREME_AGGS:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(new_agg[idx], self.extreme_agg_affinity_floor)
            )
        return new_act, new_agg

    def _update_memory_cells_from_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cells based on sustained high affinity.

        Use affinity-based path (like eligibility trace).
        Memory cells form when affinity >= threshold for memory_cell_gens generations.
        """
        active = mask > 0.5

        # High affinity AND active
        high_affinity = jnp.logical_and(
            affinity >= self.memory_cell_affinity_threshold,
            active
        )

        new_counts = jnp.where(high_affinity, memory_counts + 1, 0)
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        memory_cells: jnp.ndarray,
        dopamine: float,
        min_active: int,
        max_active: int,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Apply mutation with dopamine modulation and memory cell protection.

        COPIED FROM eligibility_trace_symmetric.
        """
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
            is_memory = bool(memory_cells[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                rate = effective_activate_rate * (0.5 + 0.5 * aff)

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

                if aff >= self.affinity_protection_threshold:
                    rate = effective_deactivate_rate * 0.1
                else:
                    rate = effective_deactivate_rate * (1.0 - aff)

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
        """Update with shared dopamine and memory cells."""
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

        # Step 1: Compute SHARED dopamine signal (like eligibility trace)
        dopamine, new_baseline = self._compute_dopamine(
            best_fitness,
            state['dopamine_baseline']
        )

        # Step 2: Update predictions and compute surprise per domain
        new_act_pred, act_surprise = self._update_predictions(
            state['act_predictions'], best_fitness, state['act_mask']
        )
        new_agg_pred, agg_surprise = self._update_predictions(
            state['agg_predictions'], best_fitness, state['agg_mask']
        )

        # Step 3: Update affinity using SHARED dopamine (with surprise modulation)
        new_act_aff = self._update_affinity_with_dopamine(
            state['act_affinity'], state['act_mask'], dopamine, act_surprise,
            state['act_memory_cells'], act_new_candidates
        )
        new_agg_aff = self._update_affinity_with_dopamine(
            state['agg_affinity'], state['agg_mask'], dopamine, agg_surprise,
            state['agg_memory_cells'], agg_new_candidates
        )

        # Apply affinity floors - CRITICAL FOR RETENTION
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Step 4: Update memory cells from affinity (not surprise)
        new_act_mc, new_act_mem_cells = self._update_memory_cells_from_affinity(
            new_act_aff, state['act_mask'],
            state['act_memory_counts'], state['act_memory_cells']
        )
        new_agg_mc, new_agg_mem_cells = self._update_memory_cells_from_affinity(
            new_agg_aff, state['agg_mask'],
            state['agg_memory_counts'], state['agg_memory_cells']
        )

        # Step 5: Apply mutation with DA modulation
        new_act_mask, act_mut_info, act_disc_to_pal = self._mutate_palette(
            k1, state['act_mask'], new_act_aff, new_act_mem_cells,
            dopamine, self.act_min_active, self.act_max_active, act_new_candidates
        )
        new_agg_mask, agg_mut_info, agg_disc_to_pal = self._mutate_palette(
            k2, state['agg_mask'], new_agg_aff, new_agg_mem_cells,
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
            'act_predictions': new_act_pred,
            'act_affinity': new_act_aff,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_predictions': new_agg_pred,
            'agg_affinity': new_agg_aff,
            # Shared dopamine system
            'dopamine_baseline': new_baseline,
            'dopamine_signal': dopamine,
            # Surprise tracking
            'act_surprise': act_surprise,
            'agg_surprise': agg_surprise,
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
            # Dopamine (shared)
            'dopamine_signal': dopamine,
            'dopamine_baseline': new_baseline,
            # Surprise (per domain)
            'act_surprise': act_surprise,
            'agg_surprise': agg_surprise,
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
            # Dopamine
            'dopamine_signal': state['dopamine_signal'],
            'dopamine_baseline': state['dopamine_baseline'],
            # Surprise
            'act_surprise': state['act_surprise'],
            'agg_surprise': state['agg_surprise'],
            # Affinity
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'act_avg_affinity': float(jnp.mean(state['act_affinity'])),
            'agg_avg_affinity': float(jnp.mean(state['agg_affinity'])),
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
