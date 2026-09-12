"""Strategy 25 Symmetric: Intrinsic Plasticity for Activation AND Aggregation Discovery.

Extends Intrinsic Plasticity to symmetric palette evolution with per-function
threshold and gain adaptation for both domains, combined with memory cells
and affinity floors for guaranteed retention.

Key mechanisms:
1. Threshold adaptation: Shift activation curves based on activity
2. Gain adaptation: Scale response magnitude based on activity
3. Homeostatic target: Maintain functions in optimal operating range
4. Cross-domain learning: Share intrinsic params between domains
5. Memory cells crystallize high-value functions
6. Protected indices for sin/extreme aggregations
7. Affinity floors for guaranteed retention

Biological basis:
- Neurons regulate their own firing threshold and gain
- Too much activity -> increase threshold, decrease gain
- Too little activity -> decrease threshold, increase gain
- Applies to both computational and integrative functions
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


# Critical indices for guaranteed retention
SIN_IDX = 4  # Sin activation - critical for parity problems
CORE_EXTREME_AGGS = [2, 3]  # max (idx 2), min (idx 3) - critical for aggregation


class IntrinsicPlasticitySymmetricStrategy(PaletteEvolutionStrategy):
    """Intrinsic plasticity with symmetric palette evolution.

    Per-function threshold/gain adaptation for both activation and aggregation
    palettes, with memory cells and affinity floors for guaranteed retention.
    """

    name = "intrinsic_plasticity_symmetric"
    description = "Symmetric intrinsic plasticity with memory cells"

    def __init__(
        self,
        # Intrinsic plasticity parameters
        threshold_lr: float = 0.08,
        gain_lr: float = 0.04,
        target_activity: float = 0.5,
        threshold_bounds: Tuple[float, float] = (-0.5, 0.5),
        gain_bounds: Tuple[float, float] = (0.5, 2.0),
        # Hebbian parameters
        hebbian_lr: float = 0.12,
        hebbian_decay: float = 0.02,
        act_affinity_protection: float = 0.6,
        agg_affinity_protection: float = 0.6,
        # Mutation rates
        act_base_activate_rate: float = 0.12,
        act_base_deactivate_rate: float = 0.05,
        agg_base_activate_rate: float = 0.10,
        agg_base_deactivate_rate: float = 0.04,
        # Activity tracking
        activity_momentum: float = 0.7,
        # Cross-domain
        cross_learning_rate: float = 0.06,
        # Memory cell parameters (from winning patterns)
        memory_threshold: float = 0.75,
        memory_sustain_generations: int = 8,
        # Affinity floors (CRITICAL for retention)
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Constraints
        max_active_act: int = 6,
        max_active_agg: int = 4,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        # General
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize symmetric intrinsic plasticity strategy."""
        # Intrinsic plasticity
        self.threshold_lr = threshold_lr
        self.gain_lr = gain_lr
        self.target_activity = target_activity
        self.threshold_bounds = threshold_bounds
        self.gain_bounds = gain_bounds

        # Hebbian
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.act_affinity_protection = act_affinity_protection
        self.agg_affinity_protection = agg_affinity_protection

        # Mutation
        self.act_base_activate_rate = act_base_activate_rate
        self.act_base_deactivate_rate = act_base_deactivate_rate
        self.agg_base_activate_rate = agg_base_activate_rate
        self.agg_base_deactivate_rate = agg_base_deactivate_rate

        # Activity
        self.activity_momentum = activity_momentum

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        # Memory cell parameters
        self.memory_threshold = memory_threshold
        self.memory_sustain_generations = memory_sustain_generations

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Constraints
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with intrinsic plasticity and memory cells."""
        # Activation
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_thresholds = jnp.zeros(NUM_ACTIVATIONS)
        act_gains = jnp.ones(NUM_ACTIVATIONS)
        act_activity_estimates = jnp.ones(NUM_ACTIVATIONS) * self.target_activity
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Aggregation
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_thresholds = jnp.zeros(NUM_AGGREGATIONS)
        agg_gains = jnp.ones(NUM_AGGREGATIONS)
        agg_activity_estimates = jnp.ones(NUM_AGGREGATIONS) * self.target_activity
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Memory cells (symmetric pattern)
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        sin_discovered = SIN_IDX in initial_act
        sin_discovery_gen = 0 if sin_discovered else -1
        extreme_agg_discovered = any(idx in initial_agg for idx in CORE_EXTREME_AGGS)
        extreme_agg_discovery_gen = 0 if extreme_agg_discovered else -1

        return {
            # Activation state
            'act_mask': act_mask,
            'act_thresholds': act_thresholds,
            'act_gains': act_gains,
            'act_activity_estimates': act_activity_estimates,
            'act_affinity': act_affinity,
            # Aggregation state
            'agg_mask': agg_mask,
            'agg_thresholds': agg_thresholds,
            'agg_gains': agg_gains,
            'agg_activity_estimates': agg_activity_estimates,
            'agg_affinity': agg_affinity,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_counts': agg_memory_counts,
            'agg_memory_cells': agg_memory_cells,
            # Discovery tracking
            'sin_discovered': sin_discovered,
            'sin_discovery_gen': sin_discovery_gen,
            'extreme_agg_discovered': extreme_agg_discovered,
            'extreme_agg_discovery_gen': extreme_agg_discovery_gen,
            # General
            'rng_key': jax.random.PRNGKey(seed + 252525),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def get_intrinsic_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return current intrinsic plasticity parameters for both domains."""
        return {
            'act_thresholds': state['act_thresholds'],
            'act_gains': state['act_gains'],
            'agg_thresholds': state['agg_thresholds'],
            'agg_gains': state['agg_gains'],
        }

    def _apply_affinity_floors(
        self, act_affinity: jnp.ndarray, agg_affinity: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for sin and extreme aggregations."""
        new_act = act_affinity.copy()
        new_agg = agg_affinity.copy()

        # Sin activation floor
        new_act = new_act.at[SIN_IDX].set(
            jnp.maximum(act_affinity[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation floors
        for idx in CORE_EXTREME_AGGS:
            new_agg = new_agg.at[idx].set(
                jnp.maximum(agg_affinity[idx], self.extreme_agg_affinity_floor)
            )

        return new_act, new_agg

    def _update_memory_cells(
        self, affinity: jnp.ndarray, memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray, mask: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell counts and crystallize sustained high-affinity functions."""
        above_threshold = affinity >= self.memory_threshold
        active = mask > 0.5

        new_counts = jnp.where(
            above_threshold & active,
            memory_counts + 1,
            jnp.zeros_like(memory_counts)
        )

        newly_memory = new_counts >= self.memory_sustain_generations
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _estimate_activity(
        self,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
        num_funcs: int,
    ) -> jnp.ndarray:
        """Estimate activity level based on fitness contribution."""
        fitness_delta = fitness - prev_fitness
        fitness_signal = 1.0 / (1.0 + jnp.exp(-fitness_delta * 10))

        active = (mask > 0.5).astype(jnp.float32)
        activity = active * (0.3 + 0.7 * fitness_signal)

        return activity

    def _update_intrinsic_params(
        self,
        thresholds: jnp.ndarray,
        gains: jnp.ndarray,
        activity_estimates: jnp.ndarray,
        new_activity: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update intrinsic plasticity parameters based on activity."""
        active = (mask > 0.5).astype(jnp.float32)

        # Update activity estimates (EMA)
        new_estimates = jnp.where(
            active > 0.5,
            self.activity_momentum * activity_estimates + (1 - self.activity_momentum) * new_activity,
            activity_estimates * 0.99
        )

        # Compute activity error
        error = new_estimates - self.target_activity

        # Update thresholds
        new_thresholds = thresholds - self.threshold_lr * error * active
        new_thresholds = jnp.clip(new_thresholds, self.threshold_bounds[0], self.threshold_bounds[1])

        # Update gains
        gain_update = 1.0 - self.gain_lr * error
        new_gains = gains * jnp.where(active > 0.5, gain_update, 1.0)
        new_gains = jnp.clip(new_gains, self.gain_bounds[0], self.gain_bounds[1])

        return new_thresholds, new_gains, new_estimates

    def _update_hebbian_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_improved: bool,
        gains: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update Hebbian affinity weighted by intrinsic gain."""
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_improved:
            tune_quality = 1.0 - jnp.abs(gains - 1.0)
            signal = self.hebbian_lr * active * (0.5 + 0.5 * tune_quality)
        else:
            signal = -self.hebbian_lr * 0.3 * active

        new_affinity = affinity + signal

        # Decay inactive
        inactive = 1.0 - active
        new_affinity = new_affinity - self.hebbian_decay * inactive * affinity

        return jnp.clip(new_affinity, 0.05, 0.95)

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_improved: bool,
        act_gains: jnp.ndarray,
        agg_gains: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update cross-domain affinity weighted by both gains."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)

        # Weight by average tune quality
        act_tune = 1.0 - jnp.abs(act_gains - 1.0)
        agg_tune = 1.0 - jnp.abs(agg_gains - 1.0)
        tune_weight = jnp.outer(act_tune, agg_tune)

        if fitness_improved:
            delta = self.cross_learning_rate * cross_active * (0.5 + 0.5 * tune_weight)
        else:
            delta = -self.cross_learning_rate * 0.3 * cross_active

        return jnp.clip(cross_affinity + delta, 0.0, 1.0)

    def _compute_protection_scores(
        self,
        affinity: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        other_mask: jnp.ndarray,
        is_act: bool,
        memory_cells: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection scores including memory cell status."""
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)

        if is_act:
            cross_score = jnp.dot(cross_affinity, other_active) / n_other
        else:
            cross_score = jnp.dot(cross_affinity.T, other_active) / n_other

        base_prot = 0.65 * affinity + 0.25 * cross_score

        # Memory cells get maximum protection
        memory_boost = memory_cells.astype(jnp.float32) * 0.10

        return jnp.clip(base_prot + memory_boost, 0.0, 1.0)

    def _apply_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        gains: jnp.ndarray,
        is_activation: bool,
        memory_cells: jnp.ndarray,
        protected_indices: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with protected indices for guaranteed retention."""
        key1, key2 = jax.random.split(key)

        if is_activation:
            num_funcs = NUM_ACTIVATIONS
            max_active = self.max_active_act
            min_active = self.min_active_act
            protection_threshold = self.act_affinity_protection
            base_activate = self.act_base_activate_rate
            base_deactivate = self.act_base_deactivate_rate
        else:
            num_funcs = NUM_AGGREGATIONS
            max_active = self.max_active_agg
            min_active = self.min_active_agg
            protection_threshold = self.agg_affinity_protection
            base_activate = self.agg_base_activate_rate
            base_deactivate = self.agg_base_deactivate_rate

        protected_set = set(protected_indices or [])

        activate_probs = jax.random.uniform(key1, (num_funcs,))
        deactivate_probs = jax.random.uniform(key2, (num_funcs,))

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(num_funcs):
            aff = float(affinity[i])
            gain = float(gains[i])
            tune_quality = 1.0 - abs(gain - 1.0)
            is_memory = bool(memory_cells[i])
            is_protected = i in protected_set

            if mask[i] < 0.5:
                # Activation logic
                current_active = int(jnp.sum(mask > 0.5))
                if current_active >= max_active:
                    continue

                # Protected indices get activation boost
                if is_protected:
                    rate = base_activate * 2.0
                else:
                    rate = base_activate * (0.5 + 0.5 * aff) * (0.7 + 0.3 * tune_quality)

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Deactivation logic

                # Memory cells never deactivate
                if is_memory:
                    continue

                # Protected indices almost never deactivate (0.1% chance)
                if is_protected:
                    if deactivate_probs[i] < 0.001:
                        new_mask = new_mask.at[i].set(0.0)
                        deactivated.append(i)
                    continue

                # Standard deactivation based on affinity and tune quality
                if aff >= protection_threshold:
                    rate = base_deactivate * 0.1
                else:
                    rate = base_deactivate * (1.0 - aff) * (2.0 - tune_quality)
                    rate = min(rate, base_deactivate * 2)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        prefix = 'act_' if is_activation else 'agg_'
        return new_mask, {f'{prefix}activated': activated, f'{prefix}deactivated': deactivated}

    def _update_discovery(
        self, state: Dict[str, Any], generation: int
    ) -> Dict[str, Any]:
        """Track discovery of sin and extreme aggregations."""
        updates = {}

        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        if not state['sin_discovered'] and SIN_IDX in act_palette:
            updates['sin_discovered'] = True
            updates['sin_discovery_gen'] = generation

        if not state['extreme_agg_discovered']:
            if any(idx in agg_palette for idx in CORE_EXTREME_AGGS):
                updates['extreme_agg_discovered'] = True
                updates['extreme_agg_discovery_gen'] = generation

        return updates

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with intrinsic plasticity and memory cells."""
        key, subkey1, subkey2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Estimate activity for both domains
        act_activity = self._estimate_activity(
            state['act_mask'], best_fitness, prev_best_fitness, NUM_ACTIVATIONS
        )
        agg_activity = self._estimate_activity(
            state['agg_mask'], best_fitness, prev_best_fitness, NUM_AGGREGATIONS
        )

        # Update intrinsic params
        new_act_thresholds, new_act_gains, new_act_estimates = self._update_intrinsic_params(
            state['act_thresholds'], state['act_gains'], state['act_activity_estimates'],
            act_activity, state['act_mask']
        )
        new_agg_thresholds, new_agg_gains, new_agg_estimates = self._update_intrinsic_params(
            state['agg_thresholds'], state['agg_gains'], state['agg_activity_estimates'],
            agg_activity, state['agg_mask']
        )

        # Update Hebbian affinity
        new_act_affinity = self._update_hebbian_affinity(
            state['act_affinity'], state['act_mask'], improved, new_act_gains
        )
        new_agg_affinity = self._update_hebbian_affinity(
            state['agg_affinity'], state['agg_mask'], improved, new_agg_gains
        )

        # Update cross-domain
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'],
            improved, new_act_gains, new_agg_gains
        )

        # Apply affinity floors
        new_act_affinity, new_agg_affinity = self._apply_affinity_floors(
            new_act_affinity, new_agg_affinity
        )

        # Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            new_act_affinity, state['act_memory_counts'],
            state['act_memory_cells'], state['act_mask'])
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            new_agg_affinity, state['agg_memory_counts'],
            state['agg_memory_cells'], state['agg_mask'])

        # Compute protection scores
        act_prot = self._compute_protection_scores(
            new_act_affinity, new_cross, state['agg_mask'], True, new_act_mem_cells
        )
        agg_prot = self._compute_protection_scores(
            new_agg_affinity, new_cross, state['act_mask'], False, new_agg_mem_cells
        )

        # Apply mutations with protected indices
        new_act_mask, act_mut = self._apply_mutation(
            subkey1, state['act_mask'], act_prot, new_act_gains, True,
            new_act_mem_cells, protected_indices=[SIN_IDX]
        )
        new_agg_mask, agg_mut = self._apply_mutation(
            subkey2, state['agg_mask'], agg_prot, new_agg_gains, False,
            new_agg_mem_cells, protected_indices=CORE_EXTREME_AGGS
        )

        # Track fitness
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_thresholds': new_act_thresholds,
            'act_gains': new_act_gains,
            'act_activity_estimates': new_act_estimates,
            'act_affinity': new_act_affinity,
            'agg_mask': new_agg_mask,
            'agg_thresholds': new_agg_thresholds,
            'agg_gains': new_agg_gains,
            'agg_activity_estimates': new_agg_estimates,
            'agg_affinity': new_agg_affinity,
            'cross_affinity': new_cross,
            # Memory cells
            'act_memory_counts': new_act_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_counts': new_agg_mem_counts,
            'agg_memory_cells': new_agg_mem_cells,
            # Discovery tracking
            'sin_discovered': state['sin_discovered'],
            'sin_discovery_gen': state['sin_discovery_gen'],
            'extreme_agg_discovered': state['extreme_agg_discovered'],
            'extreme_agg_discovery_gen': state['extreme_agg_discovery_gen'],
            # General
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        # Update discovery tracking
        discovery_updates = self._update_discovery(new_state, generation)
        new_state.update(discovery_updates)

        # Compute metrics
        act_mem_count = int(jnp.sum(new_act_mem_cells))
        agg_mem_count = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Activation intrinsic stats
            'act_mean_threshold': float(jnp.mean(new_act_thresholds)),
            'act_mean_gain': float(jnp.mean(new_act_gains)),
            'sin_threshold': float(new_act_thresholds[SIN_IDX]),
            'sin_gain': float(new_act_gains[SIN_IDX]),
            # Aggregation intrinsic stats
            'agg_mean_threshold': float(jnp.mean(new_agg_thresholds)),
            'agg_mean_gain': float(jnp.mean(new_agg_gains)),
            # Affinity stats
            'act_avg_affinity': float(jnp.mean(new_act_affinity)),
            'sin_affinity': float(new_act_affinity[SIN_IDX]),
            'agg_avg_affinity': float(jnp.mean(new_agg_affinity)),
            'max_agg_affinity': float(new_agg_affinity[2]),
            'min_agg_affinity': float(new_agg_affinity[3]),
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            'sin_is_memory': bool(new_act_mem_cells[SIN_IDX]),
            # Mutations
            **act_mut,
            **agg_mut,
            # Discovery
            'sin_discovered': new_state['sin_discovered'],
            'sin_discovery_gen': new_state['sin_discovery_gen'],
            'extreme_agg_discovered': new_state['extreme_agg_discovered'],
            'extreme_agg_discovery_gen': new_state['extreme_agg_discovery_gen'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with intrinsic and memory stats."""
        act_mem_count = int(jnp.sum(state['act_memory_cells']))
        agg_mem_count = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_act_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': SIN_IDX in self.get_active_palette(state),
            'has_extreme_aggs': any(idx in self.get_active_agg_palette(state) for idx in CORE_EXTREME_AGGS),
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Intrinsic params
            'act_mean_threshold': float(jnp.mean(state['act_thresholds'])),
            'act_mean_gain': float(jnp.mean(state['act_gains'])),
            'agg_mean_threshold': float(jnp.mean(state['agg_thresholds'])),
            'agg_mean_gain': float(jnp.mean(state['agg_gains'])),
            # Affinity
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
            # Memory cells
            'act_memory_cells': act_mem_count,
            'agg_memory_cells': agg_mem_count,
            'total_memory_cells': act_mem_count + agg_mem_count,
            # Discovery
            'sin_discovered': state['sin_discovered'],
            'sin_discovery_gen': state['sin_discovery_gen'],
            'extreme_agg_discovered': state['extreme_agg_discovered'],
            'extreme_agg_discovery_gen': state['extreme_agg_discovery_gen'],
        }
