"""Strategy 7 Symmetric: Neuromodulated Symmetric.

Extends NeuromodulatedStrategy with symmetric discovery features:
- Dual neuromodulator channels (separate DA/ACh/NE for act and agg)
- Cross-domain dopamine coupling (success in one domain influences other)
- Memory cells from sustained dopamine (high DA → memory status)
- Affinity floors and discovery tracking for both domains

Key mechanisms:
1. Dopamine (DA): Reward signal → reduce exploration when improving
2. Acetylcholine (ACh): Uncertainty signal → increase exploration when stagnating
3. Norepinephrine (NE): Arousal signal → high plasticity early/when challenged
4. Memory cells: Sustained high DA leads to memory cell status

Biological rationale:
- Neuromodulation: Global signals shape local plasticity
- Dopamine: Reward prediction error drives learning
- Acetylcholine: Uncertainty and attention modulation
- Norepinephrine: Arousal and stress response
- Cross-modal modulation: Success in one modality affects others
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

# Oscillatory activation indices that are "sticky"
OSCILLATORY_INDICES = [4, 11, 12]  # sin, burst, resonator
# Extreme aggregation indices
EXTREME_AGG_INDICES = [2, 3]  # max, min


class NeuromodulatedSymmetricStrategy(PaletteEvolutionStrategy):
    """Neuromodulated strategy with dual channels and memory cells.

    Extends activation-only neuromodulation to discover both activation AND
    aggregation functions through DA/ACh/NE modulation per domain.

    Key innovations:
    - Dual neuromodulator channels for each domain
    - Cross-domain DA coupling (success in activation helps aggregation)
    - Memory cells from sustained dopamine signaling
    - Affinity floors prevent loss of critical functions
    """

    name = "neuromodulated_symmetric"
    description = "Dual neuromodulation with DA/ACh/NE and memory cells"

    def __init__(
        self,
        # Base rates
        base_activate_rate: float = 0.22,
        base_deactivate_rate: float = 0.05,
        # Neuromodulation parameters
        dopamine_sensitivity: float = 0.5,
        acetylcholine_sensitivity: float = 0.3,
        norepinephrine_sensitivity: float = 0.2,
        modulation_ema_alpha: float = 0.3,
        # Cross-domain coupling
        cross_da_coupling: float = 0.3,  # How much DA from one domain affects other
        # Stagnation
        stagnation_threshold: int = 3,
        # Sticky functions
        deactivate_sticky_rate: float = 0.01,
        # Memory cell parameters - lower threshold, faster formation
        memory_cell_da_threshold: float = 0.5,  # DA level to become memory candidate (lowered from 0.75)
        memory_cell_gens: int = 5,  # Reduced from 10 for faster memory formation
        memory_cell_decay_rate: float = 0.05,
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Constraints
        act_min_active: int = 2,
        act_max_active: int = 6,
        agg_min_active: int = 1,
        agg_max_active: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Neuromodulated Symmetric strategy."""
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.dopamine_sensitivity = dopamine_sensitivity
        self.acetylcholine_sensitivity = acetylcholine_sensitivity
        self.norepinephrine_sensitivity = norepinephrine_sensitivity
        self.modulation_ema_alpha = modulation_ema_alpha
        self.cross_da_coupling = cross_da_coupling
        self.stagnation_threshold = stagnation_threshold
        self.deactivate_sticky_rate = deactivate_sticky_rate

        self.memory_cell_da_threshold = memory_cell_da_threshold
        self.memory_cell_gens = memory_cell_gens
        self.memory_cell_decay_rate = memory_cell_decay_rate

        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        self.act_min_active = act_min_active
        self.act_max_active = act_max_active
        self.agg_min_active = agg_min_active
        self.agg_max_active = agg_max_active
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual neuromodulator tracking."""
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

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Per-function DA tracking (for memory cell eligibility)
        act_da_levels = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_da_levels = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Discovery tracking
        act_ever_discovered = set(initial_act)
        agg_ever_discovered = set(initial_agg)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_dopamine': 0.5,
            'act_acetylcholine': 0.5,
            'act_norepinephrine': 1.0,
            'act_da_levels': act_da_levels,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_dopamine': 0.5,
            'agg_acetylcholine': 0.5,
            'agg_norepinephrine': 1.0,
            'agg_da_levels': agg_da_levels,
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
            'rng_key': jax.random.PRNGKey(seed + 77731),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'prev_fitness': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _update_neuromodulators(
        self,
        current_da: float,
        current_ach: float,
        current_ne: float,
        best_fitness: float,
        prev_fitness: float,
        stagnation_count: int,
        generation: int,
        cross_da: float = 0.5,
    ) -> Dict[str, float]:
        """Update neuromodulator levels with cross-domain coupling."""
        alpha = self.modulation_ema_alpha

        # Dopamine: Reward signal
        improvement = best_fitness - prev_fitness
        if prev_fitness > 0:
            relative_improvement = improvement / prev_fitness
        else:
            relative_improvement = improvement

        da_signal = max(0, min(1, 0.5 + relative_improvement * 10))

        # Add cross-domain DA influence
        da_signal = da_signal * (1 - self.cross_da_coupling) + cross_da * self.cross_da_coupling

        new_dopamine = (1 - alpha) * current_da + alpha * da_signal

        # Acetylcholine: Uncertainty signal
        stagnation = stagnation_count / max(self.stagnation_threshold, 1)
        ach_signal = min(1.0, stagnation)
        new_acetylcholine = (1 - alpha) * current_ach + alpha * ach_signal

        # Norepinephrine: Arousal signal
        time_decay = max(0, 1.0 - generation / 50.0)
        challenge = 1.0 - best_fitness
        ne_signal = max(time_decay, challenge * 0.5)
        new_norepinephrine = (1 - alpha) * current_ne + alpha * ne_signal

        return {
            'dopamine': float(new_dopamine),
            'acetylcholine': float(new_acetylcholine),
            'norepinephrine': float(new_norepinephrine),
        }

    def _update_per_function_da(
        self,
        da_levels: jnp.ndarray,
        mask: jnp.ndarray,
        global_da: float,
        affinity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update per-function DA levels based on activity and global DA.

        Increased DA gains to ensure memory cells form reliably.
        """
        active = (mask > 0.5).astype(jnp.float32)

        # Active functions get DA boost proportional to global DA
        # Increased from 0.3 to 0.5 for stronger reward signal
        da_boost = active * global_da * 0.5

        # Also boost based on affinity (successful functions get more DA)
        # Increased from 0.2 to 0.35 for faster memory formation
        affinity_boost = active * affinity * 0.35

        # Decay inactive (reduced to 0.05 for more stability)
        decay = (1 - active) * 0.05

        new_da = da_levels + da_boost + affinity_boost - decay
        return jnp.clip(new_da, 0.0, 1.0)

    def _update_memory_cells_from_da(
        self,
        da_levels: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cells based on sustained high DA."""
        above_threshold = da_levels >= self.memory_cell_da_threshold
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

    def _compute_effective_rates(
        self,
        dopamine: float,
        acetylcholine: float,
        norepinephrine: float,
    ) -> Tuple[float, float]:
        """Compute effective activation/deactivation rates from neuromodulators."""
        da_factor = 1.0 - self.dopamine_sensitivity * (dopamine - 0.5)
        ach_factor = 1.0 + self.acetylcholine_sensitivity * (acetylcholine - 0.5)
        ne_factor = 0.5 + self.norepinephrine_sensitivity * norepinephrine

        effective_activate = self.base_activate_rate * da_factor * ach_factor * ne_factor
        effective_deactivate = self.base_deactivate_rate * (1.0 / max(da_factor, 0.5)) * ne_factor

        effective_activate = max(0.05, min(0.5, effective_activate))
        effective_deactivate = max(0.01, min(0.2, effective_deactivate))

        return effective_activate, effective_deactivate

    def _mutate_palette_neuromodulated(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        effective_activate: float,
        effective_deactivate: float,
        sticky_indices: List[int],
        memory_cells: jnp.ndarray,
        min_active: int,
        max_active: int,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Apply neuromodulated mutation with memory cell protection."""
        newly_discovered = newly_discovered or []
        key1, key2 = jax.random.split(key)
        n_funcs = len(mask)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        discovery_to_palette = 0

        activate_probs = jax.random.uniform(key1, (n_funcs,))
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))

        for i in range(n_funcs):
            is_memory = bool(memory_cells[i])
            aff = float(affinity[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                rate = effective_activate * (0.5 + 0.5 * aff)
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

                # Use sticky rate for oscillatory/extreme functions
                if i in sticky_indices:
                    deact_rate = self.deactivate_sticky_rate
                else:
                    deact_rate = effective_deactivate * (1.0 - aff)

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
            'effective_activate': effective_activate,
            'effective_deactivate': effective_deactivate,
        }, discovery_to_palette

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
    ) -> jnp.ndarray:
        """Update affinity with memory cell protection and discovery boost."""
        newly_discovered = newly_discovered or []
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            delta = 0.1 * fitness_signal * active
        else:
            delta = 0.03 * fitness_signal * active

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

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual neuromodulation and memory cells."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Fitness signal
        fitness_signal = (best_fitness - state['prev_fitness']) / max(0.1, state['prev_fitness'])
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

        # Update neuromodulators with cross-domain coupling
        act_neuromod = self._update_neuromodulators(
            state['act_dopamine'], state['act_acetylcholine'], state['act_norepinephrine'],
            best_fitness, prev_best_fitness, new_stagnation, generation,
            cross_da=state['agg_dopamine']
        )
        agg_neuromod = self._update_neuromodulators(
            state['agg_dopamine'], state['agg_acetylcholine'], state['agg_norepinephrine'],
            best_fitness, prev_best_fitness, new_stagnation, generation,
            cross_da=state['act_dopamine']
        )

        # Update per-function DA levels
        new_act_da_levels = self._update_per_function_da(
            state['act_da_levels'], state['act_mask'],
            act_neuromod['dopamine'], state['act_affinity']
        )
        new_agg_da_levels = self._update_per_function_da(
            state['agg_da_levels'], state['agg_mask'],
            agg_neuromod['dopamine'], state['agg_affinity']
        )

        # Update affinities
        new_act_aff = self._update_affinity(
            state['act_affinity'], state['act_mask'], fitness_signal,
            state['act_memory_cells'], act_new_candidates
        )
        new_agg_aff = self._update_affinity(
            state['agg_affinity'], state['agg_mask'], fitness_signal,
            state['agg_memory_cells'], agg_new_candidates
        )

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update memory cells from DA
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells_from_da(
            new_act_da_levels, state['act_memory_counts'], state['act_memory_cells']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells_from_da(
            new_agg_da_levels, state['agg_memory_counts'], state['agg_memory_cells']
        )

        # Compute effective rates
        act_eff_act, act_eff_deact = self._compute_effective_rates(
            act_neuromod['dopamine'], act_neuromod['acetylcholine'], act_neuromod['norepinephrine']
        )
        agg_eff_act, agg_eff_deact = self._compute_effective_rates(
            agg_neuromod['dopamine'], agg_neuromod['acetylcholine'], agg_neuromod['norepinephrine']
        )

        # Apply neuromodulated mutation
        new_act_mask, act_mut_info, act_disc_to_pal = self._mutate_palette_neuromodulated(
            k1, state['act_mask'], new_act_aff, act_eff_act, act_eff_deact,
            OSCILLATORY_INDICES, new_act_mem_cells,
            self.act_min_active, self.act_max_active, act_new_candidates
        )
        new_agg_mask, agg_mut_info, agg_disc_to_pal = self._mutate_palette_neuromodulated(
            k2, state['agg_mask'], new_agg_aff, agg_eff_act, agg_eff_deact,
            EXTREME_AGG_INDICES, new_agg_mem_cells,
            self.agg_min_active, self.agg_max_active, agg_new_candidates
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
            'act_affinity': new_act_aff,
            'act_dopamine': act_neuromod['dopamine'],
            'act_acetylcholine': act_neuromod['acetylcholine'],
            'act_norepinephrine': act_neuromod['norepinephrine'],
            'act_da_levels': new_act_da_levels,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_aff,
            'agg_dopamine': agg_neuromod['dopamine'],
            'agg_acetylcholine': agg_neuromod['acetylcholine'],
            'agg_norepinephrine': agg_neuromod['norepinephrine'],
            'agg_da_levels': new_agg_da_levels,
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
            'prev_fitness': best_fitness,
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
            # Neuromodulators - activation
            'act_dopamine': act_neuromod['dopamine'],
            'act_acetylcholine': act_neuromod['acetylcholine'],
            'act_norepinephrine': act_neuromod['norepinephrine'],
            # Neuromodulators - aggregation
            'agg_dopamine': agg_neuromod['dopamine'],
            'agg_acetylcholine': agg_neuromod['acetylcholine'],
            'agg_norepinephrine': agg_neuromod['norepinephrine'],
            # Sin and extreme agg
            'has_sin': has_sin,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'sin_da_level': float(new_act_da_levels[SIN_IDX]),
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
            # Neuromodulators
            'act_dopamine': state['act_dopamine'],
            'act_acetylcholine': state['act_acetylcholine'],
            'act_norepinephrine': state['act_norepinephrine'],
            'agg_dopamine': state['agg_dopamine'],
            'agg_acetylcholine': state['agg_acetylcholine'],
            'agg_norepinephrine': state['agg_norepinephrine'],
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
