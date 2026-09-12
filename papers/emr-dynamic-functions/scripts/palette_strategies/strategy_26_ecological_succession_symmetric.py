"""Strategy 26 Symmetric: Ecological Succession.

Extends EcologicalSuccessionStrategy with symmetric discovery features:
- Dual succession phases for activation and aggregation
- Aggregation succession: Averaging → Extreme specialists
- Memory cells for climax functions (sin + max/min)
- Cross-domain climax: sin-extreme pairs become ecosystem anchors
- Affinity floors and discovery tracking

Key mechanisms:
1. Pioneer → Intermediate → Climax phase progression per domain
2. Generalist functions early, specialist functions late
3. Memory cells for discovered climax specialists
4. Cross-domain climax detection for sin-extreme synergy

Biological/Ecological basis:
- Pioneer species: Hardy generalists that colonize first
- Intermediate species: Bridge between pioneers and specialists
- Climax species: Specialists that dominate stable ecosystems
- Cross-ecosystem mutualism: Species from different ecosystems cooperate
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


class EcologicalSuccessionSymmetricStrategy(PaletteEvolutionStrategy):
    """Ecological succession with dual phase progression and memory cells.

    Extends activation-only succession to discover both activation AND
    aggregation functions through generalist→specialist progression.

    Key innovations:
    - Dual succession phases (activation and aggregation develop at different rates)
    - Aggregation succession: sum/mean (generalists) → max/min (specialists)
    - Memory cells for climax specialists (sin, burst, max, min)
    - Cross-domain climax pairs become ecosystem anchors
    """

    name = "ecological_succession_symmetric"
    description = "Dual ecological succession with climax specialists and memory cells"

    # Activation function classifications
    ACT_GENERALIST = [0, 1, 2, 5, 6]  # identity, tanh, sigmoid, relu, lrelu
    ACT_SPECIALIST = [4, 11, 12, 13, 15]  # sin, burst, osc_adapt, modulated, log_cosh
    ACT_NEUTRAL = [3, 7, 8, 9, 10, 14, 16, 17]

    # Aggregation function classifications
    AGG_GENERALIST = [0, 1]  # sum, mean (averaging operations)
    AGG_SPECIALIST = [2, 3]  # max, min (extreme operations)
    AGG_NEUTRAL = [4] if NUM_AGGREGATIONS > 4 else []  # Any others

    def __init__(
        self,
        # Phase boundaries - activation
        act_pioneer_end: int = 10,
        act_intermediate_end: int = 30,
        # Phase boundaries - aggregation (longer pioneer for stability)
        agg_pioneer_end: int = 15,
        agg_intermediate_end: int = 40,
        transition_smoothness: float = 5.0,
        # Pioneer phase parameters
        pioneer_mutation_rate: float = 0.22,
        pioneer_generalist_bias: float = 2.0,
        # Intermediate phase parameters
        intermediate_mutation_rate: float = 0.10,
        intermediate_bias: float = 1.0,
        # Climax phase parameters
        climax_mutation_rate: float = 0.04,
        climax_specialist_bias: float = 1.5,
        climax_discovery_protection: float = 0.75,
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
        # Cross-domain climax
        cross_climax_boost: float = 0.15,
        # Affinity
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.01,
        # Constraints
        act_min_active: int = 2,
        act_max_active: int = 6,
        agg_min_active: int = 1,
        agg_max_active: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Ecological Succession Symmetric strategy."""
        # Phase boundaries
        self.act_pioneer_end = act_pioneer_end
        self.act_intermediate_end = act_intermediate_end
        self.agg_pioneer_end = agg_pioneer_end
        self.agg_intermediate_end = agg_intermediate_end
        self.transition_smoothness = transition_smoothness

        # Phase rates
        self.pioneer_mutation_rate = pioneer_mutation_rate
        self.pioneer_generalist_bias = pioneer_generalist_bias
        self.intermediate_mutation_rate = intermediate_mutation_rate
        self.intermediate_bias = intermediate_bias
        self.climax_mutation_rate = climax_mutation_rate
        self.climax_specialist_bias = climax_specialist_bias
        self.climax_discovery_protection = climax_discovery_protection

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

        # Cross-domain
        self.cross_climax_boost = cross_climax_boost

        # Affinity
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # Constraints
        self.act_min_active = act_min_active
        self.act_max_active = act_max_active
        self.agg_min_active = agg_min_active
        self.agg_max_active = agg_max_active
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

        # Build function type lookups
        self.act_function_type = {}
        for i in self.ACT_GENERALIST:
            self.act_function_type[i] = 'generalist'
        for i in self.ACT_SPECIALIST:
            self.act_function_type[i] = 'specialist'
        for i in self.ACT_NEUTRAL:
            self.act_function_type[i] = 'neutral'

        self.agg_function_type = {}
        for i in self.AGG_GENERALIST:
            self.agg_function_type[i] = 'generalist'
        for i in self.AGG_SPECIALIST:
            self.agg_function_type[i] = 'specialist'
        for i in self.AGG_NEUTRAL:
            self.agg_function_type[i] = 'neutral'

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual succession tracking."""
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

        # Discovery tracking
        act_ever_discovered = set(initial_act)
        agg_ever_discovered = set(initial_agg)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_phase': 'pioneer',
            'act_discovered_specialists': [],
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_phase': 'pioneer',
            'agg_discovered_specialists': [],
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
            # Cross-domain climax pairs
            'cross_climax_pairs': [],
            # General state
            'rng_key': jax.random.PRNGKey(seed + 262631),
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

    def _get_phase_params(
        self,
        generation: int,
        pioneer_end: int,
        intermediate_end: int,
    ) -> Dict[str, Any]:
        """Get parameters for current developmental phase."""
        if generation < pioneer_end:
            phase = 'pioneer'
            progress = generation / pioneer_end
            mutation_rate = self.pioneer_mutation_rate
            generalist_bias = self.pioneer_generalist_bias
            specialist_bias = 0.5
            protection_threshold = 0.8
        elif generation < intermediate_end:
            phase = 'intermediate'
            gen_in_phase = generation - pioneer_end
            phase_length = intermediate_end - pioneer_end
            progress = gen_in_phase / phase_length
            transition_factor = min(1.0, gen_in_phase / self.transition_smoothness)

            mutation_rate = (
                self.pioneer_mutation_rate * (1 - transition_factor) +
                self.intermediate_mutation_rate * transition_factor
            )
            generalist_bias = (
                self.pioneer_generalist_bias * (1 - progress) +
                self.intermediate_bias * progress
            )
            specialist_bias = (
                0.5 * (1 - progress) +
                self.intermediate_bias * progress
            )
            protection_threshold = 0.65
        else:
            phase = 'climax'
            gen_in_phase = generation - intermediate_end
            progress = min(1.0, gen_in_phase / 20)
            transition_factor = min(1.0, gen_in_phase / self.transition_smoothness)

            mutation_rate = (
                self.intermediate_mutation_rate * (1 - transition_factor) +
                self.climax_mutation_rate * transition_factor
            )
            generalist_bias = 1.0
            specialist_bias = (
                self.intermediate_bias * (1 - transition_factor) +
                self.climax_specialist_bias * transition_factor
            )
            protection_threshold = self.climax_discovery_protection

        return {
            'phase': phase,
            'progress': progress,
            'mutation_rate': mutation_rate,
            'generalist_bias': generalist_bias,
            'specialist_bias': specialist_bias,
            'protection_threshold': protection_threshold,
        }

    def _update_memory_cells(
        self,
        affinities: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell tracking."""
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

    def _detect_cross_climax_pairs(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> List[Tuple[int, int]]:
        """Detect activation-aggregation climax specialist pairs."""
        act_active = [i for i in mask_to_indices(act_mask) if i in self.ACT_SPECIALIST]
        agg_active = [i for i in mask_to_indices(agg_mask) if i in self.AGG_SPECIALIST]
        pairs = [(a, g) for a in act_active for g in agg_active]
        return pairs

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_improved: bool,
        phase: str,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
        cross_climax_boost_indices: List[int] = None,
    ) -> jnp.ndarray:
        """Update affinity with memory cell protection and discovery boost."""
        newly_discovered = newly_discovered or []
        cross_climax_boost_indices = cross_climax_boost_indices or []

        active = (mask > 0.5).astype(jnp.float32)

        if fitness_improved:
            signal = self.affinity_lr * active
        else:
            signal = -self.affinity_lr * 0.3 * active

        new_affinity = affinity + signal

        # Memory cells resist negative changes
        negative_signal = signal < 0
        memory_protected = jnp.logical_and(negative_signal, memory_cells)
        new_affinity = jnp.where(
            memory_protected,
            affinity * (1 - self.memory_cell_decay_rate),
            new_affinity
        )

        # Decay inactive (stronger in climax)
        decay_rate = self.affinity_decay * (1.5 if phase == 'climax' else 1.0)
        inactive = 1 - active
        # Don't decay memory cells
        decay_mask = jnp.logical_and(inactive > 0.5, ~memory_cells)
        new_affinity = new_affinity - decay_rate * decay_mask * affinity

        # Discovery boost
        for idx in newly_discovered:
            new_affinity = new_affinity.at[idx].set(
                new_affinity[idx] + self.discovery_boost
            )

        # Cross-climax boost
        for idx in cross_climax_boost_indices:
            new_affinity = new_affinity.at[idx].set(
                min(0.95, float(new_affinity[idx]) + self.cross_climax_boost)
            )

        return jnp.clip(new_affinity, 0.05, 0.95)

    def _apply_succession_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        phase_params: Dict,
        function_type_lookup: Dict[int, str],
        discovered_specialists: List[int],
        specialist_indices: List[int],
        memory_cells: jnp.ndarray,
        min_active: int,
        max_active: int,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Apply mutation with succession biasing and memory cell protection."""
        newly_discovered = newly_discovered or []
        key1, key2 = jax.random.split(key)
        n_funcs = len(mask)

        activate_probs = jax.random.uniform(key1, (n_funcs,))
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))

        base_rate = phase_params['mutation_rate']
        protection_threshold = phase_params['protection_threshold']
        phase = phase_params['phase']

        new_mask = mask.copy()
        activated = []
        deactivated = []
        discovery_to_palette = 0

        for i in range(n_funcs):
            aff = float(affinity[i])
            func_type = function_type_lookup.get(i, 'neutral')
            is_memory = bool(memory_cells[i])

            # Get bias based on function type and phase
            if func_type == 'generalist':
                bias = phase_params['generalist_bias']
            elif func_type == 'specialist':
                bias = phase_params['specialist_bias']
            else:
                bias = 1.0

            if mask[i] < 0.5:
                # Inactive: maybe activate
                rate = base_rate * 0.5 * bias * (0.5 + 0.5 * aff)
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

                is_protected = (
                    aff >= protection_threshold or
                    (phase == 'climax' and i in discovered_specialists)
                )

                if is_protected:
                    rate = base_rate * 0.05
                else:
                    inv_bias = 1.0 / max(bias, 0.5)
                    rate = base_rate * 0.4 * (1.0 - aff) * inv_bias

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
            'phase': phase,
        }, discovery_to_palette

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual ecological succession and memory cells."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Get phase parameters for each domain
        act_phase_params = self._get_phase_params(
            generation, self.act_pioneer_end, self.act_intermediate_end
        )
        agg_phase_params = self._get_phase_params(
            generation, self.agg_pioneer_end, self.agg_intermediate_end
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

        # Detect cross-climax pairs
        cross_climax = self._detect_cross_climax_pairs(state['act_mask'], state['agg_mask'])
        act_climax_boost = list(set(p[0] for p in cross_climax))
        agg_climax_boost = list(set(p[1] for p in cross_climax))

        # Update affinities
        new_act_aff = self._update_affinity(
            state['act_affinity'], state['act_mask'], improved,
            act_phase_params['phase'], state['act_memory_cells'],
            act_new_candidates, act_climax_boost
        )
        new_agg_aff = self._update_affinity(
            state['agg_affinity'], state['agg_mask'], improved,
            agg_phase_params['phase'], state['agg_memory_cells'],
            agg_new_candidates, agg_climax_boost
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

        # Track discovered specialists during intermediate phase
        act_discovered = list(state['act_discovered_specialists'])
        agg_discovered = list(state['agg_discovered_specialists'])

        if act_phase_params['phase'] == 'intermediate':
            for i in current_act_palette:
                if i in self.ACT_SPECIALIST and i not in act_discovered:
                    act_discovered.append(i)
        if agg_phase_params['phase'] == 'intermediate':
            for i in current_agg_palette:
                if i in self.AGG_SPECIALIST and i not in agg_discovered:
                    agg_discovered.append(i)

        # Apply succession mutation
        new_act_mask, act_mut_info, act_disc_to_pal = self._apply_succession_mutation(
            k1, state['act_mask'], new_act_aff, act_phase_params,
            self.act_function_type, act_discovered, self.ACT_SPECIALIST,
            new_act_mem_cells, self.act_min_active, self.act_max_active,
            act_new_candidates
        )
        new_agg_mask, agg_mut_info, agg_disc_to_pal = self._apply_succession_mutation(
            k2, state['agg_mask'], new_agg_aff, agg_phase_params,
            self.agg_function_type, agg_discovered, self.AGG_SPECIALIST,
            new_agg_mem_cells, self.agg_min_active, self.agg_max_active,
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
            'act_affinity': new_act_aff,
            'act_phase': act_phase_params['phase'],
            'act_discovered_specialists': act_discovered,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_aff,
            'agg_phase': agg_phase_params['phase'],
            'agg_discovered_specialists': agg_discovered,
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
            # Cross-domain
            'cross_climax_pairs': cross_climax,
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

        # Count by type
        n_act_generalists = sum(1 for i in final_act_palette if i in self.ACT_GENERALIST)
        n_act_specialists = sum(1 for i in final_act_palette if i in self.ACT_SPECIALIST)
        n_agg_generalists = sum(1 for i in final_agg_palette if i in self.AGG_GENERALIST)
        n_agg_specialists = sum(1 for i in final_agg_palette if i in self.AGG_SPECIALIST)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': final_act_palette,
            'current_agg_palette': final_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Phases
            'act_phase': act_phase_params['phase'],
            'agg_phase': agg_phase_params['phase'],
            # Composition
            'n_act_generalists': n_act_generalists,
            'n_act_specialists': n_act_specialists,
            'n_agg_generalists': n_agg_generalists,
            'n_agg_specialists': n_agg_specialists,
            # Sin and extreme agg
            'has_sin': has_sin,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'has_extreme_agg': has_extreme_agg,
            'extreme_agg_affinities': [float(new_agg_aff[idx]) for idx in CORE_EXTREME_AGGS],
            # Cross-climax
            'n_cross_climax_pairs': len(cross_climax),
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
            # Specialists
            'act_discovered_specialists': state['act_discovered_specialists'],
            'agg_discovered_specialists': state['agg_discovered_specialists'],
            # Cross-climax
            'cross_climax_pairs': state['cross_climax_pairs'],
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
