"""Strategy 20 Symmetric: Synaptic Tagging.

Extends SynapticTaggingStrategy with symmetric discovery features:
- Dual tagging systems (separate for activation and aggregation)
- Cross-domain tagging (activation success influences aggregation tagging)
- Memory cells from captured functions (captured = permanent memory cell)
- Affinity floors and discovery tracking for both domains

Key mechanisms:
1. Tag creation: Functions active during local improvement get tagged
2. Tag decay: Tags weaken over generations
3. Capture: Global success captures tagged functions (tag → capture = memory cell)
4. Cross-domain capture: Success in one domain can trigger capture in other
5. Memory cells = captured functions with permanent protection

Biological rationale (Frey & Morris, 1997):
- Local synaptic activity creates transient "tags"
- Global events (dopamine) trigger "capture" of tagged synapses
- Only tagged + captured synapses become permanent memory
- Cross-modal consolidation: Visual memory affects motor learning
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


class SynapticTaggingSymmetricStrategy(PaletteEvolutionStrategy):
    """Synaptic tagging and capture with dual domains and memory cells.

    Two-stage learning: local tags must be captured by global success.
    Captured functions become permanent memory cells.

    Key innovations:
    - Dual tagging per domain (activation and aggregation)
    - Cross-domain tagging (success in one domain influences other)
    - Memory cells = captured functions (permanent protection)
    - Affinity floors prevent loss of critical functions
    """

    name = "synaptic_tagging_symmetric"
    description = "Dual synaptic tagging with capture and memory cells"

    def __init__(
        self,
        # Tag creation parameters
        tag_threshold: float = 0.12,            # Local improvement to create tag
        tag_strength_per_event: float = 0.25,   # Strength added per tagging event
        max_tag_strength: float = 1.0,          # Maximum tag strength
        # Tag decay parameters
        tag_decay_rate: float = 0.12,           # Tags decay per generation
        tag_min_threshold: float = 0.1,         # Below this, tag is removed
        # Capture parameters
        capture_window: int = 5,                # Gens a tag is valid for capture
        capture_threshold: float = 0.25,        # Global improvement for capture
        capture_efficiency: float = 0.5,        # Fraction of tag converted to affinity
        capture_bonus: float = 0.2,             # Bonus affinity on capture
        # Cross-domain capture
        cross_capture_weight: float = 0.3,      # How much one domain's capture affects other
        cross_tag_boost: float = 0.15,          # Tag boost from cross-domain success
        # Memory cell parameters (captured = memory cell)
        memory_cell_affinity_min: float = 0.75, # Minimum affinity for memory cells
        memory_cell_decay_rate: float = 0.05,   # Very slow decay for memory cells
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Base learning parameters
        affinity_learning_rate: float = 0.10,   # Base affinity learning rate
        mutation_rate: float = 0.12,            # Base mutation rate
        # Protection
        affinity_protection_threshold: float = 0.55,
        # Constraints
        act_min_active: int = 2,
        act_max_active: int = 6,
        agg_min_active: int = 1,
        agg_max_active: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Synaptic Tagging Symmetric strategy."""
        # Tag creation
        self.tag_threshold = tag_threshold
        self.tag_strength_per_event = tag_strength_per_event
        self.max_tag_strength = max_tag_strength

        # Tag decay
        self.tag_decay_rate = tag_decay_rate
        self.tag_min_threshold = tag_min_threshold

        # Capture
        self.capture_window = capture_window
        self.capture_threshold = capture_threshold
        self.capture_efficiency = capture_efficiency
        self.capture_bonus = capture_bonus

        # Cross-domain
        self.cross_capture_weight = cross_capture_weight
        self.cross_tag_boost = cross_tag_boost

        # Memory cells
        self.memory_cell_affinity_min = memory_cell_affinity_min
        self.memory_cell_decay_rate = memory_cell_decay_rate

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Discovery
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        # Base learning
        self.affinity_learning_rate = affinity_learning_rate
        self.mutation_rate = mutation_rate

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # Constraints
        self.act_min_active = act_min_active
        self.act_max_active = act_max_active
        self.agg_min_active = agg_min_active
        self.agg_max_active = agg_max_active
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual tagging systems."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        act_tag_gen = jnp.ones(NUM_ACTIVATIONS) * -100

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        agg_tag_gen = jnp.ones(NUM_AGGREGATIONS) * -100

        # Memory cells (captured functions)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Discovery tracking
        act_ever_discovered = set(initial_act)
        agg_ever_discovered = set(initial_agg)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_tags': act_tags,
            'act_tag_gen': act_tag_gen,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_tags': agg_tags,
            'agg_tag_gen': agg_tag_gen,
            # Memory cells (captured = memory)
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
            # Discovery tracking
            'act_ever_discovered': act_ever_discovered,
            'agg_ever_discovered': agg_ever_discovered,
            'total_act_discoveries': 0,
            'total_agg_discoveries': 0,
            'discovery_to_palette': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 202032),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_ema': 0.5,
            'fitness_history': [],
            'tag_events': 0,
            'capture_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _create_tags(
        self,
        tags: jnp.ndarray,
        tag_gen: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_improvement: float,
        generation: int,
        cross_boost: float = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Create or strengthen tags for active functions.

        Tags are created when local fitness improves above threshold.
        Cross-domain boost adds to tag strength when the other domain succeeds.
        """
        effective_improvement = fitness_improvement + cross_boost

        if effective_improvement < self.tag_threshold:
            return tags, tag_gen, 0

        new_tags = tags.copy()
        new_tag_gen = tag_gen.copy()
        n_tagged = 0

        active = (mask > 0.5).astype(jnp.float32)

        for i in range(len(tags)):
            if float(active[i]) > 0.5:
                strength_increase = self.tag_strength_per_event * (
                    effective_improvement / max(self.tag_threshold, 0.01)
                )
                new_strength = float(tags[i]) + strength_increase

                new_tags = new_tags.at[i].set(
                    min(self.max_tag_strength, new_strength)
                )
                new_tag_gen = new_tag_gen.at[i].set(float(generation))
                n_tagged += 1

        return new_tags, new_tag_gen, n_tagged

    def _decay_tags(
        self,
        tags: jnp.ndarray,
        tag_gen: jnp.ndarray,
        generation: int,
    ) -> jnp.ndarray:
        """Apply tag decay over time."""
        new_tags = tags.copy()

        for i in range(len(tags)):
            if float(tags[i]) < self.tag_min_threshold:
                continue

            tag_age = generation - int(tag_gen[i])
            effective_decay = self.tag_decay_rate * (1.0 + 0.1 * tag_age)
            new_strength = float(tags[i]) * (1.0 - effective_decay)

            if new_strength < self.tag_min_threshold:
                new_strength = 0.0

            new_tags = new_tags.at[i].set(new_strength)

        return new_tags

    def _attempt_capture(
        self,
        tags: jnp.ndarray,
        tag_gen: jnp.ndarray,
        affinity: jnp.ndarray,
        memory_cells: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_improvement: float,
        generation: int,
        cross_capture_boost: float = 0.0,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Attempt to capture tagged functions on global success.

        Captured functions become memory cells.

        Returns:
            (new_tags, new_affinity, new_memory_cells, n_captured)
        """
        newly_discovered = newly_discovered or []
        effective_improvement = fitness_improvement + cross_capture_boost

        if effective_improvement < self.capture_threshold:
            # No capture, but still apply discovery boost
            new_affinity = affinity.copy()
            for idx in newly_discovered:
                new_affinity = new_affinity.at[idx].set(
                    new_affinity[idx] + self.discovery_boost
                )
            return tags, jnp.clip(new_affinity, 0.0, 1.0), memory_cells, 0

        new_tags = tags.copy()
        new_affinity = affinity.copy()
        new_memory_cells = memory_cells.copy()
        n_captured = 0

        for i in range(len(tags)):
            if memory_cells[i]:
                # Already a memory cell
                continue

            tag_strength = float(tags[i])
            tag_age = generation - int(tag_gen[i])

            # Check if tag is valid for capture
            if tag_strength >= self.tag_min_threshold and tag_age <= self.capture_window:
                # Capture this function! It becomes a memory cell.
                affinity_boost = (
                    tag_strength * self.capture_efficiency +
                    self.capture_bonus
                )
                new_affinity = new_affinity.at[i].set(
                    max(self.memory_cell_affinity_min, float(affinity[i]) + affinity_boost)
                )

                # Mark as memory cell
                new_memory_cells = new_memory_cells.at[i].set(True)
                n_captured += 1

                # Clear the tag
                new_tags = new_tags.at[i].set(0.0)

        # Apply discovery boost
        for idx in newly_discovered:
            new_affinity = new_affinity.at[idx].set(
                new_affinity[idx] + self.discovery_boost
            )

        return new_tags, jnp.clip(new_affinity, 0.0, 1.0), new_memory_cells, n_captured

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update affinity with memory cell protection."""
        new_affinity = affinity.copy()
        active = (mask > 0.5).astype(jnp.float32)

        for i in range(len(affinity)):
            if memory_cells[i]:
                # Memory cells: only allow positive updates, resist decay
                if fitness_signal > 0 and float(active[i]) > 0.5:
                    delta = self.affinity_learning_rate * 0.3 * fitness_signal
                    new_affinity = new_affinity.at[i].set(
                        min(0.95, float(new_affinity[i]) + delta)
                    )
                # Ensure minimum affinity
                if float(new_affinity[i]) < self.memory_cell_affinity_min:
                    new_affinity = new_affinity.at[i].set(self.memory_cell_affinity_min)
            else:
                # Normal learning for non-memory cells
                if float(active[i]) > 0.5:
                    if fitness_signal >= 0:
                        delta = self.affinity_learning_rate * fitness_signal
                    else:
                        delta = self.affinity_learning_rate * 0.3 * fitness_signal
                    new_affinity = new_affinity.at[i].set(
                        max(0.05, min(0.95, float(new_affinity[i]) + delta))
                    )

        return new_affinity

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

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        tags: jnp.ndarray,
        memory_cells: jnp.ndarray,
        min_active: int,
        max_active: int,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Apply mutation with memory cell protection."""
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
            aff = float(affinity[i])
            is_memory = bool(memory_cells[i])
            tag_boost = float(tags[i]) * 0.2

            if mask[i] < 0.5:
                # Inactive: maybe activate
                if is_memory:
                    # Memory cells that were deactivated can be re-activated easily
                    effective_rate = self.mutation_rate * 2.0
                else:
                    effective_rate = self.mutation_rate * (0.5 + aff + tag_boost)

                current_active = int(jnp.sum(new_mask > 0.5))
                if activate_probs[i] < effective_rate and current_active < max_active:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
                    if i in newly_discovered:
                        discovery_to_palette += 1
            else:
                # Active: maybe deactivate
                if is_memory:
                    # Memory cells (captured) almost never deactivate
                    continue

                protection = aff + tag_boost
                if protection >= self.affinity_protection_threshold:
                    deact_rate = self.mutation_rate * 0.1
                else:
                    deact_rate = self.mutation_rate * (1.0 - protection)

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
        }, discovery_to_palette

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual synaptic tagging and memory cells."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Compute fitness signals
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        local_improvement = best_fitness - prev_best_fitness
        global_improvement = best_fitness - state['fitness_ema']
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

        # Step 1: Decay existing tags
        decayed_act_tags = self._decay_tags(state['act_tags'], state['act_tag_gen'], generation)
        decayed_agg_tags = self._decay_tags(state['agg_tags'], state['agg_tag_gen'], generation)

        # Step 2: Create/strengthen tags with cross-domain boost
        act_cross_boost = self.cross_tag_boost if jnp.sum(state['agg_memory_cells']) > 0 else 0.0
        agg_cross_boost = self.cross_tag_boost if jnp.sum(state['act_memory_cells']) > 0 else 0.0

        new_act_tags, new_act_tag_gen, n_act_tagged = self._create_tags(
            decayed_act_tags, state['act_tag_gen'], state['act_mask'],
            local_improvement, generation, act_cross_boost
        )
        new_agg_tags, new_agg_tag_gen, n_agg_tagged = self._create_tags(
            decayed_agg_tags, state['agg_tag_gen'], state['agg_mask'],
            local_improvement, generation, agg_cross_boost
        )

        # Step 3: Attempt capture with cross-domain boost
        act_capture_boost = self.cross_capture_weight * global_improvement if jnp.any(state['agg_memory_cells']) else 0.0
        agg_capture_boost = self.cross_capture_weight * global_improvement if jnp.any(state['act_memory_cells']) else 0.0

        new_act_tags, new_act_aff, new_act_mem, n_act_captured = self._attempt_capture(
            new_act_tags, new_act_tag_gen, state['act_affinity'], state['act_memory_cells'],
            state['act_mask'], global_improvement, generation, act_capture_boost, act_new_candidates
        )
        new_agg_tags, new_agg_aff, new_agg_mem, n_agg_captured = self._attempt_capture(
            new_agg_tags, new_agg_tag_gen, state['agg_affinity'], state['agg_memory_cells'],
            state['agg_mask'], global_improvement, generation, agg_capture_boost, agg_new_candidates
        )

        # Step 4: Update affinity normally
        new_act_aff = self._update_affinity(new_act_aff, state['act_mask'], new_act_mem, fitness_signal)
        new_agg_aff = self._update_affinity(new_agg_aff, state['agg_mask'], new_agg_mem, fitness_signal)

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Step 5: Apply mutation
        new_act_mask, act_mut_info, act_disc_to_pal = self._mutate_palette(
            k1, state['act_mask'], new_act_aff, new_act_tags, new_act_mem,
            self.act_min_active, self.act_max_active, act_new_candidates
        )
        new_agg_mask, agg_mut_info, agg_disc_to_pal = self._mutate_palette(
            k2, state['agg_mask'], new_agg_aff, new_agg_tags, new_agg_mem,
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
            'act_tags': new_act_tags,
            'act_tag_gen': new_act_tag_gen,
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_aff,
            'agg_tags': new_agg_tags,
            'agg_tag_gen': new_agg_tag_gen,
            # Memory cells
            'act_memory_cells': new_act_mem,
            'agg_memory_cells': new_agg_mem,
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
            'fitness_ema': new_fitness_ema,
            'fitness_history': fitness_history,
            'tag_events': state['tag_events'] + n_act_tagged + n_agg_tagged,
            'capture_events': state['capture_events'] + n_act_captured + n_agg_captured,
        }

        # Check sin and extreme agg retention
        has_sin = SIN_IDX in final_act_palette
        has_extreme_agg = any(idx in final_agg_palette for idx in CORE_EXTREME_AGGS)

        # Count tagged functions
        act_tagged = [i for i in range(NUM_ACTIVATIONS) if float(new_act_tags[i]) >= self.tag_min_threshold]
        agg_tagged = [i for i in range(NUM_AGGREGATIONS) if float(new_agg_tags[i]) >= self.tag_min_threshold]

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': final_act_palette,
            'current_agg_palette': final_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Tagging stats
            'n_act_tagged_this_gen': n_act_tagged,
            'n_agg_tagged_this_gen': n_agg_tagged,
            'n_act_captured_this_gen': n_act_captured,
            'n_agg_captured_this_gen': n_agg_captured,
            'act_tagged_functions': act_tagged,
            'agg_tagged_functions': agg_tagged,
            # Sin and extreme agg
            'has_sin': has_sin,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'sin_tag': float(new_act_tags[SIN_IDX]),
            'sin_is_memory': bool(new_act_mem[SIN_IDX]),
            'has_extreme_agg': has_extreme_agg,
            'extreme_agg_affinities': [float(new_agg_aff[idx]) for idx in CORE_EXTREME_AGGS],
            # Discovery
            'new_act_discoveries': new_act_discoveries,
            'new_agg_discoveries': new_agg_discoveries,
            'total_act_discoveries': new_state['total_act_discoveries'],
            'total_agg_discoveries': new_state['total_agg_discoveries'],
            'discovery_to_palette': new_state['discovery_to_palette'],
            # Memory cells
            'act_memory_cell_count': int(jnp.sum(new_act_mem)),
            'agg_memory_cell_count': int(jnp.sum(new_agg_mem)),
            # Global stats
            'total_tag_events': new_state['tag_events'],
            'total_capture_events': new_state['capture_events'],
        }
        metrics.update(act_mut_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        # Count tagged functions
        act_tagged = [i for i in range(NUM_ACTIVATIONS) if float(state['act_tags'][i]) >= self.tag_min_threshold]
        agg_tagged = [i for i in range(NUM_AGGREGATIONS) if float(state['agg_tags'][i]) >= self.tag_min_threshold]

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
            # Tagging
            'act_tagged_functions': act_tagged,
            'agg_tagged_functions': agg_tagged,
            'n_act_tagged': len(act_tagged),
            'n_agg_tagged': len(agg_tagged),
            'sin_affinity': float(state['act_affinity'][SIN_IDX]),
            'sin_tag': float(state['act_tags'][SIN_IDX]),
            'sin_is_memory': bool(state['act_memory_cells'][SIN_IDX]),
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
            # Global stats
            'total_tag_events': state['tag_events'],
            'total_capture_events': state['capture_events'],
        }
