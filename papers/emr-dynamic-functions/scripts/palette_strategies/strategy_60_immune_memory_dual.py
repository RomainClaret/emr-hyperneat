"""Strategy 60D: Immune Memory Dual (Adaptive Immunity for Both Domains).

Extends ImmuneMemoryStrategy to jointly evolve BOTH activation AND aggregation
function palettes using adaptive immune memory principles.

Key dual mechanisms:
1. Dual memory cells - separate memory formation for act and agg
2. Shared fitness threshold - single success triggers memory in both domains
3. Cross-domain immunity - memory cells in one domain protect partners
4. Coordinated naive pool - unexplored functions tracked separately per domain

Expected: Long-lived protection and rapid recall in both domains
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


class ImmuneMemoryDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with adaptive immune memory.

    Both activation and aggregation functions form memory cells on success.
    Memory provides long-lived protection, with cross-domain immunity
    extending protection to partner functions.
    """

    name = "immune_memory_dual"
    description = "Dual: Adaptive immunity with memory cells in both domains"

    def __init__(
        self,
        # Memory formation
        memory_formation_threshold: float = 0.75,
        memory_cell_lifespan: int = 50,
        memory_protection_strength: float = 0.9,
        # Plasma cells
        plasma_cell_duration: int = 5,
        plasma_cell_boost: float = 1.5,
        # Cross-reactivity
        cross_reactivity_enabled: bool = True,
        cross_reactivity_radius: int = 2,
        cross_protection_strength: float = 0.3,
        cross_decay_rate: float = 0.1,
        # Cross-domain immunity
        cross_domain_immunity_rate: float = 0.15,
        # Naive pool
        naive_exploration_rate: float = 0.1,
        naive_selection_bias: float = 0.8,
        # Clonal expansion
        clonal_expansion_enabled: bool = True,
        expansion_on_recall: float = 0.2,
        # General
        base_mutation_rate: float = 0.1,
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Immune Memory Dual strategy."""
        # Memory
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_cell_lifespan = memory_cell_lifespan
        self.memory_protection_strength = memory_protection_strength

        # Plasma
        self.plasma_cell_duration = plasma_cell_duration
        self.plasma_cell_boost = plasma_cell_boost

        # Cross-reactivity
        self.cross_reactivity_enabled = cross_reactivity_enabled
        self.cross_reactivity_radius = cross_reactivity_radius
        self.cross_protection_strength = cross_protection_strength
        self.cross_decay_rate = cross_decay_rate

        # Cross-domain
        self.cross_domain_immunity_rate = cross_domain_immunity_rate

        # Naive
        self.naive_exploration_rate = naive_exploration_rate
        self.naive_selection_bias = naive_selection_bias

        # Clonal
        self.clonal_expansion_enabled = clonal_expansion_enabled
        self.expansion_on_recall = expansion_on_recall

        # General
        self.base_mutation_rate = base_mutation_rate
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual immune memory."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Exposure tracking
        act_exposure = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_exposure = act_exposure.at[i].set(1.0)

        agg_exposure = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_exposure = agg_exposure.at[i].set(1.0)

        # Cross-domain immunity matrix
        cross_immunity = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_memory': {},
            'act_plasma': {},
            'act_cross_protection': jnp.zeros(NUM_ACTIVATIONS),
            'act_exposure': act_exposure,
            'act_naive_pool': set(range(NUM_ACTIVATIONS)) - set(initial_act),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_memory': {},
            'agg_plasma': {},
            'agg_cross_protection': jnp.zeros(NUM_AGGREGATIONS),
            'agg_exposure': agg_exposure,
            'agg_naive_pool': set(range(NUM_AGGREGATIONS)) - set(initial_agg),
            # Cross-domain
            'cross_immunity': cross_immunity,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 606060),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Stats
            'act_memories_formed': 0,
            'agg_memories_formed': 0,
            'total_recalls': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _get_similar_functions(self, func: int, radius: int, n_funcs: int) -> List[int]:
        """Get functions within similarity radius."""
        similar = []
        for i in range(n_funcs):
            if i != func and abs(i - func) <= radius:
                similar.append(i)
        return similar

    def _update_memory_cells(
        self,
        memory: Dict[int, Dict],
        active_palette: List[int],
        best_fitness: float,
        generation: int,
    ) -> Tuple[Dict[int, Dict], List[int], List[int]]:
        """Update memory cells for a domain."""
        new_memory = dict(memory)
        formed = []
        recalled = []

        if best_fitness >= self.memory_formation_threshold:
            for func in active_palette:
                if func not in new_memory:
                    new_memory[func] = {
                        'formation_gen': generation,
                        'fitness': best_fitness,
                        'protection': self.memory_protection_strength,
                    }
                    formed.append(func)
                else:
                    if self.clonal_expansion_enabled:
                        current = new_memory[func]['protection']
                        new_memory[func]['protection'] = min(
                            current + self.expansion_on_recall, 1.0
                        )
                        recalled.append(func)

        expired = []
        for func, info in new_memory.items():
            age = generation - info['formation_gen']
            if age > self.memory_cell_lifespan:
                expired.append(func)
            elif age > self.memory_cell_lifespan * 0.8:
                decay = (self.memory_cell_lifespan - age) / (self.memory_cell_lifespan * 0.2)
                new_memory[func]['protection'] *= decay

        for func in expired:
            del new_memory[func]

        return new_memory, formed, recalled

    def _update_plasma_cells(
        self,
        plasma: Dict[int, int],
        newly_formed: List[int],
    ) -> Dict[int, int]:
        """Update plasma cells."""
        new_plasma = {}
        for func, gens in plasma.items():
            if gens > 1:
                new_plasma[func] = gens - 1
        for func in newly_formed:
            new_plasma[func] = self.plasma_cell_duration
        return new_plasma

    def _update_cross_protection(
        self,
        cross_protection: jnp.ndarray,
        memory: Dict[int, Dict],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update within-domain cross-reactivity."""
        if not self.cross_reactivity_enabled:
            return cross_protection

        new_cross = cross_protection * (1 - self.cross_decay_rate)

        for func in memory:
            similar = self._get_similar_functions(func, self.cross_reactivity_radius, n_funcs)
            for sim_func in similar:
                current = float(new_cross[sim_func])
                new_cross = new_cross.at[sim_func].set(
                    min(current + self.cross_protection_strength, 1.0)
                )

        return new_cross

    def _update_cross_domain_immunity(
        self,
        cross_immunity: jnp.ndarray,
        act_memory: Dict[int, Dict],
        agg_memory: Dict[int, Dict],
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        best_fitness: float,
    ) -> jnp.ndarray:
        """Update cross-domain immunity based on successful pairs."""
        new_cross = cross_immunity * 0.99

        if best_fitness >= self.memory_formation_threshold:
            active_act = (act_mask > 0.5).astype(jnp.float32)
            active_agg = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active_act, active_agg)
            new_cross = new_cross + self.cross_domain_immunity_rate * co_active

        return jnp.clip(new_cross, 0.0, 1.0)

    def _get_removal_probability(
        self,
        func: int,
        memory: Dict[int, Dict],
        plasma: Dict[int, int],
        cross_protection: jnp.ndarray,
        partner_immunity: jnp.ndarray,
    ) -> float:
        """Get probability of removal."""
        base_prob = 1.0

        if func in memory:
            base_prob *= (1 - memory[func]['protection'])
        if func in plasma:
            base_prob *= 0.5
        base_prob *= (1 - float(cross_protection[func]))
        base_prob *= (1 - float(jnp.mean(partner_immunity)) * 0.3)

        return base_prob

    def _mutate_palette(
        self,
        mask: jnp.ndarray,
        memory: Dict[int, Dict],
        plasma: Dict[int, int],
        cross_protection: jnp.ndarray,
        partner_immunity: jnp.ndarray,
        exposure: jnp.ndarray,
        naive_pool: Set[int],
        stagnation: int,
        key: jax.random.PRNGKey,
        min_active: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Set[int]]:
        """Mutate palette with immune dynamics."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()
        new_exposure = exposure.copy()
        new_naive = set(naive_pool)

        current_palette = mask_to_indices(mask)
        mutation_rate = self.base_mutation_rate * (1 + stagnation * 0.1)

        if jax.random.uniform(key1) < mutation_rate:
            if len(current_palette) > min_active:
                removal_probs = []
                for func in current_palette:
                    prob = self._get_removal_probability(
                        func, memory, plasma, cross_protection, partner_immunity
                    )
                    removal_probs.append(prob)

                total = sum(removal_probs)
                if total > 0.01:
                    removal_probs = [p / total for p in removal_probs]
                    cum_prob = 0
                    sample = float(jax.random.uniform(key2))
                    remove_idx = None
                    for i, prob in enumerate(removal_probs):
                        cum_prob += prob
                        if sample < cum_prob:
                            remove_idx = i
                            break
                    if remove_idx is not None:
                        removed = current_palette[remove_idx]
                        new_mask = new_mask.at[removed].set(0.0)

            available = [i for i in range(n_funcs) if new_mask[i] < 0.5]
            if available:
                naive_available = [i for i in available if i in new_naive]
                if naive_available and jax.random.uniform(key3) < self.naive_selection_bias:
                    add_idx = int(jax.random.randint(key3, (), 0, len(naive_available)))
                    added = naive_available[add_idx]
                else:
                    add_idx = int(jax.random.randint(key3, (), 0, len(available)))
                    added = available[add_idx]

                new_mask = new_mask.at[added].set(1.0)
                new_exposure = new_exposure.at[added].add(1.0)
                new_naive.discard(added)

        return new_mask, new_exposure, new_naive

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual immune memory."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        act_palette = mask_to_indices(state['act_mask'])
        agg_palette = mask_to_indices(state['agg_mask'])

        # Update memory cells
        new_act_mem, act_formed, act_recalled = self._update_memory_cells(
            state['act_memory'], act_palette, best_fitness, generation
        )
        new_agg_mem, agg_formed, agg_recalled = self._update_memory_cells(
            state['agg_memory'], agg_palette, best_fitness, generation
        )

        # Update plasma cells
        new_act_plasma = self._update_plasma_cells(state['act_plasma'], act_formed)
        new_agg_plasma = self._update_plasma_cells(state['agg_plasma'], agg_formed)

        # Update cross-protection
        new_act_cross = self._update_cross_protection(
            state['act_cross_protection'], new_act_mem, NUM_ACTIVATIONS
        )
        new_agg_cross = self._update_cross_protection(
            state['agg_cross_protection'], new_agg_mem, NUM_AGGREGATIONS
        )

        # Update cross-domain immunity
        new_cross_imm = self._update_cross_domain_immunity(
            state['cross_immunity'], new_act_mem, new_agg_mem,
            state['act_mask'], state['agg_mask'], best_fitness
        )

        # Compute partner immunity for each domain
        act_partner_imm = jnp.sum(new_cross_imm, axis=1)
        agg_partner_imm = jnp.sum(new_cross_imm, axis=0)

        # Mutate palettes
        new_act_mask, new_act_exp, new_act_naive = self._mutate_palette(
            state['act_mask'], new_act_mem, new_act_plasma, new_act_cross,
            agg_partner_imm, state['act_exposure'], state['act_naive_pool'],
            new_stagnation, k_act, self.min_active_act, NUM_ACTIVATIONS
        )
        new_agg_mask, new_agg_exp, new_agg_naive = self._mutate_palette(
            state['agg_mask'], new_agg_mem, new_agg_plasma, new_agg_cross,
            act_partner_imm, state['agg_exposure'], state['agg_naive_pool'],
            new_stagnation, k_agg, self.min_active_agg, NUM_AGGREGATIONS
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_memory': new_act_mem,
            'act_plasma': new_act_plasma,
            'act_cross_protection': new_act_cross,
            'act_exposure': new_act_exp,
            'act_naive_pool': new_act_naive,
            'agg_mask': new_agg_mask,
            'agg_memory': new_agg_mem,
            'agg_plasma': new_agg_plasma,
            'agg_cross_protection': new_agg_cross,
            'agg_exposure': new_agg_exp,
            'agg_naive_pool': new_agg_naive,
            'cross_immunity': new_cross_imm,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'act_memories_formed': state['act_memories_formed'] + len(act_formed),
            'agg_memories_formed': state['agg_memories_formed'] + len(agg_formed),
            'total_recalls': state['total_recalls'] + len(act_recalled) + len(agg_recalled),
        }

        new_act_palette = mask_to_indices(new_act_mask)
        new_agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': new_act_palette,
            'current_agg_palette': new_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Memory
            'act_n_memory': len(new_act_mem),
            'agg_n_memory': len(new_agg_mem),
            'act_n_plasma': len(new_act_plasma),
            'agg_n_plasma': len(new_agg_plasma),
            'act_formed_this_gen': act_formed,
            'agg_formed_this_gen': agg_formed,
            # Cross-protection
            'act_mean_cross_protection': float(jnp.mean(new_act_cross)),
            'agg_mean_cross_protection': float(jnp.mean(new_agg_cross)),
            # Cross-domain immunity
            'cross_immunity_mean': float(jnp.mean(new_cross_imm)),
            # Naive
            'act_n_naive': len(new_act_naive),
            'agg_n_naive': len(new_agg_naive),
            # Stats
            'act_total_memories': new_state['act_memories_formed'],
            'agg_total_memories': new_state['agg_memories_formed'],
            'total_recalls': new_state['total_recalls'],
            # Sin status
            'has_sin': 4 in new_act_palette,
            'sin_has_memory': 4 in new_act_mem,
            'sin_cross_protection': float(new_act_cross[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual immune status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_n_memory': len(state['act_memory']),
            'agg_n_memory': len(state['agg_memory']),
            'act_n_plasma': len(state['act_plasma']),
            'agg_n_plasma': len(state['agg_plasma']),
            'act_mean_cross_protection': float(jnp.mean(state['act_cross_protection'])),
            'agg_mean_cross_protection': float(jnp.mean(state['agg_cross_protection'])),
            'cross_immunity_mean': float(jnp.mean(state['cross_immunity'])),
            'act_total_memories': state['act_memories_formed'],
            'agg_total_memories': state['agg_memories_formed'],
            'sin_has_memory': 4 in state['act_memory'],
        }
