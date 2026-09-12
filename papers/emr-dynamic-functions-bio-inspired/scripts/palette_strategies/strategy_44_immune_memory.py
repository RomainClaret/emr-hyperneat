"""Strategy 44: Immune Memory (Adaptive Immunity for Palettes).

Implements adaptive immune memory principles for palette evolution. Functions
are tested like antigens, successful ones form memory cells with long-lived
protection, and similar functions share protection through cross-reactivity.

Biological Basis:
- B-cells and T-cells form adaptive immune memory after pathogen exposure
- Memory cells provide rapid response to previously encountered antigens
- Cross-reactivity allows response to similar (but not identical) threats
- Plasma cells provide initial amplified response
- Memory cells have long lifespans (years in humans)
- Primary response is slow, secondary response is rapid

Key Insight:
- Current strategies lack explicit memory formation mechanisms
- Immune system provides sophisticated memory with similarity-based lookup
- Successful functions deserve long-term protection like memory cells
- Novel functions get more chances (like naive lymphocytes encountering new antigens)
- Cross-reactivity enables generalization from known-good to similar functions

Immune Mechanism:
    # On success: form memory cells for active functions
    if best_fitness >= memory_formation_threshold:
        for func in active_palette:
            if func not in memory_cells:
                memory_cells[func] = {
                    'formation_gen': generation,
                    'fitness_at_formation': best_fitness,
                    'protection_level': 1.0,
                }
                # Plasma cell response: temporary boost
                plasma_response[func] = plasma_cell_boost

    # Cross-reactivity: protect similar functions
    for memory_func in memory_cells:
        for similar_func in get_similar_functions(memory_func, radius):
            cross_protection[similar_func] = min(
                cross_protection[similar_func] + cross_reactivity_strength,
                max_protection
            )

    # Memory decay (but long-lived)
    for func in memory_cells:
        age = generation - memory_cells[func]['formation_gen']
        if age > memory_cell_lifespan:
            del memory_cells[func]

Expected improvements:
- Long-lived protection for proven functions
- Rapid re-activation of previously successful functions
- Generalization through cross-reactivity
- Better retention of useful discoveries
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    DEFAULT_PALETTE_INDICES,
)


class ImmuneMemoryStrategy(PaletteEvolutionStrategy):
    """Adaptive immune memory for palette evolution.

    Functions are treated like antigens - successful encounters lead to
    memory cell formation providing long-lived protection. Cross-reactivity
    allows similar functions to benefit from shared protection.
    """

    name = "immune_memory"
    description = "Adaptive immunity with memory cells and cross-reactivity"

    def __init__(
        self,
        # Memory formation
        memory_formation_threshold: float = 0.75,  # Fitness threshold for memory
        memory_cell_lifespan: int = 50,            # Generations before memory decay
        memory_protection_strength: float = 0.9,   # Protection from removal [0, 1]
        # Plasma cells (initial response)
        plasma_cell_duration: int = 5,             # Gens of amplified response
        plasma_cell_boost: float = 1.5,            # Weight multiplier during plasma
        # Cross-reactivity (generalization)
        cross_reactivity_enabled: bool = True,
        cross_reactivity_radius: int = 2,          # Function "similarity" radius
        cross_protection_strength: float = 0.3,    # Protection from cross-reactivity
        cross_decay_rate: float = 0.1,             # Cross-protection decay
        # Naive lymphocyte pool (exploration)
        naive_exploration_rate: float = 0.1,       # Chance to try new "antigens"
        naive_selection_bias: float = 0.8,         # Bias toward unexposed functions
        # Clonal expansion (on success)
        clonal_expansion_enabled: bool = True,
        expansion_on_recall: float = 0.2,          # Extra protection on re-encounter
        # General
        base_mutation_rate: float = 0.1,
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Immune Memory strategy.

        Args:
            memory_formation_threshold: Fitness needed to form memory cells
            memory_cell_lifespan: How long memory cells persist
            memory_protection_strength: How much memory protects from removal
            plasma_cell_duration: Duration of amplified initial response
            plasma_cell_boost: Weight multiplier during plasma phase
            cross_reactivity_enabled: Enable cross-protection for similar functions
            cross_reactivity_radius: "Distance" for cross-reactivity
            cross_protection_strength: Protection level from cross-reactivity
            cross_decay_rate: How fast cross-protection decays
            naive_exploration_rate: Probability of trying new functions
            naive_selection_bias: Bias toward unexposed functions
            clonal_expansion_enabled: Enable boosted response on re-encounter
            expansion_on_recall: Protection boost on memory recall
        """
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

        # Naive
        self.naive_exploration_rate = naive_exploration_rate
        self.naive_selection_bias = naive_selection_bias

        # Clonal expansion
        self.clonal_expansion_enabled = clonal_expansion_enabled
        self.expansion_on_recall = expansion_on_recall

        # General
        self.base_mutation_rate = base_mutation_rate
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with immune memory tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Exposure tracking
        exposure_count = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                exposure_count = exposure_count.at[i].set(1.0)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 444444),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Memory cells: {func_idx: {'formation_gen': int, 'fitness': float, 'protection': float}}
            'memory_cells': {},
            # Plasma cells: {func_idx: gens_remaining}
            'plasma_cells': {},
            # Cross-protection: per-function cross-reactivity protection
            'cross_protection': jnp.zeros(NUM_ACTIVATIONS),
            # Exposure tracking
            'exposure_count': exposure_count,
            'naive_pool': set(range(NUM_ACTIVATIONS)) - set(initial),  # Unexposed
            # History
            'memory_formation_events': [],  # (gen, func, fitness)
            'recall_events': [],            # (gen, func) - re-encountering memory
            'previous_mask': mask,
            'fitness_history': [],
            # Stats
            'total_memories_formed': 0,
            'total_recalls': 0,
            'peak_memory_count': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette."""
        return mask_to_indices(state['mask'])

    def _get_similar_functions(self, func: int, radius: int) -> List[int]:
        """Get functions within similarity radius.

        Uses simple function index distance as similarity measure.
        In practice, this creates groups of "related" functions.
        """
        similar = []
        for i in range(NUM_ACTIVATIONS):
            if i != func and abs(i - func) <= radius:
                similar.append(i)
        return similar

    def _update_memory_cells(
        self,
        memory_cells: Dict[int, Dict],
        active_palette: List[int],
        best_fitness: float,
        generation: int,
    ) -> Tuple[Dict[int, Dict], List[int], List[int]]:
        """Update memory cells - form new and decay old."""
        new_memory = dict(memory_cells)
        formed = []
        recalled = []

        # Form memory for successful palette
        if best_fitness >= self.memory_formation_threshold:
            for func in active_palette:
                if func not in new_memory:
                    # New memory formation
                    new_memory[func] = {
                        'formation_gen': generation,
                        'fitness': best_fitness,
                        'protection': self.memory_protection_strength,
                    }
                    formed.append(func)
                else:
                    # Memory recall - boost protection (clonal expansion)
                    if self.clonal_expansion_enabled:
                        current_protection = new_memory[func]['protection']
                        new_memory[func]['protection'] = min(
                            current_protection + self.expansion_on_recall,
                            1.0
                        )
                        recalled.append(func)

        # Decay old memories
        expired = []
        for func, info in new_memory.items():
            age = generation - info['formation_gen']
            if age > self.memory_cell_lifespan:
                expired.append(func)
            elif age > self.memory_cell_lifespan * 0.8:
                # Gradual decay near end of life
                decay_factor = (self.memory_cell_lifespan - age) / (self.memory_cell_lifespan * 0.2)
                new_memory[func]['protection'] *= decay_factor

        for func in expired:
            del new_memory[func]

        return new_memory, formed, recalled

    def _update_plasma_cells(
        self,
        plasma_cells: Dict[int, int],
        newly_formed: List[int],
    ) -> Dict[int, int]:
        """Update plasma cells - create new and decay existing."""
        new_plasma = {}

        # Existing plasma cells decay
        for func, gens_remaining in plasma_cells.items():
            if gens_remaining > 1:
                new_plasma[func] = gens_remaining - 1

        # New memory formation triggers plasma response
        for func in newly_formed:
            new_plasma[func] = self.plasma_cell_duration

        return new_plasma

    def _update_cross_protection(
        self,
        cross_protection: jnp.ndarray,
        memory_cells: Dict[int, Dict],
    ) -> jnp.ndarray:
        """Update cross-reactivity protection."""
        if not self.cross_reactivity_enabled:
            return cross_protection

        # Decay existing cross-protection
        new_cross = cross_protection * (1 - self.cross_decay_rate)

        # Add cross-protection from memory cells
        for func in memory_cells:
            similar = self._get_similar_functions(func, self.cross_reactivity_radius)
            for similar_func in similar:
                current = float(new_cross[similar_func])
                new_cross = new_cross.at[similar_func].set(
                    min(current + self.cross_protection_strength, 1.0)
                )

        return new_cross

    def _get_removal_probability(
        self,
        func: int,
        memory_cells: Dict[int, Dict],
        plasma_cells: Dict[int, int],
        cross_protection: jnp.ndarray,
    ) -> float:
        """Get probability that a function can be removed."""
        base_removal_prob = 1.0

        # Memory protection
        if func in memory_cells:
            protection = memory_cells[func]['protection']
            base_removal_prob *= (1 - protection)

        # Plasma protection (temporary boost)
        if func in plasma_cells:
            base_removal_prob *= (1 - 0.5)  # 50% reduction during plasma

        # Cross-protection
        cross_prot = float(cross_protection[func])
        base_removal_prob *= (1 - cross_prot)

        return base_removal_prob

    def _mutate_palette(
        self,
        mask: jnp.ndarray,
        memory_cells: Dict[int, Dict],
        plasma_cells: Dict[int, int],
        cross_protection: jnp.ndarray,
        exposure_count: jnp.ndarray,
        naive_pool: Set[int],
        stagnation: int,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Set[int]]:
        """Mutate palette with immune-inspired dynamics."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()
        new_exposure = exposure_count.copy()
        new_naive = set(naive_pool)

        current_palette = mask_to_indices(mask)
        mutation_rate = self.base_mutation_rate * (1 + stagnation * 0.1)

        # Should we mutate?
        if jax.random.uniform(key1) < mutation_rate:
            # Try to remove a function (if not protected)
            if len(current_palette) > self.min_active:
                # Calculate removal probabilities
                removal_probs = []
                for func in current_palette:
                    prob = self._get_removal_probability(
                        func, memory_cells, plasma_cells, cross_protection
                    )
                    removal_probs.append(prob)

                # Normalize
                total = sum(removal_probs)
                if total > 0.01:  # Some removal is possible
                    removal_probs = [p / total for p in removal_probs]

                    # Sample removal
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

            # Add a new function (biased toward naive/unexposed)
            available = [i for i in range(NUM_ACTIVATIONS) if new_mask[i] < 0.5]
            if available:
                # Bias toward naive (unexposed) functions
                naive_available = [i for i in available if i in new_naive]
                exposed_available = [i for i in available if i not in new_naive]

                if naive_available and jax.random.uniform(key3) < self.naive_selection_bias:
                    # Select from naive pool
                    add_idx = int(jax.random.randint(key3, (), 0, len(naive_available)))
                    added = naive_available[add_idx]
                else:
                    # Select from any available
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
        """Update with immune memory dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        current_palette = mask_to_indices(state['mask'])

        # Step 1: Update memory cells
        new_memory, formed, recalled = self._update_memory_cells(
            state['memory_cells'],
            current_palette,
            best_fitness,
            generation,
        )

        # Step 2: Update plasma cells
        new_plasma = self._update_plasma_cells(state['plasma_cells'], formed)

        # Step 3: Update cross-protection
        new_cross = self._update_cross_protection(
            state['cross_protection'],
            new_memory,
        )

        # Step 4: Mutate palette with immune dynamics
        new_mask, new_exposure, new_naive = self._mutate_palette(
            state['mask'],
            new_memory,
            new_plasma,
            new_cross,
            state['exposure_count'],
            state['naive_pool'],
            new_stagnation,
            k1,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update history
        memory_events = list(state['memory_formation_events'])
        for func in formed:
            memory_events.append((generation, func, best_fitness))
        if len(memory_events) > 100:
            memory_events = memory_events[-100:]

        recall_events = list(state['recall_events'])
        for func in recalled:
            recall_events.append((generation, func))
        if len(recall_events) > 100:
            recall_events = recall_events[-100:]

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        peak_memory = max(state['peak_memory_count'], len(new_memory))

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Immune state
            'memory_cells': new_memory,
            'plasma_cells': new_plasma,
            'cross_protection': new_cross,
            'exposure_count': new_exposure,
            'naive_pool': new_naive,
            # History
            'memory_formation_events': memory_events,
            'recall_events': recall_events,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Stats
            'total_memories_formed': state['total_memories_formed'] + len(formed),
            'total_recalls': state['total_recalls'] + len(recalled),
            'peak_memory_count': peak_memory,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Memory cell info
        memory_info = [
            (func, info['protection'], generation - info['formation_gen'])
            for func, info in new_memory.items()
        ]

        # Cross-protection levels
        top_cross_idx = jnp.argsort(new_cross)[-5:][::-1]
        top_cross = [(int(i), float(new_cross[i])) for i in top_cross_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Memory
            'n_memory_cells': len(new_memory),
            'n_plasma_cells': len(new_plasma),
            'memory_cells': memory_info,
            'formed_this_gen': formed,
            'recalled_this_gen': recalled,
            # Cross-reactivity
            'top_cross_protection': top_cross,
            'mean_cross_protection': float(jnp.mean(new_cross)),
            # Naive pool
            'n_naive': len(new_naive),
            'naive_pool': list(new_naive)[:10],  # First 10
            # Stats
            'total_memories_formed': new_state['total_memories_formed'],
            'total_recalls': new_state['total_recalls'],
            'peak_memory_count': peak_memory,
            'memory_utilization': len(new_memory) / NUM_ACTIVATIONS,
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_has_memory': 4 in new_memory,
            'sin_cross_protection': float(new_cross[4]),
            'sin_exposure': float(new_exposure[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with immune status."""
        palette = self.get_active_palette(state)
        memory = state['memory_cells']
        cross = state['cross_protection']
        exposure = state['exposure_count']

        # Memory details
        memory_details = [
            (func, info['protection'], state['generation'] - info['formation_gen'])
            for func, info in memory.items()
        ]

        # Top cross-protection
        top_idx = jnp.argsort(cross)[-5:][::-1]
        top_cross = [(int(i), float(cross[i])) for i in top_idx]

        # Most exposed
        top_exposed_idx = jnp.argsort(exposure)[-5:][::-1]
        top_exposed = [(int(i), float(exposure[i])) for i in top_exposed_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Memory
            'n_memory_cells': len(memory),
            'memory_cells': memory_details,
            'n_plasma_cells': len(state['plasma_cells']),
            # Cross-reactivity
            'top_cross_protection': top_cross,
            'mean_cross_protection': float(jnp.mean(cross)),
            # Naive
            'n_naive': len(state['naive_pool']),
            # Stats
            'total_memories_formed': state['total_memories_formed'],
            'total_recalls': state['total_recalls'],
            'peak_memory_count': state['peak_memory_count'],
            # Exposure
            'top_exposed': top_exposed,
            # Sin-specific
            'sin_has_memory': 4 in memory,
            'sin_cross_protection': float(cross[4]),
        }
