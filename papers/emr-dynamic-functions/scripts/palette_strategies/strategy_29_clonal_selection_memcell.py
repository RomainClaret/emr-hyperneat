"""Strategy 29 (memcell): Clonal Selection with Immune Memory Cells (activation-only).

Activation-only clonal selection that adds the immune *memory cell* mechanism on top
of the plain affinity/expression dynamics of ``strategy_29_clonal_selection``.

This variant exists to match the paper's clonal-selection description exactly while
keeping the experiment activation-only (the aggregation palette is NOT evolved here):

    a_i <- 0.98 * a_i + 0.12 * c_i                  (affinity update; c_i = fitness contribution)
    a_i >= 0.75 for 10 consecutive generations  =>  permanent memory cell

Memory cells are decay-resistant (lose at most 5% per generation), exempt from
hypermutation, and guaranteed inclusion in the palette -- a ratchet that locks in
proven functions. This reproduces the activation-side memory-cell logic of
``strategy_29_clonal_selection_symmetric`` WITHOUT its aggregation-domain coupling,
its sin-specific affinity floor, or its discovery boost (those would bias discovery
toward the known answer and are absent from both the paper's description and the
original non-memory dual clonal).

Use with the activation-only EMR module (``hmrhyperneat_dynamic_functions``),
where the runner reads only ``get_active_palette`` and aggregation is fixed at sum.
"""

from typing import Dict, Any, List, Optional, Tuple
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


class ClonalSelectionMemcellStrategy(PaletteEvolutionStrategy):
    """Immune-inspired clonal selection with permanent memory cells (activation-only).

    Functions accumulate affinity from their fitness contribution; high-affinity
    functions proliferate (expression rises) while low-affinity functions decay but
    are never eliminated. Functions that sustain affinity >= ``memory_cell_threshold``
    for ``memory_cell_gens`` consecutive generations become permanent memory cells:
    decay-resistant, exempt from hypermutation, and guaranteed palette inclusion.
    """

    name = "clonal_selection_memcell"
    description = "Immune clonal selection with permanent memory cells (activation-only)"

    def __init__(
        self,
        # Affinity parameters (match paper L163: 0.98 decay, 0.12 learning rate)
        affinity_learning_rate: float = 0.12,
        affinity_decay: float = 0.98,
        affinity_threshold: float = 0.4,   # Threshold for clonal expansion
        # Memory cell parameters (match paper L163: 0.75 affinity for 10 gens)
        memory_cell_threshold: float = 0.75,
        memory_cell_gens: int = 10,
        memory_cell_decay: float = 0.95,   # Memory cells lose at most 5% per generation
        # Expression dynamics
        proliferation_rate: float = 0.25,
        expression_decay: float = 0.08,
        expression_min: float = 0.05,
        expression_max: float = 1.0,
        # Hypermutation
        hypermutation_rate: float = 0.08,
        hypermutation_strength: float = 0.2,
        # Diversity protection
        min_diversity: int = 4,
        diversity_threshold: float = 0.2,
        # Palette selection
        palette_size: int = 6,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        # Affinity
        self.affinity_learning_rate = affinity_learning_rate
        self.affinity_decay = affinity_decay
        self.affinity_threshold = affinity_threshold

        # Memory cells
        self.memory_cell_threshold = memory_cell_threshold
        self.memory_cell_gens = memory_cell_gens
        self.memory_cell_decay = memory_cell_decay

        # Expression
        self.proliferation_rate = proliferation_rate
        self.expression_decay = expression_decay
        self.expression_min = expression_min
        self.expression_max = expression_max

        # Hypermutation
        self.hypermutation_rate = hypermutation_rate
        self.hypermutation_strength = hypermutation_strength

        # Diversity
        self.min_diversity = min_diversity
        self.diversity_threshold = diversity_threshold

        # Selection
        self.palette_size = palette_size

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with affinities, expressions, and memory-cell trackers."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                affinities = affinities.at[i].set(0.5)

        expressions = jnp.ones(NUM_ACTIVATIONS) * self.expression_min
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                expressions = expressions.at[i].set(0.6)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 292929),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Immune state
            'affinities': affinities,
            'expressions': expressions,
            # Memory-cell state
            'memory_counts': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'memory_cells': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_),
            # Tracking
            'clonal_expansions': jnp.zeros(NUM_ACTIVATIONS),
            'hypermutations': jnp.zeros(NUM_ACTIVATIONS),
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_fitness_contribution(
        self, mask: jnp.ndarray, fitness: float, prev_fitness: float,
    ) -> jnp.ndarray:
        """Active functions share credit for fitness improvement."""
        improvement = fitness - prev_fitness
        active = (mask > 0.5).astype(jnp.float32)
        n_active = jnp.maximum(jnp.sum(active), 1.0)
        return active * improvement / n_active

    def _update_affinities(
        self,
        affinities: jnp.ndarray,
        contributions: jnp.ndarray,
        memory_cells: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update affinities with decay, learning, memory-cell protection, and hypermutation."""
        key1, key2 = jax.random.split(key)

        new_affinities = self.affinity_decay * affinities
        new_affinities = new_affinities + self.affinity_learning_rate * contributions

        # Memory cells resist decay: lose at most (1 - memory_cell_decay) per generation
        new_affinities = jnp.where(
            memory_cells,
            jnp.maximum(new_affinities, affinities * self.memory_cell_decay),
            new_affinities,
        )

        # Hypermutation -- but memory cells are stable (exempt)
        mutation_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        mutation_amounts = jax.random.normal(key2, (NUM_ACTIVATIONS,)) * self.hypermutation_strength
        hypermutation_mask = mutation_probs < self.hypermutation_rate
        hypermutation_mask = jnp.logical_and(hypermutation_mask, ~memory_cells)

        new_affinities = jnp.where(
            hypermutation_mask, new_affinities + mutation_amounts, new_affinities
        )

        return jnp.clip(new_affinities, 0.0, 1.0), hypermutation_mask

    def _update_memory_cells(
        self,
        affinities: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Latch a function as a permanent memory cell after sustained high affinity.

        Counts consecutive generations with affinity >= threshold; once a function
        reaches ``memory_cell_gens`` it becomes a memory cell permanently (logical OR).
        """
        above_threshold = affinities >= self.memory_cell_threshold
        new_counts = jnp.where(above_threshold, memory_counts + 1, 0)
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)
        return new_counts, new_memory_cells

    def _update_expressions(
        self,
        expressions: jnp.ndarray,
        affinities: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> jnp.ndarray:
        """High affinity OR memory-cell status -> proliferation; otherwise decay."""
        new_expressions = expressions.copy()
        for i in range(NUM_ACTIVATIONS):
            if bool(memory_cells[i]) or affinities[i] >= self.affinity_threshold:
                new_expressions = new_expressions.at[i].set(
                    expressions[i] * (1 + self.proliferation_rate)
                )
            else:
                new_expressions = new_expressions.at[i].set(
                    expressions[i] * (1 - self.expression_decay)
                )
        return jnp.clip(new_expressions, self.expression_min, self.expression_max)

    def _ensure_diversity(self, expressions: jnp.ndarray) -> jnp.ndarray:
        """Boost lowest-expression functions to keep at least ``min_diversity`` expressible."""
        expressible = jnp.sum(expressions >= self.diversity_threshold)
        if expressible < self.min_diversity:
            n_to_boost = self.min_diversity - int(expressible)
            sorted_indices = jnp.argsort(expressions)
            for i in range(n_to_boost):
                idx = int(sorted_indices[i])
                expressions = expressions.at[idx].set(
                    max(float(expressions[idx]), self.diversity_threshold)
                )
        return expressions

    def _select_palette(
        self, expressions: jnp.ndarray, memory_cells: jnp.ndarray,
    ) -> jnp.ndarray:
        """Top-K by expression, with all memory cells guaranteed inclusion."""
        top_k = min(self.palette_size, NUM_ACTIVATIONS)
        mask = jnp.zeros(NUM_ACTIVATIONS)

        forced = [i for i in range(NUM_ACTIVATIONS) if bool(memory_cells[i])]
        for idx in forced:
            mask = mask.at[idx].set(1.0)

        remaining = top_k - len(forced)
        if remaining > 0:
            expr = expressions
            for idx in forced:
                expr = expr.at[idx].set(-jnp.inf)  # exclude already-forced from ranking
            top_indices = jnp.argsort(expr)[-remaining:]
            for idx in top_indices:
                mask = mask.at[int(idx)].set(1.0)
        return mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with clonal selection + memory-cell dynamics."""
        key, k1 = jax.random.split(state['rng_key'], 2)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # 1. Per-function fitness contributions
        contributions = self._compute_fitness_contribution(
            state['mask'], best_fitness, prev_best_fitness,
        )

        # 2. Update affinities (memory cells from the PREVIOUS generation protect decay/mutation)
        new_affinities, hypermutation_mask = self._update_affinities(
            state['affinities'], contributions, state['memory_cells'], k1,
        )

        # 3. Update memory-cell latches using the NEW affinities
        new_counts, new_memory_cells = self._update_memory_cells(
            new_affinities, state['memory_counts'], state['memory_cells'],
        )

        # 4. Expression dynamics (memory cells always proliferate)
        new_expressions = self._update_expressions(
            state['expressions'], new_affinities, new_memory_cells,
        )
        new_expressions = self._ensure_diversity(new_expressions)

        # 5. Palette selection (memory cells guaranteed)
        new_mask = self._select_palette(new_expressions, new_memory_cells)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        expanding = new_affinities >= self.affinity_threshold
        new_expansions = state['clonal_expansions'] + expanding.astype(jnp.float32)
        new_hypermutations = state['hypermutations'] + hypermutation_mask.astype(jnp.float32)

        fitness_history = (state['fitness_history'] + [best_fitness])[-20:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'affinities': new_affinities,
            'expressions': new_expressions,
            'memory_counts': new_counts,
            'memory_cells': new_memory_cells,
            'clonal_expansions': new_expansions,
            'hypermutations': new_hypermutations,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        active_palette = mask_to_indices(new_mask)
        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'mean_affinity': float(jnp.mean(new_affinities)),
            'max_affinity': float(jnp.max(new_affinities)),
            'n_memory_cells': int(jnp.sum(new_memory_cells)),
            'memory_cell_indices': [i for i in range(NUM_ACTIVATIONS) if bool(new_memory_cells[i])],
            'expanding_count': int(jnp.sum(expanding)),
            'hypermutations_this_gen': int(jnp.sum(hypermutation_mask)),
            'has_sin': 4 in active_palette,
            'sin_affinity': float(new_affinities[4]),
            'sin_is_memory_cell': bool(new_memory_cells[4]),
        }

        return new_state, metrics
