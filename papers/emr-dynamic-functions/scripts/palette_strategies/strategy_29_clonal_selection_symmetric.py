"""Strategy 29 Symmetric: Clonal Selection.

Extends ClonalSelectionDualStrategy with symmetric discovery features:
- Sin affinity floor (protected minimum)
- Extreme aggregation affinity floor (protected minimum)
- Discovery tracking for both domains
- Discovery-to-palette bridging
- Memory cell status for cross-task functions

Key innovations:
- Affinity floors prevent loss of critical functions (sin, max, min)
- Discovery tracking measures when new functions are found
- Discovery boost ensures discovered functions enter palette
- Memory cell concept: Functions that work across tasks gain permanent protection

Biological analogy:
- Clonal selection: High-affinity antibodies proliferate
- Immunological memory: Successful responses create memory cells
- Cross-reactive immunity: Antibodies that work against multiple pathogens
- Affinity maturation: Gradual improvement through hypermutation
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


class ClonalSelectionSymmetricStrategy(PaletteEvolutionStrategy):
    """Immune-inspired dual palette evolution with discovery.

    Both activation and aggregation functions have affinities that determine
    how well they match the current problem. High-affinity functions proliferate
    while low-affinity ones decay but aren't eliminated.

    Enhancements:
    - Sin affinity floor: sin never drops below threshold
    - Extreme agg affinity floor: max/min never drop below threshold
    - Discovery tracking: monitors when new functions enter palette
    - Discovery boost: newly discovered functions get one-time affinity boost
    - Memory cells: Functions that maintain high affinity across tasks
    """

    name = "clonal_selection_symmetric"
    description = "Immune-inspired clonal selection with discovery"

    def __init__(
        self,
        # Affinity parameters - activation
        act_affinity_lr: float = 0.12,
        act_affinity_decay: float = 0.98,
        act_affinity_threshold: float = 0.4,
        # Affinity parameters - aggregation
        agg_affinity_lr: float = 0.10,
        agg_affinity_decay: float = 0.97,
        agg_affinity_threshold: float = 0.35,
        # Expression dynamics - activation
        act_proliferation_rate: float = 0.25,
        act_expression_decay: float = 0.08,
        act_expression_min: float = 0.05,
        act_expression_max: float = 1.0,
        # Expression dynamics - aggregation
        agg_proliferation_rate: float = 0.20,
        agg_expression_decay: float = 0.10,
        agg_expression_min: float = 0.08,
        agg_expression_max: float = 1.0,
        # Hypermutation
        hypermutation_rate: float = 0.08,
        hypermutation_strength: float = 0.2,
        cross_hypermutation_prob: float = 0.3,
        # Cross-domain
        cross_learning_rate: float = 0.05,
        cross_boost_factor: float = 0.15,
        # Diversity protection
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        diversity_threshold: float = 0.2,
        # Palette selection
        act_palette_size: int = 6,
        agg_palette_size: int = 4,
        selection_method: str = "top_k",
        # Constraints
        min_active_act: int = 2,
        max_active_act: int = 6,
        min_active_agg: int = 1,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Memory cell parameters
        memory_cell_threshold: float = 0.75,  # Affinity to become memory cell
        memory_cell_gens: int = 10,  # Generations to maintain threshold
    ):
        """Initialize Clonal Selection Symmetric strategy."""
        # Affinity - activation
        self.act_affinity_lr = act_affinity_lr
        self.act_affinity_decay = act_affinity_decay
        self.act_affinity_threshold = act_affinity_threshold

        # Affinity - aggregation
        self.agg_affinity_lr = agg_affinity_lr
        self.agg_affinity_decay = agg_affinity_decay
        self.agg_affinity_threshold = agg_affinity_threshold

        # Expression - activation
        self.act_proliferation_rate = act_proliferation_rate
        self.act_expression_decay = act_expression_decay
        self.act_expression_min = act_expression_min
        self.act_expression_max = act_expression_max

        # Expression - aggregation
        self.agg_proliferation_rate = agg_proliferation_rate
        self.agg_expression_decay = agg_expression_decay
        self.agg_expression_min = agg_expression_min
        self.agg_expression_max = agg_expression_max

        # Hypermutation
        self.hypermutation_rate = hypermutation_rate
        self.hypermutation_strength = hypermutation_strength
        self.cross_hypermutation_prob = cross_hypermutation_prob

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_boost_factor = cross_boost_factor

        # Diversity
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg
        self.diversity_threshold = diversity_threshold

        # Selection
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.selection_method = selection_method

        # Constraints
        self.min_active_act = min_active_act
        self.max_active_act = max_active_act
        self.min_active_agg = min_active_agg
        self.max_active_agg = max_active_agg

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

        # parameters
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot
        self.memory_cell_threshold = memory_cell_threshold
        self.memory_cell_gens = memory_cell_gens

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual affinities and tracking."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        act_expressions = jnp.ones(NUM_ACTIVATIONS) * self.act_expression_min
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
                act_expressions = act_expressions.at[i].set(0.6)

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        agg_expressions = jnp.ones(NUM_AGGREGATIONS) * self.agg_expression_min
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)
                agg_expressions = agg_expressions.at[i].set(0.6)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Discovery tracking
        act_ever_discovered = set(initial_act)
        agg_ever_discovered = set(initial_agg)

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinities': act_affinities,
            'act_expressions': act_expressions,
            'act_expansions': jnp.zeros(NUM_ACTIVATIONS),
            'act_hypermutations': jnp.zeros(NUM_ACTIVATIONS),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinities': agg_affinities,
            'agg_expressions': agg_expressions,
            'agg_expansions': jnp.zeros(NUM_AGGREGATIONS),
            'agg_hypermutations': jnp.zeros(NUM_AGGREGATIONS),
            # Cross-domain
            'cross_affinity': cross_affinity,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 292929),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Discovery tracking
            'act_ever_discovered': act_ever_discovered,
            'agg_ever_discovered': agg_ever_discovered,
            'total_act_discoveries': 0,
            'total_agg_discoveries': 0,
            'discovery_to_palette': 0,
            # Memory cells
            'act_memory_counts': act_memory_counts,
            'agg_memory_counts': agg_memory_counts,
            'act_memory_cells': act_memory_cells,
            'agg_memory_cells': agg_memory_cells,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _compute_fitness_contributions(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute per-function fitness contributions for both domains."""
        improvement = fitness - prev_fitness

        # Activation contributions
        active_act = (act_mask > 0.5).astype(jnp.float32)
        n_active_act = max(jnp.sum(active_act), 1.0)
        act_contrib = active_act * improvement / n_active_act

        # Aggregation contributions
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        n_active_agg = max(jnp.sum(active_agg), 1.0)
        agg_contrib = active_agg * improvement / n_active_agg

        return act_contrib, agg_contrib

    def _apply_affinity_floors(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for critical functions."""
        # Sin affinity floor
        new_act_aff = act_affinities.at[SIN_IDX].set(
            jnp.maximum(act_affinities[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation affinity floors
        new_agg_aff = agg_affinities
        for idx in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[idx].set(
                jnp.maximum(new_agg_aff[idx], self.extreme_agg_affinity_floor)
            )

        return new_act_aff, new_agg_aff

    def _update_affinities(
        self,
        affinities: jnp.ndarray,
        contributions: jnp.ndarray,
        cross_boost: jnp.ndarray,
        key: jax.random.PRNGKey,
        lr: float,
        decay: float,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update affinities with learning, decay, hypermutation, and discovery boost."""
        k1, k2 = jax.random.split(key)
        newly_discovered = newly_discovered or []

        # Decay and learning
        new_affinities = decay * affinities
        new_affinities = new_affinities + lr * contributions
        new_affinities = new_affinities + self.cross_boost_factor * cross_boost

        # Discovery boost for newly discovered functions
        for idx in newly_discovered:
            new_affinities = new_affinities.at[idx].set(
                new_affinities[idx] + self.discovery_boost
            )

        # Memory cells resist decay
        new_affinities = jnp.where(
            memory_cells,
            jnp.maximum(new_affinities, affinities * 0.95),  # Only 5% decay for memory cells
            new_affinities
        )

        # Hypermutation
        mutation_probs = jax.random.uniform(k1, affinities.shape)
        mutation_amounts = jax.random.normal(k2, affinities.shape) * self.hypermutation_strength
        hypermutation_mask = mutation_probs < self.hypermutation_rate

        # Memory cells don't hypermutate (stable)
        hypermutation_mask = jnp.logical_and(hypermutation_mask, ~memory_cells)

        new_affinities = jnp.where(
            hypermutation_mask,
            new_affinities + mutation_amounts,
            new_affinities
        )

        return jnp.clip(new_affinities, 0.0, 1.0), hypermutation_mask

    def _update_memory_cells(
        self,
        affinities: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cell tracking based on sustained high affinity."""
        # Functions above threshold increment count
        above_threshold = affinities >= self.memory_cell_threshold
        new_counts = jnp.where(above_threshold, memory_counts + 1, 0)

        # Become memory cell if sustained long enough
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _update_expressions(
        self,
        expressions: jnp.ndarray,
        affinities: jnp.ndarray,
        threshold: float,
        prolif_rate: float,
        decay_rate: float,
        exp_min: float,
        exp_max: float,
        memory_cells: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update expression levels based on affinities."""
        new_expressions = expressions.copy()

        for i in range(len(expressions)):
            if affinities[i] >= threshold or memory_cells[i]:
                # Clonal expansion (memory cells always expand)
                new_expressions = new_expressions.at[i].set(
                    expressions[i] * (1 + prolif_rate)
                )
            else:
                # Decay
                new_expressions = new_expressions.at[i].set(
                    expressions[i] * (1 - decay_rate)
                )

        return jnp.clip(new_expressions, exp_min, exp_max)

    def _ensure_diversity(
        self,
        expressions: jnp.ndarray,
        min_diverse: int,
    ) -> jnp.ndarray:
        """Ensure minimum diversity by boosting low-expression functions."""
        expressible = jnp.sum(expressions >= self.diversity_threshold)

        if expressible < min_diverse:
            n_to_boost = min_diverse - int(expressible)
            sorted_indices = jnp.argsort(expressions)

            for i in range(min(n_to_boost, len(sorted_indices))):
                idx = int(sorted_indices[i])
                expressions = expressions.at[idx].set(
                    max(float(expressions[idx]), self.diversity_threshold)
                )

        return expressions

    def _select_palette(
        self,
        expressions: jnp.ndarray,
        palette_size: int,
        min_active: int,
        max_active: int,
        memory_cells: jnp.ndarray,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette with discovery slot guarantee.

        Returns:
            mask: Selected palette mask
            discovery_to_palette: Count of discoveries that entered palette
        """
        newly_discovered = newly_discovered or []
        n_funcs = len(expressions)
        target_size = min(max(palette_size, min_active), max_active, n_funcs)
        discovery_to_palette = 0

        # Memory cells always get priority
        memory_indices = [i for i in range(n_funcs) if memory_cells[i]]

        # Get top expressions
        top_indices = list(jnp.argsort(expressions)[-target_size:])

        # Combine: memory cells + top expressions
        selected = set(memory_indices) | set(int(i) for i in top_indices)

        # Limit to max_active
        if len(selected) > max_active:
            # Prioritize memory cells and high-expression
            sorted_selected = sorted(
                selected,
                key=lambda i: (memory_cells[i], float(expressions[i])),
                reverse=True
            )
            selected = set(sorted_selected[:max_active])

        # Discovery slot guarantee
        if self.enable_discovery_slot and newly_discovered:
            undiscovered_not_selected = [
                idx for idx in newly_discovered
                if idx not in selected
            ]
            if undiscovered_not_selected and len(selected) < max_active:
                # Pick the one with highest expression
                best_new = max(
                    undiscovered_not_selected,
                    key=lambda j: float(expressions[j])
                )
                selected.add(best_new)
                discovery_to_palette += 1

        # Count discoveries that made it
        for idx in newly_discovered:
            if idx in selected:
                discovery_to_palette += 1

        mask = jnp.zeros(n_funcs)
        for idx in selected:
            mask = mask.at[idx].set(1.0)

        return mask, discovery_to_palette

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity based on co-activation success."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        delta = self.cross_learning_rate * fitness_delta * co_active
        new_cross = cross_affinity + delta
        return jnp.clip(new_cross, 0.0, 1.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual clonal selection and discovery."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Compute fitness contributions for both domains
        act_contrib, agg_contrib = self._compute_fitness_contributions(
            state['act_mask'],
            state['agg_mask'],
            best_fitness,
            prev_best_fitness,
        )

        # Step 2: Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Compute cross-domain boosts
        act_cross_boost = jnp.dot(new_cross, state['agg_mask'] > 0.5)
        agg_cross_boost = jnp.dot(new_cross.T, state['act_mask'] > 0.5)

        # Normalize boosts
        act_cross_boost = act_cross_boost / max(jnp.sum(state['agg_mask'] > 0.5), 1)
        agg_cross_boost = agg_cross_boost / max(jnp.sum(state['act_mask'] > 0.5), 1)

        # Identify newly discovered candidates
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

        # Step 3: Update affinities with hypermutation and discovery boost
        new_act_aff, act_hypermut = self._update_affinities(
            state['act_affinities'], act_contrib, act_cross_boost, k1,
            self.act_affinity_lr, self.act_affinity_decay,
            state['act_memory_cells'], act_new_candidates
        )
        new_agg_aff, agg_hypermut = self._update_affinities(
            state['agg_affinities'], agg_contrib, agg_cross_boost, k2,
            self.agg_affinity_lr, self.agg_affinity_decay,
            state['agg_memory_cells'], agg_new_candidates
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

        # Step 4: Update expressions
        new_act_exp = self._update_expressions(
            state['act_expressions'], new_act_aff,
            self.act_affinity_threshold, self.act_proliferation_rate,
            self.act_expression_decay, self.act_expression_min, self.act_expression_max,
            new_act_mem_cells
        )
        new_agg_exp = self._update_expressions(
            state['agg_expressions'], new_agg_aff,
            self.agg_affinity_threshold, self.agg_proliferation_rate,
            self.agg_expression_decay, self.agg_expression_min, self.agg_expression_max,
            new_agg_mem_cells
        )

        # Step 5: Ensure diversity
        new_act_exp = self._ensure_diversity(new_act_exp, self.min_diversity_act)
        new_agg_exp = self._ensure_diversity(new_agg_exp, self.min_diversity_agg)

        # Step 6: Select palettes with discovery slot
        new_act_mask, act_disc_to_pal = self._select_palette(
            new_act_exp, self.act_palette_size,
            self.min_active_act, self.max_active_act,
            new_act_mem_cells, act_new_candidates
        )
        new_agg_mask, agg_disc_to_pal = self._select_palette(
            new_agg_exp, self.agg_palette_size,
            self.min_active_agg, self.max_active_agg,
            new_agg_mem_cells, agg_new_candidates
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

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Track expansions
        act_expanding = new_act_aff >= self.act_affinity_threshold
        agg_expanding = new_agg_aff >= self.agg_affinity_threshold
        new_act_expansions = state['act_expansions'] + act_expanding.astype(jnp.float32)
        new_agg_expansions = state['agg_expansions'] + agg_expanding.astype(jnp.float32)

        new_state = {
            # Activation domain
            'act_mask': new_act_mask,
            'act_affinities': new_act_aff,
            'act_expressions': new_act_exp,
            'act_expansions': new_act_expansions,
            'act_hypermutations': state['act_hypermutations'] + act_hypermut.astype(jnp.float32),
            # Aggregation domain
            'agg_mask': new_agg_mask,
            'agg_affinities': new_agg_aff,
            'agg_expressions': new_agg_exp,
            'agg_expansions': new_agg_expansions,
            'agg_hypermutations': state['agg_hypermutations'] + agg_hypermut.astype(jnp.float32),
            # Cross-domain
            'cross_affinity': new_cross,
            # General state
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': (state['fitness_history'] + [best_fitness])[-20:],
            # Discovery tracking
            'act_ever_discovered': act_ever_discovered,
            'agg_ever_discovered': agg_ever_discovered,
            'total_act_discoveries': state['total_act_discoveries'] + new_act_discoveries,
            'total_agg_discoveries': state['total_agg_discoveries'] + new_agg_discoveries,
            'discovery_to_palette': state['discovery_to_palette'] + act_disc_to_pal + agg_disc_to_pal,
            # Memory cells
            'act_memory_counts': new_act_mem_counts,
            'agg_memory_counts': new_agg_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_memory_cells': new_agg_mem_cells,
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
            # Activation affinity/expression
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'act_max_affinity': float(jnp.max(new_act_aff)),
            'act_mean_expression': float(jnp.mean(new_act_exp)),
            'act_expanding_count': int(jnp.sum(act_expanding)),
            # Aggregation affinity/expression
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'agg_max_affinity': float(jnp.max(new_agg_aff)),
            'agg_mean_expression': float(jnp.mean(new_agg_exp)),
            'agg_expanding_count': int(jnp.sum(agg_expanding)),
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # Sin and extreme agg status
            'has_sin': has_sin,
            'sin_affinity': float(new_act_aff[SIN_IDX]),
            'sin_expression': float(new_act_exp[SIN_IDX]),
            'has_extreme_agg': has_extreme_agg,
            'extreme_agg_affinities': [float(new_agg_aff[idx]) for idx in CORE_EXTREME_AGGS],
            # Discovery metrics
            'new_act_discoveries': new_act_discoveries,
            'new_agg_discoveries': new_agg_discoveries,
            'total_act_discoveries': new_state['total_act_discoveries'],
            'total_agg_discoveries': new_state['total_agg_discoveries'],
            'discovery_to_palette': new_state['discovery_to_palette'],
            # Memory cell metrics
            'act_memory_cell_count': int(jnp.sum(new_act_mem_cells)),
            'agg_memory_cell_count': int(jnp.sum(new_agg_mem_cells)),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with discovery stats."""
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
            # Activation
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'act_mean_expression': float(jnp.mean(state['act_expressions'])),
            # Aggregation
            'agg_mean_affinity': float(jnp.mean(state['agg_affinities'])),
            'agg_mean_expression': float(jnp.mean(state['agg_expressions'])),
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
            # Sin-specific
            'sin_affinity': float(state['act_affinities'][SIN_IDX]),
            'sin_expression': float(state['act_expressions'][SIN_IDX]),
            # Discovery stats
            'total_act_discoveries': state['total_act_discoveries'],
            'total_agg_discoveries': state['total_agg_discoveries'],
            'discovery_to_palette': state['discovery_to_palette'],
            'act_ever_discovered': list(state['act_ever_discovered']),
            'agg_ever_discovered': list(state['agg_ever_discovered']),
            # Memory cells
            'act_memory_cell_count': int(jnp.sum(state['act_memory_cells'])),
            'agg_memory_cell_count': int(jnp.sum(state['agg_memory_cells'])),
            'act_memory_cell_indices': [i for i in range(NUM_ACTIVATIONS) if state['act_memory_cells'][i]],
            'agg_memory_cell_indices': [i for i in range(NUM_AGGREGATIONS) if state['agg_memory_cells'][i]],
        }
