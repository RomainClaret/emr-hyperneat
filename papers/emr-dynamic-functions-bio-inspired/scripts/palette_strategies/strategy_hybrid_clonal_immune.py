"""Hybrid Strategy: Clonal Selection + Immune Memory.

Combines two immune-inspired strategies:
- Clonal Selection (29): Affinity-based proliferation/decay with hypermutation
- Immune Memory (44): Long-lived memory cells with cross-reactivity

Expected Benefits:
- Rapid reactivation of previously useful functions (Memory)
- Affinity-based selection pressure (Clonal)
- Cross-reactivity helps related functions (Memory)
- Hypermutation maintains exploration (Clonal)

Hybrid Mechanism:
    # Clonal selection: Update affinities based on fitness
    affinity[i] += learning_rate * fitness_contribution

    # Memory formation: High-affinity functions form memory cells
    if affinity[i] >= memory_threshold:
        memory_cells[i] = {
            'affinity': affinity[i],
            'lifespan': memory_lifespan,
        }

    # Memory reactivation: Dormant memories can rapidly expand
    if problem_shift_detected() and memory_cells[i].cross_reacts():
        expression[i] *= rapid_reactivation_boost

    # Palette = top K by expression, with memory bonus
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


class ClonalImmuneStrategy(PaletteEvolutionStrategy):
    """Hybrid: Clonal Selection + Immune Memory.

    Combines affinity-based proliferation (Clonal) with long-lived
    memory cells and cross-reactivity (Immune Memory).
    """

    name = "clonal_immune"
    description = "Clonal selection with immune memory and cross-reactivity"

    def __init__(
        self,
        # Clonal selection parameters
        affinity_learning_rate: float = 0.1,
        affinity_decay: float = 0.97,
        proliferation_rate: float = 0.2,
        expression_decay: float = 0.06,
        hypermutation_rate: float = 0.06,
        hypermutation_strength: float = 0.15,
        # Immune memory parameters
        memory_formation_threshold: float = 0.6,
        memory_lifespan: int = 50,
        cross_reactivity_radius: int = 2,
        cross_reactivity_strength: float = 0.3,
        rapid_reactivation_boost: float = 2.0,
        # Selection parameters
        palette_size: int = 6,
        expression_min: float = 0.05,
        expression_max: float = 1.0,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        # Clonal
        self.affinity_learning_rate = affinity_learning_rate
        self.affinity_decay = affinity_decay
        self.proliferation_rate = proliferation_rate
        self.expression_decay = expression_decay
        self.hypermutation_rate = hypermutation_rate
        self.hypermutation_strength = hypermutation_strength

        # Memory
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_lifespan = memory_lifespan
        self.cross_reactivity_radius = cross_reactivity_radius
        self.cross_reactivity_strength = cross_reactivity_strength
        self.rapid_reactivation_boost = rapid_reactivation_boost

        # Selection
        self.palette_size = palette_size
        self.expression_min = expression_min
        self.expression_max = expression_max
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with Clonal + Immune Memory tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize affinities and expression
        affinity = jnp.zeros(NUM_ACTIVATIONS)
        expression = jnp.ones(NUM_ACTIVATIONS) * self.expression_min

        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                affinity = affinity.at[i].set(0.5)
                expression = expression.at[i].set(0.6)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 294444),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Clonal state
            'affinity': affinity,
            'expression': expression,
            # Immune memory state
            'memory_cells': {},  # {func: {'affinity': float, 'formed_gen': int, 'lifespan': int}}
            'dormant_pool': {},  # {func: {'affinity': float, 'dormant_since': int}}
            # History tracking
            'recent_fitness': [],  # For detecting problem shifts
            'memory_formations': [],
            'reactivations': [],
            'hypermutations': [],
            'previous_mask': mask,
            'fitness_history': [],
            # Stats
            'total_memory_formations': 0,
            'total_reactivations': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette."""
        return mask_to_indices(state['mask'])

    def _update_affinities(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        best_fitness: float,
        improved: bool,
    ) -> jnp.ndarray:
        """Update affinities based on fitness contribution."""
        new_affinity = affinity * self.affinity_decay

        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:  # Active
                if improved:
                    boost = self.affinity_learning_rate * best_fitness
                    new_affinity = new_affinity.at[i].add(boost)
                else:
                    # Small boost for being active
                    new_affinity = new_affinity.at[i].add(0.01)

        return jnp.clip(new_affinity, 0, 1.5)

    def _update_expression(
        self,
        expression: jnp.ndarray,
        affinity: jnp.ndarray,
        memory_cells: Dict,
        dormant_pool: Dict,
    ) -> jnp.ndarray:
        """Update expression levels based on affinity and memory."""
        new_expression = expression.copy()

        for i in range(NUM_ACTIVATIONS):
            aff = float(affinity[i])

            # Clonal expansion or decay
            if aff >= 0.4:  # Above threshold
                growth = self.proliferation_rate * aff
                new_expression = new_expression.at[i].add(growth)
            else:
                decay = self.expression_decay
                new_expression = new_expression.at[i].multiply(1 - decay)

            # Memory bonus: Memory cells get extra expression
            if i in memory_cells:
                mem_bonus = 0.1 * memory_cells[i]['affinity']
                new_expression = new_expression.at[i].add(mem_bonus)

        return jnp.clip(new_expression, self.expression_min, self.expression_max)

    def _apply_hypermutation(
        self,
        affinity: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, List[int]]:
        """Apply hypermutation to maintain exploration."""
        key1, key2 = jax.random.split(key)
        new_affinity = affinity.copy()
        mutated = []

        # Mutation mask
        mutation_mask = jax.random.uniform(key1, (NUM_ACTIVATIONS,)) < self.hypermutation_rate

        if jnp.any(mutation_mask):
            # Generate random perturbations
            perturbations = jax.random.normal(key2, (NUM_ACTIVATIONS,)) * self.hypermutation_strength

            for i in range(NUM_ACTIVATIONS):
                if mutation_mask[i]:
                    new_val = float(new_affinity[i]) + float(perturbations[i])
                    new_affinity = new_affinity.at[i].set(max(0, min(new_val, 1.5)))
                    mutated.append(i)

        return new_affinity, mutated

    def _manage_memory(
        self,
        affinity: jnp.ndarray,
        expression: jnp.ndarray,
        memory_cells: Dict,
        dormant_pool: Dict,
        generation: int,
    ) -> Tuple[Dict, Dict, List[int], List[int]]:
        """Manage memory cell formation, maintenance, and dormancy."""
        new_memory = dict(memory_cells)
        new_dormant = dict(dormant_pool)
        formed = []
        reactivated = []

        for i in range(NUM_ACTIVATIONS):
            aff = float(affinity[i])
            expr = float(expression[i])

            # Memory formation
            if aff >= self.memory_formation_threshold and i not in new_memory:
                new_memory[i] = {
                    'affinity': aff,
                    'formed_gen': generation,
                    'lifespan': self.memory_lifespan,
                }
                formed.append(i)

            # Memory maintenance
            if i in new_memory:
                new_memory[i]['lifespan'] -= 1
                if new_memory[i]['lifespan'] <= 0:
                    # Move to dormant pool
                    new_dormant[i] = {
                        'affinity': new_memory[i]['affinity'],
                        'dormant_since': generation,
                    }
                    del new_memory[i]

            # Reactivation from dormant
            if i in new_dormant:
                if aff >= 0.3:  # Some renewed activity
                    # Reactivate!
                    new_memory[i] = {
                        'affinity': new_dormant[i]['affinity'],
                        'formed_gen': generation,
                        'lifespan': self.memory_lifespan // 2,
                    }
                    reactivated.append(i)
                    del new_dormant[i]

        return new_memory, new_dormant, formed, reactivated

    def _apply_cross_reactivity(
        self,
        expression: jnp.ndarray,
        memory_cells: Dict,
    ) -> jnp.ndarray:
        """Apply cross-reactivity: memory cells boost nearby functions."""
        new_expression = expression.copy()

        for mem_func, mem_info in memory_cells.items():
            mem_aff = mem_info['affinity']

            # Boost nearby functions
            for i in range(NUM_ACTIVATIONS):
                distance = abs(i - mem_func)
                if 0 < distance <= self.cross_reactivity_radius:
                    boost = self.cross_reactivity_strength * mem_aff / distance
                    new_expression = new_expression.at[i].add(boost)

        return jnp.clip(new_expression, self.expression_min, self.expression_max)

    def _select_palette(
        self,
        expression: jnp.ndarray,
        memory_cells: Dict,
    ) -> jnp.ndarray:
        """Select palette based on expression levels with memory bonus."""
        # Compute selection scores
        scores = expression.copy()

        # Memory bonus
        for func in memory_cells:
            scores = scores.at[func].add(0.2)

        # Select top K
        top_indices = jnp.argsort(scores)[-self.palette_size:]

        mask = jnp.zeros(NUM_ACTIVATIONS)
        for i in top_indices:
            mask = mask.at[i].set(1.0)

        # Ensure minimum active
        if int(jnp.sum(mask)) < self.min_active:
            for i in jnp.argsort(scores)[-self.min_active:]:
                mask = mask.at[i].set(1.0)

        return mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with Clonal + Immune hybrid dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update affinities (Clonal)
        new_affinity = self._update_affinities(
            state['affinity'],
            state['mask'],
            best_fitness,
            improved,
        )

        # Step 2: Apply hypermutation (Clonal)
        new_affinity, hypermutated = self._apply_hypermutation(new_affinity, k1)

        # Step 3: Update expression (Clonal + Memory)
        new_expression = self._update_expression(
            state['expression'],
            new_affinity,
            state['memory_cells'],
            state['dormant_pool'],
        )

        # Step 4: Manage memory (Immune Memory)
        new_memory, new_dormant, formed, reactivated = self._manage_memory(
            new_affinity,
            new_expression,
            state['memory_cells'],
            state['dormant_pool'],
            generation,
        )

        # Step 5: Apply cross-reactivity (Immune Memory)
        new_expression = self._apply_cross_reactivity(new_expression, new_memory)

        # Step 6: Apply reactivation boost
        for func in reactivated:
            new_expression = new_expression.at[func].multiply(self.rapid_reactivation_boost)
        new_expression = jnp.clip(new_expression, self.expression_min, self.expression_max)

        # Step 7: Select palette
        new_mask = self._select_palette(new_expression, new_memory)
        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        memory_formations = list(state['memory_formations'])
        if formed:
            memory_formations.extend([(generation, f) for f in formed])
            if len(memory_formations) > 50:
                memory_formations = memory_formations[-50:]

        reactivation_list = list(state['reactivations'])
        if reactivated:
            reactivation_list.extend([(generation, f) for f in reactivated])
            if len(reactivation_list) > 50:
                reactivation_list = reactivation_list[-50:]

        hypermutation_list = list(state['hypermutations'])
        if hypermutated:
            hypermutation_list.extend([(generation, f) for f in hypermutated])
            if len(hypermutation_list) > 50:
                hypermutation_list = hypermutation_list[-50:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Clonal state
            'affinity': new_affinity,
            'expression': new_expression,
            # Memory state
            'memory_cells': new_memory,
            'dormant_pool': new_dormant,
            # History
            'recent_fitness': fitness_history[-5:],
            'memory_formations': memory_formations,
            'reactivations': reactivation_list,
            'hypermutations': hypermutation_list,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Stats
            'total_memory_formations': state['total_memory_formations'] + len(formed),
            'total_reactivations': state['total_reactivations'] + len(reactivated),
        }

        active_palette = mask_to_indices(new_mask)

        # Top affinities
        top_aff_idx = jnp.argsort(new_affinity)[-5:][::-1]
        top_affinities = [(int(i), float(new_affinity[i])) for i in top_aff_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Clonal
            'hypermutated_this_gen': hypermutated,
            'n_hypermutations': len(hypermutated),
            'top_affinities': top_affinities,
            # Memory
            'n_memory_cells': len(new_memory),
            'n_dormant': len(new_dormant),
            'formed_this_gen': formed,
            'reactivated_this_gen': reactivated,
            # Stats
            'total_memory_formations': new_state['total_memory_formations'],
            'total_reactivations': new_state['total_reactivations'],
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_affinity': float(new_affinity[4]),
            'sin_in_memory': 4 in new_memory,
            'sin_expression': float(new_expression[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        palette = self.get_active_palette(state)
        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'n_memory_cells': len(state['memory_cells']),
            'n_dormant': len(state['dormant_pool']),
            'total_memory_formations': state['total_memory_formations'],
            'total_reactivations': state['total_reactivations'],
        }
