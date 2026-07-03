"""Strategy 29D: Clonal Selection Dual (Immune-Inspired Diversity for Both Domains).

Extends ClonalSelectionStrategy to jointly evolve BOTH activation AND aggregation
function palettes using immune system clonal selection mechanisms.

Cross-Domain Learning:
- Separate affinities and expressions for both domains
- Cross-domain affinity: successful act-agg combinations boost each other
- Shared hypermutation: exploration in one domain can trigger exploration in other
- Coordinated clonal expansion based on joint success

Key Dual Mechanisms:
1. Dual affinities - how well each function matches current problem
2. Dual expressions - current activation probability for each function
3. Cross-affinity learning - which act-agg combinations succeed together
4. Coordinated hypermutation - maintain exploration potential in both domains
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

# Aggregation domain constants
NUM_AGGREGATIONS = 6
DEFAULT_AGGREGATION_INDICES = [0, 1]  # sum, mean


class ClonalSelectionDualStrategy(PaletteEvolutionStrategy):
    """Immune-inspired dual palette evolution with affinity learning.

    Both activation and aggregation functions have affinities that determine
    how well they match the current problem. High-affinity functions proliferate
    while low-affinity ones decay but aren't eliminated. Cross-domain learning
    allows successful combinations to mutually reinforce.
    """

    name = "clonal_selection_dual"
    description = "Immune-inspired clonal expansion with dual domain affinity learning"

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
        cross_hypermutation_prob: float = 0.3,  # Prob that mutation in one domain triggers other
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
    ):
        """Initialize Clonal Selection Dual strategy."""
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
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual affinities and expressions."""
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
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_mask = agg_mask.at[i].set(1.0)

        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        agg_expressions = jnp.ones(NUM_AGGREGATIONS) * self.agg_expression_min
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)
                agg_expressions = agg_expressions.at[i].set(0.6)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

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
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return [i for i in range(NUM_AGGREGATIONS) if state['agg_mask'][i] > 0.5]

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

    def _update_affinities(
        self,
        affinities: jnp.ndarray,
        contributions: jnp.ndarray,
        cross_boost: jnp.ndarray,
        key: jax.random.PRNGKey,
        lr: float,
        decay: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update affinities with learning, decay, and hypermutation."""
        k1, k2 = jax.random.split(key)

        # Decay and learning
        new_affinities = decay * affinities
        new_affinities = new_affinities + lr * contributions
        new_affinities = new_affinities + self.cross_boost_factor * cross_boost

        # Hypermutation
        mutation_probs = jax.random.uniform(k1, affinities.shape)
        mutation_amounts = jax.random.normal(k2, affinities.shape) * self.hypermutation_strength
        hypermutation_mask = mutation_probs < self.hypermutation_rate

        new_affinities = jnp.where(hypermutation_mask, new_affinities + mutation_amounts, new_affinities)

        return jnp.clip(new_affinities, 0.0, 1.0), hypermutation_mask

    def _update_expressions(
        self,
        expressions: jnp.ndarray,
        affinities: jnp.ndarray,
        threshold: float,
        prolif_rate: float,
        decay_rate: float,
        exp_min: float,
        exp_max: float,
    ) -> jnp.ndarray:
        """Update expression levels based on affinities."""
        new_expressions = expressions.copy()

        for i in range(len(expressions)):
            if affinities[i] >= threshold:
                # Clonal expansion
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
    ) -> jnp.ndarray:
        """Select palette based on expression levels."""
        n_funcs = len(expressions)
        target_size = min(max(palette_size, min_active), max_active, n_funcs)

        top_indices = jnp.argsort(expressions)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        return mask

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
        """Update with dual clonal selection dynamics."""
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

        # Step 3: Update affinities with hypermutation
        new_act_aff, act_hypermut = self._update_affinities(
            state['act_affinities'], act_contrib, act_cross_boost, k1,
            self.act_affinity_lr, self.act_affinity_decay
        )
        new_agg_aff, agg_hypermut = self._update_affinities(
            state['agg_affinities'], agg_contrib, agg_cross_boost, k2,
            self.agg_affinity_lr, self.agg_affinity_decay
        )

        # Step 4: Update expressions
        new_act_exp = self._update_expressions(
            state['act_expressions'], new_act_aff,
            self.act_affinity_threshold, self.act_proliferation_rate,
            self.act_expression_decay, self.act_expression_min, self.act_expression_max
        )
        new_agg_exp = self._update_expressions(
            state['agg_expressions'], new_agg_aff,
            self.agg_affinity_threshold, self.agg_proliferation_rate,
            self.agg_expression_decay, self.agg_expression_min, self.agg_expression_max
        )

        # Step 5: Ensure diversity
        new_act_exp = self._ensure_diversity(new_act_exp, self.min_diversity_act)
        new_agg_exp = self._ensure_diversity(new_agg_exp, self.min_diversity_agg)

        # Step 6: Select palettes
        new_act_mask = self._select_palette(
            new_act_exp, self.act_palette_size,
            self.min_active_act, self.max_active_act
        )
        new_agg_mask = self._select_palette(
            new_agg_exp, self.agg_palette_size,
            self.min_active_agg, self.max_active_agg
        )

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
        }

        # Compute metrics
        act_palette = mask_to_indices(new_act_mask)
        agg_palette = self.get_active_agg_palette(new_state)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
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
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_affinity': float(new_act_aff[4]),
            'sin_expression': float(new_act_exp[4]),
            # Agg4 status
            'has_agg4': len(agg_palette) >= 4,
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
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
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
            'sin_affinity': float(state['act_affinities'][4]),
            'sin_expression': float(state['act_expressions'][4]),
        }
