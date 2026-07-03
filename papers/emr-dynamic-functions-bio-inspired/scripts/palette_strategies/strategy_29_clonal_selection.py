"""Strategy 29: Clonal Selection (Immune-Inspired Diversity).

Implements immune system clonal selection - antibodies matching antigens
proliferate through clonal expansion while maintaining diversity through
hypermutation. Applied to palette evolution, functions have affinities that
determine their expression levels.

Biological Basis:
- Antibodies (B-cells) recognize specific antigens
- Matching antibodies undergo clonal expansion (proliferation)
- Non-matching antibodies decay but aren't eliminated
- Hypermutation creates variation to match new antigens
- Immune memory preserves successful antibodies

Key Insight:
- Previous strategies don't maintain diversity under selection pressure
- Clonal selection allows dormant functions with high latent affinity
- When problem context shifts, dormant functions can rapidly expand
- Hypermutation enables discovery of new useful functions

Clonal Mechanism:
    # Affinity learning (how well function matches current problem)
    affinity[i] += learning_rate * (fitness_when_used - baseline)

    # Clonal expansion (high affinity -> high expression)
    if affinity[i] > threshold:
        expression[i] *= (1 + proliferation_rate)
    else:
        expression[i] *= (1 - decay_rate)

    # Hypermutation (maintain diversity)
    if random() < hypermutation_rate:
        affinity[i] += random_perturbation()

    # Palette = top K functions by expression level

Expected improvements:
- Maintains latent diversity (dormant but available functions)
- Rapid adaptation when problem context changes
- Protects rare but high-affinity functions
- Natural exploration through hypermutation
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


class ClonalSelectionStrategy(PaletteEvolutionStrategy):
    """Immune-inspired affinity-based selection with diversity maintenance.

    Functions have affinities that determine how well they match the current
    problem. High-affinity functions proliferate (expression increases), while
    low-affinity functions decay but aren't eliminated. Hypermutation maintains
    exploration potential.
    """

    name = "clonal_selection"
    description = "Immune-inspired clonal expansion with affinity learning and hypermutation"

    def __init__(
        self,
        # Affinity parameters
        affinity_learning_rate: float = 0.12,
        affinity_decay: float = 0.98,
        affinity_threshold: float = 0.4,  # Threshold for clonal expansion
        # Expression dynamics
        proliferation_rate: float = 0.25,  # Expansion when above threshold
        expression_decay: float = 0.08,    # Decay when below threshold
        expression_min: float = 0.05,      # Minimum expression (never fully dormant)
        expression_max: float = 1.0,       # Maximum expression
        # Hypermutation
        hypermutation_rate: float = 0.08,  # Chance of random affinity change
        hypermutation_strength: float = 0.2,  # Size of random changes
        # Diversity protection
        min_diversity: int = 4,  # Always keep N functions expressible
        diversity_threshold: float = 0.2,  # Expression level for "expressible"
        # Palette selection
        palette_size: int = 6,  # Number of active functions
        selection_method: str = "top_k",  # or "probabilistic"
        selection_temperature: float = 0.5,  # For probabilistic selection
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Clonal Selection strategy.

        Args:
            affinity_learning_rate: How fast affinities update
            affinity_decay: Passive affinity decay
            affinity_threshold: Threshold for clonal expansion
            proliferation_rate: Expression growth rate when high affinity
            expression_decay: Expression decay rate when low affinity
            expression_min: Minimum expression level
            expression_max: Maximum expression level
            hypermutation_rate: Probability of random affinity perturbation
            hypermutation_strength: Magnitude of hypermutation
            min_diversity: Minimum expressible functions
            diversity_threshold: Expression level for counting as expressible
            palette_size: Target number of active functions
            selection_method: How to select palette from expressions
            selection_temperature: Temperature for probabilistic selection
        """
        # Affinity
        self.affinity_learning_rate = affinity_learning_rate
        self.affinity_decay = affinity_decay
        self.affinity_threshold = affinity_threshold

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
        self.selection_method = selection_method
        self.selection_temperature = selection_temperature

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with affinities and expressions."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize affinities (prior beliefs about function usefulness)
        affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3  # Start neutral-positive

        # Boost initial palette affinities
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                affinities = affinities.at[i].set(0.5)

        # Initialize expression levels (current activation probability)
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
            # Tracking
            'clonal_expansions': jnp.zeros(NUM_ACTIVATIONS),  # Count of expansions
            'hypermutations': jnp.zeros(NUM_ACTIVATIONS),     # Count of mutations
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_fitness_contribution(
        self,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
    ) -> jnp.ndarray:
        """Compute per-function fitness contribution.

        Active functions share credit for fitness improvement.
        """
        improvement = fitness - prev_fitness
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)

        # Each active function gets share of improvement
        contributions = active * improvement / n_active

        return contributions

    def _update_affinities(
        self,
        affinities: jnp.ndarray,
        contributions: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update affinities based on fitness contributions and hypermutation.

        Returns:
            (new_affinities, hypermutation_mask)
        """
        key1, key2 = jax.random.split(key)

        # Decay existing affinities
        new_affinities = self.affinity_decay * affinities

        # Add contributions (positive or negative)
        new_affinities = new_affinities + self.affinity_learning_rate * contributions

        # Hypermutation: random perturbations to maintain diversity
        mutation_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        mutation_amounts = jax.random.normal(key2, (NUM_ACTIVATIONS,)) * self.hypermutation_strength
        hypermutation_mask = mutation_probs < self.hypermutation_rate

        new_affinities = jnp.where(
            hypermutation_mask,
            new_affinities + mutation_amounts,
            new_affinities
        )

        return jnp.clip(new_affinities, 0.0, 1.0), hypermutation_mask

    def _update_expressions(
        self,
        expressions: jnp.ndarray,
        affinities: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update expression levels based on affinities.

        High affinity -> proliferation (expression increases)
        Low affinity -> decay (expression decreases)
        """
        new_expressions = expressions.copy()

        for i in range(NUM_ACTIVATIONS):
            if affinities[i] >= self.affinity_threshold:
                # Clonal expansion
                new_expressions = new_expressions.at[i].set(
                    expressions[i] * (1 + self.proliferation_rate)
                )
            else:
                # Decay
                new_expressions = new_expressions.at[i].set(
                    expressions[i] * (1 - self.expression_decay)
                )

        # Clip to valid range
        new_expressions = jnp.clip(new_expressions, self.expression_min, self.expression_max)

        return new_expressions

    def _ensure_diversity(
        self,
        expressions: jnp.ndarray,
        affinities: jnp.ndarray,
    ) -> jnp.ndarray:
        """Ensure minimum diversity by boosting low-expression functions if needed."""
        expressible = jnp.sum(expressions >= self.diversity_threshold)

        if expressible < self.min_diversity:
            # Boost lowest expression functions with any affinity
            n_to_boost = self.min_diversity - int(expressible)
            sorted_indices = jnp.argsort(expressions)

            for i in range(n_to_boost):
                idx = int(sorted_indices[i])
                # Boost to diversity threshold
                expressions = expressions.at[idx].set(
                    max(float(expressions[idx]), self.diversity_threshold)
                )

        return expressions

    def _select_palette(
        self,
        expressions: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Select palette based on expression levels."""
        if self.selection_method == "top_k":
            # Select top K by expression
            top_k = min(self.palette_size, NUM_ACTIVATIONS)
            top_indices = jnp.argsort(expressions)[-top_k:]

            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_indices:
                mask = mask.at[int(idx)].set(1.0)

        else:  # probabilistic
            # Softmax selection based on expression
            probs = jax.nn.softmax(expressions / self.selection_temperature)

            # Sample without replacement (approximately)
            selected = set()
            key_remaining = key

            for _ in range(self.palette_size):
                key_remaining, subkey = jax.random.split(key_remaining)
                sample = jax.random.choice(subkey, NUM_ACTIVATIONS, p=probs)
                selected.add(int(sample))

            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in selected:
                mask = mask.at[idx].set(1.0)

        return mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with clonal selection dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Compute per-function fitness contributions
        contributions = self._compute_fitness_contribution(
            state['mask'],
            best_fitness,
            prev_best_fitness,
        )

        # Step 2: Update affinities with hypermutation
        new_affinities, hypermutation_mask = self._update_affinities(
            state['affinities'],
            contributions,
            k1,
        )

        # Step 3: Update expression levels (clonal dynamics)
        new_expressions = self._update_expressions(
            state['expressions'],
            new_affinities,
        )

        # Step 4: Ensure diversity
        new_expressions = self._ensure_diversity(
            new_expressions,
            new_affinities,
        )

        # Step 5: Select palette based on expressions
        new_mask = self._select_palette(new_expressions, k2)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track expansion and hypermutation events
        expanding = new_affinities >= self.affinity_threshold
        new_expansions = state['clonal_expansions'] + expanding.astype(jnp.float32)
        new_hypermutations = state['hypermutations'] + hypermutation_mask.astype(jnp.float32)

        # Track fitness
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Immune state
            'affinities': new_affinities,
            'expressions': new_expressions,
            # Tracking
            'clonal_expansions': new_expansions,
            'hypermutations': new_hypermutations,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Top affinities and expressions
        top_aff_idx = jnp.argsort(new_affinities)[-3:][::-1]
        top_affinities = [(int(i), float(new_affinities[i])) for i in top_aff_idx]

        top_exp_idx = jnp.argsort(new_expressions)[-3:][::-1]
        top_expressions = [(int(i), float(new_expressions[i])) for i in top_exp_idx]

        # Diversity measure
        expressible_count = int(jnp.sum(new_expressions >= self.diversity_threshold))
        expanding_count = int(jnp.sum(expanding))

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Affinity
            'mean_affinity': float(jnp.mean(new_affinities)),
            'max_affinity': float(jnp.max(new_affinities)),
            'top_affinities': top_affinities,
            # Expression
            'mean_expression': float(jnp.mean(new_expressions)),
            'max_expression': float(jnp.max(new_expressions)),
            'top_expressions': top_expressions,
            # Dynamics
            'expanding_count': expanding_count,
            'hypermutations_this_gen': int(jnp.sum(hypermutation_mask)),
            'expressible_count': expressible_count,
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_affinity': float(new_affinities[4]),
            'sin_expression': float(new_expressions[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with immune status."""
        palette = self.get_active_palette(state)
        affinities = state['affinities']
        expressions = state['expressions']

        # Top functions by affinity
        top_aff = jnp.argsort(affinities)[-5:][::-1]
        top_affinities = [(int(i), float(affinities[i])) for i in top_aff]

        # Top functions by expression
        top_exp = jnp.argsort(expressions)[-5:][::-1]
        top_expressions = [(int(i), float(expressions[i])) for i in top_exp]

        # Diversity stats
        expressible = int(jnp.sum(expressions >= self.diversity_threshold))
        dormant = int(jnp.sum(expressions < self.diversity_threshold))

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Affinity
            'top_affinities': top_affinities,
            'mean_affinity': float(jnp.mean(affinities)),
            # Expression
            'top_expressions': top_expressions,
            'mean_expression': float(jnp.mean(expressions)),
            # Diversity
            'expressible_count': expressible,
            'dormant_count': dormant,
            # Sin-specific
            'sin_affinity': float(affinities[4]),
            'sin_expression': float(expressions[4]),
            # Cumulative
            'total_expansions': int(jnp.sum(state['clonal_expansions'])),
            'total_hypermutations': int(jnp.sum(state['hypermutations'])),
        }
