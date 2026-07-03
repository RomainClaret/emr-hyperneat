"""Strategy 33: Genetic Regulatory Network (Boolean Regulatory Circuits).

Implements gene regulatory network principles for palette evolution. Functions
regulate each other through activation and inhibition relationships, creating
emergent expression patterns from evolved Boolean circuits.

Biological Basis:
- Genes regulate other genes through transcription factors
- Activation: Gene A promotes expression of Gene B
- Inhibition: Gene A suppresses expression of Gene B
- Expression levels follow Hill kinetics (sigmoidal response)
- Regulatory networks create stable attractors and oscillations

Key Insight:
- Current strategies learn correlations (Hebbian) but not causation
- GRNs create structure where function activity CAUSES other function states
- Emergent regulatory circuits self-organize
- Functions form coherent groups through evolved logic, not just correlation

GRN Mechanism:
    # Update expression based on regulatory network
    for each function j:
        activators = sum(expression[i] * max(regulation[i,j], 0))
        inhibitors = sum(expression[i] * max(-regulation[i,j], 0))

        # Hill function response
        activation = activators^n / (K^n + activators^n)
        inhibition = K^n / (K^n + inhibitors^n)

        expression[j] = basal + (1 - basal) * activation * inhibition

    # Evolve regulatory links based on fitness
    if fitness_improved:
        strengthen(links between co-active successful functions)
    else:
        weaken(links)

Expected improvements:
- Emergent regulatory circuits (not just correlations)
- Stable expression patterns (attractors)
- Coherent function groups via causal links
- Boolean logic enables complex activation patterns
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


class GeneticRegulatoryNetworkStrategy(PaletteEvolutionStrategy):
    """Boolean regulatory circuits controlling function expression.

    Functions regulate each other through activation/inhibition links.
    Expression levels follow Hill kinetics with sigmoidal responses.
    Regulatory network evolves based on fitness feedback.
    """

    name = "genetic_regulatory_network"
    description = "Gene regulatory networks with activation/inhibition circuits"

    def __init__(
        self,
        # Expression dynamics
        basal_expression: float = 0.1,          # Background expression level
        hill_coefficient: float = 2.0,          # Sigmoidal steepness
        half_max_constant: float = 0.5,         # Half-activation constant (K)
        expression_decay: float = 0.9,          # Expression decay per generation
        # Regulatory network
        initial_regulation_strength: float = 0.3,  # Initial link strength
        regulation_learning_rate: float = 0.08,    # How fast links adapt
        regulation_decay: float = 0.98,            # Passive link decay
        regulation_max: float = 1.5,               # Max regulation strength
        network_sparsity: float = 0.3,             # Initial connection density
        # Activation/Inhibition balance
        activation_bias: float = 0.6,           # Probability of activation vs inhibition
        inhibition_strength_factor: float = 0.8, # Inhibition relative to activation
        # Expression threshold
        expression_threshold: float = 0.4,       # Min expression for palette
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Genetic Regulatory Network strategy.

        Args:
            basal_expression: Background expression when no regulation
            hill_coefficient: Steepness of sigmoidal response (n)
            half_max_constant: Half-activation constant (K)
            expression_decay: How fast expression decays
            initial_regulation_strength: Starting strength for links
            regulation_learning_rate: How fast regulatory links adapt
            regulation_decay: Passive decay of regulatory strengths
            regulation_max: Maximum regulatory strength
            network_sparsity: Fraction of possible links that exist initially
            activation_bias: Probability of positive vs negative regulation
            inhibition_strength_factor: Relative strength of inhibition
            expression_threshold: Minimum expression for inclusion in palette
            palette_size: Target palette size
        """
        # Expression
        self.basal_expression = basal_expression
        self.hill_coefficient = hill_coefficient
        self.half_max_constant = half_max_constant
        self.expression_decay = expression_decay

        # Regulation
        self.initial_regulation_strength = initial_regulation_strength
        self.regulation_learning_rate = regulation_learning_rate
        self.regulation_decay = regulation_decay
        self.regulation_max = regulation_max
        self.network_sparsity = network_sparsity

        # Activation/Inhibition
        self.activation_bias = activation_bias
        self.inhibition_strength_factor = inhibition_strength_factor

        # Selection
        self.expression_threshold = expression_threshold
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _initialize_regulation_matrix(
        self,
        key: jax.random.PRNGKey,
        initial: List[int],
    ) -> jnp.ndarray:
        """Initialize regulatory network with sparse random connections."""
        key1, key2, key3 = jax.random.split(key, 3)

        # Start with zeros
        regulation = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS))

        # Randomly add connections based on sparsity
        connection_mask = jax.random.uniform(key1, (NUM_ACTIVATIONS, NUM_ACTIVATIONS)) < self.network_sparsity

        # Determine activation vs inhibition
        is_activation = jax.random.uniform(key2, (NUM_ACTIVATIONS, NUM_ACTIVATIONS)) < self.activation_bias

        # Random strengths
        strengths = jax.random.uniform(key3, (NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * self.initial_regulation_strength

        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_ACTIVATIONS):
                if i != j and connection_mask[i, j]:  # No self-regulation
                    if is_activation[i, j]:
                        regulation = regulation.at[i, j].set(float(strengths[i, j]))
                    else:
                        regulation = regulation.at[i, j].set(-float(strengths[i, j]) * self.inhibition_strength_factor)

        # Boost connections within initial palette (prior knowledge)
        for i in initial:
            for j in initial:
                if i != j and 0 <= i < NUM_ACTIVATIONS and 0 <= j < NUM_ACTIVATIONS:
                    current = regulation[i, j]
                    if current == 0:
                        regulation = regulation.at[i, j].set(self.initial_regulation_strength * 0.5)
                    elif current > 0:
                        regulation = regulation.at[i, j].set(current * 1.5)

        return regulation

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with regulatory network."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        key = jax.random.PRNGKey(seed + 333333)
        key, subkey = jax.random.split(key)

        # Initialize regulatory network
        regulation = self._initialize_regulation_matrix(subkey, initial)

        # Initialize expression levels
        expression = jnp.ones(NUM_ACTIVATIONS) * self.basal_expression
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                expression = expression.at[i].set(0.6)

        return {
            'mask': mask,
            'rng_key': key,
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # GRN state
            'regulation': regulation,
            'expression': expression,
            # Tracking
            'network_changes': 0,
            'total_activation_strength': float(jnp.sum(jnp.maximum(regulation, 0))),
            'total_inhibition_strength': float(jnp.sum(jnp.maximum(-regulation, 0))),
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _hill_activation(self, activator_sum: float) -> float:
        """Compute Hill activation function."""
        n = self.hill_coefficient
        K = self.half_max_constant
        return (activator_sum ** n) / (K ** n + activator_sum ** n + 1e-8)

    def _hill_inhibition(self, inhibitor_sum: float) -> float:
        """Compute Hill inhibition function."""
        n = self.hill_coefficient
        K = self.half_max_constant
        return (K ** n) / (K ** n + inhibitor_sum ** n + 1e-8)

    def _update_expression(
        self,
        expression: jnp.ndarray,
        regulation: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update expression levels based on regulatory network."""
        new_expression = jnp.zeros(NUM_ACTIVATIONS)

        for j in range(NUM_ACTIVATIONS):
            # Sum activation and inhibition from all regulators
            activator_sum = 0.0
            inhibitor_sum = 0.0

            for i in range(NUM_ACTIVATIONS):
                if regulation[i, j] > 0:
                    activator_sum += float(expression[i]) * float(regulation[i, j])
                elif regulation[i, j] < 0:
                    inhibitor_sum += float(expression[i]) * abs(float(regulation[i, j]))

            # Compute activation and inhibition factors
            activation = self._hill_activation(activator_sum)
            inhibition = self._hill_inhibition(inhibitor_sum)

            # Combine: expression = basal + (1-basal) * activation * inhibition
            new_expr = self.basal_expression + (1 - self.basal_expression) * activation * inhibition

            # Apply decay toward basal
            current = float(expression[j])
            decayed = current * self.expression_decay + new_expr * (1 - self.expression_decay)

            new_expression = new_expression.at[j].set(decayed)

        return jnp.clip(new_expression, 0.0, 1.0)

    def _update_regulation(
        self,
        regulation: jnp.ndarray,
        expression: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update regulatory links based on co-activity and fitness."""
        new_regulation = regulation * self.regulation_decay

        if improvement > 0:
            # Strengthen links between co-active successful functions
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_ACTIVATIONS):
                    if i != j:
                        # Co-activity: both expressed and in palette
                        co_active = (
                            expression[i] > self.expression_threshold and
                            expression[j] > self.expression_threshold and
                            mask[i] > 0.5 and mask[j] > 0.5
                        )

                        if co_active:
                            current = regulation[i, j]
                            # Strengthen existing links, or create weak activation
                            if current != 0:
                                delta = self.regulation_learning_rate * improvement * jnp.sign(current)
                            else:
                                delta = self.regulation_learning_rate * improvement * 0.5
                            new_regulation = new_regulation.at[i, j].set(current + delta)
        else:
            # Slight weakening on failure (but preserve structure)
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_ACTIVATIONS):
                    if mask[i] > 0.5 and mask[j] > 0.5:
                        current = regulation[i, j]
                        if current != 0:
                            delta = self.regulation_learning_rate * 0.3 * jnp.sign(current)
                            new_regulation = new_regulation.at[i, j].set(current - delta)

        # Clip to valid range
        new_regulation = jnp.clip(new_regulation, -self.regulation_max, self.regulation_max)

        return new_regulation

    def _select_palette_from_expression(
        self,
        expression: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select palette based on expression levels."""
        above_threshold = expression >= self.expression_threshold
        n_above = int(jnp.sum(above_threshold))

        if n_above >= self.min_active and n_above <= self.palette_size:
            mask = above_threshold.astype(jnp.float32)
        elif n_above < self.min_active:
            # Too few: take top by expression
            top_k = jnp.argsort(expression)[-self.min_active:]
            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            # Too many: take top by expression
            top_k = jnp.argsort(expression)[-self.palette_size:]
            mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in top_k:
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
        """Update with GRN dynamics."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update expression based on current regulation
        new_expression = self._update_expression(
            state['expression'],
            state['regulation'],
        )

        # Step 2: Update regulatory links based on fitness
        new_regulation = self._update_regulation(
            state['regulation'],
            new_expression,
            state['mask'],
            improvement,
        )

        # Step 3: Select palette from expression
        new_mask = self._select_palette_from_expression(new_expression)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Count network changes
        network_changes = int(jnp.sum(jnp.abs(new_regulation - state['regulation']) > 0.01))

        # Compute activation/inhibition strengths
        total_activation = float(jnp.sum(jnp.maximum(new_regulation, 0)))
        total_inhibition = float(jnp.sum(jnp.maximum(-new_regulation, 0)))

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
            # GRN state
            'regulation': new_regulation,
            'expression': new_expression,
            # Tracking
            'network_changes': network_changes,
            'total_activation_strength': total_activation,
            'total_inhibition_strength': total_inhibition,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Top expressed functions
        top_expr_idx = jnp.argsort(new_expression)[-5:][::-1]
        top_expression = [(int(i), float(new_expression[i])) for i in top_expr_idx]

        # Most regulated functions (strongest incoming regulation)
        incoming_regulation = jnp.sum(jnp.abs(new_regulation), axis=0)
        top_regulated_idx = jnp.argsort(incoming_regulation)[-3:][::-1]
        most_regulated = [(int(i), float(incoming_regulation[i])) for i in top_regulated_idx]

        # Most regulating functions (strongest outgoing regulation)
        outgoing_regulation = jnp.sum(jnp.abs(new_regulation), axis=1)
        top_regulator_idx = jnp.argsort(outgoing_regulation)[-3:][::-1]
        top_regulators = [(int(i), float(outgoing_regulation[i])) for i in top_regulator_idx]

        # Network connectivity
        n_active_links = int(jnp.sum(jnp.abs(new_regulation) > 0.01))

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Expression
            'mean_expression': float(jnp.mean(new_expression)),
            'max_expression': float(jnp.max(new_expression)),
            'top_expression': top_expression,
            # Network structure
            'n_active_links': n_active_links,
            'total_activation': total_activation,
            'total_inhibition': total_inhibition,
            'activation_inhibition_ratio': total_activation / (total_inhibition + 0.01),
            'network_changes': network_changes,
            # Top regulators
            'most_regulated': most_regulated,
            'top_regulators': top_regulators,
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_expression': float(new_expression[4]),
            'sin_incoming_regulation': float(incoming_regulation[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with GRN status."""
        palette = self.get_active_palette(state)
        expression = state['expression']
        regulation = state['regulation']

        # Top functions by expression
        top_expr = jnp.argsort(expression)[-5:][::-1]
        top_expression = [(int(i), float(expression[i])) for i in top_expr]

        # Network stats
        n_active_links = int(jnp.sum(jnp.abs(regulation) > 0.01))
        n_activating = int(jnp.sum(regulation > 0.01))
        n_inhibiting = int(jnp.sum(regulation < -0.01))

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Expression
            'top_expression': top_expression,
            'mean_expression': float(jnp.mean(expression)),
            # Network
            'n_active_links': n_active_links,
            'n_activating_links': n_activating,
            'n_inhibiting_links': n_inhibiting,
            'total_activation': state['total_activation_strength'],
            'total_inhibition': state['total_inhibition_strength'],
            # Sin-specific
            'sin_expression': float(expression[4]),
        }
