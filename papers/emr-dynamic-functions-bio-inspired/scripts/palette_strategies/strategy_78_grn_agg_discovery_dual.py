"""Strategy 78: GRN Aggregation Discovery Dual (Sin-Extreme Regulatory Coupling).

Extends GeneticRegulatoryNetworkDualStrategy with explicit sin-extreme aggregation
coupling. When sin (activation 4) expression is high, max/min expression is boosted
through regulatory links.

Bio inspiration: Gene regulatory networks where tissue-specific expression patterns
are coupled. In this case, sin activation "regulates" extreme aggregation expression.

Key innovation:
- Hardcoded sin->max and sin->min regulatory links
- Boosted learning rate for sin-extreme pairs
- Aggregation-specific Hill kinetics (gentler for 6 functions)

Expected: Better sin-extreme synergy discovery and retention.
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class GRNAggDiscoveryDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with sin-extreme gene regulatory coupling.

    Extends GRN approach with explicit sin-to-extreme aggregation regulatory
    links, enabling discovery and retention of critical sin-max/min synergies.
    """

    name = "grn_agg_discovery_dual"
    description = "Dual: GRN with sin-extreme regulatory coupling for aggregation discovery"

    def __init__(
        self,
        # Expression dynamics
        basal_expression: float = 0.1,
        hill_coefficient: float = 2.0,
        agg_hill_coefficient: float = 1.5,  # Gentler for aggregations
        half_max_constant: float = 0.5,
        expression_decay: float = 0.9,
        # Regulatory network
        initial_regulation_strength: float = 0.3,
        regulation_learning_rate: float = 0.08,
        regulation_decay: float = 0.98,
        regulation_max: float = 1.5,
        network_sparsity: float = 0.3,
        # Activation/Inhibition
        activation_bias: float = 0.6,
        inhibition_strength_factor: float = 0.8,
        # Cross-domain - KEY INNOVATION
        cross_regulation_sparsity: float = 0.15,
        cross_regulation_strength: float = 0.2,
        sin_to_max_coupling: float = 0.4,  # Hardcoded sin->max
        sin_to_min_coupling: float = 0.4,  # Hardcoded sin->min
        sin_extreme_lr_multiplier: float = 1.5,  # Boosted learning for sin-extreme
        # Expression threshold
        expression_threshold: float = 0.4,
        agg_expression_threshold: float = 0.35,  # Lower for aggregations
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize GRN Aggregation Discovery Dual strategy."""
        # Expression
        self.basal_expression = basal_expression
        self.hill_coefficient = hill_coefficient
        self.agg_hill_coefficient = agg_hill_coefficient
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

        # Cross-domain - sin-extreme coupling
        self.cross_regulation_sparsity = cross_regulation_sparsity
        self.cross_regulation_strength = cross_regulation_strength
        self.sin_to_max_coupling = sin_to_max_coupling
        self.sin_to_min_coupling = sin_to_min_coupling
        self.sin_extreme_lr_multiplier = sin_extreme_lr_multiplier

        # Selection
        self.expression_threshold = expression_threshold
        self.agg_expression_threshold = agg_expression_threshold
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _initialize_regulation_matrix(
        self,
        key: jax.random.PRNGKey,
        initial: List[int],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Initialize regulatory network with sparse random connections."""
        key1, key2, key3 = jax.random.split(key, 3)

        regulation = jnp.zeros((n_funcs, n_funcs))
        connection_mask = jax.random.uniform(key1, (n_funcs, n_funcs)) < self.network_sparsity
        is_activation = jax.random.uniform(key2, (n_funcs, n_funcs)) < self.activation_bias
        strengths = jax.random.uniform(key3, (n_funcs, n_funcs)) * self.initial_regulation_strength

        for i in range(n_funcs):
            for j in range(n_funcs):
                if i != j and connection_mask[i, j]:
                    if is_activation[i, j]:
                        regulation = regulation.at[i, j].set(float(strengths[i, j]))
                    else:
                        regulation = regulation.at[i, j].set(
                            -float(strengths[i, j]) * self.inhibition_strength_factor
                        )

        for i in initial:
            for j in initial:
                if i != j and 0 <= i < n_funcs and 0 <= j < n_funcs:
                    current = regulation[i, j]
                    if current == 0:
                        regulation = regulation.at[i, j].set(self.initial_regulation_strength * 0.5)
                    elif current > 0:
                        regulation = regulation.at[i, j].multiply(1.5)

        return regulation

    def _initialize_cross_regulation(
        self,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Initialize cross-domain regulation with sin-extreme coupling."""
        key1, key2 = jax.random.split(key)

        cross_reg = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        connection_mask = jax.random.uniform(key1, (NUM_ACTIVATIONS, NUM_AGGREGATIONS)) < self.cross_regulation_sparsity
        strengths = jax.random.uniform(key2, (NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * self.cross_regulation_strength

        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if connection_mask[i, j]:
                    cross_reg = cross_reg.at[i, j].set(float(strengths[i, j]))

        # HARDCODED sin-extreme coupling
        sin_idx = 4
        max_idx = 2
        min_idx = 3

        cross_reg = cross_reg.at[sin_idx, max_idx].set(self.sin_to_max_coupling)
        cross_reg = cross_reg.at[sin_idx, min_idx].set(self.sin_to_min_coupling)

        return cross_reg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual regulatory networks and sin-extreme coupling."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 780000)
        key, k_act, k_agg, k_cross = jax.random.split(key, 4)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize regulatory networks
        act_regulation = self._initialize_regulation_matrix(k_act, initial_act, NUM_ACTIVATIONS)
        agg_regulation = self._initialize_regulation_matrix(k_agg, initial_agg, NUM_AGGREGATIONS)
        cross_regulation = self._initialize_cross_regulation(k_cross)

        # Initialize expression levels
        act_expression = jnp.ones(NUM_ACTIVATIONS) * self.basal_expression
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_expression = act_expression.at[i].set(0.6)

        agg_expression = jnp.ones(NUM_AGGREGATIONS) * self.basal_expression
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_expression = agg_expression.at[i].set(0.6)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_regulation': act_regulation,
            'act_expression': act_expression,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_regulation': agg_regulation,
            'agg_expression': agg_expression,
            # Cross-domain
            'cross_regulation': cross_regulation,
            # Tracking
            'sin_extreme_coupling_events': 0,
            # General state
            'rng_key': key,
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _hill_activation(self, activator_sum: float, is_agg: bool = False) -> float:
        n = self.agg_hill_coefficient if is_agg else self.hill_coefficient
        K = self.half_max_constant
        return (activator_sum ** n) / (K ** n + activator_sum ** n + 1e-8)

    def _hill_inhibition(self, inhibitor_sum: float, is_agg: bool = False) -> float:
        n = self.agg_hill_coefficient if is_agg else self.hill_coefficient
        K = self.half_max_constant
        return (K ** n) / (K ** n + inhibitor_sum ** n + 1e-8)

    def _update_act_expression(
        self,
        expression: jnp.ndarray,
        regulation: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update activation expression (within-domain only)."""
        new_expression = jnp.zeros(NUM_ACTIVATIONS)

        for j in range(NUM_ACTIVATIONS):
            activator_sum = 0.0
            inhibitor_sum = 0.0

            for i in range(NUM_ACTIVATIONS):
                if regulation[i, j] > 0:
                    activator_sum += float(expression[i]) * float(regulation[i, j])
                elif regulation[i, j] < 0:
                    inhibitor_sum += float(expression[i]) * abs(float(regulation[i, j]))

            activation = self._hill_activation(activator_sum, is_agg=False)
            inhibition = self._hill_inhibition(inhibitor_sum, is_agg=False)
            new_expr = self.basal_expression + (1 - self.basal_expression) * activation * inhibition
            current = float(expression[j])
            decayed = current * self.expression_decay + new_expr * (1 - self.expression_decay)
            new_expression = new_expression.at[j].set(decayed)

        return jnp.clip(new_expression, 0.0, 1.0)

    def _update_agg_expression(
        self,
        agg_expression: jnp.ndarray,
        agg_regulation: jnp.ndarray,
        act_expression: jnp.ndarray,
        cross_regulation: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update aggregation expression with cross-domain sin influence."""
        new_expression = jnp.zeros(NUM_AGGREGATIONS)

        sin_expr = float(act_expression[4])  # sin expression

        for j in range(NUM_AGGREGATIONS):
            activator_sum = 0.0
            inhibitor_sum = 0.0

            # Within-domain regulation
            for i in range(NUM_AGGREGATIONS):
                if agg_regulation[i, j] > 0:
                    activator_sum += float(agg_expression[i]) * float(agg_regulation[i, j])
                elif agg_regulation[i, j] < 0:
                    inhibitor_sum += float(agg_expression[i]) * abs(float(agg_regulation[i, j]))

            # Cross-domain regulation from activations
            for i in range(NUM_ACTIVATIONS):
                if cross_regulation[i, j] > 0:
                    activator_sum += float(act_expression[i]) * float(cross_regulation[i, j])

            # BOOST: Extra sin->extreme coupling for max/min
            if j in CORE_EXTREME_AGGS:
                if j == 2:  # max
                    activator_sum += sin_expr * self.sin_to_max_coupling * 0.5
                elif j == 3:  # min
                    activator_sum += sin_expr * self.sin_to_min_coupling * 0.5

            activation = self._hill_activation(activator_sum, is_agg=True)
            inhibition = self._hill_inhibition(inhibitor_sum, is_agg=True)
            new_expr = self.basal_expression + (1 - self.basal_expression) * activation * inhibition
            current = float(agg_expression[j])
            decayed = current * self.expression_decay + new_expr * (1 - self.expression_decay)
            new_expression = new_expression.at[j].set(decayed)

        return jnp.clip(new_expression, 0.0, 1.0)

    def _update_regulation(
        self,
        regulation: jnp.ndarray,
        expression: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
        n_funcs: int,
        threshold: float,
    ) -> jnp.ndarray:
        """Update regulatory links based on fitness."""
        new_regulation = regulation * self.regulation_decay

        if improvement > 0:
            for i in range(n_funcs):
                for j in range(n_funcs):
                    if i != j:
                        co_active = (
                            expression[i] > threshold and
                            expression[j] > threshold and
                            mask[i] > 0.5 and mask[j] > 0.5
                        )
                        if co_active:
                            current = regulation[i, j]
                            if current != 0:
                                delta = self.regulation_learning_rate * improvement * jnp.sign(current)
                            else:
                                delta = self.regulation_learning_rate * improvement * 0.5
                            new_regulation = new_regulation.at[i, j].set(current + delta)

        return jnp.clip(new_regulation, -self.regulation_max, self.regulation_max)

    def _update_cross_regulation(
        self,
        cross_regulation: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, bool]:
        """Update cross-domain regulation with boosted sin-extreme learning."""
        new_cross = cross_regulation * self.regulation_decay
        sin_extreme_event = False

        if improvement > 0:
            # Standard co-activity learning
            active_act = (act_mask > 0.5).astype(jnp.float32)
            active_agg = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active_act, active_agg)

            # Base learning
            delta = self.regulation_learning_rate * improvement * co_active * 0.5
            new_cross = new_cross + delta

            # BOOSTED sin-extreme learning
            sin_active = act_mask[4] > 0.5
            if sin_active:
                for agg_idx in CORE_EXTREME_AGGS:
                    if agg_mask[agg_idx] > 0.5:
                        boost = self.regulation_learning_rate * improvement * self.sin_extreme_lr_multiplier
                        current = new_cross[4, agg_idx]
                        new_cross = new_cross.at[4, agg_idx].set(current + boost)
                        sin_extreme_event = True

        return jnp.clip(new_cross, 0.0, self.regulation_max), sin_extreme_event

    def _select_palette_from_expression(
        self,
        expression: jnp.ndarray,
        palette_size: int,
        min_active: int,
        n_funcs: int,
        threshold: float,
    ) -> jnp.ndarray:
        """Select palette based on expression levels."""
        above_threshold = expression >= threshold
        n_above = int(jnp.sum(above_threshold))

        if n_above >= min_active and n_above <= palette_size:
            mask = above_threshold.astype(jnp.float32)
        elif n_above < min_active:
            top_k = jnp.argsort(expression)[-min_active:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            top_k = jnp.argsort(expression)[-palette_size:]
            mask = jnp.zeros(n_funcs)
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
        """Update with GRN dynamics and sin-extreme coupling."""
        key = jax.random.split(state['rng_key'])[0]

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update activation expression
        new_act_expr = self._update_act_expression(
            state['act_expression'], state['act_regulation']
        )

        # Update aggregation expression with cross-domain influence
        new_agg_expr = self._update_agg_expression(
            state['agg_expression'], state['agg_regulation'],
            new_act_expr, state['cross_regulation']
        )

        # Update regulation
        new_act_reg = self._update_regulation(
            state['act_regulation'], new_act_expr,
            state['act_mask'], improvement, NUM_ACTIVATIONS,
            self.expression_threshold
        )
        new_agg_reg = self._update_regulation(
            state['agg_regulation'], new_agg_expr,
            state['agg_mask'], improvement, NUM_AGGREGATIONS,
            self.agg_expression_threshold
        )
        new_cross_reg, sin_extreme_event = self._update_cross_regulation(
            state['cross_regulation'], state['act_mask'], state['agg_mask'], improvement
        )

        coupling_events = state['sin_extreme_coupling_events']
        if sin_extreme_event:
            coupling_events += 1

        # Select palettes
        new_act_mask = self._select_palette_from_expression(
            new_act_expr, self.act_palette_size, self.min_active_act, NUM_ACTIVATIONS,
            self.expression_threshold
        )
        new_agg_mask = self._select_palette_from_expression(
            new_agg_expr, self.agg_palette_size, self.min_active_agg, NUM_AGGREGATIONS,
            self.agg_expression_threshold
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_regulation': new_act_reg,
            'act_expression': new_act_expr,
            'agg_mask': new_agg_mask,
            'agg_regulation': new_agg_reg,
            'agg_expression': new_agg_expr,
            'cross_regulation': new_cross_reg,
            'sin_extreme_coupling_events': coupling_events,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Expression
            'act_mean_expression': float(jnp.mean(new_act_expr)),
            'agg_mean_expression': float(jnp.mean(new_agg_expr)),
            # Network
            'act_n_active_links': int(jnp.sum(jnp.abs(new_act_reg) > 0.01)),
            'agg_n_active_links': int(jnp.sum(jnp.abs(new_agg_reg) > 0.01)),
            'cross_n_active_links': int(jnp.sum(new_cross_reg > 0.01)),
            # Sin-extreme coupling
            'sin_expression': float(new_act_expr[4]),
            'max_expression': float(new_agg_expr[2]),
            'min_expression': float(new_agg_expr[3]),
            'sin_max_coupling': float(new_cross_reg[4, 2]),
            'sin_min_coupling': float(new_cross_reg[4, 3]),
            'sin_extreme_coupling_events': coupling_events,
            # Sin status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with sin-extreme coupling status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'sin_expression': float(state['act_expression'][4]),
            'max_expression': float(state['agg_expression'][2]),
            'min_expression': float(state['agg_expression'][3]),
            'sin_max_coupling': float(state['cross_regulation'][4, 2]),
            'sin_min_coupling': float(state['cross_regulation'][4, 3]),
            'sin_extreme_coupling_events': state['sin_extreme_coupling_events'],
        }
