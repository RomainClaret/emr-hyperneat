"""Strategy 142: Regulatory-Sin-Extreme Dual.

Implements GRN-style regulatory network that promotes sin+extreme_agg
expression together while suppressing averaging aggregations.

Key Mechanism:
- Regulatory gene upregulates sin AND extreme_agg expression together
- Negative regulation for averaging aggs (sum, mean)
- Regulatory strength modulates based on fitness
- Creates positive feedback loop for sin+extreme coordination

Biological Inspiration: Gene regulatory networks (GRNs) where transcription
factors coordinate expression of functionally related genes.
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
    CORE_EXTREME_AGGS,
)


class RegulatorySinExtremeDualStrategy(PaletteEvolutionStrategy):
    """GRN-style regulatory network for sin+extreme coordination.

    Regulatory gene that upregulates sin and extreme
    aggregations while downregulating averaging aggregations.

    Critical innovation: Regulatory network creates explicit coordination
    between sin and extreme_aggs, making them more likely to be active together.
    """

    name = "regulatory_sin_extreme_dual"
    description = "Dual: GRN-style regulation promoting sin+extreme agg co-expression"

    def __init__(
        self,
        # === REGULATORY NETWORK (CORE) ===
        regulatory_strength: float = 0.3,
        upregulated_acts: List[int] = None,  # sin
        upregulated_aggs: List[int] = None,  # max, min
        downregulated_aggs: List[int] = None,  # sum, mean
        regulation_decay: float = 0.95,
        regulation_boost_on_fitness: float = 0.15,
        min_regulation_strength: float = 0.1,
        # === GUARANTEED INITIAL STATE ===
        sin_always_initial: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        # === EXPLORATION PARAMETERS ===
        exploration_rate_act: float = 0.08,
        exploration_rate_agg: float = 0.10,
        regulated_exploration_boost: float = 0.5,
        # === PRUNING PARAMETERS ===
        prune_threshold_act: float = 0.2,
        prune_threshold_agg: float = 0.25,
        stagnation_prune_after: int = 6,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === INITIAL PALETTES ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Regulatory-Sin-Extreme strategy."""
        # Regulatory network
        self.regulatory_strength = regulatory_strength
        self.upregulated_acts = upregulated_acts or [4]  # sin
        self.upregulated_aggs = upregulated_aggs or [2, 3]  # max, min
        self.downregulated_aggs = downregulated_aggs or [0, 1]  # sum, mean
        self.regulation_decay = regulation_decay
        self.regulation_boost_on_fitness = regulation_boost_on_fitness
        self.min_regulation_strength = min_regulation_strength

        # Guaranteed state
        self.sin_always_initial = sin_always_initial
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Exploration
        self.exploration_rate_act = exploration_rate_act
        self.exploration_rate_agg = exploration_rate_agg
        self.regulated_exploration_boost = regulated_exploration_boost

        # Pruning
        self.prune_threshold_act = prune_threshold_act
        self.prune_threshold_agg = prune_threshold_agg
        self.stagnation_prune_after = stagnation_prune_after

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial palettes
        default_act = list(DEFAULT_PALETTE_INDICES)
        if self.sin_always_initial and 4 not in default_act:
            default_act.append(4)

        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        if self.extreme_always_initial:
            for agg in CORE_EXTREME_AGGS:
                if agg not in default_agg:
                    default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with regulatory network."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # CRITICAL: Ensure sin and extreme in initial palette
        if self.sin_always_initial and 4 not in initial_act:
            initial_act = list(initial_act) + [4]

        initial_agg = list(initial_agg)
        if self.extreme_always_initial:
            for agg in CORE_EXTREME_AGGS:
                if agg not in initial_agg:
                    initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        act_affinities = act_affinities.at[4].set(0.8)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for agg in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[agg].set(0.75)

        # Regulatory state
        # Current regulatory gene expression level
        regulatory_expression = self.regulatory_strength

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Regulatory network (CORE)
            'regulatory_expression': regulatory_expression,
            'upregulation_events': 0,
            'downregulation_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1420000),
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

    def _apply_regulation(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        regulatory_expression: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Apply regulatory effects to affinities."""
        new_act_aff = act_affinities.copy()
        new_agg_aff = agg_affinities.copy()
        up_events = 0
        down_events = 0

        # Upregulate sin and extreme aggs
        for act in self.upregulated_acts:
            boost = regulatory_expression * 0.15
            new_act_aff = new_act_aff.at[act].set(
                min(1.0, new_act_aff[act] + boost)
            )
            if boost > 0.01:
                up_events += 1

        for agg in self.upregulated_aggs:
            boost = regulatory_expression * 0.12
            new_agg_aff = new_agg_aff.at[agg].set(
                min(1.0, new_agg_aff[agg] + boost)
            )
            if boost > 0.01:
                up_events += 1

        # Downregulate averaging aggs
        for agg in self.downregulated_aggs:
            if agg_mask[agg] > 0.5:
                penalty = regulatory_expression * 0.1
                new_agg_aff = new_agg_aff.at[agg].set(
                    max(0.1, new_agg_aff[agg] - penalty)
                )
                if penalty > 0.01:
                    down_events += 1

        return new_act_aff, new_agg_aff, up_events, down_events

    def _update_regulatory_expression(
        self,
        current_expression: float,
        fitness_delta: float,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> float:
        """Update regulatory gene expression based on context."""
        # Decay
        new_expression = current_expression * self.regulation_decay

        # Boost on fitness improvement
        if fitness_delta > 0:
            new_expression += self.regulation_boost_on_fitness * min(1.0, fitness_delta * 10)

        # Boost if upregulated members are active and performing
        upregulated_active = (
            sum(1 for a in self.upregulated_acts if act_mask[a] > 0.5) +
            sum(1 for a in self.upregulated_aggs if agg_mask[a] > 0.5)
        )
        if upregulated_active >= 2:
            new_expression += 0.05

        # Clamp
        new_expression = max(self.min_regulation_strength, min(1.0, new_expression))

        return new_expression

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update palette using regulatory network."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        regulatory_expression = state['regulatory_expression']
        rng_key = state['rng_key']

        fitness_delta = best_fitness - prev_best_fitness
        improved = fitness_delta > 0.001
        stagnated = fitness_delta < 0.0001

        # Update stagnation
        stagnation_count = state['stagnation_count']
        if stagnated:
            stagnation_count += 1
        else:
            stagnation_count = max(0, stagnation_count - 1)

        # Update regulatory expression
        regulatory_expression = self._update_regulatory_expression(
            regulatory_expression, fitness_delta, act_mask, agg_mask
        )

        # Apply regulation to affinities
        act_affinities, agg_affinities, up_events, down_events = self._apply_regulation(
            act_mask, agg_mask, act_affinities, agg_affinities, regulatory_expression
        )

        # Standard affinity update
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    bonus = 1.0
                    if i in self.upregulated_acts:
                        bonus = 1.5
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * bonus)
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    bonus = 1.0
                    if j in self.upregulated_aggs:
                        bonus = 1.4
                    elif j in self.downregulated_aggs:
                        bonus = 0.5
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * bonus)
                    )

        # Affinity floors for upregulated members
        for act in self.upregulated_acts:
            new_act_aff = new_act_aff.at[act].set(max(0.6, float(new_act_aff[act])))
        for agg in self.upregulated_aggs:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Pruning (with regulatory protection)
        new_act_mask = act_mask.copy()
        new_agg_mask = agg_mask.copy()

        rng_key, prune_key = jax.random.split(rng_key)
        prune_rand = jax.random.uniform(prune_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        if stagnation_count > self.stagnation_prune_after:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] > 0.5:
                    is_upregulated = i in self.upregulated_acts
                    threshold = self.prune_threshold_act
                    protection = regulatory_expression if is_upregulated else 0.0

                    if new_act_aff[i] < threshold and not is_upregulated:
                        if prune_rand[i] > protection:
                            new_act_mask = new_act_mask.at[i].set(0.0)

            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5:
                    is_upregulated = j in self.upregulated_aggs
                    is_downregulated = j in self.downregulated_aggs
                    threshold = self.prune_threshold_agg

                    # Downregulated have lower protection
                    if is_downregulated:
                        protection = 0.2
                    elif is_upregulated:
                        protection = regulatory_expression
                    else:
                        protection = 0.3

                    if new_agg_aff[j] < threshold and not is_upregulated:
                        idx = NUM_ACTIVATIONS + j
                        if prune_rand[idx] > protection:
                            new_agg_mask = new_agg_mask.at[j].set(0.0)

        # CRITICAL: Ensure upregulated members are never removed
        for act in self.upregulated_acts:
            new_act_mask = new_act_mask.at[act].set(1.0)
        for agg in self.upregulated_aggs:
            new_agg_mask = new_agg_mask.at[agg].set(1.0)

        # Exploration (upregulated functions more likely to explore)
        rng_key, explore_key = jax.random.split(rng_key)
        explore_rand = jax.random.uniform(explore_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        act_count = int(jnp.sum(new_act_mask))
        agg_count = int(jnp.sum(new_agg_mask))

        if act_count < self.max_active_act:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] < 0.5:
                    rate = self.exploration_rate_act
                    if i in self.upregulated_acts:
                        rate *= (1 + self.regulated_exploration_boost)
                    if explore_rand[i] < rate:
                        new_act_mask = new_act_mask.at[i].set(1.0)
                        break

        if agg_count < self.max_active_agg:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] < 0.5:
                    rate = self.exploration_rate_agg
                    if j in self.upregulated_aggs:
                        rate *= (1 + self.regulated_exploration_boost)
                    elif j in self.downregulated_aggs:
                        rate *= 0.5  # Less likely to explore
                    idx = NUM_ACTIVATIONS + j
                    if explore_rand[idx] < rate:
                        new_agg_mask = new_agg_mask.at[j].set(1.0)
                        break

        # Ensure minimums
        if int(jnp.sum(new_act_mask)) < self.min_active_act:
            new_act_mask = new_act_mask.at[0].set(1.0)
        if int(jnp.sum(new_agg_mask)) < self.min_active_agg:
            new_agg_mask = new_agg_mask.at[0].set(1.0)

        new_state = {
            **state,
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'regulatory_expression': regulatory_expression,
            'upregulation_events': state['upregulation_events'] + up_events,
            'downregulation_events': state['downregulation_events'] + down_events,
            'rng_key': rng_key,
            'generation': generation + 1,
            'stagnation_count': stagnation_count,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'regulatory_expression': regulatory_expression,
            'upregulation_events': up_events,
            'downregulation_events': down_events,
        }

        return new_state, metrics
