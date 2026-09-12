"""Strategy 138: Sin-Extreme-Codiscovery Dual.

Implements cross-domain discovery triggers: when sin is discovered, extreme_agg
exploration probability increases, and vice versa.

Key Mechanism:
- Discovery of sin triggers +40% exploration probability for extreme aggs
- Discovery of extreme agg triggers +40% exploration for sin
- Discovery window: boost applies for 10 generations after discovery
- Initial sin exploration bias ensures early discovery

Biological Inspiration: Gene co-expression networks where activation of one gene
triggers expression of related genes in the same pathway.
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
    CORE_EXTREME_AGGS,
)


class SinExtremeCodeiscoveryDualStrategy(PaletteEvolutionStrategy):
    """Sin-Extreme codiscovery with cross-domain triggers.

    Discovery of one domain triggers exploration
    in the related domain, creating a positive feedback loop.

    Critical innovation: Cross-domain affinity boost when sin OR extreme agg
    is discovered, encouraging the other to follow.
    """

    name = "sin_extreme_codiscovery_dual"
    description = "Dual: Cross-domain discovery triggers between sin and extreme aggs"

    def __init__(
        self,
        # === CODISCOVERY PARAMETERS (CORE) ===
        codiscovery_boost: float = 0.4,  # +40% exploration for partner domain
        discovery_window: int = 10,  # Generations after discovery where boost applies
        cross_domain_affinity_rate: float = 0.2,
        initial_sin_exploration_bias: float = 0.15,
        # === GUARANTEED INITIAL STATE ===
        sin_always_initial: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        # === EXPLORATION PARAMETERS ===
        base_exploration_rate_act: float = 0.08,
        base_exploration_rate_agg: float = 0.10,
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
        """Initialize Sin-Extreme-Codiscovery strategy."""
        # Codiscovery parameters
        self.codiscovery_boost = codiscovery_boost
        self.discovery_window = discovery_window
        self.cross_domain_affinity_rate = cross_domain_affinity_rate
        self.initial_sin_exploration_bias = initial_sin_exploration_bias

        # Guaranteed state
        self.sin_always_initial = sin_always_initial
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Exploration
        self.base_exploration_rate_act = base_exploration_rate_act
        self.base_exploration_rate_agg = base_exploration_rate_agg

        # Pruning
        self.prune_threshold_act = prune_threshold_act
        self.prune_threshold_agg = prune_threshold_agg
        self.stagnation_prune_after = stagnation_prune_after

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial palettes - ALWAYS include sin and extreme aggs
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
        """Initialize state with codiscovery tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # CRITICAL: Ensure sin is ALWAYS in initial palette
        if self.sin_always_initial and 4 not in initial_act:
            initial_act = list(initial_act) + [4]

        # Ensure extreme aggs in initial palette
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
        act_affinities = act_affinities.at[4].set(0.8)  # Sin high

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for agg in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[agg].set(0.75)

        # Discovery tracking
        # -1 means never discovered, otherwise generation of discovery
        act_discovery_gen = jnp.ones(NUM_ACTIVATIONS) * -1
        agg_discovery_gen = jnp.ones(NUM_AGGREGATIONS) * -1

        # Mark initial discoveries
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_discovery_gen = act_discovery_gen.at[i].set(0)
        for j in initial_agg:
            if 0 <= j < NUM_AGGREGATIONS:
                agg_discovery_gen = agg_discovery_gen.at[j].set(0)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, agg].set(0.7)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Discovery tracking (CORE)
            'act_discovery_gen': act_discovery_gen,
            'agg_discovery_gen': agg_discovery_gen,
            # Exploration boost state
            'sin_discovery_boost_remaining': self.discovery_window,  # Since sin starts in palette
            'extreme_discovery_boost_remaining': self.discovery_window,  # Since extreme starts
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'codiscovery_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1380000),
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

    def _calculate_exploration_rates(
        self,
        sin_boost_remaining: int,
        extreme_boost_remaining: int,
        generation: int,
    ) -> Tuple[float, float]:
        """Calculate current exploration rates based on discovery boosts."""
        act_rate = self.base_exploration_rate_act
        agg_rate = self.base_exploration_rate_agg

        # If extreme was recently discovered, boost sin exploration
        if extreme_boost_remaining > 0:
            act_rate *= (1 + self.codiscovery_boost)

        # If sin was recently discovered, boost extreme exploration
        if sin_boost_remaining > 0:
            agg_rate *= (1 + self.codiscovery_boost)

        # Early generation bias toward sin
        if generation < 20:
            act_rate += self.initial_sin_exploration_bias

        return act_rate, agg_rate

    def _update_discovery_tracking(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        prev_act_mask: jnp.ndarray,
        prev_agg_mask: jnp.ndarray,
        act_discovery_gen: jnp.ndarray,
        agg_discovery_gen: jnp.ndarray,
        generation: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, bool, bool]:
        """Track new discoveries and return if sin/extreme was just discovered."""
        new_act_discovery = act_discovery_gen.copy()
        new_agg_discovery = agg_discovery_gen.copy()

        sin_just_discovered = False
        extreme_just_discovered = False

        # Check for new act discoveries
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5 and prev_act_mask[i] < 0.5:
                if new_act_discovery[i] < 0:
                    new_act_discovery = new_act_discovery.at[i].set(float(generation))
                    if i == 4:
                        sin_just_discovered = True

        # Check for new agg discoveries
        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5 and prev_agg_mask[j] < 0.5:
                if new_agg_discovery[j] < 0:
                    new_agg_discovery = new_agg_discovery.at[j].set(float(generation))
                    if j in CORE_EXTREME_AGGS:
                        extreme_just_discovered = True

        return new_act_discovery, new_agg_discovery, sin_just_discovered, extreme_just_discovered

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update palette using codiscovery triggers."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        act_discovery_gen = state['act_discovery_gen']
        agg_discovery_gen = state['agg_discovery_gen']
        sin_boost = state['sin_discovery_boost_remaining']
        extreme_boost = state['extreme_discovery_boost_remaining']
        cross_affinity = state['cross_affinity']
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

        # Store previous mask for discovery tracking
        prev_act_mask = act_mask.copy()
        prev_agg_mask = agg_mask.copy()

        # Get exploration rates based on current boosts
        act_explore_rate, agg_explore_rate = self._calculate_exploration_rates(
            sin_boost, extreme_boost, generation
        )

        # Update affinities
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    bonus = 1.0
                    if i == 4:
                        bonus = 1.5
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * bonus)
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus = 1.4
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * bonus)
                    )

        # Affinity floors
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Pruning (protect sin and extreme)
        new_act_mask = act_mask.copy()
        new_agg_mask = agg_mask.copy()

        rng_key, prune_key = jax.random.split(rng_key)
        prune_rand = jax.random.uniform(prune_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        if stagnation_count > self.stagnation_prune_after:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] > 0.5 and i != 4:
                    if new_act_aff[i] < self.prune_threshold_act:
                        if prune_rand[i] > 0.5:
                            new_act_mask = new_act_mask.at[i].set(0.0)

            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5 and j not in CORE_EXTREME_AGGS:
                    if new_agg_aff[j] < self.prune_threshold_agg:
                        idx = NUM_ACTIVATIONS + j
                        if prune_rand[idx] > 0.5:
                            new_agg_mask = new_agg_mask.at[j].set(0.0)

        # Exploration with codiscovery boost
        rng_key, explore_key = jax.random.split(rng_key)
        explore_rand = jax.random.uniform(explore_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        act_count = int(jnp.sum(new_act_mask))
        agg_count = int(jnp.sum(new_agg_mask))
        codiscovery_events = 0

        # Explore activations (with sin bias if extreme was discovered)
        if act_count < self.max_active_act:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] < 0.5:
                    rate = act_explore_rate
                    if i == 4:  # Extra sin bias
                        rate *= 1.5
                    if explore_rand[i] < rate:
                        new_act_mask = new_act_mask.at[i].set(1.0)
                        if i == 4:
                            codiscovery_events += 1
                        break

        # Explore aggregations (with extreme bias if sin was discovered)
        if agg_count < self.max_active_agg:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] < 0.5:
                    rate = agg_explore_rate
                    if j in CORE_EXTREME_AGGS:
                        rate *= 1.5
                    idx = NUM_ACTIVATIONS + j
                    if explore_rand[idx] < rate:
                        new_agg_mask = new_agg_mask.at[j].set(1.0)
                        if j in CORE_EXTREME_AGGS:
                            codiscovery_events += 1
                        break

        # CRITICAL: Ensure sin and extreme aggs are never removed
        new_act_mask = new_act_mask.at[4].set(1.0)
        for agg in CORE_EXTREME_AGGS:
            new_agg_mask = new_agg_mask.at[agg].set(1.0)

        # Update discovery tracking
        act_discovery_gen, agg_discovery_gen, sin_just, extreme_just = self._update_discovery_tracking(
            new_act_mask, new_agg_mask, prev_act_mask, prev_agg_mask,
            act_discovery_gen, agg_discovery_gen, generation
        )

        # Update boost counters
        new_sin_boost = sin_boost - 1 if sin_boost > 0 else 0
        new_extreme_boost = extreme_boost - 1 if extreme_boost > 0 else 0

        if sin_just:
            new_sin_boost = self.discovery_window
            codiscovery_events += 1
        if extreme_just:
            new_extreme_boost = self.discovery_window
            codiscovery_events += 1

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
            'act_discovery_gen': act_discovery_gen,
            'agg_discovery_gen': agg_discovery_gen,
            'sin_discovery_boost_remaining': new_sin_boost,
            'extreme_discovery_boost_remaining': new_extreme_boost,
            'cross_affinity': cross_affinity,
            'rng_key': rng_key,
            'generation': generation + 1,
            'stagnation_count': stagnation_count,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'codiscovery_events': state['codiscovery_events'] + codiscovery_events,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'codiscovery_events': codiscovery_events,
            'sin_boost_remaining': new_sin_boost,
            'extreme_boost_remaining': new_extreme_boost,
            'act_explore_rate': act_explore_rate,
            'agg_explore_rate': agg_explore_rate,
        }

        return new_state, metrics
