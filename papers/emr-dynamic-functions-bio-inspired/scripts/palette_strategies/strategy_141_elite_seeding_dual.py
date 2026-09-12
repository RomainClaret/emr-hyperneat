"""Strategy 141: Elite-Seeding Dual.

Implements elite preservation with sin-extreme seeding: top genomes
periodically seed sin+extreme_agg combinations to next generation.

Key Mechanism:
- Elite genomes (top performers) get sin if missing (80% rate)
- Elite genomes get max/min if missing (90% rate)
- Re-seeding occurs at specific generations (0, 10, 20, 30)
- Ensures periodic "reminders" to use sin and extreme aggs

Biological Inspiration: Horizontal gene transfer in bacteria where beneficial
genes spread through populations independently of reproduction.
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


class EliteSeedingDualStrategy(PaletteEvolutionStrategy):
    """Elite preservation with sin-extreme seeding.

    Top performers periodically receive sin and extreme
    agg "seeds" to ensure these critical functions remain in the palette.

    Critical innovation: Periodic seeding at key generations reinforces the
    importance of sin and extreme aggregations.
    """

    name = "elite_seeding_dual"
    description = "Dual: Elite seeding of sin and extreme aggs at key generations"

    def __init__(
        self,
        # === SEEDING PARAMETERS (CORE) ===
        elite_sin_rate: float = 0.8,  # 80% of elite get sin if missing
        elite_extreme_rate: float = 0.9,  # 90% of elite get max/min if missing
        seeding_generations: List[int] = None,  # Re-seed at these gens
        seed_on_stagnation: bool = True,
        stagnation_seed_threshold: int = 10,
        # === GUARANTEED INITIAL STATE ===
        sin_always_initial: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        seeding_affinity_boost: float = 0.2,
        # === EXPLORATION PARAMETERS ===
        exploration_rate_act: float = 0.08,
        exploration_rate_agg: float = 0.10,
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
        """Initialize Elite-Seeding strategy."""
        # Seeding parameters
        self.elite_sin_rate = elite_sin_rate
        self.elite_extreme_rate = elite_extreme_rate
        self.seeding_generations = seeding_generations or [0, 10, 20, 30]
        self.seed_on_stagnation = seed_on_stagnation
        self.stagnation_seed_threshold = stagnation_seed_threshold

        # Guaranteed state
        self.sin_always_initial = sin_always_initial
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.seeding_affinity_boost = seeding_affinity_boost

        # Exploration
        self.exploration_rate_act = exploration_rate_act
        self.exploration_rate_agg = exploration_rate_agg

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
        """Initialize state with seeding tracking."""
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

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Seeding tracking
            'seeding_events': 0,
            'last_seed_gen': -1,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1410000),
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

    def _should_seed(self, generation: int, stagnation_count: int) -> bool:
        """Check if we should perform seeding this generation."""
        if generation in self.seeding_generations:
            return True
        if self.seed_on_stagnation and stagnation_count >= self.stagnation_seed_threshold:
            return True
        return False

    def _perform_seeding(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        rng_key: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Perform elite seeding of sin and extreme aggs."""
        new_act_mask = act_mask.copy()
        new_agg_mask = agg_mask.copy()
        new_act_aff = act_affinities.copy()
        new_agg_aff = agg_affinities.copy()
        events = 0

        rng_key, sin_key, agg_key = jax.random.split(rng_key, 3)

        # Seed sin if missing (with elite_sin_rate probability)
        if new_act_mask[4] < 0.5:
            if jax.random.uniform(sin_key) < self.elite_sin_rate:
                new_act_mask = new_act_mask.at[4].set(1.0)
                new_act_aff = new_act_aff.at[4].set(
                    min(1.0, new_act_aff[4] + self.seeding_affinity_boost)
                )
                events += 1
        else:
            # Boost sin affinity anyway
            new_act_aff = new_act_aff.at[4].set(
                min(1.0, new_act_aff[4] + self.seeding_affinity_boost * 0.5)
            )

        # Seed extreme aggs if missing
        agg_rand = jax.random.uniform(agg_key, shape=(len(CORE_EXTREME_AGGS),))
        for idx, agg in enumerate(CORE_EXTREME_AGGS):
            if new_agg_mask[agg] < 0.5:
                if agg_rand[idx] < self.elite_extreme_rate:
                    new_agg_mask = new_agg_mask.at[agg].set(1.0)
                    new_agg_aff = new_agg_aff.at[agg].set(
                        min(1.0, new_agg_aff[agg] + self.seeding_affinity_boost)
                    )
                    events += 1
            else:
                # Boost affinity anyway
                new_agg_aff = new_agg_aff.at[agg].set(
                    min(1.0, new_agg_aff[agg] + self.seeding_affinity_boost * 0.5)
                )

        return new_act_mask, new_agg_mask, new_act_aff, new_agg_aff, events

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update palette with elite seeding."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
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

        # Check for seeding
        seeding_events = 0
        if self._should_seed(generation, stagnation_count):
            rng_key, seed_key = jax.random.split(rng_key)
            act_mask, agg_mask, act_affinities, agg_affinities, seeding_events = self._perform_seeding(
                act_mask, agg_mask, act_affinities, agg_affinities, seed_key
            )
            # Reset stagnation after seeding
            if seeding_events > 0:
                stagnation_count = 0

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

        # Pruning
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

        # CRITICAL: Ensure sin and extreme aggs are never removed
        new_act_mask = new_act_mask.at[4].set(1.0)
        for agg in CORE_EXTREME_AGGS:
            new_agg_mask = new_agg_mask.at[agg].set(1.0)

        # Exploration
        rng_key, explore_key = jax.random.split(rng_key)
        explore_rand = jax.random.uniform(explore_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        act_count = int(jnp.sum(new_act_mask))
        agg_count = int(jnp.sum(new_agg_mask))

        if act_count < self.max_active_act:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] < 0.5:
                    if explore_rand[i] < self.exploration_rate_act:
                        new_act_mask = new_act_mask.at[i].set(1.0)
                        break

        if agg_count < self.max_active_agg:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] < 0.5:
                    idx = NUM_ACTIVATIONS + j
                    if explore_rand[idx] < self.exploration_rate_agg:
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
            'seeding_events': state['seeding_events'] + seeding_events,
            'last_seed_gen': generation if seeding_events > 0 else state['last_seed_gen'],
            'rng_key': rng_key,
            'generation': generation + 1,
            'stagnation_count': stagnation_count,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'seeding_events': seeding_events,
            'is_seeding_gen': generation in self.seeding_generations,
        }

        return new_state, metrics
