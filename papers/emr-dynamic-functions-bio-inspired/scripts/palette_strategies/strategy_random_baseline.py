"""Strategy: Random Baseline.

Level 5 ablation: No sin seeding, no mechanism, random exploration only.

Test: What is the baseline sin discovery rate with pure random exploration?
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
)


class RandomBaselineStrategy(PaletteEvolutionStrategy):
    """Pure random exploration baseline - no mechanism, no sin bias.

    Baseline for measuring what random exploration achieves.
    This represents the null hypothesis - no bio-inspired mechanism.
    """

    name = "random_baseline"
    description = "L5: Random baseline - no mechanism, no sin bias"

    def __init__(
        self,
        # NO sin initial seeding
        # NO mechanism
        # === BASIC EXPLORATION ===
        exploration_rate_act: float = 0.10,
        exploration_rate_agg: float = 0.12,
        prune_threshold_act: float = 0.25,
        prune_threshold_agg: float = 0.30,
        stagnation_prune_after: int = 5,
        # === BASIC AFFINITY (no floors) ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.12,
        affinity_decay: float = 0.95,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === INITIAL PALETTES (DEFAULT - NO SIN) ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Random Baseline strategy."""
        # Basic exploration
        self.exploration_rate_act = exploration_rate_act
        self.exploration_rate_agg = exploration_rate_agg
        self.prune_threshold_act = prune_threshold_act
        self.prune_threshold_agg = prune_threshold_agg
        self.stagnation_prune_after = stagnation_prune_after

        # Basic affinity (no floors)
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial palettes - DEFAULT (NO SIN, NO EXTREME AGGS)
        # This uses only [0, 1, 5] for acts and [0, 1] for aggs
        self.initial_act_palette = initial_act_palette or list(DEFAULT_PALETTE_INDICES)
        self.initial_agg_palette = initial_agg_palette or list(DEFAULT_AGG_PALETTE_INDICES)

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state WITHOUT sin seeding."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # NO SIN SEEDING - use default palette only
        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Uniform affinities - no bias toward any function
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'rng_key': jax.random.PRNGKey(seed + 9990000),
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

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Simple random exploration - no mechanism."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        rng_key = state['rng_key']

        fitness_delta = best_fitness - prev_best_fitness
        improved = fitness_delta > 0.001
        stagnated = fitness_delta < 0.0001

        stagnation_count = state['stagnation_count']
        if stagnated:
            stagnation_count += 1
        else:
            stagnation_count = max(0, stagnation_count - 1)

        # Simple affinity update - no special treatment
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr)
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr)
                    )

        # NO AFFINITY FLOORS - pure decay

        # Random pruning
        new_act_mask = act_mask.copy()
        new_agg_mask = agg_mask.copy()

        rng_key, prune_key = jax.random.split(rng_key)
        prune_rand = jax.random.uniform(prune_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        if stagnation_count > self.stagnation_prune_after:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] > 0.5:
                    if new_act_aff[i] < self.prune_threshold_act:
                        if prune_rand[i] > 0.5:  # 50% chance
                            new_act_mask = new_act_mask.at[i].set(0.0)

            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5:
                    if new_agg_aff[j] < self.prune_threshold_agg:
                        idx = NUM_ACTIVATIONS + j
                        if prune_rand[idx] > 0.5:
                            new_agg_mask = new_agg_mask.at[j].set(0.0)

        # Random exploration
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
            # Add random function
            rng_key, add_key = jax.random.split(rng_key)
            available = [i for i in range(NUM_ACTIVATIONS) if new_act_mask[i] < 0.5]
            if available:
                idx = int(jax.random.randint(add_key, (), 0, len(available)))
                new_act_mask = new_act_mask.at[available[idx]].set(1.0)

        if int(jnp.sum(new_agg_mask)) < self.min_active_agg:
            rng_key, add_key = jax.random.split(rng_key)
            available = [j for j in range(NUM_AGGREGATIONS) if new_agg_mask[j] < 0.5]
            if available:
                idx = int(jax.random.randint(add_key, (), 0, len(available)))
                new_agg_mask = new_agg_mask.at[available[idx]].set(1.0)

        new_state = {
            **state,
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'rng_key': rng_key,
            'generation': generation + 1,
            'stagnation_count': stagnation_count,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'stagnation_count': stagnation_count,
            'act_count': int(jnp.sum(new_act_mask)),
            'agg_count': int(jnp.sum(new_agg_mask)),
        }

        return new_state, metrics
