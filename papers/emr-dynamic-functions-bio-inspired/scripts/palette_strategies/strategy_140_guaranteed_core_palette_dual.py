"""Strategy 140: Guaranteed-Core-Palette Dual.

Implements a fixed initial palette with sin + extreme aggs that are
guaranteed to be present and protected, with gradual vulnerability decay.

Key Mechanism:
- Sin (4), max (2), min (3) always start active
- Guaranteed protection that slowly decays over 50 generations
- After full_vulnerability_after gens, guaranteed members can be removed
- Evolution adds/removes other functions while core remains stable

Biological Inspiration: Founder effect in genetics where initial population
members have outsized influence on gene pool composition.
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


class GuaranteedCorePaletteDualStrategy(PaletteEvolutionStrategy):
    """Guaranteed core palette with gradual vulnerability.

    Fixed initial palette with guaranteed sin + extreme
    aggs that are highly protected initially but can become vulnerable over time.

    Critical innovation: Guaranteed members have decaying protection, ensuring
    they're always present early while allowing eventual adaptation.
    """

    name = "guaranteed_core_palette_dual"
    description = "Dual: Guaranteed sin+extreme core with decaying protection"

    def __init__(
        self,
        # === GUARANTEED CORE (CRITICAL) ===
        guaranteed_activations: List[int] = None,  # Always start active
        guaranteed_aggregations: List[int] = None,  # Always start active
        guaranteed_protection_decay: float = 0.01,  # Per-gen decay rate
        full_vulnerability_after: int = 50,  # Gens until fully vulnerable
        min_core_protection: float = 0.5,  # Floor protection level
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        core_affinity_bonus: float = 0.3,
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
        """Initialize Guaranteed-Core-Palette strategy."""
        # Core members
        self.guaranteed_activations = guaranteed_activations or [4]  # sin
        self.guaranteed_aggregations = guaranteed_aggregations or [2, 3]  # max, min
        self.guaranteed_protection_decay = guaranteed_protection_decay
        self.full_vulnerability_after = full_vulnerability_after
        self.min_core_protection = min_core_protection

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.core_affinity_bonus = core_affinity_bonus

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

        # Initial palettes - include guaranteed members
        default_act = list(DEFAULT_PALETTE_INDICES)
        for act in self.guaranteed_activations:
            if act not in default_act:
                default_act.append(act)

        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        for agg in self.guaranteed_aggregations:
            if agg not in default_agg:
                default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with guaranteed core palette."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # CRITICAL: Ensure guaranteed members are in initial palette
        initial_act = list(initial_act)
        for act in self.guaranteed_activations:
            if act not in initial_act:
                initial_act.append(act)

        initial_agg = list(initial_agg)
        for agg in self.guaranteed_aggregations:
            if agg not in initial_agg:
                initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities - core members get bonus
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for act in self.guaranteed_activations:
            act_affinities = act_affinities.at[act].set(0.8)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)
        for agg in self.guaranteed_aggregations:
            agg_affinities = agg_affinities.at[agg].set(0.75)

        # Core protection level (starts at 1.0, decays)
        core_protection = 1.0

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Core protection (CRITICAL)
            'core_protection': core_protection,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1400000),
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
        """Update palette with guaranteed core protection."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        core_protection = state['core_protection']
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

        # Decay core protection (but keep floor)
        if generation < self.full_vulnerability_after:
            new_core_protection = max(
                self.min_core_protection,
                core_protection - self.guaranteed_protection_decay
            )
        else:
            new_core_protection = self.min_core_protection

        # Update affinities
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    bonus = 1.0
                    if i in self.guaranteed_activations:
                        bonus += self.core_affinity_bonus
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * bonus)
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    bonus = 1.0
                    if j in self.guaranteed_aggregations:
                        bonus += self.core_affinity_bonus
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * bonus)
                    )

        # Core members have affinity floor
        for act in self.guaranteed_activations:
            new_act_aff = new_act_aff.at[act].set(max(0.6, float(new_act_aff[act])))
        for agg in self.guaranteed_aggregations:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Pruning (core protected)
        new_act_mask = act_mask.copy()
        new_agg_mask = agg_mask.copy()

        rng_key, prune_key = jax.random.split(rng_key)
        prune_rand = jax.random.uniform(prune_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        if stagnation_count > self.stagnation_prune_after:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] > 0.5:
                    is_core = i in self.guaranteed_activations
                    threshold = self.prune_threshold_act
                    protection = new_core_protection if is_core else 0.0

                    if new_act_aff[i] < threshold and not is_core:
                        if prune_rand[i] > protection:
                            new_act_mask = new_act_mask.at[i].set(0.0)

            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5:
                    is_core = j in self.guaranteed_aggregations
                    threshold = self.prune_threshold_agg
                    protection = new_core_protection if is_core else 0.0

                    if new_agg_aff[j] < threshold and not is_core:
                        idx = NUM_ACTIVATIONS + j
                        if prune_rand[idx] > protection:
                            new_agg_mask = new_agg_mask.at[j].set(0.0)

        # CRITICAL: Ensure guaranteed members are never removed
        for act in self.guaranteed_activations:
            new_act_mask = new_act_mask.at[act].set(1.0)
        for agg in self.guaranteed_aggregations:
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
            'core_protection': new_core_protection,
            'rng_key': rng_key,
            'generation': generation + 1,
            'stagnation_count': stagnation_count,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'core_protection': new_core_protection,
        }

        return new_state, metrics
