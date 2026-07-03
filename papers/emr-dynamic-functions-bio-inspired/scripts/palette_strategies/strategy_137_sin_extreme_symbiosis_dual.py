"""Strategy 137: Sin-Extreme-Symbiosis Dual.

Implements obligate mutualism between sin and extreme aggregations.
If one partner is removed, the other becomes vulnerable.

Key Mechanism:
- Sin and extreme_aggs (max/min) form symbiotic pairs
- When partner is active, protection is high
- When partner is lost, orphan becomes 2x more likely to be pruned
- Symbiosis formation rate tracks how quickly new pairs form

Biological Inspiration: Obligate mutualism in nature (e.g., clownfish-anemone,
mycorrhizal fungi-plant roots) where both partners depend on each other.
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


class SinExtremeSymbiosisDualStrategy(PaletteEvolutionStrategy):
    """Sin-Extreme symbiosis with obligate mutualism.

    Explicit coupling between sin and extreme aggs
    where they protect each other. If one is lost, the other becomes vulnerable.

    Critical innovation: Symbiotic protection where sin and max/min mutually
    reinforce each other's presence in the palette.
    """

    name = "sin_extreme_symbiosis_dual"
    description = "Dual: Obligate mutualism between sin and extreme aggs"

    def __init__(
        self,
        # === SYMBIOSIS PARAMETERS (CORE) ===
        symbiosis_pairs: List[Tuple[int, int]] = None,  # (sin, max), (sin, min)
        symbiosis_protection: float = 0.8,
        orphan_vulnerability: float = 2.0,  # 2x more likely to be pruned
        symbiosis_formation_rate: float = 0.15,
        symbiosis_break_threshold: float = 0.2,  # Low fitness causes break
        # === GUARANTEED INITIAL STATE ===
        sin_always_initial: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_affinity_lr: float = 0.12,
        # === PRUNING PARAMETERS ===
        prune_threshold_act: float = 0.2,
        prune_threshold_agg: float = 0.25,
        stagnation_prune_boost: float = 0.15,
        # === EXPLORATION PARAMETERS ===
        exploration_rate_act: float = 0.08,
        exploration_rate_agg: float = 0.12,
        symbiotic_exploration_boost: float = 0.5,  # Partners more likely to explore together
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === INITIAL PALETTES ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Sin-Extreme-Symbiosis strategy."""
        # Symbiosis pairs: (activation_idx, aggregation_idx)
        self.symbiosis_pairs = symbiosis_pairs or [(4, 2), (4, 3)]  # (sin, max), (sin, min)
        self.symbiosis_protection = symbiosis_protection
        self.orphan_vulnerability = orphan_vulnerability
        self.symbiosis_formation_rate = symbiosis_formation_rate
        self.symbiosis_break_threshold = symbiosis_break_threshold

        # Guaranteed state
        self.sin_always_initial = sin_always_initial
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_affinity_lr = cross_affinity_lr

        # Pruning
        self.prune_threshold_act = prune_threshold_act
        self.prune_threshold_agg = prune_threshold_agg
        self.stagnation_prune_boost = stagnation_prune_boost

        # Exploration
        self.exploration_rate_act = exploration_rate_act
        self.exploration_rate_agg = exploration_rate_agg
        self.symbiotic_exploration_boost = symbiotic_exploration_boost

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
        """Initialize state with sin-extreme symbiotic pairs."""
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

        # Symbiosis state - track active pairs
        # Matrix: symbiosis_strength[act_idx, agg_idx] = strength of symbiotic bond
        symbiosis_strength = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        # Initialize sin-extreme pairs
        for act_idx, agg_idx in self.symbiosis_pairs:
            symbiosis_strength = symbiosis_strength.at[act_idx, agg_idx].set(1.0)

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
            # Symbiosis (CORE)
            'symbiosis_strength': symbiosis_strength,
            'orphan_count_act': jnp.zeros(NUM_ACTIVATIONS),
            'orphan_count_agg': jnp.zeros(NUM_AGGREGATIONS),
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'symbiosis_events': 0,
            'orphan_pruning_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1370000),
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

    def _check_symbiosis_status(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Check which functions have active symbiotic partners.

        Returns:
            has_partner_act: Boolean array for activations
            has_partner_agg: Boolean array for aggregations
        """
        has_partner_act = jnp.zeros(NUM_ACTIVATIONS)
        has_partner_agg = jnp.zeros(NUM_AGGREGATIONS)

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                # Check if any agg partner exists
                for j in range(NUM_AGGREGATIONS):
                    if agg_mask[j] > 0.5 and symbiosis_strength[i, j] > 0.3:
                        has_partner_act = has_partner_act.at[i].set(1.0)
                        break

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                # Check if any act partner exists
                for i in range(NUM_ACTIVATIONS):
                    if act_mask[i] > 0.5 and symbiosis_strength[i, j] > 0.3:
                        has_partner_agg = has_partner_agg.at[j].set(1.0)
                        break

        return has_partner_act, has_partner_agg

    def _update_symbiosis(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, int]:
        """Update symbiotic relationships."""
        new_symbiosis = symbiosis_strength.copy()
        events = 0

        # Strengthen existing bonds on improvement
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        if new_symbiosis[i, j] > 0.1:
                            # Strengthen bond
                            new_symbiosis = new_symbiosis.at[i, j].set(
                                min(1.0, new_symbiosis[i, j] + 0.1)
                            )
                        elif new_symbiosis[i, j] < 0.1:
                            # Chance to form new bond
                            if jax.random.uniform(jax.random.PRNGKey(i * 100 + j)) < self.symbiosis_formation_rate:
                                new_symbiosis = new_symbiosis.at[i, j].set(0.3)
                                events += 1

        # Decay bonds for inactive pairs
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if act_mask[i] < 0.5 or agg_mask[j] < 0.5:
                    new_symbiosis = new_symbiosis.at[i, j].set(
                        new_symbiosis[i, j] * 0.9
                    )

        # CRITICAL: Sin-extreme bonds never fully break
        for act_idx, agg_idx in self.symbiosis_pairs:
            new_symbiosis = new_symbiosis.at[act_idx, agg_idx].set(
                max(0.5, float(new_symbiosis[act_idx, agg_idx]))
            )

        return new_symbiosis, events

    def _calculate_protection(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        symbiosis_strength: jnp.ndarray,
        has_partner_act: jnp.ndarray,
        has_partner_agg: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate protection based on symbiotic status."""
        act_protection = jnp.zeros(NUM_ACTIVATIONS)
        agg_protection = jnp.zeros(NUM_AGGREGATIONS)

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                if has_partner_act[i] > 0.5:
                    act_protection = act_protection.at[i].set(self.symbiosis_protection)
                else:
                    # Orphan - vulnerable
                    act_protection = act_protection.at[i].set(0.1)

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                if has_partner_agg[j] > 0.5:
                    agg_protection = agg_protection.at[j].set(self.symbiosis_protection)
                else:
                    # Orphan - vulnerable
                    agg_protection = agg_protection.at[j].set(0.1)

        # CRITICAL: Sin always has base protection
        act_protection = act_protection.at[4].set(max(0.6, float(act_protection[4])))
        # Extreme aggs always have base protection
        for agg in CORE_EXTREME_AGGS:
            agg_protection = agg_protection.at[agg].set(max(0.5, float(agg_protection[agg])))

        return act_protection, agg_protection

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update palette using symbiotic relationships."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        symbiosis_strength = state['symbiosis_strength']
        orphan_count_act = state['orphan_count_act']
        orphan_count_agg = state['orphan_count_agg']
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

        # Check symbiosis status
        has_partner_act, has_partner_agg = self._check_symbiosis_status(
            act_mask, agg_mask, symbiosis_strength
        )

        # Update orphan counts
        new_orphan_act = orphan_count_act.copy()
        new_orphan_agg = orphan_count_agg.copy()
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5 and has_partner_act[i] < 0.5:
                new_orphan_act = new_orphan_act.at[i].set(new_orphan_act[i] + 1)
            else:
                new_orphan_act = new_orphan_act.at[i].set(0)

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5 and has_partner_agg[j] < 0.5:
                new_orphan_agg = new_orphan_agg.at[j].set(new_orphan_agg[j] + 1)
            else:
                new_orphan_agg = new_orphan_agg.at[j].set(0)

        # Update symbiosis
        symbiosis_strength, symbiosis_events = self._update_symbiosis(
            act_mask, agg_mask, symbiosis_strength, fitness_delta
        )

        # Calculate protection
        act_protection, agg_protection = self._calculate_protection(
            act_mask, agg_mask, symbiosis_strength, has_partner_act, has_partner_agg
        )

        # Update affinities
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if improved:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    bonus = 1.0
                    if i == 4:  # Sin
                        bonus = 1.5
                    if has_partner_act[i] > 0.5:
                        bonus *= 1.3  # Symbiotic bonus
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * bonus)
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus = 1.4
                    if has_partner_agg[j] > 0.5:
                        bonus *= 1.3
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * bonus)
                    )

        # Sin and extreme affinity floors
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Pruning with orphan vulnerability
        new_act_mask = act_mask.copy()
        new_agg_mask = agg_mask.copy()
        orphan_pruning = 0

        rng_key, prune_key = jax.random.split(rng_key)
        prune_rand = jax.random.uniform(prune_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        if stagnation_count > 5:
            # Prune low-affinity activations (orphans more vulnerable)
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] > 0.5 and i != 4:
                    base_threshold = self.prune_threshold_act
                    if new_orphan_act[i] > 3:  # Orphan for 3+ generations
                        base_threshold *= self.orphan_vulnerability
                        orphan_pruning += 1
                    if new_act_aff[i] < base_threshold:
                        if prune_rand[i] > act_protection[i]:
                            new_act_mask = new_act_mask.at[i].set(0.0)

            # Prune aggregations
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] > 0.5 and j not in CORE_EXTREME_AGGS:
                    base_threshold = self.prune_threshold_agg
                    if new_orphan_agg[j] > 3:
                        base_threshold *= self.orphan_vulnerability
                        orphan_pruning += 1
                    if new_agg_aff[j] < base_threshold:
                        idx = NUM_ACTIVATIONS + j
                        if prune_rand[idx] > agg_protection[j]:
                            new_agg_mask = new_agg_mask.at[j].set(0.0)

        # CRITICAL: Ensure sin and extreme aggs are never removed
        new_act_mask = new_act_mask.at[4].set(1.0)
        for agg in CORE_EXTREME_AGGS:
            new_agg_mask = new_agg_mask.at[agg].set(1.0)

        # Exploration (partners explore together)
        rng_key, explore_key = jax.random.split(rng_key)
        explore_rand = jax.random.uniform(explore_key, shape=(NUM_ACTIVATIONS + NUM_AGGREGATIONS,))

        act_count = int(jnp.sum(new_act_mask))
        agg_count = int(jnp.sum(new_agg_mask))

        if act_count < self.max_active_act:
            for i in range(NUM_ACTIVATIONS):
                if new_act_mask[i] < 0.5:
                    rate = self.exploration_rate_act
                    # If existing partners, boost exploration
                    if has_partner_act.sum() > 0:
                        rate *= (1 + self.symbiotic_exploration_boost)
                    if explore_rand[i] < rate:
                        new_act_mask = new_act_mask.at[i].set(1.0)
                        break

        if agg_count < self.max_active_agg:
            for j in range(NUM_AGGREGATIONS):
                if new_agg_mask[j] < 0.5:
                    rate = self.exploration_rate_agg
                    if has_partner_agg.sum() > 0:
                        rate *= (1 + self.symbiotic_exploration_boost)
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
            'symbiosis_strength': symbiosis_strength,
            'orphan_count_act': new_orphan_act,
            'orphan_count_agg': new_orphan_agg,
            'cross_affinity': cross_affinity,
            'rng_key': rng_key,
            'generation': generation + 1,
            'stagnation_count': stagnation_count,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'symbiosis_events': state['symbiosis_events'] + symbiosis_events,
            'orphan_pruning_events': state['orphan_pruning_events'] + orphan_pruning,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'symbiosis_events': symbiosis_events,
            'orphan_pruning': orphan_pruning,
            'has_partners_act': int(has_partner_act.sum()),
            'has_partners_agg': int(has_partner_agg.sum()),
        }

        return new_state, metrics
