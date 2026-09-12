"""Strategy 107: Keystone Quorum Dual.

Combines quorum sensing (strategy 27), ecosystem dynamics/keystones (strategy 46),
and clonal hybrid (strategy 91) for population-consensus-driven keystone discovery.

Key Innovation:
- Population-level VOTING determines which functions become keystones
- Minority protection keeps exploration paths open for rare but useful functions
- Keystones emerge from consensus, not just individual success
- Sin-extreme consensus triggers mutual keystone promotion

Bio inspiration: Bacterial quorum sensing coordinates population behavior.
Functions that reach population consensus become community keystones.
Minority protection ensures rare variants aren't lost.
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


class KeystoneQuorumDualStrategy(PaletteEvolutionStrategy):
    """Population voting determines keystones.

    Quorum sensing for collective keystone discovery.
    Sin and extreme aggregations reaching consensus together become super-keystones.
    """

    name = "keystone_quorum_dual"
    description = "Dual: Population consensus determines keystones with minority protection"

    def __init__(
        self,
        # === QUORUM PARAMETERS (KEY INNOVATION) ===
        quorum_threshold: float = 0.4,           # 40% consensus for keystone
        minority_threshold: float = 0.05,        # Below 5% = protected exploration
        signal_decay: float = 0.85,              # Collective memory persistence
        keystone_from_quorum_gens: int = 5,      # Generations at quorum → keystone
        # === Keystone parameters ===
        keystone_protection: float = 0.85,       # Protection level for keystones
        keystone_facilitation: float = 0.4,      # Keystone helps neighbors
        sin_extreme_quorum_boost: float = 0.3,   # Boost when sin+extreme both at quorum
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Exploration parameters ===
        exploration_rate: float = 0.20,
        minority_explore_boost: float = 1.5,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Keystone Quorum Dual strategy."""
        # Quorum (KEY)
        self.quorum_threshold = quorum_threshold
        self.minority_threshold = minority_threshold
        self.signal_decay = signal_decay
        self.keystone_from_quorum_gens = keystone_from_quorum_gens

        # Keystone
        self.keystone_protection = keystone_protection
        self.keystone_facilitation = keystone_facilitation
        self.sin_extreme_quorum_boost = sin_extreme_quorum_boost

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Exploration
        self.exploration_rate = exploration_rate
        self.minority_explore_boost = minority_explore_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with quorum and keystone tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # COLLECTIVE MEMORY (population signals)
        act_collective = jnp.zeros(NUM_ACTIVATIONS)
        agg_collective = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_collective = act_collective.at[i].set(0.3)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_collective = agg_collective.at[i].set(0.3)

        # Quorum counters
        act_above_quorum_count = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        agg_above_quorum_count = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Collective memory (quorum)
            'act_collective': act_collective,
            'agg_collective': agg_collective,
            'act_above_quorum_count': act_above_quorum_count,
            'agg_above_quorum_count': agg_above_quorum_count,
            # Keystones
            'act_keystones': set(),
            'agg_keystones': set(),
            # Cross-domain affinity
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,
            # Stats
            'quorum_promotions': 0,
            'minority_explorations': 0,
            'sin_extreme_co_quorum': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1070000),
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

    def _update_collective_memory(
        self,
        collective: jnp.ndarray,
        mask: jnp.ndarray,
        fitness: float,
        best_fitness: float,
    ) -> jnp.ndarray:
        """Update population collective memory (quorum signals)."""
        # Weight by relative fitness
        if best_fitness > 0.01:
            vote_weight = (fitness / best_fitness) ** 2
        else:
            vote_weight = 1.0

        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)
        signal = active * vote_weight / n_active

        new_collective = (
            self.signal_decay * collective +
            (1 - self.signal_decay) * signal
        )
        return jnp.clip(new_collective, 0.0, 1.0)

    def _update_quorum_status(
        self,
        collective: jnp.ndarray,
        above_count: jnp.ndarray,
        keystones: set,
        n_funcs: int,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, set, int]:
        """Update quorum counters and keystone status."""
        new_above = above_count.copy()
        new_keystones = set(keystones)
        promotions = 0

        for i in range(n_funcs):
            signal = float(collective[i])

            if signal >= self.quorum_threshold:
                new_above = new_above.at[i].set(int(above_count[i]) + 1)

                # Check for keystone promotion
                if int(new_above[i]) >= self.keystone_from_quorum_gens:
                    if i not in new_keystones:
                        new_keystones.add(i)
                        promotions += 1
            else:
                new_above = new_above.at[i].set(0)

        return new_above, new_keystones, promotions

    def _compute_sin_extreme_boost(
        self,
        act_collective: jnp.ndarray,
        agg_collective: jnp.ndarray,
    ) -> Tuple[float, float, int]:
        """Compute boost when sin and extremes are both at quorum."""
        sin_at_quorum = float(act_collective[4]) >= self.quorum_threshold
        extremes_at_quorum = any(
            float(agg_collective[j]) >= self.quorum_threshold
            for j in CORE_EXTREME_AGGS
        )

        if sin_at_quorum and extremes_at_quorum:
            return self.sin_extreme_quorum_boost, self.sin_extreme_quorum_boost, 1
        return 0.0, 0.0, 0

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        collective: jnp.ndarray,
        keystones: set,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        stagnation: int,
        generation: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette using quorum-based scoring."""
        k1, k2 = jax.random.split(key)

        # Score: affinity + collective signal + keystone bonus
        score = affinities + collective * 0.4

        for i in keystones:
            if 0 <= i < n_funcs:
                score = score.at[i].set(score[i] + 0.5)

        # Preference boost
        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + 0.6)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # EXPLORATION: Add preferred indices with probability
        exploration_prob = self.exploration_rate * (1 + stagnation * 0.1)
        if prefer_indices:
            for idx in prefer_indices:
                if 0 <= idx < n_funcs and mask[idx] < 0.5:
                    # Minority boost
                    boost = self.minority_explore_boost if collective[idx] < self.minority_threshold else 1.0
                    if float(jax.random.uniform(k1)) < exploration_prob * boost:
                        mask = mask.at[idx].set(1.0)
                    k1, _ = jax.random.split(k1)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        minority_explore = 0
        if active_count < min_diversity:
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k2, jnp.array(inactive),
                                          shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    minority_explore += 1

        return mask, minority_explore

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with quorum-keystone dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === COLLECTIVE MEMORY UPDATE ===
        new_act_collective = self._update_collective_memory(
            state['act_collective'], state['act_mask'], best_fitness, new_best
        )
        new_agg_collective = self._update_collective_memory(
            state['agg_collective'], state['agg_mask'], best_fitness, new_best
        )

        # === SIN-EXTREME CO-QUORUM BOOST ===
        sin_boost, agg_boost, co_quorum = self._compute_sin_extreme_boost(
            new_act_collective, new_agg_collective
        )
        if co_quorum:
            new_act_collective = new_act_collective.at[4].set(
                min(1.0, new_act_collective[4] + sin_boost)
            )
            for j in CORE_EXTREME_AGGS:
                new_agg_collective = new_agg_collective.at[j].set(
                    min(1.0, new_agg_collective[j] + agg_boost)
                )

        # === QUORUM STATUS UPDATE ===
        new_act_above, new_act_keystones, act_promotions = self._update_quorum_status(
            new_act_collective, state['act_above_quorum_count'],
            state['act_keystones'], NUM_ACTIVATIONS, prefer_indices=[4]
        )
        new_agg_above, new_agg_keystones, agg_promotions = self._update_quorum_status(
            new_agg_collective, state['agg_above_quorum_count'],
            state['agg_keystones'], NUM_AGGREGATIONS, prefer_indices=list(CORE_EXTREME_AGGS)
        )

        # === AFFINITY UPDATE ===
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * fitness_delta)
                    )
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta)
                    )

        # === PALETTE SELECTION ===
        new_act_mask, act_minority = self._select_palette(
            new_act_aff, new_act_collective, new_act_keystones,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            self.min_diversity_act, new_stagnation, generation, k1,
            prefer_indices=[4]
        )
        new_agg_mask, agg_minority = self._select_palette(
            new_agg_aff, new_agg_collective, new_agg_keystones,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            self.min_diversity_agg, new_stagnation, generation, k2,
            prefer_indices=list(CORE_EXTREME_AGGS)
        )

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_collective': new_act_collective,
            'agg_collective': new_agg_collective,
            'act_above_quorum_count': new_act_above,
            'agg_above_quorum_count': new_agg_above,
            'act_keystones': new_act_keystones,
            'agg_keystones': new_agg_keystones,
            'cross_affinity': state['cross_affinity'],
            'quorum_promotions': state['quorum_promotions'] + act_promotions + agg_promotions,
            'minority_explorations': state['minority_explorations'] + act_minority + agg_minority,
            'sin_extreme_co_quorum': state['sin_extreme_co_quorum'] + co_quorum,
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
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Quorum metrics (KEY)
            'sin_collective': float(new_act_collective[4]),
            'max_collective': float(new_agg_collective[2]),
            'min_collective': float(new_agg_collective[3]),
            'n_act_keystones': len(new_act_keystones),
            'n_agg_keystones': len(new_agg_keystones),
            'sin_is_keystone': 4 in new_act_keystones,
            'quorum_promotions': new_state['quorum_promotions'],
            'sin_extreme_co_quorum': new_state['sin_extreme_co_quorum'],
            'minority_explorations': new_state['minority_explorations'],
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'n_act_keystones': len(state['act_keystones']),
            'n_agg_keystones': len(state['agg_keystones']),
            'sin_is_keystone': 4 in state['act_keystones'],
            'quorum_promotions': state['quorum_promotions'],
            'generation': state['generation'],
        }
