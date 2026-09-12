"""Strategy 131: Quorum-Morphogen Consensus-Driven Spatial Dual.

Combines Quorum Sensing (#27) with Morphogen Gradient (#31).
Population voting determines morphogen source positions.

Key Innovation:
- Population-level function usage creates "votes" for morphogen sources
- Functions used by many individuals emit stronger morphogen
- Creates consensus-driven spatial exploration
- Sin and extreme aggs get voting bonuses

Biological basis: In bacterial quorum sensing, population-level signals
guide individual behavior. Here, collective usage patterns guide which
functions become morphogen sources, attracting further exploration.

Expected: Consensus-driven exploration toward population-validated functions.
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


class QuorumMorphogenDualStrategy(PaletteEvolutionStrategy):
    """Quorum-morphogen consensus-driven strategy for dual palette evolution.

    Population voting creates morphogen sources,
    guiding exploration toward consensus-validated functions.

    Critical innovation: Collective usage patterns guide spatial exploration.
    """

    name = "quorum_morphogen_dual"
    description = "Dual: Population voting determines morphogen source strength"

    def __init__(
        self,
        # === Quorum parameters ===
        quorum_threshold: float = 0.4,
        quorum_vote_weight: float = 0.2,
        quorum_decay: float = 0.90,
        # === Morphogen parameters ===
        morphogen_decay: float = 0.92,
        source_emission_rate: float = 0.3,
        quorum_source_boost: float = 0.5,
        # === Sin and extreme bonuses ===
        sin_idx: int = 4,
        sin_vote_bonus: float = 0.25,
        extreme_vote_bonus: float = 0.3,
        # === Mutation parameters ===
        base_mutation_rate: float = 0.10,
        morphogen_mutation_weight: float = 0.3,
        # === General parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Quorum-Morphogen strategy."""
        # Quorum
        self.quorum_threshold = quorum_threshold
        self.quorum_vote_weight = quorum_vote_weight
        self.quorum_decay = quorum_decay

        # Morphogen
        self.morphogen_decay = morphogen_decay
        self.source_emission_rate = source_emission_rate
        self.quorum_source_boost = quorum_source_boost

        # Sin and extreme
        self.sin_idx = sin_idx
        self.sin_vote_bonus = sin_vote_bonus
        self.extreme_vote_bonus = extreme_vote_bonus

        # Mutation
        self.base_mutation_rate = base_mutation_rate
        self.morphogen_mutation_weight = morphogen_mutation_weight

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with quorum and morphogen tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)

        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(agg_affinities[i] + 0.2)

        # Quorum votes (population-level usage signal)
        act_votes = jnp.zeros(NUM_ACTIVATIONS)
        agg_votes = jnp.zeros(NUM_AGGREGATIONS)

        # Morphogen concentrations
        act_morphogen = jnp.zeros(NUM_ACTIVATIONS)
        agg_morphogen = jnp.zeros(NUM_AGGREGATIONS)

        # Initialize votes for initial palette
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_votes = act_votes.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_votes = agg_votes.at[i].set(0.5)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Quorum
            'act_votes': act_votes,
            'agg_votes': agg_votes,
            # Morphogen
            'act_morphogen': act_morphogen,
            'agg_morphogen': agg_morphogen,
            # Stats
            'quorum_reached_count': 0,
            'consensus_mutations': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1310000),
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

    def _has_quorum(self, votes: float) -> bool:
        """Check if function has reached quorum threshold."""
        return votes >= self.quorum_threshold

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with quorum-driven morphogen sources."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === UPDATE QUORUM VOTES ===
        act_votes = state['act_votes'] * self.quorum_decay
        agg_votes = state['agg_votes'] * self.quorum_decay

        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        quorum_reached_count = state['quorum_reached_count']
        consensus_mutations = state['consensus_mutations']

        # Active functions get votes (simulating population usage)
        for i in range(NUM_ACTIVATIONS):
            if float(act_mask[i]) > 0.5:
                vote = self.quorum_vote_weight
                if improved:
                    vote *= 1.5
                if i == self.sin_idx:
                    vote += self.sin_vote_bonus
                act_votes = act_votes.at[i].add(vote)

        for i in range(NUM_AGGREGATIONS):
            if float(agg_mask[i]) > 0.5:
                vote = self.quorum_vote_weight
                if improved:
                    vote *= 1.5
                if i in CORE_EXTREME_AGGS:
                    vote += self.extreme_vote_bonus
                agg_votes = agg_votes.at[i].add(vote)

        # Clamp votes
        act_votes = jnp.clip(act_votes, 0.0, 1.0)
        agg_votes = jnp.clip(agg_votes, 0.0, 1.0)

        # === UPDATE MORPHOGEN FROM QUORUM ===
        act_morphogen = state['act_morphogen'] * self.morphogen_decay
        agg_morphogen = state['agg_morphogen'] * self.morphogen_decay

        # Functions with quorum become strong morphogen sources
        for i in range(NUM_ACTIVATIONS):
            emission = self.source_emission_rate
            if self._has_quorum(float(act_votes[i])):
                emission += self.quorum_source_boost
                quorum_reached_count += 1
            if float(act_mask[i]) > 0.5:
                act_morphogen = act_morphogen.at[i].add(emission)

        for i in range(NUM_AGGREGATIONS):
            emission = self.source_emission_rate
            if self._has_quorum(float(agg_votes[i])):
                emission += self.quorum_source_boost
            if float(agg_mask[i]) > 0.5:
                agg_morphogen = agg_morphogen.at[i].add(emission)

        # === ACTIVATION MUTATION (morphogen-weighted) ===
        candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k1) < self.base_mutation_rate:
            # Weight by morphogen + vote
            weights = []
            for i in candidates:
                w = 0.1
                w += float(act_morphogen[i]) * self.morphogen_mutation_weight
                w += float(act_votes[i]) * 0.2
                if i == self.sin_idx:
                    w += 0.3
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs))
            act_mask = act_mask.at[new_idx].set(1.0)

            if float(act_votes[new_idx]) > 0.2:
                consensus_mutations += 1

        # === AGGREGATION MUTATION ===
        candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k3) < self.base_mutation_rate:
            weights = []
            for i in candidates:
                w = 0.1
                w += float(agg_morphogen[i]) * self.morphogen_mutation_weight
                w += float(agg_votes[i]) * 0.2
                if i in CORE_EXTREME_AGGS:
                    w += 0.3
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs))
            agg_mask = agg_mask.at[new_idx].set(1.0)

        # Update affinities
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)

        # Clamp affinities
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)

        # Ensure minimum diversity
        if sum(float(act_mask[i]) for i in range(NUM_ACTIVATIONS)) < self.min_active_act:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k1, jnp.array(candidates)))
                act_mask = act_mask.at[new_idx].set(1.0)

        if sum(float(agg_mask[i]) for i in range(NUM_AGGREGATIONS)) < self.min_active_agg:
            candidates = [i for i in CORE_EXTREME_AGGS if float(agg_mask[i]) < 0.5]
            if not candidates:
                candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k3, jnp.array(candidates)))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        new_state = {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_votes': act_votes,
            'agg_votes': agg_votes,
            'act_morphogen': act_morphogen,
            'agg_morphogen': agg_morphogen,
            'quorum_reached_count': quorum_reached_count,
            'consensus_mutations': consensus_mutations,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'quorum_reached_count': quorum_reached_count,
            'consensus_mutations': consensus_mutations,
            'mean_act_votes': float(act_votes.mean()),
            'mean_agg_votes': float(agg_votes.mean()),
        }

        return new_state, metrics
