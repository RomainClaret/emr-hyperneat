"""Strategy 112: GRN + Quorum + Minority Dual.

Combines genetic regulatory network (strategy 33) with quorum sensing
(strategy 27) for population-consensus-driven regulatory circuits.

Key Innovation:
- Population CONSENSUS shapes regulatory circuit formation
- GRN attractors lock sin-extreme pairings into stable expression
- MINORITY PROTECTION prevents pruning of rare but useful functions
- Regulatory links form only after population consensus

Bio inspiration: Gene regulatory networks with population-level consensus.
Minority protection ensures rare genetic variants aren't lost.
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


class GRNQuorumMinorityDualStrategy(PaletteEvolutionStrategy):
    """Population consensus drives regulatory circuit formation.

    GRN attractors lock sin-extreme after consensus.
    Minority protection keeps exploration paths open.
    """

    name = "grn_quorum_minority_dual"
    description = "Dual: GRN attractors form after population consensus with minority protection"

    def __init__(
        self,
        # === GRN PARAMETERS ===
        hill_coefficient: float = 2.0,
        regulation_learning_rate: float = 0.08,
        sin_extreme_regulation_link: float = 0.5,
        # === QUORUM PARAMETERS (KEY INNOVATION) ===
        quorum_for_regulation: float = 0.3,      # Consensus needed for regulation
        minority_regulation_protection: float = 0.5,  # Protect minority from deactivation
        regulation_from_quorum_strength: float = 0.4,
        signal_decay: float = 0.85,
        minority_threshold: float = 0.05,
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Exploration ===
        exploration_rate: float = 0.22,
        minority_explore_boost: float = 2.0,
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
        """Initialize GRN Quorum Minority Dual strategy."""
        # GRN
        self.hill_coefficient = hill_coefficient
        self.regulation_learning_rate = regulation_learning_rate
        self.sin_extreme_regulation_link = sin_extreme_regulation_link

        # Quorum (KEY)
        self.quorum_for_regulation = quorum_for_regulation
        self.minority_regulation_protection = minority_regulation_protection
        self.regulation_from_quorum_strength = regulation_from_quorum_strength
        self.signal_decay = signal_decay
        self.minority_threshold = minority_threshold

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
        """Initialize state with GRN and quorum tracking."""
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

        # GRN regulation matrix (cross-domain)
        # regulation[i,j] = how much activation i regulates aggregation j
        regulation_matrix = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        # Initial sin → extreme regulation links
        regulation_matrix = regulation_matrix.at[4, 2].set(self.sin_extreme_regulation_link * 0.5)
        regulation_matrix = regulation_matrix.at[4, 3].set(self.sin_extreme_regulation_link * 0.5)

        # Collective memory (quorum signals)
        act_collective = jnp.zeros(NUM_ACTIVATIONS)
        agg_collective = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_collective = act_collective.at[i].set(0.3)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_collective = agg_collective.at[i].set(0.3)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # GRN
            'regulation_matrix': regulation_matrix,
            'regulation_formed': set(),  # Set of (act_idx, agg_idx) tuples
            # Collective memory (quorum)
            'act_collective': act_collective,
            'agg_collective': agg_collective,
            # Stats
            'regulation_formations': 0,
            'minority_protections': 0,
            'quorum_regulations': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1120000),
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
        """Update population collective memory."""
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

    def _update_regulation_from_quorum(
        self,
        regulation_matrix: jnp.ndarray,
        regulation_formed: set,
        act_collective: jnp.ndarray,
        agg_collective: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, set, int]:
        """Form new regulatory links when both reach quorum."""
        new_reg = regulation_matrix.copy()
        new_formed = set(regulation_formed)
        formations = 0

        # Check sin → extreme regulation formation
        if float(act_collective[4]) >= self.quorum_for_regulation:
            for j in CORE_EXTREME_AGGS:
                if float(agg_collective[j]) >= self.quorum_for_regulation:
                    if (4, j) not in new_formed:
                        # Form new regulatory link
                        strength = self.sin_extreme_regulation_link * self.regulation_from_quorum_strength
                        new_reg = new_reg.at[4, j].set(
                            min(1.0, float(new_reg[4, j]) + strength)
                        )
                        new_formed.add((4, j))
                        formations += 1

        return new_reg, new_formed, formations

    def _apply_regulation(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        regulation_matrix: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply GRN regulation: active activations boost regulated aggregations."""
        boost = jnp.zeros(NUM_AGGREGATIONS)

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                for j in range(NUM_AGGREGATIONS):
                    reg_strength = float(regulation_matrix[i, j])
                    if reg_strength > 0.01:
                        # Hill function for activation
                        act_level = float(act_affinities[i])
                        hill = (act_level ** self.hill_coefficient) / (
                            0.5 ** self.hill_coefficient + act_level ** self.hill_coefficient
                        )
                        boost = boost.at[j].add(reg_strength * hill * 0.2)

        return jnp.clip(agg_affinities + boost, 0.0, 1.0)

    def _select_palette_with_minority(
        self,
        affinities: jnp.ndarray,
        collective: jnp.ndarray,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        stagnation: int,
        generation: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int, int]:
        """Select palette with minority protection."""
        k1, k2 = jax.random.split(key)

        # Score: affinity + collective signal
        score = affinities + collective * 0.4

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

        # EXPLORATION with minority boost
        exploration_prob = self.exploration_rate * (1 + stagnation * 0.1)
        minority_protections = 0

        if prefer_indices:
            for idx in prefer_indices:
                if 0 <= idx < n_funcs and mask[idx] < 0.5:
                    # Minority gets extra boost
                    is_minority = collective[idx] < self.minority_threshold
                    boost = self.minority_explore_boost if is_minority else 1.5
                    if float(jax.random.uniform(k1)) < exploration_prob * boost:
                        mask = mask.at[idx].set(1.0)
                        if is_minority:
                            minority_protections += 1
                    k1, _ = jax.random.split(k1)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        if active_count < min_diversity:
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k2, jnp.array(inactive),
                                          shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)

        return mask, minority_protections, 0

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with GRN-quorum dynamics."""
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

        # === GRN REGULATION FROM QUORUM ===
        new_reg, new_formed, formations = self._update_regulation_from_quorum(
            state['regulation_matrix'], state['regulation_formed'],
            new_act_collective, new_agg_collective
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

        # === APPLY GRN REGULATION ===
        new_agg_aff = self._apply_regulation(
            new_act_aff, new_agg_aff, state['act_mask'], new_reg
        )

        # === PALETTE SELECTION WITH MINORITY ===
        new_act_mask, act_minority, _ = self._select_palette_with_minority(
            new_act_aff, new_act_collective, NUM_ACTIVATIONS,
            self.min_active_act, self.max_active_act, self.min_diversity_act,
            new_stagnation, generation, k1, prefer_indices=[4]
        )
        new_agg_mask, agg_minority, _ = self._select_palette_with_minority(
            new_agg_aff, new_agg_collective, NUM_AGGREGATIONS,
            self.min_active_agg, self.max_active_agg, self.min_diversity_agg,
            new_stagnation, generation, k2, prefer_indices=list(CORE_EXTREME_AGGS)
        )

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'regulation_matrix': new_reg,
            'regulation_formed': new_formed,
            'act_collective': new_act_collective,
            'agg_collective': new_agg_collective,
            'regulation_formations': state['regulation_formations'] + formations,
            'minority_protections': state['minority_protections'] + act_minority + agg_minority,
            'quorum_regulations': state['quorum_regulations'] + (1 if formations > 0 else 0),
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
            # GRN metrics (KEY)
            'sin_to_max_regulation': float(new_reg[4, 2]),
            'sin_to_min_regulation': float(new_reg[4, 3]),
            'n_regulations_formed': len(new_formed),
            'regulation_formations': new_state['regulation_formations'],
            # Quorum metrics
            'sin_collective': float(new_act_collective[4]),
            'max_collective': float(new_agg_collective[2]),
            'quorum_regulations': new_state['quorum_regulations'],
            # Minority metrics
            'minority_protections': new_state['minority_protections'],
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
            'n_regulations': len(state['regulation_formed']),
            'minority_protections': state['minority_protections'],
            'generation': state['generation'],
        }
