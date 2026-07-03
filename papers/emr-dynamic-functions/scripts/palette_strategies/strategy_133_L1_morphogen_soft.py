"""Strategy 133-L1: Morphogen-Sin-Guaranteed SOFT.

Level 1 ablation: Remove hard mask forcing, keep affinity floors and initial seeding.

Test: Does morphogen gradient work WITHOUT permanent mask guarantees?

Extensions:
- Dynamic aggregation morphogen source discovery
- Cross-morphogen influence (act-agg pairs learn from morphogen gradients)
- Universal cross-domain morphogen effects
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
    CROSS_PAIR_CATEGORIES,
    SIN_IDX,
    ACT_CATEGORIES,
    AGG_CATEGORIES,
)


class MorphogenSoftL1Strategy(PaletteEvolutionStrategy):
    """Morphogen gradients with SOFT protection (no mask forcing).

    Tests if morphogen mechanism provides value
    when sin/extreme_aggs can actually be removed.
    """

    name = "morphogen_soft_L1"
    description = "L1: Morphogen WITHOUT mask forcing (can lose sin)"

    def __init__(
        self,
        # === MORPHOGEN PARAMETERS ===
        sin_source_permanent: bool = True,
        sin_source_strength: float = 2.5,
        extreme_agg_source_strength: float = 2.0,
        morphogen_diffusion_rate: float = 0.3,
        morphogen_decay_rate: float = 0.05,
        source_creation_threshold: float = 0.6,
        gradient_threshold: float = 0.4,
        min_source_strength: float = 0.5,
        # === PHASE PARAMETERS ===
        exploration_phase_until: int = 30,
        consolidation_phase_until: int = 60,
        exploration_strength_mult: float = 1.5,
        consolidation_strength_mult: float = 0.5,
        # === INITIAL SEEDING (KEPT) ===
        sin_always_in_initial_palette: bool = True,
        extreme_always_initial: bool = True,
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_learning_rate: float = 0.12,
        sin_extreme_coupling: float = 0.5,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 5,  # Increased from 4 to allow more agg discovery
        min_diversity_act: int = 3,
        min_diversity_agg: int = 3,  # Increased from 2 to explore more aggs
        # === INITIAL PALETTES ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
        # === DYNAMIC AGG MORPHOGEN SOURCES ===
        enable_agg_morphogen_discovery: bool = True,
        agg_source_threshold: float = 0.45,       # Affinity threshold for becoming source (lowered from 0.65)
        agg_source_contribution_threshold: float = 2,  # Generations contributing before source (lowered from 3)
        max_agg_sources: int = 5,                 # Max agg morphogen sources (increased from 4)
        dynamic_source_strength: float = 1.5,    # Strength of dynamically discovered sources
        # === CROSS-MORPHOGEN INFLUENCE ===
        enable_cross_morphogen: bool = True,
        cross_morphogen_influence: float = 0.15,  # How much morphogen affects cross-affinity
        morphogen_attraction_rate: float = 0.10,  # Rate at which morphogen attracts pairs
        category_morphogen_multipliers: Dict[str, float] = None,
    ):
        """Initialize Morphogen Soft L1 strategy."""
        # Morphogen
        self.sin_source_permanent = sin_source_permanent
        self.sin_source_strength = sin_source_strength
        self.extreme_agg_source_strength = extreme_agg_source_strength
        self.morphogen_diffusion_rate = morphogen_diffusion_rate
        self.morphogen_decay_rate = morphogen_decay_rate
        self.source_creation_threshold = source_creation_threshold
        self.gradient_threshold = gradient_threshold
        self.min_source_strength = min_source_strength

        # Phases
        self.exploration_phase_until = exploration_phase_until
        self.consolidation_phase_until = consolidation_phase_until
        self.exploration_strength_mult = exploration_strength_mult
        self.consolidation_strength_mult = consolidation_strength_mult

        # Initial seeding (KEPT for L1)
        self.sin_always_in_initial_palette = sin_always_in_initial_palette
        self.extreme_always_initial = extreme_always_initial

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_coupling = sin_extreme_coupling

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial palettes
        default_act = list(DEFAULT_PALETTE_INDICES)
        if self.sin_always_in_initial_palette and 4 not in default_act:
            default_act.append(4)

        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        if self.extreme_always_initial:
            for agg in CORE_EXTREME_AGGS:
                if agg not in default_agg:
                    default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

        # Dynamic aggregation morphogen sources
        self.enable_agg_morphogen_discovery = enable_agg_morphogen_discovery
        self.agg_source_threshold = agg_source_threshold
        self.agg_source_contribution_threshold = agg_source_contribution_threshold
        self.max_agg_sources = max_agg_sources
        self.dynamic_source_strength = dynamic_source_strength

        # Cross-morphogen influence
        self.enable_cross_morphogen = enable_cross_morphogen
        self.cross_morphogen_influence = cross_morphogen_influence
        self.morphogen_attraction_rate = morphogen_attraction_rate
        self.category_morphogen_multipliers = category_morphogen_multipliers or {
            'known_synergistic': 1.5,
            'oscillatory_extreme': 1.3,
            'smooth_averaging': 1.0,
            'rectified_extreme': 1.2,
            'periodic_averaging': 1.1,
        }

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with morphogen fields."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # Initial seeding (KEPT for L1)
        if self.sin_always_in_initial_palette and 4 not in initial_act:
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
        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(0.75)

        # Morphogen fields
        act_morphogen = jnp.zeros(NUM_ACTIVATIONS)
        agg_morphogen = jnp.zeros(NUM_AGGREGATIONS)

        act_morphogen = act_morphogen.at[4].set(self.sin_source_strength)
        for agg in CORE_EXTREME_AGGS:
            agg_morphogen = agg_morphogen.at[agg].set(self.extreme_agg_source_strength)

        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, agg].set(0.7)

        # Cross-morphogen influence matrix
        cross_morphogen = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))
        # Initialize with known synergistic pairs
        for agg in CORE_EXTREME_AGGS:
            cross_morphogen = cross_morphogen.at[SIN_IDX, agg].set(0.5)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_morphogen': act_morphogen,
            'agg_morphogen': agg_morphogen,
            'cross_affinity': cross_affinity,
            'rng_key': jax.random.PRNGKey(seed + 1331000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Dynamic aggregation morphogen sources
            'agg_sources': set(CORE_EXTREME_AGGS),  # Currently active sources
            'agg_contribution_counts': jnp.zeros(NUM_AGGREGATIONS),  # Gens as active contributor
            'discovered_sources': 0,
            # Cross-morphogen influence
            'cross_morphogen': cross_morphogen,
            'cross_morphogen_updates': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _get_phase(self, generation: int) -> str:
        if generation < self.exploration_phase_until:
            return 'exploration'
        elif generation < self.consolidation_phase_until:
            return 'consolidation'
        else:
            return 'mature'

    def _update_morphogen_fields(
        self,
        act_morphogen: jnp.ndarray,
        agg_morphogen: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improved: bool,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update morphogen fields."""
        new_act = act_morphogen * (1.0 - self.morphogen_decay_rate)
        new_agg = agg_morphogen * (1.0 - self.morphogen_decay_rate)

        # Sin source maintained at high level
        if self.sin_source_permanent:
            new_act = new_act.at[4].set(max(
                float(new_act[4]),
                self.sin_source_strength * 0.9
            ))

        for agg in CORE_EXTREME_AGGS:
            new_agg = new_agg.at[agg].set(max(
                float(new_agg[agg]),
                self.extreme_agg_source_strength * 0.8
            ))

        if phase == 'exploration':
            strength_mult = self.exploration_strength_mult
        elif phase == 'consolidation':
            strength_mult = self.consolidation_strength_mult
        else:
            strength_mult = 1.0

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                contrib = 0.1 * strength_mult
                if i == 4:
                    contrib = 0.3 * strength_mult
                new_act = new_act.at[i].set(min(3.0, new_act[i] + contrib))

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                contrib = 0.15 * strength_mult
                if j in CORE_EXTREME_AGGS:
                    contrib = 0.25 * strength_mult
                new_agg = new_agg.at[j].set(min(3.0, new_agg[j] + contrib))

        # Diffusion
        new_act_diffused = new_act.copy()
        for i in range(1, NUM_ACTIVATIONS - 1):
            diffusion = self.morphogen_diffusion_rate * (
                new_act[i - 1] + new_act[i + 1] - 2 * new_act[i]
            ) * 0.5
            new_act_diffused = new_act_diffused.at[i].set(new_act[i] + diffusion)

        new_agg_diffused = new_agg.copy()
        for j in range(1, NUM_AGGREGATIONS - 1):
            diffusion = self.morphogen_diffusion_rate * (
                new_agg[j - 1] + new_agg[j + 1] - 2 * new_agg[j]
            ) * 0.5
            new_agg_diffused = new_agg_diffused.at[j].set(new_agg[j] + diffusion)

        return jnp.clip(new_act_diffused, 0, 3.0), jnp.clip(new_agg_diffused, 0, 3.0)

    def _discover_agg_morphogen_sources(
        self,
        agg_mask: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        agg_morphogen: jnp.ndarray,
        agg_sources: set,
        agg_contribution_counts: jnp.ndarray,
        improved: bool,
    ) -> Tuple[jnp.ndarray, set, jnp.ndarray, int]:
        """Discover new aggregation morphogen sources based on performance.

        Allow any aggregation to become a morphogen source based on
        sustained contribution and high affinity.
        """
        if not self.enable_agg_morphogen_discovery:
            return agg_morphogen, agg_sources, agg_contribution_counts, 0

        new_morphogen = agg_morphogen.copy()
        new_sources = set(agg_sources)
        new_counts = agg_contribution_counts.copy()
        discovered = 0

        # Update contribution counts for active aggregations
        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                if improved:
                    new_counts = new_counts.at[j].set(new_counts[j] + 1)
                else:
                    new_counts = new_counts.at[j].set(max(0, new_counts[j] - 0.5))
            else:
                new_counts = new_counts.at[j].set(max(0, new_counts[j] - 1))

        # Check for new sources
        if len(new_sources) < self.max_agg_sources:
            for j in range(NUM_AGGREGATIONS):
                if j in new_sources:
                    continue

                # Check if qualifies for source status
                affinity_ok = float(agg_affinities[j]) >= self.agg_source_threshold
                contribution_ok = float(new_counts[j]) >= self.agg_source_contribution_threshold
                is_active = agg_mask[j] > 0.5

                if affinity_ok and contribution_ok and is_active:
                    new_sources.add(j)
                    new_morphogen = new_morphogen.at[j].set(
                        max(float(new_morphogen[j]), self.dynamic_source_strength)
                    )
                    discovered += 1

                    if len(new_sources) >= self.max_agg_sources:
                        break

        # Maintain source strength for discovered sources
        for j in new_sources:
            current = float(new_morphogen[j])
            if j in CORE_EXTREME_AGGS:
                min_strength = self.extreme_agg_source_strength * 0.8
            else:
                min_strength = self.dynamic_source_strength * 0.7
            new_morphogen = new_morphogen.at[j].set(max(current, min_strength))

        return new_morphogen, new_sources, new_counts, discovered

    def _apply_cross_morphogen_influence(
        self,
        act_morphogen: jnp.ndarray,
        agg_morphogen: jnp.ndarray,
        cross_morphogen: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Let morphogen gradients influence cross-domain affinities.

        High activation morphogen attracts aggregations with high cross-affinity.
        Reciprocal influence creates learned patterns beyond sin-extreme.
        """
        if not self.enable_cross_morphogen:
            return cross_morphogen, cross_affinity, 0

        new_cross_morphogen = cross_morphogen * 0.95  # Decay
        new_cross_affinity = cross_affinity.copy()
        updates = 0

        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        # Cross-morphogen influence based on individual morphogen strengths
                        act_morph = float(act_morphogen[i])
                        agg_morph = float(agg_morphogen[j])

                        # Pairs with high combined morphogen get boosted
                        combined_morph = (act_morph + agg_morph) / 2
                        if combined_morph > 0.5:
                            # Get category multiplier
                            multiplier = 1.0
                            for category, pairs in CROSS_PAIR_CATEGORIES.items():
                                if (i, j) in pairs:
                                    multiplier = self.category_morphogen_multipliers.get(category, 1.0)
                                    break

                            # Update cross-morphogen
                            delta = self.cross_morphogen_influence * combined_morph * multiplier
                            new_cross_morphogen = new_cross_morphogen.at[i, j].set(
                                min(1.0, new_cross_morphogen[i, j] + delta)
                            )

                            # Cross-morphogen influences cross-affinity
                            affinity_delta = self.morphogen_attraction_rate * float(new_cross_morphogen[i, j])
                            new_cross_affinity = new_cross_affinity.at[i, j].set(
                                min(1.0, new_cross_affinity[i, j] + affinity_delta)
                            )
                            updates += 1

        # Ensure minimum for known synergistic pairs
        for agg in CORE_EXTREME_AGGS:
            new_cross_morphogen = new_cross_morphogen.at[SIN_IDX, agg].set(
                max(0.3, float(new_cross_morphogen[SIN_IDX, agg]))
            )

        return new_cross_morphogen, new_cross_affinity, updates

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with morphogen dynamics (L1: no mask forcing)."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        phase = self._get_phase(generation)

        # Update morphogen fields
        new_act_morphogen, new_agg_morphogen = self._update_morphogen_fields(
            state['act_morphogen'], state['agg_morphogen'],
            state['act_mask'], state['agg_mask'],
            improved, phase
        )

        # === Dynamic aggregation morphogen source discovery ===
        agg_sources = set(state.get('agg_sources', CORE_EXTREME_AGGS))
        agg_contribution_counts = state.get('agg_contribution_counts', jnp.zeros(NUM_AGGREGATIONS))

        new_agg_morphogen, new_agg_sources, new_agg_counts, discovered_sources = self._discover_agg_morphogen_sources(
            state['agg_mask'],
            state['agg_affinities'],
            new_agg_morphogen,
            agg_sources,
            agg_contribution_counts,
            improved,
        )

        # === Cross-morphogen influence ===
        cross_morphogen = state.get('cross_morphogen', jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS)))

        new_cross_morphogen, cross_affinity_updated, cross_morph_updates = self._apply_cross_morphogen_influence(
            new_act_morphogen,
            new_agg_morphogen,
            cross_morphogen,
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Update affinities
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay

        if fitness_delta > 0:
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    lr = self.agg_affinity_lr * (1 + float(new_agg_morphogen[j]) * 0.3)
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus += 0.6
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + lr * fitness_delta * bonus)
                    )

            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    lr = self.act_affinity_lr * (1 + float(new_act_morphogen[i]) * 0.3)
                    bonus = 1.0
                    if i == 4:
                        bonus += 0.5
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + lr * fitness_delta * bonus)
                    )

        # AFFINITY FLOORS (KEPT for L1)
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Select activation palette
        score = new_act_aff + new_act_morphogen * 0.4
        score = score.at[4].set(score[4] + 1.0)

        for i in range(NUM_ACTIVATIONS):
            cross_influence = 0.0
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    cross_influence = max(cross_influence, float(state['cross_affinity'][i, j]))
            score = score.at[i].set(score[i] + cross_influence * 0.25)

        target_size = min(max(self.min_diversity_act, self.min_active_act), self.max_active_act)
        top_indices = jnp.argsort(score)[-target_size:]

        new_act_mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            new_act_mask = new_act_mask.at[int(idx)].set(1.0)

        # L1 CRITICAL CHANGE: NO HARD MASK FORCING for sin
        # Original code REMOVED:
        # new_act_mask = new_act_mask.at[4].set(1.0)

        # Select agg palette
        agg_score = new_agg_aff + new_agg_morphogen * 0.4
        for agg in CORE_EXTREME_AGGS:
            agg_score = agg_score.at[agg].set(agg_score[agg] + 0.5)

        target_agg_size = min(max(self.min_diversity_agg, self.min_active_agg), self.max_active_agg)
        top_agg_indices = jnp.argsort(agg_score)[-target_agg_size:]

        new_agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for idx in top_agg_indices:
            new_agg_mask = new_agg_mask.at[int(idx)].set(1.0)

        # L1 CRITICAL CHANGE: NO HARD MASK FORCING for extreme aggs
        # Original code REMOVED:
        # has_extreme = any(new_agg_mask[agg] > 0.5 for agg in CORE_EXTREME_AGGS)
        # if not has_extreme:
        #     new_agg_mask = new_agg_mask.at[2].set(1.0)

        # Ensure minimums
        if int(jnp.sum(new_act_mask)) < self.min_active_act:
            new_act_mask = new_act_mask.at[0].set(1.0)
        if int(jnp.sum(new_agg_mask)) < self.min_active_agg:
            new_agg_mask = new_agg_mask.at[0].set(1.0)

        # Update cross-domain affinity (start from the updated value)
        new_cross = cross_affinity_updated.copy()
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if state['act_mask'][i] > 0.5 and state['agg_mask'][j] > 0.5:
                        delta = self.cross_learning_rate * fitness_delta
                        if i == SIN_IDX and j in CORE_EXTREME_AGGS:
                            delta *= (1 + self.sin_extreme_coupling)
                        new_cross = new_cross.at[i, j].set(
                            min(1.0, new_cross[i, j] + delta)
                        )

        new_state = {
            **state,
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_morphogen': new_act_morphogen,
            'agg_morphogen': new_agg_morphogen,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'fitness_history': state['fitness_history'] + [best_fitness],
            # Dynamic aggregation morphogen sources
            'agg_sources': new_agg_sources,
            'agg_contribution_counts': new_agg_counts,
            'discovered_sources': state.get('discovered_sources', 0) + discovered_sources,
            # Cross-morphogen influence
            'cross_morphogen': new_cross_morphogen,
            'cross_morphogen_updates': state.get('cross_morphogen_updates', 0) + cross_morph_updates,
        }

        # Count high cross-morphogen pairs for metrics
        high_cross_morph_pairs = 0
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if new_cross_morphogen[i, j] > 0.5:
                    high_cross_morph_pairs += 1

        # Count non-sin-extreme high cross-morphogen pairs
        non_sin_extreme_high_morph = 0
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                if new_cross_morphogen[i, j] > 0.5:
                    if i != SIN_IDX or j not in CORE_EXTREME_AGGS:
                        non_sin_extreme_high_morph += 1

        metrics = {
            'phase': phase,
            'sin_morphogen': float(new_act_morphogen[SIN_IDX]),
            'max_morphogen': float(new_agg_morphogen[2]),
            # metrics
            'discovered_sources': discovered_sources,
            'total_agg_sources': len(new_agg_sources),
            'cross_morph_updates': cross_morph_updates,
            'high_cross_morph_pairs': high_cross_morph_pairs,
            'non_sin_extreme_high_morph': non_sin_extreme_high_morph,
        }

        return new_state, metrics
