"""Strategy 70: Consolidation + Clonal Dual ("Memory Immune").

Hybrid combining Consolidation Window memory with Clonal Selection dynamics.

Key synergies:
1. Consolidation's working → LTM hierarchy stores validated knowledge
2. Clonal affinity learning identifies high-fitness functions
3. NOVEL: High-affinity functions get PRIORITY for LTM consolidation
4. Clonal expansion triggers faster working memory accumulation
5. LTM serves as "memory B-cells" - long-term immune memory

Biological basis:
- Immune system has both immediate response (working) and memory cells (LTM)
- High-affinity antibodies are preferentially stored as memory B-cells
- Sleep consolidation may reinforce "immune memory" formation
- Combines adaptive selection with structured memory

Expected: Functions that prove useful (high affinity) are robustly stored
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

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


class ConsolidationPhase:
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"


class ConsolidationClonalDualStrategy(PaletteEvolutionStrategy):
    """Clonal affinity guides memory consolidation priority.

    Functions with high affinity (matching the problem well) are
    prioritized during periodic consolidation windows for LTM transfer.
    """

    name = "consolidation_clonal_dual"
    description = "Clonal affinity guides memory consolidation priority"

    def __init__(
        self,
        # Consolidation timing
        consolidation_frequency: int = 10,
        consolidation_duration: int = 3,
        # Consolidation parameters
        replay_strength: float = 1.5,
        transfer_rate: float = 0.12,
        ltm_decay_rate: float = 0.02,
        # Affinity-guided consolidation (KEY SYNERGY)
        affinity_threshold_for_ltm: float = 0.5,  # Min affinity for LTM transfer
        affinity_transfer_bonus: float = 0.5,  # Bonus transfer rate for high-affinity
        # Clonal parameters
        affinity_lr: float = 0.12,
        affinity_decay: float = 0.97,
        affinity_threshold: float = 0.4,
        proliferation_rate: float = 0.25,
        expression_decay: float = 0.08,
        expression_min: float = 0.05,
        expression_max: float = 1.0,
        # Hypermutation (reduced during consolidation)
        active_hypermutation_rate: float = 0.08,
        consolidation_hypermutation_rate: float = 0.02,
        hypermutation_strength: float = 0.15,
        # Cross-domain
        cross_learning_rate: float = 0.10,
        cross_boost: float = 0.12,
        # Diversity
        min_diversity: int = 3,
        diversity_threshold: float = 0.15,
        # Palette selection
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        # Consolidation
        self.consolidation_frequency = consolidation_frequency
        self.consolidation_duration = consolidation_duration
        self.replay_strength = replay_strength
        self.transfer_rate = transfer_rate
        self.ltm_decay_rate = ltm_decay_rate
        self.affinity_threshold_for_ltm = affinity_threshold_for_ltm
        self.affinity_transfer_bonus = affinity_transfer_bonus

        # Clonal
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay
        self.affinity_threshold = affinity_threshold
        self.proliferation_rate = proliferation_rate
        self.expression_decay = expression_decay
        self.expression_min = expression_min
        self.expression_max = expression_max

        # Hypermutation
        self.active_hypermutation_rate = active_hypermutation_rate
        self.consolidation_hypermutation_rate = consolidation_hypermutation_rate
        self.hypermutation_strength = hypermutation_strength

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_boost = cross_boost

        # Diversity
        self.min_diversity = min_diversity
        self.diversity_threshold = diversity_threshold

        # Selection
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_consolidation_phase(self, generation: int, last_consol: int) -> Tuple[str, bool]:
        """Determine consolidation phase."""
        gens_since = generation - last_consol
        if gens_since < self.consolidation_duration:
            return ConsolidationPhase.CONSOLIDATING, False
        elif gens_since >= self.consolidation_frequency:
            return ConsolidationPhase.CONSOLIDATING, True
        return ConsolidationPhase.ACTIVE, False

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize with combined Consolidation + Clonal state."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize affinities and expressions
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        act_expressions = jnp.ones(NUM_ACTIVATIONS) * self.expression_min
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
                act_expressions = act_expressions.at[i].set(0.6)

        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        agg_expressions = jnp.ones(NUM_AGGREGATIONS) * self.expression_min
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)
                agg_expressions = agg_expressions.at[i].set(0.6)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinities': act_affinities,
            'act_expressions': act_expressions,
            'act_working': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'act_ltm': jnp.ones(NUM_ACTIVATIONS) * 0.3,

            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinities': agg_affinities,
            'agg_expressions': agg_expressions,
            'agg_working': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'agg_ltm': jnp.ones(NUM_AGGREGATIONS) * 0.3,

            # Cross-domain
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,

            # State
            'rng_key': jax.random.PRNGKey(seed + 707070),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'consolidation_phase': ConsolidationPhase.ACTIVE,
            'last_consolidation': -self.consolidation_frequency,
            'consolidations_completed': 0,

            # Stats
            'act_affinity_transfers': 0,
            'agg_affinity_transfers': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_affinities(
        self,
        affinities: jnp.ndarray,
        mask: jnp.ndarray,
        cross_boost: jnp.ndarray,
        fitness_delta: float,
        key: jax.random.PRNGKey,
        phase: str,
    ) -> Tuple[jnp.ndarray, int]:
        """Update affinities with phase-appropriate hypermutation."""
        k1, k2 = jax.random.split(key)

        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)

        # Decay and learn
        new_aff = self.affinity_decay * affinities
        contribution = active * fitness_delta / n_active
        new_aff = new_aff + self.affinity_lr * contribution
        new_aff = new_aff + self.cross_boost * cross_boost

        # Apply hypermutation (reduced during consolidation)
        mutation_rate = (
            self.consolidation_hypermutation_rate
            if phase == ConsolidationPhase.CONSOLIDATING
            else self.active_hypermutation_rate
        )

        mutation_probs = jax.random.uniform(k1, affinities.shape)
        mutation_amounts = jax.random.normal(k2, affinities.shape) * self.hypermutation_strength
        mutating = mutation_probs < mutation_rate

        new_aff = jnp.where(mutating, new_aff + mutation_amounts, new_aff)
        n_mutations = int(jnp.sum(mutating))

        return jnp.clip(new_aff, 0.0, 1.0), n_mutations

    def _update_working(
        self,
        working: jnp.ndarray,
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update working memory, boosted by affinity."""
        active = (mask > 0.5).astype(jnp.float32)

        # Base learning
        base_lr = 0.12
        # Affinity boost: high-affinity functions accumulate working memory faster
        affinity_boost = affinities * 0.5

        effective_lr = (base_lr + affinity_boost) * active
        delta = effective_lr * max(0, fitness_delta)

        return jnp.clip(working + delta, 0.0, 1.0)

    def _affinity_guided_consolidation(
        self,
        working: jnp.ndarray,
        ltm: jnp.ndarray,
        affinities: jnp.ndarray,
        mask: jnp.ndarray,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Consolidate with affinity-guided priority."""
        new_working = working.copy()
        new_ltm = ltm.copy()
        affinity_transfers = 0

        for i in range(n_funcs):
            aff = float(affinities[i])
            wm = float(working[i])

            # Replay based on working memory strength
            if wm >= 0.5:
                boost = self.replay_strength * (wm - 0.5)
                new_working = new_working.at[i].set(min(0.95, float(new_working[i]) + boost))

            # Transfer to LTM - AFFINITY GATED
            if aff >= self.affinity_threshold_for_ltm and float(mask[i]) > 0.5:
                diff = wm - float(ltm[i])
                if diff > 0:
                    # Higher affinity = faster transfer
                    effective_transfer = self.transfer_rate * (1.0 + self.affinity_transfer_bonus * aff)
                    transfer = effective_transfer * diff
                    new_ltm = new_ltm.at[i].set(min(0.95, float(new_ltm[i]) + transfer))
                    if transfer > 0.01:
                        affinity_transfers += 1

            # Decay inactive functions in LTM (slower decay for high-affinity)
            if float(mask[i]) < 0.5:
                effective_decay = self.ltm_decay_rate * (1.0 - 0.5 * aff)
                decay = effective_decay * (float(new_ltm[i]) - 0.3)
                new_ltm = new_ltm.at[i].set(max(0.1, float(new_ltm[i]) - decay))

        return new_working, new_ltm, affinity_transfers

    def _update_expressions(
        self,
        expressions: jnp.ndarray,
        affinities: jnp.ndarray,
        ltm: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update expressions combining affinity and LTM."""
        new_exp = expressions.copy()

        for i in range(len(expressions)):
            aff = float(affinities[i])
            ltm_score = float(ltm[i])

            # Combined score determines expansion
            combined = 0.6 * aff + 0.4 * ltm_score

            if combined >= self.affinity_threshold:
                # Clonal expansion
                new_exp = new_exp.at[i].set(expressions[i] * (1 + self.proliferation_rate))
            else:
                # Decay
                new_exp = new_exp.at[i].set(expressions[i] * (1 - self.expression_decay))

        return jnp.clip(new_exp, self.expression_min, self.expression_max)

    def _ensure_diversity(
        self,
        expressions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Ensure minimum diversity."""
        expressible = int(jnp.sum(expressions >= self.diversity_threshold))

        if expressible < self.min_diversity:
            n_to_boost = self.min_diversity - expressible
            sorted_indices = jnp.argsort(expressions)

            for i in range(min(n_to_boost, len(sorted_indices))):
                idx = int(sorted_indices[i])
                expressions = expressions.at[idx].set(
                    max(float(expressions[idx]), self.diversity_threshold)
                )

        return expressions

    def _select_palette(
        self,
        expressions: jnp.ndarray,
        palette_size: int,
        min_active: int,
    ) -> jnp.ndarray:
        """Select palette based on expression levels."""
        n_funcs = len(expressions)
        target_size = min(max(palette_size, min_active), n_funcs)

        top_indices = jnp.argsort(expressions)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        return mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with Consolidation + Clonal dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Determine consolidation phase
        phase, starting = self._get_consolidation_phase(generation, state['last_consolidation'])
        last_consol = generation if starting else state['last_consolidation']
        consol_count = state['consolidations_completed'] + (1 if starting else 0)

        # Compute cross-domain boosts
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)

        cross_delta = self.cross_learning_rate * fitness_delta * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        act_cross_boost = jnp.dot(new_cross, agg_active) / max(jnp.sum(agg_active), 1)
        agg_cross_boost = jnp.dot(new_cross.T, act_active) / max(jnp.sum(act_active), 1)

        # Update affinities (with phase-appropriate hypermutation)
        new_act_aff, act_mutations = self._update_affinities(
            state['act_affinities'], state['act_mask'], act_cross_boost, fitness_delta, k_act, phase
        )
        new_agg_aff, agg_mutations = self._update_affinities(
            state['agg_affinities'], state['agg_mask'], agg_cross_boost, fitness_delta, k_agg, phase
        )

        # Update working memory (affinity-boosted)
        new_act_working = self._update_working(
            state['act_working'], state['act_mask'], new_act_aff, fitness_delta
        )
        new_agg_working = self._update_working(
            state['agg_working'], state['agg_mask'], new_agg_aff, fitness_delta
        )

        # Consolidation with affinity-guided priority
        new_act_ltm, new_agg_ltm = state['act_ltm'], state['agg_ltm']
        act_aff_transfers, agg_aff_transfers = 0, 0

        if phase == ConsolidationPhase.CONSOLIDATING:
            new_act_working, new_act_ltm, act_aff_transfers = self._affinity_guided_consolidation(
                new_act_working, state['act_ltm'], new_act_aff, state['act_mask'], NUM_ACTIVATIONS
            )
            new_agg_working, new_agg_ltm, agg_aff_transfers = self._affinity_guided_consolidation(
                new_agg_working, state['agg_ltm'], new_agg_aff, state['agg_mask'], NUM_AGGREGATIONS
            )

        # Update expressions (combining affinity and LTM)
        new_act_exp = self._update_expressions(state['act_expressions'], new_act_aff, new_act_ltm)
        new_agg_exp = self._update_expressions(state['agg_expressions'], new_agg_aff, new_agg_ltm)

        # Ensure diversity
        new_act_exp = self._ensure_diversity(new_act_exp)
        new_agg_exp = self._ensure_diversity(new_agg_exp)

        # Select palettes
        new_act_mask = self._select_palette(new_act_exp, self.act_palette_size, self.min_active_act)
        new_agg_mask = self._select_palette(new_agg_exp, self.agg_palette_size, self.min_active_agg)

        new_state = {
            'act_mask': new_act_mask,
            'act_affinities': new_act_aff,
            'act_expressions': new_act_exp,
            'act_working': new_act_working,
            'act_ltm': new_act_ltm,
            'agg_mask': new_agg_mask,
            'agg_affinities': new_agg_aff,
            'agg_expressions': new_agg_exp,
            'agg_working': new_agg_working,
            'agg_ltm': new_agg_ltm,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'consolidation_phase': phase,
            'last_consolidation': last_consol,
            'consolidations_completed': consol_count,
            'act_affinity_transfers': state['act_affinity_transfers'] + act_aff_transfers,
            'agg_affinity_transfers': state['agg_affinity_transfers'] + agg_aff_transfers,
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
            'consolidation_phase': phase,
            'starting_consolidation': starting,
            # Clonal stats
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            # Memory stats
            'act_mean_ltm': float(jnp.mean(new_act_ltm)),
            'agg_mean_ltm': float(jnp.mean(new_agg_ltm)),
            'sin_ltm': float(new_act_ltm[4]),
            'sin_affinity': float(new_act_aff[4]),
            # Affinity-guided transfers
            'act_affinity_transfers': act_aff_transfers,
            'agg_affinity_transfers': agg_aff_transfers,
            # Sin status
            'has_sin': 4 in act_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with Consolidation + Clonal stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'consolidation_phase': state['consolidation_phase'],
            'consolidations_completed': state['consolidations_completed'],
            'generation': state['generation'],
            'sin_affinity': float(state['act_affinities'][4]),
            'sin_ltm': float(state['act_ltm'][4]),
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'act_mean_ltm': float(jnp.mean(state['act_ltm'])),
            'total_affinity_transfers': state['act_affinity_transfers'] + state['agg_affinity_transfers'],
        }
