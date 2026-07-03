"""Strategy 68: Circadian + Clonal Dual ("Oscillatory Immune").

Hybrid combining Circadian rhythm oscillations with Clonal selection dynamics.

Key synergies:
1. Master clock (period=20) provides global temporal structure
2. Clonal affinity learning determines function fitness matching
3. NOVEL: Clock gates hypermutation - explore during "night" (low activity), exploit during "day"
4. Phase entrainment weighted by affinity - high-affinity functions lock to clock faster

Biological basis:
- Immune system activity follows circadian patterns
- B-cell proliferation peaks at certain times of day
- Hypermutation may be more active during rest phases
- Combines rhythmic exploration with adaptive selection

Expected: Natural exploration/exploitation cycles guided by immune-like selection
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


class CircadianClonalDualStrategy(PaletteEvolutionStrategy):
    """Circadian oscillations gate clonal selection and hypermutation.

    The master clock determines when to explore (night = high hypermutation)
    vs exploit (day = affinity-based selection). Functions with high affinity
    entrain their phases to the clock faster.
    """

    name = "circadian_clonal_dual"
    description = "Circadian clock gates immune-inspired exploration/exploitation"

    def __init__(
        self,
        # Circadian parameters
        circadian_period: int = 20,
        night_phase_fraction: float = 0.3,  # Fraction of cycle that's "night"
        # Clonal parameters
        affinity_lr: float = 0.12,
        affinity_decay: float = 0.97,
        affinity_threshold: float = 0.4,
        expression_min: float = 0.05,
        expression_max: float = 1.0,
        proliferation_rate: float = 0.25,
        expression_decay: float = 0.08,
        # Hypermutation (gated by clock)
        day_hypermutation_rate: float = 0.02,  # Low during "day"
        night_hypermutation_rate: float = 0.15,  # High during "night"
        hypermutation_strength: float = 0.2,
        # Phase entrainment
        base_entrainment: float = 0.15,
        affinity_entrainment_bonus: float = 0.2,  # High-affinity = faster entrainment
        phase_noise: float = 0.08,
        # Cross-domain
        cross_learning_rate: float = 0.08,
        cross_boost: float = 0.12,
        # Diversity
        min_diversity_act: int = 3,
        min_diversity_agg: int = 2,
        diversity_threshold: float = 0.15,
        # Palette selection
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        # Circadian
        self.circadian_period = circadian_period
        self.night_phase_fraction = night_phase_fraction

        # Clonal
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay
        self.affinity_threshold = affinity_threshold
        self.expression_min = expression_min
        self.expression_max = expression_max
        self.proliferation_rate = proliferation_rate
        self.expression_decay = expression_decay

        # Hypermutation
        self.day_hypermutation_rate = day_hypermutation_rate
        self.night_hypermutation_rate = night_hypermutation_rate
        self.hypermutation_strength = hypermutation_strength

        # Entrainment
        self.base_entrainment = base_entrainment
        self.affinity_entrainment_bonus = affinity_entrainment_bonus
        self.phase_noise = phase_noise

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_boost = cross_boost

        # Diversity
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg
        self.diversity_threshold = diversity_threshold

        # Selection
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _is_night_phase(self, clock_phase: float) -> bool:
        """Check if current clock phase is in 'night' (exploration) window."""
        # Night is when phase is in the latter portion of the cycle
        normalized = (clock_phase % (2 * jnp.pi)) / (2 * jnp.pi)
        return normalized >= (1.0 - self.night_phase_fraction)

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize with combined Circadian + Clonal state."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        key = jax.random.PRNGKey(seed + 686868)
        key, k1, k2 = jax.random.split(key, 3)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize phases randomly, but initial functions start at phase 0
        act_phases = jax.random.uniform(k1, (NUM_ACTIVATIONS,)) * 2 * jnp.pi
        agg_phases = jax.random.uniform(k2, (NUM_AGGREGATIONS,)) * 2 * jnp.pi

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_phases = act_phases.at[i].set(0.0)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_phases = agg_phases.at[i].set(0.0)

        # Initialize affinities and expressions (from Clonal)
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
            'act_phases': act_phases,
            'act_affinities': act_affinities,
            'act_expressions': act_expressions,

            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_phases': agg_phases,
            'agg_affinities': agg_affinities,
            'agg_expressions': agg_expressions,

            # Cross-domain
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,

            # Clock state
            'clock_phase': 0.0,
            'cycles_completed': 0,
            'is_night': False,

            # General state
            'rng_key': key,
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,

            # Stats
            'total_night_mutations': 0,
            'total_day_mutations': 0,
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
    ) -> jnp.ndarray:
        """Update affinities based on fitness feedback."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1.0)

        # Decay and learn
        new_aff = self.affinity_decay * affinities
        contribution = active * fitness_delta / n_active
        new_aff = new_aff + self.affinity_lr * contribution

        # Add cross-domain boost
        new_aff = new_aff + self.cross_boost * cross_boost

        return jnp.clip(new_aff, 0.0, 1.0)

    def _apply_hypermutation(
        self,
        affinities: jnp.ndarray,
        is_night: bool,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int]:
        """Apply clock-gated hypermutation."""
        k1, k2 = jax.random.split(key)

        # Use appropriate mutation rate based on clock phase
        mutation_rate = (
            self.night_hypermutation_rate if is_night
            else self.day_hypermutation_rate
        )

        mutation_probs = jax.random.uniform(k1, affinities.shape)
        mutation_amounts = jax.random.normal(k2, affinities.shape) * self.hypermutation_strength
        mutating = mutation_probs < mutation_rate

        new_aff = jnp.where(mutating, affinities + mutation_amounts, affinities)
        n_mutations = int(jnp.sum(mutating))

        return jnp.clip(new_aff, 0.0, 1.0), n_mutations

    def _update_expressions(
        self,
        expressions: jnp.ndarray,
        affinities: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update expressions based on affinities (clonal expansion/decay)."""
        new_exp = expressions.copy()

        for i in range(len(expressions)):
            if affinities[i] >= self.affinity_threshold:
                # Clonal expansion
                new_exp = new_exp.at[i].set(expressions[i] * (1 + self.proliferation_rate))
            else:
                # Decay
                new_exp = new_exp.at[i].set(expressions[i] * (1 - self.expression_decay))

        return jnp.clip(new_exp, self.expression_min, self.expression_max)

    def _entrain_phases(
        self,
        phases: jnp.ndarray,
        affinities: jnp.ndarray,
        clock_phase: float,
        key: jax.random.PRNGKey,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Entrain function phases to clock, weighted by affinity."""
        new_phases = phases.copy()
        noise = jax.random.normal(key, (n_funcs,)) * self.phase_noise

        for i in range(n_funcs):
            aff = float(affinities[i])
            phase_diff = clock_phase - float(phases[i])

            # Higher affinity = faster entrainment
            entrainment = self.base_entrainment + self.affinity_entrainment_bonus * aff
            adjustment = entrainment * jnp.sin(phase_diff)

            new_phases = new_phases.at[i].set(float(phases[i]) + adjustment)

        new_phases = new_phases + noise
        return jnp.mod(new_phases, 2 * jnp.pi)

    def _ensure_diversity(
        self,
        expressions: jnp.ndarray,
        min_diverse: int,
    ) -> jnp.ndarray:
        """Ensure minimum diversity by boosting low-expression functions."""
        expressible = int(jnp.sum(expressions >= self.diversity_threshold))

        if expressible < min_diverse:
            n_to_boost = min_diverse - expressible
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
        """Update with combined Circadian + Clonal dynamics."""
        key, k_act_m, k_agg_m, k_act_p, k_agg_p = jax.random.split(state['rng_key'], 5)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Advance master clock
        phase_increment = 2 * jnp.pi / self.circadian_period
        new_clock_phase = state['clock_phase'] + phase_increment
        cycles_completed = state['cycles_completed']
        if new_clock_phase >= 2 * jnp.pi:
            new_clock_phase = new_clock_phase % (2 * jnp.pi)
            cycles_completed += 1

        is_night = self._is_night_phase(new_clock_phase)

        # Compute cross-domain boosts
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)

        cross_delta = self.cross_learning_rate * fitness_delta * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        act_cross_boost = jnp.dot(new_cross, agg_active) / max(jnp.sum(agg_active), 1)
        agg_cross_boost = jnp.dot(new_cross.T, act_active) / max(jnp.sum(act_active), 1)

        # Update affinities
        new_act_aff = self._update_affinities(
            state['act_affinities'], state['act_mask'], act_cross_boost, fitness_delta
        )
        new_agg_aff = self._update_affinities(
            state['agg_affinities'], state['agg_mask'], agg_cross_boost, fitness_delta
        )

        # Apply clock-gated hypermutation
        new_act_aff, act_mutations = self._apply_hypermutation(new_act_aff, is_night, k_act_m)
        new_agg_aff, agg_mutations = self._apply_hypermutation(new_agg_aff, is_night, k_agg_m)

        # Update expressions (clonal dynamics)
        new_act_exp = self._update_expressions(state['act_expressions'], new_act_aff)
        new_agg_exp = self._update_expressions(state['agg_expressions'], new_agg_aff)

        # Entrain phases to clock (affinity-weighted)
        new_act_phases = self._entrain_phases(
            state['act_phases'], new_act_aff, new_clock_phase, k_act_p, NUM_ACTIVATIONS
        )
        new_agg_phases = self._entrain_phases(
            state['agg_phases'], new_agg_aff, new_clock_phase, k_agg_p, NUM_AGGREGATIONS
        )

        # Ensure diversity
        new_act_exp = self._ensure_diversity(new_act_exp, self.min_diversity_act)
        new_agg_exp = self._ensure_diversity(new_agg_exp, self.min_diversity_agg)

        # Select palettes
        new_act_mask = self._select_palette(new_act_exp, self.act_palette_size, self.min_active_act)
        new_agg_mask = self._select_palette(new_agg_exp, self.agg_palette_size, self.min_active_agg)

        # Track stats
        night_mutations = act_mutations + agg_mutations if is_night else 0
        day_mutations = 0 if is_night else act_mutations + agg_mutations

        new_state = {
            'act_mask': new_act_mask,
            'act_phases': new_act_phases,
            'act_affinities': new_act_aff,
            'act_expressions': new_act_exp,
            'agg_mask': new_agg_mask,
            'agg_phases': new_agg_phases,
            'agg_affinities': new_agg_aff,
            'agg_expressions': new_agg_exp,
            'cross_affinity': new_cross,
            'clock_phase': float(new_clock_phase),
            'cycles_completed': cycles_completed,
            'is_night': is_night,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'total_night_mutations': state['total_night_mutations'] + night_mutations,
            'total_day_mutations': state['total_day_mutations'] + day_mutations,
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
            # Clock stats
            'clock_phase': float(new_clock_phase),
            'is_night': is_night,
            'cycles_completed': cycles_completed,
            # Clonal stats
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'act_mean_expression': float(jnp.mean(new_act_exp)),
            'agg_mean_expression': float(jnp.mean(new_agg_exp)),
            # Mutation stats
            'mutations_this_gen': act_mutations + agg_mutations,
            'total_night_mutations': new_state['total_night_mutations'],
            'total_day_mutations': new_state['total_day_mutations'],
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_affinity': float(new_act_aff[4]),
            'sin_expression': float(new_act_exp[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with Circadian + Clonal stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'clock_phase': state['clock_phase'],
            'is_night': state['is_night'],
            'cycles_completed': state['cycles_completed'],
            'generation': state['generation'],
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinities'])),
            'sin_affinity': float(state['act_affinities'][4]),
            'total_night_mutations': state['total_night_mutations'],
            'total_day_mutations': state['total_day_mutations'],
        }
