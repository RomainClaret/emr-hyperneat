"""Strategy 133: Morphogen-Sin-Guaranteed Dual.

Fixes the 0% sin retention bug in morphogen strategies by guaranteeing sin
in the initial palette AND giving it a permanent morphogen source.

Key Fix:
- Sin ALWAYS starts in the initial palette (not discovered - guaranteed)
- Sin has permanent morphogen source that NEVER decays
- Extreme aggs (max/min) also get strong, persistent morphogen sources

morphogen_critical_period_dual converged to
non-sin solutions because sin had no inherent advantage in the morphogen
field. Evolution found equally-valid solutions without sin.

Solution: Make sin a permanent "morphogen attractor" that never loses its
source strength, ensuring evolution retains it across all tasks.
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class MorphogenSinGuaranteedDualStrategy(PaletteEvolutionStrategy):
    """Morphogen gradients with guaranteed sin and extreme agg sources.

    Ensures sin retention by ensuring sin
    is always present and has a permanent morphogen source.

    Critical innovation: Sin's morphogen source NEVER decays, creating
    a permanent attractor in the morphogen field. Extreme aggregations
    (max/min) also get strong persistent sources.
    """

    name = "morphogen_sin_guaranteed_dual"
    description = "Dual: Morphogen gradients with guaranteed sin source"

    def __init__(
        self,
        # === GUARANTEED SOURCES (NEW - CRITICAL FIX) ===
        sin_always_in_initial_palette: bool = True,
        sin_source_permanent: bool = True,
        sin_source_strength: float = 2.5,
        extreme_agg_source_strength: float = 2.0,
        # === MORPHOGEN PARAMETERS ===
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
        # === AFFINITY PARAMETERS ===
        act_affinity_lr: float = 0.10,
        agg_affinity_lr: float = 0.15,
        affinity_decay: float = 0.98,
        cross_learning_rate: float = 0.12,
        sin_extreme_coupling: float = 0.5,
        # === TAGGING PARAMETERS ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.40,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.85,
        extreme_tag_boost: float = 1.5,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 3,
        min_diversity_agg: int = 2,
        # === INITIAL PALETTES ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Morphogen-Sin-Guaranteed strategy."""
        # GUARANTEED SOURCES (CRITICAL FIX)
        self.sin_always_in_initial_palette = sin_always_in_initial_palette
        self.sin_source_permanent = sin_source_permanent
        self.sin_source_strength = sin_source_strength
        self.extreme_agg_source_strength = extreme_agg_source_strength

        # Morphogen
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

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_coupling = sin_extreme_coupling

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial - ALWAYS include sin if guaranteed
        default_act = list(DEFAULT_PALETTE_INDICES)
        if self.sin_always_in_initial_palette and 4 not in default_act:
            default_act.append(4)

        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        # Also ensure extreme aggs in initial palette
        for agg in CORE_EXTREME_AGGS:
            if agg not in default_agg:
                default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with guaranteed sin source."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        # CRITICAL: Ensure sin is ALWAYS in initial palette
        if self.sin_always_in_initial_palette and 4 not in initial_act:
            initial_act = list(initial_act) + [4]

        # Ensure extreme aggs in initial palette
        initial_agg = list(initial_agg)
        for agg in CORE_EXTREME_AGGS:
            if agg not in initial_agg:
                initial_agg.append(agg)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities - sin starts with high affinity
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)

        # Sin gets extra high affinity
        act_affinities = act_affinities.at[4].set(0.8)

        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)

        # Extreme aggs get high affinity
        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(0.75)

        # Morphogen fields - create 2D fields for spatial gradients
        # Activations in 1D, aggregations in 1D (separate fields)
        act_morphogen = jnp.zeros(NUM_ACTIVATIONS)
        agg_morphogen = jnp.zeros(NUM_AGGREGATIONS)

        # CRITICAL: Set PERMANENT sin source
        act_morphogen = act_morphogen.at[4].set(self.sin_source_strength)

        # Set strong extreme agg sources
        for agg in CORE_EXTREME_AGGS:
            agg_morphogen = agg_morphogen.at[agg].set(self.extreme_agg_source_strength)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        # Pre-bias sin-extreme cross-domain affinity
        for agg in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, agg].set(0.7)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Morphogen fields (CRITICAL)
            'act_morphogen': act_morphogen,
            'agg_morphogen': agg_morphogen,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'capture_events': 0,
            'diversity_rescues': 0,
            'sin_source_strength_history': [],
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1330000),
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

    def _get_phase(self, generation: int) -> str:
        """Determine current developmental phase."""
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
        """Update morphogen fields with GUARANTEED sin source."""
        # Decay existing morphogen (but NOT sin if permanent)
        new_act = act_morphogen * (1.0 - self.morphogen_decay_rate)
        new_agg = agg_morphogen * (1.0 - self.morphogen_decay_rate)

        # CRITICAL: Sin source is PERMANENT - never decay below threshold
        if self.sin_source_permanent:
            new_act = new_act.at[4].set(max(
                float(new_act[4]),
                self.sin_source_strength * 0.9  # Never drops below 90%
            ))

        # Extreme agg sources also maintained at high level
        for agg in CORE_EXTREME_AGGS:
            new_agg = new_agg.at[agg].set(max(
                float(new_agg[agg]),
                self.extreme_agg_source_strength * 0.8
            ))

        # Phase-dependent source strength multiplier
        if phase == 'exploration':
            strength_mult = self.exploration_strength_mult
        elif phase == 'consolidation':
            strength_mult = self.consolidation_strength_mult
        else:
            strength_mult = 1.0

        # Active functions contribute to morphogen field
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                contrib = 0.1 * strength_mult
                if i == 4:  # Sin always strong
                    contrib = 0.3 * strength_mult
                new_act = new_act.at[i].set(
                    min(3.0, new_act[i] + contrib)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                contrib = 0.15 * strength_mult
                if j in CORE_EXTREME_AGGS:
                    contrib = 0.25 * strength_mult
                new_agg = new_agg.at[j].set(
                    min(3.0, new_agg[j] + contrib)
                )

        # Diffusion - spread morphogen to neighbors
        new_act_diffused = new_act.copy()
        for i in range(1, NUM_ACTIVATIONS - 1):
            diffusion = self.morphogen_diffusion_rate * (
                new_act[i - 1] + new_act[i + 1] - 2 * new_act[i]
            ) * 0.5
            new_act_diffused = new_act_diffused.at[i].set(
                new_act[i] + diffusion
            )

        new_agg_diffused = new_agg.copy()
        for j in range(1, NUM_AGGREGATIONS - 1):
            diffusion = self.morphogen_diffusion_rate * (
                new_agg[j - 1] + new_agg[j + 1] - 2 * new_agg[j]
            ) * 0.5
            new_agg_diffused = new_agg_diffused.at[j].set(
                new_agg[j] + diffusion
            )

        return jnp.clip(new_act_diffused, 0, 3.0), jnp.clip(new_agg_diffused, 0, 3.0)

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_morphogen: jnp.ndarray,
        agg_morphogen: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with morphogen field boost."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                # Morphogen boost
                morph_boost = float(agg_morphogen[j]) * 0.2
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.35 + morph_boost)
                )

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:  # Sin
                    tag_strength *= self.extreme_tag_boost
                morph_boost = float(act_morphogen[i]) * 0.2
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3 + morph_boost)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Attempt capture on improvement."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

        return new_act_captured, new_agg_captured, capture_count

    def _update_affinities(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
        act_morphogen: jnp.ndarray,
        agg_morphogen: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update affinities with morphogen modulation."""
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if fitness_delta > 0:
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    lr = self.agg_affinity_lr * (1 + float(agg_morphogen[j]) * 0.3)
                    bonus = 1.0
                    if j in CORE_EXTREME_AGGS:
                        bonus += 0.6
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + lr * fitness_delta * bonus)
                    )

            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    lr = self.act_affinity_lr * (1 + float(act_morphogen[i]) * 0.3)
                    bonus = 1.0
                    if i == 4:  # Sin
                        bonus += 0.5
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + lr * fitness_delta * bonus)
                    )

        # CRITICAL: Sin affinity never drops too low
        new_act_aff = new_act_aff.at[4].set(max(0.6, float(new_act_aff[4])))

        # Extreme agg affinity floor
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(max(0.5, float(new_agg_aff[agg])))

        # Cross-domain update
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        delta = self.cross_learning_rate * fitness_delta
                        if i == 4 and j in CORE_EXTREME_AGGS:
                            delta *= (1 + self.sin_extreme_coupling)
                        new_cross = new_cross.at[i, j].set(
                            min(1.0, new_cross[i, j] + delta)
                        )

        return new_act_aff, new_agg_aff, new_cross

    def _select_act_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        morphogen: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        agg_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int]:
        """Select activation palette with morphogen and guaranteed sin."""
        # Compute scores
        score = affinities + captured * 0.3 + tags * 0.2 + morphogen * 0.4

        # Cross-domain influence
        for i in range(NUM_ACTIVATIONS):
            cross_influence = 0.0
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    cross_influence = max(cross_influence, float(cross_affinity[i, j]))
            score = score.at[i].set(score[i] + cross_influence * 0.25)

        # Sin gets STRONG preference (morphogen is already high, add more)
        score = score.at[4].set(score[4] + 1.0)

        target_size = min(max(self.min_diversity_act, self.min_active_act), self.max_active_act)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # CRITICAL: ALWAYS include sin
        mask = mask.at[4].set(1.0)

        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < self.min_diversity_act:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(NUM_ACTIVATIONS) if mask[i] < 0.5]
            needed = self.min_diversity_act - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive),
                                          shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue

    def _select_agg_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        morphogen: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, int]:
        """Select aggregation palette with morphogen and guaranteed extremes."""
        score = affinities + captured * 0.3 + tags * 0.2 + morphogen * 0.4

        # Extreme aggs get bonus
        for agg in CORE_EXTREME_AGGS:
            score = score.at[agg].set(score[agg] + 0.5)

        target_size = min(max(self.min_diversity_agg, self.min_active_agg), self.max_active_agg)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(NUM_AGGREGATIONS)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # CRITICAL: ALWAYS include at least one extreme agg
        has_extreme = any(mask[agg] > 0.5 for agg in CORE_EXTREME_AGGS)
        if not has_extreme:
            mask = mask.at[2].set(1.0)  # Add max

        return mask, 0

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with morphogen dynamics and guaranteed sin retention."""
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

        # === UPDATE MORPHOGEN FIELDS ===
        new_act_morphogen, new_agg_morphogen = self._update_morphogen_fields(
            state['act_morphogen'], state['agg_morphogen'],
            state['act_mask'], state['agg_mask'],
            improved, phase
        )

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags'],
            new_act_morphogen, new_agg_morphogen
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE ===
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            new_tag_history, generation, improved
        )

        # === AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross_affinity = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], state['agg_mask'],
            state['cross_affinity'], fitness_delta,
            new_act_morphogen, new_agg_morphogen
        )

        # === PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_act_palette(
            new_act_aff, new_act_captured, new_act_tags, new_act_morphogen,
            new_cross_affinity, state['agg_mask'], k1
        )

        new_agg_mask, _ = self._select_agg_palette(
            new_agg_aff, new_agg_captured, new_agg_tags, new_agg_morphogen, k2
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        sin_strength_history = state['sin_source_strength_history'] + [float(new_act_morphogen[4])]
        if len(sin_strength_history) > 20:
            sin_strength_history = sin_strength_history[-20:]

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_morphogen': new_act_morphogen,
            'agg_morphogen': new_agg_morphogen,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            'capture_events': state['capture_events'] + capture_count,
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue,
            'sin_source_strength_history': sin_strength_history,
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
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            # Morphogen metrics
            'sin_morphogen': float(new_act_morphogen[4]),
            'max_morphogen': float(new_agg_morphogen[2]),
            'min_morphogen': float(new_agg_morphogen[3]),
            # Affinity metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'sin_affinity': float(new_act_aff[4]),
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Cross-domain
            'sin_max_affinity': float(new_cross_affinity[4, 2]),
            'sin_min_affinity': float(new_cross_affinity[4, 3]),
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
            'sin_morphogen': float(state['act_morphogen'][4]),
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinities'])),
            'capture_events': state['capture_events'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
