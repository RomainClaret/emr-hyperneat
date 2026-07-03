"""Strategy 92: Multi-Neuromodulatory + Tag-Homeostatic Hybrid.

Combines tag_homeostatic_dual with neuromodulation:
- Base: Strategy 84 (Tag+Homeostatic) - 67% Parity-5 solve, 100% sin retention
- Extension: Strategy 21 (Multi-Neuromodulatory) - DA/ACh/NE/5-HT adaptive learning

Key innovation: Neuromodulators modulate tagging and capture dynamics:
- NE (norepinephrine) on stagnation → increases tag sensitivity (exploration)
- 5-HT (serotonin) on improvement → increases capture protection (retention)
- DA (dopamine) modulates learning rate for affinity updates
- ACh (acetylcholine) modulates precision of cross-domain learning

Bio inspiration: Neuromodulators like dopamine and serotonin regulate learning
and memory consolidation in biological neural networks. This creates an adaptive
system that adjusts exploration/exploitation based on evolutionary progress.

Expected: Adaptive exploration/retention based on evolutionary state.
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


class NeuromodHybridDualStrategy(PaletteEvolutionStrategy):
    """Tag-Homeostatic base with multi-neuromodulatory extension.

    Hybrid combining:
    - Tag+Homeostatic (84): Tag-and-capture + homeostatic balance
    - Multi-Neuromodulatory (21): DA/ACh/NE/5-HT adaptive modulation

    Critical interaction: Neuromodulators control tagging dynamics:
    - High NE (stagnation) → more sensitive tagging, faster discovery
    - High 5-HT (progress) → stronger capture protection
    - High DA (reward) → faster affinity learning
    - High ACh (focus) → more precise cross-domain learning
    """

    name = "neuromod_hybrid_dual"
    description = "Dual: Tag+Homeostatic base with neuromodulatory control"

    def __init__(
        self,
        # === Neuromodulator parameters (from strategy 21) ===
        ach_baseline: float = 0.5,
        da_baseline: float = 0.5,
        ne_baseline: float = 0.5,
        serotonin_baseline: float = 0.5,
        ach_sensitivity: float = 0.3,
        da_sensitivity: float = 0.4,
        ne_sensitivity: float = 0.35,
        serotonin_sensitivity: float = 0.2,
        ach_decay: float = 0.1,
        da_decay: float = 0.15,
        ne_decay: float = 0.12,
        serotonin_decay: float = 0.05,
        # === Tagging parameters (from strategy 84/81) ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        # === Neuromodulator effects on tagging (NEW) ===
        ne_tag_sensitivity_boost: float = 0.3,  # NE increases tagging
        serotonin_capture_boost: float = 0.2,  # 5-HT increases capture protection
        da_affinity_boost: float = 0.2,  # DA increases affinity learning
        ach_precision_boost: float = 0.3,  # ACh increases cross-domain precision
        # === Homeostatic parameters (from strategy 84/82) ===
        target_extreme_ratio: float = 0.60,
        imbalance_threshold: float = 0.15,
        correction_strength: float = 1.8,
        discovery_bonus: float = 0.5,
        extreme_protection: float = 0.6,
        sin_protection: float = 0.5,
        # === Cross-domain parameters ===
        base_cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # === Mutation parameters ===
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Neuromod+Tag+Homeostatic hybrid strategy."""
        # Neuromodulators
        self.ach_baseline = ach_baseline
        self.da_baseline = da_baseline
        self.ne_baseline = ne_baseline
        self.serotonin_baseline = serotonin_baseline
        self.ach_sensitivity = ach_sensitivity
        self.da_sensitivity = da_sensitivity
        self.ne_sensitivity = ne_sensitivity
        self.serotonin_sensitivity = serotonin_sensitivity
        self.ach_decay = ach_decay
        self.da_decay = da_decay
        self.ne_decay = ne_decay
        self.serotonin_decay = serotonin_decay

        # Neuromodulator effects on tagging
        self.ne_tag_sensitivity_boost = ne_tag_sensitivity_boost
        self.serotonin_capture_boost = serotonin_capture_boost
        self.da_affinity_boost = da_affinity_boost
        self.ach_precision_boost = ach_precision_boost

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.imbalance_threshold = imbalance_threshold
        self.correction_strength = correction_strength
        self.discovery_bonus = discovery_bonus
        self.extreme_protection = extreme_protection
        self.sin_protection = sin_protection

        # Cross-domain
        self.base_cross_learning_rate = base_cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neuromodulatory + tagging + homeostatic tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Neuromodulators
            'acetylcholine': self.ach_baseline,
            'dopamine': self.da_baseline,
            'norepinephrine': self.ne_baseline,
            'serotonin': self.serotonin_baseline,
            # Stats
            'capture_events': 0,
            'homeostatic_corrections': 0,
            'discovery_bonuses_applied': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 920000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            'fitness_ema': 0.5,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_neuromodulators(
        self,
        ach: float,
        da: float,
        ne: float,
        serotonin: float,
        fitness_signal: float,
        stagnation: int,
        improved: bool,
    ) -> Tuple[float, float, float, float, Dict]:
        """Update neuromodulator levels based on evolutionary state."""
        # ACh: Increases with consistent improvement (focus)
        if improved:
            ach_delta = self.ach_sensitivity * 0.3
        else:
            ach_delta = -self.ach_sensitivity * 0.1

        # DA: Reward prediction error
        da_delta = self.da_sensitivity * fitness_signal

        # NE: Increases with stagnation (exploration)
        if stagnation > 5:
            ne_delta = self.ne_sensitivity * 0.4 * (stagnation / 20)
        elif improved:
            ne_delta = -self.ne_sensitivity * 0.2
        else:
            ne_delta = 0.0

        # 5-HT: Long-term stability
        if fitness_signal > 0.2:
            serotonin_delta = self.serotonin_sensitivity * 0.2
        elif fitness_signal < -0.2:
            serotonin_delta = -self.serotonin_sensitivity * 0.1
        else:
            serotonin_delta = 0.0

        # Decay toward baseline
        ach_delta += self.ach_decay * (self.ach_baseline - ach)
        da_delta += self.da_decay * (self.da_baseline - da)
        ne_delta += self.ne_decay * (self.ne_baseline - ne)
        serotonin_delta += self.serotonin_decay * (self.serotonin_baseline - serotonin)

        new_ach = max(0.1, min(0.9, ach + ach_delta))
        new_da = max(0.1, min(0.9, da + da_delta))
        new_ne = max(0.1, min(0.9, ne + ne_delta))
        new_serotonin = max(0.1, min(0.9, serotonin + serotonin_delta))

        metrics = {
            'ach_delta': ach_delta,
            'da_delta': da_delta,
            'ne_delta': ne_delta,
            'serotonin_delta': serotonin_delta,
        }

        return new_ach, new_da, new_ne, new_serotonin, metrics

    def _update_tags_neuromodulated(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        ne: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with NE-modulated sensitivity."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        # NE boosts tag sensitivity
        tag_sensitivity = 1.0 + (ne - 0.5) * self.ne_tag_sensitivity_boost * 2

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = tag_sensitivity
                if i == 4:  # Sin boost
                    tag_strength *= self.extreme_tag_boost
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = tag_sensitivity
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture_neuromodulated(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
        serotonin: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Attempt capture with 5-HT-modulated thresholds."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

        # 5-HT lowers capture threshold (easier capture when stable)
        threshold_modifier = 1.0 - (serotonin - 0.5) * self.serotonin_capture_boost
        effective_act_threshold = self.tag_threshold * threshold_modifier
        effective_agg_threshold = self.agg_tag_threshold * threshold_modifier

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > effective_act_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > effective_agg_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

        return new_act_captured, new_agg_captured, capture_count

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute extreme/averaging ratio for homeostatic balance."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def _update_cross_affinity_neuromodulated(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
        da: float,
        ach: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity with DA and ACh modulation."""
        new_affinity = cross_affinity.copy()

        if improvement > 0:
            # DA boosts learning rate
            learning_rate = self.base_cross_learning_rate * (1 + (da - 0.5) * self.da_affinity_boost * 2)
            # ACh increases precision (reduces noise)
            precision = 1.0 + (ach - 0.5) * self.ach_precision_boost * 2

            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    for j in range(NUM_AGGREGATIONS):
                        if agg_mask[j] > 0.5:
                            current = cross_affinity[i, j]
                            boost = learning_rate * improvement * precision
                            if i == 4 and j in CORE_EXTREME_AGGS:
                                boost *= (1 + self.sin_extreme_affinity_boost)
                            new_affinity = new_affinity.at[i, j].set(
                                min(1.0, current + boost)
                            )

        return new_affinity

    def _mutate_act_palette_neuromodulated(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        ne: float,
        serotonin: float,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Mutate activation palette with neuromodulator influence."""
        new_mask = mask.copy()
        activated = []
        deactivated = []

        # NE increases exploration (activation)
        # 5-HT increases retention (deactivation protection)
        ne_modifier = 1.0 + (ne - 0.5) * 0.5
        serotonin_modifier = 1.0 + (serotonin - 0.5) * 0.4

        for i in range(NUM_ACTIVATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            affinity_boost = float(jnp.max(cross_affinity[i, :]))

            if mask[i] < 0.5:  # Inactive
                activate_rate = self.base_activate_rate * ne_modifier
                activate_rate *= (1 + affinity_boost * 0.5)
                if i == 4:  # Sin discovery bonus
                    activate_rate += self.discovery_bonus
                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                deactivate_rate = self.base_deactivate_rate / serotonin_modifier

                # Captured protection (boosted by 5-HT)
                if captured[i] > 0.5:
                    protection = self.captured_protection + (serotonin - 0.5) * self.serotonin_capture_boost
                    deactivate_rate *= (1 - protection)

                # Sin protection
                if i == 4:
                    deactivate_rate *= (1 - self.sin_protection)

                # Affinity protection
                if affinity_boost > 0.6:
                    deactivate_rate *= 0.7

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_act or active_count > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _mutate_agg_palette_neuromodulated(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        extreme_ratio: float,
        ne: float,
        serotonin: float,
    ) -> Tuple[jnp.ndarray, Dict, int, int]:
        """Mutate aggregation palette with neuromodulator influence and homeostatic balance."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        homeostatic_corrections = 0
        discovery_bonuses = 0

        ne_modifier = 1.0 + (ne - 0.5) * 0.5
        serotonin_modifier = 1.0 + (serotonin - 0.5) * 0.4

        needs_more_extreme = extreme_ratio < self.target_extreme_ratio - self.imbalance_threshold
        needs_more_averaging = extreme_ratio > self.target_extreme_ratio + self.imbalance_threshold

        for j in range(NUM_AGGREGATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            sin_affinity = float(cross_affinity[4, j]) if act_mask[4] > 0.5 else 0.5
            is_extreme = j in EXTREME_AGGS
            is_core_extreme = j in CORE_EXTREME_AGGS

            if mask[j] < 0.5:  # Inactive
                activate_rate = self.base_activate_rate * ne_modifier

                # Homeostatic correction
                if needs_more_extreme and is_core_extreme:
                    activate_rate *= self.correction_strength
                    homeostatic_corrections += 1
                elif needs_more_averaging and not is_extreme:
                    activate_rate *= self.correction_strength
                    homeostatic_corrections += 1

                # Discovery bonus
                if is_core_extreme and extreme_ratio < 0.5:
                    activate_rate += self.discovery_bonus
                    discovery_bonuses += 1

                # Sin affinity boost
                if sin_affinity > 0.6:
                    activate_rate *= 1.3

                if p < activate_rate:
                    new_mask = new_mask.at[j].set(1.0)
                    activated.append(j)
            else:  # Active
                deactivate_rate = self.base_deactivate_rate / serotonin_modifier

                # Captured protection (boosted by 5-HT)
                if captured[j] > 0.5:
                    protection = self.captured_protection + (serotonin - 0.5) * self.serotonin_capture_boost
                    deactivate_rate *= (1 - protection)

                # Extreme protection
                if is_core_extreme:
                    deactivate_rate *= (1 - self.extreme_protection)

                # Homeostatic protection
                if needs_more_extreme and is_extreme:
                    deactivate_rate *= 0.5
                elif needs_more_averaging and not is_extreme:
                    deactivate_rate *= 0.5

                # Sin affinity protection
                if sin_affinity > 0.6:
                    deactivate_rate *= 0.7

                if p < deactivate_rate:
                    new_mask = new_mask.at[j].set(0.0)
                    deactivated.append(j)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}, homeostatic_corrections, discovery_bonuses

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with combined neuromodulatory + tag + homeostatic mechanisms."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Fitness signal for neuromodulators
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # === NEUROMODULATOR UPDATE ===
        new_ach, new_da, new_ne, new_serotonin, neuromod_metrics = self._update_neuromodulators(
            state['acetylcholine'],
            state['dopamine'],
            state['norepinephrine'],
            state['serotonin'],
            fitness_signal,
            new_stagnation,
            improved,
        )

        # === TAGGING (NE-modulated) ===
        new_act_tags, new_agg_tags = self._update_tags_neuromodulated(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags'],
            new_ne
        )

        # Update tag history
        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE (5-HT-modulated) ===
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture_neuromodulated(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            new_tag_history, generation, improved, new_serotonin
        )

        # === HOMEOSTATIC TRACKING ===
        extreme_ratio = self._compute_extreme_ratio(state['agg_mask'])

        # === CROSS-DOMAIN AFFINITY (DA/ACh-modulated) ===
        new_cross_affinity = self._update_cross_affinity_neuromodulated(
            state['cross_affinity'], state['act_mask'], state['agg_mask'],
            improvement, new_da, new_ach
        )

        # === MUTATIONS (NE/5-HT-modulated) ===
        new_act_mask, act_mut_info = self._mutate_act_palette_neuromodulated(
            k_act, state['act_mask'], new_act_captured, new_cross_affinity,
            new_ne, new_serotonin
        )
        new_agg_mask, agg_mut_info, homeostatic_corrections, discovery_bonuses = self._mutate_agg_palette_neuromodulated(
            k_agg, state['agg_mask'], new_agg_captured, new_cross_affinity,
            state['act_mask'], extreme_ratio, new_ne, new_serotonin
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            'acetylcholine': new_ach,
            'dopamine': new_da,
            'norepinephrine': new_ne,
            'serotonin': new_serotonin,
            'capture_events': state['capture_events'] + capture_count,
            'homeostatic_corrections': state['homeostatic_corrections'] + homeostatic_corrections,
            'discovery_bonuses_applied': state['discovery_bonuses_applied'] + discovery_bonuses,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
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
            # Neuromodulator metrics
            'acetylcholine': new_ach,
            'dopamine': new_da,
            'norepinephrine': new_ne,
            'serotonin': new_serotonin,
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Homeostatic metrics
            'extreme_ratio': extreme_ratio,
            'homeostatic_corrections': new_state['homeostatic_corrections'],
            'discovery_bonuses_applied': new_state['discovery_bonuses_applied'],
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
        """Return state summary with neuromodulatory + tag + homeostatic status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            # Neuromodulators
            'acetylcholine': state['acetylcholine'],
            'dopamine': state['dopamine'],
            'norepinephrine': state['norepinephrine'],
            'serotonin': state['serotonin'],
            # Tagging
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'capture_events': state['capture_events'],
            # Homeostatic
            'homeostatic_corrections': state['homeostatic_corrections'],
            'discovery_bonuses_applied': state['discovery_bonuses_applied'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
