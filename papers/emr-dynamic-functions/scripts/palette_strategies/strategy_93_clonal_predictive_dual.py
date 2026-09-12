"""Strategy 93: Clonal Predictive Dual.

Combines clonal_hybrid_dual with predictive coding:
- Base: Strategy 91 (Clonal+Tag+Homeostatic) - 100% Parity-5, 100% sin retention
- Extension: Strategy 24 (Predictive Coding) - Prediction error drives learning

Key innovation: Prediction error identifies truly useful functions (not just correlated).
Surprising functions (high prediction error when beneficial) capture FASTER.

Bio inspiration: Cortical predictive coding proposes that learning is driven by
prediction ERRORS, not raw correlations. Expected outcomes don't teach much;
unexpected successes (positive surprise) signal important discoveries.

Expected: Better discrimination between correlation and causation, faster domain adaptation.
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


class ClonalPredictiveDualStrategy(PaletteEvolutionStrategy):
    """Clonal selection with prediction-error-driven capture.

    Extend clonal_hybrid_dual
    with predictive coding mechanisms.

    Critical interaction: Functions that produce UNEXPECTED fitness improvements
    (positive prediction error) capture faster than merely correlated functions.
    This should discriminate between causally useful and coincidentally active.
    """

    name = "clonal_predictive_dual"
    description = "Dual: Clonal+Tag base with prediction-error-driven capture"

    def __init__(
        self,
        # === Clonal selection parameters (from strategy 91) ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        hypermutation_rate: float = 0.08,
        hypermutation_strength: float = 0.2,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Tagging parameters (from strategy 91/84) ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        captured_hypermutation_protection: float = 0.9,
        # === PREDICTIVE CODING PARAMETERS (NEW from strategy 24) ===
        prediction_lr: float = 0.15,           # How fast predictions update
        surprise_threshold: float = 0.15,      # Above = "confused" state
        capture_surprise_multiplier: float = 1.4,  # Surprising = faster capture
        prediction_smoothing: float = 0.4,     # EMA smoothing for errors
        surprise_exploration_boost: float = 1.3,   # Explore more when confused
        # === Homeostatic parameters (from strategy 91/84) ===
        target_extreme_ratio: float = 0.60,
        discovery_bonus: float = 0.5,
        # === Cross-domain parameters ===
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Clonal+Predictive hybrid strategy."""
        # Clonal selection
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.hypermutation_rate = hypermutation_rate
        self.hypermutation_strength = hypermutation_strength
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost
        self.captured_hypermutation_protection = captured_hypermutation_protection

        # Predictive coding (NEW)
        self.prediction_lr = prediction_lr
        self.surprise_threshold = surprise_threshold
        self.capture_surprise_multiplier = capture_surprise_multiplier
        self.prediction_smoothing = prediction_smoothing
        self.surprise_exploration_boost = surprise_exploration_boost

        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.discovery_bonus = discovery_bonus

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with clonal + tagging + predictive coding tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Clonal selection state: affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Predictive coding state (NEW)
        act_predictions = jnp.ones(NUM_ACTIVATIONS) * 0.5  # Expected usefulness
        agg_predictions = jnp.ones(NUM_AGGREGATIONS) * 0.5
        act_prediction_errors = jnp.zeros(NUM_ACTIVATIONS)  # Smoothed errors
        agg_prediction_errors = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Clonal selection
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Predictive coding (NEW)
            'act_predictions': act_predictions,
            'agg_predictions': agg_predictions,
            'act_prediction_errors': act_prediction_errors,
            'agg_prediction_errors': agg_prediction_errors,
            'global_surprise': 0.0,
            'fitness_baseline': 0.5,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'capture_events': 0,
            'surprise_captures': 0,  # NEW: captures boosted by surprise
            'hypermutation_events': 0,
            'diversity_rescues': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 930000),
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

    def _compute_actual_contributions(
        self,
        mask: jnp.ndarray,
        fitness: float,
        fitness_baseline: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute actual fitness contribution per function.

        Active functions share credit proportional to fitness improvement.
        Inactive functions get zero contribution.
        """
        if fitness_baseline > 0.01:
            fitness_signal = (fitness - fitness_baseline) / fitness_baseline
        else:
            fitness_signal = fitness - fitness_baseline

        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(float(jnp.sum(active)), 1.0)
        per_function = fitness_signal / n_active

        return active * per_function

    def _update_predictions(
        self,
        predictions: jnp.ndarray,
        actuals: jnp.ndarray,
        mask: jnp.ndarray,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update predictions based on actual outcomes.

        Active functions: move prediction toward actual.
        Inactive functions: decay toward baseline (0.5).

        Returns: (new_predictions, raw_prediction_errors)
        """
        raw_error = actuals - predictions
        active = (mask > 0.5).astype(jnp.float32)

        # Active: update prediction toward actual
        # Inactive: decay toward 0.5 baseline
        baseline = jnp.ones(n_funcs) * 0.5
        inactive_target = 0.95 * predictions + 0.05 * baseline

        new_predictions = jnp.where(
            active > 0.5,
            predictions + self.prediction_lr * raw_error,
            inactive_target
        )
        new_predictions = jnp.clip(new_predictions, 0.0, 1.0)

        return new_predictions, raw_error

    def _smooth_prediction_errors(
        self,
        current_errors: jnp.ndarray,
        new_raw_errors: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply EMA smoothing to prediction errors."""
        return (
            self.prediction_smoothing * current_errors +
            (1 - self.prediction_smoothing) * new_raw_errors
        )

    def _compute_surprise(
        self,
        act_errors: jnp.ndarray,
        agg_errors: jnp.ndarray,
    ) -> float:
        """Compute global surprise as mean absolute prediction error."""
        all_errors = jnp.concatenate([act_errors, agg_errors])
        return float(jnp.mean(jnp.abs(all_errors)))

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags for active functions."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:  # Sin boost
                    tag_strength *= self.extreme_tag_boost
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture_with_surprise(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        act_prediction_errors: jnp.ndarray,
        agg_prediction_errors: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Attempt capture with surprise-boosted threshold.

        KEY INNOVATION: Functions with positive prediction error (better than expected)
        have LOWER capture threshold (capture_surprise_multiplier > 1).
        This means surprising successes capture faster.

        Returns: (new_act_captured, new_agg_captured, capture_count, surprise_capture_count)
        """
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0
        surprise_capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0, 0

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    if new_act_captured[i] > 0.5:
                        continue  # Already captured

                    # Surprise-adjusted threshold
                    pred_error = float(act_prediction_errors[i])
                    if pred_error > 0:  # Better than expected
                        # Lower threshold = easier to capture
                        effective_threshold = self.tag_threshold / self.capture_surprise_multiplier
                        is_surprise_capture = True
                    else:
                        effective_threshold = self.tag_threshold
                        is_surprise_capture = False

                    if hist_act_tags[i] > effective_threshold:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1
                        if is_surprise_capture:
                            surprise_capture_count += 1

                for j in range(NUM_AGGREGATIONS):
                    if new_agg_captured[j] > 0.5:
                        continue

                    pred_error = float(agg_prediction_errors[j])
                    if pred_error > 0:
                        effective_threshold = self.agg_tag_threshold / self.capture_surprise_multiplier
                        is_surprise_capture = True
                    else:
                        effective_threshold = self.agg_tag_threshold
                        is_surprise_capture = False

                    if hist_agg_tags[j] > effective_threshold:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1
                        if is_surprise_capture:
                            surprise_capture_count += 1

        return new_act_captured, new_agg_captured, capture_count, surprise_capture_count

    def _update_affinities(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        act_prediction_errors: jnp.ndarray,
        agg_prediction_errors: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
        global_surprise: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Update affinities with prediction-error-modulated learning."""
        k1, k2, k3 = jax.random.split(key, 3)
        hypermutation_count = 0

        # Decay
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        # Prediction-error-modulated learning
        # Positive error (better than expected) = stronger learning
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    error_boost = 1.0 + max(0, float(act_prediction_errors[i]))
                    delta = self.act_affinity_lr * fitness_delta * error_boost
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + delta)
                    )
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    error_boost = 1.0 + max(0, float(agg_prediction_errors[j]))
                    delta = self.agg_affinity_lr * fitness_delta * error_boost
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + delta)
                    )

        # Surprise-modulated hypermutation
        # High surprise = explore more (higher hypermutation)
        if global_surprise > self.surprise_threshold:
            effective_hypermut_rate = self.hypermutation_rate * self.surprise_exploration_boost
        else:
            effective_hypermut_rate = self.hypermutation_rate

        hypermut_probs_act = jax.random.uniform(k1, (NUM_ACTIVATIONS,))
        hypermut_probs_agg = jax.random.uniform(k2, (NUM_AGGREGATIONS,))
        hypermut_amounts = jax.random.normal(k3, (NUM_ACTIVATIONS + NUM_AGGREGATIONS,)) * self.hypermutation_strength

        for i in range(NUM_ACTIVATIONS):
            if hypermut_probs_act[i] < effective_hypermut_rate:
                if act_captured[i] > 0.5:
                    # Captured = protected
                    check_rate = effective_hypermut_rate * (1 - self.captured_hypermutation_protection)
                    if hypermut_probs_act[i] >= check_rate:
                        continue
                new_act_aff = new_act_aff.at[i].set(
                    jnp.clip(new_act_aff[i] + hypermut_amounts[i], 0.0, 1.0)
                )
                hypermutation_count += 1

        for j in range(NUM_AGGREGATIONS):
            if hypermut_probs_agg[j] < effective_hypermut_rate:
                if agg_captured[j] > 0.5:
                    check_rate = effective_hypermut_rate * (1 - self.captured_hypermutation_protection)
                    if hypermut_probs_agg[j] >= check_rate:
                        continue
                new_agg_aff = new_agg_aff.at[j].set(
                    jnp.clip(new_agg_aff[j] + hypermut_amounts[NUM_ACTIVATIONS + j], 0.0, 1.0)
                )
                hypermutation_count += 1

        # Cross-domain affinity update
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(act_active, agg_active)
            delta = self.cross_learning_rate * fitness_delta * co_active
            # Sin-extreme boost
            for i in [4]:
                for j in CORE_EXTREME_AGGS:
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        delta = delta.at[i, j].set(delta[i, j] * (1 + self.sin_extreme_affinity_boost))
            new_cross = jnp.clip(new_cross + delta, 0.0, 1.0)

        return new_act_aff, new_agg_aff, new_cross, hypermutation_count

    def _select_palette_by_affinity(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        prediction_errors: jnp.ndarray,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette with prediction-error-aware scoring."""
        # Score: affinity + capture + tag + positive prediction error
        score = affinities + captured * 0.3 + tags * 0.2
        # Bonus for functions with positive prediction error (surprising successes)
        score = score + jnp.maximum(prediction_errors, 0) * 0.15

        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + self.discovery_bonus)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < min_diversity:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive), shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute extreme/averaging ratio for homeostatic tracking."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with combined clonal + tag + predictive coding mechanisms."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update fitness baseline (EMA)
        new_fitness_baseline = 0.9 * state['fitness_baseline'] + 0.1 * best_fitness

        # === PREDICTIVE CODING ===
        # Compute actual contributions
        act_actuals = self._compute_actual_contributions(
            state['act_mask'], best_fitness, state['fitness_baseline'], NUM_ACTIVATIONS
        )
        agg_actuals = self._compute_actual_contributions(
            state['agg_mask'], best_fitness, state['fitness_baseline'], NUM_AGGREGATIONS
        )

        # Update predictions and get errors
        new_act_predictions, act_raw_errors = self._update_predictions(
            state['act_predictions'], act_actuals, state['act_mask'], NUM_ACTIVATIONS
        )
        new_agg_predictions, agg_raw_errors = self._update_predictions(
            state['agg_predictions'], agg_actuals, state['agg_mask'], NUM_AGGREGATIONS
        )

        # Smooth errors
        new_act_pred_errors = self._smooth_prediction_errors(
            state['act_prediction_errors'], act_raw_errors
        )
        new_agg_pred_errors = self._smooth_prediction_errors(
            state['agg_prediction_errors'], agg_raw_errors
        )

        # Compute global surprise
        new_global_surprise = self._compute_surprise(new_act_pred_errors, new_agg_pred_errors)

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags']
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE (with surprise boost) ===
        new_act_captured, new_agg_captured, capture_count, surprise_capture_count = \
            self._attempt_capture_with_surprise(
                new_act_tags, new_agg_tags,
                state['act_captured'], state['agg_captured'],
                new_act_pred_errors, new_agg_pred_errors,
                new_tag_history, generation, improved
            )

        # === AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross_affinity, hypermut_count = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], state['agg_mask'],
            new_act_captured, new_agg_captured,
            new_act_pred_errors, new_agg_pred_errors,
            state['cross_affinity'], fitness_delta,
            new_global_surprise, k1
        )

        # === PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_palette_by_affinity(
            new_act_aff, new_act_captured, new_act_tags, new_act_pred_errors,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            self.min_diversity_act, k2, prefer_indices=[4]
        )
        new_agg_mask, agg_diversity_rescue = self._select_palette_by_affinity(
            new_agg_aff, new_agg_captured, new_agg_tags, new_agg_pred_errors,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            self.min_diversity_agg, jax.random.split(k2)[0],
            prefer_indices=list(CORE_EXTREME_AGGS)
        )

        # === HOMEOSTATIC TRACKING ===
        extreme_ratio = self._compute_extreme_ratio(new_agg_mask)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            # Predictive coding
            'act_predictions': new_act_predictions,
            'agg_predictions': new_agg_predictions,
            'act_prediction_errors': new_act_pred_errors,
            'agg_prediction_errors': new_agg_pred_errors,
            'global_surprise': new_global_surprise,
            'fitness_baseline': new_fitness_baseline,
            # Cross-domain
            'cross_affinity': new_cross_affinity,
            # Stats
            'capture_events': state['capture_events'] + capture_count,
            'surprise_captures': state['surprise_captures'] + surprise_capture_count,
            'hypermutation_events': state['hypermutation_events'] + hypermut_count,
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue + agg_diversity_rescue,
            # General state
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
            # Clonal selection metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'hypermutation_events': new_state['hypermutation_events'],
            'diversity_rescues': new_state['diversity_rescues'],
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_affinity': float(new_act_aff[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            'surprise_captures': new_state['surprise_captures'],  # NEW
            # Predictive coding metrics (NEW)
            'global_surprise': new_global_surprise,
            'sin_prediction_error': float(new_act_pred_errors[4]),
            'max_agg_prediction_error': float(new_agg_pred_errors[2]) if len(new_agg_pred_errors) > 2 else 0.0,
            'mean_act_prediction_error': float(jnp.mean(jnp.abs(new_act_pred_errors))),
            # Homeostatic metrics
            'extreme_ratio': extreme_ratio,
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
        """Return state summary with clonal + tag + predictive coding status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            # Clonal
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinities'])),
            'hypermutation_events': state['hypermutation_events'],
            'diversity_rescues': state['diversity_rescues'],
            # Tagging
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'sin_affinity': float(state['act_affinities'][4]),
            'capture_events': state['capture_events'],
            'surprise_captures': state['surprise_captures'],
            # Predictive coding
            'global_surprise': state['global_surprise'],
            'sin_prediction_error': float(state['act_prediction_errors'][4]),
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
