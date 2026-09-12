"""Strategy 97: STDP Domain-Aware Dual.

Fixes the broken STDP+Tag (strategy 89) by detecting domain shifts and
suspending LTD during transitions.

Root Cause of Strategy 89 Failure:
- LTD window (1-3 gens) depresses functions during domain shifts
- Functions don't immediately produce results in new domain → LTD kicks in
- Tag-STDP coupling creates catch-22: need high STDP for capture, but LTD drops it

Key Innovation:
- Detect domain shifts via high fitness variance
- SUSPEND LTD for N generations during transitions
- Apply LTP boost after suspension ends to help re-establish useful functions

Bio inspiration: Learning rate modulation during state changes - the brain
reduces synaptic depression during periods of high uncertainty/novelty.

Expected: Sin retains STDP weight through domain shifts, improving retention.
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class STDPDomainAwareDualStrategy(PaletteEvolutionStrategy):
    """STDP+Tag with domain shift detection and LTD suspension.

    Adjusts the STDP+Tag mechanism by detecting
    domain shifts and protecting functions during transitions.

    Critical fix: When fitness variance is high (domain shift), suspend LTD
    for N generations. This prevents valuable functions from being depressed
    before they can prove useful in the new domain.
    """

    name = "stdp_domain_aware_dual"
    description = "Dual: STDP+Tag with domain-shift-aware LTD suspension"

    def __init__(
        self,
        # === STDP parameters (from strategy 89) ===
        ltp_window: int = 5,
        ltd_window: int = 3,
        agg_ltp_window: int = 4,
        agg_ltd_window: int = 2,
        ltp_rate: float = 0.15,
        ltd_rate: float = 0.05,
        agg_ltp_rate: float = 0.20,
        agg_ltd_rate: float = 0.08,
        cross_ltp_rate: float = 0.15,
        cross_ltd_rate: float = 0.06,
        extreme_ltp_multiplier: float = 1.4,
        # === DOMAIN SHIFT DETECTION (NEW) ===
        domain_shift_variance_threshold: float = 0.15,  # Fitness variance threshold
        ltd_suspension_duration: int = 10,               # Generations to suspend LTD
        post_suspension_ltp_boost: float = 1.3,          # LTP boost after suspension
        fitness_history_window: int = 8,                 # Window for variance calculation
        # === Tagging parameters (from strategy 89) ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        fitness_delta_threshold: float = 0.01,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        # === STDP-Tag interaction ===
        stdp_tag_influence: float = 0.4,
        capture_stdp_threshold: float = 0.55,  # Lowered from 0.6 to make capture easier
        high_stdp_ltd_reduction: float = 0.5,
        # === Mutation parameters ===
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        weight_influence: float = 0.5,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        stagnation_threshold: int = 5,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize STDP Domain-Aware strategy."""
        # STDP
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.agg_ltp_window = agg_ltp_window
        self.agg_ltd_window = agg_ltd_window
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.agg_ltp_rate = agg_ltp_rate
        self.agg_ltd_rate = agg_ltd_rate
        self.cross_ltp_rate = cross_ltp_rate
        self.cross_ltd_rate = cross_ltd_rate
        self.extreme_ltp_multiplier = extreme_ltp_multiplier

        # Domain shift detection (NEW)
        self.domain_shift_variance_threshold = domain_shift_variance_threshold
        self.ltd_suspension_duration = ltd_suspension_duration
        self.post_suspension_ltp_boost = post_suspension_ltp_boost
        self.fitness_history_window = fitness_history_window

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.fitness_delta_threshold = fitness_delta_threshold
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # STDP-Tag interaction
        self.stdp_tag_influence = stdp_tag_influence
        self.capture_stdp_threshold = capture_stdp_threshold
        self.high_stdp_ltd_reduction = high_stdp_ltd_reduction

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.weight_influence = weight_influence

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.stagnation_threshold = stagnation_threshold

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with domain shift tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # STDP weights
        act_weights = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_weights = jnp.ones(NUM_AGGREGATIONS) * 0.5
        cross_weights = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # STDP weights
            'act_weights': act_weights,
            'agg_weights': agg_weights,
            'cross_weights': cross_weights,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            # History
            'act_history': [],
            'agg_history': [],
            'tag_history': [],
            'fitness_history': [],
            # Domain shift tracking (NEW)
            'ltd_suspension_counter': 0,  # Generations remaining in suspension
            'domain_shifts_detected': 0,
            'current_fitness_variance': 0.0,
            'in_domain_shift': False,
            # Tracking
            'ltp_events': 0,
            'ltd_events': 0,
            'ltd_suspended_events': 0,  # NEW: LTD events that were suspended
            'cross_ltp_events': 0,
            'capture_events': 0,
            'stdp_capture_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 970000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _detect_domain_shift(self, fitness_history: List[float]) -> Tuple[bool, float]:
        """Detect domain shift via fitness variance.

        High variance indicates unstable learning, likely from domain change.
        Returns: (is_domain_shift, variance)
        """
        if len(fitness_history) < 3:
            return False, 0.0

        # Use recent history
        recent = fitness_history[-self.fitness_history_window:]
        if len(recent) < 3:
            return False, 0.0

        variance = float(np.var(recent))

        # Also check for sudden drops (another domain shift indicator)
        if len(recent) >= 2:
            recent_delta = recent[-1] - recent[-2]
            if recent_delta < -0.2:  # Sudden fitness drop
                return True, variance

        return variance > self.domain_shift_variance_threshold, variance

    def _update_history(
        self,
        history: List,
        generation: int,
        mask: jnp.ndarray,
        max_length: int = 10,
    ) -> List:
        """Update activity history."""
        new_history = history + [(generation, mask.copy())]
        if len(new_history) > max_length:
            new_history = new_history[-max_length:]
        return new_history

    def _update_tags_with_stdp(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_weights: jnp.ndarray,
        agg_weights: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with STDP weight influence."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                stdp_boost = 1.0 + self.stdp_tag_influence * (act_weights[i] - 0.5)
                tag_strength *= stdp_boost
                if i == 4:  # Sin boost
                    tag_strength *= self.extreme_tag_boost
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                stdp_boost = 1.0 + self.stdp_tag_influence * (agg_weights[j] - 0.5)
                tag_strength *= stdp_boost
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _apply_ltp(
        self,
        weights: jnp.ndarray,
        history: List,
        current_gen: int,
        window: int,
        rate: float,
        ltp_boost: float = 1.0,  # Post-suspension boost
        is_agg: bool = False,
    ) -> Tuple[jnp.ndarray, int]:
        """Apply LTP with optional post-suspension boost."""
        new_weights = weights.copy()
        n_ltp = 0

        for gen, mask in history:
            if 0 < current_gen - gen <= window:
                for i in range(len(mask)):
                    if mask[i] > 0.5:
                        effective_rate = rate * ltp_boost
                        if is_agg and i in CORE_EXTREME_AGGS:
                            effective_rate *= self.extreme_ltp_multiplier
                        current = new_weights[i]
                        new_weights = new_weights.at[i].set(
                            current + effective_rate * (1 - current)
                        )
                        n_ltp += 1

        return jnp.clip(new_weights, 0.0, 1.0), n_ltp

    def _apply_ltd_with_suspension(
        self,
        weights: jnp.ndarray,
        history: List,
        current_gen: int,
        window: int,
        rate: float,
        ltd_suspended: bool,
    ) -> Tuple[jnp.ndarray, int, int]:
        """Apply LTD with domain-shift suspension.

        Returns: (new_weights, n_ltd_applied, n_ltd_suspended)
        """
        new_weights = weights.copy()
        n_ltd = 0
        n_suspended = 0

        for gen, mask in history:
            if 0 < current_gen - gen <= window:
                for i in range(len(mask)):
                    if mask[i] > 0.5:
                        if ltd_suspended:
                            # LTD SUSPENDED: Don't depress
                            n_suspended += 1
                            continue

                        current = float(weights[i])
                        # High STDP protection
                        effective_rate = rate
                        if current > self.capture_stdp_threshold:
                            effective_rate *= self.high_stdp_ltd_reduction

                        new_weights = new_weights.at[i].set(
                            current - effective_rate * current
                        )
                        n_ltd += 1

        return jnp.clip(new_weights, 0.0, 1.0), n_ltd, n_suspended

    def _apply_cross_ltp(
        self,
        cross_weights: jnp.ndarray,
        act_history: List,
        agg_history: List,
        current_gen: int,
        ltp_boost: float = 1.0,
    ) -> Tuple[jnp.ndarray, int]:
        """Apply cross-domain LTP."""
        new_weights = cross_weights.copy()
        n_ltp = 0

        for (act_gen, act_mask), (agg_gen, agg_mask) in zip(act_history, agg_history):
            if act_gen != agg_gen:
                continue
            if 0 < current_gen - act_gen <= self.ltp_window:
                for i in range(NUM_ACTIVATIONS):
                    if act_mask[i] > 0.5:
                        for j in range(NUM_AGGREGATIONS):
                            if agg_mask[j] > 0.5:
                                effective_rate = self.cross_ltp_rate * ltp_boost
                                if i == 4 and j in CORE_EXTREME_AGGS:
                                    effective_rate *= self.extreme_ltp_multiplier
                                current = new_weights[i, j]
                                new_weights = new_weights.at[i, j].set(
                                    current + effective_rate * (1 - current)
                                )
                                n_ltp += 1

        return jnp.clip(new_weights, 0.0, 1.0), n_ltp

    def _attempt_capture(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_weights: jnp.ndarray,
        agg_weights: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Capture with STDP weight requirement."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0
        stdp_capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0, 0

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    tag_ok = hist_act_tags[i] > self.tag_threshold
                    stdp_ok = act_weights[i] > self.capture_stdp_threshold
                    if tag_ok and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1
                        if stdp_ok:
                            stdp_capture_count += 1

                for j in range(NUM_AGGREGATIONS):
                    tag_ok = hist_agg_tags[j] > self.agg_tag_threshold
                    stdp_ok = agg_weights[j] > self.capture_stdp_threshold
                    if tag_ok and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1
                        if stdp_ok:
                            stdp_capture_count += 1

        return new_act_captured, new_agg_captured, capture_count, stdp_capture_count

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        weights: jnp.ndarray,
        captured: jnp.ndarray,
        cross_weights: jnp.ndarray,
        is_act: bool,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Mutate palette with combined protections."""
        n_funcs = NUM_ACTIVATIONS if is_act else NUM_AGGREGATIONS
        min_active = self.min_active_act if is_act else self.min_active_agg
        max_active = self.max_active_act if is_act else self.max_active_agg

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(n_funcs):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            weight = float(weights[i])
            is_captured = captured[i] > 0.5

            if is_act:
                cross_influence = float(jnp.max(cross_weights[i, :]))
            else:
                cross_influence = float(jnp.max(cross_weights[:, i]))

            combined_weight = weight * 0.6 + cross_influence * 0.4

            if mask[i] < 0.5:
                activate_rate = self.base_activate_rate * (
                    1 + self.weight_influence * (combined_weight - 0.5)
                )
                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                deactivate_rate = self.base_deactivate_rate * (
                    1 - self.weight_influence * (combined_weight - 0.5)
                )

                if is_captured:
                    deactivate_rate *= (1 - self.captured_protection)

                if is_act and i == 4:
                    deactivate_rate *= 0.5
                elif not is_act and i in CORE_EXTREME_AGGS:
                    deactivate_rate *= 0.5

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < min_active or active_count > max_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with domain-shift-aware STDP+tagging."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === UPDATE HISTORY ===
        new_act_history = self._update_history(state['act_history'], generation, state['act_mask'])
        new_agg_history = self._update_history(state['agg_history'], generation, state['agg_mask'])
        new_fitness_history = state['fitness_history'] + [best_fitness]
        if len(new_fitness_history) > self.fitness_history_window + 2:
            new_fitness_history = new_fitness_history[-(self.fitness_history_window + 2):]

        # === DOMAIN SHIFT DETECTION (NEW) ===
        is_domain_shift, fitness_variance = self._detect_domain_shift(new_fitness_history)

        # Update suspension counter
        new_ltd_suspension_counter = state['ltd_suspension_counter']
        new_domain_shifts = state['domain_shifts_detected']
        new_in_domain_shift = state['in_domain_shift']

        if is_domain_shift and not state['in_domain_shift']:
            # NEW domain shift detected - start suspension
            new_ltd_suspension_counter = self.ltd_suspension_duration
            new_domain_shifts += 1
            new_in_domain_shift = True
        elif new_ltd_suspension_counter > 0:
            new_ltd_suspension_counter -= 1
            if new_ltd_suspension_counter == 0:
                new_in_domain_shift = False

        ltd_suspended = new_ltd_suspension_counter > 0

        # Calculate LTP boost (apply after suspension ends)
        ltp_boost = 1.0
        if state['ltd_suspension_counter'] > 0 and new_ltd_suspension_counter == 0:
            # Just ended suspension - apply boost
            ltp_boost = self.post_suspension_ltp_boost

        # === STDP MECHANISM (with domain-aware LTD) ===
        new_act_weights = state['act_weights']
        new_agg_weights = state['agg_weights']
        new_cross_weights = state['cross_weights']
        new_ltp_events = state['ltp_events']
        new_ltd_events = state['ltd_events']
        new_ltd_suspended_events = state['ltd_suspended_events']
        new_cross_ltp_events = state['cross_ltp_events']

        if improved and fitness_delta > self.fitness_delta_threshold:
            # LTP (with potential post-suspension boost)
            new_act_weights, n_act_ltp = self._apply_ltp(
                state['act_weights'], new_act_history, generation,
                self.ltp_window, self.ltp_rate, ltp_boost, is_agg=False
            )
            new_agg_weights, n_agg_ltp = self._apply_ltp(
                state['agg_weights'], new_agg_history, generation,
                self.agg_ltp_window, self.agg_ltp_rate, ltp_boost, is_agg=True
            )
            new_cross_weights, n_cross_ltp = self._apply_cross_ltp(
                state['cross_weights'], new_act_history, new_agg_history,
                generation, ltp_boost
            )
            new_ltp_events += n_act_ltp + n_agg_ltp
            new_cross_ltp_events += n_cross_ltp
        elif not improved:
            # LTD (with domain-shift suspension)
            new_act_weights, n_act_ltd, n_act_suspended = self._apply_ltd_with_suspension(
                state['act_weights'], new_act_history, generation,
                self.ltd_window, self.ltd_rate, ltd_suspended
            )
            new_agg_weights, n_agg_ltd, n_agg_suspended = self._apply_ltd_with_suspension(
                state['agg_weights'], new_agg_history, generation,
                self.agg_ltd_window, self.agg_ltd_rate, ltd_suspended
            )
            new_ltd_events += n_act_ltd + n_agg_ltd
            new_ltd_suspended_events += n_act_suspended + n_agg_suspended

        # === TAGGING (with STDP influence) ===
        new_act_tags, new_agg_tags = self._update_tags_with_stdp(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags'],
            new_act_weights, new_agg_weights
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE ===
        new_act_captured, new_agg_captured, capture_count, stdp_capture_count = \
            self._attempt_capture(
                new_act_tags, new_agg_tags,
                new_act_weights, new_agg_weights,
                state['act_captured'], state['agg_captured'],
                new_tag_history, generation, improved
            )

        # === MUTATIONS ===
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, _ = self._mutate_palette(
                k_act, state['act_mask'], new_act_weights, new_act_captured,
                new_cross_weights, True
            )
            new_agg_mask, _ = self._mutate_palette(
                k_agg, state['agg_mask'], new_agg_weights, new_agg_captured,
                new_cross_weights, False
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            new_stagnation = 0

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_weights': new_act_weights,
            'agg_weights': new_agg_weights,
            'cross_weights': new_cross_weights,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'act_history': new_act_history,
            'agg_history': new_agg_history,
            'tag_history': new_tag_history,
            'fitness_history': new_fitness_history,
            # Domain shift state (NEW)
            'ltd_suspension_counter': new_ltd_suspension_counter,
            'domain_shifts_detected': new_domain_shifts,
            'current_fitness_variance': fitness_variance,
            'in_domain_shift': new_in_domain_shift,
            # Tracking
            'ltp_events': new_ltp_events,
            'ltd_events': new_ltd_events,
            'ltd_suspended_events': new_ltd_suspended_events,
            'cross_ltp_events': new_cross_ltp_events,
            'capture_events': state['capture_events'] + capture_count,
            'stdp_capture_events': state['stdp_capture_events'] + stdp_capture_count,
            # General
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
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
            # Domain shift metrics (NEW)
            'ltd_suspended': ltd_suspended,
            'ltd_suspension_counter': new_ltd_suspension_counter,
            'domain_shifts_detected': new_domain_shifts,
            'fitness_variance': fitness_variance,
            'ltd_suspended_events': new_ltd_suspended_events,
            # STDP metrics
            'sin_weight': float(new_act_weights[4]),
            'max_weight': float(new_agg_weights[2]),
            'min_weight': float(new_agg_weights[3]),
            'sin_max_cross': float(new_cross_weights[4, 2]),
            'sin_min_cross': float(new_cross_weights[4, 3]),
            'ltp_events': new_ltp_events,
            'ltd_events': new_ltd_events,
            'cross_ltp_events': new_cross_ltp_events,
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            'stdp_capture_events': new_state['stdp_capture_events'],
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with domain shift info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Domain shift
            'ltd_suspended': state['ltd_suspension_counter'] > 0,
            'domain_shifts_detected': state['domain_shifts_detected'],
            'fitness_variance': state['current_fitness_variance'],
            # STDP
            'sin_weight': float(state['act_weights'][4]),
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'ltp_events': state['ltp_events'],
            'ltd_events': state['ltd_events'],
            'ltd_suspended_events': state['ltd_suspended_events'],
            'capture_events': state['capture_events'],
            'stdp_capture_events': state['stdp_capture_events'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
