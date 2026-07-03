"""Strategy 100: STDP Adaptive Window Dual.

Attempts to fix the broken STDP+Tag (89) with dynamically adapting windows:
- Problem: Fixed windows don't match learning dynamics - during domain shifts,
  the LTD window (1-3 gens) depresses functions before they prove useful
- Hypothesis: Windows should expand during stagnation (give functions more time)
  and contract during progress (lock in discoveries faster)

Key mechanisms:
- Stagnation → windows grow (more averaging, less sensitivity to noise)
- Progress → windows shrink (faster learning, quicker capture)
- Domain shift detected → LTP expands, LTD contracts (protect during transition)

Bio inspiration: Neural time constants can change based on neuromodulatory state.
Noradrenaline/dopamine can adjust integration windows in real neural circuits.

Expected: Windows auto-adapt to learning dynamics, preventing premature depression.
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


class STDPAdaptiveWindowDualStrategy(PaletteEvolutionStrategy):
    """STDP with dynamically adapting temporal windows.

    Strategy addressing STDP+Tag failures:
    - Windows grow during stagnation (more averaging)
    - Windows shrink during progress (faster learning)
    - Domain shift: LTP expands, LTD contracts (protect)
    """

    name = "stdp_adaptive_window_dual"
    description = "Dual: STDP with adaptive window sizes"

    def __init__(
        self,
        # === Adaptive window parameters ===
        base_ltp_window: int = 5,               # Starting LTP window
        base_ltd_window: int = 3,               # Starting LTD window
        min_ltp_window: int = 2,                # Minimum LTP window
        max_ltp_window: int = 15,               # Maximum LTP window
        min_ltd_window: int = 1,                # Minimum LTD window
        max_ltd_window: int = 12,               # Maximum LTD window
        stagnation_window_growth: float = 0.2,  # How much windows grow per stag gen
        progress_window_shrink: float = 0.1,    # How much windows shrink on progress
        # Domain shift adaptation
        domain_shift_variance_threshold: float = 0.15,  # Variance indicating domain shift
        domain_shift_ltp_expansion: float = 1.5,        # LTP multiplier during shift
        domain_shift_ltd_contraction: float = 0.5,      # LTD multiplier during shift
        shift_adaptation_duration: int = 8,             # Gens to maintain shift mode
        # === STDP rates ===
        ltp_rate: float = 0.1,
        ltd_rate: float = 0.05,
        ltd_min_weight: float = 0.25,  # Floor
        # === Tag-and-capture parameters ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
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
        """Initialize STDP Adaptive Window strategy."""
        # Adaptive window parameters
        self.base_ltp_window = base_ltp_window
        self.base_ltd_window = base_ltd_window
        self.min_ltp_window = min_ltp_window
        self.max_ltp_window = max_ltp_window
        self.min_ltd_window = min_ltd_window
        self.max_ltd_window = max_ltd_window
        self.stagnation_window_growth = stagnation_window_growth
        self.progress_window_shrink = progress_window_shrink

        # Domain shift adaptation
        self.domain_shift_variance_threshold = domain_shift_variance_threshold
        self.domain_shift_ltp_expansion = domain_shift_ltp_expansion
        self.domain_shift_ltd_contraction = domain_shift_ltd_contraction
        self.shift_adaptation_duration = shift_adaptation_duration

        # STDP rates
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.ltd_min_weight = ltd_min_weight

        # Tag parameters
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

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
        """Initialize state with adaptive window tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # STDP weights
        act_stdp_weights = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_stdp_weights = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Adaptive windows (start at base)
        current_ltp_window = float(self.base_ltp_window)
        current_ltd_window = float(self.base_ltd_window)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'generation': 0,
            'best_fitness': 0.0,
            'stagnation_counter': 0,
            # STDP state
            'act_stdp_weights': act_stdp_weights,
            'agg_stdp_weights': agg_stdp_weights,
            'fitness_history': [],
            # Adaptive windows
            'current_ltp_window': current_ltp_window,
            'current_ltd_window': current_ltd_window,
            'in_domain_shift': False,
            'shift_counter': 0,
            # Tagging state
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'act_tag_gens': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'agg_tag_gens': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
        }

    def _detect_domain_shift(self, fitness_history: List[float]) -> bool:
        """Detect domain shift via high fitness variance."""
        if len(fitness_history) < 5:
            return False

        recent = fitness_history[-5:]
        variance = sum((f - sum(recent)/len(recent))**2 for f in recent) / len(recent)
        mean_fitness = sum(recent) / len(recent)

        # Normalize by mean
        if mean_fitness > 0.01:
            rel_variance = variance / (mean_fitness ** 2)
        else:
            rel_variance = variance

        return rel_variance > self.domain_shift_variance_threshold

    def _adapt_windows(
        self,
        ltp_window: float,
        ltd_window: float,
        improved: bool,
        in_domain_shift: bool,
        stagnation_counter: int
    ) -> Tuple[float, float]:
        """Adapt window sizes based on learning dynamics."""
        # Base adaptation
        if stagnation_counter > 5:
            # Stagnation → grow windows (more averaging)
            ltp_window += self.stagnation_window_growth
            ltd_window += self.stagnation_window_growth * 0.5
        elif improved:
            # Progress → shrink windows (faster learning)
            ltp_window -= self.progress_window_shrink
            ltd_window -= self.progress_window_shrink * 0.5

        # Domain shift adaptation
        if in_domain_shift:
            # Expand LTP, contract LTD
            effective_ltp = ltp_window * self.domain_shift_ltp_expansion
            effective_ltd = ltd_window * self.domain_shift_ltd_contraction
        else:
            effective_ltp = ltp_window
            effective_ltd = ltd_window

        # Clamp to bounds
        effective_ltp = max(self.min_ltp_window, min(self.max_ltp_window, effective_ltp))
        effective_ltd = max(self.min_ltd_window, min(self.max_ltd_window, effective_ltd))

        return effective_ltp, effective_ltd

    def update(
        self,
        state: Dict[str, Any],
        fitness_scores: jnp.ndarray,
        function_usage: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update with adaptive window STDP."""
        generation = state['generation']
        best_fitness = float(jnp.max(fitness_scores))
        mean_fitness = float(jnp.mean(fitness_scores))

        # Update fitness tracking
        improved = best_fitness > state['best_fitness'] + 1e-6
        state['best_fitness'] = max(state['best_fitness'], best_fitness)
        state['stagnation_counter'] = 0 if improved else state['stagnation_counter'] + 1

        # Record fitness history
        fitness_history = list(state.get('fitness_history', []))
        fitness_history.append(best_fitness)
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        # Detect domain shift
        is_domain_shift = self._detect_domain_shift(fitness_history)
        in_domain_shift = state.get('in_domain_shift', False)
        shift_counter = state.get('shift_counter', 0)

        if is_domain_shift and not in_domain_shift:
            # Entering domain shift
            in_domain_shift = True
            shift_counter = self.shift_adaptation_duration
        elif in_domain_shift:
            shift_counter -= 1
            if shift_counter <= 0:
                in_domain_shift = False

        # Adapt windows
        ltp_window = state.get('current_ltp_window', float(self.base_ltp_window))
        ltd_window = state.get('current_ltd_window', float(self.base_ltd_window))

        ltp_window, ltd_window = self._adapt_windows(
            ltp_window, ltd_window, improved, in_domain_shift,
            state['stagnation_counter']
        )

        # Get current active functions
        act_indices = mask_to_indices(state['act_mask'])
        agg_indices = mask_to_indices(state['agg_mask'])

        # Apply STDP updates with adaptive windows
        act_stdp = state['act_stdp_weights']
        agg_stdp = state['agg_stdp_weights']

        if len(fitness_history) >= 2:
            fitness_delta = fitness_history[-1] - fitness_history[-2]

            # LTP for active functions if improving
            if fitness_delta > 0:
                for i in act_indices:
                    if 0 <= i < NUM_ACTIVATIONS:
                        boost = self.ltp_rate * fitness_delta * (ltp_window / self.base_ltp_window)
                        act_stdp = act_stdp.at[i].set(
                            min(1.0, float(act_stdp[i]) + boost)
                        )
                for i in agg_indices:
                    if 0 <= i < NUM_AGGREGATIONS:
                        boost = self.ltp_rate * fitness_delta * (ltp_window / self.base_ltp_window)
                        agg_stdp = agg_stdp.at[i].set(
                            min(1.0, float(agg_stdp[i]) + boost)
                        )
            elif fitness_delta < 0 and not in_domain_shift:
                # LTD only if NOT in domain shift
                ltd_factor = ltd_window / self.base_ltd_window
                for i in act_indices:
                    if 0 <= i < NUM_ACTIVATIONS:
                        penalty = self.ltd_rate * abs(fitness_delta) * ltd_factor
                        act_stdp = act_stdp.at[i].set(
                            max(self.ltd_min_weight, float(act_stdp[i]) - penalty)
                        )
                for i in agg_indices:
                    if 0 <= i < NUM_AGGREGATIONS:
                        penalty = self.ltd_rate * abs(fitness_delta) * ltd_factor
                        agg_stdp = agg_stdp.at[i].set(
                            max(self.ltd_min_weight, float(agg_stdp[i]) - penalty)
                        )

        # Decay tags
        act_tags = state['act_tags'] * self.tag_decay
        agg_tags = state['agg_tags'] * self.tag_decay

        # Update tags from high STDP weight
        for i in range(NUM_ACTIVATIONS):
            if float(act_stdp[i]) > self.tag_threshold:
                act_tags = act_tags.at[i].set(
                    min(1.0, float(act_tags[i]) + 0.2)
                )

        for i in range(NUM_AGGREGATIONS):
            threshold = self.agg_tag_threshold
            if i in CORE_EXTREME_AGGS:
                threshold *= 0.8
            if float(agg_stdp[i]) > threshold:
                boost = 0.2 * (self.extreme_tag_boost if i in CORE_EXTREME_AGGS else 1.0)
                agg_tags = agg_tags.at[i].set(min(1.0, float(agg_tags[i]) + boost))

        # Track tag duration
        act_tag_gens = state['act_tag_gens']
        agg_tag_gens = state['agg_tag_gens']
        for i in range(NUM_ACTIVATIONS):
            if float(act_tags[i]) > 0.3:
                act_tag_gens = act_tag_gens.at[i].set(int(act_tag_gens[i]) + 1)
            else:
                act_tag_gens = act_tag_gens.at[i].set(0)
        for i in range(NUM_AGGREGATIONS):
            if float(agg_tags[i]) > 0.3:
                agg_tag_gens = agg_tag_gens.at[i].set(int(agg_tag_gens[i]) + 1)
            else:
                agg_tag_gens = agg_tag_gens.at[i].set(0)

        # Capture mechanism
        act_captured = state['act_captured']
        agg_captured = state['agg_captured']

        for i in range(NUM_ACTIVATIONS):
            if float(act_captured[i]) < 0.5 and int(act_tag_gens[i]) >= self.capture_window:
                act_captured = act_captured.at[i].set(1.0)

        for i in range(NUM_AGGREGATIONS):
            window = self.capture_window - 1 if i in CORE_EXTREME_AGGS else self.capture_window
            if float(agg_captured[i]) < 0.5 and int(agg_tag_gens[i]) >= window:
                agg_captured = agg_captured.at[i].set(1.0)

        state['act_stdp_weights'] = act_stdp
        state['agg_stdp_weights'] = agg_stdp
        state['fitness_history'] = fitness_history
        state['current_ltp_window'] = ltp_window
        state['current_ltd_window'] = ltd_window
        state['in_domain_shift'] = in_domain_shift
        state['shift_counter'] = shift_counter
        state['act_tags'] = act_tags
        state['agg_tags'] = agg_tags
        state['act_tag_gens'] = act_tag_gens
        state['agg_tag_gens'] = agg_tag_gens
        state['act_captured'] = act_captured
        state['agg_captured'] = agg_captured
        state['generation'] = generation + 1

        return state

    def mutate(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any],
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], jax.random.PRNGKey]:
        """Mutate based on STDP weights - captured protected."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_stdp = state['act_stdp_weights']
        agg_stdp = state['agg_stdp_weights']
        act_captured = state['act_captured']
        agg_captured = state['agg_captured']

        mutation_rate = 0.08

        rng_key, k1, k2, k3, k4 = jax.random.split(rng_key, 5)

        act_indices = mask_to_indices(act_mask)
        agg_indices = mask_to_indices(agg_mask)

        # Activation mutations
        if jax.random.uniform(k1) < mutation_rate:
            inactive_acts = [i for i in range(NUM_ACTIVATIONS)
                           if float(act_mask[i]) < 0.5]
            if inactive_acts:
                weights = [float(act_stdp[i]) + 0.1 for i in inactive_acts]
                total = sum(weights)
                weights = [w / total for w in weights]
                idx = jax.random.choice(k2, jnp.array(inactive_acts),
                                       p=jnp.array(weights))
                act_mask = act_mask.at[int(idx)].set(1.0)

        # Remove low-STDP (but not captured)
        if jax.random.uniform(k2) < mutation_rate and len(act_indices) > self.min_active_act:
            candidates = [i for i in act_indices
                         if float(act_captured[i]) < 0.5
                         and float(act_stdp[i]) < self.ltd_min_weight + 0.1
                         and i != 4]
            if candidates:
                stdp_vals = [(i, float(act_stdp[i])) for i in candidates]
                stdp_vals.sort(key=lambda x: x[1])
                to_remove = stdp_vals[0][0]
                act_mask = act_mask.at[to_remove].set(0.0)

        # Aggregation mutations
        if jax.random.uniform(k3) < mutation_rate:
            inactive_aggs = [i for i in range(NUM_AGGREGATIONS)
                           if float(agg_mask[i]) < 0.5]
            if inactive_aggs:
                weights = []
                for i in inactive_aggs:
                    w = float(agg_stdp[i]) + 0.1
                    if i in CORE_EXTREME_AGGS:
                        w *= 2.0
                    weights.append(w)
                total = sum(weights)
                weights = [w / total for w in weights]
                idx = jax.random.choice(k4, jnp.array(inactive_aggs),
                                       p=jnp.array(weights))
                agg_mask = agg_mask.at[int(idx)].set(1.0)

        # Remove low-STDP agg
        if jax.random.uniform(k4) < mutation_rate and len(agg_indices) > self.min_active_agg:
            candidates = [i for i in agg_indices
                         if float(agg_captured[i]) < 0.5
                         and float(agg_stdp[i]) < self.ltd_min_weight + 0.1
                         and i not in CORE_EXTREME_AGGS]
            if candidates:
                stdp_vals = [(i, float(agg_stdp[i])) for i in candidates]
                stdp_vals.sort(key=lambda x: x[1])
                to_remove = stdp_vals[0][0]
                agg_mask = agg_mask.at[to_remove].set(0.0)

        # Ensure constraints
        act_active = int(jnp.sum(act_mask))
        agg_active = int(jnp.sum(agg_mask))

        if act_active < self.min_active_act:
            inactive = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if inactive:
                rng_key, k = jax.random.split(rng_key)
                idx = jax.random.choice(k, jnp.array(inactive))
                act_mask = act_mask.at[int(idx)].set(1.0)

        if agg_active < self.min_active_agg:
            inactive = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if inactive:
                rng_key, k = jax.random.split(rng_key)
                idx = jax.random.choice(k, jnp.array(inactive))
                agg_mask = agg_mask.at[int(idx)].set(1.0)

        state['act_mask'] = act_mask
        state['agg_mask'] = agg_mask

        return state, rng_key

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Get current activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Get current aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update after each generation - wrapper for compatibility."""
        fitness_scores = jnp.array([best_fitness])
        state = self.update(state, fitness_scores, {}, {})
        rng_key = state.get('rng_key', jax.random.PRNGKey(generation))
        state, _ = self.mutate(state, {}, rng_key)
        metrics = self.get_diagnostics(state)
        return state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return summary for logging."""
        return self.get_diagnostics(state)

    def get_diagnostics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy diagnostics."""
        act_indices = mask_to_indices(state['act_mask'])
        agg_indices = mask_to_indices(state['agg_mask'])

        act_captured_list = [i for i in range(NUM_ACTIVATIONS)
                           if float(state['act_captured'][i]) > 0.5]
        agg_captured_list = [i for i in range(NUM_AGGREGATIONS)
                           if float(state['agg_captured'][i]) > 0.5]

        return {
            'generation': state['generation'],
            'act_palette_size': len(act_indices),
            'agg_palette_size': len(agg_indices),
            'has_sin': 4 in act_indices,
            'has_max': 2 in agg_indices,
            'has_min': 3 in agg_indices,
            'act_captured': act_captured_list,
            'agg_captured': agg_captured_list,
            'sin_captured': 4 in act_captured_list,
            'max_captured': 2 in agg_captured_list,
            'min_captured': 3 in agg_captured_list,
            'sin_stdp_weight': float(state['act_stdp_weights'][4]),
            'max_stdp_weight': float(state['agg_stdp_weights'][2]),
            # Adaptive window diagnostics
            'current_ltp_window': state.get('current_ltp_window', self.base_ltp_window),
            'current_ltd_window': state.get('current_ltd_window', self.base_ltd_window),
            'in_domain_shift': state.get('in_domain_shift', False),
        }
