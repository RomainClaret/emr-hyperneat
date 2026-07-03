"""Strategy 98: STDP Extended Window Dual.

Attempts to fix the broken STDP+Tag (89) by extending temporal windows:
- Problem: Strategy 89's LTD window (1-3 gens) depresses functions too quickly
  during domain shifts before they can prove useful
- Hypothesis: Longer windows with proportionally reduced rates will allow
  functions to survive through domain transitions

Key changes:
- LTP window: 5 → 8 generations (more time to earn credit)
- LTD window: 3 → 10 generations (slower depression)
- LTD rate: 0.05 → 0.02 (gentler depression)
- LTD floor: 0.3 (minimum STDP weight, can't go below)

Bio inspiration: Some synapses have naturally longer time constants. Extended
windows allow more averaging over fitness fluctuations, reducing sensitivity
to temporary domain shift noise.

Expected: Functions retain STDP weight through domain transitions.
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


class STDPExtendedWindowDualStrategy(PaletteEvolutionStrategy):
    """STDP with extended temporal windows and reduced LTD rate.

    Strategy addressing STDP+Tag failures:
    - Longer LTP/LTD windows reduce sensitivity to domain shift noise
    - Reduced LTD rate prevents rapid depression
    - LTD floor ensures functions never fully depressed
    """

    name = "stdp_extended_window_dual"
    description = "Dual: STDP with extended windows and LTD floor"

    def __init__(
        self,
        # === Extended STDP parameters ===
        ltp_window: int = 8,          # Extended from 5 (generations for LTP)
        ltd_window: int = 10,         # Extended from 3 (generations for LTD)
        ltp_rate: float = 0.08,       # LTP rate (slightly reduced for longer window)
        ltd_rate: float = 0.02,       # Reduced from 0.05 (gentler depression)
        ltd_min_weight: float = 0.3,  # Floor: minimum STDP weight
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
        """Initialize STDP Extended Window strategy."""
        # Extended STDP parameters
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
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
        """Initialize state with extended STDP tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # STDP weights (starts at 0.5, can rise or fall)
        act_stdp_weights = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_stdp_weights = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Activity tracking for STDP
        # Track fitness history to determine LTP vs LTD
        fitness_history = []

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Per-function activity tracking (when were they active and what fitness?)
        act_activity_history = {i: [] for i in range(NUM_ACTIVATIONS)}
        agg_activity_history = {i: [] for i in range(NUM_AGGREGATIONS)}

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'generation': 0,
            'best_fitness': 0.0,
            'stagnation_counter': 0,
            # STDP state
            'act_stdp_weights': act_stdp_weights,
            'agg_stdp_weights': agg_stdp_weights,
            'fitness_history': fitness_history,
            'act_activity_history': act_activity_history,
            'agg_activity_history': agg_activity_history,
            # Tagging state
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'act_tag_gens': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'agg_tag_gens': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
        }

    def _apply_stdp_update(
        self,
        stdp_weights: jnp.ndarray,
        activity_history: Dict[int, List],
        fitness_history: List[float],
        generation: int,
        active_indices: List[int],
        current_fitness: float,
        domain: str = 'act'
    ) -> jnp.ndarray:
        """Apply STDP updates with extended windows.

        LTP: If activity preceded fitness improvement
        LTD: If activity preceded fitness decrease (with floor)
        """
        if len(fitness_history) < 2:
            return stdp_weights

        # Determine overall fitness trend
        recent_trend = fitness_history[-1] - fitness_history[-2] if len(fitness_history) >= 2 else 0

        for i in active_indices:
            num_funcs = NUM_ACTIVATIONS if domain == 'act' else NUM_AGGREGATIONS
            if 0 <= i < num_funcs:
                current_weight = float(stdp_weights[i])

                # Get this function's activity history
                hist = activity_history.get(i, [])
                recent_activity = [h for h in hist if generation - h['gen'] <= self.ltp_window]

                if recent_activity:
                    # Function was active in LTP window - check for LTP/LTD
                    avg_fitness_when_active = sum(h['fitness'] for h in recent_activity) / len(recent_activity)

                    if current_fitness > avg_fitness_when_active:
                        # LTP: fitness improved since this function was active
                        delta = self.ltp_rate * (current_fitness - avg_fitness_when_active)
                        new_weight = min(1.0, current_weight + delta)
                    else:
                        # LTD: fitness decreased since this function was active
                        # But check if we're in LTD window and apply gently
                        recent_ltd = [h for h in hist if generation - h['gen'] <= self.ltd_window]
                        if recent_ltd:
                            delta = self.ltd_rate * (avg_fitness_when_active - current_fitness)
                            new_weight = max(self.ltd_min_weight, current_weight - delta)
                        else:
                            new_weight = current_weight
                else:
                    new_weight = current_weight

                stdp_weights = stdp_weights.at[i].set(new_weight)

        return stdp_weights

    def update(
        self,
        state: Dict[str, Any],
        fitness_scores: jnp.ndarray,
        function_usage: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update with extended STDP windows."""
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
        if len(fitness_history) > self.ltp_window + self.ltd_window + 5:
            fitness_history = fitness_history[-(self.ltp_window + self.ltd_window + 5):]

        # Get current active functions
        act_indices = mask_to_indices(state['act_mask'])
        agg_indices = mask_to_indices(state['agg_mask'])

        # Record activity
        act_activity_history = dict(state.get('act_activity_history', {}))
        agg_activity_history = dict(state.get('agg_activity_history', {}))

        for i in act_indices:
            if i not in act_activity_history:
                act_activity_history[i] = []
            act_activity_history[i].append({'gen': generation, 'fitness': best_fitness})
            # Keep limited history
            if len(act_activity_history[i]) > self.ltp_window + self.ltd_window + 5:
                act_activity_history[i] = act_activity_history[i][-(self.ltp_window + self.ltd_window + 5):]

        for i in agg_indices:
            if i not in agg_activity_history:
                agg_activity_history[i] = []
            agg_activity_history[i].append({'gen': generation, 'fitness': best_fitness})
            if len(agg_activity_history[i]) > self.ltp_window + self.ltd_window + 5:
                agg_activity_history[i] = agg_activity_history[i][-(self.ltp_window + self.ltd_window + 5):]

        # Apply STDP updates
        act_stdp_weights = self._apply_stdp_update(
            state['act_stdp_weights'],
            act_activity_history,
            fitness_history,
            generation,
            act_indices,
            best_fitness,
            domain='act'
        )
        agg_stdp_weights = self._apply_stdp_update(
            state['agg_stdp_weights'],
            agg_activity_history,
            fitness_history,
            generation,
            agg_indices,
            best_fitness,
            domain='agg'
        )

        # Decay tags
        act_tags = state['act_tags'] * self.tag_decay
        agg_tags = state['agg_tags'] * self.tag_decay

        # Update tags from high STDP weight
        for i in range(NUM_ACTIVATIONS):
            if float(act_stdp_weights[i]) > self.tag_threshold:
                act_tags = act_tags.at[i].set(
                    min(1.0, float(act_tags[i]) + 0.2)
                )

        for i in range(NUM_AGGREGATIONS):
            threshold = self.agg_tag_threshold
            if i in CORE_EXTREME_AGGS:
                threshold *= 0.8
            if float(agg_stdp_weights[i]) > threshold:
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

        state['act_stdp_weights'] = act_stdp_weights
        state['agg_stdp_weights'] = agg_stdp_weights
        state['fitness_history'] = fitness_history
        state['act_activity_history'] = act_activity_history
        state['agg_activity_history'] = agg_activity_history
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

        # Activation mutations - add high-STDP weight functions
        if jax.random.uniform(k1) < mutation_rate:
            inactive_acts = [i for i in range(NUM_ACTIVATIONS)
                           if float(act_mask[i]) < 0.5]
            if inactive_acts:
                # Weight by STDP weight
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
            'min_stdp_weight': float(state['agg_stdp_weights'][3]),
        }
