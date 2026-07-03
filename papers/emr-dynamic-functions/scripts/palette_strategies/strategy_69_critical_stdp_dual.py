"""Strategy 69: Critical Period + STDP Dual ("Developmental Causality").

Hybrid combining Critical Period developmental windows with STDP temporal credit.

Key synergies:
1. Critical periods define when plasticity is high vs low
2. STDP assigns temporal credit to functions that precede success
3. NOVEL: Credit window SIZE shrinks as critical period closes
   - Early (open period): wide credit window (10 gens) - learn broad associations
   - Late (closed period): narrow credit window (2 gens) - only immediate causality
4. Functions that accumulate credit during open periods are protected during closure

Biological basis:
- Critical periods in development have enhanced STDP-like plasticity
- Early sensory learning has broad temporal integration windows
- Mature circuits have precise, narrow timing requirements
- Combines developmental gating with causal learning

Expected: Early broad exploration followed by precise causal refinement
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


class CriticalSTDPDualStrategy(PaletteEvolutionStrategy):
    """Critical period gates STDP credit window size.

    During open critical periods, STDP uses wide temporal windows.
    As periods close, windows narrow to enforce precise causality.
    """

    name = "critical_stdp_dual"
    description = "Critical period modulates STDP credit window size"

    def __init__(
        self,
        # Critical period timing
        critical_period_end: int = 60,
        closure_rate: float = 0.95,  # Per 10 gens
        min_openness: float = 0.1,
        # STDP credit window (dynamic)
        initial_credit_window: int = 10,
        final_credit_window: int = 2,
        # STDP parameters
        ltp_rate: float = 0.25,
        ltd_rate: float = 0.12,
        temporal_decay: float = 0.7,
        history_length: int = 15,
        # Mutation rates (vary with openness)
        open_mutation_rate: float = 0.25,
        closed_mutation_rate: float = 0.03,
        # Protection
        credit_protection_threshold: float = 0.5,  # Protect high-credit functions
        # Cross-domain
        cross_learning_rate: float = 0.10,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        # Critical period
        self.critical_period_end = critical_period_end
        self.closure_rate = closure_rate
        self.min_openness = min_openness

        # Dynamic credit window
        self.initial_credit_window = initial_credit_window
        self.final_credit_window = final_credit_window

        # STDP
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.temporal_decay = temporal_decay
        self.history_length = history_length

        # Mutation
        self.open_mutation_rate = open_mutation_rate
        self.closed_mutation_rate = closed_mutation_rate

        # Protection
        self.credit_protection_threshold = credit_protection_threshold

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _compute_openness(self, generation: int) -> float:
        """Compute critical period openness (1.0 = fully open, min = closed)."""
        if generation >= self.critical_period_end:
            return self.min_openness

        # Gradual closure
        progress = generation / self.critical_period_end
        openness = (self.closure_rate ** (progress * 10))  # Exponential closure
        return max(self.min_openness, openness)

    def _compute_credit_window(self, openness: float) -> int:
        """Compute STDP credit window size based on openness."""
        # Interpolate between initial and final window
        window_range = self.initial_credit_window - self.final_credit_window
        dynamic_window = self.final_credit_window + int(window_range * openness)
        return max(self.final_credit_window, dynamic_window)

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize with Critical Period + STDP state."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_credit': jnp.zeros(NUM_ACTIVATIONS),
            'act_cumulative_credit': jnp.zeros(NUM_ACTIVATIONS),  # Track total credit earned
            'act_history': [],

            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_credit': jnp.zeros(NUM_AGGREGATIONS),
            'agg_cumulative_credit': jnp.zeros(NUM_AGGREGATIONS),
            'agg_history': [],

            # Cross-domain
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,

            # Critical period state
            'openness': 1.0,
            'current_credit_window': self.initial_credit_window,

            # General state
            'rng_key': jax.random.PRNGKey(seed + 696969),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,

            # Stats
            'total_ltp_events': 0,
            'total_ltd_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_temporal_weight(self, gens_before: int, credit_window: int) -> float:
        """Compute temporal weight with dynamic window."""
        if gens_before > credit_window:
            return 0.0
        return self.temporal_decay ** abs(gens_before)

    def _stdp_update_with_dynamic_window(
        self,
        credit: jnp.ndarray,
        cumulative_credit: jnp.ndarray,
        history: List[Tuple[int, jnp.ndarray, float]],
        current_gen: int,
        improved: bool,
        improvement_magnitude: float,
        credit_window: int,
        openness: float,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Apply STDP with dynamic credit window based on openness."""
        new_credit = credit * 0.85  # Faster decay during open period, slower during closed
        if openness < 0.5:
            new_credit = credit * 0.95  # Slower decay when closed

        new_cumulative = cumulative_credit.copy()
        ltp_count, ltd_count = 0, 0

        if improved and len(history) >= 2:
            # LTP with dynamic window
            for hist_gen, hist_mask, _ in history:
                gens_before = current_gen - hist_gen

                # Only credit if within dynamic window
                if 1 <= gens_before <= credit_window:
                    temporal_weight = self._compute_temporal_weight(gens_before, credit_window)

                    # Scale LTP by openness (more plastic when open)
                    effective_ltp = self.ltp_rate * (0.5 + 0.5 * openness)

                    ltp_delta = (
                        effective_ltp * temporal_weight * improvement_magnitude *
                        (hist_mask > 0.5).astype(jnp.float32)
                    )
                    new_credit = jnp.clip(new_credit + ltp_delta, 0.0, 1.0)

                    # Accumulate credit for protection
                    new_cumulative = new_cumulative + ltp_delta

                    ltp_count += int(jnp.sum(hist_mask > 0.5))

        elif not improved and len(history) >= 2:
            # LTD (also scaled by openness)
            recent_mask = history[-1][1] if history else jnp.zeros(n_funcs)
            effective_ltd = self.ltd_rate * (0.3 + 0.7 * openness)

            ltd_delta = effective_ltd * 0.5 * (recent_mask > 0.5).astype(jnp.float32)
            new_credit = jnp.clip(new_credit - ltd_delta, 0.0, 1.0)
            ltd_count = int(jnp.sum(recent_mask > 0.5))

        return new_credit, new_cumulative, ltp_count, ltd_count

    def _compute_protection_score(
        self,
        credit: jnp.ndarray,
        cumulative_credit: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        other_mask: jnp.ndarray,
        is_act: bool,
    ) -> jnp.ndarray:
        """Compute protection score combining current and cumulative credit."""
        # Base protection from cumulative credit
        max_cumulative = max(float(jnp.max(cumulative_credit)), 0.1)
        normalized_cumulative = cumulative_credit / max_cumulative

        # Current credit
        current_contrib = credit

        # Cross-domain contribution
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)
        if is_act:
            cross_score = jnp.dot(cross_affinity, other_active) / n_other
        else:
            cross_score = jnp.dot(cross_affinity.T, other_active) / n_other

        # Combine
        return 0.4 * current_contrib + 0.4 * normalized_cumulative + 0.2 * cross_score

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        openness: float,
        n_funcs: int,
        min_active: int,
        max_active: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Mutate palette with openness-dependent rates."""
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []

        # Mutation rate varies with openness
        mutation_rate = (
            self.open_mutation_rate * openness +
            self.closed_mutation_rate * (1 - openness)
        )

        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))
        current = int(jnp.sum(mask > 0.5))

        for i in range(n_funcs):
            prot = float(protection[i])

            if mask[i] < 0.5:
                # Inactive - might activate (easier when open)
                if current + len(activated) >= max_active:
                    continue
                # Higher openness = more willing to try new functions
                eff_rate = mutation_rate * (0.3 + 0.7 * openness + 0.3 * prot)
                if act_probs[i] < eff_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                # High protection = hard to remove, especially when closed
                if prot >= self.credit_protection_threshold:
                    # Protected by accumulated credit
                    dr = mutation_rate * 0.05 * (1 - prot)
                else:
                    dr = mutation_rate * (1.0 - prot)

                # Harder to deactivate when period is closed
                dr = dr * (0.2 + 0.8 * openness)

                if deact_probs[i] < dr:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum
        if jnp.sum(new_mask > 0.5) < min_active:
            return mask, {'activated': [], 'deactivated': []}

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with Critical Period + STDP dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        improvement_magnitude = max(0.0, min(1.0,
            (best_fitness - prev_best_fitness) / max(0.1, prev_best_fitness)
        ))

        # Compute current openness and dynamic credit window
        openness = self._compute_openness(generation)
        credit_window = self._compute_credit_window(openness)

        # Update histories
        act_history = state['act_history'] + [(generation, state['act_mask'].copy(), best_fitness)]
        if len(act_history) > self.history_length:
            act_history = act_history[-self.history_length:]

        agg_history = state['agg_history'] + [(generation, state['agg_mask'].copy(), best_fitness)]
        if len(agg_history) > self.history_length:
            agg_history = agg_history[-self.history_length:]

        # STDP updates with dynamic window
        new_act_credit, new_act_cumulative, act_ltp, act_ltd = self._stdp_update_with_dynamic_window(
            state['act_credit'], state['act_cumulative_credit'], act_history,
            generation, improved, improvement_magnitude, credit_window, openness, NUM_ACTIVATIONS
        )
        new_agg_credit, new_agg_cumulative, agg_ltp, agg_ltd = self._stdp_update_with_dynamic_window(
            state['agg_credit'], state['agg_cumulative_credit'], agg_history,
            generation, improved, improvement_magnitude, credit_window, openness, NUM_AGGREGATIONS
        )

        # Update cross-domain affinity
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        fitness_delta = best_fitness - prev_best_fitness

        cross_delta = self.cross_learning_rate * fitness_delta * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Compute protection scores
        act_protection = self._compute_protection_score(
            new_act_credit, new_act_cumulative, new_cross, state['agg_mask'], True
        )
        agg_protection = self._compute_protection_score(
            new_agg_credit, new_agg_cumulative, new_cross, state['act_mask'], False
        )

        # Mutate palettes
        new_act_mask, act_mut = self._mutate_palette(
            k_act, state['act_mask'], act_protection, openness,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act
        )
        new_agg_mask, agg_mut = self._mutate_palette(
            k_agg, state['agg_mask'], agg_protection, openness,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg
        )

        new_state = {
            'act_mask': new_act_mask,
            'act_credit': new_act_credit,
            'act_cumulative_credit': new_act_cumulative,
            'act_history': act_history,
            'agg_mask': new_agg_mask,
            'agg_credit': new_agg_credit,
            'agg_cumulative_credit': new_agg_cumulative,
            'agg_history': agg_history,
            'cross_affinity': new_cross,
            'openness': openness,
            'current_credit_window': credit_window,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'total_ltp_events': state['total_ltp_events'] + act_ltp + agg_ltp,
            'total_ltd_events': state['total_ltd_events'] + act_ltd + agg_ltd,
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
            # Critical period stats
            'openness': openness,
            'credit_window': credit_window,
            # STDP stats
            'sin_credit': float(new_act_credit[4]),
            'sin_cumulative_credit': float(new_act_cumulative[4]),
            'act_mean_credit': float(jnp.mean(new_act_credit)),
            'agg_mean_credit': float(jnp.mean(new_agg_credit)),
            'total_ltp_events': new_state['total_ltp_events'],
            'total_ltd_events': new_state['total_ltd_events'],
            # Sin status
            'has_sin': 4 in act_palette,
        }
        metrics.update({f'act_{k}': v for k, v in act_mut.items()})
        metrics.update({f'agg_{k}': v for k, v in agg_mut.items()})

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with Critical Period + STDP stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'openness': state['openness'],
            'credit_window': state['current_credit_window'],
            'generation': state['generation'],
            'sin_credit': float(state['act_credit'][4]),
            'sin_cumulative_credit': float(state['act_cumulative_credit'][4]),
            'total_ltp_events': state['total_ltp_events'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
