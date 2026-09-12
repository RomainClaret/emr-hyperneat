"""Strategy 64D: Critical Period Refined Dual (Multiple Sensitive Periods for Both Domains).

Extends CriticalPeriodRefinedStrategy to jointly evolve BOTH activation AND
aggregation function palettes with overlapping critical periods.

Key dual mechanisms:
1. Dual period tracking - separate periods for act and agg domains
2. Cross-domain reopening - success can reopen periods in both domains
3. Synchronized developmental stages - coordinated openness dynamics
4. Domain-specific function categories

Expected: Developmental trajectory optimization in both domains
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


# Function categories for activation domain
ACT_FUNCTION_CATEGORIES = {
    'oscillatory': [4],
    'bounded': [0, 1],
    'unbounded': [2, 6],
    'specialized': [3, 5],
    'nonlinear': [7],
}

# Aggregation categories
AGG_FUNCTION_CATEGORIES = {
    'basic': [0, 1],        # sum, mean
    'extreme': [2, 3],      # max, min
    'statistical': [4, 5],  # std, var (if available)
}


class CriticalPeriodRefinedDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with multiple overlapping critical periods.

    Both activation and aggregation domains have separate critical periods
    with gradual closure and reopening capability.
    """

    name = "critical_period_refined_dual"
    description = "Dual: Multiple critical periods with cross-domain reopening"

    def __init__(
        self,
        # Activation periods
        act_periods: List[Dict] = None,
        # Aggregation periods
        agg_periods: List[Dict] = None,
        # Closure dynamics
        base_openness: float = 1.0,
        closure_rate: float = 0.05,
        min_openness: float = 0.1,
        # Reopening
        reopening_enabled: bool = True,
        reopening_threshold: float = 0.15,
        reopening_boost: float = 0.3,
        reopening_duration: int = 5,
        reopening_cooldown: int = 10,
        cross_reopen_enabled: bool = True,
        # Exploration
        open_mutation_rate: float = 0.15,
        closed_mutation_rate: float = 0.02,
        base_mutation_rate: float = 0.1,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Critical Period Refined Dual strategy."""
        # Activation periods
        self.act_periods = act_periods or [
            {'name': 'oscillatory', 'start': 0, 'peak': 8, 'end': 25, 'functions': ACT_FUNCTION_CATEGORIES['oscillatory']},
            {'name': 'bounded', 'start': 5, 'peak': 20, 'end': 45, 'functions': ACT_FUNCTION_CATEGORIES['bounded']},
            {'name': 'unbounded', 'start': 15, 'peak': 35, 'end': 60, 'functions': ACT_FUNCTION_CATEGORIES['unbounded']},
            {'name': 'specialized', 'start': 25, 'peak': 45, 'end': 80, 'functions': ACT_FUNCTION_CATEGORIES['specialized']},
        ]

        # Aggregation periods (simpler, fewer categories)
        self.agg_periods = agg_periods or [
            {'name': 'basic', 'start': 0, 'peak': 10, 'end': 40, 'functions': AGG_FUNCTION_CATEGORIES['basic']},
            {'name': 'extreme', 'start': 20, 'peak': 40, 'end': 70, 'functions': AGG_FUNCTION_CATEGORIES['extreme']},
        ]

        # Closure
        self.base_openness = base_openness
        self.closure_rate = closure_rate
        self.min_openness = min_openness

        # Reopening
        self.reopening_enabled = reopening_enabled
        self.reopening_threshold = reopening_threshold
        self.reopening_boost = reopening_boost
        self.reopening_duration = reopening_duration
        self.reopening_cooldown = reopening_cooldown
        self.cross_reopen_enabled = cross_reopen_enabled

        # Exploration
        self.open_mutation_rate = open_mutation_rate
        self.closed_mutation_rate = closed_mutation_rate
        self.base_mutation_rate = base_mutation_rate

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual critical period tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Period states for activation
        act_period_openness = {p['name']: self.base_openness for p in self.act_periods}
        act_reopened_until = {p['name']: -1 for p in self.act_periods}
        act_last_reopen = {p['name']: -self.reopening_cooldown for p in self.act_periods}

        # Period states for aggregation
        agg_period_openness = {p['name']: self.base_openness for p in self.agg_periods}
        agg_reopened_until = {p['name']: -1 for p in self.agg_periods}
        agg_last_reopen = {p['name']: -self.reopening_cooldown for p in self.agg_periods}

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_period_openness': act_period_openness,
            'act_reopened_until': act_reopened_until,
            'act_last_reopen': act_last_reopen,
            'act_func_openness': jnp.ones(NUM_ACTIVATIONS),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_period_openness': agg_period_openness,
            'agg_reopened_until': agg_reopened_until,
            'agg_last_reopen': agg_last_reopen,
            'agg_func_openness': jnp.ones(NUM_AGGREGATIONS),
            # General state
            'rng_key': jax.random.PRNGKey(seed + 646464),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Stats
            'act_total_reopens': 0,
            'agg_total_reopens': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_period_openness(
        self,
        period: Dict,
        generation: int,
        is_reopened: bool,
    ) -> float:
        """Compute natural openness for a period."""
        start = period['start']
        peak = period['peak']
        end = period['end']

        if generation < start:
            natural = 0.0
        elif generation < peak:
            progress = (generation - start) / (peak - start)
            natural = progress * self.base_openness
        elif generation < end:
            progress = (generation - peak) / (end - peak)
            natural = self.base_openness * (1 - progress * (1 - self.min_openness))
        else:
            natural = self.min_openness

        if is_reopened:
            natural = min(natural + self.reopening_boost, self.base_openness)

        return max(natural, self.min_openness)

    def _update_all_period_openness(
        self,
        periods: List[Dict],
        period_openness: Dict[str, float],
        reopened_until: Dict[str, int],
        generation: int,
    ) -> Dict[str, float]:
        """Update openness for all periods in a domain."""
        new_openness = {}
        for period in periods:
            name = period['name']
            is_reopened = reopened_until[name] >= generation
            new_openness[name] = self._compute_period_openness(period, generation, is_reopened)
        return new_openness

    def _compute_function_openness(
        self,
        periods: List[Dict],
        period_openness: Dict[str, float],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute per-function openness."""
        func_openness = jnp.ones(n_funcs) * self.min_openness
        for period in periods:
            name = period['name']
            openness = period_openness[name]
            for func in period['functions']:
                if 0 <= func < n_funcs:
                    current = float(func_openness[func])
                    func_openness = func_openness.at[func].set(max(current, openness))
        return func_openness

    def _check_for_reopening(
        self,
        periods: List[Dict],
        period_openness: Dict[str, float],
        reopened_until: Dict[str, int],
        last_reopen: Dict[str, int],
        fitness_delta: float,
        generation: int,
    ) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
        """Check for period reopening."""
        new_reopened = dict(reopened_until)
        new_last = dict(last_reopen)
        reopened = []

        if not self.reopening_enabled or fitness_delta < self.reopening_threshold:
            return new_reopened, new_last, reopened

        for period in periods:
            name = period['name']
            if (period_openness[name] < 0.5 and
                (generation - last_reopen[name]) >= self.reopening_cooldown):
                new_reopened[name] = generation + self.reopening_duration
                new_last[name] = generation
                reopened.append(name)

        return new_reopened, new_last, reopened

    def _mutate_palette(
        self,
        mask: jnp.ndarray,
        func_openness: jnp.ndarray,
        stagnation: int,
        key: jax.random.PRNGKey,
        min_active: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Mutate palette based on openness."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()
        current_palette = mask_to_indices(mask)
        mutation_rate = self.base_mutation_rate * (1 + stagnation * 0.1)

        if jax.random.uniform(key1) < mutation_rate:
            if len(current_palette) > min_active:
                removal_weights = [(1.0 - float(func_openness[i]) + 0.1) for i in current_palette]
                total = sum(removal_weights)
                removal_probs = [w / total for w in removal_weights]
                cum_prob = 0
                sample = float(jax.random.uniform(key2))
                for i, prob in enumerate(removal_probs):
                    cum_prob += prob
                    if sample < cum_prob:
                        new_mask = new_mask.at[current_palette[i]].set(0.0)
                        break

            available = [i for i in range(n_funcs) if new_mask[i] < 0.5]
            if available:
                add_weights = [(float(func_openness[i]) + 0.1) for i in available]
                total = sum(add_weights)
                add_probs = [w / total for w in add_weights]
                cum_prob = 0
                sample = float(jax.random.uniform(key3))
                for i, prob in enumerate(add_probs):
                    cum_prob += prob
                    if sample < cum_prob:
                        new_mask = new_mask.at[available[i]].set(1.0)
                        break

        return new_mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual critical period dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness if improved else 0.0

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Check for reopening in both domains
        new_act_reopened, new_act_last, act_reopened = self._check_for_reopening(
            self.act_periods, state['act_period_openness'],
            state['act_reopened_until'], state['act_last_reopen'],
            fitness_delta, generation
        )
        new_agg_reopened, new_agg_last, agg_reopened = self._check_for_reopening(
            self.agg_periods, state['agg_period_openness'],
            state['agg_reopened_until'], state['agg_last_reopen'],
            fitness_delta, generation
        )

        # Cross-domain reopening
        if self.cross_reopen_enabled and fitness_delta >= self.reopening_threshold:
            if act_reopened and not agg_reopened:
                for period in self.agg_periods:
                    name = period['name']
                    if state['agg_period_openness'][name] < 0.5:
                        new_agg_reopened[name] = generation + self.reopening_duration // 2
                        agg_reopened.append(name)
                        break
            if agg_reopened and not act_reopened:
                for period in self.act_periods:
                    name = period['name']
                    if state['act_period_openness'][name] < 0.5:
                        new_act_reopened[name] = generation + self.reopening_duration // 2
                        act_reopened.append(name)
                        break

        # Update period openness
        new_act_period_openness = self._update_all_period_openness(
            self.act_periods, state['act_period_openness'], new_act_reopened, generation
        )
        new_agg_period_openness = self._update_all_period_openness(
            self.agg_periods, state['agg_period_openness'], new_agg_reopened, generation
        )

        # Compute function openness
        new_act_func_openness = self._compute_function_openness(
            self.act_periods, new_act_period_openness, NUM_ACTIVATIONS
        )
        new_agg_func_openness = self._compute_function_openness(
            self.agg_periods, new_agg_period_openness, NUM_AGGREGATIONS
        )

        # Mutate palettes
        new_act_mask = self._mutate_palette(
            state['act_mask'], new_act_func_openness, new_stagnation,
            k_act, self.min_active_act, NUM_ACTIVATIONS
        )
        new_agg_mask = self._mutate_palette(
            state['agg_mask'], new_agg_func_openness, new_stagnation,
            k_agg, self.min_active_agg, NUM_AGGREGATIONS
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_period_openness': new_act_period_openness,
            'act_reopened_until': new_act_reopened,
            'act_last_reopen': new_act_last,
            'act_func_openness': new_act_func_openness,
            'agg_mask': new_agg_mask,
            'agg_period_openness': new_agg_period_openness,
            'agg_reopened_until': new_agg_reopened,
            'agg_last_reopen': new_agg_last,
            'agg_func_openness': new_agg_func_openness,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'act_total_reopens': state['act_total_reopens'] + len(act_reopened),
            'agg_total_reopens': state['agg_total_reopens'] + len(agg_reopened),
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        act_open_count = sum(1 for o in new_act_period_openness.values() if o > 0.5)
        agg_open_count = sum(1 for o in new_agg_period_openness.values() if o > 0.5)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Periods
            'act_periods_open': act_open_count,
            'agg_periods_open': agg_open_count,
            'act_reopened_this_gen': act_reopened,
            'agg_reopened_this_gen': agg_reopened,
            # Openness
            'act_mean_openness': float(jnp.mean(new_act_func_openness)),
            'agg_mean_openness': float(jnp.mean(new_agg_func_openness)),
            # Stats
            'act_total_reopens': new_state['act_total_reopens'],
            'agg_total_reopens': new_state['agg_total_reopens'],
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_openness': float(new_act_func_openness[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual period status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        act_open = sum(1 for o in state['act_period_openness'].values() if o > 0.5)
        agg_open = sum(1 for o in state['agg_period_openness'].values() if o > 0.5)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_periods_open': act_open,
            'agg_periods_open': agg_open,
            'act_mean_openness': float(jnp.mean(state['act_func_openness'])),
            'agg_mean_openness': float(jnp.mean(state['agg_func_openness'])),
            'act_total_reopens': state['act_total_reopens'],
            'agg_total_reopens': state['agg_total_reopens'],
            'sin_openness': float(state['act_func_openness'][4]),
        }
