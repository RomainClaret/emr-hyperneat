"""Strategy 45: Critical Period Refined (Multiple Sensitive Periods).

Implements refined critical periods for palette evolution with multiple
overlapping sensitive periods, gradual closure (not binary), and the
ability to reopen periods with strong performance signals.

Biological Basis:
- Neocortex has multiple critical periods for different modalities
- Critical periods overlap and are developmentally staged
- Closure is gradual, not binary (sensitivity declines smoothly)
- Strong environmental signals can partially reopen closed periods
- Early periods favor basic features, late periods favor specialization
- Different brain regions have different critical period timing

Key Insight:
- Previous critical period strategies use single binary periods
- Real development has multiple overlapping windows
- Gradual closure provides smoother transitions
- Reopening capability enables recovery from suboptimal closures
- Function categories can have different optimal periods

Refined Mechanism:
    # Multiple periods with overlap
    periods = [
        {'name': 'oscillatory', 'start': 0, 'end': 20, 'functions': [sin, burst]},
        {'name': 'bounded', 'start': 10, 'end': 35, 'functions': [tanh, sigmoid]},
        {'name': 'specialized', 'start': 25, 'end': 60, 'functions': [gauss, abs]},
    ]

    # Gradual closure
    for period in periods:
        openness = compute_openness(generation, period)  # 0 to 1
        mutation_rate[period.functions] *= openness

    # Reopening on strong signal
    if fitness_delta > reopening_threshold:
        for period in recently_closed_periods:
            openness[period] = min(openness[period] + reopening_boost, 1.0)

Expected improvements:
- More biologically realistic developmental trajectory
- Better exploration of different function types at optimal times
- Graceful transitions instead of abrupt closures
- Recovery capability from premature closures
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    DEFAULT_PALETTE_INDICES,
)


# Function categories for critical periods
# Each category represents a type of activation function
FUNCTION_CATEGORIES = {
    'oscillatory': [4],        # sin (periodic/oscillatory functions)
    'bounded': [0, 1],         # tanh, sigmoid (bounded outputs)
    'unbounded': [2, 6],       # relu, elu (unbounded positive)
    'specialized': [3, 5],     # identity, abs (specialized operations)
    'nonlinear': [7],          # gauss (local nonlinear)
}


class CriticalPeriodRefinedStrategy(PaletteEvolutionStrategy):
    """Multiple overlapping critical periods with gradual closure.

    Implements biologically-inspired sensitive periods where different
    function categories have optimal windows for exploration. Periods
    close gradually and can reopen with strong performance signals.
    """

    name = "critical_period_refined"
    description = "Multiple overlapping critical periods with gradual closure and reopening"

    def __init__(
        self,
        # Period definitions (list of dicts with 'name', 'start', 'peak', 'end', 'functions')
        periods: List[Dict] = None,
        # Closure dynamics
        base_openness: float = 1.0,           # Initial openness level
        closure_rate: float = 0.05,           # Per-gen closure after peak
        min_openness: float = 0.1,            # Minimum openness (never fully closed)
        # Reopening dynamics
        reopening_enabled: bool = True,
        reopening_threshold: float = 0.15,    # Fitness delta to trigger reopening
        reopening_boost: float = 0.3,         # Openness boost on reopening
        reopening_duration: int = 5,          # Gens of reopened state
        reopening_cooldown: int = 10,         # Gens before can reopen again
        # Exploration modulation
        open_mutation_rate: float = 0.15,     # Mutation rate when open
        closed_mutation_rate: float = 0.02,   # Mutation rate when closed
        # Global settings
        base_mutation_rate: float = 0.1,
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Critical Period Refined strategy.

        Args:
            periods: List of period definitions (default: standard development)
            base_openness: Initial openness for all periods
            closure_rate: Rate of closure after peak
            min_openness: Minimum openness (periods never fully close)
            reopening_enabled: Allow periods to reopen
            reopening_threshold: Fitness improvement needed to trigger
            reopening_boost: How much openness increases on reopen
            reopening_duration: Generations the period stays reopened
            reopening_cooldown: Minimum gens between reopenings
        """
        # Default periods based on developmental neuroscience
        self.periods = periods or [
            {
                'name': 'oscillatory',
                'start': 0, 'peak': 8, 'end': 25,
                'functions': FUNCTION_CATEGORIES['oscillatory'],
                'description': 'Early: sin/oscillatory functions',
            },
            {
                'name': 'bounded',
                'start': 5, 'peak': 20, 'end': 45,
                'functions': FUNCTION_CATEGORIES['bounded'],
                'description': 'Mid: tanh/sigmoid bounded activations',
            },
            {
                'name': 'unbounded',
                'start': 15, 'peak': 35, 'end': 60,
                'functions': FUNCTION_CATEGORIES['unbounded'],
                'description': 'Later: relu/elu unbounded activations',
            },
            {
                'name': 'specialized',
                'start': 25, 'peak': 45, 'end': 80,
                'functions': FUNCTION_CATEGORIES['specialized'],
                'description': 'Late: identity/abs specialized',
            },
            {
                'name': 'nonlinear',
                'start': 30, 'peak': 50, 'end': 100,
                'functions': FUNCTION_CATEGORIES['nonlinear'],
                'description': 'Latest: gaussian local nonlinear',
            },
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

        # Exploration
        self.open_mutation_rate = open_mutation_rate
        self.closed_mutation_rate = closed_mutation_rate

        # General
        self.base_mutation_rate = base_mutation_rate
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with critical period tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Period states
        period_openness = {p['name']: self.base_openness for p in self.periods}
        reopened_until = {p['name']: -1 for p in self.periods}  # Gen until which reopened
        last_reopen = {p['name']: -self.reopening_cooldown for p in self.periods}

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 454545),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Period states
            'period_openness': period_openness,     # {period_name: openness}
            'reopened_until': reopened_until,       # {period_name: gen or -1}
            'last_reopen': last_reopen,             # {period_name: last_reopen_gen}
            # Function-level tracking
            'function_openness': jnp.ones(NUM_ACTIVATIONS),
            'function_usage': jnp.zeros(NUM_ACTIVATIONS),
            # History
            'reopen_events': [],                    # (gen, period_name, trigger_delta)
            'closure_events': [],                   # (gen, period_name)
            'previous_mask': mask,
            'fitness_history': [],
            # Stats
            'total_reopens': 0,
            'periods_currently_open': len(self.periods),
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette."""
        return mask_to_indices(state['mask'])

    def _compute_period_openness(
        self,
        period: Dict,
        generation: int,
        current_openness: float,
        is_reopened: bool,
    ) -> float:
        """Compute natural openness for a period based on developmental stage."""
        start = period['start']
        peak = period['peak']
        end = period['end']

        if generation < start:
            # Before period starts
            return 0.0
        elif generation < peak:
            # Rising phase: linear increase to peak
            progress = (generation - start) / (peak - start)
            natural_openness = progress * self.base_openness
        elif generation < end:
            # Falling phase: gradual closure
            progress = (generation - peak) / (end - peak)
            natural_openness = self.base_openness * (1 - progress * (1 - self.min_openness))
        else:
            # After period ends
            natural_openness = self.min_openness

        # If reopened, boost openness
        if is_reopened:
            natural_openness = min(natural_openness + self.reopening_boost, self.base_openness)

        return max(natural_openness, self.min_openness)

    def _update_all_period_openness(
        self,
        period_openness: Dict[str, float],
        reopened_until: Dict[str, int],
        generation: int,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Update openness for all periods."""
        new_openness = {}
        newly_closed = []

        for period in self.periods:
            name = period['name']
            is_reopened = reopened_until[name] >= generation

            old_openness = period_openness[name]
            new_val = self._compute_period_openness(
                period, generation, old_openness, is_reopened
            )
            new_openness[name] = new_val

            # Track closure events
            if old_openness > 0.5 and new_val <= 0.5:
                newly_closed.append(name)

        return new_openness, newly_closed

    def _compute_function_openness(
        self,
        period_openness: Dict[str, float],
    ) -> jnp.ndarray:
        """Compute per-function openness based on period states."""
        func_openness = jnp.ones(NUM_ACTIVATIONS) * self.min_openness

        for period in self.periods:
            name = period['name']
            openness = period_openness[name]
            for func in period['functions']:
                if 0 <= func < NUM_ACTIVATIONS:
                    current = float(func_openness[func])
                    # Take maximum openness from any period containing this function
                    func_openness = func_openness.at[func].set(max(current, openness))

        return func_openness

    def _check_for_reopening(
        self,
        period_openness: Dict[str, float],
        reopened_until: Dict[str, int],
        last_reopen: Dict[str, int],
        fitness_delta: float,
        generation: int,
    ) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
        """Check if any periods should reopen based on fitness signal."""
        new_reopened_until = dict(reopened_until)
        new_last_reopen = dict(last_reopen)
        reopened = []

        if not self.reopening_enabled:
            return new_reopened_until, new_last_reopen, reopened

        if fitness_delta < self.reopening_threshold:
            return new_reopened_until, new_last_reopen, reopened

        # Strong signal - check which periods can reopen
        for period in self.periods:
            name = period['name']
            openness = period_openness[name]

            # Can only reopen if:
            # 1. Period is partially closed (openness < 0.5)
            # 2. Cooldown has passed
            if openness < 0.5 and (generation - last_reopen[name]) >= self.reopening_cooldown:
                new_reopened_until[name] = generation + self.reopening_duration
                new_last_reopen[name] = generation
                reopened.append(name)

        return new_reopened_until, new_last_reopen, reopened

    def _mutate_palette(
        self,
        mask: jnp.ndarray,
        func_openness: jnp.ndarray,
        func_usage: jnp.ndarray,
        stagnation: int,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Mutate palette based on period openness."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()
        new_usage = func_usage.copy()

        current_palette = mask_to_indices(mask)

        # Base mutation rate adjusted by stagnation
        mutation_rate = self.base_mutation_rate * (1 + stagnation * 0.1)

        if jax.random.uniform(key1) < mutation_rate:
            # Try to remove a function
            if len(current_palette) > self.min_active:
                # Weight removal by inverse openness (more closed = more likely to remove)
                removal_weights = []
                for func in current_palette:
                    openness = float(func_openness[func])
                    # Lower openness = higher removal probability
                    weight = 1.0 - openness + 0.1  # +0.1 to avoid zero
                    removal_weights.append(weight)

                # Normalize
                total = sum(removal_weights)
                removal_probs = [w / total for w in removal_weights]

                # Sample
                cum_prob = 0
                sample = float(jax.random.uniform(key2))
                for i, prob in enumerate(removal_probs):
                    cum_prob += prob
                    if sample < cum_prob:
                        removed = current_palette[i]
                        new_mask = new_mask.at[removed].set(0.0)
                        break

            # Add a new function
            available = [i for i in range(NUM_ACTIVATIONS) if new_mask[i] < 0.5]
            if available:
                # Weight addition by openness (more open = more likely to add)
                add_weights = []
                for func in available:
                    openness = float(func_openness[func])
                    weight = openness + 0.1  # +0.1 to allow minimal chance
                    add_weights.append(weight)

                # Normalize
                total = sum(add_weights)
                add_probs = [w / total for w in add_weights]

                # Sample
                cum_prob = 0
                sample = float(jax.random.uniform(key3))
                for i, prob in enumerate(add_probs):
                    cum_prob += prob
                    if sample < cum_prob:
                        added = available[i]
                        new_mask = new_mask.at[added].set(1.0)
                        new_usage = new_usage.at[added].add(1.0)
                        break

        return new_mask, new_usage

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with critical period dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness if improved else 0.0

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Check for reopening
        new_reopened, new_last_reopen, reopened_periods = self._check_for_reopening(
            state['period_openness'],
            state['reopened_until'],
            state['last_reopen'],
            fitness_delta,
            generation,
        )

        # Step 2: Update period openness
        new_period_openness, newly_closed = self._update_all_period_openness(
            state['period_openness'],
            new_reopened,
            generation,
        )

        # Step 3: Compute function-level openness
        new_func_openness = self._compute_function_openness(new_period_openness)

        # Step 4: Mutate palette
        new_mask, new_usage = self._mutate_palette(
            state['mask'],
            new_func_openness,
            state['function_usage'],
            new_stagnation,
            k1,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update history
        reopen_events = list(state['reopen_events'])
        for period_name in reopened_periods:
            reopen_events.append((generation, period_name, fitness_delta))
        if len(reopen_events) > 50:
            reopen_events = reopen_events[-50:]

        closure_events = list(state['closure_events'])
        for period_name in newly_closed:
            closure_events.append((generation, period_name))
        if len(closure_events) > 50:
            closure_events = closure_events[-50:]

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        # Count open periods
        open_count = sum(1 for o in new_period_openness.values() if o > 0.5)

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Period states
            'period_openness': new_period_openness,
            'reopened_until': new_reopened,
            'last_reopen': new_last_reopen,
            # Function tracking
            'function_openness': new_func_openness,
            'function_usage': new_usage,
            # History
            'reopen_events': reopen_events,
            'closure_events': closure_events,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Stats
            'total_reopens': state['total_reopens'] + len(reopened_periods),
            'periods_currently_open': open_count,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Period status
        period_status = [
            (p['name'], new_period_openness[p['name']], new_reopened[p['name']] >= generation)
            for p in self.periods
        ]

        # Function openness ranking
        top_open_idx = jnp.argsort(new_func_openness)[-5:][::-1]
        top_openness = [(int(i), float(new_func_openness[i])) for i in top_open_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Periods
            'period_status': period_status,
            'periods_open': open_count,
            'periods_reopened_this_gen': reopened_periods,
            'periods_closed_this_gen': newly_closed,
            # Function openness
            'top_openness': top_openness,
            'mean_openness': float(jnp.mean(new_func_openness)),
            # Stats
            'total_reopens': new_state['total_reopens'],
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_openness': float(new_func_openness[4]),
            'sin_period_status': new_period_openness.get('oscillatory', 0),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with period status."""
        palette = self.get_active_palette(state)
        period_openness = state['period_openness']
        func_openness = state['function_openness']

        # Period details
        period_details = [
            (p['name'], period_openness[p['name']])
            for p in self.periods
        ]

        # Top function openness
        top_idx = jnp.argsort(func_openness)[-5:][::-1]
        top_openness = [(int(i), float(func_openness[i])) for i in top_idx]

        # Count open periods
        open_count = sum(1 for o in period_openness.values() if o > 0.5)

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Periods
            'period_details': period_details,
            'periods_open': open_count,
            'total_reopens': state['total_reopens'],
            # Function openness
            'top_openness': top_openness,
            'mean_openness': float(jnp.mean(func_openness)),
            # Sin-specific
            'sin_openness': float(func_openness[4]),
        }
