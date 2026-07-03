"""Strategy 83: STDP Aggregation Discovery Dual (Temporal Credit for Act-Agg Pairs).

Bio inspiration: Spike-timing-dependent plasticity - temporal order matters.
When an aggregation was active BEFORE a fitness improvement, it gets credit.
Extended to cross-domain: track activation-aggregation temporal relationships.

Key innovation:
- LTP (Long-Term Potentiation) for pairs active before improvement
- LTD (Long-Term Depression) for pairs active but no improvement
- Extreme aggregation LTP multiplier (1.4x)
- Cross-domain temporal credit assignment

Expected: Better temporal credit for activation-aggregation synergies.
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


class STDPAggDiscoveryDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with STDP-based temporal credit assignment.

    Tracks which activation-aggregation pairs were active before fitness
    improvements and strengthens their weights accordingly.
    """

    name = "stdp_agg_discovery_dual"
    description = "Dual: STDP temporal credit for act-agg pair discovery"

    def __init__(
        self,
        # STDP parameters
        ltp_window: int = 5,
        ltd_window: int = 3,
        agg_ltp_window: int = 4,  # Shorter for aggregations
        agg_ltd_window: int = 2,
        # Learning rates
        ltp_rate: float = 0.15,
        ltd_rate: float = 0.05,
        agg_ltp_rate: float = 0.20,  # Higher for aggregations
        agg_ltd_rate: float = 0.08,
        # Cross-domain STDP
        cross_ltp_rate: float = 0.15,
        cross_ltd_rate: float = 0.06,
        # Extreme aggregation boost
        extreme_ltp_multiplier: float = 1.4,
        # Mutation parameters
        base_activate_rate: float = 0.15,
        base_deactivate_rate: float = 0.10,
        weight_influence: float = 0.5,  # How much weights affect mutation
        # Stagnation
        stagnation_threshold: int = 5,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize STDP Aggregation Discovery Dual strategy."""
        # STDP windows
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.agg_ltp_window = agg_ltp_window
        self.agg_ltd_window = agg_ltd_window

        # Learning rates
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.agg_ltp_rate = agg_ltp_rate
        self.agg_ltd_rate = agg_ltd_rate

        # Cross-domain
        self.cross_ltp_rate = cross_ltp_rate
        self.cross_ltd_rate = cross_ltd_rate

        # Extreme boost
        self.extreme_ltp_multiplier = extreme_ltp_multiplier

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.weight_influence = weight_influence

        # Stagnation
        self.stagnation_threshold = stagnation_threshold

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with STDP tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # STDP weight matrices
        act_weights = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_weights = jnp.ones(NUM_AGGREGATIONS) * 0.5
        cross_weights = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Activity history for temporal credit
        act_history = []  # List of (generation, active_mask)
        agg_history = []
        fitness_history = []

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # STDP weights
            'act_weights': act_weights,
            'agg_weights': agg_weights,
            'cross_weights': cross_weights,
            # History for temporal credit
            'act_history': [],
            'agg_history': [],
            'fitness_history': [],
            # Tracking
            'ltp_events': 0,
            'ltd_events': 0,
            'cross_ltp_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 830000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_history(
        self,
        history: List,
        generation: int,
        mask: jnp.ndarray,
        max_length: int = 10,
    ) -> List:
        """Update activity history for temporal credit."""
        new_history = history + [(generation, mask.copy())]
        if len(new_history) > max_length:
            new_history = new_history[-max_length:]
        return new_history

    def _apply_ltp(
        self,
        weights: jnp.ndarray,
        history: List,
        current_gen: int,
        window: int,
        rate: float,
        is_agg: bool = False,
    ) -> Tuple[jnp.ndarray, int]:
        """Apply LTP to functions active within window before improvement."""
        new_weights = weights.copy()
        n_ltp = 0

        for gen, mask in history:
            if 0 < current_gen - gen <= window:
                for i in range(len(mask)):
                    if mask[i] > 0.5:
                        effective_rate = rate
                        # Boost for extreme aggregations
                        if is_agg and i in CORE_EXTREME_AGGS:
                            effective_rate *= self.extreme_ltp_multiplier

                        current = new_weights[i]
                        new_weights = new_weights.at[i].set(
                            current + effective_rate * (1 - current)
                        )
                        n_ltp += 1

        return jnp.clip(new_weights, 0.0, 1.0), n_ltp

    def _apply_ltd(
        self,
        weights: jnp.ndarray,
        history: List,
        current_gen: int,
        window: int,
        rate: float,
    ) -> Tuple[jnp.ndarray, int]:
        """Apply LTD to functions active without improvement."""
        new_weights = weights.copy()
        n_ltd = 0

        for gen, mask in history:
            if 0 < current_gen - gen <= window:
                for i in range(len(mask)):
                    if mask[i] > 0.5:
                        current = new_weights[i]
                        new_weights = new_weights.at[i].set(
                            current - rate * current
                        )
                        n_ltd += 1

        return jnp.clip(new_weights, 0.0, 1.0), n_ltd

    def _apply_cross_ltp(
        self,
        cross_weights: jnp.ndarray,
        act_history: List,
        agg_history: List,
        current_gen: int,
    ) -> Tuple[jnp.ndarray, int]:
        """Apply cross-domain LTP for co-active pairs."""
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
                                effective_rate = self.cross_ltp_rate
                                # Extra boost for sin + extreme
                                if i == 4 and j in CORE_EXTREME_AGGS:
                                    effective_rate *= self.extreme_ltp_multiplier

                                current = new_weights[i, j]
                                new_weights = new_weights.at[i, j].set(
                                    current + effective_rate * (1 - current)
                                )
                                n_ltp += 1

        return jnp.clip(new_weights, 0.0, 1.0), n_ltp

    def _mutate_palette_stdp(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        weights: jnp.ndarray,
        cross_weights: jnp.ndarray,
        is_act: bool,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply STDP-guided mutation."""
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

            # Cross-domain influence
            if is_act:
                cross_influence = float(jnp.max(cross_weights[i, :]))
            else:
                cross_influence = float(jnp.max(cross_weights[:, i]))

            combined_weight = weight * 0.7 + cross_influence * 0.3

            if mask[i] < 0.5:  # Inactive
                # Higher weight = higher activation rate
                activate_rate = self.base_activate_rate * (
                    1 + self.weight_influence * (combined_weight - 0.5)
                )
                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                # Higher weight = lower deactivation rate
                deactivate_rate = self.base_deactivate_rate * (
                    1 - self.weight_influence * (combined_weight - 0.5)
                )
                # Extra protection for sin and extreme aggs
                if is_act and i == 4:
                    deactivate_rate *= 0.5
                elif not is_act and i in CORE_EXTREME_AGGS:
                    deactivate_rate *= 0.5

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
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
        """Update with STDP temporal credit mechanism."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update history
        new_act_history = self._update_history(
            state['act_history'], generation, state['act_mask']
        )
        new_agg_history = self._update_history(
            state['agg_history'], generation, state['agg_mask']
        )
        new_fitness_history = state['fitness_history'] + [best_fitness]
        if len(new_fitness_history) > 10:
            new_fitness_history = new_fitness_history[-10:]

        # Apply STDP
        new_act_weights = state['act_weights']
        new_agg_weights = state['agg_weights']
        new_cross_weights = state['cross_weights']
        new_ltp_events = state['ltp_events']
        new_ltd_events = state['ltd_events']
        new_cross_ltp_events = state['cross_ltp_events']

        if improved and fitness_delta > 0.01:
            # LTP - strengthen functions active before improvement
            new_act_weights, n_act_ltp = self._apply_ltp(
                state['act_weights'], new_act_history, generation,
                self.ltp_window, self.ltp_rate, is_agg=False
            )
            new_agg_weights, n_agg_ltp = self._apply_ltp(
                state['agg_weights'], new_agg_history, generation,
                self.agg_ltp_window, self.agg_ltp_rate, is_agg=True
            )
            new_cross_weights, n_cross_ltp = self._apply_cross_ltp(
                state['cross_weights'], new_act_history, new_agg_history, generation
            )
            new_ltp_events += n_act_ltp + n_agg_ltp
            new_cross_ltp_events += n_cross_ltp
        elif not improved:
            # LTD - weaken functions active without improvement
            new_act_weights, n_act_ltd = self._apply_ltd(
                state['act_weights'], new_act_history, generation,
                self.ltd_window, self.ltd_rate
            )
            new_agg_weights, n_agg_ltd = self._apply_ltd(
                state['agg_weights'], new_agg_history, generation,
                self.agg_ltd_window, self.agg_ltd_rate
            )
            new_ltd_events += n_act_ltd + n_agg_ltd

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_palette_stdp(
                k_act, state['act_mask'], new_act_weights, new_cross_weights, True
            )
            new_agg_mask, agg_mutation_info = self._mutate_palette_stdp(
                k_agg, state['agg_mask'], new_agg_weights, new_cross_weights, False
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
            'act_history': new_act_history,
            'agg_history': new_agg_history,
            'fitness_history': new_fitness_history,
            'ltp_events': new_ltp_events,
            'ltd_events': new_ltd_events,
            'cross_ltp_events': new_cross_ltp_events,
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
            'mutation_triggered': act_mutation_info is not None,
            # STDP weight metrics
            'act_mean_weight': float(jnp.mean(new_act_weights)),
            'agg_mean_weight': float(jnp.mean(new_agg_weights)),
            'cross_mean_weight': float(jnp.mean(new_cross_weights)),
            'sin_weight': float(new_act_weights[4]),
            'max_weight': float(new_agg_weights[2]),
            'min_weight': float(new_agg_weights[3]),
            # Cross-domain weights
            'sin_max_weight': float(new_cross_weights[4, 2]),
            'sin_min_weight': float(new_cross_weights[4, 3]),
            # STDP events
            'ltp_events': new_ltp_events,
            'ltd_events': new_ltd_events,
            'cross_ltp_events': new_cross_ltp_events,
            # Sin status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with STDP status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'sin_weight': float(state['act_weights'][4]),
            'max_weight': float(state['agg_weights'][2]),
            'min_weight': float(state['agg_weights'][3]),
            'sin_max_weight': float(state['cross_weights'][4, 2]),
            'sin_min_weight': float(state['cross_weights'][4, 3]),
            'ltp_events': state['ltp_events'],
            'ltd_events': state['ltd_events'],
            'cross_ltp_events': state['cross_ltp_events'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
