"""Strategy 3D: Asymmetric Sticky Dual (Asymmetric Rates for Both Domains).

Extends AsymmetricStickyStrategy to jointly evolve BOTH activation AND aggregation
function palettes using asymmetric mutation rates and sticky discovery protection.

Cross-Domain Learning:
- Separate sticky functions for both domains
- Cross-domain sticky: if act-agg combination works, protect both
- Shared discovery history with domain-specific tracking
- Asymmetric rates tuned for each domain's characteristics

Key Dual Mechanisms:
1. Dual asymmetric rates - higher activate, lower deactivate for both domains
2. Domain-specific sticky functions - protect oscillatory acts, optimal aggs
3. Cross-domain sticky protection - successful combinations protected
4. Dual discovery history tracking
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    DEFAULT_PALETTE_INDICES,
)

# Aggregation domain constants
NUM_AGGREGATIONS = 6
DEFAULT_AGGREGATION_INDICES = [0, 1]  # sum, mean


class AsymmetricStickyDualStrategy(PaletteEvolutionStrategy):
    """Asymmetric mutation rates with sticky discoveries for both domains.

    Uses higher activation rate than deactivation rate in both activation
    and aggregation domains, with sticky protection for valuable discoveries.
    Cross-domain learning protects successful act-agg combinations.
    """

    name = "asymmetric_sticky_dual"
    description = "Asymmetric rates + sticky discoveries for both activation and aggregation"

    def __init__(
        self,
        # Activation rates
        act_activate_rate: float = 0.25,
        act_deactivate_rate: float = 0.05,
        act_sticky_functions: List[int] = None,  # [4, 11, 12] oscillatory
        # Aggregation rates
        agg_activate_rate: float = 0.20,
        agg_deactivate_rate: float = 0.06,
        agg_sticky_functions: List[int] = None,  # Optimal for parity
        # Cross-domain sticky
        cross_sticky_threshold: float = 0.9,  # Fitness to trigger cross-sticky
        cross_sticky_gens: int = 3,  # Gens at high fitness to become cross-sticky
        # Constraints
        min_active_act: int = 2,
        max_active_act: int = 6,
        min_active_agg: int = 1,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Asymmetric Sticky Dual strategy."""
        # Activation
        self.act_activate_rate = act_activate_rate
        self.act_deactivate_rate = act_deactivate_rate
        self.act_sticky_functions = act_sticky_functions or [4, 11, 12]

        # Aggregation
        self.agg_activate_rate = agg_activate_rate
        self.agg_deactivate_rate = agg_deactivate_rate
        self.agg_sticky_functions = agg_sticky_functions or [0, 1, 2, 3]  # sum,mean,max,min

        # Cross-domain
        self.cross_sticky_threshold = cross_sticky_threshold
        self.cross_sticky_gens = cross_sticky_gens

        # Constraints
        self.min_active_act = min_active_act
        self.max_active_act = max_active_act
        self.min_active_agg = min_active_agg
        self.max_active_agg = max_active_agg

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual discovery tracking."""
        # Activation domain
        initial_act = config.get('initial_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_discovered = act_mask > 0.5

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_mask = agg_mask.at[i].set(1.0)
        agg_discovered = agg_mask > 0.5

        # Cross-domain sticky matrix
        cross_sticky = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS), dtype=jnp.bool_)
        cross_sticky_count = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS), dtype=jnp.int32)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_discovered': act_discovered,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_discovered': agg_discovered,
            # Cross-domain
            'cross_sticky': cross_sticky,
            'cross_sticky_count': cross_sticky_count,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 33333),
            'generation': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return [i for i in range(NUM_AGGREGATIONS) if state['agg_mask'][i] > 0.5]

    def _mutate_act_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        discovered: jnp.ndarray,
        cross_sticky: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Apply asymmetric mutation to activation palette."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        new_mask = mask.copy()
        activated = []
        deactivated = []
        sticky_protected = []

        # Activate inactive functions
        for i in range(NUM_ACTIVATIONS):
            if mask[i] < 0.5 and activate_probs[i] < self.act_activate_rate:
                new_mask = new_mask.at[i].set(1.0)
                activated.append(i)

        # Update discovered
        new_discovered = discovered | (new_mask > 0.5)

        # Deactivate active functions (with protection)
        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5 and deactivate_probs[i] < self.act_deactivate_rate:
                # Check sticky protection
                if i in self.act_sticky_functions and new_discovered[i]:
                    sticky_protected.append(i)
                    continue

                # Check cross-sticky protection
                cross_protected = False
                for j in range(NUM_AGGREGATIONS):
                    if agg_mask[j] > 0.5 and cross_sticky[i, j]:
                        cross_protected = True
                        break
                if cross_protected:
                    sticky_protected.append(i)
                    continue

                new_mask = new_mask.at[i].set(0.0)
                deactivated.append(i)

        # Enforce constraints
        n_active = int(jnp.sum(new_mask > 0.5))
        if n_active < self.min_active_act or n_active > self.max_active_act:
            return mask, discovered, {'act_activated': [], 'act_deactivated': [], 'act_sticky_protected': []}

        return new_mask, new_discovered, {
            'act_activated': activated,
            'act_deactivated': deactivated,
            'act_sticky_protected': sticky_protected,
        }

    def _mutate_agg_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        discovered: jnp.ndarray,
        cross_sticky: jnp.ndarray,
        act_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Apply asymmetric mutation to aggregation palette."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (NUM_AGGREGATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_AGGREGATIONS,))

        new_mask = mask.copy()
        activated = []
        deactivated = []
        sticky_protected = []

        # Activate
        for i in range(NUM_AGGREGATIONS):
            if mask[i] < 0.5 and activate_probs[i] < self.agg_activate_rate:
                new_mask = new_mask.at[i].set(1.0)
                activated.append(i)

        new_discovered = discovered | (new_mask > 0.5)

        # Deactivate with protection
        for i in range(NUM_AGGREGATIONS):
            if mask[i] > 0.5 and deactivate_probs[i] < self.agg_deactivate_rate:
                if i in self.agg_sticky_functions and new_discovered[i]:
                    sticky_protected.append(i)
                    continue

                # Cross-sticky
                cross_protected = False
                for j in range(NUM_ACTIVATIONS):
                    if act_mask[j] > 0.5 and cross_sticky[j, i]:
                        cross_protected = True
                        break
                if cross_protected:
                    sticky_protected.append(i)
                    continue

                new_mask = new_mask.at[i].set(0.0)
                deactivated.append(i)

        # Enforce constraints
        n_active = int(jnp.sum(new_mask > 0.5))
        if n_active < self.min_active_agg or n_active > self.max_active_agg:
            return mask, discovered, {'agg_activated': [], 'agg_deactivated': [], 'agg_sticky_protected': []}

        return new_mask, new_discovered, {
            'agg_activated': activated,
            'agg_deactivated': deactivated,
            'agg_sticky_protected': sticky_protected,
        }

    def _update_cross_sticky(
        self,
        cross_sticky: jnp.ndarray,
        cross_sticky_count: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update cross-domain sticky matrix based on high fitness."""
        new_count = cross_sticky_count.copy()
        new_sticky = cross_sticky.copy()

        if fitness >= self.cross_sticky_threshold:
            # Increment count for active combinations
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        new_count = new_count.at[i, j].set(int(cross_sticky_count[i, j]) + 1)
                        if new_count[i, j] >= self.cross_sticky_gens:
                            new_sticky = new_sticky.at[i, j].set(True)
        else:
            # Reset counts for inactive combinations
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] < 0.5 or agg_mask[j] < 0.5:
                        new_count = new_count.at[i, j].set(0)

        return new_sticky, new_count

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update both palettes with asymmetric mutations."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Update cross-sticky
        new_cross_sticky, new_cross_count = self._update_cross_sticky(
            state['cross_sticky'],
            state['cross_sticky_count'],
            state['act_mask'],
            state['agg_mask'],
            best_fitness,
        )

        # Mutate activation palette
        new_act_mask, new_act_disc, act_info = self._mutate_act_palette(
            k1, state['act_mask'], state['act_discovered'],
            new_cross_sticky, state['agg_mask']
        )

        # Mutate aggregation palette
        new_agg_mask, new_agg_disc, agg_info = self._mutate_agg_palette(
            k2, state['agg_mask'], state['agg_discovered'],
            new_cross_sticky, state['act_mask']
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        new_state = {
            'act_mask': new_act_mask,
            'act_discovered': new_act_disc,
            'agg_mask': new_agg_mask,
            'agg_discovered': new_agg_disc,
            'cross_sticky': new_cross_sticky,
            'cross_sticky_count': new_cross_count,
            'rng_key': key,
            'generation': generation + 1,
            'best_fitness_seen': max(state['best_fitness_seen'], best_fitness),
            'strategy_name': self.name,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = self.get_active_agg_palette(new_state)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'act_discovered': mask_to_indices(new_act_disc),
            'agg_discovered': [i for i in range(NUM_AGGREGATIONS) if new_agg_disc[i]],
            'n_cross_sticky': int(jnp.sum(new_cross_sticky)),
            'has_sin': 4 in act_palette,
            'sin_discovered': bool(new_act_disc[4]),
            'has_agg4': len(agg_palette) >= 4,
        }
        metrics.update(act_info)
        metrics.update(agg_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual discovery status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'sin_discovered': bool(state['act_discovered'][4]),
            'act_discovered': mask_to_indices(state['act_discovered']),
            'agg_discovered': [i for i in range(NUM_AGGREGATIONS) if state['agg_discovered'][i]],
            'n_cross_sticky': int(jnp.sum(state['cross_sticky'])),
        }
