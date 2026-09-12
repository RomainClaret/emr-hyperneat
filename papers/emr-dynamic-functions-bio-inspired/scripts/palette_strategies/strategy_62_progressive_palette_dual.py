"""Strategy 62D: Progressive Palette Dual (Inherit-and-Expand for Both Domains).

Extends ProgressivePaletteStrategy to jointly evolve BOTH activation AND
aggregation function palettes with progressive freezing.

Key dual mechanisms:
1. Dual frozen/additions - separate frozen and addition sets per domain
2. Coordinated freezing - success can freeze both domains together
3. Cross-domain contribution - pairs contribute together
4. Synchronized task switching

Expected: Zero forgetting in both domains through progressive freezing
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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


class ProgressivePaletteDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with progressive freezing.

    Both activation and aggregation palettes have frozen and addition
    components. Frozen functions are never removed, ensuring zero
    forgetting across both domains.
    """

    name = "progressive_palette_dual"
    description = "Dual: Progressive inherit-and-expand with coordinated freezing"

    def __init__(
        self,
        # Freezing thresholds
        inherit_threshold: float = 0.85,
        partial_freeze_threshold: float = 0.7,
        freeze_delay: int = 5,
        # Addition constraints
        max_act_additions: int = 3,
        max_agg_additions: int = 2,
        max_total_act: int = 10,
        max_total_agg: int = 5,
        addition_mutation_rate: float = 0.15,
        # Contribution tracking
        contribution_window: int = 10,
        min_contribution_for_keep: float = 0.0,
        # Exploration
        exploration_rate: float = 0.2,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Progressive Palette Dual strategy."""
        # Freezing
        self.inherit_threshold = inherit_threshold
        self.partial_freeze_threshold = partial_freeze_threshold
        self.freeze_delay = freeze_delay

        # Additions
        self.max_act_additions = max_act_additions
        self.max_agg_additions = max_agg_additions
        self.max_total_act = max_total_act
        self.max_total_agg = max_total_agg
        self.addition_mutation_rate = addition_mutation_rate

        # Contribution
        self.contribution_window = contribution_window
        self.min_contribution_for_keep = min_contribution_for_keep

        # Exploration
        self.exploration_rate = exploration_rate

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual progressive tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        act_contributions = jnp.zeros(NUM_ACTIVATIONS)
        agg_contributions = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_frozen': [],
            'act_additions': list(initial_act),
            'act_contributions': act_contributions,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_frozen': [],
            'agg_additions': list(initial_agg),
            'agg_contributions': agg_contributions,
            # Freezing state
            'freeze_counter': 0,
            'task_number': 0,
            'freeze_events': [],
            # General state
            'rng_key': jax.random.PRNGKey(seed + 626262),
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

    def _update_mask_from_palette(
        self,
        frozen: List[int],
        additions: List[int],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Create mask from frozen and additions."""
        mask = jnp.zeros(n_funcs)
        for i in frozen:
            if 0 <= i < n_funcs:
                mask = mask.at[i].set(1.0)
        for i in additions:
            if 0 <= i < n_funcs:
                mask = mask.at[i].set(1.0)
        return mask

    def _update_contributions(
        self,
        contributions: jnp.ndarray,
        additions: List[int],
        fitness: float,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update contribution estimates."""
        new_contributions = contributions * 0.9
        for i in additions:
            if 0 <= i < n_funcs:
                current = float(new_contributions[i])
                new_contributions = new_contributions.at[i].set(current * 0.8 + fitness * 0.2)
        return new_contributions

    def _check_for_freeze(
        self,
        best_fitness: float,
        freeze_counter: int,
        act_frozen: List[int],
        act_additions: List[int],
        agg_frozen: List[int],
        agg_additions: List[int],
    ) -> Tuple[List[int], List[int], List[int], List[int], int, bool]:
        """Check if we should freeze both palettes."""
        freeze_event = False

        if best_fitness >= self.inherit_threshold:
            new_counter = freeze_counter + 1
            if new_counter >= self.freeze_delay:
                new_act_frozen = list(act_frozen)
                for i in act_additions:
                    if i not in new_act_frozen:
                        new_act_frozen.append(i)

                new_agg_frozen = list(agg_frozen)
                for i in agg_additions:
                    if i not in new_agg_frozen:
                        new_agg_frozen.append(i)

                freeze_event = True
                return new_act_frozen, [], new_agg_frozen, [], 0, freeze_event
            return act_frozen, act_additions, agg_frozen, agg_additions, new_counter, freeze_event

        return act_frozen, act_additions, agg_frozen, agg_additions, 0, freeze_event

    def _mutate_additions(
        self,
        frozen: List[int],
        additions: List[int],
        contributions: jnp.ndarray,
        key: jax.random.PRNGKey,
        max_additions: int,
        max_total: int,
        n_funcs: int,
    ) -> List[int]:
        """Mutate additions."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_additions = list(additions)
        total_in_palette = set(frozen) | set(additions)

        if (jax.random.uniform(key1) < self.exploration_rate and
            len(total_in_palette) < max_total):
            available = [i for i in range(n_funcs) if i not in total_in_palette]
            if available:
                new_func = available[int(jax.random.randint(key2, (), 0, len(available)))]
                new_additions.append(new_func)

        if len(new_additions) > max_additions:
            contrib_scores = [(i, float(contributions[i])) for i in new_additions]
            contrib_scores.sort(key=lambda x: x[1])
            for func, contrib in contrib_scores:
                if contrib < self.min_contribution_for_keep and len(new_additions) > 0:
                    new_additions.remove(func)
                    break

        if jax.random.uniform(key3) < self.addition_mutation_rate and new_additions:
            remove_idx = int(jax.random.randint(key3, (), 0, len(new_additions)))
            new_additions.pop(remove_idx)
            available = [i for i in range(n_funcs) if i not in frozen and i not in new_additions]
            if available:
                key3, subkey = jax.random.split(key3)
                new_func = available[int(jax.random.randint(subkey, (), 0, len(available)))]
                new_additions.append(new_func)

        return new_additions

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual progressive dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update contributions
        new_act_contrib = self._update_contributions(
            state['act_contributions'], state['act_additions'],
            best_fitness, NUM_ACTIVATIONS
        )
        new_agg_contrib = self._update_contributions(
            state['agg_contributions'], state['agg_additions'],
            best_fitness, NUM_AGGREGATIONS
        )

        # Check for coordinated freeze
        (new_act_frozen, new_act_additions,
         new_agg_frozen, new_agg_additions,
         new_freeze_counter, freeze_event) = self._check_for_freeze(
            best_fitness, state['freeze_counter'],
            state['act_frozen'], state['act_additions'],
            state['agg_frozen'], state['agg_additions']
        )

        # Mutate additions if not frozen
        if not freeze_event:
            new_act_additions = self._mutate_additions(
                new_act_frozen, new_act_additions, new_act_contrib, k_act,
                self.max_act_additions, self.max_total_act, NUM_ACTIVATIONS
            )
            new_agg_additions = self._mutate_additions(
                new_agg_frozen, new_agg_additions, new_agg_contrib, k_agg,
                self.max_agg_additions, self.max_total_agg, NUM_AGGREGATIONS
            )

        # Create masks
        new_act_mask = self._update_mask_from_palette(new_act_frozen, new_act_additions, NUM_ACTIVATIONS)
        new_agg_mask = self._update_mask_from_palette(new_agg_frozen, new_agg_additions, NUM_AGGREGATIONS)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        freeze_events = list(state['freeze_events'])
        if freeze_event:
            freeze_events.append((generation, list(new_act_frozen), list(new_agg_frozen)))

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_frozen': new_act_frozen,
            'act_additions': new_act_additions,
            'act_contributions': new_act_contrib,
            'agg_mask': new_agg_mask,
            'agg_frozen': new_agg_frozen,
            'agg_additions': new_agg_additions,
            'agg_contributions': new_agg_contrib,
            'freeze_counter': new_freeze_counter,
            'task_number': state['task_number'],
            'freeze_events': freeze_events,
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
            # Progressive state
            'act_n_frozen': len(new_act_frozen),
            'act_n_additions': len(new_act_additions),
            'agg_n_frozen': len(new_agg_frozen),
            'agg_n_additions': len(new_agg_additions),
            # Freeze status
            'freeze_event': freeze_event,
            'freeze_counter': new_freeze_counter,
            'total_freeze_events': len(freeze_events),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_is_frozen': 4 in new_act_frozen,
            'sin_is_addition': 4 in new_act_additions,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual progressive status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_frozen': state['act_frozen'],
            'act_additions': state['act_additions'],
            'agg_frozen': state['agg_frozen'],
            'agg_additions': state['agg_additions'],
            'freeze_counter': state['freeze_counter'],
            'total_freeze_events': len(state['freeze_events']),
            'sin_is_frozen': 4 in state['act_frozen'],
        }
