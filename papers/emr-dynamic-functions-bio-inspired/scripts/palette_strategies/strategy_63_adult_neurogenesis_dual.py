"""Strategy 63D: Adult Neurogenesis Dual (Birth-Maturation-Survival for Both Domains).

Extends AdultNeurogenesisStrategy to jointly evolve BOTH activation AND
aggregation function palettes using hippocampal-inspired neurogenesis.

Key dual mechanisms:
1. Dual stable/young tracking - separate pools per domain
2. Coordinated birth - new neurons can be born in both domains
3. Cross-domain maturation boost - success in partner domain helps survival
4. Independent contribution tracking per domain

Expected: Controlled exploration with survival-based integration in both domains
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


class AdultNeurogenesisDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with hippocampal-inspired neurogenesis.

    Both activation and aggregation functions can be born, mature,
    and survive or be pruned based on their contribution to fitness.
    """

    name = "adult_neurogenesis_dual"
    description = "Dual: Birth, maturation, and survival-based integration"

    def __init__(
        self,
        # Neurogenesis
        neurogenesis_rate: float = 0.08,
        maturation_period: int = 10,
        young_plasticity: float = 2.0,
        survival_threshold: float = 0.1,
        max_young_act: int = 3,
        max_young_agg: int = 2,
        # Contribution
        contribution_decay: float = 0.9,
        contribution_boost: float = 0.3,
        # Cross-domain
        cross_survival_boost: float = 0.15,
        # Stable palette
        stable_mutation_rate: float = 0.02,
        max_stable_act: int = 8,
        max_stable_agg: int = 4,
        min_stable_act: int = 2,
        min_stable_agg: int = 1,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Adult Neurogenesis Dual strategy."""
        # Neurogenesis
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.young_plasticity = young_plasticity
        self.survival_threshold = survival_threshold
        self.max_young_act = max_young_act
        self.max_young_agg = max_young_agg

        # Contribution
        self.contribution_decay = contribution_decay
        self.contribution_boost = contribution_boost

        # Cross-domain
        self.cross_survival_boost = cross_survival_boost

        # Stable
        self.stable_mutation_rate = stable_mutation_rate
        self.max_stable_act = max_stable_act
        self.max_stable_agg = max_stable_agg
        self.min_stable_act = min_stable_act
        self.min_stable_agg = min_stable_agg

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual neurogenesis tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        act_contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_contribution = act_contribution.at[i].set(0.5)

        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.5)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_stable': set(initial_act),
            'act_young': {},
            'act_contribution': act_contribution,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_stable': set(initial_agg),
            'agg_young': {},
            'agg_contribution': agg_contribution,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 636363),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Stats
            'act_total_births': 0,
            'act_total_survivals': 0,
            'act_total_prunings': 0,
            'agg_total_births': 0,
            'agg_total_survivals': 0,
            'agg_total_prunings': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _maybe_birth_neuron(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        key: jax.random.PRNGKey,
        generation: int,
        max_young: int,
        n_funcs: int,
    ) -> Tuple[Set[int], Dict[int, Dict], Optional[int]]:
        """Possibly birth a new neuron."""
        key1, key2 = jax.random.split(key)
        new_young = dict(young)
        born = None

        if (len(new_young) < max_young and
            jax.random.uniform(key1) < self.neurogenesis_rate):
            available = [i for i in range(n_funcs) if i not in stable and i not in new_young]
            if available:
                idx = int(jax.random.randint(key2, (), 0, len(available)))
                new_func = available[idx]
                new_young[new_func] = {'birth_gen': generation, 'contribution': 0.0}
                born = new_func

        return stable, new_young, born

    def _mature_neurons(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        contribution: jnp.ndarray,
        partner_mean_contrib: float,
        generation: int,
        max_stable: int,
    ) -> Tuple[Set[int], Dict[int, Dict], List[int], List[int]]:
        """Process neuron maturation."""
        new_stable = set(stable)
        new_young = {}
        survived = []
        pruned = []

        for func, info in young.items():
            age = generation - info['birth_gen']
            if age >= self.maturation_period:
                func_contrib = float(contribution[func])
                # Cross-domain boost
                effective_threshold = self.survival_threshold - partner_mean_contrib * self.cross_survival_boost

                if func_contrib >= effective_threshold:
                    if len(new_stable) < max_stable:
                        new_stable.add(func)
                        survived.append(func)
                    else:
                        pruned.append(func)
                else:
                    pruned.append(func)
            else:
                new_young[func] = info

        return new_stable, new_young, survived, pruned

    def _update_contributions(
        self,
        contribution: jnp.ndarray,
        mask: jnp.ndarray,
        young: Dict[int, Dict],
        improved: bool,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, Dict[int, Dict]]:
        """Update contribution tracking."""
        new_contribution = contribution * self.contribution_decay
        new_young = {}

        for i in range(n_funcs):
            if mask[i] > 0.5:
                current = float(new_contribution[i])
                if improved:
                    boost = self.contribution_boost
                    if i in young:
                        boost *= self.young_plasticity
                    new_contribution = new_contribution.at[i].set(current + boost)
                else:
                    new_contribution = new_contribution.at[i].set(current + 0.01)

        for func, info in young.items():
            new_info = dict(info)
            new_info['contribution'] = float(new_contribution[func])
            new_young[func] = new_info

        return jnp.clip(new_contribution, 0, 2.0), new_young

    def _create_mask(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Create mask from stable and young."""
        mask = jnp.zeros(n_funcs)
        for i in stable:
            if 0 <= i < n_funcs:
                mask = mask.at[i].set(1.0)
        for i in young.keys():
            if 0 <= i < n_funcs:
                mask = mask.at[i].set(1.0)
        return mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual neurogenesis dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update contributions
        new_act_contrib, act_young = self._update_contributions(
            state['act_contribution'], state['act_mask'],
            state['act_young'], improved, NUM_ACTIVATIONS
        )
        new_agg_contrib, agg_young = self._update_contributions(
            state['agg_contribution'], state['agg_mask'],
            state['agg_young'], improved, NUM_AGGREGATIONS
        )

        # Compute mean contributions for cross-domain boost
        act_active = mask_to_indices(state['act_mask'])
        agg_active = mask_to_indices(state['agg_mask'])
        act_mean_contrib = float(np.mean([new_act_contrib[i] for i in act_active])) if act_active else 0
        agg_mean_contrib = float(np.mean([new_agg_contrib[i] for i in agg_active])) if agg_active else 0

        # Mature neurons with cross-domain boost
        act_stable, act_young, act_survived, act_pruned = self._mature_neurons(
            set(state['act_stable']), act_young, new_act_contrib,
            agg_mean_contrib, generation, self.max_stable_act
        )
        agg_stable, agg_young, agg_survived, agg_pruned = self._mature_neurons(
            set(state['agg_stable']), agg_young, new_agg_contrib,
            act_mean_contrib, generation, self.max_stable_agg
        )

        # Birth new neurons
        act_stable, act_young, act_born = self._maybe_birth_neuron(
            act_stable, act_young, k_act, generation,
            self.max_young_act, NUM_ACTIVATIONS
        )
        agg_stable, agg_young, agg_born = self._maybe_birth_neuron(
            agg_stable, agg_young, k_agg, generation,
            self.max_young_agg, NUM_AGGREGATIONS
        )

        # Create masks
        new_act_mask = self._create_mask(act_stable, act_young, NUM_ACTIVATIONS)
        new_agg_mask = self._create_mask(agg_stable, agg_young, NUM_AGGREGATIONS)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_stable': act_stable,
            'act_young': act_young,
            'act_contribution': new_act_contrib,
            'agg_mask': new_agg_mask,
            'agg_stable': agg_stable,
            'agg_young': agg_young,
            'agg_contribution': new_agg_contrib,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'act_total_births': state['act_total_births'] + (1 if act_born else 0),
            'act_total_survivals': state['act_total_survivals'] + len(act_survived),
            'act_total_prunings': state['act_total_prunings'] + len(act_pruned),
            'agg_total_births': state['agg_total_births'] + (1 if agg_born else 0),
            'agg_total_survivals': state['agg_total_survivals'] + len(agg_survived),
            'agg_total_prunings': state['agg_total_prunings'] + len(agg_pruned),
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
            # Neurogenesis
            'act_n_stable': len(act_stable),
            'act_n_young': len(act_young),
            'agg_n_stable': len(agg_stable),
            'agg_n_young': len(agg_young),
            'act_born': act_born,
            'agg_born': agg_born,
            'act_survived': act_survived,
            'agg_survived': agg_survived,
            'act_pruned': act_pruned,
            'agg_pruned': agg_pruned,
            # Stats
            'act_survival_rate': (new_state['act_total_survivals'] / max(new_state['act_total_births'], 1)) * 100,
            'agg_survival_rate': (new_state['agg_total_survivals'] / max(new_state['agg_total_births'], 1)) * 100,
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_is_stable': 4 in act_stable,
            'sin_is_young': 4 in act_young,
            'sin_contribution': float(new_act_contrib[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual neurogenesis status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_n_stable': len(state['act_stable']),
            'act_n_young': len(state['act_young']),
            'agg_n_stable': len(state['agg_stable']),
            'agg_n_young': len(state['agg_young']),
            'act_total_births': state['act_total_births'],
            'agg_total_births': state['agg_total_births'],
            'sin_is_stable': 4 in state['act_stable'],
            'sin_contribution': float(state['act_contribution'][4]),
        }
