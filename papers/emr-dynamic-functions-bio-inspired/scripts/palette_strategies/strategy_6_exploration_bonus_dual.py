"""Strategy 6D: Exploration Bonus Dual (Novelty/Curiosity for Both Domains).

Extends ExplorationBonusStrategy to jointly evolve BOTH activation AND aggregation
function palettes using novelty search and curiosity-driven exploration.

Cross-Domain Learning:
- Separate novelty/curiosity tracking for both domains
- Cross-domain discovery bonus: finding good combinations gets extra reward
- Shared population with dual genome representation
- Elite replacement considers both domain novelties

Key Dual Mechanisms:
1. Dual novelty - genetic distance in both domains
2. Dual curiosity - under-explored functions in both domains
3. Cross-discovery bonus - novel act-agg combinations
4. Shared effective fitness combining all exploration signals
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
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

# Aggregation domain constants
NUM_AGGREGATIONS = 6
DEFAULT_AGGREGATION_INDICES = [0, 1]  # sum, mean


@dataclass
class DualExploratoryGenome:
    """Palette genome with dual domain exploration tracking."""
    act_mask: jnp.ndarray
    agg_mask: jnp.ndarray
    fitness: float = 0.0
    act_novelty: float = 0.0
    agg_novelty: float = 0.0
    act_curiosity: float = 0.0
    agg_curiosity: float = 0.0
    cross_discovery_bonus: float = 0.0
    effective_fitness: float = 0.0
    age: int = 0


class ExplorationBonusDualStrategy(PaletteEvolutionStrategy):
    """Novelty/curiosity-driven dual palette evolution.

    Extends exploration bonus mechanism to both activation and aggregation
    domains with cross-domain discovery rewards.
    """

    name = "exploration_bonus_dual"
    description = "NEAT palette genome + novelty/curiosity for both domains"

    def __init__(
        self,
        # Population
        palette_pop_size: int = 10,
        # Novelty weights
        act_novelty_weight: float = 0.3,
        agg_novelty_weight: float = 0.25,
        # Curiosity weights
        act_curiosity_weight: float = 0.2,
        agg_curiosity_weight: float = 0.15,
        # Discovery bonuses
        act_discovery_bonus: float = 0.5,
        agg_discovery_bonus: float = 0.3,
        cross_discovery_bonus: float = 0.4,
        # Mutation rates
        act_mutation_rate: float = 0.15,
        agg_mutation_rate: float = 0.12,
        # Elite replacement
        elite_replacement_rate: float = 0.3,
        # Constraints
        min_active_act: int = 2,
        max_active_act: int = 6,
        min_active_agg: int = 1,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Exploration Bonus Dual strategy."""
        self.palette_pop_size = palette_pop_size
        self.act_novelty_weight = act_novelty_weight
        self.agg_novelty_weight = agg_novelty_weight
        self.act_curiosity_weight = act_curiosity_weight
        self.agg_curiosity_weight = agg_curiosity_weight
        self.act_discovery_bonus = act_discovery_bonus
        self.agg_discovery_bonus = agg_discovery_bonus
        self.cross_discovery_bonus = cross_discovery_bonus
        self.act_mutation_rate = act_mutation_rate
        self.agg_mutation_rate = agg_mutation_rate
        self.elite_replacement_rate = elite_replacement_rate
        self.min_active_act = min_active_act
        self.max_active_act = max_active_act
        self.min_active_agg = min_active_agg
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

    def _compute_novelty(
        self,
        mask: jnp.ndarray,
        population_masks: List[jnp.ndarray],
        k: int = 3,
    ) -> float:
        """Compute novelty as average distance to k nearest neighbors."""
        if len(population_masks) <= 1:
            return 1.0

        distances = []
        for other_mask in population_masks:
            diff = float(jnp.sum((mask > 0.5) != (other_mask > 0.5)))
            distances.append(diff)

        distances.sort()
        k_nearest = distances[:min(k, len(distances))]
        return sum(k_nearest) / len(k_nearest) if k_nearest else 1.0

    def _compute_curiosity(
        self,
        mask: jnp.ndarray,
        exploration_counts: jnp.ndarray,
    ) -> float:
        """Compute curiosity bonus for under-explored functions."""
        active = mask > 0.5
        total = jnp.sum(exploration_counts)
        if total == 0:
            return 1.0
        inverse_freq = 1.0 / (exploration_counts + 1)
        return float(jnp.sum(inverse_freq * active) / max(jnp.sum(active), 1))

    def _mutate_mask(
        self,
        mask: jnp.ndarray,
        key: jax.random.PRNGKey,
        mutation_rate: float,
        min_active: int,
        max_active: int,
        discovered_ever: jnp.ndarray,
        discovery_bonus_value: float,
    ) -> Tuple[jnp.ndarray, float]:
        """Mutate a mask with discovery bonus tracking."""
        n_funcs = len(mask)
        flip_probs = jax.random.uniform(key, (n_funcs,))
        new_mask = mask.copy()
        discovery_bonus = 0.0

        for i in range(n_funcs):
            if flip_probs[i] < mutation_rate:
                if new_mask[i] > 0.5:
                    new_mask = new_mask.at[i].set(0.0)
                else:
                    new_mask = new_mask.at[i].set(1.0)
                    if discovered_ever[i] < 0.5:
                        discovery_bonus += discovery_bonus_value

        # Enforce constraints
        n_active = int(jnp.sum(new_mask > 0.5))
        if n_active < min_active or n_active > max_active:
            return mask, 0.0

        return new_mask, discovery_bonus

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize dual palette population."""
        key = jax.random.PRNGKey(seed + 66666)

        # Initial masks
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_mask = agg_mask.at[i].set(1.0)

        # Create population
        population = []
        for i in range(self.palette_pop_size):
            key, k1, k2 = jax.random.split(key, 3)
            if i == 0:
                genome = DualExploratoryGenome(act_mask=act_mask.copy(), agg_mask=agg_mask.copy())
            else:
                # Mutated versions
                new_act, _ = self._mutate_mask(
                    act_mask, k1, self.act_mutation_rate * 2,
                    self.min_active_act, self.max_active_act,
                    jnp.zeros(NUM_ACTIVATIONS), 0.0
                )
                new_agg, _ = self._mutate_mask(
                    agg_mask, k2, self.agg_mutation_rate * 2,
                    self.min_active_agg, self.max_active_agg,
                    jnp.zeros(NUM_AGGREGATIONS), 0.0
                )
                genome = DualExploratoryGenome(act_mask=new_act, agg_mask=new_agg)
            population.append(genome)

        # Track discoveries
        act_discovered = jnp.zeros(NUM_ACTIVATIONS)
        agg_discovered = jnp.zeros(NUM_AGGREGATIONS)
        cross_discovered = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        for g in population:
            act_discovered = jnp.maximum(act_discovered, (g.act_mask > 0.5).astype(jnp.float32))
            agg_discovered = jnp.maximum(agg_discovered, (g.agg_mask > 0.5).astype(jnp.float32))

        # Exploration counts
        act_counts = jnp.zeros(NUM_ACTIVATIONS)
        agg_counts = jnp.zeros(NUM_AGGREGATIONS)
        for g in population:
            act_counts = act_counts + (g.act_mask > 0.5).astype(jnp.float32)
            agg_counts = agg_counts + (g.agg_mask > 0.5).astype(jnp.float32)

        return {
            'population': population,
            'best_idx': 0,
            'rng_key': key,
            'generation': 0,
            'act_discovered': act_discovered,
            'agg_discovered': agg_discovered,
            'cross_discovered': cross_discovered,
            'act_counts': act_counts,
            'agg_counts': agg_counts,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current best activation palette."""
        return mask_to_indices(state['population'][state['best_idx']].act_mask)

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current best aggregation palette."""
        mask = state['population'][state['best_idx']].agg_mask
        return [i for i in range(NUM_AGGREGATIONS) if mask[i] > 0.5]

    def _compute_effective_fitness(
        self,
        population: List[DualExploratoryGenome],
        act_counts: jnp.ndarray,
        agg_counts: jnp.ndarray,
    ) -> List[DualExploratoryGenome]:
        """Compute effective fitness with all exploration bonuses."""
        act_masks = [g.act_mask for g in population]
        agg_masks = [g.agg_mask for g in population]

        for genome in population:
            # Novelty in both domains
            genome.act_novelty = self._compute_novelty(genome.act_mask, act_masks)
            genome.agg_novelty = self._compute_novelty(genome.agg_mask, agg_masks)

            # Curiosity in both domains
            genome.act_curiosity = self._compute_curiosity(genome.act_mask, act_counts)
            genome.agg_curiosity = self._compute_curiosity(genome.agg_mask, agg_counts)

            # Combined effective fitness
            genome.effective_fitness = (
                genome.fitness
                + self.act_novelty_weight * genome.act_novelty
                + self.agg_novelty_weight * genome.agg_novelty
                + self.act_curiosity_weight * genome.act_curiosity
                + self.agg_curiosity_weight * genome.agg_curiosity
                + genome.cross_discovery_bonus
            )

        return population

    def _evolve_population(
        self,
        population: List[DualExploratoryGenome],
        key: jax.random.PRNGKey,
        act_discovered: jnp.ndarray,
        agg_discovered: jnp.ndarray,
        cross_discovered: jnp.ndarray,
    ) -> Tuple[List[DualExploratoryGenome], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evolve dual population with exploration bonuses."""
        sorted_pop = sorted(population, key=lambda g: g.effective_fitness, reverse=True)

        new_population = []

        # Elitism with explorer replacement
        n_elite = max(1, int(self.palette_pop_size * 0.2))
        n_explorer = int(n_elite * self.elite_replacement_rate)

        # Best by task fitness
        for i in range(n_elite - n_explorer):
            new_population.append(sorted_pop[i])

        # Best by combined novelty
        novelty_sorted = sorted(
            population,
            key=lambda g: g.act_novelty + g.agg_novelty,
            reverse=True
        )
        for i in range(n_explorer):
            if len(new_population) < n_elite:
                new_population.append(novelty_sorted[i])

        # Fill rest through mutation
        while len(new_population) < self.palette_pop_size:
            key, k1, k2, k3 = jax.random.split(key, 4)

            # Tournament selection
            indices = jax.random.choice(k1, len(sorted_pop), shape=(3,), replace=False)
            parent = max([sorted_pop[int(i)] for i in indices], key=lambda g: g.effective_fitness)

            # Mutate both domains
            new_act, act_bonus = self._mutate_mask(
                parent.act_mask, k2, self.act_mutation_rate,
                self.min_active_act, self.max_active_act,
                act_discovered, self.act_discovery_bonus
            )
            new_agg, agg_bonus = self._mutate_mask(
                parent.agg_mask, k3, self.agg_mutation_rate,
                self.min_active_agg, self.max_active_agg,
                agg_discovered, self.agg_discovery_bonus
            )

            # Check for cross-domain discovery
            cross_bonus = 0.0
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if new_act[i] > 0.5 and new_agg[j] > 0.5:
                        if cross_discovered[i, j] < 0.5:
                            cross_bonus += self.cross_discovery_bonus * 0.1
                            cross_discovered = cross_discovered.at[i, j].set(1.0)

            child = DualExploratoryGenome(
                act_mask=new_act,
                agg_mask=new_agg,
                cross_discovery_bonus=act_bonus + agg_bonus + cross_bonus,
            )
            new_population.append(child)

            # Update discoveries
            act_discovered = jnp.maximum(act_discovered, (new_act > 0.5).astype(jnp.float32))
            agg_discovered = jnp.maximum(agg_discovered, (new_agg > 0.5).astype(jnp.float32))

        return new_population, act_discovered, agg_discovered, cross_discovered

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual exploration bonuses."""
        key, subkey = jax.random.split(state['rng_key'])

        population = state['population']
        population[state['best_idx']].fitness = best_fitness
        population[state['best_idx']].age += 1

        # Update exploration counts
        act_counts = state['act_counts']
        agg_counts = state['agg_counts']
        for g in population:
            act_counts = act_counts + (g.act_mask > 0.5).astype(jnp.float32)
            agg_counts = agg_counts + (g.agg_mask > 0.5).astype(jnp.float32)

        # Compute effective fitness
        population = self._compute_effective_fitness(population, act_counts, agg_counts)

        # Evolve
        new_pop, act_disc, agg_disc, cross_disc = self._evolve_population(
            population, subkey,
            state['act_discovered'], state['agg_discovered'], state['cross_discovered']
        )

        # Find new best
        new_best_idx = max(range(len(new_pop)), key=lambda i: new_pop[i].fitness)

        old_act = mask_to_indices(population[state['best_idx']].act_mask)
        new_act = mask_to_indices(new_pop[new_best_idx].act_mask)

        new_state = {
            'population': new_pop,
            'best_idx': new_best_idx,
            'rng_key': key,
            'generation': generation + 1,
            'act_discovered': act_disc,
            'agg_discovered': agg_disc,
            'cross_discovered': cross_disc,
            'act_counts': act_counts,
            'agg_counts': agg_counts,
            'strategy_name': self.name,
        }

        best = new_pop[new_best_idx]
        agg_palette = [i for i in range(NUM_AGGREGATIONS) if best.agg_mask[i] > 0.5]

        metrics = {
            'palette_changed': old_act != new_act,
            'current_palette': new_act,
            'current_agg_palette': agg_palette,
            'avg_act_novelty': sum(g.act_novelty for g in new_pop) / len(new_pop),
            'avg_agg_novelty': sum(g.agg_novelty for g in new_pop) / len(new_pop),
            'total_act_discovered': int(jnp.sum(act_disc > 0.5)),
            'total_agg_discovered': int(jnp.sum(agg_disc > 0.5)),
            'cross_combinations': int(jnp.sum(cross_disc > 0.5)),
            'has_sin': 4 in new_act,
            'has_agg4': len(agg_palette) >= 4,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual exploration stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'total_act_discovered': int(jnp.sum(state['act_discovered'] > 0.5)),
            'total_agg_discovered': int(jnp.sum(state['agg_discovered'] > 0.5)),
            'cross_combinations': int(jnp.sum(state['cross_discovered'] > 0.5)),
        }
