"""Strategy 65D: Separate Genome Dual (NEAT-style for Both Domains).

Extends SeparateGenomeStrategy to jointly evolve BOTH activation AND
aggregation function palettes using NEAT-style population genetics.

Key dual mechanisms:
1. Dual genomes - combined act+agg masks with separate innovation tracking
2. Cross-domain crossover - paired inheritance of successful combinations
3. Dual speciation - distance computed across both domains
4. Affinity-weighted selection - prefer complementary act-agg pairs

Expected: 70%+ discovery rate, <35 generations, coordinated evolution
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
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


@dataclass
class DualPaletteGenome:
    """Combined activation+aggregation genome with NEAT-style tracking."""

    act_mask: jnp.ndarray  # Binary mask for activations
    agg_mask: jnp.ndarray  # Binary mask for aggregations
    act_innovations: jnp.ndarray  # When each activation was discovered
    agg_innovations: jnp.ndarray  # When each aggregation was discovered
    fitness: float = 0.0
    age: int = 0
    species_id: int = 0
    # Cross-domain affinity tracking
    affinity: jnp.ndarray = None  # NUM_ACTIVATIONS x NUM_AGGREGATIONS

    def __post_init__(self):
        if self.affinity is None:
            self.affinity = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

    def __hash__(self):
        return hash((tuple(self.act_mask.tolist()), tuple(self.agg_mask.tolist())))


def create_dual_genome(
    act_mask: jnp.ndarray,
    agg_mask: jnp.ndarray,
    global_innovation: int,
) -> Tuple[DualPaletteGenome, int]:
    """Create a new dual palette genome.

    Args:
        act_mask: Binary activation mask
        agg_mask: Binary aggregation mask
        global_innovation: Current global innovation counter

    Returns:
        Tuple of (genome, updated_innovation_counter)
    """
    act_innovations = jnp.full(NUM_ACTIVATIONS, -1, dtype=jnp.int32)
    agg_innovations = jnp.full(NUM_AGGREGATIONS, -1, dtype=jnp.int32)

    # Assign innovation numbers to active functions
    for i in range(NUM_ACTIVATIONS):
        if act_mask[i] > 0.5:
            act_innovations = act_innovations.at[i].set(global_innovation)
            global_innovation += 1

    for i in range(NUM_AGGREGATIONS):
        if agg_mask[i] > 0.5:
            agg_innovations = agg_innovations.at[i].set(global_innovation)
            global_innovation += 1

    return DualPaletteGenome(
        act_mask=act_mask,
        agg_mask=agg_mask,
        act_innovations=act_innovations,
        agg_innovations=agg_innovations,
    ), global_innovation


def dual_genome_distance(g1: DualPaletteGenome, g2: DualPaletteGenome) -> float:
    """Compute genetic distance between two dual genomes.

    Uses combined distance across both domains.
    """
    # Activation distance
    act_diff = jnp.sum((g1.act_mask > 0.5) != (g2.act_mask > 0.5))
    act_total = max(jnp.sum(g1.act_mask > 0.5), jnp.sum(g2.act_mask > 0.5), 1)
    act_dist = float(act_diff / act_total)

    # Aggregation distance
    agg_diff = jnp.sum((g1.agg_mask > 0.5) != (g2.agg_mask > 0.5))
    agg_total = max(jnp.sum(g1.agg_mask > 0.5), jnp.sum(g2.agg_mask > 0.5), 1)
    agg_dist = float(agg_diff / agg_total)

    # Combined with activation weighted more (larger search space)
    return 0.7 * act_dist + 0.3 * agg_dist


def crossover_dual_genomes(
    parent1: DualPaletteGenome,
    parent2: DualPaletteGenome,
    key: jax.random.PRNGKey,
    global_innovation: int,
) -> Tuple[DualPaletteGenome, int]:
    """NEAT-style crossover for dual genomes.

    Favors fitter parent, inherits paired act-agg combinations.
    """
    # Determine fitter parent
    fitter = parent1 if parent1.fitness >= parent2.fitness else parent2
    weaker = parent2 if parent1.fitness >= parent2.fitness else parent1

    # Activation crossover
    child_act_mask = jnp.zeros(NUM_ACTIVATIONS)
    child_act_innovations = jnp.full(NUM_ACTIVATIONS, -1, dtype=jnp.int32)

    for i in range(NUM_ACTIVATIONS):
        p1_has = parent1.act_mask[i] > 0.5
        p2_has = parent2.act_mask[i] > 0.5

        if p1_has and p2_has:
            child_act_mask = child_act_mask.at[i].set(1.0)
            child_act_innovations = child_act_innovations.at[i].set(
                min(parent1.act_innovations[i], parent2.act_innovations[i])
            )
        elif p1_has != p2_has:
            if fitter.act_mask[i] > 0.5:
                child_act_mask = child_act_mask.at[i].set(1.0)
                child_act_innovations = child_act_innovations.at[i].set(
                    fitter.act_innovations[i]
                )

    # Aggregation crossover
    child_agg_mask = jnp.zeros(NUM_AGGREGATIONS)
    child_agg_innovations = jnp.full(NUM_AGGREGATIONS, -1, dtype=jnp.int32)

    for i in range(NUM_AGGREGATIONS):
        p1_has = parent1.agg_mask[i] > 0.5
        p2_has = parent2.agg_mask[i] > 0.5

        if p1_has and p2_has:
            child_agg_mask = child_agg_mask.at[i].set(1.0)
            child_agg_innovations = child_agg_innovations.at[i].set(
                min(parent1.agg_innovations[i], parent2.agg_innovations[i])
            )
        elif p1_has != p2_has:
            if fitter.agg_mask[i] > 0.5:
                child_agg_mask = child_agg_mask.at[i].set(1.0)
                child_agg_innovations = child_agg_innovations.at[i].set(
                    fitter.agg_innovations[i]
                )

    # Inherit affinity from fitter parent (cross-domain learning)
    child_affinity = fitter.affinity.copy()

    return DualPaletteGenome(
        act_mask=child_act_mask,
        agg_mask=child_agg_mask,
        act_innovations=child_act_innovations,
        agg_innovations=child_agg_innovations,
        affinity=child_affinity,
    ), global_innovation


def mutate_dual_genome(
    genome: DualPaletteGenome,
    key: jax.random.PRNGKey,
    global_innovation: int,
    act_mutation_rate: float = 0.15,
    agg_mutation_rate: float = 0.12,
    min_active_act: int = 2,
    min_active_agg: int = 1,
) -> Tuple[DualPaletteGenome, int]:
    """Mutate dual genome with affinity-guided bias.

    High-affinity pairs more likely to be kept together.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    new_act_mask = genome.act_mask.copy()
    new_agg_mask = genome.agg_mask.copy()
    new_act_innovations = genome.act_innovations.copy()
    new_agg_innovations = genome.agg_innovations.copy()

    # Activation mutations
    act_flip_probs = jax.random.uniform(k1, (NUM_ACTIVATIONS,))
    for i in range(NUM_ACTIVATIONS):
        if act_flip_probs[i] < act_mutation_rate:
            if new_act_mask[i] > 0.5:
                # Deactivate - but check affinity protection
                max_affinity = float(jnp.max(genome.affinity[i, :]))
                protection = 0.5 * max_affinity  # High affinity = harder to remove
                if jax.random.uniform(k3) > protection:
                    new_act_mask = new_act_mask.at[i].set(0.0)
            else:
                # Activate with new innovation
                new_act_mask = new_act_mask.at[i].set(1.0)
                if new_act_innovations[i] < 0:
                    new_act_innovations = new_act_innovations.at[i].set(global_innovation)
                    global_innovation += 1

    # Aggregation mutations
    agg_flip_probs = jax.random.uniform(k2, (NUM_AGGREGATIONS,))
    for i in range(NUM_AGGREGATIONS):
        if agg_flip_probs[i] < agg_mutation_rate:
            if new_agg_mask[i] > 0.5:
                # Deactivate with affinity protection
                max_affinity = float(jnp.max(genome.affinity[:, i]))
                protection = 0.5 * max_affinity
                if jax.random.uniform(k4) > protection:
                    new_agg_mask = new_agg_mask.at[i].set(0.0)
            else:
                new_agg_mask = new_agg_mask.at[i].set(1.0)
                if new_agg_innovations[i] < 0:
                    new_agg_innovations = new_agg_innovations.at[i].set(global_innovation)
                    global_innovation += 1

    # Ensure minimums
    if jnp.sum(new_act_mask > 0.5) < min_active_act:
        new_act_mask = genome.act_mask
        new_act_innovations = genome.act_innovations
    if jnp.sum(new_agg_mask > 0.5) < min_active_agg:
        new_agg_mask = genome.agg_mask
        new_agg_innovations = genome.agg_innovations

    return DualPaletteGenome(
        act_mask=new_act_mask,
        agg_mask=new_agg_mask,
        act_innovations=new_act_innovations,
        agg_innovations=new_agg_innovations,
        fitness=0.0,
        age=genome.age,
        species_id=genome.species_id,
        affinity=genome.affinity,
    ), global_innovation


class SeparateGenomeDualStrategy(PaletteEvolutionStrategy):
    """NEAT-style dual palette genome evolution.

    Maintains a population of combined act+agg genomes that evolve
    together with cross-domain affinity tracking.
    """

    name = "separate_genome_dual"
    description = "NEAT-style dual genome with speciation and cross-domain affinity"

    def __init__(
        self,
        palette_pop_size: int = 10,
        species_threshold: float = 0.3,
        crossover_rate: float = 0.75,
        act_mutation_rate: float = 0.15,
        agg_mutation_rate: float = 0.12,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        affinity_learning_rate: float = 0.1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            palette_pop_size: Population size for palette genomes
            species_threshold: Distance threshold for speciation
            crossover_rate: Probability of crossover vs mutation
            act_mutation_rate: Per-activation mutation probability
            agg_mutation_rate: Per-aggregation mutation probability
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            affinity_learning_rate: Rate of affinity learning from success
            initial_act_palette: Starting activation palette
            initial_agg_palette: Starting aggregation palette
        """
        self.palette_pop_size = palette_pop_size
        self.species_threshold = species_threshold
        self.crossover_rate = crossover_rate
        self.act_mutation_rate = act_mutation_rate
        self.agg_mutation_rate = agg_mutation_rate
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.affinity_learning_rate = affinity_learning_rate
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize dual palette population."""
        key = jax.random.PRNGKey(seed + 656565)

        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        global_innovation = 0
        population = []

        for i in range(self.palette_pop_size):
            key, subkey = jax.random.split(key)

            if i == 0:
                genome, global_innovation = create_dual_genome(
                    act_mask, agg_mask, global_innovation
                )
            else:
                genome, global_innovation = create_dual_genome(
                    act_mask, agg_mask, global_innovation
                )
                genome, global_innovation = mutate_dual_genome(
                    genome, subkey, global_innovation,
                    act_mutation_rate=0.2,
                    agg_mutation_rate=0.15,
                    min_active_act=self.min_active_act,
                    min_active_agg=self.min_active_agg,
                )

            population.append(genome)

        return {
            'population': population,
            'best_idx': 0,
            'global_innovation': global_innovation,
            'rng_key': key,
            'generation': 0,
            'species_count': 1,
            'strategy_name': self.name,
            'best_fitness_seen': 0.0,
            'stagnation_count': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current best activation palette."""
        best = state['population'][state['best_idx']]
        return mask_to_indices(best.act_mask)

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current best aggregation palette."""
        best = state['population'][state['best_idx']]
        return mask_to_indices(best.agg_mask)

    def _assign_species(self, population: List[DualPaletteGenome]) -> int:
        """Assign species IDs based on dual genetic distance."""
        if not population:
            return 0

        species_reps = [population[0]]
        population[0].species_id = 0

        for genome in population[1:]:
            assigned = False
            for species_id, rep in enumerate(species_reps):
                if dual_genome_distance(genome, rep) < self.species_threshold:
                    genome.species_id = species_id
                    assigned = True
                    break

            if not assigned:
                genome.species_id = len(species_reps)
                species_reps.append(genome)

        return len(species_reps)

    def _update_affinity(
        self,
        genome: DualPaletteGenome,
        fitness: float,
    ) -> DualPaletteGenome:
        """Update cross-domain affinity based on fitness."""
        if fitness <= 0:
            return genome

        new_affinity = genome.affinity.copy()
        act_active = genome.act_mask > 0.5
        agg_active = genome.agg_mask > 0.5

        # Strengthen affinity between co-active pairs when successful
        for i in range(NUM_ACTIVATIONS):
            if act_active[i]:
                for j in range(NUM_AGGREGATIONS):
                    if agg_active[j]:
                        current = float(new_affinity[i, j])
                        boost = self.affinity_learning_rate * fitness
                        new_affinity = new_affinity.at[i, j].set(
                            min(1.0, current + boost)
                        )

        genome.affinity = new_affinity
        return genome

    def _evolve_population(
        self,
        population: List[DualPaletteGenome],
        key: jax.random.PRNGKey,
        global_innovation: int,
    ) -> Tuple[List[DualPaletteGenome], int]:
        """Evolve dual palette population."""
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        new_population = []

        # Elitism
        new_population.append(sorted_pop[0])

        while len(new_population) < self.palette_pop_size:
            key, k1, k2, k3 = jax.random.split(key, 4)

            # Tournament selection
            tournament_size = 3
            indices = jax.random.choice(
                k1, len(sorted_pop),
                shape=(tournament_size,), replace=False
            )
            parent1 = sorted_pop[int(indices[0])]

            if jax.random.uniform(k2) < self.crossover_rate:
                indices2 = jax.random.choice(
                    k3, len(sorted_pop),
                    shape=(tournament_size,), replace=False
                )
                parent2 = sorted_pop[int(indices2[0])]

                child, global_innovation = crossover_dual_genomes(
                    parent1, parent2, k3, global_innovation
                )
            else:
                child, global_innovation = mutate_dual_genome(
                    parent1, k3, global_innovation,
                    self.act_mutation_rate, self.agg_mutation_rate,
                    self.min_active_act, self.min_active_agg
                )

            new_population.append(child)

        return new_population, global_innovation

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update dual palette population."""
        key, subkey = jax.random.split(state['rng_key'])

        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        population = state['population']
        best_idx = state['best_idx']

        # Update fitness and affinity of current best
        population[best_idx].fitness = best_fitness
        population[best_idx].age += 1
        population[best_idx] = self._update_affinity(population[best_idx], best_fitness)

        # Speciate
        species_count = self._assign_species(population)

        # Evolve
        new_population, global_innovation = self._evolve_population(
            population, subkey, state['global_innovation']
        )

        # Find new best
        new_best_idx = max(
            range(len(new_population)),
            key=lambda i: new_population[i].fitness
        )

        old_act = mask_to_indices(population[best_idx].act_mask)
        old_agg = mask_to_indices(population[best_idx].agg_mask)
        new_act = mask_to_indices(new_population[new_best_idx].act_mask)
        new_agg = mask_to_indices(new_population[new_best_idx].agg_mask)

        act_changed = old_act != new_act
        agg_changed = old_agg != new_agg

        new_state = {
            'population': new_population,
            'best_idx': new_best_idx,
            'global_innovation': global_innovation,
            'rng_key': key,
            'generation': generation + 1,
            'species_count': species_count,
            'strategy_name': self.name,
            'best_fitness_seen': new_best,
            'stagnation_count': new_stagnation,
        }

        # Compute mean affinity for active pairs
        best_genome = new_population[new_best_idx]
        act_active = best_genome.act_mask > 0.5
        agg_active = best_genome.agg_mask > 0.5
        active_affinities = []
        for i in range(NUM_ACTIVATIONS):
            if act_active[i]:
                for j in range(NUM_AGGREGATIONS):
                    if agg_active[j]:
                        active_affinities.append(float(best_genome.affinity[i, j]))
        mean_affinity = np.mean(active_affinities) if active_affinities else 0.0

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': new_act,
            'current_agg_palette': new_agg,
            'species_count': species_count,
            'global_innovation': global_innovation,
            'population_diversity': len(set(
                (tuple(mask_to_indices(g.act_mask)), tuple(mask_to_indices(g.agg_mask)))
                for g in new_population
            )),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'mean_active_affinity': mean_affinity,
            'has_sin': 4 in new_act,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual population stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        population = state['population']

        sin_count = sum(1 for g in population if g.act_mask[4] > 0.5)
        best = population[state['best_idx']]

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'species_count': state['species_count'],
            'pop_with_sin': sin_count,
            'global_innovation': state['global_innovation'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'best_fitness': best.fitness,
        }
