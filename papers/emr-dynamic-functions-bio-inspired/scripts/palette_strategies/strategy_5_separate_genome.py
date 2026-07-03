"""Strategy 5: Separate Palette Genome (NEAT-style).

Full redesign treating palette as a first-class evolvable component:
- Palette as independent genome with innovation tracking
- Species-based protection for new discoveries
- Crossover favoring fitter parent
- Separate population of palettes

Expected: 70%+ discovery rate, <35 generations
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
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


@dataclass
class PaletteGenome:
    """Individual palette genome with NEAT-style tracking."""

    mask: jnp.ndarray  # Binary mask for activations
    innovation_numbers: jnp.ndarray  # When each activation was discovered
    fitness: float = 0.0
    age: int = 0
    species_id: int = 0

    def __hash__(self):
        return hash(tuple(self.mask.tolist()))


def create_palette_genome(
    mask: jnp.ndarray,
    global_innovation: int,
) -> Tuple['PaletteGenome', int]:
    """Create a new palette genome.

    Args:
        mask: Binary activation mask
        global_innovation: Current global innovation counter

    Returns:
        Tuple of (genome, updated_innovation_counter)
    """
    innovation_numbers = jnp.full(NUM_ACTIVATIONS, -1, dtype=jnp.int32)

    # Assign innovation numbers to active functions
    for i in range(NUM_ACTIVATIONS):
        if mask[i] > 0.5:
            innovation_numbers = innovation_numbers.at[i].set(global_innovation)
            global_innovation += 1

    return PaletteGenome(
        mask=mask,
        innovation_numbers=innovation_numbers,
    ), global_innovation


def palette_distance(g1: PaletteGenome, g2: PaletteGenome) -> float:
    """Compute genetic distance between two palette genomes.

    Uses number of differing activations normalized by total active.
    """
    differing = jnp.sum((g1.mask > 0.5) != (g2.mask > 0.5))
    total_active = max(jnp.sum(g1.mask > 0.5), jnp.sum(g2.mask > 0.5), 1)
    return float(differing / total_active)


def crossover_palettes(
    parent1: PaletteGenome,
    parent2: PaletteGenome,
    key: jax.random.PRNGKey,
    global_innovation: int,
) -> Tuple[PaletteGenome, int]:
    """NEAT-style crossover favoring fitter parent.

    Args:
        parent1: First parent
        parent2: Second parent
        key: Random key
        global_innovation: Current innovation counter

    Returns:
        Tuple of (child_genome, updated_innovation)
    """
    child_mask = jnp.zeros(NUM_ACTIVATIONS)
    child_innovations = jnp.full(NUM_ACTIVATIONS, -1, dtype=jnp.int32)

    # Determine fitter parent
    fitter = parent1 if parent1.fitness >= parent2.fitness else parent2
    weaker = parent2 if parent1.fitness >= parent2.fitness else parent1

    for i in range(NUM_ACTIVATIONS):
        p1_has = parent1.mask[i] > 0.5
        p2_has = parent2.mask[i] > 0.5

        if p1_has and p2_has:
            # Both have it - include with high probability
            child_mask = child_mask.at[i].set(1.0)
            child_innovations = child_innovations.at[i].set(
                min(parent1.innovation_numbers[i], parent2.innovation_numbers[i])
            )
        elif p1_has != p2_has:
            # Only one has it - inherit from fitter parent
            if fitter.mask[i] > 0.5:
                child_mask = child_mask.at[i].set(1.0)
                child_innovations = child_innovations.at[i].set(
                    fitter.innovation_numbers[i]
                )
        # else: neither has it, stays 0

    return PaletteGenome(
        mask=child_mask,
        innovation_numbers=child_innovations,
    ), global_innovation


def mutate_palette_genome(
    genome: PaletteGenome,
    key: jax.random.PRNGKey,
    global_innovation: int,
    mutation_rate: float = 0.15,
    min_active: int = 2,
) -> Tuple[PaletteGenome, int]:
    """Mutate palette genome.

    Args:
        genome: Genome to mutate
        key: Random key
        global_innovation: Current innovation counter
        mutation_rate: Mutation probability per activation
        min_active: Minimum active functions

    Returns:
        Tuple of (mutated_genome, updated_innovation)
    """
    key1, key2 = jax.random.split(key)

    new_mask = genome.mask.copy()
    new_innovations = genome.innovation_numbers.copy()

    # Flip each activation with mutation_rate probability
    flip_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))

    for i in range(NUM_ACTIVATIONS):
        if flip_probs[i] < mutation_rate:
            if new_mask[i] > 0.5:
                # Deactivate
                new_mask = new_mask.at[i].set(0.0)
            else:
                # Activate with new innovation number
                new_mask = new_mask.at[i].set(1.0)
                if new_innovations[i] < 0:
                    new_innovations = new_innovations.at[i].set(global_innovation)
                    global_innovation += 1

    # Ensure minimum active
    if jnp.sum(new_mask > 0.5) < min_active:
        return genome, global_innovation  # Return unchanged

    return PaletteGenome(
        mask=new_mask,
        innovation_numbers=new_innovations,
        fitness=0.0,  # Reset fitness for new variant
        age=genome.age,
        species_id=genome.species_id,
    ), global_innovation


class SeparateGenomeStrategy(PaletteEvolutionStrategy):
    """NEAT-style separate palette genome evolution.

    Maintains a small population of palette genomes that evolve
    independently, with speciation protecting innovations.
    """

    name = "separate_genome"
    description = "NEAT-style palette genome with speciation and crossover"

    def __init__(
        self,
        palette_pop_size: int = 10,
        species_threshold: float = 0.3,
        crossover_rate: float = 0.75,
        mutation_rate: float = 0.15,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            palette_pop_size: Population size for palette genomes
            species_threshold: Distance threshold for speciation
            crossover_rate: Probability of crossover vs mutation
            mutation_rate: Per-activation mutation probability
            min_active: Minimum active functions
            initial_palette: Starting palette indices
        """
        self.palette_pop_size = palette_pop_size
        self.species_threshold = species_threshold
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize palette population.

        Creates diverse initial population through mutations.
        """
        key = jax.random.PRNGKey(seed + 55555)

        initial = config.get('initial_palette', self.initial_palette)
        initial_mask = create_initial_palette_mask(initial)

        global_innovation = 0
        population = []

        # Create initial population with variation
        for i in range(self.palette_pop_size):
            key, subkey = jax.random.split(key)

            if i == 0:
                # First is the original
                genome, global_innovation = create_palette_genome(
                    initial_mask, global_innovation
                )
            else:
                # Others are mutations of original
                genome, global_innovation = create_palette_genome(
                    initial_mask, global_innovation
                )
                genome, global_innovation = mutate_palette_genome(
                    genome, subkey, global_innovation,
                    mutation_rate=0.2,  # Slightly higher for diversity
                    min_active=self.min_active,
                )

            population.append(genome)

        # Find best palette (initially all equal)
        best_idx = 0

        return {
            'population': population,
            'best_idx': best_idx,
            'global_innovation': global_innovation,
            'rng_key': key,
            'generation': 0,
            'species_count': 1,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current best palette indices."""
        best_genome = state['population'][state['best_idx']]
        return mask_to_indices(best_genome.mask)

    def _assign_species(self, population: List[PaletteGenome]) -> int:
        """Assign species IDs based on genetic distance."""
        if not population:
            return 0

        # Simple speciation: first genome of each species is representative
        species_reps = [population[0]]
        population[0].species_id = 0

        for genome in population[1:]:
            assigned = False
            for species_id, rep in enumerate(species_reps):
                if palette_distance(genome, rep) < self.species_threshold:
                    genome.species_id = species_id
                    assigned = True
                    break

            if not assigned:
                # New species
                genome.species_id = len(species_reps)
                species_reps.append(genome)

        return len(species_reps)

    def _evolve_population(
        self,
        population: List[PaletteGenome],
        key: jax.random.PRNGKey,
        global_innovation: int,
    ) -> Tuple[List[PaletteGenome], int]:
        """Evolve palette population for one generation.

        Args:
            population: Current population
            key: Random key
            global_innovation: Innovation counter

        Returns:
            Tuple of (new_population, updated_innovation)
        """
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)

        new_population = []

        # Elitism: keep best
        new_population.append(sorted_pop[0])

        # Fill rest through reproduction
        while len(new_population) < self.palette_pop_size:
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

            # Tournament selection
            tournament_size = 3
            indices = jax.random.choice(
                subkey1, len(sorted_pop),
                shape=(tournament_size,), replace=False
            )
            parent1 = sorted_pop[int(indices[0])]

            if jax.random.uniform(subkey2) < self.crossover_rate:
                # Crossover
                indices2 = jax.random.choice(
                    subkey3, len(sorted_pop),
                    shape=(tournament_size,), replace=False
                )
                parent2 = sorted_pop[int(indices2[0])]

                child, global_innovation = crossover_palettes(
                    parent1, parent2, subkey3, global_innovation
                )
            else:
                # Mutation only
                child, global_innovation = mutate_palette_genome(
                    parent1, subkey3, global_innovation,
                    self.mutation_rate, self.min_active
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
        """Update palette population.

        1. Assign fitness to palette genomes
        2. Speciate
        3. Evolve population

        Args:
            state: Current state
            generation: Current generation
            best_fitness: Best fitness this gen
            prev_best_fitness: Previous best fitness
            population_data: Dict with fitness info

        Returns:
            Tuple of (new_state, metrics)
        """
        key, subkey = jax.random.split(state['rng_key'])

        # Assign fitness to current best palette
        population = state['population']
        best_idx = state['best_idx']

        # Update fitness of current palette
        population[best_idx].fitness = best_fitness
        population[best_idx].age += 1

        # Speciate
        species_count = self._assign_species(population)

        # Evolve population
        new_population, global_innovation = self._evolve_population(
            population, subkey, state['global_innovation']
        )

        # Find new best (highest fitness)
        new_best_idx = max(
            range(len(new_population)),
            key=lambda i: new_population[i].fitness
        )

        # Track if palette changed
        old_palette = mask_to_indices(population[best_idx].mask)
        new_palette = mask_to_indices(new_population[new_best_idx].mask)
        palette_changed = old_palette != new_palette

        new_state = {
            'population': new_population,
            'best_idx': new_best_idx,
            'global_innovation': global_innovation,
            'rng_key': key,
            'generation': generation + 1,
            'species_count': species_count,
            'strategy_name': self.name,
        }

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': new_palette,
            'species_count': species_count,
            'global_innovation': global_innovation,
            'population_diversity': len(set(
                tuple(mask_to_indices(g.mask)) for g in new_population
            )),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including population stats."""
        palette = self.get_active_palette(state)
        population = state['population']

        # Count how many genomes have sin
        sin_count = sum(1 for g in population if g.mask[4] > 0.5)

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'species_count': state['species_count'],
            'pop_with_sin': sin_count,
            'global_innovation': state['global_innovation'],
        }
