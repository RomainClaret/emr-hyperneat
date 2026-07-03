"""Strategy 6: Exploration Bonus Separate Genome.

Fixes the complete failure of Strategy 5 (separate_genome) by adding
exploration bonuses inspired by novelty search and curiosity-driven learning.

Key additions over Strategy 5:
1. Novelty bonus - reward genetic distance from population
2. Curiosity bonus - reward for under-explored activations
3. Discovery bonus - one-time reward for first-time activation discovery
4. Elite replacement - replace some elites with high-novelty explorers

Expected: 80%+ discovery (vs 0% original), 40-60% solve
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
class ExploratoryGenome:
    """Palette genome with exploration tracking."""

    mask: jnp.ndarray  # Binary mask for activations
    innovation_numbers: jnp.ndarray  # When each activation was discovered
    fitness: float = 0.0  # Task fitness
    novelty: float = 0.0  # Genetic novelty score
    curiosity: float = 0.0  # Curiosity bonus
    discovery_bonus: float = 0.0  # One-time discovery bonus
    effective_fitness: float = 0.0  # Combined score for selection
    age: int = 0
    species_id: int = 0

    def __hash__(self):
        return hash(tuple(self.mask.tolist()))


def compute_novelty(genome: ExploratoryGenome, population: List[ExploratoryGenome], k: int = 3) -> float:
    """Compute novelty as average distance to k nearest neighbors.

    Args:
        genome: Genome to score
        population: Population to compare against
        k: Number of nearest neighbors

    Returns:
        Novelty score (higher = more novel)
    """
    if len(population) <= 1:
        return 1.0

    distances = []
    for other in population:
        if other is not genome:
            # Genetic distance = number of differing activations
            diff = jnp.sum((genome.mask > 0.5) != (other.mask > 0.5))
            distances.append(float(diff))

    if not distances:
        return 1.0

    # Average distance to k nearest neighbors
    distances.sort()
    k_nearest = distances[:min(k, len(distances))]
    return sum(k_nearest) / len(k_nearest)


def compute_curiosity(genome: ExploratoryGenome, exploration_counts: jnp.ndarray) -> float:
    """Compute curiosity bonus for under-explored activations.

    Args:
        genome: Genome to score
        exploration_counts: How often each activation has been tried

    Returns:
        Curiosity bonus (higher = more unexplored activations)
    """
    active = genome.mask > 0.5
    total_exploration = jnp.sum(exploration_counts)

    if total_exploration == 0:
        return 1.0

    # Inverse frequency weighting - rare activations get higher bonus
    inverse_freq = 1.0 / (exploration_counts + 1)
    curiosity = jnp.sum(inverse_freq * active) / max(jnp.sum(active), 1)

    return float(curiosity)


class ExplorationBonusStrategy(PaletteEvolutionStrategy):
    """NEAT-style palette evolution with exploration bonuses.

    Fixes Strategy 5's failure by rewarding exploration:
    - Novelty: Genetic distance from population
    - Curiosity: Under-explored activations
    - Discovery: One-time bonus for new activations

    effective_fitness = task_fitness + novelty_weight * novelty
                      + curiosity_weight * curiosity + discovery_bonus
    """

    name = "exploration_bonus"
    description = "NEAT palette genome + novelty/curiosity bonuses"

    def __init__(
        self,
        palette_pop_size: int = 10,
        novelty_weight: float = 0.3,
        curiosity_weight: float = 0.2,
        discovery_bonus_value: float = 0.5,
        elite_replacement_rate: float = 0.3,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.75,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            palette_pop_size: Population size for palette genomes
            novelty_weight: Weight for novelty bonus (default 0.3)
            curiosity_weight: Weight for curiosity bonus (default 0.2)
            discovery_bonus_value: One-time bonus for new discoveries (default 0.5)
            elite_replacement_rate: Fraction of elites to replace with explorers
            mutation_rate: Per-activation mutation probability
            crossover_rate: Probability of crossover vs mutation
            min_active: Minimum active functions
            initial_palette: Starting palette indices
        """
        self.palette_pop_size = palette_pop_size
        self.novelty_weight = novelty_weight
        self.curiosity_weight = curiosity_weight
        self.discovery_bonus_value = discovery_bonus_value
        self.elite_replacement_rate = elite_replacement_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _create_genome(
        self,
        mask: jnp.ndarray,
        global_innovation: int,
    ) -> Tuple[ExploratoryGenome, int]:
        """Create a new genome with innovation tracking."""
        innovation_numbers = jnp.full(NUM_ACTIVATIONS, -1, dtype=jnp.int32)

        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                innovation_numbers = innovation_numbers.at[i].set(global_innovation)
                global_innovation += 1

        return ExploratoryGenome(
            mask=mask,
            innovation_numbers=innovation_numbers,
        ), global_innovation

    def _mutate_genome(
        self,
        genome: ExploratoryGenome,
        key: jax.random.PRNGKey,
        global_innovation: int,
        discovered_ever: jnp.ndarray,
    ) -> Tuple[ExploratoryGenome, int, float]:
        """Mutate genome with discovery bonus tracking.

        Returns:
            Tuple of (new_genome, updated_innovation, discovery_bonus)
        """
        new_mask = genome.mask.copy()
        new_innovations = genome.innovation_numbers.copy()
        discovery_bonus = 0.0

        flip_probs = jax.random.uniform(key, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            if flip_probs[i] < self.mutation_rate:
                if new_mask[i] > 0.5:
                    # Deactivate
                    new_mask = new_mask.at[i].set(0.0)
                else:
                    # Activate - check if first-time discovery
                    new_mask = new_mask.at[i].set(1.0)
                    if new_innovations[i] < 0:
                        new_innovations = new_innovations.at[i].set(global_innovation)
                        global_innovation += 1

                    # Discovery bonus if this activation never tried before
                    if discovered_ever[i] < 0.5:
                        discovery_bonus += self.discovery_bonus_value

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            return genome, global_innovation, 0.0

        new_genome = ExploratoryGenome(
            mask=new_mask,
            innovation_numbers=new_innovations,
            age=genome.age,
            species_id=genome.species_id,
        )

        return new_genome, global_innovation, discovery_bonus

    def _crossover(
        self,
        parent1: ExploratoryGenome,
        parent2: ExploratoryGenome,
        key: jax.random.PRNGKey,
    ) -> ExploratoryGenome:
        """NEAT-style crossover favoring fitter parent."""
        child_mask = jnp.zeros(NUM_ACTIVATIONS)
        child_innovations = jnp.full(NUM_ACTIVATIONS, -1, dtype=jnp.int32)

        # Use effective fitness for parent selection
        fitter = parent1 if parent1.effective_fitness >= parent2.effective_fitness else parent2

        for i in range(NUM_ACTIVATIONS):
            p1_has = parent1.mask[i] > 0.5
            p2_has = parent2.mask[i] > 0.5

            if p1_has and p2_has:
                # Both have it - keep it
                child_mask = child_mask.at[i].set(1.0)
                child_innovations = child_innovations.at[i].set(
                    min(parent1.innovation_numbers[i], parent2.innovation_numbers[i])
                )
            elif p1_has != p2_has:
                # Only one has it - inherit from fitter parent
                if fitter.mask[i] > 0.5:
                    child_mask = child_mask.at[i].set(1.0)
                    child_innovations = child_innovations.at[i].set(fitter.innovation_numbers[i])

        return ExploratoryGenome(
            mask=child_mask,
            innovation_numbers=child_innovations,
        )

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize palette population with diversity."""
        key = jax.random.PRNGKey(seed + 66666)

        initial = config.get('initial_palette', self.initial_palette)
        initial_mask = create_initial_palette_mask(initial)

        global_innovation = 0
        population = []

        # Create diverse initial population
        for i in range(self.palette_pop_size):
            key, subkey = jax.random.split(key)

            if i == 0:
                # First is the original
                genome, global_innovation = self._create_genome(initial_mask, global_innovation)
            else:
                # Others are mutations with higher rate for diversity
                genome, global_innovation = self._create_genome(initial_mask, global_innovation)
                discovered_ever = jnp.zeros(NUM_ACTIVATIONS)
                genome, global_innovation, _ = self._mutate_genome(
                    genome, subkey, global_innovation, discovered_ever
                )

            population.append(genome)

        # Track which activations have ever been discovered
        discovered_ever = jnp.zeros(NUM_ACTIVATIONS)
        for genome in population:
            discovered_ever = discovered_ever + (genome.mask > 0.5).astype(jnp.float32)
        discovered_ever = (discovered_ever > 0).astype(jnp.float32)

        # Track exploration counts
        exploration_counts = jnp.zeros(NUM_ACTIVATIONS)
        for genome in population:
            exploration_counts = exploration_counts + (genome.mask > 0.5).astype(jnp.float32)

        return {
            'population': population,
            'best_idx': 0,
            'global_innovation': global_innovation,
            'rng_key': key,
            'generation': 0,
            'discovered_ever': discovered_ever,
            'exploration_counts': exploration_counts,
            'novelty_archive': [],  # Archive of novel genomes
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current best palette indices."""
        best_genome = state['population'][state['best_idx']]
        return mask_to_indices(best_genome.mask)

    def _compute_effective_fitness(
        self,
        population: List[ExploratoryGenome],
        exploration_counts: jnp.ndarray,
    ) -> List[ExploratoryGenome]:
        """Compute effective fitness including exploration bonuses."""
        updated = []

        for genome in population:
            # Compute novelty
            novelty = compute_novelty(genome, population)
            genome.novelty = novelty

            # Compute curiosity
            curiosity = compute_curiosity(genome, exploration_counts)
            genome.curiosity = curiosity

            # Combine scores
            genome.effective_fitness = (
                genome.fitness
                + self.novelty_weight * novelty
                + self.curiosity_weight * curiosity
                + genome.discovery_bonus
            )

            updated.append(genome)

        return updated

    def _evolve_population(
        self,
        population: List[ExploratoryGenome],
        key: jax.random.PRNGKey,
        global_innovation: int,
        discovered_ever: jnp.ndarray,
    ) -> Tuple[List[ExploratoryGenome], int, jnp.ndarray]:
        """Evolve population with exploration bonuses."""
        # Sort by effective fitness
        sorted_pop = sorted(population, key=lambda g: g.effective_fitness, reverse=True)

        new_population = []

        # Elitism - but replace some elites with high-novelty explorers
        n_elite = max(1, int(self.palette_pop_size * 0.2))
        n_explorer_replace = int(n_elite * self.elite_replacement_rate)

        # Add best by task fitness
        for i in range(n_elite - n_explorer_replace):
            new_population.append(sorted_pop[i])

        # Add best by novelty (explorer replacements)
        novelty_sorted = sorted(population, key=lambda g: g.novelty, reverse=True)
        for i in range(n_explorer_replace):
            if len(new_population) < n_elite:
                new_population.append(novelty_sorted[i])

        # Fill rest through reproduction
        while len(new_population) < self.palette_pop_size:
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

            # Tournament selection (using effective fitness)
            tournament_size = 3
            indices = jax.random.choice(
                subkey1, len(sorted_pop),
                shape=(tournament_size,), replace=False
            )
            parent1 = max([sorted_pop[int(i)] for i in indices], key=lambda g: g.effective_fitness)

            if jax.random.uniform(subkey2) < self.crossover_rate:
                # Crossover
                indices2 = jax.random.choice(
                    subkey3, len(sorted_pop),
                    shape=(tournament_size,), replace=False
                )
                parent2 = max([sorted_pop[int(i)] for i in indices2], key=lambda g: g.effective_fitness)

                child = self._crossover(parent1, parent2, subkey3)
            else:
                # Mutation only
                child, global_innovation, discovery_bonus = self._mutate_genome(
                    parent1, subkey3, global_innovation, discovered_ever
                )
                child.discovery_bonus = discovery_bonus

                # Update discovered_ever
                discovered_ever = jnp.maximum(
                    discovered_ever,
                    (child.mask > 0.5).astype(jnp.float32)
                )

            new_population.append(child)

        return new_population, global_innovation, discovered_ever

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update palette population with exploration bonuses."""
        key, subkey = jax.random.split(state['rng_key'])

        population = state['population']
        best_idx = state['best_idx']

        # Update fitness of current best palette
        population[best_idx].fitness = best_fitness
        population[best_idx].age += 1

        # Update exploration counts
        exploration_counts = state['exploration_counts']
        for genome in population:
            exploration_counts = exploration_counts + (genome.mask > 0.5).astype(jnp.float32)

        # Compute effective fitness with exploration bonuses
        population = self._compute_effective_fitness(population, exploration_counts)

        # Evolve population
        new_population, global_innovation, discovered_ever = self._evolve_population(
            population, subkey, state['global_innovation'], state['discovered_ever']
        )

        # Find new best (by task fitness for actual selection)
        new_best_idx = max(
            range(len(new_population)),
            key=lambda i: new_population[i].fitness
        )

        # Track palette change
        old_palette = mask_to_indices(population[best_idx].mask)
        new_palette = mask_to_indices(new_population[new_best_idx].mask)
        palette_changed = old_palette != new_palette

        new_state = {
            'population': new_population,
            'best_idx': new_best_idx,
            'global_innovation': global_innovation,
            'rng_key': key,
            'generation': generation + 1,
            'discovered_ever': discovered_ever,
            'exploration_counts': exploration_counts,
            'novelty_archive': state['novelty_archive'],
            'strategy_name': self.name,
        }

        # Compute population stats
        avg_novelty = sum(g.novelty for g in new_population) / len(new_population)
        avg_curiosity = sum(g.curiosity for g in new_population) / len(new_population)
        unique_palettes = len(set(tuple(mask_to_indices(g.mask)) for g in new_population))
        n_discovered = int(jnp.sum(discovered_ever > 0.5))

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': new_palette,
            'avg_novelty': float(avg_novelty),
            'avg_curiosity': float(avg_curiosity),
            'unique_palettes': unique_palettes,
            'total_discovered': n_discovered,
            'global_innovation': global_innovation,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including exploration stats."""
        palette = self.get_active_palette(state)
        population = state['population']
        discovered_ever = state['discovered_ever']

        sin_count = sum(1 for g in population if g.mask[4] > 0.5)
        n_discovered = int(jnp.sum(discovered_ever > 0.5))

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'pop_with_sin': sin_count,
            'total_discovered': n_discovered,
            'global_innovation': state['global_innovation'],
        }
