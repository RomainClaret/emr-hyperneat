"""Strategy 32: Ant Colony Pheromone (Population Stigmergy).

Implements ant colony optimization principles for palette evolution. Functions
accumulate "pheromone" based on their contribution to fitness, creating
persistent attraction fields that guide future selection.

Biological Basis:
- Ants communicate indirectly through environment (stigmergy)
- Pheromone trails mark successful paths
- Trails evaporate over time, requiring reinforcement
- Population follows strongest trails with some exploration

Key Insight:
- Current strategies are either individual (Hebbian) or direct voting (QuorumSensing)
- Pheromone trails create persistent attraction fields
- No direct communication - only environmental modification
- Creates emergent consensus through accumulated history

Pheromone Mechanism:
    # Evaporation (decay)
    pheromone *= decay_rate

    # Deposit on successful functions
    if fitness_improved:
        pheromone[active_functions] += deposit_amount * improvement

    # Elite reinforcement
    pheromone[elite_functions] += elite_bonus

    # Selection follows pheromone with exploration
    selection_probs = softmax(pheromone / temperature)
    palette = top_k(selection_probs) with exploration_noise

Expected improvements:
- Emergent consensus through accumulated history
- Exploration via pheromone evaporation
- Elite reinforcement stabilizes good discoveries
- No explicit coordination required
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


class AntColonyPheromoneStrategy(PaletteEvolutionStrategy):
    """Population-level stigmergy through pheromone trails.

    Functions accumulate pheromone based on fitness contribution.
    Selection follows pheromone concentrations with exploration noise.
    Pheromone evaporates over time, requiring continuous reinforcement.

    Lock mechanism prevents losing discovered
    high-value functions (like sin) due to pheromone evaporation.
    """

    name = "ant_colony_pheromone"
    description = "Ant colony pheromone trails for emergent function consensus"

    def __init__(
        self,
        # Pheromone dynamics
        pheromone_decay: float = 0.85,          # Evaporation rate per generation
        pheromone_deposit: float = 0.3,         # Base deposit on success
        pheromone_min: float = 0.05,            # Minimum pheromone level
        pheromone_max: float = 3.0,             # Maximum pheromone level
        initial_pheromone: float = 0.5,         # Starting pheromone level
        # Elite reinforcement
        elite_bonus: float = 2.0,               # Extra deposit from elites
        elite_threshold: float = 0.9,           # Fitness percentile for elite
        # Selection parameters
        temperature: float = 0.5,               # Softmax temperature
        exploration_rate: float = 0.15,         # Random exploration probability
        follow_probability: float = 0.85,       # Follow pheromone vs explore
        # Palette composition
        palette_size: int = 6,                  # Target palette size
        min_active: int = 2,
        initial_palette: List[int] = None,
        # Lock discovered functions (data-driven, no hardcoding)
        lock_on_improvement: bool = True,       # Lock functions when fitness improves
        lock_threshold: float = 0.05,           # Min improvement to trigger lock (higher = more selective)
        locked_pheromone_floor: float = 2.0,    # Min pheromone for locked functions
        lock_new_only: bool = True,             # Only lock functions not in initial palette
    ):
        """Initialize Ant Colony Pheromone strategy.

        Args:
            pheromone_decay: Evaporation rate (0-1, lower = faster decay)
            pheromone_deposit: Amount deposited on fitness improvement
            pheromone_min: Minimum pheromone (prevents complete extinction)
            pheromone_max: Maximum pheromone (prevents runaway accumulation)
            initial_pheromone: Starting pheromone for all functions
            elite_bonus: Multiplier for elite function deposits
            elite_threshold: Fitness threshold for elite status
            temperature: Softmax temperature for selection
            exploration_rate: Probability of random exploration
            follow_probability: Probability of following strongest trails
            palette_size: Target number of active functions
            lock_on_improvement: Whether to lock functions when fitness improves
            lock_threshold: Minimum improvement to trigger locking
            locked_pheromone_floor: Minimum pheromone level for locked functions
            lock_new_only: If True, only lock newly discovered functions (not in initial palette)
        """
        # Pheromone dynamics
        self.pheromone_decay = pheromone_decay
        self.pheromone_deposit = pheromone_deposit
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max
        self.initial_pheromone = initial_pheromone

        # Elite reinforcement
        self.elite_bonus = elite_bonus
        self.elite_threshold = elite_threshold

        # Selection
        self.temperature = temperature
        self.exploration_rate = exploration_rate
        self.follow_probability = follow_probability

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

        # Lock mechanism (data-driven)
        self.lock_on_improvement = lock_on_improvement
        self.lock_threshold = lock_threshold
        self.locked_pheromone_floor = locked_pheromone_floor
        self.lock_new_only = lock_new_only

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with pheromone levels."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize pheromone levels
        pheromone = jnp.ones(NUM_ACTIVATIONS) * self.initial_pheromone

        # Boost initial palette pheromone
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                pheromone = pheromone.at[i].set(self.initial_pheromone * 1.5)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 323232),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Pheromone state
            'pheromone': pheromone,
            # Tracking
            'deposits_this_gen': jnp.zeros(NUM_ACTIVATIONS),
            'total_deposits': jnp.zeros(NUM_ACTIVATIONS),
            'elite_count': jnp.zeros(NUM_ACTIVATIONS),
            'previous_mask': mask,
            'fitness_history': [],
            # Locked functions (won't be removed from palette)
            'locked_functions': set(),
            'lock_generation': {},  # func_idx -> generation when locked
            'initial_palette_set': set(initial),  # Track what was in initial palette
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _evaporate_pheromone(
        self,
        pheromone: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply pheromone evaporation (decay)."""
        new_pheromone = pheromone * self.pheromone_decay
        return jnp.clip(new_pheromone, self.pheromone_min, self.pheromone_max)

    def _deposit_pheromone(
        self,
        pheromone: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
        is_elite: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Deposit pheromone on active functions based on improvement.

        Returns:
            (new_pheromone, deposits)
        """
        deposits = jnp.zeros(NUM_ACTIVATIONS)

        if improvement > 0:
            # Base deposit proportional to improvement
            base_deposit = self.pheromone_deposit * improvement

            # Elite bonus
            if is_elite:
                base_deposit *= self.elite_bonus

            # Deposit on active functions
            for i in range(NUM_ACTIVATIONS):
                if mask[i] > 0.5:
                    deposits = deposits.at[i].set(base_deposit)

        new_pheromone = pheromone + deposits
        new_pheromone = jnp.clip(new_pheromone, self.pheromone_min, self.pheromone_max)

        return new_pheromone, deposits

    def _select_palette(
        self,
        pheromone: jnp.ndarray,
        key: jax.random.PRNGKey,
        locked_functions: set = None,
    ) -> jnp.ndarray:
        """Select palette based on pheromone levels with exploration.

        Locked functions are always included in the palette.
        """
        key1, key2, key3 = jax.random.split(key, 3)
        locked_functions = locked_functions or set()

        # Start with locked functions always included
        mask = jnp.zeros(NUM_ACTIVATIONS)
        for idx in locked_functions:
            if 0 <= idx < NUM_ACTIVATIONS:
                mask = mask.at[idx].set(1.0)

        # Calculate remaining slots
        remaining_slots = max(0, self.palette_size - len(locked_functions))

        if remaining_slots == 0:
            return mask

        # Compute selection probabilities via softmax
        probs = jax.nn.softmax(pheromone / self.temperature)

        # Zero out probabilities for already-locked functions
        for idx in locked_functions:
            if 0 <= idx < NUM_ACTIVATIONS:
                probs = probs.at[idx].set(0.0)
        probs = probs / jnp.sum(probs)  # Renormalize

        # Decide: follow pheromone or explore
        follow = jax.random.uniform(key1) < self.follow_probability

        if follow:
            # Follow pheromone trails (top-k by probability, excluding locked)
            top_k = jnp.argsort(probs)[-remaining_slots:]
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            # Exploration: sample from distribution
            selected = set()
            remaining_key = key2

            for _ in range(remaining_slots):
                remaining_key, subkey = jax.random.split(remaining_key)
                # Add exploration noise to probabilities
                noise = jax.random.uniform(subkey, (NUM_ACTIVATIONS,)) * self.exploration_rate
                noisy_probs = probs + noise
                # Zero locked functions in noisy probs too
                for idx in locked_functions:
                    if 0 <= idx < NUM_ACTIVATIONS:
                        noisy_probs = noisy_probs.at[idx].set(0.0)
                noisy_probs = noisy_probs / jnp.sum(noisy_probs)

                sample = jax.random.choice(subkey, NUM_ACTIVATIONS, p=noisy_probs)
                selected.add(int(sample))

            for idx in selected:
                mask = mask.at[idx].set(1.0)

        return mask

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with ant colony pheromone dynamics.

        Includes locking mechanism to retain discovered high-value functions.
        """
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Determine if this is an elite performance
        is_elite = best_fitness >= self.elite_threshold

        # Get current palette
        current_palette = mask_to_indices(state['mask'])

        # Lock functions on significant improvement (data-driven)
        locked_functions = set(state.get('locked_functions', set()))
        lock_generation = dict(state.get('lock_generation', {}))
        initial_palette_set = state.get('initial_palette_set', set())
        newly_locked = []

        if self.lock_on_improvement and improvement >= self.lock_threshold:
            # Lock functions that contributed to this improvement
            for func_idx in current_palette:
                if func_idx in locked_functions:
                    continue  # Already locked

                # If lock_new_only, skip functions that were in initial palette
                if self.lock_new_only and func_idx in initial_palette_set:
                    continue

                # Lock this function - it was present during significant improvement
                locked_functions.add(func_idx)
                lock_generation[func_idx] = generation
                newly_locked.append(func_idx)

        # Step 1: Evaporate pheromone
        new_pheromone = self._evaporate_pheromone(state['pheromone'])

        # Ensure locked functions maintain minimum pheromone
        for func_idx in locked_functions:
            if 0 <= func_idx < NUM_ACTIVATIONS:
                current_pher = float(new_pheromone[func_idx])
                if current_pher < self.locked_pheromone_floor:
                    new_pheromone = new_pheromone.at[func_idx].set(self.locked_pheromone_floor)

        # Step 2: Deposit pheromone on successful functions
        new_pheromone, deposits = self._deposit_pheromone(
            new_pheromone,
            state['mask'],
            improvement,
            is_elite,
        )

        # Step 3: Select new palette based on pheromone (respecting locks)
        new_mask = self._select_palette(new_pheromone, k1, locked_functions)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update tracking
        new_total_deposits = state['total_deposits'] + deposits
        new_elite_count = state['elite_count']
        if is_elite:
            for i in range(NUM_ACTIVATIONS):
                if state['mask'][i] > 0.5:
                    new_elite_count = new_elite_count.at[i].add(1)

        # Track fitness
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Pheromone state
            'pheromone': new_pheromone,
            # Tracking
            'deposits_this_gen': deposits,
            'total_deposits': new_total_deposits,
            'elite_count': new_elite_count,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Lock state
            'locked_functions': locked_functions,
            'lock_generation': lock_generation,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Top pheromone functions
        top_pher_idx = jnp.argsort(new_pheromone)[-5:][::-1]
        top_pheromone = [(int(i), float(new_pheromone[i])) for i in top_pher_idx]

        # Pheromone distribution stats
        active_pheromone = [float(new_pheromone[i]) for i in active_palette]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Pheromone stats
            'mean_pheromone': float(jnp.mean(new_pheromone)),
            'max_pheromone': float(jnp.max(new_pheromone)),
            'min_pheromone': float(jnp.min(new_pheromone)),
            'active_mean_pheromone': float(np.mean(active_pheromone)) if active_pheromone else 0.0,
            'top_pheromone': top_pheromone,
            # Deposits
            'total_deposit_this_gen': float(jnp.sum(deposits)),
            'is_elite': is_elite,
            # Lock status (data-driven, no hardcoded functions)
            'locked_functions': list(locked_functions),
            'newly_locked': newly_locked,
            'num_locked': len(locked_functions),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with pheromone status."""
        palette = self.get_active_palette(state)
        pheromone = state['pheromone']
        locked_functions = state.get('locked_functions', set())
        lock_generation = state.get('lock_generation', {})

        # Top functions by pheromone
        top_pher = jnp.argsort(pheromone)[-5:][::-1]
        top_pheromone = [(int(i), float(pheromone[i])) for i in top_pher]

        # Most reinforced functions
        top_deposits = jnp.argsort(state['total_deposits'])[-5:][::-1]
        most_reinforced = [(int(i), float(state['total_deposits'][i])) for i in top_deposits]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Pheromone
            'top_pheromone': top_pheromone,
            'mean_pheromone': float(jnp.mean(pheromone)),
            'pheromone_range': (float(jnp.min(pheromone)), float(jnp.max(pheromone))),
            # Deposits
            'most_reinforced': most_reinforced,
            'total_elite_selections': int(jnp.sum(state['elite_count'])),
            # Lock status (data-driven)
            'locked_functions': list(locked_functions),
            'lock_generation': lock_generation,
            'num_locked': len(locked_functions),
        }
