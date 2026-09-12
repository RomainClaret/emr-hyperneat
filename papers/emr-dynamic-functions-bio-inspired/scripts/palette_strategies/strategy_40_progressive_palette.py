"""Strategy 40: Progressive Palette (Inherit-and-Expand).

Implements progressive neural network principles for palette evolution.
Successful palettes are frozen and inherited to future tasks. New learning
can only ADD functions, never remove inherited ones.

Biological Basis:
- Developmental canalization locks in successful phenotypes
- Neural circuits crystallize after critical periods
- New learning builds on consolidated foundations
- "Scaffold" neurons support learning of new circuits

Key Insight:
- Current strategies risk losing good discoveries to exploration
- Progressive architecture GUARANTEES no forgetting of frozen functions
- New tasks can only expand the palette, not contract it
- Creates monotonic knowledge accumulation

Progressive Mechanism:
    # On task completion (fitness > threshold):
    if best_fitness >= inherit_threshold:
        frozen_palette.extend(current_successful_functions)
        lock them permanently

    # Current palette = frozen + additions
    current_palette = frozen_palette + active_additions

    # Evolution can ONLY modify active_additions
    # frozen_palette is IMMUTABLE

    # Limit growth: max_additions_per_task
    if len(active_additions) > max_additions_per_task:
        prune least useful additions

Expected improvements:
- GUARANTEED zero forgetting of frozen functions
- Monotonic capability growth over tasks
- Explicit separation of preserved vs experimental
- Built-in continual learning architecture
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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


class ProgressivePaletteStrategy(PaletteEvolutionStrategy):
    """Inherit-and-expand architecture for zero forgetting.

    Functions are categorized as frozen (inherited) or additions (mutable).
    Frozen functions are NEVER removed. New tasks can only add to the
    palette, creating monotonic knowledge growth.
    """

    name = "progressive_palette"
    description = "Inherit successful palette, only add never remove"

    def __init__(
        self,
        # Freezing thresholds
        inherit_threshold: float = 0.85,        # Fitness to consider task "solved"
        partial_freeze_threshold: float = 0.7,  # Partial success threshold
        freeze_delay: int = 5,                  # Generations above threshold before freeze
        # Addition constraints
        max_additions_per_task: int = 3,        # Limit growth per task
        max_total_palette: int = 10,            # Maximum palette size overall
        addition_mutation_rate: float = 0.15,   # Mutation rate for additions
        # Contribution tracking for pruning
        contribution_window: int = 10,          # Window for contribution estimation
        min_contribution_for_keep: float = 0.0, # Min contribution to keep addition
        # Exploration of additions
        exploration_rate: float = 0.2,          # Chance to try new addition
        # Initial palette
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Progressive Palette strategy.

        Args:
            inherit_threshold: Fitness threshold for full freezing
            partial_freeze_threshold: Fitness for partial protection
            freeze_delay: Generations above threshold before freezing
            max_additions_per_task: Maximum new functions per task
            max_total_palette: Maximum total palette size
            addition_mutation_rate: Rate of adding/removing additions
            contribution_window: Generations to track for contribution
            min_contribution_for_keep: Minimum contribution to retain addition
            exploration_rate: Probability of exploring new function
            palette_size: Initial target palette size
        """
        # Freezing
        self.inherit_threshold = inherit_threshold
        self.partial_freeze_threshold = partial_freeze_threshold
        self.freeze_delay = freeze_delay

        # Additions
        self.max_additions_per_task = max_additions_per_task
        self.max_total_palette = max_total_palette
        self.addition_mutation_rate = addition_mutation_rate

        # Contribution
        self.contribution_window = contribution_window
        self.min_contribution_for_keep = min_contribution_for_keep

        # Exploration
        self.exploration_rate = exploration_rate

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with frozen/additions tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 404040),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Progressive state
            'frozen_palette': [],               # List of frozen function indices
            'active_additions': list(initial),  # Mutable additions
            'freeze_counter': 0,                # Generations above threshold
            'task_number': 0,                   # For multi-task tracking
            # Contribution tracking
            'addition_fitness_history': {i: [] for i in range(NUM_ACTIVATIONS)},
            'addition_contributions': jnp.zeros(NUM_ACTIVATIONS),
            # History
            'freeze_events': [],                # List of (gen, functions frozen)
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette (frozen + additions)."""
        return mask_to_indices(state['mask'])

    def _update_mask_from_palette(
        self,
        frozen: List[int],
        additions: List[int],
    ) -> jnp.ndarray:
        """Create mask from frozen palette and additions."""
        mask = jnp.zeros(NUM_ACTIVATIONS)
        for i in frozen:
            if 0 <= i < NUM_ACTIVATIONS:
                mask = mask.at[i].set(1.0)
        for i in additions:
            if 0 <= i < NUM_ACTIVATIONS:
                mask = mask.at[i].set(1.0)
        return mask

    def _update_contributions(
        self,
        contributions: jnp.ndarray,
        additions: List[int],
        fitness_history: Dict[int, List],
        current_fitness: float,
    ) -> jnp.ndarray:
        """Update contribution estimates for additions."""
        new_contributions = contributions.copy()

        for i in additions:
            history = fitness_history.get(i, [])
            history.append(current_fitness)
            if len(history) > self.contribution_window:
                history = history[-self.contribution_window:]
            fitness_history[i] = history

            if len(history) >= 3:
                # Simple contribution: mean fitness when in palette
                new_contributions = new_contributions.at[i].set(np.mean(history))

        return new_contributions

    def _check_for_freeze(
        self,
        best_fitness: float,
        freeze_counter: int,
        frozen: List[int],
        additions: List[int],
    ) -> Tuple[List[int], List[int], int, bool]:
        """Check if we should freeze the current palette."""
        freeze_event = False

        if best_fitness >= self.inherit_threshold:
            new_counter = freeze_counter + 1
            if new_counter >= self.freeze_delay:
                # Freeze successful additions
                new_frozen = list(frozen)
                for i in additions:
                    if i not in new_frozen:
                        new_frozen.append(i)
                new_additions = []  # Clear additions after freeze
                freeze_event = True
                return new_frozen, new_additions, 0, freeze_event
            else:
                return frozen, additions, new_counter, freeze_event
        else:
            return frozen, additions, 0, freeze_event

    def _mutate_additions(
        self,
        frozen: List[int],
        additions: List[int],
        contributions: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> List[int]:
        """Mutate the additions (can't touch frozen)."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_additions = list(additions)

        # Current total palette
        total_in_palette = set(frozen) | set(additions)

        # Possibly explore new addition
        if (jax.random.uniform(key1) < self.exploration_rate and
            len(total_in_palette) < self.max_total_palette):

            # Pick from functions not in palette
            available = [i for i in range(NUM_ACTIVATIONS) if i not in total_in_palette]
            if available:
                new_func = available[int(jax.random.randint(key2, (), 0, len(available)))]
                new_additions.append(new_func)

        # Possibly remove low-contribution addition
        if len(new_additions) > self.max_additions_per_task:
            # Remove lowest contribution
            contrib_scores = [(i, float(contributions[i])) for i in new_additions]
            contrib_scores.sort(key=lambda x: x[1])
            # Remove lowest if below threshold
            for func, contrib in contrib_scores:
                if contrib < self.min_contribution_for_keep and len(new_additions) > 0:
                    new_additions.remove(func)
                    break

        # Random mutation of additions
        if jax.random.uniform(key3) < self.addition_mutation_rate:
            if new_additions and len(total_in_palette) < self.max_total_palette:
                # Remove random addition and add new one
                remove_idx = int(jax.random.randint(key3, (), 0, len(new_additions)))
                removed = new_additions.pop(remove_idx)

                # Add new function
                available = [i for i in range(NUM_ACTIVATIONS)
                            if i not in frozen and i not in new_additions]
                if available:
                    key3, subkey = jax.random.split(key3)
                    new_func = available[int(jax.random.randint(subkey, (), 0, len(available)))]
                    new_additions.append(new_func)

        return new_additions

    def notify_task_switch(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task switch - freeze successful palette.

        Call this method when switching to a new task to potentially
        freeze the current palette.
        """
        best_fitness = state['best_fitness_seen']

        # Check if current task was solved
        if best_fitness >= self.inherit_threshold:
            # Freeze current additions
            new_frozen = list(state['frozen_palette'])
            for i in state['active_additions']:
                if i not in new_frozen:
                    new_frozen.append(i)

            # Clear additions for new task
            new_additions = []

            # Create new mask
            new_mask = self._update_mask_from_palette(new_frozen, new_additions)

            # Update state
            state = dict(state)
            state['frozen_palette'] = new_frozen
            state['active_additions'] = new_additions
            state['mask'] = new_mask
            state['task_number'] = state['task_number'] + 1
            state['best_fitness_seen'] = 0.0
            state['freeze_events'] = state['freeze_events'] + [(state['generation'], new_frozen)]

        return state

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with progressive palette dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update contribution tracking
        new_contributions = self._update_contributions(
            state['addition_contributions'],
            state['active_additions'],
            state['addition_fitness_history'],
            best_fitness,
        )

        # Step 2: Check for freeze
        new_frozen, new_additions, new_freeze_counter, freeze_event = self._check_for_freeze(
            best_fitness,
            state['freeze_counter'],
            state['frozen_palette'],
            state['active_additions'],
        )

        # Step 3: Mutate additions (only if not just frozen)
        if not freeze_event:
            new_additions = self._mutate_additions(
                new_frozen,
                new_additions,
                new_contributions,
                k1,
            )

        # Step 4: Create new mask
        new_mask = self._update_mask_from_palette(new_frozen, new_additions)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track freeze events
        freeze_events = list(state['freeze_events'])
        if freeze_event:
            freeze_events.append((generation, list(new_frozen)))

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
            # Progressive state
            'frozen_palette': new_frozen,
            'active_additions': new_additions,
            'freeze_counter': new_freeze_counter,
            'task_number': state['task_number'],
            # Contribution tracking
            'addition_fitness_history': state['addition_fitness_history'],
            'addition_contributions': new_contributions,
            # History
            'freeze_events': freeze_events,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Addition contributions
        addition_contribs = [(i, float(new_contributions[i])) for i in new_additions]

        # Approaching freeze
        approaching_freeze = new_freeze_counter > 0

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Progressive state
            'n_frozen': len(new_frozen),
            'frozen_palette': new_frozen,
            'n_additions': len(new_additions),
            'active_additions': new_additions,
            'total_palette_size': len(active_palette),
            # Freeze status
            'freeze_event_this_gen': freeze_event,
            'freeze_counter': new_freeze_counter,
            'approaching_freeze': approaching_freeze,
            'gens_until_freeze': self.freeze_delay - new_freeze_counter if approaching_freeze else -1,
            # Contributions
            'addition_contributions': addition_contribs,
            # Growth tracking
            'max_capacity_reached': len(active_palette) >= self.max_total_palette,
            'total_freeze_events': len(freeze_events),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_is_frozen': 4 in new_frozen,
            'sin_is_addition': 4 in new_additions,
            'sin_contribution': float(new_contributions[4]) if 4 in new_additions else 0.0,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with progressive status."""
        palette = self.get_active_palette(state)
        frozen = state['frozen_palette']
        additions = state['active_additions']
        contributions = state['addition_contributions']

        # Addition contributions
        addition_contribs = [(i, float(contributions[i])) for i in additions]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Progressive
            'frozen_palette': frozen,
            'n_frozen': len(frozen),
            'active_additions': additions,
            'n_additions': len(additions),
            # Freeze
            'freeze_counter': state['freeze_counter'],
            'task_number': state['task_number'],
            'total_freeze_events': len(state['freeze_events']),
            # Contributions
            'addition_contributions': addition_contribs,
            # Capacity
            'capacity_used': len(palette) / self.max_total_palette,
            # Sin-specific
            'sin_is_frozen': 4 in frozen,
            'sin_is_addition': 4 in additions,
        }
