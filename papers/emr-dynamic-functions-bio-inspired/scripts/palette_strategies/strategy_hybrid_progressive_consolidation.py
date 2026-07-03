"""Hybrid Strategy: Progressive + Consolidation Window.

Combines two powerful continual learning strategies:
- Progressive Palette (40): Freeze successful functions, only add
- Consolidation Window (19): Sleep/wake cycles with memory consolidation

Expected Benefits:
- Zero forgetting of frozen functions (from Progressive)
- Better transfer through consolidation (from Consolidation Window)
- Sleep phases consolidate before freezing
- Combining guarantees (Progressive) with optimization (Consolidation)

Hybrid Mechanism:
    # Wake phase: Normal evolution with Progressive additions
    if not in_sleep_phase:
        # Track contributions, possibly add new functions
        new_additions = progressive_mutate(...)

    # Sleep phase (consolidation): Evaluate what to freeze
    if in_sleep_phase:
        # Replay high-fitness patterns
        boost_high_performers()
        # Check if palette should be frozen
        if consolidation_suggests_freeze:
            freeze_successful_additions()

    # Freeze guarantee: Once frozen, NEVER remove
    current_palette = frozen + additions
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


class ProgressiveConsolidationStrategy(PaletteEvolutionStrategy):
    """Hybrid: Progressive Palette + Consolidation Window.

    Combines guaranteed zero forgetting (frozen functions never removed)
    with sleep/wake consolidation cycles for better transfer learning.
    """

    name = "progressive_consolidation"
    description = "Progressive freezing with sleep/wake consolidation cycles"

    def __init__(
        self,
        # Progressive parameters
        inherit_threshold: float = 0.85,
        freeze_delay: int = 5,
        max_additions_per_task: int = 3,
        max_total_palette: int = 10,
        # Consolidation parameters
        consolidation_frequency: int = 10,  # Every N gens, consolidate
        consolidation_duration: int = 3,    # Gens of consolidation
        replay_strength: float = 1.5,       # Boost for high-affinity during consolidation
        transfer_rate: float = 0.1,         # Addition → frozen transfer during sleep
        # Shared parameters
        contribution_window: int = 10,
        exploration_rate: float = 0.15,
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        # Progressive
        self.inherit_threshold = inherit_threshold
        self.freeze_delay = freeze_delay
        self.max_additions_per_task = max_additions_per_task
        self.max_total_palette = max_total_palette

        # Consolidation
        self.consolidation_frequency = consolidation_frequency
        self.consolidation_duration = consolidation_duration
        self.replay_strength = replay_strength
        self.transfer_rate = transfer_rate

        # Shared
        self.contribution_window = contribution_window
        self.exploration_rate = exploration_rate
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with both Progressive and Consolidation tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                contribution = contribution.at[i].set(0.3)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 191940),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Progressive state
            'frozen_palette': [],
            'active_additions': list(initial),
            'freeze_counter': 0,
            # Consolidation state
            'in_consolidation': False,
            'consolidation_countdown': 0,
            'working_memory': {},  # {func: recent_fitness}
            'long_term_memory': {},  # {func: consolidated_strength}
            # Contribution tracking
            'contribution': contribution,
            'contribution_history': {i: [] for i in range(NUM_ACTIVATIONS)},
            # History
            'freeze_events': [],
            'consolidation_events': [],
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

    def _update_working_memory(
        self,
        working_memory: Dict,
        additions: List[int],
        best_fitness: float,
    ) -> Dict:
        """Update working memory with recent fitness."""
        new_wm = dict(working_memory)
        for func in additions:
            if func not in new_wm:
                new_wm[func] = []
            new_wm[func].append(best_fitness)
            if len(new_wm[func]) > self.contribution_window:
                new_wm[func] = new_wm[func][-self.contribution_window:]
        return new_wm

    def _run_consolidation(
        self,
        working_memory: Dict,
        long_term_memory: Dict,
        contribution: jnp.ndarray,
        additions: List[int],
    ) -> Tuple[Dict, jnp.ndarray, List[int]]:
        """Run consolidation: transfer working → long-term memory."""
        new_ltm = dict(long_term_memory)
        new_contribution = contribution.copy()
        promoted_to_freeze = []

        for func in additions:
            wm_history = working_memory.get(func, [])
            if len(wm_history) >= 3:
                avg_fitness = np.mean(wm_history)

                # Update long-term memory
                current_ltm = new_ltm.get(func, 0.0)
                new_ltm[func] = current_ltm + self.transfer_rate * avg_fitness

                # Boost contribution during consolidation
                boost = self.replay_strength * avg_fitness
                new_contribution = new_contribution.at[func].add(boost)

                # Check if should be promoted to frozen
                if new_ltm[func] >= self.inherit_threshold:
                    promoted_to_freeze.append(func)

        return new_ltm, jnp.clip(new_contribution, 0, 2.0), promoted_to_freeze

    def _mutate_additions(
        self,
        frozen: List[int],
        additions: List[int],
        contribution: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> List[int]:
        """Mutate additions (can't touch frozen)."""
        key1, key2 = jax.random.split(key)
        new_additions = list(additions)
        total_in_palette = set(frozen) | set(additions)

        # Possibly explore new addition
        if (jax.random.uniform(key1) < self.exploration_rate and
            len(total_in_palette) < self.max_total_palette):
            available = [i for i in range(NUM_ACTIVATIONS) if i not in total_in_palette]
            if available:
                new_func = available[int(jax.random.randint(key2, (), 0, len(available)))]
                new_additions.append(new_func)

        # Limit additions
        if len(new_additions) > self.max_additions_per_task:
            # Remove lowest contribution
            contrib_scores = [(i, float(contribution[i])) for i in new_additions]
            contrib_scores.sort(key=lambda x: x[1])
            new_additions.remove(contrib_scores[0][0])

        return new_additions

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with Progressive + Consolidation hybrid dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        frozen = list(state['frozen_palette'])
        additions = list(state['active_additions'])
        contribution = state['contribution']
        working_memory = dict(state['working_memory'])
        long_term_memory = dict(state['long_term_memory'])

        # Check consolidation state
        in_consolidation = state['in_consolidation']
        consolidation_countdown = state['consolidation_countdown']
        consolidation_events = list(state['consolidation_events'])
        freeze_events = list(state['freeze_events'])
        promoted = []

        # Update working memory
        working_memory = self._update_working_memory(working_memory, additions, best_fitness)

        # Update contribution
        new_contribution = contribution * 0.95
        for func in mask_to_indices(state['mask']):
            if improved:
                new_contribution = new_contribution.at[func].add(0.2)
            else:
                new_contribution = new_contribution.at[func].add(0.02)
        new_contribution = jnp.clip(new_contribution, 0, 2.0)

        # Consolidation cycle management
        if in_consolidation:
            consolidation_countdown -= 1
            if consolidation_countdown <= 0:
                # End consolidation, run transfer
                long_term_memory, new_contribution, promoted = self._run_consolidation(
                    working_memory, long_term_memory, new_contribution, additions
                )
                in_consolidation = False

                # Promote to frozen
                for func in promoted:
                    if func in additions and func not in frozen:
                        frozen.append(func)
                        additions.remove(func)
                        freeze_events.append((generation, func, 'consolidation_promoted'))

        elif (generation + 1) % self.consolidation_frequency == 0:
            # Start consolidation
            in_consolidation = True
            consolidation_countdown = self.consolidation_duration
            consolidation_events.append((generation, 'start'))

        # Check for fitness-based freezing (Progressive mechanism)
        if best_fitness >= self.inherit_threshold:
            freeze_counter = state['freeze_counter'] + 1
            if freeze_counter >= self.freeze_delay:
                # Freeze all successful additions
                for func in list(additions):
                    if func not in frozen:
                        frozen.append(func)
                        freeze_events.append((generation, func, 'fitness_freeze'))
                additions = []
                freeze_counter = 0
        else:
            freeze_counter = 0

        # Wake phase: mutate additions (only when not consolidating)
        if not in_consolidation:
            additions = self._mutate_additions(frozen, additions, new_contribution, k1)

        # Create new mask
        new_mask = self._update_mask_from_palette(frozen, additions)
        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Fitness history
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
            'frozen_palette': frozen,
            'active_additions': additions,
            'freeze_counter': freeze_counter,
            # Consolidation state
            'in_consolidation': in_consolidation,
            'consolidation_countdown': consolidation_countdown,
            'working_memory': working_memory,
            'long_term_memory': long_term_memory,
            # Contribution
            'contribution': new_contribution,
            'contribution_history': state['contribution_history'],
            # History
            'freeze_events': freeze_events[-50:],
            'consolidation_events': consolidation_events[-50:],
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        active_palette = mask_to_indices(new_mask)
        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Progressive
            'n_frozen': len(frozen),
            'frozen_palette': frozen,
            'n_additions': len(additions),
            'active_additions': additions,
            # Consolidation
            'in_consolidation': in_consolidation,
            'consolidation_countdown': consolidation_countdown,
            'promoted_this_gen': promoted,
            # Stats
            'total_freeze_events': len(freeze_events),
            'total_consolidations': len([e for e in consolidation_events if e[1] == 'start']),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_is_frozen': 4 in frozen,
            'sin_contribution': float(new_contribution[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        palette = self.get_active_palette(state)
        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'n_frozen': len(state['frozen_palette']),
            'n_additions': len(state['active_additions']),
            'in_consolidation': state['in_consolidation'],
            'total_freeze_events': len(state['freeze_events']),
        }
