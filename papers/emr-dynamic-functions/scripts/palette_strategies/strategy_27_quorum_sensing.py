"""Strategy 27: Quorum Sensing (Population-Level Consensus).

Implements bacterial quorum sensing - coordinated behavior based on
population density and collective signaling. Applied to palette evolution,
functions "vote" on their usefulness and reach consensus through the population.

Biological Basis:
- Bacteria produce and sense signaling molecules (autoinducers)
- When autoinducer concentration exceeds threshold, behavior changes
- Individual bacteria coordinate through population-level signaling
- Examples: bioluminescence, biofilm formation, virulence

Key Insight:
- Previous strategies focus on individual fitness or pairwise correlations
- Quorum sensing asks: "What does the POPULATION collectively discover?"
- A function used by 30% of population at moderate fitness may be more
  valuable than one used by 5% at high fitness
- Population-level consensus is more robust than individual signals

Quorum Mechanism:
    # For each function, aggregate population "votes"
    for each individual in population:
        if individual uses function[i]:
            signal[i] += fitness_weight[individual]

    # Collective memory with decay
    collective[i] = decay * collective[i] + (1-decay) * signal[i]

    # Quorum decisions
    if collective[i] > quorum_threshold:
        # Quorum reached: function proven useful across population
        promote_to_stable_palette(i)
    elif collective[i] < minority_threshold:
        # Minority protection: allow exploration even for rare functions
        allow_in_exploration_slots(i)
    else:
        # Normal selection pressure
        standard_selection(i)

Expected improvements:
- Robust consensus (noisy individual signals → clean population signal)
- Protection of valuable rare functions (minority protection)
- Faster convergence when population agrees
- Natural exploration/exploitation balance
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


class QuorumSensingStrategy(PaletteEvolutionStrategy):
    """Population-level consensus for function selection.

    Functions accumulate "votes" from across the population. When a quorum
    is reached, the function is promoted to stable status. Minority
    protection ensures rare but useful functions aren't lost.
    """

    name = "quorum_sensing"
    description = "Population-level voting and consensus for function selection"

    def __init__(
        self,
        # Quorum parameters
        quorum_threshold: float = 0.4,        # 40% population consensus for promotion
        minority_threshold: float = 0.05,     # Below 5% = rare (protected exploration)
        signal_decay: float = 0.85,           # Collective memory persistence
        # Voting weights
        vote_weight_by_fitness: bool = True,  # High fitness = louder vote
        fitness_weight_power: float = 2.0,    # How much fitness affects vote weight
        # Function states
        stable_promotion_gens: int = 5,       # Consecutive quorum gens for stability
        unstable_after_gens: int = 10,        # Fall from stable after this many below quorum
        # Mutation rates
        stable_deactivate_rate: float = 0.01, # Very low for stable functions
        normal_activate_rate: float = 0.12,
        normal_deactivate_rate: float = 0.06,
        minority_activate_boost: float = 1.5, # Boost for minority exploration
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Quorum Sensing strategy.

        Args:
            quorum_threshold: Fraction of population for quorum
            minority_threshold: Fraction below which minority protection applies
            signal_decay: How long collective memory persists
            vote_weight_by_fitness: Whether fitness weights votes
            fitness_weight_power: Exponent for fitness weighting
            stable_promotion_gens: Consecutive generations above quorum for stability
            unstable_after_gens: Generations below quorum before losing stability
        """
        # Quorum
        self.quorum_threshold = quorum_threshold
        self.minority_threshold = minority_threshold
        self.signal_decay = signal_decay

        # Voting
        self.vote_weight_by_fitness = vote_weight_by_fitness
        self.fitness_weight_power = fitness_weight_power

        # Stability
        self.stable_promotion_gens = stable_promotion_gens
        self.unstable_after_gens = unstable_after_gens

        # Mutation
        self.stable_deactivate_rate = stable_deactivate_rate
        self.normal_activate_rate = normal_activate_rate
        self.normal_deactivate_rate = normal_deactivate_rate
        self.minority_activate_boost = minority_activate_boost

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with population collective memory."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Collective memory: population-level signal for each function
        collective_memory = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                # Initial palette starts with some signal
                collective_memory = collective_memory.at[i].set(0.3)

        # Function stability tracking
        stable_functions = []  # Functions that reached quorum consensus
        above_quorum_count = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)  # Consecutive gens above
        below_quorum_count = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)  # Consecutive gens below

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 272727),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Population sensing
            'collective_memory': collective_memory,
            'stable_functions': stable_functions,
            'above_quorum_count': above_quorum_count,
            'below_quorum_count': below_quorum_count,
            # Signal tracking
            'last_population_signal': jnp.zeros(NUM_ACTIVATIONS),
            'population_usage': jnp.zeros(NUM_ACTIVATIONS),  # Fraction using each
            # Tracking
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_population_signal(
        self,
        mask: jnp.ndarray,
        fitness: float,
        best_fitness: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute population signal from this individual's contribution.

        In a real population, we'd aggregate across all individuals.
        Here we simulate by using the elite's contribution weighted by
        relative fitness.

        Returns:
            (signal, usage) - per-function signal strength and usage indicator
        """
        active = (mask > 0.5).astype(jnp.float32)

        # Weight by fitness (normalized by best seen)
        if self.vote_weight_by_fitness and best_fitness > 0.01:
            relative_fitness = fitness / best_fitness
            vote_weight = relative_fitness ** self.fitness_weight_power
        else:
            vote_weight = 1.0

        # Each active function gets a weighted vote
        signal = active * vote_weight

        # Normalize by number of active (so total vote per individual is bounded)
        n_active = max(jnp.sum(active), 1.0)
        signal = signal / n_active

        return signal, active

    def _update_collective_memory(
        self,
        collective: jnp.ndarray,
        new_signal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update collective memory with new population signal.

        Exponential moving average maintains memory of past signals.
        """
        new_collective = (
            self.signal_decay * collective +
            (1 - self.signal_decay) * new_signal
        )
        return jnp.clip(new_collective, 0.0, 1.0)

    def _update_quorum_counts(
        self,
        collective: jnp.ndarray,
        above_count: jnp.ndarray,
        below_count: jnp.ndarray,
        stable_functions: List[int],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[int]]:
        """Update quorum status counters and stable function list."""
        new_above = above_count.copy()
        new_below = below_count.copy()
        new_stable = list(stable_functions)

        for i in range(NUM_ACTIVATIONS):
            signal = float(collective[i])

            if signal >= self.quorum_threshold:
                # Above quorum
                new_above = new_above.at[i].set(int(above_count[i]) + 1)
                new_below = new_below.at[i].set(0)

                # Check for promotion to stable
                if int(new_above[i]) >= self.stable_promotion_gens:
                    if i not in new_stable:
                        new_stable.append(i)

            else:
                # Below quorum
                new_below = new_below.at[i].set(int(below_count[i]) + 1)
                new_above = new_above.at[i].set(0)

                # Check for demotion from stable
                if i in new_stable:
                    if int(new_below[i]) >= self.unstable_after_gens:
                        new_stable.remove(i)

        return new_above, new_below, new_stable

    def _get_function_status(
        self,
        i: int,
        collective: jnp.ndarray,
        stable_functions: List[int],
    ) -> str:
        """Get status of a function: stable, quorum, normal, or minority."""
        if i in stable_functions:
            return 'stable'

        signal = float(collective[i])
        if signal >= self.quorum_threshold:
            return 'quorum'
        elif signal <= self.minority_threshold:
            return 'minority'
        else:
            return 'normal'

    def _apply_quorum_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        collective: jnp.ndarray,
        stable_functions: List[int],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with quorum-based rates."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        new_mask = mask.copy()
        activated = []
        deactivated = []
        status_counts = {'stable': 0, 'quorum': 0, 'normal': 0, 'minority': 0}

        for i in range(NUM_ACTIVATIONS):
            status = self._get_function_status(i, collective, stable_functions)
            status_counts[status] += 1
            signal = float(collective[i])

            if mask[i] < 0.5:
                # Inactive: maybe activate
                if status == 'stable' or status == 'quorum':
                    # Strong consensus: high activation probability
                    rate = self.normal_activate_rate * (1.0 + signal)
                elif status == 'minority':
                    # Minority protection: boost exploration
                    rate = self.normal_activate_rate * self.minority_activate_boost * 0.5
                else:
                    # Normal: standard rate scaled by signal
                    rate = self.normal_activate_rate * (0.3 + 0.7 * signal)

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)

            else:
                # Active: maybe deactivate
                if status == 'stable':
                    # Stable: very low deactivation
                    rate = self.stable_deactivate_rate
                elif status == 'quorum':
                    # At quorum but not stable: low deactivation
                    rate = self.normal_deactivate_rate * 0.2
                else:
                    # Normal or minority: standard rate inverse to signal
                    rate = self.normal_deactivate_rate * (1.0 - signal * 0.7)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'status_counts': status_counts,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with quorum sensing dynamics."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Compute population signal from current best
        new_signal, usage = self._compute_population_signal(
            state['mask'],
            best_fitness,
            new_best,
        )

        # Step 2: Update collective memory
        new_collective = self._update_collective_memory(
            state['collective_memory'],
            new_signal,
        )

        # Step 3: Update quorum status
        new_above, new_below, new_stable = self._update_quorum_counts(
            new_collective,
            state['above_quorum_count'],
            state['below_quorum_count'],
            state['stable_functions'],
        )

        # Step 4: Apply quorum-based mutation
        new_mask, mutation_info = self._apply_quorum_mutation(
            subkey,
            state['mask'],
            new_collective,
            new_stable,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

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
            # Population sensing
            'collective_memory': new_collective,
            'stable_functions': new_stable,
            'above_quorum_count': new_above,
            'below_quorum_count': new_below,
            # Signal tracking
            'last_population_signal': new_signal,
            'population_usage': usage,
            # Tracking
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Functions at different status levels
        at_quorum = [
            i for i in range(NUM_ACTIVATIONS)
            if new_collective[i] >= self.quorum_threshold
        ]
        at_minority = [
            i for i in range(NUM_ACTIVATIONS)
            if new_collective[i] <= self.minority_threshold
        ]

        # Top collective memory
        top_coll_idx = jnp.argsort(new_collective)[-3:][::-1]
        top_collective = [(int(i), float(new_collective[i])) for i in top_coll_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Quorum
            'n_stable': len(new_stable),
            'stable_functions': new_stable,
            'n_at_quorum': len(at_quorum),
            'at_quorum_functions': at_quorum,
            'n_at_minority': len(at_minority),
            # Collective memory
            'mean_collective': float(jnp.mean(new_collective)),
            'max_collective': float(jnp.max(new_collective)),
            'top_collective_functions': top_collective,
            'sin_collective': float(new_collective[4]),
            'sin_stable': 4 in new_stable,
            # Sin status
            'has_sin': 4 in active_palette,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with quorum status."""
        palette = self.get_active_palette(state)
        collective = state['collective_memory']

        # Top functions by collective memory
        top_idx = jnp.argsort(collective)[-5:][::-1]
        top_collective = [(int(i), float(collective[i])) for i in top_idx]

        # Status breakdown
        at_quorum = sum(1 for c in collective if c >= self.quorum_threshold)
        at_minority = sum(1 for c in collective if c <= self.minority_threshold)

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Quorum status
            'stable_functions': state['stable_functions'],
            'n_stable': len(state['stable_functions']),
            'n_at_quorum': at_quorum,
            'n_at_minority': at_minority,
            # Collective memory
            'top_collective_functions': top_collective,
            'mean_collective': float(jnp.mean(collective)),
            # Sin-specific
            'sin_collective': float(collective[4]),
            'sin_stable': 4 in state['stable_functions'],
        }
