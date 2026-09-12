"""Strategy 41: Adult Neurogenesis (Hippocampal-Inspired Palette Birth).

Implements adult neurogenesis principles for palette evolution. New functions
can be "born" into the palette, go through a maturation period with high
plasticity, and only survive if they contribute to fitness.

Biological Basis:
- Adult hippocampus generates new neurons throughout life
- New neurons are highly plastic and excitable initially
- Integration period determines if new neurons survive
- Only neurons that form useful connections persist
- Plays role in learning and memory consolidation

Key Insight:
- Current strategies only mutate existing palette, never truly "birth" new functions
- Adult neurogenesis allows controlled introduction of new capabilities
- Young functions have higher learning rates (more plastic)
- Competition for survival prevents bloat
- Creates adaptive exploration without destabilizing existing knowledge

Neurogenesis Mechanism:
    # Each generation, possibly birth new function
    if random() < neurogenesis_rate:
        new_func = sample_from_available()
        young_neurons[new_func] = {
            'birth_gen': current_gen,
            'contribution': 0.0,
            'plasticity': young_plasticity,
        }

    # Young neurons mature
    for func in young_neurons:
        age = current_gen - young_neurons[func]['birth_gen']
        if age > maturation_period:
            if young_neurons[func]['contribution'] > survival_threshold:
                # Integrate into stable palette
                stable_palette.add(func)
            else:
                # Prune - didn't form useful connections
                remove_from_palette(func)
            del young_neurons[func]

    # Young neurons have higher learning rates
    learning_rate[func] = base_lr * young_neurons[func]['plasticity']

Expected improvements:
- Controlled exploration of new functions
- Young functions don't destabilize existing palette
- Survival-based pruning prevents bloat
- Continuous capability expansion through neurogenesis
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


class AdultNeurogenesisStrategy(PaletteEvolutionStrategy):
    """Hippocampal-inspired neurogenesis for palette evolution.

    New functions can be born into the palette, undergo a maturation period
    with high plasticity, and only survive if they contribute to fitness.
    Implements the biological principle of "use it or lose it" for new neurons.
    """

    name = "adult_neurogenesis"
    description = "Adult neurogenesis with birth, maturation, and survival-based integration"

    def __init__(
        self,
        # Neurogenesis parameters
        neurogenesis_rate: float = 0.08,           # Probability of birthing new function each gen
        maturation_period: int = 10,               # Generations before survival decision
        young_plasticity: float = 2.0,             # Learning rate multiplier for young neurons
        survival_threshold: float = 0.1,           # Minimum contribution to survive
        max_young_neurons: int = 3,                # Maximum concurrent young neurons
        # Contribution tracking
        contribution_decay: float = 0.9,           # How fast contribution decays
        contribution_boost_on_improvement: float = 0.3,  # Boost when fitness improves
        # Stable palette management
        stable_mutation_rate: float = 0.02,        # Mutation rate for stable palette
        max_stable_size: int = 8,                  # Maximum stable palette size
        min_stable_size: int = 2,                  # Minimum stable palette size
        # Selection parameters
        base_weight: float = 1.0,
        young_weight_boost: float = 0.3,           # Extra weight for young neurons
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Adult Neurogenesis strategy.

        Args:
            neurogenesis_rate: Probability of generating new neuron each generation
            maturation_period: Generations before deciding if neuron survives
            young_plasticity: Learning rate multiplier for young neurons
            survival_threshold: Minimum contribution required to survive
            max_young_neurons: Maximum number of young neurons at once
            contribution_decay: Decay rate for contribution tracking
            contribution_boost_on_improvement: Contribution boost when fitness improves
            stable_mutation_rate: Mutation rate for stable palette members
            max_stable_size: Maximum size of stable palette
            min_stable_size: Minimum size of stable palette
            base_weight: Base selection weight
            young_weight_boost: Extra selection weight for young neurons
            palette_size: Target palette size
        """
        # Neurogenesis
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.young_plasticity = young_plasticity
        self.survival_threshold = survival_threshold
        self.max_young_neurons = max_young_neurons

        # Contribution
        self.contribution_decay = contribution_decay
        self.contribution_boost_on_improvement = contribution_boost_on_improvement

        # Stable palette
        self.stable_mutation_rate = stable_mutation_rate
        self.max_stable_size = max_stable_size
        self.min_stable_size = min_stable_size

        # Selection
        self.base_weight = base_weight
        self.young_weight_boost = young_weight_boost

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neurogenesis tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initial stable palette
        stable_palette = set(initial)

        # Contribution tracking for all functions
        contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                contribution = contribution.at[i].set(0.5)  # Initial contribution

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 414141),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Neurogenesis state
            'stable_palette': stable_palette,         # Set of stable function indices
            'young_neurons': {},                      # {func_idx: {'birth_gen': int, 'contribution': float}}
            'contribution': contribution,             # Per-function contribution tracking
            # History
            'births': [],                             # List of (gen, func) tuples
            'survivals': [],                          # Functions that survived maturation
            'prunings': [],                           # Functions that were pruned
            'previous_mask': mask,
            'fitness_history': [],
            # Stats
            'total_births': 0,
            'total_survivals': 0,
            'total_prunings': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette (stable + young)."""
        return mask_to_indices(state['mask'])

    def _maybe_birth_neuron(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        key: jax.random.PRNGKey,
        generation: int,
    ) -> Tuple[Set[int], Dict[int, Dict], Optional[int]]:
        """Possibly birth a new neuron."""
        key1, key2 = jax.random.split(key)
        new_young = dict(young)
        born = None

        # Check if we should birth
        if (len(new_young) < self.max_young_neurons and
            jax.random.uniform(key1) < self.neurogenesis_rate):

            # Find available functions (not in stable or young)
            available = [i for i in range(NUM_ACTIVATIONS)
                        if i not in stable and i not in new_young]

            if available:
                # Sample new function
                idx = int(jax.random.randint(key2, (), 0, len(available)))
                new_func = available[idx]

                # Birth the neuron
                new_young[new_func] = {
                    'birth_gen': generation,
                    'contribution': 0.0,
                }
                born = new_func

        return stable, new_young, born

    def _mature_neurons(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        contribution: jnp.ndarray,
        generation: int,
    ) -> Tuple[Set[int], Dict[int, Dict], List[int], List[int]]:
        """Process neuron maturation - survive or prune."""
        new_stable = set(stable)
        new_young = {}
        survived = []
        pruned = []

        for func, info in young.items():
            age = generation - info['birth_gen']

            if age >= self.maturation_period:
                # Time to decide survival
                func_contribution = float(contribution[func])

                if func_contribution >= self.survival_threshold:
                    # Survived! Integrate into stable palette
                    if len(new_stable) < self.max_stable_size:
                        new_stable.add(func)
                        survived.append(func)
                    else:
                        # Stable palette full, still prune
                        pruned.append(func)
                else:
                    # Failed to integrate, prune
                    pruned.append(func)
            else:
                # Still young
                new_young[func] = info

        return new_stable, new_young, survived, pruned

    def _update_contributions(
        self,
        contribution: jnp.ndarray,
        mask: jnp.ndarray,
        young: Dict[int, Dict],
        improved: bool,
        generation: int,
    ) -> Tuple[jnp.ndarray, Dict[int, Dict]]:
        """Update contribution tracking."""
        new_contribution = contribution * self.contribution_decay
        new_young = {}

        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                current = float(new_contribution[i])
                if improved:
                    boost = self.contribution_boost_on_improvement
                    # Young neurons get extra plasticity
                    if i in young:
                        boost *= self.young_plasticity
                    new_contribution = new_contribution.at[i].set(current + boost)
                else:
                    # Small boost just for being active
                    new_contribution = new_contribution.at[i].set(current + 0.01)

        # Update young neuron contributions
        for func, info in young.items():
            new_info = dict(info)
            new_info['contribution'] = float(new_contribution[func])
            new_young[func] = new_info

        return jnp.clip(new_contribution, 0, 2.0), new_young

    def _maybe_mutate_stable(
        self,
        stable: Set[int],
        contribution: jnp.ndarray,
        stagnation: int,
        key: jax.random.PRNGKey,
    ) -> Set[int]:
        """Possibly mutate stable palette on stagnation."""
        if stagnation < 5:
            return stable

        key1, key2 = jax.random.split(key)
        new_stable = set(stable)

        if jax.random.uniform(key1) < self.stable_mutation_rate * (stagnation / 5):
            # Remove lowest contribution stable neuron
            if len(new_stable) > self.min_stable_size:
                stable_list = list(new_stable)
                contributions = [float(contribution[i]) for i in stable_list]
                min_idx = np.argmin(contributions)
                new_stable.remove(stable_list[min_idx])

        return new_stable

    def _create_mask(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
    ) -> jnp.ndarray:
        """Create mask from stable and young palettes."""
        mask = jnp.zeros(NUM_ACTIVATIONS)

        for i in stable:
            if 0 <= i < NUM_ACTIVATIONS:
                mask = mask.at[i].set(1.0)

        for i in young.keys():
            if 0 <= i < NUM_ACTIVATIONS:
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
        """Update with adult neurogenesis dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        stable = set(state['stable_palette'])
        young = dict(state['young_neurons'])

        # Step 1: Update contributions
        new_contribution, young = self._update_contributions(
            state['contribution'],
            state['mask'],
            young,
            improved,
            generation,
        )

        # Step 2: Process maturation (survive or prune)
        stable, young, survived, pruned = self._mature_neurons(
            stable, young, new_contribution, generation
        )

        # Step 3: Maybe birth new neuron
        stable, young, born = self._maybe_birth_neuron(stable, young, k1, generation)

        # Step 4: Maybe mutate stable palette on stagnation
        stable = self._maybe_mutate_stable(stable, new_contribution, new_stagnation, k2)

        # Create new mask
        new_mask = self._create_mask(stable, young)

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update history
        births = list(state['births'])
        if born is not None:
            births.append((generation, born))
            if len(births) > 50:
                births = births[-50:]

        survivals = list(state['survivals']) + survived
        if len(survivals) > 50:
            survivals = survivals[-50:]

        prunings = list(state['prunings']) + pruned
        if len(prunings) > 50:
            prunings = prunings[-50:]

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
            # Neurogenesis state
            'stable_palette': stable,
            'young_neurons': young,
            'contribution': new_contribution,
            # History
            'births': births,
            'survivals': survivals,
            'prunings': prunings,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Stats
            'total_births': state['total_births'] + (1 if born else 0),
            'total_survivals': state['total_survivals'] + len(survived),
            'total_prunings': state['total_prunings'] + len(pruned),
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Young neuron info
        young_info = [(func, info['contribution'], generation - info['birth_gen'])
                      for func, info in young.items()]

        # Contribution ranking
        top_contrib_idx = jnp.argsort(new_contribution)[-5:][::-1]
        top_contribution = [(int(i), float(new_contribution[i])) for i in top_contrib_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neurogenesis
            'n_stable': len(stable),
            'n_young': len(young),
            'stable_palette': list(stable),
            'young_neurons': young_info,
            'born_this_gen': born,
            'survived_this_gen': survived,
            'pruned_this_gen': pruned,
            # Cumulative stats
            'total_births': new_state['total_births'],
            'total_survivals': new_state['total_survivals'],
            'total_prunings': new_state['total_prunings'],
            'survival_rate': (new_state['total_survivals'] /
                            max(new_state['total_births'], 1)) * 100,
            # Contribution
            'top_contribution': top_contribution,
            'mean_contribution': float(jnp.mean(new_contribution)),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_is_stable': 4 in stable,
            'sin_is_young': 4 in young,
            'sin_contribution': float(new_contribution[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with neurogenesis status."""
        palette = self.get_active_palette(state)
        stable = state['stable_palette']
        young = state['young_neurons']
        contribution = state['contribution']

        # Top by contribution
        top_idx = jnp.argsort(contribution)[-5:][::-1]
        top_contribution = [(int(i), float(contribution[i])) for i in top_idx]

        # Young neuron details
        young_details = [(func, info['contribution'], state['generation'] - info['birth_gen'])
                        for func, info in young.items()]

        # Survival rate
        survival_rate = (state['total_survivals'] /
                        max(state['total_births'], 1)) * 100

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Neurogenesis
            'n_stable': len(stable),
            'n_young': len(young),
            'stable_palette': list(stable),
            'young_neurons': young_details,
            # Stats
            'total_births': state['total_births'],
            'total_survivals': state['total_survivals'],
            'total_prunings': state['total_prunings'],
            'survival_rate': survival_rate,
            # Contribution
            'top_contribution': top_contribution,
            # Sin-specific
            'sin_is_stable': 4 in stable,
            'sin_is_young': 4 in young,
            'sin_contribution': float(contribution[4]),
        }
