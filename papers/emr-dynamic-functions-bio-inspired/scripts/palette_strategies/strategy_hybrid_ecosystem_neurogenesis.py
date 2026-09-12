"""Hybrid Strategy: Ecosystem Dynamics + Adult Neurogenesis.

Combines two bio-inspired strategies:
- Ecosystem Dynamics (46): Carrying capacity, succession, keystone functions
- Adult Neurogenesis (41): Birth, maturation, survival-based integration

Expected Benefits:
- Carrying capacity limits prevent palette bloat (Ecosystem)
- Controlled introduction of new functions (Neurogenesis)
- Ecological succession phases guide function roles
- Young neurons compete for survival within ecosystem constraints

Hybrid Mechanism:
    # Ecosystem provides carrying capacity and succession
    available_slots = carrying_capacity - len(stable_palette)

    # Neurogenesis only when ecosystem has capacity
    if available_slots > 0 and should_birth():
        birth_young_neuron(...)

    # Young neurons compete using ecosystem rules
    survival_score = base_contribution * role_bias[succession_stage]
    if survival_score >= threshold:
        integrate_as_stable()
    else:
        prune()

    # Keystones are protected from pruning
    if is_keystone(func):
        prevent_pruning()
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


# Function roles for succession stages
FUNCTION_ROLES = {
    0: 'intermediate',   # tanh
    1: 'climax',         # sigmoid
    2: 'pioneer',        # relu
    3: 'pioneer',        # identity
    4: 'climax',         # sin
    5: 'intermediate',   # abs
    6: 'intermediate',   # elu
    7: 'climax',         # gauss
}


class EcosystemNeurogenesisStrategy(PaletteEvolutionStrategy):
    """Hybrid: Ecosystem Dynamics + Adult Neurogenesis.

    Combines carrying capacity and succession (Ecosystem) with
    birth/maturation/survival (Neurogenesis) for controlled palette evolution.
    """

    name = "ecosystem_neurogenesis"
    description = "Ecological succession with neurogenesis-controlled births"

    def __init__(
        self,
        # Ecosystem parameters
        carrying_capacity: int = 8,
        pioneer_duration: int = 15,
        intermediate_duration: int = 40,
        keystone_threshold: float = 0.5,
        disturbance_trigger: int = 12,
        # Neurogenesis parameters
        neurogenesis_rate: float = 0.06,
        maturation_period: int = 8,
        survival_threshold: float = 0.15,
        max_young_neurons: int = 2,
        young_plasticity: float = 1.8,
        # Shared
        contribution_decay: float = 0.92,
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        # Ecosystem
        self.carrying_capacity = carrying_capacity
        self.pioneer_duration = pioneer_duration
        self.intermediate_duration = intermediate_duration
        self.keystone_threshold = keystone_threshold
        self.disturbance_trigger = disturbance_trigger

        # Neurogenesis
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.survival_threshold = survival_threshold
        self.max_young_neurons = max_young_neurons
        self.young_plasticity = young_plasticity

        # Shared
        self.contribution_decay = contribution_decay
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with Ecosystem + Neurogenesis tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                contribution = contribution.at[i].set(0.4)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 464141),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Ecosystem state
            'community_age': 0,
            'succession_stage': 'pioneer',
            'keystones': set(),
            # Neurogenesis state
            'stable_palette': set(initial),
            'young_neurons': {},  # {func: {'birth_gen': int, 'contribution': float}}
            # Contribution
            'contribution': contribution,
            # History
            'births': [],
            'survivals': [],
            'prunings': [],
            'disturbances': [],
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

    def _get_succession_stage(self, community_age: int) -> str:
        """Determine succession stage."""
        if community_age < self.pioneer_duration:
            return 'pioneer'
        elif community_age < self.intermediate_duration:
            return 'intermediate'
        else:
            return 'climax'

    def _get_role_bias(self, stage: str) -> Dict[str, float]:
        """Get selection bias for each role based on succession."""
        if stage == 'pioneer':
            return {'pioneer': 0.8, 'intermediate': 0.5, 'climax': 0.3}
        elif stage == 'intermediate':
            return {'pioneer': 0.5, 'intermediate': 0.7, 'climax': 0.5}
        else:
            return {'pioneer': 0.3, 'intermediate': 0.5, 'climax': 0.8}

    def _identify_keystones(
        self,
        contribution: jnp.ndarray,
        stable: Set[int],
    ) -> Set[int]:
        """Identify keystone functions."""
        keystones = set()
        for func in stable:
            if float(contribution[func]) >= self.keystone_threshold:
                keystones.add(func)
        return keystones

    def _maybe_birth_neuron(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        stage: str,
        key: jax.random.PRNGKey,
        generation: int,
    ) -> Tuple[Dict[int, Dict], Optional[int]]:
        """Possibly birth a new neuron (respecting ecosystem capacity)."""
        key1, key2 = jax.random.split(key)
        new_young = dict(young)
        born = None

        # Check carrying capacity
        total_current = len(stable) + len(young)
        if total_current >= self.carrying_capacity:
            return new_young, None

        # Check neurogenesis limit
        if len(young) >= self.max_young_neurons:
            return new_young, None

        # Birth probability
        if jax.random.uniform(key1) < self.neurogenesis_rate:
            available = [i for i in range(NUM_ACTIVATIONS)
                        if i not in stable and i not in young]

            if available:
                # Bias by succession stage
                role_bias = self._get_role_bias(stage)
                weights = []
                for func in available:
                    role = FUNCTION_ROLES.get(func, 'intermediate')
                    weights.append(role_bias.get(role, 0.5))

                total_w = sum(weights)
                probs = [w / total_w for w in weights]

                # Sample
                cum = 0
                sample = float(jax.random.uniform(key2))
                for i, p in enumerate(probs):
                    cum += p
                    if sample < cum:
                        new_func = available[i]
                        new_young[new_func] = {
                            'birth_gen': generation,
                            'contribution': 0.0,
                        }
                        born = new_func
                        break

        return new_young, born

    def _mature_neurons(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        contribution: jnp.ndarray,
        keystones: Set[int],
        stage: str,
        generation: int,
    ) -> Tuple[Set[int], Dict[int, Dict], List[int], List[int]]:
        """Process neuron maturation with ecosystem rules."""
        new_stable = set(stable)
        new_young = {}
        survived = []
        pruned = []

        role_bias = self._get_role_bias(stage)

        for func, info in young.items():
            age = generation - info['birth_gen']

            if age >= self.maturation_period:
                # Evaluate survival with ecosystem bias
                base_contrib = float(contribution[func])
                role = FUNCTION_ROLES.get(func, 'intermediate')
                role_factor = role_bias.get(role, 0.5)
                survival_score = base_contrib * role_factor

                if survival_score >= self.survival_threshold:
                    # Survived - integrate into stable
                    if len(new_stable) < self.carrying_capacity:
                        new_stable.add(func)
                        survived.append(func)
                    else:
                        # Over capacity - compete with weakest stable
                        weakest = None
                        weakest_score = float('inf')
                        for s_func in new_stable - keystones:
                            s_score = float(contribution[s_func])
                            if s_score < weakest_score:
                                weakest = s_func
                                weakest_score = s_score

                        if weakest is not None and survival_score > weakest_score:
                            new_stable.remove(weakest)
                            new_stable.add(func)
                            survived.append(func)
                            pruned.append(weakest)
                        else:
                            pruned.append(func)
                else:
                    pruned.append(func)
            else:
                new_young[func] = info

        return new_stable, new_young, survived, pruned

    def _trigger_disturbance(
        self,
        stable: Set[int],
        keystones: Set[int],
        stage: str,
        key: jax.random.PRNGKey,
    ) -> Set[int]:
        """Trigger disturbance - remove non-keystone climax functions."""
        new_stable = set(stable)

        # Disturbance removes climax species preferentially
        removable = []
        for func in stable - keystones:
            role = FUNCTION_ROLES.get(func, 'intermediate')
            if role == 'climax':
                removable.append(func)

        if removable and len(new_stable) > self.min_active:
            # Remove one random climax
            idx = int(jax.random.randint(key, (), 0, len(removable)))
            new_stable.remove(removable[idx])

        return new_stable

    def _create_mask(self, stable: Set[int], young: Dict[int, Dict]) -> jnp.ndarray:
        """Create mask from stable and young."""
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
        """Update with Ecosystem + Neurogenesis hybrid dynamics."""
        key, k1, k2, k3 = jax.random.split(state['rng_key'], 4)

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

        # Update contribution
        new_contribution = state['contribution'] * self.contribution_decay
        for func in mask_to_indices(state['mask']):
            boost = 0.25 if improved else 0.03
            if func in young:
                boost *= self.young_plasticity
            new_contribution = new_contribution.at[func].add(boost)
        new_contribution = jnp.clip(new_contribution, 0, 2.0)

        # Update young neuron contributions
        for func, info in young.items():
            young[func]['contribution'] = float(new_contribution[func])

        # Ecosystem: Determine succession stage
        community_age = state['community_age'] + 1
        stage = self._get_succession_stage(community_age)

        # Ecosystem: Identify keystones
        keystones = self._identify_keystones(new_contribution, stable)

        # Ecosystem: Check for disturbance
        disturbances = list(state['disturbances'])
        disturbance_triggered = False
        if new_stagnation >= self.disturbance_trigger:
            stable = self._trigger_disturbance(stable, keystones, stage, k1)
            community_age = 0  # Reset succession
            disturbances.append(generation)
            disturbance_triggered = True

        # Neurogenesis: Mature neurons
        stable, young, survived, pruned = self._mature_neurons(
            stable, young, new_contribution, keystones, stage, generation
        )

        # Neurogenesis: Maybe birth new neuron
        young, born = self._maybe_birth_neuron(stable, young, stage, k2, generation)

        # Create new mask
        new_mask = self._create_mask(stable, young)
        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update history
        births = list(state['births'])
        if born is not None:
            births.append((generation, born))
            if len(births) > 50:
                births = births[-50:]

        survivals_list = list(state['survivals']) + survived
        prunings_list = list(state['prunings']) + pruned
        if len(survivals_list) > 50:
            survivals_list = survivals_list[-50:]
        if len(prunings_list) > 50:
            prunings_list = prunings_list[-50:]

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
            # Ecosystem
            'community_age': community_age,
            'succession_stage': stage,
            'keystones': keystones,
            # Neurogenesis
            'stable_palette': stable,
            'young_neurons': young,
            # Contribution
            'contribution': new_contribution,
            # History
            'births': births,
            'survivals': survivals_list,
            'prunings': prunings_list,
            'disturbances': disturbances[-20:],
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Stats
            'total_births': state['total_births'] + (1 if born else 0),
            'total_survivals': state['total_survivals'] + len(survived),
            'total_prunings': state['total_prunings'] + len(pruned),
        }

        active_palette = mask_to_indices(new_mask)
        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Ecosystem
            'succession_stage': stage,
            'community_age': community_age,
            'n_keystones': len(keystones),
            'keystones': list(keystones),
            'disturbance_triggered': disturbance_triggered,
            # Neurogenesis
            'n_stable': len(stable),
            'n_young': len(young),
            'born_this_gen': born,
            'survived_this_gen': survived,
            'pruned_this_gen': pruned,
            # Stats
            'total_births': new_state['total_births'],
            'survival_rate': (new_state['total_survivals'] /
                            max(new_state['total_births'], 1)) * 100,
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_is_stable': 4 in stable,
            'sin_is_keystone': 4 in keystones,
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
            'succession_stage': state['succession_stage'],
            'n_stable': len(state['stable_palette']),
            'n_young': len(state['young_neurons']),
            'n_keystones': len(state['keystones']),
            'total_births': state['total_births'],
            'survival_rate': (state['total_survivals'] /
                            max(state['total_births'], 1)) * 100,
        }
