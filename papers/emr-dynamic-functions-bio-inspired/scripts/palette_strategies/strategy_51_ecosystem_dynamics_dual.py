"""Strategy 51D: Ecosystem Dynamics Dual (Ecological Succession for Both Domains).

Extends EcosystemDynamicsStrategy to jointly evolve BOTH activation AND aggregation
function palettes using ecological succession and competition dynamics.

Key dual mechanisms:
1. Dual carrying capacity - separate resource limits for act and agg
2. Dual succession stages - independent pioneer/intermediate/climax transitions
3. Cross-domain keystones - successful act-agg pairs get keystone protection
4. Coordinated disturbance - stagnation can trigger reset in both domains
5. Facilitation across domains - keystone in one domain supports the other

Expected: Self-organizing dual palette with stable attractors and natural complexity limits
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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


# Function roles in ecosystem - activation domain
ACT_FUNCTION_ROLES = {
    0: 'intermediate',   # tanh - versatile
    1: 'climax',         # sigmoid - stable
    2: 'pioneer',        # relu - simple
    3: 'pioneer',        # identity - simplest
    4: 'climax',         # sin - specialized for periodicity
    5: 'intermediate',   # gauss
    6: 'intermediate',   # lelu
    7: 'climax',         # softplus
}

# Function roles - aggregation domain
AGG_FUNCTION_ROLES = {
    0: 'pioneer',        # sum - simplest
    1: 'pioneer',        # mean - simple
    2: 'intermediate',   # max - selective
    3: 'intermediate',   # min - selective
    4: 'climax',         # product - multiplicative
    5: 'climax',         # maxabs - specialized
}


class EcosystemDynamicsDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution using ecological succession and competition.

    Both activation and aggregation palettes evolve as ecosystems with carrying
    capacity, keystone protection, succession stages, and disturbance dynamics.
    """

    name = "ecosystem_dynamics_dual"
    description = "Dual: Ecological succession with keystones in both domains"

    def __init__(
        self,
        # Carrying capacity
        act_carrying_capacity: int = 8,
        agg_carrying_capacity: int = 4,
        resource_regeneration: float = 0.5,
        base_resources: float = 10.0,
        function_resource_cost: float = 1.0,
        # Competition
        competition_strength: float = 0.3,
        # Keystone dynamics
        keystone_enabled: bool = True,
        keystone_threshold: float = 0.5,
        keystone_protection: float = 0.8,
        keystone_facilitation: float = 0.3,
        # Disturbance
        disturbance_trigger: int = 10,
        disturbance_intensity: float = 0.5,
        disturbance_recovery: int = 5,
        # Succession
        pioneer_duration: int = 15,
        intermediate_duration: int = 40,
        pioneer_bias: float = 0.7,
        climax_bias: float = 0.7,
        # Cross-domain
        cross_learning_rate: float = 0.08,
        cross_keystone_bonus: float = 0.2,
        # General
        base_mutation_rate: float = 0.1,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Ecosystem Dynamics Dual strategy."""
        self.act_carrying_capacity = act_carrying_capacity
        self.agg_carrying_capacity = agg_carrying_capacity
        self.resource_regeneration = resource_regeneration
        self.base_resources = base_resources
        self.function_resource_cost = function_resource_cost

        self.competition_strength = competition_strength

        self.keystone_enabled = keystone_enabled
        self.keystone_threshold = keystone_threshold
        self.keystone_protection = keystone_protection
        self.keystone_facilitation = keystone_facilitation

        self.disturbance_trigger = disturbance_trigger
        self.disturbance_intensity = disturbance_intensity
        self.disturbance_recovery = disturbance_recovery

        self.pioneer_duration = pioneer_duration
        self.intermediate_duration = intermediate_duration
        self.pioneer_bias = pioneer_bias
        self.climax_bias = climax_bias

        self.cross_learning_rate = cross_learning_rate
        self.cross_keystone_bonus = cross_keystone_bonus

        self.base_mutation_rate = base_mutation_rate
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual ecosystem tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize contributions
        act_contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_contribution = act_contribution.at[i].set(0.3)

        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.3)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_contribution': act_contribution,
            'act_keystones': set(),
            'act_community_age': 0,
            'act_succession_stage': 'pioneer',
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_contribution': agg_contribution,
            'agg_keystones': set(),
            'agg_community_age': 0,
            'agg_succession_stage': 'pioneer',
            # Cross-domain
            'cross_affinity': cross_affinity,
            'cross_keystones': set(),  # Act-agg pairs that are jointly keystone
            # Resources (shared)
            'resources': self.base_resources,
            'in_recovery': False,
            'recovery_gens_left': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 515151),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'total_disturbances': 0,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _get_succession_stage(self, community_age: int) -> str:
        """Determine current succession stage."""
        if community_age < self.pioneer_duration:
            return 'pioneer'
        elif community_age < self.intermediate_duration:
            return 'intermediate'
        else:
            return 'climax'

    def _get_role_bias(self, stage: str) -> Dict[str, float]:
        """Get selection bias for each role based on succession stage."""
        if stage == 'pioneer':
            return {'pioneer': self.pioneer_bias, 'intermediate': 0.5, 'climax': 1 - self.pioneer_bias}
        elif stage == 'intermediate':
            return {'pioneer': 0.5, 'intermediate': 0.7, 'climax': 0.5}
        else:
            return {'pioneer': 1 - self.climax_bias, 'intermediate': 0.5, 'climax': self.climax_bias}

    def _update_contribution(
        self,
        contribution: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update function contribution estimates."""
        new_contribution = contribution * 0.95  # Decay
        n_funcs = len(contribution)

        for i in range(n_funcs):
            if mask[i] > 0.5:
                if improved:
                    boost = 0.1 + fitness_delta * 0.5
                    new_contribution = new_contribution.at[i].add(boost)
                else:
                    new_contribution = new_contribution.at[i].add(0.02)

        return jnp.clip(new_contribution, 0, 2.0)

    def _identify_keystones(
        self,
        contribution: jnp.ndarray,
        active_palette: List[int],
    ) -> Set[int]:
        """Identify keystone functions based on contribution."""
        if not self.keystone_enabled:
            return set()

        keystones = set()
        for func in active_palette:
            if float(contribution[func]) >= self.keystone_threshold:
                keystones.add(func)

        return keystones

    def _trigger_disturbance(
        self,
        mask: jnp.ndarray,
        keystones: Set[int],
        key: jax.random.PRNGKey,
        min_active: int,
        role_map: Dict[int, str],
    ) -> jnp.ndarray:
        """Trigger disturbance - remove some functions."""
        new_mask = mask.copy()
        current_palette = mask_to_indices(mask)

        if len(current_palette) <= min_active:
            return new_mask

        n_remove = int(len(current_palette) * self.disturbance_intensity)
        n_remove = min(n_remove, len(current_palette) - min_active)

        if n_remove <= 0:
            return new_mask

        removal_weights = []
        for func in current_palette:
            role = role_map.get(func, 'intermediate')
            weight = 1.0

            if role == 'climax':
                weight *= 2.0
            elif role == 'pioneer':
                weight *= 0.5

            if func in keystones:
                weight *= 0.1

            removal_weights.append(weight)

        total = sum(removal_weights)
        if total == 0:
            return new_mask

        removal_probs = [w / total for w in removal_weights]

        removed = []
        for _ in range(n_remove):
            if not removal_probs:
                break

            key, subkey = jax.random.split(key)
            sample = float(jax.random.uniform(subkey))
            cum_prob = 0

            for i, prob in enumerate(removal_probs):
                cum_prob += prob
                if sample < cum_prob:
                    func = current_palette[i]
                    if func not in removed and new_mask[func] > 0.5:
                        new_mask = new_mask.at[func].set(0.0)
                        removed.append(func)
                    break

        return new_mask

    def _succession_mutate(
        self,
        mask: jnp.ndarray,
        contribution: jnp.ndarray,
        keystones: Set[int],
        stage: str,
        key: jax.random.PRNGKey,
        carrying_capacity: int,
        min_active: int,
        role_map: Dict[int, str],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Mutate palette based on succession dynamics."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()

        current_palette = mask_to_indices(mask)
        role_bias = self._get_role_bias(stage)

        # Check carrying capacity
        if len(current_palette) > carrying_capacity:
            n_remove = len(current_palette) - carrying_capacity
            sorted_funcs = sorted(current_palette, key=lambda f: float(contribution[f]))
            for func in sorted_funcs[:n_remove]:
                if func not in keystones:
                    new_mask = new_mask.at[func].set(0.0)

        current_palette = mask_to_indices(new_mask)

        # Normal mutation
        if jax.random.uniform(key1) < self.base_mutation_rate:
            # Remove a function
            if len(current_palette) > min_active:
                removal_weights = []
                for func in current_palette:
                    weight = 1.0 / (float(contribution[func]) + 0.1)
                    if func in keystones:
                        weight *= 0.1
                    removal_weights.append(weight)

                total = sum(removal_weights)
                if total > 0:
                    removal_probs = [w / total for w in removal_weights]
                    sample = float(jax.random.uniform(key2))
                    cum_prob = 0
                    for i, prob in enumerate(removal_probs):
                        cum_prob += prob
                        if sample < cum_prob:
                            removed = current_palette[i]
                            new_mask = new_mask.at[removed].set(0.0)
                            break

            # Add a function based on succession
            available = [i for i in range(n_funcs) if new_mask[i] < 0.5]
            if available and len(mask_to_indices(new_mask)) < carrying_capacity:
                add_weights = []
                for func in available:
                    role = role_map.get(func, 'intermediate')
                    weight = role_bias.get(role, 0.5)
                    add_weights.append(weight)

                total = sum(add_weights)
                if total > 0:
                    add_probs = [w / total for w in add_weights]
                    sample = float(jax.random.uniform(key3))
                    cum_prob = 0
                    for i, prob in enumerate(add_probs):
                        cum_prob += prob
                        if sample < cum_prob:
                            added = available[i]
                            new_mask = new_mask.at[added].set(1.0)
                            break

        return new_mask

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        delta = self.cross_learning_rate * fitness_delta * co_active
        new_cross = cross_affinity + delta
        return jnp.clip(new_cross, 0.0, 2.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual ecosystem dynamics."""
        key, k_act, k_agg, k_dist = jax.random.split(state['rng_key'], 4)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness if improved else 0.0

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update contributions in both domains
        new_act_contrib = self._update_contribution(
            state['act_contribution'], state['act_mask'], improved, fitness_delta
        )
        new_agg_contrib = self._update_contribution(
            state['agg_contribution'], state['agg_mask'], improved, fitness_delta
        )

        # Step 2: Identify keystones
        act_palette = mask_to_indices(state['act_mask'])
        agg_palette = mask_to_indices(state['agg_mask'])
        act_keystones = self._identify_keystones(new_act_contrib, act_palette)
        agg_keystones = self._identify_keystones(new_agg_contrib, agg_palette)

        # Step 3: Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Step 4: Check for disturbance
        in_recovery = state['in_recovery']
        recovery_left = state['recovery_gens_left']
        act_community_age = state['act_community_age']
        agg_community_age = state['agg_community_age']
        total_disturbances = state['total_disturbances']

        if in_recovery:
            recovery_left -= 1
            if recovery_left <= 0:
                in_recovery = False

        disturbance_triggered = False
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']

        if new_stagnation >= self.disturbance_trigger and not in_recovery:
            # Trigger disturbance in both domains
            k_dist_act, k_dist_agg = jax.random.split(k_dist)
            new_act_mask = self._trigger_disturbance(
                state['act_mask'], act_keystones, k_dist_act,
                self.min_active_act, ACT_FUNCTION_ROLES
            )
            new_agg_mask = self._trigger_disturbance(
                state['agg_mask'], agg_keystones, k_dist_agg,
                self.min_active_agg, AGG_FUNCTION_ROLES
            )
            disturbance_triggered = True
            act_community_age = 0
            agg_community_age = 0
            in_recovery = True
            recovery_left = self.disturbance_recovery
            total_disturbances += 1
        else:
            act_community_age += 1
            agg_community_age += 1

        # Step 5: Determine succession stages
        act_stage = self._get_succession_stage(act_community_age)
        agg_stage = self._get_succession_stage(agg_community_age)

        # Step 6: Succession-based mutation in both domains
        new_act_mask = self._succession_mutate(
            new_act_mask, new_act_contrib, act_keystones, act_stage, k_act,
            self.act_carrying_capacity, self.min_active_act, ACT_FUNCTION_ROLES, NUM_ACTIVATIONS
        )
        new_agg_mask = self._succession_mutate(
            new_agg_mask, new_agg_contrib, agg_keystones, agg_stage, k_agg,
            self.agg_carrying_capacity, self.min_active_agg, AGG_FUNCTION_ROLES, NUM_AGGREGATIONS
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_contribution': new_act_contrib,
            'act_keystones': act_keystones,
            'act_community_age': act_community_age,
            'act_succession_stage': act_stage,
            'agg_mask': new_agg_mask,
            'agg_contribution': new_agg_contrib,
            'agg_keystones': agg_keystones,
            'agg_community_age': agg_community_age,
            'agg_succession_stage': agg_stage,
            'cross_affinity': new_cross,
            'cross_keystones': state['cross_keystones'],
            'resources': state['resources'],
            'in_recovery': in_recovery,
            'recovery_gens_left': recovery_left,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'total_disturbances': total_disturbances,
            'fitness_history': fitness_history,
        }

        final_act_palette = mask_to_indices(new_act_mask)
        final_agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': final_act_palette,
            'current_agg_palette': final_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Ecosystem - activation
            'act_succession_stage': act_stage,
            'act_community_age': act_community_age,
            'act_n_keystones': len(act_keystones),
            # Ecosystem - aggregation
            'agg_succession_stage': agg_stage,
            'agg_community_age': agg_community_age,
            'agg_n_keystones': len(agg_keystones),
            # Disturbance
            'disturbance_triggered': disturbance_triggered,
            'in_recovery': in_recovery,
            'total_disturbances': total_disturbances,
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'cross_max_affinity': float(jnp.max(new_cross)),
            # Sin status
            'has_sin': 4 in final_act_palette,
            'sin_is_keystone': 4 in act_keystones,
            'sin_contribution': float(new_act_contrib[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual ecosystem status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Succession
            'act_succession_stage': state['act_succession_stage'],
            'agg_succession_stage': state['agg_succession_stage'],
            # Keystones
            'act_keystones': list(state['act_keystones']),
            'agg_keystones': list(state['agg_keystones']),
            # Disturbance
            'total_disturbances': state['total_disturbances'],
            'in_recovery': state['in_recovery'],
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
