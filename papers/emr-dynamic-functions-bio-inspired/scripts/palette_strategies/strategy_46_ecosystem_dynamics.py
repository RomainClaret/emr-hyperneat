"""Strategy 46: Ecosystem Dynamics (Ecological Succession for Palettes).

Implements ecological succession and competition dynamics for palette evolution.
Functions compete for resources, carrying capacity limits growth, keystone
functions support others, and disturbances trigger successional changes.

Biological Basis:
- Ecological succession progresses from pioneer to climax communities
- Carrying capacity limits population size
- Keystone species have disproportionate ecosystem impact
- Disturbances (fire, flood) reset succession
- Competition and facilitation shape community composition
- Niche partitioning allows species coexistence

Key Insight:
- Palettes can be viewed as ecosystems of functions
- Some functions are pioneers (good for exploration)
- Some are climax species (stable, optimized)
- Keystone functions support others
- Stagnation acts as a disturbance, triggering succession
- Competition prevents bloat, facilitation enables complexity

Ecosystem Mechanism:
    # Resource-based carrying capacity
    available_resources = carrying_capacity - len(active_palette)
    if available_resources < 0:
        # Competition: weakest functions die
        remove_weakest(abs(available_resources))

    # Keystone detection
    for func in active_palette:
        if contribution[func] > keystone_threshold:
            keystone_functions.add(func)
            # Keystones protect related functions
            for dependent in get_facilitated(func):
                protection[dependent] += keystone_protection

    # Disturbance on stagnation
    if stagnation > disturbance_trigger:
        # Reset to pioneer stage
        remove_climax_functions()
        allow_pioneer_colonization()

    # Succession stages
    if community_age < pioneer_duration:
        favor_pioneers()  # Simple, fast-growing
    elif community_age < intermediate_duration:
        favor_intermediate()  # Mixed community
    else:
        favor_climax()  # Stable, specialized

Expected improvements:
- Self-organizing palette with stable attractors
- Natural complexity limits through carrying capacity
- Keystone function protection
- Recovery from stagnation through succession
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


# Function roles in ecosystem
# Pioneer: fast-acting, simple, good for exploration
# Intermediate: balanced, versatile
# Climax: stable, specialized, optimal for exploitation
FUNCTION_ROLES = {
    0: 'intermediate',   # tanh - versatile, bounded
    1: 'climax',         # sigmoid - stable, specialized
    2: 'pioneer',        # relu - simple, fast
    3: 'pioneer',        # identity - simplest
    4: 'climax',         # sin - specialized, optimal for periodicity
    5: 'intermediate',   # abs - moderately specialized
    6: 'intermediate',   # elu - balanced
    7: 'climax',         # gauss - specialized, local
}


class EcosystemDynamicsStrategy(PaletteEvolutionStrategy):
    """Ecological succession and competition for palette evolution.

    Implements resource-based carrying capacity, keystone function protection,
    successional dynamics, and disturbance-triggered resets.
    """

    name = "ecosystem_dynamics"
    description = "Ecological succession with keystone functions and carrying capacity"

    def __init__(
        self,
        # Carrying capacity
        carrying_capacity: int = 8,               # Maximum palette size
        resource_regeneration: float = 0.5,       # Resources recovered per gen
        base_resources: float = 10.0,             # Starting resource pool
        function_resource_cost: float = 1.0,      # Cost per active function
        # Competition
        competition_strength: float = 0.3,        # Inter-function competition
        competition_asymmetry: float = 0.1,       # Stronger functions win more
        # Keystone dynamics
        keystone_enabled: bool = True,
        keystone_threshold: float = 0.5,          # Contribution to become keystone
        keystone_protection: float = 0.8,         # Protection strength
        keystone_facilitation: float = 0.3,       # Support to related functions
        # Disturbance (stagnation-triggered)
        disturbance_trigger: int = 10,            # Stagnation gens for disturbance
        disturbance_intensity: float = 0.5,       # Fraction of functions affected
        disturbance_recovery: int = 5,            # Gens to recover from disturbance
        # Succession
        pioneer_duration: int = 15,               # Gens of pioneer stage
        intermediate_duration: int = 40,          # Gens of intermediate stage
        pioneer_bias: float = 0.7,                # Selection bias for pioneers early
        climax_bias: float = 0.7,                 # Selection bias for climax late
        # General
        base_mutation_rate: float = 0.1,
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Ecosystem Dynamics strategy.

        Args:
            carrying_capacity: Maximum functions in palette
            resource_regeneration: Resource recovery per generation
            base_resources: Starting resource amount
            function_resource_cost: Resource cost per function
            competition_strength: How much functions compete
            competition_asymmetry: Advantage for stronger functions
            keystone_enabled: Enable keystone function dynamics
            keystone_threshold: Contribution to become keystone
            keystone_protection: How much keystones protect dependents
            keystone_facilitation: Support keystones provide
            disturbance_trigger: Stagnation generations to trigger disturbance
            disturbance_intensity: Fraction of palette affected
            disturbance_recovery: Generations to recover
            pioneer_duration: Length of pioneer succession stage
            intermediate_duration: Length of intermediate stage
            pioneer_bias: Selection bias toward pioneers early
            climax_bias: Selection bias toward climax late
        """
        # Resources
        self.carrying_capacity = carrying_capacity
        self.resource_regeneration = resource_regeneration
        self.base_resources = base_resources
        self.function_resource_cost = function_resource_cost

        # Competition
        self.competition_strength = competition_strength
        self.competition_asymmetry = competition_asymmetry

        # Keystone
        self.keystone_enabled = keystone_enabled
        self.keystone_threshold = keystone_threshold
        self.keystone_protection = keystone_protection
        self.keystone_facilitation = keystone_facilitation

        # Disturbance
        self.disturbance_trigger = disturbance_trigger
        self.disturbance_intensity = disturbance_intensity
        self.disturbance_recovery = disturbance_recovery

        # Succession
        self.pioneer_duration = pioneer_duration
        self.intermediate_duration = intermediate_duration
        self.pioneer_bias = pioneer_bias
        self.climax_bias = climax_bias

        # General
        self.base_mutation_rate = base_mutation_rate
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with ecosystem tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Per-function fitness contribution
        contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                contribution = contribution.at[i].set(0.3)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 464646),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Ecosystem state
            'resources': self.base_resources,
            'community_age': 0,                    # Gens since last disturbance
            'succession_stage': 'pioneer',         # pioneer, intermediate, climax
            'in_recovery': False,
            'recovery_gens_left': 0,
            # Function tracking
            'contribution': contribution,          # Estimated fitness contribution
            'keystone_functions': set(),           # Functions with keystone status
            'facilitated_by': {},                  # {func: keystone_supporting_it}
            # History
            'disturbance_events': [],              # (gen, intensity, cause)
            'keystone_history': [],                # (gen, keystones)
            'previous_mask': mask,
            'fitness_history': [],
            # Stats
            'total_disturbances': 0,
            'max_keystone_count': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette."""
        return mask_to_indices(state['mask'])

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
            return {
                'pioneer': self.pioneer_bias,
                'intermediate': 0.5,
                'climax': 1 - self.pioneer_bias,
            }
        elif stage == 'intermediate':
            return {
                'pioneer': 0.5,
                'intermediate': 0.7,
                'climax': 0.5,
            }
        else:  # climax
            return {
                'pioneer': 1 - self.climax_bias,
                'intermediate': 0.5,
                'climax': self.climax_bias,
            }

    def _update_resources(
        self,
        resources: float,
        n_active: int,
    ) -> float:
        """Update resource pool."""
        # Resource consumption
        consumption = n_active * self.function_resource_cost

        # Resource regeneration
        regeneration = self.resource_regeneration

        new_resources = resources - consumption + regeneration

        # Cap at carrying capacity equivalent
        max_resources = self.carrying_capacity * self.function_resource_cost
        return max(0, min(new_resources, max_resources))

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

    def _compute_facilitation(
        self,
        keystones: Set[int],
    ) -> Dict[int, int]:
        """Compute which functions are facilitated by keystones."""
        facilitated = {}

        for keystone in keystones:
            # Simple facilitation: keystones support nearby functions
            for i in range(NUM_ACTIVATIONS):
                if i != keystone and abs(i - keystone) <= 2:
                    facilitated[i] = keystone

        return facilitated

    def _compute_competition_outcome(
        self,
        func1: int,
        func2: int,
        contribution: jnp.ndarray,
    ) -> int:
        """Determine winner of competition between two functions."""
        c1 = float(contribution[func1])
        c2 = float(contribution[func2])

        # Higher contribution wins (with some asymmetry)
        if c1 > c2 * (1 + self.competition_asymmetry):
            return func1
        elif c2 > c1 * (1 + self.competition_asymmetry):
            return func2
        else:
            # Tie - random
            return func1 if np.random.random() < 0.5 else func2

    def _trigger_disturbance(
        self,
        mask: jnp.ndarray,
        keystones: Set[int],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Trigger disturbance - remove some functions."""
        new_mask = mask.copy()
        current_palette = mask_to_indices(mask)

        if len(current_palette) <= self.min_active:
            return new_mask

        # Calculate how many to remove
        n_remove = int(len(current_palette) * self.disturbance_intensity)
        n_remove = min(n_remove, len(current_palette) - self.min_active)

        if n_remove <= 0:
            return new_mask

        # Preferentially remove climax species, protect keystones
        removal_weights = []
        for func in current_palette:
            role = FUNCTION_ROLES.get(func, 'intermediate')
            weight = 1.0

            # Climax species more affected by disturbance
            if role == 'climax':
                weight *= 2.0
            elif role == 'pioneer':
                weight *= 0.5

            # Keystones are protected
            if func in keystones:
                weight *= 0.1

            removal_weights.append(weight)

        # Normalize
        total = sum(removal_weights)
        removal_probs = [w / total for w in removal_weights]

        # Remove functions
        removed = []
        for _ in range(n_remove):
            if not removal_probs:
                break

            # Sample
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
        facilitated: Dict[int, int],
        stage: str,
        resources: float,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Mutate palette based on succession dynamics."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()

        current_palette = mask_to_indices(mask)
        role_bias = self._get_role_bias(stage)

        # Check carrying capacity
        if len(current_palette) > self.carrying_capacity:
            # Over capacity - remove weakest
            n_remove = len(current_palette) - self.carrying_capacity

            # Sort by contribution (lowest first)
            sorted_funcs = sorted(current_palette, key=lambda f: float(contribution[f]))

            for func in sorted_funcs[:n_remove]:
                # Don't remove keystones
                if func not in keystones:
                    new_mask = new_mask.at[func].set(0.0)

        current_palette = mask_to_indices(new_mask)

        # Normal mutation
        if jax.random.uniform(key1) < self.base_mutation_rate:
            # Remove a function (competition)
            if len(current_palette) > self.min_active:
                removal_weights = []
                for func in current_palette:
                    weight = 1.0 / (float(contribution[func]) + 0.1)

                    # Keystones protected
                    if func in keystones:
                        weight *= 0.1

                    # Facilitated functions slightly protected
                    if func in facilitated:
                        weight *= 0.5

                    removal_weights.append(weight)

                # Normalize
                total = sum(removal_weights)
                removal_probs = [w / total for w in removal_weights]

                # Sample
                cum_prob = 0
                sample = float(jax.random.uniform(key2))
                for i, prob in enumerate(removal_probs):
                    cum_prob += prob
                    if sample < cum_prob:
                        removed = current_palette[i]
                        new_mask = new_mask.at[removed].set(0.0)
                        break

            # Add a function based on succession stage
            available = [i for i in range(NUM_ACTIVATIONS) if new_mask[i] < 0.5]
            if available and resources >= self.function_resource_cost:
                # Weight by role bias
                add_weights = []
                for func in available:
                    role = FUNCTION_ROLES.get(func, 'intermediate')
                    weight = role_bias.get(role, 0.5)

                    # Facilitated functions get bonus
                    if func in facilitated:
                        weight *= (1 + self.keystone_facilitation)

                    add_weights.append(weight)

                # Normalize
                total = sum(add_weights)
                add_probs = [w / total for w in add_weights]

                # Sample
                cum_prob = 0
                sample = float(jax.random.uniform(key3))
                for i, prob in enumerate(add_probs):
                    cum_prob += prob
                    if sample < cum_prob:
                        added = available[i]
                        new_mask = new_mask.at[added].set(1.0)
                        break

        return new_mask

    def _update_contribution(
        self,
        contribution: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update function contribution estimates."""
        new_contribution = contribution * 0.95  # Decay

        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                if improved:
                    boost = 0.1 + fitness_delta * 0.5
                    new_contribution = new_contribution.at[i].add(boost)
                else:
                    # Small boost for being active
                    new_contribution = new_contribution.at[i].add(0.02)

        return jnp.clip(new_contribution, 0, 2.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with ecosystem dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness if improved else 0.0

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        current_palette = mask_to_indices(state['mask'])

        # Step 1: Update contributions
        new_contribution = self._update_contribution(
            state['contribution'],
            state['mask'],
            improved,
            fitness_delta,
        )

        # Step 2: Identify keystones
        keystones = self._identify_keystones(new_contribution, current_palette)
        facilitated = self._compute_facilitation(keystones)

        # Step 3: Check for disturbance
        disturbance_events = list(state['disturbance_events'])
        community_age = state['community_age']
        in_recovery = state['in_recovery']
        recovery_left = state['recovery_gens_left']

        if in_recovery:
            recovery_left -= 1
            if recovery_left <= 0:
                in_recovery = False

        disturbance_triggered = False
        if new_stagnation >= self.disturbance_trigger and not in_recovery:
            # Trigger disturbance
            new_mask = self._trigger_disturbance(state['mask'], keystones, k1)
            disturbance_triggered = True
            disturbance_events.append((generation, self.disturbance_intensity, 'stagnation'))
            community_age = 0  # Reset succession
            in_recovery = True
            recovery_left = self.disturbance_recovery
        else:
            new_mask = state['mask']
            community_age += 1

        # Step 4: Determine succession stage
        stage = self._get_succession_stage(community_age)

        # Step 5: Update resources
        new_resources = self._update_resources(state['resources'], len(current_palette))

        # Step 6: Succession-based mutation
        new_mask = self._succession_mutate(
            new_mask,
            new_contribution,
            keystones,
            facilitated,
            stage,
            new_resources,
            k2,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update history
        if len(disturbance_events) > 50:
            disturbance_events = disturbance_events[-50:]

        keystone_history = list(state['keystone_history'])
        if keystones != state['keystone_functions']:
            keystone_history.append((generation, list(keystones)))
            if len(keystone_history) > 50:
                keystone_history = keystone_history[-50:]

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        max_keystones = max(state['max_keystone_count'], len(keystones))

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Ecosystem
            'resources': new_resources,
            'community_age': community_age,
            'succession_stage': stage,
            'in_recovery': in_recovery,
            'recovery_gens_left': recovery_left,
            # Function tracking
            'contribution': new_contribution,
            'keystone_functions': keystones,
            'facilitated_by': facilitated,
            # History
            'disturbance_events': disturbance_events,
            'keystone_history': keystone_history,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Stats
            'total_disturbances': state['total_disturbances'] + (1 if disturbance_triggered else 0),
            'max_keystone_count': max_keystones,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Role distribution
        role_counts = {'pioneer': 0, 'intermediate': 0, 'climax': 0}
        for func in active_palette:
            role = FUNCTION_ROLES.get(func, 'intermediate')
            role_counts[role] += 1

        # Top contributors
        top_idx = jnp.argsort(new_contribution)[-5:][::-1]
        top_contribution = [(int(i), float(new_contribution[i])) for i in top_idx]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Ecosystem
            'resources': new_resources,
            'community_age': community_age,
            'succession_stage': stage,
            'in_recovery': in_recovery,
            'disturbance_triggered': disturbance_triggered,
            # Keystones
            'n_keystones': len(keystones),
            'keystones': list(keystones),
            'n_facilitated': len(facilitated),
            # Role distribution
            'role_counts': role_counts,
            # Contribution
            'top_contribution': top_contribution,
            'mean_contribution': float(jnp.mean(new_contribution)),
            # Stats
            'total_disturbances': new_state['total_disturbances'],
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_is_keystone': 4 in keystones,
            'sin_contribution': float(new_contribution[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with ecosystem status."""
        palette = self.get_active_palette(state)
        contribution = state['contribution']

        # Role distribution
        role_counts = {'pioneer': 0, 'intermediate': 0, 'climax': 0}
        for func in palette:
            role = FUNCTION_ROLES.get(func, 'intermediate')
            role_counts[role] += 1

        # Top contributors
        top_idx = jnp.argsort(contribution)[-5:][::-1]
        top_contribution = [(int(i), float(contribution[i])) for i in top_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Ecosystem
            'resources': state['resources'],
            'community_age': state['community_age'],
            'succession_stage': state['succession_stage'],
            'in_recovery': state['in_recovery'],
            # Keystones
            'keystones': list(state['keystone_functions']),
            'n_keystones': len(state['keystone_functions']),
            # Roles
            'role_counts': role_counts,
            # Contribution
            'top_contribution': top_contribution,
            # Stats
            'total_disturbances': state['total_disturbances'],
            'max_keystone_count': state['max_keystone_count'],
            # Sin
            'sin_is_keystone': 4 in state['keystone_functions'],
        }
