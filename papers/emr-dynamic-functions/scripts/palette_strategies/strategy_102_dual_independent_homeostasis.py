"""Strategy 102: Dual Independent Homeostasis.

Aggregation-first approach with separate homeostatic systems for each domain:
- Problem: Current strategies have coupled activation-aggregation learning, so
  losing sin causes cascading loss of extreme aggs (and vice versa)
- Hypothesis: Independent homeostatic balance for each domain prevents
  cross-domain interference while still allowing coordinated captures

Key mechanisms:
- Activation domain: Maintains target oscillatory ratio (sin, burst, resonator)
- Aggregation domain: Maintains target extreme ratio (max, min)
- Cross-domain interference factor: Limits how much one domain affects the other
- Each domain has its own homeostatic pressure independent of the other

Bio inspiration: Different brain regions maintain independent homeostatic balance.
The visual cortex doesn't need motor cortex approval to regulate its own activity
levels. Each domain is autonomous but can still coordinate when beneficial.

Expected: More robust retention through domain-independent balance.
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)

# Define oscillatory activation indices (sin-related family)
OSCILLATORY_ACTS = [4, 11, 12, 13, 15]  # sin, burst, resonator, pulse, wave


class DualIndependentHomeostasisStrategy(PaletteEvolutionStrategy):
    """Separate homeostatic systems for activation and aggregation domains.

    Strategy:
    - Each domain maintains its own balance independently
    - Cross-domain interference is limited
    - Captures can still coordinate but don't force coupling
    """

    name = "dual_independent_homeostasis"
    description = "Dual: Independent homeostatic balance per domain"

    def __init__(
        self,
        # === Independent homeostasis parameters ===
        act_target_oscillatory_ratio: float = 0.40,  # Target 40% oscillatory in act
        agg_target_extreme_ratio: float = 0.60,       # Target 60% extreme in agg
        cross_domain_interference_factor: float = 0.2,  # How much domains affect each other
        homeostatic_strength: float = 0.3,            # Strength of homeostatic pressure
        # === Tag-and-capture parameters ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        # === Affinity learning ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Cross-domain parameters ===
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Dual Independent Homeostasis strategy."""
        # Independent homeostasis
        self.act_target_oscillatory_ratio = act_target_oscillatory_ratio
        self.agg_target_extreme_ratio = agg_target_extreme_ratio
        self.cross_domain_interference_factor = cross_domain_interference_factor
        self.homeostatic_strength = homeostatic_strength

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with independent homeostatic tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Independent homeostatic state
        act_homeostatic_pressure = 0.0  # Positive = need more oscillatory, negative = need less
        agg_homeostatic_pressure = 0.0  # Positive = need more extreme, negative = need less

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'generation': 0,
            'best_fitness': 0.0,
            'stagnation_counter': 0,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging state
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'act_tag_gens': jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32),
            'agg_tag_gens': jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32),
            # Homeostatic state
            'act_homeostatic_pressure': act_homeostatic_pressure,
            'agg_homeostatic_pressure': agg_homeostatic_pressure,
        }

    def _compute_homeostatic_pressure(
        self,
        mask: jnp.ndarray,
        target_ratio: float,
        target_indices: List[int],
        domain: str
    ) -> float:
        """Compute homeostatic pressure for a domain.

        Returns positive if need more target functions, negative if need less.
        """
        active_indices = mask_to_indices(mask)
        total_active = len(active_indices)

        if total_active == 0:
            return target_ratio  # Strong pressure to add any functions

        # Count target functions
        target_count = sum(1 for i in active_indices if i in target_indices)
        current_ratio = target_count / total_active

        # Compute pressure
        pressure = (target_ratio - current_ratio) * self.homeostatic_strength

        return pressure

    def update(
        self,
        state: Dict[str, Any],
        fitness_scores: jnp.ndarray,
        function_usage: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update with independent homeostatic systems."""
        generation = state['generation']
        best_fitness = float(jnp.max(fitness_scores))
        mean_fitness = float(jnp.mean(fitness_scores))

        # Update fitness tracking
        improved = best_fitness > state['best_fitness'] + 1e-6
        state['best_fitness'] = max(state['best_fitness'], best_fitness)
        state['stagnation_counter'] = 0 if improved else state['stagnation_counter'] + 1

        # Update affinities from fitness
        act_affinities = state['act_affinities'] * self.affinity_decay
        agg_affinities = state['agg_affinities'] * self.affinity_decay

        act_indices = mask_to_indices(state['act_mask'])
        agg_indices = mask_to_indices(state['agg_mask'])
        fitness_delta = best_fitness - state.get('prev_best_fitness', 0.0)

        if improved and fitness_delta > 0:
            for i in act_indices:
                if 0 <= i < NUM_ACTIVATIONS:
                    boost = self.act_affinity_lr * fitness_delta * 10
                    act_affinities = act_affinities.at[i].set(
                        min(1.0, float(act_affinities[i]) + boost)
                    )
            for i in agg_indices:
                if 0 <= i < NUM_AGGREGATIONS:
                    boost = self.agg_affinity_lr * fitness_delta * 10
                    agg_affinities = agg_affinities.at[i].set(
                        min(1.0, float(agg_affinities[i]) + boost)
                    )

        state['act_affinities'] = act_affinities
        state['agg_affinities'] = agg_affinities
        state['prev_best_fitness'] = best_fitness

        # === Compute INDEPENDENT homeostatic pressures ===
        act_pressure = self._compute_homeostatic_pressure(
            state['act_mask'],
            self.act_target_oscillatory_ratio,
            OSCILLATORY_ACTS,
            'act'
        )
        agg_pressure = self._compute_homeostatic_pressure(
            state['agg_mask'],
            self.agg_target_extreme_ratio,
            CORE_EXTREME_AGGS,
            'agg'
        )

        # Apply homeostatic pressure to affinities (INDEPENDENT for each domain)
        # Activation domain: boost oscillatory if under-represented
        if act_pressure > 0:  # Need more oscillatory
            for i in OSCILLATORY_ACTS:
                if 0 <= i < NUM_ACTIVATIONS:
                    boost = act_pressure * 0.1
                    act_affinities = act_affinities.at[i].set(
                        min(1.0, float(act_affinities[i]) + boost)
                    )
        else:  # Too many oscillatory - slightly boost others
            for i in range(NUM_ACTIVATIONS):
                if i not in OSCILLATORY_ACTS:
                    boost = abs(act_pressure) * 0.05
                    act_affinities = act_affinities.at[i].set(
                        min(1.0, float(act_affinities[i]) + boost)
                    )

        # Aggregation domain: boost extreme if under-represented
        if agg_pressure > 0:  # Need more extreme
            for i in CORE_EXTREME_AGGS:
                if 0 <= i < NUM_AGGREGATIONS:
                    boost = agg_pressure * 0.15
                    agg_affinities = agg_affinities.at[i].set(
                        min(1.0, float(agg_affinities[i]) + boost)
                    )
        else:  # Too many extreme - slightly boost averaging
            for i in AVERAGING_AGGS:
                if 0 <= i < NUM_AGGREGATIONS:
                    boost = abs(agg_pressure) * 0.05
                    agg_affinities = agg_affinities.at[i].set(
                        min(1.0, float(agg_affinities[i]) + boost)
                    )

        # Cross-domain reinforcement (limited by interference factor)
        # If sin is doing well AND extreme aggs are doing well, boost both slightly
        sin_affinity = float(act_affinities[4])
        max_affinity = float(agg_affinities[2])
        min_affinity = float(agg_affinities[3])
        extreme_avg = (max_affinity + min_affinity) / 2

        cross_bonus = sin_affinity * extreme_avg * self.cross_domain_interference_factor
        if cross_bonus > 0.05:
            # Mutual reinforcement (but limited)
            act_affinities = act_affinities.at[4].set(
                min(1.0, sin_affinity + cross_bonus * 0.5)
            )
            agg_affinities = agg_affinities.at[2].set(
                min(1.0, max_affinity + cross_bonus * 0.5)
            )
            agg_affinities = agg_affinities.at[3].set(
                min(1.0, min_affinity + cross_bonus * 0.5)
            )

        state['act_affinities'] = act_affinities
        state['agg_affinities'] = agg_affinities
        state['act_homeostatic_pressure'] = act_pressure
        state['agg_homeostatic_pressure'] = agg_pressure

        # Decay tags
        act_tags = state['act_tags'] * self.tag_decay
        agg_tags = state['agg_tags'] * self.tag_decay

        # Update tags from high-affinity functions
        for i in range(NUM_ACTIVATIONS):
            if float(act_affinities[i]) > self.tag_threshold:
                act_tags = act_tags.at[i].set(
                    min(1.0, float(act_tags[i]) + 0.2)
                )
        for i in range(NUM_AGGREGATIONS):
            threshold = self.agg_tag_threshold
            if i in CORE_EXTREME_AGGS:
                threshold *= 0.8
            if float(agg_affinities[i]) > threshold:
                boost = 0.2 * (self.extreme_tag_boost if i in CORE_EXTREME_AGGS else 1.0)
                agg_tags = agg_tags.at[i].set(min(1.0, float(agg_tags[i]) + boost))

        # Track tag duration
        act_tag_gens = state['act_tag_gens']
        agg_tag_gens = state['agg_tag_gens']
        for i in range(NUM_ACTIVATIONS):
            if float(act_tags[i]) > 0.3:
                act_tag_gens = act_tag_gens.at[i].set(int(act_tag_gens[i]) + 1)
            else:
                act_tag_gens = act_tag_gens.at[i].set(0)
        for i in range(NUM_AGGREGATIONS):
            if float(agg_tags[i]) > 0.3:
                agg_tag_gens = agg_tag_gens.at[i].set(int(agg_tag_gens[i]) + 1)
            else:
                agg_tag_gens = agg_tag_gens.at[i].set(0)

        # Capture mechanism
        act_captured = state['act_captured']
        agg_captured = state['agg_captured']

        for i in range(NUM_ACTIVATIONS):
            if float(act_captured[i]) < 0.5 and int(act_tag_gens[i]) >= self.capture_window:
                act_captured = act_captured.at[i].set(1.0)

        for i in range(NUM_AGGREGATIONS):
            window = self.capture_window - 1 if i in CORE_EXTREME_AGGS else self.capture_window
            if float(agg_captured[i]) < 0.5 and int(agg_tag_gens[i]) >= window:
                agg_captured = agg_captured.at[i].set(1.0)

        state['act_tags'] = act_tags
        state['agg_tags'] = agg_tags
        state['act_tag_gens'] = act_tag_gens
        state['agg_tag_gens'] = agg_tag_gens
        state['act_captured'] = act_captured
        state['agg_captured'] = agg_captured
        state['generation'] = generation + 1

        return state

    def mutate(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any],
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], jax.random.PRNGKey]:
        """Mutate with homeostatic-guided selection."""
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']
        act_captured = state['act_captured']
        agg_captured = state['agg_captured']
        act_pressure = state.get('act_homeostatic_pressure', 0.0)
        agg_pressure = state.get('agg_homeostatic_pressure', 0.0)

        mutation_rate = 0.08

        rng_key, k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 7)

        act_indices = mask_to_indices(act_mask)
        agg_indices = mask_to_indices(agg_mask)

        # Activation mutations - bias by homeostatic pressure
        if jax.random.uniform(k1) < mutation_rate:
            inactive_acts = [i for i in range(NUM_ACTIVATIONS)
                           if float(act_mask[i]) < 0.5]
            if inactive_acts:
                weights = []
                for i in inactive_acts:
                    w = float(act_affinities[i]) + 0.1
                    # Homeostatic bias
                    if i in OSCILLATORY_ACTS and act_pressure > 0:
                        w *= (1 + act_pressure)
                    elif i not in OSCILLATORY_ACTS and act_pressure < 0:
                        w *= (1 + abs(act_pressure) * 0.5)
                    weights.append(w)
                total = sum(weights)
                weights = [w / total for w in weights]
                idx = jax.random.choice(k2, jnp.array(inactive_acts),
                                       p=jnp.array(weights))
                act_mask = act_mask.at[int(idx)].set(1.0)

        # Remove low-affinity act (not captured)
        if jax.random.uniform(k3) < mutation_rate and len(act_indices) > self.min_active_act:
            candidates = [i for i in act_indices
                         if float(act_captured[i]) < 0.5
                         and i != 4]
            if candidates:
                affs = [(i, float(act_affinities[i])) for i in candidates]
                affs.sort(key=lambda x: x[1])
                to_remove = affs[0][0]
                act_mask = act_mask.at[to_remove].set(0.0)

        # Aggregation mutations - bias by homeostatic pressure
        if jax.random.uniform(k4) < mutation_rate:
            inactive_aggs = [i for i in range(NUM_AGGREGATIONS)
                           if float(agg_mask[i]) < 0.5]
            if inactive_aggs:
                weights = []
                for i in inactive_aggs:
                    w = float(agg_affinities[i]) + 0.1
                    # Homeostatic bias
                    if i in CORE_EXTREME_AGGS and agg_pressure > 0:
                        w *= (1 + agg_pressure * 2)
                    elif i in AVERAGING_AGGS and agg_pressure < 0:
                        w *= (1 + abs(agg_pressure))
                    weights.append(w)
                total = sum(weights)
                weights = [w / total for w in weights]
                idx = jax.random.choice(k5, jnp.array(inactive_aggs),
                                       p=jnp.array(weights))
                agg_mask = agg_mask.at[int(idx)].set(1.0)

        # Remove low-affinity agg (not captured)
        if jax.random.uniform(k6) < mutation_rate and len(agg_indices) > self.min_active_agg:
            candidates = [i for i in agg_indices
                         if float(agg_captured[i]) < 0.5
                         and i not in CORE_EXTREME_AGGS]
            if candidates:
                affs = [(i, float(agg_affinities[i])) for i in candidates]
                affs.sort(key=lambda x: x[1])
                to_remove = affs[0][0]
                agg_mask = agg_mask.at[to_remove].set(0.0)

        # Ensure constraints
        act_active = int(jnp.sum(act_mask))
        agg_active = int(jnp.sum(agg_mask))

        if act_active < self.min_active_act:
            inactive = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if inactive:
                rng_key, k = jax.random.split(rng_key)
                idx = jax.random.choice(k, jnp.array(inactive))
                act_mask = act_mask.at[int(idx)].set(1.0)

        if agg_active < self.min_active_agg:
            inactive = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if inactive:
                rng_key, k = jax.random.split(rng_key)
                idx = jax.random.choice(k, jnp.array(inactive))
                agg_mask = agg_mask.at[int(idx)].set(1.0)

        state['act_mask'] = act_mask
        state['agg_mask'] = agg_mask

        return state, rng_key

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Get current activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Get current aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update after each generation - wrapper for compatibility."""
        fitness_scores = jnp.array([best_fitness])
        state = self.update(state, fitness_scores, {}, {})
        rng_key = state.get('rng_key', jax.random.PRNGKey(generation))
        state, _ = self.mutate(state, {}, rng_key)
        metrics = self.get_diagnostics(state)
        return state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return summary for logging."""
        return self.get_diagnostics(state)

    def get_diagnostics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy diagnostics."""
        act_indices = mask_to_indices(state['act_mask'])
        agg_indices = mask_to_indices(state['agg_mask'])

        act_captured_list = [i for i in range(NUM_ACTIVATIONS)
                           if float(state['act_captured'][i]) > 0.5]
        agg_captured_list = [i for i in range(NUM_AGGREGATIONS)
                           if float(state['agg_captured'][i]) > 0.5]

        # Compute current ratios
        oscillatory_count = sum(1 for i in act_indices if i in OSCILLATORY_ACTS)
        extreme_count = sum(1 for i in agg_indices if i in CORE_EXTREME_AGGS)

        return {
            'generation': state['generation'],
            'act_palette_size': len(act_indices),
            'agg_palette_size': len(agg_indices),
            'has_sin': 4 in act_indices,
            'has_max': 2 in agg_indices,
            'has_min': 3 in agg_indices,
            'act_captured': act_captured_list,
            'agg_captured': agg_captured_list,
            'sin_captured': 4 in act_captured_list,
            'max_captured': 2 in agg_captured_list,
            'min_captured': 3 in agg_captured_list,
            'sin_affinity': float(state['act_affinities'][4]),
            'max_affinity': float(state['agg_affinities'][2]),
            # Homeostatic diagnostics
            'act_homeostatic_pressure': state.get('act_homeostatic_pressure', 0.0),
            'agg_homeostatic_pressure': state.get('agg_homeostatic_pressure', 0.0),
            'oscillatory_ratio': oscillatory_count / len(act_indices) if act_indices else 0,
            'extreme_ratio': extreme_count / len(agg_indices) if agg_indices else 0,
        }
