"""Strategy 86: Homeostatic+Reinforcement Dual Hybrid.

Combines:
- Strategy 82 (Homeostatic): Balance + discovery bonus (100% sin retention)
- Strategy 74 (Cross-Domain Reinforcement): 2x learning when both domains change (67% Parity-5)

Key innovation: Reinforcement boosts affinity learning, and high-affinity pairs
get double protection from homeostatic mechanisms.

Bio inspiration: Homeostatic plasticity maintains network stability while
neuromodulatory reinforcement signals strengthen task-relevant pathways.

Expected: Stable sin retention with improved cross-domain exploration.
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


class HomeostaticReinforcementDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution combining homeostatic balance and reinforcement.

    Hybrid combining:
    - Homeostatic plasticity (82): Balance between averaging/extreme aggregations
    - Cross-domain reinforcement (74): 2x learning when both domains improve

    Critical interaction: High-affinity pairs get double protection from
    homeostatic deactivation, creating exploration-exploitation balance.
    """

    name = "homeostatic_reinforcement_dual"
    description = "Dual: Homeostatic balance + cross-domain reinforcement hybrid"

    def __init__(
        self,
        # === Homeostatic parameters (from strategy 82) ===
        target_extreme_ratio: float = 0.60,
        imbalance_threshold: float = 0.15,
        correction_strength: float = 1.8,
        discovery_bonus: float = 0.5,
        extreme_protection: float = 0.6,
        sin_protection: float = 0.5,
        # === Reinforcement parameters (from strategy 74) ===
        base_cross_learning_rate: float = 0.15,
        reinforcement_multiplier: float = 2.0,
        affinity_protection_threshold: float = 0.6,
        affinity_protection_strength: float = 0.5,
        # === Combined: Double protection threshold ===
        double_protection_affinity: float = 0.7,  # Very high affinity = extra protection
        # === Mutation parameters ===
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        stagnation_threshold: int = 5,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Homeostatic+Reinforcement hybrid strategy."""
        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.imbalance_threshold = imbalance_threshold
        self.correction_strength = correction_strength
        self.discovery_bonus = discovery_bonus
        self.extreme_protection = extreme_protection
        self.sin_protection = sin_protection

        # Reinforcement
        self.base_cross_learning_rate = base_cross_learning_rate
        self.reinforcement_multiplier = reinforcement_multiplier
        self.affinity_protection_threshold = affinity_protection_threshold
        self.affinity_protection_strength = affinity_protection_strength

        # Combined
        self.double_protection_affinity = double_protection_affinity

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.stagnation_threshold = stagnation_threshold

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with homeostatic and reinforcement tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity matrix
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Track previous palette state for reinforcement detection
        prev_act_mask = act_mask.copy()
        prev_agg_mask = agg_mask.copy()

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'prev_act_mask': prev_act_mask,
            'prev_agg_mask': prev_agg_mask,
            # Reinforcement
            'cross_affinity': cross_affinity,
            # Tracking
            'homeostatic_corrections': 0,
            'discovery_bonuses_applied': 0,
            'reinforcement_events': 0,
            'double_protection_events': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 860000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute current extreme/averaging ratio."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def _update_cross_affinity_reinforced(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        prev_act_mask: jnp.ndarray,
        prev_agg_mask: jnp.ndarray,
        improvement: float,
    ) -> Tuple[jnp.ndarray, int]:
        """Update cross-domain affinity with reinforcement multiplier."""
        new_affinity = cross_affinity.copy()
        reinforcement_events = 0

        if improvement > 0:
            # Check if BOTH domains changed
            act_changed = not jnp.allclose(act_mask, prev_act_mask)
            agg_changed = not jnp.allclose(agg_mask, prev_agg_mask)
            both_changed = act_changed and agg_changed

            lr = self.base_cross_learning_rate
            if both_changed:
                lr *= self.reinforcement_multiplier
                reinforcement_events += 1

            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    for j in range(NUM_AGGREGATIONS):
                        if agg_mask[j] > 0.5:
                            current = cross_affinity[i, j]
                            boost = lr * improvement
                            # Extra boost for sin-extreme pairs
                            if i == 4 and j in CORE_EXTREME_AGGS:
                                boost *= 1.3
                            new_affinity = new_affinity.at[i, j].set(
                                min(1.0, current + boost)
                            )

        return new_affinity, reinforcement_events

    def _mutate_act_palette_hybrid(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict, int]:
        """Mutate activation palette with homeostatic and affinity protection."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        double_protection_count = 0

        for i in range(NUM_ACTIVATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            max_affinity = float(jnp.max(cross_affinity[i, :]))

            if mask[i] < 0.5:  # Inactive
                activate_rate = self.base_activate_rate
                # Discovery bonus for sin
                if i == 4:
                    activate_rate += self.discovery_bonus
                # Affinity boost
                if max_affinity > self.affinity_protection_threshold:
                    activate_rate *= 1.5
                if p < activate_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                deactivate_rate = self.base_deactivate_rate

                # SIN PROTECTION (homeostatic)
                if i == 4:
                    deactivate_rate *= (1 - self.sin_protection)

                # AFFINITY PROTECTION (reinforcement)
                if max_affinity > self.affinity_protection_threshold:
                    deactivate_rate *= (1 - self.affinity_protection_strength)

                # DOUBLE PROTECTION for very high affinity
                if max_affinity > self.double_protection_affinity:
                    deactivate_rate *= 0.5
                    double_protection_count += 1

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_act or active_count > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}, double_protection_count

    def _mutate_agg_palette_hybrid(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        extreme_ratio: float,
    ) -> Tuple[jnp.ndarray, Dict, int, int, int]:
        """Mutate aggregation palette with homeostatic balance and affinity protection."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        homeostatic_corrections = 0
        discovery_bonuses = 0
        double_protection_count = 0

        # Check if we need homeostatic correction
        needs_more_extreme = extreme_ratio < self.target_extreme_ratio - self.imbalance_threshold
        needs_more_averaging = extreme_ratio > self.target_extreme_ratio + self.imbalance_threshold

        for j in range(NUM_AGGREGATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            sin_affinity = float(cross_affinity[4, j]) if act_mask[4] > 0.5 else 0.5
            max_affinity = float(jnp.max(cross_affinity[:, j]))
            is_extreme = j in EXTREME_AGGS
            is_core_extreme = j in CORE_EXTREME_AGGS

            if mask[j] < 0.5:  # Inactive
                activate_rate = self.base_activate_rate

                # Homeostatic correction
                if needs_more_extreme and is_core_extreme:
                    activate_rate *= self.correction_strength
                    homeostatic_corrections += 1
                elif needs_more_averaging and not is_extreme:
                    activate_rate *= self.correction_strength
                    homeostatic_corrections += 1

                # Discovery bonus
                if is_core_extreme and extreme_ratio < 0.5:
                    activate_rate += self.discovery_bonus
                    discovery_bonuses += 1

                # Affinity boost
                if max_affinity > self.affinity_protection_threshold:
                    activate_rate *= 1.5

                if p < activate_rate:
                    new_mask = new_mask.at[j].set(1.0)
                    activated.append(j)
            else:  # Active
                deactivate_rate = self.base_deactivate_rate

                # EXTREME PROTECTION (homeostatic)
                if is_core_extreme:
                    deactivate_rate *= (1 - self.extreme_protection)

                # Homeostatic balance protection
                if needs_more_extreme and is_extreme:
                    deactivate_rate *= 0.5
                elif needs_more_averaging and not is_extreme:
                    deactivate_rate *= 0.5

                # AFFINITY PROTECTION (reinforcement)
                if max_affinity > self.affinity_protection_threshold:
                    deactivate_rate *= (1 - self.affinity_protection_strength)

                # DOUBLE PROTECTION for very high affinity
                if max_affinity > self.double_protection_affinity:
                    deactivate_rate *= 0.5
                    double_protection_count += 1

                if p < deactivate_rate:
                    new_mask = new_mask.at[j].set(0.0)
                    deactivated.append(j)

        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}, homeostatic_corrections, discovery_bonuses, double_protection_count

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with combined homeostatic+reinforcement mechanisms."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === REINFORCEMENT MECHANISM ===
        new_cross_affinity, reinforcement_events = self._update_cross_affinity_reinforced(
            state['cross_affinity'],
            state['act_mask'], state['agg_mask'],
            state['prev_act_mask'], state['prev_agg_mask'],
            improvement
        )

        # === HOMEOSTATIC BALANCE ===
        extreme_ratio = self._compute_extreme_ratio(state['agg_mask'])

        # === MUTATIONS ===
        should_mutate = new_stagnation >= self.stagnation_threshold
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        total_homeostatic = 0
        total_discovery = 0
        total_double_protection = 0

        if should_mutate:
            new_act_mask, _, double_prot_act = self._mutate_act_palette_hybrid(
                k_act, state['act_mask'], new_cross_affinity
            )
            new_agg_mask, _, homeostatic_corrections, discovery_bonuses, double_prot_agg = self._mutate_agg_palette_hybrid(
                k_agg, state['agg_mask'], new_cross_affinity, state['act_mask'], extreme_ratio
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            total_homeostatic = homeostatic_corrections
            total_discovery = discovery_bonuses
            total_double_protection = double_prot_act + double_prot_agg
            new_stagnation = 0

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'prev_act_mask': state['act_mask'],
            'prev_agg_mask': state['agg_mask'],
            'cross_affinity': new_cross_affinity,
            'homeostatic_corrections': state['homeostatic_corrections'] + total_homeostatic,
            'discovery_bonuses_applied': state['discovery_bonuses_applied'] + total_discovery,
            'reinforcement_events': state['reinforcement_events'] + reinforcement_events,
            'double_protection_events': state['double_protection_events'] + total_double_protection,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Homeostatic metrics
            'extreme_ratio': extreme_ratio,
            'homeostatic_corrections': state['homeostatic_corrections'] + total_homeostatic,
            'discovery_bonuses_applied': state['discovery_bonuses_applied'] + total_discovery,
            # Reinforcement metrics
            'reinforcement_events': state['reinforcement_events'] + reinforcement_events,
            'double_protection_events': state['double_protection_events'] + total_double_protection,
            'sin_max_affinity': float(new_cross_affinity[4, 2]),
            'sin_min_affinity': float(new_cross_affinity[4, 3]),
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'homeostatic_corrections': state['homeostatic_corrections'],
            'discovery_bonuses_applied': state['discovery_bonuses_applied'],
            'reinforcement_events': state['reinforcement_events'],
            'double_protection_events': state['double_protection_events'],
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
