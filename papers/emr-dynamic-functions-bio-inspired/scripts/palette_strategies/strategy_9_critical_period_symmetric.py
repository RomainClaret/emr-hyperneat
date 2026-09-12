"""Strategy 9 Symmetric: Critical Period with Full Symmetric Discovery.

Extends CriticalPeriodDualStrategy with enhancements:
- Sin and extreme-agg affinity floors
- Discovery-to-palette bridging
- Separate phase parameters per dimension
- Enhanced discovery metrics

Three phases for EACH domain with independent timing:
1. Exploration: Aggressive discovery with sin/extreme protection
2. Confirmation: Prune unhelpful, confirm useful
3. Consolidation: Lock confirmed, minimal changes

Key innovation: Asymmetric phase timing - aggregations can have longer exploration.
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
    CORE_EXTREME_AGGS,
    SIN_IDX,
)


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class CriticalPeriodSymmetricStrategy(PaletteEvolutionStrategy):
    """Critical period with symmetric activation-aggregation discovery.

    Key enhancements:
    1. Separate phase boundaries per dimension (aggs can explore longer)
    2. Sin and extreme-agg affinity floors
    3. Discovery-to-palette bridging
    4. Enhanced discovery metrics
    """

    name = "critical_period_symmetric"
    description = "Symmetric critical period with discovery protection"

    def __init__(
        self,
        # === ACTIVATION PHASE BOUNDARIES ===
        act_exploration_end: int = 20,
        act_confirmation_end: int = 50,
        # === AGGREGATION PHASE BOUNDARIES (longer exploration) ===
        agg_exploration_end: int = 30,
        agg_confirmation_end: int = 60,
        # === PHASE-SPECIFIC RATES: ACTIVATIONS ===
        act_exploration_activate: float = 0.35,
        act_exploration_deactivate: float = 0.02,
        act_confirmation_activate: float = 0.10,
        act_confirmation_deactivate: float = 0.15,
        act_consolidation_activate: float = 0.02,
        act_consolidation_deactivate: float = 0.01,
        # === PHASE-SPECIFIC RATES: AGGREGATIONS ===
        agg_exploration_activate: float = 0.40,  # Higher for agg exploration
        agg_exploration_deactivate: float = 0.02,
        agg_confirmation_activate: float = 0.08,
        agg_confirmation_deactivate: float = 0.12,
        agg_consolidation_activate: float = 0.02,
        agg_consolidation_deactivate: float = 0.01,
        # === CONFIRMATION TRACKING ===
        confirmation_persistence: int = 5,
        # === CROSS-DOMAIN LEARNING ===
        cross_learning_rate: float = 0.12,
        cross_protection_boost: float = 0.35,
        # === AFFINITY FLOORS ===
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # === DISCOVERY BOOST ===
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # === CONSTRAINTS ===
        min_active_act: int = 2,
        min_active_agg: int = 2,
        max_active_act: int = 8,
        max_active_agg: int = 5,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize symmetric critical period strategy."""
        # Activation phase boundaries
        self.act_exploration_end = act_exploration_end
        self.act_confirmation_end = act_confirmation_end
        # Aggregation phase boundaries
        self.agg_exploration_end = agg_exploration_end
        self.agg_confirmation_end = agg_confirmation_end

        # Activation rates
        self.act_exploration_activate = act_exploration_activate
        self.act_exploration_deactivate = act_exploration_deactivate
        self.act_confirmation_activate = act_confirmation_activate
        self.act_confirmation_deactivate = act_confirmation_deactivate
        self.act_consolidation_activate = act_consolidation_activate
        self.act_consolidation_deactivate = act_consolidation_deactivate

        # Aggregation rates
        self.agg_exploration_activate = agg_exploration_activate
        self.agg_exploration_deactivate = agg_exploration_deactivate
        self.agg_confirmation_activate = agg_confirmation_activate
        self.agg_confirmation_deactivate = agg_confirmation_deactivate
        self.agg_consolidation_activate = agg_consolidation_activate
        self.agg_consolidation_deactivate = agg_consolidation_deactivate

        self.confirmation_persistence = confirmation_persistence
        self.cross_learning_rate = cross_learning_rate
        self.cross_protection_boost = cross_protection_boost

        # enhancements
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Default palettes with sin and extreme aggs
        default_act = list(DEFAULT_PALETTE_INDICES)
        if SIN_IDX not in default_act:
            default_act.append(SIN_IDX)
        default_agg = list(DEFAULT_AGG_PALETTE_INDICES)
        for agg in CORE_EXTREME_AGGS:
            if agg not in default_agg:
                default_agg.append(agg)

        self.initial_act_palette = initial_act_palette or default_act
        self.initial_agg_palette = initial_agg_palette or default_agg

    def _get_phase(self, generation: int, is_activation: bool = True) -> str:
        """Determine current developmental phase with separate timing per dimension."""
        if is_activation:
            if generation < self.act_exploration_end:
                return CriticalPeriodPhase.EXPLORATION
            elif generation < self.act_confirmation_end:
                return CriticalPeriodPhase.CONFIRMATION
            else:
                return CriticalPeriodPhase.CONSOLIDATION
        else:
            if generation < self.agg_exploration_end:
                return CriticalPeriodPhase.EXPLORATION
            elif generation < self.agg_confirmation_end:
                return CriticalPeriodPhase.CONFIRMATION
            else:
                return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with symmetric discovery tracking."""
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Persistence tracking
        act_persistence = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_persistence = act_persistence.at[i].set(1)

        agg_persistence = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_persistence = agg_persistence.at[i].set(1)

        # Confirmed functions
        act_confirmed = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.bool_)
        agg_confirmed = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.bool_)

        # Cross-domain co-persistence
        cross_copersistence = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS), dtype=jnp.int32)

        # Affinities with floors
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.4
        act_affinities = act_affinities.at[SIN_IDX].set(0.8)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.6)

        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.45
        for agg in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[agg].set(0.75)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(max(float(agg_affinities[i]), 0.55))

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Persistence
            'act_persistence': act_persistence,
            'agg_persistence': agg_persistence,
            # Confirmed
            'act_confirmed': act_confirmed,
            'agg_confirmed': agg_confirmed,
            # Cross-domain
            'cross_copersistence': cross_copersistence,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Discovery tracking
            'discovered_aggs': set(),
            'discovery_to_palette': 0,
            # Common
            'rng_key': jax.random.PRNGKey(seed + 99999),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'act_phase': CriticalPeriodPhase.EXPLORATION,
            'agg_phase': CriticalPeriodPhase.EXPLORATION,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_persistence_and_confirmation(
        self,
        mask: jnp.ndarray,
        persistence: jnp.ndarray,
        confirmed: jnp.ndarray,
        affinities: jnp.ndarray,
        best_fitness: float,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update persistence, confirmation, and affinities."""
        active = mask > 0.5

        # Update persistence
        new_persistence = jnp.where(active, persistence + 1, jnp.zeros_like(persistence))

        # Confirm long-present functions
        long_enough = new_persistence >= self.confirmation_persistence
        new_confirmed = jnp.logical_or(confirmed, long_enough)

        # Update affinities
        new_affinities = affinities * 0.98  # Decay
        if fitness_delta > 0:
            for i in range(len(affinities)):
                if active[i]:
                    new_affinities = new_affinities.at[i].set(
                        min(1.0, new_affinities[i] + 0.1 * fitness_delta)
                    )

        return new_persistence, new_confirmed, new_affinities

    def _apply_affinity_floors(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors for sin and extreme aggs."""
        new_act_aff = act_affinities.at[SIN_IDX].set(
            max(self.sin_affinity_floor, float(act_affinities[SIN_IDX]))
        )
        new_agg_aff = agg_affinities.copy()
        for agg in CORE_EXTREME_AGGS:
            new_agg_aff = new_agg_aff.at[agg].set(
                max(self.extreme_agg_affinity_floor, float(new_agg_aff[agg]))
            )
        return new_act_aff, new_agg_aff

    def _compute_cross_protection(
        self,
        copersistence: jnp.ndarray,
        other_mask: jnp.ndarray,
        other_confirmed: jnp.ndarray,
        is_activation: bool = True,
    ) -> jnp.ndarray:
        """Compute cross-domain protection based on co-persistence."""
        other_active = (other_mask > 0.5).astype(jnp.float32)
        other_confirmed_float = other_confirmed.astype(jnp.float32)
        weighted_other = other_active * (0.5 + 0.5 * other_confirmed_float)

        if is_activation:
            copersist_score = jnp.dot(copersistence.astype(jnp.float32), weighted_other)
        else:
            copersist_score = jnp.dot(copersistence.T.astype(jnp.float32), weighted_other)

        max_score = jnp.max(copersist_score) + 1e-6
        return copersist_score / max_score

    def _select_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        phase: str,
        confirmed: jnp.ndarray,
        cross_protection: jnp.ndarray,
        activate_rates: Tuple[float, float, float],
        deactivate_rates: Tuple[float, float, float],
        min_active: int,
        max_active: int,
        n_functions: int,
        special_indices: List[int] = None,
        newly_discovered: List[int] = None,
    ) -> Tuple[jnp.ndarray, List[int], List[int], int]:
        """Select new palette with discovery bridging."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        discovery_to_palette = 0

        # Phase-specific rates
        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = activate_rates[0]
            deactivate_rate = deactivate_rates[0]
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = activate_rates[1]
            deactivate_rate = deactivate_rates[1]
        else:
            activate_rate = activate_rates[2]
            deactivate_rate = deactivate_rates[2]

        # Compute scores
        score = affinities + cross_protection * 0.3
        if special_indices:
            for idx in special_indices:
                if 0 <= idx < n_functions:
                    score = score.at[idx].set(score[idx] + 0.5)

        # Discovery boost
        if newly_discovered:
            for idx in newly_discovered:
                if 0 <= idx < n_functions:
                    score = score.at[idx].set(score[idx] + self.discovery_boost)

        # Select top functions
        current_active = int(jnp.sum(mask > 0.5))
        target_size = min(max(min_active, current_active), max_active)

        activate_probs = jax.random.uniform(key1, (n_functions,))
        deactivate_probs = jax.random.uniform(key2, (n_functions,))

        for i in range(n_functions):
            cross_prot = float(cross_protection[i])
            is_special = special_indices and i in special_indices

            if mask[i] < 0.5:
                # Might activate
                if current_active + len(activated) >= max_active:
                    continue
                rate = activate_rate * (1 + float(affinities[i]) * 0.5)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Might deactivate
                if phase == CriticalPeriodPhase.CONSOLIDATION and confirmed[i]:
                    continue
                if is_special:
                    continue  # Never deactivate sin/extreme

                protection_factor = 1.0 - self.cross_protection_boost * cross_prot
                effective_deact = deactivate_rate * max(0.1, protection_factor)

                if deactivate_probs[i] < effective_deact:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure discovered functions enter palette
        if self.enable_discovery_slot and newly_discovered:
            for idx in newly_discovered:
                if 0 <= idx < n_functions and new_mask[idx] < 0.5:
                    new_mask = new_mask.at[idx].set(1.0)
                    discovery_to_palette += 1

        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []
            discovery_to_palette = 0

        return new_mask, activated, deactivated, discovery_to_palette

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with symmetric critical period logic."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Get phases (separate per dimension)
        act_phase = self._get_phase(generation, is_activation=True)
        agg_phase = self._get_phase(generation, is_activation=False)

        # Update persistence and affinities
        new_act_persistence, new_act_confirmed, new_act_aff = self._update_persistence_and_confirmation(
            state['act_mask'], state['act_persistence'], state['act_confirmed'],
            state['act_affinities'], best_fitness, fitness_delta
        )
        new_agg_persistence, new_agg_confirmed, new_agg_aff = self._update_persistence_and_confirmation(
            state['agg_mask'], state['agg_persistence'], state['agg_confirmed'],
            state['agg_affinities'], best_fitness, fitness_delta
        )

        # Apply affinity floors
        new_act_aff, new_agg_aff = self._apply_affinity_floors(new_act_aff, new_agg_aff)

        # Update cross co-persistence
        act_active = (state['act_mask'] > 0.5).astype(jnp.int32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.int32)
        co_active = jnp.outer(act_active, agg_active)
        new_cross = state['cross_copersistence'] + co_active

        # Compute cross-protection
        act_cross_prot = self._compute_cross_protection(new_cross, state['agg_mask'], new_agg_confirmed, True)
        agg_cross_prot = self._compute_cross_protection(new_cross, state['act_mask'], new_act_confirmed, False)

        # Track newly discovered aggs
        current_agg_indices = set(mask_to_indices(state['agg_mask']))
        previously_discovered = state.get('discovered_aggs', set())
        newly_discovered_aggs = [i for i in current_agg_indices if i not in previously_discovered and i not in CORE_EXTREME_AGGS]

        # Select palettes
        act_rates = (self.act_exploration_activate, self.act_confirmation_activate, self.act_consolidation_activate)
        act_deact = (self.act_exploration_deactivate, self.act_confirmation_deactivate, self.act_consolidation_deactivate)

        new_act_mask, act_activated, act_deactivated, _ = self._select_palette(
            key_act, state['act_mask'], new_act_aff, act_phase, new_act_confirmed,
            act_cross_prot, act_rates, act_deact,
            self.min_active_act, self.max_active_act, NUM_ACTIVATIONS,
            special_indices=[SIN_IDX], newly_discovered=None
        )

        agg_rates = (self.agg_exploration_activate, self.agg_confirmation_activate, self.agg_consolidation_activate)
        agg_deact = (self.agg_exploration_deactivate, self.agg_confirmation_deactivate, self.agg_consolidation_deactivate)

        new_agg_mask, agg_activated, agg_deactivated, discovery_to_palette = self._select_palette(
            key_agg, state['agg_mask'], new_agg_aff, agg_phase, new_agg_confirmed,
            agg_cross_prot, agg_rates, agg_deact,
            self.min_active_agg, self.max_active_agg, NUM_AGGREGATIONS,
            special_indices=list(CORE_EXTREME_AGGS), newly_discovered=newly_discovered_aggs
        )

        # Update discovered set
        new_discovered = previously_discovered | set(mask_to_indices(new_agg_mask))

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_persistence': new_act_persistence,
            'agg_persistence': new_agg_persistence,
            'act_confirmed': new_act_confirmed,
            'agg_confirmed': new_agg_confirmed,
            'cross_copersistence': new_cross,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'discovered_aggs': new_discovered,
            'discovery_to_palette': state.get('discovery_to_palette', 0) + discovery_to_palette,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'act_phase': act_phase,
            'agg_phase': agg_phase,
        }

        # Metrics
        sin_retained = new_act_mask[SIN_IDX] > 0.5
        extreme_retained = all(new_agg_mask[agg] > 0.5 for agg in CORE_EXTREME_AGGS)

        metrics = {
            'act_phase': act_phase,
            'agg_phase': agg_phase,
            'n_act_confirmed': int(jnp.sum(new_act_confirmed)),
            'n_agg_confirmed': int(jnp.sum(new_agg_confirmed)),
            'sin_retained': sin_retained,
            'extreme_retained': extreme_retained,
            'discovery_to_palette': discovery_to_palette,
            'total_discovered_aggs': len(new_discovered),
            'act_activated': act_activated,
            'act_deactivated': act_deactivated,
            'agg_activated': agg_activated,
            'agg_deactivated': agg_deactivated,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with metrics."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': SIN_IDX in act_palette,
            'has_extreme_aggs': all(agg in agg_palette for agg in CORE_EXTREME_AGGS),
            'act_phase': state.get('act_phase', 'unknown'),
            'agg_phase': state.get('agg_phase', 'unknown'),
            'n_act_confirmed': int(jnp.sum(state['act_confirmed'])),
            'n_agg_confirmed': int(jnp.sum(state['agg_confirmed'])),
            'total_discovered_aggs': len(state.get('discovered_aggs', set())),
            'discovery_to_palette': state.get('discovery_to_palette', 0),
        }
