"""Strategy 82: Homeostatic Aggregation Discovery Dual (Active Balance with Discovery).

Bio inspiration: Homeostatic plasticity maintains stable activity balance while
allowing discovery of new functions. Extends agg_homeostasis_dual with explicit
discovery mechanisms.

Key innovation:
- Active balance between averaging and extreme aggregations
- Discovery bonus for underrepresented categories
- Cross-domain affinity tracking for sin-extreme pairs
- Stronger correction for extreme aggregation retention

Expected: Better aggregation discovery while maintaining balance.
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


class HomeostaticAggDiscoveryDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with homeostatic balance and active discovery.

    Maintains balance between averaging and extreme aggregations while
    actively promoting discovery of underrepresented categories.
    """

    name = "homeostatic_agg_discovery_dual"
    description = "Dual: Homeostatic balance with active aggregation discovery"

    def __init__(
        self,
        # Balance parameters
        target_extreme_ratio: float = 0.60,  # Slightly favor extreme for parity
        imbalance_threshold: float = 0.15,
        correction_strength: float = 1.8,
        # Discovery parameters
        discovery_bonus: float = 0.5,  # Extra activation rate for underrepresented
        exploration_rate: float = 0.20,
        # Mutation rates
        act_mutation_rate: float = 0.10,
        agg_base_activate: float = 0.12,
        agg_base_deactivate: float = 0.08,
        # Protection
        extreme_protection: float = 0.6,  # Reduce deactivation for extreme aggs
        sin_protection: float = 0.5,
        # Cross-domain
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # Scaling
        upscale_rate: float = 0.15,
        downscale_rate: float = 0.08,
        # Stagnation
        stagnation_threshold: int = 5,
        # Palette constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # Initial palettes
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Homeostatic Aggregation Discovery Dual strategy."""
        # Balance
        self.target_extreme_ratio = target_extreme_ratio
        self.imbalance_threshold = imbalance_threshold
        self.correction_strength = correction_strength

        # Discovery
        self.discovery_bonus = discovery_bonus
        self.exploration_rate = exploration_rate

        # Mutation
        self.act_mutation_rate = act_mutation_rate
        self.agg_base_activate = agg_base_activate
        self.agg_base_deactivate = agg_base_deactivate

        # Protection
        self.extreme_protection = extreme_protection
        self.sin_protection = sin_protection

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Scaling
        self.upscale_rate = upscale_rate
        self.downscale_rate = downscale_rate

        # Stagnation
        self.stagnation_threshold = stagnation_threshold

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with homeostasis tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        # Pre-boost sin-extreme affinity
        cross_affinity = cross_affinity.at[4, 2].set(0.5 + self.sin_extreme_affinity_boost)
        cross_affinity = cross_affinity.at[4, 3].set(0.5 + self.sin_extreme_affinity_boost)

        # Activity tracking for scaling
        act_activity = jnp.zeros(NUM_ACTIVATIONS)
        agg_activity = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Activity tracking
            'act_activity': act_activity,
            'agg_activity': agg_activity,
            # Homeostasis tracking
            'extreme_ratio_history': [],
            'corrections_applied': 0,
            'discoveries': {'averaging': 0, 'extreme': 0},
            # General state
            'rng_key': jax.random.PRNGKey(seed + 820000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_extreme_ratio(self, agg_palette: List[int]) -> float:
        """Compute ratio of extreme aggregations in palette."""
        if len(agg_palette) == 0:
            return 0.0
        extreme_count = sum(1 for a in agg_palette if a in EXTREME_AGGS)
        return extreme_count / len(agg_palette)

    def _compute_imbalance(self, agg_palette: List[int]) -> Tuple[float, str]:
        """Compute imbalance and direction."""
        extreme_ratio = self._compute_extreme_ratio(agg_palette)
        imbalance = extreme_ratio - self.target_extreme_ratio

        if imbalance > 0:
            return imbalance, 'extreme'
        else:
            return -imbalance, 'averaging'

    def _update_activity(
        self,
        activity: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update activity tracking based on current palette."""
        new_activity = activity * 0.9  # Decay
        new_activity = new_activity + (mask > 0.5).astype(jnp.float32) * 0.1
        return jnp.clip(new_activity, 0.0, 1.0)

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity with sin-extreme boost."""
        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)

        # Base learning
        delta = self.cross_learning_rate * fitness_delta * co_active
        new_cross = cross_affinity + delta

        # Extra sin-extreme learning
        if act_mask[4] > 0.5 and fitness_delta > 0:
            for agg_idx in CORE_EXTREME_AGGS:
                if agg_mask[agg_idx] > 0.5:
                    boost = self.cross_learning_rate * fitness_delta * 1.5
                    new_cross = new_cross.at[4, agg_idx].set(
                        new_cross[4, agg_idx] + boost
                    )

        return jnp.clip(new_cross, 0.0, 1.0)

    def _mutate_act_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation to activation palette with cross-domain influence."""
        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(NUM_ACTIVATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            if mask[i] < 0.5:  # Inactive
                if p < self.act_mutation_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:  # Active
                deactivate_rate = self.act_mutation_rate

                # Sin protection based on cross-domain affinity
                if i == 4:
                    avg_extreme_affinity = float(jnp.mean(cross_affinity[4, CORE_EXTREME_AGGS]))
                    protection = self.sin_protection * (1 + avg_extreme_affinity)
                    deactivate_rate *= (1 - protection)

                if p < deactivate_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_act or active_count > self.max_active_act:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _mutate_agg_palette_homeostatic(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        imbalance: float,
        direction: str,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply homeostatic mutation with discovery bonus."""
        new_mask = mask.copy()
        activated = []
        deactivated = []
        correction_applied = False

        # Compute per-aggregation rates
        activate_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_base_activate
        deactivate_rates = jnp.ones(NUM_AGGREGATIONS) * self.agg_base_deactivate

        # Discovery bonus for underrepresented category
        if imbalance > self.imbalance_threshold:
            correction_applied = True
            if direction == 'extreme':
                # Too many extreme, boost averaging discovery
                for i in AVERAGING_AGGS:
                    if mask[i] < 0.5:
                        activate_rates = activate_rates.at[i].set(
                            self.agg_base_activate + self.discovery_bonus
                        )
            else:
                # Too few extreme, boost extreme discovery
                for i in EXTREME_AGGS:
                    if mask[i] < 0.5:
                        activate_rates = activate_rates.at[i].set(
                            self.agg_base_activate + self.discovery_bonus
                        )

        # Apply protection to extreme aggregations based on sin affinity
        sin_active = act_mask[4] > 0.5
        for i in CORE_EXTREME_AGGS:
            if mask[i] > 0.5:
                protection = self.extreme_protection
                if sin_active:
                    sin_affinity = float(cross_affinity[4, i])
                    protection += sin_affinity * 0.2
                deactivate_rates = deactivate_rates.at[i].set(
                    self.agg_base_deactivate * (1 - protection)
                )

        for i in range(NUM_AGGREGATIONS):
            p = float(jax.random.uniform(key))
            key = jax.random.split(key)[0]

            if mask[i] < 0.5:
                if p < activate_rates[i]:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if p < deactivate_rates[i]:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Enforce constraints
        active_count = int(jnp.sum(new_mask > 0.5))
        if active_count < self.min_active_agg or active_count > self.max_active_agg:
            new_mask = mask
            activated = []
            deactivated = []
            correction_applied = False

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'correction_applied': correction_applied,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with homeostatic discovery mechanism."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update activity tracking
        new_act_activity = self._update_activity(state['act_activity'], state['act_mask'])
        new_agg_activity = self._update_activity(state['agg_activity'], state['agg_mask'])

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_delta,
        )

        # Compute homeostasis metrics
        agg_palette = mask_to_indices(state['agg_mask'])
        imbalance, direction = self._compute_imbalance(agg_palette)
        extreme_ratio = self._compute_extreme_ratio(agg_palette)

        extreme_ratio_history = state['extreme_ratio_history'] + [extreme_ratio]
        if len(extreme_ratio_history) > 20:
            extreme_ratio_history = extreme_ratio_history[-20:]

        # Mutate if stagnating
        new_act_mask = state['act_mask']
        new_agg_mask = state['agg_mask']
        act_changed = False
        agg_changed = False
        act_mutation_info = None
        agg_mutation_info = None
        correction_applied = False
        new_corrections = state['corrections_applied']
        new_discoveries = dict(state['discoveries'])

        if new_stagnation >= self.stagnation_threshold:
            new_act_mask, act_mutation_info = self._mutate_act_palette(
                k_act, state['act_mask'], new_cross
            )
            new_agg_mask, agg_mutation_info = self._mutate_agg_palette_homeostatic(
                k_agg, state['agg_mask'], imbalance, direction,
                new_cross, state['act_mask']
            )
            act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
            agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)
            correction_applied = agg_mutation_info.get('correction_applied', False)
            if correction_applied:
                new_corrections += 1

            # Track discoveries
            for idx in agg_mutation_info.get('activated', []):
                if idx in EXTREME_AGGS:
                    new_discoveries['extreme'] += 1
                else:
                    new_discoveries['averaging'] += 1

            new_stagnation = 0

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'cross_affinity': new_cross,
            'act_activity': new_act_activity,
            'agg_activity': new_agg_activity,
            'extreme_ratio_history': extreme_ratio_history,
            'corrections_applied': new_corrections,
            'discoveries': new_discoveries,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
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
            'mutation_triggered': act_mutation_info is not None,
            # Homeostasis metrics
            'extreme_ratio': extreme_ratio,
            'imbalance': imbalance,
            'imbalance_direction': direction,
            'correction_applied': correction_applied,
            'total_corrections': new_corrections,
            # Discovery metrics
            'extreme_discoveries': new_discoveries['extreme'],
            'averaging_discoveries': new_discoveries['averaging'],
            # Cross-domain metrics
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            'sin_max_affinity': float(new_cross[4, 2]),
            'sin_min_affinity': float(new_cross[4, 3]),
            # Sin status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        if act_mutation_info:
            metrics['act_activated'] = act_mutation_info['activated']
            metrics['act_deactivated'] = act_mutation_info['deactivated']
        if agg_mutation_info:
            metrics['agg_activated'] = agg_mutation_info['activated']
            metrics['agg_deactivated'] = agg_mutation_info['deactivated']

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with homeostasis status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        extreme_ratio = self._compute_extreme_ratio(agg_palette)
        imbalance, direction = self._compute_imbalance(agg_palette)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'extreme_ratio': extreme_ratio,
            'imbalance': imbalance,
            'imbalance_direction': direction,
            'total_corrections': state['corrections_applied'],
            'extreme_discoveries': state['discoveries']['extreme'],
            'averaging_discoveries': state['discoveries']['averaging'],
            'sin_max_affinity': float(state['cross_affinity'][4, 2]),
            'sin_min_affinity': float(state['cross_affinity'][4, 3]),
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
