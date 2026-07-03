"""Strategy 8 Dual: Hebbian Palette Discovery for Both Activation AND Aggregation.

Extends Hebbian to jointly evolve both activation and aggregation palettes.

Bio-inspired learning: "Functions that fire together, stay together."

Tracks co-occurrence of functions in high-fitness networks and uses Hebbian
learning to strengthen useful pairs while weakening bad pairs - for BOTH domains.

Key innovation: Cross-domain Hebbian learning tracks which activation-aggregation
combinations succeed together, influencing protection for both palettes.

Biological analogy:
- Hebbian plasticity: Correlation-based strengthening across all circuit types
- Cross-modal binding: Visual + auditory features that co-occur become linked
"""

from typing import Dict, Any, List, Optional, Tuple
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


class HebbianDualStrategy(PaletteEvolutionStrategy):
    """Hebbian co-occurrence learning for BOTH activation and aggregation palettes.

    Learns which function pairs work well together within each domain:
    - High fitness + pair active → strengthen association
    - Low fitness + pair active → weaken association
    - Strong pairs persist; weak pairs get pruned

    Also learns cross-domain associations:
    - Which activation-aggregation combinations correlate with success
    - Cross-domain affinity influences protection scores
    """

    name = "hebbian_dual"
    description = "Dual palette Hebbian co-occurrence learning with cross-domain"

    def __init__(
        self,
        # Hebbian learning parameters
        learning_rate: float = 0.1,
        anti_hebbian_rate: float = 0.05,
        consolidation_threshold: float = 0.7,
        consolidation_gens: int = 5,
        # Cross-domain learning
        cross_learning_rate: float = 0.08,
        cross_influence: float = 0.3,
        # Mutation parameters
        hebbian_influence: float = 0.5,
        base_activate_rate: float = 0.20,
        base_deactivate_rate: float = 0.10,
        # Aggregation-specific rates
        agg_base_activate_rate: float = 0.15,
        agg_base_deactivate_rate: float = 0.08,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,   # Optimal is 6, >6 causes antagonism
        max_active_agg: int = 4,   # Optimal is 4 for parity
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            learning_rate: Hebbian weight update rate
            anti_hebbian_rate: Anti-Hebbian (failure) update rate
            consolidation_threshold: Weight threshold for "strong" pairs
            consolidation_gens: Gens of high weight to consolidate
            cross_learning_rate: Learning rate for cross-domain associations
            cross_influence: How much cross-domain affects protection (0-1)
            hebbian_influence: How much associations affect mutation rates
            base_activate_rate: Base activation probability (activations)
            base_deactivate_rate: Base deactivation probability (activations)
            agg_base_activate_rate: Base activation probability (aggregations)
            agg_base_deactivate_rate: Base deactivation probability (aggregations)
            min_active_act: Minimum active activation functions
            min_active_agg: Minimum active aggregation functions
            max_active_act: Maximum active activation functions (prevents antagonism)
            max_active_agg: Maximum active aggregation functions
            initial_act_palette: Starting activation palette indices
            initial_agg_palette: Starting aggregation palette indices
        """
        self.learning_rate = learning_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_gens = consolidation_gens
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        self.hebbian_influence = hebbian_influence
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.agg_base_activate_rate = agg_base_activate_rate
        self.agg_base_deactivate_rate = agg_base_deactivate_rate
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual Hebbian matrices."""
        # Activation palette
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)

        # Aggregation palette
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Hebbian weight matrices - activation domain (symmetric)
        act_hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5
        act_consolidation_counts = jnp.zeros(
            (NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.int32
        )
        act_protected_pairs = jnp.zeros(
            (NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.bool_
        )

        # Hebbian weight matrices - aggregation domain (symmetric)
        agg_hebbian_weights = jnp.ones((NUM_AGGREGATIONS, NUM_AGGREGATIONS)) * 0.5
        agg_consolidation_counts = jnp.zeros(
            (NUM_AGGREGATIONS, NUM_AGGREGATIONS), dtype=jnp.int32
        )
        agg_protected_pairs = jnp.zeros(
            (NUM_AGGREGATIONS, NUM_AGGREGATIONS), dtype=jnp.bool_
        )

        # Cross-domain affinity matrix (NEW)
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation state
            'act_mask': act_mask,
            'act_hebbian_weights': act_hebbian_weights,
            'act_consolidation_counts': act_consolidation_counts,
            'act_protected_pairs': act_protected_pairs,
            # Aggregation state
            'agg_mask': agg_mask,
            'agg_hebbian_weights': agg_hebbian_weights,
            'agg_consolidation_counts': agg_consolidation_counts,
            'agg_protected_pairs': agg_protected_pairs,
            # Cross-domain state
            'cross_affinity': cross_affinity,
            # Common state
            'rng_key': jax.random.PRNGKey(seed + 88888),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Apply Hebbian/Anti-Hebbian update to weights."""
        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        if fitness_signal >= 0:
            delta = self.learning_rate * fitness_signal * co_active
        else:
            delta = self.anti_hebbian_rate * fitness_signal * co_active

        new_weights = jnp.clip(weights + delta, 0.0, 1.0)
        return new_weights

    def _hebbian_update_cross(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Apply Hebbian update to cross-domain affinity."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)

        if fitness_signal >= 0:
            delta = self.cross_learning_rate * fitness_signal * cross_active
        else:
            delta = (self.cross_learning_rate * 0.5) * fitness_signal * cross_active

        new_cross = jnp.clip(cross_affinity + delta, 0.0, 1.0)
        return new_cross

    def _update_consolidation(
        self,
        weights: jnp.ndarray,
        counts: jnp.ndarray,
        protected: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update consolidation tracking and protect strong pairs."""
        strong = weights >= self.consolidation_threshold
        new_counts = jnp.where(strong, counts + 1, 0)
        newly_protected = new_counts >= self.consolidation_gens
        new_protected = jnp.logical_or(protected, newly_protected)
        return new_counts, new_protected

    def _compute_affinity_scores(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        other_mask: jnp.ndarray,
        is_activation: bool = True,
    ) -> jnp.ndarray:
        """Compute affinity score including cross-domain influence."""
        active = (mask > 0.5).astype(jnp.float32)
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)
        n_other_active = max(jnp.sum(other_active), 1)

        # Pairwise affinity within domain
        pairwise_score = jnp.dot(weights, active) / n_active

        # Cross-domain affinity
        if is_activation:
            cross_score = jnp.dot(cross_affinity, other_active) / n_other_active
        else:
            cross_score = jnp.dot(cross_affinity.T, other_active) / n_other_active

        # Combine: pairwise (70%) + cross-domain (30%)
        affinities = (
            (1 - self.cross_influence) * pairwise_score +
            self.cross_influence * cross_score
        )

        return affinities

    def _mutate_palette_hebbian(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        protected_pairs: jnp.ndarray,
        base_activate: float,
        base_deactivate: float,
        min_active: int,
        max_active: int,
        n_functions: int,
    ) -> Tuple[jnp.ndarray, List[int], List[int]]:
        """Apply Hebbian-guided mutation to a palette.

        Key constraint: max_active prevents antagonism (too many functions hurts).
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (n_functions,))
        deactivate_probs = jax.random.uniform(key2, (n_functions,))

        # Track current active count for max constraint
        current_active = int(jnp.sum(mask > 0.5))

        for i in range(n_functions):
            affinity = float(affinities[i])

            if mask[i] < 0.5:
                # Inactive - might activate
                # CRITICAL: Skip if already at max
                if current_active + len(activated) >= max_active:
                    continue

                rate = base_activate * (1 + self.hebbian_influence * (affinity - 0.5))
                rate = max(0.05, min(0.5, rate))

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                # Check if protected by any other active function
                is_protected = False
                for j in range(n_functions):
                    if j != i and mask[j] > 0.5:
                        if protected_pairs[i, j] or protected_pairs[j, i]:
                            is_protected = True
                            break

                if is_protected:
                    continue

                # Lower affinity → higher deactivation rate
                rate = base_deactivate * (1 + self.hebbian_influence * (0.5 - affinity))
                rate = max(0.01, min(0.3, rate))

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, activated, deactivated

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual Hebbian learning and guided mutation."""
        key, key_act, key_agg = jax.random.split(state['rng_key'], 3)

        # Check improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Compute fitness signal for Hebbian update
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 10:
            fitness_history = fitness_history[-10:]

        baseline = sum(fitness_history) / len(fitness_history)
        fitness_signal = (best_fitness - baseline) / max(0.1, baseline)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # --- Hebbian updates for both domains ---
        new_act_weights = self._hebbian_update(
            state['act_hebbian_weights'], state['act_mask'], fitness_signal
        )
        new_agg_weights = self._hebbian_update(
            state['agg_hebbian_weights'], state['agg_mask'], fitness_signal
        )
        new_cross_affinity = self._hebbian_update_cross(
            state['cross_affinity'],
            state['act_mask'],
            state['agg_mask'],
            fitness_signal,
        )

        # --- Consolidation updates ---
        new_act_counts, new_act_protected = self._update_consolidation(
            new_act_weights,
            state['act_consolidation_counts'],
            state['act_protected_pairs'],
        )
        new_agg_counts, new_agg_protected = self._update_consolidation(
            new_agg_weights,
            state['agg_consolidation_counts'],
            state['agg_protected_pairs'],
        )

        # --- Compute affinities with cross-domain influence ---
        act_affinities = self._compute_affinity_scores(
            new_act_weights, state['act_mask'],
            new_cross_affinity, state['agg_mask'],
            is_activation=True,
        )
        agg_affinities = self._compute_affinity_scores(
            new_agg_weights, state['agg_mask'],
            new_cross_affinity, state['act_mask'],
            is_activation=False,
        )

        # --- Apply mutations to both palettes ---
        new_act_mask, act_activated, act_deactivated = self._mutate_palette_hebbian(
            key_act, state['act_mask'], act_affinities, new_act_protected,
            self.base_activate_rate, self.base_deactivate_rate,
            self.min_active_act, self.max_active_act, NUM_ACTIVATIONS,
        )
        new_agg_mask, agg_activated, agg_deactivated = self._mutate_palette_hebbian(
            key_agg, state['agg_mask'], agg_affinities, new_agg_protected,
            self.agg_base_activate_rate, self.agg_base_deactivate_rate,
            self.min_active_agg, self.max_active_agg, NUM_AGGREGATIONS,
        )

        act_palette_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_palette_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        new_state = {
            'act_mask': new_act_mask,
            'act_hebbian_weights': new_act_weights,
            'act_consolidation_counts': new_act_counts,
            'act_protected_pairs': new_act_protected,
            'agg_mask': new_agg_mask,
            'agg_hebbian_weights': new_agg_weights,
            'agg_consolidation_counts': new_agg_counts,
            'agg_protected_pairs': new_agg_protected,
            'cross_affinity': new_cross_affinity,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        # Compute stats
        n_act_protected = int(jnp.sum(new_act_protected) / 2)
        n_agg_protected = int(jnp.sum(new_agg_protected) / 2)
        act_strong_pairs = int(
            jnp.sum(new_act_weights >= self.consolidation_threshold) / 2
        )
        agg_strong_pairs = int(
            jnp.sum(new_agg_weights >= self.consolidation_threshold) / 2
        )

        metrics = {
            'act_palette_changed': act_palette_changed,
            'agg_palette_changed': agg_palette_changed,
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'fitness_signal': fitness_signal,
            # Activation Hebbian stats
            'n_act_protected_pairs': n_act_protected,
            'n_act_strong_pairs': act_strong_pairs,
            'avg_act_hebbian_weight': float(jnp.mean(new_act_weights)),
            'act_avg_affinity': float(jnp.mean(act_affinities)),
            # Aggregation Hebbian stats
            'n_agg_protected_pairs': n_agg_protected,
            'n_agg_strong_pairs': agg_strong_pairs,
            'avg_agg_hebbian_weight': float(jnp.mean(new_agg_weights)),
            'agg_avg_affinity': float(jnp.mean(agg_affinities)),
            # Cross-domain stats
            'cross_avg_affinity': float(jnp.mean(new_cross_affinity)),
            'cross_max_affinity': float(jnp.max(new_cross_affinity)),
            # Mutation events
            'act_activated': act_activated,
            'act_deactivated': act_deactivated,
            'agg_activated': agg_activated,
            'agg_deactivated': agg_deactivated,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including dual Hebbian stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        act_weights = state['act_hebbian_weights']
        agg_weights = state['agg_hebbian_weights']

        # Find strongest activation pair
        act_strongest_val = float(jnp.max(act_weights))
        act_strongest_idx = jnp.unravel_index(jnp.argmax(act_weights), act_weights.shape)

        # Find strongest aggregation pair
        agg_strongest_val = float(jnp.max(agg_weights))
        agg_strongest_idx = jnp.unravel_index(jnp.argmax(agg_weights), agg_weights.shape)

        # Find strongest cross-domain pair
        cross_aff = state['cross_affinity']
        cross_strongest_val = float(jnp.max(cross_aff))
        cross_strongest_idx = jnp.unravel_index(jnp.argmax(cross_aff), cross_aff.shape)

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'act_palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'stagnation_count': state['stagnation_count'],
            # Strongest pairs
            'act_strongest_pair': (int(act_strongest_idx[0]), int(act_strongest_idx[1])),
            'act_strongest_weight': act_strongest_val,
            'agg_strongest_pair': (int(agg_strongest_idx[0]), int(agg_strongest_idx[1])),
            'agg_strongest_weight': agg_strongest_val,
            'cross_strongest_pair': (int(cross_strongest_idx[0]), int(cross_strongest_idx[1])),
            'cross_strongest_weight': cross_strongest_val,
            # Protected counts
            'n_act_protected': int(jnp.sum(state['act_protected_pairs']) / 2),
            'n_agg_protected': int(jnp.sum(state['agg_protected_pairs']) / 2),
        }
