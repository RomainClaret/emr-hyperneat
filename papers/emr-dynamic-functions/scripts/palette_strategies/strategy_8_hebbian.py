"""Strategy 8: Hebbian Palette Discovery.

Bio-inspired learning: "Activations that fire together, stay together."

Tracks co-occurrence of activation functions in high-fitness networks and
uses Hebbian learning to strengthen useful pairs while weakening bad pairs.

Key mechanisms:
1. Hebbian update: w_ij += lr * (active_i * active_j * fitness_signal)
2. Anti-Hebbian: w_ij -= decay * (active_i * active_j * failure_signal)
3. Consolidation: Strong pairs (w_ij > threshold for N gens) become protected

Expected: 100% discovery, 70-80% solve (learns to avoid antagonism)
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


class HebbianStrategy(PaletteEvolutionStrategy):
    """Hebbian palette evolution with co-occurrence learning.

    Learns which activation pairs work well together:
    - High fitness + pair active → strengthen association
    - Low fitness + pair active → weaken association
    - Strong pairs persist; weak pairs get pruned

    Uses learned associations to guide mutation:
    - Activate functions that associate well with current active set
    - Deactivate functions with weak associations
    """

    name = "hebbian"
    description = "Hebbian co-occurrence learning for activation pairs"

    def __init__(
        self,
        # Hebbian learning parameters
        learning_rate: float = 0.1,
        anti_hebbian_rate: float = 0.05,
        consolidation_threshold: float = 0.7,
        consolidation_gens: int = 5,
        # Mutation parameters
        hebbian_influence: float = 0.5,
        base_activate_rate: float = 0.20,
        base_deactivate_rate: float = 0.10,
        # Constraints
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize strategy.

        Args:
            learning_rate: Hebbian weight update rate
            anti_hebbian_rate: Anti-Hebbian (failure) update rate
            consolidation_threshold: Weight threshold for "strong" pairs
            consolidation_gens: Gens of high weight to consolidate
            hebbian_influence: How much associations affect mutation rates
            base_activate_rate: Base activation probability
            base_deactivate_rate: Base deactivation probability
            min_active: Minimum active functions
            initial_palette: Starting palette indices
        """
        self.learning_rate = learning_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_gens = consolidation_gens
        self.hebbian_influence = hebbian_influence
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with Hebbian weight matrix."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize Hebbian weight matrix (symmetric)
        # w[i,j] = association strength between activations i and j
        # Start at 0.5 (neutral)
        hebbian_weights = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5

        # Consolidation tracking - how long each pair has been strong
        consolidation_counts = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.int32)

        # Protected pairs (consolidated)
        protected_pairs = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS), dtype=jnp.bool_)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 88888),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Hebbian state
            'hebbian_weights': hebbian_weights,
            'consolidation_counts': consolidation_counts,
            'protected_pairs': protected_pairs,
            # History for learning signal
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _hebbian_update(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Apply Hebbian/Anti-Hebbian update to weights.

        Args:
            weights: Current Hebbian weight matrix
            mask: Active activation mask
            fitness_signal: -1 to 1, positive = success, negative = failure

        Returns:
            Updated weight matrix
        """
        # Create outer product of active activations
        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        if fitness_signal >= 0:
            # Hebbian update: strengthen co-active pairs on success
            delta = self.learning_rate * fitness_signal * co_active
        else:
            # Anti-Hebbian: weaken co-active pairs on failure
            delta = self.anti_hebbian_rate * fitness_signal * co_active

        new_weights = weights + delta

        # Clamp to [0, 1] range
        new_weights = jnp.clip(new_weights, 0.0, 1.0)

        return new_weights

    def _update_consolidation(
        self,
        weights: jnp.ndarray,
        consolidation_counts: jnp.ndarray,
        protected_pairs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update consolidation tracking and protect strong pairs.

        Args:
            weights: Current Hebbian weights
            consolidation_counts: How long each pair has been strong
            protected_pairs: Already protected pairs

        Returns:
            Tuple of (updated_counts, updated_protected)
        """
        # Find currently strong pairs
        strong = weights >= self.consolidation_threshold

        # Increment counts for strong pairs, reset for weak
        new_counts = jnp.where(strong, consolidation_counts + 1, 0)

        # Protect pairs that have been strong for long enough
        newly_protected = new_counts >= self.consolidation_gens
        new_protected = jnp.logical_or(protected_pairs, newly_protected)

        return new_counts, new_protected

    def _compute_affinity_scores(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute affinity score for each activation based on current active set.

        High affinity = strong associations with currently active activations
        Low affinity = weak associations

        Args:
            weights: Hebbian weight matrix
            mask: Current active mask

        Returns:
            Array of affinity scores for each activation
        """
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        # Average association with all active functions
        affinities = jnp.dot(weights, active) / n_active

        return affinities

    def _mutate_palette_hebbian(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        protected_pairs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply Hebbian-guided mutation.

        High affinity inactive functions → more likely to activate
        Low affinity active functions → more likely to deactivate
        Protected pairs → cannot be broken

        Args:
            key: JAX random key
            mask: Current palette mask
            affinities: Affinity scores from Hebbian weights
            protected_pairs: Pairs that cannot be separated

        Returns:
            Tuple of (new_mask, mutation_info)
        """
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            affinity = float(affinities[i])

            if mask[i] < 0.5:
                # Inactive - might activate
                # Higher affinity → higher activation rate
                rate = self.base_activate_rate * (1 + self.hebbian_influence * (affinity - 0.5))
                rate = max(0.05, min(0.5, rate))

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                # Check if protected by any other active function
                is_protected = False
                for j in range(NUM_ACTIVATIONS):
                    if j != i and mask[j] > 0.5:
                        if protected_pairs[i, j] or protected_pairs[j, i]:
                            is_protected = True
                            break

                if is_protected:
                    continue

                # Lower affinity → higher deactivation rate
                rate = self.base_deactivate_rate * (1 + self.hebbian_influence * (0.5 - affinity))
                rate = max(0.01, min(0.3, rate))

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        mutation_info = {
            'activated': activated,
            'deactivated': deactivated,
            'avg_affinity': float(jnp.mean(affinities)),
        }

        return new_mask, mutation_info

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with Hebbian learning and guided mutation.

        1. Compute fitness signal (improvement/regression)
        2. Apply Hebbian update to weights
        3. Update consolidation tracking
        4. Apply Hebbian-guided mutation
        """
        key, subkey = jax.random.split(state['rng_key'])

        # Check improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Compute fitness signal for Hebbian update
        # Positive = success (fitness > baseline), Negative = failure
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 10:
            fitness_history = fitness_history[-10:]

        baseline = sum(fitness_history) / len(fitness_history)
        fitness_signal = (best_fitness - baseline) / max(0.1, baseline)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Apply Hebbian update
        new_weights = self._hebbian_update(
            state['hebbian_weights'],
            state['mask'],
            fitness_signal,
        )

        # Update consolidation
        new_counts, new_protected = self._update_consolidation(
            new_weights,
            state['consolidation_counts'],
            state['protected_pairs'],
        )

        # Compute affinities for mutation
        affinities = self._compute_affinity_scores(new_weights, state['mask'])

        # Apply Hebbian-guided mutation
        new_mask, mutation_info = self._mutate_palette_hebbian(
            subkey, state['mask'], affinities, new_protected
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Hebbian state
            'hebbian_weights': new_weights,
            'consolidation_counts': new_counts,
            'protected_pairs': new_protected,
            'fitness_history': fitness_history,
        }

        # Compute stats
        n_protected = int(jnp.sum(new_protected) / 2)  # Divide by 2 for symmetry
        avg_weight = float(jnp.mean(new_weights))
        strong_pairs = int(jnp.sum(new_weights >= self.consolidation_threshold) / 2)

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': mask_to_indices(new_mask),
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'fitness_signal': fitness_signal,
            # Hebbian stats
            'n_protected_pairs': n_protected,
            'n_strong_pairs': strong_pairs,
            'avg_hebbian_weight': avg_weight,
        }

        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary including Hebbian stats."""
        palette = self.get_active_palette(state)
        weights = state['hebbian_weights']
        protected = state['protected_pairs']

        # Find strongest pairs
        strongest_val = float(jnp.max(weights))
        strongest_idx = jnp.unravel_index(jnp.argmax(weights), weights.shape)

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'stagnation_count': state['stagnation_count'],
            'strongest_pair': (int(strongest_idx[0]), int(strongest_idx[1])),
            'strongest_weight': strongest_val,
            'n_protected': int(jnp.sum(protected) / 2),
        }
