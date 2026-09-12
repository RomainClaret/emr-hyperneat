"""Strategy 39: Anchor Inhibition (Probabilistic Protection).

Implements high-affinity parameter conservation for palette evolution. Functions
that contribute positively to fitness become "anchored" and are probabilistically
protected from removal, enabling natural retention of important discoveries.

Biological Basis:
- Important synapses develop high-affinity binding proteins
- LTP-stabilized synapses resist destabilization
- Synaptic tagging marks important connections for protection
- Protection is probabilistic, not absolute (allows adaptation)

Key Insight:
- Current strategies protect based on temporal patterns (Critical Period) or
  regulatory circuits (GRN), but not direct fitness contribution
- Anchor strength directly reflects a function's contribution to fitness
- P(keep) ∝ fitness_contribution → best functions naturally protected
- Simple, interpretable protection without complex circuits

Anchor Mechanism:
    # Estimate fitness contribution for each function
    contribution[f] = mean(fitness when f active) - mean(fitness when f inactive)

    # Update anchor strength based on contribution
    if contribution[f] > 0:
        anchor_strength[f] += learning_rate * contribution[f]
    else:
        anchor_strength[f] *= decay_rate

    # Selection with anchor protection
    for each function in current_palette:
        if anchor_strength[f] >= anchor_threshold:
            P(keep) = removal_resistance (e.g., 0.95)
        else:
            P(keep) = base_selection_prob

Expected improvements:
- Direct fitness-contribution-based protection
- Interpretable: anchored functions are provably useful
- Gradual protection building (no hard switches)
- Probabilistic: allows occasional exploration even of anchored functions
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


class AnchorInhibitionStrategy(PaletteEvolutionStrategy):
    """Probabilistic protection based on fitness contribution.

    Functions that contribute positively to fitness become anchored.
    Anchored functions have high probability of being retained.
    Protection strength reflects cumulative fitness contribution.
    """

    name = "anchor_inhibition"
    description = "Probabilistic protection proportional to fitness contribution"

    def __init__(
        self,
        # Anchor dynamics
        anchor_learning_rate: float = 0.15,     # How fast anchors form
        anchor_decay: float = 0.95,             # Decay when contribution is negative
        anchor_threshold: float = 0.6,          # Strength needed for protection
        anchor_max: float = 1.5,                # Maximum anchor strength
        # Protection parameters
        removal_resistance: float = 0.95,       # P(keep) for anchored functions
        base_selection_prob: float = 0.7,       # P(keep) for non-anchored functions
        # Contribution estimation
        contribution_window: int = 10,          # Generations to track for estimation
        min_samples: int = 3,                   # Minimum samples before estimating
        # Exploration
        exploration_rate: float = 0.1,          # Chance to add random function
        stagnation_exploration: int = 8,        # Stagnation before forced exploration
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Anchor Inhibition strategy.

        Args:
            anchor_learning_rate: Rate at which anchor strength increases
            anchor_decay: Multiplicative decay when contribution is negative
            anchor_threshold: Anchor strength required for protection
            anchor_max: Maximum anchor strength cap
            removal_resistance: Probability of keeping an anchored function
            base_selection_prob: Probability of keeping non-anchored function
            contribution_window: Number of generations for contribution estimation
            min_samples: Minimum observations before estimating contribution
            exploration_rate: Probability of adding random new function
            stagnation_exploration: Stagnation threshold for forced exploration
            palette_size: Target number of active functions
        """
        # Anchor dynamics
        self.anchor_learning_rate = anchor_learning_rate
        self.anchor_decay = anchor_decay
        self.anchor_threshold = anchor_threshold
        self.anchor_max = anchor_max

        # Protection
        self.removal_resistance = removal_resistance
        self.base_selection_prob = base_selection_prob

        # Contribution estimation
        self.contribution_window = contribution_window
        self.min_samples = min_samples

        # Exploration
        self.exploration_rate = exploration_rate
        self.stagnation_exploration = stagnation_exploration

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with anchor tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Initialize anchor strengths (initial palette gets small boost)
        anchor_strength = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                anchor_strength = anchor_strength.at[i].set(0.3)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 393939),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Anchor state
            'anchor_strength': anchor_strength,
            # Contribution tracking: lists of (generation, fitness, was_active_mask)
            'fitness_when_active': [[] for _ in range(NUM_ACTIVATIONS)],
            'fitness_when_inactive': [[] for _ in range(NUM_ACTIVATIONS)],
            'estimated_contribution': jnp.zeros(NUM_ACTIVATIONS),
            # History
            'previous_mask': mask,
            'fitness_history': [],
            # Tracking
            'anchor_events': [],  # List of (gen, func_idx, 'anchored'/'unanchored')
            'protection_count': jnp.zeros(NUM_ACTIVATIONS),  # How often each was protected
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_contribution_tracking(
        self,
        state: Dict[str, Any],
        generation: int,
        fitness: float,
    ) -> Tuple[List[List], List[List]]:
        """Update fitness tracking for contribution estimation."""
        mask = state['mask']
        fitness_when_active = [list(h) for h in state['fitness_when_active']]
        fitness_when_inactive = [list(h) for h in state['fitness_when_inactive']]

        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                fitness_when_active[i].append((generation, fitness))
            else:
                fitness_when_inactive[i].append((generation, fitness))

            # Trim to window
            if len(fitness_when_active[i]) > self.contribution_window:
                fitness_when_active[i] = fitness_when_active[i][-self.contribution_window:]
            if len(fitness_when_inactive[i]) > self.contribution_window:
                fitness_when_inactive[i] = fitness_when_inactive[i][-self.contribution_window:]

        return fitness_when_active, fitness_when_inactive

    def _estimate_contributions(
        self,
        fitness_when_active: List[List],
        fitness_when_inactive: List[List],
    ) -> jnp.ndarray:
        """Estimate contribution of each function to fitness."""
        contributions = jnp.zeros(NUM_ACTIVATIONS)

        for i in range(NUM_ACTIVATIONS):
            active_samples = fitness_when_active[i]
            inactive_samples = fitness_when_inactive[i]

            if len(active_samples) >= self.min_samples and len(inactive_samples) >= self.min_samples:
                # Mean fitness when active vs inactive
                mean_active = np.mean([f for _, f in active_samples])
                mean_inactive = np.mean([f for _, f in inactive_samples])
                contribution = mean_active - mean_inactive
                contributions = contributions.at[i].set(contribution)
            elif len(active_samples) >= self.min_samples:
                # Only active samples: use fitness level as proxy
                mean_active = np.mean([f for _, f in active_samples])
                contributions = contributions.at[i].set(mean_active * 0.5)
            # If neither, contribution stays at 0

        return contributions

    def _update_anchor_strength(
        self,
        anchor_strength: jnp.ndarray,
        contributions: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update anchor strengths based on contributions."""
        new_anchor = anchor_strength.copy()

        for i in range(NUM_ACTIVATIONS):
            contribution = float(contributions[i])

            if contribution > 0 and mask[i] > 0.5:
                # Positive contribution while active: strengthen anchor
                delta = self.anchor_learning_rate * contribution
                new_anchor = new_anchor.at[i].set(
                    min(float(anchor_strength[i]) + delta, self.anchor_max)
                )
            elif contribution < 0:
                # Negative contribution: decay anchor
                new_anchor = new_anchor.at[i].set(
                    float(anchor_strength[i]) * self.anchor_decay
                )
            else:
                # Neutral or inactive: slight decay
                new_anchor = new_anchor.at[i].set(
                    float(anchor_strength[i]) * 0.98
                )

        return new_anchor

    def _select_palette_with_anchors(
        self,
        anchor_strength: jnp.ndarray,
        contributions: jnp.ndarray,
        current_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
        stagnation: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select palette with anchor-based protection."""
        key1, key2, key3 = jax.random.split(key, 3)

        new_mask = jnp.zeros(NUM_ACTIVATIONS)
        protection_events = jnp.zeros(NUM_ACTIVATIONS)

        # First pass: keep anchored functions with high probability
        current_palette = mask_to_indices(current_mask)

        for i in current_palette:
            if anchor_strength[i] >= self.anchor_threshold:
                # Anchored: high keep probability
                if jax.random.uniform(key1) < self.removal_resistance:
                    new_mask = new_mask.at[i].set(1.0)
                    protection_events = protection_events.at[i].add(1)
                key1, _ = jax.random.split(key1)
            else:
                # Non-anchored: base selection probability
                if jax.random.uniform(key1) < self.base_selection_prob:
                    new_mask = new_mask.at[i].set(1.0)
                key1, _ = jax.random.split(key1)

        # Ensure minimum active
        n_active = int(jnp.sum(new_mask))
        if n_active < self.min_active:
            # Add top by anchor strength (or contribution if no anchor)
            priority = anchor_strength + contributions * 0.5
            inactive = jnp.where(new_mask < 0.5)[0]
            sorted_inactive = sorted(inactive.tolist(), key=lambda x: float(priority[x]), reverse=True)
            for idx in sorted_inactive[:self.min_active - n_active]:
                new_mask = new_mask.at[idx].set(1.0)

        # Limit to palette size
        n_active = int(jnp.sum(new_mask))
        if n_active > self.palette_size:
            # Keep top by anchor strength
            active = mask_to_indices(new_mask)
            sorted_active = sorted(active, key=lambda x: float(anchor_strength[x]), reverse=True)
            new_mask = jnp.zeros(NUM_ACTIVATIONS)
            for idx in sorted_active[:self.palette_size]:
                new_mask = new_mask.at[idx].set(1.0)

        # Exploration: add random function occasionally
        explore = (
            jax.random.uniform(key2) < self.exploration_rate or
            stagnation >= self.stagnation_exploration
        )
        if explore:
            # Pick from functions not currently in palette
            inactive = [i for i in range(NUM_ACTIVATIONS) if new_mask[i] < 0.5]
            if inactive:
                new_idx = inactive[int(jax.random.randint(key3, (), 0, len(inactive)))]
                # Replace lowest anchor if at capacity
                if int(jnp.sum(new_mask)) >= self.palette_size:
                    active = mask_to_indices(new_mask)
                    # Don't replace highly anchored functions
                    replaceable = [i for i in active if anchor_strength[i] < self.anchor_threshold]
                    if replaceable:
                        replace_idx = min(replaceable, key=lambda x: float(anchor_strength[x]))
                        new_mask = new_mask.at[replace_idx].set(0.0)
                        new_mask = new_mask.at[new_idx].set(1.0)
                else:
                    new_mask = new_mask.at[new_idx].set(1.0)

        return new_mask, protection_events

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with anchor-based protection."""
        key, k1 = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update contribution tracking
        fitness_when_active, fitness_when_inactive = self._update_contribution_tracking(
            state, generation, best_fitness
        )

        # Step 2: Estimate contributions
        contributions = self._estimate_contributions(fitness_when_active, fitness_when_inactive)

        # Step 3: Update anchor strengths
        new_anchor = self._update_anchor_strength(
            state['anchor_strength'],
            contributions,
            state['mask'],
        )

        # Step 4: Select palette with anchor protection
        new_mask, protection_events = self._select_palette_with_anchors(
            new_anchor,
            contributions,
            state['mask'],
            k1,
            new_stagnation,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track anchor events
        anchor_events = list(state['anchor_events'])
        for i in range(NUM_ACTIVATIONS):
            was_anchored = state['anchor_strength'][i] >= self.anchor_threshold
            now_anchored = new_anchor[i] >= self.anchor_threshold
            if now_anchored and not was_anchored:
                anchor_events.append((generation, i, 'anchored'))
            elif was_anchored and not now_anchored:
                anchor_events.append((generation, i, 'unanchored'))
        # Keep recent events
        if len(anchor_events) > 50:
            anchor_events = anchor_events[-50:]

        # Update protection count
        new_protection_count = state['protection_count'] + protection_events

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
            # Anchor state
            'anchor_strength': new_anchor,
            # Contribution tracking
            'fitness_when_active': fitness_when_active,
            'fitness_when_inactive': fitness_when_inactive,
            'estimated_contribution': contributions,
            # History
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Tracking
            'anchor_events': anchor_events,
            'protection_count': new_protection_count,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Anchored functions
        anchored = [i for i in range(NUM_ACTIVATIONS) if new_anchor[i] >= self.anchor_threshold]

        # Top by anchor strength
        top_anchor_idx = jnp.argsort(new_anchor)[-5:][::-1]
        top_anchors = [(int(i), float(new_anchor[i])) for i in top_anchor_idx]

        # Top by contribution
        top_contrib_idx = jnp.argsort(contributions)[-5:][::-1]
        top_contributions = [(int(i), float(contributions[i])) for i in top_contrib_idx]

        # Protection stats
        n_protected = int(jnp.sum(protection_events))

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Anchor stats
            'n_anchored': len(anchored),
            'anchored_functions': anchored,
            'mean_anchor_strength': float(jnp.mean(new_anchor)),
            'max_anchor_strength': float(jnp.max(new_anchor)),
            'top_anchors': top_anchors,
            # Contribution stats
            'top_contributions': top_contributions,
            'mean_contribution': float(jnp.mean(contributions)),
            # Protection
            'n_protected_this_gen': n_protected,
            'total_protection_events': int(jnp.sum(new_protection_count)),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_anchor_strength': float(new_anchor[4]),
            'sin_contribution': float(contributions[4]),
            'sin_is_anchored': 4 in anchored,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with anchor status."""
        palette = self.get_active_palette(state)
        anchor = state['anchor_strength']
        contributions = state['estimated_contribution']

        # Anchored functions
        anchored = [i for i in range(NUM_ACTIVATIONS) if anchor[i] >= self.anchor_threshold]

        # Top by anchor
        top_anchor = jnp.argsort(anchor)[-5:][::-1]
        top_anchors = [(int(i), float(anchor[i])) for i in top_anchor]

        # Top by contribution
        top_contrib = jnp.argsort(contributions)[-5:][::-1]
        top_contributions = [(int(i), float(contributions[i])) for i in top_contrib]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Anchors
            'n_anchored': len(anchored),
            'anchored_functions': anchored,
            'top_anchors': top_anchors,
            # Contributions
            'top_contributions': top_contributions,
            # Sin-specific
            'sin_anchor_strength': float(anchor[4]),
            'sin_contribution': float(contributions[4]),
            'sin_is_anchored': 4 in anchored,
            'sin_protection_count': int(state['protection_count'][4]),
        }
