"""Strategy 43: Opioid Neuromodulation (Reward-Driven Exploration Control).

Implements opioid neuromodulation principles for palette evolution. Success
releases opioids that reduce exploration, while low opioid levels increase
exploration. Tolerance develops with repeated success.

Biological Basis:
- Endogenous opioids (endorphins, enkephalins) modulate reward processing
- Opioid release on positive outcomes suppresses exploratory behavior
- Low opioid levels trigger seeking/exploration behavior
- Tolerance develops with repeated reward (diminishing returns)
- Withdrawal increases exploration drive
- Creates adaptive explore/exploit balance

Key Insight:
- Current strategies lack dynamic explore/exploit modulation
- Opioid system provides biologically-grounded balance mechanism
- Success → exploitation (protect what works)
- Failure → exploration (find something better)
- Tolerance prevents over-exploitation of stale solutions
- Withdrawal drives exploration after plateaus

Opioid Mechanism:
    # On fitness improvement: release opioids
    if fitness_improved:
        opioid_level += opioid_release_rate * (1 - tolerance)
        tolerance += tolerance_rate

    # Natural decay
    opioid_level *= (1 - opioid_decay_rate)
    tolerance *= (1 - tolerance_recovery_rate)

    # Modulate exploration based on opioid level
    effective_mutation_rate = base_mutation_rate * (1 - opioid_level * exploration_modulation)

    # Low opioid (withdrawal) → increased exploration
    if opioid_level < withdrawal_threshold:
        effective_mutation_rate *= withdrawal_exploration_boost

Expected improvements:
- Automatic explore/exploit balance based on performance
- Exploitation after success, exploration after stagnation
- Tolerance prevents getting stuck on local optima
- Withdrawal mechanism drives recovery from plateaus
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


class OpioidNeuromodulationStrategy(PaletteEvolutionStrategy):
    """Opioid-inspired explore/exploit balance for palette evolution.

    Implements reward-driven neuromodulation where success releases opioids
    that reduce exploration, and low opioid levels (withdrawal) increase
    exploration. Tolerance develops over repeated success.
    """

    name = "opioid_neuromodulation"
    description = "Opioid-based explore/exploit balance with tolerance and withdrawal"

    def __init__(
        self,
        # Opioid dynamics
        opioid_release_rate: float = 0.3,         # Opioid release on improvement
        opioid_decay_rate: float = 0.1,           # Natural opioid decay per gen
        max_opioid_level: float = 1.0,            # Maximum opioid saturation
        # Tolerance dynamics
        tolerance_rate: float = 0.05,             # Tolerance buildup rate
        tolerance_recovery_rate: float = 0.02,    # Tolerance recovery per gen
        max_tolerance: float = 0.9,               # Maximum tolerance level
        # Exploration modulation
        exploration_modulation: float = 0.5,      # How much opioids reduce exploration
        base_mutation_rate: float = 0.15,         # Base palette mutation rate
        min_mutation_rate: float = 0.02,          # Minimum mutation rate (even high opioid)
        # Withdrawal dynamics
        withdrawal_threshold: float = 0.15,       # Opioid level triggering withdrawal
        withdrawal_exploration_boost: float = 2.0,  # Exploration multiplier in withdrawal
        withdrawal_sensitivity_boost: float = 1.5,  # Sensitivity to new functions
        # Hedonic adaptation
        hedonic_baseline: float = 0.5,            # Baseline "pleasure" level
        hedonic_adaptation_rate: float = 0.03,    # How fast baseline adapts
        # Palette composition
        palette_size: int = 6,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Opioid Neuromodulation strategy.

        Args:
            opioid_release_rate: Amount of opioid released on fitness improvement
            opioid_decay_rate: Natural decay of opioid level per generation
            max_opioid_level: Maximum opioid saturation (1.0 = fully saturated)
            tolerance_rate: Rate of tolerance buildup on repeated success
            tolerance_recovery_rate: Rate of tolerance recovery when not improving
            max_tolerance: Maximum tolerance level
            exploration_modulation: How much high opioid levels reduce exploration
            base_mutation_rate: Base probability of palette mutation
            min_mutation_rate: Minimum mutation rate even at max opioid
            withdrawal_threshold: Opioid level below which withdrawal effects occur
            withdrawal_exploration_boost: Exploration multiplier during withdrawal
            withdrawal_sensitivity_boost: Increased sensitivity to new functions
            hedonic_baseline: Baseline pleasure level for comparison
            hedonic_adaptation_rate: How fast hedonic baseline adapts
            palette_size: Target palette size
        """
        # Opioid dynamics
        self.opioid_release_rate = opioid_release_rate
        self.opioid_decay_rate = opioid_decay_rate
        self.max_opioid_level = max_opioid_level

        # Tolerance
        self.tolerance_rate = tolerance_rate
        self.tolerance_recovery_rate = tolerance_recovery_rate
        self.max_tolerance = max_tolerance

        # Exploration
        self.exploration_modulation = exploration_modulation
        self.base_mutation_rate = base_mutation_rate
        self.min_mutation_rate = min_mutation_rate

        # Withdrawal
        self.withdrawal_threshold = withdrawal_threshold
        self.withdrawal_exploration_boost = withdrawal_exploration_boost
        self.withdrawal_sensitivity_boost = withdrawal_sensitivity_boost

        # Hedonic adaptation
        self.hedonic_baseline = hedonic_baseline
        self.hedonic_adaptation_rate = hedonic_adaptation_rate

        # Palette
        self.palette_size = palette_size
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with opioid neuromodulation tracking."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Per-function affinity tracking (how much each function is "liked")
        function_affinity = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                function_affinity = function_affinity.at[i].set(0.5)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 434343),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Opioid state
            'opioid_level': 0.0,                  # Current opioid level [0, 1]
            'tolerance': 0.0,                      # Tolerance level [0, 1]
            'hedonic_baseline': self.hedonic_baseline,  # Adaptive baseline
            'in_withdrawal': False,                # Currently in withdrawal
            'withdrawal_duration': 0,              # Generations in withdrawal
            # Function tracking
            'function_affinity': function_affinity,  # Per-function preference
            'function_exposure': jnp.zeros(NUM_ACTIVATIONS),  # How much each function used
            # History
            'opioid_history': [],                  # Track opioid levels
            'tolerance_history': [],               # Track tolerance levels
            'exploration_history': [],             # Track effective exploration rate
            'previous_mask': mask,
            'fitness_history': [],
            # Stats
            'total_rewards': 0,                    # Times fitness improved
            'total_withdrawal_episodes': 0,        # Times entered withdrawal
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette."""
        return mask_to_indices(state['mask'])

    def _update_opioid_system(
        self,
        opioid_level: float,
        tolerance: float,
        hedonic_baseline: float,
        improved: bool,
        fitness_delta: float,
    ) -> Tuple[float, float, float]:
        """Update opioid system based on fitness outcome."""
        new_opioid = opioid_level
        new_tolerance = tolerance
        new_baseline = hedonic_baseline

        if improved:
            # Calculate effective reward (affected by tolerance)
            effective_rate = self.opioid_release_rate * (1 - tolerance)

            # Release opioids proportional to improvement
            reward_magnitude = min(fitness_delta * 2, 1.0)  # Scale and cap
            opioid_release = effective_rate * reward_magnitude

            new_opioid = min(new_opioid + opioid_release, self.max_opioid_level)

            # Build tolerance
            new_tolerance = min(
                new_tolerance + self.tolerance_rate,
                self.max_tolerance
            )

            # Hedonic adaptation - baseline shifts toward current level
            new_baseline = new_baseline + self.hedonic_adaptation_rate * (new_opioid - new_baseline)
        else:
            # Tolerance recovery when not improving
            new_tolerance = max(
                new_tolerance - self.tolerance_recovery_rate,
                0.0
            )

        # Natural opioid decay
        new_opioid = new_opioid * (1 - self.opioid_decay_rate)

        return new_opioid, new_tolerance, new_baseline

    def _compute_effective_mutation_rate(
        self,
        opioid_level: float,
        in_withdrawal: bool,
    ) -> float:
        """Compute effective mutation rate based on opioid state."""
        # Base modulation: high opioid → low exploration
        modulated_rate = self.base_mutation_rate * (
            1 - opioid_level * self.exploration_modulation
        )

        # Enforce minimum
        modulated_rate = max(modulated_rate, self.min_mutation_rate)

        # Withdrawal boost
        if in_withdrawal:
            modulated_rate *= self.withdrawal_exploration_boost

        return min(modulated_rate, 1.0)  # Cap at 100%

    def _mutate_palette(
        self,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        mutation_rate: float,
        in_withdrawal: bool,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Mutate palette based on current exploration rate."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()
        new_affinity = affinity.copy()

        current_palette = mask_to_indices(mask)

        # Should we mutate?
        if jax.random.uniform(key1) < mutation_rate:
            # Remove a function (weighted by inverse affinity)
            if len(current_palette) > self.min_active:
                # Calculate removal probabilities (lower affinity → higher removal prob)
                removal_weights = []
                for i in current_palette:
                    # Inverse affinity, but withdrawal makes us more likely to try new things
                    weight = 1.0 / (float(affinity[i]) + 0.1)
                    if in_withdrawal:
                        weight *= self.withdrawal_sensitivity_boost
                    removal_weights.append(weight)

                # Normalize
                total_weight = sum(removal_weights)
                removal_probs = [w / total_weight for w in removal_weights]

                # Sample removal
                cum_prob = 0
                sample = float(jax.random.uniform(key2))
                remove_idx = 0
                for i, prob in enumerate(removal_probs):
                    cum_prob += prob
                    if sample < cum_prob:
                        remove_idx = i
                        break

                removed = current_palette[remove_idx]
                new_mask = new_mask.at[removed].set(0.0)
                # Decay affinity for removed function
                new_affinity = new_affinity.at[removed].multiply(0.8)

            # Add a new function
            available = [i for i in range(NUM_ACTIVATIONS) if new_mask[i] < 0.5]
            if available:
                # In withdrawal: more open to novel functions
                if in_withdrawal:
                    # Weighted toward unexplored functions
                    add_weights = []
                    for i in available:
                        # Lower exposure → higher add probability
                        weight = 1.0 / (float(new_affinity[i]) + 0.1)
                        add_weights.append(weight)

                    total_weight = sum(add_weights)
                    add_probs = [w / total_weight for w in add_weights]

                    cum_prob = 0
                    sample = float(jax.random.uniform(key3))
                    add_idx = 0
                    for i, prob in enumerate(add_probs):
                        cum_prob += prob
                        if sample < cum_prob:
                            add_idx = i
                            break

                    added = available[add_idx]
                else:
                    # Normal: random selection
                    add_idx = int(jax.random.randint(key3, (), 0, len(available)))
                    added = available[add_idx]

                new_mask = new_mask.at[added].set(1.0)
                # Initial affinity for new function
                new_affinity = new_affinity.at[added].set(0.3)

        return new_mask, new_affinity

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        exposure: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        opioid_level: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update function affinity based on outcomes."""
        new_affinity = affinity.copy()
        new_exposure = exposure.copy()

        # Increase exposure for active functions
        for i in range(NUM_ACTIVATIONS):
            if mask[i] > 0.5:
                new_exposure = new_exposure.at[i].add(1.0)

                if improved:
                    # Reward association: opioid enhances affinity
                    boost = 0.1 * (1 + opioid_level)
                    new_affinity = new_affinity.at[i].add(boost)

        # Decay all affinities slightly (use it or lose it)
        new_affinity = new_affinity * 0.99
        new_affinity = jnp.clip(new_affinity, 0, 2.0)

        return new_affinity, new_exposure

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with opioid neuromodulation dynamics."""
        key, k1 = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness if improved else 0.0

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Update opioid system
        new_opioid, new_tolerance, new_baseline = self._update_opioid_system(
            state['opioid_level'],
            state['tolerance'],
            state['hedonic_baseline'],
            improved,
            fitness_delta,
        )

        # Step 2: Check for withdrawal
        was_in_withdrawal = state['in_withdrawal']
        in_withdrawal = new_opioid < self.withdrawal_threshold
        withdrawal_duration = state['withdrawal_duration']

        if in_withdrawal:
            withdrawal_duration += 1
            if not was_in_withdrawal:
                # Just entered withdrawal
                withdrawal_episodes = state['total_withdrawal_episodes'] + 1
            else:
                withdrawal_episodes = state['total_withdrawal_episodes']
        else:
            withdrawal_duration = 0
            withdrawal_episodes = state['total_withdrawal_episodes']

        # Step 3: Compute effective mutation rate
        effective_mutation_rate = self._compute_effective_mutation_rate(
            new_opioid, in_withdrawal
        )

        # Step 4: Mutate palette based on exploration rate
        new_mask, new_affinity = self._mutate_palette(
            state['mask'],
            state['function_affinity'],
            effective_mutation_rate,
            in_withdrawal,
            k1,
        )

        # Step 5: Update function affinity
        new_affinity, new_exposure = self._update_affinity(
            new_affinity,
            state['function_exposure'],
            new_mask,
            improved,
            new_opioid,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update histories
        opioid_history = state['opioid_history'] + [new_opioid]
        if len(opioid_history) > 50:
            opioid_history = opioid_history[-50:]

        tolerance_history = state['tolerance_history'] + [new_tolerance]
        if len(tolerance_history) > 50:
            tolerance_history = tolerance_history[-50:]

        exploration_history = state['exploration_history'] + [effective_mutation_rate]
        if len(exploration_history) > 50:
            exploration_history = exploration_history[-50:]

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
            # Opioid state
            'opioid_level': new_opioid,
            'tolerance': new_tolerance,
            'hedonic_baseline': new_baseline,
            'in_withdrawal': in_withdrawal,
            'withdrawal_duration': withdrawal_duration,
            # Function tracking
            'function_affinity': new_affinity,
            'function_exposure': new_exposure,
            # History
            'opioid_history': opioid_history,
            'tolerance_history': tolerance_history,
            'exploration_history': exploration_history,
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
            # Stats
            'total_rewards': state['total_rewards'] + (1 if improved else 0),
            'total_withdrawal_episodes': withdrawal_episodes,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Top functions by affinity
        top_affinity_idx = jnp.argsort(new_affinity)[-5:][::-1]
        top_affinity = [(int(i), float(new_affinity[i])) for i in top_affinity_idx]

        # Opioid trend
        recent_opioid = opioid_history[-10:] if len(opioid_history) >= 10 else opioid_history
        opioid_trend = 'rising' if len(recent_opioid) >= 2 and recent_opioid[-1] > recent_opioid[0] else 'falling'

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Opioid state
            'opioid_level': new_opioid,
            'tolerance': new_tolerance,
            'hedonic_baseline': new_baseline,
            'effective_mutation_rate': effective_mutation_rate,
            'opioid_trend': opioid_trend,
            # Withdrawal
            'in_withdrawal': in_withdrawal,
            'withdrawal_duration': withdrawal_duration,
            'just_entered_withdrawal': in_withdrawal and not was_in_withdrawal,
            'just_exited_withdrawal': was_in_withdrawal and not in_withdrawal,
            # Stats
            'total_rewards': new_state['total_rewards'],
            'total_withdrawal_episodes': withdrawal_episodes,
            'reward_rate': new_state['total_rewards'] / max(generation + 1, 1),
            # Function tracking
            'top_affinity': top_affinity,
            'mean_affinity': float(jnp.mean(new_affinity)),
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_affinity': float(new_affinity[4]),
            'sin_exposure': float(new_exposure[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with opioid status."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']
        exposure = state['function_exposure']

        # Top by affinity
        top_idx = jnp.argsort(affinity)[-5:][::-1]
        top_affinity = [(int(i), float(affinity[i])) for i in top_idx]

        # Top by exposure
        top_exposure_idx = jnp.argsort(exposure)[-5:][::-1]
        top_exposure = [(int(i), float(exposure[i])) for i in top_exposure_idx]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Opioid state
            'opioid_level': state['opioid_level'],
            'tolerance': state['tolerance'],
            'hedonic_baseline': state['hedonic_baseline'],
            'in_withdrawal': state['in_withdrawal'],
            'withdrawal_duration': state['withdrawal_duration'],
            # Stats
            'total_rewards': state['total_rewards'],
            'total_withdrawal_episodes': state['total_withdrawal_episodes'],
            # Function tracking
            'top_affinity': top_affinity,
            'top_exposure': top_exposure,
            # Sin-specific
            'sin_affinity': float(affinity[4]),
            'sin_exposure': float(exposure[4]),
        }
