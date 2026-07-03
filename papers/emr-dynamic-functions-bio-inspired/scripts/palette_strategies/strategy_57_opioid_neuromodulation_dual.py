"""Strategy 57D: Opioid Neuromodulation Dual (Reward-Driven Control for Both Domains).

Extends OpioidNeuromodulationStrategy to jointly evolve BOTH activation AND aggregation
function palettes using opioid-based explore/exploit balance.

Key dual mechanisms:
1. Shared opioid system - single reward system affects both domains
2. Dual function affinity - separate preference tracking per domain
3. Coordinated withdrawal - exploration boost applies to both domains
4. Cross-domain tolerance - tolerance builds from success in either domain

Expected: Unified explore/exploit balance across both domains
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


class OpioidNeuromodulationDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution with opioid-based explore/exploit control.

    A shared opioid system modulates exploration in both activation and
    aggregation domains. Success releases opioids (exploitation), withdrawal
    triggers exploration in both domains.
    """

    name = "opioid_neuromodulation_dual"
    description = "Dual: Opioid-based explore/exploit balance for both domains"

    def __init__(
        self,
        # Opioid dynamics
        opioid_release_rate: float = 0.3,
        opioid_decay_rate: float = 0.1,
        max_opioid_level: float = 1.0,
        # Tolerance
        tolerance_rate: float = 0.05,
        tolerance_recovery_rate: float = 0.02,
        max_tolerance: float = 0.9,
        # Exploration modulation
        exploration_modulation: float = 0.5,
        base_mutation_rate: float = 0.15,
        min_mutation_rate: float = 0.02,
        # Withdrawal
        withdrawal_threshold: float = 0.15,
        withdrawal_exploration_boost: float = 2.0,
        withdrawal_sensitivity_boost: float = 1.5,
        # Hedonic adaptation
        hedonic_baseline: float = 0.5,
        hedonic_adaptation_rate: float = 0.03,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Opioid Neuromodulation Dual strategy."""
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

        # Hedonic
        self.hedonic_baseline = hedonic_baseline
        self.hedonic_adaptation_rate = hedonic_adaptation_rate

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual affinity tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize affinity for both domains
        act_affinity = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinity = act_affinity.at[i].set(0.5)

        agg_affinity = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinity = agg_affinity.at[i].set(0.5)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_affinity': act_affinity,
            'act_exposure': jnp.zeros(NUM_ACTIVATIONS),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_affinity': agg_affinity,
            'agg_exposure': jnp.zeros(NUM_AGGREGATIONS),
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Shared opioid system
            'opioid_level': 0.0,
            'tolerance': 0.0,
            'hedonic_baseline': self.hedonic_baseline,
            'in_withdrawal': False,
            'withdrawal_duration': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 575757),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            # Stats
            'total_rewards': 0,
            'total_withdrawal_episodes': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_opioid_system(
        self,
        opioid_level: float,
        tolerance: float,
        hedonic_baseline: float,
        improved: bool,
        fitness_delta: float,
    ) -> Tuple[float, float, float]:
        """Update shared opioid system."""
        new_opioid = opioid_level
        new_tolerance = tolerance
        new_baseline = hedonic_baseline

        if improved:
            effective_rate = self.opioid_release_rate * (1 - tolerance)
            reward_magnitude = min(fitness_delta * 2, 1.0)
            opioid_release = effective_rate * reward_magnitude
            new_opioid = min(new_opioid + opioid_release, self.max_opioid_level)
            new_tolerance = min(new_tolerance + self.tolerance_rate, self.max_tolerance)
            new_baseline = new_baseline + self.hedonic_adaptation_rate * (new_opioid - new_baseline)
        else:
            new_tolerance = max(new_tolerance - self.tolerance_recovery_rate, 0.0)

        new_opioid = new_opioid * (1 - self.opioid_decay_rate)
        return new_opioid, new_tolerance, new_baseline

    def _compute_effective_mutation_rate(
        self,
        opioid_level: float,
        in_withdrawal: bool,
    ) -> float:
        """Compute effective mutation rate."""
        modulated_rate = self.base_mutation_rate * (1 - opioid_level * self.exploration_modulation)
        modulated_rate = max(modulated_rate, self.min_mutation_rate)
        if in_withdrawal:
            modulated_rate *= self.withdrawal_exploration_boost
        return min(modulated_rate, 1.0)

    def _mutate_palette(
        self,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        mutation_rate: float,
        in_withdrawal: bool,
        key: jax.random.PRNGKey,
        min_active: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Mutate palette based on exploration rate."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_mask = mask.copy()
        new_affinity = affinity.copy()

        current_palette = mask_to_indices(mask)

        if jax.random.uniform(key1) < mutation_rate:
            # Remove a function
            if len(current_palette) > min_active:
                removal_weights = []
                for i in current_palette:
                    weight = 1.0 / (float(affinity[i]) + 0.1)
                    if in_withdrawal:
                        weight *= self.withdrawal_sensitivity_boost
                    removal_weights.append(weight)

                total_weight = sum(removal_weights)
                removal_probs = [w / total_weight for w in removal_weights]

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
                new_affinity = new_affinity.at[removed].multiply(0.8)

            # Add a function
            available = [i for i in range(n_funcs) if new_mask[i] < 0.5]
            if available:
                if in_withdrawal:
                    add_weights = [1.0 / (float(new_affinity[i]) + 0.1) for i in available]
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
                    add_idx = int(jax.random.randint(key3, (), 0, len(available)))
                    added = available[add_idx]

                new_mask = new_mask.at[added].set(1.0)
                new_affinity = new_affinity.at[added].set(0.3)

        return new_mask, new_affinity

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        exposure: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        opioid_level: float,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update function affinity and exposure."""
        new_affinity = affinity.copy()
        new_exposure = exposure.copy()

        for i in range(n_funcs):
            if mask[i] > 0.5:
                new_exposure = new_exposure.at[i].add(1.0)
                if improved:
                    boost = 0.1 * (1 + opioid_level)
                    new_affinity = new_affinity.at[i].add(boost)

        new_affinity = new_affinity * 0.99
        new_affinity = jnp.clip(new_affinity, 0, 2.0)

        return new_affinity, new_exposure

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improved: bool,
        opioid_level: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity on success."""
        if not improved:
            return cross_affinity * 0.99

        active_act = (act_mask > 0.5).astype(jnp.float32)
        active_agg = (agg_mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active_act, active_agg)
        boost = 0.1 * (1 + opioid_level)
        new_cross = cross_affinity + boost * co_active
        return jnp.clip(new_cross, 0.0, 2.0)

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual opioid neuromodulation."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness if improved else 0.0

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Update shared opioid system
        new_opioid, new_tolerance, new_baseline = self._update_opioid_system(
            state['opioid_level'], state['tolerance'],
            state['hedonic_baseline'], improved, fitness_delta
        )

        # Check withdrawal
        was_in_withdrawal = state['in_withdrawal']
        in_withdrawal = new_opioid < self.withdrawal_threshold
        withdrawal_duration = state['withdrawal_duration']
        withdrawal_episodes = state['total_withdrawal_episodes']

        if in_withdrawal:
            withdrawal_duration += 1
            if not was_in_withdrawal:
                withdrawal_episodes += 1
        else:
            withdrawal_duration = 0

        # Compute mutation rate
        mutation_rate = self._compute_effective_mutation_rate(new_opioid, in_withdrawal)

        # Mutate both palettes
        new_act_mask, new_act_affinity = self._mutate_palette(
            state['act_mask'], state['act_affinity'],
            mutation_rate, in_withdrawal, k_act,
            self.min_active_act, NUM_ACTIVATIONS
        )
        new_agg_mask, new_agg_affinity = self._mutate_palette(
            state['agg_mask'], state['agg_affinity'],
            mutation_rate, in_withdrawal, k_agg,
            self.min_active_agg, NUM_AGGREGATIONS
        )

        # Update affinity and exposure
        new_act_affinity, new_act_exposure = self._update_affinity(
            new_act_affinity, state['act_exposure'],
            new_act_mask, improved, new_opioid, NUM_ACTIVATIONS
        )
        new_agg_affinity, new_agg_exposure = self._update_affinity(
            new_agg_affinity, state['agg_exposure'],
            new_agg_mask, improved, new_opioid, NUM_AGGREGATIONS
        )

        # Update cross-domain affinity
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'],
            improved, new_opioid
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_affinity': new_act_affinity,
            'act_exposure': new_act_exposure,
            'agg_mask': new_agg_mask,
            'agg_affinity': new_agg_affinity,
            'agg_exposure': new_agg_exposure,
            'cross_affinity': new_cross,
            'opioid_level': new_opioid,
            'tolerance': new_tolerance,
            'hedonic_baseline': new_baseline,
            'in_withdrawal': in_withdrawal,
            'withdrawal_duration': withdrawal_duration,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
            'total_rewards': state['total_rewards'] + (1 if improved else 0),
            'total_withdrawal_episodes': withdrawal_episodes,
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
            # Opioid state
            'opioid_level': new_opioid,
            'tolerance': new_tolerance,
            'hedonic_baseline': new_baseline,
            'effective_mutation_rate': mutation_rate,
            # Withdrawal
            'in_withdrawal': in_withdrawal,
            'withdrawal_duration': withdrawal_duration,
            'just_entered_withdrawal': in_withdrawal and not was_in_withdrawal,
            'just_exited_withdrawal': was_in_withdrawal and not in_withdrawal,
            # Stats
            'total_rewards': new_state['total_rewards'],
            'total_withdrawal_episodes': withdrawal_episodes,
            # Affinity
            'act_mean_affinity': float(jnp.mean(new_act_affinity)),
            'agg_mean_affinity': float(jnp.mean(new_agg_affinity)),
            'cross_mean_affinity': float(jnp.mean(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_affinity': float(new_act_affinity[4]),
            'sin_exposure': float(new_act_exposure[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with opioid status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'opioid_level': state['opioid_level'],
            'tolerance': state['tolerance'],
            'in_withdrawal': state['in_withdrawal'],
            'withdrawal_duration': state['withdrawal_duration'],
            'total_rewards': state['total_rewards'],
            'total_withdrawal_episodes': state['total_withdrawal_episodes'],
            'act_mean_affinity': float(jnp.mean(state['act_affinity'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinity'])),
            'cross_mean_affinity': float(jnp.mean(state['cross_affinity'])),
        }
