"""Strategy 50D: Ant Colony Pheromone Dual (Stigmergy for Both Domains).

Extends AntColonyPheromoneStrategy to jointly evolve BOTH activation AND aggregation
function palettes using ant colony pheromone dynamics.

Key dual mechanisms:
1. Dual pheromone trails - separate trails for act and agg domains
2. Cross-domain pheromone - successful act-agg pairs boost each other
3. Parallel evaporation and deposit - independent dynamics in both domains
4. Coordinated exploration - trail following and random exploration in both

Expected: Emergent consensus through accumulated history in both domains
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


class AntColonyPheromoneDualStrategy(PaletteEvolutionStrategy):
    """Dual palette evolution using ant colony pheromone trails.

    Both activation and aggregation functions accumulate pheromone based on
    fitness contribution. Selection follows pheromone concentrations with
    exploration noise. Cross-domain pheromone tracks successful combinations.
    """

    name = "ant_colony_pheromone_dual"
    description = "Dual: Ant colony pheromone trails for both domains"

    def __init__(
        self,
        # Pheromone dynamics
        pheromone_decay: float = 0.85,
        pheromone_deposit: float = 0.3,
        pheromone_min: float = 0.05,
        pheromone_max: float = 3.0,
        initial_pheromone: float = 0.5,
        # Elite reinforcement
        elite_bonus: float = 2.0,
        elite_threshold: float = 0.9,
        # Selection parameters
        temperature: float = 0.5,
        exploration_rate: float = 0.15,
        follow_probability: float = 0.85,
        # Cross-domain
        cross_pheromone_rate: float = 0.08,
        cross_boost_factor: float = 0.15,
        # Palette composition
        act_palette_size: int = 6,
        agg_palette_size: int = 3,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Ant Colony Pheromone Dual strategy."""
        # Pheromone dynamics
        self.pheromone_decay = pheromone_decay
        self.pheromone_deposit = pheromone_deposit
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max
        self.initial_pheromone = initial_pheromone

        # Elite reinforcement
        self.elite_bonus = elite_bonus
        self.elite_threshold = elite_threshold

        # Selection
        self.temperature = temperature
        self.exploration_rate = exploration_rate
        self.follow_probability = follow_probability

        # Cross-domain
        self.cross_pheromone_rate = cross_pheromone_rate
        self.cross_boost_factor = cross_boost_factor

        # Palette
        self.act_palette_size = act_palette_size
        self.agg_palette_size = agg_palette_size
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual pheromone levels."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Initialize pheromone levels - activation
        act_pheromone = jnp.ones(NUM_ACTIVATIONS) * self.initial_pheromone
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_pheromone = act_pheromone.at[i].set(self.initial_pheromone * 1.5)

        # Initialize pheromone levels - aggregation
        agg_pheromone = jnp.ones(NUM_AGGREGATIONS) * self.initial_pheromone
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_pheromone = agg_pheromone.at[i].set(self.initial_pheromone * 1.5)

        # Cross-domain pheromone
        cross_pheromone = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * self.initial_pheromone

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_pheromone': act_pheromone,
            'act_deposits': jnp.zeros(NUM_ACTIVATIONS),
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_pheromone': agg_pheromone,
            'agg_deposits': jnp.zeros(NUM_AGGREGATIONS),
            # Cross-domain
            'cross_pheromone': cross_pheromone,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 505050),
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

    def _evaporate_pheromone(self, pheromone: jnp.ndarray) -> jnp.ndarray:
        """Apply pheromone evaporation."""
        new_pheromone = pheromone * self.pheromone_decay
        return jnp.clip(new_pheromone, self.pheromone_min, self.pheromone_max)

    def _deposit_pheromone(
        self,
        pheromone: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
        is_elite: bool,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Deposit pheromone on active functions."""
        deposits = jnp.zeros(n_funcs)

        if improvement > 0:
            base_deposit = self.pheromone_deposit * improvement
            if is_elite:
                base_deposit *= self.elite_bonus

            for i in range(n_funcs):
                if mask[i] > 0.5:
                    deposits = deposits.at[i].set(base_deposit)

        new_pheromone = pheromone + deposits
        new_pheromone = jnp.clip(new_pheromone, self.pheromone_min, self.pheromone_max)

        return new_pheromone, deposits

    def _select_palette(
        self,
        pheromone: jnp.ndarray,
        key: jax.random.PRNGKey,
        palette_size: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Select palette based on pheromone levels with exploration."""
        key1, key2 = jax.random.split(key)

        probs = jax.nn.softmax(pheromone / self.temperature)
        follow = jax.random.uniform(key1) < self.follow_probability

        if follow:
            top_k = jnp.argsort(probs)[-palette_size:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
        else:
            selected = set()
            remaining_key = key2
            for _ in range(palette_size):
                remaining_key, subkey = jax.random.split(remaining_key)
                noise = jax.random.uniform(subkey, (n_funcs,)) * self.exploration_rate
                noisy_probs = probs + noise
                noisy_probs = noisy_probs / jnp.sum(noisy_probs)
                sample = jax.random.choice(subkey, n_funcs, p=noisy_probs)
                selected.add(int(sample))

            mask = jnp.zeros(n_funcs)
            for idx in selected:
                mask = mask.at[idx].set(1.0)

        return mask

    def _update_cross_pheromone(
        self,
        cross_pheromone: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update cross-domain pheromone based on co-activation success."""
        # Evaporate
        new_cross = cross_pheromone * self.pheromone_decay

        if improvement > 0:
            active_act = (act_mask > 0.5).astype(jnp.float32)
            active_agg = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(active_act, active_agg)
            deposit = self.cross_pheromone_rate * improvement * co_active
            new_cross = new_cross + deposit

        return jnp.clip(new_cross, self.pheromone_min, self.pheromone_max)

    def _compute_cross_boost(
        self,
        cross_pheromone: jnp.ndarray,
        other_mask: jnp.ndarray,
        is_act: bool,
    ) -> jnp.ndarray:
        """Compute pheromone boost from cross-domain."""
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)

        if is_act:
            boost = jnp.dot(cross_pheromone, other_active) / n_other
        else:
            boost = jnp.dot(cross_pheromone.T, other_active) / n_other

        return boost * self.cross_boost_factor

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dual ant colony pheromone dynamics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        is_elite = best_fitness >= self.elite_threshold

        # Step 1: Evaporate pheromone in both domains
        new_act_pher = self._evaporate_pheromone(state['act_pheromone'])
        new_agg_pher = self._evaporate_pheromone(state['agg_pheromone'])

        # Step 2: Update cross-domain pheromone
        new_cross = self._update_cross_pheromone(
            state['cross_pheromone'],
            state['act_mask'],
            state['agg_mask'],
            improvement,
        )

        # Step 3: Compute cross-domain boosts
        act_cross_boost = self._compute_cross_boost(new_cross, state['agg_mask'], True)
        agg_cross_boost = self._compute_cross_boost(new_cross, state['act_mask'], False)

        # Add cross-domain boost to pheromone
        new_act_pher = jnp.clip(new_act_pher + act_cross_boost, self.pheromone_min, self.pheromone_max)
        new_agg_pher = jnp.clip(new_agg_pher + agg_cross_boost, self.pheromone_min, self.pheromone_max)

        # Step 4: Deposit pheromone on successful functions
        new_act_pher, act_deposits = self._deposit_pheromone(
            new_act_pher, state['act_mask'], improvement, is_elite, NUM_ACTIVATIONS
        )
        new_agg_pher, agg_deposits = self._deposit_pheromone(
            new_agg_pher, state['agg_mask'], improvement, is_elite, NUM_AGGREGATIONS
        )

        # Step 5: Select new palettes based on pheromone
        new_act_mask = self._select_palette(new_act_pher, k_act, self.act_palette_size, NUM_ACTIVATIONS)
        new_agg_mask = self._select_palette(new_agg_pher, k_agg, self.agg_palette_size, NUM_AGGREGATIONS)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_pheromone': new_act_pher,
            'act_deposits': act_deposits,
            'agg_mask': new_agg_mask,
            'agg_pheromone': new_agg_pher,
            'agg_deposits': agg_deposits,
            'cross_pheromone': new_cross,
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
            'is_elite': is_elite,
            # Pheromone stats - activation
            'act_mean_pheromone': float(jnp.mean(new_act_pher)),
            'act_max_pheromone': float(jnp.max(new_act_pher)),
            # Pheromone stats - aggregation
            'agg_mean_pheromone': float(jnp.mean(new_agg_pher)),
            'agg_max_pheromone': float(jnp.max(new_agg_pher)),
            # Cross-domain
            'cross_mean_pheromone': float(jnp.mean(new_cross)),
            'cross_max_pheromone': float(jnp.max(new_cross)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_pheromone': float(new_act_pher[4]),
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with dual pheromone status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': 4 in act_palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            'act_mean_pheromone': float(jnp.mean(state['act_pheromone'])),
            'agg_mean_pheromone': float(jnp.mean(state['agg_pheromone'])),
            'cross_mean_pheromone': float(jnp.mean(state['cross_pheromone'])),
            'sin_pheromone': float(state['act_pheromone'][4]),
        }
