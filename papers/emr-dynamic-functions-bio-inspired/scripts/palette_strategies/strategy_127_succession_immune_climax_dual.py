"""Strategy 127: Succession-Immune Climax Stabilization Dual.

Combines Ecological Succession climax (#26) with Immune Memory stabilization (#60).
Climax triggers synchronized memory consolidation burst.

Key Innovation:
- Climax detection: when fitness plateau is reached (stable high fitness)
- Climax triggers memory consolidation for ALL active functions
- Creates strong, synchronized protection for the winning configuration
- Post-climax mutations are heavily suppressed

Biological basis: In ecology, climax communities are stable endpoints.
In immune memory, successful responses are consolidated. Combining these:
when we reach a stable solution, lock it in with maximum protection.

Expected: Stable configurations that resist disruption after success.
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
)


class SuccessionImmuneClimaxDualStrategy(PaletteEvolutionStrategy):
    """Succession-immune climax stabilization for dual palette evolution.

    Climax detection triggers synchronized memory
    consolidation, creating strongly protected stable configurations.

    Critical innovation: Synchronized capture at climax for maximum stability.
    """

    name = "succession_immune_climax_dual"
    description = "Dual: Climax detection triggers synchronized memory consolidation"

    def __init__(
        self,
        # === Climax detection ===
        climax_detection_threshold: float = 0.95,
        climax_stability_window: int = 5,
        climax_improvement_tolerance: float = 0.01,
        # === Memory consolidation ===
        climax_memory_burst: bool = True,
        climax_protection_strength: float = 0.9,
        climax_mutation_suppression: float = 0.7,
        # === Pre-climax exploration ===
        exploration_mutation_rate: float = 0.12,
        post_climax_mutation_rate: float = 0.03,
        # === Memory parameters ===
        memory_decay: float = 0.98,
        base_memory_strength: float = 0.3,
        # === Sin preference ===
        sin_idx: int = 4,
        sin_memory_boost: float = 0.2,
        # === General parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Succession-Immune Climax strategy."""
        # Climax detection
        self.climax_detection_threshold = climax_detection_threshold
        self.climax_stability_window = climax_stability_window
        self.climax_improvement_tolerance = climax_improvement_tolerance

        # Memory consolidation
        self.climax_memory_burst = climax_memory_burst
        self.climax_protection_strength = climax_protection_strength
        self.climax_mutation_suppression = climax_mutation_suppression

        # Exploration
        self.exploration_mutation_rate = exploration_mutation_rate
        self.post_climax_mutation_rate = post_climax_mutation_rate

        # Memory
        self.memory_decay = memory_decay
        self.base_memory_strength = base_memory_strength

        # Sin
        self.sin_idx = sin_idx
        self.sin_memory_boost = sin_memory_boost

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with memory and climax tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.4

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.55)

        for i in CORE_EXTREME_AGGS:
            agg_affinities = agg_affinities.at[i].set(agg_affinities[i] + 0.2)

        # Memory (protection) per function
        act_memory = jnp.zeros(NUM_ACTIVATIONS)
        agg_memory = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Memory
            'act_memory': act_memory,
            'agg_memory': agg_memory,
            # Climax tracking
            'climax_reached': False,
            'climax_generation': -1,
            'stability_counter': 0,
            'memory_bursts': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1270000),
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

    def _detect_climax(self, fitness_history: List[float], current_fitness: float) -> bool:
        """Detect if we've reached climax (stable high fitness)."""
        if current_fitness < self.climax_detection_threshold:
            return False

        if len(fitness_history) < self.climax_stability_window:
            return False

        # Check if fitness has been stable
        recent = fitness_history[-self.climax_stability_window:]
        min_recent = min(recent)
        max_recent = max(recent)

        return (max_recent - min_recent) < self.climax_improvement_tolerance

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with climax-triggered memory consolidation."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        fitness_history = state['fitness_history'] + [best_fitness]
        climax_reached = state['climax_reached']
        climax_generation = state['climax_generation']
        stability_counter = state['stability_counter']
        memory_bursts = state['memory_bursts']

        # === CLIMAX DETECTION ===
        if not climax_reached:
            if self._detect_climax(fitness_history, best_fitness):
                climax_reached = True
                climax_generation = generation
                stability_counter = 0
            elif best_fitness >= self.climax_detection_threshold:
                stability_counter += 1
            else:
                stability_counter = 0

        # === UPDATE MEMORY ===
        act_memory = state['act_memory'] * self.memory_decay
        agg_memory = state['agg_memory'] * self.memory_decay

        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        act_affinities = state['act_affinities']
        agg_affinities = state['agg_affinities']

        # Memory consolidation burst at climax
        if climax_reached and self.climax_memory_burst and climax_generation == generation:
            # Burst: all active functions get maximum memory
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    act_memory = act_memory.at[i].set(self.climax_protection_strength)
                    if i == self.sin_idx:
                        act_memory = act_memory.at[i].add(self.sin_memory_boost)

            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    agg_memory = agg_memory.at[i].set(self.climax_protection_strength)
                    if i in CORE_EXTREME_AGGS:
                        agg_memory = agg_memory.at[i].add(0.1)

            memory_bursts += 1

        # Normal memory accumulation on improvement
        elif improved:
            for i in range(NUM_ACTIVATIONS):
                if float(act_mask[i]) > 0.5:
                    act_memory = act_memory.at[i].add(self.base_memory_strength)
            for i in range(NUM_AGGREGATIONS):
                if float(agg_mask[i]) > 0.5:
                    agg_memory = agg_memory.at[i].add(self.base_memory_strength)

        # Clamp memory
        act_memory = jnp.clip(act_memory, 0.0, 1.0)
        agg_memory = jnp.clip(agg_memory, 0.0, 1.0)

        # === DETERMINE MUTATION RATE ===
        if climax_reached:
            mutation_rate = self.post_climax_mutation_rate
        else:
            mutation_rate = self.exploration_mutation_rate

        # === ACTIVATION MUTATION ===
        candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k1) < mutation_rate:
            # Weight by inverse memory (low memory = more likely)
            weights = []
            for i in candidates:
                w = 0.1 + (1.0 - float(act_memory[i]))
                if i == self.sin_idx:
                    w += 0.3
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            new_idx = int(jax.random.choice(k2, jnp.array(candidates), p=probs))
            act_mask = act_mask.at[new_idx].set(1.0)

        # === AGGREGATION MUTATION ===
        candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
        if candidates and jax.random.uniform(k3) < mutation_rate:
            weights = []
            for i in candidates:
                w = 0.1 + (1.0 - float(agg_memory[i]))
                if i in CORE_EXTREME_AGGS:
                    w += 0.3
                weights.append(w)

            probs = jnp.array(weights)
            probs = probs / probs.sum()

            new_idx = int(jax.random.choice(k4, jnp.array(candidates), p=probs))
            agg_mask = agg_mask.at[new_idx].set(1.0)

        # Update affinities
        if improved:
            active_acts = jnp.where(act_mask > 0.5)[0]
            active_aggs = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
            for g in active_aggs:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)

        # Clamp affinities
        act_affinities = jnp.clip(act_affinities, 0.0, 1.0)
        agg_affinities = jnp.clip(agg_affinities, 0.0, 1.0)

        # Ensure minimum diversity
        if sum(float(act_mask[i]) for i in range(NUM_ACTIVATIONS)) < self.min_active_act:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k1, jnp.array(candidates)))
                act_mask = act_mask.at[new_idx].set(1.0)

        if sum(float(agg_mask[i]) for i in range(NUM_AGGREGATIONS)) < self.min_active_agg:
            candidates = [i for i in CORE_EXTREME_AGGS if float(agg_mask[i]) < 0.5]
            if not candidates:
                candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k3, jnp.array(candidates)))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        new_state = {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_memory': act_memory,
            'agg_memory': agg_memory,
            'climax_reached': climax_reached,
            'climax_generation': climax_generation,
            'stability_counter': stability_counter,
            'memory_bursts': memory_bursts,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        metrics = {
            'climax_reached': climax_reached,
            'climax_generation': climax_generation,
            'memory_bursts': memory_bursts,
            'mean_act_memory': float(act_memory.mean()),
            'mean_agg_memory': float(agg_memory.mean()),
        }

        return new_state, metrics
