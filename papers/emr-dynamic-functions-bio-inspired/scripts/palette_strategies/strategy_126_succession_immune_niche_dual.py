"""Strategy 126: Succession-Immune Niche Memory Dual.

Combines Ecological Succession niches (#26) with Immune Memory (#60).
Separate memory pools for activation and aggregation niches.
Cross-niche discoveries get bonus memory strength.

Key Innovation:
- Activation niches: oscillatory (sin), nonlinear, other
- Aggregation niches: extreme (max/min), averaging, other
- Each niche has its own memory pool with capacity limit
- Cross-niche discoveries (e.g., sin found while extreme active) get bonus

Biological basis: Memory B cells are organized by antigen type.
Cross-reactive antibodies that recognize multiple antigens are especially valuable.

Expected: Balanced discovery across niches with strong cross-domain synergy.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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


class SuccessionImmuneNicheDualStrategy(PaletteEvolutionStrategy):
    """Succession-Immune niche memory for dual palette evolution.

    Separate memory pools for activation and aggregation
    niches. Cross-niche discoveries get bonus memory strength.

    Critical innovation: Domain-specific memory with cross-niche synergy.
    """

    name = "succession_immune_niche_dual"
    description = "Dual: Niche-specific memory pools with cross-domain synergy"

    def __init__(
        self,
        # === Niche memory parameters ===
        niche_memory_capacity: int = 3,
        cross_niche_memory_bonus: float = 0.3,
        base_memory_lifespan: int = 30,
        memory_protection_strength: float = 0.9,
        # === Succession phases ===
        pioneer_phase_end: int = 15,
        transition_phase_end: int = 40,
        pioneer_mutation_rate: float = 0.15,
        transition_mutation_rate: float = 0.08,
        climax_mutation_rate: float = 0.02,
        # === Sin index ===
        sin_idx: int = 4,
        # === General parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        tag_threshold: float = 0.5,
        tag_decay: float = 0.9,
        captured_protection: float = 0.85,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Succession-Immune Niche strategy."""
        # Niche memory
        self.niche_memory_capacity = niche_memory_capacity
        self.cross_niche_memory_bonus = cross_niche_memory_bonus
        self.base_memory_lifespan = base_memory_lifespan
        self.memory_protection_strength = memory_protection_strength

        # Succession phases
        self.pioneer_phase_end = pioneer_phase_end
        self.transition_phase_end = transition_phase_end
        self.pioneer_mutation_rate = pioneer_mutation_rate
        self.transition_mutation_rate = transition_mutation_rate
        self.climax_mutation_rate = climax_mutation_rate

        # Sin index
        self.sin_idx = sin_idx

        # General
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay
        self.tag_threshold = tag_threshold
        self.tag_decay = tag_decay
        self.captured_protection = captured_protection

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

        # Define niches
        self.oscillatory_acts = {4, 11, 12, 13, 15}  # sin, burst, resonator, etc.
        self.nonlinear_acts = {0, 1, 2, 3, 5, 6, 7}  # sigmoid, tanh, relu, etc.
        self.extreme_aggs = {2, 3}  # max, min
        self.averaging_aggs = {0, 1}  # sum, mean

    def _get_act_niche(self, idx: int) -> int:
        if idx in self.oscillatory_acts:
            return 0
        elif idx in self.nonlinear_acts:
            return 1
        else:
            return 2

    def _get_agg_niche(self, idx: int) -> int:
        if idx in self.extreme_aggs:
            return 0
        elif idx in self.averaging_aggs:
            return 1
        else:
            return 2

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with niche memory pools."""
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

        # Niche memories: {niche_id: {idx: (strength, expiry_gen)}}
        act_niche_memories: Dict[int, Dict[int, Tuple[float, int]]] = {0: {}, 1: {}, 2: {}}
        agg_niche_memories: Dict[int, Dict[int, Tuple[float, int]]] = {0: {}, 1: {}, 2: {}}

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Niche memories
            'act_niche_memories': act_niche_memories,
            'agg_niche_memories': agg_niche_memories,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            # Stats
            'memory_formations': 0,
            'cross_niche_bonuses': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 1260000),
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

    def _get_phase(self, generation: int) -> str:
        if generation < self.pioneer_phase_end:
            return "pioneer"
        elif generation < self.transition_phase_end:
            return "transition"
        else:
            return "climax"

    def _get_mutation_rate(self, generation: int) -> float:
        phase = self._get_phase(generation)
        if phase == "pioneer":
            return self.pioneer_mutation_rate
        elif phase == "transition":
            return self.transition_mutation_rate
        else:
            return self.climax_mutation_rate

    def _is_memory_protected(
        self,
        idx: int,
        is_activation: bool,
        niche_memories: Dict[int, Dict[int, Tuple[float, int]]],
        generation: int,
    ) -> bool:
        niche = self._get_act_niche(idx) if is_activation else self._get_agg_niche(idx)
        memories = niche_memories[niche]
        if idx in memories:
            strength, expiry = memories[idx]
            if generation < expiry:
                return True
        return False

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with niche memory mechanics."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        phase = self._get_phase(generation)
        mutation_rate = self._get_mutation_rate(generation)

        # Deep copy niche memories
        act_niche_memories = {k: dict(v) for k, v in state['act_niche_memories'].items()}
        agg_niche_memories = {k: dict(v) for k, v in state['agg_niche_memories'].items()}

        # Cleanup expired memories
        for niche in act_niche_memories:
            expired = [idx for idx, (_, exp) in act_niche_memories[niche].items() if generation >= exp]
            for idx in expired:
                del act_niche_memories[niche][idx]
        for niche in agg_niche_memories:
            expired = [idx for idx, (_, exp) in agg_niche_memories[niche].items() if generation >= exp]
            for idx in expired:
                del agg_niche_memories[niche][idx]

        memory_formations = state['memory_formations']
        cross_niche_bonuses = state['cross_niche_bonuses']

        # Get active functions for cross-niche detection
        act_mask = state['act_mask']
        agg_mask = state['agg_mask']
        active_aggs = set(int(i) for i in jnp.where(agg_mask > 0.5)[0])
        active_acts = set(int(i) for i in jnp.where(act_mask > 0.5)[0])

        # === ACTIVATION UPDATE ===
        act_affinities = state['act_affinities']
        act_tags = state['act_tags'] * self.tag_decay
        act_captured = state['act_captured']

        # Pioneer: explore oscillatory niche (contains sin)
        if phase == "pioneer" and jax.random.uniform(k1) < mutation_rate:
            # Prefer oscillatory niche
            candidates = [i for i in range(NUM_ACTIVATIONS)
                         if float(act_mask[i]) < 0.5 and self._get_act_niche(i) == 0]
            if not candidates:
                candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k2, jnp.array(candidates)))
                act_mask = act_mask.at[new_idx].set(1.0)

                # Add to memory with potential cross-niche bonus
                niche = self._get_act_niche(new_idx)
                strength = 1.0
                if any(self._get_agg_niche(g) == 0 for g in active_aggs):  # Extreme agg active
                    strength += self.cross_niche_memory_bonus
                    cross_niche_bonuses += 1

                expiry = generation + int(self.base_memory_lifespan * strength)
                act_niche_memories[niche][new_idx] = (strength, expiry)
                memory_formations += 1

                # Enforce capacity
                if len(act_niche_memories[niche]) > self.niche_memory_capacity:
                    weakest = min(act_niche_memories[niche].keys(),
                                 key=lambda k: act_niche_memories[niche][k][0])
                    del act_niche_memories[niche][weakest]

        elif phase != "pioneer" and jax.random.uniform(k1) < mutation_rate:
            candidates = [i for i in range(NUM_ACTIVATIONS) if float(act_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k2, jnp.array(candidates)))
                act_mask = act_mask.at[new_idx].set(1.0)

                niche = self._get_act_niche(new_idx)
                strength = 0.5 if phase == "climax" else 1.0
                expiry = generation + int(self.base_memory_lifespan * strength)
                act_niche_memories[niche][new_idx] = (strength, expiry)

        # === AGGREGATION UPDATE ===
        agg_affinities = state['agg_affinities']
        agg_tags = state['agg_tags'] * self.tag_decay
        agg_captured = state['agg_captured']

        # Pioneer: prioritize extreme niche
        if phase == "pioneer" and jax.random.uniform(k3) < mutation_rate:
            candidates = [i for i in CORE_EXTREME_AGGS if float(agg_mask[i]) < 0.5]
            if not candidates:
                candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k4, jnp.array(candidates)))
                agg_mask = agg_mask.at[new_idx].set(1.0)

                niche = self._get_agg_niche(new_idx)
                strength = 1.0
                if any(self._get_act_niche(a) == 0 for a in active_acts):  # Oscillatory act active
                    strength += self.cross_niche_memory_bonus
                    cross_niche_bonuses += 1

                expiry = generation + int(self.base_memory_lifespan * strength)
                agg_niche_memories[niche][new_idx] = (strength, expiry)
                memory_formations += 1

                if len(agg_niche_memories[niche]) > self.niche_memory_capacity:
                    weakest = min(agg_niche_memories[niche].keys(),
                                 key=lambda k: agg_niche_memories[niche][k][0])
                    del agg_niche_memories[niche][weakest]

        elif phase != "pioneer" and jax.random.uniform(k3) < mutation_rate:
            candidates = [i for i in range(NUM_AGGREGATIONS) if float(agg_mask[i]) < 0.5]
            if candidates:
                new_idx = int(jax.random.choice(k4, jnp.array(candidates)))
                agg_mask = agg_mask.at[new_idx].set(1.0)

        # Update affinities on improvement
        if improved:
            active_acts_list = jnp.where(act_mask > 0.5)[0]
            active_aggs_list = jnp.where(agg_mask > 0.5)[0]

            for a in active_acts_list:
                act_affinities = act_affinities.at[a].add(self.affinity_lr)
            for g in active_aggs_list:
                agg_affinities = agg_affinities.at[g].add(self.affinity_lr)

        # Clamp
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
            'act_niche_memories': act_niche_memories,
            'agg_niche_memories': agg_niche_memories,
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'memory_formations': memory_formations,
            'cross_niche_bonuses': cross_niche_bonuses,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': state['fitness_history'] + [best_fitness],
        }

        metrics = {
            'phase': phase,
            'memory_formations': memory_formations - state['memory_formations'],
            'cross_niche_bonuses': cross_niche_bonuses - state['cross_niche_bonuses'],
        }

        return new_state, metrics
