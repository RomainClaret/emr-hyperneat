"""Strategy 116: Neurogenesis + Cross-Reactive Dual.

Combines adult neurogenesis (strategy 41) with immune memory cross-reactivity
(strategy 114) for evolutionary momentum through birth + memory.

Key Innovation:
- New functions BORN near memory cells survive better
- Birth near sin-memory creates sin-associated newcomers
- Cross-reactivity creates evolutionary momentum toward sin-extreme
- Young functions with cross-protection get survival boost

Bio inspiration: Adult neurogenesis with immune-style memory.
New neurons near established circuits integrate better.
Memory cross-reactivity creates protected lineages.
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


class NeurogenesisCrossReactiveDualStrategy(PaletteEvolutionStrategy):
    """Birth near memory creates evolutionary momentum.

    Neurogenesis + cross-reactivity.
    New functions near sin-memory survive better.
    """

    name = "neurogenesis_cross_reactive_dual"
    description = "Dual: New functions born near memory cells survive better"

    def __init__(
        self,
        # === NEUROGENESIS PARAMETERS (KEY) ===
        neurogenesis_rate: float = 0.08,         # Rate of new function birth
        young_cross_protection_boost: float = 1.5,  # Boost for young + cross-protected
        maturation_memory_integration: bool = True,
        maturation_period: int = 8,              # Generations until mature
        young_plasticity: float = 2.0,           # Young functions are more plastic
        # === MEMORY PARAMETERS ===
        memory_formation_threshold: float = 0.7,
        cross_protection_strength: float = 0.5,
        memory_cell_lifespan: int = 35,
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        # === Exploration ===
        exploration_rate: float = 0.22,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Neurogenesis Cross-Reactive Dual strategy."""
        # Neurogenesis (KEY)
        self.neurogenesis_rate = neurogenesis_rate
        self.young_cross_protection_boost = young_cross_protection_boost
        self.maturation_memory_integration = maturation_memory_integration
        self.maturation_period = maturation_period
        self.young_plasticity = young_plasticity

        # Memory
        self.memory_formation_threshold = memory_formation_threshold
        self.cross_protection_strength = cross_protection_strength
        self.memory_cell_lifespan = memory_cell_lifespan

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay

        # Exploration
        self.exploration_rate = exploration_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neurogenesis and memory tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # Birth generation tracking (for maturation)
        act_birth_gen = jnp.full(NUM_ACTIVATIONS, -100.0)  # -100 = mature
        agg_birth_gen = jnp.full(NUM_AGGREGATIONS, -100.0)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_birth_gen = act_birth_gen.at[i].set(0.0)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_birth_gen = agg_birth_gen.at[i].set(0.0)

        # Memory cells
        act_memory = {}  # {idx: {'formation_gen': int, 'protection': float}}
        agg_memory = {}

        # Cross-protection
        act_cross_protection = jnp.zeros(NUM_ACTIVATIONS)
        agg_cross_protection = jnp.zeros(NUM_AGGREGATIONS)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Neurogenesis tracking
            'act_birth_gen': act_birth_gen,
            'agg_birth_gen': agg_birth_gen,
            # Memory
            'act_memory': act_memory,
            'agg_memory': agg_memory,
            # Cross-protection
            'act_cross_protection': act_cross_protection,
            'agg_cross_protection': agg_cross_protection,
            # Stats
            'neurogenesis_events': 0,
            'young_survivals': 0,
            'memory_formations': 0,
            'cross_protection_boosts': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1160000),
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

    def _is_young(self, birth_gen: float, current_gen: int) -> bool:
        """Check if a function is still in maturation period."""
        age = current_gen - birth_gen
        return 0 <= age < self.maturation_period

    def _attempt_neurogenesis(
        self,
        mask: jnp.ndarray,
        birth_gen: jnp.ndarray,
        memory: Dict,
        generation: int,
        stagnation: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Attempt to birth new functions near memory cells."""
        n_funcs = len(mask)
        new_mask = mask.copy()
        new_birth = birth_gen.copy()
        births = 0

        # Neurogenesis rate scales with stagnation
        rate = self.neurogenesis_rate * (1 + stagnation * 0.1)

        # Prefer indices get priority
        if prefer_indices:
            for idx in prefer_indices:
                if 0 <= idx < n_funcs and mask[idx] < 0.5:
                    key, subkey = jax.random.split(key)
                    # Higher birth rate near memory cells
                    near_memory = any(abs(idx - m) <= 1 for m in memory.keys())
                    boost = 2.0 if near_memory else 1.0
                    if float(jax.random.uniform(subkey)) < rate * boost:
                        new_mask = new_mask.at[idx].set(1.0)
                        new_birth = new_birth.at[idx].set(float(generation))
                        births += 1

        return new_mask, new_birth, births

    def _update_memory(
        self,
        memory: Dict,
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        best_fitness: float,
        generation: int,
        n_funcs: int,
    ) -> Tuple[Dict, int]:
        """Update memory cells."""
        new_memory = dict(memory)
        formations = 0

        # Form memory
        if best_fitness >= self.memory_formation_threshold:
            for i in range(n_funcs):
                if mask[i] > 0.5 and affinities[i] > 0.5:
                    if i not in new_memory:
                        new_memory[i] = {
                            'formation_gen': generation,
                            'protection': 0.9,
                        }
                        formations += 1
                    else:
                        new_memory[i]['protection'] = min(
                            1.0, new_memory[i]['protection'] + 0.1
                        )

        # Decay old memories
        expired = []
        for i, info in new_memory.items():
            age = generation - info['formation_gen']
            if age > self.memory_cell_lifespan:
                expired.append(i)
            elif age > self.memory_cell_lifespan * 0.7:
                decay_factor = (self.memory_cell_lifespan - age) / (self.memory_cell_lifespan * 0.3)
                new_memory[i]['protection'] *= decay_factor

        for i in expired:
            del new_memory[i]

        return new_memory, formations

    def _update_cross_protection(
        self,
        cross_protection: jnp.ndarray,
        memory: Dict,
        n_funcs: int,
        is_sin_domain: bool = True,
    ) -> Tuple[jnp.ndarray, int]:
        """Update cross-protection based on memory."""
        new_cross = cross_protection * 0.95  # Decay
        boosts = 0

        if is_sin_domain:
            # Sin's memory protects aggregations
            if 4 in memory:
                # Return unchanged for activations (this is for aggs)
                pass
        else:
            # For aggregations: sin's memory protects extremes
            # This is handled in the main update
            pass

        return new_cross, boosts

    def _compute_survival_score(
        self,
        idx: int,
        affinity: float,
        cross_protection: float,
        birth_gen: float,
        generation: int,
        memory: Dict,
    ) -> float:
        """Compute survival score with young + cross-protection boost."""
        score = affinity

        # Cross-protection boost
        score += cross_protection * 0.3

        # Young + cross-protection boost (KEY INNOVATION)
        is_young = self._is_young(birth_gen, generation)
        if is_young:
            # Young functions with cross-protection survive better
            if cross_protection > 0.3:
                score += self.young_cross_protection_boost * 0.2
            # Young functions are more plastic
            score *= self.young_plasticity * 0.5 + 0.5

        # Memory boost
        if idx in memory:
            score += 0.3 * memory[idx]['protection']

        return score

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        cross_protection: jnp.ndarray,
        birth_gen: jnp.ndarray,
        memory: Dict,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        generation: int,
        stagnation: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette with neurogenesis and cross-protection."""
        k1, k2 = jax.random.split(key)

        # Compute survival scores
        score = jnp.zeros(n_funcs)
        young_survivals = 0

        for i in range(n_funcs):
            s = self._compute_survival_score(
                i, float(affinities[i]), float(cross_protection[i]),
                float(birth_gen[i]), generation, memory
            )
            score = score.at[i].set(s)
            if self._is_young(float(birth_gen[i]), generation) and s > 0.5:
                young_survivals += 1

        # Preference boost
        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + 0.6)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Exploration
        exploration_prob = self.exploration_rate * (1 + stagnation * 0.1)
        if prefer_indices:
            for idx in prefer_indices:
                if 0 <= idx < n_funcs and mask[idx] < 0.5:
                    if float(jax.random.uniform(k1)) < exploration_prob * 1.5:
                        mask = mask.at[idx].set(1.0)
                    k1, _ = jax.random.split(k1)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        if active_count < min_diversity:
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k2, jnp.array(inactive),
                                          shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)

        return mask, young_survivals

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with neurogenesis + cross-reactive dynamics."""
        key, k1, k2, k3, k4 = jax.random.split(state['rng_key'], 5)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === AFFINITY UPDATE ===
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    # Young functions learn faster
                    lr = self.act_affinity_lr
                    if self._is_young(float(state['act_birth_gen'][i]), generation):
                        lr *= self.young_plasticity
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + lr * fitness_delta)
                    )
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    lr = self.agg_affinity_lr
                    if self._is_young(float(state['agg_birth_gen'][j]), generation):
                        lr *= self.young_plasticity
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + lr * fitness_delta)
                    )

        # === MEMORY UPDATE ===
        new_act_memory, act_formations = self._update_memory(
            state['act_memory'], state['act_mask'], new_act_aff,
            best_fitness, generation, NUM_ACTIVATIONS
        )
        new_agg_memory, agg_formations = self._update_memory(
            state['agg_memory'], state['agg_mask'], new_agg_aff,
            best_fitness, generation, NUM_AGGREGATIONS
        )

        # === CROSS-PROTECTION UPDATE ===
        new_act_cross = state['act_cross_protection'] * 0.95
        new_agg_cross = state['agg_cross_protection'] * 0.95
        cross_boosts = 0

        # Sin's memory protects extreme aggregations
        if 4 in new_act_memory:
            sin_protection = new_act_memory[4]['protection']
            for j in CORE_EXTREME_AGGS:
                new_agg_cross = new_agg_cross.at[j].set(
                    min(1.0, float(new_agg_cross[j]) + sin_protection * self.cross_protection_strength)
                )
                cross_boosts += 1

        # Extreme memory protects sin
        for j in CORE_EXTREME_AGGS:
            if j in new_agg_memory:
                ext_protection = new_agg_memory[j]['protection']
                new_act_cross = new_act_cross.at[4].set(
                    min(1.0, float(new_act_cross[4]) + ext_protection * self.cross_protection_strength)
                )
                cross_boosts += 1

        # === NEUROGENESIS ===
        new_act_mask, new_act_birth, act_births = self._attempt_neurogenesis(
            state['act_mask'], state['act_birth_gen'], new_act_memory,
            generation, new_stagnation, k1, prefer_indices=[4]
        )
        new_agg_mask, new_agg_birth, agg_births = self._attempt_neurogenesis(
            state['agg_mask'], state['agg_birth_gen'], new_agg_memory,
            generation, new_stagnation, k2, prefer_indices=list(CORE_EXTREME_AGGS)
        )

        # === PALETTE SELECTION ===
        new_act_mask, act_young_survivals = self._select_palette(
            new_act_aff, new_act_cross, new_act_birth, new_act_memory,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            self.min_diversity_act, generation, new_stagnation, k3,
            prefer_indices=[4]
        )
        new_agg_mask, agg_young_survivals = self._select_palette(
            new_agg_aff, new_agg_cross, new_agg_birth, new_agg_memory,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            self.min_diversity_agg, generation, new_stagnation, k4,
            prefer_indices=list(CORE_EXTREME_AGGS)
        )

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_birth_gen': new_act_birth,
            'agg_birth_gen': new_agg_birth,
            'act_memory': new_act_memory,
            'agg_memory': new_agg_memory,
            'act_cross_protection': new_act_cross,
            'agg_cross_protection': new_agg_cross,
            'neurogenesis_events': state['neurogenesis_events'] + act_births + agg_births,
            'young_survivals': state['young_survivals'] + act_young_survivals + agg_young_survivals,
            'memory_formations': state['memory_formations'] + act_formations + agg_formations,
            'cross_protection_boosts': state['cross_protection_boosts'] + cross_boosts,
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
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neurogenesis metrics (KEY)
            'neurogenesis_events': new_state['neurogenesis_events'],
            'young_survivals': new_state['young_survivals'],
            'sin_is_young': self._is_young(float(new_act_birth[4]), generation),
            # Memory metrics
            'n_act_memory': len(new_act_memory),
            'sin_has_memory': 4 in new_act_memory,
            'memory_formations': new_state['memory_formations'],
            # Cross-protection metrics
            'sin_cross_protection': float(new_act_cross[4]),
            'max_cross_protection': float(new_agg_cross[2]),
            'cross_protection_boosts': new_state['cross_protection_boosts'],
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'neurogenesis_events': state['neurogenesis_events'],
            'young_survivals': state['young_survivals'],
            'cross_protection_boosts': state['cross_protection_boosts'],
            'generation': state['generation'],
        }
