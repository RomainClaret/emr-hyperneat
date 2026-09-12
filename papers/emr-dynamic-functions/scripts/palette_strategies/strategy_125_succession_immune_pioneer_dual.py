"""Strategy 125: Succession-Immune Pioneer Memory Dual.

Combines Ecological Succession (#26) with Immune Memory (#60).
Pioneer species that succeed become long-lived founder memories.

Key Innovation:
- Pioneer phase: Functions discovered early get "founder" status opportunity
- Intermediate phase: Successful founders form long-lived immune memories
- Climax phase: Founder memories are strongly protected from removal
- Cross-domain: Founders in one domain protect partners in other domain

Biological basis: In ecology, pioneer species establish ecosystems. In immunology,
first-responders to pathogens become memory cells. Combining: early discoveries
that prove successful become strongly protected long-term memories.

Expected: Early sin discovery creates founder memory that persists through task changes.
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


class SuccessionPhase:
    """Ecological succession phases."""
    PIONEER = "pioneer"
    INTERMEDIATE = "intermediate"
    CLIMAX = "climax"


class SuccessionImmunePioneerDualStrategy(PaletteEvolutionStrategy):
    """Succession-based memory formation with pioneer founder advantage.

    Combines ecological succession developmental phases
    with immune memory persistence. Early pioneers that prove successful
    form founder memories with exceptional longevity.

    Critical innovation: Functions discovered during pioneer phase AND
    successful during intermediate phase become "founder memories" with
    much longer lifespan and stronger protection than regular memories.
    """

    name = "succession_immune_pioneer_dual"
    description = "Dual: Pioneer successes become founder memories with long persistence"

    # Function classifications (from ecological_succession)
    ACT_GENERALIST = [0, 1, 2, 5, 6]
    ACT_SPECIALIST = [4, 11, 12, 13, 15]  # sin is specialist!
    AGG_GENERALIST = [0, 1]
    AGG_SPECIALIST = [2, 3]  # max, min are specialists!

    def __init__(
        self,
        # === Succession timing ===
        pioneer_end: int = 15,
        intermediate_end: int = 40,
        # === Pioneer memory parameters (NEW) ===
        pioneer_memory_threshold: float = 0.6,    # Fitness to form pioneer memory
        pioneer_memory_lifespan: int = 50,        # Much longer than regular
        regular_memory_lifespan: int = 25,        # Regular memory lifespan
        founder_cross_protection: float = 0.5,    # Cross-domain founder protection
        founder_protection_strength: float = 0.95,# Founder memories very protected
        regular_protection_strength: float = 0.8, # Regular memory protection
        # === Immune memory parameters (from #60) ===
        memory_formation_threshold: float = 0.75,
        plasma_cell_duration: int = 5,
        plasma_cell_boost: float = 1.5,
        # === Succession exploration rates ===
        pioneer_mutation_rate: float = 0.25,
        pioneer_generalist_bias: float = 2.0,
        intermediate_mutation_rate: float = 0.12,
        climax_mutation_rate: float = 0.04,
        climax_specialist_bias: float = 1.5,
        # === Affinity parameters ===
        affinity_lr: float = 0.10,
        affinity_decay: float = 0.01,
        cross_learning_rate: float = 0.08,
        # === Constraints ===
        max_active_act: int = 6,
        max_active_agg: int = 4,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Succession-Immune Pioneer strategy."""
        # Succession timing
        self.pioneer_end = pioneer_end
        self.intermediate_end = intermediate_end

        # Pioneer memory (NEW)
        self.pioneer_memory_threshold = pioneer_memory_threshold
        self.pioneer_memory_lifespan = pioneer_memory_lifespan
        self.regular_memory_lifespan = regular_memory_lifespan
        self.founder_cross_protection = founder_cross_protection
        self.founder_protection_strength = founder_protection_strength
        self.regular_protection_strength = regular_protection_strength

        # Immune memory
        self.memory_formation_threshold = memory_formation_threshold
        self.plasma_cell_duration = plasma_cell_duration
        self.plasma_cell_boost = plasma_cell_boost

        # Succession rates
        self.pioneer_mutation_rate = pioneer_mutation_rate
        self.pioneer_generalist_bias = pioneer_generalist_bias
        self.intermediate_mutation_rate = intermediate_mutation_rate
        self.climax_mutation_rate = climax_mutation_rate
        self.climax_specialist_bias = climax_specialist_bias

        # Affinity
        self.affinity_lr = affinity_lr
        self.affinity_decay = affinity_decay
        self.cross_learning_rate = cross_learning_rate

        # Constraints
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

        # Build type lookups
        self.act_type = {}
        for i in self.ACT_GENERALIST:
            self.act_type[i] = 'generalist'
        for i in self.ACT_SPECIALIST:
            self.act_type[i] = 'specialist'

        self.agg_type = {}
        for i in self.AGG_GENERALIST:
            self.agg_type[i] = 'generalist'
        for i in self.AGG_SPECIALIST:
            self.agg_type[i] = 'specialist'

    def _get_phase(self, generation: int) -> str:
        """Get current succession phase."""
        if generation < self.pioneer_end:
            return SuccessionPhase.PIONEER
        elif generation < self.intermediate_end:
            return SuccessionPhase.INTERMEDIATE
        else:
            return SuccessionPhase.CLIMAX

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with succession and immune tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5
        agg_affinity = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Cross-domain
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        # Pioneer discovery tracking (NEW)
        act_pioneers: Set[int] = set(initial_act)  # Functions discovered during pioneer
        agg_pioneers: Set[int] = set(initial_agg)

        # Memory state (Dict: func -> {'formation_gen', 'is_founder', 'lifespan', 'protection'})
        act_memory: Dict[int, Dict] = {}
        agg_memory: Dict[int, Dict] = {}

        # Plasma cells (Dict: func -> gens_remaining)
        act_plasma: Dict[int, int] = {}
        agg_plasma: Dict[int, int] = {}

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinity': act_affinity,
            'agg_affinity': agg_affinity,
            'cross_affinity': cross_affinity,
            # Pioneer tracking (NEW)
            'act_pioneers': act_pioneers,
            'agg_pioneers': agg_pioneers,
            # Memory state
            'act_memory': act_memory,
            'agg_memory': agg_memory,
            'act_plasma': act_plasma,
            'agg_plasma': agg_plasma,
            # Stats
            'act_founder_memories': 0,
            'agg_founder_memories': 0,
            'act_regular_memories': 0,
            'agg_regular_memories': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1250000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': SuccessionPhase.PIONEER,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _update_pioneers(
        self,
        pioneers: Set[int],
        mask: jnp.ndarray,
        phase: str,
        n_funcs: int,
    ) -> Set[int]:
        """Update pioneer set during pioneer phase."""
        if phase != SuccessionPhase.PIONEER:
            return pioneers

        new_pioneers = set(pioneers)
        for i in range(n_funcs):
            if mask[i] > 0.5 and i not in new_pioneers:
                new_pioneers.add(i)

        return new_pioneers

    def _update_memory(
        self,
        memory: Dict[int, Dict],
        plasma: Dict[int, int],
        pioneers: Set[int],
        active_palette: List[int],
        best_fitness: float,
        generation: int,
        phase: str,
    ) -> Tuple[Dict[int, Dict], Dict[int, int], int, int]:
        """Update memory cells with founder distinction.

        KEY INNOVATION: Pioneers that succeed become founder memories.
        """
        new_memory = dict(memory)
        new_plasma = {}
        founder_formed = 0
        regular_formed = 0

        # Decay plasma cells
        for func, gens in plasma.items():
            if gens > 1:
                new_plasma[func] = gens - 1

        # Form memories during intermediate+ phases
        if phase != SuccessionPhase.PIONEER and best_fitness >= self.memory_formation_threshold:
            for func in active_palette:
                if func not in new_memory:
                    # Determine if founder (was a pioneer)
                    is_founder = func in pioneers

                    if is_founder:
                        lifespan = self.pioneer_memory_lifespan
                        protection = self.founder_protection_strength
                        founder_formed += 1
                    else:
                        lifespan = self.regular_memory_lifespan
                        protection = self.regular_protection_strength
                        regular_formed += 1

                    new_memory[func] = {
                        'formation_gen': generation,
                        'is_founder': is_founder,
                        'lifespan': lifespan,
                        'protection': protection,
                        'fitness': best_fitness,
                    }
                    new_plasma[func] = self.plasma_cell_duration

        # Expire old memories
        expired = []
        for func, info in new_memory.items():
            age = generation - info['formation_gen']
            if age > info['lifespan']:
                expired.append(func)
            elif age > info['lifespan'] * 0.8:
                # Decay protection for old memories
                decay = (info['lifespan'] - age) / (info['lifespan'] * 0.2)
                new_memory[func]['protection'] = info['protection'] * decay

        for func in expired:
            del new_memory[func]

        return new_memory, new_plasma, founder_formed, regular_formed

    def _get_function_bias(self, func_idx: int, phase: str, is_activation: bool) -> float:
        """Get activation bias based on phase and function type."""
        if is_activation:
            func_type = self.act_type.get(func_idx, 'neutral')
        else:
            func_type = self.agg_type.get(func_idx, 'neutral')

        if phase == SuccessionPhase.PIONEER:
            return self.pioneer_generalist_bias if func_type == 'generalist' else 0.5
        elif phase == SuccessionPhase.INTERMEDIATE:
            return 1.0
        else:  # CLIMAX
            return self.climax_specialist_bias if func_type == 'specialist' else 1.0

    def _get_removal_probability(
        self,
        func: int,
        memory: Dict[int, Dict],
        plasma: Dict[int, int],
        partner_memory: Dict[int, Dict],
        partner_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        is_activation: bool,
    ) -> float:
        """Get probability of removal considering founder protection."""
        base_prob = 1.0

        # Memory protection
        if func in memory:
            info = memory[func]
            base_prob *= (1 - info['protection'])

        # Plasma boost
        if func in plasma:
            base_prob *= 0.5

        # Cross-domain founder protection (NEW)
        # If partner domain has founder memories, protect this function
        for partner_func, partner_info in partner_memory.items():
            if partner_mask[partner_func] > 0.5 and partner_info.get('is_founder', False):
                if is_activation:
                    aff = float(cross_affinity[func, partner_func])
                else:
                    aff = float(cross_affinity[partner_func, func])
                if aff > 0.5:
                    base_prob *= (1 - self.founder_cross_protection * aff)

        return max(0.01, base_prob)

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        phase: str,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update affinity."""
        active = (mask > 0.5).astype(jnp.float32)

        if improved:
            signal = self.affinity_lr * active
        else:
            signal = -self.affinity_lr * 0.3 * active

        new_affinity = affinity + signal
        decay_rate = self.affinity_decay * (1.5 if phase == SuccessionPhase.CLIMAX else 1.0)
        new_affinity = new_affinity - decay_rate * (1 - active) * affinity

        return jnp.clip(new_affinity, 0.05, 0.95)

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improved: bool,
        phase: str,
    ) -> jnp.ndarray:
        """Update cross-domain affinity."""
        act_active = (act_mask > 0.5).astype(jnp.float32)
        agg_active = (agg_mask > 0.5).astype(jnp.float32)
        cross_active = jnp.outer(act_active, agg_active)

        if improved:
            delta = self.cross_learning_rate * cross_active
        else:
            delta = -self.cross_learning_rate * 0.3 * cross_active

        new_cross = cross_affinity + delta
        decay_rate = self.affinity_decay * (1.5 if phase == SuccessionPhase.CLIMAX else 1.0)
        inactive = 1.0 - cross_active
        new_cross = new_cross - decay_rate * inactive * (cross_affinity - 0.5)

        return jnp.clip(new_cross, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        affinity: jnp.ndarray,
        memory: Dict[int, Dict],
        plasma: Dict[int, int],
        partner_memory: Dict[int, Dict],
        partner_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        phase: str,
        is_activation: bool,
        max_active: int,
        min_active: int,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply succession mutation with memory protection."""
        key1, key2 = jax.random.split(key)

        if phase == SuccessionPhase.PIONEER:
            base_rate = self.pioneer_mutation_rate
        elif phase == SuccessionPhase.INTERMEDIATE:
            base_rate = self.intermediate_mutation_rate
        else:
            base_rate = self.climax_mutation_rate

        new_mask = mask.copy()
        activated = []
        deactivated = []

        activate_probs = jax.random.uniform(key1, (n_funcs,))
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))

        for i in range(n_funcs):
            aff = float(affinity[i])
            bias = self._get_function_bias(i, phase, is_activation)

            if mask[i] < 0.5:
                # Inactive - might activate
                current_active = int(jnp.sum(mask > 0.5))
                if current_active >= max_active:
                    continue

                rate = base_rate * 0.5 * bias * (0.5 + 0.5 * aff)
                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                removal_prob = self._get_removal_probability(
                    i, memory, plasma, partner_memory, partner_mask, cross_affinity, is_activation
                )

                rate = base_rate * 0.4 * removal_prob * (1.0 - aff)

                # Reduce rate in climax
                if phase == SuccessionPhase.CLIMAX:
                    rate *= 0.3

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with succession-immune mechanics."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        phase = self._get_phase(generation)
        phase_changed = phase != state['phase']

        act_palette = mask_to_indices(state['act_mask'])
        agg_palette = mask_to_indices(state['agg_mask'])

        # Update pioneer sets (only during pioneer phase)
        new_act_pioneers = self._update_pioneers(
            state['act_pioneers'], state['act_mask'], phase, NUM_ACTIVATIONS
        )
        new_agg_pioneers = self._update_pioneers(
            state['agg_pioneers'], state['agg_mask'], phase, NUM_AGGREGATIONS
        )

        # Update memories with founder distinction
        new_act_mem, new_act_plasma, act_founder, act_regular = self._update_memory(
            state['act_memory'], state['act_plasma'], new_act_pioneers,
            act_palette, best_fitness, generation, phase
        )
        new_agg_mem, new_agg_plasma, agg_founder, agg_regular = self._update_memory(
            state['agg_memory'], state['agg_plasma'], new_agg_pioneers,
            agg_palette, best_fitness, generation, phase
        )

        # Update affinities
        new_act_aff = self._update_affinity(
            state['act_affinity'], state['act_mask'], improved, phase, NUM_ACTIVATIONS
        )
        new_agg_aff = self._update_affinity(
            state['agg_affinity'], state['agg_mask'], improved, phase, NUM_AGGREGATIONS
        )
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improved, phase
        )

        # Mutate palettes
        new_act_mask, act_mut = self._mutate_palette(
            k_act, state['act_mask'], new_act_aff, new_act_mem, new_act_plasma,
            new_agg_mem, state['agg_mask'], new_cross, phase,
            True, self.max_active_act, self.min_active_act, NUM_ACTIVATIONS
        )
        new_agg_mask, agg_mut = self._mutate_palette(
            k_agg, state['agg_mask'], new_agg_aff, new_agg_mem, new_agg_plasma,
            new_act_mem, state['act_mask'], new_cross, phase,
            False, self.max_active_agg, self.min_active_agg, NUM_AGGREGATIONS
        )

        # Track fitness
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinity': new_act_aff,
            'agg_affinity': new_agg_aff,
            'cross_affinity': new_cross,
            'act_pioneers': new_act_pioneers,
            'agg_pioneers': new_agg_pioneers,
            'act_memory': new_act_mem,
            'agg_memory': new_agg_mem,
            'act_plasma': new_act_plasma,
            'agg_plasma': new_agg_plasma,
            'act_founder_memories': state['act_founder_memories'] + act_founder,
            'agg_founder_memories': state['agg_founder_memories'] + agg_founder,
            'act_regular_memories': state['act_regular_memories'] + act_regular,
            'agg_regular_memories': state['agg_regular_memories'] + agg_regular,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'fitness_history': fitness_history,
        }

        new_act_palette = mask_to_indices(new_act_mask)
        new_agg_palette = mask_to_indices(new_agg_mask)

        # Count current founder vs regular memories
        n_act_founders = sum(1 for info in new_act_mem.values() if info.get('is_founder'))
        n_agg_founders = sum(1 for info in new_agg_mem.values() if info.get('is_founder'))

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': new_act_palette,
            'current_agg_palette': new_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Phase metrics
            'phase': phase,
            'phase_changed': phase_changed,
            # Pioneer metrics (NEW)
            'n_act_pioneers': len(new_act_pioneers),
            'n_agg_pioneers': len(new_agg_pioneers),
            'sin_is_pioneer': 4 in new_act_pioneers,
            'max_is_pioneer': 2 in new_agg_pioneers,
            # Memory metrics
            'n_act_memory': len(new_act_mem),
            'n_agg_memory': len(new_agg_mem),
            'n_act_founders': n_act_founders,
            'n_agg_founders': n_agg_founders,
            'act_founder_formed': act_founder,
            'agg_founder_formed': agg_founder,
            # Founder status
            'sin_has_founder_memory': (4 in new_act_mem and new_act_mem[4].get('is_founder', False)),
            'max_has_founder_memory': (2 in new_agg_mem and new_agg_mem[2].get('is_founder', False)),
            # Affinity
            'act_avg_affinity': float(jnp.mean(new_act_aff)),
            'agg_avg_affinity': float(jnp.mean(new_agg_aff)),
            'sin_affinity': float(new_act_aff[4]),
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Status
            'has_sin': 4 in new_act_palette,
            'has_max': 2 in new_agg_palette,
            'has_min': 3 in new_agg_palette,
            # Mutations
            'act_activated': act_mut['activated'],
            'act_deactivated': act_mut['deactivated'],
            'agg_activated': agg_mut['activated'],
            'agg_deactivated': agg_mut['deactivated'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        n_act_founders = sum(1 for info in state['act_memory'].values() if info.get('is_founder'))
        n_agg_founders = sum(1 for info in state['agg_memory'].values() if info.get('is_founder'))

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'phase': state['phase'],
            'generation': state['generation'],
            # Pioneer info
            'n_act_pioneers': len(state['act_pioneers']),
            'n_agg_pioneers': len(state['agg_pioneers']),
            'sin_is_pioneer': 4 in state['act_pioneers'],
            # Memory info
            'n_act_memory': len(state['act_memory']),
            'n_agg_memory': len(state['agg_memory']),
            'n_act_founders': n_act_founders,
            'n_agg_founders': n_agg_founders,
            'sin_has_founder_memory': (4 in state['act_memory'] and
                                       state['act_memory'][4].get('is_founder', False)),
            # Affinity
            'sin_affinity': float(state['act_affinity'][4]),
            'avg_act_affinity': float(jnp.mean(state['act_affinity'])),
            'avg_agg_affinity': float(jnp.mean(state['agg_affinity'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
        }
