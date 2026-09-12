"""Strategy 115: Immune + Clonal + Cross V2 Dual.

Combines immune memory (strategy 44), clonal hybrid (strategy 91), and
cross_domain_v2 (strategy 103) for TRIPLE PROTECTION of sin-extreme.

Key Innovation:
- AFFINITY-based protection (clonal selection)
- CAPTURE-based protection (tag and capture)
- MEMORY-based protection (immune memory cells)
- Three independent protection mechanisms = near-impossible forgetting

Bio inspiration: Immune system with multiple defense layers.
Clonal selection, memory cells, and cross-reactivity all protect antigens.
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


class ImmuneClonalCrossV2DualStrategy(PaletteEvolutionStrategy):
    """Triple protection: affinity + capture + memory.

    Three independent protection mechanisms
    for sin-extreme make forgetting nearly impossible.
    """

    name = "immune_clonal_cross_v2_dual"
    description = "Dual: Triple protection (affinity + capture + memory) for sin-extreme"

    def __init__(
        self,
        # === CLONAL AFFINITY PARAMETERS ===
        clonal_affinity_lr: float = 0.12,
        affinity_decay: float = 0.98,
        hypermutation_rate: float = 0.08,
        # === MEMORY PARAMETERS (KEY) ===
        memory_cell_lifespan: int = 30,
        memory_formation_threshold: float = 0.7,
        memory_clonal_interaction: bool = True,
        # === CROSS-DOMAIN PARAMETERS ===
        cross_affinity_growth: float = 1.15,
        sin_extreme_multiplier: float = 1.4,
        # === TRIPLE PROTECTION THRESHOLD (KEY) ===
        triple_protection_threshold: float = 0.8,
        # === Tagging parameters ===
        tag_threshold: float = 0.5,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        # === Exploration ===
        exploration_rate: float = 0.20,
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
        """Initialize Immune Clonal Cross V2 Dual strategy."""
        # Clonal
        self.clonal_affinity_lr = clonal_affinity_lr
        self.affinity_decay = affinity_decay
        self.hypermutation_rate = hypermutation_rate

        # Memory (KEY)
        self.memory_cell_lifespan = memory_cell_lifespan
        self.memory_formation_threshold = memory_formation_threshold
        self.memory_clonal_interaction = memory_clonal_interaction

        # Cross-domain
        self.cross_affinity_growth = cross_affinity_growth
        self.sin_extreme_multiplier = sin_extreme_multiplier

        # Triple protection (KEY)
        self.triple_protection_threshold = triple_protection_threshold

        # Tagging
        self.tag_threshold = tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window

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
        """Initialize state with triple protection tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities (Protection Layer 1)
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # Tagging/Capture (Protection Layer 2)
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Memory (Protection Layer 3)
        # {idx: {'formation_gen': int, 'protection': float}}
        act_memory = {}
        agg_memory = {}

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities (Layer 1)
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tags/Capture (Layer 2)
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Memory (Layer 3)
            'act_memory': act_memory,
            'agg_memory': agg_memory,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'triple_protection_events': 0,
            'memory_formations': 0,
            'capture_events': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1150000),
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

    def _update_memory(
        self,
        memory: Dict[int, Dict],
        mask: jnp.ndarray,
        affinities: jnp.ndarray,
        best_fitness: float,
        generation: int,
        n_funcs: int,
    ) -> Tuple[Dict, int]:
        """Update memory cells (Protection Layer 3)."""
        new_memory = dict(memory)
        formations = 0

        # Form memory for high-affinity, active functions
        if best_fitness >= self.memory_formation_threshold:
            for i in range(n_funcs):
                if mask[i] > 0.5 and affinities[i] > 0.6:
                    if i not in new_memory:
                        new_memory[i] = {
                            'formation_gen': generation,
                            'protection': 0.9,
                        }
                        formations += 1
                    else:
                        # Boost existing memory
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

    def _update_tags_and_capture(
        self,
        mask: jnp.ndarray,
        tags: jnp.ndarray,
        captured: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Update tags and capture (Protection Layer 2)."""
        n_funcs = len(tags)
        new_tags = tags * self.tag_decay
        new_captured = captured.copy()
        capture_count = 0

        # Tag active functions
        for i in range(n_funcs):
            if mask[i] > 0.5:
                new_tags = new_tags.at[i].set(min(1.0, new_tags[i] + 0.3))

        # Capture on improvement
        if improved:
            for hist_gen, hist_tags in tag_history:
                if generation - hist_gen <= self.capture_window:
                    for i in range(n_funcs):
                        if hist_tags[i] > self.tag_threshold and new_captured[i] < 0.5:
                            new_captured = new_captured.at[i].set(1.0)
                            capture_count += 1

        return new_tags, new_captured, capture_count

    def _compute_protection_score(
        self,
        idx: int,
        affinity: float,
        captured: float,
        memory: Dict,
    ) -> Tuple[float, int]:
        """Compute combined protection score from all 3 layers."""
        score = 0.0
        layers_active = 0

        # Layer 1: Affinity
        if affinity > 0.6:
            score += 0.3
            layers_active += 1

        # Layer 2: Capture
        if captured > 0.5:
            score += 0.3
            layers_active += 1

        # Layer 3: Memory
        if idx in memory:
            score += 0.3 * memory[idx]['protection']
            layers_active += 1

        # Triple protection bonus
        if layers_active >= 3:
            score += 0.2

        return score, layers_active

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        memory: Dict,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        stagnation: int,
        generation: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int, int]:
        """Select palette using combined protection score."""
        k1, k2 = jax.random.split(key)

        # Compute scores with triple protection
        score = jnp.zeros(n_funcs)
        triple_protections = 0

        for i in range(n_funcs):
            protection, layers = self._compute_protection_score(
                i, float(affinities[i]), float(captured[i]), memory
            )
            base_score = affinities[i] + tags[i] * 0.15 + protection
            score = score.at[i].set(base_score)
            if layers >= 3:
                triple_protections += 1

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

        return mask, 0, triple_protections

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_delta: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity with multiplicative growth."""
        new_cross = cross_affinity.copy()

        if fitness_delta > 0:
            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(act_active, agg_active)

            # Multiplicative growth for sin-extreme pairs
            growth = self.cross_affinity_growth
            for j in CORE_EXTREME_AGGS:
                if agg_active[j] > 0.5 and act_active[4] > 0.5:
                    growth = self.cross_affinity_growth * self.sin_extreme_multiplier

            delta = 0.1 * fitness_delta * co_active
            new_cross = jnp.clip(new_cross * (1 + delta * (growth - 1)), 0.0, 1.0)

        return new_cross

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with triple protection dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === LAYER 1: AFFINITY UPDATE ===
        new_act_aff = state['act_affinities'] * self.affinity_decay
        new_agg_aff = state['agg_affinities'] * self.affinity_decay
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if state['act_mask'][i] > 0.5:
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.clonal_affinity_lr * fitness_delta)
                    )
            for j in range(NUM_AGGREGATIONS):
                if state['agg_mask'][j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.clonal_affinity_lr * fitness_delta)
                    )

        # === LAYER 2: TAGS AND CAPTURE ===
        new_act_tags, new_act_captured, act_captures = self._update_tags_and_capture(
            state['act_mask'], state['act_tags'], state['act_captured'],
            state['tag_history'], generation, improved
        )
        new_agg_tags, new_agg_captured, agg_captures = self._update_tags_and_capture(
            state['agg_mask'], state['agg_tags'], state['agg_captured'],
            [(g, t) for g, t in state['tag_history'][-5:]],
            generation, improved
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === LAYER 3: MEMORY UPDATE ===
        new_act_memory, act_formations = self._update_memory(
            state['act_memory'], state['act_mask'], new_act_aff,
            best_fitness, generation, NUM_ACTIVATIONS
        )
        new_agg_memory, agg_formations = self._update_memory(
            state['agg_memory'], state['agg_mask'], new_agg_aff,
            best_fitness, generation, NUM_AGGREGATIONS
        )

        # === CROSS-DOMAIN UPDATE ===
        new_cross = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], fitness_delta
        )

        # === PALETTE SELECTION ===
        new_act_mask, _, act_triple = self._select_palette(
            new_act_aff, new_act_captured, new_act_tags, new_act_memory,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            self.min_diversity_act, new_stagnation, generation, k1,
            prefer_indices=[4]
        )
        new_agg_mask, _, agg_triple = self._select_palette(
            new_agg_aff, new_agg_captured, new_agg_tags, new_agg_memory,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            self.min_diversity_agg, new_stagnation, generation, k2,
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
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'act_memory': new_act_memory,
            'agg_memory': new_agg_memory,
            'cross_affinity': new_cross,
            'triple_protection_events': state['triple_protection_events'] + act_triple + agg_triple,
            'memory_formations': state['memory_formations'] + act_formations + agg_formations,
            'capture_events': state['capture_events'] + act_captures + agg_captures,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Check sin's protection layers
        sin_affinity = float(new_act_aff[4]) > 0.6
        sin_captured = float(new_act_captured[4]) > 0.5
        sin_memory = 4 in new_act_memory
        sin_layers = sum([sin_affinity, sin_captured, sin_memory])

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Triple protection metrics (KEY)
            'sin_protection_layers': sin_layers,
            'sin_affinity_protected': sin_affinity,
            'sin_capture_protected': sin_captured,
            'sin_memory_protected': sin_memory,
            'triple_protection_events': new_state['triple_protection_events'],
            # Memory metrics
            'n_act_memory': len(new_act_memory),
            'n_agg_memory': len(new_agg_memory),
            'memory_formations': new_state['memory_formations'],
            # Capture metrics
            'capture_events': new_state['capture_events'],
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

        sin_affinity = float(state['act_affinities'][4]) > 0.6
        sin_captured = float(state['act_captured'][4]) > 0.5
        sin_memory = 4 in state['act_memory']

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'sin_protection_layers': sum([sin_affinity, sin_captured, sin_memory]),
            'triple_protection_events': state['triple_protection_events'],
            'generation': state['generation'],
        }
