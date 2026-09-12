"""Strategy 91: Clonal Selection + Tag-Homeostatic Hybrid.

Combines tag_homeostatic_dual with clonal selection:
- Base: Strategy 84 (Tag+Homeostatic) - 67% Parity-5 solve, 100% sin retention
- Extension: Strategy 29 (Clonal Selection) - Immune-inspired affinity learning

Key innovation: Captured functions are protected from hypermutation. The immune
diversity pool provides exploration while tagging protects valuable discoveries.

Bio inspiration: Immune system maintains a diverse repertoire of antibodies
while protecting high-affinity clones. Combined with synaptic tagging, this
creates a dual protection mechanism: affinity-based AND tag-based.

Expected: Better exploration (clonal diversity) with robust retention (capture protection).
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class ClonalHybridDualStrategy(PaletteEvolutionStrategy):
    """Tag-Homeostatic base with clonal selection extension.

    Hybrid combining:
    - Tag+Homeostatic (84): Tag-and-capture + homeostatic balance
    - Clonal Selection (29): Immune-inspired affinity and hypermutation

    Critical interaction: Captured functions are protected from hypermutation,
    creating a stable core while the rest of the repertoire explores.
    """

    name = "clonal_hybrid_dual"
    description = "Dual: Tag+Homeostatic base with immune clonal selection"

    def __init__(
        self,
        # === Clonal selection parameters (from strategy 29) ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        proliferation_rate: float = 0.25,
        hypermutation_rate: float = 0.08,
        hypermutation_strength: float = 0.2,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Tagging parameters (from strategy 84/81) ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        captured_hypermutation_protection: float = 0.9,  # NEW: Captured = no hypermutation
        # === Homeostatic parameters (from strategy 84/82) ===
        target_extreme_ratio: float = 0.60,
        imbalance_threshold: float = 0.15,
        discovery_bonus: float = 0.5,
        extreme_protection: float = 0.6,
        sin_protection: float = 0.5,
        # === Cross-domain parameters ===
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
        # Alias for compatibility with test scripts
        initial_palette: List[int] = None,
    ):
        """Initialize Clonal+Tag+Homeostatic hybrid strategy."""
        # Handle initial_palette alias
        if initial_palette is not None and initial_act_palette is None:
            initial_act_palette = initial_palette

        # Clonal selection
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.proliferation_rate = proliferation_rate
        self.hypermutation_rate = hypermutation_rate
        self.hypermutation_strength = hypermutation_strength
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost
        self.captured_hypermutation_protection = captured_hypermutation_protection

        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.imbalance_threshold = imbalance_threshold
        self.discovery_bonus = discovery_bonus
        self.extreme_protection = extreme_protection
        self.sin_protection = sin_protection

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with clonal + tagging + homeostatic tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Clonal selection state: affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity (from tag_homeostatic)
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Clonal selection
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            'act_expansions': jnp.zeros(NUM_ACTIVATIONS),
            'agg_expansions': jnp.zeros(NUM_AGGREGATIONS),
            'act_hypermutations': jnp.zeros(NUM_ACTIVATIONS),
            'agg_hypermutations': jnp.zeros(NUM_AGGREGATIONS),
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'capture_events': 0,
            'hypermutation_events': 0,
            'diversity_rescues': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 910000),
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

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags for active functions."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:  # Sin boost
                    tag_strength *= self.extreme_tag_boost
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Attempt to capture tagged functions on improvement."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

        return new_act_captured, new_agg_captured, capture_count

    def _update_affinities(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Update affinities with hypermutation (captured protected)."""
        k1, k2, k3 = jax.random.split(key, 3)
        hypermutation_count = 0

        # Decay and learn
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        # Learning boost from fitness
        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * fitness_delta)
                    )
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta)
                    )

        # Hypermutation (but NOT for captured functions)
        hypermut_probs_act = jax.random.uniform(k1, (NUM_ACTIVATIONS,))
        hypermut_probs_agg = jax.random.uniform(k2, (NUM_AGGREGATIONS,))
        hypermut_amounts_act = jax.random.normal(k3, (NUM_ACTIVATIONS,)) * self.hypermutation_strength
        hypermut_amounts_agg = jax.random.normal(jax.random.split(k3)[0], (NUM_AGGREGATIONS,)) * self.hypermutation_strength

        for i in range(NUM_ACTIVATIONS):
            if hypermut_probs_act[i] < self.hypermutation_rate:
                # Captured = protected from hypermutation
                if act_captured[i] > 0.5:
                    effective_rate = self.hypermutation_rate * (1 - self.captured_hypermutation_protection)
                    if hypermut_probs_act[i] >= effective_rate:
                        continue  # Skip hypermutation
                new_act_aff = new_act_aff.at[i].set(
                    jnp.clip(new_act_aff[i] + hypermut_amounts_act[i], 0.0, 1.0)
                )
                hypermutation_count += 1

        for j in range(NUM_AGGREGATIONS):
            if hypermut_probs_agg[j] < self.hypermutation_rate:
                # Captured = protected from hypermutation
                if agg_captured[j] > 0.5:
                    effective_rate = self.hypermutation_rate * (1 - self.captured_hypermutation_protection)
                    if hypermut_probs_agg[j] >= effective_rate:
                        continue
                new_agg_aff = new_agg_aff.at[j].set(
                    jnp.clip(new_agg_aff[j] + hypermut_amounts_agg[j], 0.0, 1.0)
                )
                hypermutation_count += 1

        # Cross-domain affinity update
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(act_active, agg_active)
            delta = self.cross_learning_rate * fitness_delta * co_active
            # Extra boost for sin-extreme pairs
            for i in [4]:  # sin
                for j in CORE_EXTREME_AGGS:
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        delta = delta.at[i, j].set(delta[i, j] * (1 + self.sin_extreme_affinity_boost))
            new_cross = jnp.clip(new_cross + delta, 0.0, 1.0)

        return new_act_aff, new_agg_aff, new_cross, hypermutation_count

    def _select_palette_by_affinity(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        n_funcs: int,
        min_active: int,
        max_active: int,
        min_diversity: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette based on affinity with capture protection and diversity."""
        # Combined score: affinity + capture + tag
        score = affinities + captured * 0.3 + tags * 0.2

        # Preference boost
        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + self.discovery_bonus)

        # Top-k selection
        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Diversity rescue: ensure minimum different functions
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < min_diversity:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive), shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute extreme/averaging ratio for homeostatic tracking."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with combined clonal + tag + homeostatic mechanisms."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags']
        )

        # Update tag history
        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE ===
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            new_tag_history, generation, improved
        )

        # === AFFINITY UPDATE WITH HYPERMUTATION ===
        new_act_aff, new_agg_aff, new_cross_affinity, hypermut_count = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], state['agg_mask'],
            new_act_captured, new_agg_captured,
            state['cross_affinity'], fitness_delta, k1
        )

        # === PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_palette_by_affinity(
            new_act_aff, new_act_captured, new_act_tags,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            self.min_diversity_act, k2, prefer_indices=[4]  # Prefer sin
        )
        new_agg_mask, agg_diversity_rescue = self._select_palette_by_affinity(
            new_agg_aff, new_agg_captured, new_agg_tags,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            self.min_diversity_agg, jax.random.split(k2)[0],
            prefer_indices=list(CORE_EXTREME_AGGS)
        )

        # === HOMEOSTATIC TRACKING ===
        extreme_ratio = self._compute_extreme_ratio(new_agg_mask)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_expansions': state['act_expansions'],
            'agg_expansions': state['agg_expansions'],
            'act_hypermutations': state['act_hypermutations'] + hypermut_count,
            'agg_hypermutations': state['agg_hypermutations'],
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            'capture_events': state['capture_events'] + capture_count,
            'hypermutation_events': state['hypermutation_events'] + hypermut_count,
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue + agg_diversity_rescue,
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
            # Clonal selection metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'hypermutation_events': new_state['hypermutation_events'],
            'diversity_rescues': new_state['diversity_rescues'],
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_affinity': float(new_act_aff[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Homeostatic metrics
            'extreme_ratio': extreme_ratio,
            # Cross-domain
            'sin_max_affinity': float(new_cross_affinity[4, 2]),
            'sin_min_affinity': float(new_cross_affinity[4, 3]),
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with clonal + tag + homeostatic status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            # Clonal
            'act_mean_affinity': float(jnp.mean(state['act_affinities'])),
            'agg_mean_affinity': float(jnp.mean(state['agg_affinities'])),
            'hypermutation_events': state['hypermutation_events'],
            'diversity_rescues': state['diversity_rescues'],
            # Tagging
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'sin_affinity': float(state['act_affinities'][4]),
            'capture_events': state['capture_events'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
