"""Strategy 94: Clonal Dendritic Dual.

Combines clonal_hybrid_dual with dendritic computation:
- Base: Strategy 91 (Clonal+Tag+Homeostatic) - 100% Parity-5, 100% sin retention
- Extension: Strategy 28 (Dendritic Computation) - Zone-based local processing

Key Innovation: Zone-based protection - sin's zone-mates (burst, resonator)
act as buffer. When the oscillatory zone captures, ALL its members get
protection, not just sin. This creates redundant pathways.

Zone Assignments for Dual Palette:
  Activations:
    - Zone 0 (oscillatory): sin(4), burst(11), resonator(12), osc_adapt(13), receptive(15)
    - Zone 1 (monotonic): identity(3), tanh(0), sigmoid(1), relu(2)
    - Zone 2 (spatial): gauss(5), softplus(7), band_pass(16), integrate(17)
    - Zone 3 (nonlinear): lelu(6), rs_adapt(8), fs_fast(9), lts_low(10), gain_mod(14)

  Aggregations:
    - Zone 0 (extreme): max(2), min(3), maxabs(5)
    - Zone 1 (averaging): sum(0), mean(1), product(4)

Bio inspiration: Dendritic compartments compute locally. When one member
of a compartment is valuable, the entire compartment gets protected,
creating redundant discovery pathways.

Expected: More robust protection through zone-level capture.
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
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class ClonalDendriticDualStrategy(PaletteEvolutionStrategy):
    """Clonal selection with zone-based dendritic protection.

    Extend clonal_hybrid_dual
    with dendritic computation's zone-based organization.

    Critical innovation: Zone capture - when a function captures, its
    entire zone gets partial protection. This creates redundant discovery
    pathways so losing one function doesn't lose the whole zone.
    """

    name = "clonal_dendritic_dual"
    description = "Dual: Clonal+Tag base with zone-based dendritic protection"

    # Activation zone assignments
    ACT_ZONES = {
        # Zone 0: Oscillatory (sin's zone)
        4: 0, 11: 0, 12: 0, 13: 0, 15: 0,  # sin, burst, resonator, osc_adapt, receptive
        # Zone 1: Monotonic
        0: 1, 1: 1, 2: 1, 3: 1,  # tanh, sigmoid, relu, identity
        # Zone 2: Spatial
        5: 2, 7: 2, 16: 2, 17: 2,  # gauss, softplus, band_pass, integrate
        # Zone 3: Nonlinear
        6: 3, 8: 3, 9: 3, 10: 3, 14: 3,  # lelu, rs_adapt, fs_fast, lts_low, gain_mod
    }
    ACT_N_ZONES = 4
    ACT_ZONE_NAMES = ['oscillatory', 'monotonic', 'spatial', 'nonlinear']

    # Aggregation zone assignments
    AGG_ZONES = {
        # Zone 0: Extreme
        2: 0, 3: 0, 5: 0,  # max, min, maxabs
        # Zone 1: Averaging
        0: 1, 1: 1, 4: 1,  # sum, mean, product
    }
    AGG_N_ZONES = 2
    AGG_ZONE_NAMES = ['extreme', 'averaging']

    def __init__(
        self,
        # === Clonal selection parameters (from strategy 91) ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        hypermutation_rate: float = 0.08,
        hypermutation_strength: float = 0.2,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Tagging parameters (from strategy 91) ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        captured_hypermutation_protection: float = 0.9,
        # === ZONE-BASED PARAMETERS (NEW from strategy 28) ===
        zone_capture_threshold: float = 0.6,       # Zone capture requires high zone activity
        zone_capture_spread: float = 0.4,          # How much capture spreads to zone-mates
        zone_hypermutation_protection: float = 0.7,  # Zone-mates get partial protection
        zone_learning_rate: float = 0.15,          # Zone memory learning rate
        zone_decay: float = 0.92,                  # Zone memory decay
        # === Homeostatic parameters ===
        target_extreme_ratio: float = 0.60,
        discovery_bonus: float = 0.5,
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
    ):
        """Initialize Clonal+Dendritic hybrid strategy."""
        # Clonal selection
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
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
        self.captured_hypermutation_protection = captured_hypermutation_protection

        # Zone-based (NEW)
        self.zone_capture_threshold = zone_capture_threshold
        self.zone_capture_spread = zone_capture_spread
        self.zone_hypermutation_protection = zone_hypermutation_protection
        self.zone_learning_rate = zone_learning_rate
        self.zone_decay = zone_decay

        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.discovery_bonus = discovery_bonus

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

        # Build zone member lists
        self._build_zone_members()

    def _build_zone_members(self):
        """Build zone member lists for quick lookup."""
        self.act_zone_members = {z: [] for z in range(self.ACT_N_ZONES)}
        for func_idx, zone_idx in self.ACT_ZONES.items():
            if func_idx < NUM_ACTIVATIONS:
                self.act_zone_members[zone_idx].append(func_idx)

        self.agg_zone_members = {z: [] for z in range(self.AGG_N_ZONES)}
        for func_idx, zone_idx in self.AGG_ZONES.items():
            if func_idx < NUM_AGGREGATIONS:
                self.agg_zone_members[zone_idx].append(func_idx)

    def _get_act_zone(self, func_idx: int) -> int:
        """Get zone for activation function."""
        return self.ACT_ZONES.get(func_idx, 0)

    def _get_agg_zone(self, func_idx: int) -> int:
        """Get zone for aggregation function."""
        return self.AGG_ZONES.get(func_idx, 0)

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with clonal + zone tracking."""
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

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Zone memories (NEW)
        act_zone_memories = jnp.zeros(self.ACT_N_ZONES)
        agg_zone_memories = jnp.zeros(self.AGG_N_ZONES)
        act_zone_captured = jnp.zeros(self.ACT_N_ZONES)  # Zone-level capture
        agg_zone_captured = jnp.zeros(self.AGG_N_ZONES)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Clonal
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Zone (NEW)
            'act_zone_memories': act_zone_memories,
            'agg_zone_memories': agg_zone_memories,
            'act_zone_captured': act_zone_captured,
            'agg_zone_captured': agg_zone_captured,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'capture_events': 0,
            'zone_capture_events': 0,
            'hypermutation_events': 0,
            'diversity_rescues': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 940000),
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

    def _update_zone_memories(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_zone_memories: jnp.ndarray,
        agg_zone_memories: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update zone memories based on zone activity and fitness."""
        new_act_zones = act_zone_memories * self.zone_decay
        new_agg_zones = agg_zone_memories * self.zone_decay

        if fitness_delta > 0:
            # Credit active zones
            for z in range(self.ACT_N_ZONES):
                zone_active = sum(1 for i in self.act_zone_members[z] if act_mask[i] > 0.5)
                if zone_active > 0:
                    new_act_zones = new_act_zones.at[z].set(
                        min(1.0, new_act_zones[z] + self.zone_learning_rate * fitness_delta)
                    )

            for z in range(self.AGG_N_ZONES):
                zone_active = sum(1 for i in self.agg_zone_members[z] if agg_mask[i] > 0.5)
                if zone_active > 0:
                    new_agg_zones = new_agg_zones.at[z].set(
                        min(1.0, new_agg_zones[z] + self.zone_learning_rate * fitness_delta)
                    )

        return new_act_zones, new_agg_zones

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
                # Boost for oscillatory zone (sin's zone)
                if self._get_act_zone(i) == 0:
                    tag_strength *= 1.3
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                # Boost for extreme zone
                if self._get_agg_zone(j) == 0:
                    tag_strength *= 1.3
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture_with_zone_spread(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        act_zone_captured: jnp.ndarray,
        agg_zone_captured: jnp.ndarray,
        act_zone_memories: jnp.ndarray,
        agg_zone_memories: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]:
        """Attempt capture with zone-level spread.

        KEY INNOVATION: When a function captures AND its zone memory is high,
        the entire zone gets partial capture status. This protects zone-mates.

        Returns: (new_act_captured, new_agg_captured, new_act_zone_captured,
                  new_agg_zone_captured, capture_count, zone_capture_count)
        """
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        new_act_zone_captured = act_zone_captured.copy()
        new_agg_zone_captured = agg_zone_captured.copy()
        capture_count = 0
        zone_capture_count = 0

        if not improved:
            return (new_act_captured, new_agg_captured, new_act_zone_captured,
                    new_agg_zone_captured, 0, 0)

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                # Capture activations
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

                        # ZONE SPREAD: If zone memory is high, capture zone
                        zone = self._get_act_zone(i)
                        if (act_zone_memories[zone] > self.zone_capture_threshold
                            and new_act_zone_captured[zone] < 0.5):
                            new_act_zone_captured = new_act_zone_captured.at[zone].set(1.0)
                            zone_capture_count += 1
                            # Spread partial capture to zone-mates
                            for mate in self.act_zone_members[zone]:
                                if new_act_captured[mate] < 0.5:
                                    new_act_captured = new_act_captured.at[mate].set(
                                        self.zone_capture_spread
                                    )

                # Capture aggregations
                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

                        # Zone spread for aggregations
                        zone = self._get_agg_zone(j)
                        if (agg_zone_memories[zone] > self.zone_capture_threshold
                            and new_agg_zone_captured[zone] < 0.5):
                            new_agg_zone_captured = new_agg_zone_captured.at[zone].set(1.0)
                            zone_capture_count += 1
                            for mate in self.agg_zone_members[zone]:
                                if new_agg_captured[mate] < 0.5:
                                    new_agg_captured = new_agg_captured.at[mate].set(
                                        self.zone_capture_spread
                                    )

        return (new_act_captured, new_agg_captured, new_act_zone_captured,
                new_agg_zone_captured, capture_count, zone_capture_count)

    def _update_affinities(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        act_zone_captured: jnp.ndarray,
        agg_zone_captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Update affinities with zone-aware hypermutation protection."""
        k1, k2, k3 = jax.random.split(key, 3)
        hypermutation_count = 0

        # Decay
        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        # Learning
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

        # Hypermutation with zone-aware protection
        hypermut_probs_act = jax.random.uniform(k1, (NUM_ACTIVATIONS,))
        hypermut_probs_agg = jax.random.uniform(k2, (NUM_AGGREGATIONS,))
        hypermut_amounts = jax.random.normal(k3, (NUM_ACTIVATIONS + NUM_AGGREGATIONS,)) * self.hypermutation_strength

        for i in range(NUM_ACTIVATIONS):
            if hypermut_probs_act[i] < self.hypermutation_rate:
                protection = 0.0
                # Individual capture protection
                if act_captured[i] > 0.5:
                    protection = max(protection, self.captured_hypermutation_protection)
                # Zone capture protection
                zone = self._get_act_zone(i)
                if act_zone_captured[zone] > 0.5:
                    protection = max(protection, self.zone_hypermutation_protection)

                effective_rate = self.hypermutation_rate * (1 - protection)
                if hypermut_probs_act[i] >= effective_rate:
                    continue
                new_act_aff = new_act_aff.at[i].set(
                    jnp.clip(new_act_aff[i] + hypermut_amounts[i], 0.0, 1.0)
                )
                hypermutation_count += 1

        for j in range(NUM_AGGREGATIONS):
            if hypermut_probs_agg[j] < self.hypermutation_rate:
                protection = 0.0
                if agg_captured[j] > 0.5:
                    protection = max(protection, self.captured_hypermutation_protection)
                zone = self._get_agg_zone(j)
                if agg_zone_captured[zone] > 0.5:
                    protection = max(protection, self.zone_hypermutation_protection)

                effective_rate = self.hypermutation_rate * (1 - protection)
                if hypermut_probs_agg[j] >= effective_rate:
                    continue
                new_agg_aff = new_agg_aff.at[j].set(
                    jnp.clip(new_agg_aff[j] + hypermut_amounts[NUM_ACTIVATIONS + j], 0.0, 1.0)
                )
                hypermutation_count += 1

        # Cross-domain update
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(act_active, agg_active)
            delta = self.cross_learning_rate * fitness_delta * co_active
            for i in [4]:  # sin
                for j in CORE_EXTREME_AGGS:
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        delta = delta.at[i, j].set(delta[i, j] * (1 + self.sin_extreme_affinity_boost))
            new_cross = jnp.clip(new_cross + delta, 0.0, 1.0)

        return new_act_aff, new_agg_aff, new_cross, hypermutation_count

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        zone_captured: jnp.ndarray,
        n_funcs: int,
        n_zones: int,
        zone_assignments: Dict[int, int],
        min_active: int,
        max_active: int,
        min_diversity: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette with zone capture bonus."""
        # Score: affinity + capture + tag + zone capture bonus
        score = affinities + captured * 0.3 + tags * 0.2

        # Zone capture bonus
        for i in range(n_funcs):
            zone = zone_assignments.get(i, 0)
            if zone < n_zones and zone_captured[zone] > 0.5:
                score = score.at[i].set(score[i] + 0.15)

        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + self.discovery_bonus)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Diversity rescue
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
        """Compute extreme/averaging ratio."""
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
        """Update with clonal + zone mechanisms."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === ZONE MEMORY UPDATE ===
        new_act_zone_mem, new_agg_zone_mem = self._update_zone_memories(
            state['act_mask'], state['agg_mask'],
            state['act_zone_memories'], state['agg_zone_memories'],
            fitness_delta
        )

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags']
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE WITH ZONE SPREAD ===
        (new_act_captured, new_agg_captured, new_act_zone_captured,
         new_agg_zone_captured, capture_count, zone_capture_count) = \
            self._attempt_capture_with_zone_spread(
                new_act_tags, new_agg_tags,
                state['act_captured'], state['agg_captured'],
                state['act_zone_captured'], state['agg_zone_captured'],
                new_act_zone_mem, new_agg_zone_mem,
                new_tag_history, generation, improved
            )

        # === AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross_affinity, hypermut_count = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], state['agg_mask'],
            new_act_captured, new_agg_captured,
            new_act_zone_captured, new_agg_zone_captured,
            state['cross_affinity'], fitness_delta, k1
        )

        # === PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_palette(
            new_act_aff, new_act_captured, new_act_tags, new_act_zone_captured,
            NUM_ACTIVATIONS, self.ACT_N_ZONES, self.ACT_ZONES,
            self.min_active_act, self.max_active_act, self.min_diversity_act,
            k2, prefer_indices=[4]
        )
        new_agg_mask, agg_diversity_rescue = self._select_palette(
            new_agg_aff, new_agg_captured, new_agg_tags, new_agg_zone_captured,
            NUM_AGGREGATIONS, self.AGG_N_ZONES, self.AGG_ZONES,
            self.min_active_agg, self.max_active_agg, self.min_diversity_agg,
            jax.random.split(k2)[0], prefer_indices=list(CORE_EXTREME_AGGS)
        )

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
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'act_zone_memories': new_act_zone_mem,
            'agg_zone_memories': new_agg_zone_mem,
            'act_zone_captured': new_act_zone_captured,
            'agg_zone_captured': new_agg_zone_captured,
            'cross_affinity': new_cross_affinity,
            'capture_events': state['capture_events'] + capture_count,
            'zone_capture_events': state['zone_capture_events'] + zone_capture_count,
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

        # Dominant zones
        dominant_act_zone = int(jnp.argmax(new_act_zone_mem))
        dominant_agg_zone = int(jnp.argmax(new_agg_zone_mem))

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Zone metrics (NEW)
            'dominant_act_zone': self.ACT_ZONE_NAMES[dominant_act_zone],
            'dominant_agg_zone': self.AGG_ZONE_NAMES[dominant_agg_zone],
            'oscillatory_zone_captured': bool(new_act_zone_captured[0] > 0.5),
            'extreme_zone_captured': bool(new_agg_zone_captured[0] > 0.5),
            'zone_capture_events': new_state['zone_capture_events'],
            # Clonal metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'hypermutation_events': new_state['hypermutation_events'],
            'diversity_rescues': new_state['diversity_rescues'],
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_affinity': float(new_act_aff[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Homeostatic
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
        """Return state summary with zone info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        dominant_act = int(jnp.argmax(state['act_zone_memories']))
        dominant_agg = int(jnp.argmax(state['agg_zone_memories']))

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            # Zone status
            'dominant_act_zone': self.ACT_ZONE_NAMES[dominant_act],
            'dominant_agg_zone': self.AGG_ZONE_NAMES[dominant_agg],
            'oscillatory_zone_captured': bool(state['act_zone_captured'][0] > 0.5),
            'extreme_zone_captured': bool(state['agg_zone_captured'][0] > 0.5),
            'zone_capture_events': state['zone_capture_events'],
            # Clonal
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'sin_affinity': float(state['act_affinities'][4]),
            'capture_events': state['capture_events'],
            'hypermutation_events': state['hypermutation_events'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
