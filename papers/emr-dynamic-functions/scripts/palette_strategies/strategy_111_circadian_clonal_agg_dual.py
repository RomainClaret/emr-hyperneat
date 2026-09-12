"""Strategy 111: Circadian + Clonal + Aggregation Dual.

Combines circadian rhythm oscillations (strategy 34) with clonal hybrid dual
(strategy 91) for phase-synchronized sin-extreme discovery.

Key Innovation:
- Sin and extreme aggregations are PHASE-COUPLED
- When sin peaks, extremes are forced to peak with it
- Phase synchronization creates automatic co-activation
- Clonal selection provides diversity while phases ensure pairing

Bio inspiration: In the brain, circadian rhythms coordinate neural activity
across regions. Certain processes are phase-locked to peak together,
ensuring coordinated function. Sleep-wake cycles affect learning and memory
consolidation across multiple brain areas simultaneously.

Expected: Phase coupling ensures sin-extreme always activate together,
creating natural retention through temporal synchronization.
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


class CircadianClonalAggDualStrategy(PaletteEvolutionStrategy):
    """Circadian phase synchronization with clonal selection.

    Phase coupling ensures sin and extremes
    always peak together, creating automatic co-activation.

    Critical innovation: Sin-extreme phase coupling forces coordinated
    discovery and retention through temporal synchronization.
    """

    name = "circadian_clonal_agg_dual"
    description = "Dual: Circadian phase synchronization for sin-extreme co-activation"

    def __init__(
        self,
        # === PHASE COUPLING PARAMETERS (KEY INNOVATION) ===
        circadian_period: int = 15,                    # Gens per full cycle
        sin_extreme_phase_coupling: float = 0.9,       # How tightly phases lock
        phase_learning_rate: float = 0.15,             # Phase adaptation speed
        phase_noise: float = 0.05,                     # Phase drift
        activity_threshold: float = 0.4,               # Minimum for palette
        # === Clonal selection (from strategy 91) ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        hypermutation_rate: float = 0.08,
        hypermutation_strength: float = 0.2,
        # === Tagging (from strategy 91) ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        captured_hypermutation_protection: float = 0.9,
        # === Cross-domain ===
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
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
        """Initialize Circadian + Clonal + Aggregation strategy."""
        # Circadian phase coupling (KEY INNOVATION)
        self.circadian_period = circadian_period
        self.sin_extreme_phase_coupling = sin_extreme_phase_coupling
        self.phase_learning_rate = phase_learning_rate
        self.phase_noise = phase_noise
        self.activity_threshold = activity_threshold

        # Clonal selection
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.hypermutation_rate = hypermutation_rate
        self.hypermutation_strength = hypermutation_strength

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.captured_hypermutation_protection = captured_hypermutation_protection

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

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

    def _initialize_phases(
        self,
        key: jax.random.PRNGKey,
        initial_act: List[int],
        initial_agg: List[int],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize phases with sin-extreme coupling."""
        k1, k2 = jax.random.split(key)

        # Random initial phases
        act_phases = jax.random.uniform(k1, (NUM_ACTIVATIONS,)) * 2 * jnp.pi
        agg_phases = jax.random.uniform(k2, (NUM_AGGREGATIONS,)) * 2 * jnp.pi

        # PHASE COUPLING: Sin starts at phase 0
        act_phases = act_phases.at[4].set(0.0)

        # Extreme aggregations are phase-locked to sin
        for j in CORE_EXTREME_AGGS:
            agg_phases = agg_phases.at[j].set(0.0)

        return act_phases, agg_phases

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with circadian phases + clonal dynamics."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Clonal affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # Tagging
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        key = jax.random.PRNGKey(seed + 1110000)
        key, phase_key = jax.random.split(key)

        # CIRCADIAN PHASES (KEY)
        act_phases, agg_phases = self._initialize_phases(phase_key, initial_act, initial_agg)
        circadian_phase = 0.0

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Clonal affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # CIRCADIAN STATE (KEY)
            'circadian_phase': circadian_phase,
            'act_phases': act_phases,
            'agg_phases': agg_phases,
            'cycles_completed': 0,
            # Stats
            'capture_events': 0,
            'phase_sync_events': 0,
            'sin_extreme_coactivations': 0,
            'diversity_rescues': 0,
            # General
            'rng_key': key,
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

    def _compute_activity(
        self,
        circadian_phase: float,
        phases: jnp.ndarray,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute activity level based on phase alignment with clock."""
        activity = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            phase_diff = circadian_phase - float(phases[i])
            # Cosine oscillation: peaks when phase_diff = 0
            oscillation = (1 + jnp.cos(phase_diff)) / 2  # Range [0, 1]
            # Activity with baseline
            act = 0.2 + 0.8 * oscillation  # Range [0.2, 1.0]
            activity = activity.at[i].set(act)

        return activity

    def _adapt_phases_with_coupling(
        self,
        act_phases: jnp.ndarray,
        agg_phases: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        circadian_phase: float,
        improvement: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Adapt phases with SIN-EXTREME COUPLING.

        KEY INNOVATION: When sin or extremes are active with improvement,
        their phases are pulled together, creating forced co-activation.
        """
        k1, k2 = jax.random.split(key)
        phase_sync_events = 0

        new_act_phases = act_phases.copy()
        new_agg_phases = agg_phases.copy()

        # Add random drift
        noise_act = jax.random.normal(k1, (NUM_ACTIVATIONS,)) * self.phase_noise
        noise_agg = jax.random.normal(k2, (NUM_AGGREGATIONS,)) * self.phase_noise

        if improvement > 0:
            # Active functions entrain to clock
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    phase_diff = circadian_phase - float(act_phases[i])
                    adjustment = self.phase_learning_rate * improvement * jnp.sin(phase_diff)
                    new_act_phases = new_act_phases.at[i].set(
                        float(act_phases[i]) + adjustment
                    )

            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    phase_diff = circadian_phase - float(agg_phases[j])
                    adjustment = self.phase_learning_rate * improvement * jnp.sin(phase_diff)
                    new_agg_phases = new_agg_phases.at[j].set(
                        float(agg_phases[j]) + adjustment
                    )

            # SIN-EXTREME PHASE COUPLING
            # When sin is active, pull extreme phases toward sin's phase
            if act_mask[4] > 0.5:  # Sin is active
                sin_phase = float(new_act_phases[4])
                for j in CORE_EXTREME_AGGS:
                    if agg_mask[j] > 0.5:
                        # Strong coupling: pull extreme phase toward sin phase
                        phase_diff = sin_phase - float(new_agg_phases[j])
                        coupling_adjustment = (
                            self.sin_extreme_phase_coupling *
                            improvement *
                            jnp.sin(phase_diff)
                        )
                        new_agg_phases = new_agg_phases.at[j].set(
                            float(new_agg_phases[j]) + coupling_adjustment
                        )
                        phase_sync_events += 1

            # Also pull sin toward active extremes
            for j in CORE_EXTREME_AGGS:
                if agg_mask[j] > 0.5:
                    extreme_phase = float(new_agg_phases[j])
                    phase_diff = extreme_phase - float(new_act_phases[4])
                    coupling_adjustment = (
                        self.sin_extreme_phase_coupling * 0.5 *  # Weaker reverse
                        improvement *
                        jnp.sin(phase_diff)
                    )
                    new_act_phases = new_act_phases.at[4].set(
                        float(new_act_phases[4]) + coupling_adjustment
                    )

        # Apply noise
        new_act_phases = new_act_phases + noise_act
        new_agg_phases = new_agg_phases + noise_agg

        # Wrap to [0, 2π]
        new_act_phases = jnp.mod(new_act_phases, 2 * jnp.pi)
        new_agg_phases = jnp.mod(new_agg_phases, 2 * jnp.pi)

        return new_act_phases, new_agg_phases, phase_sync_events

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
                if i == 4:
                    tag_strength *= 1.3
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= 1.3
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
        """Attempt capture."""
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
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Update affinities with cross-domain learning."""
        sin_extreme_coactivations = 0

        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

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

        # Cross-domain update
        new_cross = cross_affinity.copy()
        if fitness_delta > 0:
            act_active = (act_mask > 0.5).astype(jnp.float32)
            agg_active = (agg_mask > 0.5).astype(jnp.float32)
            co_active = jnp.outer(act_active, agg_active)
            delta = self.cross_learning_rate * fitness_delta * co_active

            # Sin-extreme boost
            if act_mask[4] > 0.5:
                for j in CORE_EXTREME_AGGS:
                    if agg_mask[j] > 0.5:
                        delta = delta.at[4, j].set(
                            delta[4, j] * (1 + self.sin_extreme_affinity_boost)
                        )
                        sin_extreme_coactivations += 1

            new_cross = jnp.clip(new_cross + delta, 0.0, 1.0)

        return new_act_aff, new_agg_aff, new_cross, sin_extreme_coactivations

    def _select_palette_by_activity(
        self,
        activity: jnp.ndarray,
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
        """Select palette based on circadian activity + affinity."""
        # Combined score: activity + affinity + capture + tag
        score = activity * 0.5 + affinities * 0.3 + captured * 0.15 + tags * 0.05

        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + 0.5)

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

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with circadian phase dynamics + clonal selection."""
        key, k1, k2, k3 = jax.random.split(state['rng_key'], 4)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === ADVANCE CIRCADIAN CLOCK ===
        phase_increment = 2 * jnp.pi / self.circadian_period
        new_circadian_phase = state['circadian_phase'] + phase_increment
        cycles_completed = state['cycles_completed']
        if new_circadian_phase >= 2 * jnp.pi:
            new_circadian_phase = new_circadian_phase % (2 * jnp.pi)
            cycles_completed += 1

        # === ADAPT PHASES WITH COUPLING ===
        new_act_phases, new_agg_phases, phase_sync_events = self._adapt_phases_with_coupling(
            state['act_phases'], state['agg_phases'],
            state['act_mask'], state['agg_mask'],
            new_circadian_phase, fitness_delta, k1
        )

        # === COMPUTE ACTIVITY ===
        act_activity = self._compute_activity(new_circadian_phase, new_act_phases, NUM_ACTIVATIONS)
        agg_activity = self._compute_activity(new_circadian_phase, new_agg_phases, NUM_AGGREGATIONS)

        # === TAGGING ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags']
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE ===
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            new_tag_history, generation, improved
        )

        # === AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross, sin_extreme_coact = self._update_affinities(
            state['act_affinities'], state['agg_affinities'],
            state['act_mask'], state['agg_mask'],
            state['cross_affinity'], fitness_delta
        )

        # === PALETTE SELECTION ===
        new_act_mask, act_diversity_rescue = self._select_palette_by_activity(
            act_activity, new_act_aff, new_act_captured, new_act_tags,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            self.min_diversity_act, k2, prefer_indices=[4]
        )
        new_agg_mask, agg_diversity_rescue = self._select_palette_by_activity(
            agg_activity, new_agg_aff, new_agg_captured, new_agg_tags,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            self.min_diversity_agg, k3, prefer_indices=list(CORE_EXTREME_AGGS)
        )

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
            'cross_affinity': new_cross,
            'circadian_phase': float(new_circadian_phase),
            'act_phases': new_act_phases,
            'agg_phases': new_agg_phases,
            'cycles_completed': cycles_completed,
            'capture_events': state['capture_events'] + capture_count,
            'phase_sync_events': state['phase_sync_events'] + phase_sync_events,
            'sin_extreme_coactivations': state['sin_extreme_coactivations'] + sin_extreme_coact,
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

        # Phase alignment metrics
        sin_phase = float(new_act_phases[4])
        max_phase = float(new_agg_phases[2])
        min_phase = float(new_agg_phases[3])
        sin_max_alignment = jnp.cos(sin_phase - max_phase)
        sin_min_alignment = jnp.cos(sin_phase - min_phase)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Circadian metrics (KEY)
            'circadian_phase': float(new_circadian_phase),
            'clock_position': float(new_circadian_phase) / (2 * jnp.pi),
            'cycles_completed': cycles_completed,
            'sin_phase': sin_phase,
            'max_phase': max_phase,
            'min_phase': min_phase,
            'sin_max_alignment': float(sin_max_alignment),
            'sin_min_alignment': float(sin_min_alignment),
            'phase_sync_events': new_state['phase_sync_events'],
            # Activity metrics
            'sin_activity': float(act_activity[4]),
            'max_activity': float(agg_activity[2]),
            'min_activity': float(agg_activity[3]),
            # Co-activation
            'sin_extreme_coactivations': new_state['sin_extreme_coactivations'],
            # Tagging
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with circadian + clonal status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        sin_phase = float(state['act_phases'][4])
        max_phase = float(state['agg_phases'][2])
        sin_max_alignment = jnp.cos(sin_phase - max_phase)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # Circadian status
            'circadian_phase': state['circadian_phase'],
            'cycles_completed': state['cycles_completed'],
            'sin_phase': sin_phase,
            'sin_max_alignment': float(sin_max_alignment),
            'phase_sync_events': state['phase_sync_events'],
            # Co-activation
            'sin_extreme_coactivations': state['sin_extreme_coactivations'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
