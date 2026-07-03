"""Strategy 121: Morphogen-Critical Period Phased Gradients Dual.

Combines Morphogen Gradients (#31) with Critical Period phases (#9).
Morphogen source strength varies by developmental phase.

Key Innovation:
- Exploration phase: Strong morphogen gradients encourage wide function discovery
- Confirmation phase: Moderate gradients focus on promising areas
- Consolidation phase: Weak gradients, rely on already-discovered functions
- Phase transitions smoothly modulate spatial organization

Biological basis: During brain development, morphogen concentration varies
across developmental stages. Early stages have strong gradients (exploration),
while later stages have weaker gradients as connections stabilize.

Expected: Spatial organization guides discovery, then phase transitions lock in findings.
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
    CORE_EXTREME_AGGS,
)


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class MorphogenCriticalPeriodDualStrategy(PaletteEvolutionStrategy):
    """Morphogen gradients with critical period phase modulation.

    Combines spatial organization (morphogen) with
    temporal development (critical periods). Source strength decreases
    as development progresses.

    Critical innovation: Strong exploration gradients encourage early
    discovery. Gradients weaken during confirmation/consolidation to
    stabilize discovered functions.
    """

    name = "morphogen_critical_period_dual"
    description = "Dual: Morphogen gradients modulated by critical period phases"

    # Activation positions (from morphogen_gradient_dual)
    ACT_POSITIONS = {
        0: (0.2, 0.8), 1: (0.3, 0.7), 2: (0.1, 0.7), 3: (0.2, 0.6),
        4: (0.8, 0.2), 5: (0.3, 0.6), 6: (0.1, 0.6), 7: (0.2, 0.2),
        8: (0.5, 0.5), 9: (0.6, 0.5), 10: (0.5, 0.4), 11: (0.7, 0.3),
        12: (0.9, 0.3), 13: (0.8, 0.4), 14: (0.3, 0.3), 15: (0.7, 0.2),
        16: (0.1, 0.3), 17: (0.2, 0.4),
    }

    AGG_POSITIONS = {
        0: (0.5, 0.8),  # sum
        1: (0.4, 0.7),  # mean
        2: (0.7, 0.6),  # max
        3: (0.8, 0.5),  # min
        4: (0.6, 0.3),  # product
        5: (0.9, 0.4),  # maxabs
    }

    def __init__(
        self,
        # === Critical period timing ===
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # === Phase-dependent strength multipliers (NEW) ===
        exploration_strength_multiplier: float = 1.5,   # Strong gradients early
        confirmation_strength_multiplier: float = 0.8,  # Moderate gradients
        consolidation_strength_multiplier: float = 0.3, # Weak gradients late
        # === Morphogen parameters ===
        n_sources: int = 3,
        act_gradient_decay: float = 3.0,
        agg_gradient_decay: float = 3.5,
        act_concentration_threshold: float = 0.35,
        agg_concentration_threshold: float = 0.30,
        # === Source dynamics ===
        source_learning_rate: float = 0.08,
        source_momentum: float = 0.7,
        source_position_decay: float = 0.02,
        # === Strengths ===
        initial_strength: float = 1.0,
        strength_learning_rate: float = 0.05,
        strength_decay: float = 0.98,
        strength_min: float = 0.3,
        strength_max: float = 2.0,
        # === Cross-domain ===
        cross_source_influence: float = 0.8,
        # === Phase-specific thresholds (NEW) ===
        exploration_threshold_mult: float = 0.7,   # Lower threshold = more discovery
        consolidation_threshold_mult: float = 1.3, # Higher threshold = more selective
        # === Palette limits ===
        max_act_palette: int = 8,
        min_act_palette: int = 3,
        max_agg_palette: int = 4,
        min_agg_palette: int = 2,
        # === Capture mechanism ===
        capture_threshold: float = 0.6,
        capture_protection: float = 0.85,
        # === Early consolidation ===
        early_consolidation_threshold: float = 0.95,
        # === General ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Morphogen-Critical Period strategy."""
        # Critical period
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

        # Phase strength multipliers (NEW)
        self.exploration_strength_multiplier = exploration_strength_multiplier
        self.confirmation_strength_multiplier = confirmation_strength_multiplier
        self.consolidation_strength_multiplier = consolidation_strength_multiplier

        # Morphogen
        self.n_sources = n_sources
        self.act_gradient_decay = act_gradient_decay
        self.agg_gradient_decay = agg_gradient_decay
        self.act_concentration_threshold = act_concentration_threshold
        self.agg_concentration_threshold = agg_concentration_threshold

        # Source dynamics
        self.source_learning_rate = source_learning_rate
        self.source_momentum = source_momentum
        self.source_position_decay = source_position_decay

        # Strengths
        self.initial_strength = initial_strength
        self.strength_learning_rate = strength_learning_rate
        self.strength_decay = strength_decay
        self.strength_min = strength_min
        self.strength_max = strength_max

        # Cross-domain
        self.cross_source_influence = cross_source_influence

        # Phase thresholds (NEW)
        self.exploration_threshold_mult = exploration_threshold_mult
        self.consolidation_threshold_mult = consolidation_threshold_mult

        # Palette limits
        self.max_act_palette = max_act_palette
        self.min_act_palette = min_act_palette
        self.max_agg_palette = max_agg_palette
        self.min_agg_palette = min_agg_palette

        # Capture
        self.capture_threshold = capture_threshold
        self.capture_protection = capture_protection

        # Early consolidation
        self.early_consolidation_threshold = early_consolidation_threshold

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

        self._build_position_arrays()

    def _build_position_arrays(self):
        """Build position arrays for both domains."""
        self.act_positions = jnp.zeros((NUM_ACTIVATIONS, 2))
        for i, pos in self.ACT_POSITIONS.items():
            if i < NUM_ACTIVATIONS:
                self.act_positions = self.act_positions.at[i, 0].set(pos[0])
                self.act_positions = self.act_positions.at[i, 1].set(pos[1])

        self.agg_positions = jnp.zeros((NUM_AGGREGATIONS, 2))
        for i, pos in self.AGG_POSITIONS.items():
            if i < NUM_AGGREGATIONS:
                self.agg_positions = self.agg_positions.at[i, 0].set(pos[0])
                self.agg_positions = self.agg_positions.at[i, 1].set(pos[1])

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase with early consolidation."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def _get_phase_params(self, phase: str, generation: int) -> Dict[str, float]:
        """Get phase-specific parameters.

        KEY INNOVATION: Strength and threshold vary by phase.
        """
        if phase == CriticalPeriodPhase.EXPLORATION:
            strength_mult = self.exploration_strength_multiplier
            threshold_mult = self.exploration_threshold_mult
            source_lr = self.source_learning_rate * 1.5  # Faster adaptation early
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            # Smooth transition
            gen_in_phase = generation - self.exploration_end
            phase_length = self.confirmation_end - self.exploration_end
            progress = gen_in_phase / max(phase_length, 1)

            strength_mult = (
                self.exploration_strength_multiplier * (1 - progress) +
                self.confirmation_strength_multiplier * progress
            )
            threshold_mult = 1.0  # Normal threshold
            source_lr = self.source_learning_rate
        else:
            strength_mult = self.consolidation_strength_multiplier
            threshold_mult = self.consolidation_threshold_mult
            source_lr = self.source_learning_rate * 0.3  # Slow adaptation late

        return {
            'strength_mult': strength_mult,
            'threshold_mult': threshold_mult,
            'source_lr': source_lr,
        }

    def _initialize_sources(self, key, initial_act, initial_agg):
        """Initialize sources near initial functions."""
        positions = jnp.zeros((self.n_sources, 2))
        strengths = jnp.ones(self.n_sources) * self.initial_strength

        all_initial_pos = []
        for i in initial_act:
            if i < NUM_ACTIVATIONS:
                all_initial_pos.append(self.ACT_POSITIONS.get(i, (0.5, 0.5)))
        for i in initial_agg:
            if i < NUM_AGGREGATIONS:
                all_initial_pos.append(self.AGG_POSITIONS.get(i, (0.5, 0.5)))

        if all_initial_pos:
            all_initial_pos = jnp.array(all_initial_pos)
            for s in range(self.n_sources):
                idx = s % len(all_initial_pos)
                key, subkey = jax.random.split(key)
                offset = jax.random.uniform(subkey, (2,), minval=-0.1, maxval=0.1)
                positions = positions.at[s].set(
                    jnp.clip(all_initial_pos[idx] + offset, 0.0, 1.0)
                )
        else:
            key, subkey = jax.random.split(key)
            positions = jax.random.uniform(subkey, (self.n_sources, 2))

        return positions, strengths, key

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        key = jax.random.PRNGKey(seed + 1210000)
        source_positions, source_strengths, key = self._initialize_sources(
            key, initial_act, initial_agg
        )
        source_velocities = jnp.zeros((self.n_sources, 2))

        act_success = jnp.zeros(NUM_ACTIVATIONS)
        agg_success = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_success = act_success.at[i].set(0.3)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_success = agg_success.at[i].set(0.3)

        # Capture tracking (NEW)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'source_positions': source_positions,
            'source_strengths': source_strengths,
            'source_velocities': source_velocities,
            'act_success': act_success,
            'agg_success': agg_success,
            'act_concentrations': jnp.zeros(NUM_ACTIVATIONS),
            'agg_concentrations': jnp.zeros(NUM_AGGREGATIONS),
            # Capture state (NEW)
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            # Phase tracking
            'phase': CriticalPeriodPhase.EXPLORATION,
            'phase_history': [],
            # Stats
            'capture_events': 0,
            # General
            'rng_key': key,
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_concentrations(
        self,
        source_positions: jnp.ndarray,
        source_strengths: jnp.ndarray,
        func_positions: jnp.ndarray,
        gradient_decay: float,
        strength_multiplier: float,
        cross_influence: float = 1.0,
    ) -> jnp.ndarray:
        """Compute morphogen concentration with phase-modulated strength."""
        n_funcs = func_positions.shape[0]
        concentrations = jnp.zeros(n_funcs)

        for i in range(n_funcs):
            func_pos = func_positions[i]
            total = 0.0

            for s in range(self.n_sources):
                source_pos = source_positions[s]
                # Apply phase strength multiplier
                strength = source_strengths[s] * cross_influence * strength_multiplier
                distance = jnp.sqrt(jnp.sum((func_pos - source_pos) ** 2))
                total += strength * jnp.exp(-gradient_decay * distance)

            concentrations = concentrations.at[i].set(total)

        return concentrations

    def _select_palette(
        self,
        concentrations: jnp.ndarray,
        captured: jnp.ndarray,
        threshold: float,
        threshold_mult: float,
        min_size: int,
        max_size: int,
    ) -> jnp.ndarray:
        """Select palette based on concentrations with capture protection."""
        n_funcs = concentrations.shape[0]
        effective_threshold = threshold * threshold_mult

        # Captured functions get concentration boost
        boosted_conc = concentrations + captured * self.capture_protection

        above = boosted_conc >= effective_threshold
        n_above = int(jnp.sum(above))

        if min_size <= n_above <= max_size:
            return above.astype(jnp.float32)
        elif n_above < min_size:
            top_k = jnp.argsort(boosted_conc)[-min_size:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
            return mask
        else:
            top_k = jnp.argsort(boosted_conc)[-max_size:]
            mask = jnp.zeros(n_funcs)
            for idx in top_k:
                mask = mask.at[int(idx)].set(1.0)
            return mask

    def _update_success(
        self,
        success: jnp.ndarray,
        mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update success memory."""
        new_success = 0.9 * success
        n_funcs = success.shape[0]
        for i in range(n_funcs):
            if mask[i] > 0.5:
                new_success = new_success.at[i].add(max(0, improvement))
        return jnp.clip(new_success, 0.0, 1.0)

    def _update_capture(
        self,
        captured: jnp.ndarray,
        concentrations: jnp.ndarray,
        mask: jnp.ndarray,
        improved: bool,
        phase: str,
    ) -> Tuple[jnp.ndarray, int]:
        """Update capture state during confirmation phase.

        Functions that consistently have high concentration get captured.
        """
        new_captured = captured.copy()
        capture_count = 0

        # Only capture during confirmation (not exploration or consolidation)
        if phase != CriticalPeriodPhase.CONFIRMATION:
            return new_captured, 0

        if improved:
            for i in range(len(concentrations)):
                if (mask[i] > 0.5 and
                    concentrations[i] > self.capture_threshold and
                    captured[i] < 0.5):
                    new_captured = new_captured.at[i].set(1.0)
                    capture_count += 1

        return new_captured, capture_count

    def _compute_source_gradients(
        self,
        source_positions: jnp.ndarray,
        act_success: jnp.ndarray,
        agg_success: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute gradients for source movement toward successful functions."""
        gradients = jnp.zeros((self.n_sources, 2))

        for s in range(self.n_sources):
            source_pos = source_positions[s]
            total_weight = 0.0
            weighted_dir = jnp.zeros(2)

            # From activations
            for i in range(NUM_ACTIVATIONS):
                weight = float(act_success[i])
                if act_mask[i] > 0.5:
                    weight *= 2.0
                # Extra weight for sin
                if i == 4:
                    weight *= 1.5
                if weight > 0.01:
                    func_pos = self.act_positions[i]
                    direction = func_pos - source_pos
                    distance = jnp.sqrt(jnp.sum(direction ** 2)) + 0.01
                    weighted_dir += weight * direction / distance
                    total_weight += weight

            # From aggregations
            for i in range(NUM_AGGREGATIONS):
                weight = float(agg_success[i]) * self.cross_source_influence
                if agg_mask[i] > 0.5:
                    weight *= 2.0
                # Extra weight for extreme aggs
                if i in CORE_EXTREME_AGGS:
                    weight *= 1.5
                if weight > 0.01:
                    func_pos = self.agg_positions[i]
                    direction = func_pos - source_pos
                    distance = jnp.sqrt(jnp.sum(direction ** 2)) + 0.01
                    weighted_dir += weight * direction / distance
                    total_weight += weight

            if total_weight > 0.01:
                gradients = gradients.at[s].set(weighted_dir / total_weight)

        return gradients

    def _update_sources(
        self,
        positions: jnp.ndarray,
        strengths: jnp.ndarray,
        velocities: jnp.ndarray,
        gradients: jnp.ndarray,
        improvement: float,
        source_lr: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update source positions and strengths with phase-specific LR."""
        new_velocities = self.source_momentum * velocities + source_lr * gradients
        new_positions = positions + new_velocities

        center = jnp.array([0.5, 0.5])
        center_pull = self.source_position_decay * (center - new_positions)
        new_positions = jnp.clip(new_positions + center_pull, 0.0, 1.0)

        new_strengths = self.strength_decay * strengths
        if improvement > 0:
            new_strengths = new_strengths + self.strength_learning_rate * improvement
        new_strengths = jnp.clip(new_strengths, self.strength_min, self.strength_max)

        return new_positions, new_strengths, new_velocities

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with phase-modulated morphogen gradients."""
        key = state['rng_key']

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Determine phase
        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']
        phase_params = self._get_phase_params(phase, generation)

        # Update success
        new_act_success = self._update_success(state['act_success'], state['act_mask'], improvement)
        new_agg_success = self._update_success(state['agg_success'], state['agg_mask'], improvement)

        # Compute gradients
        gradients = self._compute_source_gradients(
            state['source_positions'], new_act_success, new_agg_success,
            state['act_mask'], state['agg_mask']
        )

        # Update sources with phase-specific LR
        new_positions, new_strengths, new_velocities = self._update_sources(
            state['source_positions'], state['source_strengths'],
            state['source_velocities'], gradients, improvement,
            phase_params['source_lr']
        )

        # Compute concentrations with phase-modulated strength
        new_act_conc = self._compute_concentrations(
            new_positions, new_strengths, self.act_positions,
            self.act_gradient_decay, phase_params['strength_mult']
        )
        new_agg_conc = self._compute_concentrations(
            new_positions, new_strengths, self.agg_positions,
            self.agg_gradient_decay, phase_params['strength_mult'],
            self.cross_source_influence
        )

        # Update capture
        new_act_captured, act_cap_count = self._update_capture(
            state['act_captured'], new_act_conc, state['act_mask'], improved, phase
        )
        new_agg_captured, agg_cap_count = self._update_capture(
            state['agg_captured'], new_agg_conc, state['agg_mask'], improved, phase
        )

        # Select palettes with phase-specific threshold
        new_act_mask = self._select_palette(
            new_act_conc, new_act_captured,
            self.act_concentration_threshold, phase_params['threshold_mult'],
            self.min_act_palette, self.max_act_palette
        )
        new_agg_mask = self._select_palette(
            new_agg_conc, new_agg_captured,
            self.agg_concentration_threshold, phase_params['threshold_mult'],
            self.min_agg_palette, self.max_agg_palette
        )

        # Track phase history
        phase_history = state['phase_history'] + [(generation, phase)]
        if len(phase_history) > 50:
            phase_history = phase_history[-50:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'source_positions': new_positions,
            'source_strengths': new_strengths,
            'source_velocities': new_velocities,
            'act_success': new_act_success,
            'agg_success': new_agg_success,
            'act_concentrations': new_act_conc,
            'agg_concentrations': new_agg_conc,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'phase': phase,
            'phase_history': phase_history,
            'capture_events': state['capture_events'] + act_cap_count + agg_cap_count,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        source_pos_list = [
            (float(new_positions[s, 0]), float(new_positions[s, 1]))
            for s in range(self.n_sources)
        ]

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Phase metrics (NEW)
            'phase': phase,
            'phase_changed': phase_changed,
            'strength_mult': phase_params['strength_mult'],
            'threshold_mult': phase_params['threshold_mult'],
            # Concentration stats
            'act_mean_concentration': float(jnp.mean(new_act_conc)),
            'agg_mean_concentration': float(jnp.mean(new_agg_conc)),
            'sin_concentration': float(new_act_conc[4]),
            'max_concentration': float(new_agg_conc[2]),
            'min_concentration': float(new_agg_conc[3]),
            # Capture stats
            'capture_events': new_state['capture_events'],
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'max_captured': bool(new_agg_captured[2] > 0.5),
            # Sources
            'source_positions': source_pos_list,
            'source_strengths': [float(s) for s in new_strengths],
            # Function status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            'sin_success': float(new_act_success[4]),
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
            'phase': state['phase'],
            'generation': state['generation'],
            'act_mean_concentration': float(jnp.mean(state['act_concentrations'])),
            'agg_mean_concentration': float(jnp.mean(state['agg_concentrations'])),
            'sin_concentration': float(state['act_concentrations'][4]),
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'capture_events': state['capture_events'],
            'n_sources': self.n_sources,
            'source_strengths': [float(s) for s in state['source_strengths']],
        }
