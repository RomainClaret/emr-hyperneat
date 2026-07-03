"""Strategy 67: STDP + Consolidation Dual ("Temporal Credit Memory").

Hybrid combining STDP temporal credit assignment with Consolidation Window memory.

Key synergies:
1. STDP temporal credit (5-gen LTP window) tracks which functions PRECEDE success
2. Consolidation's working → LTM hierarchy stores validated functions
3. NOVEL: High STDP credit functions get PRIORITY in consolidation transfer
4. Cross-domain credit: act-agg pairs that co-precede success are jointly consolidated

Biological basis:
- Sleep consolidation prioritizes memories tagged during waking (SHY hypothesis)
- STDP provides the "tagging" signal that guides consolidation priority
- Temporal credit + memory consolidation = robust knowledge retention

Expected: Best of both worlds - temporal precision + long-term stability
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


class ConsolidationPhase:
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"


class STDPConsolidationDualStrategy(PaletteEvolutionStrategy):
    """STDP temporal credit guides memory consolidation priority.

    Functions that receive high STDP credit (preceded fitness improvement)
    are prioritized during periodic consolidation windows.
    """

    name = "stdp_consolidation_dual"
    description = "STDP temporal credit guides memory consolidation priority"

    def __init__(
        self,
        # STDP parameters
        ltp_window: int = 5,
        ltd_window: int = 3,
        history_length: int = 10,
        ltp_rate: float = 0.25,
        ltd_rate: float = 0.10,
        temporal_decay: float = 0.7,
        # Consolidation timing
        consolidation_frequency: int = 10,
        consolidation_duration: int = 3,
        # Consolidation parameters
        replay_strength: float = 1.5,
        replay_threshold: float = 0.5,
        transfer_rate: float = 0.12,
        ltm_decay_rate: float = 0.02,
        # Priority consolidation (KEY SYNERGY)
        credit_priority_weight: float = 0.4,  # How much STDP credit affects consolidation priority
        # Active phase parameters
        active_mutation_rate: float = 0.18,
        consolidation_mutation_rate: float = 0.02,
        # Cross-domain
        cross_learning_rate: float = 0.10,
        # Protection
        affinity_protection_threshold: float = 0.55,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        # STDP
        self.ltp_window = ltp_window
        self.ltd_window = ltd_window
        self.history_length = history_length
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.temporal_decay = temporal_decay

        # Consolidation
        self.consolidation_frequency = consolidation_frequency
        self.consolidation_duration = consolidation_duration
        self.replay_strength = replay_strength
        self.replay_threshold = replay_threshold
        self.transfer_rate = transfer_rate
        self.ltm_decay_rate = ltm_decay_rate
        self.credit_priority_weight = credit_priority_weight

        # Mutation
        self.active_mutation_rate = active_mutation_rate
        self.consolidation_mutation_rate = consolidation_mutation_rate

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _get_consolidation_phase(self, generation: int, last_consol: int) -> Tuple[str, bool]:
        """Determine consolidation phase."""
        gens_since = generation - last_consol
        if gens_since < self.consolidation_duration:
            return ConsolidationPhase.CONSOLIDATING, False
        elif gens_since >= self.consolidation_frequency:
            return ConsolidationPhase.CONSOLIDATING, True
        return ConsolidationPhase.ACTIVE, False

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize with combined STDP + Consolidation state."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_working': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'act_ltm': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'act_credit': jnp.zeros(NUM_ACTIVATIONS),  # STDP credit
            'act_history': [],  # (gen, mask, fitness) for STDP

            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_working': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'agg_ltm': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'agg_credit': jnp.zeros(NUM_AGGREGATIONS),
            'agg_history': [],

            # Cross-domain
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,

            # General state
            'rng_key': jax.random.PRNGKey(seed + 676767),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'consolidation_phase': ConsolidationPhase.ACTIVE,
            'last_consolidation': -self.consolidation_frequency,
            'consolidations_completed': 0,
            'fitness_ema': 0.5,

            # Stats
            'total_ltp_events': 0,
            'total_ltd_events': 0,
            'act_priority_transfers': 0,
            'agg_priority_transfers': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _compute_temporal_weight(self, gens_before: int) -> float:
        """Closer to improvement = stronger effect."""
        return self.temporal_decay ** abs(gens_before)

    def _stdp_update(
        self,
        credit: jnp.ndarray,
        history: List[Tuple[int, jnp.ndarray, float]],
        current_gen: int,
        improved: bool,
        improvement_magnitude: float,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, int, int]:
        """Apply STDP credit update."""
        new_credit = credit * 0.9  # Decay existing credit
        ltp_count, ltd_count = 0, 0

        if improved and len(history) >= 2:
            # LTP: boost credit for functions active before improvement
            for hist_gen, hist_mask, _ in history:
                gens_before = current_gen - hist_gen
                if 1 <= gens_before <= self.ltp_window:
                    temporal_weight = self._compute_temporal_weight(gens_before)
                    ltp_delta = (
                        self.ltp_rate * temporal_weight * improvement_magnitude *
                        (hist_mask > 0.5).astype(jnp.float32)
                    )
                    new_credit = jnp.clip(new_credit + ltp_delta, 0.0, 1.0)
                    ltp_count += int(jnp.sum(hist_mask > 0.5))

        elif not improved and len(history) >= self.ltd_window:
            # LTD: reduce credit for stagnant functions
            recent_mask = history[-1][1] if history else jnp.zeros(n_funcs)
            ltd_delta = self.ltd_rate * 0.5 * (recent_mask > 0.5).astype(jnp.float32)
            new_credit = jnp.clip(new_credit - ltd_delta, 0.0, 1.0)
            ltd_count = int(jnp.sum(recent_mask > 0.5))

        return new_credit, ltp_count, ltd_count

    def _priority_consolidation(
        self,
        working: jnp.ndarray,
        ltm: jnp.ndarray,
        credit: jnp.ndarray,
        mask: jnp.ndarray,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Consolidate with STDP credit priority."""
        new_working = working.copy()
        new_ltm = ltm.copy()
        priority_transfers = 0

        for i in range(n_funcs):
            # Compute priority: base affinity + credit bonus
            base_affinity = float(working[i])
            credit_bonus = float(credit[i]) * self.credit_priority_weight
            priority = base_affinity + credit_bonus

            # Replay for high-priority functions
            if priority >= self.replay_threshold:
                boost = self.replay_strength * (priority - self.replay_threshold)
                new_working = new_working.at[i].set(min(0.95, float(new_working[i]) + boost))

                # Transfer to LTM (priority-weighted)
                if float(mask[i]) > 0.5:
                    diff = float(working[i]) - float(ltm[i])
                    # Higher credit = higher transfer rate
                    effective_transfer = self.transfer_rate * (1.0 + float(credit[i]))
                    transfer = effective_transfer * diff
                    new_ltm = new_ltm.at[i].set(min(0.95, float(new_ltm[i]) + transfer))
                    if transfer > 0.01:
                        priority_transfers += 1

            # Decay inactive functions in LTM
            if float(mask[i]) < 0.5:
                decay = self.ltm_decay_rate * (float(new_ltm[i]) - 0.5)
                new_ltm = new_ltm.at[i].set(max(0.05, float(new_ltm[i]) - decay))

        return new_working, new_ltm, priority_transfers

    def _compute_effective_affinity(
        self,
        working: jnp.ndarray,
        ltm: jnp.ndarray,
        credit: jnp.ndarray,
        phase: str,
    ) -> jnp.ndarray:
        """Compute effective affinity combining working, LTM, and credit."""
        if phase == ConsolidationPhase.CONSOLIDATING:
            base = 0.3 * working + 0.6 * ltm
        else:
            base = 0.5 * working + 0.4 * ltm

        # Add credit contribution
        credit_contrib = 0.1 * credit
        return jnp.clip(base + credit_contrib, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        effective: jnp.ndarray,
        phase: str,
        n_funcs: int,
        min_active: int,
        max_active: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply phase-appropriate mutation."""
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []

        mut_rate = (
            self.consolidation_mutation_rate
            if phase == ConsolidationPhase.CONSOLIDATING
            else self.active_mutation_rate
        )

        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))
        current = int(jnp.sum(mask > 0.5))

        for i in range(n_funcs):
            aff = float(effective[i])

            if mask[i] < 0.5:
                # Inactive - might activate
                if current + len(activated) >= max_active:
                    continue
                eff_rate = mut_rate * (0.5 + aff)
                if act_probs[i] < eff_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active - might deactivate
                if aff >= self.affinity_protection_threshold:
                    dr = mut_rate * 0.1
                else:
                    dr = mut_rate * (1.0 - aff)
                if phase == ConsolidationPhase.CONSOLIDATING:
                    dr *= 0.2  # Reduce deactivation during consolidation
                if deact_probs[i] < dr:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < min_active:
            return mask, {'activated': [], 'deactivated': []}

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with STDP credit + consolidation priority."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Improvement magnitude for STDP
        improvement_magnitude = max(0.0, min(1.0,
            (best_fitness - prev_best_fitness) / max(0.1, prev_best_fitness)
        ))

        # Determine consolidation phase
        phase, starting = self._get_consolidation_phase(generation, state['last_consolidation'])
        last_consol = generation if starting else state['last_consolidation']
        consol_count = state['consolidations_completed'] + (1 if starting else 0)

        # Update histories for STDP
        act_history = state['act_history'] + [(generation, state['act_mask'].copy(), best_fitness)]
        if len(act_history) > self.history_length:
            act_history = act_history[-self.history_length:]

        agg_history = state['agg_history'] + [(generation, state['agg_mask'].copy(), best_fitness)]
        if len(agg_history) > self.history_length:
            agg_history = agg_history[-self.history_length:]

        # Apply STDP credit updates
        new_act_credit, act_ltp, act_ltd = self._stdp_update(
            state['act_credit'], act_history, generation,
            improved, improvement_magnitude, NUM_ACTIVATIONS
        )
        new_agg_credit, agg_ltp, agg_ltd = self._stdp_update(
            state['agg_credit'], agg_history, generation,
            improved, improvement_magnitude, NUM_AGGREGATIONS
        )

        # Update working memory
        alpha = 0.2
        new_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fs = max(-1.0, min(1.0, (best_fitness - new_ema) / max(0.1, new_ema)))

        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)

        lr = 0.08 if phase == ConsolidationPhase.CONSOLIDATING else 0.15
        new_act_working = jnp.clip(
            state['act_working'] + lr * fs * act_active, 0.0, 1.0
        )
        new_agg_working = jnp.clip(
            state['agg_working'] + lr * fs * agg_active, 0.0, 1.0
        )

        # Update cross-domain affinity
        cross_delta = self.cross_learning_rate * fs * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Consolidation with credit priority
        new_act_ltm, new_agg_ltm = state['act_ltm'], state['agg_ltm']
        act_priority_transfers, agg_priority_transfers = 0, 0

        if phase == ConsolidationPhase.CONSOLIDATING:
            new_act_working, new_act_ltm, act_priority_transfers = self._priority_consolidation(
                new_act_working, state['act_ltm'], new_act_credit,
                state['act_mask'], NUM_ACTIVATIONS
            )
            new_agg_working, new_agg_ltm, agg_priority_transfers = self._priority_consolidation(
                new_agg_working, state['agg_ltm'], new_agg_credit,
                state['agg_mask'], NUM_AGGREGATIONS
            )

        # Compute effective affinity
        act_eff = self._compute_effective_affinity(
            new_act_working, new_act_ltm, new_act_credit, phase
        )
        agg_eff = self._compute_effective_affinity(
            new_agg_working, new_agg_ltm, new_agg_credit, phase
        )

        # Mutate palettes
        new_act_mask, act_mut = self._mutate_palette(
            k_act, state['act_mask'], act_eff, phase,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act
        )
        new_agg_mask, agg_mut = self._mutate_palette(
            k_agg, state['agg_mask'], agg_eff, phase,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg
        )

        new_state = {
            'act_mask': new_act_mask,
            'act_working': new_act_working,
            'act_ltm': new_act_ltm,
            'act_credit': new_act_credit,
            'act_history': act_history,
            'agg_mask': new_agg_mask,
            'agg_working': new_agg_working,
            'agg_ltm': new_agg_ltm,
            'agg_credit': new_agg_credit,
            'agg_history': agg_history,
            'cross_affinity': new_cross,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'consolidation_phase': phase,
            'last_consolidation': last_consol,
            'consolidations_completed': consol_count,
            'fitness_ema': new_ema,
            'total_ltp_events': state['total_ltp_events'] + act_ltp + agg_ltp,
            'total_ltd_events': state['total_ltd_events'] + act_ltd + agg_ltd,
            'act_priority_transfers': state['act_priority_transfers'] + act_priority_transfers,
            'agg_priority_transfers': state['agg_priority_transfers'] + agg_priority_transfers,
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
            'consolidation_phase': phase,
            'starting_consolidation': starting,
            # STDP stats
            'sin_credit': float(new_act_credit[4]),
            'act_mean_credit': float(jnp.mean(new_act_credit)),
            'agg_mean_credit': float(jnp.mean(new_agg_credit)),
            # Memory stats
            'sin_ltm': float(new_act_ltm[4]),
            'act_mean_ltm': float(jnp.mean(new_act_ltm)),
            'agg_mean_ltm': float(jnp.mean(new_agg_ltm)),
            # Priority consolidation
            'act_priority_transfers': act_priority_transfers,
            'agg_priority_transfers': agg_priority_transfers,
            # Sin status
            'has_sin': 4 in act_palette,
        }
        metrics.update({f'act_{k}': v for k, v in act_mut.items()})
        metrics.update({f'agg_{k}': v for k, v in agg_mut.items()})

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with STDP + Consolidation stats."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'consolidation_phase': state['consolidation_phase'],
            'consolidations_completed': state['consolidations_completed'],
            'generation': state['generation'],
            'sin_credit': float(state['act_credit'][4]),
            'sin_ltm': float(state['act_ltm'][4]),
            'total_ltp_events': state['total_ltp_events'],
            'act_priority_transfers': state['act_priority_transfers'],
            'cross_avg_affinity': float(jnp.mean(state['cross_affinity'])),
        }
