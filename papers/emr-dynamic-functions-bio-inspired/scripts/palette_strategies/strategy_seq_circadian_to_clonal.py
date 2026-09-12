"""Sequential Hybrid Strategy: Circadian -> Clonal.

Phase 1 (gens 0 to switch_gen): Circadian rhythm for fast discovery.
Phase 2 (gens switch_gen+): Clonal selection for strong retention.

Rationale: Circadian's periodic exploration is best for initial function
discovery, while clonal's immune memory (affinity learning + hypermutation)
is best for retaining useful functions under changing fitness landscapes.
The sequential composition exploits each mechanism's strength in the right phase.

Compare against the PARALLEL hybrid (strategy_68, circadian+clonal) which
achieves 89.3% on Parity-4.
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
    mask_to_indices,
)
from .strategy_52_circadian_rhythm_dual import CircadianRhythmDualStrategy
from .strategy_29_clonal_selection_dual import ClonalSelectionDualStrategy


class SequentialCircadianToClonalStrategy(PaletteEvolutionStrategy):
    """Phase 1: Circadian rhythm (fast discovery). Phase 2: Clonal selection (strong retention).

    At switch_gen, the circadian phase's discovered palette is transferred to
    clonal selection as the initial configuration, giving clonal a warm start
    with the functions circadian found useful.
    """

    name = "seq_circadian_clonal_dual"
    description = "Sequential: Circadian discovery then Clonal retention"

    def __init__(
        self,
        switch_gen: int = 30,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.switch_gen = switch_gen
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES
        self.phase1 = CircadianRhythmDualStrategy(
            initial_act_palette=self.initial_act_palette,
            initial_agg_palette=self.initial_agg_palette,
        )
        self.phase2 = ClonalSelectionDualStrategy(
            initial_act_palette=self.initial_act_palette,
            initial_agg_palette=self.initial_agg_palette,
        )

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize with phase 1 (circadian) active."""
        phase1_state = self.phase1.initialize(config, seed)

        return {
            'phase': 1,
            'switch_gen': self.switch_gen,
            'phase1_state': phase1_state,
            'phase2_state': None,  # Initialized at switch time with discovered palette
            'seed': seed,
            'generation': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        if state['phase'] == 1:
            return self.phase1.get_active_palette(state['phase1_state'])
        return self.phase2.get_active_palette(state['phase2_state'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        if state['phase'] == 1:
            return self.phase1.get_active_agg_palette(state['phase1_state'])
        return self.phase2.get_active_agg_palette(state['phase2_state'])

    def _switch_to_phase2(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer circadian discoveries to clonal selection."""
        # Get circadian's discovered palettes
        discovered_act = self.phase1.get_active_palette(state['phase1_state'])
        discovered_agg = self.phase1.get_active_agg_palette(state['phase1_state'])

        # Initialize clonal with the discovered palette as starting point
        config = {
            'initial_palette': discovered_act,
            'initial_agg_palette': discovered_agg,
        }
        phase2_state = self.phase2.initialize(config, state['seed'] + 999)

        state['phase'] = 2
        state['phase2_state'] = phase2_state
        return state

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Route to active phase, switching at switch_gen."""

        # Check for phase switch
        if state['phase'] == 1 and generation >= state['switch_gen']:
            state = self._switch_to_phase2(state)

        if state['phase'] == 1:
            state['phase1_state'], metrics = self.phase1.post_generation_update(
                state['phase1_state'], generation, best_fitness,
                prev_best_fitness, population_data
            )
        else:
            state['phase2_state'], metrics = self.phase2.post_generation_update(
                state['phase2_state'], generation, best_fitness,
                prev_best_fitness, population_data
            )

        state['generation'] = generation + 1

        # Add phase info to metrics
        metrics['sequential_phase'] = state['phase']
        metrics['switch_gen'] = state['switch_gen']

        return state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with phase info."""
        if state['phase'] == 1:
            inner = self.phase1.get_state_summary(state['phase1_state'])
        else:
            inner = self.phase2.get_state_summary(state['phase2_state'])

        inner['sequential_phase'] = state['phase']
        inner['switch_gen'] = state['switch_gen']
        inner['strategy'] = self.name
        return inner
