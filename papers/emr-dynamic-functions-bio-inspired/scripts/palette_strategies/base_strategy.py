"""Base class for palette evolution strategies."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

# Number of available activation functions
NUM_ACTIVATIONS = 18

# Number of available aggregation functions
NUM_AGGREGATIONS = 6

# Activation function names by index
ACTIVATION_NAMES = [
    'tanh',      # 0
    'sigmoid',   # 1
    'relu',      # 2
    'identity',  # 3
    'sin',       # 4 - KEY for parity
    'gauss',     # 5
    'lelu',      # 6
    'softplus',  # 7
    'rs_adapt',  # 8  - Izhikevich-inspired
    'fs_fast',   # 9  - Izhikevich-inspired
    'lts_low',   # 10 - Izhikevich-inspired
    'burst',     # 11 - oscillatory
    'resonator', # 12 - oscillatory
    # bio-inspired
    'osc_adapt', # 13 - oscillatory + adaptive
    'gain_mod',  # 14 - cortical divisive normalization
    'receptive', # 15 - localized oscillatory
    'band_pass', # 16 - intermediate value filter
    'integrate', # 17 - membrane dynamics
]

# Aggregation function names by index
AGGREGATION_NAMES = [
    'sum',      # 0 - default
    'mean',     # 1
    'max',      # 2
    'min',      # 3
    'product',  # 4
    'maxabs',   # 5
]

# Aggregation categories
# Averaging aggregations - sum/mean normalize inputs
AVERAGING_AGGS = [0, 1]  # sum, mean

# Extreme-value aggregations - select strongest signals
# Critical for hard parity problems (P5, P6)
EXTREME_AGGS = [2, 3, 4, 5]  # max, min, product, maxabs

# Core extreme aggregations (most important for hard parity)
CORE_EXTREME_AGGS = [2, 3]  # max, min - required for P5/P6

# ============================================================================
# Aggregation Categories for Symmetric Discovery
# ============================================================================

# Aggregation categories for discovery and protection
AGG_CATEGORIES = {
    'averaging': [0, 1],      # sum, mean - smooth integration
    'extreme': [2, 3, 4, 5],  # max, min, product, maxabs - selective
    'selective': [2, 3],      # max, min - most important for hard parity
    'multiplicative': [4],    # product - nonlinear scaling
}

# Activation categories for symmetric treatment
ACT_CATEGORIES = {
    'smooth': [0, 1, 7],            # tanh, sigmoid, softplus
    'rectified': [2, 6],            # relu, lelu
    'periodic': [4, 5],             # sin, gauss - critical for parity
    'oscillatory': [11, 12, 13],    # burst, resonator, osc_adapt
    'adaptive': [8, 9, 10, 14],     # Izhikevich-inspired + gain_mod
    'specialized': [15, 16, 17],    # receptive, band_pass, integrate
}

# ============================================================================
# Cross-Domain Pair Categories for Affinity Learning
# ============================================================================

# Known synergistic pairs
CROSS_PAIR_KNOWN_SYNERGISTIC = [(4, 2), (4, 3)]  # sin-max, sin-min

# Hypothesized useful pairs for exploration
CROSS_PAIR_CATEGORIES = {
    'known_synergistic': [(4, 2), (4, 3)],  # sin-max, sin-min (proven)
    'oscillatory_extreme': [
        (11, 2), (11, 3),  # burst + max/min
        (12, 2), (12, 3),  # resonator + max/min
        (13, 2), (13, 3),  # osc_adapt + max/min
    ],
    'smooth_averaging': [
        (0, 0), (0, 1),    # tanh + sum/mean
        (1, 0), (1, 1),    # sigmoid + sum/mean
    ],
    'rectified_extreme': [
        (2, 2), (2, 3),    # relu + max/min
        (6, 2), (6, 3),    # lelu + max/min
    ],
    'periodic_averaging': [
        (4, 0), (4, 1),    # sin + sum/mean (alternative to extremes)
        (5, 0), (5, 1),    # gauss + sum/mean
    ],
}

# Sin index for reference
SIN_IDX = 4

# Default activation palette (no sin!)
DEFAULT_PALETTE_INDICES = [0, 1, 2, 3]  # tanh, sigmoid, relu, identity

# Default aggregation palette
DEFAULT_AGG_PALETTE_INDICES = [0, 1]  # sum, mean (safe defaults)


def create_initial_palette_mask(palette_spec):
    """Create binary mask from palette specification.

    Args:
        palette_spec: Either 'default', list of indices, or existing mask array

    Returns:
        Binary mask array of shape (13,)
    """
    if isinstance(palette_spec, str):
        if palette_spec == 'default':
            indices = DEFAULT_PALETTE_INDICES
        elif palette_spec == 'sin_default':
            indices = [0, 1, 2, 3, 4]  # default + sin
        elif palette_spec == 'full':
            indices = list(range(NUM_ACTIVATIONS))
        else:
            raise ValueError(f"Unknown palette spec: {palette_spec}")
    elif isinstance(palette_spec, list):
        indices = palette_spec
    else:
        # Assume it's already a mask
        return jnp.array(palette_spec, dtype=jnp.float32)

    mask = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.float32)
    for idx in indices:
        mask = mask.at[idx].set(1.0)
    return mask


def create_initial_agg_palette_mask(palette_spec):
    """Create binary mask for aggregation palette.

    Args:
        palette_spec: Either 'default', list of indices, or existing mask array

    Returns:
        Binary mask array of shape (NUM_AGGREGATIONS,)
    """
    if isinstance(palette_spec, str):
        if palette_spec == 'default':
            indices = DEFAULT_AGG_PALETTE_INDICES
        elif palette_spec == 'full':
            indices = list(range(NUM_AGGREGATIONS))
        else:
            raise ValueError(f"Unknown agg palette spec: {palette_spec}")
    elif isinstance(palette_spec, list):
        indices = palette_spec
    else:
        # Assume it's already a mask
        return jnp.array(palette_spec, dtype=jnp.float32)

    mask = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.float32)
    for idx in indices:
        mask = mask.at[idx].set(1.0)
    return mask


def mask_to_indices(mask):
    """Convert binary mask to list of active indices."""
    return list(jnp.where(mask > 0.5)[0].tolist())


@dataclass
class TrialMetrics:
    """Metrics collected during a single trial."""

    strategy: str
    seed: int
    solved: bool
    solved_gen: Optional[int]
    best_fitness: float
    generations_run: int
    elapsed_seconds: float

    # Palette tracking
    sin_discovery_gen: Optional[int] = None
    sin_lost_gen: Optional[int] = None
    palette_history: List[List[int]] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)

    # Mutation events
    mutation_events: List[Dict] = field(default_factory=list)

    # Strategy-specific
    strategy_metrics: Dict[str, Any] = field(default_factory=dict)


class PaletteEvolutionStrategy(ABC):
    """Base class for palette evolution strategies.

    Each strategy implements a different approach to discovering
    useful activation functions through palette evolution.
    """

    name: str = "base"
    description: str = "Abstract base strategy"

    @abstractmethod
    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize strategy state.

        Args:
            config: Configuration dict with initial_palette, etc.
            seed: Random seed for reproducibility

        Returns:
            State dictionary for this strategy
        """
        pass

    @abstractmethod
    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices.

        Args:
            state: Current strategy state

        Returns:
            List of active activation function indices
        """
        pass

    @abstractmethod
    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update after each generation.

        Called after each generation completes. Can modify palette
        based on fitness, stagnation, population statistics, etc.

        Args:
            state: Current strategy state
            generation: Current generation number
            best_fitness: Best fitness this generation
            prev_best_fitness: Best fitness from previous generation
            population_data: Optional dict with population-level info
                           (for fitness-guided strategies)

        Returns:
            Tuple of (new_state, metrics_dict)
            - new_state: Updated state
            - metrics_dict: Dict of metrics to log
        """
        pass

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return summary for logging.

        Args:
            state: Current strategy state

        Returns:
            Dict with key state info for logging
        """
        palette = self.get_active_palette(state)
        return {
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'has_burst': 11 in palette,
            'has_resonator': 12 in palette,
        }

    def check_sin_status(
        self,
        state: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Check if sin was discovered or lost.

        Args:
            state: Current state
            prev_state: Previous state (if available)

        Returns:
            Tuple of (discovered_this_gen, lost_this_gen)
            - discovered_this_gen: True if sin just became active
            - lost_this_gen: True if sin just became inactive
        """
        current_has_sin = 4 in self.get_active_palette(state)

        if prev_state is not None:
            prev_has_sin = 4 in self.get_active_palette(prev_state)
            discovered = current_has_sin and not prev_has_sin
            lost = not current_has_sin and prev_has_sin
            return discovered, lost

        return None, None
