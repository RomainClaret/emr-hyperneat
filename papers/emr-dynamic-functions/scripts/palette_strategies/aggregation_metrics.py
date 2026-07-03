"""Aggregation Discovery Metrics.

This module provides per-aggregation tracking for symmetric activation-aggregation
discovery in bio-inspired palette evolution strategies.

Key metrics tracked:
- Per-aggregation discovery timing (generation when first found)
- Per-aggregation stability (generations retained after discovery)
- Per-aggregation loss events (when discovered function is lost)
- Cross-domain coupling strength (learned affinity between act-agg pairs)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base_strategy import (
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    AGGREGATION_NAMES,
    ACTIVATION_NAMES,
    CORE_EXTREME_AGGS,
    AGG_CATEGORIES,
    CROSS_PAIR_CATEGORIES,
)


@dataclass
class AggregationDiscoveryMetrics:
    """Track individual aggregation function discovery and retention.

    Provides symmetric tracking for aggregations matching how activations
    are tracked (discovery_gen, stability, loss events).
    """

    # Per-aggregation discovery generation (None if never discovered)
    sum_discovery_gen: Optional[int] = None      # Index 0
    mean_discovery_gen: Optional[int] = None     # Index 1
    max_discovery_gen: Optional[int] = None      # Index 2
    min_discovery_gen: Optional[int] = None      # Index 3
    product_discovery_gen: Optional[int] = None  # Index 4
    maxabs_discovery_gen: Optional[int] = None   # Index 5

    # Per-aggregation stability (generations retained after discovery)
    sum_stability_gens: int = 0
    mean_stability_gens: int = 0
    max_stability_gens: int = 0
    min_stability_gens: int = 0
    product_stability_gens: int = 0
    maxabs_stability_gens: int = 0

    # Per-aggregation loss generation (None if never lost after discovery)
    sum_lost_gen: Optional[int] = None
    mean_lost_gen: Optional[int] = None
    max_lost_gen: Optional[int] = None
    min_lost_gen: Optional[int] = None
    product_lost_gen: Optional[int] = None
    maxabs_lost_gen: Optional[int] = None

    # Cross-domain coupling: final cross-affinity values for key pairs
    final_cross_affinity: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Cross-pair formation tracking
    cross_pairs_formed: List[Tuple[int, int, int]] = field(default_factory=list)  # (act, agg, gen)

    # History tracking (optional, for detailed analysis)
    agg_palette_history: List[List[int]] = field(default_factory=list)

    def _get_attr_name(self, agg_idx: int, attr_type: str) -> str:
        """Get attribute name for given aggregation index and type."""
        agg_name = AGGREGATION_NAMES[agg_idx]
        return f"{agg_name}_{attr_type}"

    def update_from_palette(
        self,
        agg_palette: List[int],
        prev_palette: List[int],
        generation: int,
    ) -> Dict[str, Any]:
        """Update metrics based on current vs previous palette.

        Args:
            agg_palette: Current active aggregation indices
            prev_palette: Previous generation's active aggregation indices
            generation: Current generation number

        Returns:
            Dict with discovery/loss events this generation
        """
        events = {
            'discovered': [],
            'lost': [],
            'retained': [],
        }

        for agg_idx in range(NUM_AGGREGATIONS):
            agg_name = AGGREGATION_NAMES[agg_idx]
            was_present = agg_idx in prev_palette
            is_present = agg_idx in agg_palette

            disc_attr = f"{agg_name}_discovery_gen"
            stab_attr = f"{agg_name}_stability_gens"
            lost_attr = f"{agg_name}_lost_gen"

            # Discovery event
            if is_present and getattr(self, disc_attr) is None:
                setattr(self, disc_attr, generation)
                events['discovered'].append((agg_idx, agg_name))

            # Stability tracking (increment if discovered and present)
            if is_present and getattr(self, disc_attr) is not None:
                current_stab = getattr(self, stab_attr)
                setattr(self, stab_attr, current_stab + 1)
                events['retained'].append((agg_idx, agg_name))

            # Loss event (was discovered, was present, now gone)
            if was_present and not is_present:
                if getattr(self, disc_attr) is not None and getattr(self, lost_attr) is None:
                    setattr(self, lost_attr, generation)
                    events['lost'].append((agg_idx, agg_name))

        # Track palette history
        self.agg_palette_history.append(list(agg_palette))

        return events

    def record_cross_pair_formation(
        self,
        act_idx: int,
        agg_idx: int,
        generation: int,
        affinity: float,
    ):
        """Record formation of a cross-domain pair.

        Args:
            act_idx: Activation function index
            agg_idx: Aggregation function index
            generation: Generation when pair formed
            affinity: Cross-affinity value at formation
        """
        self.cross_pairs_formed.append((act_idx, agg_idx, generation))
        self.final_cross_affinity[(act_idx, agg_idx)] = affinity

    def update_final_cross_affinity(self, cross_affinity_matrix):
        """Update final cross-affinity from matrix at end of trial.

        Args:
            cross_affinity_matrix: (NUM_ACTIVATIONS, NUM_AGGREGATIONS) array
        """
        for i in range(NUM_ACTIVATIONS):
            for j in range(NUM_AGGREGATIONS):
                self.final_cross_affinity[(i, j)] = float(cross_affinity_matrix[i, j])

    def get_discovery_rate(self) -> float:
        """Calculate overall aggregation discovery rate."""
        discovered = 0
        for agg_idx in range(NUM_AGGREGATIONS):
            agg_name = AGGREGATION_NAMES[agg_idx]
            if getattr(self, f"{agg_name}_discovery_gen") is not None:
                discovered += 1
        return discovered / NUM_AGGREGATIONS * 100

    def get_extreme_discovery_rate(self) -> float:
        """Calculate discovery rate for extreme aggregations (max, min)."""
        discovered = 0
        for agg_idx in CORE_EXTREME_AGGS:
            agg_name = AGGREGATION_NAMES[agg_idx]
            if getattr(self, f"{agg_name}_discovery_gen") is not None:
                discovered += 1
        return discovered / len(CORE_EXTREME_AGGS) * 100

    def get_retention_rate(self) -> float:
        """Calculate retention rate (discovered and not lost)."""
        discovered = 0
        retained = 0
        for agg_idx in range(NUM_AGGREGATIONS):
            agg_name = AGGREGATION_NAMES[agg_idx]
            if getattr(self, f"{agg_name}_discovery_gen") is not None:
                discovered += 1
                if getattr(self, f"{agg_name}_lost_gen") is None:
                    retained += 1
        if discovered == 0:
            return 0.0
        return retained / discovered * 100

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this trial."""
        return {
            'discovery_rate': self.get_discovery_rate(),
            'extreme_discovery_rate': self.get_extreme_discovery_rate(),
            'retention_rate': self.get_retention_rate(),
            'max_discovery_gen': self.max_discovery_gen,
            'min_discovery_gen': self.min_discovery_gen,
            'max_stability_gens': self.max_stability_gens,
            'min_stability_gens': self.min_stability_gens,
            'cross_pairs_formed': len(self.cross_pairs_formed),
            'non_sin_extreme_pairs': len([
                p for p in self.cross_pairs_formed
                if p[0] != 4 or p[1] not in CORE_EXTREME_AGGS
            ]),
        }


def calculate_aggregation_discovery_rates(
    trials: List[AggregationDiscoveryMetrics],
) -> Dict[str, Any]:
    """Calculate per-aggregation discovery rates across trials.

    Args:
        trials: List of AggregationDiscoveryMetrics from multiple trials

    Returns:
        Dict with per-aggregation and aggregate statistics
    """
    if not trials:
        return {'error': 'No trials provided'}

    results = {
        'per_agg': {},
        'extreme_first_rate': 0.0,      # % trials where extreme found before averaging
        'max_before_min_rate': 0.0,     # % trials where max found before min
        'avg_extreme_discovery_gap': 0.0,  # Generations between max and min discovery
        'avg_cross_pairs_formed': 0.0,
        'avg_non_sin_pairs': 0.0,
    }

    # Per-aggregation analysis
    for agg_idx, agg_name in enumerate(AGGREGATION_NAMES):
        discovered_count = 0
        discovery_gens = []
        stability_gens = []
        lost_count = 0

        for trial in trials:
            disc_gen = getattr(trial, f'{agg_name}_discovery_gen')
            stab_gen = getattr(trial, f'{agg_name}_stability_gens')
            lost_gen = getattr(trial, f'{agg_name}_lost_gen')

            if disc_gen is not None:
                discovered_count += 1
                discovery_gens.append(disc_gen)
                stability_gens.append(stab_gen)

            if lost_gen is not None:
                lost_count += 1

        results['per_agg'][agg_name] = {
            'discovery_rate': discovered_count / len(trials) * 100,
            'avg_discovery_gen': np.mean(discovery_gens) if discovery_gens else None,
            'avg_stability_gens': np.mean(stability_gens) if stability_gens else None,
            'loss_rate': lost_count / max(discovered_count, 1) * 100,
            'retention_rate': (discovered_count - lost_count) / max(discovered_count, 1) * 100,
        }

    # Extreme-first analysis
    extreme_first = 0
    max_before_min = 0
    discovery_gaps = []

    for trial in trials:
        max_gen = trial.max_discovery_gen
        min_gen = trial.min_discovery_gen
        sum_gen = trial.sum_discovery_gen
        mean_gen = trial.mean_discovery_gen

        # Check if any extreme found before any averaging
        extreme_gens = [g for g in [max_gen, min_gen] if g is not None]
        avg_gens = [g for g in [sum_gen, mean_gen] if g is not None]

        if extreme_gens and (not avg_gens or min(extreme_gens) < min(avg_gens)):
            extreme_first += 1

        # Max before min?
        if max_gen is not None and min_gen is not None:
            if max_gen <= min_gen:
                max_before_min += 1
            discovery_gaps.append(abs(max_gen - min_gen))

    results['extreme_first_rate'] = extreme_first / len(trials) * 100
    results['max_before_min_rate'] = max_before_min / len(trials) * 100
    results['avg_extreme_discovery_gap'] = np.mean(discovery_gaps) if discovery_gaps else 0.0

    # Cross-pair analysis
    cross_pairs_counts = [len(t.cross_pairs_formed) for t in trials]
    non_sin_pairs_counts = [
        len([p for p in t.cross_pairs_formed if p[0] != 4 or p[1] not in CORE_EXTREME_AGGS])
        for t in trials
    ]

    results['avg_cross_pairs_formed'] = np.mean(cross_pairs_counts)
    results['avg_non_sin_pairs'] = np.mean(non_sin_pairs_counts)

    return results


def print_aggregation_metrics_table(
    results: Dict[str, Any],
    title: str = "Aggregation Discovery Analysis",
):
    """Print formatted table of aggregation discovery results.

    Args:
        results: Output from calculate_aggregation_discovery_rates
        title: Table title
    """
    print("=" * 80)
    print(title)
    print("=" * 80)

    # Per-aggregation table
    print(f"\n{'Aggregation':<12} | {'Disc%':>7} | {'AvgGen':>7} | {'Stability':>10} | {'Retained%':>10}")
    print("-" * 60)

    for agg_name in AGGREGATION_NAMES:
        stats = results['per_agg'].get(agg_name, {})
        disc = stats.get('discovery_rate', 0)
        avg_gen = stats.get('avg_discovery_gen')
        stab = stats.get('avg_stability_gens')
        ret = stats.get('retention_rate', 0)

        avg_gen_str = f"{avg_gen:.1f}" if avg_gen is not None else "N/A"
        stab_str = f"{stab:.1f}" if stab is not None else "N/A"

        print(f"{agg_name:<12} | {disc:>6.1f}% | {avg_gen_str:>7} | {stab_str:>10} | {ret:>9.1f}%")

    # Summary stats
    print("\n" + "-" * 60)
    print(f"Extreme-First Rate: {results['extreme_first_rate']:.1f}%")
    print(f"Max Before Min Rate: {results['max_before_min_rate']:.1f}%")
    print(f"Avg Extreme Discovery Gap: {results['avg_extreme_discovery_gap']:.1f} generations")
    print(f"Avg Cross-Pairs Formed: {results['avg_cross_pairs_formed']:.1f}")
    print(f"Avg Non-Sin-Extreme Pairs: {results['avg_non_sin_pairs']:.1f}")
    print("=" * 80)
