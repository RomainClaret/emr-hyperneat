"""Fitness sanitization and the run statistics: fitness and species summaries, the topology summary, and
the recurrence metrics record."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def sanitize_fitness(fitnesses: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
    """NaN fitness to -inf, as TensorNEAT's own pipeline does, and how many there were.

    The generation loop calls a jitted ``tell`` directly, which bypasses TensorNEAT's guard,
    so without this a NaN (an overflowing settling, a NaN target) reaches selection and the
    reported best fitness. A population without NaN is returned unchanged.

    Returns:
        ``(fitnesses, nan_count)``.
    """
    nan = jnp.isnan(fitnesses)
    count = int(jnp.sum(nan))
    if count == 0:
        return fitnesses, 0
    return jnp.where(nan, -jnp.inf, fitnesses), count


def sanitize_fitness_on_device(fitnesses: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """sanitize_fitness without the host read: the same values, the count as a device scalar.

    Traceable, for the feed-forward generation. A population without NaN
    keeps its values bit for bit: the select returns them unchanged.

    Returns:
        ``(fitnesses, nan_count)``, the count an int32 scalar.
    """
    nan = jnp.isnan(fitnesses)
    return jnp.where(nan, -jnp.inf, fitnesses), jnp.sum(nan, dtype=jnp.int32)


def float32_threshold(target: float) -> np.float32:
    """The smallest float32 not below ``target``: for a float32 ``b``, ``b >= float32_threshold(t)``
    exactly when ``float(b) >= t``.

    run_until_threshold's Python loop compares a generation's float32 best, as a Python float, with
    the target. A target rounded to the nearest float32 can lie below it (``np.float32(0.95) <
    0.95``): HMR's GPU loop compared with that rounding and could stop a generation earlier than a
    Python-float comparison.

    Returns:
        A float32 (NaN for a NaN target, which stops nothing: no fitness compares above it).
    """
    with np.errstate(over='ignore'):
        rounded = np.float32(target)
    if np.isnan(rounded) or float(rounded) >= target:
        return rounded
    return np.nextafter(rounded, np.float32(np.inf))


def fitness_statistics(fitnesses: jnp.ndarray,
                       nan_count: int) -> Tuple[float, float, float, float]:
    """Best, mean, min and std of a population as told, the last three over its finite entries.

    A NaN fitness is told as -inf (``sanitize_fitness``); counting it would report a mean of -inf.

    Args:
        fitnesses: (pop_size,) fitness after ``sanitize_fitness``.
        nan_count: How many entries ``sanitize_fitness`` replaced.

    Returns:
        ``(best, mean, min, std)``; the last three are NaN when no entry is finite.
    """
    best = float(jnp.max(fitnesses))
    finite = fitnesses if nan_count == 0 else fitnesses[jnp.isfinite(fitnesses)]
    if not finite.size:
        return best, float('nan'), float('nan'), float('nan')
    return best, float(jnp.mean(finite)), float(jnp.min(finite)), float(jnp.std(finite))


def species_summary(species_state: Any,
                    fitnesses: jnp.ndarray) -> Tuple[int, List[int], List[float]]:
    """The CPPN species of an evaluated population, as TensorNEAT's speciation assigned them.

    ``species_state`` is ``state.species`` of the state the population was evaluated under, before
    ``tell``: its ``idx2species`` holds each genome's species key and its ``member_count`` the size
    of each slot (NaN for an empty slot). After ``tell`` the same fields describe the next
    population, so passing the new state would pair this generation's fitness with the next
    generation's species.

    Args:
        species_state: TensorNEAT's species state (``state.species``).
        fitnesses: (pop_size,) fitness the population was told with.

    Returns:
        ``(num_species, sizes, best)``: the number of non-empty species, their sizes and the best
        fitness of each, in TensorNEAT's slot order.
    """
    counts, keys, members, fitness = (np.asarray(a) for a in jax.device_get((
        species_state.member_count, species_state.species_keys, species_state.idx2species,
        fitnesses)))
    live = np.flatnonzero(np.nan_to_num(counts, nan=0.0) > 0)
    sizes = [int(counts[s]) for s in live]
    best = [float(np.max(fitness[members == keys[s]])) for s in live]
    return len(live), sizes, best


def topology_summary(
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    masks: jnp.ndarray,
    hh_counts: Optional[jnp.ndarray] = None,
) -> Dict[str, float]:
    """Population mean and spread of each genome's hidden nodes and expressed connections.

    A hidden node is an active substrate position. A connection is expressed when its weight
    survived pruning: a nonzero entry of W1 (input -> hidden) or W2 (hidden -> output), plus the
    genome's discovered h->h connections.

    Args:
        W1: (pop_size, num_inputs, positions) input weights as evaluated.
        W2: (pop_size, positions, num_outputs) output weights as evaluated.
        masks: (pop_size, positions) active-position mask.
        hh_counts: (pop_size,) h->h connection counts, or None on a feed-forward run.

    Returns:
        ``hidden_nodes_mean``, ``hidden_nodes_std``, ``connections_mean`` and ``connections_std``,
        the population fields of ``AlgorithmMetrics``.
    """
    connections = jnp.sum(W1 != 0, axis=(1, 2)) + jnp.sum(W2 != 0, axis=(1, 2))
    if hh_counts is not None:
        connections = connections + hh_counts
    hidden, connections = (np.asarray(a) for a in jax.device_get(
        (jnp.sum(masks, axis=1), connections)))
    return {
        'hidden_nodes_mean': float(np.mean(hidden)),
        'hidden_nodes_std': float(np.std(hidden)),
        'connections_mean': float(np.mean(connections)),
        'connections_std': float(np.std(connections)),
    }


@dataclass
class EMRRecurrenceMetrics:
    """Extended metrics for A/B testing recurrence configurations.

    Includes all metrics from RecurrenceMetrics plus multi-hop specific stats.
    """
    # Connection statistics
    total_connections: int = 0
    forward_connections: int = 0
    backward_connections: int = 0
    lateral_connections: int = 0
    self_loop_connections: int = 0

    # Hidden->hidden specific
    hidden_to_hidden_connections: int = 0
    direct_hh_connections: int = 0  # Before multi-hop expansion

    # Topology metrics
    avg_in_degree: float = 0.0
    avg_out_degree: float = 0.0
    has_cycles: bool = False

    # Performance metrics
    fitness: float = 0.0
    generations_to_solve: Optional[int] = None
    activate_time_used: int = 0

    # Timing breakdown (milliseconds) - fine-grained step timing
    time_cppn_ask_ms: float = 0.0        # CPPN population ask + transform
    time_cppn_query_ms: float = 0.0      # CPPN queries for I/O weights
    time_variance_ms: float = 0.0         # Variance computation + masks
    time_hh_discovery_ms: float = 0.0     # h->h discovery
    time_build_matrices_ms: float = 0.0   # W1/W2 matrix construction
    time_evaluation_ms: float = 0.0       # Forward pass evaluation
    time_neat_evolution_ms: float = 0.0   # NEAT tell() evolution

    # Legacy timing fields (kept for compatibility)
    substrate_discovery_ms: float = 0.0
    phase2_discovery_ms: float = 0.0
    forward_pass_ms: float = 0.0

    # Multi-hop specific
    multi_hop_connections_added: int = 0
    effective_iteration_level: int = 0
    transitive_closure_density: float = 0.0
    multi_hop_discovery_ms: float = 0.0

    # Cache status
    cache_hit: bool = False
    cache_hit_count: int = 0  # Cumulative hits across generations
    cache_refresh_count: int = 0

    # Substrate efficiency metrics
    active_positions_mean: float = 0.0
    active_positions_min: int = 0
    active_positions_max: int = 0
    total_positions: int = 0
    position_utilization: float = 0.0  # active / total ratio

    # Weight matrix density
    w1_density: float = 0.0  # Non-zero ratio in W1
    w2_density: float = 0.0  # Non-zero ratio in W2

    # Configuration echo (actual values used)
    variance_threshold_used: float = 0.0
    band_threshold_used: float = 0.0
    max_weight_used: float = 0.0

    # Population diversity (from last generation)
    min_fitness: float = 0.0
    max_fitness: float = 0.0
    std_fitness: float = 0.0
