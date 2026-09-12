"""Chunk-size arithmetic shared by the EMR execution strategies.

Every strategy that trades speed for memory has to answer the same question: how
many genomes, positions or samples fit in a given byte budget. That arithmetic was
previously duplicated as private methods on two subclasses and as a module-level
helper on a third, so the three could drift. It is collected here as pure
functions -- no algorithm state, no JAX -- which also makes it directly testable.

All budgets are in GB and use 1e9 bytes per GB, matching the constants the
original call sites were tuned against.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

BYTES_PER_FLOAT32 = 4
BYTES_PER_GB = 1e9

# Retired estimate, kept only so the corrected model can state its error factor:
# the old sizing assumed the CPPN query allocates ~one intermediate alongside its
# output (2 float32 per (genome, position)). The measured cost is the double-vmapped
# `DefaultGenome.forward` while-loop carry: 2 x max_nodes float32 per (genome,
# position) -- 100 at the default max_nodes=50, an undercount of 50x that produced
# two byte-exact recorded depth-7 OOM allocations (6,555,023,048 B and
# 25,001,623,112 B).
LEGACY_CPPN_QUERY_OVERHEAD = 2

# Floats the double-vmapped CPPN forward carries per (genome, position) pair:
# node activations plus the accumulation buffer, each padded to max_nodes.
CPPN_CARRY_FLOATS_PER_NODE = 2

# Floors below which chunking costs more in Python overhead than it saves in memory.
MIN_POSITION_CHUNK = 1000
MIN_POPULATION_CHUNK = 10
MIN_RECURRENT_CHUNK = 10

# Population chunking keeps a margin so that a slightly-off estimate does not OOM.
POPULATION_SAFETY_MARGIN = 0.9


def cppn_query_carry_bytes(pop_size: int, n_positions: int, max_nodes: int) -> int:
    """Bytes the double-vmapped CPPN query materializes for its forward carry.

    Measured, not modeled: the `DefaultGenome.forward` while-loop carries
    `CPPN_CARRY_FLOATS_PER_NODE x max_nodes` float32 per (genome, position) pair.
    Validated byte-exact against both recorded depth-7 OOMs (pop 1000,
    max_nodes 50): 16,384 positions -> 6.5536e9 B (observed 6,555,023,048,
    +0.022%) and 62,500 positions -> 2.5e10 B (observed 25,001,623,112,
    +0.0065%).

    Args:
        pop_size: Genomes queried together.
        n_positions: Target positions in the (level or chunk) slab.
        max_nodes: The CPPN genome's padded node capacity.

    Returns:
        Carry bytes for one unchunked query over the slab.
    """
    return (pop_size * n_positions * CPPN_CARRY_FLOATS_PER_NODE
            * max_nodes * BYTES_PER_FLOAT32)


def optimal_position_chunk_size(
    pop_size: int,
    num_sources: int,
    target_gb: float,
    max_chunk: int,
    max_nodes: int,
) -> int:
    """Positions per CPPN query chunk that fit inside a GPU memory target.

    The dominant term is the forward carry (`cppn_query_carry_bytes`); the
    per-source output rows are added for honesty but are ~1% of it at the
    default max_nodes. The previous version modeled only the output rows
    (`LEGACY_CPPN_QUERY_OVERHEAD`), undercounting by 2*max_nodes/2 = 50x and
    sizing 62,500-position shards against a 0.5 GB target.

    Args:
        pop_size: Population size, since every genome queries every position.
        num_sources: Number of source coordinates (inputs or outputs).
        target_gb: Target GPU memory for one shard, in GB.
        max_chunk: Upper clamp, the configured `cppn_pos_chunk_size`.
        max_nodes: The CPPN genome's padded node capacity (carry term).

    Returns:
        Positions per chunk, clamped to [MIN_POSITION_CHUNK, max_chunk].
    """
    bytes_per_position = pop_size * BYTES_PER_FLOAT32 * (
        CPPN_CARRY_FLOATS_PER_NODE * max_nodes + num_sources)
    target_bytes = target_gb * BYTES_PER_GB
    optimal_chunk = int(target_bytes / bytes_per_position)
    return max(MIN_POSITION_CHUNK, min(optimal_chunk, max_chunk))


def population_chunk_for_query(
    pop_size: int,
    n_positions: int,
    max_nodes: int,
    target_gb: float,
) -> int:
    """Genomes per query chunk under the carry budget -- the AUTO (-1) semantics.

    Returns `pop_size` unchanged whenever the full population's carry fits the
    budget, so every configuration that fit before keeps its measured behavior
    byte-identical; chunking engages only where the unchunked query would have
    OOMed. Per-genome results are independent, so chunk-and-concatenate is
    value-identical to the unchunked query.

    Headroom note: with the product budget the fat-carry design runs d7/pop1000
    at the 11 GB card's preallocation ceiling (peak 9,554 MiB, 84 C) at the
    default 0.5 GB target. Deeper substrates or larger palettes may need
    `target_shard_gb` exposed rather than assumed.

    Args:
        pop_size: Population size.
        n_positions: Positions in the slab being queried (for the streaming
            path, the largest level slab).
        max_nodes: The CPPN genome's padded node capacity.
        target_gb: Carry budget for one query, in GB.

    Returns:
        Genomes per chunk, in [1, pop_size].
    """
    if pop_size <= 0 or n_positions <= 0:
        return max(pop_size, 1)
    if cppn_query_carry_bytes(pop_size, n_positions, max_nodes) <= target_gb * BYTES_PER_GB:
        return pop_size
    per_genome = cppn_query_carry_bytes(1, n_positions, max_nodes)
    return max(1, min(pop_size, int(target_gb * BYTES_PER_GB / per_genome)))


def population_chunk_size(
    pop_size: int,
    total_positions: int,
    target_cpu_gb: float,
    num_inputs: int = 3,
    num_outputs: int = 1,
    max_population_chunk: Optional[int] = None,
) -> int:
    """Genomes per chunk so that the weight matrices fit in a CPU RAM target.

    Beyond roughly depth 10 the full W1/W2/mask set for a whole population exceeds
    host RAM, so the population dimension has to be split. Per genome the cost is
    `(num_inputs + num_outputs) * total_positions` float32 weights plus one byte
    per position for the mask.

    Worked examples (verified against this implementation, not copied):
        depth 10 (5.6M positions), pop 1000 -> 95.2 MB per genome
            target_cpu_gb=16 -> 151 genomes;  target_cpu_gb=32 -> 302
        depth 11 (22.4M positions), pop 1000 -> 380.8 MB per genome
            target_cpu_gb=16 -> 37 genomes;   target_cpu_gb=32 -> 75

    The predecessor's docstring gave 168/336 and 42/84 here. Those come from
    dividing the raw budget by the per-genome cost and omit POPULATION_SAFETY_MARGIN,
    which the code has always applied -- the numbers above are what it returns.

    Args:
        pop_size: Full population size.
        total_positions: Candidate positions in the substrate.
        target_cpu_gb: Target host RAM for the weight matrices, in GB.
        num_inputs: Number of input nodes.
        num_outputs: Number of output nodes.
        max_population_chunk: Explicit override; returned (clamped) when set.

    Returns:
        Genomes per chunk, clamped to [MIN_POPULATION_CHUNK, pop_size].
    """
    if max_population_chunk is not None:
        return min(max_population_chunk, pop_size)

    bytes_per_genome = (
        (num_inputs + num_outputs) * total_positions * BYTES_PER_FLOAT32
        + total_positions  # bool mask
    )
    target_bytes = target_cpu_gb * BYTES_PER_GB
    chunk_size = int((target_bytes * POPULATION_SAFETY_MARGIN) / bytes_per_genome)
    return max(MIN_POPULATION_CHUNK, min(chunk_size, pop_size))


def recurrent_chunk_sizes(
    pop_size: int,
    positions: int,
    n_samples: int,
    activate_time: int,
    target_memory_gb: float = 2.0,
) -> Tuple[int, int]:
    """Population and sample chunk sizes for recurrent evaluation.

    `lax.scan` inside `vmap` is unrolled by XLA, so every iteration's hidden state
    is held live at once and memory grows with `activate_time`. Chunking both the
    population and the sample axis is what keeps that bounded; chunking population
    alone is not enough, which is the case this function exists for.

    Args:
        pop_size: Total population size.
        positions: Total hidden positions.
        n_samples: Samples in the evaluation batch.
        activate_time: Forward-pass iterations per evaluation.
        target_memory_gb: Target peak memory, in GB.

    Returns:
        Tuple of (pop_chunk_size, sample_chunk_size). The two are chosen equal
        before clamping, which minimizes their product's memory for a fixed budget.
    """
    per_genome_per_sample = positions * activate_time * BYTES_PER_FLOAT32
    target_bytes = target_memory_gb * BYTES_PER_GB
    chunk_product = target_bytes / per_genome_per_sample
    chunk_size = int(math.sqrt(chunk_product))
    pop_chunk = max(MIN_RECURRENT_CHUNK, min(chunk_size, pop_size))
    sample_chunk = max(MIN_RECURRENT_CHUNK, min(chunk_size, n_samples))
    return pop_chunk, sample_chunk


# Live copies the h->h discovery pipeline holds per genome at its peak, in bytes
# per active-set entry pair: dense weights (4), multi-hop weights (4), two bool
# validity masks (1+1), and the constraint/broadcast intermediates (~8) --
# recorded byte-exact pop x A^2 x 4 single allocations at A=692 and A=888 with
# several such tensors live at once.
DISCOVERY_BYTES_PER_PAIR = 18


def discovery_population_chunk(
    pop_size: int,
    num_active: int,
    target_gb: float = 2.0,
) -> int:
    """Genomes per h->h discovery chunk under the dense (A, A) working set.

    The discovery wall: `discover_sparse_hh_vectorized_multi_hop` materializes
    population-dense (pop, A, A) tensors over the active-set union. Per-genome
    work is independent once the union is fixed, so the population axis chunks
    exactly. Returns `pop_size` whenever the full working set fits, keeping
    every previously-feasible configuration byte-identical.

    Args:
        pop_size: Population size.
        num_active: Active-set union size A.
        target_gb: Working-set budget for one chunk, in GB.

    Returns:
        Genomes per chunk, in [1, pop_size].
    """
    if pop_size <= 0 or num_active <= 0:
        return max(pop_size, 1)
    per_genome = num_active * num_active * DISCOVERY_BYTES_PER_PAIR
    if pop_size * per_genome <= target_gb * BYTES_PER_GB:
        return pop_size
    return max(1, min(pop_size, int(target_gb * BYTES_PER_GB / per_genome)))


# Calibrated multiplier from single-buffer floor to true GPU working set for the
# STANDARD path. Two measured anchors (RTX 2080 Ti, 9.35 GB usable):
# d7/pop1000 (floor 2.88 GB) OOMed 0/10, so the true peak exceeds 9.35 GB and the
# factor must be >= 3.25; d6/pop1000 (floor 0.72 GB) ran clean, so 4x keeps it on
# the standard path with margin. 4 is calibrated, not derived -- the same lesson
# as the NEAT arm's measured ~6x transform working set.
STANDARD_WORKING_SET_FACTOR = 4


def standard_mode_floor_bytes(
    pop_size: int,
    total_positions: int,
    num_inputs: int = 3,
    num_outputs: int = 1,
) -> int:
    """Single-buffer floor of the STANDARD path's device-resident set.

    Counts W1 (num_inputs f32), W2 (num_outputs f32), the 4-float query output
    family (the recorded 1.398 GB d7 allocation is byte-exact pop x positions x
    16 B), and the 1-byte active mask, all per (genome, position). Excludes
    evaluation buffers and XLA transients -- hence a floor, corrected by
    `STANDARD_WORKING_SET_FACTOR`.
    """
    per_pair = BYTES_PER_FLOAT32 * (num_inputs + num_outputs + 4) + 1
    return pop_size * total_positions * per_pair
