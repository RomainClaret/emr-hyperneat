"""Several devices: the device split of the split routes (its query blocks and evaluation ranges), the
strategy names and their routes, the parallel grid, and the island model's plan, seeds and migration."""
import numbers
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.emrhyperneat_base import MultiGPUStrategy
from emr_hyperneat.engine.evaluation import evaluate_population_on_device
from emr_hyperneat.engine.substrate_build import QUERY_POPULATION_CHUNK, query_population_on_device


class DeviceSplit(NamedTuple):
    """The devices a split route spreads a generation's queries and evaluation over.

    A grid of ``population_devices`` x ``position_devices`` devices: device (i, j) is
    ``jax.devices()[device_ids[i * position_devices + j]]``. The query splits the genomes over the
    first axis and the grid positions over the second; the evaluation splits the genomes over every
    device. Results are gathered on ``device_ids[0]``, where the rest of the generation runs.
    """
    population_devices: int
    position_devices: int
    device_ids: Tuple[int, ...]


def make_device_split(population_devices: int, position_devices: int = 1) -> DeviceSplit:
    """A split over the first ``population_devices x position_devices`` devices JAX sees.

    Raises:
        ValueError: A count that is not a positive integer, or more devices than JAX sees.
    """
    for name, value in (
        ('population_devices', population_devices),
        ('position_devices', position_devices),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    count = population_devices * position_devices
    visible = len(jax.devices())
    if count > visible:
        raise ValueError(
            f"a {population_devices} x {position_devices} device split needs {count} devices; "
            f"JAX sees {visible}"
        )
    return DeviceSplit(population_devices, position_devices, tuple(range(count)))


def split_ranges(size: int, parts: int) -> List[Tuple[int, int]]:
    """``numpy.array_split``'s contiguous ranges of ``range(size)`` in ``parts`` pieces: the first
    ``size % parts`` one element longer, empty ranges when ``parts > size``."""
    base, extra = divmod(size, parts)
    ranges, start = [], 0
    for index in range(parts):
        stop = start + base + (1 if index < extra else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def query_blocks(
    pop_size: int,
    total_positions: int,
    split: DeviceSplit,
    pop_chunk_size: int = QUERY_POPULATION_CHUNK,
) -> List[Tuple[int, Tuple[int, int], Tuple[int, int]]]:
    """The blocks `query_population_split` queries: ``(device_id, genome range, position range)``,
    empty ones left out.

    A device queries its genomes in chunks of at most ``pop_chunk_size``, as one device does, so its
    largest call is ``min(pop_chunk_size, genomes) x positions`` whatever the population and the
    device count.
    """
    blocks = []
    for i, genomes in enumerate(split_ranges(pop_size, split.population_devices)):
        for j, positions in enumerate(split_ranges(total_positions, split.position_devices)):
            if genomes[1] > genomes[0] and positions[1] > positions[0]:
                blocks.append(
                    (split.device_ids[i * split.position_devices + j], genomes, positions)
                )
    return blocks


def query_population_split(
    state: Any,
    cppns_transformed: Tuple,
    rows: Sequence[jnp.ndarray],
    cppn_forward: Any,
    split: DeviceSplit,
    pop_chunk_size: int = QUERY_POPULATION_CHUNK,
) -> jnp.ndarray:
    """`query_population_on_device` spread over a split's devices: (pop_size, num_sources,
    total_positions).

    Each block of `query_blocks` runs `query_population_on_device` on its device with that device's
    compiled query, in chunks of ``pop_chunk_size`` genomes. The query is elementwise in genomes and
    positions, so every output is the one-device query's (bit for bit on a CPU, where every device
    compiles the same program). The blocks are dispatched without waiting for one another, then
    gathered on the split's first device.

    Args:
        state: Algorithm state.
        cppns_transformed: Tuple of 4 arrays, each (pop_size, ...).
        rows: Each source's rows (`substrate_query_rows`).
        cppn_forward: JIT-compiled CPPN forward function.
        split: The devices and their grid.
        pop_chunk_size: Genomes per query call on a device.

    Returns:
        (pop_size, num_sources, total_positions) CPPN outputs, on the split's first device.
    """
    devices = jax.devices()
    gather = devices[split.device_ids[0]]
    states: Dict[int, Any] = {}
    by_genomes: Dict[Tuple[int, int], List[jnp.ndarray]] = {}
    for device_id, (g0, g1), (p0, p1) in query_blocks(
        cppns_transformed[0].shape[0], rows[0].shape[0], split, pop_chunk_size
    ):
        device = devices[device_id]
        if device_id not in states:
            states[device_id] = jax.device_put(state, device)
        block = query_population_on_device(
            states[device_id],
            tuple(jax.device_put(part[g0:g1], device) for part in cppns_transformed),
            tuple(jax.device_put(source_rows[p0:p1], device) for source_rows in rows),
            cppn_forward, pop_chunk_size=pop_chunk_size, device_id=device_id)
        by_genomes.setdefault((g0, g1), []).append(block)
    rows_of_blocks = []
    for blocks in by_genomes.values():
        gathered = [jax.device_put(block, gather) for block in blocks]
        rows_of_blocks.append(
            gathered[0] if len(gathered) == 1 else jnp.concatenate(gathered, axis=2)
        )
    return (
        rows_of_blocks[0] if len(rows_of_blocks) == 1 else jnp.concatenate(rows_of_blocks, axis=0)
    )


def evaluation_ranges(
    pop_size: int, eval_chunk_size: Optional[int], devices: int
) -> Tuple[List[Tuple[int, int]], bool]:
    """The genome ranges a split's devices evaluate (empty ones left out), and whether they are one
    device's chunks.

    Unchunked (no ``eval_chunk_size``, or a population not above it): `split_ranges`'s near-equal
    ranges. Chunked: one device's chunks of ``eval_chunk_size`` genomes, dealt to the devices in
    contiguous groups, so every device evaluates whole chunks of one device's evaluation, the last
    device the last, partial one.
    """
    if not (eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size):
        return [r for r in split_ranges(pop_size, devices) if r[1] > r[0]], False
    num_chunks = -(-pop_size // eval_chunk_size)
    return [(a * eval_chunk_size, min(b * eval_chunk_size, pop_size))
            for a, b in split_ranges(num_chunks, devices) if b > a], True


def evaluate_population_split(
    forward_one: Callable,
    genome_args: Tuple,
    chunk_by_scan: bool,
    eval_chunk_size: Optional[int],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    fitness_mode: str,
    split: DeviceSplit,
) -> jnp.ndarray:
    """`evaluate_population_on_device` spread over every device of a split: (pop_size,) fitness on
    its first device.

    Each device evaluates a range of `evaluation_ranges` as one device evaluates a population: the
    genome arrays of the range and the data are placed on it, and it runs the one-device evaluation
    (whole chunks of one device's evaluation when that evaluation is chunked). A genome's fitness
    does not depend on the other genomes of its batch (two or more), so each equals one device's
    (bit for bit on a CPU). The ranges are dispatched without waiting for one another, then gathered
    on the split's first device.
    """
    devices = jax.devices()
    gather = devices[split.device_ids[0]]
    ranges, whole_chunks = evaluation_ranges(
        genome_args[0].shape[0], eval_chunk_size, len(split.device_ids)
    )
    parts = []
    for device_id, (g0, g1) in zip(split.device_ids, ranges):
        device = devices[device_id]
        parts.append(
            evaluate_population_on_device(
                forward_one,
                tuple(
                    None if arg is None else jax.device_put(arg[g0:g1], device)
                    for arg in genome_args
                ),
                chunk_by_scan,
                eval_chunk_size,
                jax.device_put(inputs, device),
                jax.device_put(targets, device),
                fitness_mode,
                whole_chunks=whole_chunks,
            )
        )
    gathered = [jax.device_put(part, gather) for part in parts]
    return gathered[0] if len(gathered) == 1 else jnp.concatenate(gathered)


# Main Class: EMRHyperNEAT

#: What a strategy name means to EMRHyperNEAT, through the constructor and the configuration key
#: multi_gpu_strategy alike: the primary strategies, the aliases, then the island model and its
#: alias. Every MultiGPUStrategy member has a route. The constructor does not use the base class's
#: normalization, which sends FULL_PIPELINE_PARALLEL, MULTI_GPU and DATA_PARALLEL to
#: PIPELINE_CHUNKED and CPPN_CHUNKED to POSITION_SHARDING_CHUNKED. 0.1.0's configuration key ran
#: CPPN_CHUNKED as STREAMING and POSITION_SHARDING_CHUNKED as FULL_PIPELINE_PARALLEL; the HMR engine
#: ran both as a CPPN query split by grid position.
STRATEGY_NAME_MAP: Dict[str, MultiGPUStrategy] = {
    'SINGLE_GPU': MultiGPUStrategy.SINGLE_GPU,
    'FULL_PIPELINE_PARALLEL': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
    'STREAMING': MultiGPUStrategy.STREAMING,
    'EVAL_ONLY_PARALLEL': MultiGPUStrategy.EVAL_ONLY_PARALLEL,
    'POPULATION_PARALLEL_PROCESS': MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
    'POPULATION_PMAP': MultiGPUStrategy.POPULATION_PMAP,
    'POSITION_SHARDING': MultiGPUStrategy.POSITION_SHARDING,
    'HYBRID': MultiGPUStrategy.HYBRID,
    'BASELINE': MultiGPUStrategy.SINGLE_GPU,
    'MULTI_GPU': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
    'DATA_PARALLEL': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
    'PMAP_PARALLEL': MultiGPUStrategy.EVAL_ONLY_PARALLEL,
    'PIPELINE_CHUNKED': MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
    'CPPN_CHUNKED': MultiGPUStrategy.POSITION_SHARDING,
    'POSITION_SHARDING_CHUNKED': MultiGPUStrategy.POSITION_SHARDING,
    'POPULATION_PARALLEL_SEQUENTIAL': MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
    'PERSISTENT_PARALLEL': MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
    'POPULATION_PARALLEL': MultiGPUStrategy.POPULATION_PMAP,
    'POSITION_PMAP': MultiGPUStrategy.POSITION_SHARDING,
    'SHARD_MAP': MultiGPUStrategy.POSITION_SHARDING,
    'ISLAND_MODEL': MultiGPUStrategy.ISLAND_MODEL,
    'ISLAND_MODEL_V2': MultiGPUStrategy.ISLAND_MODEL,
}

#: The split routes: the standard generation, its CPPN query and evaluation spread over a
#: DeviceSplit
SPLIT_STRATEGIES = (
    MultiGPUStrategy.POPULATION_PMAP,
    MultiGPUStrategy.POSITION_SHARDING,
    MultiGPUStrategy.HYBRID,
)

#: Keys of the emr_hyperneat.parallel block, the split routes' device grid
PARALLEL_KEYS = frozenset({'population_devices', 'position_devices'})


def parse_parallel_grid(
    block: Any, strategy: MultiGPUStrategy
) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """emr_hyperneat.parallel, the split routes' device grid: ``(population_devices,
    position_devices)``, None where a count is left to the route, or None without the block.

    POPULATION_PMAP splits the genomes and POSITION_SHARDING the grid positions, so each reads its
    own count (every visible device when it is absent) and refuses the other above 1. HYBRID splits
    both and needs both counts.

    Raises:
        ValueError: The block under another strategy, an unknown key, a count that is not a positive
            integer, a count the route does not split by, or HYBRID without both counts (or without
            the block).
    """
    if block is None:
        if strategy == MultiGPUStrategy.HYBRID:
            raise ValueError(
                "multi_gpu_strategy HYBRID splits the genomes and the grid positions at once: set "
                "emr_hyperneat.parallel.population_devices and position_devices"
            )
        return None
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f"emr_hyperneat.parallel sets the device grid of the split routes (POPULATION_PMAP, "
            f"POSITION_SHARDING, HYBRID); multi_gpu_strategy {strategy.name} reads "
            "none of it"
        )
    if not isinstance(block, dict):
        raise ValueError(
            f"emr_hyperneat.parallel must be a mapping of {sorted(PARALLEL_KEYS)}, got {block!r}"
        )
    unknown = sorted(set(block) - PARALLEL_KEYS)
    if unknown:
        raise ValueError(
            f"emr_hyperneat.parallel: unknown keys {unknown}; the keys are {sorted(PARALLEL_KEYS)}"
        )
    for key, value in block.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"emr_hyperneat.parallel.{key} must be a positive integer, got {value!r}"
            )
    population, position = block.get('population_devices'), block.get('position_devices')
    if strategy == MultiGPUStrategy.POPULATION_PMAP and position not in (None, 1):
        raise ValueError(
            "multi_gpu_strategy POPULATION_PMAP splits the genomes only: set position_devices with "
            "HYBRID, or leave it out"
        )
    if strategy == MultiGPUStrategy.POSITION_SHARDING and population not in (None, 1):
        raise ValueError("multi_gpu_strategy POSITION_SHARDING splits the grid positions only: set "
                         "population_devices with HYBRID, or leave it out")
    if strategy == MultiGPUStrategy.HYBRID and (population is None or position is None):
        raise ValueError(
            "multi_gpu_strategy HYBRID splits the genomes and the grid positions at once: set both "
            "emr_hyperneat.parallel.population_devices and position_devices"
        )
    return population, position


#: Keys of the emr_hyperneat.islands block, the island model's (multi_gpu_strategy ISLAND_MODEL)
ISLAND_KEYS = frozenset(
    {'count', 'populations', 'devices', 'migration_interval', 'migrants', 'topology'}
)
#: Who sends migrants to whom: ring (island i to island i + 1), star (every island to every other,
#: HMR's V2), random (each island to one other, drawn from the run's seed and the migration event)
ISLAND_TOPOLOGIES = ('ring', 'star', 'random')


class IslandPlan(NamedTuple):
    """An ISLAND_MODEL run's islands (emr_hyperneat.islands, checked by parse_islands)."""
    count: int
    populations: Tuple[int, ...]
    devices: Optional[Tuple[int, ...]]
    migration_interval: int
    migrants: int
    topology: str


def island_seed(seed: int, island: int) -> int:
    """An island's seed: the run's for island 0, and for island i one drawn from (seed, i) by
    numpy's SeedSequence.

    HMR seeded island i with seed + i, so island 1 of seed 42 was island 0 of seed 43, and two
    replicates of a seeded study shared a population. These seeds depend on neither the populations
    nor the devices.
    """
    if island == 0:
        return int(seed)
    return int(np.random.SeedSequence([int(seed), int(island)]).generate_state(1)[0] & 0x7FFFFFFF)


def migration_targets(plan: IslandPlan, seed: int, event: int) -> List[List[int]]:
    """For each island, the islands its emigrants go to at migration ``event`` (0, 1, ...): the next
    one (ring), every other (star), or one other drawn from the run's seed and the event (random),
    never from an island's key, so the islands' own random streams are those of runs without
    migration."""
    count = plan.count
    if plan.topology == 'ring':
        return [[(i + 1) % count] for i in range(count)]
    if plan.topology == 'star':
        return [[j for j in range(count) if j != i] for i in range(count)]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(event))
    return [
        [(i + int(jax.random.randint(jax.random.fold_in(key, i), (), 1, count))) % count]
        for i in range(count)
    ]


def remap_hidden_keys(nodes: np.ndarray, conns: np.ndarray, io_keys: np.ndarray, first_key: int
                      ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Migrants' genes (genomes x rows x attributes, the key in column 0 of a node row and the two
    ends in columns 0 and 1 of a connection row) with their hidden node keys moved to ``first_key``
    and up, one map for all of them, so migrants that share a node in the sender share it in the
    receiver. A receiver numbers its new nodes from its population's largest key and aligns
    crossover by key; without the move a migrant's hidden node would be taken for an unrelated node
    of the receiver that happens to bear the same key. EMR's CPPN connections carry no historical
    marker (DefaultConn), so the node keys are the whole remap.

    Returns:
        The remapped nodes and connections, and the next free key.
    """
    nodes, conns = np.array(nodes, copy=True), np.array(conns, copy=True)
    keys = nodes[..., 0]
    hidden = np.unique(keys[~np.isnan(keys) & ~np.isin(keys, io_keys)])
    mapping = {float(old): float(first_key + rank) for rank, old in enumerate(hidden)}
    if mapping:
        def moved(values):
            out = values.copy()
            for old, new in mapping.items():
                out[values == old] = new
            return out
        nodes[..., 0] = moved(keys)
        conns[..., 0] = moved(conns[..., 0])
        conns[..., 1] = moved(conns[..., 1])
    return nodes, conns, first_key + len(mapping)


def parse_islands(block: Any, strategy: MultiGPUStrategy, population_size: int, genome_elitism: int,
                  visible_devices: int) -> Optional[IslandPlan]:
    """emr_hyperneat.islands, the island model's islands, checked at create_config.

    ``count`` islands (at least 2) share ``population_size``: equally by default, which needs
    ``population_size`` divisible by ``count``, or as ``populations`` gives, which must sum to it.
    An island needs more genomes than ``genome_elitism``, or it only copies itself. ``devices``
    places island i (default: i mod the visible devices); placement never changes results.
    ``migration_interval`` is required (0: no migration); ``migrants`` and ``topology`` are read
    only with migration, and a receiver's best ``genome_elitism`` genomes are never replaced.

    Raises:
        ValueError: The block under another strategy, ISLAND_MODEL without it, an unknown key, or a
            missing or wrong value (the message names the key).
    """
    if block is None:
        if strategy == MultiGPUStrategy.ISLAND_MODEL:
            raise ValueError(
                "multi_gpu_strategy ISLAND_MODEL needs emr_hyperneat.islands (count, "
                "migration_interval, "
                "and optionally populations, devices, migrants, topology)"
            )
        return None
    if strategy != MultiGPUStrategy.ISLAND_MODEL:
        raise ValueError(
            f"emr_hyperneat.islands configures the island model (multi_gpu_strategy ISLAND_MODEL); "
            f"multi_gpu_strategy {strategy.name} reads none of it"
        )
    if not isinstance(block, dict):
        raise ValueError(
            f"emr_hyperneat.islands must be a mapping of {sorted(ISLAND_KEYS)}, got {block!r}"
        )
    unknown = sorted(set(block) - ISLAND_KEYS)
    if unknown:
        raise ValueError(
            f"emr_hyperneat.islands: unknown keys {unknown}; the keys are {sorted(ISLAND_KEYS)}"
        )

    def integer(key: str, value: Any, low: int) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < low:
            raise ValueError(
                f"emr_hyperneat.islands.{key} must be an integer of at least {low}, got {value!r}"
            )
        return int(value)

    for key in ('count', 'migration_interval'):
        if key not in block:
            raise ValueError(f"emr_hyperneat.islands.{key} is required")
    count = integer('count', block['count'], 2)
    interval = integer('migration_interval', block['migration_interval'], 0)
    if 'populations' in block:
        given = block['populations']
        if not isinstance(given, (list, tuple)) or len(given) != count:
            raise ValueError(
                f"emr_hyperneat.islands.populations must list {count} island populations, got "
                f"{given!r}"
            )
        populations = tuple(integer('populations', value, 1) for value in given)
        if sum(populations) != population_size:
            raise ValueError(
                f"emr_hyperneat.islands.populations sum to {sum(populations)}, not population_size "
                f"{population_size}: the islands share the population"
            )
    else:
        if population_size % count:
            near = [
                population_size // count + (1 if i < population_size % count else 0)
                for i in range(count)
            ]
            raise ValueError(
                f"population_size {population_size} does not split into {count} equal islands: set "
                f"emr_hyperneat.islands.populations, for example {near}"
            )
        populations = (population_size // count,) * count
    small = [i for i, population in enumerate(populations) if population <= genome_elitism]
    if small:
        raise ValueError(f"emr_hyperneat.islands: islands {small} of {list(populations)} genomes "
                         f"hold no more than neat_selection.genome_elitism {genome_elitism}, so "
                         "they would only copy themselves; give them more genomes or lower "
                         "genome_elitism")
    devices = None
    if 'devices' in block:
        given = block['devices']
        if not isinstance(given, (list, tuple)) or len(given) != count:
            raise ValueError(
                f"emr_hyperneat.islands.devices must list {count} device indices, got {given!r}"
            )
        devices = tuple(integer('devices', value, 0) for value in given)
        if max(devices) >= visible_devices:
            raise ValueError(
                f"emr_hyperneat.islands.devices {list(devices)} names a device beyond the "
                f"{visible_devices} visible"
            )
    if interval == 0:
        idle = sorted({'migrants', 'topology'} & set(block))
        if idle:
            raise ValueError(f"emr_hyperneat.islands.{idle[0]} is read only with migration, and "
                             "migration_interval is 0")
        return IslandPlan(count, populations, devices, 0, 0, 'ring')
    if 'migrants' not in block:
        raise ValueError("emr_hyperneat.islands.migrants is required with migration "
                         "(migration_interval above 0)")
    migrants = integer('migrants', block['migrants'], 1)
    topology = block.get('topology', 'ring')
    if topology not in ISLAND_TOPOLOGIES:
        raise ValueError(
            f"emr_hyperneat.islands.topology must be one of {list(ISLAND_TOPOLOGIES)}, got "
            f"{topology!r}"
        )
    inflow = migrants * (count - 1) if topology == 'star' else migrants
    crowded = [
        i for i, population in enumerate(populations) if inflow > population - genome_elitism
    ]
    if crowded:
        raise ValueError(
            f"emr_hyperneat.islands: {inflow} immigrants per migration would replace part of the "
            f"best genome_elitism {genome_elitism} genomes of islands {crowded}; lower migrants"
        )
    return IslandPlan(count, populations, devices, interval, migrants, topology)
