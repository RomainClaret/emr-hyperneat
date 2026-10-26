"""Execution strategies for EMR-HyperNEAT.

EMR has three ways of getting through one generation. They compute the same thing
and differ only in where the intermediate arrays live:

- `STANDARD` queries every candidate position at once, with the highest peak
  memory. This is the path the published results and the golden tests lock,
  so it is the default and it is never routed through this package.
- `STREAMING` builds the weight matrices level by level, so one level's queries, and the next
  level's input queries for its variance, are resident at once; W1 and W2 are still assembled
  at full size.
- `DEPTH_OFFLOAD` also keeps the assembled matrices in host RAM (or a
  memmap) and moves each chunk to the device only to evaluate it. This is the
  only mode that reaches depth 11+, where the matrices exceed device memory
  outright.

`STREAMING` swaps one step inside the normal generation loop, so it composes with
everything else the loop does. `DEPTH_OFFLOAD` cannot: never materializing the
full matrices means its build and its evaluation are fused, so it replaces the
generation instead of a step within it. That asymmetry is why the dispatcher has
two extension points rather than one.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

__all__ = [
    "ExecutionMode",
    "ExecutionConfig",
    "EXECUTION_KEYS",
    "GENERATION_LOOPS",
    "resolve_mode",
    "parse_execution_config",
]


#: execution.generation_loop: how run_until_threshold iterates its generations on one device.
#: 'python': one generation call per generation, every feature. 'while_loop': the HMR engine's GPU
#: loop, the plain feed-forward generation compiled inside jax.lax.while_loop; performance only, up
#: to rounding.
GENERATION_LOOPS = ("python", "while_loop")


class ExecutionMode(str, Enum):
    """How a generation moves its intermediate arrays through memory.

    This is a memory-residency axis and is orthogonal to `MultiGPUStrategy`, which
    is a device-parallelism axis. Do not merge them: a run can stream *and* be
    population-parallel, and collapsing the two would make that inexpressible.

    `AUTO` is resolved to one of the concrete modes by `resolve_mode`, using depth.
    It is never the value a strategy sees.
    """

    AUTO = "auto"
    STANDARD = "standard"
    STREAMING = "streaming"
    DEPTH_OFFLOAD = "depth_offload"


@dataclass
class ExecutionConfig:
    """Every knob controlling execution mode and its memory budgets.

    The config file and the imperative `configure_offloading` / `configure_sharding`
    setters write the same fields here, so a YAML-driven experiment reaches every
    one of them.

    Defaults reproduce the pre-unification behavior of the standard path exactly:
    `mode` is STANDARD, so nothing about an existing run changes unless asked.
    """

    # mode selection
    mode: ExecutionMode = ExecutionMode.STANDARD
    depth_threshold: int = 9
    """Depth at or above which AUTO selects DEPTH_OFFLOAD."""
    streaming_depth_threshold: int = 8
    """Depth at or above which AUTO selects STREAMING."""

    # CPPN query chunking
    cppn_pos_chunk_size: int = 100_000
    target_shard_gb: float = 0.5
    max_lookahead_positions: int = 5_000_000
    """Levels larger than this skip the variance lookahead.

    Skipping is not free: zeroed variance halts subdivision at and below that
    level, so the discovered substrate changes. It is therefore opt-in rather than
    an invisible consequence of depth (see `truncate_variance_lookahead`).
    """
    truncate_variance_lookahead: bool = False
    """Whether `max_lookahead_positions` may alter the substrate. Default off.

    Off, a level larger than the budget is computed in slabs: bounded memory, exact
    substrate. On, it is zeroed, which halts subdivision at and below that level.
    """
    skip_dead_levels: bool = True
    """Stop descending once a level has no reached positions.

    Behaviorally identical (an unreached level contributes exactly zeros through
    `jnp.where(mask & ..., W1_raw, 0.0)` against a pre-zeroed buffer), but it changes
    the cost profile, so it is on by default and can be turned off. Acts on
    the depth-offload path only. Reach and fitness are the same with it off
    (`test_depth_lookahead.py`); the saving grows with depth, because at the default
    threshold subdivision stops at level 3 and every level below is empty.
    """

    # evaluation chunking
    eval_chunk_levels: int = 3
    eval_chunk_positions: int = 1_000_000
    target_eval_shard_gb: float = 1.0
    recurrent_target_memory_gb: float = 2.0

    # host residency
    target_cpu_gb: float = 32.0
    max_population_chunk: Optional[int] = None
    use_cpu_grid: bool = False
    """Keep the grid itself in host RAM. Needed from depth 12, where the grid
    structure alone exceeds device memory."""

    # spill to disk
    use_memmap: bool = True
    memmap_path: str = "/tmp/emr_memmap"
    enable_disk_offload: bool = False
    disk_offload_path: str = "/tmp/emr_offload"

    # active-set compaction
    compaction_bucket: str = "auto"
    """Shape of the compacted recurrent evaluation space: `auto` or `exact`.

    The recurrent evaluation runs on the population's union of reached positions, a size
    that changes from generation to generation once the union leaves part of the grid unused
    (depth 3 and deeper). `exact` evaluates on exactly that many positions, so every step
    after compaction meets a new shape and recompiles (18 to 54 executables per generation
    on two RTX 2080 Ti at depth 3, population 1000). `auto` pads the compact space to
    a bucketed size with virtual positions after the real ones: zero weights in and out, no
    hidden-to-hidden connection, mask False, so a bucket size compiles once and results do
    not change. Acts on the standard and streaming paths; depth offload keeps exact shapes.
    """

    # the generation loop
    generation_loop: str = "python"
    """How run_until_threshold iterates its generations on one device: `python` or `while_loop`.

    `while_loop` (the HMR engine's GPU loop) compiles the plain feed-forward generation inside
    `jax.lax.while_loop` and reads the results back once. Performance only, up to rounding: the
    fitness can differ from `python`'s by a float32 ulp, the generation count, stop and key do not.
    It refuses every setting outside that generation (hidden-to-hidden connections, per-node
    functions other than `global`, LEO, neuromodulation, multi-task, streaming, depth offload,
    palettes, the node-list model, the multi-device routes).
    """

    # diagnostics
    verbose: bool = False

    def resolve(self, max_depth: int) -> ExecutionMode:
        """Resolve AUTO against a depth. See `resolve_mode`."""
        return resolve_mode(self, max_depth)


def resolve_mode(
    config: ExecutionConfig,
    max_depth: int,
    *,
    population: "Optional[int]" = None,
    num_inputs: int = 3,
    num_outputs: int = 1,
    device_bytes: "Optional[int]" = None,
) -> ExecutionMode:
    """Turn a possibly-AUTO mode into the concrete mode to run.

    Thresholds are checked from the most memory-conservative downwards, so a depth
    past both thresholds gets DEPTH_OFFLOAD rather than STREAMING.

    Memory escalation: the depth thresholds alone routed
    d7/pop1000 to STANDARD on an 11 GB GPU, where its working set does not fit
    (0/10 OOM, protocol GPU pass). When the caller supplies `population` and
    `device_bytes`, a STANDARD selection is escalated to DEPTH_OFFLOAD if the
    standard path's calibrated working-set estimate exceeds the device. With
    either argument absent the escalation is inert, so every existing caller
    and every CPU run resolves exactly as before.

    Args:
        config: The execution configuration.
        max_depth: Configured maximum substrate depth.
        population: Genomes per generation (for the escalation).
        num_inputs: Substrate input count (default XOR shape).
        num_outputs: Substrate output count.
        device_bytes: Usable accelerator memory, when the caller wants the
            escalation (pass None on CPU).

    Returns:
        A concrete mode, never AUTO.
    """
    if config.mode is not ExecutionMode.AUTO:
        return config.mode
    if max_depth >= config.depth_threshold:
        return ExecutionMode.DEPTH_OFFLOAD
    if max_depth >= config.streaming_depth_threshold:
        return ExecutionMode.STREAMING
    if population is not None and device_bytes is not None:
        from .chunking import STANDARD_WORKING_SET_FACTOR, standard_mode_floor_bytes
        total_positions = (4 ** (max_depth + 2) - 4) // 3
        floor = standard_mode_floor_bytes(
            population, total_positions, num_inputs, num_outputs)
        if floor * STANDARD_WORKING_SET_FACTOR > device_bytes:
            return ExecutionMode.DEPTH_OFFLOAD
    return ExecutionMode.STANDARD


# Keys accepted directly under the `execution:` block, mapped to dataclass fields.
_DIRECT_KEYS = (
    "depth_threshold",
    "streaming_depth_threshold",
    "cppn_pos_chunk_size",
    "target_shard_gb",
    "max_lookahead_positions",
    "truncate_variance_lookahead",
    "skip_dead_levels",
    "eval_chunk_levels",
    "eval_chunk_positions",
    "target_eval_shard_gb",
    "recurrent_target_memory_gb",
    "target_cpu_gb",
    "max_population_chunk",
    "use_cpu_grid",
    "use_memmap",
    "memmap_path",
    "enable_disk_offload",
    "disk_offload_path",
    "compaction_bucket",
    "generation_loop",
    "verbose",
)
#: Every key of the `execution:` block; EMRHyperNEAT.create_config raises on any other, so a
#: misspelled setting cannot silently run the default.
EXECUTION_KEYS = frozenset(("mode",) + _DIRECT_KEYS)


def parse_execution_config(
    emr_config: Dict[str, Any],
    base: Optional[ExecutionConfig] = None,
) -> ExecutionConfig:
    """Build an `ExecutionConfig` from the `emr_hyperneat` config block.

    Accepts both the new `execution:` block and the older top-level
    `enable_streaming` flag. The old flag is not deprecated away here: it is what
    the existing streaming tests and several runners set, and it stays a valid
    spelling of `mode: streaming`. An explicit `execution.mode` wins over it.

    Keys absent from the config are left as they are on `base` rather than reset to
    the class defaults. That is what lets `configure_offloading()` be called before
    `create_config()` without the config parse silently discarding it: the two
    ways of setting execution options have to compose, not compete.

    Args:
        emr_config: The `emr_hyperneat` sub-dictionary of an algorithm config.
        base: Existing configuration to update. A fresh default when omitted.

    Returns:
        The updated configuration (the `base` object itself when one was given).
        With no `execution` block and no base, this is the standard path unchanged.

    Raises:
        ValueError: If `execution.mode` is not one of the `ExecutionMode` values,
            or if `execution` is not a mapping.
    """
    config = ExecutionConfig() if base is None else base
    block = emr_config.get("execution") or {}

    if not isinstance(block, dict):
        raise ValueError(
            f"emr_hyperneat.execution must be a mapping, got {type(block).__name__}"
        )

    raw_mode = block.get("mode")
    if raw_mode is not None:
        try:
            config.mode = ExecutionMode(str(raw_mode).lower())
        except ValueError as exc:
            valid = ", ".join(mode.value for mode in ExecutionMode)
            raise ValueError(
                f"Unknown execution mode {raw_mode!r}. Valid modes: {valid}"
            ) from exc
    elif emr_config.get("enable_streaming"):
        config.mode = ExecutionMode.STREAMING

    for key in _DIRECT_KEYS:
        if key in block:
            setattr(config, key, block[key])
    if config.compaction_bucket not in ("auto", "exact"):
        raise ValueError(
            "execution.compaction_bucket must be 'auto' or 'exact', got "
            f"{config.compaction_bucket!r}"
        )
    if config.generation_loop not in GENERATION_LOOPS:
        raise ValueError(
            "execution.generation_loop must be 'python' or 'while_loop', got "
            f"{config.generation_loop!r}"
        )

    return config
