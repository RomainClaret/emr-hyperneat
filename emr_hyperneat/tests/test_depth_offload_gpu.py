"""GPU-only checks for the `depth_offload` execution mode.

These cannot run on CPU. Offload exists to keep a generation's intermediate arrays off
a device that cannot hold them, and a CPU backend has no such limit to hit, so the CPU
suite can only show that the path is *reached*. `test_depth_offload.py` does that: it
verifies that `run_until_threshold` dispatches to the offload path and that the result
matches `run_generation`. What is left, and what these tests cover, is the reason the
mode exists: that a configuration whose standard-path working set does not fit the
device still completes.

The depth is chosen from the device rather than pinned, so this is correct on an 11 GB
card and on an 80 GB one: the test asks the package's own calibrated estimator which
depth the standard path could not hold, and skips if no depth in range qualifies.

The device figure is what JAX can allocate (`bytes_limit`, the preallocated pool),
which is also what `resolve_mode`'s escalation uses; on an RTX 2080 Ti that is 8.4 GB
of 11. The estimate is a floor times an approximate factor, so the chosen depth must
exceed the pool by a margin, or a passing offload run proves nothing about whether
standard would have failed. The negative is asserted directly, last, since a device
out-of-memory can leave the allocator fragmented for whatever runs after it.
"""

import pytest

import jax

from conftest import (
    EMRHyperNEAT,
    create_base_config,
    XORProblem,
    DEFAULT_SEED,
)
from emr_hyperneat.execution import ExecutionMode
from emr_hyperneat.execution.chunking import (
    STANDARD_WORKING_SET_FACTOR,
    standard_mode_floor_bytes,
)

# Small enough that evaluation is not the cost under test, large enough that the
# working-set estimate crosses a consumer card at a depth that still runs.
POPULATION = 50

# The contract is that the loop ran, dispatched to offload, and came back without an
# out-of-memory failure; the second generation also drives the offload path's h->h
# cache pass-through on the device (the pass that once repeated generation 0). It costs
# little: 63 s against 48 s for one generation at depth 9 on an RTX 2080 Ti. Longer
# looping is covered on CPU, where it is cheap.
GENERATIONS = 2

# Above any achievable fitness, so the loop always spends its whole budget.
UNREACHABLE_FITNESS = 2.0

# Searched from the shallowest depth whose arrays plausibly exceed a device, up to the
# deepest the package claims to reach.
DEPTH_SEARCH_RANGE = range(8, 14)

# The estimate must exceed the pool by at least this factor. It is a floor times
# STANDARD_WORKING_SET_FACTOR (an approximation), so a depth that clears the pool by
# 1.1x (depth 9 on an 8.4 GB pool at population 50: 9.23 GB) may in fact fit; 2x is
# past any plausible error in the factor. On that pool this selects depth 10 (4.4x).
MIN_MARGIN = 2.0


def _gpu_devices():
    return [d for d in jax.devices() if d.platform == 'gpu']


def _device_bytes():
    """Usable device memory, by the same route `resolve_execution_mode` uses."""
    devices = _gpu_devices()
    if not devices:
        return None
    try:
        stats = devices[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - backends differ on what they expose
        return None
    return stats.get('bytes_limit')


def _total_positions(max_depth):
    """Candidate positions on the multi-resolution grid, as `resolve_mode` counts them."""
    return (4 ** (max_depth + 2) - 4) // 3


def _smallest_depth_exceeding(device_bytes, pop_size=POPULATION):
    """Shallowest depth whose standard-path working set does not fit `device_bytes`.

    Uses the package's own estimator, so the test agrees with the escalation rule
    rather than second-guessing it with a hand-rolled figure.
    """
    for depth in DEPTH_SEARCH_RANGE:
        floor = standard_mode_floor_bytes(pop_size, _total_positions(depth))
        if floor * STANDARD_WORKING_SET_FACTOR >= MIN_MARGIN * device_bytes:
            return depth
    return None


@pytest.fixture(scope='module')
def deep_depth():
    """The depth under test, or a skip explaining which precondition was missing."""
    if not _gpu_devices():
        pytest.skip("Requires a GPU device")
    device_bytes = _device_bytes()
    if not device_bytes:
        pytest.skip("Backend reports no bytes_limit; cannot size the test to the device")
    depth = _smallest_depth_exceeding(device_bytes)
    if depth is None:
        pytest.skip(
            f"No depth in {DEPTH_SEARCH_RANGE} exceeds {device_bytes / 1e9:.1f} GB "
            f"at population {POPULATION}")
    return depth


def _config(max_depth, mode):
    config = create_base_config(max_depth=max_depth, population_size=POPULATION)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
        'mode': mode,
    }
    return config


@pytest.mark.cuda
@pytest.mark.slow
class TestOffloadOnDevice:
    """The mode's actual promise, which only a memory-limited device can test."""

    def test_offload_completes_where_standard_would_not_fit(self, deep_depth):
        """`run_until_threshold` + `depth_offload` survives a depth standard cannot hold.

        The CPU suite proves the dispatch happens. This proves the dispatch is worth
        having: on the standard on-device path this configuration would hit exactly
        the out-of-memory failure offload exists to prevent.
        """
        algorithm = EMRHyperNEAT()
        config_obj = algorithm.create_config(_config(deep_depth, 'depth_offload'))
        assert algorithm.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD

        problem = XORProblem()
        state = algorithm.initialize(config_obj, problem, seed=DEFAULT_SEED)
        result = algorithm.run_until_threshold(
            state, problem, UNREACHABLE_FITNESS, GENERATIONS)

        assert result['generations'] == GENERATIONS
        # Only execution/offload.py ever increments this, so a positive count is
        # independent evidence that the offload path ran rather than the standard one.
        assert algorithm._gpu_transfers_count > 0
        best = float(result['best_fitness'])
        assert best == best, "best_fitness is NaN"  # NaN is the OOM-adjacent failure
        assert best > 0

    def test_auto_escalates_to_offload_on_this_device(self, deep_depth):
        """`auto` must reach the same decision from the device, not just the depth.

        The memory-aware escalation in `resolve_mode` is inert on CPU, where the caller
        passes `device_bytes=None`, so this branch has no CPU coverage at all.
        """
        algorithm = EMRHyperNEAT()
        algorithm.create_config(_config(deep_depth, 'auto'))
        assert algorithm.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD

    def test_standard_does_not_fit_at_that_depth(self, deep_depth):
        """The negative half of the contract: standard really is out of memory here.

        Without this, the offload test above can pass at a depth the standard path
        could also have handled, proving nothing. Defined last on purpose: an XLA
        out-of-memory can fragment the allocator for whatever runs next in-process.
        Matched on the message, not the type: the exception class has moved between
        jaxlib versions, the RESOURCE_EXHAUSTED text has not.
        """
        algorithm = EMRHyperNEAT()
        config_obj = algorithm.create_config(_config(deep_depth, 'standard'))
        assert algorithm.resolve_execution_mode() is ExecutionMode.STANDARD
        problem = XORProblem()
        state = algorithm.initialize(config_obj, problem, seed=DEFAULT_SEED)
        with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
            algorithm.run_until_threshold(state, problem, UNREACHABLE_FITNESS, 1)
