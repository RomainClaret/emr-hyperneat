"""The compiled feed-forward forward of the substrate API against the dense one.

`substrate_forward(..., compiled=True)` evaluates a feed-forward substrate genome by genome on its
own mask positions, zero-padded to the recurrent forward's power-of-two bucket, the genomes of a
bucket in fixed slots of one jitted call, the inputs in chunks of ``chunk_size`` rows (rounded up to
a multiple of 8). It computes the dense forward's function on the mask's positions, so on a
substrate as built (zero outside the mask) it equals the dense forward within float32 rounding, per
output function within the allowances measured below, not bit for bit. It is one program per bucket
size, chunk size, settings and operand shapes: on the CPU under single-threaded Eigen a row's
outputs do not depend on its chunk, its place in the chunk, its genome's slot, the other genomes
or a repeat, bit for bit, with one output or three. Under XLA's default multi-threaded Eigen a
Xeon E5-2683 v4 rounded rows apart by their place in the chunk (three outputs); a repeat stays
bit for bit.

Also covered: an empty genome and a substrate cut to one silent position, row counts around the
chunk size, two chunk sizes as two programs (apart at bucket 2,048 with three outputs), the compile
count (another neurotransmitter vector or strength compiles nothing, a new bucket compiles once,
another genome count in a bucket nothing, a setting the forward does not read nothing, another
output count each bucket once more), the function cache's key and bound, the refusals, the mask rule
on an edited substrate, NaN and inf where the dense forward has them, the exception to that (a value
within the two forwards' rounding difference of a float32 overflow threshold can overflow in one of
them only: a softplus output, whose argument the two forwards sum in different orders, and a
softplus node after a step the compiled program fuses), and the neuromodulation checks.
"""

import functools
import platform
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    CPU_CROSS_PROGRAM_ULPS,
    GPU_TRAJECTORY_RTOL,
    assert_close_across_programs,
    cpu_multi_threaded_eigen,
    create_config_with_dynamic_functions,
    create_config_with_recurrence,
    exact_backend,
)
from emr_hyperneat import (
    build_recurrent_substrate,
    query_recurrent_substrate,
    substrate_forward,
)
from emr_hyperneat import emrhyperneat as engine
from emr_hyperneat import substrate as api
import test_substrate_api as feedforward  # type: ignore[import-not-found]

DEPTH = feedforward.DEPTH
POP = feedforward.POP
#: The output functions every configuration is compared under: the configured sigmoid,
#: the tanh and sin of a caller that draws pictures, and two unbounded ones
OUTPUTS = ("sigmoid", "tanh", "sin", "relu", "identity")
# Compiled against dense on the CPU (an Apple M4 Max, jax 0.6.1,
# XLA_FLAGS=--xla_cpu_multi_thread_eigen=false): every configuration of
# test_substrate_api.py's CONFIGS and NM_CONFIGS under each of OUTPUTS, on XOR's rows, 256 random
# rows and a 32 x 32 raster of (x, y, 1). The two differ in float32 rounding only: the
# hidden-to-output sum runs over the genome's positions instead of the grid's, in another order,
# and XLA fuses the compiled program (a multiply and an add rounded once where the dense forward,
# op by op, rounds twice: rs_adapt, resonator, osc_adapt, integrate, exact and global sum and mean
# aggregation, the paper rule's gains * pre + bias, the gain bg + s m at a strength that is not a
# power of two, the inversion). The largest difference, in ulps of max(1, the genome's largest
# pre-activation magnitude |h @ W2|), which the rounding of both causes scales with at these
# settings (sin's and the unbounded functions' with the argument, not the output; product
# aggregation under neuromodulation can sum terms far larger than the sum, past these allowances:
# 0.47 in the sum itself, up to 865 ulps of the genome's largest |h @ W2|, with an aggregation
# palette holding product at strengths 0.7 to 3.5, docs/architecture.md):
#   sigmoid 1.6 (9.5e-7), tanh 3.9 (3.8e-6), sin 6.6 (3.0e-5), relu 4.0 (1.5e-5),
#   identity 8.0 (3.1e-5, 8 ulps of an output of 187).
# The allowance is eight times that, to the next power of two; a dropped node moves an output by
# 1e-2 or more (test_the_allowance_sees_a_dropped_node). On 19 recorded cases of picture drawing
# (16,384 rows, three outputs) the largest differences were 1.79e-3 under sin outputs, 3.5e-5 under
# sigmoid and 1.5e-5 under tanh (docs/architecture.md).
ALLOWANCE_ULPS = {"sigmoid": 16, "tanh": 32, "sin": 64, "relu": 32, "identity": 64}
SAMPLES = 256
RASTER = 32


def _raster(side: int) -> np.ndarray:
    axis = (2.0 * np.arange(side, dtype=np.float32) + 1.0) / side - 1.0
    x, y = np.meshgrid(axis, axis)
    return np.stack([x.ravel(), y.ravel(), np.ones(side * side, dtype=np.float32)], axis=1)


def _inputs(problem: Any) -> np.ndarray:
    """XOR's rows, SAMPLES random rows in [-1, 1] and a RASTER x RASTER raster of (x, y, 1)."""
    random = np.random.default_rng(0).uniform(-1.0, 1.0, (SAMPLES, 3)).astype(np.float32)
    return np.concatenate([np.asarray(problem.inputs, dtype=np.float32), random, _raster(RASTER)])


@functools.lru_cache(maxsize=None)
def _population(kind: str, name: str) -> Tuple[Any, np.ndarray]:
    """A configuration's population as test_substrate_api.py builds it, and the inputs (the arrays
    are not modified)."""
    configs = feedforward.CONFIGS if kind == "ff" else feedforward.NM_CONFIGS
    algo, problem, state = feedforward._setup(configs[name]())
    return algo.population_substrates(state), _inputs(problem)


def _with_output(substrate: Any, function: str) -> Any:
    return substrate._replace(config=substrate.config._replace(output_activation=function))


def _pre_activation_scale(substrate: Any, inputs: Any,
                          neuromodulation: Optional[Any] = None) -> np.ndarray:
    """Per genome, max(1, the largest |h @ W2|): the dense forward with the identity output and the
    inversion off."""
    if neuromodulation is None and substrate.config.neuromodulation is not None:
        neuromodulation = api.substrate_neuromodulation(substrate)
    if neuromodulation is not None:
        neuromodulation = neuromodulation._replace(
            settings=neuromodulation.settings._replace(output_inversion=False))
    pre = np.asarray(substrate_forward(_with_output(substrate, "identity"), inputs,
                                       neuromodulation=neuromodulation))
    return np.maximum(1.0, np.abs(pre).max(axis=(1, 2))).astype(np.float32)


def _assert_within_allowance(compiled: Any, dense: Any, scale: np.ndarray, function: str) -> None:
    """Compiled against dense: equal finiteness, and within the function's allowance in ulps of each
    genome's scale (on a GPU, also GPU_TRAJECTORY_RTOL of it)."""
    compiled, dense = np.asarray(compiled), np.asarray(dense)
    assert compiled.shape == dense.shape and compiled.dtype == np.float32
    np.testing.assert_array_equal(np.isfinite(compiled), np.isfinite(dense), err_msg="finiteness")
    np.testing.assert_array_equal(np.isnan(compiled), np.isnan(dense), err_msg="NaN")
    allowance = ALLOWANCE_ULPS[function] * np.spacing(scale)
    if not exact_backend():
        allowance = np.maximum(allowance, GPU_TRAJECTORY_RTOL * scale)
    finite = np.isfinite(dense)
    # Infinite outputs must agree in sign
    np.testing.assert_array_equal(compiled[~finite & ~np.isnan(dense)],
                                  dense[~finite & ~np.isnan(dense)])
    with np.errstate(invalid="ignore"):
        excess = np.where(finite, np.abs(compiled - dense) - allowance[:, None, None], -1.0)
    assert np.all(excess <= 0), (
        f"{function}: {int((excess > 0).sum())} outputs past the allowance of "
        f"{ALLOWANCE_ULPS[function]} ulps of their genome's scale; largest excess "
        f"{excess.max():.3e}")


def _assert_same_program(actual: Any, desired: Any) -> None:
    """One compiled program on the same operands: bit for bit on the CPU, to GPU_TRAJECTORY_RTOL
    off it."""
    if exact_backend():
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(desired))
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(desired),
                                   rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


def _assert_same_rows(actual: Any, desired: Any) -> None:
    """One compiled program's outputs for rows or genomes that sit elsewhere in its operands
    (another place in the chunk, another slot, other genomes beside them): bit for bit on the CPU
    under single-threaded Eigen. Under XLA's default multi-threaded Eigen (cpu_multi_threaded_eigen)
    a Xeon E5-2683 v4 (AVX2, jax 0.6.1) rounded a row apart by where it sits: it put 28 of 12,780
    three-output values up to 1.2e-7 apart by their place in the chunk, so there they are compared
    as two programs are (conftest's assert_close_across_programs; for a genome's slot and the other
    genomes, which changed none of 58,834,944 outputs on 19 populations of 12 genomes, the
    comparison is a precaution); off the CPU as `_assert_same_program`."""
    if exact_backend() and cpu_multi_threaded_eigen():
        assert_close_across_programs(actual, desired)
    else:
        _assert_same_program(actual, desired)


def _take(substrate: Any, genomes: Any) -> Any:
    """The substrate of the given genomes, in that order."""
    index = np.asarray(genomes)

    def one(array):
        return None if array is None else jnp.asarray(np.asarray(array)[index])
    return substrate._replace(mask=one(substrate.mask), W1=one(substrate.W1), W2=one(substrate.W2),
                              act_indices=one(substrate.act_indices),
                              agg_indices=one(substrate.agg_indices))


def _cleared(substrate: Any, genomes: Any, positions: Optional[Any] = None) -> Any:
    """``substrate`` with the given genomes' mask entries cleared, every active one when
    ``positions`` is None, and their W1 columns and W2 rows zeroed: a substrate as built."""
    mask, W1, W2 = (np.array(a) for a in (substrate.mask, substrate.W1, substrate.W2))
    for genome in genomes:
        where = (np.flatnonzero(mask[genome]) if positions is None else np.asarray(positions))
        mask[genome, where] = False
        W1[genome][:, where] = 0.0
        W2[genome][where] = 0.0
    return substrate._replace(mask=jnp.asarray(mask), W1=jnp.asarray(W1), W2=jnp.asarray(W2))


# Compiled against dense

CASES = [("ff", name) for name in sorted(feedforward.CONFIGS)] + [
    ("nm", name) for name in sorted(feedforward.NM_CONFIGS)]


@pytest.mark.parametrize("kind, name", CASES, ids=[f"{k}-{n}" for k, n in CASES])
def test_compiled_matches_dense(kind, name):
    """Every configuration of test_substrate_api.py, under each output function: the compiled
    forward within the function's allowance of the dense one, the same finiteness."""
    substrate, inputs = _population(kind, name)
    buckets = {engine._bucket_size_for(int(c)) for c in np.asarray(substrate.mask).sum(axis=1)}
    assert len(buckets) > 1, "the population spans more than one bucket"
    for function in OUTPUTS:
        variant = _with_output(substrate, function)
        scale = _pre_activation_scale(variant, inputs)
        dense = substrate_forward(variant, inputs)
        compiled = substrate_forward(variant, inputs, compiled=True)
        _assert_within_allowance(compiled, dense, scale, function)


def test_the_allowance_sees_a_dropped_node():
    """Zeroing each genome's strongest hidden-to-output weight moves outputs far past the
    allowance, so the comparison sees a forward that loses a node."""
    substrate, inputs = _population("ff", "cppn_output_full18")
    W2 = np.array(substrate.W2)
    for genome in range(W2.shape[0]):
        W2[genome, np.argmax(np.abs(W2[genome]).sum(axis=1))] = 0.0
    dropped = substrate._replace(W2=jnp.asarray(W2))
    compiled = np.asarray(substrate_forward(substrate, inputs, compiled=True))
    dense = np.asarray(substrate_forward(dropped, inputs))
    scale = _pre_activation_scale(substrate, inputs)
    allowance = ALLOWANCE_ULPS["sigmoid"] * np.spacing(scale)
    apart = np.abs(compiled - dense).max(axis=(1, 2)) > allowance
    assert apart.sum() >= POP // 2, apart
    assert float(np.abs(compiled - dense).max()) > 1e-2


def test_a_caller_built_neuromodulation_and_both_inversions():
    """A caller-built SubstrateNeuromodulation, on a configuration with or without neuromodulation
    settings, with a vector whose fourth channel inverts and with the inversion off: the compiled
    forward within the allowance of the dense one under the same."""
    substrate, inputs = _population("nm", "default")
    nm = api.substrate_neuromodulation(substrate, neurotransmitter=[0.9, 0.1, 0.5, 0.3])
    plain = substrate._replace(config=substrate.config._replace(neuromodulation=None))
    for given in (nm, nm._replace(settings=nm.settings._replace(output_inversion=False))):
        scale = _pre_activation_scale(substrate, inputs, given)
        dense = substrate_forward(substrate, inputs, neuromodulation=given)
        for target in (substrate, plain):
            compiled = substrate_forward(target, inputs, neuromodulation=given, compiled=True)
            _assert_within_allowance(compiled, dense, scale, "sigmoid")
    inverted = np.asarray(substrate_forward(substrate, inputs, neuromodulation=nm, compiled=True))
    not_inverted = np.asarray(substrate_forward(substrate, inputs, compiled=True,
                                                neuromodulation=nm._replace(
                                                    settings=nm.settings._replace(
                                                        output_inversion=False))))
    assert float(np.abs(inverted - not_inverted).max()) > 1e-2


# Within the compiled forward

def _three_outputs(substrate: Any) -> Any:
    """``substrate`` with three outputs, the count of a caller that draws pictures: W2's column, then
    -0.5 and 0.25 times it (zero outside the mask still, so as built)."""
    W2 = np.asarray(substrate.W2)
    return substrate._replace(
        W2=jnp.asarray(np.concatenate([W2, -0.5 * W2, 0.25 * W2], axis=2)),
        config=substrate.config._replace(output_coords=((-1.0, 1.0), (0.0, 1.0), (1.0, 1.0))))


def _assert_place_in_chunk_does_not_matter(substrate: Any, inputs: np.ndarray, chunk: int) -> None:
    """One row repeated over two chunks and a part: every copy gets the same outputs; and the rows
    rolled by 37 get their own outputs back."""
    repeated = np.repeat(inputs[5:6], 2 * api._chunk_rows(chunk) + 5, axis=0)
    copies = np.asarray(substrate_forward(substrate, repeated, compiled=True, chunk_size=chunk))
    _assert_same_rows(copies, np.broadcast_to(copies[:, :1], copies.shape))
    first = np.asarray(substrate_forward(substrate, inputs, compiled=True, chunk_size=chunk))
    shift = 37
    rolled = np.asarray(substrate_forward(substrate, np.roll(inputs, shift, axis=0), compiled=True,
                                          chunk_size=chunk))
    _assert_same_rows(np.roll(rolled, -shift, axis=1), first)


def _assert_slots_and_genomes_do_not_matter(substrate: Any, inputs: np.ndarray, chunk: int,
                                            first: np.ndarray) -> None:
    """The genomes reversed (a genome moves to another slot wherever its bucket's call holds two or
    more genomes and it is not the middle one) give each genome's outputs back, and so do the
    smallest and the largest genome alone (the same bucket's program, the other slots zero)."""
    order = np.arange(POP)[::-1]
    reversed_ = np.asarray(substrate_forward(_take(substrate, order), inputs, compiled=True,
                                             chunk_size=chunk))
    _assert_same_rows(reversed_[np.argsort(order)], first)
    counts = np.asarray(substrate.mask).sum(axis=1)
    for genome in (int(np.argmin(counts)), int(np.argmax(counts))):
        alone = substrate_forward(_take(substrate, [genome]), inputs, compiled=True,
                                  chunk_size=chunk)
        _assert_same_rows(np.asarray(alone)[0], first[genome])


def test_one_program_bit_for_bit_across_chunks_slots_genomes_and_repeats():
    """With one chunk size, a row's outputs do not depend on its chunk or its place in it (one row
    repeated, the rows rolled), on its genome's slot (the genomes reversed), on the other genomes
    (each genome alone, the same bucket's program) or on a repeat; with one output and with three
    (where a chunk of 4,095 rows, without the rounding to a multiple of 8, computes rows 4,088 to
    4,091 apart from the others). Bit for bit on the CPU under single-threaded Eigen; under XLA's
    default multi-threaded Eigen, rows and genomes that sit elsewhere as two programs
    (`_assert_same_rows`), a repeat still bit for bit; on a GPU within GPU_TRAJECTORY_RTOL."""
    substrate, inputs = _population("nm", "cppn_activations")
    chunk = 100
    first = np.asarray(substrate_forward(substrate, inputs, compiled=True, chunk_size=chunk))
    _assert_same_program(substrate_forward(substrate, inputs, compiled=True, chunk_size=chunk),
                         first)
    _assert_place_in_chunk_does_not_matter(substrate, inputs, chunk)
    _assert_slots_and_genomes_do_not_matter(substrate, inputs, chunk, first)
    three = _three_outputs(substrate)
    three_first = np.asarray(substrate_forward(three, inputs, compiled=True, chunk_size=chunk))
    _assert_slots_and_genomes_do_not_matter(three, inputs, chunk, three_first)
    for chunk in (100, 4095):
        _assert_place_in_chunk_does_not_matter(three, inputs, chunk)
    # 4,095 rows round up to the default's 4,096: one program
    assert api._chunk_rows(4095) == api._chunk_rows(None) == 4096
    _assert_same_program(substrate_forward(three, inputs, compiled=True, chunk_size=4095),
                         substrate_forward(three, inputs, compiled=True))


#: Two chunk sizes at bucket 2,048 with three outputs, on the CPU. XLA:CPU computes the
#: hidden-to-output product, (rows x 2,048) @ (2,048 x 3), by a kernel that depends on the shapes,
#: so two chunk sizes round it apart even under single-threaded Eigen, outside the premise of
#: conftest's CPU_CROSS_PROGRAM_ULPS. Measured on an Apple M4 Max (jax 0.6.1, single-threaded
#: Eigen), depth-4 full-grid genomes under four test configurations, three outputs, the tests'
#: 1,284 rows: chunks of 8 to 2,560 rows against the default's 4,096, up to 8.0 ulps of the
#: genome's largest |h @ W2|, and chunks of 3,000 rows or more bit for bit (a substrate built as a
#: caller that draws pictures builds one gave 6 ulps: 1.2e-3 on pre-activations of 3,012); bit for
#: bit with one output. Eight times that is 64. The test allows the larger of that and
#: CPU_CROSS_PROGRAM_ULPS, since under XLA's default multi-threaded Eigen the threaded product may
#: round apart on its own as well, by an amount not measured on this comparison (a Xeon E5-2683 v4
#: (AVX2) passes it there).
CHUNK_SIZE_ULPS_AT_2048 = 64


def _full_grid_substrate(kind: str, name: str, depth: int, num_outputs: int, genomes: int,
                         seed: int) -> Any:
    """``genomes`` genomes keeping every position of a depth-``depth`` grid under a configuration
    of test_substrate_api.py with ``num_outputs`` outputs: weights uniform in +-max_weight and
    function indices drawn from the population's range (every position kept, so the substrate is
    zero outside its mask, as built)."""
    population, _ = _population(kind, name)
    grid = api.get_hierarchical_grid(depth)
    positions = np.asarray(grid.all_positions)
    count = positions.shape[0]
    rng = np.random.default_rng(seed)
    bound = float(population.config.max_weight)

    def indices(array: Any) -> Any:
        if array is None:
            return None
        top = int(np.asarray(array).max()) + 1
        return jnp.asarray(rng.integers(0, top, (genomes, count)).astype(np.int32))

    config = population.config._replace(
        max_depth=depth, output_coords=tuple((float(i), 1.0) for i in range(num_outputs)))
    W1 = rng.uniform(-bound, bound, (genomes, 3, count)).astype(np.float32)
    W2 = rng.uniform(-bound, bound, (genomes, count, num_outputs)).astype(np.float32)
    return api.Substrate(
        positions=jnp.asarray(positions), levels=api._levels(grid),
        mask=jnp.ones((genomes, count), dtype=bool), W1=jnp.asarray(W1), W2=jnp.asarray(W2),
        act_indices=indices(population.act_indices), agg_indices=indices(population.agg_indices),
        config=config)


def test_two_chunk_sizes_are_two_programs():
    """Two chunk sizes compile two programs, which agree within conftest's CPU_CROSS_PROGRAM_ULPS on
    the CPU (a Xeon E5-2683 v4 (AVX2) under XLA's default threading rounds two programs apart); at
    bucket 2,048 with three outputs (a depth-4 full-grid genome) they round the hidden-to-output
    product apart under any threading, within the larger of CHUNK_SIZE_ULPS_AT_2048 and
    CPU_CROSS_PROGRAM_ULPS of the largest |h @ W2|."""
    substrate, inputs = _population("ff", "combined_true")
    small = substrate_forward(substrate, inputs, compiled=True, chunk_size=64)
    default = substrate_forward(substrate, inputs, compiled=True)
    if exact_backend():
        assert_close_across_programs(small, default)
    else:
        np.testing.assert_allclose(np.asarray(small), np.asarray(default),
                                   rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)
    # The identity output, so the comparison reads the sum itself
    full = _with_output(_full_grid_substrate("ff", "combined_true", 4, 3, 2, seed=11), "identity")
    assert engine._bucket_size_for(int(full.mask.shape[1])) == 2048
    chunked = substrate_forward(full, inputs, compiled=True, chunk_size=1000)
    default = substrate_forward(full, inputs, compiled=True)
    if exact_backend():
        assert_close_across_programs(chunked, default,
                                     ulps=max(CHUNK_SIZE_ULPS_AT_2048, CPU_CROSS_PROGRAM_ULPS))
    else:
        # On a GPU an elementwise rtol of 1e-4 fails: an NVIDIA T4 put 15 of 7,704 outputs up to
        # 1.45e-4 apart and an NVIDIA L4 14 up to 8.4e-5 (jax 0.6.1,
        # JAX_DEFAULT_MATMUL_PRECISION=highest), at small sums whose rounding scales with the
        # largest terms, not with the sum. So the allowance scales with the largest output, 158 on
        # these rows (its value on the CPU): 128 ulps of it, 2.0e-3, 13 times the T4's largest
        # difference.
        default = np.asarray(default)
        scale = float(np.max(np.abs(default[np.isfinite(default)])))
        np.testing.assert_allclose(np.asarray(chunked), default, rtol=0,
                                   atol=128 * float(np.spacing(np.float32(scale))))


@pytest.mark.parametrize("rows", [1, 63, 65])
def test_row_counts_around_the_chunk_size(rows):
    """1, chunk_size - 1 and chunk_size + 1 rows: the shape of the dense forward, and each row's
    outputs bit for bit those of a longer call with the same chunk size."""
    substrate, inputs = _population("ff", "cppn_output_full18_uniform")
    chunk = 64
    longer = np.asarray(substrate_forward(substrate, inputs[:3 * chunk], compiled=True,
                                          chunk_size=chunk))
    outputs = substrate_forward(substrate, inputs[:rows], compiled=True, chunk_size=chunk)
    assert outputs.shape == (POP, rows, 1) == substrate_forward(substrate, inputs[:rows]).shape
    _assert_same_program(outputs, longer[:, :rows])


def test_no_rows():
    substrate, _ = _population("ff", "plain")
    outputs = substrate_forward(substrate, np.zeros((0, 3), dtype=np.float32), compiled=True)
    assert outputs.shape == (POP, 0, 1)


# Empty genomes, silent positions, the mask rule

#: A neurotransmitter vector whose fourth channel inverts the outputs (the default, 1.0, does not)
INVERTING = (0.5, 0.5, 0.5, 0.3)


@pytest.mark.parametrize("kind, name", [("ff", "cppn_output_full18_uniform"),
                                        ("ff", "combined_true"), ("nm", "output_relu"),
                                        ("nm", "rule_steep_gate")])
def test_an_empty_genome_gives_the_output_activation_of_zero(kind, name):
    """A genome whose mask is cleared (its weights zero, as built) gives what the dense forward
    gives: the output activation of zero at every input, then the inversion under neuromodulation
    (a fourth channel of 0.3: relu's 0 becomes 0.7, while the sigmoid's 0.5 is its own inversion);
    the other genomes keep their outputs bit for bit."""
    substrate, inputs = _population(kind, name)
    empty = [0, 7]
    edited = _cleared(substrate, empty)
    nm = nm_before = None
    if substrate.config.neuromodulation is not None:
        nm = api.substrate_neuromodulation(edited, neurotransmitter=INVERTING)
        nm_before = api.substrate_neuromodulation(substrate, neurotransmitter=INVERTING)
    compiled = np.asarray(substrate_forward(edited, inputs, neuromodulation=nm, compiled=True))
    dense = np.asarray(substrate_forward(edited, inputs, neuromodulation=nm))
    zero = float(engine.ACTIVATION_FUNCTIONS[substrate.config.output_activation](jnp.float32(0.0)))
    if nm is not None:
        zero = INVERTING[3] * zero + (1.0 - INVERTING[3]) * (1.0 - zero)
    one_ulp = float(np.spacing(np.float32(1.0)))
    for genome in empty:
        np.testing.assert_allclose(compiled[genome], zero, rtol=0, atol=one_ulp)
        np.testing.assert_allclose(dense[genome], zero, rtol=0, atol=one_ulp)
    before = np.asarray(substrate_forward(substrate, inputs, neuromodulation=nm_before,
                                          compiled=True))
    others = np.setdiff1d(np.arange(POP), empty)
    _assert_same_program(compiled[others], before[others])


@pytest.mark.parametrize("kind, name", [("ff", "cppn_output_full18"), ("nm", "output_relu")])
def test_a_substrate_cut_to_one_silent_position(kind, name):
    """A caller that compacts the substrate to its genomes' positions keeps one silent position when
    no genome has any (outside every mask, zero weights): every genome gives the output activation
    of zero, as in the dense forward, then under neuromodulation the inversion (receptors and gains
    restricted to the kept position; a fourth channel of 0.3 turns relu's 0 into 0.7); and a silent
    position inside a mask (zero weights) adds nothing."""
    substrate, inputs = _population(kind, name)
    empty = _cleared(substrate, range(POP))
    keep = np.array([0])
    silent = empty._replace(
        positions=empty.positions[keep], levels=np.asarray(empty.levels)[keep],
        mask=empty.mask[:, keep], W1=empty.W1[:, :, keep], W2=empty.W2[:, keep],
        act_indices=None if empty.act_indices is None else empty.act_indices[:, keep],
        agg_indices=None if empty.agg_indices is None else empty.agg_indices[:, keep])
    zero = float(engine.ACTIVATION_FUNCTIONS[substrate.config.output_activation](jnp.float32(0.0)))
    nm = None
    if substrate.config.neuromodulation is not None:
        built = api.substrate_neuromodulation(empty, neurotransmitter=INVERTING)
        nm = built._replace(receptors=built.receptors[:, keep],
                            base_gains=built.base_gains[:, keep])
        zero = INVERTING[3] * zero + (1.0 - INVERTING[3]) * (1.0 - zero)
    compiled = np.asarray(substrate_forward(silent, inputs, neuromodulation=nm, compiled=True))
    np.testing.assert_array_equal(compiled, np.asarray(substrate_forward(silent, inputs,
                                                                         neuromodulation=nm)))
    np.testing.assert_allclose(compiled, zero, rtol=0, atol=float(np.spacing(np.float32(1.0))))
    inside = silent._replace(mask=jnp.ones_like(silent.mask))
    np.testing.assert_array_equal(
        np.asarray(substrate_forward(inside, inputs, neuromodulation=nm, compiled=True)), compiled)


def test_the_mask_rule_on_an_edited_substrate():
    """A position the mask clears while its weights stay is dropped by the compiled forward, which
    then computes the substrate with those weights zeroed, and still summed by the dense forward."""
    substrate, inputs = _population("ff", "cppn_output_full18_uniform")
    mask = np.array(substrate.mask)
    knocked = {genome: int(np.flatnonzero(mask[genome])[-1]) for genome in range(POP)}
    for genome, position in knocked.items():
        mask[genome, position] = False
    edited = substrate._replace(mask=jnp.asarray(mask))
    zeroed = edited
    for genome, position in knocked.items():
        zeroed = _cleared(zeroed, [genome], [position])
    compiled = np.asarray(substrate_forward(edited, inputs, compiled=True))
    _assert_same_program(compiled, substrate_forward(zeroed, inputs, compiled=True))
    _assert_within_allowance(compiled, substrate_forward(zeroed, inputs),
                             _pre_activation_scale(zeroed, inputs), "sigmoid")
    dense = np.asarray(substrate_forward(edited, inputs))
    np.testing.assert_array_equal(dense, np.asarray(substrate_forward(substrate, inputs)))
    assert float(np.abs(compiled - dense).max()) > 1e-2


def test_nan_and_inf_are_kept():
    """A node whose drive overflows (relu of inf) gives NaN through a zero output weight and inf
    through a nonzero one, in both forwards at the same outputs; the other genomes stay finite."""
    algo, problem, state = feedforward._setup(create_config_with_dynamic_functions(
        "global_relu", max_depth=DEPTH, population_size=POP))
    substrate = _with_output(algo.population_substrates(state), "identity")
    mask, W1, W2 = (np.array(a) for a in (substrate.mask, substrate.W1, substrate.W2))
    nan_genome, inf_genome = 3, 11
    for genome, weight in ((nan_genome, 0.0), (inf_genome, 1.0)):
        position = int(np.flatnonzero(mask[genome])[0])
        W1[genome][:, position] = 3.0e38
        W2[genome][position] = weight
    edited = substrate._replace(W1=jnp.asarray(W1), W2=jnp.asarray(W2))
    # Positive rows: every product with the large weight is positive, so the drive is inf in any
    # order of summation
    inputs = np.random.default_rng(1).uniform(0.5, 1.0, (50, 3)).astype(np.float32)
    dense = np.asarray(substrate_forward(edited, inputs))
    compiled = np.asarray(substrate_forward(edited, inputs, compiled=True))
    assert np.isnan(dense[nan_genome]).all() and np.isposinf(dense[inf_genome]).all()
    others = np.setdiff1d(np.arange(POP), [nan_genome, inf_genome])
    assert np.isfinite(dense[others]).all()
    # The scale over the finite pre-activations (the identity output's), 1 for a genome with none
    finite_pre = np.where(np.isfinite(dense), np.abs(dense), 0.0)
    scale = np.maximum(1.0, finite_pre.max(axis=(1, 2))).astype(np.float32)
    _assert_within_allowance(compiled, dense, scale, "identity")


#: float32's largest x whose exp is finite (0x42b17217, 88.722832) and the next float (0x42b17218,
#: 88.722839, exp inf): ln(FLT_MAX) = 88.7228391 lies between them, so softplus, log1p(exp(x)),
#: overflows from the second on
_LAST_FINITE = np.uint32(0x42B17217).view(np.float32)
_FIRST_OVERFLOWING = np.uint32(0x42B17218).view(np.float32)
#: (gain, drive) pairs whose paper-rule pre-activation gain * drive + gain (the bias equal to the
#: gain) lands on the two sides of that threshold: rounded twice (the product, then the sum) and
#: rounded once (a fused multiply-add). In the first the twice-rounded value overflows, in the
#: second the once-rounded one (found by an exact search; the test checks it).
_STRADDLES = ((np.float32(3.3), np.uint32(0x41CF15EE).view(np.float32)),
              (np.float32(3.7), np.uint32(0x41B7D54A).view(np.float32)))


def _rounded_to_float32(value: Fraction) -> np.float32:
    """The float32 nearest to an exact value, ties to even."""
    near = np.float32(float(value))
    best = None
    for candidate in (np.nextafter(near, np.float32(-np.inf)), near,
                      np.nextafter(near, np.float32(np.inf))):
        distance = abs(Fraction(float(candidate)) - value)
        even = int(candidate.view(np.uint32)) % 2 == 0
        if best is None or distance < best[0] or (distance == best[0] and even):
            best = (distance, candidate)
    return best[1]


def test_a_value_at_the_overflow_threshold_can_overflow_in_one_forward_only():
    """The exception to equal finiteness on a substrate as built, at a hidden node, pinned. The
    compiled program rounds the paper rule's gains * pre + bias once (XLA:CPU fuses the multiply and
    the add into a multiply-add; it does the same for exact and global sum and mean aggregation's
    multiply and sum, and for the gain bg + s m when s is not a power of two), where the dense
    forward, op by op, rounds the product and the sum apart. A softplus node whose pre-activation
    the two roundings put on either side of ln(FLT_MAX) then overflows in one forward only. Built
    with exact operands (an input of 1 times the drive's weight, receptor channel 0 under the vector
    (1, 0, 0, 1), base gain 0 and strength 4, so gain = bias = 4 * receptor), so the dense forward's
    pre-activation is the twice-rounded value on every backend: genome 0 overflows in the dense
    forward only and genome 1 in the compiled forward only. Off the CPU the compiled program may
    round either way. The output's route is test_an_output_at_the_overflow_threshold_...: the
    hidden-to-output sum."""
    for index, (gain, drive) in enumerate(_STRADDLES):
        exact = Fraction(float(gain)) * Fraction(float(drive)) + Fraction(float(gain))
        once = _rounded_to_float32(exact)
        twice = _rounded_to_float32(
            Fraction(float(_rounded_to_float32(Fraction(float(gain)) * Fraction(float(drive)))))
            + Fraction(float(gain)))
        expected = ((_FIRST_OVERFLOWING, _LAST_FINITE) if index == 0
                    else (_LAST_FINITE, _FIRST_OVERFLOWING))
        assert (twice, once) == expected, (index, twice, once)
    softplus = engine.ACTIVATION_FUNCTIONS["softplus"]
    assert np.isfinite(softplus(jnp.float32(_LAST_FINITE)))
    assert np.isposinf(softplus(jnp.float32(_FIRST_OVERFLOWING)))

    population, _ = _population("nm", "default")
    genomes = len(_STRADDLES)
    W1 = np.zeros((genomes, 3, 1), dtype=np.float32)
    W1[:, 0, 0] = [drive for _, drive in _STRADDLES]
    keep = np.array([0])
    substrate = population._replace(
        positions=population.positions[keep], levels=np.asarray(population.levels)[keep],
        mask=jnp.ones((genomes, 1), dtype=bool), W1=jnp.asarray(W1),
        W2=jnp.ones((genomes, 1, 1), dtype=jnp.float32), act_indices=None, agg_indices=None,
        config=population.config._replace(hidden_activation="softplus",
                                          output_activation="identity", neuromodulation=None))
    settings = population.config.neuromodulation._replace(
        node_rule="paper", modulation_mode="full", modulation_strength=4.0,
        output_inversion=False)
    receptors = np.zeros((genomes, 1, int(settings.num_nt_types)), dtype=np.float32)
    receptors[:, 0, 0] = [gain / np.float32(4.0) for gain, _ in _STRADDLES]
    nm = api.SubstrateNeuromodulation(jnp.asarray(receptors),
                                      jnp.zeros((genomes, 1), dtype=jnp.float32),
                                      jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
                                      settings)
    row = np.array([[1.0, 0.5, -0.25]], dtype=np.float32)
    dense = np.asarray(substrate_forward(substrate, row, neuromodulation=nm))[:, 0, 0]
    compiled = np.asarray(substrate_forward(substrate, row, neuromodulation=nm,
                                            compiled=True))[:, 0, 0]
    assert np.isposinf(dense[0]) and np.isfinite(dense[1]), dense
    # The finite outputs, softplus(88.722832) * sigmoid(receptor)
    for value, (gain, _) in ((dense[1], _STRADDLES[1]), (compiled[0], _STRADDLES[0])):
        if np.isfinite(value):
            gate = float(jax.nn.sigmoid(jnp.float32(gain / np.float32(4.0))))
            np.testing.assert_allclose(value, float(_LAST_FINITE) * gate, rtol=1e-5)
    if exact_backend():
        assert np.isfinite(compiled[0]) and np.isposinf(compiled[1]), (
            f"compiled {compiled}: the compiled program no longer rounds gains * pre + bias once "
            "on the CPU, so the exception documented in docs/architecture.md (the substrate on "
            "its own) may be gone; update it and this test")
    else:
        assert all(np.isposinf(value) or np.isfinite(value) for value in compiled), compiled


#: The raster the softplus-output search starts from, and how many lines it scans
_THRESHOLD_RASTER = 64
_THRESHOLD_LINES = 3


def _overflows(values: Any) -> np.ndarray:
    """Where the engine's softplus, applied eagerly, overflows: z past ln(FLT_MAX)."""
    return ~np.isfinite(np.asarray(engine.ACTIVATION_FUNCTIONS["softplus"](jnp.asarray(values))))


def test_an_output_at_the_overflow_threshold_can_overflow_in_one_forward_only():
    """The exception's second route: the output. The dense forward sums h @ W2 over the grid's
    positions and the compiled one over the genome's own, in another order, so wherever the two
    sums lie on either side of ln(FLT_MAX) a softplus output is finite in one forward and inf in the
    other, on a substrate exactly as built, without neuromodulation or a fused step, with inputs in
    [-1, 1]. Searched on combined_approx (one genome's sum reaches the threshold on the raster): on
    a few raster lines where the dense sum crosses it, the 81 float32 inputs around the crossing.
    Each forward's softplus output overflows exactly where its own sum (the identity output) does,
    so the two differ in finiteness exactly where their sums straddle the threshold. On an Apple
    M4 Max some rows straddle (31 of the 243 scanned, the sums one ulp apart); elsewhere any count
    is accepted."""
    substrate, _ = _population("ff", "combined_approx")
    identity, softplus = _with_output(substrate, "identity"), _with_output(substrate, "softplus")
    side = _THRESHOLD_RASTER
    axis = (2.0 * np.arange(side, dtype=np.float32) + 1.0) / side - 1.0
    sums = np.asarray(substrate_forward(identity, _raster(side)))[:, :, 0].reshape(-1, side, side)
    over = _overflows(sums)
    crossings = [(genome, line, column) for genome in range(sums.shape[0])
                 for line, column in np.argwhere(~over[genome][:, :-1] & over[genome][:, 1:])]
    assert crossings, "no genome's dense sum reaches the threshold on the raster any more"
    straddles = 0
    for genome, line, column in crossings[:_THRESHOLD_LINES]:
        alone_identity, alone_softplus = (_take(s, [genome]) for s in (identity, softplus))
        y = axis[line]
        low, high = axis[column], axis[column + 1]
        # Bisect x in float32 to the dense sum's crossing on this line
        while True:
            middle = np.float32((np.float64(low) + np.float64(high)) / 2.0)
            if middle in (low, high):
                break
            at = np.asarray(substrate_forward(alone_identity, np.array([[middle, y, 1.0]],
                                                                       dtype=np.float32)))
            low, high = (low, middle) if _overflows(at[0, 0, 0]) else (middle, high)
        xs = [low]
        for _ in range(40):
            xs.insert(0, np.nextafter(xs[0], np.float32(-2.0)))
            xs.append(np.nextafter(xs[-1], np.float32(2.0)))
        rows = np.array([[x, y, 1.0] for x in xs], dtype=np.float32)
        dense_sum = np.asarray(substrate_forward(alone_identity, rows))[0, :, 0]
        compiled_sum = np.asarray(substrate_forward(alone_identity, rows, compiled=True))[0, :, 0]
        dense = np.asarray(substrate_forward(alone_softplus, rows))[0, :, 0]
        compiled = np.asarray(substrate_forward(alone_softplus, rows, compiled=True))[0, :, 0]
        np.testing.assert_array_equal(~np.isfinite(dense), _overflows(dense_sum))
        np.testing.assert_array_equal(~np.isfinite(compiled), _overflows(compiled_sum))
        apart = np.isfinite(dense) != np.isfinite(compiled)
        np.testing.assert_array_equal(apart, _overflows(dense_sum) != _overflows(compiled_sum))
        straddles += int(apart.sum())
    if exact_backend() and platform.machine() == "arm64":
        assert straddles > 0, (
            "no row's two sums straddle ln(FLT_MAX) any more on this machine: the exception "
            "documented in docs/architecture.md (the substrate on its own) still holds "
            "wherever they do; move this search to a genome and lines where they do")


# Compiles and the function cache

_COMPILES: Dict[str, Any] = {"active": False, "count": 0, "registered": False}


def _watch_compiles() -> None:
    """Count backend compilations while _COMPILES['active'] (the listener of
    test_compaction_bucketing.py)."""
    if not _COMPILES["registered"]:
        def listener(event, duration, **kwargs):
            if _COMPILES["active"] and event == "/jax/core/compile/backend_compile_duration":
                _COMPILES["count"] += 1
        jax.monitoring.register_event_duration_secs_listener(listener)
        _COMPILES["registered"] = True


def _compiles(call) -> Tuple[Any, int]:
    _COMPILES["count"] = 0
    _COMPILES["active"] = True
    try:
        result = np.asarray(call())
    finally:
        _COMPILES["active"] = False
    return result, _COMPILES["count"]


def _keep_first(substrate: Any, count: int) -> Any:
    """Every genome keeps the first ``count`` positions of the grid, its weights there as built
    elsewhere zero."""
    mask = np.zeros(np.asarray(substrate.mask).shape, dtype=bool)
    mask[:, :count] = True
    W1, W2 = np.array(substrate.W1), np.array(substrate.W2)
    W1[:, :, count:] = 0.0
    W2[:, count:] = 0.0
    return substrate._replace(mask=jnp.asarray(mask), W1=jnp.asarray(W1), W2=jnp.asarray(W2))


def test_the_vector_and_the_strength_compile_nothing_and_a_bucket_compiles_once():
    """The neurotransmitter vector and the strength are operands: other values compile nothing and
    change the outputs. A bucket size compiles once, one program for its slots: a new bucket
    compiles one program, and another number of genomes in it (one alone, in a bucket of many slots
    and in one of a single slot) compiles nothing."""
    _watch_compiles()
    substrate, inputs = _population("nm", "cppn_activations")
    nm = api.substrate_neuromodulation(substrate)
    first = np.asarray(substrate_forward(substrate, inputs, neuromodulation=nm, compiled=True))
    counts = np.asarray(substrate.mask).sum(axis=1)
    smallest = int(np.argmin(counts))
    assert api._compiled_slots(engine._bucket_size_for(int(counts[smallest]))) > 1
    # Sliced before the counted call: an eager slice compiles a program of its own
    smallest_nm = nm._replace(receptors=nm.receptors[smallest:smallest + 1],
                              base_gains=nm.base_gains[smallest:smallest + 1])
    smallest_alone = _take(substrate, [smallest])
    alone, count = _compiles(lambda: substrate_forward(smallest_alone, inputs,
                                                       neuromodulation=smallest_nm, compiled=True))
    assert count == 0, f"{count} compiled for one genome in a bucket of several slots"
    _assert_same_program(alone[0], first[smallest])
    other = nm._replace(neurotransmitter=jnp.asarray([0.2, 0.8, 0.3, 0.6], dtype=jnp.float32),
                        settings=nm.settings._replace(modulation_strength=3.5))
    second, count = _compiles(lambda: substrate_forward(substrate, inputs, neuromodulation=other,
                                                        compiled=True))
    assert count == 0, f"{count} compiled for another vector and strength"
    assert float(np.abs(second - first).max()) > 1e-3
    # 200 positions: bucket 256, which no genome of these populations reaches
    wide = _keep_first(substrate, 200)
    wide_nm = api.substrate_neuromodulation(wide)
    _, count = _compiles(lambda: substrate_forward(wide, inputs, neuromodulation=wide_nm,
                                                   compiled=True))
    assert count == 1, f"{count} compiled for a new bucket"
    alone = _take(wide, [4])
    alone_nm = wide_nm._replace(receptors=wide_nm.receptors[4:5],
                                base_gains=wide_nm.base_gains[4:5])
    _, count = _compiles(lambda: substrate_forward(alone, inputs, neuromodulation=alone_nm,
                                                   compiled=True))
    assert count == 0, f"{count} compiled for one genome in a bucket already compiled"


def test_another_output_count_compiles_each_bucket_once_more():
    """Another number of outputs (a caller may serve one and three) is another operand shape, not
    another key: the same cache entry, whose jitted function compiles each bucket's program once
    more, so a caller warms its buckets once per output count; a repeat compiles nothing."""
    _watch_compiles()
    substrate, inputs = _population("ff", "plain")
    sizes = {engine._bucket_size_for(int(c)) for c in np.asarray(substrate.mask).sum(axis=1)}
    substrate_forward(substrate, inputs, compiled=True)
    three = _three_outputs(substrate)
    flags = (substrate.act_indices is not None, substrate.agg_indices is not None)
    assert (api._feedforward_forward_fn(three.config, *flags)
            is api._feedforward_forward_fn(substrate.config, *flags))
    _, count = _compiles(lambda: substrate_forward(three, inputs, compiled=True))
    assert count == len(sizes), f"{count} compiled for {len(sizes)} buckets at three outputs"
    _, count = _compiles(lambda: substrate_forward(three, inputs, compiled=True))
    assert count == 0, f"{count} compiled on a repeat"


def test_the_function_cache_key_and_bound():
    """Each setting the forward reads is in the key where it reads it; coordinates, thresholds, the
    receptor derivation and the strength are not, nor a setting the combination does not read
    (test_a_setting_the_forward_does_not_read_compiles_nothing), nor the bucket size and the rows,
    which the entry's jitted function takes from its operands' shapes; and the cache holds 64."""
    fn = api._feedforward_forward_fn
    substrate, _ = _population("nm", "default")
    config = substrate.config
    settings = config.neuromodulation
    plain = fn(config, False, False)
    assert fn(config, False, False, None) is plain
    elsewhere = config._replace(input_coords=((0.0, -1.0),) * 3, variance_threshold=0.05,
                                weight_threshold=0.2, max_weight=5.0, locality_radius=0.3)
    assert fn(elsewhere, False, False) is plain
    # Read without per-node functions
    for changed in (dict(hidden_activation="sin"), dict(output_activation="tanh"),
                    dict(aggregation_mode="global")):
        assert fn(config._replace(**changed), False, False) is not plain, changed
    # Read with per-node activations: the palette, and without one the number of activations
    paletted = config._replace(palette=(0, 1, 2, 3))
    assert fn(paletted._replace(palette=(0, 1, 2)), True, False) is not fn(paletted, True, False)
    unpaletted = config._replace(palette=None)
    assert (fn(unpaletted._replace(num_activations=6), True, False)
            is not fn(unpaletted, True, False))
    # Read with per-node aggregation: exact or approximated, and under exact aggregation the
    # aggregations evaluated; approximated aggregation reads the indices alone
    exact = config._replace(true_aggregation=True)
    per_node = fn(exact, False, True)
    for changed in (dict(true_aggregation=False), dict(num_aggregations=3),
                    dict(aggregation_palette=(1, 2))):
        assert fn(exact._replace(**changed), False, True) is not per_node, changed
    approximated = config._replace(true_aggregation=False)
    assert (fn(approximated._replace(num_aggregations=3, aggregation_palette=(1, 2)), False, True)
            is fn(approximated, False, True))
    # Read under the 'global' mode: which function
    global_max = config._replace(aggregation_mode="global", global_aggregation="max")
    assert (fn(global_max._replace(global_aggregation="min"), False, False)
            is not fn(global_max, False, False))
    for args in ((True, False), (False, True)):
        assert fn(config, *args) is not plain, args
    modulated = fn(config, False, False, settings)
    assert modulated is not plain
    assert fn(config, False, False, settings._replace(
        modulation_strength=5.0, receptor_derivation="abs")) is modulated
    for change in (dict(node_rule="steep_gate"), dict(modulation_mode="gating_only"),
                   dict(num_nt_types=3), dict(output_inversion=False)):
        assert fn(config, False, False, settings._replace(**change)) is not modulated
    cache = api._compiled_feedforward_forward
    assert cache.cache_info().maxsize == 64
    # Building an entry traces nothing until it is called, so filling the cache is cheap: 69
    # palettes are 69 combinations of the settings
    for entry in range(69):
        fn(config._replace(palette=(entry,)), True, False)
    assert cache.cache_info().currsize == 64


#: A workload past the cache's bound: the nine bucket sizes of depth 4 (8 to 2,048 positions) under
#: eight combinations of the settings the compiled forward reads (four output functions, two hidden
#: activations), 72 (settings, bucket) combinations, more than the cache's 64 entries
CYCLE_OUTPUTS = ("sigmoid", "tanh", "relu", "identity")
CYCLE_HIDDEN = ("tanh", "sin")
CYCLE_COUNTS = (5, 12, 24, 48, 100, 200, 400, 800, 1300)


def test_a_second_pass_over_more_than_64_settings_and_buckets_traces_nothing(monkeypatch):
    """An entry of the cache is one combination of the settings, whose jitted function keeps a
    program per bucket, so bucket sizes never evict one another: over 72 (settings, bucket)
    combinations (eight of the settings at nine bucket sizes, more than the cache's 64 entries) the
    second pass traces nothing (the per-genome function counted as it is traced), and its outputs
    are the first pass's bit for bit (one program per entry and shapes; on a GPU within
    GPU_TRAJECTORY_RTOL). A key that held the bucket size would evict each entry before its next
    use and trace and compile every bucket on every pass."""
    grid = _full_grid_substrate("ff", "plain", 4, 1, len(CYCLE_COUNTS), seed=5)
    positions = int(grid.mask.shape[1])
    assert positions > 1024, "depth 4 reaches the bucket of 2,048"
    mask = np.zeros((len(CYCLE_COUNTS), positions), dtype=bool)
    for genome, count in enumerate(CYCLE_COUNTS):
        mask[genome, :count] = True
    W1 = np.where(mask[:, None, :], np.asarray(grid.W1), 0.0).astype(np.float32)
    W2 = np.where(mask[:, :, None], np.asarray(grid.W2), 0.0).astype(np.float32)
    grid = grid._replace(mask=jnp.asarray(mask), W1=jnp.asarray(W1), W2=jnp.asarray(W2))
    sizes = {engine._bucket_size_for(count) for count in CYCLE_COUNTS}
    assert len(sizes) == len(CYCLE_COUNTS), sizes
    combinations = [(output, hidden) for output in CYCLE_OUTPUTS for hidden in CYCLE_HIDDEN]
    assert len(combinations) * len(sizes) > api._compiled_feedforward_forward.cache_info().maxsize
    inputs = _raster(8)
    traced = {"count": 0}
    original = api.dense_substrate_forward

    def counted(*args, **kwargs):
        traced["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "dense_substrate_forward", counted)
    api._compiled_feedforward_forward.cache_clear()
    passes = []
    for _ in range(2):
        traced["count"] = 0
        outputs = [np.asarray(substrate_forward(grid._replace(config=grid.config._replace(
            output_activation=output, hidden_activation=hidden)), inputs, compiled=True))
            for output, hidden in combinations]
        passes.append((traced["count"], outputs))
    first, second = passes
    assert first[0] >= len(combinations) * len(sizes), f"{first[0]} traced on the first pass"
    assert second[0] == 0, f"{second[0]} traced again on the second pass"
    for before, after in zip(first[1], second[1]):
        _assert_same_program(after, before)


#: A setting the forward does not read, by configuration, and another value of it: the hidden
#: activation with per-node activations (a caller may still write it with them on), the palette
#: without them and the number of activations with a palette, the global function outside the
#: 'global' mode, the aggregation settings without per-node aggregation or under a global
#: function, the aggregations evaluated under approximated aggregation (its correction factors
#: read the indices alone), and under neuromodulation the inversion below four channels and the
#: modulation mode under 'anti'.
UNREAD = [
    (("ff", "cppn_output_full18"), dict(hidden_activation="sin"), None),
    (("ff", "cppn_output_full18"), dict(num_activations=6), None),
    (("ff", "plain"), dict(palette=(0, 1, 2), num_activations=6, global_aggregation="min",
                           true_aggregation=False, num_aggregations=3,
                           aggregation_palette=(1, 2)), None),
    (("ff", "aggregation_global_max"), dict(true_aggregation=False, num_aggregations=3), None),
    (("ff", "combined_approx"), dict(num_aggregations=3, aggregation_palette=(1, 2)), None),
    (("nm", "nt_2"), None, dict(output_inversion=False)),
    (("nm", "rule_anti"), None, dict(modulation_mode="gating_only")),
]


@pytest.mark.parametrize("population, config_change, settings_change", UNREAD,
                         ids=[f"{p[0]}-{p[1]}-{i}" for i, (p, _, _) in enumerate(UNREAD)])
def test_a_setting_the_forward_does_not_read_compiles_nothing(population, config_change,
                                                              settings_change):
    """Changing a setting the combination does not read leaves the dense forward's outputs as they
    were, bit for bit (so the setting is unread), and the compiled forward's too, with nothing
    compiled (so the key leaves it out)."""
    _watch_compiles()
    substrate, inputs = _population(*population)
    config = substrate.config
    if config_change is not None:
        config = config._replace(**config_change)
    if settings_change is not None:
        config = config._replace(neuromodulation=config.neuromodulation._replace(**settings_change))
    changed = substrate._replace(config=config)
    np.testing.assert_array_equal(np.asarray(substrate_forward(changed, inputs)),
                                  np.asarray(substrate_forward(substrate, inputs)))
    before = np.asarray(substrate_forward(substrate, inputs, compiled=True))
    after, count = _compiles(lambda: substrate_forward(changed, inputs, compiled=True))
    assert count == 0, f"{count} compiled for a setting the forward does not read"
    np.testing.assert_array_equal(after, before)


# Refusals and checks

def test_a_global_softmax_hidden_activation_is_refused():
    """'softmax' normalizes over every position of the grid, which this forward's genome slots do
    not add (the recurrent forward does, full_grid_softmax): refused as a global hidden activation,
    and evaluated where nodes have their own functions (the hidden activation is then unread)."""
    substrate, inputs = _population("ff", "plain")
    softmax = substrate._replace(config=substrate.config._replace(hidden_activation="softmax"))
    substrate_forward(softmax, inputs)
    with pytest.raises(ValueError, match="'softmax' hidden activation normalizes"):
        substrate_forward(softmax, inputs, compiled=True)
    nm_substrate, _ = _population("nm", "default")
    nm_softmax = nm_substrate._replace(
        config=nm_substrate.config._replace(hidden_activation="softmax"))
    with pytest.raises(ValueError, match="'softmax' hidden activation normalizes"):
        substrate_forward(nm_softmax, inputs, compiled=True)
    per_node, _ = _population("ff", "cppn_output")
    per_node = per_node._replace(config=per_node.config._replace(hidden_activation="softmax"))
    _assert_same_program(substrate_forward(per_node, inputs, compiled=True),
                         substrate_forward(_population("ff", "cppn_output")[0], inputs,
                                           compiled=True))


def test_chunk_size_and_inputs_are_checked():
    substrate, inputs = _population("ff", "plain")
    with pytest.raises(ValueError, match="return_states needs a recurrent substrate"):
        substrate_forward(substrate, inputs, return_states=True, compiled=True)
    with pytest.raises(ValueError, match="chunk_size applies to the compiled feed-forward forward"):
        substrate_forward(substrate, inputs, chunk_size=64)
    for wrong in (0, -1, True, 2.5, "64"):
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            substrate_forward(substrate, inputs, compiled=True, chunk_size=wrong)
    np.testing.assert_array_equal(
        np.asarray(substrate_forward(substrate, inputs, compiled=True, chunk_size=np.int64(64))),
        np.asarray(substrate_forward(substrate, inputs, compiled=True, chunk_size=64)))
    for wrong in (inputs[0], inputs[:, :2], inputs[None]):
        with pytest.raises(ValueError, match=r"inputs must be \(n_samples, 3\)"):
            substrate_forward(substrate, wrong, compiled=True)


def test_a_recurrent_substrate_ignores_compiled_and_refuses_chunk_size():
    """compiled=True changes nothing on a recurrent substrate (its forward is compiled per genome
    already); chunk_size is refused there."""
    config = create_config_with_recurrence("hidden_only", max_depth=DEPTH, population_size=POP,
                                           iteration_level=1, hh_cache_enabled=False)
    algo, problem, state = feedforward._setup(config)
    transformed = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    population = algo.population_substrates(state)
    recurrent = build_recurrent_substrate(population, query_recurrent_substrate(
        algo._jitted_cppn_forward, state, transformed, population))
    inputs = np.asarray(problem.inputs, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(substrate_forward(recurrent, inputs, compiled=True)),
                                  np.asarray(substrate_forward(recurrent, inputs)))
    for compiled in (False, True):
        with pytest.raises(ValueError,
                           match="a recurrent substrate's forward takes its inputs whole"):
            substrate_forward(recurrent, inputs, compiled=compiled, chunk_size=64)


def test_the_neuromodulation_checks_are_the_dense_forwards():
    """The compiled forward refuses what the dense neuromodulated forward refuses, with its
    messages, and receptors without the channels the node rule reads, which the dense forward fails
    on inside the computation."""
    substrate, inputs = _population("nm", "default")
    nm = api.substrate_neuromodulation(substrate)
    short = nm._replace(neurotransmitter=jnp.asarray([0.5, 0.5, 0.2]))
    for compiled in (False, True):
        with pytest.raises(ValueError, match="reads 4 entries"):
            substrate_forward(substrate, inputs, neuromodulation=short, compiled=compiled)
        with pytest.raises(ValueError, match="must cover the substrate"):
            substrate_forward(substrate, inputs, compiled=compiled,
                              neuromodulation=nm._replace(base_gains=nm.base_gains[:1]))
    with pytest.raises(ValueError, match="reads 3 receptor channels"):
        substrate_forward(substrate, inputs, compiled=True,
                          neuromodulation=nm._replace(receptors=nm.receptors[..., :2]))
    with pytest.raises(TypeError):
        substrate_forward(substrate, inputs,
                          neuromodulation=nm._replace(receptors=nm.receptors[..., :2]))


def test_a_strength_or_vector_not_finite_in_float32_is_refused():
    """With compiled=True a modulation strength that is not a finite number in float32 is refused,
    with the recurrent path's message, and so is a neurotransmitter vector with an entry the node
    rule or the inversion reads that is not finite in float32. The zero padding past a genome's
    count is inert only at a finite strength and finite entries m reads: at a padded position m =
    0, so s m = inf * 0 is NaN, and the zero W2 rows carry it to every output of the genome, where
    the dense forward has none (the reverse at some genomes under node rule 'steep_gate'); the
    inversion's entry acts after the output function in both forwards alike, and is refused with the
    others for one rule over the entries read. The dense forward checks neither and evaluates them;
    a finite strength of another type evaluates as the same float, and an entry neither the node
    rule nor the inversion reads may be anything."""
    substrate, inputs = _population("nm", "default")
    nm = api.substrate_neuromodulation(substrate)
    for strength in (float("inf"), -float("inf"), float("nan"), 1e40, True, "2.0"):
        settings = nm.settings._replace(modulation_strength=strength)
        configured = substrate._replace(config=substrate.config._replace(neuromodulation=settings))
        with pytest.raises(ValueError, match="modulation_strength must be a finite number"):
            substrate_forward(substrate, inputs, compiled=True,
                              neuromodulation=nm._replace(settings=settings))
        with pytest.raises(ValueError, match="modulation_strength must be a finite number"):
            substrate_forward(configured, inputs, compiled=True)
    infinite = nm._replace(settings=nm.settings._replace(modulation_strength=float("inf")))
    assert np.asarray(substrate_forward(substrate, inputs, neuromodulation=infinite)).shape == (
        POP, inputs.shape[0], 1)
    as_int = nm._replace(settings=nm.settings._replace(modulation_strength=2))
    _assert_same_program(
        substrate_forward(substrate, inputs, neuromodulation=as_int, compiled=True),
        substrate_forward(substrate, inputs, neuromodulation=nm, compiled=True))
    # 1e39 is finite in float64 and inf in float32, the precision the forward reads the vector at
    for vector in ((np.inf, 0.5, 0.5, 1.0), (0.5, -np.inf, 0.5, 1.0), (0.5, 0.5, np.nan, 1.0),
                   (0.5, 0.5, 0.5, np.inf), (0.5, 0.5, 1e39, 1.0)):
        given = nm._replace(neurotransmitter=np.asarray(vector, dtype=np.float64))
        with pytest.raises(ValueError, match="neurotransmitter vector's entries"):
            substrate_forward(substrate, inputs, neuromodulation=given, compiled=True)
        assert np.asarray(substrate_forward(substrate, inputs, neuromodulation=given)).shape == (
            POP, inputs.shape[0], 1)
    # With the inversion off the fourth entry is unread
    off = nm._replace(settings=nm.settings._replace(output_inversion=False))
    unread = off._replace(neurotransmitter=jnp.asarray([0.5, 0.5, 0.5, np.inf]))
    _assert_same_program(
        substrate_forward(substrate, inputs, neuromodulation=unread, compiled=True),
        substrate_forward(substrate, inputs, neuromodulation=off, compiled=True))


def test_the_capability_is_named():
    assert "compiled_forward" in api.SUBSTRATE_CAPABILITIES
