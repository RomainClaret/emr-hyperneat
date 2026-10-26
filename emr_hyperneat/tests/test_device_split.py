"""The split routes' backends: the one-output CPPN query and the evaluation spread over devices,
the rest of the generation where it runs on one device.

The query backend: `query_population_split` runs each block of `query_blocks` (a range of genomes
at a range of grid positions) on its device with that device's compiled query, in chunks of at most
QUERY_POPULATION_CHUNK genomes, and gathers the blocks on the split's first device. The query is
elementwise in genomes and positions, so on forced CPU devices it must equal the one-device query
bit for bit, for every grid of devices, and a generation run with the split must equal SINGLE_GPU's:
every fitness array and the final key. A device holds one chunk's queries, so the memory its program
needs must not grow with the device count (the recurrent FULL_PIPELINE_PARALLEL route asked for
69.48 GiB on its first GPU because it queried every genome at once).

The eval backend: `evaluate_population_split` spreads the population over every device of the
split. Each device evaluates a range of genomes as one device evaluates a population
(`evaluate_population_on_device`); when one device's evaluation is chunked
(`recurrence.eval_chunk_size`), the devices get its own chunks, evaluated in its form (the scan or
the loop). A genome's fitness does not depend on the other genomes of its batch, so a generation run
with both backends must equal SINGLE_GPU's bit for bit, for every evaluation form: feed-forward, the
plain and per-node recurrent forwards, per-node functions, aggregation, neuromodulation, LEO, and
both chunked forms. The negative control evaluates a device's single chunk with the vmap instead of
the scan: one float32 ulp, which the comparison must see.

The routes POPULATION_PMAP, POSITION_SHARDING and HYBRID: run_until_threshold holds the split for
the run of the one-device loop and releases it after, failed or not. emr_hyperneat.parallel sets
the grid. On two forced devices each route records itself and its devices and calls both backends
once per generation; a grid of one device runs the one-device generation; the routes refuse what
would bypass the backends (an explicit streaming or depth offload, AUTO resolved to depth offload,
the compiled generation loop), a population that leaves a device fewer than two genomes, and a grid
larger than the visible devices. On one device each runs the one-device loop. Their equality with
SINGLE_GPU, bit for bit over five generations, is test_multi_device_parity.py's split harness.

The multi-device cases run in subprocesses with forced CPU devices, as test_multi_device_parity.py
does. The backends' harness on 2, 3 and 4 devices is marked slow.
"""

from __future__ import annotations

import json
import os
import subprocess

import jax
import numpy as np
import pytest

from conftest import MultiGPUStrategy, EMRHyperNEAT, XORProblem, create_base_config
from emr_hyperneat import (
    emrhyperneat as engine,
)
from test_multi_device_parity import PYTHON, _host_devices_env
from test_node_list import _node_list_config


class TestSplitRanges:
    @pytest.mark.parametrize('size', [0, 1, 2, 5, 22, 100, 1364])
    @pytest.mark.parametrize('parts', [1, 2, 3, 4, 7])
    def test_equals_numpy_array_split(self, size, parts):
        expected = [(int(chunk[0]), int(chunk[-1]) + 1) if chunk.size else None
                    for chunk in np.array_split(np.arange(size), parts)]
        got = engine.split_ranges(size, parts)
        assert len(got) == parts
        assert [r if r[1] > r[0] else None for r in got] == expected
        assert got[0][0] == 0 and got[-1][1] == size


class TestMakeDeviceSplit:
    def test_one_device(self):
        assert engine.make_device_split(1) == engine.DeviceSplit(1, 1, (0,))

    @pytest.mark.parametrize(
        'population_devices,position_devices',
        [(0, 1), (1, 0), (-2, 1), (True, 1), (1.0, 1), ('2', 1)],
    )
    def test_counts_must_be_positive_integers(self, population_devices, position_devices):
        with pytest.raises(ValueError, match='must be a positive integer'):
            engine.make_device_split(population_devices, position_devices)

    def test_more_devices_than_jax_sees(self):
        with pytest.raises(ValueError, match='needs 64 devices'):
            engine.make_device_split(8, 8)


class TestQueryBlocks:
    def test_blocks_cover_the_population_and_the_positions_once(self):
        split = engine.DeviceSplit(2, 3, (0, 1, 2, 3, 4, 5))
        blocks = engine.query_blocks(22, 1364, split)
        assert [b[0] for b in blocks] == [0, 1, 2, 3, 4, 5]
        covered = np.zeros((22, 1364), dtype=int)
        for _, (g0, g1), (p0, p1) in blocks:
            covered[g0:g1, p0:p1] += 1
        assert (covered == 1).all()

    def test_empty_blocks_are_left_out(self):
        # Three genomes over four population devices: the fourth device has none
        blocks = engine.query_blocks(3, 10, engine.DeviceSplit(4, 1, (0, 1, 2, 3)))
        assert [b[0] for b in blocks] == [0, 1, 2]


class TestParallelBlock:
    """emr_hyperneat.parallel, the split routes' device grid (parse_parallel_grid, through
    create_config)."""

    @pytest.mark.parametrize(
        'name', ['SINGLE_GPU', 'POPULATION_PMAP', 'POSITION_SHARDING', 'EVAL_ONLY_PARALLEL']
    )
    def test_absent(self, name):
        assert engine.parse_parallel_grid(None, MultiGPUStrategy[name]) is None

    def test_hybrid_needs_both_counts(self):
        for block in (None, {'population_devices': 2}, {'position_devices': 2}):
            with pytest.raises(
                ValueError, match='HYBRID splits the genomes and the grid positions at once'
            ):
                engine.parse_parallel_grid(block, MultiGPUStrategy.HYBRID)
        assert engine.parse_parallel_grid({'population_devices': 2, 'position_devices': 3},
                                          MultiGPUStrategy.HYBRID) == (2, 3)

    @pytest.mark.parametrize(
        'name', ['SINGLE_GPU', 'STREAMING', 'EVAL_ONLY_PARALLEL', 'FULL_PIPELINE_PARALLEL']
    )
    def test_other_strategies_read_none_of_it(self, name):
        with pytest.raises(ValueError, match=f'multi_gpu_strategy {name} reads none of it'):
            engine.parse_parallel_grid({'population_devices': 2}, MultiGPUStrategy[name])

    @pytest.mark.parametrize('block, match', [
        ([2, 1], 'must be a mapping'),
        ({'population_devices': 2, 'devices': 2}, r"unknown keys \['devices'\]"),
        ({'population_devices': 0}, 'population_devices must be a positive integer'),
        ({'population_devices': True}, 'population_devices must be a positive integer'),
        ({'population_devices': 2.0}, 'population_devices must be a positive integer'),
        ({'position_devices': '2'}, 'position_devices must be a positive integer'),
    ])
    def test_wrong_blocks(self, block, match):
        with pytest.raises(ValueError, match=match):
            engine.parse_parallel_grid(block, MultiGPUStrategy.POPULATION_PMAP)

    def test_each_route_reads_its_axis(self):
        population, position = MultiGPUStrategy.POPULATION_PMAP, MultiGPUStrategy.POSITION_SHARDING
        assert engine.parse_parallel_grid({'population_devices': 3}, population) == (3, None)
        assert engine.parse_parallel_grid(
            {'population_devices': 3, 'position_devices': 1}, population
        ) == (3, 1)
        assert engine.parse_parallel_grid({'position_devices': 2}, position) == (None, 2)
        with pytest.raises(ValueError, match='POPULATION_PMAP splits the genomes only'):
            engine.parse_parallel_grid({'position_devices': 2}, population)
        with pytest.raises(ValueError, match='POSITION_SHARDING splits the grid positions only'):
            engine.parse_parallel_grid({'population_devices': 2}, position)

    def test_through_create_config(self):
        config = create_base_config(max_depth=2, population_size=8)
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['multi_gpu_strategy'] = 'POPULATION_PMAP'
        emr['parallel'] = {'population_devices': 2}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm._parallel_grid == (2, None)
        emr['multi_gpu_strategy'] = 'SINGLE_GPU'
        with pytest.raises(ValueError, match='multi_gpu_strategy SINGLE_GPU reads none of it'):
            EMRHyperNEAT().create_config(config)

    def test_a_constructed_hybrid_needs_its_grid(self):
        """A strategy from the constructor is checked against the block as the configuration key's
        is."""
        with pytest.raises(
            ValueError, match='HYBRID splits the genomes and the grid positions at once'
        ):
            EMRHyperNEAT(strategy=MultiGPUStrategy.HYBRID).create_config(
                create_base_config(max_depth=2, population_size=8))

    def test_hmr_names(self):
        """HMR's names for the split routes (STRATEGY_NAME_MAP). HMR ran CPPN_CHUNKED and
        POSITION_SHARDING_CHUNKED as a CPPN query split by position; 0.1.0's configuration key ran
        them as STREAMING and FULL_PIPELINE_PARALLEL."""
        expected = {'POPULATION_PMAP': 'POPULATION_PMAP', 'POPULATION_PARALLEL': 'POPULATION_PMAP',
                    'POSITION_SHARDING': 'POSITION_SHARDING', 'POSITION_PMAP': 'POSITION_SHARDING',
                    'SHARD_MAP': 'POSITION_SHARDING', 'CPPN_CHUNKED': 'POSITION_SHARDING',
                    'POSITION_SHARDING_CHUNKED': 'POSITION_SHARDING', 'HYBRID': 'HYBRID'}
        for name, route in expected.items():
            assert engine.STRATEGY_NAME_MAP[name] == MultiGPUStrategy[route], name
            assert (
                EMRHyperNEAT(strategy=MultiGPUStrategy[name]).strategy == MultiGPUStrategy[route]
            ), name

    def test_node_list_refuses_them(self):
        with pytest.raises(ValueError, match='strategy POPULATION_PMAP'):
            EMRHyperNEAT().create_config(_node_list_config(multi_gpu_strategy='POPULATION_PMAP'))


@pytest.mark.skipif(len(jax.devices()) >= 2, reason='the one-device fallback needs one device')
class TestOneDevice:
    @pytest.mark.parametrize('name', ['POPULATION_PMAP', 'POSITION_SHARDING', 'HYBRID'])
    def test_runs_the_one_device_loop(self, name, capsys):
        """On one device a split route runs the one-device loop and says so, as the other
        multi-device routes do; the engine holds no split."""
        config = create_base_config(max_depth=2, population_size=8)
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['multi_gpu_strategy'] = name
        if name == 'HYBRID':
            emr['parallel'] = {'population_devices': 2, 'position_devices': 2}
        algorithm = EMRHyperNEAT()
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
        result = algorithm.run_until_threshold(
            state, problem, target_fitness=2.0, max_generations=1
        )
        assert (result['route'], result['devices'], algorithm.get_num_devices()) == (
            'SINGLE_GPU',
            1,
            1,
        )
        assert algorithm._device_split is None
        assert f'{name} requested but only 1 GPU available' in capsys.readouterr().out


class TestEvaluationRanges:
    def test_unchunked_ranges_are_split_ranges(self):
        assert engine.evaluation_ranges(22, None, 4) == (
            [(0, 6), (6, 12), (12, 17), (17, 22)],
            False,
        )
        # A population not above the chunk is evaluated unchunked, as on one device
        assert engine.evaluation_ranges(22, 30, 2) == ([(0, 11), (11, 22)], False)
        assert engine.evaluation_ranges(3, None, 4) == ([(0, 1), (1, 2), (2, 3)], False)

    def test_chunked_ranges_are_whole_chunks_of_one_device(self):
        # Five chunks of five (the last one of two) over four devices: two chunks, then one each
        assert engine.evaluation_ranges(22, 5, 4) == ([(0, 10), (10, 15), (15, 20), (20, 22)], True)
        # Two chunks over three devices: the third device has none
        assert engine.evaluation_ranges(20, 10, 3) == ([(0, 10), (10, 20)], True)


SCRIPT = r'''
import json
import sys
import warnings
sys.path.insert(0, {tests!r})
import jax
import numpy as np
from conftest import EMRHyperNEAT, XORProblem, create_base_config, create_config_with_recurrence
from emr_hyperneat import (
    emrhyperneat as engine)
assert len(jax.devices()) == {devices}, jax.devices()
warnings.simplefilter('ignore')
n = {devices}
grids = [(n, 1), (1, n)] + ([(2, n // 2)] if n == 4 else [])
found = {{'query': {{}}, 'placement': {{}}, 'generation': {{}}, 'memory': {{}}}}


def bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def engine_at(depth, pop, config=None):
    algo = EMRHyperNEAT()
    config = config or create_base_config(max_depth=depth, population_size=pop)
    state = algo.initialize(algo.create_config(config), XORProblem(), seed=42)
    cppns = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    return algo, state, cppns


# 1. The split query equals the one-device query, on the first device
for pop in (22, 230):
    algo, state, cppns = engine_at(4, pop)
    for side, rows in zip(('in', 'out'), algo._substrate_query_rows()):
        reference = np.asarray(
            engine.query_population_on_device(state, cppns, rows, algo._jitted_cppn_forward)
        )
        for grid in grids:
            got = engine.query_population_split(state, cppns, rows, algo._jitted_cppn_forward,
                                                engine.make_device_split(*grid))
            found['query'][f'{{pop}}_{{side}}_{{grid[0]}}x{{grid[1]}}'] = [
                np.asarray(got).shape == reference.shape
                and bool(np.array_equal(bits(got), bits(reference))),
                [d.id for d in got.devices()],
            ]

# The negative control: genome ranges that each start one genome early (a share's first genome
# queried twice)
real_split_ranges = engine.split_ranges
engine.split_ranges = lambda size, parts: [
    (max(0, a - 1), b) for a, b in real_split_ranges(size, parts)
]
algo, state, cppns = engine_at(4, 22)
rows = algo._substrate_query_rows()[0]
reference = np.asarray(
    engine.query_population_on_device(state, cppns, rows, algo._jitted_cppn_forward)
)
bad = np.asarray(engine.query_population_split(state, cppns, rows, algo._jitted_cppn_forward,
                                               engine.make_device_split(n)))
engine.split_ranges = real_split_ranges
found['negative_control'] = bad.shape == reference.shape and bool(
    np.array_equal(bits(bad), bits(reference))
)

# 2. Each block runs on its own device
algo, state, cppns = engine_at(4, 22)
rows = algo._substrate_query_rows()[0]
for grid in grids:
    split = engine.make_device_split(*grid)
    on = []
    for device_id, (g0, g1), (p0, p1) in engine.query_blocks(22, rows[0].shape[0], split):
        device = jax.devices()[device_id]
        block = engine.query_population_on_device(
            jax.device_put(state, device),
            tuple(jax.device_put(p[g0:g1], device) for p in cppns),
            tuple(jax.device_put(r[p0:p1], device) for r in rows),
            algo._jitted_cppn_forward,
            device_id=device_id,
        )
        on.append([device_id, [d.id for d in block.devices()]])
    found['placement'][f'{{grid[0]}}x{{grid[1]}}'] = on


# 3. A generation with the query split equals SINGLE_GPU's, every fitness array and the final key
def run(config, split_grid, generations=3):
    algo = EMRHyperNEAT()
    problem = XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=42)
    if split_grid is not None:
        algo._device_split = engine.make_device_split(*split_grid)
    told = []
    tell = algo._compiled_tell

    def capture(state, fitnesses, tell=tell):
        told.append(bits(fitnesses).tolist())
        return tell(state, fitnesses)

    algo._compiled_tell = capture
    for _ in range(generations):
        state, _ = algo.run_generation(state, problem, verbose=False)
    return told, np.asarray(state.randkey).tolist()


cases = {{
    'ff': create_base_config(max_depth=3, population_size=22),
    'hidden_only_1': create_config_with_recurrence(
        'hidden_only', max_depth=3, population_size=22, iteration_level=1, hh_cache_enabled=False
    ),
}}
for label, config in cases.items():
    config['algorithm_params']['emrhyperneat']['neat_selection'] = {{'genome_elitism': 2}}
    reference = run(config, None)
    for grid in grids:
        found['generation'][f'{{label}}_{{grid[0]}}x{{grid[1]}}'] = run(config, grid) == reference
jax.clear_caches()


# 4. The memory of the largest per-device query program, against one device's (population 1000,
# depth 4)
def program_bytes(algo, state, cppns, rows, split, chunk):
    largest = 0
    for device_id, (g0, g1), (p0, p1) in engine.query_blocks(
        cppns[0].shape[0], rows[0].shape[0], split, chunk
    ):
        device = jax.devices()[device_id]
        fn = engine._get_device_specific_query_fn(device_id, algo._jitted_cppn_forward)
        size = min(chunk, g1 - g0)
        stats = (
            fn.lower(
                jax.device_put(state, device),
                jax.device_put(rows[0][p0:p1], device),
                *(jax.device_put(p[g0 : g0 + size], device) for p in cppns),
            )
            .compile()
            .memory_analysis()
        )
        largest = max(
            largest,
            stats.temp_size_in_bytes + stats.argument_size_in_bytes + stats.output_size_in_bytes,
        )
    return largest


algo, state, cppns = engine_at(4, 1000)
rows = algo._substrate_query_rows()[0]
chunk = engine.QUERY_POPULATION_CHUNK
one = program_bytes(algo, state, cppns, rows, engine.make_device_split(1), chunk)
found['memory']['one_device'] = one
for grid in grids:
    found['memory'][f'{{grid[0]}}x{{grid[1]}}'] = program_bytes(
        algo, state, cppns, rows, engine.make_device_split(*grid), chunk
    )
# The negative control: each device querying its whole share of the population at once
found['memory']['whole_share'] = program_bytes(
    algo, state, cppns, rows, engine.make_device_split(n), -(-1000 // n)
)
jax.clear_caches()


# 5. A generation with both backends (the query and the evaluation split) equals SINGLE_GPU's
from conftest import (
    create_config_with_aggregation,
    create_config_with_dynamic_functions,
    create_config_with_leo,
    create_config_with_neuromodulation,
)


def both(config, split_grid, generations=3):
    return run(config, split_grid, generations)


def recurrent(preset, level, chunk=None):
    config = create_config_with_recurrence(
        preset, max_depth=3, population_size=22, iteration_level=level, hh_cache_enabled=False
    )
    if chunk is not None:
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'][
            'eval_chunk_size'
        ] = chunk
    return config


def chunked_feedforward(chunk):
    config = create_base_config(max_depth=3, population_size=22)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = {{
        'eval_chunk_size': chunk
    }}
    return config


eval_cases = {{
    'ff': create_base_config(max_depth=3, population_size=22),
    'ff_chunk6': chunked_feedforward(6),
    'hidden_only_1': recurrent('hidden_only', 1),
    'hidden_only_2': recurrent('hidden_only', 2),
    'hidden_only_2_chunk5': recurrent('hidden_only', 2, chunk=5),
    'full_recurrent_2': recurrent('full_recurrent', 2),
    'cppn_output_4': create_config_with_dynamic_functions(
        'cppn_output_4', max_depth=3, population_size=22
    ),
    'agg_weight_interp': create_config_with_aggregation(
        'weight_interp', max_depth=3, population_size=22
    ),
    'context_gating': create_config_with_neuromodulation(
        'context_gating', max_depth=3, population_size=22
    ),
    'true_neuromodulation': create_config_with_neuromodulation(
        'true_neuromodulation_4nt', max_depth=3, population_size=22
    ),
    'leo': create_config_with_leo('default', max_depth=3, population_size=22),
}}
found['evaluation'] = {{}}
for label, config in eval_cases.items():
    config['algorithm_params']['emrhyperneat']['neat_selection'] = {{'genome_elitism': 2}}
    reference = both(config, None)
    for grid in grids:
        found['evaluation'][f'{{label}}_{{grid[0]}}x{{grid[1]}}'] = both(config, grid) == reference
    jax.clear_caches()

# 6. Each range is evaluated on its own device
seen = []
real_on_device = engine.evaluate_population_on_device


def recording(forward_one, genome_args, *rest, **kwargs):
    seen.append(sorted(d.id for d in genome_args[0].devices()))
    return real_on_device(forward_one, genome_args, *rest, **kwargs)


engine.evaluate_population_on_device = recording
config = create_base_config(max_depth=3, population_size=22)
both(config, (n, 1), generations=1)
engine.evaluate_population_on_device = real_on_device
found['eval_placement'] = seen

# Negative control: a device holding one of one device's chunks evaluates it with the vmap, not the
# scan
config = recurrent('hidden_only', 2, chunk=10)
config['algorithm_params']['emrhyperneat']['population_size'] = 20
config['algorithm_params']['emrhyperneat']['neat_selection'] = {{'genome_elitism': 2}}
reference = both(config, None)
found['eval_chunk_form'] = both(config, (n, 1)) == reference
real_ranges = engine.evaluation_ranges
engine.evaluation_ranges = lambda *args: (real_ranges(*args)[0], False)
found['eval_negative_control'] = both(config, (n, 1)) == reference
engine.evaluation_ranges = real_ranges
print('SPLIT', json.dumps(found))
'''


@pytest.fixture(scope='module', params=[2, 3, 4], ids=['2_devices', '3_devices', '4_devices'])
def split_run(request):
    """Run every case on `request.param` forced CPU devices; return (devices, results)."""
    tests = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        PYTHON + ['-c', SCRIPT.format(tests=tests, devices=request.param)],
        env=_host_devices_env(request.param),
        capture_output=True,
        text=True,
        # As test_multi_device_parity.py's split harness: beside a 12-worker suite an x86 Xeon took
        # 1,701 and 1,764 s on two and three devices, and over 1,800 s on four
        timeout=3600,
        cwd=tests,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith('SPLIT ')][-1]
    return request.param, json.loads(line[len('SPLIT '):])


def query_problems(devices, found):
    """The query backend's checks: the split query equals the one-device query bit for bit and lands
    on the split's first device; each block runs on its own device; a generation run with the split
    query equals SINGLE_GPU's (every fitness array, the final key); a device's query program needs
    no more memory than one device's. Negative controls: overlapping genome ranges, and a device
    querying its whole share at once."""
    problems = []
    if not found['query'] or not all(equal for equal, _ in found['query'].values()):
        problems.append(f"split query differs from the one-device query: {found['query']}")
    if not all(on == [0] for _, on in found['query'].values()):
        problems.append(f"split query not gathered on device 0: {found['query']}")
    if found['negative_control']:
        problems.append("overlapping genome ranges still equal the one-device query: the "
                        "comparison is blind")
    for grid, blocks in found['placement'].items():
        if sorted(device for device, _ in blocks) != list(range(devices)) or not all(
                on == [device] for device, on in blocks):
            problems.append(f"{grid}: blocks not on their devices: {blocks}")
    if not found['generation'] or not all(found['generation'].values()):
        problems.append(
            f"a generation with the split query differs from SINGLE_GPU's: {found['generation']}"
        )
    memory = found['memory']
    one = memory['one_device']
    for grid, size in memory.items():
        if grid not in ('one_device', 'whole_share') and size > one:
            problems.append(
                f"{grid}: a device's query program needs {size} bytes, one device's {one}"
            )
    # A device that queries its whole share at once (1000 / n genomes, above the 100-genome chunk at
    # 2, 3 and 4 devices) needs more than one device's chunked program
    if memory['whole_share'] <= one:
        problems.append(f"the whole-share control needs no more memory than one device ({memory})")
    return problems


def eval_problems(devices, found):
    """The eval backend's checks: a generation run with both backends equals SINGLE_GPU's bit for
    bit (every fitness array and the final key) for every evaluation form, unchunked and chunked;
    each device evaluates its own range. Negative control: the vmap instead of the scan on a device
    holding one chunk."""
    problems = []
    failing = [label for label, equal in found['evaluation'].items() if not equal]
    if not found['evaluation'] or failing:
        problems.append(f"a generation with both backends differs from SINGLE_GPU's: {failing}")
    if found['eval_placement'] != [[device] for device in range(devices)]:
        problems.append(f"ranges not evaluated each on its own device: {found['eval_placement']}")
    if not found['eval_chunk_form']:
        problems.append("the chunked evaluation, one chunk per device, differs from SINGLE_GPU's")
    if found['eval_negative_control']:
        problems.append("the vmap on a device's single chunk equals the scan's fitness: the "
                        "comparison is blind")
    return problems


@pytest.mark.slow
def test_split_backends(split_run):
    """Both backends on 2, 3 and 4 forced CPU devices, one subprocess per device count (see the two
    checks above)."""
    devices, found = split_run
    problems = query_problems(devices, found) + eval_problems(devices, found)
    assert not problems, f"{devices} devices:\n" + "\n".join(problems)


ROUTE_SCRIPT = r'''
import json
import sys
import warnings
sys.path.insert(0, {tests!r})
import jax
from conftest import (
    EMRHyperNEAT,
    XORProblem,
    create_base_config,
    create_config_with_dynamic_functions,
    create_config_with_streaming,
)
from emr_hyperneat import (
    emrhyperneat as engine)
from emr_hyperneat.execution import ExecutionMode
assert len(jax.devices()) == 2, jax.devices()
warnings.simplefilter('ignore')
calls = {{'query': 0, 'evaluate': 0}}


def counting(name, real):
    def wrapper(*args, **kwargs):
        calls[name] += 1
        return real(*args, **kwargs)
    return wrapper


engine.query_population_split = counting('query', engine.query_population_split)
engine.evaluate_population_split = counting('evaluate', engine.evaluate_population_split)
found = {{}}


def attempt(
    label, strategy, population=8, parallel=None, execution=None, config=None, prepare=None
):
    """Two generations of STRATEGY through run_until_threshold: the route, its devices,
    get_num_devices and the mode it ran, and how often each backend ran; or what refused it.
    Whether the engine still holds a split after."""
    config = config or create_base_config(max_depth=3, population_size=population)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['multi_gpu_strategy'] = strategy
    if parallel is not None:
        emr['parallel'] = parallel
    if execution is not None:
        emr['execution'] = execution
    if population < 4:
        config['algorithm_params']['emrhyperneat']['neat_selection'] = {{'genome_elitism': 1}}
    calls.update(query=0, evaluate=0)
    algorithm = EMRHyperNEAT()
    try:
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
        if prepare is not None:
            prepare(algorithm)
        result = algorithm.run_until_threshold(
            state, problem, target_fitness=2.0, max_generations=2
        )
        found[label] = {{'route': [result['route'], result['devices'], algorithm.get_num_devices(),
                                   result['execution_mode']], 'calls': dict(calls)}}
    except Exception as error:
        found[label] = {{'refused': type(error).__name__ + ': ' + str(error)}}
    found[label]['split_after'] = getattr(algorithm, '_device_split', None) is not None
    jax.clear_caches()


def resolve_to_offload(algorithm):
    # AUTO escalated to depth offload, as on a GPU whose memory the standard generation's estimate
    # exceeds
    algorithm.resolve_execution_mode = lambda: ExecutionMode.DEPTH_OFFLOAD


def fail_generation(algorithm):
    def failing(*args, **kwargs):
        raise RuntimeError('a generation failed')
    algorithm.run_generation_verbose = failing


def palette_evolution():
    config = create_config_with_dynamic_functions('cppn_output_4', max_depth=3, population_size=8)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'][
        'palette_evolution'
    ] = {{'enabled': True}}
    return config


attempt('single', 'SINGLE_GPU')
attempt('population', 'POPULATION_PMAP')
attempt('position', 'POSITION_SHARDING')
attempt('cppn_chunked', 'CPPN_CHUNKED')
attempt('hybrid_1x2', 'HYBRID', parallel={{'population_devices': 1, 'position_devices': 2}})
attempt('population_on_one', 'POPULATION_PMAP', parallel={{'population_devices': 1}})
attempt('palette_evolution', 'POPULATION_PMAP', config=palette_evolution())
attempt('hybrid_2x2', 'HYBRID', parallel={{'population_devices': 2, 'position_devices': 2}})
attempt('too_few_genomes', 'POPULATION_PMAP', population=3)
attempt('depth_offload', 'POPULATION_PMAP', execution={{'mode': 'depth_offload'}})
attempt('streaming', 'POSITION_SHARDING', execution={{'mode': 'streaming'}})
attempt(
    'enable_streaming',
    'POPULATION_PMAP',
    config=create_config_with_streaming(max_depth=3, population_size=8),
)
attempt('auto_offload', 'POPULATION_PMAP', prepare=resolve_to_offload)
attempt('while_loop', 'POSITION_SHARDING', execution={{'generation_loop': 'while_loop'}})
attempt('failed_generation', 'POPULATION_PMAP', prepare=fail_generation)
print('ROUTES', json.dumps(found))
'''

#: What each route of ROUTE_SCRIPT records: [route, devices, get_num_devices, execution mode] and
#: how often each backend ran in two generations. The one-output query runs twice a generation (the
#: input rows, then the output rows); a multi-output CPPN (palette evolution's per-node functions)
#: keeps its query on the first device: the query backend splits one-output queries only
ROUTE_RUNS = {
    'single': (['SINGLE_GPU', 1, 1, 'standard'], {'query': 0, 'evaluate': 0}),
    'population': (['POPULATION_PMAP', 2, 2, 'standard'], {'query': 4, 'evaluate': 2}),
    'position': (['POSITION_SHARDING', 2, 2, 'standard'], {'query': 4, 'evaluate': 2}),
    'cppn_chunked': (['POSITION_SHARDING', 2, 2, 'standard'], {'query': 4, 'evaluate': 2}),
    'hybrid_1x2': (['HYBRID', 2, 2, 'standard'], {'query': 4, 'evaluate': 2}),
    'population_on_one': (['POPULATION_PMAP', 1, 1, 'standard'], {'query': 0, 'evaluate': 0}),
    'palette_evolution': (['POPULATION_PMAP', 2, 2, 'standard'], {'query': 0, 'evaluate': 2}),
}
#: What refuses each other case of ROUTE_SCRIPT (a part of the message)
ROUTE_REFUSALS = {
    'hybrid_2x2': 'needs 4 devices',
    'too_few_genomes': 'fewer than two genomes',
    'depth_offload': "would ignore execution.mode 'depth_offload'",
    'streaming': "would ignore execution.mode 'streaming'",
    'enable_streaming': 'streaming',
    'auto_offload': 'runs the depth_offload generation',
    'while_loop': "'while_loop' runs one device",
    'failed_generation': 'RuntimeError: a generation failed',
}


@pytest.fixture(scope='module')
def routes_on_two_devices():
    tests = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        PYTHON + ['-c', ROUTE_SCRIPT.format(tests=tests)],
        env=_host_devices_env(2),
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=tests,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith('ROUTES ')][-1]
    return json.loads(line[len('ROUTES '):])


def test_routes_on_two_devices(routes_on_two_devices):
    """On two forced CPU devices: each split route records itself, its devices and the standard
    mode, and runs both backends in every generation (CPPN_CHUNKED is POSITION_SHARDING; a grid of
    one device runs neither; SINGLE_GPU runs neither); palette evolution runs. Refused: a grid
    larger than the visible devices, fewer than two genomes per device, an explicit depth offload or
    streaming, the legacy enable_streaming, AUTO resolved to depth offload, and the compiled
    generation loop. The engine holds no split after any run, a failed generation's included."""
    found = routes_on_two_devices
    problems = []
    for label, (route, calls) in ROUTE_RUNS.items():
        if found[label].get('route') != route or found[label].get('calls') != calls:
            problems.append(f"{label}: {found[label]}, expected {route} and {calls}")
    for label, part in ROUTE_REFUSALS.items():
        if part not in found[label].get('refused', ''):
            problems.append(f"{label}: {found[label]}, expected a refusal saying {part!r}")
    held = [label for label, case in found.items() if case['split_after']]
    if held:
        problems.append(f"the engine still holds a split after {held}")
    assert set(found) == set(ROUTE_RUNS) | set(ROUTE_REFUSALS), sorted(found)
    assert not problems, "\n".join(problems)
