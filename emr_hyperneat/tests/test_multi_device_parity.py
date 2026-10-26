"""Every multi-device route of run_until_threshold scores a genome as SINGLE_GPU does.

In 0.1.0 nothing compared them, and three routes did not: EVAL_ONLY_PARALLEL scored its zero
padding rows when the sample count was not a multiple of the device count, FULL_PIPELINE_PARALLEL
counted its last sample once per padding copy, and POPULATION_PARALLEL_PROCESS scored every genome 0
with hidden-to-hidden connections. Each case runs one generation in a subprocess with forced CPU
devices and compares the fitness vector the route passes to NEAT at generation 0 (captured by
wrapping `_compiled_tell`) with SINGLE_GPU's for the same seed, and the random key the run returns:
every route must draw the keys SINGLE_GPU draws (in 0.1.0 FULL_PIPELINE_PARALLEL split the key
every generation by default and EVAL_ONLY_PARALLEL and POPULATION_PARALLEL_PROCESS never did,
whatever `extra_randkey_split` said). The routes sum in another order than SINGLE_GPU, so the
vectors agree to about 1e-7, not bit for bit; the defects they replace were 5e-3 and more.

The split routes POPULATION_PMAP and POSITION_SHARDING join these cases. They run SINGLE_GPU's own
generation and spread only its one-output CPPN query and its evaluation over the devices, so on
forced CPU devices they are held to more: SINGLE_GPU's run bit for bit over SPLIT_GENERATIONS
generations (SPLIT_SCRIPT, marked slow), HYBRID's 2 x 2 grid included on four devices.
"""

import json
import os
import subprocess
import sys

import pytest

ATOL = 1e-5

#: The harness subprocesses run under faulthandler, so a crash leaves its Python traceback in the
#: assertion message: one two-device split harness process crashed with SIGSEGV inside an XLA CPU
#: compile (macOS, jaxlib 0.6.1, under a load average of about 300; the same script passed twice
#: after). Each harness also releases the programs it compiled as it goes (jax.clear_caches()): a
#: process kept every one of them, 1,080 after twelve split runs, and on a Linux node at the
#: default vm.max_map_count (65530) such an accumulation crashed test_node_list.py inside an XLA
#: compile (docs/testing.md)
PYTHON = [sys.executable, '-X', 'faulthandler']


class SampleProblem:
    """n samples, not exhaustive: FULL_PIPELINE_PARALLEL shards it from 16 samples per device."""
    input_shape = (3,)
    output_shape = (1,)
    jitable = True
    use_bias = True
    is_exhaustive_dataset = False
    fitness_threshold = 0.95

    def __init__(self, n_samples):
        import numpy as np
        rng = np.random.default_rng(7)
        points = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
        self.inputs = [[float(x), float(y), 1.0] for x, y in points]
        self.targets = [[float(x * y > 0)] for x, y in points]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


def _label(strategy, n_samples, preset, level, split):
    """A case's label: strategy ('ref' for SINGLE_GPU), samples, preset and level, key split."""
    route = 'ref' if strategy == 'SINGLE_GPU' else strategy
    recurrence = 'ff' if preset is None else f'{preset}_{level}'
    return f'{route}_{n_samples}_{recurrence}' + ('_split' if split else '')


# (strategy, samples (4 = XOR), recurrence preset or None, iteration level, extra_randkey_split or
# None for the key left out). Every route case is compared with the SINGLE_GPU case of the same
# samples, preset, level and key split.
_CASE_FIELDS = [
    ('SINGLE_GPU', 4, None, 0, None),
    ('SINGLE_GPU', 4, 'hidden_only', 1, None),
    ('SINGLE_GPU', 4, 'hidden_only', 2, None),
    ('SINGLE_GPU', 4, 'full_recurrent', 2, None),
    ('SINGLE_GPU', 48, None, 0, None),
    ('SINGLE_GPU', 49, None, 0, None),
    ('SINGLE_GPU', 49, 'hidden_only', 1, None),
    ('EVAL_ONLY_PARALLEL', 4, None, 0, None),
    ('EVAL_ONLY_PARALLEL', 4, 'hidden_only', 1, None),
    ('EVAL_ONLY_PARALLEL', 4, 'hidden_only', 2, None),
    ('EVAL_ONLY_PARALLEL', 4, 'full_recurrent', 2, None),
    ('EVAL_ONLY_PARALLEL', 48, None, 0, None),
    ('EVAL_ONLY_PARALLEL', 49, None, 0, None),
    ('FULL_PIPELINE_PARALLEL', 48, None, 0, None),
    ('FULL_PIPELINE_PARALLEL', 49, None, 0, None),
    ('FULL_PIPELINE_PARALLEL', 49, 'hidden_only', 1, None),
    ('POPULATION_PARALLEL_PROCESS', 4, None, 0, None),
    ('POPULATION_PARALLEL_PROCESS', 4, 'hidden_only', 1, None),
    ('POPULATION_PARALLEL_PROCESS', 4, 'hidden_only', 2, None),
    ('POPULATION_PARALLEL_PROCESS', 4, 'full_recurrent', 2, None),
    ('POPULATION_PARALLEL_PROCESS', 49, None, 0, None),
    ('POPULATION_PMAP', 4, None, 0, None),
    ('POPULATION_PMAP', 4, 'hidden_only', 1, None),
    ('POPULATION_PMAP', 4, 'hidden_only', 2, None),
    ('POPULATION_PMAP', 4, 'full_recurrent', 2, None),
    ('POPULATION_PMAP', 49, None, 0, None),
    ('POSITION_SHARDING', 4, None, 0, None),
    ('POSITION_SHARDING', 4, 'hidden_only', 2, None),
    ('POSITION_SHARDING', 49, 'hidden_only', 1, None),
    ('SINGLE_GPU', 4, None, 0, True),
    ('SINGLE_GPU', 48, None, 0, True),
    ('EVAL_ONLY_PARALLEL', 4, None, 0, True),
    ('FULL_PIPELINE_PARALLEL', 48, None, 0, True),
    ('POPULATION_PARALLEL_PROCESS', 4, None, 0, True),
    ('POPULATION_PMAP', 4, None, 0, True),
    ('POSITION_SHARDING', 48, None, 0, True),
]
CASES = [(_label(*fields),) + fields for fields in _CASE_FIELDS]
ROUTE_CASES = [case for case in CASES if not case[0].startswith('ref_')]

PARITY_SCRIPT = r'''
import json
import sys
import warnings
sys.path.insert(0, {tests!r})
import jax
import numpy as np
from conftest import EMRHyperNEAT, XORProblem, create_base_config, create_config_with_recurrence
from test_multi_device_parity import CASES, SampleProblem
assert len(jax.devices()) == {devices}, jax.devices()
warnings.simplefilter('ignore')
found = {{}}
for label, strategy, n_samples, preset, level, split in CASES:
    problem = XORProblem() if n_samples == 4 else SampleProblem(n_samples)
    if preset is None:
        config = create_base_config(max_depth=3, population_size=8)
    else:
        config = create_config_with_recurrence(preset, max_depth=3, population_size=8,
                                               iteration_level=level, hh_cache_enabled=False)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multi_gpu_strategy'] = strategy
    if split is not None:  # None leaves the key out: its documented default, off
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['extra_randkey_split'] = split
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
    captured = []
    tell = algorithm._compiled_tell

    def capture(state, fitnesses, tell=tell, captured=captured):
        captured.append(np.asarray(fitnesses, dtype=np.float64).tolist())
        return tell(state, fitnesses)

    algorithm._compiled_tell = capture
    try:
        result = algorithm.run_until_threshold(state, problem, target_fitness=2.0,
                                               max_generations=1)
        found[label] = {{'route': result['route'], 'devices': result['devices'],
                         'fitness': captured[0],
                         'randkey': np.asarray(result['state'].randkey).tolist()}}
    except Exception as error:
        found[label] = {{'error': type(error).__name__ + ': ' + str(error)[:500]}}
    jax.clear_caches()
print('PARITY', json.dumps(found))
'''


def _host_devices_env(count):
    """The environment of a subprocess with `count` CPU devices.

    The platform selectors a GPU machine's profile sets (JAX_PLATFORM_NAME, JAX_BACKEND_TARGET) are
    removed: inherited, they keep JAX from starting on the CPU.
    """
    env = {key: value for key, value in os.environ.items()
           if key not in ('JAX_PLATFORM_NAME', 'JAX_BACKEND_TARGET')}
    env.update(JAX_PLATFORMS='cpu', XLA_FLAGS=f'--xla_force_host_platform_device_count={count}')
    return env


@pytest.fixture(scope='module', params=[2, 3, 4], ids=['2_devices', '3_devices', '4_devices'])
def parity(request):
    """Run every case on `request.param` forced CPU devices; return (devices, results by label)."""
    tests = os.path.dirname(os.path.abspath(__file__))
    script = PARITY_SCRIPT.format(tests=tests, devices=request.param)
    result = subprocess.run(PYTHON + ['-c', script], env=_host_devices_env(request.param),
                            capture_output=True, text=True, timeout=1800, cwd=tests)
    assert result.returncode == 0, result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith('PARITY ')][-1]
    return request.param, json.loads(line[len('PARITY '):])


@pytest.mark.parametrize('label,strategy,n_samples,preset,level,split', ROUTE_CASES,
                         ids=[c[0] for c in ROUTE_CASES])
def test_route_scores_as_single_gpu(parity, label, strategy, n_samples, preset, level, split):
    """The route ran on every device and matched SINGLE_GPU at generation 0.

    It handed NEAT SINGLE_GPU's fitness vector and returned SINGLE_GPU's random key.
    """
    devices, found = parity
    reference = found[_label('SINGLE_GPU', n_samples, preset, level, split)]
    case = found[label]
    assert 'error' not in reference, reference
    assert 'error' not in case, case
    # FULL_PIPELINE_PARALLEL runs on one device below 16 samples per device (run_until_threshold's
    # MIN_SAMPLES_PER_GPU): on four devices the 48 and 49 samples take that fallback
    expected = (
        ('SINGLE_GPU', 1)
        if strategy == 'FULL_PIPELINE_PARALLEL' and n_samples < 16 * devices
        else (strategy, devices)
    )
    assert (case['route'], case['devices']) == expected
    gap = max(abs(a - b) for a, b in zip(case['fitness'], reference['fitness']))
    assert len(case['fitness']) == len(reference['fitness']) and gap <= ATOL, (
        f'{label} on {devices} devices: largest gap {gap:.3g} to SINGLE_GPU')
    assert case['randkey'] == reference['randkey'], f'{label}: other random keys than SINGLE_GPU'


ZERO_SCRIPT = r'''
import json
import sys
import warnings
sys.path.insert(0, {tests!r})
from conftest import EMRHyperNEAT, XORProblem, create_base_config
from test_multi_device_parity import SampleProblem
warnings.simplefilter('ignore')
found = {{}}
for strategy, problem in (('SINGLE_GPU', XORProblem()), ('EVAL_ONLY_PARALLEL', XORProblem()),
                          ('FULL_PIPELINE_PARALLEL', SampleProblem(49)),
                          ('POPULATION_PARALLEL_PROCESS', XORProblem()),
                          ('POPULATION_PMAP', XORProblem()), ('POSITION_SHARDING', XORProblem()),
                          ('HYBRID', XORProblem())):
    config = create_base_config(max_depth=3, population_size=8)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['multi_gpu_strategy'] = strategy
    if strategy == 'HYBRID':
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['parallel'] = {{
            'population_devices': 1, 'position_devices': 2}}
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
    try:
        result = algorithm.run_until_threshold(state, problem, target_fitness=2.0,
                                               max_generations=0)
        found[strategy] = [result['route'], result['generations']]
    except Exception as error:
        found[strategy] = type(error).__name__ + ': ' + str(error)[:300]
print('ZERO', json.dumps(found))
'''


def test_zero_generations_on_every_route():
    """max_generations=0 returns zero generations on every route.

    In 0.1.0 each multi-device route read a variable its loop never bound and raised
    UnboundLocalError.
    """
    tests = os.path.dirname(os.path.abspath(__file__))
    script = ZERO_SCRIPT.format(tests=tests)
    result = subprocess.run(PYTHON + ['-c', script], env=_host_devices_env(2),
                            capture_output=True, text=True, timeout=900, cwd=tests)
    assert result.returncode == 0, result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith('ZERO ')][-1]
    assert json.loads(line[len('ZERO '):]) == {
        'SINGLE_GPU': ['SINGLE_GPU', 0],
        'EVAL_ONLY_PARALLEL': ['EVAL_ONLY_PARALLEL', 0],
        'FULL_PIPELINE_PARALLEL': ['FULL_PIPELINE_PARALLEL', 0],
        'POPULATION_PARALLEL_PROCESS': ['POPULATION_PARALLEL_PROCESS', 0],
        'POPULATION_PMAP': ['POPULATION_PMAP', 0],
        'POSITION_SHARDING': ['POSITION_SHARDING', 0],
        'HYBRID': ['HYBRID', 0],
    }


# The split routes, bit for bit over several generations

#: Generations of each split harness run; the population (uneven over 3 and 4 devices) and the depth
SPLIT_GENERATIONS = 5
SPLIT_POPULATION = 22
SPLIT_DEPTH = 3

#: The configurations the split harness runs (SPLIT_SCRIPT's form_config builds each): feed-forward,
#: the plain recurrent forward at levels 1 and 2, chunked, full recurrence, per-node functions,
#: aggregation, neuromodulation, LEO, and palette evolution, a host step between generations
SPLIT_FORMS = (
    'ff',
    'hidden_only_1',
    'hidden_only_2',
    'hidden_only_2_chunk5',
    'full_recurrent_2',
    'cppn_output_4',
    'agg_weight_interp',
    'true_neuromodulation',
    'leo',
    'palette_evolution',
)


def split_grids(devices):
    """The split routes the harness runs on DEVICES devices, as (label, strategy,
    emr_hyperneat.parallel or None): the population and the position split over every device, and
    HYBRID's 2 x 2 grid on four."""
    grids = [
        ('POPULATION_PMAP', 'POPULATION_PMAP', None),
        ('POSITION_SHARDING', 'POSITION_SHARDING', None),
    ]
    if devices == 4:
        grids.append(('HYBRID_2x2', 'HYBRID', {'population_devices': 2, 'position_devices': 2}))
    return grids


SPLIT_SCRIPT = r'''
import copy
import hashlib
import json
import sys
import warnings
sys.path.insert(0, {tests!r})
import jax
import numpy as np
from conftest import (
    EMRHyperNEAT,
    XORProblem,
    create_base_config,
    create_config_with_aggregation,
    create_config_with_dynamic_functions,
    create_config_with_leo,
    create_config_with_neuromodulation,
    create_config_with_recurrence,
)
from emr_hyperneat import (
    emrhyperneat as engine)
from test_multi_device_parity import (
    SPLIT_DEPTH,
    SPLIT_FORMS,
    SPLIT_GENERATIONS,
    SPLIT_POPULATION,
    split_grids,
)

assert len(jax.devices()) == {devices}, jax.devices()
warnings.simplefilter('ignore')


def form_config(form, population=SPLIT_POPULATION):
    depth = SPLIT_DEPTH
    if form == 'ff':
        config = create_base_config(max_depth=depth, population_size=population)
    elif form.startswith(('hidden_only', 'full_recurrent')):
        preset, level = form.split('_chunk')[0].rsplit('_', 1)
        config = create_config_with_recurrence(preset, max_depth=depth, population_size=population,
                                               iteration_level=int(level), hh_cache_enabled=False)
        if '_chunk' in form:
            chunk = int(form.split('_chunk')[1])
            config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'][
                'eval_chunk_size'
            ] = chunk
    elif form == 'cppn_output_4':
        config = create_config_with_dynamic_functions(
            'cppn_output_4', max_depth=depth, population_size=population
        )
    elif form == 'agg_weight_interp':
        config = create_config_with_aggregation(
            'weight_interp', max_depth=depth, population_size=population
        )
    elif form == 'true_neuromodulation':
        config = create_config_with_neuromodulation('true_neuromodulation_4nt', max_depth=depth,
                                                    population_size=population)
    elif form == 'leo':
        config = create_config_with_leo('default', max_depth=depth, population_size=population)
    elif form == 'palette_evolution':
        # The palette steps after one generation without improvement, so it changes within the run
        config = create_config_with_dynamic_functions(
            'cppn_output_4', max_depth=depth, population_size=population
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'][
            'palette_evolution'
        ] = {{'enabled': True, 'stagnation_generations': 1}}
    else:
        raise ValueError(form)
    # Elitism 2: at the default 15 a population of 22 keeps most genomes unchanged
    config['algorithm_params']['emrhyperneat']['neat_selection'] = {{'genome_elitism': 2}}
    return config


def bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32).tolist()


def digest(tree):
    sha = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        sha.update(repr((array.dtype.str, array.shape)).encode())
        sha.update(np.ascontiguousarray(array).tobytes())
    return sha.hexdigest()


def run(config, strategy, parallel=None):
    """A run of SPLIT_GENERATIONS generations through run_until_threshold, recorded generation by
    generation: the fitness vector NEAT was told, the species, the h->h and active counts, the
    weight densities, and which devices the split held; then the final state, the champion and its
    score on the training data."""
    config = copy.deepcopy(config)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['multi_gpu_strategy'] = strategy
    if parallel is not None:
        emr['parallel'] = parallel
    algorithm = EMRHyperNEAT()
    problem = XORProblem()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=42)
    generations = []
    verbose = algorithm.run_generation_verbose

    def recording(state, problem, *args, **kwargs):
        new_state, metrics = verbose(state, problem, *args, **kwargs)
        custom = metrics.custom_metrics
        extended = algorithm._extended_metrics
        split = algorithm._device_split
        generations.append({{
            'fitness': bits(algorithm._last_fitnesses),
            'species': [int(metrics.num_species), [int(size) for size in metrics.species_sizes]],
            'counts': [None if custom.get(key) is None else float(custom[key])
                       for key in ('total_hh_connections', 'min_hidden_nodes', 'max_hidden_nodes',
                                   'nan_fitness_count', 'total_positions')],
            'densities': [float(extended.w1_density), float(extended.w2_density)],
            'split': None if split is None else list(split.device_ids)}})
        return new_state, metrics

    algorithm.run_generation_verbose = recording
    try:
        result = algorithm.run_until_threshold(
            state,
            problem,
            target_fitness=2.0,
            max_generations=SPLIT_GENERATIONS,
            collect_history=True,
        )
    except Exception as error:
        return {{'error': type(error).__name__ + ': ' + str(error)[:500]}}
    nodes, conns, fitness, index = algorithm.get_best_genome_with_fitness()
    try:
        on_data = float(algorithm.evaluate_on_data(result['state'], algorithm._cached_inputs,
                                                   algorithm._cached_targets)['fitness'])
    except Exception as error:
        on_data = type(error).__name__ + ': ' + str(error)[:200]
    return {{
        'route': result['route'],
        'devices': result['devices'],
        'generations': generations,
        'history': [float(best) for best in result['history']],
        'state': digest(result['state']),
        'randkey': np.asarray(result['state'].randkey).tolist(),
        'champion': [digest((nodes, conns)), float(fitness), int(index)],
        'on_data': on_data,
        'split_after': algorithm._device_split is not None,
    }}


found = {{'runs': {{}}}}
for form in SPLIT_FORMS:
    config = form_config(form)
    found['runs'][form] = {{'SINGLE_GPU': run(config, 'SINGLE_GPU')}}
    for label, strategy, parallel in split_grids({devices}):
        found['runs'][form][label] = run(config, strategy, parallel)
    jax.clear_caches()

# Negative control: on a device holding one of one device's chunks, the vmap instead of the scan
# (one float32 ulp in a generation, test_device_split.py); the comparison must see it over the run
config = form_config('hidden_only_2', population=20)
config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['eval_chunk_size'] = 10
reference = run(config, 'SINGLE_GPU')
real_ranges = engine.evaluation_ranges
engine.evaluation_ranges = lambda *args: (real_ranges(*args)[0], False)
control = run(config, 'POPULATION_PMAP')
engine.evaluation_ranges = real_ranges
found['negative_control'] = [reference, control]
print('SPLIT', json.dumps(found))
'''


@pytest.fixture(scope='module', params=[2, 3, 4], ids=['2_devices', '3_devices', '4_devices'])
def split_parity(request):
    """Run SPLIT_SCRIPT on `request.param` forced CPU devices; return (devices, results)."""
    tests = os.path.dirname(os.path.abspath(__file__))
    script = SPLIT_SCRIPT.format(tests=tests, devices=request.param)
    result = subprocess.run(PYTHON + ['-c', script], env=_host_devices_env(request.param),
                            capture_output=True, text=True, timeout=3600, cwd=tests)
    assert result.returncode == 0, result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith('SPLIT ')][-1]
    return request.param, json.loads(line[len('SPLIT '):])


def comparable(record):
    """A split harness record without what names the route: what must equal SINGLE_GPU's."""
    if 'error' in record:
        return record
    kept = {key: value for key, value in record.items() if key not in ('route', 'devices')}
    kept['generations'] = [{key: value for key, value in generation.items() if key != 'split'}
                           for generation in record['generations']]
    return kept


def split_problems(devices, found):
    """What differs between each split route and SINGLE_GPU: the route record, the split held in
    every generation and released after the run, and everything comparable() keeps, bit for bit."""
    problems = []
    for form in SPLIT_FORMS:
        runs = found['runs'][form]
        reference = runs['SINGLE_GPU']
        if 'error' in reference:
            problems.append(f"{form}: SINGLE_GPU failed: {reference['error']}")
            continue
        if len(reference['generations']) != SPLIT_GENERATIONS or any(
                generation['split'] is not None for generation in reference['generations']):
            problems.append(f"{form}: SINGLE_GPU ran {len(reference['generations'])} "
                            f"generations or held a split")
        for label, strategy, parallel in split_grids(devices):
            case = runs[label]
            if 'error' in case:
                problems.append(f"{form}/{label}: {case['error']}")
                continue
            used = (
                devices
                if parallel is None
                else parallel['population_devices'] * parallel['position_devices']
            )
            if (case['route'], case['devices']) != (strategy, used):
                problems.append(
                    f"{form}/{label}: recorded {case['route']} on {case['devices']} devices"
                )
            if any(generation['split'] != list(range(used)) for generation in case['generations']):
                problems.append(f"{form}/{label}: a generation ran without the split: "
                                f"{[generation['split'] for generation in case['generations']]}")
            if case['split_after']:
                problems.append(f"{form}/{label}: the engine still holds the split after the run")
            if comparable(case) != comparable(reference):
                differing = sorted(key for key in comparable(reference)
                                   if comparable(case).get(key) != comparable(reference)[key])
                problems.append(f"{form}/{label}: differs from SINGLE_GPU in {differing}")
    reference, control = found['negative_control']
    if 'error' in reference or 'error' in control:
        problems.append(
            f"negative control failed to run: {reference.get('error')} {control.get('error')}"
        )
    elif comparable(control) == comparable(reference):
        problems.append("the vmap on a device's single chunk gives SINGLE_GPU's run: the "
                        "comparison is blind")
    return problems


@pytest.mark.slow
def test_split_routes_run_single_gpus_generations(split_parity):
    """POPULATION_PMAP and POSITION_SHARDING on 2, 3 and 4 forced CPU devices, HYBRID's 2 x 2 grid
    on four: each run of SPLIT_GENERATIONS generations equals SINGLE_GPU's bit for bit, in every
    form of SPLIT_FORMS. Every generation's fitness vector, species, counts and densities, the
    best-fitness history, the final state (population, species and key), the champion and its score
    on the training data; the route records itself and its devices, holds the split in every
    generation and releases it after the run. Negative control: the vmap on a device's single chunk.
    """
    devices, found = split_parity
    problems = split_problems(devices, found)
    assert not problems, f"{devices} devices:\n" + "\n".join(problems)
