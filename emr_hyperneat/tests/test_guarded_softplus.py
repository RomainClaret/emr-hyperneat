"""dynamic_functions.guarded_softplus: an opt-in guarded softplus, off by default.

``softplus``, log1p(exp(x)) and index 7 of ACTIVATION_LIST, is inf in float32 from x = 88.72284
(0x42b17218, the first float32 past ln of float32's largest value) on, as published: a node or an
output past it gives inf, and NaN where the inf meets a zero weight. With the key set
(``emr_hyperneat.dynamic_functions.guarded_softplus``, which ``substrate_config`` carries as
``SubstrateConfig.guarded_softplus``) every softplus the evaluation applies, a palette's index 7,
the global hidden activation, the output activation and an activation named for a multi-task
evaluation, is ``e = exp(x); where(isinf(e), x, log1p(e))`` (``activation_function('softplus',
True)``): every value the published formula gives finite, bit for bit, and x itself where exp(x)
overflows. The guarded function is in neither ACTIVATION_LIST nor the registry, so no name, index,
palette or draw changes. With the key off no computed value moves: the generation's forward and the
substrate API's dense, compiled and recurrent forwards trace the programs 0.3.1 traces. With it on,
wherever no softplus overflows, the function and every forward run op by op agree with the published
one bit for bit, while a compiled evaluation that applies a softplus is another program than the
published setting's and agrees within what two programs round apart (CPU_CROSS_PROGRAM_ULPS).

Covered: the function against the published one; the activation list and the registry; the
per-node forwards; the configuration key and SubstrateConfig's field, a bool that every forward
reads through one check; what moves with the key off (the field count, ``_asdict()``, the
capabilities and the configuration keys); the generation (feed-forward global and per-node,
recurrent global and per-node under both settling rules, neuromodulated with a global and with
per-node activations, feed-forward and recurrent, multi-task) and the substrate API's forwards
against it, each forward's hidden and output lookups apart; and the compiled-program caches, which
never share a program between two configurations that differ in the key alone, and add none where
no softplus is applied.
"""
import functools
from typing import Any, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    DEFAULT_SEED,
    NEUROMODULATION_PRESETS,
    RECURRENCE_PRESETS,
    MultiTaskProblem,
    XORProblem,
    assert_close_across_programs,
    cpu_multi_threaded_eigen,
    create_base_config,
    create_config_with_multi_task,
    exact_backend,
)
from emr_hyperneat import (
    EMRHyperNEAT,
    SubstrateConfig,
    build_recurrent_substrate,
    query_recurrent_substrate,
    substrate_forward,
)
from emr_hyperneat import emrhyperneat as engine
from emr_hyperneat import substrate as api

#: The first float32 at which the published softplus is inf: 88.72283935546875
OVERFLOW_BITS = 0x42B17218
#: The large-input problems' input value: a pre-activation of up to 3 x (2 x 100 + 1) at max_weight
#: 3, far past the overflow
SCALE = 100.0
DEPTH = 2
POP = 10
#: Tried in order until the unguarded generation has a genome the overflow makes non-finite
SEEDS = (DEFAULT_SEED,) + tuple(range(6))
#: XOR's rows: at max_weight 3 a hidden pre-activation is at most 9 there, so with a sigmoid output
#: no softplus of a feed-forward substrate overflows
SMALL_ROWS = np.asarray(XORProblem().inputs, dtype=np.float32)
NT_VECTOR = jnp.array([0.5, 0.5, 0.5, 1.0])


def _bits(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).view(np.uint32)


def _assert_bitwise(actual: Any, desired: Any, message: str = '') -> None:
    np.testing.assert_array_equal(_bits(actual), _bits(desired), err_msg=message)


def _all_float32(low: float, high: float) -> np.ndarray:
    """Every float32 in [low, high), for 0 < low < high."""
    bits = np.arange(np.float32(low).view(np.uint32), np.float32(high).view(np.uint32),
                     dtype=np.uint32)
    return bits.view(np.float32)


def test_the_published_softplus_overflows_from_88_72():
    """Control: the registry's softplus, every float32 of [64, 128), jitted. On the CPU the first
    inf is at 0x42b17218 and every larger input gives inf; off the CPU the threshold lies within an
    ulp of it."""
    x = _all_float32(64.0, 128.0)
    y = np.asarray(jax.jit(engine.ACTIVATION_FUNCTIONS['softplus'])(jnp.asarray(x)))
    first = int(np.flatnonzero(np.isposinf(y))[0])
    assert np.isfinite(y[:first]).all() and np.isposinf(y[first:]).all()
    if exact_backend():
        assert int(x[first].view(np.uint32)) == OVERFLOW_BITS
    else:
        assert abs(int(x[first].view(np.uint32)) - OVERFLOW_BITS) <= 1


def test_the_activation_list_and_the_registry_are_unchanged():
    """Control: the 18 functions and their indices, which palettes, decodes and draws read, and the
    registry's names."""
    assert engine.ACTIVATION_LIST == [
        'tanh', 'sigmoid', 'relu', 'identity', 'sin', 'gauss', 'lelu', 'softplus', 'rs_adapt',
        'fs_fast', 'lts_low', 'burst', 'resonator', 'osc_adapt', 'gain_mod', 'receptive',
        'band_pass', 'integrate']
    assert engine.NUM_ACTIVATIONS == 18 and engine.ACTIVATION_LIST.index('softplus') == 7
    assert engine.PALETTE_CONFIGS['full'] == list(range(18))
    assert sorted(engine.ACTIVATION_FUNCTIONS) == sorted(engine.ACTIVATION_LIST + [
        'abs', 'scaled_tanh', 'softmax'])


def test_the_lookup_gives_the_registry_unless_the_key_names_softplus():
    """activation_function: the registry's own object for every name with the key off, and for every
    name but 'softplus' with it on; the guarded function is not in the registry. An unknown name
    raises the registry's KeyError."""
    guarded = engine.activation_function('softplus', True)
    for name, function in engine.ACTIVATION_FUNCTIONS.items():
        assert engine.activation_function(name) is function, name
        assert engine.activation_function(name, False) is function, name
        if name != 'softplus':
            assert engine.activation_function(name, True) is function, name
    assert guarded is not engine.ACTIVATION_FUNCTIONS['softplus']
    assert all(guarded is not function for function in engine.ACTIVATION_FUNCTIONS.values())
    assert engine.activation_function('softplus', True) is guarded
    with pytest.raises(KeyError):
        engine.activation_function('not_a_function', True)


def test_the_guard_keeps_every_finite_value_and_gives_x_where_exp_overflows():
    """The guarded softplus against the published one, jitted, on every float32 of [1, 2) (where
    jax.nn.softplus moves 1.69M of the 8.39M values), of [64, 128) (where the published form
    overflows) and of (-128, -64], 2**20 uniform values in [-100, 100), and the special values:
    bit for bit wherever the published value is finite, x itself (finite) wherever it is inf at a
    finite x, inf at inf and NaN at NaN as before."""
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, np.finfo(np.float32).max,
                         -np.finfo(np.float32).max, np.finfo(np.float32).tiny], dtype=np.float32)
    x = np.concatenate([
        _all_float32(1.0, 2.0), _all_float32(64.0, 128.0), -_all_float32(64.0, 128.0),
        np.random.default_rng(20261112).uniform(-100.0, 100.0, 2 ** 20).astype(np.float32),
        specials])
    published = np.asarray(jax.jit(engine.ACTIVATION_FUNCTIONS['softplus'])(jnp.asarray(x)))
    guarded = np.asarray(jax.jit(engine.activation_function('softplus', True))(jnp.asarray(x)))
    finite = np.isfinite(published)
    _assert_bitwise(guarded[finite], published[finite], 'a finite value moved')
    overflow = np.isposinf(published) & np.isfinite(x)
    assert overflow.sum() > 5_000_000, overflow.sum()
    _assert_bitwise(guarded[overflow], x[overflow], 'not x where exp(x) overflows')
    assert np.isfinite(guarded[np.isfinite(x)]).all()
    tail = guarded[-len(specials):]
    assert np.isposinf(tail[2]) and tail[3] == 0.0 and np.isnan(tail[4])


def test_the_per_node_forwards_guard_index_7_alone():
    """The grouped forwards, hidden_activation, node_activation and node_activation_rule with the
    key on: the published values everywhere but at index-7 nodes past the overflow, which give
    their pre-activation; with it off (given or not) the published values bit for bit. A palette
    without index 7, or fewer than eight leading functions, applies no softplus, and the key
    changes nothing there."""
    rng = np.random.default_rng(7)
    pre = jnp.asarray(rng.uniform(-150.0, 150.0, (64, 36)).astype(np.float32))
    indices = jnp.asarray(np.arange(36) % 18, dtype=jnp.int32)
    softplus_nodes = np.asarray(indices) == 7
    palette = jnp.asarray([7, 4, 0, 13], dtype=jnp.int32)
    paletted = jnp.asarray(np.array([7, 4, 0, 13] * 9), dtype=jnp.int32)

    def check(off, on, nodes):
        off, on = np.asarray(off), np.asarray(on)
        changed = nodes[None, :] & np.isposinf(off)
        assert changed.any(), 'no softplus node overflows here'
        _assert_bitwise(on[~changed], off[~changed])
        _assert_bitwise(on[changed], np.asarray(pre)[changed])

    grouped = engine.grouped_activation_forward
    full = grouped(pre, indices, 18)
    _assert_bitwise(grouped(pre, indices, 18, guarded_softplus=False), full)
    check(full, grouped(pre, indices, 18, guarded_softplus=True), softplus_nodes)
    seven = grouped(pre, indices, 7)
    _assert_bitwise(grouped(pre, indices, 7, guarded_softplus=True), seven)

    with_palette = engine.grouped_activation_forward_with_palette(pre, paletted, palette)
    _assert_bitwise(engine.grouped_activation_forward_with_palette(
        pre, paletted, palette, guarded_softplus=False), with_palette)
    check(with_palette, engine.grouped_activation_forward_with_palette(
        pre, paletted, palette, guarded_softplus=True), np.asarray(paletted) == 7)
    without = jnp.asarray([4, 0, 13], dtype=jnp.int32)
    _assert_bitwise(engine.grouped_activation_forward_with_palette(
        pre, paletted, without, guarded_softplus=True),
        engine.grouped_activation_forward_with_palette(pre, paletted, without))

    check(engine.hidden_activation(pre, indices, num_activations=18),
          engine.hidden_activation(pre, indices, num_activations=18, guarded_softplus=True),
          softplus_nodes)
    check(engine.hidden_activation(pre, paletted, palette=palette),
          engine.hidden_activation(pre, paletted, palette=palette, guarded_softplus=True),
          np.asarray(paletted) == 7)
    check(engine.node_activation('every_iteration', indices, None, None, 18)(pre),
          engine.node_activation('every_iteration', indices, None, None, 18,
                                 guarded_softplus=True)(pre), softplus_nodes)
    rule_off, operands_off = engine.node_activation_rule('every_iteration', paletted, None,
                                                         palette, 18)
    rule_on, operands_on = engine.node_activation_rule('every_iteration', paletted, None, palette,
                                                       18, guarded_softplus=True)
    check(rule_off(pre, *operands_off), rule_on(pre, *operands_on), np.asarray(paletted) == 7)
    # 'tanh' settling applies tanh whatever the key
    tanh_rule, _ = engine.node_activation_rule('tanh', indices, None, palette, 18,
                                               guarded_softplus=True)
    assert tanh_rule is jnp.tanh


GLOBAL_SOFTPLUS = {'mode': 'global', 'hidden_activation': 'softplus'}
FULL_PALETTE = {'mode': 'cppn_output', 'num_activations': 18, 'palette': 'full'}
TRUE_NEUROMODULATION = NEUROMODULATION_PRESETS['true_neuromodulation_4nt']


def _config(recurrence: str, dynamic_functions: Dict[str, Any],
            neuromodulation: Optional[Dict[str, Any]] = None, output: str = 'sigmoid',
            guarded: Any = None, settling_rule: Optional[str] = None) -> Dict[str, Any]:
    """A depth-2 configuration; ``guarded`` sets dynamic_functions.guarded_softplus unless None,
    ``settling_rule`` recurrence.settling_rule unless None."""
    config = create_base_config(max_depth=DEPTH, population_size=POP)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    recurrence_block = dict(RECURRENCE_PRESETS[recurrence])
    if recurrence != 'feedforward':
        # Per-genome discovery without the h->h cache: what the substrate API builds
        recurrence_block.update(iteration_level=1, hh_cache_enabled=False)
    if settling_rule is not None:
        recurrence_block['settling_rule'] = settling_rule
    emr['recurrence'] = recurrence_block
    emr['dynamic_functions'] = dict(dynamic_functions, output_activation=output)
    if guarded is not None:
        emr['dynamic_functions']['guarded_softplus'] = guarded
    emr['aggregation'] = {'mode': 'disabled'}
    emr['neuromodulation'] = dict(neuromodulation) if neuromodulation else {'enabled': False}
    substrate = config['algorithm_params']['emrhyperneat']['substrate']
    substrate['output_activation'] = output
    if dynamic_functions.get('mode') == 'global':
        substrate['hidden_activation'] = dynamic_functions['hidden_activation']
    return config


def test_the_key_is_a_bool_off_by_default():
    """dynamic_functions.guarded_softplus: False when absent, read as given when true or false, and
    any other value refused; the resolved hidden and output functions and substrate_config follow
    it, under per-node activations too (where the hidden function is per node and the output one
    is resolved alone)."""
    assert 'guarded_softplus' in EMRHyperNEAT.DYNAMIC_FUNCTIONS_KEYS
    published = engine.ACTIVATION_FUNCTIONS['softplus']
    guarded = engine.activation_function('softplus', True)
    for value, function in ((None, published), (False, published), (True, guarded)):
        algo = EMRHyperNEAT()
        algo.create_config(_config('feedforward', GLOBAL_SOFTPLUS, output='softplus',
                                   guarded=value))
        assert algo.df_guarded_softplus is bool(value)
        assert algo._hidden_act_fn is function and algo._output_act_fn is function
        assert algo.substrate_config().guarded_softplus is bool(value)
        algo = EMRHyperNEAT()
        algo.create_config(_config('feedforward', FULL_PALETTE, output='softplus', guarded=value))
        assert algo._hidden_act_fn is None and algo._output_act_fn is function
        assert algo.substrate_config().guarded_softplus is bool(value)
    algo = EMRHyperNEAT()
    algo.create_config(_config('feedforward', {'mode': 'global', 'hidden_activation': 'sin'},
                               guarded=True))
    assert algo._hidden_act_fn is engine.ACTIVATION_FUNCTIONS['sin']
    assert algo._output_act_fn is engine.ACTIVATION_FUNCTIONS['sigmoid']
    for value in (1, 0, 'true', 'false', 0.0):
        with pytest.raises(ValueError, match='guarded_softplus must be true or false'):
            EMRHyperNEAT().create_config(_config('feedforward', GLOBAL_SOFTPLUS, guarded=value))
    # None given explicitly is not the key's absence
    config = _config('feedforward', GLOBAL_SOFTPLUS)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'][
        'guarded_softplus'] = None
    with pytest.raises(ValueError, match='guarded_softplus must be true or false'):
        EMRHyperNEAT().create_config(config)


def test_what_moves_with_the_key_off():
    """With the key off the setting changes no computed value, but these values of the objects move:
    SubstrateConfig has a 42nd field, appended last with its default, so its ``_asdict()``,
    ``len``, ``repr`` and equality (so the hash) against a 0.3.1 tuple of 41 values change, and
    ``SubstrateConfig._make`` needs all 42 values, while a 0.3.1 positional construction still
    builds; substrate_config() returns it; SUBSTRATE_CAPABILITIES names one more capability;
    DYNAMIC_FUNCTIONS_KEYS holds one more key, so the message for an unknown dynamic_functions key
    lists one more setting."""
    assert len(SubstrateConfig._fields) == 42 and SubstrateConfig._fields[-1] == 'guarded_softplus'
    config = _api_config()
    fields = config._asdict()
    assert list(fields)[-1] == 'guarded_softplus' and fields['guarded_softplus'] is False
    assert len(config) == 42 and repr(config).endswith('guarded_softplus=False)')
    published_tuple = tuple(config)[:-1]  # what a 0.3.1 SubstrateConfig of these settings held
    assert config != published_tuple and SubstrateConfig(*published_tuple) == config
    with pytest.raises(TypeError, match='Expected 42 arguments, got 41'):
        SubstrateConfig._make(published_tuple)
    assert SubstrateConfig._make(tuple(config)) == config
    algo = EMRHyperNEAT()
    algo.create_config(_config('feedforward', GLOBAL_SOFTPLUS))
    assert list(algo.substrate_config()._asdict().items())[-1] == ('guarded_softplus', False)
    assert len(api.SUBSTRATE_CAPABILITIES) == 7 and 'guarded_softplus' in api.SUBSTRATE_CAPABILITIES
    assert len(EMRHyperNEAT.DYNAMIC_FUNCTIONS_KEYS) == 20
    unknown = _config('feedforward', GLOBAL_SOFTPLUS)
    unknown['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'][
        'not_a_setting'] = True
    listed = (r"dynamic_functions\.not_a_setting is not a setting; the settings of "
              r"dynamic_functions are \[.*'guarded_softplus'")
    with pytest.raises(ValueError, match=listed):
        EMRHyperNEAT().create_config(unknown)


def test_the_substrate_config_field_is_last_and_off():
    """SubstrateConfig.guarded_softplus, appended after every 0.3.1 field (a configuration written
    without it reads the published softplus), named in SUBSTRATE_CAPABILITIES; validate refuses a
    value other than a bool."""
    assert SubstrateConfig._fields[-1] == 'guarded_softplus'
    assert SubstrateConfig._field_defaults['guarded_softplus'] is False
    assert 'guarded_softplus' in api.SUBSTRATE_CAPABILITIES
    config = _api_config()
    assert config.validate() is config
    assert config._replace(guarded_softplus=True).validate().guarded_softplus is True
    assert config._replace(guarded_softplus=np.bool_(True)).validate().guarded_softplus
    for value in (1, 'yes', None) + NOT_BOOLS:
        replaced = config._replace(guarded_softplus=value)
        with pytest.raises(ValueError, match='guarded_softplus must be True or False'):
            replaced.validate()


class LargeInputXOR(XORProblem):
    """XOR with its two inputs at 0 and SCALE (the bias stays 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs = [[a * SCALE, b * SCALE, 1.0] for a, b, _ in self.inputs]


#: A per-node palette of softplus and sin: about half the nodes of a population apply softplus
#: through the grouped forward (with the 18-function palette a first population's channel decodes to
#: a few middle entries, and on seven seeds no softplus node overflowed on LargeInputXOR)
SOFTPLUS_PALETTE = {'mode': 'cppn_output', 'palette': ['softplus', 'sin']}
#: name -> recurrence preset, dynamic_functions block, neuromodulation block, output activation. The
#: identity output keeps an infinite h @ W2 infinite, where a sigmoid can turn it into 0 or 1. The
#: three neuromodulated per-node cases reach the per-node node rule under true neuromodulation,
#: which the substrate API builds at four sites of its own: the dense feed-forward forward
#: (_neuromodulated_forward), the compiled one (one_neuromodulated), and the recurrent forward's
#: settling activation (activation, the 'emr' rule) and first step (first_activation, the
#: 'drive_once' rule, which the engine takes with true neuromodulation under hh_activation_mode
#: 'initial_only' alone: the configured activation at the first step, tanh after)
CASES = {
    'global': ('feedforward', GLOBAL_SOFTPLUS, None, 'softplus'),
    'palette': ('feedforward', SOFTPLUS_PALETTE, None, 'identity'),
    'neuromodulated': ('feedforward', GLOBAL_SOFTPLUS, TRUE_NEUROMODULATION, 'sigmoid'),
    'neuromodulated_palette': ('feedforward', SOFTPLUS_PALETTE, TRUE_NEUROMODULATION, 'identity'),
    'recurrent': ('hidden_only', GLOBAL_SOFTPLUS, None, 'sigmoid'),
    'recurrent_palette': ('hidden_only', SOFTPLUS_PALETTE, None, 'sigmoid'),
    'recurrent_palette_drive_once': ('hidden_only', SOFTPLUS_PALETTE, None, 'sigmoid'),
    'recurrent_neuromodulated_palette': ('hidden_only', SOFTPLUS_PALETTE, TRUE_NEUROMODULATION,
                                         'sigmoid'),
    'recurrent_neuromodulated_palette_drive_once': (
        'hidden_only', dict(SOFTPLUS_PALETTE, hh_activation_mode='initial_only'),
        TRUE_NEUROMODULATION, 'sigmoid'),
}
#: The cases settled by the prototype's rule, 'drive_once', whose first step applies the configured
#: activation through a path of its own (first_activation in the loop, the first step's partial in
#: the API, and first_activation in its neuromodulated forward)
SETTLING_RULES = {'recurrent_palette_drive_once': 'drive_once',
                  'recurrent_neuromodulated_palette_drive_once': 'drive_once'}
FEEDFORWARD = ('global', 'palette', 'neuromodulated', 'neuromodulated_palette')
RECURRENT = ('recurrent', 'recurrent_palette', 'recurrent_palette_drive_once',
             'recurrent_neuromodulated_palette', 'recurrent_neuromodulated_palette_drive_once')


class Generation(NamedTuple):
    algo: Any
    state: Any  # the state the generation evaluated
    fitness: np.ndarray  # the generation's fitness after the NaN sanitizer (_cached_fitnesses)
    nan_count: int


@functools.lru_cache(maxsize=None)
def _generation(case: str, guarded: bool, seed: int, large: bool = True) -> Generation:
    """One generation of a CASES configuration on LargeInputXOR (XOR when not ``large``)."""
    recurrence, dynamic_functions, neuromodulation, output = CASES[case]
    config = _config(recurrence, dynamic_functions, neuromodulation, output,
                     guarded=True if guarded else None, settling_rule=SETTLING_RULES.get(case))
    algo = EMRHyperNEAT()
    problem = LargeInputXOR() if large else XORProblem()
    state = algo.initialize(algo.create_config(config), problem, seed=seed)
    _, metrics = algo.run_generation(state, problem, verbose=False)
    return Generation(algo, state, np.asarray(algo._cached_fitnesses),
                      int(metrics.custom_metrics['nan_fitness_count']))


@functools.lru_cache(maxsize=None)
def _overflowing_seed(case: str) -> int:
    """The first of SEEDS whose unguarded generation has a genome of non-finite fitness."""
    for seed in SEEDS:
        if not np.isfinite(_generation(case, False, seed).fitness).all():
            return seed
    pytest.fail(f'{case}: no seed of {SEEDS} overflows a softplus on LargeInputXOR')


@pytest.mark.parametrize('case', sorted(CASES))
def test_a_generation_applies_the_guard(case):
    """On LargeInputXOR the unguarded generation has genomes the overflow makes NaN (recorded at
    -inf), and the same population with the key on has none: every fitness is finite."""
    seed = _overflowing_seed(case)
    off, on = _generation(case, False, seed), _generation(case, True, seed)
    assert not off.algo.df_guarded_softplus and on.algo.df_guarded_softplus
    assert off.nan_count > 0 or not np.isfinite(off.fitness).all()
    assert on.nan_count == 0 and np.isfinite(on.fitness).all(), on.fitness


@pytest.mark.parametrize('case', ['global', 'palette'])
def test_where_nothing_overflows_the_generation_does_not_move(case):
    """On XOR with a sigmoid output no softplus of a feed-forward substrate reaches the overflow
    (a hidden pre-activation is at most 9), so the key changes no fitness beyond what two compiled
    programs round apart (CPU_CROSS_PROGRAM_ULPS; the two settings evaluate through two programs,
    test_the_recurrent_forward_applies_the_guard): bit for bit on an Apple M4 Max under
    single-threaded Eigen."""
    recurrence, dynamic_functions, _, _ = CASES[case]
    fitness = {}
    for guarded in (False, True):
        config = _config(recurrence, dynamic_functions, None, 'sigmoid',
                         guarded=True if guarded else None)
        algo = EMRHyperNEAT()
        problem = XORProblem()
        state = algo.initialize(algo.create_config(config), problem, seed=DEFAULT_SEED)
        algo.run_generation(state, problem, verbose=False)
        fitness[guarded] = np.asarray(algo._cached_fitnesses)
    assert np.isfinite(fitness[False]).all()
    assert_close_across_programs(fitness[True], fitness[False])


@pytest.mark.parametrize('case', FEEDFORWARD)
def test_the_substrate_api_reproduces_the_guarded_generation(case):
    """substrate_config carries the key, and the dense forward of the guarded generation's
    population reproduces its fitness bit for bit, as without the key
    (test_substrate_api.py)."""
    on = _generation(case, True, _overflowing_seed(case))
    substrate = on.algo.population_substrates(on.state)
    assert substrate.config.guarded_softplus is True
    outputs = substrate_forward(substrate, on.algo._cached_inputs)
    fitness = jax.vmap(lambda out: engine._eval_single_fitness(
        out, on.algo._cached_targets, on.algo.fitness_mode))(outputs)
    _assert_bitwise(fitness, on.fitness)


def _unguarded(substrate: Any) -> Any:
    return substrate._replace(config=substrate.config._replace(guarded_softplus=False))


def _with_output(substrate: Any, function: str) -> Any:
    return substrate._replace(config=substrate.config._replace(output_activation=function))


#: The feed-forward cases with a global hidden activation. Under a relu hidden layer, which does not
#: overflow, a softplus output passes the overflow at rows of SCALE through the lookup of the output
#: function alone, which each forward makes for itself: the dense forward, the dense neuromodulated
#: one (_neuromodulated_forward) and the compiled one (shared by one and one_neuromodulated)
OUTPUT_CASES = ('global', 'neuromodulated')


@pytest.mark.parametrize('case', FEEDFORWARD)
def test_the_feedforward_forwards_apply_the_guard(case):
    """The dense and the compiled forward (and under neuromodulation its node rule): on XOR's rows
    under a sigmoid output nothing overflows and the key changes no output, bit for bit (the dense
    forward runs op by op, the same operations either way; the compiled forward's two settings are
    two programs, bit for bit on an Apple M4 Max under single-threaded Eigen, held within
    CPU_CROSS_PROGRAM_ULPS as two programs are); on rows of SCALE under the identity output (which
    keeps an infinite h @ W2 infinite) the unguarded forwards give non-finite outputs and the
    guarded ones none. In OUTPUT_CASES a softplus output under a relu hidden layer does the same
    through each forward's output lookup, dense and compiled, with and without neuromodulation; the
    dense forward without neuromodulation computes h @ W2 op by op, so its guarded output is h @ W2
    itself, bit for bit, where the published one is inf, and the published value elsewhere."""
    on = _generation(case, True, _overflowing_seed(case))
    population = on.algo.population_substrates(on.state)
    small, guarded = _with_output(population, 'sigmoid'), _with_output(population, 'identity')
    large = np.asarray(LargeInputXOR().inputs, dtype=np.float32)
    _assert_bitwise(substrate_forward(small, SMALL_ROWS),
                    substrate_forward(_unguarded(small), SMALL_ROWS), 'dense')
    assert_close_across_programs(substrate_forward(small, SMALL_ROWS, compiled=True),
                                 substrate_forward(_unguarded(small), SMALL_ROWS, compiled=True))
    for compiled in (False, True):
        assert not np.isfinite(np.asarray(substrate_forward(
            _unguarded(guarded), large, compiled=compiled))).all(), f'compiled={compiled}'
        assert np.isfinite(np.asarray(substrate_forward(guarded, large, compiled=compiled))).all()
    if case not in OUTPUT_CASES:
        return
    relu = guarded._replace(config=guarded.config._replace(hidden_activation='relu'))
    softplus = _with_output(relu, 'softplus')
    outputs = {}
    for compiled in (False, True):
        published = np.asarray(substrate_forward(_unguarded(softplus), large, compiled=compiled))
        guarded_out = np.asarray(substrate_forward(softplus, large, compiled=compiled))
        assert not np.isfinite(published).all(), f'compiled={compiled}: nothing overflows'
        assert np.isfinite(guarded_out).all(), f'compiled={compiled}'
        outputs[compiled] = published, guarded_out
    if case == 'global':
        # The identity output's values are h @ W2, the softplus output's argument
        published, guarded_out = outputs[False]
        pre_output = np.asarray(substrate_forward(_with_output(relu, 'identity'), large))
        overflow = np.isposinf(published)
        assert overflow.any() and not np.isnan(published).any()
        _assert_bitwise(guarded_out[overflow], pre_output[overflow])
        _assert_bitwise(guarded_out[~overflow], published[~overflow])


@functools.lru_cache(maxsize=None)
def _recurrent_substrate(case: str) -> Any:
    """The guarded generation's population with its hidden-to-hidden connections (the API's)."""
    on = _generation(case, True, _overflowing_seed(case))
    algo, state = on.algo, on.state
    transformed = algo._compiled_transform_batch(state, algo._compiled_ask(state))
    population = algo.population_substrates(state)
    return build_recurrent_substrate(population, query_recurrent_substrate(
        algo._jitted_cppn_forward, state, transformed, population))


@pytest.mark.parametrize('case', RECURRENT)
def test_the_recurrent_forward_applies_the_guard(case):
    """The recurrent forward settles every step through the guarded softplus. A genome none of whose
    settled states overflows without the key has the same states with it, bit for bit under
    single-threaded Eigen on the CPU (the threaded matmul may round them apart, as the suite
    allows), and outputs within CPU_CROSS_PROGRAM_ULPS: the two settings are two compiled programs,
    which can round the output product apart. Under the 'drive_once' rule with per-node activations
    and without neuromodulation, on an Apple M4 Max under single-threaded Eigen, the two programs'
    outputs were up to 28 ulps apart on XOR's rows, the states bit for bit. A genome some state of
    which overflows without the key has finite states and outputs with it. Under the 'drive_once'
    rule off the CPU, whose scatter-add accumulates in no fixed order, a genome that does not
    overflow has states and outputs of the same shape and finite where they were (below). XOR's rows
    leave some genomes below the overflow (the settling can still reach it), rows of SCALE take some
    past it. For the global case, a softplus output under a relu settling, which does not overflow,
    isolates the output lookup that the forwards without and with neuromodulation and the completed
    one share: at rows of SCALE the states are finite either way, the unguarded outputs are not, the
    guarded ones are."""
    guarded = _recurrent_substrate(case)
    assert guarded.config.hidden_to_hidden and guarded.config.guarded_softplus
    large = np.asarray(LargeInputXOR().inputs, dtype=np.float32)
    if case == 'recurrent':
        softplus = guarded._replace(config=guarded.config._replace(
            hidden_activation='relu', output_activation='softplus'))
        out_off, states_off = substrate_forward(_unguarded(softplus), large, return_states=True)
        out_on, states_on = substrate_forward(softplus, large, return_states=True)
        assert all(np.isfinite(np.asarray(s)).all() for s in [*states_off, *states_on])
        assert not np.isfinite(np.asarray(out_off)).all(), 'no softplus output overflows'
        assert np.isfinite(np.asarray(out_on)).all()
    # On a GPU a scatter-add accumulates in no fixed order and the 'drive_once' rule's recurrent
    # input is one at every step, so a settling carries a run's rounding through the states: an
    # NVIDIA T4 (jax 0.6.1, JAX_DEFAULT_MATMUL_PRECISION=highest) put the two settings' states of
    # the palette case up to 8.8e-4 apart, as a repeat of one program moves them there
    # (test_substrate_recurrent_neuromodulation.py, GPU_SCATTER_ATOL). There the states and outputs
    # of the two settings are compared by shape and finiteness; the CPU compares their values
    gpu_drive_once = guarded.config.settling_rule == 'drive_once' and not exact_backend()
    flags = {}
    for label, rows in (('small', SMALL_ROWS), ('large', large)):
        out_off, states_off = substrate_forward(_unguarded(guarded), rows, return_states=True)
        out_on, states_on = substrate_forward(guarded, rows, return_states=True)
        out_off, out_on = np.asarray(out_off), np.asarray(out_on)
        flags[label] = [not np.isfinite(np.asarray(s)).all() for s in states_off]
        for genome, overflow in enumerate(flags[label]):
            if overflow:
                assert np.isfinite(np.asarray(states_on[genome])).all(), (label, genome)
                assert np.isfinite(out_on[genome]).all(), (label, genome)
                continue
            if gpu_drive_once:
                assert np.asarray(states_on[genome]).shape == np.asarray(states_off[genome]).shape
                assert np.isfinite(np.asarray(states_on[genome])).all(), (label, genome)
                assert out_on[genome].shape == out_off[genome].shape
                np.testing.assert_array_equal(np.isfinite(out_on[genome]),
                                              np.isfinite(out_off[genome]))
                continue
            if exact_backend() and not cpu_multi_threaded_eigen():
                _assert_bitwise(states_on[genome], states_off[genome], f'{label} genome {genome}')
            else:
                assert_close_across_programs(states_on[genome], states_off[genome])
            assert_close_across_programs(out_on[genome], out_off[genome])
    assert not all(flags['small']) and any(flags['large']), flags


#: Values of SubstrateConfig.guarded_softplus that are not a bool. Read by their truth, as a
#: forward that does not validate would read the field without its check, the first two would run
#: the guarded softplus and the last two the published one
NOT_BOOLS = ('false', 'no', 0, None)


def _population(case: str) -> Any:
    """The guarded generation's population of a feed-forward case (the API's substrate)."""
    on = _generation(case, True, _overflowing_seed(case))
    return on.algo.population_substrates(on.state)


def _with_guard(substrate: Any, value: Any) -> Any:
    return substrate._replace(config=substrate.config._replace(guarded_softplus=value))


def test_a_value_other_than_a_bool_is_refused_on_every_path():
    """Every forward reads SubstrateConfig.guarded_softplus through one check, as validate does:
    the feed-forward forwards and the plain recurrent one do not validate (0.3.0's behavior), and a
    value other than a bool raises instead of being read by its truth. The dense and the compiled
    feed-forward forward, with and without neuromodulation, refuse it whether or not the
    configuration applies a softplus (the compiled forward's key leaves the setting out where none
    is applied); the recurrent forward, plain and neuromodulated, refuses it too. numpy's bool is
    read as Python's, bit for bit."""
    feedforward = _population('palette')
    neuromodulated = _population('neuromodulated_palette')
    global_population = _population('global')
    no_softplus = global_population._replace(config=global_population.config._replace(
        hidden_activation='tanh', output_activation='sigmoid'))

    def dense(substrate):
        return substrate_forward(substrate, SMALL_ROWS)

    def compiled(substrate):
        return substrate_forward(substrate, SMALL_ROWS, compiled=True)

    paths = (('dense', feedforward, dense), ('compiled', feedforward, compiled),
             ('dense without softplus', no_softplus, dense),
             ('compiled without softplus', no_softplus, compiled),
             ('dense neuromodulated', neuromodulated, dense),
             ('compiled neuromodulated', neuromodulated, compiled),
             ('recurrent', _recurrent_substrate('recurrent_palette'), dense),
             ('recurrent neuromodulated', _recurrent_substrate('recurrent_neuromodulated_palette'),
              dense))
    for value in NOT_BOOLS:
        for label, substrate, forward in paths:
            refused = _with_guard(substrate, value)
            with pytest.raises(ValueError, match='guarded_softplus must be True or False'):
                forward(refused)
                pytest.fail(f'{label}: guarded_softplus={value!r} ran')
    for substrate in (feedforward, neuromodulated):
        for forward in (dense, compiled):
            _assert_bitwise(forward(_with_guard(substrate, np.bool_(True))), forward(substrate))


def test_evaluate_with_neurotransmitter_applies_the_guard_to_softplus():
    """activation_name 'softplus' resolves through the key: after the same neuromodulated
    generation, the unguarded evaluation on LargeInputXOR's rows has non-finite fitness and the
    guarded one none; under 'tanh' the key changes nothing, bit for bit."""
    seed = _overflowing_seed('neuromodulated')
    off, on = _generation('neuromodulated', False, seed), _generation('neuromodulated', True, seed)
    problem = LargeInputXOR()
    rows = jnp.asarray(problem.inputs, dtype=jnp.float32)
    targets = jnp.asarray(problem.targets, dtype=jnp.float32)
    fitness = {g.algo.df_guarded_softplus: np.asarray(g.algo.evaluate_with_neurotransmitter(
        NT_VECTOR, rows, targets, activation_name='softplus')) for g in (off, on)}
    assert not np.isfinite(fitness[False]).all() and np.isfinite(fitness[True]).all()
    _assert_bitwise(
        on.algo.evaluate_with_neurotransmitter(NT_VECTOR, rows, targets, activation_name='tanh'),
        off.algo.evaluate_with_neurotransmitter(NT_VECTOR, rows, targets, activation_name='tanh'))


class LargeInputTask(MultiTaskProblem):
    """A logic-gate task with its inputs at 0 and SCALE."""

    def __init__(self, task_name: str) -> None:
        super().__init__(task_name)
        self.inputs = self.inputs * SCALE


def test_a_multi_task_generation_applies_the_guard():
    """multitask.hidden_activation 'softplus' resolves through the key: two tasks on inputs of
    SCALE, NaN fitness without it (counted and recorded at -inf), none with it."""
    aggregated = {}
    for guarded in (False, True):
        config = create_config_with_multi_task(['xor', 'and'], recurrence_preset='feedforward',
                                               max_depth=DEPTH, population_size=POP)
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['multitask']['hidden_activation'] = 'softplus'
        if guarded:
            emr.setdefault('dynamic_functions', {})['guarded_softplus'] = True
        algo = EMRHyperNEAT()
        problems = [LargeInputTask('xor'), LargeInputTask('and')]
        state = algo.initialize(algo.create_config(config), problems[0], seed=DEFAULT_SEED)
        _, metrics = algo.run_generation_multitask(state, problems=problems,
                                                   neurotransmitters=[NT_VECTOR, NT_VECTOR])
        aggregated[guarded] = (np.asarray(algo._last_multitask['aggregated']),
                               int(metrics.custom_metrics['nan_fitness_count']))
    assert aggregated[False][1] > 0 and not np.isfinite(aggregated[False][0]).all()
    assert aggregated[True][1] == 0 and np.isfinite(aggregated[True][0]).all()


def _api_config(**fields: Any) -> SubstrateConfig:
    """A substrate configuration under which every softplus path reads the key: the 18-function
    palette, a softplus hidden and output activation."""
    return SubstrateConfig(input_coords=((-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)),
                           output_coords=((0.0, 1.0),), max_depth=DEPTH, num_cppn_outputs=2,
                           activation_mode='cppn_output', palette=tuple(range(18)),
                           num_activations=18, hidden_activation='softplus',
                           output_activation='softplus', **fields)


def test_two_configurations_differing_only_by_the_key_never_share_a_compiled_program():
    """Every cache that keeps a compiled program, or the function one is made from, keys the guard:
    the compiled feed-forward forward (per combination of the settings it reads, with and without
    neuromodulation), the recurrent forward (plain, per node, neuromodulated, the whole grid's
    softmax), the per-node node rule (and so its settling steps) and the multi-task evaluators. A
    repeat with the key on is the entry the first call made."""
    config = _api_config()
    guarded = config._replace(guarded_softplus=True)
    settings = api.NeuromodulationSettings()
    for has_act, has_agg in ((False, False), (True, False), (False, True), (True, True)):
        for neuromodulation in (None, settings):
            args = (has_act, has_agg, neuromodulation)
            plain = api._feedforward_forward_fn(config, *args)
            assert api._feedforward_forward_fn(guarded, *args) is not plain, args
            assert api._feedforward_forward_fn(guarded, *args) is api._feedforward_forward_fn(
                guarded, *args), args
    recurrent = config._replace(hidden_to_hidden=True)
    recurrent_guarded = recurrent._replace(guarded_softplus=True)
    for has_act in (False, True):
        for neuromodulation in (None, settings):
            for complete in (False, True):
                args = (16, has_act, False, False, neuromodulation)
                plain = api._recurrent_forward_fn(recurrent, *args, complete=complete)
                on = api._recurrent_forward_fn(recurrent_guarded, *args, complete=complete)
                assert on is not plain, (has_act, neuromodulation, complete)
                assert api._recurrent_forward_fn(recurrent_guarded, *args, complete=complete) is on
            # The guarded branch routes `complete` too: one program per value, as unguarded
            assert (api._recurrent_forward_fn(recurrent_guarded, *args, complete=True)
                    is not api._recurrent_forward_fn(recurrent_guarded, *args, complete=False)), (
                has_act, neuromodulation)
    indices = jnp.asarray(np.arange(16) % 18, dtype=jnp.int32)
    entries = tuple(range(18))
    rule_off, _ = engine.node_activation_rule('every_iteration', indices, None, entries, 18)
    rule_on, _ = engine.node_activation_rule('every_iteration', indices, None, entries, 18,
                                             guarded_softplus=True)
    assert rule_on is not rule_off and engine._is_stable_node_fn(rule_on)
    assert engine.node_activation_rule('every_iteration', indices, None, entries, 18,
                                       guarded_softplus=True)[0] is rule_on
    # The key the unguarded rule is kept under is 0.3.1's
    assert engine._NODE_RULES[('activation', entries, 18)] is rule_off
    whole_on, _ = engine.node_activation_rule('every_iteration', None,
                                              engine.activation_function('softplus', True))
    assert whole_on is engine.activation_function('softplus', True)
    assert engine._is_stable_node_fn(whole_on)
    algo = EMRHyperNEAT()
    algo.create_config(_config('feedforward', GLOBAL_SOFTPLUS, TRUE_NEUROMODULATION))
    make = algo._create_eval_population_true_neuromodulated
    published = make(hidden_act_fn=engine.ACTIVATION_FUNCTIONS['softplus'])
    with_guard = make(hidden_act_fn=engine.activation_function('softplus', True))
    assert with_guard is not published
    assert make(hidden_act_fn=engine.activation_function('softplus', True)) is with_guard


def test_the_key_compiles_nothing_new_where_no_softplus_is_applied():
    """Where a configuration applies no softplus, the compiled feed-forward forward and the per-node
    node rule with the key on are the ones without it (the key is in their cache keys only where a
    softplus is applied, as _feedforward_key leaves out every setting a combination does not read):
    a palette without index 7, fewer than eight leading functions, a hidden activation other than
    softplus without per-node activations (with them it is unread), an output other than softplus.
    One more function of the palette, or softplus as the output, puts it back in."""
    fn = api._feedforward_forward_fn
    plain = _api_config()._replace(palette=(0, 4, 13), hidden_activation='tanh',
                                   output_activation='sigmoid')
    settings = api.NeuromodulationSettings()

    def guarded(config):
        return config._replace(guarded_softplus=True)

    for has_act, has_agg in ((False, False), (True, False), (False, True), (True, True)):
        for neuromodulation in (None, settings):
            args = (has_act, has_agg, neuromodulation)
            assert fn(guarded(plain), *args) is fn(plain, *args), args
    seven = plain._replace(palette=None, num_activations=7)
    assert fn(guarded(seven), True, False) is fn(seven, True, False)
    eight = seven._replace(num_activations=8)
    assert fn(guarded(eight), True, False) is not fn(eight, True, False)
    with_seven = plain._replace(palette=(0, 4, 7))
    assert fn(guarded(with_seven), True, False) is not fn(with_seven, True, False)
    hidden = plain._replace(hidden_activation='softplus')
    assert fn(guarded(hidden), True, False) is fn(hidden, True, False)
    assert fn(guarded(hidden), False, False) is not fn(hidden, False, False)
    output = plain._replace(output_activation='softplus')
    assert fn(guarded(output), True, False) is not fn(output, True, False)
    indices = jnp.asarray(np.arange(16) % 18, dtype=jnp.int32)
    for entries, count in (((0, 4, 13), 18), (None, 7)):
        rule_off, _ = engine.node_activation_rule('every_iteration', indices, None, entries, count)
        rule_on, _ = engine.node_activation_rule('every_iteration', indices, None, entries, count,
                                                 guarded_softplus=True)
        assert rule_on is rule_off, (entries, count)
    rule_off, _ = engine.node_activation_rule('every_iteration', indices, None, None, 8)
    rule_on, _ = engine.node_activation_rule('every_iteration', indices, None, None, 8,
                                             guarded_softplus=True)
    assert rule_on is not rule_off
