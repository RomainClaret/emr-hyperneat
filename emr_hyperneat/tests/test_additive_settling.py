"""HMR FULL's neuromodulated recurrence under recurrence.settling_rule 'hmr'.

FULL (HMR's full neuromodulation variant, the rule HMR's neuromodulation benchmark ran) settled a
neuromodulated recurrent hidden layer additively: the node of the input drive, then activate_time
steps of h = h + update(recurrent(h)). Single-task, the update was tanh(gates * r) under static and
context gating and the node rule with tanh under true neuromodulation (FULL's own gains and gate);
on the multi-task path the update was the node itself, with the task's activation (the paper's
rule). EMR runs it under settling_rule 'hmr' with neuromodulation (0.1.0 had no settling_rule). The
multi-task evaluator is checked against FULL's own on the same arrays, as recorded in the research
repository (hmr_reference.py); the single-task updates against FULL's loop as written. EMR's
settling rule is the negative control.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (
    DEFAULT_SEED,
    MultiTaskProblem,
    EMRHyperNEAT,
    XORProblem,
    create_config_with_multi_task,
    create_config_with_neuromodulation,
    exact_backend,
)
from emr_hyperneat.emrhyperneat import (
    _neuromodulated_rule,
    settle_hidden_additive,
    true_neuromodulation_node_rule,
)
from hmr_reference import hmr_value, live  # type: ignore[import-not-found]

STEPS = 5


def _network(seed, pop=None, positions=12, cases=4, inputs=2, conns=30, num_nt=4):
    keys = jax.random.split(jax.random.PRNGKey(seed), 9)
    lead = () if pop is None else (pop,)
    return dict(
        X=jax.random.uniform(keys[0], (cases, inputs)),
        Y=(jax.random.uniform(keys[1], (cases, 1)) > 0.5).astype(jnp.float32),
        W1=jax.random.normal(keys[2], lead + (inputs, positions)),
        W2=jax.random.normal(keys[3], lead + (positions, 1)),
        rd=jax.random.uniform(keys[4], lead + (positions, num_nt), minval=-1.0, maxval=1.0),
        bg=jax.random.uniform(keys[5], lead + (positions,), minval=0.5, maxval=1.0),
        frm=jax.random.randint(keys[6], lead + (conns,), 0, positions),
        to=jax.random.randint(keys[7], lead + (conns,), 0, positions),
        w=jax.random.normal(keys[8], lead + (conns,)),
        valid=jnp.broadcast_to(jnp.arange(conns) % 4 != 0, lead + (conns,)),
    )


def _fulls_loop(first, net, update):
    """FULL's loop as written (FULL:5365-5373 and 5629-5644): HMR's scatter-add, then h = h +
    update(r). Compiled, as FULL's evaluators ran it: XLA contracts h + t * gates to a fused
    multiply-add there, which an op-by-op run rounds otherwise (one ulp)."""
    n = first.shape[1]
    safe_from, safe_to = jnp.clip(net['frm'], 0, n - 1), jnp.clip(net['to'], 0, n - 1)
    weights = jnp.where(net['valid'], net['w'], 0.0)

    def loop(h):
        for _ in range(STEPS):
            contrib = jnp.zeros_like(h).at[:, safe_to].add(h[:, safe_from] * weights)
            h = h + update(contrib)
        return h

    return jax.jit(loop)(first)


def _agree(ours, theirs):
    if exact_backend():
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    else:
        np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-5, atol=1e-6)


class TestTheRule:
    def test_gating(self):
        """Static and context gating: tanh(gates * r), FULL:5365-5373."""
        net = _network(1)
        gates = jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(2), (12,)))
        first = jnp.tanh(net['X'] @ net['W1']) * gates
        ours = settle_hidden_additive(first, net['frm'], net['to'], net['w'], net['valid'], STEPS,
                                      _neuromodulated_rule('gain', jnp.tanh), (gates,))
        _agree(ours, _fulls_loop(first, net, lambda r: jnp.tanh(r * gates)))

    @pytest.mark.parametrize('modulation_mode', ['full', 'gating_only', 'gain_bias_only'])
    def test_true_neuromodulation(self, modulation_mode):
        """FULL's single-task rule (FULL:5565-5644): gains bg + s m, gate sigmoid(5 m), tanh in the
        update."""
        net = _network(3)
        nt = jnp.array([0.9, 0.1, 0.5, 1.0])
        m = net['rd'][:, :3] @ nt[:3]
        gains, gates = net['bg'] + 5.0 * m, jax.nn.sigmoid(5.0 * m)
        update = {
            'full': lambda r: jnp.tanh(gains * r) * gates,
            'gating_only': lambda r: jnp.tanh(r) * gates,
            'gain_bias_only': lambda r: jnp.tanh(gains * r),
        }[modulation_mode]
        first_rule, first_ops = true_neuromodulation_node_rule(
            (jnp.tanh, ()), net['rd'], net['bg'], nt, 5.0, 4, modulation_mode, 'hmr_full'
        )
        first = first_rule(net['X'] @ net['W1'], *first_ops)
        ours = settle_hidden_additive(
            first, net['frm'], net['to'], net['w'], net['valid'], STEPS, first_rule, first_ops
        )
        _agree(ours, _fulls_loop(first, net, update))

    def test_zero_steps_is_the_first_value(self):
        net = _network(4)
        first = jnp.tanh(net['X'] @ net['W1'])
        np.testing.assert_array_equal(np.asarray(settle_hidden_additive(
            first, net['frm'], net['to'], net['w'], net['valid'], 0, jnp.tanh)), np.asarray(first))


def _full_config():
    return {
        'algorithm_params': {
            'hmrhyperneat': {
                'population_size': 6,
                'substrate': {
                    'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                    'output_coords': [[0.0, 1.0]],
                },
                'hmr_hyperneat': {
                    'initial_depth': 0,
                    'max_depth': 2,
                    'variance_threshold': 0.03,
                    'recurrence': {'preset': 'hidden_only'},
                    'neuromodulation': {
                        'mode': 'true_neuromodulation_4nt_option_a_tanh',
                        'modulation_strength': 5.0,
                    },
                },
            }
        }
    }


def _emr_config(settling_rule):
    params = dict(_full_config()['algorithm_params'])
    block = dict(params.pop('hmrhyperneat'))
    emr = dict(block.pop('hmr_hyperneat'), variance_rule='legacy')
    emr['recurrence'] = {
        'preset': 'hidden_only',
        'iteration_level': 1,
        'settling_rule': settling_rule,
    }
    if settling_rule == 'hmr':
        emr['dynamic_functions'] = {'hh_activation_mode': 'initial_only'}
    block['emr_hyperneat'] = emr
    params['emrhyperneat'] = block
    return {'algorithm_params': params}


def _evaluate(evaluator, net, nt):
    return np.asarray(evaluator(net['W1'], net['W2'], net['rd'], net['bg'], nt, net['X'], net['Y'],
                                net['frm'], net['to'], net['w'], net['valid'], 'mse', 'full', None))


class TestMultiTaskAgainstLiveFull:
    """FULL's multi-task evaluator (its _create_eval_population_true_neuromodulated with h->h)
    against EMR's on the same arrays: the paper's rule as node and update, the output inverted by
    the fourth channel."""

    @pytest.fixture(scope='class')
    def fulls(self):
        """FULL's evaluator, called with the arguments of EMR's (hmr_reference.py)."""
        def evaluator(*args):
            return hmr_value(
                'additive_settling.fulls',
                lambda: live('full_multitask_fitness', _full_config(), STEPS, *args),
                inputs=(_full_config(), STEPS, args),
            )
        return evaluator

    @staticmethod
    def _ours(settling_rule):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(_emr_config(settling_rule))
        return algorithm._create_eval_population_true_neuromodulated(
            has_hh=True, activate_time=STEPS, hidden_act_fn=jnp.tanh
        )

    @pytest.mark.parametrize('nt', [[0.9, 0.1, 0.5, 1.0], [0.2, 0.8, 0.3, 0.0]])
    def test_equal_on_the_same_arrays(self, fulls, nt):
        net = _network(5, pop=6)
        nt = jnp.array(nt)
        np.testing.assert_allclose(
            _evaluate(self._ours('hmr'), net, nt), _evaluate(fulls, net, nt), rtol=0, atol=1e-6
        )

    def test_negative_control_emrs_rule(self, fulls):
        net = _network(5, pop=6)
        nt = jnp.array([0.9, 0.1, 0.5, 1.0])
        assert (
            float(np.max(np.abs(_evaluate(self._ours('emr'), net, nt) - _evaluate(fulls, net, nt))))
            > 1e-3
        )


def _neuromodulated(mode, **recurrence):
    config = create_config_with_neuromodulation(mode, recurrence_preset='hidden_only', max_depth=2,
                                                population_size=10)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['recurrence'].update(settling_rule='hmr', hh_cache_enabled=False, **recurrence)
    emr.setdefault('dynamic_functions', {})['hh_activation_mode'] = 'initial_only'
    return config


class TestGenerations:

    @pytest.mark.parametrize(
        'mode', ['static_gating', 'context_gating', 'true_neuromodulation_4nt']
    )
    def test_a_generation_and_its_champion(self, mode):
        """The rule runs in a generation, and evaluate_on_data reproduces the champion's fitness."""
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(_neuromodulated(mode)), XORProblem(), seed=DEFAULT_SEED
        )
        state, _ = algorithm.run_generation(state, XORProblem())
        result = algorithm.evaluate_on_data(
            state, algorithm._cached_inputs, algorithm._cached_targets
        )
        np.testing.assert_allclose(
            float(result['fitness']), algorithm._last_eval['champion_fitness'], rtol=0, atol=1e-6
        )

    def test_it_is_another_network(self):
        """Negative control: the same configuration under EMR's settling rule scores otherwise."""
        def fitness(rule):
            config = _neuromodulated('static_gating')
            config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'][
                'settling_rule'
            ] = rule
            if rule == 'emr':
                config['algorithm_params']['emrhyperneat']['emr_hyperneat'][
                    'dynamic_functions'
                ].pop('hh_activation_mode')
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(
                algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED
            )
            algorithm.run_generation(state, XORProblem())
            return np.asarray(algorithm._cached_fitnesses)
        assert float(np.max(np.abs(fitness('hmr') - fitness('emr')))) > 1e-4


class TestRefusals:
    def test_modulatory_neurons(self):
        with pytest.raises(ValueError, match="'hmr' with modulatory neurons is refused"):
            EMRHyperNEAT().create_config(_neuromodulated('modulatory_neurons'))

    def test_per_node_hh_aggregation(self):
        config = _neuromodulated('static_gating')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['aggregation'] = {
            'mode': 'cppn_output', 'hh_aggregation_mode': 'dynamic'}
        with pytest.raises(
            ValueError, match='per-node hidden-to-hidden aggregation would be ignored'
        ):
            EMRHyperNEAT().create_config(config)

    def test_every_iteration_with_per_node_functions(self):
        config = _neuromodulated('static_gating')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output', 'num_activations': 4, 'hh_activation_mode': 'every_iteration'}
        with pytest.raises(ValueError, match="applies tanh at every step after the first"):
            EMRHyperNEAT().create_config(config)


class TestMultiTaskGeneration:
    """run_generation_multitask under settling_rule 'hmr' (it raised before: the multi-task check
    counted the rule among the features its evaluation would drop) scores each task with FULL's
    multi-task rule."""

    @staticmethod
    def _run(settling_rule):
        from emr_hyperneat.emrhyperneat \
            import get_nt_for_task
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            fitness_aggregation='mean',
            recurrence_preset='hidden_only',
            population_size=12,
            max_depth=2,
        )
        recurrence = config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']
        recurrence.update(settling_rule=settling_rule, hh_cache_enabled=False)
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(config), MultiTaskProblem('xor'), seed=DEFAULT_SEED
        )
        problems = [MultiTaskProblem('xor'), MultiTaskProblem('and')]
        nts = [get_nt_for_task(p.name, 4) for p in problems]
        algorithm.run_generation_multitask(
            state, problems=problems, neurotransmitters=nts, aggregation_method='mean'
        )
        return np.stack([np.asarray(f) for f in algorithm._last_multitask['per_task']])

    def test_it_runs_another_network(self):
        assert float(np.max(np.abs(self._run('hmr') - self._run('emr')))) > 1e-4
