"""HMR ANTI's neuromodulation rule as neuromodulation.node_rule 'anti'.

ANTI (HMR's anti-correlated-receptors neuromodulation variant) modulates every node from every
neurotransmitter channel, m = receptors @ nt, with gains clip(bg + s m, 0.1, 5) and the gate
sigmoid(m): act(gains pre) * sigmoid(m), no bias and no output inversion. The paper's rule reads
three channels and inverts the outputs with the fourth; HMR FULL's has unclipped gains and the gate
sigmoid(5 m). The parity test runs EMR's rule on ANTI's own arrays against ANTI's own population
evaluator, both as recorded in the research repository (hmr_reference.py); the other two rules are
the negative control.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (DEFAULT_SEED, EMRHyperNEAT, create_config_with_multi_task,
                      create_config_with_neuromodulation)
from emr_hyperneat.emrhyperneat import (
    refuse_anti_branch_gating,
    true_neuromodulation_node_rule,
)
from hmr_reference import hmr_value, live  # type: ignore[import-not-found]

XOR_X = jnp.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=jnp.float32)
XOR_Y = jnp.array([[0.], [1.], [1.], [0.]], dtype=jnp.float32)


class _TwoInputXOR:
    jitable = True
    input_shape = (2,)
    output_shape = (1,)
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]
    input_coords = [[-1.0, 0.0], [1.0, 0.0]]
    output_coords = [[0.0, 1.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


def _hmr_config():
    """ANTI's true neuromodulation: Option A from the weight (tanh), four NT types, strength 5."""
    return {
        'algorithm_params': {
            'hmrhyperneat': {
                'population_size': 10,
                'substrate': {
                    'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                    'output_coords': [[0.0, 1.0]],
                },
                'hmr_hyperneat': {
                    'initial_depth': 0,
                    'max_depth': 3,
                    'variance_threshold': 0.03,
                    'neuromodulation': {
                        'mode': 'true_neuromodulation_4nt_option_a_tanh',
                        'modulation_strength': 5.0,
                    },
                },
                'neat_species': {'compatibility_threshold': 2.5, 'max_stagnation': 40},
            }
        }
    }


def _as_emr(config, **neuromodulation):
    params = dict(config['algorithm_params'])
    block = dict(params.pop('hmrhyperneat'))
    emr = dict(block.pop('hmr_hyperneat'), variance_rule='legacy')
    emr['neuromodulation'] = {**emr['neuromodulation'], **neuromodulation}
    block['emr_hyperneat'] = emr
    params['emrhyperneat'] = block
    return {**config, 'algorithm_params': params}


@pytest.fixture(scope='module')
def antis_arrays():
    """One generation of ANTI: its arrays, its NT for the single-task run, and its own evaluator's
    fitness (hmr_reference.py)."""
    args = (_hmr_config(), _TwoInputXOR(), DEFAULT_SEED, 'xor', XOR_X, XOR_Y)
    return hmr_value('anti_node_rule.antis_arrays', lambda: live('anti_arrays', *args), inputs=args)


def _ours(rule, arrays, nt):
    algorithm = EMRHyperNEAT()
    problem = _TwoInputXOR()
    state = algorithm.initialize(
        algorithm.create_config(_as_emr(_hmr_config(), node_rule=rule)), problem, seed=DEFAULT_SEED
    )
    algorithm.run_generation_verbose(state, problem, skip_metrics=True)
    return np.asarray(algorithm._eval_population_true_neuromodulated(*arrays, nt, XOR_X, XOR_Y))


class TestAgainstLiveAnti:
    def test_equal_on_antis_arrays(self, antis_arrays):
        arrays, nt, theirs = antis_arrays
        np.testing.assert_allclose(_ours('anti', arrays, nt), theirs, rtol=0, atol=1e-6)

    @pytest.mark.parametrize('rule', ['paper', 'hmr_full'])
    def test_negative_control_the_other_rules(self, antis_arrays, rule):
        arrays, nt, theirs = antis_arrays
        assert float(np.max(np.abs(_ours(rule, arrays, nt) - theirs))) > 1e-3


def test_the_rule():
    """The node, from its formula: every channel, clipped gains, no bias, the gate sigmoid(m)."""
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    receptors = jax.random.uniform(k1, (7, 4), minval=-1.0, maxval=1.0)
    base_gains = jax.random.uniform(k2, (7,), minval=0.5, maxval=1.0)
    nt = jax.random.uniform(k3, (4,))
    pre = jax.random.normal(k4, (5, 7))
    rule, operands = true_neuromodulation_node_rule(
        (jnp.tanh, ()), receptors, base_gains, nt, 5.0, 4, 'full', 'anti'
    )
    m = receptors @ nt
    expected = jnp.tanh(jnp.clip(base_gains + 5.0 * m, 0.1, 5.0) * pre) * jax.nn.sigmoid(m)
    np.testing.assert_array_equal(np.asarray(rule(pre, *operands)), np.asarray(expected))
    # Negative control: the paper's rule reads three channels and adds a bias
    paper, paper_operands = true_neuromodulation_node_rule(
        (jnp.tanh, ()), receptors, base_gains, nt, 5.0, 4, 'full', 'paper'
    )
    assert float(jnp.max(jnp.abs(paper(pre, *paper_operands) - expected))) > 1e-3


class TestChecks:
    @staticmethod
    def _create(**neuromodulation):
        EMRHyperNEAT().create_config(_as_emr(_hmr_config(), **neuromodulation))

    def test_no_output_inversion(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(_as_emr(_hmr_config(), node_rule='anti'))
        assert algorithm.neuromod_config.use_output_inversion is False

    @pytest.mark.parametrize('neuromodulation, match', [
        ({'node_rule': 'anti', 'use_output_inversion': True}, 'inverts no output'),
        ({'node_rule': 'anti', 'modulation_mode': 'gating_only'}, 'is not one of its forms'),
    ])
    def test_refused(self, neuromodulation, match):
        with pytest.raises(ValueError, match=match):
            self._create(**neuromodulation)

    @pytest.mark.parametrize('rule', ['anti', 'hmr_full'])
    def test_a_rule_outside_true_neuromodulation(self, rule):
        """Another mode would ignore a node rule, so it raises.
        """
        config = create_config_with_neuromodulation('static_gating')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'][
            'node_rule'
        ] = rule
        with pytest.raises(ValueError, match='is a rule of mode true_neuromodulation'):
            EMRHyperNEAT().create_config(config)

    def test_the_paper_rule_outside_true_neuromodulation_is_accepted(self):
        config = create_config_with_neuromodulation('static_gating')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'][
            'node_rule'
        ] = 'paper'
        EMRHyperNEAT().create_config(config)


class TestBranchGating:
    """Multi-task branch gating gives each region one receptor channel's modulation, which would
    replace the rule's own over every channel; ANTI had no routing. Refused at configuration and
    when a call names a branch mode.
    """

    def test_the_check(self):
        refuse_anti_branch_gating('anti', 'none')
        refuse_anti_branch_gating('paper', 'spatial')
        with pytest.raises(ValueError, match="'spatial' would replace it"):
            refuse_anti_branch_gating('anti', 'spatial')

    def test_configured(self):
        config = create_config_with_multi_task(['xor', 'and'], recurrence_preset='feedforward')
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['neuromodulation']['node_rule'] = 'anti'
        emr['multitask']['branch_gating_mode'] = 'spatial'
        with pytest.raises(ValueError, match='multitask.branch_gating_mode .* would replace it'):
            EMRHyperNEAT().create_config(config)

    def test_named_by_a_call(self):
        algorithm = EMRHyperNEAT()
        problem = _TwoInputXOR()
        state = algorithm.initialize(
            algorithm.create_config(_as_emr(_hmr_config(), node_rule='anti')),
            problem,
            seed=DEFAULT_SEED,
        )
        algorithm.run_generation_verbose(state, problem, skip_metrics=True)
        nt = jnp.array([0.9, 0.1, 0.5, 1.0])
        algorithm.evaluate_with_neurotransmitter(nt, XOR_X, XOR_Y)
        with pytest.raises(ValueError, match="'hierarchical' would replace it"):
            algorithm.evaluate_with_neurotransmitter(
                nt, XOR_X, XOR_Y, branch_gating_mode='hierarchical'
            )
