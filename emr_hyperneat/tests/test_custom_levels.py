"""HMR's 'custom' neuromodulation with its level flags.

HMR's neuromodulation variant evaluated one level, the highest of true neuromodulation, modulatory
neurons, context gating and static gating; its gating read the CPPN's gate output (context gating as
logits) and its modulatory neurons the CPPN's factor and node-type outputs. So 'custom' with one of
those levels is that level with its CPPN source, and static with context gating ran context gating
(HMR's 'xdg_style'). A combination whose lower levels made the CPPN evolve outputs nothing read is
refused, as 'full' is. 'custom' with true neuromodulation alone is HMR's Option A or B, as its flags
say.
"""
import numpy as np
import pytest

from conftest import DEFAULT_SEED, EMRHyperNEAT, XORProblem, create_config_with_neuromodulation
from emr_hyperneat.emrhyperneat import (
    _translate_hmr_neuromodulation,
)

SETTINGS = {'gate_scaling': 'sigmoid', 'context_influence': 0.5}


@pytest.mark.parametrize(
    'flags, expected',
    [
        ({'static_gating': True}, {'mode': 'static_gating', 'gate_source': 'cppn'}),
        ({'context_gating': True}, {'mode': 'context_gating', 'gate_source': 'cppn'}),
        (
            {'static_gating': True, 'context_gating': True},
            {'mode': 'context_gating', 'gate_source': 'cppn'},
        ),
        ({'modulatory_neurons': True}, {'mode': 'modulatory_neurons', 'modulatory_source': 'cppn'}),
    ],
)
def test_the_level_hmr_evaluated(flags, expected):
    translated = _translate_hmr_neuromodulation(
        {'mode': 'custom', 'enabled': True, **flags, **SETTINGS}
    )
    assert translated == {**SETTINGS, **expected}


def test_static_and_context_gating_is_xdg_style():
    custom = _translate_hmr_neuromodulation(
        {'mode': 'custom', 'static_gating': True, 'context_gating': True}
    )
    assert custom == _translate_hmr_neuromodulation({'mode': 'xdg_style'})


@pytest.mark.parametrize(
    'flags, evaluated',
    [
        ({'static_gating': True, 'modulatory_neurons': True}, 'modulatory_neurons'),
        ({'context_gating': True, 'true_neuromodulation': True}, 'true_neuromodulation'),
        ({'modulatory_neurons': True, 'true_neuromodulation': True}, 'true_neuromodulation'),
        (
            {'static_gating': True, 'context_gating': True, 'modulatory_neurons': True},
            'modulatory_neurons',
        ),
    ],
)
def test_a_combination_with_unread_outputs(flags, evaluated):
    with pytest.raises(ValueError, match=f'HMR evaluated {evaluated} alone'):
        _translate_hmr_neuromodulation({'mode': 'custom', **flags})


def test_no_level():
    with pytest.raises(ValueError, match='with no level flag'):
        _translate_hmr_neuromodulation({'mode': 'custom', 'enabled': True})


def test_a_contradicting_source():
    with pytest.raises(ValueError, match="contradicts HMR's 'custom'"):
        _translate_hmr_neuromodulation(
            {'mode': 'custom', 'static_gating': True, 'gate_source': 'weights'}
        )


def _fitness(neuromodulation):
    config = create_config_with_neuromodulation('static_gating')
    config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'] = neuromodulation
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
    algorithm.run_generation(state, XORProblem())
    return np.asarray(algorithm._cached_fitnesses)


def test_a_custom_run_is_the_translated_run():
    """End to end: HMR's 'custom' with static gating runs EMR's static gating from the CPPN's gate
    output, bit for bit; the weights-derived gates (EMR's default source) are the negative control.
    """
    custom = _fitness({'mode': 'custom', 'enabled': True, 'static_gating': True})
    explicit = _fitness({'mode': 'static_gating', 'gate_source': 'cppn'})
    weights = _fitness({'mode': 'static_gating'})
    np.testing.assert_array_equal(custom, explicit)
    assert float(np.max(np.abs(custom - weights))) > 1e-4
