"""The prototype's LEO gate as leo.mode 'compound_unrescaled'.

The prototype's LEO variant kept a weight W = tanh(w) * max_weight where |W| > 0.1 and tanh(leo) >
0.2. EMR's 'compound' thresholds the tanh-normalized weight and rescales the kept range linearly, so
its networks differ.

A generation-level comparison with the prototype's variant is not possible: its initialize rebuilt
the CPPN with the LEO output into self._state but returned the population built before, which has no
LEO output, so every network its runners evaluated at generation 0 was empty (pinned below on the
prototype's values recorded in the research repository, prototype_reference.py). The rebuilt
population, never used, took 150 genomes whatever the configuration. The gate is checked instead
against the prototype's formula as written, on every call a generation makes.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import emr_hyperneat.emrhyperneat as E
from conftest import DEFAULT_SEED, EMRHyperNEAT, create_base_config
from emr_hyperneat.emrhyperneat import (
    leo_sparsification_compound_unrescaled,
    leo_sparsify,
)
from prototype_reference import prototype_value, live  # type: ignore[import-not-found]

POP = 12


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


def _prototype_config():
    return {'algorithm_params': {'emrhyperneat': {
        'population_size': POP,
        'substrate': {'input_coords': [[-1.0, 0.0], [1.0, 0.0]], 'output_coords': [[0.0, 1.0]]},
        'emr_hyperneat': {'initial_depth': 0, 'max_depth': 3, 'variance_threshold': 0.03},
        'neat_species': {'compatibility_threshold': 2.5, 'max_stagnation': 40},
    }}}


def _emr_config(mode):
    params = dict(_prototype_config()['algorithm_params'])
    block = dict(params.pop('emrhyperneat'))
    block['emr_hyperneat'] = dict(block.pop('emr_hyperneat'), variance_rule='legacy',
                                  leo={'enabled': True, 'threshold': 0.2, 'mode': mode})
    params['emrhyperneat'] = block
    return {'algorithm_params': params}


def _w1(mode):
    algorithm = EMRHyperNEAT()
    problem = _TwoInputXOR()
    state = algorithm.initialize(
        algorithm.create_config(_emr_config(mode)), problem, seed=DEFAULT_SEED
    )
    algorithm.run_generation(state, problem)
    return np.asarray(algorithm._cached_W1)


class TestTheGateInAGeneration:
    def test_every_call_is_the_prototype_formula(self, monkeypatch):
        """Both gate calls of a generation (W1 and W2) return the prototype's W = tanh(w) *
        max_weight, kept where |W| > 0.1 and tanh(leo) > 0.2, bit for bit on the arguments they
        received."""
        calls = []
        gate = E.leo_sparsify

        def recording(weights, leo_outputs, leo_mode, weight_threshold, leo_threshold, max_weight):
            out = gate(weights, leo_outputs, leo_mode, weight_threshold, leo_threshold, max_weight)
            calls.append(
                (
                    np.asarray(weights),
                    np.asarray(leo_outputs),
                    leo_mode,
                    weight_threshold,
                    leo_threshold,
                    max_weight,
                    np.asarray(out),
                )
            )
            return out

        monkeypatch.setattr(E, 'leo_sparsify', recording)
        W1 = _w1('compound_unrescaled')
        assert len(calls) == 2
        for weights, leo, mode, weight_threshold, leo_threshold, max_weight, out in calls:
            assert (mode, weight_threshold, leo_threshold) == ('compound_unrescaled', 0.1, 0.2)
            scaled = weights * np.float32(max_weight)
            np.testing.assert_array_equal(
                out, np.where((np.abs(scaled) > 0.1) & (leo > 0.2), scaled, 0.0)
            )
        assert np.count_nonzero(W1) > 0

    def test_negative_control_compound(self):
        assert float(np.max(np.abs(_w1('compound') - _w1('compound_unrescaled')))) > 1e-3


def test_the_gate():
    weights = jnp.array([0.005, -0.02, 0.5, -0.9, 0.9])
    leo = jnp.array([0.9, 0.9, 0.9, 0.1, 0.3])
    gated = leo_sparsification_compound_unrescaled(weights, leo, 0.1, 0.2, 8.0)
    # |W| = 0.04 and 0.16: the first under 0.1; the fourth's LEO under 0.2
    np.testing.assert_array_equal(
        np.asarray(gated), np.asarray(jnp.array([0.0, -0.16, 4.0, 0.0, 7.2]))
    )
    np.testing.assert_array_equal(
        np.asarray(leo_sparsify(weights, leo, 'compound_unrescaled', 0.1, 0.2, 8.0)), np.asarray(gated)
    )


def test_locality_is_refused():
    config = create_base_config(population_size=10)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr.update(leo={'enabled': True, 'mode': 'compound_unrescaled'}, locality_radius=0.5)
    with pytest.raises(ValueError, match="'compound_unrescaled' with locality_radius is refused"):
        EMRHyperNEAT().create_config(config)


class TestPrototypeLeoVariant:
    """Why no generation of the prototype's LEO variant is compared: pinned facts about the frozen
    engine."""

    def test_its_runners_got_a_population_without_the_leo_output(self):
        args = (_prototype_config(), _TwoInputXOR(), DEFAULT_SEED)
        recorded = prototype_value(
            'leo_compound_unrescaled.leo_variant', lambda: live('leo_variant', *args), inputs=args
        )
        assert not np.any(recorded['population_keys'] == recorded['leo_output'])
        assert recorded['nonzero_weights'] == 0
        # The rebuilt population, never returned, ignores the configured size
        assert recorded['rebuilt_population'] == 150 != POP
