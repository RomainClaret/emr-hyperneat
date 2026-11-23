"""Setting checks that the configuration-parsing tests leave out.

Each case is either a refusal (a setting the run would silently ignore, or a name that is not one)
or the acceptance that bounds it (the same block where it is read):
- neuromodulation.enabled agrees with the mode or preset, also when the level is given by one of
  the prototype's names, which the engine translates;
- the known multi-task activation names are accepted at create_config;
- evaluate_with_neurotransmitter refuses an empty activation name (only None means tanh);
- dynamic_functions.sparsity, dynamic_functions.modular and palette_evolution settings raise
  outside the modes that read them, and are accepted there.
"""
from typing import Any, Dict

import jax.numpy as jnp
import pytest
from conftest import (  # type: ignore[import-not-found]
    DEFAULT_SEED,
    XORProblem,
    create_base_config,
    create_config_with_neuromodulation,
)
from emr_hyperneat import EMRHyperNEAT


def _config(**sections: Any) -> Dict[str, Any]:
    config = create_base_config(population_size=8)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat'].update(sections)
    return config


class TestNeuromodulationEnabledAgreesWithTheLevel:
    """neuromodulation.enabled must agree with the mode or preset."""

    @pytest.mark.parametrize('block', [{'enabled': False, 'mode': 'static_gating'},
                                       {'enabled': False, 'preset': 'xdg_style'},
                                       {'enabled': False, 'mode': 'xdg_style'}])
    def test_enabled_false_with_a_level_raises(self, block):
        """The prototype's mode names are translated, and the translation drops enabled: it is read
        from the block given."""
        with pytest.raises(ValueError, match='enabled is false but a mode or preset is set'):
            EMRHyperNEAT().create_config(_config(neuromodulation=block))

    @pytest.mark.parametrize('block', [{'enabled': False, 'mode': 'disabled'}, {'enabled': False},
                                       {'enabled': True, 'mode': 'static_gating'},
                                       {'mode': 'static_gating'}])
    def test_consistent_blocks_are_accepted(self, block):
        EMRHyperNEAT().create_config(_config(neuromodulation=block))


class TestMultiTaskActivationNames:
    """Every multi-task activation name is checked at create_config, not at the first generation."""

    def test_known_names_are_accepted(self):
        config = _config(neuromodulation={'enabled': True, 'mode': 'true_neuromodulation'},
                         multitask={'enabled': True, 'hidden_activation': 'tanh',
                                    'per_task_activation': {'xor': 'sin', 'and': 'sigmoid'}})
        EMRHyperNEAT().create_config(config)


class TestEvaluateWithNeurotransmitterActivationName:
    """Only None means tanh: an empty name is an unknown one, as the docstring says."""

    def test_an_empty_name_raises(self):
        config = create_config_with_neuromodulation('true_neuromodulation_4nt',
                                                    recurrence_preset='feedforward',
                                                    population_size=10)
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(),
                                     seed=DEFAULT_SEED)
        algorithm.run_generation(state, XORProblem())
        with pytest.raises(ValueError, match="unknown activation ''"):
            algorithm.evaluate_with_neurotransmitter(jnp.array([0.5, 0.5, 0.5, 1.0]),
                                                     algorithm._cached_inputs,
                                                     algorithm._cached_targets,
                                                     activation_name='')


class TestBlocksOutsideTheModesThatReadThem:
    """Each block is read in one mode; given in another, it raises rather than being ignored."""

    @pytest.mark.parametrize('dynamic_functions', [
        {'mode': 'disabled', 'sparsity': {'level_0': 0.1}},
        {'mode': 'weight_interpretation', 'interpretation': 'sparsity_threshold',
         'sparsity': {'level_0': 0.1}},
        {'mode': 'modular', 'sparsity': {'level_0': 0.1}}])
    def test_sparsity_outside_hierarchical_sparsity_raises(self, dynamic_functions):
        with pytest.raises(ValueError, match="dynamic_functions.sparsity sets the level fractions"):
            EMRHyperNEAT().create_config(_config(dynamic_functions=dynamic_functions))

    @pytest.mark.parametrize('modular', [{'activation_method': 'sign'}, {'sparsity_method': 'wta'},
                                         {'scaling_method': 'stp_inspired'},
                                         {'sparsity': {'level_0': 0.1}}])
    def test_modular_settings_outside_mode_modular_raise(self, modular):
        with pytest.raises(ValueError, match="configure dynamic_functions.mode 'modular' only"):
            EMRHyperNEAT().create_config(
                _config(dynamic_functions={'mode': 'cppn_output', 'modular': modular}))

    def test_disabled_palette_evolution_with_settings_raises(self):
        with pytest.raises(ValueError, match="palette_evolution settings are given but"):
            EMRHyperNEAT().create_config(_config(dynamic_functions={
                'mode': 'cppn_output',
                'palette_evolution': {'enabled': False, 'mutation_rate': 0.1}}))

    @pytest.mark.parametrize('sections', [
        {'dynamic_functions': {'mode': 'weight_interpretation',
                               'interpretation': 'hierarchical_sparsity',
                               'sparsity': {'level_0': 0.1}}},
        {'dynamic_functions': {'mode': 'modular',
                               'modular': {'sparsity_method': 'hierarchical',
                                           'sparsity': {'level_0': 0.1}}}},
        {'dynamic_functions': {'mode': 'disabled',
                               'modular': {'aggregation_method': 'magnitude_bio'}},
         'aggregation': {'mode': 'modular'}},
        {'dynamic_functions': {'mode': 'cppn_output', 'palette_evolution': {'enabled': False}}}])
    def test_blocks_where_they_are_read_are_accepted(self, sections):
        EMRHyperNEAT().create_config(_config(**sections))
