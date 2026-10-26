"""Settings that used to be dropped without a word now raise.

Each setting below was accepted and then ignored, so a run did something other than its
configuration said. A setting the engine cannot honor raises.
"""
import pytest

from conftest import (
    DEFAULT_SEED,
    MultiTaskProblem,
    EMRHyperNEAT,
    create_base_config,
    create_config_with_multi_task,
    create_config_with_neuromodulation,
)


def _emr(config):
    return config['algorithm_params']['emrhyperneat']['emr_hyperneat']


class TestNeuromodulationLevelFlagsAgainstHmrModeNames:
    """HMR merged a block's level flags into the preset its mode named (get_neuromodulation_config:
    the custom values update the preset's) and evaluated the highest level set. The translation of
    HMR's names dropped the flags, so 'xdg_style' with modulatory_neurons true ran context gating
    where HMR ran modulatory neurons."""

    @pytest.mark.parametrize('block, flag', [
        ({'enabled': True, 'mode': 'xdg_style', 'modulatory_neurons': True}, 'modulatory_neurons'),
        ({'enabled': True, 'mode': 'xdg_style', 'static_gating': False}, 'static_gating'),
        ({'enabled': True, 'mode': 'modulatory_only', 'context_gating': True}, 'context_gating'),
        ({'enabled': True, 'mode': 'true_neuromodulation_4nt_option_a_tanh', 'static_gating': True},
         'static_gating'),
        ({'enabled': True, 'mode': 'true_neuromodulation_4nt', 'true_neuromodulation': False},
         'true_neuromodulation'),
    ])
    def test_a_flag_that_changes_the_named_levels_raises(self, block, flag):
        config = create_base_config(max_depth=3, population_size=20)
        _emr(config)['neuromodulation'] = block
        with pytest.raises(ValueError, match=f'neuromodulation.{flag}'):
            EMRHyperNEAT().create_config(config)

    def test_flags_that_repeat_the_named_levels_are_accepted(self):
        config = create_base_config(max_depth=3, population_size=20)
        _emr(config)['neuromodulation'] = {
            'enabled': True,
            'mode': 'xdg_style',
            'static_gating': True,
            'context_gating': True,
            'modulatory_neurons': False,
        }
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm.neuromod_config.mode == 'context_gating'


@pytest.mark.parametrize('flag', ['allow_backward', 'allow_lateral', 'allow_self_loops'])
def test_a_connection_class_with_hidden_to_hidden_off_raises(flag):
    """enabled true with allow_hidden_to_hidden false built a feed-forward network and kept the
    class flag."""
    config = create_base_config(max_depth=3, population_size=20)
    _emr(config)['recurrence'] = {'enabled': True, 'allow_hidden_to_hidden': False, flag: True}
    with pytest.raises(ValueError, match=f'recurrence.{flag}'):
        EMRHyperNEAT().create_config(config)


def test_modular_aggregation_method_outside_modular_aggregation_raises():
    """dynamic_functions.modular.aggregation_method is read by aggregation.mode 'modular' only; with
    aggregation disabled it was ignored
    (scripts/benchmarks/emr_optimization/benchmark_aggregation_functions.py's
    'modular_act_agg' arm set it that way, commented 'Uses modular aggregation_method instead')."""
    config = create_base_config(max_depth=3, population_size=20)
    _emr(config)['dynamic_functions'] = {
        'mode': 'modular', 'num_activations': 4,
        'modular': {'activation_method': 'magnitude_bio', 'aggregation_method': 'magnitude_bio',
                    'sparsity_method': 'none', 'scaling_method': 'none'}}
    _emr(config)['aggregation'] = {'mode': 'disabled'}
    with pytest.raises(ValueError, match='modular.aggregation_method'):
        EMRHyperNEAT().create_config(config)


class TestMultiTaskCallSettings:
    """run_generation_multitask's call forms read their settings from the call (HMR's
    MultiTaskConfig form) or from a configuration object set by hand, where create_config's checks
    never ran."""

    @staticmethod
    def _setup():
        config = create_config_with_multi_task(
            task_names=['xor', 'and'],
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
            population_size=20,
        )
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(config), MultiTaskProblem('xor'), seed=DEFAULT_SEED
        )
        return algorithm, state

    @staticmethod
    def _tasks():
        return {
            name: (MultiTaskProblem(name).inputs, MultiTaskProblem(name).targets)
            for name in ('xor', 'and')
        }

    @pytest.mark.parametrize('settings, match', [
        (dict(orthogonality_bonus=0.5), 'orthogonality_bonus'),
        (dict(specialization_bonus=0.5), 'specialization_bonus'),
        (dict(generalist_bonus_type='bogus'), 'generalist_bonus_type'),
        (dict(confusion_eval_frequency=1), 'confusion_eval_frequency'),
    ])
    def test_hmr_call_form_refuses_what_it_would_ignore(self, settings, match):
        from emr_hyperneat.emrhyperneat import MultiTaskConfig
        algorithm, state = self._setup()
        with pytest.raises(ValueError, match=match):
            algorithm.run_generation_multitask(
                state, MultiTaskConfig(tasks=self._tasks(), aggregation='mean', **settings)
            )

    def test_per_node_functions_with_a_direct_call_raise(self):
        """The configuration check needs multitask.enabled; a direct list-form call ran without it
        and ignored the per-node functions (the multi-task evaluation applies one hidden activation
        and sum aggregation).
        """
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt', recurrence_preset='feedforward', population_size=20
        )
        config['algorithm_params']['emrhyperneat']['substrate'] = {
            'input_coords': [[-1.0, 0.0], [1.0, 0.0]], 'output_coords': [[0.0, 1.0]],
            'output_activation': 'sigmoid', 'hidden_activation': 'tanh'}
        _emr(config)['dynamic_functions'] = {'mode': 'cppn_output'}
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(config), MultiTaskProblem('xor'), seed=DEFAULT_SEED
        )
        with pytest.raises(ValueError, match='would ignore'):
            algorithm.run_generation_multitask(
                state, [MultiTaskProblem('xor'), MultiTaskProblem('and')]
            )
