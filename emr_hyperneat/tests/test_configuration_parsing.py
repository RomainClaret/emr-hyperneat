"""Tests for configuration parsing in EMR-HyperNEAT.

Tests cover:
- Recurrence configuration parsing (_parse_extended_config)
- Dynamic functions configuration parsing (_parse_dynamic_functions_config)
- Neuromodulation configuration parsing (_parse_neuromodulation_config)
- Preset loading and validation
- Error handling for invalid configurations
"""

import numpy as np
import pytest
from conftest import (
    EMRHyperNEAT,
    create_base_config,
    create_config_with_recurrence,
    create_config_with_dynamic_functions,
    create_config_with_aggregation,
    create_config_with_neuromodulation,
    RECURRENCE_PRESETS,
    DYNAMIC_FUNCTION_PRESETS,
    AGGREGATION_PRESETS,
    NEUROMODULATION_PRESETS,
    XORProblem,
    DEFAULT_SEED,
)
from emr_hyperneat.emrhyperneat import EMRConfig, ForwardPassMode


class TestRecurrenceConfigParsing:
    """Tests for recurrence/connection type configuration parsing."""

    def test_feedforward_preset_parses(self, algorithm, xor_problem):
        """Verify feedforward preset parses correctly."""
        config = create_config_with_recurrence('feedforward')
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Verify recurrence is disabled
        assert algorithm.extended_config is not None
        assert algorithm.extended_config.enabled == False
        # Note: iteration_level may still be set even when disabled

    @pytest.mark.parametrize("preset", list(RECURRENCE_PRESETS.keys()))
    def test_all_recurrence_presets_parse(self, algorithm, xor_problem, preset):
        """Verify all recurrence presets parse without errors."""
        config = create_config_with_recurrence(preset)
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Verify config was applied
        assert algorithm.extended_config is not None
        expected = RECURRENCE_PRESETS[preset]
        if preset != 'feedforward':
            assert algorithm.extended_config.enabled == expected.get('enabled', True)

    def test_custom_recurrence_config(self, algorithm, xor_problem):
        """Verify custom recurrence configuration works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'allow_backward': True,
            'allow_lateral': False,
            'allow_self_loops': True,
            'iteration_level': 3,
            'multi_hop_algorithm': 'fori_loop',
            'hop_decay_factor': 0.5,
        }

        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.extended_config.allow_backward == True
        assert algorithm.extended_config.allow_lateral == False
        assert algorithm.extended_config.allow_self_loops == True
        assert algorithm.extended_config.iteration_level == 3

    def test_invalid_preset_raises_error(self, algorithm):
        """Verify invalid preset name raises error."""
        with pytest.raises(ValueError, match="Unknown recurrence preset"):
            create_config_with_recurrence('invalid_preset')

    def test_preset_decides_whether_recurrence_is_on(self):
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {'preset': 'hidden_only', 'enabled': True,
                             'allow_hidden_to_hidden': True}
        EMRHyperNEAT().create_config(config)
        hmr['recurrence']['allow_hidden_to_hidden'] = False
        with pytest.raises(ValueError, match="contradicts preset 'hidden_only'"):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('preset, key, value', [
        ('hidden_only', 'enabled', False),
        ('hidden_only', 'allow_hidden_to_hidden', False),
        ('full_recurrent', 'enabled', 0),
        ('feedforward', 'enabled', True),
        ('feedforward', 'allow_hidden_to_hidden', True),
    ])
    def test_a_key_contradicting_the_preset_is_refused(self, preset, key, value):
        """A preset decides enabled and allow_hidden_to_hidden, so a value that contradicts it
        raises and an agreeing one is accepted: a block cannot say one thing and run the other."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {'preset': preset, key: value}
        with pytest.raises(ValueError, match=f"recurrence.{key}={value!r} contradicts preset "
                                             f"'{preset}'"):
            EMRHyperNEAT().create_config(config)
        hmr['recurrence'] = {'preset': preset, key: not value}
        EMRHyperNEAT().create_config(config)

    def test_iteration_level_configuration(self, algorithm, xor_problem):
        """Verify iteration_level is set correctly for recurrence presets."""
        config = create_config_with_recurrence('hidden_only')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # iteration_level should be set
        assert algorithm.extended_config.iteration_level >= 0
        # Note: activate_time may or may not be auto-computed depending on implementation


class TestDynamicFunctionsConfigParsing:
    """Tests for dynamic activation functions configuration parsing."""

    def test_disabled_mode_parses(self, algorithm, xor_problem):
        """Verify disabled mode parses correctly."""
        config = create_config_with_dynamic_functions('disabled')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.dynamic_functions_mode == 'disabled'

    @pytest.mark.parametrize("mode", list(DYNAMIC_FUNCTION_PRESETS.keys()))
    def test_all_dynamic_function_modes_parse(self, algorithm, xor_problem, mode):
        """Verify all dynamic function modes parse without errors."""
        config = create_config_with_dynamic_functions(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_global_mode_sets_activations(self, algorithm, xor_problem):
        """Verify global mode sets the hidden activation correctly."""
        config = create_config_with_dynamic_functions('global_sin')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.dynamic_functions_mode == 'global'
        assert algorithm.df_hidden_activation == 'sin'

    def test_cppn_output_mode_sets_num_activations(self, algorithm, xor_problem):
        """Verify CPPN output mode sets number of activations."""
        config = create_config_with_dynamic_functions('cppn_output_6')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.dynamic_functions_mode == 'cppn_output'
        assert algorithm.df_num_activations == 6

    def test_weight_interpretation_modes(self, algorithm, xor_problem):
        """Verify weight interpretation modes parse correctly."""
        for interp in ['sign', 'magnitude_bio', 'variance']:
            config = create_config_with_dynamic_functions(f'weight_interp_{interp}')
            config_obj = algorithm.create_config(config)
            algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.dynamic_functions_mode == 'weight_interpretation'
            assert algorithm.df_interpretation == interp

    def test_hh_activation_mode_options(self, algorithm, xor_problem):
        """Verify H->H activation mode options work."""
        for hh_mode in ['tanh', 'every_iteration']:
            config = create_config_with_dynamic_functions(
                'cppn_output_4',
                recurrence_preset='hidden_only',
                hh_activation_mode=hh_mode
            )
            config_obj = algorithm.create_config(config)
            algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.hh_activation_mode == hh_mode

    def test_invalid_mode_raises_error(self, algorithm):
        """Verify invalid dynamic functions mode raises error."""
        with pytest.raises(ValueError, match="Unknown dynamic functions mode"):
            create_config_with_dynamic_functions('invalid_mode')

    def test_custom_num_activations(self, algorithm, xor_problem):
        """Verify custom number of activations works."""
        config = create_config_with_dynamic_functions('cppn_output_4', num_activations=8)
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.df_num_activations == 8


class TestAggregationConfigParsing:
    """Tests for aggregation configuration parsing."""

    def test_disabled_mode_parses(self, algorithm, xor_problem):
        """Verify disabled aggregation mode parses correctly."""
        config = create_config_with_aggregation('disabled')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.agg_mode == 'disabled'

    @pytest.mark.parametrize("mode", list(AGGREGATION_PRESETS.keys()))
    def test_all_aggregation_modes_parse(self, algorithm, xor_problem, mode):
        """Verify all aggregation modes parse without errors."""
        config = create_config_with_aggregation(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("func", ['sum', 'mean', 'max', 'min', 'product'])
    def test_global_aggregation_options(self, xor_problem, func):
        """Verify all global aggregation functions can be configured."""
        # Create fresh algorithm instance for each function
        algorithm = EMRHyperNEAT()
        config = create_config_with_aggregation(f'global_{func}')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Verify mode is set to global (function may default if not parsed)
        assert algorithm.agg_mode == 'global'

    def test_hh_aggregation_mode_options(self, algorithm, xor_problem):
        """Verify H->H aggregation mode options work."""
        for hh_mode in ['sum', 'dynamic']:
            config = create_config_with_aggregation(
                'weight_interp',
                recurrence_preset='hidden_only',
                hh_aggregation_mode=hh_mode
            )
            config_obj = algorithm.create_config(config)
            algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.hh_aggregation_mode == hh_mode

    def test_invalid_mode_raises_error(self, algorithm):
        """Verify invalid aggregation mode raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation mode"):
            create_config_with_aggregation('invalid_mode')


class TestNeuromodulationConfigParsing:
    """Tests for neuromodulation configuration parsing."""

    def test_disabled_mode_parses(self, algorithm, xor_problem):
        """Verify disabled neuromodulation parses correctly."""
        config = create_config_with_neuromodulation('disabled')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # When disabled, neuromod_config may be None or have enabled=False
        if algorithm.neuromod_config is not None:
            assert algorithm.neuromod_config.enabled == False or algorithm.neuromod_config.mode == 'disabled'

    @pytest.mark.parametrize("mode", list(NEUROMODULATION_PRESETS.keys()))
    def test_all_neuromodulation_modes_parse(self, algorithm, xor_problem, mode):
        """Verify all neuromodulation modes parse without errors."""
        config = create_config_with_neuromodulation(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_static_gating_config(self, algorithm, xor_problem):
        """Verify static gating configuration parses correctly."""
        config = create_config_with_neuromodulation('static_gating')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Check that neuromod_config exists and has correct mode
        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'static_gating'

    def test_true_neuromodulation_config(self, algorithm, xor_problem):
        """Verify TRUE neuromodulation configuration parses correctly."""
        config = create_config_with_neuromodulation('true_neuromodulation_4nt')
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Check that neuromod_config exists and has correct mode
        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'true_neuromodulation'
        assert algorithm.neuromod_config.num_nt_types == 4

    def test_receptor_derivation_options(self, algorithm, xor_problem):
        """Verify all receptor derivation methods parse."""
        for derivation in ['tanh', 'orthogonal', 'phase_shifted']:
            config = create_config_with_neuromodulation(
                'true_neuromodulation_4nt',
                receptor_derivation=derivation
            )
            config_obj = algorithm.create_config(config)
            algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.neuromod_config.receptor_derivation == derivation

    def test_invalid_mode_raises_error(self, algorithm):
        """Verify invalid neuromodulation mode raises error."""
        with pytest.raises(ValueError, match="Unknown neuromodulation mode"):
            create_config_with_neuromodulation('invalid_mode')


class TestBaseConfigParsing:
    """Tests for base configuration parsing."""

    @pytest.mark.parametrize("pop_size", [50, 100, 200])
    def test_population_size_parsing(self, xor_problem, pop_size):
        """Verify different population sizes can be configured."""
        # Create fresh algorithm instance
        algorithm = EMRHyperNEAT()
        config = create_base_config(population_size=pop_size)
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Initialization should complete without error
        # Note: population_size may be stored in different attributes depending on implementation

    def test_max_depth_parsing(self, algorithm, xor_problem):
        """Verify max depth is parsed correctly."""
        for depth in [2, 3, 4, 5]:
            config = create_base_config(max_depth=depth)
            config_obj = algorithm.create_config(config)
            algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.max_depth == depth

    def test_threshold_parameters(self, algorithm, xor_problem):
        """Verify threshold parameters are parsed correctly."""
        config = create_base_config(
            variance_threshold=0.05,
            division_threshold=0.05,
            band_threshold=0.2,
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.variance_threshold == 0.05
        assert algorithm.division_threshold == 0.05
        assert algorithm.band_threshold == 0.2

    def test_substrate_coordinates(self, algorithm, xor_problem):
        """Verify substrate coordinates configuration is accepted."""
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['substrate'] = {
            'input_coords': [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],
            'output_coords': [(0.0, 1.0)],
            'output_activation': 'sigmoid',
            'hidden_activation': 'tanh',
        }
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Initialization should complete without error
        # Substrate coordinates are processed internally during initialization


class TestConfigurationCombinations:
    """Tests for valid combinations of configurations."""

    def test_recurrence_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify recurrence + dynamic functions combination works."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only'
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.dynamic_functions_mode == 'cppn_output'

    def test_recurrence_with_neuromodulation(self, algorithm, xor_problem):
        """Verify recurrence + neuromodulation combination works."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            recurrence_preset='with_backward'
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'true_neuromodulation'

    def test_full_feature_combination(self, algorithm, xor_problem):
        """Verify all features can be enabled together."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']

        # Enable all features
        hmr['recurrence'] = RECURRENCE_PRESETS['hidden_only'].copy()
        hmr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}
        hmr['aggregation'] = {'mode': 'global', 'global_function': 'sum'}
        hmr['neuromodulation'] = NEUROMODULATION_PRESETS['true_neuromodulation_4nt'].copy()

        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # All features should be enabled
        assert algorithm.extended_config.enabled == True
        assert algorithm.dynamic_functions_mode == 'cppn_output'
        assert algorithm.agg_mode == 'global'
        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'true_neuromodulation'


class TestRetiredKnobs:
    """Older configs may still carry retired keys."""

    def test_use_vectorized_discovery_is_ignored_not_fatal(self, xor_problem):
        # use_vectorized_discovery is a retired knob that nothing consults; a config
        # that still carries it must build and run.
        from conftest import create_config_with_recurrence, run_quick_evolution
        config = create_config_with_recurrence("hidden_only")
        config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["recurrence"][
            "use_vectorized_discovery"] = False
        result = run_quick_evolution(
            EMRHyperNEAT(), config, xor_problem, generations=1, seed=7)
        assert result.error is None, result.error


class TestSelectorGuards:
    """A selector that names a path must reach that path, or say it cannot.

    Each guard here covers a setting that would otherwise be accepted, echoed back
    under verbose, and then quietly ignored, and the run would look like a
    successful run of the thing you asked for.
    """

    @staticmethod
    def _config(**emr_overrides):
        config = create_base_config(max_depth=2, population_size=10)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat'].update(emr_overrides)
        return config

    def test_unknown_multi_gpu_strategy_raises(self):
        """An unknown strategy name raises instead of warning and running the default strategy."""
        config = self._config(multi_gpu_strategy='SINGLE_GPUS')
        with pytest.raises(ValueError, match="Unknown multi_gpu_strategy 'SINGLE_GPUS'"):
            EMRHyperNEAT().create_config(config)

    def test_population_parallel_process_refuses_per_node_functions(self, monkeypatch):
        """Its worker processes evaluate tanh nodes with sum aggregation whatever the configuration
        asks for, so per-node functions are refused. The check runs at dispatch on two or more
        devices (one device runs the standard loop, which computes every feature)."""
        import jax
        from emr_hyperneat.emrhyperneat_base import MultiGPUStrategy
        config = self._config(dynamic_functions={'mode': 'cppn_output', 'num_activations': 4})
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.POPULATION_PARALLEL_PROCESS)
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(),
                                     seed=DEFAULT_SEED)
        cpu = jax.devices()[0]
        monkeypatch.setattr(jax, 'devices', lambda backend=None: [cpu, cpu])
        with pytest.raises(ValueError, match='POPULATION_PARALLEL_PROCESS evaluates tanh'):
            algorithm.run_until_threshold(state, XORProblem(), target_fitness=2.0,
                                          max_generations=1)

    def test_unknown_dynamic_functions_mode_raises(self):
        """A typo raises instead of falling through the dispatch to the tanh baseline."""
        config = self._config(dynamic_functions={'mode': 'cppn-output'})
        with pytest.raises(ValueError, match='dynamic_functions.mode'):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('mode', sorted(
        EMRHyperNEAT.DYNAMIC_FUNCTION_MODES - {'modular'}))
    def test_every_documented_mode_still_parses(self, mode):
        """The guard must not narrow what already worked."""
        config = self._config(dynamic_functions={'mode': mode})
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm.dynamic_functions_mode == mode

    def test_modular_mode_is_dispatched(self):
        """Modular mode reaches its own branch (0.1.0 parsed it, warned that it was not implemented
        and evaluated fixed tanh). The default layers decode magnitude_bio indices (per node, so no
        single hidden function); activation method 'none' gives every node tanh. Behavior:
        test_dynamic_functions.py TestModularMode."""
        import warnings
        config = self._config(dynamic_functions={'mode': 'modular'})
        algorithm = EMRHyperNEAT()
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            algorithm.create_config(config)
        assert algorithm.dynamic_functions_mode == 'modular'
        assert algorithm.df_modular_config['activation_method'] == 'magnitude_bio'
        assert algorithm._hidden_act_fn is None
        config = self._config(
            dynamic_functions={'mode': 'modular', 'modular': {'activation_method': 'none'}}
        )
        algorithm.create_config(config)
        assert algorithm._hidden_act_fn is not None

    @pytest.mark.parametrize('mode', ['streaming', 'depth_offload'])
    def test_legacy_variance_rule_is_silent_outside_the_standard_path(self, mode):
        """Every mode applies `legacy` (test_depth_lookahead.py checks the substrates), so
        selecting it outside the standard path gives no warning."""
        config = self._config(variance_rule='legacy', execution={'mode': mode})
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            algorithm = EMRHyperNEAT()
            algorithm.create_config(config)
        assert algorithm.variance_rule == 'legacy'

    def test_legacy_variance_rule_is_silent_on_the_standard_path(self):
        """The reproduction pin itself must stay quiet where it does apply."""
        config = self._config(variance_rule='legacy', execution={'mode': 'standard'})
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            algorithm = EMRHyperNEAT()
            algorithm.create_config(config)
        assert algorithm.variance_rule == 'legacy'

    def test_missing_config_block_warns(self):
        """An HMR-shaped or misnested config warns: it would default every EMR setting."""
        with pytest.warns(UserWarning, match="No 'emr_hyperneat' configuration block"):
            EMRHyperNEAT().create_config(
                {'algorithm_params': {'emrhyperneat': {'population_size': 10}}})

    def test_missing_config_block_warns_once(self):
        """The subclass defers to the base parse; two warnings would be noise."""
        with pytest.warns(UserWarning) as record:
            EMRHyperNEAT().create_config(
                {'algorithm_params': {'emrhyperneat': {'population_size': 10}}})
        block_warnings = [
            w for w in record
            if "No 'emr_hyperneat' configuration block" in str(w.message)
        ]
        assert len(block_warnings) == 1

    def test_well_formed_config_does_not_warn(self):
        """The guards must be silent on the configs the suite already uses."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            EMRHyperNEAT().create_config(self._config())


class TestForwardModeLabels:
    """FULL_SPARSE is not implemented: it is refused rather than run with no h->h, as DENSE_ONLY."""

    def test_full_sparse_is_refused(self):
        from emr_hyperneat.emrhyperneat import ForwardPassMode
        import importlib
        import inspect
        import pkgutil
        from emr_hyperneat import emrhyperneat as module
        import emr_hyperneat.engine as engine

        # the facade and every engine module, where its code moved
        sources = [inspect.getsource(module)] + [
            inspect.getsource(importlib.import_module(info.name))
            for info in pkgutil.iter_modules(engine.__path__, 'emr_hyperneat.engine.')]
        assert all('extreme recurrence' not in source for source in sources)

        algorithm = EMRHyperNEAT()
        with pytest.raises(NotImplementedError, match='FULL_SPARSE is not implemented'):
            algorithm.set_forward_mode(ForwardPassMode.FULL_SPARSE)
        assert algorithm.get_forward_mode() is not ForwardPassMode.FULL_SPARSE


class TestMultiHopSecondFactorParsing:
    """`recurrence.multi_hop_second_factor` reaches the engine through both parser branches."""

    def test_default_is_masked_in_both_branches(self, algorithm, xor_problem):
        preset_config = create_config_with_recurrence('hidden_only')
        algorithm.initialize(algorithm.create_config(preset_config), xor_problem, seed=DEFAULT_SEED)
        assert algorithm.extended_config.multi_hop_second_factor == 'masked'
        explicit = create_base_config()
        explicit['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = {
            'enabled': True, 'allow_hidden_to_hidden': True, 'iteration_level': 2,
        }
        algorithm.initialize(algorithm.create_config(explicit), xor_problem, seed=DEFAULT_SEED)
        assert algorithm.extended_config.multi_hop_second_factor == 'masked'

    def test_raw_is_honored_in_both_branches(self, algorithm, xor_problem):
        preset_config = create_config_with_recurrence('hidden_only')
        preset_config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['multi_hop_second_factor'] = 'raw'
        algorithm.initialize(algorithm.create_config(preset_config), xor_problem, seed=DEFAULT_SEED)
        assert algorithm.extended_config.multi_hop_second_factor == 'raw'
        explicit = create_base_config()
        explicit['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = {
            'enabled': True, 'allow_hidden_to_hidden': True, 'iteration_level': 2,
            'multi_hop_second_factor': 'raw',
        }
        algorithm.initialize(algorithm.create_config(explicit), xor_problem, seed=DEFAULT_SEED)
        assert algorithm.extended_config.multi_hop_second_factor == 'raw'

    def test_unknown_value_is_refused(self, algorithm):
        config = create_config_with_recurrence('hidden_only')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence']['multi_hop_second_factor'] = 'sum'
        with pytest.raises(ValueError):
            algorithm.create_config(config)


class TestPresetIterationLevelDefault:
    """The recurrent presets discover direct connections only (iteration level 1); level 2 stays
    available by setting it."""

    @staticmethod
    def _recurrence(config, **section):
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = section
        return config

    @pytest.mark.parametrize('preset', ['hidden_only', 'with_backward', 'with_lateral', 'with_self',
                                        'full_recurrent'])
    def test_preset_runs_level_one(self, algorithm, preset):
        algorithm.create_config(self._recurrence(create_base_config(), preset=preset))
        assert algorithm.extended_config.iteration_level == 1
        algorithm.create_config(self._recurrence(create_base_config(), preset=preset, iteration_level=2))
        assert algorithm.extended_config.iteration_level == 2

    def test_explicit_block_defaults_to_level_one(self, algorithm):
        algorithm.create_config(self._recurrence(create_base_config(), enabled=True,
                                                 allow_hidden_to_hidden=True))
        assert algorithm.extended_config.iteration_level == 1

    def test_feedforward_preset_keeps_level_zero(self, algorithm):
        algorithm.create_config(self._recurrence(create_base_config(), preset='feedforward'))
        assert algorithm.extended_config.iteration_level == 0

    def test_substrate_config_follows_the_engine(self, algorithm):
        from emr_hyperneat.substrate import SubstrateConfig
        assert SubstrateConfig(input_coords=((0.0, -1.0),), output_coords=((0.0, 1.0),),
                               max_depth=2).iteration_level == 1
        algorithm.create_config(self._recurrence(create_base_config(), preset='hidden_only'))
        assert algorithm.substrate_config().iteration_level == 1


class TestHmrNameAliases:
    def test_cppn_num_outputs(self, algorithm):
        """HMR's runners read cppn_num_outputs; EMR names it num_cppn_outputs."""
        algorithm.create_config(create_base_config())
        assert algorithm.cppn_num_outputs == algorithm.num_cppn_outputs == 1


class TestActivateTimeResolution:
    """The settling steps follow the configuration.

    Either block sets them, `substrate.activate_time` or the recurrence block's value, and both must
    agree. With hidden-to-hidden connections on, a preset's own count or 2 (2**max_depth + 1)
    applies only when neither is set, and an explicit value must be an integer of at least 1;
    without them any value is accepted and unused. `set_recurrence_config` resolves the step count
    and `iteration_level` through the same rule.
    """

    RECURRENT = ['hidden_only', 'with_backward', 'with_lateral', 'with_self', 'full_recurrent']
    # Neither key set: full_recurrent's own 20, the others 2 (2**max_depth + 1) at create_base_config's depth 2
    DEFAULT_STEPS = {'hidden_only': 10, 'with_backward': 10, 'with_lateral': 10, 'with_self': 10,
                     'full_recurrent': 20}
    NOT_AN_INTEGER_OF_AT_LEAST_ONE = [True, False, np.bool_(True), 2.5, 20.0, '7', 0, -1]

    @staticmethod
    def _config(recurrence=None, substrate=None):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['recurrence'] = dict(recurrence or {})
        config['algorithm_params']['emrhyperneat']['substrate'].update(substrate or {})
        return config

    @pytest.mark.parametrize('preset', RECURRENT)
    def test_the_substrate_value_takes_effect(self, algorithm, preset):
        algorithm.create_config(self._config({'preset': preset}, {'activate_time': 7}))
        assert algorithm.activate_time == 7
        assert algorithm.substrate_config().activate_time == 7

    @pytest.mark.parametrize('preset', RECURRENT)
    def test_the_recurrence_value_takes_effect(self, algorithm, preset):
        algorithm.create_config(self._config({'preset': preset, 'activate_time': 7}))
        assert algorithm.activate_time == 7

    def test_the_explicit_block_takes_either_value(self, algorithm):
        block = {'enabled': True, 'allow_hidden_to_hidden': True}
        algorithm.create_config(self._config(block, {'activate_time': 7}))
        assert algorithm.activate_time == 7
        algorithm.create_config(self._config({**block, 'activate_time': 6}))
        assert algorithm.activate_time == 6

    @pytest.mark.parametrize('preset', RECURRENT)
    def test_both_blocks_agreeing(self, algorithm, preset):
        algorithm.create_config(self._config({'preset': preset, 'activate_time': 7}, {'activate_time': 7}))
        assert algorithm.activate_time == 7

    @pytest.mark.parametrize('preset', RECURRENT)
    def test_both_blocks_disagreeing_raise(self, algorithm, preset):
        with pytest.raises(ValueError, match=r"recurrence\.activate_time=3 and substrate\.activate_time=7"):
            algorithm.create_config(self._config({'preset': preset, 'activate_time': 3}, {'activate_time': 7}))

    @pytest.mark.parametrize('preset', RECURRENT)
    def test_neither_set_keeps_the_preset_count_or_the_derived_one(self, algorithm, preset):
        algorithm.create_config(self._config({'preset': preset}))
        assert algorithm.activate_time == self.DEFAULT_STEPS[preset]
        assert algorithm.iteration_level == 1
        assert algorithm.get_forward_mode() is ForwardPassMode.HYBRID_SPARSE_HH

    def test_neither_set_in_the_explicit_block(self, algorithm):
        algorithm.create_config(self._config({'enabled': True, 'allow_hidden_to_hidden': True}))
        assert algorithm.activate_time == 10

    @pytest.mark.parametrize('value', [7, True, 0, 2.5, '7'])
    def test_feed_forward_accepts_any_value_and_uses_none(self, algorithm, value):
        algorithm.create_config(self._config({'preset': 'feedforward'}, {'activate_time': value}))
        assert algorithm.activate_time == value
        assert algorithm.get_forward_mode() is ForwardPassMode.DENSE_ONLY
        assert algorithm.substrate_config().activate_time is None
        # Both blocks, disagreeing: feed-forward keeps the recurrence block's value, as before, and raises nothing
        algorithm.create_config(self._config({'preset': 'feedforward', 'activate_time': 3}, {'activate_time': value}))
        assert algorithm.activate_time == 3

    def test_feed_forward_without_either_key_is_unchanged(self, algorithm):
        algorithm.create_config(self._config({'preset': 'feedforward'}))
        assert (algorithm.activate_time, algorithm.iteration_level) == (5, 0)
        algorithm.create_config(self._config())
        assert (algorithm.activate_time, algorithm.iteration_level) == (5, 0)

    @pytest.mark.parametrize('value', NOT_AN_INTEGER_OF_AT_LEAST_ONE)
    @pytest.mark.parametrize('key', ['substrate', 'recurrence'])
    def test_a_value_that_is_not_an_integer_of_at_least_one_is_refused(self, algorithm, key, value):
        recurrence, substrate = {'preset': 'hidden_only'}, {}
        (recurrence if key == 'recurrence' else substrate)['activate_time'] = value
        with pytest.raises(ValueError, match=rf"{key}\.activate_time must be an integer of at least 1"):
            algorithm.create_config(self._config(recurrence, substrate))

    @pytest.mark.parametrize('key', ['substrate', 'recurrence'])
    def test_numpy_integers_are_accepted(self, algorithm, key):
        recurrence, substrate = {'preset': 'hidden_only'}, {}
        (recurrence if key == 'recurrence' else substrate)['activate_time'] = np.int64(7)
        algorithm.create_config(self._config(recurrence, substrate))
        assert algorithm.activate_time == 7 and type(algorithm.activate_time) is int

    @pytest.mark.parametrize('value', [True, np.bool_(True), 2.5, 20.0, '7', 0])
    def test_substrate_config_refuses_the_same_values_with_the_same_message(self, algorithm, value):
        from emr_hyperneat.substrate import SubstrateConfig
        config = SubstrateConfig(input_coords=((0.0, -1.0),), output_coords=((0.0, 1.0),), max_depth=2,
                                 hidden_to_hidden=True, activate_time=value)
        with pytest.raises(ValueError, match="activate_time must be an integer of at least 1") as api:
            config.validate()
        with pytest.raises(ValueError) as yaml:
            algorithm.create_config(self._config({'preset': 'hidden_only'}, {'activate_time': value}))
        assert str(yaml.value) == 'substrate.' + str(api.value)

    def test_substrate_config_accepts_numpy_integers_and_any_feed_forward_value(self):
        from emr_hyperneat.substrate import SubstrateConfig
        base = dict(input_coords=((0.0, -1.0),), output_coords=((0.0, 1.0),), max_depth=2)
        assert SubstrateConfig(**base, hidden_to_hidden=True, activate_time=np.int64(7)).validate().activate_time == 7
        for value in (True, 0, 2.5, '7'):
            SubstrateConfig(**base, hidden_to_hidden=False, activate_time=value).validate()

    def test_set_recurrence_config_sets_the_steps_and_the_level(self, algorithm):
        algorithm.create_config(self._config({'preset': 'hidden_only'}))
        algorithm.set_recurrence_config(EMRConfig(iteration_level=2, activate_time=3))
        assert (algorithm.activate_time, algorithm.iteration_level) == (3, 2)
        assert algorithm.get_forward_mode() is ForwardPassMode.HYBRID_SPARSE_HH
        assert algorithm.substrate_config().activate_time == 3

    def test_set_recurrence_config_without_steps_derives_them(self, algorithm):
        algorithm.create_config(self._config({'preset': 'feedforward'}))
        assert algorithm.activate_time == 5  # the feed-forward count, 2**max_depth + 1
        algorithm.set_recurrence_config(EMRConfig(iteration_level=1))
        assert (algorithm.activate_time, algorithm.iteration_level) == (10, 1)
        assert algorithm.substrate_config().activate_time == 10

    def test_set_recurrence_config_keeps_the_configured_substrate_value(self, algorithm):
        algorithm.create_config(self._config({'preset': 'hidden_only'}, {'activate_time': 7}))
        algorithm.set_recurrence_config(EMRConfig(iteration_level=2))
        assert (algorithm.activate_time, algorithm.iteration_level) == (7, 2)
        algorithm.set_recurrence_config(EMRConfig(iteration_level=1, activate_time=7))
        assert algorithm.activate_time == 7
        with pytest.raises(ValueError, match=r"activate_time=3 and substrate\.activate_time=7"):
            algorithm.set_recurrence_config(EMRConfig(iteration_level=1, activate_time=3))

    @pytest.mark.parametrize('value', [True, 2.5, 0])
    def test_set_recurrence_config_refuses_where_the_steps_act(self, algorithm, value):
        algorithm.create_config(self._config({'preset': 'hidden_only'}))
        with pytest.raises(ValueError, match="activate_time must be an integer of at least 1"):
            algorithm.set_recurrence_config(EMRConfig(iteration_level=1, activate_time=value))
        algorithm.set_recurrence_config(EMRConfig(enabled=False, allow_hidden_to_hidden=False, iteration_level=0,
                                                  activate_time=value))
        assert algorithm.get_forward_mode() is ForwardPassMode.DENSE_ONLY

    @pytest.mark.parametrize('fields, mode', [
        (dict(enabled=False, allow_hidden_to_hidden=True, iteration_level=1), ValueError),
        (dict(enabled=True, allow_hidden_to_hidden=False, iteration_level=1), ForwardPassMode.DENSE_ONLY),
        (dict(enabled=True, allow_hidden_to_hidden=True, iteration_level=0), ForwardPassMode.HYBRID_SPARSE_HH),
        (dict(enabled=True, allow_hidden_to_hidden=True, iteration_level=1), ForwardPassMode.HYBRID_SPARSE_HH),
        (dict(enabled=False, allow_hidden_to_hidden=False, iteration_level=0), ForwardPassMode.DENSE_ONLY),
    ])
    def test_set_recurrence_config_picks_the_forward_mode_as_the_configuration_does(self, algorithm, fields, mode):
        algorithm.create_config(self._config({'preset': 'hidden_only'}))
        configured = EMRHyperNEAT()
        if mode is ValueError:
            # Recurrence switched off with hidden-to-hidden connections on: both refuse, naming
            # both keys
            refusal = "enabled: false with allow_hidden_to_hidden: true"
            with pytest.raises(ValueError, match=refusal):
                algorithm.set_recurrence_config(EMRConfig(**fields))
            with pytest.raises(ValueError, match=refusal):
                configured.create_config(self._config(fields))
            return
        algorithm.set_recurrence_config(EMRConfig(**fields))
        assert algorithm.get_forward_mode() is mode
        configured.create_config(self._config(fields))
        assert configured.get_forward_mode() is mode

    def test_a_refused_switch_leaves_the_recurrence_as_it_was(self, algorithm):
        """set_recurrence_config refuses before it writes anything; an object without `enabled`
        counts as on."""
        from types import SimpleNamespace

        def recurrence():
            return (algorithm.extended_config, algorithm.get_forward_mode(),
                    algorithm.iteration_level, algorithm.activate_time)

        algorithm.create_config(self._config({'preset': 'hidden_only'}))
        before = recurrence()
        with pytest.raises(ValueError, match="enabled: false with allow_hidden_to_hidden: true"):
            algorithm.set_recurrence_config(EMRConfig(
                enabled=False, allow_hidden_to_hidden=True, iteration_level=2, activate_time=3))
        assert recurrence() == before
        algorithm.set_recurrence_config(SimpleNamespace(
            allow_hidden_to_hidden=True, iteration_level=1, activate_time=4))
        assert algorithm.get_forward_mode() is ForwardPassMode.HYBRID_SPARSE_HH

    def test_set_recurrence_config_before_create_config_derives_nothing(self):
        """A caller can set recurrence before create_config, when the depth is not known yet."""
        algorithm = EMRHyperNEAT()
        algorithm.set_recurrence_config(EMRConfig(iteration_level=1, activate_time=5))
        assert (algorithm.activate_time, algorithm.iteration_level) == (5, 1)
        fresh = EMRHyperNEAT()
        fresh.set_recurrence_config(EMRConfig(iteration_level=1))
        assert getattr(fresh, 'activate_time', None) is None and fresh.iteration_level == 1
        fresh.create_config(self._config({'preset': 'hidden_only'}))
        assert fresh.activate_time == 10

    def test_set_recurrence_config_reads_only_four_fields(self, algorithm):
        """A caller can pass HMR's UnifiedExtendedConfig rather than an EMRConfig."""
        from types import SimpleNamespace
        algorithm.create_config(self._config({'preset': 'feedforward'}))
        algorithm.set_recurrence_config(SimpleNamespace(enabled=True, allow_hidden_to_hidden=True,
                                                        iteration_level=1, activate_time=4))
        assert (algorithm.activate_time, algorithm.iteration_level) == (4, 1)
        assert algorithm.get_forward_mode() is ForwardPassMode.HYBRID_SPARSE_HH


class TestOneMinusMseRoutes:
    """EVAL_ONLY_PARALLEL and the FULL_PIPELINE_PARALLEL pipelines build their own weights and score
    1-MSE, as depth offload does, so a fitness mode, a locality penalty or a fitness rescaling they
    would drop is refused.
    """

    SETTINGS = [
        ('fitness_mode', 'accuracy'),
        ('locality_radius', 0.5),
        ('fitness_rescale_variance', 0.25),
    ]

    @pytest.mark.parametrize('key, value', SETTINGS)
    def test_eval_only_parallel(self, key, value):
        config = create_base_config()
        _emr(config)[key] = value
        _emr(config)['multi_gpu_strategy'] = 'EVAL_ONLY_PARALLEL'
        with pytest.raises(ValueError, match=f"EVAL_ONLY_PARALLEL .*would ignore {key}="):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('key, value', SETTINGS)
    def test_full_pipeline_parallel(self, monkeypatch, key, value):
        import jax
        from emr_hyperneat.emrhyperneat import MultiGPUStrategy
        from test_multi_gpu_strategies import ShardableProblem
        config = create_base_config(population_size=8)
        _emr(config)[key] = value
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.FULL_PIPELINE_PARALLEL)
        state = algorithm.initialize(
            algorithm.create_config(config), ShardableProblem(), seed=DEFAULT_SEED
        )
        cpu = jax.devices()[0]
        monkeypatch.setattr(jax, 'devices', lambda backend=None: [cpu, cpu])
        with pytest.raises(
            ValueError, match=f"FULL_PIPELINE_PARALLEL evaluates tanh.*would ignore {key}="
        ):
            algorithm.run_until_threshold(
                state, ShardableProblem(), target_fitness=2.0, max_generations=1
            )

    @pytest.mark.parametrize('variance', [0, -0.5])
    def test_fitness_rescale_variance_must_be_positive(self, variance):
        config = create_base_config()
        _emr(config)['fitness_rescale_variance'] = variance
        with pytest.raises(
            ValueError, match='fitness_rescale_variance must be a positive variance'
        ):
            EMRHyperNEAT().create_config(config)

    def test_connection_class_under_a_feedforward_preset(self):
        config = create_base_config()
        _emr(config)['recurrence'] = {'preset': 'feedforward', 'allow_lateral': True}
        with pytest.raises(
            ValueError, match="recurrence.allow_lateral is set but preset 'feedforward'"
        ):
            EMRHyperNEAT().create_config(config)


class TestNeatOperatorSettings:
    """neat_mutation, neat_species and neat_selection reach TensorNEAT.

    A configuration's values replace the defaults rather than being dropped: the neuromodulation
    runner's EMR backend asks for a compatibility threshold of 2.5, where the default is 3.0.
    """

    @staticmethod
    def _built(**blocks):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat'].update(blocks)
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        return algorithm.neat_algo.species_controller, algorithm.neat_algo.genome.mutation

    def test_values_reach_tensorneat(self):
        species, mutation = self._built(
            neat_species={'compatibility_threshold': 2.5, 'max_stagnation': 40,
                          'species_elitism': 3},
            neat_mutation={'conn_add_prob': 0.3, 'node_add_prob': 0.1},
            neat_selection={'genome_elitism': 5, 'survival_threshold': 0.3})
        assert (species.compatibility_threshold, species.max_stagnation,
                species.species_elitism) == (2.5, 40, 3)
        assert (species.genome_elitism, species.survival_threshold) == (5, 0.3)
        assert (mutation.conn_add, mutation.node_add, mutation.conn_delete,
                mutation.node_delete) == (0.3, 0.1, 0.5, 0.2)

    def test_without_the_keys_the_defaults_are_unchanged(self):
        species, mutation = self._built()
        assert (species.compatibility_threshold, species.max_stagnation,
                species.species_elitism) == (3.0, 20, 15)
        assert (species.genome_elitism, species.survival_threshold) == (15, 0.2)
        assert (mutation.conn_add, mutation.node_add, mutation.conn_delete,
                mutation.node_delete) == (0.5, 0.2, 0.5, 0.2)

    @pytest.mark.parametrize('block, key', [('neat_species', 'compat_threshold'),
                                            ('neat_mutation', 'add_prob'),
                                            ('neat_selection', 'elitism')])
    def test_an_unknown_setting_is_refused(self, block, key):
        with pytest.raises(ValueError, match=f'{block} has no setting'):
            self._built(**{block: {key: 1}})


def _emr(config):
    return config['algorithm_params']['emrhyperneat']['emr_hyperneat']


class TestNeuromodulationAndMultiTaskChecks:
    """The neuromodulation and multitask blocks refuse what EMR would not apply: an unknown key,
    preset or setting raises rather than being ignored, a preset runs with its modulation, and the
    parser passes on the multi-task settings the evaluation reads."""

    @pytest.mark.parametrize('block, match', [
        ({'mode': 'true_neuromodulation', 'num_nt': 4}, 'neuromodulation.num_nt is not a setting; .*num_nt_types'),
        ({'mode': 'true_neuromodulation', 'static_gating': True}, 'a level is chosen with neuromodulation.mode'),
        ({'preset': 'full'}, "combines the levels"),
        ({'preset': 'mystery'}, "neuromodulation.preset must be one of"),
        ({'enabled': True}, "enabled is true but no mode or preset"),
        ({'enabled': False, 'mode': 'true_neuromodulation'}, "enabled is false but a mode or preset is set"),
        ({'enabled': False, 'preset': 'static_gating'}, "enabled is false but a mode or preset is set"),
        ({'preset': 'xdg_style', 'mode': 'static_gating'}, "contradicts preset 'xdg_style'"),
        ({'mode': 'static_gating', 'branch_gating': True}, 'branch_gating is not implemented'),
    ])
    def test_neuromodulation_refusals(self, block, match):
        config = create_base_config()
        _emr(config)['neuromodulation'] = block
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('block', [{'enabled': False}, {'enabled': False, 'mode': 'disabled'},
                                       {'enabled': True, 'mode': 'static_gating'}, {'preset': 'disabled'}])
    def test_consistent_enabled_is_accepted(self, block):
        config = create_base_config()
        _emr(config)['neuromodulation'] = block
        EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('preset, mode', [('true_neuromodulation_4nt', 'true_neuromodulation'),
                                              ('xdg_style', 'context_gating'), ('static_gating', 'static_gating'),
                                              ('modulatory_only', 'modulatory_neurons')])
    def test_a_preset_sets_its_mode(self, preset, mode):
        """A preset gives level flags and no mode, and the evaluation dispatches on the mode, so the
        parser sets the preset's mode; use_self_connection_query is kept on that path."""
        config = create_base_config()
        _emr(config)['neuromodulation'] = {'preset': preset, 'use_self_connection_query': True}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm.neuromod_config.mode == mode
        assert algorithm.neuromod_config.use_self_connection_query

    def test_a_preset_runs_as_its_explicit_mode(self):
        def fitness(block):
            config = create_base_config(max_depth=3, population_size=20)
            _emr(config)['neuromodulation'] = block
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
            algorithm.run_generation(state, XORProblem())
            return np.asarray(algorithm._cached_fitnesses)
        preset = fitness({'preset': 'true_neuromodulation_4nt'})
        explicit = fitness({'mode': 'true_neuromodulation', 'num_nt_types': 4, 'receptor_from_weight': True,
                            'receptor_derivation': 'tanh', 'modulation_strength': 2.0})
        disabled = fitness({'mode': 'disabled'})
        np.testing.assert_array_equal(preset, explicit)
        assert not np.array_equal(preset, disabled)

    @pytest.mark.parametrize('block, match', [
        ({'enabled': False, 'num_tasks': 2}, 'multitask.enabled is not true'),
        ({'enabled': True, 'fitness_mode': 'f1'}, 'multitask.fitness_mode must be one of'),
        ({'enabled': True, 'fitness_aggregation': 'median'}, 'multitask.fitness_aggregation must be one of'),
        ({'enabled': True, 'generalist_bonus_type': 'max'}, 'multitask.generalist_bonus_type must be one of'),
        ({'enabled': True, 'two_module_mode': 'both'}, 'multitask.two_module_mode must be one of'),
        ({'enabled': True, 'modulation_mode': 'gain_only'}, 'multitask.modulation_mode must be one of'),
        ({'enabled': True, 'hidden_activation': 'swish'}, "unknown multi-task activation 'swish'"),
        ({'enabled': True, 'per_task_activation': {'xor': 'sine'}}, "unknown multi-task activation 'sine'"),
        ({'enabled': True, 'tasks': {}}, 'multitask.tasks is not a setting'),
    ])
    def test_multitask_refusals(self, block, match):
        config = create_base_config()
        _emr(config)['multitask'] = block
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_multitask_settings_reach_the_evaluation(self):
        """The multi-task evaluation reads these, so the parser must pass them on."""
        config = create_base_config()
        settings = {'fitness_mode': 'acc_mse', 'modulation_mode': 'gating_only', 'modulation_strength_override': 1.5,
                    'two_module_mode': 'parallel', 'generalist_bonus_type': 'min_bonus',
                    'generalist_bonus_weight': 0.2, 'generalist_threshold': 0.8, 'softmin_temperature': 0.3}
        _emr(config)['multitask'] = dict(enabled=True, num_tasks=2, **settings)
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        for key, value in settings.items():
            assert getattr(algorithm.multitask_config, key) == value, key

    @pytest.mark.parametrize('mode', ['acc_mse', 'accuracy_confident', 'ce', 'f1'])
    def test_top_level_fitness_mode(self, mode):
        config = create_base_config()
        _emr(config)['fitness_mode'] = mode
        if mode == 'f1':
            with pytest.raises(ValueError, match='emr_hyperneat.fitness_mode must be one of'):
                EMRHyperNEAT().create_config(config)
        else:
            EMRHyperNEAT().create_config(config)

    def test_acc_mse_and_accuracy_confident(self):
        """The single-task definitions equal the multi-task evaluators', not plain MSE."""
        import jax.numpy as jnp
        from emr_hyperneat.emrhyperneat import _eval_single_fitness, compute_fitness
        outputs = jnp.array([[0.9], [0.2], [0.6], [0.4]])
        targets = jnp.array([[1.0], [0.0], [0.0], [1.0]])
        accuracy, mse = 0.5, float(np.mean((np.asarray(outputs) - np.asarray(targets)) ** 2))
        confidence = float(np.mean(np.abs(np.asarray(outputs) - 0.5) * 2))
        for mode, expected in (('acc_mse', accuracy + 0.01 * (1 - mse)),
                               ('accuracy_confident', accuracy + 0.1 * confidence)):
            assert float(_eval_single_fitness(outputs, targets, mode)) == pytest.approx(expected, abs=1e-6)
            assert float(compute_fitness(outputs, targets, mode)) == pytest.approx(expected, abs=1e-6)

    def test_evaluate_with_neurotransmitter_refuses_an_unknown_activation(self):
        """An unknown activation name raises rather than being evaluated as tanh."""
        import jax.numpy as jnp
        config = create_config_with_neuromodulation('true_neuromodulation_4nt', recurrence_preset='feedforward',
                                                    population_size=10)
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
        algorithm.run_generation(state, XORProblem())
        with pytest.raises(ValueError, match="unknown activation 'sine'"):
            algorithm.evaluate_with_neurotransmitter(jnp.array([0.5, 0.5, 0.5, 1.0]), algorithm._cached_inputs,
                                                     algorithm._cached_targets, activation_name='sine')


class TestNoSilentDrops:
    """Keys EMR does not read, unknown values and contradictions raise, so that a run never goes
    ahead without what its configuration asked for."""

    @pytest.mark.parametrize('key, value, hint', [
        ('aggregation', {'mode': 'cppn_output'}, 'aggregation is its own block'),
        ('activation_palette', ['tanh', 'sin'], 'dynamic_functions.palette'),
        ('interp_method', 'sign', 'dynamic_functions.interpretation'),
        ('global_activation', 'sin', 'dynamic_functions.hidden_activation'),
        ('bogus', 1, 'the settings of dynamic_functions'),
    ])
    def test_unknown_dynamic_functions_keys(self, key, value, hint):
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': 'global', key: value}
        with pytest.raises(ValueError, match=f'dynamic_functions.{key} is not a setting; .*{hint}'):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('key, hint', [('aggregation_func', 'aggregation.global_function'),
                                           ('global_aggregation', 'aggregation.global_function'),
                                           ('bogus', 'the settings of aggregation')])
    def test_unknown_aggregation_keys(self, key, hint):
        config = create_base_config()
        _emr(config)['aggregation'] = {'mode': 'global', key: 'max'}
        with pytest.raises(ValueError, match=f'aggregation.{key} is not a setting; .*{hint}'):
            EMRHyperNEAT().create_config(config)

    def test_max_connections_warns_and_changes_nothing(self):
        """recurrence.max_connections is parsed and never read (the hidden-to-hidden cap is
        max_sparse_conns): it warns, and two values give the same generation bit for bit."""
        def fitness(value):
            config = create_config_with_recurrence(
                'hidden_only', population_size=10, hh_cache_enabled=False
            )
            _emr(config)['recurrence']['max_connections'] = value
            algorithm = EMRHyperNEAT()
            with pytest.warns(UserWarning, match=f'max_connections={value} has no effect'):
                configured = algorithm.create_config(config)
            state = algorithm.initialize(configured, XORProblem(), seed=DEFAULT_SEED)
            algorithm.run_generation(state, XORProblem())
            return np.asarray(algorithm._cached_fitnesses)
        np.testing.assert_array_equal(fitness(100), fitness(10000))

    def test_generation_loop_value_and_scope(self):
        """execution.generation_loop: another value than 'python' or 'while_loop' raises, and
        'while_loop' refuses at run_until_threshold what it cannot apply, naming it
        (test_generation_loop.py)."""
        config = create_base_config()
        _emr(config)['execution'] = {'generation_loop': 'while-loop'}
        with pytest.raises(ValueError,
                           match="execution.generation_loop must be 'python' or 'while_loop'"):
            EMRHyperNEAT().create_config(config)
        config = create_config_with_recurrence('hidden_only', population_size=10,
                                               hh_cache_enabled=False)
        _emr(config)['execution'] = {'generation_loop': 'while_loop'}
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(),
                                     seed=DEFAULT_SEED)
        with pytest.raises(ValueError, match='cannot apply: hidden-to-hidden connections'):
            algorithm.run_until_threshold(state, XORProblem(), target_fitness=2.0,
                                          max_generations=1)

    def test_live_strategy_needs_the_cppn_decode(self):
        """The live palette strategy (test_live_palette_strategy.py) sets the palette of the
        per-node CPPN decode, so with another mode it raises."""
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': 'global', 'live_strategy': {'enabled': True}}
        with pytest.raises(ValueError, match="needs dynamic_functions.mode 'cppn_output'"):
            EMRHyperNEAT().create_config(config)

    def test_palette_evolution_needs_the_cppn_decode(self):
        """Palette evolution (test_palette_evolution.py) evolves the per-node CPPN decode's palette,
        so with another mode it raises."""
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {
            'mode': 'disabled',
            'palette_evolution': {'enabled': False},
        }
        EMRHyperNEAT().create_config(config)
        _emr(config)['dynamic_functions']['palette_evolution'] = {'enabled': True}
        with pytest.raises(
            ValueError, match="palette_evolution evolves the palette of the per-node CPPN decode"
        ):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('block, nested', [
        ({'mode': 'cppn_output', 'cppn_output_indices': {'activations': 1}}, 'cppn_output_indices'),
        ({'mode': 'disabled', 'critical_periods': {'enable': True}}, 'critical_periods'),
        ({'mode': 'modular', 'modular': {'activation': 'sign'}}, 'modular'),
        ({'mode': 'modular', 'modular': {'sparsity': {'level0': 0.1}}}, 'modular.sparsity'),
    ])
    def test_unknown_nested_keys(self, block, nested):
        import warnings
        config = create_base_config()
        _emr(config)['dynamic_functions'] = block
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with pytest.raises(ValueError, match=f'dynamic_functions.{nested}.\\w+ is not a setting'):
                EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('block, match', [
        ({'mode': 'cppn_output', 'critical_periods': {'enabled': True}}, 'critical_periods is not implemented'),
    ])
    def test_blocks_nothing_reads(self, block, match):
        """Nothing applies these blocks, so they raise instead of being parsed and only printed."""
        config = create_base_config()
        _emr(config)['dynamic_functions'] = block
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_unknown_aggregation_mode_and_function(self):
        config = create_base_config()
        _emr(config)['aggregation'] = {'mode': 'maximum'}
        with pytest.raises(ValueError, match="aggregation.mode must be one of"):
            EMRHyperNEAT().create_config(config)
        _emr(config)['aggregation'] = {'mode': 'global', 'global_function': 'maximum'}
        with pytest.raises(ValueError, match="aggregation.global_function must be one of"):
            EMRHyperNEAT().create_config(config)
        _emr(config)['aggregation'] = {'mode': 'global', 'hh_aggregation_mode': 'max'}
        with pytest.raises(ValueError, match="aggregation.hh_aggregation_mode must be 'sum' or 'dynamic'"):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize(
        'section, block, match',
        [
            (
                'dynamic_functions',
                {'mode': 'weight_interpretation', 'interpretation': 'magnitde'},
                'dynamic_functions.interpretation must be one of',
            ),
            (
                'aggregation',
                {'mode': 'weight_interpretation', 'interpretation': 'sign'},
                'aggregation.interpretation must be one of',
            ),
            (
                'aggregation',
                {'mode': 'global', 'hh_aggregation_mode': 'max'},
                'hh_aggregation_mode must be',
            ),
            (
                'dynamic_functions',
                {'mode': 'modular', 'modular': {'activation_method': 'fan_in'}},
                'modular.activation_method must be one of',
            ),
            (
                'dynamic_functions',
                {'mode': 'modular', 'modular': {'sparsity_method': 'topk'}},
                'modular.sparsity_method must be one of',
            ),
            (
                'dynamic_functions',
                {'mode': 'modular', 'modular': {'scaling_method': 'linear'}},
                'modular.scaling_method must be one of',
            ),
            (
                'dynamic_functions',
                {'mode': 'modular', 'modular': {'aggregation_method': 'sign'}},
                'modular.aggregation_method must be one of',
            ),
            (
                'dynamic_functions',
                {'mode': 'modular', 'modular': {'activation': 'sign'}},
                'dynamic_functions.modular.activation is not a setting',
            ),
            (
                'dynamic_functions',
                {'mode': 'weight_interpretation', 'sparsity': {'level_3': 0.5}},
                'dynamic_functions.sparsity.level_3 is not a setting',
            ),
            (
                'dynamic_functions',
                {'mode': 'weight_interpretation', 'sparsity': {'level_0': 0.0}},
                'is the fraction of values kept',
            ),
            (
                'dynamic_functions',
                {'mode': 'modular', 'modular': {'sparsity': {'wta_k_percent': 1.5}}},
                'is the fraction of values kept',
            ),
        ],
    )
    def test_unknown_interpretations_and_modular_settings(self, section, block, match):
        """An unknown interpretation raises (aggregation's 'sign' is activation's name), and so does
        an unknown modular setting, now that modular is implemented."""
        config = create_base_config()
        _emr(config)[section] = block
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_modular_aggregation_is_quiet(self):
        """Modular aggregation is implemented, so it runs without a warning (0.1.0 ran
        sum)."""
        import warnings
        config = create_base_config()
        _emr(config)['aggregation'] = {'mode': 'modular'}
        _emr(config)['dynamic_functions'] = {
            'mode': 'disabled',
            'modular': {'aggregation_method': 'variance'},
        }
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('extra, what', [
        ({'recurrence': {'preset': 'hidden_only'}}, 'hidden-to-hidden recurrence'),
        ({'neuromodulation': {'mode': 'static_gating'}}, 'neuromodulation'),
    ])
    @pytest.mark.parametrize('dynamic_functions', [
        {'mode': 'weight_interpretation', 'interpretation': 'hierarchical_sparsity'},
        {'mode': 'modular', 'modular': {'sparsity_method': 'wta'}},
    ])
    def test_sparsity_is_feed_forward_only(self, dynamic_functions, extra, what):
        config = create_base_config()
        _emr(config)['dynamic_functions'] = dynamic_functions
        _emr(config).update(extra)
        with pytest.raises(
            ValueError, match=f'applies to the feed-forward evaluation only; .*{what}'
        ):
            EMRHyperNEAT().create_config(config)

    def test_stp_scaling_runs_with_recurrence(self):
        """Scaling multiplies the incoming weights, which the recurrent evaluation reads too."""
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'interpretation': 'stp_inspired',
        }
        _emr(config)['recurrence'] = {'preset': 'hidden_only'}
        EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize(
        'section, name', [('aggregation', 'sign'), ('aggregation', 'magnitude')]
    )
    def test_unknown_interpretations(self, section, name):
        """An aggregation interpretation EMR does not implement raises instead of running
        magnitude_bio ('sign' and 'magnitude' are activation interpretations)."""
        config = create_base_config()
        _emr(config)[section] = {'mode': 'weight_interpretation', 'interpretation': name}
        with pytest.raises(ValueError, match=f'{section}.interpretation must be one of'):
            EMRHyperNEAT().create_config(config)

    def test_hmr_shaped_configuration(self):
        config = create_base_config()
        block = config['algorithm_params'].pop('emrhyperneat')
        block['hmr_hyperneat'] = block.pop('emr_hyperneat')
        config['algorithm_params']['hmrhyperneat'] = block
        with pytest.raises(ValueError, match='shaped for the HMR engine'):
            EMRHyperNEAT().create_config(config)

    def test_recurrent_aggregation_fallback_follows_the_interpretation(self, monkeypatch):
        """Both sides of weight-interpretation aggregation, input-to-hidden and hidden-to-hidden,
        call aggregation_from_weight_interpretation with the configured interpretation, so
        sign_uniformity and fan_in reach the hidden-to-hidden side instead of magnitude_bio."""
        import emr_hyperneat.emrhyperneat as E
        calls = []
        original = E.aggregation_from_weight_interpretation

        def spy(W1, interpretation, num_aggregations, *decode):
            calls.append(interpretation)
            return original(W1, interpretation, num_aggregations, *decode)
        monkeypatch.setattr(E, 'aggregation_from_weight_interpretation', spy)
        config = create_config_with_recurrence('hidden_only', max_depth=3, population_size=20)
        _emr(config)['aggregation'] = {'mode': 'weight_interpretation', 'interpretation': 'fan_in',
                                       'hh_aggregation_mode': 'dynamic'}
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
        algorithm.run_generation(state, XORProblem())
        assert len(calls) >= 2 and set(calls) == {'fan_in'}, calls

    def test_depth_offload_refuses_leo(self):
        config = create_base_config()
        _emr(config)['leo'] = {'enabled': True}
        _emr(config)['execution'] = {'mode': 'depth_offload'}
        with pytest.raises(ValueError, match="would ignore leo"):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('key, hint', [('extended', 'emr_hyperneat.recurrence'),
                                           ('strategy', 'emr_hyperneat.multi_gpu_strategy'),
                                           ('num_cppn_outputs', 'dynamic_functions.num_cppn_outputs'),
                                           ('activate_time', 'recurrence.activate_time'),
                                           ('pop_size', 'population_size'),
                                           ('bogus', 'the settings of emr_hyperneat')])
    def test_unknown_emr_hyperneat_keys(self, key, hint):
        config = create_base_config()
        _emr(config)[key] = {'preset': 'hidden_only'} if key == 'extended' else 2
        with pytest.raises(ValueError, match=f'emr_hyperneat.{key} is not a setting; .*{hint}'):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('section, block', [
        ('leo', {'enabled': True, 'treshold': 0.3}),
        ('geometry_seeding', {'enabled': True, 'seed_weigth': -0.5}),
        ('execution', {'mode': 'standard', 'compaction': 'auto'}),
    ])
    def test_unknown_leo_geometry_and_execution_keys(self, section, block):
        """A key these three blocks do not read raises, so a misspelled threshold, seed weight or
        execution setting cannot run the default."""
        config = create_base_config()
        _emr(config)[section] = block
        misspelled = next(key for key in block if key not in ('enabled', 'mode'))
        with pytest.raises(
            ValueError, match=f'{section}.{misspelled} is not a setting; the settings of {section}'
        ):
            EMRHyperNEAT().create_config(config)

    def test_every_leo_geometry_and_execution_key_is_accepted(self):
        from emr_hyperneat.execution import (
            EXECUTION_KEYS,
        )
        algorithm = EMRHyperNEAT()
        assert algorithm.LEO_KEYS == {'enabled', 'threshold', 'mode', 'output_index'}
        assert algorithm.GEOMETRY_SEEDING_KEYS == {'enabled', 'seed_weight'}
        assert {
            'mode',
            'compaction_bucket',
            'skip_dead_levels',
            'eval_chunk_levels',
            'generation_loop',
        } <= EXECUTION_KEYS
        config = create_base_config()
        _emr(config)['leo'] = {
            'enabled': True,
            'threshold': 0.25,
            'mode': 'compound',
            'output_index': 1,
        }
        _emr(config)['geometry_seeding'] = {'enabled': True, 'seed_weight': -0.5}
        _emr(config)['execution'] = {
            'mode': 'standard',
            'compaction_bucket': 'auto',
            'skip_dead_levels': True,
        }
        algorithm.create_config(config)
        assert (algorithm.use_leo, algorithm.leo_threshold, algorithm.geometry_seeding_weight) == (
            True,
            0.25,
            -0.5,
        )

    @pytest.mark.parametrize('key, hint', [('recurrence_preset', 'recurrence.preset'),
                                           ('cache_enabled', 'recurrence.hh_cache_enabled'),
                                           ('bogus', 'the settings of recurrence')])
    def test_unknown_recurrence_keys(self, key, hint):
        config = create_base_config()
        _emr(config)['recurrence'] = {'preset': 'hidden_only', key: True}
        with pytest.raises(ValueError, match=f'recurrence.{key} is not a setting; .*{hint}'):
            EMRHyperNEAT().create_config(config)

    def test_preset_decides_whether_recurrence_is_on(self):
        config = create_base_config()
        _emr(config)['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
        }
        EMRHyperNEAT().create_config(config)
        _emr(config)['recurrence']['allow_hidden_to_hidden'] = False
        with pytest.raises(ValueError, match="contradicts preset 'hidden_only'"):
            EMRHyperNEAT().create_config(config)

    def test_connection_classes_without_recurrence(self):
        """allow_backward with hidden-to-hidden connections off raises, with no preset and under a
        feed-forward preset, instead of leaving the block feed-forward."""
        config = create_base_config()
        _emr(config)['recurrence'] = {'allow_backward': True, 'iteration_level': 2}
        with pytest.raises(ValueError, match='recurrence.allow_backward is set but hidden-to-hidden'):
            EMRHyperNEAT().create_config(config)
        _emr(config)['recurrence'] = {'preset': 'feedforward', 'allow_lateral': True}
        with pytest.raises(ValueError, match="recurrence.allow_lateral is set but preset 'feedforward'"):
            EMRHyperNEAT().create_config(config)
        _emr(config)['recurrence'] = {'enabled': False, 'allow_backward': False, 'eval_chunk_size': 8}
        EMRHyperNEAT().create_config(config)
        _emr(config)['recurrence'] = {'preset': 'hidden_only', 'allow_self_loops': True}
        EMRHyperNEAT().create_config(config)

    def test_top_level_iteration_level(self):
        config = create_base_config()
        _emr(config)['iteration_level'] = 1
        EMRHyperNEAT().create_config(config)
        _emr(config)['iteration_level'] = 2
        with pytest.raises(ValueError, match='the multi-hop level is recurrence.iteration_level'):
            EMRHyperNEAT().create_config(config)

    def test_unknown_multi_gpu_strategy(self):
        config = create_base_config()
        _emr(config)['multi_gpu_strategy'] = 'SINGLE_GPUS'
        with pytest.raises(ValueError, match="Unknown multi_gpu_strategy 'SINGLE_GPUS'"):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('variance', [0, -0.5])
    def test_fitness_rescale_variance_must_be_positive(self, variance):
        """A variance that is not positive raises; ignoring it would score the run 1-MSE."""
        config = create_base_config()
        _emr(config)['fitness_rescale_variance'] = variance
        with pytest.raises(ValueError, match='fitness_rescale_variance must be a positive variance'):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('palette', [[0, 18], [0, -1], [], 'not_a_preset', [0, 'gaussian'], [True, 4]])
    def test_invalid_palettes(self, palette):
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': 'cppn_output', 'palette': palette}
        with pytest.raises(ValueError, match='palette'):
            EMRHyperNEAT().create_config(config)

    def test_palette_names_map_to_indices(self):
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': 'cppn_output', 'palette': ['tanh', 'sin', 13]}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert np.asarray(algorithm.df_palette).tolist() == [0, 4, 13]

    @pytest.mark.parametrize('section', ['dynamic_functions', 'substrate'])
    @pytest.mark.parametrize('key', ['hidden_activation', 'output_activation'])
    def test_unknown_activation_names(self, section, key):
        """An unknown name raises instead of running tanh (hidden) or sigmoid (output)."""
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': 'global'}
        block = (_emr(config)['dynamic_functions'] if section == 'dynamic_functions'
                 else config['algorithm_params']['emrhyperneat']['substrate'])
        block[key] = 'elu'
        with pytest.raises(ValueError, match=f"{key} 'elu' .* is not an activation function"):
            EMRHyperNEAT().create_config(config)

    def test_substrate_activations_feed_global_mode(self):
        """As in HMR's dynamic-functions variant, the substrate section's activations are the defaults
        of dynamic_functions.hidden_activation and output_activation."""
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': 'global'}
        config['algorithm_params']['emrhyperneat']['substrate'].update(
            {'hidden_activation': 'sin', 'output_activation': 'tanh'})
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert (algorithm.df_hidden_activation, algorithm.df_output_activation) == ('sin', 'tanh')
        assert algorithm.substrate_config().hidden_activation == 'sin'
        assert algorithm.substrate_config().output_activation == 'tanh'

    @pytest.mark.parametrize('mode', ['disabled', 'cppn_output'])
    def test_hidden_activation_outside_global_mode(self, mode):
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': mode, 'hidden_activation': 'sin'}
        with pytest.raises(ValueError, match="applies in dynamic_functions.mode 'global' only"):
            EMRHyperNEAT().create_config(config)

    def test_output_activation_on_the_plain_path(self):
        """A configured output activation applies on the plain path (HMR's dynamic-functions variant
        applied it; 0.1.0 ignored it on this path): the run takes mode 'global' with a tanh hidden
        activation, which equals the plain evaluation bit for bit (test_hmr_names.py). With
        per-node aggregation the dynamic evaluation applies it."""
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': 'disabled', 'output_activation': 'tanh'}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert (algorithm.dynamic_functions_mode, algorithm.df_hidden_activation,
                algorithm.df_output_activation) == ('global', 'tanh', 'tanh')
        _emr(config)['aggregation'] = {'mode': 'global', 'global_function': 'max'}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm.substrate_config().output_activation == 'tanh'

    @pytest.mark.parametrize(
        'block, match',
        [
            # gate_from_cppn is HMR's name of gate_source 'cppn' (TestCppnNeuromodulation); in
            # another mode it raises
            (
                {'mode': 'true_neuromodulation', 'gate_from_cppn': True},
                "gate_source 'cppn' gates connections",
            ),
            (
                {'mode': 'true_neuromodulation', 'num_nt': 4},
                'neuromodulation.num_nt is not a setting; .*num_nt_types',
            ),
            (
                {'mode': 'true_neuromodulation', 'static_gating': True},
                'a level is chosen with neuromodulation.mode',
            ),
            ({'mode': 'true_neuromodulation_fancy'}, "neuromodulation.mode must be one of"),
            ({'mode': 'full'}, "evaluated Level 3 only"),
            ({'mode': 'xdg_style', 'gate_source': 'weights'}, "contradicts HMR's mode 'xdg_style'"),
            ({'preset': 'full'}, "combines the levels"),
            ({'preset': 'mystery'}, "neuromodulation.preset must be one of"),
            ({'enabled': True}, "enabled is true but no mode or preset"),
            ({'preset': 'xdg_style', 'mode': 'static_gating'}, "contradicts preset 'xdg_style'"),
            ({'mode': 'static_gating', 'branch_gating': True}, 'branch_gating is not implemented'),
        ],
    )
    def test_neuromodulation_refusals(self, block, match):
        config = create_base_config()
        _emr(config)['neuromodulation'] = block
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize(
        'preset, mode',
        [
            ('true_neuromodulation_4nt', 'true_neuromodulation'),
            ('xdg_style', 'context_gating'),
            ('static_gating', 'static_gating'),
            ('modulatory_only', 'modulatory_neurons'),
        ],
    )
    def test_a_preset_sets_its_mode(self, preset, mode):
        """EMR's presets set level flags and no mode, and evaluation dispatches on the mode, so a
        preset sets its mode, or it would run without modulation."""
        config = create_base_config()
        _emr(config)['neuromodulation'] = {'preset': preset, 'use_self_connection_query': True}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm.neuromod_config.mode == mode
        assert algorithm.neuromod_config.use_self_connection_query      # kept through the preset

    def test_a_preset_runs_as_its_explicit_mode(self):
        def fitness(block):
            config = create_base_config(max_depth=3, population_size=20)
            _emr(config)['neuromodulation'] = block
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(
                algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED
            )
            algorithm.run_generation(state, XORProblem())
            return np.asarray(algorithm._cached_fitnesses)
        preset = fitness({'preset': 'true_neuromodulation_4nt'})
        explicit = fitness(
            {
                'mode': 'true_neuromodulation',
                'num_nt_types': 4,
                'receptor_from_weight': True,
                'receptor_derivation': 'tanh',
                'modulation_strength': 2.0,
            }
        )
        disabled = fitness({'mode': 'disabled'})
        np.testing.assert_array_equal(preset, explicit)
        assert not np.array_equal(preset, disabled)

    @pytest.mark.parametrize(
        'block, match',
        [
            ({'enabled': False, 'num_tasks': 2}, 'multitask.enabled is not true'),
            ({'enabled': True, 'fitness_mode': 'f1'}, 'multitask.fitness_mode must be one of'),
            (
                {'enabled': True, 'fitness_aggregation': 'median'},
                'multitask.fitness_aggregation must be one of',
            ),
            (
                {'enabled': True, 'generalist_bonus_type': 'max'},
                'multitask.generalist_bonus_type must be one of',
            ),
            (
                {'enabled': True, 'two_module_mode': 'both'},
                'multitask.two_module_mode must be one of',
            ),
            ({'enabled': True, 'tasks': {}}, 'multitask.tasks is not a setting'),
        ],
    )
    def test_multitask_refusals(self, block, match):
        config = create_base_config()
        _emr(config)['multitask'] = block
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_multitask_settings_reach_the_evaluation(self):
        """The multi-task evaluation reads these, so the parser passes them on."""
        config = create_base_config()
        settings = {
            'fitness_mode': 'acc_mse',
            'modulation_mode': 'gating_only',
            'modulation_strength_override': 1.5,
            'two_module_mode': 'parallel',
            'generalist_bonus_type': 'min_bonus',
            'generalist_bonus_weight': 0.2,
            'generalist_threshold': 0.8,
            'softmin_temperature': 0.3,
        }
        _emr(config)['multitask'] = dict(enabled=True, num_tasks=2, **settings)
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        for key, value in settings.items():
            assert getattr(algorithm.multitask_config, key) == value, key

    @pytest.mark.parametrize('mode', ['acc_mse', 'accuracy_confident', 'f1'])
    def test_top_level_fitness_mode(self, mode):
        config = create_base_config()
        _emr(config)['fitness_mode'] = mode
        if mode == 'f1':
            with pytest.raises(ValueError, match='emr_hyperneat.fitness_mode must be one of'):
                EMRHyperNEAT().create_config(config)
        else:
            EMRHyperNEAT().create_config(config)

    def test_hmr_fitness_modes(self):
        """acc_mse and accuracy_confident as HMR's neuromodulation variant defined them (they
        computed MSE here)."""
        import jax.numpy as jnp
        from emr_hyperneat.emrhyperneat import (
            _eval_single_fitness,
        )
        outputs = jnp.array([[0.9], [0.2], [0.6], [0.4]])
        targets = jnp.array([[1.0], [0.0], [0.0], [1.0]])
        accuracy, mse = 0.5, float(np.mean((np.asarray(outputs) - np.asarray(targets)) ** 2))
        assert (
            abs(
                float(_eval_single_fitness(outputs, targets, 'acc_mse'))
                - (accuracy + 0.01 * (1 - mse))
            )
            < 1e-6
        )
        confidence = float(np.mean(np.abs(np.asarray(outputs) - 0.5) * 2))
        assert abs(float(_eval_single_fitness(outputs, targets, 'accuracy_confident'))
                   - (accuracy + 0.1 * confidence)) < 1e-6

    def test_hmrs_option_a_mode_names(self):
        """HMR's Option A mode names are EMR's true_neuromodulation with the name's NT count and
        receptor derivation and HMR's receptor source; the run equals the explicit EMR block bit for
        bit."""
        def run(block):
            config = create_base_config(max_depth=3, population_size=20)
            _emr(config)['neuromodulation'] = block
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(
                algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED
            )
            algorithm.run_generation(state, XORProblem())
            return algorithm, np.asarray(algorithm._cached_fitnesses)
        named, named_fitness = run(
            {'mode': 'true_neuromodulation_4nt_option_a_tanh', 'modulation_strength': 5.0}
        )
        explicit, explicit_fitness = run(
            {
                'mode': 'true_neuromodulation',
                'num_nt_types': 4,
                'receptor_from_weight': True,
                'receptor_derivation': 'tanh',
                'use_self_connection_query': True,
                'modulation_strength': 5.0,
            }
        )
        assert named.neuromod_config.mode == 'true_neuromodulation'
        assert (named.neuromod_config.num_nt_types, named.neuromod_config.receptor_derivation) == (
            4,
            'tanh',
        )
        assert named.neuromod_config.use_self_connection_query
        np.testing.assert_array_equal(named_fitness, explicit_fitness)

    @pytest.mark.parametrize(
        'name, num_nt, derivation',
        [
            ('true_neuromodulation_option_a_abs', 2, 'abs'),
            ('true_neuromodulation_3nt_option_a_tanh', 3, 'tanh'),
            ('true_neuromodulation_4nt_option_a_fourier', 4, 'fourier'),
            ('true_neuromodulation_6nt_option_a_tanh', 6, 'tanh'),
        ],
    )
    def test_every_option_a_name_maps(self, name, num_nt, derivation):
        config = create_base_config()
        _emr(config)['neuromodulation'] = {'mode': name}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert (
            algorithm.neuromod_config.num_nt_types,
            algorithm.neuromod_config.receptor_derivation,
        ) == (num_nt, derivation)

    def test_hmrs_custom_mode(self):
        config = create_base_config()
        _emr(config)['neuromodulation'] = {
            'mode': 'custom',
            'enabled': True,
            'true_neuromodulation': True,
            'num_nt_types': 3,
            'modulation_strength': 4.0,
            'receptor_from_weight': True,
            'receptor_derivation': 'tanh',
        }
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        config_nm = algorithm.neuromod_config
        assert (config_nm.mode, config_nm.num_nt_types, config_nm.modulation_strength) == (
            'true_neuromodulation', 3, 4.0)

    @pytest.mark.parametrize(
        'block, match',
        [
            (
                {'mode': 'true_neuromodulation_4nt_option_a_tanh', 'num_nt_types': 5},
                "contradicts HMR's mode",
            ),
            # HMR's Option B names translate (test_neuromodulation.py TestOptionB); a contradiction
            # raises
            (
                {'mode': 'true_neuromodulation_4nt', 'num_nt_types': 2},
                "is HMR's Option B with four NT types",
            ),
            # HMR evaluated the highest level of a 'custom' combination (test_custom_levels.py)
            (
                {'mode': 'custom', 'static_gating': True, 'true_neuromodulation': True},
                'HMR evaluated true_neuromodulation alone',
            ),
        ],
    )
    def test_hmr_mode_name_refusals(self, block, match):
        config = create_base_config()
        _emr(config)['neuromodulation'] = block
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_population_parallel_process_refuses_per_node_functions(self, monkeypatch):
        import jax
        from emr_hyperneat.emrhyperneat import MultiGPUStrategy
        config = create_base_config()
        _emr(config)['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy.POPULATION_PARALLEL_PROCESS)
        state = algorithm.initialize(
            algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED
        )
        cpu = jax.devices()[0]
        monkeypatch.setattr(jax, 'devices', lambda backend=None: [cpu, cpu])
        with pytest.raises(ValueError, match='POPULATION_PARALLEL_PROCESS evaluates tanh'):
            algorithm.run_until_threshold(
                state, XORProblem(), target_fitness=2.0, max_generations=1
            )

    @pytest.mark.parametrize('section, block', [
        ('geometry_seeding', {'enabled': True, 'seed_weigth': -0.5}),
        ('geometry_seeding', {'enabled': True, 'use_7d_inputs': False}),
        ('execution', {'mode': 'standard', 'compaction': 'auto'}),
    ])
    def test_unknown_geometry_and_execution_keys(self, section, block):
        config = create_base_config()
        _emr(config)[section] = block
        misspelled = next(key for key in block if key not in ('enabled', 'mode'))
        with pytest.raises(ValueError, match=f'{section}.{misspelled} is not a setting'):
            EMRHyperNEAT().create_config(config)

    def test_every_geometry_and_execution_key_is_accepted(self):
        from emr_hyperneat.execution import EXECUTION_KEYS
        algorithm = EMRHyperNEAT()
        assert algorithm.GEOMETRY_SEEDING_KEYS == {'enabled', 'seed_weight'}
        assert {'mode', 'compaction_bucket', 'skip_dead_levels', 'eval_chunk_levels'} <= EXECUTION_KEYS
        config = create_base_config()
        _emr(config)['geometry_seeding'] = {'enabled': True, 'seed_weight': -0.5}
        _emr(config)['execution'] = {'mode': 'standard', 'compaction_bucket': 'auto', 'skip_dead_levels': True}
        algorithm.create_config(config)
        assert algorithm.geometry_seeding_enabled


def _silently(config):
    """create_config with every UserWarning an error."""
    import warnings
    algorithm = EMRHyperNEAT()
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        algorithm.create_config(config)
    return algorithm


class TestUnknownKeysOutsideEmrHyperneatRaise:
    """The levels outside emr_hyperneat (the top level, algorithm_params,
    algorithm_params.emrhyperneat, substrate, swarm, neat, cppn) raise on a key nothing reads, naming
    the right key when it is known, as the blocks inside emr_hyperneat do. 0.1.0 ignored
    such a key without a word."""

    @pytest.mark.parametrize('level, key, hint', [
        ('', 'neat', 'EMR reads algorithm_params.emrhyperneat.neat'),
        ('', 'emr_hyperneat', 'EMR reads algorithm_params.emrhyperneat.emr_hyperneat'),
        ('', 'bogus', 'the settings of the top level are'),
        ('algorithm_params', 'population_size',
         'EMR reads algorithm_params.emrhyperneat.population_size'),
        ('algorithm_params', 'generations', "run_until_threshold's max_generations"),
        ('algorithm_params', 'hmrhyperneat', 'the block of the HMR engine'),
        ('algorithm_params.emrhyperneat', 'pop_size', 'population_size here, or neat.pop_size'),
        ('algorithm_params.emrhyperneat', 'hmr_hyperneat', "HMR's block name"),
        ('algorithm_params.emrhyperneat', 'bogus',
         'the settings of algorithm_params.emrhyperneat are'),
    ])
    def test_outer_levels(self, level, key, hint):
        config = create_base_config()
        block = {'': config, 'algorithm_params': config['algorithm_params'],
                 'algorithm_params.emrhyperneat': config['algorithm_params']['emrhyperneat']}[level]
        block[key] = {}
        where = f'{level}.{key}' if level else key
        with pytest.raises(ValueError, match=f'^{where} is not a setting; .*{hint}'):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('block, key, hint', [
        ('substrate', 'input_coord', 'the settings of substrate are'),
        ('swarm', 'agents', 'the settings of swarm are'),
        ('neat', 'conn_add_prob', 'the setting is neat_mutation.conn_add_prob'),
        ('neat', 'compatibility_threshold', 'the setting is neat_species.compatibility_threshold'),
        ('neat', 'survival_threshold', 'the setting is neat_selection.survival_threshold'),
        ('neat', 'species_threshold',
         'the speciation threshold is neat_species.compatibility_threshold'),
        ('cppn', 'genomes', 'the settings of cppn are'),
    ])
    def test_blocks(self, block, key, hint):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat'].setdefault(block, {})[key] = 1
        match = f'^{block}.{key} is not a setting; .*{hint}'
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('block, key', [('genome', 'num_input'), ('activation', 'option')])
    def test_cppn_blocks(self, block, key):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['cppn'] = {block: {key: 1}}
        match = f'^cppn.{block}.{key} is not a setting; the settings of cppn.{block}'
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_the_error_names_the_other_keys(self):
        config = create_base_config()
        substrate = config['algorithm_params']['emrhyperneat']['substrate']
        substrate.update(input_coord=1, output_coord=1)
        with pytest.raises(ValueError, match=r"also not settings: \['output_coord'\]"):
            EMRHyperNEAT().create_config(config)

    def test_flat_configuration(self):
        """Without algorithm_params.emrhyperneat the top level is that block (both read sites)."""
        flat = create_base_config()['algorithm_params']['emrhyperneat']
        _silently(flat)
        flat['bogus'] = 1
        match = r"^bogus is not a setting; the settings of the top level .*'emr_hy"
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(flat)

    def test_every_key_the_read_sites_read_is_silent(self):
        config = create_base_config()
        algo = config['algorithm_params']['emrhyperneat']
        algo['substrate'].update(activate_time=5)
        algo['swarm'] = {'enabled': False, 'num_agents': 1, 'team_layout': 'line'}
        algo['neat'] = {'pop_size': 8, 'population_size': 8, 'species_size': 10}
        algo['cppn'] = {'genome': {'num_inputs': 5, 'num_outputs': 1},
                        'activation': {'options': ['tanh', 'sin'], 'default': 'tanh',
                                       'mutate_rate': 0.5}}
        algo['neat_species'] = {'compatibility_threshold': 3.0}
        config['config_file'] = None
        _silently(config)

    def test_the_suite_configurations_are_silent(self):
        """Every factory of conftest with every preset it takes."""
        from conftest import (GEOMETRY_SEEDING_PRESETS, create_config_with_combined_recurrence,
                              create_config_with_geometry_seeding, create_config_with_multi_task,
                              create_config_with_streaming)
        configs = [create_base_config(), create_config_with_streaming(),
                   create_config_with_streaming(enable_streaming=False),
                   create_config_with_multi_task(['xor', 'and']),
                   create_config_with_combined_recurrence()]
        configs += [create_config_with_recurrence(p) for p in RECURRENCE_PRESETS]
        configs += [create_config_with_dynamic_functions(p) for p in DYNAMIC_FUNCTION_PRESETS]
        configs += [create_config_with_aggregation(p) for p in AGGREGATION_PRESETS]
        configs += [create_config_with_neuromodulation(p) for p in NEUROMODULATION_PRESETS]
        configs += [create_config_with_geometry_seeding(p) for p in GEOMETRY_SEEDING_PRESETS]
        for config in configs:
            _silently(config)

    def test_a_per_node_paper_runner_configuration_is_silent(self):
        """The shape of emr-dynamic-functions E2 (E1, E7, E8 and E9 alike), which carries the inert
        neat.species_size."""
        config = {'algorithm_params': {'emrhyperneat': {
            'emr_hyperneat': {
                'extra_randkey_split': True, 'initial_depth': 0, 'variance_rule': 'legacy',
                'max_depth': 2, 'variance_threshold': 0.03,
                'dynamic_functions': {'mode': 'global', 'hidden_activation': 'sin', 'palette': [4],
                                      'palette_evolution': {'enabled': False}},
            },
            'substrate': {'input_coords': [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],
                          'output_coords': [(0.0, 1.0)]},
            'neat': {'pop_size': 150, 'species_size': 10},
        }}}
        _silently(config)

    def test_hmr_shaped_configuration_raises_without_a_warning_first(self):
        """Without an emr_hyperneat block nothing is checked here: the base class refuses the
        configuration (or warns that every setting takes its default)."""
        import warnings
        config = create_base_config()
        block = config['algorithm_params'].pop('emrhyperneat')
        block['hmr_hyperneat'] = block.pop('emr_hyperneat')
        config['algorithm_params']['hmrhyperneat'] = block
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            with pytest.raises(ValueError, match='shaped for the HMR engine'):
                EMRHyperNEAT().create_config(config)
        assert not [w for w in record if 'is not a setting' in str(w.message)]


class TestFileConfigurationRefused:
    """A top-level preset, config_file or overrides asked the vendored ConfigManager for a
    configuration loaded from a file, and none ships: every release raised ConfigError after two
    warnings. They raise a ValueError first, saying what to pass instead; set to None or empty they
    do nothing, as before."""

    @pytest.mark.parametrize('key, value', [
        ('preset', 'default'), ('config_file', 'experiment.yaml'),
        ('overrides', {'neat': {'pop_size': 8}})])
    def test_refused_before_anything_else(self, key, value):
        import warnings
        config = create_base_config()
        _emr(config)['bogus'] = 1   # a later check would raise "bogus is not a setting"
        config[key] = value
        algorithm = EMRHyperNEAT()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            match = f'^top-level {key} asks for a configuration loaded from a file, and none ships'
            with pytest.raises(ValueError, match=match):
                algorithm.create_config(config)
        assert not record

    def test_the_base_class_refuses_too(self):
        from emr_hyperneat.emrhyperneat_base import EMRHyperNEATChunking
        with pytest.raises(ValueError, match='^top-level preset and config_file ask for'):
            EMRHyperNEATChunking.create_config(EMRHyperNEAT(), {'preset': 'default',
                                                                'config_file': 'x.yaml'})

    @pytest.mark.parametrize('key, value', [
        ('preset', None), ('config_file', ''), ('overrides', {})])
    def test_empty_values_do_nothing(self, key, value):
        config = create_base_config()
        config[key] = value
        _silently(config)


class TestOneMinusMseEvaluations:
    """Depth offload, POPULATION_PARALLEL_PROCESS, EVAL_ONLY_PARALLEL and the FULL_PIPELINE_PARALLEL
    pipelines score 1-MSE on weights they build without the locality penalty, and apply no fitness
    rescaling, so these settings raise there instead of being dropped without a word."""

    SETTINGS = [('fitness_mode', 'accuracy'), ('locality_radius', 0.5), ('fitness_rescale_variance', 0.25)]

    @pytest.mark.parametrize('key, value', SETTINGS)
    def test_depth_offload(self, key, value):
        config = create_base_config()
        _emr(config)[key] = value
        _emr(config)['execution'] = {'mode': 'depth_offload'}
        with pytest.raises(ValueError, match=f"depth_offload' .*would ignore {key}="):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('key, value', SETTINGS)
    def test_eval_only_parallel(self, key, value):
        config = create_base_config()
        _emr(config)[key] = value
        _emr(config)['multi_gpu_strategy'] = 'EVAL_ONLY_PARALLEL'
        with pytest.raises(ValueError, match=f"EVAL_ONLY_PARALLEL .*would ignore {key}="):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('strategy, problem, match', [
        ('POPULATION_PARALLEL_PROCESS', 'xor', 'POPULATION_PARALLEL_PROCESS evaluates tanh'),
        ('FULL_PIPELINE_PARALLEL', 'shardable', 'FULL_PIPELINE_PARALLEL evaluates tanh'),
    ])
    @pytest.mark.parametrize('key, value', SETTINGS)
    def test_multi_device_routes(self, monkeypatch, strategy, problem, match, key, value):
        """Checked at dispatch on two or more devices (one device runs the standard loop)."""
        import jax
        from emr_hyperneat.emrhyperneat_base import MultiGPUStrategy
        from test_multi_gpu_strategies import ShardableProblem
        task = XORProblem() if problem == 'xor' else ShardableProblem()
        config = create_base_config(population_size=8)
        _emr(config)[key] = value
        algorithm = EMRHyperNEAT(strategy=MultiGPUStrategy[strategy])
        state = algorithm.initialize(algorithm.create_config(config), task, seed=DEFAULT_SEED)
        cpu = jax.devices()[0]
        monkeypatch.setattr(jax, 'devices', lambda backend=None: [cpu, cpu])
        with pytest.raises(ValueError, match=f"{match}.*would ignore {key}="):
            algorithm.run_until_threshold(state, task, target_fitness=2.0, max_generations=1)
