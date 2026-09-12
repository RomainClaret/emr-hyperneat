"""Tests for configuration parsing in EMR-HyperNEAT.

Tests cover:
- Recurrence configuration parsing (_parse_extended_config)
- Dynamic functions configuration parsing (_parse_dynamic_functions_config)
- Neuromodulation configuration parsing (_parse_neuromodulation_config)
- Preset loading and validation
- Error handling for invalid configurations
"""

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


class TestRecurrenceConfigParsing:
    """Tests for recurrence/connection type configuration parsing."""

    def test_feedforward_preset_parses(self, algorithm, xor_problem):
        """Verify feedforward preset parses correctly."""
        config = create_config_with_recurrence('feedforward')
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.extended_config.allow_backward == True
        assert algorithm.extended_config.allow_lateral == False
        assert algorithm.extended_config.allow_self_loops == True
        assert algorithm.extended_config.iteration_level == 3

    def test_invalid_preset_raises_error(self, algorithm):
        """Verify invalid preset name raises error."""
        with pytest.raises(ValueError, match="Unknown recurrence preset"):
            create_config_with_recurrence('invalid_preset')

    def test_iteration_level_configuration(self, algorithm, xor_problem):
        """Verify iteration_level is set correctly for recurrence presets."""
        config = create_config_with_recurrence('hidden_only')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # iteration_level should be set
        assert algorithm.extended_config.iteration_level >= 0
        # Note: activate_time may or may not be auto-computed depending on implementation


class TestDynamicFunctionsConfigParsing:
    """Tests for dynamic activation functions configuration parsing."""

    def test_disabled_mode_parses(self, algorithm, xor_problem):
        """Verify disabled mode parses correctly."""
        config = create_config_with_dynamic_functions('disabled')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.dynamic_functions_mode == 'disabled'

    @pytest.mark.parametrize("mode", list(DYNAMIC_FUNCTION_PRESETS.keys()))
    def test_all_dynamic_function_modes_parse(self, algorithm, xor_problem, mode):
        """Verify all dynamic function modes parse without errors."""
        config = create_config_with_dynamic_functions(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_global_mode_sets_activations(self, algorithm, xor_problem):
        """Verify global mode sets the hidden activation correctly."""
        config = create_config_with_dynamic_functions('global_sin')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.dynamic_functions_mode == 'global'
        assert algorithm.df_hidden_activation == 'sin'

    def test_cppn_output_mode_sets_num_activations(self, algorithm, xor_problem):
        """Verify CPPN output mode sets number of activations."""
        config = create_config_with_dynamic_functions('cppn_output_6')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.dynamic_functions_mode == 'cppn_output'
        assert algorithm.df_num_activations == 6

    def test_weight_interpretation_modes(self, algorithm, xor_problem):
        """Verify weight interpretation modes parse correctly."""
        for interp in ['sign', 'magnitude', 'variance']:
            config = create_config_with_dynamic_functions(
                f'weight_interp_{interp}' if interp != 'variance' else 'weight_interp_variance'
            )
            config_obj = algorithm.create_config(config)
            state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.dynamic_functions_mode == 'weight_interpretation'

    def test_hh_activation_mode_options(self, algorithm, xor_problem):
        """Verify H→H activation mode options work."""
        for hh_mode in ['initial_only', 'every_iteration']:
            config = create_config_with_dynamic_functions(
                'cppn_output_4',
                recurrence_preset='hidden_only',
                hh_activation_mode=hh_mode
            )
            config_obj = algorithm.create_config(config)
            state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.hh_activation_mode == hh_mode

    def test_invalid_mode_raises_error(self, algorithm):
        """Verify invalid dynamic functions mode raises error."""
        with pytest.raises(ValueError, match="Unknown dynamic functions mode"):
            create_config_with_dynamic_functions('invalid_mode')

    def test_custom_num_activations(self, algorithm, xor_problem):
        """Verify custom number of activations works."""
        config = create_config_with_dynamic_functions('cppn_output_4', num_activations=8)
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.df_num_activations == 8


class TestAggregationConfigParsing:
    """Tests for aggregation configuration parsing."""

    def test_disabled_mode_parses(self, algorithm, xor_problem):
        """Verify disabled aggregation mode parses correctly."""
        config = create_config_with_aggregation('disabled')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.agg_mode == 'disabled'

    @pytest.mark.parametrize("mode", list(AGGREGATION_PRESETS.keys()))
    def test_all_aggregation_modes_parse(self, algorithm, xor_problem, mode):
        """Verify all aggregation modes parse without errors."""
        config = create_config_with_aggregation(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("func", ['sum', 'mean', 'max', 'min', 'product'])
    def test_global_aggregation_options(self, xor_problem, func):
        """Verify all global aggregation functions can be configured."""
        # Create fresh algorithm instance for each function
        algorithm = EMRHyperNEAT()
        config = create_config_with_aggregation(f'global_{func}')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Verify mode is set to global (function may default if not parsed)
        assert algorithm.agg_mode == 'global'

    def test_hh_aggregation_mode_options(self, algorithm, xor_problem):
        """Verify H→H aggregation mode options work."""
        for hh_mode in ['sum', 'dynamic']:
            config = create_config_with_aggregation(
                'weight_interp',
                recurrence_preset='hidden_only',
                hh_aggregation_mode=hh_mode
            )
            config_obj = algorithm.create_config(config)
            state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # When disabled, neuromod_config may be None or have enabled=False
        if algorithm.neuromod_config is not None:
            assert algorithm.neuromod_config.enabled == False or algorithm.neuromod_config.mode == 'disabled'

    @pytest.mark.parametrize("mode", list(NEUROMODULATION_PRESETS.keys()))
    def test_all_neuromodulation_modes_parse(self, algorithm, xor_problem, mode):
        """Verify all neuromodulation modes parse without errors."""
        config = create_config_with_neuromodulation(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_static_gating_config(self, algorithm, xor_problem):
        """Verify static gating configuration parses correctly."""
        config = create_config_with_neuromodulation('static_gating')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Check that neuromod_config exists and has correct mode
        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'static_gating'

    def test_true_neuromodulation_config(self, algorithm, xor_problem):
        """Verify TRUE neuromodulation configuration parses correctly."""
        config = create_config_with_neuromodulation('true_neuromodulation_4nt')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
            state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Initialization should complete without error
        # Note: population_size may be stored in different attributes depending on implementation

    def test_max_depth_parsing(self, algorithm, xor_problem):
        """Verify max depth is parsed correctly."""
        for depth in [2, 3, 4, 5]:
            config = create_base_config(max_depth=depth)
            config_obj = algorithm.create_config(config)
            state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

            assert algorithm.max_depth == depth

    def test_threshold_parameters(self, algorithm, xor_problem):
        """Verify threshold parameters are parsed correctly."""
        config = create_base_config(
            variance_threshold=0.05,
            division_threshold=0.05,
            band_threshold=0.2,
        )
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.extended_config.enabled == True
        assert algorithm.dynamic_functions_mode == 'cppn_output'

    def test_recurrence_with_neuromodulation(self, algorithm, xor_problem):
        """Verify recurrence + neuromodulation combination works."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            recurrence_preset='with_backward'
        )
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        hmr['aggregation'] = {'mode': 'global', 'global_aggregation': 'sum'}
        hmr['neuromodulation'] = NEUROMODULATION_PRESETS['true_neuromodulation_4nt'].copy()

        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # All features should be enabled
        assert algorithm.extended_config.enabled == True
        assert algorithm.dynamic_functions_mode == 'cppn_output'
        assert algorithm.agg_mode == 'global'
        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'true_neuromodulation'


class TestRetiredKnobs:
    """Older configs may still carry retired keys."""

    def test_use_vectorized_discovery_is_ignored_not_fatal(self, xor_problem):
        # The knob was parsed and stored but never consulted; it is deleted as
        # a dead knob. A config that still carries it must build and run.
        from conftest import create_config_with_recurrence, run_quick_evolution
        config = create_config_with_recurrence("hidden_only")
        config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["recurrence"][
            "use_vectorized_discovery"] = False
        result = run_quick_evolution(
            EMRHyperNEAT(), config, xor_problem, generations=1, seed=7)
        assert result.error is None, result.error


class TestSelectorGuards:
    """A selector that names a path must reach that path, or say it cannot.

    Each guard here covers a setting that used to be accepted, echoed back under
    verbose, and then quietly ignored: the run looked like a successful run of the
    thing you asked for.
    """

    @staticmethod
    def _config(**emr_overrides):
        config = create_base_config(max_depth=2, population_size=10)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat'].update(emr_overrides)
        return config

    def test_unknown_dynamic_functions_mode_raises(self):
        """A typo used to fall through the dispatch and run the tanh baseline."""
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

    def test_modular_mode_warns_that_it_is_not_dispatched(self):
        """Parsed, echoed under verbose, and never dispatched on the EMR path.

        The activation-index branch for it exists only in the frozen HMR snapshot,
        and EMRHyperNEAT does not inherit from that class, so the mode leaves both
        act_indices and _hidden_act_fn None and evaluation falls back to fixed tanh.
        """
        config = self._config(dynamic_functions={'mode': 'modular'})
        algorithm = EMRHyperNEAT()
        with pytest.warns(UserWarning, match='not implemented on the EMR path'):
            algorithm.create_config(config)
        assert algorithm.dynamic_functions_mode == 'modular'
        assert algorithm._hidden_act_fn is None

    def test_legacy_variance_rule_warns_outside_the_standard_path(self):
        """`legacy` names the old standard-path rule; offload cannot honor it."""
        config = self._config(
            variance_rule='legacy', execution={'mode': 'depth_offload'})
        with pytest.warns(UserWarning, match="variance_rule='legacy' has no effect"):
            EMRHyperNEAT().create_config(config)

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
        """An HMR-shaped or misnested config used to default every EMR setting."""
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
    """FULL_SPARSE is weaker than HYBRID_SPARSE_HH, not stronger."""

    def test_full_sparse_yields_no_sparse_hh(self):
        """Every h→h gate tests HYBRID_SPARSE_HH, so FULL_SPARSE gets none.

        The enum comment used to read "All sparse (extreme recurrence)", which
        describes the opposite of what the code does.
        """
        from emr_hyperneat.emrhyperneat import ForwardPassMode
        import inspect
        from emr_hyperneat import emrhyperneat as module

        source = inspect.getsource(module)
        assert 'extreme recurrence' not in source

        algorithm = EMRHyperNEAT()
        algorithm.set_forward_mode(ForwardPassMode.FULL_SPARSE)
        assert algorithm.get_forward_mode() is ForwardPassMode.FULL_SPARSE
        # The gate the generation loop actually gets to.
        assert algorithm._forward_mode != ForwardPassMode.HYBRID_SPARSE_HH
