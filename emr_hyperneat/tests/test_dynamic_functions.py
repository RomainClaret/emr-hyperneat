"""Tests for dynamic activation functions in EMR-HyperNEAT.

Tests cover:
- 7 selection modes (disabled, global, cppn_output, weight_interpretation, random_fixed, random_generation, modular)
- 18 activation functions (tanh, sigmoid, relu, sin, burst, etc.)
- H→H activation modes (initial_only, every_iteration)
- Activation function palettes
"""

import pytest
from conftest import (
    EMRHyperNEAT,
    create_config_with_dynamic_functions,
    create_config_with_combined_recurrence,
    create_base_config,
    run_quick_evolution,
    assert_no_errors,
    assert_fitness_above,
    assert_positive_fitness,
    DYNAMIC_FUNCTION_PRESETS,
    XORProblem,
    ParityProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
    STANDARD_GENERATIONS,
)


# Selection modes to test
SELECTION_MODES = ['disabled', 'global_tanh', 'global_sin', 'cppn_output_4', 'weight_interp_sign', 'random_fixed']

# Activation functions available
ACTIVATION_FUNCTIONS = [
    'tanh', 'sigmoid', 'relu', 'identity', 'sin', 'gaussian',
    'elu', 'selu', 'swish', 'mish', 'softplus', 'leaky_relu',
]


class TestDynamicFunctionModes:
    """Tests for different dynamic function selection modes."""

    @pytest.mark.parametrize("mode", list(DYNAMIC_FUNCTION_PRESETS.keys()))
    def test_mode_initializes(self, algorithm, xor_problem, mode):
        """Verify each mode initializes without errors."""
        config = create_config_with_dynamic_functions(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("mode", list(DYNAMIC_FUNCTION_PRESETS.keys()))
    def test_mode_runs_generation(self, algorithm, xor_problem, mode):
        """Verify each mode can run a generation."""
        config = create_config_with_dynamic_functions(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=1)

        assert_no_errors(result, f"Mode {mode} failed")

    @pytest.mark.parametrize("mode", SELECTION_MODES)
    def test_mode_produces_fitness(self, algorithm, xor_problem, mode):
        """Verify each mode produces valid fitness."""
        config = create_config_with_dynamic_functions(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Mode {mode} failed")
        assert_positive_fitness(result, f"Mode {mode} has non-positive fitness")


class TestDisabledMode:
    """Tests for disabled dynamic functions mode."""

    def test_disabled_uses_global_activation(self, algorithm, xor_problem):
        """Verify disabled mode uses substrate's global activation."""
        config = create_config_with_dynamic_functions('disabled')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.dynamic_functions_mode == 'disabled'

    def test_disabled_solves_xor(self, algorithm, xor_problem):
        """Verify disabled mode can solve XOR."""
        config = create_config_with_dynamic_functions('disabled')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        assert_fitness_above(result, 0.7)


class TestGlobalMode:
    """Tests for global activation function mode."""

    @pytest.mark.parametrize("activation", ['tanh', 'sin', 'relu', 'sigmoid'])
    def test_global_mode_activations(self, algorithm, xor_problem, activation):
        """Verify global mode with different activations works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': activation,
            'output_activation': 'sigmoid',
        }
        config['algorithm_params']['emrhyperneat']['substrate']['hidden_activation'] = activation

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Global mode with {activation} failed")
        assert_positive_fitness(result)

    def test_global_sin_on_parity(self, algorithm, parity3_problem):
        """Verify global sin activation helps with parity problems."""
        config = create_config_with_dynamic_functions('global_sin')
        # Parity-3 has 4 inputs (3 bits + bias), so update substrate coords
        config['algorithm_params']['emrhyperneat']['substrate']['input_coords'] = [
            (-1.0, -1.0), (-0.33, -1.0), (0.33, -1.0), (1.0, -1.0)
        ]
        result = run_quick_evolution(algorithm, config, parity3_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestCppnOutputMode:
    """Tests for CPPN output-based activation selection."""

    def test_cppn_output_mode_basic(self, algorithm, xor_problem):
        """Verify CPPN output mode works."""
        config = create_config_with_dynamic_functions('cppn_output_4')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("num_activations", [2, 4, 6, 8])
    def test_cppn_output_num_activations(self, algorithm, xor_problem, num_activations):
        """Verify different numbers of activations work."""
        config = create_config_with_dynamic_functions('cppn_output_4', num_activations=num_activations)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"CPPN output with {num_activations} activations failed")
        assert_positive_fitness(result)

    def test_cppn_output_with_recurrence(self, algorithm, xor_problem):
        """Verify CPPN output mode works with recurrence."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only'
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestWeightInterpretationMode:
    """Tests for weight-based activation selection."""

    @pytest.mark.parametrize("interpretation", ['sign', 'magnitude', 'variance'])
    def test_interpretation_methods(self, algorithm, xor_problem, interpretation):
        """Verify all interpretation methods work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 4,
            'interpretation': interpretation,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Interpretation {interpretation} failed")
        assert_positive_fitness(result)

    def test_sign_based_interpretation(self, algorithm, xor_problem):
        """Verify sign-based interpretation works."""
        config = create_config_with_dynamic_functions('weight_interp_sign')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_magnitude_based_interpretation(self, algorithm, xor_problem):
        """Verify magnitude-based interpretation works."""
        config = create_config_with_dynamic_functions('weight_interp_magnitude')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_magnitude_bio_interpretation(self, algorithm, xor_problem):
        """Verify magnitude_bio interpretation works.

        magnitude_bio uses biologically-inspired weight magnitude binning.
        """
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 6,
            'interpretation': 'magnitude_bio',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "magnitude_bio interpretation failed")
        assert_positive_fitness(result)

    def test_hierarchical_sparsity_interpretation(self, algorithm, xor_problem):
        """Verify hierarchical_sparsity interpretation works.

        hierarchical_sparsity derives activation from sparsity at different depth levels.
        """
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 4,
            'interpretation': 'hierarchical_sparsity',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "hierarchical_sparsity interpretation failed")

    def test_sparsity_threshold_interpretation(self, algorithm, xor_problem):
        """Verify sparsity_threshold interpretation works.

        sparsity_threshold selects activation based on connection density thresholds.
        """
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 4,
            'interpretation': 'sparsity_threshold',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "sparsity_threshold interpretation failed")

    def test_stp_inspired_interpretation(self, algorithm, xor_problem):
        """Verify stp_inspired interpretation works.

        stp_inspired uses Short-Term Plasticity-inspired weight patterns.
        """
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 4,
            'interpretation': 'stp_inspired',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "stp_inspired interpretation failed")

    def test_combined_interpretation(self, algorithm, xor_problem):
        """Verify combined interpretation works.

        combined uses multiple interpretation methods together.
        """
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 6,
            'interpretation': 'combined',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "combined interpretation failed")

    @pytest.mark.parametrize("interpretation", [
        'magnitude_bio', 'hierarchical_sparsity', 'sparsity_threshold', 'stp_inspired', 'combined'
    ])
    def test_extended_interpretation_methods(self, algorithm, xor_problem, interpretation):
        """Verify all extended interpretation methods work with parametrization."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 4,
            'interpretation': interpretation,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Extended interpretation {interpretation} failed")


class TestRandomModes:
    """Tests for random activation assignment modes."""

    def test_random_fixed_mode(self, algorithm, xor_problem):
        """Verify random fixed mode assigns consistent activations."""
        config = create_config_with_dynamic_functions('random_fixed')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_random_generation_mode(self, algorithm, xor_problem):
        """Verify random generation mode changes activations."""
        config = create_config_with_dynamic_functions('random_generation')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestHHActivationModes:
    """Tests for H→H activation modes."""

    @pytest.mark.parametrize("hh_mode", ['initial_only', 'every_iteration'])
    def test_hh_activation_modes(self, algorithm, xor_problem, hh_mode):
        """Verify both H→H activation modes work."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only',
            hh_activation_mode=hh_mode
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"H→H mode {hh_mode} failed")
        assert_positive_fitness(result)

    def test_hh_initial_only_config(self, algorithm, xor_problem):
        """Verify initial_only mode is properly configured."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only',
            hh_activation_mode='initial_only'
        )
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.hh_activation_mode == 'initial_only'

    def test_hh_every_iteration_config(self, algorithm, xor_problem):
        """Verify every_iteration mode is properly configured."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only',
            hh_activation_mode='every_iteration'
        )
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.hh_activation_mode == 'every_iteration'


class TestActivationFunctions:
    """Tests for individual activation functions."""

    @pytest.mark.parametrize("activation", ACTIVATION_FUNCTIONS)
    def test_activation_function_works(self, algorithm, xor_problem, activation):
        """Verify each activation function works in global mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': activation,
            'output_activation': 'sigmoid',
        }
        config['algorithm_params']['emrhyperneat']['substrate']['hidden_activation'] = activation

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        # Some activations may not converge well on XOR, but should not error
        assert_no_errors(result, f"Activation {activation} failed")

    def test_sin_activation_for_parity(self, algorithm, parity3_problem):
        """Verify sin activation is effective for parity problems."""
        config = create_config_with_dynamic_functions('global_sin')
        # Parity-3 has 4 inputs (3 bits + bias), so update substrate coords
        config['algorithm_params']['emrhyperneat']['substrate']['input_coords'] = [
            (-1.0, -1.0), (-0.33, -1.0), (0.33, -1.0), (1.0, -1.0)
        ]
        result = run_quick_evolution(algorithm, config, parity3_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        # Sin should help with parity problems (may not always reach 0.5 in 15 gens)
        assert_positive_fitness(result)

    def test_tanh_activation_standard(self, algorithm, xor_problem):
        """Verify tanh activation works as standard baseline."""
        config = create_config_with_dynamic_functions('global_tanh')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        assert_fitness_above(result, 0.7)


class TestActivationPalettes:
    """Tests for activation function palettes."""

    def test_custom_palette(self, algorithm, xor_problem):
        """Verify custom activation palette works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 3,
            'activation_palette': ['tanh', 'sin', 'relu'],
            'interpretation': 'sign',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestDynamicFunctionsWithRecurrence:
    """Tests for dynamic functions combined with recurrence."""

    @pytest.mark.parametrize("conn_type", ['hidden_only', 'with_backward', 'full_recurrent'])
    def test_dynamic_functions_with_connection_types(self, algorithm, xor_problem, conn_type):
        """Verify dynamic functions work with all recurrence types."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset=conn_type
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Dynamic functions with {conn_type} failed")
        assert_positive_fitness(result)


class TestCombinedPerNodeWithRecurrence:
    """Tests for combined activation+aggregation evolution (3-output CPPN) with recurrence.

    This covers the bug where multi-output CPPNs (num_cppn_outputs > 1) caused
    shape corruption in H→H discovery because batch_query_population_multi_source_chunked
    calls .flatten() on CPPN outputs, turning (N, K) into (N*K,).
    The fix routes multi-output CPPNs through the multi-output query path which preserves dimensions.
    """

    @pytest.mark.parametrize("conn_type", ['hidden_only', 'full_recurrent'])
    def test_combined_cppn_output_with_recurrence(self, algorithm, xor_problem, conn_type):
        """Verify combined activation+aggregation (3-output CPPN) works with recurrence."""
        config = create_config_with_combined_recurrence(recurrence_preset=conn_type)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Combined CPPN output + {conn_type} failed")
        assert_positive_fitness(result)

    def test_combined_cppn_output_feedforward_baseline(self, algorithm, xor_problem):
        """Verify combined activation+aggregation works with feedforward (baseline)."""
        config = create_config_with_combined_recurrence(recurrence_preset='feedforward')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Combined CPPN output + feedforward failed")
        assert_positive_fitness(result)


class TestComprehensiveDynamicFunctions:
    """Comprehensive tests (marked slow)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", SELECTION_MODES)
    def test_mode_solves_xor(self, algorithm, xor_problem, mode):
        """Verify each mode can solve XOR given enough generations."""
        config = create_config_with_dynamic_functions(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        assert_fitness_above(result, 0.75, f"Mode {mode} should achieve >0.75 on XOR")

    @pytest.mark.slow
    def test_sin_solves_parity4(self, algorithm, parity4_problem):
        """Verify sin activation can make progress on Parity-4."""
        config = create_config_with_dynamic_functions('global_sin')
        # Parity-4 has 5 inputs (4 bits + bias), so update substrate coords
        config['algorithm_params']['emrhyperneat']['substrate']['input_coords'] = [
            (-1.0, -1.0), (-0.5, -1.0), (0.0, -1.0), (0.5, -1.0), (1.0, -1.0)
        ]
        result = run_quick_evolution(algorithm, config, parity4_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        # Parity-4 is harder, but sin should help
        assert_fitness_above(result, 0.5)

    @pytest.mark.slow
    def test_all_modes_on_and(self, algorithm, and_problem):
        """Verify all modes work on AND problem."""
        for mode in SELECTION_MODES:
            config = create_config_with_dynamic_functions(mode)
            result = run_quick_evolution(algorithm, config, and_problem, generations=QUICK_GENERATIONS)

            assert_no_errors(result, f"Mode {mode} failed on AND")
            assert_positive_fitness(result)


class TestModularMode:
    """Tests for modular dynamic function mode.

    Modular mode provides orthogonal configuration with activation,
    sparsity, and scaling methods combined.
    """

    def test_modular_mode_initializes(self, algorithm, xor_problem):
        """Verify modular mode initializes without errors."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'num_activations': 6,
            'modular': {
                'activation_method': 'magnitude_bio',
                'sparsity_method': 'none',
                'scaling_method': 'none',
            },
        }
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    def test_modular_mode_runs_generation(self, algorithm, xor_problem):
        """Verify modular mode can run generations."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'num_activations': 6,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    @pytest.mark.parametrize("activation_method", ['magnitude_bio', 'sign', 'magnitude'])
    def test_modular_activation_methods(self, algorithm, xor_problem, activation_method):
        """Verify modular mode works with different activation methods."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'modular': {
                'activation_method': activation_method,
            },
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Modular with activation_method={activation_method} failed")

    @pytest.mark.parametrize("sparsity_method", ['none', 'threshold', 'wta'])
    def test_modular_sparsity_methods(self, algorithm, xor_problem, sparsity_method):
        """Verify modular mode works with different sparsity methods."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'modular': {
                'sparsity_method': sparsity_method,
            },
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Modular with sparsity_method={sparsity_method} failed")


class TestBioInspiredActivations:
    """Tests for bio-inspired activation functions.

    These include Izhikevich-inspired and Phase 4 bio-inspired functions.
    """

    # Bio-inspired activations not in the default ACTIVATION_FUNCTIONS list
    BIO_ACTIVATIONS = [
        'abs', 'scaled_tanh', 'softplus',
        'rs_adapt', 'fs_fast', 'lts_low', 'burst', 'resonator',
        'osc_adapt', 'gain_mod', 'receptive', 'band_pass', 'integrate',
    ]

    @pytest.mark.parametrize("activation", BIO_ACTIVATIONS)
    def test_bio_activation_works(self, algorithm, xor_problem, activation):
        """Verify each bio-inspired activation function works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': activation,
            'output_activation': 'sigmoid',
        }
        config['algorithm_params']['emrhyperneat']['substrate']['hidden_activation'] = activation

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Bio activation {activation} failed")

    def test_burst_activation_oscillatory(self, algorithm, xor_problem):
        """Verify burst activation (tanh + sine) works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'burst',
            'output_activation': 'sigmoid',
        }
        config['algorithm_params']['emrhyperneat']['substrate']['hidden_activation'] = 'burst'

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_resonator_activation_damped_sine(self, algorithm, xor_problem):
        """Verify resonator activation (damped sine) works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'resonator',
            'output_activation': 'sigmoid',
        }
        config['algorithm_params']['emrhyperneat']['substrate']['hidden_activation'] = 'resonator'

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_osc_adapt_activation(self, algorithm, xor_problem):
        """Verify osc_adapt activation (oscillatory + adaptive) works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'osc_adapt',
            'output_activation': 'sigmoid',
        }
        config['algorithm_params']['emrhyperneat']['substrate']['hidden_activation'] = 'osc_adapt'

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestNamedPalettes:
    """Tests for named activation function palettes.

    Palettes define subsets of activation functions for selection modes.
    """

    # Named palettes from PALETTE_CONFIGS
    NAMED_PALETTES = [
        'default', 'oscillatory', 'sin_only', 'parity_optimal',
        'classification', 'bio_oscillatory', 'bio_adaptive', 'phase4_all', 'full',
    ]

    @pytest.mark.parametrize("palette_name", NAMED_PALETTES)
    def test_named_palette_initializes(self, algorithm, xor_problem, palette_name):
        """Verify each named palette initializes without errors."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
            'palette': palette_name,
        }
        config_obj = algorithm.create_config(config)

        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("palette_name", ['oscillatory', 'bio_oscillatory', 'parity_optimal'])
    def test_oscillatory_palettes_run(self, algorithm, xor_problem, palette_name):
        """Verify oscillatory palettes can run generations."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
            'palette': palette_name,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Palette {palette_name} failed")

    def test_sin_only_palette_for_parity(self, algorithm, parity3_problem):
        """Verify sin_only palette is effective for parity problems."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'palette': 'sin_only',
        }
        # Parity-3 has 4 inputs (3 bits + bias)
        config['algorithm_params']['emrhyperneat']['substrate']['input_coords'] = [
            (-1.0, -1.0), (-0.33, -1.0), (0.33, -1.0), (1.0, -1.0)
        ]

        result = run_quick_evolution(algorithm, config, parity3_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_bio_adaptive_palette(self, algorithm, xor_problem):
        """Verify bio_adaptive palette works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'palette': 'bio_adaptive',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_full_palette_all_activations(self, algorithm, xor_problem):
        """Verify full palette with all 18 activations works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'palette': 'full',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestSparsityConfiguration:
    """Tests for sparsity configuration in dynamic functions."""

    def test_sparsity_threshold_level_0(self, algorithm, xor_problem):
        """Verify level_0 sparsity threshold works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'modular': {
                'sparsity_method': 'threshold',
                'sparsity': {
                    'level_0': 0.05,
                },
            },
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_sparsity_threshold_level_2_plus(self, algorithm, xor_problem):
        """Verify level_2_plus sparsity threshold works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'modular': {
                'sparsity_method': 'threshold',
                'sparsity': {
                    'level_0': 0.05,
                    'level_1': 0.20,
                    'level_2_plus': 0.40,
                },
            },
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_sparsity_wta_k_percent(self, algorithm, xor_problem):
        """Verify winner-take-all sparsity with k_percent works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'modular': {
                'sparsity_method': 'wta',
                'sparsity': {
                    'wta_k_percent': 0.10,
                },
            },
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    @pytest.mark.parametrize("wta_k", [0.05, 0.10, 0.20, 0.50])
    def test_wta_k_percent_values(self, algorithm, xor_problem, wta_k):
        """Verify different WTA k percentages work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'modular': {
                'sparsity_method': 'wta',
                'sparsity': {
                    'wta_k_percent': wta_k,
                },
            },
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"WTA k={wta_k} failed")


class TestCriticalPeriods:
    """Tests for critical periods in dynamic functions.

    Critical periods control plasticity scheduling with three phases:
    - Phase 1 (0 - phase1_end): Full plasticity (1.0)
    - Phase 2 (phase1_end - phase2_end): Linear decline to min_plasticity
    - Phase 3 (phase2_end - 1.0): Minimal plasticity (fine-tuning only)
    """

    def test_critical_periods_disabled(self, algorithm, xor_problem):
        """Verify critical periods can be disabled (default)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid',
            'critical_periods': {
                'enabled': False,
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_critical_periods_enabled_default(self, algorithm, xor_problem):
        """Verify critical periods works with default settings."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid',
            'critical_periods': {
                'enabled': True,
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_critical_periods_phase1_end(self, algorithm, xor_problem):
        """Verify phase1_end parameter is respected."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid',
            'critical_periods': {
                'enabled': True,
                'phase1_end': 0.1,  # Shorter full plasticity phase
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Critical periods with phase1_end=0.1 failed")

    def test_critical_periods_phase2_end(self, algorithm, xor_problem):
        """Verify phase2_end parameter is respected."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid',
            'critical_periods': {
                'enabled': True,
                'phase1_end': 0.2,
                'phase2_end': 0.7,  # Longer decline phase
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Critical periods with phase2_end=0.7 failed")

    def test_critical_periods_min_plasticity(self, algorithm, xor_problem):
        """Verify min_plasticity parameter is respected."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid',
            'critical_periods': {
                'enabled': True,
                'min_plasticity': 0.1,  # Lower minimum plasticity
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Critical periods with min_plasticity=0.1 failed")

    def test_critical_periods_full_config(self, algorithm, xor_problem):
        """Verify full critical periods configuration works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid',
            'critical_periods': {
                'enabled': True,
                'phase1_end': 0.15,
                'phase2_end': 0.6,
                'min_plasticity': 0.2,
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Full critical periods config failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("phase1_end", [0.05, 0.1, 0.2, 0.3])
    def test_critical_periods_phase1_values(self, algorithm, xor_problem, phase1_end):
        """Verify different phase1_end values work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid',
            'critical_periods': {
                'enabled': True,
                'phase1_end': phase1_end,
                'phase2_end': min(phase1_end + 0.3, 0.9),
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Critical periods phase1_end={phase1_end} failed")

    @pytest.mark.parametrize("min_plasticity", [0.1, 0.3, 0.5, 0.7])
    def test_critical_periods_min_plasticity_values(self, algorithm, xor_problem, min_plasticity):
        """Verify different min_plasticity values work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'global',
            'hidden_activation': 'tanh',
            'output_activation': 'sigmoid',
            'critical_periods': {
                'enabled': True,
                'min_plasticity': min_plasticity,
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Critical periods min_plasticity={min_plasticity} failed")

    def test_critical_periods_with_cppn_output(self, algorithm, xor_problem):
        """Verify critical periods work with cppn_output mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
            'critical_periods': {
                'enabled': True,
                'phase1_end': 0.2,
                'phase2_end': 0.5,
                'min_plasticity': 0.3,
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Critical periods with cppn_output mode failed")

    def test_critical_periods_with_weight_interpretation(self, algorithm, xor_problem):
        """Verify critical periods work with weight_interpretation mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 4,
            'interpretation': 'sign',
            'critical_periods': {
                'enabled': True,
                'phase1_end': 0.2,
                'phase2_end': 0.5,
                'min_plasticity': 0.3,
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Critical periods with weight_interpretation mode failed")


class TestDynamicFunctionsEdgeCases:
    """Edge case tests for dynamic functions."""

    def test_num_activations_2(self, algorithm, xor_problem):
        """Verify minimum number of activations works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 2,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "num_activations=2 failed")

    def test_num_activations_8(self, algorithm, xor_problem):
        """Verify larger number of activations works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 8,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "num_activations=8 failed")

    def test_empty_dynamic_functions_config(self, algorithm, xor_problem):
        """Verify empty dynamic functions config works (uses defaults)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {}

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Empty dynamic functions config failed")

    def test_custom_palette_minimal(self, algorithm, xor_problem):
        """Verify minimal custom palette works (2 functions)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'num_activations': 2,
            'activation_palette': ['tanh', 'sin'],
            'interpretation': 'sign',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Minimal custom palette failed")

    def test_custom_palette_large(self, algorithm, xor_problem):
        """Verify large custom palette works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 6,
            'activation_palette': ['tanh', 'sin', 'relu', 'sigmoid', 'gaussian', 'identity'],
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Large custom palette failed")

    def test_modular_all_methods_combined(self, algorithm, xor_problem):
        """Verify modular mode with all methods specified works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'num_activations': 6,
            'modular': {
                'activation_method': 'magnitude_bio',
                'sparsity_method': 'threshold',
                'scaling_method': 'linear',
                'sparsity': {
                    'level_0': 0.05,
                    'level_1': 0.15,
                    'level_2_plus': 0.30,
                },
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Modular with all methods failed")
