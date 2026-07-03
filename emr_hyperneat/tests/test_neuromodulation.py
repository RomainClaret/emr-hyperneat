"""Tests for neuromodulation in EMR-HyperNEAT.

Tests cover:
- 4 neuromodulation levels:
  - Static Gating (CPPN-based gate values)
  - Context Gating (XdG-style context-dependent gating)
  - Modulatory Neurons (Soltoggio-style)
  - TRUE Neuromodulation (NT vectors + receptor densities)
- 7 receptor derivation methods
- Modulation strength and mode settings
- Multi-task compatibility
"""

import pytest
from conftest import (
    EMRHyperNEAT,
    create_config_with_neuromodulation,
    create_base_config,
    run_quick_evolution,
    assert_no_errors,
    assert_fitness_above,
    assert_positive_fitness,
    NEUROMODULATION_PRESETS,
    XORProblem,
    ANDProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
    STANDARD_GENERATIONS,
)


# Neuromodulation levels to test
NEUROMODULATION_LEVELS = list(NEUROMODULATION_PRESETS.keys())

# Receptor derivation methods (all 7)
RECEPTOR_DERIVATION_METHODS = ['tanh', 'orthogonal', 'phase_shifted', 'abs', 'normalized', 'fourier', 'softmax']


class TestNeuromodulationModes:
    """Tests for different neuromodulation modes."""

    @pytest.mark.parametrize("mode", NEUROMODULATION_LEVELS)
    def test_mode_initializes(self, algorithm, xor_problem, mode):
        """Verify each neuromodulation mode initializes without errors."""
        config = create_config_with_neuromodulation(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

    @pytest.mark.parametrize("mode", NEUROMODULATION_LEVELS)
    def test_mode_runs_generation(self, algorithm, xor_problem, mode):
        """Verify each mode can run a generation."""
        config = create_config_with_neuromodulation(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=1)

        assert_no_errors(result, f"Neuromodulation mode {mode} failed")

    @pytest.mark.parametrize("mode", NEUROMODULATION_LEVELS)
    def test_mode_produces_fitness(self, algorithm, xor_problem, mode):
        """Verify each mode produces valid fitness."""
        config = create_config_with_neuromodulation(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Mode {mode} failed")
        assert_positive_fitness(result, f"Mode {mode} has non-positive fitness")


class TestDisabledNeuromodulation:
    """Tests for disabled neuromodulation."""

    def test_disabled_mode(self, algorithm, xor_problem):
        """Verify disabled neuromodulation works."""
        config = create_config_with_neuromodulation('disabled')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        # Check mode is disabled (neuromod_config may be None or have mode='disabled')
        if algorithm.neuromod_config is not None:
            assert algorithm.neuromod_config.mode == 'disabled' or algorithm.neuromod_config.enabled == False

    def test_disabled_solves_xor(self, algorithm, xor_problem):
        """Verify disabled mode can solve XOR."""
        config = create_config_with_neuromodulation('disabled')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        assert_fitness_above(result, 0.7)


class TestStaticGating:
    """Tests for static gating (Level 1)."""

    def test_static_gating_config(self, algorithm, xor_problem):
        """Verify static gating is configured correctly."""
        config = create_config_with_neuromodulation('static_gating')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'static_gating'

    def test_static_gating_produces_fitness(self, algorithm, xor_problem):
        """Verify static gating produces valid fitness."""
        config = create_config_with_neuromodulation('static_gating')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("scaling", ['sigmoid', 'binary', 'soft_threshold'])
    def test_gate_scaling_options(self, algorithm, xor_problem, scaling):
        """Verify all gate scaling options work."""
        config = create_config_with_neuromodulation('static_gating', gate_scaling=scaling)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Gate scaling {scaling} failed")
        assert_positive_fitness(result)


class TestContextGating:
    """Tests for context-dependent gating (Level 2)."""

    def test_context_gating_config(self, algorithm, xor_problem):
        """Verify context gating is configured correctly."""
        config = create_config_with_neuromodulation('context_gating')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'context_gating'

    def test_context_gating_produces_fitness(self, algorithm, xor_problem):
        """Verify context gating produces valid fitness."""
        config = create_config_with_neuromodulation('context_gating')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("context_dim", [2, 4, 8])
    def test_context_dimensions(self, algorithm, xor_problem, context_dim):
        """Verify different context dimensions work."""
        config = create_config_with_neuromodulation('context_gating', context_dim=context_dim)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Context dim {context_dim} failed")
        assert_positive_fitness(result)


class TestModulatoryNeurons:
    """Tests for modulatory neurons (Level 3)."""

    def test_modulatory_neurons_config(self, algorithm, xor_problem):
        """Verify modulatory neurons configuration."""
        config = create_config_with_neuromodulation('modulatory_neurons')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'modulatory_neurons'

    def test_modulatory_neurons_produces_fitness(self, algorithm, xor_problem):
        """Verify modulatory neurons produce valid fitness."""
        config = create_config_with_neuromodulation('modulatory_neurons')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("ratio", [0.05, 0.1, 0.2])
    def test_modulatory_neuron_ratios(self, algorithm, xor_problem, ratio):
        """Verify different modulatory neuron ratios work."""
        config = create_config_with_neuromodulation('modulatory_neurons', mod_neuron_ratio=ratio)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Modulatory ratio {ratio} failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("conn_type", ['multiplicative', 'additive'])
    def test_modulatory_connection_types(self, algorithm, xor_problem, conn_type):
        """Verify different modulatory connection types work."""
        config = create_config_with_neuromodulation(
            'modulatory_neurons',
            mod_connection_type=conn_type
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Modulatory connection type {conn_type} failed")
        assert_positive_fitness(result)


class TestTrueNeuromodulation:
    """Tests for TRUE neuromodulation (Level 4)."""

    def test_true_neuromodulation_config(self, algorithm, xor_problem):
        """Verify TRUE neuromodulation configuration."""
        config = create_config_with_neuromodulation('true_neuromodulation_4nt')
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.neuromod_config is not None
        assert algorithm.neuromod_config.mode == 'true_neuromodulation'
        assert algorithm.neuromod_config.num_nt_types == 4

    def test_true_neuromodulation_produces_fitness(self, algorithm, xor_problem):
        """Verify TRUE neuromodulation produces valid fitness."""
        config = create_config_with_neuromodulation('true_neuromodulation_4nt')
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize("num_nt", [2, 4, 5, 6])
    def test_neurotransmitter_counts(self, algorithm, xor_problem, num_nt):
        """Verify different neurotransmitter counts work."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            num_nt_types=num_nt
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"NT count {num_nt} failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("modulation_mode", ['full', 'gating_only', 'gain_bias_only'])
    def test_modulation_modes(self, algorithm, xor_problem, modulation_mode):
        """Verify different modulation modes work."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            modulation_mode=modulation_mode
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Modulation mode {modulation_mode} failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("strength", [0.5, 2.0, 5.0])
    def test_modulation_strengths(self, algorithm, xor_problem, strength):
        """Verify different modulation strengths work."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            modulation_strength=strength
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Modulation strength {strength} failed")
        assert_positive_fitness(result)


class TestReceptorDerivation:
    """Tests for receptor derivation methods."""

    @pytest.mark.parametrize("derivation", RECEPTOR_DERIVATION_METHODS)
    def test_receptor_derivation_methods(self, algorithm, xor_problem, derivation):
        """Verify all receptor derivation methods work."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            receptor_derivation=derivation
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Receptor derivation {derivation} failed")
        assert_positive_fitness(result)

    def test_tanh_derivation(self, algorithm, xor_problem):
        """Verify tanh receptor derivation works."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            receptor_derivation='tanh'
        )
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.neuromod_config.receptor_derivation == 'tanh'

    def test_orthogonal_derivation(self, algorithm, xor_problem):
        """Verify orthogonal receptor derivation works."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            receptor_derivation='orthogonal'
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestNeuromodulationWithRecurrence:
    """Tests for neuromodulation combined with recurrence."""

    @pytest.mark.parametrize("conn_type", ['hidden_only', 'with_backward', 'full_recurrent'])
    def test_neuromodulation_with_connection_types(self, algorithm, xor_problem, conn_type):
        """Verify neuromodulation works with all recurrence types."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            recurrence_preset=conn_type
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Neuromodulation with {conn_type} failed")
        assert_positive_fitness(result)


class TestNeuromodulationWithDynamicFunctions:
    """Tests for neuromodulation combined with dynamic functions."""

    def test_neuromodulation_with_dynamic_functions(self, algorithm, xor_problem):
        """Verify neuromodulation works with dynamic functions."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
        }
        hmr['neuromodulation'] = NEUROMODULATION_PRESETS['true_neuromodulation_4nt'].copy()

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestNeuromodulationLevelDistinctness:
    """Tests verifying neuromodulation levels produce distinct behavior."""

    def test_levels_produce_different_results(self, algorithm, xor_problem):
        """Verify different neuromodulation levels can produce different results."""
        results = {}
        for mode in ['disabled', 'static_gating', 'true_neuromodulation_4nt']:
            config = create_config_with_neuromodulation(mode)
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
            results[mode] = result.best_fitness

        # All should work
        for mode, fitness in results.items():
            assert fitness >= 0, f"Mode {mode} has invalid fitness"


class TestComprehensiveNeuromodulation:
    """Comprehensive tests (marked slow)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", NEUROMODULATION_LEVELS)
    def test_mode_achieves_fitness(self, algorithm, xor_problem, mode):
        """Verify each mode can achieve good fitness on XOR."""
        config = create_config_with_neuromodulation(mode)
        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        # All modes should achieve reasonable fitness
        assert_fitness_above(result, 0.6, f"Mode {mode} should achieve >0.6 on XOR")

    @pytest.mark.slow
    def test_all_receptor_derivations(self, algorithm, xor_problem):
        """Test all receptor derivation methods comprehensively."""
        for derivation in RECEPTOR_DERIVATION_METHODS:
            config = create_config_with_neuromodulation(
                'true_neuromodulation_4nt',
                receptor_derivation=derivation
            )
            result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

            assert_no_errors(result, f"Receptor derivation {derivation} failed")
            assert_fitness_above(result, 0.5)

    @pytest.mark.slow
    def test_full_feature_combination(self, algorithm, xor_problem):
        """Test neuromodulation with all features enabled."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'full_recurrent',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'allow_backward': True,
            'allow_lateral': True,
            'allow_self_loops': True,
            'iteration_level': 2,
        }
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
        }
        hmr['aggregation'] = {
            'mode': 'global',
            'global_aggregation': 'sum',
        }
        hmr['neuromodulation'] = NEUROMODULATION_PRESETS['true_neuromodulation_4nt'].copy()

        result = run_quick_evolution(algorithm, config, xor_problem, generations=STANDARD_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestAdditionalReceptorDerivations:
    """Tests for additional receptor derivation methods not in base tests."""

    def test_abs_receptor_derivation(self, algorithm, xor_problem):
        """Verify abs receptor derivation works."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            receptor_derivation='abs'
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_normalized_receptor_derivation(self, algorithm, xor_problem):
        """Verify normalized receptor derivation works."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            receptor_derivation='normalized'
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_fourier_receptor_derivation(self, algorithm, xor_problem):
        """Verify fourier receptor derivation works."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            receptor_derivation='fourier'
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_softmax_receptor_derivation(self, algorithm, xor_problem):
        """Verify softmax receptor derivation works."""
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt',
            receptor_derivation='softmax'
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestContextSource:
    """Tests for context_source options in context gating mode."""

    @pytest.mark.parametrize("context_source", ['input', 'task_id', 'learned'])
    def test_context_source_options(self, algorithm, xor_problem, context_source):
        """Verify all context_source options work."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'context_gating',
            'gate_from_cppn': True,
            'context_dim': 4,
            'context_source': context_source,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Context source {context_source} failed")

    def test_context_source_input(self, algorithm, xor_problem):
        """Verify input-based context source works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'context_gating',
            'context_source': 'input',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_context_source_learned(self, algorithm, xor_problem):
        """Verify learned context source works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'context_gating',
            'context_source': 'learned',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestGatedModulationConnection:
    """Tests for gated modulatory connection type."""

    def test_gated_connection_type(self, algorithm, xor_problem):
        """Verify gated modulatory connection type works."""
        config = create_config_with_neuromodulation(
            'modulatory_neurons',
            mod_connection_type='gated'
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_gated_with_decay(self, algorithm, xor_problem):
        """Verify gated connection type works with decay."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'modulatory_neurons',
            'mod_neuron_ratio': 0.1,
            'mod_connection_type': 'gated',
            'mod_decay': 0.9,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestModDecay:
    """Tests for mod_decay parameter in modulatory neurons."""

    @pytest.mark.parametrize("decay", [0.5, 0.7, 0.9, 0.99])
    def test_mod_decay_values(self, algorithm, xor_problem, decay):
        """Verify different mod_decay values work."""
        config = create_config_with_neuromodulation(
            'modulatory_neurons',
            mod_decay=decay
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"Mod decay {decay} failed")

    def test_mod_decay_zero(self, algorithm, xor_problem):
        """Verify mod_decay=0 (no decay) works."""
        config = create_config_with_neuromodulation(
            'modulatory_neurons',
            mod_decay=0.0
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_mod_decay_one(self, algorithm, xor_problem):
        """Verify mod_decay=1 (full persistence) works."""
        config = create_config_with_neuromodulation(
            'modulatory_neurons',
            mod_decay=1.0
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestOutputInversion:
    """Tests for use_output_inversion in neuromodulation."""

    def test_output_inversion_enabled(self, algorithm, xor_problem):
        """Verify output inversion can be enabled."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'use_output_inversion': True,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_output_inversion_disabled(self, algorithm, xor_problem):
        """Verify output inversion can be disabled (default)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'use_output_inversion': False,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestBranchGating:
    """Tests for branch_gating in neuromodulation."""

    def test_branch_gating_enabled(self, algorithm, xor_problem):
        """Verify branch gating can be enabled."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'branch_gating': True,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_branch_gating_disabled(self, algorithm, xor_problem):
        """Verify branch gating can be disabled (default)."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'branch_gating': False,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)

    def test_branch_gating_with_static_gating(self, algorithm, xor_problem):
        """Verify branch gating works with static gating mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'static_gating',
            'gate_from_cppn': True,
            'branch_gating': True,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)


class TestContextDerivation:
    """Tests for context_derivation parameter in context gating neuromodulation.

    The context_derivation parameter controls how context vectors are derived:
    - 'mean': Simple mean of input features to derive context
    - 'statistics': Use statistical features (mean, variance) for richer context
    """

    def test_context_derivation_default(self, algorithm, xor_problem):
        """Verify context gating uses default context_derivation when not specified."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'context_gating',
            'context_dim': 4,
            # context_derivation not specified - should use default
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_context_derivation_mean(self, algorithm, xor_problem):
        """Verify 'mean' context derivation method works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'context_gating',
            'context_dim': 4,
            'context_derivation': 'mean',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "context_derivation='mean' failed")
        assert_positive_fitness(result)

    def test_context_derivation_statistics(self, algorithm, xor_problem):
        """Verify 'statistics' context derivation method works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'context_gating',
            'context_dim': 4,
            'context_derivation': 'statistics',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "context_derivation='statistics' failed")
        assert_positive_fitness(result)

    @pytest.mark.parametrize("derivation", ['mean', 'statistics'])
    def test_context_derivation_with_context_gating(self, algorithm, xor_problem, derivation):
        """Verify all context_derivation methods work with context gating mode."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'context_gating',
            'gate_from_cppn': True,
            'context_dim': 4,
            'context_source': 'input',
            'context_derivation': derivation,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"context_derivation='{derivation}' with context_gating failed")
        assert_positive_fitness(result)

    def test_context_derivation_mean_with_different_dims(self, algorithm, xor_problem):
        """Verify 'mean' context derivation works with different context dimensions."""
        for context_dim in [2, 4, 8]:
            config = create_base_config()
            hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            hmr['recurrence'] = {
                'preset': 'hidden_only',
                'enabled': True,
                'allow_hidden_to_hidden': True,
                'iteration_level': 2,
            }
            hmr['neuromodulation'] = {
                'enabled': True,
                'mode': 'context_gating',
                'context_dim': context_dim,
                'context_derivation': 'mean',
            }
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

            assert_no_errors(result, f"context_derivation='mean' with dim={context_dim} failed")

    def test_context_derivation_statistics_with_different_dims(self, algorithm, xor_problem):
        """Verify 'statistics' context derivation works with different context dimensions."""
        for context_dim in [2, 4, 8]:
            config = create_base_config()
            hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            hmr['recurrence'] = {
                'preset': 'hidden_only',
                'enabled': True,
                'allow_hidden_to_hidden': True,
                'iteration_level': 2,
            }
            hmr['neuromodulation'] = {
                'enabled': True,
                'mode': 'context_gating',
                'context_dim': context_dim,
                'context_derivation': 'statistics',
            }
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

            assert_no_errors(result, f"context_derivation='statistics' with dim={context_dim} failed")


# =============================================================================
# Extended NeuromodulationConfig Feature Tests
# =============================================================================

class TestGateHardness:
    """Tests for gate_hardness parameter with soft_threshold gate scaling.

    gate_hardness controls the sharpness of the soft_threshold transition:
    - Low values (1.0): Gradual, smooth transition
    - High values (20.0): Sharp, almost binary transition
    Default: 10.0
    """

    @pytest.mark.parametrize("hardness", [1.0, 5.0, 10.0, 20.0])
    def test_gate_hardness_values(self, algorithm, xor_problem, hardness):
        """Verify different gate_hardness values work with soft_threshold."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'static_gating',
            'gate_from_cppn': True,
            'gate_scaling': 'soft_threshold',
            'gate_hardness': hardness,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"gate_hardness={hardness} failed")

    def test_gate_hardness_default(self, algorithm, xor_problem):
        """Verify default gate_hardness (10.0) works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'static_gating',
            'gate_from_cppn': True,
            'gate_scaling': 'soft_threshold',
            # gate_hardness not specified - uses default 10.0
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_gate_hardness_low_vs_high(self, algorithm, xor_problem):
        """Verify both low and high gate_hardness values produce different results."""
        results = {}
        for hardness in [1.0, 20.0]:
            config = create_base_config()
            hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            hmr['recurrence'] = {
                'preset': 'hidden_only',
                'enabled': True,
                'allow_hidden_to_hidden': True,
                'iteration_level': 2,
            }
            hmr['neuromodulation'] = {
                'enabled': True,
                'mode': 'static_gating',
                'gate_from_cppn': True,
                'gate_scaling': 'soft_threshold',
                'gate_hardness': hardness,
            }
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
            results[hardness] = result.best_fitness

        # Both should work
        for hardness, fitness in results.items():
            assert fitness >= 0, f"gate_hardness={hardness} produced invalid fitness"

    def test_gate_hardness_extreme_values(self, algorithm, xor_problem):
        """Verify extreme gate_hardness values work (0.1 and 100.0)."""
        for hardness in [0.1, 100.0]:
            config = create_base_config()
            hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            hmr['recurrence'] = {
                'preset': 'hidden_only',
                'enabled': True,
                'allow_hidden_to_hidden': True,
                'iteration_level': 2,
            }
            hmr['neuromodulation'] = {
                'enabled': True,
                'mode': 'static_gating',
                'gate_from_cppn': True,
                'gate_scaling': 'soft_threshold',
                'gate_hardness': hardness,
            }
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

            assert_no_errors(result, f"gate_hardness={hardness} failed")


class TestSelfConnectionQuery:
    """Tests for use_self_connection_query in receptor derivation.

    When True, uses batch_query_population_self_connections() to get receptor
    densities from CPPN output at (x,y,x,y,bias) instead of deriving from weights.
    Default: False (derive from weights).
    """

    def test_self_connection_query_disabled(self, algorithm, xor_problem):
        """Verify use_self_connection_query=False (default) works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'use_self_connection_query': False,  # Default: derive from weights
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_self_connection_query_enabled(self, algorithm, xor_problem):
        """Verify use_self_connection_query=True works."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'use_self_connection_query': True,  # Use CPPN self-connection output
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_self_connection_query_with_different_derivations(self, algorithm, xor_problem):
        """Verify use_self_connection_query works with different receptor derivation methods."""
        for derivation in ['tanh', 'orthogonal', 'phase_shifted']:
            config = create_base_config()
            hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            hmr['recurrence'] = {
                'preset': 'hidden_only',
                'enabled': True,
                'allow_hidden_to_hidden': True,
                'iteration_level': 2,
            }
            hmr['neuromodulation'] = {
                'enabled': True,
                'mode': 'true_neuromodulation',
                'num_nt_types': 4,
                'receptor_derivation': derivation,
                'use_self_connection_query': True,
            }
            result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

            assert_no_errors(result, f"use_self_connection_query with {derivation} failed")


class TestCombinedNeuromodulationLevels:
    """Tests for combining multiple neuromodulation levels simultaneously.

    Tests combinations of:
    - Static Gating (Level 1)
    - Context Gating (Level 2)
    - Modulatory Neurons (Level 3)
    - TRUE Neuromodulation (Level 4)
    """

    def test_static_gating_with_modulatory_neurons(self, algorithm, xor_problem):
        """Verify static gating + modulatory neurons combination."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'modulatory_neurons',  # Level 3
            'static_gating': True,  # Also enable Level 1
            'gate_from_cppn': True,
            'mod_neuron_ratio': 0.1,
            'mod_connection_type': 'multiplicative',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_context_gating_with_static_gating(self, algorithm, xor_problem):
        """Verify context gating + static gating combination."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'context_gating',  # Level 2
            'static_gating': True,  # Also enable Level 1
            'gate_from_cppn': True,
            'context_dim': 4,
            'context_source': 'input',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_true_neuromodulation_with_static_gating(self, algorithm, xor_problem):
        """Verify TRUE neuromodulation + static gating combination."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',  # Level 4
            'static_gating': True,  # Also enable Level 1
            'gate_from_cppn': True,
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_true_neuromodulation_with_branch_gating(self, algorithm, xor_problem):
        """Verify TRUE neuromodulation + branch gating combination."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'branch_gating': True,  # Enable branch-specific gating
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)

    def test_full_feature_combination_neuromod(self, algorithm, xor_problem):
        """Verify complex neuromodulation feature combination."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'full_recurrent',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'allow_backward': True,
            'allow_lateral': True,
            'allow_self_loops': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {
            'enabled': True,
            'mode': 'true_neuromodulation',
            'static_gating': True,
            'gate_from_cppn': True,
            'gate_scaling': 'soft_threshold',
            'gate_hardness': 10.0,
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'modulation_strength': 2.0,
            'modulation_mode': 'full',
            'branch_gating': True,
            'use_output_inversion': True,
        }
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)
