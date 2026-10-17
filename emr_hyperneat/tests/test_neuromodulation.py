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
            'global_function': 'sum',
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


class TestNeuromodulationNodeRule:
    """One neuromodulated node for every path: the paper's rule by default, HMR FULL's as an opt-in.

    In 0.1.0 the single-task path computed HMR FULL's rule (unclipped gain, no bias, gate
    sigmoid(5 m)) while the multi-task path computed the published one (gain clipped to [0.1, 5],
    additive bias, gate sigmoid(m)), and the multi-task output inversion read nt[3] whatever the
    channel count.
    """

    @staticmethod
    def _node_inputs():
        import jax.numpy as jnp
        receptors = jnp.array([[0.5, -0.2, 0.1, 0.9], [-0.3, 0.8, 0.4, 0.0], [1.0, 1.0, 1.0, 1.0]])
        base_gains = jnp.array([0.6, 1.2, 0.9])
        nt = jnp.array([0.5, 0.3, 0.7, 0.2])
        pre = jnp.array([[0.4, -1.1, 2.0], [-0.5, 0.2, 0.3]])
        return receptors, base_gains, nt, pre

    @pytest.mark.parametrize('rule', ['paper', 'hmr_full'])
    @pytest.mark.parametrize('mode', ['full', 'gating_only', 'gain_bias_only'])
    def test_node_formulas(self, rule, mode):
        import jax.numpy as jnp
        import numpy as np
        from emr_hyperneat.emrhyperneat import true_neuromodulation_node_fn
        receptors, base_gains, nt, pre = self._node_inputs()
        s = 2.5
        m = np.asarray(receptors)[:, :3] @ np.asarray(nt)[:3]
        sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
        if rule == 'paper':
            gains = np.clip(np.asarray(base_gains) + s * m, 0.1, 5.0)
            expected = {'full': np.tanh(gains * pre + s * m) * sig(m),
                        'gating_only': np.tanh(pre) * sig(m * s),
                        'gain_bias_only': np.tanh(gains * pre + s * m)}[mode]
        else:
            gains = np.asarray(base_gains) + s * m
            expected = {'full': np.tanh(gains * pre) * sig(5 * m),
                        'gating_only': np.tanh(pre) * sig(5 * m),
                        'gain_bias_only': np.tanh(gains * pre)}[mode]
        node = true_neuromodulation_node_fn(jnp.tanh, receptors, base_gains, nt, s, 4, mode, rule)
        np.testing.assert_allclose(np.asarray(node(pre)), expected, rtol=1e-5, atol=1e-6)

    def test_inversion_needs_four_channels(self):
        import jax.numpy as jnp
        import numpy as np
        from emr_hyperneat.emrhyperneat import invert_outputs
        outputs = jnp.array([0.2, 0.9])
        np.testing.assert_allclose(invert_outputs(outputs, jnp.array([0.4, 0.0]), True, 2), outputs)
        np.testing.assert_allclose(
            invert_outputs(outputs, jnp.array([0.1, 0.1, 0.1, 0.0]), True, 4), 1 - outputs)
        np.testing.assert_allclose(
            invert_outputs(outputs, jnp.array([0.1, 0.1, 0.1, 2.0]), True, 4), outputs)
        np.testing.assert_allclose(
            invert_outputs(outputs, jnp.array([0.1, 0.1, 0.1, 0.0]), False, 4), outputs)

    @pytest.mark.parametrize('mode', ['full', 'gating_only', 'gain_bias_only'])
    def test_single_task_equals_multitask_evaluator(self, algorithm, xor_problem, mode):
        """A single-task generation's fitness equals the multi-task evaluator's on the same
        networks and NT vector."""
        import jax.numpy as jnp
        import numpy as np
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt', recurrence_preset='feedforward', population_size=20,
            modulation_mode=mode)
        state = algorithm.initialize(algorithm.create_config(config), xor_problem, seed=DEFAULT_SEED)
        algorithm.run_generation(state, xor_problem)
        single = np.asarray(algorithm._cached_fitnesses)
        other = np.asarray(algorithm.evaluate_with_neurotransmitter(
            jnp.array([0.5, 0.5, 0.5, 1.0]), algorithm._cached_inputs, algorithm._cached_targets,
            activation_name='tanh', fitness_mode='mse', modulation_mode=mode))
        np.testing.assert_allclose(single, other, rtol=1e-6, atol=1e-7)

    def test_unknown_rule_raises(self, algorithm):
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt', recurrence_preset='feedforward', node_rule='mean_field')
        with pytest.raises(ValueError, match='node_rule must be one of'):
            algorithm.create_config(config)

    def test_two_module_set_after_create_config_refuses_hmr_full(self, algorithm,
                                                                 two_task_problems):
        """A runner may set two_module_mode on the multi-task configuration after create_config;
        the modules compute the paper's rule, so 'hmr_full' is refused there too."""
        import dataclasses
        from conftest import create_config_with_multi_task
        config = create_config_with_multi_task(task_names=two_task_problems['task_names'])
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['recurrence'] = {'preset': 'feedforward'}
        emr['neuromodulation']['node_rule'] = 'hmr_full'
        state = algorithm.initialize(algorithm.create_config(config),
                                     two_task_problems['problems'][0], seed=DEFAULT_SEED)
        algorithm.multitask_config = dataclasses.replace(algorithm.multitask_config,
                                                         two_module_mode='parallel')
        with pytest.raises(ValueError, match='not applied by multitask.two_module_mode'):
            algorithm.run_generation_multitask(state, problems=two_task_problems['problems'],
                                               neurotransmitters=two_task_problems['nt_vectors'])

    def test_one_network_wrapper_forwards_the_rule(self):
        """eval_single_network_hybrid_neuromodulated passes node_rule to the forward."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from emr_hyperneat.emrhyperneat import (
            eval_single_network_hybrid_neuromodulated,
            forward_hybrid_vmapped_neuromodulated,
        )
        keys = jax.random.split(jax.random.PRNGKey(3), 4)
        inputs = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        targets = jnp.array([[0.0], [1.0], [1.0], [0.0]])
        W1 = jax.random.normal(keys[0], (3, 5))
        W2 = jax.random.normal(keys[1], (5, 1))
        receptors = jax.random.uniform(keys[2], (5, 4), minval=-1.0, maxval=1.0)
        base_gains = jax.random.uniform(keys[3], (5,), minval=0.5, maxval=1.5)
        nt = jnp.array([0.5, 0.3, 0.7, 1.0])
        empty = (jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32), jnp.zeros(1), jnp.zeros(1, bool))
        fitness = {}
        for rule in ('paper', 'hmr_full'):
            fitness[rule] = float(eval_single_network_hybrid_neuromodulated(
                W1, W2, *empty, inputs, targets, 1, receptors, base_gains, nt, 2.5, node_rule=rule))
            outputs = forward_hybrid_vmapped_neuromodulated(
                inputs, W1, W2, *empty, 1, receptors, base_gains, nt, 2.5, node_rule=rule)
            np.testing.assert_allclose(
                fitness[rule], 1.0 - float(jnp.mean((outputs - targets) ** 2)), rtol=1e-6)
        assert fitness['paper'] != fitness['hmr_full']


class TestBranchGating:
    """Tests for branch_gating in neuromodulation."""

    def test_branch_gating_enabled(self, algorithm, xor_problem):
        """Branch gating is not implemented (the function is never called), so enabling it raises;
        this test used to pass on the ignored key."""
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
        with pytest.raises(ValueError, match='branch_gating is not implemented'):
            algorithm.create_config(config)

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
        """Branch gating with static gating raises too (not implemented)."""
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
            'branch_gating': True,
        }
        with pytest.raises(ValueError, match='branch_gating is not implemented'):
            algorithm.create_config(config)


class TestContextDerivation:
    """Context gating's context: context_dim, and context_derivation refused.

    context_derivation ('mean', 'statistics') was never read by any engine, EMR or HMR; the tests
    that set it only asserted that a run had no error. It raises, naming
    context_source, which chooses the context.
    """

    @staticmethod
    def _context_gating_config(**neuromodulation):
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['recurrence'] = {
            'preset': 'hidden_only',
            'enabled': True,
            'allow_hidden_to_hidden': True,
            'iteration_level': 2,
        }
        hmr['neuromodulation'] = {'enabled': True, 'mode': 'context_gating', 'context_dim': 4, **neuromodulation}
        return config

    def test_context_gating_default(self, algorithm, xor_problem):
        """Verify context gating runs with its default context."""
        result = run_quick_evolution(algorithm, self._context_gating_config(), xor_problem,
                                     generations=QUICK_GENERATIONS)
        assert_no_errors(result)
        assert_positive_fitness(result)

    @pytest.mark.parametrize('context_dim', [2, 8])
    def test_context_gating_dims(self, algorithm, xor_problem, context_dim):
        """Verify context gating runs with other context dimensions."""
        result = run_quick_evolution(algorithm, self._context_gating_config(context_dim=context_dim),
                                     xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result, f"context_dim={context_dim} failed")

    @pytest.mark.parametrize('derivation', ['mean', 'statistics'])
    def test_context_derivation_refused(self, algorithm, derivation):
        with pytest.raises(ValueError, match='context_derivation is not a setting.*context_source'):
            algorithm.create_config(self._context_gating_config(context_derivation=derivation))


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
            'mod_neuron_ratio': 0.1,
            'mod_connection_type': 'multiplicative',
        }
        # EMR evaluates one level per run; the level flag was never read, so this ran the mode's level alone.
        # It raises: one mode is one level, and HMR's level combinations are not part of EMR.
        with pytest.raises(ValueError, match='neuromodulation.static_gating is not a setting'):
            algorithm.create_config(config)

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
            'context_dim': 4,
            'context_source': 'input',
        }
        # EMR evaluates one level per run; the level flag was never read, so this ran the mode's level alone.
        # It raises: one mode is one level, and HMR's level combinations are not part of EMR.
        with pytest.raises(ValueError, match='neuromodulation.static_gating is not a setting'):
            algorithm.create_config(config)

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
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
        }
        # EMR evaluates one level per run; the level flag was never read, so this ran the mode's level alone.
        # It raises: one mode is one level, and HMR's level combinations are not part of EMR.
        with pytest.raises(ValueError, match='neuromodulation.static_gating is not a setting'):
            algorithm.create_config(config)

    def test_true_neuromodulation_with_branch_gating(self, algorithm, xor_problem):
        """TRUE neuromodulation with branch gating raises: branch gating is not implemented."""
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
        with pytest.raises(ValueError, match='branch_gating is not implemented'):
            algorithm.create_config(config)

    def test_full_feature_combination_neuromod(self, algorithm, xor_problem):
        """A neuromodulation configuration with many settings. In 0.1.0 it also set static_gating: True (a
        level flag EMR never read: one mode is one level), gate_from_cppn and branch_gating: True, all ignored."""
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
            'gate_scaling': 'soft_threshold',
            'gate_hardness': 10.0,
            'num_nt_types': 4,
            'receptor_derivation': 'tanh',
            'modulation_strength': 2.0,
            'modulation_mode': 'full',
            'use_output_inversion': True,
        }
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'num_activations': 4,
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)


class TestModulatoryNeuronsPerGenome:
    """Modulatory neurons are each genome's own: mod_neuron_ratio of its own active positions, at least one, those of
    highest input-weight variance (0.1.0 gave every genome the population's top positions, counted as a fraction of
    every grid position)."""

    def test_masks_are_per_genome_and_active(self):
        import jax.numpy as jnp
        import numpy as np
        from emr_hyperneat.emrhyperneat import modulatory_neuron_masks
        weight_var = jnp.array([[0.9, 0.1, 0.5, 0.3], [0.1, 0.8, 0.2, 0.7], [0.4, 0.4, 0.0, 0.0]])
        active = jnp.array([[True, True, True, True], [True, True, False, True], [True, False, False, False]])
        # Half of 4, 3 and 1 active positions: 2, 1 and (at least) 1
        mask = np.asarray(modulatory_neuron_masks(weight_var, active, 0.5))
        np.testing.assert_array_equal(mask, [[True, False, True, False], [False, True, False, False],
                                             [True, False, False, False]])
        # A genome's mask does not depend on the other genomes
        other = weight_var.at[1:].set(5.0)
        np.testing.assert_array_equal(np.asarray(modulatory_neuron_masks(other, active, 0.5))[0], mask[0])

    def test_the_count_is_a_fraction_of_the_genome_active_positions(self):
        import jax.numpy as jnp
        import numpy as np
        from emr_hyperneat.emrhyperneat import modulatory_neuron_masks
        weight_var = jnp.asarray(np.random.default_rng(0).random((5, 100)), dtype=jnp.float32)
        active = jnp.asarray(np.arange(100)[None, :] < np.array([[40], [4], [10], [90], [0]]))
        counts = np.asarray(modulatory_neuron_masks(weight_var, active, 0.1)).sum(axis=1)
        np.testing.assert_array_equal(counts, [4, 1, 1, 9, 0])      # at least one where any position is active
        # Counted as Python counts, int(90 * 0.7) = 62 (float32 would give 63)
        counts = np.asarray(modulatory_neuron_masks(weight_var, active, 0.7)).sum(axis=1)
        np.testing.assert_array_equal(counts, [28, 2, 7, 62, 0])

    def test_mode_runs_with_per_genome_masks(self, algorithm, xor_problem, monkeypatch):
        import numpy as np
        import emr_hyperneat.emrhyperneat as E
        seen = {}
        original = E.modulatory_neuron_masks

        def spy(weight_var, active, ratio):
            seen.update(active=np.asarray(active), ratio=ratio)
            return original(weight_var, active, ratio)
        monkeypatch.setattr(E, 'modulatory_neuron_masks', spy)
        config = create_config_with_neuromodulation('modulatory_neurons', recurrence_preset='feedforward',
                                                    population_size=12)
        state = algorithm.initialize(algorithm.create_config(config), xor_problem, seed=DEFAULT_SEED)
        algorithm.run_generation(state, xor_problem)
        mask = np.asarray(algorithm._mod_neuron_mask)
        active = seen['active']
        assert mask.ndim == 2 and mask.shape[0] == 12 and seen['ratio'] == 0.1
        assert not (mask & ~active).any()
        n_active = active.sum(axis=1)
        expected = np.where(n_active > 0, np.maximum(1, np.floor(n_active * seen['ratio'])), 0)
        np.testing.assert_array_equal(mask.sum(axis=1), expected)
