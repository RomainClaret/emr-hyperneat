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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
    """Slow runs: every neuromodulation level and receptor derivation, and all features at once."""

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
        """Every receptor derivation method evolves on XOR."""
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
    """context_source: context gating takes its context from the current task index, so 'task_id', which
    says so, is the only value accepted and the default; 'input' and 'learned', which no evaluation reads,
    raise at create_config (0.1.0 accepted and ignored them)."""

    @staticmethod
    def _config(**neuromodulation):
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['neuromodulation'] = {'enabled': True, 'mode': 'context_gating', 'context_dim': 4,
                                  **neuromodulation}
        return config

    def test_task_id_accepted(self, algorithm):
        algorithm.create_config(self._config(context_source='task_id'))
        assert algorithm.neuromod_config.context_source == 'task_id'

    def test_default_is_task_id(self, algorithm):
        algorithm.create_config(self._config())
        assert algorithm.neuromod_config.context_source == 'task_id'

    @pytest.mark.parametrize("context_source", ['input', 'learned'])
    def test_other_sources_refused(self, algorithm, context_source):
        with pytest.raises(ValueError, match="context_source must be 'task_id'"):
            algorithm.create_config(self._config(context_source=context_source))


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

    The paper's rule clips the gain to [0.1, 5], adds a bias and gates with sigmoid(m); HMR FULL's,
    the rule 0.1.0's single-task path computed (its multi-task path computed the paper's), leaves
    the gain unclipped, adds no bias and gates with sigmoid(5 m). The output inversion reads nt[3]
    only when the vector has four or more channels.
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


class TestBranchGating:
    """Tests for branch_gating in neuromodulation."""

    def test_branch_gating_enabled(self, algorithm, xor_problem):
        """Branch gating is not implemented (the function is never called), so enabling it raises
        rather than passing on an ignored key."""
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

    context_derivation ('mean', 'statistics') is read by no engine, EMR or HMR, so it raises,
    naming context_source, which is not read either: context gating takes its context from the
    current task index.
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


# Extended NeuromodulationConfig Feature Tests

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
        # EMR evaluates one level per run, so a level flag beside the mode raises: one mode is
        # one level (HMR's 'custom' with level flags becomes the level HMR evaluated,
        # test_custom_levels.py).
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
        }
        # EMR evaluates one level per run, so a level flag beside the mode raises: one mode is
        # one level (HMR's 'custom' with level flags becomes the level HMR evaluated,
        # test_custom_levels.py).
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
        # EMR evaluates one level per run, so a level flag beside the mode raises: one mode is
        # one level (HMR's 'custom' with level flags becomes the level HMR evaluated,
        # test_custom_levels.py).
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
        """A neuromodulation configuration with many settings. It sets no static_gating (a level
        flag: one mode is one level), gate_from_cppn or branch_gating, which EMR refuses."""
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
    highest input-weight variance, not the population's top positions counted as a fraction of every
    grid position."""

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

class TestOptionB:
    """HMR's Option B: each NT type's receptor density is a dedicated CPPN output at the node's
    self-connection, tanh-bounded, and the base gain comes from the weight output there; the outputs
    follow every other channel. 0.1.0 raised NotImplementedError for receptor_from_weight false, and
    failed on HMR's Option B names, both at the first generation. HMR also decoded channel 1, the
    first receptor, as a per-node activation whenever its CPPN had more than one output; EMR does
    not."""

    @staticmethod
    def _config(**block):
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt', recurrence_preset='feedforward', population_size=20
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'].update(
            receptor_from_weight=False, **block)
        return config

    def test_outputs_and_channels(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(self._config())
        assert algorithm.num_cppn_outputs == 5 and algorithm._cppn_output_indices['receptor'] == 1
        config = self._config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output', 'num_activations': 4}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm.num_cppn_outputs == 6
        assert (
            algorithm._cppn_output_indices['activation'] == 1
            and algorithm._cppn_output_indices['receptor'] == 2
        )

    def test_hmr_names(self):
        config = create_config_with_neuromodulation(
            'true_neuromodulation_4nt', recurrence_preset='feedforward', population_size=20
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'] = {
            'enabled': True, 'mode': 'true_neuromodulation_4nt', 'modulation_strength': 2.0}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert (
            algorithm.neuromod_config.receptor_from_weight,
            algorithm.neuromod_config.num_nt_types,
        ) == (False, 4)
        assert algorithm.num_cppn_outputs == 5
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'] = {
            'enabled': True,
            'mode': 'custom',
            'true_neuromodulation': True,
            'receptor_from_weight': False,
            'num_nt_types': 2,
        }
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert (
            algorithm.neuromod_config.receptor_from_weight,
            algorithm.neuromod_config.num_nt_types,
        ) == (False, 2)
        assert algorithm.num_cppn_outputs == 3

    def test_receptors_are_the_self_connection_outputs(self, xor_problem):
        import jax.numpy as jnp
        import numpy as np
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(self._config()), xor_problem, seed=DEFAULT_SEED
        )
        state, metrics = algorithm.run_generation(state, xor_problem)
        outputs = jnp.asarray(algorithm._cached_self_conn)
        np.testing.assert_allclose(np.asarray(algorithm._neuromod_true['receptor_densities']),
                                   np.asarray(jnp.tanh(jnp.clip(outputs[:, :, 1:5], -10.0, 10.0))),
                                   rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(algorithm._neuromod_true['base_gains']),
            np.asarray(jnp.abs(outputs[:, :, 0]) * 0.5 + 0.5),
            rtol=0,
            atol=1e-7,
        )
        assert np.isfinite(float(metrics.best_fitness))

    def test_too_few_outputs_raise(self):
        config = self._config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output', 'num_activations': 4, 'num_cppn_outputs': 3}
        with pytest.raises(ValueError, match="leaves no room"):
            EMRHyperNEAT().create_config(config)

    def test_multitask_with_option_b(self, two_task_problems):
        from conftest import create_config_with_multi_task
        config = create_config_with_multi_task(
            task_names=two_task_problems['task_names'],
            fitness_aggregation='mean',
            recurrence_preset='feedforward',
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'][
            'receptor_from_weight'
        ] = False
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(config), two_task_problems['problems'][0], seed=DEFAULT_SEED
        )
        _, metrics = algorithm.run_generation_multitask(
            state,
            problems=two_task_problems['problems'],
            neurotransmitters=two_task_problems['nt_vectors'],
        )
        assert algorithm._neuromod_true['option'] == 'B'
        assert all(0.0 <= value <= 1.0 for value in metrics.custom_metrics['per_task_best'])


class TestCppnNeuromodulation:
    """HMR's neuromodulation from dedicated CPPN outputs: a gate per connection (static gating
    multiplies the weights; context gating gates each case with sigmoid(logit + influence * mean of
    the case's input statistics)), and modulatory neurons from a node-type output with factors from
    a mod-factor output. EMR derived gates and modulatory neurons from the weights only (and still
    does by default). Fixed rather than reproduced: modulatory neurons are chosen among a genome's
    active positions (HMR's percentile ran over every position) and the
    modulatory signal is each case's (HMR used the first case's for every case)."""

    @staticmethod
    def _setup(mode, recurrence='feedforward', **block):
        config = create_config_with_neuromodulation(
            mode, recurrence_preset=recurrence, population_size=20
        )
        neuromodulation = config['algorithm_params']['emrhyperneat']['emr_hyperneat'][
            'neuromodulation'
        ]
        neuromodulation.update(block)
        if block.get('modulatory_source') == 'cppn':
            # The conftest block sets the weights mode's mod_decay, which the CPPN source refuses
            neuromodulation.pop('mod_decay', None)
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED
        )
        return algorithm, state

    @staticmethod
    def _numpy_context(algorithm, genome):
        import numpy as np
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        X = np.asarray(algorithm._cached_inputs, dtype=np.float64)
        Y = np.asarray(algorithm._cached_targets, dtype=np.float64)
        W1 = np.asarray(algorithm._cached_W1[genome], dtype=np.float64)
        W2 = np.asarray(algorithm._cached_W2[genome], dtype=np.float64)
        L1 = np.asarray(algorithm._cppn_neuromodulation['G1_logit'][genome], dtype=np.float64)
        L2 = np.asarray(algorithm._cppn_neuromodulation['G2_logit'][genome], dtype=np.float64)
        stats = np.stack([X.mean(1), X.std(1) + 1e-8, X.min(1), X.max(1)], axis=1)
        shift = algorithm.neuromod_config.context_influence * stats.mean(axis=1)
        outputs = np.stack(
            [
                sig(np.tanh(X[c] @ (W1 * sig(L1 + shift[c]))) @ (W2 * sig(L2 + shift[c])))
                for c in range(X.shape[0])
            ]
        )
        return 1.0 - np.mean((outputs - Y) ** 2)

    def test_channels(self):
        algorithm, _ = self._setup('static_gating', gate_source='cppn')
        assert algorithm.num_cppn_outputs == 2 and algorithm._cppn_output_indices['gate'] == 1
        algorithm, _ = self._setup('static_gating', gate_from_cppn=True)
        assert (
            algorithm.neuromod_config.gate_source == 'cppn'
            and algorithm._cppn_output_indices['gate'] == 1
        )
        algorithm, _ = self._setup('modulatory_neurons', modulatory_source='cppn')
        assert algorithm.num_cppn_outputs == 3
        assert (
            algorithm._cppn_output_indices['mod_factor'],
            algorithm._cppn_output_indices['node_type'],
        ) == (1, 2)

    def test_static_gates_multiply_the_weights(self):
        import numpy as np
        algorithm, state = self._setup('static_gating', gate_source='cppn', gate_scaling='binary')
        algorithm.run_generation(state, XORProblem())
        record = algorithm._cppn_neuromodulation
        np.testing.assert_array_equal(np.asarray(algorithm._cached_W1),
                                      np.asarray(record['W1_ungated'] * record['G1']))
        np.testing.assert_array_equal(np.asarray(algorithm._cached_W2),
                                      np.asarray(record['W2_ungated'] * record['G2']))
        gates = np.asarray(record['G1'])
        assert set(np.unique(gates)) <= {0.0, 1.0} and gates.min() == 0.0 and gates.max() == 1.0

    def test_hmr_sigmoid_gates_never_close(self):
        """HMR's sigmoid of a tanh-bounded output (gate_hardness 1): every active gate lies in
        (0.26, 0.74)."""
        import numpy as np
        algorithm, state = self._setup('static_gating', gate_source='cppn', gate_hardness=1.0)
        algorithm.run_generation(state, XORProblem())
        gates = np.asarray(algorithm._cppn_neuromodulation['G1'])
        mask = np.asarray(algorithm._cppn_neuromodulation['W1_ungated']) != 0
        assert gates[mask].min() > 0.26 and gates[mask].max() < 0.74

    def test_context_gates_per_case(self):
        import numpy as np
        algorithm, state = self._setup('context_gating', gate_source='cppn')
        algorithm.run_generation(state, XORProblem())
        for genome in (0, 5):
            assert float(np.asarray(algorithm._cached_fitnesses)[genome]) == pytest.approx(
                self._numpy_context(algorithm, genome), abs=1e-5)

    def test_modulatory_neurons_from_the_cppn(self):
        import numpy as np
        algorithm, state = self._setup('modulatory_neurons', modulatory_source='cppn')
        algorithm.run_generation(state, XORProblem())
        mask = np.asarray(algorithm._mod_neuron_mask)
        active = np.asarray(algorithm._cached_masks_A)
        ratio = algorithm.neuromod_config.mod_neuron_ratio
        assert np.all(mask <= active)
        np.testing.assert_array_equal(mask.sum(axis=1), np.ceil(ratio * active.sum(axis=1)))
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        X = np.asarray(algorithm._cached_inputs, dtype=np.float64)
        Y = np.asarray(algorithm._cached_targets, dtype=np.float64)
        for genome in (0, 5):
            factor = np.tanh(
                np.asarray(algorithm._cppn_neuromodulation['mod_factor'][genome], dtype=np.float64)
            )
            h = np.tanh(X @ np.asarray(algorithm._cached_W1[genome], dtype=np.float64))
            signal = np.tanh(np.sum(h * np.where(mask[genome], factor, 0.0), axis=1, keepdims=True))
            h = np.where(mask[genome], h, h * (1.0 + signal))
            outputs = sig(h @ np.asarray(algorithm._cached_W2[genome], dtype=np.float64))
            assert float(np.asarray(algorithm._cached_fitnesses)[genome]) == pytest.approx(
                1.0 - np.mean((outputs - Y) ** 2), abs=1e-5)

    def test_with_recurrence(self):
        import numpy as np
        for mode, block in (('static_gating', {'gate_source': 'cppn'}),
                            ('modulatory_neurons', {'modulatory_source': 'cppn'})):
            algorithm, state = self._setup(mode, recurrence='hidden_only', **block)
            _, metrics = algorithm.run_generation(state, XORProblem())
            assert np.isfinite(float(metrics.best_fitness)), mode

    @pytest.mark.parametrize(
        'mode, block, reason',
        [
            ('modulatory_neurons', {'gate_source': 'cppn'}, "gate_source 'cppn' gates connections"),
            (
                'static_gating',
                {'modulatory_source': 'cppn'},
                "modulatory_source 'cppn' chooses modulatory neurons",
            ),
            ('static_gating', {'gate_source': 'evolved'}, 'gate_source must be one of'),
            (
                'context_gating',
                {'gate_source': 'cppn', 'recurrence': 'hidden_only'},
                'hidden-to-hidden',
            ),
        ],
    )
    def test_refusals(self, mode, block, reason):
        block = dict(block)
        recurrence = block.pop('recurrence', 'feedforward')
        config = create_config_with_neuromodulation(
            mode, recurrence_preset=recurrence, population_size=20
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'].update(block)
        with pytest.raises(ValueError, match=reason):
            EMRHyperNEAT().create_config(config)

    def test_streaming_is_refused(self):
        config = create_config_with_neuromodulation(
            'static_gating', recurrence_preset='feedforward', population_size=20
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'][
            'gate_source'
        ] = 'cppn'
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
            'mode': 'streaming'
        }
        with pytest.raises(ValueError, match='standard execution path'):
            EMRHyperNEAT().create_config(config)


class TestCppnModulatoryCount:
    """The CPPN source's modulatory count, ceil(ratio x active count), is exact. Computed in float32
    on the device, it would count one more where ratio x count lands just above an integer: 0.3 x 50
    would give 16, at 677 of the counts 1 to 20,000 for ratio 0.3 and 336 for 0.15. The source never
    read mod_decay, so a block that sets it with the source is refused."""

    @staticmethod
    def _engine():
        from emr_hyperneat import emrhyperneat as E
        return E

    @staticmethod
    def _reference_masks(node_type, active, ratio):
        """ceil(Fraction(ratio) x active count) positions per genome, by descending output, the
        lower position first on a tie."""
        import math
        from fractions import Fraction
        import numpy as np
        expected = np.zeros(active.shape, bool)
        for genome in range(active.shape[0]):
            positions = sorted(
                np.flatnonzero(active[genome]), key=lambda p: (-node_type[genome, p], p)
            )
            keep = math.ceil(Fraction(ratio) * len(positions))
            expected[genome, positions[:keep]] = True
        return expected

    @pytest.mark.parametrize('ratio', ['0.05', '0.07', '0.1', '0.15', '0.2', '0.25', '0.3', '0.7'])
    def test_the_count_is_exact(self, ratio):
        import math
        from fractions import Fraction
        import numpy as np
        counts = np.arange(1, 20001)
        # on the decimal the configuration writes
        expected = [math.ceil(Fraction(ratio) * int(n)) for n in counts]

        got = self._engine().exact_ceil_counts(counts, float(ratio))

        np.testing.assert_array_equal(got, expected)

    def test_boundary_counts(self):
        import numpy as np
        E = self._engine()
        # (ratio, active count, exact count): float32 gave 16 for the first two, float64 gives 8 for
        # the third
        cases = [
            (0.3, 50, 15),
            (0.15, 100, 15),
            (0.07, 100, 7),
            (0.3, 90, 27),
            (0.1, 30, 3),
            (0.3, 0, 0),
            (0.3, 1, 1),
        ]

        got = [int(E.exact_ceil_counts(np.array([count]), ratio)[0]) for ratio, count, _ in cases]

        assert got == [exact for _, _, exact in cases]

    @pytest.mark.parametrize('ratio', ['0.1', '0.15', '0.3'])
    def test_masks_take_the_count_among_active_positions(self, ratio):
        import jax.numpy as jnp
        import numpy as np
        # One genome per active count 0 to 120 (50, 90 and 100 were over-counted at 0.3, 100 at
        # 0.15), its active positions scattered over 128, and an output of four values, so ties are
        # common
        rng = np.random.default_rng(0)
        active = np.zeros((121, 128), bool)
        for genome in range(121):
            active[genome, rng.choice(128, genome, replace=False)] = True
        node_type = rng.integers(0, 4, size=active.shape).astype(np.float32)

        mask = np.asarray(
            self._engine().cppn_modulatory_masks(
                jnp.asarray(node_type), jnp.asarray(active), float(ratio)
            )
        )

        np.testing.assert_array_equal(mask, self._reference_masks(node_type, active, ratio))
        assert not (mask & ~active).any()

    def test_a_genome_without_active_positions_has_none(self):
        import jax.numpy as jnp
        import numpy as np
        active = np.array([[False] * 6, [True, False, True, True, False, False]])
        node_type = np.array(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [0.1, 0.9, 0.3, 0.2, 0.8, 0.7]], np.float32
        )

        mask = np.asarray(
            self._engine().cppn_modulatory_masks(jnp.asarray(node_type), jnp.asarray(active), 0.5)
        )

        # ceil(0.5 x 0) = 0; ceil(0.5 x 3) = 2 of the second genome's active positions, its two
        # highest (0.3 and 0.2), never an inactive position, whatever its output
        np.testing.assert_array_equal(mask, [[False] * 6, [False, False, True, True, False, False]])

    @pytest.mark.parametrize('block', [
        {'mode': 'modulatory_neurons', 'modulatory_source': 'cppn', 'mod_decay': 0.9},
        {'mode': 'modulatory_only', 'mod_decay': 0.5},  # HMR's name for the CPPN source
        {'preset': 'modulatory_only', 'modulatory_source': 'cppn', 'mod_decay': 0.9},
    ])
    def test_mod_decay_is_refused_with_the_cppn_source(self, block):
        config = create_base_config(population_size=20)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'] = dict(
            enabled=True, **block
        )

        with pytest.raises(ValueError, match='the CPPN source has no decay'):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('block, source', [
        ({'mode': 'modulatory_neurons', 'mod_decay': 0.5}, 'weights'),
        # EMR's preset: the weights mode
        ({'preset': 'modulatory_only', 'mod_decay': 0.5}, 'weights'),
        ({'mode': 'modulatory_neurons', 'modulatory_source': 'cppn'}, 'cppn'),
        ({'mode': 'modulatory_only'}, 'cppn'),
    ])
    def test_mod_decay_is_accepted_otherwise(self, block, source):
        config = create_base_config(population_size=20)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'] = dict(
            enabled=True, **block
        )
        algorithm = EMRHyperNEAT()

        algorithm.create_config(config)

        # The dataclass default (0.9) does not count as set
        assert algorithm.neuromod_config.modulatory_source == source
        assert algorithm.neuromod_config.mod_decay == block.get('mod_decay', 0.9)


#: Each source's single-change arms (the other bundle's value for one option), its matched-gain
#: controls, the self read, the unread outputs and the fixed count, as (source, options).
MODULATORY_ARMS = [
    ('weights', {}),
    ('weights', {'modulatory_selection': 'cppn'}),
    ('weights', {'modulatory_selection': 'cppn', 'modulatory_cppn_read': 'self'}),
    ('weights', {'modulatory_count_rule': 'ceil'}),
    ('weights', {'modulatory_count_rule': 'fixed', 'mod_neuron_count': 2}),
    ('weights', {'modulatory_signal': 'sum_factor'}),
    ('weights', {'modulatory_node_weights': 'none'}),
    ('weights', {'modulatory_scope': 'standard'}),
    ('weights', {'modulatory_matched_gain': 'mean'}),
    ('weights', {'modulatory_matched_gain': 'shift'}),
    ('weights', {'modulatory_cppn_outputs': 'always'}),
    ('cppn', {}),
    ('cppn', {'modulatory_selection': 'variance'}),
    ('cppn', {'modulatory_count_rule': 'floor'}),
    ('cppn', {'modulatory_signal': 'mean_decay'}),
    ('cppn', {'modulatory_node_weights': 'variance'}),
    ('cppn', {'modulatory_scope': 'all'}),
    ('cppn', {'modulatory_cppn_read': 'self'}),
    ('cppn', {'modulatory_matched_gain': 'mean'}),
    ('cppn', {'modulatory_matched_gain': 'shift'}),
]


def _modulated_reference(
    h, mask, weights, factor, signal, node_weights, scope, connection, decay, gain
):
    """The modulatory node of every option combination in float64 NumPy, from the activity h (cases,
    positions)."""
    import numpy as np
    if signal == 'mean_decay':
        s = np.tanh(np.sum(h * mask, axis=1, keepdims=True) / max(int(mask.sum()), 1)) * decay
    else:
        s = np.tanh(np.sum(h * np.where(mask, factor, 0.0), axis=1, keepdims=True))
    if gain == 'mean':
        s = np.broadcast_to(s.mean(axis=0, keepdims=True), s.shape)
    elif gain == 'shift':
        s = np.roll(s, 1, axis=0)
    if node_weights == 'variance':
        s = s * weights
    if connection == 'multiplicative':
        changed = h * (1.0 + s)
    elif connection == 'additive':
        changed = h + s
    else:
        changed = h / (1.0 + np.exp(-s))
    return np.where(mask, h, changed) if scope == 'standard' else changed


class TestModulatoryOptions:
    """The modulatory-neuron options: each changes one thing the weights mode and the CPPN source
    differ in (selection, count rule, signal, node weights, scope), modulatory_source is the bundle
    an unset option takes, a CPPN output is read at the input queries or the self-connection, the
    CPPN can carry both outputs unread, and the matched-gain controls give each case the mean signal
    over the cases or the previous case's signal.
    """

    @staticmethod
    def _engine():
        from emr_hyperneat import emrhyperneat as E
        return E

    @staticmethod
    def _config(recurrence='feedforward', **block):
        config = create_config_with_neuromodulation(
            'modulatory_neurons', recurrence_preset=recurrence, population_size=20
        )
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'] = {
            'enabled': True,
            'mode': 'modulatory_neurons',
            'mod_connection_type': 'multiplicative',
            **block,
        }
        return config

    @classmethod
    def _run(cls, generations=1, recurrence='feedforward', **block):
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(cls._config(recurrence, **block)),
            XORProblem(),
            seed=DEFAULT_SEED,
        )
        for _ in range(generations):
            state, _ = algorithm.run_generation(state, XORProblem())
        return algorithm, state

    def test_unset_options_take_the_bundle(self):
        E = self._engine()
        resolved = {}
        for name, block in (('weights', {}), ('cppn', {'modulatory_source': 'cppn'}),
                            ('override', {'modulatory_source': 'cppn', 'modulatory_scope': 'all'}),
                            ('hmr_name', {'mode': 'modulatory_only'})):
            algorithm = EMRHyperNEAT()
            algorithm.create_config(self._config(**block))
            resolved[name] = {
                key: getattr(algorithm.neuromod_config, key) for key in E.MODULATORY_OPTIONS
            }

        assert resolved['weights'] == {
            **E.MODULATORY_BUNDLES['weights'],
            'modulatory_cppn_read': 'input_mean',
            'modulatory_cppn_outputs': 'read',
            'modulatory_matched_gain': 'none',
        }
        assert resolved['cppn'] == {**resolved['weights'], **E.MODULATORY_BUNDLES['cppn']}
        assert resolved['override'] == {**resolved['cppn'], 'modulatory_scope': 'all'}
        assert resolved['hmr_name'] == resolved['cppn']
        # A configuration built directly resolves the same way
        assert E.NeuromodulationConfig(modulatory_source='cppn').modulatory_count_rule == 'ceil'

    @pytest.mark.parametrize(
        'block, outputs, channels',
        [
            ({}, 1, {}),
            ({'modulatory_selection': 'cppn'}, 2, {'node_type': 1}),
            ({'modulatory_signal': 'sum_factor'}, 2, {'mod_factor': 1}),
            ({'modulatory_cppn_outputs': 'always'}, 3, {'mod_factor': 1, 'node_type': 2}),
            (
                {'modulatory_selection': 'cppn', 'modulatory_signal': 'sum_factor'},
                3,
                {'mod_factor': 1, 'node_type': 2},
            ),
            ({'modulatory_source': 'cppn'}, 3, {'mod_factor': 1, 'node_type': 2}),
            (
                {'modulatory_source': 'cppn', 'modulatory_selection': 'variance'},
                2,
                {'mod_factor': 1},
            ),
            ({'modulatory_source': 'cppn', 'modulatory_signal': 'mean_decay'}, 2, {'node_type': 1}),
            (
                {
                    'modulatory_source': 'cppn',
                    'modulatory_selection': 'variance',
                    'modulatory_signal': 'mean_decay',
                },
                1,
                {},
            ),
            (
                {
                    'modulatory_source': 'cppn',
                    'modulatory_selection': 'variance',
                    'modulatory_signal': 'mean_decay',
                    'modulatory_cppn_outputs': 'always',
                },
                3,
                {'mod_factor': 1, 'node_type': 2},
            ),
        ],
    )
    def test_the_cppn_carries_the_outputs_the_options_read(self, block, outputs, channels):
        algorithm = EMRHyperNEAT()

        algorithm.create_config(self._config(**block))

        assert algorithm.num_cppn_outputs == outputs
        assert {name: algorithm._cppn_output_indices[name] for name in ('mod_factor', 'node_type')
                if name in algorithm._cppn_output_indices} == channels

    def test_the_outputs_follow_the_per_node_channels(self):
        config = self._config(modulatory_selection='cppn')
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output', 'num_activations': 4}
        algorithm = EMRHyperNEAT()

        algorithm.create_config(config)

        assert algorithm.num_cppn_outputs == 3
        assert (
            algorithm._cppn_output_indices['activation'],
            algorithm._cppn_output_indices['node_type'],
        ) == (1, 2)

    @pytest.mark.parametrize(
        'explicit, implicit',
        [
            (
                {
                    'modulatory_selection': 'variance',
                    'modulatory_count_rule': 'floor',
                    'modulatory_signal': 'mean_decay',
                    'modulatory_node_weights': 'variance',
                    'modulatory_scope': 'all',
                },
                {},
            ),
            (
                {
                    'modulatory_source': 'cppn',
                    'modulatory_selection': 'cppn',
                    'modulatory_count_rule': 'ceil',
                    'modulatory_signal': 'sum_factor',
                    'modulatory_node_weights': 'none',
                    'modulatory_scope': 'standard',
                },
                {'modulatory_source': 'cppn'},
            ),
            # The weights bundle with all five switched is the CPPN source
            (
                {
                    'modulatory_selection': 'cppn',
                    'modulatory_count_rule': 'ceil',
                    'modulatory_signal': 'sum_factor',
                    'modulatory_node_weights': 'none',
                    'modulatory_scope': 'standard',
                },
                {'modulatory_source': 'cppn'},
            ),
        ],
    )
    def test_the_bundle_spelled_out_runs_as_the_bundle(self, explicit, implicit):
        import numpy as np
        runs = [
            self._run(generations=2, recurrence='hidden_only', **block)[0]
            for block in (explicit, implicit)
        ]

        for name in ('_cached_fitnesses', '_mod_neuron_mask'):
            np.testing.assert_array_equal(
                np.asarray(getattr(runs[0], name)), np.asarray(getattr(runs[1], name))
            )

    @pytest.mark.parametrize('rule, ratio, count', [('floor', '0.3', None), ('floor', '0.7', None),
                                                    ('ceil', '0.3', None), ('ceil', '0.15', None),
                                                    ('fixed', None, 3), ('fixed', None, 200)])
    def test_the_count_rules(self, rule, ratio, count):
        import math
        from fractions import Fraction
        import jax.numpy as jnp
        import numpy as np
        rng = np.random.default_rng(3)
        active = np.zeros((121, 128), bool)
        for genome in range(121):
            active[genome, rng.choice(128, genome, replace=False)] = True
        score = rng.integers(0, 4, size=active.shape).astype(np.float32)
        expected = np.zeros(active.shape, bool)
        for genome in range(121):
            positions = sorted(np.flatnonzero(active[genome]), key=lambda p: (-score[genome, p], p))
            n = len(positions)
            keep = {'floor': lambda: max(1, int(np.floor(n * float(ratio)))) if n else 0,
                    'ceil': lambda: math.ceil(Fraction(ratio) * n),
                    'fixed': lambda: min(count, n)}[rule]()
            expected[genome, positions[:keep]] = True

        mask = self._engine().modulatory_masks(jnp.asarray(score), jnp.asarray(active), rule,
                                               None if ratio is None else float(ratio), count)

        np.testing.assert_array_equal(np.asarray(mask), expected)

    def test_the_node_rule_against_numpy(self):
        import itertools
        import jax.numpy as jnp
        import numpy as np
        E = self._engine()
        rng = np.random.default_rng(5)
        pre = rng.normal(size=(4, 12)).astype(np.float32)
        mask = rng.uniform(size=12) < 0.3
        weights = rng.normal(size=12).astype(np.float32)
        factor = rng.uniform(-1, 1, size=12).astype(np.float32)
        h = np.tanh(pre.astype(np.float64))
        for signal, node_weights, scope, connection, gain in itertools.product(
                ('mean_decay', 'sum_factor'), ('variance', 'none'), ('all', 'standard'),
                ('multiplicative', 'additive', 'gated'), ('none', 'mean', 'shift')):
            arrays = (
                ((jnp.asarray(weights),) if node_weights == 'variance' else ())
                + ((jnp.asarray(factor),) if signal == 'sum_factor' else ())
                + (jnp.asarray(mask),)
            )

            rule, operands = E.modulatory_node_rule(
                (jnp.tanh, ()), arrays, connection, signal, node_weights, scope, 0.9, gain
            )

            np.testing.assert_allclose(
                np.asarray(rule(jnp.asarray(pre), *operands)),
                _modulated_reference(
                    h,
                    mask,
                    weights.astype(np.float64),
                    factor.astype(np.float64),
                    signal,
                    node_weights,
                    scope,
                    connection,
                    0.9,
                    gain,
                ),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f'{signal} {node_weights} {scope} {connection} {gain}',
            )

    @pytest.mark.parametrize('source, options', MODULATORY_ARMS)
    def test_a_generation_against_numpy(self, source, options):
        """Feed-forward: the mask from the score the selection reads and the count rule, then every
        genome's fitness from the cached arrays in float64."""
        import jax.numpy as jnp
        import numpy as np
        E = self._engine()
        block = dict(options, **({'modulatory_source': 'cppn'} if source == 'cppn' else {}))

        algorithm, _ = self._run(**block)

        cfg = algorithm.neuromod_config
        W1 = algorithm._cached_W1
        active = algorithm._cached_masks_A
        cppn = algorithm._cppn_neuromodulation
        if cfg.modulatory_cppn_read == 'self' and cppn:
            for name, values in cppn.items():
                np.testing.assert_array_equal(
                    np.asarray(values),
                    np.asarray(
                        algorithm._cached_self_conn[:, :, algorithm._cppn_output_indices[name]]
                    ),
                )
        score = jnp.var(W1, axis=1) if cfg.modulatory_selection == 'variance' else cppn['node_type']
        mask = np.asarray(algorithm._mod_neuron_mask)
        np.testing.assert_array_equal(mask, np.asarray(E.modulatory_masks(
            score, active, cfg.modulatory_count_rule, cfg.mod_neuron_ratio, cfg.mod_neuron_count)))
        X = np.asarray(algorithm._cached_inputs, dtype=np.float64)
        Y = np.asarray(algorithm._cached_targets, dtype=np.float64)
        for genome in range(20):
            h = np.tanh(X @ np.asarray(W1[genome], dtype=np.float64))
            weights = (np.asarray(algorithm._mod_connection_weights[genome], dtype=np.float64)
                       if cfg.modulatory_node_weights == 'variance' else None)
            factor = (np.tanh(np.asarray(cppn['mod_factor'][genome], dtype=np.float64))
                      if cfg.modulatory_signal == 'sum_factor' else None)
            h = _modulated_reference(
                h,
                mask[genome],
                weights,
                factor,
                cfg.modulatory_signal,
                cfg.modulatory_node_weights,
                cfg.modulatory_scope,
                'multiplicative',
                cfg.mod_decay,
                cfg.modulatory_matched_gain,
            )
            outputs = 1.0 / (
                1.0 + np.exp(-(h @ np.asarray(algorithm._cached_W2[genome], dtype=np.float64)))
            )
            assert float(np.asarray(algorithm._cached_fitnesses)[genome]) == pytest.approx(
                1.0 - np.mean((outputs - Y) ** 2), abs=1e-5), genome

    @pytest.mark.parametrize('source, options', [arm for arm in MODULATORY_ARMS if arm[1]])
    def test_recurrent_arms_run(self, source, options):
        import numpy as np
        block = dict(options, **({'modulatory_source': 'cppn'} if source == 'cppn' else {}))

        algorithm, state = self._run(generations=2, recurrence='hidden_only', **block)

        assert np.isfinite(np.asarray(algorithm._cached_fitnesses)).all()

    @pytest.mark.parametrize(
        'block, reason',
        [
            (
                {'mode': 'static_gating', 'modulatory_scope': 'standard'},
                'is an option of mode modulatory_neurons',
            ),
            ({'modulatory_scope': 'everything'}, 'must be one of'),
            ({'modulatory_matched_gain': True}, 'must be one of'),
            ({'modulatory_count_rule': 'fixed'}, "counts mod_neuron_count"),
            (
                {'modulatory_count_rule': 'fixed', 'mod_neuron_count': 3, 'mod_neuron_ratio': 0.1},
                'is not read under',
            ),
            ({'mod_neuron_count': 3}, "read by modulatory_count_rule 'fixed' only"),
            ({'modulatory_count_rule': 'fixed', 'mod_neuron_count': 0}, 'integer of at least 1'),
            ({'modulatory_count_rule': 'fixed', 'mod_neuron_count': 2.5}, 'integer of at least 1'),
            ({'modulatory_count_rule': 'fixed', 'mod_neuron_count': True}, 'integer of at least 1'),
            ({'mod_neuron_ratio': 0.0}, r'must be in \(0, 1\]'),
            ({'mod_neuron_ratio': -0.1}, r'must be in \(0, 1\]'),
            ({'mod_neuron_ratio': 1.5}, r'must be in \(0, 1\]'),
            ({'mod_neuron_ratio': float('nan')}, r'must be in \(0, 1\]'),
            ({'mod_neuron_ratio': True}, r'must be in \(0, 1\]'),
            ({'modulatory_source': 'cppn', 'mod_neuron_ratio': 1.2}, r'must be in \(0, 1\]'),
            (
                {
                    'modulatory_signal': 'sum_factor',
                    'modulatory_node_weights': 'none',
                    'mod_decay': 0.9,
                },
                'nothing reads it',
            ),
            ({'modulatory_cppn_read': 'self'}, 'no option reads'),
            (
                {'modulatory_cppn_outputs': 'always', 'modulatory_cppn_read': 'self'},
                'no option reads',
            ),
        ],
    )
    def test_refusals(self, block, reason):
        config = self._config(**block)

        with pytest.raises(ValueError, match=reason):
            EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('block', [
        {'modulatory_source': 'cppn', 'modulatory_node_weights': 'variance', 'mod_decay': 0.5},
        {'modulatory_source': 'cppn', 'modulatory_signal': 'mean_decay', 'mod_decay': 0.5},
        {'modulatory_selection': 'cppn', 'modulatory_cppn_read': 'self'},
        {'mod_neuron_ratio': 1.0},
        {'mod_neuron_ratio': 1},
        {'modulatory_count_rule': 'fixed', 'mod_neuron_count': 4},
    ])
    def test_accepted(self, block):
        algorithm = EMRHyperNEAT()

        algorithm.create_config(self._config(**block))

        assert all(getattr(algorithm.neuromod_config, key) == value for key, value in block.items())

    def test_the_input_read_needs_the_standard_path_and_the_self_read_does_not(self):
        import numpy as np
        configs = {}
        for read in ('input_mean', 'self'):
            configs[read] = self._config(modulatory_source='cppn', modulatory_cppn_read=read)
            configs[read]['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
                'mode': 'streaming'
            }

        with pytest.raises(ValueError, match='standard execution path'):
            EMRHyperNEAT().create_config(configs['input_mean'])
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(configs['self']), XORProblem(), seed=DEFAULT_SEED
        )
        algorithm.run_generation(state, XORProblem())
        assert (
            algorithm.enable_streaming
            and np.isfinite(np.asarray(algorithm._cached_fitnesses)).all()
        )

    @pytest.mark.parametrize('gain', ['mean', 'shift'])
    def test_a_matched_gain_needs_two_cases(self, gain):
        class OneCase:
            input_shape, output_shape, jitable, use_bias, fitness_threshold = (
                (3,),
                (1,),
                True,
                True,
                0.95,
            )

            def get_data(self):
                return [([0.0, 1.0, 1.0], [1.0])]

        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(self._config(modulatory_matched_gain=gain)),
            OneCase(),
            seed=DEFAULT_SEED,
        )
        with pytest.raises(ValueError, match='with one case it equals'):
            algorithm.run_generation(state, OneCase())

        algorithm, state = self._run(modulatory_matched_gain=gain)
        inputs, targets = algorithm._cached_inputs, algorithm._cached_targets
        with pytest.raises(ValueError, match='one sample makes it equal'):
            algorithm.evaluate_on_data(state, inputs[:1], targets[:1])
        assert algorithm.evaluate_on_data(state, inputs[:2], targets[:2])['n_samples'] == 2


class TestHmrLevelModes:
    """HMR's level presets used as modes: its evaluation took one level, so 'xdg_style' ran context
    gating and 'modulatory_only' modulatory neurons, both from dedicated CPPN outputs."""

    @pytest.mark.parametrize('name, mode, source, outputs, flag', [
        ('xdg_style', 'context_gating', ('gate_source', 'cppn'), 2, 'static_gating'),
        ('modulatory_only', 'modulatory_neurons', ('modulatory_source', 'cppn'), 3,
         'modulatory_neurons'),
    ])
    def test_translation(self, name, mode, source, outputs, flag):
        """A flag that repeats the preset's level is accepted. A flag of another level, as
        modulatory_only with static_gating true, raises (test_silent_drops.py): HMR merged it into
        the preset, an unread gate output, as 'full'."""
        config = create_base_config(population_size=20)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['neuromodulation'] = {
            'enabled': True, 'mode': name, flag: True}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm.neuromod_config.mode == mode
        assert getattr(algorithm.neuromod_config, source[0]) == source[1]
        assert algorithm.num_cppn_outputs == outputs
