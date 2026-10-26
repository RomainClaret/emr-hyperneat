"""HMR's names and aliases in the engine.

Runners written for the frozen HMR engine use names that EMR lacked: NUM_ACTIVATIONS, NT_PRESETS
with HMR's mnist and iris vectors, the CPPN-output helpers of its neuromodulation variant, the
attribute names of the per-network activation functions, the multires generation names of its base
class, configure_offloading's skip_dead_levels and HMR's result keys of run_until_threshold. HMR's
three context-gated multi-task methods never ran there (they read gate logits nothing set) and are
refused with a pointer to EMR's own API.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import (DEFAULT_SEED, EMRHyperNEAT, XORProblem, create_base_config,
                      create_config_with_neuromodulation)
from emr_hyperneat.emrhyperneat_base import (
    EMRHyperNEATChunking,
)
from emr_hyperneat.emrhyperneat import (
    ACTIVATION_LIST,
    NT_PRESETS,
    NT_PRESETS_2,
    NT_PRESETS_3_FIBONACCI,
    NT_PRESETS_4,
    NT_PRESETS_5,
    NT_PRESETS_6,
    NUM_ACTIVATIONS,
    get_cppn_num_outputs,
    get_cppn_receptor_base_idx,
    get_nt_for_task,
)
from hmr_reference import hmr_value, live  # type: ignore[import-not-found]


class TestModuleNames:
    def test_num_activations_counts_the_list(self):
        assert NUM_ACTIVATIONS == len(ACTIVATION_LIST) == 18

    def test_num_activations_equals_hmrs(self):
        assert NUM_ACTIVATIONS == hmr_value(
            'hmr_names.num_activations', lambda: live('num_activations')
        )

    def test_nt_presets_is_the_two_type_table(self):
        assert NT_PRESETS is NT_PRESETS_2

    def test_hmr_task_vectors_equal_hmrs(self):
        hmr = hmr_value('hmr_names.nt_task_vectors', lambda: live('nt_task_vectors'))
        pairs = (
            (NT_PRESETS_3_FIBONACCI, hmr['NT_PRESETS_3_FIBONACCI']),
            (NT_PRESETS_4, hmr['NT_PRESETS_4']),
            (NT_PRESETS_5, hmr['NT_PRESETS_5']),
            (NT_PRESETS_6, hmr['NT_PRESETS_6']),
        )
        for mine, hmrs in pairs:
            for task in ('mnist', 'iris'):
                np.testing.assert_array_equal(np.asarray(mine[task]), np.asarray(hmrs[task]))

    def test_get_nt_for_task_reads_them(self):
        np.testing.assert_array_equal(np.asarray(get_nt_for_task('mnist', 4)),
                                      np.asarray([0.6, 0.4, 0.7, 1.0], dtype=np.float32))
        # Negative control: a name in no table still gets 1/n per channel
        np.testing.assert_array_equal(
            np.asarray(get_nt_for_task('cifar', 4)), np.full(4, 0.25, dtype=np.float32)
        )


class TestCppnOutputHelpers:
    """The helpers return the CPPN outputs create_config builds for the same configuration (EMR's
    layout: the weight, the activation channel, then the neuromodulation block's dedicated outputs).
    """

    CASES = [
        ('disabled', None, {}),
        ('cppn_output', None, {}),
        ('disabled', 'static_gating', {}),  # gate from the weights: no output
        ('disabled', 'static_gating', {'gate_source': 'cppn'}),
        ('cppn_output', 'static_gating', {'gate_source': 'cppn'}),
        ('disabled', 'context_gating', {'gate_source': 'cppn'}),
        ('disabled', 'true_neuromodulation_4nt', {}),  # Option A: receptors from the weights
        # Option B: four outputs
        ('disabled', 'true_neuromodulation_4nt', {'receptor_from_weight': False}),
        ('cppn_output', 'true_neuromodulation_4nt', {'receptor_from_weight': False}),
        ('disabled', 'modulatory_neurons', {'modulatory_source': 'cppn'}),
    ]

    @staticmethod
    def _built(df_mode, preset, block):
        if preset is None:
            config = create_base_config(population_size=10)
        else:
            config = create_config_with_neuromodulation(
                preset, recurrence_preset='feedforward', population_size=10
            )
            neuromodulation = config['algorithm_params']['emrhyperneat']['emr_hyperneat'][
                'neuromodulation'
            ]
            neuromodulation.update(block)
            if block.get('modulatory_source') == 'cppn':
                # the CPPN source refuses the weights mode's mod_decay
                neuromodulation.pop('mod_decay', None)
        if df_mode == 'cppn_output':
            config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
                'mode': 'cppn_output', 'num_activations': 4}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        block_out = config['algorithm_params']['emrhyperneat']['emr_hyperneat'].get(
            'neuromodulation'
        )
        return algorithm, block_out

    @pytest.mark.parametrize('df_mode,preset,block', CASES)
    def test_equal_to_the_built_cppn(self, df_mode, preset, block):
        algorithm, block_out = self._built(df_mode, preset, block)
        assert get_cppn_num_outputs(df_mode, block_out) == algorithm.num_cppn_outputs
        # The parsed configuration object gives the same count as its block
        assert (
            get_cppn_num_outputs(df_mode, algorithm.neuromod_config) == algorithm.num_cppn_outputs
        )
        receptor = algorithm._cppn_output_indices.get('receptor')
        expected_base = receptor if receptor is not None else algorithm.num_cppn_outputs
        assert get_cppn_receptor_base_idx(df_mode, block_out) == expected_base

    def test_option_b_base_equals_hmrs(self):
        hmr_base, hmr_outputs = hmr_value(
            'hmr_names.option_b_outputs', lambda: live('option_b_outputs', 4), inputs=(4,)
        )
        algorithm, block_out = self._built(
            'cppn_output', 'true_neuromodulation_4nt', {'receptor_from_weight': False}
        )
        assert get_cppn_receptor_base_idx('cppn_output', block_out) == hmr_base == 2
        assert get_cppn_num_outputs('cppn_output', block_out) == hmr_outputs


class TestEngineNames:
    def test_activation_attribute_names(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(create_base_config(population_size=10))
        assert algorithm._hidden_act_func is algorithm._hidden_act_fn
        assert algorithm._output_act_func is algorithm._output_act_fn
        replacement = lambda x: x  # noqa: E731
        algorithm._hidden_act_func = replacement
        assert algorithm._hidden_act_fn is replacement

    def test_multires_names_run_this_class_generation(self):
        problem = XORProblem()
        runs = []
        for name in (
            'run_generation',
            'run_generation_vmapped_multires',
            'run_generation_vmapped_multiresA',
        ):
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(
                algorithm.create_config(create_base_config(population_size=12)),
                problem,
                seed=DEFAULT_SEED,
            )
            state, metrics = getattr(algorithm, name)(state, problem)
            runs.append(
                (
                    float(metrics.best_fitness),
                    float(metrics.mean_fitness),
                    np.asarray(state.randkey),
                )
            )
        for best, mean, key in runs[1:]:
            assert (best, mean) == runs[0][:2]
            np.testing.assert_array_equal(key, runs[0][2])
        # The names are this class's own: the base class has no generation loop to bind them to
        for name in ('run_generation_vmapped_multires', 'run_generation_vmapped_multiresA'):
            assert name in vars(EMRHyperNEAT) and not hasattr(EMRHyperNEATChunking, name), name

    def test_configure_offloading_skip_dead_levels(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(create_base_config(population_size=10))
        assert algorithm.execution_config.skip_dead_levels is True
        algorithm.configure_offloading(skip_dead_levels=False)
        assert (
            algorithm.execution_config.skip_dead_levels is False
            and algorithm._skip_dead_levels is False
        )
        algorithm.configure_offloading()  # None keeps the configured value
        assert algorithm.execution_config.skip_dead_levels is False
        with pytest.raises(ValueError, match='skip_dead_levels must be a bool'):
            algorithm.configure_offloading(skip_dead_levels='no')

    @pytest.mark.parametrize('name,args', [
        ('evaluate_with_task_context', (None, None, None, None)),
        ('evaluate_multi_task', (None, None)),
        ('run_multi_task_generation', (None, None)),
    ])
    def test_hmr_multi_task_methods_are_refused(self, name, args):
        with pytest.raises(NotImplementedError, match='never ran'):
            getattr(EMRHyperNEAT(), name)(*args)


class TestHmrResultKeys:
    @staticmethod
    def _run(target, generations):
        problem = XORProblem()
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(create_base_config(population_size=12)),
            problem,
            seed=DEFAULT_SEED,
        )
        return algorithm.run_until_threshold(
            state, problem, target, generations, collect_history=True
        )

    def test_unreached_target(self):
        result = self._run(2.0, 3)
        assert result['converged'] is False and result['solved'] is False
        assert result['final_fitness'] == result['history'][-1]
        assert result['best_fitness'] == max(result['history'])
        assert set(result['sparse_hh_stats']) == {
            'discovery_time_ms',
            'hh_connections',
            'total_connections',
            'avg_connections_per_genome',
        }

    def test_reached_target(self):
        # every fitness reaches it, so the run stops after one generation
        result = self._run(-1.0, 3)
        assert (
            result['converged'] is True and result['solved'] is True and result['generations'] == 1
        )
        assert result['final_fitness'] == result['best_fitness'] == result['history'][0]

    def test_zero_generations(self):
        result = self._run(2.0, 0)
        assert result['generations'] == 0 and result['converged'] is False
        assert result['final_fitness'] == result['best_fitness'] == -float('inf')

    def test_best_and_final_are_told_apart(self):
        """best_fitness is the run's best, final_fitness the last generation's. Elitism keeps the
        champion, so in an evolved run the two coincide; the generations' bests are scripted here
        (0.6, 0.8, 0.7) instead."""
        problem = XORProblem()
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(create_base_config(population_size=12)),
            problem,
            seed=DEFAULT_SEED,
        )
        scripted = iter([0.6, 0.8, 0.7])
        algorithm.run_generation_verbose = lambda state, problem: (
            state,
            SimpleNamespace(best_fitness=next(scripted)),
        )
        result = algorithm.run_until_threshold(state, problem, 2.0, 3, collect_history=True)
        assert result['route'] == 'SINGLE_GPU' and result['generations'] == 3
        assert result['history'] == [0.6, 0.8, 0.7]
        assert result['best_fitness'] == 0.8 and result['final_fitness'] == 0.7
        assert result['converged'] is False and result['solved'] is False


class TestPlainOutputActivation:
    """HMR's dynamic-functions variant applied the configured output activation in plain mode (no
    per-node function, aggregation or neuromodulation), where 0.1.0 applied a sigmoid. The plain
    evaluation equals mode 'global' with a tanh hidden activation bit for bit, so a plain run with
    another output activation takes that path.
    """

    @staticmethod
    def _fitnesses(dynamic_functions, generations=2):
        config = create_base_config(population_size=12)
        # Elitism 2: at the default 15 a population of 12 is copied unmutated, so generation 2
        # would score generation 1's genomes again.
        config['algorithm_params']['emrhyperneat']['neat_selection'] = {'genome_elitism': 2}
        if dynamic_functions is not None:
            config['algorithm_params']['emrhyperneat']['emr_hyperneat'][
                'dynamic_functions'
            ] = dynamic_functions
        problem = XORProblem()
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
        out = []
        for _ in range(generations):
            state, _ = algorithm.run_generation(state, problem)
            out.append(np.asarray(algorithm._last_fitnesses))
        return algorithm, out

    def test_plain_equals_global_tanh_sigmoid(self):
        _, plain = self._fitnesses(None)
        _, glob = self._fitnesses(
            {'mode': 'global', 'hidden_activation': 'tanh', 'output_activation': 'sigmoid'}
        )
        for x, y in zip(plain, glob):
            np.testing.assert_array_equal(x, y)

    def test_plain_applies_another_output_activation(self):
        algorithm, plain = self._fitnesses({'mode': 'disabled', 'output_activation': 'tanh'})
        assert algorithm.dynamic_functions_mode == 'global' and algorithm._plain_output_activation
        _, glob = self._fitnesses(
            {'mode': 'global', 'hidden_activation': 'tanh', 'output_activation': 'tanh'}
        )
        for x, y in zip(plain, glob):
            np.testing.assert_array_equal(x, y)
        # Negative control: the output activation is applied (a sigmoid output scores differently)
        _, sigmoid = self._fitnesses(None)
        assert not np.array_equal(plain[0], sigmoid[0])


class TestHmrAttributeNames:
    def test_aggregation_mode(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(create_base_config(population_size=10))
        assert algorithm.aggregation_mode == algorithm.agg_mode == 'disabled'

    def test_depth_offload_enabled(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(create_base_config(population_size=10))
        assert algorithm._depth_offload_enabled is False
        config = create_base_config(population_size=10)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['execution'] = {
            'mode': 'depth_offload'
        }
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        assert algorithm._depth_offload_enabled is True


def _hmr_shaped_true_neuromodulation(population=10, depth=3):
    """The co-evolving-NT harness's HMR configuration (Option A, tanh receptors, 4 NT types,
    strength 5.0)."""
    return {
        'algorithm_params': {
            'hmrhyperneat': {
                'population_size': population,
                'substrate': {
                    'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                    'output_coords': [[0.0, 1.0]],
                },
                'hmr_hyperneat': {
                    'initial_depth': 0,
                    'max_depth': depth,
                    'variance_threshold': 0.03,
                    'neuromodulation': {
                        'mode': 'true_neuromodulation_4nt_option_a_tanh',
                        'modulation_strength': 5.0,
                    },
                },
                'neat_species': {'compatibility_threshold': 2.5, 'max_stagnation': 40},
            },
        },
    }


class _TwoInputXOR:
    """The co-evolving-NT harness's problem: XOR on two inputs, no bias, as its configuration's
    coordinates."""
    jitable = True
    input_shape = (2,)
    output_shape = (1,)
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]
    input_coords = [[-1.0, 0.0], [1.0, 0.0]]
    output_coords = [[0.0, 1.0]]

    def get_data(self):
        return list(zip(self.inputs, self.targets))


def _as_emr(config):
    params = dict(config['algorithm_params'])
    block = dict(params.pop('hmrhyperneat'))
    block['emr_hyperneat'] = dict(block.pop('hmr_hyperneat'), variance_rule='legacy')
    params['emrhyperneat'] = block
    return {**config, 'algorithm_params': params}


class TestSingleNetworkEvaluators:
    """HMR's single-network evaluators of its neuromodulation variant (5 experiment scripts call
    them): the population evaluator on a population of one, equal to its population row within
    float32 rounding (a batch of one compiles to another shape), and to HMR's own single-network
    evaluator on HMR's arrays, as recorded in the research repository (hmr_reference.py).
    """

    XOR_X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
    XOR_Y = np.array([[0.], [1.], [1.], [0.]], dtype=np.float32)
    NT = np.array([0.95, 0.05, 0.95, 1.0], dtype=np.float32)

    @staticmethod
    def _emr_arrays():
        algorithm = EMRHyperNEAT()
        problem = _TwoInputXOR()
        state = algorithm.initialize(
            algorithm.create_config(_as_emr(_hmr_shaped_true_neuromodulation())),
            problem,
            seed=DEFAULT_SEED,
        )
        algorithm.run_generation_verbose(state, problem, skip_metrics=True)
        arrays = (
            algorithm._cached_W1,
            algorithm._cached_W2,
            algorithm._neuromod_true['receptor_densities'],
            algorithm._neuromod_true['base_gains'],
        )
        return algorithm, arrays

    def test_single_equals_the_population_row(self):
        algorithm, (W1, W2, rd, bg) = self._emr_arrays()
        population = np.asarray(
            algorithm._eval_population_true_neuromodulated(
                W1, W2, rd, bg, self.NT, self.XOR_X, self.XOR_Y
            )
        )
        for genome in (0, 3, len(population) - 1):
            single = float(algorithm._eval_single_network_true_neuromodulated(
                W1[genome], W2[genome], rd[genome], bg[genome], self.NT, self.XOR_X, self.XOR_Y))
            # Within float32 rounding: a population of one compiles to another batch shape (one ulp
            # seen)
            np.testing.assert_allclose(single, float(population[genome]), rtol=0, atol=1e-6)
        fitness, hidden = algorithm._eval_single_network_true_neuromodulated_with_hidden(
            W1[0], W2[0], rd[0], bg[0], self.NT, self.XOR_X, self.XOR_Y)
        np.testing.assert_allclose(float(fitness), float(population[0]), rtol=0, atol=1e-6)
        assert hidden.shape[0] == self.XOR_X.shape[0]
        tasks = np.stack([self.NT, np.array([0.1, 0.9, 0.1, 1.0], dtype=np.float32)])
        mean = algorithm._eval_single_network_true_neuromodulated_multitask(
            W1[0], W2[0], rd[0], bg[0], tasks, [self.XOR_X, self.XOR_X], [self.XOR_Y, self.XOR_Y])
        expected = np.mean([float(algorithm._eval_single_network_true_neuromodulated(
            W1[0], W2[0], rd[0], bg[0], nt, self.XOR_X, self.XOR_Y)) for nt in tasks])
        np.testing.assert_allclose(float(mean), expected, rtol=0, atol=1e-7)

    def test_equal_to_hmrs_on_hmrs_arrays(self):
        genomes = (0, 4, 9)
        args = (
            _hmr_shaped_true_neuromodulation(),
            _TwoInputXOR(),
            DEFAULT_SEED,
            genomes,
            self.NT,
            self.XOR_X,
            self.XOR_Y,
        )
        hmr = hmr_value('hmr_names.single_network_on_hmrs_arrays',
                        lambda: live('single_network_on_hmrs_arrays', *args), inputs=args)
        W1, W2, rd, bg = hmr['W1'], hmr['W2'], hmr['rd'], hmr['bg']
        algorithm, _ = self._emr_arrays()
        for row, theirs in enumerate(hmr['theirs']):
            ours = float(algorithm._eval_single_network_true_neuromodulated(
                W1[row], W2[row], rd[row], bg[row], self.NT, self.XOR_X, self.XOR_Y))
            np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-6)
