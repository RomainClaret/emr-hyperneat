"""Tests for dynamic activation functions in EMR-HyperNEAT.

Tests cover:
- 7 selection modes (disabled, global, cppn_output, weight_interpretation, random_fixed, random_generation, modular)
- 18 activation functions (tanh, sigmoid, relu, sin, burst, etc.)
- H->H activation modes (every_iteration, tanh)
- Activation function palettes
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emr_hyperneat.emrhyperneat import (
    ACTIVATION_FUNCTIONS as ENGINE_ACTIVATIONS,
    activation_interpretation_signal,
    aggregation_from_weight_interpretation,
    continuous_to_index,
    level_sparsity_quantiles,
    sparsify_by_quantile,
    sparsify_winner_take_all,
    stp_depression,
)
from conftest import (
    EMRHyperNEAT,
    create_config_with_dynamic_functions,
    create_config_with_combined_recurrence,
    create_config_with_recurrence,
    create_base_config,
    assert_trajectories_agree,
    run_quick_evolution,
    assert_no_errors,
    assert_fitness_above,
    assert_positive_fitness,
    DYNAMIC_FUNCTION_PRESETS,
    XORProblem,
    DEFAULT_SEED,
    QUICK_GENERATIONS,
    STANDARD_GENERATIONS,
)


# Selection modes to test
SELECTION_MODES = ['disabled', 'global_tanh', 'global_sin', 'cppn_output_4', 'weight_interp_sign', 'random_fixed']

# The engine's activation names outside TestBioInspiredActivations.BIO_ACTIVATIONS, and no
# others: a name the engine does not have ('gaussian', 'elu', 'selu', 'swish', 'mish',
# 'leaky_relu') raises.
ACTIVATION_FUNCTIONS = ['tanh', 'sigmoid', 'relu', 'identity', 'sin', 'gauss', 'lelu', 'softmax']


def assert_palette_members(algorithm, names):
    """Every hidden node of the last champion computes one of the named functions."""
    record = algorithm.get_best_activation_indices()
    histogram = record['activation_histogram']
    assert set(histogram) <= set(names), (histogram, names)
    assert sum(histogram.values()) == record['num_hidden_nodes'] > 0


def _fitness_and_indices(dynamic_functions, aggregation=None, generations=1):
    """The population's fitness per generation and the last generation's activation indices (XOR,
    depth 3, population 20, the suite's seed). Generation 1 evaluates the seeded initial population.
    """
    config = create_base_config(max_depth=3, population_size=20)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['dynamic_functions'] = dynamic_functions
    if aggregation is not None:
        emr['aggregation'] = aggregation
    algorithm = EMRHyperNEAT()
    problem = XORProblem()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
    fitness = []
    for _ in range(generations):
        state, _metrics = algorithm.run_generation(state, problem)
        fitness.append(np.asarray(algorithm._cached_fitnesses))
    indices = algorithm._cached_act_indices
    return np.stack(fitness), (None if indices is None else np.asarray(indices))


def test_activation_lists_cover_the_engine():
    """The parametrized activation tests cover exactly the engine's names."""
    covered = set(ACTIVATION_FUNCTIONS) | set(TestBioInspiredActivations.BIO_ACTIVATIONS)
    assert covered == set(ENGINE_ACTIVATIONS)


class TestDynamicFunctionModes:
    """Tests for different dynamic function selection modes."""

    @pytest.mark.parametrize("mode", list(DYNAMIC_FUNCTION_PRESETS.keys()))
    def test_mode_initializes(self, algorithm, xor_problem, mode):
        """Verify each mode initializes without errors."""
        config = create_config_with_dynamic_functions(mode)
        config_obj = algorithm.create_config(config)

        # Should not raise
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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

    @pytest.mark.parametrize("interpretation", ['sign', 'magnitude_bio', 'variance'])
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
        config = create_config_with_dynamic_functions('weight_interp_magnitude_bio')
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
        """hierarchical_sparsity runs: magnitude_bio indices, then each level keeps its fraction of
        every sample's active values (HMR's; 0.1.0 ran the name as magnitude_bio)."""
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
        """sparsity_threshold runs: magnitude_bio indices, then a per-node kept fraction from the
        incoming weight magnitude (HMR's)."""
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
        """stp_inspired runs: magnitude_bio indices, incoming weights scaled by a per-node
        depression (HMR's)."""
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
        """combined runs: indices from magnitude, variance and the sign ratio together (HMR's)."""
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

    @pytest.mark.parametrize("interpretation", ['magnitude_bio', 'variance', 'sign'])
    def test_palette_members_on_every_hidden_node(self, algorithm, xor_problem, interpretation):
        """With a palette, every hidden node computes a palette function.

        The decoded palette position must be mapped to its member: evaluation expects palette
        members, and a node left on a position that is not itself a member would output 0.
        """
        from emr_hyperneat.emrhyperneat import ACTIVATION_LIST

        palette = [0, 4, 5, 13]  # tanh, sin, gauss, osc_adapt: positions 1-3 are not members
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {'mode': 'weight_interpretation',
                                    'interpretation': interpretation, 'palette': palette}
        run_quick_evolution(algorithm, config, xor_problem, generations=2)
        record = algorithm.get_best_activation_indices()
        histogram = record['activation_histogram']
        names = {ACTIVATION_LIST[i] for i in palette}
        assert set(histogram) <= names, (histogram, names)
        assert sum(histogram.values()) == record['num_hidden_nodes'] > 0

    @pytest.mark.parametrize('interpretation', ['magnitude', 'combined', 'sign', 'variance'])
    def test_index_interpretations_decode_their_own_indices(self, interpretation):
        """Each of these interpretations decodes its own indices, HMR's `magnitude` (magnitude_bio
        inverted) and `combined` included: generation 1 (the seeded population in both runs) differs
        from magnitude_bio's."""
        _, indices = _fitness_and_indices({'mode': 'weight_interpretation', 'num_activations': 6,
                                           'interpretation': interpretation})
        _, reference = _fitness_and_indices({'mode': 'weight_interpretation', 'num_activations': 6,
                                             'interpretation': 'magnitude_bio'})
        assert not np.array_equal(indices, reference)

    def test_magnitude_is_the_inverse_of_magnitude_bio(self):
        weights = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 10)) * 3
        np.testing.assert_allclose(
            np.asarray(activation_interpretation_signal(weights, 'magnitude')),
            -np.asarray(activation_interpretation_signal(weights, 'magnitude_bio')),
        )

    @pytest.mark.parametrize(
        'interpretation', ['hierarchical_sparsity', 'sparsity_threshold', 'stp_inspired']
    )
    def test_mechanism_interpretations_change_the_evaluation_not_the_indices(self, interpretation):
        """They select the activation as magnitude_bio does and add a sparsity or a scaling."""
        fitness, indices = _fitness_and_indices(
            {
                'mode': 'weight_interpretation',
                'num_activations': 6,
                'interpretation': interpretation,
            }
        )
        fitness_mb, indices_mb = _fitness_and_indices(
            {
                'mode': 'weight_interpretation',
                'num_activations': 6,
                'interpretation': 'magnitude_bio',
            }
        )
        np.testing.assert_array_equal(indices, indices_mb)
        assert not np.array_equal(fitness, fitness_mb)

    def test_stp_scales_the_evaluated_weights(self, algorithm, xor_problem):
        """The champion is evaluated with its incoming weights times the depression its own weights
        give."""
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'weight_interpretation', 'num_activations': 6, 'interpretation': 'stp_inspired'}
        run_quick_evolution(algorithm, config, xor_problem, generations=1)
        record = algorithm._last_eval
        champion = record['champion_index']
        unscaled = np.asarray(algorithm._cached_W1)[champion]
        expected = unscaled * np.asarray(stp_depression(jnp.asarray(unscaled[None])))[0][None, :]
        np.testing.assert_allclose(np.asarray(record['champion_args'][0])[0], expected, rtol=1e-6)

    def test_unknown_interpretation_raises(self):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'weight_interpretation', 'interpretation': 'not_a_real_method'}
        with pytest.raises(ValueError, match='dynamic_functions.interpretation must be one of'):
            EMRHyperNEAT().create_config(config)


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


def _random_indices(dynamic_functions=None, aggregation=None, seed=DEFAULT_SEED, generations=2,
                    algorithm=None):
    """Each generation's activation (or, with an aggregation block, aggregation) indices of a run."""
    config = create_base_config(max_depth=3, population_size=20)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['dynamic_functions'] = dynamic_functions or {'mode': 'disabled'}
    if aggregation is not None:
        emr['aggregation'] = aggregation
    algorithm = algorithm or EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=seed)
    draws = []
    for _ in range(generations):
        state, _metrics = algorithm.run_generation(state, XORProblem())
        draws.append(np.asarray(algorithm._cached_act_indices if aggregation is None
                                else algorithm._cached_ih_agg_indices))
    return np.stack(draws)


class TestRandomScopes:
    """The random modes' draw.

    Scope 'genome', the default, draws each population slot independently from the run's key, as
    HMR drew its negative controls. 0.1.0 drew one assignment for every genome from a fixed key, so
    every genome and every seed of a random-control arm had the same functions: that draw is scope
    'shared', kept bit for bit to reproduce such runs.
    """

    @pytest.mark.parametrize('mode', ['random_fixed', 'random_generation'])
    def test_genome_scope_draws_per_slot_and_per_run(self, mode):
        first = _random_indices({'mode': mode, 'num_activations': 6}, seed=42)
        second = _random_indices({'mode': mode, 'num_activations': 6}, seed=43)
        assert not np.array_equal(first[0, 0], first[0, 1])  # two slots of one generation
        assert not np.array_equal(first[0], second[0])  # two runs
        assert np.array_equal(first[0], first[1]) == (mode == 'random_fixed')

    def test_genome_scope_with_a_seed_is_the_same_in_every_run(self):
        block = {'mode': 'random_generation', 'num_activations': 6, 'random_seed': 7}
        first, second = _random_indices(block, seed=42), _random_indices(block, seed=43)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first[0, 0], first[0, 1])

    @pytest.mark.parametrize('mode', ['random_fixed', 'random_generation'])
    def test_shared_scope_is_one_assignment_for_every_genome_and_run(self, mode):
        block = {'mode': mode, 'num_activations': 6, 'random_scope': 'shared'}
        first, second = _random_indices(block, seed=42), _random_indices(block, seed=43)
        np.testing.assert_array_equal(first, second)
        assert (first[0] == first[0, :1]).all()
        # The draw itself, as 0.1.0 made it: PRNGKey(42) fixed, PRNGKey(generation) per generation
        # (the first evaluated generation is number 0)
        key = jax.random.PRNGKey(42 if mode == 'random_fixed' else 0)
        np.testing.assert_array_equal(
            first[0, 0], np.asarray(jax.random.randint(key, (first.shape[-1],), 0, 6)))

    def test_palette_members_in_both_scopes(self):
        for scope in ('genome', 'shared'):
            draws = _random_indices({'mode': 'random_generation', 'palette': [0, 4, 5, 13],
                                     'random_scope': scope})
            assert set(np.unique(draws)) <= {0, 4, 5, 13}

    @pytest.mark.parametrize('mode', ['random_fixed', 'random_generation'])
    def test_aggregation_draws_per_slot(self, mode):
        draws = _random_indices(aggregation={'mode': mode})
        assert not np.array_equal(draws[0, 0], draws[0, 1])
        shared = _random_indices(aggregation={'mode': mode, 'random_scope': 'shared'})
        assert (shared[0] == shared[0, :1]).all()

    def test_a_reused_object_draws_again_for_the_next_run(self):
        """initialize resets the fixed draw and the generation count rather than keeping them on
        the object across runs, so a reused object gives what a fresh one gives."""
        algorithm = EMRHyperNEAT()
        first = _random_indices({'mode': 'random_fixed', 'num_activations': 6}, seed=42,
                                algorithm=algorithm)
        second = _random_indices({'mode': 'random_fixed', 'num_activations': 6}, seed=43,
                                 algorithm=algorithm)
        assert not np.array_equal(first[0], second[0])
        block = {'mode': 'random_generation', 'num_activations': 6, 'random_scope': 'shared'}
        _random_indices(block, algorithm=algorithm)
        np.testing.assert_array_equal(_random_indices(block, algorithm=algorithm),
                                      _random_indices(block))

    def test_unknown_scope_raises(self):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'random_fixed', 'random_scope': 'population'}
        with pytest.raises(ValueError, match='random_scope must be one of'):
            EMRHyperNEAT().create_config(config)


def _recurrent_random_fixed(block, preset, depth, generations):
    """A recurrent run with random_fixed activation (or aggregation), XOR, population 20: the
    algorithm, its last state, and per generation the evaluated position count, the evaluated
    positions (None: all) and the indices the evaluation used."""
    config = create_config_with_recurrence(preset, max_depth=depth, population_size=20,
                                           iteration_level=1, hh_cache_enabled=False)
    config['algorithm_params']['emrhyperneat']['emr_hyperneat'][block] = (
        {'mode': 'random_fixed', 'num_activations': 6} if block == 'dynamic_functions'
        else {'mode': 'random_fixed', 'num_aggregations': 4})
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
    rows = []
    for _ in range(generations):
        state, metrics = algorithm.run_generation(state, XORProblem())
        used = (algorithm._cached_act_indices if block == 'dynamic_functions'
                else algorithm._cached_ih_agg_indices)
        rows.append((metrics.custom_metrics['total_positions'],
                     algorithm._last_eval['eval_positions'], np.asarray(used)))
    return algorithm, state, rows


class TestRandomFixedWithRecurrence:
    """random_fixed keeps a function per population slot and grid position. With hidden-to-hidden
    connections a generation evaluates its active positions only, a count that changes between
    generations, so the fixed draw covers the whole position axis and is gathered at the positions
    evaluated: a draw of generation 0's count, reused whole, fails at the first generation whose
    count differs ("Incompatible shapes for broadcasting"). A feed-forward run evaluates the whole
    axis, so the gather does not change its draw."""

    @pytest.mark.parametrize('block', ['dynamic_functions', 'aggregation'])
    @pytest.mark.parametrize('preset', ['hidden_only', 'full_recurrent'])
    @pytest.mark.parametrize('depth', [2, 3])
    def test_a_recurrent_run_survives_a_changing_active_count(self, depth, preset, block):
        _, _, rows = _recurrent_random_fixed(block, preset, depth, generations=4)
        counts = [count for count, _, _ in rows]
        assert len(set(counts)) > 1, counts  # the case that failed
        for count, _, used in rows:
            assert used.shape == (20, count)

    @pytest.mark.parametrize('block', ['dynamic_functions', 'aggregation'])
    def test_a_position_keeps_its_function_across_generations(self, block):
        algorithm, _, rows = _recurrent_random_fixed(block, 'hidden_only', 3, generations=4)
        drawn = np.asarray(algorithm._random_fixed_indices if block == 'dynamic_functions'
                           else algorithm._random_fixed_agg_indices)
        assert drawn.shape == (20, algorithm._cached_W1.shape[2])  # every grid position
        by_position = {}
        for _, positions, used in rows:
            positions = np.arange(drawn.shape[1]) if positions is None else np.asarray(positions)
            np.testing.assert_array_equal(used, drawn[:, positions])
            for column, position in enumerate(positions.tolist()):
                by_position.setdefault(position, []).append(used[:, column])
        repeated = [columns for columns in by_position.values() if len(columns) > 1]
        assert repeated
        for columns in repeated:
            for column in columns[1:]:
                np.testing.assert_array_equal(column, columns[0])

    def test_the_champion_is_rescored_with_its_generation_s_functions(self):
        """evaluate_on_data re-runs the last generation's champion with the indices gathered at
        that generation's active positions."""
        algorithm, state, rows = _recurrent_random_fixed('dynamic_functions', 'hidden_only', 3,
                                                         generations=4)
        assert len({count for count, _, _ in rows}) > 1
        result = algorithm.evaluate_on_data(state, algorithm._cached_inputs,
                                            algorithm._cached_targets)
        assert_trajectories_agree({'rescored': [result['fitness']],
                                   'selected': [float(np.max(algorithm._cached_fitnesses))]},
                                  'recurrent random_fixed champion')


class TestHHActivationModes:
    """Tests for H->H activation modes."""

    @pytest.mark.parametrize("hh_mode", ['tanh', 'every_iteration'])
    def test_hh_activation_modes(self, algorithm, xor_problem, hh_mode):
        """Verify both H->H activation modes work."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only',
            hh_activation_mode=hh_mode
        )
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, f"H->H mode {hh_mode} failed")
        assert_positive_fitness(result)

    def test_hh_tanh_config(self, algorithm, xor_problem):
        """Verify tanh mode is properly configured."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only',
            hh_activation_mode='tanh'
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

        assert algorithm.hh_activation_mode == 'tanh'

    def test_hh_initial_only_is_rejected(self, algorithm):
        """initial_only would apply no configured activation, so it is refused."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only',
            hh_activation_mode='initial_only'
        )
        with pytest.raises(ValueError, match="no longer accepted"):
            algorithm.create_config(config)

    def test_hh_default_is_every_iteration(self, algorithm):
        """The configured activation applies at every settling step by default."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4', recurrence_preset='hidden_only')
        algorithm.create_config(config)
        assert algorithm.hh_activation_mode == 'every_iteration'

    def test_hh_every_iteration_config(self, algorithm, xor_problem):
        """Verify every_iteration mode is properly configured."""
        config = create_config_with_dynamic_functions(
            'cppn_output_4',
            recurrence_preset='hidden_only',
            hh_activation_mode='every_iteration'
        )
        config_obj = algorithm.create_config(config)
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        """A palette given by name (dynamic_functions.palette, not `activation_palette`, which EMR
        refuses) reaches every hidden node."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'palette': ['tanh', 'sin', 'relu'],
            'interpretation': 'sign',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result)
        assert_positive_fitness(result)
        assert_palette_members(algorithm, ['tanh', 'sin', 'relu'])


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

    Multi-output CPPNs (num_cppn_outputs > 1) go through the multi-output query path, which
    preserves their (N, K) outputs: batch_query_population_multi_source_chunked calls
    .flatten() on CPPN outputs, which would turn (N, K) into (N*K,) and corrupt the shapes in
    H->H discovery.
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
    """Slow runs: every selection mode on XOR and AND, and a global sin on four-bit parity."""

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
    """Tests for modular dynamic function mode: HMR's three orthogonal layers (activation,
    sparsity, scaling) and its aggregation method.

    Each layer combination equals the weight interpretation it composes, bit for bit; a
    sparsity or scaling layer changes the evaluation and not the activation indices; the
    aggregation method equals its interpretation; palettes stay respected.
    """

    @pytest.mark.parametrize('layers, equivalent', [
        ({'activation_method': 'none'}, {'mode': 'global', 'hidden_activation': 'tanh'}),
        ({'activation_method': 'magnitude_bio'},
         {'mode': 'weight_interpretation', 'interpretation': 'magnitude_bio'}),
        ({'activation_method': 'magnitude_bio', 'sparsity_method': 'hierarchical'},
         {'mode': 'weight_interpretation', 'interpretation': 'hierarchical_sparsity'}),
        ({'activation_method': 'magnitude_bio', 'sparsity_method': 'threshold'},
         {'mode': 'weight_interpretation', 'interpretation': 'sparsity_threshold'}),
        ({'activation_method': 'magnitude_bio', 'scaling_method': 'stp_inspired'},
         {'mode': 'weight_interpretation', 'interpretation': 'stp_inspired'}),
    ])
    def test_layers_equal_their_weight_interpretation(self, layers, equivalent):
        modular, _ = _fitness_and_indices(
            {'mode': 'modular', 'num_activations': 6, 'modular': layers}, generations=2
        )
        reference, _ = _fitness_and_indices(dict(equivalent, num_activations=6), generations=2)
        np.testing.assert_array_equal(modular, reference)

    @pytest.mark.parametrize(
        'layer',
        [
            {'sparsity_method': 'hierarchical'},
            {'sparsity_method': 'threshold'},
            {'sparsity_method': 'wta'},
            {'scaling_method': 'stp_inspired'},
        ],
    )
    def test_each_layer_changes_the_evaluation_not_the_indices(self, layer):
        """Generation 1 evaluates the seeded population in both runs: same activation indices, other
        fitness."""
        base = {
            'mode': 'modular',
            'num_activations': 6,
            'modular': {'activation_method': 'magnitude_bio'},
        }
        with_layer = {'mode': 'modular', 'num_activations': 6,
                      'modular': dict(activation_method='magnitude_bio', **layer)}
        fitness_base, indices_base = _fitness_and_indices(base)
        fitness_layer, indices_layer = _fitness_and_indices(with_layer)
        np.testing.assert_array_equal(indices_layer, indices_base)
        assert not np.array_equal(fitness_layer, fitness_base)

    @pytest.mark.parametrize('method', ['magnitude_bio', 'variance', 'sign_uniformity', 'fan_in'])
    def test_aggregation_method_equals_its_interpretation(self, method):
        modular, _ = _fitness_and_indices(
            {'mode': 'disabled', 'modular': {'aggregation_method': method}},
            aggregation={'mode': 'modular'},
            generations=2,
        )
        reference, _ = _fitness_and_indices(
            {'mode': 'disabled'},
            generations=2,
            aggregation={'mode': 'weight_interpretation', 'interpretation': method},
        )
        np.testing.assert_array_equal(modular, reference)

    def test_aggregation_method_none_is_sum(self):
        modular, _ = _fitness_and_indices(
            {'mode': 'disabled', 'modular': {'aggregation_method': 'none'}},
            aggregation={'mode': 'modular'},
            generations=2,
        )
        reference, _ = _fitness_and_indices({'mode': 'disabled'}, generations=2)
        np.testing.assert_array_equal(modular, reference)

    def test_palette_members_under_modular(self, algorithm, xor_problem):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'modular', 'palette': ['tanh', 'sin', 'gauss', 'osc_adapt'],
            'modular': {'activation_method': 'variance', 'sparsity_method': 'wta'}}
        run_quick_evolution(algorithm, config, xor_problem, generations=2)
        assert_palette_members(algorithm, ['tanh', 'sin', 'gauss', 'osc_adapt'])

    @pytest.mark.parametrize('wta_k', [0.05, 0.5, 1.0])
    def test_wta_k_percent_values(self, algorithm, xor_problem, wta_k):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'modular',
            'modular': {'sparsity_method': 'wta', 'sparsity': {'wta_k_percent': wta_k}},
        }
        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)
        assert_no_errors(result, f"WTA k={wta_k} failed")

    @pytest.mark.parametrize(
        'dynamic_functions',
        [
            {
                'mode': 'weight_interpretation',
                'num_activations': 6,
                'interpretation': 'hierarchical_sparsity',
            },
            {
                'mode': 'modular',
                'num_activations': 6,
                'modular': {
                    'activation_method': 'variance',
                    'sparsity_method': 'wta',
                    'scaling_method': 'stp_inspired',
                },
            },
        ],
    )
    def test_streaming_evaluates_the_same(self, dynamic_functions):
        """Streaming builds the matrices its own way and shares the decode and the evaluation."""
        def fitness(mode):
            config = create_base_config(max_depth=3, population_size=20)
            emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
            emr['dynamic_functions'] = dynamic_functions
            emr['execution'] = {'mode': mode}
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(
                algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED
            )
            values = []
            for _ in range(2):
                state, _metrics = algorithm.run_generation(state, XORProblem())
                values.append(np.asarray(algorithm._cached_fitnesses))
            assert algorithm.enable_streaming == (mode == 'streaming')
            return np.stack(values)
        np.testing.assert_array_equal(fitness('streaming'), fitness('standard'))

    def test_wta_keeping_everything_is_no_sparsity(self):
        """k_percent 1.0 keeps every active value, so the evaluation is the activation-only one."""
        kept, _ = _fitness_and_indices(
            {
                'mode': 'modular',
                'num_activations': 6,
                'modular': {
                    'activation_method': 'magnitude_bio',
                    'sparsity_method': 'wta',
                    'sparsity': {'wta_k_percent': 1.0},
                },
            }
        )
        base, _ = _fitness_and_indices({'mode': 'modular', 'num_activations': 6,
                                        'modular': {'activation_method': 'magnitude_bio'}})
        np.testing.assert_array_equal(kept, base)

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
        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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
        """Verify each activation_method value parses and runs without raising.

        Its effect is checked by TestModularMode.test_layers_equal_their_weight_interpretation.
        """
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
        """Verify each sparsity_method value parses and runs without raising.

        Its effect is checked by
        TestModularMode.test_each_layer_changes_the_evaluation_not_the_indices.
        """
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

    These include the Izhikevich-inspired functions and those of the 'phase4_all' palette.
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

        algorithm.initialize(config_obj, xor_problem, seed=DEFAULT_SEED)

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

    def test_level_quantiles(self):
        quantiles = np.asarray(
            level_sparsity_quantiles(
                (0, 1, 5, 21), {'level_0': 0.05, 'level_1': 0.20, 'level_2_plus': 0.40}
            )
        )
        np.testing.assert_allclose(quantiles, [0.95] + [0.8] * 4 + [0.6] * 16, rtol=1e-6)

    def test_the_level_block_reaches_the_evaluation(self):
        """A different level_2_plus changes the fitness of hierarchical sparsity: the block reaches
        the
        evaluation."""
        def fitness(kept):
            return _fitness_and_indices({'mode': 'weight_interpretation', 'num_activations': 6,
                                         'interpretation': 'hierarchical_sparsity',
                                         'sparsity': {'level_2_plus': kept}})[0]
        assert not np.array_equal(fitness(0.40), fitness(0.90))

    def test_quantile_sparsity_counts_active_nodes_only(self):
        """HMR took the quantile over every position, inactive ones included at the value their
        function gives at zero input; here only active nodes count. With an inactive node at 1.0
        (gauss at 0), HMR's median of [0.1, 0.5, 0.7, 0.9, 1.0] is 0.7 and drops -0.5; over the four
        active values it is 0.5."""
        hidden = jnp.array([[0.1, -0.5, 0.9, 1.0, 0.7]])
        mask = jnp.array([True, True, True, False, True])
        kept = sparsify_by_quantile(hidden, jnp.full(5, 0.5), mask)
        np.testing.assert_array_equal(np.asarray(kept), np.float32([[0.0, -0.5, 0.9, 1.0, 0.7]]))

    def test_winner_take_all_counts_active_nodes_only(self):
        hidden = jnp.array([[0.1, -0.5, 0.9, 1.0, 0.7]])
        mask = jnp.array([True, True, True, False, True])
        kept = sparsify_winner_take_all(hidden, 0.5, mask)      # 2 of 4 active nodes
        np.testing.assert_array_equal(np.asarray(kept), np.float32([[0.0, 0.0, 0.9, 1.0, 0.7]]))

    def test_winner_take_all_keeps_at_least_one(self):
        hidden = jnp.array([[0.1, -0.5, 0.9]])
        kept = sparsify_winner_take_all(hidden, 0.01, jnp.array([True, True, True]))
        np.testing.assert_array_equal(np.asarray(kept), np.float32([[0.0, 0.0, 0.9]]))

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
    """dynamic_functions.critical_periods is not implemented (HMR's variants parse it and print a
    line, and nothing changes a mutation rate), so `enabled: True` raises; a disabled block is
    accepted."""

    def test_disabled_is_accepted(self):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'global', 'hidden_activation': 'tanh', 'critical_periods': {'enabled': False}}
        EMRHyperNEAT().create_config(config)

    @pytest.mark.parametrize('mode', ['global', 'cppn_output', 'weight_interpretation'])
    def test_enabled_raises(self, mode):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': mode, 'critical_periods': {'enabled': True, 'phase1_end': 0.2, 'phase2_end': 0.5,
                                                'min_plasticity': 0.3}}
        with pytest.raises(ValueError, match='critical_periods is not implemented'):
            EMRHyperNEAT().create_config(config)


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
        """A two-function palette given by indices in dynamic_functions.palette reaches every hidden
        node."""
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'weight_interpretation',
            'palette': [0, 4],
            'interpretation': 'sign',
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Minimal custom palette failed")
        assert_palette_members(algorithm, ['tanh', 'sin'])

    def test_custom_palette_large(self, algorithm, xor_problem):
        """A six-function palette of engine names ('gauss', not 'gaussian') reaches every hidden
        node under cppn_output."""
        names = ['tanh', 'sin', 'relu', 'sigmoid', 'gauss', 'identity']
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'cppn_output',
            'palette': names,
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Large custom palette failed")
        assert_palette_members(algorithm, names)

    def test_modular_all_methods_combined(self, algorithm, xor_problem):
        """Modular mode with all three layers, and the level fractions hierarchical sparsity reads
        (0.1.0's version of this test named scaling_method 'linear', which no engine had, and ran
        fixed tanh).
        """
        config = create_base_config()
        hmr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        hmr['dynamic_functions'] = {
            'mode': 'modular',
            'num_activations': 6,
            'modular': {
                'activation_method': 'magnitude_bio',
                'sparsity_method': 'hierarchical',
                'scaling_method': 'stp_inspired',
                'sparsity': {
                    'level_0': 0.05,
                    'level_1': 0.15,
                    'level_2_plus': 0.30,
                },
            },
        }

        result = run_quick_evolution(algorithm, config, xor_problem, generations=QUICK_GENERATIONS)

        assert_no_errors(result, "Modular with all methods failed")
        assert (algorithm._df_sparsity_method, algorithm._df_scaling_method) == (
            'hierarchical',
            'stp_inspired',
        )
        assert algorithm._df_sparsity_params['level_1'] == 0.15


class TestIndexDecode:
    """index_decode. The default 'tanh', 0.1.0's decode, applies tanh to values that are already in [-1, 1] (the
    CPPN's channels, whose output transform is tanh, and every weight-interpretation signal), so a palette of 9 or
    more functions never reaches its end entries: 18 functions, positions 2 to 15. 'uniform' bins the value itself,
    every bin equally wide."""

    def test_tanh_misses_the_end_bins_and_uniform_does_not(self):
        import jax.numpy as jnp
        from emr_hyperneat.emrhyperneat import continuous_to_index
        values = jnp.linspace(-1.0, 1.0, 18001)
        tanh_bins = np.asarray(continuous_to_index(values, 18))
        uniform_bins = np.asarray(continuous_to_index(values, 18, 'uniform'))
        assert (tanh_bins.min(), tanh_bins.max()) == (2, 15)
        assert (uniform_bins.min(), uniform_bins.max()) == (0, 17)
        counts = np.bincount(uniform_bins, minlength=18)
        assert counts.max() - counts.min() <= 2           # equally wide bins

    def test_default_is_the_old_decode(self):
        from emr_hyperneat.emrhyperneat import continuous_to_index, continuous_to_palette_index
        values = jax.random.uniform(jax.random.PRNGKey(0), (1000,), minval=-1.0, maxval=1.0)
        old = np.clip(np.floor((np.tanh(np.asarray(values)) + 1.0) / 2.0 * 7), 0, 6).astype(np.int32)
        np.testing.assert_array_equal(np.asarray(continuous_to_index(values, 7)), old)
        palette = np.array([4, 11, 12, 5, 0, 1, 2])
        np.testing.assert_array_equal(np.asarray(continuous_to_palette_index(values, palette)), palette[old])

    def test_uniform_reaches_every_position_of_an_18_function_palette(self):
        def reached(decode):
            config = create_base_config(max_depth=4, population_size=100)
            config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
                'mode': 'cppn_output', 'palette': list(range(18)), 'index_decode': decode}
            algorithm = EMRHyperNEAT()
            state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
            algorithm.run_generation(state, XORProblem())
            return set(np.unique(np.asarray(algorithm._cached_act_indices)).tolist())
        assert reached('tanh') <= set(range(2, 16))
        assert reached('uniform') == set(range(18))

    def test_aggregation_decode(self):
        from emr_hyperneat.emrhyperneat import aggregation_from_weight_interpretation
        weights = jax.random.normal(jax.random.PRNGKey(1), (4, 3, 50)) * 3
        for interpretation in ('magnitude_bio', 'variance', 'sign_uniformity', 'fan_in'):
            uniform = aggregation_from_weight_interpretation(weights, interpretation, 6, 'uniform')
            assert int(uniform.min()) >= 0 and int(uniform.max()) <= 5
        default = aggregation_from_weight_interpretation(weights, 'fan_in', 6)
        np.testing.assert_array_equal(np.asarray(default),
                                      np.asarray(aggregation_from_weight_interpretation(weights, 'fan_in', 6, 'tanh')))

    @pytest.mark.parametrize('section', ['dynamic_functions', 'aggregation'])
    def test_unknown_decode_raises(self, section):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat'][section] = {
            'mode': 'cppn_output', 'index_decode': 'linear'}
        with pytest.raises(ValueError, match=f'{section}.index_decode must be one of'):
            EMRHyperNEAT().create_config(config)

    def test_the_substrate_config_carries_both_decodes(self):
        config = create_base_config()
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['dynamic_functions'] = {'mode': 'cppn_output', 'index_decode': 'uniform'}
        emr['aggregation'] = {'mode': 'cppn_output', 'index_decode': 'uniform'}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        substrate_config = algorithm.substrate_config()
        assert (substrate_config.activation_index_decode, substrate_config.aggregation_index_decode) == (
            'uniform', 'uniform')


def _with_raw_queries(dynamic_functions, aggregation=None):
    """One generation, and the CPPN's raw input-to-node outputs (pop, inputs, positions, outputs)
    for the population it evaluated, recomputed outside the generation loop."""
    from emr_hyperneat import (
        emrhyperneat as engine,
    )
    from emr_hyperneat.emrhyperneat_base import (
        get_hierarchical_grid,
    )
    config = create_base_config(max_depth=3, population_size=20)
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    emr['dynamic_functions'] = dynamic_functions
    if aggregation is not None:
        emr['aggregation'] = aggregation
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=DEFAULT_SEED)
    transformed = algorithm._compiled_transform_batch(state, algorithm._compiled_ask(state))
    raw = engine.batch_query_population_multi_source_multi_output(
        state,
        transformed,
        algorithm._cached_input_coords,
        get_hierarchical_grid(algorithm.max_depth).all_positions,
        True,
        algorithm._jitted_cppn_forward,
    )
    algorithm.run_generation(state, XORProblem())
    raw = np.asarray(raw)
    return algorithm, raw if raw.ndim == 4 else raw[..., None]


class TestHmrReads:
    """cppn_read 'input_mean' and weight_source 'raw': HMR's dynamic-functions and aggregation
    variants averaged a node's function channel over its input-to-node queries, where EMR reads the
    node's self-connection, and their weight interpretations read the raw weight channel, where EMR
    reads W1 (scaled and thresholded). Opt-ins, for runs that reproduce HMR's; checked against the
    queries recomputed outside the loop."""

    def test_input_mean_is_the_channel_averaged_over_the_input_queries(self):
        algorithm, raw = _with_raw_queries(
            {'mode': 'cppn_output', 'num_activations': 6, 'cppn_read': 'input_mean'}
        )
        channel = algorithm._cppn_output_indices['activation']
        expected = continuous_to_index(jnp.mean(jnp.asarray(raw[..., channel]), axis=1), 6)
        np.testing.assert_array_equal(
            np.asarray(algorithm._cached_act_indices), np.asarray(expected)
        )

    @pytest.mark.parametrize('interpretation', ['magnitude', 'sign', 'combined'])
    def test_raw_weight_source_reads_the_weight_channel(self, interpretation):
        algorithm, raw = _with_raw_queries(
            {
                'mode': 'weight_interpretation',
                'num_activations': 6,
                'interpretation': interpretation,
                'weight_source': 'raw',
            }
        )
        expected = continuous_to_index(
            activation_interpretation_signal(jnp.asarray(raw[..., 0]), interpretation), 6
        )
        np.testing.assert_array_equal(
            np.asarray(algorithm._cached_act_indices), np.asarray(expected)
        )

    def test_raw_weight_source_for_aggregation(self):
        algorithm, raw = _with_raw_queries({'mode': 'disabled'}, {
            'mode': 'weight_interpretation', 'interpretation': 'variance', 'weight_source': 'raw',
            'num_aggregations': 4})
        expected = aggregation_from_weight_interpretation(jnp.asarray(raw[..., 0]), 'variance', 4)
        np.testing.assert_array_equal(
            np.asarray(algorithm._cached_ih_agg_indices), np.asarray(expected)
        )

    def test_the_defaults_are_emr_reads(self):
        """Without the opt-ins nothing changes: self-connection reads and W1."""
        default, _ = _fitness_and_indices({'mode': 'cppn_output', 'num_activations': 6})
        explicit, _ = _fitness_and_indices(
            {
                'mode': 'cppn_output',
                'num_activations': 6,
                'cppn_read': 'self',
                'weight_source': 'w1',
            }
        )
        np.testing.assert_array_equal(default, explicit)

    @pytest.mark.parametrize(
        'dynamic_functions, extra, match',
        [
            (
                {'mode': 'cppn_output', 'cppn_read': 'input_mean'},
                {'recurrence': {'preset': 'hidden_only'}},
                'hidden-to-hidden recurrence',
            ),
            (
                {'mode': 'cppn_output', 'cppn_read': 'input_mean'},
                {'execution': {'mode': 'streaming'}},
                'needs the standard execution mode',
            ),
            (
                {'mode': 'weight_interpretation', 'cppn_read': 'input_mean'},
                {},
                "needs mode 'cppn_output'",
            ),
            (
                {'mode': 'cppn_output', 'cppn_read': 'input_mean', 'num_cppn_outputs': 1},
                {},
                'dedicated CPPN channel',
            ),
            ({'mode': 'cppn_output', 'weight_source': 'raw'}, {}, 'reads no weight interpretation'),
            ({'mode': 'cppn_output', 'cppn_read': 'node'}, {}, 'cppn_read must be one of'),
            (
                {'mode': 'weight_interpretation', 'weight_source': 'cppn'},
                {},
                'weight_source must be one of',
            ),
        ],
    )
    def test_refusals(self, dynamic_functions, extra, match):
        config = create_base_config()
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        emr['dynamic_functions'] = dynamic_functions
        emr.update(extra)
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_the_substrate_api_refuses_them(self):
        config = create_base_config()
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'cppn_output', 'cppn_read': 'input_mean'}
        algorithm = EMRHyperNEAT()
        algorithm.create_config(config)
        with pytest.raises(ValueError, match="HMR's reads"):
            algorithm.substrate_config()
