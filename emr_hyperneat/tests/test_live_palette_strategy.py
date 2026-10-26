"""HMR's live palette strategy.

HMR's dynamic-functions variant let a runner's strategy object set the palette after each generation
(`set_palette_strategy`, `dynamic_functions.live_strategy`). When the palette grew, it perturbed a
share of the population toward the new bins, but it added the strength to column 0 of every node of
the chosen genomes, which is TensorNEAT's node key, not the bias, and it split `state.randkey` to
choose them. EMR adds the strength to the bias of the CPPN's activation output node alone and draws
the genomes from a key folded from `state.randkey`, which stays as it is. The negative control is
HMR's own function on the same state, as recorded in the research repository (hmr_reference.py).
"""

import jax
import numpy as np
import pytest

from conftest import (DEFAULT_SEED, EMRHyperNEAT, XORProblem, create_base_config,
                      create_config_with_dynamic_functions)
from hmr_reference import hmr_value, live  # type: ignore[import-not-found]

DEPTH, POP = 2, 20
PALETTE = [0, 1, 2, 3]


def _config(live_strategy=None, **dynamic_functions):
    overrides = {'palette': list(PALETTE), **dynamic_functions}
    if live_strategy is not None:
        overrides['live_strategy'] = live_strategy
    return create_config_with_dynamic_functions(
        'cppn_output_4', max_depth=DEPTH, population_size=POP, **overrides
    )


class _Strategy:
    """A strategy object in HMR's interface: its palette is read after each generation; it records
    what it is told."""

    def __init__(self, palettes):
        self.palettes = palettes      # generation told -> palette from then on
        self.told = []

    def get_active_palette(self):
        told = self.told[-1][0] if self.told else 0
        return list(self.palettes[max(g for g in self.palettes if g <= told)])

    def post_generation_update(self, generation, best_fitness, fitness_history=None, **kwargs):
        self.told.append((generation, best_fitness, list(fitness_history)))


def _run(config, generations, strategy=None):
    algorithm = EMRHyperNEAT()
    problem = XORProblem()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
    if strategy is not None:
        algorithm.set_palette_strategy(strategy)
    fitnesses, timings = [], []
    for _ in range(generations):
        state, metrics = algorithm.run_generation(state, problem)
        fitnesses.append(np.asarray(algorithm._cached_fitnesses))
        timings.append(metrics.custom_metrics['step_timings'])
    return algorithm, state, fitnesses, timings


def _after_one_generation(live_strategy):
    algorithm = EMRHyperNEAT()
    problem = XORProblem()
    state = algorithm.initialize(
        algorithm.create_config(_config(live_strategy)), problem, seed=DEFAULT_SEED
    )
    algorithm.set_palette_strategy(_Strategy({0: PALETTE}))
    state, _ = algorithm.run_generation(state, problem)
    return algorithm, state


def _equal_or_both_nan(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


class TestPerturbation:
    LIVE = {'enabled': True, 'perturbation_rate': 0.25, 'perturbation_strength': 0.3}

    def test_column_1_is_the_bias(self):
        """The column the fix writes: TensorNEAT's node rows are the key, then the bias."""
        algorithm, _ = _after_one_generation(self.LIVE)
        gene = algorithm.neat_algo.genome.node_gene
        assert list(gene.fixed_attrs) == ['index'] and gene.custom_attrs[0] == 'bias'

    def test_only_the_activation_output_bias_moves(self):
        algorithm, state = _after_one_generation(self.LIVE)
        before = np.asarray(state.pop_nodes)
        after_state = algorithm._perturb_activation_outputs(state, 4, 5)
        after = np.asarray(after_state.pop_nodes)
        output_key = int(
            np.asarray(algorithm.neat_algo.genome.output_idx)[
                algorithm._cppn_output_indices['activation']
            ]
        )
        moved = ~_equal_or_both_nan(after, before)
        genomes, rows, columns = np.nonzero(moved)
        assert set(columns) == {1}
        assert np.all(before[genomes, rows, 0] == output_key)
        assert len(set(genomes)) == len(genomes) == int(POP * 0.25)
        np.testing.assert_allclose(after[moved] - before[moved], 0.3, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(after_state.randkey), np.asarray(state.randkey))
        for ours, theirs in zip(jax.tree_util.tree_leaves(after_state.pop_conns),
                                jax.tree_util.tree_leaves(state.pop_conns)):
            assert np.all(_equal_or_both_nan(np.asarray(ours), np.asarray(theirs)))

    def test_negative_control_hmrs_function(self):
        """HMR's function on the same state moves column 0 (the node keys) of every node of the
        chosen genomes and splits the run's key."""
        _, state = _after_one_generation(self.LIVE)
        # rate, strength, decay, generation, old and new palette sizes
        settings = (0.25, 0.3, 0.0, 1, 4, 5)
        after, their_key = hmr_value('live_palette_strategy.hmr_perturbation',
                                     lambda: live('hmr_perturbation', state, *settings),
                                     inputs=(state.pop_nodes, state.randkey, settings))
        before = np.asarray(state.pop_nodes)
        keys_moved = ~_equal_or_both_nan(after[:, :, 0], before[:, :, 0])
        assert keys_moved.any()
        assert np.all(_equal_or_both_nan(after[:, :, 1], before[:, :, 1]))   # no bias moved
        assert not np.array_equal(their_key, np.asarray(state.randkey))

    def test_decay(self):
        algorithm, state = _after_one_generation({'enabled': True, 'perturbation_rate': 0.5,
                                                  'perturbation_decay': 1.0})
        algorithm._live_generation = 50          # rate 0.5 * (1 - 1.0 * 50 / 100)
        after = algorithm._perturb_activation_outputs(state, 4, 5)
        moved = ~_equal_or_both_nan(np.asarray(after.pop_nodes), np.asarray(state.pop_nodes))
        assert len(set(np.nonzero(moved)[0])) == int(POP * 0.25)
        algorithm._live_generation = 100         # rate 0: nothing moves
        assert algorithm._perturb_activation_outputs(state, 4, 5) is state


class TestLiveStrategy:
    def test_an_unchanging_strategy_changes_nothing(self):
        """The step reads the palette and tells the strategy each generation; a palette that does
        not change leaves the run as it is without the strategy, bit for bit."""
        strategy = _Strategy({0: PALETTE})
        _, _, with_strategy, timings = _run(_config({'enabled': True}), 3, strategy)
        _, _, without, _ = _run(_config(), 3)
        for ours, theirs in zip(with_strategy, without):
            np.testing.assert_array_equal(ours, theirs)
        assert [generation for generation, _, _ in strategy.told] == [1, 2, 3]
        best = [float(np.max(fitness)) for fitness in with_strategy]
        assert [fitness for _, fitness, _ in strategy.told] == best
        assert strategy.told[-1][2] == best
        assert all('step6b_palette_update' in timing for timing in timings)

    def test_a_grown_palette(self, monkeypatch):
        calls = []
        perturb = EMRHyperNEAT._perturb_activation_outputs

        def recording(self, state, old, new):
            calls.append((self._live_generation, old, new))
            return perturb(self, state, old, new)

        monkeypatch.setattr(EMRHyperNEAT, '_perturb_activation_outputs', recording)
        algorithm, _, _, _ = _run(
            _config({'enabled': True}), 3, _Strategy({0: PALETTE, 2: PALETTE + [4]})
        )
        assert calls == [(2, 4, 5)]
        assert np.asarray(algorithm.df_palette).tolist() == PALETTE + [4]

    def test_a_shrunk_palette_is_not_perturbed(self, monkeypatch):
        monkeypatch.setattr(EMRHyperNEAT, '_perturb_activation_outputs',
                            lambda *args: pytest.fail('perturbed on a shrink'))
        algorithm, _, _, _ = _run(
            _config({'enabled': True}), 2, _Strategy({0: PALETTE, 1: [0, 1, 2]})
        )
        assert np.asarray(algorithm.df_palette).tolist() == [0, 1, 2]

    def test_enabled_without_a_strategy(self):
        with pytest.raises(ValueError, match='no strategy is set'):
            _run(_config({'enabled': True}), 1)

    def test_the_call_alone_enables_hmrs_defaults(self):
        algorithm, _, _, _ = _run(_config(), 1, _Strategy({0: PALETTE}))
        assert algorithm._live_strategy == {'perturbation_rate': 0.15, 'perturbation_strength': 0.3,
                                            'perturbation_decay': 0.0}

    def test_a_palette_outside_the_list(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(_config({'enabled': True}))
        with pytest.raises(ValueError, match='df_palette must hold indices'):
            algorithm.set_palette_strategy(_Strategy({0: [0, 99]}))

    def test_an_object_without_a_palette(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(_config({'enabled': True}))
        with pytest.raises(TypeError, match='get_active_palette'):
            algorithm.set_palette_strategy(object())

    def test_initialize_runs_when_it_takes_no_argument(self):
        class NoArgument(_Strategy):
            def initialize(self):
                self.initialized = True

        class WithArguments(_Strategy):
            def initialize(self, config, seed):
                raise AssertionError('called')

        algorithm = EMRHyperNEAT()
        algorithm.create_config(_config({'enabled': True}))
        strategy = NoArgument({0: PALETTE})
        algorithm.set_palette_strategy(strategy)
        assert strategy.initialized
        algorithm.set_palette_strategy(WithArguments({0: PALETTE}))


class TestRefusals:
    @pytest.mark.parametrize('block, match', [
        ({'enabled': True, 'perturbation_rate': 1.5}, 'perturbation_rate must be'),
        ({'enabled': True, 'perturbation_strength': float('nan')}, 'perturbation_strength must be'),
        ({'enabled': True, 'perturbation_decay': -0.1}, 'perturbation_decay must be'),
        ({'enabled': 'yes'}, 'enabled must be a bool'),
        ({'enabled': False, 'perturbation_rate': 0.2}, 'would be ignored'),
        ({'enabled': True, 'perturbation_size': 0.2}, 'perturbation_size'),
    ])
    def test_the_block(self, block, match):
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(_config(block))

    def test_a_disabled_block_alone_is_accepted(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(_config({'enabled': False}))
        assert algorithm._live_strategy is None

    def test_without_the_per_node_decode(self):
        config = create_base_config(population_size=POP)
        config['algorithm_params']['emrhyperneat']['emr_hyperneat']['dynamic_functions'] = {
            'mode': 'global', 'hidden_activation': 'tanh', 'live_strategy': {'enabled': True}}
        with pytest.raises(ValueError, match="needs dynamic_functions.mode 'cppn_output'"):
            EMRHyperNEAT().create_config(config)
        algorithm = EMRHyperNEAT()
        algorithm.create_config(create_base_config(population_size=POP))
        with pytest.raises(ValueError, match="needs dynamic_functions.mode 'cppn_output'"):
            algorithm.set_palette_strategy(_Strategy({0: PALETTE}))

    def test_with_palette_evolution(self):
        config = create_config_with_dynamic_functions(
            'cppn_output_4', max_depth=DEPTH, population_size=POP,
            palette_evolution={'enabled': True}, live_strategy={'enabled': True})
        with pytest.raises(ValueError, match='both set the palette'):
            EMRHyperNEAT().create_config(config)
