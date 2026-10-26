"""HMR's base-engine memory and speed settings on this engine.

sparse_forward_threshold, fuse_w1_computation and skip_unused_masks tuned HMR's base-engine
generation: a sparse forward pass over active positions, a fused weight expression, and masks B and
C skipped. The base class still parses them, and nothing reads them. This class computes the
reached-position mask alone and narrows the evaluation by compaction, so every value runs the same
computation: the tests below check it bit for bit on five kinds of run. The values are checked
(0.1.0 read them as given).
"""
import jax
import numpy as np
import pytest

from conftest import (DEFAULT_SEED, EMRHyperNEAT, XORProblem, create_base_config,
                      create_config_with_recurrence)


def _config(kind: str, **settings):
    if kind == 'hidden_only':
        config = create_config_with_recurrence(
            'hidden_only', max_depth=3, population_size=12, hh_cache_enabled=False
        )
    else:
        config = create_base_config(population_size=12)
    # Elitism 2, so that the next state holds mutated genomes: at the default 15 a population of
    # 12 is copied unmutated.
    config['algorithm_params']['emrhyperneat']['neat_selection'] = {'genome_elitism': 2}
    emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
    if kind == 'act_cppn4':
        emr['dynamic_functions'] = {'mode': 'cppn_output', 'num_activations': 4}
    elif kind in ('streaming', 'depth_offload'):
        emr['execution'] = {'mode': kind}
    emr.update(settings)
    return config


def _generation(config):
    """One generation: the best and mean fitness and every array of the next state, as numpy."""
    problem = XORProblem()
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(algorithm.create_config(config), problem, seed=DEFAULT_SEED)
    state, metrics = algorithm.run_generation(state, problem)
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state)]
    return float(metrics.best_fitness), float(metrics.mean_fitness), leaves


KINDS = ['feedforward', 'hidden_only', 'act_cppn4', 'streaming', 'depth_offload']
SETTINGS = [
    {'sparse_forward_threshold': -1},
    {'sparse_forward_threshold': 5},
    {'fuse_w1_computation': True},
    {'skip_unused_masks': True},
    {'sparse_forward_threshold': -1, 'fuse_w1_computation': True, 'skip_unused_masks': True},
]


@pytest.mark.parametrize('kind', KINDS)
def test_every_value_runs_the_same_generation(kind):
    best, mean, leaves = _generation(_config(kind))
    for settings in SETTINGS:
        other_best, other_mean, other_leaves = _generation(_config(kind, **settings))
        assert (other_best, other_mean) == (best, mean), settings
        assert len(other_leaves) == len(leaves)
        for ours, theirs in zip(leaves, other_leaves):
            np.testing.assert_array_equal(theirs, ours, err_msg=str(settings))


@pytest.mark.parametrize('settings', [
    {'sparse_forward_threshold': -2},
    {'sparse_forward_threshold': 1.5},
    {'sparse_forward_threshold': True},
    {'fuse_w1_computation': 'yes'},
    {'skip_unused_masks': 1},
])
def test_wrong_values_raise(settings):
    key = next(iter(settings))
    with pytest.raises(ValueError, match=f'emr_hyperneat.{key} must be'):
        EMRHyperNEAT().create_config(_config('feedforward', **settings))
