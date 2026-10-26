"""The island model, multi_gpu_strategy ISLAND_MODEL: islands of the population evolve apart,
generation by generation in lockstep, and exchange their best genomes every migration_interval
generations.

Island i is SINGLE_GPU's run at its own population and at island_seed(seed, i): the run's seed for
island 0, a seed drawn from (seed, i) by numpy's SeedSequence for the others, never HMR's seed + i,
which made island 1 of seed 42 island 0 of seed 43. So each island must equal that SINGLE_GPU run
bit for bit: every generation's fitness vector, the best-fitness history and the final state, key
included. Placement (emr_hyperneat.islands.devices, or island i on device i mod the visible ones)
must change nothing but where the arrays live. The configuration is checked at create_config, each
refusal naming its key.

Migration: after generation M (a multiple of migration_interval) each island's best `migrants`
genomes of its own evaluation go, with the fitness they earned, to its receivers (ring, star or
random), and replace the worst genomes of each receiver's tell at M + 1, their hidden node keys
moved above the receiver's largest. So until the tell at M + 1 every island runs as without
migration, the tell at M + 1 holds exactly those immigrants in exactly those slots, and every
island's random key is the one it has without migration (TensorNEAT's key splits ignore the
population, and no target is drawn from an island's key).

The multi-device cases run in a subprocess with forced CPU devices, as test_multi_device_parity.py
does, and are marked slow.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

import jax
import numpy as np
import pytest

from conftest import (MultiGPUStrategy, EMRHyperNEAT, XORProblem, create_base_config,
                      create_config_with_recurrence)
from emr_hyperneat import (
    emrhyperneat as engine,
)
from test_multi_device_parity import PYTHON, _host_devices_env

GENERATIONS = 5


def island_config(populations, islands=None, recurrent=False, **emr):
    """An ISLAND_MODEL configuration of len(populations) islands (depth 3, genome_elitism 2: at the
    default 15 an island of 20 genomes keeps most of them unchanged)."""
    total = sum(populations)
    if recurrent:
        config = create_config_with_recurrence(
            'hidden_only',
            max_depth=3,
            population_size=total,
            iteration_level=2,
            hh_cache_enabled=False,
        )
    else:
        config = create_base_config(max_depth=3, population_size=total)
    config['algorithm_params']['emrhyperneat']['neat_selection'] = {'genome_elitism': 2}
    block = {'count': len(populations), 'populations': list(populations), 'migration_interval': 0}
    block.update(islands or {})
    config['algorithm_params']['emrhyperneat']['emr_hyperneat'].update(
        multi_gpu_strategy='ISLAND_MODEL', islands=block, **emr)
    return config


def digest(tree):
    sha = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        sha.update(repr((array.dtype.str, array.shape)).encode())
        sha.update(np.ascontiguousarray(array).tobytes())
    return sha.hexdigest()


@pytest.fixture
def told(monkeypatch):
    """Every generation's fitness vector, per engine (in order of first generation), as float32
    bits."""
    records = {}
    real = engine.EMRHyperNEAT.run_generation_verbose

    def recording(self, state, problem, *args, **kwargs):
        new_state, metrics = real(self, state, problem, *args, **kwargs)
        records.setdefault(id(self), []).append(
            np.asarray(self._last_fitnesses, dtype=np.float32).view(np.uint32).tolist())
        return new_state, metrics

    monkeypatch.setattr(engine.EMRHyperNEAT, 'run_generation_verbose', recording)
    return records


def single_gpu_run(config_of_population, population, seed, told):
    """SINGLE_GPU's run at (population, seed): its fitness vectors, history and final state's
    digest."""
    told.clear()
    algorithm = EMRHyperNEAT()
    state = algorithm.initialize(
        algorithm.create_config(config_of_population(population)), XORProblem(), seed=seed
    )
    result = algorithm.run_until_threshold(
        state, XORProblem(), target_fitness=2.0, max_generations=GENERATIONS, collect_history=True
    )
    return {'fitness': list(told.values())[0], 'history': [float(x) for x in result['history']],
            'state': digest(result['state'])}


class TestIslandSeed:
    def test_island_zero_takes_the_runs_seed(self):
        assert [engine.island_seed(seed, 0) for seed in (0, 7, 42, 2**31 - 1)] == [
            0,
            7,
            42,
            2**31 - 1,
        ]

    @staticmethod
    def seeds_shared(seed_of):
        """Islands 1 to 7 of runs 0 to 999 whose seed is another run's island-0 seed, or another
        island's seed."""
        derived = [seed_of(seed, island) for seed in range(1000) for island in range(1, 8)]
        return len(set(derived) & set(range(1000))) + len(derived) - len(set(derived))

    def test_no_island_shares_a_seed(self):
        assert self.seeds_shared(engine.island_seed) == 0

    def test_negative_control_seed_plus_island(self):
        """HMR's seed + i: island 1 of seed 42 is island 0 of seed 43."""
        assert self.seeds_shared(lambda seed, island: seed + island) > 0

    def test_seeds_are_positive_32_bit_and_fixed(self):
        assert engine.island_seed(42, 1) == 1181570228
        assert all(
            0 <= engine.island_seed(seed, island) < 2**31
            for seed in range(50)
            for island in range(8)
        )


class TestConfiguration:
    """emr_hyperneat.islands at create_config: each refusal names its key."""

    @staticmethod
    def refused(config, match):
        with pytest.raises(ValueError, match=match):
            EMRHyperNEAT().create_config(config)

    def test_names(self):
        for name in ('ISLAND_MODEL', 'ISLAND_MODEL_V2'):
            assert engine.STRATEGY_NAME_MAP[name] == MultiGPUStrategy.ISLAND_MODEL
            assert (
                EMRHyperNEAT(strategy=MultiGPUStrategy[name]).strategy
                == MultiGPUStrategy.ISLAND_MODEL
            )

    def test_accepted(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(island_config([16, 24, 20]))
        assert algorithm._island_plan == engine.IslandPlan(3, (16, 24, 20), None, 0, 0, 'ring')
        config = island_config([20, 20])
        del config['algorithm_params']['emrhyperneat']['emr_hyperneat']['islands']['populations']
        algorithm.create_config(config)
        assert algorithm._island_plan.populations == (20, 20)

    def test_the_block_and_the_strategy_go_together(self):
        config = island_config([20, 20])
        config['algorithm_params']['emrhyperneat']['emr_hyperneat'][
            'multi_gpu_strategy'
        ] = 'SINGLE_GPU'
        self.refused(config, 'multi_gpu_strategy SINGLE_GPU reads none of it')
        config = island_config([20, 20])
        del config['algorithm_params']['emrhyperneat']['emr_hyperneat']['islands']
        self.refused(config, 'ISLAND_MODEL needs emr_hyperneat.islands')

    @pytest.mark.parametrize(
        'change, match',
        [
            ({'count': 1}, 'islands.count must be an integer of at least 2'),
            ({'count': True}, 'islands.count must be an integer'),
            (
                {'migration_interval': -1},
                'islands.migration_interval must be an integer of at least 0',
            ),
            ({'size': 3}, r"unknown keys \['size'\]"),
            ({'populations': [20, 21]}, 'islands.populations sum to 41, not population_size 40'),
            ({'populations': [40]}, 'islands.populations must list 2 island populations'),
            (
                {'populations': [38, 2]},
                r'islands: islands \[1\] of \[38, 2\] genomes hold no more than '
                r'neat_selection.genome_elitism 2',
            ),
            ({'devices': [0]}, 'islands.devices must list 2 device indices'),
            ({'devices': [0, 64]}, r'islands.devices \[0, 64\] names a device beyond'),
            ({'migrants': 2}, 'islands.migrants is read only with migration'),
            ({'topology': 'star'}, 'islands.topology is read only with migration'),
            ({'migration_interval': 5}, 'islands.migrants is required with migration'),
            (
                {'migration_interval': 5, 'migrants': 1, 'topology': 'mesh'},
                'islands.topology must be one of',
            ),
            (
                {'migration_interval': 5, 'migrants': 19},
                r'19 immigrants per migration would replace part of the best '
                r'genome_elitism 2 genomes of islands \[0, '
                r'1\]',
            ),
        ],
    )
    def test_refusals(self, change, match):
        self.refused(island_config([20, 20], islands=change), match)

    def test_migration_settings(self):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(
            island_config(
                [16, 24, 20], islands={'migration_interval': 3, 'migrants': 2, 'topology': 'star'}
            )
        )
        assert algorithm._island_plan == engine.IslandPlan(3, (16, 24, 20), None, 3, 2, 'star')
        algorithm.create_config(
            island_config([20, 20], islands={'migration_interval': 1, 'migrants': 1})
        )
        assert algorithm._island_plan.topology == 'ring'

    @pytest.mark.parametrize(
        'recurrence, dynamic, match',
        [
            ({'hh_cache_enabled': True}, None, 'recurrence.hh_cache_enabled'),
            (
                None,
                {
                    'mode': 'cppn_output',
                    'num_activations': 4,
                    'palette_evolution': {'enabled': True},
                },
                'dynamic_functions.palette_evolution',
            ),
        ],
    )
    def test_what_migration_cannot_carry(self, recurrence, dynamic, match):
        config = island_config(
            [20, 20], islands={'migration_interval': 2, 'migrants': 1}, recurrent=True
        )
        emr = config['algorithm_params']['emrhyperneat']['emr_hyperneat']
        if recurrence:
            emr['recurrence'].update(recurrence)
        if dynamic:
            emr['dynamic_functions'] = dynamic
        self.refused(config, f'islands with migration cannot apply {match}')

    def test_required_keys(self):
        for key in ('count', 'migration_interval'):
            config = island_config([20, 20])
            del config['algorithm_params']['emrhyperneat']['emr_hyperneat']['islands'][key]
            self.refused(config, f'islands.{key} is required')

    def test_an_unequal_default_split_prints_one_to_paste(self):
        config = island_config([38, 38, 37, 37])
        del config['algorithm_params']['emrhyperneat']['emr_hyperneat']['islands']['populations']
        config['algorithm_params']['emrhyperneat']['population_size'] = 150
        self.refused(config, r'population_size 150 does not split into 4 equal islands: set '
                             r'emr_hyperneat.islands.populations, for example \[38, 38, 37, 37\]')

    def test_the_default_elitism_freezes_small_islands(self):
        config = island_config([15, 15])
        del config['algorithm_params']['emrhyperneat']['neat_selection']
        self.refused(config, r'islands \[0, 1\] of \[15, 15\] genomes hold no more than '
                             r'neat_selection.genome_elitism 15')

    def test_hmr_island_config_object_stays_refused(self):
        with pytest.raises(ValueError, match='island_config configure base-class strategies'):
            EMRHyperNEAT(strategy=MultiGPUStrategy.ISLAND_MODEL, island_config=object())


class TestRoute:
    """On one device: each island is SINGLE_GPU's run at its seed and population, bit for bit."""

    @pytest.mark.parametrize('populations, recurrent', [([20, 20], False), ([16, 24, 20], True)])
    def test_islands_are_single_gpu_runs(self, told, populations, recurrent):
        config = island_config(populations, recurrent=recurrent)
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=42)
        told.clear()
        result = algorithm.run_until_threshold(
            state,
            XORProblem(),
            target_fitness=2.0,
            max_generations=GENERATIONS,
            collect_history=True,
        )
        runs = list(told.values())
        # Island i on device i mod the visible devices (no islands.devices here): the route counts
        # the devices that host islands (2 on two GPUs, where this test once expected the one CPU
        # device it was written on)
        hosts = len({i % len(jax.devices()) for i in range(len(populations))})
        assert (result['route'], result['devices'], result['generations']) == (
            'ISLAND_MODEL',
            hosts,
            GENERATIONS,
        )
        assert [island['device'] for island in result['islands']] == [
            i % len(jax.devices()) for i in range(len(populations))]
        assert len(runs) == len(populations) and all(len(run) == GENERATIONS for run in runs)

        def config_of(population):
            single = island_config([population], recurrent=recurrent)
            emr = single['algorithm_params']['emrhyperneat']['emr_hyperneat']
            del emr['islands']
            emr['multi_gpu_strategy'] = 'SINGLE_GPU'
            return single

        for index, (island, run) in enumerate(zip(result['islands'], runs)):
            assert (island['island'], island['seed'], island['population']) == (
                index, engine.island_seed(42, index), populations[index])
            reference = single_gpu_run(config_of, populations[index], island['seed'], told)
            assert (
                run == reference['fitness']
            ), f'island {index}: another fitness vector than SINGLE_GPU'
            assert island['history'] == reference['history']
            assert digest(island['state']) == reference['state']
        # The negative control: island 1 against SINGLE_GPU at HMR's seed 43
        hmr = single_gpu_run(config_of, populations[1], 43, told)
        assert runs[1] != hmr['fitness'] and digest(result['islands'][1]['state']) != hmr['state']

        # The records: the best over islands and generations, the best last-generation champion's
        # island
        assert result['best_fitness'] == max(island['best_fitness'] for island in result['islands'])
        assert result['history'] == [max(island['history'][g] for island in result['islands'])
                                     for g in range(GENERATIONS)]
        last = [island['history'][-1] for island in result['islands']]
        assert result['champion_island'] == int(np.argmax(last))
        assert result['state'] is result['islands'][result['champion_island']]['state']
        nodes, conns, fitness, index = algorithm.get_best_genome_with_fitness()
        assert fitness == last[result['champion_island']]
        scored = algorithm.evaluate_on_data(
            result['state'], algorithm._cached_inputs, algorithm._cached_targets
        )
        assert scored['fitness'] == pytest.approx(fitness, abs=0.0)

    def test_stops_when_an_island_reaches_the_target(self):
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(island_config([20, 20])), XORProblem(), seed=42
        )
        result = algorithm.run_until_threshold(
            state,
            XORProblem(),
            target_fitness=0.75,
            max_generations=GENERATIONS,
            collect_history=True,
        )
        # Island 0 of seed 42 passes 0.75 at generation 2 (histories in
        # test_islands_are_single_gpu_runs)
        assert result['generations'] < GENERATIONS and result['best_fitness'] >= 0.75
        assert all(len(island['history']) == result['generations'] for island in result['islands'])

    def test_zero_generations(self):
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(island_config([20, 20])), XORProblem(), seed=42
        )
        result = algorithm.run_until_threshold(
            state, XORProblem(), target_fitness=2.0, max_generations=0
        )
        assert (result['route'], result['generations']) == ('ISLAND_MODEL', 0)

    @pytest.mark.parametrize(
        'emr, match',
        [
            ({'execution': {'mode': 'depth_offload'}}, "runs the depth_offload generation"),
            ({'execution': {'mode': 'streaming'}}, "runs the streaming generation"),
            (
                {'execution': {'generation_loop': 'while_loop'}},
                "generation_loop 'while_loop' compiles one run",
            ),
        ],
    )
    def test_refusals_at_run_time(self, emr, match):
        algorithm = EMRHyperNEAT()
        state = algorithm.initialize(
            algorithm.create_config(island_config([20, 20], **emr)), XORProblem(), seed=42
        )
        with pytest.raises(ValueError, match=match):
            algorithm.run_until_threshold(
                state, XORProblem(), target_fitness=2.0, max_generations=1
            )


PLACEMENT_SCRIPT = r'''
import json
import sys
import warnings
sys.path.insert(0, {tests!r})
import jax
from conftest import EMRHyperNEAT, XORProblem
from test_islands import GENERATIONS, digest, island_config
assert len(jax.devices()) == {devices}, jax.devices()
warnings.simplefilter('ignore')
populations = [16, 24, 20, 20][:max(3, {devices})]
found = {{}}
for label, placement in (
    ('first_device', [0] * len(populations)),
    ('default', None),
    ('reversed', [({devices} - 1 - i) % {devices} for i in range(len(populations))]),
):
    # Migration on: star, every second generation
    islands = {{'migration_interval': 2, 'migrants': 1, 'topology': 'star'}}
    if placement is not None:
        islands['devices'] = placement
    algorithm = EMRHyperNEAT()
    config = island_config(populations, islands=islands, recurrent=True)
    state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=42)
    result = algorithm.run_until_threshold(
        state, XORProblem(), target_fitness=2.0, max_generations=GENERATIONS, collect_history=True
    )
    found[label] = {{
        'route': [result['route'], result['devices'], algorithm.get_num_devices()],
        'islands': [
            [
                island['device'],
                digest(island['state']),
                island['history'],
                sorted(
                    {{
                        d.id
                        for leaf in jax.tree_util.tree_leaves(island['state'])
                        for d in leaf.devices()
                    }}
                ),
            ]
            for island in result['islands']
        ],
    }}
    jax.clear_caches()
print('PLACEMENT', json.dumps(found))
'''


@pytest.fixture(scope='module', params=[2, 3, 4], ids=['2_devices', '3_devices', '4_devices'])
def placement_run(request):
    tests = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        PYTHON + ['-c', PLACEMENT_SCRIPT.format(tests=tests, devices=request.param)],
        env=_host_devices_env(request.param),
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=tests,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith('PLACEMENT ')][-1]
    return request.param, json.loads(line[len('PLACEMENT '):])


@pytest.mark.slow
def test_placement_changes_nothing_but_the_devices(placement_run):
    """On 2, 3 and 4 forced CPU devices, hidden_only at level 2, 3 or 4 islands migrating (star,
    every second generation): every island on the first device, the default placement (island i on
    device i mod n) and a reversed one give the same islands bit for bit (final state and history),
    and each island's arrays live on its device; the route records the devices that host islands."""
    devices, found = placement_run
    reference = [island[1:3] for island in found['first_device']['islands']]
    for label, run in found.items():
        assert [
            island[1:3] for island in run['islands']
        ] == reference, f'{devices} devices, {label}'
        assert all(
            island[3] == [island[0]] for island in run['islands']
        ), f'{label}: arrays off their devices'
        hosts = len({island[0] for island in run['islands']})
        assert run['route'] == ['ISLAND_MODEL', hosts, hosts], label
    assert [island[0] for island in found['default']['islands']] == [
        i % devices for i in range(len(found['default']['islands']))]


class TestMigrationTargets:
    @staticmethod
    def plan(count, topology):
        return engine.IslandPlan(count, (20,) * count, None, 2, 1, topology)

    def test_ring_and_star(self):
        assert engine.migration_targets(self.plan(3, 'ring'), 42, 0) == [[1], [2], [0]]
        assert engine.migration_targets(self.plan(3, 'star'), 42, 5) == [[1, 2], [0, 2], [0, 1]]

    def test_random_is_one_other_island_drawn_from_the_seed_and_the_event(self):
        plan = self.plan(4, 'random')
        draws = [engine.migration_targets(plan, 42, event) for event in range(20)]
        assert all(
            len(targets) == 1 and targets[0] != i
            for draw in draws
            for i, targets in enumerate(draw)
        )
        assert draws == [engine.migration_targets(plan, 42, event) for event in range(20)]
        assert len({json.dumps(draw) for draw in draws}) > 1
        assert draws != [engine.migration_targets(plan, 43, event) for event in range(20)]


class TestRemapHiddenKeys:
    """Two migrants of one sender: inputs and outputs keep their keys; hidden nodes move to 100 and
    up, one map for both, so the node they share stays shared; every connection follows its nodes.
    """

    IO = np.asarray([0.0, 1.0, 2.0])
    NAN = np.nan

    def genes(self):
        nodes = np.asarray([
            [[0, .1], [1, .2], [2, .3], [7, .4], [9, .5], [self.NAN, self.NAN]],
            [[0, .1], [1, .2], [2, .3], [9, .6], [12, .7], [self.NAN, self.NAN]],
        ])
        conns = np.asarray([
            [[0, 7, .5], [7, 9, .6], [9, 2, .7], [self.NAN, self.NAN, self.NAN]],
            [[1, 9, .5], [9, 12, .6], [12, 2, .7], [0, 2, .8]],
        ])
        return nodes, conns

    @staticmethod
    def consistent(nodes, conns):
        for genome_nodes, genome_conns in zip(nodes, conns):
            keys = set(genome_nodes[~np.isnan(genome_nodes[:, 0]), 0])
            ends = genome_conns[~np.isnan(genome_conns[:, 0]), :2]
            if not set(ends.ravel()) <= keys:
                return False
        return True

    def test_the_hidden_keys_move_together(self):
        nodes, conns = self.genes()
        moved_nodes, moved_conns, next_key = engine.remap_hidden_keys(nodes, conns, self.IO, 100)
        assert next_key == 103
        assert moved_nodes[0, :, 0].tolist()[:5] == [0, 1, 2, 100, 101]
        assert moved_nodes[1, :, 0].tolist()[:5] == [0, 1, 2, 101, 102]
        assert np.array_equal(moved_nodes[..., 1:], nodes[..., 1:], equal_nan=True)
        assert np.array_equal(moved_conns[..., 2:], conns[..., 2:], equal_nan=True)
        assert self.consistent(moved_nodes, moved_conns)
        assert np.isnan(moved_nodes[:, 5]).all() and np.isnan(moved_conns[0, 3]).all()

    def test_negative_control_a_skipped_node_leaves_a_dangling_connection(self):
        nodes, conns = self.genes()
        moved_nodes, moved_conns, _ = engine.remap_hidden_keys(nodes, conns, self.IO, 100)
        moved_nodes[0, 3, 0] = 7.0  # node 7's move undone in the nodes only
        assert not self.consistent(moved_nodes, moved_conns)


def migration_run(populations, islands, recurrent=False):
    """An island run of GENERATIONS generations: per island, every generation's own fitness
    (float32 bits) and what its tell received (state and fitness, numpy), and the result."""
    told = {}
    real = engine.EMRHyperNEAT._before_tell

    def recording(self, state, fitnesses):
        state_told, fitness_told = real(self, state, fitnesses)
        told.setdefault(id(self), []).append({
            'own': np.asarray(fitnesses, dtype=np.float32).view(np.uint32).tolist(),
            'own_nodes': np.asarray(state.pop_nodes), 'own_conns': np.asarray(state.pop_conns),
            'nodes': np.asarray(state_told.pop_nodes), 'conns': np.asarray(state_told.pop_conns),
            'fitness': np.asarray(fitness_told)})
        return state_told, fitness_told

    engine.EMRHyperNEAT._before_tell = recording
    try:
        algorithm = EMRHyperNEAT()
        config = island_config(populations, islands=islands, recurrent=recurrent)
        state = algorithm.initialize(algorithm.create_config(config), XORProblem(), seed=42)
        result = algorithm.run_until_threshold(
            state,
            XORProblem(),
            target_fitness=2.0,
            max_generations=GENERATIONS,
            collect_history=True,
        )
    finally:
        engine.EMRHyperNEAT._before_tell = real
    return list(told.values()), result, algorithm


class TestMigration:
    """One device. Each case against the same islands without migration."""

    @pytest.mark.parametrize('populations, islands, recurrent', [
        ([20, 20], {'migration_interval': 2, 'migrants': 2}, False),
        ([40, 16], {'migration_interval': 2, 'migrants': 2}, False),
        ([16, 24, 20], {'migration_interval': 2, 'migrants': 1, 'topology': 'star'}, True),
        ([16, 24, 20], {'migration_interval': 1, 'migrants': 3, 'topology': 'random'}, False),
    ], ids=['ring', 'ring_unequal', 'star_hidden_only', 'random_every_generation'])
    def test_immigrants_replace_the_worst_of_the_next_tell(self, populations, islands, recurrent):
        tells, result, algorithm = migration_run(populations, islands, recurrent)
        quiet, quiet_result, _ = migration_run(populations, {}, recurrent)
        interval, migrants = islands['migration_interval'], islands['migrants']
        plan = algorithm._island_plan
        io_keys = np.asarray(
            list(algorithm.neat_algo.genome.input_idx)
            + list(algorithm.neat_algo.genome.output_idx),
            dtype=np.float64,
        )
        assert [event['generation'] for event in result['migrations']] == list(
            range(interval, GENERATIONS + 1, interval))
        # Until the first delivery every tell is the run without migration's
        for island, (run, reference) in enumerate(zip(tells, quiet)):
            for generation in range(interval + 1):
                assert run[generation]['own'] == reference[generation]['own'], (island, generation)
        # The first delivery: at the tell of generation interval + 1 (index interval)
        event = result['migrations'][0]
        g = interval
        for receiver, received in enumerate(tells):
            senders = [i for i, targets in enumerate(event['targets']) if receiver in targets]
            tell = received[g]
            own = np.asarray(tell['own'], dtype=np.uint32).view(np.float32)
            order = np.argsort(own[::-1], kind='stable')
            slots = (len(own) - 1 - order)[:migrants * len(senders)]
            next_key = int(np.nanmax(tell['own_nodes'][:, :, 0])) + 1
            for k, sender in enumerate(senders):
                sent = tells[sender][g - 1]
                sender_own = np.asarray(sent['own'], dtype=np.uint32).view(np.float32)
                top = np.argsort(-sender_own, kind='stable')[:migrants]
                nodes, conns, next_key = engine.remap_hidden_keys(
                    sent['own_nodes'][top], sent['own_conns'][top], io_keys, next_key
                )
                taken = slots[k * migrants:(k + 1) * migrants]
                assert np.array_equal(tell['nodes'][taken], nodes, equal_nan=True), (
                    receiver,
                    sender,
                )
                assert np.array_equal(tell['conns'][taken], conns, equal_nan=True), (
                    receiver,
                    sender,
                )
                assert np.array_equal(tell['fitness'][taken], sender_own[top]), (receiver, sender)
            kept = np.setdiff1d(np.arange(len(own)), slots)
            assert np.array_equal(tell['nodes'][kept], tell['own_nodes'][kept], equal_nan=True)
            assert np.array_equal(tell['fitness'][kept], own[kept])
        # Every island's random key is the one it has without migration
        assert [np.asarray(island['state'].randkey).tolist() for island in result['islands']] == [
            np.asarray(island['state'].randkey).tolist() for island in quiet_result['islands']]
        # The populations differ after the first delivery (the immigrants entered selection)
        assert any(digest(a['state']) != digest(b['state'])
                   for a, b in zip(result['islands'], quiet_result['islands']))
        # The champion's record matches the returned state
        scored = algorithm.evaluate_on_data(
            result['state'], algorithm._cached_inputs, algorithm._cached_targets
        )
        assert scored['champion_index'] == algorithm.get_best_genome_with_fitness()[3]
        assert plan.migration_interval == interval
