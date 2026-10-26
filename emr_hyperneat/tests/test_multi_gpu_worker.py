"""The process route's worker helpers: per-device workers, and shard results combined or refused.

In 0.1.0 every worker set CUDA_VISIBLE_DEVICES itself, after the package import had initialized
JAX, so all of them ran on the first GPU, and a failed shard kept a fitness of 0 that evolution then
selected on. The GPU half of the first check (a worker sees one GPU) needs two GPUs and runs on a
GPU machine; on the CPU these tests check what each worker process starts with.
"""

import multiprocessing as mp
import os

import numpy as np
import pytest

from emr_hyperneat.multi_gpu_worker import (
    EvalShardOutput,
    combine_shard_fitnesses,
    start_device_workers,
)


def _shard(start, end, values, error=None):
    return EvalShardOutput(shard_start=start, shard_end=end,
                           fitnesses=np.asarray(values, dtype=np.float32), error=error)


def _worker_environment(monkeypatch, parent_value):
    """Start two device workers with the parent's CUDA_VISIBLE_DEVICES set to `parent_value`.

    `parent_value` None leaves the variable unset. Returns what each worker process sees, and the
    parent's variable afterwards.
    """
    if parent_value is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', parent_value)
    pools = start_device_workers(2, mp.get_context('spawn'))
    try:
        seen = [(pool.submit(os.getenv, 'CUDA_VISIBLE_DEVICES').result(timeout=120),
                 pool.submit(os.getenv, 'XLA_PYTHON_CLIENT_PREALLOCATE').result(timeout=120))
                for pool in pools]
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    return seen, os.environ.get('CUDA_VISIBLE_DEVICES')


def _worker_numeric_settings():
    """The matmul precision and x64 mode a worker's JAX started with (run inside the worker)."""
    import jax
    return jax.config.jax_default_matmul_precision, bool(jax.config.jax_enable_x64)


class TestStartDeviceWorkers:
    """Each worker process starts with its own device in CUDA_VISIBLE_DEVICES, not preallocating."""

    def test_each_worker_starts_with_its_device(self, monkeypatch):
        """No parent setting: worker d sees device d; the parent's environment is left as it was."""
        monkeypatch.delenv('XLA_PYTHON_CLIENT_PREALLOCATE', raising=False)
        monkeypatch.delenv('XLA_PYTHON_CLIENT_MEM_FRACTION', raising=False)
        monkeypatch.delenv('XLA_PYTHON_CLIENT_ALLOCATOR', raising=False)
        seen, after = _worker_environment(monkeypatch, None)
        assert seen == [('0', 'false'), ('1', 'false')]
        assert after is None and 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ

    def test_workers_take_the_parents_device_list_in_order(self, monkeypatch):
        """Under Slurm the parent's variable lists its GPUs; worker d gets the d-th of them."""
        monkeypatch.delenv('XLA_PYTHON_CLIENT_PREALLOCATE', raising=False)
        monkeypatch.delenv('XLA_PYTHON_CLIENT_MEM_FRACTION', raising=False)
        monkeypatch.delenv('XLA_PYTHON_CLIENT_ALLOCATOR', raising=False)
        seen, after = _worker_environment(monkeypatch, '5,7')
        assert [device for device, _ in seen] == ['5', '7']
        assert after == '5,7'

    def test_workers_take_the_parents_numeric_settings(self, monkeypatch):
        """A matmul precision and an x64 mode the parent set in process reach every worker, and the
        parent's environment is left as it was. 0.1.0's workers ran without them."""
        import jax
        monkeypatch.delenv('JAX_DEFAULT_MATMUL_PRECISION', raising=False)
        monkeypatch.delenv('JAX_ENABLE_X64', raising=False)
        before = (jax.config.jax_default_matmul_precision, jax.config.jax_enable_x64)
        jax.config.update('jax_default_matmul_precision', 'highest')
        jax.config.update('jax_enable_x64', True)
        try:
            # One worker per visible GPU, at most two: a worker pointed at a GPU the host lacks
            # cannot start its JAX (on one L4: "No visible GPU devices"). Two on the CPU
            workers = min(2, len(jax.devices())) if jax.default_backend() == 'gpu' else 2
            pools = start_device_workers(workers, mp.get_context('spawn'))
            try:
                seen = [pool.submit(_worker_numeric_settings).result(timeout=120)
                        for pool in pools]
            finally:
                for pool in pools:
                    pool.shutdown(wait=True)
        finally:
            jax.config.update('jax_default_matmul_precision', before[0])
            jax.config.update('jax_enable_x64', before[1])
        assert seen == [('highest', True)] * workers
        assert 'JAX_DEFAULT_MATMUL_PRECISION' not in os.environ
        assert 'JAX_ENABLE_X64' not in os.environ

    def test_a_users_memory_setting_is_inherited(self, monkeypatch):
        """With an XLA_PYTHON_CLIENT_* memory variable set, the workers keep the user's settings."""
        monkeypatch.setenv('XLA_PYTHON_CLIENT_MEM_FRACTION', '.4')
        monkeypatch.setenv('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
        seen, _ = _worker_environment(monkeypatch, None)
        assert [prealloc for _, prealloc in seen] == ['true', 'true']


class TestCombineShardFitnesses:
    """Shard results land in population order; a failed shard raises, not scoring its genomes 0."""

    def test_shards_in_population_order(self):
        """The last shard may be larger (it takes the remainder of the population)."""
        results = [_shard(3, 8, [0.5, 0.6, 0.7, 0.8, 0.9]), _shard(0, 3, [0.1, 0.2, 0.3])]
        np.testing.assert_allclose(combine_shard_fitnesses(results, 8),
                                   [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])

    def test_a_failed_shard_raises(self):
        """A worker's error reaches the caller with the genomes it covered."""
        results = [_shard(0, 4, [0.4] * 4), _shard(4, 8, [0.0] * 4, error='ValueError: shape')]
        with pytest.raises(RuntimeError, match='genomes 4-8 failed: ValueError: shape'):
            combine_shard_fitnesses(results, 8)
