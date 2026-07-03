"""No-op stub of the geenns metrics storage adapter.

The original module persists per-generation metrics to pluggable backends
(memory/cached/streaming) and pulls in a metrics-cache subtree. The standalone
algorithm runs evolution without persistence, so this stub provides the two
symbols BaseAlgorithm imports (`storage_registry`, `get_optimal_adapter`) as
no-ops that never store anything.
"""
from __future__ import annotations


class _NoOpStorage:
    """A metrics-storage handle that silently accepts any method call."""

    def _noop(self, *args, **kwargs):
        return None

    def __getattr__(self, _name):
        return self._noop


class _Registry:
    """Stand-in for the storage registry; `create(...)` yields a no-op storage."""

    def create(self, *args, **kwargs) -> _NoOpStorage:
        return _NoOpStorage()

    def __getattr__(self, _name):
        return lambda *a, **k: None

    def __contains__(self, _name):
        return False

    def __iter__(self):
        return iter(())


storage_registry = _Registry()


def get_optimal_adapter(*args, **kwargs):
    """Original returns the name of the best storage adapter; None disables it."""
    return None
