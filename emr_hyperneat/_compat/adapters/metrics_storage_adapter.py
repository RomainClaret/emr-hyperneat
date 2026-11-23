"""In-memory stand-in for the geenns metrics storage adapter.

The original module stores per-generation metrics through pluggable backends (in memory, or
cached and streamed to disk) and pulls in a metrics-cache subtree. The standalone algorithm keeps
every generation's metrics in memory: this module provides the two symbols BaseAlgorithm imports
(`storage_registry`, `get_optimal_adapter`), and the registry gives an in-memory storage whatever
adapter is named, so nothing is written to disk.
"""
from __future__ import annotations

from typing import Any, Iterator, List


class InMemoryMetricsStorage:
    """Every appended generation's metrics, kept as given, in the order they were appended.

    It implements what `BaseAlgorithm.run_experiment` and `metrics_history` use: `append`,
    `clear`, iteration, `len`, a truth value (false when empty) and indexing (negative indices
    included).
    """

    def __init__(self, **_options: Any) -> None:
        # The original's disk-backed adapters take options (memory_size, window_size, cache_dir);
        # none applies to a storage held in memory
        self._history: List[Any] = []

    def append(self, metrics: Any) -> None:
        """Keep one generation's metrics after the others."""
        self._history.append(metrics)

    def clear(self) -> None:
        """Drop every kept generation."""
        self._history.clear()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._history)

    def __len__(self) -> int:
        return len(self._history)

    def __getitem__(self, index: Any) -> Any:
        return self._history[index]


class _Registry:
    """Stand-in for the storage registry; `create(...)` gives an in-memory storage, whatever the
    adapter's name."""

    def create(self, name: Any = None, **options: Any) -> InMemoryMetricsStorage:
        return InMemoryMetricsStorage(**options)

    def __getattr__(self, _name):
        return lambda *a, **k: None

    def __contains__(self, _name):
        return False

    def __iter__(self):
        return iter(())


storage_registry = _Registry()


def get_optimal_adapter(*args: Any, **kwargs: Any) -> str:
    """The original picks an adapter by the run's size; here every run keeps its metrics in
    memory."""
    return 'memory'
