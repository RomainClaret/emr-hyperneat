"""No-op stub of the geenns experiment reporter.

The original module streams progress/telemetry events to the research framework's
reporting backend. The standalone algorithm does not need telemetry, so this stub
provides the single symbol BaseAlgorithm imports (`get_global_reporter`) returning
a reporter whose every method is a no-op.
"""
from __future__ import annotations


class _NoOpReporter:
    """A reporter that silently accepts any method call."""

    def _noop(self, *args, **kwargs):
        return None

    def __getattr__(self, _name):
        # Any attribute access (report_progress, report_completion, ...) -> no-op.
        return self._noop


_GLOBAL_REPORTER = _NoOpReporter()


def get_global_reporter(*args, **kwargs) -> _NoOpReporter:
    return _GLOBAL_REPORTER
