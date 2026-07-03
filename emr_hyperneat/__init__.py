"""EMR-HyperNEAT (Eager Multi-Resolution HyperNEAT).

A GPU-optimized variant of ES-HyperNEAT. It replaces the sequential quadtree of
ES-HyperNEAT with eager tensor evaluation on a pre-computed hierarchical grid plus
variance-based masking, turning substrate discovery into batch matrix arithmetic
that vmaps across an entire population.

Public API
----------
    from emr_hyperneat import EMRHyperNEAT

Published features (all on the main class): base EMR substrate discovery,
per-node dynamic activation functions, neuromodulation (incl. multi-task), and
hidden-to-hidden recurrence caching. Per-node aggregation functions are present
but only partial in the current version (exercised via the bio-inspired palette
strategies, not yet as a standalone control).
"""
from .emrhyperneat import EMRHyperNEAT

__all__ = ["EMRHyperNEAT"]
