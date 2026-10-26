"""EMR-HyperNEAT (Eager Multi-Resolution HyperNEAT).

A GPU-optimized variant of ES-HyperNEAT. It replaces the sequential quadtree of
ES-HyperNEAT with eager tensor evaluation on a pre-computed hierarchical grid plus
variance-based masking, turning substrate discovery into batch matrix arithmetic
that vmaps across an entire population.

Public API
----------
    from emr_hyperneat import EMRHyperNEAT

    # The substrate on its own (see emr_hyperneat.substrate):
    from emr_hyperneat import build_substrate, substrate_forward, substrate_queries
    # and its hidden-to-hidden connections, as per-genome discovery builds them:
    from emr_hyperneat import hh_query_rows, build_recurrent_substrate

Published features (all on the main class): base EMR substrate discovery,
per-node dynamic activation functions, neuromodulation (incl. multi-task), and
hidden-to-hidden recurrence caching. Per-node aggregation functions (global,
CPPN-assigned or weight-interpreted) are a standalone control; their published use is
the bio-inspired palette strategies.
"""
from ._version import __version__
from .emrhyperneat import EMRHyperNEAT
from .substrate import (
    HH_CONNECTION_CLASSES,
    SPARSE_CAP_POLICIES,
    SparseCapExceeded,
    SparseCapWarning,
    Substrate,
    SubstrateConfig,
    SubstrateQueries,
    build_recurrent_substrate,
    build_substrate,
    hh_connection_classes,
    hh_pair_weights,
    hh_query_rows,
    query_recurrent_substrate,
    query_substrate,
    substrate_forward,
    substrate_queries,
)

__all__ = [
    "EMRHyperNEAT",
    "HH_CONNECTION_CLASSES",
    "SPARSE_CAP_POLICIES",
    "SparseCapExceeded",
    "SparseCapWarning",
    "Substrate",
    "SubstrateConfig",
    "SubstrateQueries",
    "__version__",
    "build_recurrent_substrate",
    "build_substrate",
    "hh_connection_classes",
    "hh_pair_weights",
    "hh_query_rows",
    "query_recurrent_substrate",
    "query_substrate",
    "substrate_forward",
    "substrate_queries",
]
