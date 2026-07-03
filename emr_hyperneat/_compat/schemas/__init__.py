"""
Pydantic schemas for experiment configuration.

This module provides type-safe, validated configuration schemas that serve as
the single source of truth for all experiment parameters. Schemas replace
manual parameter dictionaries and DEFAULT_VALUES registries.

Key benefits:
- Single source of truth (add parameter once)
- Type safety (catches errors at config time)
- Auto-validation (ranges, types, dependencies)
- Auto-documentation (from field descriptions)
- Works for TensorNEAT + PUREPLES + Web UI

Usage:
    from emr_hyperneat._compat.schemas import NEATConfig

    config = NEATConfig(population_size=150, mutation__conn_add_prob=0.3)
    config.model_dump()  # Dict for builders
    config.model_json_schema()  # JSON schema for Web UI
"""

from .base import (
    ActivationConfig,
    AggregationConfig,
    BiasConfig,
    WeightConfig,
    ResponseConfig,
)
from .neat_schema import (
    NEATConfig,
    MutationConfig,
    GenomeConfig,
    SpeciesConfig,
    SelectionConfig,
)
from .hyperneat_schema import (
    HyperNEATConfig,
    CPPNConfig,
    CPPNGenomeConfig,
    CPPNMutationConfig,
    CPPNActivationConfig,
    SubstrateConfig,
)
from .experiment_schema import (
    ExperimentConfig,
    PerformanceConfig,
    MetricsConfig,
    CheckpointConfig,
    InfrastructureConfig,
    OutputConfig,
)

__all__ = [
    # Base configs
    'ActivationConfig',
    'AggregationConfig',
    'BiasConfig',
    'WeightConfig',
    'ResponseConfig',

    # NEAT configs
    'NEATConfig',
    'MutationConfig',
    'GenomeConfig',
    'SpeciesConfig',
    'SelectionConfig',

    # HyperNEAT configs
    'HyperNEATConfig',
    'CPPNConfig',
    'CPPNGenomeConfig',
    'CPPNMutationConfig',
    'CPPNActivationConfig',
    'SubstrateConfig',

    # Experiment configs
    'ExperimentConfig',
    'PerformanceConfig',
    'MetricsConfig',
    'CheckpointConfig',
    'InfrastructureConfig',
    'OutputConfig',
]
