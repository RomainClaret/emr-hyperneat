"""
Base configuration schemas shared across NEAT and HyperNEAT.

These schemas define common gene configurations (weights, biases, responses),
activation functions, and aggregation functions used by both NEAT and HyperNEAT.

All defaults verified against TensorNEAT library source code:
- tensorneat/genome/gene/conn/default.py (weight parameters)
- tensorneat/genome/gene/node/default.py (bias/response parameters)
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Set, Optional


def impl_meta(
    supported_by: Set[str],
    pureples_key: Optional[str] = None,
    tensorneat_path: Optional[str] = None,
    notes: Optional[str] = None
) -> dict:
    """Generate implementation metadata for Field json_schema_extra.

    This helper function creates standardized metadata annotations for schema parameters,
    documenting which implementations support each parameter and providing mapping information.

    Args:
        supported_by: Set of implementation names as strings.
            Examples:
            - {"pureples"} - Only PUREPLES supports this parameter
            - {"tensorneat"} - Only TensorNEAT supports this parameter
            - {"pureples", "tensorneat"} - Both implementations support it
            - {"pureples", "tensorneat", "neat-python"} - Multiple implementations
        pureples_key: NEAT-python config key for PUREPLES mapping validation (optional)
        tensorneat_path: TensorNEAT parameter path if different from schema path (optional)
        notes: Implementation-specific notes or caveats (optional)

    Returns:
        Dictionary suitable for Field's json_schema_extra parameter

    Examples:
        PUREPLES-only parameter:
        ```python
        enabled_mutate_rate: float = Field(
            default=0.01,
            json_schema_extra=impl_meta(
                supported_by={"pureples"},
                pureples_key="enabled_mutate_rate",
                notes="PUREPLES-specific behavior"
            )
        )
        ```

        Multi-implementation parameter:
        ```python
        max_nodes: int = Field(
            default=50,
            json_schema_extra=impl_meta(
                supported_by={"pureples", "tensorneat"},
                pureples_key="max_nodes",
                tensorneat_path="genome.max_nodes",
                notes="Both implementations support complexity limits"
            )
        )
        ```
    """
    return {
        "implementation_support": {
            "supported_by": list(supported_by),
            "pureples_config_key": pureples_key,
            "tensorneat_equivalent": tensorneat_path,
            "notes": notes
        }
    }


class WeightConfig(BaseModel):
    """Connection weight gene configuration.

    Controls initialization and mutation of connection weights in neural networks.
    Verified against: tensorneat/genome/gene/conn/default.py:13-21
    """

    # Initialization
    init_mean: float = Field(
        default=0.0,
        description="Mean value for weight initialization (normal distribution)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    init_std: float = Field(
        default=1.0,
        ge=0.0,
        description="Standard deviation for weight initialization",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    init_type: Literal['gaussian', 'uniform'] = Field(
        default='gaussian',
        description="Distribution type for weight initialization",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    min_value: float = Field(
        default=-5.0,
        description="Minimum allowed weight value (clipping bound)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    max_value: float = Field(
        default=5.0,
        description="Maximum allowed weight value (clipping bound)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Mutation
    mutate_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of mutating a weight value",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    mutate_power: float = Field(
        default=0.15,
        ge=0.0,
        description="Standard deviation of Gaussian noise added during mutation",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    replace_rate: float = Field(
        default=0.015,
        ge=0.0,
        le=1.0,
        description="Probability of completely replacing a weight with new random value",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    class Config:
        json_schema_extra = {
            "description": "Connection weight configuration (init + mutation)",
            "examples": [{
                "init_mean": 0.0,
                "init_std": 1.0,
                "mutate_rate": 0.2,
                "mutate_power": 0.15
            }]
        }


class BiasConfig(BaseModel):
    """Node bias gene configuration.

    Controls initialization and mutation of node bias values in neural networks.
    Verified against: tensorneat/genome/gene/node/default.py:25-33
    """

    # Initialization
    init_mean: float = Field(
        default=0.0,
        description="Mean value for bias initialization (normal distribution)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    init_std: float = Field(
        default=1.0,
        ge=0.0,
        description="Standard deviation for bias initialization (stdev not std!)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    init_type: Literal['gaussian', 'uniform'] = Field(
        default='gaussian',
        description="Distribution type for bias initialization",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    min_value: float = Field(
        default=-5.0,
        description="Minimum allowed bias value (clipping bound)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    max_value: float = Field(
        default=5.0,
        description="Maximum allowed bias value (clipping bound)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Mutation
    mutate_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of mutating a bias value",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    mutate_power: float = Field(
        default=0.15,
        ge=0.0,
        description="Standard deviation of Gaussian noise added during mutation",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    replace_rate: float = Field(
        default=0.015,
        ge=0.0,
        le=1.0,
        description="Probability of completely replacing a bias with new random value",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    class Config:
        json_schema_extra = {
            "description": "Node bias configuration (init + mutation)",
            "examples": [{
                "init_mean": 0.0,
                "init_std": 1.0,
                "mutate_rate": 0.2,
                "mutate_power": 0.15
            }]
        }


class ResponseConfig(BaseModel):
    """Node response gene configuration.

    Controls node response multiplier (output scaling factor).
    Verified against: tensorneat/genome/gene/node/default.py
    """

    # Initialization
    init_mean: float = Field(
        default=1.0,
        description="Mean value for response initialization",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    init_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation for response initialization (stdev not std!)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    min_value: float = Field(
        default=-5.0,
        description="Minimum allowed response value",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    max_value: float = Field(
        default=5.0,
        description="Maximum allowed response value",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Mutation
    mutate_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of mutating response value (default 0 = disabled)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    mutate_power: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of mutation noise (default 0 = disabled)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    replace_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of replacing response value (default 0 = disabled)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    class Config:
        json_schema_extra = {
            "description": "Node response configuration (output scaling)",
            "examples": [{
                "init_mean": 1.0,
                "init_std": 0.0,
                "mutate_rate": 0.0
            }]
        }


class ActivationConfig(BaseModel):
    """Activation function configuration.

    Defines available activation functions and default selection.
    Used for both NEAT and HyperNEAT networks.
    """

    default: Literal['tanh', 'sigmoid', 'relu', 'identity', 'sin', 'cos', 'gauss', 'abs', 'square', 'cube'] = Field(
        default='sigmoid',
        description="Default activation function for new nodes",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    options: List[Literal['tanh', 'sigmoid', 'relu', 'identity', 'sin', 'cos', 'gauss', 'abs', 'square', 'cube']] = Field(
        default=['sigmoid', 'tanh', 'relu', 'identity', 'sin', 'gauss'],
        description="Available activation functions for mutation",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    mutate_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of changing node activation function (default 0 = disabled)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    class Config:
        json_schema_extra = {
            "description": "Activation function configuration",
            "examples": [{
                "default": "sigmoid",
                "options": ["sigmoid", "tanh", "relu"],
                "mutate_rate": 0.1
            }]
        }


class AggregationConfig(BaseModel):
    """Aggregation function configuration.

    Defines how multiple inputs to a node are combined.
    TensorNEAT currently only supports 'sum' aggregation.
    """

    default: Literal['sum', 'product', 'min', 'max', 'mean', 'median'] = Field(
        default='sum',
        description="Default aggregation function for new nodes",
        json_schema_extra=impl_meta(
            supported_by={"pureples", "tensorneat"},
            notes="⚠️ TensorNEAT only supports 'sum' - other values will fail"
        )
    )

    options: List[Literal['sum', 'product', 'min', 'max', 'mean', 'median']] = Field(
        default=['sum'],
        description="Available aggregation functions (TensorNEAT supports 'sum' only)",
        json_schema_extra=impl_meta(
            supported_by={"pureples", "tensorneat"},
            notes="⚠️ TensorNEAT limited to ['sum'], PUREPLES supports all aggregation functions"
        )
    )

    mutate_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of changing node aggregation function (default 0 = disabled)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    class Config:
        json_schema_extra = {
            "description": "Aggregation function configuration",
            "examples": [{
                "default": "sum",
                "options": ["sum"],
                "mutate_rate": 0.0
            }]
        }
