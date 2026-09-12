"""
HyperNEAT algorithm configuration schema.

Defines complete HyperNEAT configuration including CPPN (Compositional Pattern
Producing Network) evolution and substrate parameters.

HyperNEAT extends NEAT by evolving CPPNs that generate connection weights for
a fixed substrate topology, enabling indirect encoding of large networks.

All defaults verified against TensorNEAT library and YAML configs:
- tensorneat_config_builder.py lines 150-177 (CPPN and substrate params)
- configs/implementations/tensorneat/hyperneat_default.yaml
"""

from pydantic import BaseModel, Field
from typing import List, Literal
from .neat_schema import NEATConfig, MutationConfig, SpeciesConfig, SelectionConfig
from .base import (
    WeightConfig,
    BiasConfig,
    ResponseConfig,
    ActivationConfig,
)


class CPPNActivationConfig(BaseModel):
    """CPPN-specific activation function configuration.

    CPPNs use specialized activation functions for generating spatial patterns:
    - sin: Periodic patterns (regularity) - NEAT-python supports sin but NOT cos
    - gauss: Localized patterns (symmetry)
    - sigmoid, tanh: Standard non-linearities
    - relu, identity: Linear/piecewise patterns

    NOTE: 'cos' is NOT supported by NEAT-python despite being theoretically useful for CPPNs.
    """

    default: Literal['sigmoid', 'tanh', 'sin', 'gauss', 'relu', 'identity'] = Field(
        default='sigmoid',
        description="Default activation function for new CPPN nodes (NEAT-python compatible)"
    )

    options: List[Literal['sigmoid', 'tanh', 'sin', 'gauss', 'relu', 'identity', 'abs', 'square', 'cube']] = Field(
        default=['sigmoid', 'tanh', 'sin', 'gauss', 'relu', 'identity'],
        description="Available activation functions for CPPN mutation (NEAT-python compatible)"
    )

    mutate_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of changing CPPN node activation function"
    )

    class Config:
        json_schema_extra = {
            "description": "CPPN activation functions (spatial pattern generation)",
            "examples": [{
                "default": "sigmoid",
                "options": ["sigmoid", "tanh", "sin", "gauss"],
                "mutate_rate": 0.1
            }]
        }


class CPPNMutationConfig(BaseModel):
    """CPPN structural mutation configuration.

    Controls topology evolution of the CPPN that generates substrate weights.
    Verified against: tensorneat_config_builder.py lines 170-173
    """

    # Connection mutations
    conn_add_prob: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of adding new CPPN connection"
    )
    conn_delete_prob: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of deleting CPPN connection"
    )

    # Node mutations
    node_add_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of adding new CPPN node"
    )
    node_delete_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of deleting CPPN node"
    )

    class Config:
        json_schema_extra = {
            "description": "CPPN structural mutation probabilities",
            "examples": [{
                "conn_add_prob": 0.2,
                "node_add_prob": 0.1
            }]
        }


class CPPNGenomeConfig(BaseModel):
    """CPPN genome structure configuration.

    The CPPN takes spatial coordinates as input and outputs connection weights.
    Inputs: (x1, y1, z1, x2, y2, z2) - source and target node coordinates
    Outputs: connection weight, bias, etc.
    """

    # CPPN topology (determined by substrate)
    num_inputs: int = Field(
        default=4,  # 2D substrate: (x1, y1, x2, y2)
        ge=2,
        description="CPPN input dimensions (2*substrate_dimensions)"
    )
    num_outputs: int = Field(
        default=1,  # Connection weight
        ge=1,
        description="CPPN output dimensions (weight, bias, etc.)"
    )

    # Complexity limits
    max_nodes: int = Field(
        default=50,
        ge=1,
        description="Maximum number of nodes in CPPN"
    )

    # Value bounds (for weight/bias outputs)
    min_value: float = Field(
        default=-5.0,
        description="Minimum CPPN output value"
    )
    max_value: float = Field(
        default=5.0,
        description="Maximum CPPN output value"
    )

    # Gene configurations (CPPN-specific)
    weight: WeightConfig = Field(
        default_factory=lambda: WeightConfig(
            mutate_rate=0.2,
            mutate_power=0.15,
            replace_rate=0.015
        ),
        description="CPPN connection weight configuration"
    )
    bias: BiasConfig = Field(
        default_factory=lambda: BiasConfig(
            init_std=1.0,
            mutate_rate=0.2,
            mutate_power=0.15,
            replace_rate=0.015,
            min_value=-5.0,
            max_value=5.0
        ),
        description="CPPN node bias configuration"
    )
    response: ResponseConfig = Field(
        default_factory=lambda: ResponseConfig(
            init_std=0.0,
            mutate_rate=0.2,
            mutate_power=0.15,
            replace_rate=0.015,
            min_value=-5.0,
            max_value=5.0
        ),
        description="CPPN node response configuration"
    )
    activation: CPPNActivationConfig = Field(
        default_factory=CPPNActivationConfig,
        description="CPPN activation function configuration"
    )

    class Config:
        json_schema_extra = {
            "description": "CPPN genome structure and gene configurations",
            "examples": [{
                "num_inputs": 4,
                "num_outputs": 1,
                "max_nodes": 50,
                "min_value": -5.0,
                "max_value": 5.0
            }]
        }


class CPPNConfig(BaseModel):
    """Complete CPPN evolution configuration.

    The CPPN (Compositional Pattern Producing Network) is evolved to generate
    substrate connection weights based on spatial coordinates. This enables
    indirect encoding of large, regular network patterns.

    Verified against: tensorneat_config_builder.py lines 158-177
    """

    # CPPN population (can be different from substrate population)
    population_size: int = Field(
        default=150,
        ge=2,
        description="CPPN population size (can differ from substrate population)"
    )

    # CPPN evolution
    genome: CPPNGenomeConfig = Field(
        default_factory=CPPNGenomeConfig,
        description="CPPN genome structure and gene configurations"
    )
    mutation: CPPNMutationConfig = Field(
        default_factory=CPPNMutationConfig,
        description="CPPN structural mutation probabilities"
    )

    # CPPN speciation (typically uses same settings as substrate NEAT)
    species: SpeciesConfig = Field(
        default_factory=SpeciesConfig,
        description="CPPN speciation configuration"
    )

    # CPPN selection
    selection: SelectionConfig = Field(
        default_factory=lambda: SelectionConfig(
            genome_elitism=2,  # cppn_genome_elitism
            survival_threshold=0.1,
            crossover_rate=0.7
        ),
        description="CPPN selection and reproduction configuration"
    )

    class Config:
        json_schema_extra = {
            "description": "CPPN evolution configuration for HyperNEAT",
            "examples": [{
                "population_size": 150,
                "genome": {
                    "num_inputs": 4,
                    "num_outputs": 1,
                    "max_nodes": 50
                },
                "selection": {
                    "genome_elitism": 2
                }
            }]
        }


class SubstrateConfig(BaseModel):
    """Substrate network configuration.

    The substrate is the actual neural network whose weights are generated by
    the CPPN. It has a fixed topology (grid-based or custom) and evolves only
    through CPPN evolution, not direct structural mutations.

    Verified against: tensorneat_config_builder.py lines 150-156
    """

    # Weight generation
    weight_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum absolute CPPN output to create substrate connection (pruning)"
    )
    max_weight: float = Field(
        default=5.0,
        ge=0.0,
        description="Maximum absolute value for substrate connection weights (clipping)"
    )

    # Activation
    activate_time: int = Field(
        default=10,
        ge=1,
        description="Number of activation iterations for substrate network (recurrent settling)"
    )

    # Output transformation
    output_activation: Literal['sigmoid', 'tanh', 'relu', 'linear', 'identity'] = Field(
        default='sigmoid',
        description="Activation function for substrate output layer"
    )
    hidden_activation: Literal['sigmoid', 'tanh', 'relu', 'linear', 'identity'] = Field(
        default='tanh',
        description="Activation function for substrate hidden layers"
    )

    class Config:
        json_schema_extra = {
            "description": "Substrate network configuration (CPPN-generated weights)",
            "examples": [{
                "weight_threshold": 0.3,
                "max_weight": 5.0,
                "activate_time": 10,
                "output_activation": "sigmoid"
            }]
        }


class HyperNEATConfig(NEATConfig):
    """Complete HyperNEAT configuration.

    HyperNEAT extends NEAT by:
    1. Evolving CPPNs (not substrate networks directly)
    2. Using CPPNs to generate substrate connection weights
    3. Evaluating fitness on substrate network performance

    The substrate has fixed topology while the CPPN evolves to discover
    effective weight patterns based on geometric regularities.

    Inherits all NEAT configuration and adds CPPN + substrate.
    """

    # Override algorithm type (supports both HyperNEAT and ES-HyperNEAT)
    algorithm: Literal['hyperneat', 'es_hyperneat', 'eshyperneat'] = Field(
        default='hyperneat',
        description="Algorithm type (hyperneat, es_hyperneat, or eshyperneat)"
    )

    # HyperNEAT-specific components
    cppn: CPPNConfig = Field(
        default_factory=CPPNConfig,
        description="CPPN evolution configuration"
    )
    substrate: SubstrateConfig = Field(
        default_factory=SubstrateConfig,
        description="Substrate network configuration"
    )

    class Config:
        json_schema_extra = {
            "title": "HyperNEAT Configuration",
            "description": "Complete HyperNEAT algorithm configuration (NEAT + CPPN + Substrate)",
            "examples": [{
                "population_size": 150,
                "max_generations": 100,
                "algorithm": "hyperneat",
                "cppn": {
                    "population_size": 150,
                    "genome": {
                        "num_inputs": 4,
                        "num_outputs": 1
                    }
                },
                "substrate": {
                    "weight_threshold": 0.3,
                    "activate_time": 10
                }
            }]
        }
