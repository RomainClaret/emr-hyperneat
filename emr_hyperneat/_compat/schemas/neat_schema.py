"""
NEAT algorithm configuration schema.

Defines complete NEAT (NeuroEvolution of Augmenting Topologies) configuration
including population, mutation, genome structure, speciation, and selection.

All defaults verified against TensorNEAT library source code:
- tensorneat/algorithm/neat/neat.py (population, species parameters)
- tensorneat/genome/operations/mutation/default.py (structural mutations)
- tensorneat/genome/default.py (network structure limits)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from .base import (
    WeightConfig,
    BiasConfig,
    ResponseConfig,
    ActivationConfig,
    AggregationConfig,
    impl_meta,
)


class MutationConfig(BaseModel):
    """Structural mutation configuration.

    Controls probabilities for adding/deleting nodes and connections during evolution.
    Verified against: tensorneat/genome/operations/mutation/default.py:22-27
    """

    # Connection mutations
    conn_add_prob: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of adding a new connection between existing nodes",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    conn_delete_prob: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of deleting an existing connection",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Node mutations
    node_add_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of adding a new hidden node (splits connection)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    node_delete_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of deleting an existing hidden node",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    class Config:
        json_schema_extra = {
            "description": "Structural mutation probabilities (topology changes)",
            "examples": [{
                "conn_add_prob": 0.2,
                "conn_delete_prob": 0.2,
                "node_add_prob": 0.1,
                "node_delete_prob": 0.1
            }]
        }


class GenomeConfig(BaseModel):
    """Genome structure configuration.

    Defines network topology, initial structure, and complexity limits.
    Verified against: tensorneat/genome/default.py
    """

    # Network topology
    num_inputs: int = Field(
        default=2,
        ge=1,
        description="Number of input nodes (determined by problem)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    num_outputs: int = Field(
        default=1,
        ge=1,
        description="Number of output nodes (determined by problem)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    num_hidden: int = Field(
        default=0,
        ge=0,
        description="Number of hidden nodes in initial topology",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Bias handling (PUREPLES-specific)
    use_bias: bool = Field(
        default=True,
        description="Whether to include bias as external input (PUREPLES only). When True, problem provides additional bias input (e.g., XOR: 2+1=3 inputs). Auto-detected num_inputs will reflect this setting.",
        json_schema_extra=impl_meta(
            supported_by={"pureples"},
            notes="PUREPLES-specific: Controls whether problem includes bias in input data. TensorNEAT uses internal bias nodes instead."
        )
    )

    # Complexity limits
    # NOTE: These are TensorNEAT-specific, NOT in official NEAT-python docs!
    max_nodes: int = Field(
        default=50,
        ge=1,
        description="Maximum number of nodes allowed in genome. TensorNEAT-only (genome.max_nodes), NOT in NEAT-python.",
        json_schema_extra=impl_meta(
            supported_by={"tensorneat"},  # TensorNEAT-only, NOT PUREPLES!
            notes="TensorNEAT-specific: genome.max_nodes attribute. NOT in official NEAT-python documentation."
        )
    )
    max_connections: int = Field(
        default=100,
        ge=1,
        description="Maximum number of connections allowed in genome. TensorNEAT-only (genome.max_conns), NOT in NEAT-python.",
        json_schema_extra=impl_meta(
            supported_by={"tensorneat"},  # TensorNEAT-only, NOT PUREPLES!
            notes="TensorNEAT-specific: genome.max_conns attribute. NOT in official NEAT-python documentation."
        )
    )

    # Mutation constraints
    single_structural_mutation: bool = Field(
        default=False,
        description="If True, only one structural mutation (add/delete node/connection) per genome per generation",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Initial topology
    feed_forward: bool = Field(
        default=True,
        description="Whether network is feedforward (no recurrent connections)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    initial_connection: Literal['full', 'full_direct', 'full_nodirect', 'partial', 'partial_direct', 'partial_nodirect', 'unconnected'] = Field(
        default='full',
        description="Initial connection pattern (NEAT-python): full/full_direct/full_nodirect (all inputs→outputs), partial/partial_direct/partial_nodirect (fraction), or unconnected",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    connection_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of possible connections to create when initial_connection='partial'",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    enabled_default: bool = Field(
        default=True,
        description="Whether new connections are enabled by default",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Enabled connection mutation parameters (PUREPLES-specific behavior)
    enabled_mutate_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Probability of toggling connection enabled state",
        json_schema_extra=impl_meta(
            supported_by={"pureples"},
            pureples_key="enabled_mutate_rate",
            notes="PUREPLES-specific: controls enabled state mutation probability"
        )
    )
    enabled_rate_to_false_add: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of disabling new connections when added (PUREPLES-specific)",
        json_schema_extra=impl_meta(
            supported_by={"pureples"},
            pureples_key="enabled_rate_to_false_add",
            notes="PUREPLES-specific: controls initial enabled state of new connections"
        )
    )
    enabled_rate_to_true_add: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of enabling new connections when added (PUREPLES-specific)",
        json_schema_extra=impl_meta(
            supported_by={"pureples"},
            pureples_key="enabled_rate_to_true_add",
            notes="PUREPLES-specific: complements enabled_rate_to_false_add"
        )
    )

    # Structural mutation strategy (PUREPLES-specific)
    structural_mutation_surer: str = Field(
        default='default',
        description="Strategy for structural mutations: 'default' or 'surer' (PUREPLES-specific)",
        json_schema_extra=impl_meta(
            supported_by={"pureples"},
            pureples_key="structural_mutation_surer",
            notes="PUREPLES-specific: 'surer' ensures at least one structural mutation per generation"
        )
    )

    # Gene configurations (nested)
    weight: WeightConfig = Field(
        default_factory=WeightConfig,
        description="Connection weight gene configuration"
    )
    bias: BiasConfig = Field(
        default_factory=BiasConfig,
        description="Node bias gene configuration"
    )
    response: ResponseConfig = Field(
        default_factory=ResponseConfig,
        description="Node response gene configuration"
    )
    activation: ActivationConfig = Field(
        default_factory=ActivationConfig,
        description="Activation function configuration"
    )
    aggregation: AggregationConfig = Field(
        default_factory=AggregationConfig,
        description="Aggregation function configuration"
    )

    # Advanced genome features
    input_transform: Optional[str] = Field(
        default=None,
        description="Optional input transformation function name (None = no transformation). TensorNEAT-only.",
        json_schema_extra=impl_meta(
            supported_by={"tensorneat"},
            notes="Advanced feature: Allows preprocessing inputs before network computation. Rarely used."
        )
    )
    init_hidden_layers: tuple = Field(
        default=(),
        description="Initial hidden layer sizes for topology initialization (empty tuple = minimal topology). TensorNEAT-only.",
        json_schema_extra=impl_meta(
            supported_by={"tensorneat"},
            notes="Specifies initial hidden nodes instead of starting with direct input→output connections"
        )
    )

    class Config:
        json_schema_extra = {
            "description": "Genome structure and gene configurations",
            "examples": [{
                "num_inputs": 2,
                "num_outputs": 1,
                "max_nodes": 50,
                "feed_forward": True
            }]
        }


class SpeciesConfig(BaseModel):
    """Speciation configuration.

    Controls how genomes are grouped into species for protected innovation.
    Verified against: tensorneat/algorithm/neat/neat.py
    """

    enabled: bool = Field(
        default=True,
        description="Whether speciation is enabled",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    compatibility_threshold: float = Field(
        default=2.0,
        ge=0.0,
        description="Genomic distance threshold for species membership",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    species_size: int = Field(
        default=10,
        ge=1,
        description="Target number of species to maintain",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Stagnation
    max_stagnation: int = Field(
        default=15,
        ge=1,
        description="Generations without improvement before species considered stagnant",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    species_elitism: int = Field(
        default=2,
        ge=0,
        description="Number of species protected from stagnation removal",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    bad_species_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fitness threshold below which species are considered bad (PUREPLES-specific, optional)",
        json_schema_extra=impl_meta(
            supported_by={"pureples"},
            notes="PUREPLES-specific: Optional fitness threshold for identifying underperforming species"
        )
    )

    # Species fitness
    species_fitness_func: Literal['max', 'mean', 'min'] = Field(
        default='max',
        description="How to calculate species fitness (max, mean, or min of members)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Compatibility coefficients (for genomic distance calculation)
    disjoint_coefficient: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of disjoint genes in compatibility distance",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    weight_coefficient: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight of weight differences in compatibility distance",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    spawn_number_change_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Smoothing factor for gradual species population adjustments between generations (0=instant change, 1=no change)",
        json_schema_extra=impl_meta(
            supported_by={"tensorneat"},
            notes="Controls interpolation between previous size and target spawn number: new_size = prev * rate + target * (1-rate)"
        )
    )
    species_number_calculate_by: Literal['rank', 'fitness'] = Field(
        default='rank',
        description="Method for calculating offspring allocation to species ('rank' uses linear ranking, 'fitness' uses fitness-proportional)",
        json_schema_extra=impl_meta(supported_by={"tensorneat"})
    )

    class Config:
        json_schema_extra = {
            "description": "Speciation and genomic distance configuration",
            "examples": [{
                "compatibility_threshold": 2.0,
                "max_stagnation": 15,
                "species_elitism": 2
            }]
        }


class SelectionConfig(BaseModel):
    """Selection and reproduction configuration.

    Controls survival, elitism, and parent selection for next generation.
    Verified against: tensorneat/algorithm/neat/neat.py
    """

    # Elitism (direct copying to next generation)
    genome_elitism: int = Field(
        default=2,
        ge=0,
        description="Number of best genomes copied unchanged to next generation",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Survival
    survival_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of each species allowed to reproduce (top X%)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    min_species_size: int = Field(
        default=1,
        ge=1,
        description="Minimum number of genomes in a species to survive",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Reproduction
    # NOTE: These parameters are NOT in official NEAT-python documentation!
    # Verified against: https://neat-python.readthedocs.io/en/latest/config_file.html
    # [DefaultReproduction] only has: elitism, survival_threshold, min_species_size
    crossover_rate: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Probability of crossover vs cloning for reproduction. NOT IN NEAT-PYTHON - crossover is hardcoded. Invalid parameter.",
        json_schema_extra=impl_meta(
            supported_by=set(),  # NOT supported by any implementation!
            notes="NOT in official NEAT-python docs. Crossover is hardcoded in NEAT algorithm. This parameter has no effect."
        )
    )
    interspecies_crossover_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Probability of mating between different species. NOT IN NEAT-PYTHON. Invalid parameter.",
        json_schema_extra=impl_meta(
            supported_by=set(),  # NOT supported by any implementation!
            notes="NOT in official NEAT-python docs or TensorNEAT constructor. This parameter has no effect."
        )
    )

    # Tournament selection (optional)
    tournament_size: Optional[int] = Field(
        default=None,
        ge=2,
        description="Tournament size for parent selection (None = fitness proportional). TensorNEAT-only, NOT in NEAT-python.",
        json_schema_extra=impl_meta(
            supported_by={"tensorneat"},  # TensorNEAT-only, NOT PUREPLES!
            notes="TensorNEAT-specific parameter. NOT in official NEAT-python documentation."
        )
    )

    class Config:
        json_schema_extra = {
            "description": "Selection and reproduction configuration",
            "examples": [{
                "genome_elitism": 2,
                "survival_threshold": 0.1,
                "crossover_rate": 0.7
            }]
        }


class FitnessConfig(BaseModel):
    """Fitness evaluation configuration.

    Controls fitness calculation and termination criteria.
    """

    threshold: float = Field(
        default=0.98,
        description="Fitness threshold for termination (success criterion)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    criterion: Literal['max', 'min', 'mean'] = Field(
        default='max',
        description="Fitness criterion: maximize, minimize, or mean",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    normalization: Literal['raw', 'rank', 'normalized'] = Field(
        default='raw',
        description="Fitness normalization method",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    no_termination: bool = Field(
        default=False,
        description="If True, evolution continues even after fitness threshold is reached",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Fitness sharing (optional)
    sharing_enabled: bool = Field(
        default=False,
        description="Whether to enable fitness sharing within species",
        json_schema_extra=impl_meta(
            supported_by={"pureples"},
            notes="PUREPLES-specific: Enable fitness sharing within species for diversity"
        )
    )
    sharing_delta: float = Field(
        default=0.1,
        ge=0.0,
        description="Distance threshold for fitness sharing",
        json_schema_extra=impl_meta(
            supported_by={"pureples"},
            notes="PUREPLES-specific: Distance threshold for fitness sharing calculations"
        )
    )

    class Config:
        json_schema_extra = {
            "description": "Fitness evaluation and termination",
            "examples": [{
                "threshold": 0.98,
                "criterion": "max",
                "normalization": "raw"
            }]
        }


class NEATConfig(BaseModel):
    """Complete NEAT algorithm configuration.

    Top-level configuration combining all NEAT components:
    - Population and evolution parameters
    - Genome structure and gene configurations
    - Mutation operators
    - Speciation and compatibility
    - Selection and reproduction
    - Fitness evaluation

    This is the single source of truth for NEAT experiments.
    """

    # Population
    population_size: int = Field(
        default=150,
        ge=2,
        description="Number of genomes in population",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    reset_on_extinction: bool = Field(
        default=False,
        description="If True, create new random population when all species go extinct",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )

    # Evolution control
    max_generations: int = Field(
        default=100,
        ge=1,
        description="Maximum number of generations to evolve",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility (None = random)",
        json_schema_extra=impl_meta(supported_by={"pureples", "tensorneat"})
    )
    verbose: bool = Field(
        default=True,
        description="Enable debug output during evolution. TensorNEAT-only.",
        json_schema_extra=impl_meta(
            supported_by={"tensorneat"},
            notes="Controls console output during evolution. Useful for debugging."
        )
    )

    # Component configurations (nested)
    genome: GenomeConfig = Field(
        default_factory=GenomeConfig,
        description="Genome structure and gene configurations"
    )
    mutation: MutationConfig = Field(
        default_factory=MutationConfig,
        description="Structural mutation probabilities"
    )
    species: SpeciesConfig = Field(
        default_factory=SpeciesConfig,
        description="Speciation and compatibility configuration"
    )
    selection: SelectionConfig = Field(
        default_factory=SelectionConfig,
        description="Selection and reproduction configuration"
    )
    fitness: FitnessConfig = Field(
        default_factory=FitnessConfig,
        description="Fitness evaluation configuration"
    )

    @field_validator('population_size')
    @classmethod
    def validate_population_size(cls, v: int) -> int:
        """Ensure population size is reasonable."""
        if v < 2:
            raise ValueError("Population size must be at least 2")
        if v > 10000:
            raise ValueError("Population size > 10000 is likely impractical")
        return v

    @field_validator('max_generations')
    @classmethod
    def validate_max_generations(cls, v: int) -> int:
        """Ensure max generations is reasonable."""
        if v < 1:
            raise ValueError("Max generations must be at least 1")
        if v > 100000:
            raise ValueError("Max generations > 100000 is likely impractical")
        return v

    class Config:
        json_schema_extra = {
            "title": "NEAT Configuration",
            "description": "Complete NEAT algorithm configuration",
            "examples": [{
                "population_size": 150,
                "max_generations": 100,
                "genome": {
                    "num_inputs": 2,
                    "num_outputs": 1,
                    "max_nodes": 50
                },
                "mutation": {
                    "conn_add_prob": 0.2,
                    "node_add_prob": 0.1
                },
                "selection": {
                    "genome_elitism": 2,
                    "survival_threshold": 0.1
                }
            }]
        }
