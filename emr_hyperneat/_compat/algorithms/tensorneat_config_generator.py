"""
TensorNEAT Configuration Generator - Schema-Based

Schema-based configuration generator using Pydantic for type safety and validation.

This generator:
1. Accepts NEATConfig/HyperNEATConfig schemas instead of dicts
2. Uses Pydantic schemas as single source of truth
3. Eliminates DEFAULT_VALUES registry (replaced by schema defaults)
4. Uses direct attribute access (type-safe)
5. Provides better error messages (Pydantic validation)

Naming coherence with PUREPLESConfigGenerator pattern.
"""

from typing import Any, Optional, Union
import logging

from emr_hyperneat._compat.schemas import NEATConfig, HyperNEATConfig
from emr_hyperneat._compat.core.parameter_support_validator import ParameterSupportValidator

logger = logging.getLogger(__name__)


class TensorNEATConfigGenerator:
    """Schema-based configuration generator for TensorNEAT implementations.

    Uses Pydantic schemas as single source of truth for all parameters.
    Provides type-safe configuration generation with validation.

    Naming coherence with PUREPLESConfigGenerator pattern.

    Usage:
        # From schema
        config = NEATConfig(population_size=150)
        generator = TensorNEATConfigGenerator(config)
        neat = generator.build_neat_config(jax, ACT, AGG)

        # From dict (for backward compatibility)
        generator = TensorNEATConfigGenerator.from_dict(params)
    """

    def __init__(self, config: Union[NEATConfig, HyperNEATConfig], validate: bool = True):
        """Initialize builder with validated schema.

        Args:
            config: NEATConfig or HyperNEATConfig schema instance
            validate: Whether to validate parameter support (default: True)

        Raises:
            ValueError: If TensorNEAT-specific constraints are violated (e.g., aggregation!='sum')
        """
        self.config = config
        self.is_hyperneat = isinstance(config, HyperNEATConfig)

        # Validate parameter support for TensorNEAT implementation
        if validate:
            result = ParameterSupportValidator.validate(config, 'tensorneat')
            if result.warnings:
                result.log_warnings()
            if not result.valid:
                error_msg = "\n".join(result.errors)
                raise ValueError(f"TensorNEAT configuration validation failed:\n{error_msg}")

    @classmethod
    def from_dict(cls, params: dict) -> 'TensorNEATConfigGenerator':
        """Create builder from parameter dictionary (backward compatibility).

        Args:
            params: Parameter dictionary from web UI, YAML, or grid search

        Returns:
            TensorNEATConfigGenerator instance

        Raises:
            ValidationError: If parameters are invalid
        """
        # FIX: Enhanced HyperNEAT detection with more indicators and better logging
        algorithm = params.get('algorithm', 'neat')

        # Check multiple indicators for HyperNEAT configuration
        is_hyperneat = (
            algorithm in ['hyperneat', 'es_hyperneat', 'eshyperneat'] or
            'cppn' in params or  # Nested CPPN configuration
            'substrate' in params or  # Nested substrate configuration
            'cppn_pop_size' in params or  # Legacy flat CPPN param
            'weight_threshold' in params  # Legacy flat substrate param
        )

        try:
            if is_hyperneat:
                logger.debug(f"Creating HyperNEATConfig from dict (algorithm={algorithm})")
                config = HyperNEATConfig(**params)
            else:
                logger.debug(f"Creating NEATConfig from dict (algorithm={algorithm})")
                config = NEATConfig(**params)
        except Exception as e:
            logger.error(f"Failed to create config from dict: {e}")
            logger.error(f"Algorithm detected: {algorithm}, is_hyperneat: {is_hyperneat}")
            logger.error(f"Params keys (first 20): {list(params.keys())[:20]}")
            raise

        return cls(config)

    @staticmethod
    def _pureples_gaussian(x):
        """PUREPLES-compatible Gaussian activation function.

        PUREPLES uses: exp(-x²/2)
        TensorNEAT uses: exp(-x²)

        This function matches PUREPLES exactly for ES-HyperNEAT CPPN evolution.

        Args:
            x: Input tensor

        Returns:
            exp(-x²/2)
        """
        import jax.numpy as jnp
        return jnp.exp(-0.5 * x**2)

    def _get_activation_function(self, name: str, jax_module, act_module):
        """Convert activation function name to JAX/TensorNEAT function.

        Args:
            name: Activation function name from schema
            jax_module: JAX module
            act_module: TensorNEAT ACT module

        Returns:
            Activation function object
        """
        # Map string names to TensorNEAT ACT module functions
        # Only include functions actually available in TensorNEAT
        activation_map = {
            'sigmoid': act_module.sigmoid,
            'tanh': act_module.tanh,
            'relu': act_module.relu,
            'identity': act_module.identity,
            'sin': act_module.sin,
            'gauss': act_module.gauss,  # TensorNEAT's exp(-z²)
            'gauss_pureples': self._pureples_gaussian,  # PUREPLES exp(-z²/2)
            # Note: TensorNEAT doesn't have: cos, abs, square, cube
        }
        return activation_map.get(name, act_module.sigmoid)

    def _get_aggregation_function(self, name: str, agg_module):
        """Convert aggregation function name to TensorNEAT function.

        Args:
            name: Aggregation function name from schema
            agg_module: TensorNEAT AGG module

        Returns:
            Aggregation function object
        """
        # TensorNEAT only supports 'sum' aggregation currently
        aggregation_map = {
            'sum': agg_module.sum,
            # Other aggregation functions not supported by TensorNEAT yet
        }
        return aggregation_map.get(name, agg_module.sum)

    def _get_species_fitness_function(self, name: str, jax_module) -> Any:
        """Convert species fitness function name to JAX function.

        Args:
            name: Species fitness function name ('max', 'mean', 'min')
            jax_module: JAX module for numpy functions

        Returns:
            JAX numpy function object
        """
        fitness_func_map = {
            'max': jax_module.numpy.max,
            'mean': jax_module.numpy.mean,
            'min': jax_module.numpy.min
        }
        return fitness_func_map.get(name, jax_module.numpy.max)

    def build_neat_config(self, jax_module, act_module, agg_module) -> Any:
        """Build complete NEAT algorithm configuration.

        Creates TensorNEAT NEAT algorithm with validated parameters from schema.

        Args:
            jax_module: JAX module (for activation functions)
            act_module: TensorNEAT ACT module (activation functions)
            agg_module: TensorNEAT AGG module (aggregation functions)

        Returns:
            Configured TensorNEAT NEAT algorithm instance
        """
        from tensorneat import algorithm
        from tensorneat.genome import (
            DefaultGenome,
            RecurrentGenome,
            DefaultNode,
            DefaultConn,
            DefaultMutation,
            DefaultCrossover,
            DefaultDistance
        )

        # Get validated parameters from schema
        genome_cfg = self.config.genome
        mutation_cfg = self.config.mutation
        species_cfg = self.config.species
        selection_cfg = self.config.selection

        # Create mutation configuration
        mutation = DefaultMutation(
            node_add=mutation_cfg.node_add_prob,
            node_delete=mutation_cfg.node_delete_prob,
            conn_add=mutation_cfg.conn_add_prob,
            conn_delete=mutation_cfg.conn_delete_prob
        )

        # Create node gene configuration
        node_gene = DefaultNode(
            # Bias parameters
            bias_init_mean=genome_cfg.bias.init_mean,
            bias_init_std=genome_cfg.bias.init_std,
            bias_mutate_power=genome_cfg.bias.mutate_power,
            bias_mutate_rate=genome_cfg.bias.mutate_rate,
            bias_replace_rate=genome_cfg.bias.replace_rate,
            bias_lower_bound=genome_cfg.bias.min_value,
            bias_upper_bound=genome_cfg.bias.max_value,
            # Response parameters
            response_init_mean=genome_cfg.response.init_mean,
            response_init_std=genome_cfg.response.init_std,
            response_mutate_power=genome_cfg.response.mutate_power,
            response_mutate_rate=genome_cfg.response.mutate_rate,
            response_replace_rate=genome_cfg.response.replace_rate,
            response_lower_bound=genome_cfg.response.min_value,
            response_upper_bound=genome_cfg.response.max_value,
            # Activation parameters
            activation_default=self._get_activation_function(genome_cfg.activation.default, jax_module, act_module),
            activation_options=tuple(
                self._get_activation_function(name, jax_module, act_module)
                for name in genome_cfg.activation.options
            ),
            activation_replace_rate=genome_cfg.activation.mutate_rate,
            # Aggregation parameters
            aggregation_default=self._get_aggregation_function(genome_cfg.aggregation.default, agg_module),
            aggregation_options=tuple(
                self._get_aggregation_function(name, agg_module)
                for name in genome_cfg.aggregation.options
            ),
            aggregation_replace_rate=genome_cfg.aggregation.mutate_rate
        )

        # Create connection gene configuration
        conn_gene = DefaultConn(
            weight_init_mean=genome_cfg.weight.init_mean,
            weight_init_std=genome_cfg.weight.init_std,
            weight_mutate_power=genome_cfg.weight.mutate_power,
            weight_mutate_rate=genome_cfg.weight.mutate_rate,
            weight_replace_rate=genome_cfg.weight.replace_rate,
            weight_lower_bound=genome_cfg.weight.min_value,
            weight_upper_bound=genome_cfg.weight.max_value
        )

        # Get output activation from schema
        output_activation_name = genome_cfg.activation.default
        output_transform = getattr(act_module, output_activation_name, act_module.sigmoid)

        # Handle input_transform
        input_transform = None
        if genome_cfg.input_transform:
            input_transform = None  # Currently not implemented

        # Convert init_hidden_layers list to tuple
        init_hidden = tuple(genome_cfg.init_hidden_layers) if genome_cfg.init_hidden_layers else ()

        # Select genome class based on feed_forward config
        # TensorNEAT uses genome CLASS selection, not a boolean parameter:
        # - feed_forward=True → DefaultGenome (network_type="feedforward")
        # - feed_forward=False → RecurrentGenome (network_type="recurrent")
        GenomeClass = DefaultGenome if genome_cfg.feed_forward else RecurrentGenome

        # Create genome configuration
        genome_obj = GenomeClass(
            num_inputs=genome_cfg.num_inputs,
            num_outputs=genome_cfg.num_outputs,
            max_nodes=genome_cfg.max_nodes,
            max_conns=genome_cfg.max_connections,
            init_hidden_layers=init_hidden,  # From config (default: empty tuple)
            node_gene=node_gene,
            conn_gene=conn_gene,
            mutation=mutation,
            crossover=DefaultCrossover(),
            distance=DefaultDistance(
                compatibility_disjoint=species_cfg.disjoint_coefficient,
                compatibility_weight=species_cfg.weight_coefficient
            ),
            output_transform=output_transform,
            input_transform=input_transform  # From config (default: None)
        )

        # Get species fitness function from schema
        species_fitness_func = self._get_species_fitness_function(species_cfg.species_fitness_func, jax_module)

        # Create NEAT algorithm with validated schema parameters
        neat_algorithm = algorithm.NEAT(
            genome=genome_obj,
            pop_size=self.config.population_size,
            species_size=species_cfg.species_size,
            max_stagnation=species_cfg.max_stagnation,
            species_elitism=species_cfg.species_elitism,
            spawn_number_change_rate=species_cfg.spawn_number_change_rate,  # Added
            genome_elitism=selection_cfg.genome_elitism,
            survival_threshold=selection_cfg.survival_threshold,
            min_species_size=selection_cfg.min_species_size,  # Added (was in schema but not passed!)
            compatibility_threshold=species_cfg.compatibility_threshold,
            species_fitness_func=species_fitness_func,  # Added
            species_number_calculate_by=species_cfg.species_number_calculate_by,  # Added
            verbose=self.config.verbose  # Debug output control
        )

        return neat_algorithm

    def build_hyperneat_cppn_config(self, jax_module, act_module, agg_module) -> Any:
        """Build CPPN NEAT configuration for HyperNEAT.

        Creates the CPPN NEAT algorithm that evolves pattern-generating networks.

        Args:
            jax_module: JAX module
            act_module: TensorNEAT ACT module
            agg_module: TensorNEAT AGG module

        Returns:
            Configured CPPN NEAT algorithm

        Raises:
            ValueError: If config is not HyperNEATConfig
        """
        if not self.is_hyperneat:
            raise ValueError("build_hyperneat_cppn_config requires HyperNEATConfig")

        return self._build_cppn_neat_config(act_module)

    def _build_cppn_neat_config(self, act_module) -> Any:
        """Build NEAT configuration for CPPN evolution.

        CPPNs (Compositional Pattern Producing Networks) are evolved by NEAT
        to generate connection weights for the substrate network.

        Args:
            act_module: TensorNEAT ACT module

        Returns:
            Configured NEAT algorithm for CPPN evolution
        """
        from tensorneat.algorithm import NEAT
        from tensorneat.genome import DefaultGenome
        from tensorneat.genome.gene.node import DefaultNode
        from tensorneat.genome.gene.conn import DefaultConn
        from tensorneat.genome.operations.mutation import DefaultMutation

        # Get CPPN config from HyperNEAT schema
        cppn_cfg = self.config.cppn
        cppn_genome_cfg = cppn_cfg.genome
        cppn_mutation_cfg = cppn_cfg.mutation
        cppn_selection_cfg = cppn_cfg.selection

        # CPPN activation functions - critical for pattern generation
        cppn_activations = [
            act_module.gauss,
            act_module.sin,
            act_module.tanh,
            act_module.sigmoid
        ]

        # Create CPPN node gene with schema parameters
        node_gene = DefaultNode(
            bias_mutate_rate=cppn_genome_cfg.bias.mutate_rate,
            bias_replace_rate=cppn_genome_cfg.bias.replace_rate,
            bias_mutate_power=cppn_genome_cfg.bias.mutate_power,
            activation_options=cppn_activations,
            activation_replace_rate=cppn_genome_cfg.activation.mutate_rate
        )

        # Create CPPN connection gene with schema parameters
        conn_gene = DefaultConn(
            weight_mutate_rate=cppn_genome_cfg.weight.mutate_rate,
            weight_replace_rate=cppn_genome_cfg.weight.replace_rate,
            weight_mutate_power=cppn_genome_cfg.weight.mutate_power
        )

        # Create CPPN mutation operator
        mutation = DefaultMutation(
            conn_add=cppn_mutation_cfg.conn_add_prob,
            conn_delete=cppn_mutation_cfg.conn_delete_prob,
            node_add=cppn_mutation_cfg.node_add_prob,
            node_delete=cppn_mutation_cfg.node_delete_prob
        )

        # Create NEAT algorithm for CPPN evolution
        cppn_neat = NEAT(
            pop_size=cppn_cfg.population_size,
            species_size=cppn_cfg.species.species_size,
            survival_threshold=cppn_selection_cfg.survival_threshold,
            genome_elitism=cppn_selection_cfg.genome_elitism,
            genome=DefaultGenome(
                num_inputs=cppn_genome_cfg.num_inputs,
                num_outputs=cppn_genome_cfg.num_outputs,
                max_nodes=cppn_genome_cfg.max_nodes,
                node_gene=node_gene,
                conn_gene=conn_gene,
                mutation=mutation,
                output_transform=act_module.tanh,  # CPPN output activation
                init_hidden_layers=()  # Start minimal
            )
        )

        return cppn_neat

    def get_substrate_config(self) -> dict:
        """Get substrate configuration for HyperNEAT.

        Returns:
            Dictionary of substrate parameters

        Raises:
            ValueError: If config is not HyperNEATConfig
        """
        if not self.is_hyperneat:
            raise ValueError("get_substrate_config requires HyperNEATConfig")

        substrate_cfg = self.config.substrate

        return {
            'weight_threshold': substrate_cfg.weight_threshold,
            'max_weight': substrate_cfg.max_weight,
            'activate_time': substrate_cfg.activate_time,
            'output_activation': substrate_cfg.output_activation,
            'hidden_activation': substrate_cfg.hidden_activation,
        }

    def summary(self) -> str:
        """Generate configuration summary for logging.

        Returns:
            Human-readable configuration summary
        """
        algo_type = "HyperNEAT" if self.is_hyperneat else "NEAT"
        genome_cfg = self.config.genome

        lines = [
            "TensorNEAT Configuration Summary (Schema-Based V2)",
            "=" * 60,
            f"Algorithm: {algo_type}",
            f"Population Size: {self.config.population_size}",
            f"Max Generations: {self.config.max_generations}",
            f"Species Size: {self.config.species.species_size}",
            f"Network: {genome_cfg.num_inputs} inputs → {genome_cfg.num_outputs} outputs",
            f"Max Nodes: {genome_cfg.max_nodes}",
            f"Max Connections: {genome_cfg.max_connections}",
            "",
            "Mutation Rates:",
            f"  Connection Add: {self.config.mutation.conn_add_prob}",
            f"  Connection Delete: {self.config.mutation.conn_delete_prob}",
            f"  Node Add: {self.config.mutation.node_add_prob}",
            f"  Node Delete: {self.config.mutation.node_delete_prob}",
            f"  Weight Mutate: {genome_cfg.weight.mutate_rate}",
            f"  Bias Mutate: {genome_cfg.bias.mutate_rate}",
            "",
            "Selection:",
            f"  Genome Elitism: {self.config.selection.genome_elitism}",
            f"  Species Elitism: {self.config.species.species_elitism}",
            f"  Survival Threshold: {self.config.selection.survival_threshold}",
            "=" * 60,
        ]

        if self.is_hyperneat:
            lines.extend([
                "",
                "HyperNEAT Specific:",
                f"  CPPN Population: {self.config.cppn.population_size}",
                f"  CPPN Inputs: {self.config.cppn.genome.num_inputs}",
                f"  CPPN Outputs: {self.config.cppn.genome.num_outputs}",
                f"  Weight Threshold: {self.config.substrate.weight_threshold}",
                f"  Activate Time: {self.config.substrate.activate_time}",
                "=" * 60,
            ])

        return "\n".join(lines)
